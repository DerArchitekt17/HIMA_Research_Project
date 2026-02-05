"""
Benchmark the single fine-tuned SOAP model against reference notes.

Loads LLM on available GPUs in NF4, applies the single LoRA adapter, and
processes samples in parallel.

For each validation case:
  1. Generate complete SOAP note from dialogue
  2. Compute ROUGE and BERTScore against reference SOAP note

Usage:
    python benchmark.py --output results/benchmark_results.json
    python benchmark.py --num_samples 5 --output results/test_run.json
"""
# Workaround for CVE-2025-32434 check on older PyTorch (< 2.6)
# MUST be done before any transformers import
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.trainer
transformers.trainer.check_torch_load_is_safe = lambda: None

import os
import json
import argparse
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import PeftModel
from rouge_score import rouge_scorer

from bert_score import score as bert_score
# Must be absolute path - relative paths like ./bertscore_model are rejected as invalid repo IDs
BERTSCORE_MODEL_DIR = os.path.abspath(os.path.join(os.getenv("SLURM_SUBMIT_DIR", "."), "bertscore_model"))

# Register missing ministral3 text config
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# Processing CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=0,
                    help="Number of samples to benchmark (0 = all)")
parser.add_argument("--output", type=str, default="benchmark_results/hima_single_benchmark.json",
                    help="Path to save results JSON")
parser.add_argument("--max_new_tokens", type=int, default=2048,
                    help="Max new tokens for generation")
args = parser.parse_args()

# Set paths and environment variables
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
MODEL_DIR = os.path.join(BASE_DIR, "basemodel")
DATA_DIR = os.path.join(BASE_DIR, "data")
ADAPTER_DIR = os.path.join(BASE_DIR, "finetuned_models", "final_adapter")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Validation data loading
print("Loading validation data ...")
validation_data = load_jsonl(os.path.join(DATA_DIR, "validation/validation_single.jsonl"))
print(f"Loaded {len(validation_data)} records")

# Prepare dataset
dataset = []
for rec in validation_data:
    msgs = rec["messages"]
    system_prompt = next(m["content"] for m in msgs if m["role"] == "system")
    user_message = next(m["content"] for m in msgs if m["role"] == "user")
    reference = next(m["content"] for m in msgs if m["role"] == "assistant")
    dataset.append({
        "system_prompt": system_prompt,
        "user_message": user_message,
        "reference": reference,
    })

if args.num_samples > 0:
    dataset = dataset[:args.num_samples]
    print(f"Using first {args.num_samples} samples")


# Worker function
def worker(gpu_id: int, samples: list[dict], max_new_tokens: int,
           result_queue: mp.Queue):
    """Load model on gpu_id, process assigned samples, put results in queue."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading base model ...", flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print(f"[GPU {gpu_id}] Loading LoRA adapter ...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    def generate(system_prompt, user_message):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        n_in = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
            )
        n_out = outputs[0].shape[0] - n_in
        elapsed = time.time() - t0
        print(f"    [GPU {gpu_id}] {n_in}→{n_out} tokens "
              f"({n_out/max(elapsed,0.01):.1f} tok/s)", flush=True)
        return tok.decode(outputs[0][n_in:], skip_special_tokens=True)

    # Process assigned samples
    sample_times = []
    for idx, sample in enumerate(samples):
        sample_start = time.time()
        print(f"[GPU {gpu_id}] Sample {sample['_global_idx']+1}/{sample['_total']} ...",
              flush=True)

        output = generate(sample["system_prompt"], sample["user_message"])

        result_queue.put({
            "index": sample["_global_idx"],
            "output": output,
            "reference": sample["reference"],
        })

        # Track timing and estimate remaining
        sample_elapsed = time.time() - sample_start
        sample_times.append(sample_elapsed)

        # After 3 samples, start showing estimates
        if len(sample_times) >= 3:
            avg_time = sum(sample_times) / len(sample_times)
            remaining_samples = len(samples) - (idx + 1)
            eta_seconds = avg_time * remaining_samples
            eta_hours = eta_seconds / 3600
            print(f"[GPU {gpu_id}] Sample took {sample_elapsed:.1f}s | "
                  f"Avg: {avg_time:.1f}s | "
                  f"ETA this GPU: {eta_hours:.2f}h ({remaining_samples} samples left)", flush=True)

    total_time = sum(sample_times) if sample_times else 0
    avg_time = sum(sample_times) / len(sample_times) if sample_times else 0
    print(f"[GPU {gpu_id}] Done — processed {len(samples)} samples in {total_time/3600:.2f}h "
          f"(avg {avg_time:.1f}s/sample)", flush=True)


def main():
    # Enabling multi GPU processing
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} GPU(s) for inference on {len(dataset)} samples ...")
    print(f"Each GPU processes ~{len(dataset)//max(num_gpus,1)} samples in parallel.")
    print(f"ETA estimates will appear after the first few samples complete.\n")
    inference_start_time = time.time()

    # Tag samples with global index
    for i, s in enumerate(dataset):
        s["_global_idx"] = i
        s["_total"] = len(dataset)

    if num_gpus >= 2:
        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()
        # Round-robin split across available GPUs
        shards = [[] for _ in range(num_gpus)]
        for i, s in enumerate(dataset):
            shards[i % num_gpus].append(s)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=worker,
                           args=(gpu_id, shards[gpu_id], args.max_new_tokens, result_queue))
            p.start()
            processes.append(p)

        # Collect results WHILE workers run (prevents queue flush deadlock)
        results = []
        completed = 0
        while completed < len(dataset):
            try:
                result = result_queue.get(timeout=300)  # 5 min timeout per result
                results.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"Collected {completed}/{len(dataset)} results...", flush=True)
            except Exception:
                # Check if all workers are done
                alive = sum(1 for p in processes if p.is_alive())
                if alive == 0:
                    print(f"All workers exited. Collected {completed}/{len(dataset)} results.", flush=True)
                    break
                print(f"Waiting for results... ({alive} workers alive)", flush=True)

        # Now join the processes
        for p in processes:
            p.join(timeout=10)
    else:
        # Single GPU fallback!
        result_queue = mp.Queue()
        worker(0, dataset, args.max_new_tokens, result_queue)
        results = []
        for _ in range(len(dataset)):
            results.append(result_queue.get())

    # Sort by original index
    results.sort(key=lambda r: r["index"])
    inference_elapsed = time.time() - inference_start_time

    # Free GPU memory from inference workers before BERTScore
    torch.cuda.empty_cache()
    print(f"\nCollected {len(results)} results.")
    if len(results) == 0:
        print("ERROR: No results collected! Check worker logs above.")
        return
    print(f"Total inference time: {inference_elapsed/3600:.2f}h ({inference_elapsed/len(results):.1f}s/sample avg)")

# Computing metrics
    # Computing ROUGE scores
    print("\nComputing ROUGE scores ...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for res in results:
        scores = scorer.score(res["reference"], res["output"])
        res["rouge"] = {
            k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
            for k, v in scores.items()
        }

    # Computing BERTScore (on GPU) - batched to avoid memory explosion
    print("Computing BERTScore ...")
    bertscore_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {bertscore_device} for BERTScore computation")
    print(f"Using local model: {BERTSCORE_MODEL_DIR}")

    # Process in batches to avoid O(n*L^2) memory growth
    # A100 40GB: use 4, B200/H100 80GB+: can use 8-16
    BERTSCORE_BATCH_SIZE = 4
    all_P, all_R, all_F1 = [], [], []

    for batch_start in range(0, len(results), BERTSCORE_BATCH_SIZE):
        batch_end = min(batch_start + BERTSCORE_BATCH_SIZE, len(results))
        batch_results = results[batch_start:batch_end]

        refs = [r["reference"] for r in batch_results]
        cands = [r["output"] for r in batch_results]

        print(f"  BERTScore batch {batch_start//BERTSCORE_BATCH_SIZE + 1}/"
              f"{(len(results) + BERTSCORE_BATCH_SIZE - 1)//BERTSCORE_BATCH_SIZE} "
              f"(samples {batch_start+1}-{batch_end})")

        P, R, F1 = bert_score(cands, refs, model_type=BERTSCORE_MODEL_DIR,
                              num_layers=17, verbose=False, device=bertscore_device)
        all_P.extend(P.tolist())
        all_R.extend(R.tolist())
        all_F1.extend(F1.tolist())

        # Clear CUDA cache between batches to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, res in enumerate(results):
        res["bertscore"] = {
            "precision": all_P[i],
            "recall": all_R[i],
            "f1": all_F1[i],
        }

    # Result aggregation and summary terminal output
    n = len(results)

    def avg(key_fn):
        return sum(key_fn(r) for r in results) / n

    print(f"\n{'='*90}")
    print(f"BENCHMARK RESULTS ({n} samples)")
    print(f"{'='*90}")

    print(f"\n  ROUGE-1:    P={avg(lambda r: r['rouge']['rouge1']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge']['rouge1']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge']['rouge1']['fmeasure']):.4f}")
    print(f"  ROUGE-2:    P={avg(lambda r: r['rouge']['rouge2']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge']['rouge2']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge']['rouge2']['fmeasure']):.4f}")
    print(f"  ROUGE-L:    P={avg(lambda r: r['rouge']['rougeL']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge']['rougeL']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge']['rougeL']['fmeasure']):.4f}")
    print(f"  BERTScore:  P={avg(lambda r: r['bertscore']['precision']):.4f}  "
          f"R={avg(lambda r: r['bertscore']['recall']):.4f}  "
          f"F1={avg(lambda r: r['bertscore']['f1']):.4f}")

    print(f"\n{'='*90}")

    # Write results to disk
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    summary = {
        "num_samples": n,
        "rouge1": {
            "precision": avg(lambda r: r["rouge"]["rouge1"]["precision"]),
            "recall": avg(lambda r: r["rouge"]["rouge1"]["recall"]),
            "f1": avg(lambda r: r["rouge"]["rouge1"]["fmeasure"]),
        },
        "rouge2": {
            "precision": avg(lambda r: r["rouge"]["rouge2"]["precision"]),
            "recall": avg(lambda r: r["rouge"]["rouge2"]["recall"]),
            "f1": avg(lambda r: r["rouge"]["rouge2"]["fmeasure"]),
        },
        "rougeL": {
            "precision": avg(lambda r: r["rouge"]["rougeL"]["precision"]),
            "recall": avg(lambda r: r["rouge"]["rougeL"]["recall"]),
            "f1": avg(lambda r: r["rouge"]["rougeL"]["fmeasure"]),
        },
        "bertscore": {
            "precision": avg(lambda r: r["bertscore"]["precision"]),
            "recall": avg(lambda r: r["bertscore"]["recall"]),
            "f1": avg(lambda r: r["bertscore"]["f1"]),
        },
    }

    output_data = {
        "summary": summary,
        "per_sample": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
