"""
Benchmark the 4 fine-tuned SOAP agents against reference sections.

Loads LLMs among available GPUs in NF4, applies all 4 LoRA adapters, and
processes samples in parallel.

For each validation case:
  1. Run all 4 agents (S, O, A, P) sequentially on a dialogue
  2. Compute ROUGE and BERTScore per agent against its reference section
  3. Assemble outputs into a combined SOAP note and score against combined reference

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

# Workaround for HF transformers bug: Ministral 3B tokenizer has vocab_file=None,
# which crashes convert_slow_tokenizer when it calls vocab_file.endswith(...)
import sys, transformers.convert_slow_tokenizer
_cst_mod = sys.modules["transformers.convert_slow_tokenizer"]
_orig_convert = _cst_mod.convert_slow_tokenizer
def _safe_convert(transformer_tokenizer, *args, **kwargs):
    if getattr(transformer_tokenizer, "vocab_file", None) is None:
        transformer_tokenizer.vocab_file = ""
    return _orig_convert(transformer_tokenizer, *args, **kwargs)
_cst_mod.convert_slow_tokenizer = _safe_convert

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BERTSCORE_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "bertscore")

# Register missing ministral3 text config
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

AGENTS = ["subjective", "objective", "assessment", "plan"]
SECTION_LABELS = {
    "subjective": "**1. Subjective:**",
    "objective": "**2. Objective:**",
    "assessment": "**3. Assessment:**",
    "plan": "**4. Plan:**",
}

# Processing CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=0,
                    help="Number of samples to benchmark (0 = all)")
parser.add_argument("--output", type=str, default="benchmark_results/hima_multi_benchmark.json",
                    help="Path to save results JSON")
parser.add_argument("--max_new_tokens", type=int, default=2048,
                    help="Max new tokens per agent generation")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size per GPU for generation (reduce if OOM)")
args = parser.parse_args()

# Set paths and environment variables
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ministral3_3B")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Extraction of raw dialogue
def extract_dialogue(user_message: str) -> str:
    """Extract raw dialogue from user message by stripping the instruction prefix."""
    parts = user_message.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else user_message


# Loading uniform test data
print("Loading single-LLM test data ...")
multi_agent_benchmark = load_jsonl(os.path.join(DATA_DIR, "test/test_full.jsonl"))
print(f"Loaded {len(multi_agent_benchmark)} single-LLM records")

single_llm_refs = {}
for rec in multi_agent_benchmark:
    msgs = rec["messages"]
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    ref = next(m["content"] for m in msgs if m["role"] == "assistant")
    dialogue = extract_dialogue(user_msg)
    single_llm_refs[dialogue] = ref

print("Loading agent test data ...")
agent_data = {}
for agent in AGENTS:
    path = os.path.join(DATA_DIR, f"test/test_{agent}.jsonl")
    agent_data[agent] = load_jsonl(path)

num_records = len(agent_data["subjective"])
print(f"Loaded {num_records} records per agent")

dataset = []
skipped = 0
for i in range(num_records):
    ref_sections = {}
    system_prompts = {}
    user_messages = {}
    for agent in AGENTS:
        msgs = agent_data[agent][i]["messages"]
        system_prompts[agent] = next(m["content"] for m in msgs if m["role"] == "system")
        user_messages[agent] = next(m["content"] for m in msgs if m["role"] == "user")
        ref_sections[agent] = next(m["content"] for m in msgs if m["role"] == "assistant")

    dialogue = extract_dialogue(user_messages["subjective"])
    ref_combined = single_llm_refs.get(dialogue)
    if ref_combined is None:
        skipped += 1
        continue

    dataset.append({
        "ref_sections": ref_sections,
        "ref_combined": ref_combined,
        "agent_system_prompts": system_prompts,
        "agent_user_messages": user_messages,
    })

print(f"Matched {len(dataset)} records ({skipped} skipped — no single-LLM match)")

if args.num_samples > 0:
    dataset = dataset[:args.num_samples]
    print(f"Using first {args.num_samples} samples")


# Worker function
def worker(gpu_id: int, samples: list[dict], max_new_tokens: int,
           batch_size: int, result_queue: mp.Queue):
    """Load model on gpu_id, process samples stage-by-stage in batches."""
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
    tok.padding_side = "left"  # Required for batched decoder generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print(f"[GPU {gpu_id}] Loading LoRA adapters ...", flush=True)
    for i, agent in enumerate(AGENTS):
        adapter_dir = os.path.join(SCRIPT_DIR, "finetuned_adapters", agent, "final_adapter")
        if i == 0:
            model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=agent)
        else:
            model.load_adapter(adapter_dir, adapter_name=agent)

    print(f"[GPU {gpu_id}] Adapters: {list(model.peft_config.keys())}", flush=True)

    def batched_generate(adapter_name, prompts):
        """Generate for all prompts using the given adapter, in sub-batches."""
        model.set_adapter(adapter_name)
        model.eval()
        all_texts = []
        total_tokens = 0
        t0 = time.time()

        for b_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[b_start : b_start + batch_size]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
            padded_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                )

            for i in range(len(batch_prompts)):
                text = tok.decode(outputs[i][padded_len:], skip_special_tokens=True)
                all_texts.append(text)
                total_tokens += outputs[i].shape[0] - padded_len

        elapsed = time.time() - t0
        print(f"  [GPU {gpu_id}][{adapter_name}] {len(prompts)} samples, "
              f"{total_tokens} tokens in {elapsed:.1f}s "
              f"({total_tokens/max(elapsed,0.01):.0f} tok/s)", flush=True)
        return all_texts

    # Stage-by-stage batched generation
    n = len(samples)
    agent_outputs = {agent: [None] * n for agent in AGENTS}
    soap_so_far = [""] * n
    stage_times = []

    for agent in AGENTS:
        stage_t0 = time.time()
        print(f"[GPU {gpu_id}] Stage: {agent} ({n} samples) ...", flush=True)

        prompts = []
        for i, s in enumerate(samples):
            user_msg = s["agent_user_messages"][agent]
            if soap_so_far[i]:
                user_msg += f"\n\nPreviously generated SOAP sections:\n{soap_so_far[i].rstrip()}"
            messages = [
                {"role": "system", "content": s["agent_system_prompts"][agent]},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        texts = batched_generate(agent, prompts)

        for i in range(n):
            agent_outputs[agent][i] = texts[i]
            soap_so_far[i] += f"{SECTION_LABELS[agent]}\n{texts[i]}\n\n"

        stage_elapsed = time.time() - stage_t0
        stage_times.append(stage_elapsed)
        remaining_stages = len(AGENTS) - len(stage_times)
        avg_stage = sum(stage_times) / len(stage_times)
        eta_h = (avg_stage * remaining_stages) / 3600
        print(f"[GPU {gpu_id}] Stage {agent} done in {stage_elapsed:.1f}s | "
              f"ETA: {eta_h:.2f}h ({remaining_stages} stages left)", flush=True)

    # Enqueue results
    for i, s in enumerate(samples):
        combined = (
            f"**1. Subjective:**\n{agent_outputs['subjective'][i]}\n\n"
            f"**2. Objective:**\n{agent_outputs['objective'][i]}\n\n"
            f"**3. Assessment:**\n{agent_outputs['assessment'][i]}\n\n"
            f"**4. Plan:**\n{agent_outputs['plan'][i]}"
        )
        result_queue.put({
            "index": s["_global_idx"],
            "agent_outputs": {a: agent_outputs[a][i] for a in AGENTS},
            "combined_output": combined,
            "ref_sections": s["ref_sections"],
            "ref_combined": s["ref_combined"],
        })

    total_time = sum(stage_times)
    print(f"[GPU {gpu_id}] Done — {n} samples in {total_time/3600:.2f}h "
          f"({total_time/max(n,1):.1f}s/sample avg)", flush=True)


def bertscore_worker(gpu_id: int, tasks: list[dict], model_dir: str,
                     result_queue: mp.Queue):
    """Compute BERTScore for assigned tasks on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] BERTScore worker starting with {len(tasks)} task(s)...",
          flush=True)

    for task in tasks:
        label = task["label"]
        print(f"  [GPU {gpu_id}] {label} ({len(task['cands'])} samples)...", flush=True)
        P, R, F1 = bert_score(
            task["cands"], task["refs"], model_type=model_dir,
            num_layers=17, verbose=False, device=device,
        )
        result_queue.put({
            "label": label,
            "P": P.tolist(),
            "R": R.tolist(),
            "F1": F1.tolist(),
        })
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] BERTScore worker done.", flush=True)


def main():
    mp.set_start_method("spawn", force=True)
    # Enabling multi GPU processing
    num_gpus = torch.cuda.device_count()
    batch_size = args.batch_size
    print(f"\nUsing {num_gpus} GPU(s) for inference on {len(dataset)} samples ...")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Each GPU processes ~{len(dataset)//max(num_gpus,1)} samples.")
    print(f"Pipeline: 4 agents (S→O→A→P), each stage batched.\n")
    inference_start_time = time.time()

    # Tag samples with global index
    for i, s in enumerate(dataset):
        s["_global_idx"] = i
        s["_total"] = len(dataset)

    if num_gpus >= 2:
        result_queue = mp.Queue()
        # Round-robin split across available GPUs
        shards = [[] for _ in range(num_gpus)]
        for i, s in enumerate(dataset):
            shards[i % num_gpus].append(s)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=worker,
                           args=(gpu_id, shards[gpu_id], args.max_new_tokens,
                                 batch_size, result_queue))
            p.start()
            processes.append(p)

        # Collect results WHILE workers run (prevents queue flush issues)
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
        worker(0, dataset, args.max_new_tokens, batch_size, result_queue)
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

    # Computing ROUGE scores
    print("\nComputing ROUGE scores ...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for res in results:
        res["rouge_per_agent"] = {}
        for agent in AGENTS:
            scores = scorer.score(res["ref_sections"][agent], res["agent_outputs"][agent])
            res["rouge_per_agent"][agent] = {
                k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
                for k, v in scores.items()
            }

        scores = scorer.score(res["ref_combined"], res["combined_output"])
        res["rouge_combined"] = {
            k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
            for k, v in scores.items()
        }

    # BERTScore — parallel across all available GPUs
    print("\nComputing BERTScore ...")
    print(f"Using local model: {BERTSCORE_MODEL_DIR}")

    # Build all 5 BERTScore tasks (4 agents + 1 combined)
    bs_tasks = []
    for agent in AGENTS:
        bs_tasks.append({
            "label": f"agent_{agent}",
            "cands": [r["agent_outputs"][agent] for r in results],
            "refs": [r["ref_sections"][agent] for r in results],
        })
    bs_tasks.append({
        "label": "combined",
        "cands": [r["combined_output"] for r in results],
        "refs": [r["ref_combined"] for r in results],
    })

    if num_gpus >= 2:
        # Distribute tasks round-robin across GPUs
        print(f"Distributing {len(bs_tasks)} BERTScore tasks across {num_gpus} GPUs ...")
        gpu_tasks = [[] for _ in range(num_gpus)]
        for i, task in enumerate(bs_tasks):
            gpu_tasks[i % num_gpus].append(task)

        bs_queue = mp.Queue()
        bs_processes = []
        for gpu_id in range(num_gpus):
            if gpu_tasks[gpu_id]:
                p = mp.Process(
                    target=bertscore_worker,
                    args=(gpu_id, gpu_tasks[gpu_id], BERTSCORE_MODEL_DIR, bs_queue),
                )
                p.start()
                bs_processes.append(p)

        # Collect all results
        bs_results = {}
        for _ in range(len(bs_tasks)):
            res = bs_queue.get(timeout=600)
            bs_results[res["label"]] = res

        for p in bs_processes:
            p.join(timeout=10)
    else:
        # Single GPU fallback
        print("Using cuda:0 for BERTScore computation")
        bs_results = {}
        for task in bs_tasks:
            print(f"  {task['label']} ({len(task['cands'])} samples)...", flush=True)
            P, R, F1 = bert_score(
                task["cands"], task["refs"], model_type=BERTSCORE_MODEL_DIR,
                num_layers=17, verbose=True, device="cuda:0",
            )
            bs_results[task["label"]] = {
                "label": task["label"],
                "P": P.tolist(),
                "R": R.tolist(),
                "F1": F1.tolist(),
            }
            torch.cuda.empty_cache()

    # Assign BERTScore results back to per-sample results
    for agent in AGENTS:
        bs = bs_results[f"agent_{agent}"]
        for i, res in enumerate(results):
            if "bertscore_per_agent" not in res:
                res["bertscore_per_agent"] = {}
            res["bertscore_per_agent"][agent] = {
                "precision": bs["P"][i],
                "recall": bs["R"][i],
                "f1": bs["F1"][i],
            }

    bs = bs_results["combined"]
    for i, res in enumerate(results):
        res["bertscore_combined"] = {
            "precision": bs["P"][i],
            "recall": bs["R"][i],
            "f1": bs["F1"][i],
        }
    # Result aggregation and summary terminal output
    n = len(results)

    def avg(key_fn):
        return sum(key_fn(r) for r in results) / n

    print(f"\n{'='*90}")
    print(f"BENCHMARK RESULTS ({n} samples)")
    print(f"{'='*90}")

    for agent in AGENTS:
        print(f"\n  {agent.upper()}")
        print(f"    ROUGE-1:    P={avg(lambda r: r['rouge_per_agent'][agent]['rouge1']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_agent'][agent]['rouge1']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_agent'][agent]['rouge1']['fmeasure']):.4f}")
        print(f"    ROUGE-2:    P={avg(lambda r: r['rouge_per_agent'][agent]['rouge2']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_agent'][agent]['rouge2']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_agent'][agent]['rouge2']['fmeasure']):.4f}")
        print(f"    ROUGE-L:    P={avg(lambda r: r['rouge_per_agent'][agent]['rougeL']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_agent'][agent]['rougeL']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_agent'][agent]['rougeL']['fmeasure']):.4f}")
        print(f"    BERTScore:  P={avg(lambda r: r['bertscore_per_agent'][agent]['precision']):.4f}  "
              f"R={avg(lambda r: r['bertscore_per_agent'][agent]['recall']):.4f}  "
              f"F1={avg(lambda r: r['bertscore_per_agent'][agent]['f1']):.4f}")

    print(f"\n  COMBINED")
    print(f"    ROUGE-1:    P={avg(lambda r: r['rouge_combined']['rouge1']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge_combined']['rouge1']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge_combined']['rouge1']['fmeasure']):.4f}")
    print(f"    ROUGE-2:    P={avg(lambda r: r['rouge_combined']['rouge2']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge_combined']['rouge2']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge_combined']['rouge2']['fmeasure']):.4f}")
    print(f"    ROUGE-L:    P={avg(lambda r: r['rouge_combined']['rougeL']['precision']):.4f}  "
          f"R={avg(lambda r: r['rouge_combined']['rougeL']['recall']):.4f}  "
          f"F1={avg(lambda r: r['rouge_combined']['rougeL']['fmeasure']):.4f}")
    print(f"    BERTScore:  P={avg(lambda r: r['bertscore_combined']['precision']):.4f}  "
          f"R={avg(lambda r: r['bertscore_combined']['recall']):.4f}  "
          f"F1={avg(lambda r: r['bertscore_combined']['f1']):.4f}")

    print(f"\n{'='*90}")

    # Write results to disk
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    summary = {
        "num_samples": n,
        "per_agent": {},
        "combined": {
            "rouge1": {
                "precision": avg(lambda r: r["rouge_combined"]["rouge1"]["precision"]),
                "recall": avg(lambda r: r["rouge_combined"]["rouge1"]["recall"]),
                "f1": avg(lambda r: r["rouge_combined"]["rouge1"]["fmeasure"]),
            },
            "rouge2": {
                "precision": avg(lambda r: r["rouge_combined"]["rouge2"]["precision"]),
                "recall": avg(lambda r: r["rouge_combined"]["rouge2"]["recall"]),
                "f1": avg(lambda r: r["rouge_combined"]["rouge2"]["fmeasure"]),
            },
            "rougeL": {
                "precision": avg(lambda r: r["rouge_combined"]["rougeL"]["precision"]),
                "recall": avg(lambda r: r["rouge_combined"]["rougeL"]["recall"]),
                "f1": avg(lambda r: r["rouge_combined"]["rougeL"]["fmeasure"]),
            },
        },
    }
    summary["combined"]["bertscore"] = {
        "precision": avg(lambda r: r["bertscore_combined"]["precision"]),
        "recall": avg(lambda r: r["bertscore_combined"]["recall"]),
        "f1": avg(lambda r: r["bertscore_combined"]["f1"]),
    }
    for agent in AGENTS:
        summary["per_agent"][agent] = {
            "rouge1": {
                "precision": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge1"]["precision"]),
                "recall": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge1"]["recall"]),
                "f1": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge1"]["fmeasure"]),
            },
            "rouge2": {
                "precision": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge2"]["precision"]),
                "recall": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge2"]["recall"]),
                "f1": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rouge2"]["fmeasure"]),
            },
            "rougeL": {
                "precision": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rougeL"]["precision"]),
                "recall": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rougeL"]["recall"]),
                "f1": avg(lambda r, a=agent: r["rouge_per_agent"][a]["rougeL"]["fmeasure"]),
            },
        }
        summary["per_agent"][agent]["bertscore"] = {
            "precision": avg(lambda r, a=agent: r["bertscore_per_agent"][a]["precision"]),
            "recall": avg(lambda r, a=agent: r["bertscore_per_agent"][a]["recall"]),
            "f1": avg(lambda r, a=agent: r["bertscore_per_agent"][a]["f1"]),
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
