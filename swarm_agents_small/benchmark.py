"""
Benchmark the 12 swarm Draft-Critique-Refine agents against reference sections.

Loads base model + 12 LoRA adapters (3 roles x 4 dimensions) across available
GPUs in NF4, and processes samples in parallel via multiprocessing.

For each validation case:
  1. For each SOAP dimension (S, O, A, P):
     a. Drafter generates initial section from dialogue (+prior context)
     b. Critic reviews the draft against the source dialogue
     c. Refiner produces the definitive section incorporating feedback
  2. Compute ROUGE and BERTScore per dimension (Refiner output vs reference)
  3. Assemble final sections into a combined SOAP note and score vs combined ref

Ablation: Drafter-only metrics are computed alongside Refiner metrics to
quantify the contribution of the Critique-Refine loop.

Usage:
    python benchmark.py --output benchmark_results/hima_swarm_benchmark.json
    python benchmark.py --num_samples 5 --output benchmark_results/test_run.json
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
BERTSCORE_MODEL_DIR = os.path.join(os.getenv("SLURM_SUBMIT_DIR", "."), "bertscore_model")

# Register missing ministral3 text config
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

DIMENSIONS = ["subjective", "objective", "assessment", "plan"]
ROLES = ["drafter", "critic", "refiner"]
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
parser.add_argument("--output", type=str,
                    default="benchmark_results/hima_swarm_benchmark.json",
                    help="Path to save results JSON")
parser.add_argument("--max_new_tokens", type=int, default=2048,
                    help="Max new tokens per generation call")
args = parser.parse_args()

# Set paths and environment variables
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
MODEL_DIR = os.path.join(BASE_DIR, "basemodel")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_dialogue(user_message: str) -> str:
    """Extract raw dialogue from user message by stripping the instruction prefix."""
    parts = user_message.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else user_message


# Load combined reference (single-LLM full SOAP notes for fair cross-architecture comparison)
print("Loading single-LLM validation data ...")
benchmark_ref = load_jsonl(os.path.join(DATA_DIR, "benchmark_swarm_agents.jsonl"))
print(f"Loaded {len(benchmark_ref)} single-LLM records")

single_llm_refs = {}
for rec in benchmark_ref:
    msgs = rec["messages"]
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    ref = next(m["content"] for m in msgs if m["role"] == "assistant")
    dialogue = extract_dialogue(user_msg)
    single_llm_refs[dialogue] = ref

# Load drafter validation data (contains dialogue + gold sections)
print("Loading drafter validation data ...")
drafter_data = {}
for dim in DIMENSIONS:
    path = os.path.join(DATA_DIR, f"validation/validation_drafter_{dim}.jsonl")
    drafter_data[dim] = load_jsonl(path)

num_records = len(drafter_data["subjective"])
print(f"Loaded {num_records} records per dimension")

# Extract system prompts for each role from their validation data
print("Loading system prompts for all roles ...")
role_system_prompts = {}
for role in ROLES:
    role_system_prompts[role] = {}
    for dim in DIMENSIONS:
        path = os.path.join(DATA_DIR, f"validation/validation_{role}_{dim}.jsonl")
        first_rec = load_jsonl(path)[0]
        msgs = first_rec["messages"]
        role_system_prompts[role][dim] = next(
            m["content"] for m in msgs if m["role"] == "system"
        )

# Build dataset
dataset = []
skipped = 0
for i in range(num_records):
    ref_sections = {}
    for dim in DIMENSIONS:
        msgs = drafter_data[dim][i]["messages"]
        ref_sections[dim] = next(m["content"] for m in msgs if m["role"] == "assistant")

    # Dialogue comes from the drafter's user message
    user_msg = next(
        m["content"] for m in drafter_data["subjective"][i]["messages"]
        if m["role"] == "user"
    )
    dialogue = extract_dialogue(user_msg)

    ref_combined = single_llm_refs.get(dialogue)
    if ref_combined is None:
        skipped += 1
        continue

    dataset.append({
        "dialogue": dialogue,
        "ref_sections": ref_sections,
        "ref_combined": ref_combined,
    })

print(f"Matched {len(dataset)} records ({skipped} skipped - no single-LLM match)")

if args.num_samples > 0:
    dataset = dataset[:args.num_samples]
    print(f"Using first {args.num_samples} samples")


# ---------------------------------------------------------------------------
# Worker function - runs on a single GPU
# ---------------------------------------------------------------------------

def worker(gpu_id: int, samples: list[dict], max_new_tokens: int,
           system_prompts: dict, result_queue: mp.Queue):
    """Load base model + 12 LoRA adapters on gpu_id, run DCR pipeline per sample."""
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

    # Load all 12 LoRA adapters (3 roles x 4 dimensions)
    print(f"[GPU {gpu_id}] Loading 12 LoRA adapters ...", flush=True)
    first = True
    for dim in DIMENSIONS:
        for role in ROLES:
            adapter_name = f"{role}_{dim}"
            adapter_dir = os.path.join(
                BASE_DIR, "finetuned_models", dim, role, "final_adapter"
            )
            if first:
                model = PeftModel.from_pretrained(
                    model, adapter_dir, adapter_name=adapter_name
                )
                first = False
            else:
                model.load_adapter(adapter_dir, adapter_name=adapter_name)

    print(f"[GPU {gpu_id}] Adapters: {list(model.peft_config.keys())}", flush=True)

    def gen(adapter_name, sys_prompt, user_message):
        model.set_adapter(adapter_name)
        model.eval()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        print(
            f"    [GPU {gpu_id}][{adapter_name}] {n_in}->{n_out} tokens "
            f"({n_out / max(elapsed, 0.01):.1f} tok/s)",
            flush=True,
        )
        return tok.decode(outputs[0][n_in:], skip_special_tokens=True)

    # Process assigned samples
    sample_times = []
    for idx, sample in enumerate(samples):
        sample_start = time.time()
        print(
            f"[GPU {gpu_id}] Sample {sample['_global_idx'] + 1}/{sample['_total']} ...",
            flush=True,
        )

        drafter_outputs = {}
        critic_outputs = {}
        refiner_outputs = {}
        soap_so_far = ""

        for dim in DIMENSIONS:
            # --- Drafter ---
            drafter_msg = (
                f"Draft the {dim.capitalize()} section from the following "
                f"consultation dialogue:\n\n{sample['dialogue']}"
            )
            if soap_so_far:
                drafter_msg += (
                    f"\n\nPreviously generated SOAP sections:\n{soap_so_far.rstrip()}"
                )
            draft = gen(
                f"drafter_{dim}", system_prompts["drafter"][dim], drafter_msg
            )
            drafter_outputs[dim] = draft

            # --- Critic ---
            critic_msg = (
                f"Review the following draft {dim.capitalize()} section against "
                f"the source consultation dialogue.\n\n"
                f"Source consultation dialogue:\n{sample['dialogue']}\n\n"
                f"Draft {dim.capitalize()} section to review:\n{draft}"
            )
            critique = gen(
                f"critic_{dim}", system_prompts["critic"][dim], critic_msg
            )
            critic_outputs[dim] = critique

            # --- Refiner ---
            refiner_msg = (
                f"Produce the final revised {dim.capitalize()} section.\n\n"
                f"Source consultation dialogue:\n{sample['dialogue']}\n\n"
                f"Initial draft:\n{draft}\n\n"
                f"Peer critique:\n{critique}"
            )
            final = gen(
                f"refiner_{dim}", system_prompts["refiner"][dim], refiner_msg
            )
            refiner_outputs[dim] = final

            # Accumulate context for next dimension's drafter
            soap_so_far += f"{SECTION_LABELS[dim]}\n{final}\n\n"

        combined = (
            f"**1. Subjective:**\n{refiner_outputs['subjective']}\n\n"
            f"**2. Objective:**\n{refiner_outputs['objective']}\n\n"
            f"**3. Assessment:**\n{refiner_outputs['assessment']}\n\n"
            f"**4. Plan:**\n{refiner_outputs['plan']}"
        )

        result_queue.put({
            "index": sample["_global_idx"],
            "drafter_outputs": drafter_outputs,
            "critic_outputs": critic_outputs,
            "refiner_outputs": refiner_outputs,
            "combined_output": combined,
            "ref_sections": sample["ref_sections"],
            "ref_combined": sample["ref_combined"],
        })

        # Track timing and estimate remaining
        sample_elapsed = time.time() - sample_start
        sample_times.append(sample_elapsed)

        if len(sample_times) >= 3:
            avg_time = sum(sample_times) / len(sample_times)
            remaining = len(samples) - (idx + 1)
            eta_hours = (avg_time * remaining) / 3600
            print(
                f"[GPU {gpu_id}] Sample took {sample_elapsed:.1f}s | "
                f"Avg: {avg_time:.1f}s | "
                f"ETA this GPU: {eta_hours:.2f}h ({remaining} samples left)",
                flush=True,
            )

    total_time = sum(sample_times) if sample_times else 0
    avg_time = sum(sample_times) / len(sample_times) if sample_times else 0
    print(
        f"[GPU {gpu_id}] Done - processed {len(samples)} samples in "
        f"{total_time / 3600:.2f}h (avg {avg_time:.1f}s/sample)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# BERTScore worker - runs on a single GPU, processes assigned scoring tasks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} GPU(s) for inference on {len(dataset)} samples ...")
    print(f"Each GPU processes ~{len(dataset) // max(num_gpus, 1)} samples in parallel.")
    print(f"DCR pipeline: 3 generations per dimension x 4 dimensions = 12 calls/sample")
    print(f"ETA estimates will appear after the first few samples complete.\n")
    inference_start_time = time.time()

    # Tag samples with global index
    for i, s in enumerate(dataset):
        s["_global_idx"] = i
        s["_total"] = len(dataset)

    if num_gpus >= 2:
        result_queue = mp.Queue()

        # Round-robin split across GPUs
        shards = [[] for _ in range(num_gpus)]
        for i, s in enumerate(dataset):
            shards[i % num_gpus].append(s)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(gpu_id, shards[gpu_id], args.max_new_tokens,
                      role_system_prompts, result_queue),
            )
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
                alive = sum(1 for p in processes if p.is_alive())
                if alive == 0:
                    print(
                        f"All workers exited. Collected {completed}/{len(dataset)} results.",
                        flush=True,
                    )
                    break
                print(f"Waiting for results... ({alive} workers alive)", flush=True)

        # Now join the processes
        for p in processes:
            p.join(timeout=10)
    else:
        # Single GPU fallback
        result_queue = mp.Queue()
        worker(0, dataset, args.max_new_tokens, role_system_prompts, result_queue)
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
    print(f"Total inference time: {inference_elapsed / 3600:.2f}h "
          f"({inference_elapsed / len(results):.1f}s/sample avg)")

    # -------------------------------------------------------------------
    # Computing metrics
    # -------------------------------------------------------------------

    # ROUGE scores
    print("\nComputing ROUGE scores ...")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    for res in results:
        # Per-dimension: Refiner output vs gold section
        res["rouge_per_dim"] = {}
        for dim in DIMENSIONS:
            scores = scorer.score(
                res["ref_sections"][dim], res["refiner_outputs"][dim]
            )
            res["rouge_per_dim"][dim] = {
                k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
                for k, v in scores.items()
            }

        # Per-dimension: Drafter output vs gold section (ablation)
        res["rouge_per_dim_drafter"] = {}
        for dim in DIMENSIONS:
            scores = scorer.score(
                res["ref_sections"][dim], res["drafter_outputs"][dim]
            )
            res["rouge_per_dim_drafter"][dim] = {
                k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
                for k, v in scores.items()
            }

        # Combined
        scores = scorer.score(res["ref_combined"], res["combined_output"])
        res["rouge_combined"] = {
            k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
            for k, v in scores.items()
        }

    # BERTScore - parallel across all available GPUs
    print("\nComputing BERTScore ...")
    print(f"Using local model: {BERTSCORE_MODEL_DIR}")

    # Build all 9 BERTScore tasks (4 refiner + 4 drafter + 1 combined)
    bs_tasks = []
    for dim in DIMENSIONS:
        bs_tasks.append({
            "label": f"refiner_{dim}",
            "cands": [r["refiner_outputs"][dim] for r in results],
            "refs": [r["ref_sections"][dim] for r in results],
        })
    for dim in DIMENSIONS:
        bs_tasks.append({
            "label": f"drafter_{dim}",
            "cands": [r["drafter_outputs"][dim] for r in results],
            "refs": [r["ref_sections"][dim] for r in results],
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
    for dim in DIMENSIONS:
        bs = bs_results[f"refiner_{dim}"]
        for i, res in enumerate(results):
            if "bertscore_per_dim" not in res:
                res["bertscore_per_dim"] = {}
            res["bertscore_per_dim"][dim] = {
                "precision": bs["P"][i],
                "recall": bs["R"][i],
                "f1": bs["F1"][i],
            }

    for dim in DIMENSIONS:
        bs = bs_results[f"drafter_{dim}"]
        for i, res in enumerate(results):
            if "bertscore_per_dim_drafter" not in res:
                res["bertscore_per_dim_drafter"] = {}
            res["bertscore_per_dim_drafter"][dim] = {
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

    # -------------------------------------------------------------------
    # Result aggregation and summary terminal output
    # -------------------------------------------------------------------

    n = len(results)

    def avg(key_fn):
        return sum(key_fn(r) for r in results) / n

    print(f"\n{'=' * 90}")
    print(f"BENCHMARK RESULTS - SWARM DCR ({n} samples)")
    print(f"{'=' * 90}")

    for dim in DIMENSIONS:
        print(f"\n  {dim.upper()} - Refiner (final)")
        print(f"    ROUGE-1:    P={avg(lambda r: r['rouge_per_dim'][dim]['rouge1']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim'][dim]['rouge1']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim'][dim]['rouge1']['fmeasure']):.4f}")
        print(f"    ROUGE-2:    P={avg(lambda r: r['rouge_per_dim'][dim]['rouge2']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim'][dim]['rouge2']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim'][dim]['rouge2']['fmeasure']):.4f}")
        print(f"    ROUGE-L:    P={avg(lambda r: r['rouge_per_dim'][dim]['rougeL']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim'][dim]['rougeL']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim'][dim]['rougeL']['fmeasure']):.4f}")
        print(f"    BERTScore:  P={avg(lambda r: r['bertscore_per_dim'][dim]['precision']):.4f}  "
              f"R={avg(lambda r: r['bertscore_per_dim'][dim]['recall']):.4f}  "
              f"F1={avg(lambda r: r['bertscore_per_dim'][dim]['f1']):.4f}")

        print(f"  {dim.upper()} - Drafter (ablation)")
        print(f"    ROUGE-1:    P={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge1']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge1']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge1']['fmeasure']):.4f}")
        print(f"    ROUGE-2:    P={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge2']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge2']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rouge2']['fmeasure']):.4f}")
        print(f"    ROUGE-L:    P={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rougeL']['precision']):.4f}  "
              f"R={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rougeL']['recall']):.4f}  "
              f"F1={avg(lambda r: r['rouge_per_dim_drafter'][dim]['rougeL']['fmeasure']):.4f}")
        print(f"    BERTScore:  P={avg(lambda r: r['bertscore_per_dim_drafter'][dim]['precision']):.4f}  "
              f"R={avg(lambda r: r['bertscore_per_dim_drafter'][dim]['recall']):.4f}  "
              f"F1={avg(lambda r: r['bertscore_per_dim_drafter'][dim]['f1']):.4f}")

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

    print(f"\n{'=' * 90}")

    # -------------------------------------------------------------------
    # Write results JSON
    # -------------------------------------------------------------------

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    summary = {
        "num_samples": n,
        "per_dimension_refiner": {},
        "per_dimension_drafter": {},
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
            "bertscore": {
                "precision": avg(lambda r: r["bertscore_combined"]["precision"]),
                "recall": avg(lambda r: r["bertscore_combined"]["recall"]),
                "f1": avg(lambda r: r["bertscore_combined"]["f1"]),
            },
        },
    }

    for dim in DIMENSIONS:
        # Refiner metrics
        summary["per_dimension_refiner"][dim] = {
            "rouge1": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge1"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge1"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge1"]["fmeasure"]),
            },
            "rouge2": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge2"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge2"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rouge2"]["fmeasure"]),
            },
            "rougeL": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rougeL"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rougeL"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim"][d]["rougeL"]["fmeasure"]),
            },
            "bertscore": {
                "precision": avg(lambda r, d=dim: r["bertscore_per_dim"][d]["precision"]),
                "recall": avg(lambda r, d=dim: r["bertscore_per_dim"][d]["recall"]),
                "f1": avg(lambda r, d=dim: r["bertscore_per_dim"][d]["f1"]),
            },
        }

        # Drafter ablation metrics
        summary["per_dimension_drafter"][dim] = {
            "rouge1": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge1"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge1"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge1"]["fmeasure"]),
            },
            "rouge2": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge2"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge2"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rouge2"]["fmeasure"]),
            },
            "rougeL": {
                "precision": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rougeL"]["precision"]),
                "recall": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rougeL"]["recall"]),
                "f1": avg(lambda r, d=dim: r["rouge_per_dim_drafter"][d]["rougeL"]["fmeasure"]),
            },
            "bertscore": {
                "precision": avg(lambda r, d=dim: r["bertscore_per_dim_drafter"][d]["precision"]),
                "recall": avg(lambda r, d=dim: r["bertscore_per_dim_drafter"][d]["recall"]),
                "f1": avg(lambda r, d=dim: r["bertscore_per_dim_drafter"][d]["f1"]),
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

