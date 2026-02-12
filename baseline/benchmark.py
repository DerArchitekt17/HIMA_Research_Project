"""
Benchmark ground-truth SOAP notes against their source dialogues.

Computes ROUGE and BERTScore between the full SOAP note (assistant message)
and the raw dialogue (user message) for every record in benchmark_base_data.jsonl.

No model inference is performed — this is a pure metrics-computation script.

Usage:
    python benchmark.py --output benchmark_results/base_data_benchmark.json
    python benchmark.py --num_samples 50 --output benchmark_results/test_run.json
"""

import os
import json
import argparse
import time
import torch
import torch.multiprocessing as mp
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Must be absolute path - relative paths are rejected as invalid repo IDs
BERTSCORE_MODEL_DIR = os.path.abspath(
    os.path.join(os.getenv("SLURM_SUBMIT_DIR", "."), "bertscore_model")
)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=0,
                    help="Number of samples to benchmark (0 = all)")
parser.add_argument("--output", type=str,
                    default="benchmark_results/base_data_benchmark.json",
                    help="Path to save results JSON")
args = parser.parse_args()


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

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} GPU(s) for metrics computation")

    # Load ground-truth data
    BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
    data_path = os.path.join(BASE_DIR, "benchmark_base_data.jsonl")
    print(f"Loading data from {data_path} ...")
    raw = load_jsonl(data_path)
    print(f"Loaded {len(raw)} records")

    # Extract dialogues and SOAP notes
    dataset = []
    for rec in raw:
        msgs = rec["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        soap_note = next(m["content"] for m in msgs if m["role"] == "assistant")
        dialogue = extract_dialogue(user_msg)
        dataset.append({"dialogue": dialogue, "soap_note": soap_note})

    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        print(f"Using first {args.num_samples} samples")

    n = len(dataset)
    print(f"Computing metrics for {n} samples ...\n")

    # Computing ROUGE scores
    print("Computing ROUGE scores ...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []
    for sample in dataset:
        scores = scorer.score(sample["dialogue"], sample["soap_note"])
        results.append({
            "rouge": {
                k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
                for k, v in scores.items()
            },
        })

    print("ROUGE done.\n")

    # BERTScore — parallel across all available GPUs
    print("Computing BERTScore ...")
    print(f"Using local model: {BERTSCORE_MODEL_DIR}")

    all_cands = [s["soap_note"] for s in dataset]
    all_refs = [s["dialogue"] for s in dataset]

    if num_gpus >= 2:
        # Shard samples across GPUs
        print(f"Distributing {n} samples across {num_gpus} GPUs for BERTScore ...")
        shard_size = (n + num_gpus - 1) // num_gpus
        bs_tasks = []
        for gpu_id in range(num_gpus):
            start = gpu_id * shard_size
            end = min(start + shard_size, n)
            if start >= n:
                break
            bs_tasks.append({
                "label": f"shard_{gpu_id}",
                "cands": all_cands[start:end],
                "refs": all_refs[start:end],
            })

        bs_queue = mp.Queue()
        bs_processes = []
        for gpu_id, task in enumerate(bs_tasks):
            p = mp.Process(
                target=bertscore_worker,
                args=(gpu_id, [task], BERTSCORE_MODEL_DIR, bs_queue),
            )
            p.start()
            bs_processes.append(p)

        # Collect results and reassemble in order
        bs_results = {}
        for _ in range(len(bs_tasks)):
            res = bs_queue.get(timeout=600)
            bs_results[res["label"]] = res

        for p in bs_processes:
            p.join(timeout=10)

        # Merge shards back in GPU order
        P_all, R_all, F1_all = [], [], []
        for gpu_id in range(len(bs_tasks)):
            bs = bs_results[f"shard_{gpu_id}"]
            P_all.extend(bs["P"])
            R_all.extend(bs["R"])
            F1_all.extend(bs["F1"])
    else:
        # Single GPU fallback
        print("Using cuda:0 for BERTScore computation")
        P, R, F1 = bert_score(
            all_cands, all_refs, model_type=BERTSCORE_MODEL_DIR,
            num_layers=17, verbose=True, device="cuda:0",
        )
        P_all = P.tolist()
        R_all = R.tolist()
        F1_all = F1.tolist()
        torch.cuda.empty_cache()

    # Assign BERTScore to per-sample results
    for i, res in enumerate(results):
        res["bertscore"] = {
            "precision": P_all[i],
            "recall": R_all[i],
            "f1": F1_all[i],
        }

    # Aggregation
    def avg(key_fn):
        return sum(key_fn(r) for r in results) / n

    print(f"\n{'='*70}")
    print(f"BASE DATA BENCHMARK — SOAP vs Dialogue ({n} samples)")
    print(f"{'='*70}")
    print(f"  ROUGE-1:    P={avg(lambda r: r['rouge']['rouge1']['precision']):.4f}  "
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
    print(f"{'='*70}")

    # Write results
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

    output_data = {"summary": summary, "per_sample": results}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
