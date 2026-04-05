# Agentic AI Architectures for Clinical Consultation Intelligence

This repository contains the code, data, and experiment configurations for reproducing the results presented in the research paper *"Agentic Architectures for Clinical Consultation Intelligence"*.

The project investigates the use of fine-tuned large language models to automatically generate structured SOAP (Subjective, Objective, Assessment, Plan) clinical notes from patient-doctor consultation dialogues. Three architectural approaches are compared: a **single-agent** model that generates the complete note end-to-end, a **multi-agent** system with four specialist models (each responsible for one SOAP dimension), and a **swarm-agent** system using a Draft-Critique-Refine (DCR) pipeline with 12 specialist adapters. All 3 architectures are evaluated at two model scales (8B and 3B parameters) to study the effect of model size on clinical note generation quality.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Architectures](#architectures)
- [Results](#results)
- [Reproduction](#reproduction)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

## Overview

Clinical documentation is a time-intensive task for healthcare professionals. This project explores whether agentic LLM architectures can reliably generate SOAP notes from consultation transcripts. The primary study uses [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) as the base model. A companion study using [Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) enables direct comparison across model sizes. All models are fine-tuned with 4-bit QLoRA and evaluated using ROUGE and BERTScore metrics.

## Repository Structure

```
.
├── README.md
├── MedSynth_huggingface_final.csv      # Source dataset (10,240 records)
├── data_preperation.ipynb              # Data cleaning, balancing, and split pipeline
├── environment.yml                     # Conda environment specification
├── apptainer_runtime.def               # Apptainer container definition
├── create_conda_env.sh                 # Conda environment setup script
├── build_apptainer_image.sh            # Apptainer image build script
├── setup_native_runtime.sh             # Native runtime setup (no container/conda)
├── download_models.sh                  # Model download script (~15 GB)
│
├── models/                             # Shared base models (downloaded once)
│   ├── ministral3_8B/                  # Ministral-3-8B-Reasoning-2512
│   ├── ministral3_3B/                  # Ministral-3-3B-Reasoning-2512
│   └── bertscore/                      # roberta-large (for BERTScore evaluation)
│
├── single_agent/                       # Single-agent architecture (8B)
│   ├── train.py                        # QLoRA fine-tuning script
│   ├── benchmark.py                    # Evaluation script (ROUGE + BERTScore)
│   ├── run_training.slurm              # SLURM job for training
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training_single.jsonl       # ~7,222 training examples
│   │   ├── validation_single.jsonl     # ~802 validation examples
│   │   └── test/
│   │       └── test_full.jsonl         # 2,006 test examples (1 per ICD code)
│   ├── finetuned_adapters/             # LoRA adapters (output of training)
│   └── benchmark_results/
│
├── multi_agents/                       # Multi-agent architecture (8B)
│   ├── train.py                        # Per-agent QLoRA fine-tuning script
│   ├── benchmark.py                    # Multi-agent evaluation script
│   ├── run_training.slurm              # SLURM job for training (parallel agents)
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training/                   # 4 JSONL files (4 dimensions)
│   │   ├── validation/                 # 4 JSONL files (4 dimensions)
│   │   └── test/                       # test_full.jsonl + 4 per-dimension test files
│   ├── finetuned_adapters/             # LoRA adapters per SOAP dimension
│   └── benchmark_results/
│
├── swarm_agents/                       # Swarm-agent architecture (8B, DCR pipeline)
│   ├── train.py                        # Per-role/dimension QLoRA fine-tuning
│   ├── benchmark.py                    # DCR pipeline evaluation script
│   ├── run_training.slurm              # SLURM job for training (12 agents)
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training/                   # 12 JSONL files (3 roles × 4 dimensions)
│   │   ├── validation/                 # 12 JSONL files (3 roles × 4 dimensions)
│   │   └── test/                       # test_full.jsonl + 12 per-role/dimension test files
│   ├── finetuned_adapters/             # LoRA adapters per role × dimension
│   └── benchmark_results/
│
├── baseline/                           # Baseline: ground-truth SOAP vs. dialogue
│   ├── benchmark.py                    # Metrics-only script (no model inference)
│   ├── data/
│   │   └── test/
│   │       └── test_full.jsonl         # Ground-truth SOAP + dialogue pairs
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   └── benchmark_results/
│
├── single_agent_small/                 # 3B model study - single-agent / Same data as 8B counterpart
│
├── multi_agents_small/                 # 3B model study - multi-agent / Same data as 8B counterpart
│
└── swarm_agents_small/                 # 3B model study - swarm-agent / Same data as 8B counterpart
```

## Dataset

The project uses the **MedSynth** dataset (`MedSynth_huggingface_final.csv`), a synthetic medical dataset containing 10,240 patient-doctor consultation dialogues paired with structured SOAP notes and ICD-10 diagnosis codes. Read their paper: ([MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs](https://arxiv.org/pdf/2508.01401))

The data preparation pipeline ([data_preperation.ipynb](data_preperation.ipynb)) applies the following steps:

1. **Cleaning**: Unicode normalization (NFKC), removal of zero-width and control characters
2. **SOAP extraction**: Regex-based parsing of Subjective, Objective, Assessment, and Plan sections
3. **ICD-10 balancing**: Filtering to 5 examples per diagnosis code, yielding 10,030 records across 2,006 unique diagnoses
4. **Stratified split** (70/10/20): 1 random sample per ICD-10 code is held out as the **test set** (2,006 examples). From the remaining 4 samples per code, 10% are randomly sampled for **validation** (~802 examples) and the rest form the **training set** (~7,222 examples). A fixed seed ensures reproducibility.

## Architectures

### Single-Agent

A single fine-tuned model receives a consultation dialogue and generates the complete SOAP note including ICD-10 codes in one pass.

| Parameter         | Value                                  |
|-------------------|----------------------------------------|
| Base model        | Ministral-3-8B-Reasoning-2512          |
| Method            | QLoRA (4-bit NF4, bfloat16)            |
| LoRA rank / alpha | 64 / 32                                |
| Target modules    | q, k, v, o, gate, up, down projections |
| Epochs            | 3                                      |
| Batch size        | 2 (gradient accumulation: 4)           |
| Learning rate     | 2e-4 (cosine schedule)                 |
| Hardware          | 3x NVIDIA A100-40GB                    |

### Multi-Agent

Four independently fine-tuned specialist models, each generating one SOAP section:

- **Subjective Agent**: Chief complaint, HPI, review of systems
- **Objective Agent**: Vital signs, physical exam, test results
- **Assessment Agent**: Diagnosis, ICD-10 codes, differentials
- **Plan Agent**: Medications, referrals, follow-up, patient education

Each agent uses the same QLoRA configuration as the single-agent model, except that each agent trains on a single GPU:

| Parameter         | Value                                  |
|-------------------|----------------------------------------|
| Base model        | Ministral-3-8B-Reasoning-2512          |
| Method            | QLoRA (4-bit NF4, bfloat16)            |
| LoRA rank / alpha | 64 / 32                                |
| Target modules    | q, k, v, o, gate, up, down projections |
| Epochs            | 3                                      |
| Batch size        | 2 (gradient accumulation: 4)           |
| Learning rate     | 2e-4 (cosine schedule)                 |
| Hardware          | 3x NVIDIA A100-40GB                    |

During training, all 4 agents were trained sequentially accross all available GPUs.

### Swarm-Agent (Draft-Critique-Refine)

A swarm of 12 fine-tuned adapters organized into a three-stage pipeline for each SOAP dimension:

1. **Drafter**: Generates an initial section from the dialogue (plus prior context)
2. **Critic**: Reviews the draft against the source dialogue, flagging hallucinations, omissions, and formatting issues
3. **Refiner**: Produces the final section by incorporating the critic's feedback

This yields 12 adapters in total (3 roles × 4 SOAP dimensions). All share the same base model and QLoRA configuration:

| Parameter         | Value                                  |
|-------------------|----------------------------------------|
| Base model        | Ministral-3-8B-Reasoning-2512          |
| Method            | QLoRA (4-bit NF4, bfloat16)            |
| LoRA rank / alpha | 64 / 32                                |
| Target modules    | q, k, v, o, gate, up, down projections |
| Epochs            | 3                                      |
| Batch size        | 2 (gradient accumulation: dynamic)     |
| Learning rate     | 2e-4 (cosine schedule)                 |
| Max seq. length   | 8,192                                  |
| Hardware          | 3x NVIDIA A100-40GB                    |

Training data for the critic and refiner agents is generated synthetically by cross-pairing drafts within ICD-10 codes (see [data_preperation.ipynb](data_preperation.ipynb)).

### Inference

During inference, each architecture generates SOAP notes from consultation dialogues using greedy decoding. Samples are distributed across GPUs via round-robin.

| Parameter             | Single-Agent             | Multi-Agent                           | Swarm-Agent                                |
|-----------------------|--------------------------|---------------------------------------|--------------------------------------------|
| Adapters loaded       | 1                        | 4 (hot-swapped on same base model)    | 12 (hot-swapped: 3 roles × 4 dimensions)  |
| Passes per sample     | 1 (full SOAP note)       | 4 sequential (S → O → A → P)          | 12 sequential (D→C→R × 4 dimensions)      |
| Context accumulation  | N/A                      | Yes (each agent sees prior sections)  | Yes (within and across dimensions)         |
| Decoding strategy     | Greedy                   | Greedy                                | Greedy                                     |
| Max new tokens        | 2,048                    | 2,048 per agent                       | 2,048 per agent                            |
| Hardware              | 3x A100-40GB, 64 GB RAM  | 3x A100-40GB, 128 GB RAM              | 3x A100-40GB, 256 GB RAM                  |

## Results

### Baseline (Ground-Truth SOAP vs. Dialogue)

As a reference point, the ground-truth SOAP notes from the dataset are scored directly against their source consultation dialogues. No model inference is involved - this measures the inherent textual overlap between the raw dialogue and the structured SOAP output. Because SOAP notes reorganize, summarize, and add clinical structure to the conversation, scores are expected to be low. Any fine-tuned model should substantially exceed this baseline.

Evaluation on the held-out test set (2,006 samples):

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.6638  |  0.3560 | 0.4611 |
| ROUGE-2   |   0.3273  |  0.1755 | 0.2274 |
| ROUGE-L   |   0.4686  |  0.2514 | 0.3256 |
| BERTScore |   0.8198  |  0.8296 | 0.8246 |

### 8B Model (Ministral-3-8B-Reasoning-2512)

Benchmark evaluation on the held-out test set (2,006 samples). All scores are F1.

#### Single-Agent

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Multi-Agent (Combined)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Multi-Agent (Per Agent)

| Agent      | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |            |            |            |              |
| Objective  |            |            |            |              |
| Assessment |            |            |            |              |
| Plan       |            |            |            |              |

#### Swarm-Agent (Combined - Refiner Output)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Swarm-Agent (Per Dimension - Refiner Output)

| Dimension  | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |            |            |            |              |
| Objective  |            |            |            |              |
| Assessment |            |            |            |              |
| Plan       |            |            |            |              |

#### Swarm-Agent Ablation (Drafter-Only vs. Refiner)

Comparing Drafter output (before critique-refine) against the final Refiner output to quantify the contribution of the DCR loop:

| Dimension  | Stage   | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|---------|------------|------------|------------|--------------|
| Subjective | Drafter |            |            |            |              |
| Subjective | Refiner |            |            |            |              |
| Objective  | Drafter |            |            |            |              |
| Objective  | Refiner |            |            |            |              |
| Assessment | Drafter |            |            |            |              |
| Assessment | Refiner |            |            |            |              |
| Plan       | Drafter |            |            |            |              |
| Plan       | Refiner |            |            |            |              |

### 3B Model (Ministral-3-3B-Reasoning-2512)

The 3B study uses the same data, hyperparameters, and evaluation pipeline as the 8B study, with [Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) as the base model. This enables a direct comparison of architectural patterns across model scales.

#### Single-Agent (3B)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Multi-Agent (3B, Combined)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Multi-Agent (3B, Per Agent)

| Agent      | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |            |            |            |              |
| Objective  |            |            |            |              |
| Assessment |            |            |            |              |
| Plan       |            |            |            |              |

#### Swarm-Agent (3B, Combined - Refiner Output)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |           |         |       |
| ROUGE-2   |           |         |       |
| ROUGE-L   |           |         |       |
| BERTScore |           |         |       |

#### Swarm-Agent (3B, Per Dimension - Refiner Output)

| Dimension  | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |            |            |            |              |
| Objective  |            |            |            |              |
| Assessment |            |            |            |              |
| Plan       |            |            |            |              |

#### Swarm-Agent (3B) Ablation (Drafter-Only vs. Refiner)

Comparing Drafter output (before critique-refine) against the final Refiner output to quantify the contribution of the DCR loop:

| Dimension  | Stage   | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|---------|------------|------------|------------|--------------|
| Subjective | Drafter |            |            |            |              |
| Subjective | Refiner |            |            |            |              |
| Objective  | Drafter |            |            |            |              |
| Objective  | Refiner |            |            |            |              |
| Assessment | Drafter |            |            |            |              |
| Assessment | Refiner |            |            |            |              |
| Plan       | Drafter |            |            |            |              |
| Plan       | Refiner |            |            |            |              |

## Reproduction

### Prerequisites

- Linux environment with NVIDIA GPUs (3x A100-40GB recommended) and minimum CUDA 13.0
- [Weights & Biases](https://wandb.ai/) account (optional, for experiment tracking)
- [Hugging Face](https://huggingface.co/) account with access to the Ministral model

**Choose one of the two setup options below:**
| | Option 1: SLURM + Apptainer | Option 2: Native |
|---|---|---|
| **Environment** | HPC cluster with job scheduler | Any Linux machine with GPUs |
| **Isolation** | Conda env + Apptainer container | Packages installed directly on system |
| **Additional tools** | [Conda](https://docs.conda.io/), [Apptainer](https://apptainer.org/), SLURM | None |
| **Run training/benchmarks** | `sbatch <dir>/run_training.slurm` | `bash <dir>/run_training.sh` |

Pick **one** option and follow it consistently. Do not mix `.slurm` and `.sh` scripts.

### Setup

#### Option 1: SLURM + Apptainer (HPC Clusters)

Use this option on shared HPC systems where you cannot install packages globally and jobs are submitted via SLURM. A Conda environment provides the build toolchain for Apptainer, which packages all Python dependencies into a portable container image.

1. **Create the Conda environment**
   ```bash
   ./create_conda_env.sh
   conda activate hima_research
   ```

2. **Build the Apptainer image** (~15 minutes)
   ```bash
   ./build_apptainer_image.sh
   ```
   This produces `apptainer_runtime.sif` in the project root, which the `.slurm` scripts reference to execute training and benchmarking inside the container.

#### Option 2: Native Installation

Use this option when you have direct access to a GPU machine and can install packages system-wide (or into an existing virtual environment). No container or Conda environment is needed.

1. **Run the native setup script**
   ```bash
   bash setup_native_runtime.sh
   ```
   This installs all Python dependencies (including flash-attn), verifies CUDA availability, and sets environment variables.

#### Common Steps (both options)

1. **Authenticate with Hugging Face**
   ```bash
   hf auth login --token YOUR_TOKEN
   ```

2. **Download models to the shared `models/` directory** (~15 GB)
   ```bash
   ./download_models.sh
   ```
   This downloads the base models once into `models/ministral3_8B/`, `models/ministral3_3B/`, and `models/bertscore/`. All architectures reference these shared copies.

3. **Login to Weights & Biases** (optional)
   ```bash
   wandb login
   ```

### Training

Submit training for the desired architecture using the method that matches your setup option:

**Option 1 - SLURM:**
```bash
# 8B model
sbatch single_agent/run_training.slurm
sbatch multi_agents/run_training.slurm
sbatch swarm_agents/run_training.slurm

# 3B model
sbatch single_agent_small/run_training.slurm
sbatch multi_agents_small/run_training.slurm
sbatch swarm_agents_small/run_training.slurm
```

**Option 2 - Native:**
```bash
# 8B model
nohup bash single_agent/run_training.sh > single_agent/training.log 2>&1 &
nohup bash multi_agents/run_training.sh > multi_agents/training.log 2>&1 &
nohup bash swarm_agents/run_training.sh > swarm_agents/training.log 2>&1 &

# 3B model
nohup bash single_agent_small/run_training.sh > single_agent_small/training.log 2>&1 &
nohup bash multi_agents_small/run_training.sh > multi_agents_small/training.log 2>&1 &
nohup bash swarm_agents_small/run_training.sh > swarm_agents_small/training.log 2>&1 &
```

(Optional / If available) Asyncronous WandB syncing (~30 sec interval)
```bash
nohup ./wandb_sync.sh > wandb_sync.log 2>&1 &
```

OR

(Optional) After training, manually sync WandB runs to your online account:
```bash
for exp in single_agent multi_agents swarm_agents single_agent_small multi_agents_small swarm_agents_small; do
    shopt -s nullglob
    local runs=( ${exp}/wandb/offline-run-* )
    shopt -u nullglob
    if [[ ${#runs[@]} -gt 0 ]]; then
      echo "[$(date)] [${exp}] Found ${#runs[@]} offline run(s). Syncing..."
      wandb sync --include-offline ${exp}/wandb/offline-run-* \
        || echo "[$(date)] [${exp}] wandb sync returned non-zero; continuing."
      synced=$((synced + ${#runs[@]}))
    fi
  done
```

### Benchmarking

Configure benchmark parameters (e.g., number of samples) in the respective `run_benchmark.slurm` or `run_benchmark.sh` file, then run:

**Option 1 - SLURM:**
```bash
# Baseline (no model inference - scores ground-truth SOAP vs. dialogue)
sbatch baseline/run_benchmark.slurm

# 8B model
sbatch single_agent/run_benchmark.slurm
sbatch multi_agents/run_benchmark.slurm
sbatch swarm_agents/run_benchmark.slurm

# 3B model
sbatch single_agent_small/run_benchmark.slurm
sbatch multi_agents_small/run_benchmark.slurm
sbatch swarm_agents_small/run_benchmark.slurm
```

**Option 2 - Native:**
```bash
# Baseline
bash baseline/run_benchmark.sh

# 8B model
nohup bash single_agent/run_benchmark.sh > single_agent/benchmark.log 2>&1 &
nohup bash multi_agents/run_benchmark.sh > multi_agents/benchmark.log 2>&1 &
nohup bash swarm_agents/run_benchmark.sh > swarm_agents/benchmark.log 2>&1 &

# 3B model
nohup bash single_agent_small/run_benchmark.sh > single_agent_small/benchmark.log 2>&1 &
nohup bash multi_agents_small/run_benchmark.sh > multi_agents_small/benchmark.log 2>&1 &
nohup bash swarm_agents_small/run_benchmark.sh > swarm_agents_small/benchmark.log 2>&1 &
```

Results are saved to `benchmark_results/` in each architecture folder. Trained LoRA adapters are saved to `finetuned_adapters/` in each architecture folder.

## Requirements

Key dependencies (full list in [environment.yml](environment.yml)):

|     Package      |  Ver.  |
|------------------|--------|
| Python           | 3.x    |
| PyTorch          | 2.8.0  |
| Transformers     | 4.57.1 |
| PEFT             | 0.14.0 |
| TRL              | latest |
| BitsAndBytes     | latest |
| Weights & Biases | latest |
| rouge-score      | latest |
| bert-score       | latest |

## Citation

> **Note**: This paper has been submitted for academic grading and scientific peer review and is not yet published. A formal citation will be added upon publication.

```
@unpublished{<placeholder>,
  title   = {Agentic AI Architectures for Clinical Consultation Intelligence},
  author  = {<author names>},
  year    = {2026},
  note    = {Submitted for grading and peer review}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt this work for non-commercial purposes, provided appropriate credit is given.
