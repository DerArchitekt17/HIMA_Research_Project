# Agentic Architectures for Clinical Consultation Intelligence

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
├── single_agent/                       # Single-agent architecture
│   ├── train.py                        # QLoRA fine-tuning script
│   ├── benchmark.py                    # Evaluation script (ROUGE + BERTScore)
│   ├── run_training.slurm              # SLURM job for training
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training_single.jsonl       # 8,024 training examples
│   │   └── validation_single.jsonl     # 2,006 validation examples
│   └── benchmark_results/
│       └── hima_single_benchmark_n0.json # Benchmark results - all samples
│
├── multi_agents/                       # Multi-agent architecture
│   ├── train.py                        # Per-agent QLoRA fine-tuning script
│   ├── benchmark.py                    # Multi-agent evaluation script
│   ├── run_training.slurm              # SLURM job for training (parallel agents)
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training/                   # 4 JSONL files (4 dimensions)
│   │   ├── validation/                 # 4 JSONL files (4 dimensions)
│   │   └── benchmark_multi_agent.jsonl # Combined reference for evaluation
│   └── benchmark_results/              # Benchmark results - all samples
│
├── swarm_agents/                       # Swarm-agent architecture (DCR pipeline)
│   ├── train.py                        # Per-role/dimension QLoRA fine-tuning
│   ├── benchmark.py                    # DCR pipeline evaluation script
│   ├── run_training.slurm              # SLURM job for training (12 agents)
│   ├── run_training.sh                 # Native training script
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   ├── data/
│   │   ├── training/                   # 12 JSONL files (3 roles × 4 dimensions)
│   │   ├── validation/                 # 12 JSONL files (3 roles × 4 dimensions)
│   │   └── benchmark_swarm_agents.jsonl
│   └── benchmark_results/              # Benchmark results - all samples
│
├── baseline/                           # Baseline: ground-truth SOAP vs. dialogue
│   ├── benchmark.py                    # Metrics-only script (no model inference)
│   ├── benchmark_baseline.jsonl        # Ground-truth SOAP + dialogue pairs
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── run_benchmark.sh                # Native benchmark script
│   └── benchmark_results/
│       └── base_data_benchmark_n0.json # Benchmark results - all samples
│
├── single_agent_small/                 # 3B model study - single-agent / Same data as 8B counterpart
│
├── multi_agents_small/                 # 3B model study - multi-agent / Same data as 8B counterpart
│
└── swarm_agents_small/                 # 3B model study - swarm-agent / Same data as 8B counterpart
```

## Dataset

The project uses the **MedSynth** dataset (`MedSynth_huggingface_final.csv`), a synthetic medical dataset containing 10,240 patient-doctor consultation dialogues paired with structured SOAP notes and ICD-10 diagnosis codes.

The data preparation pipeline ([data_preperation.ipynb](data_preperation.ipynb)) applies the following steps:

1. **Cleaning**: Unicode normalization (NFKC), removal of zero-width and control characters
2. **SOAP extraction**: Regex-based parsing of Subjective, Objective, Assessment, and Plan sections
3. **ICD-10 balancing**: Filtering to 5 examples per diagnosis code, yielding 10,030 records across 2,006 unique diagnoses
4. **Stratified split**: 8,024 training / 2,006 validation examples, stratified by ICD-10 code

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

Evaluation on the full validation set (2,006 samples):

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.664   |  0.355  | 0.461 |
| ROUGE-2   |   0.328   |  0.176  | 0.228 |
| ROUGE-L   |   0.470   |  0.251  | 0.326 |
| BERTScore |   0.820   |  0.830  | 0.825 |

### 8B Model (Ministral-3-8B-Reasoning-2512)

Benchmark evaluation on the full validation set (2,006 samples). All scores are F1.

#### Single-Agent

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.781   |  0.772  | 0.774 |
| ROUGE-2   |   0.544   |  0.547  | 0.548 |
| ROUGE-L   |   0.632   |  0.624  | 0.626 |
| BERTScore |   0.929   |  0.927  | 0.928 |

#### Multi-Agent (Combined)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.781   |  0.739  | 0.756 |
| ROUGE-2   |   0.544   |  0.514  | 0.526 |
| ROUGE-L   |   0.615   |  0.582  | 0.596 |
| BERTScore |   0.922   |  0.920  | 0.921 |

#### Multi-Agent (Per Agent)

| Agent      | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |    0.735   |   0.497    |   0.599    |    0.938     |
| Objective  |    0.763   |   0.622    |   0.711    |    0.941     |
| Assessment |    0.430   |   0.299    |   0.389    |    0.886     |
| Plan       |    0.734   |   0.519    |   0.573    |    0.925     |

#### Swarm-Agent (Combined - Refiner Output)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.767   |  0.757  | 0.758 |
| ROUGE-2   |   0.537   |  0.529  | 0.530 |
| ROUGE-L   |   0.605   |  0.596  | 0.597 |
| BERTScore |   0.922   |  0.920  | 0.921 |

#### Swarm-Agent (Per Dimension - Refiner Output)

| Dimension  | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |    0.731   |   0.492    |   0.593    |    0.937     |
| Objective  |    0.777   |   0.639    |   0.730    |    0.944     |
| Assessment |    0.469   |   0.331    |   0.423    |    0.891     |
| Plan       |    0.729   |   0.522    |   0.578    |    0.926     |

#### Swarm-Agent Ablation (Drafter-Only vs. Refiner)

Comparing Drafter output (before critique-refine) against the final Refiner output to quantify the contribution of the DCR loop:

| Dimension  | Stage   | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|---------|------------|------------|------------|--------------|
| Subjective | Drafter |    0.743   |   0.507    |   0.609    |    0.939     |
| Subjective | Refiner |    0.731   |   0.492    |   0.593    |    0.937     |
| Objective  | Drafter |    0.772   |   0.631    |   0.720    |    0.941     |
| Objective  | Refiner |    0.777   |   0.639    |   0.730    |    0.944     |
| Assessment | Drafter |    0.459   |   0.321    |   0.419    |    0.890     |
| Assessment | Refiner |    0.469   |   0.331    |   0.423    |    0.891     |
| Plan       | Drafter |    0.723   |   0.508    |   0.559    |    0.924     |
| Plan       | Refiner |    0.729   |   0.522    |   0.578    |    0.926     |

### 3B Model (Ministral-3-3B-Reasoning-2512)

The 3B study uses the same data, hyperparameters, and evaluation pipeline as the 8B study, with [Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) as the base model. This enables a direct comparison of architectural patterns across model scales.

#### Single-Agent (3B)

> Results pending - benchmarks not yet completed.

#### Multi-Agent (3B, Combined)

> Results pending - benchmarks not yet completed.

#### Multi-Agent (3B, Per Agent)

> Results pending - benchmarks not yet completed.

#### Swarm-Agent (3B, Combined)

> Results pending - benchmarks not yet completed.

#### Swarm-Agent (3B, Per Dimension)

> Results pending - benchmarks not yet completed.

## Reproduction

### Prerequisites

- Linux environment with NVIDIA GPUs (3x A100-40GB recommended)
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
   This produces `apptainer_runtime.sif`, which the `.slurm` scripts reference to execute training and benchmarking inside the container.

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
   huggingface-cli login --token YOUR_TOKEN
   ```

2. **Download models and copy them into all architecture dirs** (~15 GB)
   ```bash
   ./download_models.sh
   ```

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
cd single_agent && nohup bash run_training.sh > training.log 2>&1 & cd ..
cd multi_agents && nohup bash run_training.sh > training.log 2>&1 & cd ..
cd swarm_agents && nohup bash run_training.sh > training.log 2>&1 & cd ..

# 3B model
cd single_agent_small && nohup bash run_training.sh > training.log 2>&1 & cd ..
cd multi_agents_small && nohup bash run_training.sh > training.log 2>&1 & cd ..
cd swarm_agents_small && nohup bash run_training.sh > training.log 2>&1 & cd ..
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
cd baseline && bash run_benchmark.sh && cd ..

# 8B model
cd single_agent && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..
cd multi_agents && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..
cd swarm_agents && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..

# 3B model
cd single_agent_small && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..
cd multi_agents_small && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..
cd swarm_agents_small && nohup bash run_benchmark.sh > benchmark.log 2>&1 & cd ..
```

Results are saved to `benchmark_results/` in each architecture folder.

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
  title   = {Agentic Architectures for Clinical Consultation Intelligence},
  author  = {<author names>},
  year    = {2026},
  note    = {Submitted for grading and peer review}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt this work for non-commercial purposes, provided appropriate credit is given.
