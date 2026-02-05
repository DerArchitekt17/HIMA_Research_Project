# Agentic Architectures for Clinical Consultation Intelligence

This repository contains the code, data, and experiment configurations for reproducing the results presented in the research paper *"Agentic Architectures for Clinical Consultation Intelligence"*.

The project investigates the use of fine-tuned large language models to automatically generate structured SOAP (Subjective, Objective, Assessment, Plan) clinical notes from patient-doctor consultation dialogues. Two architectural approaches are compared: a **single-agent** model that generates the complete note end-to-end, and a **multi-agent** system with four specialist models, each responsible for one SOAP dimension.

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

Clinical documentation is a time-intensive task for healthcare professionals. This project explores whether agentic LLM architectures can reliably generate SOAP notes from consultation transcripts. All models are built on [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512), fine-tuned with 4-bit QLoRA, and evaluated using ROUGE and BERTScore metrics on a held-out validation set of 500 samples.

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
├── download_models.sh                  # Model download script (~15 GB)
│
├── single_agent/                       # Single-agent architecture
│   ├── train.py                        # QLoRA fine-tuning script
│   ├── benchmark.py                    # Evaluation script (ROUGE + BERTScore)
│   ├── run_training.slurm              # SLURM job for training
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── data/
│   │   ├── training_single.jsonl       # 8,024 training examples
│   │   └── validation_single.jsonl     # 2,006 validation examples
│   └── benchmark_results/
│       └── hima_single_benchmark_n500_research.json
│
├── multi_agents/                       # Multi-agent architecture
│   ├── train.py                        # Per-agent QLoRA fine-tuning script
│   ├── benchmark.py                    # Multi-agent evaluation script
│   ├── run_training.slurm              # SLURM job for training (parallel agents)
│   ├── run_benchmark.slurm             # SLURM job for benchmarking
│   ├── data/
│   │   ├── training/
│   │   │   ├── training_subjective.jsonl
│   │   │   ├── training_objective.jsonl
│   │   │   ├── training_assessment.jsonl
│   │   │   └── training_plan.jsonl
│   │   ├── validation/
│   │   │   ├── validation_subjective.jsonl
│   │   │   ├── validation_objective.jsonl
│   │   │   ├── validation_assessment.jsonl
│   │   │   └── validation_plan.jsonl
│   │   └── benchmark_multi_agent.jsonl # Combined reference for evaluation
│   └── benchmark_results/
│       └── hima_multi_benchmark_n500_research.json
│
└── swarm_agents/                       # Swarm-agent architecture (planned)
    └── README.md
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
| Hardware          | 2x NVIDIA A100-40GB                    |

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
| Hardware          | 1x NVIDIA A100-40GB per agent          |

During training, two agents are trained in parallel across two GPUs (batch 1: subjective + objective, batch 2: assessment + plan).

### Inference

During inference, each architecture generates SOAP notes from consultation dialogues using greedy decoding. Samples are distributed across GPUs via round-robin.

| Parameter             | Single-Agent             | Multi-Agent                           |
|-----------------------|--------------------------|---------------------------------------|
| Adapters loaded       | 1                        | 4 (hot-swapped on same base model)    |
| Passes per sample     | 1 (full SOAP note)       | 4 sequential (S → O → A → P)          |
| Context accumulation  | N/A                      | Yes (each agent sees prior sections)  |
| Decoding strategy     | Greedy                   | Greedy                                |
| Max new tokens        | 2,048                    | 2,048 per agent                       |
| Hardware              | 2x A100-40GB, 64 GB RAM  | 2x A100-40GB, 256 GB RAM              |

## Results

Benchmark evaluation on 500 validation samples. All scores are F1.

### Single-Agent

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.781   |  0.772  | 0.774 |
| ROUGE-2   |   0.544   |  0.547  | 0.548 |
| ROUGE-L   |   0.632   |  0.624  | 0.626 |
| BERTScore |   0.929   |  0.927  | 0.928 |

### Multi-Agent (Combined)

|  Metric   | Precision | Recall  |  F1   |
|-----------|-----------|---------|-------|
| ROUGE-1   |   0.781   |  0.739  | 0.756 |
| ROUGE-2   |   0.544   |  0.514  | 0.526 |
| ROUGE-L   |   0.615   |  0.582  | 0.596 |
| BERTScore |   0.922   |  0.920  | 0.921 |

### Multi-Agent (Per Agent)

| Agent      | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------|------------|------------|------------|--------------|
| Subjective |    0.735   |   0.497    |   0.599    |    0.938     |
| Objective  |    0.763   |   0.622    |   0.711    |    0.941     |
| Assessment |    0.430   |   0.299    |   0.389    |    0.886     |
| Plan       |    0.734   |   0.519    |   0.573    |    0.925     |

## Reproduction

### Prerequisites

- Linux environment with SLURM workload manager
- NVIDIA GPUs (2x A100-40GB recommended)
- [Conda](https://docs.conda.io/) and [Apptainer](https://apptainer.org/)
- [Weights & Biases](https://wandb.ai/) account
- [Hugging Face](https://huggingface.co/) account with access to the Ministral model

### Setup

1. **Create the conda environment**
   ```bash
   ./create_conda_env.sh
   ```

2. **Build the Apptainer image** (~15 minutes)
   ```bash
   ./build_apptainer_image.sh
   ```

3. **Login to Weights & Biases**
   ```bash
   wandb login
   ```

4. **Authenticate with Hugging Face**
   ```bash
   huggingface-cli login --token YOUR_TOKEN
   ```

5. **Download models and copy them into all architecture dirs** (~15 GB)
   ```bash
   ./download_models.sh
   ```

### Training

Submit the SLURM training job for the desired architecture:

```bash
# Single-agent
sbatch single_agent/run_training.slurm

# Multi-agent (trains all 4 agents in two parallel batches)
sbatch multi_agents/run_training.slurm
```

### Benchmarking

Configure benchmark parameters (e.g., number of samples) in the respective `run_benchmark.slurm` file, then submit:

```bash
# Single-agent
sbatch single_agent/run_benchmark.slurm

# Multi-agent
sbatch multi_agents/run_benchmark.slurm
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

> **Note**: This paper has been submitted for academic grading and is not yet published. A formal citation will be added upon publication.

```
@unpublished{<placeholder>,
  title   = {Agentic Architectures for Clinical Consultation Intelligence},
  author  = {<author names>},
  year    = {2026},
  note    = {Submitted for grading}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt this work for non-commercial purposes, provided appropriate credit is given.
