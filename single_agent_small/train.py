"""
Fine-tune Ministral-3-8B with QLoRA on medical consultation -> SOAP note data.
Trains a single model to generate complete SOAP notes (all 4 dimensions).
Designed for offline, multi-GPU execution via accelerate + DeepSpeed.

Usage:
    Invoked by run_training.slurm.
    No agent argument needed - trains single unified model.
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
import glob
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb

# Distributed training setup (set by accelerate launch)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN_PROCESS = LOCAL_RANK == 0

if IS_MAIN_PROCESS:
    print(f"Distributed setup: {WORLD_SIZE} GPU(s) detected")

# Register missing ministral3 text config
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# Set paths and environment variables
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
MODEL_DIR = os.path.join(BASE_DIR, "basemodel")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "finetuned_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure online functions are configured as offline services (reinforcement)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

# Only initialize wandb on main process to avoid duplicate logging
if IS_MAIN_PROCESS:
    wandb.init(project=os.getenv("WANDB_PROJECT", ""), name=os.getenv("WANDB_NAME", ""), dir=BASE_DIR)

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantisation config (4-bit QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Loading model — each process loads onto its assigned GPU via LOCAL_RANK
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map={"": LOCAL_RANK},
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if IS_MAIN_PROCESS:
    model.print_trainable_parameters()

# Loading and preparing dataset
MAX_SEQ_LENGTH = 4096

dataset = load_dataset(
    "json",
    data_files={
        "training": os.path.join(DATA_DIR, "training_single.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation_single.jsonl"),
    },
)

def preprocess(examples):
    texts = []
    for msgs in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["training"].column_names)

# Dynamic gradient accumulation to keep effective batch size constant
# Baseline: 2 GPUs x batch 2 x grad_accum 4 = effective batch 16
PER_DEVICE_BATCH = 2
TARGET_EFFECTIVE_BATCH = 16
grad_accum = max(1, TARGET_EFFECTIVE_BATCH // (PER_DEVICE_BATCH * WORLD_SIZE))

if IS_MAIN_PROCESS:
    effective = PER_DEVICE_BATCH * WORLD_SIZE * grad_accum
    print(f"Batch config: {PER_DEVICE_BATCH}/device x {WORLD_SIZE} GPUs x {grad_accum} accum = {effective} effective")

# Training config
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=False,
    report_to="wandb" if IS_MAIN_PROCESS else "none",
    max_seq_length=MAX_SEQ_LENGTH,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    ddp_find_unused_parameters=True,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["training"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    peft_config=None,  # already applied above
)

# Train & save for fault tolerance
# Resume from checkpoint if one exists, otherwise start fresh
checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoint-*")
checkpoints = glob.glob(checkpoint_dir)
if checkpoints:
    # Find latest checkpoint by step number
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    if IS_MAIN_PROCESS:
        print(f"Resuming from {latest}")
    trainer.train(resume_from_checkpoint=latest)
else:
    trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
if IS_MAIN_PROCESS:
    wandb.finish()
    print(f"Training complete. Adapter saved to {os.path.join(OUTPUT_DIR, 'final_adapter')}")
