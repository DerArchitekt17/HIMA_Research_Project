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
from accelerate import Accelerator
import wandb

acc = Accelerator()

# Configure multi GPU use
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device_map = {"": local_rank}   # or {"": torch.cuda.current_device()}

# Configure WandB logging
if not acc.is_main_process:
    # absolutely prevent rank>0 from creating a run
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

# Register missing ministral3 text config
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# Set paths and environment variables
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
MODEL_DIR = os.path.join(BASE_DIR, "basemodel")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "finetuned_models")
WANDB_DIR = os.path.join(BASE_DIR, "wandb")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WANDB_DIR, exist_ok=True)

# Ensure online functions are configured as offline services (reinforcement)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = WANDB_DIR

wandb.init(project=os.getenv("WANDB_PROJECT", ""), name=os.getenv("WANDB_NAME", ""), dir=WANDB_DIR)

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

# Loading model - device_map="auto" distributes across all available GPUs
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# Log GPU distribution
print(f"Model distributed across devices: {set(model.hf_device_map.values())}")
print(f"Total GPUs available: {torch.cuda.device_count()}")

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
model.print_trainable_parameters()

# Loading and preparing dataset
MAX_SEQ_LENGTH = 4096

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "training/training_single.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation/validation_single.jsonl"),
    },
)

def preprocess(examples):
    texts = []
    for msgs in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Training config
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
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
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    max_seq_length=MAX_SEQ_LENGTH,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
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
    print(f"Resuming from {latest}")
    trainer.train(resume_from_checkpoint=latest)
else:
    trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
wandb.finish()
print(f"Training complete. Adapter saved to {os.path.join(OUTPUT_DIR, 'final_adapter')}")
