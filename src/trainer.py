#!/usr/bin/env python3
import os
import json
import shutil
import time
import torch
import wandb
from typing import Optional, Dict, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model

from .config import SYS_CFG, TRAIN_CFG
from .data_registry import DATASETS, load_and_format_dataset
from .utils_training import get_targets, infer_hot_k


# Helpers
def save_run_artifacts(out_dir: str, run_cfg: Dict[str, Any], targets: List[str], hotmap_json: Optional[str]):
    """Saves config and targets BEFORE training starts (Crash recovery)."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save Run Config
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # 2. Save Target Modules List
    with open(os.path.join(out_dir, "targets.json"), "w") as f:
        json.dump(targets, f, indent=2)

    # 3. Backup the Hotmap used
    if hotmap_json and os.path.exists(hotmap_json):
        shutil.copy(hotmap_json, os.path.join(out_dir, "hotmap_used.json"))


# MAIN TRAINING ROUTINE
def run_training(
    dataset_key: str,
    run_name: str,
    mode: str = "hot", # 'hot' or 'full'
    hotmap_json: Optional[str] = None,
    override_lr: float = None,
    override_epochs: int = None,
    train_samples: Optional[int] = None # For quick debug runs
):
    # Global Setup
    set_seed(TRAIN_CFG.seed)

    # Paths & Configs
    out_dir = SYS_CFG.get_output_dir(f"runs/{dataset_key}/{run_name}")
    ds_cfg = DATASETS[dataset_key]
    lr_eff = override_lr or ds_cfg["lr"]
    epochs_eff = override_epochs or ds_cfg["epochs"]
    hot_k = infer_hot_k(hotmap_json) if mode == "hot" else None

    # W&B Init (Robust Logging)
    if TRAIN_CFG.use_wandb:
        wandb.init(
            project=TRAIN_CFG.wandb_project,
            name=f"{dataset_key}_{run_name}",
            config={
                "dataset": dataset_key,
                "run_name": run_name,
                "mode": mode,
                "model_id": TRAIN_CFG.model_id,
                "seed": TRAIN_CFG.seed,
                "max_len": TRAIN_CFG.max_len,
                "bs_per_device": TRAIN_CFG.per_device_bs,
                "grad_acc": TRAIN_CFG.grad_acc,
                "epochs": epochs_eff,
                "learning_rate": lr_eff,
                "lora_r": TRAIN_CFG.r,
                "lora_alpha": TRAIN_CFG.alpha,
                "hotmap_path": hotmap_json,
                "hot_k": hot_k,
                "samples": train_samples or "full"
            }
        )

    # Model & Tokenizer
    print(f"⏳ Loading Model: {TRAIN_CFG.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CFG.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TRAIN_CFG.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Target Selection
    targets = get_targets(model, mode, hotmap_json)

    # PEFT Application
    peft_config = LoraConfig(
        r=TRAIN_CFG.r,
        lora_alpha=TRAIN_CFG.alpha,
        lora_dropout=TRAIN_CFG.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets
    )
    model = get_peft_model(model, peft_config)

    # Log Trainable Params
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\n📊 [{dataset_key}/{run_name}] Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    if TRAIN_CFG.use_wandb:
        wandb.log({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_frac": trainable/total
        })

    # Save Artifacts EARLY (Safety)
    save_run_artifacts(
        out_dir=out_dir,
        run_cfg={
            "dataset": dataset_key,
            "mode": mode,
            "lr": lr_eff,
            "epochs": epochs_eff,
            "lora": {"r": TRAIN_CFG.r, "alpha": TRAIN_CFG.alpha},
            "train_samples": train_samples
        },
        targets=targets,
        hotmap_json=hotmap_json
    )

    # Data Loading
    train_ds = load_and_format_dataset(
        dataset_key,
        tokenizer,
        TRAIN_CFG.max_len,
        n_samples=train_samples,
        seed=TRAIN_CFG.seed
    )

    # Training Arguments
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=TRAIN_CFG.per_device_bs,
        gradient_accumulation_steps=TRAIN_CFG.grad_acc,
        learning_rate=lr_eff,
        num_train_epochs=epochs_eff,
        bf16=True,
        logging_steps=TRAIN_CFG.logging_steps,
        save_strategy="epoch",      # Save checkpoint every epoch
        save_total_limit=2,         # Keep only last 2 checkpoints to save disk
        report_to="wandb" if TRAIN_CFG.use_wandb else "none",
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Execute Training
    print("🚀 Training Started...")
    t0 = time.time()
    trainer.train()
    t1 = time.time()

    duration = round(t1 - t0, 1)
    print(f"✅ Training Complete in {duration}s")
    if TRAIN_CFG.use_wandb:
        wandb.log({"train_seconds": duration})

    # Save Final Adapter
    final_path = os.path.join(out_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"💾 Saved Final Adapter -> {final_path}")

    if TRAIN_CFG.use_wandb:
        wandb.finish()

    # Cleanup
    del trainer, model
    torch.cuda.empty_cache()
