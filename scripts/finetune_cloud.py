"""
云端 QLoRA 微调脚本 — Qwen3-14B
在 AutoDL / Vast.ai 等云GPU平台运行

用法:
  python scripts/finetune_cloud.py              # 完整流程
  python scripts/finetune_cloud.py --prepare-only  # 仅格式化数据

硬件建议:
  - 最低 24GB 显存 (RTX 3090/A5000) — 使用4-bit量化
  - 推荐 A100 40GB 或 RTX 4090
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("finetune")

from src.training.shared import format_training_sample

FINETUNE_CONFIG = {
    "model_name": "Qwen/Qwen3-14B",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 1024,
    "warmup_ratio": 0.03,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "use_gradient_checkpointing": True,
}

def load_training_data(data_path: str) -> list[dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"加载 {len(samples)} 条训练样本")
    return samples


def format_sample(sample: dict) -> dict:
    return format_training_sample(sample)


def prepare_dataset(samples: list[dict], output_path: str):
    formatted = [format_sample(s) for s in samples]
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"数据集已保存: {output_path} ({len(formatted)} 条)")


def train():
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForSeq2Seq,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("pip install transformers datasets peft accelerate bitsandbytes")
        return

    config = FINETUNE_CONFIG
    logger.info("=" * 60)
    logger.info("Qwen3-14B QLoRA 微调配置:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    logger.info(f"加载模型: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if config["use_4bit"]:
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("使用4-bit量化加载模型 (QLoRA)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("使用bfloat16加载模型 (LoRA)")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_path = "training_data/training_data_formatted.jsonl"
    if not Path(dataset_path).exists():
        logger.error(f"数据集不存在: {dataset_path}")
        return

    dataset = Dataset.from_json(dataset_path)

    def tokenize_function(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for msgs in examples["messages"]:
            full_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False
            )
            if len(full_ids) > config["max_seq_length"]:
                full_ids = full_ids[:config["max_seq_length"]]
            labels = [-100] * len(full_ids)
            for i, msg in enumerate(msgs):
                if msg["role"] == "assistant":
                    end_ids = tokenizer.apply_chat_template(
                        msgs[:i + 1], tokenize=True, add_generation_prompt=False
                    )
                    start = 0 if i == 0 else len(
                        tokenizer.apply_chat_template(
                            msgs[:i], tokenize=True, add_generation_prompt=False
                        )
                    )
                    end = len(end_ids)
                    for j in range(start, min(end, len(full_ids))):
                        labels[j] = full_ids[j]
            all_input_ids.append(full_ids)
            all_attention_mask.append([1] * len(full_ids))
            all_labels.append(labels)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    warmup_steps = max(1, int(0.03 * len(tokenized_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])))

    training_args = TrainingArguments(
        output_dir="./qwen3-lora-output",
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=warmup_steps,
        learning_rate=config["learning_rate"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        gradient_checkpointing=config["use_gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("开始训练...")
    trainer.train()

    output_dir = "./qwen3-lora-finetuned"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA 权重已保存: {output_dir}")

    merged_dir = "./qwen3-lora-merged"
    logger.info(f"合并模型到: {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"合并模型已保存: {merged_dir}")


def prepare_and_train(data_path: str = "training_data/training_data.jsonl"):
    samples = load_training_data(data_path)
    if not samples:
        logger.error("无训练数据，请先运行: d2t training import <你的数据>")
        return

    os.makedirs("training_data", exist_ok=True)
    formatted_path = "training_data/training_data_formatted.jsonl"
    prepare_dataset(samples, formatted_path)

    train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-14B 云端 QLoRA 微调")
    parser.add_argument("--prepare-only", action="store_true", help="仅格式化数据")
    parser.add_argument("--data", default="training_data/training_data.jsonl")
    args = parser.parse_args()

    if args.prepare_only:
        samples = load_training_data(args.data)
        os.makedirs("training_data", exist_ok=True)
        prepare_dataset(samples, "training_data/training_data_formatted.jsonl")
    else:
        prepare_and_train(args.data)
