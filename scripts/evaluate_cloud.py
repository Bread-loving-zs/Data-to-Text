"""
云端评估脚本 — 对比微调前后效果并保存详细结果

用法:
  PYTHONPATH=. python scripts/evaluate_cloud.py
  PYTHONPATH=. python scripts/evaluate_cloud.py --num-samples 10
  PYTHONPATH=. python scripts/evaluate_cloud.py --base-only
  PYTHONPATH=. python scripts/evaluate_cloud.py --lora-only

结果保存到 results/evaluation_results.json
推送到 GitHub 后可在本地分析:
  git add results/ && git commit -m 'eval results' && git push
"""

import os
import json
import re
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

from src.data.utils import load_jsonl

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("evaluate_cloud")

from src.training.shared import format_training_sample, SYSTEM_PROMPT

LOCAL_MODEL_PATH = "/root/autodl-tmp/Qwen3-14B"
HF_MODEL_NAME = "Qwen/Qwen3-14B"
MODEL_PATH = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else HF_MODEL_NAME
LORA_PATH = "./qwen3-lora-finetuned"
DATA_PATH = "training_data/training_data.jsonl"
RESULTS_DIR = "results"


def load_samples(data_path: str, num_samples: int) -> list[dict]:
    samples = load_jsonl(data_path)
    logger.info(f"共加载 {len(samples)} 条样本，取前 {num_samples} 条")
    return samples[:num_samples]


def get_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    generated = generated.strip()
    generated = re.sub(r'<\s*/?\s*think\s*>', '', generated)
    generated = generated.strip()
    return generated


def compute_word_overlap(generated: str, reference: str) -> float:
    def tokenize(text):
        return set(text.replace("\n", " ").replace(" ", "").replace("　", ""))
    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0
    intersection = gen_tokens & ref_tokens
    union = gen_tokens | ref_tokens
    return len(intersection) / len(union)


def check_facts(generated: str, reference: str) -> dict:
    ref_numbers = set(re.findall(r'\d+\.?\d*', reference))
    gen_numbers = set(re.findall(r'\d+\.?\d*', generated))
    if not ref_numbers:
        return {
            "total": 0, "found": 0, "accuracy": 1.0,
            "reference_numbers": [], "generated_numbers": sorted(gen_numbers)[:20],
            "matched_numbers": [],
        }
    found = ref_numbers & gen_numbers
    return {
        "total": len(ref_numbers),
        "found": len(found),
        "accuracy": len(found) / len(ref_numbers),
        "reference_numbers": sorted(ref_numbers)[:20],
        "generated_numbers": sorted(gen_numbers)[:20],
        "matched_numbers": sorted(found)[:20],
    }


def evaluate_model(model, tokenizer, samples: list[dict], label: str) -> dict:
    results = []
    for i, sample in enumerate(samples):
        formatted = format_training_sample(sample)
        messages = formatted["messages"][:2]
        reference = formatted["messages"][2]["content"]

        logger.info(f"[{label}] 生成样本 {i+1}/{len(samples)}...")
        start = time.time()
        generated = generate(model, tokenizer, messages)
        elapsed = time.time() - start

        overlap = compute_word_overlap(generated, reference)
        facts = check_facts(generated, reference)

        results.append({
            "index": i,
            "input_preview": messages[1]["content"][:300],
            "reference": reference,
            "generated": generated,
            "word_overlap": round(overlap, 4),
            "fact_accuracy": round(facts["accuracy"], 4),
            "fact_details": facts,
            "time_seconds": round(elapsed, 1),
        })

        print(f"\n{'='*60}")
        print(f"[{label}] 样本 {i+1} | 词汇重叠: {overlap:.2%} | 事实准确率: {facts['accuracy']:.2%} | 耗时: {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"【参考输出】(前200字):\n{reference[:200]}...")
        print(f"\n【{label}输出】(前200字):\n{generated[:200]}...")

    avg_overlap = sum(r["word_overlap"] for r in results) / len(results)
    avg_facts = sum(r["fact_accuracy"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)

    summary = {
        "num_samples": len(results),
        "avg_word_overlap": round(avg_overlap, 4),
        "avg_fact_accuracy": round(avg_facts, 4),
        "avg_time_seconds": round(avg_time, 1),
    }

    print(f"\n{'#'*60}")
    print(f"{label}平均: 词汇重叠={avg_overlap:.2%} 事实准确率={avg_facts:.2%} 耗时={avg_time:.1f}s")
    print(f"{'#'*60}")

    return {"samples": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="云端评估脚本")
    parser.add_argument("--num-samples", type=int, default=5, help="测试样本数")
    parser.add_argument("--base-only", action="store_true", help="只测试基础模型")
    parser.add_argument("--lora-only", action="store_true", help="只测试微调模型")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    samples = load_samples(DATA_PATH, args.num_samples)

    base_result = None
    lora_result = None

    if not args.lora_only:
        logger.info("=" * 60)
        logger.info("测试基础模型 (未微调)")
        logger.info("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        logger.info("基础模型加载完成")

        base_result = evaluate_model(model, tokenizer, samples, "基础模型")

        del model
        torch.cuda.empty_cache()

    if not args.base_only:
        if not Path(LORA_PATH).exists():
            logger.warning(f"LoRA 权重不存在: {LORA_PATH}，跳过微调模型测试")
        else:
            logger.info("=" * 60)
            logger.info("测试微调模型 (LoRA)")
            logger.info("=" * 60)

            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=get_bnb_config(),
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, LORA_PATH)
            model.eval()
            logger.info("微调模型加载完成")

            lora_result = evaluate_model(model, tokenizer, samples, "微调模型")

            del model
            torch.cuda.empty_cache()

    comparison = None
    if base_result and lora_result:
        bs = base_result["summary"]
        ls = lora_result["summary"]
        comparison = {
            "word_overlap_improvement": round(ls["avg_word_overlap"] - bs["avg_word_overlap"], 4),
            "fact_accuracy_improvement": round(ls["avg_fact_accuracy"] - bs["avg_fact_accuracy"], 4),
            "time_change_seconds": round(ls["avg_time_seconds"] - bs["avg_time_seconds"], 1),
        }
        print(f"\n{'='*60}")
        print(f"微调前后对比:")
        print(f"{'='*60}")
        print(f"{'指标':<15} {'基础模型':>10} {'微调模型':>10} {'提升':>10}")
        print(f"{'-'*45}")
        print(f"{'词汇重叠率':<13} {bs['avg_word_overlap']:>10.2%} {ls['avg_word_overlap']:>10.2%} {comparison['word_overlap_improvement']*100:>+9.1f}%")
        print(f"{'事实准确率':<13} {bs['avg_fact_accuracy']:>10.2%} {ls['avg_fact_accuracy']:>10.2%} {comparison['fact_accuracy_improvement']*100:>+9.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_path": MODEL_PATH,
            "lora_path": LORA_PATH,
            "num_samples": args.num_samples,
        },
        "base_model": base_result,
        "lora_model": lora_result,
        "comparison": comparison,
    }

    results_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {results_file}")
    logger.info("推送到 GitHub 后可在本地分析: git add results/ && git commit -m 'eval' && git push")


if __name__ == "__main__":
    main()
