"""
云端推理测试脚本 — 对比微调前后效果
在 AutoDL 上运行

用法:
  PYTHONPATH=. python src/training/test_inference_cloud.py
  PYTHONPATH=. python src/training/test_inference_cloud.py --num-samples 5
  PYTHONPATH=. python src/training/test_inference_cloud.py --base-only
  PYTHONPATH=. python src/training/test_inference_cloud.py --lora-only
"""

import os
import json
import logging
import argparse
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_inference")

from src.training.shared import format_training_sample, SYSTEM_PROMPT

MODEL_PATH = "/root/autodl-tmp/Qwen3-14B"
LORA_PATH = "./qwen3-lora-finetuned"
DATA_PATH = "training_data/training_data.jsonl"


def load_samples(data_path: str, num_samples: int) -> list[dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"共加载 {len(samples)} 条样本")
    return samples[:num_samples]


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    import torch
    import re
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
    import re
    ref_numbers = set(re.findall(r'\d+\.?\d*', reference))
    gen_numbers = set(re.findall(r'\d+\.?\d*', generated))
    if not ref_numbers:
        return {"total": 0, "found": 0, "accuracy": 1.0}
    found = ref_numbers & gen_numbers
    return {
        "total": len(ref_numbers),
        "found": len(found),
        "accuracy": len(found) / len(ref_numbers)
    }


def test_base_model(num_samples: int):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    logger.info("=" * 60)
    logger.info("测试基础模型 (未微调)")
    logger.info("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("基础模型加载完成")

    samples = load_samples(DATA_PATH, num_samples)
    results = []

    for i, sample in enumerate(samples):
        formatted = format_training_sample(sample)
        messages = formatted["messages"][:2]
        reference = formatted["messages"][2]["content"]

        logger.info(f"生成样本 {i+1}/{len(samples)}...")
        start = time.time()
        generated = generate(model, tokenizer, messages)
        elapsed = time.time() - start

        overlap = compute_word_overlap(generated, reference)
        facts = check_facts(generated, reference)

        results.append({
            "index": i,
            "overlap": overlap,
            "fact_accuracy": facts["accuracy"],
            "time": elapsed,
        })

        print(f"\n{'='*60}")
        print(f"样本 {i+1} | 词汇重叠: {overlap:.2%} | 事实准确率: {facts['accuracy']:.2%} | 耗时: {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"【参考输出】(前300字):\n{reference[:300]}...")
        print(f"\n【模型输出】(前300字):\n{generated[:300]}...")

    avg_overlap = sum(r["overlap"] for r in results) / len(results)
    avg_facts = sum(r["fact_accuracy"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)

    print(f"\n{'#'*60}")
    print(f"基础模型平均: 词汇重叠={avg_overlap:.2%} 事实准确率={avg_facts:.2%} 耗时={avg_time:.1f}s")
    print(f"{'#'*60}")

    del model
    torch.cuda.empty_cache()

    return {"avg_overlap": avg_overlap, "avg_facts": avg_facts, "results": results}


def test_lora_model(num_samples: int):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    logger.info("=" * 60)
    logger.info("测试微调模型 (LoRA)")
    logger.info("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    logger.info("微调模型加载完成")

    samples = load_samples(DATA_PATH, num_samples)
    results = []

    for i, sample in enumerate(samples):
        formatted = format_training_sample(sample)
        messages = formatted["messages"][:2]
        reference = formatted["messages"][2]["content"]

        logger.info(f"生成样本 {i+1}/{len(samples)}...")
        start = time.time()
        generated = generate(model, tokenizer, messages)
        elapsed = time.time() - start

        overlap = compute_word_overlap(generated, reference)
        facts = check_facts(generated, reference)

        results.append({
            "index": i,
            "overlap": overlap,
            "fact_accuracy": facts["accuracy"],
            "time": elapsed,
        })

        print(f"\n{'='*60}")
        print(f"样本 {i+1} | 词汇重叠: {overlap:.2%} | 事实准确率: {facts['accuracy']:.2%} | 耗时: {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"【参考输出】(前300字):\n{reference[:300]}...")
        print(f"\n【微调模型输出】(前300字):\n{generated[:300]}...")

    avg_overlap = sum(r["overlap"] for r in results) / len(results)
    avg_facts = sum(r["fact_accuracy"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)

    print(f"\n{'#'*60}")
    print(f"微调模型平均: 词汇重叠={avg_overlap:.2%} 事实准确率={avg_facts:.2%} 耗时={avg_time:.1f}s")
    print(f"{'#'*60}")

    del model
    torch.cuda.empty_cache()

    return {"avg_overlap": avg_overlap, "avg_facts": avg_facts, "results": results}


def main():
    parser = argparse.ArgumentParser(description="云端推理测试")
    parser.add_argument("--num-samples", type=int, default=3, help="测试样本数")
    parser.add_argument("--base-only", action="store_true", help="只测试基础模型")
    parser.add_argument("--lora-only", action="store_true", help="只测试微调模型")
    args = parser.parse_args()

    base_result = None
    lora_result = None

    if not args.lora_only:
        base_result = test_base_model(args.num_samples)

    if not args.base_only:
        lora_result = test_lora_model(args.num_samples)

    if base_result and lora_result:
        print(f"\n{'='*60}")
        print(f"微调前后对比:")
        print(f"{'='*60}")
        print(f"{'指标':<15} {'基础模型':>10} {'微调模型':>10} {'提升':>10}")
        print(f"{'-'*45}")
        print(f"{'词汇重叠率':<13} {base_result['avg_overlap']:>10.2%} {lora_result['avg_overlap']:>10.2%} {(lora_result['avg_overlap']-base_result['avg_overlap'])*100:>+9.1f}%")
        print(f"{'事实准确率':<13} {base_result['avg_facts']:>10.2%} {lora_result['avg_facts']:>10.2%} {(lora_result['avg_facts']-base_result['avg_facts'])*100:>+9.1f}%")

    output_file = "inference_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"base": base_result, "lora": lora_result}, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
