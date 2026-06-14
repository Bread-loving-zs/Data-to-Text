import json
import re
from pathlib import Path
from typing import Optional
from collections import Counter
from dataclasses import dataclass, field

from src.config import setup_logging
from src.data.utils import load_jsonl

logger = setup_logging(__name__)


@dataclass
class EvalResult:
    sample_index: int
    generated_text: str
    reference_text: str
    word_overlap_ratio: float
    num_facts_correct: int
    num_facts_total: int
    fact_accuracy: float
    has_overview: bool
    has_trend: bool
    has_risk_warning: bool
    has_suggestion: bool
    structure_score: float
    overall_score: float
    errors: list[str] = field(default_factory=list)


class Evaluator:
    def __init__(self, test_data_path: Optional[Path] = None):
        self.test_samples = []
        if test_data_path and test_data_path.exists():
            self.load_test_samples(test_data_path)

    def load_test_samples(self, filepath: Path) -> list[dict]:
        samples = load_jsonl(filepath)
        self.test_samples = samples
        logger.info(f"加载 {len(samples)} 条测试样本")
        return samples

    def evaluate_single(self, generated_text: str, sample: dict, index: int = 0) -> EvalResult:
        reference = sample.get("output", "")
        input_data = sample.get("input", {})

        word_overlap = self._compute_word_overlap(generated_text, reference)

        fact_correct, fact_total = self._check_factual_accuracy(generated_text, input_data)
        fact_accuracy = fact_correct / fact_total if fact_total > 0 else 0.0

        has_overview = self._has_section(generated_text, ["概述", "概况", "整体情况"])
        has_trend = self._has_section(generated_text, ["趋势", "变化", "走向"])
        has_risk_warning = self._has_section(generated_text, ["风险", "预警", "不合格率较高", "问题"])
        has_suggestion = self._has_section(generated_text, ["建议", "措施", "对策", "总结建议"])

        structure_hits = sum([has_overview, has_trend, has_risk_warning, has_suggestion])
        structure_score = structure_hits / 4.0

        overall = (
            0.25 * min(word_overlap * 3, 1.0)
            + 0.40 * fact_accuracy
            + 0.35 * structure_score
        )

        return EvalResult(
            sample_index=index,
            generated_text=generated_text,
            reference_text=reference,
            word_overlap_ratio=round(word_overlap, 4),
            num_facts_correct=fact_correct,
            num_facts_total=fact_total,
            fact_accuracy=round(fact_accuracy, 4),
            has_overview=has_overview,
            has_trend=has_trend,
            has_risk_warning=has_risk_warning,
            has_suggestion=has_suggestion,
            structure_score=round(structure_score, 4),
            overall_score=round(overall, 4),
        )

    def evaluate_batch(self, generator, samples: Optional[list[dict]] = None) -> list[EvalResult]:
        samples = samples or self.test_samples
        if not samples:
            logger.warning("无测试样本可评估")
            return []

        results = []
        for i, sample in enumerate(samples):
            intent = sample.get("intent", {})
            query_results = sample.get("query_results", {})
            try:
                generated = generator.generate(intent, query_results)
            except Exception as e:
                logger.error(f"样本 {i} 生成失败: {e}")
                generated = f"[生成失败: {e}]"

            result = self.evaluate_single(generated, sample, index=i)
            results.append(result)

        return results

    def compare_models(self, before_results: list[EvalResult],
                       after_results: list[EvalResult]) -> dict:
        if len(before_results) != len(after_results):
            logger.warning(f"评估结果数量不一致: {len(before_results)} vs {len(after_results)}")

        def avg(items, attr):
            vals = [getattr(r, attr, 0) for r in items]
            return sum(vals) / len(vals) if vals else 0

        return {
            "num_samples": len(before_results),
            "before": {
                "word_overlap": round(avg(before_results, "word_overlap_ratio"), 4),
                "fact_accuracy": round(avg(before_results, "fact_accuracy"), 4),
                "structure_score": round(avg(before_results, "structure_score"), 4),
                "overall_score": round(avg(before_results, "overall_score"), 4),
            },
            "after": {
                "word_overlap": round(avg(after_results, "word_overlap_ratio"), 4),
                "fact_accuracy": round(avg(after_results, "fact_accuracy"), 4),
                "structure_score": round(avg(after_results, "structure_score"), 4),
                "overall_score": round(avg(after_results, "overall_score"), 4),
            },
            "improvement": {
                "word_overlap": round(avg(after_results, "word_overlap_ratio") - avg(before_results, "word_overlap_ratio"), 4),
                "fact_accuracy": round(avg(after_results, "fact_accuracy") - avg(before_results, "fact_accuracy"), 4),
                "structure_score": round(avg(after_results, "structure_score") - avg(before_results, "structure_score"), 4),
                "overall_score": round(avg(after_results, "overall_score") - avg(before_results, "overall_score"), 4),
            }
        }

    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+\.?\d*", text1))
        words2 = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+\.?\d*", text2))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        return len(intersection) / len(words1 | words2)

    def _check_factual_accuracy(self, generated: str, input_data: dict) -> tuple[int, int]:
        input_values = self._extract_all_values(input_data)
        if not input_values:
            return 0, 0

        correct = 0
        total = 0
        tolerance = 0.001

        for key, val in input_values.items():
            if isinstance(val, float):
                found = False
                patterns = [
                    rf"{re.escape(key)}[^\d]*{val:.2f}",
                    rf"{re.escape(key)}[^\d]*{val:.4f}",
                    rf"{re.escape(key)}[^\d]*{val * 100:.2f}%",
                ]
                for pattern in patterns:
                    if re.search(pattern, generated):
                        found = True
                        break
                if found:
                    correct += 1
                total += 1

            elif isinstance(val, str) and len(val) > 1:
                if val in generated:
                    correct += 1
                total += 1

        return correct, total

    def _extract_all_values(self, data: dict, prefix: str = "") -> dict[str, float | str]:
        values = {}
        for k, v in data.items():
            if k == "query":
                continue
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[full_key] = float(v)
            elif isinstance(v, str) and v:
                values[full_key] = v
            elif isinstance(v, dict):
                values.update(self._extract_all_values(v, full_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        values.update(self._extract_all_values(item, f"{full_key}[{i}]"))
        return values

    def _has_section(self, text: str, keywords: list[str]) -> bool:
        return any(kw in text for kw in keywords)


def run_evaluation_pipeline(generator, test_data_path: str, output_path: Optional[str] = None):
    evaluator = Evaluator(Path(test_data_path))

    if not evaluator.test_samples:
        logger.error(f"测试数据为空: {test_data_path}")
        return

    logger.info(f"开始评估 {len(evaluator.test_samples)} 条测试样本 ...")
    results = evaluator.evaluate_batch(generator)

    avg_overall = sum(r.overall_score for r in results) / len(results)
    avg_fact = sum(r.fact_accuracy for r in results) / len(results)
    avg_struct = sum(r.structure_score for r in results) / len(results)

    logger.info(f"评估完成: overall={avg_overall:.4f}, fact_accuracy={avg_fact:.4f}, structure={avg_struct:.4f}")

    if output_path:
        output_data = {
            "summary": {
                "num_samples": len(results),
                "avg_overall_score": round(avg_overall, 4),
                "avg_fact_accuracy": round(avg_fact, 4),
                "avg_structure_score": round(avg_struct, 4),
            },
            "details": [
                {
                    "index": r.sample_index,
                    "overall_score": r.overall_score,
                    "fact_accuracy": r.fact_accuracy,
                    "structure_score": r.structure_score,
                    "word_overlap": r.word_overlap_ratio,
                }
                for r in results
            ]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存: {output_path}")

    return results