import re
from typing import Optional

from src.config import setup_logging

logger = setup_logging(__name__)


class FactChecker:
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.warnings: list[dict] = []

    def check(self, generated_text: str, context: dict) -> dict:
        self.warnings = []

        facts_in_text = self._extract_numbers(generated_text)

        facts_in_context = self._extract_numbers_from_context(context)

        self._compare(facts_in_text, facts_in_context)

        total_checks = len(facts_in_text)
        passed = total_checks - len(self.warnings)
        if total_checks == 0:
            accuracy = 0.0
            verdict = "无法校验"
        else:
            accuracy = passed / total_checks
            verdict = "通过" if accuracy >= 0.9 else ("警告" if accuracy >= 0.7 else "不通过")

        return {
            "total_facts": total_checks,
            "passed": passed,
            "warnings": self.warnings,
            "accuracy": round(accuracy, 4),
            "verdict": verdict,
        }

    def _extract_numbers(self, text: str) -> list[dict]:
        results = []
        patterns = [
            (r"(不合格率|检出率|达标率|合格率)[^\d]*?(\d+\.?\d*)\s*%", "percentage"),
            (r"(\d+\.?\d*)\s*%\s*(?:的)?\s*(不合格率|检出率|达标率|合格率)", "percentage_rev"),
            (r"(抽检|检测|监测)[^\d]*?(\d+)\s*(批|批次|个)", "count"),
            (r"(不合格|超标|问题)\s*(\d+)\s*(批|批次)", "fail_count"),
            (r"P\s*[<>=]\s*(\d+\.?\d*)", "p_value"),
            (r"相关系数[^\d]*?(\d+\.?\d*)", "correlation"),
            (r"(上升|下降|持平)\s*(?:趋势|态势)", "trend"),
        ]

        for pattern, fact_type in patterns:
            for match in re.finditer(pattern, text):
                if fact_type == "percentage":
                    context_word = match.group(1)
                    value = float(match.group(2))
                    results.append({
                        "type": "rate",
                        "label": context_word,
                        "value": value,
                        "raw": match.group(0),
                    })
                elif fact_type == "percentage_rev":
                    value = float(match.group(1))
                    context_word = match.group(2)
                    results.append({
                        "type": "rate",
                        "label": context_word,
                        "value": value,
                        "raw": match.group(0),
                    })
                elif fact_type == "count":
                    context_word = match.group(1)
                    value = int(match.group(2))
                    results.append({
                        "type": "count",
                        "label": context_word,
                        "value": value,
                        "raw": match.group(0),
                    })
                elif fact_type == "fail_count":
                    context_word = match.group(1)
                    value = int(match.group(2))
                    results.append({
                        "type": "fail_count",
                        "label": context_word,
                        "value": value,
                        "raw": match.group(0),
                    })
                elif fact_type == "p_value":
                    value = float(match.group(1))
                    results.append({
                        "type": "p_value",
                        "label": "P值",
                        "value": value,
                        "raw": match.group(0),
                    })
                elif fact_type == "trend":
                    results.append({
                        "type": "trend",
                        "label": "趋势",
                        "value": match.group(1),
                        "raw": match.group(0),
                    })

        return results

    def _extract_numbers_from_context(self, context: dict) -> dict[str, float]:
        facts: dict[str, float] = {}
        province = context.get("province_summary") or {}
        category = context.get("category_details") or []
        trend = context.get("trend_analysis") or {}
        stats = context.get("statistics") or {}

        if "total_inspections" in province:
            facts["抽检总批次"] = float(province["total_inspections"])
        if "total_fails" in province:
            facts["不合格总批次"] = float(province["total_fails"])
        if "fail_rate" in province:
            val = float(province["fail_rate"])
            facts["全省不合格率"] = val * 100 if val <= 1 else val

        for item in category[:20]:
            name = item.get("sp_s_20") or item.get("xiangmumingcheng") or ""
            if name and "rate" in item:
                val = float(item["rate"])
                facts[f"{name}_不合格率"] = val * 100 if val <= 1 else val
            if name and "total" in item:
                facts[f"{name}_批次"] = float(item["total"])

        for k, v in trend.items():
            if isinstance(v, (int, float)):
                facts[f"趋势_{k}"] = float(v)

        for table_key, si in stats.items():
            if isinstance(si, dict):
                for sk in ("mean_rate", "median_rate", "max_rate", "min_rate"):
                    if sk in si and si[sk] is not None:
                        facts[f"统计_{table_key}_{sk}"] = float(si[sk]) * 100

        return facts

    def _compare(self, text_facts: list[dict], context_facts: dict[str, float]):
        for tf in text_facts:
            if tf["type"] == "trend":
                continue

            matched = self._find_match(tf, context_facts)
            if matched:
                actual = context_facts[matched]
                if tf["type"] in ("rate", "p_value"):
                    rel_diff = abs(tf["value"] - actual) / max(abs(actual), 1e-9)
                    if rel_diff > self.tolerance:
                        self.warnings.append({
                            "raw": tf["raw"],
                            "generated": tf["value"],
                            "expected": actual,
                            "deviation": round(rel_diff, 4),
                            "severity": "high" if rel_diff > 0.2 else "medium",
                        })
                elif tf["type"] in ("count", "fail_count"):
                    if tf["value"] != int(actual):
                        rel_diff = abs(tf["value"] - actual) / max(abs(actual), 1e-9)
                        self.warnings.append({
                            "raw": tf["raw"],
                            "generated": tf["value"],
                            "expected": int(actual),
                            "deviation": round(rel_diff, 4),
                            "severity": "high" if rel_diff > 0.2 else "medium",
                        })

    def _find_match(self, text_fact: dict, context: dict[str, float]) -> Optional[str]:
        label = text_fact.get("label", "")
        for key in context:
            if label in key or key in label:
                return key
        return None