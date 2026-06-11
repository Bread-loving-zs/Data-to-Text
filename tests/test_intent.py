import pytest

from src.agent.intent import IntentRecognizer


def test_rule_based_full():
    rec = IntentRecognizer()
    result = rec._rule_based("帮我生成一份2024年一季度的辣椒监督抽检报告")
    assert result["year"] == 2024, f"期望 year=2024, 实际 {result['year']}"
    assert result["quarter"] == 1, f"期望 quarter=1, 实际 {result['quarter']}"
    assert result["food_category"] == "辣椒", f"期望 food_category=辣椒, 实际 {result['food_category']}"
    assert result["report_type"] == "监督抽检", f"期望 report_type=监督抽检, 实际 {result['report_type']}"


def test_rule_based_year_only():
    rec = IntentRecognizer()
    result = rec._rule_based("2023年抽检情况")
    assert result["year"] == 2023
    assert result["food_category"] is None


def test_rule_based_risk_monitoring():
    rec = IntentRecognizer()
    result = rec._rule_based("生成2024年一季度风险监测报告，重点看蔬菜制品")
    assert result["year"] == 2024
    assert result["quarter"] == 1
    assert result["report_type"] == "风险监测"
    assert result["food_category"] == "蔬菜制品"


def test_rule_based_region():
    rec = IntentRecognizer()
    result = rec._rule_based("四川省的食品安全抽检情况")
    assert result["region"] == "四川"


def test_result_is_complete():
    rec = IntentRecognizer()
    assert not rec._result_is_complete({"year": 2024, "food_category": "辣椒"})
    assert rec._result_is_complete({
        "year": 2024, "food_category": "辣椒", "report_type": "监督抽检"
    })
    assert rec._result_is_complete({
        "year": 2024, "food_category": "辣椒", "report_type": "综合分析"
    }) is False


def test_merge_results():
    rec = IntentRecognizer()
    rule = {"year": 2024, "food_category": "辣椒"}
    llm = {"year": 2024, "food_category": "辣椒", "report_type": "监督抽检", "quarter": 1}
    merged = rec._merge_results(rule, llm)
    assert merged["food_category"] == "辣椒"
    assert merged["report_type"] == "监督抽检"
    assert merged["quarter"] == 1