import pytest

from src.agent.fact_checker import FactChecker


def test_extract_numbers_percentage():
    fc = FactChecker()
    text = "蔬菜不合格率为5.20%，水果不合格率为3.10%"
    facts = fc._extract_numbers(text)
    rates = [f for f in facts if f["type"] == "rate"]
    assert len(rates) >= 2, f"期望至少2个百分比率, 实际: {len(rates)}"


def test_extract_numbers_count():
    fc = FactChecker()
    text = "共抽检1000批次，不合格50批次"
    facts = fc._extract_numbers(text)
    assert len(facts) >= 2, f"期望至少2个数字, 实际: {len(facts)}"


def test_extract_p_value():
    fc = FactChecker()
    text = "P<0.005，差异有统计学意义"
    facts = fc._extract_numbers(text)
    p_values = [f for f in facts if f["type"] == "p_value"]
    assert len(p_values) >= 1, f"期望至少1个P值, 实际: {len(p_values)}"


def test_check_pass():
    fc = FactChecker(tolerance=0.05)
    text = "全省不合格率为5.00%，共抽检1000批次，不合格50批次"
    context = {
        "province_summary": {
            "total_inspections": 1000,
            "total_fails": 50,
            "fail_rate": 0.05,
        }
    }
    result = fc.check(text, context)
    assert result["verdict"] == "通过", f"期望通过, 实际: {result['verdict']}"
    assert len(result["warnings"]) == 0, f"期望无警告, 实际: {result['warnings']}"


def test_check_warning():
    fc = FactChecker(tolerance=0.05)
    text = "全省不合格率为50.00%，共抽检1000批次"
    context = {
        "province_summary": {
            "total_inspections": 1000,
            "fail_rate": 0.05,
        }
    }
    result = fc.check(text, context)
    assert result["verdict"] != "通过", f"期望不通过, 实际: {result['verdict']}"
    assert len(result["warnings"]) > 0, f"期望有警告, 实际: {result['warnings']}"


def test_check_empty_text():
    fc = FactChecker()
    result = fc.check("", {})
    assert result["accuracy"] == 1.0
    assert result["total_facts"] == 0


def test_find_match():
    fc = FactChecker()
    context = {"不合格率": 5.0, "辣椒_不合格率": 3.2}
    assert fc._find_match({"label": "不合格率", "type": "rate"}, context) == "不合格率"
    assert fc._find_match({"label": "辣椒", "type": "rate"}, context) == "辣椒_不合格率"
    assert fc._find_match({"label": "不存在", "type": "rate"}, context) is None