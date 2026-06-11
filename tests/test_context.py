import pytest
import pandas as pd

from src.agent.context import ContextAssembler
from src.data.models import ReportContext


def test_assemble_empty():
    assembler = ContextAssembler()
    intent = {"year": 2024, "food_category": "蔬菜"}
    ctx = assembler.assemble(intent, {})
    assert ctx is not None
    assert ctx.year == 2024
    assert ctx.food_category == "蔬菜"
    assert ctx.province_summary is None


def test_assemble_with_prov_data():
    assembler = ContextAssembler()
    df = pd.DataFrame([{
        "niandu": 2024, "jdpc": 1000, "bhgpc": 50, "bhgl": 0.05,
        "trend_3y": "上升", "p_value": 0.01,
    }])
    ctx = assembler.assemble({"year": 2024}, {"prov_trend": df})
    assert ctx.province_summary is not None
    assert ctx.province_summary["total_inspections"] == 1000
    assert ctx.province_summary["total_fails"] == 50
    assert ctx.province_summary["fail_rate"] == 0.05
    assert ctx.trend_analysis is not None


def test_assemble_with_statistics():
    assembler = ContextAssembler()
    stats = {
        "prov_trend": {
            "mean_rate": 0.052,
            "median_rate": 0.048,
            "max_rate": 0.12,
            "min_rate": 0.01,
            "std_rate": 0.03,
            "total_inspections": 5000,
            "total_fails": 260,
            "overall_rate": 0.052,
            "sample_count": 10,
        }
    }
    ctx = assembler.assemble({"year": 2024}, {"_statistics": stats})
    assert ctx.statistics is not None
    assert ctx.statistics == stats


def test_to_prompt_text_basic():
    ctx = ReportContext(
        year=2024, quarter=1, food_category="辣椒",
        province_summary={"total_inspections": 1000, "total_fails": 45, "fail_rate": 0.045}
    )
    assembler = ContextAssembler()
    text = assembler.to_prompt_text(ctx)
    assert "2024年" in text
    assert "辣椒" in text
    assert "1000" in text
    assert "45" in text
    assert "4.50%" in text


def test_to_prompt_text_with_statistics():
    ctx = ReportContext(
        year=2024,
        statistics={
            "prov_trend": {
                "mean_rate": 0.052,
                "median_rate": 0.048,
                "max_rate": 0.12,
                "min_rate": 0.01,
                "std_rate": 0.03,
                "total_inspections": 5000,
                "total_fails": 260,
                "overall_rate": 0.052,
                "sample_count": 10,
            }
        }
    )
    assembler = ContextAssembler()
    text = assembler.to_prompt_text(ctx)
    assert "统计指标" in text
    assert "5.20%" in text
    assert "4.80%" in text


def test_build_category_details():
    assembler = ContextAssembler()
    df = pd.DataFrame([{
        "sp_s_20": "辣椒", "total": 500, "fail": 25, "rate": 0.05,
        "trend_3y": "上升", "p_value": 0.03,
    }])
    result = assembler._build_category_details({"category_trend": df})
    assert result is not None
    assert len(result) == 1
    assert result[0]["sp_s_20"] == "辣椒"
    assert result[0]["rate"] == 0.05


def test_build_risk_items():
    assembler = ContextAssembler()
    df = pd.DataFrame([{
        "xiangmumingcheng": "农药残留", "sp_s_20": "辣椒",
        "y1_rate": 0.03, "y2_rate": 0.05, "y3_rate": 0.08,
        "rate_trend": "持续上升",
    }])
    result = assembler._build_risk_items({"risk_items": df})
    assert result is not None
    assert len(result) == 1
    assert result[0]["y1_rate"] == 0.03


def test_build_seasonal_analysis():
    assembler = ContextAssembler()
    df = pd.DataFrame([{
        "xiangmumingcheng": "微生物", "q1_result": "高风险",
        "q2_result": "中风险", "q3_result": "高风险", "q4_result": "低风险",
    }])
    result = assembler._build_seasonal_analysis({"seasonal": df})
    assert result is not None
    assert result[0]["q1"] == "高风险"