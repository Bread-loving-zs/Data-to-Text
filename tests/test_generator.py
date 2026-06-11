import pytest

from src.agent.generator import ReportGenerator
from src.agent.context import ContextAssembler
from src.data.models import ReportContext


def test_init():
    gen = ReportGenerator()
    assert gen.model is not None
    assert gen.host is not None
    assert gen.backend is not None
    assert gen.context_assembler is not None
    assert gen.fact_checker is not None


def test_fallback_report():
    gen = ReportGenerator()
    context_text = (
        "## 报告基本信息\n"
        "- 年份：2024年\n"
        "- 食品大类：蔬菜制品\n"
        "\n## 全省抽检概况\n"
        "- 抽检总批次：1000\n"
        "- 不合格批次：50\n"
        "- 不合格率：5.00%\n"
    )
    report = gen._fallback_report(context_text)
    assert report is not None
    assert isinstance(report, str)
    assert len(report) > 0
    assert "食品安全抽检分析报告" in report
    assert "概述" in report
    assert "总结与建议" in report


def test_context_assembler_integration():
    gen = ReportGenerator()
    assembler = gen.context_assembler
    assert isinstance(assembler, ContextAssembler)
    ctx = ReportContext(year=2024, food_category="蔬菜")
    text = assembler.to_prompt_text(ctx)
    assert "2024年" in text
    assert "蔬菜" in text