import re
import time
from dataclasses import dataclass
from typing import Optional

from src.agent.llm_client import LLMClient
from src.agent.context import ContextAssembler
from src.agent.fact_checker import FactChecker
from src.config import (
    INFERENCE_BACKEND,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    PROMPT_DIR,
    setup_logging,
)
from src.data.models import ReportContext

logger = setup_logging(__name__)


@dataclass
class GenerationMetrics:
    total_generations: int = 0
    success_count: int = 0
    fallback_count: int = 0
    fact_check_pass_count: int = 0
    avg_generation_time_ms: float = 0.0
    fact_check_accuracy: float = 0.0


def _load_system_prompt() -> str:
    prompt_file = PROMPT_DIR / "system_report.txt"
    if prompt_file.exists():
        logger.debug(f"从文件加载 system prompt: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")
    return (
        "你是一位资深的食品安全抽检分析专家。请根据提供的抽检数据，"
        "撰写一份专业的食品安全抽检分析报告。\n\n"
        "## 撰写要求\n"
        "1. 语言专业严谨，使用\"不合格率\"\"检出率\"\"超标倍数\"等标准术语\n"
        "2. 数据呈现清晰，关键数字精确到小数点后两位\n"
        "3. 对趋势变化给出合理分析（上升/下降/持平）\n"
        "4. 对有统计学意义的结果（P<0.05），明确标注\"差异有统计学意义\"\n"
        "5. 对高风险项目给出预警提示\n"
        "6. 报告结构清晰，包含：概述、整体情况、分类分析、趋势分析、风险预警、总结建议\n\n"
        "## 输出格式\n"
        "使用 Markdown 格式，包含标题层级（# ## ###）、表格和必要的列表。"
    )


REPORT_SYSTEM_PROMPT = _load_system_prompt()


class ReportGenerator:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 backend: Optional[str] = None):
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.backend = backend or INFERENCE_BACKEND
        self.context_assembler = ContextAssembler()
        self.fact_checker = FactChecker()
        self.metrics = GenerationMetrics()
        self._llm = LLMClient(model, host, backend)

    def generate(self, intent: dict, query_results: dict) -> str:
        start_time = time.time()
        ctx = self.context_assembler.assemble(intent, query_results)
        context_text = self.context_assembler.to_prompt_text(ctx)
        report = self._call_llm(context_text)

        is_fallback = "降级模式" in report or "本报告基于抽检数据自动生成，涵盖抽检概况" in report
        elapsed_ms = (time.time() - start_time) * 1000

        self.metrics.total_generations += 1
        if is_fallback:
            self.metrics.fallback_count += 1
        else:
            self.metrics.success_count += 1

        fact_result = self.fact_checker.check(report, ctx.model_dump())
        if fact_result["verdict"] == "通过":
            self.metrics.fact_check_pass_count += 1

        prev_success = self.metrics.total_generations - 1
        if prev_success > 0:
            self.metrics.avg_generation_time_ms = (
                (self.metrics.avg_generation_time_ms * prev_success + elapsed_ms)
                / self.metrics.total_generations
            )
        else:
            self.metrics.avg_generation_time_ms = elapsed_ms

        if self.metrics.total_generations > 0:
            self.metrics.fact_check_accuracy = (
                self.metrics.fact_check_pass_count / self.metrics.total_generations
            )

        if fact_result["warnings"]:
            logger.warning(
                f"事实校验: {fact_result['verdict']} "
                f"({fact_result['passed']}/{fact_result['total_facts']}), "
                f"偏差项: {len(fact_result['warnings'])}"
            )
            for w in fact_result["warnings"][:5]:
                logger.warning(
                    f"  [{w['severity']}] '{w['raw']}' -> "
                    f"生成={w['generated']}, 期望={w['expected']}, 偏差={w['deviation']:.2%}"
                )
        else:
            logger.info(f"事实校验: 通过 ({fact_result['total_facts']} 项)")

        return report

    def generate_from_context(self, ctx: ReportContext) -> str:
        context_text = self.context_assembler.to_prompt_text(ctx)
        return self._call_llm(context_text)

    def get_metrics(self) -> GenerationMetrics:
        return self.metrics

    def reset_metrics(self) -> None:
        self.metrics = GenerationMetrics()

    def _call_llm(self, context_text: str) -> str:
        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"请根据以下数据撰写食品安全抽检分析报告：\n\n{context_text}"},
        ]
        content = self._llm.chat(messages, temperature=0.3, max_tokens=4096, timeout=120.0)
        if content is None:
            return self._fallback_report(context_text)
        return content

    def _fallback_report(self, context_text: str) -> str:
        data = {
            "total": None, "fail": None, "rate": None,
            "categories": [],
            "trend_items": [],
            "stat_items": [],
        }

        sections = re.split(r"\n## ", context_text)
        for section in sections:
            section_stripped = section.strip()
            if section_stripped.startswith("全省抽检概况"):
                for line in section_stripped.split("\n"):
                    m_total = re.search(r"抽检总批次[：:]\s*([\d.]+)", line)
                    m_fail = re.search(r"不合格批次[：:]\s*([\d.]+)", line)
                    m_rate = re.search(r"不合格率[：:]\s*(.+?)(?:%|\n|$)", line)
                    if m_total:
                        data["total"] = m_total.group(1)
                    if m_fail:
                        data["fail"] = m_fail.group(1)
                    if m_rate:
                        data["rate"] = m_rate.group(1).strip() + "%"

            elif section_stripped.startswith("分类抽检详情"):
                for line in section_stripped.split("\n"):
                    m = re.match(r"-\s*(.+?)[：:]\s*抽检(\d+)批次[，,]\s*不合格(\d+)批次[，,]\s*不合格率(.+)", line)
                    if m:
                        data["categories"].append({
                            "name": m.group(1).strip(),
                            "total": m.group(2),
                            "fail": m.group(3),
                            "rate": m.group(4).strip(),
                        })

            elif section_stripped.startswith("三年趋势分析"):
                for line in section_stripped.split("\n"):
                    m = re.match(r"-\s*(.+?)[：:]\s*(.+)", line)
                    if m:
                        key = m.group(1).strip()
                        val = m.group(2).strip()
                        if key not in ("slope", "r_squared", "chi2_3y", "chi2_2y"):
                            data["trend_items"].append((key, val))

            elif section_stripped.startswith("统计指标"):
                for line in section_stripped.split("\n"):
                    if line.startswith("### "):
                        continue
                    m = re.match(r"-\s*(.+?)[：:]\s*(.+)", line)
                    if m:
                        data["stat_items"].append((m.group(1).strip(), m.group(2).strip()))

        has_data = (
            data["total"] is not None
            or data["categories"]
            or data["trend_items"]
            or data["stat_items"]
        )

        if not has_data:
            return self._simple_fallback(context_text)

        report_parts = [
            "# 食品安全抽检分析报告\n\n",
            "## 一、概述\n\n",
        ]

        if data["total"] and data["fail"] and data["rate"]:
            report_parts.append(
                f"本报告基于抽检数据自动生成。全省共抽检 {data['total']} 批次，"
                f"不合格 {data['fail']} 批次，不合格率 {data['rate']}。\n"
            )
            report_parts.append("注：本报告为自动降级生成，LLM 服务暂时不可用。\n")
        elif data["total"]:
            report_parts.append(
                f"本报告基于抽检数据自动生成。全省共抽检 {data['total']} 批次。\n"
            )
            report_parts.append("注：本报告为自动降级生成，LLM 服务暂时不可用。\n")
        else:
            report_parts.append(
                "本报告基于抽检数据自动生成。\n"
                "注：本报告为自动降级生成，LLM 服务暂时不可用。\n"
            )

        report_parts.append("\n## 二、整体情况\n\n")

        if data["total"] or data["fail"] or data["rate"]:
            report_parts.append("| 指标 | 数值 |\n")
            report_parts.append("|------|------|\n")
            if data["total"]:
                report_parts.append(f"| 抽检总批次 | {data['total']} |\n")
            if data["fail"]:
                report_parts.append(f"| 不合格批次 | {data['fail']} |\n")
            if data["rate"]:
                report_parts.append(f"| 不合格率 | {data['rate']} |\n")

        if data["trend_items"]:
            report_parts.append("\n### 三年趋势\n\n")
            report_parts.append("| 指标 | 数值 |\n")
            report_parts.append("|------|------|\n")
            for key, val in data["trend_items"]:
                report_parts.append(f"| {key} | {val} |\n")

        if data["categories"]:
            report_parts.append("\n### 分类抽检详情\n\n")
            report_parts.append("| 分类 | 抽检批次 | 不合格批次 | 不合格率 |\n")
            report_parts.append("|------|----------|------------|----------|\n")
            for cat in data["categories"]:
                report_parts.append(
                    f"| {cat['name']} | {cat['total']} | {cat['fail']} | {cat['rate']} |\n"
                )

        if data["stat_items"]:
            report_parts.append("\n### 统计摘要\n\n")
            report_parts.append("| 指标 | 数值 |\n")
            report_parts.append("|------|------|\n")
            for key, val in data["stat_items"]:
                report_parts.append(f"| {key} | {val} |\n")

        report_parts.append("\n## 三、总结与建议\n\n")
        report_parts.append("建议持续关注不合格率较高的品类和检测项目，加强源头监管和过程控制。\n")
        if data["categories"]:
            high_rate = sorted(
                data["categories"], key=lambda x: float(x["rate"].replace("%", "")) if x["rate"].replace("%", "").replace(".", "").isdigit() else 0, reverse=True
            )
            if high_rate:
                top_cat = high_rate[0]
                report_parts.append(
                    f"重点关注品类：{top_cat['name']}（不合格率 {top_cat['rate']}）。\n"
                )

        report_parts.append("\n---\n*本报告由 Data-to-Text 系统自动生成（降级模式）*")
        return "".join(report_parts)

    def _simple_fallback(self, context_text: str) -> str:
        lines = context_text.strip().split("\n")
        report_parts = [
            "# 食品安全抽检分析报告\n",
            "## 一、概述\n",
            "本报告基于抽检数据自动生成，涵盖抽检概况、分类分析和趋势研判。\n",
            "## 二、数据摘要\n",
        ]
        in_section = False
        for line in lines:
            if line.startswith("## "):
                report_parts.append(f"\n## {line[3:].strip()}\n")
                in_section = True
            elif line.startswith("- ") and in_section:
                report_parts.append(f"{line}\n")

        report_parts.append("\n## 三、总结与建议\n")
        report_parts.append("建议持续关注不合格率较高的品类和检测项目，加强源头监管和过程控制。\n")
        report_parts.append("\n---\n*本报告由 Data-to-Text 系统自动生成*")
        return "".join(report_parts)
