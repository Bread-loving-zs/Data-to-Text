from pathlib import Path
from typing import Optional
import pandas as pd

from src.data.loader import DataLoader
from src.report.charts import ChartGenerator
from src.report.converter import MarkdownToDocx
from src.config import OUTPUT_DIR, setup_logging

logger = setup_logging(__name__)


class ReportTemplate:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_gen = ChartGenerator(self.output_dir)

    def build_report_with_charts(self, report_markdown: str, query_results: dict,
                                 report_name: str = "抽检分析报告") -> Path:
        chart_paths = self._generate_charts_from_data(query_results)
        enhanced_md = self._insert_charts_into_markdown(report_markdown, chart_paths)
        converter = MarkdownToDocx()
        converter.convert(enhanced_md)

        output_path = self.output_dir / f"{report_name}.docx"
        converter.save(output_path)
        return output_path

    def _generate_charts_from_data(self, query_results: dict) -> dict:
        chart_paths = {}

        scjc = query_results.get("market_inspection")
        if scjc is not None and not scjc.empty and "rate" in scjc.columns:
            df = scjc.dropna(subset=["rate"]).sort_values("rate", ascending=False).head(10)
            labels = []
            values = []
            for _, row in df.iterrows():
                label = str(row.get("sp_s_20", row.get("xiangmumingcheng", "未命名")))
                if len(label) > 8:
                    label = label[:7] + "..."
                labels.append(label)
                values.append(float(row["rate"]) * 100)
            if labels:
                chart_paths["rate_bar"] = self.chart_gen.bar_chart(
                    labels, values, title="不合格率排名 (Top 10)", filename="rate_top10.png"
                )

        risk = query_results.get("risk_items")
        if risk is not None and not risk.empty:
            if "y3_rate" in risk.columns:
                df_risk = risk.dropna(subset=["y3_rate"]).head(8)
                labels = []
                values = []
                for _, row in df_risk.iterrows():
                    label = str(row.get("xiangmumingcheng", row.get("sp_s_20", "未命名")))
                    if len(label) > 8:
                        label = label[:7] + "..."
                    labels.append(label)
                    values.append(float(row["y3_rate"]) * 100 if pd.notna(row.get("y3_rate")) else 0)
                if labels:
                    chart_paths["risk_bar"] = self.chart_gen.bar_chart(
                        labels, values, title="风险项目不合格率", filename="risk_items.png",
                        color="#E67E22"
                    )

        trend = query_results.get("prov_trend")
        if trend is not None and not trend.empty:
            row = trend.iloc[0]
            periods = ["前年", "去年", "今年"]
            vals = []
            for prefix in ["last_last_niandu", "last_niandu", "niandu"]:
                col = f"{prefix}_bhgl"
                if col in trend.columns and pd.notna(row.get(col)):
                    vals.append(float(row[col]) * 100)
                else:
                    vals.append(0)
            if vals and any(v > 0 for v in vals):
                chart_paths["trend_line"] = self.chart_gen.trend_line_chart(
                    periods, vals, title="三年不合格率趋势", filename="trend_3y.png"
                )

        seasonal = query_results.get("seasonal")
        if seasonal is not None and not seasonal.empty and "mark_quarter" in seasonal.columns:
            marked = seasonal[seasonal["mark_quarter"].notna() & (seasonal["mark_quarter"] != "")]
            if not marked.empty:
                quarters = marked["mark_quarter"].value_counts()
                if len(quarters) > 0:
                    chart_paths["seasonal_pie"] = self.chart_gen.pie_chart(
                        list(quarters.index), list(quarters.values),
                        title="高风险季度分布", filename="seasonal_dist.png"
                    )

        return chart_paths

    def _insert_charts_into_markdown(self, markdown: str, chart_paths: dict) -> str:
        chart_md = "\n\n## 统计图表\n\n"
        chart_descriptions = {
            "rate_bar": "图1：不合格率排名",
            "risk_bar": "图2：风险项目不合格率",
            "trend_line": "图3：三年不合格率趋势",
            "seasonal_pie": "图4：高风险季度分布",
        }
        for key, filepath in chart_paths.items():
            desc = chart_descriptions.get(key, key)
            chart_md += f"![{desc}]({filepath})\n\n"

        if "## 统计图表" in markdown:
            idx = markdown.index("## 统计图表")
            return markdown[:idx] + chart_md

        return markdown + chart_md