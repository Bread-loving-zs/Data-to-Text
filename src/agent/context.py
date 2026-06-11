import json
from typing import Optional
import numpy as np
import pandas as pd

from src.agent.statistics import StatisticsEngine
from src.data.models import ReportContext
from src.config import setup_logging

logger = setup_logging(__name__)


class ContextAssembler:
    def __init__(self, stats_engine: Optional[StatisticsEngine] = None):
        self.stats = stats_engine or StatisticsEngine()

    def assemble(self, intent: dict, query_results: dict[str, pd.DataFrame]) -> ReportContext:
        ctx = ReportContext(
            year=intent.get("year") or 2024,
            quarter=intent.get("quarter"),
            food_category=intent.get("food_category"),
            food_subcategory=intent.get("food_subcategory"),
            item_name=intent.get("item_name"),
        )

        province_data = self._build_province_summary(query_results, ctx.year)
        ctx.province_summary = province_data

        category_data = self._build_category_details(query_results)
        ctx.category_details = category_data

        trend_data = self._build_trend_analysis(query_results)
        ctx.trend_analysis = trend_data

        risk_data = self._build_risk_items(query_results)
        ctx.risk_items = risk_data

        seasonal_data = self._build_seasonal_analysis(query_results)
        ctx.seasonal_analysis = seasonal_data

        high_rate_data = self._build_high_rate_items(query_results)
        ctx.high_rate_items = high_rate_data

        stats = query_results.get("_statistics")
        if stats:
            ctx.statistics = stats
            logger.debug(f"统计结果已集成: {list(stats.keys())}")

        query_status = query_results.get("_query_status")
        if query_status:
            failed_tables = {k: v for k, v in query_status.items() if v != "success"}
            if failed_tables:
                logger.warning(f"数据查询存在失败的表: {list(failed_tables.keys())}")

        validation = ctx.validate()
        if not validation["valid"] or validation["warnings"]:
            logger.warning(f"ReportContext 验证问题: {validation['warnings']}")

        if ctx.statistics is None:
            ctx.statistics = {}
        ctx.statistics["_meta"] = {
            "validation": validation,
            "query_status": query_status,
        }

        return ctx

    def _build_province_summary(self, results: dict[str, pd.DataFrame], year: int) -> Optional[dict]:
        df = results.get("prov_batch_detail")
        if df is None or df.empty:
            df = results.get("prov_trend")
        if df is None or df.empty:
            return None

        summary = {}
        if "jdpc" in df.columns:
            summary["total_inspections"] = int(df["jdpc"].sum())
        if "bhgpc" in df.columns:
            summary["total_fails"] = int(df["bhgpc"].sum())
        if "bhgl" in df.columns and len(df) > 0:
            rate_vals = df["bhgl"].dropna()
            summary["fail_rate"] = round(float(rate_vals.iloc[0]), 4) if len(rate_vals) > 0 else None
        if "trend_3y" in df.columns and len(df) > 0:
            summary["three_year_trend"] = str(df["trend_3y"].iloc[0]) if pd.notna(df["trend_3y"].iloc[0]) else None
        if "yoy_change_new" in df.columns and len(df) > 0:
            val = df["yoy_change_new"].iloc[0]
            if pd.notna(val):
                summary["yoy_change"] = round(float(val), 4)
        if "p_value" in df.columns and len(df) > 0:
            val = df["p_value"].iloc[0]
            if pd.notna(val):
                summary["p_value"] = float(val)
        summary["year"] = year
        return summary

    def _build_category_details(self, results: dict[str, pd.DataFrame]) -> Optional[list[dict]]:
        df = results.get("category_trend")
        if df is None or df.empty:
            df = results.get("market_inspection")
        if df is None or df.empty:
            return None

        details = []
        for _, row in df.head(20).iterrows():
            item = {}
            for col in ["sp_s_17", "sp_s_20", "xiangmumingcheng", "xiangmufenlei"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item[col] = str(row[col])
            for col in ["niandu_jdpc", "total", "jdpc"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item["total"] = int(float(row[col]))
                    break
            for col in ["niandu_bhgpc", "fail", "bhgpc"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item["fail"] = int(float(row[col]))
                    break
            for col in ["niandu_bhgl", "rate", "bhgl"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item["rate"] = round(float(row[col]), 4)
                    break
            if "trend_3y" in df.columns and pd.notna(row.get("trend_3y")):
                item["trend"] = str(row["trend_3y"])
            if "p_value" in df.columns and pd.notna(row.get("p_value")):
                item["p_value"] = float(row["p_value"])
            if item:
                details.append(item)
        return details if details else None

    def _build_trend_analysis(self, results: dict[str, pd.DataFrame]) -> Optional[dict]:
        df = results.get("prov_trend")
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        analysis = {}
        year_cols = [
            ("last_last_niandu", "前年"),
            ("last_niandu", "去年"),
            ("niandu", "今年"),
        ]
        for prefix, label in year_cols:
            jdpc_col = f"{prefix}_jdpc"
            bhgpc_col = f"{prefix}_bhgpc"
            bhgl_col = f"{prefix}_bhgl"
            if jdpc_col in df.columns and pd.notna(row.get(jdpc_col)):
                analysis[f"{label}_批次"] = int(float(row[jdpc_col]))
            if bhgl_col in df.columns and pd.notna(row.get(bhgl_col)):
                analysis[f"{label}_不合格率"] = round(float(row[bhgl_col]), 4)

        for col in ["trend_3y", "slope", "r_squared", "p_value", "chi2_3y", "is_continuous_unqualified"]:
            if col in df.columns and pd.notna(row.get(col)):
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    analysis[col] = round(float(val), 6)
                else:
                    analysis[col] = str(val)

        return analysis if analysis else None

    def _build_risk_items(self, results: dict[str, pd.DataFrame]) -> Optional[list[dict]]:
        df = results.get("risk_items")
        if df is None or df.empty:
            return None
        items = []
        for _, row in df.head(15).iterrows():
            item = {}
            for col in ["sp_s_17", "sp_s_20", "xiangmufenlei", "xiangmumingcheng"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item[col] = str(row[col])
            for year_prefix, label in [("y1", "y1"), ("y2", "y2"), ("y3", "y3")]:
                rate_col = f"{year_prefix}_rate"
                result_col = f"{year_prefix}_result"
                if result_col in df.columns and pd.notna(row.get(result_col)):
                    item[f"{label}_result"] = str(row[result_col])
                elif rate_col in df.columns and pd.notna(row.get(rate_col)):
                    item[f"{label}_rate"] = round(float(row[rate_col]), 4)
            if "rate_trend" in df.columns and pd.notna(row.get("rate_trend")):
                item["trend"] = str(row["rate_trend"])
            items.append(item)
        return items if items else None

    def _build_seasonal_analysis(self, results: dict[str, pd.DataFrame]) -> Optional[list[dict]]:
        df = results.get("seasonal")
        if df is None or df.empty:
            return None
        items = []
        for _, row in df.head(20).iterrows():
            item = {}
            for col in ["sp_s_20", "xiangmumingcheng", "mark_quarter"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item[col] = str(row[col])
            for q in ["q1", "q2", "q3", "q4"]:
                q_col = f"{q}_result"
                if q_col in df.columns and pd.notna(row.get(q_col)):
                    item[q] = str(row[q_col])
            items.append(item)
        return items if items else None

    def _build_high_rate_items(self, results: dict[str, pd.DataFrame]) -> Optional[list[dict]]:
        df = results.get("high_rate")
        if df is None or df.empty:
            return None
        items = []
        for _, row in df.head(15).iterrows():
            item = {}
            for col in ["sp_s_20", "xiangmufenlei", "xiangmumingcheng"]:
                if col in df.columns and pd.notna(row.get(col)):
                    item[col] = str(row[col])
            for col in ["total", "fail", "rate"]:
                if col in df.columns and pd.notna(row.get(col)):
                    val = row[col]
                    item[col] = round(float(val), 4) if col == "rate" else int(float(val))
            items.append(item)
        return items if items else None

    def to_prompt_text(self, ctx: ReportContext) -> str:
        parts = []

        parts.append(f"## 报告基本信息")
        parts.append(f"- 年份：{ctx.year}年")
        if ctx.quarter:
            parts.append(f"- 季度：第{ctx.quarter}季度")
        if ctx.food_category:
            parts.append(f"- 食品大类：{ctx.food_category}")
        if ctx.food_subcategory:
            parts.append(f"- 食品细类：{ctx.food_subcategory}")
        if ctx.item_name:
            parts.append(f"- 检测项目：{ctx.item_name}")

        if ctx.province_summary:
            ps = ctx.province_summary
            parts.append(f"\n## 全省抽检概况")
            if ps.get("total_inspections"):
                parts.append(f"- 抽检总批次：{ps['total_inspections']}")
            if ps.get("total_fails") is not None:
                parts.append(f"- 不合格批次：{ps['total_fails']}")
            if ps.get("fail_rate") is not None:
                parts.append(f"- 不合格率：{ps['fail_rate'] * 100:.2f}%")
            if ps.get("three_year_trend"):
                parts.append(f"- 三年趋势：{ps['three_year_trend']}")
            if ps.get("yoy_change") is not None:
                parts.append(f"- 同比变化：{ps['yoy_change']:.4f}")
            if ps.get("p_value") is not None:
                significance = "P<0.05，差异有统计学意义" if ps["p_value"] < 0.05 else f"P={ps['p_value']:.4f}，差异无统计学意义"
                parts.append(f"- 统计检验：{significance}")

        if ctx.trend_analysis:
            ta = ctx.trend_analysis
            parts.append(f"\n## 三年趋势分析")
            for k, v in ta.items():
                parts.append(f"- {k}：{v}")

        if ctx.category_details:
            parts.append(f"\n## 分类抽检详情")
            for item in ctx.category_details[:10]:
                name = item.get("sp_s_20") or item.get("sp_s_17") or item.get("xiangmumingcheng") or "未命名"
                total = item.get("total", "N/A")
                fail = item.get("fail", "N/A")
                rate = item.get("rate", "N/A")
                if isinstance(rate, float):
                    rate = f"{rate * 100:.2f}%"
                parts.append(f"- {name}：抽检{total}批次，不合格{fail}批次，不合格率{rate}")

        if ctx.risk_items:
            parts.append(f"\n## 风险项目")
            for item in ctx.risk_items[:10]:
                name = item.get("xiangmumingcheng") or item.get("sp_s_20") or "未命名"
                trend = item.get("trend", "")
                parts.append(f"- {name} 趋势：{trend}")

        if ctx.seasonal_analysis:
            parts.append(f"\n## 季节性风险分析")
            for item in ctx.seasonal_analysis[:10]:
                name = item.get("xiangmumingcheng") or item.get("sp_s_20") or "未命名"
                qs = f"Q1:{item.get('q1','N/A')} Q2:{item.get('q2','N/A')} Q3:{item.get('q3','N/A')} Q4:{item.get('q4','N/A')}"
                parts.append(f"- {name}：{qs}")

        if ctx.statistics:
            parts.append(f"\n## 统计指标")
            for table_key, stats_info in ctx.statistics.items():
                if table_key == "_meta":
                    continue
                if isinstance(stats_info, dict):
                    parts.append(f"\n### {table_key}")
                    if "mean_rate" in stats_info and stats_info["mean_rate"] is not None:
                        parts.append(f"- 平均不合格率：{stats_info['mean_rate'] * 100:.2f}%")
                    if "median_rate" in stats_info and stats_info["median_rate"] is not None:
                        parts.append(f"- 中位数不合格率：{stats_info['median_rate'] * 100:.2f}%")
                    if "max_rate" in stats_info and stats_info["max_rate"] is not None:
                        parts.append(f"- 最高不合格率：{stats_info['max_rate'] * 100:.2f}%")
                    if "min_rate" in stats_info and stats_info["min_rate"] is not None:
                        parts.append(f"- 最低不合格率：{stats_info['min_rate'] * 100:.2f}%")
                    if "std_rate" in stats_info and stats_info["std_rate"] is not None:
                        parts.append(f"- 不合格率标准差：{stats_info['std_rate']:.4f}")
                    if "total_inspections" in stats_info and stats_info["total_inspections"] is not None:
                        parts.append(f"- 抽检总批次：{stats_info['total_inspections']}")
                    if "total_fails" in stats_info and stats_info["total_fails"] is not None:
                        parts.append(f"- 不合格总批次：{stats_info['total_fails']}")
                    if "overall_rate" in stats_info and stats_info["overall_rate"] is not None:
                        parts.append(f"- 总体不合格率：{stats_info['overall_rate'] * 100:.2f}%")
                    if "sample_count" in stats_info:
                        parts.append(f"- 样本数：{stats_info['sample_count']}")

        return "\n".join(parts)

    def to_json(self, ctx: ReportContext) -> str:
        return json.dumps(ctx.model_dump(), ensure_ascii=False, indent=2, default=str)