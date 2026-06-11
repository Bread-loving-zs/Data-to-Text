from pydantic import BaseModel
from typing import Optional


class BatchDetail(BaseModel):
    niandu: int
    jdpc: float
    hgpc: float
    bhgpc: float
    bhgl: float
    wilson_ci_95: Optional[str] = None
    ci_length: Optional[float] = None
    bhgl_new: Optional[float] = None


class TrendData(BaseModel):
    last_last_niandu_jdpc: Optional[float] = None
    last_last_niandu_hgpc: Optional[float] = None
    last_last_niandu_bhgpc: Optional[float] = None
    last_last_niandu_bhgl: Optional[float] = None
    last_niandu_jdpc: Optional[float] = None
    last_niandu_hgpc: Optional[float] = None
    last_niandu_bhgpc: Optional[float] = None
    last_niandu_bhgl: Optional[float] = None
    niandu_jdpc: Optional[float] = None
    niandu_hgpc: Optional[float] = None
    niandu_bhgpc: Optional[float] = None
    niandu_bhgl: Optional[float] = None
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    yoy_change: Optional[float] = None
    trend_3y: Optional[str] = None
    chi2_3y: Optional[float] = None
    p_value: Optional[float] = None
    chi2_2y: Optional[float] = None
    p_value_2: Optional[float] = None
    is_continuous_unqualified: Optional[str] = None
    yoy_change_new: Optional[float] = None


class RiskItem(BaseModel):
    sp_s_17: Optional[str] = None
    sp_s_20: Optional[str] = None
    xiangmufenlei: Optional[str] = None
    xiangmumingcheng: Optional[str] = None
    total: Optional[float] = None
    fail: Optional[float] = None
    rate: Optional[float] = None


class YearComparison(BaseModel):
    y1_fail: Optional[float] = None
    y1_total: Optional[float] = None
    y1_rate: Optional[float] = None
    y1_result: Optional[str] = None
    y2_fail: Optional[float] = None
    y2_total: Optional[float] = None
    y2_rate: Optional[float] = None
    y2_result: Optional[str] = None
    y3_fail: Optional[float] = None
    y3_total: Optional[float] = None
    y3_rate: Optional[float] = None
    y3_result: Optional[str] = None
    rate_trend: Optional[str] = None


class SeasonalRisk(BaseModel):
    sp_s_17: Optional[str] = None
    sp_s_20: Optional[str] = None
    xiangmufenlei: Optional[str] = None
    xiangmumingcheng: Optional[str] = None
    q1_result: Optional[str] = None
    q2_result: Optional[str] = None
    q3_result: Optional[str] = None
    q4_result: Optional[str] = None
    mark_quarter: Optional[str] = None


class TrainingSample(BaseModel):
    input: dict
    output: str


class ReportContext(BaseModel):
    year: int
    quarter: Optional[int] = None
    food_category: Optional[str] = None
    food_subcategory: Optional[str] = None
    item_name: Optional[str] = None
    province_summary: Optional[dict] = None
    category_details: Optional[list[dict]] = None
    trend_analysis: Optional[dict] = None
    risk_items: Optional[list[dict]] = None
    seasonal_analysis: Optional[list[dict]] = None
    high_rate_items: Optional[list[dict]] = None
    statistics: Optional[dict] = None

    def validate(self) -> dict:
        warnings: list[str] = []
        valid = True

        if self.year < 2000 or self.year > 2099:
            warnings.append(f"年份 {self.year} 不在合理范围 (2000-2099)")
            valid = False

        if self.quarter is not None and (self.quarter < 1 or self.quarter > 4):
            warnings.append(f"季度 {self.quarter} 不在合理范围 (1-4)")
            valid = False

        if self.province_summary is not None:
            has_key_fields = any(k in self.province_summary for k in ("total_inspections", "fail_rate"))
            if not has_key_fields:
                warnings.append("province_summary 缺少关键字段 (total_inspections 或 fail_rate)")

        if self.category_details is not None and len(self.category_details) == 0:
            warnings.append("category_details 不为 None 但为空列表")

        if self.trend_analysis is not None and len(self.trend_analysis) == 0:
            warnings.append("trend_analysis 不为 None 但为空 dict")

        data_fields = (
            self.province_summary,
            self.category_details,
            self.trend_analysis,
            self.risk_items,
            self.seasonal_analysis,
            self.high_rate_items,
        )
        if all(f is None for f in data_fields):
            warnings.append("所有数据字段均为 None，报告上下文无效")
            valid = False

        return {"valid": valid, "warnings": warnings}