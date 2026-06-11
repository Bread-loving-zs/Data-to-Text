import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


class StatisticsEngine:
    @staticmethod
    def chi_square_test(observed: list[float], expected: Optional[list[float]] = None) -> dict:
        if expected is None:
            expected = [np.mean(observed)] * len(observed)
        try:
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            return {
                "chi2": round(chi2, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "conclusion": "差异有统计学意义" if p_value < 0.05 else "差异无统计学意义"
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def yoy_change(current: float, previous: float) -> dict:
        if previous == 0:
            return {"yoy_change": None, "yoy_change_pct": None, "direction": "N/A"}
        change = current - previous
        change_pct = (change / previous) * 100
        direction = "上升" if change > 0 else ("下降" if change < 0 else "持平")
        return {
            "yoy_change": round(change, 4),
            "yoy_change_pct": round(change_pct, 2),
            "direction": direction
        }

    @staticmethod
    def trend_analysis(values: list[float], periods: Optional[list[str]] = None) -> dict:
        if len(values) < 2:
            return {"error": "数据点不足，至少需要2个"}
        x = np.arange(len(values))
        y = np.array(values, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return {"error": "有效数据点不足"}
        x_clean = x[mask]
        y_clean = y[mask]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        direction = "上升" if slope > 0 else ("下降" if slope < 0 else "持平")
        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_value ** 2, 4),
            "p_value": round(p_value, 6),
            "direction": direction,
            "significant": p_value < 0.05
        }

    @staticmethod
    def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> dict:
        if total == 0:
            return {"lower": None, "upper": None, "rate": None}
        rate = successes / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denominator = 1 + z**2 / total
        centre = (rate + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * total)) / total) / denominator
        return {
            "rate": round(rate, 6),
            "lower": round(max(0, centre - margin), 6),
            "upper": round(min(1, centre + margin), 6),
            "confidence": confidence
        }

    @staticmethod
    def summarize_rates(df: pd.DataFrame, rate_col: str = "rate",
                        total_col: str = "total", fail_col: str = "fail") -> dict:
        rates = df[rate_col].dropna()
        totals = df[total_col].dropna() if total_col in df.columns else pd.Series(dtype=float)
        fails = df[fail_col].dropna() if fail_col in df.columns else pd.Series(dtype=float)

        return {
            "mean_rate": round(rates.mean(), 6) if len(rates) > 0 else None,
            "max_rate": round(rates.max(), 6) if len(rates) > 0 else None,
            "min_rate": round(rates.min(), 6) if len(rates) > 0 else None,
            "median_rate": round(rates.median(), 6) if len(rates) > 0 else None,
            "std_rate": round(rates.std(), 6) if len(rates) > 0 else None,
            "total_inspections": int(totals.sum()) if len(totals) > 0 else None,
            "total_fails": int(fails.sum()) if len(fails) > 0 else None,
            "overall_rate": round(fails.sum() / totals.sum(), 6) if len(totals) > 0 and totals.sum() > 0 else None,
            "sample_count": len(rates)
        }

    @staticmethod
    def detect_anomalies(values: list[float], method: str = "iqr") -> list[int]:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 4:
            return []
        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            anomalies = []
            for i, v in enumerate(values):
                if not np.isnan(v) and (v < lower or v > upper):
                    anomalies.append(i)
            return anomalies
        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                return []
            anomalies = []
            for i, v in enumerate(values):
                if not np.isnan(v) and abs(v - mean) / std > 2.5:
                    anomalies.append(i)
            return anomalies
        return []

    @staticmethod
    def compare_groups(group_data: dict[str, list[float]]) -> dict:
        groups = {k: [x for x in v if not np.isnan(x)] for k, v in group_data.items()}
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        if len(groups) < 2:
            return {"error": "至少需要2个有效分组"}
        group_names = list(groups.keys())
        group_values = list(groups.values())
        try:
            f_stat, p_value = stats.f_oneway(*group_values)
            return {
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "conclusion": "组间差异有统计学意义" if p_value < 0.05 else "组间差异无统计学意义",
                "groups": {name: {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4),
                                   "n": len(vals)}
                           for name, vals in zip(group_names, group_values)}
            }
        except Exception as e:
            return {"error": str(e)}