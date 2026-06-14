from typing import Callable, Optional
import pandas as pd

from src.data.loader import DataLoader
from src.config import FOOD_CATEGORY_COLS, setup_logging

logger = setup_logging(__name__)


class DataQuerier:
    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()

    def _load_data(self, key: str, load_fn: Callable, *args, **kwargs) -> Optional[pd.DataFrame]:
        try:
            df = load_fn(*args, **kwargs)
            return df
        except Exception as e:
            logger.warning(f"加载 {key} 失败: {e}")
            return None

    def query_by_intent(self, intent: dict) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        status: dict[str, str] = {}
        year = intent.get("year")
        food_cat = intent.get("food_category")
        food_sub = intent.get("food_subcategory")
        item = intent.get("item_name")
        region = intent.get("region")

        prov_trend = self._load_data("prov_trend", self.loader.load_prov_trend)
        if prov_trend is not None:
            results["prov_trend"] = prov_trend
            status["prov_trend"] = "success"
        else:
            status["prov_trend"] = "error"

        prov_detail = self._load_data("prov_batch_detail", self.loader.load_prov_batch_detail)
        if prov_detail is not None:
            if year:
                prov_detail = prov_detail[prov_detail["niandu"].astype(int) == int(year)]
            results["prov_batch_detail"] = prov_detail
            status["prov_batch_detail"] = "success"
        else:
            status["prov_batch_detail"] = "error"

        risk = self._load_data("risk_items", self.loader.load_risk_items, "all")
        if risk is not None:
            results["risk_items"] = self._filter_by_food(risk, food_cat, food_sub, item)
            status["risk_items"] = "success"
        else:
            status["risk_items"] = "error"

        scjc = self._load_data("market_inspection", self.loader.load_market_inspection, "all")
        if scjc is not None:
            scjc = self._filter_by_food(scjc, food_cat, food_sub, item)
            if year and "niandu" in scjc.columns:
                scjc = scjc[scjc["niandu"].astype(int) == int(year)]
            results["market_inspection"] = scjc
            status["market_inspection"] = "success"
        else:
            status["market_inspection"] = "error"

        high_rate = self._load_data("high_rate", self.loader.load_high_rate)
        if high_rate is not None:
            results["high_rate"] = self._filter_by_food(high_rate, food_cat, food_sub, item)
            status["high_rate"] = "success"
        else:
            status["high_rate"] = "error"

        exceedance = self._load_data("exceedance", self.loader.load_exceedance)
        if exceedance is not None:
            exceedance = self._filter_by_food(exceedance, food_cat, food_sub, item)
            if year and "niandu" in exceedance.columns:
                exceedance = exceedance[exceedance["niandu"].astype(int) == int(year)]
            results["exceedance"] = exceedance
            status["exceedance"] = "success"
        else:
            status["exceedance"] = "error"

        near_limit = self._load_data("near_limit", self.loader.load_near_limit)
        if near_limit is not None:
            near_limit = self._filter_by_food(near_limit, food_cat, food_sub, item)
            if year and "niandu" in near_limit.columns:
                near_limit = near_limit[near_limit["niandu"].astype(int) == int(year)]
            results["near_limit"] = near_limit
            status["near_limit"] = "success"
        else:
            status["near_limit"] = "error"

        dl_trend = self._load_data("category_trend", self.loader.load_category_trend, "sp_s_17")
        if dl_trend is not None:
            results["category_trend"] = self._filter_by_food(dl_trend, food_cat, food_sub, item)
            status["category_trend"] = "success"
        else:
            status["category_trend"] = "error"

        xm_trend = self._load_data("item_trend", self.loader.load_xm_trend)
        if xm_trend is not None:
            if item:
                xm_trend = xm_trend[xm_trend["xiangmumingcheng"].astype(str).str.contains(item, na=False, regex=False)]
            results["item_trend"] = xm_trend
            status["item_trend"] = "success"
        else:
            status["item_trend"] = "error"

        seasonal = self._load_data("seasonal", self.loader.load_seasonal, "jjxfx_report")
        if seasonal is not None:
            if food_sub:
                seasonal = seasonal[seasonal["sp_s_20"].astype(str).str.contains(food_sub, na=False, regex=False)]
            elif food_cat:
                if "sp_s_17" in seasonal.columns:
                    seasonal = seasonal[seasonal["sp_s_17"].astype(str).str.contains(food_cat, na=False, regex=False)]
            results["seasonal"] = seasonal
            status["seasonal"] = "success"
        else:
            status["seasonal"] = "error"

        success_count = sum(1 for v in status.values() if v == "success")
        total_count = len(status)
        logger.info(f"数据查询完成: {success_count}/{total_count} 表成功")
        if success_count == 0:
            logger.error(f"所有数据表查询均失败! 失败详情: {status}")

        results["_query_status"] = status

        return results

    def _filter_by_food(self, df: pd.DataFrame, food_cat: Optional[str],
                        food_sub: Optional[str], item: Optional[str]) -> pd.DataFrame:
        result = df.copy()
        if food_cat:
            for col in reversed(FOOD_CATEGORY_COLS):
                if col in result.columns:
                    filtered = result[result[col].astype(str).str.contains(food_cat, na=False, regex=False)]
                    if not filtered.empty:
                        result = filtered
                        break
        if food_sub:
            if "sp_s_20" in result.columns:
                result = result[result["sp_s_20"].astype(str).str.contains(food_sub, na=False, regex=False)]
        if item:
            if "xiangmumingcheng" in result.columns:
                result = result[result["xiangmumingcheng"].astype(str).str.contains(item, na=False, regex=False)]
        return result
