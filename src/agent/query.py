from typing import Optional
import pandas as pd

from src.data.loader import DataLoader
from src.config import FOOD_CATEGORY_COLS, setup_logging

logger = setup_logging(__name__)


class DataQuerier:
    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()

    def query_by_intent(self, intent: dict) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        status: dict[str, str] = {}
        year = intent.get("year")
        food_cat = intent.get("food_category")
        food_sub = intent.get("food_subcategory")
        item = intent.get("item_name")
        region = intent.get("region")

        try:
            prov_trend = self.loader.load_prov_trend()
            results["prov_trend"] = prov_trend
            status["prov_trend"] = "success"
        except Exception as e:
            logger.warning(f"加载 prov_trend 失败: {e}")
            status["prov_trend"] = f"error: {e}"

        try:
            prov_detail = self.loader.load_prov_batch_detail()
            if year:
                prov_detail = prov_detail[prov_detail["niandu"].astype(int) == int(year)]
            results["prov_batch_detail"] = prov_detail
            status["prov_batch_detail"] = "success"
        except Exception as e:
            logger.warning(f"加载 prov_batch_detail 失败: {e}")
            status["prov_batch_detail"] = f"error: {e}"

        try:
            wgx = self._filter_by_food(self.loader.load_risk_items("all"), food_cat, food_sub, item)
            results["risk_items"] = wgx
            status["risk_items"] = "success"
        except Exception as e:
            logger.warning(f"加载 risk_items 失败: {e}")
            status["risk_items"] = f"error: {e}"

        try:
            scjc = self._filter_by_food(self.loader.load_market_inspection("all"), food_cat, food_sub, item)
            if year and "niandu" in scjc.columns:
                scjc = scjc[scjc["niandu"].astype(int) == int(year)]
            results["market_inspection"] = scjc
            status["market_inspection"] = "success"
        except Exception as e:
            logger.warning(f"加载 market_inspection 失败: {e}")
            status["market_inspection"] = f"error: {e}"

        try:
            high_rate = self._filter_by_food(self.loader.load_high_rate(), food_cat, food_sub, item)
            results["high_rate"] = high_rate
            status["high_rate"] = "success"
        except Exception as e:
            logger.warning(f"加载 high_rate 失败: {e}")
            status["high_rate"] = f"error: {e}"

        try:
            exceedance = self._filter_by_food(self.loader.load_exceedance(), food_cat, food_sub, item)
            if year and "niandu" in exceedance.columns:
                exceedance = exceedance[exceedance["niandu"].astype(int) == int(year)]
            results["exceedance"] = exceedance
            status["exceedance"] = "success"
        except Exception as e:
            logger.warning(f"加载 exceedance 失败: {e}")
            status["exceedance"] = f"error: {e}"

        try:
            near_limit = self._filter_by_food(self.loader.load_near_limit(), food_cat, food_sub, item)
            if year and "niandu" in near_limit.columns:
                near_limit = near_limit[near_limit["niandu"].astype(int) == int(year)]
            results["near_limit"] = near_limit
            status["near_limit"] = "success"
        except Exception as e:
            logger.warning(f"加载 near_limit 失败: {e}")
            status["near_limit"] = f"error: {e}"

        try:
            dl_trend = self._filter_by_food(self.loader.load_category_trend("sp_s_17"), food_cat, food_sub, item)
            results["category_trend"] = dl_trend
            status["category_trend"] = "success"
        except Exception as e:
            logger.warning(f"加载 category_trend 失败: {e}")
            status["category_trend"] = f"error: {e}"

        try:
            xm_trend = self.loader.load_xm_trend()
            if item:
                xm_trend = xm_trend[xm_trend["xiangmumingcheng"].astype(str).str.contains(item, na=False, regex=False)]
            results["item_trend"] = xm_trend
            status["item_trend"] = "success"
        except Exception as e:
            logger.warning(f"加载 item_trend 失败: {e}")
            status["item_trend"] = f"error: {e}"

        try:
            seasonal = self.loader.load_seasonal("jjxfx_report")
            if food_sub:
                seasonal = seasonal[seasonal["sp_s_20"].astype(str).str.contains(food_sub, na=False, regex=False)]
            elif food_cat:
                if "sp_s_17" in seasonal.columns:
                    seasonal = seasonal[seasonal["sp_s_17"].astype(str).str.contains(food_cat, na=False, regex=False)]
            results["seasonal"] = seasonal
            status["seasonal"] = "success"
        except Exception as e:
            logger.warning(f"加载 seasonal 失败: {e}")
            status["seasonal"] = f"error: {e}"

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