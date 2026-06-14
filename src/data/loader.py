import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, resolve_csv_mapping, setup_logging

logger = setup_logging(__name__)


class DataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._cache: dict[str, pd.DataFrame] = {}
        self._mapping = resolve_csv_mapping(self.data_dir)
        self._file_to_key: dict[str, str] = {v: k for k, v in self._mapping.items()}

    _EXPECTED_COLUMNS: dict[str, list[str]] = {
        "prov_trend": ["niandu", "jdpc", "bhgpc", "bhgl"],
        "prov_batch_detail": ["niandu", "jdpc", "bhgpc", "bhgl"],
        "wgx_all": ["xiangmumingcheng", "sp_s_17"],
        "scjc_all": ["niandu", "xiangmumingcheng", "sp_s_17"],
        "high_rate": ["sp_s_20", "rate"],
        "dl_trend": ["sp_s_17"],
        "xm_trend": ["xiangmumingcheng"],
        "jjxfx_report": ["sp_s_20"],
    }

    def _read_csv(self, filename: str) -> pd.DataFrame:
        if filename in self._cache:
            return self._cache[filename].copy()
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        df = pd.read_csv(filepath)
        self._check_columns(filename, df)
        self._cache[filename] = df
        return df.copy()

    def _check_columns(self, filename: str, df: pd.DataFrame):
        key = self._file_to_key.get(filename)
        if key is None or key not in self._EXPECTED_COLUMNS:
            return
        expected = self._EXPECTED_COLUMNS[key]
        actual = set(df.columns)
        missing = [c for c in expected if c not in actual]
        if missing:
            logger.warning(f"表 [{key}] 缺少期望列: {missing}，实际列: {sorted(actual)}")

    def load(self, key: str) -> pd.DataFrame:
        if key not in self._mapping:
            available = list(self._mapping.keys())
            raise KeyError(f"未知数据表 '{key}'，可用: {available}")
        return self._read_csv(self._mapping[key])

    def load_prov_trend(self) -> pd.DataFrame:
        return self.load("prov_trend")

    def load_prov_batch_detail(self) -> pd.DataFrame:
        return self.load("prov_batch_detail")

    def load_category_trend(self, category_col: str = "sp_s_17") -> pd.DataFrame:
        key_map = {
            "sp_s_17": "dl_trend",
            "sp_s_2": "cycs_trend",
            "sp_s_4": "cy_trend",
            "sp_s_68": "cyhj_trend",
            "sp_s_220": "sc_trend",
            "sp_s_201": "qy_trend",
        }
        key = key_map.get(category_col, "dl_trend")
        return self.load(key)

    def load_category_detail(self, category_col: str = "sp_s_17") -> pd.DataFrame:
        key_map = {
            "sp_s_17": "dl_batch_detail",
            "sp_s_2": "cycs_batch_detail",
            "sp_s_4": "cy_batch_detail",
            "sp_s_68": "cyhj_batch_detail",
            "sp_s_220": "sc_batch_detail",
            "sp_s_201": "qy_batch_detail",
        }
        key = key_map.get(category_col, "dl_batch_detail")
        return self.load(key)

    def load_risk_items(self, risk_type: str = "all") -> pd.DataFrame:
        key_map = {
            "all": "wgx_all",
            "bcy": "wgx_bcy",
            "hj": "wgx_hj",
            "sc": "wgx_sc",
        }
        key = key_map.get(risk_type, "wgx_all")
        return self.load(key)

    def load_market_inspection(self, market_type: str = "all") -> pd.DataFrame:
        key_map = {
            "all": "scjc_all",
            "bcy": "scjc_bcy",
            "hj": "scjc_hj",
            "sc": "scjc_sc",
        }
        key = key_map.get(market_type, "scjc_all")
        return self.load(key)

    def load_seasonal(self, seasonal_type: str = "jjxfx") -> pd.DataFrame:
        key_map = {
            "jjxfx": "jjxfx",
            "jjxfx_report": "jjxfx_report",
            "jjxqs": "jjxqs",
            "jjxqs_report": "jjxqs_report",
        }
        key = key_map.get(seasonal_type, "jjxfx")
        return self.load(key)

    def load_exceedance(self) -> pd.DataFrame:
        return self.load("cbbs")

    def load_high_rate(self) -> pd.DataFrame:
        return self.load("high_rate")

    def load_near_limit(self) -> pd.DataFrame:
        return self.load("jxz_table")

    def load_xm_trend(self) -> pd.DataFrame:
        return self.load("xm_trend")

    def load_xm_batch_detail(self) -> pd.DataFrame:
        return self.load("xm_batch_detail")

    def load_xl_trend(self) -> pd.DataFrame:
        return self.load("xl_trend")

    def load_xl_batch_detail(self) -> pd.DataFrame:
        return self.load("xl_batch_detail")

    def load_xl_xm_trend(self) -> pd.DataFrame:
        return self.load("xl_xm_trend")

    def load_xl_xm_batch_detail(self) -> pd.DataFrame:
        return self.load("xl_xm_batch_detail")

    def load_dl_xm_trend(self) -> pd.DataFrame:
        return self.load("dl_xm_trend")

    def load_dl_xm_batch_detail(self) -> pd.DataFrame:
        return self.load("dl_xm_batch_detail")

    def list_available(self) -> list[str]:
        return list(self._mapping.keys())

    def clear_cache(self):
        self._cache.clear()