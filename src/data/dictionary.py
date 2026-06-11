import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import DICT_FILE, setup_logging

logger = setup_logging(__name__)


class DataDictionary:
    def __init__(self, dict_path: Optional[Path] = None):
        self.dict_path = dict_path or DICT_FILE
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df.copy()
        if not self.dict_path.exists():
            raise FileNotFoundError(f"数据字典文件不存在: {self.dict_path}")
        self._df = pd.read_excel(self.dict_path)
        return self._df.copy()

    def get_field_info(self, field_name: str) -> Optional[dict]:
        df = self.load()
        cols = df.columns.tolist()
        name_col = cols[0] if cols else None
        if name_col is None:
            return None
        match = df[df[name_col].astype(str).str.strip() == field_name.strip()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def get_all_fields(self) -> list[str]:
        df = self.load()
        cols = df.columns.tolist()
        name_col = cols[0] if cols else None
        if name_col is None:
            return []
        return df[name_col].astype(str).str.strip().tolist()

    def search_fields(self, keyword: str) -> pd.DataFrame:
        df = self.load()
        cols = df.columns.tolist()
        name_col = cols[0] if cols else None
        if name_col is None:
            return pd.DataFrame()
        mask = df[name_col].astype(str).str.contains(keyword, case=False, na=False)
        return df[mask].copy()