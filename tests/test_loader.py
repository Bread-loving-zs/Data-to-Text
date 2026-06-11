import pytest
from pathlib import Path

from src.data.loader import DataLoader
from src.config import DATA_DIR


def _data_dir_exists():
    return DATA_DIR.is_dir()


@pytest.fixture
def loader():
    if not _data_dir_exists():
        pytest.skip("数据目录不存在")
    return DataLoader()


def test_init(loader):
    assert loader.data_dir is not None
    assert isinstance(loader._cache, dict)
    assert isinstance(loader._mapping, dict)
    assert len(loader._mapping) > 0


def test_list_available(loader):
    available = loader.list_available()
    assert isinstance(available, list)
    assert len(available) > 0
    assert "prov_trend" in available
    assert "prov_batch_detail" in available


def test_load_prov_trend(loader):
    df = loader.load_prov_trend()
    assert df is not None
    assert len(df) > 0
    trend_cols = [c for c in df.columns if "bhgl" in c or "jdpc" in c or "bhgpc" in c]
    assert len(trend_cols) > 0, f"缺少趋势关键列，实际列: {list(df.columns)[:10]}"


def test_load_prov_batch_detail(loader):
    df = loader.load_prov_batch_detail()
    assert df is not None
    assert len(df) > 0
    assert "niandu" in df.columns
    assert "jdpc" in df.columns
    assert "bhgpc" in df.columns
    assert "bhgl" in df.columns


def test_cache(loader):
    df1 = loader.load_prov_trend()
    df2 = loader.load_prov_trend()
    assert df1.equals(df2)


def test_clear_cache(loader):
    loader.load_prov_trend()
    assert len(loader._cache) > 0
    loader.clear_cache()
    assert len(loader._cache) == 0