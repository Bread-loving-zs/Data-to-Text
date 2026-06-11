import pytest
from pathlib import Path

from src.data.loader import DataLoader
from src.agent.query import DataQuerier
from src.config import DATA_DIR


def _data_dir_exists():
    return DATA_DIR.is_dir()


@pytest.fixture
def querier():
    if not _data_dir_exists():
        pytest.skip("数据目录不存在")
    loader = DataLoader()
    return DataQuerier(loader=loader)


def test_query_by_intent_basic(querier):
    intent = {"year": 2024, "food_category": None}
    results = querier.query_by_intent(intent)
    assert isinstance(results, dict)
    assert "_query_status" in results
    status = results["_query_status"]
    success_count = sum(1 for v in status.values() if v == "success")
    assert success_count > 0, f"期望至少1个表查询成功, 实际: {status}"


def test_query_status(querier):
    intent = {"year": 2024}
    results = querier.query_by_intent(intent)
    assert "_query_status" in results
    status = results["_query_status"]
    assert isinstance(status, dict)
    for table_key, table_status in status.items():
        assert table_status in ("success",) or table_status.startswith("error:"), \
            f"表 {table_key} 状态异常: {table_status}"


def test_filter_by_food(querier):
    import pandas as pd
    df = pd.DataFrame({
        "sp_s_17": ["大米", "小麦粉", "辣椒"],
        "sp_s_20": ["粳米", "标准粉", "辣椒粉"],
        "total": [100, 200, 300],
    })
    filtered = querier._filter_by_food(df, food_cat="大米", food_sub=None, item=None)
    assert len(filtered) == 1
    assert filtered.iloc[0]["sp_s_17"] == "大米"

    filtered = querier._filter_by_food(df, food_cat=None, food_sub="标准粉", item=None)
    assert len(filtered) == 1
    assert filtered.iloc[0]["sp_s_20"] == "标准粉"


def test_empty_intent(querier):
    intent = {}
    results = querier.query_by_intent(intent)
    assert isinstance(results, dict)
    assert "_query_status" in results