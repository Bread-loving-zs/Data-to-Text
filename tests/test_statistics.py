import pytest

from src.agent.statistics import StatisticsEngine


def test_chi_square():
    result = StatisticsEngine.chi_square_test([10, 12, 15, 8, 11])
    assert "chi2" in result, f"缺少 chi2: {result}"
    assert "p_value" in result, f"缺少 p_value: {result}"
    assert "significant" in result, f"缺少 significant: {result}"
    assert isinstance(result["chi2"], float)
    assert isinstance(result["p_value"], float)


def test_yoy_change():
    result = StatisticsEngine.yoy_change(0.15, 0.10)
    assert result["yoy_change"] == 0.05
    assert result["yoy_change_pct"] == 50.0
    assert result["direction"] == "上升"

    result = StatisticsEngine.yoy_change(0.05, 0.10)
    assert result["direction"] == "下降"

    result = StatisticsEngine.yoy_change(0.10, 0.10)
    assert result["direction"] == "持平"


def test_trend_analysis():
    result = StatisticsEngine.trend_analysis([0.05, 0.08, 0.12, 0.15])
    assert "slope" in result
    assert "r_squared" in result
    assert result["direction"] == "上升"
    assert result["slope"] > 0

    result = StatisticsEngine.trend_analysis([0.15, 0.12, 0.08, 0.05])
    assert result["direction"] == "下降"


def test_trend_insufficient_data():
    result = StatisticsEngine.trend_analysis([0.5])
    assert "error" in result


def test_wilson_ci():
    result = StatisticsEngine.wilson_confidence_interval(45, 1000, 0.95)
    assert result["rate"] is not None
    assert result["lower"] is not None
    assert result["upper"] is not None
    assert result["lower"] <= result["rate"] <= result["upper"]


def test_wilson_ci_zero():
    result = StatisticsEngine.wilson_confidence_interval(0, 0)
    assert result["rate"] is None


def test_detect_anomalies_iqr():
    values = [0.05, 0.06, 0.04, 0.07, 0.05, 0.50, 0.06]
    anomalies = StatisticsEngine.detect_anomalies(values, method="iqr")
    assert 5 in anomalies, f"期望索引5为异常值, 实际: {anomalies}"


def test_detect_anomalies_zscore():
    values = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99]
    anomalies = StatisticsEngine.detect_anomalies(values, method="zscore")
    assert 9 in anomalies, f"期望索引9为异常值, 实际: {anomalies}"


def test_compare_groups():
    result = StatisticsEngine.compare_groups({
        "A": [0.05, 0.06, 0.04, 0.07],
        "B": [0.12, 0.15, 0.11, 0.14],
    })
    assert "f_statistic" in result
    assert "p_value" in result
    assert result["significant"]