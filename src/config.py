import os
import re
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "自动分析报告数据模板" / "数据表")))
DICT_FILE = Path(os.environ.get("DICT_FILE", str(PROJECT_ROOT / "自动分析报告数据模板" / "sheng_jdcj_dictionary数据字典.xlsx")))

OUTPUT_DIR = PROJECT_ROOT / "output"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
PROMPT_DIR = PROJECT_ROOT / "prompts"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "")
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY", "")

VLLM_API_URL = os.environ.get("VLLM_API_URL", "")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "ollama")

INTENT_USE_LLM = os.environ.get("INTENT_USE_LLM", "false").lower() == "true"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging(name: str = "data_to_text", level: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level or LOG_LEVEL, logging.INFO))
    return logger


logger = setup_logging(__name__)


CSV_FILE_MAPPING = {
    "cbbs": "jdcj_prov_cbbs_202606021555.csv",
    "high_rate": "jdcj_prov_high_rate_202606021555.csv",
    "jjxfx": "jdcj_prov_jjxfx_202606021556.csv",
    "jjxfx_report": "jdcj_prov_jjxfx_report_202606021556.csv",
    "jjxqs": "jdcj_prov_jjxqs_202606021556.csv",
    "jjxqs_report": "jdcj_prov_jjxqs_report_202606021556.csv",
    "scjc_all": "jdcj_prov_scjc_all_202606021556.csv",
    "scjc_bcy": "jdcj_prov_scjc_bcy_202606021556.csv",
    "scjc_hj": "jdcj_prov_scjc_hj_202606021556.csv",
    "scjc_sc": "jdcj_prov_scjc_sc_202606021556.csv",
    "wgx_all": "jdcj_prov_wgx_all_202606021557.csv",
    "wgx_bcy": "jdcj_prov_wgx_bcy_202606021557.csv",
    "wgx_hj": "jdcj_prov_wgx_hj_202606021557.csv",
    "wgx_sc": "jdcj_prov_wgx_sc_202606021557.csv",
    "cy_batch_detail": "sheng_jdcj_cy_batch_detail_202606021557.csv",
    "cy_trend": "sheng_jdcj_cy_trend_202606021557.csv",
    "cycs_batch_detail": "sheng_jdcj_cycs_batch_detail_202606021558.csv",
    "cycs_trend": "sheng_jdcj_cycs_trend_202606021558.csv",
    "cyhj_batch_detail": "sheng_jdcj_cyhj_batch_detail_202606021558.csv",
    "cyhj_trend": "sheng_jdcj_cyhj_trend_202606021558.csv",
    "dl_batch_detail": "sheng_jdcj_dl_batch_detail_202606021558.csv",
    "dl_trend": "sheng_jdcj_dl_trend_202606021559.csv",
    "dl_xm_batch_detail": "sheng_jdcj_dl_xm_batch_detail_202606021559.csv",
    "dl_xm_trend": "sheng_jdcj_dl_xm_trend_202606021559.csv",
    "jxz_table": "sheng_jdcj_jxz_table_202606021559.csv",
    "prov_batch_detail": "sheng_jdcj_prov_batch_detail_202606021559.csv",
    "prov_trend": "sheng_jdcj_prov_trend_202606021559.csv",
    "qy_batch_detail": "sheng_jdcj_qy_batch_detail_202606021559.csv",
    "qy_trend": "sheng_jdcj_qy_trend_202606021600.csv",
    "sc_batch_detail": "sheng_jdcj_sc_batch_detail_202606021600.csv",
    "sc_trend": "sheng_jdcj_sc_trend_202606021600.csv",
    "xl_batch_detail": "sheng_jdcj_xl_batch_detail_202606021600.csv",
    "xl_trend": "sheng_jdcj_xl_trend_202606021600.csv",
    "xl_xm_batch_detail": "sheng_jdcj_xl_xm_batch_detail_202606021601.csv",
    "xl_xm_trend": "sheng_jdcj_xl_xm_trend_202606021601.csv",
    "xm_batch_detail": "sheng_jdcj_xm_batch_detail_202606021601.csv",
    "xm_trend": "sheng_jdcj_xm_trend_202606021601.csv",
}

CSV_FILE_PATTERNS: dict[str, str] = {}
for _key, _filename in CSV_FILE_MAPPING.items():
    _match = re.match(r"^(.+?)_\d{12,}$", Path(_filename).stem)
    if _match:
        CSV_FILE_PATTERNS[_key] = _match.group(1)


def resolve_csv_mapping(data_dir: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}

    for key, hardcoded_filename in CSV_FILE_MAPPING.items():
        filepath = data_dir / hardcoded_filename
        if filepath.exists():
            resolved[key] = hardcoded_filename
            continue
        prefix = CSV_FILE_PATTERNS.get(key)
        if prefix is None:
            resolved[key] = hardcoded_filename
            continue
        if not data_dir.is_dir():
            resolved[key] = hardcoded_filename
            continue
        matched = None
        prefix_sep = prefix + "_"
        for f in data_dir.glob("*.csv"):
            stem = f.stem
            if stem.startswith(prefix_sep):
                suffix = stem[len(prefix_sep):]
                if re.fullmatch(r"\d{12,}", suffix):
                    if matched is not None:
                        logger.warning(
                            f"键 [{key}] 匹配到多个文件: {matched} 和 {f.name}，使用 {f.name}"
                        )
                    matched = f.name
        if matched:
            resolved[key] = matched
        else:
            logger.warning(
                f"键 [{key}] 在 {data_dir} 中未找到匹配的 CSV 文件（前缀: {prefix}）"
            )
            resolved[key] = hardcoded_filename

    return resolved

SCHEMA_META = {
    "batch_detail": {
        "base_cols": ["jdpc", "hgpc", "bhgpc", "bhgl", "wilson_ci_95", "ci_length", "bhgl_new"],
        "desc": "单年度批次明细"
    },
    "trend": {
        "base_cols": ["slope", "r_squared", "yoy_change", "trend_3y", "chi2_3y", "p_value",
                       "chi2_2y", "p_value_2", "is_continuous_unqualified", "yoy_change_new"],
        "desc": "三年趋势对比"
    },
    "wgx": {
        "base_cols": ["rate_trend"],
        "desc": "违规项三年对比"
    },
    "scjc": {
        "base_cols": ["total", "fail", "rate"],
        "desc": "市场抽检明细"
    },
}

FOOD_CATEGORY_COLS = ["sp_s_17", "sp_s_18", "sp_s_19", "sp_s_20"]
FOOD_CATEGORY_NAMES = ["大类食品", "食品亚类", "食品次亚类", "细类食品"]