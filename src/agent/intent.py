import json
import re
from typing import Optional

from src.agent.llm_client import LLMClient
from src.config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    INFERENCE_BACKEND, INTENT_USE_LLM,
    PROMPT_DIR, setup_logging,
)

logger = setup_logging(__name__)

KNOWN_CATEGORIES = [
    "水果制品", "蔬菜制品", "保健食品", "方便食品", "水产制品",
    "调味品", "肉制品", "乳制品", "饮料", "糕点", "辣椒",
    "酒类", "食用油", "粮食加工品", "炒货", "茶叶", "豆制品",
    "蛋制品", "蜂产品", "糖果制品", "薯类和膨化食品", "饼干",
    "冷冻饮品", "速冻食品", "罐头", "食糖", "淀粉及淀粉制品",
    "特殊膳食食品", "婴幼儿配方食品", "餐饮食品", "食盐",
]

KNOWN_REGIONS = [
    "北京", "上海", "广州", "深圳", "广东", "浙江", "江苏", "山东",
    "四川", "湖北", "湖南", "河南", "河北", "福建", "安徽", "辽宁",
    "陕西", "重庆", "天津", "江西", "广西", "云南", "贵州", "山西",
    "吉林", "黑龙江", "甘肃", "海南", "宁夏", "青海", "西藏", "新疆",
    "内蒙古",
]

DIMENSION_KEYWORDS = {
    "全省": "全省",
    "全省范围": "全省",
    "大类食品": "大类食品",
    "食品大类": "大类食品",
    "细类食品": "细类食品",
    "食品细类": "细类食品",
    "地市": "地市",
    "各市": "地市",
    "环节": "环节",
    "抽样场所": "抽样场所",
    "采样场所": "抽样场所",
}

CATEGORY_ALIASES = {
    "辣椒": ["辣椒类", "辣椒制品", "鲜辣椒"],
    "饮料": ["饮品", "饮料类"],
    "肉制品": ["肉类", "肉制品类", "肉类制品"],
    "乳制品": ["奶制品", "乳类", "乳品类"],
    "糕点": ["糕点类", "烘焙食品", "面包", "蛋糕"],
    "调味品": ["调味料", "调料", "调味品类"],
    "酒类": ["酒", "酒类产品", "酒精饮料"],
    "食用油": ["油", "油脂", "食用油类"],
    "粮食加工品": ["粮食", "大米", "面粉", "谷物"],
    "水产制品": ["水产", "水产品", "海鲜", "鱼类"],
}

REGION_ALIASES = {
    "四川": ["川", "蜀"],
    "广东": ["粤"],
    "浙江": ["浙"],
    "江苏": ["苏"],
    "山东": ["鲁"],
    "湖北": ["鄂"],
    "湖南": ["湘"],
    "河南": ["豫"],
    "河北": ["冀"],
    "福建": ["闽"],
    "安徽": ["皖"],
    "辽宁": ["辽"],
    "陕西": ["陕", "秦"],
    "重庆": ["渝"],
    "江西": ["赣"],
    "广西": ["桂"],
    "云南": ["滇", "云"],
    "贵州": ["黔", "贵"],
    "山西": ["晋"],
    "吉林": ["吉"],
    "黑龙江": ["黑"],
    "甘肃": ["甘", "陇"],
    "海南": ["琼"],
    "上海": ["沪"],
}

ANALYSIS_TYPE_KEYWORDS = {
    "监督抽检": "监督抽检",
    "风险监测": "风险监测",
    "综合分析": "综合分析",
    "综合": "综合分析",
}

REPORT_TYPE_FUZZY = {
    "监督抽检": ["抽检", "监督", "监督检查"],
    "风险监测": ["监测", "风险", "风险预警"],
    "综合分析": ["综合", "全面", "整体"],
}


class IntentRecognizer:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 use_llm: Optional[bool] = None, backend: Optional[str] = None):
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.use_llm = use_llm if use_llm is not None else INTENT_USE_LLM
        self.backend = backend or INFERENCE_BACKEND
        self._system_prompt = self._load_system_prompt()
        self._llm = LLMClient(model, host, backend)

    def _load_system_prompt(self) -> str:
        prompt_file = PROMPT_DIR / "intent_recognition.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return (
            "你是一个食品安全抽检报告意图识别助手。从用户指令中提取以下实体，以JSON格式输出：\n"
            '{"year": 年份, "quarter": 季度(1-4/null), "food_category": "食品大类/null", '
            '"food_subcategory": "食品细类/null", "item_name": "检测项目/null", '
            '"region": "地区/null", "report_type": "监督抽检/风险监测/综合分析", '
            '"dimension": "全省/大类食品/细类食品/地市/环节/抽样场所/null"}\n'
            "只输出JSON，不要输出其他内容。"
        )

    def recognize(self, user_input: str, use_llm: Optional[bool] = None) -> dict:
        if use_llm is None:
            use_llm = self.use_llm

        rule_result = self._rule_based(user_input)

        if not use_llm:
            logger.debug(f"规则识别: {rule_result}")
            return rule_result

        if self._result_is_complete(rule_result):
            logger.debug(f"规则识别完整: {rule_result}")
            return rule_result

        logger.info("规则识别不完整，启用LLM增强...")
        llm_result = self._call_llm(user_input)
        merged = self._merge_results(rule_result, llm_result)
        logger.debug(f"融合结果: {merged}")
        return merged

    def _result_is_complete(self, result: dict) -> bool:
        report_type = result.get("report_type")
        return bool(
            result.get("year")
            and report_type and report_type != "综合分析"
            and result.get("food_category")
        )

    def _merge_results(self, rule: dict, llm: dict) -> dict:
        merged = {}
        for key in rule:
            if rule[key] is not None and rule[key]:
                merged[key] = rule[key]
            elif llm.get(key) is not None and llm.get(key):
                merged[key] = llm[key]
            else:
                merged[key] = rule[key]
        for key in llm:
            if key not in merged:
                if llm[key] is not None and llm[key]:
                    merged[key] = llm[key]
        return merged

    def _rule_based(self, text: str) -> dict:
        result = {
            "year": self._extract_year(text),
            "quarter": self._extract_quarter(text),
            "food_category": self._extract_food_category(text),
            "food_subcategory": self._extract_food_subcategory(text),
            "item_name": None,
            "region": self._extract_region(text),
            "report_type": self._extract_report_type(text),
            "dimension": self._extract_dimension(text),
        }
        return result

    def _extract_year(self, text: str) -> Optional[int]:
        year_match = re.search(r"(\d{4})\s*年", text)
        if year_match:
            return int(year_match.group(1))
        year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
        if year_match:
            return int(year_match.group(1))
        return None

    def _extract_quarter(self, text: str) -> Optional[int]:
        q_map = {"一": 1, "二": 2, "三": 3, "四": 4,
                 "1": 1, "2": 2, "3": 3, "4": 4}
        match = re.search(r"[一二三四1-4]\s*季度", text)
        if match:
            q_str = match.group()
            for k, v in q_map.items():
                if k in q_str:
                    return v
        match = re.search(r"第\s*([一二三四1-4])\s*季度", text)
        if match:
            return q_map.get(match.group(1))
        match = re.search(r"Q\s*([1-4])", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_food_category(self, text: str) -> Optional[str]:
        matched = []
        for cat in KNOWN_CATEGORIES:
            if cat in text:
                matched.append((text.index(cat), cat))
        if matched:
            matched.sort(key=lambda x: x[0])
            return matched[0][1]

        for std_name, aliases in CATEGORY_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    return std_name
        return None

    def _extract_food_subcategory(self, text: str) -> Optional[str]:
        patterns = [
            r"食品细类[是为：:]\s*([^\s,，。]+)",
            r"细类[是为：:]\s*([^\s,，。]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_region(self, text: str) -> Optional[str]:
        for region in sorted(KNOWN_REGIONS, key=len, reverse=True):
            if region in text:
                return region

        for std_name, aliases in REGION_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    return std_name
        return None

    def _extract_report_type(self, text: str) -> str:
        for keyword, report_type in ANALYSIS_TYPE_KEYWORDS.items():
            if keyword in text:
                return report_type

        for std_name, keywords in REPORT_TYPE_FUZZY.items():
            for keyword in keywords:
                if keyword in text:
                    return std_name

        return "综合分析"

    def _extract_dimension(self, text: str) -> Optional[str]:
        for keyword, dimension in sorted(DIMENSION_KEYWORDS.items(),
                                          key=lambda x: len(x[0]), reverse=True):
            if keyword in text:
                return dimension
        return None

    def _call_llm(self, user_message: str) -> dict:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        content = self._llm.chat(messages, temperature=0.1, max_tokens=1024,
                                  response_format={"type": "json_object"} if self.backend == "vllm" else None)
        if content is None:
            return self._empty_result()
        return self._parse_json(content)

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "year": None, "quarter": None, "food_category": None,
            "food_subcategory": None, "item_name": None, "region": None,
            "report_type": "综合分析", "dimension": None
        }