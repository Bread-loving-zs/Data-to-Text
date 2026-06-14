import json

SYSTEM_PROMPT = (
    "你是一位资深的食品安全抽检分析专家。请根据提供的抽检数据，"
    "撰写一段专业的食品安全抽检分析文本。"
    "要求：语言专业严谨，数据呈现清晰，在分析中应涵盖："
    "1) 确认分析类型（监督抽检/风险监测/综合分析）；"
    "2) 提取关键指标（不合格率、批次、趋势方向）；"
    "3) 判断统计显著性（P<0.05 则标注'差异有统计学意义'）；"
    "4) 识别高风险项目；"
    "5) 提出针对性建议。"
)

ANALYSIS_TYPE_TEMPLATES = {
    "监督抽检": "请撰写一份监督抽检分析报告，重点关注不合格率、不合格批次分布、"
               "以及同比/环比变化趋势。",
    "风险监测": "请撰写一份风险监测分析报告，重点关注风险项目检出率、"
               "超标情况、季节性风险和三年趋势。",
    "综合分析": "请撰写一份食品安全综合分析报告，涵盖抽检概况、分类分析、"
               "趋势研判和风险预警。",
}

SECTION_NAMES = {
    "首次检出问题项目情况": "首次检出",
    "首次检出不合格项目情况": "首次检出",
    "超标倍数情况": "超标倍数",
    "顽固性检验项目分析": "顽固性检验项目",
    "监督抽检中的趋势分析": "趋势分析",
    "风险监测中的趋势分析": "趋势分析",
    "监督抽检中的近限值分析": "近限值分析",
    "风险监测中的近限值分析": "近限值分析",
    "季节性风险": "季节性风险",
    "其他问题": "其他问题",
}

SECTION_INSTRUCTIONS = {
    "首次检出": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「首次检出不合格项目情况」分析报告。"
                   "需要说明首次检出不合格项目的定义、总体数量、主要食品类别、"
                   "各地区分布以及各抽样环节的分布情况。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「首次检出问题项目情况」分析报告。"
                   "需要说明首次检出问题项目的定义、总体数量、主要食品类别、"
                   "各地区分布以及各抽样环节的分布情况。",
    },
    "超标倍数": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「超标倍数情况」分析报告。"
                   "需要说明超标倍数的定义、总体超标情况、超标倍数最高的前10项。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「超标倍数情况」分析报告。"
                   "需要说明超标倍数的定义、总体超标情况、超标倍数最高的前10项。",
    },
    "顽固性检验项目": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「顽固性检验项目分析」报告。"
                   "需要说明顽固性检验项目的定义和判定标准、全省/全市概况、"
                   "各地市/各县区分布情况、各抽样环节分布以及项目类别分布。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「顽固性检验项目分析」报告。"
                   "需要说明顽固性检验项目的定义和判定标准、全省/全市概况、"
                   "各地市/各县区分布情况、各抽样环节分布以及项目类别分布。",
    },
    "趋势分析": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「趋势分析」报告。"
                   "需要说明三年总体趋势、各类食品趋势、各地区趋势、"
                   "各环节和场所趋势、以及细类项目不合格率的升降变化。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「趋势分析」报告。"
                   "需要说明三年总体趋势、各类食品趋势、各地区趋势、"
                   "各环节和场所趋势、以及细类项目风险发现率的升降变化。",
    },
    "近限值分析": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「近限值分析」报告。"
                   "需要说明近限值分析的方法和目的，列出检验结果异常聚集在最大允许限值附近的食品细类。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「近限值分析」报告。"
                   "需要说明近限值分析的方法和目的，列出检验结果异常聚集在最大允许限值附近的食品细类。",
    },
    "季节性风险": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「季节性风险」分析报告。"
                   "需要说明季节性风险的判定标准、各季度风险分布、"
                   "以及三年季节性趋势变化。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「季节性风险」分析报告。"
                   "需要说明季节性风险的判定标准、各季度风险分布、"
                   "以及三年季节性趋势变化。",
    },
    "其他问题": {
        "监督抽检": "请根据以下食品安全监督抽检数据，撰写「其他问题」分析报告。"
                   "需要说明项目不合格项次检出率较高的食品类别、"
                   "以及连续多年不合格的食品细类和检验项目。",
        "风险监测": "请根据以下食品安全风险监测数据，撰写「其他问题」分析报告。"
                   "需要说明项目问题率较高的食品类别、"
                   "以及连续多年出现问题的食品细类和检验项目。",
    },
}


def format_value(v, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(v, dict):
        lines = []
        for kk, vv in v.items():
            if kk == "query":
                continue
            if isinstance(vv, (dict, list)):
                lines.append(f"{prefix}- {kk}:")
                lines.append(format_value(vv, indent + 1))
            else:
                if isinstance(vv, float):
                    lines.append(f"{prefix}- {kk}: {vv:.4f}")
                elif isinstance(vv, int) and not isinstance(vv, bool):
                    lines.append(f"{prefix}- {kk}: {vv}")
                else:
                    lines.append(f"{prefix}- {kk}: {vv}")
        return "\n".join(lines)
    elif isinstance(v, list):
        lines = []
        for item in v:
            if isinstance(item, dict):
                lines.append(format_value(item, indent))
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    else:
        if isinstance(v, float):
            return f"{prefix}{v:.4f}"
        return f"{prefix}{v}"


def format_alpaca_sample(sample: dict) -> dict:
    input_data = sample.get("input", {})
    output_text = sample.get("output", "")

    analysis_type = input_data.get("analysis_type", "综合分析")
    region = input_data.get("region", "")
    year = input_data.get("year", "")
    section_name = input_data.get("section", "")

    section_key = SECTION_NAMES.get(section_name, "综合分析")
    instruction = SECTION_INSTRUCTIONS.get(section_key, {}).get(
        analysis_type,
        f"请根据以下食品安全{analysis_type}数据，撰写分析报告。"
    )

    meta_parts = []
    if region:
        meta_parts.append(f"地区：{region}")
    if year:
        meta_parts.append(f"年份：{year}")
    if section_name:
        meta_parts.append(f"分析章节：{section_name}")

    data_lines = []
    for key, value in input_data.items():
        if key in ("analysis_type", "region", "region_level",
                    "year", "section", "section_key", "row_count"):
            continue
        if isinstance(value, list):
            data_lines.append(f"\n### {key}")
            data_lines.append(format_value(value))
        elif isinstance(value, dict):
            data_lines.append(f"\n### {key}")
            data_lines.append(format_value(value))
        else:
            if isinstance(value, float):
                data_lines.append(f"- {key}: {value:.4f}")
            else:
                data_lines.append(f"- {key}: {value}")

    input_text = "\n".join(meta_parts)
    if data_lines:
        input_text += "\n\n## 结构化数据\n" + "\n".join(data_lines)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def format_training_sample(sample: dict) -> dict:
    input_data = sample.get("input", {})
    output_text = sample.get("output", "")

    query = input_data.get("query", "")
    analysis_type = input_data.get("analysis_type", "综合分析")
    region = input_data.get("region", "")
    year = input_data.get("year", "")
    section = input_data.get("section", "")
    subsection = input_data.get("subsection", "")
    sub_subsection = input_data.get("sub_subsection", "")

    ag_instruction = ANALYSIS_TYPE_TEMPLATES.get(
        analysis_type, ANALYSIS_TYPE_TEMPLATES["综合分析"]
    )

    parts = [ag_instruction]
    if query:
        parts.append(f"\n用户需求：{query}")

    meta_parts = []
    if region:
        meta_parts.append(f"地区：{region}")
    if year:
        meta_parts.append(f"年份：{year}")
    if section:
        meta_parts.append(f"章节：{section}")
    if subsection:
        meta_parts.append(f"子章节：{subsection}")
    if sub_subsection:
        meta_parts.append(f"具体分析：{sub_subsection}")
    if meta_parts:
        parts.append("\n## 分析上下文")
        parts.extend(meta_parts)

    parts.append(f"\n## 结构化数据")
    data_lines = []
    for key, value in input_data.items():
        if key in ("query", "analysis_type", "region", "region_level",
                    "year", "section", "subsection", "sub_subsection",
                    "section_path", "row_count"):
            continue
        if isinstance(value, dict):
            data_lines.append(f"\n### {key}")
            data_lines.append(format_value(value))
        elif isinstance(value, list):
            data_lines.append(f"\n### {key}")
            data_lines.append(format_value(value))
        else:
            if isinstance(value, float):
                data_lines.append(f"- {key}: {value:.4f}")
            else:
                data_lines.append(f"- {key}: {value}")
    parts.append("\n".join(data_lines) if data_lines else json.dumps(input_data, ensure_ascii=False))

    user_text = "\n".join(parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output_text},
        ]
    }