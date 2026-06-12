import json
import re
from pathlib import Path
from typing import Optional

from src.config import TRAINING_DATA_DIR, setup_logging
from src.data.loader import DataLoader
from src.training.shared import format_training_sample, format_alpaca_sample, SECTION_NAMES

logger = setup_logging(__name__)

SECTION_PATTERN_L1 = re.compile(r'^[一二三四五六七八九十]+、(.+)$')
SECTION_PATTERN_L2 = re.compile(r'^[（(][一二三四五六七八九十]+[）)]\s*(.+)$')
SECTION_PATTERN_L3 = re.compile(r'^\d+[、，.]\s*(.+)$')
SECTION_PATTERN_L4 = re.compile(r'^[（(]\d+[）)]\s*(.+)$')

BAREL2_PATTERNS = [
    "监督抽检中的趋势分析",
    "风险监测中的趋势分析",
]

CITY_NAMES = ["北海", "南宁", "崇左", "来宾", "柳州", "桂林", "梧州", "河池", "防城港"]


class TrainingDataPreparer:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or TRAINING_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()

    def process_real_samples(self, samples: list[dict], output_filename: str = "training_data.jsonl",
                              mode: str = "w") -> Path:
        return self._process_and_write_samples(
            samples=samples,
            output_filename=output_filename,
            mode=mode,
            format_func=None,
            log_label="训练样本"
        )

    def load_existing_samples(self, filename: str = "training_data.jsonl") -> list[dict]:
        filepath = self.output_dir / filename
        if not filepath.exists():
            return []
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"跳过无效JSON行 #{line_num}")
        return samples

    def _process_and_write_samples(
        self,
        samples: list[dict],
        output_filename: str,
        mode: str = "w",
        format_func: Optional[callable] = None,
        key_extractor: Optional[callable] = None,
        log_label: str = "训练样本"
    ) -> Path:
        cleaned = []
        for s in samples:
            if self._validate_sample(s):
                cleaned.append(s)
            else:
                logger.warning(f"跳过无效样本: {str(s)[:80]}...")

        deduped = self._deduplicate(cleaned)
        if len(cleaned) != len(deduped):
            logger.info(f"去重: {len(cleaned)} -> {len(deduped)} 条")

        processed = [format_func(s) for s in deduped] if format_func else deduped

        if key_extractor is None:
            if format_func:
                key_extractor = lambda item: json.dumps(item.get("input", ""), ensure_ascii=False, sort_keys=True)
            else:
                key_extractor = lambda item: json.dumps(item.get("input", {}), ensure_ascii=False, sort_keys=True)

        output_path = self.output_dir / output_filename
        write_mode = "w" if mode == "w" else "a"

        if write_mode == "w":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in processed:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            existing = self.load_existing_samples(output_filename)
            existing_keys = {key_extractor(s) for s in existing}
            new_count = 0
            with open(output_path, "a", encoding="utf-8") as f:
                for item in processed:
                    key = key_extractor(item)
                    if key not in existing_keys:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        existing_keys.add(key)
                        new_count += 1
            logger.info(f"追加 {new_count} 条新样本")

        logger.info(f"已写入 {len(processed)} 条{log_label} -> {output_path}")
        return output_path

    def load_samples_from_jsonl(self, filepath: Path) -> list[dict]:
        if not filepath.exists():
            logger.error(f"文件不存在: {filepath}")
            return []
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"跳过无效行: {line[:60]}...")
        logger.info(f"从 {filepath} 加载 {len(samples)} 条样本")
        return samples

    def load_samples_from_csv_pairs(self, input_csv: Path, output_csv: Path) -> list[dict]:
        import pandas as pd
        if not input_csv.exists():
            logger.error(f"输入文件不存在: {input_csv}")
            return []
        if not output_csv.exists():
            logger.error(f"输出文件不存在: {output_csv}")
            return []

        input_df = pd.read_csv(input_csv)
        output_df = pd.read_csv(output_csv)

        min_len = min(len(input_df), len(output_df))
        samples = []
        for i in range(min_len):
            input_row = input_df.iloc[i].dropna().to_dict()
            output_text = str(output_df.iloc[i].iloc[0]) if len(output_df.columns) > 0 else ""

            input_clean = {}
            for k, v in input_row.items():
                if isinstance(v, float) and v == int(v):
                    input_clean[str(k)] = int(v)
                else:
                    input_clean[str(k)] = v

            samples.append({"input": input_clean, "output": output_text})

        logger.info(f"从CSV对加载 {len(samples)} 条样本")
        return samples

    def _extract_report_metadata(self, filename: str) -> dict:
        base = Path(filename).stem
        metadata = {"report_name": filename}

        pattern = re.compile(
            r'^(?:(\d{4})年)?(?:广西)?(.*?)(监督抽检|风险监测)\d*$'
        )
        match = pattern.match(base)
        if match:
            year_str = match.group(1)
            region_suffix = match.group(2).strip()
            analysis_type = match.group(3)

            metadata["analysis_type"] = analysis_type
            if year_str:
                metadata["year"] = int(year_str)

            if region_suffix:
                metadata["region"] = f"广西{region_suffix}" if region_suffix else "广西"
                is_city = any(city in region_suffix for city in CITY_NAMES)
                metadata["region_level"] = "市级" if is_city else "省级"
            else:
                metadata["region"] = "广西"
                metadata["region_level"] = "省级"
        else:
            if "监督抽检" in base:
                metadata["analysis_type"] = "监督抽检"
            elif "风险监测" in base:
                metadata["analysis_type"] = "风险监测"
            metadata["region"] = "未知"
            metadata["region_level"] = "未知"

        return metadata

    @staticmethod
    def _detect_section_header(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        l1 = SECTION_PATTERN_L1.match(text)
        l2 = SECTION_PATTERN_L2.match(text)
        l3 = SECTION_PATTERN_L3.match(text)
        l4 = SECTION_PATTERN_L4.match(text)

        if l4:
            return ("L4", l4.group(1), None)
        if l3:
            return ("L3", l3.group(1), None)
        if l2:
            return ("L2", l2.group(1), None)
        if l1:
            return ("L1", l1.group(1), None)
        return (None, None, None)

    def load_samples_from_docx_pairs(self, data_dir: Path, report_docx: Path) -> list[dict]:
        try:
            from docx import Document
        except ImportError:
            logger.error("缺少 python-docx 依赖: pip install python-docx")
            return []

        if not report_docx.exists():
            logger.error(f"报告文件不存在: {report_docx}")
            return []

        logger.info(f"解析 DOCX 报告: {report_docx}")
        metadata = self._extract_report_metadata(report_docx.name)
        logger.info(f"  元数据: {metadata.get('analysis_type')} | "
                    f"{metadata.get('region')} | {metadata.get('year')} | "
                    f"{metadata.get('region_level')}")

        doc = Document(report_docx)

        current_section = ""
        current_subsection = ""
        current_sub_subsection = ""
        current_section_path: list[str] = []

        tables = []
        current_text_lines: list[str] = []
        current_table = None

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "tbl":
                if current_table is not None and current_text_lines:
                    section_path = list(current_section_path)
                    tables.append({
                        "table": current_table,
                        "text": "\n".join(current_text_lines).strip(),
                        "section": current_section,
                        "subsection": current_subsection,
                        "sub_subsection": current_sub_subsection,
                        "section_path": section_path,
                    })
                current_table = self._parse_docx_table(element)
                current_text_lines = []
            elif tag == "p":
                text = self._extract_paragraph_text(element)
                if text:
                    level, title, _ = self._detect_section_header(text)
                    if level == "L1":
                        current_section = title
                        current_subsection = ""
                        current_sub_subsection = ""
                        current_section_path = [title]
                    elif level == "L2":
                        current_subsection = title
                        current_sub_subsection = ""
                        if len(current_section_path) >= 1:
                            current_section_path = current_section_path[:1]
                        current_section_path.append(title)
                    elif level == "L3":
                        current_sub_subsection = title
                        if len(current_section_path) >= 2:
                            current_section_path = current_section_path[:2]
                        current_section_path.append(title)
                    current_text_lines.append(text)

        if current_table is not None and current_text_lines:
            tables.append({
                "table": current_table,
                "text": "\n".join(current_text_lines).strip(),
                "section": current_section,
                "subsection": current_subsection,
                "sub_subsection": current_sub_subsection,
                "section_path": list(current_section_path),
            })

        samples = self._convert_table_text_pairs(tables, metadata)
        logger.info(f"从 DOCX 提取 {len(samples)} 条数据-文本对")
        return samples

    @staticmethod
    def _parse_docx_table(element) -> list[list[str]]:
        rows = []
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for tr in element.findall(".//w:tr", ns):
            cells = []
            for tc in tr.findall(".//w:tc", ns):
                cell_texts = []
                for p in tc.findall(".//w:p", ns):
                    text_parts = []
                    for t in p.findall(".//w:t", ns):
                        if t.text:
                            text_parts.append(t.text)
                    if text_parts:
                        cell_texts.append("".join(text_parts))
                cells.append(" ".join(cell_texts).strip())
            if cells:
                rows.append(cells)
        return rows

    @staticmethod
    def _extract_paragraph_text(element) -> str:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parts = []
        for t in element.findall(".//w:t", ns):
            if t.text:
                parts.append(t.text)
        return "".join(parts).strip()

    @staticmethod
    def _convert_table_text_pairs(pairs: list[dict], metadata: dict) -> list[dict]:
        samples = []
        for pair in pairs:
            table = pair["table"]
            text = pair["text"]
            if len(table) < 2 or not text:
                continue

            headers = [h.strip() for h in table[0]]
            data_rows = []
            for row in table[1:]:
                row_dict = {}
                for i, cell in enumerate(row):
                    col_name = headers[i] if i < len(headers) else f"col_{i}"
                    row_dict[col_name] = cell
                data_rows.append(row_dict)

            input_data = {
                "analysis_type": metadata.get("analysis_type", "未知"),
                "region": metadata.get("region", "未知"),
                "region_level": metadata.get("region_level", "未知"),
                "year": metadata.get("year"),
                "section": pair.get("section", ""),
                "subsection": pair.get("subsection", ""),
                "sub_subsection": pair.get("sub_subsection", ""),
                "section_path": pair.get("section_path", []),
                "headers": headers,
                "rows": data_rows,
                "row_count": len(data_rows),
            }

            samples.append({
                "input": input_data,
                "output": text,
                "source": "docx",
                "report_name": metadata.get("report_name", ""),
            })
        return samples

    def load_samples_from_docx_by_section(self, data_dir: Path, report_docx: Path) -> list[dict]:
        try:
            from docx import Document
        except ImportError:
            logger.error("缺少 python-docx 依赖: pip install python-docx")
            return []

        if not report_docx.exists():
            logger.error(f"报告文件不存在: {report_docx}")
            return []

        logger.info(f"按分析类型解析 DOCX: {report_docx.name}")
        metadata = self._extract_report_metadata(report_docx.name)
        analysis_type = metadata.get("analysis_type", "未知")
        region = metadata.get("region", "未知")
        year = metadata.get("year")
        region_level = metadata.get("region_level", "未知")
        logger.info(f"  {analysis_type} | {region} | {year} | {region_level}")

        doc = Document(report_docx)

        sections: dict[str, dict] = {}
        current_l2_section = ""
        current_l2_title = ""
        pending_text_before_table: list[str] = []

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "tbl":
                table_data = self._parse_docx_table(element)
                if current_l2_section and table_data:
                    if current_l2_section not in sections:
                        sections[current_l2_section] = {
                            "title": current_l2_title,
                            "tables": [],
                            "texts": [],
                        }
                    sections[current_l2_section]["tables"].append(table_data)
                    if pending_text_before_table:
                        sections[current_l2_section]["texts"].extend(pending_text_before_table)
                        pending_text_before_table = []
            elif tag == "p":
                text = self._extract_paragraph_text(element)
                if not text:
                    continue
                level, title, _ = self._detect_section_header(text)
                if level == "L2":
                    current_l2_section = title
                    current_l2_title = title
                    pending_text_before_table = [text]
                elif text in BAREL2_PATTERNS:
                    current_l2_section = text
                    current_l2_title = text
                    pending_text_before_table = [text]
                elif level == "L1":
                    current_l2_section = ""
                    current_l2_title = ""
                    pending_text_before_table = []
                elif current_l2_section and not text.startswith("表"):
                    pending_text_before_table.append(text)

        samples = []
        for section_name, section_data in sections.items():
            section_key = SECTION_NAMES.get(section_name)
            if not section_key:
                logger.warning(f"  跳过未知章节: {section_name}")
                continue

            tables = section_data["tables"]
            text_lines = section_data["texts"]

            description_lines = []
            section_title = section_data["title"]
            for line in text_lines:
                if line == section_name or line == section_title:
                    continue
                level, _, _ = self._detect_section_header(line)
                if level:
                    continue
                if line.startswith("表"):
                    continue
                if line.startswith("注:"):
                    continue
                if line.startswith("附表"):
                    continue
                if line.startswith("（按"):
                    continue
                description_lines.append(line)

            description = "\n".join(description_lines).strip()

            structured_tables = []
            for t in tables:
                if len(t) < 2:
                    continue
                headers = [h.strip() for h in t[0]]
                data_rows = []
                for row in t[1:]:
                    row_dict = {}
                    for i, cell in enumerate(row):
                        col_name = headers[i] if i < len(headers) else f"col_{i}"
                        row_dict[col_name] = cell
                    data_rows.append(row_dict)
                structured_tables.append({
                    "headers": headers,
                    "rows": data_rows,
                    "row_count": len(data_rows),
                })

            input_data = {
                "analysis_type": analysis_type,
                "region": region,
                "region_level": region_level,
                "year": year,
                "section": section_name,
                "section_key": section_key,
                "tables": structured_tables,
                "table_count": len(structured_tables),
            }

            samples.append({
                "input": input_data,
                "output": description,
                "source": "docx_section",
                "report_name": metadata.get("report_name", ""),
            })

        logger.info(f"  提取 {len(samples)} 个分析类型样本")
        return samples

    def process_real_samples_alpaca(self, samples: list[dict],
                                     output_filename: str = "training_data_alpaca.jsonl",
                                     mode: str = "w") -> Path:
        return self._process_and_write_samples(
            samples=samples,
            output_filename=output_filename,
            mode=mode,
            format_func=format_alpaca_sample,
            log_label="Alpaca 格式训练样本"
        )

    def get_statistics(self, filename: str = "training_data.jsonl") -> dict:
        samples = self.load_existing_samples(filename)
        if not samples:
            return {"total": 0}

        input_keys = set()
        analysis_types = set()
        for s in samples:
            input_keys.update(s.get("input", {}).keys())
            at = s.get("input", {}).get("analysis_type", "未知")
            analysis_types.add(at)

        output_lengths = [len(s.get("output", "")) for s in samples]

        return {
            "total": len(samples),
            "input_fields": sorted(input_keys),
            "analysis_types": sorted(analysis_types),
            "avg_output_length": round(sum(output_lengths) / len(output_lengths), 0) if output_lengths else 0,
            "min_output_length": min(output_lengths) if output_lengths else 0,
            "max_output_length": max(output_lengths) if output_lengths else 0,
        }

    def export_for_finetuning(self, input_filename: str = "training_data.jsonl",
                               output_filename: str = "training_data_formatted.jsonl") -> Path:
        samples = self.load_existing_samples(input_filename)
        if not samples:
            logger.warning("无训练数据可导出")
            return self.output_dir / output_filename

        formatted = []
        for s in samples:
            formatted.append(self._format_for_training(s))

        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            for item in formatted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"微调格式导出: {len(formatted)} 条 -> {output_path}")
        return output_path

    def _format_for_training(self, sample: dict) -> dict:
        return format_training_sample(sample)

    def _validate_sample(self, sample: dict) -> bool:
        if not isinstance(sample, dict):
            return False
        if "input" not in sample or "output" not in sample:
            return False
        if not isinstance(sample["input"], dict) or not sample["input"]:
            return False
        if not isinstance(sample["output"], str) or not sample["output"].strip():
            return False
        if len(sample["output"].strip()) < 1:
            return False
        return True

    def _deduplicate(self, samples: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for s in samples:
            key = json.dumps(s.get("input", {}), ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result

