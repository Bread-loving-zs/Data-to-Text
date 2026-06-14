import re
from pathlib import Path
from collections import Counter, defaultdict
from src.data.utils import load_jsonl, save_jsonl

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training_data"


def is_empty_table_alpaca(sample):
    input_text = sample.get("input", "")
    if "### tables\n\n- table_count: 0" in input_text:
        return True
    if "### tables\n" in input_text and "- table_count: 0" in input_text:
        return True
    tables_match = re.search(r'### tables\s*\n(.*?)(?=\n###|\Z)', input_text, re.DOTALL)
    if tables_match:
        tables_content = tables_match.group(1).strip()
        if not tables_content or tables_content == "- table_count: 0":
            return True
    return False


def is_empty_table_raw(sample):
    inp = sample.get("input", {})
    rows = inp.get("rows", [])
    row_count = inp.get("row_count", 0)
    if row_count == 0 or (isinstance(rows, list) and len(rows) == 0):
        return True
    if isinstance(rows, list) and len(rows) <= 1:
        if len(rows) == 0:
            return True
        if len(rows) == 1 and isinstance(rows[0], dict):
            vals = [str(v).strip() for v in rows[0].values()]
            if all(v == "" for v in vals):
                return True
            first_vals = [v for v in vals if v]
            if len(first_vals) <= 2 and any(v in first_vals for v in ["2022年", "2023年", "2024年", "2025年"]):
                return True
    return False


def is_trivial_output(output):
    stripped = output.strip()
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return True
    non_table_lines = [l for l in lines if not l.startswith("表") and not l.startswith("附表")]
    if not non_table_lines:
        return True
    trivial_phrases = ["未发现风险", "无数据", "暂无数据"]
    all_trivial = True
    for l in non_table_lines:
        is_trivial_line = False
        for p in trivial_phrases:
            if p in l:
                is_trivial_line = True
                break
        if not is_trivial_line and len(l) > 15:
            all_trivial = False
            break
    if all_trivial and len(non_table_lines) <= 5:
        return True
    return False


def fix_text_adhesion(output):
    fixed = output
    fixed = fixed.replace(';', '；')
    fixed = re.sub(r'；\s*；', '；', fixed)
    fixed = re.sub(r'；。', '。', fixed)
    fixed = re.sub(r'；\s*详见', '。详见', fixed)

    severity_pattern = re.compile(
        r'(极度严重|比较严重|比较轻微|比较轻|严重|轻微)'
        r'([^；。\n\s和与及])'
    )
    fixed = severity_pattern.sub(r'\1；\2', fixed)

    fixed = re.sub(r'；\s*；', '；', fixed)

    return fixed


def extract_year_from_alpaca(sample):
    input_text = sample.get("input", "")
    match = re.search(r'年份：(\d{4})', input_text)
    if match:
        return int(match.group(1))
    return None


def extract_year_from_raw(sample):
    inp = sample.get("input", {})
    year = inp.get("year")
    if year:
        return int(year)
    return None


def extract_section_from_alpaca(sample):
    input_text = sample.get("input", "")
    match = re.search(r'分析章节：(.+)', input_text)
    if match:
        return match.group(1).strip()
    return None


def extract_section_from_raw(sample):
    inp = sample.get("input", {})
    return inp.get("section_key") or inp.get("section", "")


def balance_by_year(samples, extract_year_fn, max_ratio=3.0):
    year_counts = Counter()
    for s in samples:
        y = extract_year_fn(s)
        if y:
            year_counts[y] += 1

    if not year_counts or len(year_counts) <= 1:
        return samples

    print(f"  年份分布: {dict(sorted(year_counts.items()))}")

    min_count = min(year_counts.values())
    max_count = max(year_counts.values())

    if max_count <= min_count * max_ratio:
        return samples

    year_samples = defaultdict(list)
    no_year = []
    for s in samples:
        y = extract_year_fn(s)
        if y:
            year_samples[y].append(s)
        else:
            no_year.append(s)

    target_per_year = min(max_count, int(min_count * max_ratio))

    balanced = []
    for year, year_list in sorted(year_samples.items()):
        if len(year_list) > target_per_year:
            section_counter = Counter()
            for s in year_list:
                section = extract_section_from_alpaca(s) if extract_year_fn == extract_year_from_alpaca else extract_section_from_raw(s)
                section_counter[section or "unknown"] += 1

            num_sections = len(section_counter)
            per_section_target = max(target_per_year // num_sections, 1)

            section_selected = defaultdict(list)
            for s in year_list:
                section = extract_section_from_alpaca(s) if extract_year_fn == extract_year_from_alpaca else extract_section_from_raw(s)
                section_key = section or "unknown"
                if len(section_selected[section_key]) < per_section_target:
                    section_selected[section_key].append(s)

            selected = []
            for sec_list in section_selected.values():
                selected.extend(sec_list)

            if len(selected) < target_per_year:
                selected_set = set(id(s) for s in selected)
                remaining = [s for s in year_list if id(s) not in selected_set]
                selected.extend(remaining[:target_per_year - len(selected)])

            balanced.extend(selected[:target_per_year])
        else:
            balanced.extend(year_list)

    balanced.extend(no_year)
    return balanced


def deduplicate_by_output(samples):
    output_map = defaultdict(list)
    for s in samples:
        output = s.get("output", "").strip()
        output_map[output].append(s)

    kept = []
    removed = 0
    for output, group in output_map.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            best = max(group, key=lambda s: len(s.get("input", "")))
            kept.append(best)
            removed += len(group) - 1

    return kept, removed


def optimize_alpaca_data():
    filepath = TRAINING_DIR / "training_data_alpaca.jsonl"
    if not filepath.exists():
        print(f"文件不存在: {filepath}")
        return

    samples = load_jsonl(filepath)
    print(f"\n=== 优化 Alpaca 格式训练数据 ===")
    print(f"原始样本数: {len(samples)}")

    stats = {"original": len(samples)}

    empty_count = 0
    trivial_count = 0
    filtered = []
    for s in samples:
        if is_empty_table_alpaca(s):
            empty_count += 1
            continue
        if is_trivial_output(s.get("output", "")):
            trivial_count += 1
            continue
        filtered.append(s)

    print(f"过滤空表样本: {empty_count} 条")
    print(f"过滤无实质输出样本: {trivial_count} 条")
    stats["empty_table"] = empty_count
    stats["trivial_output"] = trivial_count

    adhesion_fixed = 0
    for s in filtered:
        original_output = s.get("output", "")
        fixed_output = fix_text_adhesion(original_output)
        if original_output != fixed_output:
            s["output"] = fixed_output
            adhesion_fixed += 1

    print(f"修复文本粘连: {adhesion_fixed} 条")
    stats["adhesion_fixed"] = adhesion_fixed

    deduped, dup_removed = deduplicate_by_output(filtered)
    print(f"输出去重: {len(filtered)} -> {len(deduped)} (移除 {dup_removed} 条)")
    stats["after_dedup"] = len(deduped)

    output_path = TRAINING_DIR / "training_data_alpaca_optimized.jsonl"
    save_jsonl(deduped, output_path)
    print(f"优化后样本数: {len(deduped)}")
    print(f"已保存到: {output_path}")

    year_counts = Counter()
    section_counts = Counter()
    for s in deduped:
        y = extract_year_from_alpaca(s)
        if y:
            year_counts[y] += 1
        sec = extract_section_from_alpaca(s)
        if sec:
            section_counts[sec] += 1

    print(f"\n优化后年份分布: {dict(sorted(year_counts.items()))}")
    print(f"优化后章节分布:")
    for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec}: {cnt}")

    return stats


def optimize_raw_data():
    filepath = TRAINING_DIR / "training_data.jsonl"
    if not filepath.exists():
        print(f"文件不存在: {filepath}")
        return

    samples = load_jsonl(filepath)
    print(f"\n=== 优化原始格式训练数据 ===")
    print(f"原始样本数: {len(samples)}")

    stats = {"original": len(samples)}

    empty_count = 0
    trivial_count = 0
    filtered = []
    for s in samples:
        if is_empty_table_raw(s):
            empty_count += 1
            continue
        if is_trivial_output(s.get("output", "")):
            trivial_count += 1
            continue
        filtered.append(s)

    print(f"过滤空表样本: {empty_count} 条")
    print(f"过滤无实质输出样本: {trivial_count} 条")
    stats["empty_table"] = empty_count
    stats["trivial_output"] = trivial_count

    deduped, dup_removed = deduplicate_by_output(filtered)
    print(f"输出去重: {len(filtered)} -> {len(deduped)} (移除 {dup_removed} 条)")
    stats["after_dedup"] = len(deduped)

    output_path = TRAINING_DIR / "training_data_optimized.jsonl"
    save_jsonl(deduped, output_path)
    print(f"优化后样本数: {len(deduped)}")
    print(f"已保存到: {output_path}")

    year_counts = Counter()
    for s in deduped:
        y = extract_year_from_raw(s)
        if y:
            year_counts[y] += 1
    print(f"优化后年份分布: {dict(sorted(year_counts.items()))}")

    return stats


def regenerate_formatted_data():
    raw_path = TRAINING_DIR / "training_data_optimized.jsonl"
    if not raw_path.exists():
        print(f"优化后的原始数据不存在，跳过格式化数据生成")
        return

    samples = load_jsonl(raw_path)
    print(f"\n=== 重新生成 Messages 格式训练数据 ===")
    print(f"输入样本数: {len(samples)}")

    from src.training.shared import format_training_sample

    formatted = []
    for s in samples:
        try:
            fmt = format_training_sample(s)
            formatted.append(fmt)
        except Exception as e:
            print(f"  格式化失败: {e}")

    output_path = TRAINING_DIR / "training_data_formatted_optimized.jsonl"
    save_jsonl(formatted, output_path)
    print(f"已生成 {len(formatted)} 条 Messages 格式样本")
    print(f"已保存到: {output_path}")


def regenerate_alpaca_from_raw():
    raw_path = TRAINING_DIR / "training_data_optimized.jsonl"
    if not raw_path.exists():
        print(f"优化后的原始数据不存在，跳过 Alpaca 格式生成")
        return

    samples = load_jsonl(raw_path)
    print(f"\n=== 从优化后的原始数据重新生成 Alpaca 格式 ===")
    print(f"输入样本数: {len(samples)}")

    from src.training.shared import format_alpaca_sample

    formatted = []
    for s in samples:
        try:
            fmt = format_alpaca_sample(s)
            formatted.append(fmt)
        except Exception as e:
            print(f"  格式化失败: {e}")

    output_path = TRAINING_DIR / "training_data_alpaca_from_raw_optimized.jsonl"
    save_jsonl(formatted, output_path)
    print(f"已生成 {len(formatted)} 条 Alpaca 格式样本")
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    alpaca_stats = optimize_alpaca_data()
    raw_stats = optimize_raw_data()

    try:
        regenerate_formatted_data()
    except Exception as e:
        print(f"生成 Messages 格式数据失败: {e}")

    try:
        regenerate_alpaca_from_raw()
    except Exception as e:
        print(f"从原始数据生成 Alpaca 格式失败: {e}")

    print("\n=== 优化总结 ===")
    if alpaca_stats:
        print(f"Alpaca 数据: {alpaca_stats['original']} -> {alpaca_stats['after_dedup']}")
        print(f"  - 空表过滤: -{alpaca_stats.get('empty_table', 0)}")
        print(f"  - 无实质输出过滤: -{alpaca_stats.get('trivial_output', 0)}")
        print(f"  - 文本粘连修复: {alpaca_stats.get('adhesion_fixed', 0)}")
        print(f"  - 输出去重: -{alpaca_stats['original'] - alpaca_stats.get('empty_table', 0) - alpaca_stats.get('trivial_output', 0) - alpaca_stats['after_dedup']}")
    if raw_stats:
        print(f"原始数据: {raw_stats['original']} -> {raw_stats['after_dedup']}")
