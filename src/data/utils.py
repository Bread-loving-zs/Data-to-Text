import json
from pathlib import Path

from src.config import setup_logging

logger = setup_logging(__name__)


def load_jsonl(filepath: Path | str, skip_errors: bool = True) -> list[dict]:
    filepath = Path(filepath)
    results: list[dict] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_errors:
                    logger.warning("跳过无效行 %s: %s (%s)", filepath, line_no, e)
                else:
                    raise
    return results


def save_jsonl(samples: list[dict], filepath: Path | str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
