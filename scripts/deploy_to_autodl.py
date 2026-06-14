import base64
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
REMOTE_BASE = "/root/autodl-tmp/data-to-text"

SRC_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/cli.py",
    "src/agent/__init__.py",
    "src/agent/llm_client.py",
    "src/agent/intent.py",
    "src/agent/query.py",
    "src/agent/context.py",
    "src/agent/generator.py",
    "src/agent/statistics.py",
    "src/agent/fact_checker.py",
    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/dictionary.py",
    "src/data/models.py",
    "src/data/utils.py",
    "src/training/__init__.py",
    "src/training/shared.py",
    "src/training/prepare_data.py",
    "src/training/evaluate.py",
    "src/report/__init__.py",
    "src/report/template.py",
    "src/report/converter.py",
    "src/report/charts.py",
]

SCRIPT_FILES = [
    "scripts/finetune_cloud.py",
    "scripts/inference_cloud.py",
]

PROMPT_FILES = [
    "prompts/intent_recognition.txt",
    "prompts/system_report.txt",
]

DATA_FILES = [
    "training_data/training_data.jsonl",
    "training_data/training_data_formatted.jsonl",
]


def _deploy_file_heredoc(local_path: str, remote_path: str) -> list[str]:
    local_file = ROOT / local_path
    if not local_file.exists():
        return [f"# WARNING: {local_path} not found, skipping"]

    data = local_file.read_bytes()
    b64 = base64.b64encode(data).decode()
    filename = local_file.name

    return [
        f"cat << 'DEPLOY_EOF' | python3",
        f"import base64, pathlib",
        f"pathlib.Path('{remote_path}').parent.mkdir(parents=True, exist_ok=True)",
        f"b64 = '{b64}'",
        f"data = base64.b64decode(b64)",
        f"pathlib.Path('{remote_path}').write_bytes(data)",
        f"print('{filename} deployed (' + str(len(data)) + ' bytes)')",
        f"DEPLOY_EOF",
    ]


def generate_commands(mode: str = "full") -> str:
    lines = []
    lines.append("#" + "=" * 59)
    lines.append("# AutoDL 部署脚本 — Data-to-Text 项目")
    lines.append("#" + "=" * 59)
    lines.append("")

    lines.append("# ====== 第一步：环境初始化 ======")
    lines.append("source /etc/network_turbo 2>/dev/null || true")
    lines.append("pip install pydantic httpx pandas openpyxl scipy -q")
    lines.append("pip install transformers peft datasets accelerate bitsandbytes -q")
    lines.append("pip uninstall torchvision -y 2>/dev/null || true")
    lines.append("")

    lines.append("# ====== 第二步：部署项目文件 ======")
    all_files = SRC_FILES + SCRIPT_FILES + PROMPT_FILES
    if mode in ("full", "inference"):
        all_files += DATA_FILES

    for local_path in all_files:
        remote_path = f"{REMOTE_BASE}/{local_path}"
        lines.append(f"# --- {local_path} ---")
        lines.extend(_deploy_file_heredoc(local_path, remote_path))
        lines.append("")

    lines.append("cd /root/autodl-tmp/data-to-text")
    lines.append("")

    if mode == "full":
        lines.append("# ====== 第三步：QLoRA 微调 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/finetune_cloud.py")
        lines.append("")

        lines.append("# ====== 第四步：推理对比测试 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/inference_cloud.py --num-samples 5")
        lines.append("")

    elif mode == "finetune":
        lines.append("# ====== 第三步：QLoRA 微调 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/finetune_cloud.py")
        lines.append("")

    elif mode == "inference":
        lines.append("# ====== 第三步：推理对比测试 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/inference_cloud.py --num-samples 5")
        lines.append("")

    elif mode == "inference-base":
        lines.append("# ====== 仅测试基础模型 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/inference_cloud.py --base-only --num-samples 5")
        lines.append("")

    elif mode == "inference-lora":
        lines.append("# ====== 仅测试微调模型 ======")
        lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
        lines.append("  python scripts/inference_cloud.py --lora-only --num-samples 5")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoDL 部署命令生成器")
    parser.add_argument("output", nargs="?", help="输出文件路径（默认输出到终端）")
    parser.add_argument("--mode", default="full",
                        choices=["full", "finetune", "inference", "inference-base", "inference-lora"],
                        help="部署模式：full=微调+推理, finetune=仅微调, inference=仅推理, inference-base=仅基础模型, inference-lora=仅微调模型")
    args = parser.parse_args()

    output = generate_commands(args.mode)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"部署命令已写入: {args.output}")
    else:
        print(output)
