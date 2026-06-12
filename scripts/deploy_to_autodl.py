import base64
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES_TO_DEPLOY = [
    ("src/training/shared.py", "/root/autodl-tmp/data-to-text/src/training/shared.py"),
    ("src/training/test_inference_cloud.py", "/root/autodl-tmp/data-to-text/src/training/test_inference_cloud.py"),
]



def generate_commands():
    lines = []
    lines.append("=" * 60)
    lines.append("复制以下命令到 AutoDL 终端执行")
    lines.append("=" * 60)
    lines.append("")

    for local_path, remote_path in FILES_TO_DEPLOY:
        local_file = ROOT / local_path
        data = local_file.read_bytes()
        b64 = base64.b64encode(data).decode()
        filename = local_file.name

        lines.append(f"# --- 部署 {filename} ---")
        lines.append(f"cat << 'DEPLOY_EOF' | python3")
        lines.append(f"import base64, pathlib")
        lines.append(f"pathlib.Path('{remote_path}').parent.mkdir(parents=True, exist_ok=True)")
        lines.append(f"b64 = '{b64}'")
        lines.append(f"data = base64.b64decode(b64)")
        lines.append(f"pathlib.Path('{remote_path}').write_bytes(data)")
        lines.append(f"print('{filename} deployed (' + str(len(data)) + ' bytes)')")
        lines.append(f"DEPLOY_EOF")
        lines.append("")

    lines.append("# --- 运行推理对比测试 ---")
    lines.append("cd /root/autodl-tmp/data-to-text")
    lines.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
    lines.append("  python src/training/test_inference_cloud.py --num-samples 5")

    return "\n".join(lines)


if __name__ == "__main__":
    output = generate_commands()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(output, encoding="utf-8")
        print(f"部署命令已写入: {sys.argv[1]}")
    else:
        print(output)
