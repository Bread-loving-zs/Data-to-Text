import base64
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES_TO_DEPLOY = [
    ("src/training/shared.py", "/root/autodl-tmp/data-to-text/src/training/shared.py"),
    ("src/training/test_inference_cloud.py", "/root/autodl-tmp/data-to-text/src/training/test_inference_cloud.py"),
]

print("=" * 60)
print("复制以下命令到 AutoDL 终端执行")
print("=" * 60)
print()

for local_path, remote_path in FILES_TO_DEPLOY:
    local_file = ROOT / local_path
    data = local_file.read_bytes()
    b64 = base64.b64encode(data).decode()
    filename = local_file.name
    print(f"# --- 部署 {filename} ---")
    print(f"echo '{b64}' | base64 -d > {remote_path}")
    print()

print("# --- 运行推理对比测试 ---")
print("cd /root/autodl-tmp/data-to-text")
print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \\")
print("  python src/training/test_inference_cloud.py --num-samples 5")
