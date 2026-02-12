#!/usr/bin/env python3
import cgi
import html
import os
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "webapp_runs"
INFER_SCRIPT = ROOT_DIR / "scripts" / "inference.py"
DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "model.pth"

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
ALLOWED_SUFFIX = {".tif", ".tiff", ".png"}


def _render_index(message: str = "") -> str:
    msg_html = f"<p><b>{html.escape(message)}</b></p>" if message else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>遥感分割推理 WebApp</title>
  <style>
    body {{ font-family: sans-serif; max-width: 860px; margin: 32px auto; line-height: 1.5; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
    code {{ background: #f6f6f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>U-Net++ 单图推理（CPU）</h2>
  <div class="card">
    <p>上传单张影像（支持 .tif/.tiff/.png），服务会调用 <code>scripts/inference.py</code> 并强制 CPU 推理。</p>
    {msg_html}
    <form action="/infer" method="post" enctype="multipart/form-data">
      <input type="file" name="image" required />
      <button type="submit">开始推理并下载结果</button>
    </form>
    <p>结果将打包为 zip，包含推理产物（如 <code>pred_class.tif</code> 与矢量文件）。</p>
  </div>
</body>
</html>"""


def _zip_dir(source_dir: Path, zip_path: Path) -> int:
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(source_dir))
                file_count += 1
    return file_count


def _run_inference(input_path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["FORCE_CPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["IMG_PATH"] = str(input_path)
    env["OUT_DIR"] = str(out_dir)
    if "CHECKPOINT_PATH" not in env and DEFAULT_MODEL_PATH.exists():
        env["CHECKPOINT_PATH"] = str(DEFAULT_MODEL_PATH)

    cmd = ["python3", str(INFER_SCRIPT)]
    return subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )


class AppHandler(BaseHTTPRequestHandler):
    def _send_html(self, body: str, status: int = 200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download(self, file_path: Path, filename: str):
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(_render_index())
            return
        self._send_html(_render_index("未找到页面。"), status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/infer":
            self._send_html(_render_index("无效请求路径。"), status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_html(_render_index("上传内容为空。"), status=400)
            return
        if length > MAX_UPLOAD_MB * 1024 * 1024:
            self._send_html(
                _render_index(f"文件过大，超过 {MAX_UPLOAD_MB}MB 限制。"),
                status=413,
            )
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self._send_html(_render_index("请上传一个影像文件。"), status=400)
            return

        suffix = Path(file_item.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIX:
            self._send_html(
                _render_index("仅支持 .tif/.tiff/.png 文件。"),
                status=400,
            )
            return

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = RUNS_DIR / run_id
        in_dir = run_dir / "input"
        out_dir = run_dir / "infer"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = in_dir / f"uploaded{suffix}"
        with input_path.open("wb") as f:
            shutil.copyfileobj(file_item.file, f)

        try:
            result = _run_inference(input_path, out_dir)
        except subprocess.TimeoutExpired:
            self._send_html(_render_index("推理超时（>3600秒）。"), status=504)
            return

        if result.returncode != 0:
            stderr = html.escape(result.stderr[-4000:]) if result.stderr else "无错误输出"
            stdout = html.escape(result.stdout[-4000:]) if result.stdout else "无标准输出"
            self._send_html(
                _render_index(
                    f"推理失败。<br/><pre>STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}</pre>"
                ),
                status=500,
            )
            return

        zip_path = run_dir / "result.zip"
        file_count = _zip_dir(out_dir, zip_path)
        if file_count == 0:
            self._send_html(_render_index("推理完成，但未生成可下载文件。"), status=500)
            return

        self._send_download(zip_path, f"infer_result_{run_id}.zip")

    def log_message(self, fmt, *args):
        return


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = HTTPServer((HOST, PORT), AppHandler)
    print(f"WebApp running at http://{HOST}:{PORT}")
    print("Upload one image, run CPU inference, then download zipped outputs.")
    server.serve_forever()


if __name__ == "__main__":
    main()
