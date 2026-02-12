#!/usr/bin/env python3
import html
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
TEMPLATE_PATH = ROOT_DIR / "index.html"
RUNS_DIR = PROJECT_ROOT / "webapp_runs"
INFER_SCRIPT = PROJECT_ROOT / "scripts" / "inference.py"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pth"

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "3600"))
ALLOWED_SUFFIX = {".tif", ".tiff", ".png"}

app = FastAPI(title="U-Net++ Inference App", version="1.0.0")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _render_index(message: str = "", raw_html: bool = False) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if not message:
        block = ""
    elif raw_html:
        block = message
    else:
        block = f"<p><b>{html.escape(message)}</b></p>"
    return template.replace("{{MESSAGE_BLOCK}}", block)


def _zip_dir(source_dir: Path, zip_path: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(source_dir))
                count += 1
    return count


def _run_inference(input_path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["FORCE_CPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["IMG_PATH"] = str(input_path)
    env["OUT_DIR"] = str(out_dir)
    if "CHECKPOINT_PATH" not in env and DEFAULT_MODEL_PATH.exists():
        env["CHECKPOINT_PATH"] = str(DEFAULT_MODEL_PATH)

    cmd = [sys.executable, str(INFER_SCRIPT)]
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=INFERENCE_TIMEOUT,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/infer", response_class=HTMLResponse)
def infer(image: UploadFile = File(...)):
    if not image.filename:
        return HTMLResponse(_render_index("请上传一个影像文件。"), status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        return HTMLResponse(_render_index("仅支持 .tif/.tiff/.png 文件。"), status_code=400)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    in_dir = run_dir / "input"
    out_dir = run_dir / "infer"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = in_dir / f"uploaded{suffix}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    if input_path.stat().st_size <= 0:
        return HTMLResponse(_render_index("上传内容为空。"), status_code=400)
    if input_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        return HTMLResponse(
            _render_index(f"文件过大，超过 {MAX_UPLOAD_MB}MB 限制。"),
            status_code=413,
        )

    try:
        result = _run_inference(input_path, out_dir)
    except subprocess.TimeoutExpired:
        return HTMLResponse(_render_index("推理超时。"), status_code=504)
    except Exception as e:
        return HTMLResponse(
            _render_index(f"推理进程异常：{e}"),
            status_code=500,
        )

    if result.returncode != 0:
        err_text = result.stderr or result.stdout or "无错误输出"
        err_block = (
            "<p><b>推理失败。</b></p>"
            f"<pre>{html.escape(err_text[-5000:])}</pre>"
        )
        return HTMLResponse(_render_index(err_block, raw_html=True), status_code=500)

    zip_path = run_dir / "result.zip"
    file_count = _zip_dir(out_dir, zip_path)
    if file_count == 0:
        return HTMLResponse(_render_index("推理完成，但未生成可下载文件。"), status_code=500)

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"infer_result_{run_id}.zip",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, reload=False)
