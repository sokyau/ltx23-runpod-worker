import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
import runpod
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
SPACE_SRC = APP_DIR / "space_src"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SPACE_SRC))

import app as space_app  # noqa: E402


def _download_file(url: str, suffix: str = "") -> Path:
    parsed = urlparse(url)
    suffix = suffix or Path(parsed.path).suffix or ".bin"
    target = Path(tempfile.mktemp(suffix=suffix))
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def _load_image(value: str | None) -> Image.Image | None:
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        path = _download_file(value, ".jpg")
    else:
        path = Path(value)
    return Image.open(path).convert("RGB")


def _maybe_download(value: str | None, suffix: str) -> str | None:
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return str(_download_file(value, suffix))
    return value


def _maybe_upload(path: Path, key_prefix: str) -> dict:
    bucket = os.getenv("S3_BUCKET")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    access_key = os.getenv("S3_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
    if not (bucket and access_key and secret_key):
        return {"url": "", "path": str(path)}

    region = os.getenv("S3_REGION", "auto")
    key = f"{key_prefix.rstrip('/')}/{path.name}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    public_base = os.getenv("S3_PUBLIC_BASE_URL", "").rstrip("/")
    url = f"{public_base}/{key}" if public_base else ""
    return {"url": url, "path": f"s3://{bucket}/{key}"}


def handler(job):
    data = job.get("input") or {}
    job_id = job.get("id") or str(uuid.uuid4())

    first_image = _load_image(data.get("first_image_url") or data.get("input_image_url"))
    last_image = _load_image(data.get("last_image_url"))
    input_audio = _maybe_download(data.get("input_audio_url"), ".wav")

    if first_image is None and last_image is None:
        return {"status": "failed", "error": "first_image_url or last_image_url is required"}

    prompt = data.get("prompt") or "Make this image come alive with cinematic motion, smooth animation"
    duration = float(data.get("duration", data.get("duration_seconds", 3)))
    enhance_prompt = bool(data.get("enhance_prompt", False))
    seed = int(data.get("seed", 1007968632))
    height = int(data.get("height", 704))
    width = int(data.get("width", 1280))

    video_path, used_seed = space_app.generate_video(
        first_image=first_image,
        last_image=last_image,
        input_audio=input_audio,
        prompt=prompt,
        duration=duration,
        enhance_prompt=enhance_prompt,
        seed=seed,
        randomize_seed=False,
        height=height,
        width=width,
    )

    if not video_path:
        return {"status": "failed", "error": "generation returned no video_path", "seed": used_seed}

    output_path = OUTPUT_DIR / f"ltx23-{job_id}.mp4"
    shutil.copyfile(video_path, output_path)
    persisted = _maybe_upload(output_path, data.get("output_key_prefix", "formula-faith/runpod/ltx23"))

    return {
        "status": "generated",
        "provider": "runpod",
        "provider_model": "ltx-2.3-first-last-frame-hf-port",
        "output": {
            "type": "video/mp4",
            "url": persisted["url"],
            "path": persisted["path"],
        },
        "seed": used_seed,
        "duration_seconds": duration,
        "prompt": prompt,
    }


runpod.serverless.start({"handler": handler})
