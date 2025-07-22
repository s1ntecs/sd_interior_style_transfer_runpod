import io
import os
from typing import Dict, Tuple
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError
from runpod.http_client import SyncClientSession

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}


def extract_disposition_params(content_disposition: str) -> Dict[str, str]:
    parts = (p.strip() for p in content_disposition.split(";"))
    return {
        k.strip().lower(): v.strip().strip('"')
        for part in parts if "=" in part
        for k, v in [part.split("=", 1)]
    }


def _guess_original_name(url: str, content_disposition: str | None) -> str:
    if content_disposition:
        params = extract_disposition_params(content_disposition)
        if params.get("filename"):
            return params["filename"]

    return os.path.basename(urlparse(url).path) or "download"


def _save_image_as_png(raw_bytes: bytes, out_path_png: str) -> Tuple[bool, str]:
    """
    Пытается открыть raw_bytes как изображение и сохранить в PNG.
    Возвращает (успех, сообщение_или_путь).
    """
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            # Конверт в RGB, если у изображения палитра/альфа и т.д.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            img.save(out_path_png, format="PNG")
        return True, out_path_png
    except UnidentifiedImageError:
        return False, "Not an image or unsupported image format for PIL"


def file(file_url: str,
         file_name: str,
         directory_path: str) -> dict:
    """
    Скачивает файл по URL.
    Если это изображение — сохраняет его под именем file_name с расширением .png.
    Иначе сохраняет как есть (с оригинальным расширением).
    Возвращает словарь с путями и типом.
    """
    os.makedirs(directory_path, exist_ok=True)

    resp = SyncClientSession().get(file_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    original_name = _guess_original_name(file_url, resp.headers.get("Content-Disposition"))
    raw_bytes = resp.content

    # Путь, куда положим итоговый PNG
    png_path = os.path.join(directory_path, f"{file_name}.png")

    is_image, result = _save_image_as_png(raw_bytes, png_path)
    if is_image:
        return {
            "file_path": os.path.abspath(result),
            "type": "png",
            "original_name": original_name
        }

    # не изображение — сохраняем как есть
    # достаём расширение (если нет — без него)
    _, ext = os.path.splitext(original_name)
    out_path = os.path.join(directory_path, f"{file_name}{ext}")
    with open(out_path, "wb") as f:
        f.write(raw_bytes)

    return {
        "file_path": os.path.abspath(out_path),
        "type": ext.lstrip(".").lower(),
        "original_name": original_name
    }
