import io
import os
from typing import Dict, Tuple
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError
from runpod.http_client import SyncClientSession

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}


def file(file_url: str, file_name: str, directory_path: str) -> None:
    """
    Скачивает файл в directory_path под именем file_name.
    Если это картинка — принудительно сохраняем как PNG (имя уже должно быть *.png).
    Ничего не возвращает (как в твоём коде).
    """
    os.makedirs(directory_path, exist_ok=True)
    out_path = os.path.join(directory_path, file_name)

    resp = SyncClientSession().get(file_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.content

    # Попробуем как изображение
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            im.save(out_path, format="PNG")
        return
    except UnidentifiedImageError:
        # не изображение — сохраняем как есть
        pass

    # Если хотели png, но это не картинка — всё равно просто пишем байты
    with open(out_path, "wb") as f:
        f.write(raw)
