#!/usr/bin/env python3
import os
import shutil
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from helpers.comfyui import ComfyUI


@dataclass
class HFFile:
    repo_id: str          # репозиторий на HF
    filename: str         # путь к файлу в репо
    dest_path: str        # итоговый путь (с именем файла) внутри проекта


FILES = [
    HFFile(
        repo_id="frankjoshua/albedobaseXL_v21",
        filename="albedobaseXL_v21.safetensors",
        dest_path="ComfyUI/models/checkpoints/albedobaseXL_v21.safetensors"
    ),
    HFFile(
        repo_id="h94/IP-Adapter",
        filename="models/image_encoder/model.safetensors",
        dest_path="ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    ),
    HFFile(
        repo_id="h94/IP-Adapter",
        filename="sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        dest_path="ComfyUI/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"
    ),
    HFFile(
        repo_id="SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe",
        filename="depth-zoe-xl-v1.0-controlnet.safetensors",
        dest_path="ComfyUI/models/controlnet/depth-zoe-xl-v1.0-controlnet.safetensors"
    ),
]


def safe_copy(src: str, dst: str):
    """Создать каталог и скопировать файл без симлинков."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Если src == dst (hf_hub_download уже положил туда), просто пропускаем
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)


def download_sd_ckp():
    """Скачиваем всё из списка FILES через huggingface_hub."""
    for item in FILES:
        try:
            # local_dir_use_symlinks=False чтобы получить реальный файл, не линк
            local_tmp = hf_hub_download(
                repo_id=item.repo_id,
                filename=item.filename,
                local_dir="/tmp/hf_cache",
                local_dir_use_symlinks=False
            )
            safe_copy(local_tmp, item.dest_path)
            print(f"✓ {item.filename} -> {item.dest_path}")
        except Exception as e:
            print(f"✗ Ошибка при скачивании {item.filename} из {item.repo_id}: {e}")


def download_comfy():
    """Если у тебя в helpers.comfyui есть метод предзагрузки — вызываем корректно."""
    comfy = ComfyUI("127.0.0.1:8188")
    # Заметка: у тебя было `comfyUI.download_pre_start_models` без вызова.
    if hasattr(comfy, "download_pre_start_models"):
        comfy.download_pre_start_models()
    else:
        # либо просто заглушка
        pass


if __name__ == "__main__":
    download_sd_ckp()
    download_comfy()
