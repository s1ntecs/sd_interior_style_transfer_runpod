#!/usr/bin/env python3
import os

from helpers.comfyui import ComfyUI
from huggingface_hub import hf_hub_download

REPO_ID = "frankjoshua/albedobaseXL_v21"

FILES_TO_DOWNLOAD = [
    ("albedobaseXL_v21.safetensors", "ComfyUI/models/checkpoints"),
]


def download_comfy():
    comfyUI = ComfyUI("127.0.0.1:8188")
    comfyUI.download_pre_start_models


def download_sd_ckp():
    """
    Скачивает из репозитория на Hugging Face все файлы из списка FILES_TO_DOWNLOAD
    и сохраняет их в соответствующие папки внутри проекта ComfyUI.
    """
    for remote_path, local_subdir in FILES_TO_DOWNLOAD:
        # Создаем локальную папку, если её ещё нет
        os.makedirs(local_subdir, exist_ok=True)

        try:
            # Скачиваем файл
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path,
                local_dir=local_subdir,
                local_dir_use_symlinks=False
            )
            print(f"Скачан файл: {remote_path} → {local_path}")
        except Exception as e:
            print(f"Ошибка при скачивании {remote_path}: {e}")


if __name__ == "__main__":
    download_sd_ckp()
    download_comfy()
