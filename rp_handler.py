import base64
import io
import os
import shutil
import json
import mimetypes
import random
import time
from PIL import Image
from typing import Any, Dict, List

import numpy as np
import torch
from helpers.comfyui import ComfyUI

import runpod
from rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
MAX_SEED = np.iinfo(np.int32).max

mimetypes.add_type("image/webp", ".webp")

with open("style-transfer-api.json", "r") as file:
    STYLE_TRANSFER_WORKFLOW_JSON = file.read()

with open("style-transfer-with-structure-api.json", "r") as file:
    STYLE_TRANSFER_WITH_STRUCTURE_WORKFLOW_JSON = file.read()

logger = RunPodLogger()


class Predictor():
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(
            STYLE_TRANSFER_WORKFLOW_JSON,
            handle_inputs=False,
            handle_weights=True
        )

    def pil_to_b64(self,
                   img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: str, filename: str = "image.png"):
        image = Image.open(input_file)
        image.save(os.path.join(INPUT_DIR, filename))

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(path)
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def collect_b64_files_to_dict(
        self,
        files: List[str],
        job: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Превращает список путей к файлам в словарь формата:
        {
            "images_base64": [...],
            "time": <float | None>,
        }

        Неизображения игнорируются.
        """
        images_b64: List[str] = []

        for path in files:
            mime, _ = mimetypes.guess_type(path)
            if mime and mime.startswith("image/"):
                # Можно напрямую закодировать файл, но раз уже есть pil_to_b64 — используем его
                try:
                    with Image.open(path) as im:
                        images_b64.append(self.pil_to_b64(im))
                except Exception:
                    # fallback — вдруг PIL не смог открыть, но это всё же картинка
                    with open(path, "rb") as f:
                        images_b64.append(base64.b64encode(f.read()).decode())

        return {
            "images_base64": images_b64,
            "time": round(time.time() - job["created"], 2) if job and "created" in job else None,
        }

    def update_workflow(self, workflow, **kwargs):
        workflow["3"]["inputs"]["steps"] = kwargs['steps']
        workflow["3"]["inputs"]["cfg"] = kwargs['cfg']
        workflow["3"]["inputs"]["sampler_name"] = "dpmpp_2m_sde_gpu"

        workflow["2"]["inputs"]["ckpt_name"] = "albedobaseXL_v21.safetensors"
        workflow["6"]["inputs"]["text"] = kwargs["prompt"]
        workflow["7"]["inputs"]["text"] =\
            f"nsfw, nude, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        sampler["denoise"] = kwargs["structure_denoising_strength"]
        workflow["18"]["inputs"]["strength"] = kwargs[
            "structure_denoising_strength"
        ]
        workflow["24"]["inputs"]["amount"] = kwargs["batch_size"]

    def predict(
        self,
        job,
        style_image_url: str,
        structure_image_url: str,
        prompt: str,
        negative_prompt: str,
        number_of_images: int,
        structure_depth_strength: float,
        structure_denoising_strength: float,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        rp_file(style_image_url,
                "image.png",
                INPUT_DIR)
        rp_file(structure_image_url,
                "structure.png",
                INPUT_DIR)

        workflow = json.loads(STYLE_TRANSFER_WITH_STRUCTURE_WORKFLOW_JSON)

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            batch_size=number_of_images,
            structure_depth_strength=structure_depth_strength,
            structure_denoising_strength=structure_denoising_strength,
        )

        wf = self.comfyUI.load_workflow(workflow, handle_weights=True)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        files = self.log_and_collect_files(OUTPUT_DIR)
        result = self.collect_b64_files_to_dict(
            files,
            job=job
        )
        return result


for d in (INPUT_DIR, OUTPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

comfy_obj = Predictor()
comfy_obj.setup()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        style_image_url = payload.get("style_image_url")
        structure_image_url = payload.get("structure_image_url")
        if not style_image_url or not structure_image_url:
            return {"error": "'style_image_url' and 'structure_image_url' is required"}
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}
        negative_prompt = payload.get(
            "negative_prompt", "")
        number_of_images = payload.get(
            "number_of_images", 1)
        structure_depth_strength = payload.get(
            "structure_depth_strength", 1.0)
        structure_denoising_strength = payload.get(
            "structure_denoising_strength", 0.65)
        seed = int(payload.get(
            "seed",
            random.randint(0, MAX_SEED)))
        result = comfy_obj.predict(
            job=job,
            style_image_url=style_image_url,
            structure_image_url=structure_image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=number_of_images,
            structure_denoising_strength=structure_denoising_strength,
            structure_depth_strength=structure_depth_strength,
            seed=seed
        )
        return result
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}

# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
