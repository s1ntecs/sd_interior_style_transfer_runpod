import os
from typing import Dict
from urllib.parse import urlparse
from PIL import Image

from runpod.http_client import SyncClientSession

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}


def extract_disposition_params(content_disposition: str) -> Dict[str, str]:
    parts = (p.strip() for p in content_disposition.split(";"))

    params = {
        key.strip().lower(): value.strip().strip('"')
        for part in parts
        if "=" in part
        for key, value in [part.split("=", 1)]
    }

    return params


def file(file_url: str,
         file_name: str,
         directory_path: str) -> dict:
    """
    Downloads a single file from a given URL, file is given a file_name.
    """
    os.makedirs(directory_path, exist_ok=True)

    download_response = SyncClientSession().get(file_url,
                                                headers=HEADERS,
                                                timeout=30)
    download_response.raise_for_status()

    content_disposition = download_response.headers.get("Content-Disposition")

    original_file_name = ""
    if content_disposition:
        params = extract_disposition_params(content_disposition)

        original_file_name = params.get("filename", "")

    if not original_file_name:
        download_path = urlparse(file_url).path
        original_file_name = os.path.basename(download_path)

    file_type = os.path.splitext(original_file_name)[1].replace(".", "")

    output_file_path = os.path.join(directory_path, f"{file_name}.{file_type}")
    with open(output_file_path, "wb") as output_file:
        output_file.write(download_response.content)
    return {
        "file_path": os.path.abspath(output_file_path),
        "type": file_type,
        "original_name": original_file_name,
    }
