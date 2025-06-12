import os
from huggingface_hub import login, snapshot_download, HfFolder
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def download_dataset(repo_id, save_dir, token=None):
    print(f"\nStarting download for: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=save_dir,
            local_dir=os.path.join(save_dir, repo_id.replace("/", "_")),
            token=token,
        )
        print(f"Successfully downloaded or updated: {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")


def NetTurbo():
    import subprocess
    import os

    result = subprocess.run(
        'bash -c "source /etc/network_turbo && env | grep proxy"',
        shell=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    for line in output.splitlines():
        if "=" in line:
            var, value = line.split("=", 1)
            os.environ[var] = value


if __name__ == "__main__":
    NetTurbo()

    # "https://hf-mirror.com"
    # 'hf-cdn.sufy.com'
    os.environ["HF_ENDPOINT"] = "hf-cdn.sufy.com"

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    HfFolder.save_token(hf_token)

    try:
        login(token=hf_token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(
            f"Hugging Face login failed: {e}. Downloads might rely on cached credentials or public access."
        )

    base_save_dir = "/root/autodl-tmp/HuggingFace_Datasets"
    os.makedirs(base_save_dir, exist_ok=True)

    repo_ids = ["ItzLoghotXD/Ghibli"]

    unique_repo_ids = sorted(list(set(repo_ids)))

    print(f"Found {len(unique_repo_ids)} unique datasets to download.")

    for repo_id in tqdm(unique_repo_ids, desc="Downloading datasets"):

        specific_dataset_dir = os.path.join(base_save_dir, repo_id.replace("/", "--"))

        download_dataset(repo_id, save_dir=specific_dataset_dir, token=hf_token)

    print("\nAll dataset download tasks complete.")
