import os
import requests
from tqdm import tqdm
from pathlib import Path


def download_with_progress(url, destination):
    """Download file with tqdm progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_arcface():
    """Download ArcFace - the best face recognition model"""
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5"
    destination = os.path.expanduser("~/.deepface/weights/arcface_weights.h5")

    if os.path.exists(destination):
        file_size = os.path.getsize(destination) / (1024 * 1024)
        print(f"✓ ArcFace already downloaded ({file_size:.1f} MB)")
        print(f"  Location: {destination}")
        return

    print("=== ArcFace Model Downloader ===")
    print("Model: ArcFace (state-of-the-art)")
    print("Accuracy: 99.83% on LFW benchmark")
    print("Size: ~130 MB\n")

    download_with_progress(url, destination)

    print(f"\n✓ Download complete!")
    print(f"  Location: {destination}")
    print(f"\nNext step: Update your script to use MODEL_NAME = 'ArcFace'")


if __name__ == "__main__":
    download_arcface()
