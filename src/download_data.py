"""
Downloads the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

Requires a Kaggle API token at ~/.kaggle/kaggle.json.
See: https://www.kaggle.com/docs/api
"""

import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading '{DATASET_SLUG}' from Kaggle...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(DATA_DIR),
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print("Download failed. Common causes:")
        print("  - kaggle.json not found at ~/.kaggle/kaggle.json")
        print("  - kaggle.json permissions too open (run: chmod 600 ~/.kaggle/kaggle.json)")
        print(f"\nKaggle CLI output:\n{result.stderr}")
        return

    zip_path = DATA_DIR / "chest-xray-pneumonia.zip"
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)

    extracted = DATA_DIR / "chest_xray"
    if extracted.exists():
        print(f"Done. Data available at: {extracted}")
    else:
        print(f"Extraction finished — check the contents of {DATA_DIR} manually.")


if __name__ == "__main__":
    main()
