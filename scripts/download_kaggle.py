import sys
import os
import shutil
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR, RAW_TWCS_PATH

def download_and_setup_dataset():
    import kagglehub
    print("Downloading 'thoughtvector/customer-support-on-twitter' via kagglehub...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print(f"Dataset downloaded to cache at: {path}")

    downloaded_dir = Path(path)
    # Find twcs.csv or .zip inside the downloaded dir
    csv_files = list(downloaded_dir.glob("**/*.csv"))
    print(f"Found CSV files: {csv_files}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {downloaded_dir}")

    src_csv = csv_files[0]
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {src_csv} to {RAW_TWCS_PATH}...")
    shutil.copy2(src_csv, RAW_TWCS_PATH)
    print(f"Successfully placed {RAW_TWCS_PATH} (Size: {RAW_TWCS_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_and_setup_dataset()
