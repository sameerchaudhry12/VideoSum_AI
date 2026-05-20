# download_datasets.py
# Run this once to download FER-2013, RAVDESS, and MELD datasets
# Usage: python download_datasets.py

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FER2013_DIR, RAVDESS_DIR, MELD_DIR

# ── Helper ────────────────────────────────────────────────────────────────────

def extract_zip(zip_path, extract_to):
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Done. Removed zip file.")


# ── 1. FER-2013 (via Kaggle API) ──────────────────────────────────────────────

def download_fer2013():
    print("\n[1/3] Downloading FER-2013 from Kaggle...")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("  ERROR: kaggle.json not found at ~/.kaggle/kaggle.json")
        print("  Go to kaggle.com > Settings > API > Create New Token")
        return False

    zip_path = os.path.join(FER2013_DIR, "fer2013.zip")

    os.system(
        f'kaggle datasets download -d msambare/fer2013 -p "{FER2013_DIR}"'
    )

    # The dataset downloads as fer2013.zip
    downloaded = os.path.join(FER2013_DIR, "fer2013.zip")
    if os.path.exists(downloaded):
        extract_zip(downloaded, FER2013_DIR)
        print(f"  FER-2013 ready at: {FER2013_DIR}")
        return True
    else:
        print("  ERROR: Download failed. Check your kaggle.json and internet connection.")
        return False


# ── 2. RAVDESS (direct download from Zenodo) ──────────────────────────────────

def download_ravdess():
    print("\n[2/3] Downloading RAVDESS from Zenodo...")
    print("  This is ~600MB — will take a few minutes.")

    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = os.path.join(RAVDESS_DIR, "ravdess.zip")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        print(f"\r  [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress)
        print()
        extract_zip(zip_path, RAVDESS_DIR)
        print(f"  RAVDESS ready at: {RAVDESS_DIR}")
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("  Try downloading manually from: https://zenodo.org/record/1188976")
        print(f"  Extract to: {RAVDESS_DIR}")
        return False


# ── 3. MELD (from GitHub releases) ───────────────────────────────────────────

def download_meld():
    print("\n[3/3] Downloading MELD dataset...")
    print("  Downloading CSV files only (we only need text + labels, not video).")

    base = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
    files = {
        "train_sent_emo.csv": f"{base}/train_sent_emo.csv",
        "dev_sent_emo.csv":   f"{base}/dev_sent_emo.csv",
        "test_sent_emo.csv":  f"{base}/test_sent_emo.csv",
    }

    all_ok = True
    for filename, url in files.items():
        dest = os.path.join(MELD_DIR, filename)
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            print(f"OK ({size / 1024:.1f} KB)")
        except Exception as e:
            print(f"FAILED — {e}")
            all_ok = False

    if all_ok:
        print(f"  MELD ready at: {MELD_DIR}")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Downloader — Emotion Video Summarization")
    print("=" * 60)

    results = {
        "FER-2013": download_fer2013(),
        "RAVDESS":  download_ravdess(),
        "MELD":     download_meld(),
    }

    print("\n" + "=" * 60)
    print("  Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"    {name}: {status}")

    if all(results.values()):
        print("\n  All datasets downloaded. Run preprocess.py next.")
    else:
        print("\n  Some downloads failed. Fix errors above and re-run.")
    print("=" * 60)