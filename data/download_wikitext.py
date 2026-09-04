"""
Downloads the official Wikitext-2 dataset (train and validation splits).
"""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "wikitext-2")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "train.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
}

def download_data():
    for fname, url in URLS.items():
        out_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            print(f"Saved {fname} ({os.path.getsize(out_path):,} bytes).")
        else:
            print(f"{fname} already exists.")

if __name__ == "__main__":
    download_data()
