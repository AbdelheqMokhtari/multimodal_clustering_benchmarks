#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from pyquery import PyQuery as pq


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_URL = "http://vision.cs.uiuc.edu/pascal-sentences/"
OUT_ROOT = Path("pascal/raw")
IMG_DIR = OUT_ROOT / "images"
TXT_DIR = OUT_ROOT / "texts"

VERBOSE = False  

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Pascal Sentences Downloader)"
}


# --------------------------------------------------
# Logging helper
# --------------------------------------------------
def log(msg: str):
    if VERBOSE:
        print(msg)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def fetch_page(url: str) -> pq:
    log(f"[↓] Fetching page: {url}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    log("[✓] Page fetched successfully")
    return pq(r.text)


def download_file(url: str, dst: Path):
    if dst.exists():
        log(f"[↷] Exists, skipping: {dst.relative_to(OUT_ROOT)}")
        return False

    log(f"[↓] Downloading: {dst.relative_to(OUT_ROOT)}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    dst.write_bytes(r.content)
    log(f"[✓] Saved: {dst.relative_to(OUT_ROOT)}")
    return True


# --------------------------------------------------
# Main logic
# --------------------------------------------------
def main():
    log("\n=== Pascal Sentences Downloader ===\n")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    dom = fetch_page(BASE_URL)

    total = 0
    downloaded = 0

    for row in dom("body table tr").items():
        img = row("img")
        if not img:
            continue

        img_src = img.attr("src")
        if not img_src:
            continue

        cls, img_name = os.path.split(img_src)
        if not cls:
            continue

        img_id = Path(img_name).stem

        log(f"\n[→] Processing: class={cls}, id={img_id}")

        # Output paths
        img_out_dir = IMG_DIR / cls
        txt_out_dir = TXT_DIR / cls
        img_out_dir.mkdir(exist_ok=True)
        txt_out_dir.mkdir(exist_ok=True)

        img_out = img_out_dir / img_name
        txt_out = txt_out_dir / f"{img_id}.txt"

        # Download image
        img_url = urljoin(BASE_URL, img_src)
        if download_file(img_url, img_out):
            downloaded += 1

        # Save captions
        if txt_out.exists():
            log(f"[↷] Captions exist, skipping: {txt_out.relative_to(OUT_ROOT)}")
        else:
            log(f"[✎] Writing captions: {txt_out.relative_to(OUT_ROOT)}")
            with open(txt_out, "w", encoding="utf-8") as f:
                count = 0
                for td in row("table tr td").items():
                    text = td.text().strip()
                    if text:
                        f.write(text + "\n")
                        count += 1
            log(f"[✓] Saved {count} captions")

        total += 1

    log("\n=== Finished ===")
    log(f"Total samples processed : {total}")
    log(f"Images downloaded       : {downloaded}")
    log(f"Images directory        : {IMG_DIR}")
    log(f"Texts directory         : {TXT_DIR}\n")


if __name__ == "__main__":
    main()
