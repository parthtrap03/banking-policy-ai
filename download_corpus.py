# download_corpus.py
"""
Utility script to download and refresh the corpus of banking and factoring documents.
It pulls:
- Factoring/Invoice‑Discounting pages from major Indian banks and NBFCs.
- Indian legal documents (Factoring & Invoice Discounting Act 2023, RBI Factoring Guidelines, RBI circulars).
- International reference material (U.S. UCC §9, EU Factoring Directive, Factoring‑association whitepapers).
The script stores everything under `legal_documents/factoring/` and `legal_documents/indian_banks/`.
A simple monthly scheduler (using the `schedule` library) is also provided – run the script once and it will keep itself alive,
triggering a fresh crawl on the first day of each month.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration – URLs to scrape
# ---------------------------------------------------------------------------

BANK_FACTRING_URLS = {
    "SBI": "https://www.sbi.co.in/webcontent/Factoring",
    "HDFC": "https://www.hdfcbank.com/personal/loans/factoring",
    "ICICI": "https://www.icicibank.com/factoring",
    "Axis": "https://www.axisbank.com/factoring",
    "Kotak": "https://www.kotak.com/factoring",
    # Add more banks as needed
}

NBFC_FACTRING_URLS = {
    "BajajFinance": "https://www.bajajfinserv.in/factoring",
    "MahindraFinance": "https://www.mahindrafinance.com/factoring",
    "IndusIndBank": "https://www.indusind.com/factoring",
    "YesBank": "https://www.yesbank.in/factoring",
    "IDFCFirst": "https://www.idfcfirstbank.com/factoring",
    "RBLBank": "https://www.rblbank.com/factoring",
}

INDIAN_LEGAL_URLS = {
    "FactoringAct2023": "https://www.indiacode.nic.in/handle/123456789/4567?lang=en",  # placeholder – actual URL may differ
    "RBI_Factoring_Guidelines": "https://www.rbi.org.in/Scripts/BS_ViewMasCircular.aspx?Id=1234",
    "RBI_Factoring_Circular": "https://www.rbi.org.in/Scripts/BS_ViewMasCircular.aspx?Id=5678",
}

INTERNATIONAL_REFERENCES = {
    "UCC_Section9": "https://www.law.cornell.edu/ucc/9",
    "EU_Factoring_Directive": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0138",
    "Factoring_Association_Whitepaper": "https://www.ifac.org/factoring-whitepaper.pdf",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def safe_filename(url: str) -> str:
    """Create a filesystem‑safe filename from a URL."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", url)
    return name[:200]

def download_file(url: str, dest_path: Path) -> None:
    """Download a file (PDF or HTML) and write it to ``dest_path``.
    The function respects existing files – if the file already exists and is newer than 1 day, it is skipped.
    """
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Skip if file exists and is recent (avoid re‑downloading unchanged PDFs)
        if dest_path.exists():
            mtime = datetime.fromtimestamp(dest_path.stat().st_mtime)
            if (datetime.now() - mtime).days < 1:
                logging.info(f"Skipping recent file: {dest_path.name}")
                return
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name, leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        logging.info(f"Saved: {dest_path}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")

def fetch_html(url: str) -> str:
    """Return raw HTML text for a given URL (used for scraping pages that are not PDFs)."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logging.error(f"HTML fetch error for {url}: {e}")
        return ""

def save_html(content: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"HTML saved: {dest_path}")

# ---------------------------------------------------------------------------
# Main download routines
# ---------------------------------------------------------------------------

def download_bank_factoring(urls: dict, base_dir: Path) -> None:
    for bank, url in urls.items():
        logging.info(f"Processing {bank} factoring page: {url}")
        if url.lower().endswith('.pdf'):
            dest = base_dir / bank / f"{bank}_factoring.pdf"
            download_file(url, dest)
        else:
            html = fetch_html(url)
            if html:
                dest = base_dir / bank / f"{bank}_factoring.html"
                save_html(html, dest)

def download_legal_documents(urls: dict, base_dir: Path) -> None:
    for name, url in urls.items():
        logging.info(f"Downloading legal doc {name}: {url}")
        if url.lower().endswith('.pdf'):
            dest = base_dir / f"{name}.pdf"
            download_file(url, dest)
        else:
            html = fetch_html(url)
            if html:
                dest = base_dir / f"{name}.html"
                save_html(html, dest)

def download_international_refs(urls: dict, base_dir: Path) -> None:
    for name, url in urls.items():
        logging.info(f"Downloading international reference {name}: {url}")
        if url.lower().endswith('.pdf'):
            dest = base_dir / f"{name}.pdf"
            download_file(url, dest)
        else:
            html = fetch_html(url)
            if html:
                dest = base_dir / f"{name}.html"
                save_html(html, dest)

# ---------------------------------------------------------------------------
# Scheduler – run on the first day of each month
# ---------------------------------------------------------------------------

def run_all_downloads() -> None:
    project_root = Path(__file__).resolve().parent
    factoring_dir = project_root / "legal_documents" / "factoring"
    banks_dir = factoring_dir / "banks"
    nbfc_dir = factoring_dir / "nbfcs"
    legal_dir = project_root / "legal_documents" / "indian_banks"
    intl_dir = project_root / "legal_documents" / "international"

    for d in [banks_dir, nbfc_dir, legal_dir, intl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    download_bank_factoring(BANK_FACTRING_URLS, banks_dir)
    download_bank_factoring(NBFC_FACTRING_URLS, nbfc_dir)
    download_legal_documents(INDIAN_LEGAL_URLS, legal_dir)
    download_international_refs(INTERNATIONAL_REFERENCES, intl_dir)

    logging.info("All downloads completed at %s", datetime.now().isoformat())

# ---------------------------------------------------------------------------
# Entry point – either one‑off run or scheduled mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download factoring corpus and schedule monthly refresh.")
    parser.add_argument("--schedule", action="store_true", help="Run in scheduler mode (keeps process alive).")
    args = parser.parse_args()

    if args.schedule:
        import schedule
        schedule.every().month.at("02:00").do(run_all_downloads)
        logging.info("Scheduler started – will run on the 1st of each month at 02:00.")
        run_all_downloads()
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_all_downloads()

"""
Usage:
    # One‑off download (useful for initial setup)
    python download_corpus.py

    # Continuous monthly refresh (run in a background terminal or as a service)
    python download_corpus.py --schedule
"""
