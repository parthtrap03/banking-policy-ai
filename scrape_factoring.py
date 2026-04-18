"""
Broader factoring/banking corpus scraper.

Scrapes educational, regulatory, and industry pages on factoring, forfaiting,
invoice discounting, receivables financing, TReDS, and related Indian banking
regulation. Saves cleaned text into legal_documents/factoring/ and
legal_documents/regulatory/.

Run: python scrape_factoring.py
"""

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

ROOT = Path(__file__).parent
OUT_EDU = ROOT / "legal_documents" / "factoring"
OUT_REG = ROOT / "legal_documents" / "regulatory"
OUT_EDU.mkdir(parents=True, exist_ok=True)
OUT_REG.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FactoringCorpusBot/1.0; educational use)"
}

# Educational / explainer sources — verified landing pages
EDUCATIONAL_URLS = [
    "https://corporatefinanceinstitute.com/resources/commercial-lending/forfaiting/",
    "https://en.wikipedia.org/wiki/Factoring_(finance)",
    "https://en.wikipedia.org/wiki/Forfaiting",
    "https://en.wikipedia.org/wiki/Invoice_discounting",
    "https://en.wikipedia.org/wiki/Letter_of_credit",
    "https://en.wikipedia.org/wiki/Accounts_receivable",
    "https://en.wikipedia.org/wiki/Supply_chain_finance",
    "https://en.wikipedia.org/wiki/Reverse_factoring",
    "https://en.wikipedia.org/wiki/Trade_finance",
    "https://en.wikipedia.org/wiki/Discounting",
    "https://en.wikipedia.org/wiki/Working_capital",
    "https://en.wikipedia.org/wiki/Securitization",
    "https://en.wikipedia.org/wiki/Assignment_(law)",
    "https://en.wikipedia.org/wiki/Uniform_Commercial_Code",
]

# Indian regulatory / institutional sources
REGULATORY_URLS = [
    "https://en.wikipedia.org/wiki/Reserve_Bank_of_India",
    "https://en.wikipedia.org/wiki/Non-banking_financial_company",
    "https://www.rxil.in/",
    "https://www.rxil.in/aboutTreds",
]


def safe_name(url: str) -> str:
    parsed = urlparse(url)
    base = parsed.netloc + parsed.path
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base)[:180] + ".txt"


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # drop boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    # prefer main content containers when present
    main = soup.find("main") or soup.find("article") or soup.find(id="mw-content-text") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    # collapse blank-line runs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def scrape(url: str, dest_dir: Path) -> bool:
    path = dest_dir / safe_name(url)
    if path.exists() and path.stat().st_size > 500:
        logging.info(f"skip (cached): {path.name}")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        text = clean_text(r.text)
        if len(text) < 500:
            logging.warning(f"too short, skipping: {url}")
            return False
        path.write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
        logging.info(f"saved: {path.name} ({len(text):,} chars)")
        return True
    except Exception as e:
        logging.error(f"failed {url}: {e}")
        return False


def main():
    ok = fail = 0
    for url in EDUCATIONAL_URLS:
        if scrape(url, OUT_EDU):
            ok += 1
        else:
            fail += 1
        time.sleep(1)
    for url in REGULATORY_URLS:
        if scrape(url, OUT_REG):
            ok += 1
        else:
            fail += 1
        time.sleep(1)
    logging.info(f"done. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
