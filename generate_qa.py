"""
Generate instruction-tuning Q&A pairs from the factoring/banking corpus
using Groq (llama-3.1-8b-instant).

For each document, chunks the text and asks the LLM to produce JSON Q&A
pairs in two modes: legal (regulatory, Act/section-grounded) and
financial (mechanism, calculations, scenarios).

Output: qa_dataset.jsonl with {instruction, input, output, source, category}
Target: ~2000 pairs.

Run: python generate_qa.py
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "legal_documents"
OUT_PATH = ROOT / "qa_dataset_v2.jsonl"

MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY missing from .env")

client = Groq(api_key=API_KEY)

TARGET_PAIRS = 500
PAIRS_PER_CHUNK = 4
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200
MIN_ANSWER_CHARS = 60        # drop terse one-phrase answers client-side

PROMPT = """You are generating high-quality instruction-tuning data for a banking/factoring expert assistant.

Read the passage below and produce exactly {n} Q&A pairs grounded strictly in the passage.
Split evenly: half LEGAL/REGULATORY (statutes, RBI rules, recourse, assignment, definitions, compliance), half FINANCIAL/MECHANICAL (discount rates, advances, factor reserve, calculations, mechanism, parties, advantages/disadvantages).

Hard rules:
- Do NOT invent facts not in the passage.
- Questions: self-contained (no "according to the text"), specific, never yes/no.
- Answers: 3-5 full sentences, 50-200 words. NEVER a single word or phrase — always explain mechanism, cite the specific number/term/section, and give context. An answer like "The forfaiter" or "$250,000" is REJECTED; rewrite as a full explanation.
- Use exact numbers, percentages, timeframes, statute sections, and proper nouns from the passage.
- Mix difficulty: definitional, comparative, and scenario-based.
- If the passage lacks substantive factual content, return [].

Return ONLY a JSON array, no prose:
[
  {{"question": "...", "answer": "...", "category": "legal"}},
  {{"question": "...", "answer": "...", "category": "financial"}}
]

PASSAGE:
---
{passage}
---
"""


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return [c for c in chunks if len(c.strip()) >= 500]


def extract_json_array(s: str):
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def generate_for_chunk(passage: str, n: int = PAIRS_PER_CHUNK, retries: int = 2):
    prompt = PROMPT.format(n=n, passage=passage)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content
            pairs = extract_json_array(raw)
            valid = [
                p for p in pairs
                if isinstance(p, dict)
                and p.get("question") and p.get("answer")
                and len(p["answer"].strip()) >= MIN_ANSWER_CHARS
                and len(p["question"].strip()) >= 15
                and p.get("category") in ("legal", "financial")
            ]
            return valid
        except Exception as e:
            logging.warning(f"attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return []


def load_docs():
    docs = []
    for p in DOCS_DIR.rglob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) >= 500:
            docs.append((p, text))
    return docs


def already_done_sources():
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line).get("source"))
        except json.JSONDecodeError:
            pass
    return done


def main():
    docs = load_docs()
    logging.info(f"found {len(docs)} source documents")
    done_sources = already_done_sources()
    total = sum(1 for _ in OUT_PATH.open(encoding="utf-8")) if OUT_PATH.exists() else 0
    logging.info(f"resuming — already have {total} pairs")

    with OUT_PATH.open("a", encoding="utf-8") as out:
        for path, text in docs:
            if total >= TARGET_PAIRS:
                break
            src = str(path.relative_to(ROOT))
            if src in done_sources:
                logging.info(f"skip (done): {src}")
                continue
            chunks = chunk_text(text)
            logging.info(f"{src}: {len(chunks)} chunks")
            for idx, chunk in enumerate(chunks):
                if total >= TARGET_PAIRS:
                    break
                pairs = generate_for_chunk(chunk)
                for p in pairs:
                    record = {
                        "instruction": p["question"],
                        "input": "",
                        "output": p["answer"],
                        "category": p["category"],
                        "source": src,
                        "chunk": idx,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                out.flush()
                logging.info(f"  chunk {idx+1}/{len(chunks)}: +{len(pairs)} pairs (total={total})")
                time.sleep(2.0)  # gentle rate limit
    logging.info(f"done. total pairs: {total}")


if __name__ == "__main__":
    main()
