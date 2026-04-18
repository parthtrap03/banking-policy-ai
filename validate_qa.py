"""
Validate qa_dataset.jsonl: deduplicate, drop malformed records, check
answer grounding against source docs via simple token overlap.

Writes qa_dataset.clean.jsonl and prints a report.

Run: python validate_qa.py
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

ROOT = Path(__file__).parent
IN_PATH = ROOT / "qa_dataset.jsonl"
OUT_PATH = ROOT / "qa_dataset.clean.jsonl"
DOCS_DIR = ROOT / "legal_documents"

MIN_ANSWER_LEN = 40
MAX_ANSWER_LEN = 1500
MIN_QUESTION_LEN = 15
GROUNDING_THRESHOLD = 0.25  # fraction of answer content-tokens that must appear in source


STOPWORDS = set("""
a an the and or but if then of to in on at by for with from as is are was were be been being this that these those it its
""".split())


def tokenize(s: str):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOPWORDS and len(w) > 2]


def load_source_tokens():
    cache = {}
    for p in DOCS_DIR.rglob("*.txt"):
        rel = str(p.relative_to(ROOT))
        cache[rel] = set(tokenize(p.read_text(encoding="utf-8", errors="ignore")))
    return cache


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found — run generate_qa.py first")

    src_tokens = load_source_tokens()
    seen_questions = set()
    stats = Counter()
    kept = []

    for line in IN_PATH.read_text(encoding="utf-8").splitlines():
        stats["total"] += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["malformed"] += 1
            continue

        q = (rec.get("instruction") or "").strip()
        a = (rec.get("output") or "").strip()
        cat = rec.get("category")
        source = rec.get("source")

        if not q or not a or cat not in ("legal", "financial"):
            stats["missing_fields"] += 1
            continue
        if len(q) < MIN_QUESTION_LEN or not (MIN_ANSWER_LEN <= len(a) <= MAX_ANSWER_LEN):
            stats["bad_length"] += 1
            continue

        qnorm = re.sub(r"\s+", " ", q.lower())
        if qnorm in seen_questions:
            stats["duplicate"] += 1
            continue
        seen_questions.add(qnorm)

        src_toks = src_tokens.get(source, set())
        if src_toks:
            ans_toks = tokenize(a)
            if ans_toks:
                overlap = sum(1 for t in ans_toks if t in src_toks) / len(ans_toks)
                if overlap < GROUNDING_THRESHOLD:
                    stats["ungrounded"] += 1
                    continue

        kept.append(rec)
        stats[f"kept_{cat}"] += 1

    OUT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8")
    logging.info(f"wrote {len(kept)} clean pairs to {OUT_PATH.name}")
    for k, v in stats.most_common():
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
