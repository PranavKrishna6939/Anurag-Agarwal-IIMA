"""Hybrid RAG-NER extraction pipeline (Chunk Retriever + DeepSeek API + Structured Legal NER)."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset_builder import SECTOR_PROFILES


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

ALL_DOMAIN_TERMS = sorted({
    kw.lower()
    for prof in SECTOR_PROFILES.values()
    for kw in prof["keywords"]
})

ENTITY_QUERIES = {
    "parties": "BETWEEN Claimant Respondent party names arbitration",
    "amounts": "total claims INR Crore counterclaim net arbitral award interest",
    "statutory": "Section 34 Section 37 Arbitration Conciliation Act High Court Supreme Court",
    "technical": "disputes delay months escalation clause scope variation infrastructure",
}


@dataclass
class NEREvalReport:
    overall_precision: float
    overall_recall: float
    overall_f1: float
    total_gold: int
    total_predicted: int
    per_type: pd.DataFrame
    extracted_df: pd.DataFrame


def chunk_document(text: str, max_chars: int = 420) -> List[str]:
    """Split a legal judgment into paragraph-aligned chunks for RAG retrieval."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s
            if buf:
                chunks.append(buf)
    return chunks or [text]


def retrieve_relevant_chunks(chunks: List[str], query: str, top_k: int = 3) -> List[str]:
    """Retrieve top-k chunks relevant to an entity extraction query via cosine similarity."""
    if len(chunks) <= top_k:
        return chunks
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(chunks + [query])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in sorted(top_idx)]


def extract_with_deepseek(
    context_text: str,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
) -> Optional[List[Dict[str, str]]]:
    """Optional live DeepSeek RAG-NER extraction when DEEPSEEK_API_KEY is configured."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None

    prompt = (
        "Extract structured legal arbitration entities as a JSON array of objects with keys "
        "'type' and 'text'. Allowed types: CLAIMANT, RESPONDENT, SECTOR, FORUM, STATUTE, "
        "CLAUSE_TYPE, CLAIM_AMOUNT_CR, COUNTERCLAIM_CR, AWARD_AMOUNT_CR, INTEREST_RATE, "
        "DELAY_MONTHS, DOMAIN_TERM. Return JSON only.\n\n"
        f"Context:\n{context_text}"
    )
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=25,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [{"type": str(x["type"]), "text": str(x["text"])} for x in parsed if "type" in x and "text" in x]
    except Exception:
        return None
    return None


def extract_entities_hybrid(text: str, use_deepseek: bool = False) -> List[Dict[str, str]]:
    """
    RAG-grounded structured entity extraction:
      1. Chunks judgment & retrieves top-k passages per legal facet.
      2. Calls DeepSeek API when explicitly enabled & keyed; otherwise applies
         structured legal pattern + domain vocabulary span extraction.
    """
    chunks = chunk_document(text)
    retrieved_blocks = []
    for q in ENTITY_QUERIES.values():
        retrieved_blocks.extend(retrieve_relevant_chunks(chunks, q, top_k=2))
    context = "\n".join(dict.fromkeys(retrieved_blocks))

    if use_deepseek:
        llm_out = extract_with_deepseek(context)
        if llm_out is not None:
            return llm_out

    entities: List[Dict[str, str]] = []

    def add(etype: str, val: Optional[str]) -> None:
        if val and val.strip():
            entities.append({"type": etype, "text": val.strip()})

    m_parties = re.search(r"BETWEEN:\s*(.+?)\s*\(Claimant\)\s*AND\s*(.+?)\s*\(Respondent\)", context, re.I)
    if m_parties:
        add("CLAIMANT", m_parties.group(1))
        add("RESPONDENT", m_parties.group(2))

    m_forum = re.search(r"IN THE\s+(.+?)\s*\|", context)
    if m_forum:
        add("FORUM", m_forum.group(1).title().replace("Of", "of"))

    m_stat = re.search(r"STATUTORY PROVISION:\s*(.+)", context)
    if m_stat:
        add("STATUTE", m_stat.group(1))

    m_sec = re.search(r"INFRASTRUCTURE SECTOR:\s*(.+)", context)
    if m_sec:
        add("SECTOR", m_sec.group(1))

    m_claim = re.search(r"total claims of\s*(INR\s*[\d,.]+\s*Crore)", context, re.I)
    if m_claim:
        add("CLAIM_AMOUNT_CR", m_claim.group(1))

    m_counter = re.search(r"counterclaim of\s*(INR\s*[\d,.]+\s*Crore)", context, re.I)
    if m_counter:
        add("COUNTERCLAIM_CR", m_counter.group(1))

    m_award = re.search(r"net arbitral award of\s*(INR\s*[\d,.]+\s*Crore)", context, re.I)
    if m_award:
        add("AWARD_AMOUNT_CR", m_award.group(1))

    m_int = re.search(r"interest at\s*([\d.]+%)", context, re.I)
    if m_int:
        add("INTEREST_RATE", m_int.group(1))

    m_delay = re.search(r"delays spanning\s*(\d+\s*months)", context, re.I)
    if m_delay:
        add("DELAY_MONTHS", m_delay.group(1))

    m_clause = re.search(r"Disputes arose concerning\s+(.+?)\s+and\s+project execution", context, re.I)
    if m_clause:
        add("CLAUSE_TYPE", m_clause.group(1).title())

    # Match domain terminology spans inside retrieved chunks
    lower_ctx = context.lower()
    for term in ALL_DOMAIN_TERMS:
        if term in lower_ctx:
            add("DOMAIN_TERM", term)

    # Deduplicate (type, normalized text)
    seen = set()
    unique = []
    for e in entities:
        key = (e["type"], e["text"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def evaluate_rag_ner(cases: List[dict], use_deepseek: bool = False) -> NEREvalReport:
    """Run RAG-NER across the full corpus and compute strict span Precision/Recall/F1."""
    tp_by_type: Dict[str, int] = defaultdict(int)
    fp_by_type: Dict[str, int] = defaultdict(int)
    fn_by_type: Dict[str, int] = defaultdict(int)

    extracted_rows = []
    total_gold = 0
    total_pred = 0

    for c in cases:
        preds = extract_entities_hybrid(c["text"], use_deepseek=use_deepseek)
        golds = c.get("gold_entities", [])

        pred_set = {(e["type"], e["text"].strip().lower()) for e in preds}
        gold_set = {(e["type"], e["text"].strip().lower()) for e in golds}

        total_gold += len(gold_set)
        total_pred += len(pred_set)

        for etype, txt in pred_set:
            if (etype, txt) in gold_set:
                tp_by_type[etype] += 1
            else:
                fp_by_type[etype] += 1
        for etype, txt in gold_set:
            if (etype, txt) not in pred_set:
                fn_by_type[etype] += 1

        for e in preds:
            extracted_rows.append({
                "case_id": c["case_id"],
                "sector": c["sector"],
                "entity_type": e["type"],
                "entity_text": e["text"],
            })

    all_types = sorted(set(tp_by_type) | set(fp_by_type) | set(fn_by_type))
    rows = []
    for et in all_types:
        tp, fp, fn = tp_by_type[et], fp_by_type[et], fn_by_type[et]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({
            "Entity_Type": et,
            "Support": tp + fn,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
        })

    tp_all = sum(tp_by_type.values())
    fp_all = sum(fp_by_type.values())
    fn_all = sum(fn_by_type.values())
    p_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    r_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0

    return NEREvalReport(
        overall_precision=round(p_all, 4),
        overall_recall=round(r_all, 4),
        overall_f1=round(f1_all, 4),
        total_gold=total_gold,
        total_predicted=total_pred,
        per_type=pd.DataFrame(rows).set_index("Entity_Type"),
        extracted_df=pd.DataFrame(extracted_rows),
    )
