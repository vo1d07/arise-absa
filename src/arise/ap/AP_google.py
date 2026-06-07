"""
Iterative Aspect Proposer + Aspect Validator (ACSD) in ONE file.

Input:  raw dataset CSV with columns:
  - text
  - merged_entities  (for building span bank; Python-literal string or JSON string)

Output: CSV containing ONLY Aspect Validator's final (post-iteration) outputs:
  - sample_id
  - validator_final   (JSON string)

Method:
    0) Build span bank from merged_entities (span-level prototypes) + embeddings
    1) Aspect Proposer proposes candidate aspect categories (global_candidates + evidence hints)
    2) Aspect Validator verifies each (x,c) using span-prototype retrieval + evidence-gated LLM
    3) Iteration: for candidates with score < threshold or mentioned=false,
        - Aspect Validator critic produces structured critique (why wrong + suggest add/drop) - Aspect Proposer revises candidate set using critiques (edit-based, schema-locked)
        until convergence or max_rounds

Prereqs:
    pip install pandas pyarrow numpy google-genai sentence-transformers tqdm

Env:
    export GOOGLE_API_KEY="..."

Example:
    python3 A-IPV_google.py \
    --in_csv datasets/dataset.csv \
    --out_csv outputs/AIPV_output.csv \
    --threshold 0.5 \
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:
    genai = None
    genai_types = None

from evaluation_AIPV import eval_a_ipv_aspect_detection


# -----------------------------
# Schema
# -----------------------------
ASPECTS_8 = [
    "On-campus Service",
    "Counseling Service",
    "Mental Health Service",
    "Wellness Service",
    "Therapy Service",
    "Hotline Service",
    "Service Availability",
    "General",
]

ASPECT_DESC = {
    "On-campus Service": "Emphasize resources that come from the campus.",
    "Counseling Service": "Emphasize the counseling process.",
    "Mental Health Service": "Emphasize the resources for mental health.",
    "Wellness Service": "Emphasize resources for wellness and various support systems.",
    "Therapy Service": "Emphasize the therapy process.",
    "Hotline Service": "Emphasize the 24 hour and emergency services.",
    "Service Availability": "Emphasize the availability/accessibility of the various resources.",
    "General": "For those which are not specifically mentioned, label them as general.",
}

TEXT_COL = "text"
MERGED_COL = "merged_entities"
GOLD_COL = "gold_struct"


# -----------------------------
# Utilities
# -----------------------------
def parse_json_cell(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse_json_with_fallback(s: str) -> Any:
    """
    Fast path: parse the whole string as JSON.
    Fallback path: extract and parse the first JSON value when the model adds extra text.
    """
    raw = (s or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    candidates: List[str] = []
    if raw:
        candidates.append(raw)

    decoder = json.JSONDecoder()
    starts = [i for i in (raw.find("{"), raw.find("[")) if i != -1]
    if starts:
        start = min(starts)
        snippet = raw[start:]
        candidates.append(snippet)
        try:
            obj, _ = decoder.raw_decode(snippet)
            return obj
        except json.JSONDecodeError:
            pass

    # Try strict JSON first, then lightly normalized JSON, then Python-literal fallback.
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

        normalized = cand
        normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
        normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)

        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            pass

        try:
            obj = ast.literal_eval(normalized)
            if isinstance(obj, (dict, list, tuple, set, str, int, float, bool)) or obj is None:
                return obj
        except Exception:
            pass

    raise ValueError("Could not parse model output as JSON.")


def safe_slice(text: str, start: int, end: int) -> str:
    start = max(0, min(len(text), int(start)))
    end = max(0, min(len(text), int(end)))
    if end < start:
        start, end = end, start
    return text[start:end]


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    start = 0
    for m in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = m.end()
        sent = text[start:end].strip()
        if sent:
            lstrip = len(text[start:end]) - len(text[start:end].lstrip())
            s0 = start + lstrip
            s1 = end
            spans.append((s0, s1, text[s0:s1]))
        start = end
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            lstrip = len(text[start:]) - len(text[start:].lstrip())
            s0 = start + lstrip
            spans.append((s0, len(text), text[s0:len(text)]))
    return spans


def keep_substrings(evs: List[str], text: str) -> List[str]:
    out = []
    for e in evs or []:
        if isinstance(e, str) and e and e in text:
            out.append(e)
    # de-dup preserve order
    seen = set()
    dedup = []
    for e in out:
        if e not in seen:
            seen.add(e)
            dedup.append(e)
    return dedup


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def sanitize_text_for_api(text: Any) -> str:
    s = str(text or "")
    out_chars: List[str] = []
    for ch in s:
        code = ord(ch)
        # Remove lone surrogate code points and disallowed control chars.
        if 0xD800 <= code <= 0xDFFF:
            continue
        if code < 32 and ch not in "\t\n\r":
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars).strip()
    return cleaned if cleaned else " "


def sanitize_message_content(text: Any) -> str:
    return sanitize_text_for_api(text)


# -----------------------------
# Build span bank
# -----------------------------
def build_span_bank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ridx, row in df.iterrows():
        text = str(row.get(TEXT_COL, "") or "")
        ents = parse_json_cell(row.get(MERGED_COL)) or []
        if not isinstance(ents, list):
            continue
        for eidx, ent in enumerate(ents):
            if not isinstance(ent, dict):
                continue
            start = ent.get("start")
            end = ent.get("end")
            aspects = ent.get("aspects") or []
            sentiments = ent.get("sentiments") or []
            raw_labels = ent.get("raw_labels") or []
            if start is None or end is None:
                continue
            span_text = safe_slice(text, int(start), int(end)).strip()
            if not span_text:
                continue
            for a in aspects:
                a = str(a).strip()
                if a not in ASPECTS_8:
                    continue
                rows.append(
                    {
                        "sample_id": int(ridx),
                        "entity_id": int(eidx),
                        "category": a,
                        "start": int(start),
                        "end": int(end),
                        "span_text": span_text,
                        "sentiments": sentiments,
                        "raw_labels": raw_labels,
                        "full_text": text,
                    }
                )
    span_bank = pd.DataFrame(rows)
    if span_bank.empty:
        raise RuntimeError("span_bank is empty. Check merged_entities parsing and aspect names.")
    span_bank = span_bank.drop_duplicates(subset=["sample_id", "start", "end", "category"]).reset_index(drop=True)
    span_bank["row_id"] = np.arange(len(span_bank), dtype=int)
    return span_bank


# -----------------------------
# Embedding
# -----------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model: str, api_key: Optional[str] = None, batch_size: int = 128):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. Install with: pip install sentence-transformers"
            ) from e
        self.encoder = SentenceTransformer(model)
        self.model = model
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}

    def embed_one(self, text: str) -> np.ndarray:
        key = sanitize_text_for_api(text)
        if key in self.cache:
            return self.cache[key]
        vec = self.encoder.encode(key, convert_to_numpy=True)
        vec = np.array(vec, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        self.cache[key] = vec
        return vec

    def embed_many(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding span bank"):
            batch = [sanitize_text_for_api(t) for t in texts[i : i + self.batch_size]]
            batch_vecs = self.encoder.encode(batch, convert_to_numpy=True)
            vecs.extend([np.array(v, dtype=np.float32) for v in batch_vecs])
        arr = np.array(vecs, dtype=np.float32)
        return l2_normalize(arr)


def google_text_from_response(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts: List[str] = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            ptxt = getattr(part, "text", None)
            if isinstance(ptxt, str) and ptxt:
                parts.append(ptxt)
    return "\n".join(parts)


def genai_generate_content_with_retry(
    client,
    model: str,
    contents: Any,
    config: Any,
    max_retries: int = 6,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
):
    """Call `client.models.generate_content` with exponential backoff + jitter on transient errors.

    Retries when the exception message or attributes indicate a transient server error
    (e.g., HTTP 5xx, 'UNAVAILABLE'). Non-transient exceptions are re-raised immediately.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:  # pragma: no cover - runtime behaviour
            msg = str(e) or ""
            is_transient = False

            # Common signals for transient service errors
            if hasattr(e, "code"):
                try:
                    code = int(getattr(e, "code"))
                    if 500 <= code < 600:
                        is_transient = True
                except Exception:
                    pass

            if not is_transient:
                if "503" in msg or "UNAVAILABLE" in msg.upper() or "UNAVAILABLE" in getattr(e, "status", ""):
                    is_transient = True

            if not is_transient:
                raise

            # Last attempt -> re-raise to surface the error
            if attempt >= max_retries - 1:
                raise

            backoff = min(max_backoff, initial_backoff * (2 ** attempt))
            # jitter 50%-100% of backoff
            sleep_for = backoff * (0.5 + random.random() * 0.5)
            print(f"[Warning] genai transient error: {e}; retrying in {sleep_for:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_for)


# -----------------------------
# Retrieval over span bank
# -----------------------------
class SpanRetriever:
    def __init__(self, span_bank: pd.DataFrame, span_embeds: np.ndarray):
        assert len(span_bank) == span_embeds.shape[0]
        self.span_bank = span_bank.reset_index(drop=True)
        self.embeds = span_embeds.astype(np.float32)

        self.cat_to_ids: Dict[str, np.ndarray] = {}
        for c in ASPECTS_8:
            ids = self.span_bank.index[self.span_bank["category"] == c].to_numpy(dtype=int)
            self.cat_to_ids[c] = ids

        # Hard negatives: everything not in c (small data -> store ids, compute sims on-demand)
        self.neg_ids: Dict[str, np.ndarray] = {}
        all_ids = np.arange(len(self.span_bank), dtype=int)
        for c in ASPECTS_8:
            self.neg_ids[c] = all_ids[self.span_bank["category"].to_numpy() != c]

    def topk_positive(self, qvec: np.ndarray, category: str, k: int, exclude_sample_id: Optional[int] = None) -> List[Dict[str, Any]]:
        ids = self.cat_to_ids.get(category, np.array([], dtype=int))
        if ids.size == 0:
            return []
        
        # Dynamic self-exclusion: filter out spans from the current sample
        if exclude_sample_id is not None:
            mask = self.span_bank.iloc[ids]["sample_id"].to_numpy() != exclude_sample_id
            ids = ids[mask]
            if ids.size == 0:
                return []
        
        sims = self.embeds[ids] @ qvec
        top_idx = np.argsort(-sims)[: max(0, k)]
        out = []
        for j in top_idx:
            rid = int(ids[j])
            row = self.span_bank.iloc[rid].to_dict()
            out.append(
                {
                    "row_id": rid,
                    "sim": float(sims[j]),
                    "span_text": row.get("span_text", ""),
                    "category": row.get("category", ""),
                    "sample_id": int(row.get("sample_id", -1)),
                    "start": int(row.get("start", -1)),
                    "end": int(row.get("end", -1)),
                    "full_text": row.get("full_text", ""),
                }
            )
        return out

    def topm_hard_negatives(self, qvec: np.ndarray, category: str, m: int, exclude_sample_id: Optional[int] = None) -> List[Dict[str, Any]]:
        ids = self.neg_ids.get(category, np.array([], dtype=int))
        if ids.size == 0:
            return []
        
        # Dynamic self-exclusion: filter out spans from the current sample
        if exclude_sample_id is not None:
            mask = self.span_bank.iloc[ids]["sample_id"].to_numpy() != exclude_sample_id
            ids = ids[mask]
            if ids.size == 0:
                return []
        
        sims = self.embeds[ids] @ qvec
        top_idx = np.argsort(-sims)[: max(0, m)]
        out = []
        for j in top_idx:
            rid = int(ids[j])
            row = self.span_bank.iloc[rid].to_dict()
            out.append(
                {
                    "row_id": rid,
                    "sim": float(sims[j]),
                    "span_text": row.get("span_text", ""),
                    "category": row.get("category", ""),
                    "sample_id": int(row.get("sample_id", -1)),
                    "start": int(row.get("start", -1)),
                    "end": int(row.get("end", -1)),
                    "full_text": row.get("full_text", ""),
                }
            )
        return out


# -----------------------------
# Aspect Proposer (propose + revise)
# -----------------------------
class Proposer:
    def __init__(
        self,
        model: str,
        predefined_categories: List[str],
        category_descriptions: Dict[str, str],
        top_k: int = 6,
        per_sentence_top_k: int = 4,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        if genai is None:
            raise ImportError("google-genai is required. Install with: pip install google-genai")
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=resolved_api_key)
        self.model = model
        self.categories = [c.strip() for c in predefined_categories]
        self.desc = category_descriptions
        self.top_k = min(top_k, len(self.categories))
        self.per_sentence_top_k = min(per_sentence_top_k, len(self.categories))
        self.temperature = temperature

    def _schema_text(self) -> str:
        return "\n".join([f"- {c}: {self.desc.get(c,'')}".rstrip() for c in self.categories])

    def _system(self) -> str:
        return """You are an information extraction model for student mental health feedback.

Task:
Given one student feedback comment, identify which aspects are explicitly discussed.

Valid aspect labels:
- On-campus Service
- Counseling Service
- Mental Health Service
- Wellness Service
- Therapy Service
- Hotline Service
- Service Availability
- General

Rules:
1. Output only aspects explicitly supported by the text.
2. Use only labels from the valid list.
3. Do not invent new labels.
4. Return JSON only.
"""

    def _chat(self, user_content: str) -> str:
        config: Any = {
            "temperature": self.temperature,
            "system_instruction": sanitize_message_content(self._system()),
            "response_mime_type": "application/json",
        }
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(**config)
        resp = genai_generate_content_with_retry(
            self.client,
            model=self.model,
            contents=sanitize_message_content(user_content),
            config=config,
        )
        return google_text_from_response(resp)

    @staticmethod
    def _parse_json_strict(s: str) -> Dict[str, Any]:
        data = parse_json_with_fallback(s)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _default_plan(self) -> Dict[str, Any]:
        return {
            "global_candidates": [
                {"category": c, "score": 0.01, "evidence": []}
                for c in self.categories[: self.top_k]
            ]
        }

    def plan(self, text: str) -> Dict[str, Any]:
        # Escape user-provided content to avoid JSON serialization errors
        text_escaped = json.dumps(text)
        schema_escaped = json.dumps(self._schema_text())
        
        prompt = f"""Student feedback:
{text_escaped}

Use only these valid aspect labels:
{schema_escaped}

Select up to {self.top_k} relevant aspects.
For each selected aspect, provide 1-3 evidence substrings copied EXACTLY from the input text and a score in [0,1].

Return ONLY JSON:
{{
    "global_candidates": [
        {{
            "category": "<exact schema category>",
            "score": 0.0,
            "evidence": ["<verbatim substring>", "..."]
        }}
    ]
}}
"""
        raw = self._chat(prompt)
        try:
            data = self._parse_json_strict(raw)
        except Exception as e:
            preview = (raw or "").strip().replace("\n", " ")[:240]
            print(f"[Warning] proposer.plan JSON parse failed: {e}; raw preview: {preview}")
            return self._default_plan()

        global_candidates = []
        for item in data.get("global_candidates", []) or []:
            cat = item.get("category", "")
            if cat not in self.categories:
                continue
            score = float(item.get("score", 0.0))
            ev = keep_substrings(item.get("evidence", []), text)
            global_candidates.append({"category": cat, "score": max(0.0, min(1.0, score)), "evidence": ev})

        if not global_candidates:
            global_candidates = self._default_plan()["global_candidates"]

        return {"global_candidates": global_candidates}

    def revise(self, text: str, current_plan: Dict[str, Any], critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Edit-based revision of global candidates, schema-locked.
        """
        cur = current_plan.get("global_candidates", []) or []
        cur_min = [{"category": d.get("category"), "evidence": d.get("evidence", [])} for d in cur]

        # Escape user-provided content to avoid JSON serialization errors
        text_escaped = json.dumps(text)
        schema_escaped = json.dumps(self._schema_text())
        cur_min_escaped = json.dumps(cur_min, ensure_ascii=False)
        critiques_escaped = json.dumps(critiques, ensure_ascii=False)
        
        prompt = f"""
            Text x:
            {text_escaped}

            Schema:
            {schema_escaped}

            Current candidates:
            {cur_min_escaped}

            Critiques from verifier (Aspect Validator). Use them to edit the candidate set:
            {critiques_escaped}

            Task:
            Revise the candidate categories by adding/removing categories ONLY from the schema.
            For each retained/added category, provide 1-2 evidence substrings copied verbatim from Text x.

            Return ONLY JSON:
            {{
            "global_candidates": [
                {{"category": "<exact schema category>", "evidence": ["<verbatim substring>", "..."]}}
            ]
            }}

            Rules:
            - Do NOT include categories outside the schema.
            - Evidence must be exact substrings from Text x.
            - Keep the set small but do not miss clearly relevant categories.
            """

        raw = self._chat(prompt)
        try:
            data = self._parse_json_strict(raw)
        except Exception as e:
            preview = (raw or "").strip().replace("\n", " ")[:240]
            print(f"[Warning] proposer.revise JSON parse failed: {e}; raw preview: {preview}")
            return current_plan

        out = []
        for item in data.get("global_candidates", []) or []:
            cat = item.get("category", "")
            if cat not in self.categories:
                continue
            ev = keep_substrings(item.get("evidence", []), text)
            out.append({"category": cat, "score": 0.0, "evidence": ev})

        # Fallback: if revision fails, keep current
        if not out:
            return current_plan

        # Trim to top_k to control cost downstream
        out = out[: self.top_k]
        return {"global_candidates": out}


def evidence_from_proposer_plan(proposer_plan: Dict[str, Any], category: str) -> List[str]:
    out: List[str] = []
    for item in (proposer_plan or {}).get("global_candidates", []) or []:
        if item.get("category") == category:
            out.extend([e for e in (item.get("evidence") or []) if isinstance(e, str)])
    # de-dup preserve order
    seen = set()
    dedup = []
    for e in out:
        if e not in seen:
            seen.add(e)
            dedup.append(e)
    return dedup


# -----------------------------
# Aspect Validator decision (two-stage evidence) + Critic
# -----------------------------
class Validator:
    def __init__(
        self,
        chat_model: str,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        max_pos_to_show: int = 5,
        max_parent_texts: int = 2,
        max_neg_to_show: int = 4,
    ):
        if genai is None:
            raise ImportError("google-genai is required. Install with: pip install google-genai")
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=resolved_api_key)
        self.model = chat_model
        self.temperature = temperature
        self.max_pos_to_show = max_pos_to_show
        self.max_parent_texts = max_parent_texts
        self.max_neg_to_show = max_neg_to_show

    def _system_decide(self) -> str:
        return (
            "You are Aspect Validator for aspect category detection (ACD) task. "
            "Given (text x, category c), decide whether c is mentioned in x. "
            "You MUST provide evidence substrings copied verbatim from x. "
            "If you cannot quote valid evidence from x, you MUST set mentioned=false."
        )

    def _system_critic(self) -> str:
        return (
            "You are Aspect Validator for aspect CATEGORY detection (ACD) task in CRITIC mode. "
            "Given (text x, target aspect category c) and the current verification result, "
            "explain why the category may be wrong/uncertain and suggest edits. "
            "You MUST quote supporting spans verbatim from x (with character spans)."
        )

    @staticmethod
    def _parse_json(s: str) -> Dict[str, Any]:
        data = parse_json_with_fallback(s)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    @staticmethod
    def _evidence_is_valid(x: str, ev: Dict[str, Any]) -> bool:
        t = ev.get("text")
        sp = ev.get("span")
        if not isinstance(t, str) or not t:
            return False
        if t not in x:
            return False
        if not (isinstance(sp, list) and len(sp) == 2):
            return False
        s0, s1 = int(sp[0]), int(sp[1])
        if s0 < 0 or s1 > len(x) or s1 <= s0:
            return False
        return x[s0:s1] == t

    @staticmethod
    def _autofill_span_from_text(x: str, ev_text: str) -> Optional[List[int]]:
        if not isinstance(ev_text, str) or not ev_text:
            return None
        i = x.find(ev_text)
        if i == -1:
            return None
        return [i, i + len(ev_text)]

    def _chat(self, system: str, user: str) -> str:
        config: Any = {
            "temperature": self.temperature,
            "system_instruction": sanitize_message_content(system),
            "response_mime_type": "application/json",
        }
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(**config)
        resp = genai_generate_content_with_retry(
            self.client,
            model=self.model,
            contents=sanitize_message_content(user),
            config=config,
        )
        return google_text_from_response(resp)

    def decide(
        self,
        x: str,
        c: str,
        p_evidence: List[str],
        pos_spans: List[Dict[str, Any]],
        neg_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        desc = ASPECT_DESC.get(c, "")
        pe = p_evidence[:2]
        pe_str = "\n".join([f"- {e}" for e in pe]) if pe else "(none)"

        pos_lines = []
        for i, ex in enumerate(pos_spans[: self.max_pos_to_show], start=1):
            pos_lines.append(f"{i}. {ex['span_text']} (sim={ex['sim']:.3f})")
        pos_block = "\n".join(pos_lines) if pos_lines else "(none)"

        parent_lines = []
        for ex in pos_spans[: self.max_parent_texts]:
            ft = (ex.get("full_text") or "").strip()
            if ft:
                parent_lines.append(f"- {ft}")
        parent_block = "\n".join(parent_lines) if parent_lines else "(omitted)"

        neg_lines = []
        for i, ex in enumerate(neg_spans[: self.max_neg_to_show], start=1):
            neg_lines.append(f"{i}. [{ex['category']}] {ex['span_text']} (sim={ex['sim']:.3f})")
        neg_block = "\n".join(neg_lines) if neg_lines else "(none)"

        # Escape user-provided content to avoid JSON serialization errors
        x_escaped = json.dumps(x)
        c_escaped = json.dumps(c)
        desc_escaped = json.dumps(desc)
        pe_str_escaped = json.dumps(pe_str)
        pos_block_escaped = json.dumps(pos_block)
        parent_block_escaped = json.dumps(parent_block)
        neg_block_escaped = json.dumps(neg_block)

        user = f"""
            Text x: {x_escaped}

            Current category c: {c_escaped}
            Category definition/boundary (follow this strictly):
            {desc_escaped}

            Aspect Proposer evidence hints (may be incomplete):
            {pe_str_escaped}

            Retrieved positive span prototypes for category {c_escaped} (from ground-truth):
            {pos_block_escaped}

            (Optionally) some parent texts for prototypes:
            {parent_block_escaped}

            Retrieved hard negatives (similar spans from OTHER categories):
            {neg_block_escaped}

            Task:
            Decide if category {c_escaped} is mentioned in x.

            Output ONLY valid JSON:
            {{
            "category": {c_escaped},
            "mentioned": true/false,
            "evidence": [
                {{"text": "<verbatim substring from x>", "span": [start, end]}}
            ],
            "score": 0.0,
            "confusable_with": ["<optional other categories>"]
            }}

            Rules:
            - evidence.text MUST be a verbatim substring of x (NOT from prototypes).
            - span must be character indices into x for that evidence.text.
            - If you cannot find valid evidence in x, set mentioned=false and evidence=[].
            - If evidence seems to match a hard negative category better, set mentioned=false OR lower score and list confusable_with.
            """
        raw = self._chat(self._system_decide(), user)
        try:
            data = self._parse_json(raw)
        except Exception as e:
            return {"category": c, "mentioned": False, "evidence": [], "score": 0.0, "error": f"json_parse_error: {e}"}

        mentioned = bool(data.get("mentioned", False))
        try:
            score = float(data.get("score", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))

        raw_evidence = data.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = []

        # Stage 1 strict
        strict_valid = []
        for ev in raw_evidence:
            if isinstance(ev, dict) and self._evidence_is_valid(x, ev):
                strict_valid.append(ev)

        # Stage 2 relaxed fallback
        relaxed_valid = []
        if not strict_valid:
            for ev in raw_evidence:
                if not isinstance(ev, dict):
                    continue
                ev_text = ev.get("text")
                if not isinstance(ev_text, str) or not ev_text:
                    continue
                if ev_text not in x:
                    continue
                auto_span = self._autofill_span_from_text(x, ev_text)
                if auto_span is None:
                    continue
                relaxed_valid.append({"text": ev_text, "span": auto_span})

        final_evs = strict_valid if strict_valid else relaxed_valid

        if mentioned and not final_evs:
            mentioned = False
            score = 0.0

        conf = data.get("confusable_with", [])
        if not isinstance(conf, list):
            conf = []

        return {"category": c, "mentioned": mentioned, "evidence": final_evs, "score": score, "confusable_with": conf}

    def critique(self, x: str, c: str, decision: Dict[str, Any], tau: float) -> Dict[str, Any]:
        """
        Structured critique used for iteration. Schema-locked suggestions.
        """
        # Escape user-provided content to avoid JSON serialization errors
        x_escaped = json.dumps(x)
        c_escaped = json.dumps(c)
        desc_escaped = json.dumps(ASPECT_DESC.get(c, ""))
        decision_escaped = json.dumps(decision, ensure_ascii=False)
        
        user = f"""
            Text x: {x_escaped}

            Target category: {c_escaped}
            Category definition/boundary:
            {desc_escaped}

            Current verification result (Aspect Validator decide):
            {decision_escaped}

            Threshold tau for this category: {tau}

            Task:
            If the category is wrong/uncertain (below threshold or mentioned=false), explain WHY using spans from x,
            and suggest how to revise the candidate set.

            Output ONLY JSON:
            {{
            "target_category": {c_escaped},
            "why_wrong": ["..."],
            "supporting_spans": [{{"text":"<substring from x>","span":[start,end]}}],
            "confusable_with": ["<schema categories>"],
            "suggest_add": ["<schema categories>"],
            "suggest_drop": ["<schema categories>"]
            }}

            Rules:
            - supporting_spans MUST be copied verbatim from x with correct character spans.
            - suggest_add/suggest_drop/confusable_with MUST be categories from the schema only.
            - Keep suggestions minimal.
            """
        raw = self._chat(self._system_critic(), user)
        try:
            data = self._parse_json(raw)
        except Exception as e:
            return {
                "target_category": c,
                "why_wrong": [f"critic_json_parse_error: {e}"],
                "supporting_spans": [],
                "confusable_with": [],
                "suggest_add": [],
                "suggest_drop": [],
            }

        # Validate spans
        spans = data.get("supporting_spans", [])
        if not isinstance(spans, list):
            spans = []
        valid_spans = []
        for ev in spans:
            if isinstance(ev, dict) and self._evidence_is_valid(x, ev):
                valid_spans.append(ev)

        def _filter_cats(lst):
            if not isinstance(lst, list):
                return []
            out = []
            for z in lst:
                z = str(z).strip()
                if z in ASPECTS_8 and z not in out:
                    out.append(z)
            return out

        return {
            "target_category": c,
            "why_wrong": data.get("why_wrong", []) if isinstance(data.get("why_wrong", []), list) else [],
            "supporting_spans": valid_spans,
            "confusable_with": _filter_cats(data.get("confusable_with", [])),
            "suggest_add": _filter_cats(data.get("suggest_add", [])),
            "suggest_drop": _filter_cats(data.get("suggest_drop", [])),
        }


# -----------------------------
# Iteration driver
# -----------------------------
def iterative_refine(
    x: str,
    proposer: Proposer,
    validator: Validator,
    retriever: SpanRetriever,
    embedder: SentenceTransformerEmbedder,
    threshold: float,
    max_rounds: int = 2,
    top_k_pos: int = 5,
    top_m_neg: int = 4,
    exclude_sample_id: Optional[int] = None,
    disable_validator: bool = False,
) -> Dict[str, Any]:
    # initial plan
    plan_t = proposer.plan(x)
    Ct = [d.get("category") for d in plan_t.get("global_candidates", []) or []]
    Ct = [c for c in Ct if c in ASPECTS_8]

    if disable_validator:
        proposer_only = []
        for item in plan_t.get("global_candidates", []) or []:
            c = item.get("category")
            if c not in ASPECTS_8:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            proposer_only.append(
                {
                    "category": c,
                    "mentioned": bool(score > 0.0),
                    "evidence": [],
                    "score": max(0.0, min(1.0, score)),
                    "confusable_with": [],
                    "retrieval": {"query": "", "pos": [], "neg": []},
                }
            )
        return {
            "final_candidates": Ct,
            "validator_output": {"categories": proposer_only},
            "history": [],
        }

    history = []

    for r in range(max_rounds):
        decisions = []
        critiques = []

        # verify current candidates
        for c in Ct:
            p_ev = evidence_from_proposer_plan(plan_t, c)
            q = " ".join(p_ev[:2]).strip() if p_ev else x
            qvec = embedder.embed_one(q)
            pos = retriever.topk_positive(qvec, c, top_k_pos, exclude_sample_id=exclude_sample_id)
            neg = retriever.topm_hard_negatives(qvec, c, top_m_neg, exclude_sample_id=exclude_sample_id)

            dec = validator.decide(x=x, c=c, p_evidence=p_ev, pos_spans=pos, neg_spans=neg)
            dec["retrieval"] = {
                "query": q,
                "pos": [{"row_id": e["row_id"], "sim": e["sim"], "span_text": e["span_text"]} for e in pos],
                "neg": [{"row_id": e["row_id"], "sim": e["sim"], "category": e["category"], "span_text": e["span_text"]} for e in neg],
            }
            decisions.append(dec)

        # critiques for low confidence
        for dec in decisions:
            c = dec["category"]
            score = float(dec.get("score", 0.0) or 0.0)
            mentioned = bool(dec.get("mentioned", False))
            if (not mentioned) or (score < threshold):
                critiques.append(validator.critique(x=x, c=c, decision=dec, tau=threshold))


        # stop if no critiques
        if not critiques:
            history.append({"round": r, "candidates": Ct, "decisions": decisions, "critiques": [], "plan": []})
            break

        # revise candidates
        revised = proposer.revise(text=x, current_plan=plan_t, critiques=critiques)
        Ct_next = [d.get("category") for d in revised.get("global_candidates", []) or []]
        Ct_next = [c for c in Ct_next if c in ASPECTS_8]

        # convergence
        if set(Ct_next) == set(Ct):
            plan_t = revised
            break

        plan_t = revised
        Ct = Ct_next

        history.append({"round": r, "candidates": Ct, "decisions": decisions, "critiques": critiques, "plan": plan_t})

    # final verification for final candidates (use last decisions if last round used them and Ct unchanged)
    final_decisions = history[-1]["decisions"] if history else []
    final_candidates = [d.get("category") for d in plan_t.get("global_candidates", []) or []]
    return {
        "final_candidates": [c for c in final_candidates if c in ASPECTS_8],
        "validator_output": {"categories": final_decisions},
        "history": history,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="datasets/with_gold_struct.csv")
    ap.add_argument("--out_csv", default="outputs/AIPV_output.csv")
    ap.add_argument("--threshold", type=float, default=0.4)

    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--p_model", default="gemini-2.5-flash")
    ap.add_argument("--v_model", default="gemini-2.5-flash")

    ap.add_argument("--p_top_k", type=int, default=8)
    ap.add_argument("--max_rounds", type=int, default=2)
    ap.add_argument("--top_k_pos", type=int, default=5)
    ap.add_argument("--top_m_neg", type=int, default=3)
    ap.add_argument("--disable_validator", action="store_true", help="Ablation mode: skip Aspect Validator and use proposer candidates only")

    ap.add_argument("--sleep_s", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert TEXT_COL in df.columns, f"Missing column '{TEXT_COL}'"
    assert MERGED_COL in df.columns, f"Missing column '{MERGED_COL}'"

    df_run = df
    enable_self_exclusion = True  # RAV(x) = dataset \ {x}

    # Guard against empty dataset
    if len(df_run) == 0:
        raise ValueError("No samples in dataset.")

    # Build span bank + embeddings from the SAME dataset being processed (RAG-self mode)
    span_bank = build_span_bank(df_run)
    embedder = SentenceTransformerEmbedder(model=args.embed_model)
    span_embeds = embedder.embed_many(span_bank["span_text"].tolist())
    retriever = SpanRetriever(span_bank, span_embeds)

    # Agents
    proposer = Proposer(
        model=args.p_model,
        predefined_categories=ASPECTS_8,
        category_descriptions=ASPECT_DESC,
        top_k=args.p_top_k,
        temperature=0.0,
    )
    validator = Validator(chat_model=args.v_model, temperature=0.0)

    outputs = []
    for ridx, row in tqdm(df_run.iterrows(), total=len(df_run), desc=f"Iterative P2V"):
        x = str(row.get(TEXT_COL, "") or "").strip()
        if not x:
            outputs.append({
                "sample_id": int(ridx),
                "text": row.get(TEXT_COL, ""),
                "merged_entities": row.get(MERGED_COL, ""),
                "validator_final": json.dumps({"categories": []}, ensure_ascii=False),
                "candidates_final": json.dumps([], ensure_ascii=False),
                "history": json.dumps([], ensure_ascii=False)
            })
            continue

        # Dynamic self-exclusion: exclude current sample from RAG retrieval
        exclude_id = int(ridx) if enable_self_exclusion else None
        
        res = iterative_refine(
            x=x,
            proposer=proposer,
            validator=validator,
            retriever=retriever,
            embedder=embedder,
            threshold=args.threshold,
            max_rounds=args.max_rounds,
            top_k_pos=args.top_k_pos,
            top_m_neg=args.top_m_neg,
            exclude_sample_id=exclude_id,
            disable_validator=args.disable_validator,
        )

        # Output ONLY Aspect Validator final (post-iteration). Keep minimal: categories list.
        validator_final = res["validator_output"]
        candidates_final = res.get("final_candidates", [])
        outputs.append({
            "sample_id": int(ridx),
            "text": row.get(TEXT_COL, ""),
            "merged_entities": row.get(MERGED_COL, ""),
            "gold_struct": row.get(GOLD_COL, ""),
            "validator_final": json.dumps(validator_final, ensure_ascii=False),
            "candidates_final": json.dumps(candidates_final, ensure_ascii=False),
            "history": json.dumps(res.get("history", []), ensure_ascii=False, separators=(",", ":"))
        })

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    out_df = pd.DataFrame(outputs)
    # Ensure out_csv is a file path (not directory)
    if os.path.isdir(args.out_csv):
        args.out_csv = os.path.join(args.out_csv, "A-IPV_google.csv")
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} ({len(out_df)} rows)")
    print(f"    Threshold: {args.threshold}")
    print(f"    Columns: sample_id, text, merged_entities, gold_struct, validator_final, candidates_final, history")

    summary, per_aspect = eval_a_ipv_aspect_detection(out_df, gold_col=GOLD_COL, aspects_list=ASPECTS_8)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save per-aspect results
    per_aspect_file = os.path.splitext(args.out_csv)[0] + "_per_aspect.csv"
    per_aspect.to_csv(per_aspect_file, index=False)
    print(f"Saved per-aspect results to {per_aspect_file}")

if __name__ == "__main__":
    main()
