#!/usr/bin/env python3
"""
Memory-Augmented Sentiment Propose–Validate–Refine with LLM

Uses Google Gemini models for sentiment classification.

Input (from gate_by_threshold.py output):
  - CSV with at least columns:
      - sample_id
      - text
      - candidates_final   (JSON string list of aspect names)

Output:
  - CSV with new column `sentiment_final` (JSON string mapping aspect -> {label, evidence})
    - CSV with new column `validator_iterations` (JSON string mapping aspect -> per-iteration validator records)
  - JSON file `sentiment_memory.json` storing memory buckets {POS, NEG, NEU}

Example:
    export GOOGLE_API_KEY="..."
    python3 S-IPV_google.py \
    --in_csv outputs/AIPV_output.csv \
    --out_csv outputs/SIPV_output.csv \
    --out_memory outputs/memory_bank.json \
    --tau_sent 0.5 --t_max 3
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation_SIPV import eval_s_ipv_sentiment_conditional

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:
    genai = None
    genai_types = None

# -----------------------------
# Constants
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

LABELS3 = ["NEG", "NEU", "POS"]

TEXT_COL = "text"
GATED_COL = "candidates_final"
SAMPLE_ID_COL = "sample_id"


def sanitize_text_for_api(text: Any) -> str:
    s = str(text or "")
    out_chars: List[str] = []
    for ch in s:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            continue
        if code < 32 and ch not in "\t\n\r":
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars).strip()
    return cleaned if cleaned else " "


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

# -----------------------------
# Memory structure
# -----------------------------
@dataclass
class MemoryEntry:
    """Single memory entry storing aspect-sentiment evidence."""
    aspect: str
    sentiment: str  # POS/NEG/NEU
    evidence: List[str]  # list of evidence phrases
    text_id: Optional[Any]  # sample_id from original data
    confidence: float  # prediction confidence score
    embedding: Optional[np.ndarray] = None  # embedding of evidence text
    cold_start: bool = False  # whether this entry was created under cold-start validation


class Memory:
    """Structured memory storing aspect-sentiment examples with embeddings."""
    
    def __init__(self):
        self.entries: List[MemoryEntry] = []
    
    def add_entry(self, aspect: str, sentiment: str, evidence: List[str], 
                  text_id: Optional[Any], confidence: float, embedding: Optional[np.ndarray] = None,
                  cold_start: bool = False):
        """Add a new memory entry."""
        entry = MemoryEntry(
            aspect=aspect,
            sentiment=sentiment,
            evidence=[normalize_phrase(e) for e in evidence if e.strip()],
            text_id=text_id,
            confidence=confidence,
            embedding=embedding,
            cold_start=cold_start
        )
        self.entries.append(entry)
    
    def get_by_aspect(self, aspect: str) -> List[MemoryEntry]:
        """Get all entries for a specific aspect."""
        return [e for e in self.entries if e.aspect == aspect]
    
    def get_by_sentiment(self, sentiment: str) -> List[MemoryEntry]:
        """Get all entries for a specific sentiment."""
        return [e for e in self.entries if e.sentiment == sentiment]
    
    def get_by_aspect_sentiment(self, aspect: str, sentiment: str) -> List[MemoryEntry]:
        """Get entries matching both aspect and sentiment."""
        return [e for e in self.entries if e.aspect == aspect and e.sentiment == sentiment]
    
    def get_all_phrases_by_sentiment(self, sentiment: str) -> List[str]:
        """Get all evidence phrases for a sentiment (for backward compatibility)."""
        phrases = []
        for entry in self.entries:
            if entry.sentiment == sentiment:
                phrases.extend(entry.evidence)
        return list(set(phrases))  # deduplicate
    
    def get_all_phrases(self) -> List[str]:
        """Get all evidence phrases across all sentiments."""
        phrases = []
        for entry in self.entries:
            phrases.extend(entry.evidence)
        return list(set(phrases))
    
    def to_json(self) -> Dict[str, Any]:
        """Export memory to JSON format."""
        return {
            "entries": [
                {
                    "aspect": e.aspect,
                    "sentiment": e.sentiment,
                    "evidence": e.evidence,
                    "text_id": e.text_id,
                    "confidence": e.confidence,
                    "has_embedding": e.embedding is not None,
                    "cold_start": e.cold_start
                }
                for e in self.entries
            ],
            "total_entries": len(self.entries),
            "by_sentiment": {
                "POS": len([e for e in self.entries if e.sentiment == "POS"]),
                "NEG": len([e for e in self.entries if e.sentiment == "NEG"]),
                "NEU": len([e for e in self.entries if e.sentiment == "NEU"])
            }
        }


def normalize_phrase(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


# Global embedding model cache
_EMBEDDING_MODEL = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get or initialize the global embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[Warning] Could not load sentence-transformers: {e}")
            _EMBEDDING_MODEL = False
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL else None


def compute_text_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Compute sentence embedding for text using sentence-transformers."""
    model = get_embedding_model(model_name)
    if model is None:
        return None
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        return None


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def topk_by_similarity(query_emb: np.ndarray, entries: List[MemoryEntry], k: int) -> List[Tuple[MemoryEntry, float]]:
    """Retrieve top-k memory entries by cosine similarity to query embedding."""
    candidates = []
    for entry in entries:
        if entry.embedding is not None:
            sim = cosine_similarity(query_emb, entry.embedding)
            candidates.append((entry, sim))
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]


def sigmoid(z: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-min(max(z, -50), 50)))


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def get_aspect_description(aspect: str) -> str:
    """Generate canonical text description for an aspect."""
    # Simple heuristic: convert aspect name to natural language
    aspect_descriptions = {
        "On-campus Service": "on-campus mental health services and facilities",
        "Counseling Service": "counseling services and support",
        "Mental Health Service": "mental health care and treatment services",
        "Wellness Service": "wellness programs and activities",
        "Therapy Service": "therapy and therapeutic interventions",
        "Hotline Service": "crisis hotline and emergency support",
        "Service Availability": "availability and accessibility of services",
        "General": "general mental health support and services"
    }
    return aspect_descriptions.get(aspect, aspect.lower().replace("-", " "))


def memory_write(memory: Memory, aspect: str, label: str, evidence_phrases: List[str], 
                 text_id: Optional[Any] = None, confidence: float = 0.0, cold_start: bool = False):
    """Write a new entry to memory with aspect-sentiment-evidence structure."""
    if not evidence_phrases:
        return
    
    # Combine evidence phrases for embedding
    combined_evidence = " ".join([ph for ph in evidence_phrases if isinstance(ph, str) and ph.strip()])
    
    # Compute embedding (optional, may be None if library not available)
    embedding = compute_text_embedding(combined_evidence) if combined_evidence else None
    
    memory.add_entry(
        aspect=aspect,
        sentiment=label,
        evidence=evidence_phrases,
        text_id=text_id,
        confidence=confidence,
        embedding=embedding,
        cold_start=cold_start
    )


# -----------------------------
# Utilities
# -----------------------------
_sent_re = re.compile(r"[.!?]+(?:\s+|$)")


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    start = 0
    for m in _sent_re.finditer(text):
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


def find_phrases(text: str, phrases: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ph in phrases:
        p = normalize_phrase(ph)
        if not p:
            continue
        i = text.lower().find(p)
        if i >= 0:
            out.append({"span": [i, i + len(p)], "phrase": text[i:i + len(p)]})
    return out


def parse_candidates_cell(x: Any) -> List[str]:
    """Parse candidates_final cell from JSON string, Python-literal string, or list-like object."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    parsed: Any = x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return []

    if not isinstance(parsed, (list, tuple, set)):
        return []

    out: List[str] = []
    seen = set()
    for item in parsed:
        c = str(item).strip()
        if c and c in ASPECTS_8 and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_json_with_fallback(raw_text: str) -> Any:
    """
    Parse model output as JSON with a lightweight fallback.
   
    Fast path parses the whole string. Fallback strips markdown fences and
    decodes the first JSON object/array if extra text is present.
    """
    s = (raw_text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    candidates: List[str] = []
    if s:
        candidates.append(s)

    decoder = json.JSONDecoder()
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if starts:
        start = min(starts)
        snippet = s[start:]
        candidates.append(snippet)
        try:
            obj, _ = decoder.raw_decode(snippet)
            return obj
        except json.JSONDecodeError:
            pass

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





# -----------------------------
# Model loading
# -----------------------------
@dataclass
class SentimentModels:
    llm_client: Optional[Any] = None
    llm_model: Optional[str] = None


class LLMSentimentClassifier:
    """LLM-based sentiment classifier using Google GenAI API."""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-genai is required. Install with: pip install google-genai")
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=resolved_api_key)
        self.model = model
        self.temperature = temperature

    def _chat(self, system: str, user: str) -> str:
        config = {
            "temperature": self.temperature,
            "system_instruction": sanitize_text_for_api(system),
            "response_mime_type": "application/json",
        }
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(**config)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=sanitize_text_for_api(user),
            config=config,
        )
        return google_text_from_response(resp)

    def infer_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:
        """
        Classify sentiment for aspect in text.
        Returns: (label: POS/NEG/NEU, confidence: 0.0-1.0)
        """
        system = """You are a sentiment classification model for student mental health feedback.

    Task:
    Given:
    1. one student feedback comment
    2. one target aspect

    Predict the sentiment toward that target aspect.

    Valid sentiment labels:
    - NEG
    - NEU
    - POS

    Rules:
    1. Classify sentiment only for the given target aspect.
    2. Return exactly one label from NEG, NEU, POS.
    3. Return JSON only.
    """

        user = f"""Target aspect:
    {aspect}

    Student feedback:
    {text}

    Respond ONLY with JSON: {{"label": "POS|NEG|NEU", "confidence": 0.0-1.0}}
    """
        
        try:
            response = self._chat(system, user)
            data = parse_json_with_fallback(response)
            label = str(data.get("label", "NEU")).upper()
            confidence = float(data.get("confidence", 0.5))
            
            if label not in {"POS", "NEG", "NEU"}:
                label = "NEU"
            confidence = max(0.0, min(1.0, confidence))
            
            return label, confidence
        except Exception as e:
            print(f"[Warning] LLM inference error: {e}. Falling back to lexicon.")
            return "NEU", 0.33

    def infer_sentiment_with_evidence(self, text: str, aspect: str, feedback: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Classify sentiment and extract supporting evidence spans.
        Returns: (label: POS/NEG/NEU, confidence: 0.0-1.0, evidence_spans: List[{span, phrase}])
        """
        system = """You are a sentiment classification model for student mental health feedback.

    Task:
    Given:
    1. one student feedback comment
    2. one target aspect

    Predict the sentiment toward that target aspect.

    Valid sentiment labels:
    - NEG
    - NEU
    - POS

    Rules:
    1. Classify sentiment only for the given target aspect.
    2. Return exactly one label from NEG, NEU, POS.
    3. Return JSON only.
    """
        
        feedback_text = ""
        if feedback:
            feedback_text = "\n\nPREVIOUS ITERATION FEEDBACK (improve on these issues):\n"
            for i, fb in enumerate(feedback, 1):
                feedback_text += f"\nIteration {i}:\n"
                # original spans and reasons
                spans = fb.get("spans", [])
                reasons = fb.get("reasons", [])
                if spans:
                    feedback_text += "  Issues with evidence:\n"
                    for j, span in enumerate(spans):
                        reason = reasons[j] if j < len(reasons) else "Issue detected"
                        feedback_text += f"    - '{span}': {reason}\n"

                # detailed validator feedback (if any)
                validator_feedback = fb.get("validator_feedback", {})
                if validator_feedback:
                    evidence_relevance = validator_feedback.get("evidence_aspect_relevance", {})
                    if evidence_relevance:
                        feedback_text += "  Evidence-Aspect Relevance:\n"
                        for span, rel_info in evidence_relevance.items():
                            if isinstance(rel_info, dict) and rel_info.get("warning"):
                                feedback_text += f"    - '{span}': {rel_info.get('message', 'Not relevant')}\n"

                    llm_conf = validator_feedback.get("llm_confidence", {})
                    if isinstance(llm_conf, dict) and llm_conf.get("warning"):
                        feedback_text += f"  LLM Confidence Issue: {llm_conf.get('warning')}\n"

                    suggested = validator_feedback.get("suggested_label", {})
                    if isinstance(suggested, dict) and suggested.get("label") and suggested.get("label") != fb.get("suggested_label", "?"):
                        feedback_text += f"  Validator suggests considering: {suggested.get('label')} (confidence: {suggested.get('best_label_score', 0):.2f})\n"

                suggested_label = fb.get("suggested_label")
                if suggested_label:
                    feedback_text += f"  Validator's Suggested Label: {suggested_label}\n"

                feedback_text += "\n"

            feedback_text += "Based on these feedbacks, please choose DIFFERENT evidence and/or reconsider the sentiment label.\n"
        
        user = f"""Target aspect:
    {aspect}

    Student feedback:
    {text}

    Identify and extract 1-3 key phrases or sentences that support your sentiment classification.
    These phrases MUST be exact excerpts from the given text.{feedback_text}

    Respond ONLY with JSON: {{"label": "POS|NEG|NEU", "confidence": 0.0-1.0, "evidence_spans": ["phrase 1", "phrase 2", ...]}}
    """
        
        try:
            response = self._chat(system, user)
            data = parse_json_with_fallback(response)
            label = str(data.get("label", "NEU")).upper()
            confidence = float(data.get("confidence", 0.5))
            evidence_spans = data.get("evidence_spans", [])
            
            if label not in {"POS", "NEG", "NEU"}:
                label = "NEU"
            confidence = max(0.0, min(1.0, confidence))
            
            # Convert evidence spans to the required format with actual text positions
            evidence = find_phrases(text, evidence_spans) if evidence_spans else []
            
            return label, confidence, evidence
        except Exception as e:
            print(f"[Warning] LLM inference error: {e}. Falling back to defaults.")
            return "NEU", 0.33, []
    
    # def validate_evidence_consistency_multi_signal(self, aspect: str, label: str, evidence_phrases: List[str],
    #                                               memory: Memory, hyperparams: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    #     """
    #     Multi-signal validation using memory retrieval, aspect relevance, and sentiment consistency.
    #     Implements the pseudocode from the design document.
        
    #     Returns: (S_c: validation_score 0.0-1.0, diagnostics: dict with all signals)
    #     """

    #     # Hyperparameters for fusion
    #     if hyperparams is None:
    #         hyperparams = {
    #             "K": 12,
    #             "tau": 0.07,
    #             "sim_min": 0.25,
    #             "alpha": 2.0,
    #             "beta": 1.0,
    #             "gamma": 1.2,
    #             "delta": 0.8,
    #             "bias": -1.0
    #         }
        
    #     K = int(hyperparams["K"])
    #     tau = hyperparams["tau"]
    #     sim_min = hyperparams["sim_min"]
    #     alpha = hyperparams["alpha"]
    #     beta = hyperparams["beta"]
    #     gamma = hyperparams["gamma"]
    #     delta = hyperparams["delta"]
    #     bias = hyperparams["bias"]
        
    #     # Combine evidence phrases
    #     evidence_text = " ".join(evidence_phrases[:3])  # limit to top 3
    #     if not evidence_text.strip():
    #         return 0.0, {"reason": "empty_evidence", "S_c": 0.0}
        
    #     # 0) Embed query evidence
    #     query_emb = compute_text_embedding(evidence_text)
    #     if query_emb is None:
    #         # Fallback to simple LLM validation if embeddings not available
    #         return self._validate_fallback_llm(aspect, label, evidence_phrases, memory)
        
    #     # 1) Retrieve from memory (prefer non-cold-start entries; fallback to cold-start if none)
    #     all_entries = memory.get_by_aspect(aspect)
    #     aspect_entries = [e for e in all_entries if not e.cold_start]
    #     if not aspect_entries:
    #         aspect_entries = [e for e in all_entries if e.cold_start]
    #     if not aspect_entries:
    #         # No memory for this aspect yet - validate based on aspect relevance and sentiment consistency only
    #         # Use a moderate default score that can still pass reasonable thresholds
    #         aspect_emb = compute_text_embedding(get_aspect_description(aspect))
    #         if aspect_emb is not None:
    #             S_rel_raw = cosine_similarity(query_emb, aspect_emb)
    #             S_rel = 0.5 * (S_rel_raw + 1.0)
    #         else:
    #             S_rel = 0.5
            
    #         S_sent, S_sent_margin = self._compute_sentiment_consistency(evidence_text, aspect, label)
            
    #         # For cold-start (no memory), rely on aspect relevance + sentiment consistency
    #         # Use a more lenient score to bootstrap the memory
    #         S_c = sigmoid(beta * S_rel + gamma * S_sent + bias - 0.5)  # slight boost for cold-start
            
    #         return S_c, {
    #             "reason": "cold_start_no_memory_for_aspect",
    #             "S_c": float(S_c),
    #             "S_mem_margin": 0.0,
    #             "S_rel": float(S_rel),
    #             "S_sent": float(S_sent),
    #             "S_sent_margin": float(S_sent_margin),
    #             "avg_sim": 0.0,
    #             "feedback_tags": [],
    #             "cold_start": True
    #         }
        
    #     topk = topk_by_similarity(query_emb, aspect_entries, K)
    #     if not topk:
    #         # Memory entries exist but no embeddings - fallback to LLM validation
    #         return self._validate_fallback_llm(aspect, label, evidence_phrases, memory)
        
    #     sims = [sim for (_, sim) in topk]
    #     avg_sim = sum(sims) / max(len(sims), 1)
        
    #     # Retrieval reliability
    #     rel_retr = clamp((avg_sim - sim_min) / (1.0 - sim_min + 1e-9), 0.0, 1.0)
        
    #     # 2) Memory consistency score (support margin)
    #     w_raw = [math.exp(sim / tau) for sim in sims]
    #     Z = sum(w_raw) + 1e-12
    #     w = [wi / Z for wi in w_raw]
        
    #     mass = {"POS": 0.0, "NEG": 0.0, "NEU": 0.0}
    #     for idx, (entry, sim) in enumerate(topk):
    #         mass[entry.sentiment] += w[idx]
        
    #     p_y = mass[label]
    #     p_best_other = max([mass[y] for y in ["POS", "NEG", "NEU"] if y != label])
    #     S_mem_margin = clamp(p_y - p_best_other, -1.0, 1.0)
    #     S_mem = 0.5 * (S_mem_margin + 1.0)  # map to [0,1]
        
    #     # 3) Evidence-aspect relevance (semantic alignment)
    #     aspect_text = get_aspect_description(aspect)
    #     aspect_emb = compute_text_embedding(aspect_text)
    #     if aspect_emb is not None:
    #         S_rel_raw = cosine_similarity(query_emb, aspect_emb)
    #         S_rel = 0.5 * (S_rel_raw + 1.0)  # map to [0,1]
    #     else:
    #         S_rel = 0.5  # neutral if can't compute
        
    #     # 4) Evidence sentiment self-consistency
    #     # Use LLM to judge sentiment of evidence independently
    #     S_sent, S_sent_margin = self._compute_sentiment_consistency(evidence_text, aspect, label)
        
    #     # 5) Fuse signals into S_c
    #     mem_term = rel_retr * (alpha * S_mem + delta * rel_retr)
    #     logit = mem_term + beta * S_rel + gamma * S_sent + bias
    #     S_c = sigmoid(logit)
        
    #     # 6) Diagnostics
    #     diagnostics = {
    #         "S_c": float(S_c),
    #         "avg_sim": float(avg_sim),
    #         "rel_retr": float(rel_retr),
    #         "mass": {k: float(v) for k, v in mass.items()},
    #         "S_mem": float(S_mem),
    #         "S_mem_margin": float(S_mem_margin),
    #         "S_rel": float(S_rel),
    #         "S_sent": float(S_sent),
    #         "S_sent_margin": float(S_sent_margin),
    #         "logit": float(logit),
    #         "topk_count": len(topk),
    #         "feedback_tags": []
    #     }
        
    #     # Generate feedback
    #     if S_rel < 0.4:
    #         diagnostics["feedback_tags"].append("evidence_not_about_aspect")
    #     if S_mem_margin < -0.2 and rel_retr > 0.4:
    #         diagnostics["feedback_tags"].append("memory_contradicts_label")
    #     if S_sent < 0.4:
    #         diagnostics["feedback_tags"].append("evidence_sentiment_mismatch")
        
    #     reasoning = f"S_c={S_c:.3f} (mem={S_mem:.2f}, rel={S_rel:.2f}, sent={S_sent:.2f}, avg_sim={avg_sim:.2f})"
    #     if diagnostics["feedback_tags"]:
    #         reasoning += f" | Issues: {', '.join(diagnostics['feedback_tags'])}"
        
    #     return S_c, diagnostics

    def validate_evidence_consistency(self, aspect: str, label: str, evidence_phrases: List[str], 
                                       memory: Memory, hyperparams: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:

        diagnostics = {"feedback_tags": []}
        # New validator score: accumulate cosine similarities per label over top-k memory entries per query span.
        if hyperparams is None:
            hyperparams = {"K": 12}

        K = int(hyperparams.get("K", 12))

        # Prepare query spans (limit to top 3 as before)
        query_spans = [ph.strip() for ph in evidence_phrases if isinstance(ph, str) and ph.strip()][:3]
        if not query_spans:
            return 0.0, {"reason": "empty_evidence", "S_c": 0.0}

        # Retrieve memory entries (prefer non-cold-start entries; fallback to cold-start if none)
        all_entries = memory.get_by_aspect(aspect)
        aspect_entries = [e for e in all_entries if not e.cold_start]
        if not aspect_entries:
            aspect_entries = [e for e in all_entries if e.cold_start]
        if not aspect_entries:
            # Cold-start: no memory for this aspect yet. Use aspect relevance + sentiment consistency.
            evidence_text = " ".join(query_spans)
            query_emb = compute_text_embedding(evidence_text) if evidence_text.strip() else None
            aspect_emb = compute_text_embedding(get_aspect_description(aspect))

            if query_emb is not None and aspect_emb is not None:
                S_rel_raw = cosine_similarity(query_emb, aspect_emb)
                S_rel = 0.5 * (S_rel_raw + 1.0)
            else:
                S_rel = 0.5
            if S_rel < 0.5:
                diagnostics["feedback_tags"].append(f"span_not_relevant_to_aspect: '{evidence_text}'")

            llm_probs = self._compute_sentiment_consistency(evidence_text, aspect) 

            p_y = llm_probs[label]
            p_best_other = max([llm_probs[y] for y in ["POS", "NEG", "NEU"] if y != label])
            S_sent_margin = clamp(p_y - p_best_other, -1.0, 1.0)
            S_sent = 0.5 * (S_sent_margin + 1.0)

            if S_sent < 0.5:
                diagnostics["feedback_tags"].append("evidence_sentiment_mismatch")

            beta = float(hyperparams.get("beta", 1.0))
            gamma = float(hyperparams.get("gamma", 1.2))
            bias = float(hyperparams.get("bias", -1.0))
            S_c = sigmoid(beta * S_rel + gamma * S_sent + bias - 0.5)

            diagnostics.update({
                "reason": "cold_start_no_memory_for_aspect",
                "S_c": float(S_c),
                "S_rel": float(S_rel),
                "S_sent": float(S_sent),
                "S_sent_margin": float(S_sent_margin),
                "avg_sim": 0.0,
                "topk_count": 0,
                "cold_start": True,
            })

            return S_c, diagnostics
        
        label_sums = {"POS": 0.0, "NEU": 0.0, "NEG": 0.0}
        label_counts = {"POS": 0, "NEU": 0, "NEG": 0}
        total_retrieved = 0
        sims_accum: List[float] = []

        aspect_emb = compute_text_embedding(get_aspect_description(aspect))
        S_rels: Dict[any, float] = {}

        for span in query_spans:
            query_emb = compute_text_embedding(span)
            if query_emb is None:
                continue
            S_rel = 1.0
            if aspect_emb is not None:
                S_rel_raw = cosine_similarity(query_emb, aspect_emb)
                S_rel = clamp(0.5 * (S_rel_raw + 1.0), 0.0, 1.0)
            S_rels[span] = float(S_rel)
            if S_rel < 0.5:
                diagnostics["feedback_tags"].append(f"span_not_relevant_to_aspect: '{span}'")
            topk = topk_by_similarity(query_emb, aspect_entries, K)
            if not topk:
                continue
            for entry, sim in topk:
                if entry.sentiment in label_sums:
                    # S_rel体现query span与aspect的相关性（该query是否可靠？），sim体现了query span与memory entry的相关性，两者乘积作为对该entry的支持度贡献
                    label_sums[entry.sentiment] += float(sim) * S_rel
                    label_counts[entry.sentiment] += 1
                sims_accum.append(sim)
            total_retrieved += len(topk)

        if total_retrieved == 0:
            return 0.0, {"reason": "no_embeddings_for_memory_or_queries", "S_c": 0.0, "topk_count": 0}

        avg_sim = sum(sims_accum) / max(len(sims_accum), 1)
        eps = 1e-6
        label_scores = {
            k: (label_sums[k] / (label_counts[k] + eps))
            for k in label_sums
        }

        # Similarity support per label (margin vs best other label), mapped to [-1, 1]
        sim_support = {}
        for k in label_scores:
            other_best = max([v for kk, v in label_scores.items() if kk != k])
            sim_support[k] = clamp(float(label_scores[k] - other_best), -1.0, 1.0)

        # LLM support per label (margin vs best other label), mapped to [-1, 1]
        evidence_text = " ".join(query_spans)
        llm_probs = self._compute_sentiment_consistency(evidence_text, aspect)

        llm_support = {}
        for k in llm_probs:
            other_best = max([v for kk, v in llm_probs.items() if kk != k])
            llm_support[k] = clamp(float(llm_probs[k] - other_best), -1.0, 1.0)
        
        if llm_support.get(label, 0.0) < 0.5:
            diagnostics["feedback_tags"].append("evidence_sentiment_mismatch")

        # Fuse supports into a single strength signal per label
        alpha = float(hyperparams.get("alpha", 0.6))
        combined_support = {
            k: clamp(alpha * sim_support.get(k, 0.0) + (1.0 - alpha) * llm_support.get(k, 0.0), -1.0, 1.0)
            for k in label_scores
        }

        best_label = max(combined_support.items(), key=lambda kv: kv[1])[0]
        proposed_support = float(combined_support.get(label, 0.0))

        # Gate score in [0, 1]
        S_c = 0.5 * (proposed_support + 1.0)

        diagnostics.update({
            "S_c": float(S_c),
            "avg_sim": float(avg_sim),
            "topk_count": int(total_retrieved),
            "spans_used": list(query_spans),
            "S_rels": S_rels,
            "label_sums": {k: float(v) for k, v in label_sums.items()},
            "label_counts": {k: int(v) for k, v in label_counts.items()},
            "label_scores": {k: float(v) for k, v in label_scores.items()},
            "sim_support": {k: float(v) for k, v in sim_support.items()},
            "llm_support": {k: float(v) for k, v in llm_support.items()},
            "combined_support": {k: float(v) for k, v in combined_support.items()},
            "best_label": best_label,
            "proposed_support": proposed_support,
            "alpha": alpha,
            "method": "sim_llm_margin_fusion"
        })

        return S_c, diagnostics
    
    def _compute_sentiment_consistency(self, evidence: str, aspect: str):
        """Use LLM to judge if evidence expresses the proposed sentiment."""
        system = "You are a sentiment classifier. Judge the sentiment expressed in the text regarding an aspect. Return ONLY JSON."
        user = f"""Text: {evidence}
Aspect: {aspect}

What sentiment does this text express about the aspect?
Return JSON: {{"POS": 0.0-1.0, "NEG": 0.0-1.0, "NEU": 0.0-1.0}} (probabilities summing to ~1.0)"""
        
        try:
            response = self._chat(system, user)
            data = parse_json_with_fallback(response)
            p_pos = float(data.get("POS", 0.33))
            p_neg = float(data.get("NEG", 0.33))
            p_neu = float(data.get("NEU", 0.33))
            
            # Normalize
            total = p_pos + p_neg + p_neu + 1e-9
            probs = {"POS": p_pos/total, "NEG": p_neg/total, "NEU": p_neu/total}

            return probs

        except Exception as e:
            print(f"[Warning] Sentiment consistency check failed: {e}")
            return {"POS": 1.0 / 3.0, "NEG": 1.0 / 3.0, "NEU": 1.0 / 3.0}
    
    def _validate_fallback_llm(self, aspect: str, label: str, evidence_phrases: List[str], memory: Memory) -> Tuple[float, Dict[str, Any]]:
        """Fallback validation when embeddings not available."""
        relevant_entries = memory.get_by_aspect_sentiment(aspect, label)
        mem_context = ""
        if relevant_entries:
            top_entries = sorted(relevant_entries, key=lambda e: e.confidence, reverse=True)[:3]
            mem_phrases = []
            for entry in top_entries:
                mem_phrases.extend(entry.evidence[:2])
            mem_context = f"Previously memorized {label} phrases for '{aspect}': {', '.join(mem_phrases[:5])}"
        else:
            mem_context = f"No memorized {label} patterns for '{aspect}' yet."
        
        evidence_text = "\n".join(f"- {ph}" for ph in evidence_phrases)
        system = "You are a sentiment validation expert. Return ONLY JSON with 'score' (0.0-1.0) and 'reasoning'."
        user = f"""Aspect: {aspect}\nLabel: {label}\nEvidence:\n{evidence_text}\n\n{mem_context}\n\nHow well does evidence support the label? JSON: {{\"score\": 0.0-1.0, \"reasoning\": \"...\"}}"""
        
        try:
            response = self._chat(system, user)
            data = parse_json_with_fallback(response)
            score = float(data.get("score", 0.5))
            reasoning = str(data.get("reasoning", ""))
            return clamp(score, 0.0, 1.0), {"S_c": score, "reasoning": reasoning, "method": "fallback_llm"}
        except Exception:
            return 0.5, {"S_c": 0.5, "method": "fallback_error"}


def load_models(models_dir: str, llm_model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> SentimentModels:
    """Initialize LLM client for sentiment classification."""
    llm_client = None
    
    try:
        llm_client = LLMSentimentClassifier(model=llm_model, api_key=api_key)
        print(f"[Info] Using LLM ({llm_model}) for sentiment classification")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    return SentimentModels(llm_client=llm_client, llm_model=llm_model)


# -----------------------------
# Agents
# -----------------------------
@dataclass
class Pred:
    label: str
    score: float
    evidence: List[Dict[str, Any]]
    rationale: str


@dataclass
class Verdict:
    score: float
    reasons: List[str]
    counter_evidence: List[Dict[str, Any]]
    suggested_fix: Dict[str, List[str]]
    cold_start: bool = False


def infer_with_head(models: SentimentModels, aspect: str, text: str) -> Tuple[str, float]:
    """Use LLM for sentiment classification."""
    if models.llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return models.llm_client.infer_sentiment(text, aspect)


def infer_with_head_and_evidence(models: SentimentModels, aspect: str, text: str, feedback: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Use LLM for sentiment classification and extract supporting evidence spans."""
    if models.llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return models.llm_client.infer_sentiment_with_evidence(text, aspect, feedback)


def proposer_propose(text: str, aspect: str, memory: Memory, models: SentimentModels, prev_feedback: Optional[List[Dict[str, Any]]] = None) -> Pred:
    """Sentiment Proposer: Direct LLM inference on full text, with feedback from previous iterations."""
    # Direct LLM inference on full text, with feedback from previous iterations
    label, score, evidence = infer_with_head_and_evidence(models, aspect, text, feedback=prev_feedback)
    
    rationale = f"Direct LLM inference on full text."
    if prev_feedback:
        rationale += f" Feedback from {len(prev_feedback)} previous iteration(s) applied."

    return Pred(label=label, score=score, evidence=evidence, rationale=rationale)


def validator_validate(text: str, aspect: str, pred: Pred, memory: Memory, models: SentimentModels) -> Verdict:
    """
    Sentiment Validator: Multi-signal validation using memory consistency, aspect relevance, sentiment consistency.
    Implements the multi-signal S_c formula from the design document.
    """
    reasons: List[str] = []

    if not pred.evidence:
        reasons.append("No evidence spans in text; reject.")
        return Verdict(score=0.0, reasons=reasons, counter_evidence=[], suggested_fix={"focus_phrases": [], "avoid_phrases": []})

    # Extract key phrases from evidence
    evidence_phrases = [e.get("phrase", "") for e in pred.evidence if e.get("phrase")]
    
    if not evidence_phrases:
        reasons.append("Evidence spans have no phrases; reject.")
        return Verdict(score=0.0, reasons=reasons, counter_evidence=[], suggested_fix={"focus_phrases": [], "avoid_phrases": []})
    
    if models.llm_client is None:
        print("[Warning] LLM client not available, using default validation score.")
        validator_score = 0.5
        diagnostics = {"method": "no_llm"}
    else:
        validator_score, diagnostics = models.llm_client.validate_evidence_consistency(
            aspect, pred.label, evidence_phrases, memory
        )
    cold_start = bool(diagnostics.get("cold_start", False)) if isinstance(diagnostics, dict) else False
    
    # Build informative reason string from diagnostics
    if "reasoning" in diagnostics:
        reasons.append(diagnostics["reasoning"])
    else:
        reason_parts = []
        if "S_mem_margin" in diagnostics:
            reason_parts.append(f"mem_margin={diagnostics['S_mem_margin']:.2f}")
        if "S_rel" in diagnostics:
            reason_parts.append(f"rel={diagnostics['S_rel']:.2f}")
        if "S_sent" in diagnostics:
            reason_parts.append(f"sent={diagnostics['S_sent']:.2f}")
        if "avg_sim" in diagnostics:
            reason_parts.append(f"sim={diagnostics['avg_sim']:.2f}")
        if reason_parts:
            reasons.append(f"Multi-signal: {', '.join(reason_parts)}")
    
    # Use feedback tags for suggested fixes
    suggest_focus = evidence_phrases[:3]
    if "feedback_tags" in diagnostics:
        if "memory_contradicts_label" in diagnostics["feedback_tags"]:
            reasons.append("Warning: Memory suggests different sentiment")
        if "evidence_not_about_aspect" in diagnostics["feedback_tags"]:
            reasons.append("Warning: Evidence may not relate to aspect")
        if "evidence_sentiment_mismatch" in diagnostics["feedback_tags"]:
            reasons.append("Warning: Evidence sentiment unclear, may not support label")
        if "span_not_relevant_to_aspect" in diagnostics["feedback_tags"]:
            reasons.append("Warning: Some evidence spans may not be relevant to aspect, consider focusing on other relevant spans")

    return Verdict(
        score=validator_score,
        reasons=reasons,
        counter_evidence=[],
        suggested_fix={"focus_phrases": suggest_focus},
        cold_start=cold_start,
    )


def gate_accept(pred: Pred, verdict: Verdict, tau_sent: float) -> bool:
    return verdict.score >= tau_sent


# -----------------------------
# Pipeline
# -----------------------------

def SVR_sentiment_pipeline_rows(
    rows: List[Dict[str, Any]],
    models: SentimentModels,
    T_max: int = 3,
    tau_sent: float = 0.5,
    disable_validator: bool = False,
    disable_memory: bool = False,
):
    """S-IPV Sentiment Pipeline: Sentiment Proposer -> Sentiment Validator -> Refine loop."""
    M = Memory()
    outputs: List[Dict[str, Any]] = []
    stats = {"total": len(rows), "with_aspects": 0, "empty_aspects": 0, "processed_aspects": 0}

    for row in tqdm(rows, total=len(rows), desc="Iterative P2V"):
        text = str(row.get(TEXT_COL, "") or "")
        aspects_raw = row.get(GATED_COL)
        aspects = parse_candidates_cell(aspects_raw)

        sentiment_map: Dict[str, Dict[str, Any]] = {}
        validator_iterations: Dict[str, List[Dict[str, Any]]] = {}

        # Handle empty aspects
        if not aspects or len(aspects) == 0:
            stats["empty_aspects"] += 1
            outputs.append({
                SAMPLE_ID_COL: row.get(SAMPLE_ID_COL, None),
                TEXT_COL: text,
                GATED_COL: [],
                "sentiment_final": {},
                "validator_iterations": {},
                "has_aspects": False,
            })
            continue

        stats["with_aspects"] += 1

        for aspect in aspects:
            if disable_validator:
                pred = proposer_propose(text, aspect, M, models, prev_feedback=None)
                sentiment_map[aspect] = {"label": pred.label, "evidence": pred.evidence, "score": pred.score}
                validator_iterations[aspect] = []
                if not disable_memory:
                    evidence_phrases = [e.get("phrase", "") for e in pred.evidence]
                    text_id = row.get(SAMPLE_ID_COL, None)
                    memory_write(M, aspect, pred.label, evidence_phrases, text_id, pred.score, cold_start=False)
                continue

            feedback: List[Dict[str, Any]] = []  # list of iteration feedback dicts
            accepted = False
            best_candidate: Optional[Dict[str, Any]] = None

            for t in range(int(T_max)):
                pred = proposer_propose(text, aspect, M, models, prev_feedback=feedback if feedback else None)
                validation_memory = M if not disable_memory else Memory()
                verdict = validator_validate(text, aspect, pred, validation_memory, models)

                best_candidate_score = best_candidate["verdict"].score if best_candidate else -1.0
                if (best_candidate is None) or (verdict.score > best_candidate_score):
                    best_candidate = {"pred": pred, "verdict": verdict}

                gate_pass = gate_accept(pred, verdict, tau_sent)
                validator_iterations.setdefault(aspect, []).append({
                    "iter": t + 1,
                    "label": pred.label,
                    "evidence": pred.evidence,
                    "score": verdict.score,
                    "accepted": gate_pass,
                    "reasons": verdict.reasons,
                    "cold_start": verdict.cold_start,
                })

                if gate_pass:
                    label = pred.label
                    evidence = pred.evidence
                    sentiment_map[aspect] = {"label": label, "evidence": evidence, "score": verdict.score}
                    # Write to memory with full context
                    if not disable_memory:
                        evidence_phrases = [e.get("phrase", "") for e in evidence]
                        text_id = row.get(SAMPLE_ID_COL, None)
                        memory_write(M, aspect, label, evidence_phrases, text_id, verdict.score, cold_start=verdict.cold_start)
                    accepted = True
                    break
                
                # Accumulate feedback from this iteration
                if best_candidate:
                    iter_focus = best_candidate["verdict"].suggested_fix.get("focus_phrases", []) if best_candidate["verdict"].suggested_fix else []
                    iter_reasons = best_candidate["verdict"].reasons or []
                    # Skip first reason (usually summary), use reasons[1:]
                    iter_reasons_filtered = iter_reasons[1:] if len(iter_reasons) > 1 else iter_reasons
                    
                    feedback.append({
                        "spans": list(iter_focus),
                        "reasons": list(iter_reasons_filtered)
                    })

            if not accepted:
                sentiment_map[aspect] = {
                    "label": "UNSURE",
                    "evidence": best_candidate["pred"].evidence if best_candidate else [],
                    "score": best_candidate["verdict"].score if best_candidate else 0.0
                }

        stats["processed_aspects"] += len(sentiment_map)
        outputs.append({
            SAMPLE_ID_COL: row.get(SAMPLE_ID_COL, None),
            TEXT_COL: text,
            "merged_entities": row.get("merged_entities", []),
            "gold_struct": row.get("gold_struct", None),
            GATED_COL: aspects,
            "sentiment_final": sentiment_map,
            "validator_iterations": validator_iterations,
            "has_aspects": True,
        })

    return outputs, M, stats


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Memory-Augmented S-IPV Sentiment Inference with LLM")
    ap.add_argument("--in_csv", required=True, help="A-IPV output CSV")
    ap.add_argument("--out_csv", default="outputs/SIPV_output.csv")
    ap.add_argument("--out_memory", default="outputs/sentiment_memory.json", help="Output JSON for memory buckets")
    
    # S-IPV parameters
    ap.add_argument("--t_max", type=int, default=3, help="Max iterations per (x, aspect)")
    ap.add_argument("--tau_sent", type=float, default=0.5, help="Fixed threshold for all aspect sentiments")
    ap.add_argument("--disable_validator", action="store_true", help="Ablation mode: skip sentiment validator and output proposer result directly")
    ap.add_argument("--disable_memory", action="store_true", help="Ablation mode: disable memory read/write during sentiment validation")
    ap.add_argument(
        "--missing_policy",
        type=str,
        default="wrong",
        choices=["wrong", "skip"],
        help="How to handle missing sentiment predictions during evaluation",
    )
    
    # LLM options
    ap.add_argument("--llm_model", default="gemini-2.5-flash", help="Google model name (e.g., gemini-2.5-flash, gemini-2.5-pro)")
    ap.add_argument("--google_api_key", default=None, help="Google API key (default: GOOGLE_API_KEY or GEMINI_API_KEY env var)")
    
    args = ap.parse_args()

    if os.path.dirname(args.out_csv):
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if os.path.dirname(args.out_memory):
        os.makedirs(os.path.dirname(args.out_memory), exist_ok=True)

    df = pd.read_csv(args.in_csv)
    assert TEXT_COL in df.columns, f"Missing column '{TEXT_COL}'"
    assert GATED_COL in df.columns, f"Missing column '{GATED_COL}' (run gate_by_threshold.py first)"

    models = load_models(
        models_dir="",
        llm_model=args.llm_model,
        api_key=args.google_api_key
    )

    rows = df.to_dict(orient="records")
    print(f"Inferencing on {len(df)} samples with threshold={args.tau_sent:.3f}...")
    results, memory, pipeline_stats = SVR_sentiment_pipeline_rows(
        rows,
        models,
        T_max=args.t_max,
        tau_sent=args.tau_sent,
        disable_validator=args.disable_validator,
        disable_memory=args.disable_memory,
    )

    out_df = pd.DataFrame(results)
    out_df["sentiment_final"] = out_df["sentiment_final"].apply(lambda m: json.dumps(m, ensure_ascii=False))
    out_df["validator_iterations"] = out_df["validator_iterations"].apply(lambda m: json.dumps(m, ensure_ascii=False))
    out_df.to_csv(args.out_csv, index=False)

    with open(args.out_memory, "w", encoding="utf-8") as f:
        json.dump(memory.to_json(), f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {args.out_csv} ({len(out_df)} rows)")
    print(f"    Threshold: {args.tau_sent}")
    print("    Columns: sample_id, text, merged_entities, gold_struct, candidates_final, sentiment_final, validator_iterations, has_aspects")
    print(f"[OK] wrote {args.out_memory}")

    if "gold_struct" in out_df.columns:
        summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(
            out_df,
            gold_col="gold_struct",
            pred_col="sentiment_final",
            missing_policy=args.missing_policy,
        )

        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        per_label_file = os.path.splitext(args.out_csv)[0] + "_per_label.csv"
        conf_file = os.path.splitext(args.out_csv)[0] + "_confusion.csv"
        per_aspect_file = os.path.splitext(args.out_csv)[0] + "_per_aspect.csv"

        per_label_df.to_csv(per_label_file, index=False)
        conf_df.to_csv(conf_file)
        per_aspect_df.to_csv(per_aspect_file, index=False)

        print(f"Saved per-label results to {per_label_file}")
        print(f"Saved confusion matrix to {conf_file}")
        print(f"Saved per-aspect results to {per_aspect_file}")
    else:
        print("[Warn] gold_struct not found in output; skipping S-IPV evaluation.")

    print("Pipeline Stats:")
    print(f"  Total samples: {pipeline_stats['total']}")
    print(f"  Samples with aspects: {pipeline_stats['with_aspects']}")
    print(f"  Samples with empty aspects: {pipeline_stats['empty_aspects']}")
    print(f"  Total aspects processed: {pipeline_stats['processed_aspects']}")


if __name__ == "__main__":
    main()
