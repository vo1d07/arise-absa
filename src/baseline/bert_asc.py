"""
bert_asc.py

Core idea:
  For each (text, aspect), construct an auxiliary sentence grounded in the text:
      aspect seed words -> semantic candidate tokens -> syntactic/opinion modifiers
  Then train a sentence-pair encoder:
      [CLS] auxiliary_sentence [SEP] text [SEP]
  to predict sentiment labels, including "none" for aspect absence.

Designed for AWARE-SMILE CSV files with either:
  - gold_struct column:
      {"aspect_set": [...], "by_aspect": {"Service Availability": "Negative", ...}}
    or
      {"by_aspect": {"Service Availability": {"label": "Negative", ...}}}
  - merged_entities column:
      "[{'aspects': [...], 'sentiments': [...], ...}, ...]"

Example:
  python bert_asc.py \
    --train train.csv --dev dev.csv --test test.csv \
    --text_col text \
    --model_name microsoft/deberta-v3-base \
    --output_dir runs/bert_asc \
    --epochs 5 --batch_size 16 --max_length 256

With only a single CSV file:
  python bert_asc.py --data all.csv --split 0.8 0.1 0.1 --output_dir runs/bert_asc

Notes:
  - It replaces L-LDA seed extraction with a transparent supervised PMI-style seed extractor.
  - It uses the encoder input embedding space for semantic token matching.
  - spaCy is optional. With spaCy installed, dependency modifiers are used; otherwise a window-based fallback is used.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_parse_obj(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip()
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


def normalize_label(x: Any) -> str:
    if x is None:
        return "none"
    s = str(x).strip()
    if not s:
        return "none"
    low = s.lower()
    if low in {"pos", "positive", "positve"}:
        return "positive"
    if low in {"neg", "negative"}:
        return "negative"
    if low in {"neu", "neutral"}:
        return "neutral"
    if low in {"conflict", "conflicting"}:
        return "conflict"
    if low in {"none", "null", "nan", "no", "absent"}:
        return "none"
    return low


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", str(text).lower())


def canonical_aspect(a: Any) -> str:
    return str(a).strip()


def load_spacy(model: str = "en_core_web_sm"):
    try:
        import spacy
        try:
            return spacy.load(model)
        except Exception:
            return spacy.blank("en")
    except Exception:
        return None


# -----------------------------
# Gold parsing
# -----------------------------

def parse_gold_from_row(
    row: pd.Series,
    gold_col: str = "gold_struct",
    merged_col: str = "merged_entities",
) -> Dict[str, str]:
    """
    Return {aspect: sentiment_label}. If multiple labels occur for one aspect,
    choose the majority label; ties prefer negative > positive > neutral > conflict.
    """
    aspect_to_labels: Dict[str, List[str]] = defaultdict(list)

    if gold_col in row and not pd.isna(row[gold_col]):
        obj = safe_parse_obj(row[gold_col])
        if isinstance(obj, dict):
            by_aspect = obj.get("by_aspect")
            if isinstance(by_aspect, dict):
                for asp, val in by_aspect.items():
                    asp = canonical_aspect(asp)
                    if isinstance(val, dict):
                        lab = val.get("label", val.get("sentiment", val.get("polarity")))
                    elif isinstance(val, list):
                        lab = val[0] if val else None
                    else:
                        lab = val
                    aspect_to_labels[asp].append(normalize_label(lab))
                return resolve_aspect_labels(aspect_to_labels)

            aspect_set = obj.get("aspect_set")
            if isinstance(aspect_set, list):
                for asp in aspect_set:
                    aspect_to_labels[canonical_aspect(asp)].append("positive")
                return resolve_aspect_labels(aspect_to_labels)

    if merged_col in row and not pd.isna(row[merged_col]):
        ents = safe_parse_obj(row[merged_col])
        if isinstance(ents, list):
            for ent in ents:
                if not isinstance(ent, dict):
                    continue
                aspects = ent.get("aspects", [])
                sentiments = ent.get("sentiments", [])
                if isinstance(aspects, str):
                    aspects = [aspects]
                if isinstance(sentiments, str):
                    sentiments = [sentiments]
                if not sentiments:
                    sentiments = ["positive"] * len(aspects)
                for i, asp in enumerate(aspects):
                    lab = sentiments[i] if i < len(sentiments) else sentiments[-1]
                    aspect_to_labels[canonical_aspect(asp)].append(normalize_label(lab))

    return resolve_aspect_labels(aspect_to_labels)


def resolve_aspect_labels(aspect_to_labels: Dict[str, List[str]]) -> Dict[str, str]:
    priority = {"negative": 4, "positive": 3, "neutral": 2, "conflict": 1, "none": 0}
    out = {}
    for asp, labels in aspect_to_labels.items():
        labels = [normalize_label(x) for x in labels if normalize_label(x) != "none"]
        if not labels:
            continue
        counts = Counter(labels)
        out[asp] = sorted(counts.items(), key=lambda kv: (kv[1], priority.get(kv[0], 0)), reverse=True)[0][0]
    return out


def attach_gold(df: pd.DataFrame, text_col: str, gold_col: str, merged_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str)
    df["_gold_by_aspect"] = df.apply(lambda r: parse_gold_from_row(r, gold_col, merged_col), axis=1)
    return df


def infer_aspects(dfs: Sequence[pd.DataFrame]) -> List[str]:
    aspects = set()
    for df in dfs:
        for d in df["_gold_by_aspect"]:
            aspects.update(d.keys())
    return sorted(aspects)


# -----------------------------
# Seed extraction
# -----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "i", "you", "he", "she", "they", "we", "my", "your", "our", "their", "me", "him", "her",
    "them", "not", "no", "do", "does", "did", "have", "has", "had", "can", "could", "would", "should",
    "will", "just", "very", "really", "so", "too", "there", "here", "about", "into", "than", "then",
}


def extract_pmi_seeds(
    df: pd.DataFrame,
    aspects: List[str],
    text_col: str,
    top_k: int = 20,
    min_count: int = 2,
) -> Dict[str, List[str]]:
    """
    Supervised PMI-style seed extraction:
      score(token, aspect) = log((count(token in positive aspect docs)+1) / expected)
    This is a practical replacement for L-LDA in the original BERT-ASC paper.
    """
    global_counts = Counter()
    aspect_counts = {a: Counter() for a in aspects}
    aspect_doc_counts = Counter()

    for _, row in df.iterrows():
        toks = [t for t in simple_tokenize(row[text_col]) if t not in STOPWORDS and len(t) > 2]
        unique_toks = set(toks)
        global_counts.update(unique_toks)
        gold = row["_gold_by_aspect"]
        for asp in gold:
            if asp in aspect_counts:
                aspect_counts[asp].update(unique_toks)
                aspect_doc_counts[asp] += 1

    total_docs = max(len(df), 1)
    seeds: Dict[str, List[str]] = {}
    for asp in aspects:
        scored = []
        for tok, c in aspect_counts[asp].items():
            if c < min_count:
                continue
            p_tok_given_asp = (c + 1) / (aspect_doc_counts[asp] + 2)
            p_tok = (global_counts[tok] + 1) / (total_docs + 2)
            score = math.log(p_tok_given_asp / p_tok)
            scored.append((score, c, tok))
        scored.sort(reverse=True)
        chosen = [tok for _, _, tok in scored[:top_k]]
        if not chosen:
            chosen = [w.lower() for w in re.findall(r"[A-Za-z]+", asp) if len(w) > 2]
        seeds[asp] = chosen
    return seeds


def load_or_make_seeds(args, train_df: pd.DataFrame, aspects: List[str]) -> Dict[str, List[str]]:
    if args.seeds_json:
        with open(args.seeds_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {canonical_aspect(k): [str(x).lower() for x in v] for k, v in raw.items()}
    seeds = extract_pmi_seeds(
        train_df,
        aspects,
        args.text_col,
        top_k=args.seed_top_k,
        min_count=args.seed_min_count,
    )
    return seeds


# -----------------------------
# Grounding / auxiliary construction
# -----------------------------

class EmbeddingSimilarity:
    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        emb_layer = encoder.get_input_embeddings()
        self.emb = emb_layer.weight.detach().cpu()
        self.cache: Dict[str, np.ndarray] = {}

    def word_vec(self, word: str) -> Optional[np.ndarray]:
        word = word.lower().strip()
        if not word:
            return None
        if word in self.cache:
            return self.cache[word]
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        vec = self.emb[ids].mean(dim=0).numpy()
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            return None
        vec = vec / norm
        self.cache[word] = vec
        return vec

    def cosine(self, w1: str, w2: str) -> float:
        v1 = self.word_vec(w1)
        v2 = self.word_vec(w2)
        if v1 is None or v2 is None:
            return -1.0
        return float(np.dot(v1, v2))


def get_candidate_tokens(
    text: str,
    aspect: str,
    seeds: List[str],
    sim: EmbeddingSimilarity,
    nlp=None,
    threshold: float = 0.35,
    top_k: int = 3,
) -> List[Tuple[str, int, float]]:
    doc_tokens = []
    if nlp is not None:
        doc = nlp(text)
        for i, tok in enumerate(doc):
            t = tok.text.lower()
            if re.match(r"^[a-z][a-z'\-]*$", t) and t not in STOPWORDS and len(t) > 2:
                doc_tokens.append((t, i))
    else:
        for i, t in enumerate(simple_tokenize(text)):
            if t not in STOPWORDS and len(t) > 2:
                doc_tokens.append((t, i))

    aspect_terms = [w.lower() for w in re.findall(r"[A-Za-z]+", aspect)]
    all_seeds = list(dict.fromkeys([s.lower() for s in seeds] + aspect_terms))

    scored = []
    for tok, idx in doc_tokens:
        best = max((sim.cosine(tok, seed) for seed in all_seeds), default=-1.0)
        exact_bonus = 0.15 if tok in all_seeds else 0.0
        score = best + exact_bonus
        if score >= threshold:
            scored.append((tok, idx, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def extract_modifiers_spacy(text: str, candidate_indices: List[int], nlp) -> List[str]:
    if nlp is None or not candidate_indices:
        return []
    doc = nlp(text)
    mods = []
    cand_set = set(candidate_indices)

    for idx in candidate_indices:
        if idx < 0 or idx >= len(doc):
            continue
        tok = doc[idx]

        for child in tok.children:
            if child.dep_ in {"amod", "advmod"} and child.pos_ in {"ADJ", "ADV", "VERB"}:
                mods.append(child.text.lower())

        head = tok.head
        if tok.dep_ in {"nsubj", "nsubjpass"} and head.pos_ in {"ADJ", "VERB"}:
            mods.append(head.text.lower())

        for child in head.children:
            if child.i != tok.i and child.dep_ in {"acomp", "amod", "advmod"} and child.pos_ in {"ADJ", "ADV", "VERB"}:
                mods.append(child.text.lower())

    seen = set()
    out = []
    for m in mods:
        if m not in STOPWORDS and m not in seen and re.match(r"^[a-z][a-z'\-]*$", m):
            out.append(m)
            seen.add(m)
    return out[:3]


def extract_modifiers_window(text: str, candidates: List[str], window: int = 3) -> List[str]:
    toks = simple_tokenize(text)
    mods = []
    opinionish_suffix = ("able", "ive", "ous", "ful", "less", "ing", "ed", "al")
    for c in candidates:
        for i, t in enumerate(toks):
            if t == c:
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                for w in toks[lo:hi]:
                    if w != c and w not in STOPWORDS and (w.endswith(opinionish_suffix) or len(w) > 5):
                        mods.append(w)
    return list(dict.fromkeys(mods))[:3]


def build_auxiliary_sentence(
    text: str,
    aspect: str,
    seeds: List[str],
    sim: EmbeddingSimilarity,
    nlp=None,
    threshold: float = 0.35,
    top_k: int = 3,
    template: str = "phrase",
) -> str:
    candidates = get_candidate_tokens(
        text=text,
        aspect=aspect,
        seeds=seeds,
        sim=sim,
        nlp=nlp,
        threshold=threshold,
        top_k=top_k,
    )
    cand_words = [c[0] for c in candidates]
    cand_indices = [c[1] for c in candidates]

    if nlp is not None:
        modifiers = extract_modifiers_spacy(text, cand_indices, nlp)
    else:
        modifiers = extract_modifiers_window(text, cand_words)

    phrase_parts = list(dict.fromkeys(cand_words + modifiers))
    if not phrase_parts:
        phrase = aspect
    else:
        phrase = " ".join(phrase_parts)

    if template == "question":
        return f"What is the sentiment of {phrase}?"
    if template == "about":
        return f"What do you think about {phrase}?"
    return phrase


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class PairExample:
    text: str
    aspect: str
    aux: str
    label: str


class ASCPairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        aspects: List[str],
        labels: List[str],
        text_col: str,
        seeds: Dict[str, List[str]],
        sim: EmbeddingSimilarity,
        tokenizer,
        max_length: int,
        nlp=None,
        threshold: float = 0.35,
        top_k: int = 3,
        template: str = "phrase",
        cache_aux: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.examples: List[PairExample] = []

        aux_cache = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building pair examples"):
            text = str(row[text_col])
            gold = row["_gold_by_aspect"]
            for asp in aspects:
                lab = normalize_label(gold.get(asp, "none"))
                key = (text, asp)
                if cache_aux and key in aux_cache:
                    aux = aux_cache[key]
                else:
                    aux = build_auxiliary_sentence(
                        text=text,
                        aspect=asp,
                        seeds=seeds.get(asp, []),
                        sim=sim,
                        nlp=nlp,
                        threshold=threshold,
                        top_k=top_k,
                        template=template,
                    )
                    aux_cache[key] = aux
                self.examples.append(PairExample(text=text, aspect=asp, aux=aux, label=lab))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.aux,
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[ex.label], dtype=torch.long)
        item["aspect"] = ex.aspect
        return item


# -----------------------------
# Model
# -----------------------------

class PairClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        if class_weights is None:
            class_weights = torch.ones(num_labels, dtype=torch.float)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        out = self.encoder(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return {"loss": loss, "logits": logits}


# -----------------------------
# Training / evaluation
# -----------------------------

def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


@torch.no_grad()
def evaluate(model, loader, labels: List[str], device: str) -> Dict[str, Any]:
    model.eval()
    id2label = {i: l for i, l in enumerate(labels)}
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs = batch_to_device(batch, device)
        logits = model(**inputs)["logits"]
        preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
        golds = inputs["labels"].detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(golds)

    y_true_lab = [id2label[i] for i in y_true]
    y_pred_lab = [id2label[i] for i in y_pred]

    acc_all = accuracy_score(y_true_lab, y_pred_lab)

    non_none_idx = [i for i, y in enumerate(y_true_lab) if y != "none"]
    if non_none_idx:
        asc_true = [y_true_lab[i] for i in non_none_idx]
        asc_pred = [y_pred_lab[i] for i in non_none_idx]
        asc_acc = accuracy_score(asc_true, asc_pred)
        asc_micro_p, asc_micro_r, asc_micro_f1, _ = precision_recall_fscore_support(
            asc_true,
            asc_pred,
            labels=[l for l in labels if l != "none"],
            average="micro",
            zero_division=0,
        )
        asc_macro_f1 = f1_score(
            asc_true,
            asc_pred,
            labels=[l for l in labels if l != "none"],
            average="macro",
            zero_division=0,
        )
    else:
        asc_acc = 0.0
        asc_micro_p = 0.0
        asc_micro_r = 0.0
        asc_micro_f1 = 0.0
        asc_macro_f1 = 0.0

    acd_true = [0 if y == "none" else 1 for y in y_true_lab]
    acd_pred = [0 if y == "none" else 1 for y in y_pred_lab]
    p, r, f1, _ = precision_recall_fscore_support(acd_true, acd_pred, average="binary", zero_division=0)

    pred_dist = Counter(y_pred_lab)
    gold_dist = Counter(y_true_lab)

    return {
        "pair_acc_all_labels": acc_all,
        "acd_precision": p,
        "acd_recall": r,
        "acd_f1": f1,
        "asc_acc_gold_aspects": asc_acc,
        "asc_micro_precision_gold_aspects": asc_micro_p,
        "asc_micro_recall_gold_aspects": asc_micro_r,
        "asc_micro_f1_gold_aspects": asc_micro_f1,
        "asc_macro_f1_gold_aspects": asc_macro_f1,
        "gold_non_none_rate": float(np.mean([y != "none" for y in y_true_lab])),
        "pred_non_none_rate": float(np.mean([y != "none" for y in y_pred_lab])),
        "gold_label_dist": dict(gold_dist),
        "pred_label_dist": dict(pred_dist),
    }


def train(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        df = pd.read_csv(args.data)
        df = attach_gold(df, args.text_col, args.gold_col, args.merged_col)
        train_ratio, dev_ratio, test_ratio = args.split
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=args.seed, shuffle=True)
        rel_test = test_ratio / (dev_ratio + test_ratio)
        dev_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=args.seed, shuffle=True)
    else:
        train_df = attach_gold(pd.read_csv(args.train), args.text_col, args.gold_col, args.merged_col)
        dev_df = attach_gold(pd.read_csv(args.dev), args.text_col, args.gold_col, args.merged_col)
        test_df = attach_gold(pd.read_csv(args.test), args.text_col, args.gold_col, args.merged_col)

    if args.aspects_json:
        aspects = json.loads(Path(args.aspects_json).read_text(encoding="utf-8"))
        aspects = [canonical_aspect(a) for a in aspects]
    else:
        aspects = infer_aspects([train_df, dev_df, test_df])

    observed_labels = set()
    for df in [train_df, dev_df, test_df]:
        for d in df["_gold_by_aspect"]:
            observed_labels.update(normalize_label(v) for v in d.values())
    labels = ["none"] + sorted([l for l in observed_labels if l != "none"])

    seeds = load_or_make_seeds(args, train_df, aspects)
    (out_dir / "aspects.json").write_text(json.dumps(aspects, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Pairwise ACD/ASC creates one example for every (text, aspect).
    # The "none" label usually dominates, so unweighted CE can collapse to predicting none.
    tmp_label_counts = Counter()
    for _, row in train_df.iterrows():
        gold = row["_gold_by_aspect"]
        for asp in aspects:
            tmp_label_counts[normalize_label(gold.get(asp, "none"))] += 1

    if args.use_class_weights:
        counts = np.array([tmp_label_counts.get(l, 0) for l in labels], dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        weights_np = counts.sum() / (len(labels) * counts)
        if args.none_weight is not None and "none" in labels:
            weights_np[labels.index("none")] = args.none_weight
        weights = torch.tensor(weights_np, dtype=torch.float)
        print("Class counts:", dict(tmp_label_counts))
        print("Class weights:", {l: float(weights[i]) for i, l in enumerate(labels)})
    else:
        weights = torch.ones(len(labels), dtype=torch.float)

    model = PairClassifier(args.model_name, num_labels=len(labels), dropout=args.dropout, class_weights=weights)

    # Similarity uses the encoder's input embedding space before training.
    sim = EmbeddingSimilarity(tokenizer, model.encoder)

    nlp = None if args.no_spacy else load_spacy(args.spacy_model)
    if nlp is None:
        print("spaCy unavailable or disabled: using window-based modifier fallback.")
    else:
        print("spaCy available: using dependency-style modifier extraction when possible.")

    train_ds = ASCPairDataset(
        train_df, aspects, labels, args.text_col, seeds, sim, tokenizer, args.max_length,
        nlp=nlp, threshold=args.sim_threshold, top_k=args.candidate_top_k,
        template=args.aux_template,
    )
    dev_ds = ASCPairDataset(
        dev_df, aspects, labels, args.text_col, seeds, sim, tokenizer, args.max_length,
        nlp=nlp, threshold=args.sim_threshold, top_k=args.candidate_top_k,
        template=args.aux_template,
    )
    test_ds = ASCPairDataset(
        test_df, aspects, labels, args.text_col, seeds, sim, tokenizer, args.max_length,
        nlp=nlp, threshold=args.sim_threshold, top_k=args.candidate_top_k,
        template=args.aux_template,
    )

    # Save a few generated auxiliary examples for sanity check.
    preview = []
    for ex in train_ds.examples[: min(200, len(train_ds.examples))]:
        if ex.label != "none":
            preview.append({"text": ex.text, "aspect": ex.aspect, "aux": ex.aux, "label": ex.label})
        if len(preview) >= 50:
            break
    pd.DataFrame(preview).to_csv(out_dir / "aux_preview.csv", index=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_dev = -1.0
    best_path = out_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = batch_to_device(batch, device)
            out = model(**inputs)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())

        dev_metrics = evaluate(model, dev_loader, labels, device)
        dev_score = dev_metrics["acd_f1"] + dev_metrics["asc_macro_f1_gold_aspects"]
        row = {"epoch": epoch, "train_loss": total_loss / max(len(train_loader), 1), **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=2))

        if dev_score > best_dev:
            best_dev = dev_score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, labels, device)
    print("TEST")
    print(json.dumps(test_metrics, indent=2))

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    return test_metrics


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--dev", type=str, default=None)
    p.add_argument("--test", type=str, default=None)
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--gold_col", type=str, default="gold_struct")
    p.add_argument("--merged_col", type=str, default="merged_entities")

    p.add_argument("--aspects_json", type=str, default=None)
    p.add_argument("--seeds_json", type=str, default=None)
    p.add_argument("--seed_top_k", type=int, default=20)
    p.add_argument("--seed_min_count", type=int, default=2)

    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--none_weight", type=float, default=0.25)

    p.add_argument("--sim_threshold", type=float, default=0.35)
    p.add_argument("--candidate_top_k", type=int, default=3)
    p.add_argument("--aux_template", type=str, choices=["phrase", "question", "about"], default="phrase")

    p.add_argument("--no_spacy", action="store_true")
    p.add_argument("--spacy_model", type=str, default="en_core_web_sm")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data and not (args.train and args.dev and args.test):
        raise SystemExit("Provide either --data or all of --train --dev --test.")
    train(args)
