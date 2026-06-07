"""
chatabsa.py

ChatABSA prompting baseline for AWARE-SMILE.

Adopted ChatABSA:
  - constrained aspect category prompt
  - strict JSON output
  - zero/few-shot in-context examples
  - optional TF-IDF retrieval for few-shot examples
  - label normalization and JSON parsing

Modes:
  joint:
    predict aspect-sentiment pairs in one LLM call.
  two_stage:
    predict aspects first, then classify sentiment for predicted or gold aspects.

Examples:
  export OPENAI_API_KEY=...
  python chatabsa.py \
    --data ../datasets/dataset.csv \
    --split 0.8 0.1 0.1 \
    --output_dir runs/chatabsa \
    --provider openai \
    --model gpt-4o-mini \
    --mode joint \
    --shots 5 \
    --retrieval_shots

Gold-aspect sentiment evaluation:
  python chatabsa.py \
    --data ../datasets/dataset.csv \
    --output_dir runs/chatabsa_gold_sp \
    --provider openai \
    --model gpt-4o-mini \
    --mode two_stage \
    --sentiment_aspects gold \
    --shots 5 \
    --retrieval_shots
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# ensure project root is on sys.path so ASQP_MA package imports work when running from subfolders
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ASQP_MA.evaluation_SIPV import eval_s_ipv_sentiment_conditional


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if s in {"pos", "positive"}:
        return "positive"
    if s in {"neg", "negative"}:
        return "negative"
    if s in {"neu", "neutral"}:
        return "neutral"
    if s in {"mix", "mixed", "conflict", "conflicting"}:
        return "mixed"
    if s in {"none", "null", "nan", "absent", "no sentiment"}:
        return "none"
    return s


def canonical_aspect(a: Any) -> str:
    return str(a).strip()


def resolve_aspect_labels(aspect_to_labels: Dict[str, List[str]]) -> Dict[str, str]:
    priority = {"negative": 5, "positive": 4, "mixed": 3, "neutral": 2, "none": 0}
    out = {}
    for asp, labels in aspect_to_labels.items():
        labs = [normalize_label(x) for x in labels if normalize_label(x) != "none"]
        if not labs:
            continue
        counts = Counter(labs)
        out[asp] = sorted(counts.items(), key=lambda kv: (kv[1], priority.get(kv[0], 1)), reverse=True)[0][0]
    return out


def parse_gold_from_row(row: pd.Series, gold_col: str, merged_col: str) -> Dict[str, str]:
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


def infer_sentiments(dfs: Sequence[pd.DataFrame]) -> List[str]:
    labels = set()
    for df in dfs:
        for d in df["_gold_by_aspect"]:
            labels.update(normalize_label(v) for v in d.values() if normalize_label(v) != "none")
    preferred = ["negative", "positive", "neutral", "mixed"]
    out = [x for x in preferred if x in labels]
    out += sorted(labels - set(out))
    return out


def load_data(args):
    if args.data:
        df = attach_gold(pd.read_csv(args.data), args.text_col, args.gold_col, args.merged_col)
        train_ratio, dev_ratio, test_ratio = args.split
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=args.seed, shuffle=True)
        rel_test = test_ratio / (dev_ratio + test_ratio)
        dev_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=args.seed, shuffle=True)
    else:
        train_df = attach_gold(pd.read_csv(args.train), args.text_col, args.gold_col, args.merged_col)
        dev_df = attach_gold(pd.read_csv(args.dev), args.text_col, args.gold_col, args.merged_col)
        test_df = attach_gold(pd.read_csv(args.test), args.text_col, args.gold_col, args.merged_col)
    return train_df, dev_df, test_df


class FewShotSelector:
    def __init__(self, train_df: pd.DataFrame, text_col: str, shots: int, retrieval: bool, seed: int):
        self.train_df = train_df.reset_index(drop=True)
        self.text_col = text_col
        self.shots = shots
        self.retrieval = retrieval
        self.rng = random.Random(seed)
        self.vectorizer = None
        self.train_matrix = None
        if retrieval and shots > 0:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
            self.train_matrix = self.vectorizer.fit_transform(self.train_df[text_col].astype(str).tolist())

    def select(self, query_text: str) -> pd.DataFrame:
        if self.shots <= 0:
            return self.train_df.iloc[[]]
        if self.retrieval and self.vectorizer is not None:
            q = self.vectorizer.transform([query_text])
            sims = cosine_similarity(q, self.train_matrix).reshape(-1)
            idx = np.argsort(-sims)[: self.shots]
            return self.train_df.iloc[idx]
        idx = self.rng.sample(range(len(self.train_df)), k=min(self.shots, len(self.train_df)))
        return self.train_df.iloc[idx]


def gold_pairs_to_json(gold: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"aspect": a, "sentiment": s} for a, s in sorted(gold.items())]


def build_joint_prompt(text: str, examples: pd.DataFrame, aspects: List[str], sentiments: List[str], text_col: str) -> str:
    lines = [
        "Aspect-Based Sentiment Analysis requires identifying aspect categories mentioned in a text and determining the sentiment toward each aspect.",
        f"The aspect category must be selected only from this list: [{', '.join(aspects)}].",
        f"The sentiment must be selected only from this list: [{', '.join(sentiments)}].",
        "Return only a valid JSON list. Each item must have exactly two fields: aspect and sentiment.",
        "If no aspect category is mentioned, return an empty list: [].",
        "Do not output explanations.",
    ]
    if len(examples) > 0:
        lines.append("\nIn-context examples:")
        for _, row in examples.iterrows():
            lines.append(f'Text: {json.dumps(str(row[text_col]), ensure_ascii=False)}')
            lines.append(f'Output: {json.dumps(gold_pairs_to_json(row["_gold_by_aspect"]), ensure_ascii=False)}')
    lines.append("\nQuery:")
    lines.append(f'Text: {json.dumps(text, ensure_ascii=False)}')
    lines.append("Output:")
    return "\n".join(lines)


def build_aspect_prompt(text: str, examples: pd.DataFrame, aspects: List[str], text_col: str) -> str:
    lines = [
        "Aspect category detection requires identifying all aspect categories mentioned in a text.",
        f"The aspect category must be selected only from this list: [{', '.join(aspects)}].",
        "Return only a valid JSON object with one field: aspects.",
        'The value of aspects must be a list of category names. If no aspect category is mentioned, return {"aspects": []}.',
        "Do not output explanations.",
    ]
    if len(examples) > 0:
        lines.append("\nIn-context examples:")
        for _, row in examples.iterrows():
            lines.append(f'Text: {json.dumps(str(row[text_col]), ensure_ascii=False)}')
            lines.append(f'Output: {json.dumps({"aspects": sorted(row["_gold_by_aspect"].keys())}, ensure_ascii=False)}')
    lines.append("\nQuery:")
    lines.append(f'Text: {json.dumps(text, ensure_ascii=False)}')
    lines.append("Output:")
    return "\n".join(lines)


def build_sentiment_prompt(text: str, aspects_to_label: List[str], examples: pd.DataFrame, sentiments: List[str], text_col: str) -> str:
    lines = [
        "Aspect-conditioned sentiment classification requires determining the sentiment toward each given aspect category in the text.",
        f"The sentiment must be selected only from this list: [{', '.join(sentiments)}].",
        "Return only a valid JSON list. Each item must have exactly two fields: aspect and sentiment.",
        "Do not add new aspects. Only classify the given aspects.",
        "Do not output explanations.",
    ]
    if len(examples) > 0:
        lines.append("\nIn-context examples:")
        for _, row in examples.iterrows():
            pairs = gold_pairs_to_json(row["_gold_by_aspect"])
            if pairs:
                lines.append(f'Text: {json.dumps(str(row[text_col]), ensure_ascii=False)}')
                lines.append(f'Given aspects: {json.dumps([p["aspect"] for p in pairs], ensure_ascii=False)}')
                lines.append(f'Output: {json.dumps(pairs, ensure_ascii=False)}')
    lines.append("\nQuery:")
    lines.append(f'Text: {json.dumps(text, ensure_ascii=False)}')
    lines.append(f'Given aspects: {json.dumps(aspects_to_label, ensure_ascii=False)}')
    lines.append("Output:")
    return "\n".join(lines)


def call_openai(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict information extraction system. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You are a strict information extraction system. Return only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join([block.text for block in resp.content if getattr(block, "type", None) == "text"])


def call_google(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        },
    )
    return resp.text or ""


def call_llm(args, prompt: str) -> str:
    last_err = None
    for attempt in range(args.max_retries):
        try:
            if args.provider == "openai":
                return call_openai(args.model, prompt, args.temperature, args.max_tokens)
            if args.provider == "anthropic":
                return call_anthropic(args.model, prompt, args.temperature, args.max_tokens)
            if args.provider == "google":
                return call_google(args.model, prompt, args.temperature, args.max_tokens)
            raise ValueError(args.provider)
        except Exception as e:
            last_err = e
            wait = args.retry_wait * (attempt + 1)
            print(f"[WARN] LLM call failed ({attempt+1}/{args.max_retries}): {e}. Waiting {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def extract_json(text: str) -> Any:
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        start, end = s.find(open_ch), s.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            c = s[start:end+1]
            try:
                return json.loads(c)
            except Exception:
                try:
                    return ast.literal_eval(c)
                except Exception:
                    pass
    return None


def normalize_aspect(pred: Any, aspects: List[str]) -> str | None:
    if pred is None:
        return None
    s = str(pred).strip()
    if not s:
        return None
    lookup = {a.lower(): a for a in aspects}
    if s.lower() in lookup:
        return lookup[s.lower()]
    norm = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    norm_lookup = {re.sub(r"[^a-z0-9]+", " ", a.lower()).strip(): a for a in aspects}
    return norm_lookup.get(norm)


def parse_joint_prediction(raw: str, aspects: List[str], sentiments: List[str]) -> Dict[str, str]:
    obj = extract_json(raw)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("predictions"), list):
            items = obj["predictions"]
        else:
            items = [obj]
    else:
        items = []

    out = {}
    valid_sents = set(sentiments)
    for item in items:
        if not isinstance(item, dict):
            continue
        asp = item.get("aspect", item.get("aspect_category", item.get("category")))
        sent = item.get("sentiment", item.get("sentiment_polarity", item.get("polarity")))
        asp = normalize_aspect(asp, aspects)
        sent = normalize_label(sent)
        if asp is not None and sent in valid_sents:
            out[asp] = sent
    return out


def parse_aspect_prediction(raw: str, aspects: List[str]) -> List[str]:
    obj = extract_json(raw)
    if isinstance(obj, dict):
        vals = obj.get("aspects", [])
        if isinstance(vals, str):
            vals = [vals]
    elif isinstance(obj, list):
        vals = obj
    else:
        vals = []
    out = []
    for v in vals:
        if isinstance(v, dict):
            v = v.get("aspect", v.get("aspect_category", v.get("category")))
        asp = normalize_aspect(v, aspects)
        if asp and asp not in out:
            out.append(asp)
    return out


def make_multilabel_matrix(dicts: Sequence[Dict[str, str]], aspects: List[str]) -> np.ndarray:
    idx = {a: i for i, a in enumerate(aspects)}
    y = np.zeros((len(dicts), len(aspects)), dtype=int)
    for i, d in enumerate(dicts):
        for a in d:
            if a in idx:
                y[i, idx[a]] = 1
    return y


def eval_aspects(gold_dicts, pred_dicts, aspects):
    y_true = make_multilabel_matrix(gold_dicts, aspects)
    y_pred = make_multilabel_matrix(pred_dicts, aspects)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true.reshape(-1), y_pred.reshape(-1), average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "aspect_micro_precision": float(micro_p),
        "aspect_micro_recall": float(micro_r),
        "aspect_micro_f1": float(micro_f1),
        "aspect_macro_precision": float(macro_p),
        "aspect_macro_recall": float(macro_r),
        "aspect_macro_f1": float(macro_f1),
        "aspect_weighted_precision": float(weighted_p),
        "aspect_weighted_recall": float(weighted_r),
        "aspect_weighted_f1": float(weighted_f1),
        "aspect_subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "gold_positive_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
    }


def eval_sentiment_gold_aspects(gold_dicts, pred_dicts, sentiments):
    y_true, y_pred = [], []
    valid = set(sentiments)
    for gold, pred in zip(gold_dicts, pred_dicts):
        for asp, gsent in gold.items():
            if gsent not in valid:
                continue
            psent = pred.get(asp, "__missing__")
            if psent not in valid:
                psent = "__missing__"
            y_true.append(gsent)
            y_pred.append(psent)

    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=sentiments, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=sentiments, average="weighted", zero_division=0)
    return {
        "sent_acc_gold_aspects": acc,
        "sent_macro_precision": float(p),
        "sent_macro_recall": float(r),
        "sent_macro_f1": float(f1),
        "sent_weighted_precision": float(wp),
        "sent_weighted_recall": float(wr),
        "sent_weighted_f1": float(wf1),
        "gold_sent_dist": dict(Counter(y_true)),
        "pred_sent_dist": dict(Counter(y_pred)),
    }


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_data(args)

    aspects = [canonical_aspect(x) for x in json.loads(Path(args.aspects_json).read_text(encoding="utf-8"))] if args.aspects_json else infer_aspects([train_df])
    sentiments = [normalize_label(x) for x in json.loads(Path(args.sentiments_json).read_text(encoding="utf-8"))] if args.sentiments_json else infer_sentiments([train_df])

    (out_dir / "aspects.json").write_text(json.dumps(aspects, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sentiments.json").write_text(json.dumps(sentiments, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Provider:", args.provider)
    print("Model:", args.model)
    print("Mode:", args.mode)
    print("Shots:", args.shots, "Retrieval:", args.retrieval_shots)
    print("Aspects:", aspects)
    print("Sentiments:", sentiments)

    selector = FewShotSelector(train_df, args.text_col, args.shots, args.retrieval_shots, args.seed)
    eval_df = test_df.reset_index(drop=True)
    if args.limit:
        eval_df = eval_df.iloc[:args.limit].copy()

    rows, gold_dicts, pred_dicts = [], [], []

    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Running ChatABSA"):
        text = str(row[args.text_col])
        gold = row["_gold_by_aspect"]
        examples = selector.select(text)

        if args.mode == "joint":
            prompt = build_joint_prompt(text, examples, aspects, sentiments, args.text_col)
            raw = call_llm(args, prompt)
            pred = parse_joint_prediction(raw, aspects, sentiments)
            ap_raw, sp_raw = "", raw
        else:
            ap_prompt = build_aspect_prompt(text, examples, aspects, args.text_col)
            ap_raw = call_llm(args, ap_prompt)
            pred_aspects = parse_aspect_prediction(ap_raw, aspects)
            aspects_to_label = sorted(gold.keys()) if args.sentiment_aspects == "gold" else pred_aspects
            if aspects_to_label:
                sp_prompt = build_sentiment_prompt(text, aspects_to_label, examples, sentiments, args.text_col)
                sp_raw = call_llm(args, sp_prompt)
                pred = parse_joint_prediction(sp_raw, aspects, sentiments)
                if args.sentiment_aspects == "predicted":
                    pred = {a: s for a, s in pred.items() if a in set(pred_aspects)}
            else:
                sp_raw, pred = "[]", {}

        gold_dicts.append(gold)
        pred_dicts.append(pred)
        rows.append({
            "idx": i,
            "text": text,
            "gold": json.dumps(gold, ensure_ascii=False),
            "pred": json.dumps(pred, ensure_ascii=False),
            "ap_raw": ap_raw,
            "sp_raw": sp_raw,
        })

        if (len(rows) % args.save_every) == 0:
            pd.DataFrame(rows).to_csv(out_dir / "predictions.partial.csv", index=False)
        if args.sleep > 0:
            time.sleep(args.sleep)

    aspect_metrics = eval_aspects(gold_dicts, pred_dicts, aspects)

    # Build DataFrame compatible with eval_s_ipv_sentiment_conditional
    # map labels (e.g. 'positive' -> 'POS') so evaluator recognizes them
    label_map_to_sipv = {
        "negative": "NEG",
        "positive": "POS",
        "neutral": "NEU",
        "mixed": "MIXED",
        "neg": "NEG",
        "pos": "POS",
        "neu": "NEU",
        "mix": "MIXED",
    }

    rows = []
    for gold, pred in zip(gold_dicts, pred_dicts):
        # gold and pred are dicts: aspect -> sentiment (normalized like 'positive')
        by_aspect = {asp: {"label": label_map_to_sipv.get(str(lab).lower(), lab)} for asp, lab in gold.items()}
        gold_struct = {"aspect_set": list(gold.keys()), "by_aspect": by_aspect}
        pred_struct = {}
        for asp in set(list(gold.keys()) + list(pred.keys())):
            lab = pred.get(asp, None)
            mapped = label_map_to_sipv.get(str(lab).lower(), lab) if lab is not None else None
            pred_struct[asp] = {"label": mapped}
        rows.append({"gold_struct": gold_struct, "sentiment_final": pred_struct})

    df_eval = pd.DataFrame(rows)
    summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(df_eval)

    sent_metrics = {
        "sent_acc_gold_aspects": float(summary.get("accuracy", 0.0)),
        "sent_micro_precision": float(summary.get("micro_precision", 0.0)),
        "sent_micro_recall": float(summary.get("micro_recall", 0.0)),
        "sent_micro_f1": float(summary.get("micro_f1", 0.0)),
        "sent_macro_precision": None,
        "sent_macro_recall": None,
        "sent_macro_f1": float(summary.get("macro_f1", 0.0)),
        "sent_weighted_precision": None,
        "sent_weighted_recall": None,
        "sent_weighted_f1": None,
        "gold_sent_dist": per_label_df.set_index("label")["support"].to_dict() if not per_label_df.empty else {},
        "pred_sent_dist": conf_df.sum(axis=0).to_dict() if not conf_df.empty else {},
    }

    final = {
        "model": f"ChatABSA/{args.provider}/{args.model}",
        "mode": args.mode,
        "shots": args.shots,
        "retrieval_shots": args.retrieval_shots,
        "sentiment_aspects": args.sentiment_aspects,
        "aspect": aspect_metrics,
        "sentiment_gold_aspects": sent_metrics,
    }

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("TEST")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--dev", type=str, default=None)
    p.add_argument("--test", type=str, default=None)
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--gold_col", type=str, default="gold_struct")
    p.add_argument("--merged_col", type=str, default="merged_entities")
    p.add_argument("--aspects_json", type=str, default=None)
    p.add_argument("--sentiments_json", type=str, default=None)

    p.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_wait", type=float, default=3.0)
    p.add_argument("--sleep", type=float, default=0.0)

    p.add_argument("--mode", choices=["joint", "two_stage"], default="joint")
    p.add_argument("--sentiment_aspects", choices=["predicted", "gold"], default="predicted")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--retrieval_shots", action="store_true")

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data and not (args.train and args.dev and args.test):
        raise SystemExit("Provide either --data or all of --train --dev --test.")
    run(args)
