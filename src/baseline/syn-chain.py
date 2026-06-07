#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
syn_chain.py

Syn-Chain baseline for AWARE-SMILE gold-aspect conditional sentiment prediction.

It adapts Syn-Chain to SP:
  Step 1: use spaCy dependency parse and ask LLM to analyze syntax related to the aspect.
  Step 2: ask LLM to extract the user's opinion toward the aspect.
  Step 3: ask LLM to classify sentiment toward the aspect.

Install:
  pip install pandas numpy scikit-learn tqdm openai spacy
  python -m spacy download en_core_web_sm

Example:
  export OPENAI_API_KEY=...
  python syn_chain.py \
    --data ../datasets/dataset.csv \
    --split 0.8 0.1 0.1 \
    --output_dir runs/syn_chain \
    --provider openai \
    --model gpt-4o-mini
"""

from __future__ import annotations

import argparse, ast, json, os, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# ensure project root on sys.path so ASQP_MA package imports work when running from subfolders
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
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
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


def make_examples(df: pd.DataFrame, text_col: str, sentiments: List[str]) -> List[Dict[str, Any]]:
    out, valid = [], set(sentiments)
    for row_idx, row in df.reset_index(drop=True).iterrows():
        for aspect, label in row["_gold_by_aspect"].items():
            label = normalize_label(label)
            if label in valid:
                out.append({"row_idx": row_idx, "text": str(row[text_col]), "aspect": aspect, "gold": label})
    return out


def load_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError as e:
        raise SystemExit("spaCy is not installed. Install it with: pip install spacy") from e
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise SystemExit(f"spaCy model {model_name} is not installed. Run: python -m spacy download {model_name}") from e


def doc_to_conllu_like(doc) -> str:
    rows = ["ID\tTEXT\tLEMMA\tPOS\tTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"]
    for i, tok in enumerate(doc, start=1):
        head = tok.head.i + 1 if tok.head is not tok else 0
        rows.append("\t".join([
            str(i), tok.text, tok.lemma_, tok.pos_, tok.tag_, "_",
            str(head), tok.dep_, "_", "_"
        ]))
    return "\n".join(rows)


CONLLU_EXPLANATION = (
    "Each row represents a token. HEAD indicates the syntactic head token ID, and DEPREL describes "
    "the dependency relation between the current token and its head."
)


def system_step1(words: int) -> str:
    return f"You analyze syntactic dependency information for ABSA. Ensure accuracy. Reply within {words} words."


def system_step2(words: int) -> str:
    return f"You extract the user's opinion toward a given aspect for ABSA. Ensure accuracy. Reply within {words} words."


def system_step3(sentiments: List[str]) -> str:
    return (
        "You are a sentiment analysis expert. Classify the sentiment toward the given aspect. "
        f"The sentiment must be one of: {', '.join(sentiments)}. "
        "Return exactly one JSON object with fields sentiment and reason."
    )


def build_step1_prompt(text: str, aspect: str, conllu: str) -> str:
    return (
        f"Sentence: {json.dumps(text, ensure_ascii=False)}\n\n"
        f"CoNLL-U-like dependency information:\n{conllu}\n\n"
        f"{CONLLU_EXPLANATION}\n\n"
        f"Based on the syntactic dependency information, analyze information related to the aspect {json.dumps(aspect, ensure_ascii=False)}."
    )


def build_step2_prompt(text: str, aspect: str, r1: str) -> str:
    return (
        f"Sentence: {json.dumps(text, ensure_ascii=False)}\n\n"
        f"Syntactic analysis related to the aspect:\n{r1}\n\n"
        f"Considering the context and information related to {json.dumps(aspect, ensure_ascii=False)}, "
        f"what is the speaker or user's opinion toward this aspect?"
    )


def build_step3_prompt(text: str, aspect: str, r2: str, sentiments: List[str]) -> str:
    return (
        f"Sentence: {json.dumps(text, ensure_ascii=False)}\n"
        f"Aspect: {json.dumps(aspect, ensure_ascii=False)}\n\n"
        f"Opinion analysis:\n{r2}\n\n"
        f"Based on common sense and the user's opinion, what is the sentiment polarity toward the aspect?\n"
        f"Return only JSON, for example: {{\"sentiment\": \"positive\", \"reason\": \"...\"}}."
    )


def build_direct_prompt(text: str, aspect: str, sentiments: List[str]) -> str:
    return (
        f"Aspect-conditioned sentiment classification.\n"
        f"The sentiment must be one of: {', '.join(sentiments)}.\n"
        f"Text: {json.dumps(text, ensure_ascii=False)}\n"
        f"Aspect: {json.dumps(aspect, ensure_ascii=False)}\n"
        f"Return only JSON: {{\"sentiment\": \"positive|negative|neutral|mixed\", \"reason\": \"...\"}}."
    )


def call_openai(model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join([b.text for b in resp.content if getattr(b, "type", None) == "text"])


def call_google(model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    m = genai.GenerativeModel(model, system_instruction=system)
    resp = m.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
    return resp.text or ""


def call_llm(args, system: str, prompt: str, max_tokens: int | None = None) -> str:
    max_tokens = max_tokens or args.max_tokens
    last_err = None
    for attempt in range(args.max_retries):
        try:
            if args.provider == "openai":
                return call_openai(args.model, system, prompt, args.temperature, max_tokens)
            if args.provider == "anthropic":
                return call_anthropic(args.model, system, prompt, args.temperature, max_tokens)
            if args.provider == "google":
                return call_google(args.model, system, prompt, args.temperature, max_tokens)
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
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = s[start:end+1]
        try:
            return json.loads(frag)
        except Exception:
            try:
                return ast.literal_eval(frag)
            except Exception:
                return None
    return None


def parse_sentiment(raw: str, sentiments: List[str]) -> str:
    valid = set(sentiments)
    obj = extract_json(raw)
    if isinstance(obj, dict):
        y = normalize_label(obj.get("sentiment", obj.get("polarity", obj.get("label"))))
        if y in valid:
            return y

    low = raw.lower()
    patterns = [
        r"sentiment(?: polarity)?(?: toward [^.\n]+)?(?: is|:)\s*(positive|negative|neutral|mixed)",
        r'"sentiment"\s*:\s*"(positive|negative|neutral|mixed)"',
        r"\b(positive|negative|neutral|mixed)\b",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            y = normalize_label(m.group(1))
            if y in valid:
                return y
    return "__parse_error__"


def evaluate(gold: List[str], pred: List[str], sentiments: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(gold, pred) if gold else 0.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(gold, pred, labels=sentiments, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(gold, pred, labels=sentiments, average="weighted", zero_division=0)
    return {
        "sent_acc_gold_aspects": float(acc),
        "sent_macro_precision": float(macro_p),
        "sent_macro_recall": float(macro_r),
        "sent_macro_f1": float(macro_f1),
        "sent_weighted_precision": float(weighted_p),
        "sent_weighted_recall": float(weighted_r),
        "sent_weighted_f1": float(weighted_f1),
        "gold_sent_dist": dict(Counter(gold)),
        "pred_sent_dist": dict(Counter(pred)),
    }


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, _, test_df = load_data(args)
    if args.sentiments_json:
        sentiments = [normalize_label(x) for x in json.loads(Path(args.sentiments_json).read_text(encoding="utf-8"))]
    else:
        sentiments = infer_sentiments([train_df])

    print("Provider:", args.provider)
    print("Model:", args.model)
    print("Mode:", args.mode)
    print("Sentiments:", sentiments)

    nlp = None if args.mode == "direct" else load_spacy_model(args.spacy_model)
    examples = make_examples(test_df, args.text_col, sentiments)
    if args.limit:
        examples = examples[: args.limit]

    rows, gold_labels, pred_labels = [], [], []
    parse_cache: Dict[str, str] = {}

    for i, ex in tqdm(enumerate(examples), total=len(examples), desc="Running Syn-Chain"):
        text, aspect, gold = ex["text"], ex["aspect"], ex["gold"]
        r1 = r2 = r3 = ""

        if args.mode == "direct":
            r3 = call_llm(args, system_step3(sentiments), build_direct_prompt(text, aspect, sentiments), args.step3_tokens)
        else:
            if text not in parse_cache:
                parse_cache[text] = doc_to_conllu_like(nlp(text))
            conllu = parse_cache[text]

            r1 = call_llm(args, system_step1(args.step1_words), build_step1_prompt(text, aspect, conllu), args.step1_tokens)
            if args.mode == "syn_1_3":
                r2 = r1
            else:
                r2 = call_llm(args, system_step2(args.step2_words), build_step2_prompt(text, aspect, r1), args.step2_tokens)
            r3 = call_llm(args, system_step3(sentiments), build_step3_prompt(text, aspect, r2, sentiments), args.step3_tokens)

        pred = parse_sentiment(r3, sentiments)
        gold_labels.append(gold)
        pred_labels.append(pred)
        rows.append({
            "idx": i,
            "row_idx": ex["row_idx"],
            "text": text,
            "aspect": aspect,
            "gold": gold,
            "pred": pred,
            "step1_syntax_analysis": r1,
            "step2_opinion_analysis": r2,
            "step3_sentiment_output": r3,
        })

        if len(rows) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "predictions.partial.csv", index=False)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Compute weighted metrics using original normalized labels
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=sentiments, average="weighted", zero_division=0
    )

    # Build DataFrame compatible with eval_s_ipv_sentiment_conditional
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

    eval_rows = []
    for r in rows:
        asp = r["aspect"]
        gt = r["gold"]
        pr = r["pred"]
        gt_mapped = label_map_to_sipv.get(str(gt).lower(), gt)
        pr_mapped = label_map_to_sipv.get(str(pr).lower(), pr) if pr is not None else None
        gold_struct = {"aspect_set": [asp], "by_aspect": {asp: {"label": gt_mapped}}}
        pred_struct = {asp: {"label": pr_mapped}}
        eval_rows.append({"gold_struct": gold_struct, "sentiment_final": pred_struct})

    df_eval = pd.DataFrame(eval_rows)
    summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(df_eval)

    sent_metrics = {
        "sent_acc_gold_aspects": float(summary.get("accuracy", 0.0)),
        "sent_micro_precision": float(summary.get("micro_precision", 0.0)),
        "sent_micro_recall": float(summary.get("micro_recall", 0.0)),
        "sent_micro_f1": float(summary.get("micro_f1", 0.0)),
        "sent_macro_precision": None,
        "sent_macro_recall": None,
        "sent_macro_f1": float(summary.get("macro_f1", 0.0)),
        "sent_weighted_precision": float(weighted_p),
        "sent_weighted_recall": float(weighted_r),
        "sent_weighted_f1": float(weighted_f1),
        "gold_sent_dist": per_label_df.set_index("label")["support"].to_dict() if not per_label_df.empty else dict(Counter(gold_labels)),
        "pred_sent_dist": conf_df.sum(axis=0).to_dict() if not conf_df.empty else dict(Counter(pred_labels)),
    }

    final = {"model": f"Syn-Chain/{args.provider}/{args.model}", "mode": args.mode, "sentiments": sentiments, "metrics": sent_metrics}

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sentiments.json").write_text(json.dumps(sentiments, indent=2, ensure_ascii=False), encoding="utf-8")

    print("TEST")
    print(json.dumps(final, indent=2, ensure_ascii=False))


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
    p.add_argument("--sentiments_json", type=str, default=None)

    p.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--step1_tokens", type=int, default=350)
    p.add_argument("--step2_tokens", type=int, default=250)
    p.add_argument("--step3_tokens", type=int, default=200)
    p.add_argument("--step1_words", type=int, default=200)
    p.add_argument("--step2_words", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_wait", type=float, default=3.0)
    p.add_argument("--sleep", type=float, default=0.0)

    p.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    p.add_argument("--mode", choices=["syn_chain", "syn_1_3", "direct"], default="syn_chain")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data and not (args.train and args.dev and args.test):
        raise SystemExit("Provide either --data or all of --train --dev --test.")
    run(args)
