"""
xgboost_baseline.py

XGBoost + TF-IDF baseline for SMILE-CASD-style ABSA.

It trains:
  1) Aspect prediction: one binary XGBClassifier per aspect.
  2) Gold-aspect sentiment prediction: one multi-class XGBClassifier.

Example:
  python xgboost_baseline.py \
    --train train.csv --dev dev.csv --test test.csv \
    --output_dir runs/xgboost_tfidf \
    --tune_threshold --use_scale_pos_weight
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from ASQP_MA.evaluation_SIPV import eval_s_ipv_sentiment_conditional

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("xgboost is not installed. Install it with: pip install xgboost") from e


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
    low = str(x).strip().lower()
    if low in {"pos", "positive", "positve"}:
        return "positive"
    if low in {"neg", "negative"}:
        return "negative"
    if low in {"neu", "neutral"}:
        return "neutral"
    if low in {"mix", "mixed", "conflict", "conflicting"}:
        return "mixed"
    if low in {"none", "null", "nan", "no", "absent", ""}:
        return "none"
    return low


def canonical_aspect(a: Any) -> str:
    return str(a).strip()


def resolve_aspect_labels(aspect_to_labels: Dict[str, List[str]]) -> Dict[str, str]:
    priority = {"negative": 5, "positive": 4, "mixed": 3, "neutral": 2, "none": 0}
    out = {}
    for asp, labels in aspect_to_labels.items():
        labels = [normalize_label(x) for x in labels if normalize_label(x) != "none"]
        if not labels:
            continue
        counts = Counter(labels)
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
            for y in d.values():
                y = normalize_label(y)
                if y != "none":
                    labels.add(y)
    preferred = ["negative", "positive", "neutral", "mixed"]
    out = [x for x in preferred if x in labels]
    out += sorted(labels - set(out))
    return out


def make_aspect_matrix(df: pd.DataFrame, aspects: List[str]) -> np.ndarray:
    aspect2id = {a: i for i, a in enumerate(aspects)}
    y = np.zeros((len(df), len(aspects)), dtype=np.int64)
    for i, d in enumerate(df["_gold_by_aspect"]):
        for asp in d:
            if asp in aspect2id:
                y[i, aspect2id[asp]] = 1
    return y


def make_sentiment_examples(df: pd.DataFrame, text_col: str):
    texts, aspects, labels = [], [], []
    for _, row in df.iterrows():
        text = str(row[text_col])
        for asp, lab in row["_gold_by_aspect"].items():
            lab = normalize_label(lab)
            if lab != "none":
                texts.append(text)
                aspects.append(asp)
                labels.append(lab)
    return texts, aspects, labels


def aspect_conditioned_text(texts: Sequence[str], aspects: Sequence[str], template: str) -> List[str]:
    out = []
    for text, asp in zip(texts, aspects):
        if template == "plain":
            prefix = asp
        elif template == "statement":
            prefix = f"The sentiment toward {asp}."
        else:
            prefix = f"What is the sentiment toward {asp}?"
        out.append(prefix + " [SEP] " + str(text))
    return out


def evaluate_aspect(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
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


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    best_t, best_m = 0.5, None
    for t in np.arange(0.05, 0.96, 0.05):
        m = evaluate_aspect(y_true, y_prob, float(t))
        if best_m is None or m["aspect_micro_f1"] > best_m["aspect_micro_f1"]:
            best_t, best_m = float(t), m
    return best_t, best_m


def evaluate_sentiment(y_true: List[int], y_pred: List[int], sentiments: List[str]) -> Dict[str, Any]:
    id2sent = {i: s for i, s in enumerate(sentiments)}
    label_map_to_sipv = {
        "negative": "NEG",
        "positive": "POS",
        "neutral": "NEU",
        "mixed": "MIXED",
    }

    if not y_true:
        return {
            "sent_acc_gold_aspects": 0.0,
            "sent_micro_precision": 0.0,
            "sent_micro_recall": 0.0,
            "sent_micro_f1": 0.0,
            "sent_macro_precision": 0.0,
            "sent_macro_recall": 0.0,
            "sent_macro_f1": 0.0,
            "sent_weighted_precision": 0.0,
            "sent_weighted_recall": 0.0,
            "sent_weighted_f1": 0.0,
            "gold_sent_dist": {},
            "pred_sent_dist": {},
        }

    rows = []
    for gt_idx, pr_idx in zip(y_true, y_pred):
        gt = id2sent[gt_idx]
        pr = id2sent[pr_idx]
        gt_mapped = label_map_to_sipv.get(str(gt).lower(), str(gt).upper())
        pr_mapped = label_map_to_sipv.get(str(pr).lower(), str(pr).upper())
        # The S-IPV evaluator expects a gold_struct with aspect_set and by_aspect.
        # For gold-aspect sentiment evaluation we can use a synthetic aspect key.
        aspect_key = "__aspect__"
        rows.append(
            {
                "gold_struct": {"aspect_set": [aspect_key], "by_aspect": {aspect_key: {"label": gt_mapped}}},
                "sentiment_final": {aspect_key: {"label": pr_mapped}},
            }
        )

    summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(
        pd.DataFrame(rows),
        aspects_list=["__aspect__"],
    )

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "sent_acc_gold_aspects": float(summary.get("accuracy", acc)),
        "sent_micro_precision": float(summary.get("micro_precision", 0.0)),
        "sent_micro_recall": float(summary.get("micro_recall", 0.0)),
        "sent_micro_f1": float(summary.get("micro_f1", 0.0)),
        "sent_macro_precision": None,
        "sent_macro_recall": None,
        "sent_macro_f1": float(summary.get("macro_f1", 0.0)),
        "sent_weighted_precision": float(weighted_p),
        "sent_weighted_recall": float(weighted_r),
        "sent_weighted_f1": float(weighted_f1),
        "gold_sent_dist": per_label_df.set_index("label")["support"].to_dict() if not per_label_df.empty else dict(Counter([id2sent[i] for i in y_true])),
        "pred_sent_dist": conf_df.sum(axis=0).to_dict() if not conf_df.empty else dict(Counter([id2sent[i] for i in y_pred])),
    }


def make_xgb_binary(args, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )


def make_xgb_multiclass(args, num_class: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )


def predict_aspect_probs(models: List[XGBClassifier], X) -> np.ndarray:
    probs = []
    for model in models:
        p = model.predict_proba(X)
        probs.append(p[:, 1] if p.shape[1] > 1 else np.zeros(X.shape[0]))
    return np.vstack(probs).T


def load_data(args):
    if args.data:
        df = attach_gold(pd.read_csv(args.data), args.text_col, args.gold_col, args.merged_col)
        tr, dv, te = args.split
        train_df, temp_df = train_test_split(df, train_size=tr, random_state=args.seed, shuffle=True)
        rel_test = te / (dv + te)
        dev_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=args.seed, shuffle=True)
    else:
        train_df = attach_gold(pd.read_csv(args.train), args.text_col, args.gold_col, args.merged_col)
        dev_df = attach_gold(pd.read_csv(args.dev), args.text_col, args.gold_col, args.merged_col)
        test_df = attach_gold(pd.read_csv(args.test), args.text_col, args.gold_col, args.merged_col)
    return train_df, dev_df, test_df


def main(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_data(args)

    if args.aspects_json:
        aspects = [canonical_aspect(x) for x in json.loads(Path(args.aspects_json).read_text(encoding="utf-8"))]
    else:
        aspects = infer_aspects([train_df, dev_df, test_df])

    if args.sentiments_json:
        sentiments = [normalize_label(x) for x in json.loads(Path(args.sentiments_json).read_text(encoding="utf-8"))]
    else:
        # Use training labels only. XGBoost requires classes seen during training
        # to be encoded as consecutive integers.
        sentiments = infer_sentiments([train_df])

    (out_dir / "aspects.json").write_text(json.dumps(aspects, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sentiments.json").write_text(json.dumps(sentiments, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Aspects:", aspects)
    print("Sentiments:", sentiments)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer=args.analyzer,
        ngram_range=(args.min_ngram, args.max_ngram),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_df[args.text_col].astype(str).tolist())
    X_dev = vectorizer.transform(dev_df[args.text_col].astype(str).tolist())
    X_test = vectorizer.transform(test_df[args.text_col].astype(str).tolist())

    y_train = make_aspect_matrix(train_df, aspects)
    y_dev = make_aspect_matrix(dev_df, aspects)
    y_test = make_aspect_matrix(test_df, aspects)

    aspect_models = []
    for j, asp in enumerate(aspects):
        yj = y_train[:, j]
        pos = int(yj.sum())
        neg = int(len(yj) - pos)
        spw = (neg / max(pos, 1)) if args.use_scale_pos_weight else 1.0
        print(f"Training aspect {j+1}/{len(aspects)}: {asp} | pos={pos}, neg={neg}, spw={spw:.2f}")
        model = make_xgb_binary(args, spw)
        model.fit(X_train, yj)
        aspect_models.append(model)

    dev_prob = predict_aspect_probs(aspect_models, X_dev)
    test_prob = predict_aspect_probs(aspect_models, X_test)

    if args.tune_threshold:
        best_t, dev_metrics = tune_threshold(y_dev, dev_prob)
        print("Best aspect threshold:", best_t)
        print("Dev aspect metrics:", json.dumps(dev_metrics, indent=2))
    else:
        best_t = args.threshold

    aspect_test_metrics = evaluate_aspect(y_test, test_prob, threshold=best_t)
    aspect_test_metrics["aspect_threshold"] = best_t

    tr_texts, tr_asps, tr_labs = make_sentiment_examples(train_df, args.text_col)
    te_texts, te_asps, te_labs = make_sentiment_examples(test_df, args.text_col)
    sent2id = {s: i for i, s in enumerate(sentiments)}

    def filter_seen_sentiments(texts, asps, labs):
        kept_texts, kept_asps, kept_labs = [], [], []
        skipped = Counter()
        for t, a, y in zip(texts, asps, labs):
            if y in sent2id:
                kept_texts.append(t)
                kept_asps.append(a)
                kept_labs.append(y)
            else:
                skipped[y] += 1
        return kept_texts, kept_asps, kept_labs, skipped

    tr_texts, tr_asps, tr_labs, tr_skipped = filter_seen_sentiments(tr_texts, tr_asps, tr_labs)
    te_texts, te_asps, te_labs, te_skipped = filter_seen_sentiments(te_texts, te_asps, te_labs)
    if tr_skipped or te_skipped:
        print("Skipped sentiment labels unseen in train:", {"train": dict(tr_skipped), "test": dict(te_skipped)})

    y_sent_train = np.array([sent2id[y] for y in tr_labs], dtype=np.int64)
    y_sent_test = np.array([sent2id[y] for y in te_labs], dtype=np.int64)

    sent_train_inputs = aspect_conditioned_text(tr_texts, tr_asps, args.sent_pair_template)
    sent_test_inputs = aspect_conditioned_text(te_texts, te_asps, args.sent_pair_template)

    sent_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer=args.analyzer,
        ngram_range=(args.min_ngram, args.max_ngram),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    Xs_train = sent_vectorizer.fit_transform(sent_train_inputs)
    Xs_test = sent_vectorizer.transform(sent_test_inputs)

    print("Training sentiment classifier | examples:", len(y_sent_train), "dist:", Counter(tr_labs))
    sent_model = make_xgb_multiclass(args, num_class=len(sentiments))
    sent_model.fit(Xs_train, y_sent_train)
    sent_pred = sent_model.predict(Xs_test).astype(int).tolist()
    sentiment_test_metrics = evaluate_sentiment(y_sent_test.tolist(), sent_pred, sentiments)

    final = {
        "model": "XGBoost + TF-IDF",
        "aspect": aspect_test_metrics,
        "sentiment_gold_aspects": sentiment_test_metrics,
    }
    print("\nTEST")
    print(json.dumps(final, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    joblib.dump(vectorizer, out_dir / "aspect_tfidf.joblib")
    joblib.dump(aspect_models, out_dir / "aspect_xgb_models.joblib")
    joblib.dump(sent_vectorizer, out_dir / "sentiment_tfidf.joblib")
    joblib.dump(sent_model, out_dir / "sentiment_xgb_model.joblib")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--dev", type=str, default=None)
    p.add_argument("--test", type=str, default=None)
    p.add_argument("--split", type=float, nargs=3, default=[0.75, 0.125, 0.125])
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--gold_col", type=str, default="gold_struct")
    p.add_argument("--merged_col", type=str, default="merged_entities")
    p.add_argument("--aspects_json", type=str, default=None)
    p.add_argument("--sentiments_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--analyzer", choices=["word", "char", "char_wb"], default="word")
    p.add_argument("--min_ngram", type=int, default=1)
    p.add_argument("--max_ngram", type=int, default=2)
    p.add_argument("--max_features", type=int, default=20000)
    p.add_argument("--min_df", type=int, default=1)
    p.add_argument("--max_df", type=float, default=0.95)

    p.add_argument("--n_estimators", type=int, default=400)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--tree_method", type=str, default="hist")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--use_scale_pos_weight", action="store_true")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tune_threshold", action="store_true")
    p.add_argument("--sent_pair_template", choices=["question", "statement", "plain"], default="question")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data and not (args.train and args.dev and args.test):
        raise SystemExit("Provide either --data or all of --train --dev --test.")
    main(args)
