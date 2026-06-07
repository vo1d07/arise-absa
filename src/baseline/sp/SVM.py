import argparse
import ast
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASQP_MA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ASQP_MA_DIR not in sys.path:
    sys.path.append(ASQP_MA_DIR)

from evaluation_SIPV import eval_s_ipv_sentiment_conditional


DEFAULT_IN_CSV = "ASQP_MA/datasets/dataset.csv"
DEFAULT_OUT_MODEL = "ASQP_MA/aspect_svm_baseline.joblib"


def parse_gold_struct(cell) -> Dict:
    if isinstance(cell, dict):
        return cell
    if cell is None:
        return {}
    if isinstance(cell, float) and pd.isna(cell):
        return {}
    if isinstance(cell, str):
        text = cell.strip()
        if not text:
            return {}
        for loader in (ast.literal_eval, json.loads):
            try:
                obj = loader(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def aspect_set_from_gold(cell) -> List[str]:
    gold = parse_gold_struct(cell)
    aspects = gold.get("aspect_set")
    if isinstance(aspects, list):
        items = aspects
    else:
        by_aspect = gold.get("by_aspect", {})
        items = list(by_aspect.keys()) if isinstance(by_aspect, dict) else []
    return sorted({str(item).strip() for item in items if str(item).strip()})


def sentiment_map_to_short(label):
    mapping = {
        "NEG": "NEG",
        "Negative": "NEG",
        "negative": "NEG",
        "NEU": "NEU",
        "Neutral": "NEU",
        "neutral": "NEU",
        "POS": "POS",
        "Positive": "POS",
        "positive": "POS",
    }
    return mapping.get(label)


def collect_majority_sentiment_by_aspect(df: pd.DataFrame, gold_col: str) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {}
    for cell in df[gold_col]:
        gold = parse_gold_struct(cell)
        by_aspect = gold.get("by_aspect", {}) if isinstance(gold, dict) else {}
        if not isinstance(by_aspect, dict):
            continue
        for aspect, info in by_aspect.items():
            if not isinstance(info, dict):
                continue
            short_label = sentiment_map_to_short(info.get("label"))
            if short_label is None:
                continue
            counts.setdefault(aspect, {})
            counts[aspect][short_label] = counts[aspect].get(short_label, 0) + 1

    majority: Dict[str, str] = {}
    order = ["NEG", "NEU", "POS"]
    for aspect, c in counts.items():
        majority[aspect] = sorted(order, key=lambda x: (-c.get(x, 0), order.index(x)))[0]
    return majority


def build_sentiment_structs(pred_aspect_sets: List[List[str]], majority_map: Dict[str, str]):
    rows = []
    for aspects in pred_aspect_sets:
        row = {}
        for aspect in aspects:
            row[aspect] = {"label": majority_map.get(aspect, "NEU")}
        rows.append(row)
    return rows


def build_label_matrix(aspect_sets: List[List[str]], labels: List[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.zeros((len(aspect_sets), len(labels)), dtype=int)
    for row_idx, aspects in enumerate(aspect_sets):
        for aspect in set(aspects):
            col_idx = label_to_idx.get(aspect)
            if col_idx is not None:
                y[row_idx, col_idx] = 1
    return y


def fit_binary_svm(texts: List[str], targets: np.ndarray, c_value: float):
    if targets.min() == targets.max():
        return int(targets[0])

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )
    model.fit(texts, targets)
    return model


def decision_for_model(model, texts: List[str]) -> np.ndarray:
    if isinstance(model, (int, np.integer)):
        # Constant classifier: positive class score 1, negative class score -1.
        score = 1.0 if int(model) == 1 else -1.0
        return np.full(len(texts), score, dtype=float)
    return np.asarray(model.decision_function(texts), dtype=float)


def scores_to_labels(scores: np.ndarray, labels: List[str], threshold: float) -> List[List[str]]:
    predictions: List[List[str]] = []
    for row in scores:
        chosen = [label for label, score in zip(labels, row) if score >= threshold]
        if not chosen:
            chosen = [labels[int(np.argmax(row))]]
        predictions.append(chosen)
    return predictions


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@dataclass
class AspectSVMBaseline:
    labels: List[str]
    threshold: float
    models: Dict[str, object]

    def decision_scores(self, texts: Sequence[str]) -> np.ndarray:
        text_list = [str(text) for text in texts]
        columns = [decision_for_model(self.models[label], text_list) for label in self.labels]
        return np.column_stack(columns) if columns else np.zeros((len(text_list), 0), dtype=float)

    def predict(self, texts: Sequence[str]) -> List[List[str]]:
        scores = self.decision_scores(texts)
        return scores_to_labels(scores, self.labels, self.threshold)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "labels": self.labels,
                "threshold": self.threshold,
                "models": self.models,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "AspectSVMBaseline":
        payload = joblib.load(path)
        return AspectSVMBaseline(
            labels=payload["labels"],
            threshold=payload["threshold"],
            models=payload["models"],
        )


def train_aspect_svm_baseline(
    df: pd.DataFrame,
    text_col: str = "text",
    gold_col: str = "gold_struct",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    threshold: float = 0.0,
    c_value: float = 1.0,
):
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")
    if gold_col not in df.columns:
        raise KeyError(f"Missing gold column: {gold_col}")

    texts = df[text_col].fillna("").astype(str).tolist()
    aspect_sets = [aspect_set_from_gold(cell) for cell in df[gold_col]]
    labels = sorted({aspect for aspects in aspect_sets for aspect in aspects})

    if not labels:
        raise ValueError(f"No aspect labels found in column '{gold_col}'.")

    y = build_label_matrix(aspect_sets, labels)
    indices = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    X_train = [texts[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    models: Dict[str, object] = {}
    for label_idx, label in enumerate(labels):
        models[label] = fit_binary_svm(X_train, y_train[:, label_idx], c_value)

    baseline = AspectSVMBaseline(labels=labels, threshold=threshold, models=models)
    majority_sentiment = collect_majority_sentiment_by_aspect(df.iloc[train_idx], gold_col)

    val_scores = baseline.decision_scores(X_val)
    val_pred = (val_scores >= threshold).astype(int)
    if len(val_pred):
        empty_rows = np.where(val_pred.sum(axis=1) == 0)[0]
        for row_idx in empty_rows:
            val_pred[row_idx, int(np.argmax(val_scores[row_idx]))] = 1
    val_summary = evaluate_multilabel(y_val, val_pred)
    val_summary["n_samples"] = int(len(X_val))

    test_scores = baseline.decision_scores(X_test)
    test_pred = (test_scores >= threshold).astype(int)
    if len(test_pred):
        empty_rows = np.where(test_pred.sum(axis=1) == 0)[0]
        for row_idx in empty_rows:
            test_pred[row_idx, int(np.argmax(test_scores[row_idx]))] = 1
    test_summary = evaluate_multilabel(y_test, test_pred)
    test_summary["n_samples"] = int(len(X_test))

    test_pred_sets = baseline.predict(X_test)
    test_eval_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_eval_df["sentiment_final"] = build_sentiment_structs(test_pred_sets, majority_sentiment)

    sipv_summary, sipv_per_label, sipv_confusion, sipv_per_aspect = eval_s_ipv_sentiment_conditional(
        test_eval_df,
        gold_col=gold_col,
        pred_col="sentiment_final",
        missing_policy="wrong",
    )

    return (
        baseline,
        val_summary,
        test_summary,
        sipv_summary,
        sipv_per_label,
        sipv_confusion,
        sipv_per_aspect,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal SVM baseline for aspect prediction.")
    parser.add_argument("--in_csv", type=str, default=DEFAULT_IN_CSV)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--gold_col", type=str, default="gold_struct")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--output_model", type=str, default=DEFAULT_OUT_MODEL)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    (
        baseline,
        val_summary,
        test_summary,
        sipv_summary,
        sipv_per_label,
        sipv_confusion,
        sipv_per_aspect,
    ) = train_aspect_svm_baseline(
        df,
        text_col=args.text_col,
        gold_col=args.gold_col,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        threshold=args.threshold,
        c_value=args.C,
    )

    print("VAL summary:", val_summary)
    print("TEST summary:", test_summary)
    print("SIPV conditional summary:", sipv_summary)
    print("SIPV per-label:")
    print(sipv_per_label.to_string(index=False))
    print("SIPV confusion (rows=true, cols=pred):")
    print(sipv_confusion.to_string())
    print("SIPV per-aspect:")
    print(sipv_per_aspect.to_string(index=False))

    baseline.save(args.output_model)
    print(f"Model saved to: {args.output_model}")


if __name__ == "__main__":
    main()
