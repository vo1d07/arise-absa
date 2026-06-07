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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASQP_MA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ASQP_MA_DIR not in sys.path:
    sys.path.append(ASQP_MA_DIR)

from evaluation_AIPV import eval_a_ipv_aspect_detection


DEFAULT_IN_CSV = "ASQP_MA/datasets/dataset.csv"
DEFAULT_OUT_MODEL = "ASQP_MA/tfidf_lr_a_ipv_baseline.joblib"


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


def build_label_matrix(aspect_sets: List[List[str]], labels: List[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.zeros((len(aspect_sets), len(labels)), dtype=int)
    for row_idx, aspects in enumerate(aspect_sets):
        for aspect in set(aspects):
            col_idx = label_to_idx.get(aspect)
            if col_idx is not None:
                y[row_idx, col_idx] = 1
    return y


def fit_binary_lr(texts: List[str], targets: np.ndarray):
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
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(texts, targets)
    return model


def predict_proba_for_model(model, texts: List[str]) -> np.ndarray:
    if isinstance(model, (int, np.integer)):
        return np.full(len(texts), float(model), dtype=float)
    return np.asarray(model.predict_proba(texts))[:, 1]


def scores_to_labels(scores: np.ndarray, labels: List[str], threshold: float) -> List[List[str]]:
    predictions: List[List[str]] = []
    for row in scores:
        chosen = [label for label, score in zip(labels, row) if score >= threshold]
        predictions.append(chosen)
    return predictions


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@dataclass
class AspectLRBaseline:
    labels: List[str]
    threshold: float
    models: Dict[str, object]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        text_list = [str(text) for text in texts]
        columns = [predict_proba_for_model(self.models[label], text_list) for label in self.labels]
        return np.column_stack(columns) if columns else np.zeros((len(text_list), 0), dtype=float)

    def predict(self, texts: Sequence[str]) -> List[List[str]]:
        scores = self.predict_proba(texts)
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


def train_aspect_lr_baseline(
    df: pd.DataFrame,
    text_col: str = "text",
    gold_col: str = "gold_struct",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    threshold: float = 0.5,
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

    x_train = [texts[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    x_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    models: Dict[str, object] = {}
    for label_idx, label in enumerate(labels):
        models[label] = fit_binary_lr(x_train, y_train[:, label_idx])

    baseline = AspectLRBaseline(labels=labels, threshold=threshold, models=models)

    val_scores = baseline.predict_proba(x_val)
    val_pred = (val_scores >= threshold).astype(int)
    val_summary = evaluate_multilabel(y_val, val_pred)
    val_summary["n_samples"] = int(len(x_val))

    test_scores = baseline.predict_proba(x_test)
    test_pred = (test_scores >= threshold).astype(int)
    test_summary = evaluate_multilabel(y_test, test_pred)
    test_summary["n_samples"] = int(len(x_test))

    val_pred_sets = baseline.predict(x_val)
    test_pred_sets = baseline.predict(x_test)

    val_eval_df = pd.DataFrame(
        {
            "gold_struct": [{"aspect_set": list(aspect_sets[i])} for i in val_idx],
            "candidates_final": [list(p) for p in val_pred_sets],
        }
    )
    test_eval_df = pd.DataFrame(
        {
            "gold_struct": [{"aspect_set": list(aspect_sets[i])} for i in test_idx],
            "candidates_final": [list(p) for p in test_pred_sets],
        }
    )

    val_aipv_summary, val_per_aspect = eval_a_ipv_aspect_detection(
        val_eval_df,
        gold_col="gold_struct",
        pred_col="candidates_final",
        aspects_list=labels,
    )
    test_aipv_summary, test_per_aspect = eval_a_ipv_aspect_detection(
        test_eval_df,
        gold_col="gold_struct",
        pred_col="candidates_final",
        aspects_list=labels,
    )

    return (
        baseline,
        val_summary,
        test_summary,
        val_aipv_summary,
        val_per_aspect,
        test_aipv_summary,
        test_per_aspect,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal LR baseline for AIPV aspect prediction.")
    parser.add_argument("--in_csv", type=str, default=DEFAULT_IN_CSV)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--gold_col", type=str, default="gold_struct")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_model", type=str, default=DEFAULT_OUT_MODEL)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    (
        baseline,
        val_summary,
        test_summary,
        val_aipv_summary,
        val_per_aspect,
        test_aipv_summary,
        test_per_aspect,
    ) = train_aspect_lr_baseline(
        df,
        text_col=args.text_col,
        gold_col=args.gold_col,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        threshold=args.threshold,
    )

    print("VAL summary:", val_summary)
    print("TEST summary:", test_summary)
    print("AIPV eval VAL summary:", val_aipv_summary)
    print(val_per_aspect.to_string(index=False))
    print("AIPV eval TEST summary:", test_aipv_summary)
    print(test_per_aspect.to_string(index=False))

    baseline.save(args.output_model)
    print(f"Model saved to: {args.output_model}")


if __name__ == "__main__":
    main()
