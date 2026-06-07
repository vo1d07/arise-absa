"""
deberta_v3_baseline.py

Supervised DeBERTa-v3-base baseline for AWARE-SMILE ABSA.

It trains two models:
  1) Aspect Prediction / ACD:
       input: text
       output: multi-label aspect vector
       loss: BCEWithLogitsLoss

  2) Sentiment Prediction / ASC:
       input: aspect-conditioned pair: aspect + text
       output: sentiment label for gold aspects
       loss: CrossEntropyLoss

Supported CSV formats:
  - text column, default: text
  - gold_struct column:
      {"aspect_set": [...], "by_aspect": {"Service Availability": "Negative", ...}}
    or
      {"by_aspect": {"Service Availability": {"label": "Negative", ...}}}
  - merged_entities column:
      "[{'aspects': [...], 'sentiments': [...], ...}, ...]"

Example:
  python deberta_v3_baseline.py \
    --train train.csv --dev dev.csv --test test.csv \
    --output_dir runs/deberta_v3_base \
    --epochs 5 --batch_size 16 \
    --use_pos_weight --use_class_weights --tune_threshold
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import sys
# ensure project root is on sys.path so ASQP_MA can be imported when running from subfolders
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ASQP_MA.evaluation_SIPV import eval_s_ipv_sentiment_conditional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


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
    if low in {"mix", "mixed", "conflict", "conflicting"}:
        return "mixed"
    if low in {"none", "null", "nan", "no", "absent"}:
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


def parse_gold_from_row(row: pd.Series, gold_col: str = "gold_struct", merged_col: str = "merged_entities") -> Dict[str, str]:
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
    return [x for x in preferred if x in labels] + sorted(labels - set(preferred))


class AspectDataset(Dataset):
    def __init__(self, df, text_col, aspects, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.aspects = aspects
        self.aspect2id = {a: i for i, a in enumerate(aspects)}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        labels = np.zeros(len(self.aspects), dtype=np.float32)
        for asp in row["_gold_by_aspect"].keys():
            if asp in self.aspect2id:
                labels[self.aspect2id[asp]] = 1.0
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item


class SentimentDataset(Dataset):
    def __init__(self, df, text_col, sentiments, tokenizer, max_length, pair_template="question"):
        self.items = []
        self.sentiments = sentiments
        self.sent2id = {s: i for i, s in enumerate(sentiments)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pair_template = pair_template
        for _, row in df.iterrows():
            text = str(row[text_col])
            for asp, lab in row["_gold_by_aspect"].items():
                lab = normalize_label(lab)
                if lab in self.sent2id:
                    self.items.append((text, asp, lab))

    def __len__(self):
        return len(self.items)

    def make_query(self, aspect: str) -> str:
        if self.pair_template == "plain":
            return aspect
        if self.pair_template == "statement":
            return f"The sentiment toward {aspect}."
        return f"What is the sentiment toward {aspect}?"

    def __getitem__(self, idx):
        text, asp, lab = self.items[idx]
        query = self.make_query(asp)
        enc = self.tokenizer(query, text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.sent2id[lab], dtype=torch.long)
        # keep aspect string for evaluation with external evaluator
        item["aspect"] = asp
        return item


class EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, problem_type: str, dropout: float = 0.1, class_weights=None, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.problem_type = problem_type
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        out = self.encoder(**inputs)
        pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if self.problem_type == "multilabel":
                loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, labels)
            else:
                loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return {"loss": loss, "logits": logits}


def multilabel_subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def evaluate_aspect(model, loader, device, threshold=0.5) -> Dict[str, float]:
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval aspect", leave=False):
            labels = batch["labels"].numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v) and k != "labels"}
            logits = model(**inputs)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true.append(labels)
            all_prob.append(probs)
    y_true = np.vstack(all_true)
    y_prob = np.vstack(all_prob)
    y_pred = (y_prob >= threshold).astype(int)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true.reshape(-1), y_pred.reshape(-1), average="binary", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
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
        "aspect_subset_accuracy": multilabel_subset_accuracy(y_true, y_pred),
        "gold_positive_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
    }


def find_best_threshold(model, loader, device):
    best_t, best_m = 0.5, None
    for t in np.arange(0.10, 0.91, 0.05):
        m = evaluate_aspect(model, loader, device, threshold=float(t))
        if best_m is None or m["aspect_micro_f1"] > best_m["aspect_micro_f1"]:
            best_t, best_m = float(t), m
    return best_t, best_m


def evaluate_sentiment(model, loader, sentiments, device) -> Dict[str, float]:
    model.eval()
    id2sent = {i: s for i, s in enumerate(sentiments)}
    y_true_labels = []
    y_pred_labels = []
    aspects = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval sentiment", leave=False):
            labels_idx = batch["labels"].cpu().numpy().tolist()
            inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v) and k != "labels"}
            logits = model(**inputs)["logits"]
            preds_idx = logits.argmax(dim=-1).cpu().numpy().tolist()
            # aspects come from non-tensor collated field
            batch_aspects = batch.get("aspect")
            for i in range(len(labels_idx)):
                y_true_labels.append(id2sent[labels_idx[i]])
                y_pred_labels.append(id2sent[preds_idx[i]])
                # if batch_aspects is a list, index it; else if it's a tensor, convert
                if isinstance(batch_aspects, (list, tuple)):
                    aspects.append(batch_aspects[i])
                else:
                    aspects.append(str(batch_aspects))

    if not y_true_labels:
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
        }

    # Map our label strings to the S-IPV evaluator's expected labels (e.g. 'negative' -> 'NEG')
    label_map_to_sipv = {
        "negative": "NEG",
        "pos": "POS",
        "positive": "POS",
        "neutral": "NEU",
        "neu": "NEU",
        "mixed": "MIXED",
        "mix": "MIXED",
    }

    # Build a DataFrame compatible with eval_s_ipv_sentiment_conditional
    rows = []
    for gt, pr, asp in zip(y_true_labels, y_pred_labels, aspects):
        gt_mapped = label_map_to_sipv.get(str(gt).lower(), gt)
        pr_mapped = label_map_to_sipv.get(str(pr).lower(), pr)
        gold_struct = {"by_aspect": {asp: {"label": gt_mapped}}, "aspect_set": [asp]}
        pred_struct = {asp: {"label": pr_mapped}}
        rows.append({"gold_struct": gold_struct, "sentiment_final": pred_struct})

    df = pd.DataFrame(rows)
    # call S-IPV evaluator (uses its own label mapping and missing policy)
    summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(df)

    return {
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


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


def train_one_model(model, train_loader, dev_loader, device, args, task_name: str, eval_fn, metric_key: str, output_dir: Path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)
    best_score = -1.0
    best_path = output_dir / f"best_{task_name}.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{task_name} epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            inputs = batch_to_device(batch, device)
            out = model(**inputs)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        dev_metrics = eval_fn(model, dev_loader)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_loader)), **{f"dev_{k}": v for k, v in dev_metrics.items() if isinstance(v, (int, float))}}
        history.append(row)
        print(json.dumps(row, indent=2))
        score = dev_metrics[metric_key]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
    pd.DataFrame(history).to_csv(output_dir / f"{task_name}_history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


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


def main(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, dev_df, test_df = load_data(args)

    aspects = [canonical_aspect(a) for a in json.loads(Path(args.aspects_json).read_text(encoding="utf-8"))] if args.aspects_json else infer_aspects([train_df, dev_df, test_df])
    sentiments = [normalize_label(x) for x in json.loads(Path(args.sentiments_json).read_text(encoding="utf-8"))] if args.sentiments_json else infer_sentiments([train_df, dev_df, test_df])
    (output_dir / "aspects.json").write_text(json.dumps(aspects, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sentiments.json").write_text(json.dumps(sentiments, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    aspect_train_ds = AspectDataset(train_df, args.text_col, aspects, tokenizer, args.max_length)
    aspect_dev_ds = AspectDataset(dev_df, args.text_col, aspects, tokenizer, args.max_length)
    aspect_test_ds = AspectDataset(test_df, args.text_col, aspects, tokenizer, args.max_length)
    sent_train_ds = SentimentDataset(train_df, args.text_col, sentiments, tokenizer, args.max_length, args.sent_pair_template)
    sent_dev_ds = SentimentDataset(dev_df, args.text_col, sentiments, tokenizer, args.max_length, args.sent_pair_template)
    sent_test_ds = SentimentDataset(test_df, args.text_col, sentiments, tokenizer, args.max_length, args.sent_pair_template)

    aspect_train_loader = DataLoader(aspect_train_ds, batch_size=args.batch_size, shuffle=True)
    aspect_dev_loader = DataLoader(aspect_dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    aspect_test_loader = DataLoader(aspect_test_ds, batch_size=args.eval_batch_size, shuffle=False)
    sent_train_loader = DataLoader(sent_train_ds, batch_size=args.batch_size, shuffle=True)
    sent_dev_loader = DataLoader(sent_dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    sent_test_loader = DataLoader(sent_test_ds, batch_size=args.eval_batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Device:", device)
    print("Aspects:", aspects)
    print("Sentiments:", sentiments)
    print("Sentiment train examples:", len(sent_train_ds))

    aspect_mat = np.vstack([aspect_train_ds[i]["labels"].numpy() for i in range(len(aspect_train_ds))])
    pos = aspect_mat.sum(axis=0)
    neg = aspect_mat.shape[0] - pos
    pos_weight = torch.tensor(np.maximum(neg / np.maximum(pos, 1.0), 1.0), dtype=torch.float) if args.use_pos_weight else None
    if pos_weight is not None:
        print("Aspect pos_weight:", {a: float(pos_weight[i]) for i, a in enumerate(aspects)})

    aspect_model = EncoderClassifier(args.model_name, len(aspects), "multilabel", args.dropout, pos_weight=pos_weight)
    aspect_model = train_one_model(
        aspect_model, aspect_train_loader, aspect_dev_loader, device, args, "aspect",
        lambda m, loader: evaluate_aspect(m, loader, device, threshold=args.threshold),
        "aspect_micro_f1", output_dir
    )
    if args.tune_threshold:
        best_t, dev_best = find_best_threshold(aspect_model, aspect_dev_loader, device)
        print("Best aspect threshold:", best_t)
        print("Best dev threshold metrics:", json.dumps(dev_best, indent=2))
    else:
        best_t = args.threshold
    aspect_test_metrics = evaluate_aspect(aspect_model, aspect_test_loader, device, threshold=best_t)
    aspect_test_metrics["aspect_threshold"] = best_t

    sent_counts = Counter([lab for _, _, lab in sent_train_ds.items])
    if args.use_class_weights:
        counts = np.array([sent_counts.get(s, 0) for s in sentiments], dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        sent_weights = torch.tensor(counts.sum() / (len(sentiments) * counts), dtype=torch.float)
        print("Sentiment class weights:", {s: float(sent_weights[i]) for i, s in enumerate(sentiments)})
    else:
        sent_weights = None
    sent_model = EncoderClassifier(args.model_name, len(sentiments), "singlelabel", args.dropout, class_weights=sent_weights)
    sent_model = train_one_model(
        sent_model, sent_train_loader, sent_dev_loader, device, args, "sentiment",
        lambda m, loader: evaluate_sentiment(m, loader, sentiments, device),
        "sent_macro_f1", output_dir
    )
    sent_test_metrics = evaluate_sentiment(sent_model, sent_test_loader, sentiments, device)

    final = {"model_name": args.model_name, "aspect": aspect_test_metrics, "sentiment_gold_aspects": sent_test_metrics}
    print("TEST")
    print(json.dumps(final, indent=2))
    (output_dir / "test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output_dir / "tokenizer")


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
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tune_threshold", action="store_true")
    p.add_argument("--use_pos_weight", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--sent_pair_template", choices=["question", "statement", "plain"], default="question")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.data and not (args.train and args.dev and args.test):
        raise SystemExit("Provide either --data or all of --train --dev --test.")
    main(args)
