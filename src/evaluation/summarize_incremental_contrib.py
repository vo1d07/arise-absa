#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd


def parse_name(name: str):
    # Examples:
    # A2_run1_aipv_summary.json
    # S3_run2_sipv_summary.json
    m = re.match(r"^(A\d|S\d)_run(\d+)_(aipv|sipv)_summary\.json$", name)
    if not m:
        return None
    group, run, track = m.group(1), int(m.group(2)), m.group(3)
    return group, run, track


def load_metrics(metrics_dir: str):
    rows = []
    for fn in os.listdir(metrics_dir):
        parsed = parse_name(fn)
        if parsed is None:
            continue
        group, run, track = parsed
        path = os.path.join(metrics_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if track == "aipv":
            rows.append(
                {
                    "track": "A",
                    "group": group,
                    "run": run,
                    "micro_f1": float(data.get("micro_f1", 0.0)),
                    "macro_f1": float(data.get("macro_f1", 0.0)),
                    "accuracy": float(data.get("subset_accuracy", 0.0)),
                }
            )
        else:
            rows.append(
                {
                    "track": "S",
                    "group": group,
                    "run": run,
                    "micro_f1": float(data.get("micro_f1", 0.0)),
                    "macro_f1": float(data.get("macro_f1", 0.0)),
                    "accuracy": float(data.get("accuracy", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def group_order(track: str):
    # Decremental ablation order:
    # A0/S0 are full models; following groups remove modules step by step.
    return ["A0", "A1", "A2", "A3", "A4"] if track == "A" else ["S0", "S1", "S2", "S3"]


def summarize(df: pd.DataFrame):
    out_rows = []
    for track in ["A", "S"]:
        sub = df[df["track"] == track].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby(["track", "group"], as_index=False)
            .agg(
                micro_f1_mean=("micro_f1", "mean"),
                micro_f1_std=("micro_f1", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                n_runs=("run", "nunique"),
            )
        )

        order = group_order(track)
        agg["ord"] = agg["group"].apply(lambda g: order.index(g) if g in order else 999)
        agg = agg.sort_values("ord").drop(columns=["ord"]).reset_index(drop=True)

        prev_micro = None
        prev_macro = None
        prev_acc = None
        for _, row in agg.iterrows():
            d_micro = None if prev_micro is None else row["micro_f1_mean"] - prev_micro
            d_macro = None if prev_macro is None else row["macro_f1_mean"] - prev_macro
            d_acc = None if prev_acc is None else row["accuracy_mean"] - prev_acc

            out_rows.append(
                {
                    "track": row["track"],
                    "group": row["group"],
                    "n_runs": int(row["n_runs"]),
                    "micro_f1_mean": float(row["micro_f1_mean"]),
                    "micro_f1_std": float(0.0 if pd.isna(row["micro_f1_std"]) else row["micro_f1_std"]),
                    "delta_micro_f1_vs_prev": d_micro,
                    "macro_f1_mean": float(row["macro_f1_mean"]),
                    "macro_f1_std": float(0.0 if pd.isna(row["macro_f1_std"]) else row["macro_f1_std"]),
                    "delta_macro_f1_vs_prev": d_macro,
                    "accuracy_mean": float(row["accuracy_mean"]),
                    "accuracy_std": float(0.0 if pd.isna(row["accuracy_std"]) else row["accuracy_std"]),
                    "delta_accuracy_vs_prev": d_acc,
                }
            )

            prev_micro = row["micro_f1_mean"]
            prev_macro = row["macro_f1_mean"]
            prev_acc = row["accuracy_mean"]

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser(description="Summarize incremental module contribution metrics")
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = load_metrics(args.metrics_dir)
    if df.empty:
        raise RuntimeError(f"No summary json files found in: {args.metrics_dir}")

    summary = summarize(df)
    summary.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
