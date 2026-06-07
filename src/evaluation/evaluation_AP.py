import argparse
import ast
import json
import math
import pandas as pd
from collections import defaultdict

def _safe_div(n, d):
    return n / d if d != 0 else 0.0

def _f1(p, r):
    return _safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0


def _parse_struct_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (dict, list, set, tuple)):
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

def eval_a_ipv_aspect_detection(
    df: pd.DataFrame,
    gold_col: str = "gold_struct",
    pred_col: str = "candidates_final",
    aspects_list: list = [
        "On-campus Service",
        "Counseling Service",
        "Mental Health Service",
        "Wellness Service",
        "Therapy Service",
        "Hotline Service",
        "Service Availability",
        "General"
    ],
    normalize_fn=None,
):
    """
    A-IPV aspect detection (multi-label) 指标：
      - micro-P / micro-R / micro-F1
      - macro-F1: 对每个 aspect 单独算 F1，再平均
      - subset accuracy: 预测集合==金标集合的比例

    df[gold_col] 期望结构：
      {
        "aspect_set": [...],
        "by_aspect": {...}
      }
    df[pred_col] 期望是 list[str] 或 set[str]（每条样本预测到的 aspects）
    """

    if aspects_list is None:
        all_aspects = set()
        for x in df[gold_col]:
            if isinstance(x, dict) and "aspect_set" in x:
                all_aspects.update(x.get("aspect_set", []) or [])
        # 同时把预测里出现的也纳入，避免漏掉
        for p in df[pred_col]:
            if isinstance(p, (list, set, tuple)):
                all_aspects.update(list(p))
        aspects_list = sorted(all_aspects)
    else:
        aspects_list = list(aspects_list)

    # micro counts
    micro_TP = micro_FP = micro_FN = 0

    # per-aspect counts
    per = {a: {"TP": 0, "FP": 0, "FN": 0} for a in aspects_list}

    # subset accuracy
    subset_match = 0
    n = 0

    for _, row in df.iterrows():
        g = _parse_struct_cell(row.get(gold_col, None))
        p = _parse_struct_cell(row.get(pred_col, None))

        # gold set
        if isinstance(g, dict):
            gset = set(g.get("aspect_set", []) or [])
        else:
            gset = set()

        # pred set
        if isinstance(p, (list, set, tuple)):
            pset = set(p)
        elif p is None or (isinstance(p, float) and pd.isna(p)):
            pset = set()
        else:
            pset = set()

        if normalize_fn is not None:
            gset = set(normalize_fn(a) for a in gset)
            pset = set(normalize_fn(a) for a in pset)

        # subset accuracy
        n += 1
        if pset == gset:
            subset_match += 1

        # micro + per-aspect counts
        for a in aspects_list:
            in_p = a in pset
            in_g = a in gset

            if in_p and in_g:
                micro_TP += 1
                per[a]["TP"] += 1
            elif in_p and (not in_g):
                micro_FP += 1
                per[a]["FP"] += 1
            elif (not in_p) and in_g:
                micro_FN += 1
                per[a]["FN"] += 1
            # else: TN，不用记

    # 4) micro
    micro_P = _safe_div(micro_TP, micro_TP + micro_FP)
    micro_R = _safe_div(micro_TP, micro_TP + micro_FN)
    micro_F1 = _f1(micro_P, micro_R)

    # 5) macro-F1 (aspect average)
    per_aspect_rows = []
    f1s = []
    for a in aspects_list:
        TP = per[a]["TP"]
        FP = per[a]["FP"]
        FN = per[a]["FN"]
        P = _safe_div(TP, TP + FP)
        R = _safe_div(TP, TP + FN)
        F1 = _f1(P, R)
        f1s.append(F1)
        per_aspect_rows.append({
            "aspect": a,
            "TP": TP, "FP": FP, "FN": FN,
            "precision": P,
            "recall": R,
            "f1": F1,
            "support": TP + FN,
        })

    macro_F1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0.0
    subset_acc = subset_match / n if n > 0 else 0.0

    summary = {
        "micro_precision": micro_P,
        "micro_recall": micro_R,
        "micro_f1": micro_F1,
        "macro_f1": macro_F1,
        "subset_accuracy": subset_acc,
        "n_samples": n,
        "n_aspects": len(aspects_list),
        "micro_TP": micro_TP,
        "micro_FP": micro_FP,
        "micro_FN": micro_FN,
    }

    per_aspect_df = pd.DataFrame(per_aspect_rows).sort_values(
        by=["support", "f1"], ascending=[False, True]
    ).reset_index(drop=True)

    return summary, per_aspect_df


def main():
    ap = argparse.ArgumentParser(description="Evaluate A-IPV aspect detection results")
    ap.add_argument("--in_csv", default="outputs/AIPV_output.csv", help="CSV file with A-IPV predictions (containing validator_final column)")
    ap.add_argument("--gold_col", default="gold_struct", help="Column name for gold annotations in CSV")
    ap.add_argument("--pred_col", default="candidates_final", help="Name to extract from validator_final; if validator_final is list then use directly")
    ap.add_argument("--output_summary", default=None, help="Output file for summary metrics (JSON)")
    ap.add_argument("--output_details", default=None, help="Output file for per-aspect details (CSV)")
    args = ap.parse_args()

    # Read CSV
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} samples from {args.in_csv}")

    # Evaluate
    summary, per_aspect_df = eval_a_ipv_aspect_detection(
        df, 
        gold_col=args.gold_col,
        pred_col=args.pred_col,
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Samples: {summary['n_samples']}")
    print(f"Aspects: {summary['n_aspects']}")
    print(f"\nMicro Metrics:")
    print(f"  Precision: {summary['micro_precision']:.4f}")
    print(f"  Recall:    {summary['micro_recall']:.4f}")
    print(f"  F1:        {summary['micro_f1']:.4f}")
    print(f"\nMacro F1:      {summary['macro_f1']:.4f}")
    print(f"Subset Acc:    {summary['subset_accuracy']:.4f}")
    print(f"\nMicro Counts: TP={summary['micro_TP']} FP={summary['micro_FP']} FN={summary['micro_FN']}")
    print("="*60)

    print("\nPer-Aspect Breakdown:")
    print(per_aspect_df.to_string(index=False))
    print("="*60)

    # Save outputs
    if args.output_summary:
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to {args.output_summary}")

    if args.output_details:
        per_aspect_df.to_csv(args.output_details, index=False)
        print(f"[OK] Per-aspect details saved to {args.output_details}")


if __name__ == "__main__":
    main()