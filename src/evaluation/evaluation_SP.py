import argparse
import ast
import json
import pandas as pd
from collections import defaultdict

def _safe_div(n, d):
    return n / d if d != 0 else 0.0

def _f1(p, r):
    return (2*p*r/(p+r)) if (p+r) else 0.0


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

def eval_s_ipv_sentiment_conditional(
    df: pd.DataFrame,
    gold_col: str = "gold_struct",
    pred_col: str = "sentiment_final",
    aspects_list=None,             # 可选：只评这几个 aspect（默认评所有 gold 出现过的）
    label_map_gold=None,           # gold label 归一化映射
    label_map_pred=None,           # pred label 归一化映射
    labels_eval=("NEG", "NEU", "POS"),
    missing_policy="wrong",        # "wrong"：缺失算错；"skip"：缺失跳过不计
):
    """
    Gold-aspect conditional evaluation:
      只在 gold 的 (i, a) 上评 sentiment，不评 A-IPV 的召回/精度。

    返回：
      summary: dict (accuracy, macro_f1, micro_f1, counts...)
      per_label: DataFrame (每个情感类的 P/R/F1/support)
      confusion: DataFrame (y_true x y_pred)
      per_aspect: DataFrame (每个 aspect 的 accuracy/support，便于排查)
    """

    # 默认映射（你现在数据/输出常见写法）
    if label_map_gold is None:
        label_map_gold = {
            "Negative": "NEG", "NEG": "NEG", "neg": "NEG", "negative": "NEG",
            "Neutral":  "NEU", "NEU": "NEU", "neu": "NEU", "neutral":  "NEU",
            "Positive": "POS", "POS": "POS", "pos": "POS", "positive": "POS",
            "MIXED": "MIXED", None: None
        }
    if label_map_pred is None:
        label_map_pred = {
            "NEG": "NEG", "NEU": "NEU", "POS": "POS",
            "Negative": "NEG", "Neutral": "NEU", "Positive": "POS",
            "neg": "NEG", "neu": "NEU", "pos": "POS",
            None: None
        }

    labels_eval = list(labels_eval)
    labels_all = labels_eval + (["MISSING"] if missing_policy == "wrong" else [])

    # 1) 收集要评的 aspects（默认：数据里出现过的 gold aspects）
    if aspects_list is None:
        aspects_set = set()
        for g in df[gold_col]:
            g = _parse_struct_cell(g)
            if isinstance(g, dict):
                aspects_set.update((g.get("aspect_set") or []))
        aspects_list = sorted(aspects_set)
    else:
        aspects_list = list(aspects_list)

    # 2) 扁平化为 (i, a) 对
    y_true = []
    y_pred = []
    per_aspect_counts = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    per_aspect_pairs = defaultdict(lambda: {"y_true": [], "y_pred": []})

    total_pairs = 0
    skipped_pairs = 0

    for _, row in df.iterrows():
        g = _parse_struct_cell(row.get(gold_col, None))
        p = _parse_struct_cell(row.get(pred_col, None))

        if not isinstance(g, dict):
            continue

        gold_by_aspect = g.get("by_aspect", {}) or {}
        gold_aspects = set(g.get("aspect_set", []) or [])

        # 只在 gold aspects 上评
        for a in gold_aspects:
            if a not in aspects_list:
                continue

            gt_raw = None
            if isinstance(gold_by_aspect.get(a, None), dict):
                gt_raw = gold_by_aspect[a].get("label", None)

            gt = label_map_gold.get(gt_raw, gt_raw)

            # pred 取 label
            pred_label_raw = None
            if isinstance(p, dict) and isinstance(p.get(a, None), dict):
                pred_label_raw = p[a].get("label", None)
            pred_label = label_map_pred.get(pred_label_raw, pred_label_raw)

            # 缺失处理
            if pred_label is None:
                per_aspect_counts[a]["missing"] += 1
                if missing_policy == "skip":
                    skipped_pairs += 1
                    continue
                pred_label = "MISSING"

            # 只把可评的 label 放进来
            if gt not in labels_eval:
                # gold 若是 MIXED 或其它，你可以选择跳过
                skipped_pairs += 1
                continue

            y_true.append(gt)
            y_pred.append(pred_label)
            total_pairs += 1

            per_aspect_counts[a]["total"] += 1
            if pred_label == gt:
                per_aspect_counts[a]["correct"] += 1
            per_aspect_pairs[a]["y_true"].append(gt)
            per_aspect_pairs[a]["y_pred"].append(pred_label)

    # 3) accuracy
    correct = sum(1 for t, pr in zip(y_true, y_pred) if t == pr)
    accuracy = _safe_div(correct, len(y_true))

    # 4) micro-F1（单标签多类：micro-F1 == accuracy；这里仍按定义算一遍，含 MISSING）
    # micro counts over labels_eval（可选包含 MISSING：会把错计入 FP/FN）
    micro_TP = 0
    micro_FP = 0
    micro_FN = 0
    for t, pr in zip(y_true, y_pred):
        if pr == t and t in labels_eval:
            micro_TP += 1
        else:
            # 预测成了某个 eval label 但不对 -> FP
            if pr in labels_eval:
                micro_FP += 1
            # 真值是某个 eval label 但没预测对 -> FN
            if t in labels_eval:
                micro_FN += 1

    micro_P = _safe_div(micro_TP, micro_TP + micro_FP)
    micro_R = _safe_div(micro_TP, micro_TP + micro_FN)
    micro_F1 = _f1(micro_P, micro_R)

    # 5) macro-F1（按情感类别平均）
    per_label_rows = []
    f1s = []
    for lab in labels_eval:
        TP = FP = FN = 0
        for t, pr in zip(y_true, y_pred):
            if pr == lab and t == lab:
                TP += 1
            elif pr == lab and t != lab:
                FP += 1
            elif pr != lab and t == lab:
                FN += 1
        P = _safe_div(TP, TP + FP)
        R = _safe_div(TP, TP + FN)
        F1 = _f1(P, R)
        f1s.append(F1)
        per_label_rows.append({
            "label": lab,
            "precision": P,
            "recall": R,
            "f1": F1,
            "support": TP + FN
        })
    macro_F1 = sum(f1s) / len(f1s) if f1s else 0.0

    per_label_df = pd.DataFrame(per_label_rows).sort_values(by="label").reset_index(drop=True)

    # 6) confusion matrix
    # 行：true，列：pred
    conf = pd.DataFrame(0, index=labels_eval, columns=labels_all)
    for t, pr in zip(y_true, y_pred):
        if t in conf.index and pr in conf.columns:
            conf.loc[t, pr] += 1

    # 7) per-aspect metrics（accuracy + macro P/R/F1 over sentiment labels）
    per_aspect_rows = []
    for a in aspects_list:
        tot = per_aspect_counts[a]["total"]
        cor = per_aspect_counts[a]["correct"]
        mis = per_aspect_counts[a]["missing"]
        if tot == 0 and mis == 0:
            continue

        a_true = per_aspect_pairs[a]["y_true"]
        a_pred = per_aspect_pairs[a]["y_pred"]

        p_list = []
        r_list = []
        f1_list = []
        for lab in labels_eval:
            TP = FP = FN = 0
            for t, pr in zip(a_true, a_pred):
                if pr == lab and t == lab:
                    TP += 1
                elif pr == lab and t != lab:
                    FP += 1
                elif pr != lab and t == lab:
                    FN += 1
            P = _safe_div(TP, TP + FP)
            R = _safe_div(TP, TP + FN)
            F1 = _f1(P, R)
            p_list.append(P)
            r_list.append(R)
            f1_list.append(F1)

        macro_P_a = sum(p_list) / len(p_list) if p_list else 0.0
        macro_R_a = sum(r_list) / len(r_list) if r_list else 0.0
        macro_F1_a = sum(f1_list) / len(f1_list) if f1_list else 0.0

        per_aspect_rows.append({
            "aspect": a,
            "precision": macro_P_a,
            "recall": macro_R_a,
            "f1": macro_F1_a,
            "accuracy": _safe_div(cor, tot),
            "support": tot,
            "missing": mis
        })
    per_aspect_df = pd.DataFrame(per_aspect_rows).sort_values(
        by=["support", "f1"], ascending=[False, True]
    ).reset_index(drop=True)

    summary = {
        "accuracy": accuracy,
        "macro_f1": macro_F1,
        "micro_f1": micro_F1,
        "micro_precision": micro_P,
        "micro_recall": micro_R,
        "n_pairs_used": len(y_true),
        "n_pairs_total_gold": total_pairs + skipped_pairs,
        "n_pairs_skipped": skipped_pairs,
        "missing_policy": missing_policy,
    }

    return summary, per_label_df, conf, per_aspect_df


def main():
    ap = argparse.ArgumentParser(description="Evaluate S-IPV sentiment results (gold-aspect conditional)")
    ap.add_argument("--in_csv", default="outputs/SIPV_eval_output.csv", help="CSV file with S-IPV predictions")
    ap.add_argument("--gold_col", default="gold_struct", help="Column name for gold annotations in CSV")
    ap.add_argument("--pred_col", default="sentiment_final", help="Column name for sentiment predictions in CSV")
    ap.add_argument("--missing_policy", default="wrong", choices=["wrong", "skip"], help="How to handle missing predictions")
    ap.add_argument("--output_summary", default=None, help="Output file for summary metrics (JSON)")
    ap.add_argument("--output_per_label", default=None, help="Output file for per-label metrics (CSV)")
    ap.add_argument("--output_confusion", default=None, help="Output file for confusion matrix (CSV)")
    ap.add_argument("--output_per_aspect", default=None, help="Output file for per-aspect metrics (CSV)")
    args = ap.parse_args()

    # Read CSV
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} samples from {args.in_csv}")

    # Evaluate
    summary, per_label_df, conf_df, per_aspect_df = eval_s_ipv_sentiment_conditional(
        df,
        gold_col=args.gold_col,
        pred_col=args.pred_col,
        missing_policy=args.missing_policy,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Pairs used:        {summary['n_pairs_used']}")
    print(f"Pairs total gold:  {summary['n_pairs_total_gold']}")
    print(f"Pairs skipped:     {summary['n_pairs_skipped']}")
    print(f"Missing policy:    {summary['missing_policy']}")
    print("\nOverall Metrics:")
    print(f"  Accuracy:        {summary['accuracy']:.4f}")
    print(f"  Micro Precision: {summary['micro_precision']:.4f}")
    print(f"  Micro Recall:    {summary['micro_recall']:.4f}")
    print(f"  Micro F1:        {summary['micro_f1']:.4f}")
    print(f"  Macro F1:        {summary['macro_f1']:.4f}")
    print("=" * 60)

    print("\nPer-Label Breakdown:")
    print(per_label_df.to_string(index=False))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(conf_df.to_string())
    print("\nPer-Aspect Breakdown:")
    print(per_aspect_df.to_string(index=False))
    print("=" * 60)

    # Save outputs
    if args.output_summary:
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to {args.output_summary}")

    if args.output_per_label:
        per_label_df.to_csv(args.output_per_label, index=False)
        print(f"[OK] Per-label metrics saved to {args.output_per_label}")

    if args.output_confusion:
        conf_df.to_csv(args.output_confusion)
        print(f"[OK] Confusion matrix saved to {args.output_confusion}")

    if args.output_per_aspect:
        per_aspect_df.to_csv(args.output_per_aspect, index=False)
        print(f"[OK] Per-aspect metrics saved to {args.output_per_aspect}")


if __name__ == "__main__":
    main()