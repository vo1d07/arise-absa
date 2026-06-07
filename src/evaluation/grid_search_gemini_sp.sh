#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Gemini / Google grid search for S-IPV.
#
# Usage example:
#   bash grid_search_gemini_sipv.sh \
#     --in_csv outputs/baseline_test_seed42_sipv.csv \
#     --out_dir outputs/grid_search_gemini_sipv \
#     --runs 1 \
#     --tau_sent_list 0.4,0.5,0.6 \
#     --t_max_list 1,2,3 \
#     --missing_policy wrong
#
# Optional parameters:
#   --missing_policy {wrong|skip}  : How to handle missing sentiment predictions during evaluation
#                                     "wrong" = treat missing as error (default)
#                                     "skip"  = skip missing predictions

SIPV_IN_CSV="outputs/SIPV_evalset.csv"
OUT_DIR="outputs/grid_search_gemini_sipv"
RUNS=1

# Model choice
LLM_MODEL="gemini-2.5-flash"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# Grid defaults (comma-separated)
TAU_SENT_LIST="0.5"
T_MAX_LIST="3"

# Evaluation option
MISSING_POLICY="wrong"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in_csv) SIPV_IN_CSV="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --llm_model) LLM_MODEL="$2"; shift 2 ;;
    --google_api_key) GOOGLE_API_KEY="$2"; shift 2 ;;
    --tau_sent_list) TAU_SENT_LIST="$2"; shift 2 ;;
    --t_max_list) T_MAX_LIST="$2"; shift 2 ;;
    --missing_policy) MISSING_POLICY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR/sipv" "$OUT_DIR/metrics" "$OUT_DIR/logs"

LEADERBOARD_CSV="$OUT_DIR/leaderboard.csv"
PARAMS_CSV="$OUT_DIR/grid_params.csv"
FAILED_CSV="$OUT_DIR/failed_runs.csv"
RANKED_CSV="$OUT_DIR/leaderboard_ranked_by_macro_f1.csv"

rm -f "$FAILED_CSV" "$RANKED_CSV"

echo "config_id,run,tau_sent,t_max,accuracy,micro_f1,macro_f1,micro_precision,micro_recall,n_pairs_used,pred_csv,summary_json" > "$LEADERBOARD_CSV"
echo "config_id,tau_sent,t_max" > "$PARAMS_CSV"
echo "config_id,run,tau_sent,t_max,stage,log_file" > "$FAILED_CSV"

refresh_ranked_leaderboard() {
  python3 - "$LEADERBOARD_CSV" "$RANKED_CSV" << 'PY'
import csv
import sys

in_csv = sys.argv[1]
out_csv = sys.argv[2]

with open(in_csv, "r", encoding="utf-8") as f:
  rows = list(csv.DictReader(f))

rows.sort(key=lambda x: (float(x.get("macro_f1", 0.0)), float(x.get("micro_f1", 0.0)), float(x.get("accuracy", 0.0))), reverse=True)

fieldnames = [
  "rank",
  "config_id",
  "run",
  "tau_sent",
  "t_max",
  "macro_f1",
  "micro_f1",
  "accuracy",
  "micro_precision",
  "micro_recall",
  "n_pairs_used",
  "pred_csv",
  "summary_json",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
  writer = csv.DictWriter(f, fieldnames=fieldnames)
  writer.writeheader()
  for i, row in enumerate(rows, start=1):
    writer.writerow({
      "rank": i,
      "config_id": row.get("config_id", ""),
      "run": row.get("run", ""),
      "tau_sent": row.get("tau_sent", ""),
      "t_max": row.get("t_max", ""),
      "macro_f1": row.get("macro_f1", ""),
      "micro_f1": row.get("micro_f1", ""),
      "accuracy": row.get("accuracy", ""),
      "micro_precision": row.get("micro_precision", ""),
      "micro_recall": row.get("micro_recall", ""),
      "n_pairs_used": row.get("n_pairs_used", ""),
      "pred_csv": row.get("pred_csv", ""),
      "summary_json": row.get("summary_json", ""),
    })
PY
}

run_sipv() {
  if [[ -n "$GOOGLE_API_KEY" ]]; then
    GOOGLE_API_KEY="$GOOGLE_API_KEY" python3 SP_google.py "$@"
  else
    python3 SP_google.py "$@"
  fi
}

IFS=',' read -r -a ARR_TAU_SENT <<< "$TAU_SENT_LIST"
IFS=',' read -r -a ARR_T_MAX <<< "$T_MAX_LIST"

GRID_SIZE=$(( ${#ARR_TAU_SENT[@]} * ${#ARR_T_MAX[@]} ))
TOTAL_RUNS=$(( GRID_SIZE * RUNS ))

print_progress() {
  local done="$1"
  local total="$2"
  local width=32
  local percent=0
  local filled=0
  local bar=""
  local empty=""

  if [[ "$total" -gt 0 ]]; then
    percent=$(( done * 100 / total ))
    filled=$(( done * width / total ))
  fi

  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  empty=$(printf '%*s' "$((width - filled))" '' | tr ' ' '-')
  printf "\rProgress: [%s%s] %3d%% (%d/%d)" "$bar" "$empty" "$percent" "$done" "$total"
}

echo "Gemini S-IPV grid search started"
echo "Input CSV: $SIPV_IN_CSV"
echo "Output dir: $OUT_DIR"
echo "Grid size: $GRID_SIZE configs"
echo "Runs per config: $RUNS"
print_progress 0 "$TOTAL_RUNS"

cfg_idx=0
completed_runs=0
for tau_sent in "${ARR_TAU_SENT[@]}"; do
  for t_max in "${ARR_T_MAX[@]}"; do
    cfg_idx=$((cfg_idx + 1))
    cfg_id=$(printf "G%04d" "$cfg_idx")

    echo "$cfg_id,$tau_sent,$t_max" >> "$PARAMS_CSV"

    echo "===== CONFIG $cfg_id / $(printf "%04d" "$GRID_SIZE") ====="
    echo "tau_sent=$tau_sent t_max=$t_max"

    for r in $(seq 1 "$RUNS"); do
      pred_csv="$OUT_DIR/sipv/${cfg_id}_run${r}.csv"
      memory_json="$OUT_DIR/sipv/${cfg_id}_run${r}_memory.json"
      summary_json="$OUT_DIR/metrics/${cfg_id}_run${r}_sipv_summary.json"
      per_label_csv="$OUT_DIR/metrics/${cfg_id}_run${r}_sipv_per_label.csv"
      confusion_csv="$OUT_DIR/metrics/${cfg_id}_run${r}_sipv_confusion.csv"
      per_aspect_csv="$OUT_DIR/metrics/${cfg_id}_run${r}_sipv_per_aspect.csv"
      log_file="$OUT_DIR/logs/${cfg_id}_run${r}.log"
      failed_stage=""

      echo "  -> run $r/$RUNS"

      if ! run_sipv \
        --in_csv "$SIPV_IN_CSV" \
        --out_csv "$pred_csv" \
        --out_memory "$memory_json" \
        --llm_model "$LLM_MODEL" \
        --google_api_key "$GOOGLE_API_KEY" \
        --tau_sent "$tau_sent" \
        --t_max "$t_max" \
        --missing_policy "$MISSING_POLICY" > "$log_file" 2>&1; then
        failed_stage="sipv_inference"
      fi

      if [[ -z "$failed_stage" ]]; then
        if ! python3 evaluation_SIPV.py \
          --in_csv "$pred_csv" \
          --gold_col gold_struct \
          --pred_col sentiment_final \
          --missing_policy "$MISSING_POLICY" \
          --output_summary "$summary_json" \
          --output_per_label "$per_label_csv" \
          --output_confusion "$confusion_csv" \
          --output_per_aspect "$per_aspect_csv" >> "$log_file" 2>&1; then
          failed_stage="sipv_eval"
        fi
      fi

      if [[ -z "$failed_stage" ]]; then
        if ! python3 - "$summary_json" "$LEADERBOARD_CSV" "$cfg_id" "$r" "$tau_sent" "$t_max" "$pred_csv" << 'PY'
import json
import sys

summary_json = sys.argv[1]
leaderboard_csv = sys.argv[2]
cfg_id = sys.argv[3]
run = sys.argv[4]
tau_sent = sys.argv[5]
t_max = sys.argv[6]
pred_csv = sys.argv[7]

with open(summary_json, "r", encoding="utf-8") as f:
    s = json.load(f)

row = [
    cfg_id,
    run,
    tau_sent,
    t_max,
    str(s.get("accuracy", 0.0)),
    str(s.get("micro_f1", 0.0)),
    str(s.get("macro_f1", 0.0)),
    str(s.get("micro_precision", 0.0)),
    str(s.get("micro_recall", 0.0)),
    str(s.get("n_pairs_used", 0)),
    pred_csv,
    summary_json,
]

with open(leaderboard_csv, "a", encoding="utf-8") as f:
    f.write(",".join(row) + "\n")
PY
        then
          failed_stage="leaderboard_append"
        fi
      fi

      if [[ -n "$failed_stage" ]]; then
        echo "[WARN] ${cfg_id} run ${r} failed at ${failed_stage}. See ${log_file}" >&2
        echo "$cfg_id,$r,$tau_sent,$t_max,$failed_stage,$log_file" >> "$FAILED_CSV"
      fi

      completed_runs=$((completed_runs + 1))
      print_progress "$completed_runs" "$TOTAL_RUNS"
    done

    refresh_ranked_leaderboard
    echo "  -> ranked leaderboard updated: $RANKED_CSV"
  done
done

printf "\n"

refresh_ranked_leaderboard

echo "Grid search completed."
echo "Params table: $PARAMS_CSV"
echo "Leaderboard: $LEADERBOARD_CSV"
echo "Ranked leaderboard: $RANKED_CSV"