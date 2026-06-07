#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_incremental_contrib_google.sh \
#     --aipv_in_csv datasets/dataset_for_aipv.csv \
#     --sipv_in_csv outputs/AIPV_output.csv \
#     --out_dir outputs/incremental_contrib_google \
#     --runs 1 \
#     --threshold 0.5 \
#     --tau_sent 0.5

# Defaults tuned for Google-based scripts
AIPV_IN_CSV="datasets/dataset.csv"
SIPV_IN_CSV=""
OUT_DIR="outputs/incremental_contrib_google"
RUNS=1
THRESHOLD=0.4
TAU_SENT=0.5
P_MODEL="gemini-2.5-flash"
V_MODEL="gemini-2.5-flash"
LLM_MODEL="gemini-2.5-flash"
EMBED_MODEL="all-MiniLM-L6-v2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in_csv) AIPV_IN_CSV="$2"; shift 2 ;;
    --aipv_in_csv) AIPV_IN_CSV="$2"; shift 2 ;;
    --sipv_in_csv) SIPV_IN_CSV="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --tau_sent) TAU_SENT="$2"; shift 2 ;;
    --p_model) P_MODEL="$2"; shift 2 ;;
    --v_model) V_MODEL="$2"; shift 2 ;;
    --llm_model) LLM_MODEL="$2"; shift 2 ;;
    --embed_model) EMBED_MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR/aipv" "$OUT_DIR/sipv" "$OUT_DIR/metrics"

for r in $(seq 1 "$RUNS"); do
  echo "===== RUN $r / $RUNS ====="

  if [[ -n "$SIPV_IN_CSV" ]]; then
    SIPV_INPUT_CSV="$SIPV_IN_CSV"
  else
    SIPV_INPUT_CSV="$OUT_DIR/aipv/A0_run${r}.csv"
  fi

  # A-IPV variants (Google implementation files)
  python3 AP_google.py \
    --in_csv "$AIPV_IN_CSV" \
    --out_csv "$OUT_DIR/aipv/A0_run${r}.csv" \
    --threshold "$THRESHOLD" \
    --embed_model "$EMBED_MODEL" \
    --p_model "$P_MODEL" \
    --v_model "$V_MODEL" \
    --max_rounds 2 \
    --top_k_pos 5 \
    --top_m_neg 3

  python3 AP_google.py \
    --in_csv "$AIPV_IN_CSV" \
    --out_csv "$OUT_DIR/aipv/A1_run${r}.csv" \
    --threshold "$THRESHOLD" \
    --embed_model "$EMBED_MODEL" \
    --p_model "$P_MODEL" \
    --v_model "$V_MODEL" \
    --max_rounds 1 \
    --top_k_pos 5 \
    --top_m_neg 3

  python3 AP_google.py \
    --in_csv "$AIPV_IN_CSV" \
    --out_csv "$OUT_DIR/aipv/A2_run${r}.csv" \
    --threshold "$THRESHOLD" \
    --embed_model "$EMBED_MODEL" \
    --p_model "$P_MODEL" \
    --v_model "$V_MODEL" \
    --max_rounds 1 \
    --top_k_pos 5 \
    --top_m_neg 0

  python3 AP_google.py \
    --in_csv "$AIPV_IN_CSV" \
    --out_csv "$OUT_DIR/aipv/A3_run${r}.csv" \
    --threshold "$THRESHOLD" \
    --embed_model "$EMBED_MODEL" \
    --p_model "$P_MODEL" \
    --v_model "$V_MODEL" \
    --max_rounds 1 \
    --top_k_pos 0 \
    --top_m_neg 0

  python3 AP_google.py \
    --in_csv "$AIPV_IN_CSV" \
    --out_csv "$OUT_DIR/aipv/A4_run${r}.csv" \
    --threshold "$THRESHOLD" \
    --embed_model "$EMBED_MODEL" \
    --p_model "$P_MODEL" \
    --v_model "$V_MODEL" \
    --disable_validator \
    --max_rounds 1 \
    --top_k_pos 0 \
    --top_m_neg 0

  # evaluate A groups
  for g in A0 A1 A2 A3 A4; do
    python3 evaluation_AIPV.py \
      --in_csv "$OUT_DIR/aipv/${g}_run${r}.csv" \
      --gold_col gold_struct \
      --pred_col candidates_final \
      --output_summary "$OUT_DIR/metrics/${g}_run${r}_aipv_summary.json" \
      --output_details "$OUT_DIR/metrics/${g}_run${r}_aipv_per_aspect.csv"
  done

  # S-IPV groups (Google implementation files)
  python3 SP_google.py \
    --in_csv "$SIPV_INPUT_CSV" \
    --out_csv "$OUT_DIR/sipv/S0_run${r}.csv" \
    --out_memory "$OUT_DIR/sipv/S0_run${r}_memory.json" \
    --llm_model "$LLM_MODEL" \
    --tau_sent "$TAU_SENT" \
    --t_max 3

  python3 SP_google.py \
    --in_csv "$SIPV_INPUT_CSV" \
    --out_csv "$OUT_DIR/sipv/S1_run${r}.csv" \
    --out_memory "$OUT_DIR/sipv/S1_run${r}_memory.json" \
    --llm_model "$LLM_MODEL" \
    --tau_sent "$TAU_SENT" \
    --t_max 1

  python3 SP_google.py \
    --in_csv "$SIPV_INPUT_CSV" \
    --out_csv "$OUT_DIR/sipv/S2_run${r}.csv" \
    --out_memory "$OUT_DIR/sipv/S2_run${r}_memory.json" \
    --llm_model "$LLM_MODEL" \
    --tau_sent "$TAU_SENT" \
    --t_max 1 \
    --disable_memory

  python3 SP_google.py \
    --in_csv "$SIPV_INPUT_CSV" \
    --out_csv "$OUT_DIR/sipv/S3_run${r}.csv" \
    --out_memory "$OUT_DIR/sipv/S3_run${r}_memory.json" \
    --llm_model "$LLM_MODEL" \
    --tau_sent "$TAU_SENT" \
    --t_max 1 \
    --disable_validator \
    --disable_memory

  # evaluate S groups
  for g in S0 S1 S2 S3; do
    python3 evaluation_SIPV.py \
      --in_csv "$OUT_DIR/sipv/${g}_run${r}.csv" \
      --gold_col gold_struct \
      --pred_col sentiment_final \
      --output_summary "$OUT_DIR/metrics/${g}_run${r}_sipv_summary.json" \
      --output_per_label "$OUT_DIR/metrics/${g}_run${r}_sipv_per_label.csv" \
      --output_confusion "$OUT_DIR/metrics/${g}_run${r}_sipv_confusion.csv" \
      --output_per_aspect "$OUT_DIR/metrics/${g}_run${r}_sipv_per_aspect.csv"
  done

done

echo "All Google runs completed."
python3 summarize_incremental_contrib.py --metrics_dir "$OUT_DIR/metrics" --out_csv "$OUT_DIR/incremental_contrib_summary.csv"
echo "Summary saved to $OUT_DIR/incremental_contrib_summary.csv"
