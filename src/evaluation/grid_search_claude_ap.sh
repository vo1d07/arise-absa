#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Claude / Anthropic grid search for A-IPV.
#
# Usage example:
#   bash grid_search_claude_aipv.sh \
#     --in_csv datasets/dataset.csv \
#     --out_dir outputs/grid_search_claude_aipv \
#     --runs 1 \
#     --thresholds 0.4,0.5,0.6 \
#     --p_top_ks 4,6,8 \
#     --max_rounds_list 2,3,4 \
#     --top_k_pos_list 3,5,8 \
#     --top_m_neg_list 2,5,8

AIPV_IN_CSV="datasets/dataset.csv"
OUT_DIR="outputs/grid_search_claude_aipv"
RUNS=1

# Model choices
P_MODEL="claude-haiku-4-5-20251001"
V_MODEL="claude-haiku-4-5-20251001"
EMBED_MODEL="all-MiniLM-L6-v2"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Grid defaults (comma-separated)
THRESHOLDS="0.4"
P_TOP_KS="8"
MAX_ROUNDS_LIST="2"
TOP_K_POS_LIST="5"
TOP_M_NEG_LIST="3"

SLEEP_S=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--in_csv) AIPV_IN_CSV="$2"; shift 2 ;;
		--out_dir) OUT_DIR="$2"; shift 2 ;;
		--runs) RUNS="$2"; shift 2 ;;
		--p_model) P_MODEL="$2"; shift 2 ;;
		--v_model) V_MODEL="$2"; shift 2 ;;
		--embed_model) EMBED_MODEL="$2"; shift 2 ;;
		--anthropic_api_key) ANTHROPIC_API_KEY="$2"; shift 2 ;;
		--thresholds) THRESHOLDS="$2"; shift 2 ;;
		--p_top_ks) P_TOP_KS="$2"; shift 2 ;;
		--max_rounds_list) MAX_ROUNDS_LIST="$2"; shift 2 ;;
		--top_k_pos_list) TOP_K_POS_LIST="$2"; shift 2 ;;
		--top_m_neg_list) TOP_M_NEG_LIST="$2"; shift 2 ;;
		--sleep_s) SLEEP_S="$2"; shift 2 ;;
		*) echo "Unknown argument: $1"; exit 1 ;;
	esac
done

mkdir -p "$OUT_DIR/aipv" "$OUT_DIR/metrics" "$OUT_DIR/logs"

LEADERBOARD_CSV="$OUT_DIR/leaderboard.csv"
PARAMS_CSV="$OUT_DIR/grid_params.csv"
FAILED_CSV="$OUT_DIR/failed_runs.csv"
RANKED_CSV="$OUT_DIR/leaderboard_ranked_by_macro_f1.csv"

rm -f "$FAILED_CSV" "$RANKED_CSV"

echo "config_id,run,threshold,p_top_k,max_rounds,top_k_pos,top_m_neg,micro_f1,macro_f1,subset_accuracy,micro_precision,micro_recall,pred_csv,summary_json" > "$LEADERBOARD_CSV"
echo "config_id,threshold,p_top_k,max_rounds,top_k_pos,top_m_neg" > "$PARAMS_CSV"
echo "config_id,run,threshold,p_top_k,max_rounds,top_k_pos,top_m_neg,stage,log_file" > "$FAILED_CSV"

refresh_ranked_leaderboard() {
	python3 - "$LEADERBOARD_CSV" "$RANKED_CSV" << 'PY'
import csv
import sys

in_csv = sys.argv[1]
out_csv = sys.argv[2]

with open(in_csv, "r", encoding="utf-8") as f:
	rows = list(csv.DictReader(f))

rows.sort(key=lambda x: (float(x.get("macro_f1", 0.0)), float(x.get("micro_f1", 0.0))), reverse=True)

fieldnames = [
	"rank",
	"config_id",
	"run",
	"threshold",
	"p_top_k",
	"max_rounds",
	"top_k_pos",
	"top_m_neg",
	"macro_f1",
	"micro_f1",
	"subset_accuracy",
	"micro_precision",
	"micro_recall",
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
			"threshold": row.get("threshold", ""),
			"p_top_k": row.get("p_top_k", ""),
			"max_rounds": row.get("max_rounds", ""),
			"top_k_pos": row.get("top_k_pos", ""),
			"top_m_neg": row.get("top_m_neg", ""),
			"macro_f1": row.get("macro_f1", ""),
			"micro_f1": row.get("micro_f1", ""),
			"subset_accuracy": row.get("subset_accuracy", ""),
			"micro_precision": row.get("micro_precision", ""),
			"micro_recall": row.get("micro_recall", ""),
			"pred_csv": row.get("pred_csv", ""),
			"summary_json": row.get("summary_json", ""),
		})
PY
}

IFS=',' read -r -a ARR_THRESHOLDS <<< "$THRESHOLDS"
IFS=',' read -r -a ARR_P_TOP_KS <<< "$P_TOP_KS"
IFS=',' read -r -a ARR_MAX_ROUNDS <<< "$MAX_ROUNDS_LIST"
IFS=',' read -r -a ARR_TOP_K_POS <<< "$TOP_K_POS_LIST"
IFS=',' read -r -a ARR_TOP_M_NEG <<< "$TOP_M_NEG_LIST"

GRID_SIZE=$(( ${#ARR_THRESHOLDS[@]} * ${#ARR_P_TOP_KS[@]} * ${#ARR_MAX_ROUNDS[@]} * ${#ARR_TOP_K_POS[@]} * ${#ARR_TOP_M_NEG[@]} ))
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

run_aipv() {
	if [[ -n "$ANTHROPIC_API_KEY" ]]; then
		ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python3 AP_anthropic.py "$@"
	else
		python3 AP_anthropic.py "$@"
	fi
}

echo "Claude A-IPV grid search started"
echo "Input CSV: $AIPV_IN_CSV"
echo "Output dir: $OUT_DIR"
echo "Grid size: $GRID_SIZE configs"
echo "Runs per config: $RUNS"
print_progress 0 "$TOTAL_RUNS"

cfg_idx=0
completed_runs=0
for threshold in "${ARR_THRESHOLDS[@]}"; do
	for p_top_k in "${ARR_P_TOP_KS[@]}"; do
		for max_rounds in "${ARR_MAX_ROUNDS[@]}"; do
			for top_k_pos in "${ARR_TOP_K_POS[@]}"; do
				for top_m_neg in "${ARR_TOP_M_NEG[@]}"; do
					cfg_idx=$((cfg_idx + 1))
					cfg_id=$(printf "G%04d" "$cfg_idx")

					echo "$cfg_id,$threshold,$p_top_k,$max_rounds,$top_k_pos,$top_m_neg" >> "$PARAMS_CSV"

					echo "===== CONFIG $cfg_id / $(printf "%04d" "$GRID_SIZE") ====="
					echo "threshold=$threshold p_top_k=$p_top_k max_rounds=$max_rounds top_k_pos=$top_k_pos top_m_neg=$top_m_neg"

					for r in $(seq 1 "$RUNS"); do
						pred_csv="$OUT_DIR/aipv/${cfg_id}_run${r}.csv"
						summary_json="$OUT_DIR/metrics/${cfg_id}_run${r}_aipv_summary.json"
						details_csv="$OUT_DIR/metrics/${cfg_id}_run${r}_aipv_per_aspect.csv"
						log_file="$OUT_DIR/logs/${cfg_id}_run${r}.log"
						failed_stage=""

						echo "  -> run $r/$RUNS"

						if ! run_aipv \
							--in_csv "$AIPV_IN_CSV" \
							--out_csv "$pred_csv" \
							--threshold "$threshold" \
							--embed_model "$EMBED_MODEL" \
							--p_model "$P_MODEL" \
							--v_model "$V_MODEL" \
							--p_top_k "$p_top_k" \
							--max_rounds "$max_rounds" \
							--top_k_pos "$top_k_pos" \
							--top_m_neg "$top_m_neg" \
							--sleep_s "$SLEEP_S" > "$log_file" 2>&1; then
							failed_stage="aipv_inference"
						fi

						if [[ -z "$failed_stage" ]]; then
							if ! python3 evaluation_AIPV.py \
								--in_csv "$pred_csv" \
								--gold_col gold_struct \
								--pred_col candidates_final \
								--output_summary "$summary_json" \
								--output_details "$details_csv" >> "$log_file" 2>&1; then
								failed_stage="aipv_eval"
							fi
						fi

						if [[ -z "$failed_stage" ]]; then
							if ! python3 - "$summary_json" "$LEADERBOARD_CSV" "$cfg_id" "$r" "$threshold" "$p_top_k" "$max_rounds" "$top_k_pos" "$top_m_neg" "$pred_csv" << 'PY'
import json
import sys

summary_json = sys.argv[1]
leaderboard_csv = sys.argv[2]
cfg_id = sys.argv[3]
run = sys.argv[4]
threshold = sys.argv[5]
p_top_k = sys.argv[6]
max_rounds = sys.argv[7]
top_k_pos = sys.argv[8]
top_m_neg = sys.argv[9]
pred_csv = sys.argv[10]

with open(summary_json, "r", encoding="utf-8") as f:
	s = json.load(f)

row = [
	cfg_id,
	run,
	threshold,
	p_top_k,
	max_rounds,
	top_k_pos,
	top_m_neg,
	str(s.get("micro_f1", 0.0)),
	str(s.get("macro_f1", 0.0)),
	str(s.get("subset_accuracy", 0.0)),
	str(s.get("micro_precision", 0.0)),
	str(s.get("micro_recall", 0.0)),
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
							echo "$cfg_id,$r,$threshold,$p_top_k,$max_rounds,$top_k_pos,$top_m_neg,$failed_stage,$log_file" >> "$FAILED_CSV"
						fi

						completed_runs=$((completed_runs + 1))
						print_progress "$completed_runs" "$TOTAL_RUNS"
						done

					refresh_ranked_leaderboard
					echo "  -> ranked leaderboard updated: $RANKED_CSV"
				done
				done
		done
	done
done

printf "\n"

refresh_ranked_leaderboard

echo "Grid search completed."
echo "Params table: $PARAMS_CSV"
echo "Leaderboard: $LEADERBOARD_CSV"
echo "Ranked leaderboard: $RANKED_CSV"