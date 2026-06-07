# ARISE-ABSA

ARISE-ABSA is a repository for aspect-based sentiment analysis (ABSA) on student feedback about university mental health services.

The repository contains:
- ARISE pipelines for aspect prediction and sentiment prediction using iterative proposer-validator loops with LLMs
- Multiple baseline systems (BERT/DeBERTa/classical ML/prompt-based)
- Evaluation scripts and experiment automation scripts
- Dataset assets and analysis notebooks

## Project Scope

This project focuses on two linked subtasks:

1. Aspect Prediction (AP)
- Predict which service aspects are mentioned in each feedback text.
- Aspect schema used in ARISE scripts:
	- On-campus Service
	- Counseling Service
	- Mental Health Service
	- Wellness Service
	- Therapy Service
	- Hotline Service
	- Service Availability
	- General

2. Sentiment Prediction (SP)
- Predict sentiment for each detected aspect.
- Labels in ARISE scripts are normalized to NEG/NEU/POS.

## Repository Structure

Current repository file layout:

~~~text
appendix/
	appendix.pdf
datasets/
	annotator1.jsonl
	annotator2.jsonl
	annotator3.jsonl
	clustered_aspects_original.csv
	clustered_aspects_refined.csv
	dataset.csv
	low_level_aspects.csv
    SP_evalset.csv
src/
	arise/
		ap/
			AP_anthropic.py
			AP_google.py
			AP_openai.py
		sp/
			SP_anthropic.py
			SP_google.py
			SP_openai.py
	baseline/
		bert_asc.py
		chatabsa.py
		deberta.py
		llm_only.py
		syn-chain.py
		xgboost_baseline.py
		ap/
			LR.py
			pretrained.py
			SVM.py
		sp/
			LR.py
			pretrained.py
			SVM.py
	datasets/
		dataset_analysis.ipynb
		low_level_aspect_analysis.ipynb
		low_level_aspect_cluster.ipynb
		majority_voting.ipynb
		stats_analysis.ipynb
	evaluation/
		evaluation_AP.py
		evaluation_SP.py
		grid_search_claude_ap.sh
		grid_search_claude_sp.sh
		grid_search_gemini_ap.sh
		grid_search_gemini_sp.sh
		grid_search_openai_ap.sh
		grid_search_openai_sp.sh
		run_incremental_contrib_anthropic.sh
		run_incremental_contrib_google.sh
		run_incremental_contrib_openai.sh
		summarize_incremental_contrib.py
~~~

## Data Overview

### Main dataset

File: datasets/dataset.csv

Columns:
- text
- merged_entities

### SP evaluation datsset

File: datasets/SP_evalset.csv

Used to evaluate SP performance by given ground-truth aspect label.

### Annotator files

Files:
- datasets/annotator1.jsonl
- datasets/annotator2.jsonl
- datasets/annotator3.jsonl

JSONL keys:
- id
- text
- doc_id
- Comments
- cats
- entities

## ARISE Pipeline Scripts

### AP (Aspect Prediction)

Provider-specific scripts:
- src/arise/ap/AP_openai.py
- src/arise/ap/AP_anthropic.py
- src/arise/ap/AP_google.py

Common important options:
- --in_csv
- --out_csv
- --threshold
- --embed_model
- --p_model
- --v_model
- --p_top_k
- --max_rounds
- --top_k_pos
- --top_m_neg
- --disable_validator

### SP (Sentiment Prediction)

Provider-specific scripts:
- src/arise/sp/SP_openai.py
- src/arise/sp/SP_anthropic.py
- src/arise/sp/SP_google.py

Common important options:
- --in_csv
- --out_csv
- --out_memory
- --t_max
- --tau_sent
- --disable_validator
- --disable_memory
- --missing_policy
- provider API key argument

## Environment Setup

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
	pandas numpy tqdm scikit-learn \
	torch transformers sentence-transformers xgboost \
	openai anthropic google-generativeai httpx pyarrow
~~~

Optional:

~~~bash
pip install spacy
python -m spacy download en_core_web_sm
~~~

## API Keys

Set the API key for your selected provider.

~~~bash
# OpenAI
export OPENAI_API_KEY="openai_key"

# Anthropic
export ANTHROPIC_API_KEY="anthropic_key"

# Google Gemini
export GOOGLE_API_KEY="google_key"
~~~

### Option A: OpenAI AP then SP

~~~bash
python src/arise/ap/AP_openai.py \
	--in_csv datasets/dataset.csv \
	--out_csv outputs/AP_openai.csv \
	--threshold 0.5 \
	--p_model gpt-4o-mini \
	--v_model gpt-4o-mini

python src/arise/sp/SP_openai.py \
	--in_csv datasets/SP_evalset.csv \
	--out_csv outputs/SP_openai.csv \
	--out_memory outputs/sentiment_memory_openai.json \
	--llm_model gpt-4o-mini \
	--tau_sent 0.5 \
	--t_max 3
~~~

### Option B: Anthropic AP then SP

~~~bash
python src/arise/ap/AP_anthropic.py \
	--in_csv datasets/dataset.csv \
	--out_csv outputs/AP_anthropic.csv

python src/arise/sp/SP_anthropic.py \
	--in_csv datasets/SP_evalset.csv \
	--out_csv outputs/SP_anthropic.csv \
	--out_memory outputs/sentiment_memory_anthropic.json
~~~

### Option C: Google AP then SP

~~~bash
python src/arise/ap/AP_google.py \
	--in_csv datasets/dataset.csv \
	--out_csv outputs/AP_google.csv

python src/arise/sp/SP_google.py \
	--in_csv datasets/SP_evalset.csv \
	--out_csv outputs/SP_google.csv \
	--out_memory outputs/sentiment_memory_google.json
~~~

## Evaluation

AP evaluation:

~~~bash
python src/evaluation/evaluation_AP.py \
	--in_csv outputs/AP_openai.csv \
	--gold_col gold_struct \
	--pred_col candidates_final \
	--output_summary outputs/ap_summary.json \
	--output_details outputs/ap_per_aspect.csv
~~~

SP evaluation:

~~~bash
python src/evaluation/evaluation_SP.py \
	--in_csv outputs/SP_openai.csv \
	--gold_col gold_struct \
	--pred_col sentiment_final \
	--missing_policy wrong \
	--output_summary outputs/sp_summary.json \
	--output_per_label outputs/sp_per_label.csv \
	--output_confusion outputs/sp_confusion.csv \
	--output_per_aspect outputs/sp_per_aspect.csv
~~~

## Baselines

Baseline scripts are under src/baseline and include:
- bert_asc.py
- deberta.py
- chatabsa.py
- llm_only.py
- syn-chain.py
- xgboost_baseline.py
- Classical models in src/baseline/ap and src/baseline/sp

## Experiment Automation

Automation scripts are in src/evaluation:
- grid_search_claude_ap.sh
- grid_search_claude_sp.sh
- grid_search_gemini_ap.sh
- grid_search_gemini_sp.sh
- grid_search_openai_ap.sh
- grid_search_openai_sp.sh
- run_incremental_contrib_anthropic.sh
- run_incremental_contrib_google.sh
- run_incremental_contrib_openai.sh
- summarize_incremental_contrib.py

Incremental contribution summary example:

~~~bash
python src/evaluation/summarize_incremental_contrib.py \
	--metrics_dir outputs/incremental_contrib/metrics \
	--out_csv outputs/incremental_contrib_summary.csv
~~~
