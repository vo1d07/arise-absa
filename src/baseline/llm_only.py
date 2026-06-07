import os
import re
import json
import ast
import time
import random
import argparse
import sys
import requests
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASQP_MA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ASQP_MA_DIR not in sys.path:
    sys.path.append(ASQP_MA_DIR)

from evaluation_AIPV import eval_a_ipv_aspect_detection
from evaluation_SIPV import eval_s_ipv_sentiment_conditional


# =========================================================
# Global label space
# =========================================================
ASPECTS_8 = [
    "On-campus Service",
    "Counseling Service",
    "Mental Health Service",
    "Wellness Service",
    "Therapy Service",
    "Hotline Service",
    "Service Availability",
    "General",
]

SENTIMENT_SET = ["NEG", "NEU", "POS"]

TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}


# =========================================================
# Utilities
# =========================================================
def normalize_aspect_list(x):
    if not isinstance(x, list):
        return []
    out = []
    for a in x:
        if a in ASPECTS_8 and a not in out:
            out.append(a)
    return out


def normalize_sentiment(x):
    mapping = {
        "NEG": "NEG",
        "NEU": "NEU",
        "POS": "POS",
        "Negative": "NEG",
        "Neutral": "NEU",
        "Positive": "POS",
        "negative": "NEG",
        "neutral": "NEU",
        "positive": "POS",
    }
    return mapping.get(x, None)


def safe_json_loads(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    unescaped = text.replace('""', '"')
    try:
        return json.loads(unescaped)
    except Exception:
        pass

    # first json object/array
    obj_match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if obj_match:
        candidate = obj_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate.replace('""', '"'))
            except Exception:
                return None

    return None


def extract_sentiment_label(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None

    parsed = safe_json_loads(text)
    if isinstance(parsed, dict):
        label = normalize_sentiment(parsed.get("label", None))
        if label is not None:
            return label

    cleaned = text.strip()
    if not cleaned:
        return None

    m = re.search(r"\b(NEG|NEU|POS)\b", cleaned, flags=re.I)
    if m:
        return normalize_sentiment(m.group(1).upper())

    m = re.search(r"\b(negative|neutral|positive)\b", cleaned, flags=re.I)
    if m:
        return normalize_sentiment(m.group(1).title())

    return None


def _parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        sec = float(value)
        return max(0.0, sec)
    except Exception:
        # HTTP-date format is ignored here; fallback to exponential backoff.
        return None


def post_json_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int = 6,
    backoff_base_sec: float = 1.5,
    backoff_max_sec: float = 45.0,
) -> requests.Response:
    """
    POST JSON with retries for transient failures (e.g., 429 rate limit).
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if resp.status_code in TRANSIENT_HTTP_STATUS:
                if attempt >= max_retries:
                    resp.raise_for_status()

                retry_after = _parse_retry_after_seconds(resp.headers.get("Retry-After"))
                if retry_after is None:
                    retry_after = min(
                        backoff_max_sec,
                        backoff_base_sec * (2 ** attempt) + random.uniform(0, 0.5),
                    )
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                raise

            status_code = None
            if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                status_code = exc.response.status_code

            if status_code is not None and status_code not in TRANSIENT_HTTP_STATUS:
                raise

            sleep_sec = min(
                backoff_max_sec,
                backoff_base_sec * (2 ** attempt) + random.uniform(0, 0.5),
            )
            time.sleep(sleep_sec)

    raise RuntimeError("Unexpected retry loop exit")


def parse_struct_cell(cell: Any) -> Dict[str, Any]:
    if isinstance(cell, dict):
        return cell
    if cell is None:
        return {"aspect_set": [], "by_aspect": {}}
    if isinstance(cell, float) and pd.isna(cell):
        return {"aspect_set": [], "by_aspect": {}}
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return {"aspect_set": [], "by_aspect": {}}
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"aspect_set": [], "by_aspect": {}}


# =========================================================
# LLM Client Abstraction
# =========================================================
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        pass


# =========================================================
# OpenAI-compatible client
# Works for:
# - OpenAI
# - many custom/vLLM servers
# - OpenRouter-like endpoints
# - local OpenAI-compatible serving
# =========================================================
@dataclass
class OpenAICompatibleClient(BaseLLMClient):
    base_url: str
    api_key: str
    model: str
    timeout: int = 120

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        resp = post_json_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            timeout=self.timeout,
        )
        data = resp.json()

        return data["choices"][0]["message"]["content"]


# =========================================================
# Anthropic client
# =========================================================
@dataclass
class AnthropicClient(BaseLLMClient):
    api_key: str
    model: str
    timeout: int = 120
    base_url: str = "https://api.anthropic.com/v1/messages"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        resp = post_json_with_retries(
            url=self.base_url,
            headers=headers,
            payload=payload,
            timeout=self.timeout,
        )
        data = resp.json()

        parts = data.get("content", [])
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        return "\n".join(texts).strip()


# =========================================================
# Gemini REST client
# =========================================================
@dataclass
class GeminiClient(BaseLLMClient):
    api_key: str
    model: str
    timeout: int = 120
    base_url_template: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        url = self.base_url_template.format(model=self.model)
        url = f"{url}?key={self.api_key}"

        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
            "systemInstruction": {
                "parts": [
                    {"text": system_prompt}
                ]
            },
            "contents": [
                {
                    "parts": [
                        {"text": user_prompt}
                    ]
                }
            ]
        }

        resp = post_json_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            timeout=self.timeout,
        )
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if "text" in p]
        return "\n".join(texts).strip()


# =========================================================
# Factory
# =========================================================
def build_llm_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMClient:
    provider = provider.lower()

    if provider == "openai_compatible":
        if not api_key:
            raise ValueError("api_key is required for openai_compatible")
        if not base_url:
            raise ValueError("base_url is required for openai_compatible")
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)

    if provider == "anthropic":
        if not api_key:
            raise ValueError("api_key is required for anthropic")
        return AnthropicClient(api_key=api_key, model=model)

    if provider == "gemini":
        if not api_key:
            raise ValueError("api_key is required for gemini")
        return GeminiClient(api_key=api_key, model=model)

    raise ValueError(f"Unsupported provider: {provider}")


# =========================================================
# AP LLM-only baseline
# =========================================================
A_IPV_SYSTEM_PROMPT = "You are a classifier."

def build_a_ipv_user_prompt(text: str) -> str:
    aspect_list = "\n".join([f"- {a}" for a in ASPECTS_8])
    return f"""Task: choose all mentioned aspects from the label set.

Label set:
{aspect_list}

Examples:
1. Input: "Counseling sessions are too short, need more availability"
   Output: {{"aspects": ["Counseling Service", "Service Availability"]}}

2. Input: "Group therapy works great for stress relief"
   Output: {{"aspects": ["Therapy Service"]}}

3. Input: "Overall the services are good"
   Output: {{"aspects": ["General"]}}

Text:
{text}

Output ONLY valid JSON in this format, with no additional text or markdown:
{{"aspects": ["label1", "label2"]}}
"""


def run_a_ipv_llm_baseline(
    df: pd.DataFrame,
    llm_client: BaseLLMClient,
    text_col: str = "text",
    output_col: str = "a_ipv_llm_pred",
    raw_output_col: str = "a_ipv_llm_raw",
    temperature: float = 0.0,
    max_tokens: int = 256,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    preds = []
    raws = []

    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="A-IPV LLM inference",
        leave=True,
    )
    for _, row in iterator:
        text = str(row[text_col])

        raw = llm_client.generate(
            system_prompt=A_IPV_SYSTEM_PROMPT,
            user_prompt=build_a_ipv_user_prompt(text),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raws.append(raw)

        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict):
            aspects = normalize_aspect_list(parsed.get("aspects", []))
        else:
            aspects = []

        preds.append(aspects)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    out_df = df.copy()
    out_df[raw_output_col] = raws
    out_df[output_col] = preds
    return out_df


# =========================================================
# SP LLM-only baseline
# Gold-aspect conditional version
# input: text + aspect
# output: sentiment
# =========================================================
S_IPV_SYSTEM_PROMPT = "You are a classifier."

def build_s_ipv_user_prompt(text: str, aspect: str) -> str:
    sentiment_list = ", ".join(SENTIMENT_SET)
    return f"""Classify sentiment for the target aspect.
Aspect: {aspect}
Labels: {sentiment_list}

Text:
{text}

Output JSON:
{{"label": "NEG"}}
"""


def run_s_ipv_llm_baseline(
    df: pd.DataFrame,
    llm_client: BaseLLMClient,
    text_col: str = "text",
    gold_col: str = "gold_struct",
    output_col: str = "s_ipv_llm_pred",
    raw_output_col: str = "s_ipv_llm_raw",
    temperature: float = 0.0,
    max_tokens: int = 128,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    pred_all = []
    raw_all = []

    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="S-IPV LLM inference",
        leave=True,
    )
    for _, row in iterator:
        text = str(row[text_col])
        gold = parse_struct_cell(row[gold_col])

        pred_dict = {}
        raw_dict = {}

        aspect_set = gold.get("aspect_set", []) or []

        for aspect in aspect_set:
            raw = llm_client.generate(
                system_prompt=S_IPV_SYSTEM_PROMPT,
                user_prompt=build_s_ipv_user_prompt(text, aspect),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_dict[aspect] = raw

            label = extract_sentiment_label(raw)

            pred_dict[aspect] = {"label": label}

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        pred_all.append(pred_dict)
        raw_all.append(raw_dict)

    out_df = df.copy()
    out_df[raw_output_col] = raw_all
    out_df[output_col] = pred_all
    return out_df


def main():
    ap = argparse.ArgumentParser(description="Pure-LLM baseline for A-IPV/S-IPV with unified VAL/TEST output style.")
    ap.add_argument("--task", type=str, choices=["aipv", "sipv"], required=True,
                    help="Which task to run: aipv or sipv")
    ap.add_argument("--in_csv", type=str, required=True,
                    help="Path to input CSV")
    ap.add_argument("--text_col", type=str, default="text",
                    help="Column name for text")
    ap.add_argument("--gold_col", type=str, default="gold_struct",
                    help="Column name for gold structure")

    ap.add_argument("--provider", type=str, default="openai_compatible",
                    choices=["openai_compatible", "anthropic", "gemini"],
                    help="LLM provider")
    ap.add_argument("--model", type=str, default="gpt-4o-mini",
                    help="Model name")
    ap.add_argument("--api_key", type=str, default=None,
                    help="API key (optional; if omitted, read from --api_key_env)")
    ap.add_argument("--api_key_env", type=str, default="API_KEY",
                    help="Environment variable name for API key")
    ap.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                    help="Base URL for openai_compatible provider")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="Override max tokens; task-specific defaults are used when omitted")
    ap.add_argument("--sleep_sec", type=float, default=0.0)
    ap.add_argument("--missing_policy", type=str, default="wrong", choices=["wrong", "skip"],
                    help="How to handle missing sentiment predictions for SIPV evaluation")

    ap.add_argument("--output_dir", type=str, default="runs/pure_llm",
                    help="Directory to save outputs")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(
            "Missing API key. Provide --api_key or set environment variable "
            f"{args.api_key_env}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    client = build_llm_client(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
    )

    df = pd.read_csv(args.in_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    if args.gold_col not in df.columns:
        raise ValueError(f"Missing gold column: {args.gold_col}")

    # Zero-shot evaluation on full dataset.
    df_test = df.reset_index(drop=True)

    if args.task == "aipv":
        pred_col = "candidates_final"
        raw_col = "a_ipv_llm_raw"
        max_tokens = args.max_tokens if args.max_tokens is not None else 256

        test_pred_df = run_a_ipv_llm_baseline(
            df_test,
            client,
            text_col=args.text_col,
            output_col=pred_col,
            raw_output_col=raw_col,
            temperature=args.temperature,
            max_tokens=max_tokens,
            sleep_sec=args.sleep_sec,
        )

        test_summary, test_per = eval_a_ipv_aspect_detection(
            test_pred_df,
            gold_col=args.gold_col,
            pred_col=pred_col,
            aspects_list=ASPECTS_8,
        )

    else:
        pred_col = "sentiment_final"
        raw_col = "s_ipv_llm_raw"
        max_tokens = args.max_tokens if args.max_tokens is not None else 256

        test_pred_df = run_s_ipv_llm_baseline(
            df_test,
            client,
            text_col=args.text_col,
            gold_col=args.gold_col,
            output_col=pred_col,
            raw_output_col=raw_col,
            temperature=args.temperature,
            max_tokens=max_tokens,
            sleep_sec=args.sleep_sec,
        )

        test_summary, test_per, test_conf, test_per_aspect = eval_s_ipv_sentiment_conditional(
            test_pred_df,
            gold_col=args.gold_col,
            pred_col=pred_col,
            missing_policy=args.missing_policy,
        )

    task_tag = args.task.upper()
    test_pred_df.to_csv(os.path.join(args.output_dir, f"{task_tag}_test_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, f"{task_tag}_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)
    test_per.to_csv(os.path.join(args.output_dir, f"{task_tag}_test_per.csv"), index=False)
    
    # Save per-aspect metrics for SIPV
    if args.task == "sipv":
        test_per_aspect.to_csv(os.path.join(args.output_dir, f"{task_tag}_test_per_aspect.csv"), index=False)

    print("TEST summary:", test_summary)
    print(test_per.head(20))

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()