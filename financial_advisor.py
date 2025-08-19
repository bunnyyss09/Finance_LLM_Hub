import os
from typing import Optional

import requests
from huggingface_hub import InferenceClient


# Remote inference (no local downloads). Change via FIN_ADVISOR_MODEL_ID.
# Default to Meta Llama 3 8B per user request.
MODEL_ID = os.environ.get("FIN_ADVISOR_MODEL_ID", "meta-llama/Meta-Llama-3-8B")
FALLBACK_MODEL_ID = os.environ.get("FIN_ADVISOR_FALLBACK_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

SYSTEM_PROMPT = (
    "You are a professional financial advisor. Provide concise, actionable guidance (3-7 sentences)."
    " Prefer diversified, low-cost portfolios; include risk and tax notes when relevant."
    " Do not repeat the question and never output role tags or fabricated dialogs."
)

_client: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    return _client


def _generate_text_generation(client: InferenceClient, prompt: str) -> str:
    return client.text_generation(
        prompt,
        max_new_tokens=int(os.environ.get("FIN_ADVISOR_MAX_NEW_TOKENS", 256)),
        temperature=float(os.environ.get("FIN_ADVISOR_TEMPERATURE", 0.7)),
        top_p=float(os.environ.get("FIN_ADVISOR_TOP_P", 0.9)),
        return_full_text=False,
        stop_sequences=[
            "[USER]", "[ASSISTANT]", "User:", "Assistant:",
            "USER:", "ASSISTANT:", "User reply:", "Advisor reply:",
            "User message:", "System:", "Question:", "Q:",
            "Human:", "### User", "### Assistant", "<|eot_id|>"
        ]
    )


def _generate_conversational(client: InferenceClient, prompt: str) -> str:
    # Try standard text-generation as a convenience wrapper
    return client.text_generation(
        prompt,
        max_new_tokens=int(os.environ.get("FIN_ADVISOR_MAX_NEW_TOKENS", 256)),
        temperature=float(os.environ.get("FIN_ADVISOR_TEMPERATURE", 0.7)),
        top_p=float(os.environ.get("FIN_ADVISOR_TOP_P", 0.9)),
        return_full_text=False,
    )


def _generate_conversational_http(model_id: str, token: Optional[str], prompt: str) -> str:
    # Directly call the HF Inference API with the conversational task payload
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": {
            "past_user_inputs": [],
            "generated_responses": [],
            "text": prompt,
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "generated_text" in data:
        return data.get("generated_text", "")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("generated_text", "")
    return str(data)


def _build_chat_prompt(user_query: str) -> str:
    # Minimal prompt to reduce role-tag echoes
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {user_query.strip()}\n"
        f"Answer:"
    )


def _postprocess(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    # Remove common headers/prefixes at the start
    starting_prefixes = [
        "Assistant:", "advisor:", "Advisor:", "Advisor reply:",
        "[ASSISTANT]", "assistant:", "ASSISTANT:", "Answer:", "A:"
    ]
    for prefix in starting_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
            break

    # Truncate at the earliest occurrence of any user-like marker AFTER the beginning
    stop_markers = [
        "[USER]", "User:", "USER:", "User reply:", "User message:",
        "System:", "Question:", "Q:", "Human:", "### User", "<|eot_id|>"
    ]
    cut_positions = []
    for m in stop_markers:
        idx = cleaned.find(m)
        if idx > 0:
            cut_positions.append(idx)
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)].rstrip()

    # Remove any lingering role tags
    for m in [
        "[USER]", "[ASSISTANT]", "User:", "Assistant:", "USER:", "ASSISTANT:",
        "User reply:", "Advisor reply:", "User message:", "System:",
        "Question:", "Q:", "Answer:", "A:", "### User", "### Assistant"
    ]:
        cleaned = cleaned.replace(m, "")

    return cleaned.strip()


def ask_financial_advisor(query: str) -> str:
    try:
        if not isinstance(query, str) or not query.strip():
            return "Please provide a valid question."

        prompt_text = _build_chat_prompt(query)
        client = _get_client()

        # Try standard text-generation first
        try:
            text = _generate_text_generation(client, prompt_text)
            cleaned = _postprocess((text or "").strip())
            if cleaned:
                return cleaned
        except Exception as primary_err:
            err_msg = str(primary_err).lower()
            # If backend indicates conversational-only, try the HTTP conversational task
            if ("conversational" in err_msg or "not supported for task text-generation" in err_msg
                or "supported task: conversational" in err_msg):
                try:
                    text = _generate_conversational_http(MODEL_ID, HF_TOKEN, prompt_text)
                    cleaned = _postprocess((text or "").strip())
                    if cleaned:
                        return cleaned
                except Exception:
                    pass
            # Otherwise fall through to fallback model below

        # Fallback to a compatible model for text-generation
        fb_client = InferenceClient(model=FALLBACK_MODEL_ID, token=HF_TOKEN)
        try:
            text = _generate_text_generation(fb_client, prompt_text)
        except Exception:
            # Try conversational on fallback via HTTP too
            text = _generate_conversational_http(FALLBACK_MODEL_ID, HF_TOKEN, prompt_text)
        return _postprocess((text or "").strip()) or "I couldn’t generate advice just now. Please try again."
    except Exception as e:
        return f"Advisor unavailable: {e}. Verify your Hugging Face token and model access."


