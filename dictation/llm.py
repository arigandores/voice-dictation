"""LLM correction via Ollama."""

import httpx

from dictation.config import OLLAMA_URL, get_llm_system
from dictation import state


def _build_request(text):
    """Build the Ollama API request body."""
    model = state.config.get("llm_model", "qwen3.5:9b")
    lang = state.config.get("language", "auto")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": get_llm_system(lang)},
            {"role": "user", "content": text},
        ],
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }


def llm_correct(text):
    if not text.strip():
        return text
    try:
        body = _build_request(text)
        body["stream"] = False
        resp = httpx.post(OLLAMA_URL, json=body, timeout=30.0)
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()
        return result if result else text
    except Exception as e:
        print(f"[llm] Error: {e}")
        return text


def llm_correct_streaming(text, on_token=None):
    """Stream LLM correction, calling on_token(partial_text) as tokens arrive."""
    if not text.strip():
        return text
    try:
        body = _build_request(text)
        body["stream"] = True
        collected = []
        with httpx.stream("POST", OLLAMA_URL, json=body, timeout=30.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                token = chunk.get("message", {}).get("content", "")
                if token:
                    collected.append(token)
                    if on_token:
                        on_token("".join(collected))
        result = "".join(collected).strip()
        return result if result else text
    except Exception as e:
        print(f"[llm] Streaming error: {e}")
        return text


def warmup_llm(model_name=None):
    """Warm up an Ollama model."""
    model = model_name or state.config.get("llm_model", "qwen3.5:9b")
    print(f"[llm] Warming up {model}...")
    try:
        httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1},
            },
            timeout=60.0,
        )
        print(f"[llm] {model} ready")
    except Exception as e:
        print(f"[llm] Warning: Ollama not responding: {e}")
        print("[llm] Make sure Ollama is running!")
