"""LLM correction via Ollama."""

import httpx

from dictation.config import OLLAMA_URL, get_llm_system
from dictation import state


def llm_correct(text):
    if not text.strip():
        return text
    model = state.config.get("llm_model", "qwen3.5:9b")
    lang = state.config.get("language", "auto")
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": get_llm_system(lang)},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                },
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()
        return result if result else text
    except Exception as e:
        print(f"[llm] Error: {e}")
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
