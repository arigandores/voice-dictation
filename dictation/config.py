"""Configuration constants, loading, and saving."""

import os
import json
import miniaudio
import httpx

CONFIG_PATH = os.path.join(os.path.expanduser("~"), "dictation_config.json")

DEFAULT_CONFIG = {
    "hotkey_record": "ctrl+shift+space",
    "hotkey_quit": "ctrl+shift+q",
    "language": "auto",
    "asr_backend": "whisper",       # "whisper" or "canary"
    "whisper_model": "large-v3",
    "llm_model": "qwen3.5:9b",
    "llm_enabled": True,
    "input_device": None,           # None = system default
}

WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

LANGUAGE_OPTIONS = {
    "auto": "Авто (определять автоматически)",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "uk": "Українська",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
}

WHISPER_MODELS = [
    "large-v3",
    "large-v3-turbo",
    "~/whisper-env/faster-whisper-ru-turbo",
]

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

SAMPLE_RATE = 16000
CHANNELS = 1

LLM_SYSTEM_TEMPLATES = {
    "ru": (
        "Ты корректор транскрипции голосового ввода. "
        "Исправляй ошибки распознавания речи, расставляй знаки препинания и заглавные буквы. "
        "IT-термины и англицизмы (API, Docker, git, pull request, deploy, endpoint и т.д.) пиши на английском. "
        "Отвечай ТОЛЬКО исправленным текстом на русском языке. Никаких пояснений."
    ),
    "en": (
        "You are a voice transcription corrector. "
        "Fix speech recognition errors, add punctuation and capitalization. "
        "Reply ONLY with the corrected text in English. No explanations."
    ),
    "default": (
        "You are a voice transcription corrector. "
        "Fix speech recognition errors, add punctuation and capitalization. "
        "Keep the same language as the input. "
        "Reply ONLY with the corrected text. No explanations."
    ),
}


def get_llm_system(lang="auto"):
    return LLM_SYSTEM_TEMPLATES.get(lang, LLM_SYSTEM_TEMPLATES["default"])


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(saved)
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, CONFIG_PATH)
    except OSError as e:
        print(f"[config] Failed to save config: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_input_devices():
    """Return list of device names for capture devices via miniaudio."""
    try:
        devs = miniaudio.Devices()
        return [d["name"] for d in devs.get_captures()]
    except Exception:
        return []


def fetch_ollama_models():
    """Fetch available models from Ollama API."""
    try:
        resp = httpx.get(OLLAMA_TAGS_URL, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
