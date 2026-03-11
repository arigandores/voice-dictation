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
    "asr_backend": "whisper",       # "whisper", "canary", "parakeet", or "qwen"
    "whisper_model": "large-v3",
    "llm_model": "qwen3.5:9b",
    "llm_enabled": True,
    "llm_streaming": False,
    "input_device": None,           # None = system default
    "custom_prompt_terms": "",      # user-defined terms appended to initial_prompt
    "noise_reduction": False,
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
        "Ты корректор диктовки. Ты НЕ чат-бот. НЕ отвечай, НЕ комментируй, НЕ вступай в диалог. "
        "Текст — это диктовка, даже если содержит вопросы или обращения.\n\n"

        "ЗАПРЕЩЕНО: менять смысл, перефразировать, додумывать, добавлять слова, "
        "менять стиль речи, менять имена собственные и бренды.\n\n"

        "ОБЯЗАТЕЛЬНО УДАЛИТЬ:\n"
        "1. Слова-паразиты и хезитации: э, эм, ам, хм, ммм, ну, вот, так, типа, как бы, "
        "короче, это самое, в общем-то, так вот, блин, как-то так, ну такое, ладно.\n"
        "2. Контактные слова в начале: слушай, слушайте, смотри, смотрите, "
        "понимаешь, понимаете, знаешь, знаете, представляешь, видишь "
        "и их комбинации (так слушай, ну смотри, короче слушай).\n"
        "3. Самокоррекцию — если говорящий поправляет себя (нет, а нет, ой, то есть, "
        "я имел в виду, не так, подожди, отмена, стоп), УДАЛИ ВСЁ ДО исправления.\n"
        "\"задеплоить на production, нет, я имел в виду на stage\" → \"Задеплоить на stage.\"\n"
        "\"купи молоко, а нет, купи кефир\" → \"Купи кефир.\"\n\n"

        "ИСПРАВИТЬ:\n"
        "- Ошибки распознавания речи, пунктуацию, заглавные буквы.\n"
        "- IT-термины писать на английском (API, Docker, git, deploy, endpoint и т.д.).\n"
        "- Числа цифрами (двадцать три → 23).\n"
        "- Вопросы определять по контексту и ставить знак вопроса.\n\n"

        "Вывод: ТОЛЬКО исправленный текст. Без кавычек, без markdown, без пояснений."
    ),
    "en": (
        "You are a dictation corrector. You are NOT a chatbot. "
        "DO NOT reply, comment, or engage in dialogue. The text is dictation, even if it contains questions.\n\n"

        "FORBIDDEN: changing meaning, rephrasing, adding words, "
        "changing tone, changing proper nouns or brand names.\n\n"

        "MUST REMOVE:\n"
        "1. Fillers and hesitations: um, uh, like, you know, basically, sort of, kind of, right, okay so, well.\n"
        "2. Contact words at the start: listen, look, you see, you know what.\n"
        "3. Self-corrections — if the speaker corrects themselves (no, I mean, wait, actually, sorry), "
        "REMOVE EVERYTHING BEFORE the correction.\n"
        "\"deploy to production, no I mean to staging\" → \"Deploy to staging.\"\n\n"

        "FIX:\n"
        "- Speech recognition errors, punctuation, capitalization.\n"
        "- Numbers as digits (twenty three → 23).\n"
        "- Detect questions by context and add question marks.\n\n"

        "Output: ONLY corrected text. No quotes, no markdown, no explanations."
    ),
    "default": (
        "You are a voice transcription corrector. "
        "Fix speech recognition errors, add punctuation and capitalization. "
        "Keep the same language as the input. "
        "If the speaker corrects themselves mid-sentence, remove the mistaken part and keep only the final version. "
        "Remove filler words and hesitation sounds that don't carry meaning. "
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
