"""ASR backends: Whisper and NVIDIA Canary."""

import os
import time
import tempfile

from dictation.config import WHISPER_DEVICE, WHISPER_COMPUTE, SAMPLE_RATE
from dictation import state
from dictation.vad import load_vad, filter_silence

_asr_model = None
_asr_backend = None  # tracks which backend is loaded


def load_whisper_model():
    model_path = state.config.get("whisper_model", "large-v3")
    if model_path.startswith("~"):
        model_path = os.path.expanduser(model_path)
    print(f"[whisper] Loading model: {model_path}")
    if state.app:
        from dictation.ui.overlay import OverlayApp
        state.app.set_status(OverlayApp.STATUS_LOADING, "(Whisper)")
    from faster_whisper import WhisperModel
    t = time.time()
    model = WhisperModel(model_path, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(f"[whisper] Model loaded in {time.time()-t:.1f}s")
    return model


def load_canary_model():
    print("[canary] Loading NVIDIA Canary-1B-v2 (this may take a while on first run)...")
    if state.app:
        from dictation.ui.overlay import OverlayApp
        state.app.set_status(OverlayApp.STATUS_LOADING, "(Canary)")
    load_vad()
    import nemo.collections.asr as nemo_asr
    t = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-v2")
    model = model.to("cuda")
    model.eval()

    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 5
    decode_cfg.beam.len_pen = 1.0
    model.change_decoding_strategy(decode_cfg)
    print(f"[canary] Beam search configured (beam_size=5, len_pen=1.0)")

    print(f"[canary] Model loaded in {time.time()-t:.1f}s")
    return model


_HOTWORDS = (
    "API Docker git pull request deploy endpoint Kubernetes "
    "CI/CD backend frontend merge commit pipeline container debug "
    "microservice REST JSON webhook release server database "
    "refactoring framework library repository branch staging production"
)


def transcribe_whisper(model, audio_data):
    lang = state.config.get("language", "auto")

    initial_prompt = None
    if lang in ("ru", "auto"):
        initial_prompt = (
            "Мы обсуждали архитектуру backend-сервиса. Нужно настроить CI/CD pipeline, "
            "задеплоить Docker-контейнер на Kubernetes. Я создал pull request, "
            "прошёл code review и замержил в main. Проверил REST API endpoint, "
            "отправил JSON через webhook. Всё работает стабильно после последнего release."
        )

    # Append user-defined custom terms
    custom = state.config.get("custom_prompt_terms", "").strip()
    if custom:
        terms_line = f" Термины: {custom}."
        if initial_prompt:
            initial_prompt += terms_line
        else:
            initial_prompt = terms_line

    segments, info = model.transcribe(
        audio_data,
        language=None if lang == "auto" else lang,
        beam_size=5,
        patience=1.5,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        initial_prompt=initial_prompt,
        hotwords=_HOTWORDS,
        hallucination_silence_threshold=2.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    # Collect segments and compute average confidence
    texts = []
    log_probs = []
    for seg in segments:
        texts.append(seg.text.strip())
        if seg.avg_logprob is not None:
            log_probs.append(seg.avg_logprob)

    text = " ".join(texts)
    avg_logprob = sum(log_probs) / len(log_probs) if log_probs else None

    if lang == "auto":
        print(f"[whisper] Detected language: {info.language} ({info.language_probability:.0%})")
    if avg_logprob is not None:
        print(f"[whisper] Confidence: avg_logprob={avg_logprob:.3f}")

    return text, avg_logprob


def transcribe_canary(model, audio_data):
    import soundfile as sf
    lang = state.config.get("language", "auto")
    source_lang = "ru" if lang == "auto" else lang

    fd, tmpfile = tempfile.mkstemp(suffix=".wav", prefix="dictation_canary_")
    os.close(fd)
    sf.write(tmpfile, audio_data, SAMPLE_RATE)

    try:
        result = model.transcribe([tmpfile], source_lang=source_lang, target_lang=source_lang, batch_size=1)
    finally:
        try:
            os.remove(tmpfile)
        except OSError:
            pass

    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        text = item.text if hasattr(item, "text") else str(item)
    else:
        text = str(result)

    if lang == "auto":
        print(f"[canary] (source_lang={source_lang})")
    return text.strip(), None  # Canary doesn't expose per-segment logprob


def load_asr_model():
    """Load the ASR model based on current config."""
    global _asr_model, _asr_backend
    backend = state.config.get("asr_backend", "whisper")
    if backend == "canary":
        _asr_model = load_canary_model()
    else:
        _asr_model = load_whisper_model()
    _asr_backend = backend


def reload_asr_model():
    """Reload ASR model (called when settings change)."""
    global _asr_model, _asr_backend

    with state.processing_lock:
        print("[config] Reloading ASR model...")

        old_model = _asr_model
        _asr_model = None
        _asr_backend = None
        del old_model

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        load_asr_model()

    if state.app:
        from dictation.ui.overlay import OverlayApp
        state.app.set_status(OverlayApp.STATUS_IDLE)
    print("[config] ASR model reloaded")


def is_loaded():
    """Return True if an ASR model is currently loaded."""
    return _asr_model is not None


def get_backend():
    """Return the name of the currently loaded ASR backend."""
    return _asr_backend


def transcribe(audio_data):
    """Transcribe using the currently loaded ASR backend."""
    if _asr_backend == "canary":
        return transcribe_canary(_asr_model, audio_data)
    else:
        return transcribe_whisper(_asr_model, audio_data)


def transcribe_quick(audio_data):
    """Fast transcription for live preview (Whisper only, greedy decoding)."""
    if _asr_model is None or _asr_backend != "whisper":
        return ""
    try:
        lang = state.config.get("language", "auto")
        segments, _ = _asr_model.transcribe(
            audio_data,
            language=None if lang == "auto" else lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        return " ".join(seg.text.strip() for seg in segments)
    except Exception:
        return ""
