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
    print(f"[canary] Model loaded in {time.time()-t:.1f}s")
    return model


def transcribe_whisper(model, audio_data):
    lang = state.config.get("language", "auto")
    segments, info = model.transcribe(
        audio_data,
        language=None if lang == "auto" else lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(seg.text.strip() for seg in segments)
    if lang == "auto":
        print(f"[whisper] Detected language: {info.language} ({info.language_probability:.0%})")
    return text


def transcribe_canary(model, audio_data):
    import soundfile as sf
    lang = state.config.get("language", "auto")
    source_lang = "ru" if lang == "auto" else lang

    tmpfile = os.path.join(tempfile.gettempdir(), "dictation_canary_tmp.wav")
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
    return text.strip()


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


def transcribe(audio_data):
    """Transcribe using the currently loaded ASR backend."""
    if _asr_backend == "canary":
        return transcribe_canary(_asr_model, audio_data)
    else:
        return transcribe_whisper(_asr_model, audio_data)
