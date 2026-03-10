"""
Voice Dictation Tool
====================
Press Ctrl+Shift+Space to start/stop recording.
Transcribes with Whisper or Canary, corrects with LLM, pastes result.
Press Ctrl+Shift+Q to quit.

Backends:
  - faster-whisper (GPU) -- fast startup, auto language detection
  - NVIDIA Canary-1B-v2 (GPU) -- lower VRAM, 25 European languages
LLM: Ollama (configurable model) for post-correction.
"""

import sys
import os
import threading

# Fix Windows console encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dictation import state
from dictation.config import LANGUAGE_OPTIONS
from dictation.asr import load_asr_model
from dictation.llm import warmup_llm
from dictation.hotkeys import register_hotkeys
from dictation.ui.overlay import OverlayApp


def background_init():
    """Load models in background thread, then register hotkeys."""
    asr_ok = True
    try:
        load_asr_model()
    except Exception as e:
        asr_ok = False
        print(f"[error] Failed to load ASR model: {e}")
        print("[error] Check that the model exists and CUDA is available.")

    if state.config.get("llm_enabled", True):
        warmup_llm()

    register_hotkeys()

    hotkey = state.config["hotkey_record"].upper()
    backend = state.config.get("asr_backend", "whisper").capitalize()
    llm = state.config.get("llm_model", "qwen3.5:9b")

    if asr_ok:
        print(f"\nReady! Press {hotkey} to start/stop dictation.")
        print(f"  ASR: {backend}  |  LLM: {llm}\n")
        if state.app:
            state.app.set_status(OverlayApp.STATUS_IDLE)
    else:
        print(f"\n[error] ASR model failed to load. Recording will not work.")
        print(f"[error] Try changing ASR settings via right-click -> Settings.\n")
        if state.app:
            state.app.set_status(OverlayApp.STATUS_LOADING, "(ошибка ASR)")


def main():
    backend = state.config.get("asr_backend", "whisper").capitalize()
    llm = state.config.get("llm_model", "qwen3.5:9b")

    print("=" * 50)
    print(f"  Voice Dictation ({backend} + {llm})")
    print("=" * 50)
    print(f"  Record:  press {state.config['hotkey_record'].upper()}")
    print(f"  Quit:    {state.config['hotkey_quit'].upper()}")
    print(f"  Settings: right-click the overlay")
    print("=" * 50)

    state.app = OverlayApp()
    threading.Thread(target=background_init, daemon=True).start()
    state.app.run()


if __name__ == "__main__":
    main()
