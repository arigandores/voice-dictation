# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows voice dictation tool that records audio via hotkey, transcribes with GPU-accelerated ASR (Whisper or NVIDIA Canary), optionally corrects via local LLM (Ollama), and pastes the result into the active window via Ctrl+V.

## Running

```powershell
# Activate the virtual environment first
~\whisper-env\Scripts\Activate.ps1
python main.py
```

Requires NVIDIA GPU with CUDA, Ollama running locally (port 11434), and Python 3.12.

## Architecture

The project is split into a `dictation/` package with the following modules:

```
main.py                     # Entry point: console banner, starts OverlayApp + background init
dictation/
    __init__.py
    config.py               # Constants (paths, defaults, language options, URLs, sample rate),
                            #   load_config/save_config, get_input_devices, fetch_ollama_models
    state.py                # Shared mutable state: config dict, app ref, locks, recording flags,
                            #   transcription history (add_to_history)
    vad.py                  # Silero VAD: load_vad, filter_silence
    asr.py                  # ASR backends: load/transcribe for Whisper and Canary,
                            #   load_asr_model, reload_asr_model, transcribe dispatcher
    llm.py                  # LLM correction via Ollama: llm_correct, warmup_llm
    audio.py                # Audio capture (miniaudio), processing pipeline:
                            #   record_audio, type_text, process_recording
    hotkeys.py              # Global hotkey registration via keyboard library
    ui/
        __init__.py
        overlay.py          # OverlayApp: always-on-top tkinter status widget
        history.py          # HistoryWindow: scrollable transcription history
        settings.py         # SettingsWindow: settings dialog for all config
```

**Threading model:** tkinter mainloop on main thread. Background thread loads ASR model + warms LLM. Recording/processing happens on daemon threads triggered by hotkey. `state.processing_lock` prevents concurrent transcriptions. `state.recording_lock` guards `state.is_recording`.

**Text insertion:** copies to clipboard via `pyperclip`, then simulates Ctrl+V via `keyboard`.

**State management:** All shared mutable state (config dict, app reference, locks, history) lives in `dictation/state.py`. Modules import `state` and access attributes directly.

## Key Dependencies

- `faster-whisper`: Whisper ASR (GPU, required)
- `nemo_toolkit[asr]` + `silero-vad`: Canary ASR (optional)
- `miniaudio`: audio capture
- `keyboard`: global hotkeys (requires admin on Linux, works on Windows)
- `httpx`: Ollama API calls
- `tkinter`: UI (bundled with Python)

## Conventions

- UI text and labels are in Russian
- Config persists as JSON with atomic write (write to .tmp, then `os.replace`)
- All print output uses `[tag]` prefixes: `[whisper]`, `[canary]`, `[llm]`, `[rec]`, `[vad]`, `[config]`, `[error]`
- ASR models are hot-swappable at runtime — changing settings reloads without restart
- Lazy imports for heavy dependencies (torch, faster_whisper, nemo) to keep startup fast
