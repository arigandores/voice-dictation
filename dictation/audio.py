"""Audio recording and processing pipeline."""

import math
import time
import numpy as np
import miniaudio
import keyboard
import pyperclip

from dictation.config import SAMPLE_RATE, CHANNELS
from dictation import state
from dictation.vad import filter_silence
from dictation.llm import llm_correct

MAX_RECORDING_SECONDS = 300  # 5 minutes


def _find_capture_device_id(name):
    """Find miniaudio device_id by name, or None for default."""
    if not name:
        return None
    try:
        devs = miniaudio.Devices()
        for d in devs.get_captures():
            if d["name"] == name:
                return d["id"]
    except Exception:
        pass
    print(f"[rec] Device '{name}' not found, using default")
    return None


def record_audio():
    buf = []

    state.stop_recording_event.clear()
    state.waveform_levels.clear()

    device_name = state.config.get("input_device")
    device_id = _find_capture_device_id(device_name)

    def capture_gen():
        while True:
            data = yield
            raw = bytes(data)
            buf.append(raw)
            # Push RMS amplitude for waveform visualization
            samples = np.frombuffer(raw, dtype=np.float32)
            if len(samples) > 0:
                rms = math.sqrt(np.mean(samples ** 2))
                state.waveform_levels.append(rms)

    cap = miniaudio.CaptureDevice(
        input_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=CHANNELS,
        sample_rate=SAMPLE_RATE,
        device_id=device_id,
    )
    gen = capture_gen()
    next(gen)  # prime the generator
    cap.start(gen)

    mic_name = device_name or "default"
    print(f"[rec] Recording from '{mic_name}'... (press hotkey again to stop)")
    with state.recording_lock:
        state.is_recording = True
    if state.app:
        from dictation.ui.overlay import OverlayApp
        state.app.set_status(OverlayApp.STATUS_RECORDING)

    rec_start = time.time()
    try:
        while not state.stop_recording_event.wait(timeout=0.05):
            if time.time() - rec_start > MAX_RECORDING_SECONDS:
                print(f"[rec] Max duration ({MAX_RECORDING_SECONDS}s) reached, stopping")
                break
    finally:
        with state.recording_lock:
            state.is_recording = False
        cap.stop()
        cap.close()

    if not buf:
        return np.array([], dtype=np.float32)

    audio = np.frombuffer(b"".join(buf), dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE
    print(f"[rec] Recorded {duration:.1f}s")
    return audio


def type_text(text):
    pyperclip.copy(text)
    time.sleep(0.15)
    keyboard.send("ctrl+v")


def process_recording():
    if not state.processing_lock.acquire(blocking=False):
        return
    try:
        from dictation import asr
        from dictation.ui.overlay import OverlayApp

        if not asr.is_loaded():
            print("[skip] ASR model not loaded yet")
            if state.app:
                state.app.set_status(OverlayApp.STATUS_IDLE)
            return

        try:
            audio = record_audio()
        except Exception as e:
            print(f"[error] Microphone error: {e}")
            if state.app:
                state.app.set_status(OverlayApp.STATUS_IDLE)
            return
        if len(audio) < SAMPLE_RATE * 0.3:
            print("[skip] Too short")
            if state.app:
                state.app.set_status(OverlayApp.STATUS_IDLE)
            return

        # VAD filtering for Canary backend
        if asr.get_backend() == "canary":
            audio = filter_silence(audio)
            if len(audio) < SAMPLE_RATE * 0.3:
                print("[skip] No speech detected")
                if state.app:
                    state.app.set_status(OverlayApp.STATUS_IDLE)
                return

        # Step 1: Transcription
        if state.app:
            state.app.set_status(OverlayApp.STATUS_TRANSCRIBING)
        t = time.time()
        backend_tag = asr.get_backend() or "whisper"
        raw_text = asr.transcribe(audio)
        asr_time = time.time() - t
        print(f"[{backend_tag}] ({asr_time:.2f}s) {raw_text}")

        if not raw_text.strip():
            print("[skip] Empty transcription")
            if state.app:
                state.app.set_status(OverlayApp.STATUS_IDLE)
            return

        # Step 2: LLM correction (if enabled)
        if state.config.get("llm_enabled", True):
            if state.app:
                state.app.set_status(OverlayApp.STATUS_CORRECTING)
            t = time.time()
            corrected = llm_correct(raw_text)
            llm_time = time.time() - t
            print(f"[llm] ({llm_time:.2f}s) {corrected}")
        else:
            corrected = raw_text
            llm_time = 0

        # Step 3: Paste + History
        type_text(corrected)
        total = asr_time + llm_time
        state.add_to_history(raw_text, corrected, backend_tag, total)
        print(f"[done] Total: {total:.2f}s")
        if state.app:
            state.app.set_status(OverlayApp.STATUS_DONE, f"({total:.1f}s)")
    finally:
        state.processing_lock.release()
