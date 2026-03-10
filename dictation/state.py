"""Shared mutable state: config, locks, history, app reference."""

import collections
import datetime
import queue
import threading

from dictation.config import load_config

config = load_config()
audio_queue = queue.Queue()
is_recording = False
recording_lock = threading.Lock()
processing_lock = threading.Lock()
stop_recording_event = threading.Event()
app = None  # OverlayApp instance

# Real-time audio amplitude for waveform visualization
WAVEFORM_MAXLEN = 80
waveform_levels = collections.deque(maxlen=WAVEFORM_MAXLEN)

# Transcription history
_history = []
_history_lock = threading.Lock()
MAX_HISTORY = 100


def add_to_history(raw, corrected, backend, duration):
    entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "raw": raw,
        "corrected": corrected,
        "backend": backend,
        "duration": round(duration, 2),
    }
    with _history_lock:
        _history.append(entry)
        if len(_history) > MAX_HISTORY:
            _history.pop(0)
