"""Silero VAD for silence filtering (Canary backend only)."""

import time
import numpy as np

from dictation.config import SAMPLE_RATE

_vad_model = None


def load_vad():
    global _vad_model
    if _vad_model is not None:
        return
    print("[vad] Loading Silero VAD (ONNX)...")
    import torch
    from silero_vad import load_silero_vad
    t = time.time()
    _vad_model = load_silero_vad(onnx=True)
    print(f"[vad] Loaded in {time.time()-t:.1f}s")


def filter_silence(audio_np):
    """Remove silence using Silero VAD."""
    if _vad_model is None:
        return audio_np
    import torch
    from silero_vad import get_speech_timestamps, collect_chunks
    tensor = torch.from_numpy(audio_np).float()
    timestamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=SAMPLE_RATE)
    if not timestamps:
        print("[vad] No speech detected")
        return np.array([], dtype=np.float32)
    filtered = collect_chunks(timestamps, tensor)
    original_dur = len(audio_np) / SAMPLE_RATE
    filtered_dur = len(filtered) / SAMPLE_RATE
    print(f"[vad] {original_dur:.1f}s -> {filtered_dur:.1f}s (removed {original_dur - filtered_dur:.1f}s silence)")
    return filtered.numpy()
