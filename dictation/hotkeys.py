"""Global hotkey management."""

import os
import threading
import keyboard

from dictation import state

_hotkey_ids = []


def on_record_hotkey():
    with state.recording_lock:
        currently_recording = state.is_recording
    if currently_recording:
        state.stop_recording_event.set()
    else:
        from dictation.audio import process_recording
        threading.Thread(target=process_recording, daemon=True).start()


def register_hotkeys():
    global _hotkey_ids
    for hid in _hotkey_ids:
        try:
            keyboard.remove_hotkey(hid)
        except (KeyError, ValueError):
            pass
    _hotkey_ids.clear()

    try:
        hid = keyboard.add_hotkey(
            state.config["hotkey_record"],
            on_record_hotkey,
            suppress=True,
            trigger_on_release=False,
        )
        _hotkey_ids.append(hid)
    except Exception as e:
        print(f"[hotkey] Failed to register record hotkey: {e}")

    try:
        hid = keyboard.add_hotkey(
            state.config["hotkey_quit"],
            lambda: state.app._shutdown() if state.app else os._exit(0),
            suppress=True,
            trigger_on_release=False,
        )
        _hotkey_ids.append(hid)
    except Exception as e:
        print(f"[hotkey] Failed to register quit hotkey: {e}")
