"""Global hotkey management."""

import os
import threading
import keyboard

from dictation import state

_hotkey_ids = []
_record_start_lock = threading.Lock()


def on_record_hotkey():
    with state.recording_lock:
        currently_recording = state.is_recording
    if currently_recording:
        state.stop_recording_event.set()
    else:
        if not _record_start_lock.acquire(blocking=False):
            return
        try:
            from dictation.audio import process_recording
            threading.Thread(target=process_recording, daemon=True).start()
        finally:
            _record_start_lock.release()


def _on_quit_hotkey():
    if state.app:
        state.app.root.after(0, state.app._shutdown)
    else:
        os._exit(0)


def unregister_hotkeys():
    """Unregister all hotkeys. Safe to call from any thread."""
    for hid in _hotkey_ids:
        try:
            keyboard.remove_hotkey(hid)
        except (KeyError, ValueError):
            pass
    _hotkey_ids.clear()


def register_hotkeys():
    unregister_hotkeys()

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
            _on_quit_hotkey,
            suppress=True,
            trigger_on_release=False,
        )
        _hotkey_ids.append(hid)
    except Exception as e:
        print(f"[hotkey] Failed to register quit hotkey: {e}")
