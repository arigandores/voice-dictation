"""
Voice Dictation Tool
====================
Press Ctrl+Shift+Space to start/stop recording.
Transcribes with Whisper or Canary, corrects with LLM, pastes result.
Press Ctrl+Shift+Q to quit.

Backends:
  - faster-whisper (GPU) — fast startup, auto language detection
  - NVIDIA Canary-1B-v2 (GPU) — lower VRAM, 25 European languages
LLM: Ollama (configurable model) for post-correction.
"""

import sys
import os
import time
import json
import datetime
import threading
import queue
import tempfile
import numpy as np
import miniaudio
import keyboard
import pyperclip
import httpx
import tkinter as tk
from tkinter import ttk

# Fix Windows console encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────── Config ────────────────────
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "dictation_config.json")

DEFAULT_CONFIG = {
    "hotkey_record": "ctrl+shift+space",
    "hotkey_quit": "ctrl+shift+q",
    "language": "auto",
    "asr_backend": "whisper",       # "whisper" or "canary"
    "whisper_model": "large-v3",
    "llm_model": "qwen3.5:9b",
    "llm_enabled": True,
    "input_device": None,           # None = system default
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
        "Ты корректор транскрипции голосового ввода. "
        "Исправляй ошибки распознавания речи, расставляй знаки препинания и заглавные буквы. "
        "IT-термины и англицизмы (API, Docker, git, pull request, deploy, endpoint и т.д.) пиши на английском. "
        "Отвечай ТОЛЬКО исправленным текстом на русском языке. Никаких пояснений."
    ),
    "en": (
        "You are a voice transcription corrector. "
        "Fix speech recognition errors, add punctuation and capitalization. "
        "Reply ONLY with the corrected text in English. No explanations."
    ),
    "default": (
        "You are a voice transcription corrector. "
        "Fix speech recognition errors, add punctuation and capitalization. "
        "Keep the same language as the input. "
        "Reply ONLY with the corrected text. No explanations."
    ),
}


def get_llm_system():
    lang = config.get("language", "auto")
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
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CONFIG_PATH)


def get_input_devices():
    """Return list of (name, name) for capture devices via miniaudio."""
    try:
        devs = miniaudio.Devices()
        return [(d["name"], d["name"]) for d in devs.get_captures()]
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


# ──────────────────── Globals ────────────────────
config = load_config()
audio_queue = queue.Queue()
is_recording = False
recording_lock = threading.Lock()
processing_lock = threading.Lock()
stop_recording_event = threading.Event()
app = None  # OverlayApp instance

# ASR model (Whisper or Canary)
_asr_model = None
_asr_backend = None  # tracks which backend is loaded

# Canary-specific: Silero VAD model
_vad_model = None

# Transcription history: list of dicts {time, raw, corrected, backend, duration}
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


# ──────────────────── Overlay UI ────────────────────
class OverlayApp:
    """Small always-on-top status window."""

    STATUS_IDLE = "idle"
    STATUS_RECORDING = "recording"
    STATUS_TRANSCRIBING = "transcribing"
    STATUS_CORRECTING = "correcting"
    STATUS_DONE = "done"
    STATUS_LOADING = "loading"

    COLORS = {
        "idle": "#2d2d2d",
        "recording": "#c0392b",
        "transcribing": "#2980b9",
        "correcting": "#8e44ad",
        "done": "#27ae60",
        "loading": "#e67e22",
    }

    LABELS = {
        "idle": "⏳ Готов к записи",
        "recording": "🔴 Запись...",
        "transcribing": "📝 Транскрипция",
        "correcting": "✨ Коррекция",
        "done": "✅ Готово!",
        "loading": "⏳ Загрузка модели",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dictation")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)

        self.width = 220
        self.height = 58
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - self.width - 20
        y = 20
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Draggable
        self._drag_data = {"x": 0, "y": 0}
        self.root.bind("<Button-1>", self._on_drag_start)
        self.root.bind("<B1-Motion>", self._on_drag_motion)

        # Right-click menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="📋 История", command=self._open_history)
        self.menu.add_command(label="⚙ Настройки", command=self._open_settings)
        self.menu.add_separator()
        self.menu.add_command(label="✕ Выход", command=self._quit)
        self.root.bind("<Button-3>", self._show_menu)

        # Main frame
        self.frame = tk.Frame(self.root, bg=self.COLORS["idle"], padx=10, pady=8)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Status label
        self.label = tk.Label(
            self.frame,
            text=self.LABELS["idle"],
            font=("Segoe UI", 11),
            fg="white",
            bg=self.COLORS["idle"],
            anchor="w",
        )
        self.label.pack(fill=tk.X)

        # Info label (hotkey + backend)
        info_text = self._make_info_text()
        self.hotkey_label = tk.Label(
            self.frame,
            text=info_text,
            font=("Segoe UI", 8),
            fg="#aaaaaa",
            bg=self.COLORS["idle"],
            anchor="w",
        )
        self.hotkey_label.pack(fill=tk.X)

        self._anim_dots = 0
        self._anim_job = None
        self._current_status = self.STATUS_IDLE
        self._done_hide_job = None

    def _make_info_text(self):
        hotkey = config["hotkey_record"].upper()
        backend = config.get("asr_backend", "whisper").capitalize()
        if config.get("llm_enabled", True):
            llm = config.get("llm_model", "qwen3.5:9b")
        else:
            llm = "LLM выкл"
        return f"{hotkey}  |  {backend}  |  {llm}"

    def set_status(self, status, extra=""):
        self.root.after(0, self._update_status, status, extra)

    def _update_status(self, status, extra=""):
        self._current_status = status
        color = self.COLORS.get(status, self.COLORS["idle"])
        text = self.LABELS.get(status, status)
        if extra:
            text += f" {extra}"

        self.frame.config(bg=color)
        self.label.config(text=text, bg=color)
        self.hotkey_label.config(bg=color)

        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        if self._done_hide_job:
            self.root.after_cancel(self._done_hide_job)
            self._done_hide_job = None

        if status in (self.STATUS_RECORDING, self.STATUS_TRANSCRIBING, self.STATUS_CORRECTING, self.STATUS_LOADING):
            self._anim_dots = 0
            self._animate(status)

        if status == self.STATUS_DONE:
            self._done_hide_job = self.root.after(2000, lambda: self._update_status(self.STATUS_IDLE))

        self.root.deiconify()
        self.root.attributes("-topmost", True)

    def _animate(self, status):
        if self._current_status != status:
            return
        self._anim_dots = (self._anim_dots + 1) % 4
        dots = "." * self._anim_dots
        base = self.LABELS.get(status, "")
        self.label.config(text=base + dots)
        self._anim_job = self.root.after(400, self._animate, status)

    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag_motion(self, event):
        x = self.root.winfo_x() + event.x - self._drag_data["x"]
        y = self.root.winfo_y() + event.y - self._drag_data["y"]
        self.root.geometry(f"+{x}+{y}")

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def _open_history(self):
        HistoryWindow(self.root)

    def _open_settings(self):
        SettingsWindow(self.root)

    def _quit(self):
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown: unregister hotkeys, destroy tkinter, then exit."""
        try:
            for hid in _hotkey_ids:
                try:
                    keyboard.remove_hotkey(hid)
                except (KeyError, ValueError):
                    pass
            _hotkey_ids.clear()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    def update_info_display(self):
        self.hotkey_label.config(text=self._make_info_text())

    def run(self):
        self.root.mainloop()


class HistoryWindow:
    """Scrollable transcription history. Click to copy."""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("История транскрипций")
        self.win.geometry("600x500")
        self.win.resizable(True, True)
        self.win.attributes("-topmost", True)
        self.win.grab_set()

        # Top bar
        top = tk.Frame(self.win, padx=10, pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="История транскрипций", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        tk.Button(top, text="Очистить", font=("Segoe UI", 9), command=self._clear).pack(side=tk.RIGHT)

        # Hint
        tk.Label(self.win, text="Нажмите на запись, чтобы скопировать в буфер обмена",
                 font=("Segoe UI", 8), fg="#888", padx=10).pack(anchor="w")

        # Scrollable list
        container = tk.Frame(self.win)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.win.bind("<Destroy>", self._on_destroy)

        self._populate()

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_destroy(self, event):
        if event.widget == self.win:
            self.canvas.unbind_all("<MouseWheel>")

    def _populate(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        with _history_lock:
            entries = list(reversed(_history))

        if not entries:
            tk.Label(self.scroll_frame, text="Пока нет транскрипций",
                     font=("Segoe UI", 11), fg="#999", pady=40).pack()
            return

        for i, entry in enumerate(entries):
            card = tk.Frame(self.scroll_frame, bg="#f5f5f5" if i % 2 == 0 else "#ffffff",
                            padx=12, pady=8, cursor="hand2")
            card.pack(fill=tk.X, pady=1)

            # Header: time + backend + duration
            header = tk.Frame(card, bg=card["bg"])
            header.pack(fill=tk.X)
            tk.Label(header, text=entry["time"], font=("Segoe UI", 9, "bold"),
                     fg="#555", bg=card["bg"]).pack(side=tk.LEFT)
            tk.Label(header, text=f"  {entry['backend']}  •  {entry['duration']}s",
                     font=("Segoe UI", 8), fg="#999", bg=card["bg"]).pack(side=tk.LEFT)

            # Corrected text (main)
            text = entry["corrected"]
            lbl = tk.Label(card, text=text, font=("Segoe UI", 10), bg=card["bg"],
                           anchor="w", wraplength=550, justify=tk.LEFT)
            lbl.pack(fill=tk.X, pady=(3, 0))

            # Show raw if different
            if entry["raw"] != entry["corrected"]:
                raw_lbl = tk.Label(card, text=f"raw: {entry['raw']}", font=("Segoe UI", 8),
                                   fg="#aaa", bg=card["bg"], anchor="w", wraplength=550, justify=tk.LEFT)
                raw_lbl.pack(fill=tk.X)
                raw_lbl.bind("<Button-1>", lambda e, t=text: self._copy(t))

            # Bind click on all card widgets
            for w in (card, header, lbl):
                w.bind("<Button-1>", lambda e, t=text: self._copy(t))
            for child in header.winfo_children():
                child.bind("<Button-1>", lambda e, t=text: self._copy(t))

    def _copy(self, text):
        pyperclip.copy(text)
        try:
            orig_title = self.win.title()
            self.win.title("✓ Скопировано!")
            self.win.after(1000, lambda: self.win.title(orig_title))
        except tk.TclError:
            pass

    def _clear(self):
        with _history_lock:
            _history.clear()
        self._populate()


class SettingsWindow:
    """Settings dialog for all configuration."""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Настройки диктовки")
        self.win.geometry("460x550")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self.win.grab_set()

        frame = tk.Frame(self.win, padx=20, pady=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(frame, text="Настройки", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(frame, text="", font=("Segoe UI", 2)).pack()

        # ── ASR Backend ──
        tk.Label(frame, text="Модель транскрипции (ASR)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5, 5))

        self.asr_row = tk.Frame(frame)
        self.asr_row.pack(fill=tk.X, pady=(0, 3))

        self.asr_var = tk.StringVar(value=config.get("asr_backend", "whisper"))
        tk.Label(self.asr_row, text="Движок:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        asr_combo = ttk.Combobox(self.asr_row, textvariable=self.asr_var, values=["whisper", "canary"],
                                 state="readonly", font=("Segoe UI", 10), width=15)
        asr_combo.pack(side=tk.LEFT)
        self.asr_hint = tk.Label(self.asr_row, text="", font=("Segoe UI", 8), fg="#888")
        self.asr_hint.pack(side=tk.LEFT, padx=(10, 0))
        self.asr_var.trace_add("write", self._on_asr_change)
        self._on_asr_change()

        # Whisper model path (shown only when whisper selected)
        self.whisper_row = tk.Frame(frame)
        tk.Label(self.whisper_row, text="Whisper модель:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.whisper_var = tk.StringVar(value=config.get("whisper_model", "large-v3"))
        self.whisper_combo = ttk.Combobox(self.whisper_row, textvariable=self.whisper_var,
                                          values=WHISPER_MODELS, font=("Segoe UI", 10), width=35)
        self.whisper_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Show/hide based on initial backend
        self._toggle_model_row()

        # ── LLM Model ──
        llm_header = tk.Frame(frame)
        llm_header.pack(fill=tk.X, pady=(10, 5))
        tk.Label(llm_header, text="Коррекция через LLM", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.llm_enabled_var = tk.BooleanVar(value=config.get("llm_enabled", True))
        self.llm_check = tk.Checkbutton(llm_header, variable=self.llm_enabled_var,
                                         command=self._on_llm_toggle)
        self.llm_check.pack(side=tk.LEFT, padx=(10, 0))

        self.llm_row = tk.Frame(frame)
        self.llm_row.pack(fill=tk.X, pady=(0, 3))
        tk.Label(self.llm_row, text="Ollama модель:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)

        current_llm = config.get("llm_model", "qwen3.5:9b")
        self.llm_var = tk.StringVar(value=current_llm)
        self.llm_combo = ttk.Combobox(self.llm_row, textvariable=self.llm_var,
                                      values=[current_llm], font=("Segoe UI", 10), width=25)
        self.llm_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(self.llm_row, text="⟳", command=lambda: threading.Thread(target=self._refresh_models, daemon=True).start(),
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(5, 0))

        # Fetch Ollama models in background to avoid blocking UI
        threading.Thread(target=self._refresh_models, daemon=True).start()

        # Hide LLM row if disabled
        if not self.llm_enabled_var.get():
            self.llm_row.pack_forget()

        # ── Language ──
        tk.Label(frame, text="Язык распознавания", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))
        lang_row = tk.Frame(frame)
        lang_row.pack(fill=tk.X, pady=(0, 5))
        self.lang_var = tk.StringVar(value=config.get("language", "auto"))
        lang_codes = list(LANGUAGE_OPTIONS.keys())
        tk.Label(lang_row, text="Язык:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        lang_combo = ttk.Combobox(lang_row, textvariable=self.lang_var, values=lang_codes,
                                  state="readonly", font=("Segoe UI", 10), width=8)
        lang_combo.pack(side=tk.LEFT)
        self.lang_display = tk.Label(lang_row, text=LANGUAGE_OPTIONS.get(self.lang_var.get(), ""),
                                     font=("Segoe UI", 10), fg="#555")
        self.lang_display.pack(side=tk.LEFT, padx=(10, 0))
        self.lang_var.trace_add("write", self._on_lang_change)

        # ── Input Device ──
        tk.Label(frame, text="Микрофон", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))
        mic_row = tk.Frame(frame)
        mic_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(mic_row, text="Устройство:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)

        input_devices = get_input_devices()
        self._mic_map = {d[1]: d[0] for d in input_devices}
        mic_names = ["По умолчанию"] + [d[1] for d in input_devices]

        current_dev = config.get("input_device")
        if current_dev is not None:
            if isinstance(current_dev, str) and current_dev in self._mic_map:
                current_name = current_dev
            else:
                current_name = "По умолчанию"  # old int index or stale name
        else:
            current_name = "По умолчанию"

        self.mic_var = tk.StringVar(value=current_name)
        mic_combo = ttk.Combobox(mic_row, textvariable=self.mic_var, values=mic_names,
                                 state="readonly", font=("Segoe UI", 10), width=35)
        mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── Hotkeys ──
        tk.Label(frame, text="Горячие клавиши", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))

        row1 = tk.Frame(frame)
        row1.pack(fill=tk.X, pady=3)
        tk.Label(row1, text="Запись:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.record_var = tk.StringVar(value=config["hotkey_record"])
        self.record_entry = tk.Entry(row1, textvariable=self.record_var, font=("Segoe UI", 10),
                                     state="readonly", readonlybackground="white")
        self.record_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(row1, text="Изменить", command=lambda: self._capture_hotkey("record")).pack(side=tk.LEFT)

        row2 = tk.Frame(frame)
        row2.pack(fill=tk.X, pady=3)
        tk.Label(row2, text="Выход:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.quit_var = tk.StringVar(value=config["hotkey_quit"])
        self.quit_entry = tk.Entry(row2, textvariable=self.quit_var, font=("Segoe UI", 10),
                                   state="readonly", readonlybackground="white")
        self.quit_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(row2, text="Изменить", command=lambda: self._capture_hotkey("quit")).pack(side=tk.LEFT)

        # Buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        tk.Button(btn_frame, text="Сохранить", font=("Segoe UI", 10), command=self._save, width=12).pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(btn_frame, text="Отмена", font=("Segoe UI", 10), command=self.win.destroy, width=12).pack(side=tk.RIGHT)

        self._capturing = None
        self._capture_window = None

    def _on_asr_change(self, *args):
        backend = self.asr_var.get()
        if backend == "whisper":
            self.asr_hint.config(text="~3s загрузка, авто-язык")
        else:
            self.asr_hint.config(text="~20s загрузка, меньше VRAM")
        self._toggle_model_row()

    def _toggle_model_row(self):
        """Show whisper model row only when whisper is selected."""
        if not hasattr(self, "whisper_row"):
            return
        if self.asr_var.get() == "whisper":
            # Pack right after the asr_row
            self.whisper_row.pack(fill=tk.X, pady=(0, 3), after=self.asr_row)
        else:
            self.whisper_row.pack_forget()

    def _on_llm_toggle(self):
        if self.llm_enabled_var.get():
            self.llm_row.pack(fill=tk.X, pady=(0, 3), after=self.llm_check.master)
        else:
            self.llm_row.pack_forget()

    def _on_lang_change(self, *args):
        lang = self.lang_var.get()
        self.lang_display.config(text=LANGUAGE_OPTIONS.get(lang, ""))

    def _refresh_models(self):
        models = fetch_ollama_models()
        if models:
            current = self.llm_var.get()
            if current and current not in models:
                models.insert(0, current)
            try:
                self.win.after(0, lambda m=models: self._safe_update_models(m))
            except tk.TclError:
                pass

    def _safe_update_models(self, models):
        try:
            self.llm_combo.configure(values=models)
        except tk.TclError:
            pass

    def _capture_hotkey(self, which):
        if self._capture_window:
            self._capture_window.destroy()

        self._capturing = which
        cw = tk.Toplevel(self.win)
        cw.title("Нажмите комбинацию")
        cw.geometry("300x100")
        cw.resizable(False, False)
        cw.attributes("-topmost", True)
        cw.grab_set()
        self._capture_window = cw

        tk.Label(cw, text="Нажмите нужную комбинацию клавиш...", font=("Segoe UI", 10), pady=20).pack()
        tk.Label(cw, text="(Esc для отмены)", font=("Segoe UI", 8), fg="#999").pack()

        def capture_thread():
            hotkey = keyboard.read_hotkey(suppress=False)
            try:
                if hotkey == "esc":
                    cw.after(0, cw.destroy)
                    return
                def update():
                    if which == "record":
                        self.record_var.set(hotkey)
                    else:
                        self.quit_var.set(hotkey)
                    cw.destroy()
                cw.after(0, update)
            except tk.TclError:
                pass

        threading.Thread(target=capture_thread, daemon=True).start()

    def _save(self):
        global config

        old_asr = config.get("asr_backend", "whisper")
        old_whisper = config.get("whisper_model", "large-v3")
        old_llm = config.get("llm_model", "qwen3.5:9b")
        old_llm_enabled = config.get("llm_enabled", True)

        config["hotkey_record"] = self.record_var.get()
        config["hotkey_quit"] = self.quit_var.get()
        config["language"] = self.lang_var.get()
        config["asr_backend"] = self.asr_var.get()
        config["whisper_model"] = self.whisper_var.get()
        config["llm_model"] = self.llm_var.get()
        config["llm_enabled"] = self.llm_enabled_var.get()
        mic_name = self.mic_var.get()
        config["input_device"] = self._mic_map.get(mic_name) if mic_name != "По умолчанию" else None
        save_config(config)

        # Re-register hotkeys
        register_hotkeys()

        if app:
            app.update_info_display()

        new_asr = config["asr_backend"]
        new_whisper = config["whisper_model"]
        new_llm = config["llm_model"]

        # Reload ASR model if backend or whisper model changed
        asr_changed = (old_asr != new_asr) or (new_asr == "whisper" and old_whisper != new_whisper)
        if asr_changed:
            print(f"[config] ASR backend changed to: {new_asr}")
            threading.Thread(target=reload_asr_model, daemon=True).start()

        # Warm up LLM if model changed or LLM was just enabled
        llm_now_enabled = config.get("llm_enabled", True)
        if llm_now_enabled and (old_llm != new_llm or not old_llm_enabled):
            print(f"[config] LLM model: {new_llm}")
            threading.Thread(target=warmup_llm, args=(new_llm,), daemon=True).start()

        lang = config["language"]
        lang_label = LANGUAGE_OPTIONS.get(lang, lang)
        print(f"[config] Settings saved. Language: {lang_label}")

        self.win.destroy()


# ──────────────────── VAD (for Canary) ────────────────────

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


# ──────────────────── ASR Backends ────────────────────

def load_whisper_model():
    model_path = config.get("whisper_model", "large-v3")
    # Expand ~ in path
    if model_path.startswith("~"):
        model_path = os.path.expanduser(model_path)
    print(f"[whisper] Loading model: {model_path}")
    if app:
        app.set_status(OverlayApp.STATUS_LOADING, "(Whisper)")
    from faster_whisper import WhisperModel
    t = time.time()
    model = WhisperModel(model_path, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(f"[whisper] Model loaded in {time.time()-t:.1f}s")
    return model


def load_canary_model():
    print("[canary] Loading NVIDIA Canary-1B-v2 (this may take a while on first run)...")
    if app:
        app.set_status(OverlayApp.STATUS_LOADING, "(Canary)")
    load_vad()
    import nemo.collections.asr as nemo_asr
    t = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-v2")
    model = model.to("cuda")
    model.eval()
    print(f"[canary] Model loaded in {time.time()-t:.1f}s")
    return model


def transcribe_whisper(model, audio_data):
    lang = config.get("language", "auto")
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
    lang = config.get("language", "auto")
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
    backend = config.get("asr_backend", "whisper")
    if backend == "canary":
        _asr_model = load_canary_model()
    else:
        _asr_model = load_whisper_model()
    _asr_backend = backend


def reload_asr_model():
    """Reload ASR model (called when settings change). Waits for any active recording to finish."""
    global _asr_model, _asr_backend

    # Wait for any active processing to finish before swapping models
    with processing_lock:
        print("[config] Reloading ASR model...")

        # Free old model
        old_model = _asr_model
        _asr_model = None
        _asr_backend = None
        del old_model

        # Try to free GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        load_asr_model()

    if app:
        app.set_status(OverlayApp.STATUS_IDLE)
    print("[config] ASR model reloaded")


def transcribe(audio_data):
    """Transcribe using the currently loaded ASR backend."""
    if _asr_backend == "canary":
        return transcribe_canary(_asr_model, audio_data)
    else:
        return transcribe_whisper(_asr_model, audio_data)


# ──────────────────── LLM ────────────────────

def llm_correct(text):
    if not text.strip():
        return text
    model = config.get("llm_model", "qwen3.5:9b")
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": get_llm_system()},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                },
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()
        return result if result else text
    except Exception as e:
        print(f"[llm] Error: {e}")
        return text


def warmup_llm(model_name=None):
    """Warm up an Ollama model."""
    model = model_name or config.get("llm_model", "qwen3.5:9b")
    print(f"[llm] Warming up {model}...")
    try:
        httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1},
            },
            timeout=60.0,
        )
        print(f"[llm] {model} ready")
    except Exception as e:
        print(f"[llm] Warning: Ollama not responding: {e}")
        print("[llm] Make sure Ollama is running!")


# ──────────────────── Audio / Recording ────────────────────

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
    global is_recording
    buf = []

    stop_recording_event.clear()

    device_name = config.get("input_device")
    device_id = _find_capture_device_id(device_name)

    def capture_gen():
        while True:
            data = yield
            buf.append(bytes(data))

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
    with recording_lock:
        is_recording = True
    if app:
        app.set_status(OverlayApp.STATUS_RECORDING)

    try:
        while not stop_recording_event.is_set():
            time.sleep(0.05)
    finally:
        with recording_lock:
            is_recording = False
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
    if not processing_lock.acquire(blocking=False):
        return
    try:
        if _asr_model is None:
            print("[skip] ASR model not loaded yet")
            if app:
                app.set_status(OverlayApp.STATUS_IDLE)
            return

        try:
            audio = record_audio()
        except Exception as e:
            print(f"[error] Microphone error: {e}")
            if app:
                app.set_status(OverlayApp.STATUS_IDLE)
            return
        if len(audio) < SAMPLE_RATE * 0.3:
            print("[skip] Too short")
            if app:
                app.set_status(OverlayApp.STATUS_IDLE)
            return

        # VAD filtering for Canary backend
        if _asr_backend == "canary":
            audio = filter_silence(audio)
            if len(audio) < SAMPLE_RATE * 0.3:
                print("[skip] No speech detected")
                if app:
                    app.set_status(OverlayApp.STATUS_IDLE)
                return

        # Step 1: Transcription
        if app:
            app.set_status(OverlayApp.STATUS_TRANSCRIBING)
        t = time.time()
        backend_tag = "canary" if _asr_backend == "canary" else "whisper"
        raw_text = transcribe(audio)
        asr_time = time.time() - t
        print(f"[{backend_tag}] ({asr_time:.2f}s) {raw_text}")

        if not raw_text.strip():
            print("[skip] Empty transcription")
            if app:
                app.set_status(OverlayApp.STATUS_IDLE)
            return

        # Step 2: LLM correction (if enabled)
        if config.get("llm_enabled", True):
            if app:
                app.set_status(OverlayApp.STATUS_CORRECTING)
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
        add_to_history(raw_text, corrected, backend_tag, total)
        print(f"[done] Total: {total:.2f}s")
        if app:
            app.set_status(OverlayApp.STATUS_DONE, f"({total:.1f}s)")
    finally:
        processing_lock.release()


# ──────────────────── Hotkey toggle logic ────────────────────

def on_record_hotkey():
    global is_recording
    with recording_lock:
        currently_recording = is_recording
    if currently_recording:
        stop_recording_event.set()
    else:
        threading.Thread(target=process_recording, daemon=True).start()


_hotkey_ids = []


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
            config["hotkey_record"],
            on_record_hotkey,
            suppress=True,
            trigger_on_release=False,
        )
        _hotkey_ids.append(hid)
    except Exception as e:
        print(f"[hotkey] Failed to register record hotkey: {e}")

    try:
        hid = keyboard.add_hotkey(
            config["hotkey_quit"],
            lambda: app._shutdown() if app else os._exit(0),
            suppress=True,
            trigger_on_release=False,
        )
        _hotkey_ids.append(hid)
    except Exception as e:
        print(f"[hotkey] Failed to register quit hotkey: {e}")


# ──────────────────── Main ────────────────────

def background_init():
    """Load models in background thread, then register hotkeys."""
    try:
        load_asr_model()
    except Exception as e:
        print(f"[error] Failed to load ASR model: {e}")
        print("[error] Check that the model exists and CUDA is available.")
        if app:
            app.set_status(OverlayApp.STATUS_IDLE)

    if config.get("llm_enabled", True):
        warmup_llm()

    register_hotkeys()

    hotkey = config["hotkey_record"].upper()
    backend = config.get("asr_backend", "whisper").capitalize()
    llm = config.get("llm_model", "qwen3.5:9b")
    print(f"\nReady! Press {hotkey} to start/stop dictation.")
    print(f"  ASR: {backend}  |  LLM: {llm}\n")
    if app:
        app.set_status(OverlayApp.STATUS_IDLE)


def main():
    global app

    backend = config.get("asr_backend", "whisper").capitalize()
    llm = config.get("llm_model", "qwen3.5:9b")

    print("=" * 50)
    print(f"  Voice Dictation ({backend} + {llm})")
    print("=" * 50)
    print(f"  Record:  press {config['hotkey_record'].upper()}")
    print(f"  Quit:    {config['hotkey_quit'].upper()}")
    print(f"  Settings: right-click the overlay")
    print("=" * 50)

    app = OverlayApp()
    threading.Thread(target=background_init, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()
