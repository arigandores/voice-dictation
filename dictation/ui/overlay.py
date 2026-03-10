"""Small always-on-top status overlay widget."""

import os
import tkinter as tk

from dictation import state
from dictation.hotkeys import _hotkey_ids


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
        hotkey = state.config["hotkey_record"].upper()
        backend = state.config.get("asr_backend", "whisper").capitalize()
        if state.config.get("llm_enabled", True):
            llm = state.config.get("llm_model", "qwen3.5:9b")
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
        from dictation.ui.history import HistoryWindow
        HistoryWindow(self.root)

    def _open_settings(self):
        from dictation.ui.settings import SettingsWindow
        SettingsWindow(self.root)

    def _quit(self):
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown: unregister hotkeys, destroy tkinter, then exit."""
        try:
            import keyboard
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
