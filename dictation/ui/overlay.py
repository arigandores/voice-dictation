"""Small always-on-top status overlay widget with waveform visualizer."""

import math
import os
import tkinter as tk

from dictation import state


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
        "idle": "\u23f3 \u0413\u043e\u0442\u043e\u0432 \u043a \u0437\u0430\u043f\u0438\u0441\u0438",
        "recording": "\U0001f534 \u0417\u0430\u043f\u0438\u0441\u044c...",
        "transcribing": "\U0001f4dd \u0422\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0446\u0438\u044f",
        "correcting": "\u2728 \u041a\u043e\u0440\u0440\u0435\u043a\u0446\u0438\u044f",
        "done": "\u2705 \u0413\u043e\u0442\u043e\u0432\u043e!",
        "loading": "\u23f3 \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043c\u043e\u0434\u0435\u043b\u0438",
    }

    # Waveform settings
    WAVE_HEIGHT = 34
    WAVE_POINTS = 40       # number of amplitude points in the wave
    WAVE_UPDATE_MS = 50
    WAVE_MAX_RMS = 0.12    # RMS reference for full-scale

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dictation")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)

        self.width = 220
        self._height_normal = 58
        self._height_recording = 58 + self.WAVE_HEIGHT + 8
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - self.width - 20
        y = 20
        self.root.geometry(f"{self.width}x{self._height_normal}+{x}+{y}")

        # Draggable
        self._drag_data = {"x": 0, "y": 0}
        self.root.bind("<Button-1>", self._on_drag_start)
        self.root.bind("<B1-Motion>", self._on_drag_motion)

        # Right-click menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="\U0001f4cb \u0418\u0441\u0442\u043e\u0440\u0438\u044f", command=self._open_history)
        self.menu.add_command(label="\u2699 \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438", command=self._open_settings)
        self.menu.add_separator()
        self.menu.add_command(label="\u2715 \u0412\u044b\u0445\u043e\u0434", command=self._quit)
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

        # Live transcription preview (hidden by default)
        self._live_label = tk.Label(
            self.frame,
            text="",
            font=("Segoe UI", 8),
            fg="#dddddd",
            bg=self.COLORS["recording"],
            anchor="w",
            wraplength=self.width - 24,
            justify="left",
        )

        # Waveform canvas (hidden by default)
        self._wave_canvas = tk.Canvas(
            self.frame,
            height=self.WAVE_HEIGHT,
            bg=self.COLORS["recording"],
            highlightthickness=0,
        )
        self._wave_display = [0.0] * self.WAVE_POINTS
        self._wave_last_rms = 0.0  # held value between audio chunks
        self._wave_job = None

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
            llm = "LLM \u0432\u044b\u043a\u043b"
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

        # Waveform: show during recording, hide otherwise
        if status == self.STATUS_RECORDING:
            self._show_waveform(color)
        else:
            self._hide_waveform()

        if status in (self.STATUS_RECORDING, self.STATUS_TRANSCRIBING, self.STATUS_CORRECTING, self.STATUS_LOADING):
            self._anim_dots = 0
            self._animate(status)

        if status == self.STATUS_DONE:
            self._done_hide_job = self.root.after(2000, lambda: self._update_status(self.STATUS_IDLE))

        self.root.deiconify()
        self.root.attributes("-topmost", True)

    def _show_waveform(self, bg_color):
        """Show waveform canvas and start updating."""
        self._wave_canvas.config(bg=bg_color)
        self._wave_canvas.pack(fill=tk.X, pady=(6, 0))
        self._wave_display = [0.0] * self.WAVE_POINTS
        self._wave_last_rms = 0.0
        # Show live preview if streaming enabled
        self._live_text = ""
        if state.config.get("llm_streaming", False):
            self._live_label.config(text="", bg=bg_color)
            self._live_label.pack(fill=tk.X, pady=(4, 0))
        # Resize window to fit waveform + preview
        self._resize_recording_window()
        self._wave_update()

    def _hide_waveform(self):
        """Hide waveform canvas and stop updating."""
        if self._wave_job:
            self.root.after_cancel(self._wave_job)
            self._wave_job = None
        self._wave_canvas.pack_forget()
        self._live_label.pack_forget()
        self._live_text = ""
        # Resize window back to normal
        geo = self.root.geometry()
        pos = geo.split("+", 1)[1]
        self.root.geometry(f"{self.width}x{self._height_normal}+{pos}")

    def _resize_recording_window(self):
        """Resize overlay window to fit waveform and optional live preview."""
        h = self._height_recording
        if state.config.get("llm_streaming", False):
            h += 40  # extra space for live transcription text
        geo = self.root.geometry()
        pos = geo.split("+", 1)[1]
        self.root.geometry(f"{self.width}x{h}+{pos}")

    def set_live_text(self, text):
        """Update live transcription preview (called from background thread)."""
        self.root.after(0, self._update_live_text, text)

    def _update_live_text(self, text):
        if self._current_status != self.STATUS_RECORDING:
            return
        # Show last ~80 chars to fit the small overlay
        display = text.strip()
        if len(display) > 80:
            display = "..." + display[-77:]
        self._live_label.config(text=display)

    def _wave_update(self):
        """Read new amplitude samples, smooth, and redraw."""
        if self._current_status != self.STATUS_RECORDING:
            return

        # Drain all pending samples from the shared deque
        levels = state.waveform_levels
        new_samples = []
        while True:
            try:
                new_samples.append(levels.popleft())
            except IndexError:
                break

        # Compute target: use new data or decay from last known level
        if new_samples:
            self._wave_last_rms = sum(new_samples) / len(new_samples)
        else:
            # Slow decay — keeps the wave smooth when chunks arrive unevenly
            self._wave_last_rms *= 0.85

        # Scroll left and push current level
        self._wave_display.pop(0)
        self._wave_display.append(self._wave_last_rms)

        self._draw_waveform()
        self._wave_job = self.root.after(self.WAVE_UPDATE_MS, self._wave_update)

    def _draw_waveform(self):
        """Draw a smooth filled waveform polygon mirrored from center."""
        c = self._wave_canvas
        c.delete("all")

        w = c.winfo_width()
        h = self.WAVE_HEIGHT
        if w < 10:
            w = self.width - 20

        n = self.WAVE_POINTS
        mid_y = h / 2
        max_half = mid_y - 2  # max amplitude in pixels

        # Build normalized amplitudes
        amps = []
        for rms in self._wave_display:
            norm = min(rms / self.WAVE_MAX_RMS, 1.0)
            norm = math.sqrt(norm)
            amps.append(norm)

        # Build polygon points: top curve (left to right), then bottom curve (right to left)
        step = w / max(n - 1, 1)
        top_points = []
        bottom_points = []
        for i, amp in enumerate(amps):
            x = i * step
            offset = max(1, amp * max_half)
            top_points.append(x)
            top_points.append(mid_y - offset)
            bottom_points.append(x)
            bottom_points.append(mid_y + offset)

        # Reverse bottom so the polygon closes properly
        bottom_points_rev = []
        for i in range(len(bottom_points) - 2, -1, -2):
            bottom_points_rev.append(bottom_points[i])
            bottom_points_rev.append(bottom_points[i + 1])

        poly = top_points + bottom_points_rev

        if len(poly) >= 6:
            # Filled smooth wave shape
            c.create_polygon(
                poly,
                fill="#ffffff",
                outline="",
                smooth=True,
                splinesteps=12,
            )
            # Thinner brighter center line for detail
            center_line = []
            for i, amp in enumerate(amps):
                x = i * step
                offset = max(1, amp * max_half * 0.4)
                center_line.append(x)
                center_line.append(mid_y - offset)
            # Add bottom half
            for i in range(len(amps) - 1, -1, -1):
                x = i * step
                offset = max(1, amps[i] * max_half * 0.4)
                center_line.append(x)
                center_line.append(mid_y + offset)

            c.create_polygon(
                center_line,
                fill="#e8b4b0",
                outline="",
                smooth=True,
                splinesteps=12,
            )

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
            from dictation.hotkeys import unregister_hotkeys
            unregister_hotkeys()
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
