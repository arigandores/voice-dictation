"""Scrollable transcription history window."""

import tkinter as tk
import pyperclip

from dictation import state


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

        # Mouse wheel scrolling (scoped to this window only)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.win.bind("<Destroy>", self._on_destroy)

        self._populate()

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_destroy(self, event):
        pass

    def _populate(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        with state._history_lock:
            entries = list(reversed(state._history))

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
            self.win.after(1000, lambda t=orig_title: self._safe_set_title(t))
        except tk.TclError:
            pass

    def _safe_set_title(self, title):
        try:
            self.win.title(title)
        except tk.TclError:
            pass

    def _clear(self):
        with state._history_lock:
            state._history.clear()
        self._populate()
