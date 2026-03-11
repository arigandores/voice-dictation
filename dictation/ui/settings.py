"""Settings dialog for all configuration."""

import threading
import tkinter as tk
from tkinter import ttk

from dictation import state
from dictation.config import (
    LANGUAGE_OPTIONS, WHISPER_MODELS, fetch_ollama_models, get_input_devices, save_config,
)


class SettingsWindow:
    """Settings dialog for all configuration."""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Настройки диктовки")
        self.win.geometry("460x680")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self.win.grab_set()

        frame = tk.Frame(self.win, padx=20, pady=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(frame, text="Настройки", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(frame, text="", font=("Segoe UI", 2)).pack()

        # -- ASR Backend --
        tk.Label(frame, text="Модель транскрипции (ASR)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5, 5))

        self.asr_row = tk.Frame(frame)
        self.asr_row.pack(fill=tk.X, pady=(0, 3))

        self.asr_var = tk.StringVar(value=state.config.get("asr_backend", "whisper"))
        tk.Label(self.asr_row, text="Движок:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        asr_combo = ttk.Combobox(self.asr_row, textvariable=self.asr_var, values=["whisper", "canary", "parakeet", "qwen"],
                                 state="readonly", font=("Segoe UI", 10), width=15)
        asr_combo.pack(side=tk.LEFT)
        self.asr_hint = tk.Label(self.asr_row, text="", font=("Segoe UI", 8), fg="#888")
        self.asr_hint.pack(side=tk.LEFT, padx=(10, 0))
        self.asr_var.trace_add("write", self._on_asr_change)
        self._on_asr_change()

        # Whisper model path (shown only when whisper selected)
        self.whisper_row = tk.Frame(frame)
        tk.Label(self.whisper_row, text="Whisper модель:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.whisper_var = tk.StringVar(value=state.config.get("whisper_model", "large-v3"))
        self.whisper_combo = ttk.Combobox(self.whisper_row, textvariable=self.whisper_var,
                                          values=WHISPER_MODELS, font=("Segoe UI", 10), width=35)
        self.whisper_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Show/hide based on initial backend
        self._toggle_model_row()

        # -- LLM Model --
        llm_header = tk.Frame(frame)
        llm_header.pack(fill=tk.X, pady=(10, 5))
        tk.Label(llm_header, text="Коррекция через LLM", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.llm_enabled_var = tk.BooleanVar(value=state.config.get("llm_enabled", True))
        self.llm_check = tk.Checkbutton(llm_header, variable=self.llm_enabled_var,
                                         command=self._on_llm_toggle)
        self.llm_check.pack(side=tk.LEFT, padx=(10, 0))

        self.llm_row = tk.Frame(frame)
        self.llm_row.pack(fill=tk.X, pady=(0, 3))
        tk.Label(self.llm_row, text="Ollama модель:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)

        current_llm = state.config.get("llm_model", "qwen3.5:9b")
        self.llm_var = tk.StringVar(value=current_llm)
        self.llm_combo = ttk.Combobox(self.llm_row, textvariable=self.llm_var,
                                      values=[current_llm], font=("Segoe UI", 10), width=25)
        self.llm_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(self.llm_row, text="⟳", command=lambda: threading.Thread(target=self._refresh_models, daemon=True).start(),
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(5, 0))

        # Fetch Ollama models in background
        threading.Thread(target=self._refresh_models, daemon=True).start()

        # Hide LLM row if disabled
        if not self.llm_enabled_var.get():
            self.llm_row.pack_forget()

        # -- Language --
        tk.Label(frame, text="Язык распознавания", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))
        lang_row = tk.Frame(frame)
        lang_row.pack(fill=tk.X, pady=(0, 5))
        self.lang_var = tk.StringVar(value=state.config.get("language", "auto"))
        lang_codes = list(LANGUAGE_OPTIONS.keys())
        tk.Label(lang_row, text="Язык:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        lang_combo = ttk.Combobox(lang_row, textvariable=self.lang_var, values=lang_codes,
                                  state="readonly", font=("Segoe UI", 10), width=8)
        lang_combo.pack(side=tk.LEFT)
        self.lang_display = tk.Label(lang_row, text=LANGUAGE_OPTIONS.get(self.lang_var.get(), ""),
                                     font=("Segoe UI", 10), fg="#555")
        self.lang_display.pack(side=tk.LEFT, padx=(10, 0))
        self.lang_var.trace_add("write", self._on_lang_change)

        # -- Input Device --
        tk.Label(frame, text="Микрофон", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))
        mic_row = tk.Frame(frame)
        mic_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(mic_row, text="Устройство:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)

        input_devices = get_input_devices()
        mic_names = ["По умолчанию"] + input_devices

        current_dev = state.config.get("input_device")
        if current_dev and isinstance(current_dev, str) and current_dev in input_devices:
            current_name = current_dev
        else:
            current_name = "По умолчанию"

        self.mic_var = tk.StringVar(value=current_name)
        mic_combo = ttk.Combobox(mic_row, textvariable=self.mic_var, values=mic_names,
                                 state="readonly", font=("Segoe UI", 10), width=35)
        mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- Custom Prompt Terms --
        tk.Label(frame, text="Дополнительные термины", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 3))
        tk.Label(frame, text="Через запятую (добавляются в промпт Whisper):", font=("Segoe UI", 8), fg="#888").pack(anchor="w")
        self.custom_terms_var = tk.StringVar(value=state.config.get("custom_prompt_terms", ""))
        terms_entry = tk.Entry(frame, textvariable=self.custom_terms_var, font=("Segoe UI", 10))
        terms_entry.pack(fill=tk.X, pady=(2, 5))

        # -- Audio Processing --
        tk.Label(frame, text="Обработка аудио", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.noise_var = tk.BooleanVar(value=state.config.get("noise_reduction", False))
        tk.Checkbutton(frame, text="Шумоподавление (noisereduce)", variable=self.noise_var,
                        font=("Segoe UI", 10)).pack(anchor="w")

        # -- LLM Streaming --
        self.streaming_var = tk.BooleanVar(value=state.config.get("llm_streaming", False))
        tk.Checkbutton(frame, text="Streaming LLM (быстрее отклик)", variable=self.streaming_var,
                        font=("Segoe UI", 10)).pack(anchor="w")

        # -- Hotkeys --
        tk.Label(frame, text="Горячие клавиши", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 5))

        row1 = tk.Frame(frame)
        row1.pack(fill=tk.X, pady=3)
        tk.Label(row1, text="Запись:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.record_var = tk.StringVar(value=state.config["hotkey_record"])
        self.record_entry = tk.Entry(row1, textvariable=self.record_var, font=("Segoe UI", 10),
                                     state="readonly", readonlybackground="white")
        self.record_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(row1, text="Изменить", command=lambda: self._capture_hotkey("record")).pack(side=tk.LEFT)

        row2 = tk.Frame(frame)
        row2.pack(fill=tk.X, pady=3)
        tk.Label(row2, text="Выход:", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.quit_var = tk.StringVar(value=state.config["hotkey_quit"])
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
        elif backend == "parakeet":
            self.asr_hint.config(text="0.6B, пунктуация, 25 языков")
        elif backend == "qwen":
            self.asr_hint.config(text="Qwen3-ASR-1.7B, 30 языков")
        else:
            self.asr_hint.config(text="~20s загрузка, меньше VRAM")
        self._toggle_model_row()

    def _toggle_model_row(self):
        """Show whisper model row only when whisper is selected."""
        if not hasattr(self, "whisper_row"):
            return
        if self.asr_var.get() == "whisper":
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
        import keyboard

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
        from dictation.asr import reload_asr_model
        from dictation.hotkeys import register_hotkeys
        from dictation.llm import warmup_llm

        old_asr = state.config.get("asr_backend", "whisper")
        old_whisper = state.config.get("whisper_model", "large-v3")
        old_llm = state.config.get("llm_model", "qwen3.5:9b")
        old_llm_enabled = state.config.get("llm_enabled", True)

        state.config["hotkey_record"] = self.record_var.get()
        state.config["hotkey_quit"] = self.quit_var.get()
        state.config["language"] = self.lang_var.get()
        state.config["asr_backend"] = self.asr_var.get()
        state.config["whisper_model"] = self.whisper_var.get()
        state.config["llm_model"] = self.llm_var.get()
        state.config["llm_enabled"] = self.llm_enabled_var.get()
        state.config["llm_streaming"] = self.streaming_var.get()
        state.config["custom_prompt_terms"] = self.custom_terms_var.get().strip()
        state.config["noise_reduction"] = self.noise_var.get()
        mic_name = self.mic_var.get()
        state.config["input_device"] = mic_name if mic_name != "По умолчанию" else None
        save_config(state.config)

        # Re-register hotkeys
        register_hotkeys()

        if state.app:
            state.app.update_info_display()

        new_asr = state.config["asr_backend"]
        new_whisper = state.config["whisper_model"]
        new_llm = state.config["llm_model"]

        # Reload ASR model if backend or whisper model changed
        asr_changed = (old_asr != new_asr) or (new_asr == "whisper" and old_whisper != new_whisper)
        if asr_changed:
            print(f"[config] ASR backend changed to: {new_asr}")
            threading.Thread(target=reload_asr_model, daemon=True).start()

        # Warm up LLM if model changed or LLM was just enabled
        llm_now_enabled = state.config.get("llm_enabled", True)
        if llm_now_enabled and (old_llm != new_llm or not old_llm_enabled):
            print(f"[config] LLM model: {new_llm}")
            threading.Thread(target=warmup_llm, args=(new_llm,), daemon=True).start()

        lang = state.config["language"]
        lang_label = LANGUAGE_OPTIONS.get(lang, lang)
        print(f"[config] Settings saved. Language: {lang_label}")

        self.win.destroy()
