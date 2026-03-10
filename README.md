# Voice Dictation

Локальный голосовой ввод с AI-коррекцией. Аналог SuperWhisper, работающий полностью на своём железе.

**Пайплайн:** Нажал хоткей → надиктовал → нажал хоткей → ASR транскрибирует → LLM исправляет → текст вставляется в активное окно.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA_CUDA-green)

## Возможности

- **Два ASR-движка на выбор:**
  - **faster-whisper** — быстрый старт (~3с), авто-определение языка, 99+ языков
  - **NVIDIA Canary-1B-v2** — меньше VRAM (~6 GB), Silero VAD для удаления тишины, 25 европейских языков
- **LLM-коррекция** через Ollama (любая модель) — исправляет ошибки, расставляет пунктуацию, IT-термины пишет на английском
- **GUI-оверлей** — всегда поверх окон, показывает статус (запись / транскрипция / коррекция / готово)
- **Настройки через GUI** — правый клик по оверлею: выбор ASR-движка, LLM-модели, языка, хоткеев
- **История транскрипций** — клик по записи копирует текст в буфер обмена
- **LLM можно отключить** — для "сырой" транскрипции без коррекции
- Всё работает локально, никакие данные не уходят в облако

## Системные требования

| Компонент | Минимум |
|-----------|---------|
| GPU | NVIDIA с 10+ GB VRAM, CUDA 12+ |
| RAM | 16 GB |
| OS | Windows 10/11 |
| Python | 3.12 |

### Использование VRAM

| Конфигурация | ASR | LLM | Итого |
|-------------|-----|-----|-------|
| Whisper + Qwen3.5 9B | ~3 GB | ~6.6 GB | ~10 GB |
| Canary + Qwen3.5 9B | ~6 GB | ~6.6 GB | ~12.7 GB |
| Whisper + Qwen3.5 4B | ~3 GB | ~3.4 GB | ~6.4 GB |

## Установка

### 1. Python и виртуальное окружение

```powershell
pip install uv
uv python install 3.12
uv venv whisper-env --python 3.12
~\whisper-env\Scripts\Activate.ps1
```

### 2. PyTorch с CUDA

```powershell
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Зависимости

```powershell
# Основные
uv pip install numpy sounddevice keyboard pyperclip httpx

# Whisper backend
uv pip install faster-whisper

# Canary backend (опционально)
uv pip install nemo_toolkit[asr] silero-vad soundfile
```

### 4. Ollama + LLM

```powershell
winget install Ollama.Ollama
ollama pull qwen3.5:9b
```

### 5. Whisper модель (опционально — файнтюн для русского)

```powershell
python -c "
from huggingface_hub import snapshot_download
snapshot_download('dvislobokov/faster-whisper-large-v3-turbo-russian', local_dir='whisper-env/faster-whisper-ru-turbo')
"
```

Стандартные модели (`large-v3`, `large-v3-turbo`) скачиваются автоматически при первом запуске.

### 6. Проверка

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from faster_whisper import WhisperModel; print('Whisper OK')"
curl http://localhost:11434/api/tags
```

## Запуск

```powershell
~\whisper-env\Scripts\Activate.ps1
python dictation.py
```

При запуске появится маленький оверлей в правом верхнем углу экрана.

## Использование

| Действие | Способ |
|----------|--------|
| Начать запись | Нажать хоткей (по умолчанию `Ctrl+Shift+Space`) |
| Остановить запись | Нажать хоткей ещё раз |
| Настройки | Правый клик по оверлею → Настройки |
| История | Правый клик по оверлею → История |
| Выход | `Ctrl+Shift+Q` или правый клик → Выход |

### Процесс

1. Поставь курсор в нужное поле
2. Нажми хоткей — начнётся запись (оверлей станет красным)
3. Говори
4. Нажми хоткей ещё раз — запись остановится
5. Подожди ~3 сек — текст вставится через Ctrl+V

## Настройки

Все настройки доступны через GUI (правый клик по оверлею → Настройки):

| Параметр | Описание |
|----------|----------|
| **Движок ASR** | `whisper` или `canary` — переключение перезагрузит модель автоматически |
| **Whisper модель** | Путь или имя модели (`large-v3`, `large-v3-turbo`, кастомный путь) |
| **Коррекция через LLM** | Вкл/выкл — если выключить, текст вставляется без обработки |
| **Ollama модель** | Выбор из установленных моделей (подтягиваются автоматически) |
| **Язык** | `auto` для авто-определения, или конкретный язык |
| **Горячие клавиши** | Настройка хоткеев записи и выхода |

Конфиг сохраняется в `~/dictation_config.json`. При сохранении настроек модели перезагружаются автоматически.

## Архитектура

```
┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│ Микрофон │──>│  Silero VAD│──>│ Whisper /     │──>│ Ollama LLM   │──>│  Ctrl+V  │
│ (запись)  │   │ (Canary)   │   │ Canary (GPU) │   │ (коррекция)  │   │ (вставка)│
└──────────┘   └────────────┘   └──────────────┘   └──────────────┘   └──────────┘
  16kHz mono    Только Canary    Транскрипция       Опционально        В активное окно
```

### Пайплайн

1. **Запись** — `sounddevice` захватывает аудио (16 кГц, моно, float32)
2. **VAD** (только Canary) — Silero VAD удаляет тишину
3. **Транскрипция** — Whisper или Canary обрабатывает аудио на GPU
4. **Коррекция** (опционально) — LLM исправляет ошибки, расставляет пунктуацию
5. **Вставка** — текст копируется в буфер и вставляется через Ctrl+V

## Альтернативные модели

### Whisper

| Модель | Размер | Скорость |
|--------|--------|----------|
| `large-v3` | ~3 GB | Средняя |
| `large-v3-turbo` | ~1.6 GB | Быстрая |
| `faster-whisper-ru-turbo` (файнтюн) | ~1.6 GB | Быстрая |

### LLM (Ollama)

| Модель | Время | VRAM | Качество |
|--------|-------|------|----------|
| gemma3:4b | ~2.3s | 3.3 GB | Хорошее |
| qwen3.5:4b | ~2.4s | 3.4 GB | Хорошее |
| **qwen3.5:9b** | **~2.5s** | **6.6 GB** | **Лучшее** |

## Troubleshooting

### Ollama не отвечает

```
[llm] Warning: Ollama not responding
```

Убедись, что Ollama запущен: `curl http://localhost:11434/api/tags`

### CUDA не найдена

```powershell
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Warning: Triton / Megatron / OneLogger

Нормально для Windows. Не влияет на работу.

### Текст вставляется не туда

Текст вставляется в текущее активное окно через Ctrl+V. Убедись, что курсор в нужном поле до начала записи.

## Лицензия

MIT
