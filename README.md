# VLM Camera

Local, real-time Vision-Language-Model analysis of any webcam or RTSP network
camera, running fully on-device on Apple Silicon via
[MLX-VLM](https://github.com/Blaizzy/mlx-vlm) (or on any OS via a local
[Ollama](https://ollama.com) server). Includes an always-on Telegram bot that
can answer free-form questions about the live feed or summarize any recent
window on demand.

## Screenshots

**Web UI — live preview + structured per-frame VLM output**

![Web UI](docs/webui.png)

**Telegram bot — ask questions, request a snapshot, or a recent-activity summary**

![Telegram bot](docs/telegram.jpg)

## Features

- **Local inference.** Models are cached in `~/.cache/huggingface` after first
  download; subsequent launches run fully offline.
- **Backends.** MLX-VLM (Apple Silicon, fastest) or Ollama (any OS). Ollama VLM
  models on `localhost:11434` are auto-detected.
- **Cameras.** Built-in/USB webcams auto-probed; network cameras (Tapo,
  Hikvision, Reolink, any RTSP source) configured via a local `cameras.json`.
- **Monitor Mode.** Domain-specific profiles (Kid / Office / Retail / Home
  Security) that emit structured JSON per frame, push high-priority alerts to
  Telegram, and deliver periodic activity summaries.
- **Telegram Q&A.** Send natural-language questions (ZH or EN) — a VLM-backed
  intent classifier routes between _visual-current_ ("how many people?",
  "is the floor clean?"), _history_ ("summarize the past 15 min"), and
  _snapshot_ (returns the current frame as a photo).
- **Thread-safe MLX.** All model calls are funneled through a single dedicated
  worker thread so the capture loop and the Telegram listener never race on
  Metal command buffers.

## Quick Start

### Requirements

- macOS on Apple Silicon (for MLX backend) — or any OS with a local Ollama
  server running a VLM model (LLaVA, Qwen2.5-VL, MiniCPM-V, Gemma 3, …)
- Python 3.11+
- A webcam or an RTSP-capable IP camera

### Install

```bash
git clone https://github.com/Drlucaslu/vlm-camera.git
cd vlm-camera
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Configure (optional but recommended)

```bash
cp .env.example .env                     # Telegram token + chat ID
cp cameras.json.example cameras.json     # one entry per network camera
```

Edit `.env`:

```ini
TELEGRAM_BOT_TOKEN=123456:ABC-DEF…
TELEGRAM_CHAT_ID=12345678
```

Edit `cameras.json` — any number of entries, any brand:

```json
[
  { "name": "Tapo Living Room (HD)",
    "url": "rtsp://USER:PASS@192.168.1.10:554/stream1" },
  { "name": "Hikvision Garage",
    "url": "rtsp://USER:PASS@192.168.1.20:554/Streaming/Channels/101" }
]
```

Both files are gitignored — credentials never leave the machine.

### Run

```bash
./start.sh
```

Open <http://127.0.0.1:7860>.

## Usage

### Web UI

1. Pick a **Model** and a **Camera** (local index or one of your `cameras.json`
   entries, or "Network Camera (custom RTSP URL)" to paste a URL live).
2. Choose a **Preset** prompt (Person Action / Scene / Object / Custom) and an
   **Interval** (how often to run inference).
3. Optionally expand **Monitor Mode** to enable a profile that emits
   structured JSON, sends alerts, and posts periodic summaries.
4. Hit **Start**. Results stream on the right; the preview stays live.

### Telegram

Send any of the following to your bot:

| Message | What happens |
|---|---|
| `房间里有几个人？` / `how many people?` | VLM runs on the latest frame with your question as the prompt |
| `地面干净吗？` / `is the floor clean?` | Same — free-form visual question |
| `过去 15 分钟怎么样？` / `summary last hour` | Text-only VLM summarizes the recent capture-loop results |
| `截图` / `/snapshot` | Returns the current frame as a photo |
| `/help` · `/status` | Help + running state |

Only messages from the configured `TELEGRAM_CHAT_ID` are answered; others are
dropped.

## Configuration Reference

### `cameras.json`

A JSON list of `{ "name": ..., "url": ... }`. Each URL embeds its own
credentials and may use any scheme OpenCV/FFmpeg understands (`rtsp://`,
`http://`, `https://`). The file is loaded at startup; restart the app after
editing.

For Tapo, set up a **Camera Account** in the Tapo app (Advanced Settings →
Camera Account) and enable RTSP/ONVIF, then use
`rtsp://USER:PASS@HOST:554/stream1` (HD) or `/stream2` (SD).

### `.env`

| Variable | Required? | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Optional | Enables push notifications + the incoming listener |
| `TELEGRAM_CHAT_ID` | Optional | Locks the bot to one chat (your user ID) |
| `OLLAMA_BASE_URL` | Optional | Defaults to `http://localhost:11434` |
| `GRADIO_SERVER_PORT` | Optional | Defaults to `7860` |

## Architecture

- **`app.py`** — Gradio UI, capture loop, VLM orchestration.
- **`listener.py`** — Telegram long-polling listener + intent classifier.
- **`monitor.py`** — Scene-profile definitions, structured-output parser,
  activity log, alert manager, and periodic summary scheduler.
- **`notify.py`** — Telegram send-only helpers (stdlib `urllib` only).

All MLX operations (model load, inference, unload) are funneled through a
single `vlm-worker` thread. This is load-bearing: MLX arrays and Metal command
buffers are bound to the thread that allocated them, so a naive "just add a
lock" approach crashes with `failed assertion 'A command encoder is already
encoding to this command buffer'` once a second thread (e.g. the Telegram
listener) starts issuing inference requests.

## Status

Hobby project, built iteratively with [Claude Code](https://claude.com/claude-code).
No tests, no versioning. Use at your own risk, especially if you point it at a
camera that sees people who haven't agreed to be analyzed.

## License

Not yet chosen. Treat as "all rights reserved" for now.
