"""Telegram notifications — stdlib-only, fire-and-forget.

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from environment. Methods never
raise; they log on failure and return False so the capture loop keeps running.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
import urllib.request
from io import BytesIO

log = logging.getLogger("vlm-camera.notify")

_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def _url(self, method: str) -> str:
        return _API.format(token=self.bot_token, method=method)

    def send(self, text: str) -> bool:
        if not self.chat_id:
            return False
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        try:
            req = urllib.request.Request(
                self._url("sendMessage"),
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read()).get("ok", False)
        except Exception:
            log.debug("Telegram send failed", exc_info=True)
            return False

    def send_photo(self, jpeg_bytes: bytes, caption: str = "") -> bool:
        if not self.chat_id:
            return False
        boundary = f"----vlmcamera{uuid.uuid4().hex}"
        body = BytesIO()

        def part(name: str, value: str):
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(value.encode())
            body.write(b"\r\n")

        part("chat_id", str(self.chat_id))
        if caption:
            part("caption", caption[:1024])  # Telegram caption limit
            part("parse_mode", "Markdown")

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="photo"; filename="frame.jpg"\r\n')
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(jpeg_bytes)
        body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())

        try:
            req = urllib.request.Request(
                self._url("sendPhoto"),
                data=body.getvalue(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read()).get("ok", False)
        except Exception:
            log.debug("Telegram send_photo failed", exc_info=True)
            return False


def from_env() -> TelegramNotifier | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.info("Telegram disabled: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return None
    log.info("Telegram notifier enabled for chat %s", chat_id)
    return TelegramNotifier(token, chat_id)
