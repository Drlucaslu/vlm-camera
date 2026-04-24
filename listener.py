"""Telegram incoming listener — long-polling, stdlib-only.

Runs a single background thread that pulls user messages from Telegram via
getUpdates and dispatches them through a VLM-backed intent classifier to
either a visual-question-over-current-frame path, a history-summary path,
or a raw snapshot reply.

The classifier + answer functions are passed in as callables (ListenerServices)
so this module has no dependency on app.py — avoiding a circular import.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

import notify

log = logging.getLogger("vlm-camera.listener")

_API = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 25  # Telegram long-poll seconds

_HELP_TEXT = (
    "🤖 *VLM Camera Bot*\n"
    "问我关于摄像头画面的问题，或要一份过去一段时间的总结。\n\n"
    "*示例：*\n"
    "• `房间里有几个人？`\n"
    "• `地面干净吗？`\n"
    "• `桌上有什么东西？`\n"
    "• `过去 15 分钟怎么样？`\n"
    "• `截图` 或 `/snapshot`\n"
    "• `/help` · `/status`"
)


@dataclass
class ListenerServices:
    notifier: "notify.TelegramNotifier"
    classify: Callable[[str], dict]
    ask_visual: Callable[[str], str]
    ask_history: Callable[[int], str]
    snapshot_jpeg: Callable[[], bytes | None]
    status: Callable[[], str]


class TelegramListener:
    def __init__(self, services: ListenerServices, allowed_chat_id: int) -> None:
        self._svc = services
        self._allowed = int(allowed_chat_id)
        self._offset: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tg-listener", daemon=True)
        self._thread.start()
        log.info("Telegram listener started (chat=%s)", self._allowed)

    def stop(self) -> None:
        self._stop.set()

    # ---------- main loop ----------

    def _loop(self) -> None:
        try:
            self._prime_offset()
        except Exception:
            log.debug("Could not prime offset; starting from scratch", exc_info=True)

        while not self._stop.is_set():
            try:
                updates = self._get_updates()
            except Exception:
                log.debug("getUpdates failed; retry in 5s", exc_info=True)
                self._stop.wait(5)
                continue

            for upd in updates:
                self._offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = msg.get("chat", {}).get("id")
                if chat_id != self._allowed:
                    log.info("Drop message from chat %s (allowed=%s)", chat_id, self._allowed)
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                try:
                    self._handle(text)
                except Exception as e:
                    log.exception("Handler crashed")
                    self._svc.notifier.send(f"⚠️ 处理失败：{e}")

    def _prime_offset(self) -> None:
        """Skip any backlog accumulated while we were offline."""
        url = _API.format(token=self._svc.notifier.bot_token, method="getUpdates")
        data = urllib.parse.urlencode({"offset": -1, "limit": 1}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
        result = payload.get("result", [])
        if result:
            self._offset = result[-1]["update_id"] + 1
            log.info("Primed Telegram offset at %s (skipped backlog)", self._offset)

    def _get_updates(self) -> list:
        params: dict = {"timeout": _POLL_TIMEOUT}
        if self._offset is not None:
            params["offset"] = self._offset
        url = _API.format(token=self._svc.notifier.bot_token, method="getUpdates")
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=_POLL_TIMEOUT + 10) as resp:
            return json.loads(resp.read()).get("result", [])

    # ---------- dispatch ----------

    def _handle(self, text: str) -> None:
        low = text.strip().lower()

        # Fast slash-command path — no VLM call needed
        if low in ("/help", "help", "/start", "start"):
            self._svc.notifier.send(_HELP_TEXT)
            return
        if low in ("/status",):
            self._svc.notifier.send(f"📟 {self._svc.status()}")
            return
        if low in ("/snapshot", "/photo", "截图", "发图", "snapshot"):
            self._snapshot_reply()
            return

        # Everything else goes through VLM intent classifier
        intent = self._svc.classify(text)
        itype = intent.get("type", "visual")
        log.info("Intent=%s for %r", intent, text[:80])

        if itype == "not_ready":
            self._svc.notifier.send(
                "📷 摄像头/模型未启动。请到 Web UI 点 *Start* 后再问我。"
            )
        elif itype == "help":
            self._svc.notifier.send(_HELP_TEXT)
        elif itype == "snapshot":
            self._snapshot_reply()
        elif itype == "history":
            minutes = int(intent.get("minutes") or 15)
            minutes = max(1, min(minutes, 720))  # 1 min – 12 h
            self._svc.notifier.send(f"⏳ 正在总结过去 *{minutes}* 分钟…")
            answer = self._svc.ask_history(minutes)
            self._svc.notifier.send(f"📊 *过去 {minutes} 分钟*\n\n{answer}")
        else:  # "visual" or anything unknown — default to visual
            self._svc.notifier.send("🔍 正在看当前画面…")
            answer = self._svc.ask_visual(text)
            self._svc.notifier.send(f"👁️ {answer}")

    def _snapshot_reply(self) -> None:
        jpeg = self._svc.snapshot_jpeg()
        if jpeg is None:
            self._svc.notifier.send("📷 目前没有画面（摄像头未启动？）")
            return
        self._svc.notifier.send_photo(jpeg, caption="📷 当前画面")


def start_from_env(services: ListenerServices) -> TelegramListener | None:
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not chat_id:
        log.info("Listener disabled: TELEGRAM_CHAT_ID not set")
        return None
    try:
        cid = int(chat_id)
    except ValueError:
        log.warning("TELEGRAM_CHAT_ID=%r is not an integer; listener disabled", chat_id)
        return None
    lis = TelegramListener(services, cid)
    lis.start()
    return lis
