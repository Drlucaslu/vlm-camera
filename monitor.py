"""Kid-room monitor: structured VLM output -> alerts + periodic summaries.

Design:
- VLM is instructed (via MONITOR_PROMPT_ZH) to emit a compact JSON per frame.
- AlertManager trusts the VLM's risk_level but requires N consecutive high-risk
  frames to fire, and applies a per-category cooldown so we don't spam Telegram.
- ActivityLog is a rolling ring buffer. condense() merges consecutive identical
  activities into runs before feeding the local LLM — this is the key "filter
  out redundant info" step so the summary call stays fast and on-topic.
- SummaryScheduler runs in a background thread; when the window elapses it
  asks the already-loaded VLM (text-only) to write one Chinese summary and
  pushes it to Telegram.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Callable

from PIL import Image

log = logging.getLogger("vlm-camera.monitor")

# ---------------------------------------------------------------------------
# Scene monitor profiles
# ---------------------------------------------------------------------------
# Every profile emits the same universal fields (activity, risk_level,
# risk_reason) so the alert + summary pipeline stays generic. Domain-specific
# secondary fields (num_children, num_customers, cleanliness, ...) are allowed
# in the JSON but are only used visually in the result pane — the alerting
# code trusts risk_level/risk_reason only.

@dataclass(frozen=True)
class MonitorProfile:
    id: str
    name: str  # shown in UI + included in Telegram summary header
    prompt: str
    summary_intro: str


# --- Kid Monitor ---------------------------------------------------------
_KID_DANGERS = [
    "打闹/肢体冲突",
    "攀爬桌椅或柜子",
    "靠近或翻越窗户",
    "玩插座/电线/充电器",
    "玩剪刀/刀具/玻璃/易碎物",
    "玩打火机/火源/蜡烛",
    "吞咽或往嘴里塞小物件",
    "倒地不动或持续哭泣",
    "屋内出现陌生人",
    "从高处坠落的姿势",
]
_KID_PROMPT = (
    "你是一个儿童房安全监控助手。看这张画面，只输出一行合法 JSON，"
    "不要写解释、不要用代码块围栏。\n"
    "\n"
    "字段与类型：\n"
    "- activity: string — 用 3 到 10 个汉字直接描述孩子正在做什么，"
    "例如\"写作业\"、\"玩积木\"、\"看平板\"、\"空房间无人\"。"
    "严禁照抄本说明里的任何字样或圆括号里的例子。\n"
    "- num_children: integer — 画面中看到的孩子数量，数字类型（不是字符串），没看到填 0。\n"
    "- risk_level: 只能是 \"none\" / \"low\" / \"medium\" / \"high\" 其中之一。\n"
    "- risk_reason: string — risk_level 为 none 时必须是空字符串 \"\"；"
    "否则从下面清单里原文选一项填入：\n"
    "  " + " | ".join(_KID_DANGERS) + "\n"
    "\n"
    "判定规则（非常重要，严格遵守）：\n"
    "- 日常写作业、学习、看书、看平板、用电脑、画画、玩玩具、休息、发呆、空房间 → risk_level 必须是 \"none\"。\n"
    "- 只有清单里所列、而且正在发生的行为才是 \"high\"。\n"
    "- 有清单里行为的明显苗头才用 \"medium\"。\n"
    "- \"low\" 基本不用。\n"
    "\n"
    "输出格式示例（仅供格式参考，不要照抄内容）：\n"
    '{"activity": "坐在桌前写作业", "num_children": 1, "risk_level": "none", "risk_reason": ""}'
)
_KID_SUMMARY = (
    "你是一位细心的家长助手。下面是过去一段时间孩子房间的活动日志摘要，"
    "请用中文写 2-4 句自然总结：孩子整体在做什么、有没有值得留意的地方。"
    "不要逐条罗列、不要重复时间戳。"
)


# --- Office Monitor ------------------------------------------------------
_OFFICE_ALERTS = [
    "人员受伤或倒地",
    "激烈争执或肢体冲突",
    "明火或浓烟",
    "非授权人员闯入",
    "设备倒塌或漏水",
    "可疑的破坏或翻找行为",
]
_OFFICE_PROMPT = (
    "你是一个办公室监控助手。看这张画面，只输出一行合法 JSON，"
    "不要写解释、不要用代码块围栏。\n"
    "\n"
    "字段与类型：\n"
    "- activity: string — 用 3 到 15 个汉字描述此刻办公室主要状态，"
    "例如\"多人围桌开会\"、\"个人专注工作\"、\"员工休息聊天\"、\"办公室无人\"。"
    "严禁照抄本说明里的任何字样或圆括号里的例子。\n"
    "- num_people: integer — 画面中可见人数，没看到填 0。\n"
    "- focus_state: 字符串，只能是 \"focused\" / \"meeting\" / \"casual\" / \"idle\" / \"unknown\" 之一。\n"
    "- risk_level: 只能是 \"none\" / \"low\" / \"medium\" / \"high\" 之一。\n"
    "- risk_reason: string — risk_level 为 none 时必须是空字符串 \"\"；"
    "否则从下面清单里原文选一项填入：\n"
    "  " + " | ".join(_OFFICE_ALERTS) + "\n"
    "\n"
    "判定规则（严格遵守）：\n"
    "- 正常工作、开会、讨论、打电话、走动、短暂休息 → risk_level 必须是 \"none\"。\n"
    "- 只有清单里所列、而且正在发生的事才是 \"high\"。\n"
    "- 冒烟但未着火、陌生人徘徊等苗头用 \"medium\"。\"low\" 基本不用。\n"
    "\n"
    "输出格式示例（仅供格式参考，不要照抄内容）：\n"
    '{"activity": "多人围桌开会", "num_people": 4, "focus_state": "meeting", "risk_level": "none", "risk_reason": ""}'
)
_OFFICE_SUMMARY = (
    "你是一位办公室运营助手。下面是过去一段时间办公室的活动日志摘要，"
    "请用中文写 2-4 句话概括：整体工作氛围、主要活动、人员流动大致情况，"
    "以及有没有需要关注的异常。不要逐条罗列、不要重复时间戳。"
)


# --- Retail Store Monitor ------------------------------------------------
_RETAIL_ALERTS = [
    "顾客进店无人接待",
    "员工忽视顾客玩手机",
    "桌面或柜台明显脏乱",
    "垃圾或污渍堆积",
    "顾客之间发生争执",
    "人员受伤或倒地",
    "非授权人员闯入",
    "明火或浓烟",
]
_RETAIL_PROMPT = (
    "你是一个门店运营监控助手。看这张画面，只输出一行合法 JSON，"
    "不要写解释、不要用代码块围栏。\n"
    "\n"
    "字段与类型：\n"
    "- activity: string — 用 3 到 15 个汉字描述当前门店场景，"
    "例如\"顾客在浏览商品\"、\"员工正在接待顾客\"、\"员工无事可做\"、\"门店无人\"、\"员工正在打扫\"。"
    "严禁照抄本说明里的任何字样或圆括号里的例子。\n"
    "- num_customers: integer — 可见顾客数，没看到填 0。\n"
    "- num_staff: integer — 可见员工数，没看到填 0。\n"
    "- staff_engagement: 字符串，只能是 \"active\" / \"passive\" / \"none\" / \"n/a\" 之一。"
    "active=员工正在积极服务；passive=员工在场但玩手机或聊天忽视顾客；"
    "none=有顾客但现场无员工；n/a=此刻画面中没有顾客。\n"
    "- cleanliness: 字符串，只能是 \"good\" / \"fair\" / \"poor\" 之一，描述桌面/柜台/地面的整洁程度。\n"
    "- risk_level: 只能是 \"none\" / \"low\" / \"medium\" / \"high\" 之一。\n"
    "- risk_reason: string — risk_level 为 none 时必须是空字符串 \"\"；"
    "否则从下面清单里原文选一项填入：\n"
    "  " + " | ".join(_RETAIL_ALERTS) + "\n"
    "\n"
    "判定规则（严格遵守）：\n"
    "- 正常接待、结账、整理货品、无人时段、日常打扫 → risk_level 必须是 \"none\"。\n"
    "- 只有清单里所列、而且正在发生的才是 \"high\"。\n"
    "- 员工消极但勉强在场、桌面略乱等苗头用 \"medium\"。\"low\" 基本不用。\n"
    "\n"
    "输出格式示例（仅供格式参考，不要照抄内容）：\n"
    '{"activity": "员工正在为顾客介绍商品", "num_customers": 1, "num_staff": 1, "staff_engagement": "active", "cleanliness": "good", "risk_level": "none", "risk_reason": ""}'
)
_RETAIL_SUMMARY = (
    "你是一位门店运营助手。下面是过去一段时间门店的活动日志摘要，"
    "请用中文写 2-4 句话概括：客流大致情况、员工服务状态、卫生状况，"
    "以及有没有值得店长处理的问题。不要逐条罗列、不要重复时间戳。"
)


# --- Home Security -------------------------------------------------------
_SECURITY_ALERTS = [
    "陌生人闯入",
    "非正常时段有人活动",
    "门窗被强行打开",
    "明火或浓烟",
    "玻璃或物品破碎",
    "人员倒地不动",
    "屋内宠物异常（剧烈挣扎或受困）",
]
_SECURITY_PROMPT = (
    "你是一个家庭安防监控助手。看这张画面，只输出一行合法 JSON，"
    "不要写解释、不要用代码块围栏。\n"
    "\n"
    "字段与类型：\n"
    "- activity: string — 用 3 到 15 个汉字描述当前场景，"
    "例如\"房间无人\"、\"家人在沙发上看电视\"、\"宠物在地上休息\"、\"门口有陌生人\"。"
    "严禁照抄本说明里的任何字样或圆括号里的例子。\n"
    "- num_people: integer — 可见人数，没看到填 0。\n"
    "- num_pets: integer — 可见宠物数，没看到填 0。\n"
    "- risk_level: 只能是 \"none\" / \"low\" / \"medium\" / \"high\" 之一。\n"
    "- risk_reason: string — risk_level 为 none 时必须是空字符串 \"\"；"
    "否则从下面清单里原文选一项填入：\n"
    "  " + " | ".join(_SECURITY_ALERTS) + "\n"
    "\n"
    "判定规则（严格遵守）：\n"
    "- 房间无人、家庭成员在正常活动、宠物正常休息或走动 → risk_level 必须是 \"none\"。\n"
    "- 只有清单里所列、而且正在发生的才是 \"high\"。\n"
    "- 有苗头（门虚掩、有人在门外张望）用 \"medium\"。\"low\" 基本不用。\n"
    "\n"
    "输出格式示例（仅供格式参考，不要照抄内容）：\n"
    '{"activity": "客厅无人安静", "num_people": 0, "num_pets": 1, "risk_level": "none", "risk_reason": ""}'
)
_SECURITY_SUMMARY = (
    "你是一位家庭安防助手。下面是过去一段时间家中监控的活动日志摘要，"
    "请用中文写 2-4 句话概括：整体是否安静平稳、有没有人员进出或异常事件。"
    "不要逐条罗列、不要重复时间戳。"
)


PROFILES: dict[str, MonitorProfile] = {
    "kid": MonitorProfile("kid", "Kid Monitor (ZH)", _KID_PROMPT, _KID_SUMMARY),
    "office": MonitorProfile("office", "Office Monitor (ZH)", _OFFICE_PROMPT, _OFFICE_SUMMARY),
    "retail": MonitorProfile("retail", "Retail Store Monitor (ZH)", _RETAIL_PROMPT, _RETAIL_SUMMARY),
    "security": MonitorProfile("security", "Home Security (ZH)", _SECURITY_PROMPT, _SECURITY_SUMMARY),
}

# Backward-compatible export — some early code referenced this by name.
MONITOR_PROMPT_ZH = PROFILES["kid"].prompt
DANGER_CATEGORIES = _KID_DANGERS


# ---------------------------------------------------------------------------
# JSON parsing (tolerant to junk around the object)
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


# Phrases that indicate the model copied the prompt template instead of
# answering. Checked case-insensitively as a substring match on `activity`.
_LEAK_MARKERS = (
    "一句话描述", "描述孩子", "学习/画画", "学习/玩耍", "填写", "字段",
    "string —", "integer —",
)


def parse_vlm_json(text: str) -> dict | None:
    if not text:
        return None
    raw: dict | None = None
    try:
        raw = json.loads(text.strip())
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                raw = json.loads(m.group(0))
            except Exception:
                raw = None
    if not isinstance(raw, dict):
        return None

    # Defensive normalisation — the small VLM sometimes echoes the schema
    # verbatim or returns "1" as a string. Drop echoes so they don't
    # contaminate the activity log, and coerce types on the happy fields.
    activity = (raw.get("activity") or "").strip()
    if any(marker in activity for marker in _LEAK_MARKERS):
        log.debug("Dropping prompt-echo activity: %r", activity)
        return None
    if not activity:
        return None
    raw["activity"] = activity

    try:
        raw["num_children"] = int(raw.get("num_children") or 0)
    except (TypeError, ValueError):
        raw["num_children"] = 0

    risk = str(raw.get("risk_level", "none")).strip().lower()
    if risk not in ("none", "low", "medium", "high"):
        risk = "none"
    raw["risk_level"] = risk

    return raw


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------
@dataclass
class ActivityEntry:
    ts: float
    activity: str
    risk_level: str
    num_children: int
    parsed_ok: bool


class ActivityLog:
    """Thread-safe rolling log of per-frame activity tags."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[ActivityEntry] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def append(self, entry: ActivityEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                del self._entries[: len(self._entries) - self._max]

    def snapshot_since(self, since_ts: float) -> list[ActivityEntry]:
        with self._lock:
            return [e for e in self._entries if e.ts >= since_ts]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def condense(entries: list[ActivityEntry]) -> list[tuple[str, float, float, int]]:
    """Collapse consecutive identical activity tags into (activity, start, end, frames)."""
    runs: list[tuple[str, float, float, int]] = []
    cur_label: str | None = None
    cur_start = 0.0
    cur_end = 0.0
    cur_n = 0
    for e in entries:
        if not e.parsed_ok:
            continue
        act = (e.activity or "").strip()
        if not act or act.lower() in ("unknown", "none", "n/a"):
            continue
        if act == cur_label:
            cur_end = e.ts
            cur_n += 1
        else:
            if cur_label is not None:
                runs.append((cur_label, cur_start, cur_end, cur_n))
            cur_label = act
            cur_start = e.ts
            cur_end = e.ts
            cur_n = 1
    if cur_label is not None:
        runs.append((cur_label, cur_start, cur_end, cur_n))
    return runs


def build_summary_input(entries: list[ActivityEntry], window_min: int) -> str | None:
    """Produce the compact text we feed to the local LLM. Returns None if no data."""
    runs = condense(entries)
    if not runs:
        return None

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M")

    lines = [f"过去约 {window_min} 分钟内共 {len(entries)} 次采样，合并连续相同活动后的时间线："]
    for label, start, end, n in runs:
        dur = max(1, int(round((end - start) / 60)))
        lines.append(f"- {fmt(start)}–{fmt(end)}  {label}  (~{dur} 分钟, {n} 帧)")

    # Top activities by frame count (rough time-share)
    counter = Counter((e.activity or "").strip() for e in entries if e.parsed_ok)
    counter.pop("", None)
    top = counter.most_common(5)
    if top:
        total = sum(counter.values())
        lines.append("\n活动占比（按帧数近似）：")
        for act, n in top:
            lines.append(f"- {act}: {100 * n / total:.0f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------
class AlertManager:
    def __init__(
        self,
        notifier,  # TelegramNotifier or None
        consecutive_threshold: int = 2,
        cooldown_sec: float = 120.0,
    ) -> None:
        self.notifier = notifier
        self.threshold = consecutive_threshold
        self.cooldown = cooldown_sec
        self._streak: dict[str, int] = {}  # category -> consecutive high count
        self._last_sent: dict[str, float] = {}  # category -> last send ts

    def observe(self, parsed: dict, pil: Image.Image | None) -> str | None:
        """Call once per frame. Returns the alert text if one was sent, else None."""
        risk = (parsed.get("risk_level") or "none").lower()
        reason = (parsed.get("risk_reason") or "").strip() or "未标注"
        activity = (parsed.get("activity") or "").strip() or "未知"

        if risk != "high":
            # Reset streaks when risk subsides
            self._streak.clear()
            return None

        cat = reason
        self._streak[cat] = self._streak.get(cat, 0) + 1
        if self._streak[cat] < self.threshold:
            return None

        now = time.time()
        if now - self._last_sent.get(cat, 0.0) < self.cooldown:
            return None

        self._last_sent[cat] = now
        text = (
            f"⚠️ *儿童房告警*\n"
            f"*类别*: {cat}\n"
            f"*当前活动*: {activity}\n"
            f"*时间*: {datetime.now().strftime('%H:%M:%S')}"
        )
        sent = False
        if self.notifier is not None:
            if pil is not None:
                buf = BytesIO()
                pil.copy().convert("RGB").save(buf, format="JPEG", quality=80)
                sent = self.notifier.send_photo(buf.getvalue(), caption=text)
            if not sent:
                sent = self.notifier.send(text)
        log.info("ALERT fired (sent=%s): %s", sent, cat)
        return text


# ---------------------------------------------------------------------------
# Summary scheduler
# ---------------------------------------------------------------------------
SummaryFn = Callable[[str], str]  # takes the compact text, returns VLM summary


class SummaryScheduler:
    """Fires periodic summaries in a background thread."""

    def __init__(
        self,
        log_store: ActivityLog,
        window_min: int,
        notifier,
        summarize_fn: SummaryFn,
        summary_intro: str,
        profile_name: str,
    ) -> None:
        self._log = log_store
        self._window_min = window_min
        self._notifier = notifier
        self._summarize = summarize_fn
        self._summary_intro = summary_intro
        self._profile_name = profile_name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_ts: float = 0.0

    @property
    def next_fire_ts(self) -> float:
        return self._next_ts

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._next_ts = time.time() + self._window_min * 60
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("SummaryScheduler started; window=%d min", self._window_min)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Sleep in small slices so stop() responds quickly
            while not self._stop.is_set() and time.time() < self._next_ts:
                self._stop.wait(timeout=1.0)
            if self._stop.is_set():
                return
            try:
                self._fire()
            except Exception:
                log.exception("Summary fire failed")
            self._next_ts = time.time() + self._window_min * 60

    def _fire(self) -> None:
        since = time.time() - self._window_min * 60
        entries = self._log.snapshot_since(since)
        compact = build_summary_input(entries, self._window_min)
        if not compact:
            log.info("Summary skipped: no parsed activity in window")
            return

        prompt = f"{self._summary_intro}\n\n{compact}"
        try:
            summary = self._summarize(prompt).strip()
        except Exception as e:
            log.error("Summarize call failed: %s", e)
            summary = f"（本地模型总结失败：{e}）\n\n{compact}"

        msg = (
            f"🕒 *{self._profile_name} — {self._window_min} 分钟活动总结*\n"
            f"{datetime.now().strftime('%H:%M:%S')}\n\n"
            f"{summary}"
        )
        if self._notifier is not None:
            self._notifier.send(msg)
        log.info("Summary fired: %s", summary[:120])


# ---------------------------------------------------------------------------
# Top-level facade used by app.py
# ---------------------------------------------------------------------------
@dataclass
class MonitorConfig:
    enabled: bool = False
    profile: MonitorProfile = field(default_factory=lambda: PROFILES["kid"])
    summary_window_min: int = 30
    alert_consecutive: int = 2
    alert_cooldown_sec: float = 120.0


@dataclass
class MonitorState:
    log: ActivityLog = field(default_factory=ActivityLog)
    alerts: AlertManager | None = None
    scheduler: SummaryScheduler | None = None
    last_alert: str = ""


def start_monitor(
    cfg: MonitorConfig,
    notifier,
    summarize_fn: SummaryFn,
) -> MonitorState:
    state = MonitorState()
    state.alerts = AlertManager(
        notifier,
        consecutive_threshold=cfg.alert_consecutive,
        cooldown_sec=cfg.alert_cooldown_sec,
    )
    state.scheduler = SummaryScheduler(
        state.log,
        cfg.summary_window_min,
        notifier,
        summarize_fn,
        summary_intro=cfg.profile.summary_intro,
        profile_name=cfg.profile.name,
    )
    state.scheduler.start()
    return state


def tick(state: MonitorState, pil: Image.Image, vlm_text: str) -> None:
    """Called after each VLM inference. Parses JSON, logs activity, maybe alerts."""
    parsed = parse_vlm_json(vlm_text) or {}
    ok = bool(parsed)
    state.log.append(
        ActivityEntry(
            ts=time.time(),
            activity=(parsed.get("activity") or "").strip(),
            risk_level=(parsed.get("risk_level") or "none").strip().lower(),
            num_children=int(parsed.get("num_children") or 0) if ok else 0,
            parsed_ok=ok,
        )
    )
    if ok and state.alerts is not None:
        fired = state.alerts.observe(parsed, pil)
        if fired:
            state.last_alert = fired
