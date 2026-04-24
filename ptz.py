"""PTZ (pan-tilt-zoom) controllers — brand-specific wrappers around stepped
motor controls, exposed behind a single PTZController protocol.

Currently implements TapoPTZ (via pytapo). Other brands (Hikvision, Reolink,
generic ONVIF) can slot in later without touching callers.
"""

from __future__ import annotations

import logging
from typing import Protocol
from urllib.parse import urlparse

log = logging.getLogger("vlm-camera.ptz")


class LimitReachedError(Exception):
    """Raised when the motor hit its physical pan or tilt limit.

    Callers should treat this as "we've arrived at the edge, stop pushing that
    direction" — NOT as a failure. Every PTZ motor has endstops; running into
    them is part of normal operation when you're calibrating or sweeping.
    """


class PTZController(Protocol):
    def pan_left(self) -> None: ...
    def pan_right(self) -> None: ...
    def tilt_up(self) -> None: ...
    def tilt_down(self) -> None: ...


class TapoPTZ:
    """Tapo PTZ cameras (TC71, C200, C210, C220, C225, C500, C520, ...).

    Wraps pytapo's moveMotorStep, which emits one fixed-size motor step per
    call (typically ~15° on TC71). Callers do multiple steps to pan further.
    Raises LimitReachedError instead of a generic Exception when the camera
    reports MOTOR_LOCKED_ROTOR (error_code -64304) so the patrol routine can
    tell "at the edge" apart from "lost connection".
    """

    def __init__(self, host: str, user: str, password: str) -> None:
        from pytapo import Tapo  # imported lazily so the dep is optional
        self._tapo = Tapo(host, user, password)
        self._host = host

    def _step(self, angle: int) -> None:
        try:
            self._tapo.moveMotorStep(angle)
        except Exception as e:
            msg = str(e)
            if "MOTOR_LOCKED_ROTOR" in msg or "-64304" in msg:
                raise LimitReachedError(msg) from e
            raise

    def pan_left(self) -> None:
        self._step(180)  # 180° = CCW = physically left

    def pan_right(self) -> None:
        self._step(0)  # 0° = CW = physically right

    def tilt_up(self) -> None:
        self._step(90)

    def tilt_down(self) -> None:
        self._step(270)

    def __repr__(self) -> str:
        return f"TapoPTZ({self._host})"


def from_camera_entry(entry: dict) -> PTZController | None:
    """Build a controller from a cameras.json entry.

    Expects `entry["ptz"]` to be a dict like {"type": "tapo"} (credentials
    are parsed from the RTSP URL in entry["url"] — no duplication needed).
    Returns None if the entry has no ptz config or the brand is unsupported.
    """
    ptz_cfg = entry.get("ptz")
    if not isinstance(ptz_cfg, dict):
        return None
    ptz_type = ptz_cfg.get("type", "").lower()
    if ptz_type != "tapo":
        log.warning("Unsupported PTZ type %r for %s", ptz_type, entry.get("name"))
        return None

    url = entry.get("url", "")
    parsed = urlparse(url)
    if not (parsed.hostname and parsed.username and parsed.password):
        log.warning("Cannot parse creds from URL for PTZ camera %s", entry.get("name"))
        return None

    try:
        return TapoPTZ(parsed.hostname, parsed.username, parsed.password)
    except Exception as e:
        log.warning("Failed to init TapoPTZ for %s: %s", entry.get("name"), e)
        return None
