"""
VLM Camera - Local real-time webcam analysis using Vision Language Models on Apple Silicon.
Uses MLX-VLM for fast local inference and Gradio for the UI.
Supports offline operation with cached HuggingFace models and local Ollama VLM models.
"""

import base64
import json
import logging
import os
import time
import threading
import urllib.request
from datetime import datetime
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlm-camera")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model = None
_processor = None
_config = None
_loaded_model_id: str | None = None
_backend: str | None = None  # "mlx" or "ollama"
_ollama_model: str | None = None
_capture: cv2.VideoCapture | None = None
_running = False
_lock = threading.Lock()
_latest_frame: np.ndarray | None = None  # shared: written by bg thread, read by UI

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# HuggingFace cache helpers
# ---------------------------------------------------------------------------

def is_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is already downloaded in the local cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    folder_name = "models--" + model_id.replace("/", "--")
    model_path = os.path.join(cache_dir, folder_name)
    if not os.path.isdir(model_path):
        return False
    # Check that snapshots dir has at least one entry (model actually downloaded)
    snapshots = os.path.join(model_path, "snapshots")
    return os.path.isdir(snapshots) and len(os.listdir(snapshots)) > 0


# ---------------------------------------------------------------------------
# Ollama VLM detection
# ---------------------------------------------------------------------------

# Model families known to support vision/image input
_OLLAMA_VLM_FAMILIES = [
    "llava", "moondream", "qwen2.5-vl", "qwen2-vl", "minicpm-v",
    "llama3.2-vision", "gemma3", "granite3.2-vision",
]


def detect_ollama_vlm_models() -> dict[str, str]:
    """Detect available VLM-capable models from local Ollama server."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        log.info("Ollama not reachable at %s", OLLAMA_BASE_URL)
        return {}

    models = {}
    for m in data.get("models", []):
        name = m["name"]
        name_lower = name.lower()
        for family in _OLLAMA_VLM_FAMILIES:
            if family in name_lower:
                label = f"Ollama: {name}"
                models[label] = f"ollama:{name}"
                break
    if models:
        log.info("Detected Ollama VLM models: %s", list(models.keys()))
    else:
        log.info("No VLM-capable models found in Ollama (text-only models are skipped)")
    return models


# ---------------------------------------------------------------------------
# Available models & presets
# ---------------------------------------------------------------------------
MODELS = {
    "SmolVLM 2B 4-bit (fastest)": "mlx-community/SmolVLM-Instruct-4bit",
    "Qwen2.5-VL 3B 4-bit (recommended)": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "Gemma 3 4B 4-bit": "mlx-community/gemma-3-4b-it-4bit",
}

# Add locally detected Ollama VLM models
_ollama_models = detect_ollama_vlm_models()
MODELS.update(_ollama_models)

PRESETS = {
    "Person Action (ZH)": "图中的人在做什么？",
    "Person Action (EN)": "What is the person doing in this image?",
    "Scene Description (ZH)": "请描述这个画面中的场景。",
    "Scene Description (EN)": "Describe the scene in this image.",
    "Object Detection (ZH)": "图中有哪些物体？请列出来。",
    "Object Detection (EN)": "What objects can you see in this image? List them.",
    "Custom": "",
}

INTERVALS = ["1.0s", "2.0s", "3.0s", "5.0s", "10.0s"]


def detect_cameras(max_index: int = 5) -> dict[str, int | str]:
    """Probe available cameras and return {label: source} dict.

    Source is an int index for local USB/built-in cameras, or a string URL
    (e.g. rtsp://...) for network cameras.
    """
    cameras: dict[str, int | str] = {}
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras[f"Camera {i} ({w}x{h})"] = i
            cap.release()
    # Preset network cameras — credentials & host pulled from env so nothing
    # sensitive lands in the repo. Set TAPO_USER / TAPO_PASS / TAPO_HOST
    # (e.g. in a local .env or shell profile) to enable these entries.
    tapo_user = os.environ.get("TAPO_USER")
    tapo_pass = os.environ.get("TAPO_PASS")
    tapo_host = os.environ.get("TAPO_HOST", "192.168.4.243")
    if tapo_user and tapo_pass:
        base = f"rtsp://{tapo_user}:{tapo_pass}@{tapo_host}:554"
        cameras[f"Tapo @ {tapo_host} (HD)"] = f"{base}/stream1"
        cameras[f"Tapo @ {tapo_host} (SD)"] = f"{base}/stream2"
    # Generic: user supplies RTSP URL at runtime (see UI)
    cameras["Network Camera (custom RTSP URL)"] = "__rtsp_url__"
    safe = {k: (v if not (isinstance(v, str) and v.startswith("rtsp://")) else "rtsp://***") for k, v in cameras.items()}
    log.info("Detected cameras: %s", safe)
    return cameras


CAMERAS = detect_cameras()

DEFAULT_RTSP_URL = "rtsp://user:pass@host:554/stream1"

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> str:
    global _model, _processor, _config, _loaded_model_id, _backend, _ollama_model
    model_id = MODELS[model_name]
    if _loaded_model_id == model_id:
        log.info("Model already loaded: %s", model_name)
        return f"Model already loaded: {model_name}"

    # --- Ollama backend ---
    if model_id.startswith("ollama:"):
        ollama_name = model_id[len("ollama:"):]
        log.info("Loading Ollama model: %s ...", ollama_name)
        try:
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/show",
                data=json.dumps({"name": ollama_name}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.error("Ollama model not available: %s", e)
            return f"Error: Ollama model '{ollama_name}' not available — {e}"

        _model = _processor = _config = None
        _backend = "ollama"
        _ollama_model = ollama_name
        _loaded_model_id = model_id
        log.info("Ollama model ready: %s", ollama_name)
        return f"Loaded: {model_name} (Ollama, fully offline)"

    # --- MLX-VLM backend ---
    try:
        cached = is_model_cached(model_id)
        if cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
            log.info("Model found in local cache, loading offline: %s", model_id)
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            log.info("Model not cached, will download from HuggingFace: %s", model_id)

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        log.info("Loading model: %s (%s) ...", model_name, model_id)
        _model, _processor = load(model_id)
        _config = load_config(model_id)
        _backend = "mlx"
        _ollama_model = None
        _loaded_model_id = model_id
        mode = "offline/cached" if cached else "downloaded"
        log.info("Model loaded successfully (%s): %s", mode, model_name)
        return f"Loaded: {model_name} ({mode})"
    except Exception as e:
        _model = _processor = _config = None
        _loaded_model_id = None
        _backend = None
        log.error("Failed to load model: %s", e, exc_info=True)
        # If offline load failed, suggest re-downloading
        if cached:
            return f"Error loading cached model (cache may be corrupted): {e}"
        return f"Error loading model: {e}"


def _resize_image(pil_image: Image.Image, max_dim: int = 768) -> Image.Image:
    """Resize image for faster inference while preserving aspect ratio."""
    img = pil_image.copy()
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img


def run_inference(pil_image: Image.Image, prompt: str, max_tokens: int) -> tuple[str, float]:
    """Run VLM inference on a single PIL image. Returns (text, elapsed_seconds)."""
    if _backend == "ollama":
        return _run_ollama_inference(pil_image, prompt, max_tokens)
    return _run_mlx_inference(pil_image, prompt, max_tokens)


def _run_mlx_inference(pil_image: Image.Image, prompt: str, max_tokens: int) -> tuple[str, float]:
    """MLX-VLM local inference. Must be called from the same thread that loaded the model."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(_processor, _config, prompt, num_images=1)
    img = _resize_image(pil_image)

    t0 = time.time()
    result = generate(
        _model, _processor, formatted,
        image=[img],
        verbose=False,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    elapsed = time.time() - t0
    return result.text.strip(), elapsed


def _run_ollama_inference(pil_image: Image.Image, prompt: str, max_tokens: int) -> tuple[str, float]:
    """Ollama API inference with image support."""
    img = _resize_image(pil_image)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = json.dumps({
        "model": _ollama_model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }).encode()

    t0 = time.time()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0
    return result.get("response", "").strip(), elapsed


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera(source: int | str = 0) -> str:
    global _capture
    if _capture is not None and _capture.isOpened():
        log.info("Camera already open")
        return "Camera already open"

    log.info("Opening camera source %r ...", source)
    # Network streams benefit from TCP transport for reliability
    if isinstance(source, str) and source.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        _capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        _capture = cv2.VideoCapture(source)
    opened = _capture.isOpened()
    log.info("cv2.VideoCapture(%r).isOpened() = %s", source, opened)

    if not opened:
        _capture = None
        log.error("Failed to open camera %r", source)
        return f"Failed to open camera {source}"

    # Try to grab a test frame
    ret, frame = _capture.read()
    log.info("Test frame read: ret=%s, shape=%s", ret, frame.shape if ret else "N/A")
    if not ret:
        log.warning("Camera opened but cannot read frames")

    return f"Camera opened: {source}"


def close_camera():
    global _capture
    with _lock:
        if _capture is not None:
            log.info("Closing camera")
            _capture.release()
            _capture = None


def grab_frame() -> Image.Image | None:
    if _capture is None or not _capture.isOpened():
        return None
    ret, frame = _capture.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Main capture loop (runs in a background thread)
# ---------------------------------------------------------------------------

def capture_loop(
    model_name: str,
    prompt: str,
    max_tokens: int,
    interval: float,
    results_state: list,
    frame_counter: list,
    stop_event: threading.Event,
):
    """Background thread: sole owner of camera reads AND of MLX state. Grabs frame -> inference -> append result."""
    global _latest_frame, _running
    log.info("Capture loop started")

    try:
        # MLX arrays are thread-bound (the default GPU stream is thread-local).
        # Load the model in THIS thread so every subsequent inference call stays
        # on the same stream. Loading from another thread raises
        # "There is no Stream(gpu, 0) in current thread."
        load_msg = load_model(model_name)
        log.info("capture_loop: %s", load_msg)
        if _model is None and _backend != "ollama":
            log.error("Model load failed in worker thread, aborting capture loop")
            return

        _run_capture(prompt, max_tokens, interval, results_state, frame_counter, stop_event)
    finally:
        _running = False
        log.info("Capture loop exited")


def _run_capture(prompt, max_tokens, interval, results_state, frame_counter, stop_event):
    global _latest_frame
    _no_frame_count = 0
    while not stop_event.is_set():
        # --- read camera (only this thread touches _capture.read) ---
        with _lock:
            pil = grab_frame()
            if pil is not None:
                _latest_frame = np.array(pil)

        if pil is None:
            _no_frame_count += 1
            if _no_frame_count % 10 == 1:
                log.warning("No frame from camera (attempt %d)", _no_frame_count)
            time.sleep(0.5)
            continue

        _no_frame_count = 0
        frame_counter[0] += 1
        frame_no = frame_counter[0]
        log.info("Frame %d captured, size=%s, running inference ...", frame_no, pil.size)

        try:
            text, elapsed = run_inference(pil, prompt, max_tokens)
            log.info("Frame %d inference done in %.2fs: %s", frame_no, elapsed, text[:80])
        except Exception as e:
            log.error("Frame %d inference error: %s", frame_no, e, exc_info=True)
            text = f"[Error] {e}"
            elapsed = 0.0

        now = datetime.now().strftime("%H:%M:%S")
        if _loaded_model_id and _loaded_model_id.startswith("ollama:"):
            model_label = _loaded_model_id[len("ollama:"):]
        else:
            model_label = _loaded_model_id.split("/")[-1] if _loaded_model_id else "?"
        entry = {
            "frame": frame_no,
            "time": now,
            "infer": f"{elapsed:.2f}s",
            "model": model_label,
            "text": text,
        }
        results_state.append(entry)

        # Wait for the remaining interval, but keep updating preview frames
        remaining = interval - elapsed
        deadline = time.time() + remaining
        while not stop_event.is_set() and time.time() < deadline:
            with _lock:
                pil_preview = grab_frame()
                if pil_preview is not None:
                    _latest_frame = np.array(pil_preview)
            stop_event.wait(timeout=0.1)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

_stop_event: threading.Event | None = None
_bg_thread: threading.Thread | None = None
_results: list = []
_frame_counter: list = [0]


def on_start(model_name, camera_name, rtsp_url, preset_name, custom_prompt, interval_str, max_tokens):
    global _running, _stop_event, _bg_thread, _results, _frame_counter

    log.info("=== START clicked === model=%s camera=%s preset=%s interval=%s max_tokens=%s",
             model_name, camera_name, preset_name, interval_str, max_tokens)

    if _running:
        log.info("Already running, ignoring")
        return "Already running", gr.update(), format_results(_results)

    # Open camera — resolve sentinel for network stream into the user-provided URL
    source = CAMERAS.get(camera_name, 0)
    if source == "__rtsp_url__":
        source = (rtsp_url or "").strip()
        if not source:
            return "Error: please fill in the RTSP URL for the network camera.", gr.update(), ""
    cam_msg = open_camera(source)
    if _capture is None:
        log.error("Camera not opened, aborting start")
        return cam_msg, gr.update(), ""

    # Determine prompt
    prompt = custom_prompt.strip()
    if not prompt:
        prompt = PRESETS.get(preset_name, "Describe this image.")
    if not prompt:
        prompt = "Describe this image."

    interval = float(interval_str.replace("s", ""))
    log.info("Starting capture loop: prompt=%r interval=%.1fs max_tokens=%d",
             prompt, interval, int(max_tokens))

    _results = []
    _frame_counter = [0]
    _stop_event = threading.Event()
    _running = True

    _bg_thread = threading.Thread(
        target=capture_loop,
        args=(model_name, prompt, int(max_tokens), interval, _results, _frame_counter, _stop_event),
        daemon=True,
    )
    _bg_thread.start()

    return f"{cam_msg} | Loading model & running …", gr.update(), ""


def on_stop():
    global _running, _stop_event, _latest_frame, _bg_thread
    global _model, _processor, _config, _loaded_model_id, _backend, _ollama_model

    if _stop_event is not None:
        _stop_event.set()
    _running = False

    # Wait for the worker thread to exit before tearing MLX state down.
    # Otherwise new Start() spawns a second thread while the old one still
    # holds cv2 + MLX arrays.
    if _bg_thread is not None and _bg_thread.is_alive():
        _bg_thread.join(timeout=5.0)
        if _bg_thread.is_alive():
            log.warning("Worker thread did not exit in 5s")
    _bg_thread = None

    close_camera()
    _latest_frame = None

    # MLX arrays are bound to the stream of the thread that created them.
    # That thread is gone, so the cached model is no longer usable from a
    # fresh worker thread. Drop the cache so the next Start() reloads.
    _model = None
    _processor = None
    _config = None
    _loaded_model_id = None
    _backend = None
    _ollama_model = None

    return "Stopped"


def on_preset_change(preset_name):
    return PRESETS.get(preset_name, "")


def poll_updates():
    """Called by a Gradio timer to refresh camera preview + results (reads shared frame, never touches camera)."""
    img_out = _latest_frame  # written by bg thread, safe to read (numpy array is immutable ref swap)
    results_md = format_results(_results)
    status = f"Running | Frames: {_frame_counter[0]}" if _running else "Stopped"
    return img_out, results_md, status


def format_results(results: list) -> str:
    if not results:
        return "*Waiting for frames …*"
    lines = []
    for r in reversed(results[-50:]):  # Show latest 50, newest first
        header = f"**frame={r['frame']}** | {r['time']} | infer={r['infer']} | model={r['model']}"
        lines.append(f"{header}\n\n{r['text']}\n\n---\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="VLM Camera") as app:
        gr.Markdown("# VLM Camera — Local Real-Time Webcam Analysis", elem_classes="header")

        with gr.Row():
            # ── Left panel: config + camera preview ──
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[1],  # default Qwen2.5-VL
                    label="MODEL",
                )
                # Default to first local webcam if present, else the RTSP option
                _cam_default = next(
                    (k for k, v in CAMERAS.items() if isinstance(v, int)),
                    list(CAMERAS.keys())[0],
                )
                camera_dd = gr.Dropdown(
                    choices=list(CAMERAS.keys()),
                    value=_cam_default,
                    label="CAMERA",
                )
                rtsp_url_txt = gr.Textbox(
                    value=DEFAULT_RTSP_URL,
                    label="RTSP URL (for Network Camera)",
                    placeholder="rtsp://user:pass@host:554/stream1",
                    visible=False,
                )

                # Camera preview — moved up so it's visible near the source selector
                camera_img = gr.Image(label="Camera Preview", height=360)

                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Person Action (ZH)",
                        label="PRESET",
                    )
                    interval_dd = gr.Dropdown(
                        choices=INTERVALS, value="2.0s", label="INTERVAL"
                    )

                with gr.Row():
                    max_tokens_num = gr.Number(value=200, label="MAX TOKENS", precision=0)
                    backend_txt = gr.Textbox(
                        value="MLX-VLM (offline)" if all(
                            is_model_cached(v) for v in MODELS.values() if not v.startswith("ollama:")
                        ) else "MLX-VLM",
                        label="BACKEND", interactive=False,
                    )

                prompt_txt = gr.Textbox(
                    value=PRESETS["Person Action (ZH)"],
                    label="PROMPT",
                    lines=2,
                    placeholder="Enter custom prompt …",
                )

                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

                status_txt = gr.Textbox(label="STATUS", interactive=False, value="Ready")

                gr.Markdown(
                    "*MLX models auto-cache locally after first download (offline thereafter). "
                    "Ollama VLM models are auto-detected if available.*"
                )

            # ── Right panel: results ──
            with gr.Column(scale=1):
                results_md = gr.Markdown(
                    "*Press **Start** to begin analysis …*",
                    elem_classes="result-box",
                )

        # ── Events ──
        preset_dd.change(on_preset_change, inputs=[preset_dd], outputs=[prompt_txt])

        # Show the RTSP URL box only when the network camera option is picked
        def _toggle_rtsp(cam_name: str):
            return gr.update(visible=CAMERAS.get(cam_name) == "__rtsp_url__")

        camera_dd.change(_toggle_rtsp, inputs=[camera_dd], outputs=[rtsp_url_txt])

        start_btn.click(
            on_start,
            inputs=[model_dd, camera_dd, rtsp_url_txt, preset_dd, prompt_txt, interval_dd, max_tokens_num],
            outputs=[status_txt, camera_img, results_md],
        )
        stop_btn.click(on_stop, outputs=[status_txt])

        # Polling timer: refresh camera + results every 0.5s
        timer = gr.Timer(0.5)
        timer.tick(poll_updates, outputs=[camera_img, results_md, status_txt])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="green"),
        css="""
        .result-box { max-height: 80vh; overflow-y: auto; }
        .header { text-align: center; margin-bottom: 0.5em; }
        """,
    )
