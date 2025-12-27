import sys
import time
import json
import base64
import io
import math
import os
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
import pyautogui
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from PySide6 import QtCore, QtGui, QtWidgets

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:235b-cloud"
DEFAULT_SCALE = 0.60
DEFAULT_INTERVAL = 0.90
DEFAULT_TEMP = 0.20

pyautogui.FAILSAFE = True

MOUSE_ACTIONS: Set[str] = {
    "move", "move_rel", "click", "double_click", "right_click",
    "mouse_down", "mouse_up", "drag",
}

# -------------------- App data paths (OS-correct, PyInstaller-friendly) --------------------
def app_data_dir() -> str:
    # Use Qt's standard app data location:
    # Windows: %APPDATA%/BlackForge
    # macOS:   ~/Library/Application Support/BlackForge
    # Linux:   ~/.local/share/BlackForge (or XDG)
    base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".blackforge")
    d = os.path.join(base, "BlackForge")
    os.makedirs(d, exist_ok=True)
    return d

APP_DIR = app_data_dir()
APP_SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
APP_LOG_PATH = os.path.join(APP_DIR, "blackforge.log")

# -------------------- Logging --------------------
LOG = logging.getLogger("blackforge")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = RotatingFileHandler(APP_LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    LOG.addHandler(h)

# -------------------- Prompts --------------------
SYSTEM_PROMPT_AGENT = r"""You control my computer.

CRITICAL OUTPUT RULE (NO EXCEPTIONS):
- Your entire reply MUST be exactly ONE JSON object.
- Output ONLY the JSON. No extra words. No markdown. No backticks. No commentary.

POLICY (IMPORTANT):
- Prefer keyboard navigation FIRST whenever possible:
  Tab / Shift+Tab / Enter / Space,
  Ctrl+L (address bar), Ctrl+T (new tab), Ctrl+W (close tab),
  Alt+Tab, arrow keys, etc.
- The mouse is unreliable. Do NOT use any mouse action unless there is truly no safe keyboard alternative.
- Use the mouse ONLY when keyboard cannot reliably do it.
- If screenshots are NOT provided, avoid mouse actions unless absolutely necessary.

AVAILABLE ACTIONS (choose exactly ONE per step):

Mouse:
{"action":"move","x":int,"y":int}
{"action":"move_rel","dx":int,"dy":int}
{"action":"click","x":int,"y":int}
{"action":"double_click","x":int,"y":int}
{"action":"right_click","x":int,"y":int}
{"action":"mouse_down","x":int,"y":int,"button":"left|right|middle"}
{"action":"mouse_up","x":int,"y":int,"button":"left|right|middle"}
{"action":"drag","x1":int,"y1":int,"x2":int,"y2":int,"button":"left|right|middle","duration":number}

Typing:
{"action":"type","text":"..."}

Keys (single press):
{"action":"key","key":"<keyname>"}

Key hold/release:
{"action":"key_down","key":"<keyname>"}
{"action":"key_up","key":"<keyname>"}
{"action":"hold","keys":["shift","tab"],"seconds":0.2}
{"action":"press","keys":["tab","tab","enter"]}

Hotkeys:
{"action":"hotkey","keys":["ctrl","l"]}
{"action":"hotkey","keys":["ctrl","t"]}
{"action":"hotkey","keys":["ctrl","w"]}
{"action":"hotkey","keys":["alt","tab"]}
{"action":"hotkey","keys":["ctrl","shift","esc"]}

Scroll:
{"action":"scroll","scroll":int}

Wait:
{"action":"wait","seconds":number}

Notify the user:
{"action":"notify","text":"..."}

Done:
{"action":"done"}

COORDINATES AND SCALE:
- If screenshots are provided, x,y are in the screenshot coordinate system (scaled).
- The user message includes SCALE each step.
- You MUST output x,y matching the screenshot coordinate system.

SAFETY:
- One action per step.
- Prefer small safe steps.
- If uncertain: {"action":"wait","seconds":1}
"""

SYSTEM_PROMPT_REPAIR = """You are a strict JSON fixer.
Given some model output text, extract or produce exactly ONE valid JSON object with an "action" field.
Output ONLY the JSON object. If no valid action can be found, output: {"action":"wait","seconds":1}
"""

# -------------------- Key helpers --------------------
KEY_MAP = {
    "up": "up", "down": "down", "left": "left", "right": "right",
    "enter": "enter", "return": "enter",
    "esc": "esc", "escape": "esc",
    "tab": "tab",
    "backspace": "backspace",
    "delete": "delete", "del": "delete",
    "space": "space",
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "win", "windows": "win", "cmd": "win", "super": "win",
}
VALID_PYAUTOGUI_KEYS = set(getattr(pyautogui, "KEYBOARD_KEYS", []))


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(round(v))
        return int(round(float(str(v).strip())))
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, float):
            return v
        if isinstance(v, int):
            return float(v)
        return float(str(v).strip())
    except Exception:
        return default


def b64_png_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _map_key(k: str) -> str:
    k = (k or "").strip().lower()
    k = KEY_MAP.get(k, k)
    if k == "win":
        return "winleft"
    return k


def _is_valid_key(k: str) -> bool:
    k0 = (k or "").strip().lower()
    k0 = KEY_MAP.get(k0, k0)
    if len(k0) == 1 and (k0.isalnum() or k0 in [" ", ".", ",", "/", "\\", ";", "'", "[", "]", "-", "=", "`"]):
        return True
    if k0 == "win":
        return True
    return (k0 in VALID_PYAUTOGUI_KEYS) or (_map_key(k0) in VALID_PYAUTOGUI_KEYS)


def looks_vision_capable(model_name: str) -> bool:
    m = (model_name or "").lower()
    return ("-vl" in m) or ("vision" in m) or ("multimodal" in m) or ("mm" in m)


# -------------------- JSON extraction --------------------
def find_action_json_objects(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    objs: List[Dict[str, Any]] = []
    start = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict) and "action" in obj:
                            objs.append(obj)
                    except Exception:
                        pass
                    start = None

    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "action" in obj:
            objs.append(obj)
    except Exception:
        pass

    return objs


def extract_action(content: str) -> Optional[Dict[str, Any]]:
    cands = find_action_json_objects(content)
    return cands[-1] if cands else None


# -------------------- Screenshot helpers --------------------
def add_soft_grid(img: Image.Image, minor: int = 80, major: int = 320) -> Image.Image:
    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    w, h = base.size
    minor_color = (255, 255, 255, 18)
    major_color = (255, 255, 255, 42)

    minor = max(12, int(minor))
    major = max(0, int(major))

    for x in range(0, w, minor):
        col = major_color if (major > 0 and x % major == 0) else minor_color
        d.line([(x, 0), (x, h)], fill=col, width=1)
    for y in range(0, h, minor):
        col = major_color if (major > 0 and y % major == 0) else minor_color
        d.line([(0, y), (w, y)], fill=col, width=1)

    return Image.alpha_composite(base, overlay)


def draw_cursor_marker(img: Image.Image, x: int, y: int):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    d = ImageDraw.Draw(img)
    d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 80, 80, 230), width=2)
    d.line((x - 14, y, x + 14, y), fill=(255, 80, 80, 180), width=1)
    d.line((x, y - 14, x, y + 14), fill=(255, 80, 80, 180), width=1)


# -------------------- Minimal sound --------------------
class SoundBank(QtCore.QObject):
    def play_action(self):
        QtWidgets.QApplication.beep()

    def play_done(self):
        QtWidgets.QApplication.beep()


# -------------------- Ollama client --------------------
class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def set_base_url(self, base_url: str):
        self.base_url = (base_url or "").strip().rstrip("/") or DEFAULT_BASE_URL

    def list_models(self, timeout: int = 8) -> List[str]:
        url = f"{self.base_url}/api/tags"
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return []
            data = r.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return sorted(set(models))
        except Exception as e:
            LOG.warning("list_models failed: %s", e)
            return []

    def chat_stream(self, model: str, messages: List[Dict[str, Any]], temperature: float, timeout: int = 300):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": float(temperature)},
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:1200]}")
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if "error" in obj:
                        raise RuntimeError(str(obj["error"]))
                    msg = obj.get("message") or {}
                    yield (msg.get("content") or ""), bool(obj.get("done"))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error talking to Ollama at {self.base_url}: {e}") from e


# -------------------- Settings --------------------
@dataclass
class AppSettings:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    goal: str = "Do the next sensible step in the currently focused window."
    interval: float = DEFAULT_INTERVAL
    scale: float = DEFAULT_SCALE
    temperature: float = DEFAULT_TEMP

    vision: bool = True

    auto_execute: bool = False
    max_steps: int = 0

    block_alt_f4: bool = False
    block_win_r: bool = False

    allow_mouse: bool = True

    # Strict keyboard-first enforcement:
    prefer_keyboard_strict: bool = True     # auto-replan if mouse chosen
    mouse_retry_limit: int = 2              # how many forced keyboard replans
    always_confirm_mouse: bool = True       # require approval for mouse even in auto

    grid_minor: int = 80
    grid_major: int = 320
    capture_stamp: bool = True
    show_cursor_marker: bool = True

    mouse_move_min: float = 0.05
    mouse_move_per_1000px: float = 0.16
    mouse_move_max: float = 0.35
    click_settle_delay: float = 0.04
    mouse_tween: str = "easeInOutQuad"

    tint_r: int = 90
    tint_g: int = 140
    tint_b: int = 255
    tint_a: int = 55
    border_a: int = 70

    blur_radius: float = 14.0
    blur_fps: float = 30.0
    blur_downscale: float = 0.45


def _sanitize_settings(st: AppSettings) -> AppSettings:
    st.scale = float(max(0.25, min(1.0, float(st.scale))))
    st.interval = float(max(0.1, min(10.0, float(st.interval))))
    st.temperature = float(max(0.0, min(2.0, float(st.temperature))))
    st.max_steps = int(max(0, min(20000, int(st.max_steps))))
    st.blur_radius = float(max(0.0, min(40.0, float(st.blur_radius))))
    st.blur_fps = float(max(1.0, min(60.0, float(st.blur_fps))))
    st.blur_downscale = float(max(0.2, min(1.0, float(st.blur_downscale))))

    st.grid_minor = int(max(20, min(240, int(st.grid_minor))))
    st.grid_major = int(max(80, min(900, int(st.grid_major))))

    st.mouse_move_min = float(max(0.0, min(1.5, float(st.mouse_move_min))))
    st.mouse_move_per_1000px = float(max(0.0, min(2.0, float(st.mouse_move_per_1000px))))
    st.mouse_move_max = float(max(0.0, min(3.0, float(st.mouse_move_max))))
    st.click_settle_delay = float(max(0.0, min(1.0, float(st.click_settle_delay))))

    st.mouse_retry_limit = int(max(0, min(5, int(st.mouse_retry_limit))))
    st.prefer_keyboard_strict = bool(st.prefer_keyboard_strict)
    st.always_confirm_mouse = bool(st.always_confirm_mouse)

    if looks_vision_capable(st.model):
        st.vision = True
    return st


def load_settings() -> AppSettings:
    try:
        if os.path.exists(APP_SETTINGS_PATH):
            with open(APP_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = AppSettings()
            for k, v in data.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            return _sanitize_settings(st)
    except Exception as e:
        LOG.warning("load_settings failed: %s", e)
    return _sanitize_settings(AppSettings())


def save_settings(st: AppSettings) -> bool:
    try:
        st = _sanitize_settings(st)
        tmp = json.dumps(asdict(st), indent=2)
        with open(APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
            f.write(tmp)
        return True
    except Exception as e:
        LOG.warning("save_settings failed: %s", e)
        return False


# -------------------- Screenshot bridge --------------------
class ScreenshotBridge(QtCore.QObject):
    request = QtCore.Signal(float)
    ready = QtCore.Signal(str)


# -------------------- Action validation --------------------
ALLOWED_ACTIONS = {
    "move", "move_rel", "click", "double_click", "right_click", "mouse_down", "mouse_up", "drag",
    "type", "key", "key_down", "key_up", "hold", "press", "hotkey",
    "scroll", "wait", "notify", "done",
}


def normalize_action(act: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(act, dict):
        return {"action": "wait", "seconds": 1}
    a = str(act.get("action", "")).strip().lower()
    if a not in ALLOWED_ACTIONS:
        return {"action": "wait", "seconds": 1}
    act = dict(act)
    act["action"] = a
    return act


# -------------------- Agent Worker --------------------
class AgentWorker(QtCore.QThread):
    state = QtCore.Signal(str)
    stats = QtCore.Signal(int, str)
    last_response = QtCore.Signal(str)
    error = QtCore.Signal(str)
    notify_user = QtCore.Signal(str)
    action_sound = QtCore.Signal()
    done_sound = QtCore.Signal()

    proposed_action = QtCore.Signal(dict)
    waiting_approval = QtCore.Signal(bool)

    def __init__(self, bridge: ScreenshotBridge):
        super().__init__()
        self.bridge = bridge
        self.client = OllamaClient(DEFAULT_BASE_URL)
        self.settings = AppSettings()

        self._stop = False

        self._pending_shot: Optional[str] = None
        self._shot_mutex = QtCore.QMutex()
        self._shot_wait = QtCore.QWaitCondition()
        self.bridge.ready.connect(self._on_shot_ready, QtCore.Qt.QueuedConnection)

        self._messages: List[Dict[str, Any]] = []
        self._messages_mutex = QtCore.QMutex()

        self._last_step_image_retry = -1

        self._approval_mutex = QtCore.QMutex()
        self._approval_wait = QtCore.QWaitCondition()
        self._approved: Optional[bool] = None

    def configure(self, st: AppSettings):
        self.settings = _sanitize_settings(st)
        self.client.set_base_url(self.settings.base_url)
        with QtCore.QMutexLocker(self._messages_mutex):
            self._messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]

    def stop(self):
        self._stop = True
        with QtCore.QMutexLocker(self._shot_mutex):
            self._shot_wait.wakeAll()
        with QtCore.QMutexLocker(self._approval_mutex):
            self._approved = False
            self._approval_wait.wakeAll()

    def approve(self, ok: bool):
        with QtCore.QMutexLocker(self._approval_mutex):
            self._approved = bool(ok)
            self._approval_wait.wakeAll()

    def _on_shot_ready(self, b64: str):
        with QtCore.QMutexLocker(self._shot_mutex):
            self._pending_shot = b64
            self._shot_wait.wakeAll()

    def _get_screenshot_b64_blocking(self) -> str:
        with QtCore.QMutexLocker(self._shot_mutex):
            self._pending_shot = None
        self.bridge.request.emit(self.settings.scale)

        self._shot_mutex.lock()
        try:
            waited = 0
            while self._pending_shot is None and not self._stop and waited < 5500:
                self._shot_wait.wait(self._shot_mutex, 250)
                waited += 250
            return self._pending_shot or ""
        finally:
            self._shot_mutex.unlock()

    def _merge_stream_text(self, accum: str, incoming: str) -> str:
        if not incoming:
            return accum
        if incoming.startswith(accum) and len(incoming) >= len(accum):
            return incoming
        return accum + incoming

    def _scaled_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        sw, sh = pyautogui.size()
        s = max(1e-9, float(self.settings.scale))
        sx = int(x / s)
        sy = int(y / s)
        return clamp(sx, 0, sw - 1), clamp(sy, 0, sh - 1)

    def _get_tween(self):
        name = (self.settings.mouse_tween or "").strip()
        return getattr(pyautogui, name, pyautogui.easeInOutQuad)

    def _move_to_smooth(self, sx: int, sy: int):
        cx, cy = pyautogui.position()
        dist = math.hypot(sx - cx, sy - cy)
        dur = float(self.settings.mouse_move_min) + (dist / 1000.0) * float(self.settings.mouse_move_per_1000px)
        dur = max(0.0, min(float(self.settings.mouse_move_max), dur))
        pyautogui.moveTo(sx, sy, duration=dur, tween=self._get_tween())

    def _repair_json_once(self, bad_text: str) -> Optional[Dict[str, Any]]:
        try:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT_REPAIR},
                {"role": "user", "content": bad_text[-12000:]},
            ]
            content_full = ""
            for chunk, done in self.client.chat_stream(self.settings.model, msgs, temperature=0.0, timeout=60):
                if self._stop:
                    return None
                if chunk:
                    content_full = self._merge_stream_text(content_full, chunk)
                if done:
                    break
            content_full = (content_full or "").strip()
            act = extract_action(content_full)
            return normalize_action(act) if act else None
        except Exception as e:
            LOG.warning("repair_json failed: %s", e)
            return None

    def _manual_approval_blocking(self, act: Dict[str, Any]) -> bool:
        self.proposed_action.emit(act)
        self.waiting_approval.emit(True)
        with QtCore.QMutexLocker(self._approval_mutex):
            self._approved = None

        self._approval_mutex.lock()
        try:
            while self._approved is None and not self._stop:
                self._approval_wait.wait(self._approval_mutex, 250)
            return bool(self._approved) and not self._stop
        finally:
            self._approval_mutex.unlock()
            self.waiting_approval.emit(False)

    def _replan_without_mouse(self, step: int, why: str) -> Optional[Dict[str, Any]]:
        nudge = {
            "role": "user",
            "content": (
                f"REPLAN REQUIRED (Step {step}): You chose a mouse action but mouse is unreliable.\n"
                f"Reason: {why}\n"
                f"RULE: For this step, you MUST NOT use any mouse action. "
                f"Choose a keyboard-only action (type/key/hotkey/press/hold/scroll/wait/notify/done).\n"
                f"Reply with ONLY one JSON action object.\n"
            ),
        }

        with QtCore.QMutexLocker(self._messages_mutex):
            self._messages.append(nudge)
            messages_snapshot = list(self._messages)

        content_full = ""
        for chunk, done in self.client.chat_stream(
            self.settings.model,
            messages_snapshot,
            float(self.settings.temperature),
            timeout=180
        ):
            if self._stop:
                return None
            if chunk:
                content_full = self._merge_stream_text(content_full, chunk)
                self.last_response.emit(content_full[-6000:])
            if done:
                break

        content_full = (content_full or "").strip()
        with QtCore.QMutexLocker(self._messages_mutex):
            self._messages.append({"role": "assistant", "content": content_full})

        act = extract_action(content_full)
        return normalize_action(act) if act else None

    def _execute_action(self, act: Dict[str, Any]) -> str:
        act = normalize_action(act)
        a = act["action"]

        if a == "done":
            return "done"

        if a == "notify":
            msg = str(act.get("text", "")).strip()
            if msg:
                self.notify_user.emit(msg)
            return "notify"

        if a == "wait":
            secs = safe_float(act.get("seconds", 1.0), 1.0)
            time.sleep(max(0.0, min(10.0, secs)))
            return f"wait({secs:.2f})"

        if not self.settings.auto_execute:
            return f"proposed({json.dumps(act, ensure_ascii=False)})"

        # --- Keyboard ---
        if a == "type":
            text = str(act.get("text", ""))
            if len(text) > 4000:
                text = text[:4000]
            pyautogui.write(text, interval=0.02)
            return f"type({len(text)}c)"

        if a in ("key", "key_down", "key_up"):
            key_raw = str(act.get("key", "")).strip()
            if not _is_valid_key(key_raw):
                return f"unknown_key({key_raw})"
            mapped = _map_key(key_raw)
            if a == "key":
                pyautogui.press(mapped)
                return f"key({key_raw})"
            if a == "key_down":
                pyautogui.keyDown(mapped)
                return f"keyDown({key_raw})"
            pyautogui.keyUp(mapped)
            return f"keyUp({key_raw})"

        if a == "hotkey":
            keys_raw = act.get("keys", [])
            if not isinstance(keys_raw, list) or not keys_raw:
                return "unknown_hotkey(empty)"
            keys_clean = [str(k).strip().lower() for k in keys_raw if str(k).strip()][:8]

            if self.settings.block_alt_f4 and keys_clean == ["alt", "f4"]:
                self.notify_user.emit("Blocked Alt+F4 (toggle off in settings)")
                return "blocked(alt+f4)"
            if self.settings.block_win_r and keys_clean == ["win", "r"]:
                self.notify_user.emit("Blocked Win+R (toggle off in settings)")
                return "blocked(win+r)"

            if any(not _is_valid_key(k) for k in keys_clean):
                return f"unknown_hotkey({keys_clean})"

            mapped = [_map_key(k) for k in keys_clean]
            pyautogui.hotkey(*mapped)
            return f"hotkey({'+'.join(keys_clean)})"

        if a == "press":
            keys_raw = act.get("keys", [])
            if not isinstance(keys_raw, list) or not keys_raw:
                return "press(empty)"
            keys_clean = [str(k).strip().lower() for k in keys_raw if str(k).strip()][:25]
            for k in keys_clean:
                if not _is_valid_key(k):
                    return f"press(invalid:{k})"
            for k in keys_clean:
                pyautogui.press(_map_key(k))
                time.sleep(0.02)
            return f"press({len(keys_clean)})"

        if a == "hold":
            keys_raw = act.get("keys", [])
            secs = max(0.01, min(5.0, safe_float(act.get("seconds", 0.25), 0.25)))
            if not isinstance(keys_raw, list) or not keys_raw:
                return "hold(empty)"
            keys_clean = [str(k).strip().lower() for k in keys_raw if str(k).strip()][:8]
            for k in keys_clean:
                if not _is_valid_key(k):
                    return f"hold(invalid:{k})"
            mapped = [_map_key(k) for k in keys_clean]
            for kk in mapped:
                pyautogui.keyDown(kk)
            time.sleep(secs)
            for kk in reversed(mapped):
                pyautogui.keyUp(kk)
            return f"hold({'+'.join(keys_clean)},{secs:.2f}s)"

        if a == "scroll":
            amt = clamp(safe_int(act.get("scroll", 0), 0), -5000, 5000)
            pyautogui.scroll(amt)
            return f"scroll({amt})"

        # --- Mouse fallback ---
        if not bool(self.settings.allow_mouse):
            self.notify_user.emit("Mouse is disabled in settings.")
            return "blocked(mouse_disabled)"

        if a in ("move", "click", "double_click", "right_click"):
            x = safe_int(act.get("x", 0))
            y = safe_int(act.get("y", 0))
            sx, sy = self._scaled_to_screen(x, y)

            if a == "move":
                self._move_to_smooth(sx, sy)
                return f"moveTo({sx},{sy})"

            self._move_to_smooth(sx, sy)
            time.sleep(float(self.settings.click_settle_delay))

            if a == "double_click":
                pyautogui.doubleClick(sx, sy)
                return f"doubleClick({sx},{sy})"
            if a == "right_click":
                pyautogui.click(sx, sy, button="right")
                return f"rightClick({sx},{sy})"
            pyautogui.click(sx, sy)
            return f"click({sx},{sy})"

        if a == "move_rel":
            dx = clamp(safe_int(act.get("dx", 0)), -2000, 2000)
            dy = clamp(safe_int(act.get("dy", 0)), -2000, 2000)
            cx, cy = pyautogui.position()
            self._move_to_smooth(cx + dx, cy + dy)
            return f"moveRel({dx},{dy})"

        if a in ("mouse_down", "mouse_up"):
            button = str(act.get("button", "left")).lower().strip()
            if button not in ("left", "right", "middle"):
                button = "left"
            x = act.get("x", None)
            y = act.get("y", None)
            if x is not None and y is not None:
                sx, sy = self._scaled_to_screen(safe_int(x), safe_int(y))
                self._move_to_smooth(sx, sy)
                time.sleep(float(self.settings.click_settle_delay))
            if a == "mouse_down":
                pyautogui.mouseDown(button=button)
                return f"mouseDown({button})"
            pyautogui.mouseUp(button=button)
            return f"mouseUp({button})"

        if a == "drag":
            button = str(act.get("button", "left")).lower().strip()
            if button not in ("left", "right", "middle"):
                button = "left"
            x1 = safe_int(act.get("x1", 0))
            y1 = safe_int(act.get("y1", 0))
            x2 = safe_int(act.get("x2", 0))
            y2 = safe_int(act.get("y2", 0))
            dur = max(0.0, min(2.0, safe_float(act.get("duration", 0.2), 0.2)))

            sx1, sy1 = self._scaled_to_screen(x1, y1)
            sx2, sy2 = self._scaled_to_screen(x2, y2)

            self._move_to_smooth(sx1, sy1)
            time.sleep(float(self.settings.click_settle_delay))
            pyautogui.dragTo(sx2, sy2, duration=dur, button=button)
            return f"drag({sx1},{sy1}->{sx2},{sy2},{button})"

        return f"unknown_action({a})"

    def run(self):
        self._stop = False
        step = 0

        while not self._stop:
            step += 1
            if int(self.settings.max_steps) > 0 and step > int(self.settings.max_steps):
                self.state.emit("Idle")
                break

            try:
                self.state.emit("Thinking")

                sw, sh = pyautogui.size()
                mx, my = pyautogui.position()
                mx_s = int(mx * float(self.settings.scale))
                my_s = int(my * float(self.settings.scale))

                exec_mode = "AUTO" if self.settings.auto_execute else "MANUAL (needs approval)"

                user_msg: Dict[str, Any] = {
                    "role": "user",
                    "content": (
                        f"GOAL: {self.settings.goal}\n"
                        f"Step {step}: Choose ONE action.\n"
                        f"EXECUTION MODE: {exec_mode}\n"
                        f"SCALE: {self.settings.scale}\n"
                        f"SCREEN: {sw}x{sh}\n"
                        f"SCALED_CURSOR: {mx_s},{my_s}\n"
                        f"MOUSE_ALLOWED: {self.settings.allow_mouse}\n"
                        f"KEYBOARD_FIRST_STRICT: {self.settings.prefer_keyboard_strict}\n"
                        f"REMINDER: reply with ONLY one JSON object.\n"
                    ),
                }

                attached_images = False
                if self.settings.vision:
                    shot = self._get_screenshot_b64_blocking()
                    if shot:
                        user_msg["images"] = [shot]
                        attached_images = True
                    else:
                        user_msg["content"] += "\n(No screenshot available.)"

                with QtCore.QMutexLocker(self._messages_mutex):
                    self._messages.append(user_msg)
                    messages_snapshot = list(self._messages)

                content_full = ""
                try:
                    for content_chunk, done in self.client.chat_stream(
                        self.settings.model,
                        messages_snapshot,
                        float(self.settings.temperature),
                        timeout=300
                    ):
                        if self._stop:
                            break
                        if content_chunk:
                            content_full = self._merge_stream_text(content_full, content_chunk)
                            self.last_response.emit(content_full[-6000:])
                        if done:
                            break
                except Exception as e:
                    msg = str(e).lower()
                    if attached_images and step != self._last_step_image_retry and (
                        "image" in msg or "vision" in msg or "does not support" in msg or "unsupported" in msg
                    ):
                        self._last_step_image_retry = step
                        self.notify_user.emit(
                            "Model rejected screenshots — retrying WITHOUT images. "
                            "If you want vision, use a vision-capable model."
                        )
                        with QtCore.QMutexLocker(self._messages_mutex):
                            if self._messages and self._messages[-1].get("role") == "user":
                                self._messages[-1].pop("images", None)
                            messages_snapshot = list(self._messages)

                        content_full = ""
                        for content_chunk, done in self.client.chat_stream(
                            self.settings.model,
                            messages_snapshot,
                            float(self.settings.temperature),
                            timeout=300
                        ):
                            if self._stop:
                                break
                            if content_chunk:
                                content_full = self._merge_stream_text(content_full, content_chunk)
                                self.last_response.emit(content_full[-6000:])
                            if done:
                                break
                    else:
                        raise

                content_full = (content_full or "").strip()
                LOG.info("Step %d model output tail: %s", step, content_full[-500:].replace("\n", " "))

                with QtCore.QMutexLocker(self._messages_mutex):
                    self._messages.append({"role": "assistant", "content": content_full})

                act = extract_action(content_full)
                if not act:
                    repaired = self._repair_json_once(content_full)
                    act = repaired if repaired else {"action": "wait", "seconds": 1}

                act = normalize_action(act)

                # -------- STRICT KEYBOARD-FIRST ENFORCEMENT --------
                if self.settings.prefer_keyboard_strict and act["action"] in MOUSE_ACTIONS:
                    retries = 0
                    while retries < int(self.settings.mouse_retry_limit) and act["action"] in MOUSE_ACTIONS and not self._stop:
                        retries += 1
                        self.notify_user.emit("Mouse action proposed — forcing keyboard-only replan…")
                        replanned = self._replan_without_mouse(step, why=f"Mouse action '{act['action']}' is discouraged.")
                        if replanned:
                            act = replanned
                        else:
                            break

                # If still mouse, optionally require approval even in AUTO mode
                force_manual_for_mouse = bool(self.settings.always_confirm_mouse) and act["action"] in MOUSE_ACTIONS

                # Manual approval (either manual mode or forced-for-mouse)
                if (not self.settings.auto_execute) or force_manual_for_mouse:
                    self.state.emit("Awaiting approval")
                    if not self._manual_approval_blocking(act):
                        self.stats.emit(step, "rejected")
                        time.sleep(0.2)
                        continue

                self.state.emit("Executing")

                # Execute approved step even if manual mode:
                orig_auto = self.settings.auto_execute
                if not orig_auto:
                    self.settings.auto_execute = True
                try:
                    result = self._execute_action(act)
                finally:
                    self.settings.auto_execute = orig_auto

                self.stats.emit(step, result)

                if result == "done":
                    self.done_sound.emit()
                    self.state.emit("Idle")
                    break

                if not result.startswith(("unknown", "blocked", "proposed")):
                    self.action_sound.emit()

            except pyautogui.FailSafeException:
                self.notify_user.emit("PyAutoGUI FAILSAFE triggered (mouse to a corner). Stopping agent.")
                self.state.emit("Idle")
                break
            except Exception as e:
                LOG.exception("Worker error")
                self.error.emit(str(e))
                self.state.emit("Error")

            time.sleep(max(0.1, float(self.settings.interval)))


# -------------------- Tinted Glass Dock --------------------
class FloatingDock(QtWidgets.QWidget):
    start_stop_clicked = QtCore.Signal()
    open_settings_clicked = QtCore.Signal()
    approve_clicked = QtCore.Signal()
    reject_clicked = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._state = "Idle"
        self._step = 0
        self._last = "-"
        self._awaiting_approval = False

        self._st = AppSettings()
        self._bg_pix: Optional[QtGui.QPixmap] = None
        self._last_blur_ts = 0.0

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self._blur_timer = QtCore.QTimer(self)
        self._blur_timer.timeout.connect(self.update_blur_background)
        self._blur_timer.start(33)

        self._build_ui()
        self._position_top_center()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        self.lbl = QtWidgets.QLabel("Idle • step 0 • -")
        self.lbl.setStyleSheet("color: rgba(255,255,255,220); font-size: 12px; font-weight: 650;")
        root.addWidget(self.lbl, 1)

        self.btn_reject = QtWidgets.QPushButton("Reject")
        self.btn_reject.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_reject.setFixedHeight(30)
        self.btn_reject.setStyleSheet("""
            QPushButton { background: rgba(255,120,120,18); border: 1px solid rgba(255,120,120,55);
                          color: rgba(255,255,255,230); border-radius: 10px; padding: 6px 12px; font-weight: 800; }
            QPushButton:hover { background: rgba(255,120,120,28); }
        """)
        self.btn_reject.clicked.connect(self.reject_clicked.emit)
        root.addWidget(self.btn_reject, 0)

        self.btn_approve = QtWidgets.QPushButton("Approve")
        self.btn_approve.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_approve.setFixedHeight(30)
        self.btn_approve.setStyleSheet("""
            QPushButton { background: rgba(120,255,160,18); border: 1px solid rgba(120,255,160,55);
                          color: rgba(255,255,255,230); border-radius: 10px; padding: 6px 12px; font-weight: 800; }
            QPushButton:hover { background: rgba(120,255,160,28); }
        """)
        self.btn_approve.clicked.connect(self.approve_clicked.emit)
        root.addWidget(self.btn_approve, 0)

        self.btn_settings = QtWidgets.QPushButton("⚙")
        self.btn_settings.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_settings.setToolTip("Settings")
        self.btn_settings.setFixedSize(34, 30)
        self.btn_settings.setStyleSheet("""
            QPushButton { background: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,40);
                          color: rgba(255,255,255,220); border-radius: 10px; font-weight: 800; }
            QPushButton:hover { background: rgba(255,255,255,28); }
        """)
        root.addWidget(self.btn_settings, 0)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_start.setToolTip("Start/Stop agent")
        self.btn_start.setFixedHeight(30)
        self.btn_start.setStyleSheet("""
            QPushButton { background: rgba(120,200,255,22); border: 1px solid rgba(120,200,255,55);
                          color: rgba(255,255,255,230); border-radius: 10px; padding: 6px 12px; font-weight: 800; }
            QPushButton:hover { background: rgba(120,200,255,32); }
        """)
        root.addWidget(self.btn_start, 0)

        self.btn_start.clicked.connect(self.start_stop_clicked.emit)
        self.btn_settings.clicked.connect(self.open_settings_clicked.emit)

        self.btn_approve.setVisible(False)
        self.btn_reject.setVisible(False)

    def _position_top_center(self):
        screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        w, h = 820, 50
        x = geo.x() + (geo.width() - w) // 2
        y = geo.y() + 12
        self.setGeometry(x, y, w, h)

    def apply_style(self, st: AppSettings):
        self._st = st
        interval_ms = int(max(16, 1000.0 / max(1.0, float(st.blur_fps))))
        self._blur_timer.setInterval(interval_ms)
        self.update_blur_background(force=True)
        self.update()

    def set_running(self, running: bool):
        self._running = running
        self.btn_start.setText("Stop" if running else "Start")

    def set_state(self, state: str):
        self._state = state or "Idle"
        self._update_label()

    def set_stats(self, step: int, last: str):
        self._step = int(step)
        self._last = last or "-"
        self._update_label()

    def set_awaiting_approval(self, on: bool):
        self._awaiting_approval = bool(on)
        self.btn_approve.setVisible(self._awaiting_approval)
        self.btn_reject.setVisible(self._awaiting_approval)
        self._update_label()

    def _update_label(self):
        suffix = " • APPROVE/REJECT" if self._awaiting_approval else ""
        self.lbl.setText(f"{self._state} • step {self._step} • {self._last}{suffix}")

    def update_blur_background(self, force: bool = False):
        now = time.time()
        if not force:
            min_dt = max(0.016, 1.0 / max(1.0, float(self._st.blur_fps)))
            if now - self._last_blur_ts < min_dt:
                return
        self._last_blur_ts = now

        try:
            g = self.geometry()
            x, y, w, h = g.x(), g.y(), g.width(), g.height()
            sw, sh = pyautogui.size()
            rx = clamp(x, 0, max(0, sw - 1))
            ry = clamp(y, 0, max(0, sh - 1))
            rw = clamp(w, 1, sw - rx)
            rh = clamp(h, 1, sh - ry)
            if rw < 2 or rh < 2:
                return

            shot = pyautogui.screenshot(region=(rx, ry, rw, rh))
            if shot.mode != "RGBA":
                shot = shot.convert("RGBA")

            ds = float(self._st.blur_downscale)
            if ds < 0.999:
                w2 = max(1, int(rw * ds))
                h2 = max(1, int(rh * ds))
                small = shot.resize((w2, h2), resample=Image.BILINEAR)
                small = small.filter(ImageFilter.GaussianBlur(radius=float(self._st.blur_radius) * ds))
                blurred = small.resize((rw, rh), resample=Image.BILINEAR)
            else:
                blurred = shot.filter(ImageFilter.GaussianBlur(radius=float(self._st.blur_radius)))

            blurred = ImageEnhance.Brightness(blurred).enhance(0.92)
            blurred = ImageEnhance.Color(blurred).enhance(1.06)

            data = blurred.tobytes("raw", "RGBA")
            qimg = QtGui.QImage(data, blurred.size[0], blurred.size[1], QtGui.QImage.Format_RGBA8888)
            self._bg_pix = QtGui.QPixmap.fromImage(qimg)
            self.update()
        except Exception:
            self._bg_pix = None
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect()
        r = 16
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), r, r)
        p.setClipPath(path)

        if self._bg_pix and not self._bg_pix.isNull():
            p.drawPixmap(rect, self._bg_pix)

        tint = QtGui.QColor(int(self._st.tint_r), int(self._st.tint_g), int(self._st.tint_b), int(self._st.tint_a))
        p.fillPath(path, tint)

        grad = QtGui.QLinearGradient(rect.left(), rect.top(), rect.left(), rect.top() + rect.height() * 0.6)
        grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 65))
        grad.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillPath(path, grad)

        p.setClipping(False)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, int(self._st.border_a)))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(path)


# -------------------- Settings dialog --------------------
class SettingsDialog(QtWidgets.QDialog):
    saved = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BlackForge — Settings")
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(720)
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        self.ed_base = QtWidgets.QLineEdit()
        self.ed_model = QtWidgets.QLineEdit()
        self.ed_goal = QtWidgets.QLineEdit()

        self.sp_temp = QtWidgets.QDoubleSpinBox(); self.sp_temp.setRange(0.0, 2.0); self.sp_temp.setSingleStep(0.05)
        self.sp_scale = QtWidgets.QDoubleSpinBox(); self.sp_scale.setRange(0.25, 1.0); self.sp_scale.setSingleStep(0.05)
        self.sp_interval = QtWidgets.QDoubleSpinBox(); self.sp_interval.setRange(0.1, 10.0); self.sp_interval.setSingleStep(0.1)

        self.cb_vision = QtWidgets.QCheckBox("Send screenshots (Vision) — requires a vision-capable model")
        self.cb_auto = QtWidgets.QCheckBox("Auto-execute (dangerous)")
        self.cb_mouse = QtWidgets.QCheckBox("Allow mouse fallback (still keyboard-first)")

        self.cb_keyboard_strict = QtWidgets.QCheckBox("Strict keyboard-first: auto-replan if mouse chosen")
        self.cb_confirm_mouse = QtWidgets.QCheckBox("Always confirm mouse actions (even in auto mode)")
        self.sp_mouse_retry = QtWidgets.QSpinBox(); self.sp_mouse_retry.setRange(0, 5)

        self.sp_max_steps = QtWidgets.QSpinBox(); self.sp_max_steps.setRange(0, 20000); self.sp_max_steps.setToolTip("0 = unlimited")

        self.cb_block_altf4 = QtWidgets.QCheckBox("Block Alt+F4")
        self.cb_block_winr = QtWidgets.QCheckBox("Block Win+R")

        self.sp_blur_fps = QtWidgets.QDoubleSpinBox(); self.sp_blur_fps.setRange(1.0, 60.0); self.sp_blur_fps.setSingleStep(1.0)
        self.sp_blur_radius = QtWidgets.QDoubleSpinBox(); self.sp_blur_radius.setRange(0.0, 40.0); self.sp_blur_radius.setSingleStep(1.0)
        self.sp_blur_ds = QtWidgets.QDoubleSpinBox(); self.sp_blur_ds.setRange(0.2, 1.0); self.sp_blur_ds.setSingleStep(0.05)

        self.btn_refresh = QtWidgets.QPushButton("Refresh models")
        self.btn_refresh.setCursor(QtCore.Qt.PointingHandCursor)

        form = QtWidgets.QFormLayout()
        form.addRow("Base URL", self.ed_base)
        form.addRow("Model", self.ed_model)
        form.addRow("", self.btn_refresh)
        form.addRow("Goal", self.ed_goal)
        form.addRow("Temperature", self.sp_temp)
        form.addRow("Scale", self.sp_scale)
        form.addRow("Interval (s)", self.sp_interval)
        form.addRow("Max steps", self.sp_max_steps)
        form.addRow("", self.cb_vision)
        form.addRow("", self.cb_auto)
        form.addRow("", self.cb_mouse)
        form.addRow("", self.cb_keyboard_strict)
        form.addRow("Mouse replan attempts", self.sp_mouse_retry)
        form.addRow("", self.cb_confirm_mouse)
        form.addRow("Glass blur FPS", self.sp_blur_fps)
        form.addRow("Glass blur radius", self.sp_blur_radius)
        form.addRow("Glass blur downscale", self.sp_blur_ds)

        lay.addLayout(form)
        lay.addWidget(self.cb_block_altf4)
        lay.addWidget(self.cb_block_winr)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_close.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_save.setCursor(QtCore.Qt.PointingHandCursor)
        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_save)
        lay.addLayout(btns)

        self.btn_close.clicked.connect(self.close)
        self.btn_save.clicked.connect(self.saved.emit)

    def load_from_settings(self, st: AppSettings):
        self.ed_base.setText(st.base_url)
        self.ed_model.setText(st.model)
        self.ed_goal.setText(st.goal)
        self.sp_temp.setValue(float(st.temperature))
        self.sp_scale.setValue(float(st.scale))
        self.sp_interval.setValue(float(st.interval))
        self.sp_max_steps.setValue(int(st.max_steps))

        self.cb_vision.setChecked(bool(st.vision))
        self.cb_auto.setChecked(bool(st.auto_execute))
        self.cb_mouse.setChecked(bool(st.allow_mouse))

        self.cb_keyboard_strict.setChecked(bool(st.prefer_keyboard_strict))
        self.sp_mouse_retry.setValue(int(st.mouse_retry_limit))
        self.cb_confirm_mouse.setChecked(bool(st.always_confirm_mouse))

        self.cb_block_altf4.setChecked(bool(st.block_alt_f4))
        self.cb_block_winr.setChecked(bool(st.block_win_r))
        self.sp_blur_fps.setValue(float(st.blur_fps))
        self.sp_blur_radius.setValue(float(st.blur_radius))
        self.sp_blur_ds.setValue(float(st.blur_downscale))

    def apply_to_settings(self, st: AppSettings) -> AppSettings:
        st.base_url = self.ed_base.text().strip() or DEFAULT_BASE_URL
        st.model = self.ed_model.text().strip() or DEFAULT_MODEL
        st.goal = self.ed_goal.text().strip() or st.goal
        st.temperature = float(self.sp_temp.value())
        st.scale = float(self.sp_scale.value())
        st.interval = float(self.sp_interval.value())
        st.max_steps = int(self.sp_max_steps.value())

        st.vision = bool(self.cb_vision.isChecked())
        st.auto_execute = bool(self.cb_auto.isChecked())
        st.allow_mouse = bool(self.cb_mouse.isChecked())

        st.prefer_keyboard_strict = bool(self.cb_keyboard_strict.isChecked())
        st.mouse_retry_limit = int(self.sp_mouse_retry.value())
        st.always_confirm_mouse = bool(self.cb_confirm_mouse.isChecked())

        st.block_alt_f4 = bool(self.cb_block_altf4.isChecked())
        st.block_win_r = bool(self.cb_block_winr.isChecked())

        st.blur_fps = float(self.sp_blur_fps.value())
        st.blur_radius = float(self.sp_blur_radius.value())
        st.blur_downscale = float(self.sp_blur_ds.value())

        return _sanitize_settings(st)


# -------------------- Controller --------------------
class BlackForgeApp(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.client = OllamaClient(self.settings.base_url)

        self.bridge = ScreenshotBridge()
        self.worker = AgentWorker(self.bridge)
        self.sounds = SoundBank()

        self.dock = FloatingDock()
        self.dlg = SettingsDialog(self.dock)

        self._wire()
        self._apply_settings_to_ui()

        self.dock.show()
        self.dock.raise_()

    def _wire(self):
        self.dock.start_stop_clicked.connect(self.toggle_start_stop)
        self.dock.open_settings_clicked.connect(self.open_settings)

        self.dock.approve_clicked.connect(lambda: self.worker.approve(True))
        self.dock.reject_clicked.connect(lambda: self.worker.approve(False))

        self.worker.state.connect(self.dock.set_state)
        self.worker.stats.connect(self.dock.set_stats)
        self.worker.action_sound.connect(self.sounds.play_action)
        self.worker.done_sound.connect(self.sounds.play_done)
        self.worker.notify_user.connect(self._toast)
        self.worker.error.connect(self._on_error)

        self.worker.waiting_approval.connect(self.dock.set_awaiting_approval)

        self.bridge.request.connect(self.capture_for_worker, QtCore.Qt.QueuedConnection)

        self.dlg.saved.connect(self.save_settings_from_dialog)
        self.dlg.btn_refresh.clicked.connect(self.refresh_models)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self._on_quit)

    def _on_quit(self):
        try:
            self.stop_agent()
        except Exception:
            pass

    def _toast(self, text: str):
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text)

    def _on_error(self, e: str):
        self.dock.set_state("Error")
        self._toast(f"LLM error: {e}")

    def open_settings(self):
        self.dlg.load_from_settings(self.settings)
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()

    def _apply_settings_to_ui(self):
        self.settings = _sanitize_settings(self.settings)
        self.dock.apply_style(self.settings)
        self.worker.configure(self.settings)
        self.dock.set_running(self.worker.isRunning())

    def save_settings_from_dialog(self):
        self.settings = self.dlg.apply_to_settings(self.settings)
        ok = save_settings(self.settings)
        self.worker.configure(self.settings)
        self.dock.apply_style(self.settings)
        self._toast("Saved ✅" if ok else "Save failed ❌")

    def refresh_models(self):
        tmp = self.dlg.apply_to_settings(AppSettings(**asdict(self.settings))) if self.dlg.isVisible() else self.settings
        self.client.set_base_url(tmp.base_url)
        models = self.client.list_models()
        cur = self.dlg.ed_model.text().strip() or tmp.model

        if models and self.dlg.isVisible():
            menu = QtWidgets.QMenu(self.dlg)
            for m in models[:300]:
                act = menu.addAction(m)
                act.triggered.connect(lambda _=False, mm=m: self.dlg.ed_model.setText(mm))
            menu.addSeparator()
            keep = menu.addAction(f"(keep typed) {cur}")
            keep.triggered.connect(lambda: self.dlg.ed_model.setText(cur))
            menu.exec(QtGui.QCursor.pos())

        self._toast(f"Found {len(models)} model(s)")

    def toggle_start_stop(self):
        if self.worker.isRunning():
            self.stop_agent()
        else:
            self.start_agent()

    def start_agent(self):
        if self.worker.isRunning():
            return
        self.worker.configure(self.settings)
        self.worker.start()
        self.dock.set_running(True)
        self._toast("Agent started")

    def stop_agent(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self.dock.set_running(False)
        self.dock.set_state("Idle")
        self.dock.set_awaiting_approval(False)
        self._toast("Agent stopped")

    # ---------- Screenshot capture ----------
    def capture_for_worker(self, scale: float):
        was_visible = self.dock.isVisible()
        self.dock.setVisible(False)
        QtWidgets.QApplication.processEvents()
        time.sleep(0.05)

        try:
            img = pyautogui.screenshot()
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            if scale != 1.0:
                w, h = img.size
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.BILINEAR)

            if self.settings.capture_stamp:
                d = ImageDraw.Draw(img)
                stamp = f"CAPTURE {time.strftime('%H:%M:%S')} scale={scale}"
                d.text((10, 10), stamp, fill=(240, 240, 240, 200))

            if self.settings.show_cursor_marker:
                mx, my = pyautogui.position()
                mx_s = int(mx * scale)
                my_s = int(my * scale)
                draw_cursor_marker(img, mx_s, my_s)

            img = add_soft_grid(img, minor=int(self.settings.grid_minor), major=int(self.settings.grid_major))
            b64 = b64_png_from_pil(img)

        except Exception as e:
            b64 = ""
            self._toast(f"Screenshot error: {e}")
        finally:
            if was_visible:
                self.dock.setVisible(True)
            QtWidgets.QApplication.processEvents()

        self.bridge.ready.emit(b64)


def main():
    # Helps some frozen builds behave nicely
    QtCore.QCoreApplication.setApplicationName("BlackForge")
    QtCore.QCoreApplication.setOrganizationName("BlackForge")
    app = QtWidgets.QApplication(sys.argv)
    _ = BlackForgeApp()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
