"""Command-line tool and Python library for OpenAI-compatible image APIs.

Two modes:
  - generate (generations): create new images from a prompt alone.
  - edit (edits): create new images from one or more reference images
    (optionally with a mask) plus a prompt.

CLI: the API key and base URL come from environment variables or a .env file
next to this script (see config.example); run ``python promptbrush.py --help``
for usage. Logs go to stderr and saved image paths go to stdout, one per line,
so scripts can consume them.

As a library: ``create_image`` sends one request and returns the image bytes,
``MODELS`` is the measured model capability table, ``check_size`` /
``fit_size`` handle the size rules, and ``log_sink`` hands the current
thread's log lines to the caller.
"""

import argparse
import base64
import contextlib
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

__version__ = "0.1.0"

__all__ = [
    "MODELS", "DEFAULT_MODEL", "DEFAULT_SIZE", "LENIENT_MODERATION_MODEL",
    "ModelInfo", "Generation",
    "ImageAPIError", "create_image", "check_size", "fit_size",
    "detect_image_ext", "png_dimensions", "log_sink",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env"
PROMPT_FILE = SCRIPT_DIR / "prompt.md"

ENV_PREFIX = "PROMPTBRUSH_"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelInfo:
    """Measured model behavior. It depends on the provider; treat it as a guide."""
    seconds: str   # time per image
    size: str      # "exact" (renders -s) | "aspect" (ratio only) | "fixed" (1024x1024)
    aligned: bool  # edits keep the original framing (same ratio, no shift), so the
                   # result can be compared pixel by pixel or pasted back
    mask: bool     # edits respect the mask: nothing outside its transparent area changes
    note: str = ""


# Mainstream image models verified on one OpenAI-compatible relay service
# (2026-10; select with -m, all support -i editing). Edit behavior comes from
# the same local replacement on one 1536x1024 photo:
#   - only gpt-image-2-vip respects the mask, and it clips hard to it, cutting
#     off any part of the new object that extends past the mask; the other
#     models give the same result with or without a mask, with slight color
#     shifts and detail changes outside the selection;
#   - the gemini models always output 1024x1024 and reframe the shot, so the
#     result can't be compared with the original pixel by pixel;
#   - every model can locate the region to change from "original + annotated
#     copy", and none of them draws the annotations into the result;
#   - gpt-image-2 occasionally returns two images for a multi-image edit (one
#     was requested); both are saved.
MODELS: dict[str, ModelInfo] = {
    "gpt-image-2.5-sunburst": ModelInfo(
        "40–90s", "exact", True, False,
        "best quality; keeps the most of the original in edits"),
    "gpt-image-2.5-flare": ModelInfo(
        "25–45s", "exact", True, False, "the default"),
    "gpt-image-2": ModelInfo(
        "20–40s", "aspect", True, False, "~1.57 MP; most lenient moderation"),
    "gpt-image-2-vip": ModelInfo(
        "30–75s", "exact", True, True, "most sizes rendered exactly; up to 4K"),
    "gpt-image-1.5": ModelInfo(
        "15–30s", "aspect", True, False, "~1.57 MP"),
    "gemini-nano-banana-2.1": ModelInfo(
        "15–30s", "fixed", False, False),
    "gemini-3-pro-image": ModelInfo(
        "25–45s", "fixed", False, False, "also known as Nano Banana Pro"),
    "gemini-3.1-flash-image": ModelInfo(
        "10–20s", "fixed", False, False, "fastest"),
    "jimeng-4.5": ModelInfo(
        "40–55s", "aspect", True, False, '~4.15 MP; adds an "AI" watermark in a corner'),
}

# flare is the default: second only to sunburst on every leaderboard and about
# 2.5x faster. sunburst has the best quality but occasionally exceeds
# Cloudflare's ~2-minute limit and returns 524, which triggers a retry (the
# origin may already have generated and billed the image); pick it with -m
# when you need it.
DEFAULT_MODEL = "gpt-image-2.5-flare"
DEFAULT_SIZE = "1600x1200"
DEFAULT_N = 1
# Each image is a request that is billed: a slip of the finger (-n 100) is refused
MAX_N = 10
DEFAULT_OUTPUT_DIR = "."
# The model with the most lenient moderation, suggested when an image is
# blocked by moderation (HTTP 451)
LENIENT_MODERATION_MODEL = "gpt-image-2"
# Error codes that mean the same, from APIs that answer moderation with 400
MODERATION_CODES = {"moderation_blocked", "content_policy_violation"}
# ImageAPIError.code of a successful response that holds no image
NO_IMAGE = "no_image"

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# Format found by detect_image_ext → MIME type for uploaded reference images
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif"}
SIZE_PATTERN = re.compile(r"^(auto|[1-9]\d*x[1-9]\d*)$")
MAX_ASPECT = 3.0
# Measured size behavior varies by model:
#   - gpt-image-2.5-*: renders the -s size exactly and rejects invalid sizes
#     outright, or occasionally hangs until a CF 524 (and the retries waste
#     minutes), so the rules below are enforced locally up front;
#   - gpt-image-2: uses only the aspect ratio at a fixed ~1.57 MP total, and
#     switches to another ratio beyond 3:1;
#   - other models have their own rules (e.g. gemini is fixed at 1024x1024),
#     so only the aspect ratio gets a warning.
STRICT_SIZE_PREFIX = "gpt-image-2.5"
SIZE_MULTIPLE = 16
MAX_EDGE = 3840
MIN_PIXELS = 1024 * 640
MAX_PIXELS = 3840 * 2160

# Network retries: back off and retry automatically when the server drops the
# connection, times out or rate-limits
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # retry k waits RETRY_BACKOFF ** k seconds
MAX_RETRY_AFTER = 60  # upper limit (seconds) for honoring a server's Retry-After
# 429 is rate limiting; 500/502/503/504 are ordinary 5xx; 520-527 are
# Cloudflare origin errors (524 = origin timed out). Generation time varies,
# so a retry often finishes within the CF limit.
RETRY_STATUS = {429, 500, 502, 503, 504} | set(range(520, 528))

# Timeouts are all (connect, read): short for connecting, long for waiting on
# generation. Generating an image can take minutes, so the read timeout is
# generous to avoid mistaking a normal generation for a failure.
CONNECT_TIMEOUT = 15
TIMEOUT_GENERATE = (CONNECT_TIMEOUT, 300)
TIMEOUT_EDIT = (CONNECT_TIMEOUT, 360)
TIMEOUT_DOWNLOAD = (CONNECT_TIMEOUT, 120)

# gpt-image-2 was measured to ignore n and return one image per request, so
# -n N is always split into N single-image requests sent concurrently; too
# much concurrency easily triggers 429.
MAX_PARALLEL = 4

PROMPT_PREVIEW_LEN = 80
ERROR_BODY_LEN = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_COLORS = {
    "info": "\033[36m",   # cyan
    "ok": "\033[32m",     # green
    "warn": "\033[33m",   # yellow
    "error": "\033[31m",  # red
    "retry": "\033[35m",  # magenta
}
_RESET = "\033[0m"


def _supports_color() -> bool:
    """Use color only on a terminal and when NO_COLOR (https://no-color.org) is unset."""
    return sys.stderr.isatty() and not os.environ.get("NO_COLOR")


_USE_COLOR = _supports_color()
# Thread-local state: tag labels log lines of concurrent requests (e.g. "2/3");
# for sink see log_sink
_job = threading.local()
_log_lock = threading.Lock()


def log(level: str, msg: str) -> None:
    """Print one log line to stderr, prefixed with its level and colored on a terminal.

    stdout is reserved for saved image paths so scripts can consume it. print
    writes the text and the newline separately, so a lock keeps each line
    whole when threads log concurrently. If the current thread has a
    ``log_sink``, the line goes there instead and nothing is printed.
    """
    tag = getattr(_job, "tag", None)
    if tag:
        msg = f"[{tag}] {msg}"
    sink = getattr(_job, "sink", None)
    if sink is not None:
        sink(level, msg)
        return
    if _USE_COLOR and level in _COLORS:
        prefix = f"{_COLORS[level]}[{level}]{_RESET}"
    else:
        prefix = f"[{level}]"
    with _log_lock:
        print(f"{prefix} {msg}", file=sys.stderr)


@contextlib.contextmanager
def log_sink(sink: Callable[[str, str], None]) -> Iterator[None]:
    """Within the with block, pass this thread's logs to ``sink(level, msg)``.

    Nothing goes to stderr meanwhile. Lets library callers collect progress
    such as retries and usage; affects only the current thread and can be
    nested.
    """
    previous = getattr(_job, "sink", None)
    _job.sink = sink
    try:
        yield
    finally:
        _job.sink = previous


def _brief(text: str, limit: int = ERROR_BODY_LEN) -> str:
    """Collapse a response body to a single truncated line for logging (a stray
    carriage return would overwrite the start of the line on a terminal)."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------------------
# Config and file reading
# ---------------------------------------------------------------------------
def read_text_file(path: Path) -> str:
    """Read a text file, trying UTF-8 (with or without the BOM Windows Notepad
    writes) and then GBK."""
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path}: not UTF-8 or GBK encoded")


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file: KEY=VALUE per line, skipping blank lines, # comments
    and an optional ``export`` prefix.

    A quoted value is the text inside the quotes, ignoring anything after
    them; in an unquoted value, a ``#`` preceded by whitespace starts a
    trailing comment (the # in ``a#b`` doesn't). Returns an empty dict if the
    file doesn't exist."""
    if not path.exists():
        return {}
    values = {}
    for raw in read_text_file(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.removeprefix("export ").split("=", 1)
        value = raw_value.strip()
        quote = value[:1]
        if quote in ("'", '"') and (end := value.find(quote, 1)) != -1:
            value = value[1:end]
        else:
            value = re.split(r"\s#", raw_value, maxsplit=1)[0].strip()
        values[key.strip()] = value
    return values


def get_setting(name: str, default: str = "") -> str:
    """Read ``PROMPTBRUSH_<name>``: environment variable > .env > default."""
    key = ENV_PREFIX + name
    value = os.environ.get(key) or read_env_file(ENV_FILE).get(key)
    return (value or default).strip()


def load_api_key() -> str:
    key = get_setting("API_KEY")
    if not key:
        raise ValueError(
            f"API key not configured: set the {ENV_PREFIX}API_KEY environment "
            f"variable, or copy config.example to {ENV_FILE} and fill it in")
    return key


def api_endpoints(base_url: str | None = None) -> tuple[str, str]:
    """Return the (generations, edits) endpoint URLs; ``base_url`` defaults to
    ``PROMPTBRUSH_BASE_URL``."""
    base = (base_url or get_setting("BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    return f"{base}/images/generations", f"{base}/images/edits"


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"No prompt given: pass one as an argument or write it to {PROMPT_FILE}")
    text = read_text_file(PROMPT_FILE).strip()
    if not text:
        raise ValueError(f"{PROMPT_FILE} is empty: write your prompt in it")
    return text


def find_reference_images(paths: list[str]) -> list[Path]:
    """Expand file/directory paths into a deduplicated, ordered list of images.

    Directories contribute only files in supported formats. But a path the
    user gave that doesn't exist, has an unsupported format, or is a directory
    without images is always an error: with a reference image missing, the
    result isn't what the user asked for, so stop before the paid request
    rather than carry on with the remaining images.
    """
    images: list[Path] = []
    problems: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            # hidden files are skipped: macOS leaves "._photo.png" beside photos copied to
            # other disks, and it isn't an image
            found = [f for f in sorted(path.iterdir()) if f.is_file()
                     and not f.name.startswith(".")
                     and f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
            if not found:
                problems.append(f"{raw} (no images in directory)")
            images.extend(found)
        elif not path.is_file():
            problems.append(f"{raw} (not found)")
        elif path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            problems.append(f"{raw} (unsupported format)")
        else:
            images.append(path)
    if problems:
        exts = ", ".join(sorted(SUPPORTED_IMAGE_EXTS))
        raise ValueError(f"Invalid reference image paths: {'; '.join(problems)}. "
                         f"Supported formats: {exts}")

    seen: set[Path] = set()
    unique: list[Path] = []
    for img in images:
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(img)
    return unique


def detect_image_ext(data: bytes) -> str | None:
    """Detect the image format from magic bytes; return its extension, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return None


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read (width, height) from a PNG's IHDR chunk; None if it isn't a PNG."""
    if detect_image_ext(data) != ".png" or len(data) < 24:
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def write_unique(directory: Path, stem: str, ext: str, data: bytes) -> Path:
    """Write to the first unused ``stem_<n>ext`` file name and return its path.

    Files are created in exclusive mode, so runs in the same second and
    concurrent requests never overwrite each other.
    """
    k = 0
    while True:
        path = directory / f"{stem}_{k}{ext}"
        try:
            with path.open("xb") as f:
                f.write(data)
            return path
        except FileExistsError:
            k += 1


# ---------------------------------------------------------------------------
# Size rules
# ---------------------------------------------------------------------------
def _size_problems(w: int, h: int) -> list[str]:
    """Check a size against the measured rules of strict models; return the
    rules it breaks (empty if valid)."""
    problems = []
    if w % SIZE_MULTIPLE or h % SIZE_MULTIPLE:
        problems.append(f"width and height must be multiples of {SIZE_MULTIPLE}")
    if max(w, h) > MAX_EDGE:
        problems.append(f"each side must be at most {MAX_EDGE}")
    if max(w, h) / min(w, h) > MAX_ASPECT:
        problems.append(f"aspect ratio must be at most {MAX_ASPECT:.0f}:1")
    if not MIN_PIXELS <= w * h <= MAX_PIXELS:
        problems.append(f"total pixels must be between {MIN_PIXELS:,} "
                        f"and {MAX_PIXELS:,}")
    return problems


def check_size(model: str, size: str) -> None:
    """Check a size against the measured rules: raise ValueError for strict
    models, only warn for the others."""
    if not SIZE_PATTERN.match(size):
        raise ValueError(f"Size must be WIDTHxHEIGHT (e.g. 1536x1024) or auto, got {size!r}")
    if size == "auto":
        return
    w, h = map(int, size.split("x"))
    if not model.startswith(STRICT_SIZE_PREFIX):
        ratio = max(w, h) / min(w, h)
        if ratio > MAX_ASPECT:
            log("warn", f"Aspect ratio {ratio:.1f}:1 exceeds {MAX_ASPECT:.0f}:1; "
                        "in testing the server did not render at this ratio")
        return
    if problems := _size_problems(w, h):
        raise ValueError(
            f"{model} does not support size {size}: {'; '.join(problems)}. "
            "Common valid sizes: 1024x1024, 1536x1024, 1600x1200, 3840x2160 or auto")


def fit_size(width: int, height: int) -> str:
    """Turn any width and height into a ``WxH`` size that every model accepts.

    Keeps the aspect ratio and pixel count as close as possible. Used when
    editing an existing image so the output has the original's proportions:
    the result satisfies every rule of strict models (see ``check_size``),
    and originals wider than 3:1 can only get 3:1.
    """
    if width < 1 or height < 1:
        raise ValueError(f"Invalid width/height: {width}x{height}")
    ratio = min(max(width / height, 1 / MAX_ASPECT), MAX_ASPECT)
    pixels = min(max(width * height, MIN_PIXELS), MAX_PIXELS)
    h = math.sqrt(pixels / ratio)
    w = h * ratio
    shrink = min(1.0, MAX_EDGE / max(w, h))
    w0 = round(w * shrink / SIZE_MULTIPLE)
    h0 = round(h * shrink / SIZE_MULTIPLE)
    # Rounding to multiples of 16 can cross a limit, so search nearby for the
    # valid size closest in aspect ratio
    best = None
    for dw in range(-3, 4):
        for dh in range(-3, 4):
            cw, ch = (w0 + dw) * SIZE_MULTIPLE, (h0 + dh) * SIZE_MULTIPLE
            if cw < 1 or ch < 1 or _size_problems(cw, ch):
                continue
            score = (abs(math.log(cw / ch / ratio)), abs(cw * ch - pixels))
            if best is None or score < best[0]:
                best = (score, cw, ch)
    assert best is not None, "there is always a valid size nearby"
    return f"{best[1]}x{best[2]}"


# ---------------------------------------------------------------------------
# HTTP requests
# ---------------------------------------------------------------------------
def _retry_after(resp: requests.Response) -> float:
    """Read the Retry-After header, in seconds or as an HTTP date, capped at
    MAX_RETRY_AFTER; 0 if missing or unparsable."""
    value = resp.headers.get("Retry-After", "").strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return 0
    return min(max(seconds, 0), MAX_RETRY_AFTER) if math.isfinite(seconds) else 0


def _retry_wait(reason: str, attempt: int, min_wait: float = 0) -> None:
    """Log why we're retrying, then back off exponentially (at least ``min_wait`` s)."""
    wait = max(RETRY_BACKOFF ** attempt, min_wait)
    log("retry", f"{reason}, retrying in {wait:.0f}s ({attempt}/{MAX_RETRIES - 1})")
    time.sleep(wait)


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Retry transient network glitches and server disconnects with
    exponential backoff.

    On a connection error, a timeout, or a retryable status code (see
    ``RETRY_STATUS``), back off and retry, waiting at least as long as the
    server's Retry-After if it sends one. When the attempts run out, the last
    response is returned or the last exception re-raised.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                _retry_wait(f"HTTP {resp.status_code}", attempt,
                            _retry_after(resp))
                continue
            return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                _retry_wait(f"Connection error ({type(exc).__name__})", attempt)

    log("error", f"Still failing after {MAX_RETRIES - 1} retries")
    raise last_exc  # type: ignore[misc]


class ImageAPIError(Exception):
    """The API reported a failure: a non-200 HTTP status, a response that isn't a
    JSON object, or one with an ``error`` field.

    ``status`` is the HTTP status code (200 when the body is what's wrong).
    Network failures that outlast the retries don't use this class; the
    requests exception is raised as is.
    """

    def __init__(self, message: str, status: int, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code  # the body's error.code, when it has one

    @property
    def moderated(self) -> bool:
        """Whether content moderation blocked the image: HTTP 451 (relays), or an
        error code saying so (OpenAI answers 400 ``moderation_blocked``)."""
        return self.status == 451 or self.code in MODERATION_CODES


def _error_code(body: object) -> str:
    """The ``error.code`` of an OpenAI-style error body, or ""."""
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else ""


@dataclass
class Generation:
    """Result of one request. ``images`` holds the decoded image bytes, minus
    any items that failed."""
    images: list[bytes]
    usage: dict | None
    elapsed: float  # seconds, including retries


def _item_bytes(item: dict) -> bytes | None:
    """Return one item's image bytes, from b64_json or else by downloading url.

    gpt-image-2 was measured to return only urls, and a failed download throws
    away an image that's already been paid for, so downloads are retried too.
    """
    if not isinstance(item, dict):
        return None
    b64, url = item.get("b64_json"), item.get("url")
    if b64 and isinstance(b64, str):
        data = base64.b64decode(b64)
    elif url and isinstance(url, str):
        log("info", f"Downloading image: {url}")
        resp = request_with_retry("GET", url, timeout=TIMEOUT_DOWNLOAD)
        resp.raise_for_status()
        data = resp.content
    else:
        return None
    # what an expired link or a relay's error page answers with, rather than an image
    if detect_image_ext(data) is None:
        raise ValueError(f"not an image: {_brief(data.decode('utf-8', 'replace'))}")
    return data


def decode_images(result: dict) -> list[bytes]:
    """Extract the bytes of every image in a response.

    An image that fails to decode or download is skipped on its own, without
    affecting the other paid-for images; malformed fields likewise only affect
    their own part and never cost an image that can still be retrieved.
    """
    items = result.get("data")
    if not isinstance(items, list) or not items:
        log("warn", "No image data received")
        return []
    images: list[bytes] = []
    for i, item in enumerate(items, 1):
        try:
            data = _item_bytes(item)
        except (ValueError, requests.exceptions.RequestException) as exc:
            log("error", f"Failed to fetch image {i}: {exc}")
            continue
        if not data:
            log("warn", f"Image {i} has no usable data, skipping")
            continue
        images.append(data)
    return images


def save_images(images: list[bytes], output_dir: str) -> list[str]:
    """Write image bytes to disk, picking each extension from the actual format,
    and return the saved file paths."""
    if not images:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved: list[str] = []
    for data in images:
        ext = detect_image_ext(data) or ".bin"  # decode_images passes images only
        filename = write_unique(out, f"image_{ts}", ext, data)
        saved.append(str(filename))
        dims = png_dimensions(data)
        size_note = f" ({dims[0]}x{dims[1]})" if dims else ""
        log("ok", f"Saved {filename}{size_note}")
    return saved


def _post_images(url: str, **kwargs) -> Generation:
    """Send the request (with retries), validate the response and extract the
    images; the shared final step of both modes.

    Keyword arguments go to requests unchanged. Raises ``ImageAPIError`` when
    the API reports a failure.
    """
    start = time.monotonic()
    resp = request_with_retry("POST", url, **kwargs)
    elapsed = time.monotonic() - start
    log("info", f"Request finished in {elapsed:.0f}s")

    try:
        result = resp.json()
    except ValueError:
        result = None
    if resp.status_code != 200:
        raise ImageAPIError(f"HTTP {resp.status_code}: {_brief(resp.text)}",
                            resp.status_code, _error_code(result))
    if result is None:
        raise ImageAPIError(f"Response is not valid JSON: {_brief(resp.text)}",
                            resp.status_code)
    if not isinstance(result, dict):
        raise ImageAPIError(f"Unexpected response format: {_brief(resp.text)}",
                            resp.status_code)
    if result.get("error"):
        raise ImageAPIError(f"API returned an error: {_brief(str(result['error']))}",
                            resp.status_code, _error_code(result))
    if not isinstance(result.get("data"), list) or not result["data"]:
        # often a relay's own error, in a shape of its own ({"code": 1001, "msg": ...})
        raise ImageAPIError(f"No image in the response: {_brief(resp.text)}",
                            resp.status_code, NO_IMAGE)

    usage = result.get("usage")
    if isinstance(usage, dict):
        log("info", f"Usage: total={usage.get('total_tokens', 'N/A')} "
                    f"input={usage.get('input_tokens', 'N/A')} "
                    f"output={usage.get('output_tokens', 'N/A')}")
    else:
        usage = None
    return Generation(decode_images(result), usage, elapsed)


def _moderation_hint(model: str) -> str:
    """Advice for the user when content moderation blocks an image (HTTP 451)."""
    hint = "Image blocked by content moderation: try rephrasing the prompt"
    if model != LENIENT_MODERATION_MODEL:
        hint += (f", or switch to -m {LENIENT_MODERATION_MODEL}, whose moderation "
                 "is more lenient (the same prompt passes more often in testing)")
    return hint


def _request_and_save(url: str, output_dir: str, *, model: str,
                      **kwargs) -> list[str]:
    """For the CLI: send one request and save the results; API errors are only
    logged and yield an empty list.

    ``model`` is only used for error hints; other keyword arguments go to
    requests unchanged.
    """
    try:
        gen = _post_images(url, **kwargs)
    except ImageAPIError as exc:
        log("error", str(exc))
        if exc.moderated:
            log("warn", _moderation_hint(model))
        return []
    return save_images(gen.images, output_dir)


def _request_parallel(url: str, output_dir: str, n: int, *, model: str,
                      **kwargs) -> list[str]:
    """Send the same request n times (at most ``MAX_PARALLEL`` at once) and
    collect the saved files.

    A failed request is only logged and doesn't affect the others. Worker
    threads are daemons, so on Ctrl+C the main thread can exit immediately
    instead of waiting for in-flight requests.
    """
    results: list[list[str]] = [[] for _ in range(n)]
    gate = threading.BoundedSemaphore(MAX_PARALLEL)

    def worker(k: int) -> None:
        if n > 1:
            _job.tag = f"{k + 1}/{n}"
        with gate:
            try:
                results[k] = _request_and_save(url, output_dir, model=model,
                                               **kwargs)
            except requests.exceptions.RequestException as exc:
                log("error", f"Network request failed: {exc}")
            except (OSError, ValueError) as exc:
                log("error", str(exc))

    threads = [threading.Thread(target=worker, args=(k,), daemon=True)
               for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [path for paths in results for path in paths]


def _log_request(mode: str, model: str, size: str, n: int,
                 prompt: str, refs: list[Path] | None = None) -> None:
    log("info", f"Mode: {mode}")
    log("info", f"Model: {model} | Size: {size} | Count: {n}")
    if refs:
        log("info", f"Reference images: {', '.join(p.name for p in refs)}")
    preview = prompt if len(prompt) <= PROMPT_PREVIEW_LEN \
        else prompt[:PROMPT_PREVIEW_LEN] + "..."
    log("info", f"Prompt: {preview}")


# ---------------------------------------------------------------------------
# Generate / edit
# ---------------------------------------------------------------------------
def _image_part(stem: str, content: bytes, *,
                label: str) -> tuple[str, bytes, str]:
    """Return the multipart part (filename, content, MIME) for a reference image.

    The format is detected from the content, not the extension: non-image
    content fails before the paid request, and a mismatched extension (e.g. a
    JPEG saved as .png) is replaced by the real one so it matches the MIME type.
    """
    ext = detect_image_ext(content)
    if ext is None:
        raise ValueError(f"{label} is not a PNG/JPEG/WebP/GIF image")
    return stem + ext, content, IMAGE_MIME[ext]


def create_image(
    prompt: str,
    *,
    images: Sequence[bytes] = (),
    mask: bytes | None = None,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Generation:
    """For library use: send one single-image request and return the result
    without saving it.

    Without ``images`` this generates; with them it edits (with several, the
    first one is the image being edited). ``mask`` is a PNG whose transparent
    areas may be changed and must be the same size as the first reference
    image; see ``MODELS`` for which models respect it. ``api_key`` /
    ``base_url`` default to the environment or .env, like the CLI.

    Raises ``ValueError`` before sending anything if the size is invalid or an
    image isn't valid; ``ImageAPIError`` if the API reports a failure; and the
    requests exception once network retries are exhausted. Progress is logged
    to stderr by default; ``log_sink`` can take it over.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if mask is not None and not images:
        raise ValueError("mask only works in edit mode: pass reference images too")
    if mask is not None and detect_image_ext(mask) != ".png":
        raise ValueError("mask must be a PNG image")
    check_size(model, size)

    headers = {"Authorization": f"Bearer {api_key or load_api_key()}"}
    gen_url, edit_url = api_endpoints(base_url)
    if not images:
        payload = {"model": model, "prompt": prompt, "n": 1, "size": size}
        return _post_images(gen_url, headers=headers, json=payload,
                            timeout=TIMEOUT_GENERATE)

    files = [("image[]", _image_part(f"image_{k}", data,
                                     label=f"Reference image {k}"))
             for k, data in enumerate(images, 1)]
    if mask is not None:
        files.append(("mask", ("mask.png", mask, "image/png")))
    data = {"model": model, "prompt": prompt, "n": "1", "size": size}
    return _post_images(edit_url, headers=headers, data=data, files=files,
                        timeout=TIMEOUT_EDIT)


def generate_image(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    n: int = DEFAULT_N,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """Generate n images from a prompt alone (as n single-image requests)."""
    headers = {"Authorization": f"Bearer {load_api_key()}"}
    payload = {"model": model, "prompt": prompt, "n": 1, "size": size}

    _log_request("generate (generations)", model, size, n, prompt)
    return _request_parallel(
        api_endpoints()[0], output_dir, n, model=model, headers=headers,
        json=payload, timeout=TIMEOUT_GENERATE,
    )


def edit_image(
    prompt: str,
    image_paths: list[Path],
    *,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    n: int = DEFAULT_N,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """Generate n new images from reference images plus a prompt (as n
    single-image requests)."""
    headers = {"Authorization": f"Bearer {load_api_key()}"}

    # Read into bytes up front instead of passing file handles, so retries and
    # concurrent requests can resend the full content
    files = [("image[]", _image_part(img.stem, img.read_bytes(),
                                     label=f"Reference image {img}"))
             for img in image_paths]
    data = {"model": model, "prompt": prompt, "n": "1", "size": size}

    _log_request("edit (edits)", model, size, n, prompt, refs=image_paths)
    return _request_parallel(
        api_endpoints()[1], output_dir, n, model=model, headers=headers,
        data=data, files=files, timeout=TIMEOUT_EDIT,
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
_EPILOG = f"""examples:
  python promptbrush.py 'a cat'
  python promptbrush.py -i ref.png 'turn the cat into a dog'
  python promptbrush.py -i img1.png -i img2.jpg 'blend these two images'
  python promptbrush.py -i ./refs/ -n 2 -o out/ 'a new image based on these'
  python promptbrush.py -s 3840x2160 'a wide landscape'   # 4K
  python promptbrush.py -m gpt-image-2.5-sunburst '…'     # best quality, ~2.5x slower

output: logs go to stderr; saved image paths go to stdout, one per line

configuration (environment variables first, then .env next to this script):
  {ENV_PREFIX}API_KEY   required
  {ENV_PREFIX}BASE_URL  default {DEFAULT_BASE_URL}

if no prompt is given, it is read from {PROMPT_FILE.name} (see prompt.example.md)

measured models (select with -m; time is per image):
""" + "\n".join(f"  {name:<24}{info.seconds:<8}{info.note}"
                for name, info in MODELS.items())


def _image_count(value: str) -> int:
    if not value.isdecimal() or not 1 <= int(value) <= MAX_N:
        raise argparse.ArgumentTypeError(f"expected a number from 1 to {MAX_N}, got {value!r}")
    return int(value)


def _image_size(value: str) -> str:
    if not SIZE_PATTERN.match(value):
        raise argparse.ArgumentTypeError(
            f"expected WIDTHxHEIGHT (e.g. 1024x1024, both ≥1) or auto, got {value!r}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Call an OpenAI-compatible image API: "
                    "generate mode without -i, edit mode with -i.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt", nargs="*",
        help=f"the prompt; read from {PROMPT_FILE.name} when omitted",
    )
    parser.add_argument(
        "-i", "--image", action="append", default=[], metavar="PATH",
        help="reference image file or directory (repeatable, one path per -i)",
    )
    parser.add_argument(
        "-s", "--size", type=_image_size, default=DEFAULT_SIZE,
        help=f"WIDTHxHEIGHT or auto (default: {DEFAULT_SIZE}). Rendered exactly "
             f"by gpt-image-2.5 models, which need multiples of {SIZE_MULTIPLE}, "
             f"sides ≤{MAX_EDGE}, aspect ratio ≤{MAX_ASPECT:.0f}:1 and "
             f"{MIN_PIXELS:,}–{MAX_PIXELS:,} pixels; gpt-image-2 uses only the aspect ratio",
    )
    parser.add_argument(
        "-n", "--num", type=_image_count, default=DEFAULT_N,
        help=f"number of images, 1 to {MAX_N} (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"model name (default: {DEFAULT_MODEL}); measured models are listed below",
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help="output directory (default: current directory)",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    # intermixed: options and prompt fragments can be mixed freely, as in
    # `a cat -n 2 wearing a hat`. Everything after `--` is prompt; we split
    # there ourselves because parse_intermixed_args in some Python versions
    # (3.10 in testing) errors out on `--`.
    if "--" not in argv:
        return build_parser().parse_intermixed_args(argv)
    k = argv.index("--")
    args = build_parser().parse_intermixed_args(argv[:k])
    args.prompt += argv[k + 1:]
    return args


def _check_output_dir(path: str) -> None:
    """Make sure the output directory can be written to, or created: found out only
    after the request, the images would be paid for and lost. Nothing is created
    yet, so a run that saves nothing leaves no empty directory behind."""
    out = Path(path).absolute()
    existing = next(p for p in (out, *out.parents) if p.exists())
    if not existing.is_dir():
        raise ValueError(f"Output path {existing} exists and is not a directory")
    if not os.access(existing, os.W_OK | os.X_OK):
        raise ValueError(f"Can't write to {existing}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)  # on bad arguments argparse prints usage and exits with 2
    options = {"model": args.model, "size": args.size, "n": args.num,
               "output_dir": args.output}
    try:
        check_size(args.model, args.size)
        _check_output_dir(args.output)
        prompt = " ".join(args.prompt).strip() or load_prompt()
        if args.image:
            # Fail on invalid paths instead of falling back to generate mode or
            # dropping reference images: the user meant to edit, and silently
            # changing the request would waste paid credits
            ref_images = find_reference_images(args.image)
            saved = edit_image(prompt, ref_images, **options)
        else:
            saved = generate_image(prompt, **options)
    # Network errors are already logged in the request threads; all that's left
    # here are size errors and failures reading the key, prompt or reference images
    except (OSError, ValueError) as exc:
        log("error", str(exc))
        return 1
    except KeyboardInterrupt:
        log("warn", "Cancelled")
        return 130

    if not saved:
        log("warn", "No images were generated")
        return 1
    log("ok", f"Done: saved {len(saved)} image{'s' if len(saved) > 1 else ''}")
    for path in saved:
        print(path)
    return 0


def cli() -> None:
    """Entry point of the installed ``promptbrush`` command."""
    # Pipes on Windows (and legacy locales) use a code page that can't encode
    # every character in the help text or the logs; print "?" instead of crashing
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
