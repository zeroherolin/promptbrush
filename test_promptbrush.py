"""Unit tests and local integration tests for promptbrush.py.

Run: python -m unittest -v test_promptbrush

Isolation is enforced by setUpModule rather than left to each test:
  - ENV_FILE / PROMPT_FILE point into an empty temp directory and every
    PROMPTBRUSH_* environment variable is cleared, so the real key, base URL
    and prompt are out of reach;
  - every network connection except to loopback is blocked and fails the
    test, so a paid API can never be called by accident.
Tests that run a subprocess aren't covered by these patches; each one explains
in a comment why it is safe.
"""

import base64
import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import requests
import urllib3.util.connection

import promptbrush as pb

PROJECT_DIR = Path(pb.__file__).parent

# Minimal PNG header with an IHDR chunk: 640x480
PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
       + (640).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "
GIF = b"GIF89a" + b"\x00" * 8


# ---------------------------------------------------------------------------
# Global isolation
# ---------------------------------------------------------------------------
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_blocked_hosts: list[str] = []


def setUpModule():
    sandbox = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(sandbox.cleanup)
    real_connect = urllib3.util.connection.create_connection

    def loopback_only(address, *args, **kwargs):
        if address[0] not in _LOOPBACK:
            _blocked_hosts.append(address[0])
            # AssertionError rather than OSError: an OSError would be taken for
            # a network glitch and retried
            raise AssertionError(f"tests must not access the network: {address[0]}")
        return real_connect(address, *args, **kwargs)

    for target, attr, value in (
            (pb, "ENV_FILE", Path(sandbox.name) / ".env"),
            (pb, "PROMPT_FILE", Path(sandbox.name) / "prompt.md"),
            (urllib3.util.connection, "create_connection", loopback_only)):
        patcher = mock.patch.object(target, attr, value)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(pb.ENV_PREFIX)}
    env_patcher = mock.patch.dict(os.environ, clean_env, clear=True)
    env_patcher.start()
    unittest.addModuleCleanup(env_patcher.stop)
    # Worker threads may swallow the exception, so check once more at teardown
    unittest.addModuleCleanup(_assert_no_external_access)


def _assert_no_external_access():
    if _blocked_hosts:
        raise AssertionError(
            f"a test tried to access the network: {sorted(set(_blocked_hosts))}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def fake_response(status=200, *, json_body=None, text=None, headers=None,
                  content=b""):
    """Build a requests.Response stand-in good enough for promptbrush."""
    resp = mock.Mock(spec=requests.Response)
    resp.status_code = status
    resp.headers = headers or {}
    resp.content = content
    if json_body is not None:
        resp.text = json.dumps(json_body)
        resp.json.return_value = json_body
    else:
        resp.text = text if text is not None else ""
        resp.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", resp.text, 0)
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error")
    return resp


def parse_multipart(req: dict) -> tuple[dict[str, str], list]:
    """Parse a multipart request seen by the local server: (text fields, image parts)."""
    msg = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {req['headers']['Content-Type']}\r\n\r\n".encode()
        + req["body"])
    fields, images = {}, []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "image[]":
            images.append(part)
        else:
            # requests sends text fields without a charset; decode the raw UTF-8 bytes
            fields[name] = part.get_payload(decode=True).decode()
    return fields, images


class LogCaptureTestCase(unittest.TestCase):
    """Replace pb.log with a recorder and stdout with self.stdout, so tests can
    assert on the logs and on the image paths main prints, and test output
    stays clean."""

    def setUp(self):
        self.logs: list[tuple[str, str]] = []
        for patcher in (
                mock.patch.object(pb, "log", side_effect=lambda lv, msg:
                                  self.logs.append((lv, msg))),
                mock.patch("sys.stdout", new_callable=io.StringIO)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stdout = sys.stdout

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def logged(self, level: str) -> list[str]:
        return [msg for lv, msg in self.logs if lv == level]


class TestIsolation(unittest.TestCase):
    """Check that the global isolation above actually works."""

    def test_real_config_and_prompt_unreachable(self):
        self.assertNotEqual(pb.ENV_FILE, PROJECT_DIR / ".env")
        self.assertNotEqual(pb.PROMPT_FILE, PROJECT_DIR / "prompt.md")
        self.assertFalse([k for k in os.environ if k.startswith(pb.ENV_PREFIX)])
        with self.assertRaisesRegex(ValueError, "API key not configured"):
            pb.load_api_key()

    def test_external_network_blocked(self):
        with self.assertRaisesRegex(AssertionError, "must not access the network"):
            requests.get("http://example.com", timeout=1)
        _blocked_hosts.remove("example.com")  # expected block, not a violation


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class TestLog(unittest.TestCase):
    def capture(self, fn, *, color=False):
        with mock.patch.object(pb, "_USE_COLOR", color), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            fn()
        self.assertEqual(out.getvalue(), "")  # stdout is reserved for image paths
        return err.getvalue().splitlines()

    def test_plain_format_and_thread_tag(self):
        def in_thread():
            pb._job.tag = "2/3"
            pb.log("info", "saved")

        def run():
            pb.log("warn", "main thread")
            t = threading.Thread(target=in_thread)
            t.start()
            t.join()

        self.assertEqual(self.capture(run),
                         ["[warn] main thread", "[info] [2/3] saved"])

    def test_color_keeps_level_and_unknown_level_stays_plain(self):
        def run():
            pb.log("error", "broken")
            pb.log("debug", "details")

        self.assertEqual(self.capture(run, color=True),
                         ["\033[31m[error]\033[0m broken", "[debug] details"])

    def test_color_only_on_tty_without_no_color(self):
        tty, pipe = mock.Mock(), mock.Mock()
        tty.isatty.return_value, pipe.isatty.return_value = True, False
        base = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        cases = [(tty, {}, True), (tty, {"NO_COLOR": "1"}, False),
                 (tty, {"NO_COLOR": ""}, True),  # per the spec, empty means unset
                 (pipe, {}, False)]
        for stream, env, expected in cases:
            with self.subTest(tty=stream is tty, env=env), \
                    mock.patch("sys.stderr", stream), \
                    mock.patch.dict(os.environ, base | env, clear=True):
                self.assertIs(pb._supports_color(), expected)

    def test_concurrent_lines_not_interleaved(self):
        # Must write to a real pipe: StringIO is implemented in C and never
        # interleaves even without the lock, so it can't catch the bug.
        # The subprocess only calls log; it reads no key and never goes online.
        script = textwrap.dedent("""
            import sys, threading
            import promptbrush as pb
            sys.setswitchinterval(1e-6)
            def worker(k):
                for _ in range(1000):
                    pb.log("info", f"thread{k} " + "x" * 50)
            threads = [threading.Thread(target=worker, args=(k,))
                       for k in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        """)
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_DIR,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout, "")
        lines = proc.stderr.splitlines()
        self.assertEqual(len(lines), 8000)
        bad = [line for line in lines
               if not re.fullmatch(r"\[info\] thread\d x{50}", line)]
        self.assertEqual(bad, [])

    def test_sink_takes_over_current_thread_only(self):
        got = []

        def other_thread():
            pb.log("info", "other thread")

        def run():
            with pb.log_sink(lambda lv, msg: got.append((lv, msg))):
                pb._job.tag = "1/2"
                pb.log("retry", "retrying")
                with pb.log_sink(lambda lv, msg: got.append(("inner", msg))):
                    pb.log("info", "inner")
                pb.log("ok", "outer again")
                t = threading.Thread(target=other_thread)
                t.start()
                t.join()
                pb._job.tag = None
            pb.log("warn", "exited")

        self.assertEqual(self.capture(run), ["[info] other thread", "[warn] exited"])
        self.assertEqual(got, [("retry", "[1/2] retrying"), ("inner", "[1/2] inner"),
                               ("ok", "[1/2] outer again")])

    def test_brief_flattens_and_truncates(self):
        self.assertEqual(pb._brief("  a\nb\n "), "a b")
        self.assertEqual(pb._brief("x" * 4, limit=4), "xxxx")
        self.assertEqual(pb._brief("x" * 5, limit=4), "xxxx…")


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
class TestLoadFiles(LogCaptureTestCase):
    # Chinese text on purpose ("draw a cat"): its GBK bytes are invalid UTF-8,
    # which is what makes the fallback kick in
    def test_utf8(self):
        p = self.tmp / "a.md"
        p.write_text("画一只猫", encoding="utf-8")
        self.assertEqual(pb.read_text_file(p), "画一只猫")

    def test_gbk_fallback(self):
        p = self.tmp / "a.md"
        p.write_bytes("画一只猫".encode("gbk"))
        self.assertEqual(pb.read_text_file(p), "画一只猫")

    def test_undecodable_raises_readable_value_error(self):
        p = self.tmp / "a.md"
        p.write_bytes(b"\xff\xfe\x80\x81" * 3)
        with self.assertRaises(ValueError) as cm:
            pb.read_text_file(p)
        self.assertIn("not UTF-8 or GBK", str(cm.exception))
        self.assertNotIn("codec", str(cm.exception))

    def test_load_prompt(self):
        prompt_file = self.tmp / "prompt.md"
        with mock.patch.object(pb, "PROMPT_FILE", prompt_file):
            with self.assertRaises(FileNotFoundError):
                pb.load_prompt()
            prompt_file.write_text("\n  \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "is empty"):
                pb.load_prompt()
            prompt_file.write_text("  a cat\n", encoding="utf-8")
            self.assertEqual(pb.load_prompt(), "a cat")


class TestSettings(LogCaptureTestCase):
    def setUp(self):
        super().setUp()
        self.env_file = self.tmp / ".env"
        patcher = mock.patch.object(pb, "ENV_FILE", self.env_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_env_file_syntax(self):
        self.env_file.write_text(textwrap.dedent("""\
            # comment
            PROMPTBRUSH_API_KEY=sk-plain

            export PROMPTBRUSH_BASE_URL = "https://a.example/v1"
            QUOTED='with = sign'
            MISMATCHED="half'
            EMPTY=
            not a pair
            INLINE=sk-abc  # trailing comment
            QUOTED_HASH="a # b" # a # inside quotes is not a comment
            HASH_NO_SPACE=a#b
            ONLY_COMMENT= # nothing but a comment
            HASH_FIRST=#abc
        """), encoding="utf-8")
        self.assertEqual(pb.read_env_file(self.env_file), {
            "PROMPTBRUSH_API_KEY": "sk-plain",
            "PROMPTBRUSH_BASE_URL": "https://a.example/v1",
            "QUOTED": "with = sign",
            "MISMATCHED": "\"half'",
            "EMPTY": "",
            "INLINE": "sk-abc",
            "QUOTED_HASH": "a # b",
            "HASH_NO_SPACE": "a#b",
            "ONLY_COMMENT": "",
            "HASH_FIRST": "#abc",
        })

    def test_missing_env_file_is_empty(self):
        self.assertEqual(pb.read_env_file(self.tmp / "nope"), {})

    def test_env_var_overrides_file_and_blank_counts_as_unset(self):
        self.env_file.write_text("PROMPTBRUSH_API_KEY=from-file\n")
        self.assertEqual(pb.load_api_key(), "from-file")
        with mock.patch.dict(os.environ, {"PROMPTBRUSH_API_KEY": " from-env "}):
            self.assertEqual(pb.load_api_key(), "from-env")
        with mock.patch.dict(os.environ, {"PROMPTBRUSH_API_KEY": ""}):
            self.assertEqual(pb.load_api_key(), "from-file")

    def test_missing_or_blank_key_explains_how_to_configure(self):
        for content in ("", "PROMPTBRUSH_API_KEY=   \n"):
            self.env_file.write_text(content)
            with self.subTest(content=content), \
                    self.assertRaisesRegex(ValueError, "PROMPTBRUSH_API_KEY"):
                pb.load_api_key()

    def test_endpoints_default_and_trailing_slash(self):
        self.assertEqual(pb.api_endpoints(), (
            "https://api.openai.com/v1/images/generations",
            "https://api.openai.com/v1/images/edits"))
        self.env_file.write_text("PROMPTBRUSH_BASE_URL=https://p.example/v1/\n")
        self.assertEqual(pb.api_endpoints()[1],
                         "https://p.example/v1/images/edits")
        # an explicitly passed URL wins over the config
        self.assertEqual(pb.api_endpoints("https://q.example/v1/")[0],
                         "https://q.example/v1/images/generations")


class TestFindReferenceImages(LogCaptureTestCase):
    def test_files_dirs_dedupe_and_filtering(self):
        refs = self.tmp / "refs"
        refs.mkdir()
        for name in ("b.PNG", "a.jpg", "notes.txt", ".DS_Store"):
            (refs / name).write_bytes(b"x")
        (refs / "fake.png").mkdir()  # a subdirectory named like an image is ignored
        single = self.tmp / "c.webp"
        single.write_bytes(b"x")

        found = pb.find_reference_images(
            [str(single), str(refs), str(refs / "a.jpg")])

        self.assertEqual([p.name for p in found], ["c.webp", "a.jpg", "b.PNG"])
        self.assertEqual(self.logs, [])

    def test_any_invalid_path_rejected_listing_every_problem(self):
        # With one reference image missing the result is wrong, so never send a
        # paid request with the rest
        good = self.tmp / "ok.png"
        good.write_bytes(b"x")
        pdf = self.tmp / "doc.pdf"
        pdf.write_bytes(b"x")
        empty = self.tmp / "empty"
        empty.mkdir()
        (empty / "notes.txt").write_bytes(b"x")

        with self.assertRaises(ValueError) as cm:
            pb.find_reference_images(
                [str(good), "missing.png", str(pdf), str(empty)])
        msg = str(cm.exception)
        for part in ("missing.png (not found)", "doc.pdf (unsupported format)",
                     "empty (no images in directory)", ".webp"):
            self.assertIn(part, msg)
        self.assertNotIn("ok.png", msg)


class TestImageBytes(unittest.TestCase):
    def test_known_formats(self):
        for data, ext in ((PNG, ".png"), (JPG, ".jpg"), (WEBP, ".webp"),
                          (GIF, ".gif"), (b"GIF87a" + b"\x00" * 8, ".gif")):
            with self.subTest(ext=ext):
                self.assertEqual(pb.detect_image_ext(data), ext)

    def test_unknown(self):
        self.assertIsNone(pb.detect_image_ext(b"<html>404</html>"))
        self.assertIsNone(pb.detect_image_ext(b""))

    def test_png_dimensions(self):
        self.assertEqual(pb.png_dimensions(PNG), (640, 480))
        self.assertIsNone(pb.png_dimensions(JPG))
        self.assertIsNone(pb.png_dimensions(b"\x89PNG\r\n\x1a\n"))  # truncated


class TestWriteUnique(LogCaptureTestCase):
    def test_sequential_names_never_overwrite(self):
        (self.tmp / "img_1.png").write_bytes(b"old")
        names = [pb.write_unique(self.tmp, "img", ".png", bytes([k])).name
                 for k in range(3)]
        self.assertEqual(names, ["img_0.png", "img_2.png", "img_3.png"])
        self.assertEqual((self.tmp / "img_1.png").read_bytes(), b"old")

    def test_concurrent_writers_get_distinct_files(self):
        barrier = threading.Barrier(16)
        paths = []

        def write(k):
            barrier.wait()
            paths.append(pb.write_unique(self.tmp, "img", ".png", bytes([k])))

        threads = [threading.Thread(target=write, args=(k,)) for k in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(paths)), 16)
        self.assertEqual(sorted(p.read_bytes() for p in paths),
                         [bytes([k]) for k in range(16)])


# ---------------------------------------------------------------------------
# Command-line parsing
# ---------------------------------------------------------------------------
class TestParseArgs(unittest.TestCase):
    def parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as cm:
            pb.parse_args(argv)
        self.assertEqual(cm.exception.code, 2)
        return err.getvalue()

    def test_defaults(self):
        args = pb.parse_args([])
        self.assertEqual(args.prompt, [])
        self.assertEqual(args.image, [])
        self.assertEqual(args.size, pb.DEFAULT_SIZE)
        self.assertEqual(args.num, pb.DEFAULT_N)
        self.assertEqual(args.model, pb.DEFAULT_MODEL)
        self.assertEqual(args.output, pb.DEFAULT_OUTPUT_DIR)

    def test_options_intermixed_with_prompt(self):
        args = pb.parse_args(
            ["a cat", "-n", "2", "in a hat", "-i", "a.png", "--image=b.png",
             "--size=auto", "-m", "m1", "-o", "out"])
        self.assertEqual(args.prompt, ["a cat", "in a hat"])
        self.assertEqual(args.image, ["a.png", "b.png"])
        self.assertEqual((args.num, args.size, args.model, args.output),
                         (2, "auto", "m1", "out"))

    def test_double_dash_allows_prompt_starting_with_dash(self):
        self.assertEqual(pb.parse_args(["--", "-odd start"]).prompt,
                         ["-odd start"])
        args = pb.parse_args(["a cat", "-n", "2", "--", "-s", "--", "x"])
        self.assertEqual((args.prompt, args.num, args.size),
                         (["a cat", "-s", "--", "x"], 2, pb.DEFAULT_SIZE))
        self.parse_error(["-n", "--", "x"])  # option without a value: still an error

    def test_invalid_num(self):
        for bad in ("0", "-2", "abc", "1.5", ""):
            with self.subTest(n=bad):
                self.assertIn("expected a positive integer",
                              self.parse_error(["-n", bad, "x"]))

    def test_invalid_size(self):
        for bad in ("1024", "1024*1024", "big", "x1024", "0x0", "0x1024",
                    "1024x0", "01x5"):
            with self.subTest(size=bad):
                self.assertIn("WIDTHxHEIGHT", self.parse_error(["-s", bad, "x"]))

    def test_typo_option_rejected_not_sent_as_prompt(self):
        self.assertIn("--szie", self.parse_error(["--szie", "1024x1024", "x"]))

    def test_missing_option_value(self):
        self.parse_error(["a cat", "-i"])


class TestCheckSize(LogCaptureTestCase):
    STRICT = "gpt-image-2.5-flare"

    def test_default_model_is_strict(self):
        self.assertTrue(pb.DEFAULT_MODEL.startswith(pb.STRICT_SIZE_PREFIX))
        pb.check_size(pb.DEFAULT_MODEL, pb.DEFAULT_SIZE)  # the defaults must be valid

    def test_strict_accepts_measured_valid_sizes(self):
        # All measured to work, including the boundary of every rule
        for size in ("auto", "1024x1024", "1600x1200", "3840x2160",
                     "2160x3840", "3840x1600", "2976x992", "1024x640",
                     "816x816"):
            with self.subTest(size=size):
                pb.check_size(self.STRICT, size)
        self.assertEqual(self.logs, [])

    def test_strict_rejects_measured_invalid_sizes(self):
        # All measured to be rejected by the server, each with the rule it breaks
        cases = {
            "1000x1000": "multiples of 16",
            "1600x1208": "multiples of 16",
            "4000x1504": "each side must be at most 3840",
            "2992x992": "aspect ratio must be at most 3:1",
            "800x800": "total pixels",
            "3840x3840": "total pixels",
        }
        for size, reason in cases.items():
            with self.subTest(size=size), \
                    self.assertRaisesRegex(ValueError, reason) as cm:
                pb.check_size(self.STRICT, size)
            self.assertIn(size, str(cm.exception))

    def test_strict_lists_every_violation(self):
        with self.assertRaises(ValueError) as cm:
            pb.check_size("gpt-image-2.5-sunburst", "10000x1001")
        msg = str(cm.exception)
        for reason in ("multiples of 16", "each side must be at most",
                       "aspect ratio must be at most", "total pixels"):
            self.assertIn(reason, msg)

    def test_lenient_models_only_warn(self):
        for model in ("gpt-image-2", "gpt-image-2-vip", "gemini-3-pro-image"):
            self.logs.clear()
            with self.subTest(model=model):
                pb.check_size(model, "1000x1000")
                self.assertEqual(self.logs, [])
                pb.check_size(model, "4000x1000")
                self.assertIn("exceeds 3:1", self.logged("warn")[0])


class TestModels(unittest.TestCase):
    def test_defaults_are_measured_models(self):
        for name in (pb.DEFAULT_MODEL, pb.LENIENT_MODERATION_MODEL):
            self.assertIn(name, pb.MODELS)

    def test_fields_consistent(self):
        for name, info in pb.MODELS.items():
            with self.subTest(model=name):
                self.assertIn(info.size, {"exact", "aspect", "fixed"})
                # Models fixed at 1024x1024 reframe the shot, so they can't be aligned
                if info.size == "fixed":
                    self.assertFalse(info.aligned)
                if info.mask:
                    self.assertTrue(info.aligned)

    def test_listed_in_help(self):
        help_text = pb.build_parser().format_help()
        for name in pb.MODELS:
            self.assertIn(name, help_text)


class TestFitSize(unittest.TestCase):
    def test_valid_sizes_kept(self):
        for size in ("1024x1024", "1536x1024", "1600x1200", "3840x2160",
                     "2976x992", "816x816"):
            with self.subTest(size=size):
                self.assertEqual(pb.fit_size(*map(int, size.split("x"))), size)

    def test_result_always_valid_and_keeps_ratio(self):
        for w in (1, 7, 100, 333, 640, 1000, 1179, 1920, 2556, 4032, 9000):
            for h in (1, 9, 100, 480, 1080, 1333, 3024, 5000):
                with self.subTest(w=w, h=h):
                    fw, fh = map(int, pb.fit_size(w, h).split("x"))
                    self.assertEqual(pb._size_problems(fw, fh), [])
                    target = min(max(w / h, 1 / pb.MAX_ASPECT), pb.MAX_ASPECT)
                    self.assertAlmostEqual(fw / fh / target, 1, delta=0.03)

    def test_scales_into_pixel_range(self):
        self.assertEqual(pb.fit_size(4032, 3024), "3312x2480")  # phone photo shrunk
        self.assertEqual(pb.fit_size(512, 512), "816x816")      # small image enlarged

    def test_invalid(self):
        with self.assertRaises(ValueError):
            pb.fit_size(0, 100)


class TestCliEntry(unittest.TestCase):
    """Run the script entry point in a real subprocess to check that exit codes
    make it out through sys.exit.

    Both cases exit during argument parsing, before reading a key or sending
    any request.
    """

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "promptbrush.py", *args], cwd=PROJECT_DIR,
            capture_output=True, text=True, timeout=30)

    def test_help(self):
        proc = self.run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("examples:", proc.stdout)

    def test_help_survives_a_narrow_output_encoding(self):
        # Windows pipes use an ANSI code page that can't encode every character
        # in the help text; the CLI must print "?" instead of crashing
        proc = subprocess.run(
            [sys.executable, "promptbrush.py", "--help"], cwd=PROJECT_DIR,
            env={**os.environ, "PYTHONIOENCODING": "ascii", "PYTHONUTF8": "0"},
            capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(b"examples:", proc.stdout)

    def test_bad_argument(self):
        proc = self.run_cli("-n", "0", "x")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("expected a positive integer", proc.stderr)


# ---------------------------------------------------------------------------
# Network retries
# ---------------------------------------------------------------------------
class TestRequestWithRetry(LogCaptureTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(pb.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def sleeps(self):
        return [c.args[0] for c in self.sleep.call_args_list]

    def test_success_first_try_passes_arguments_through(self):
        with mock.patch.object(pb.requests, "request",
                               return_value=fake_response(200)) as req:
            resp = pb.request_with_retry("POST", "u", json={"a": 1}, timeout=5)
        self.assertEqual(resp.status_code, 200)
        req.assert_called_once_with("POST", "u", json={"a": 1}, timeout=5)
        self.assertEqual(self.sleeps(), [])

    def test_retryable_status_then_success(self):
        seq = [fake_response(502), fake_response(524), fake_response(200)]
        with mock.patch.object(pb.requests, "request", side_effect=seq) as req:
            self.assertEqual(pb.request_with_retry("POST", "u").status_code, 200)
        self.assertEqual(req.call_count, 3)
        self.assertEqual(self.sleeps(), [2.0, 4.0])

    def test_non_retryable_status_returned_immediately(self):
        with mock.patch.object(pb.requests, "request",
                               return_value=fake_response(400)) as req:
            self.assertEqual(pb.request_with_retry("POST", "u").status_code, 400)
        self.assertEqual(req.call_count, 1)

    def test_exhausted_returns_last_response(self):
        with mock.patch.object(pb.requests, "request",
                               return_value=fake_response(503)) as req:
            self.assertEqual(pb.request_with_retry("POST", "u").status_code, 503)
        self.assertEqual(req.call_count, pb.MAX_RETRIES)
        self.assertEqual(self.sleeps(), [2.0, 4.0, 8.0, 16.0])

    def test_retry_after_respected_and_capped(self):
        seq = [fake_response(429, headers={"Retry-After": "30"}),
               fake_response(429, headers={"Retry-After": "9999"}),
               fake_response(429, headers={"Retry-After": "1"}),
               fake_response(429, headers={"Retry-After": "Wed, 21 Oct 2015"}),
               fake_response(200)]
        with mock.patch.object(pb.requests, "request", side_effect=seq):
            pb.request_with_retry("POST", "u")
        # max(backoff, min(Retry-After, cap)); the date form falls back to backoff
        self.assertEqual(self.sleeps(), [30.0, pb.MAX_RETRY_AFTER, 8.0, 16.0])

    def test_connection_errors_recover(self):
        seq = [requests.exceptions.ConnectionError("reset"),
               requests.exceptions.ChunkedEncodingError("broken"),
               fake_response(200)]
        with mock.patch.object(pb.requests, "request", side_effect=seq):
            self.assertEqual(pb.request_with_retry("POST", "u").status_code, 200)
        self.assertEqual(len(self.logged("retry")), 2)

    def test_connection_errors_exhausted_raise_last(self):
        err = requests.exceptions.ReadTimeout("slow")
        with mock.patch.object(pb.requests, "request", side_effect=err) as req:
            with self.assertRaises(requests.exceptions.ReadTimeout):
                pb.request_with_retry("POST", "u")
        self.assertEqual(req.call_count, pb.MAX_RETRIES)
        self.assertTrue(self.logged("error"))

    def test_other_request_exceptions_not_retried(self):
        err = requests.exceptions.InvalidURL("bad")
        with mock.patch.object(pb.requests, "request", side_effect=err) as req:
            with self.assertRaises(requests.exceptions.InvalidURL):
                pb.request_with_retry("POST", "u")
        self.assertEqual(req.call_count, 1)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
class TestDecodeAndSave(LogCaptureTestCase):
    """decode_images extracts the image bytes and save_images writes them;
    chained, they are the final step of the CLI."""

    @staticmethod
    def save(result: dict, output_dir: str) -> list[str]:
        return pb.save_images(pb.decode_images(result), output_dir)

    def test_b64_formats_get_correct_extension(self):
        result = {"data": [{"b64_json": b64(PNG)}, {"b64_json": b64(JPG)},
                           {"b64_json": b64(WEBP)}]}
        saved = self.save(result, str(self.tmp / "out"))
        self.assertEqual([Path(p).suffix for p in saved],
                         [".png", ".jpg", ".webp"])
        self.assertEqual(Path(saved[1]).read_bytes(), JPG)

    def test_creates_nested_output_dir(self):
        out = self.tmp / "a" / "b"
        saved = self.save({"data": [{"b64_json": b64(PNG)}]}, str(out))
        self.assertTrue(Path(saved[0]).is_relative_to(out))

    def test_same_second_runs_do_not_overwrite(self):
        result = {"data": [{"b64_json": b64(PNG)}, {"b64_json": b64(PNG)}]}
        with mock.patch.object(pb, "datetime") as dt:
            dt.now.return_value.strftime.return_value = "20260101_000000"
            first = self.save(result, str(self.tmp))
            second = self.save(result, str(self.tmp))
        self.assertEqual([Path(p).name for p in first + second],
                         [f"image_20260101_000000_{k}.png" for k in range(4)])

    def test_saved_log_shows_actual_png_size(self):
        self.save({"data": [{"b64_json": b64(PNG)}]}, str(self.tmp))
        self.assertTrue(self.logged("ok")[0].endswith("(640x480)"))

    def test_url_download(self):
        with mock.patch.object(pb.requests, "request",
                               return_value=fake_response(200, content=PNG)) as req:
            saved = self.save({"data": [{"url": "http://x/1"}]},
                                    str(self.tmp))
        req.assert_called_once_with("GET", "http://x/1",
                                    timeout=pb.TIMEOUT_DOWNLOAD)
        self.assertEqual(Path(saved[0]).read_bytes(), PNG)

    def test_url_download_retried_on_transient_failure(self):
        seq = [requests.exceptions.ConnectionError("reset"),
               fake_response(502), fake_response(200, content=PNG)]
        with mock.patch.object(pb.requests, "request", side_effect=seq) as req, \
                mock.patch.object(pb.time, "sleep"):
            saved = self.save({"data": [{"url": "http://x/1"}]},
                                    str(self.tmp))
        self.assertEqual(req.call_count, 3)
        self.assertEqual(Path(saved[0]).read_bytes(), PNG)

    def test_failed_item_skipped_others_kept(self):
        bad_download = fake_response(404, content=b"<html>404</html>")
        result = {"data": [{"url": "http://x/gone"},
                           {"b64_json": "!!!not base64"},
                           {},
                           {"b64_json": b64(PNG)}]}
        with mock.patch.object(pb.requests, "request",
                               return_value=bad_download) as req:
            saved = self.save(result, str(self.tmp))
        self.assertEqual(req.call_count, 1)  # 404 is not retried
        self.assertEqual(len(saved), 1)
        self.assertEqual(Path(saved[0]).read_bytes(), PNG)
        self.assertEqual(len(self.logged("error")), 2)  # 404 + bad base64
        self.assertEqual(len(self.logged("warn")), 1)   # empty item
        self.assertEqual(list(self.tmp.glob("*.png")), [Path(saved[0])])

    def test_malformed_fields_do_not_lose_valid_images(self):
        result = {"usage": 5, "data": ["oops", None, {"b64_json": b64(PNG)}]}
        saved = self.save(result, str(self.tmp))
        self.assertEqual([Path(p).read_bytes() for p in saved], [PNG])
        self.assertEqual(len(self.logged("warn")), 2)

    def test_data_not_a_list(self):
        self.assertEqual(self.save({"data": {"url": "x"}}, str(self.tmp)),
                         [])
        self.assertIn("No image data received", self.logged("warn"))

    def test_unrecognized_bytes_saved_as_bin(self):
        saved = self.save({"data": [{"b64_json": b64(b"mystery")}]},
                                str(self.tmp))
        self.assertTrue(saved[0].endswith(".bin"))
        self.assertTrue(self.logged("warn"))

    def test_empty_data(self):
        self.assertEqual(self.save({"data": []}, str(self.tmp)), [])
        self.assertIn("No image data received", self.logged("warn"))
        self.assertFalse(any(self.tmp.iterdir()))  # no files created without images


# ---------------------------------------------------------------------------
# Request orchestration
# ---------------------------------------------------------------------------
class TestRequestAndSave(LogCaptureTestCase):
    def run_with(self, resp, model=pb.DEFAULT_MODEL):
        with mock.patch.object(pb, "request_with_retry", return_value=resp) as req:
            saved = pb._request_and_save("u", str(self.tmp), model=model)
        req.assert_called_once_with("POST", "u")  # model must not reach requests
        return saved

    def test_http_error_body_truncated(self):
        self.assertEqual(self.run_with(fake_response(401, text="x" * 1000)), [])
        self.assertEqual(self.logged("error"),
                         ["HTTP 401: " + "x" * pb.ERROR_BODY_LEN + "…"])
        self.assertEqual(self.logged("warn"), [])

    def test_moderation_451_suggests_alternative(self):
        body = {"error": {"message": "The generated images appear to be unsafe"}}
        self.assertEqual(self.run_with(fake_response(451, json_body=body)), [])
        self.assertTrue(self.logged("error")[0].startswith("HTTP 451"))
        self.assertIn(f"-m {pb.LENIENT_MODERATION_MODEL}", self.logged("warn")[0])

    def test_moderation_451_on_lenient_model_does_not_suggest_itself(self):
        body = {"error": {"message": "unsafe"}}
        self.run_with(fake_response(451, json_body=body),
                      model=pb.LENIENT_MODERATION_MODEL)
        (hint,) = self.logged("warn")
        self.assertIn("rephrasing the prompt", hint)
        self.assertNotIn("-m", hint)

    def test_non_json_200(self):
        self.assertEqual(self.run_with(fake_response(200, text="<html>")), [])
        self.assertIn("not valid JSON", self.logged("error")[0])

    def test_json_not_object(self):
        self.assertEqual(self.run_with(fake_response(200, json_body=[1])), [])
        self.assertIn("Unexpected response format", self.logged("error")[0])

    def test_error_field_in_200(self):
        body = {"error": {"message": "content policy"}}
        self.assertEqual(self.run_with(fake_response(200, json_body=body)), [])
        self.assertIn("content policy", self.logged("error")[0])

    def test_null_error_field_is_not_an_error(self):
        body = {"error": None, "data": [{"b64_json": b64(PNG)}]}
        self.assertEqual(len(self.run_with(fake_response(200, json_body=body))), 1)
        self.assertEqual(self.logged("error"), [])

    def test_usage_logged(self):
        body = {"data": [{"b64_json": b64(PNG)}],
                "usage": {"total_tokens": 7, "input_tokens": 3, "output_tokens": 4}}
        self.run_with(fake_response(200, json_body=body))
        self.assertIn("Usage: total=7 input=3 output=4", self.logged("info"))


class TestRequestParallel(LogCaptureTestCase):
    def test_splits_into_n_requests_and_collects(self):
        tags = []

        def fake(url, output_dir, **kw):
            tags.append(pb._job.tag)
            return [f"{output_dir}/{pb._job.tag}"]

        with mock.patch.object(pb, "_request_and_save", side_effect=fake) as rs:
            saved = pb._request_parallel("u", "out", 3, model="m", json={"n": 1})
        self.assertEqual(rs.call_count, 3)
        rs.assert_called_with("u", "out", model="m", json={"n": 1})
        self.assertEqual(sorted(tags), ["1/3", "2/3", "3/3"])
        self.assertEqual(saved, ["out/1/3", "out/2/3", "out/3/3"])  # collected in order

    def test_single_request_has_no_tag(self):
        def fake(url, output_dir, **kw):
            return [str(getattr(pb._job, "tag", None))]

        with mock.patch.object(pb, "_request_and_save", side_effect=fake):
            self.assertEqual(pb._request_parallel("u", "out", 1, model="m"),
                             ["None"])

    def test_one_failure_does_not_lose_others(self):
        effects = iter([["a"], requests.exceptions.ConnectionError("down"),
                        PermissionError("denied"), ["d"]])
        lock = threading.Lock()

        def fake(*a, **kw):
            with lock:
                effect = next(effects)
            if isinstance(effect, Exception):
                raise effect
            return effect

        with mock.patch.object(pb, "_request_and_save", side_effect=fake):
            saved = pb._request_parallel("u", "out", 4, model="m")
        self.assertEqual(sorted(saved), ["a", "d"])
        errors = self.logged("error")
        self.assertEqual(len(errors), 2)
        self.assertTrue(any(m.startswith("Network request failed") for m in errors))

    def test_concurrency_capped(self):
        # Every request blocks until released: done right, exactly MAX_PARALLEL
        # are in flight; with a broken cap the rest pour in during the wait.
        # Doesn't depend on how long requests take.
        total = pb.MAX_PARALLEL * 3
        active = 0
        lock = threading.Lock()
        release = threading.Event()

        def fake(*a, **kw):
            nonlocal active
            with lock:
                active += 1
            release.wait(10)
            with lock:
                active -= 1
            return ["x"]

        saved = []
        with mock.patch.object(pb, "_request_and_save", side_effect=fake):
            runner = threading.Thread(target=lambda: saved.extend(
                pb._request_parallel("u", "out", total, model="m")))
            runner.start()
            deadline = time.monotonic() + 5
            while active < pb.MAX_PARALLEL and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)
            with lock:
                in_flight = active
            release.set()
            runner.join(10)
        self.assertEqual(in_flight, pb.MAX_PARALLEL)
        self.assertEqual(len(saved), total)


class TestModeEndpoints(LogCaptureTestCase):
    def test_each_mode_uses_own_endpoint_and_timeout(self):
        # A wrong timeout goes unnoticed until a slow generation is mistaken
        # for a failure, so pin it down here
        img = self.tmp / "a.png"
        img.write_bytes(PNG)
        with mock.patch.object(pb, "load_api_key", return_value="k"), \
                mock.patch.object(pb, "_request_parallel", return_value=[]) as rp:
            pb.generate_image("x")
            pb.edit_image("x", [img])
        gen, edit = rp.call_args_list
        gen_url, edit_url = pb.api_endpoints()
        self.assertEqual((gen.args[0], gen.kwargs["timeout"]),
                         (gen_url, pb.TIMEOUT_GENERATE))
        self.assertEqual((edit.args[0], edit.kwargs["timeout"]),
                         (edit_url, pb.TIMEOUT_EDIT))
        self.assertEqual((gen.kwargs["model"], edit.kwargs["model"]),
                         (pb.DEFAULT_MODEL, pb.DEFAULT_MODEL))


class TestEditImageFiles(LogCaptureTestCase):
    def setUp(self):
        super().setUp()
        for patcher in (
                mock.patch.object(pb, "load_api_key", return_value="k"),
                mock.patch.object(pb, "_request_parallel", return_value=[])):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.request_parallel = pb._request_parallel

    def ref(self, name, data):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def test_format_detected_from_content_not_extension(self):
        refs = [self.ref("a.png", PNG), self.ref("b.webp", WEBP),
                self.ref("c.JPEG", JPG), self.ref("d.gif", GIF),
                self.ref("photo.png", JPG)]  # a .png that is really a JPEG
        pb.edit_image("x", refs)
        files = self.request_parallel.call_args.kwargs["files"]
        self.assertEqual([(f, name, mime) for f, (name, _, mime) in files], [
            ("image[]", "a.png", "image/png"),
            ("image[]", "b.webp", "image/webp"),
            ("image[]", "c.jpg", "image/jpeg"),
            ("image[]", "d.gif", "image/gif"),
            ("image[]", "photo.jpg", "image/jpeg"),
        ])
        self.assertEqual([data for _, (_, data, _) in files],
                         [PNG, WEBP, JPG, GIF, JPG])

    def test_non_image_content_rejected_before_any_request(self):
        refs = [self.ref("a.png", PNG), self.ref("broken.png", b"<html>404")]
        with self.assertRaisesRegex(ValueError, "broken.png"):
            pb.edit_image("x", refs)
        self.request_parallel.assert_not_called()


class TestCreateImage(LogCaptureTestCase):
    """Library API: arguments are validated before any request, and API errors
    are raised as exceptions."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(pb, "_post_images",
                                    return_value=pb.Generation([PNG], None, 1.0))
        self.post = patcher.start()
        self.addCleanup(patcher.stop)

    def test_generate_uses_explicit_key_and_base_url(self):
        gen = pb.create_image("a cat", size="1024x1024", api_key="k1",
                              base_url="https://p.example/v1")
        self.assertEqual(gen.images, [PNG])
        url = self.post.call_args.args[0]
        kw = self.post.call_args.kwargs
        self.assertEqual(url, "https://p.example/v1/images/generations")
        self.assertEqual(kw["headers"], {"Authorization": "Bearer k1"})
        self.assertEqual(kw["json"], {"model": pb.DEFAULT_MODEL, "prompt": "a cat",
                                      "n": 1, "size": "1024x1024"})
        self.assertEqual(kw["timeout"], pb.TIMEOUT_GENERATE)

    def test_edit_with_images_and_mask(self):
        pb.create_image("edit", images=[PNG, JPG], mask=PNG, api_key="k",
                        base_url="https://p.example/v1")
        url = self.post.call_args.args[0]
        kw = self.post.call_args.kwargs
        self.assertEqual(url, "https://p.example/v1/images/edits")
        self.assertEqual(kw["timeout"], pb.TIMEOUT_EDIT)
        self.assertEqual(kw["data"]["n"], "1")
        self.assertEqual([(f, name, mime) for f, (name, _, mime) in kw["files"]], [
            ("image[]", "image_1.png", "image/png"),
            ("image[]", "image_2.jpg", "image/jpeg"),
            ("mask", "mask.png", "image/png"),
        ])

    def test_invalid_input_rejected_before_any_request(self):
        cases = {
            "prompt must not be empty": dict(prompt="  "),
            "mask only works in edit mode": dict(mask=PNG),
            "mask must be a PNG": dict(images=[PNG], mask=JPG),
            "Reference image 2": dict(images=[PNG, b"<html>"]),
            "multiples of 16": dict(size="1000x1000"),
        }
        for reason, kwargs in cases.items():
            kwargs = {"prompt": "x", "api_key": "k"} | kwargs
            with self.subTest(reason=reason), \
                    self.assertRaisesRegex(ValueError, reason):
                pb.create_image(kwargs.pop("prompt"), **kwargs)
        self.post.assert_not_called()


class TestCreateImageErrors(LogCaptureTestCase):
    def test_api_errors_raised_with_status(self):
        cases = [(fake_response(451, json_body={"error": "unsafe"}), 451, True),
                 (fake_response(400, text="bad"), 400, False),
                 (fake_response(200, json_body={"error": "x"}), 200, False)]
        for resp, status, moderated in cases:
            with self.subTest(status=status), \
                    mock.patch.object(pb, "request_with_retry", return_value=resp), \
                    self.assertRaises(pb.ImageAPIError) as cm:
                pb.create_image("x", api_key="k")
            self.assertEqual((cm.exception.status, cm.exception.moderated),
                             (status, moderated))
        self.assertEqual(self.logged("warn"), [])  # no moderation hint in library calls


class TestLogRequest(LogCaptureTestCase):
    def test_long_prompt_truncated_and_refs_listed(self):
        prompt = "c" * (pb.PROMPT_PREVIEW_LEN + 5)
        pb._log_request("edit (edits)", "m", "auto", 2, prompt,
                        refs=[Path("/x/a.png"), Path("b.jpg")])
        self.assertEqual(self.logged("info"), [
            "Mode: edit (edits)",
            "Model: m | Size: auto | Count: 2",
            "Reference images: a.png, b.jpg",
            "Prompt: " + "c" * pb.PROMPT_PREVIEW_LEN + "...",
        ])

    def test_short_prompt_kept_and_no_refs_line(self):
        pb._log_request("generate (generations)", "m", "auto", 1, "a cat")
        self.assertEqual(self.logged("info")[-1], "Prompt: a cat")
        self.assertFalse(any(m.startswith("Reference images")
                             for m in self.logged("info")))


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------
class TestMain(LogCaptureTestCase):
    def setUp(self):
        super().setUp()
        self.prompt_file = self.tmp / "prompt.md"
        for target, value in (("PROMPT_FILE", self.prompt_file),
                              ("load_api_key", lambda: "k")):
            patcher = mock.patch.object(pb, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_invalid_image_paths_do_not_fall_back_to_generation(self):
        with mock.patch.object(pb, "generate_image") as gen, \
                mock.patch.object(pb, "edit_image") as edit:
            self.assertEqual(pb.main(["-i", "nope.png", "a cat"]), 1)
            ok = self.tmp / "ok.png"
            ok.write_bytes(PNG)
            # Some valid paths still mustn't mean carrying on with fewer references
            self.assertEqual(pb.main(["-i", str(ok), "-i", "typo.png", "a cat"]), 1)
        gen.assert_not_called()
        edit.assert_not_called()
        self.assertEqual(len(self.logged("error")), 2)
        self.assertTrue(all("Invalid reference image paths" in m
                            for m in self.logged("error")))

    def test_saved_paths_printed_to_stdout(self):
        with mock.patch.object(pb, "generate_image",
                               return_value=["a.png", "out/b.png"]):
            self.assertEqual(pb.main(["x"]), 0)
        self.assertEqual(self.stdout.getvalue(), "a.png\nout/b.png\n")

    def test_nothing_on_stdout_when_nothing_saved(self):
        for effect in ([], ValueError("broken")):
            kwargs = ({"return_value": effect} if isinstance(effect, list)
                      else {"side_effect": effect})
            with self.subTest(effect=effect), \
                    mock.patch.object(pb, "generate_image", **kwargs):
                self.assertEqual(pb.main(["x"]), 1)
        self.assertEqual(self.stdout.getvalue(), "")

    def test_prompt_from_file_and_options_forwarded(self):
        self.prompt_file.write_text("  prompt from the file\n", encoding="utf-8")
        with mock.patch.object(pb, "generate_image", return_value=["a.png"]) as gen:
            code = pb.main(["-n", "3", "-s", "auto", "-m", "m2", "-o", "out"])
        self.assertEqual(code, 0)
        gen.assert_called_once_with("prompt from the file", model="m2",
                                    size="auto", n=3, output_dir="out")

    def test_blank_cli_prompt_falls_back_to_file(self):
        self.prompt_file.write_text("fallback", encoding="utf-8")
        with mock.patch.object(pb, "generate_image", return_value=["a"]) as gen:
            pb.main(["  "])
        self.assertEqual(gen.call_args.args[0], "fallback")

    def test_missing_prompt_file(self):
        self.assertEqual(pb.main([]), 1)
        self.assertIn("No prompt given", self.logged("error")[0])

    def test_edit_mode_selected(self):
        img = self.tmp / "ref.png"
        img.write_bytes(PNG)
        with mock.patch.object(pb, "edit_image", return_value=["a"]) as edit:
            self.assertEqual(pb.main(["-i", str(img), "touch it up"]), 0)
        self.assertEqual(edit.call_args.args[:2], ("touch it up", [img]))

    def test_exit_codes(self):
        # Network errors are handled in the worker threads; only file errors
        # and Ctrl+C reach main
        cases = [
            ([], 1),
            (PermissionError("denied"), 1),
            (KeyboardInterrupt(), 130),
        ]
        for effect, code in cases:
            with self.subTest(effect=effect):
                kwargs = ({"return_value": effect} if isinstance(effect, list)
                          else {"side_effect": effect})
                with mock.patch.object(pb, "generate_image", **kwargs):
                    self.assertEqual(pb.main(["x"]), code)

    def test_lenient_model_warns_on_extreme_aspect_but_still_sends(self):
        cases = {"4000x1000": True, "1000x3500": True, "3000x1000": False,
                 "1x1": False, "auto": False}
        for size, warns in cases.items():
            self.logs.clear()
            with self.subTest(size=size), \
                    mock.patch.object(pb, "generate_image",
                                      return_value=["a"]) as gen:
                code = pb.main(["-m", "gpt-image-2", "-s", size, "x"])
                self.assertEqual(code, 0)
                self.assertEqual(gen.call_args.kwargs["size"], size)
                self.assertEqual(
                    any("exceeds 3:1" in m for m in self.logged("warn")), warns)

    def test_strict_model_rejects_bad_size_before_any_request(self):
        with mock.patch.object(pb, "generate_image") as gen, \
                mock.patch.object(pb, "edit_image") as edit:
            code = pb.main(["-s", "1000x1000", "x"])  # the default model is strict
        self.assertEqual(code, 1)
        gen.assert_not_called()
        edit.assert_not_called()
        self.assertIn("multiples of 16", self.logged("error")[0])


# ---------------------------------------------------------------------------
# Integration: a local HTTP server, exercising the real requests stack
# ---------------------------------------------------------------------------
class _FakeImageAPI(BaseHTTPRequestHandler):
    """Mimics the measured server behavior: POST returns an image url that the
    client then downloads with GET."""
    requests_seen: list[dict] = []
    downloads: list[str] = []
    status_plan: list[int] = []      # statuses for successive POSTs, then 200
    get_status_plan: list[int] = []  # statuses for successive GETs, then 200
    hold = threading.Event()         # when set, POST hangs like a long generation
    release = threading.Event()

    def log_message(self, *args):  # silence
        pass

    def _reply(self, status, body: bytes, ctype: str):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client already gone (Ctrl+C test)

    def do_POST(self):
        cls = type(self)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        cls.requests_seen.append(
            {"path": self.path, "headers": dict(self.headers), "body": body})
        if cls.hold.is_set():
            cls.release.wait(30)
        status = cls.status_plan.pop(0) if cls.status_plan else 200
        if status != 200:
            return self._reply(status, b'{"error":"busy"}', "application/json")
        host, port = self.server.server_address
        url = f"http://{host}:{port}/files/{len(cls.requests_seen)}.png"
        payload = {"data": [{"url": url}], "usage": {"total_tokens": 1}}
        self._reply(200, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        cls = type(self)
        cls.downloads.append(self.path)
        status = cls.get_status_plan.pop(0) if cls.get_status_plan else 200
        if status != 200:
            return self._reply(status, b"<html>error</html>", "text/html")
        self._reply(200, PNG, "image/png")


class TestIntegrationLocalServer(LogCaptureTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeImageAPI)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        _FakeImageAPI.release.set()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        super().setUp()
        _FakeImageAPI.requests_seen = []
        _FakeImageAPI.downloads = []
        _FakeImageAPI.status_plan = []
        _FakeImageAPI.get_status_plan = []
        _FakeImageAPI.hold.clear()
        _FakeImageAPI.release.clear()
        self.addCleanup(_FakeImageAPI.release.set)
        # Go through the real config path: environment variables point to a
        # throwaway key and the local server
        self.env = {"PROMPTBRUSH_API_KEY": "sk-local-test",
                    "PROMPTBRUSH_BASE_URL": self.base_url}
        env_patcher = mock.patch.dict(os.environ, self.env)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        sleep = mock.patch.object(pb.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        self.out = self.tmp / "out"

    def ref_image(self, name="a.png", data=PNG):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def test_generate_n_split_into_single_requests(self):
        code = pb.main(["draw a cat", "-n", "3", "-s", "1024x1024",
                        "-o", str(self.out)])
        self.assertEqual(code, 0)
        reqs = _FakeImageAPI.requests_seen
        self.assertEqual(len(reqs), 3)
        for req in reqs:
            self.assertEqual(req["path"], "/v1/images/generations")
            self.assertEqual(req["headers"]["Authorization"],
                             "Bearer sk-local-test")
            self.assertEqual(json.loads(req["body"]),
                             {"model": pb.DEFAULT_MODEL, "prompt": "draw a cat",
                              "n": 1, "size": "1024x1024"})
        self.assertEqual(len(_FakeImageAPI.downloads), 3)
        files = sorted(self.out.iterdir())
        self.assertEqual(len(files), 3)
        self.assertTrue(all(f.read_bytes() == PNG for f in files))

    def test_edit_multipart(self):
        a, b = self.ref_image("a.png", PNG), self.ref_image("b.jpg", JPG)
        # Non-ASCII prompt on purpose ("blend"): it must reach the server as UTF-8
        code = pb.main(["-i", str(a), "-i", str(b), "融合", "-o", str(self.out)])
        self.assertEqual(code, 0)
        (req,) = _FakeImageAPI.requests_seen
        self.assertEqual(req["path"], "/v1/images/edits")
        self.assertEqual(req["headers"]["Authorization"], "Bearer sk-local-test")
        fields, images = parse_multipart(req)
        self.assertEqual(fields, {"model": pb.DEFAULT_MODEL, "prompt": "融合",
                                  "n": "1", "size": pb.DEFAULT_SIZE})
        self.assertEqual([p.get_filename() for p in images], ["a.png", "b.jpg"])
        self.assertEqual([p.get_content_type() for p in images],
                         ["image/png", "image/jpeg"])
        self.assertEqual([p.get_payload(decode=True) for p in images], [PNG, JPG])

    def test_edit_n2_sends_two_complete_single_image_requests(self):
        img = self.ref_image()
        self.assertEqual(pb.main(["-i", str(img), "x", "-n", "2",
                                  "-o", str(self.out)]), 0)
        self.assertEqual(len(_FakeImageAPI.requests_seen), 2)
        for req in _FakeImageAPI.requests_seen:
            fields, images = parse_multipart(req)
            self.assertEqual(fields["n"], "1")
            self.assertEqual([p.get_payload(decode=True) for p in images], [PNG])
        self.assertEqual(len(list(self.out.iterdir())), 2)

    def test_retry_resends_full_multipart(self):
        img = self.ref_image()
        _FakeImageAPI.status_plan = [503]
        self.assertEqual(pb.main(["-i", str(img), "x", "-o", str(self.out)]), 0)
        self.assertEqual(len(_FakeImageAPI.requests_seen), 2)
        for req in _FakeImageAPI.requests_seen:
            _, images = parse_multipart(req)
            self.assertEqual([p.get_payload(decode=True) for p in images], [PNG])

    def test_download_retried_then_saved(self):
        _FakeImageAPI.get_status_plan = [502, 503]
        self.assertEqual(pb.main(["x", "-o", str(self.out)]), 0)
        self.assertEqual(len(_FakeImageAPI.downloads), 3)
        self.assertEqual([f.read_bytes() for f in self.out.iterdir()], [PNG])

    def test_expired_download_url_not_saved_as_image(self):
        _FakeImageAPI.get_status_plan = [404]
        self.assertEqual(pb.main(["x", "-o", str(self.out)]), 1)
        self.assertEqual(len(_FakeImageAPI.downloads), 1)
        self.assertFalse(self.out.exists())  # no empty dir when nothing was fetched

    def test_partial_failure_keeps_successful_images(self):
        _FakeImageAPI.status_plan = [400]  # one of the 3 requests is rejected
        self.assertEqual(pb.main(["x", "-n", "3", "-o", str(self.out)]), 0)
        self.assertEqual(len(list(self.out.iterdir())), 2)
        self.assertTrue(any("HTTP 400" in m for m in self.logged("error")))

    def test_server_error_reported(self):
        _FakeImageAPI.status_plan = [400]
        self.assertEqual(pb.main(["x", "-o", str(self.out)]), 1)
        self.assertTrue(any("HTTP 400" in m for m in self.logged("error")))
        self.assertFalse(self.out.exists())

    def test_missing_api_key_sends_nothing(self):
        del os.environ["PROMPTBRUSH_API_KEY"]
        self.assertEqual(pb.main(["x", "-o", str(self.out)]), 1)
        self.assertIn("API key", self.logged("error")[0])
        self.assertEqual(_FakeImageAPI.requests_seen, [])

    def test_library_edit_with_mask_returns_bytes_without_saving(self):
        gen = pb.create_image("swap the hat", images=[PNG], mask=PNG, size="1024x1024")
        self.assertEqual(gen.images, [PNG])
        self.assertEqual(gen.usage, {"total_tokens": 1})
        (req,) = _FakeImageAPI.requests_seen
        msg = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {req['headers']['Content-Type']}\r\n\r\n".encode()
            + req["body"])
        names = [part.get_param("name", header="content-disposition")
                 for part in msg.iter_parts()]
        self.assertEqual(names, ["model", "prompt", "n", "size", "image[]", "mask"])
        self.assertEqual(msg.get_payload()[-1].get_payload(decode=True), PNG)
        self.assertFalse(self.out.exists())

    def test_cli_stdout_is_saved_paths_and_logs_go_to_stderr(self):
        # The subprocess gets the throwaway key and local URL from environment
        # variables (which win over any real .env in the project), so it never
        # touches the real key or the external network
        proc = subprocess.run(
            [sys.executable, "promptbrush.py", "x", "-n", "2", "-o", str(self.out)],
            cwd=PROJECT_DIR, env={**os.environ, **self.env},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(proc.stdout.splitlines()),
                         sorted(str(f) for f in self.out.iterdir()))
        self.assertEqual(len(proc.stdout.splitlines()), 2)
        self.assertIn("[ok] Done: saved 2 images", proc.stderr)

    @unittest.skipIf(sys.platform == "win32",
                     "a child process can't be sent SIGINT on Windows")
    def test_ctrl_c_exits_promptly_while_requests_in_flight(self):
        # The subprocess gets the throwaway key and local URL from environment
        # variables (which win over any real .env in the project), so it never
        # touches the real key or the external network
        _FakeImageAPI.hold.set()
        proc = subprocess.Popen(
            [sys.executable, "promptbrush.py", "x", "-n", "2", "-o", str(self.out)],
            cwd=PROJECT_DIR, env={**os.environ, **self.env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.kill)  # no-op if it already exited
        deadline = time.monotonic() + 10
        while len(_FakeImageAPI.requests_seen) < 2:
            self.assertLess(time.monotonic(), deadline,
                            "requests never reached the local server")
            time.sleep(0.05)

        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=5)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(proc.returncode, 130, err)
        self.assertIn("Cancelled", err)
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
