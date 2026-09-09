#!/usr/bin/env python3
"""
codewalk — a local three-pane codebase walkthrough: tree | file | chat.

    python codewalk.py [ROOT] [--port 8765] [--model opus] [--no-open]

Left:   the repo file tree (from `git ls-files` when ROOT is a git repo), with a filter box.
Center: the open file, syntax-highlighted (Pygments, server-side), with line numbers and tabs.
Right:  a chat that shells out to the `claude` CLI, streaming the answer back.

Claude gets read-only tools (Read, Grep, Glob) confined to ROOT, so it pulls files up itself,
and steers the center pane with two directives:

    [[open:path/to/file.py:120-140|label]]     open that file, highlight and scroll to the range
    [[edit:path/to/file.py:120-140]]           propose a replacement, shown as a diff with Apply
    ...replacement lines...
    [[/edit]]

Standard library only, plus Pygments for highlighting (optional — falls back to plain text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import glob
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from pygments import lex
    from pygments.lexers import get_lexer_for_filename, TextLexer
    from pygments.token import STANDARD_TYPES
except ImportError:
    lex = None

MAX_FILES = 20000
MAX_BYTES = 4_000_000
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".ipynb_checkpoints", ".venv", "venv", "env", "dist", "build",
    ".tox", ".next", ".cache", "target", ".idea", ".vscode-test",
}
SKIP_EXT = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".o", ".a", ".class", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar",
    ".xz", ".bz2", ".7z", ".mp4", ".mov", ".mp3", ".wav", ".woff", ".woff2", ".ttf",
    ".eot", ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".h5",
    ".parquet", ".db", ".sqlite", ".lock",
}


# ----------------------------------------------------------------------------------
# the repository
# ----------------------------------------------------------------------------------

class Repo:
    """The file tree, plus tokenized reads of individual files."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root.rstrip(os.sep)) or self.root
        self._files: list[str] | None = None
        self._files_at = 0.0
        self._lock = threading.Lock()
        self.is_git = os.path.isdir(os.path.join(self.root, ".git"))

    # -- path safety -----------------------------------------------------------
    def resolve(self, rel: str) -> str | None:
        """Absolute path for a repo-relative path, or None if it escapes the root."""
        if not rel:
            return None
        candidate = os.path.realpath(os.path.join(self.root, rel))
        root = os.path.realpath(self.root)
        if candidate != root and not candidate.startswith(root + os.sep):
            return None
        return candidate

    # -- tree ------------------------------------------------------------------
    def files(self) -> list[str]:
        with self._lock:
            if self._files is not None and time.time() - self._files_at < 5:
                return self._files
            found = self._git_files() if self.is_git else None
            if found is None:
                found = self._walk_files()
            found.sort()
            self._files = found[:MAX_FILES]
            self._files_at = time.time()
            return self._files

    def _git_files(self) -> list[str] | None:
        try:
            out = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.root, capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        keep = []
        for rel in out.stdout.split("\n"):
            if not rel or os.path.splitext(rel)[1].lower() in SKIP_EXT:
                continue
            if any(part in SKIP_DIRS for part in rel.split("/")):
                continue
            if os.path.isfile(os.path.join(self.root, rel)):
                keep.append(rel)
        return keep

    def _walk_files(self) -> list[str]:
        keep = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in filenames:
                if fn.startswith(".") or os.path.splitext(fn)[1].lower() in SKIP_EXT:
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root).replace(os.sep, "/")
                keep.append(rel)
                if len(keep) >= MAX_FILES:
                    return keep
        return keep

    # -- one file --------------------------------------------------------------
    def read(self, rel: str) -> dict:
        rel = rel.lstrip("/")
        full = self.resolve(rel)
        if full is None:
            return {"error": "That path is outside the project."}
        if not os.path.isfile(full):
            return {"error": f"No such file: {rel}"}
        try:
            size = os.path.getsize(full)
            if size > MAX_BYTES:
                return {"error": f"{rel} is {size // 1024} KB — too large to display."}
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            return {"error": f"Could not read {rel}: {e}"}
        if "\0" in text[:4096]:
            return {"error": f"{rel} looks like a binary file."}
        return {
            "path": rel,
            "name": os.path.basename(rel),
            "digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "mtime": os.path.getmtime(full),
            "lines": tokenize(text, full),
        }

    def mtime(self, rel: str) -> float:
        full = self.resolve(rel)
        try:
            return os.path.getmtime(full) if full else 0.0
        except OSError:
            return 0.0

    def apply(self, rel: str, start: int, end: int, replacement: str, digest: str) -> dict:
        full = self.resolve(rel.lstrip("/"))
        if full is None or not os.path.isfile(full):
            return {"ok": False, "error": f"No such file: {rel}"}
        with open(full, encoding="utf-8") as fh:
            text = fh.read()
        if hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] != digest:
            return {"ok": False, "error": "The file changed since that suggestion. Reload and ask again."}
        old = text.split("\n")
        shown = len(old) - 1 if old and old[-1] == "" else len(old)
        if not (1 <= start <= end <= shown):
            return {"ok": False, "error": f"Lines {start}-{end} are outside this {shown}-line file."}
        body = "\n".join(old[: start - 1] + replacement.split("\n") + old[end:])
        try:
            shutil.copyfile(full, full + ".codewalk.bak")
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(body)
        except OSError as e:
            return {"ok": False, "error": f"Could not write the file: {e}"}
        return {
            "ok": True,
            "digest": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
            "added": len(replacement.split("\n")),
        }


def tokenize(text: str, path: str) -> list[list[dict]]:
    """Split into lines of {c: css-class, t: text}. Tokens never straddle a newline."""
    lines: list[list[dict]] = [[]]
    if lex is None:
        for i, ln in enumerate(text.split("\n")):
            if i:
                lines.append([])
            if ln:
                lines[-1].append({"c": "", "t": ln})
        if lines and not lines[-1]:
            lines.pop()
        return lines

    try:
        lexer = get_lexer_for_filename(path, stripnl=False)
    except Exception:
        lexer = TextLexer(stripnl=False)

    for ttype, value in lex(text, lexer):
        cls = css_class(ttype)
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append({"c": cls, "t": part})
    if lines and not lines[-1]:
        lines.pop()
    return lines


def css_class(ttype) -> str:
    t = ttype
    while t is not None:
        if t in STANDARD_TYPES:
            return STANDARD_TYPES[t]
        t = t.parent
    return ""


# ----------------------------------------------------------------------------------
# talking to claude
# ----------------------------------------------------------------------------------

RULES = """\
You are walking a researcher through the codebase rooted at {root}, which they are reading in a \
local three-pane dashboard: file tree, open file, and this conversation.
{inherited}
Use your Read, Grep and Glob tools freely to pull up whatever you need — do not ask them to paste \
code, and do not guess at contents you have not read. You have no other tools; you cannot run \
anything or write to disk.

Answer in plain prose. No headings, no bullet lists, no code fences, no preamble, no restating the \
question. A few short paragraphs at most unless they ask for depth. They are a strong engineer, so \
be concrete and skip the basics.

You drive their center pane with two directives, and you should use them constantly:

1. To point at code, write [[open:PATH:START-END|short label]] inline, exactly where you would \
otherwise have written "see model.py lines 120-140". PATH is relative to the project root. The pane \
opens that file, highlights the range and scrolls to it. A whole file is [[open:PATH|label]]. \
Several ranges in one file: [[open:PATH:10-20,44|label]]. Line numbers must be the real ones from \
the file you read.

2. To propose a change, write

[[edit:PATH:START-END]]
the complete replacement text for those lines
[[/edit]]

which the dashboard shows as a diff with an Apply button; the user decides. Give the full \
replacement for that line range, indented exactly as it must appear in the file. Only when they ask \
for a change, and one edit block per message.

3. To pace a walkthrough, end a step with

[[continue: what comes next]]

which the dashboard turns into a Continue button. Use it whenever you have more to say than fits in
one bite. A step is ONE idea: at most two short paragraphs and one to three [[open:...]] directives,
then stop and let them press it. Never deliver a whole tour in a single message — a reader cannot
follow at the speed you write. If there is nothing more to say, leave it out.

The user can attach code to a question. When they do, the exact text arrives above their message \
under a REFERENCES heading, with the path and line numbers. That attached code is what they are \
asking about — answer about it directly, and do not re-read those lines unless you need surrounding \
context.
{brief}"""


INHERITED_NOTE = """
This conversation is a fork of the terminal session where the user and you designed and built this
code together, so that history is already above — the decisions, the reasoning, the things you tried
and rejected. Draw on it. Do not re-derive choices that were already made there, and do not
reintroduce yourself as if you were meeting this code for the first time.
"""


def find_session(spec: str, context_dir: str) -> str | None:
    """Resolve a session id, or 'latest' -> the newest Claude Code session for context_dir."""
    if spec and spec != "latest":
        return spec
    proj = re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(context_dir))
    found = glob.glob(os.path.join(os.path.expanduser("~/.claude/projects"), proj, "*.jsonl"))
    if not found:
        return None
    newest = max(found, key=os.path.getmtime)
    return os.path.splitext(os.path.basename(newest))[0]


class Claude:
    """One `claude -p` session over the repo, resumed across turns.

    When `parent` is given, the first turn forks that session instead of starting cold, so the
    chat inherits the terminal conversation's context. Forking (rather than resuming in place)
    means the user's own terminal session is never written to.
    """

    def __init__(self, model: str, root: str, parent: str | None = None, brief: str = ""):
        self.model = model
        self.root = root
        self.parent = parent
        self.brief = (
            "\n\n=== YOUR BRIEF FOR THIS SESSION ===\n\n"
            + brief.strip()
            + "\n\nThis brief was written for this session specifically. Where it conflicts with the "
              "general guidance above, the brief wins.\n"
        ) if brief.strip() else ""

        self.session_id: str | None = None
        self.lock = threading.Lock()

    @property
    def inherited(self) -> str:
        return INHERITED_NOTE if self.parent else ""

    def _argv(self, prompt: str, system: str, mode: str) -> list[str]:
        argv = [
            "claude", "-p", prompt,
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--append-system-prompt", system,
            "--restricted",
            "--permission-prompts", "none",
            "--allowedTools", "Read", "Grep", "Glob",
            "--disallowedTools", "Write", "Edit", "MultiEdit", "NotebookEdit",
            "Bash", "WebFetch", "WebSearch", "Task",
        ]
        if self.model:
            argv += ["--model", self.model]
        if mode == "resume" and self.session_id:
            argv += ["--resume", self.session_id]
        elif mode == "fork" and self.parent:
            argv += ["--resume", self.parent, "--fork-session"]
        return argv

    def ask(self, prompt: str, system: str, emit):
        """Run one turn. emit(kind, payload) for 'delta' | 'tool' | 'notice' | 'error' | 'done'."""
        with self.lock:
            modes = []
            if self.session_id:
                modes.append("resume")
            elif self.parent:
                modes.append("fork")
            modes.append("fresh")

            for mode in modes:
                try:
                    ok = self._run(self._argv(prompt, system, mode), emit)
                except FileNotFoundError:
                    emit("error", "The `claude` CLI is not on PATH.")
                    return
                if ok:
                    if mode == "fork":
                        self.parent = None      # the fork is ours now; resume it from here
                    return
                if mode == "resume":
                    self.session_id = None
                    emit("notice", "Session expired — starting a fresh one.")
                elif mode == "fork":
                    self.parent = None
                    emit("notice", "Could not inherit the terminal session — starting cold.")
            emit("error", "Claude exited without an answer. Check the terminal for details.")

    def _run(self, argv, emit) -> bool:
        proc = subprocess.Popen(
            argv, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        got_text = False
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")

            if ev.get("session_id"):
                self.session_id = ev["session_id"]

            if kind == "stream_event":
                inner = ev.get("event", {})
                if inner.get("type") == "content_block_delta":
                    piece = inner.get("delta", {}).get("text") or ""
                    if piece:
                        got_text = True
                        emit("delta", piece)

            elif kind == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        emit("tool", describe_tool(block))
                    elif block.get("type") == "text" and not got_text and block.get("text"):
                        got_text = True
                        emit("delta", block["text"])

            elif kind == "result":
                if not got_text and ev.get("result"):
                    got_text = True
                    emit("delta", ev["result"])

        err = proc.stderr.read()
        proc.wait()
        if not got_text:
            sys.stderr.write(f"[codewalk] claude exited {proc.returncode}: {err.strip()[:500]}\n")
            return False
        emit("done", "")
        return True


def render_transcript(turns: list, numbered_from: int = 1) -> str:
    """The dashboard conversation, written so the terminal session can absorb it verbatim."""
    if not turns:
        return "(no exchanges — the dashboard was closed without asking anything)\n"
    out = [f"# Dashboard conversation — {len(turns)} exchange(s)", ""]
    for i, t in enumerate(turns, numbered_from):
        out.append(f"## {i}. User")
        if t["context"].strip():
            out.append("")
            out.append("> attached context:")
            for line in t["context"].strip().split("\n"):
                out.append("> " + line)
        out.append("")
        out.append(t["q"].strip())
        out.append("")
        out.append(f"## {i}. Claude")
        if t["tools"]:
            out.append("")
            out.append("_(read: " + "; ".join(t["tools"][:12]) + ")_")
        out.append("")
        out.append(t["a"].strip())
        out.append("")
    return "\n".join(out) + "\n"


def describe_tool(block: dict) -> str:
    """A one-line, human-readable note for the activity strip."""
    name = block.get("name", "tool")
    args = block.get("input", {}) or {}
    if name == "Read":
        return "Reading " + short(args.get("file_path", "?"))
    if name == "Grep":
        where = args.get("path") or args.get("glob") or ""
        return f"Searching for {args.get('pattern', '?')!r}" + (f" in {short(where)}" if where else "")
    if name == "Glob":
        return f"Listing {args.get('pattern', '?')}"
    return name


def short(path: str) -> str:
    parts = str(path).replace(os.sep, "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else str(path)


# ----------------------------------------------------------------------------------
# http
# ----------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "codewalk"
    repo: Repo
    claude: Claude
    context_label: str = ""
    handoff: bool = False
    transcript: list = []
    transcript_path: str | None = None
    handback = threading.Event()
    last_ping: float = 0.0
    sync_path: str | None = None
    synced_upto: int = 0
    first_question: str = ""

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _query(self) -> dict:
        if "?" not in self.path:
            return {}
        return {k: v[0] for k, v in urllib.parse.parse_qs(self.path.split("?", 1)[1]).items()}

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/ping":
            Handler.last_ping = time.time()      # the tab is still open (fetch() == GET)
            self._json({"ok": True})
        elif path == "/api/tree":
            self._json({
                "root": self.repo.root,
                "name": self.repo.name,
                "git": self.repo.is_git,
                "context": Handler.context_label,
                "handoff": Handler.handoff,
                "first_question": Handler.first_question,
                "sync": bool(Handler.sync_path),
                "files": self.repo.files(),
            })
        elif path == "/api/file":
            res = self.repo.read(self._query().get("path", ""))
            self._json(res, 200 if "error" not in res else 404)
        elif path == "/api/watch":
            rel = self._query().get("path", "")
            try:
                seen = float(self._query().get("mtime", "0"))
            except ValueError:
                seen = 0.0
            deadline = time.time() + 25
            while time.time() < deadline:
                now = self.repo.mtime(rel)
                if now and now != seen:
                    res = self.repo.read(rel)
                    if "error" not in res:
                        self._json({"changed": True, **res})
                        return
                time.sleep(0.4)
            self._json({"changed": False})
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/apply":
            b = self._body()
            try:
                res = self.repo.apply(
                    str(b["path"]), int(b["start"]), int(b["end"]),
                    str(b["text"]), str(b.get("digest", "")),
                )
            except (KeyError, TypeError, ValueError):
                res = {"ok": False, "error": "Malformed edit."}
            self._json(res)
        elif path == "/api/sync":
            b = self._body()
            self._json(Handler.sync(bool(b.get("close"))))
            if b.get("close"):
                Handler.handback.set()
        elif path == "/api/handback":
            self._json({"ok": True, "turns": len(Handler.transcript)})
            Handler.handback.set()
        elif path == "/api/ask":
            b = self._body()
            self._ask(b.get("q", ""), b.get("context", ""))
        else:
            self._send(404, "text/plain", b"not found")

    def _ask(self, question: str, context: str):
        question = (question or "").strip()
        if not question:
            self._json({"error": "empty"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        prompt = (context.strip() + "\n\n" + question) if context.strip() else question
        system = RULES.format(
            root=self.repo.root, inherited=self.claude.inherited, brief=self.claude.brief,
        )
        q: queue.Queue = queue.Queue()

        threading.Thread(
            target=self.claude.ask,
            args=(prompt, system, lambda k, v: q.put((k, v))),
            daemon=True,
        ).start()

        answer, tools = [], []
        while True:
            try:
                kind, payload = q.get(timeout=600)
            except queue.Empty:
                kind, payload = "error", "Timed out waiting for Claude."
            if kind == "delta":
                answer.append(payload)
            elif kind == "tool":
                tools.append(payload)
            try:
                self.wfile.write(f"data: {json.dumps({'k': kind, 'v': payload})}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                kind = "done"
            if kind in ("done", "error"):
                Handler.record(question, context, "".join(answer), tools)
                return

    @classmethod
    def sync(cls, closing: bool) -> dict:
        """Publish everything said since the last sync, for the terminal session to pick up."""
        if not cls.sync_path:
            return {"ok": False, "error": "This dashboard was not started with --sync-file."}
        fresh = cls.transcript[cls.synced_upto:]
        if not fresh and not closing:
            return {"ok": False, "error": "Nothing new to sync."}
        header = (
            "# Synced from the codewalk dashboard"
            + (" (session closed)" if closing else "")
            + f" — {len(fresh)} new exchange(s)\n\n"
        )
        body = header + render_transcript(fresh, numbered_from=cls.synced_upto + 1)
        tmp = cls.sync_path + ".part"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, cls.sync_path)      # atomic: the watcher never sees a half file
        except OSError as e:
            return {"ok": False, "error": f"Could not write the sync file: {e}"}
        cls.synced_upto = len(cls.transcript)
        return {"ok": True, "synced": len(fresh)}

    @classmethod
    def record(cls, question, context, answer, tools):
        cls.transcript.append({
            "q": question, "context": context, "a": answer, "tools": tools,
        })
        if cls.transcript_path:
            try:
                with open(cls.transcript_path, "w", encoding="utf-8") as fh:
                    fh.write(render_transcript(cls.transcript))
            except OSError:
                pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>codewalk</title>
<style>
:root {
  --bg-editor: #1e1e1e; --bg-side: #252526; --bg-title: #3c3c3c; --bg-input: #3c3c3c;
  --bg-hover: #2a2d2e; --bg-active: #37373d;
  --border: #3c3c3c; --border-2: #2b2b2b;
  --fg: #d4d4d4; --fg-dim: #cccccc; --fg-faint: #858585;
  --accent: #007acc; --btn: #0e639c; --btn-hover: #1177bb;
  /* Row washes are deliberately faint — the left rail is the signal, the tint is only
     a hint, so syntax colours underneath stay readable. Turn them up here if you want. */
  --hl-bg: rgba(215,186,125,0.09); --hl-rail: #d7ba7d; --sel-bg: rgba(0,122,204,0.20);
  --chip-bg: #3a3116;
  --add: rgba(78,201,176,0.14); --del: rgba(244,71,71,0.13);
  --red: #f44747; --green: #4ec9b0; --purple: #68217a;
  --mono: ui-monospace, "Cascadia Code", "Droid Sans Mono", Consolas, "Courier New", monospace;
  --ui: system-ui, "Segoe UI", Ubuntu, "Droid Sans", sans-serif;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; background: var(--bg-editor); color: var(--fg);
  font-family: var(--ui); font-size: 13px; display: flex; flex-direction: column; overflow: hidden;
}
:focus-visible { outline: 1px solid var(--accent); outline-offset: -1px; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #424242; }
::-webkit-scrollbar-thumb:hover { background: #4f4f4f; }
::-webkit-scrollbar-track { background: transparent; }

.titlebar {
  flex: 0 0 30px; background: var(--bg-title); display: flex; align-items: center;
  padding: 0 10px; gap: 12px; font-size: 12px; color: var(--fg-dim); user-select: none;
}
.titlebar .brand { color: var(--fg-faint); letter-spacing: 0.09em; text-transform: uppercase; font-size: 10px; }
#handback, #syncbar button {
  font: inherit; font-size: 11px; font-family: var(--ui); background: var(--btn); color: #fff;
  border: 0; border-radius: 2px; padding: 3px 10px; cursor: pointer;
}
#syncbar { margin-left: auto; display: flex; gap: 6px; align-items: center; }
#handback { margin-left: auto; }
#syncbar + #handback { margin-left: 6px; }
#handback:hover:not(:disabled), #syncbar button:hover:not(:disabled) { background: var(--btn-hover); }
#handback:disabled, #syncbar button:disabled { background: #2d2d2d; color: var(--fg-faint); cursor: default; }
#syncbar button.ghost { background: transparent; color: var(--fg-dim); border: 1px solid var(--border); }
#syncbar button.ghost:hover:not(:disabled) { background: var(--bg-hover); color: #fff; }
#syncnote { font-size: 11px; color: var(--green); margin-left: 2px; }
.titlebar .center { flex: 1; text-align: center; color: var(--fg-faint); }

.main { flex: 1 1 auto; display: flex; min-height: 0; }
.pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
#explorer { flex: 0 0 250px; background: var(--bg-side); }
#editor { flex: 1 1 auto; background: var(--bg-editor); }
#chat { flex: 0 0 400px; background: var(--bg-side); border-left: 1px solid var(--border-2); }
.sash { flex: 0 0 4px; cursor: col-resize; background: transparent; }
.sash:hover, .sash.dragging { background: var(--accent); }

.paneHead {
  flex: 0 0 35px; display: flex; align-items: center; gap: 8px; padding: 0 14px;
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fg-dim);
  user-select: none;
}
.paneHead .spacer { flex: 1; }
.iconbtn {
  font: inherit; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase;
  background: transparent; color: var(--fg-faint); border: 1px solid var(--border);
  border-radius: 2px; padding: 2px 7px; cursor: pointer;
}
.iconbtn:hover:not(:disabled) { color: var(--fg-dim); background: var(--bg-hover); }
.iconbtn:disabled { opacity: 0.4; cursor: default; }

/* explorer */
#filter {
  margin: 0 8px 6px; padding: 4px 7px; font: inherit; font-size: 12px;
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 2px; color: var(--fg-dim);
}
#filter:focus { outline: none; border-color: var(--accent); }
#tree { flex: 1 1 auto; overflow: auto; padding-bottom: 12px; font-size: 13px; }
.node {
  display: flex; align-items: center; gap: 5px; padding: 1px 8px 1px 0;
  cursor: pointer; white-space: nowrap; user-select: none; color: var(--fg-dim);
}
.node:hover { background: var(--bg-hover); }
.node.active { background: var(--bg-active); }
.node .tw { flex: 0 0 12px; color: var(--fg-faint); font-size: 10px; text-align: center; }
.node .dot { flex: 0 0 6px; height: 6px; border-radius: 50%; background: var(--fg-faint); }
.node.dir { color: var(--fg); }
.node .lbl { overflow: hidden; text-overflow: ellipsis; }
.node .dim { color: var(--fg-faint); font-size: 11px; margin-left: 4px; }
.kids.collapsed { display: none; }

/* tabs */
.tabs { flex: 0 0 35px; background: var(--bg-side); display: flex; align-items: stretch; overflow-x: auto; border-bottom: 1px solid var(--border-2); }
.tab {
  display: flex; align-items: center; gap: 8px; padding: 0 10px 0 14px; cursor: pointer;
  background: var(--bg-side); color: var(--fg-faint); font-size: 13px;
  border-right: 1px solid var(--border-2); white-space: nowrap;
}
.tab.active { background: var(--bg-editor); color: var(--fg-dim); border-top: 1px solid var(--accent); }
.tab .x { color: var(--fg-faint); font-size: 14px; line-height: 1; padding: 0 2px; border-radius: 2px; }
.tab .x:hover { background: var(--bg-hover); color: var(--fg); }
.subbar { flex: 0 0 26px; display: flex; align-items: center; border-bottom: 1px solid var(--border-2); }
.crumbs { flex: 1 1 auto; display: flex; align-items: center; padding: 0 16px; font-size: 11.5px; color: var(--fg-faint); white-space: nowrap; overflow-x: auto; }
.crumbs::-webkit-scrollbar { height: 0; }
.nav { flex: 0 0 auto; display: flex; align-items: center; gap: 2px; padding: 0 8px; }
.navbtn {
  font: inherit; font-size: 13px; line-height: 1; background: transparent; color: var(--fg-dim);
  border: 1px solid transparent; border-radius: 2px; padding: 2px 7px; cursor: pointer;
}
.navbtn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--border); }
.navbtn:disabled { color: #4a4a4a; cursor: default; }
.navpos { font-size: 10.5px; color: var(--fg-faint); font-variant-numeric: tabular-nums; padding: 0 4px; min-width: 34px; text-align: center; }
.chip.active { background: var(--hl-rail); color: #1e1e1e; border-color: var(--hl-rail); }

#scroll { flex: 1 1 auto; overflow: auto; }
#code { font-family: var(--mono); font-size: 13px; line-height: 19px; min-width: max-content; padding: 6px 0 40vh; }
.ln { display: flex; border-left: 3px solid transparent; scroll-margin: 30vh; }
.ln .no { flex: 0 0 62px; text-align: right; padding-right: 18px; color: var(--fg-faint); font-variant-numeric: tabular-nums; user-select: none; cursor: pointer; }
.ln:hover .no { color: var(--fg-dim); }
.ln .src { white-space: pre; padding-right: 32px; }
.ln.sel { background: var(--sel-bg); border-left-color: var(--accent); }
.ln.hl { background: var(--hl-bg); border-left-color: var(--hl-rail); }
.ln.hl.sel { background: var(--sel-bg); }
.empty { padding: 40px; color: var(--fg-faint); font-size: 13px; max-width: 46ch; line-height: 1.7; }

/* pygments -> vs code dark+ */
.k, .kc, .kd, .kr, .kt { color: #569cd6; }
.kn, .ow { color: #c586c0; }
.s, .s1, .s2, .sb, .sc, .sd, .sh, .sx, .sr, .se, .si, .ss { color: #ce9178; }
.sa { color: #569cd6; }
.m, .mi, .mf, .mh, .mo, .il { color: #b5cea8; }
.c, .c1, .cm, .cs, .cp, .cpf { color: #6a9955; font-style: italic; }
.nf, .fm, .nd { color: #dcdcaa; }
.nc, .ne, .nb, .nn, .nt { color: #4ec9b0; }
.bp { color: #569cd6; }
.n, .nv, .vc, .vg, .vi, .na, .py, .nl { color: #9cdcfe; }
.o, .p { color: #d4d4d4; }
.gd { color: #f44747; } .gi { color: #4ec9b0; } .gh, .gu { color: #569cd6; }
.err { color: #f44747; }

/* chat */
#log { flex: 1 1 auto; overflow-y: auto; padding: 14px 16px 6px; }
.msg { margin-bottom: 18px; }
.who { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fg-faint); margin-bottom: 5px; }
.msg.me .who { color: #569cd6; }
.msg.me .body { background: #2d2d2d; border-left: 2px solid var(--accent); padding: 7px 10px; white-space: pre-wrap; color: var(--fg-dim); font-size: 12.5px; }
.body { line-height: 1.65; }
.body p { margin: 0 0 0.7em; }
.body p:last-child { margin-bottom: 0; }
.body code { font-family: var(--mono); font-size: 12px; background: #2d2d2d; padding: 1px 4px; border-radius: 2px; color: #ce9178; }
.chip {
  font-family: var(--mono); font-size: 11.5px; background: var(--chip-bg); color: var(--hl-rail);
  border: 1px solid #5a4a20; border-radius: 2px; padding: 1px 5px; cursor: pointer; white-space: nowrap;
}
.chip:hover { background: var(--hl-rail); color: #1e1e1e; }
.acts { margin-bottom: 8px; display: flex; flex-direction: column; gap: 2px; }
.act { font-family: var(--mono); font-size: 11px; color: var(--fg-faint); display: flex; gap: 6px; }
.act::before { content: "\203A"; color: var(--green); }
.thinking { color: var(--fg-faint); font-style: italic; }
.thinking::after { content: ""; animation: dots 1.4s steps(4, end) infinite; }
@keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
.errline { color: var(--red); font-size: 12px; margin-top: 6px; }
.contbar { margin-top: 12px; display: flex; align-items: center; gap: 10px; }
.contbtn {
  font: inherit; font-size: 12px; font-family: var(--ui); background: var(--btn); color: #fff;
  border: 0; border-radius: 2px; padding: 6px 12px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 8px;
}
.contbtn:hover:not(:disabled) { background: var(--btn-hover); }
.contbtn:disabled { background: #2d2d2d; color: var(--fg-faint); cursor: default; }
.contbtn .next { opacity: 0.75; font-style: italic; }
.contbar .kbd { font-family: var(--mono); font-size: 10.5px; color: var(--fg-faint); }
.notice { color: var(--fg-faint); font-size: 12px; font-style: italic; }

.diff { border: 1px solid var(--border); background: var(--bg-editor); margin: 10px 0; }
.diff .head { display: flex; align-items: center; gap: 8px; padding: 5px 10px; background: var(--bg-side); border-bottom: 1px solid var(--border); font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--fg-faint); }
.diff .head .spacer { flex: 1; }
.diff pre { margin: 0; font-family: var(--mono); font-size: 12px; line-height: 18px; overflow-x: auto; padding: 6px 0; }
.diff .d { padding: 0 10px; white-space: pre; }
.diff .d.minus { background: var(--del); color: #d4a0a0; }
.diff .d.plus { background: var(--add); color: #b7e0d5; }
.btn { font: inherit; font-size: 11px; background: var(--btn); color: #fff; border: 0; border-radius: 2px; padding: 3px 10px; cursor: pointer; }
.btn:hover:not(:disabled) { background: var(--btn-hover); }
.btn.ghost { background: transparent; color: var(--fg-faint); border: 1px solid var(--border); }
.btn.ghost:hover { color: var(--fg-dim); }
.btn:disabled { opacity: 0.5; cursor: default; }

#composer { flex: 0 0 auto; border-top: 1px solid var(--border-2); padding: 10px 14px 12px; display: flex; flex-direction: column; gap: 8px; }
#ctx { font-family: var(--mono); font-size: 11.5px; color: #9cdcfe; display: flex; gap: 8px; align-items: center; }
.refbar { display: flex; flex-wrap: wrap; gap: 5px; }
.ref {
  display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11.5px;
  background: #1e3a5f; color: #9cdcfe; border: 1px solid #2d5a8c; border-radius: 2px;
  padding: 2px 4px 2px 7px; max-width: 100%;
}
.ref .lbl { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
.ref .lbl:hover { color: #cfe8ff; }
.ref .x { cursor: pointer; color: #6f9dc9; padding: 0 2px; border-radius: 2px; font-size: 13px; line-height: 1; }
.ref .x:hover { background: #2d5a8c; color: #fff; }
.msg.me .refbar { margin-bottom: 6px; }
.msg.me .ref { cursor: pointer; }
#addSel {
  position: fixed; z-index: 50; font: inherit; font-size: 11px; font-family: var(--ui);
  background: var(--btn); color: #fff; border: 1px solid #1177bb; border-radius: 3px;
  padding: 3px 9px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5); white-space: nowrap;
}
#addSel:hover { background: var(--btn-hover); }
#addSel kbd { font-family: var(--mono); font-size: 10px; opacity: 0.7; margin-left: 5px; }
#ctx button { font: inherit; background: none; border: 0; color: var(--fg-faint); cursor: pointer; text-decoration: underline; padding: 0; }
.starters { display: flex; flex-wrap: wrap; gap: 5px; }
.starters button { font: inherit; font-size: 11.5px; background: #2d2d2d; color: var(--fg-faint); border: 1px solid var(--border); border-radius: 2px; padding: 3px 8px; cursor: pointer; }
.starters button:hover { color: var(--fg-dim); background: var(--bg-hover); }
.row { display: flex; gap: 8px; align-items: flex-end; }
textarea { flex: 1; resize: none; font-family: var(--ui); font-size: 13px; color: var(--fg-dim); background: var(--bg-input); border: 1px solid var(--border); border-radius: 2px; padding: 7px 9px; min-height: 32px; max-height: 160px; line-height: 1.5; }
textarea:focus { outline: none; border-color: var(--accent); }

.status { flex: 0 0 22px; background: var(--accent); color: #fff; display: flex; align-items: center; gap: 16px; padding: 0 12px; font-size: 11.5px; user-select: none; }
.status .spacer { flex: 1; }
.status.busy { background: var(--purple); }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
</head>
<body>

<div class="titlebar">
  <span class="brand">codewalk</span>
  <span class="center" id="titleRoot">—</span>
  <span id="syncbar" hidden><button id="syncNow">Sync to terminal</button><button id="syncClose" class="ghost">Sync &amp; close</button></span>
  <button id="handback" hidden>Return to terminal</button>
</div>

<div class="main">
  <section class="pane" id="explorer">
    <div class="paneHead"><span id="expName">Explorer</span><span class="spacer"></span><span class="dim" id="fileCount" style="text-transform:none;letter-spacing:0"></span></div>
    <input id="filter" type="search" placeholder="Filter files…" aria-label="Filter files">
    <div id="tree"></div>
  </section>

  <div class="sash" id="sash1" role="separator" aria-orientation="vertical" tabindex="0" aria-label="Resize explorer"></div>

  <section class="pane" id="editor">
    <div class="tabs" id="tabs"></div>
    <div class="subbar">
      <div class="crumbs" id="crumbs"></div>
      <div class="nav">
        <span class="navpos" id="navPos"></span>
        <button class="navbtn" id="navBack" title="Back (Alt+Left)" disabled>&#8592;</button>
        <button class="navbtn" id="navFwd" title="Forward (Alt+Right)" disabled>&#8594;</button>
      </div>
    </div>
    <div id="scroll"><div id="code"></div>
      <div class="empty" id="empty">No file open. Pick one from the tree — or just ask a question on the right, and I will open whatever I need as I explain it.</div>
    </div>
  </section>

  <div class="sash" id="sash2" role="separator" aria-orientation="vertical" tabindex="0" aria-label="Resize chat"></div>

  <section class="pane" id="chat">
    <div class="paneHead"><span>Walkthrough</span><span class="spacer"></span><button class="iconbtn" id="follow" title="Let Claude move the editor as it explains">Follow: on</button><button class="iconbtn" id="clearHl" disabled>Clear highlight</button></div>
    <div id="log"></div>
    <div id="composer">
      <div id="ctx" hidden></div>
      <div class="refbar" id="refs" hidden></div>
      <div class="starters" id="starters"></div>
      <div class="row">
        <textarea id="box" rows="1" placeholder="Ask about this codebase…" aria-label="Message"></textarea>
        <button class="btn" id="send">Send</button>
      </div>
    </div>
  </section>
</div>

<button id="addSel" hidden>Add to chat<kbd>Ctrl L</kbd></button>

<div class="status" id="status">
  <span id="stFile">no file</span>
  <span id="stSel"></span>
  <span class="spacer"></span>
  <span id="stState">ready</span>
</div>

<script>
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const codeEl = $("code"), logEl = $("log"), boxEl = $("box"), sendEl = $("send");

  let allFiles = [], openTabs = [], active = null, busy = false, watchToken = 0;

  const DOT = { py: "#4b8bbe", js: "#e6c07b", ts: "#3178c6", jsx: "#e6c07b", tsx: "#3178c6",
    md: "#6a9955", json: "#cbcb41", yml: "#cb4141", yaml: "#cb4141", toml: "#cbcb41",
    html: "#e34c26", css: "#563d7c", sh: "#89e051", rs: "#dea584", go: "#00add8",
    c: "#555555", h: "#555555", cpp: "#f34b7d", cu: "#76b900", ipynb: "#da5b0b" };
  const dotColor = (p) => DOT[(p.split(".").pop() || "").toLowerCase()] || "#6e7681";

  // ---------------- tree ----------------
  const expanded = new Set([""]);

  function buildTree(paths) {
    const root = { dirs: new Map(), files: [] };
    for (const p of paths) {
      const parts = p.split("/");
      let node = root;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!node.dirs.has(parts[i])) node.dirs.set(parts[i], { dirs: new Map(), files: [] });
        node = node.dirs.get(parts[i]);
      }
      node.files.push({ name: parts[parts.length - 1], path: p });
    }
    return root;
  }

  function renderTree() {
    const q = $("filter").value.trim().toLowerCase();
    const el = $("tree");
    el.textContent = "";
    if (q) {
      const hits = allFiles.filter((p) => p.toLowerCase().includes(q)).slice(0, 400);
      $("fileCount").textContent = hits.length + (hits.length === 400 ? "+" : "") + " match";
      for (const p of hits) el.appendChild(fileNode(p, p, 0));
      if (!hits.length) {
        const d = document.createElement("div");
        d.className = "node"; d.style.color = "var(--fg-faint)";
        d.textContent = "  no match";
        el.appendChild(d);
      }
      return;
    }
    $("fileCount").textContent = allFiles.length + " files";
    el.appendChild(dirBody(buildTree(allFiles), "", 0));
  }

  function dirBody(node, prefix, depth) {
    const frag = document.createDocumentFragment();
    for (const [name, child] of Array.from(node.dirs.entries()).sort((a, b) => a[0].localeCompare(b[0]))) {
      const full = prefix ? prefix + "/" + name : name;
      const open = expanded.has(full);
      const row = document.createElement("div");
      row.className = "node dir";
      row.style.paddingLeft = 8 + depth * 12 + "px";
      const tw = document.createElement("span");
      tw.className = "tw";
      tw.textContent = open ? "▾" : "▸";
      const lbl = document.createElement("span");
      lbl.className = "lbl";
      lbl.textContent = name;
      row.append(tw, lbl);
      row.onclick = () => { open ? expanded.delete(full) : expanded.add(full); renderTree(); };
      frag.appendChild(row);
      const kids = document.createElement("div");
      kids.className = "kids" + (open ? "" : " collapsed");
      if (open) kids.appendChild(dirBody(child, full, depth + 1));
      frag.appendChild(kids);
    }
    for (const f of node.files.sort((a, b) => a.name.localeCompare(b.name))) {
      frag.appendChild(fileNode(f.path, f.name, depth));
    }
    return frag;
  }

  function fileNode(path, label, depth) {
    const row = document.createElement("div");
    row.className = "node file" + (active === path ? " active" : "");
    row.style.paddingLeft = 8 + depth * 12 + "px";
    row.dataset.path = path;
    const tw = document.createElement("span");
    tw.className = "tw";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = dotColor(path);
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = label;
    row.append(tw, dot, lbl);
    row.onclick = () => openFile(path);
    return row;
  }

  function expandTo(path) {
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) expanded.add(parts.slice(0, i).join("/"));
  }

  // ---------------- navigation history ----------------
  // Every place the editor is sent — a tree click, a tab, one of Claude's chips — is an
  // entry here, so the arrows replay a walkthrough the way browser history replays a session.
  let hist = [], hIdx = -1, navLock = false;
  const HIST_MAX = 200;

  function pushHist(path, spec) {
    if (navLock) return;
    const cur = hist[hIdx];
    if (cur && cur.path === path && cur.spec === (spec || "")) return;
    hist = hist.slice(0, hIdx + 1);
    hist.push({ path: path, spec: spec || "" });
    if (hist.length > HIST_MAX) hist.shift();
    hIdx = hist.length - 1;
    updateNav();
  }

  function updateNav() {
    $("navBack").disabled = hIdx <= 0;
    $("navFwd").disabled = hIdx >= hist.length - 1;
    $("navPos").textContent = hist.length ? (hIdx + 1) + "/" + hist.length : "";
  }

  async function navGo(delta) {
    const i = hIdx + delta;
    if (i < 0 || i >= hist.length) return;
    hIdx = i;
    navLock = true;
    try { await openFile(hist[i].path, hist[i].spec); }
    finally { navLock = false; updateNav(); }
  }

  $("navBack").onclick = () => navGo(-1);
  $("navFwd").onclick = () => navGo(1);
  document.addEventListener("keydown", (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); navGo(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); navGo(1); }
  });
  window.addEventListener("mouseup", (e) => {
    if (e.button === 3) { e.preventDefault(); navGo(-1); }
    if (e.button === 4) { e.preventDefault(); navGo(1); }
  });

  // ---------------- tabs + file ----------------
  function tab(path) { return openTabs.find((t) => t.path === path); }

  async function openFile(path, spec, focus) {
    let t = tab(path);
    if (!t) {
      const data = await fetch("/api/file?path=" + encodeURIComponent(path)).then((r) => r.json());
      if (data.error) { flash(data.error); return null; }
      t = { path: path, data: data, hl: new Set(), sel: null, scroll: 0 };
      openTabs.push(t);
    }
    if (active && active !== path) {
      const prev = tab(active);
      if (prev) prev.scroll = $("scroll").scrollTop;
    }
    active = path;
    if (spec) { t.hl = parseSpec(spec); }
    expandTo(path);
    renderTree();
    renderTabs();
    renderCode(t, !!spec);
    watchFile(t);
    pushHist(path, spec);
    return t;
  }

  function closeTab(path) {
    const i = openTabs.findIndex((t) => t.path === path);
    if (i < 0) return;
    openTabs.splice(i, 1);
    if (active === path) {
      const next = openTabs[Math.min(i, openTabs.length - 1)];
      active = next ? next.path : null;
      if (next) { renderCode(next, false); watchFile(next); }
      else { codeEl.textContent = ""; $("empty").hidden = false; $("stFile").textContent = "no file"; $("crumbs").textContent = ""; }
    }
    renderTabs(); renderTree();
  }

  function renderTabs() {
    const el = $("tabs");
    el.textContent = "";
    for (const t of openTabs) {
      const d = document.createElement("div");
      d.className = "tab" + (t.path === active ? " active" : "");
      const dot = document.createElement("span");
      dot.className = "dot"; dot.style.background = dotColor(t.path);
      dot.style.width = "6px"; dot.style.height = "6px"; dot.style.borderRadius = "50%"; dot.style.display = "inline-block";
      const n = document.createElement("span");
      n.textContent = t.data.name;
      const x = document.createElement("span");
      x.className = "x"; x.textContent = "×";
      x.onclick = (e) => { e.stopPropagation(); closeTab(t.path); };
      d.append(dot, n, x);
      d.onclick = () => openFile(t.path);
      el.appendChild(d);
    }
  }

  function renderCode(t, scrollToHl) {
    $("empty").hidden = true;
    codeEl.textContent = "";
    const frag = document.createDocumentFragment();
    t.data.lines.forEach((segs, i) => {
      const n = i + 1;
      const row = document.createElement("div");
      row.className = "ln" + (t.hl.has(n) ? " hl" : "") + (t.sel && n >= t.sel.a && n <= t.sel.b ? " sel" : "");
      row.dataset.n = String(n);
      const no = document.createElement("span");
      no.className = "no"; no.textContent = String(n);
      const src = document.createElement("span");
      src.className = "src";
      if (!segs.length) src.appendChild(document.createTextNode("​"));
      for (const s of segs) {
        if (s.c) { const sp = document.createElement("span"); sp.className = s.c; sp.textContent = s.t; src.appendChild(sp); }
        else src.appendChild(document.createTextNode(s.t));
      }
      row.append(no, src);
      frag.appendChild(row);
    });
    codeEl.appendChild(frag);

    $("stFile").textContent = t.path + "  ·  " + t.data.lines.length + " lines";
    document.title = t.data.name + " — codewalk";
    const cr = $("crumbs");
    cr.textContent = "";
    t.path.split("/").forEach((part, i, arr) => {
      const s = document.createElement("span");
      s.textContent = part;
      cr.appendChild(s);
      if (i < arr.length - 1) { const g = document.createElement("span"); g.textContent = " › "; g.style.color = "var(--border)"; cr.appendChild(g); }
    });
    $("clearHl").disabled = !t.hl.size;
    paintSel();

    if (scrollToHl && t.hl.size) {
      const first = codeEl.querySelector('.ln[data-n="' + Math.min.apply(null, Array.from(t.hl)) + '"]');
      if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
    } else {
      $("scroll").scrollTop = t.scroll || 0;
    }
  }

  function parseSpec(spec) {
    const out = new Set();
    for (const part of String(spec).split(",")) {
      const m = part.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
      if (!m) continue;
      const a = Number(m[1]), b = m[2] ? Number(m[2]) : Number(m[1]);
      for (let i = Math.min(a, b); i <= Math.max(a, b); i++) out.add(i);
    }
    return out;
  }

  let follow = true;
  const followBtn = $("follow");
  followBtn.onclick = () => {
    follow = !follow;
    followBtn.textContent = "Follow: " + (follow ? "on" : "off");
    followBtn.style.color = follow ? "var(--hl-rail)" : "";
  };
  followBtn.style.color = "var(--hl-rail)";

  $("clearHl").onclick = () => {
    const t = tab(active);
    if (!t) return;
    t.hl = new Set();
    codeEl.querySelectorAll(".ln.hl").forEach((r) => r.classList.remove("hl"));
    $("clearHl").disabled = true;
  };

  // ---------------- gutter selection ----------------
  let dragFrom = null;
  codeEl.addEventListener("pointerdown", (e) => {
    const no = e.target.closest(".no");
    if (!no) return;
    e.preventDefault();
    const t = tab(active); if (!t) return;
    dragFrom = Number(no.parentElement.dataset.n);
    t.sel = { a: dragFrom, b: dragFrom };
    paintSel();
  });
  codeEl.addEventListener("pointermove", (e) => {
    if (dragFrom == null) return;
    const row = e.target.closest(".ln"); if (!row) return;
    const t = tab(active); if (!t) return;
    const n = Number(row.dataset.n);
    t.sel = { a: Math.min(dragFrom, n), b: Math.max(dragFrom, n) };
    paintSel();
  });
  window.addEventListener("pointerup", () => { dragFrom = null; });

  function paintSel() {
    const t = tab(active);
    const sel = t && t.sel;
    codeEl.querySelectorAll(".ln").forEach((r) => {
      const n = Number(r.dataset.n);
      r.classList.toggle("sel", !!sel && n >= sel.a && n <= sel.b);
    });
    const ctx = $("ctx");
    if (sel) {
      ctx.hidden = false; ctx.textContent = "";
      const s = document.createElement("span");
      s.textContent = t.path + ":" + (sel.a === sel.b ? sel.a : sel.a + "-" + sel.b);
      const add = document.createElement("button");
      add.textContent = "add to chat";
      add.onclick = () => { addRef(t.path, sel.a, sel.b); t.sel = null; paintSel(); };
      const c = document.createElement("button");
      c.textContent = "clear";
      c.onclick = () => { t.sel = null; paintSel(); };
      ctx.append(s, add, c);
      $("stSel").textContent = (sel.b - sel.a + 1) + " selected";
    } else { ctx.hidden = true; $("stSel").textContent = ""; }
  }

  // ---------------- references you attach ----------------
  // The mirror image of my [[open:...]] chips: you point at code, it rides along with the question.
  let refs = [];
  const MAX_REF_LINES = 200;

  function lineText(t, a, b) {
    return t.data.lines.slice(a - 1, b).map((segs) => segs.map((x) => x.t).join("")).join("\n");
  }

  function addRef(path, a, b) {
    const t = tab(path);
    if (!t) return;
    if (refs.some((r) => r.path === path && r.a === a && r.b === b)) return;
    let text = lineText(t, a, Math.min(b, a + MAX_REF_LINES - 1));
    const clipped = b - a + 1 > MAX_REF_LINES;
    refs.push({ path: path, a: a, b: b, text: text, clipped: clipped });
    renderRefs();
    boxEl.focus();
  }

  function refLabel(r) {
    return shortPath(r.path) + ":" + (r.a === r.b ? r.a : r.a + "-" + r.b);
  }

  function refChip(r, onRemove) {
    const el = document.createElement("span");
    el.className = "ref";
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = refLabel(r);
    lbl.title = r.path + " lines " + r.a + "-" + r.b + (r.clipped ? " (first " + MAX_REF_LINES + " sent)" : "");
    lbl.onclick = () => openFile(r.path, r.a + "-" + r.b);
    el.appendChild(lbl);
    if (onRemove) {
      const x = document.createElement("span");
      x.className = "x";
      x.textContent = "\u00d7";
      x.onclick = onRemove;
      el.appendChild(x);
    }
    return el;
  }

  function renderRefs() {
    const bar = $("refs");
    bar.textContent = "";
    bar.hidden = !refs.length;
    refs.forEach((r, i) => {
      bar.appendChild(refChip(r, () => { refs.splice(i, 1); renderRefs(); }));
    });
  }

  // -- attach from a free text selection inside the code ----------------------
  function selectedLineRange() {
    const s = window.getSelection();
    if (!s || s.isCollapsed || !s.rangeCount) return null;
    const node = (n) => (n && n.nodeType === 3 ? n.parentElement : n);
    const a = node(s.anchorNode), b = node(s.focusNode);
    if (!a || !b || !codeEl.contains(a) || !codeEl.contains(b)) return null;
    const ra = a.closest(".ln"), rb = b.closest(".ln");
    if (!ra || !rb) return null;
    const x = Number(ra.dataset.n), y = Number(rb.dataset.n);
    return { a: Math.min(x, y), b: Math.max(x, y), rect: s.getRangeAt(0).getBoundingClientRect() };
  }

  const addBtn = $("addSel");
  function hideFloat() { addBtn.hidden = true; }
  function showFloat(r) {
    addBtn.hidden = false;
    const top = Math.max(34, r.rect.top - 30);
    addBtn.style.top = top + "px";
    addBtn.style.left = Math.min(window.innerWidth - 140, Math.max(8, r.rect.left)) + "px";
    addBtn.onclick = () => {
      addRef(active, r.a, r.b);
      window.getSelection().removeAllRanges();
      hideFloat();
    };
  }
  $("scroll").addEventListener("mouseup", () => {
    setTimeout(() => {
      const r = selectedLineRange();
      if (r && active) showFloat(r); else hideFloat();
    }, 0);
  });
  $("scroll").addEventListener("scroll", hideFloat);
  document.addEventListener("mousedown", (e) => { if (e.target !== addBtn) hideFloat(); });

  // -- keyboard: ctrl-L attaches whatever is selected -------------------------
  document.addEventListener("keydown", (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "l") return;
    const t = tab(active);
    if (!t) return;
    const r = selectedLineRange();
    if (r) {
      e.preventDefault();
      addRef(active, r.a, r.b);
      window.getSelection().removeAllRanges();
      hideFloat();
    } else if (t.sel) {
      e.preventDefault();
      addRef(active, t.sel.a, t.sel.b);
      t.sel = null;
      paintSel();
    }
  });

  // ---------------- watching ----------------
  async function watchFile(t) {
    const token = ++watchToken;
    while (token === watchToken && active === t.path) {
      try {
        const r = await fetch("/api/watch?path=" + encodeURIComponent(t.path) + "&mtime=" + t.data.mtime).then((x) => x.json());
        if (token !== watchToken) return;
        if (r.changed) {
          t.data = r;
          renderCode(t, false);
          flash("reloaded");
        }
      } catch (e) {
        $("stState").textContent = "disconnected";
        await new Promise((res) => setTimeout(res, 2000));
      }
    }
  }
  function flash(msg) {
    $("stState").textContent = msg;
    setTimeout(() => { if (!busy) $("stState").textContent = "ready"; }, 2000);
  }

  // ---------------- reply rendering ----------------
  const OPEN_RE = /\[\[open:\s*([^\]|:]+?)\s*(?::\s*([\d,\s-]+?))?\s*(?:\|\s*([^\]]*?)\s*)?\]\]/g;
  const EDIT_RE = /\[\[edit:\s*([^\]|:]+?)\s*:\s*(\d+)\s*-\s*(\d+)\s*\]\]\n?([\s\S]*?)\[\[\/edit\]\]/g;
  const CONT_RE = /\[\[continue(?::\s*([^\]]*?))?\s*\]\]/;

  function renderReply(el, text) {
    el.textContent = "";
    text = text.replace(CONT_RE, "");   // the marker is a control signal, not prose
    let cursor = 0, m;
    EDIT_RE.lastIndex = 0;
    while ((m = EDIT_RE.exec(text))) {
      prose(el, text.slice(cursor, m.index));
      el.appendChild(diffCard(m[1], Number(m[2]), Number(m[3]), m[4].replace(/\n$/, "")));
      cursor = m.index + m[0].length;
    }
    const tail = text.slice(cursor);
    const open = tail.lastIndexOf("[[edit:");
    if (open !== -1) {
      prose(el, tail.slice(0, open));
      const p = document.createElement("p");
      p.className = "notice"; p.textContent = "preparing an edit…";
      el.appendChild(p);
    } else prose(el, tail);
  }

  function prose(el, text) {
    if (!text.trim()) return;
    for (const para of text.split(/\n{2,}/)) {
      if (!para.trim()) continue;
      const p = document.createElement("p");
      // Inline code is resolved FIRST: a directive written inside backticks is Claude
      // talking about the syntax, not asking to navigate. Only bare text gets scanned.
      for (const bit of para.split(/(`[^`\n]+`)/)) {
        if (!bit) continue;
        if (bit.length > 2 && bit.startsWith("`") && bit.endsWith("`")) {
          const c = document.createElement("code");
          c.textContent = bit.slice(1, -1);
          p.appendChild(c);
        } else {
          scanDirectives(p, bit);
        }
      }
      el.appendChild(p);
    }
  }

  function scanDirectives(p, text) {
    let last = 0, m;
    OPEN_RE.lastIndex = 0;
    while ((m = OPEN_RE.exec(text))) {
      const path = m[1], spec = m[2] || "", label = m[3];
      if (!isPath(path)) continue;
      p.appendChild(document.createTextNode(text.slice(last, m.index)));
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.textContent = label ? label + " \u00b7 " + shortPath(path) + (spec ? ":" + spec : "")
                               : shortPath(path) + (spec ? ":" + spec : "");
      chip.onclick = () => {
        const body = chip.closest(".body");
        if (body) body.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        openFile(path, spec);
      };
      p.appendChild(chip);
      last = m.index + m[0].length;
    }
    p.appendChild(document.createTextNode(text.slice(last)));
  }

  // A real path has word characters and no whitespace — this rejects the literal
  // "..." written when quoting the directive syntax in prose.
  function isPath(v) {
    return /[A-Za-z0-9_]/.test(v) && !/\s/.test(v);
  }

  function allDirectives(text) {
    const out = [];
    const bare = text.replace(/`[^`\n]+`/g, "");   // ignore anything inside inline code
    OPEN_RE.lastIndex = 0;
    let m;
    while ((m = OPEN_RE.exec(bare))) {
      if (isPath(m[1])) out.push({ path: m[1], spec: m[2] || "" });
    }
    return out;
  }

  function markActiveChip(scope, idx) {
    const chips = scope.querySelectorAll(".chip");
    chips.forEach((c, i) => c.classList.toggle("active", i === idx));
    if (idx >= 0 && chips[idx]) {
      const r = chips[idx].getBoundingClientRect(), lr = logEl.getBoundingClientRect();
      if (r.top < lr.top || r.bottom > lr.bottom) chips[idx].scrollIntoView({ block: "nearest" });
    }
  }

  function shortPath(p) {
    const parts = p.split("/");
    return parts.length > 2 ? ".../" + parts.slice(-2).join("/") : p;
  }

  function inline(parent, text) {
    if (!text) return;
    for (const bit of text.split(/(`[^`\n]+`)/)) {
      if (!bit) continue;
      if (bit.length > 2 && bit.startsWith("`") && bit.endsWith("`")) {
        const c = document.createElement("code");
        c.textContent = bit.slice(1, -1);
        parent.appendChild(c);
      } else parent.appendChild(document.createTextNode(bit));
    }
  }

  function diffCard(path, start, end, replacement) {
    const card = document.createElement("div");
    card.className = "diff";
    const head = document.createElement("div");
    head.className = "head";
    const title = document.createElement("span");
    title.textContent = shortPath(path) + " · lines " + start + "–" + end;
    const spacer = document.createElement("span");
    spacer.className = "spacer";
    const show = document.createElement("button");
    show.className = "btn ghost"; show.textContent = "Show";
    show.onclick = () => openFile(path, start + "-" + end);
    const apply = document.createElement("button");
    apply.className = "btn"; apply.textContent = "Apply";
    head.append(title, spacer, show, apply);

    const pre = document.createElement("pre");
    for (const l of replacement.split("\n")) {
      const d = document.createElement("div");
      d.className = "d plus"; d.textContent = "+ " + l;
      pre.appendChild(d);
    }
    // fill in the removed lines once we have the file
    fetch("/api/file?path=" + encodeURIComponent(path)).then((r) => r.json()).then((data) => {
      if (data.error) return;
      card.dataset.digest = data.digest;
      const olds = data.lines.slice(start - 1, end)
        .map((segs) => segs.map((s) => s.t).join(""));
      const first = pre.firstChild;
      for (const l of olds) {
        const d = document.createElement("div");
        d.className = "d minus"; d.textContent = "- " + l;
        pre.insertBefore(d, first);
      }
    });

    apply.onclick = async () => {
      apply.disabled = true; apply.textContent = "Applying…";
      try {
        const res = await fetch("/api/apply", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: path, start: start, end: end, text: replacement, digest: card.dataset.digest || "" }),
        }).then((r) => r.json());
        if (res.ok) {
          apply.textContent = "Applied";
          title.textContent = "applied · " + shortPath(path);
          const t = tab(path);
          if (t) t.data = await fetch("/api/file?path=" + encodeURIComponent(path)).then((r) => r.json());
          await openFile(path, start + "-" + (start + res.added - 1));
        } else {
          apply.textContent = "Apply"; apply.disabled = false;
          const e = document.createElement("div");
          e.className = "errline"; e.textContent = res.error;
          card.appendChild(e);
        }
      } catch (err) {
        apply.textContent = "Apply"; apply.disabled = false;
      }
    };
    card.append(head, pre);
    return card;
  }

  // ---------------- chat ----------------
  function bubble(who, cls) {
    const m = document.createElement("div");
    m.className = "msg " + (cls || "");
    const w = document.createElement("div");
    w.className = "who"; w.textContent = who;
    const b = document.createElement("div");
    b.className = "body";
    m.append(w, b);
    logEl.appendChild(m);
    logEl.scrollTop = logEl.scrollHeight;
    return m;
  }

  // A step ends with [[continue: ...]]; this is the button that asks for the next one.
  function addContinue(msg, label) {
    const bar = document.createElement("div");
    bar.className = "contbar";
    const btn = document.createElement("button");
    btn.className = "contbtn";
    const cap = document.createElement("span");
    cap.textContent = "Continue";
    btn.appendChild(cap);
    if (label) {
      const nx = document.createElement("span");
      nx.className = "next";
      nx.textContent = "\u2192 " + label;
      btn.appendChild(nx);
    }
    btn.onclick = () => ask("Continue.");
    const hint = document.createElement("span");
    hint.className = "kbd";
    hint.textContent = "or press Enter";
    bar.append(btn, hint);
    msg.appendChild(bar);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function retireContinues() {
    logEl.querySelectorAll(".contbar").forEach((b) => b.remove());
  }

  function pendingContinue() {
    const bars = logEl.querySelectorAll(".contbar .contbtn");
    return bars.length ? bars[bars.length - 1] : null;
  }

  function setBusy(on, label) {
    busy = on; sendEl.disabled = on;
    $("status").classList.toggle("busy", on);
    $("stState").textContent = label || (on ? "thinking" : "ready");
  }

  async function ask(text) {
    if (busy) return;
    const q = (text || "").trim();
    if (!q) return;

    retireContinues();
    const t = tab(active);
    const sent = refs.slice();
    let context = "";
    if (sent.length) {
      context = "REFERENCES the user attached to this question:\n\n";
      for (const r of sent) {
        context += "--- " + r.path + " lines " + r.a + "-" + r.b + " ---\n"
                 + r.text + "\n"
                 + (r.clipped ? "--- (truncated; read the file for the rest) ---\n" : "")
                 + "\n";
      }
      context = context.trimEnd() + "\n";
    }
    if (t) {
      context += (context ? "\n" : "") + "[The user is looking at " + t.path;
      if (t.sel) context += ", lines " + t.sel.a + "-" + t.sel.b + " selected";
      context += ".]";
    }

    const mine = bubble("You", "me");
    if (sent.length) {
      const bar = document.createElement("div");
      bar.className = "refbar";
      for (const r of sent) bar.appendChild(refChip(r, null));
      mine.insertBefore(bar, mine.querySelector(".body"));
    }
    mine.querySelector(".body").textContent = q;
    refs = [];
    renderRefs();
    const msg = bubble("Claude");
    const acts = document.createElement("div");
    acts.className = "acts";
    const out = msg.querySelector(".body");
    msg.insertBefore(acts, out);
    out.innerHTML = '<p class="thinking">Thinking</p>';
    setBusy(true);

    let acc = "", autoIdx = -1;
    const atBottom = () => logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 90;

    try {
      const res = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: q, context: context }),
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          let ev;
          try { ev = JSON.parse(part.slice(6)); } catch (e) { continue; }
          const stick = atBottom();
          if (ev.k === "delta") {
            acc += ev.v;
            renderReply(out, acc);
            const ds = allDirectives(acc);
            if (follow && ds.length - 1 > autoIdx) {
              autoIdx = ds.length - 1;
              openFile(ds[autoIdx].path, ds[autoIdx].spec);
            }
            markActiveChip(out, autoIdx);
          } else if (ev.k === "tool") {
            const a = document.createElement("div");
            a.className = "act"; a.textContent = ev.v;
            acts.appendChild(a);
            setBusy(true, "reading");
          } else if (ev.k === "notice") {
            const n = document.createElement("p");
            n.className = "notice"; n.textContent = ev.v;
            out.appendChild(n);
          } else if (ev.k === "error") {
            if (!acc) out.textContent = "";
            const e = document.createElement("div");
            e.className = "errline"; e.textContent = ev.v;
            out.appendChild(e);
          }
          if (stick) logEl.scrollTop = logEl.scrollHeight;
        }
      }
      if (acc) {
        renderReply(out, acc);
        const cont = CONT_RE.exec(acc);
        if (cont) addContinue(msg, (cont[1] || "").trim());
        const ds = allDirectives(acc);
        if (ds.length && autoIdx < 0) {
          autoIdx = 0;
          openFile(ds[0].path, ds[0].spec);
        }
        markActiveChip(out, autoIdx);
      }
    } catch (err) {
      const e = document.createElement("div");
      e.className = "errline";
      e.textContent = "Lost the connection to codewalk. Is the server still running?";
      out.appendChild(e);
    } finally {
      setBusy(false);
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  const STARTERS = [
    "Give me the tour: what is in this project and how does it fit together?",
    "Where does execution start?",
    "What are the least obvious design choices here?",
    "Where is this most likely to break?",
  ];
  for (const s of STARTERS) {
    const b = document.createElement("button");
    b.textContent = s;
    b.onclick = () => ask(s);
    $("starters").appendChild(b);
  }

  sendEl.onclick = () => { const v = boxEl.value; boxEl.value = ""; boxEl.style.height = "auto"; ask(v); };
  boxEl.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    e.preventDefault();
    if (!boxEl.value.trim()) {
      const c = pendingContinue();
      if (c) { c.click(); return; }
    }
    sendEl.click();
  });
  boxEl.addEventListener("input", () => {
    boxEl.style.height = "auto";
    boxEl.style.height = Math.min(boxEl.scrollHeight, 160) + "px";
  });
  $("filter").addEventListener("input", renderTree);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "p") { e.preventDefault(); $("filter").focus(); $("filter").select(); }
  });

  // ---------------- sashes ----------------
  function makeSash(sash, pane, side) {
    let dragging = false;
    sash.addEventListener("pointerdown", (e) => { dragging = true; sash.classList.add("dragging"); sash.setPointerCapture(e.pointerId); });
    sash.addEventListener("pointerup", () => { dragging = false; sash.classList.remove("dragging"); });
    sash.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const w = side === "left" ? e.clientX : window.innerWidth - e.clientX;
      pane.style.flex = "0 0 " + Math.min(700, Math.max(160, w)) + "px";
      try { localStorage.setItem("codewalk." + pane.id, String(Math.round(w))); } catch (err) {}
    });
    try {
      const saved = Number(localStorage.getItem("codewalk." + pane.id));
      if (saved) pane.style.flex = "0 0 " + saved + "px";
    } catch (err) {}
  }
  makeSash($("sash1"), $("explorer"), "left");
  makeSash($("sash2"), $("chat"), "right");

  // ---------------- sync to the terminal session ----------------
  // One-way the other way: publish what was said here so the terminal Claude can pick it up.
  function setupSync() {
    const bar = $("syncbar"), now = $("syncNow"), close = $("syncClose");
    bar.hidden = false;
    const note = document.createElement("span");
    note.id = "syncnote";
    bar.appendChild(note);

    async function push(closing) {
      now.disabled = close.disabled = true;
      note.style.color = "";
      note.textContent = closing ? "Syncing and closing\u2026" : "Syncing\u2026";
      let res = null;
      try {
        res = await fetch("/api/sync", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ close: !!closing }),
        }).then((r) => r.json());
      } catch (e) {
        res = { ok: false, error: "Could not reach the server." };
      }
      if (closing && res && res.ok) {
        document.body.innerHTML =
          "<div style='padding:60px;font-family:system-ui;color:#858585;font-size:14px;line-height:1.7'>" +
          "Synced to the terminal \u2014 " + res.synced + " exchange(s) sent. You can close this tab.</div>";
        return;
      }
      now.disabled = close.disabled = false;
      if (res && res.ok) {
        note.style.color = "var(--green)";
        note.textContent = "sent " + res.synced;
        setTimeout(() => { note.textContent = ""; }, 4000);
      } else {
        note.style.color = "var(--red)";
        note.textContent = (res && res.error) || "sync failed";
      }
    }

    now.onclick = () => push(false);
    close.onclick = () => push(true);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); push(false); }
    });
  }

  // ---------------- boot ----------------
  fetch("/api/tree").then((r) => r.json()).then((d) => {
    allFiles = d.files;
    $("titleRoot").textContent = d.root + (d.context ? "   \u2014   " + d.context : "");
    $("expName").textContent = d.name;
    if (d.handoff) {
      const hb = $("handback");
      hb.hidden = false;
      hb.title = "Close this and hand the conversation back to the terminal";
      hb.onclick = async () => {
        hb.disabled = true;
        hb.textContent = "Handing back\u2026";
        try { await fetch("/api/handback", { method: "POST" }); } catch (e) {}
        document.body.innerHTML =
          "<div style='padding:60px;font-family:system-ui;color:#858585;font-size:14px;line-height:1.7'>" +
          "Handed back to the terminal. Everything said here has gone with you \u2014 " +
          "you can close this tab.</div>";
      };
    }
    if (d.handoff) setInterval(() => { fetch("/api/ping").catch(() => {}); }, 3000);
    if (d.sync) setupSync();
    renderTree();
    const b = bubble("Claude").querySelector(".body");
    b.innerHTML =
      (d.context
        ? "<p>Picking up from our terminal session \u2014 I already have the context of what we built here.</p>"
        : "<p>Ask me anything about <code>" + d.name + "</code>. I will read whatever files I need and " +
          "open them in the center pane, highlighting the lines as I explain them.</p>") +
      "<p class='notice'>Click a chip in my answer to jump there. Drag the line-number gutter or select code and press Ctrl-L to attach it to a question. Ctrl-P filters the tree.</p>";
    if (d.first_question) ask(d.first_question);
  });
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Local three-pane codebase walkthrough.")
    ap.add_argument("root", nargs="?", default=".", help="project root (default: .)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="opus", help="alias passed to `claude --model`")
    ap.add_argument(
        "--resume", metavar="SESSION", nargs="?", const="latest", default=None,
        help="start the chat from an existing Claude Code session so it already knows what you "
             "built together. A session id, or 'latest' (the default when the flag is bare) for "
             "the most recent session in --context-dir. The original is forked, never modified.",
    )
    ap.add_argument(
        "--context-dir", default=None, metavar="DIR",
        help="directory whose sessions '--resume latest' searches (default: the current directory)",
    )
    ap.add_argument(
        "--handoff", action="store_true",
        help="block until you press 'Return to terminal' in the page, then print the whole "
             "dashboard conversation to stdout so the calling session absorbs it. Implies --resume.",
    )
    ap.add_argument(
        "--transcript", metavar="PATH", default=None,
        help="write the running transcript here (default: alongside the log in /tmp)",
    )
    ap.add_argument("--wait", type=int, default=570, metavar="SECONDS",
                    help="how long --handoff blocks before giving up (default 570)")
    ap.add_argument("--brief", default="", metavar="TEXT",
                    help="instructions for the dashboard session: what to walk through and how")
    ap.add_argument("--brief-file", default=None, metavar="PATH",
                    help="a file of instructions for the dashboard session (added to --brief)")
    ap.add_argument("--start", default="", metavar="QUESTION",
                    help="ask this the moment the page opens, so the walkthrough begins by itself")
    ap.add_argument("--sync-file", default=None, metavar="PATH",
                    help="show Sync buttons that publish the conversation to PATH, for a terminal "
                         "session to pick up. Works with or without --handoff.")
    ap.add_argument("--idle-exit", type=int, default=15, metavar="SECONDS",
                    help="with --handoff, return to the terminal this long after the tab is closed")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if args.handoff and not args.resume:
        args.resume = "latest"

    if not os.path.isdir(args.root):
        sys.exit(f"codewalk: not a directory: {args.root}")

    parent = None
    if args.resume:
        ctx = args.context_dir or os.getcwd()
        parent = find_session(args.resume, ctx)
        if parent is None:
            sys.exit(f"codewalk: no Claude Code session found for {os.path.abspath(ctx)}")
        Handler.context_label = "forked from " + parent[:8]

    brief = args.brief
    if args.brief_file:
        try:
            with open(args.brief_file, encoding="utf-8") as fh:
                brief = (brief + "\n\n" + fh.read()).strip()
        except OSError as e:
            sys.exit(f"codewalk: could not read --brief-file: {e}")

    Handler.first_question = args.start
    Handler.sync_path = os.path.abspath(args.sync_file) if args.sync_file else None
    Handler.repo = Repo(args.root)
    Handler.claude = Claude(args.model, Handler.repo.root, parent, brief)
    n = len(Handler.repo.files())

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"codewalk  {Handler.repo.root}  ({n} files{', git' if Handler.repo.is_git else ''})")
    print(f"          {url}   (ctrl-c to stop)")
    if parent:
        print(f"          chat forks session {parent} (your original is not modified)")
    if Handler.sync_path:
        print(f"          sync button writes {Handler.sync_path}")
    if lex is None:
        print("          note: Pygments not found — showing plain text")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    if not args.handoff:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\ncodewalk: stopped")
        return

    # Handoff: block here so the terminal conversation pauses, then hand back everything
    # that was said in the page by printing it — the caller's next tool result IS the transcript.
    Handler.handoff = True
    Handler.transcript_path = args.transcript or f"/tmp/codewalk-{args.port}.md"
    print(f"          waiting for you in the browser — press 'Return to terminal' when done")
    print(f"          (transcript: {Handler.transcript_path})", flush=True)

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    returned, why = False, ""
    deadline = time.time() + args.wait
    try:
        while time.time() < deadline:
            if Handler.handback.wait(timeout=1.0):
                returned, why = True, "you pressed Return to terminal"
                break
            if Handler.last_ping:
                idle = time.time() - Handler.last_ping
                if idle > args.idle_exit:
                    returned, why = True, "the dashboard tab was closed"
                    break
    except KeyboardInterrupt:
        returned, why = True, "interrupted at the terminal"
    httpd.shutdown()

    print()
    print("=" * 72)
    if returned:
        print(f"RETURNED FROM DASHBOARD — {why}")
    else:
        print("STILL IN THE DASHBOARD (wait expired) — transcript so far; re-run to keep waiting")
    print("=" * 72)
    print()
    print(render_transcript(Handler.transcript))
    if not returned:
        sys.exit(3)


if __name__ == "__main__":
    main()
