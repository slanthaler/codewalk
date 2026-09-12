#!/usr/bin/env python3
"""
codewalk — a local three-pane workspace for talking about code: tree | file | chat.

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
import difflib
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


def cache_root() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "codewalk")


def default_shadow(root: str) -> str:
    """Outside the repo on purpose: the file tree comes from `git ls-files`, so a shadow
    inside the root would either pollute the tree or be invisible to it."""
    tag = hashlib.sha256(os.path.abspath(root).encode("utf-8")).hexdigest()[:8]
    return os.path.join(cache_root(), "shadow", f"{os.path.basename(root.rstrip(os.sep))}-{tag}")


class Shadow:
    """A sparse mirror of the repo holding proposals the user has not agreed to yet.

    Sparse is the point: a file exists here only because there is a pending proposal for it,
    so presence *is* the signal and no read ever has to ask which tree is authoritative.
    """

    def __init__(self, root: str, repo: "Repo"):
        self.root = os.path.abspath(root)
        self.repo = repo
        os.makedirs(self.root, exist_ok=True)

    def resolve(self, rel: str) -> str | None:
        rel = rel.lstrip("/")
        if not rel:
            return None
        candidate = os.path.realpath(os.path.join(self.root, rel))
        root = os.path.realpath(self.root)
        if candidate != root and not candidate.startswith(root + os.sep):
            return None
        return candidate

    def entries(self) -> list[dict]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                try:
                    new_text = open(full, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                base = self.repo.resolve(rel)
                old_text = ""
                is_new = True
                if base and os.path.isfile(base):
                    is_new = False
                    try:
                        old_text = open(base, encoding="utf-8", errors="replace").read()
                    except OSError:
                        old_text = ""
                added = removed = 0
                for line in difflib.unified_diff(
                    old_text.splitlines(), new_text.splitlines(), n=0, lineterm=""
                ):
                    if line.startswith("+") and not line.startswith("+++"):
                        added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        removed += 1
                out.append({
                    "path": rel, "new": is_new, "added": added, "removed": removed,
                    "same": (not is_new) and old_text == new_text,
                })
        out.sort(key=lambda e: e["path"])
        return out

    def read(self, rel: str) -> dict:
        full = self.resolve(rel)
        if full is None or not os.path.isfile(full):
            return {"error": f"No proposal for {rel}"}
        try:
            text = open(full, encoding="utf-8", errors="replace").read()
        except OSError as e:
            return {"error": f"Could not read the proposal: {e}"}
        base = self.repo.resolve(rel)
        old = ""
        is_new = True
        if base and os.path.isfile(base):
            is_new = False
            try:
                old = open(base, encoding="utf-8", errors="replace").read()
            except OSError:
                old = ""
        # Mark which lines are new relative to the repo, so the pane can rail them.
        sm = difflib.SequenceMatcher(None, old.splitlines(), text.splitlines())
        changed = []
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert"):
                changed.extend(range(j1 + 1, j2 + 1))
        return {
            "path": rel,
            "name": os.path.basename(rel),
            "digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "mtime": os.path.getmtime(full),
            "lines": tokenize(text, full),
            "proposed": True,
            "new": is_new,
            "changed": changed,
            "diff": self.diff_text(rel),
        }

    def diff_text(self, rel: str) -> list[dict]:
        """Unified diff as rows the chat/pane can render without a diff library."""
        full = self.resolve(rel)
        if full is None or not os.path.isfile(full):
            return []
        new_text = open(full, encoding="utf-8", errors="replace").read()
        base = self.repo.resolve(rel)
        old_text = ""
        if base and os.path.isfile(base):
            old_text = open(base, encoding="utf-8", errors="replace").read()
        rows = []
        for line in difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(),
            fromfile=rel, tofile=rel, lineterm="",
        ):
            if line.startswith("+++") or line.startswith("---"):
                continue
            kind = "hunk" if line.startswith("@@") else (
                "plus" if line.startswith("+") else "minus" if line.startswith("-") else "same"
            )
            rows.append({"k": kind, "t": line})
        return rows

    def apply(self, rel: str) -> dict:
        """Move a proposal into the repo, backing up whatever was there."""
        full = self.resolve(rel)
        if full is None or not os.path.isfile(full):
            return {"ok": False, "error": f"No proposal for {rel}"}
        dest = self.repo.resolve(rel)
        if dest is None:
            return {"ok": False, "error": "That path is outside the project."}
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isfile(dest):
                shutil.copyfile(dest, dest + ".codewalk.bak")
            shutil.copyfile(full, dest)
            os.remove(full)
            self._prune()
        except OSError as e:
            return {"ok": False, "error": f"Could not apply: {e}"}
        return {"ok": True, "path": rel}

    def discard(self, rel: str) -> dict:
        full = self.resolve(rel)
        if full is None or not os.path.isfile(full):
            return {"ok": False, "error": f"No proposal for {rel}"}
        try:
            os.remove(full)
            self._prune()
        except OSError as e:
            return {"ok": False, "error": f"Could not discard: {e}"}
        return {"ok": True, "path": rel}

    def _prune(self) -> None:
        """Drop directories the last proposal just left behind."""
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=False):
            if dirpath == self.root or dirnames or filenames:
                continue
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


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
You are Claude, working with someone on the codebase rooted at {root}. You are inside codewalk: a \
local three-pane dashboard they are looking at RIGHT NOW.

    left     the file tree
    center   the file you last opened, scrolled to the lines you last highlighted
    right    this conversation

The center pane is yours and it persists — whatever you opened last is still on their screen, so \
you can talk about it as something they can see. Never ask them to paste code, to open a file \
themselves, or to scroll somewhere; open it for them.
{inherited}
Use your Read, Grep and Glob tools freely to pull up whatever you need, and never cite a line you \
have not read — a wrong line number costs you their trust faster than a vague answer does. You \
cannot run commands; you can write files, under the rules in directive 5.

Each reply becomes a titled CELL in the right-hand pane, pinned so its first line sits at the top \
of their view while the rest fills in underneath. They start reading the moment your first words \
land, so those words must carry the point: no preamble, no restating the question, no throat \
clearing. They can also jump back through earlier cells by title, which is what the titles are for.

Answer what they actually asked, at the length that answer takes. This is an ordinary working \
conversation about code — a question, a debugging session, a design argument, a review, a change \
they want made — and the pane is what makes it better than a terminal, not a format to fill.

What the dashboard does change is that showing beats telling. Whatever they ask, put the relevant \
code in front of them: find it, open it at the exact lines with [[open:...]], and say what matters \
about it. "Which transformer architecture is this" is answered by opening the class and the two \
lines that give it away, not by describing them from memory. A short answer anchored on real lines \
is worth more than a long one that is not.

This holds when the code does not exist yet. If you are thinking through a new implementation with \
them, keep opening the code it would touch, sit next to, or replace — a design conversation \
grounded in the actual call sites stays honest in a way an abstract one does not.

Answer in plain prose. No headings, no bullet lists. A few short paragraphs at \
most unless they ask for depth. They are a strong engineer, so be concrete and skip the basics. \
Say why the code is the way it is — the shape of the trade-off, what would break if it were \
written the obvious way — rather than narrating what the lines they can already see literally do.

You drive their center pane with six directives, and you should use them constantly:

1. To point at code, write [[open:PATH:START-END|short label]] inline, exactly where you would \
otherwise have written "see model.py lines 120-140". PATH is relative to the project root. The pane \
opens that file, highlights the range and scrolls to it. A whole file is [[open:PATH|label]]. \
Several ranges in one file: [[open:PATH:10-20,44|label]]. Line numbers must be the real ones from \
the file you read.

Be sparing. Each one MOVES their screen, and a reader who is looking at the code cannot follow a \
pane that jumps every sentence — they end up reading none of it. One or two per part, on the lines \
that carry the point, and stay in that file while you talk about it. Naming a file or a function \
in prose is often enough; open it only when they need to see the lines to follow you.

2. To propose a change, write

[[edit:PATH:START-END]]
the complete replacement text for those lines
[[/edit]]

which the dashboard renders as a real diff — deletions and insertions, line by line — with an \
Apply button the user may or may not press. Give the full replacement for that line range, \
indented exactly as it must appear in the file.

Reach for this whenever a change is what you are discussing, not only when they ask you to make \
one. "What if this took a mask instead" is a diff; so is showing two ways to do the same thing as \
two blocks. Seeing the exact lines that would go and the exact lines that would arrive is what \
makes the discussion precise, and Apply stays theirs to ignore.


3. Write the WHOLE answer in one reply, at the length the answer actually takes. When that is \
long enough to need pacing, cut it into parts with

[[continue: what the next part covers]]

between them. The reader sees the first part and a Continue button; each press reveals the next \
part of what you already wrote. Nothing is re-run and nothing is generated on the press, so this \
costs you nothing — it is purely how fast the text arrives in front of them.

A part is ONE coherent idea, two or three short paragraphs, with one or two [[open:...]] \
directives on the lines that part is about. Cut where the subject changes, never mid-argument, and \
make the label say what comes next so the button is a real choice. Nothing follows the last part: \
no trailing directive, no summary of what you just said.

Short answers have no parts at all. A question with an answer gets the answer.

4. Begin EVERY reply — without exception, including short answers to questions — with

[[title: three to six words]]

on its own first line, naming what that reply is about. It is stripped from the prose and used as \
the heading on that cell, so they can scan back through the conversation and find it again. \
Name the specific subject, not the genre: "RevIN running statistics" or "why the mask is \
concatenated", never "Explanation" or "Next step". Do not repeat the title in the body text.

5. To draft a NEW file, or a rewrite too large to read as a diff, write it into the shadow \
directory at {shadow}. Mirror the repo's own layout inside it: a proposed \
src/models/encoder.py goes to {shadow}/src/models/encoder.py. The dashboard lists everything \
there as a pending proposal the user can open, diff against the real file, and apply or discard. \
Nothing you put there touches their code, so draft freely — but say in the chat what you wrote and \
why, because a file appearing with no explanation is not a discussion.

You may also write directly into the repo at {root}, and that is a real change to their code. Do \
it only when they have actually agreed to that specific change; while anything is still being \
discussed, draft it in the shadow instead. When in doubt, shadow.

6. To show code that is NOT in the repo — a proposed new function, a sketch, a snippet from \
elsewhere — use a fenced ``` block. That is the one place fences belong. Code that IS in the repo \
must be shown with [[open:...]] instead, so they see it in context in the center pane with the \
real line numbers around it.

The user can attach code to a question. When they do, the exact text arrives above their message \
under a REFERENCES heading, with the path and line numbers. That attached code is what they are \
asking about — answer about it directly, and do not re-read those lines unless you need surrounding \
context.
{brief}"""


VOICE_NOTE = """
THE USER IS LISTENING TO THIS ANSWER. Every word you write is spoken aloud by a synthetic voice
while they watch the code pane. You are not writing text that happens to be read out — you are
TALKING to someone who cannot see your sentence and cannot go back over it.

This changes how you write, not just which words you pick. Say a thing the way you would say it to
a colleague sitting next to you, out loud, with no whiteboard.

Never write a shape, signature or expression in symbols. Say what it IS:

  BAD:  inputs arrive as batch by variates by num_patches by patch_length (bvnp)
  GOOD: inputs arrive as a four dimensional tensor: batch, variate, patch number, and patch length

  BAD:  it reshapes (b v n d) to (b*n, v, d) and applies attention over v
  GOOD: it folds the patch axis into the batch axis, so attention runs across variates

  BAD:  a single full attention over (bv, n+v, d) would be quadratic
  GOOD: attending over sequence and variates at once would cost the square of both together

The precision belongs on the screen, not in the ear: point at the line with [[open:...]] and say in
words what it does. They can read the exact shape there while you talk.

Also: short sentences, subject first, no parentheses mid-sentence — a spoken aside has no brackets,
so make it its own sentence. No lists of symbols. No abbreviations they would have to unpack, like
"bvnp" or "MHA"; say the words.

When a sentence really must be written with symbols for the screen, put the spoken version after it
in a [[say: ...]] directive, and that is what will be read aloud in place of that sentence:

  The mixing transformer runs `(b*n, v, d)` attention separately. [[say: the mixing transformer
  runs attention across variates separately, with the patch axis folded into the batch.]]

Use [[say: ...]] sparingly — reformulating the sentence itself is almost always better.
"""


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


# Aliases `claude --model` accepts. The page offers these; --model can name anything
# else and it is added to the list so the session's own choice stays selectable.
MODELS = ["opus", "sonnet", "haiku"]


class Script:
    """A canned stand-in for Claude, so the page can be worked on without a model call.

    Duck-types the parts of Claude the handler touches. Answers come from a file of
    turns separated by "=== " lines and are served in order, the last one repeating;
    they stream out word by word because anything that looks at partial text -- the
    narration, the pane following, the live diff cards -- only misbehaves mid-stream.
    """

    def __init__(self, path: str, model: str = "demo"):
        self.model = model
        self.brief = ""
        self.pending = None
        self.i = 0
        self.lock = threading.Lock()
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        body = re.sub(r"\A(?:#[^\n]*\n|\s*\n)+", "", body)     # strip the file header
        turns = [t.strip() for t in re.split(r"^===[^\n]*\n", body, flags=re.M)]
        self.turns = [t for t in turns if t] or ["Nothing in the demo script."]

    @property
    def inherited(self) -> str:
        return ""

    def ask(self, prompt: str, system: str, emit):
        with self.lock:
            text = self.turns[min(self.i, len(self.turns) - 1)]
            self.i += 1
        emit("tool", "Read  demo-walkthrough.md")
        time.sleep(0.4)
        for word in re.findall(r"\S+\s*", text):
            emit("delta", word)
            time.sleep(0.035)
        emit("done", "")


class Claude:
    """One `claude -p` session over the repo, resumed across turns.

    When `parent` is given, the first turn forks that session instead of starting cold, so the
    chat inherits the terminal conversation's context. Forking (rather than resuming in place)
    means the user's own terminal session is never written to.
    """

    def __init__(self, model: str, root: str, parent: str | None = None, brief: str = "",
                 shadow: str | None = None):
        self.model = model
        self.root = root
        self.parent = parent
        self.shadow = shadow
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
            "--allowedTools", "Read", "Grep", "Glob", "Write", "Edit", "MultiEdit",
            "--disallowedTools", "NotebookEdit", "Bash", "WebFetch", "WebSearch", "Task",
        ]
        if self.shadow:
            argv += ["--add-dir", self.shadow]
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


# Whatever reads a transcript is an agent in a terminal, and an agent handed text ending
# in an unanswered question will answer it. So the transcript says what it is before it
# says anything else: someone else's conversation, delivered as context, not as work.
TRANSCRIPT_NOTE = """\
# HOW TO READ THIS
#
# What follows is a TRANSCRIPT, not a request. It is the conversation the user had with
# the codewalk session in their browser, delivered here so this session knows what was
# said. Nothing in it is addressed to you and nothing in it is assigned to you.
#
# Read it, say in one or two sentences that you have it and where things stand, and then
# STOP. Do not implement, verify, test, review, refactor or draft anything it mentions.
# The last message in it was written TO the user, not BY them: if it ends on a question,
# that question is still open and only the user can answer it. Pressing Sync is not an
# answer. Wait for them to type here.
"""


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
# voice
# ----------------------------------------------------------------------------------

PIPER_DIRS = ["~/.local/share/piper/piper", "~/.local/share/piper"]


def find_piper() -> tuple[str, str]:
    """(executable, model) for a local Piper install, or ("", "")."""
    exe = os.environ.get("CODEWALK_PIPER") or ""
    if not exe:
        for d in PIPER_DIRS:
            cand = os.path.join(os.path.expanduser(d), "piper")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                exe = cand
                break
        else:
            exe = shutil.which("piper") or ""
    if not exe:
        return "", ""
    model = os.environ.get("CODEWALK_PIPER_MODEL") or ""
    if not model:
        # Prefer a high-quality voice, then whatever is there, so an install with
        # several models does not depend on directory order.
        here = os.path.dirname(os.path.abspath(exe))
        onnx = sorted(glob.glob(os.path.join(here, "*.onnx")))
        if not onnx:
            return "", ""
        model = next((m for m in onnx if "-high" in m), onnx[0])
    return (exe, model) if os.path.isfile(model) else ("", "")


class Voice:
    """Piper, kept warm.

    Loading the model costs half a second, which is audible between every sentence, so
    each speaking rate gets a resident process fed line-delimited JSON. Piper writes the
    finished wav's path back on stdout, which is the completion signal. Rate cannot be
    set per line, hence one process per rate rather than one process.
    """

    MAX_PROCS = 3

    def __init__(self, exe: str, model: str):
        self.exe, self.model = exe, model
        self.procs: dict[float, dict] = {}
        self.lock = threading.Lock()
        self.dir = os.path.join(cache_root(), "tts")
        os.makedirs(self.dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.exe and self.model)

    def _proc(self, scale: float) -> dict:
        p = self.procs.get(scale)
        if p and p["proc"].poll() is None:
            p["used"] = time.time()
            return p
        if len(self.procs) >= self.MAX_PROCS:
            old = min(self.procs, key=lambda k: self.procs[k]["used"])
            self._kill(old)
        proc = subprocess.Popen(
            [self.exe, "--model", self.model, "--json-input", "--quiet",
             "--length_scale", f"{scale:.3f}", "--sentence_silence", "0.1"],
            cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        p = {"proc": proc, "lock": threading.Lock(), "used": time.time()}
        self.procs[scale] = p
        return p

    def _kill(self, scale: float) -> None:
        p = self.procs.pop(scale, None)
        if not p:
            return
        try:
            p["proc"].stdin.close()
            p["proc"].terminate()
        except Exception:
            pass

    def say(self, text: str, rate: float) -> bytes:
        """Synthesize one chunk. Returns wav bytes, or b'' if Piper failed."""
        text = " ".join(text.split())
        if not text:
            return b""
        scale = round(min(2.0, max(0.4, 1.0 / max(0.3, rate))), 3)
        for attempt in (1, 2):                 # a dead process is respawned once
            with self.lock:
                p = self._proc(scale)
            out = os.path.join(self.dir, f"cw{os.getpid()}-{threading.get_ident()}.wav")
            try:
                with p["lock"]:
                    p["proc"].stdin.write(json.dumps({"text": text, "output_file": out}) + "\n")
                    p["proc"].stdin.flush()
                    line = p["proc"].stdout.readline()
                if not line:
                    raise OSError("piper closed")
                with open(out, "rb") as fh:
                    data = fh.read()
                return data
            except Exception:
                with self.lock:
                    self._kill(scale)
                if attempt == 2:
                    return b""
            finally:
                try:
                    os.unlink(out)
                except OSError:
                    pass
        return b""

    def shutdown(self) -> None:
        with self.lock:
            for scale in list(self.procs):
                self._kill(scale)

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
    shadow: "Shadow" = None      # type: ignore[assignment]
    first_question: str = ""
    models: list = []
    voice: "Voice" = None      # type: ignore[assignment]
    voice_on: bool = False     # the page is reading answers aloud

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
                "tts": Handler.voice.available,
                "model": self.claude.model,
                "models": Handler.models,
                "files": self.repo.files(),
            })
        elif path == "/api/file":
            res = self.repo.read(self._query().get("path", ""))
            self._json(res, 200 if "error" not in res else 404)
        elif path == "/api/proposals":
            self._json({"items": Handler.shadow.entries(), "root": Handler.shadow.root})
        elif path == "/api/proposal":
            res = Handler.shadow.read(self._query().get("path", ""))
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
        elif path == "/api/proposal/apply":
            b = self._body()
            self._json(self._proposal_bulk(b, Handler.shadow.apply))
        elif path == "/api/proposal/discard":
            b = self._body()
            self._json(self._proposal_bulk(b, Handler.shadow.discard))
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
            Handler.voice_on = bool(b.get("voice"))
            self._ask(b.get("q", ""), b.get("context", ""))
        elif path == "/api/tts":
            b = self._body()
            try:
                rate = float(b.get("rate") or 1.0)
            except (TypeError, ValueError):
                rate = 1.0
            wav = Handler.voice.say(str(b.get("text") or ""), rate)
            if not wav:
                self._json({"ok": False, "error": "tts failed"}, 503)
                return
            self._send(200, "audio/wav", wav)
        elif path == "/api/model":
            b = self._body()
            want = str(b.get("model") or "")
            if want not in Handler.models:
                self._json({"ok": False, "error": "unknown model"}, 400)
                return
            self.claude.model = want
            self._json({"ok": True, "model": want})
        else:
            self._send(404, "text/plain", b"not found")

    def _proposal_bulk(self, b: dict, fn) -> dict:
        """Run apply/discard over one path or every pending proposal."""
        if b.get("all"):
            paths = [e["path"] for e in Handler.shadow.entries()]
        else:
            paths = [str(b.get("path") or "")]
        done, failed = [], []
        for rel in paths:
            res = fn(rel)
            (done if res.get("ok") else failed).append(res.get("path") or rel)
        return {"ok": not failed, "done": done, "failed": failed,
                "left": len(Handler.shadow.entries())}

    def _system(self) -> str:
        # The brief still outranks everything; the voice note sits just above it because
        # how to phrase an answer is a smaller matter than what the answer is about.
        brief = (VOICE_NOTE + self.claude.brief) if Handler.voice_on else self.claude.brief
        return RULES.format(
            root=self.repo.root, inherited=self.claude.inherited, brief=brief,
            shadow=Handler.shadow.root,
        )

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
        system = self._system()
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

    def _event(self, kind: str, payload: str) -> None:
        try:
            self.wfile.write(f"data: {json.dumps({'k': kind, 'v': payload})}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @classmethod
    def sync(cls, closing: bool) -> dict:
        """Publish everything said since the last sync, for the terminal session to pick up."""
        if not cls.sync_path:
            return {"ok": False, "error": "This dashboard was not started with --sync-file."}
        fresh = cls.transcript[cls.synced_upto:]
        if not fresh and not closing:
            return {"ok": False, "error": "Nothing new to sync."}
        header = (
            TRANSCRIPT_NOTE
            + "#\n# Synced from the codewalk dashboard"
            + (" (session closed)" if closing else " — the user is still in the browser")
            + f", {len(fresh)} new exchange(s).\n\n"
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
/* The proposals list is the bottom half of the explorer, not an overlay: pending work
   belongs next to the tree it will change, and must never cover the conversation. */
#propPane {
  flex: 0 0 auto; max-height: 45%; min-height: 0; display: flex; flex-direction: column;
  border-top: 1px solid var(--border); background: var(--bg-side);
}
.ptHead { display: flex; align-items: center; gap: 6px; padding: 6px 8px;
          border-bottom: 1px solid var(--border); font-size: 10px; letter-spacing: 0.09em;
          text-transform: uppercase; color: var(--fg-faint); }
.ptHead b { color: var(--hl-rail); font-weight: 700; }
.ptHead .spacer { flex: 1; }
.ptHead .btn { font-size: 10px; padding: 2px 7px; }
#ptList { overflow-y: auto; flex: 1 1 auto; min-height: 0; }
.ptRow { display: flex; align-items: center; gap: 6px; padding: 5px 8px;
         border-bottom: 1px solid #333; font-size: 11.5px; }
.ptRow:hover { background: var(--bg-hover); }
.ptRow .p { flex: 1; font-family: var(--mono); font-size: 11px; color: var(--fg-dim);
            cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            direction: rtl; unicode-bidi: plaintext; text-align: left; }
.ptRow .p:hover { color: #fff; }
.ptRow .tag { font-size: 9px; letter-spacing: 0.06em; text-transform: uppercase;
              color: var(--hl-rail); }
.ptRow .n { font-family: var(--mono); font-size: 10px; }
.ptRow .n .a { color: var(--green); }
.ptRow .n .r { color: var(--red); }
.ptRow button { font: inherit; font-size: 10px; font-family: var(--ui); border: 0;
                border-radius: 2px; padding: 1px 6px; cursor: pointer; }
.ptRow .ok { background: var(--btn); color: #fff; }
.ptRow .no { background: transparent; color: var(--fg-faint); border: 1px solid var(--border); }
.ptFoot { padding: 4px 8px; font-size: 9.5px; color: var(--fg-faint);
          font-family: var(--mono); border-top: 1px solid var(--border);
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; unicode-bidi: plaintext; }
.tab .prop { color: var(--hl-rail); font-size: 9px; letter-spacing: 0.06em; margin-left: 4px; }
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
select.iconbtn { padding: 2px 4px; text-transform: none; letter-spacing: 0; }
.iconbtn.wants { color: var(--hl-rail); border-color: var(--hl-rail); }
.rateWrap { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--fg-faint); }
.rateWrap input { width: 74px; accent-color: var(--hl-rail); cursor: pointer; }
.rateWrap #rateVal { font-family: var(--mono); min-width: 34px; text-align: right; }

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
#logwrap { flex: 1 1 auto; position: relative; min-height: 0; display: flex; }
#log { flex: 1 1 auto; overflow-y: auto; padding: 14px 16px 6px; }
#jumpDown {
  position: absolute; top: 8px; left: 50%; transform: translateX(-50%); z-index: 5;
  font: inherit; font-size: 11px; font-family: var(--ui); background: var(--btn); color: #fff;
  border: 0; border-radius: 10px; padding: 3px 11px; cursor: pointer; opacity: 0.92;
  box-shadow: 0 1px 6px rgba(0,0,0,0.45);
}
#jumpDown:hover { background: var(--btn-hover); opacity: 1; }
.msg { margin-bottom: 18px; scroll-margin-top: 0; }
/* Grows so the last exchange can still be pulled to the top of the viewport. */
#tailpad { height: 0; }
.msg.cell { transition: background .35s; border-radius: 2px; }
.msg.cell.jumped { background: #2a2f38; }
#cellPrev, #cellNext { font-size: 13px; line-height: 1; padding: 2px 6px; }
.who { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fg-faint); margin-bottom: 5px; }
.msg.me .who { color: #569cd6; }
.who.titled { color: var(--fg-dim); font-size: 11px; letter-spacing: 0.04em; text-transform: none;
              font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
.msg.me .body { background: #2d2d2d; border-left: 2px solid var(--accent); padding: 7px 10px; white-space: pre-wrap; color: var(--fg-dim); font-size: 12.5px; }
.body { line-height: 1.65; }
.body p { margin: 0 0 0.7em; }
.body p:last-child { margin-bottom: 0; }
.body code { font-family: var(--mono); font-size: 12px; background: #2d2d2d; padding: 1px 4px; border-radius: 2px; color: #ce9178; }
.body strong { font-weight: 600; color: #fff; }
.body em { font-style: italic; color: var(--fg); }
.chip {
  font-family: var(--mono); font-size: 11.5px; background: var(--chip-bg); color: var(--hl-rail);
  border: 1px solid #5a4a20; border-radius: 2px; padding: 1px 5px; cursor: pointer; white-space: nowrap;
}
.chip:hover { background: var(--hl-rail); color: #1e1e1e; }
.acts { margin-bottom: 8px; display: flex; flex-direction: column; gap: 2px; }
.act { font-family: var(--mono); font-size: 11px; color: var(--fg-faint); display: flex; gap: 6px; }
.act::before { content: "\203A"; color: var(--green); }
.act.replay { align-self: flex-start; cursor: pointer; background: none; border: 0; padding: 0;
              color: var(--fg-faint); }
.act.replay::before { content: none; }
.act.replay:hover { color: var(--hl-rail); }
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

.snip {
  font-family: var(--mono); font-size: 12px; line-height: 1.5; margin: 9px 0; padding: 8px 10px;
  background: var(--bg-editor); border: 1px solid var(--border); border-left: 2px solid #6a9955;
  border-radius: 2px; overflow-x: auto; white-space: pre; color: var(--fg-dim);
}
.snip.live { opacity: 0.75; }
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
    <div id="propPane" hidden>
      <div class="ptHead"><span>Proposed <b id="propN">0</b></span><span class="spacer"></span><button id="ptAll" class="btn">Apply all</button><button id="ptNone" class="btn ghost">Discard</button></div>
      <div id="ptList"></div>
      <div class="ptFoot" id="ptFoot"></div>
    </div>
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
    <div class="paneHead"><span class="spacer"></span><select class="iconbtn" id="model" title="Which model answers. Takes effect on the next question; the conversation so far is kept."></select><button class="iconbtn" id="voice" title="Read the answers aloud and move the editor in time with the voice">Voice: off</button><span class="rateWrap" id="rateWrap" hidden title="Speaking speed — drag, or Alt+, / Alt+. "><input type="range" id="rate" min="0.5" max="2" step="0.25" value="1" aria-label="Speaking speed"><span id="rateVal">1×</span></span><button class="iconbtn" id="follow" title="Let Claude move the editor as it explains">Follow: on</button><button class="iconbtn" id="clearHl" disabled>Clear highlight</button><button class="iconbtn" id="cellPrev" title="Previous step (Alt+Up)" disabled>&#8593;</button><button class="iconbtn" id="cellNext" title="Next step (Alt+Down)" disabled>&#8595;</button></div>
    <div id="logwrap">
      <button id="jumpDown" hidden title="Jump to the latest step">&#8595; latest</button>
      <div id="log"><div id="tailpad"></div></div>
    </div>
    <div id="composer">
      <div id="ctx" hidden></div>
      <div class="refbar" id="refs" hidden></div>
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
  // entry here, so the arrows replay the visit the way browser history replays a session.
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
    const isProp = path.startsWith(PROP);
    if (!t) {
      const rel = isProp ? path.slice(PROP.length) : path;
      const url = (isProp ? "/api/proposal?path=" : "/api/file?path=") + encodeURIComponent(rel);
      const data = await fetch(url).then((r) => r.json());
      if (data.error) { flash(data.error); return null; }
      t = { path: path, data: data, hl: new Set(), sel: null, scroll: 0, prop: isProp };
      openTabs.push(t);
    }
    if (active && active !== path) {
      const prev = tab(active);
      if (prev) prev.scroll = $("scroll").scrollTop;
    }
    active = path;
    if (spec) { t.hl = parseSpec(spec); }
    // Opening a proposal rails exactly the lines that differ from the repo.
    else if (isProp && t.data.changed && !t.hl.size) t.hl = new Set(t.data.changed);
    // A proposal has no place in the repo tree and nothing to watch on disk.
    if (!isProp) {
      expandTo(path);
      renderTree();
    }
    renderTabs();
    renderCode(t, !!spec);
    if (!isProp) watchFile(t);
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
      if (t.prop) {
        const pr = document.createElement("span");
        pr.className = "prop";
        pr.textContent = t.data.new ? "NEW" : "PROPOSED";
        n.appendChild(pr);
      }
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
  // The spoken form of a sentence that had to be written in symbols. It never shows in
  // the chat; the narrator swaps it in for the sentence it follows.
  const SAY_RE = /\[\[say:\s*([\s\S]*?)\]\]/g;
  // The same directive, used to cut one answer into the parts a reader walks through.
  const CONT_G = /\[\[continue(?::\s*([^\]]*?))?\s*\]\]/g;   // a part break

  // Every [[continue:]] in the text, as {at, end, label}.
  function breaks(text) {
    const out = [];
    let m;
    CONT_G.lastIndex = 0;
    while ((m = CONT_G.exec(text))) {
      out.push({ at: m.index, end: m.index + m[0].length, label: (m[1] || "").trim() });
    }
    return out;
  }

  // The first `n` parts of an answer, as one piece of text.
  function revealed(text, n) {
    const bs = breaks(text);
    return n > bs.length ? text : text.slice(0, bs[n - 1].at);
  }

  // The label on the button that reveals part n+1, or null when there is no more.
  function breakLabel(text, n) {
    const bs = breaks(text);
    return n <= bs.length ? bs[n - 1].label : null;
  }
  const TITLE_RE = /\[\[title:\s*([^\]]*?)\s*\]\]/;

  function renderReply(el, text) {
    el.textContent = "";
    text = text.replace(CONT_G, "").replace(TITLE_RE, "")     // control signals, not prose
               .replace(SAY_RE, "");
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

  // A fenced block is code that is NOT in the repo yet — a proposal, a sketch, an
  // alternative. Code that IS in the repo belongs in the center pane via [[open:]].
  const FENCE_RE = /```[\w.+-]*\n?([\s\S]*?)```/g;

  function snippet(code, live) {
    const pre = document.createElement("pre");
    pre.className = "snip" + (live ? " live" : "");
    pre.textContent = code.replace(/\n$/, "");
    return pre;
  }

  function prose(el, text) {
    if (!text.trim()) return;
    let last = 0, m;
    FENCE_RE.lastIndex = 0;
    while ((m = FENCE_RE.exec(text))) {
      paras(el, text.slice(last, m.index));
      el.appendChild(snippet(m[1], false));
      last = m.index + m[0].length;
    }
    const tail = text.slice(last);
    // A fence still being written: show it as a block immediately rather than as
    // paragraphs full of backticks that reflow the moment it closes.
    const openFence = tail.search(/```[\w.+-]*\n/);
    if (openFence !== -1) {
      paras(el, tail.slice(0, openFence));
      el.appendChild(snippet(tail.slice(openFence).replace(/^```[\w.+-]*\n?/, ""), true));
    } else {
      paras(el, tail);
    }
  }

  function paras(el, text) {
    if (!text.trim()) return;
    for (const para of text.split(/\n{2,}/)) {
      if (!para.trim()) continue;
      const p = document.createElement("p");
      // Directives are resolved FIRST, because a label may itself contain backticks
      // -- [[open:path|`thing`]] is normal for Claude to write, and splitting on inline
      // code before matching would tear the directive in half and print it raw. A
      // directive that is entirely wrapped in backticks is still just quoted syntax,
      // and scanDirectives leaves it as code.
      scanDirectives(p, para);
      el.appendChild(p);
    }
  }

  // Bold and italic. Underscore forms are deliberately NOT supported: identifiers
  // like input_patch_len appear constantly in this prose and would false-italicize.
  const EMPH_RE = /(\*\*[^*\n]+\*\*|\*[^*\n]+\*)/;

  function emitText(p, text) {
    if (!text) return;
    for (const bit of text.split(EMPH_RE)) {
      if (!bit) continue;
      if (bit.length > 4 && bit.startsWith("**") && bit.endsWith("**")) {
        const b = document.createElement("strong");
        b.textContent = bit.slice(2, -2);
        p.appendChild(b);
      } else if (bit.length > 2 && bit.startsWith("*") && bit.endsWith("*")) {
        const i = document.createElement("em");
        i.textContent = bit.slice(1, -1);
        p.appendChild(i);
      } else {
        p.appendChild(document.createTextNode(bit));
      }
    }
  }

  // Inline code and emphasis, for the stretches between directives.
  function codeAndEmph(p, text) {
    if (!text) return;
    for (const bit of text.split(/(`[^`\n]+`)/)) {
      if (!bit) continue;
      if (bit.length > 2 && bit.startsWith("`") && bit.endsWith("`")) {
        const c = document.createElement("code");
        c.textContent = bit.slice(1, -1);
        p.appendChild(c);
      } else emitText(p, bit);
    }
  }

  // Inline-code spans, the one place both the renderer and the narrator agree on
  // what counts as a directive: one that sits INSIDE a span is quoted syntax, one that
  // merely CONTAINS a span in its label is a real directive.
  function codeSpans(text) {
    const spans = [];
    const re = /`[^`\n]+`/g;
    let m;
    while ((m = re.exec(text))) spans.push([m.index, m.index + m[0].length]);
    return spans;
  }

  function scanDirectives(p, text) {
    const spans = codeSpans(text);
    let last = 0, m;
    OPEN_RE.lastIndex = 0;
    while ((m = OPEN_RE.exec(text))) {
      const path = m[1], spec = m[2] || "", label = (m[3] || "").replace(/`/g, "");
      if (!isPath(path)) continue;
      // Quoted syntax: left where it is, and rendered as the code it is.
      if (spans.some(([a, b]) => m.index >= a && m.index + m[0].length <= b)) continue;
      codeAndEmph(p, text.slice(last, m.index));
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
    codeAndEmph(p, text.slice(last));
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
    logEl.insertBefore(m, $("tailpad"));
    return m;
  }

  // ---------------- proposals ----------------
  // Files the agent drafted in the shadow tree. They are real files on disk, but outside
  // the repo, so nothing the user is reading changes until they say so here.
  const PROP = "@proposed/";
  let propItems = [];

  async function refreshProposals() {
    let d = null;
    try { d = await fetch("/api/proposals").then((r) => r.json()); } catch (e) { return; }
    propItems = d.items || [];
    // The section simply is not there when nothing is pending — no empty state to ignore.
    $("propPane").hidden = propItems.length === 0;
    $("propN").textContent = propItems.length;
    $("ptFoot").textContent = d.root || "";
    renderProposals();
  }

  function renderProposals() {
    const list = $("ptList");
    list.textContent = "";
    for (const it of propItems) {
      const row = document.createElement("div");
      row.className = "ptRow";

      const p = document.createElement("span");
      p.className = "p";
      p.textContent = it.path;
      p.title = "Open this proposal in the editor";
      p.onclick = () => openFile(PROP + it.path);

      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = it.new ? "new" : (it.same ? "identical" : "changed");

      const n = document.createElement("span");
      n.className = "n";
      if (!it.new) {
        const a = document.createElement("span"); a.className = "a"; a.textContent = "+" + it.added;
        const r = document.createElement("span"); r.className = "r"; r.textContent = " -" + it.removed;
        n.append(a, r);
      }

      const ok = document.createElement("button");
      ok.className = "ok"; ok.textContent = "Apply";
      ok.onclick = () => propAct("/api/proposal/apply", { path: it.path });

      const no = document.createElement("button");
      no.className = "no"; no.textContent = "Discard";
      no.onclick = () => propAct("/api/proposal/discard", { path: it.path });

      row.append(p, tag, n, ok, no);
      list.appendChild(row);
    }
  }

  async function propAct(url, body) {
    let res = null;
    try {
      res = await fetch(url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
    } catch (e) { flash("Could not reach the server."); return; }
    if (res && res.failed && res.failed.length) flash("Failed: " + res.failed.join(", "));
    // Applied files change the repo underneath any tab showing them.
    for (const t of openTabs.slice()) {
      if (t.path.startsWith(PROP)) closeTab(t.path);
    }
    for (const t of openTabs.slice()) {
      if ((res.done || []).includes(t.path)) { closeTab(t.path); openFile(t.path); }
    }
    refreshProposals();
  }

  function setupProposals() {
    $("ptAll").onclick = () => propAct("/api/proposal/apply", { all: true });
    $("ptNone").onclick = () => propAct("/api/proposal/discard", { all: true });
    refreshProposals();
  }

  // ---------------- cells ----------------
  // Each exchange is a cell. A new answer is pinned near the TOP of the viewport, not the
  // bottom, so reading starts immediately and the text fills in below rather than pushing
  // the eye down the page. TOP_FRAC of the pane is left above it so the question stays visible.
  const TOP_FRAC = 0.18;

  function cells() { return Array.from(logEl.querySelectorAll(".msg.cell")); }

  // The last cell can only be pulled up if there is something under it to scroll into.
  function sizePad() {
    const c = cells();
    const last = c[c.length - 1];
    const pad = $("tailpad");
    if (!last) { pad.style.height = "0px"; return; }
    const room = logEl.clientHeight * (1 - TOP_FRAC) - last.offsetHeight;
    pad.style.height = Math.max(0, room) + "px";
  }

  // The pin is HELD, not applied once: an answer that is still being written (or has not
  // started arriving yet) changes height constantly, and a single scrollTop assignment at
  // the wrong moment silently clamps. Re-asserting on every frame of growth is what makes
  // "click Continue while it is still thinking" land in the right place.
  let pinTarget = null, pinHeld = false, cellIdx = -1;

  function holdPin(msg) {
    pinTarget = msg;
    pinHeld = true;
    applyPin();
  }

  function applyPin() {
    if (!pinHeld || !pinTarget || !pinTarget.isConnected) return;
    sizePad();
    const delta = pinTarget.getBoundingClientRect().top - logEl.getBoundingClientRect().top;
    logEl.scrollTop += delta - logEl.clientHeight * TOP_FRAC;
  }

  function releasePin() { pinHeld = false; }

  function atLatest() {
    const c = cells();
    if (!c.length) return true;
    const last = c[c.length - 1];
    const top = last.getBoundingClientRect().top - logEl.getBoundingClientRect().top;
    return top < logEl.clientHeight * 0.75;
  }

  function updateJump() {
    $("jumpDown").hidden = atLatest();
  }

  function gotoCell(i) {
    const c = cells();
    if (!c.length) return;
    cellIdx = Math.max(0, Math.min(c.length - 1, i));
    const m = c[cellIdx];
    holdPin(m);
    releasePin();                 // a deliberate jump is a destination, not a follow
    m.classList.add("jumped");
    setTimeout(() => m.classList.remove("jumped"), 400);
    syncCellNav();
    updateJump();
  }

  function syncCellNav() {
    const n = cells().length;
    $("cellPrev").disabled = !(n && cellIdx > 0);
    $("cellNext").disabled = !(n && cellIdx < n - 1);
  }

  function setupCells() {
    $("cellPrev").onclick = () => gotoCell(cellIdx - 1);
    $("cellNext").onclick = () => gotoCell(cellIdx + 1);
    $("jumpDown").onclick = () => gotoCell(cells().length - 1);
    // Only real input releases the pin — never our own scrollTop writes.
    for (const ev of ["wheel", "touchmove", "mousedown"]) {
      logEl.addEventListener(ev, releasePin, { passive: true });
    }
    logEl.addEventListener("scroll", updateJump, { passive: true });
    window.addEventListener("keydown", (e) => {
      if (e.altKey && !e.ctrlKey && !e.metaKey) {
        if (e.key === "ArrowUp") { e.preventDefault(); gotoCell(cellIdx - 1); return; }
        if (e.key === "ArrowDown") { e.preventDefault(); gotoCell(cellIdx + 1); return; }
      }
      if (document.activeElement === logEl && /^(Arrow|Page|Home|End)/.test(e.key)) releasePin();
    });
    window.addEventListener("resize", () => { sizePad(); applyPin(); updateJump(); });
    syncCellNav();
    updateJump();
  }

  // The answer arrives whole, split into parts by [[continue: ...]]. The button walks
  // through what is already written -- no second call, nothing to guess, and the reader
  // sets the pace instead of the model.
  function addContinue(msg, label, onclick) {
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
    btn.dataset.label = label || "";
    btn.onclick = onclick;
    const hint = document.createElement("span");
    hint.className = "kbd";
    hint.textContent = "or press Enter";
    bar.append(btn, hint);
    msg.appendChild(bar);
    sizePad();      // the button is part of the cell; never yank the view to it
    applyPin();
  }

  // The "who" label doubles as the cell heading once a title arrives.
  function setCellTitle(msg, title) {
    if (!title || msg.dataset.title === title) return;
    msg.dataset.title = title;
    const w = msg.querySelector(".who");
    w.textContent = title;
    w.classList.add("titled");
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

  // ---------------- narration ----------------
  // The chat is read aloud while the center pane follows the voice, not the stream:
  // an [[open:...]] fires the moment speech reaches the spot in the sentence where it
  // was written, which is how a person points at code while talking about it.
  const synth = window.speechSynthesis || null;
  let narrOn = localStorage.getItem("cw.voice") === "1";
  // Quarter steps, so the slider lands on a speed you can name rather than on 1.13.
  const RATE_STEP = 0.25, RATE_MIN = 0.5, RATE_MAX = 2;
  const snapRate = (v) => Math.min(RATE_MAX, Math.max(RATE_MIN,
                            Math.round(v / RATE_STEP) * RATE_STEP));
  let narrRate = snapRate(Number(localStorage.getItem("cw.rate")) || 1);
  let narrVoice = null;
  let narrQ = [];          // queued {say, fire, idx, scope}
  let narrBusy = false;
  let narrToken = 0;       // bumped on every stop; stale callbacks check it
  let narrCurrent = null;  // the chunk in the speaker's mouth, so a rate change can repeat it

  const VOICE_RANK = ["ryan", "lessac", "piper", "google us english", "microsoft aria", "samantha",
                      "english (america)", "en-us", "english"];
  function pickVoice() {
    if (!synth) return;
    const vs = synth.getVoices().filter((v) => /^en/i.test(v.lang));
    if (!vs.length) return;
    const want = localStorage.getItem("cw.voiceName");
    narrVoice = vs.find((v) => v.name === want) ||
      VOICE_RANK.map((k) => vs.find((v) => (v.name + " " + v.lang).toLowerCase().includes(k)))
                .find(Boolean) || vs[0];
  }
  if (synth) { pickVoice(); synth.onvoiceschanged = pickVoice; }

  // Two engines. When the server has Piper, sentences are synthesized server-side and
  // played as audio, and the NEXT ones are synthesized while the current one plays --
  // that lookahead is what removes the silence between sentences. Without Piper it
  // falls back to the browser's own speechSynthesis, which cannot be run ahead.
  let ttsOK = false;
  const audioEl = new Audio();
  const AHEAD = 3;
  // A failed synthesis used to set ttsOK = false for good, so one hiccup -- a server
  // restarted under an open tab, say -- silently moved the rest of the answer onto the
  // browser's own speech engine, mid-paragraph, at a different speed and in a different
  // voice. Now a failure costs that one sentence its retry, and the server is tried
  // again for the next; only a run of them gives up, and then it says so.
  let ttsFails = 0;
  const TTS_GIVE_UP = 3;
  // Whether this server HAS Piper, as opposed to whether it is answering right now.
  // If it has it, a failure is transient and the answer is to wait, not to switch
  // engines mid-paragraph: on this machine the browser's own speech also routes to
  // Piper through speech-dispatcher, so the fallback sounds like the same voice
  // coming apart rather than like a different one. The queue is held instead.
  let ttsServer = false;
  let narrHeld = false;

  // Chrome refuses audio.play() until the page has been clicked, and a reload with
  // Voice already on has no click yet. The old speechSynthesis path was never gated
  // this way, so this has to be handled rather than assumed: a refusal must not eat
  // the sentence -- it goes back on the queue and waits for the first gesture.
  let narrBlocked = false;
  function narrUnblock() {
    if (!narrBlocked) return;
    narrBlocked = false;
    voiceBtn.classList.remove("wants");
    setVoiceBtn();
    narrPump();
  }
  for (const ev of ["pointerdown", "keydown"]) {
    document.addEventListener(ev, narrUnblock, true);
  }

  function narrDrop(it) {
    if (it.ctrl) { try { it.ctrl.abort(); } catch (e) {} }
    if (it.url) { URL.revokeObjectURL(it.url); it.url = null; }
    for (const t of it.timers || []) clearTimeout(t);
    it.timers = null;
    it.ctrl = null; it.wav = null;
  }

  // Light the chips this sentence points at: the first one as it starts, the rest
  // scheduled against the audio's own duration.
  function narrMarks(it, duration) {
    it.timers = [];
    for (const mk of it.marks || []) {
      const at = duration && mk.frac > 0.02 ? duration * mk.frac : 0;
      if (!at) { mk.fire(); continue; }
      it.timers.push(setTimeout(mk.fire, at * 1000));
    }
  }

  function narrStop() {
    narrToken++;
    for (const it of narrQ) narrDrop(it);
    if (narrCurrent) narrDrop(narrCurrent);
    narrQ = [];
    narrBusy = false;
    narrCurrent = null;
    narrHeld = false;
    audioEl.pause();
    audioEl.removeAttribute("src");
    if (synth) synth.cancel();
  }

  function narrEnqueue(items) {
    if (!narrOn || !items.length) return;
    for (const it of items) narrQ.push(it);
    narrAhead();
    narrPump();
  }

  // Start synthesis for the next few queued sentences. The rate is baked into the
  // audio, so anything fetched at a stale rate is thrown away when the slider moves.
  function narrAhead() {
    if (!ttsOK) return;
    let n = 0;
    for (const it of narrQ) {
      if (n++ >= AHEAD) break;
      narrSynth(it);
    }
  }

  function narrSynth(it) {
    if (!it.say || it.wav) return;
    it.rate = narrRate;
    it.ctrl = new AbortController();
    const ctrl = it.ctrl;
    it.wav = fetch("/api/tts", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: it.say, rate: it.rate }), signal: ctrl.signal,
    }).then((r) => (r.ok ? r.blob() : null)).catch(() => null);
  }

  function narrPump() {
    if (narrBusy || narrBlocked || narrHeld || !narrQ.length) return;
    const it = narrQ.shift();
    if (!it.say) {                       // a directive with nothing to say around it
      narrMarks(it, 0);
      narrPump();
      return;
    }
    narrBusy = true;
    narrCurrent = it;
    const tok = narrToken;
    const next = () => {
      if (tok !== narrToken) return;     // a stop happened while we were talking
      narrDrop(it);
      narrBusy = false;
      narrCurrent = null;
      narrPump();
    };
    narrAhead();

    if (ttsOK) {
      narrSynth(it);
      it.wav.then((blob) => {
        if (tok !== narrToken) return;
        if (!blob) {
          if (!it.retried) {                 // one hiccup: ask again for this sentence
            it.retried = true;
            it.wav = null;
            narrQ.unshift(it);
            narrBusy = false;
            narrCurrent = null;
            narrPump();
            return;
          }
          if (++ttsFails >= TTS_GIVE_UP) {
            if (ttsServer) {          // it has Piper; hold the queue until it answers
              narrQ.unshift(it);
              it.wav = null;
              it.retried = false;
              narrBusy = false;
              narrCurrent = null;
              narrHeld = true;
              setVoiceBtn();
              flash("voice: the server stopped answering — click Voice to retry");
              return;
            }
            ttsOK = false;            // no Piper here at all: the browser is all there is
            setVoiceBtn();
            flash("voice: the server stopped answering");
          }
          next();
          return;
        }
        ttsFails = 0;
        it.url = URL.createObjectURL(blob);
        audioEl.src = it.url;
        audioEl.playbackRate = 1;
        audioEl.onended = next;
        audioEl.onerror = next;
        audioEl.onloadedmetadata = () => {
          if (tok !== narrToken || it.timers) return;
          narrMarks(it, isFinite(audioEl.duration) ? audioEl.duration : 0);
        };
        audioEl.play().catch((err) => {
          if (tok !== narrToken) return;
          if (err && err.name === "NotAllowedError") {
            narrQ.unshift(it);          // hold the sentence, do not skip it
            for (const t of it.timers || []) clearTimeout(t);
            it.timers = null;           // the chips will be lit again on the retry
            narrBusy = false;
            narrCurrent = null;
            narrBlocked = true;
            voiceBtn.classList.add("wants");
            setVoiceBtn();
            flash("click the page to start the voice");
            return;
          }
          next();
        });
      });
      return;
    }
    if (!synth) { next(); return; }
    const u = new SpeechSynthesisUtterance(it.say);
    if (narrVoice) { u.voice = narrVoice; u.lang = narrVoice.lang; }
    u.rate = narrRate;
    u.onboundary = (e) => {              // speechSynthesis does report progress
      const frac = e.charIndex / Math.max(1, it.say.length);
      for (const mk of it.marks || []) {
        if (!mk.done && frac >= mk.frac) { mk.done = true; mk.fire(); }
      }
    };
    u.onstart = () => {                  // whatever sits at the very start of it
      for (const mk of it.marks || []) {
        if (mk.frac <= 0.02 && !mk.done) { mk.done = true; mk.fire(); }
      }
    };
    u.onend = next;
    u.onerror = next;
    synth.speak(u);
  }

  // ONE definition of where a spoken unit ends, used by both the splitter and the
  // stable-prefix cut. They have to agree exactly: if the prefix could end anywhere
  // the splitter does not also split, a unit would be short in one pass and whole in
  // the next, the chunk count would not grow, and the text in between would be lost.
  const UNIT_END = /[.!?]+[)"'”]*(?=\s|$)|\n\n/g;

  function narrUnits(text) {
    const out = [];
    let last = 0, m;
    UNIT_END.lastIndex = 0;
    while ((m = UNIT_END.exec(text))) {
      out.push(text.slice(last, m.index + m[0].length));
      last = UNIT_END.lastIndex;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  // The start of the unit that contains `at`.
  function unitStart(text, at) {
    let start = 0, m;
    UNIT_END.lastIndex = 0;
    while ((m = UNIT_END.exec(text)) && UNIT_END.lastIndex <= at) start = UNIT_END.lastIndex;
    return start;
  }

  // Where it is safe to stop reading a half-written answer: the end of the last
  // complete unit, never inside an unfinished fence, edit block or directive.
  function narrStable(text) {
    let cut = text.length;
    for (const open of ["```", "[[edit:"]) {
      const i = text.lastIndexOf(open);
      if (i !== -1) {
        const close = open === "```" ? text.indexOf("```", i + 3) : text.indexOf("[[/edit]]", i);
        if (close === -1) cut = Math.min(cut, i);
      }
    }
    const bracket = text.lastIndexOf("[[");
    if (bracket !== -1 && text.indexOf("]]", bracket) === -1) cut = Math.min(cut, bracket);
    const head = text.slice(0, cut);
    let end = 0, m;
    UNIT_END.lastIndex = 0;
    while ((m = UNIT_END.exec(head))) end = UNIT_END.lastIndex;
    let stable = head.slice(0, end);
    // Never end a pass on a directive. Its label is spoken with the words that follow,
    // so a cut between them makes the label a chunk of its own in this pass and part of
    // the next chunk in the following one -- the count does not grow, and the sentence
    // after the directive is never spoken. A directive alone on its own line, which is
    // how a step usually points at the code it is about to discuss, hits this every time.
    const trimmed = stable.replace(/\s+$/, "");
    if (trimmed.endsWith("]]")) {
      const open = trimmed.lastIndexOf("[[");
      if (open !== -1) stable = stable.slice(0, open);
    }
    // A sentence followed by [[say: ...]] must never be spoken in its written form.
    // The directive sits after the full stop, so it falls outside the prefix that ends
    // there: take it in when it is complete, and wait when it might still be coming.
    const rawRest = text.slice(stable.length);
    const rest = rawRest.replace(/^\s+/, "");
    const gap = rawRest.length - rest.length;
    const dropLast = () => stable.slice(0, unitStart(stable, stable.length - 1));
    if (rest.startsWith("[[say:")) {
      const close = rest.indexOf("]]");
      return close === -1 ? dropLast() : text.slice(0, stable.length + gap + close + 2);
    }
    // Nothing yet, or only the few characters that could still become "[[say:".
    if (rest === "" || /^(\[(\[(s(a(y(:)?)?)?)?)?)$/.test(rest)) return dropLast();
    return stable;
  }

  const NARR_DROP = /\[\[[^\]]*\]\]/g;

  // Written for the eye, said for the ear. "(bv, n+v, d)" is fine on screen and
  // unbearable spoken, so the shapes and operators that show up constantly in this
  // kind of prose are turned into the words a person would actually say. This is the
  // safety net: the model is also told to phrase things for listening, but it will
  // always reach for symbols sometimes, and the reader should not have to suffer them.
  const SHAPE = /\(([^()\n]{1,60})\)/g;          // a parenthesised tuple, maybe a shape
  const SHAPEY = /^[\w\s,+*/\-]+$/;               // ...only if it is all symbols and names

  // A character espeak-ng has no name for is read as its CODEPOINT: the transpose
  // mark in QK\u1d40 comes out as "letter one D four zero", and a sentence with a few of
  // those turns into a fast stream of letters and digits -- which is what "the voice
  // suddenly sped up and became babble" actually was. Nothing gets past this: the
  // compatibility forms are normalized to their plain letters, the symbols worth saying
  // are said, and anything left that is not plainly sayable is dropped.
  const SYMBOLS = [
    [/[\u2018\u2019]/g, "'"], [/[\u201c\u201d]/g, '"'],
    [/[\u2014\u2013\u2012]/g, ", "], [/\u00b7/g, " "], [/\u2026/g, "..."],
    [/\u2207/g, " grad "], [/\u2202/g, " partial "], [/\u222b/g, " integral "],
    [/\u2211/g, " sum "], [/\u220f/g, " product "], [/\u221e/g, " infinity "],
    [/\u2208/g, " in "], [/\u2209/g, " not in "], [/\u2200/g, " for all "],
    [/\u2203/g, " there exists "], [/[\u2225\u2016]/g, " norm "], [/\u2299/g, " elementwise times "],
    [/\u2297/g, " tensor product "], [/\u21d2/g, " implies "], [/\u2261/g, " identical to "],
    [/\u2192/g, " to "], [/\u2190/g, " from "], [/\u2194/g, " to and from "],
    [/\u221d/g, " proportional to "], [/\u226a/g, " much less than "],
    [/\u226b/g, " much greater than "], [/[\u27e8\u27e9]/g, " "],
  ];
  // Latin, Greek and the punctuation espeak can actually pronounce. Everything else goes.
  const UNSAYABLE = /[^\u0000-\u024f\u0370-\u03ff\u1e00-\u1eff\u221a\u2248\u2260\u2264\u2265\u00d7\u00f7\u00b1]/g;

  const SAY = [
    [/\s*(?:->|→|=>)\s*/g, " to "],
    [/\s*==\s*/g, " equals "],
    [/\s*!=\s*/g, " is not "],
    [/\s*>=\s*/g, " at least "],
    [/\s*<=\s*/g, " at most "],
    [/(\w)\s*=\s*(\w)/g, "$1 equals $2"],
    [/(\w)\s*\+\s*(\w)/g, "$1 plus $2"],
    [/(\w)\s*\*\s*(\w)/g, "$1 times $2"],
    [/(\d)\s*[x\u00d7]\s*(\d)/g, "$1 by $2"],
    [/(\w)\s*\|\s*(\w)/g, "$1 or $2"],
  ];

  function speechify(s) {
    s = s.normalize("NFKC");                  // superscripts and the like become letters
    for (const [re, to] of SYMBOLS) s = s.replace(re, to);
    s = s.replace(UNSAYABLE, " ");
    s = s.replace(SHAPE, (m, inner) => {
      if (!inner.includes(",") || !SHAPEY.test(inner)) return m;
      // A shape is read the way it is said out loud: b by n by d, not b comma n comma d.
      return " " + inner.split(",").map((t) => t.trim()).filter(Boolean).join(" by ") + " ";
    });
    for (const [re, to] of SAY) s = s.replace(re, to);
    return s;
  }

  function narrClean(s) {
    return speechify(s.replace(NARR_DROP, " ")
            .replace(/`([^`\n]+)`/g, "$1")
            .replace(/\*\*([^*\n]+)\*\*/g, "$1")
            .replace(/\*([^*\n]+)\*/g, "$1"))
            .replace(/\s+([,.;:!?])/g, "$1")
            .replace(/\s+/g, " ")
            .trim();
  }

  // Split the answer into speakable chunks: one sentence each, with the chips that
  // fall inside it and how far through the words each one sits.
  //
  // A chip is not punctuation -- it is the words the sentence uses at that point, so
  // the label is spoken inline and the sentence stays one utterance. Speaking the
  // label separately is what made the voice sound strange: Piper gives every request
  // sentence-final intonation and a pause, so "the primary path, with" landed like a
  // finished sentence and the label after it started a new one.
  function narrChunks(text, scope) {
    text = text.replace(CONT_G, "").replace(TITLE_RE, "");
    text = text.replace(EDIT_RE, (m, p) => " Proposing an edit to " + shortPath(p) + ". ");
    text = text.replace(FENCE_RE, " (code block in the chat.) ");

    // Substitute each directive with the words it shows, remembering where the chip
    // lands in the resulting prose.
    const spans = codeSpans(text);
    let prose = "", last = 0, idx = 0, m;
    const marks = [];
    OPEN_RE.lastIndex = 0;
    while ((m = OPEN_RE.exec(text))) {
      if (!isPath(m[1])) continue;
      if (spans.some(([a, b]) => m.index >= a && m.index + m[0].length <= b)) continue;
      prose += text.slice(last, m.index);
      marks.push({ at: prose.length, idx: idx++ });
      prose += (m[3] || "").replace(/`/g, "").trim();
      last = m.index + m[0].length;
    }
    prose += text.slice(last);

    // [[say: ...]] replaces the sentence it follows, in the ear only. Chips inside
    // that sentence keep their place by moving to the start of the spoken version --
    // the words no longer line up, but the code they point at still does.
    SAY_RE.lastIndex = 0;
    let sm;
    while ((sm = SAY_RE.exec(prose))) {
      // The directive usually comes after the sentence's full stop, which puts it in a
      // unit of its own; the sentence it replaces is then the one before.
      let from = unitStart(prose, sm.index);
      if (!prose.slice(from, sm.index).trim() && from > 0) from = unitStart(prose, from - 1);
      const said = " " + sm[1].trim() + " ";
      const delta = said.length - (sm.index + sm[0].length - from);
      for (const k of marks) {
        if (k.at >= from && k.at < sm.index + sm[0].length) k.at = from;
        else if (k.at >= sm.index + sm[0].length) k.at += delta;
      }
      prose = prose.slice(0, from) + said + prose.slice(sm.index + sm[0].length);
      SAY_RE.lastIndex = from + said.length;
    }

    const fireFor = (i) => () => {
      const ds = allDirectives(scope.raw || "");
      const d = ds[i];
      if (d) openFile(d.path, d.spec);
      if (scope.el) markActiveChip(scope.el, i);
    };

    // One chunk per unit, and never more than one: this list has to be a stable PREFIX
    // as the answer streams in, because the queue is advanced by counting what has
    // already been sent. Merge two units and a sentence that stood alone in one pass
    // vanishes into its neighbour in the next, taking the text between them with it.
    const out = [];
    let pos = 0;
    for (const unit of narrUnits(prose)) {
      const start = pos, end = pos + unit.length;
      pos = end;
      const mine = marks.filter((k) => k.at >= start && k.at < end);
      const say = narrClean(unit);
      if (!say) {
        for (const k of mine) out.push({ say: "", marks: [{ frac: 0, fire: fireFor(k.idx) }] });
        continue;
      }
      out.push({
        say: say,
        // Where in the audio each chip belongs, as a fraction of the sentence. Piper
        // reports no word timings, so this is proportional to characters -- close
        // enough that the pane moves on the phrase that names it.
        marks: mine.map((k) => ({ frac: (k.at - start) / unit.length, fire: fireFor(k.idx) })),
      });
    }
    return out;
  }

  // Every finished answer keeps its raw text, so it can be read aloud again later.
  function addReplay(acts, msg, out) {
    if (!synth || acts.querySelector(".replay")) return;
    const b = document.createElement("button");
    b.className = "act replay";
    b.title = "Read this step aloud";
    b.textContent = "\u25b6 listen";
    b.onclick = () => narrReplay(out, msg.dataset.raw || "");
    acts.appendChild(b);
  }

  // A live answer: feed whatever has become stable since the last call.
  function narrFeed(state, acc, el, final) {
    if (!narrOn) return;
    state.raw = acc;
    state.el = el;
    const src = final ? acc : narrStable(acc);
    if (!src) return;
    const chunks = narrChunks(src, state);
    if (chunks.length <= state.sent) return;
    narrEnqueue(chunks.slice(state.sent));
    state.sent = chunks.length;
  }

  function narrReplay(el, raw) {
    narrStop();
    if (!synth) return;
    const state = { raw: raw, el: el, sent: 0 };
    narrOn = true; setVoiceBtn();
    narrEnqueue(narrChunks(raw, state));
  }

  // Model picker. A change lands on the next turn: the session is resumed, so the
  // conversation so far is kept and only the model answering it changes.
  const modelEl = $("model");
  function setupModel(cur, list) {
    for (const m of list || []) {
      const o = document.createElement("option");
      o.value = m; o.textContent = m;
      modelEl.appendChild(o);
    }
    modelEl.value = cur || "";
    modelEl.onchange = async () => {
      const want = modelEl.value;
      try {
        const r = await fetch("/api/model", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: want }),
        });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "rejected");
        flash(want + " from the next question");
      } catch (e) {
        flash("could not switch model");
      }
    };
  }

  // A test hook: the page is one closure, so an automated check of the narrator has
  // no other way to see whether a sentence was held or dropped.
  window.__cw = {
    narr: () => ({
      on: narrOn, blocked: narrBlocked, busy: narrBusy, queued: narrQ.length,
      rate: narrRate, tts: ttsOK, playing: !audioEl.paused,
      at: audioEl.currentTime, say: narrCurrent ? narrCurrent.say : null,
    }),
  };

  const voiceBtn = $("voice"), rateEl = $("rate"), rateWrap = $("rateWrap");
  function setVoiceBtn() {
    voiceBtn.textContent = narrBlocked ? "Voice: click to start"
                         : !narrOn ? "Voice: off"
                         : narrHeld ? "Voice: retry"
                         : ttsOK ? "Voice: on"
                         : "Voice: browser";      // no Piper here; the browser is speaking
    voiceBtn.title = narrHeld
      ? "The voice server stopped answering. Click to pick up where it stopped."
      : narrOn && !ttsOK
      ? "This server has no Piper, so this is the browser's own speech engine"
      : "Read the answers aloud and move the editor in time with the voice";
    voiceBtn.style.color = narrOn ? "var(--hl-rail)" : "";
    rateWrap.hidden = !narrOn;
    rateEl.value = String(narrRate);
    $("rateVal").textContent = (narrRate % 1 ? narrRate.toFixed(2).replace(/0$/, "")
                                            : narrRate.toFixed(0)) + "\u00d7";
  }

  voiceBtn.onclick = () => {
    narrUnblock();
    if (narrHeld) {                   // a held queue: the click is "try again", not "off"
      narrHeld = false;
      ttsFails = 0;
      setVoiceBtn();
      narrPump();
      return;
    }
    narrOn = !narrOn;
    localStorage.setItem("cw.voice", narrOn ? "1" : "0");
    if (!narrOn) narrStop();
    setVoiceBtn();
  };
  // The rate is a live regulator: a speaking utterance cannot be re-rated in place,
  // so the current sentence is re-spoken at the new speed and the queue rides along.
  // While the slider is being dragged, only the readout moves — restarting the voice
  // on every pixel would stutter.
  let rateTimer = null;
  function setRate(v, immediate) {
    const was = narrRate;
    narrRate = snapRate(v);
    if (narrRate === was) { setVoiceBtn(); return; }   // the drag has not left this step
    localStorage.setItem("cw.rate", String(narrRate));
    setVoiceBtn();
    clearTimeout(rateTimer);
    const apply = () => {
      if (!narrBusy) return;
      const rest = narrQ.map((it) => ({ say: it.say, marks: it.marks }));
      const cur = narrCurrent;
      narrStop();
      narrOn = true;
      if (cur) rest.unshift({ say: cur.say, marks: [] });    // repeat it, do not re-open
      narrEnqueue(rest);
    };
    rateTimer = setTimeout(apply, immediate ? 0 : 260);
  }
  rateEl.oninput = () => setRate(Number(rateEl.value), false);
  rateEl.onchange = () => setRate(Number(rateEl.value), true);
  document.addEventListener("keydown", (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === ",") { e.preventDefault(); setRate(narrRate - RATE_STEP, true); }
    if (e.key === ".") { e.preventDefault(); setRate(narrRate + RATE_STEP, true); }
  });
  setVoiceBtn();

  async function ask(text) {
    if (busy) return;
    const q = (text || "").trim();
    if (!q) return;

    narrStop();
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
    msg.classList.add("cell");
    const acts = document.createElement("div");
    acts.className = "acts";
    const out = msg.querySelector(".body");
    msg.insertBefore(acts, out);
    out.innerHTML = '<p class="thinking">Thinking</p>';
    setBusy(true);
    cellIdx = cells().length - 1;
    holdPin(msg);
    syncCellNav();
    updateJump();

    let acc = "", autoIdx = -1;
    const narr = { raw: "", el: out, sent: 0 };

    // How much of the answer the reader has asked for. Everything past the current
    // [[continue:]] is written but not shown, not spoken, and not allowed to move the
    // pane: the model finishes its thought, the reader walks through it.
    let shown = 1;
    const show = () => {
      const text = revealed(acc, shown);
      const label = breakLabel(acc, shown);
      renderReply(out, text);
      msg.querySelectorAll(".contbar").forEach((b) => b.remove());
      if (label !== null) {
        addContinue(msg, label, () => {
          shown += 1;
          msg.dataset.shown = String(shown);
          show();
          applyPin();
        });
      }
      // A part followed by a break is finished even while the rest still streams, so
      // the narrator may speak all of it; only the tail of the last part has to wait.
      narrFeed(narr, text, out, label !== null || !busy);
      if (!narrOn) {
        const ds = allDirectives(text);
        if (follow && ds.length - 1 > autoIdx) {
          autoIdx = ds.length - 1;
          openFile(ds[autoIdx].path, ds[autoIdx].spec);
        }
        markActiveChip(out, autoIdx);
      }
    };

    try {
      const res = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: q, context: context, voice: narrOn }),
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
          if (ev.k === "delta") {
            acc += ev.v;
            const tm = TITLE_RE.exec(acc);
            if (tm) setCellTitle(msg, (tm[1] || "").trim());
            show();
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
          applyPin();  // the answer is growing; keep its top where the eye already is
          updateJump();
        }
      }
      if (acc) {
        const tm = TITLE_RE.exec(acc);
        if (tm) setCellTitle(msg, (tm[1] || "").trim());
        msg.dataset.raw = acc;
        msg.dataset.shown = "1";
        addReplay(acts, msg, out);
        show();
        if (!narrOn && autoIdx < 0) {
          const ds = allDirectives(revealed(acc, shown));
          if (ds.length) {
            autoIdx = 0;
            openFile(ds[0].path, ds[0].spec);
            markActiveChip(out, autoIdx);
          }
        }
      }
    } catch (err) {
      const e = document.createElement("div");
      e.className = "errline";
      e.textContent = "Lost the connection to codewalk. Is the server still running?";
      out.appendChild(e);
    } finally {
      setBusy(false);
      applyPin();
      releasePin();
      syncCellNav();
      updateJump();
      show();               // the last part is complete now: release it to the narrator
      refreshProposals();   // the agent may have drafted a file this turn
    }
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
    // Escape shuts the voice up without turning narration off for the next step.
    if (e.key === "Escape" && narrBusy) { e.preventDefault(); narrStop(); }
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
    setupCells();
    ttsOK = !!d.tts;
    ttsServer = !!d.tts;
    if (!ttsOK && !synth) { narrOn = false; voiceBtn.hidden = true; }
    setVoiceBtn();
    setupModel(d.model, d.models);
    setupProposals();
    if (d.sync) setupSync();
    renderTree();
    const b = bubble("Claude").querySelector(".body");
    b.innerHTML =
      (d.context
        ? "<p>Picking up from our terminal session \u2014 I already have the context of what we built here.</p>"
        : "<p>Ask me anything about <code>" + d.name + "</code>. I will read whatever files I need and " +
          "open them in the center pane, on the lines I am talking about.</p>") +
      "<p class='notice'>Click a chip in my answer to jump there. Drag the line-number gutter or select code and press Ctrl-L to attach it to a question. Ctrl-P filters the tree.</p>";
    if (d.first_question) ask(d.first_question);
  });
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Local three-pane workspace for talking about code.")
    ap.add_argument("root", nargs="?", default=".", help="project root (default: .)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="opus", help="alias passed to `claude --model`")
    ap.add_argument("--demo", nargs="?", const="", metavar="SCRIPT",
                    help="serve canned answers from a script file instead of calling the "
                         "model, for working on the page itself (default: "
                         "demo-walkthrough.md next to codewalk.py)")
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
                    help="ask this the moment the page opens, so the conversation starts by itself")
    ap.add_argument("--sync-file", default=None, metavar="PATH",
                    help="show Sync buttons that publish the conversation to PATH, for a terminal "
                         "session to pick up. Works with or without --handoff.")
    ap.add_argument("--idle-exit", type=int, default=15, metavar="SECONDS",
                    help="with --handoff, return to the terminal this long after the tab is closed")
    ap.add_argument("--shadow", default=None, metavar="DIR",
                    help="where proposed files are drafted before you agree to them "
                         "(default: ~/.cache/codewalk/shadow/<repo>-<hash>)")
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
    Handler.shadow = Shadow(args.shadow or default_shadow(Handler.repo.root), Handler.repo)
    if args.demo is not None:
        script = args.demo or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "demo-walkthrough.md")
        try:
            Handler.claude = Script(script)
        except OSError as e:
            sys.exit(f"codewalk: could not read --demo script: {e}")
        Handler.models = ["demo"]
    else:
        Handler.claude = Claude(
            args.model, Handler.repo.root, parent, brief, shadow=Handler.shadow.root,
        )
        Handler.models = MODELS + ([args.model] if args.model not in MODELS else [])
    Handler.voice = Voice(*find_piper())
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
    print(TRANSCRIPT_NOTE)
    print(render_transcript(Handler.transcript))
    if not returned:
        sys.exit(3)


if __name__ == "__main__":
    main()
