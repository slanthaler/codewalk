#!/usr/bin/env python3
"""
codewalk — a dashboard that agents submit code reviews to.

    codewalk.py serve                                  # leave this open
    codewalk.py submit --title "..." --notes notes.md  # what an agent runs
    codewalk.py wait <id>                              # re-attach to one

One dashboard, many agents. An agent changes files, writes a notes file saying what to look
at and in what order, and submits it with a title. The dashboard lists what is pending; you
open one and get three panes:

    left     that repository's file tree, changed files badged
    center   the open file, syntax-highlighted, with the diff spliced in place, the agent's
             notes and your comments pinned to the lines they are about
    right    the review: every change in reading order, your verdicts, and Send

`submit` BLOCKS until you press Send, then prints your verdicts and comments to stdout — so
the agent runs one command and reads the result as that command's output. Each review
carries its own repository root and base commit, so several agents in several checkouts can
be waiting at once. The dashboard never writes to any repository: a verdict is a statement
back to the agent, never an edit.

The notes file is markdown. Everything before the first heading is the brief; each
`## <path>[:<line>[-<line>]] <title>` heading is one stop, and the file's order is the
reading order:

    New attention block, wired in behind a config flag. Read the stops in order.

    ## model/config.py:40-52  Settings for the new block
    Added `attn: AttnCfg`. Defaults keep the old behaviour.

    ## model/build.py:88-94  Where it gets built
    One branch in `build_model`. Nothing else in the wiring changed.

Standard library only, plus Pygments for highlighting (optional — falls back to plain text).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
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
IDLE_EXIT = 15          # seconds after the tab stops pinging before we give up on it
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
        self.is_git = os.path.exists(os.path.join(self.root, ".git"))

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
            "lines": tokenize(text, full),
        }


# ----------------------------------------------------------------------------------
# what changed
# ----------------------------------------------------------------------------------

class Changes:
    """What the agent has done, read as a git diff against a base commit.

    This answers the question a reviewer actually asks — what is different from where we
    started — and it answers it from git rather than from a second copy of the tree. It
    reads the WORKING tree, and counts untracked files too, so a change shows up whether
    the agent committed it, staged it, or just wrote the file, and the reviewer never has
    to know which of those happened.
    """

    def __init__(self, repo: "Repo", ref: str | None):
        self.repo = repo
        self.ref = ref
        self._lock = threading.Lock()
        self._base: str | None = None
        self._entries: list[dict] = []
        self._at = 0.0

    # -- git -------------------------------------------------------------------
    def _git(self, *args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=self.repo.root,
                capture_output=True, text=True, errors="replace", timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    def base(self) -> str | None:
        """What the diff is measured from. Resolved once — it must not move while a review
        is happening, since that would silently reframe everything on screen."""
        if self._base is not None:
            return self._base or None
        if not self.repo.is_git:
            self._base = ""
            return None
        for name in ([self.ref] if self.ref else ["HEAD"]):
            if not name:
                continue
            # A branch name means the point we forked from it, not its tip.
            sha = (self._git("merge-base", name, "HEAD") or "").strip()
            if not sha:
                sha = (self._git("rev-parse", "--verify", name + "^{commit}") or "").strip()
            if sha:
                self._base = sha
                return sha
        self._base = ""
        return None

    def label(self) -> str:
        """What to print under the list: the name asked for, plus where it landed."""
        sha = self.base()
        if not sha:
            return ""
        name = self.ref or "HEAD"
        if sha.startswith(name):     # the ref *is* the commit; say it once
            return sha[:8]
        return f"{name} @ {sha[:8]}"

    # -- the list --------------------------------------------------------------
    def entries(self) -> list[dict]:
        with self._lock:
            if self._entries and time.time() - self._at < 2:
                return self._entries
            self._entries = self._scan()
            self._at = time.time()
            return self._entries

    def _scan(self) -> list[dict]:
        sha = self.base()
        if not sha:
            return []
        status: dict[str, str] = {}
        raw = self._git("diff", "--name-status", "--no-renames", "-z", sha)
        if raw is None:
            return []
        parts = [p for p in raw.split("\0") if p]
        for i in range(0, len(parts) - 1, 2):
            status[parts[i + 1]] = parts[i][:1]

        counts: dict[str, tuple[int, int]] = {}
        raw = self._git("diff", "--numstat", "--no-renames", "-z", sha) or ""
        for rec in raw.split("\0"):
            if not rec:
                continue
            bits = rec.split("\t")
            if len(bits) != 3:
                continue
            add, rem, path = bits
            # "-" means git called it binary; there is nothing to show line by line.
            counts[path] = (int(add) if add.isdigit() else -1,
                            int(rem) if rem.isdigit() else -1)

        # A file the agent has written but never added is invisible to `git diff`, and a
        # brand new file is the single most likely thing for it to have written. Without
        # this the one change you most need to see is the one that does not show up.
        raw = self._git("ls-files", "--others", "--exclude-standard", "-z") or ""
        for path in raw.split("\0"):
            if not path or path in status:
                continue
            text = self.new_text(path)
            status[path] = "A"
            counts[path] = (len(text.splitlines()), 0)

        out = []
        for path, letter in sorted(status.items()):
            if os.path.splitext(path)[1].lower() in SKIP_EXT:
                continue
            if any(part in SKIP_DIRS for part in path.split("/")):
                continue
            add, rem = counts.get(path, (0, 0))
            out.append({
                "path": path,
                "state": {"A": "new", "D": "deleted"}.get(letter, "changed"),
                "added": add, "removed": rem,
                "binary": add < 0,
            })
        return out

    def has(self, rel: str) -> bool:
        return any(e["path"] == rel for e in self.entries())

    # -- one file --------------------------------------------------------------
    def old_text(self, rel: str) -> str:
        sha = self.base()
        if not sha:
            return ""
        return self._git("show", f"{sha}:{rel}") or ""

    def new_text(self, rel: str) -> str:
        full = self.repo.resolve(rel)
        if full is None or not os.path.isfile(full):
            return ""
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def read(self, rel: str) -> dict:
        rel = rel.lstrip("/")
        if self.repo.resolve(rel) is None:
            return {"error": "That path is outside the project."}
        entry = next((e for e in self.entries() if e["path"] == rel), None)
        if entry is None:
            return {"error": f"{rel} has not changed since {self.label() or 'the base'}."}
        if entry["binary"]:
            return {"error": f"{rel} is a binary file."}

        old, new = self.old_text(rel), self.new_text(rel)
        deleted = entry["state"] == "deleted"
        # A deleted file still gets shown — as the text that would go. Otherwise the one
        # change you cannot comment on is the most destructive one.
        text = old if deleted else new
        if len(text.encode("utf-8", "replace")) > MAX_BYTES:
            return {"error": f"{rel} is too large to display."}

        # A deletion is the whole file going, so splice it against nothing and every line
        # reads as removed — the same view, not a special case.
        view = splice(rel, old, "" if deleted else new)
        return {
            "path": rel,
            "name": os.path.basename(rel),
            "digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "state": entry["state"],
            "deleted": deleted,
            "base": self.label(),
            **view,
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


def splice(path: str, old_text: str, new_text: str) -> dict:
    """Two versions of a file as ONE readable column: context, then what went, then what
    arrived, in place.

    The gutter numbers the file as it is AFTER the change, and a removed line shows a dash
    instead of a number — one coherent run of numbers top to bottom, which is what you want
    when the question is "what does this file look like now". It also means a line number
    in a comment is a line number in the file, with no translation anywhere.
    """
    # Empty text is NOT one empty line. `"".split()` gives [""], a phantom line that difflib
    # happily matches against a real blank line in the other version — so a brand new file
    # came out as two hunks with a fabricated line of context between them, and had to be
    # approved twice. A side that does not exist has no lines at all.
    old = old_text.split("\n") if old_text else []
    new = new_text.split("\n") if new_text else []
    old_tok, new_tok = tokenize(old_text, path), tokenize(new_text, path)
    # tokenize() drops a trailing empty line; keep the two views the same length.
    old_tok += [[] for _ in range(len(old) - len(old_tok))]
    new_tok += [[] for _ in range(len(new) - len(new_tok))]

    ops = difflib.SequenceMatcher(None, old, new).get_opcodes()
    # Adjacent non-equal opcodes are one change to a reader, so read them as one hunk.
    merged: list[list] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal" and merged and merged[-1][0] != "equal":
            merged[-1][2], merged[-1][4] = i2, j2
        else:
            merged.append([tag, i1, i2, j1, j2])

    lines: list = []
    nums: list[str] = []
    marks: list[str] = []
    hunks: list[dict] = []
    n = 0
    for tag, i1, i2, j1, j2 in merged:
        if tag == "equal":
            for j in range(j1, j2):
                n += 1
                lines.append(new_tok[j]); nums.append(str(n)); marks.append("")
            continue
        at = len(lines)
        for i in range(i1, i2):
            lines.append(old_tok[i]); nums.append("−"); marks.append("del")
        for j in range(j1, j2):
            n += 1
            lines.append(new_tok[j]); nums.append(str(n)); marks.append("add")
        hunks.append({
            "at": at, "rows": len(lines) - at,
            "removed": i2 - i1, "added": j2 - j1,
            "line": n - (j2 - j1) + 1 if j2 > j1 else n + 1,
        })
    return {"lines": lines, "nums": nums, "marks": marks, "hunks": hunks}


def css_class(ttype) -> str:
    t = ttype
    while t is not None:
        if t in STANDARD_TYPES:
            return STANDARD_TYPES[t]
        t = t.parent
    return ""


def short(path: str) -> str:
    bits = path.split("/")
    return path if len(bits) <= 2 else ".../" + "/".join(bits[-2:])


# ----------------------------------------------------------------------------------
# the agent's notes, read as an ordered itinerary
# ----------------------------------------------------------------------------------

# The anchor is the first whitespace-delimited token of the heading, and a path may not
# contain a colon — which is what makes "where" and "which lines" unambiguous to split.
ANCHOR_RE = re.compile(r"^(?P<path>[^\s:]+)(?::(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?)?$")


class Notes:
    """The agent's notes: a brief, then an ordered list of stops.

    The order is the agent's, and it is the one thing this program will not second-guess:
    the agent wrote the code and is the only party that knows which part has to be read
    first for the rest to make sense. Files it gave no stop to are not hidden — the page
    lists them after the last stop — but they are not promoted into the walk either.
    """

    def __init__(self, text: str, repo: Repo):
        self.brief = ""
        self.stops: list[dict] = []
        self.problems: list[str] = []
        self._parse(text or "", repo)

    def _parse(self, text: str, repo: Repo) -> None:
        head: list[str] = []
        sink = head
        for raw in text.split("\n"):
            m = re.match(r"^##\s+(\S+)\s*(.*)$", raw)
            if not m:
                sink.append(raw)
                continue
            cur = self._stop(m.group(1), m.group(2).strip(), repo)
            if cur is None:
                sink = []        # the heading was not an anchor; its prose goes nowhere
                continue
            self.stops.append(cur)
            sink = cur["note"]
        self.brief = "\n".join(head).strip()
        for s in self.stops:
            s["note"] = "\n".join(s["note"]).strip()

    def _stop(self, anchor: str, title: str, repo: Repo) -> dict | None:
        m = ANCHOR_RE.match(anchor)
        if not m:
            self.problems.append(f"Not a path:line anchor, so this note has no stop: `{anchor}`")
            return None
        rel = m.group("path").lstrip("./")
        a = int(m.group("a")) if m.group("a") else 0
        b = int(m.group("b")) if m.group("b") else a
        full = repo.resolve(rel)
        missing = full is None or not os.path.isfile(full)
        if missing:
            self.problems.append(f"No such file, so stop {len(self.stops) + 1} has nothing to "
                                 f"open: `{rel}`")
        return {
            "n": len(self.stops) + 1, "path": rel, "a": a, "b": max(a, b),
            "title": title, "note": [], "missing": missing,
        }

    def payload(self) -> dict:
        return {"brief": self.brief, "stops": self.stops, "problems": self.problems}


# ----------------------------------------------------------------------------------
# what you send back
# ----------------------------------------------------------------------------------

class Comments:
    """Your side of the round, held here and printed when you press Send.

    Two kinds of thing go back: a VERDICT on something the agent proposed — a stop, or a
    hunk it gave no stop to — and a COMMENT, which is prose about particular lines. They
    are kept apart because they answer different questions. "Is this right?" is not the
    same question as "here is what I noticed", and an agent that cannot tell them apart
    will rewrite code you only asked about.

    The browser is the only writer and sends the whole state each time, so there is no
    incremental state to get out of step. It is kept server-side at all because a tab that
    gets closed by accident should not take the review you typed with it.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.items: list[dict] = []        # comments
        self.verdicts: list[dict] = []     # approve / reject
        self.general = ""
        self.sent = threading.Event()

    def put(self, body: dict) -> int:
        with self.lock:
            self.items = [i for i in (body.get("items") or []) if isinstance(i, dict)]
            self.verdicts = [v for v in (body.get("verdicts") or []) if isinstance(v, dict)]
            self.general = str(body.get("general") or "")
            return self._count()

    def _count(self) -> int:
        return len(self.items) + len(self.verdicts) + (1 if self.general.strip() else 0)

    def count(self) -> int:
        with self.lock:
            return self._count()

    def draft(self) -> dict:
        """Everything said so far, so a review you stepped away from is still there when
        you come back to it."""
        with self.lock:
            return {"items": list(self.items), "verdicts": list(self.verdicts),
                    "general": self.general}

    def render(self, root: str, base: str, submitted: bool, changed: list,
               stops: list | None = None) -> str:
        with self.lock:
            items, verdicts, general = list(self.items), list(self.verdicts), self.general
            n = self._count()

        ok = [v for v in verdicts if v.get("state") == "approve"]
        no = [v for v in verdicts if v.get("state") == "reject"]
        tally = ", ".join(
            p for p in (
                f"{len(ok)} approved" if ok else "",
                f"{len(no)} rejected" if no else "",
                f"{len(items)} comment(s)" if items else "",
                "a general note" if general.strip() else "",
            ) if p
        )
        out = [f"# codewalk review — {tally}" if tally else "# codewalk review — nothing said"]
        out.append(f"# {root}" + (f"   ·   base {base}" if base else ""))
        if not submitted:
            out += ["#",
                    "# The tab was closed without pressing Send. This is what had been "
                    "written by then;",
                    "# ask before treating it as the finished review."]
        out.append("")
        if not n:
            out.append("Nothing was flagged." if submitted else
                       "Nothing had been written down.")
            return "\n".join(out)

        # One section per thing reviewed, so a verdict and the comments explaining it read
        # together instead of twice. Stop order first, because that is the order the agent
        # asked for and so the order the reading actually happened in.
        sections: dict = {}
        order: list = []
        for it in verdicts + items:
            key = str(it.get("key") or (it.get("path"), it.get("a"), it.get("b")))
            if key not in sections:
                sections[key] = {"head": it, "state": "", "notes": [], "rows": []}
                order.append(key)
            sec = sections[key]
            if it.get("state"):
                sec["state"] = str(it["state"])
            if it.get("note"):
                sec["notes"].append(str(it["note"]))
            if it.get("rows") and not sec["rows"]:
                sec["rows"] = it["rows"]

        order.sort(key=lambda k: (sections[k]["head"].get("stop") or 10**6,
                                  str(sections[k]["head"].get("path")),
                                  sections[k]["head"].get("a") or 0))
        for key in order:
            out.append(self._one(sections[key]))

        if general.strip():
            out.append("## general")
            out += ["> " + l for l in general.strip().split("\n")]
            out.append("")

        # What was never judged at all. Silence is not approval, and it has to be reported
        # per STOP as well as per path: a stop skipped inside a file that was otherwise
        # reviewed used to disappear without trace, because the file itself looked covered.
        keys = {str(i.get("key")) for i in verdicts + items}
        touched = {str(i.get("path")) for i in verdicts + items}
        judged = {str(v.get("key")) for v in verdicts}
        missed = []
        for st in (stops or []):
            if "stop:%d" % st["n"] in judged:
                continue
            where = st["path"] + (f":{st['a']}" if st["a"] and st["a"] == st["b"]
                                  else f":{st['a']}-{st['b']}" if st["a"] else "")
            said = " (commented on, but no verdict)" if "stop:%d" % st["n"] in keys else ""
            missed.append(f"stop {st['n']} \u2014 {where}"
                          + (f'  "{st["title"]}"' if st.get("title") else "") + said)
        rest = [e["path"] for e in changed if e["path"] not in touched]
        if missed or rest:
            out.append("## not reviewed")
            if missed:
                out.append("> These stops were passed over:")
                out += ["> " + "  " + m for m in missed]
            if rest:
                out.append("> No verdict and no comment on these changed paths:")
                out += ["> " + "  " + p for p in rest]
            out.append("")
        return "\n".join(out)

    def _one(self, sec: dict) -> str:
        it = sec["head"]
        path = str(it.get("path") or "?")
        a, b = it.get("a") or 0, it.get("b") or 0
        where = path + (f":{a}" if a and a == b else f":{a}-{b}" if a else "")
        head = f"## stop {it['stop']} — {where}" if it.get("stop") else f"## {where}"
        if it.get("title"):
            head += f'  "{it["title"]}"'
        if sec["state"]:
            head += "   [" + {"approve": "APPROVED", "reject": "REJECTED"}.get(
                sec["state"], sec["state"].upper()) + "]"
        out = [head]
        rows = sec["rows"] or []
        for r in rows[:60]:
            if not isinstance(r, dict):
                continue
            mark = {"add": "+", "del": "-"}.get(str(r.get("m") or ""), " ")
            out.append(f"  {mark} {str(r.get('n') or ''):>5}  {r.get('t') or ''}")
        if len(rows) > 60:
            out.append(f"  … {len(rows) - 60} more line(s)")
        for note in sec["notes"]:
            for l in note.strip().split("\n"):
                out.append("> " + l)
        if sec["state"] == "reject" and not sec["notes"]:
            out.append("> (rejected with no reason given)")
        out.append("")
        return "\n".join(out)


# ----------------------------------------------------------------------------------
# one submitted review, and the queue of them
# ----------------------------------------------------------------------------------

class Request:
    """One review an agent has asked for, and everything needed to serve it.

    It carries its own repo root and base commit, because the agents submitting are working
    in different checkouts at the same time. Nothing here is shared between reviews except
    the port they arrive on.
    """

    PENDING = "pending"      # queued, nobody has opened it
    READING = "reading"      # the dashboard has it open
    ANSWERED = "answered"    # sent back; the agent has it or is about to
    GONE = "gone"            # the agent stopped waiting

    def __init__(self, title: str, root: str, notes_text: str, base: str | None, agent: str):
        self.id = hashlib.sha256(
            f"{time.time()}{root}{title}".encode("utf-8")).hexdigest()[:8]
        self.title = title.strip() or "untitled review"
        self.agent = agent.strip()
        self.created = time.time()
        self.state = Request.PENDING
        self.last_seen = time.time()      # the agent is still on the other end
        self.opened = 0.0
        self.repo = Repo(root)
        self.changes = Changes(self.repo, base)
        self.notes = Notes(notes_text, self.repo)
        self.comments = Comments()
        self.answered = threading.Event()

    def touch(self) -> None:
        self.last_seen = time.time()
        if self.state == Request.GONE:
            self.state = Request.PENDING

    def waiting(self) -> bool:
        """Is anyone still listening? An agent that has given up leaves a review nobody
        will ever read the answer to, and saying so is kinder than letting it look live."""
        return time.time() - self.last_seen < 90

    def card(self) -> dict:
        """What the list shows. Counted lazily, because a queue of twenty reviews should not
        diff twenty repositories to draw one screen."""
        changed = self.changes.entries()
        return {
            "id": self.id,
            "title": self.title,
            "agent": self.agent,
            "repo": self.repo.name,
            "root": self.repo.root,
            "base": self.changes.label(),
            "files": len(changed),
            "added": sum(e["added"] for e in changed if e["added"] > 0),
            "removed": sum(e["removed"] for e in changed if e["removed"] > 0),
            "stops": len(self.notes.stops),
            "state": self.state,
            "waiting": self.waiting(),
            "age": int(time.time() - self.created),
            "answered_with": self.comments.count() if self.state == Request.ANSWERED else 0,
        }

    def boot(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "agent": self.agent,
            "root": self.repo.root,
            "name": self.repo.name,
            "git": self.repo.is_git,
            "base": self.changes.label(),
            "files": self.repo.files(),
            "changes": self.changes.entries(),
            "review": self.notes.payload(),
            "state": self.state,
            "waiting": self.waiting(),
            "draft": self.comments.draft(),
        }


class Queue:
    """Every review the dashboard knows about, newest first.

    Memory only. The dashboard is the long-running process and the agents are the
    short-lived ones, so there is nothing a restart could usefully preserve: a client whose
    review vanished finds out on its next poll and says so.
    """

    KEEP_ANSWERED = 25

    def __init__(self):
        self.lock = threading.Lock()
        self.items: list[Request] = []

    def add(self, req: Request) -> None:
        with self.lock:
            self.items.insert(0, req)
            done = [r for r in self.items if r.state == Request.ANSWERED]
            for old in done[Queue.KEEP_ANSWERED:]:
                self.items.remove(old)

    def get(self, rid: str) -> Request | None:
        with self.lock:
            for r in self.items:
                if r.id == rid:
                    return r
        return None

    def cards(self) -> dict:
        with self.lock:
            items = list(self.items)
        for r in items:
            if r.state in (Request.PENDING, Request.READING) and not r.waiting():
                r.state = Request.GONE
        pending = [r.card() for r in items if r.state in (Request.PENDING, Request.READING)]
        stale = [r.card() for r in items if r.state == Request.GONE]
        done = [r.card() for r in items if r.state == Request.ANSWERED]
        return {"pending": pending, "stale": stale, "done": done[:8]}


# ----------------------------------------------------------------------------------
# the server
# ----------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "codewalk"
    queue: "Queue" = None    # type: ignore[assignment]

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

    # -- routing ---------------------------------------------------------------
    # Everything about one review lives under /api/r/<id>/, so the page never has to hold a
    # review id and a path in the same string, and two reviews of two repositories cannot
    # read each other's files.
    def _review(self, rest: str):
        rid, _, tail = rest.partition("/")
        req = Handler.queue.get(rid)
        if req is None:
            self._json({"error": "No such review. The dashboard may have been restarted."}, 404)
            return None, ""
        return req, tail.split("?", 1)[0]

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = PAGE.replace("__STAMP__", page_stamp())
            self._send(200, "text/html; charset=utf-8", body.encode("utf-8"))
        elif path == "/api/queue":
            self._json({"stamp": page_stamp(), **Handler.queue.cards()})
        elif path == "/api/wait":
            self._wait()
        elif path.startswith("/api/r/"):
            req, tail = self._review(path[len("/api/r/"):])
            if req is None:
                return
            if tail == "":
                if req.state == Request.PENDING:
                    req.state = Request.READING
                    req.opened = time.time()
                self._json(req.boot())
            elif tail == "file":
                res = req.repo.read(self._query().get("path", ""))
                self._json(res, 200 if "error" not in res else 404)
            elif tail == "change":
                res = req.changes.read(self._query().get("path", ""))
                self._json(res, 200 if "error" not in res else 404)
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/submit":
            self._submit()
        elif path.startswith("/api/r/"):
            req, tail = self._review(path[len("/api/r/"):])
            if req is None:
                return
            if tail in ("draft", "send"):
                b = self._body()
                n = req.comments.put(b)
                if tail == "send":
                    req.state = Request.ANSWERED
                    req.answered.set()
                self._json({"ok": True, "count": n})
            elif tail == "defer":
                # Put it back on the pile without answering: a review you open and then
                # decide to come back to should not look like one you are in the middle of.
                if req.state == Request.READING:
                    req.state = Request.PENDING
                self._json({"ok": True})
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")

    # -- what agents call ------------------------------------------------------
    def _submit(self):
        b = self._body()
        root = str(b.get("root") or "")
        if not os.path.isdir(root):
            self._json({"ok": False, "error": f"not a directory: {root}"}, 400)
            return
        req = Request(
            title=str(b.get("title") or ""),
            root=root,
            notes_text=str(b.get("notes") or ""),
            base=(str(b.get("base")) if b.get("base") else None),
            agent=str(b.get("agent") or ""),
        )
        Handler.queue.add(req)
        card = req.card()
        print(f"  + {req.id}  {req.title}   ({card['repo']}: {card['files']} file(s), "
              f"{card['stops']} stop(s))", flush=True)
        self._json({"ok": True, "id": req.id, **card})

    def _wait(self):
        """Long poll. The agent holds this open; it returns the moment you press Send."""
        q = self._query()
        req = Handler.queue.get(str(q.get("id") or ""))
        if req is None:
            self._json({"state": "unknown"}, 404)
            return
        try:
            secs = max(1.0, min(60.0, float(q.get("for") or 25)))
        except ValueError:
            secs = 25.0
        req.touch()
        if req.answered.wait(timeout=secs):
            self._json({
                "state": Request.ANSWERED,
                "text": req.comments.render(req.repo.root, req.changes.label(), True,
                                            req.changes.entries(), req.notes.stops),
            })
            return
        self._json({"state": req.state, "title": req.title})


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
     a hint, so syntax colours underneath stay readable. */
  --hl-bg: rgba(215,186,125,0.09); --hl-rail: #d7ba7d; --sel-bg: rgba(0,122,204,0.20);
  --add: rgba(78,201,176,0.15); --del: rgba(244,71,71,0.14);
  /* The same two colours, faint. Every change stays visible so the shape of the diff is
     still readable at a glance, but only the one you are deciding on is lit. */
  --add-off: rgba(78,201,176,0.045); --del-off: rgba(244,71,71,0.04);
  /* Git status, and nothing else. Deliberately not the highlight amber: what a file's
     state is and what is being pointed at are unrelated questions. */
  --git-new: #4fa3e3; --git-mod: #e2a03f; --git-del: #f44747;
  /* The two voices in the margin. They must never be confusable: one is the agent saying
     what it did, the other is you saying what you think of it. */
  --them: #4ec9b0; --them-bg: rgba(78,201,176,0.07);
  --me: #c586c0; --me-bg: rgba(197,134,192,0.10);
  --red: #f44747; --green: #4ec9b0;
  /* A verdict is not a note and must not read like one: yes and no, in the two colours
     every reviewer already knows, and used for nothing else in the page. */
  --ok: #4ec9b0; --no: #f05f5f;
  --gutter: 62px;
  --mono: ui-monospace, "Cascadia Code", "Droid Sans Mono", Consolas, "Courier New", monospace;
  --ui: system-ui, "Segoe UI", Ubuntu, "Droid Sans", sans-serif;
}
* { box-sizing: border-box; }
/* Every pane here is laid out with an explicit `display`, which outranks the hidden
   attribute's own rule — so without this, hiding a view does nothing at all. */
[hidden] { display: none !important; }
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
.titlebar .center { flex: 1; text-align: center; color: var(--fg-faint); }
.titlebar .base { font-family: var(--mono); font-size: 11px; color: var(--fg-faint); }

.main { flex: 1 1 auto; display: flex; min-height: 0; }

/* The queue. It shows what a review IS and which project it is in, and nothing else:
   anything more and you are reading the list instead of picking from it. */
#home { flex: 1 1 auto; overflow-y: auto; background: var(--bg-editor); }
.hwrap { max-width: 620px; margin: 0 auto; padding: 14vh 24px 80px; }
.qcard {
  display: flex; align-items: baseline; gap: 12px; width: 100%; text-align: left;
  font: inherit; cursor: pointer; background: transparent; border: 0;
  border-bottom: 1px solid var(--border-2); padding: 14px 4px; color: var(--fg);
}
.qcard:hover { background: var(--bg-hover); }
.qcard .n {
  flex: 0 0 18px; font-family: var(--mono); font-size: 11px; color: var(--fg-faint);
  text-align: right;
}
.qcard:hover .n { color: var(--hl-rail); }
.qcard .t { font-size: 15px; color: #fff; }
.qcard .m {
  margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--fg-faint);
  white-space: nowrap;
}
.qcard.stale { opacity: 0.45; }
.qcard.stale .t { color: var(--fg-dim); }
.qnone { color: var(--fg-faint); font-size: 12.5px; padding: 14px 4px; }
#qnote {
  font-size: 12px; color: var(--ok); padding: 0 4px 14px; letter-spacing: 0.01em;
}
#back {
  font: inherit; font-size: 14px; line-height: 1; background: transparent; color: var(--fg-dim);
  border: 1px solid transparent; border-radius: 2px; padding: 1px 8px; cursor: pointer;
}
#back:hover { background: var(--bg-hover); border-color: var(--border); }
.pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
#explorer { flex: 0 0 250px; background: var(--bg-side); }
#editor { flex: 1 1 auto; background: var(--bg-editor); }
#review { flex: 0 0 420px; background: var(--bg-side); border-left: 1px solid var(--border-2); }
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
.btn { font: inherit; font-size: 11px; background: var(--btn); color: #fff; border: 0; border-radius: 2px; padding: 4px 11px; cursor: pointer; }
.btn:hover:not(:disabled) { background: var(--btn-hover); }
.btn.ghost { background: transparent; color: var(--fg-faint); border: 1px solid var(--border); }
.btn.ghost:hover { color: var(--fg-dim); }
.btn:disabled { opacity: 0.5; cursor: default; }

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
.node .dot { flex: 0 0 6px; height: 6px; border-radius: 50%; background: transparent; }
.node.dir { color: var(--fg); }
.node .lbl { overflow: hidden; text-overflow: ellipsis; }
.kids.collapsed { display: none; }
/* The dot carries one meaning: what git says about this file. An untouched file has no
   dot at all, so a coloured one always means something happened. */
.node.file.chg .dot, .tab .dot.chg { background: var(--git-mod); }
.node.file.chg-new .dot, .tab .dot.chg-new { background: var(--git-new); }
.node.file.chg-deleted .dot, .tab .dot.chg-deleted { background: var(--git-del); }
.node.file.chg .lbl { color: #fff; }
.node.file.chg-deleted .lbl { text-decoration: line-through; text-decoration-thickness: 1px; color: var(--fg-dim); }
.node .num { margin-left: auto; padding-left: 8px; font-family: var(--mono); font-size: 9.5px; }
.node .num .a { color: var(--green); }
.node .num .r { color: var(--red); }
/* A collapsed directory still has to say that something inside it moved. */
.node.dir.chg > .lbl { color: #fff; }
.node.dir.chg > .lbl::after { content: "\2022"; color: var(--git-mod); font-size: 14px; margin-left: 6px; vertical-align: middle; }
/* Which files are part of the walk, and which ones you have written on. */
.node .mk { flex: 0 0 auto; font-size: 9.5px; font-family: var(--mono); letter-spacing: 0; }
.node .mk.stop { color: var(--them); }
.node .mk.cmt { color: var(--me); }

/* tabs */
.tabs { flex: 0 0 35px; background: var(--bg-side); display: flex; align-items: stretch; overflow-x: auto; border-bottom: 1px solid var(--border-2); }
.tab {
  display: flex; align-items: center; gap: 8px; padding: 0 10px 0 14px; cursor: pointer;
  background: var(--bg-side); color: var(--fg-faint); font-size: 13px;
  border-right: 1px solid var(--border-2); white-space: nowrap;
}
.tab.active { background: var(--bg-editor); color: var(--fg-dim); border-top: 1px solid var(--accent); }
.tab .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: transparent; }
.tab .state { color: var(--hl-rail); font-size: 9px; letter-spacing: 0.06em; margin-left: 4px; }
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

/* code */
#scroll { flex: 1 1 auto; overflow: auto; }
#code { font-family: var(--mono); font-size: 13px; line-height: 19px; min-width: max-content; padding: 10px 0 40vh; }
/* One uninterrupted column of code with evenly spaced line numbers. Nothing is allowed to
   take vertical space between two lines except an actual annotation: the per-hunk controls
   are lifted out of the flow and the state of a hunk is carried by the rail in its gutter,
   so a diff reads top to bottom the way the file does. */
.ln { display: flex; }
#scroll:focus { outline: none; }
#scroll:focus-visible { outline: none; }
/* The gutter stays put when the code scrolls sideways, and keeps its own flat background
   rather than the diff tint, so the numbers read as one straight column. */
.ln .no {
  flex: 0 0 var(--gutter); position: sticky; left: 0; z-index: 1;
  background: var(--bg-editor); border-left: 3px solid transparent;
  text-align: right; padding-right: 18px; color: var(--fg-faint);
  font-variant-numeric: tabular-nums; user-select: none; cursor: pointer;
}
.ln:hover .no { color: var(--fg-dim); }
.ln .src { white-space: pre; padding-right: 32px; }
.ln.sel { background: var(--sel-bg); }
.ln.sel .no { border-left-color: var(--accent); }
.ln.hl { background: var(--hl-bg); }
.ln.hl .no { border-left-color: var(--hl-rail); }
.ln.hl.sel { background: var(--sel-bg); }
/* A verdict shows as the rail beside the lines it was given on — no label, no row, and
   visible however far sideways the code is scrolled. */
.ln.v-ok .no { border-left-color: var(--ok); }
.ln.v-no .no { border-left-color: var(--no); }
/* Tinted background, untouched foreground: the code stays as legible as it is in the file
   itself, and the tint is the only thing the diff adds. Changes you are not currently
   deciding on are tinted far more weakly — present, but not competing for the decision. */
.ln.del { background: var(--del-off); }
.ln.add { background: var(--add-off); }
.ln.cur.del { background: var(--del); }
.ln.cur.add { background: var(--add); }
.ln.cur.del .no, .ln.cur.add .no { color: var(--fg-dim); }
.empty { padding: 40px; color: var(--fg-faint); font-size: 13px; max-width: 48ch; line-height: 1.7; }

/* The bar that opens each hunk. The pane scrolls in BOTH directions, so the controls are
   stuck to the left edge of the viewport and to the top of their own hunk, which is the
   only way to promise they are on screen when they are wanted. */
.hunk { position: relative; }
/* Lifted clear of the text flow: it is one line tall, pinned beside the gutter, and it
   costs the code no vertical space at all. It only appears while the pointer is on its own
   hunk, because the keys are how this is normally done. */
.hctl {
  position: absolute; top: 0; left: 0; height: 19px; z-index: 4;
  display: flex; align-items: center; pointer-events: none;
}
.hctl .grp {
  position: sticky; left: var(--gutter); display: inline-flex; align-items: center; gap: 1px;
  padding: 0 3px; margin-left: var(--gutter); pointer-events: none;
  background: var(--bg-editor); border-radius: 2px;
  opacity: 0; transition: opacity .12s;
}
.hunk:hover .hctl .grp { opacity: 1; pointer-events: auto; }
/* The verdict pair, wherever a verdict can be given: two glyphs, no chrome. */
.vb {
  font: inherit; font-size: 11px; line-height: 1; border: 0; border-radius: 2px;
  background: transparent; color: var(--fg-faint); padding: 1px 4px; cursor: pointer;
}
.vb.yes:hover, .vb.yes.on { color: var(--ok); }
.vb.nope:hover, .vb.nope.on { color: var(--no); }
.vb.cmt:hover { color: var(--me); }
.vb.on { font-weight: 700; }

/* the margin: what they said, what you said */
.band {
  font-family: var(--ui); font-size: 12.5px; line-height: 1.6; white-space: normal;
  margin: 9px 0; padding: 5px 14px 7px 12px; border-left: 2px solid var(--border);
  max-width: 88ch;
}
.band.them { background: var(--them-bg); border-left-color: var(--them); }
.band.mine { background: var(--me-bg); border-left-color: var(--me); }
.band .who { font-size: 9px; letter-spacing: 0.11em; text-transform: uppercase; color: var(--fg-faint); margin-bottom: 3px; display: flex; align-items: center; gap: 6px; }
.band .who .vb, .band .who .act { opacity: 0; transition: opacity .12s; }
.band:hover .who .vb, .band:hover .who .act, .band .who .vb.on { opacity: 1; }
.band.them .who b { color: var(--them); font-weight: 600; }
.band.mine .who b { color: var(--me); font-weight: 600; }
.band .who .spacer { flex: 1; }
.band .who .act { cursor: pointer; color: var(--fg-faint); text-transform: none; letter-spacing: 0; font-size: 10.5px; }
.band .who .act:hover { color: #fff; }
.band p { margin: 0 0 0.6em; }
.band p:last-child { margin-bottom: 0; }
.band code { font-family: var(--mono); font-size: 11.5px; background: #2d2d2d; padding: 1px 4px; border-radius: 2px; color: #ce9178; }
.band textarea { width: 100%; max-width: 70ch; }
.band .row { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
.band .hint { font-size: 10.5px; color: var(--fg-faint); }

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

/* the review pane */
/* How much is left, where you cannot scroll away from it. Not knowing whether a review is
   three items or thirty is the thing that makes one feel endless. */
#progress {
  flex: 0 0 auto; display: flex; align-items: center; gap: 9px; padding: 2px 14px 9px;
  border-bottom: 1px solid var(--border-2);
}
#progress .bar {
  flex: 1 1 auto; height: 4px; border-radius: 2px; background: #333; overflow: hidden;
  display: flex;
}
#progress .bar i { display: block; height: 100%; width: 0; transition: width .18s; }
#progress .bar i.ok { background: var(--ok); }
#progress .bar i.no { background: var(--no); }
#progress .txt {
  flex: 0 0 auto; font-family: var(--mono); font-size: 10.5px; color: var(--fg-faint);
  white-space: nowrap;
}
#progress .txt b { color: var(--fg-dim); font-weight: 600; }
#rbody { flex: 1 1 auto; overflow-y: auto; padding: 12px 14px 20px; }
#brief { font-size: 12.5px; line-height: 1.65; color: var(--fg-dim); padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid var(--border); }
#brief p { margin: 0 0 0.7em; }
#brief p:last-child { margin-bottom: 0; }
#brief code { font-family: var(--mono); font-size: 11.5px; background: #2d2d2d; padding: 1px 4px; border-radius: 2px; color: #ce9178; }
#brief .probs { color: var(--hl-rail); font-size: 11.5px; margin-top: 8px; }
.sect { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fg-faint); margin: 18px 0 8px; }
.sect:first-child { margin-top: 2px; }
/* One row per thing to review. Dense on purpose: this pane answers "where am I and what is
   left", and it can only do that if the whole walk fits on a screen. */
.wrow {
  display: flex; align-items: baseline; gap: 7px; cursor: pointer;
  padding: 3px 6px 3px 4px; border-left: 2px solid transparent; border-radius: 2px;
}
.wrow:hover { background: var(--bg-hover); }
.wrow.on { background: var(--bg-active); border-left-color: var(--hl-rail); }
.wrow.on .k { color: var(--hl-rail); }
.wrow .k {
  flex: 0 0 15px; font-family: var(--mono); font-size: 11px; color: var(--fg-faint);
  text-align: right;
}
.wrow .t { font-size: 12.5px; color: var(--fg-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wrow.on .t { color: #fff; font-weight: 600; }
.wrow .a { margin-left: auto; font-family: var(--mono); font-size: 10.5px; color: var(--fg-faint); white-space: nowrap; }
.wrow .c { font-size: 9px; color: var(--me); }
.wrow.ok .k { color: var(--ok); }
.wrow.no .k { color: var(--no); }
.wrow.ok .t { color: var(--fg-faint); }
.wrow.no .t { color: var(--no); }
.wrow.gone .t { color: var(--red); text-decoration: line-through; }
.cmt {
  border-left: 2px solid var(--me); background: var(--me-bg); padding: 5px 9px; margin-bottom: 5px;
  font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: var(--fg-dim);
}
.cmt .where { display: block; font-family: var(--mono); font-size: 10.5px; color: var(--me); margin-bottom: 2px; cursor: pointer; white-space: nowrap; }
.cmt .where:hover { color: #fff; }
.cmt .x { float: right; cursor: pointer; color: var(--fg-faint); padding: 0 2px; }
.cmt .x:hover { color: #fff; }
.dimnote { font-size: 11.5px; color: var(--fg-faint); line-height: 1.6; }

#rfoot { flex: 0 0 auto; border-top: 1px solid var(--border-2); padding: 10px 14px 12px; display: flex; flex-direction: column; gap: 8px; }
textarea {
  resize: none; font-family: var(--ui); font-size: 12.5px; color: var(--fg-dim);
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 2px;
  padding: 7px 9px; line-height: 1.5;
}
textarea:focus { outline: none; border-color: var(--accent); }
#general { width: 100%; min-height: 34px; max-height: 140px; }
#rfoot .row { display: flex; gap: 10px; align-items: center; }
#count { font-size: 11.5px; color: var(--fg-faint); }
#count b { color: var(--me); font-weight: 600; }
#count .y { color: var(--ok); font-weight: 600; }
#count .n { color: var(--no); font-weight: 600; }
#tally { font-family: var(--mono); font-size: 11px; letter-spacing: 0; }
#tally .y { color: var(--ok); } #tally .n { color: var(--no); }

#addSel {
  position: fixed; z-index: 50; font: inherit; font-size: 11px; font-family: var(--ui);
  background: var(--me); color: #1e1e1e; border: 0; border-radius: 3px;
  padding: 3px 9px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5); white-space: nowrap;
}
#addSel kbd { font-family: var(--mono); font-size: 10px; opacity: 0.65; margin-left: 5px; }

.status { flex: 0 0 22px; background: var(--accent); color: #fff; display: flex; align-items: center; gap: 16px; padding: 0 12px; font-size: 11.5px; user-select: none; }
.status .spacer { flex: 1; }
/* The keys, where you cannot miss them. A shortcut nobody knows about is not a shortcut. */
.status .legend { font-family: var(--mono); font-size: 10.5px; opacity: 0.85; letter-spacing: 0; }
.status .legend b { font-weight: 700; opacity: 1; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
</head>
<body>

<div class="titlebar">
  <button id="back" hidden title="Back to the queue (Esc)">&#8592;</button>
  <span class="brand">codewalk</span>
  <span class="center" id="titleRoot">dashboard</span>
  <span class="base" id="titleBase"></span>
</div>

<div id="home">
  <div class="hwrap">
    <div id="qnote" hidden></div>
    <div id="qlist"></div>
    <div id="qempty" class="qnone">nothing pending</div>
  </div>
</div>

<div class="main" id="work" hidden>
  <section class="pane" id="explorer">
    <div class="paneHead"><span id="expName">Explorer</span><span class="spacer"></span><span id="fileCount" style="text-transform:none;letter-spacing:0;color:var(--fg-faint)"></span></div>
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
    <div id="scroll" tabindex="0"><div id="code"></div>
      <div class="empty" id="empty">Nothing open yet.</div>
    </div>
  </section>

  <div class="sash" id="sash2" role="separator" aria-orientation="vertical" tabindex="0" aria-label="Resize review"></div>

  <section class="pane" id="review">
    <div class="paneHead">
      <span>Review</span><span class="spacer"></span>
      <span id="tally"></span>
      <span class="navpos" id="stopPos"></span>
      <button class="iconbtn" id="stopPrev" title="Previous stop (Alt+Up)" disabled>&#8593;</button>
      <button class="iconbtn" id="stopNext" title="Next stop (Alt+Down)" disabled>&#8595;</button>
    </div>
    <div id="progress"><div class="bar"><i class="ok"></i><i class="no"></i></div><span class="txt"></span></div>
    <div id="rbody">
      <div id="brief"></div>
      <div id="stops"></div>
      <div id="loose"></div>
    </div>
    <div id="rfoot">
      <textarea id="general" rows="1" placeholder="Anything that is not about one place in the code… (Enter sends, Shift+Enter for a line)" aria-label="General comment"></textarea>
      <div class="row">
        <button class="btn" id="send">Send review</button>
        <span id="count">no comments yet</span>
      </div>
    </div>
  </section>
</div>

<button id="addSel" hidden>Comment<kbd>Ctrl L</kbd></button>

<div class="status" id="status">
  <span id="stFile">no file</span>
  <span id="stPos"></span>
  <span id="stSel"></span>
  <span class="spacer"></span>
  <span class="legend"><b>a</b>/<b>k</b> approve &nbsp;<b>r</b> reject &nbsp;<b>c</b> comment &nbsp;<b>n</b>/<b>p</b> next/prev &nbsp;<b>u</b> undo</span>
  <span id="stState">ready</span>
</div>

<script>
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const codeEl = $("code");
  const CHG = "Δ ";          // tab-key prefix: the diff view of a path, not the file

  const STAMP = "__STAMP__";            // which build of this page is in the tab
  let rid = null;                       // the review on screen, if any
  const R = (tail) => "/api/r/" + rid + (tail ? "/" + tail : "");

  let allFiles = [], openTabs = [], active = null;
  let chgItems = [], chgByPath = new Map(), chgBase = "";
  let stops = [], curStop = -1;
  let comments = [], nextId = 1;
  // key -> {key, state, path, a, b, stop, title, rows}. A verdict answers the question
  // "is this right", which is not the question a comment answers, so it is held apart.
  const verdicts = new Map();
  let editing = null;             // {path, rowA, rowB, stop} while a comment box is open

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

  // The tree is `git ls-files` plus whatever the changes list knows that it does not.
  // A deleted file is gone from disk so git cannot report it at all; a newly written one
  // may not have been there when the list was built. Both would otherwise be invisible —
  // and they are the two states you most need to see.
  function treeFiles() {
    const seen = new Set(allFiles);
    const extra = chgItems.map((e) => e.path).filter((p) => !seen.has(p));
    return extra.length ? allFiles.concat(extra).sort() : allFiles;
  }

  function renderTree() {
    const q = $("filter").value.trim().toLowerCase();
    const el = $("tree");
    const files = treeFiles();
    el.textContent = "";
    if (q) {
      const hits = files.filter((p) => p.toLowerCase().includes(q)).slice(0, 400);
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
    $("fileCount").textContent = files.length + " files";
    el.appendChild(dirBody(buildTree(files), "", 0));
  }

  function dirBody(node, prefix, depth) {
    const frag = document.createDocumentFragment();
    for (const [name, child] of Array.from(node.dirs.entries()).sort((a, b) => a[0].localeCompare(b[0]))) {
      const full = prefix ? prefix + "/" + name : name;
      const open = expanded.has(full);
      const row = document.createElement("div");
      row.className = "node dir" + (chgUnder(full) ? " chg" : "");
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
    const ch = chgByPath.get(path) || null;
    const row = document.createElement("div");
    row.className = "node file"
      + (active === path || active === CHG + path ? " active" : "")
      + (ch ? " chg chg-" + ch.state : "");
    row.style.paddingLeft = 8 + depth * 12 + "px";
    const tw = document.createElement("span");
    tw.className = "tw";
    const dot = document.createElement("span");
    dot.className = "dot";
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = label;
    row.append(tw, dot, lbl);
    const nStop = stops.filter((s) => s.path === path).length;
    const nCmt = comments.filter((c) => c.path === path).length;
    if (nStop || nCmt) {
      const mk = document.createElement("span");
      mk.className = "mk " + (nCmt ? "cmt" : "stop");
      mk.textContent = nCmt ? "●" + (nCmt > 1 ? nCmt : "") : "○";
      mk.title = (nStop ? nStop + " stop(s)" : "") + (nStop && nCmt ? ", " : "")
               + (nCmt ? nCmt + " comment(s) of yours" : "");
      row.appendChild(mk);
    }
    if (ch && !ch.binary) {
      const n = document.createElement("span");
      n.className = "num";
      const a = document.createElement("span"); a.className = "a"; a.textContent = "+" + ch.added;
      const r = document.createElement("span"); r.className = "r"; r.textContent = " -" + ch.removed;
      n.append(a, r);
      row.appendChild(n);
      row.title = ch.state + " · open the diff";
    }
    // A changed file opens as its diff. The plain file is still one click away on the tab.
    row.onclick = () => openFile(ch ? CHG + path : path);
    return row;
  }

  function chgUnder(dir) {
    const pre = dir + "/";
    for (const p of chgByPath.keys()) if (p.startsWith(pre)) return true;
    return false;
  }

  function expandTo(path) {
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) expanded.add(parts.slice(0, i).join("/"));
  }

  function realPath(p) { return p.startsWith(CHG) ? p.slice(CHG.length) : p; }
  function viewFor(p) { return chgByPath.has(p) ? CHG + p : p; }
  function shortPath(p) {
    const b = p.split("/");
    return b.length <= 2 ? p : ".../" + b.slice(-2).join("/");
  }

  // ---------------- navigation history ----------------
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
  window.addEventListener("mouseup", (e) => {
    if (e.button === 3) { e.preventDefault(); navGo(-1); }
    if (e.button === 4) { e.preventDefault(); navGo(1); }
  });

  // ---------------- tabs + file ----------------
  function tab(path) { return openTabs.find((t) => t.path === path); }

  async function openFile(path, spec) {
    let t = tab(path);
    const isChg = path.startsWith(CHG);
    if (!t) {
      const rel = realPath(path);
      let data = isChg ? diffCache.get(rel) : null;
      if (!data) {
        const url = R(isChg ? "change" : "file") + "?path=" + encodeURIComponent(rel);
        try { data = await fetch(url).then((r) => r.json()); }
        catch (e) { flash("Could not reach the server."); return null; }
        if (data.error) { flash(data.error); return null; }
        if (isChg) diffCache.set(rel, data);
      }
      t = { path: path, rel: rel, data: data, hl: new Set(), sel: null, scroll: 0, chg: isChg,
            rows: lineRows(data) };
      openTabs.push(t);
    }
    if (active && active !== path) {
      const prev = tab(active);
      if (prev) prev.scroll = $("scroll").scrollTop;
    }
    active = path;
    t.hl = spec ? parseSpec(spec) : t.hl;
    expandTo(realPath(path));
    renderTabs();
    renderTree();
    renderCode(t, !!spec);
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
      if (next) renderCode(next, false);
      else {
        codeEl.textContent = ""; $("empty").hidden = false;
        $("stFile").textContent = "no file"; $("crumbs").textContent = "";
      }
    }
    renderTabs(); renderTree();
  }

  function renderTabs() {
    const el = $("tabs");
    el.textContent = "";
    for (const t of openTabs) {
      const d = document.createElement("div");
      d.className = "tab" + (t.path === active ? " active" : "");
      const st = chgByPath.get(t.rel);
      const dot = document.createElement("span");
      dot.className = "dot" + (st ? " chg chg-" + st.state : "");
      const n = document.createElement("span");
      n.textContent = t.data.name;
      if (t.chg) {
        const pr = document.createElement("span");
        pr.className = "state";
        pr.textContent = String(t.data.state || "changed").toUpperCase();
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

  // file line number -> row index. In a diff view the gutter numbers the file as it is
  // now, so a line number in a note or a comment is a line number in the file and needs
  // no translation; this is only the lookup back to the row that carries it.
  function lineRows(data) {
    const m = new Map();
    const nums = data.nums;
    for (let i = 0; i < data.lines.length; i++) {
      const key = nums ? nums[i] : String(i + 1);
      if (key && key !== "−" && !m.has(key)) m.set(key, i);
    }
    return m;
  }

  function rowOf(t, line) {
    const r = t.rows.get(String(line));
    return r === undefined ? -1 : r;
  }

  // Everything that wants to sit in the margin of this file, by the row it attaches to.
  // Where a note sits: immediately above the lines it is about.
  //
  // One subtlety earns the whole function. An anchor is a line number, and the removed lines
  // of a hunk have no numbers — they come first, marked with a dash. So a stop that begins
  // at a hunk's first *arriving* line would have its note slipped between the lines that went
  // and the lines that came, inside the change it was explaining. When that is the case the
  // note moves up over the removed lines too, so it sits above the whole change. A stop
  // pointing into the middle of a long hunk is left exactly where it points.
  function bandsFor(t) {
    const rh = rowHunks(t);
    const by = new Map();
    const put = (row, el) => {
      if (!by.has(row)) by.set(row, []);
      by.get(row).push(el);
    };
    const place = (row) => {
      if (row < 0 || row === undefined) return -1;
      const h = rh ? rh[row] : null;
      if (h && row === h.at + h.removed) return h.at;   // first arriving line of the hunk
      return row;
    };
    for (const s of stops) {
      if (s.path !== t.rel || s.missing) continue;
      put(place(s.a ? rowOf(t, s.a) : 0), stopBand(s));
    }
    for (const c of comments) {
      if (c.path !== t.rel) continue;
      put(place(c.rowA >= 0 && t.data === c.data ? c.rowA : rowOf(t, c.a)), cmtBand(c));
    }
    return by;
  }

  function renderCode(t, scrollToHl) {
    flushEdit();
    const curKey = (walk()[walkIdx] || {}).key || "";
    $("empty").hidden = true;
    codeEl.textContent = "";
    const frag = document.createDocumentFragment();
    const rh = rowHunks(t);
    const bands = bandsFor(t);
    const keys = rowKeys(t);
    // Rows of a hunk go inside a box of their own, which is what the absolutely positioned
    // controls are measured against. The box adds no spacing of its own, so the column of
    // code runs through it unbroken.
    let into = frag;
    for (const el of (bands.get(-1) || [])) frag.appendChild(el);
    t.data.lines.forEach((segs, i) => {
      const n = i + 1;
      const mark = t.data.marks ? t.data.marks[i] : "";
      const h = rh ? rh[i] : null;
      if (h && h.at === i) {
        into = document.createElement("div");
        into.className = "hunk";
        into.appendChild(hunkCtl(t, h));
        frag.appendChild(into);
      }
      if (bands.has(i)) for (const el of bands.get(i)) into.appendChild(el);
      // The rail says what was decided about the whole change; the lit tint says which part
      // of it you are deciding on now. They are different questions and different keys.
      const hv = h ? verdictOf(hunkTarget(t, h).key) : "";
      const hkey = keys[i] || "";
      const row = document.createElement("div");
      row.className = "ln" + (mark ? " " + mark : "")
        + (hv === "approve" ? " v-ok" : hv === "reject" ? " v-no" : "")
        + (hkey && hkey === curKey ? " cur" : "")
        + (t.hl.has(n) ? " hl" : "") + (t.sel && n >= t.sel.a && n <= t.sel.b ? " sel" : "");
      row.dataset.n = String(n);
      if (hkey) row.dataset.key = hkey;
      const no = document.createElement("span");
      no.className = "no";
      no.textContent = t.data.nums ? t.data.nums[i] : String(n);
      const src = document.createElement("span");
      src.className = "src";
      if (!segs.length) src.appendChild(document.createTextNode("​"));
      for (const s of segs) {
        if (s.c) { const sp = document.createElement("span"); sp.className = s.c; sp.textContent = s.t; src.appendChild(sp); }
        else src.appendChild(document.createTextNode(s.t));
      }
      row.append(no, src);
      into.appendChild(row);
      if (h && i === h.at + h.rows - 1) into = frag;
    });
    codeEl.appendChild(frag);

    $("stFile").textContent = t.path + "  ·  " + t.data.lines.length + " lines";
    document.title = t.data.name + " — codewalk";
    const cr = $("crumbs");
    cr.textContent = "";
    realPath(t.path).split("/").forEach((part, i, arr) => {
      const s = document.createElement("span");
      s.textContent = part;
      cr.appendChild(s);
      if (i < arr.length - 1) {
        const g = document.createElement("span");
        g.textContent = " ›"; g.style.color = "var(--border)";
        cr.appendChild(g);
      }
    });
    paintSel();

    if (scrollToHl && t.hl.size) {
      revealTop(codeEl.querySelector('.ln[data-n="'
        + Math.min.apply(null, Array.from(t.hl)) + '"]'));
    } else if (t.scroll) {
      $("scroll").scrollTop = t.scroll;
    } else if (t.data.hunks && t.data.hunks.length) {
      // A diff view aims at its first hunk. Landing at line 1 of a long file and making
      // the reader hunt is the thing this view exists to prevent.
      revealTop(codeEl.querySelector(".hunk"));
    }
  }

  // The TOP of the viewport, not the middle. The first line of a change is where reading
  // starts; everything above it belongs to the change before. If a note sits above the line,
  // that is the real top of the thing being shown, so it goes there instead — and a row
  // inside a hunk is pushed clear of the sticky bar that would otherwise cover it.
  // Scrolling the pane ourselves must not be mistaken for you scrolling it; without this
  // the jump to an item immediately re-asked the viewport where it was and talked itself
  // into a different answer.
  let scrollLock = 0;

  // How much of the code above a change comes with it. Landing a change flush against the
  // top edge says nothing about where in the file it sits, and the surrounding lines are
  // most of what tells you whether a change belongs where it is.
  const LEAD_LINES = 4;

  function lead() {
    const line = codeEl.querySelector(".ln");
    const h = line ? line.getBoundingClientRect().height : 19;
    return Math.round(h * LEAD_LINES) + 4;
  }

  function revealTop(el) {
    if (!el) return;
    scrollLock = Date.now();
    const sc = $("scroll");
    // If a note was placed above these lines, it is part of what you came to read.
    const prev = el.previousElementSibling;
    const target = (prev && prev.classList.contains("band")) ? prev : el;
    sc.scrollTop += target.getBoundingClientRect().top - sc.getBoundingClientRect().top - lead();
  }

  function rowHunks(t) {
    if (!t.data.hunks) return null;
    const out = new Array(t.data.lines.length).fill(null);
    for (const h of t.data.hunks) {
      for (let i = h.at; i < h.at + h.rows && i < out.length; i++) out[i] = h;
    }
    return out;
  }

  // Three glyphs beside the first line of a hunk, and nothing else. The count that used to
  // be here is in the review pane, where it does not cost a row of code.
  function hunkCtl(t, h) {
    const target = hunkTarget(t, h);
    const wrap = document.createElement("div");
    wrap.className = "hctl";
    const grp = document.createElement("span");
    grp.className = "grp";
    const b = document.createElement("button");
    b.className = "vb cmt";
    b.textContent = "\u270e";
    b.title = "comment on this change (c)";
    b.onclick = () => startComment(t, h.at, h.at + h.rows - 1, target.stop, null,
                                   target.stop ? null : target.key);
    grp.append(verdictBtns(target, t, "vb"), b);
    wrap.appendChild(grp);
    return wrap;
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

  function prose(el, text) {
    for (const para of String(text).split(/\n\s*\n/)) {
      if (!para.trim()) continue;
      const p = document.createElement("p");
      para.trim().split(/`([^`]+)`/).forEach((s, i) => {
        if (i % 2) { const c = document.createElement("code"); c.textContent = s; p.appendChild(c); }
        else if (s) p.appendChild(document.createTextNode(s));
      });
      el.appendChild(p);
    }
  }

  // ---------------- the margin ----------------
  function stopBand(s) {
    const el = document.createElement("div");
    el.className = "band them";
    const who = document.createElement("div");
    who.className = "who";
    const b = document.createElement("b");
    b.textContent = "stop " + s.n + (s.title ? " · " + s.title : "");
    who.appendChild(b);
    const sp = document.createElement("span"); sp.className = "spacer"; who.appendChild(sp);
    who.appendChild(verdictBtns(stopTarget(s), null, "vb"));
    const act = document.createElement("span");
    act.className = "act"; act.textContent = "comment";
    act.onclick = () => commentOnStop(s);
    who.appendChild(act);
    el.appendChild(who);
    if (s.note) prose(el, s.note);
    return el;
  }

  function cmtBand(c) {
    const el = document.createElement("div");
    el.className = "band mine";
    const who = document.createElement("div");
    who.className = "who";
    const b = document.createElement("b");
    b.textContent = "you" + (c.stop ? " · on stop " + c.stop : "");
    who.appendChild(b);
    const sp = document.createElement("span"); sp.className = "spacer"; who.appendChild(sp);
    const ed = document.createElement("span");
    ed.className = "act"; ed.textContent = "edit";
    ed.onclick = () => editComment(c);
    const rm = document.createElement("span");
    rm.className = "act"; rm.textContent = "×";
    rm.onclick = () => dropComment(c.id);
    who.append(ed, rm);
    el.appendChild(who);
    const p = document.createElement("p");
    p.style.whiteSpace = "pre-wrap";
    p.textContent = c.note;
    el.appendChild(p);
    return el;
  }

  // ---------------- writing a comment ----------------
  // The box opens on the code, not in a side panel: what you are looking at when you have
  // something to say is the line, so that is where the cursor goes.
  function startComment(t, rowA, rowB, stopN, existing, keyArg, ask) {
    cancelEdit();
    const anchor = codeEl.querySelector('.ln[data-n="' + (rowB + 1) + '"]');
    if (!anchor) return;
    const box = document.createElement("div");
    box.className = "band mine";
    const who = document.createElement("div");
    who.className = "who";
    const b = document.createElement("b");
    const at = anchorOf(t, rowA, rowB);
    b.textContent = (existing ? "edit · " : "you · ") + shortPath(t.rel)
                  + ":" + (at.a === at.b ? at.a : at.a + "-" + at.b)
                  + (stopN ? "  (stop " + stopN + ")" : "");
    who.appendChild(b);
    box.appendChild(who);
    const ta = document.createElement("textarea");
    ta.rows = 3;
    ta.placeholder = ask || "What about these lines?";
    if (existing) ta.value = existing.note;
    const row = document.createElement("div");
    row.className = "row";
    const save = document.createElement("button");
    save.className = "btn";
    save.textContent = existing ? "Save"
      : (verdictOf(keyArg || (stopN ? "stop:" + stopN : "")) === "reject"
         ? "Add comment" : "Reject with this");
    const cancel = document.createElement("button");
    cancel.className = "btn ghost"; cancel.textContent = "Cancel";
    const hint = document.createElement("span");
    hint.className = "hint"; hint.textContent = "Enter to save \u00b7 Shift+Enter for a line \u00b7 Esc to cancel";
    row.append(save, cancel, hint);
    box.append(ta, row);
    anchor.insertAdjacentElement("afterend", box);
    editing = { el: box, tab: t, rowA: rowA, rowB: rowB, stop: stopN || null };
    ta.focus();

    const commit = () => {
      const note = ta.value.trim();
      if (!note) { cancelEdit(); return; }
      const live = editing !== null;   // false when flushEdit() already tore the box down
      const stop = stopN || stopAt(t.rel, at.a, at.b);
      const key = keyArg
        || (stop ? "stop:" + stop : "line:" + t.rel + ":" + at.a + "-" + at.b);
      const back = { path: t.path, spec: at.a + "-" + at.b };
      // Writing a note IS rejecting: nobody stops to type when they are happy. So the
      // comment carries the verdict with it, and one keypress of yours stays one undo.
      const wasVerdict = verdicts.get(key);
      const rejects = verdictOf(key) !== "reject";
      if (existing) {
        const was = existing.note;
        pushUndo("edit of your comment on " + shortPath(t.rel), back, () => {
          existing.note = was;
          if (rejects) { if (wasVerdict) verdicts.set(key, wasVerdict); else verdicts.delete(key); }
        });
        existing.note = note;
      } else {
        pushUndo("your comment on " + shortPath(t.rel) + ":" + at.a, back, () => {
          comments = comments.filter((x) => x.id !== id);
          if (rejects) { if (wasVerdict) verdicts.set(key, wasVerdict); else verdicts.delete(key); }
        });
        const id = nextId++;
        comments.push({
          id: id, path: t.rel, view: t.path, key: key,
          a: at.a, b: at.b, rowA: rowA, rowB: rowB, data: t.data,
          rows: quote(t, rowA, rowB), note: note,
          stop: stop, title: stopTitle(stop),
        });
      }
      if (rejects) {
        verdicts.set(key, {
          key: key, path: t.rel, a: at.a, b: at.b, stop: stop, title: stopTitle(stop),
          rows: quote(t, rowA, rowB), state: "reject",
        });
      }
      editing = null;
      box.remove();
      if (live) afterComments(t);
      else { renderReview(); renderTree(); saveDraft(); }
    };
    editing.commit = commit;
    save.onclick = commit;
    cancel.onclick = () => cancelEdit();
    ta.addEventListener("keydown", (e) => {
      // Enter finishes the comment. A note is a sentence or two, so the newline is the rare
      // case and it gets the modifier, not the other way round.
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
      if (e.key === "Escape") { e.preventDefault(); cancelEdit(); }
    });
    box.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  function cancelEdit() {
    if (editing) { editing.el.remove(); editing = null; }
  }

  // The box is about to be torn down by something other than you deciding. Anything you
  // had written is kept — it is one click to delete a rough comment, and no clicks at all
  // get the sentence back once the pane has been rebuilt under it.
  function flushEdit() {
    if (!editing) return;
    const commit = editing.commit;
    cancelEdit();
    if (commit) commit();
  }

  // The gutter numbers a diff view by the file as it is now, so a row of removed code has
  // no number of its own. Its anchor is the first numbered row of the same hunk, which is
  // the line the removal sits at in the file that exists.
  function anchorOf(t, rowA, rowB) { return anchorIn(t.data, rowA, rowB); }

  function anchorIn(data, rowA, rowB) {
    const nums = data.nums;
    if (!nums) return { a: rowA + 1, b: rowB + 1 };
    let a = 0, b = 0;
    for (let i = rowA; i <= rowB && i < nums.length; i++) {
      const v = Number(nums[i]);
      if (!v) continue;
      if (!a) a = v;
      b = v;
    }
    if (!a) {               // nothing but removed lines: borrow the next numbered line
      for (let i = rowB + 1; i < nums.length; i++) {
        const v = Number(nums[i]);
        if (v) { a = b = v; break; }
      }
    }
    return { a: a || 1, b: b || a || 1 };
  }

  function quote(t, rowA, rowB) { return quoteIn(t.data, rowA, rowB); }

  function quoteIn(data, rowA, rowB) {
    const out = [];
    for (let i = rowA; i <= rowB && i < data.lines.length; i++) {
      out.push({
        m: data.marks ? data.marks[i] : "",
        n: data.nums ? data.nums[i] : String(i + 1),
        t: data.lines[i].map((x) => x.t).join(""),
      });
    }
    return out;
  }

  // What a verdict can be about. A hunk that a stop already covers belongs to that stop:
  // the stop is the thing the agent proposed, and answering the same lines twice under two
  // names would be two answers to one question.
  function stopTarget(s) {
    return { key: "stop:" + s.n, stop: s.n, title: s.title, path: s.path,
             a: s.a, b: s.b, rows: null };
  }

  function hunkTarget(t, h) { return hunkTargetIn(t.rel, t.data, h); }

  // The key a hunk has when no stop is allowed to speak for it.
  function hunkKeyBare(path, h) { return "hunk:" + path + ":" + h.at; }

  // The stop whose declared range covers one line of one file, if any.
  function stopForLine(path, line) {
    if (!line) return null;
    for (const s of stops) {
      if (s.path !== path || s.missing || !s.a) continue;
      if (line >= s.a && line <= s.b) return s;
    }
    return null;
  }

  // Which change each row belongs to, for the purpose of lighting one of them up.
  //
  // A new file is a single hunk covering every line, so using the hunk here lit the whole
  // file when a stop pointed at twelve lines of it. A stop's own range is what the agent
  // declared and what you are deciding on, so inside a stop the stop wins; outside it, the
  // rows fall back to their hunk, which is still a change you have not judged.
  function rowKeys(t) {
    const rh = rowHunks(t);
    const nums = t.data.nums;
    const out = new Array(t.data.lines.length).fill("");
    for (let i = 0; i < out.length; i++) {
      const h = rh ? rh[i] : null;
      if (!h) continue;
      let line = nums ? Number(nums[i]) : i + 1;
      if (!line) {                       // a removed row borrows the next real line
        for (let j = i + 1; j < out.length; j++) {
          const v = Number(nums[j]);
          if (v) { line = v; break; }
        }
      }
      const s = stopForLine(t.rel, line);
      out[i] = s ? "stop:" + s.n : hunkKeyBare(t.rel, h);
    }
    return out;
  }

  function hunkTargetIn(path, data, h) {
    const at = anchorIn(data, h.at, h.at + h.rows - 1);
    const stop = stopAt(path, at.a, at.b);
    return {
      key: stop ? "stop:" + stop : "hunk:" + path + ":" + h.at,
      stop: stop, title: stop ? stopTitle(stop) : "",
      path: path, a: at.a, b: at.b,
      get rows() { return quoteIn(data, h.at, h.at + h.rows - 1); },
    };
  }

  function verdictOf(key) {
    const v = verdicts.get(key);
    return v ? v.state : "";
  }

  // Clicking the verdict you already gave takes it back, which is the only undo it needs.
  // A rejection with no reason is almost useless to the agent, so saying no opens the box
  // for why — but it does not force one: the verdict stands either way, and the output
  // says plainly when none was given.
  function setVerdict(target, state, t) {
    const had = verdictOf(target.key);
    const prev = verdicts.get(target.key);
    const back = target.a ? { path: viewFor(target.path), spec: target.a + "-" + target.b } : null;
    pushUndo((had === state ? "took back " : "") + (state === "approve" ? "approve" : "reject")
             + " on " + (target.stop ? "stop " + target.stop : shortPath(target.path)),
             back,
             () => { if (prev) verdicts.set(target.key, prev); else verdicts.delete(target.key); });
    if (had === state) verdicts.delete(target.key);
    else verdicts.set(target.key, Object.assign({}, target, { state: state }));
    const want = had !== state && state === "reject"
                 && !comments.some((c) => c.key === target.key);
    const tb = t || tab(active);
    afterComments(tb);
    // Approving is done with that change, so it moves you to the next one — from wherever
    // the approval was given, not from wherever you happened to be standing. Rejecting does
    // not: the box for the reason is about to open on these very lines.
    if (had !== state && state === "approve") {
      const at = walkIndexOf(target.key);
      if (at >= 0) { gotoWalk(at + 1); return; }
    }
    if (!want || !tb) return;
    const a = target.a ? rowOf(tb, target.a) : 0;
    const b = target.b ? rowOf(tb, target.b) : 0;
    startComment(tb, a < 0 ? 0 : a, b < 0 ? (a < 0 ? 0 : a) : b, target.stop, null,
                 target.key, "Why not?");
  }

  function verdictBtns(target, t, cls) {
    const wrap = document.createElement("span");
    wrap.style.display = "inline-flex";
    wrap.style.gap = "5px";
    const st = verdictOf(target.key);
    const pair = [["approve", "\u2713", "yes"], ["reject", "\u2717", "nope"]];
    for (const p of pair) {
      const b = document.createElement("button");
      b.className = (cls || "vb") + " " + p[2] + (st === p[0] ? " on" : "");
      b.textContent = p[1];
      b.title = st === p[0]
        ? "take back the " + (p[0] === "approve" ? "approval" : "rejection")
        : (p[0] === "approve" ? "approve this" : "reject this");
      b.onclick = (e) => { e.stopPropagation(); setVerdict(target, p[0], t); };
      wrap.appendChild(b);
    }
    return wrap;
  }

  // ---------------- the walk, and taking it back ----------------
  // ONE order over the whole review: the agent's stops as it ordered them, then every hunk
  // it gave no stop to, file by file. It is deliberately independent of what happens to be
  // open — a change in a file you have not opened yet is exactly the one you must not skip,
  // so the page fetches every changed file's diff up front and walks all of them.
  let walkIdx = -1;
  let prefetched = false;
  const diffCache = new Map();        // path -> /api/change payload

  function walk() {
    const out = [];
    stops.forEach((st, i) => out.push({
      kind: "stop", si: i, s: st, path: st.path, key: "stop:" + st.n,
      label: st.title || shortPath(st.path),
      at: st.a ? (st.a === st.b ? String(st.a) : st.a + "-" + st.b) : "",
    }));
    for (const e of chgItems) {
      const d = diffCache.get(e.path);
      if (!d || !d.hunks) continue;
      d.hunks.forEach((h, hi) => {
        const tg = hunkTargetIn(e.path, d, h);
        if (tg.stop) return;          // a stop already speaks for these lines
        out.push({
          kind: "hunk", path: e.path, hi: hi, h: h, key: tg.key, tg: tg,
          label: shortPath(e.path), at: String(tg.a),
          counts: "−" + h.removed + " +" + h.added,
        });
      });
    }
    return out;
  }

  function walkIndexOf(key) {
    const w = walk();
    for (let i = 0; i < w.length; i++) if (w[i].key === key) return i;
    return -1;
  }

  async function gotoWalk(i) {
    const w = walk();
    if (!w.length) return;
    if (i >= w.length) {
      flash(prefetched ? "that was the last change \u2014 Send when you are ready"
                       : "still reading the other diffs\u2026");
      return;
    }
    const n = Math.max(0, Math.min(w.length - 1, i));
    const it = w[n];
    if (it.kind === "stop") { await gotoStop(it.si); return; }
    curStop = -1;
    walkIdx = -1;                 // so markCurrent() does not treat this as a no-op
    const t = await openFile(CHG + it.path);
    renderReview();
    markCurrent(n, true);
    if (!t) return;
    revealTop(codeEl.querySelectorAll(".hunk")[it.hi]);
  }

  // Every changed file's diff, up front, so the walk is complete before you start it. One
  // at a time and rendering as they land: the list grows in front of you instead of the
  // page sitting blank until the last one arrives.
  async function prefetchDiffs() {
    for (const e of chgItems) {
      if (e.binary || diffCache.has(e.path)) continue;
      try {
        const d = await fetch(R("change") + "?path=" + encodeURIComponent(e.path))
                        .then((r) => r.json());
        if (!d.error) diffCache.set(e.path, d);
      } catch (err) { /* a file we cannot read is reported by the list itself */ }
      renderReview();
    }
    prefetched = true;
    renderReview();
  }

  // What a bare a/r/c acts on: the SELECTED item — the one highlighted in the review pane
  // and named in the status bar. It used to be whatever sat at the top of the viewport,
  // which is wrong whenever two changes are on screen at once: jumping to a stop that lives
  // above another one left both visible and the verdict went to the wrong one.
  function keyTarget() {
    const cur = walk()[walkIdx];
    if (cur) return cur.kind === "stop" ? stopTarget(cur.s) : cur.tg;
    return viewportTarget();      // nothing selected yet, so fall back to what is on screen
  }

  // Only for deciding what you have scrolled to, never for deciding what a key acts on.
  function viewportTarget() {
    const t = tab(active);
    if (t && t.data.hunks && t.data.hunks.length) {
      const sc = $("scroll");
      // The same offset reveal uses, so "what is at the top" means "what you are reading".
      const top = sc.getBoundingClientRect().top + lead();
      const boxes = codeEl.querySelectorAll(".hunk");
      let best = null;
      for (let i = 0; i < boxes.length; i++) {
        const r = boxes[i].getBoundingClientRect();
        if (r.bottom > top) { best = i; break; }
      }
      if (best !== null && t.data.hunks[best]) return hunkTarget(t, t.data.hunks[best]);
    }
    if (stops[curStop] && !stops[curStop].missing) return stopTarget(stops[curStop]);
    return null;
  }

  // Every change to the review, with the inverse and the place it happened. u walks back
  // up it: undoing a verdict you gave three files ago without being shown where it was
  // would be a worse trick than making you find it yourself.
  const undoLog = [];

  function pushUndo(label, where, fn) {
    undoLog.push({ label: label, where: where, undo: fn });
    if (undoLog.length > 200) undoLog.shift();
  }

  async function undoLast() {
    const it = undoLog.pop();
    if (!it) { flash("nothing to undo"); return; }
    it.undo();
    if (it.where && it.where.path) {
      await openFile(it.where.path, it.where.spec || "");
    }
    afterComments(tab(active));
    flash("undone: " + it.label);
  }

  function stopAt(path, a, b) {
    for (const s of stops) {
      if (s.path !== path || s.missing) continue;
      if (!s.a) return s.n;                       // a stop on the whole file
      if (a <= s.b && b >= s.a) return s.n;
    }
    return null;
  }

  function stopTitle(n) {
    const s = stops.find((x) => x.n === n);
    return s ? s.title : "";
  }

  function dropComment(id) {
    const gone = comments.find((c) => c.id === id);
    if (gone) {
      pushUndo("deleting your comment on " + shortPath(gone.path) + ":" + gone.a,
               { path: gone.view || viewFor(gone.path), spec: gone.a + "-" + gone.b },
               () => { comments.push(gone); });
    }
    comments = comments.filter((c) => c.id !== id);
    afterComments(tab(active));
  }

  async function editComment(c) {
    const t = await openFile(c.view || viewFor(c.path));
    if (!t) return;
    // A comment restored from the server knows the lines it is about but not the rows they
    // were drawn on, so those are looked up again in whatever is open now.
    const a = c.rowA >= 0 ? c.rowA : rowOf(t, c.a);
    const b = c.rowB >= 0 ? c.rowB : rowOf(t, c.b);
    startComment(t, a < 0 ? 0 : a, b < 0 ? (a < 0 ? 0 : a) : b, c.stop, c, c.key);
  }

  function afterComments(t) {
    if (t) { const sc = $("scroll"), at = sc.scrollTop; renderCode(t, false); sc.scrollTop = at; }
    renderReview();
    renderTree();
    saveDraft();
  }

  async function commentOnStop(s) {
    const t = await gotoStop(stops.indexOf(s));
    if (!t) return;
    const a = s.a ? rowOf(t, s.a) : 0;
    const b = s.b ? rowOf(t, s.b) : 0;
    startComment(t, a < 0 ? 0 : a, b < 0 ? (a < 0 ? 0 : a) : b, s.n, null, "stop:" + s.n);
  }

  // ---------------- selection ----------------
  let dragFrom = null;
  codeEl.addEventListener("pointerdown", (e) => {
    // The bare keys belong to the editor, so clicking in it has to be what hands them over.
    $("scroll").focus({ preventScroll: true });
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
    $("stSel").textContent = sel ? (sel.b - sel.a + 1) + " selected · Ctrl+L to comment" : "";
  }

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
    addBtn.style.top = Math.max(34, r.rect.top - 30) + "px";
    addBtn.style.left = Math.min(window.innerWidth - 140, Math.max(8, r.rect.left)) + "px";
    addBtn.onclick = () => {
      const t = tab(active);
      window.getSelection().removeAllRanges();
      hideFloat();
      if (t) startComment(t, r.a - 1, r.b - 1);
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

  // ---------------- the review pane ----------------
  function renderReview() {
    renderWalkPane();
    renderLoose();
    const vs = Array.from(verdicts.values());
    const ok = vs.filter((v) => v.state === "approve").length;
    const no = vs.filter((v) => v.state === "reject").length;
    const nc = comments.length + ($("general").value.trim() ? 1 : 0);
    const c = $("count");
    c.textContent = "";
    if (!ok && !no && !nc) c.textContent = "nothing said yet";
    else {
      const bits = [];
      if (ok) bits.push(["y", ok + " approved"]);
      if (no) bits.push(["n", no + " rejected"]);
      if (nc) bits.push(["", nc + " comment" + (nc === 1 ? "" : "s")]);
      bits.forEach((bit, i) => {
        if (i) c.appendChild(document.createTextNode(" · "));
        const sp = document.createElement("span");
        if (bit[0]) sp.className = bit[0];
        sp.textContent = bit[1];
        c.appendChild(sp);
      });
      c.appendChild(document.createTextNode(" to send"));
    }
    // The same two numbers in the pane head, so you can see how far the walk has got
    // without reading to the bottom of it.
    const tl = $("tally");
    tl.textContent = "";
    if (ok || no) {
      const y = document.createElement("span"); y.className = "y"; y.textContent = "✓" + ok;
      const z = document.createElement("span"); z.className = "n"; z.textContent = " ✗" + no;
      tl.append(y, z);
    }
    const wl = walk().length;
    $("stopPos").textContent = wl ? (walkIdx + 1) + "/" + wl : "";
    $("stopPrev").disabled = walkIdx <= 0;
    $("stopNext").disabled = !wl || walkIdx >= wl - 1;
    renderProgress();
  }

  // An overview, and only that: position, what it is, and your verdict. The agent's own
  // note is already in the margin beside the code it is about, and printing it twice made
  // this pane a wall of prose you had to scroll past to see where you were.
  function renderWalkPane() {
    const el = $("stops");
    el.textContent = "";
    const w = walk();
    if (!w.length) {
      const d = document.createElement("div");
      d.className = "dimnote";
      d.textContent = chgItems.length
        ? "Reading the diffs\u2026"
        : "Nothing differs from " + (chgBase || "the base") + ".";
      el.appendChild(d);
      return;
    }
    let section = null;
    w.forEach((it, i) => {
      const want = it.kind === "stop" ? "stops \u00b7 in reading order"
                                      : "other changes \u00b7 no stop";
      if (want !== section) {
        section = want;
        const mine = w.filter((x) => x.kind === it.kind);
        const done = mine.filter((x) => verdictOf(x.key)).length;
        const h = document.createElement("div");
        h.className = "sect";
        h.textContent = want + "  \u00b7  " + done + "/" + mine.length;
        el.appendChild(h);
      }
      el.appendChild(walkRow(it, i));
    });
    // A file git calls binary has no diff to walk, and saying so beats leaving it out.
    const blind = chgItems.filter((e) => e.binary || (prefetched && !diffCache.has(e.path)));
    if (blind.length) {
      const h = document.createElement("div");
      h.className = "sect";
      h.textContent = "changed, nothing to show";
      el.appendChild(h);
      for (const e of blind) {
        const d = document.createElement("div");
        d.className = "dimnote";
        d.textContent = e.path + " \u2014 " + (e.binary ? "binary" : "could not be read");
        el.appendChild(d);
      }
    }
  }

  // The bar, the count, and the one line of it that sits in the editor's own status bar —
  // because while you are reading code that is the only one of the three you can see.
  function renderProgress() {
    const w = walk();
    const total = w.length;
    let ok = 0, no = 0, said = 0;
    for (const it of w) {
      const st = verdictOf(it.key);
      if (st === "approve") ok++;
      else if (st === "reject") no++;
      else if (comments.some((c) => c.key === it.key)) said++;
    }
    const el = $("progress");
    const bars = el.querySelectorAll(".bar i");
    bars[0].style.width = total ? (100 * ok / total) + "%" : "0";
    bars[1].style.width = total ? (100 * no / total) + "%" : "0";
    const txt = el.querySelector(".txt");
    txt.textContent = "";
    if (!total) { txt.textContent = prefetched ? "nothing to review" : "reading\u2026"; }
    else {
      const b = document.createElement("b");
      b.textContent = (ok + no) + " of " + total;
      txt.append(b, document.createTextNode(" judged"
        + (said ? " \u00b7 " + said + " commented" : "")
        + (total - ok - no - said > 0 ? " \u00b7 " + (total - ok - no - said) + " to go" : "")));
    }
    const sp = $("stPos");
    const cur = w[walkIdx];
    sp.textContent = total && walkIdx >= 0
      ? "change " + (walkIdx + 1) + " of " + total
        + (cur ? "  \u00b7  " + cur.label : "")
        + (prefetched ? "" : "  (still reading)")
      : "";
  }

  // Move the highlight without rebuilding the pane: this runs on every scroll frame, and
  // a full re-render there would make the editor feel heavy for no reason.
  function markCurrent(i, scrollIntoList) {
    if (i < 0 || i === walkIdx) return;
    walkIdx = i;
    // Relight the code without rebuilding it: the rows carry the key of the change they
    // belong to, so this is a class toggle per row and nothing more.
    const key = (walk()[i] || {}).key || "";
    codeEl.querySelectorAll(".ln").forEach((r) => {
      r.classList.toggle("cur", !!r.dataset.key && r.dataset.key === key);
    });
    const rows = $("stops").querySelectorAll(".wrow");
    rows.forEach((r, n) => r.classList.toggle("on", n === i));
    $("stopPos").textContent = rows.length ? (i + 1) + "/" + rows.length : "";
    $("stopPrev").disabled = i <= 0;
    $("stopNext").disabled = i >= rows.length - 1;
    renderProgress();
    if (scrollIntoList && rows[i]) rows[i].scrollIntoView({ block: "nearest" });
  }

  // What the eye is on, told by the viewport rather than by the last key pressed. Without
  // this the highlight only moved for n/p, so it quietly lied as soon as you scrolled.
  // Scrolling moves the selection only when what is selected has gone off screen
  // altogether. Reading around the item you are judging is not a decision to judge
  // something else, and the old rule turned every scroll into one.
  function syncToViewport() {
    if (Date.now() - scrollLock < 500) return;
    const cur = walk()[walkIdx];
    if (cur && onScreen(cur)) return;
    const tg = viewportTarget();
    if (!tg) return;
    markCurrent(walkIndexOf(tg.key), true);
  }

  function onScreen(item) {
    const t = tab(active);
    if (!t) return false;
    let el = null;
    if (item.kind === "stop") {
      if (t.rel !== item.s.path || !item.s.a) return false;
      const r = rowOf(t, item.s.a);
      if (r >= 0) el = codeEl.querySelector('.ln[data-n="' + (r + 1) + '"]');
    } else {
      if (t.rel !== item.path) return false;
      el = codeEl.querySelectorAll(".hunk")[item.hi] || null;
    }
    if (!el) return false;
    const box = $("scroll").getBoundingClientRect();
    const at = el.getBoundingClientRect();
    return at.bottom > box.top && at.top < box.bottom;
  }

  let syncTimer = 0;
  $("scroll").addEventListener("scroll", () => {
    clearTimeout(syncTimer);
    syncTimer = setTimeout(syncToViewport, 90);
  });

  function walkRow(it, i) {
    const st = verdictOf(it.key);
    const row = document.createElement("div");
    row.className = "wrow" + (i === walkIdx ? " on" : "")
                  + (st === "approve" ? " ok" : st === "reject" ? " no" : "")
                  + (it.kind === "stop" && it.s.missing ? " gone" : "");
    const mark = document.createElement("span");
    mark.className = "k";
    mark.textContent = st === "approve" ? "\u2713" : st === "reject" ? "\u2717"
                     : it.kind === "stop" ? it.s.n + "." : "\u00b7";
    const lbl = document.createElement("span");
    lbl.className = "t";
    lbl.textContent = it.label;
    const at = document.createElement("span");
    at.className = "a";
    at.textContent = (it.at ? ":" + it.at : "") + (it.counts ? "  " + it.counts : "")
                   + (it.kind === "stop" && it.s.missing ? "  (no such file)" : "");
    row.append(mark, lbl, at);
    const mine = comments.filter((c) => c.key === it.key);
    if (mine.length) {
      const c = document.createElement("span");
      c.className = "c";
      c.textContent = "\u25cf" + (mine.length > 1 ? mine.length : "");
      c.title = mine.length + " comment(s) of yours here";
      row.appendChild(c);
    }
    row.onclick = () => gotoWalk(i);
    return row;
  }

  function cmtRow(c) {
    const el = document.createElement("div");
    el.className = "cmt";
    const x = document.createElement("span");
    x.className = "x"; x.textContent = "×"; x.title = "delete";
    x.onclick = (e) => { e.stopPropagation(); dropComment(c.id); };
    const w = document.createElement("span");
    w.className = "where";
    w.textContent = shortPath(c.path) + ":" + (c.a === c.b ? c.a : c.a + "-" + c.b);
    w.onclick = (e) => { e.stopPropagation(); openFile(c.view || viewFor(c.path), c.a + "-" + c.b); };
    el.append(x, w, document.createTextNode(c.note));
    return el;
  }

  // Comments that belong to nothing in the walk — on an unchanged file, say. They are the
  // only ones this pane shows in full, because there is nowhere else they would be seen.
  function renderLoose() {
    const el = $("loose");
    el.textContent = "";
    const keys = new Set(walk().map((w) => w.key));
    const loose = comments.filter((c) => !keys.has(c.key));
    if (!loose.length) return;
    const h = document.createElement("div");
    h.className = "sect";
    h.textContent = "your comments elsewhere";
    el.appendChild(h);
    loose.sort((a, b) => a.path.localeCompare(b.path) || a.a - b.a);
    for (const c of loose) el.appendChild(cmtRow(c));
  }

  async function gotoStop(i) {
    if (i < 0 || i >= stops.length) return null;
    curStop = i;
    walkIdx = i;              // stops come first in the walk, so the index is the same
    const s = stops[i];
    renderReview();
    const row = $("stops").querySelector(".wrow.on");
    if (row) row.scrollIntoView({ block: "nearest" });
    if (s.missing) return null;
    const t = await openFile(viewFor(s.path), s.a ? s.a + "-" + s.b : "");
    return t;
  }

  $("stopPrev").onclick = () => gotoWalk(walkIdx - 1);
  $("stopNext").onclick = () => gotoWalk(walkIdx + 1);

  // ---------------- sending ----------------
  function payload() {
    return {
      items: comments.map((c) => ({
        key: c.key, path: c.path, a: c.a, b: c.b, stop: c.stop, title: c.title,
        rows: c.rows, note: c.note,
      })),
      verdicts: Array.from(verdicts.values()).map((v) => ({
        key: v.key, path: v.path, a: v.a, b: v.b, stop: v.stop, title: v.title,
        rows: v.rows, state: v.state,
      })),
      general: $("general").value,
    };
  }

  // Kept server-side as you type, so a tab closed by accident does not take the review
  // with it. It is a draft either way: only Send says you are finished.
  let draftTimer = 0;
  function saveDraft() {
    clearTimeout(draftTimer);
    draftTimer = setTimeout(() => {
      fetch(R("draft"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
      }).catch(() => {});
    }, 400);
  }

  async function send() {
    const n = comments.length + verdicts.size + ($("general").value.trim() ? 1 : 0);
    if (!n && !confirm("Send an empty review?\n\nNo verdict and no comment on anything —\nwhich tells the agent very little.")) return;
    // Catching a skipped change here is worth more than reporting it afterwards: you are
    // still in front of it, and one keypress fixes it.
    const open = walk().filter((it) => !verdictOf(it.key));
    if (open.length && !confirm(
        open.length + " change" + (open.length === 1 ? "" : "s")
        + " still have no verdict:\n\n  "
        + open.slice(0, 12).map((it) => (it.kind === "stop" ? "stop " + it.s.n + " — " : "")
                                        + it.label + (it.at ? ":" + it.at : "")).join("\n  ")
        + (open.length > 12 ? "\n  … and " + (open.length - 12) + " more" : "")
        + "\n\nSend anyway?")) return;
    const btn = $("send");
    btn.disabled = true;
    btn.textContent = "Sending…";
    let res = null;
    try {
      res = await fetch(R("send"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
      }).then((r) => r.json());
    } catch (e) { res = null; }
    if (!res || !res.ok) {
      btn.disabled = false; btn.textContent = "Send review";
      flash("Could not reach the server.");
      return;
    }
    // Back to the queue. This window is the one you keep open, so a review ending must
    // return you to the list, not to a dead end telling you to close it.
    const title = $("titleRoot").textContent;
    btn.disabled = false;
    btn.textContent = "Send review";
    showHome();
    qnote(n
      ? "Sent \u2014 " + n + " item" + (n === 1 ? "" : "s") + " on \u201c" + title + "\u201d"
      : "Sent \u2014 nothing flagged on \u201c" + title + "\u201d");
  }

  let noteTimer = 0;
  function qnote(text) {
    const el = $("qnote");
    el.textContent = text;
    el.hidden = false;
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => { el.hidden = true; }, 8000);
  }

  $("send").onclick = send;
  // The same gesture as saving a comment, one level up: finish writing, send the review.
  $("general").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("general").addEventListener("input", () => {
    const g = $("general");
    g.style.height = "auto";
    g.style.height = Math.min(g.scrollHeight, 140) + "px";
    renderReview();
    saveDraft();
  });

  function flash(msg) {
    $("stState").textContent = msg;
    setTimeout(() => { $("stState").textContent = "ready"; }, 2500);
  }

  // ---------------- keys ----------------
  $("filter").addEventListener("input", renderTree);
  document.addEventListener("keydown", (e) => {
    const typing = /^(TEXTAREA|INPUT)$/.test(document.activeElement.tagName);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p") {
      e.preventDefault(); $("filter").focus(); $("filter").select(); return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "l") {
      const t = tab(active);
      if (!t) return;
      const r = selectedLineRange();
      e.preventDefault();
      if (r) {
        window.getSelection().removeAllRanges();
        hideFloat();
        startComment(t, r.a - 1, r.b - 1);
      } else if (t.sel) {
        const sel = t.sel;
        t.sel = null;
        startComment(t, sel.a - 1, sel.b - 1);
      }
      return;
    }
    if (e.key === "Escape" && editing) { e.preventDefault(); cancelEdit(); return; }

    // Bare keys, scoped to the editor: a whole review without the mouse. They are off while
    // you are typing, and off in the tree, where the filter box owns the alphabet.
    const el = document.activeElement;
    const live = rid && !typing && !(el && $("explorer").contains(el));
    if (live && !e.altKey && !e.ctrlKey && !e.metaKey) {
      const k = e.key.toLowerCase();
      if ("arcnpuk".indexOf(k) >= 0 && k.length === 1) {
        e.preventDefault();
        const t = tab(active);
        if (k === "n") { gotoWalk(walkIdx + 1); return; }
        if (k === "p") { gotoWalk(walkIdx - 1); return; }
        if (k === "u") { undoLast(); return; }
        // An explicit selection of lines beats everything: you picked those out by hand.
        if (k === "c" && t) {
          const r = selectedLineRange();
          if (r) {
            window.getSelection().removeAllRanges();
            hideFloat();
            startComment(t, r.a - 1, r.b - 1);
            return;
          }
          if (t.sel) {
            const sel = t.sel; t.sel = null;
            startComment(t, sel.a - 1, sel.b - 1);
            return;
          }
        }
        actOnSelected(k);
        return;
      }
    }

    if (!e.altKey || e.ctrlKey || e.metaKey || typing) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); navGo(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); navGo(1); }
    if (e.key === "ArrowUp") { e.preventDefault(); gotoWalk(walkIdx - 1); }
    if (e.key === "ArrowDown") { e.preventDefault(); gotoWalk(walkIdx + 1); }
  });

  // a, r and c all act on the selected item, which may not be in the file on screen — you
  // can wander off with the tabs or the tree. Going there first is what makes the three keys
  // mean the same thing wherever you are standing.
  async function actOnSelected(k) {
    const tg = keyTarget();
    const verb = k === "r" ? "reject" : k === "c" ? "comment on" : "approve";
    if (!tg) { flash("nothing selected to " + verb); return; }
    let t = tab(active);
    if (!t || t.rel !== tg.path) {
      await gotoWalk(walkIdx >= 0 ? walkIdx : 0);
      t = tab(active);
      if (!t || t.rel !== tg.path) { flash("could not open " + shortPath(tg.path)); return; }
    }
    if (k !== "c") {
      setVerdict(tg, k === "r" ? "reject" : "approve", t);
      return;
    }
    const a = tg.a ? rowOf(t, tg.a) : 0;
    const b = tg.b ? rowOf(t, tg.b) : 0;
    startComment(t, a < 0 ? 0 : a, b < 0 ? (a < 0 ? 0 : a) : b, tg.stop, null, tg.key);
  }

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
  makeSash($("sash2"), $("review"), "right");

  // ---------------- the queue, and opening one review from it ----------------
  // The dashboard outlives every review in it, so the page has two states and a lot of
  // per-review state to throw away between them. Doing that in one place is the only way
  // the second review you open behaves like the first.
  function resetReview() {
    openTabs = []; active = null;
    allFiles = []; chgItems = []; chgByPath = new Map(); chgBase = "";
    stops = []; curStop = -1; walkIdx = -1; prefetched = false;
    comments = []; verdicts.clear(); diffCache.clear(); undoLog.length = 0;
    hist = []; hIdx = -1; editing = null;
    codeEl.textContent = ""; $("tabs").textContent = ""; $("tree").textContent = "";
    $("brief").textContent = ""; $("stops").textContent = ""; $("loose").textContent = "";
    $("general").value = ""; $("crumbs").textContent = "";
    $("send").disabled = false; $("send").textContent = "Send review";
    $("empty").hidden = false;
    $("empty").textContent = "Nothing open yet.";
    $("stFile").textContent = "no file"; $("stPos").textContent = "";
    updateNav();
  }

  function showHome() {
    rid = null;
    $("home").hidden = false;
    $("work").hidden = true;
    $("status").hidden = true;
    $("back").hidden = true;
    $("titleRoot").textContent = "dashboard";
    $("titleBase").textContent = "";
    document.title = "codewalk";
    refreshQueue();
  }

  function ago(secs) {
    if (secs < 60) return secs + "s ago";
    if (secs < 3600) return Math.round(secs / 60) + "m ago";
    return Math.round(secs / 3600) + "h ago";
  }

  let queue = [];

  async function refreshQueue() {
    let d = null;
    try { d = await fetch("/api/queue").then((r) => r.json()); }
    catch (e) { return; }
    // The server has been restarted with different code since this tab loaded. Drafts live
    // on the server, so nothing is lost by reloading — but not out from under a review in
    // progress, where it would be startling.
    if (d.stamp && d.stamp !== STAMP) {
      if (!rid) { location.reload(); return; }
      $("titleBase").textContent = "dashboard updated \u2014 reload when you finish";
    }
    queue = (d.pending || []).concat(d.stale || []);
    const list = $("qlist");
    list.textContent = "";
    queue.forEach((c, i) => list.appendChild(qcard(c, i)));
    $("qempty").hidden = queue.length > 0;
    // The tab is the notification: a count in the title is the whole of it, and it is the
    // one thing you can see while this window is behind your editor.
    if (!rid) {
      const n = (d.pending || []).length;
      document.title = n ? "codewalk (" + n + ")" : "codewalk";
    }
  }

  function qcard(c, i) {
    const el = document.createElement("button");
    el.className = "qcard" + (c.state === "gone" ? " stale" : "");
    const n = document.createElement("span");
    n.className = "n"; n.textContent = i < 9 ? String(i + 1) : "";
    const t = document.createElement("span");
    t.className = "t"; t.textContent = c.title;
    const m = document.createElement("span");
    m.className = "m";
    m.textContent = c.repo + (c.state === "gone" ? "  \u00b7  agent gone" : "");
    el.append(n, t, m);
    el.onclick = () => openReview(c.id);
    return el;
  }

  async function openReview(id) {
    resetReview();
    rid = id;
    let d = null;
    try { d = await fetch(R("")).then((r) => r.json()); }
    catch (e) { showHome(); return; }
    if (d.error) { showHome(); return; }

    $("home").hidden = true;
    $("work").hidden = false;
    $("status").hidden = false;
    $("back").hidden = false;
    $("titleRoot").textContent = d.title;
    document.title = d.title + " — codewalk";

    allFiles = d.files || [];
    chgItems = d.changes || [];
    chgBase = d.base || "";
    chgByPath = new Map(chgItems.map((e) => [e.path, e]));
    stops = (d.review && d.review.stops) || [];
    $("titleBase").textContent = (d.name || "") + (chgBase ? "  vs " + chgBase : "");
    $("expName").textContent = d.name;

    // What you had already said about this one, if you have been here before. Leaving puts
    // a review back on the pile, and a pile you cannot step away from is not a queue.
    const draft = d.draft || {};
    for (const v of (draft.verdicts || [])) {
      if (v && v.key) verdicts.set(v.key, v);
    }
    for (const c of (draft.items || [])) {
      if (!c || !c.key) continue;
      comments.push(Object.assign({}, c, {
        id: nextId++, view: viewFor(c.path), rowA: -1, rowB: -1, data: null,
      }));
    }
    $("general").value = draft.general || "";

    const brief = $("brief");
    const text = (d.review && d.review.brief) || "";
    if (text) prose(brief, text);
    else {
      const p = document.createElement("div");
      p.className = "dimnote";
      p.textContent = stops.length
        ? "No brief \u2014 start at stop 1."
        : "No notes. Everything that differs from " + (chgBase || "the base")
          + " is listed below; select lines and press Ctrl+L to comment.";
      brief.appendChild(p);
    }
    if (!d.waiting) {
      const warn = document.createElement("div");
      warn.className = "probs";
      warn.textContent = "\u26a0 The agent that asked for this has stopped waiting. "
                       + "Anything you send will not reach it.";
      brief.appendChild(warn);
    }
    const probs = (d.review && d.review.problems) || [];
    if (probs.length) {
      const box = document.createElement("div");
      box.className = "probs";
      for (const p of probs) {
        const l = document.createElement("div");
        l.textContent = "\u26a0 " + p;
        box.appendChild(l);
      }
      brief.appendChild(box);
    }

    renderTree();
    renderReview();
    // You land where the agent wants you looking. The rest of the diffs stream in behind
    // it, so the first stop is on screen while they arrive.
    if (stops.length) await gotoWalk(0);
    else if (chgItems.length) await openFile(CHG + chgItems[0].path);
    else $("empty").textContent = "Nothing differs from " + (chgBase || "the base") + ".";
    await prefetchDiffs();
    if (!stops.length) gotoWalk(0);
  }

  // Leaving without sending puts it back on the pile rather than holding it open.
  async function leaveReview() {
    if (rid) { try { await fetch(R("defer"), { method: "POST" }); } catch (e) {} }
    showHome();
  }

  $("back").onclick = leaveReview;
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^(TEXTAREA|INPUT)$/.test(document.activeElement.tagName)) return;
    if (!rid) {
      // On the queue, the numbers open things. Nothing else there needs a key.
      if (e.key >= "1" && e.key <= "9") {
        const c = queue[Number(e.key) - 1];
        if (c) { e.preventDefault(); openReview(c.id); }
      }
      return;
    }
    if (e.key === "Escape" && !editing) { e.preventDefault(); leaveReview(); }
  });

  setInterval(refreshQueue, 2500);
  showHome();
})();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------------
# the two sides: the dashboard you leave open, and what an agent calls
# ----------------------------------------------------------------------------------

DEFAULT_PORT = int(os.environ.get("CODEWALK_PORT") or 8765)


def api(port: int, path: str, payload=None, timeout: float = 70.0) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"error": f"HTTP {e.code}"}


def no_dashboard(port: int) -> str:
    return (f"codewalk: no dashboard listening on port {port}.\n"
            f"          start one with:  codewalk.py serve --port {port}")


def cmd_serve(args) -> int:
    Handler.queue = Queue()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"codewalk: could not listen on port {args.port}: {e}")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"codewalk dashboard  {url}")
    print("          waiting for agents to submit reviews (ctrl-c to stop)")
    if lex is None:
        print("          note: Pygments not found — showing plain text")
    print(flush=True)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ncodewalk: dashboard stopped")
    return 0


def cmd_submit(args) -> int:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"codewalk: not a directory: {root}")
    notes = ""
    if args.notes:
        try:
            with open(args.notes, encoding="utf-8", errors="replace") as fh:
                notes = fh.read()
        except OSError as e:
            sys.exit(f"codewalk: could not read --notes: {e}")

    try:
        res = api(args.port, "/api/submit", {
            "title": args.title, "root": root, "notes": notes,
            "base": args.base, "agent": args.agent,
        }, timeout=30)
    except (urllib.error.URLError, OSError):
        sys.exit(no_dashboard(args.port))
    if not res.get("ok"):
        sys.exit("codewalk: the dashboard refused the review: "
                 + str(res.get("error") or "unknown reason"))

    print(f"codewalk: submitted {res['id']} — \"{res['title']}\"")
    print(f"          {res['files']} changed path(s), {res['stops']} stop(s), "
          f"vs {res['base'] or 'the base'}")
    print(f"          waiting in the dashboard at http://127.0.0.1:{args.port}/", flush=True)
    return wait_for(args.port, res["id"], args.wait)


def cmd_wait(args) -> int:
    return wait_for(args.port, args.id, args.wait)


def wait_for(port: int, rid: str, limit: int) -> int:
    """Hold the agent here until the review comes back.

    A chain of short polls rather than one long one: a connection held open for an hour is
    at the mercy of everything between here and the socket, and this way a dashboard that
    goes away is noticed within half a minute instead of never.
    """
    deadline = time.time() + limit
    while time.time() < deadline:
        try:
            res = api(port, f"/api/wait?id={urllib.parse.quote(rid)}&for=25", timeout=40)
        except (urllib.error.URLError, OSError):
            print()
            print("=" * 72)
            print("REVIEW LOST — the dashboard is no longer listening")
            print("=" * 72)
            print()
            print(no_dashboard(port))
            return 4
        state = res.get("state")
        if state == "answered":
            print()
            print("=" * 72)
            print("REVIEW RETURNED — it was sent from the dashboard")
            print("=" * 72)
            print()
            print(res.get("text") or "(empty)")
            return 0
        if state == "unknown":
            print()
            print("=" * 72)
            print(f"REVIEW {rid} IS NOT IN THE QUEUE — the dashboard was restarted")
            print("=" * 72)
            return 4
    print()
    print("=" * 72)
    print(f"STILL WAITING — nothing sent within {limit}s")
    print("=" * 72)
    print()
    print(f"The review is still in the dashboard. Re-attach with:\n"
          f"    codewalk.py wait {rid} --port {port}")
    return 3


def page_stamp() -> str:
    """A fingerprint of the page this process serves.

    The dashboard is meant to stay open for days while this program is restarted under it,
    and a tab holding yesterday's JavaScript against today's server looks exactly like a
    bug in the UI — which it was, twice. The page carries its own stamp and reloads itself
    when the one the server reports stops matching.
    """
    global _STAMP
    if not _STAMP:
        _STAMP = hashlib.sha256(PAGE.encode("utf-8")).hexdigest()[:10]
    return _STAMP


_STAMP = ""


def main():
    ap = argparse.ArgumentParser(
        description="A dashboard agents submit code reviews to, and the client they "
                    "submit with.")
    subs = ap.add_subparsers(dest="cmd")

    s = subs.add_parser("serve", help="run the dashboard (leave this open)")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = subs.add_parser("submit", help="submit a review and block until it comes back")
    s.add_argument("--title", required=True,
                   help="what this change IS, in a few words — it is what the dashboard "
                        'list shows, e.g. "residual MLP architecture"')
    s.add_argument("--notes", default=None, metavar="PATH",
                   help="the notes file: a brief, then one `## path:lines  title` heading "
                        "per stop, in the order they should be read")
    s.add_argument("--root", default=".", help="the repository being reviewed (default: .)")
    s.add_argument("--base", default=None, metavar="REF",
                   help="what the diff is measured against (default: HEAD)")
    s.add_argument("--agent", default="", metavar="NAME",
                   help="who is asking, when several of you are working at once")
    s.add_argument("--wait", type=int, default=3600, metavar="SECONDS",
                   help="give up after this long (default 3600). The review stays in the "
                        "dashboard either way")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.set_defaults(fn=cmd_submit)

    s = subs.add_parser("wait", help="re-attach to a review already in the dashboard")
    s.add_argument("id")
    s.add_argument("--wait", type=int, default=3600, metavar="SECONDS")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.set_defaults(fn=cmd_wait)

    args = ap.parse_args()
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
