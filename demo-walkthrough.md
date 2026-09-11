# Demo walkthrough script
#
# Canned answers for `codewalk.py --demo`, so the page, the narration and the
# pane-following can be worked on without spending a model call on every reload.
# Run it against this repo — the line ranges below point at codewalk.py itself:
#
#     python codewalk.py ~/Progs/codewalk --demo --port 8766
#
# Turns are separated by a line starting with "=== ". Text after === is a comment
# (what the answer is for). Answers are served in order and the last one repeats.
# Lines starting with "#" at the very top of the file are this header only.

=== the tour
[[title: what codewalk is]]

Codewalk is one Python file that serves a three pane page and drives a Claude Code
session behind it. The whole server is here [[open:codewalk.py:1-20|the module docstring]],
and the contract it describes is the interesting part: the model does not print code
into the chat, it points at code with directives and the center pane follows.

The pieces are a repo reader, a shadow tree for edits the user has not accepted, the
subprocess wrapper around `claude -p`, and a single page of HTML at the bottom. They
are stacked in that order in the file, which is also the order of dependency.

[[continue: how a file gets read and tokenized]]

=== reading files
Reading a file is deliberately boring [[open:codewalk.py:137-165|Repo.read]] — it returns
the text, a digest and the line count, and nothing else. The digest matters: an edit
proposed against a file that has since changed is refused rather than applied to the
wrong lines, e.g. after you edited it yourself in another window.

Tokenizing happens on the server [[open:codewalk.py:363-400|tokenize]] because shipping a
syntax highlighter to the page would be a bigger dependency than the whole program.
It is a Pygments pass flattened into rows of spans, which is the shape the pane wants.

[[continue: the shadow tree for proposed edits]]

=== the shadow tree
The shadow tree is the part I would defend hardest [[open:codewalk.py:207-262|class Shadow]].
When the model writes a file it does not write it into your repo; it writes into a mirror
under the cache directory, and the page shows you a diff with Apply and Discard.

It lives outside the repo on purpose [[open:codewalk.py:200-205|default_shadow]]. The file
tree comes from `git ls-files`, so a shadow inside the root would either pollute the tree
or be invisible to it — neither is what you want when you are reviewing a proposal.

[[continue: how the model session is kept alive]]

=== the claude session
One `claude -p` process per turn, resumed by session id [[open:codewalk.py:616-645|_argv]].
The flags are the policy: read, grep, glob, write and edit are allowed, Bash and the web
are not, and the shadow directory is added so writes land there instead of in your repo.

The speculative part is the one that will bite someone [[open:codewalk.py:669-700|prefetch]].
When a step ends with a Continue, the next answer is guessed in a forked session while you
are still reading. If you press Continue, it is already there; if you ask something else,
the guess is thrown away. A fork is used rather than a resume so the canonical session is
never advanced by a guess that nobody accepted.

[[continue: where this is fragile]]

=== weaknesses
Here is what I would fix first. The transcript is held in memory and written on sync
[[open:codewalk.py:1216-1250|Handler.sync]], so a crash between syncs loses the conversation
even though the session id survives it.

There is no back pressure on the model either: a long answer streams as fast as it arrives
and the reader is assumed to keep up, which is exactly the assumption the narration was
added to break. And the demo script you are reading right now is the only test — there is
no test suite in this repo at all.

[[continue: anything you want to poke at]]
