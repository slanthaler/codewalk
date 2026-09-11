# Demo script
#
# Canned answers for `codewalk.py --demo`, so the page, the narration and the pane
# following can be worked on without spending a model call on every reload. Run it
# against this repo — the line ranges point at codewalk.py itself:
#
#     python codewalk.py ~/Progs/codewalk --demo --port 8766
#
# Turns are separated by a line starting with "=== "; text after === is a comment.
# Answers are served in order and the last one repeats. Within an answer, a
# [[continue: ...]] is a part break: the reader sees one part at a time.

=== a long answer in parts
[[title: what codewalk is]]

Codewalk is one Python file that serves a three pane page and drives a Claude Code session behind
it [[open:codewalk.py:1-20|the module docstring]]. The contract it describes is the interesting
part: the model does not print code into the chat, it points at code, and the center pane follows.

The pieces are a repo reader, a shadow tree for edits the user has not accepted, the subprocess
wrapper around the claude CLI, and a single page of HTML at the bottom. They are stacked in that
order in the file, which is also the order of dependency.

[[continue: how a file gets read]]

Reading a file is deliberately boring [[open:codewalk.py:137-165|Repo.read]]. It returns the text,
a digest and the line count, and nothing else. The digest matters: an edit proposed against a file
that has since changed is refused rather than applied to the wrong lines, e.g. after you edited it
yourself in another window.

Tokenizing happens on the server because shipping a syntax highlighter to the page would be a
bigger dependency than the whole program. It is a Pygments pass flattened into rows of spans,
which is the shape the pane wants.

[[continue: the shadow tree for proposed edits]]

The shadow tree is the part I would defend hardest [[open:codewalk.py:207-262|class Shadow]]. When
the model writes a file it does not write it into your repo; it writes into a mirror under the
cache directory, and the page shows you a diff with Apply and Discard.

It lives outside the repo on purpose. The file tree comes from git ls-files, so a shadow inside the
root would either pollute the tree or be invisible to it — neither is what you want when you are
reviewing a proposal.

[[continue: where this is fragile]]

The transcript is held in memory and written on sync, so a crash between syncs loses the
conversation even though the session id survives it.

There is no back pressure on the model either: a long answer streams as fast as it arrives and the
reader is assumed to keep up, which is the assumption both the narration and these part breaks were
added to break.

=== a short answer, no parts at all
[[title: where the tokenizer lives]]

Server side [[open:codewalk.py:363-400|tokenize]], as a Pygments pass flattened into rows of
{class, text} spans. The page never sees a lexer.

=== the shapes that used to break narration
[[title: two shapes worth a test]]

The file list comes from git when git will answer [[open:codewalk.py:91-120|`Repo.files`]], and
falls back to walking the tree when it will not: a label written in backticks, which Claude does
constantly, used to tear the directive in half and print it raw in the chat.

The other shape is a directive standing alone on its own line:

[[open:codewalk.py:169-195|Repo.apply, the guarded write]]

Everything after such a line used to go unspoken, because the label was a chunk by itself in one
streaming pass and part of the next chunk in the pass after it. Writing the syntax as
`[[open:codewalk.py:1-5]]` inside backticks stays quoted: it is code, not a chip.

=== written for the eye, said for the ear
[[title: shapes in the ear]]

The transformer stack folds the patch axis into the batch axis before attending across variates,
which on screen is written `(b*n, v, d)`. [[say: the transformer stack folds the patch axis into
the batch axis before attending across variates.]]

Shapes are the usual case: a tensor written as `(batch, variates, patches, patch_length)` is four
dimensional, and the voice says it as batch by variates by patches by patch length, which is why
the written form stays on screen where it can be read.
