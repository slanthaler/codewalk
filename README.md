# codewalk

A dashboard that agents submit code reviews to. You keep one open; they queue up in it.

```bash
codewalk.py serve                                     # leave this open
codewalk.py submit --title "..." --notes notes.md     # what an agent runs
codewalk.py wait <id>                                 # re-attach to one
```

`submit` posts a review and **blocks** until you send it back; your verdicts and comments are
printed to its stdout, which is how they reach the agent that called it. Each review carries
its own repository root and base commit, so several agents in several checkouts can be waiting
at once.

The main view is the pending list — one card per review, titled with what the change *is*.
Press `1`–`9` or click to open one, and you get three panes:

```
left     that repository's file tree, changed files badged
center   the open file, syntax-highlighted, with the diff spliced in place, the agent's
         notes and your comments pinned to the lines they are about
right    every change in reading order, your verdicts, and Send
```

Sending returns you to the queue; `Esc` puts a review back on the pile without sending. The dashboard never writes to any
repository — there is no accept, revert or finalize, and `git` stays your undo.

## The notes file

Markdown. Everything before the first heading is the brief. Each `##` heading is one stop:
the first token is the anchor, the rest of the line is the title, and the file's order is the
reading order — code flow, not filename order.

```markdown
A second architecture, behind a config flag. The default is unchanged.

## config.py:19-28  The settings for the new architecture
Four new fields. `arch` stays a string so the config still serialises to JSON.

## model.py:85-97  Where it gets built
One branch in `build_model`, still the only place a model is constructed.

## blocks.py:11-38  The block itself
Pre-norm, which is the one decision in here worth arguing about.
```

Line numbers are lines in the file as it is now. Changed files with no stop are listed after
the last one, so the ordering hides nothing. Without `--notes` you get a plain diff review.

## What comes back

Two kinds of thing: a **verdict** on something the agent proposed — approve or reject, on a
stop or on a hunk it gave no stop to — and a **comment**, prose about particular lines. They
answer different questions, so they stay apart.

```
# codewalk review — 1 approved, 1 rejected, 1 comment(s), a general note
# /home/sam/Progs/mlp-demo   ·   base HEAD @ e63a591c

## stop 1 — config.py:19-28  "The settings for the new architecture"   [APPROVED]
  +    21      arch: str = "mlp"

## stop 5 — model.py:39-53  "Initialisation, where I am least confident"   [REJECTED]
       45          nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
> put the 1/sqrt(2*depth) scaling in now, I will use depth 16

## general
> good split, ship the init fix and resubmit

## not reviewed
> No verdict and no comment on these changed paths:
>   README.md
```

A verdict prints together with the comments explaining it, in stop order. The last section
names what was never judged — skipped stops as well as untouched files — because silence is
not approval. The dashboard also asks before sending a review with changes still unjudged.

An agent that stops waiting leaves its review marked in the list, since anything you send it
then goes nowhere. `--wait` expiring on the agent's side exits 3 and loses nothing: the review
stays queued and `codewalk.py wait <id>` picks it up again.

## Keys

Bare keys, active whenever you are not typing and not in the file tree:

| key | does |
|-----|------|
| `a` or `k` | approve the change you are looking at, and move to the next one |
| `r` | reject it, and open the box for why |
| `c` | comment on the selected change, which also rejects it |
| `n` / `p` | next / previous, over every change: the stops in the agent's order, then every hunk no stop covers, across all files |
| `u` | undo the last thing you did and go back to where you did it |
| `Esc` | back to the queue |

Also `Ctrl+L` to comment on a selection, `Alt+←`/`Alt+→` to walk back and forward through
places visited, `Ctrl+P` to filter the tree, `Ctrl+Enter` to save a comment — or to send the
whole review, from the general box at the bottom — and dragging the line-number gutter to
select a range.

Writing a comment counts as a rejection: nobody stops to type when they are happy, so a
note carries the verdict with it. Press `a` afterwards if you want to approve it anyway.
Clicking a verdict again takes it back. Rejecting opens the box for the reason — a nudge, not
a requirement; a bare rejection is reported as one. `n` lands on the top of each change, with
its note above it. The right-hand pane is an overview: a progress bar, then one row per change
with its verdict, and the change you are looking at highlighted as you scroll. The agent's
notes live in the margin next to the code, not in that list.

`CODEWALK_PORT` sets the default port for all three commands.

Standard library only, plus Pygments for highlighting (optional — falls back to plain text).

## Licence

MIT. See `LICENSE`.
