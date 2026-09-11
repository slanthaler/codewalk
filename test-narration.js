// Does the narrator speak every word?
//
//   node test-narration.js [extra-answers.json]
//
// The chunker is pulled straight out of codewalk.py and fed the demo script one
// word at a time, the way a streaming answer arrives. Every chunk it hands to the
// speech queue is collected and compared against the prose it should have read.
// This is the check that catches the whole class of bug where a unit is short in
// one pass and whole in the next, so the queue never advances over the middle.
//
// An optional JSON file of extra answer strings (real transcripts, say) is tested too.

const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "codewalk.py");
const src = fs.readFileSync(SRC, "utf8");

function grab(from, to) {
  const i = src.indexOf(from);
  const j = src.indexOf(to, i);
  if (i < 0 || j < 0) throw new Error("cannot find " + from.trim() + " in codewalk.py");
  return src.slice(i, j);
}

const core = [
  grab("  const OPEN_RE", "  function renderReply"),
  grab("  const FENCE_RE", "  function snippet"),
  grab("  // ONE definition of where a spoken unit ends", "  function narrReplay"),
  grab("  function isPath", "  function allDirectives"),
  grab("  function allDirectives", "  function markActiveChip"),
  grab("  function shortPath", "  function inline"),
].join("").replace(/\n  function narrFeed/, "\nfunction narrFeed");

let narrOn = true, narrRate = 1, spoken = [];
function narrEnqueue(items) { for (const it of items) spoken.push(it); }
// A direct eval keeps its const/let to itself, so the pieces the test needs are
// handed back out explicitly. The point is to run the page's own code, not a copy.
const F = {};
eval(core + "\nObject.assign(F, {CONT_RE, TITLE_RE, EDIT_RE, FENCE_RE, OPEN_RE, narrFeed, narrClean});");


const demo = fs.readFileSync(path.join(__dirname, "demo-walkthrough.md"), "utf8");
const turns = demo.replace(/^(?:#[^\n]*\n|\s*\n)+/, "").split(/^===[^\n]*\n/m)
                  .map((t) => t.trim()).filter(Boolean);
if (process.argv[2]) turns.push(...JSON.parse(fs.readFileSync(process.argv[2], "utf8")));

const norm = (t) => t.replace(/\s+([,.;:!?])/g, "$1").replace(/[\s,]+/g, " ").trim();
let bad = 0;

turns.forEach((turn, n) => {
  spoken = [];
  const state = { raw: "", el: null, sent: 0 };
  let acc = "";
  for (const word of turn.match(/\S+\s*/g) || []) {
    acc += word;
    F.narrFeed(state, acc, null, false);      // streaming
  }
  F.narrFeed(state, acc, null, true);         // the turn ends

  const said = spoken.map((c) => c.say).filter(Boolean).join(" ");
  const want = F.narrClean(
    turn.replace(F.CONT_RE, "").replace(F.TITLE_RE, "").replace(F.EDIT_RE, " ")
        .replace(F.FENCE_RE, " ")
        .replace(F.OPEN_RE, (m, p, spec, label) => (label ? " " + label + " " : " ")),
  );
  const ok = norm(said) === norm(want);
  if (!ok) {
    bad++;
    const a = norm(said), b = norm(want);
    let i = 0;
    while (i < a.length && a[i] === b[i]) i++;
    console.log(`turn ${n}: LOSES TEXT at char ${i}`);
    console.log("   said: ..." + a.slice(Math.max(0, i - 40), i + 100));
    console.log("   want: ..." + b.slice(Math.max(0, i - 40), i + 100));
  } else {
    console.log(`turn ${n}: ok  (${spoken.length} chunks, ${spoken.filter((c) => c.idx !== null).length} chips)`);
  }
});

console.log(bad ? `\nFAIL: ${bad}/${turns.length} turns lose text` : `\nPASS: ${turns.length} turns speak every word`);
process.exit(bad ? 1 : 0);
