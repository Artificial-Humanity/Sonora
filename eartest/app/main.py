"""Sonora ear test: blind pairwise A/B between two acoustic checkpoints.

The verdict surface for a corpus or model change. `score_holdout.py` prices what a
change COSTS on dev-clean read speech; this is the only instrument we own that can see
what it BOUGHT, because the thing being bought is expressive delivery and no metric
here measures it.

⚠ BLINDING IS STRUCTURAL, NOT A PROMISE. The renderer writes the unblinding map
(opaque id -> arm) to a SIBLING `_keys/` directory, outside the test directory this
container mounts. There is no code path from this app to the arm names because the
bytes are not in the container. Unblinding is a separate step, after the verdicts land.

⚠ The key started out INSIDE the test directory, which would have made blinding a
promise that this file never opens a readable path. Mounting the whole test directory
is also what keeps `items.json` current: a single-file bind mount pins an inode, so a
re-render that replaced the file by rename would serve the OLD manifest forever.

⚠ NO RUNNING TALLY IN THE UI. Progress is shown, the A-vs-B split is not. A listener
who can see "A is winning 7-2" hears the eighth item differently, and the whole point
of a forced-choice test is that each judgement is independent.

⚠ SET A AND SET B ARE NEVER POOLED. They ask different questions of a corpus change
that pulled two levers of very different size. Pooling them averages a 22% treatment
with a 1.7% one and reports a number belonging to neither.

Verdicts land in verdicts/verdicts.csv (current state, rewritten atomically) and
verdicts/verdicts_history.csv (append-only). Same shape as the audition app, and
deliberately a DIFFERENT file: ratings.csv is the dataset SSOT and holds judgements
about corpus clips, not about checkpoints.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(os.environ.get("EARTEST_ROOT", "/test"))
CLIPS = ROOT / "clips"
ITEMS_JSON = ROOT / "items.json"
VERDICT_DIR = ROOT / "verdicts"
VERDICTS = VERDICT_DIR / "verdicts.csv"
HISTORY = VERDICT_DIR / "verdicts_history.csv"
FIELDS = ["item", "set", "choice", "confidence", "note", "ts"]

# Serialize read-modify-write of verdicts.csv against our own threads. Same reason the
# audition app does it: uvicorn runs handlers in a threadpool, and two saves landing
# together silently drop one.
_LOCK = Lock()

app = FastAPI(title="Sonora ear test")

# What each set is asking. Shown in the UI, because a listener told only "pick one"
# picks on whatever they notice first, which is usually loudness or speed.
SET_BRIEF = {
    "A_domain_vat": {
        "title": "Set A — domain and V/A/T",
        "ask": ("Delivery lane is BLANK for both. This is the part of the corpus change "
                "that was actually large. Listen for naturalness, prosody and life — "
                "does one read sound more like a person and less like a reader?"),
    },
    "B_delivery_lane": {
        "title": "Set B — delivery lane",
        "ask": ("Same text, one delivery lane requested. First ask whether the lane is "
                "audible AT ALL, then which arm does it better. This lane is taught by "
                "very few clips, so 'no difference' is a real and expected answer."),
    },
}


def _items():
    if not ITEMS_JSON.is_file():
        raise HTTPException(503, f"no test at {ITEMS_JSON} — run render_ear_ab.py first")
    return json.loads(ITEMS_JSON.read_text())


def _read():
    if not VERDICTS.is_file():
        return {}
    with VERDICTS.open(newline="", encoding="utf-8") as f:
        return {r["item"]: r for r in csv.DictReader(f)}


def _write(rows):
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VERDICTS.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(VERDICTS)


def _append_history(row):
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


class Verdict(BaseModel):
    item: str
    choice: str          # "A" | "B" | "same"
    confidence: str = ""  # "sure" | "unsure"
    note: str = ""


@app.get("/api/items")
def items(set: str = "all"):
    data = _items()
    have = _read()
    out = []
    for it in data["items"]:
        if set not in ("all", it["set"]):
            continue
        v = have.get(it["id"], {})
        out.append(it | {"choice": v.get("choice", ""),
                         "confidence": v.get("confidence", ""),
                         "note": v.get("note", "")})
    # ⚠ THE TEST DESCRIBES ITSELF. A bench knows what question its sets ask; this app
    # does not, and hardcoding it here meant a new test rendered with different set
    # names showed the wrong tabs and the wrong question. SET_BRIEF is the fallback for
    # manifests written before 2026-08-30.
    return {"test": data.get("test", "?"),
            "sets": data.get("sets") or SET_BRIEF, "items": out}


@app.get("/audio")
def audio(clip: str):
    # ⚠ WHITELIST, not a path join. `clip` arrives from the browser; anything that is
    # not an opaque id named by items.json is refused, so a crafted value cannot walk
    # out of clips/ into the rest of the mount.
    allowed = {c for it in _items()["items"] for c in (it["A"], it["B"])}
    if clip not in allowed:
        raise HTTPException(404, "unknown clip")
    p = CLIPS / f"{clip}.wav"
    if not p.is_file():
        raise HTTPException(404, f"missing render: {clip}")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/verdict")
def verdict(v: Verdict):
    if v.choice not in ("A", "B", "same", ""):
        raise HTTPException(400, "choice must be A, B, same, or empty to clear")
    known = {it["id"]: it["set"] for it in _items()["items"]}
    if v.item not in known:
        raise HTTPException(404, f"unknown item {v.item}")
    row = {"item": v.item, "set": known[v.item], "choice": v.choice,
           "confidence": v.confidence, "note": v.note,
           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _LOCK:
        rows = _read()
        if v.choice == "":
            rows.pop(v.item, None)
        else:
            rows[v.item] = row
        _write(rows)
    _append_history(row)
    return {"ok": True}


@app.get("/api/progress")
def progress():
    # Counts only. NOT the A/B split — see the module docstring.
    done = _read()
    per = {}
    for it in _items()["items"]:
        s = per.setdefault(it["set"], {"total": 0, "done": 0})
        s["total"] += 1
        if done.get(it["id"], {}).get("choice"):
            s["done"] += 1
    return per


INDEX = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sonora ear test</title><style>
:root{color-scheme:dark}
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#12141a;color:#e6e8ee}
header{padding:14px 20px;border-bottom:1px solid #2a2f3a;display:flex;gap:14px;
  align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#12141a;z-index:5}
h1{font-size:16px;margin:0;font-weight:600}
.tab{padding:6px 12px;border:1px solid #2a2f3a;border-radius:6px;cursor:pointer;background:#191d26}
.tab.on{background:#2b4a7d;border-color:#3d6ab5}
main{max-width:820px;margin:0 auto;padding:22px 20px 80px}
.brief{background:#191d26;border-left:3px solid #3d6ab5;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:22px;color:#aab2c4}
.brief b{color:#e6e8ee}
.meta{color:#79839a;font-size:13px;margin-bottom:6px}
.text{font-size:19px;margin:10px 0 22px;line-height:1.45}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
button{font:inherit;color:inherit;background:#191d26;border:1px solid #2a2f3a;
  border-radius:8px;padding:14px;cursor:pointer}
button:hover{border-color:#3d6ab5}
.play{font-size:17px;font-weight:600;padding:22px}
.play.playing{background:#2b4a7d;border-color:#3d6ab5}
.choices{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px}
.choices button.sel{background:#2f6b3f;border-color:#4d9c63}
.conf{display:flex;gap:10px;margin-bottom:14px}
.conf button{flex:1;padding:9px;font-size:14px}
.conf button.sel{background:#6b552f;border-color:#9c854d}
textarea{width:100%;background:#191d26;border:1px solid #2a2f3a;border-radius:8px;
  color:inherit;font:inherit;padding:10px;box-sizing:border-box}
nav{display:flex;gap:10px;margin-top:16px}
nav button{flex:1;padding:11px}
.prog{color:#79839a;font-size:13px;margin-left:auto}
kbd{background:#232833;border:1px solid #2a2f3a;border-radius:4px;padding:1px 6px;font-size:12px}
.hint{color:#79839a;font-size:13px;margin-top:22px;line-height:1.9}
.done{color:#4d9c63}
</style></head><body>
<header><h1>Sonora ear test</h1>
  <div id="tabs" style="display:flex;gap:10px"></div>
  <div class="prog" id="prog"></div>
</header>
<main>
  <div class="brief" id="brief"></div>
  <div class="meta" id="meta"></div>
  <div class="text" id="text"></div>
  <div class="pair">
    <button class="play" id="pa">▶ A <kbd>1</kbd></button>
    <button class="play" id="pb">▶ B <kbd>2</kbd></button>
  </div>
  <div class="choices">
    <button data-c="A"></button><button data-c="same"></button><button data-c="B"></button>
  </div>
  <div class="conf">
    <button data-f="sure">Sure</button><button data-f="unsure">Not sure</button>
  </div>
  <textarea id="note" rows="2" placeholder="Optional: what did you hear?"></textarea>
  <nav><button id="prev">← Previous</button><button id="next">Next →</button></nav>
  <div class="hint">
    <kbd>1</kbd> play A · <kbd>2</kbd> play B · <kbd>a</kbd>/<kbd>s</kbd>/<kbd>b</kbd> choose ·
    <kbd>←</kbd>/<kbd>→</kbd> move · replay as often as you like.<br>
    Which clip is A changes from item to item, so the sides tell you nothing.
    The middle answer is a real answer — do not force a choice.
  </div>
</main>
<script>
let ALL=[],ITEMS=[],SETS={},i=0,SET=null,au=new Audio();
const $=id=>document.getElementById(id);
const DEFAULT_LABELS={A:"A is better",same:"No difference",B:"B is better"};
async function load(){
  const r=await(await fetch("/api/items")).json();
  ALL=r.items;SETS=r.sets;
  const names=Object.keys(SETS).filter(n=>ALL.some(x=>x.set===n));
  if(!SET||!names.includes(SET))SET=names[0];
  $("tabs").innerHTML=names.map((n,k)=>
    `<div class="tab${n===SET?" on":""}" data-set="${n}">Set ${n.replace(/^([A-Z])[_ ].*/,"$1")||k+1}</div>`).join("");
  document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{SET=t.dataset.set;load();});
  select();
}
function select(){
  ITEMS=ALL.filter(x=>x.set===SET); i=0;
  const f=ITEMS.findIndex(x=>!x.choice); if(f>=0)i=f;
  render();
}
function render(){
  const it=ITEMS[i]; if(!it)return;
  const L=SETS[SET].labels||DEFAULT_LABELS;
  document.querySelectorAll(".choices button").forEach(b=>{
    const k={a:"a",same:"s",b:"b"}[b.dataset.c.toLowerCase()];
    b.innerHTML=(L[b.dataset.c]||DEFAULT_LABELS[b.dataset.c])+` <kbd>${k}</kbd>`;});
  $("brief").innerHTML="<b>"+SETS[SET].title+"</b><br>"+SETS[SET].ask;
  $("meta").textContent=`item ${i+1} of ${ITEMS.length}  ·  speaker ${it.spk}  ·  V/A/T ${it.vat.join(", ")}  ·  delivery ${it.delivery_ui}`;
  $("text").textContent="“"+it.text+"”";
  document.querySelectorAll(".choices button").forEach(b=>
    b.classList.toggle("sel",b.dataset.c===it.choice));
  document.querySelectorAll(".conf button").forEach(b=>
    b.classList.toggle("sel",b.dataset.f===it.confidence));
  $("note").value=it.note||"";
  const d=ITEMS.filter(x=>x.choice).length;
  $("prog").innerHTML=`<span class="${d===ITEMS.length?'done':''}">${d} / ${ITEMS.length} judged</span>`;
}
function play(which){
  const it=ITEMS[i]; au.pause();
  au.src="/audio?clip="+(which==="A"?it.A:it.B); au.play();
  $("pa").classList.toggle("playing",which==="A");
  $("pb").classList.toggle("playing",which==="B");
}
au.onended=()=>{$("pa").classList.remove("playing");$("pb").classList.remove("playing")};
async function save(){
  const it=ITEMS[i];
  // ⚠ mutate in place; ALL and ITEMS share objects, so the tab counters stay right
  // without a refetch that would jump the listener back to the first unjudged item.
  await fetch("/api/verdict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({item:it.id,choice:it.choice||"",confidence:it.confidence||"",note:it.note||""})});
  render();
}
function choose(c){const it=ITEMS[i];it.choice=(it.choice===c?"":c);save();}
function conf(f){const it=ITEMS[i];it.confidence=(it.confidence===f?"":f);save();}
function move(d){au.pause();i=Math.max(0,Math.min(ITEMS.length-1,i+d));render();}
$("pa").onclick=()=>play("A"); $("pb").onclick=()=>play("B");
$("prev").onclick=()=>move(-1); $("next").onclick=()=>move(1);
document.querySelectorAll(".choices button").forEach(b=>b.onclick=()=>choose(b.dataset.c));
document.querySelectorAll(".conf button").forEach(b=>b.onclick=()=>conf(b.dataset.f));
$("note").onchange=()=>{ITEMS[i].note=$("note").value;save();};
document.onkeydown=e=>{
  if(e.target.tagName==="TEXTAREA")return;
  const k=e.key.toLowerCase();
  if(k==="1")play("A"); else if(k==="2")play("B");
  else if(k==="a")choose("A"); else if(k==="b")choose("B"); else if(k==="s")choose("same");
  else if(e.key==="ArrowLeft")move(-1); else if(e.key==="ArrowRight")move(1);
  else return; e.preventDefault();};
load();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX
