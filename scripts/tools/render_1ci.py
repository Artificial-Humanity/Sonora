"""Phase 1S Step 1c-i: the Chatterbox and Zonos clones of the two Qwen anchors, planned and launched.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c (Design 1c-i, Measures,
Criteria, Budget). Four cells, each anchor x each engine, 300 clips each:

    M4-s3_chatterbox   M4-s3_zonos   F6-s2_chatterbox   F6-s2_zonos

This tool writes the banks and launches the existing render path; it renders nothing itself.
`synth_bank.sh` runs the bank (check_bank, the engine's own renderer, loudness, the QC gate) in
probe mode (`SYNTH_BANK_PROBE=1`), so nothing reaches qc_verdict, EIV scoring, the defect
detectors or the audition queue.

    # host: verify the inputs and write the four campaign dirs (no GPU, no model)
    .venv/bin/python scripts/tools/render_1ci.py plan \\
        --out-root /data/model-training/datasets/probes/qwen-1ci

    # host: print the render command for one cell; `--run` executes it (GPU: a JARVIS slot)
    .venv/bin/python scripts/tools/render_1ci.py render M4-s3_chatterbox \\
        --out-root /data/model-training/datasets/probes/qwen-1ci

`plan` refuses unless the text bank's canonical sha256 (`probe_text_bank.canonical_sha256`) and its
`meta.json` both equal TEXT_SHA256 and it holds `BANK_SIZE` lines, and unless each anchor clip's
sha256 equals its pin in ANCHORS. It writes `<out-root>/<cell>/bank.json` and `plan_meta.json`:

* **One campaign dir per cell.** synth_bank.sh takes its QC campaign dir as `dirname "$OUT"` and
  qc_gate.py globs every `*_manifest.jsonl` under it, so OUT is `<out-root>/<cell>/audio` and a cell
  shares its gate with no other cell.
* **The bank** is the format check_bank.py and the renderers read: `version`, `campaign` and one
  line per text, in the text bank's order. Each line carries `render_index` (1-based; criterion C2
  compares clips 1-50 with 251-300), `text_id`, `seed`, `direction` and `ref_file`. It carries no
  `intended` V/A/T, so no EIV scoring can be triggered by it.
* **The reference** is the anchor clip, pinned by path in `ref_file` (`ref_select.
  pinned_reference_file`): the renderer checks its sha256 before every clip and applies no casting,
  F0 ceiling or damping.
* **Direction** is `book_ingest.build_direction` for each line with the tag in DIRECTION_TAGS and
  lane `Neutral`, and must come out the same for all 300 lines or the plan refuses. The tags hold
  nothing but the engine, so what the lines get is build_direction's defaults under the Neutral lane
  rules; `plan_meta.json` records the tag and the result.
* **Seeds:** line i (1-based render order) has seed `SEED_BASE + i` in every cell, so the same line
  is drawn from the same seed whichever voice and engine render it. The renderers call
  `torch.manual_seed(seed)` before each clip.
* **plan_meta.json** also records the commit and whether the tree was clean, the text bank's path
  and sha256, the anchor's path, sha256, text and gender, the seed rule, the engine pin
  (chatterbox: `SONORA_PIN_CHATTERBOX` from `scripts/container_env.sh`; zonos: the source checkout's
  commit, which synth_bank.sh puts on PYTHONPATH), what the throughput fields time, and BATCHED_RUN.

`plan` refuses to replace a cell's `bank.json` with a different one, and to re-plan a cell whose
`audio/` already holds a manifest at another commit: a resumed render would mix two plans.

`render` re-runs every check `plan` made, rebuilds the cell's bank and refuses unless it equals the
`bank.json` on disk, unless `plan_meta.json` names this checkout's HEAD, and unless the tree is clean
(`--allow-dirty` overrides the last and is recorded). It then prints the synth_bank.sh command, and
runs it only with `--run`, appending the command, commit and exit status to the cell's
`render_log.jsonl`. synth_bank.sh needs BANK and OUT under /data and the checkout's `.venv`, so
`--run` refuses without them; the printed command only warns. A run that stops resumes: the
renderers skip a clip whose wav exists.

⚠ THROUGHPUT IS LOGGED BY THE RENDERERS, NOT HERE. Every manifest row carries `render_seconds` (the
engine's generate call), `ref_condition_seconds` (the reference conditioning, timed separately),
`audio_seconds` and `rtf`, all from `time.monotonic()`; the model load is in neither. What each call
covers is in synth_chatterbox.py's and synth_zonos.py's docstrings.

⚠ THE BATCHED THROUGHPUT RUN IS NOT AVAILABLE (BATCHED_RUN). The pre-registered 100-line batched
Chatterbox run needs a batch API, and chatterbox-tts 0.1.7's `ChatterboxTTS.generate` takes one
text. `batched` prints that and exits non-zero; no custom batching is written here.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess

import sys as _sys  # noqa: E402
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import book_ingest  # noqa: E402
import probe_text_bank  # noqa: E402
import ref_select  # noqa: E402
from synth_common import append_jsonl, write_json_atomic  # noqa: E402

TOOL = "scripts/tools/render_1ci.py"
SPEC = ("Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c — pre-registration "
        "(Design 1c-i, Measures, Criteria, Budget)")

TEXT_BANK = "/data/model-training/datasets/probes/qwen-1ci-text/lines.jsonl"
TEXT_META = "/data/model-training/datasets/probes/qwen-1ci-text/meta.json"
TEXT_SHA256 = "4022c8e3b241ee15c4615ef1817dea45df8154e206c2f8f0f997d5a2b89ebee1"

# The locked anchors (§ 1c-0 verdicts), male first, each pinned to the sha256 of its clip as the
# owner approved it. Their text and gender come from the anchor folder's manifest.jsonl.
ANCHOR_DIR = "/data/model-training/datasets/probes/qwen-1c0-anchors"
ANCHORS = (
    ("M4-s3", "f3398bb813d784a2a8ae65daf7feb1827de013ca512fa24f1d40e6c966e4fa52"),
    ("F6-s2", "11977e770851d2acdb5516cb743fc2b0e1f73d23bac5ce603a4f64fcd5574209"),
)
ENGINES = ("chatterbox", "zonos")
LANE = "Neutral"
# The tag handed to build_direction for every line. Nothing but the engine: the values are
# build_direction's, under its Neutral lane rules (module docstring, Direction).
DIRECTION_TAGS = {"chatterbox": {}, "zonos": {}}
SEED_BASE = 1_000_000
SEED_RULE = "seed = SEED_BASE + render_index (1-based), the same for a line in every cell"
VERSION = "1ci-v1"

SYNTH_BANK = os.path.join(_SONORA_REPO, "scripts", "stages", "synth_bank.sh")
PROBE_ENV = {"SYNTH_BANK_PROBE": "1"}
CONTAINER_ENV = os.path.join(_SONORA_REPO, "scripts", "container_env.sh")
# synth_bank.sh's zonos block runs Zonos from this checkout (PYTHONPATH), not from a wheel.
ZONOS_SOURCE = "/data/toolchain/Zonos"
# synth_bank.sh refuses a BANK or OUT outside this tree, the one mounted into the containers,
# and runs its host steps (check_bank, loudness, the QC gate) with the checkout's venv.
DATA_ROOT = "/data/"
VENV_PYTHON = os.path.join(_SONORA_REPO, ".venv", "bin", "python")

BATCHED_RUN = ("not available: the pre-registered 100-line batched Chatterbox throughput run needs a "
               "batch API, and chatterbox-tts 0.1.7's ChatterboxTTS.generate takes one text. It is "
               "not run and no custom batching is written; throughput is batch size 1 only.")
TIMING = ("per manifest row, by time.monotonic(): render_seconds = the engine's generate call; "
          "ref_condition_seconds = the reference conditioning, timed separately; audio_seconds; "
          "rtf = render_seconds / audio_seconds; the model load is in neither. See "
          "synth_chatterbox.py and synth_zonos.py.")


def refuse(msg):
    raise SystemExit("REFUSING: " + msg)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_text():
    """The text bank's rows in bank order, after checking them against TEXT_SHA256."""
    bank, meta = TEXT_BANK, TEXT_META
    rows = _read_jsonl(bank)
    got = probe_text_bank.canonical_sha256(rows)
    if got != TEXT_SHA256:
        refuse("%s has canonical sha256 %s, not the registered %s." % (bank, got, TEXT_SHA256))
    with open(meta, encoding="utf-8") as fh:
        recorded = json.load(fh).get("sha256")
    if recorded != TEXT_SHA256:
        refuse("%s records sha256 %s, not the registered %s." % (meta, recorded, TEXT_SHA256))
    if len(rows) != probe_text_bank.BANK_SIZE:
        refuse("%s holds %d lines, not %d." % (bank, len(rows), probe_text_bank.BANK_SIZE))
    if len({r["id"] for r in rows}) != len(rows):
        refuse("%s repeats a line id." % bank)
    return rows


def load_anchors():
    """[{id, path, sha256, text, gender}] in ANCHORS order, each clip checked against its pin."""
    anchor_dir = ANCHOR_DIR
    by_id = {r["id"]: r for r in _read_jsonl(os.path.join(anchor_dir, "manifest.jsonl"))}
    anchors = []
    for anchor_id, pin in ANCHORS:
        row = by_id.get(anchor_id)
        if row is None:
            refuse("%s/manifest.jsonl has no row %s." % (anchor_dir, anchor_id))
        path = os.path.join(anchor_dir, row["file"])
        if not os.path.isfile(path):
            refuse("anchor clip %s does not exist." % path)
        got = ref_select.file_sha256(path)
        if got != pin:
            refuse("anchor clip %s has sha256 %s, not the pinned %s." % (path, got, pin))
        spec = {"path": path, "sha256": pin, "text": row["text"], "id": anchor_id,
                "gender": row["gender"]}
        problems = ref_select.ref_file_problems(spec)
        if problems:
            refuse("anchor %s cannot be pinned by file: %s." % (anchor_id, "; ".join(problems)))
        anchors.append(spec)
    return anchors


def cell_name(anchor_id, engine):
    return "%s_%s" % (anchor_id, engine)


def cell_names():
    return [cell_name(a, e) for a, _pin in ANCHORS for e in ENGINES]


def direction_for(engine, rows):
    """build_direction's payload for `engine` under LANE, refused unless every line gets the same."""
    tag = dict(DIRECTION_TAGS[engine], engine=engine)
    out = [book_ingest.build_direction(tag, r["text"], lane=LANE) for r in rows]
    if any(e != engine for e, _d in out):
        refuse("build_direction re-routed a %s line to another engine." % engine)
    first = out[0][1]
    if any(d != first for _e, d in out):
        refuse("build_direction gave the %s lines more than one direction." % engine)
    return first


def seed_for(render_index):
    return SEED_BASE + render_index


def build_bank(anchor, engine, rows, direction):
    cell = cell_name(anchor["id"], engine)
    lines = [{"id": "%s-%03d" % (cell, i), "engine": engine, "render_index": i,
              "text_id": r["id"], "text": r["text"], "seed": seed_for(i),
              "direction": dict(direction), "ref_file": dict(anchor)}
             for i, r in enumerate(rows, 1)]
    return {"version": VERSION, "campaign": "1ci-" + cell, "lines": lines}


def tool_commit():
    """{"commit": this checkout's HEAD, "tree_clean": `git status --porcelain` empty}, each None
    when git cannot say."""
    def git(*argv):
        try:
            return subprocess.run(["git", "-C", _SONORA_REPO, *argv], capture_output=True,
                                  text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": head.strip() if head else None,
            "tree_clean": None if status is None else not status.strip()}


def engine_pin(engine):
    """What fixes the engine's code: the wheel pin, or the source checkout's commit."""
    if engine == "chatterbox":
        with open(CONTAINER_ENV, encoding="utf-8") as fh:
            m = re.search(r'^SONORA_PIN_CHATTERBOX="([^"]+)"', fh.read(), re.M)
        if not m:
            refuse("%s has no SONORA_PIN_CHATTERBOX." % CONTAINER_ENV)
        return {"wheel": m.group(1), "source": "scripts/container_env.sh SONORA_PIN_CHATTERBOX"}
    try:
        commit = subprocess.run(["git", "-C", ZONOS_SOURCE, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ZONOS_SOURCE, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"source_checkout": ZONOS_SOURCE, "commit": commit,
            "tree_clean": None if dirty is None else not dirty}


def plan_meta(anchor, engine, direction, git, n_lines):
    return {
        "tool": TOOL, "spec": SPEC, "commit": git["commit"], "tree_clean": git["tree_clean"],
        "planned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cell": cell_name(anchor["id"], engine), "engine": engine, "anchor": dict(anchor),
        "text": {"bank": TEXT_BANK, "meta": TEXT_META, "sha256": TEXT_SHA256, "lines": n_lines,
                 "canonical_form": probe_text_bank.CANONICAL_FORM},
        "render_order": "the text bank's line order; render_index is 1-based",
        "direction": {"lane": LANE, "tag": dict(DIRECTION_TAGS[engine], engine=engine),
                      "via": "book_ingest.build_direction", "direction": direction},
        "seeds": {"rule": SEED_RULE, "SEED_BASE": SEED_BASE, "first": seed_for(1),
                  "last": seed_for(n_lines)},
        "engine_pin": engine_pin(engine),
        "synth_bank": {"env": PROBE_ENV, "out": "audio"},
        "throughput": TIMING, "batched_run": BATCHED_RUN,
    }


def _has_manifest(audio):
    return os.path.isdir(audio) and any(n.endswith("_manifest.jsonl") for n in os.listdir(audio))


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _require_clean(git, allow_dirty):
    if git["commit"] is None:
        refuse("git cannot name this checkout's HEAD.")
    if not git["tree_clean"] and not allow_dirty:
        refuse("%s has uncommitted changes or untracked files (pass --allow-dirty to "
               "override; it is recorded)." % _SONORA_REPO)


def planned_cells():
    """{cell: (anchor, engine, bank)} built from the verified inputs. Every plan check is here."""
    rows, anchors = load_text(), load_anchors()
    directions = {e: direction_for(e, rows) for e in ENGINES}
    return {cell_name(a["id"], e): (a, e, build_bank(a, e, rows, directions[e]))
            for a in anchors for e in ENGINES}, directions


def cmd_plan(args):
    git = tool_commit()
    _require_clean(git, args.allow_dirty)
    out_root = os.path.abspath(args.out_root)
    cells, directions = planned_cells()
    # Every refusal before the first write, so a refused plan leaves nothing half-written.
    for cell, (_a, _e, bank) in cells.items():
        cdir = os.path.join(out_root, cell)
        if os.path.exists(os.path.join(cdir, "bank.json")) and \
                _load_json(os.path.join(cdir, "bank.json")) != bank:
            refuse("%s/bank.json differs from this plan. Plan into a new --out-root." % cdir)
        meta = os.path.join(cdir, "plan_meta.json")
        if _has_manifest(os.path.join(cdir, "audio")) and \
                (not os.path.exists(meta) or _load_json(meta).get("commit") != git["commit"]):
            refuse("%s/audio holds renders planned at another commit; re-planning would mix "
                   "them. Plan into a new --out-root." % cdir)
    for cell, (anchor, engine, bank) in cells.items():
        cdir = os.path.join(out_root, cell)
        os.makedirs(cdir, exist_ok=True)
        write_json_atomic(os.path.join(cdir, "bank.json"), bank)
        write_json_atomic(os.path.join(cdir, "plan_meta.json"),
                          plan_meta(anchor, engine, directions[engine], git, len(bank["lines"])))
        print("%-17s %d lines  ref %s  seeds %d-%d -> %s" % (
            cell, len(bank["lines"]), anchor["id"], bank["lines"][0]["seed"],
            bank["lines"][-1]["seed"], cdir))
    for engine in ENGINES:
        print("direction %-10s %s" % (engine, json.dumps(directions[engine], sort_keys=True)))
    print("commit %s%s" % (git["commit"], "" if git["tree_clean"] else " (DIRTY TREE)"))
    print("batched run: " + BATCHED_RUN)


def render_command(out_root, cell, engine):
    cdir = os.path.join(out_root, cell)
    return [SYNTH_BANK, os.path.join(cdir, "bank.json"), os.path.join(cdir, "audio"), engine]


def _shell_line(argv):
    return " ".join("%s=%s" % kv for kv in PROBE_ENV.items()) + " " + shlex.join(argv)


def cmd_render(args):
    git = tool_commit()
    _require_clean(git, args.allow_dirty)
    out_root = os.path.abspath(args.out_root)
    if args.cell not in cell_names():
        refuse("no cell %r; the cells are %s." % (args.cell, ", ".join(cell_names())))
    cells, _directions = planned_cells()
    _anchor, engine, bank = cells[args.cell]
    cdir = os.path.join(out_root, args.cell)
    try:
        on_disk, meta = _load_json(os.path.join(cdir, "bank.json")), \
            _load_json(os.path.join(cdir, "plan_meta.json"))
    except (OSError, ValueError) as e:
        refuse("%s has no readable plan (%s). Run `plan` first." % (cdir, e))
    if on_disk != bank:
        refuse("%s/bank.json is not the bank these inputs and this code plan. Re-plan." % cdir)
    if meta.get("commit") != git["commit"]:
        refuse("%s was planned at %s, and this checkout is %s. Re-plan at this commit."
               % (cdir, meta.get("commit"), git["commit"]))
    argv = render_command(out_root, args.cell, engine)
    line = _shell_line(argv)
    blockers = []
    if not (out_root + "/").startswith(DATA_ROOT) or re.search(r"[\s'\"]", out_root):
        blockers.append("synth_bank.sh needs BANK and OUT under %s, without spaces or quotes; "
                        "%s is not." % (DATA_ROOT, out_root))
    if not os.access(VENV_PYTHON, os.X_OK):
        blockers.append("synth_bank.sh runs check_bank, loudness and the QC gate with %s, which "
                        "does not exist." % VENV_PYTHON)
    print(line)
    for b in blockers:
        print("WARNING: " + b)
    if not args.run:
        print("(printed only; --run executes it. The GPU is booked separately.)")
        return 0
    if blockers:
        refuse("--run with the blockers above.")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    code = subprocess.run(argv, env=dict(os.environ, **PROBE_ENV)).returncode
    append_jsonl(os.path.join(cdir, "render_log.jsonl"), [{
        "command": line, "commit": git["commit"], "tree_clean": git["tree_clean"],
        "allow_dirty": bool(args.allow_dirty), "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "exit_code": code}])
    return code


def cmd_batched(_args):
    print("batched run: " + BATCHED_RUN)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="verify the inputs and write the four campaign dirs")
    p.add_argument("--out-root", required=True)
    p.add_argument("--allow-dirty", action="store_true")
    r = sub.add_parser("render", help="print (or with --run, execute) one cell's synth_bank.sh")
    r.add_argument("cell", help="one of: " + ", ".join(cell_names()))
    r.add_argument("--out-root", required=True)
    r.add_argument("--run", action="store_true", help="execute it (GPU); default: print only")
    r.add_argument("--allow-dirty", action="store_true")
    sub.add_parser("batched", help="the pre-registered batched run (not available; says why)")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "render": cmd_render, "batched": cmd_batched}[args.cmd](args)


if __name__ == "__main__":
    _sys.exit(main())
