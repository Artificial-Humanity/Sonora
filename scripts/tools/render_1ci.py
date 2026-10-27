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
  lane `Neutral`, and must come out the same for all 300 lines, with every value the tag sets kept
  exactly (no clamp may move it), or the plan refuses. Chatterbox's tag holds nothing, so it gets
  build_direction's defaults under the Neutral lane rules (exaggeration 0.4, cfg_weight 0.5). Zonos's
  tag holds the lead resident's decision (DIRECTION_TAGS, ZONOS_DIRECTION_DECISION): pitch_std 28,
  speaking_rate 14, emotion off. `plan_meta.json` records the tag, the decision and the result.
* **Seeds:** line i (1-based render order) has seed `SEED_BASE + i` in every cell, so the same line
  is drawn from the same seed whichever voice and engine render it. The renderers call
  `torch.manual_seed(seed)` before each clip.
* **plan_meta.json** also records the commit and whether the tree was clean, the text bank's path
  and sha256, the anchor's path, sha256, text and gender, the seed rule, the engine pin
  (chatterbox: `SONORA_PIN_CHATTERBOX` from `scripts/container_env.sh`; zonos: the source checkout's
  commit, which synth_bank.sh puts on PYTHONPATH), what the throughput fields time, and BATCHED_RUN.

`plan` refuses to replace a cell's `bank.json` with a different one, and to re-plan a cell whose
`audio/` already holds a manifest at another commit: a resumed render would mix two plans. Such a
cell is resumed with `render` against its existing plan instead (below).

`render` re-runs every check `plan` made, refuses unless no file in RENDER_AFFECTING_PATHS differs
between the commit `plan_meta.json` names and this checkout's HEAD (`git diff --quiet`; a plan
commit git cannot compare with HEAD is refused too), refuses unless the engine pin it derives now
(`engine_pin`, the same way `plan` did) equals `plan_meta.engine_pin`, rebuilds the cell's bank and
refuses unless it equals the `bank.json` on disk, and refuses unless the tree is clean
(`--allow-dirty` overrides the last and is recorded). A HEAD that moved on only outside those files
(a pull of unrelated work into the shared checkout between cells) is accepted, so a multi-day
campaign is not stopped by it. What this guarantees: the bank, the render-affecting files and the
engine pin are checked once, when the cell is launched.

⚠ THE GATE RUNS ONLY AT LAUNCH. synth_bank.sh and the renderers read their code from this checkout
while the cell runs, and nothing re-checks it. A pull into the shared checkout, or a change to the
Zonos source checkout, during a running cell is NOT detected, and the clips after it come from the
new code. Do not pull or change either checkout while a cell is rendering.

`render` then prints the synth_bank.sh command, and runs it only with `--run`, appending the
command, both commits, the exit status and the cell's rendered/expected/missing counts to the
cell's `render_log.jsonl`. synth_bank.sh needs BANK and OUT under /data and the checkout's `.venv`,
so `--run` refuses without them; the printed command only warns.

⚠ A PARTLY FAILED CELL. A renderer whose reference fails for one clip (CASTING FAILED) prints it,
writes no manifest row for that clip and carries on, so synth_bank.sh can exit 0 with clips
missing; only a cell where every clip fails is stopped by the QC gate. After synth_bank.sh returns,
`render --run` therefore counts the unique bank ids in `<cell>/audio/<engine>_manifest.jsonl`
against the bank, and exits non-zero (PARTIAL_EXIT when synth_bank.sh itself exited 0), naming the
missing ids, when any are missing. Re-running `render --run` resumes: the renderers skip a clip
whose wav exists and render only the rest. `status <cell>` reports the same counts without
launching anything, and `render` without `--run` prints them when the cell has a manifest.

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
# The tag handed to build_direction for every line (with the engine added), under LANE. Every
# value set here must come out of build_direction unchanged, or the plan refuses.
#   chatterbox: nothing, so build_direction's defaults under the Neutral lane rules apply
#     (exaggeration 0.5 capped to NARRATION_MAX_EXAGGERATION 0.4, cfg_weight 0.5).
#   zonos: ZONOS_DIRECTION_DECISION. build_direction's own defaults (pitch_std 45, speaking_rate
#     15) sit outside zonos.md's Neutral row for pitch_std (20-35), so they are not used.
DIRECTION_TAGS = {"chatterbox": {},
                  "zonos": {"pitch_std": 28.0, "speaking_rate": 14.0, "emotion": None}}
ZONOS_DIRECTION_DECISION = (
    "pitch_std 28.0, speaking_rate 14.0, emotion None (off) on every Zonos line of the 1,200-line "
    "run (both Zonos cells): the Sonora lead resident's decision for Step 1c-i (2026-10-27). It is the most-used audited Neutral narration "
    "setting in delivery-v1-narration and delivery-v1-narration-r2, and it sits inside "
    "scripts/assets/director_skills/zonos.md's Neutral row (pitch_std 20-35, speaking_rate 14-15, "
    "emotion omitted).")
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

# Every tracked file whose content decides what `render` launches or what a cell's render
# produces. `render` accepts a HEAD other than the plan commit only when none of these differ
# between the two (`git diff --quiet <plan> HEAD -- <paths>`): the renders then come from the same
# code as the plan, whatever else the shared checkout has pulled. The set is:
#   * the launch path: synth_bank.sh, the pins it sources, and the two in-container steps run
#     before each renderer (the ai-mgr user setup and the environment freeze);
#   * the steps synth_bank.sh runs in probe mode: check_bank, the two engines' renderers, the
#     loudness normaliser and the QC gate (which decides which clips are kept);
#   * this tool and the modules its plan is built from (the bank, the direction, the text check);
#   * every repo module those import at module level (matcha.delivery: the lane rules
#     build_direction and ref_select apply; schemas; qc_gate's phonation and VAD measures).
# test_render_1ci.py checks it: every listed path is tracked, every repo module imported at
# module level by a listed file is listed or in NOT_RENDER_AFFECTING, and every script
# synth_bank.sh or container_env.sh names is listed or in NOT_RENDER_AFFECTING.
RENDER_AFFECTING_PATHS = (
    "scripts/stages/synth_bank.sh",
    "scripts/container_env.sh",
    "scripts/container_as_ai_mgr.sh",
    "scripts/capture_container_env.sh",
    "scripts/stages/check_bank.py",
    "scripts/stages/synth_chatterbox.py",
    "scripts/stages/synth_zonos.py",
    "scripts/stages/normalize_loudness.py",
    "scripts/stages/qc_gate.py",
    "scripts/tools/render_1ci.py",
    "scripts/lib/book_ingest.py",
    "scripts/lib/probe_text_bank.py",
    "scripts/lib/ref_select.py",
    "scripts/lib/synth_common.py",
    "scripts/lib/schemas.py",
    "scripts/lib/derive_vat_corpus.py",
    "scripts/lib/silero_vad.py",
    "matcha/__init__.py",
    "matcha/delivery.py",
)
# Reached from the files above but not part of a 1c-i render, each with the reason.
NOT_RENDER_AFFECTING = {
    "scripts/lib/gemma_client.py": "book_ingest imports it for the director pass; plan never calls it",
    "scripts/lib/gemma_server.py": "book_ingest imports it for the director pass; plan never calls it",
    "scripts/stages/qc_verdict.py": "probe mode skips the direction check",
    "scripts/stages/eiv_score.sh": "probe mode skips the direction check",
    "scripts/stages/qc_engine_defects.py": "probe mode skips the defect detectors (PROBE_MODE_GAP)",
    "scripts/stages/register_audition.py": "probe mode skips the audition queue",
    "scripts/stages/librivox_align.sh": "another orchestrator; container_env.sh's header names it",
    "scripts/stages/synth_dia.py": "another engine's block",
    "scripts/stages/synth_moss_vg.py": "another engine's block",
    "scripts/stages/synth_moss85.py": "another engine's block",
    "scripts/stages/synth_qwen.py": "another engine's block",
    "scripts/stages/synth_qwen_base.py": "another engine's block (Step 1c-ii's engine, not 1c-i's)",
    "scripts/stages/synth_qwen_custom.py": "another engine's block (Step 1c-ii's engine, not 1c-i's)",
    "scripts/stages/synth_orpheus.py": "another engine's block",
    "scripts/stages/synth_vibevoice.py": "another engine's block",
}

# KNOWN GAP, documented and not closed here. Probe mode skips qc_engine_defects.py, which reads
# the campaign's rows in ratings.csv (a probe never enters it), so Zonos's leading-gap detector
# ("a blip, then silence") does not run on any probe cell.
PROBE_MODE_GAP = (
    "probe mode skips scripts/stages/qc_engine_defects.py (it needs the campaign's ratings.csv "
    "rows), so the Zonos leading-gap check ('blip then silence', qc_engine_defects."
    "leading_gap_score) does NOT run on these cells; the 1c-i evaluator must run it on the "
    "Zonos cells itself.")

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
        try:
            got = ref_select.file_sha256(path)
        except OSError as e:
            refuse("anchor clip %s cannot be read: %s." % (path, e))
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
    """build_direction's payload for `engine` under LANE, refused unless every line gets the same
    and every value DIRECTION_TAGS sets comes out unchanged (a clamp that moved one would render
    something other than what was decided)."""
    tag = dict(DIRECTION_TAGS[engine], engine=engine)
    out = [book_ingest.build_direction(tag, r["text"], lane=LANE) for r in rows]
    if any(e != engine for e, _d in out):
        refuse("build_direction re-routed a %s line to another engine." % engine)
    first = out[0][1]
    if any(d != first for _e, d in out):
        refuse("build_direction gave the %s lines more than one direction." % engine)
    moved = {k: (v, first.get(k)) for k, v in DIRECTION_TAGS[engine].items()
             if k not in first or first[k] != v or type(first[k]) is not type(v)}
    if moved:
        refuse("build_direction changed the %s direction DIRECTION_TAGS sets: %s." % (
            engine, ", ".join("%s %r -> %r" % (k, a, b) for k, (a, b) in sorted(moved.items()))))
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
        # An unreadable checkout would record commit None at plan AND at render, and the
        # launch check compares values, so None == None would pass: refuse instead.
        refuse("cannot read the Zonos source checkout %s (git rev-parse/status failed), so "
               "the engine code this cell renders with cannot be recorded." % ZONOS_SOURCE)
    if dirty:
        refuse("the Zonos source checkout %s has uncommitted changes, so its commit does not "
               "name the engine code that would run. Clean it first." % ZONOS_SOURCE)
    return {"source_checkout": ZONOS_SOURCE, "commit": commit, "tree_clean": True}


def plan_meta(anchor, engine, direction, git, n_lines):
    return {
        "tool": TOOL, "spec": SPEC, "commit": git["commit"], "tree_clean": git["tree_clean"],
        "planned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cell": cell_name(anchor["id"], engine), "engine": engine, "anchor": dict(anchor),
        "text": {"bank": TEXT_BANK, "meta": TEXT_META, "sha256": TEXT_SHA256, "lines": n_lines,
                 "canonical_form": probe_text_bank.CANONICAL_FORM},
        "render_order": "the text bank's line order; render_index is 1-based",
        "direction": {"lane": LANE, "tag": dict(DIRECTION_TAGS[engine], engine=engine),
                      "via": "book_ingest.build_direction", "direction": direction,
                      "decision": (ZONOS_DIRECTION_DECISION if engine == "zonos" else
                                   "build_direction's defaults under the Neutral lane rules")},
        "seeds": {"rule": SEED_RULE, "SEED_BASE": SEED_BASE, "first": seed_for(1),
                  "last": seed_for(n_lines)},
        "engine_pin": engine_pin(engine),
        "synth_bank": {"env": PROBE_ENV, "out": "audio", "known_gap": PROBE_MODE_GAP},
        "render_affecting_paths": list(RENDER_AFFECTING_PATHS),
        "throughput": TIMING, "batched_run": BATCHED_RUN,
    }


def _has_manifest(audio):
    return os.path.isdir(audio) and any(n.endswith("_manifest.jsonl") for n in os.listdir(audio))


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git(*argv):
    return subprocess.run(["git", "-C", _SONORA_REPO, *argv], capture_output=True, text=True)


def render_inputs_changed(plan_commit, head):
    """The RENDER_AFFECTING_PATHS that differ between `plan_commit` and `head`, [] when none do.

    Refuses when git cannot compare the two (a commit this repository does not have) or when a
    listed path is not tracked at `head`: `git diff` passes a pathspec that matches nothing, so a
    misspelt entry would otherwise vouch for a file it never compared."""
    if not isinstance(plan_commit, str) or not plan_commit:
        refuse("plan_meta.json names no plan commit (%r). Plan into a new --out-root."
               % (plan_commit,))
    if plan_commit == head:
        return []
    paths = list(RENDER_AFFECTING_PATHS)
    tracked = _git("ls-tree", "-r", "--name-only", head, "--", *paths)
    if tracked.returncode != 0:
        refuse("git cannot list %s's tree: %s" % (head, tracked.stderr.strip()))
    missing = sorted(set(paths) - set(tracked.stdout.split()))
    if missing:
        refuse("RENDER_AFFECTING_PATHS names files %s does not track: %s." % (
            head, ", ".join(missing)))
    quiet = _git("diff", "--quiet", plan_commit, head, "--", *paths)
    if quiet.returncode == 0:
        return []
    if quiet.returncode != 1:
        refuse("git cannot compare the plan commit %s with this checkout's %s (%s). Render from "
               "a checkout that has the plan commit." % (plan_commit, head, quiet.stderr.strip()))
    changed = _git("diff", "--name-only", plan_commit, head, "--", *paths).stdout.split()
    # A disagreement between the two calls is refused, never read as "unchanged".
    if not changed:
        refuse("git diff --quiet says the render-affecting files differ between %s and %s, and "
               "--name-only names none of them." % (plan_commit, head))
    return changed


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
            refuse("%s/audio holds renders planned at another commit (%s; this checkout is %s); "
                   "re-planning would mix two plans. To resume those renders, run `render %s` "
                   "against the existing plan: it accepts this checkout when no file in "
                   "RENDER_AFFECTING_PATHS differs from the plan commit, and otherwise names the "
                   "files that do. To start over at this commit, plan into a new --out-root."
                   % (cdir, _load_json(meta).get("commit") if os.path.exists(meta) else
                      "no plan_meta.json", git["commit"], cell))
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


# `render --run` and `status` exit with this when the cell's manifest lacks a row for some bank id
# and synth_bank.sh did not already fail (a synth_bank.sh failure keeps its own exit status).
PARTIAL_EXIT = 3
MISSING_SHOWN = 10          # missing ids printed by name; the rest are counted


def manifest_status(cdir, engine, bank):
    """How much of `bank` the cell's `<engine>_manifest.jsonl` holds.

    {"manifest", "expected", "rendered", "missing", "wav_without_row", "unreadable_lines"}:
    `rendered` counts the unique bank ids with a row, `missing` lists the others in bank order, and
    `wav_without_row` the missing ids whose wav exists anyway (a renderer skips an existing wav, so
    a re-run never gives those a row). A line that does not parse (a write cut short) is counted in
    `unreadable_lines` and gives no id."""
    audio = os.path.join(cdir, "audio")
    path = os.path.join(audio, "%s_manifest.jsonl" % engine)
    expected = [ln["id"] for ln in bank["lines"]]
    got, unreadable = set(), 0
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    unreadable += 1
                    continue
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    got.add(row["id"])
    missing = [i for i in expected if i not in got]
    return {"manifest": path, "expected": len(expected), "rendered": len(expected) - len(missing),
            "missing": missing,
            "wav_without_row": [i for i in missing
                                if os.path.exists(os.path.join(audio, i + ".wav"))],
            "unreadable_lines": unreadable}


def _some(ids):
    shown = ", ".join(ids[:MISSING_SHOWN])
    return shown + (" and %d more" % (len(ids) - MISSING_SHOWN) if len(ids) > MISSING_SHOWN else "")


def report_status(cell, status, synth_bank_code=None):
    """Print the cell's rendered/expected line, and for a short cell what is missing and the fix.
    Returns 0 for a complete cell, else PARTIAL_EXIT."""
    print("%s: %d of %d bank ids have a manifest row (%s)" % (
        cell, status["rendered"], status["expected"], status["manifest"]))
    if status["unreadable_lines"]:
        print("WARNING: %d line(s) of %s do not parse and were not counted."
              % (status["unreadable_lines"], status["manifest"]))
    if not status["missing"]:
        return 0
    lines = ["PARTIAL RENDER: %s is missing %d of %d clips: %s." % (
        cell, len(status["missing"]), status["expected"], _some(status["missing"]))]
    if synth_bank_code == 0:
        lines.append("synth_bank.sh exited 0, but a renderer skips a clip whose reference fails "
                     "(CASTING FAILED in its output) and carries on, so the cell is incomplete.")
    if status["wav_without_row"]:
        lines.append("%d missing id(s) have a wav but no manifest row (%s): a re-run skips an "
                     "existing wav, so move those wavs into <cell>/audio/_superseded/ first "
                     "(normalize_loudness skips that folder; any other subfolder of audio/ "
                     "would be gain-adjusted again)."
                     % (len(status["wav_without_row"]), _some(status["wav_without_row"])))
    lines.append("Re-run `render %s --run` to resume: the renderers skip every clip whose wav "
                 "exists and render only the missing ones." % cell)
    print("\n".join(lines), file=_sys.stderr)
    return PARTIAL_EXIT


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
    plan_commit = meta.get("commit")
    changed = render_inputs_changed(plan_commit, git["commit"])
    if changed:
        refuse("%s was planned at %s, and this checkout is %s; these render-affecting files "
               "differ between the two: %s. Render from a checkout at the plan commit (`git worktree "
               "add --detach <dir> %s`; it needs its own .venv), or plan into a new --out-root "
               "at this commit."
               % (cdir, plan_commit, git["commit"], ", ".join(changed), plan_commit))
    if plan_commit != git["commit"]:
        print("HEAD %s is not the plan commit %s; no render-affecting file differs between them."
              % (git["commit"], plan_commit))
    pin_now = engine_pin(engine)
    if meta.get("engine_pin") != pin_now:
        refuse("%s was planned with the %s engine pin %s, and it is now %s; the engine code is "
               "not the code the plan recorded. Restore the planned pin, or plan into a new "
               "--out-root with this one."
               % (cdir, engine, json.dumps(meta.get("engine_pin"), sort_keys=True),
                  json.dumps(pin_now, sort_keys=True)))
    if on_disk != bank:
        refuse("%s/bank.json is not the bank these inputs and this code plan. Re-plan." % cdir)
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
        if _has_manifest(os.path.join(cdir, "audio")):
            report_status(args.cell, manifest_status(cdir, engine, bank))
        print("(printed only; --run executes it. The GPU is booked separately.)")
        return 0
    if blockers:
        refuse("--run with the blockers above.")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    code = subprocess.run(argv, env=dict(os.environ, **PROBE_ENV)).returncode
    status = manifest_status(cdir, engine, bank)
    append_jsonl(os.path.join(cdir, "render_log.jsonl"), [{
        "command": line, "plan_commit": plan_commit, "head_commit": git["commit"],
        "tree_clean": git["tree_clean"],
        "allow_dirty": bool(args.allow_dirty), "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "exit_code": code, "rendered": status["rendered"], "expected": status["expected"],
        "missing": status["missing"], "wav_without_row": status["wav_without_row"]}])
    if code != 0:
        print("synth_bank.sh exited %d." % code, file=_sys.stderr)
    partial = report_status(args.cell, status, synth_bank_code=code)
    return code if code != 0 else partial


def cmd_status(args):
    """The cell's rendered/expected counts from its bank.json and manifest; launches nothing and
    re-checks no plan input (`render` does that)."""
    if args.cell not in cell_names():
        refuse("no cell %r; the cells are %s." % (args.cell, ", ".join(cell_names())))
    cdir = os.path.join(os.path.abspath(args.out_root), args.cell)
    try:
        bank, meta = _load_json(os.path.join(cdir, "bank.json")), \
            _load_json(os.path.join(cdir, "plan_meta.json"))
    except (OSError, ValueError) as e:
        refuse("%s has no readable plan (%s). Run `plan` first." % (cdir, e))
    return report_status(args.cell, manifest_status(cdir, meta.get("engine"), bank))


def cmd_batched(_args):
    print("batched run: " + BATCHED_RUN)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="verify the inputs and write the four campaign dirs")
    p.add_argument("--out-root", required=True)
    p.add_argument("--allow-dirty", action="store_true")
    r = sub.add_parser(
        "render", help="print (or with --run, execute) one cell's synth_bank.sh",
        description="The checks (plan inputs, render-affecting files, engine pin, bank) run once, "
                    "at launch: a pull into the shared checkout or a change to the Zonos source "
                    "while the cell runs is not detected, so do not pull mid-cell. --run exits "
                    "%d when synth_bank.sh exits 0 but bank ids have no manifest row; re-running "
                    "resumes." % PARTIAL_EXIT)
    r.add_argument("cell", help="one of: " + ", ".join(cell_names()))
    r.add_argument("--out-root", required=True)
    r.add_argument("--run", action="store_true", help="execute it (GPU); default: print only")
    r.add_argument("--allow-dirty", action="store_true")
    s = sub.add_parser("status", help="one cell's rendered/expected counts; launches nothing")
    s.add_argument("cell", help="one of: " + ", ".join(cell_names()))
    s.add_argument("--out-root", required=True)
    sub.add_parser("batched", help="the pre-registered batched run (not available; says why)")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "render": cmd_render, "status": cmd_status,
            "batched": cmd_batched}[args.cmd](args)


if __name__ == "__main__":
    _sys.exit(main())
