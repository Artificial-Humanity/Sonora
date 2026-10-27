"""Phase 1S Step 1c-ii: Qwen Base clones, CustomVoice presets and Orpheus jess, planned and launched.

Pre-registered in Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c (Design 1c-ii,
Measures, Criteria, Budget). Five measured cells of 300 clips on the same pinned 300-line text
bank as Step 1c-i, plus one batched throughput cell:

    M4-s3_qwen_base   F6-s2_qwen_base     (the locked 1c-0 anchors, cloned)
    ryan_qwen_custom  aiden_qwen_custom   (CustomVoice's only English-native presets)
    jess_orpheus                          (already onboarded; deferred from 1c-i for GPU budget)
    M4-s3_qwen_base_batched               (first 100 lines, throughput only, via `batched`)

This tool writes the banks and launches the existing render path; it renders nothing itself.
`synth_bank.sh` runs each bank in probe mode (`SYNTH_BANK_PROBE=1`), exactly as in 1c-i, so
nothing reaches qc_verdict, EIV scoring, the defect detectors or the audition queue.

    # host: verify the inputs and write the five campaign dirs (no GPU, no model)
    .venv/bin/python scripts/tools/render_1cii.py plan \
        --out-root /data/model-training/datasets/probes/qwen-1cii

    # host: print the render command for one cell; `--run` executes it (GPU: a JARVIS slot)
    .venv/bin/python scripts/tools/render_1cii.py render M4-s3_qwen_base \
        --out-root /data/model-training/datasets/probes/qwen-1cii

    # host: plan the batched throughput cell (batch size is the operator's measured
    # largest-that-fits; render it with `render M4-s3_qwen_base_batched --run`)
    .venv/bin/python scripts/tools/render_1cii.py batched --batch-size 8 \
        --out-root /data/model-training/datasets/probes/qwen-1cii

Shared machinery (the text bank check, the anchor pins, manifest accounting, the launch
refusals) is imported from render_1ci rather than copied: the 1c-i tool is frozen while its
cells render, and one definition of "the same pinned inputs" is the point. What is NOT shared
is anything cell-shaped — cells, direction tags, engine pins and the render-affecting list are
this tool's own, because the engines differ.

Direction is `book_ingest.build_direction` under lane `Neutral` with a fixed tag per cell, and
must come out exactly as the tag states (per line for Orpheus, whose render_text is the line
itself), or the plan refuses. The qwen_base cells pin their anchor by `ref_file`, so no casting
runs; the qwen_custom cells carry one fixed neutral delivery clause; the orpheus cell is jess
with the plain line, no inline tags. Seeds follow 1c-i's rule (`SEED_BASE + render_index`), so
the same line is drawn from the same seed in every cell of both steps.

`render` re-runs every check `plan` made and refuses unless no file in this tool's
RENDER_AFFECTING_PATHS differs from the plan commit, the derived engine pin still equals the
recorded one, and the rebuilt bank equals the bank on disk — the same launch gate as 1c-i,
including its limit: THE GATE RUNS ONLY AT LAUNCH. A pull into the shared checkout during a
running cell is not detected. Do not pull mid-cell.

⚠ THROUGHPUT IS LOGGED BY THE RENDERERS, NOT HERE. Every manifest row carries
`render_seconds`, `ref_condition_seconds`, `audio_seconds` and `rtf` from
`synth_common.render_timing`; what each covers is in each renderer's docstring.

⚠ THE BATCHED CELL IS THROUGHPUT ONLY. `generate_voice_clone` accepts a list of texts and
broadcasts a one-item prompt across them (CODE-VERIFIED, qwen-tts 0.1.1), but a batched call
draws every clip from one RNG stream, so per-clip seeds do not hold and its clips can never be
measured as a cell. Its speed-up is read against M4-s3_qwen_base's batch-size-1 rows.
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
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b)
                            for _b in ("lib", "tools"))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import book_ingest  # noqa: E402
import probe_text_bank  # noqa: E402
import ref_select  # noqa: E402
import render_1ci as r1  # noqa: E402
from synth_common import append_jsonl, write_json_atomic  # noqa: E402

TOOL = "scripts/tools/render_1cii.py"
SPEC = ("Notes: Sonora/phase-1s-step1-preregistration.md § Step 1c — pre-registration "
        "(Design 1c-ii, Measures, Criteria, Budget) and § Onboarding checklist")

LANE = r1.LANE
SEED_BASE = r1.SEED_BASE
SEED_RULE = r1.SEED_RULE
VERSION = "1cii-v1"

# The one fixed delivery clause for both CustomVoice cells: neutral narration,
# in the short one-clause shape director_skills/qwen_custom.md mandates (the
# 12-key VoiceDesign format is unverified on that checkpoint and is not sent).
CUSTOM_INSTRUCT = "Calm and measured."

# The measured cells. `anchor` names a locked 1c-0 anchor (render_1ci.ANCHORS)
# pinned by ref_file; the preset and closed-voice cells have no reference.
# `tag` is handed to build_direction for every line (with the engine added):
# every value set here must come out unchanged, or the plan refuses.
CELLS = (
    {"name": "M4-s3_qwen_base", "engine": "qwen_base", "anchor": "M4-s3", "tag": {}},
    {"name": "F6-s2_qwen_base", "engine": "qwen_base", "anchor": "F6-s2", "tag": {}},
    {"name": "ryan_qwen_custom", "engine": "qwen_custom", "anchor": None,
     "tag": {"speaker": "ryan", "instruct": CUSTOM_INSTRUCT}},
    {"name": "aiden_qwen_custom", "engine": "qwen_custom", "anchor": None,
     "tag": {"speaker": "aiden", "instruct": CUSTOM_INSTRUCT}},
    {"name": "jess_orpheus", "engine": "orpheus", "anchor": None,
     "tag": {"voice": "jess"}},
)
BATCHED_CELL = "M4-s3_qwen_base_batched"
BATCHED_ANCHOR = "M4-s3"
BATCHED_LINES = 100

DIRECTION_DECISIONS = {
    "qwen_base": ("no direction exists to decide: the clone takes no instruction, "
                  "so the tag is empty and the reference clip is the whole direction"),
    "qwen_custom": ("speaker ryan/aiden (the checkpoint's only English-native presets) "
                    "with the one fixed clause %r on every line: neutral narration in "
                    "the short shape qwen_custom.md mandates" % CUSTOM_INSTRUCT),
    "orpheus": ("voice jess (orpheus.md's measured-reliable female voice) with each "
                "line verbatim and no inline tags: this measures the voice, not "
                "directability"),
}

# Weights pins, from the .sha256 sidecars beside the weights
# (/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-{Base,CustomVoice}.sha256) and the
# onboarding checklist's recorded revisions. engine_pin re-hashes both
# safetensors and refuses a mismatch, so a swapped or truncated download
# cannot render under a recorded pin.
QWEN_WEIGHTS = {
    "qwen_base": {
        "dir": "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model_safetensors_sha256":
            "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c",
        "speech_tokenizer_sha256":
            "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    },
    "qwen_custom": {
        "dir": "/data/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "model_safetensors_sha256":
            "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
        "speech_tokenizer_sha256":
            "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    },
}
ORPHEUS_WEIGHTS = "/data/models/canopylabs/orpheus-3b-0.1-ft"

SYNTH_BANK = r1.SYNTH_BANK
CONTAINER_ENV = r1.CONTAINER_ENV

# Every tracked file whose content decides what `render` launches or what a 1c-ii cell's
# render produces — the same construction as render_1ci's list (launch path, probe-mode
# steps, this tool and its plan modules, every repo module those import at module level),
# with this step's three renderers and their qwen_load in place of 1c-i's two engines.
# render_1ci.py itself is listed: this tool imports its plan checks, so its content
# decides what a 1c-ii launch verifies. test_render_1cii.py checks the coverage the
# same way test_render_1ci.py checks 1c-i's.
RENDER_AFFECTING_PATHS = (
    "scripts/stages/synth_bank.sh",
    "scripts/container_env.sh",
    "scripts/container_as_ai_mgr.sh",
    "scripts/capture_container_env.sh",
    "scripts/stages/check_bank.py",
    "scripts/stages/synth_qwen_base.py",
    "scripts/stages/synth_qwen_custom.py",
    "scripts/stages/synth_orpheus.py",
    "scripts/stages/normalize_loudness.py",
    "scripts/stages/qc_gate.py",
    "scripts/tools/render_1cii.py",
    "scripts/tools/render_1ci.py",
    "scripts/lib/book_ingest.py",
    "scripts/lib/probe_text_bank.py",
    "scripts/lib/qwen_load.py",
    "scripts/lib/ref_select.py",
    "scripts/lib/synth_common.py",
    "scripts/lib/schemas.py",
    "scripts/lib/derive_vat_corpus.py",
    "scripts/lib/silero_vad.py",
    "matcha/__init__.py",
    "matcha/delivery.py",
)
# Reached from the files above but not part of a 1c-ii render, each with the reason.
NOT_RENDER_AFFECTING = {
    "scripts/lib/gemma_client.py": "book_ingest imports it for the director pass; plan never calls it",
    "scripts/lib/gemma_server.py": "book_ingest imports it for the director pass; plan never calls it",
    "scripts/stages/qc_verdict.py": "probe mode skips the direction check",
    "scripts/stages/eiv_score.sh": "probe mode skips the direction check",
    "scripts/stages/qc_engine_defects.py": "probe mode skips the defect detectors (PROBE_MODE_GAP)",
    "scripts/stages/register_audition.py": "probe mode skips the audition queue",
    "scripts/stages/librivox_align.sh": "another orchestrator; container_env.sh's header names it",
    "scripts/stages/synth_chatterbox.py": "another engine's block (Step 1c-i's engine)",
    "scripts/stages/synth_zonos.py": "another engine's block (Step 1c-i's engine)",
    "scripts/stages/synth_dia.py": "another engine's block",
    "scripts/stages/synth_moss_vg.py": "another engine's block",
    "scripts/stages/synth_moss85.py": "another engine's block",
    "scripts/stages/synth_qwen.py": "another engine's block (VoiceDesign, the anchor source)",
    "scripts/stages/synth_vibevoice.py": "another engine's block",
}

PROBE_MODE_GAP = (
    "probe mode skips scripts/stages/qc_engine_defects.py (it needs the campaign's "
    "ratings.csv rows), so no per-engine defect detector runs on these cells; the "
    "1c-ii evaluator must run them itself.")

BATCHED_RUN = ("the pre-registered 100-line batched Qwen Base throughput run: cell "
               "M4-s3_qwen_base_batched, planned by `batched --batch-size N` (N is the "
               "operator's measured largest batch that fits) and rendered by the same "
               "`render` path. Its clips are throughput-only: a batched call draws every "
               "clip from one RNG stream, so per-clip seeds do not hold.")
TIMING = ("per manifest row, by time.monotonic(): render_seconds = the engine's generate "
          "call (amortised per clip in the batched cell, which also records "
          "batch_size/batch_seconds); ref_condition_seconds = the reference conditioning "
          "(the clone-prompt build for qwen_base, 0.0 for the preset/closed-voice "
          "engines); audio_seconds; rtf = render_seconds / audio_seconds; the model load "
          "is in neither. See each renderer.")


def refuse(msg):
    raise SystemExit("REFUSING: " + msg)


def cell_names():
    return [c["name"] for c in CELLS] + [BATCHED_CELL]


def _cell(name):
    for c in CELLS:
        if c["name"] == name:
            return c
    if name == BATCHED_CELL:
        return {"name": name, "engine": "qwen_base", "anchor": BATCHED_ANCHOR, "tag": {}}
    refuse("no cell %r; the cells are %s." % (name, ", ".join(cell_names())))


def directions_for(cell, rows):
    """(per-line directions, the summary recorded in plan_meta), or a refusal.

    Every cell's direction is fixed except Orpheus's render_text, which is each line
    verbatim. A value the tag sets that build_direction moved, dropped or re-routed —
    or a direction that varies where it must not — means the clips would not render
    what was decided, so the plan refuses instead of recording it."""
    engine = cell["engine"]
    tag = dict(cell["tag"], engine=engine)
    out = [book_ingest.build_direction(tag, r["text"], lane=LANE) for r in rows]
    if any(e != engine for e, _d in out):
        refuse("build_direction re-routed a %s line to another engine." % engine)
    dirs = [d for _e, d in out]
    moved = {}
    for row, d in zip(rows, dirs):
        for k, v in cell["tag"].items():
            if (k not in d or d[k] != v) and k not in moved:
                moved[k] = (v, d.get(k))
    if moved:
        refuse("build_direction changed the %s direction the cell tag sets: %s." % (
            cell["name"],
            ", ".join("%s %r -> %r" % (k, a, b) for k, (a, b) in sorted(moved.items()))))
    if engine == "orpheus":
        bad = [r["id"] for r, d in zip(rows, dirs)
               if d != {"voice": cell["tag"]["voice"], "render_text": r["text"]}]
        if bad:
            refuse("orpheus directions are not voice + the line verbatim for %d "
                   "line(s), first: %s." % (len(bad), bad[0]))
        return dirs, {"voice": cell["tag"]["voice"],
                      "render_text": "<each line verbatim>"}
    if any(d != dirs[0] for d in dirs):
        refuse("build_direction gave the %s lines more than one direction." % cell["name"])
    return dirs, dirs[0]


def build_bank(cell, anchor, rows, directions, batch_size=None):
    lines = []
    for i, (r, d) in enumerate(zip(rows, directions), 1):
        line = {"id": "%s-%03d" % (cell["name"], i), "engine": cell["engine"],
                "render_index": i, "text_id": r["id"], "text": r["text"],
                "seed": r1.seed_for(i), "direction": dict(d)}
        if anchor is not None:
            line["ref_file"] = dict(anchor)
        lines.append(line)
    bank = {"version": VERSION, "campaign": "1cii-" + cell["name"], "lines": lines}
    if batch_size is not None:
        bank["batch_size"] = batch_size
    return bank


def _pin_from(path, pattern, what):
    with open(path, encoding="utf-8") as fh:
        m = re.search(pattern, fh.read(), re.M)
    if not m:
        refuse("%s has no %s." % (path, what))
    return m.group(1)


def _transformers_pin():
    return _pin_from(CONTAINER_ENV, r'^SONORA_PIN_TRANSFORMERS="([^"]+)"',
                     "SONORA_PIN_TRANSFORMERS")


def _qwen_wheel_pin():
    """The one qwen-tts pin synth_bank.sh installs; refused if its blocks disagree."""
    with open(SYNTH_BANK, encoding="utf-8") as fh:
        pins = sorted(set(re.findall(r"qwen-tts==[0-9][0-9.]*", fh.read())))
    if len(pins) != 1:
        refuse("%s installs %s; the qwen blocks must agree on one qwen-tts pin."
               % (SYNTH_BANK, pins or "no qwen-tts pin at all"))
    return pins[0]


def _verified_sha(path, pinned, what):
    if not os.path.isfile(path):
        refuse("%s does not exist, so the %s cannot be recorded." % (path, what))
    try:
        got = ref_select.file_sha256(path)
    except OSError as e:
        refuse("%s cannot be read: %s." % (path, e))
    if got != pinned:
        refuse("%s has sha256 %s, not the pinned %s. The weights on disk are not "
               "the weights the onboarding checklist recorded." % (path, got, pinned))
    return got


def engine_pin(engine):
    """What fixes the engine's code and weights, verified rather than transcribed."""
    if engine in QWEN_WEIGHTS:
        w = QWEN_WEIGHTS[engine]
        return {"wheel": _qwen_wheel_pin(), "transformers": _transformers_pin(),
                "weights_dir": w["dir"],
                "model_safetensors_sha256": _verified_sha(
                    os.path.join(w["dir"], "model.safetensors"),
                    w["model_safetensors_sha256"], engine + " weights pin"),
                "speech_tokenizer_sha256": _verified_sha(
                    os.path.join(w["dir"], "speech_tokenizer", "model.safetensors"),
                    w["speech_tokenizer_sha256"], engine + " tokenizer pin")}
    if engine == "orpheus":
        if not os.path.isdir(ORPHEUS_WEIGHTS):
            refuse("the Orpheus weights dir %s does not exist, so the engine code "
                   "this cell renders with cannot be recorded." % ORPHEUS_WEIGHTS)
        # snac is installed unpinned by synth_bank.sh's orpheus block; recorded
        # as the gap it is (the campaign _env freeze captures what resolved),
        # not papered over with a pin this tool cannot enforce.
        return {"weights_dir": ORPHEUS_WEIGHTS, "transformers": _transformers_pin(),
                "snac": "unpinned in synth_bank.sh's orpheus block"}
    refuse("no engine pin is defined for %r." % (engine,))


def plan_meta(cell, anchor, summary, git, n_lines, batch_size=None):
    meta = {
        "tool": TOOL, "spec": SPEC, "commit": git["commit"],
        "tree_clean": git["tree_clean"],
        "planned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "cell": cell["name"], "engine": cell["engine"],
        "anchor": dict(anchor) if anchor is not None else None,
        "text": {"bank": r1.TEXT_BANK, "meta": r1.TEXT_META, "sha256": r1.TEXT_SHA256,
                 "lines": n_lines, "canonical_form": probe_text_bank.CANONICAL_FORM},
        "render_order": "the text bank's line order; render_index is 1-based",
        "direction": {"lane": LANE, "tag": dict(cell["tag"], engine=cell["engine"]),
                      "via": "book_ingest.build_direction", "direction": summary,
                      "decision": DIRECTION_DECISIONS[cell["engine"]]},
        "seeds": {"rule": SEED_RULE, "SEED_BASE": SEED_BASE, "first": r1.seed_for(1),
                  "last": r1.seed_for(n_lines)},
        "engine_pin": engine_pin(cell["engine"]),
        "synth_bank": {"env": r1.PROBE_ENV, "out": "audio", "known_gap": PROBE_MODE_GAP},
        "render_affecting_paths": list(RENDER_AFFECTING_PATHS),
        "throughput": TIMING, "batched_run": BATCHED_RUN,
    }
    if batch_size is not None:
        meta["batch_size"] = batch_size
        meta["batched_of"] = ("the first %d lines of %s_qwen_base; speed-up is read "
                              "against that cell's batch-size-1 rows"
                              % (BATCHED_LINES, BATCHED_ANCHOR))
    return meta


def planned_cells(batch_size=None):
    """{cell: (cell, anchor|None, bank, summary)} from the verified inputs.

    The five measured cells always; the batched cell only when `batch_size` is
    given (its bank cannot be rebuilt without knowing it)."""
    rows = r1.load_text()
    anchors = {a["id"]: a for a in r1.load_anchors()}
    out = {}
    for cell in CELLS:
        anchor = anchors[cell["anchor"]] if cell["anchor"] else None
        dirs, summary = directions_for(cell, rows)
        out[cell["name"]] = (cell, anchor, build_bank(cell, anchor, rows, dirs), summary)
    if batch_size is not None:
        cell = _cell(BATCHED_CELL)
        anchor = anchors[BATCHED_ANCHOR]
        head = rows[:BATCHED_LINES]
        dirs, summary = directions_for(cell, head)
        out[cell["name"]] = (cell, anchor,
                             build_bank(cell, anchor, head, dirs, batch_size), summary)
    return out


def _refuse_existing(cdir, bank, git):
    """The pre-write refusals, shared by plan and batched (1c-i's, cell by cell)."""
    if os.path.exists(os.path.join(cdir, "bank.json")) and \
            r1._load_json(os.path.join(cdir, "bank.json")) != bank:
        refuse("%s/bank.json differs from this plan. Plan into a new --out-root." % cdir)
    meta = os.path.join(cdir, "plan_meta.json")
    if r1._has_manifest(os.path.join(cdir, "audio")) and \
            (not os.path.exists(meta) or r1._load_json(meta).get("commit") != git["commit"]):
        refuse("%s/audio holds renders planned at another commit (%s; this checkout "
               "is %s); re-planning would mix two plans. Resume with `render` against "
               "the existing plan, or plan into a new --out-root."
               % (cdir, r1._load_json(meta).get("commit") if os.path.exists(meta)
                  else "no plan_meta.json", git["commit"]))


def _write_cell(out_root, cell, anchor, bank, summary, git, batch_size=None):
    cdir = os.path.join(out_root, cell["name"])
    os.makedirs(cdir, exist_ok=True)
    write_json_atomic(os.path.join(cdir, "bank.json"), bank)
    write_json_atomic(os.path.join(cdir, "plan_meta.json"),
                      plan_meta(cell, anchor, summary, git, len(bank["lines"]),
                                batch_size))
    via = ("ref %s" % anchor["id"]) if anchor is not None else (
        cell["tag"].get("speaker") or cell["tag"].get("voice") or "-")
    print("%-24s %d lines  %-8s  seeds %d-%d -> %s" % (
        cell["name"], len(bank["lines"]), via, bank["lines"][0]["seed"],
        bank["lines"][-1]["seed"], cdir))


def cmd_plan(args):
    git = r1.tool_commit()
    r1._require_clean(git, args.allow_dirty)
    out_root = os.path.abspath(args.out_root)
    cells = planned_cells()
    # Engine pins are verified before the first write too: plan_meta derives them
    # again per cell, but a weights refusal discovered mid-loop would leave
    # earlier cells written under a plan that as a whole was refused.
    for engine in sorted({c["engine"] for c, _a, _b, _s in cells.values()}):
        engine_pin(engine)
    for cell, _a, bank, _s in cells.values():
        _refuse_existing(os.path.join(out_root, cell["name"]), bank, git)
    for cell, anchor, bank, summary in cells.values():
        _write_cell(out_root, cell, anchor, bank, summary, git)
        print("direction %-24s %s" % (cell["name"], json.dumps(summary, sort_keys=True)))
    print("commit %s%s" % (git["commit"], "" if git["tree_clean"] else " (DIRTY TREE)"))
    print("batched run: " + BATCHED_RUN)


def cmd_batched(args):
    """Plan the throughput cell only; `render` executes it like any other cell."""
    if args.batch_size < 2:
        refuse("--batch-size must be at least 2 (1 is the measured cells' own path).")
    git = r1.tool_commit()
    r1._require_clean(git, args.allow_dirty)
    out_root = os.path.abspath(args.out_root)
    cell, anchor, bank, summary = planned_cells(batch_size=args.batch_size)[BATCHED_CELL]
    engine_pin(cell["engine"])
    _refuse_existing(os.path.join(out_root, cell["name"]), bank, git)
    _write_cell(out_root, cell, anchor, bank, summary, git, batch_size=args.batch_size)
    print("render it with: render %s --out-root %s --run" % (BATCHED_CELL, out_root))
    print("commit %s%s" % (git["commit"], "" if git["tree_clean"] else " (DIRTY TREE)"))


def render_inputs_changed(plan_commit, head):
    """This tool's RENDER_AFFECTING_PATHS that differ between the commits, [] when none.

    render_1ci's launch gate hard-codes its own list and that tool is frozen while
    1c-i renders, so the comparison is restated here over this tool's list; the
    refusals are the same (uncomparable commits and untracked listed paths refuse
    rather than vouch)."""
    if not isinstance(plan_commit, str) or not plan_commit:
        refuse("plan_meta.json names no plan commit (%r). Plan into a new --out-root."
               % (plan_commit,))
    if plan_commit == head:
        return []
    paths = list(RENDER_AFFECTING_PATHS)
    tracked = r1._git("ls-tree", "-r", "--name-only", head, "--", *paths)
    if tracked.returncode != 0:
        refuse("git cannot list %s's tree: %s" % (head, tracked.stderr.strip()))
    missing = sorted(set(paths) - set(tracked.stdout.split()))
    if missing:
        refuse("RENDER_AFFECTING_PATHS names files %s does not track: %s."
               % (head, ", ".join(missing)))
    quiet = r1._git("diff", "--quiet", plan_commit, head, "--", *paths)
    if quiet.returncode == 0:
        return []
    if quiet.returncode != 1:
        refuse("git cannot compare the plan commit %s with this checkout's %s (%s). "
               "Render from a checkout that has the plan commit."
               % (plan_commit, head, quiet.stderr.strip()))
    changed = r1._git("diff", "--name-only", plan_commit, head, "--", *paths).stdout.split()
    if not changed:
        refuse("git diff --quiet says the render-affecting files differ between %s "
               "and %s, and --name-only names none of them." % (plan_commit, head))
    return changed


def render_command(out_root, cell, engine):
    cdir = os.path.join(out_root, cell)
    return [SYNTH_BANK, os.path.join(cdir, "bank.json"), os.path.join(cdir, "audio"), engine]


def _shell_line(argv):
    return " ".join("%s=%s" % kv for kv in r1.PROBE_ENV.items()) + " " + shlex.join(argv)


def cmd_render(args):
    git = r1.tool_commit()
    r1._require_clean(git, args.allow_dirty)
    out_root = os.path.abspath(args.out_root)
    cell = _cell(args.cell)
    cdir = os.path.join(out_root, args.cell)
    try:
        on_disk = r1._load_json(os.path.join(cdir, "bank.json"))
        meta = r1._load_json(os.path.join(cdir, "plan_meta.json"))
    except (OSError, ValueError) as e:
        refuse("%s has no readable plan (%s). Run `plan` (or `batched`) first." % (cdir, e))
    if args.cell == BATCHED_CELL:
        bs = meta.get("batch_size")
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 2:
            refuse("%s records batch_size %r; the batched cell needs an int "
                   "batch_size >= 2. Re-plan it with `batched --batch-size N`."
                   % (cdir, bs))
    # The batched bank cannot be rebuilt without its recorded batch size.
    cells = planned_cells(batch_size=meta.get("batch_size"))
    if args.cell not in cells:
        refuse("%s was not planned by this tool (no batch_size recorded for the "
               "batched cell?). Re-plan it." % cdir)
    _c, _anchor, bank, _summary = cells[args.cell]
    engine = cell["engine"]
    plan_commit = meta.get("commit")
    changed = render_inputs_changed(plan_commit, git["commit"])
    if changed:
        refuse("%s was planned at %s, and this checkout is %s; these render-affecting "
               "files differ between the two: %s. Render from a checkout at the plan "
               "commit (`git worktree add --detach <dir> %s`; it needs its own .venv), "
               "or plan into a new --out-root at this commit."
               % (cdir, plan_commit, git["commit"], ", ".join(changed), plan_commit))
    if plan_commit != git["commit"]:
        print("HEAD %s is not the plan commit %s; no render-affecting file differs "
              "between them." % (git["commit"], plan_commit))
    pin_now = engine_pin(engine)
    if meta.get("engine_pin") != pin_now:
        refuse("%s was planned with the %s engine pin %s, and it is now %s; the "
               "engine code is not the code the plan recorded. Restore the planned "
               "pin, or plan into a new --out-root with this one."
               % (cdir, engine, json.dumps(meta.get("engine_pin"), sort_keys=True),
                  json.dumps(pin_now, sort_keys=True)))
    if on_disk != bank:
        refuse("%s/bank.json is not the bank these inputs and this code plan. Re-plan."
               % cdir)
    argv = render_command(out_root, args.cell, engine)
    line = _shell_line(argv)
    blockers = []
    if not (out_root + "/").startswith(r1.DATA_ROOT) or re.search(r"[\s'\"]", out_root):
        blockers.append("synth_bank.sh needs BANK and OUT under %s, without spaces "
                        "or quotes; %s is not." % (r1.DATA_ROOT, out_root))
    if not os.access(r1.VENV_PYTHON, os.X_OK):
        blockers.append("synth_bank.sh runs check_bank, loudness and the QC gate "
                        "with %s, which does not exist." % r1.VENV_PYTHON)
    print(line)
    for b in blockers:
        print("WARNING: " + b)
    if not args.run:
        if r1._has_manifest(os.path.join(cdir, "audio")):
            r1.report_status(args.cell, r1.manifest_status(cdir, engine, bank))
        print("(printed only; --run executes it. The GPU is booked separately.)")
        return 0
    if blockers:
        refuse("--run with the blockers above.")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    code = subprocess.run(argv, env=dict(os.environ, **r1.PROBE_ENV)).returncode
    status = r1.manifest_status(cdir, engine, bank)
    append_jsonl(os.path.join(cdir, "render_log.jsonl"), [{
        "command": line, "plan_commit": plan_commit, "head_commit": git["commit"],
        "tree_clean": git["tree_clean"],
        "allow_dirty": bool(args.allow_dirty), "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "exit_code": code, "rendered": status["rendered"],
        "expected": status["expected"],
        "missing": status["missing"], "wav_without_row": status["wav_without_row"]}])
    if code != 0:
        print("synth_bank.sh exited %d." % code, file=_sys.stderr)
    partial = r1.report_status(args.cell, status, synth_bank_code=code)
    return code if code != 0 else partial


def cmd_status(args):
    """The cell's rendered/expected counts; launches nothing, re-checks no inputs."""
    _cell(args.cell)
    cdir = os.path.join(os.path.abspath(args.out_root), args.cell)
    try:
        bank = r1._load_json(os.path.join(cdir, "bank.json"))
        meta = r1._load_json(os.path.join(cdir, "plan_meta.json"))
    except (OSError, ValueError) as e:
        refuse("%s has no readable plan (%s). Run `plan` first." % (cdir, e))
    return r1.report_status(args.cell, r1.manifest_status(cdir, meta.get("engine"), bank))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="verify the inputs and write the five campaign dirs")
    p.add_argument("--out-root", required=True)
    p.add_argument("--allow-dirty", action="store_true")
    b = sub.add_parser("batched", help="plan the 100-line batched throughput cell")
    b.add_argument("--out-root", required=True)
    b.add_argument("--batch-size", type=int, required=True,
                   help="the measured largest batch that fits (throughput only)")
    b.add_argument("--allow-dirty", action="store_true")
    r = sub.add_parser(
        "render", help="print (or with --run, execute) one cell's synth_bank.sh",
        description="The checks (plan inputs, render-affecting files, engine pin, "
                    "bank) run once, at launch: a pull into the shared checkout "
                    "while the cell runs is not detected, so do not pull mid-cell.")
    r.add_argument("cell", help="one of: " + ", ".join(cell_names()))
    r.add_argument("--out-root", required=True)
    r.add_argument("--run", action="store_true", help="execute it (GPU); default: print only")
    r.add_argument("--allow-dirty", action="store_true")
    s = sub.add_parser("status", help="one cell's rendered/expected counts; launches nothing")
    s.add_argument("cell", help="one of: " + ", ".join(cell_names()))
    s.add_argument("--out-root", required=True)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "batched": cmd_batched, "render": cmd_render,
            "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    _sys.exit(main())



