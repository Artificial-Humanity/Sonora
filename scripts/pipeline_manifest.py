"""Which pipeline stages a dataset-generation pass must run, and which shell wires each.

This file exists because **"nothing invokes this script" is the normal, healthy state in
`scripts/`** — 100 non-test `.py` files after #26 step 2 (106 tracked, less the 6 gate
scripts), most of them operator tools — so being
uninvoked carried no signal, and a *stage* that stopped being invoked looked exactly like
the ~70 files that never were. The buckets (`scripts/README.md`) now say what a file IS;
this file is what makes a stage's WIRING checkable.
`qc_verdict.py` sat in that fog for a month: named in `synth_bank.sh` inside a comment, and
never run (issue #24). 695 directed clips reached the ear with no intended-vs-measured check.

The fix is the technique `matcha/direction.py` already established for conditioning
channels — one declaration, plus a coverage test that iterates it:

    Adding a dimension without a control now fails a test instead of being noticed later
    by ear.

Applied here: **adding a stage without wiring it, or unwiring a stage, now fails
`tests/test_stage_coverage.py` instead of being noticed a month later by its absence from
the output.** The test checks BOTH directions, so this file cannot quietly drift from the
shells it describes: a stage declared here and not invoked fails, and a script invoked by
an orchestrator and not declared here fails too.

## How to change it

* Wiring a new stage into an orchestrator → add it to that orchestrator's `stages`.
* Removing a stage → remove it here, in the same commit. If the removal is deliberate but
  the script stays referenced (a printed next-step hint, say), move it to
  `deliberately_not_invoked` **with the reason**, because that is a decision and the next
  reader needs it, not a rediscovery.
* Adding a `.sh` under `scripts/` → it must be listed in exactly one of `ORCHESTRATORS`,
  `DYNAMIC_DISPATCH` or `NOT_ORCHESTRATORS`. That is what stops the next orchestrator from
  arriving with its stages undeclared.

## What this file does NOT do

It records *that* a stage is wired, not that it is wired *usefully*. Several stages are
inert without a particular flag — `qc_verdict.py` without `--append-flags` writes verdicts
no axis failure ever escapes — and those argument-level assertions live per stage in
`tests/test_audit_sampling.py`, along with the one ordering constraint that matters
(verdicts frozen before `register_audition` runs, or the `gate_calibration` round is
spent). This manifest is the coverage half: it is what makes the *set* complete.

## Two mechanisms that defeat a text search, and why the test parses instead

1. **A comment is indistinguishable from a call.** This is the #24 failure verbatim. Any
   guard that a comment can satisfy is not a guard.
2. **An `echo` is indistinguishable from a call.** Every stage in `synth_bank.sh` prints
   the command to re-run it by hand on failure, so the recovery hints name the very
   scripts under test. Dropping the invocation and keeping the hint would leave a
   comment-aware guard green.

Both are handled by `tests/test_audit_sampling.py::_invocations`, which the coverage test
imports rather than copies. ⚠ Both were only PARTLY closed until 2026-08-12: the comment
filter dropped whole-line comments only, and the echo filter matched a LEADING BARE `echo`
only — so a trailing comment and `>&2 echo "…"` each walked a fully unwired stage through
the guard. `librivox_align.sh`'s `stage_pool` line is a live fixture for the leading-bare
form; the other spellings are covered by literal fixtures in `test_audit_sampling.py`.
"""

from typing import NamedTuple


class Stage(NamedTuple):
    """One script an orchestrator must invoke, and what it is for."""

    script: str  # repo-relative path; the test asserts it exists
    role: str


STAGES = "scripts/stages/"

# Shells an orchestrator may run without them being stages: two are `.`-sourced definition
# files, and the third runs inside the throwaway container to create the ai-mgr passwd
# entry. Named explicitly rather than by exempting everything in NOT_ORCHESTRATORS, so
# wiring (say) a developer tool into a data pass still has to be declared.
STRUCTURAL_HELPERS = (
    "container_env.sh",
    "capture_container_env.sh",
    "container_as_ai_mgr.sh",
)

# Statically wired orchestrators: every stage below must appear as a real invocation, and
# every scripts/ file really invoked must appear below.
ORCHESTRATORS = {
    STAGES + "synth_bank.sh": {
        "purpose": "the synthetic-bank generation pass: render, normalise, gate, register",
        "stages": (
            Stage(STAGES + "check_bank.py", "refuses a malformed bank before any GPU time is spent"),
            # One branch per engine. WHICH engines run is a runtime choice; that each engine
            # the script can select is actually reachable is not.
            Stage(STAGES + "synth_chatterbox.py", "engine render — chatterbox (live, trusted-provisional)"),
            Stage(STAGES + "synth_dia.py", "engine render — dia (benched; branch retained)"),
            Stage(STAGES + "synth_moss85.py", "engine render — moss85 (not a directed teacher; cloning candidate)"),
            Stage(STAGES + "synth_moss_vg.py", "engine render — moss_vg (live, scrutinized tier)"),
            Stage(STAGES + "synth_orpheus.py", "engine render — orpheus (live, normal tier)"),
            Stage(STAGES + "synth_qwen.py", "engine render — qwen (live, trusted tier)"),
            Stage(STAGES + "synth_vibevoice.py", "engine render — vibevoice (benched; branch retained)"),
            Stage(STAGES + "synth_zonos.py", "engine render — zonos (live, normal tier)"),
            Stage(
                STAGES + "normalize_loudness.py",
                "-23 LUFS, fatal on failure, and it must precede QC — the measures are level-sensitive",
            ),
            Stage(STAGES + "qc_gate.py", "QC stage 1 — measurable defects (mandatory after every pass)"),
            Stage(STAGES + "qc_verdict.py", "QC stage 2 — intended-vs-measured direction check (issue #24)"),
            Stage(STAGES + "qc_engine_defects.py", "per-engine defect roll-up across the campaign"),
            Stage(STAGES + "register_audition.py", "writes the campaign's clips into ratings.csv"),
        ),
        # A stage that is itself an orchestrator. Easy to miss in an audit that greps for
        # `.py`: the bank pass produces the EIV scores by running the EIV lane, and without
        # them `qc_verdict --eiv` has nothing to merge.
        "invokes_orchestrators": ("scripts/stages/eiv_score.sh",),
        "non_stage_scripts": (),
        "deliberately_not_invoked": {},
    },
    STAGES + "librivox_align.sh": {
        "purpose": "the real-audio force-align pass: segment a book into the clip pool",
        "stages": (
            Stage(STAGES + "librivox_align.py", "force-aligns canonical text to the reading and cuts clips"),
            Stage(STAGES + "qc_gate.py", "the QC gate follows EVERY dataset pass — real audio is not exempt"),
        ),
        "invokes_orchestrators": (),
        "non_stage_scripts": (),
        "deliberately_not_invoked": {
            "scripts/tools/stage_pool.py": (
                "The aligner fills the POOL; the pool is not the audition queue. Registering here "
                "auto-queued 652 unaudited rows off one book on 2026-08-01, which is the flood the "
                "pool/staging split exists to prevent. The script is PRINTED as the next step and "
                "run by hand. It is also this manifest's only live fixture for the echo filter: if "
                "the parser ever stopped dropping a LEADING BARE `echo`, this entry is what fails — the "
                "redirected and braced spellings have their own fixtures in test_audit_sampling.py."
            ),
        },
    },
    "scripts/stages/eiv_score.sh": {
        "purpose": "the EIV labelling pass (LAION Empathic-Insight-Voice heads)",
        "stages": (Stage("scripts/stages/eiv_score.py", "scores wavs into an append-only, resumable jsonl"),),
        "invokes_orchestrators": (),
        "non_stage_scripts": (),
        "deliberately_not_invoked": {},
    },
    "scripts/stages/score_holdout.sh": {
        "purpose": "the holdout scorer — the SSOT for cross-run checkpoint comparison",
        "stages": (Stage("scripts/stages/score_holdout.py", "teacher-forced forward passes over the frozen holdout"),),
        "invokes_orchestrators": (),
        # setup.py builds monotonic_align (Cython) in a /tmp copy of the package. A build
        # step, not a pipeline stage, and named here so it is allowed rather than ignored.
        "non_stage_scripts": ("setup.py",),
        "deliberately_not_invoked": {},
    },
}

# Orchestrators whose target is chosen at runtime, so static reachability cannot see their
# stages at all. Declared, not enforced — the value here is that the next audit knows the
# blind spot exists instead of concluding the lane is dead.
DYNAMIC_DISPATCH = {
    "scripts/litert_export/run.sh": (
        "Takes the script name as $1 and `exec`s it, so none of the export-lane scripts in "
        "`scripts/litert_export/` appear as called by anything. Deliberate: it is the wrapper that "
        "separated code from data on 2026-08-06 (SONORA_LITERT_WORK / SONORA_REPO). If a hardcoded "
        "script list ever lands in it, that list belongs in ORCHESTRATORS above."
    ),
}

# Everything else under scripts/. Listed so that a NEW shell cannot arrive without a
# decision about what it is.
NOT_ORCHESTRATORS = {
    "scripts/container_env.sh": "sourced, not run — the shared pinned-image and uv bootstrap definitions",
    "scripts/capture_container_env.sh": "sourced helper — records the resolved environment beside the artifacts (D-M2)",
    "scripts/container_as_ai_mgr.sh": "runs INSIDE the throwaway container to create the ai-mgr passwd entry",
    "scripts/teacher_audition/render_longcat.sh": (
        "Preserved audition provenance for rated clips, and an INTERFACE record for a benched "
        "engine (teacher_audition/README.md). Invokes a toolchain script outside this repo."
    ),
    "scripts/teacher_audition/realtime_05b_study.sh": (
        "Preserved audition provenance — added for exactly that reason in 4e87241. Invokes a "
        "toolchain script outside this repo."
    ),
    # --- the review lane (workflow/), not the data pipeline -----------------------------
    # ⚠ These live outside `scripts/` since 2026-08-17 and are declared here anyway, because
    # the enumeration behind this gate is REPO-WIDE rather than `scripts/*.sh`. That widening
    # (2026-08-12) was made when every shell happened to live under `scripts/`, so it changed
    # no result and read as belt-and-braces. `workflow/` is the first thing to exercise it —
    # leaving these undeclared would be exactly the exemption-by-construction it removed.
    "workflow/scripts/request_review.sh": (
        "Review lane: runs `claude -p` as Janis over a commit range, and blocks. Touches no "
        "pipeline stage and writes no artifact under /data — its output is issues in the "
        "tracker. See workflow/WORKFLOW.md §2."
    ),
    "workflow/scripts/review_cycle.sh": (
        "Review lane: drives request_review.sh and a `claude -p` worker unattended until the "
        "branch converges. ⚠ It NEVER pushes, and denies `git push` and `merge_branch.sh` to "
        "the worker it spawns."
    ),
    "workflow/scripts/merge_branch.sh": (
        "Review lane: the merge gate. Refuses to merge a branch into main while any of its "
        "issues is open, review or escalated; then merges and pushes. ⚠ The only thing in this "
        "repo that reaches `main` on its own — see workflow/WORKFLOW.md §3."
    ),
}
