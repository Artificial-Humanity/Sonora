# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""publish_tier — enforce the train-only / publishable split on dataset metadata.

WHY THIS EXISTS (owner decision 2026-07-28). Chatterbox watermarks every output
with Perth unconditionally — no flag, no config, no env var — and the native
component is unavailable on this box, so our renderers disable it with a no-op
patch. The owner's call:

    strip the watermark, use those clips for TRAINING ONLY, never publish them.

That is a responsible-AI decision, not a licence term: Chatterbox is MIT and the
licence permits stripping. Which makes it a promise we own rather than one anyone
enforces for us — so it needs a mechanism, not a note.

Before this script, `metadata.jsonl` carried a flat `"license": "CC-BY-4.0"` on
every row with no publication tier and no engine attribution. A Chatterbox clip
folded into v1 would have been published to HuggingFace by default, watermark
stripped, with nothing marking it. The failure mode was silent, which is the worst
kind for a promise.

Two fields close it:
  * `engine_license`  — the WEIGHTS licence of the engine that rendered the clip
                        (not our dataset licence; those are different things and
                        conflating them is how licence walls get breached).
  * `publish`         — true = may go in the CC-BY-4.0 release; false = train-only.

UNKNOWN ENGINES ARE A HARD ERROR, deliberately. VibeVoice's and MOSS-VoiceGenerator's
weights licences are not recorded anywhere in the repo. Rather than guess one into a
published dataset, this refuses to run and says so. Add the entry once verified.

Usage:
    .venv/bin/python scripts/synthesis/publish_tier.py --meta <metadata.jsonl>            # check (default)
    .venv/bin/python scripts/synthesis/publish_tier.py --meta <metadata.jsonl> --apply    # backfill missing fields
"""
import argparse
import json
import pathlib
import shutil
import sys

# engine -> (weights licence, may appear in the published release)
#
# `publishable=False` means the clip is fine to TRAIN on but must never reach the
# CC-BY-4.0 release. Today that is a watermark-provenance call, not a licence one.
ENGINE_POLICY = {
    "qwen":       ("Apache-2.0", True),
    "moss85":     ("Apache-2.0", True),
    "dia":        ("Apache-2.0", True),
    "longcat":    ("MIT", True),
    "orpheus":    ("Apache-2.0", True),
    "zonos":      ("Apache-2.0", True),
    # Watermark stripped by our renderer patch -> train-only (owner 2026-07-28).
    "chatterbox": ("MIT", False),
    # NOT LISTED, deliberately: vibevoice, moss_vg. Their weights licences are
    # unverified. Verify against the model card, then add — do not infer one from
    # a sibling repo or a code licence.
    #
    # C-L5: `librivox` is not an ENGINE and its clips are not synthesised — they are
    # human recordings, public-domain text read by LibriVox volunteers, and the LibriVox
    # catalogue places its recordings in the public domain. There is no model licence to
    # verify, so the "verify the model card" instruction above is not merely unanswered
    # for this row, it is the wrong question. Absent this entry the `unknown` check below
    # hard-errors the day the first staged real-audio clip reaches a metadata.jsonl —
    # which is a matter of when, not whether: `stage_pool` writes `engine: "librivox"`
    # for the force-align lane and three books are already fetched.
    #
    # Publishable: the recordings are PD and the text is PD/CC0. That is a stronger
    # position than any synthetic row here, none of which can point at a human who
    # consented; it is listed explicitly rather than left to a default so the reasoning
    # is on the record.
    "librivox":   ("PD (LibriVox recording; PD/CC0 source text)", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to metadata.jsonl")
    ap.add_argument("--apply", action="store_true", help="backfill missing fields in place")
    args = ap.parse_args()

    path = pathlib.Path(args.meta)
    if not path.is_file():
        sys.exit(f"no such file: {path}")

    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: malformed JSON — {e}")

    unknown = sorted({r.get("engine", "?") for r in rows} - set(ENGINE_POLICY))
    if unknown:
        sys.exit(
            f"ERROR: no publication policy for engine(s): {', '.join(unknown)}\n"
            f"  Verify each one's WEIGHTS licence on its model card and add it to\n"
            f"  ENGINE_POLICY in {pathlib.Path(__file__).name}. Refusing to guess a\n"
            f"  licence into a dataset that gets published."
        )

    missing, violations, changed = [], [], 0
    for r in rows:
        lic, publishable = ENGINE_POLICY[r["engine"]]
        if "engine_license" not in r or "publish" not in r:
            missing.append(r.get("id", r.get("file", "?")))
            if args.apply:
                r["engine_license"] = lic
                r["publish"] = publishable
                changed += 1
        elif r["publish"] and not publishable:
            violations.append((r.get("id", "?"), r["engine"]))

    # A train-only clip marked publishable is the exact failure this script exists
    # to catch, so it is fatal in both modes — --apply must not paper over it.
    if violations:
        print(f"FAIL: {len(violations)} clip(s) marked publish=true from a train-only engine:")
        for cid, eng in violations[:20]:
            print(f"  {cid}  ({eng})")
        sys.exit(1)

    if args.apply and changed:
        backup = path.with_suffix(path.suffix + ".pre-publish-tier.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"backfilled {changed} row(s); original kept at {backup.name}")
    elif missing:
        print(f"FAIL: {len(missing)} row(s) lack engine_license/publish. Re-run with --apply.")
        sys.exit(1)

    n_pub = sum(1 for r in rows if r.get("publish"))
    print(f"OK: {len(rows)} row(s) — {n_pub} publishable, {len(rows) - n_pub} train-only")


if __name__ == "__main__":
    main()
