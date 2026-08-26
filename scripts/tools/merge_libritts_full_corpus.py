# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# ///
"""Phase 1 rung 3 — append the rest of LibriTTS-R to the v6 corpus, making v7.

HOW TO RUN IT: the repo **.venv**, not `uv run` (owner, 2026-08-01).

    .venv/bin/python scripts/tools/merge_libritts_full_corpus.py \
        --base data/libritts_r_emilia_expressive_vat_v6 \
        --add  data/_derived_train_clean_360 \
        --add  data/_derived_train_other_500 \
        --out  data/libritts_r_full_vat_v7

THE ONE THING THIS SCRIPT IS FOR, and it is the same one `merge_emilia_corpus` exists for:
v6's rows pass through BYTE-IDENTICAL and the new rows are appended with speaker indices
that continue v6's numbering. `make_warmstart --donor-speakers` widens `spk_emb.weight`
only after proving the donor's index map is a PREFIX of the new one, because row *i* of a
speaker table is a person. Renumber a single existing speaker and every voice the model
learned moves one seat to the left, silently.

WHY THIS IS NOT `merge_emilia_corpus` WITH DIFFERENT PATHS
----------------------------------------------------------
Emilia needed its own script because it needed a different LABELLING LANE: per-speaker z
destroys a corpus with a median of 3 clips per speaker, so those rows are labelled on a
global anchor. The new LibriTTS-R subsets are the SAME DOMAIN as the base and go through
the ordinary `derive_vat_corpus` per-speaker lane. So this script does no labelling at
all — it takes already-derived corpus directories and does one job: renumber and append.

⚠ WHY DERIVING THE NEW SUBSETS SEPARATELY IS SAFE, since it looks like it should not be.
A per-speaker z is computed within one speaker's own clips. Measured 2026-08-25 under
`LC_ALL=C` with a positive control: `train-clean-360`, `train-other-500`, `train-clean-100`
and `dev-clean` are pairwise speaker-DISJOINT (all four intersections empty). No speaker
spans two subsets, so deriving a subset alone produces exactly the values deriving it
alongside the others would. That disjointness is also what lets v6's rows stay untouched;
if it ever stops holding, this script's collision check below fails rather than quietly
producing a corpus whose older half was relabelled.
⚠ It does mean the independence gate is read PER DERIVED DIRECTORY, not over the pooled
corpus. Two readings, not one — do not report either as the corpus-wide figure.

WHAT THIS DOES NOT DO
---------------------
No dropping. `derive_vat_corpus` already applied the duration, sample-rate, digit,
vocabulary and G2P filters, and a second filter here would be a second definition of a
rule that already has one. If a row reached the input, it is in the output. The only way
out of this script is an ABORT that writes nothing.
"""

import argparse
import collections
import hashlib
import json
import os
import sys

_SONORA_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE = "data/libritts_r_emilia_expressive_vat_v6"
# The namespace new LibriTTS speakers join. It already holds the Emilia ids too — a wart
# inherited from v5 and NOT corrected here, because the key of this map is what the prefix
# proof reads. Renaming or re-bucketing it would change nothing about the corpus and would
# invalidate every warm start that has ever been proven against it.
LIBRITTS_NS = "libritts_id_to_index"


def die(msg):
    sys.exit("merge_libritts_full_corpus: %s" % msg)


def read_corpus(d):
    """(train_lines, val_lines, speakers.json) with rows kept as exact strings."""
    out = {}
    for name in ("train_op.txt", "val_op.txt"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            die("%s has no %s — is it a corpus directory?" % (d, name))
        with open(p, encoding="utf-8") as f:
            out[name] = [ln for ln in f.read().split("\n") if ln]
    sp = os.path.join(d, "speakers.json")
    if not os.path.exists(sp):
        die("%s has no speakers.json" % d)
    with open(sp, encoding="utf-8") as f:
        out["speakers"] = json.load(f)
    return out


def namespaces(spk):
    """Every id->index map in a speakers.json, by name."""
    return {k: v for k, v in spk.items() if isinstance(v, dict)}


def index_to_id(ns_maps):
    """index -> (namespace, id), refusing two ids on one row."""
    out = {}
    for ns, m in ns_maps.items():
        for sid, idx in m.items():
            if idx in out:
                die("two speakers on index %d: %s/%s and %s/%s — the donor corpus is "
                    "already broken, not this merge" % (idx, out[idx][0], out[idx][1], ns, sid))
            out[idx] = (ns, sid)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=BASE, help="corpus whose rows must survive byte-identical")
    ap.add_argument("--add", action="append", default=[], metavar="DIR",
                    help="a derive_vat_corpus output directory to append (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an --out that already holds a corpus")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()
    if not args.add:
        die("nothing to add — pass at least one --add DIR")

    # ⚠ #295. `--out` == `--base` would rewrite the base corpus in place, and the base is the
    # byte-identity REFERENCE — destroying it destroys the only thing that can prove the
    # append was legal, and the warm start's prefix proof with it. Compared by realpath so a
    # symlink or a trailing slash cannot slip past. An --out that already holds a corpus is
    # refused for the weaker version of the same reason: a merge is not an in-place edit.
    if os.path.realpath(args.out) == os.path.realpath(args.base):
        die("--out is the same directory as --base. This would overwrite the base corpus, "
            "which is the reference the byte-identity check is made against. Nothing written.")
    existing = [n for n in ("train_op.txt", "val_op.txt", "speakers.json")
                if os.path.exists(os.path.join(args.out, n))]
    if existing and not args.force:
        die("--out %s already holds %s. Refusing to overwrite a corpus; pass --force if "
            "that is genuinely what you want." % (args.out, ", ".join(existing)))

    # ⚠ #293. The same pre-flight `merge_emilia_corpus` runs, and for the reason its own
    # comment gives: the wall classifies the FILELIST'S OWN DIRECTORY, so a new corpus dir
    # needs a manifest entry even though it holds no audio. Asked early because the answer
    # does not change and finding out at load time costs the whole build. The authoritative
    # check is `license_check` on the written filelists at the end.
    from matcha.data.license_wall import classify_path
    from matcha.data.license_wall import enforce as license_check
    for p in (args.out, args.base, *args.add):
        hit = classify_path(p)
        if hit is None:
            die("%s matches no declared dataset. The licence wall would refuse this corpus "
                "at load; declare it in configs/data_licenses.yaml first." % p)
        if hit[1] == "nc":
            die("%s -> %s (%s) is NON-COMMERCIAL. NC data is de-risk-only and must not "
                "enter a corpus." % (p, hit[0], hit[2]))

    base = read_corpus(args.base)
    base_ns = namespaces(base["speakers"])
    base_n = base["speakers"].get("n_spks")
    base_idx = index_to_id(base_ns)
    if base_n != len(base_idx):
        die("base n_spks=%s but its maps hold %d distinct indices" % (base_n, len(base_idx)))
    # Contiguity: the warm start widens a table by APPENDING rows, so a hole in the base
    # would mean row k of spk_emb belongs to nobody and every later row is off by one.
    missing = sorted(set(range(base_n)) - set(base_idx))
    if missing:
        die("base index space is not contiguous — missing %s" % missing[:8])
    print("base %s: %d rows train / %d val, %d speakers across %s"
          % (args.base, len(base["train_op.txt"]), len(base["val_op.txt"]),
             base_n, ", ".join(sorted(base_ns))))

    known = {sid for m in base_ns.values() for sid in m}
    next_index = base_n
    added_map = {}
    new_train, new_val = [], []
    per_dir = []

    for d in args.add:
        add = read_corpus(d)
        add_ns = namespaces(add["speakers"])
        if LIBRITTS_NS not in add_ns:
            die("%s has no %s map — this script appends LibriTTS-lane corpora only" % (d, LIBRITTS_NS))
        if len(add_ns) != 1:
            die("%s carries more than one namespace (%s); a multi-namespace donor needs a "
                "decision about which ids join which map, and that is not this script's to "
                "make" % (d, ", ".join(sorted(add_ns))))
        local = add_ns[LIBRITTS_NS]

        # ⚠ THE CHECK THAT MAKES THE APPEND LEGAL. A speaker already in the base must never
        # be re-added: it would get a SECOND index, so the same person occupies two rows of
        # the embedding table and each row sees half their audio. Directory-level
        # disjointness was measured, but the corpus is the authority, not the filesystem.
        clash = sorted(set(local) & known)
        if clash:
            die("%d speaker id(s) in %s are ALREADY in the base (%s ...). Appending them "
                "would give one person two embedding rows. Nothing written."
                % (len(clash), d, ", ".join(clash[:6])))

        remap = {}
        for sid in sorted(local, key=lambda s: local[s]):
            remap[local[sid]] = next_index
            added_map[sid] = next_index
            known.add(sid)
            next_index += 1

        n_before = len(new_train) + len(new_val)
        for name, sink in (("train_op.txt", new_train), ("val_op.txt", new_val)):
            for ln in add[name]:
                parts = ln.split("|")
                if len(parts) != 4:
                    die("%s/%s: row has %d fields, expected 4 (path|spk|phonemes|vat):\n  %s"
                        % (d, name, len(parts), ln[:120]))
                try:
                    old = int(parts[1])
                except ValueError:
                    die("%s/%s: speaker field %r is not an index" % (d, name, parts[1]))
                if old not in remap:
                    die("%s/%s: row uses speaker index %d, which its own speakers.json "
                        "does not define" % (d, name, old))
                parts[1] = str(remap[old])
                sink.append("|".join(parts))
        per_dir.append((d, len(local), len(new_train) + len(new_val) - n_before))
        print("  + %s: %d speakers -> indices %d..%d, %d rows"
              % (d, len(local), remap[min(remap)], next_index - 1,
                 len(new_train) + len(new_val) - n_before))

    # The prefix proof, asserted here rather than left for make_warmstart to discover: every
    # base speaker must still hold its exact original index in the merged map.
    merged_ns = {ns: dict(m) for ns, m in base_ns.items()}
    merged_ns[LIBRITTS_NS].update(added_map)
    for ns, m in base_ns.items():
        for sid, idx in m.items():
            if merged_ns[ns][sid] != idx:
                die("PREFIX PROOF FAILED: %s/%s moved %d -> %d" % (ns, sid, idx, merged_ns[ns][sid]))
    merged_idx = index_to_id(merged_ns)
    if len(merged_idx) != next_index:
        die("merged map holds %d indices but the counter says %d" % (len(merged_idx), next_index))
    holes = sorted(set(range(next_index)) - set(merged_idx))
    if holes:
        die("merged index space is not contiguous — missing %s" % holes[:8])

    train = base["train_op.txt"] + new_train
    val = base["val_op.txt"] + new_val
    if not val:
        die("merged corpus has an EMPTY val split")
    # A vat field that changed width between the base and an addition would train two
    # different conditioning contracts in one corpus, and the trunk only checks the first
    # batch it happens to see.
    widths = collections.Counter(len(ln.split("|")[3].split(",")) for ln in train + val)
    if len(widths) != 1:
        die("rows disagree about the VAT vector width: %s" % dict(widths))

    print("\nv7: %d train + %d val = %d rows, %d speakers (base %d + new %d)"
          % (len(train), len(val), len(train) + len(val), next_index, base_n, next_index - base_n))
    print("    vat width %d (uniform)" % next(iter(widths)))
    if args.dry_run:
        print("dry run — nothing written")
        return

    os.makedirs(args.out, exist_ok=True)
    for name, part in (("train_op.txt", train), ("val_op.txt", val)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print("wrote %d rows -> %s" % (len(part), os.path.join(args.out, name)))

    # ⚠ #296, SECOND ATTEMPT. The first fix was a measurement in FORM ONLY and Janis
    # reopened it: both sides went through the same normalisation, so `identical` was True
    # by construction and the die() was unreachable. `read_corpus` drops every empty line,
    # `train` is literally `base[name] + new_train`, and re-reading the output then
    # re-applying the identical strip cannot disagree with it. A base carrying an interior
    # blank line, or no trailing newline, came out different on disk while the report said
    # true — measured, not argued.
    #
    # So the comparison is now against BYTES OF FILES, which is the only thing that can
    # actually differ: the base file as it sits on disk, and the leading bytes of what was
    # written. The digests are of those same bytes, so `sha256sum` on either file
    # reproduces them — the previous ones hashed joined lines and matched nothing.
    identity = {}
    for name in ("train_op.txt", "val_op.txt"):
        with open(os.path.join(args.base, name), "rb") as f:
            base_bytes = f.read()
        with open(os.path.join(args.out, name), "rb") as f:
            out_bytes = f.read()
        head = out_bytes[:len(base_bytes)]
        b = hashlib.sha256(base_bytes).hexdigest()
        o = hashlib.sha256(head).hexdigest()
        identity[name] = {"base_bytes": len(base_bytes), "sha256_base_file": b,
                          "sha256_out_leading_bytes": o, "identical": head == base_bytes}
        if head != base_bytes:
            # Unlike the first attempt, this die() is reachable — so it cleans up after
            # itself, the same invariant the licence refusal below keeps.
            for n2 in ("train_op.txt", "val_op.txt"):
                try:
                    os.remove(os.path.join(args.out, n2))
                except OSError:
                    pass
            die("BYTE-IDENTITY FAILED on %s: the output's leading %d bytes do not reproduce "
                "the base file. The corpus at %s would NOT be a legal warm-start donor. "
                "Partial write removed." % (name, len(base_bytes), args.out))
    print("byte-identity: verified against file bytes (%s)"
          % ", ".join("%s %d bytes" % (n, v["base_bytes"]) for n, v in identity.items()))

    # ⚠ The authoritative licence check needs the files on disk, so it runs AFTER the write —
    # which means a refusal here leaves filelists in `--out` and no `speakers.json`. That
    # half-corpus would then trip the "--out already holds a corpus" guard above and demand
    # `--force` on the retry, turning one honest refusal into two. So the partial write is
    # removed: the invariant this script promises is that a failure writes nothing.
    try:
        license_check([os.path.join(args.out, n) for n in ("train_op.txt", "val_op.txt")])
    except Exception:
        for n in ("train_op.txt", "val_op.txt"):
            try:
                os.remove(os.path.join(args.out, n))
            except OSError:
                pass
        print("licence wall REFUSED — partial write removed, %s left with no corpus"
              % args.out, file=sys.stderr)
        raise
    print("licence wall: accepted")
    merged_ns["n_spks"] = next_index
    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({"n_spks": next_index, **{k: v for k, v in merged_ns.items() if k != "n_spks"}},
                  f, indent=2)
    with open(os.path.join(args.out, "derivation_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base": args.base,
            "added": [{"dir": d, "speakers": s, "rows": r} for d, s, r in per_dir],
            "n_spks": next_index,
            "train_rows": len(train),
            "val_rows": len(val),
            "base_rows_byte_identical": identity,
            "note": ("Rows are appended, never relabelled. `base_rows_byte_identical` is "
                     "MEASURED from the written files, not asserted. The per-directory independence "
                     "gate readings live in each --add dir's own derivation_report.json and "
                     "are NOT pooled here."),
        }, f, indent=2)
    print("wrote %s" % os.path.join(args.out, "speakers.json"))


if __name__ == "__main__":
    main()
