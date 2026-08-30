"""Shared machinery for the blind ear benches.

Two benches use this and they differ ONLY in what they contrast:

* `render_ear_ab.py`      — same conditioning, two CHECKPOINTS.
* `render_ear_lane_control.py` — one checkpoint, two CONDITIONING settings.

Everything else is common, and it lives here rather than in both files because the
parts that must not drift are the parts a copy would drift first: the seed discipline,
the refusal on incomparable checkpoints, and the opaque naming that makes the test blind.

⚠ THE SEED IS THE EXPERIMENT. `synthesise` samples at temperature 0.667, so two renders
of one prompt differ even from ONE checkpoint under ONE setting. Both sides of a pair
are rendered under the SAME seed, derived from the PAIR key rather than from the side,
so the only thing varying inside a pair is the thing under test.

⚠ crc32, NOT hash(): str hashing is salted per process, so a render resumed in a second
process would reseed the remaining work and silently turn sampling noise into the
measured effect.
"""

import csv
import hashlib
import json
import zlib
from pathlib import Path

import soundfile as sf
import torch

from matcha import delivery
from matcha.cli import (detect_lane, load_matcha, load_vocoder_24k,
                        process_text_for_lane, to_waveform)

DEVICE = torch.device("cpu")

# Fixed render settings, IDENTICAL for every side of every pair in every bench. Taken
# from the Vocalizer/derisk defaults so the benches and the interactive surface agree
# about what "neutral settings" means. Changing one invalidates comparison against every
# previously rendered test, so they are constants here and not flags.
N_TIMESTEPS = 10
TEMPERATURE = 0.667
LENGTH_SCALE = 1.0
GUIDANCE = 1.0


def opaque(pair_key, side_key, salt):
    """The served filename. Carries no information about what the clip is."""
    return hashlib.sha1(f"{salt}|{pair_key}|{side_key}".encode()).hexdigest()[:16]


def seed_for(pair_key, base):
    """One seed per PAIR, shared by both sides. See the module docstring."""
    return base + zlib.crc32(pair_key.encode()) % 65536


def require_comparable(ckpts):
    """Refuse a comparison that cannot be one.

    Different speaker-table sizes mean one speaker id is a different voice on each side,
    and a different `vat_dim` means the conditioning vector means different things. Both
    render happily and produce a confident wrong verdict, which is what this prevents.
    """
    shapes = {n: detect_lane(p) for n, p in ckpts.items()}
    if len(set(shapes.values())) != 1:
        for n, s in shapes.items():
            print(f"  {n}: lane={s[0]} n_spks={s[1]} vat_dim={s[2]}")
        raise SystemExit("REFUSING: checkpoints disagree on lane/n_spks/vat_dim.")
    lane_kind, n_spks, vat_dim = next(iter(shapes.values()))
    if lane_kind != "vat" or vat_dim != delivery.VAT_DIM:
        raise SystemExit(
            f"REFUSING: these benches render contract-v2 checkpoints (lane=vat, "
            f"vat_dim={delivery.VAT_DIM}); got lane={lane_kind} vat_dim={vat_dim}.")
    return lane_kind, n_spks, vat_dim


class Bench:
    """Renders sides on demand, loading each checkpoint at most once."""

    def __init__(self, out, salt, seed, lane_kind):
        self.out = Path(out)
        (self.out / "clips").mkdir(parents=True, exist_ok=True)
        self.salt, self.seed, self.lane_kind = salt, seed, lane_kind
        self.vocoder, self.sample_rate = load_vocoder_24k(DEVICE)
        self._model = self._loaded = None
        self.key, self.written = {}, 0

    def _model_for(self, ckpt):
        if self._loaded != ckpt:
            print(f"== loading {ckpt}")
            self._model, self._loaded = load_matcha("custom", ckpt, DEVICE), ckpt
        return self._model

    def render(self, pair_key, side_key, ckpt, text, spk, vat, lane, label):
        """Render one side of one pair. Returns its opaque id.

        `label` is what DISTINGUISHES this side inside its pair — a checkpoint name for
        an A/B bench, a conditioning name for a contrast bench. It goes in the key, not
        in the filename, and the unblinder reports by it.
        """
        name = opaque(pair_key, side_key, self.salt)
        self.key[name] = {"label": label, "pair": pair_key, "side": side_key}
        path = self.out / "clips" / f"{name}.wav"
        if path.exists():
            return name
        enc = process_text_for_lane(1, text, DEVICE, self.lane_kind)
        vec = delivery.vat_vector(*vat, lane)
        torch.manual_seed(seed_for(pair_key, self.seed))
        with torch.no_grad():
            o = self._model_for(ckpt).synthesise(
                enc["x"], enc["x_lengths"], n_timesteps=N_TIMESTEPS,
                temperature=TEMPERATURE, length_scale=LENGTH_SCALE,
                spks=torch.tensor([spk], dtype=torch.long),
                vat=torch.tensor([vec]), guidance=GUIDANCE)
            wav = to_waveform(o["mel"], self.vocoder, None)
        sf.write(path, wav.cpu().numpy(), self.sample_rate, "PCM_24")
        self.written += 1
        if self.written % 10 == 0:
            print(f"   {self.written} clips")
        return name

    def write(self, test_name, sets, served, meta, key_out=None):
        """items.json (served, blind) and the key (NOT served).

        ⚠ THE KEY GOES OUTSIDE `out` BY DEFAULT. The app mounts the test directory
        whole, so a key inside it would be in the container and blinding would rest on
        the app not opening a path it can reach.
        """
        # ⚠⚠ A COLLECTED VERDICT IS A REFERENCE INTO items.json. It records "the
        # listener chose A", and only this manifest says which clip A was. Rewriting it
        # with a different A/B assignment does not corrupt anything visibly — it
        # silently RE-POINTS every verdict already collected, and the unblinder goes on
        # printing confident numbers for a test that no longer happened. Regenerating a
        # manifest is normal and safe when the assignment is unchanged; changing it
        # under existing verdicts is not, so this refuses rather than warns.
        manifest = {"test": test_name, "sample_rate": self.sample_rate,
                    "sets": sets, "items": served}
        old = self.out / "items.json"
        vcsv = self.out / "verdicts" / "verdicts.csv"
        if old.is_file() and vcsv.is_file():
            with vcsv.open(newline="", encoding="utf-8") as f:
                judged = {r["item"] for r in csv.DictReader(f) if r.get("choice")}
            if judged:
                was = {i["id"]: (i["A"], i["B"])
                       for i in json.loads(old.read_text())["items"]}
                now = {i["id"]: (i["A"], i["B"]) for i in served}
                moved = sorted(k for k in judged
                               if k in was and k in now and was[k] != now[k])
                if moved:
                    raise SystemExit(
                        f"REFUSING: {len(moved)} already-judged pairs would change "
                        f"sides, which re-points every verdict collected for them "
                        f"(e.g. {moved[:3]}). Render to a NEW --out, or bump --salt for "
                        f"a deliberate re-blinding.")
        old.write_text(json.dumps(manifest, indent=2))
        kp = Path(key_out) if key_out else (
            self.out.parent / "_keys" / f"{self.out.name}.key.json")
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(json.dumps(
            {"test_dir": str(self.out), "salt": self.salt, "clips": self.key} | meta,
            indent=2))
        (self.out / "render_meta.json").write_text(json.dumps(
            {"n_timesteps": N_TIMESTEPS, "temperature": TEMPERATURE,
             "length_scale": LENGTH_SCALE, "guidance": GUIDANCE, "seed": self.seed,
             "sample_rate": self.sample_rate, "pairs": len(served),
             "clips_written": self.written} | meta, indent=2))
        print(f"\n  {len(served)} pairs -> {self.out}")
        print(f"  {self.written} clips rendered this run")
        print(f"  unblinding key (keep out of the app's mount): {kp}")
        return kp
