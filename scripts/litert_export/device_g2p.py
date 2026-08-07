# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The DEVICE text front end — an independent replica of `matcha.text.op_g2p`.

This is the executable specification for the mobile port's G2P. There is no shipped
mobile app; front-end development has not started in earnest, so nothing here is a field
repair — it is the spec being made correct before anyone ports it.

D-C1, ON THE DEVICE. `kotlin_replica.phonemize` was
`DICT.get(token) or phon_word(token)`: a flat dictionary lookup with a neural fallback and
no apostrophe handling of any kind. The host had carried a contraction table since
2026-08-02 — that was D-C1, the finding whose absence poisoned corpus versions v1 through
v3 — and the replica never got it. Verified against the shipped assets rather than
inferred: **0 of 274,927 dictionary keys contain an apostrophe**, and `'` is absent from
the neural charset in `g2p_meta.json`, where unknown characters are silently skipped. So
every contraction took the exact path D-C1 named, and the failure was invisible because
the wrong answer arrived as a successful neural hit:

    don't -> dont -> dˈɔnt    (the model is trained on dˈoʊnt)
    we'll -> well -> wˈɛl     (identical to the word *well*)
    won't -> wont -> wˈɔnt    (the adjective, wrong vowel)

Measured at 0.37% of v3 tokens and concentrated in dialogue, which is 64% of the corpus.

THE SPLIT THIS MODULE MAKES, and why it is not just "copy the table over":

  * The TABLES are data, and ship as an asset (`g2p_contractions.json`) emitted by
    `convert_vat.py` from `matcha.text.op_g2p`'s own definitions. One source of truth, so
    the device cannot fall behind the host by omission — which is the precise shape of the
    defect above. A device that cannot find the asset fails hard; it does NOT fall back to
    the flat lookup, because a silent fallback is how this stayed hidden for a corpus and
    a half.
  * The ALGORITHM is reimplemented here, deliberately, and imports nothing from `matcha`.
    A replica that called the host's phonemizer would prove only that a function equals
    itself. What the Kotlin port has to get right is the algorithm — lookup order,
    decomposition, the possessive allomorph, which apostrophes are part of a word and
    which are quotation marks — and that is what gate G7 in `convert_vat.py` checks, by
    comparing phoneme STRINGS between this module and `op_g2p` over `PARITY_PROBES`.

Torch-free and litert-lazy: the neural OOV graph is loaded only when a word actually
misses the dictionary, so the parity gate and its test run without a GPU.
"""

import gzip
import json
import os
import re

from unidecode import unidecode

# --- asset resolution ---------------------------------------------------
#
# Same two-root search as `kotlin_replica._find`, and for the same reason (F-M3): the
# conversion outputs and the vendored assets live in different directories, and the G2P
# dictionary ships gzipped under a name the bare-name opener missed.
ASSETS = os.environ.get("SONORA_LITERT_ASSETS",
                        "/data/models/litert-community/Matcha-TTS")

CONTRACTIONS_ASSET = "g2p_contractions.json"

# The manifest key that records which front end the corpus was derived with. The host
# reads config.json and nothing else, so a setting that changes the phonemes has to be
# in it — see F-H2 for the same argument about the control surface.
G2P_MANIFEST_KEY = "g2p"


def find_asset(name, *roots):
    """First existing path for `name` across `roots`, accepting a `.gz` sibling."""
    for root in roots:
        if not root:
            continue
        for cand in (os.path.join(root, name), os.path.join(root, name + ".gz")):
            if os.path.isfile(cand):
                return cand
    return None


# --- normalization ------------------------------------------------------
#
# `op_g2p.phonemize` runs english_cleaners2's pre-normalization minus espeak itself, and
# the device has to run the SAME chain in the SAME order or the two front ends disagree on
# text before they ever disagree on phonemes. Reimplemented here from the observable
# behaviour of matcha.text.cleaners rather than imported (see the module docstring).
#
# A Kotlin port needs an ASCII fold equivalent to `unidecode` for step one. That is a real
# porting dependency, not an incidental one: curly quotes and accented borrowings are
# ordinary in the book lane, and folding them differently changes tokenization.
_ABBREVIATIONS = [
    (re.compile(rf"\b{short}\.", re.IGNORECASE), full)
    for short, full in (
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
        ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    )
]
_BRACKETS_RE = re.compile(r"[\[\]\(\)\{\}]")
_WHITESPACE_RE = re.compile(r"\s+")
_HYPHEN_RE = re.compile(r"[-–]+")

# Word tokens carry apostrophes; punctuation passes through only if it is vocab
# punctuation. Must match op_g2p._TOKEN_RE exactly — a mark this drops and the host keeps
# is a missing pause on device, which reads as "the model ignores commas".
_TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:—…\"«»“”¡¿]")
_WORD_RE = re.compile(r"[a-z']+")
_COMBINING_TILDE = "̃"


def normalize(text):
    """Text -> the normalized string the tokenizer sees. Mirrors op_g2p.phonemize's head."""
    text = unidecode(text)
    text = text.lower()
    for regex, replacement in _ABBREVIATIONS:
        text = regex.sub(replacement, text)
    text = _BRACKETS_RE.sub("", text)
    # Hyphens and dashes between words are separators, as in espeak.
    text = _HYPHEN_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text)


# --- the possessive allomorph -------------------------------------------
#
# Table-driven from the asset, but the RULE is code: which of the three "'s" endings a
# base takes is decided by its final phoneme, and enumerating that per-noun is exactly the
# enumeration the productive class exists to avoid.
def possessive_suffix(base_ipa, classes):
    """The 's allomorph for a base's final phoneme: ᵻz / s / z."""
    tail = base_ipa.rstrip(classes["stress_marks"])
    if tail.endswith(tuple(classes["sibilant"])):
        return classes["after_sibilant"]
    if tail.endswith(tuple(classes["voiceless"])):
        return classes["after_voiceless"]
    return classes["default"]


class DeviceG2P:
    """Dictionary-primary, neural-fallback phonemizer as the mobile host must implement it.

    Args:
        assets_dir: Directory holding g2p_dict.txt(.gz), g2p_meta.json and the neural
            graph. Defaults to $SONORA_LITERT_ASSETS.
        artifacts_dir: Export output directory, searched FIRST — a freshly converted
            `g2p_contractions.json` there must win over a stale vendored one.
        tables: Pre-loaded contraction tables, used by gate G7 so it can certify the
            exact bytes it is about to write instead of a file that does not exist yet.
        use_neural_oov: Load the DeepPhonemizer graph for dictionary misses.
    """

    def __init__(self, assets_dir=None, artifacts_dir=None, tables=None,
                 use_neural_oov=True):
        self.assets_dir = os.path.abspath(assets_dir or ASSETS)
        self.artifacts_dir = os.path.abspath(artifacts_dir) if artifacts_dir else None
        self.use_neural_oov = use_neural_oov
        self.tables = tables if tables is not None else self._load_tables()
        self._check_tables()

        dict_path = find_asset("g2p_dict.txt", self.artifacts_dir, self.assets_dir)
        if dict_path is None:
            raise SystemExit(
                f"!! device G2P: no g2p_dict.txt in {self.artifacts_dir or '-'} or "
                f"{self.assets_dir}\n   (vendored asset; ships as g2p_dict.txt.gz)")
        opener = gzip.open if dict_path.endswith(".gz") else open
        self.dict = {}
        with opener(dict_path, "rt", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    word, ipa = line.rstrip("\n").split("\t", 1)
                    # The one known dictionary gap: a combining nasal tilde in 7 French
                    # loanwords, outside the locked 178-symbol vocab. The host strips it;
                    # a device that does not would emit a symbol with no embedding row.
                    self.dict[word] = ipa.replace(_COMBINING_TILDE, "")

        self._neural = None
        self._neural_meta = None
        self.oov_words = set()
        # Apostrophe words that neither the table nor decomposition resolved, and that
        # fell through to the bare-letters guess (mostly names: O'Brien, d'Artagnan).
        # Tracked apart from neural hits on purpose: counting them together is how D-C1
        # stayed invisible through two corpus versions.
        self.apostrophe_fallback_words = set()

    # -- tables ----------------------------------------------------------
    def _load_tables(self):
        path = find_asset(CONTRACTIONS_ASSET, self.artifacts_dir, self.assets_dir)
        if path is None:
            raise SystemExit(
                f"!! device G2P: {CONTRACTIONS_ASSET} is missing.\n"
                f"   Looked in: {self.artifacts_dir or '-'}\n"
                f"          and: {self.assets_dir}\n"
                "   Produced by convert_vat.py from matcha.text.op_g2p.\n\n"
                "   This is NOT recoverable by falling back to a plain dictionary lookup.\n"
                "   That fallback is D-C1: the dictionary holds no apostrophe keys and the\n"
                "   neural charset has no \"'\", so every contraction resolves to its\n"
                "   apostrophe-stripped letters and arrives looking like a clean hit —\n"
                "   we'll becomes wˈɛl, which is the word *well*.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _check_tables(self):
        missing = [k for k in ("contractions", "clitics", "possessive")
                   if not self.tables.get(k)]
        if missing:
            raise SystemExit(
                f"!! device G2P: {CONTRACTIONS_ASSET} is missing {missing}. "
                "Re-run convert_vat.py.")
        if self.tables.get("homographs"):
            # The host can be constructed with homographs=True (D-M4). If a derivation
            # ever is, the device needs the resolver AND its evidence rules, which are not
            # a table — so this must fail at construction rather than quietly phonemize
            # `live` as the adjective the corpus was trained away from.
            raise SystemExit(
                "!! device G2P: this asset was exported from a homograph-resolving host "
                "(D-M4 is ON).\n   The device has no context resolver, so the two front "
                "ends cannot agree.\n   Port matcha/text/homographs.py before exporting "
                "with homographs enabled.")

    # -- lookup ----------------------------------------------------------
    def _neural_init(self):
        from ai_edge_litert.compiled_model import CompiledModel

        meta_path = find_asset("g2p_meta.json", self.artifacts_dir, self.assets_dir)
        graph_path = find_asset("dp_g2p_matcha_fp16.tflite",
                                self.artifacts_dir, self.assets_dir)
        if meta_path is None or graph_path is None:
            raise SystemExit(
                "!! device G2P: the neural OOV graph or its meta is missing "
                f"({self.assets_dir}). Pass use_neural_oov=False to run dictionary-only.")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self._neural_meta = {
            "c2i": {k: v for k, v in meta["char2idx"].items() if len(k) == 1},
            "i2p": {int(k): v for k, v in meta["idx2ph"].items()},
            "rep": meta["char_repeats"],
            "start": meta["start"],
            "end": meta["end"],
            "maxt": meta["MAXT"],
            "special": set(meta["special"]),
        }
        self._neural = CompiledModel.from_file(graph_path)

    def neural_word(self, word):
        """MatchaG2P.phonemizeWord: one OOV word through the DeepPhonemizer graph."""
        import numpy as np

        if self._neural is None:
            self._neural_init()
        meta = self._neural_meta
        ids = [meta["start"]]
        for char in word:
            if char in meta["c2i"]:
                ids += [meta["c2i"][char]] * meta["rep"]
        ids.append(meta["end"])
        length = min(len(ids), meta["maxt"])
        padded = [ids[i] if i < length else 0 for i in range(meta["maxt"])]

        model = self._neural
        signatures = model.get_signature_list()
        key = list(signatures)[0]
        in_details = model.get_input_tensor_details(key)
        out_details = model.get_output_tensor_details(key)
        input_buffers = model.create_input_buffers(0)
        output_buffers = model.create_output_buffers(0)
        name = signatures[key]["inputs"][0]
        input_buffers[0].write(
            np.ascontiguousarray([padded], dtype=np.dtype(in_details[name]["dtype"])))
        model.run_by_index(0, input_buffers, output_buffers)
        oname = signatures[key]["outputs"][0]
        detail = out_details[oname]
        count = int(np.prod(detail["shape"]))
        logits = output_buffers[0].read(
            count, np.dtype(detail["dtype"])).reshape(detail["shape"])[0]

        pieces, previous = [], -1
        for t in range(length):
            best = int(logits[t].argmax())
            if best == previous:
                continue
            previous = best
            phoneme = meta["i2p"].get(best)
            if phoneme is None or phoneme in meta["special"] or best == 0:
                continue
            pieces.append("".join(ch for ch in phoneme if ch != "-"))
        return "".join(pieces).replace(_COMBINING_TILDE, "")

    def _plain_word(self, word):
        """Dictionary-then-neural for a word with no apostrophes, or None."""
        ipa = self.dict.get(word)
        if ipa is not None:
            return ipa
        if self.use_neural_oov:
            ipa = self.neural_word(word)
            if ipa:
                return ipa
        return None

    def _apostrophe_word(self, word):
        """Contraction/clitic resolution, or None if neither applies.

        Order matters: the irregular table wins over decomposition, because the forms
        decomposition gets wrong are the whole reason the table exists (don't is /doʊnt/,
        not do + n't; we're is /wɪɹ/, not we + ɚ).
        """
        fixed = self.tables["contractions"].get(word)
        if fixed is not None:
            return fixed

        # A trailing apostrophe is the plural possessive (cats'), which adds no phoneme.
        if word.endswith("'"):
            return self._plain_word(word[:-1])

        for clitic, suffix_ipa in self.tables["clitics"].items():
            if word.endswith(clitic) and len(word) > len(clitic):
                base = self._plain_word(word[: -len(clitic)])
                if base:
                    return base + suffix_ipa
                return None

        if word.endswith("'s") and len(word) > 2:
            base = self._plain_word(word[:-2])
            if base:
                return base + possessive_suffix(base, self.tables["possessive"])
        return None

    def phonemize_word(self, word):
        """Lower-case word -> IPA, or None when unresolvable. Apostrophes allowed."""
        ipa = self.dict.get(word)
        if ipa is not None:
            return ipa

        if "'" in word:
            ipa = self._apostrophe_word(word)
            if ipa:
                return ipa
            # Not a contraction we know. The bare-letters guess is still the best
            # available, but it is recorded so it cannot hide inside the hit rate.
            bare = word.replace("'", "")
            if bare:
                ipa = self._plain_word(bare)
                if ipa:
                    self.apostrophe_fallback_words.add(word)
                    return ipa

        # Never send an apostrophe word to the neural graph: the charset has no "'", the
        # character is silently skipped, and the result is a confident wrong answer.
        if self.use_neural_oov and "'" not in word:
            ipa = self.neural_word(word)
            if ipa:
                return ipa

        self.oov_words.add(word)
        return None

    def phonemize(self, text):
        """Normalized English text -> espeak-style IPA string."""
        tokens = _TOKEN_RE.findall(normalize(text))
        out, first = [], True
        for token in tokens:
            if not _WORD_RE.fullmatch(token):
                out.append(token)
                continue
            # A leading apostrophe is usually an opening quote and is dropped — except on
            # the archaic forms where it belongs to the word ('tis, 'em), which is why the
            # table is consulted before the strip. A trailing one needs no special case:
            # plural possessive and closing quote both resolve to the bare base.
            word = token if token in self.tables["contractions"] else token.lstrip("'")
            if not word or word == "'":
                continue
            # Letters pass through verbatim when nothing resolves. This does NOT surface
            # in a vocab check — espeak IPA reuses ASCII letters, so a-z is inside the
            # 178-symbol vocab and an unphonemized word validates clean. `oov_words` is
            # the only signal, and every caller must read it.
            ipa = self.phonemize_word(word) or word
            if not first:
                out.append(" ")
            out.append(ipa)
            first = False
        return "".join(out)


# --- the parity corpus --------------------------------------------------
#
# Gate G7 and tests/test_device_g2p_parity.py both read this. Every line is here because
# it exercises a way the two front ends can disagree, not because it is representative
# prose. The contraction table itself is probed separately and exhaustively — the gate
# builds one sentence per entry — so a table that grows is covered without editing this.
PARITY_PROBES = (
    # The three D-C1 exemplars, in context.
    "Don't go — we'll be late, and it won't matter.",
    "He's here, she'd know, they've gone, I'm sure, you're right.",
    # Productive clitics on bases the table does not enumerate.
    "The cat's dish, James's hat, the horses' stalls, the church's bell.",
    "Dickens's world, the fox's den, the lamp's glow, the moth's wing.",
    # Voiceless / voiced / sibilant allomorph boundaries.
    "The cat's, the dog's, the horse's, the judge's, the cliff's, the myth's.",
    # Apostrophes that are NOT part of a word.
    "'Hello,' he said, 'it is late.'",
    "The boys' coats and the ladies' gloves.",
    # Archaic and fixed forms — ordinary in the book lane.
    "'Tis a fine morning, and 'twas brillig; hand 'em over by o'clock.",
    "Ma'am, ne'er a word, e'er again, o'er the hill, y'all.",
    # Apostrophe names: no table entry, no clean decomposition. These take the
    # bare-letters fallback on BOTH sides, and the gate's job is that they take it the
    # same way rather than one side reaching for the neural graph.
    "O'Brien and d'Artagnan rode with Ma'evan.",
    # Abbreviation expansion, which runs before tokenization.
    "Mr. Smith and Dr. Jones walked to St. Paul's with Col. Reed.",
    # Brackets, hyphens and dashes as separators.
    "A well-known (and much-loved) tune — the second-best kind.",
    # Non-ASCII the fold has to flatten identically.
    "The naïve café serves crème brûlée — “quite good,” said the maître d'.",
    # Punctuation that must survive to the symbol sequence.
    "Wait… what? Yes; no: maybe! «Alors» ¿verdad? ¡Claro!",
    # Dictionary misses, so the neural path is compared too.
    "Zyzzyqx blorptastic frunge.",
    # Plain prose, as the control.
    "The quick brown fox jumps over the lazy dog.",
    "Hello, this is Matcha running on the mobile GPU.",
)


def probe_sentences(tables):
    """PARITY_PROBES plus one sentence per contraction-table entry.

    Driven from the table so an entry added to `op_g2p._CONTRACTIONS` is probed by the
    next export without anyone remembering to extend the list here. Framed as a sentence
    rather than a bare word because tokenization is half of what can differ.
    """
    probes = list(PARITY_PROBES)
    for word in sorted(tables["contractions"]):
        probes.append(f"And then {word} said the thing.")
    return probes


def compare(host_phonemize, device_phonemize, sentences):
    """Run both front ends over `sentences` -> list of (sentence, host, device) mismatches."""
    bad = []
    for sentence in sentences:
        want = host_phonemize(sentence)
        got = device_phonemize(sentence)
        if want != got:
            bad.append((sentence, want, got))
    return bad
