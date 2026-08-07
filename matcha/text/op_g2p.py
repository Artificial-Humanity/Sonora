"""Espeak-free G2P for training filelists (north star §8.3).

Primary: the OpenPhonemizer 275k espeak-IPA dictionary (Clear BSD) from
litert-community/Matcha-TTS. OOV fallback: the DeepPhonemizer (MIT) TFLite
graph from the same repo, run via ai-edge-litert when available; otherwise
OOV words are reported so the caller can decide.

Output is espeak-style IPA over the locked 178-symbol vocab (symbols.py).
The one known dictionary gap — a combining nasal tilde (U+0303) in 7 French
loanwords — is stripped (validated 2026-07-12: with that rule, 99.997% of
entries phonemize entirely into the locked vocab).

Asset directory resolution order: explicit argument, $SONORA_G2P_ASSETS,
then the first _ASSET_CANDIDATES entry that exists on disk.

The old default — ../Reference/models/... relative to the repo root — was the
umbrella-workspace layout, deleted 2026-07-22 when the repos went flat. It was
still the hard-coded default on 2026-08-01, so every caller without the env var
died in __init__ on a missing g2p_dict.txt.gz. The assets live under /data with
the other model weights; that path is checked first now, and the old one is kept
last so a machine still carrying the umbrella layout keeps working.
"""

import gzip
import json
import os
import re

from matcha.text.cleaners import (
    collapse_whitespace,
    convert_to_ascii,
    expand_abbreviations,
    lowercase,
    remove_brackets,
)
from matcha.text.homographs import resolve as resolve_homograph
from matcha.text.symbols import symbols

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASSET_CANDIDATES = (
    "/data/models/litert-community/Matcha-TTS",
    os.path.join(_REPO_ROOT, "..", "Reference", "models", "litert-community", "Matcha-TTS"),
)


def _default_assets():
    """First candidate that actually holds the dictionary, else the first one.

    Falling back to candidate 0 rather than to None keeps the failure legible: a
    missing-file error naming /data/models beats a None that surfaces later as a
    confusing TypeError.
    """
    for cand in _ASSET_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "g2p_dict.txt.gz")):
            return cand
    return _ASSET_CANDIDATES[0]

# Word tokens may carry apostrophes; everything else passes through only if
# it is vocab punctuation (hyphens/brackets are separators, handled upstream).
_TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:—…\"«»“”¡¿]")
_WORD_RE = re.compile(r"[a-z']+")
_COMBINING_TILDE = "̃"

_VOCAB = set(symbols)

# --- Apostrophe handling ------------------------------------------------
#
# The dictionary holds no apostrophe keys at all (measured 2026-08-02: 0 of
# 274,927 entries) and the neural fallback's charset has no "'" — chars absent
# from c2i are silently skipped in _neural_word. So before this table existed,
# an apostrophe word was phonemized from its apostrophe-stripped letters, and
# the failure was invisible because the result counted as a neural "hit":
#   don't -> dont -> dˈɔnt   (should be dˈoʊnt)
#   we'll -> well -> wˈɛl    (identical to the word *well*)
#   won't -> wont -> wˈɔnt   (the adjective, wrong vowel)
# Measured at 0.37% of v3 tokens, concentrated in dialogue.
#
# Two mechanisms replace that. _CONTRACTIONS is the closed irregular class,
# where splitting at the apostrophe gives the wrong answer (don't is /doʊnt/,
# not do + n't; we're is /wɪɹ/, not we + ɚ). _CLITICS is the productive class,
# which attaches to any base — possessives on arbitrary nouns and names work
# without enumerating them.
#
# Every IPA string below follows this dictionary's own espeak conventions,
# cross-checked against rime-mates that are in the dictionary:
#   dote dˈoʊt · own ˈoʊn · cant kˈænt · heel hˈiːl · their ðɛɹ · your jʊɹ
#   its ɪts · hes hiːz · lets lˈɛts · student stˈuːdənt (syllabic n = ənt)
_CONTRACTIONS = {
    # negation
    "don't": "dˈoʊnt", "won't": "wˈoʊnt", "can't": "kˈænt",
    "shan't": "ʃˈænt", "ain't": "ˈeɪnt", "didn't": "dˈɪdənt",
    "doesn't": "dˈʌzənt", "isn't": "ˈɪzənt", "wasn't": "wˈɑːzənt",
    "aren't": "ˈɑːɹənt", "weren't": "wˈɜːnt", "hasn't": "hˈæzənt",
    "haven't": "hˈævənt", "hadn't": "hˈædənt", "wouldn't": "wˈʊdənt",
    "shouldn't": "ʃˈʊdənt", "couldn't": "kˈʊdənt", "mustn't": "mˈʌsənt",
    "needn't": "nˈiːdənt", "mightn't": "mˈaɪtənt", "oughtn't": "ˈɔːtənt",
    "daren't": "dˈɛɹənt",
    # pronoun + auxiliary
    "i'm": "ˈaɪm", "i'll": "ˈaɪl", "i'd": "ˈaɪd", "i've": "ˈaɪv",
    "you're": "jʊɹ", "you'll": "jˈuːl", "you'd": "jˈuːd", "you've": "jˈuːv",
    "he's": "hiːz", "he'll": "hiːl", "he'd": "hiːd",
    "she's": "ʃiːz", "she'll": "ʃiːl", "she'd": "ʃiːd",
    "it's": "ɪts", "it'll": "ˈɪtəl", "it'd": "ˈɪtəd",
    "we're": "wɪɹ", "we'll": "wiːl", "we'd": "wiːd", "we've": "wiːv",
    "they're": "ðɛɹ", "they'll": "ðeɪl", "they'd": "ðeɪd", "they've": "ðeɪv",
    # demonstrative / interrogative + auxiliary
    "that's": "ðæts", "that'll": "ðˈætəl", "there's": "ðɛɹz",
    "there'll": "ðˈɛɹəl", "here's": "hˈɪɹz", "what's": "wˌʌts",
    "what'll": "wˈʌtəl", "who's": "hˈuːz", "who'd": "hˈuːd",
    "who'll": "hˈuːl", "who've": "hˈuːv", "let's": "lˈɛts",
    # archaic and fixed forms — common in the book lane (Dickens, LibriVox)
    "'tis": "tˈɪz", "'twas": "twˈʌz", "'twould": "twˈʊd", "'em": "əm",
    "o'clock": "əklˈɑːk", "ma'am": "mˈæm", "y'all": "jˈɔːl",
    "e'er": "ˈɛɹ", "ne'er": "nˈɛɹ", "o'er": "ˈoʊɹ",
}

# Productive suffixes: base is phonemized normally, these are appended.
# "'s" is not here — it takes the allomorph rule in _possessive_suffix.
_CLITICS = {
    "n't": "nt", "'ll": "l", "'ve": "v", "'re": "ɚ", "'d": "d", "'m": "m",
}

# Allomorph classes for "'s", in this dictionary's spelling. Verified against
# the corpus: horses hˈɔːɹsᵻz, churches tʃˈɜːtʃᵻz (14,026 entries end "ᵻz",
# so the sibilant form is ᵻz, not ɪz); cats kˈæts; dogs dˈɑːɡz.
_SIBILANT_ENDINGS = ("tʃ", "dʒ", "s", "z", "ʃ", "ʒ")
_VOICELESS_ENDINGS = ("p", "t", "k", "f", "θ")
_STRESS_MARKS = "ˈˌː"


def _possessive_suffix(base_ipa):
    """The 's allomorph for a base's final phoneme: ᵻz / s / z."""
    tail = base_ipa.rstrip(_STRESS_MARKS)
    if tail.endswith(_SIBILANT_ENDINGS):
        return "ᵻz"
    if tail.endswith(_VOICELESS_ENDINGS):
        return "s"
    return "z"


def contraction_tables():
    """The apostrophe tables as serializable data, for export to the mobile front end.

    These are the SOURCE OF TRUTH and the device gets a copy, rather than the port
    transcribing them — D-C1 is a defect of omission, and a table that has to be kept in
    sync by hand is the same defect waiting to happen. `convert_vat.py` writes this to
    `g2p_contractions.json` beside the graphs, gate G7 certifies that the device front end
    reproduces this one's phonemes from it, and `scripts/litert_export/device_g2p.py`
    consumes it.

    The possessive block is the RULE's inputs rather than a per-word list, because "'s"
    is productive: it attaches to any noun, so enumerating it is what the allomorph
    classes exist to avoid.
    """
    return {
        "contractions": dict(_CONTRACTIONS),
        "clitics": dict(_CLITICS),
        "possessive": {
            "sibilant": list(_SIBILANT_ENDINGS),
            "voiceless": list(_VOICELESS_ENDINGS),
            "after_sibilant": "ᵻz",
            "after_voiceless": "s",
            "default": "z",
            "stress_marks": _STRESS_MARKS,
        },
    }


class OpenPhonemizerG2P:
    """Dictionary-primary, neural-fallback espeak-IPA phonemizer."""

    def __init__(self, assets_dir=None, use_neural_oov=True, homographs=False):
        # `homographs` is OFF by default and that is deliberate (review finding D-M4).
        # Turning it on changes the phonemes of words the dictionary currently gets
        # wrong, which is a corpus change wherever it is applied — so it is the caller's
        # explicit choice, never a side effect of upgrading. Measure with
        # scripts/measure_homographs.py before enabling it on a derivation.
        self.homographs = homographs
        self.assets_dir = os.path.abspath(
            assets_dir or os.environ.get("SONORA_G2P_ASSETS") or _default_assets()
        )
        self.dict = {}
        dict_path = os.path.join(self.assets_dir, "g2p_dict.txt.gz")
        with gzip.open(dict_path, "rt", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    word, ipa = line.rstrip("\n").split("\t", 1)
                    self.dict[word] = ipa.replace(_COMBINING_TILDE, "")
        self._neural = None
        self._neural_meta = None
        self.use_neural_oov = use_neural_oov
        self.stats = {
            "dict_hits": 0,
            "neural_hits": 0,
            "oov_misses": 0,
            "contraction_hits": 0,
            "apostrophe_fallbacks": 0,
            "homograph_flips": 0,
        }
        # (word, sense) -> count, so a report can show WHICH words moved rather than
        # only how many did. Populated whether or not `homographs` is on: with the
        # flag off the decisions are counted and discarded, which is what makes a
        # dry-run measurement possible without touching the output.
        self.homograph_decisions = {}
        self.oov_words = set()
        # Apostrophe words that neither table nor decomposition resolved and
        # that fell through to the bare-letters path (mostly names: O'Brien).
        # Tracked separately so the residue is visible in the report instead
        # of counting as a clean neural hit, which is how D-C1 stayed hidden.
        self.apostrophe_fallback_words = set()

    def _neural_init(self):
        import numpy as np  # noqa: F401  (hard dep of the tflite path only)
        from ai_edge_litert.compiled_model import CompiledModel

        with open(os.path.join(self.assets_dir, "g2p_meta.json"), encoding="utf-8") as f:
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
        self._neural = CompiledModel.from_file(
            os.path.join(self.assets_dir, "dp_g2p_matcha_fp16.tflite")
        )

    def _neural_word(self, word):
        """DeepPhonemizer TFLite forward for one OOV word (kotlin_replica logic)."""
        import numpy as np

        if self._neural is None:
            self._neural_init()
        m = self._neural_meta
        ids = [m["start"]]
        for c in word:
            if c in m["c2i"]:
                ids += [m["c2i"][c]] * m["rep"]
        ids.append(m["end"])
        length = min(len(ids), m["maxt"])
        padded = [ids[i] if i < length else 0 for i in range(m["maxt"])]
        model = self._neural
        signatures = model.get_signature_list()
        key = list(signatures)[0]
        in_details = model.get_input_tensor_details(key)
        out_details = model.get_output_tensor_details(key)
        input_buffers = model.create_input_buffers(0)
        output_buffers = model.create_output_buffers(0)
        name = signatures[key]["inputs"][0]
        input_buffers[0].write(
            np.ascontiguousarray([padded], dtype=np.dtype(in_details[name]["dtype"]))
        )
        model.run_by_index(0, input_buffers, output_buffers)
        oname = signatures[key]["outputs"][0]
        detail = out_details[oname]
        count = int(np.prod(detail["shape"]))
        logits = output_buffers[0].read(count, np.dtype(detail["dtype"])).reshape(detail["shape"])[0]
        pieces, previous = [], -1
        for t in range(length):
            best = int(logits[t].argmax())
            if best == previous:
                continue
            previous = best
            phoneme = m["i2p"].get(best)
            if phoneme is None or phoneme in m["special"] or best == 0:
                continue
            pieces.append("".join(ch for ch in phoneme if ch != "-"))
        return "".join(pieces).replace(_COMBINING_TILDE, "")

    def _apostrophe_word(self, word):
        """Contraction/clitic resolution, or None if neither applies.

        Order matters: the irregular table wins over decomposition, because
        the whole point of the table is the forms decomposition gets wrong.
        """
        fixed = _CONTRACTIONS.get(word)
        if fixed is not None:
            return fixed

        # Trailing apostrophe = plural possessive (cats'), which adds nothing.
        if word.endswith("'"):
            return self._plain_word(word[:-1])

        for clitic, suffix_ipa in _CLITICS.items():
            if word.endswith(clitic) and len(word) > len(clitic):
                base = self._plain_word(word[: -len(clitic)])
                if base:
                    return base + suffix_ipa
                return None

        if word.endswith("'s") and len(word) > 2:
            base = self._plain_word(word[:-2])
            if base:
                return base + _possessive_suffix(base)
        return None

    def _plain_word(self, word):
        """Dictionary-then-neural lookup for a word with no apostrophes.

        Returns None rather than counting a miss — the caller decides whether
        an unresolved base is fatal (contraction decomposition) or just the
        normal OOV path.
        """
        ipa = self.dict.get(word)
        if ipa is not None:
            return ipa
        if self.use_neural_oov:
            ipa = self._neural_word(word)
            if ipa:
                return ipa
        return None

    def _homograph_ipa(self, word, words, index):
        """Context-selected IPA for a homograph, or None to fall through to the dict.

        `words` is the sentence's word tokens with None at every punctuation position,
        so slicing it gives neighbours that already stop at a punctuation barrier.

        The decision is always COMPUTED and recorded; it is only APPLIED when the
        instance was constructed with homographs=True. That split is what lets
        measure_homographs.py report exactly what would change without changing it.
        """
        decision = resolve_homograph(
            word,
            prev=words[index - 1] if index >= 1 else None,
            prev2=words[index - 2] if index >= 2 else None,
            nxt=words[index + 1] if index + 1 < len(words) else None,
        )
        if decision is None:
            return None
        key = (word, decision.sense)
        self.homograph_decisions[key] = self.homograph_decisions.get(key, 0) + 1
        if not self.homographs:
            return None
        # Counted as a dict hit as well: the pronunciation is the dictionary's own, via
        # an inflected form, so the hit-rate totals stay reconcilable with the token
        # count rather than developing a fifth bucket nobody sums.
        self.stats["dict_hits"] += 1
        self.stats["homograph_flips"] += 1
        return decision.ipa

    def phonemize_word(self, word):
        """Lower-case word -> espeak-style IPA, or None when unresolvable.

        The word may carry apostrophes; see the _CONTRACTIONS comment for why
        they cannot be stripped before lookup.
        """
        ipa = self.dict.get(word)
        if ipa is not None:
            self.stats["dict_hits"] += 1
            return ipa

        if "'" in word:
            ipa = self._apostrophe_word(word)
            if ipa:
                self.stats["contraction_hits"] += 1
                return ipa
            # Not a contraction we know (usually a name: O'Brien, d'Artagnan).
            # The bare-letters guess is still the best available, but it is
            # counted apart so it cannot hide inside the neural hit rate.
            bare = word.replace("'", "")
            if bare:
                ipa = self._plain_word(bare)
                if ipa:
                    self.stats["apostrophe_fallbacks"] += 1
                    self.apostrophe_fallback_words.add(word)
                    return ipa

        if self.use_neural_oov and "'" not in word:
            ipa = self._neural_word(word)
            if ipa:
                self.stats["neural_hits"] += 1
                return ipa

        self.stats["oov_misses"] += 1
        self.oov_words.add(word)
        return None

    def phonemize(self, text):
        """Normalized English text -> espeak-style IPA string.

        Applies the same pre-normalization as english_cleaners2 (ascii,
        lowercase, abbreviations, bracket removal) minus espeak itself.
        Digits are NOT expanded — feed normalized text (e.g. the LJSpeech
        normalized column); leftover digits surface in validate().
        """
        text = convert_to_ascii(text)
        text = lowercase(text)
        text = expand_abbreviations(text)
        text = remove_brackets(text)
        # Hyphens/dashes between words act as separators, like espeak.
        text = re.sub(r"[-–]+", " ", text)
        text = collapse_whitespace(text)
        # Materialised rather than streamed: homograph resolution needs the two tokens
        # to the left and one to the right, and a punctuation mark between them is a
        # barrier (see homographs._evidence) rather than something to look through.
        tokens = [m.group(0) for m in _TOKEN_RE.finditer(text)]
        words = [t if _WORD_RE.fullmatch(t) else None for t in tokens]
        out, first = [], True
        for index, token in enumerate(tokens):
            if _WORD_RE.fullmatch(token):
                # A leading apostrophe is usually an opening quote, so it is
                # dropped — except on the archaic forms where it is part of
                # the word ('tis, 'em). A trailing one needs no special case:
                # plural possessive and closing quote both resolve to the
                # bare base.
                word = token if token in _CONTRACTIONS else token.lstrip("'")
                if not word or word == "'":
                    continue
                ipa = self._homograph_ipa(word, words, index)
                if ipa is None:
                    ipa = self.phonemize_word(word)
                if ipa is None:
                    # Letters passed through verbatim. NOTE: this does *not*
                    # surface in validate() — a-z and ' are all inside the
                    # 178-symbol vocab, since espeak IPA reuses ASCII letters
                    # (validate("zyzzyqx") == []). The only signal is
                    # stats["oov_misses"] / oov_words, which every caller
                    # must check; phonemize_filelist treats it as fatal.
                    ipa = word
                if not first:
                    out.append(" ")
                out.append(ipa)
                first = False
            else:
                out.append(token)
        return "".join(out)

    @staticmethod
    def validate(ipa):
        """Returns the sorted set of characters not in the locked vocab."""
        return sorted(set(ipa) - _VOCAB)
