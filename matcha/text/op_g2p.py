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
then ../Reference/models/litert-community/Matcha-TTS relative to the repo
root (the workspace layout on both dev machines).
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
from matcha.text.symbols import symbols

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_ASSETS = os.path.join(_REPO_ROOT, "..", "Reference", "models", "litert-community", "Matcha-TTS")

# Word tokens may carry apostrophes; everything else passes through only if
# it is vocab punctuation (hyphens/brackets are separators, handled upstream).
_TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:—…\"«»“”¡¿]")
_WORD_RE = re.compile(r"[a-z']+")
_COMBINING_TILDE = "̃"

_VOCAB = set(symbols)


class OpenPhonemizerG2P:
    """Dictionary-primary, neural-fallback espeak-IPA phonemizer."""

    def __init__(self, assets_dir=None, use_neural_oov=True):
        self.assets_dir = os.path.abspath(
            assets_dir or os.environ.get("SONORA_G2P_ASSETS") or _DEFAULT_ASSETS
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
        self.stats = {"dict_hits": 0, "neural_hits": 0, "oov_misses": 0}
        self.oov_words = set()

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

    def phonemize_word(self, word):
        """Lower-case word -> espeak-style IPA, or None when unresolvable."""
        ipa = self.dict.get(word)
        if ipa is not None:
            self.stats["dict_hits"] += 1
            return ipa
        if self.use_neural_oov:
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
        out, first = [], True
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if _WORD_RE.fullmatch(token):
                word = token.strip("'")
                if not word:
                    continue
                ipa = self.phonemize_word(word)
                if ipa is None:
                    ipa = word  # surfaces as a vocab violation in validate()
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
