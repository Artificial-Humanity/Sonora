"""D-C1 on the device: the front end that ships is not the front end that trained.

The host got a contraction table on 2026-08-02, when D-C1 was found to have poisoned the
IPA of corpus versions v1 through v3. `kotlin_replica.phonemize` — the executable spec for
the mobile port, since no mobile app exists yet — was still
`DICT.get(token) or phon_word(token)` on 2026-08-07: a flat dictionary lookup with a
neural fallback and no apostrophe handling at all.

That is not a near-miss. Measured against the shipped assets: **0 of 274,927 dictionary
keys contain an apostrophe**, and `'` is absent from the neural charset, where unknown
characters are silently skipped. So a contraction could only ever resolve to its
apostrophe-stripped letters, and it came back looking like a successful lookup —
`we'll` -> `wˈɛl`, which is the word *well*, and `she'd` -> `ʃˈɛd`, the word *shed*.

What is guarded here:

  * the device front end reproduces the training front end's phoneme string EXACTLY, over
    a probe corpus that includes every entry of the contraction table;
  * the check is not vacuous — the flat lookup this replaced still fails it, loudly;
  * a missing `g2p_contractions.json` is FATAL rather than a fallback to the flat lookup,
    because a silent fallback is precisely how the defect survived two corpus versions; and
  * the converter emits the asset and gates on it (G7), so the tables cannot drift.
"""

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "litert_export"))

device_g2p = pytest.importorskip("device_g2p")
op_g2p = pytest.importorskip("matcha.text.op_g2p")

CONVERTER = REPO / "scripts" / "litert_export" / "convert_vat.py"
REPLICA = REPO / "scripts" / "litert_export" / "kotlin_replica.py"

_ASSETS = pathlib.Path(op_g2p._default_assets())
needs_assets = pytest.mark.skipif(
    not (_ASSETS / "g2p_dict.txt.gz").is_file(),
    reason=f"G2P dictionary not present under {_ASSETS}",
)


@pytest.fixture(scope="module")
def tables():
    """The exported tables as the DEVICE sees them — through JSON, not as a Python dict."""
    return json.loads(json.dumps(op_g2p.contraction_tables(), ensure_ascii=False))


@pytest.fixture(scope="module")
def host():
    # Dictionary-only on both sides: the neural graph is identical code on both paths and
    # loading it costs seconds per test session. The OOV probe below covers it.
    return op_g2p.OpenPhonemizerG2P(use_neural_oov=False)


@pytest.fixture(scope="module")
def device(tables):
    return device_g2p.DeviceG2P(tables=tables, use_neural_oov=False)


@needs_assets
def test_device_and_host_agree_on_every_probe(host, device, tables):
    """The gate itself, run without a converter: identical phoneme strings, no exceptions.

    `probe_sentences` is table-driven, so every contraction entry is exercised and a new
    entry is covered by the next run rather than by someone remembering this file.
    """
    sentences = device_g2p.probe_sentences(tables)
    bad = device_g2p.compare(host.phonemize, device.phonemize, sentences)
    assert not bad, "\n".join(
        f"{s!r}\n  host: {w!r}\n  dev : {g!r}" for s, w, g in bad[:10])
    assert len(sentences) > len(tables["contractions"])


@needs_assets
def test_the_host_table_is_anchored_to_absolute_phonemes(host):
    """D-C1's own guard, which it never had — and which parity CANNOT provide.

    Both front ends read one table, so an entry deleted from `_CONTRACTIONS` disappears
    from the host and the device together and G7 still passes: parity is preserved while
    both sides are wrong. Found while checking that the gate was not vacuous. The only
    thing that catches it is an absolute assertion about the phonemes, so this test names
    the values rather than comparing two implementations.

    The 2026-08-02 fix shipped with no test of its own; this is it.
    """
    assert host.phonemize("don't") == "dˈoʊnt"
    assert host.phonemize("we'll") == "wiːl"
    assert host.phonemize("won't") == "wˈoʊnt"
    assert host.phonemize("we're") == "wɪɹ"
    assert host.phonemize("'tis") == "tˈɪz"
    # The bare-letters answers, so a regression is recognisable as the defect it is:
    # `wont` and `well` are real dictionary entries, and they are what the contractions
    # collapsed into. (`dont` is NOT in the dictionary — it reached dˈɔnt through the
    # neural graph, which is why the wrong answer arrived as a confident one.)
    assert host.phonemize("wont") == "wˈɔnt" != host.phonemize("won't")
    assert host.phonemize("well") == "wˈɛl" != host.phonemize("we'll")
    tables = op_g2p.contraction_tables()
    assert len(tables["contractions"]) >= 69, "the irregular table lost entries"
    assert set(tables["clitics"]) == {"n't", "'ll", "'ve", "'re", "'d", "'m"}


@needs_assets
def test_the_three_exemplars_are_not_their_stripped_letters(device):
    """D-C1's own examples, as phonemes rather than as a description of phonemes."""
    assert device.phonemize("don't") == "dˈoʊnt"
    assert device.phonemize("we'll") == "wiːl"
    assert device.phonemize("won't") == "wˈoʊnt"
    # The failure mode: each of these is what the flat lookup returned, and each is a
    # different English word the model has heard thousands of times.
    for contraction, other_word in (("we'll", "well"), ("she'd", "shed"),
                                    ("he'll", "hell"), ("i'll", "ill")):
        assert device.phonemize(contraction) != device.phonemize(other_word), \
            f"{contraction} still phonemizes as {other_word}"


@needs_assets
def test_the_productive_classes_reach_words_no_table_enumerates(host, device):
    """Possessives and clitics attach to arbitrary bases, which is why they are a RULE.

    A table-only fix would cover `don't` and leave `James's` as garbage — which is what
    the flat lookup did to it (dʒˈːmˈɛs).
    """
    for phrase in ("james's hat", "the fox's den", "the church's bell",
                   "dickens's world", "the cliff's edge", "the moth's wing",
                   "the boys' coats", "we've gone", "they'd know"):
        assert device.phonemize(phrase) == host.phonemize(phrase), phrase
    # The three allomorphs, by the final phoneme of the base rather than by spelling.
    assert device.phonemize("cat's").endswith("ts")      # voiceless -> s
    assert device.phonemize("dog's").endswith("z")       # voiced    -> z
    assert device.phonemize("horse's").endswith("ᵻz")    # sibilant  -> ᵻz


@needs_assets
def test_the_neural_oov_path_agrees_too():
    """Dictionary misses go through the same graph on both sides — separately, not shared.

    device_g2p reimplements the DeepPhonemizer runner rather than calling op_g2p's, so
    this compares two implementations. A shared call would prove a function equals itself.
    """
    litert = pytest.importorskip("ai_edge_litert")
    del litert
    tables = json.loads(json.dumps(op_g2p.contraction_tables(), ensure_ascii=False))
    host = op_g2p.OpenPhonemizerG2P(use_neural_oov=True)
    device = device_g2p.DeviceG2P(tables=tables, use_neural_oov=True,
                                  assets_dir=host.assets_dir)
    for sentence in ("Zyzzyqx blorptastic frunge.",
                     "O'Brien and d'Artagnan rode with Ma'evan."):
        assert device.phonemize(sentence) == host.phonemize(sentence), sentence


@needs_assets
def test_the_flat_lookup_this_replaced_still_fails(host, device, tables):
    """The parity check is not vacuous: the shipped-until-2026-08-07 front end fails it.

    A gate that passes on the broken implementation as well as the fixed one certifies
    nothing. This reconstructs the old `kotlin_replica.phonemize` in the shape it had.
    """
    word_re = re.compile(r"[a-z']+")
    token_re = re.compile(r"[a-z']+|[.,!?;:—…\"]")

    def flat(text):
        ipa, first = [], True
        for match in token_re.finditer(text.lower()):
            token = match.group(0)
            if word_re.fullmatch(token):
                phonemes = device.dict.get(token) or token.replace("'", "")
                if not first:
                    ipa.append(" ")
                ipa.append(phonemes)
                first = False
            else:
                ipa.append(token)
        return "".join(ipa)

    sentences = device_g2p.probe_sentences(tables)
    bad = device_g2p.compare(host.phonemize, flat, sentences)
    assert len(bad) > len(sentences) // 2, \
        "the flat lookup passes parity — the probe corpus is not exercising apostrophes"


@needs_assets
def test_the_asset_on_disk_is_what_the_device_reads(tmp_path, host):
    """The path the device actually takes: JSON file in, phonemes out.

    Everything above hands the tables in as an object, which is how gate G7 certifies
    bytes it has not written yet. A device has no such shortcut — it opens the file the
    converter left beside the graphs, and that is the case that has to work.
    """
    serialized = json.dumps(op_g2p.contraction_tables(), ensure_ascii=False,
                            indent=2, sort_keys=True)
    (tmp_path / device_g2p.CONTRACTIONS_ASSET).write_text(serialized, encoding="utf-8")
    device = device_g2p.DeviceG2P(artifacts_dir=str(tmp_path),
                                  assets_dir=str(_ASSETS), use_neural_oov=False)
    for sentence in ("Don't go — we'll be late, and it won't matter.",
                     "James's hat and the horses' stalls.",
                     "'Tis a fine morning; hand 'em over."):
        assert device.phonemize(sentence) == host.phonemize(sentence), sentence


def test_a_missing_table_is_fatal_not_a_fallback(tmp_path):
    """The defining property. A device that cannot find the asset must REFUSE.

    Falling back to the plain dictionary is not a degraded mode, it is D-C1 — and it
    reports success, which is why the defect outlived two corpus versions.
    """
    with pytest.raises(SystemExit) as excinfo:
        device_g2p.DeviceG2P(assets_dir=str(tmp_path), artifacts_dir=str(tmp_path))
    message = str(excinfo.value)
    assert device_g2p.CONTRACTIONS_ASSET in message
    assert "convert_vat.py" in message


def test_a_homograph_enabled_export_is_refused(tables):
    """D-M4 coupling: if the corpus front end resolves homographs, the device cannot.

    Resolution needs context evidence, not a lookup table, so there is nothing to ship.
    Failing at construction keeps the two front ends from drifting the moment the owner
    turns D-M4 on for a derivation.
    """
    poisoned = dict(tables, homographs={"live": "lˈɪv"})
    with pytest.raises(SystemExit) as excinfo:
        device_g2p.DeviceG2P(tables=poisoned, use_neural_oov=False)
    assert "homographs.py" in str(excinfo.value)


def test_the_normalization_chain_matches_the_host():
    """Order matters, and it runs before tokenization — so a difference here moves words.

    Abbreviation expansion after lowercasing, brackets removed, hyphens and dashes as
    separators. Torch-free and asset-free: this is string handling, not phonemes.
    """
    from matcha.text import cleaners

    for text in ("Mr. Smith and Dr. Jones went to St. Paul's.",
                 "A well-known (and much-loved) tune — the second-best kind.",
                 "The naïve café serves crème brûlée.",
                 "  ragged   whitespace\tand\nnewlines  "):
        want = cleaners.convert_to_ascii(text)
        want = cleaners.lowercase(want)
        want = cleaners.expand_abbreviations(want)
        want = cleaners.remove_brackets(want)
        want = re.sub(r"[-–]+", " ", want)
        want = cleaners.collapse_whitespace(want)
        assert device_g2p.normalize(text) == want, text


def test_the_converter_emits_the_asset_and_gates_on_it():
    """G7 and the asset are one change: a table that ships ungated drifts."""
    src = CONVERTER.read_text(encoding="utf-8")
    assert "g2p_parity_gate()" in src
    assert "G7 host/device G2P parity" in src
    assert 'os.path.join(ART, "g2p_contractions.json")' in src
    # Written from the string the gate certified, not re-serialized afterwards.
    assert "f.write(g2p_json)" in src
    # The manifest tells the host which front end these graphs expect.
    assert "g2p=dict(" in src and "homographs=False" in src


def test_the_replica_no_longer_carries_its_own_flat_lookup():
    """The spec the mobile port is written against is this file, so it has to be right."""
    src = REPLICA.read_text(encoding="utf-8")
    # Comment lines dropped: the replaced expression is quoted in the comment that
    # explains why it went, and a test that cannot tell code from prose would forbid
    # recording what was fixed.
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "DICT.get(token) or phon_word(token)" not in code
    assert "device_g2p.DeviceG2P(" in code
    # A missing table is named at preflight, beside its producer, rather than surfacing
    # as a wrong vowel six frames deeper.
    assert "device_g2p.CONTRACTIONS_ASSET" in src
