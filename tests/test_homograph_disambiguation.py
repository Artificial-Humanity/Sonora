"""D-M4: one pronunciation per dictionary key, and the other sense trains on it.

The OpenPhonemizer dictionary is flat, so `live` ships as lˈaɪv (the adjective) and every
"they live on buds and blossoms" in the corpus trains against the wrong vowel. The espeak
lane this replaced disambiguated by POS; the finding was left open on the grounds that a
partial heuristic would silently change pronunciations across the whole corpus.

What is guarded here is the pair of properties that make it not silent:

  * the resolver ABSTAINS unless the context forces a reading, and it never returns the
    pronunciation the dictionary already gives — so an unrecognised context leaves the
    corpus exactly as it is; and
  * op_g2p applies the decision only when explicitly constructed with homographs=True,
    while still COUNTING it either way, which is what makes the measurement a dry run.

The four abstention guards each have a test naming the sentence that produced them in the
v3c audit. They are not hypothetical: before those guards, "from mouth to mouth",
"pronounced it perfect", "a plum tree growing on it close to the fence" and "dove-like"
all flipped, and 6 of 288 flips were wrong. After them the audit is 281 flips at 0 known
errors, which is the number the corpus decision rests on.
"""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from matcha.text.homographs import (  # noqa: E402
    ADJ,
    ATTRIBUTIVE,
    HOMOGRAPHS,
    NOT_RESOLVABLE,
    PARTICIPLE,
    PAST,
    VERB,
    is_homograph,
    resolve,
)


# --- the table itself -----------------------------------------------------------------

def test_every_alternate_differs_from_the_default():
    """An alternate equal to the default is a table error, not a homograph."""
    for word, entry in HOMOGRAPHS.items():
        assert entry.senses, f"{word} has no senses; it belongs in NOT_RESOLVABLE"
        for sense, ipa in entry.senses.items():
            assert ipa != entry.default, f"{word}/{sense} restates the default"


def test_every_entry_records_its_provenance():
    """No invented IPA: each alternate names the dictionary entry it came from."""
    for word, entry in HOMOGRAPHS.items():
        assert entry.source.strip(), f"{word} has no source"


def test_resolvable_and_unreachable_sets_are_disjoint():
    overlap = set(HOMOGRAPHS) & set(NOT_RESOLVABLE)
    assert not overlap, f"{overlap} is both resolvable and not"


def test_is_homograph_covers_both_sets():
    assert is_homograph("live")
    assert is_homograph("bow")          # known, but out of reach of a POS rule
    assert not is_homograph("blossom")


def test_defaults_match_the_shipped_dictionary():
    """A dictionary revision that moves a default changes what "flip" means.

    Skipped where the assets are not mounted; on ai-lab-0 they always are.
    """
    import gzip
    import os

    from matcha.text.op_g2p import _default_assets

    path = os.path.join(_default_assets(), "g2p_dict.txt.gz")
    if not os.path.isfile(path):
        pytest.skip("g2p_dict.txt.gz not mounted")
    shipped = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                w, ipa = line.rstrip("\n").split("\t", 1)
                shipped[w] = ipa
    for word, entry in HOMOGRAPHS.items():
        assert word in shipped, f"{word} is not in the dictionary at all"
        assert shipped[word] == entry.default, (
            f"{word}: dictionary now says {shipped[word]}, table says {entry.default}")


# --- abstention is the default --------------------------------------------------------

def test_no_context_abstains():
    assert resolve("live") is None
    assert resolve("use", prev=None, prev2=None, nxt=None) is None


def test_unknown_word_abstains():
    assert resolve("blossom", prev="they") is None


def test_evidence_for_the_sense_already_shipped_abstains():
    """`refuse`-shaped case: the dictionary already gives the verb, so there is no flip.

    `close` after a copula is adjectival, and klˈoʊs IS the adjective, so the resolver
    must return None rather than a Decision that happens to equal the default.
    """
    assert resolve("close", prev="was") is None
    assert resolve("perfect", prev="is") is None


def test_punctuation_barrier_abstains():
    """op_g2p passes None across a comma; the resolver must not reach through it."""
    assert resolve("present", prev=None, prev2="to") is None


# --- the positive rules ---------------------------------------------------------------

@pytest.mark.parametrize("prev", ["to", "will", "must", "could", "don't", "can"])
def test_infinitival_and_modal_select_the_verb(prev):
    d = resolve("live", prev=prev)
    assert d is not None and d.ipa == "lˈɪv" and d.sense == VERB


@pytest.mark.parametrize("prev", ["they", "he", "we", "i", "who"])
def test_subject_pronoun_selects_a_finite_verb(prev):
    d = resolve("lives", prev=prev)
    assert d is not None and d.ipa == "lˈɪvz"


@pytest.mark.parametrize("prev", ["had", "have", "having", "was", "were", "is", "be"])
def test_perfect_and_passive_select_the_participle(prev):
    d = resolve("read", prev=prev)
    assert d is not None and d.ipa == "ɹˈɛd" and d.sense == PARTICIPLE


def test_copula_and_degree_select_the_adjective():
    assert resolve("content", prev="am").sense == ADJ
    assert resolve("content", prev="quite").sense == ADJ
    assert resolve("content", prev="am").ipa == "kəntˈɛnt"


def test_negator_takes_its_reading_from_the_auxiliary_beyond_it():
    """"will not close" is a verb; "is not content" is an adjective."""
    assert resolve("close", prev="not", prev2="did").sense == VERB
    assert resolve("content", prev="not", prev2="is").sense == ADJ
    # A bare negator with nothing behind it is not evidence.
    assert resolve("close", prev="not", prev2=None) is None


def test_determiner_licenses_the_attributive_only_before_a_noun():
    d = resolve("learned", prev="the", nxt="professor")
    assert d is not None and d.ipa == "lˈɜːnᵻd" and d.sense == ATTRIBUTIVE
    # "the learned of the age" — a preposition cannot start the noun phrase.
    assert resolve("learned", prev="the", nxt="of") is None


def test_causative_object_pronoun_selects_the_verb():
    """"let me present to you my friend" — the corpus sentence this rule exists for."""
    d = resolve("present", prev="me", prev2="let")
    assert d is not None and d.ipa == "pɹɪzˈɛnt"


def test_determiner_never_licenses_a_verb():
    for word in ("live", "use", "close", "present", "object"):
        assert resolve(word, prev="the", nxt="thing") is None or \
            resolve(word, prev="the", nxt="thing").sense == ATTRIBUTIVE


# --- the four abstention guards, each named by the sentence that produced it ----------

def test_reduplication_after_to_is_not_an_infinitive():
    """"news spread from mouth to mouth" — the `to` is a preposition."""
    assert resolve("mouth", prev="to", prev2="mouth") is None
    # The same word with an ordinary infinitival `to` still fires.
    assert resolve("mouth", prev="to", prev2="began") is not None


def test_small_clause_predicative_is_not_a_finite_verb():
    """"pronounced it perfect", "held it close", "have everything perfect"."""
    assert resolve("perfect", prev="it", prev2="pronounced") is None
    assert resolve("close", prev="it", prev2="held") is None
    assert resolve("perfect", prev="everything", prev2="have") is None
    # Outside the small-clause frame the pronoun is a subject again.
    assert resolve("close", prev="we", prev2="that") is not None


def test_prepositional_object_is_not_a_subject():
    """"a big plum tree growing on it close to the line fence"."""
    assert resolve("close", prev="it", prev2="on") is None


def test_hyphen_split_compound_abstains():
    """"dove-like" reaches the tokenizer as "dove like" after hyphen splitting."""
    assert resolve("dove", prev="be", nxt="like") is None
    assert resolve("dove", prev="they", nxt="into") is not None


# --- the two words that only fire on the perfect --------------------------------------

def test_past_simple_read_is_left_alone():
    """"he read the book yesterday" is not distinguishable from the present here."""
    assert resolve("read", prev="he") is None
    assert resolve("read", prev="to") is None
    assert resolve("read", prev="had") is not None


def test_wound_never_fires_on_a_finite_clause():
    """"to wound" is wˈuːnd, and "I wound and I heal" is the present of the same verb.

    Both flipped wrongly before `wound` was cut back to the participle.
    """
    assert resolve("wound", prev="to") is None
    assert resolve("wound", prev="i") is None
    assert resolve("wound", prev="she") is None
    assert resolve("wound", prev="had").ipa == "wˈaʊnd"


def test_dove_has_a_past_but_no_infinitive():
    assert resolve("dove", prev="they").sense == PAST
    assert resolve("dove", prev="to") is None


# --- op_g2p integration ---------------------------------------------------------------

def _stub_g2p(homographs):
    """An OpenPhonemizerG2P with a hand-built dictionary and no assets on disk.

    Constructed through __new__ so the test does not depend on the 275k dictionary or
    the TFLite graph being mounted; only phonemize()'s own logic is under test.
    """
    from matcha.text.op_g2p import OpenPhonemizerG2P

    g2p = OpenPhonemizerG2P.__new__(OpenPhonemizerG2P)
    g2p.dict = {"they": "ðˈeɪ", "live": "lˈaɪv", "here": "hˈɪɹ", "the": "ðə",
                "life": "lˈaɪf", "is": "ˈɪz"}
    g2p.homographs = homographs
    g2p.homograph_decisions = {}
    g2p.use_neural_oov = False
    g2p._neural = None
    g2p.oov_words = set()
    g2p.apostrophe_fallback_words = set()
    g2p.stats = {"dict_hits": 0, "neural_hits": 0, "oov_misses": 0,
                 "contraction_hits": 0, "apostrophe_fallbacks": 0, "homograph_flips": 0}
    return g2p


def test_off_by_default_leaves_the_phonemes_alone():
    off = _stub_g2p(homographs=False)
    assert off.phonemize("they live here") == "ðˈeɪ lˈaɪv hˈɪɹ"
    assert off.stats["homograph_flips"] == 0


def test_the_decision_is_counted_even_when_it_is_not_applied():
    """This is what makes measure_homographs.py a dry run rather than a rehearsal."""
    off = _stub_g2p(homographs=False)
    off.phonemize("they live here")
    assert off.homograph_decisions == {("live", VERB): 1}


def test_on_applies_the_alternate():
    on = _stub_g2p(homographs=True)
    assert on.phonemize("they live here") == "ðˈeɪ lˈɪv hˈɪɹ"
    assert on.stats["homograph_flips"] == 1
    # Still reconcilable against the token count: every word is one dict hit.
    assert on.stats["dict_hits"] == 3


def test_a_context_with_no_evidence_is_identical_both_ways():
    text = "the life is here"
    assert _stub_g2p(False).phonemize(text) == _stub_g2p(True).phonemize(text)
