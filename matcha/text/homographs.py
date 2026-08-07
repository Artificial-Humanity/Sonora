"""Context-aware homograph resolution for the espeak-free G2P lane (review finding D-M4).

The OpenPhonemizer dictionary is a flat `word -> ipa` map, so a heterophonic homograph
gets ONE pronunciation and the other sense trains against the wrong vowel. The espeak
lane this replaced disambiguated by part of speech; losing that was a regression, filed
as D-M4 and left open on the grounds that "a partial heuristic would silently change
pronunciations across the whole corpus".

Two things keep that from happening here.

**Nothing is invented.** Every alternate below is the dictionary's OWN pronunciation of
an inflected form, minus its suffix — `conducted kəndˈʌktᵻd` -> `kəndˈʌkt` is exactly the
verb the noun entry (`kˈɑːndʌkt`) is missing — or, where no inflection carries it, a
rime-mate that IS in the dictionary. Each entry records which. The derivation was done
mechanically and audited; three of its proposals were rejected and are listed in
NOT_RESOLVABLE with the reason.

**Nothing fires without positive evidence.** `resolve()` abstains unless the preceding
token forces a reading, and it only ever REPLACES the dictionary's pronunciation when
the evidence selects a sense the dictionary does not already give. Absence of evidence
leaves the corpus exactly as it is today. That is why the resolver is safe to measure
before it is safe to enable, and why `op_g2p` keeps it off by default.

The residue is real and is not papered over: past-simple `read` ("he read it yesterday")
is indistinguishable from the present without tense agreement across the clause, so only
the perfect and passive fire. Words whose two senses share a part of speech — `bow`,
`row`, `bass`, `sow`, `lead` — are out of reach of any POS rule at all. See
NOT_RESOLVABLE.
"""

import collections

# --- Sense keys ---------------------------------------------------------------------
#
# A sense key names the reading that the EVIDENCE selects, not the word's part of
# speech in the abstract. `participle` is kept apart from `verb` because the perfect
# and passive auxiliaries are the only decisive evidence for `read` and `wound`, while
# a modal or an infinitival `to` says nothing about them.
VERB = "verb"          # bare infinitive: after `to`, a modal, or do-support
PAST = "past"          # finite past: after a subject pronoun
PARTICIPLE = "participle"  # after a perfect or passive auxiliary
ADJ = "adj"
ATTRIBUTIVE = "attributive"

Homograph = collections.namedtuple("Homograph", "default senses source")


def _h(default, senses, source):
    return Homograph(default=default, senses=senses, source=source)


# --- The table ----------------------------------------------------------------------
#
# `default` is what g2p_dict.txt.gz ships, recorded so a dictionary revision that moves
# it is caught by the test suite rather than silently changing what "flip" means.
# `senses` maps a sense key to the pronunciation for that reading. `source` is the
# dictionary entry the alternate was derived from — the provenance, in the same style
# as op_g2p's _CONTRACTIONS table.
HOMOGRAPHS = {
    # -- noun/verb stress pairs. The verb falls out of the -ed/-ing form. -------------
    "absent": _h("ˈæbsənt", {VERB: "æbsˈɛnt"}, "absented æbsˈɛntᵻd"),
    "abstract": _h("ˈæbstɹækt", {VERB: "ɐbstɹˈækt"}, "abstracted ɐbstɹˈæktᵻd"),
    "alternate": _h("ɔːltˈɜːnət", {VERB: "ˈɔːltɚnˌeɪt"}, "alternated ˈɔːltɚnˌeɪɾᵻd"),
    "attribute": _h("ˈætɹɪbjˌuːt", {VERB: "ɐtɹˈɪbjuːt"}, "attributed ɐtɹˈɪbjuːɾᵻd"),
    "compound": _h("kˈɑːmpaʊnd", {VERB: "kɑːmpˈaʊnd"}, "compounded kɑːmpˈaʊndᵻd"),
    "concert": _h("kˈɑːnsɚt", {VERB: "kənsˈɜːt"}, "concerted kənsˈɜːɾᵻd"),
    "conduct": _h("kˈɑːndʌkt", {VERB: "kəndˈʌkt"}, "conducted kəndˈʌktᵻd"),
    "conflict": _h("kˈɑːnflɪkt", {VERB: "kənflˈɪkt"}, "conflicted kənflˈɪktᵻd"),
    "conscript": _h("kˈɑːnskɹɪpt", {VERB: "kɑːnskɹˈɪpt"}, "conscripted kɑːnskɹˈɪptᵻd"),
    "consort": _h("kˈɑːnsɔːɹt", {VERB: "kənsˈɔːɹt"}, "consorted kənsˈɔːɹɾᵻd"),
    "construct": _h("kˈɑːnstɹʌkt", {VERB: "kənstɹˈʌkt"}, "constructed kənstɹˈʌktᵻd"),
    "contest": _h("kˈɑːntɛst", {VERB: "kəntˈɛst"}, "contested kəntˈɛstᵻd"),
    "contract": _h("kˈɑːntɹækt", {VERB: "kəntɹˈækt"}, "contracted kəntɹˈæktᵻd"),
    "contrast": _h("kˈɑːntɹæst", {VERB: "kəntɹˈæst"}, "contrasted kəntɹˈæstᵻd"),
    "converse": _h("kˈɑːnvɜːs", {VERB: "kənvˈɜːs"}, "conversed kənvˈɜːst"),
    "convict": _h("kˈɑːnvɪkt", {VERB: "kənvˈɪkt"}, "convicted kənvˈɪktᵻd"),
    "defect": _h("dˈiːfɛkt", {VERB: "dᵻfˈɛkt"}, "defected dᵻfˈɛktᵻd"),
    "desert": _h("dˈɛzɚt", {VERB: "dᵻzˈɜːt"}, "deserted dᵻzˈɜːɾᵻd"),
    "discount": _h("dˈɪskaʊnt", {VERB: "dɪskˈaʊnt"}, "discounted dɪskˈaʊntᵻd"),
    "escort": _h("ˈɛskɔːɹt", {VERB: "ɛskˈɔːɹt"}, "escorted ɛskˈɔːɹɾᵻd"),
    "exploit": _h("ˈɛksplɔɪt", {VERB: "ɛksplˈɔɪt"}, "exploited ɛksplˈɔɪɾᵻd"),
    "export": _h("ˈɛkspoːɹt", {VERB: "ɛkspˈoːɹt"}, "exported ɛkspˈoːɹɾᵻd"),
    "extract": _h("ˈɛkstɹækt", {VERB: "ɛkstɹˈækt"}, "extracted ɛkstɹˈæktᵻd"),
    "finance": _h("fˈaɪnæns", {VERB: "faɪnˈæns"}, "financed faɪnˈænst"),
    "implant": _h("ˈɪmplænt", {VERB: "ɪmplˈænt"}, "implanted ɪmplˈæntᵻd"),
    "incense": _h("ˈɪnsɛns", {VERB: "ɪnsˈɛns"}, "incensed ɪnsˈɛnst"),
    "increase": _h("ˈɪŋkɹiːs", {VERB: "ɪŋkɹˈiːs"}, "increased ɪŋkɹˈiːst"),
    "insert": _h("ˈɪnsɜːt", {VERB: "ɪnsˈɜːt"}, "inserted ɪnsˈɜːɾᵻd"),
    "insult": _h("ˈɪnsʌlt", {VERB: "ɪnsˈʌlt"}, "insulted ɪnsˈʌltᵻd"),
    "intern": _h("ˈɪntɜːn", {VERB: "ɪntˈɜːn"}, "interned ɪntˈɜːnd"),
    "mandate": _h("mˈændeɪt", {VERB: "mændˈeɪt"}, "mandated mændˈeɪɾᵻd"),
    "object": _h("ˈɑːbdʒɛkt", {VERB: "ɑːbdʒˈɛkt"}, "objected ɑːbdʒˈɛktᵻd"),
    "perfect": _h("pˈɜːfɛkt", {VERB: "pɚfˈɛkt"}, "perfected pɚfˈɛktᵻd"),
    "present": _h("pɹˈɛzənt", {VERB: "pɹɪzˈɛnt"}, "presented pɹɪzˈɛntᵻd"),
    "progress": _h("pɹˈɑːɡɹɛs", {VERB: "pɹəɡɹˈɛs"}, "progressed pɹəɡɹˈɛst"),
    "project": _h("pɹˈɑːdʒɛkt", {VERB: "pɹədʒˈɛkt"}, "projected pɹədʒˈɛktᵻd"),
    "protest": _h("pɹˈoʊtɛst", {VERB: "pɹətˈɛst"}, "protested pɹətˈɛstᵻd"),
    "recall": _h("ɹˈiːkɔːl", {VERB: "ɹᵻkˈɔːl"}, "recalled ɹᵻkˈɔːld"),
    "record": _h("ɹˈɛkɚd", {VERB: "ɹᵻkˈoːɹd"}, "recorded ɹᵻkˈoːɹdᵻd"),
    "segment": _h("sˈɛɡmənt", {VERB: "sɛɡmˈɛnt"}, "segmented sɛɡmˈɛntɪd"),
    "separate": _h("sˈɛpɹət", {VERB: "sˈɛpɚɹˌeɪt"}, "separated sˈɛpɚɹˌeɪɾᵻd"),
    "survey": _h("sˈɜːveɪ", {VERB: "sɚvˈeɪ"}, "surveying sɚvˈeɪɪŋ"),
    "suspect": _h("sˈʌspɛkt", {VERB: "səspˈɛkt"}, "suspected səspˈɛktᵻd"),
    "torment": _h("tˈoːɹmɛnt", {VERB: "toːɹmˈɛnt"}, "tormented toːɹmˈɛntᵻd"),
    "transport": _h("tɹˈænspoːɹt", {VERB: "tɹænspˈoːɹt"}, "transported tɹænspˈoːɹɾᵻd"),
    "upgrade": _h("ˈʌpɡɹeɪd", {VERB: "ʌpɡɹˈeɪd"}, "upgraded ʌpɡɹˈeɪdᵻd"),

    # -- voicing alternations: the noun ends /s/, the verb /z/. ----------------------
    "abuse": _h("ɐbjˈuːs", {VERB: "ɐbjˈuːz"}, "abused ɐbjˈuːzd"),
    "close": _h("klˈoʊs", {VERB: "klˈoʊz"}, "closed klˈoʊzd · closing klˈoʊzɪŋ"),
    "house": _h("hˈaʊs", {VERB: "hˈaʊz"}, "housed hˈaʊzd · housing hˈaʊzɪŋ"),
    "misuse": _h("mɪsjˈuːs", {VERB: "mɪsjˈuːz"}, "misused mɪsjˈuːzd"),
    "mouth": _h("mˈaʊθ", {VERB: "mˈaʊð"}, "mouthed mˈaʊðd"),
    "use": _h("jˈuːs", {VERB: "jˈuːz"}, "used jˈuːzd · using jˈuːzɪŋ"),

    # -- vowel alternations. ---------------------------------------------------------
    # `live`/`lives` are the highest-value entries in this table: the dictionary ships
    # the ADJECTIVE (lˈaɪv, as in "live broadcast"), and in narrative prose essentially
    # every occurrence is the verb.
    "live": _h("lˈaɪv", {VERB: "lˈɪv"}, "lived lˈɪvd · living lˈɪvɪŋ · give ɡˈɪv"),
    "lives": _h("lˈaɪvz", {VERB: "lˈɪvz"}, "lived lˈɪvd · gives ɡˈɪvz"),
    "content": _h("kˈɑːntɛnt", {ADJ: "kəntˈɛnt", VERB: "kəntˈɛnt"},
                  "contented kəntˈɛntᵻd · discontent dɪskəntˈɛnt"),
    # No VERB key: "dove" is the bird or the past of dive, never a bare infinitive.
    "dove": _h("dˈʌv", {PAST: "dˈoʊv", PARTICIPLE: "dˈoʊv"},
               "drove dɹˈoʊv · wove wˈoʊv · grove ɡɹˈoʊv"),
    "entrance": _h("ˈɛntɹəns", {VERB: "ɛntɹˈæns"}, "entranced ɛntɹˈænst"),
    "learned": _h("lˈɜːnd", {ATTRIBUTIVE: "lˈɜːnᵻd"}, "learnedly lˈɜːnᵻdli"),
    # Only the perfect and the passive fire — see the module docstring.
    "read": _h("ɹˈiːd", {PARTICIPLE: "ɹˈɛd"}, "red ɹˈɛd · said sˈɛd · bed bˈɛd"),
    "tear": _h("tˈɪɹ", {VERB: "tˈɛɹ"}, "tearing tˈɛɹɪŋ"),
    # Perfect and passive only, for the same reason as `read` and with an extra one:
    # "to wound" is the transitive verb meaning injure, which IS wˈuːnd, and after a
    # subject pronoun "I wound and I heal" (present, injure) is indistinguishable from
    # "she wound herself around the branch" (past of wind). Both fired wrongly in the
    # v3c audit. Four correct flips are given up to avoid those two.
    "wound": _h("wˈuːnd", {PARTICIPLE: "wˈaʊnd"},
                "unwound ʌnwˈaʊnd · found fˈaʊnd · bound bˈaʊnd"),
}

# --- What this cannot reach, and why -------------------------------------------------
#
# Recorded as data so the gap is auditable rather than an absence someone has to notice.
# `measure_homographs.py` reports occurrences of these alongside the resolvable ones.
NOT_RESOLVABLE = {
    "bow": "both senses are nouns (bˈoʊ the weapon, bˈaʊ the bend and the ship's front); "
           "no POS rule separates them",
    "row": "both senses are nouns (ɹˈoʊ the line, ɹˈaʊ the quarrel)",
    "bass": "both senses are nouns (bˈeɪs the voice, bˈæs the fish)",
    "sow": "both senses occur as nouns and verbs independently of the vowel",
    "lead": "the metal (lˈɛd) and the position (lˈiːd) are BOTH nouns, so a determiner "
            "cannot choose; 'took the lead' would flip wrongly",
    "minute": "the attributive rule cannot separate 'a minute quantity' (maɪnjˈuːt) from "
              "'a minute more' (mˈɪnɪt), and the corpus is almost entirely the latter",
    "invalid": "the dictionary's only inflection is the rare verb 'invalided'; the noun "
               "ˈɪnvəlɪd is not derivable from any entry",
    "excuse": "the dictionary gives /s/ for every form including 'excused ɛkskjˈuːst', so "
              "the verb's /z/ would have to be invented rather than sourced",
    "number": "'numbering nˈʌmbɚɹɪŋ' yields a liaison artifact, not a second sense",
    "produce": "the dictionary ships the verb (pɹədˈuːs); the noun ˈpɹoʊduːs is not "
               "derivable from any inflection it holds",
    "polish": "Polish/polish is carried by CASE, and cleaners.lowercase() runs before the "
              "G2P sees the token — the evidence is gone upstream",
    "august": "August/august is carried by CASE, same as polish",
}

# --- Evidence ------------------------------------------------------------------------
#
# Closed classes only. A homograph immediately after one of these has its reading forced;
# anything else abstains. Punctuation is a barrier — the caller passes None across it —
# because "…, present…" says nothing about what precedes the comma.

# Infinitival `to`, modals and do-support: a bare verb has to follow. The negative
# contractions are here rather than under _NEGATORS because op_g2p tokenises them whole
# — "don't close" arrives as one token, with the auxiliary already inside it.
_MODALS = frozenset("""to will would shall should can could may might must cannot
    do does did let don't doesn't didn't won't can't couldn't wouldn't shouldn't
    mustn't shan't""".split())

# Perfect and passive auxiliaries.
_HAVE = frozenset("have has had having".split())
_BE = frozenset("am is are was were be been being".split())

# A homograph immediately after a subject pronoun is the clause's verb.
_SUBJECTS = frozenset("""i you he she it we they who nothing something anything
    everything nobody somebody anybody everybody everyone someone anyone""".split())

# Causative `let me present`, `make him repeat`. `have`/`had` are deliberately NOT here:
# they head the small-clause pattern below far more often than the causative one, and
# the two want opposite answers.
_OBJECTS = frozenset("me him her us them you it".split())
_CAUSATIVE = frozenset("let make help".split())

# Small clauses: `pronounced it perfect`, `held it close`, `have everything perfect`.
# The homograph is a predicative complement — an adjective — and the pronoun before it
# is the matrix verb's OBJECT, not the clause's subject. Without this the subject-pronoun
# rule reads all three as verbs, which was 3 of the 6 false positives in the first
# corpus audit. Closed list of matrix verbs; anything outside it abstains anyway.
_SMALL_CLAUSE_HEADS = frozenset("""have has had make makes made keep keeps kept hold
    holds held find finds found think thinks thought call calls called consider considers
    considered pronounce pronounces pronounced deem deems deemed leave leaves left get
    gets got set sets render renders rendered want wants wanted like likes liked see sees
    saw feel feels felt declare declares declared""".split())
_SMALL_CLAUSE_SUBJECTS = _OBJECTS | frozenset(
    "everything something anything nothing everyone someone anyone one".split())

# Determiners, quantifiers and numerals — the left edge of a noun phrase.
_DETERMINERS = frozenset("""the a an this that these those my your his her its our their
    whose some any no every each another one two three four five six seven eight nine ten
    such both either neither many much few several all what which first second third
    last next same other""".split())

# Degree adverbs: predicative-adjective evidence, like the copulas.
_DEGREE = frozenset("""very so quite too rather fairly perfectly entirely wholly
    altogether extremely""".split())

_NEGATORS = frozenset("not never".split())

_PREPOSITIONS = frozenset("""of in on at for with by from into onto upon about through
    without within under over between among against during after before since until till
    toward towards across along around behind beneath beside besides beyond near off out
    past per than""".split())

# Tokens that cannot begin a noun phrase, used by the attributive rule to tell
# "the learned professions" from "the learned of the age".
_NON_NOMINAL_NEXT = _PREPOSITIONS | frozenset(
    "to and or but nor if that which who whom whose when where while".split())


def _evidence(word, prev, prev2, nxt):
    """Sense keys the left context licenses, most specific first.

    Returns a tuple because one context can license more than one reading and only the
    word's own table knows which of them it distinguishes: a copula is evidence for a
    predicative adjective AND for a passive participle, and `content` defines the first
    while `read` defines the second.
    """
    if prev is None:
        return ()

    # -- the three abstentions, before any evidence is read -------------------------
    #
    # Each removes a false positive found by auditing every flip on v3c; each is a
    # context that LOOKS decisive and is not.

    # "from mouth to mouth", "face to face": the `to` is a preposition inside a
    # reduplication, not an infinitive marker.
    if prev == "to" and prev2 == word:
        return ()

    # "pronounced it perfect" — predicative complement, not a finite verb.
    if prev in _SMALL_CLAUSE_SUBJECTS and prev2 in _SMALL_CLAUSE_HEADS:
        return ()

    # "a plum tree growing on it close to the fence": `it` is the object of a
    # preposition, so it cannot be the subject of a following verb.
    if prev in _OBJECTS and prev2 in _PREPOSITIONS:
        return ()

    # "dove-like" reaches us as "dove like", because op_g2p splits hyphens into spaces
    # before tokenising. The left context is then whatever preceded the compound, which
    # says nothing about the compound's first element.
    if nxt == "like":
        return ()

    if prev in _NEGATORS:
        # "will not close" is a verb; "is not perfect" is an adjective. The negator
        # itself is silent — what decides is the auxiliary on its far side.
        if prev2 in _MODALS or prev2 in _HAVE:
            return (VERB,)
        if prev2 in _BE:
            return (ADJ, PARTICIPLE)
        return ()

    if prev in _HAVE:
        return (PARTICIPLE, VERB)
    if prev in _BE:
        return (ADJ, PARTICIPLE)
    if prev in _MODALS:
        return (VERB,)
    if prev in _SUBJECTS:
        # A finite clause, so the past is on the table as well as the bare form. The
        # order matters for `dove`, which has a past and no infinitive.
        return (PAST, VERB)
    if prev in _OBJECTS and prev2 in _CAUSATIVE:
        return (VERB,)
    if prev in _DEGREE:
        return (ADJ,)
    if prev in _DETERMINERS:
        # A determiner excludes the verb outright. It licenses the attributive reading
        # only when a noun can actually follow — "the learned professions", not
        # "the learned of the age" and not "a minute or two".
        if nxt is not None and nxt not in _NON_NOMINAL_NEXT:
            return (ATTRIBUTIVE,)
        return ()
    return ()


Decision = collections.namedtuple("Decision", "ipa sense rule")


def resolve(word, prev=None, prev2=None, nxt=None):
    """Context-selected pronunciation for `word`, or None to leave it to the dictionary.

    All four arguments are lower-cased word tokens; the caller passes None wherever a
    punctuation mark or a sentence edge intervenes.

    Returns a Decision only when the evidence positively selects a sense the dictionary
    does NOT already give. Every other case — not a homograph, no evidence, evidence for
    the sense already shipped — returns None, which is what keeps this from rewriting
    the corpus on a guess.
    """
    entry = HOMOGRAPHS.get(word)
    if entry is None or not entry.senses:
        return None
    for sense in _evidence(word, prev, prev2, nxt):
        ipa = entry.senses.get(sense)
        if ipa is not None:
            if ipa == entry.default:
                return None
            return Decision(ipa=ipa, sense=sense, rule=f"{prev} -> {sense}")
    return None


def is_homograph(word):
    """True for any word this module knows about, resolvable or not.

    Used by the measurement script so the report counts the words it cannot fix as well
    as the ones it can.
    """
    return word in HOMOGRAPHS or word in NOT_RESOLVABLE
