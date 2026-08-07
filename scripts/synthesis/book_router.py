# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""book_router — Stage A of the book-prose lane, made executable.

`book-prose-lane.md` has specified this router since 2026-07-18 and describes the lane
as live, but no script ever existed: `book_queue.txt` and `books_ledger.json` were being
maintained by hand, which is why ten entries sat PENDING with no verdict.

WHAT IT DOES (per book-prose-lane.md § Stage A):

  SKIP       already in books_ledger (any lane)
  REAL-AUDIO a librivox.org URL — an explicit owner pick, routed to the force-align lane
  SYNTHESIZE an SE/PG URL — straight through, no overlap ask (owner rule 2026-07-22:
             "a submission is itself the lane decision")

THE LIBRITTS OVERLAP CHECK, AND WHY IT ANNOTATES RATHER THAN SKIPS
------------------------------------------------------------------
LibriTTS-R derives from LibriSpeech, which derives from LibriVox, so overlap with a
LibriVox pick is the DEFAULT, not the exception. `BOOKS.txt` ships with train-clean-100
and is keyed by **Gutenberg etext-id** — the same canonical key as our ledger — so the
join is exact rather than the title/author guess the note assumed.

We match on BOTH etext-id and normalized title, and report which key hit. The title pass
is a safeguard against a work LibriTTS holds under a different Gutenberg edition than
LibriVox reports — note that BOOKS.txt does list some works twice (Peter Pan is there as
both PG 16 and, as "Peter and Wendy", PG 26654), so the id join is stronger than expected
but not guaranteed complete.

Overlap NEVER auto-skips a librivox.org URL. Per the lane's LibriVox stance the link is
an explicit owner pick ("never an incidental crawl candidate"), and the 2026-07-22 rule
change retired the overlap ask for the symmetric SE/PG case on the grounds that a
submission is itself the decision. What overlap means here is narrower and worth knowing:
LibriTTS already has *clean aligned audio* for this text, so the marginal value of
force-aligning it ourselves is lower — unless the pick is a DRAMATIC READING, where the
whole point is multi-voice character performance that LibriTTS's single-narrator
recordings do not contain. The verdict line says which.

Default is a dry run. Pass --apply to write the ledger and mark the queue.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import shutil
import sys
import time
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import synth_common  # noqa: E402
import urllib.request

DATASETS = pathlib.Path("/data/model-training/datasets")
QUEUE = DATASETS / "book_queue.txt"
LEDGER = DATASETS / "books_ledger.json"
BOOKS_TXT = DATASETS / "LibriTTS_R/train-clean-100/BOOKS.txt"

UA = {"User-Agent": "Mozilla/5.0 (research; Sonora dataset source router)"}
API = "https://librivox.org/api/feed/audiobooks/"

# LibriVox rate-limits; be a good citizen on a 30-book batch.
SLEEP_BETWEEN = 1.5

STOPWORDS = {"the", "a", "an"}


# ---------------------------------------------------------------- helpers
def norm_title(t: str) -> str:
    """Lowercase, drop punctuation and a leading article, collapse space.

    Deliberately lossy: it has to make "The Awakening" and "Awakening" agree. It must
    NOT be clever enough to equate genuinely different titles — same-work/different-title
    cases belong in ALIASES, where each entry is verified by hand.
    """
    t = (t or "").lower().replace("&", " and ")   # "Guest & Other" == "Guest and Other"
    # Apostrophes are DELETED, not spaced: a URL slug writes "draculas", the catalogue
    # writes "Dracula's", and spacing would give "dracula s" — two words vs one.
    t = re.sub(r"['\u2018\u2019]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    words = [w for w in t.split() if w]
    while words and words[0] in STOPWORDS:
        words.pop(0)
    return " ".join(words)


# Works LibriTTS holds under a different Gutenberg edition than LibriVox reports.
# Curated, not inferred — populate only with VERIFIED pairs that the id join misses.
# Empty so far: the Peter Pan / Peter and Wendy case that motivated it turned out to be
# covered, because BOOKS.txt lists both editions.
ALIASES: dict[str, str] = {}


def load_libritts_books() -> tuple[dict[str, str], dict[str, str]]:
    """-> ({etext_id: title}, {normalized_title: etext_id}) for train-clean-100."""
    by_id: dict[str, str] = {}
    by_title: dict[str, str] = {}
    if not BOOKS_TXT.is_file():
        print(f"!! {BOOKS_TXT} not found — overlap check DISABLED", file=sys.stderr)
        return by_id, by_title
    for line in BOOKS_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        eid, title = parts[0], parts[1] if len(parts) > 1 else ""
        if not eid.isdigit():
            continue
        by_id[eid] = title
        by_title.setdefault(norm_title(title), eid)
    return by_id, by_title


def api_get(params: dict) -> dict:
    """Query the LibriVox feed. A 404 is how this API says "no results" — not an error."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"books": []}
        raise


def author_from_slug(url: str) -> str:
    """'...-by-bram-stoker' -> 'stoker' (surname), '' when absent."""
    slug = urllib.parse.urlparse(url).path.strip("/").split("/")[-1]
    m = re.search(r"-by-(.+)$", slug)
    if not m:
        return ""
    parts = [w for w in m.group(1).split("-") if w and not w.isdigit()]
    return parts[-1] if parts else ""


def slug_to_query(url: str) -> str:
    """librivox.org/the-awakening-by-kate-chopin/ -> 'the awakening'.

    Strips the trailing '-by-<author>' and any '-version-N' / dramatic-reading marker,
    because the API's title search does not match those.
    """
    slug = urllib.parse.urlparse(url).path.strip("/").split("/")[-1]
    slug = re.sub(r"-by-.*$", "", slug)
    slug = re.sub(r"-(version|dramatic-reading)\b.*$", "", slug)
    # Some slugs carry the site name that the catalogue title omits:
    # "librivox-short-humor-collection-001" is catalogued "Short Humor Collection 001".
    slug = re.sub(r"^librivox-", "", slug)
    return slug.replace("-", " ").strip()


def lookup_librivox(url: str) -> dict | None:
    """Best-effort project metadata for a librivox.org URL.

    The API's title filter is a LITERAL match ("^" makes it a literal prefix), so it is
    defeated by exactly the things that differ between a URL slug and a catalogue title:
    a leading article ("a-christmas-carol" vs catalogued "Christmas Carol"), punctuation
    ("frankenstein-or-the-modern-prometheus" vs "Frankenstein, or The Modern
    Prometheus"), and dropped apostrophes ("draculas-guest" never prefix-matches
    "Dracula's Guest"). Six of thirteen queue entries failed on these.

    So: query on short distinctive prefixes AND on the author from the slug, then do the
    real matching locally on normalized titles.

    Returns None rather than a guess when nothing matches. An early version fell back to
    "first result of the broadest probe", which silently matched LibriVox Short Humor
    Collection 001 to LibriVox *Language Learning* Collection Vol. 001 — a wrong row in
    the ledger is worse than an owner confirm.
    """
    q = norm_title(slug_to_query(url))   # article-stripped, depunctuated
    if not q:
        return None
    words = q.split()

    def matches(books: list[dict]) -> dict | None:
        # Prefix predicates match on WORD boundaries only. Without that,
        # "draculas guest and other weird tales" startswith "dracula" is True and the
        # router silently returns a different book by the same author.
        for pred in (lambda t: t == q,
                     lambda t: t.startswith(q + " "),
                     lambda t: q.startswith(t + " ") and len(t.split()) >= 2):
            for b in books:
                if pred(norm_title(b.get("title", ""))):
                    return b
        return None

    probes: list[dict] = [{"title": f"^{' '.join(words[:n])}", "limit": 50}
                          for n in (len(words), 2, 1)]
    # Author probe rescues titles whose punctuation the literal prefix cannot reproduce.
    author = author_from_slug(url)
    if author:
        probes.append({"author": f"^{author}", "limit": 100})

    tried: set[str] = set()
    for probe in probes:
        key = json.dumps(probe, sort_keys=True)
        if key in tried:
            continue
        tried.add(key)
        try:
            books = api_get(probe).get("books") or []
        except Exception as exc:  # noqa: BLE001 - network best-effort
            print(f"    (api error on {probe}: {type(exc).__name__})", file=sys.stderr)
            books = []
        time.sleep(SLEEP_BETWEEN)
        hit = matches(books)
        if hit:
            return hit
    return None


# One definition, in synth_common — `librivox_fetch` needs the same rule to resolve a
# book's canonical ledger key (A-M10). Re-exported under the old name; callers and tests
# already import it from here.
etext_id_from = synth_common.etext_id_from


def is_dramatic(url: str, title: str = "") -> bool:
    blob = f"{url} {title}".lower()
    return "dramatic" in blob or "play" in blob


# ---------------------------------------------------------------- routing
def route(url: str, ledger: dict, lt_by_id: dict, lt_by_title: dict) -> dict:
    rec: dict = {"url": url, "flags": []}
    host = urllib.parse.urlparse(url).netloc.lower()

    if "librivox.org" in host:
        rec["lane"], rec["verdict"] = "force-align", "REAL-AUDIO"
        meta = lookup_librivox(url)
        time.sleep(SLEEP_BETWEEN)
        if meta:
            rec["title"] = meta.get("title")
            rec["url_text_source"] = meta.get("url_text_source")
            rec["sections"] = meta.get("num_sections")
            rec["totaltime"] = meta.get("totaltime")
            rec["etext_id"] = etext_id_from(meta.get("url_text_source"))
        else:
            rec["flags"].append("?? LibriVox API returned no match — confirm by hand")
    elif "standardebooks.org" in host or "gutenberg.org" in host:
        rec["lane"], rec["verdict"] = "synthesize", "SYNTHESIZE"
        rec["title"] = url.rstrip("/").split("/")[-1].replace("-", " ")
        m = re.search(r"ebooks/(\d+)", url)
        if m:
            rec["etext_id"] = m.group(1)
    else:
        rec["lane"], rec["verdict"] = None, "UNKNOWN"
        rec["flags"].append("?? unrecognised host — route by hand")
        return rec

    # A-M9. This built a `pg:` key and nothing else, so a LibriVox book whose Gutenberg
    # edition we had not resolved was checked against NOTHING — every re-route re-created
    # it and reset `status` to "pending force-align ingest", regressing a book that was
    # already fetched and aligned. Silently: the router prints SKIP only for what it found.
    # `resolve_ledger_key` checks every key the book could be filed under.
    key, entry = synth_common.resolve_ledger_key(
        ledger, url=url, etext_id=rec.get("etext_id"))
    rec["ledger_key"] = key
    if entry is not None:
        rec["verdict"] = "SKIP"
        rec["flags"].append(
            f"already in ledger as {key} (status: {entry.get('status') or 'unrecorded'})")
        return rec

    # --- LibriTTS overlap: annotate, never auto-skip -------------------
    hit_id = rec.get("etext_id") in lt_by_id if rec.get("etext_id") else False
    nt = norm_title(rec.get("title") or "")
    nt = ALIASES.get(nt, nt)
    hit_title = nt in lt_by_title if nt else False
    if hit_id or hit_title:
        which = "etext-id" if hit_id else "title (different PG edition)"
        lt_title = lt_by_id.get(rec.get("etext_id")) or lt_by_title.get(nt, "")
        rec["libritts_overlap"] = {"matched_on": which, "libritts_entry": lt_title}
        rec["flags"].append(
            f"LibriTTS-R train-clean-100 already covers this text (matched on {which})"
            + (
                " — but this is a DRAMATIC READING, which LibriTTS's single-narrator "
                "recordings do not contain, so the pick still adds what we lack"
                if is_dramatic(url, rec.get("title") or "")
                else " — marginal value is lower; confirm the pick"
            )
        )
    return rec


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    ap.add_argument("--queue", default=str(QUEUE))
    ap.add_argument("--ledger", default=str(LEDGER))
    args = ap.parse_args()

    qpath, lpath = pathlib.Path(args.queue), pathlib.Path(args.ledger)
    ledger = json.loads(lpath.read_text(encoding="utf-8"))
    lt_by_id, lt_by_title = load_libritts_books()
    print(f"LibriTTS-R train-clean-100: {len(lt_by_id)} books indexed by etext-id\n")

    lines = qpath.read_text(encoding="utf-8").splitlines()
    today = _dt.date.today().isoformat()
    results, out_lines = [], []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*"):
            out_lines.append(line)
            continue
        url = s.split("|")[0].strip()
        print(f"→ {url}")
        rec = route(url, ledger, lt_by_id, lt_by_title)
        rec["routed"] = today
        results.append(rec)
        print(f"    {rec['verdict']:11s} {rec.get('title') or ''}")
        for f in rec["flags"]:
            print(f"    ⚠ {f}")
        out_lines.append(f"* {line}  → {rec['verdict']} ({today})")

    print(f"\n{len(results)} entries routed")
    for v in ("REAL-AUDIO", "SYNTHESIZE", "SKIP", "UNKNOWN"):
        n = sum(1 for r in results if r["verdict"] == v)
        if n:
            print(f"  {v:11s} {n}")
    n_ov = sum(1 for r in results if r.get("libritts_overlap"))
    print(f"  {'overlap':11s} {n_ov} also present in LibriTTS-R (annotated, not skipped)")

    if not args.apply:
        print("\nDRY RUN — pass --apply to write the ledger and mark the queue")
        return 0

    shutil.copy2(lpath, lpath.with_suffix(".json.bak"))
    shutil.copy2(qpath, qpath.with_suffix(".txt.bak"))

    # A-H3. `ledger` was read before a routing pass that makes one network call per book,
    # so writing it back whole erased whatever librivox_fetch recorded in the meantime.
    # Applying the same edits to the CURRENT contents under a lock keeps both.
    def _apply(led):
        for rec in results:
            # A SKIP means "this book is already recorded". Writing it back anyway is how
            # the regression happened: `rec` for a SKIP is the shallow record `route()`
            # built before it returned early, so it would stamp `verdict: SKIP` over the
            # entry's real lane verdict and drop everything learned since.
            if rec["verdict"] == "SKIP":
                continue
            # A-M9 again, on the WRITE side. This derived the key independently of the
            # SKIP check above, so a book already filed under `lv:` was written a second
            # time under `pg:` the moment its etext id became known — two entries, two
            # states, one book. Resolved against the ledger as it is NOW (update_json
            # re-read it under the lock), so the existing key wins if there is one.
            key, entry = synth_common.resolve_ledger_key(
                led, url=rec["url"], etext_id=rec.get("etext_id"))
            if key is None:
                continue
            entry = dict(entry or {})
            was = entry.get("status")
            entry.update(
                {k: v for k, v in rec.items() if k not in ("flags", "ledger_key")},
                ledger_key=key,
            )
            # Never regress a book that is further along. The router's job is to decide a
            # LANE; it knows nothing about whether the audio has been fetched or aligned,
            # and stamping "pending" over "fetched; awaiting align" loses work that
            # nothing else records.
            entry["status"] = was or (
                "pending force-align ingest" if rec["lane"] == "force-align"
                else "pending book_ingest")
            if rec["flags"]:
                entry["router_flags"] = rec["flags"]
            led[key] = entry

    synth_common.update_json(lpath, _apply)
    synth_common.write_text_atomic(qpath, "\n".join(out_lines) + "\n")
    print(f"\nwrote {lpath} and marked {qpath} (backups alongside)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
