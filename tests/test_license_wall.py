"""The licence wall, exercised as a wall — no corpus, no /data, no GPU.

WHY THIS FILE EXISTS (2026-09-09)
---------------------------------
`matcha/data/license_wall.py` had **no tests for `enforce` or `classify_path`**. The only
coverage in the repo was of `refuse_holdout`, a different guard in the same module. What
that cost is measurable and specific: `expresso` was declared with `dirs: [expresso]` while
the only copy on disk lived at `/data/huggingface/datasets/ylacombe___expresso`, so
`classify_path` returned `None` for every file in it and **the declaration matched nothing
for the whole time it existed**.

⚠ THE DATA WAS STILL REFUSED — as *undeclared*, down a different branch. That is the
subtle part and the reason nobody noticed: the wall's OBSERVABLE BEHAVIOUR was correct,
while the mechanism everyone believed was doing it sat idle. A test that only asserted
"NC data is refused" would have passed throughout and proved nothing about the entry.

So the tests below assert **which branch fired and what it named**, not merely that
something raised.
"""

import os
import pathlib
import re
import subprocess
import tempfile

import pytest

wall = pytest.importorskip("matcha.data.license_wall")


@pytest.fixture(autouse=True)
def _clean_manifest_cache():
    """Each test sees the real manifest, freshly read."""
    wall._manifest_cache = None
    wall._publish_cache = None
    yield
    wall._manifest_cache = None
    wall._publish_cache = None


# --- classify_path: the HF-cache spellings ------------------------------------------------

@pytest.mark.parametrize("path", [
    "/data/datasets/expresso/read/x.wav",                             # plain
    "/data/huggingface/datasets/ylacombe___expresso/read/x.wav",      # datasets cache
    "/data/huggingface/hub/datasets--ylacombe--expresso/x.wav",       # hub cache
])
def test_every_on_disk_spelling_of_a_declared_dataset_is_recognised(path):
    """⚠ THE MIDDLE ONE IS THE REAL BUG. It was the only copy that ever existed on this
    machine, and it classified as None until 2026-09-09."""
    hit = wall.classify_path(path)
    assert hit is not None, f"{path} classifies as undeclared — the declaration is decorative"
    assert hit[0] == "expresso"
    assert hit[1] == "blocked"


def test_the_org_half_of_a_cache_name_is_NOT_matched():
    """A publisher is not a dataset. Matching the org half would let one entry's class be
    inherited by every dataset that publisher ships."""
    assert wall.classify_path("/data/huggingface/datasets/ylacombe___something_else/x.wav") is None
    assert wall.classify_path("/data/huggingface/hub/datasets--ylacombe--unrelated/x.wav") is None


@pytest.mark.parametrize("component", ["emilia_kept", "emilia_kept_24k"])
def test_the_yodas_keeps_do_not_collide_with_the_blocked_emilia_entry(component):
    """⚠ LOAD-BEARING. `emilia_original` declares `dirs: [emilia, Emilia]` and is blocked;
    v5-v7 train on the YODAS keeps. Component matching is exact, which is the only thing
    keeping those apart — and the HF-cache splitting added in 2026-09-09 must not weaken it.
    """
    hit = wall.classify_path(f"/data/datasets/{component}/x.wav")
    assert hit is not None and hit[1] == "permissive", hit


def test_the_full_emilia_dataset_repo_is_not_silently_permitted():
    """`amphion___Emilia-Dataset` splits to `Emilia-Dataset`, which is declared nowhere, so
    it must refuse as UNDECLARED rather than resolve to the permissive YODAS entry."""
    assert wall.classify_path("/data/huggingface/datasets/amphion___Emilia-Dataset/x.wav") is None


# --- enforce: which branch fired, and what it named ---------------------------------------

def _filelist(tmp_path, audio_dir, home="emilia_kept_24k", name="train.txt"):
    """Write a filelist whose OWN path is declared, pointing at `audio_dir`.

    ⚠ THE FILELIST PATH IS CLASSIFIED TOO — `enforce` checks `[filelist, *seen_dirs]`. A
    filelist sitting in an undeclared directory raises the UNDECLARED error before the
    blocked branch is ever reached, which is correct behaviour and it broke the first draft
    of three tests here: a bare `tmp_path/train.txt` is undeclared, so they were asserting
    on the wrong refusal. `home` therefore names a declared, permissive directory, and the
    audio paths are what each test is actually varying.
    """
    d = tmp_path / home
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(f"{audio_dir}/a.wav|speaker|text\n{audio_dir}/b.wav|speaker|text\n",
                 encoding="utf-8")
    return str(p)


def test_the_filelist_path_itself_must_be_declared(tmp_path):
    """Documents the property the helper above works around, so it is a stated rule rather
    than a quirk someone rediscovers."""
    p = tmp_path / "train.txt"
    p.write_text("/data/datasets/emilia_kept_24k/a.wav|s|t\n", encoding="utf-8")
    with pytest.raises(wall.LicenseWallError, match="undeclared"):
        wall.enforce([str(p)])


def test_a_blocked_dataset_is_refused_AS_BLOCKED_naming_the_dataset(tmp_path):
    fl = _filelist(tmp_path, "/data/huggingface/datasets/ylacombe___expresso/read")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.enforce([fl])
    msg = str(e.value)
    assert "non-permissive" in msg, msg
    # ⚠ `"-> expresso ("`, NOT `"expresso"` (#418). The offending PATH contains the word and
    # BOTH branches print the path, so the bare form is true whichever fired — a clause that
    # cannot fail, in the one file whose docstring makes "what it named" the standard. The
    # arrow is the format `f"{p} -> {name} ({lic})"` uses to name the DATASET.
    assert "-> expresso (" in msg, "the refusal does not name the dataset"
    # ⚠ NOT the undeclared branch — that is the bug this file was written about.
    assert "undeclared" not in msg, (
        "blocked data refused as UNDECLARED: classify_path is not matching the entry, so "
        "the declaration is decorative again")


def test_an_undeclared_path_is_refused_AS_UNDECLARED_naming_the_path(tmp_path):
    fl = _filelist(tmp_path, "/data/datasets/something_nobody_declared")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.enforce([fl])
    assert "undeclared" in str(e.value)


def test_a_permissive_corpus_passes(tmp_path):
    """A wall that fires on the working path gets switched off."""
    fl = _filelist(tmp_path, "/data/datasets/emilia_kept_24k")
    wall.enforce([fl])


# --- no override exists, and that is the point --------------------------------------------

@pytest.mark.parametrize("value", ["derisk", "off", "0", "permissive", "enforce", ""])
def test_NO_environment_value_can_permit_blocked_data(tmp_path, monkeypatch, value):
    """⚠⚠ THE REGRESSION GUARD FOR THE 2026-09-09 RULING.

    `SONORA_LICENSE_WALL=derisk` used to permit class-`nc` data behind a taint banner. The
    owner removed it with the `nc` class: a dormant permission is the one route restricted
    material could take back into a lineage. `derisk` is parametrised here alongside values
    that never existed, because the failure to catch is someone RE-ADDING a mode — under
    any name — not that one specific string comes back.
    """
    monkeypatch.setenv("SONORA_LICENSE_WALL", value)
    fl = _filelist(tmp_path, "/data/datasets/expresso")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.enforce([fl])
    # ⚠ #414. Which BRANCH, per this file's own docstring: "the wall raised" would also be
    # true if `expresso` fell out of the manifest and was refused as undeclared.
    msg = str(e.value)
    # ⚠ the naming clause is the arrow form, not the bare word — see #418 above.
    assert "non-permissive" in msg and "-> expresso (" in msg, msg
    assert "undeclared" not in msg, f"refused as UNDECLARED, not as blocked: {msg}"


def test_enforce_takes_no_mode_argument():
    """The hatch must not come back as a parameter either."""
    import inspect
    params = list(inspect.signature(wall.enforce).parameters)
    assert params == ["filelist_paths"], (
        f"enforce grew a parameter: {params}. If that is a new mode, it is the escape hatch "
        f"the owner removed on 2026-09-09 wearing a different hat.")


# --- the manifest itself -------------------------------------------------------------------

def test_the_nc_class_is_gone_and_stays_gone():
    """⚠ RETIRED, NOT EMPTIED (owner 2026-09-09). An empty permissive-looking class is a
    hiding place: the next NC arrival gets filed under it and inherits a risk acceptance
    that no longer exists. So the NAME must not reappear, even with no members."""
    import yaml
    with open(wall._MANIFEST_PATH, encoding="utf-8") as f:
        entries = yaml.safe_load(f)["datasets"]
    classes = {e["class"] for e in entries.values()}
    assert "nc" not in classes, (
        f"the `nc` class is back in data_licenses.yaml: {sorted(classes)}. It was retired "
        f"with the Expresso ruling; a non-permissive dataset is `blocked`.")
    assert classes <= {"permissive", "blocked"}, f"undeclared class kind: {sorted(classes)}"


def test_anything_not_permissive_fails_closed(tmp_path, monkeypatch):
    """⚠ `enforce` checks `!= "permissive"`, not `== "blocked"`, so a typo'd or newly
    invented class refuses instead of sailing through. Proven by injecting one."""
    monkeypatch.setattr(wall, "_manifest_cache",
                        {"weirdcorpus": ("weirdcorpus", "probably_fine", "Some-License")})
    fl = _filelist(tmp_path, "/data/datasets/weirdcorpus", home="weirdcorpus")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.enforce([fl])
    assert "non-permissive" in str(e.value)


# --- the publish wall ---------------------------------------------------------------------
#
# A licence is not the only reason an artifact must not ship, so `publish` is a separate axis
# from `class` — folding it in would make the manifest state something false about the licence.
# ⚠ The firewall in `docs/audiobook-corpus-policy.md` has the same shape and is NOT reachable
# through this axis (#473) — that corpus is `class: blocked` and `enforce` refuses it at
# training time, one wall earlier. `private_audiobook_v1` below is a FIXTURE NAME standing in
# for some future corpus, not a claim that such a corpus can get here.
#
# ⚠⚠ THESE FIXTURES USED TO BE THE CROSSED DELIVERY BANK, AND THAT EXAMPLE IS RETIRED.
# Owner ruling 12 answered 2026-09-09 that the bank was diagnostic only; the owner REVISED it
# on 2026-09-10 and the bank is PUBLISHABLE. A test asserting `crossed_bank_v8` is
# `publish: forbidden` teaches a reader the opposite of the live ruling, which is why the
# fixture corpus was renamed rather than left alone. Nothing about the guard changed.
# (#474: this sentence said `private_audiobook_v1` — the NEW name — because the rename that
# it exists to explain was done with a blanket replace that rewrote the explanation too.)
#
# ⚠ NO LIVE MANIFEST ENTRY IS `publish: forbidden` TODAY, so these tests inject the policy.
# That is the honest way to test a guard whose corpus has not been built — and the
# alternative, shipping the mechanism untested until one arrives, is how #391 shipped a guard
# that could not fire.

def _publish_manifest(monkeypatch, policy, reason="Private lineage: nothing trained on it leaves the machine."):
    """Inject a corpus carrying `policy` and a permissive neighbour, bypassing the yaml."""
    monkeypatch.setattr(wall, "_manifest_cache", {
        "private_audiobook_v1": ("private_audiobook_v1", "permissive", "CC-BY-4.0"),
        "ordinary_corpus": ("ordinary_corpus", "permissive", "CC-BY-4.0"),
    })
    monkeypatch.setattr(wall, "_publish_cache", {
        "private_audiobook_v1": ("private_audiobook_v1", policy, reason),
        "ordinary_corpus": ("ordinary_corpus", "allowed", ""),
    })


def test_a_corpus_marked_publish_forbidden_refuses_at_export(monkeypatch):
    _publish_manifest(monkeypatch, "forbidden")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable(["data/private_audiobook_v1/train_op.txt"], what="the ckpt")
    msg = str(e.value)
    assert "-> private_audiobook_v1 (publish: forbidden)" in msg, msg
    assert "Private lineage" in msg, "the recorded reason is not surfaced to the operator"
    # ⚠ IT MUST NOT READ AS A LICENCE PROBLEM. An operator who thinks this is a licence
    # refusal goes looking at the licence, finds it clean, and concludes the wall is broken.
    assert "NOT a licence refusal" in msg, msg


def test_a_publishable_corpus_passes(monkeypatch):
    _publish_manifest(monkeypatch, "forbidden")
    wall.refuse_unpublishable(["data/ordinary_corpus/train_op.txt"])


def test_the_LIVE_v7_corpus_is_still_publishable():
    """Against the real manifest — a guard that fires on the working path gets removed."""
    wall.refuse_unpublishable(["data/libritts_r_full_vat_v7/train_op.txt",
                               "data/libritts_r_full_vat_v7/val_op.txt"])


def test_publish_is_orthogonal_to_licence_class(monkeypatch):
    """⚠ THE WHOLE POINT OF THE SEPARATE AXIS. The injected corpus is `permissive` — a clean
    licence — and still must not ship. If these two ever collapse into one field, this fails.
    """
    _publish_manifest(monkeypatch, "forbidden")
    assert wall.classify_path("data/private_audiobook_v1/x.txt")[1] == "permissive"
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(["data/private_audiobook_v1/train_op.txt"])


# --- lineage extraction, against the shape a real checkpoint actually has ------------------

def test_lineage_comes_out_of_datamodule_hyper_parameters():
    """⚠ THE KEYS ARE MEASURED, NOT ASSUMED. Read out of a real 263 MB
    `vat7_finetune` checkpoint on 2026-09-09 by unzipping `archive/data.pkl` and reading the
    literal strings — no torch on the host, and nothing executed. The values found were
    `data/libritts_r_full_vat_v7/{train_op,val_op}.txt`.
    """
    ck = {"datamodule_hyper_parameters": {
        "train_filelist_path": "data/libritts_r_full_vat_v7/train_op.txt",
        "valid_filelist_path": "data/libritts_r_full_vat_v7/val_op.txt",
        "batch_size": 32}}
    assert wall.lineage_filelists(ck) == ["data/libritts_r_full_vat_v7/train_op.txt",
                                          "data/libritts_r_full_vat_v7/val_op.txt"]


def test_a_checkpoint_with_no_lineage_yields_nothing_not_a_false_clean():
    """Empty means UNKNOWN. The caller must not read it as verified-clean, and the export
    path's comment says so — pre-wall checkpoints exist."""
    assert wall.lineage_filelists({"state_dict": {}}) == []
    assert wall.lineage_filelists({"datamodule_hyper_parameters": {}}) == []


# --- and the export path must actually CALL it (#391's lesson) -----------------------------

_LINEAGE_READ = "lineage_filelists(ck)"
_WALL_CALL = "refuse_unpublishable(lineage,"
_UNPACK = 'hp = ck["hyper_parameters"]'


def _wall_runs_before_unpack(src):
    """True when the publish wall's CALL precedes the checkpoint unpack in `src`.

    ⚠ INDEXED ON THE CALL, NOT THE NAME (#427). It was `src.index("refuse_unpublishable")`,
    and `str.index` returns the FIRST occurrence — which in `convert_vat.py` is the local
    `from … import` line one line above the call, not the call. So the assertion measured
    where the IMPORT sat: moving the call below the unpack while leaving the import in place
    kept it green, which is the one mutation its failure message names. Measured both ways in
    `test_the_ordering_check_fires_when_the_call_moves` below.
    """
    return src.index(_WALL_CALL) < src.index(_UNPACK)


def test_the_export_path_calls_the_publish_wall():
    """⚠ A GUARD NOTHING INVOKES IS NOT A GUARD. #391 shipped one that could never fire; the
    tests around it all passed because they exercised the function, never the call site.
    """
    src = (pathlib.Path(wall._MANIFEST_PATH).parent.parent
           / "scripts" / "litert_export" / "convert_vat.py").read_text(encoding="utf-8")
    assert _LINEAGE_READ in src and _WALL_CALL in src, (
        "convert_vat.py no longer calls the publish wall over the checkpoint's own lineage — "
        "the ruling is unenforced")
    # before the graph is built, not after
    assert _wall_runs_before_unpack(src), (
        "the publish wall runs after the checkpoint is unpacked; refuse before doing work")


def test_the_ordering_check_fires_when_the_call_moves():
    """The positive control for the assertion above: an empty result and a broken instrument
    are indistinguishable, so prove this one can report dirty. The mutation is the finding's
    own case — the call moved below the unpack, the import left where it was.
    """
    src = (pathlib.Path(wall._MANIFEST_PATH).parent.parent
           / "scripts" / "litert_export" / "convert_vat.py").read_text(encoding="utf-8")
    call_line = next(ln for ln in src.splitlines(keepends=True) if _WALL_CALL in ln)
    unpack_line = next(ln for ln in src.splitlines(keepends=True) if _UNPACK in ln)
    moved = src.replace(call_line, "").replace(unpack_line, unpack_line + call_line)

    assert moved.index("refuse_unpublishable") < moved.index(_UNPACK), (
        "the mutation is supposed to leave the IMPORT above the unpack — that is what made "
        "the old name-indexed assertion unable to fire")
    assert not _wall_runs_before_unpack(moved), (
        "the ordering check passed a source whose wall call runs after the unpack")


# --- #420: the realistic shape is a filelists-only corpus whose AUDIO is the bank ----------

def _merged_publish_manifest(monkeypatch):
    """A v8 built the way v5-v7 were: its dir appended to `merged_vat_corpora` (allowed), its
    audio living under the bank's own directory (forbidden)."""
    monkeypatch.setattr(wall, "_manifest_cache", {
        "libritts_r_full_vat_v8": ("merged_vat_corpora", "permissive", "CC-BY-4.0"),
        "private_audiobook_v1": ("private_audiobook_v1", "permissive", "CC-BY-4.0"),
        "ordinary_corpus": ("ordinary_corpus", "permissive", "CC-BY-4.0"),
    })
    monkeypatch.setattr(wall, "_publish_cache", {
        "libritts_r_full_vat_v8": ("merged_vat_corpora", "allowed", ""),
        "private_audiobook_v1": ("private_audiobook_v1", "forbidden", "Private lineage: nothing trained on it leaves the machine."),
        "ordinary_corpus": ("ordinary_corpus", "allowed", ""),
    })


def _merged_filelist(tmp_path, audio_dir):
    d = tmp_path / "data" / "libritts_r_full_vat_v8"
    d.mkdir(parents=True)
    fl = d / "train_op.txt"
    fl.write_text(f"{audio_dir}/a.wav|1|x|0\n{audio_dir}/b.wav|1|y|0\n", encoding="utf-8")
    return fl


def test_a_merged_corpus_whose_AUDIO_is_forbidden_refuses(tmp_path, monkeypatch):
    """⚠ THE CASE #420 NAMED. The filelist path classifies as allowed; only the audio rows
    name the bank. The old wall walked the path alone and passed this."""
    _merged_publish_manifest(monkeypatch)
    fl = _merged_filelist(tmp_path, "/data/private_audiobook_v1/wavs")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable([str(fl)])
    assert "-> private_audiobook_v1 (publish: forbidden)" in str(e.value), str(e.value)


def test_a_merged_corpus_whose_audio_is_all_publishable_passes(tmp_path, monkeypatch):
    _merged_publish_manifest(monkeypatch)
    fl = _merged_filelist(tmp_path, "/data/ordinary_corpus/wavs")
    assert wall.refuse_unpublishable([str(fl)]) == []


def test_a_repo_relative_filelist_resolves_against_root(tmp_path, monkeypatch):
    """The export runs from the LiteRT work dir, so without `root` nothing would open.

    ⚠ THE NO-ROOT HALF RUNS FROM AN EMPTY DIR, AND THAT IS THE POINT (#429). With `root=None`
    the open is relative to the process cwd, which under pytest is the repo root — where
    `data/` is the live corpus mount. So "cannot open it" was a property of the HOST (no
    `libritts_r_full_vat_v8/` there yet), not of the test: the day a v8 is derived the file
    opens, its audio is not in the injected cache, and the assertion flips to `[]`. Measured.
    ⚠ Not `chdir(tmp_path)`, which the finding's remedy named: the fixture writes the
    filelist THERE, so that cwd opens it and the call raises instead. Measured too.
    """
    _merged_publish_manifest(monkeypatch)
    _merged_filelist(tmp_path, "/data/private_audiobook_v1/wavs")
    rel = "data/libritts_r_full_vat_v8/train_op.txt"
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable([rel], root=str(tmp_path))
    # and WITHOUT root it cannot open the file: reported as unread, not passed as clean
    nowhere = tmp_path / "cwd-with-no-data-dir"
    nowhere.mkdir()
    monkeypatch.chdir(nowhere)
    assert wall.refuse_unpublishable([rel]) == [rel]


def test_an_unreadable_filelist_still_checks_its_own_path(monkeypatch):
    _merged_publish_manifest(monkeypatch)
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(["data/private_audiobook_v1/train_op.txt"])


# --- #422: a misspelled key must be a load error, not a silent `allowed` ------------------

def _manifest_file(tmp_path, monkeypatch, entry_body):
    p = tmp_path / "data_licenses.yaml"
    p.write_text("datasets:\n  bank:\n" + entry_body, encoding="utf-8")
    monkeypatch.setattr(wall, "_MANIFEST_PATH", str(p))
    return p


def test_a_misspelled_publish_KEY_refuses_at_load(tmp_path, monkeypatch):
    _manifest_file(tmp_path, monkeypatch,
                   "    dirs: [bank]\n    license: L\n    class: permissive\n    publsh: forbidden\n")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable(["data/bank/t.txt"])
    assert "publsh" in str(e.value)


def test_a_misspelled_publish_VALUE_refuses_at_load(tmp_path, monkeypatch):
    _manifest_file(tmp_path, monkeypatch,
                   "    dirs: [bank]\n    license: L\n    class: permissive\n    publish: forbiden\n")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.classify_path("data/bank/t.txt")
    assert "forbiden" in str(e.value)


def test_a_correctly_spelled_forbidden_entry_loads_and_fires(tmp_path, monkeypatch):
    """Positive control for the two above: the validator must let a RIGHT entry through."""
    _manifest_file(tmp_path, monkeypatch,
                   "    dirs: [bank]\n    license: L\n    class: permissive\n    publish: forbidden\n")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable(["data/bank/t.txt"])
    assert "publish: forbidden" in str(e.value)


def test_the_live_manifest_passes_the_key_and_value_validation():
    assert wall._manifest() and wall._publish()


# --- #424: the two maps fill themselves independently --------------------------------------

def test_injecting_only_the_licence_map_does_not_break_the_publish_wall(monkeypatch):
    monkeypatch.setattr(wall, "_manifest_cache",
                        {"weirdcorpus": ("weirdcorpus", "probably_fine", "Some-License")})
    assert wall._publish_cache is None
    # no AttributeError, no refusal; the (nonexistent) filelist is reported unread
    assert wall.refuse_unpublishable(["data/weirdcorpus/t.txt"]) == ["data/weirdcorpus/t.txt"]


# --- #421: ancestors ride in the checkpoint ------------------------------------------------

def test_lineage_includes_the_donor_stages(monkeypatch):
    ck = {wall.LINEAGE_KEY: ["data/private_audiobook_v1/train_op.txt"],
          "datamodule_hyper_parameters": {
              "train_filelist_path": "data/libritts_r_full_vat_v7/train_op.txt"}}
    assert wall.lineage_filelists(ck) == ["data/private_audiobook_v1/train_op.txt",
                                          "data/libritts_r_full_vat_v7/train_op.txt"]
    _merged_publish_manifest(monkeypatch)
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(wall.lineage_filelists(ck))


def test_the_warm_start_and_the_module_carry_the_lineage():
    """Source-level, because there is no torch on this host: the init must be given the
    donor lineage, and every later save must write it back out."""
    root = pathlib.Path(wall._MANIFEST_PATH).parent.parent
    ws = (root / "scripts" / "lib" / "make_warmstart.py").read_text(encoding="utf-8")
    # ⚠ `carried_lineage`, not `lineage_filelists`, since #425 — the difference is what a
    # pre-wall donor writes, and it is exercised for real in the tests below rather than
    # asserted as a string here.
    assert "model.sonora_lineage = carried_lineage(donor)" in ws
    mod = (root / "matcha" / "models" / "baselightningmodule.py").read_text(encoding="utf-8")
    assert "def on_save_checkpoint" in mod and "checkpoint[LINEAGE_KEY]" in mod
    # ⚠ `carried_lineage` on the LOAD side too, since #425. `make_warmstart.py` only WRITES an
    # init; the lane starts a fine-tune with `ckpt_path=<init>`, so the init is adopted here.
    assert "self.sonora_lineage = carried_lineage(checkpoint)" in mod


# --- #425/#426: an UNKNOWN ancestry must survive the warm start it is discovered at --------
#
# The chain below is what the module's hooks do, written out as dicts because there is no
# torch on this host: `on_save_checkpoint` writes `LINEAGE_KEY` from `self.sonora_lineage`,
# and `on_load_checkpoint` sets `self.sonora_lineage` from `carried_lineage(checkpoint)`.
# `test_the_warm_start_and_the_module_carry_the_lineage` above pins that those two lines are
# still the ones in the module; these exercise what they carry.
#
# ⚠ MODEL THE LOAD STEP WITH `carried_lineage`, NOT `lineage_filelists` (#430). This comment
# said the latter for one commit, and it is the block a later test author copies to write the
# next chain test — which is exactly how the pass-1 test for #425 modelled the load door as
# `lineage_filelists`, missed that the door converts, and earned the reopen. The difference
# only shows on a checkpoint with nothing recorded, which is the case that matters here.

def _save(lineage, corpus=None):
    """One checkpoint written by the module's hooks: the lineage it holds, plus — for a real
    training run — the datamodule hparams Lightning records for the stage itself."""
    ck = {wall.LINEAGE_KEY: list(lineage)}
    if corpus:
        ck["datamodule_hyper_parameters"] = {"train_filelist_path": f"{corpus}/train_op.txt",
                                             "valid_filelist_path": f"{corpus}/val_op.txt"}
    return ck


def test_a_pre_wall_donor_yields_the_unknown_marker_not_an_empty_list():
    assert wall.carried_lineage({"state_dict": {}}) == [wall.LINEAGE_UNKNOWN]


def test_a_donor_that_recorded_its_corpus_carries_no_marker():
    """The positive control: the marker must appear ONLY for an unrecorded ancestry, or every
    checkpoint in the current lineage would report UNKNOWN and the signal would be worthless."""
    donor = _save([], corpus="data/ordinary_corpus")
    assert wall.carried_lineage(donor) == ["data/ordinary_corpus/train_op.txt",
                                           "data/ordinary_corpus/val_op.txt"]
    assert wall.LINEAGE_UNKNOWN not in wall.carried_lineage(donor)


def test_the_unknown_marker_survives_a_warm_start_and_a_fine_tune(monkeypatch):
    """⚠ THE CASE #425 NAMED, end to end. Before this the donor's UNKNOWN was an empty list,
    the fine-tune added its own corpus, and the descendant read as a complete record."""
    _merged_publish_manifest(monkeypatch)
    init = _save(wall.carried_lineage({"state_dict": {}}))          # make_warmstart writes it
    loaded = wall.carried_lineage(init)                              # on_load_checkpoint
    finetune = _save(loaded, corpus="data/libritts_r_full_vat_v8")   # on_save_checkpoint

    lineage = wall.lineage_filelists(finetune)
    assert wall.LINEAGE_UNKNOWN in lineage, "the donor's UNKNOWN did not survive the warm start"
    assert "data/libritts_r_full_vat_v8/train_op.txt" in lineage, "the fine-tune's own corpus"

    gaps = wall.lineage_gaps(lineage, wall.refuse_unpublishable(lineage))
    assert any("warm-started from a pre-wall donor" in g for g in gaps), gaps


# ⚠ THE INIT THAT ACTUALLY EXISTS, not the one make_warmstart would write today (#425 pass 2).
# Measured 2026-09-10 by reading `archive/data.pkl` out of each zip, no torch: all eight
# `*_init.ckpt` under /data/model-training/sonora/warmstart/ — derisk_energy, vat3, vat3c,
# vat4, vat5, vat6, vat7, vat7r — contain neither `datamodule_hyper_parameters` nor
# `sonora_lineage`. The probe has a positive control: `matcha_vctk.ckpt` and a real training
# checkpoint both come back True for the hparams, so the eight Falses are a result and not a
# broken probe. Those inits are what `ckpt_path=` hands to `on_load_checkpoint`.
_AN_INIT_ON_DISK = {"epoch": 42, "global_step": 0, "state_dict": {}}


def test_an_init_that_already_exists_on_disk_is_adopted_as_unknown():
    """The load side of #425. Fixing only `make_warmstart.py` left every init already built
    going through the old path, because the lane adopts an init through `ckpt_path=`."""
    assert wall.lineage_filelists(_AN_INIT_ON_DISK) == [], (
        "the fixture no longer matches what was measured on disk — re-probe before trusting "
        "the test below, because it is the emptiness that makes this case the defect")
    assert wall.carried_lineage(_AN_INIT_ON_DISK) == [wall.LINEAGE_UNKNOWN]


def test_a_run_resumed_from_its_own_checkpoint_is_not_marked_unknown():
    """The positive control for the load side: `on_load_checkpoint` also fires on an ordinary
    auto-resume, and a run that recorded its corpus must NOT acquire the marker — or every
    resumed run would report UNKNOWN and the signal would mean nothing."""
    resumed = _save([], corpus="data/libritts_r_full_vat_v8")   # Lightning writes dm hparams
    assert wall.carried_lineage(resumed) == ["data/libritts_r_full_vat_v8/train_op.txt",
                                             "data/libritts_r_full_vat_v8/val_op.txt"]
    assert wall.LINEAGE_UNKNOWN not in wall.carried_lineage(resumed)


def test_the_two_doors_agree(monkeypatch):
    """Adopting an on-disk init through the LOAD hook must reach the same descendant lineage
    as one written by a fixed `make_warmstart`. The bug was the two doors disagreeing."""
    _merged_publish_manifest(monkeypatch)
    via_writer = _save(wall.carried_lineage({"state_dict": {}}))
    via_loader = _save(wall.carried_lineage(_AN_INIT_ON_DISK))
    assert wall.lineage_filelists(via_writer) == wall.lineage_filelists(via_loader)


def test_the_marker_is_not_a_path_so_an_unaware_reader_still_fails_closed():
    """The fail-closed property `carried_lineage` claims, measured rather than asserted in
    prose: a caller that knows nothing about the marker gets it back as unread, which every
    caller already treats as not-clean."""
    assert wall.refuse_unpublishable([wall.LINEAGE_UNKNOWN]) == [wall.LINEAGE_UNKNOWN]
    assert wall.publish_policy(wall.LINEAGE_UNKNOWN) is None


def test_an_empty_lineage_is_a_reported_gap_not_a_silent_pass():
    """#426: `refuse_unpublishable([])` returns `[]`, so the export's report loop had nothing
    to print and a checkpoint with nothing to check logged exactly like a cleared one."""
    assert wall.refuse_unpublishable([]) == []
    gaps = wall.lineage_gaps([], [])
    assert len(gaps) == 1 and "no datamodule hparams and no lineage key" in gaps[0], gaps


def test_a_complete_lineage_reports_no_gaps(tmp_path, monkeypatch):
    """The positive control for `lineage_gaps`: prove it can report clean, or 'no gaps' is
    indistinguishable from a function that never reports anything."""
    _merged_publish_manifest(monkeypatch)
    fl = _merged_filelist(tmp_path, "/data/ordinary_corpus/wavs")
    lineage = [str(fl)]
    assert wall.lineage_gaps(lineage, wall.refuse_unpublishable(lineage)) == []


def test_the_marker_is_reported_once_and_not_as_an_unopenable_file():
    """It lands in `unread` by design, so `lineage_gaps` must not also report it as a file
    that would not open — one condition, one message."""
    lineage = [wall.LINEAGE_UNKNOWN, "data/ordinary_corpus/train_op.txt"]
    unread = [wall.LINEAGE_UNKNOWN, "data/ordinary_corpus/train_op.txt"]
    gaps = wall.lineage_gaps(lineage, unread)
    assert sum("pre-wall donor" in g for g in gaps) == 1, gaps
    assert not any(wall.LINEAGE_UNKNOWN in g and "could not open" in g for g in gaps), gaps
    assert any("could not open data/ordinary_corpus/train_op.txt" in g for g in gaps), gaps


def test_both_readers_use_the_shared_gap_report():
    """#426 was one reader saying less than the other about the same condition. Source-level
    for the two files that need torch to run."""
    root = pathlib.Path(wall._MANIFEST_PATH).parent.parent
    for rel in (("scripts", "litert_export", "convert_vat.py"),
                ("scripts", "tools", "check_publishable.py")):
        src = root.joinpath(*rel).read_text(encoding="utf-8")
        assert "lineage_gaps(lineage, unread)" in src, rel


# --- the promotion tool's exit codes, which are its whole machine-readable interface -------

# The repo venv has no torch (AGENTS.md §3), so the exit-code tests below need a torch-capable
# interpreter from somewhere else or they cannot run at all.
#
# ⚠ THIS USED TO READ `REVIEWER_TORCH_PY` FROM `FerroStep/workflow/config.env`, WHICH WAS
# DELETED WITH THE REVIEW CYCLE (2026-09-15). That key existed to grant the reviewer an
# interpreter that could open a checkpoint; with no reviewer there is no config to read, so
# the path moved to an environment variable with the same default it always had.
#
# ⚠ Returning None makes the dependent tests SKIP, not pass. That is deliberate and it is
# also the risk: a skip is invisible in a green run. If these ever need to be load-bearing on
# a given host, set the variable there rather than assuming the default resolves.
# ⚠ DERIVED FROM `run.sh`, NOT COPIED. `scripts/litert_export/run.sh` OWNS this default — it
# composes it from `SONORA_LITERT_WORK` — and #434 was three passes of prose claiming how many
# copies existed, each correct when written and unable to fail afterwards. A literal here would
# have been a fourth.
REPO = pathlib.Path(__file__).resolve().parent.parent
_RUN_SH = REPO / "scripts" / "litert_export" / "run.sh"


def _default_torch_py():
    src = _RUN_SH.read_text(encoding="utf-8")
    work = re.search(r'SONORA_LITERT_WORK="\$\{SONORA_LITERT_WORK:-([^}"]+)\}"', src)
    py = re.search(r'PY="\$\{SONORA_LITERT_PY:-\$\{SONORA_LITERT_WORK\}([^}"]+)\}"', src)
    assert work and py, "run.sh no longer composes the interpreter path the way this derives it"
    return work.group(1) + py.group(1)


TORCH_PY_DEFAULT = _default_torch_py()

# ⚠ `run.sh` IS THE SOURCE, NOT A COPY, and it is excluded from the comparison for a reason
# worth stating: it never contains the literal. It composes the path from `SONORA_LITERT_WORK`,
# so asserting "run.sh contains run.sh's value" fails on a correct file — measured, on the
# first run of this test. The source is derived FROM; only the copies are compared TO it.
_INTERPRETER_SOURCE = "scripts/litert_export/run.sh"

# Every OTHER file allowed to spell the path out, and why.
# ⚠ ENROLMENT IS NOT PERMISSION TO DRIFT — the test below compares each against `run.sh`.
_INTERPRETER_COPIES = {
    "scripts/tools/check_publishable.py": "a pasteable command in the docstring, so the reader "
                                          "does not have to complete a shape",
}


def test_every_literal_copy_of_the_interpreter_is_enrolled():
    """⚠ COMPLETENESS BY CONSTRUCTION, restored 2026-09-15 after the review lane took the
    original with it. A number in prose is not a mechanism (#434); a scan is.

    ⚠ The scan is over TRACKED files — an untracked scratch copy is not the repo's problem,
    and including it would make this fail on anyone's working directory.
    """
    out = subprocess.run(["git", "grep", "-l", "litert-conversion/.venv/bin/python"],
                         capture_output=True, text=True, cwd=str(REPO))
    # rc 1 = no matches, which is itself a finding: the population cannot be empty.
    found = {l.strip() for l in out.stdout.splitlines() if l.strip()}
    found.discard("tests/test_license_wall.py")     # this file names it only in prose
    assert found, "no file spells the interpreter path — the enrolment list guards nothing"
    unenrolled = sorted(found - set(_INTERPRETER_COPIES) - {_INTERPRETER_SOURCE})
    assert not unenrolled, (
        "these files spell out the interpreter path and are not enrolled in "
        "_INTERPRETER_COPIES, so nothing compares them against run.sh: %s" % unenrolled)


def test_every_enrolled_copy_matches_run_sh():
    """⚠ POSITIVE CONTROL for the enrolment: a list nobody compares is a list."""
    for rel in _INTERPRETER_COPIES:
        src = (REPO / rel).read_text(encoding="utf-8")
        assert TORCH_PY_DEFAULT in src, (
            "%s is enrolled but does not contain %r — run.sh changed and this copy did not "
            "follow" % (rel, TORCH_PY_DEFAULT))


def _torch_interpreter():
    """A torch-capable interpreter, or None if this host has none."""
    p = os.environ.get("SONORA_TORCH_PY", TORCH_PY_DEFAULT).strip()
    return p if p and os.access(p, os.X_OK) else None


def test_an_unreadable_checkpoint_is_could_not_run_not_refused():
    """⚠ A LOAD FAILURE EXITED 1, THE CODE THE DOCSTRING ASSIGNS TO "refused".

    `torch.load` raised through `main()`, so a bad path produced a traceback and status 1 — the
    same answer as the wall refusing an artifact that must not ship. §7 is a hand checklist whose
    only machine-readable output is this number, so the two cases have to differ.

    ⚠⚠ EVERY SHAPE OF UNREADABLE, NOT JUST A MISSING PATH (#436). The first version of this test
    covered only `/no/such.ckpt`, so the handler could have been narrowed back to a tuple of
    types and stayed green — and the tuple it replaced genuinely missed one: measured under the
    granted interpreter, a file that exists but is not a checkpoint raises
    `_pickle.UnpicklingError`, which escaped and exited 1. The other three were caught, which is
    exactly why the hole read as closed. Each row below is a DIFFERENT exception type reaching
    the same handler:

        garbage text file  -> UnpicklingError (PickleError, NOT an OSError)
        truncated zip      -> RuntimeError
        a directory        -> IsADirectoryError (OSError)
        missing path       -> FileNotFoundError (OSError)

    Runs the real tool, because the defect is in what the PROCESS exits with and no assertion
    about an `except` clause can see that. Skipped with its reason printed where no interpreter
    with torch exists — the repo venv deliberately has none.
    """
    py = _torch_interpreter()
    if not py:
        pytest.skip("no interpreter with torch on this host (REVIEWER_TORCH_PY absent)")
    repo = pathlib.Path(wall._MANIFEST_PATH).parent.parent
    tool = repo / "scripts" / "tools" / "check_publishable.py"
    donor = pathlib.Path("/data/model-training/sonora/warmstart/vat7_init.ckpt")
    if not donor.exists():
        pytest.skip("no warmstart init on this host: no control, and no bytes to truncate")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "garbage.ckpt").write_text("not a checkpoint at all\n", encoding="utf-8")
        (tmp / "truncated.ckpt").write_bytes(donor.read_bytes()[:200])
        cases = {
            "garbage text file": tmp / "garbage.ckpt",
            "truncated checkpoint": tmp / "truncated.ckpt",
            "a directory": tmp,
            "missing path": tmp / "no" / "such.ckpt",
        }
        for label, path in cases.items():
            r = subprocess.run([py, str(tool), str(path)], cwd=str(repo),
                               capture_output=True, text=True, timeout=300)
            assert r.returncode == 3, (
                f"{label} exited {r.returncode}; 1 means REFUSED and 3 means it could not run, "
                f"so this input is reported as a publish refusal. stderr: {r.stderr[-400:]}")
            assert "NOT a publish refusal" in r.stderr, f"{label}: {r.stderr[-300:]}"

    # ⚠ THE CONTROL: prove the tool still reaches a real verdict, or every 3 above could mean it
    # is broken for all inputs and the assertions pass on a corpse.
    real = subprocess.run([py, str(tool), str(donor)], cwd=str(repo),
                          capture_output=True, text=True, timeout=300)
    assert real.returncode == 2, (
        f"the control checkpoint exited {real.returncode}, expected 2 (lineage UNKNOWN) — so the "
        f"3s above are not attributable to the inputs. stderr: {real.stderr[-400:]}")
