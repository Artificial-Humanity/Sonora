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


# --- the publish wall (owner ruling 12, 2026-09-09) ---------------------------------------
#
# A licence is not the only reason an artifact must not ship. The crossed delivery bank is
# CC-BY-4.0 and genuinely `permissive`; publishing a model trained on it is forbidden because
# it holds ~20 cloned REAL LibriTTS-R voices. Licence and voice identity are different
# questions, so `publish` is a separate axis from `class` — folding it in would make the
# manifest state something false about the licence.
#
# ⚠ THE BANK DOES NOT EXIST YET, so nothing in the LIVE manifest is `publish: forbidden` and
# these tests inject the policy. That is the honest way to test a guard for a corpus that has
# not been built — and the alternative, shipping the mechanism untested until the bank
# arrives, is how #391 shipped a guard that could not fire.

def _publish_manifest(monkeypatch, policy, reason="Diagnostic only (owner ruling 12)."):
    """Inject a corpus carrying `policy` and a permissive neighbour, bypassing the yaml."""
    monkeypatch.setattr(wall, "_manifest_cache", {
        "crossed_bank_v8": ("crossed_bank_v8", "permissive", "CC-BY-4.0"),
        "ordinary_corpus": ("ordinary_corpus", "permissive", "CC-BY-4.0"),
    })
    monkeypatch.setattr(wall, "_publish_cache", {
        "crossed_bank_v8": ("crossed_bank_v8", policy, reason),
        "ordinary_corpus": ("ordinary_corpus", "allowed", ""),
    })


def test_a_corpus_marked_publish_forbidden_refuses_at_export(monkeypatch):
    _publish_manifest(monkeypatch, "forbidden")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable(["data/crossed_bank_v8/train_op.txt"], what="the ckpt")
    msg = str(e.value)
    assert "-> crossed_bank_v8 (publish: forbidden)" in msg, msg
    assert "Diagnostic only" in msg, "the recorded reason is not surfaced to the operator"
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
    assert wall.classify_path("data/crossed_bank_v8/x.txt")[1] == "permissive"
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(["data/crossed_bank_v8/train_op.txt"])


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

def test_the_export_path_calls_the_publish_wall():
    """⚠ A GUARD NOTHING INVOKES IS NOT A GUARD. #391 shipped one that could never fire; the
    tests around it all passed because they exercised the function, never the call site.
    """
    src = (pathlib.Path(wall._MANIFEST_PATH).parent.parent
           / "scripts" / "litert_export" / "convert_vat.py").read_text(encoding="utf-8")
    assert "refuse_unpublishable(lineage_filelists(ck)" in src, (
        "convert_vat.py no longer calls the publish wall — the ruling is unenforced")
    # before the graph is built, not after
    assert src.index("refuse_unpublishable") < src.index('hp = ck["hyper_parameters"]'), (
        "the publish wall runs after the checkpoint is unpacked; refuse before doing work")


# --- #420: the realistic shape is a filelists-only corpus whose AUDIO is the bank ----------

def _merged_publish_manifest(monkeypatch):
    """A v8 built the way v5-v7 were: its dir appended to `merged_vat_corpora` (allowed), its
    audio living under the bank's own directory (forbidden)."""
    monkeypatch.setattr(wall, "_manifest_cache", {
        "libritts_r_full_vat_v8": ("merged_vat_corpora", "permissive", "CC-BY-4.0"),
        "crossed_bank_v8": ("crossed_bank_v8", "permissive", "CC-BY-4.0"),
        "ordinary_corpus": ("ordinary_corpus", "permissive", "CC-BY-4.0"),
    })
    monkeypatch.setattr(wall, "_publish_cache", {
        "libritts_r_full_vat_v8": ("merged_vat_corpora", "allowed", ""),
        "crossed_bank_v8": ("crossed_bank_v8", "forbidden", "Diagnostic only (owner ruling 12)."),
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
    fl = _merged_filelist(tmp_path, "/data/crossed_bank_v8/wavs")
    with pytest.raises(wall.LicenseWallError) as e:
        wall.refuse_unpublishable([str(fl)])
    assert "-> crossed_bank_v8 (publish: forbidden)" in str(e.value), str(e.value)


def test_a_merged_corpus_whose_audio_is_all_publishable_passes(tmp_path, monkeypatch):
    _merged_publish_manifest(monkeypatch)
    fl = _merged_filelist(tmp_path, "/data/ordinary_corpus/wavs")
    assert wall.refuse_unpublishable([str(fl)]) == []


def test_a_repo_relative_filelist_resolves_against_root(tmp_path, monkeypatch):
    """The export runs from the LiteRT work dir, so without `root` nothing would open."""
    _merged_publish_manifest(monkeypatch)
    _merged_filelist(tmp_path, "/data/crossed_bank_v8/wavs")
    rel = "data/libritts_r_full_vat_v8/train_op.txt"
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable([rel], root=str(tmp_path))
    # and WITHOUT root it cannot open the file: reported as unread, not passed as clean
    assert wall.refuse_unpublishable([rel]) == [rel]


def test_an_unreadable_filelist_still_checks_its_own_path(monkeypatch):
    _merged_publish_manifest(monkeypatch)
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(["data/crossed_bank_v8/train_op.txt"])


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
    ck = {wall.LINEAGE_KEY: ["data/crossed_bank_v8/train_op.txt"],
          "datamodule_hyper_parameters": {
              "train_filelist_path": "data/libritts_r_full_vat_v7/train_op.txt"}}
    assert wall.lineage_filelists(ck) == ["data/crossed_bank_v8/train_op.txt",
                                          "data/libritts_r_full_vat_v7/train_op.txt"]
    _merged_publish_manifest(monkeypatch)
    with pytest.raises(wall.LicenseWallError):
        wall.refuse_unpublishable(wall.lineage_filelists(ck))


def test_the_warm_start_and_the_module_carry_the_lineage():
    """Source-level, because there is no torch on this host: the init must be given the
    donor lineage, and every later save must write it back out."""
    root = pathlib.Path(wall._MANIFEST_PATH).parent.parent
    ws = (root / "scripts" / "lib" / "make_warmstart.py").read_text(encoding="utf-8")
    assert "model.sonora_lineage = lineage_filelists(donor)" in ws
    mod = (root / "matcha" / "models" / "baselightningmodule.py").read_text(encoding="utf-8")
    assert "def on_save_checkpoint" in mod and "checkpoint[LINEAGE_KEY]" in mod
    assert "self.sonora_lineage = lineage_filelists(checkpoint)" in mod
