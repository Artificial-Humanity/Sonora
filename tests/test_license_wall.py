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

import pytest

wall = pytest.importorskip("matcha.data.license_wall")


@pytest.fixture(autouse=True)
def _clean_manifest_cache():
    """Each test sees the real manifest, freshly read."""
    wall._manifest_cache = None
    yield
    wall._manifest_cache = None


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
