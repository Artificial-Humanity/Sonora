"""Reading a training filelist, and naming the source a row came from.

⚠⚠ ONE COPY, ON PURPOSE. These three functions were written twice within a week — once in
`build_pitch_error_filelist.py` and once, subtly differently, in `measure_speaker_hnr.py`
— and the two copies disagreed about what `--dataset LibriTTS_R` means. The survey's copy
returned the dataset name for a non-LibriTTS row, so its "(unkeyed)" test never fired and
the restriction it advertised did not restrict anything. That is the same failure shape as
the duplicated pitch estimator: two functions that look alike, are compared against each
other by downstream code, and quietly encode different rules.

⚠ `partition` AND `dataset` ARE DIFFERENT QUESTIONS and the tools need both. `dataset_of`
answers "which corpus was this row drawn from" (`LibriTTS_R`, `emilia_kept_24k`,
`expressive_registers_24k`), which is a licensing and recording-chain question.
`partition_of` answers "which slice of that corpus" (`train-other-500` vs
`train-clean-100`), which is a recording-conditions question and the one that rides along
with any quality measurement. Filter on the first, report and match on the second.
"""

from pathlib import Path


def dataset_of(wav):
    """The component under `datasets/` — `LibriTTS_R`, `emilia_kept_24k`, ..."""
    parts = Path(wav).parts
    try:
        return parts[parts.index("datasets") + 1]
    except (ValueError, IndexError):
        return "(unkeyed)"


def partition_of(wav, depth=2):
    """`train-other-500` etc., taken below the dataset root.

    ⚠ Falls back to the DATASET NAME rather than a silent "(unkeyed)" for corpora that are
    not LibriTTS-R. v7 holds three of them — 325,094 LibriTTS-R rows, 10,653
    emilia_kept_24k and 799 expressive_registers_24k — and keying on
    `parts.index("LibriTTS_R")` alone means an Emilia speaker drawn into a bin is reported
    as "(unkeyed)" beside the partitions rather than as the recording-condition confound
    it is.
    """
    ds = dataset_of(wav)
    parts = Path(wav).parts
    if ds != "LibriTTS_R":
        return ds
    i = parts.index("LibriTTS_R")
    tail = parts[i + 1:i + 1 + depth - 1] or ("(root)",)
    return "/".join(tail)


def read_corpus(path, split, vat_dim):
    """Rows verbatim, so phonemes, speaker id and VAT are exactly what training saw.

    Returns `(line, wav, speaker_index, phoneme_count)` per row.

    ⚠ THE VAT WIDTH IS CHECKED HERE because a filelist of the wrong width parses cleanly
    and means a DIFFERENT MODEL. `vat_dim` is passed in rather than imported so this module
    stays free of the matcha package, but every caller must pass `matcha.delivery.VAT_DIM`
    and not a literal.
    """
    names = {"train": ["train_op.txt"], "val": ["val_op.txt"],
             "both": ["train_op.txt", "val_op.txt"]}[split]
    rows = []
    for name in names:
        fp = Path(path) / name
        if not fp.is_file():
            raise SystemExit("REFUSING: no %s. --corpus must point at a corpus directory "
                             "holding the filelists." % fp)
        for lineno, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 4:
                raise SystemExit("REFUSING: %s line %d states %d `|`-separated fields, "
                                 "wanted 4." % (fp, lineno, len(parts)))
            wav, spk, phon, vat = parts
            width = len(vat.split(","))
            if width != vat_dim:
                raise SystemExit(
                    "REFUSING: %s line %d carries a %d-wide conditioning vector and this "
                    "checkout states VAT_DIM as %d. A filelist of the wrong width parses "
                    "cleanly and means a DIFFERENT MODEL." % (fp, lineno, width, vat_dim))
            try:
                spk_i = int(spk)
            except ValueError:
                raise SystemExit("REFUSING: %s line %d speaker %r is not an index."
                                 % (fp, lineno, spk))
            rows.append((line, wav, spk_i, len(phon.split())))
    if not rows:
        raise SystemExit("REFUSING: 0 rows read. An empty sample reports a tidy zero "
                         "gradient and looks exactly like a measurement.")
    return rows
