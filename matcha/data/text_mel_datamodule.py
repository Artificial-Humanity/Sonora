import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.data.license_wall import enforce as enforce_license_wall
from matcha.data.license_wall import refuse_holdout
from matcha.delivery import VAT_BASE_DIM, lane_of_vector
from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Batch indices so each batch holds utterances of similar length.

    Why this exists (2026-08-01): the loader ran `shuffle=True` with no length
    awareness, so every batch padded to its longest member. That was tolerable while
    MAX_SECONDS was 16 s; at 22 s a single long utterance inflates its whole batch by
    ~40%, and the cost lands on whichever batch happens to catch one. Padding waste is
    also pure compute — the padded frames are masked out of the loss.

    Standard bucketing, deliberately not a full length-sort: indices are shuffled, cut
    into megabatches of `batch_size * multiplier`, sorted WITHIN each megabatch, then
    split into batches whose order is shuffled again. So batches are internally
    homogeneous while the epoch still sees a fresh, non-length-ordered sequence — a
    global sort would make every epoch identical and feed the model all its short
    utterances first.

    Length key is the TEXT field, not the audio: it needs no I/O at setup, and for TTS
    mel length is near-proportional to phoneme count. It only has to rank, not measure.
    """

    def __init__(self, lengths, batch_size, multiplier=20, shuffle=True, seed=42):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.megabatch = batch_size * multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.megabatch):
            chunk = sorted(idx[i : i + self.megabatch], key=lambda k: self.lengths[k])
            batches += [chunk[j : j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1).shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        load_vat=False,
        vat_dim=None,
        # E-M3: train_dataloader documents this as the bucketing off-switch and reads it
        # with getattr, but it was never an __init__ param — so yaml could not set it and
        # the getattr always returned its default. Declaring it makes the documented
        # switch actually reachable. (It was declared to let a curve be bisected across
        # the bucketing boundary; since the E-M5 masking fix, bucketing no longer moves
        # the logged loss at all, so this is a throughput knob again and nothing more.)
        bucket_multiplier=20,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        # The license wall (north star §8.2): every dataset in the filelists
        # must be declared permissive in configs/data_licenses.yaml.
        enforce_license_wall(
            [self.hparams.train_filelist_path, self.hparams.valid_filelist_path]
        )
        refuse_holdout(
            [self.hparams.train_filelist_path, self.hparams.valid_filelist_path]
        )

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            load_vat=self.hparams.load_vat,
            vat_dim=self.hparams.vat_dim,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            load_vat=self.hparams.load_vat,
            vat_dim=self.hparams.vat_dim,
        )

    def _loader_mp_kwargs(self):
        # Workers must be spawned, not forked: forking the GPU-initialized
        # trainer (dozens of HIP/MIOpen threads, KFD SVM ranges) wedges for
        # minutes per fork on gfx1151 (amdgpu restore-workqueue churn,
        # observed 2026-07-20). Spawned workers never touch the parent's GPU
        # address space; persistent workers pay the spawn cost once per fit
        # instead of every epoch.
        if self.hparams.num_workers == 0:
            return {}
        return {"multiprocessing_context": "spawn", "persistent_workers": True}

    def train_dataloader(self):
        # bucket_multiplier=0 restores the old length-blind shuffle, so this can be
        # turned off from config without editing code if it ever needs bisecting.
        mult = getattr(self.hparams, "bucket_multiplier", 20)
        if not mult:
            return DataLoader(
                dataset=self.trainset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                collate_fn=TextMelBatchCollate(self.hparams.n_spks),
                **self._loader_mp_kwargs(),
            )
        lengths = [len(r[2] if self.hparams.n_spks > 1 else r[1])
                   for r in self.trainset.filepaths_and_text]
        return DataLoader(
            dataset=self.trainset,
            batch_sampler=LengthBucketBatchSampler(
                lengths, self.hparams.batch_size, multiplier=mult, seed=self.hparams.seed),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            **self._loader_mp_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            **self._loader_mp_kwargs(),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        load_vat=False,
        vat_dim=None,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.filelist_path = filelist_path
        self.vat_dim = vat_dim
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.load_vat = load_vat

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        # VAT filelists append a final `v,a,t` float field:
        #   path|text|v,a,t  or  path|spk|text|v,a,t
        vat = None
        if self.load_vat:
            vat = torch.tensor([float(v) for v in filepath_and_text[-1].split(",")], dtype=torch.float32)
            # THE FILELIST WIDTH SEAM. Nothing here ever counted the fields, so a
            # filelist whose width disagrees with the model's vat_dim loads happily and
            # fails much later, deep in the trunk's Conv1d. Under the delivery migration
            # (vat_dim 3 -> 4) both mistakes are one edit away: pointing a 4-channel run
            # at a v3c filelist, or a 3-channel run at a delivery-bearing one. Check it
            # where the number is first read, and name the file.
            if self.vat_dim is not None and vat.numel() != self.vat_dim:
                raise ValueError(
                    f"{self.filelist_path}: VAT field has {vat.numel()} value(s) "
                    f"({filepath_and_text[-1]!r}) but the model expects "
                    f"vat_dim={self.vat_dim}. Filelist and model config disagree — see "
                    "scripts/gates/test_vat_dim_seams.py, the gate for exactly these seams."
                )
            # ...and the WIDTH is only half of what "a valid conditioning vector" means
            # (TR-L3). The count was checked and the delivery block's shape was not, on
            # the one path that trains on it: `lane_of_vector`'s one-hot refusal is called
            # only by readers (CLI, manifests), never by the loader. A future writer
            # emitting `...,0.5,0.5,0,0,0` would train silently as an interpolated
            # category — the meaning-error `matcha/delivery.py` says has no meaning —
            # rather than raising. Derive and merge both write `int(x)` and the shipped v5
            # was verified 0/1 on disk, so this is a seam guard, not a live fix.
            if self.vat_dim is not None and vat.numel() > VAT_BASE_DIM:
                lane_of_vector(vat.tolist())     # raises unless the block is one-hot
            filepath_and_text = filepath_and_text[:-1]
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": cleaned_text,
            "durations": durations,
            "vat": vat,
        }

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        import soundfile as sf
        audio_np, sr = sf.read(filepath, dtype='float32')
        if audio_np.ndim == 1:
            audio = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            audio = torch.from_numpy(audio_np).t()
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        vats = []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            vats.append(item.get("vat"))
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None
        # THE SILENT SEAM — the dangerous one of the three.
        #
        # This was all-or-none: one item missing a VAT dropped conditioning for the
        # ENTIRE batch, silently, because `vat=None` is a legitimate value everywhere
        # downstream (it means "neutral", and the encoder/decoder helpfully substitute
        # zeros). So a partially-labelled filelist does not crash and does not warn; it
        # trains a conditioned model on unconditioned batches and the only symptom is a
        # channel that never learns. That is precisely the failure mode the delivery
        # channel is most likely to hit, since delivery is blank for 17 clips by design
        # and unknown ≡ zero is a real, intended value.
        #
        # Mixed presence is never legitimate. Make it loud.
        present = sum(v is not None for v in vats)
        if present and present != len(vats):
            missing = [fp for fp, v in zip(filepaths, vats) if v is None]
            raise ValueError(
                f"VAT present for {present}/{len(vats)} items in this batch. Mixed "
                "presence would silently drop conditioning for the whole batch. "
                f"First without: {missing[0]} ({len(missing)} total). A clip with no "
                "label must carry an explicit neutral row, not a missing field."
            )
        vat = torch.stack(vats) if present else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
            "vat": vat,
        }
