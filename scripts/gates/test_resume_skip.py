"""Gates for resuming mid-epoch without replaying the epoch (matcha/data/text_mel_datamodule.py).

IN-CONTAINER (needs torch + lightning, CPU only):
    scripts/stages/run_in_rocm.sh scripts/gates/test_resume_skip.py

WHY IT EXISTS (2026-09-27). vat7_dit_spike resumed from its step-3,505 checkpoint and trained
the epoch's FIRST batches again (frames 1,092 then 484, exactly the fresh run's steps 1 and 2).
LengthBucketBatchSampler keeps no position, so the resumed iterator starts at batch 0, while
Lightning restores its own count (batch_progress.current.ready = 3,506) and ends the epoch
after len - 3,506 more batches. The resumed epoch therefore trained batches 0..7,011 -- the
first 3,506 twice -- and never reached 7,012..10,517. The fix skips exactly the batches
Lightning counts as done.

  1. skip_once(k): the next pass yields the full order from k, the pass after it the full order;
     len() is unchanged (Lightning ends the epoch on len, not on what the sampler yields);
  2. resume_skip_batches reads Lightning's restored count, and yields 0 with no trainer, a fresh
     run, or a count at the epoch's end;
  2b. no skip unless Lightning is RESTARTING: a dataloader reload after a short
     (limit_train_batches) epoch still holds the previous epoch's count (review);
  3. END TO END in real Lightning: save at a mid-epoch step checkpoint the way the launcher does
     (ModelCheckpoint every_n_train_steps), resume, and the resumed epoch trains exactly the
     batches not yet trained, in order, each once -- WITH THE PRODUCTION LOADER (worker
     processes, spawn, persistent, the kwargs taken from TextMelDataModule itself) and with
     num_workers=0. The first version tested only num_workers=0, where torch iterates the batch
     sampler once; with workers it iterates it TWICE while building the loader iterator and
     throws the first away, so a skip taken eagerly was lost in production and the gate passed
     anyway (review);
  3a. CONTROL: the same resume without the skip replays the epoch's start -- the defect,
     reproduced, so gate 3 cannot pass vacuously.
  4. EACH EPOCH ITS OWN ORDER, in real Lightning with both loaders. Lightning calls `set_epoch`
     only on `dataloader.sampler` and `batch_sampler.sampler`, and this sampler IS the batch
     sampler, so until 2026-09-27 its epoch never left 0 and every epoch of every bucketed run
     repeated one order;
  4a. CONTROL: with the hook removed, epoch 1 repeats epoch 0 -- the defect, reproduced;
  5. a resume in the middle of epoch 1 trains epoch 1's untrained remainder: the resumed
     iterator is built before Lightning's own set_epoch, so the epoch is set at construction.
"""
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _SONORA_REPO not in _sys.path:
    _sys.path.insert(0, _SONORA_REPO)

import os
import random
import tempfile
import types
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from matcha.data.text_mel_datamodule import (LengthBucketBatchSampler, TextMelDataModule,
                                             make_bucket_sampler, resume_skip_batches)

warnings.filterwarnings("ignore")
_FAILURES = []


def check(name, ok, detail=""):
    if not ok:
        _FAILURES.append(name)
    print(f"{name}: {detail}  {'PASS' if ok else 'FAIL'}")
    return ok


N, BS, MULT, SEED = 200, 4, 5, 1234
LENGTHS = [random.Random(7 + i).randint(5, 400) for i in range(N)]


def fake(ready, restarting=True):
    bp = types.SimpleNamespace(current=types.SimpleNamespace(ready=ready))
    return types.SimpleNamespace(fit_loop=types.SimpleNamespace(
        restarting=restarting, epoch_loop=types.SimpleNamespace(batch_progress=bp)))


def collate(b):
    return torch.tensor(b)


class _Idx(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return i


class _DM(L.LightningDataModule):
    def __init__(self, resume_aware, num_workers, epoch_hook=True):
        super().__init__()
        self.resume_aware = resume_aware
        self.num_workers = num_workers
        self.epoch_hook = epoch_hook

    def train_dataloader(self):
        sampler = make_bucket_sampler(LENGTHS, BS, MULT, SEED,
                                      trainer=self.trainer if self.resume_aware else None)
        if not self.epoch_hook:
            del sampler.sampler  # the pre-fix shape: nothing for Lightning to call
        # The production loader's multiprocessing kwargs, from the production code.
        mp = TextMelDataModule._loader_mp_kwargs(
            types.SimpleNamespace(hparams=types.SimpleNamespace(num_workers=self.num_workers)))
        return torch.utils.data.DataLoader(_Idx(), batch_sampler=sampler, collate_fn=collate,
                                           num_workers=self.num_workers, **mp)


class _M(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.seen = []
        self.by_epoch = {}

    def training_step(self, batch, _):
        self.by_epoch.setdefault(self.current_epoch, []).append(batch.tolist())
        if self.current_epoch == 0:
            self.seen.append(batch.tolist())
        return (self.w * batch.float().mean()).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-6)


def run(resume_aware, num_workers):
    d = tempfile.mkdtemp(prefix="resume_skip_")
    kw = dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
              enable_model_summary=False, default_root_dir=d)
    ck = ModelCheckpoint(dirpath=d, every_n_train_steps=20, save_top_k=-1,
                         filename="checkpoint_{epoch:03d}_{step:07d}")
    m1 = _M()
    L.Trainer(max_steps=25, callbacks=[ck], **kw).fit(m1, datamodule=_DM(resume_aware, num_workers))
    path = next(os.path.join(d, f) for f in sorted(os.listdir(d)) if "step=0000020" in f)
    m2 = _M()
    L.Trainer(max_steps=50, **kw).fit(m2, datamodule=_DM(resume_aware, num_workers), ckpt_path=path)
    return m2.seen


def order(epoch):
    s = LengthBucketBatchSampler(LENGTHS, BS, multiplier=MULT, seed=SEED)
    s.set_epoch(epoch)
    return list(s)


def run_epochs(num_workers, epoch_hook):
    d = tempfile.mkdtemp(prefix="sampler_epoch_")
    m = _M()
    L.Trainer(max_epochs=2, accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
              enable_model_summary=False, enable_checkpointing=False, default_root_dir=d
              ).fit(m, datamodule=_DM(True, num_workers, epoch_hook))
    return m.by_epoch


def run_resume_in_epoch_1(num_workers):
    d = tempfile.mkdtemp(prefix="sampler_epoch_resume_")
    kw = dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
              enable_model_summary=False, default_root_dir=d)
    ck = ModelCheckpoint(dirpath=d, every_n_train_steps=60, save_top_k=-1,
                         filename="checkpoint_{epoch:03d}_{step:07d}")
    L.Trainer(max_steps=65, callbacks=[ck], **kw).fit(_M(), datamodule=_DM(True, num_workers))
    path = next(os.path.join(d, f) for f in sorted(os.listdir(d)) if "step=0000060" in f)
    m = _M()
    L.Trainer(max_steps=100, **kw).fit(m, datamodule=_DM(True, num_workers), ckpt_path=path)
    return m.by_epoch.get(1, [])


def main():
    # 1. skip_once
    s = LengthBucketBatchSampler(LENGTHS, BS, multiplier=MULT, seed=SEED)
    full = list(s)
    s.skip_once(20)
    first, second = list(s), list(s)
    check("1 skip_once skips once, len unchanged",
          first == full[20:] and second == full and len(s) == len(full) == 50,
          f"first={len(first)} second={len(second)} len={len(s)}")

    # 2. resume_skip_batches
    got = [resume_skip_batches(None, 50), resume_skip_batches(fake(0), 50),
           resume_skip_batches(fake(21), 50), resume_skip_batches(fake(50), 50),
           resume_skip_batches(types.SimpleNamespace(), 50)]
    check("2 resume_skip_batches reads the restored count", got == [0, 0, 21, 0, 0], f"got={got}")
    got = resume_skip_batches(fake(21, restarting=False), 50)
    check("2b no skip unless Lightning is restarting", got == 0, f"got={got}")

    # 3. end to end, with the production loader and without workers
    for nw in (2, 0):
        after_fix = run(resume_aware=True, num_workers=nw)
        check(f"3 [num_workers={nw}] resumed epoch trains exactly the untrained batches",
              after_fix == full[20:], f"resumed batches={len(after_fix)} first={after_fix[:1]} "
              f"expected first={full[20:21]}")
        after_ctrl = run(resume_aware=False, num_workers=nw)
        check(f"3a [num_workers={nw}] control: without the skip, the resume replays the start",
              after_ctrl == full[:30], f"resumed batches={len(after_ctrl)} first={after_ctrl[:1]}")

    # 4. each epoch its own order; 4a control; 5 resume inside epoch 1
    o0, o1 = order(0), order(1)
    for nw in (2, 0):
        got = run_epochs(nw, epoch_hook=True)
        check(f"4 [num_workers={nw}] each epoch trains its own order",
              got.get(0) == o0 and got.get(1) == o1 and o0 != o1,
              f"epoch0 ok={got.get(0) == o0} epoch1 ok={got.get(1) == o1}")
        got = run_epochs(nw, epoch_hook=False)
        check(f"4a [num_workers={nw}] control: without the hook, epoch 1 repeats epoch 0",
              got.get(1) == o0, f"epoch1==epoch0 order: {got.get(1) == o0}")
        got = run_resume_in_epoch_1(nw)
        check(f"5 [num_workers={nw}] a resume inside epoch 1 trains its untrained remainder",
              got == o1[10:], f"resumed batches={len(got)} first={got[:1]} expected={o1[10:11]}")


if __name__ == "__main__":
    main()
    print()
    if _FAILURES:
        print(f"FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all resume-skip gates PASS")
