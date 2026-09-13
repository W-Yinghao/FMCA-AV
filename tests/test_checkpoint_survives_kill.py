import re
import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent


class _Tiny(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def training_step(self, batch, index):
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, batch, index):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _FirstWrite(Callback):
    """Epoch at which a checkpoint first exists, observed from the NEXT epoch."""

    def __init__(self, directory: Path) -> None:
        self.directory, self.epoch = directory, None

    def on_train_epoch_start(self, trainer, module) -> None:
        if self.epoch is None and self.directory.is_dir():
            if any(self.directory.glob("*.ckpt")):
                self.epoch = int(trainer.current_epoch)


def _first_write(**checkpoint_kwargs):
    directory = Path(tempfile.mkdtemp()) / "ck"
    watch = _FirstWrite(directory)
    trainer = L.Trainer(
        max_epochs=12, accelerator="cpu", logger=False, enable_progress_bar=False,
        enable_model_summary=False, check_val_every_n_epoch=10, num_sanity_val_steps=0,
        callbacks=[ModelCheckpoint(dirpath=str(directory), **checkpoint_kwargs), watch])
    loader = DataLoader(TensorDataset(torch.randn(16, 4)), batch_size=8)
    trainer.fit(_Tiny(), loader, loader)
    return watch.epoch, sorted(p.name for p in directory.glob("*.ckpt"))


class CheckpointSurvivesKillTests(unittest.TestCase):
    def test_epoch_based_saving_writes_nothing_until_fit_returns(self) -> None:
        """The configuration that silently loses a killed run.

        Pinned so nobody restores it: on a 24h partition cap an 800-epoch
        unit never reaches a normal return, so this setting means every
        kill restarts from zero.
        """
        first, _ = _first_write(save_last=True, save_top_k=0, every_n_epochs=1)
        self.assertIsNone(first)

    def test_step_based_saving_writes_during_training(self) -> None:
        first, files = _first_write(save_last=True, save_top_k=0, every_n_train_steps=4)
        self.assertIsNotNone(first)
        self.assertLessEqual(first, 4)
        self.assertEqual(files, ["last.ckpt"])     # one file, overwritten

    def test_the_runner_uses_step_based_saving(self) -> None:
        source = (REPO / "scripts" / "run_gate1_unit.py").read_text()
        # Up to the closing paren at the call's own indentation -- a lazy
        # match stops inside str(unit_dir / "checkpoints") instead.
        block = re.search(r"checkpoint = ModelCheckpoint\(\n(.*?)\n        \)",
                          source, re.S).group(1)
        self.assertIn("every_n_train_steps", block)
        self.assertNotIn("every_n_epochs", block)
        self.assertIn("save_last=True", block)


if __name__ == "__main__":
    unittest.main()
