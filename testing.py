#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prithvi-CAFE Terratorch-style testing script

What it does:
- Downloads Prithvi-CAFE.tar.gz from Google Drive (if missing)
- Extracts into ./Prithvi-CAFE (if missing)
- Auto-detects:
    * checkpoint (.ckpt)
    * dataset root containing img_dir/test and ann_dir/test
- Builds IgnoreLabelSegTask (same structure as training, with ignore_index=-1)
- Uses Lightning Trainer + trainer.test(...) to run Terratorch metrics
- Prints the full test metrics dict (same keys as your training code)
"""

import os
# ---------- Force this process to use physical GPU 1 ----------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # physical GPU 1 will appear as cuda:0

import tarfile
import time
import csv
import glob

import numpy as np
import torch
import torch.nn.functional as F
import rasterio

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, Resize
)
from albumentations.pytorch import ToTensorV2

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassAccuracy,
)

from terratorch.tasks import SemanticSegmentationTask

# custom encoder/decoder (registered with terratorch factory)
from encoder import AdaptedPrithvi
from decoder import PT2Decoder  # noqa

# ------------- GDrive archive config -------------
GDRIVE_FILE_ID = "1QNefJQrlxXVwcLManl4bIL8Lb1Dontpu"
ARCHIVE_NAME = "Prithvi-CAFE.tar.gz"
EXTRACTED_ROOT = "Prithvi-CAFE"  # top-level folder inside the tar.gz

IGNORE_INDEX = -1
BATCH_SIZE = 4
NUM_WORKERS = 4

# ------------- CUDA sanity prints -------------
def cuda_debug_prints():
    print("CUDA_VISIBLE_DEVICES   =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count()  =", torch.cuda.device_count())
    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        print("torch.cuda.current_device() =", cur)
        print("torch.cuda.get_device_name(cur) =", torch.cuda.get_device_name(cur))
    print("-" * 60)


# =========================
# Download + extract utils
# =========================
def download_archive_if_needed():
    """Download Prithvi-CAFE.tar.gz from Google Drive if not present."""
    if os.path.exists(ARCHIVE_NAME):
        print(f"[INFO] Archive already exists: {ARCHIVE_NAME}")
        return

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download the demo data.\n"
            "Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print("[INFO] Downloading tar.gz from:")
    print(f"  {url}")
    gdown.download(url, ARCHIVE_NAME, quiet=False)
    print(f"[INFO] Downloaded to {ARCHIVE_NAME}")


def extract_archive_if_needed():
    """Extract Prithvi-CAFE.tar.gz into ./Prithvi-CAFE if not already present."""
    if os.path.isdir(EXTRACTED_ROOT):
        print(f"[INFO] Extracted folder already exists: {EXTRACTED_ROOT}")
        return

    if not os.path.exists(ARCHIVE_NAME):
        raise FileNotFoundError(
            f"[ERROR] {ARCHIVE_NAME} not found.\n"
            f"Expected archive in: {os.path.abspath(ARCHIVE_NAME)}"
        )

    print(f"[INFO] Extracting {ARCHIVE_NAME} into {EXTRACTED_ROOT} with Python tarfile...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tf:
        tf.extractall(".")
    print("[INFO] Extraction complete.")


def find_checkpoint(root_dir: str) -> str:
    """Find first .ckpt under root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".ckpt"):
                ckpt_path = os.path.join(dirpath, f)
                print(f"[INFO] Found checkpoint: {ckpt_path}")
                return ckpt_path
    raise FileNotFoundError(f"No .ckpt file found under {root_dir}")


def find_data_root(root_dir: str) -> str:
    """
    Find directory that contains both img_dir and ann_dir,
    and under them a test/ subfolder.
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if "img_dir" in dirnames and "ann_dir" in dirnames:
            img_dir = os.path.join(dirpath, "img_dir")
            ann_dir = os.path.join(dirpath, "ann_dir")
            if (
                os.path.isdir(os.path.join(img_dir, "test"))
                and os.path.isdir(os.path.join(ann_dir, "test"))
            ):
                print(f"[INFO] Found dataset root: {dirpath}")
                return dirpath
    raise FileNotFoundError(
        f"Could not find dataset root with img_dir/test and ann_dir/test under {root_dir}"
    )


# =========================
# Dataset (preserve -1)
# =========================
class FullBandFloodDataset(Dataset):
    """
    Reads full multi-band image (.tif) and mask (.tif).
    Preserves -1 ignore pixels. Non-negative labels are binarized: >0 -> 1, 0 -> 0.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))]
        )
        self.masks = sorted(
            [f for f in os.listdir(mask_dir) if f.lower().endswith((".tif", ".tiff"))]
        )
        assert len(self.images) == len(
            self.masks
        ), f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ip = os.path.join(self.image_dir, self.images[idx])
        mp = os.path.join(self.mask_dir, self.masks[idx])

        with rasterio.open(ip) as src:
            img = src.read()  # (C,H,W)
        with rasterio.open(mp) as src:
            msk_raw = src.read(1)  # (H,W)

        # preserve -1; binarize only non-negative labels
        msk = np.where(msk_raw < 0, -1, (msk_raw > 0).astype(np.uint8))

        # Albumentations expects HWC
        img = np.moveaxis(img, 0, -1)  # (H,W,C)

        if self.transform:
            out = self.transform(image=img, mask=msk)
            img = out["image"]   # tensor [C,H,W]
            msk = out["mask"]    # tensor [H,W]

        msk = msk.to(dtype=torch.int64)
        return {
            "image": img,
            "mask": msk,
            "filename": os.path.basename(ip),
        }


# =========================
# Test-only DataModule
# =========================
class FloodTestDataModule(pl.LightningDataModule):
    def __init__(self, img_test_dir, mask_test_dir, batch_size, test_transform):
        super().__init__()
        self.img_test_dir = img_test_dir
        self.mask_test_dir = mask_test_dir
        self.batch_size = batch_size
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage in ("test", None):
            self.test_dataset = FullBandFloodDataset(
                self.img_test_dir, self.mask_test_dir, self.test_transform
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )


# =========================
# Augmentations
# =========================
test_transform = A.Compose([Resize(512, 512), ToTensorV2()])


# =========================
# Custom Task that IGNORES label -1
# (same logic as your main training script)
# =========================
class IgnoreLabelSegTask(SemanticSegmentationTask):
    """
    - Ignores label == ignore_index for loss & metrics
    - Accepts focal-loss params via loss_hparams={'gamma':..., 'alpha':...}
    - Robustly extracts logits from Terratorch ModelOutput/dict/tensor
    """

    def __init__(
        self,
        *args,
        ignore_index: int = -1,
        loss_hparams: dict | None = None,
        **kwargs,
    ):
        # capture and remove unsupported kwarg before super()
        self._loss_hparams = loss_hparams or {}
        self._focal_gamma = float(self._loss_hparams.get("gamma", 2.0))
        self._focal_alpha = self._loss_hparams.get("alpha", None)
        if "loss_hparams" in kwargs:
            kwargs.pop("loss_hparams")

        super().__init__(*args, **kwargs)

        self.ignore_index = int(ignore_index)

        # discover #classes
        try:
            self.num_classes_ = int(kwargs.get("model_args", {}).get("num_classes", 2))
        except Exception:
            self.num_classes_ = 2

        self.train_f1 = MulticlassF1Score(
            num_classes=self.num_classes_,
            average="macro",
            ignore_index=self.ignore_index,
        )
        self.val_f1 = MulticlassF1Score(
            num_classes=self.num_classes_,
            average="macro",
            ignore_index=self.ignore_index,
        )
        self.test_f1 = MulticlassF1Score(
            num_classes=self.num_classes_,
            average="macro",
            ignore_index=self.ignore_index,
        )

        self.train_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )

        self.train_acc = MulticlassAccuracy(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )
        self.test_acc = MulticlassAccuracy(
            num_classes=self.num_classes_, ignore_index=self.ignore_index
        )

    # ---------- helpers ----------
    @staticmethod
    def _extract_logits(model_out):
        """
        Accepts: Tensor, Mapping-like (incl. Terratorch ModelOutput), object with .logits,
                 tuple/list. Returns a 4D tensor [B, C, H, W].
        """
        import collections.abc as cabc

        # 0) bare tensor
        if isinstance(model_out, torch.Tensor):
            return model_out

        def _first(t):
            if isinstance(t, (list, tuple)):
                return t[0] if len(t) else t
            return t

        # 1) mapping-like
        if isinstance(model_out, dict) or isinstance(model_out, cabc.Mapping) or (
            hasattr(model_out, "keys") and hasattr(model_out, "__contains__")
        ):
            preferred_keys = (
                "logits",
                "y_hat",
                "pred",
                "preds",
                "out",
                "seg",
                "segmentation",
            )
            for k in preferred_keys:
                if k in model_out:
                    t = _first(model_out[k])
                    if isinstance(t, torch.Tensor):
                        return t
            vals = model_out.values() if hasattr(model_out, "values") else []
            for v in vals:
                v = _first(v)
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    return v

        # 2) attribute `.logits`
        if hasattr(model_out, "logits"):
            t = _first(getattr(model_out, "logits"))
            if isinstance(t, torch.Tensor):
                return t

        # 3) to_dict()
        if hasattr(model_out, "to_dict") and callable(getattr(model_out, "to_dict")):
            d = model_out.to_dict()
            if isinstance(d, dict):
                preferred_keys = (
                    "logits",
                    "y_hat",
                    "pred",
                    "preds",
                    "out",
                    "seg",
                    "segmentation",
                )
                for k in preferred_keys:
                    if k in d and isinstance(d[k], torch.Tensor):
                        return d[k]
                for v in d.values():
                    v = _first(v)
                    if isinstance(v, torch.Tensor) and v.dim() == 4:
                        return v

        # 4) tuple/list -> first element
        if isinstance(model_out, (list, tuple)) and len(model_out) > 0:
            t = _first(model_out[0])
            if isinstance(t, torch.Tensor):
                return t

        # 5) scan attributes for a likely 4D tensor
        preferred_hits, tensor_attrs = [], []
        for name in dir(model_out):
            if name.startswith("_"):
                continue
            try:
                v = getattr(model_out, name)
            except Exception:
                continue
            v = _first(v)
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                tensor_attrs.append((name, v))
                if any(tag in name.lower() for tag in ("logit", "y_hat", "pred", "seg", "out")):
                    preferred_hits.append((name, v))
        if preferred_hits:
            return preferred_hits[0][1]
        if tensor_attrs:
            return tensor_attrs[0][1]

        raise TypeError(f"Could not extract logits from output of type {type(model_out)}")

    @staticmethod
    def _masked_logits_targets(logits, y, ignore_index):
        """
        Select valid pixels (y != ignore_index) and flatten.
        logits: [B,C,H,W], y: [B,H,W] -> returns logits_valid [N,C], y_valid [N], valid_mask [B,H,W]
        """
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got shape {tuple(logits.shape)}")
        B, C, H, W = logits.shape
        y = y.view(B, H, W)
        valid = (y != ignore_index)
        if not torch.any(valid):
            return None, None, valid

        logits_perm = logits.permute(0, 2, 3, 1)  # [B,H,W,C]
        logits_flat = logits_perm.reshape(-1, C)  # [B*H*W, C]
        y_flat = y.reshape(-1)                    # [B*H*W]
        valid_flat = valid.reshape(-1)

        return logits_flat[valid_flat], y_flat[valid_flat], valid

    def _focal_ce_loss(self, logits_valid, y_valid):
        ce = F.cross_entropy(logits_valid, y_valid.long(), reduction="none")
        pt = torch.exp(-ce)
        gamma = self._focal_gamma
        alpha_cfg = self._focal_alpha
        if alpha_cfg is None:
            focal = ((1 - pt) ** gamma) * ce
        else:
            if isinstance(alpha_cfg, (list, tuple)):
                alpha = torch.as_tensor(
                    alpha_cfg, device=logits_valid.device, dtype=torch.float32
                )
                alpha_t = alpha[y_valid.long()]
            else:
                alpha_t = torch.full_like(ce, float(alpha_cfg))
            focal = alpha_t * ((1 - pt) ** gamma) * ce
        return focal.mean()

    def _compute_loss_masked(self, logits, y):
        if logits.dtype != torch.float32:
            logits = logits.float()
        logits_valid, y_valid, valid_mask = self._masked_logits_targets(
            logits, y, self.ignore_index
        )
        if logits_valid is None:
            # No valid pixels — return zero loss
            return logits.new_tensor(0.0), valid_mask

        loss_name = str(self.hparams.get("loss", "cross_entropy")).lower()
        if loss_name == "focal":
            loss = self._focal_ce_loss(logits_valid, y_valid)
        else:
            loss = F.cross_entropy(logits_valid, y_valid.long(), reduction="mean")
        return loss, valid_mask

    # ---------- core step ----------
    def _step_impl(self, batch, stage: str):
        x = batch["image"].float()
        y = batch["mask"].long()
        bsz = x.shape[0]  # <-- batch size

        model_out = self.forward(x)
        logits = self._extract_logits(model_out)
        loss, valid_mask = self._compute_loss_masked(logits, y)
        preds = torch.argmax(logits, dim=1)
        has_valid = bool(torch.any(valid_mask))

        if stage == "train":
            if has_valid:
                self.train_f1.update(preds, y)
                self.train_iou.update(preds, y)
                self.train_acc.update(preds, y)
            self.log("train/loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            self.log("train/Multiclass_F1_Score", self.train_f1,
                     on_step=False, on_epoch=True, batch_size=bsz)
            self.log("train/Multiclass_Jaccard_Index", self.train_iou,
                     on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("train/Multiclass_Accuracy", self.train_acc,
                     on_step=False, on_epoch=True, batch_size=bsz)

        elif stage == "val":
            if has_valid:
                self.val_f1.update(preds, y)
                self.val_iou.update(preds, y)
                self.val_acc.update(preds, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            self.log("val/Multiclass_F1_Score", self.val_f1,
                     on_step=False, on_epoch=True, batch_size=bsz)
            self.log("val/Multiclass_Jaccard_Index", self.val_iou,
                     on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("val/Multiclass_Accuracy", self.val_acc,
                     on_step=False, on_epoch=True, batch_size=bsz)

        else:  # test
            if has_valid:
                self.test_f1.update(preds, y)
                self.test_iou.update(preds, y)
                self.test_acc.update(preds, y)
            self.log("test/loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            self.log("test/Multiclass_F1_Score", self.test_f1,
                     on_step=False, on_epoch=True, batch_size=bsz)
            self.log("test/Multiclass_Jaccard_Index", self.test_iou,
                     on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("test/Multiclass_Accuracy", self.test_acc,
                     on_step=False, on_epoch=True, batch_size=bsz)

        return loss


# =========================
# Build + load model
# =========================
def build_model():
    return IgnoreLabelSegTask(
        model_args={
            # ------ encoder / backbone ------
            "backbone": "AdaptedPrithvi",
            "backbone_in_channels": 6,
            "rescale": True,
            "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"],
            "backbone_num_frames": 1,
            # ------ decoder ------
            "decoder": "UPerDecoder",
            # ------ head ------
            "head_channel_list": [64],
            "head_dropout": 0.1,
            "num_classes": 2,
        },
        plot_on_val=False,
        loss="ce",
        lr=7.46e-05,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 1.69e-05},
        scheduler="StepLR",
        scheduler_hparams={"step_size": 16, "gamma": 0.909027866016032},
        model_factory="EncoderDecoderFactory",
        ignore_index=IGNORE_INDEX,
    )


def load_checkpoint_into_model(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    return model


# =========================
# MAIN
# =========================
def main():
    pl.seed_everything(42, workers=True)
    cuda_debug_prints()

    # 1) Download + extract
    download_archive_if_needed()
    extract_archive_if_needed()

    # 2) Locate checkpoint + data root
    ckpt_path = find_checkpoint(EXTRACTED_ROOT)
    data_root = find_data_root(EXTRACTED_ROOT)

    img_test_dir = os.path.join(data_root, "img_dir", "test")
    ann_test_dir = os.path.join(data_root, "ann_dir", "test")

    # 3) DataModule (test only)
    test_dm = FloodTestDataModule(
        img_test_dir=img_test_dir,
        mask_test_dir=ann_test_dir,
        batch_size=BATCH_SIZE,
        test_transform=test_transform,
    )
    test_dm.setup(stage="test")
    print(f"[INFO] Test samples: {len(test_dm.test_dataset)}")

    # 4) Model + weights
    model = build_model()
    model = load_checkpoint_into_model(model, ckpt_path)

    # 5) Trainer (test only)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1,
        logger=False,  # you can plug a logger if you want TB logs
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    print("[INFO] Running Terratorch test...")
    test_results = trainer.test(model=model, datamodule=test_dm)

    # 6) Print full metrics dict (comparable to your main script)
    print("\n======================")
    print("  TERRATORCH TEST METRICS")
    print("======================")
    if test_results:
        res = test_results[0]
        for k in sorted(res.keys()):
            print(f"{k}: {res[k]}")
    else:
        print("[WARN] trainer.test returned no results")


if __name__ == "__main__":
    main()
