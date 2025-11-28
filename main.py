import os
# ---------- Force this process to use physical GPU 1 ----------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # physical GPU 1 will appear as cuda:0

import os, glob, time, csv
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

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger

import terratorch
from terratorch.tasks import SemanticSegmentationTask

# custom encoder/decoder (registered with terratorch factory)
from encoder import AdaptedPrithvi
from decoder import PT2Decoder  # noqa

# torchmetrics
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassAccuracy,
)

# ------------- CUDA sanity prints -------------
def cuda_debug_prints():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count()  =", torch.cuda.device_count())
    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        print("torch.cuda.current_device() =", cur)
        print("torch.cuda.get_device_name(cur) =", torch.cuda.get_device_name(cur))

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
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".tif",".tiff"))])
        self.masks  = sorted([f for f in os.listdir(mask_dir)  if f.lower().endswith((".tif",".tiff"))])
        assert len(self.images) == len(self.masks), \
            f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        ip = os.path.join(self.image_dir, self.images[idx])
        mp = os.path.join(self.mask_dir,  self.masks[idx])

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
        return {"image": img, "mask": msk, "filename": os.path.basename(ip)}

# =========================
# DataModule
# =========================
class FloodDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=8, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_root, "img_dir/train")
        self.val_dir   = os.path.join(data_root, "img_dir/val")
        self.test_dir  = os.path.join(data_root, "img_dir/test")
        self.mask_train_dir = os.path.join(data_root, "ann_dir/train")
        self.mask_val_dir   = os.path.join(data_root, "ann_dir/val")
        self.mask_test_dir  = os.path.join(data_root, "ann_dir/test")
        self.train_transform = train_transform
        self.val_transform   = val_transform
        self.test_transform  = test_transform

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = FullBandFloodDataset(self.train_dir, self.mask_train_dir, self.train_transform)
            self.val_dataset   = FullBandFloodDataset(self.val_dir,   self.mask_val_dir,   self.val_transform)
        if stage in ("test", None):
            self.test_dataset  = FullBandFloodDataset(self.test_dir,  self.mask_test_dir,  self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# =========================
# Augmentations
# =========================
train_transform = A.Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    Resize(512, 512),
    ToTensorV2()
])
val_transform = A.Compose([Resize(512, 512), ToTensorV2()])
test_transform = A.Compose([Resize(512, 512), ToTensorV2()])

# =========================
# Paths & Logger
# =========================
DATA_ROOT = '/media/turtle-ssd/users/skaushik/Sen1Flood11/S2_std'
BATCH_SIZE = 4

os.makedirs("glacier_logs", exist_ok=True)
logger = TensorBoardLogger(save_dir="glacier_logs", name="glacier_fused")

ckpt_dir = "/home/skaushik/Prithvi/Prithvi-EO-2.0-main/glacier/AC_v2/UperNet_HPO_2/std"
os.makedirs(ckpt_dir, exist_ok=True)

# =========================
# Callbacks
# =========================
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename="epoch-{epoch:02d}-val_jacc-{val/Multiclass_Jaccard_Index:.4f}",
    monitor="val/Multiclass_Jaccard_Index",
    mode="max",
    save_top_k=,1
    every_n_epochs=1,
    save_on_train_epoch_end=False,
    auto_insert_metric_name=False
)
early_stopping_callback = EarlyStopping(
    monitor="val/Multiclass_Jaccard_Index",
    patience=20,
    mode="max",
    verbose=True
)

class CSVLoggerCallback(Callback):
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, "val_metrics.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "val/Multiclass_F1_Score", "val/loss"])
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        f1 = metrics.get("val/Multiclass_F1_Score", None)
        loss = metrics.get("val/loss", None)
        ep = trainer.current_epoch
        if f1 is not None and loss is not None:
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow([ep, float(f1), float(loss)])

class EpochTimeLogger(Callback):
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, "epoch_times.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "seconds"])
    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.time() - self._t0
        ep = trainer.current_epoch
        print(f"[TIME] Epoch {ep} took {dt:.2f}s")
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{dt:.4f}"])
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            try:
                trainer.logger.experiment.add_scalar("time/epoch_seconds", dt, ep)
            except Exception:
                pass

# =========================
# Custom Task that IGNORES label -1
# =========================
class IgnoreLabelSegTask(SemanticSegmentationTask):
    """
    - Ignores label == ignore_index for loss & metrics
    - Accepts focal-loss params via loss_hparams={'gamma':..., 'alpha':...}
    - Robustly extracts logits from Terratorch ModelOutput/dict/tensor
    """
    def __init__(self, *args, ignore_index: int = -1, loss_hparams: dict | None = None, **kwargs):
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

        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes_, average="macro", ignore_index=self.ignore_index)
        self.val_f1   = MulticlassF1Score(num_classes=self.num_classes_, average="macro", ignore_index=self.ignore_index)
        self.test_f1  = MulticlassF1Score(num_classes=self.num_classes_, average="macro", ignore_index=self.ignore_index)

        self.train_iou = MulticlassJaccardIndex(num_classes=self.num_classes_, ignore_index=self.ignore_index)
        self.val_iou   = MulticlassJaccardIndex(num_classes=self.num_classes_, ignore_index=self.ignore_index)
        self.test_iou  = MulticlassJaccardIndex(num_classes=self.num_classes_, ignore_index=self.ignore_index)

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes_, ignore_index=self.ignore_index)
        self.val_acc   = MulticlassAccuracy(num_classes=self.num_classes_, ignore_index=self.ignore_index)
        self.test_acc  = MulticlassAccuracy(num_classes=self.num_classes_, ignore_index=self.ignore_index)

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
            preferred_keys = ("logits", "y_hat", "pred", "preds", "out", "seg", "segmentation")
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
                preferred_keys = ("logits", "y_hat", "pred", "preds", "out", "seg", "segmentation")
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
                alpha = torch.as_tensor(alpha_cfg, device=logits_valid.device, dtype=torch.float32)
                alpha_t = alpha[y_valid.long()]
            else:
                alpha_t = torch.full_like(ce, float(alpha_cfg))
            focal = alpha_t * ((1 - pt) ** gamma) * ce
        return focal.mean()

    def _compute_loss_masked(self, logits, y):
        if logits.dtype != torch.float32:
            logits = logits.float()
        logits_valid, y_valid, valid_mask = self._masked_logits_targets(logits, y, self.ignore_index)
        if logits_valid is None:
            # No valid pixels — return zero loss
            return logits.new_tensor(0.0), valid_mask

        loss_name = str(self.hparams.get("loss", "cross_entropy")).lower()
        if loss_name == "focal":
            loss = self._focal_ce_loss(logits_valid, y_valid)
        else:
            loss = F.cross_entropy(logits_valid, y_valid.long(), reduction="mean")
        return loss, valid_mask

    # ---------- steps ----------
    def _step_impl(self, batch, stage: str):
        x = batch["image"].float()
        y = batch["mask"].long()
        bsz = x.shape[0]  # <-- get batch size
    
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
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("train/Multiclass_F1_Score", self.train_f1, on_step=False, on_epoch=True, batch_size=bsz)
            self.log("train/Multiclass_Jaccard_Index", self.train_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("train/Multiclass_Accuracy", self.train_acc, on_step=False, on_epoch=True, batch_size=bsz)
    
        elif stage == "val":
            if has_valid:
                self.val_f1.update(preds, y)
                self.val_iou.update(preds, y)
                self.val_acc.update(preds, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("val/Multiclass_F1_Score", self.val_f1, on_step=False, on_epoch=True, batch_size=bsz)
            self.log("val/Multiclass_Jaccard_Index", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("val/Multiclass_Accuracy", self.val_acc, on_step=False, on_epoch=True, batch_size=bsz)
    
        else:  # test
            if has_valid:
                self.test_f1.update(preds, y)
                self.test_iou.update(preds, y)
                self.test_acc.update(preds, y)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("test/Multiclass_F1_Score", self.test_f1, on_step=False, on_epoch=True, batch_size=bsz)
            self.log("test/Multiclass_Jaccard_Index", self.test_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
            self.log("test/Multiclass_Accuracy", self.test_acc, on_step=False, on_epoch=True, batch_size=bsz)
    
        return loss
    
    # =========================
# Data
# =========================
data_module = FloodDataModule(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

# =========================
# Model (Terratorch) with ignore -1
# =========================
model = IgnoreLabelSegTask(
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
    loss="ce",                     # or "cross_entropy"
    #loss_hparams={"gamma": 1.5},      # focal params (captured locally)
    lr=7.46e-05,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 1.69e-05},
    scheduler="StepLR",
    scheduler_hparams={"step_size": 16, "gamma": 0.909027866016032},
    model_factory="EncoderDecoderFactory",
    ignore_index=-1,
)

# =========================
# Trainer
# =========================
pl.seed_everything(42, workers=True)
cuda_debug_prints()

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=200,
    logger=logger,
    log_every_n_steps=20,
    callbacks=[checkpoint_callback, early_stopping_callback, CSVLoggerCallback(ckpt_dir), EpochTimeLogger(ckpt_dir)],
    deterministic=False,
    enable_progress_bar=True,
    num_sanity_val_steps=0,  # avoid compute-before-update warnings during sanity check
    # precision="16-mixed",
)

# =========================
# Train
# =========================
print("[RUN] trainer.fit starting ...")
#trainer.fit(model, datamodule=data_module)
print("[RUN] trainer.fit finished")

# =========================
# Test best (or all) checkpoints
# =========================
def append_test_metrics(csv_path: str, ckpt_path: str, metrics_list):
    if not metrics_list: return
    metrics = metrics_list[0]
    row = {"ckpt_path": ckpt_path}
    for k, v in metrics.items():
        try: row[k] = float(v)
        except Exception: row[k] = v
    file_exists = os.path.exists(csv_path)
    fieldnames = ["ckpt_path"] + sorted([k for k in row.keys() if k != "ckpt_path"])
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header: fieldnames = header
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow(row)

test_summary_csv = os.path.join(ckpt_dir, "test_metrics.csv")
best_ckpt = checkpoint_callback.best_model_path
if best_ckpt and os.path.exists(best_ckpt):
    print(f"[RUN] Testing best checkpoint: {best_ckpt}")
    res = trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt)
    append_test_metrics(test_summary_csv, best_ckpt, res)
else:
    ckpts = sorted(glob.glob(f"{ckpt_dir}/*.ckpt"))
    if len(ckpts) == 0:
        print("[WARN] No checkpoints found to test.")
    for p in ckpts:
        print(f"[RUN] Testing checkpoint: {p}")
        res = trainer.test(model, datamodule=data_module, ckpt_path=p)
        append_test_metrics(test_summary_csv, p, res)

print(f"[DONE] Test metrics logged to: {test_summary_csv}")


##Infereces, 

#import os
#import torch
#import rasterio
#import numpy as np
#
## ---- Load trained model from checkpoint ----
#best_ckpt_path = "/home/skaushik/Prithvi/Prithvi-EO-2.0-main/glacier/AC_v2/UperNet_HPO_2/epoch-89-val_jacc-0.9115.ckpt"
#
#model = SemanticSegmentationTask.load_from_checkpoint(
#    best_ckpt_path,
#    model_args=model.hparams.model_args,
#    model_factory=model.hparams.model_factory,
#)
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#model.eval()
#
## ---- Ensure test dataset is ready ----
#data_module.setup(stage="test")
#test_loader = data_module.test_dataloader()
#test_dataset = data_module.test_dataset
#
## ---- Where to save predictions ----
#output_dir = "/home/skaushik/Prithvi/Prithvi-EO-2.0-main/glacier/AC_v2/UperNet_HPO_2/s1f1_predictions"
#os.makedirs(output_dir, exist_ok=True)
#
#def save_prediction(pred, filename, ref_image_path):
#    # NOTE: assumes pred shape matches ref raster shape.
#    # If you used Resize(320,320) on non-320x320 chips, you'll need to handle rescaling.
#    with rasterio.open(ref_image_path) as src:
#        transform = src.transform
#        crs = src.crs
#        height, width = pred.shape
#
#        with rasterio.open(
#            os.path.join(output_dir, filename),
#            "w",
#            driver="GTiff",
#            height=height,
#            width=width,
#            count=1,
#            dtype="uint8",
#            crs=crs,
#            transform=transform,
#        ) as dst:
#            dst.write(pred.astype("uint8"), 1)
#
## ---- Run inference exactly like training/test ----
#with torch.no_grad():
#    for batch in test_loader:
#        imgs = batch["image"].to(device)       # already full stack + transforms
#        filenames = batch["filename"]
#
#        outputs = model(imgs)
#        # SemanticSegmentationTask usually returns an object with `.output`
#        logits = outputs.output if hasattr(outputs, "output") else outputs
#        preds = torch.argmax(logits, dim=1).cpu().numpy()
#
#        for pred, fname in zip(preds, filenames):
#            ref_image_path = os.path.join(test_dataset.image_dir, fname)
#            save_prediction(pred, fname, ref_image_path)
#
#print(f"? Predictions saved in: {output_dir}")
