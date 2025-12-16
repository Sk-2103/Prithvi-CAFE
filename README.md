
# Prithvi-Complimentary Adaptive Fusion Encoder (CAFE)
### Unlocking the full potential of multi-band satellite imagery for flood inundation mapping

The **Prithvi-CAFE** framework introduces a powerful *adaptive hybrid encoder* that fuses **Transformer-based global reasoning (Prithvi-EO-2.0)** with **CNN-based local spatial sensitivity**, enabling high-resolution, reliable flood inundation mapping across multi-channel/sensor inputs.

Prithvi-CAFE integrates:

- 🌍 **Prithvi-EO-2.0 (600M) backbone with lightweight Adapters**  
- 🔁 **Multi-scale multi-stage fusion of ViT + CNN **  
- 🧠 **Terratorch-compatible custom UPerNet decoders**  
- 📡 **Support for any number of input channels (Sentinel-1/2, PlanetScope, DEM, etc.)**  
- ⚡ **End-to-end PyTorch Lightning training + testing pipeline**

## Model Architecture

![Block Diagram](block_png.jpeg)

# 📦 Installation

```bash
git clone https://github.com/Sk-2103/Prithvi-CAFE.git
cd <path>

pip install -r requirements.txt
```

### Required libraries
- terratorch  
- pytorch-lightning  
- torchmetrics  
- rasterio  
- albumentations  

---

# 📂 Dataset Structure

```
dataset_root/
│
├── img_dir/
│     ├── train/
│     ├── val/
│     └── test/
│
└── ann_dir/
      ├── train/
      ├── val/
      └── test/
```

- Images: multi-band satellite stacks (TIF)  
- Masks:  
  - 0 = background  
  - 1 = flood  
  - -1 = ignore (not used in loss/metrics)

---

# 🏋️ Training

```bash
python main.py
```
# 📊 Download model weight (Sen1Flood11)
Model weight and test data link: [Download Model Weights (Google Drive)](https://drive.google.com/file/d/1QNefJQrlxXVwcLManl4bIL8Lb1Dontpu/view?usp=drive_link)


#  Testing

python testing.py

What it does:
- Downloads Prithvi-CAFE.tar.gz from Google Drive (if missing)
- Extracts into ./Prithvi-CAFE (if missing)
- Auto-detects:
    * checkpoint (.ckpt)
    * dataset root containing img_dir/test and ann_dir/test
- Prints the full test metrics dict (same keys as your training code)
"""

---
### 📊 Full Test Metrics (Sen1Flood11)

We provide access to trained weights and the Sen1Flood11 test data, enabling fully automated testing of the model and reproduction of the reported results.
The same model can be directly tested on similar flood-mapping datasets with only minor path/config modifications.

The model was evaluated on the Sen1Flood11 test split using the Lightning test loop, yielding the following metrics:

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%|█████████████████████████████████████████████████| 23/23 [00:26<00:00,  0.88it/s]

#### **Global Metrics**

| Metric                           | Value     |
|---------------------------------|-----------|
| Multiclass Accuracy             | **0.9778** |
| Multiclass F1 Score             | **0.9778** |
| Multiclass Jaccard Index (mIoU) | **0.9056** |
| Micro Jaccard Index             | **0.9566** |
| Test Loss                       | **0.0815** |

#### **Per-Class Metrics**

| Metric                 | Background (0) | Flood (1) |
|------------------------|----------------|-----------|
| Accuracy               | 0.9903         | 0.8910    |
| IoU (Jaccard Index)    | 0.9771         | 0.8341    |


### 🌎 Full Test Metrics — Geographically Held-Out Test Site (Bolivia)

This evaluation measures how well **Prithvi-CAFE** generalizes to a completely unseen geographic region (Bolivia).  
The model was tested on a geographically distinct flood event not used during training.

#### **Global Metrics**

| Metric                           | Value     |
|---------------------------------|-----------|
| Multiclass Accuracy             | **0.9687** |
| Multiclass F1 Score             | **0.9687** |
| Multiclass Jaccard Index (mIoU) | **0.8888** |
| Micro Jaccard Index             | **0.9394** |
| Test Loss                       | **0.0807** |

#### **Per-Class Metrics**

| Metric                 | Background (0) | Flood (1) |
|------------------------|----------------|-----------|
| Accuracy               | 0.9891         | 0.8608    |
| IoU (Jaccard Index)    | 0.9638         | 0.8137    |





# 🔍 Inference Example

```python
best_ckpt_path = ".../epoch-89-val_jacc-0.9115.ckpt"

model = SemanticSegmentationTask.load_from_checkpoint(
    best_ckpt_path,
    model_args=model.hparams.model_args,
    model_factory=model.hparams.model_factory,
)

preds = torch.argmax(logits, dim=1)
```

---

# 🧠 Conceptual Overview

### Prithvi-CAFE = Prithvi Transformer + CNN + Adaptive Fusion

- Prithvi-EO-2.0 extracts global contextual features  
- Residual CNN + CAM captures spatial/local texture cues  
- M2FAF aligns and fuses multi-scale features  
- Decoder reconstructs dense segmentation at full resolution  

---


















