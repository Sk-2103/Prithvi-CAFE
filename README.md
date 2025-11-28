
# Prithvi-Complimentary Adaptive Fusion Encoder (CAFE)
### Unlocking the full potential of multi-band satellite imagery for flood inundation mapping

The **Prithvi-CAFE** framework introduces a powerful *adaptive hybrid encoder* that fuses **Transformer-based global reasoning (Prithvi-EO-2.0)** with **CNN-based local spatial sensitivity**, enabling high-resolution, reliable flood inundation mapping across multi-channel/sensor inputs.

Prithvi-CAFE integrates:

- 🌍 **Prithvi-EO-2.0 (600M) backbone with lightweight Adapters**  
- 🔁 **Multi-scale multi-stage fusion of ViT + CNN via FAT-Net**  
- 🧠 **Terratorch-compatible custom UPerNet decoders**  
- 📡 **Support for any number of input channels (Sentinel-1/2, PlanetScope, DEM, etc.)**  
- ⚡ **End-to-end PyTorch Lightning training + testing pipeline**


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

🧪 Testing Prithvi-CAFE on Sen1Flood11

We provide access to trained weights and the Sen1Flood11 test data, enabling fully automated testing of the model and reproduction of the reported results.
The same model can be directly tested on similar flood-mapping datasets with only minor path/config modifications.

The model was evaluated on the Sen1Flood11 test split using the Lightning test loop, yielding the following metrics:

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%|█████████████████████████████████████████████████| 23/23 [00:26<00:00,  0.88it/s]


| Metric                                | Value      |
| ------------------------------------- | ---------- |
| `test/Multiclass_Accuracy`            | **0.9778** |
| `test/Multiclass_F1_Score`            | **0.9778** |
| `test/Multiclass_Jaccard_Index`       | **0.9046** |
| `test/Multiclass_Jaccard_Index_Micro` | **0.9566** |
| `test/loss`                           | **0.0815** |

---

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
- Residual CNN + CBAM captures spatial/local texture cues  
- FAT-Net aligns and fuses multi-scale features  
- Decoder reconstructs dense segmentation at full resolution  

---




