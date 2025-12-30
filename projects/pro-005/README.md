# Pro-005 — SqueezeNet Transfer Learning (5-Flowers)

A minimal PyTorch/Colab notebook that fine-tunes **SqueezeNet1_1** (pretrained on ImageNet) for **5-class flower classification** and logs training to **Weights & Biases (wandb)**.

## Contents
- Data transforms (train/val)
- Dataset loading with `ImageFolder`
- Pretrained **SqueezeNet1_1** + classifier replacement for 5 classes
- Training + validation loop (loss/accuracy)
- Experiment tracking with `wandb`

## Requirements
- Python 3.x
- `torch`, `torchvision`, `numpy`
- `wandb`
- (Optional) Google Colab + GPU

Install (example):
```bash
pip install torch torchvision wandb numpy
```

## Dataset layout
The notebook expects a folder structure compatible with `torchvision.datasets.ImageFolder`:

```
flowers/
  train/
    class_0/
    class_1/
    class_2/
    class_3/
    class_4/
  val/
    class_0/
    class_1/
    class_2/
    class_3/
    class_4/
```

In the notebook these are set as:
- `train_dir = /content/drive/MyDrive/Colab Notebooks/flowers/train`
- `val_dir   = /content/drive/MyDrive/Colab Notebooks/flowers/val`

> If you are **not** using Google Drive, change `train_dir` and `val_dir` to your local paths.

## Data preprocessing
Input images are resized to **224×224**, normalized with ImageNet mean/std, and training data uses a random horizontal flip:

- Train: `Resize(224) → RandomHorizontalFlip → ToTensor → Normalize`
- Val:   `Resize(224) → ToTensor → Normalize`

## Model (SqueezeNet1_1) — architecture overview
SqueezeNet is a lightweight CNN designed to achieve strong accuracy with **very few parameters**.  
Its key idea is the **Fire module**:

### Fire module
Each Fire module has two parts:
1. **Squeeze layer**: `1×1` conv to reduce channels (cheap, reduces parameters)
2. **Expand layer**: a mix of `1×1` conv and `3×3` conv whose outputs are **concatenated**

This keeps computation small while preserving representational power.

### Classifier head in this notebook
The pretrained backbone is loaded:
```python
model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
```
Then the last conv in the classifier is replaced to output **5 classes**:
```python
model.classifier[1] = nn.Conv2d(512, 5, kernel_size=1)
```

### Freezing strategy
All parameters are frozen, then only the final classifier conv is unfrozen:
```python
for p in model.parameters(): p.requires_grad = False
for p in model.classifier[1].parameters(): p.requires_grad = True
```
So training updates **only** the new classification layer (fast and stable for small datasets).

## Training
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (LR = `1e-3`)
- Epochs: `50`
- Batch size: `16`
- Device: CUDA if available

The loop logs step-level metrics every `log_every` steps (default 10) and logs epoch metrics every epoch:
- `train/loss`, `train/acc`, `val/loss`, `val/acc`, learning rate

## wandb logging
The notebook starts a wandb run:
```python
wandb.init(project="SqueesNet-5flowers", config={...})
```
and watches the model:
```python
wandb.watch(model, log="all", log_freq=10)
```
> Note: `log="all"` can be heavy/slow. If Colab becomes slow, change to `log="gradients"` or remove `watch()`.


## How to run (Colab)
1. Ensure your dataset exists at the configured `train_dir` and `val_dir`.
2. Run all cells top-to-bottom.
3. Check training logs in the notebook and in your wandb dashboard.

---
**File:** `projects/pro-005/src/squeezNet.ipynb`
