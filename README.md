# AI2612 Self-Supervised Learning Project

This project implements ResNet-18 based self-supervised pretraining on Tiny ImageNet and downstream few-shot classification fine-tuning.

Implemented pretext tasks:

- Rotation Prediction: predict 0, 90, 180, or 270 degree image rotation.
- Jigsaw Puzzle: classify which fixed 3x3 tile permutation was applied.
- Relative Patch Position: predict the relative position of two image patches.

## Setup

```bash
pip install -r requirements.txt
```

Put Tiny ImageNet under one of these layouts:

```text
data/imagenet/train/<wnid>/images/*.JPEG
data/imagenet/val/images/*.JPEG
data/imagenet/val/val_annotations.txt
```

or:

```text
data/imagenet/tiny-imagenet-200/train/...
```

## Self-Supervised Pretraining

```bash
python train_pretext.py --data-dir data/imagenet --task rotation --epochs 20 --batch-size 128 --lr 1e-3
python train_pretext.py --data-dir data/imagenet --task jigsaw --epochs 20 --batch-size 128 --lr 1e-3
python train_pretext.py --data-dir data/imagenet --task relative_patch --epochs 20 --batch-size 128 --lr 1e-3
```

Checkpoints and per-epoch JSON logs are saved under `outputs/pretext/`.

## Fine-Tuning and Random-Init Baseline

Fine-tune a self-supervised encoder with 10 labeled images per class:

```bash
python finetune.py --data-dir data/imagenet --pretrained outputs/pretext/rotation_bs128_lr0.001_ep20/best.pt --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3
```

Train the same classifier from random initialization:

```bash
python finetune.py --data-dir data/imagenet --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3
```

Fine-tuning reports Top-1 and Top-5 accuracy on the Tiny ImageNet validation split.

## Hyper-Parameter Sweep

```bash
python run_experiments.py --data-dir data/imagenet --tasks rotation jigsaw relative_patch --batch-sizes 64 128 --lrs 1e-3 3e-4 --pretext-epochs 20 --finetune-epochs 30
```

The sweep writes `outputs/experiment_summary.csv`, which is suitable for the report section comparing batch size, learning rate, training time, and accuracy.

## Suggested Report Structure

1. Method: describe ResNet-18 backbone, data augmentation, and each pretext label construction.
2. Pretraining results: include pretext validation accuracy/loss curves.
3. Hyper-parameter analysis: compare batch size, learning rate, epochs, runtime, and memory/resource observations.
4. Downstream evaluation: compare SSL-initialized fine-tuning with random initialization using Top-1 and Top-5 accuracy.
5. Discussion: explain which pretext task transfers best and possible reasons.
