# AI2612 Self-Supervised Learning Project

This project implements ResNet-18 based self-supervised pretraining on Tiny ImageNet, CIFAR-10, and CIFAR-100, plus downstream few-shot classification fine-tuning.

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

Put CIFAR data under one of these layouts:

```text
data/cifar/cifar-10-batches-py
data/cifar/cifar-100-python
```

## Self-Supervised Pretraining

```bash
python train_pretext.py --dataset tiny-imagenet --data-dir data/tiny-imagenet-200 --task rotation --epochs 20 --batch-size 128 --lr 1e-3
python train_pretext.py --dataset cifar10 --data-dir data/cifar --task rotation --epochs 20 --batch-size 128 --lr 1e-3 --image-size 32
python train_pretext.py --dataset cifar100 --data-dir data/cifar --task jigsaw --epochs 20 --batch-size 128 --lr 1e-3 --image-size 32
```

Checkpoints and per-epoch JSON logs are saved under `outputs/pretext/`.

## Fine-Tuning and Random-Init Baseline

Fine-tune a self-supervised encoder with 10 labeled images per class:

```bash
python finetune.py --dataset tiny-imagenet --data-dir data/tiny-imagenet-200 --pretrained outputs/pretext/tiny-imagenet/rotation_bs128_lr0.001_ep20/best.pt --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3
```

Train the same classifier from random initialization:

```bash
python finetune.py --dataset cifar10 --data-dir data/cifar --pretrained outputs/pretext/cifar10/rotation_bs128_lr0.001_ep20/best.pt --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3 --image-size 32
python finetune.py --dataset cifar10 --data-dir data/cifar --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3 --image-size 32
```

Adapter tuning freezes the backbone and trains lightweight residual adapters plus the classifier:

```bash
python finetune.py --dataset cifar10 --data-dir data/cifar --pretrained outputs/pretext/cifar10/jigsaw_bs128_lr0.001_ep20/best.pt --shots-per-class 100 --epochs 30 --batch-size 128 --lr 1e-3 --image-size 32 --adapter-dim 16
```

Fine-tuning reports Top-1 and Top-5 accuracy on the held-out validation/test split.

## Hyper-Parameter Sweep

```bash
python run_experiments.py --dataset tiny-imagenet --data-dir data/tiny-imagenet-200 --tasks rotation jigsaw relative_patch --batch-sizes 64 128 --lrs 1e-3 3e-4 --pretext-epochs 20 --finetune-epochs 30
```

The sweep writes `outputs/experiment_summary.csv`, which is suitable for the report section comparing batch size, learning rate, training time, and accuracy.

## Suggested Report Structure

1. Method: describe ResNet-18 backbone, data augmentation, and each pretext label construction.
2. Pretraining results: include pretext validation accuracy/loss curves.
3. Hyper-parameter analysis: compare batch size, learning rate, epochs, runtime, and memory/resource observations.
4. Downstream evaluation: compare SSL-initialized fine-tuning with random initialization using Top-1 and Top-5 accuracy.
5. Discussion: explain which pretext task transfers best and possible reasons.
