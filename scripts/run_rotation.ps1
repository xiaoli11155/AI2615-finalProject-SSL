python train_pretext.py --data-dir data/imagenet --task rotation --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
python finetune.py --data-dir data/imagenet --pretrained outputs/pretext/rotation_bs128_lr0.001_ep20/best.pt --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3 --num-workers 4
python finetune.py --data-dir data/imagenet --shots-per-class 10 --epochs 30 --batch-size 128 --lr 1e-3 --num-workers 4
