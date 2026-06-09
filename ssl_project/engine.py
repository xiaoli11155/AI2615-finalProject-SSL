from __future__ import annotations

import time

import torch
from tqdm import tqdm

from ssl_project.utils.metrics import topk_accuracy


def build_grad_scaler(device, amp: bool):
    enabled = amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, amp: bool):
    enabled = amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def move_batch(batch, device):
    inputs, targets = batch
    if isinstance(inputs, (tuple, list)):
        inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
    else:
        inputs = inputs.to(device, non_blocking=True)
    return inputs, targets.to(device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, amp=False):
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_seen = 0
    start = time.time()
    scaler = build_grad_scaler(device, amp)
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in pbar:
        inputs, targets = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = targets.size(0)
        top1 = topk_accuracy(logits.detach(), targets, topk=(1,))[0]
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_seen += bs
        pbar.set_postfix(loss=total_loss / total_seen, top1=total_top1 / total_seen)
    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, topk=(1, 5)):
    model.eval()
    total_loss = 0.0
    totals = [0.0 for _ in topk]
    total_seen = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs, targets = move_batch(batch, device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        accs = topk_accuracy(logits, targets, topk=topk)
        total_loss += loss.item() * bs
        totals = [old + acc * bs for old, acc in zip(totals, accs)]
        total_seen += bs
    metrics = {"loss": total_loss / max(1, total_seen)}
    for k, total in zip(topk, totals):
        metrics[f"top{k}"] = total / max(1, total_seen)
    return metrics
