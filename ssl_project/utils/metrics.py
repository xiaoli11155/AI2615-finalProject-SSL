import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        k = min(k, logits.size(1))
        out.append(correct[:k].reshape(-1).float().sum().item() / targets.size(0))
    return out
