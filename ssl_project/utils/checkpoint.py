from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args) if hasattr(args, "__dict__") else args,
        },
        path,
    )


def load_encoder_from_pretext(classifier_model, checkpoint_path, strict: bool = False):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"]
    encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    missing, unexpected = classifier_model.encoder.load_state_dict(encoder_state, strict=strict)
    return missing, unexpected
