
import os
import torch

def save_checkpoint(model, optimiser, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
    }
    path = os.path.join(save_dir, f"ckpt_{step}.pt")
    torch.save(ckpt, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(model, optimiser, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    step = ckpt["step"]
    print(f"loaded checkpoint from step {step}")
    return step