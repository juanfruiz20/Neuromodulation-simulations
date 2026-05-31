import torch


def save_ckpt(
    path,
    G,
    D,
    optim_G,
    optim_D,
    scaler_G,
    scaler_D,
    epoch,
    best_val,
    config
):
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optim_G": optim_G.state_dict() if optim_G is not None else None,
        "optim_D": optim_D.state_dict() if optim_D is not None else None,
        "scaler_G": scaler_G.state_dict() if scaler_G is not None else None,
        "scaler_D": scaler_D.state_dict() if scaler_D is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "config": config,
    }

    torch.save(ckpt, path)
