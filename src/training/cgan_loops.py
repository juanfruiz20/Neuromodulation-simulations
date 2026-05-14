from contextlib import nullcontext
from typing import Dict

import torch


def train_one_epoch(
    G,
    D,
    loader,
    optim_G,
    optim_D,
    scaler_G,
    scaler_D,
    recon_criterion,
    adv_criterion,
    device,
    epoch: int,
    lambda_adv: float,
    grad_clip_G: float = 2.0,
    grad_clip_D: float = 1.0,
    use_amp: bool = True
) -> Dict[str, float]:
    G.train()
    D.train()

    sums = {
        "g_total": 0.0,
        "g_recon": 0.0,
        "g_adv": 0.0,
        "d_total": 0.0,
        "d_real_mean": 0.0,
        "d_fake_mean": 0.0,
    }

    n_batches = 0

    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        source_mask = X[:, 0:1]

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and device == "cuda")
            else nullcontext()
        )

        # -------------------------
        # A) Train Discriminator
        # -------------------------
        optim_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            with amp_ctx:
                y_hat_det = G(X)

        with amp_ctx:
            pred_real = D(X, y)
            pred_fake = D(X, y_hat_det)

            loss_d_real = adv_criterion(
                pred_real,
                torch.ones_like(pred_real)
            )

            loss_d_fake = adv_criterion(
                pred_fake,
                torch.zeros_like(pred_fake)
            )

            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        if not torch.isfinite(loss_d):
            print(
                f"[WARN] Non-finite loss_D at epoch={epoch}, "
                f"step={step}. Batch skipped."
            )
            continue

        if device == "cuda" and use_amp:
            scaler_D.scale(loss_d).backward()
            scaler_D.unscale_(optim_D)

            if grad_clip_D is not None and grad_clip_D > 0:
                torch.nn.utils.clip_grad_norm_(
                    D.parameters(),
                    max_norm=float(grad_clip_D)
                )

            scaler_D.step(optim_D)
            scaler_D.update()

        else:
            loss_d.backward()

            if grad_clip_D is not None and grad_clip_D > 0:
                torch.nn.utils.clip_grad_norm_(
                    D.parameters(),
                    max_norm=float(grad_clip_D)
                )

            optim_D.step()

        # -------------------------
        # B) Train Generator
        # -------------------------
        optim_G.zero_grad(set_to_none=True)

        with amp_ctx:
            y_hat = G(X)

            loss_recon, _ = recon_criterion(
                y_hat,
                y,
                source_mask=source_mask,
                epoch=epoch,
                return_dict=True
            )

            pred_fake_for_g = D(X, y_hat)

            loss_adv_g = adv_criterion(
                pred_fake_for_g,
                torch.ones_like(pred_fake_for_g)
            )

            loss_g = loss_recon + lambda_adv * loss_adv_g

        if not torch.isfinite(loss_g):
            print(
                f"[WARN] Non-finite loss_G at epoch={epoch}, "
                f"step={step}. Batch skipped."
            )
            continue

        if device == "cuda" and use_amp:
            scaler_G.scale(loss_g).backward()
            scaler_G.unscale_(optim_G)

            if grad_clip_G is not None and grad_clip_G > 0:
                torch.nn.utils.clip_grad_norm_(
                    G.parameters(),
                    max_norm=float(grad_clip_G)
                )

            scaler_G.step(optim_G)
            scaler_G.update()

        else:
            loss_g.backward()

            if grad_clip_G is not None and grad_clip_G > 0:
                torch.nn.utils.clip_grad_norm_(
                    G.parameters(),
                    max_norm=float(grad_clip_G)
                )

            optim_G.step()

        sums["g_total"] += float(loss_g.item())
        sums["g_recon"] += float(loss_recon.item())
        sums["g_adv"] += float(loss_adv_g.item())
        sums["d_total"] += float(loss_d.item())
        sums["d_real_mean"] += float(pred_real.detach().mean().item())
        sums["d_fake_mean"] += float(pred_fake.detach().mean().item())

        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def eval_one_epoch(
    G,
    D,
    loader,
    recon_criterion,
    adv_criterion,
    device,
    epoch: int,
    lambda_adv: float,
    use_amp: bool = True
) -> Dict[str, float]:
    G.eval()
    D.eval()

    sums = {
        "g_total": 0.0,
        "g_recon": 0.0,
        "g_adv": 0.0,
        "d_total": 0.0,
        "d_real_mean": 0.0,
        "d_fake_mean": 0.0,
        "loss_global": 0.0,
        "loss_focus": 0.0,
        "loss_grad": 0.0,
        "loss_tube": 0.0,
        "loss_profile": 0.0,
    }

    n_batches = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device == "cuda")
        else nullcontext()
    )

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        source_mask = X[:, 0:1]

        with amp_ctx:
            y_hat = G(X)

            loss_recon, recon_comp = recon_criterion(
                y_hat,
                y,
                source_mask=source_mask,
                epoch=epoch,
                return_dict=True
            )

            pred_fake_for_g = D(X, y_hat)

            loss_adv_g = adv_criterion(
                pred_fake_for_g,
                torch.ones_like(pred_fake_for_g)
            )

            loss_g = loss_recon + lambda_adv * loss_adv_g

            pred_real = D(X, y)
            pred_fake = D(X, y_hat.detach())

            loss_d_real = adv_criterion(
                pred_real,
                torch.ones_like(pred_real)
            )

            loss_d_fake = adv_criterion(
                pred_fake,
                torch.zeros_like(pred_fake)
            )

            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        if (not torch.isfinite(loss_g)) or (not torch.isfinite(loss_d)):
            print(
                f"[WARN] Non-finite validation loss at epoch={epoch}. "
                "Batch skipped."
            )
            continue

        sums["g_total"] += float(loss_g.item())
        sums["g_recon"] += float(loss_recon.item())
        sums["g_adv"] += float(loss_adv_g.item())
        sums["d_total"] += float(loss_d.item())
        sums["d_real_mean"] += float(pred_real.detach().mean().item())
        sums["d_fake_mean"] += float(pred_fake.detach().mean().item())

        sums["loss_global"] += float(recon_comp["loss_global"])
        sums["loss_focus"] += float(recon_comp["loss_focus"])
        sums["loss_grad"] += float(recon_comp["loss_grad"])
        sums["loss_tube"] += float(recon_comp["loss_tube"])
        sums["loss_profile"] += float(recon_comp["loss_profile"])

        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}