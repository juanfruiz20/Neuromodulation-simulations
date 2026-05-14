import torch
import torch.nn as nn

from src.losses.basic_ops import (
    grad3d,
    masked_mean,
    ramp_linear,
    make_focus_roi,
)

from src.losses.tube_geometry import (
    build_tube_roi_and_tcoord,
    compute_profile_loss,
)


class StableTubeAwareTUSLoss(nn.Module):
    def __init__(
        self,
        lambda_global=0.8,
        lambda_focus=1.5,
        lambda_grad=0.08,
        lambda_tube=1.0,
        lambda_profile=0.6,

        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,

        tube_radius_vox=3,
        profile_bins=16,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        tube_warmup_start=3,
        tube_warmup_end=12,
        profile_warmup_start=5,
        profile_warmup_end=15,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_grad = float(lambda_grad)
        self.lambda_tube = float(lambda_tube)
        self.lambda_profile = float(lambda_profile)

        self.focus_frac = float(focus_frac)
        self.focus_min_thr = float(focus_min_thr)
        self.focus_dilate_ks = int(focus_dilate_ks)

        self.tube_radius_vox = int(tube_radius_vox)
        self.profile_bins = int(profile_bins)

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.tube_warmup_start = int(tube_warmup_start)
        self.tube_warmup_end = int(tube_warmup_end)
        self.profile_warmup_start = int(profile_warmup_start)
        self.profile_warmup_end = int(profile_warmup_end)

        self.eps = float(eps)

    def get_schedule(self, epoch: int):
        r_tube = ramp_linear(
            epoch,
            self.tube_warmup_start,
            self.tube_warmup_end
        )

        r_profile = ramp_linear(
            epoch,
            self.profile_warmup_start,
            self.profile_warmup_end
        )

        return {
            "r_tube": r_tube,
            "r_profile": r_profile,
            "lambda_tube_eff": self.lambda_tube * r_tube,
            "lambda_profile_eff": self.lambda_profile * r_profile,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        epoch: int = 1,
        return_dict: bool = False
    ):
        pred32 = pred.float()
        target32 = target.float()
        pred_pos = pred32.clamp_min(0.0)

        sched = self.get_schedule(epoch)

        focus_roi, peak = make_focus_roi(
            target32,
            frac=self.focus_frac,
            min_thr=self.focus_min_thr,
            dilate_ks=self.focus_dilate_ks
        )

        rel = (target32 / peak).clamp(0.0, 1.0)

        w_global = (
            1.0
            + self.global_peak_weight * rel.pow(self.global_peak_gamma)
        )

        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        focus_num = masked_mean(
            (pred_pos - target32).pow(2),
            focus_roi,
            eps=self.eps
        )

        focus_den = masked_mean(
            target32.pow(2),
            focus_roi,
            eps=self.eps
        ).clamp_min(self.eps)

        loss_focus = focus_num / focus_den

        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = (
            torch.abs(pdx - tdx)
            + torch.abs(pdy - tdy)
            + torch.abs(pdz - tdz)
        )

        grad_ref = (
            torch.abs(tdx)
            + torch.abs(tdy)
            + torch.abs(tdz)
        )

        grad_num = masked_mean(
            grad_diff,
            focus_roi,
            eps=self.eps
        )

        grad_den = masked_mean(
            grad_ref,
            focus_roi,
            eps=self.eps
        ).clamp_min(self.eps)

        loss_grad = grad_num / grad_den

        tube_roi, t_coord, _, _ = build_tube_roi_and_tcoord(
            source_mask=source_mask,
            target=target32,
            radius_vox=self.tube_radius_vox
        )

        if sched["lambda_tube_eff"] > 0.0:
            tube_num = masked_mean(
                (pred_pos - target32).pow(2),
                tube_roi,
                eps=self.eps
            )

            tube_den = masked_mean(
                target32.pow(2),
                tube_roi,
                eps=self.eps
            ).clamp_min(self.eps)

            loss_tube = tube_num / tube_den

        else:
            loss_tube = torch.zeros(
                (),
                device=pred.device,
                dtype=torch.float32
            )

        if sched["lambda_profile_eff"] > 0.0:
            loss_profile = compute_profile_loss(
                pred=pred_pos,
                target=target32,
                tube_roi=tube_roi,
                t_coord=t_coord,
                num_bins=self.profile_bins,
                eps=self.eps
            )

        else:
            loss_profile = torch.zeros(
                (),
                device=pred.device,
                dtype=torch.float32
            )

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_tube_eff"] * loss_tube
            + sched["lambda_profile_eff"] * loss_profile
        )

        if return_dict:
            comp = {
                "loss_total_recon": float(total.detach().item()),
                "loss_global": float(loss_global.detach().item()),
                "loss_focus": float(loss_focus.detach().item()),
                "loss_grad": float(loss_grad.detach().item()),
                "loss_tube": float(loss_tube.detach().item()),
                "loss_profile": float(loss_profile.detach().item()),
                "r_tube": float(sched["r_tube"]),
                "r_profile": float(sched["r_profile"]),
                "lambda_tube_eff": float(sched["lambda_tube_eff"]),
                "lambda_profile_eff": float(sched["lambda_profile_eff"]),
            }

            return total, comp

        return total