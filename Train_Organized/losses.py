import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    grad3d, masked_mean, ramp_linear, interp_linear,
    make_focus_roi, make_peak_mask, approx_max, soft_argmax3d
)


class StableFocusAwareTUSLoss(nn.Module):
    """
    Progressive, focus-aware loss for TUS prediction.
    Includes global L1, focus MSE, gradients, peak, and localization terms.
    """

    def __init__(
        self,
        lambda_global=1.0,
        lambda_focus=2.0,
        lambda_peak=0.50,
        lambda_loc=0.25,
        lambda_grad=0.05,

        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,
        peak_frac=0.85,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        peak_warmup_start=10,
        peak_warmup_end=30,
        loc_warmup_start=20,
        loc_warmup_end=45,

        beta_peak_start=8.0,
        beta_peak_end=25.0,
        beta_loc_start=8.0,
        beta_loc_end=35.0,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_peak = float(lambda_peak)
        self.lambda_loc = float(lambda_loc)
        self.lambda_grad = float(lambda_grad)

        self.focus_frac = float(focus_frac)
        self.focus_min_thr = float(focus_min_thr)
        self.focus_dilate_ks = int(focus_dilate_ks)
        self.peak_frac = float(peak_frac)

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.peak_warmup_start = int(peak_warmup_start)
        self.peak_warmup_end = int(peak_warmup_end)
        self.loc_warmup_start = int(loc_warmup_start)
        self.loc_warmup_end = int(loc_warmup_end)

        self.beta_peak_start = float(beta_peak_start)
        self.beta_peak_end = float(beta_peak_end)
        self.beta_loc_start = float(beta_loc_start)
        self.beta_loc_end = float(beta_loc_end)

        self.eps = float(eps)

    def get_schedule(self, epoch: int):
        """Compute learning schedules for current epoch."""
        r_peak = ramp_linear(epoch, self.peak_warmup_start,
                             self.peak_warmup_end)
        r_loc = ramp_linear(epoch, self.loc_warmup_start, self.loc_warmup_end)

        beta_peak = interp_linear(
            r_peak, self.beta_peak_start, self.beta_peak_end)
        beta_loc = interp_linear(r_loc, self.beta_loc_start, self.beta_loc_end)

        return {
            "r_peak": r_peak,
            "r_loc": r_loc,
            "beta_peak": beta_peak,
            "beta_loc": beta_loc,
            "lambda_peak_eff": self.lambda_peak * r_peak,
            "lambda_loc_eff": self.lambda_loc * r_loc,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 1, return_dict: bool = False):
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
        peak_mask, _ = make_peak_mask(target32, frac=self.peak_frac)

        # 1) Global weighted L1
        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * \
            rel.pow(self.global_peak_gamma)
        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        # 2) Focus ROI relative MSE
        focus_num = masked_mean(
            (pred_pos - target32).pow(2), focus_roi, eps=self.eps)
        focus_den = masked_mean(target32.pow(
            2), focus_roi, eps=self.eps).clamp_min(self.eps)
        loss_focus = focus_num / focus_den

        # 3) Focus grad relative loss
        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + \
            torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(grad_diff, focus_roi, eps=self.eps)
        grad_den = masked_mean(grad_ref, focus_roi,
                               eps=self.eps).clamp_min(self.eps)
        loss_grad = grad_num / grad_den

        # 4) Peak relative loss
        if sched["lambda_peak_eff"] > 0.0:
            pred_peak = approx_max(
                pred_pos, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            targ_peak = approx_max(
                target32, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            loss_peak = (torch.abs(pred_peak - targ_peak) /
                         (torch.abs(targ_peak) + self.eps)).mean()
        else:
            loss_peak = torch.zeros(
                (), device=pred.device, dtype=torch.float32)

        # 5) Peak location loss
        if sched["lambda_loc_eff"] > 0.0:
            pred_loc = soft_argmax3d(
                pred_pos, mask=focus_roi, beta=sched["beta_loc"])
            targ_loc = soft_argmax3d(
                target32, mask=focus_roi, beta=sched["beta_loc"])
            loss_loc = F.l1_loss(pred_loc, targ_loc)
        else:
            loss_loc = torch.zeros((), device=pred.device, dtype=torch.float32)

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_peak_eff"] * loss_peak
            + sched["lambda_loc_eff"] * loss_loc
        )

        if return_dict:
            comp = {
                "loss_total": float(total.detach().item()),
                "loss_global": float(loss_global.detach().item()),
                "loss_focus": float(loss_focus.detach().item()),
                "loss_grad": float(loss_grad.detach().item()),
                "loss_peak": float(loss_peak.detach().item()),
                "loss_loc": float(loss_loc.detach().item()),
                "r_peak": float(sched["r_peak"]),
                "r_loc": float(sched["r_loc"]),
                "beta_peak": float(sched["beta_peak"]),
                "beta_loc": float(sched["beta_loc"]),
                "lambda_peak_eff": float(sched["lambda_peak_eff"]),
                "lambda_loc_eff": float(sched["lambda_loc_eff"]),
            }
            return total, comp

        return total
