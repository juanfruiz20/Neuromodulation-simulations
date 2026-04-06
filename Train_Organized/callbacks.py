import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from contextlib import nullcontext


class VisualCallback:
    """
    Save visualizations of fixed validation cases.
    Runs every N epochs.
    """

    def __init__(
        self,
        save_dir,
        val_dataset,
        device,
        every_n_epochs=10,
        fixed_indices=(0, 5, 10),
        use_amp=True,
        save_raw_npz=False,
    ):
        self.device = device
        self.every_n_epochs = int(every_n_epochs)
        self.use_amp = bool(use_amp)
        self.save_raw_npz = bool(save_raw_npz)

        self.out_dir = os.path.join(save_dir, "visuals")
        os.makedirs(self.out_dir, exist_ok=True)

        n_val = len(val_dataset)
        valid_indices = [idx for idx in fixed_indices if 0 <= idx < n_val]

        if len(valid_indices) == 0:
            valid_indices = list(range(min(3, n_val)))

        self.samples = []
        for idx in valid_indices:
            x, y = val_dataset[idx]

            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
            if not torch.is_tensor(y):
                y = torch.from_numpy(y)

            self.samples.append((int(idx), x.clone().cpu(), y.clone().cpu()))

        print(
            f"🖼️ VisualCallback active with fixed cases: {[s[0] for s in self.samples]}")

    def should_run(self, epoch: int) -> bool:
        return self.every_n_epochs > 0 and epoch % self.every_n_epochs == 0

    @staticmethod
    def _to_numpy_3d(t: torch.Tensor):
        return t.detach().float().cpu().squeeze().numpy()

    @staticmethod
    def _norm_2d(a: np.ndarray):
        a = a.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn + 1e-8)

    @classmethod
    def _make_overlay(cls, gt2d: np.ndarray, pred2d: np.ndarray):
        """
        RGB overlay:
        - red   = pred
        - green = GT
        - yellow = overlap
        """
        g = cls._norm_2d(gt2d)
        p = cls._norm_2d(pred2d)
        rgb = np.stack([p, g, np.zeros_like(g)], axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    @staticmethod
    def _peak_idx(vol: np.ndarray):
        return np.unravel_index(np.argmax(vol), vol.shape)

    def _save_case_figure(self, pred: np.ndarray, gt: np.ndarray, epoch: int, case_idx: int):
        pred = np.clip(pred, 0.0, None)
        err = np.abs(pred - gt)

        z_gt, y_gt, x_gt = self._peak_idx(gt)
        z_pr, y_pr, x_pr = self._peak_idx(pred)

        peak_err_vox = np.sqrt(
            (z_pr - z_gt) ** 2 +
            (y_pr - y_gt) ** 2 +
            (x_pr - x_gt) ** 2
        )

        # slices fixed at GT peak
        gt_ax = gt[z_gt, :, :]
        pr_ax = pred[z_gt, :, :]
        er_ax = err[z_gt, :, :]

        gt_co = gt[:, y_gt, :]
        pr_co = pred[:, y_gt, :]
        er_co = err[:, y_gt, :]

        gt_sa = gt[:, :, x_gt]
        pr_sa = pred[:, :, x_gt]
        er_sa = err[:, :, x_gt]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        rows = [
            ("Axial", gt_ax, pr_ax, er_ax),
            ("Coronal", gt_co, pr_co, er_co),
            ("Sagittal", gt_sa, pr_sa, er_sa),
        ]

        vmax_main = max(float(gt.max()), float(pred.max()), 1e-8)
        vmax_err = max(float(err.max()), 1e-8)

        for r, (name, g, p, e) in enumerate(rows):
            axes[r, 0].imshow(g, cmap="jet", origin="lower",
                              vmin=0, vmax=vmax_main)
            axes[r, 0].set_title(f"{name} | GT")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(p, cmap="jet", origin="lower",
                              vmin=0, vmax=vmax_main)
            axes[r, 1].set_title(f"{name} | Pred")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(e, cmap="magma", origin="lower",
                              vmin=0, vmax=vmax_err)
            axes[r, 2].set_title(f"{name} | |Error|")
            axes[r, 2].axis("off")

            overlay = self._make_overlay(g, p)
            axes[r, 3].imshow(overlay, origin="lower")
            axes[r, 3].set_title(f"{name} | Overlay")
            axes[r, 3].axis("off")

        gt_peak = float(gt.max())
        pr_peak = float(pred.max())
        peak_rel_err = abs(pr_peak - gt_peak) / (abs(gt_peak) + 1e-8)

        fig.suptitle(
            f"Epoch {epoch:03d} | Case {case_idx} | "
            f"GT peak idx=({z_gt},{y_gt},{x_gt}) | "
            f"Pred peak idx=({z_pr},{y_pr},{x_pr}) | "
            f"PeakErr={peak_err_vox:.2f} vox | "
            f"PeakRelErr={peak_rel_err:.4f}",
            fontsize=12
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(
            self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.png")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if self.save_raw_npz:
            out_npz = os.path.join(
                self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.npz")
            np.savez_compressed(
                out_npz,
                pred=pred.astype(np.float32),
                gt=gt.astype(np.float32),
                peak_gt=np.array([z_gt, y_gt, x_gt], dtype=np.int32),
                peak_pred=np.array([z_pr, y_pr, x_pr], dtype=np.int32),
                peak_err_vox=np.array([peak_err_vox], dtype=np.float32),
                peak_rel_err=np.array([peak_rel_err], dtype=np.float32),
            )

    @torch.no_grad()
    def __call__(self, model, epoch: int):
        if not self.should_run(epoch):
            return

        was_training = model.training
        model.eval()

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (self.use_amp and self.device == "cuda")
            else nullcontext()
        )

        print(
            f"🖼️ Saving validation visuals at epoch {epoch:03d}...")

        for case_idx, x_cpu, y_cpu in self.samples:
            X = x_cpu.unsqueeze(0).to(self.device, non_blocking=True)

            with amp_ctx:
                pred = model(X)

            pred_np = self._to_numpy_3d(pred.clamp_min(0.0))
            gt_np = self._to_numpy_3d(y_cpu)

            self._save_case_figure(pred_np, gt_np, epoch, case_idx)

        if was_training:
            model.train()
