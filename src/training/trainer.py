import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.helpers.dataloader import TusDataset
from src.helpers.reproducibility import seed_all
from src.helpers.loaders import make_loader


from src.losses.tube_aware_loss import StableTubeAwareTUSLoss

from src.modelos.ResUnet3D import ResUNet3D_HQ
from src.modelos.Discriminator3D import PatchDiscriminator3D

from src.metrics.validation_metrics import eval_extra_metrics

from src.visualization.visual_callback import VisualCallback

from src.training.schedules import adv_weight_schedule
from src.training.checkpoints import save_ckpt
from src.training.logging_utils import (
    init_csv,
    append_csv,
    log_epoch_to_tensorboard,
    log_extra_metrics_to_tensorboard,
)
from src.training.cgan_loops import train_one_epoch, eval_one_epoch

try:
    from torch.amp.grad_scaler import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler

def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123

    SAVE_DIR = "checkpoints_cgan_exp04_300epoch"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    # no se usa durante train
    TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")
    TB_DIR = os.path.join(SAVE_DIR, "tensorboard")

    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 300
    LR_G = 1e-4
    LR_D = 5e-5
    WEIGHT_DECAY_G = 1e-4
    WEIGHT_DECAY_D = 0.0

    BETAS_G = (0.5, 0.999)
    BETAS_D = (0.5, 0.999)

    BASE_G = 16
    BASE_D = 16

    USE_SE = True
    OUT_POSITIVE = True
    USE_AMP = True

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {10, 20, 30, 40, 50, 60, 80, 100}

    # adversarial schedule
    ADV_START_EPOCH = 1
    ADV_RAMP_END_EPOCH = 15
    ADV_START_VALUE = 1e-3
    ADV_END_VALUE = 1e-2

    # metrics / visuals cadence
    EXTRA_METRICS_EVERY_N_EPOCHS = 10
    VISUAL_EVERY_N_EPOCHS = 10
    VISUAL_FIXED_INDICES = (0, 5, 10)
    VISUAL_SAVE_RAW_NPZ = True

    # voxel size para peak_loc_err_mm
    DX_MM = 0.5

    # =========================
    # SETUP
    # =========================
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    writer = SummaryWriter(log_dir=TB_DIR)

    # =========================
    # DATA
    # =========================
    train_ds = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = make_loader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = make_loader(
        dataset=val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # =========================
    # MODELS (FROM SCRATCH)
    # =========================
    G = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=BASE_G,
        norm_kind="group",
        use_se=USE_SE,
        out_positive=OUT_POSITIVE
    ).to(device)

    D = PatchDiscriminator3D(
        in_ch=3,   # 2 canales de X + 1 canal del campo
        base=BASE_D
    ).to(device)

    # =========================
    # OPTIMS / SCALERS
    # =========================
    optim_G = optim.Adam(
        G.parameters(),
        lr=LR_G,
        betas=BETAS_G,
        weight_decay=WEIGHT_DECAY_G
    )

    optim_D = optim.Adam(
        D.parameters(),
        lr=LR_D,
        betas=BETAS_D,
        weight_decay=WEIGHT_DECAY_D
    )

    try:
        scaler_G = GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))
        scaler_D = GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))
    except TypeError:
        scaler_G = GradScaler(enabled=(device == "cuda" and USE_AMP))
        scaler_D = GradScaler(enabled=(device == "cuda" and USE_AMP))

    # =========================
    # LOSSES
    # =========================
    recon_criterion = StableTubeAwareTUSLoss(
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
    )

    # LSGAN
    adv_criterion = nn.MSELoss()

    # =========================
    # VISUAL CALLBACK (VAL)
    # =========================
    visual_cb = VisualCallback(
        save_dir=SAVE_DIR,
        val_dataset=val_ds,
        device=device,
        every_n_epochs=VISUAL_EVERY_N_EPOCHS,
        fixed_indices=VISUAL_FIXED_INDICES,
        use_amp=USE_AMP,
        save_raw_npz=VISUAL_SAVE_RAW_NPZ,
        writer=writer,
    )

    # =========================
    # CONFIG SAVE
    # =========================
    best_val = float("inf")
    config = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "LR_G": LR_G,
        "LR_D": LR_D,
        "WEIGHT_DECAY_G": WEIGHT_DECAY_G,
        "WEIGHT_DECAY_D": WEIGHT_DECAY_D,
        "BETAS_G": BETAS_G,
        "BETAS_D": BETAS_D,
        "ADV": {
            "ADV_START_EPOCH": ADV_START_EPOCH,
            "ADV_RAMP_END_EPOCH": ADV_RAMP_END_EPOCH,
            "ADV_START_VALUE": ADV_START_VALUE,
            "ADV_END_VALUE": ADV_END_VALUE,
        },
        "LOSS": {
            "lambda_global": 0.8,
            "lambda_focus": 1.5,
            "lambda_grad": 0.08,
            "lambda_tube": 1.0,
            "lambda_profile": 0.6,
            "tube_radius_vox": 3,
            "profile_bins": 16,
            "tube_warmup_start": 3,
            "tube_warmup_end": 12,
            "profile_warmup_start": 5,
            "profile_warmup_end": 15,
        },
        "VISUAL": {
            "VISUAL_EVERY_N_EPOCHS": VISUAL_EVERY_N_EPOCHS,
            "VISUAL_FIXED_INDICES": list(VISUAL_FIXED_INDICES),
            "VISUAL_SAVE_RAW_NPZ": VISUAL_SAVE_RAW_NPZ,
        },
        "EXTRA_METRICS": {
            "EXTRA_METRICS_EVERY_N_EPOCHS": EXTRA_METRICS_EVERY_N_EPOCHS,
            "DX_MM": DX_MM,
        }
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # =========================
    # TRAIN
    # =========================
    try:
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            lambda_adv = adv_weight_schedule(
                epoch=epoch,
                start_epoch=ADV_START_EPOCH,
                ramp_end_epoch=ADV_RAMP_END_EPOCH,
                start_value=ADV_START_VALUE,
                end_value=ADV_END_VALUE
            )

            train_stats = train_one_epoch(
                G=G,
                D=D,
                loader=train_loader,
                optim_G=optim_G,
                optim_D=optim_D,
                scaler_G=scaler_G,
                scaler_D=scaler_D,
                recon_criterion=recon_criterion,
                adv_criterion=adv_criterion,
                device=device,
                epoch=epoch,
                lambda_adv=lambda_adv,
                grad_clip_G=2.0,
                grad_clip_D=1.0,
                use_amp=USE_AMP
            )

            val_stats = eval_one_epoch(
                G=G,
                D=D,
                loader=val_loader,
                recon_criterion=recon_criterion,
                adv_criterion=adv_criterion,
                device=device,
                epoch=epoch,
                lambda_adv=lambda_adv,
                use_amp=USE_AMP
            )

            extra_val_stats = None
            if epoch % EXTRA_METRICS_EVERY_N_EPOCHS == 0:
                extra_val_stats = eval_extra_metrics(
                    G=G,
                    loader=val_loader,
                    device=device,
                    dx_mm=DX_MM,
                    use_amp=USE_AMP,
                )
                log_extra_metrics_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    stats=extra_val_stats,
                    split="val_extra"
                )

            dt = time.time() - t0

            print(
                f"Epoch {epoch:03d} | "
                f"train_G={train_stats['g_total']:.6f} | "
                f"train_recon={train_stats['g_recon']:.6f} | "
                f"train_adv={train_stats['g_adv']:.6f} | "
                f"train_D={train_stats['d_total']:.6f} | "
                f"Dreal={train_stats['d_real_mean']:.3f} | "
                f"Dfake={train_stats['d_fake_mean']:.3f} || "
                f"val_G={val_stats['g_total']:.6f} | "
                f"val_recon={val_stats['g_recon']:.6f} | "
                f"val_adv={val_stats['g_adv']:.6f} | "
                f"val_D={val_stats['d_total']:.6f} | "
                f"vDreal={val_stats['d_real_mean']:.3f} | "
                f"vDfake={val_stats['d_fake_mean']:.3f} | "
                f"glob={val_stats['loss_global']:.4f} | "
                f"focus={val_stats['loss_focus']:.4f} | "
                f"grad={val_stats['loss_grad']:.4f} | "
                f"tube={val_stats['loss_tube']:.4f} | "
                f"profile={val_stats['loss_profile']:.4f} | "
                f"lambda_adv={lambda_adv:.5f} | "
                f"lrG={optim_G.param_groups[0]['lr']:.2e} | "
                f"lrD={optim_D.param_groups[0]['lr']:.2e} | "
                f"time={dt:.1f}s"
            )

            if extra_val_stats is not None:
                print(
                    f"[VAL extra] "
                    f"peak_rel={extra_val_stats['peak_rel_err']:.4f} | "
                    f"peak_loc_mm={extra_val_stats['peak_loc_err_mm']:.3f} | "
                    f"dice50={extra_val_stats['dice50']:.4f} | "
                    f"mse_brain={extra_val_stats['mse_brain']:.4f}"
                )

            append_csv(
                CSV_PATH,
                epoch=epoch,
                train_stats=train_stats,
                val_stats=val_stats,
                extra_val_stats=extra_val_stats,
                lambda_adv=lambda_adv,
                lr_G=optim_G.param_groups[0]["lr"],
                lr_D=optim_D.param_groups[0]["lr"],
                time_sec=dt
            )

            log_epoch_to_tensorboard(
                writer=writer,
                epoch=epoch,
                train_stats=train_stats,
                val_stats=val_stats,
                lambda_adv=lambda_adv,
                lr_G=optim_G.param_groups[0]["lr"],
                lr_D=optim_D.param_groups[0]["lr"]
            )

            save_ckpt(
                os.path.join(SAVE_DIR, "last.pth"),
                G, D,
                optim_G, optim_D,
                scaler_G, scaler_D,
                epoch, best_val, config
            )

            if val_stats["g_total"] < best_val:
                best_val = val_stats["g_total"]
                save_ckpt(
                    os.path.join(SAVE_DIR, "best.pth"),
                    G, D,
                    optim_G, optim_D,
                    scaler_G, scaler_D,
                    epoch, best_val, config
                )
                print(f"New best val_G: {best_val:.6f}")

            if (epoch % SAVE_EVERY_N_EPOCHS == 0) or (epoch in SPECIFIC_SAVE_EPOCHS):
                save_ckpt(
                    os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pth"),
                    G, D,
                    optim_G, optim_D,
                    scaler_G, scaler_D,
                    epoch, best_val, config
                )
                print(f"Saved checkpoint: epoch_{epoch:03d}.pth")

            visual_cb(G, epoch)

        print("Training done. Best val_G:", best_val)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
