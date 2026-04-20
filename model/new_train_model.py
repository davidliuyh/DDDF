"""
GAN training adapted for multi-channel (vector displacement) data.

Extends train_model.py to auto-detect the number of channels from the
training data shape and use the configurable-channel UNet3D / PatchDiscriminator3D
from new_model.py.
"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.new_model as nnmodel
from tqdm.auto import tqdm


def train_gan(training_data_path, save_file_name, batch_size=64, epochs=50,
              lr_g=1e-4, lr_d=1e-4, lambda_pixel=10.0, n_disc_layers=3,
              lambda_fm=10.0, d_update_interval=2, use_spectral_norm=True,
              checkpoint_interval=10, resume_checkpoint=None, overwrite=False,
              lambda_gp=10.0, use_multiscale_disc=False, disc_base_channels=32):
    """Conditional GAN training with WGAN-GP for multi-channel fields.

    Auto-detects channel count from data:
    - 3D patches (N, D, H, W): single-channel, unsqueeze to (N, 1, D, H, W)
    - 4D patches (N, D, H, W, C): multi-channel, permute to (N, C, D, H, W)
    """
    final_path = f'{save_file_name}-e{epochs}.pth'
    if not overwrite and os.path.exists(final_path):
        print(f"Model already exists, skipping training: {final_path}")
        return

    data = np.load(training_data_path)
    input_patches  = data['input_patches']
    target_patches = data['target_patches']

    # ── Auto-detect channels ──────────────────────────────────────────────
    if input_patches.ndim == 4:
        # (N, D, H, W) → scalar, add channel dim
        n_channels = 1
        spatial_shape = input_patches.shape[1:4]
        input_t  = torch.from_numpy(input_patches).unsqueeze(1).float()
        target_t = torch.from_numpy(target_patches).unsqueeze(1).float()
    elif input_patches.ndim == 5:
        # (N, D, H, W, C) → multi-channel, permute to (N, C, D, H, W)
        n_channels = input_patches.shape[-1]
        spatial_shape = input_patches.shape[1:4]
        input_t  = torch.from_numpy(input_patches).permute(0, 4, 1, 2, 3).float()
        target_t = torch.from_numpy(target_patches).permute(0, 4, 1, 2, 3).float()
    else:
        raise ValueError(f"Expected 4D or 5D patches, got {input_patches.ndim}D")

    print(f"Detected {n_channels} channel(s), spatial shape {spatial_shape}")

    min_spatial = min(spatial_shape)
    num_pools   = max(1, min(5, int(np.floor(np.log2(min_spatial)))))
    print(f"Auto-selected number of pooling levels: {num_pools} (min side={min_spatial})")

    # Optional config-driven generator width (fallback keeps backward compatibility).
    try:
        import new_config as cfg  # available when training is launched from src workflow
        base_channels = int(getattr(cfg, 'vec_unet_base_channels', 16))
    except Exception:
        base_channels = 16
    print(f"Generator base_channels: {base_channels}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_t, target_t),
        batch_size=batch_size, shuffle=True)
    n_samples = input_t.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net_G = nnmodel.UNet3D(
        n_classes=n_channels, in_channels=n_channels,
        trilinear=True, base_channels=base_channels, num_pools=num_pools,
    ).to(device)

    disc_in_channels = 2 * n_channels  # condition + prediction
    sn_impl = 'native'
    if use_multiscale_disc:
        net_D = nnmodel.MultiScaleDiscriminator3D(
            in_channels=disc_in_channels,
            base_channels=disc_base_channels,
            n_layers_coarse=n_disc_layers,
            use_spectral_norm=use_spectral_norm,
            spectral_norm_impl=sn_impl,
        ).to(device)
    else:
        net_D = nnmodel.PatchDiscriminator3D(
            in_channels=disc_in_channels,
            base_channels=disc_base_channels,
            n_layers=n_disc_layers,
            use_spectral_norm=use_spectral_norm,
            spectral_norm_impl=sn_impl,
        ).to(device)

    total_G = sum(p.numel() for p in net_G.parameters())
    total_D = sum(p.numel() for p in net_D.parameters())
    print(f"Generator parameters: {total_G:,}")
    print(f"Discriminator parameters: {total_D:,}")
    print(f"n_channels={n_channels}, lambda_pixel={lambda_pixel}, lambda_fm={lambda_fm}, "
          f"n_disc_layers={n_disc_layers}, "
          f"d_update_interval={d_update_interval}, "
          f"use_spectral_norm={use_spectral_norm}, sn_impl={sn_impl}, batch_size={batch_size}, "
            f"use_multiscale_disc={use_multiscale_disc}, disc_base_channels={disc_base_channels}")

    opt_G = torch.optim.AdamW(net_G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.AdamW(net_D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=epochs, eta_min=lr_g * 0.01)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=epochs, eta_min=lr_d * 0.01)

    criterion_pixel = nn.MSELoss()

    torch.cuda.empty_cache()
    start_epoch = 0

    def _d_forward(ic, res):
        """Returns list of logit tensors (one per sub-discriminator)."""
        out = net_D(ic, res)
        return out if isinstance(out, list) else [out]

    def _d_forward_feats(ic, res):
        """Returns (list_of_logits, flat_list_of_feature_tensors)."""
        out = net_D(ic, res, return_features=True)
        if isinstance(out, list):
            logits = [item[0] for item in out]
            feats  = [f for item in out for f in item[1]]
        else:
            logits = [out[0]]
            feats  = out[1]
        return logits, feats

    def _gradient_penalty(cond_x, real_y, fake_y):
        bsz = real_y.size(0)
        alpha = torch.rand(bsz, 1, 1, 1, 1, device=device)
        interp = (alpha * real_y + (1.0 - alpha) * fake_y).requires_grad_(True)
        gp_terms = []
        for pred_interp in _d_forward(cond_x, interp):
            grad_outputs = torch.ones_like(pred_interp, device=device)
            gradients = torch.autograd.grad(
                outputs=pred_interp, inputs=interp,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True, only_inputs=True,
            )[0]
            gradients = gradients.view(bsz, -1)
            gp_terms.append(((gradients.norm(2, dim=1) - 1.0) ** 2).mean())
        return torch.stack(gp_terms).mean()

    # ── Auto-resume ────────────────────────────────────────────────────────
    if overwrite and resume_checkpoint == 'auto':
        print("overwrite=True, skipping checkpoints and starting from scratch.")
        resume_checkpoint = None

    if resume_checkpoint == 'auto':
        ckpt_dir    = os.path.dirname(save_file_name) or '.'
        ckpt_prefix = os.path.basename(save_file_name)
        candidates  = [f for f in os.listdir(ckpt_dir)
                       if f.startswith(ckpt_prefix) and f.endswith('.ckpt')]
        if candidates:
            candidates.sort(key=lambda f: int(f.rsplit('-e', 1)[-1].replace('.ckpt', '')))
            resume_checkpoint = os.path.join(ckpt_dir, candidates[-1])
        else:
            print("No checkpoint found, starting from scratch.")
            resume_checkpoint = None

    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        net_G.load_state_dict(ckpt['net_G_state'])
        try:
            net_D.load_state_dict(ckpt['net_D_state'])
            opt_G.load_state_dict(ckpt['optimizer_G_state'])
            opt_D.load_state_dict(ckpt['optimizer_D_state'])
            if 'scheduler_G_state' in ckpt:
                scheduler_G.load_state_dict(ckpt['scheduler_G_state'])
                scheduler_D.load_state_dict(ckpt['scheduler_D_state'])
        except RuntimeError as e:
            print(f"  Warning: checkpoint incompatible; continuing with fresh D/optimizer.\n  ({e})")
        start_epoch = ckpt['epoch']
        print(f"Completed {start_epoch} epochs, continue training to epoch {epochs}.")

    # ── Training loop ──────────────────────────────────────────────────────
    train_start_time = time.perf_counter()
    sum_epoch_seconds = 0.0

    for epoch in range(start_epoch, epochs):
        print(f'Starting epoch {epoch+1}/{epochs}...')
        epoch_start_time = time.perf_counter()
        net_G.train(); net_D.train()
        sum_G = sum_D = 0.0

        epoch_bar = tqdm(
            loader, total=len(loader),
            desc=f'Epoch {epoch+1}/{epochs}',
            unit='batch', dynamic_ncols=True, leave=True,
        )

        for batch_idx, (xb, yb) in enumerate(epoch_bar, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            # Per-patch mean subtraction (over spatial dims, keep channel)
            xb_dm = (xb - xb.mean(dim=[2, 3, 4], keepdim=True)).contiguous()
            yb_dm = (yb - yb.mean(dim=[2, 3, 4], keepdim=True)).contiguous()

            fake_dm = net_G(xb_dm)
            fake_dm = (fake_dm - fake_dm.mean(dim=[2, 3, 4], keepdim=True)).contiguous()

            def _switch_sn_impl_and_reset_d(err):
                nonlocal net_D, opt_D, scheduler_D, sn_impl
                if use_spectral_norm and 'CUBLAS_STATUS_INVALID_VALUE' in str(err):
                    if sn_impl == 'cpu_power_iter':
                        return False
                    print('Switching discriminator SN to cpu_power_iter...')
                    sn_impl = 'cpu_power_iter'
                    if use_multiscale_disc:
                        net_D = nnmodel.MultiScaleDiscriminator3D(
                            in_channels=disc_in_channels,
                            base_channels=disc_base_channels,
                            n_layers_coarse=n_disc_layers,
                            use_spectral_norm=True, spectral_norm_impl=sn_impl,
                        ).to(device)
                    else:
                        net_D = nnmodel.PatchDiscriminator3D(
                            in_channels=disc_in_channels,
                            base_channels=disc_base_channels, n_layers=n_disc_layers,
                            use_spectral_norm=True, spectral_norm_impl=sn_impl,
                        ).to(device)
                    opt_D = torch.optim.AdamW(net_D.parameters(), lr=lr_d, betas=(0.5, 0.999))
                    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt_D, T_max=epochs, eta_min=lr_d * 0.01)
                    torch.cuda.empty_cache()
                    return True
                return False

            # ── Train Discriminator ───────────────────────────────────────
            if batch_idx % d_update_interval == 0:
                opt_D.zero_grad()
                try:
                    pred_real_list = _d_forward(xb_dm, yb_dm)
                    pred_fake_list = _d_forward(xb_dm, fake_dm.detach())
                except RuntimeError as e:
                    if _switch_sn_impl_and_reset_d(e):
                        pred_real_list = _d_forward(xb_dm, yb_dm)
                        pred_fake_list = _d_forward(xb_dm, fake_dm.detach())
                    else:
                        raise
                gp = _gradient_penalty(xb_dm, yb_dm, fake_dm.detach())
                n_d = len(pred_real_list)
                loss_D = (
                    sum(pf.mean() - pr.mean()
                        for pf, pr in zip(pred_fake_list, pred_real_list)) / n_d
                    + lambda_gp * gp
                )
                try:
                    loss_D.backward()
                    opt_D.step()
                except RuntimeError as e:
                    if _switch_sn_impl_and_reset_d(e):
                        opt_D.zero_grad()
                        pred_real_list = _d_forward(xb_dm, yb_dm)
                        pred_fake_list = _d_forward(xb_dm, fake_dm.detach())
                        gp = _gradient_penalty(xb_dm, yb_dm, fake_dm.detach())
                        n_d = len(pred_real_list)
                        loss_D = (
                            sum(pf.mean() - pr.mean()
                                for pf, pr in zip(pred_fake_list, pred_real_list)) / n_d
                            + lambda_gp * gp
                        )
                        loss_D.backward()
                        opt_D.step()
                    else:
                        raise
            else:
                loss_D = torch.tensor(0.0, device=device)
                gp = torch.tensor(0.0, device=device)

            # ── Train Generator ───────────────────────────────────────────
            opt_G.zero_grad()
            if lambda_fm > 0:
                try:
                    with torch.no_grad():
                        _, feats_real = _d_forward_feats(xb_dm, yb_dm)
                    pred_fake_list, feats_fake = _d_forward_feats(xb_dm, fake_dm)
                except RuntimeError as e:
                    if _switch_sn_impl_and_reset_d(e):
                        with torch.no_grad():
                            _, feats_real = _d_forward_feats(xb_dm, yb_dm)
                        pred_fake_list, feats_fake = _d_forward_feats(xb_dm, fake_dm)
                    else:
                        raise
                loss_fm = torch.stack([F.mse_loss(ff, fr)
                                       for ff, fr in zip(feats_fake, feats_real)]).mean()
            else:
                try:
                    pred_fake_list = _d_forward(xb_dm, fake_dm)
                except RuntimeError as e:
                    if _switch_sn_impl_and_reset_d(e):
                        pred_fake_list = _d_forward(xb_dm, fake_dm)
                    else:
                        raise
                loss_fm = torch.tensor(0.0, device=device)

            n_d = len(pred_fake_list)
            loss_adv   = sum(-pf.mean() for pf in pred_fake_list) / n_d
            loss_pixel = criterion_pixel(fake_dm, yb_dm)
            loss_G = (loss_adv + lambda_pixel * loss_pixel +
                      lambda_fm * loss_fm)
            try:
                loss_G.backward()
                opt_G.step()
            except RuntimeError as e:
                if _switch_sn_impl_and_reset_d(e):
                    opt_G.zero_grad()
                    if lambda_fm > 0:
                        with torch.no_grad():
                            _, feats_real = _d_forward_feats(xb_dm, yb_dm)
                        pred_fake_list, feats_fake = _d_forward_feats(xb_dm, fake_dm)
                        loss_fm = torch.stack([F.mse_loss(ff, fr)
                                               for ff, fr in zip(feats_fake, feats_real)]).mean()
                    else:
                        pred_fake_list = _d_forward(xb_dm, fake_dm)
                        loss_fm = torch.tensor(0.0, device=device)
                    n_d = len(pred_fake_list)
                    loss_adv   = sum(-pf.mean() for pf in pred_fake_list) / n_d
                    loss_pixel = criterion_pixel(fake_dm, yb_dm)
                    loss_G = (loss_adv + lambda_pixel * loss_pixel +
                              lambda_fm * loss_fm)
                    loss_G.backward()
                    opt_G.step()
                else:
                    raise

            sum_G += loss_G.item() * xb.size(0)
            sum_D += loss_D.item() * xb.size(0)

            batch_elapsed = time.perf_counter() - epoch_start_time
            avg_batch_seconds = batch_elapsed / batch_idx
            remaining_batches = len(loader) - batch_idx
            epoch_eta_dt = datetime.now() + timedelta(seconds=avg_batch_seconds * remaining_batches)
            epoch_bar.set_postfix(
                G=f'{loss_G.item():.4f}',
                D=f'{loss_D.item():.4f}',
                px=f'{loss_pixel.item():.4f}',
                fm=f'{loss_fm.item():.4f}',
                gp=f'{gp.item():.4f}',
                eta=epoch_eta_dt.strftime('%H:%M:%S'),
            )

        epoch_bar.close()
        scheduler_G.step()
        scheduler_D.step()

        n = n_samples
        epoch_seconds = time.perf_counter() - epoch_start_time
        sum_epoch_seconds += epoch_seconds
        done_epochs = epoch - start_epoch + 1
        avg_epoch_seconds = sum_epoch_seconds / done_epochs
        remaining_epochs = epochs - (epoch + 1)
        train_eta_dt = datetime.now() + timedelta(seconds=avg_epoch_seconds * remaining_epochs)
        print(
            f"Epoch {epoch+1}/{epochs},  loss_G={sum_G/n:.6f},  loss_D={sum_D/n:.6f},  "
            f"lr_G={scheduler_G.get_last_lr()[0]:.2e},  lr_D={scheduler_D.get_last_lr()[0]:.2e},  "
            f"epoch_time={epoch_seconds:.1f}s,  avg_epoch_time={avg_epoch_seconds:.1f}s,  "
            f"train_eta={train_eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = f'{save_file_name}-e{epoch+1}.ckpt'
            torch.save({
                'epoch':             epoch + 1,
                'net_G_state':       net_G.state_dict(),
                'net_D_state':       net_D.state_dict(),
                'optimizer_G_state': opt_G.state_dict(),
                'optimizer_D_state': opt_D.state_dict(),
                'scheduler_G_state': scheduler_G.state_dict(),
                'scheduler_D_state': scheduler_D.state_dict(),
                'loss_G':            sum_G / n,
                'loss_D':            sum_D / n,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if epochs >= 100 and (epoch + 1) % 100 == 0:
            torch.save(net_G.state_dict(), f'{save_file_name}-e{epoch+1}.pth')

    torch.save(net_G.state_dict(), f'{save_file_name}-e{epochs}.pth')
    total_train_seconds = time.perf_counter() - train_start_time
    print(f"Total training time: {total_train_seconds:.1f}s")
    print(f"GAN training complete, generator saved to: {save_file_name}-e{epochs}.pth")
