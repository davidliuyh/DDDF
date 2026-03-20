import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.model as nnmodel



def train_unet(training_data_path, save_file_name, batch_size=64, epochs=10, learning_rate=1e-3, weight_decay=1e-5,
               checkpoint_interval=100, resume_checkpoint=None, overwrite=False):
    """
    Train a UNet3D model with checkpoint resume support.

    Args:
        checkpoint_interval: Save one checkpoint every this many epochs.
        resume_checkpoint: Checkpoint path (.ckpt); if provided, resume from it.
                           If set to 'auto', find the latest matching checkpoint.
        overwrite: If False and final model file exists, skip training.
    """
    
    final_path = f'{save_file_name}-e{epochs}.pth'
    if not overwrite and os.path.exists(final_path):
        print(f"Model already exists, skipping training: {final_path}")
        return
    input_patches = np.load(training_data_path)['input_patches']
    target_patches = np.load(training_data_path)['target_patches']

    spatial_shape = input_patches.shape[1:4]
    min_spatial = min(spatial_shape)
    max_pools_allowed = int(np.floor(np.log2(min_spatial)))
    auto_num_pools = max(1, min(5, max_pools_allowed))

    print(f"Training patch spatial shape: {spatial_shape}")
    print(f"Auto-selected number of pooling levels: {auto_num_pools} (min side={min_spatial})")

    # Convert to PyTorch tensors and build dataset.
    input_tensor = torch.from_numpy(input_patches).unsqueeze(1).float()
    target_tensor = torch.from_numpy(target_patches).unsqueeze(1).float()

    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    # Rebuild model with one output channel for regression.
    model = nnmodel.UNet3D(
        n_classes=1,
        trilinear=True,
        base_channels=16,
        num_pools=auto_num_pools
    )

    # Parameter counts.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    torch.cuda.empty_cache()  # Clear cached GPU memory from earlier runs if any.

    start_epoch = 0

    # Auto-find latest checkpoint.
    if resume_checkpoint == 'auto':
        ckpt_dir = os.path.dirname(save_file_name) or '.'
        ckpt_prefix = os.path.basename(save_file_name)
        candidates = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith(ckpt_prefix) and f.endswith('.ckpt')
        ]
        if candidates:
            candidates.sort(key=lambda f: int(f.rsplit('-e', 1)[-1].replace('.ckpt', '')))
            resume_checkpoint = os.path.join(ckpt_dir, candidates[-1])
        else:
            print("No checkpoint found, starting from scratch.")
            resume_checkpoint = None

    # Resume from checkpoint.
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # Number of completed epochs.
        print(f"Completed {start_epoch} epochs, continue training to epoch {epochs}.")

    for epoch in range(start_epoch, epochs):
        print(f'Starting epoch {epoch+1}/{epochs}...')
        model.train()
        running_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            # Subtract per-patch mean before computing loss so the network
            # cannot learn a mean-field bias correlated with the IC large-scale
            # modes, which would otherwise leak spurious large-scale power.
            out_dm = out - out.mean(dim=[2, 3, 4], keepdim=True)
            yb_dm  = yb  - yb.mean(dim=[2, 3, 4], keepdim=True)
            loss = criterion(out_dm, yb_dm)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

            if batch_idx % 10 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] batch {batch_idx+1}/{len(loader)}, "
                      f"batch_loss={loss.item():.6f}")

        epoch_loss = running_loss / len(dataset)
        print(f'Epoch {epoch+1}/{epochs}, loss={epoch_loss:.6f}, lr={learning_rate:.2e}')

        # Save checkpoint periodically (full training state).
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = f'{save_file_name}-e{epoch+1}.ckpt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Save trained model to file.
        if epochs >= 100 and (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{save_file_name}-e{epoch+1}.pth')

    # Save trained model to file.
    torch.save(model.state_dict(), f'{save_file_name}-e{epochs}.pth')


def train_gan(training_data_path, save_file_name, batch_size=64, epochs=50,
              lr_g=1e-4, lr_d=1e-4, lambda_pixel=10.0, n_disc_layers=3,
              lambda_fm=10.0, d_update_interval=2, use_spectral_norm=True,
              checkpoint_interval=10, resume_checkpoint=None, overwrite=False):
    """Conditional GAN (LSGAN / Pix2Pix3D) training.

    Generator  : UNet3D (same architecture as train_unet)
    Discriminator: PatchDiscriminator3D conditioned on IC

    Loss_D = 0.5 * MSE(D(IC,real) - 1)² + 0.5 * MSE(D(IC,fake))²
    Loss_G = 0.5 * MSE(D(IC,fake) - 1)²              [adversarial]
           + lambda_pixel * MSE(fake, real)            [pixel]
           + lambda_fm    * FeatureMatching(fake,real)  [FM]

    Anti-collapse improvements:
    - use_spectral_norm : apply spectral normalisation to every D Conv3d,
                          limiting the Lipschitz constant and slowing D.
    - d_update_interval : update D only every N batches so G has more
                          gradient steps relative to D.
    - lambda_fm         : feature-matching loss that aligns D's
                          intermediate activations for fake vs. real,
                          giving G a richer, more stable training signal.
    - CosineAnnealingLR : smoothly decay both lr_g and lr_d to 1 % of
                          their initial values over the full training run.

    Checkpoint (.ckpt) stores G + D + both optimizers + both schedulers.
    Final .pth stores only the generator state dict → compatible with
    the existing apply_unet_to_field inference path.
    """
    final_path = f'{save_file_name}-e{epochs}.pth'
    if not overwrite and os.path.exists(final_path):
        print(f"Model already exists, skipping training: {final_path}")
        return

    data = np.load(training_data_path)
    input_patches  = data['input_patches']
    target_patches = data['target_patches']

    spatial_shape = input_patches.shape[1:4]
    min_spatial   = min(spatial_shape)
    num_pools     = max(1, min(5, int(np.floor(np.log2(min_spatial)))))
    print(f"Training patch spatial shape: {spatial_shape}")
    print(f"Auto-selected number of pooling levels: {num_pools} (min side={min_spatial})")

    input_t  = torch.from_numpy(input_patches).unsqueeze(1).float()
    target_t = torch.from_numpy(target_patches).unsqueeze(1).float()
    loader   = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_t, target_t),
        batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net_G = nnmodel.UNet3D(n_classes=1, trilinear=True,
                            base_channels=16, num_pools=num_pools).to(device)
    net_D = nnmodel.PatchDiscriminator3D(in_channels=2,
                                          base_channels=32,
                                          n_layers=n_disc_layers,
                                          use_spectral_norm=use_spectral_norm).to(device)

    total_G = sum(p.numel() for p in net_G.parameters())
    total_D = sum(p.numel() for p in net_D.parameters())
    print(f"Generator parameters: {total_G:,}")
    print(f"Discriminator parameters: {total_D:,}")
    print(f"lambda_pixel={lambda_pixel}, lambda_fm={lambda_fm}, "
          f"n_disc_layers={n_disc_layers}, d_update_interval={d_update_interval}, "
          f"use_spectral_norm={use_spectral_norm}")

    opt_G = torch.optim.AdamW(net_G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.AdamW(net_D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Cosine annealing: decay lr to 1% of initial over full training run
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=epochs, eta_min=lr_g * 0.01)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=epochs, eta_min=lr_d * 0.01)

    criterion_GAN   = nn.MSELoss()           # LSGAN
    criterion_pixel = nn.MSELoss()

    torch.cuda.empty_cache()
    start_epoch = 0

    # ── Auto-resume ────────────────────────────────────────────────────────────
    # overwrite=True means "retrain from scratch" — skip any existing checkpoint.
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
            print(f"  Warning: discriminator checkpoint is incompatible with current architecture; continuing with randomly initialized D/optimizer.\n  ({e})")
        start_epoch = ckpt['epoch']
        print(f"Completed {start_epoch} epochs, continue training to epoch {epochs}.")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        print(f'Starting epoch {epoch+1}/{epochs}...')
        net_G.train(); net_D.train()
        sum_G = sum_D = 0.0

        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)
            # per-patch mean subtraction (consistent with train_unet)
            xb_dm = (xb - xb.mean(dim=[2, 3, 4], keepdim=True)).contiguous()
            yb_dm = (yb - yb.mean(dim=[2, 3, 4], keepdim=True)).contiguous()

            fake_dm = net_G(xb_dm)
            fake_dm = (fake_dm - fake_dm.mean(dim=[2, 3, 4], keepdim=True)).contiguous()

            def _disable_sn_and_reset_d(err):
                nonlocal net_D, opt_D, scheduler_D, use_spectral_norm
                if use_spectral_norm and 'CUBLAS_STATUS_INVALID_VALUE' in str(err):
                    print('Detected CUBLAS_STATUS_INVALID_VALUE; disabling discriminator spectral norm and rebuilding D automatically.')
                    use_spectral_norm = False
                    net_D = nnmodel.PatchDiscriminator3D(
                        in_channels=2,
                        base_channels=32,
                        n_layers=n_disc_layers,
                        use_spectral_norm=False,
                    ).to(device)
                    opt_D = torch.optim.AdamW(net_D.parameters(), lr=lr_d, betas=(0.5, 0.999))
                    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt_D, T_max=epochs, eta_min=lr_d * 0.01)
                    torch.cuda.empty_cache()
                    return True
                return False

            # ── Train Discriminator (every d_update_interval batches) ─────────
            if batch_idx % d_update_interval == 0:
                opt_D.zero_grad()
                try:
                    pred_real = net_D(xb_dm, yb_dm)
                    pred_fake = net_D(xb_dm, fake_dm.detach())
                except RuntimeError as e:
                    if _disable_sn_and_reset_d(e):
                        pred_real = net_D(xb_dm, yb_dm)
                        pred_fake = net_D(xb_dm, fake_dm.detach())
                    else:
                        raise
                ones  = torch.ones_like(pred_real)
                zeros = torch.zeros_like(pred_fake)
                loss_D = 0.5 * (criterion_GAN(pred_real, ones) +
                                criterion_GAN(pred_fake, zeros))
                loss_D.backward()
                opt_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # ── Train Generator ───────────────────────────────────────────────
            opt_G.zero_grad()
            if lambda_fm > 0:
                # Get real intermediate features without gradients
                try:
                    with torch.no_grad():
                        _, feats_real = net_D(xb_dm, yb_dm, return_features=True)
                    pred_fake_G, feats_fake = net_D(xb_dm, fake_dm, return_features=True)
                except RuntimeError as e:
                    if _disable_sn_and_reset_d(e):
                        with torch.no_grad():
                            _, feats_real = net_D(xb_dm, yb_dm, return_features=True)
                        pred_fake_G, feats_fake = net_D(xb_dm, fake_dm, return_features=True)
                    else:
                        raise
                loss_fm = torch.stack([F.mse_loss(ff, fr)
                                       for ff, fr in zip(feats_fake, feats_real)]).mean()
            else:
                try:
                    pred_fake_G = net_D(xb_dm, fake_dm)
                except RuntimeError as e:
                    if _disable_sn_and_reset_d(e):
                        pred_fake_G = net_D(xb_dm, fake_dm)
                    else:
                        raise
                loss_fm = torch.tensor(0.0, device=device)

            loss_adv   = 0.5 * criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_pixel = criterion_pixel(fake_dm, yb_dm)
            loss_G = loss_adv + lambda_pixel * loss_pixel + lambda_fm * loss_fm
            loss_G.backward()
            opt_G.step()

            sum_G += loss_G.item() * xb.size(0)
            sum_D += loss_D.item() * xb.size(0)

            if batch_idx % 10 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] batch {batch_idx+1}/{len(loader)}, "
                      f"G={loss_G.item():.4f} "
                      f"(adv={loss_adv.item():.4f}, px={loss_pixel.item():.4f}, "
                      f"fm={loss_fm.item():.4f}), "
                      f"D={loss_D.item():.4f}")

        scheduler_G.step()
        scheduler_D.step()

        n = len(input_patches)
        print(f"Epoch {epoch+1}/{epochs},  loss_G={sum_G/n:.6f},  loss_D={sum_D/n:.6f},  "
              f"lr_G={scheduler_G.get_last_lr()[0]:.2e},  lr_D={scheduler_D.get_last_lr()[0]:.2e}")

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

    # Save generator-only .pth for inference
    torch.save(net_G.state_dict(), f'{save_file_name}-e{epochs}.pth')
    print(f"GAN training complete, generator saved to: {save_file_name}-e{epochs}.pth")