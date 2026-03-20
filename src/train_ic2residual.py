import sys, os
from pathlib import Path

# Make imports robust to notebook working directory.
# This notebook lives in .../DDDF/src, so we add both src and repo root.
nb_dir = Path.cwd()
if not (nb_dir / 'config.py').exists() and (nb_dir / 'src' / 'config.py').exists():
    nb_dir = nb_dir / 'src'
repo_root = nb_dir.parent

for p in (str(nb_dir), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import importlib
import MAS_library as MASL
import Pk_library as PKL
import matplotlib.pyplot as plt
import torch
import dddf
from model.gen_patches import extract_patches
import model.model as nnmodel
import model.train_model as train_module

import config as cfg
import pipeline as _pipeline_module
from pipeline import (load_snapshot_pair, compute_target_psi_div,
                      psi_div_to_delta, free_gpu_memory, compute_best_fit,
                      highpass_field)
from inference import apply_model_to_field

# ── User settings ─────────────────────────────────────────────────
N_p        = cfg.N_p
data_dir   = cfg.data_dir
model_dir  = cfg.model_dir
# ──────────────────────────────────────────────────────────────────

# ── Derived constants ──────────────────────────────────────────────
L         = cfg.L
boxsize   = cfg.boxsize
MAS       = cfg.MAS
threads   = cfg.threads
grid_size = N_p

dl        = dddf.DDDF(cfg.init_redshift(N_p), cfg.final_snapshot_z, cfg.Omega_m, threads)
veck_main = dl.Veck(dl, N_p, boxsize, padding=0)

print(f'hostname: {os.popen("hostname").read().strip()}')
print(f'N_p={N_p}, boxsize={boxsize}, data_dir={data_dir}, model_dir={model_dir}')

# ── Section settings ──────────────────────────────────────────────
realization = 0
coef_file   = cfg.best_fit_avg_coef_path(data_dir, cfg.L, N_p)  # None -> refit from scratch
overwrite   = False
# ──────────────────────────────────────────────────────────────────

snapshot_info = load_snapshot_pair(
    dl,
    cfg.snapshot_paths(realization, N_p),
    cfg.snapshot_format(N_p),
    boxsize, grid_size,
)

init_delta     = snapshot_info[0]['delta']
target_psi_div = compute_target_psi_div(dl, snapshot_info, N_p, boxsize, veck_main)

# psi_div_1  = dl.div_psi_1(init_delta)
# psi_div_2  = dl.div_psi_2(init_delta, veck_main)
# ZA_disp    = dl.disp_from_psi_div(psi_div_1, veck_main, N_p)
# LPT2_disp  = dl.disp_from_psi_div(psi_div_1 + psi_div_2, veck_main, N_p)

best_fit_psi_div, best_fit_delta, target_delta = compute_best_fit(
    dl, init_delta, target_psi_div,
    snapshot_info[0]['pos'], snapshot_info[1]['delta'],
    veck_main, N_p, boxsize, MAS,
    realization, data_dir, L,
    coef_file=coef_file,
    overwrite=overwrite,
)

patch_size = cfg.patch_size
padding    = cfg.padding
overlap    = cfg.overlap
rotate     = cfg.rotate

# ── Section settings ──────────────────────────────────────────────
train_realizations = cfg.train_realizations   # e.g. list(range(16)); edit in config.py
coef_file   = cfg.best_fit_avg_coef_path(data_dir, cfg.L, N_p)  # None -> refit from scratch
# ──────────────────────────────────────────────────────────────────

train_path = cfg.training_data_path(train_realizations, patch_size, padding, overlap, rotate, N_p, data_dir=data_dir)

if os.path.exists(train_path):
    cached = np.load(train_path)
    input_patches  = cached['input_patches']
    target_patches = cached['target_patches']
    print(f'Combined training data already exists, loaded: {train_path}')
    print(f'{input_patches.shape[0]} patch pairs, shape {input_patches.shape[1:]}')
else:
    all_input_patches  = []
    all_target_patches = []

    for r in train_realizations:
        train_path_r = cfg.training_data_path(r, patch_size, padding, overlap, rotate, N_p, data_dir=data_dir)
        if os.path.exists(train_path_r):
            cached = np.load(train_path_r)
            inp_r = cached['input_patches']
            tgt_r = cached['target_patches']
            print(f'  [r{r}] training file exists, loaded: {train_path_r}')
        else:
            snap_info_r = load_snapshot_pair(
                dl,
                cfg.snapshot_paths(r, N_p),
                cfg.snapshot_format(N_p),
                boxsize, grid_size,
            )
            tgt_psi_div_r = compute_target_psi_div(dl, snap_info_r, N_p, boxsize, veck_main)
            bf_psi_div_r, _, _ = compute_best_fit(
                dl, snap_info_r[0]['delta'], tgt_psi_div_r,
                snap_info_r[0]['pos'], snap_info_r[1]['delta'],
                veck_main, N_p, boxsize, MAS,
                r, data_dir, L,
                coef_file=coef_file,
                overwrite=False,
            )
            residual_r = tgt_psi_div_r - bf_psi_div_r
            print(f'  [r{r}] residual χ² = {np.mean(residual_r**2):.6e}')

            inp_r = extract_patches(snap_info_r[0]['delta'], patch_size, padding, overlap, rotate)
            tgt_r = extract_patches(residual_r,              patch_size, padding, overlap, rotate)

            os.makedirs(os.path.dirname(train_path_r), exist_ok=True)
            np.savez(
                train_path_r,
                input_patches=inp_r,
                target_patches=tgt_r,
                patch_size=patch_size,
                padding=padding,
                overlap=overlap,
                realization=r,
            )
            print(f'  [r{r}] saved: {train_path_r}')
            del snap_info_r, tgt_psi_div_r, bf_psi_div_r, residual_r

        all_input_patches.append(inp_r)
        all_target_patches.append(tgt_r)

    input_patches  = np.concatenate(all_input_patches,  axis=0)
    target_patches = np.concatenate(all_target_patches, axis=0)
    assert input_patches.shape == target_patches.shape
    print(f'{input_patches.shape[0]} patch pairs from {len(train_realizations)} realization(s), shape {input_patches.shape[1:]}')

    np.savez(train_path, input_patches=input_patches, target_patches=target_patches,
             patch_size=patch_size, padding=padding, overlap=overlap)
    print(f'combined training data saved: {train_path}')

free_gpu_memory()

free_gpu_memory()

importlib.reload(cfg)
importlib.reload(nnmodel)          # Must reload model before reloading train_module.
importlib.reload(train_module)

# ── Section settings ──────────────────────────────────────────────
overwrite_train = False   # True -> force retrain; False -> skip if model exists.
train_mode = cfg.train_mode  # 'unet' | 'gan'  <- switch in config.py
# ──────────────────────────────────────────────────────────────────


if train_mode == 'gan':
    model_name = cfg.gan_model_name(train_realizations, patch_size, padding, rotate, N_p, model_dir=model_dir)
    train_module.train_gan(
        training_data_path=train_path,
        save_file_name=model_name,

        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr_g=cfg.gan_lr_g,
        lr_d=cfg.gan_lr_d,
        lambda_pixel=cfg.gan_lambda_pixel,
        n_disc_layers=cfg.gan_n_disc_layers,
        lambda_fm=cfg.gan_lambda_fm,
        d_update_interval=cfg.gan_d_update_interval,
        use_spectral_norm=cfg.gan_use_spectral_norm,
        checkpoint_interval=cfg.checkpoint_interval,
        resume_checkpoint='auto',
        overwrite=overwrite_train,
    )
else:
    model_name = cfg.unet_model_name(train_realizations, patch_size, padding, rotate, N_p, model_dir=model_dir)
    train_module.train_unet(
        training_data_path=train_path,
        save_file_name=model_name,

        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        checkpoint_interval=cfg.checkpoint_interval,
        resume_checkpoint='auto',
        overwrite=overwrite_train,
    )

importlib.reload(cfg)
importlib.reload(_pipeline_module)
from pipeline import (load_snapshot_pair, compute_target_psi_div,
                      psi_div_to_delta, free_gpu_memory, compute_best_fit,
                      highpass_field)

infer_patch_size = cfg.infer_patch_size
infer_padding    = cfg.infer_padding
infer_overlap    = cfg.infer_overlap
infer_epochs     = cfg.infer_epochs

# ── Section settings ──────────────────────────────────────────────
infer_checkpoint         = cfg.infer_checkpoint      # None → auto-derive; str → explicit .pth path
infer_train_realizations = cfg.train_realizations    # realizations this model was trained on (for chi²)
# ──────────────────────────────────────────────────────────────────

if infer_checkpoint is None:
    if cfg.train_mode == 'gan':
        _auto_model = cfg.gan_model_name(infer_train_realizations,
                                         cfg.patch_size, cfg.padding,
                                         cfg.rotate, N_p, model_dir=model_dir)
    else:
        _auto_model = cfg.unet_model_name(infer_train_realizations,
                                          cfg.patch_size, cfg.padding,
                                          cfg.rotate, N_p, model_dir=model_dir)
    checkpoint_path = f'{_auto_model}-e{infer_epochs}.pth'
else:
    checkpoint_path = infer_checkpoint

state_dict = torch.load(checkpoint_path, map_location='cpu')

num_pools = len({int(k.split('.')[1]) for k in state_dict if k.startswith('downs.')})
num_pools = max(num_pools, 1)

loaded_model = nnmodel.UNet3D(n_classes=1, trilinear=True,
                               base_channels=16, num_pools=num_pools)
loaded_model.load_state_dict(state_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)
print(f'Mode: {cfg.train_mode}. Model loaded to {device}. '
      f'Inference patch size = {infer_patch_size}, padding = {infer_padding}, overlap = {infer_overlap}')
print(f'Loaded: {checkpoint_path}  (pools={num_pools}, device={device})')

k_cut = 0.06   # h/Mpc — matches the tophat0.4 best-fit filter

residual_pred   = apply_model_to_field(init_delta, loaded_model,
                                      infer_patch_size, infer_padding, infer_overlap, device)
residual_pred   = highpass_field(residual_pred, k_cut, boxsize)
final_psi_div   = best_fit_psi_div + residual_pred

delta_nbody     = target_delta
delta_bestfit   = best_fit_delta
delta_final     = psi_div_to_delta(final_psi_div,  dl, snapshot_info[0]['pos'],
                                    veck_main, N_p, boxsize, MAS)
delta_recovered = psi_div_to_delta(target_psi_div, dl, snapshot_info[0]['pos'],
                                    veck_main, N_p, boxsize, MAS)
delta_residual  = delta_nbody - delta_final

labels = ['N-body', 'best-fit', 'best-fit + IC2RES', 'Recovered']
deltas = [delta_nbody, delta_bestfit, delta_final, delta_recovered]

# Overdensity slices (Residual replaces Recovered)
slice_labels = ['N-body', 'best-fit', 'best-fit + IC2RES', 'Residual']
slice_deltas = [delta_nbody, delta_bestfit, delta_final, delta_residual]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, d, lab in zip(axes.flat, slice_deltas, slice_labels):
    img = np.mean(d[:1], axis=0).T
    if lab == 'Residual':
        im = ax.imshow(img, cmap='RdBu_r', vmin=-2, vmax=2, origin='lower')
        plt.colorbar(im, ax=ax)
    else:
        ax.imshow(img, cmap='gray_r', vmin=-1, vmax=3, origin='lower')
    ax.set_title(lab)
plt.suptitle(f'[r{realization}] Overdensity slice z=0')
plt.tight_layout(); plt.show()

# Power spectra
Pks = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False) for d in deltas]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xscale('log')
ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
ax.set_ylabel(r'$P/P_{\rm N\text{-}body}$')
ax.set_xlim(0.01, 1.0); ax.set_ylim(0, 1.5)
ax.axvline(veck_main.Nyquist_freq, color='grey', ls='--', lw=1, alpha=0.7)
for pk, lab in zip(Pks, labels):
    ax.plot(Pks[0].k3D, pk.Pk[:, 0] / Pks[0].Pk[:, 0], label=lab)
ax.legend(); plt.title(f'[r{realization}] Matter power spectrum z=0'); plt.show()

# # Bispectrum
# k1, k2 = 0.2, 0.2
# theta   = np.linspace(0, np.pi, 25)
# Bks = [PKL.Bk(d, boxsize, k1, k2, theta, MAS=MAS, threads=8) for d in deltas]

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$B/B_{\rm N\text{-}body}$')
# ax.set_xlim(0, np.pi); ax.set_ylim(0, 1.5)
# ax.set_title(f'[r{realization}] Bispectrum z=0, k1={k1}, k2={k2}')
# for bk, lab in zip(Bks, labels):
#     ax.plot(theta, bk.B / Bks[0].B, label=lab)
# ax.legend(); plt.show()

# χ² vs N-body (k ≤ 0.3)
mask_k = Pks[0].k3D <= 0.3
for pk, lab in zip(Pks[1:], labels[1:]):
    ratio = pk.Pk[:, 0] / Pks[0].Pk[:, 0]
    print(f'[r{realization}] {lab} χ² (k≤0.3) = {np.mean((ratio[mask_k] - 1)**2):.6e}')

realization_test = 18

snapshot_info_test = load_snapshot_pair(
    dl,
    cfg.snapshot_paths(realization_test, N_p),
    cfg.snapshot_format(N_p),
    boxsize, grid_size,
)

init_delta_test     = snapshot_info_test[0]['delta']
target_psi_div_test = compute_target_psi_div(dl, snapshot_info_test, N_p, boxsize, veck_main)

best_fit_psi_div_test, best_fit_delta_test, target_delta_test = compute_best_fit(
    dl, init_delta_test, target_psi_div_test,
    snapshot_info_test[0]['pos'], snapshot_info_test[1]['delta'],
    veck_main, N_p, boxsize, MAS,
    realization_test, data_dir, L,
    coef_file=coef_file,
    overwrite=False,
)

residual_pred_test  = apply_model_to_field(
    init_delta_test, loaded_model, infer_patch_size, infer_padding, infer_overlap, device)
residual_pred_test  = highpass_field(residual_pred_test, k_cut, boxsize)
delta_final_test    = psi_div_to_delta(
    best_fit_psi_div_test + residual_pred_test,
    dl, snapshot_info_test[0]['pos'], veck_main, N_p, boxsize, MAS)
delta_recovered_test = psi_div_to_delta(
    target_psi_div_test, dl, snapshot_info_test[0]['pos'], veck_main, N_p, boxsize, MAS)
delta_residual_test  = target_delta_test - delta_final_test

labels_t = ['N-body', 'best-fit', 'best-fit + IC2RES', 'Recovered']
deltas_t = [target_delta_test, best_fit_delta_test, delta_final_test, delta_recovered_test]

# Overdensity slices (Residual replaces Recovered)
slice_labels_t = ['N-body', 'best-fit', 'best-fit + IC2RES', 'Residual']
slice_deltas_t = [target_delta_test, best_fit_delta_test, delta_final_test, delta_residual_test]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, d, lab in zip(axes.flat, slice_deltas_t, slice_labels_t):
    img = np.mean(d[:1], axis=0).T
    if lab == 'Residual':
        im = ax.imshow(img, cmap='RdBu_r', vmin=-2, vmax=2, origin='lower')
        plt.colorbar(im, ax=ax)
    else:
        ax.imshow(img, cmap='gray_r', vmin=-1, vmax=3, origin='lower')
    ax.set_title(lab)
plt.suptitle(f'[r{realization_test}] Overdensity slice z=0')
plt.tight_layout(); plt.show()

# Power spectra
Pks_t = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False) for d in deltas_t]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xscale('log')
ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
ax.set_ylabel(r'$P/P_{\rm N\text{-}body}$')
ax.set_xlim(0.01, 1.0); ax.set_ylim(0, 1.5)
ax.axvline(veck_main.Nyquist_freq, color='grey', ls='--', lw=1, alpha=0.7)
for pk, lab in zip(Pks_t, labels_t):
    ax.plot(Pks_t[0].k3D, pk.Pk[:, 0] / Pks_t[0].Pk[:, 0], label=lab)
ax.legend(); plt.title(f'[r{realization_test}] Matter power spectrum z=0'); plt.show()

# # Bispectrum
# k1, k2 = 0.2, 0.2
# theta   = np.linspace(0, np.pi, 25)
# Bks_t = [PKL.Bk(d, boxsize, k1, k2, theta, MAS=MAS, threads=8) for d in deltas_t]

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$B/B_{\rm N\text{-}body}$')
# ax.set_xlim(0, np.pi); ax.set_ylim(0, 1.5)
# ax.set_title(f'[r{realization_test}] Bispectrum z=0, k1={k1}, k2={k2}')
# for bk, lab in zip(Bks_t, labels_t):
#     ax.plot(theta, bk.B / Bks_t[0].B, label=lab)
# ax.legend(); plt.show()

# χ²
mask_k = Pks_t[0].k3D <= 0.3
for pk, lab in zip(Pks_t[1:], labels_t[1:]):
    ratio = pk.Pk[:, 0] / Pks_t[0].Pk[:, 0]
    print(f'[r{realization_test}] {lab} χ² (k≤0.3) = {np.mean((ratio[mask_k] - 1)**2):.6e}')

# 5. Checkpoint diagnostics: G/D vs epoch + improvement suggestions (GAN first)
from pathlib import Path
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── User settings ─────────────────────────────────────────────────
analysis_mode = cfg.train_mode            # 'gan' or 'unet'
analysis_realizations = cfg.train_realizations
analysis_model_dir = model_dir
# Optionally specify multiple model prefixes (without -eXX.ckpt) to compare runs.
analysis_prefixes = []
# Example:
# analysis_prefixes = [
#     cfg.gan_model_name(list(range(8)), cfg.patch_size, cfg.padding, cfg.rotate, N_p, analysis_model_dir),
#     cfg.gan_model_name(list(range(16)), cfg.patch_size, cfg.padding, cfg.rotate, N_p, analysis_model_dir),
# ]
# ──────────────────────────────────────────────────────────────────

if not analysis_prefixes:
    if analysis_mode == 'gan':
        analysis_prefixes = [cfg.gan_model_name(
            analysis_realizations, cfg.patch_size, cfg.padding, cfg.rotate, N_p, analysis_model_dir
        )]
    else:
        analysis_prefixes = [cfg.unet_model_name(
            analysis_realizations, cfg.patch_size, cfg.padding, cfg.rotate, N_p, analysis_model_dir
        )]

def _epoch_from_path(p):
    m = re.search(r'-e(\d+)\.ckpt$', str(p))
    return int(m.group(1)) if m else -1

def load_ckpt_curve(prefix):
    ckpts = sorted(glob.glob(f'{prefix}-e*.ckpt'), key=_epoch_from_path)
    records = []
    for p in ckpts:
        try:
            c = torch.load(p, map_location='cpu')
        except Exception as e:
            print(f'[WARN] Failed to read: {p} ({e})')
            continue

        rec = {'epoch': _epoch_from_path(p), 'path': p}
        if 'loss_G' in c:
            rec['loss_G'] = float(c.get('loss_G', np.nan))
            rec['loss_D'] = float(c.get('loss_D', np.nan))
            try:
                rec['lr_G'] = float(c['optimizer_G_state']['param_groups'][0]['lr'])
                rec['lr_D'] = float(c['optimizer_D_state']['param_groups'][0]['lr'])
            except Exception:
                rec['lr_G'] = np.nan
                rec['lr_D'] = np.nan
        else:
            rec['loss'] = float(c.get('loss', np.nan))
            try:
                rec['lr'] = float(c['optimizer_state_dict']['param_groups'][0]['lr'])
            except Exception:
                rec['lr'] = np.nan
        records.append(rec)
    return records

all_runs = {}
for pref in analysis_prefixes:
    run_name = Path(pref).name
    recs = load_ckpt_curve(pref)
    if recs:
        all_runs[run_name] = recs
    else:
        print(f'[INFO] No checkpoints found: {pref}-e*.ckpt')

if not all_runs:
    print('No usable ckpt found. Please verify model path and file names.')
else:
    if analysis_mode == 'gan':
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        for name, recs in all_runs.items():
            epochs = np.array([r['epoch'] for r in recs], dtype=float)
            g = np.array([r.get('loss_G', np.nan) for r in recs], dtype=float)
            d = np.array([r.get('loss_D', np.nan) for r in recs], dtype=float)
            lr_g = np.array([r.get('lr_G', np.nan) for r in recs], dtype=float)
            lr_d = np.array([r.get('lr_D', np.nan) for r in recs], dtype=float)

            axes[0].plot(epochs, g, marker='o', lw=1.8, label=f'{name} | G')
            axes[0].plot(epochs, d, marker='x', lw=1.2, ls='--', label=f'{name} | D')
            axes[1].plot(epochs, lr_g, marker='o', lw=1.4, label=f'{name} | lr_G')
            axes[1].plot(epochs, lr_d, marker='x', lw=1.2, ls='--', label=f'{name} | lr_D')

        axes[0].set_ylabel('Loss')
        axes[0].set_title('GAN training diagnostics from checkpoints')
        axes[0].grid(alpha=0.25)
        axes[0].legend(ncol=2, fontsize=9)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning rate')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.25)
        axes[1].legend(ncol=2, fontsize=9)
        plt.tight_layout(); plt.show()

        # ── Heuristic analysis ───────────────────────────────────────
        print('\n[Automatic diagnostics and improvement suggestions]')
        for name, recs in all_runs.items():
            epochs = np.array([r['epoch'] for r in recs], dtype=float)
            g = np.array([r.get('loss_G', np.nan) for r in recs], dtype=float)
            d = np.array([r.get('loss_D', np.nan) for r in recs], dtype=float)

            ok = np.isfinite(g) & np.isfinite(d)
            if ok.sum() < 4:
                print(f'\n{name}: Too few valid points (<4). Consider reducing checkpoint_interval to 1 or 2 first, then inspect the curves.')
                continue

            epochs = epochs[ok]; g = g[ok]; d = d[ok]
            w = max(3, len(g) // 3)

            g_first, g_last = np.mean(g[:w]), np.mean(g[-w:])
            d_first, d_last = np.mean(d[:w]), np.mean(d[-w:])

            g_cv = np.std(g[-w:]) / (np.mean(g[-w:]) + 1e-12)
            d_cv = np.std(d[-w:]) / (np.mean(d[-w:]) + 1e-12)

            xw = np.arange(w, dtype=float)
            g_slope = np.polyfit(xw, g[-w:], 1)[0]
            d_slope = np.polyfit(xw, d[-w:], 1)[0]

            print(f'\n{name}:')
            print(f'  G: {g_first:.4e} -> {g_last:.4e}, D: {d_first:.4e} -> {d_last:.4e}')
            print(f'  Volatility coefficient: G={g_cv:.3f}, D={d_cv:.3f}; late-stage slope: dG/de={g_slope:.3e}, dD/de={d_slope:.3e}')

            # Balance check.
            if (d_last < 0.3 * g_last) and (g_slope >= 0):
                print('  Diagnosis: discriminator is too strong; generator is suppressed.')
                print('  Suggestion (training hyperparams): lower lr_D or raise lr_G (e.g., lr_D=0.5*lr_G), and increase d_update_interval from 2 to 3.')
                print('  Suggestion (model): reduce n_disc_layers or strengthen spectral_norm (if numerically stable).')
            elif (g_last < 0.4 * d_last) and (d_slope > 0):
                print('  Diagnosis: discriminator is too weak or under-trained.')
                print('  Suggestion (training hyperparams): moderately increase lr_D, or reduce d_update_interval back to 1-2.')
                print('  Suggestion (model): increase n_disc_layers (3->4) or increase D base_channels.')
            else:
                print('  Diagnosis: G/D are roughly balanced.')

            # Stability check.
            if (g_cv > 0.25) or (d_cv > 0.25):
                print('  Diagnosis: late-stage oscillation is relatively high.')
                print('  Suggestion (training stability): reduce batch_size or enable/strengthen spectral_norm; moderately increase lambda_fm.')

            # Convergence / underfitting check.
            if (g_last > 0.9 * g_first) and (d_last > 0.9 * d_first):
                print('  Diagnosis: insufficient convergence (possible underfitting).')
                print('  Suggestion (model): increase UNet capacity (base_channels or num_pools) and train for more epochs.')
                print('  Suggestion (data): expand train_realizations, enable rotate=True, and increase overlap for more sample diversity.')

        print('\n[General improvement priority]')
        print('1) Start from data: expand train_realizations (e.g., 8->16/32) and enable rotate.')
        print('2) Then tune losses: adjust lambda_fm vs lambda_pixel while preserving pixel fidelity.')
        print('3) Finally tune models: gradually adjust D depth and G capacity; avoid changing too many parameters at once.')

    else:
        # UNet-only fallback
        plt.figure(figsize=(10, 4.5))
        for name, recs in all_runs.items():
            epochs = np.array([r['epoch'] for r in recs], dtype=float)
            loss = np.array([r.get('loss', np.nan) for r in recs], dtype=float)
            plt.plot(epochs, loss, marker='o', lw=1.8, label=name)
        plt.xlabel('Epoch'); plt.ylabel('MSE loss')
        plt.title('UNet training diagnostics from checkpoints')
        plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.show()
        print('Current mode is UNet, so D curves are not included. To analyze G/D, switch train_mode to gan.')

