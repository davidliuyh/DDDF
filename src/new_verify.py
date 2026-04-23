"""
Verification helpers for the vector-Ψ WN-based IC2RES pipeline.

Uses NewDDDF + full vector displacement (psi_bestfit → psi_residual)
to recover curl/vorticity information.
"""

import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import Pk_library as PKL

import model.new_model as nnmodel
import new_config as cfg
from new_dddf import NewDDDF
from new_pipeline import (
    psi_to_delta,
    psi_div_to_delta,
    compute_best_fit,
    highpass_vector_field,
    fourier_upsample_field,
    fourier_downsample_field,
)
from new_inference import apply_model_to_field


def _infer_num_pools(state_dict):
    num_pools = len({int(k.split(".")[1]) for k in state_dict if k.startswith("downs.")})
    return max(num_pools, 1)


def _resolve_checkpoint(model=None, infer_train_realizations=None, infer_epochs=None):
    infer_train_realizations = (
        cfg.train_realizations if infer_train_realizations is None else infer_train_realizations
    )
    infer_epochs = cfg.infer_epochs if infer_epochs is None else infer_epochs

    if model is not None:
        base = str(model)
        candidates = [base]
        if not base.endswith(".pth"):
            candidates.append(f"{base}.pth")
            candidates.append(f"{base}-e{infer_epochs}.pth")
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Model checkpoint not found. Tried: " + ", ".join(candidates)
        )

    if cfg.infer_checkpoint is not None:
        checkpoint_path = cfg.infer_checkpoint
    else:
        auto_model = cfg.vec_gan_model_name(
            infer_train_realizations,
            cfg.patch_size,
            cfg.padding,
            cfg.vec_rotate,
            cfg.N_p,
        )
        checkpoint_path = f"{auto_model}-e{infer_epochs}.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_model(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    num_pools = _infer_num_pools(state_dict)
    # Infer channels from the first conv layer weight shape
    inc_key = 'inc.double_conv.0.weight'
    if inc_key in state_dict:
        in_channels = state_dict[inc_key].shape[1]
        base_channels = state_dict[inc_key].shape[0]
    else:
        in_channels = 3  # default for vector pipeline
        base_channels = 16
    n_classes = in_channels  # symmetric: in_channels == n_classes
    model = nnmodel.UNet3D(
        n_classes=n_classes, in_channels=in_channels,
        trilinear=True, base_channels=base_channels, num_pools=num_pools,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_pools


def verify_realization(
    realization,
    model=None,
    k_cut=0.055,
    k_width=0.01,
    coef_file=None,
    infer_train_realizations=None,
    infer_epochs=None,
    bispec_k1=0.2,
    bispec_k2=0.2,
    bispec_n_theta=25,
):
    """Run verification plots for one realization (WN-based pipeline).

    Parameters
    ----------
    realization : int
        Seed / realization index.
    model : str or None
        Explicit checkpoint path.  None → auto-resolve.
    k_cut, k_width : float
        High-pass filter for predicted residual.
    coef_file : str or None
        Best-fit coefficient file.  None → use per-realization coef on disk.
    bispec_k1, bispec_k2 : float
        Fixed triangle sides used in the bispectrum comparison.
    bispec_n_theta : int
        Number of angle bins between 0 and pi for bispectrum plotting.
    """
    importlib.reload(cfg)

    N_p = cfg.N_p
    L = cfg.L
    boxsize = cfg.boxsize
    MAS = cfg.MAS
    threads = cfg.threads
    data_dir = cfg.data_dir
    grid_size = N_p

    dl = NewDDDF(cfg.Omega_m, threads)
    veck_main = dl.Veck(dl, N_p, boxsize, padding=0)

    # ── Load WN initial conditions ────────────────────────────────────────
    wn_info = dl.get_snapshot_wn(
        cfg.wn_psi1_path(realization, N_p),
        cfg.wn_qinit_path(realization, N_p),
        boxsize, grid_size)
    q_init     = wn_info['q_init']
    init_delta = wn_info['delta']

    # ── Load final N-body snapshot ────────────────────────────────────────
    final_info = dl.get_snapshot(
        cfg.final_snapshot_path(realization, N_p),
        cfg.snapshot_format(N_p),
        boxsize, grid_size,
    )
    final_delta = final_info['delta']

    # ── Target displacement (vector + divergence) ──────────────────────
    target_psi_div, target_psi = dl.compute_target_psi_wn(
        q_init, final_info['pos'], N_p, boxsize, veck_main)

    # ── Best-fit (returns 4-tuple including vector psi) ───────────────
    best_fit_psi_div, best_fit_psi, best_fit_delta, target_delta = compute_best_fit(
        dl, init_delta, target_psi_div,
        q_init, final_delta,
        veck_main, N_p, boxsize, MAS,
        realization, data_dir, L,
        coef_file=coef_file,
        overwrite=False,
    )

    # ── Load model and predict residual ───────────────────────────────────
    checkpoint_path = _resolve_checkpoint(
        model=model,
        infer_train_realizations=infer_train_realizations,
        infer_epochs=infer_epochs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model, num_pools = _load_model(checkpoint_path, device)
    print(
        f"GAN Loaded: {checkpoint_path} "
        f"(pools={num_pools}, device={device})"
    )

    residual_pred = apply_model_to_field(
        best_fit_psi,
        loaded_model,
        cfg.infer_patch_size,
        cfg.infer_padding,
        cfg.infer_overlap,
        device,
        batch_size=cfg.infer_batch_size,
    )
    residual_pred = highpass_vector_field(residual_pred, k_cut, boxsize, width=k_width)

    final_psi = best_fit_psi + residual_pred
    delta_final = psi_to_delta(
        final_psi, dl, q_init, N_p, boxsize, MAS)
    delta_recovered = psi_to_delta(
        target_psi, dl, q_init, N_p, boxsize, MAS)
    delta_residual = delta_recovered - delta_final

    labels = ["N-body", "best-fit", "best-fit + IC2RES"]
    deltas = [target_delta, best_fit_delta, delta_final]

    # ── Overdensity slices ────────────────────────────────────────────────
    slice_labels = ["N-body", "best-fit", "best-fit + IC2RES", "Residual to N-body"]
    slice_deltas = [target_delta, best_fit_delta, delta_final, delta_residual]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, d, lab in zip(axes.flat, slice_deltas, slice_labels):
        img = np.mean(d[:1], axis=0).T
        if lab == "Residual to N-body":
            im = ax.imshow(img, cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(img, cmap="gray_r", vmin=-1, vmax=3, origin="lower")
        ax.set_title(lab)
    plt.suptitle(f"[r{realization}] Overdensity slice z=0 (WN)")
    plt.tight_layout()
    plt.show()

    # ── Residual distribution ─────────────────────────────────────────────
    residual_values = delta_residual.ravel()
    residual_mean = float(np.mean(residual_values))
    residual_std  = float(np.std(residual_values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual_values, bins=120, range=(-1.0, 1.0),
            density=True, alpha=0.8, color="tab:blue")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"$\Delta\delta = \delta_{\rm recovered} - \delta_{\rm final}$")
    ax.set_ylabel("PDF")
    ax.set_title(
        f"[r{realization}] Residual distribution "
        f"(mean={residual_mean:.3e}, std={residual_std:.3e})")
    plt.tight_layout()
    plt.show()

    # ── Power spectra ─────────────────────────────────────────────────────
    pks = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
           for d in deltas]
    k_values = pks[0].k3D
    nyquist = float(veck_main.Nyquist_freq)
    nyquist_idx = int(np.argmin(np.abs(k_values - nyquist)))
    nyquist_k = float(k_values[nyquist_idx])
    print(f"[r{realization}] Nyquist target k = {nyquist:.6f}, "
          f"nearest-bin k = {nyquist_k:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P/P_{\rm N\text{-}body}$")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0, 1.5)
    ax.axvline(nyquist, color="grey", ls="--", lw=1, alpha=0.7)
    for pk, lab in zip(pks, labels):
        ax.plot(k_values, pk.Pk[:, 0] / pks[0].Pk[:, 0], label=lab)
    ax.legend()
    ax.set_title(f"[r{realization}] P/P_Nbody z=0 (WN)")

    plt.tight_layout()
    plt.show()

    for pk, lab in zip(pks, labels):
        ratio_nyquist = float(pk.Pk[nyquist_idx, 0] / pks[0].Pk[nyquist_idx, 0])
        print(f"[r{realization}] {lab} P/P_N-body @Nyquist = {ratio_nyquist * 100:.2f}%")

    # ── Bispectrum ───────────────────────────────────────────────────────
    k1, k2 = bispec_k1, bispec_k2
    theta = np.linspace(0, np.pi, bispec_n_theta)
    bks = [PKL.Bk(d, boxsize, k1, k2, theta, MAS=MAS, threads=threads) for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$B/B_{\rm N\text{-}body}$")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"[r{realization}] Bispectrum z=0 (WN), k1={k1}, k2={k2}")
    for bk, lab in zip(bks, labels):
        ratio = np.divide(bk.B, bks[0].B, out=np.zeros_like(bk.B), where=np.abs(bks[0].B) > 0)
        ax.plot(theta, ratio, label=lab)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── chi^2 (k <= 0.3) ─────────────────────────────────────────────────
    mask_k = k_values <= 0.3
    chi2 = {}
    for pk, lab in zip(pks[1:], labels[1:]):
        ratio = pk.Pk[:, 0] / pks[0].Pk[:, 0]
        val = np.mean((ratio[mask_k] - 1) ** 2)
        chi2[lab] = float(val)
        print(f"[r{realization}] {lab} chi^2 (k<=0.3) = {val:.6e}")

    return {
        "realization": realization,
        "checkpoint": checkpoint_path,
        "chi2": chi2,
        "labels": labels,
    }


# ── HR (fiducial 512³) verification ──────────────────────────────────────────

def _resolve_hr_checkpoint(model=None, infer_train_realizations=None, infer_epochs=None):
    infer_train_realizations = (
        cfg.train_realizations if infer_train_realizations is None else infer_train_realizations
    )
    infer_epochs = cfg.hr_epochs if infer_epochs is None else infer_epochs

    if model is not None:
        base = str(model)
        candidates = [base]
        if not base.endswith(".pth"):
            candidates.append(f"{base}.pth")
            candidates.append(f"{base}-e{infer_epochs}.pth")
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "HR model checkpoint not found. Tried: " + ", ".join(candidates)
        )

    auto_model = cfg.hr_vec_gan_model_name(
        infer_train_realizations,
        cfg.patch_size,
        cfg.padding,
        cfg.hr_vec_rotate,
        cfg.N_p_HR,
    )
    checkpoint_path = f"{auto_model}-e{infer_epochs}.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"HR model checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def verify_realization_hr(
    realization,
    model=None,
    k_cut=0.055,
    k_width=0.01,
    coef_file=None,
    infer_train_realizations=None,
    infer_epochs=None,
    bispec_k1=0.2,
    bispec_k2=0.2,
    bispec_n_theta=25,
):
    """Run verification for the HR pipeline on one realization.

    Flow: LR best-fit → Fourier upsample to 512³ → GAN residual at 512³ →
          Fourier downsample to 256³ → paint delta at LR → compare P(k).

    Also compares at HR resolution against the fiducial 512³ N-body.
    """
    importlib.reload(cfg)

    N_p       = cfg.N_p         # 256 (LR)
    N_p_HR    = cfg.N_p_HR      # 512
    L         = cfg.L
    boxsize   = cfg.boxsize
    MAS       = cfg.MAS
    threads   = cfg.threads
    data_dir  = cfg.data_dir

    dl = NewDDDF(cfg.Omega_m, threads)
    veck_LR = dl.Veck(dl, N_p, boxsize, padding=0)
    veck_HR = dl.Veck(dl, N_p_HR, boxsize, padding=0)

    # ── LR: Load WN initial conditions ────────────────────────────────────
    wn_info_LR = dl.get_snapshot_wn(
        cfg.wn_psi1_path(realization, N_p),
        cfg.wn_qinit_path(realization, N_p),
        boxsize, N_p)
    q_init_LR     = wn_info_LR['q_init']
    init_delta_LR = wn_info_LR['delta']

    # ── LR: Load final N-body snapshot ────────────────────────────────────
    final_info_LR = dl.get_snapshot(
        cfg.final_snapshot_path(realization, N_p),
        cfg.snapshot_format(N_p),
        boxsize, N_p,
    )

    # ── LR: Target displacement and best-fit ──────────────────────────────
    target_psi_div_LR, target_psi_LR = dl.compute_target_psi_wn(
        q_init_LR, final_info_LR['pos'], N_p, boxsize, veck_LR)

    bf_psi_div_LR, bf_psi_LR, bf_delta_LR, target_delta_LR = compute_best_fit(
        dl, init_delta_LR, target_psi_div_LR,
        q_init_LR, final_info_LR['delta'],
        veck_LR, N_p, boxsize, MAS,
        realization, data_dir, L,
        coef_file=coef_file,
        overwrite=False,
    )
    del final_info_LR, target_psi_div_LR

    # ── Fourier upsample best-fit to HR ───────────────────────────────────
    print(f'Fourier upsampling best-fit psi {N_p}³ → {N_p_HR}³ ...')
    bf_psi_HR = fourier_upsample_field(bf_psi_LR, N_p_HR, threads)

    # ── Load HR GAN and predict residual ──────────────────────────────────
    checkpoint_path = _resolve_hr_checkpoint(
        model=model,
        infer_train_realizations=infer_train_realizations,
        infer_epochs=infer_epochs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model, num_pools = _load_model(checkpoint_path, device)
    print(
        f"HR GAN Loaded: {checkpoint_path} "
        f"(pools={num_pools}, device={device})"
    )

    residual_pred_HR = apply_model_to_field(
        bf_psi_HR,
        loaded_model,
        cfg.infer_patch_size,
        cfg.infer_padding,
        cfg.infer_overlap,
        device,
        batch_size=cfg.infer_batch_size,
    )
    residual_pred_HR = highpass_vector_field(
        residual_pred_HR, k_cut, boxsize, width=k_width)

    final_psi_HR = bf_psi_HR + residual_pred_HR
    del bf_psi_HR, residual_pred_HR, loaded_model
    torch.cuda.empty_cache()

    # ── Fourier downsample final psi to LR ────────────────────────────────
    print(f'Fourier downsampling final psi {N_p_HR}³ → {N_p}³ ...')
    final_psi_LR = fourier_downsample_field(final_psi_HR, N_p, threads)

    # ── Paint delta at LR ─────────────────────────────────────────────────
    delta_final_LR = psi_to_delta(
        final_psi_LR, dl, q_init_LR, N_p, boxsize, MAS)

    # ── Also paint delta from target psi at LR for comparison ─────────────
    delta_recovered_LR = psi_to_delta(
        target_psi_LR, dl, q_init_LR, N_p, boxsize, MAS)
    delta_residual_LR = delta_recovered_LR - delta_final_LR

    # ── HR verification: paint at 512³ ────────────────────────────────────
    print(f'Loading HR N-body for verification ...')
    wn_info_HR = dl.get_snapshot_wn(
        cfg.wn_psi1_path(realization, N_p_HR),
        cfg.wn_qinit_path(realization, N_p_HR),
        boxsize, N_p_HR)
    q_init_HR = wn_info_HR['q_init']

    final_info_HR = dl.get_snapshot(
        cfg.final_snapshot_path(realization, N_p_HR),
        cfg.snapshot_format(N_p_HR),
        boxsize, N_p_HR,
    )
    target_delta_HR = final_info_HR['delta']

    # HR N-body displacement, then Fourier downsample to LR and paint with LR q_init
    _, target_psi_HR = dl.compute_target_psi_wn(
        q_init_HR, final_info_HR['pos'], N_p_HR, boxsize, veck_HR)
    target_delta_HR_from_psi = psi_to_delta(
        target_psi_HR, dl, q_init_HR, N_p_HR, boxsize, MAS)
    target_psi_HR_to_LR = fourier_downsample_field(target_psi_HR, N_p, threads)
    target_delta_HR_disp_to_LR = psi_to_delta(
        target_psi_HR_to_LR, dl, q_init_LR, N_p, boxsize, MAS)
    del target_psi_HR, target_psi_HR_to_LR

    # HR density Fourier-downsampled to LR
    target_delta_HR_density_to_LR = fourier_downsample_field(target_delta_HR, N_p, threads)

    delta_final_HR = psi_to_delta(
        final_psi_HR, dl, q_init_HR, N_p_HR, boxsize, MAS)
    del final_psi_HR

    # ── Overdensity slices (LR) ───────────────────────────────────────────
    slice_labels = ["N-body (LR)", "best-fit (LR)", "HR→LR IC2RES", "Residual to N-body"]
    slice_deltas = [target_delta_LR, bf_delta_LR, delta_final_LR, delta_residual_LR]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, d, lab in zip(axes.flat, slice_deltas, slice_labels):
        img = np.mean(d[:1], axis=0).T
        if lab == "Residual to N-body":
            im = ax.imshow(img, cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(img, cmap="gray_r", vmin=-1, vmax=3, origin="lower")
        ax.set_title(lab)
    plt.suptitle(f"[r{realization}] HR→LR Overdensity slice z=0")
    plt.tight_layout()
    plt.show()

    # ── Residual distribution (LR) ───────────────────────────────────────
    residual_values = delta_residual_LR.ravel()
    residual_mean = float(np.mean(residual_values))
    residual_std  = float(np.std(residual_values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual_values, bins=120, range=(-1.0, 1.0),
            density=True, alpha=0.8, color="tab:blue")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"$\Delta\delta = \delta_{\rm recovered} - \delta_{\rm final}$")
    ax.set_ylabel("PDF")
    ax.set_title(
        f"[r{realization}] HR→LR Residual distribution "
        f"(mean={residual_mean:.3e}, std={residual_std:.3e})")
    plt.tight_layout()
    plt.show()

    # ── Power spectra: LR comparison ──────────────────────────────────────
    labels_LR = ["N-body (LR)", "best-fit (LR)", "HR→LR IC2RES"]
    deltas_LR = [target_delta_LR, bf_delta_LR, delta_final_LR]

    pks_LR = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
              for d in deltas_LR]
    k_values = pks_LR[0].k3D
    nyquist_LR = float(veck_LR.Nyquist_freq)
    nyquist_idx = int(np.argmin(np.abs(k_values - nyquist_LR)))

    # ── Target spectra used by HR training target construction ───────────
    pk_target_HR_from_psi = PKL.Pk(
        target_delta_HR_from_psi, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
    pk_target_LR_from_downsampled_psi = PKL.Pk(
        target_delta_HR_disp_to_LR, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
    ratio_target_HR_from_psi_to_LR = (
        np.interp(k_values, pk_target_HR_from_psi.k3D, pk_target_HR_from_psi.Pk[:, 0])
        / pks_LR[0].Pk[:, 0]
    )
    ratio_target_LR_from_downsampled_psi_to_LR = (
        pk_target_LR_from_downsampled_psi.Pk[:, 0] / pks_LR[0].Pk[:, 0]
    )

    pk_HR_disp_to_LR = PKL.Pk(
        target_delta_HR_disp_to_LR, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
    pk_HR_density_to_LR = PKL.Pk(
        target_delta_HR_density_to_LR, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)

    # No-downsample HR density ratio to LR N-body, shown up to LR k max only
    pk_HR_nodown = PKL.Pk(
        target_delta_HR, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
    ratio_HR_nodown_to_LR = np.interp(k_values, pk_HR_nodown.k3D, pk_HR_nodown.Pk[:, 0]) / pks_LR[0].Pk[:, 0]

    ratio_curves = [
        ("N-body (LR)", pks_LR[0].Pk[:, 0] / pks_LR[0].Pk[:, 0]),
        ("best-fit (LR)", pks_LR[1].Pk[:, 0] / pks_LR[0].Pk[:, 0]),
        ("HR→LR IC2RES", pks_LR[2].Pk[:, 0] / pks_LR[0].Pk[:, 0]),
        ("HR N-body disp↓ + LR q_init assign / N-body (LR)",
         pk_HR_disp_to_LR.Pk[:, 0] / pks_LR[0].Pk[:, 0]),
        ("HR density↓ / N-body (LR)",
         pk_HR_density_to_LR.Pk[:, 0] / pks_LR[0].Pk[:, 0]),
        ("HR density (no↓) / N-body (LR), k<=kmax(LR)",
         ratio_HR_nodown_to_LR),
        ("target_psi + 512 q_init (512^3) / N-body (LR)",
         ratio_target_HR_from_psi_to_LR),
        ("downsample(target_psi) + 256 q_init / N-body (LR)",
         ratio_target_LR_from_downsampled_psi_to_LR),
    ]

    overlap_pairs = []
    overlap_indices = set()
    abs_tol = 1.0e-10
    rel_tol = 1.0e-7
    for i in range(len(ratio_curves)):
        name_i, y_i = ratio_curves[i]
        for j in range(i + 1, len(ratio_curves)):
            name_j, y_j = ratio_curves[j]
            diff_abs = np.abs(y_i - y_j)
            denom = np.maximum(np.maximum(np.abs(y_i), np.abs(y_j)), 1.0e-12)
            diff_rel = diff_abs / denom
            max_abs = float(np.nanmax(diff_abs))
            max_rel = float(np.nanmax(diff_rel))
            if max_abs < abs_tol or max_rel < rel_tol:
                overlap_pairs.append((name_i, name_j, max_abs, max_rel))
                overlap_indices.add(i)
                overlap_indices.add(j)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P/P_{\rm N\text{-}body}$")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0, 1.5)
    ax.axvline(nyquist_LR, color="grey", ls="--", lw=1, alpha=0.7)
    for idx, (lab, curve) in enumerate(ratio_curves):
        label = lab + (" [OVERLAP]" if idx in overlap_indices else "")
        ax.plot(k_values, curve, label=label)

    if overlap_pairs:
        overlap_lines = ["Overlapping curves:"]
        overlap_lines.extend([
            f"#{idx + 1}: {a} == {b}"
            for idx, (a, b, _, _) in enumerate(overlap_pairs)
        ])
        ax.text(
            0.02,
            0.03,
            "\n".join(overlap_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
        )
    ax.legend()
    ax.set_title(f"[r{realization}] P/P_Nbody z=0 (HR→LR)")
    plt.tight_layout()
    plt.show()

    if overlap_pairs:
        print(f"[r{realization}] Overlap summary ({len(overlap_pairs)} pairs):")
        for idx, (a, b, max_abs, max_rel) in enumerate(overlap_pairs, start=1):
            print(
                f"[r{realization}] OVERLAP #{idx}: {a} == {b} "
                f"(max|Δ|={max_abs:.3e}, max rel Δ={max_rel:.3e})"
            )

    for pk, lab in zip(pks_LR, labels_LR):
        ratio_nyquist = float(pk.Pk[nyquist_idx, 0] / pks_LR[0].Pk[nyquist_idx, 0])
        print(f"[r{realization}] {lab} P/P_N-body @Nyquist(LR) = {ratio_nyquist * 100:.2f}%")

    # ── Bispectrum: LR comparison ───────────────────────────────────────
    k1, k2 = bispec_k1, bispec_k2
    theta = np.linspace(0, np.pi, bispec_n_theta)
    bks_LR = [PKL.Bk(d, boxsize, k1, k2, theta, MAS=MAS, threads=threads) for d in deltas_LR]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$B/B_{\rm N\text{-}body}$")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"[r{realization}] Bispectrum z=0 (HR→LR), k1={k1}, k2={k2}")
    for bk, lab in zip(bks_LR, labels_LR):
        ratio = np.divide(bk.B, bks_LR[0].B, out=np.zeros_like(bk.B), where=np.abs(bks_LR[0].B) > 0)
        ax.plot(theta, ratio, label=lab)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Power spectra: HR comparison ──────────────────────────────────────
    labels_HR = ["N-body (HR)", "HR IC2RES"]
    deltas_HR = [target_delta_HR, delta_final_HR]

    pks_HR = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False)
              for d in deltas_HR]
    k_HR = pks_HR[0].k3D
    nyquist_HR = float(veck_HR.Nyquist_freq)
    nyquist_HR_idx = int(np.argmin(np.abs(k_HR - nyquist_HR)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P/P_{\rm N\text{-}body}$")
    ax.set_xlim(0.01, 2.0)
    ax.set_ylim(0, 1.5)
    ax.axvline(nyquist_LR, color="grey", ls="--", lw=1, alpha=0.5, label="Nyquist LR")
    ax.axvline(nyquist_HR, color="grey", ls=":", lw=1, alpha=0.5, label="Nyquist HR")
    for pk, lab in zip(pks_HR, labels_HR):
        ax.plot(k_HR, pk.Pk[:, 0] / pks_HR[0].Pk[:, 0], label=lab)
    ax.legend()
    ax.set_title(f"[r{realization}] P/P_Nbody z=0 (HR 512³)")
    plt.tight_layout()
    plt.show()

    for pk, lab in zip(pks_HR, labels_HR):
        ratio = float(pk.Pk[nyquist_HR_idx, 0] / pks_HR[0].Pk[nyquist_HR_idx, 0])
        print(f"[r{realization}] {lab} P/P_N-body @Nyquist(HR) = {ratio * 100:.2f}%")

    # ── chi^2 (k <= 0.3) at LR ───────────────────────────────────────────
    mask_k = k_values <= 0.3
    chi2 = {}
    for pk, lab in zip(pks_LR[1:], labels_LR[1:]):
        ratio = pk.Pk[:, 0] / pks_LR[0].Pk[:, 0]
        val = np.mean((ratio[mask_k] - 1) ** 2)
        chi2[lab] = float(val)
        print(f"[r{realization}] {lab} chi^2 (k<=0.3, LR) = {val:.6e}")

    return {
        "realization": realization,
        "checkpoint": checkpoint_path,
        "chi2": chi2,
        "labels_LR": labels_LR,
        "labels_HR": labels_HR,
    }
