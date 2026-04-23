"""Verification helpers for the ZA-baseline vector-Ψ WN IC2RES pipeline."""

import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import Pk_library as PKL

import model.new_model as nnmodel
import new_config_ZA as cfg
from new_dddf import NewDDDF
from new_pipeline_ZA import (
    psi_to_delta,
    psi_div_to_delta,
    compute_za_baseline,
    highpass_vector_field,
)
from new_inference_ZA import apply_model_to_field


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

    # ── ZA baseline (returns 4-tuple including vector psi) ─────────────
    baseline_psi_div, baseline_psi, baseline_delta, target_delta = compute_za_baseline(
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
        baseline_psi,
        loaded_model,
        cfg.infer_patch_size,
        cfg.infer_padding,
        cfg.infer_overlap,
        device,
        batch_size=cfg.infer_batch_size,
    )
    residual_pred = highpass_vector_field(residual_pred, k_cut, boxsize, width=k_width)

    final_psi = baseline_psi + residual_pred
    delta_final = psi_to_delta(
        final_psi, dl, q_init, N_p, boxsize, MAS)
    delta_recovered = psi_to_delta(
        target_psi, dl, q_init, N_p, boxsize, MAS)
    delta_residual = delta_recovered - delta_final

    labels = ["N-body", "ZA baseline", "ZA baseline + IC2RES"]
    deltas = [target_delta, baseline_delta, delta_final]

    # ── Overdensity slices ────────────────────────────────────────────────
    slice_labels = ["N-body", "ZA baseline", "ZA baseline + IC2RES", "Residual to N-body"]
    slice_deltas = [target_delta, baseline_delta, delta_final, delta_residual]
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
    k1, k2 = 0.4, 0.4
    theta = np.linspace(0, np.pi, 25)
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
