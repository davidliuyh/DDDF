"""Verification helpers for IC2RES models.

Pick a realization and a model checkpoint, then generate the same
slice/power-spectrum/bispectrum plots previously produced in the notebook.
"""

import os

import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import Pk_library as PKL

import dddf
import model.model as nnmodel

import config as cfg
from pipeline import (
    load_snapshot_pair,
    compute_target_psi_div,
    psi_div_to_delta,
    compute_best_fit,
    highpass_field,
)
from inference import apply_model_to_field


def _infer_num_pools(state_dict):
    """Infer generator depth from checkpoint key names."""
    num_pools = len({int(k.split(".")[1]) for k in state_dict if k.startswith("downs.")})
    return max(num_pools, 1)


def _resolve_checkpoint(model=None, infer_train_realizations=None, infer_epochs=None):
    """Resolve checkpoint path using the same logic as notebook model loading.

    Resolution order:
    1) If ``model`` is provided, treat it as explicit checkpoint/base path.
    2) Else if ``cfg.infer_checkpoint`` is set, use it directly.
    3) Else auto-derive from GAN model name and train realizations,
       then append ``-e{infer_epochs}.pth``.

    For explicit ``model`` values, accepted forms are:
    - Full ``.pth`` path
    - Path without ``.pth``
    - Base model name (e.g. output of cfg.gan_model_name)
      where this helper will try ``-e{infer_epochs}.pth``.
    """
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
        auto_model = cfg.gan_model_name(
            infer_train_realizations,
            cfg.patch_size,
            cfg.padding,
            cfg.rotate,
            cfg.N_p,
            model_dir=cfg.model_dir,
        )
        checkpoint_path = f"{auto_model}-e{infer_epochs}.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_model(checkpoint_path, device):
    """Load generator from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    num_pools = _infer_num_pools(state_dict)
    model = nnmodel.UNet3D(n_classes=1, trilinear=True, base_channels=16, num_pools=num_pools)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_pools


def verify_realization(
    realization,
    model=None,
    k_cut=0.061,
    k_width=0.01,
    coef_file=None,
    infer_train_realizations=None,
    infer_epochs=None,
):
    """Run verification plots for one realization.

    Parameters
    ----------
    realization : int
        Snapshot realization index to evaluate.
    model : str or None
        Explicit checkpoint/base model path. If None, uses the exact same auto
        logic as notebook "Load Trained Model".
        infer_train_realizations : list[int] or int or None
            Realizations used for auto-deriving checkpoint when ``model is None``.
            Defaults to ``cfg.train_realizations``.
        infer_epochs : int or None
            Epoch suffix used for auto/base-name checkpoint resolution.
            Defaults to ``cfg.infer_epochs``.
    k_cut, k_width : float
        High-pass filter settings for predicted residual.
    coef_file : str or None
        Optional best-fit coefficient file path.
    """
    importlib.reload(cfg)  # ensure config changes are picked up when running multiple times

    N_p = cfg.N_p
    L = cfg.L
    boxsize = cfg.boxsize
    MAS = cfg.MAS
    threads = cfg.threads
    data_dir = cfg.data_dir

    dl = dddf.DDDF(cfg.init_redshift(N_p), cfg.final_snapshot_z, cfg.Omega_m, threads)
    veck_main = dl.Veck(dl, N_p, boxsize, padding=0)
    grid_size = N_p

    snapshot_info = load_snapshot_pair(
        dl,
        cfg.snapshot_paths(realization, N_p),
        cfg.snapshot_format(N_p),
        boxsize,
        grid_size,
    )

    init_delta = snapshot_info[0]["delta"]
    target_psi_div = compute_target_psi_div(dl, snapshot_info, N_p, boxsize, veck_main)

    if coef_file is None:
        coef_file = cfg.best_fit_avg_coef_path(data_dir, L, N_p)

    best_fit_psi_div, best_fit_delta, target_delta = compute_best_fit(
        dl,
        init_delta,
        target_psi_div,
        snapshot_info[0]["pos"],
        snapshot_info[1]["delta"],
        veck_main,
        N_p,
        boxsize,
        MAS,
        realization,
        data_dir,
        L,
        coef_file=coef_file,
        overwrite=False,
    )

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
        init_delta,
        loaded_model,
        cfg.infer_patch_size,
        cfg.infer_padding,
        cfg.infer_overlap,
        device,
    )
    residual_pred = highpass_field(residual_pred, k_cut, boxsize, width=k_width)

    final_psi_div = best_fit_psi_div + residual_pred
    delta_final = psi_div_to_delta(
        final_psi_div,
        dl,
        snapshot_info[0]["pos"],
        veck_main,
        N_p,
        boxsize,
        MAS,
    )
    delta_recovered = psi_div_to_delta(
        target_psi_div,
        dl,
        snapshot_info[0]["pos"],
        veck_main,
        N_p,
        boxsize,
        MAS,
    )
    delta_residual = delta_recovered - delta_final

    labels = ["N-body", "best-fit", "best-fit + IC2RES", "Recovered"]
    deltas = [target_delta, best_fit_delta, delta_final, delta_recovered]

    # Overdensity slices
    slice_labels = ["N-body", "best-fit", "best-fit + IC2RES", "Residual to Recovered"]
    slice_deltas = [target_delta, best_fit_delta, delta_final, delta_residual]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, d, lab in zip(axes.flat, slice_deltas, slice_labels):
        img = np.mean(d[:1], axis=0).T
        if lab == "Residual to Recovered":
            im = ax.imshow(img, cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(img, cmap="gray_r", vmin=-1, vmax=3, origin="lower")
        ax.set_title(lab)
    plt.suptitle(f"[r{realization}] Overdensity slice z=0")
    plt.tight_layout()
    plt.show()

    # Residual distribution
    residual_values = delta_residual.ravel()
    residual_mean = float(np.mean(residual_values))
    residual_std = float(np.std(residual_values))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        residual_values,
        bins=120,
        range=(-1.0, 1.0),
        density=True,
        alpha=0.8,
        color="tab:blue",
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"$\Delta\delta = \delta_{\rm recovered} - \delta_{\rm final}$")
    ax.set_ylabel("PDF")
    ax.set_title(
        f"[r{realization}] Residual distribution "
        f"(mean={residual_mean:.3e}, std={residual_std:.3e})"
    )
    plt.tight_layout()
    plt.show()

    # Power spectra
    pks = [PKL.Pk(d, boxsize, axis=0, MAS=MAS, threads=threads, verbose=False) for d in deltas]
    k_values = pks[0].k3D
    nyquist = float(veck_main.Nyquist_freq)
    nyquist_idx = int(np.argmin(np.abs(k_values - nyquist)))
    nyquist_k = float(k_values[nyquist_idx])
    print(
        f"[r{realization}] Nyquist target k = {nyquist:.6f}, "
        f"nearest-bin k = {nyquist_k:.6f}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P/P_{\rm N\text{-}body}$")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0, 1.5)
    ax.axvline(veck_main.Nyquist_freq, color="grey", ls="--", lw=1, alpha=0.7)
    for pk, lab in zip(pks, labels):
        ax.plot(pks[0].k3D, pk.Pk[:, 0] / pks[0].Pk[:, 0], label=lab)
    ax.legend()
    plt.title(f"[r{realization}] Matter power spectrum z=0")
    plt.show()
    for pk, lab in zip(pks, labels):
        ratio_nyquist = float(pk.Pk[nyquist_idx, 0] / pks[0].Pk[nyquist_idx, 0])
        print(f"[r{realization}] {lab} P/P_N-body @Nyquist = {ratio_nyquist * 100:.2f}%")

    # # Bispectrum
    # k1, k2 = 0.2, 0.2
    # theta = np.linspace(0, np.pi, 25)
    # bks = [PKL.Bk(d, boxsize, k1, k2, theta, MAS=MAS, threads=threads) for d in deltas]

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.set_xlabel(r"$\theta$")
    # ax.set_ylabel(r"$B/B_{\rm N\text{-}body}$")
    # ax.set_xlim(0, np.pi)
    # ax.set_ylim(0, 1.5)
    # ax.set_title(f"[r{realization}] Bispectrum z=0, k1={k1}, k2={k2}")
    # for bk, lab in zip(bks, labels):
    #     ax.plot(theta, bk.B / bks[0].B, label=lab)
    # ax.legend()
    # plt.show()

    # chi^2 vs N-body (k <= 0.3)
    mask_k = pks[0].k3D <= 0.3
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
