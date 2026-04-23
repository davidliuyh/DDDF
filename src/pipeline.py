"""
Data-pipeline helpers for the vector-Ψ IC→Residual workflow.

Supports full 3-component displacement fields (psi) in addition to
the scalar divergence representation (psi_div).
"""

import gc
import os
import numpy as np
import pyfftw
import MAS_library as MASL
import torch
import config as cfg


# ── Snapshot loading ──────────────────────────────────────────────────────────

def load_snapshot_pair(dl, snapshots, fmt, boxsize, grid_size):
    """Load the IC and final snapshots, return a list of snapshot_info dicts."""
    return [dl.get_snapshot(s, fmt, boxsize, grid_size) for s in snapshots]


# ── Displacement / divergence ─────────────────────────────────────────────────

def compute_target_psi_div(dl, snapshot_info, N_p, boxsize, veck_main):
    """Compute ∇·Ψ_target from an (IC, final) snapshot pair."""
    par_disp = (snapshot_info[1]['pos'] - snapshot_info[0]['pos']).astype(dl.real_dtype)
    par_disp = np.where(par_disp < -boxsize / 2, par_disp + boxsize, par_disp)
    par_disp = np.where(par_disp > boxsize / 2, par_disp - boxsize, par_disp)

    disp_field = np.zeros((N_p, N_p, N_p, 3), dtype=dl.real_dtype)
    dl.disp_from_par(disp_field, snapshot_info[0]['pos'], par_disp, N_p, boxsize)
    return dl.divergence(disp_field, veck_main)


# ── Density field ─────────────────────────────────────────────────────────────

def psi_div_to_delta(psi_div, dl, init_pos, veck_main, N_p, boxsize,
                     MAS='CIC', verbose=False):
    """Paint particles displaced by psi_div onto a grid; return δ = ρ/ρ̄ − 1."""
    delta = np.zeros((N_p, N_p, N_p), dtype=dl.real_dtype)
    pos = dl.par_pos_from_psi_div(psi_div, init_pos, veck_main, N_p, boxsize)
    MASL.MA(pos, delta, boxsize, MAS, verbose=verbose)
    return delta / np.mean(delta) - 1.0


# ── k-space high-pass filter ──────────────────────────────────────────────────

def highpass_field(field, k_cut, boxsize, width=None):
    """Apply a high-pass spectral filter to a scalar field."""
    N = field.shape[0]
    fk = np.fft.rfftn(field)

    ki = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
    kn = np.fft.rfftfreq(N, d=boxsize / N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(ki, ki, kn, indexing='ij')
    k_abs = np.sqrt(kx**2 + ky**2 + kz**2)

    if width:
        mask = 1.0 / (1.0 + np.exp(-(k_abs - k_cut) / width))
    else:
        mask = (k_abs >= k_cut).astype(np.float64)

    fk *= mask
    return np.fft.irfftn(fk, s=field.shape).astype(field.dtype)


# ── GPU memory ────────────────────────────────────────────────────────────────

def free_gpu_memory():
    """Release cached GPU memory and report current allocation."""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')


# ── Vector displacement → density ─────────────────────────────────────────────

def psi_to_delta(psi_field, dl, init_pos, N_p, boxsize, MAS='CIC', verbose=False):
    """Paint particles displaced by the full vector Ψ onto a grid; return δ = ρ/ρ̄ − 1.

    Parameters
    ----------
    psi_field : ndarray, shape (N_p, N_p, N_p, 3)
        Full 3-component displacement field.
    dl : DDDF
    init_pos : ndarray, shape (N_part, 3)
        Initial (Lagrangian) particle positions.
    N_p : int
    boxsize : float
    MAS : str
    verbose : bool

    Returns
    -------
    ndarray, shape (N_p, N_p, N_p) – overdensity field δ
    """
    par_disp = np.zeros((N_p**3, 3), dtype=dl.real_dtype)
    dl.assign_disp(par_disp, init_pos, psi_field, N_p, boxsize)
    pos = (par_disp + init_pos) % boxsize
    pos = np.where(pos >= boxsize, 0.0, pos)

    delta = np.zeros((N_p, N_p, N_p), dtype=dl.real_dtype)
    MASL.MA(pos, delta, boxsize, MAS, verbose=verbose)
    return delta / np.mean(delta) - 1.0


# ── k-space high-pass filter for vector fields ────────────────────────────────

def highpass_vector_field(field, k_cut, boxsize, width=None):
    """Apply a high-pass spectral filter to each component of a vector field.

    Parameters
    ----------
    field : ndarray, shape (N, N, N, 3)
    k_cut : float
    boxsize : float
    width : float or None

    Returns
    -------
    ndarray, same shape and dtype as field
    """
    out = np.empty_like(field)
    for i in range(field.shape[-1]):
        out[..., i] = highpass_field(field[..., i], k_cut, boxsize, width=width)
    return out


# ── Fourier-space resolution transfer ─────────────────────────────────────────

def _fourier_upsample_3d(field_3d, target_N, threads=32):
    """Upsample a single 3D scalar field via FFT zero-padding.

    FFT → fftshift (center DC) → zero-pad symmetrically → ifftshift → IFFT.
    Scales by (target_N / N)³ to preserve the physical amplitude.

    Parameters
    ----------
    field_3d : ndarray, shape (N, N, N)
    target_N : int  (must be > N, both even)
    threads  : int

    Returns
    -------
    ndarray, shape (target_N, target_N, target_N), same dtype as input
    """
    N = field_3d.shape[0]
    assert target_N > N and target_N % 2 == 0 and N % 2 == 0

    fk = pyfftw.builders.fftn(field_3d, threads=threads)()
    fk_shifted = np.fft.fftshift(fk)

    pad_each = (target_N - N) // 2
    fk_padded = np.pad(fk_shifted, pad_each, mode='constant', constant_values=0)

    fk_out = np.fft.ifftshift(fk_padded)
    scale = (target_N / N) ** 3
    result = pyfftw.builders.ifftn(fk_out * scale, threads=threads)().real
    return result.astype(field_3d.dtype)


def _fourier_downsample_3d(field_3d, target_N, threads=32):
    """Downsample a single 3D scalar field via FFT truncation.

    FFT → fftshift (center DC) → crop symmetrically → ifftshift → IFFT.
    Scales by (target_N / N)³ to preserve the physical amplitude.

    Parameters
    ----------
    field_3d : ndarray, shape (N, N, N)
    target_N : int  (must be < N, both even)
    threads  : int

    Returns
    -------
    ndarray, shape (target_N, target_N, target_N), same dtype as input
    """
    N = field_3d.shape[0]
    assert target_N < N and target_N % 2 == 0 and N % 2 == 0

    fk = pyfftw.builders.fftn(field_3d, threads=threads)()
    fk_shifted = np.fft.fftshift(fk)

    crop = (N - target_N) // 2
    fk_cropped = fk_shifted[crop:crop + target_N,
                            crop:crop + target_N,
                            crop:crop + target_N]

    fk_out = np.fft.ifftshift(fk_cropped)
    scale = (target_N / N) ** 3
    result = pyfftw.builders.ifftn(fk_out * scale, threads=threads)().real
    return result.astype(field_3d.dtype)


def fourier_upsample_field(field, target_N, threads=32):
    """Upsample a 3D scalar or 4D vector field via FFT zero-padding.

    Parameters
    ----------
    field : ndarray, shape (N, N, N) or (N, N, N, C)
    target_N : int
    threads : int

    Returns
    -------
    ndarray, shape (target_N,)*3 or (target_N,)*3 + (C,)
    """
    if field.ndim == 3:
        return _fourier_upsample_3d(field, target_N, threads)
    elif field.ndim == 4:
        C = field.shape[-1]
        out = np.empty((target_N, target_N, target_N, C), dtype=field.dtype)
        for i in range(C):
            out[..., i] = _fourier_upsample_3d(field[..., i], target_N, threads)
        return out
    else:
        raise ValueError(f'Expected 3D or 4D field, got {field.ndim}D')


def fourier_downsample_field(field, target_N, threads=32):
    """Downsample a 3D scalar or 4D vector field via FFT truncation.

    Parameters
    ----------
    field : ndarray, shape (N, N, N) or (N, N, N, C)
    target_N : int
    threads : int

    Returns
    -------
    ndarray, shape (target_N,)*3 or (target_N,)*3 + (C,)
    """
    if field.ndim == 3:
        return _fourier_downsample_3d(field, target_N, threads)
    elif field.ndim == 4:
        C = field.shape[-1]
        out = np.empty((target_N, target_N, target_N, C), dtype=field.dtype)
        for i in range(C):
            out[..., i] = _fourier_downsample_3d(field[..., i], target_N, threads)
        return out
    else:
        raise ValueError(f'Expected 3D or 4D field, got {field.ndim}D')


# ── Baseline pipeline (returns vector psi) ────────────────────────────────────

def compute_baseline(
    dl, init_delta, target_psi_div, init_pos, final_delta,
    veck_main, N_p, boxsize, MAS,
    realization, filter_dir, L,
    coef_file=None,
    overwrite=False,
    k_cut=0.4,
):
    """Compute (or load) the baseline ∇·Ψ, Ψ vector, and density fields.

    Returns
    -------
    baseline_psi_div : ndarray (N_p, N_p, N_p)
    baseline_psi     : ndarray (N_p, N_p, N_p, 3) – curl-free displacement
    baseline_delta   : ndarray (N_p, N_p, N_p)
    target_delta     : ndarray (N_p, N_p, N_p)
    """
    fit_path         = cfg.best_fit_path(realization, filter_dir, L, N_p)
    best_fit_delta_p = cfg.best_fit_delta_path(realization, filter_dir, L, N_p)
    target_delta_p   = cfg.target_delta_path(realization, filter_dir, L, N_p)

    if coef_file is not None:
        print('coef_file is ignored in this baseline pipeline.')
    _ = k_cut  # preserved in signature for drop-in compatibility

    # ── baseline_psi_div ──────────────────────────────────────────────────────
    if not overwrite and os.path.exists(fit_path):
        print(f'Loading baseline psi_div from {fit_path}')
        baseline_psi_div = np.load(fit_path)['best_fit_psi_div']
    else:
        baseline_psi_div = dl.div_psi_1(init_delta)

        os.makedirs(os.path.dirname(fit_path), exist_ok=True)
        np.savez(fit_path, best_fit_psi_div=baseline_psi_div)
        print(f'Saved baseline psi_div to {fit_path}')

    # ── baseline_psi (curl-free vector displacement from psi_div) ─────────────
    baseline_psi = dl.disp_from_psi_div(baseline_psi_div, veck_main, N_p)

    # ── baseline_delta ─────────────────────────────────────────────────────────
    if not overwrite and os.path.exists(best_fit_delta_p):
        baseline_delta = np.load(best_fit_delta_p)['best_fit_delta']
    else:
        baseline_delta = psi_div_to_delta(
            baseline_psi_div, dl, init_pos, veck_main, N_p, boxsize, MAS)
        np.savez(best_fit_delta_p, best_fit_delta=baseline_delta)

    # ── target_delta ──────────────────────────────────────────────────────────
    if not overwrite and os.path.exists(target_delta_p):
        target_delta = np.load(target_delta_p)['target_delta']
    else:
        target_delta = final_delta.copy()
        np.savez(target_delta_p, target_delta=target_delta)

    return baseline_psi_div, baseline_psi, baseline_delta, target_delta


def compute_best_fit(
    dl, init_delta, target_psi_div, init_pos, final_delta,
    veck_main, N_p, boxsize, MAS,
    realization, filter_dir, L,
    coef_file=None,
    overwrite=False,
    k_cut=0.4,
):
    """Backward-compatible alias returning the scalar-triplet output shape."""
    baseline_psi_div, _, baseline_delta, target_delta = compute_baseline(
        dl, init_delta, target_psi_div, init_pos, final_delta,
        veck_main, N_p, boxsize, MAS,
        realization, filter_dir, L,
        coef_file=coef_file,
        overwrite=overwrite,
        k_cut=k_cut,
    )
    return baseline_psi_div, baseline_delta, target_delta
