"""
Data-pipeline helpers for the vector-Ψ IC→Residual workflow.

Extends pipeline.py to support full 3-component displacement fields (psi)
instead of only the scalar divergence (psi_div), enabling recovery of
curl/vorticity information lost in the Helmholtz decomposition.
"""

import gc
import os
import numpy as np
import pyfftw
import MAS_library as MASL
import torch
import new_config_ZA as cfg


# ── Re-export unchanged helpers from pipeline ─────────────────────────────────
from pipeline import (
    load_snapshot_pair,
    compute_target_psi_div,
    psi_div_to_delta,
    free_gpu_memory,
)


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
    from pipeline import highpass_field as _hp_scalar
    out = np.empty_like(field)
    for i in range(field.shape[-1]):
        out[..., i] = _hp_scalar(field[..., i], k_cut, boxsize, width=width)
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


# ── ZA baseline pipeline (returns vector psi) ─────────────────────────────────

def compute_za_baseline(
    dl, init_delta, target_psi_div, init_pos, final_delta,
    veck_main, N_p, boxsize, MAS,
    realization, filter_dir, L,
    coef_file=None,
    overwrite=False,
    k_cut=0.4,
):
    """Compute (or load) the ZA baseline ∇·Ψ, Ψ vector, and density fields.

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
        print('coef_file is ignored in ZA baseline pipeline.')
    _ = k_cut  # preserved in signature for drop-in compatibility

    # ── baseline_psi_div (ZA) ─────────────────────────────────────────────────
    if not overwrite and os.path.exists(fit_path):
        print(f'Loading ZA baseline psi_div from {fit_path}')
        baseline_psi_div = np.load(fit_path)['best_fit_psi_div']
    else:
        baseline_psi_div = dl.div_psi_1(init_delta)

        os.makedirs(os.path.dirname(fit_path), exist_ok=True)
        np.savez(fit_path, best_fit_psi_div=baseline_psi_div)
        print(f'Saved ZA baseline psi_div to {fit_path}')

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
