"""
Data-pipeline helpers for the vector-Ψ IC→Residual workflow.

Extends pipeline.py to support full 3-component displacement fields (psi)
instead of only the scalar divergence (psi_div), enabling recovery of
curl/vorticity information lost in the Helmholtz decomposition.
"""

import gc
import os
import numpy as np
import MAS_library as MASL
import torch
import new_config as cfg


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


# ── Best-fit pipeline (extended to return vector psi) ─────────────────────────

def compute_best_fit(
    dl, init_delta, target_psi_div, init_pos, final_delta,
    veck_main, N_p, boxsize, MAS,
    realization, filter_dir, L,
    coef_file=None,
    overwrite=False,
    k_cut=0.4,
):
    """Compute (or load) the best-fit ∇·Ψ, best-fit Ψ vector, and density fields.

    Returns
    -------
    best_fit_psi_div : ndarray (N_p, N_p, N_p)
    best_fit_psi     : ndarray (N_p, N_p, N_p, 3) – curl-free displacement
    best_fit_delta   : ndarray (N_p, N_p, N_p)
    target_delta     : ndarray (N_p, N_p, N_p)
    """
    fit_path         = cfg.best_fit_path(realization, filter_dir, L, N_p)
    best_fit_delta_p = cfg.best_fit_delta_path(realization, filter_dir, L, N_p)
    target_delta_p   = cfg.target_delta_path(realization, filter_dir, L, N_p)

    # ── best_fit_psi_div ──────────────────────────────────────────────────────
    if not overwrite and os.path.exists(fit_path):
        print(f'Loading best-fit psi_div from {fit_path}')
        best_fit_psi_div = np.load(fit_path)['best_fit_psi_div']
    else:
        psi_div_list = [target_psi_div] + dl.add_funcs(init_delta, veck_main)
        k_filters    = [dl.k_tophat_filter(0.0, k_cut, veck_main),
                        dl.k_end_filter(veck_main)]
        n_fit_layers = len(k_filters) - 1
        n_coef       = len(psi_div_list) - 1

        if coef_file is not None and not os.path.exists(coef_file):
            print(f'Coefficient file not found: {coef_file}, fitting from scratch')
            coef_file = None

        if coef_file is not None:
            layer_coef_all = np.load(coef_file)['layer_best_fit_coef_all']
        else:
            layer_coef_all = np.zeros((n_fit_layers, n_coef), dtype=dl.real_dtype)

        best_fit_psi_div = np.zeros_like(target_psi_div)
        for layer, k_filter in enumerate(k_filters[:-1]):
            print(f'  layer {layer}')
            filtered    = [dl.kx2x_convolve(k_filter, p) for p in psi_div_list]
            filtered[0] = dl.kx2x_convolve(k_filter, target_psi_div - best_fit_psi_div)

            if coef_file is not None:
                coef = layer_coef_all[layer]
            else:
                coef = dl.solve_best_fit(filtered)
                layer_coef_all[layer] = coef

            print(f'  coef = {coef}')
            best_fit_psi_div += np.einsum('l,lijk->ijk', coef, filtered[1:])

        os.makedirs(os.path.dirname(fit_path), exist_ok=True)
        np.savez(fit_path, best_fit_psi_div=best_fit_psi_div)
        print(f'Saved best-fit psi_div to {fit_path}')

        if coef_file is None:
            default_coef_path = cfg.best_fit_coef_path(realization, filter_dir, L, N_p)
            np.savez(default_coef_path, layer_best_fit_coef_all=layer_coef_all)
            print(f'Saved coefficients to {default_coef_path}')

    # ── best_fit_psi (curl-free vector displacement from psi_div) ─────────────
    best_fit_psi = dl.disp_from_psi_div(best_fit_psi_div, veck_main, N_p)

    # ── best_fit_delta ────────────────────────────────────────────────────────
    if not overwrite and os.path.exists(best_fit_delta_p):
        best_fit_delta = np.load(best_fit_delta_p)['best_fit_delta']
    else:
        best_fit_delta = psi_div_to_delta(
            best_fit_psi_div, dl, init_pos, veck_main, N_p, boxsize, MAS)
        np.savez(best_fit_delta_p, best_fit_delta=best_fit_delta)

    # ── target_delta ──────────────────────────────────────────────────────────
    if not overwrite and os.path.exists(target_delta_p):
        target_delta = np.load(target_delta_p)['target_delta']
    else:
        target_delta = final_delta.copy()
        np.savez(target_delta_p, target_delta=target_delta)

    return best_fit_psi_div, best_fit_psi, best_fit_delta, target_delta
