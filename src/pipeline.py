"""
Data-pipeline helpers for the IC→Residual workflow.

Extracted from the notebook so that the notebook only contains
high-level orchestration logic.
"""

import gc
import os
import numpy as np
import MAS_library as MASL
import torch
import config as cfg


# ── Snapshot loading ──────────────────────────────────────────────────────────

def load_snapshot_pair(dl, snapshots, fmt, boxsize, grid_size):
    """Load the IC and final snapshots, return a list of snapshot_info dicts."""
    return [dl.get_snapshot(s, fmt, boxsize, grid_size) for s in snapshots]


# ── Displacement / divergence ─────────────────────────────────────────────────

def compute_target_psi_div(dl, snapshot_info, N_p, boxsize, veck_main):
    """Compute ∇·Ψ_target from an (IC, final) snapshot pair.

    Handles periodic wrapping of particle displacements.
    """
    par_disp = np.astype(
        snapshot_info[1]['pos'] - snapshot_info[0]['pos'], dl.real_dtype
    )
    par_disp = np.where(par_disp <  boxsize / 2, par_disp, par_disp - boxsize)
    par_disp = np.where(par_disp > -boxsize / 2, par_disp, par_disp + boxsize)

    disp_field = np.zeros((N_p, N_p, N_p, 3), dtype=dl.real_dtype)
    dl.disp_from_par(disp_field, snapshot_info[0]['pos'], par_disp, N_p, boxsize)
    return dl.divergence(disp_field, veck_main)


# ── Density field ─────────────────────────────────────────────────────────────

def psi_div_to_delta(psi_div, dl, init_pos, veck_main, N_p, boxsize,
                     MAS='CIC', verbose=False):
    """Paint particles displaced by psi_div onto a grid; return δ = ρ/ρ̄ − 1."""
    delta = np.zeros((N_p, N_p, N_p), dtype=dl.real_dtype)
    pos   = dl.par_pos_from_psi_div(psi_div, init_pos, veck_main, N_p, boxsize)
    MASL.MA(pos, delta, boxsize, MAS, verbose=verbose)
    return delta / np.mean(delta) - 1.0


# ── k-space high-pass filter ──────────────────────────────────────────────────

def highpass_field(field, k_cut, boxsize, width=None):
    """Apply a high-pass spectral filter to field.

    Parameters
    ----------
    field   : ndarray, shape (N, N, N), real-valued
    k_cut   : float, transition wavenumber in h/Mpc
    boxsize : float, box size in Mpc/h
    width   : float or None
        Sigmoid transition width in h/Mpc.
        - None / 0 : hard step (zero all k < k_cut, keep all k >= k_cut)
        - > 0      : soft sigmoid  w(k) = 1 / (1 + exp(-(k - k_cut) / width))
                     The network can still contribute at k < k_cut but with
                     smoothly reduced amplitude, preventing hard artifact edges.
                     Typical value: 0.05–0.15 h/Mpc.

    Returns
    -------
    ndarray, same shape and dtype as field
    """
    N = field.shape[0]
    fk = np.fft.rfftn(field)

    ki  = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi   # full  axes
    kn  = np.fft.rfftfreq(N, d=boxsize / N) * 2 * np.pi  # last  axis (rfft)
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


# ── Best-fit pipeline ─────────────────────────────────────────────────────────

def compute_best_fit(
    dl, init_delta, target_psi_div, init_pos, final_delta,
    veck_main, N_p, boxsize, MAS,
    realization, filter_dir, L,
    coef_file=None,
    overwrite=False,
    k_cut=0.4,
):
    """Compute (or load) the large-scale best-fit ∇·Ψ and density fields.

    Parameters
    ----------
    dl : DDDF
    init_delta : ndarray — initial density contrast
    target_psi_div : ndarray — target ∇·Ψ (N-body)
    init_pos : ndarray — initial particle positions
    final_delta : ndarray — N-body final density contrast
    veck_main : Veck
    N_p, boxsize, MAS : simulation settings
    realization : int
    filter_dir, L : path components
    coef_file : str or None
        Path to a .npz with 'layer_best_fit_coef_all'.
        Pass None to fit coefficients from scratch (and save them to the
        default coef path defined in config).
    overwrite : bool
        False (default): load from disk if the output file already exists.
        True: always recompute and overwrite saved files.
    k_cut : float
        Upper k-space cutoff of the tophat filter in h/Mpc (default 0.4).

    Returns
    -------
    best_fit_psi_div : ndarray
    best_fit_delta   : ndarray
    target_delta     : ndarray
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

    return best_fit_psi_div, best_fit_delta, target_delta