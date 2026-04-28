"""Inference helpers for the WN IC->Residual displacement pipeline."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import config as cfg
import model.model as nnmodel
from dddf import DDDF
from pipeline import compute_baseline, highpass_vector_field


_TWO_LPT_RUNTIME_MODULE: Optional[ModuleType] = None


def apply_model_to_field(field, model, patch_size, pad, overlap, device=None, batch_size=8):
    """Apply a trained model to every patch of a periodic 3D or 4D field.

    Parameters
    ----------
    field : np.ndarray
        Input field: (N, N, N) scalar or (N, N, N, C) vector.
    model : torch.nn.Module
    patch_size, pad : int
    overlap : float
    device : torch.device or None

    Returns
    -------
    np.ndarray – same shape and dtype as ``field``.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_vector = field.ndim == 4
    N = field.shape[0]
    step = max(1, int(patch_size * (1 - overlap)))

    if is_vector:
        C = field.shape[-1]
        output = np.zeros((N, N, N, C), dtype=np.float64)
        weight = np.zeros((N, N, N), dtype=np.float64)
    else:
        output = np.zeros((N, N, N), dtype=np.float64)
        weight = np.zeros((N, N, N), dtype=np.float64)

    model.eval()
    t0 = time.time()

    grid = range(0, N, step)
    total_patches = len(grid) ** 3

    def flush_batch(patch_tensors, patch_meta):
        if not patch_tensors:
            return

        inp = torch.stack(patch_tensors, dim=0).float().to(device, non_blocking=True)
        pred = model(inp).cpu().numpy()

        for i, (z, y, x, cz, cy, cx) in enumerate(patch_meta):
            if is_vector:
                # (B, C, D, H, W) -> (D, H, W, C)
                pred_i = pred[i].transpose(1, 2, 3, 0)
            else:
                # (B, 1, D, H, W) -> (D, H, W)
                pred_i = pred[i, 0]

            crop = pred_i[
                pad:pad + patch_size,
                pad:pad + patch_size,
                pad:pad + patch_size,
            ]

            output[z:z + cz, y:y + cy, x:x + cx] += crop[:cz, :cy, :cx]
            weight[z:z + cz, y:y + cy, x:x + cx] += 1

    with torch.no_grad():
        patch_tensors = []
        patch_meta = []

        with tqdm(total=total_patches, desc='apply_model (patch-batch)') as pbar:
            for z in grid:
                for y in grid:
                    for x in grid:
                        zs = np.arange(z - pad, z + patch_size + pad) % N
                        ys = np.arange(y - pad, y + patch_size + pad) % N
                        xs = np.arange(x - pad, x + patch_size + pad) % N

                        block = field[np.ix_(zs, ys, xs)]

                        if is_vector:
                            # (D, H, W, C) -> (C, D, H, W)
                            inp = torch.from_numpy(block).permute(3, 0, 1, 2)
                        else:
                            # (D, H, W) -> (1, D, H, W)
                            inp = torch.from_numpy(block).unsqueeze(0)

                        z_end = min(z + patch_size, N)
                        y_end = min(y + patch_size, N)
                        x_end = min(x + patch_size, N)
                        cz, cy, cx = z_end - z, y_end - y, x_end - x

                        patch_tensors.append(inp)
                        patch_meta.append((z, y, x, cz, cy, cx))

                        if len(patch_tensors) >= batch_size:
                            flush_batch(patch_tensors, patch_meta)
                            pbar.update(len(patch_tensors))
                            patch_tensors.clear()
                            patch_meta.clear()

            if patch_tensors:
                flush_batch(patch_tensors, patch_meta)
                pbar.update(len(patch_tensors))

    if is_vector:
        w = weight[..., np.newaxis]
        output = np.where(w > 0, output / w, 0.0).astype(field.dtype)
    else:
        output = np.where(weight > 0, output / weight, 0.0).astype(field.dtype)

    elapsed = time.time() - t0
    print(
        f'apply_model done: step={step}, overlap={overlap}, '
        f'batch_size={batch_size}, elapsed={elapsed:.1f}s'
    )
    return output


def _infer_num_pools(state_dict):
    num_pools = len({int(k.split(".")[1]) for k in state_dict if k.startswith("downs.")})
    return max(num_pools, 1)


def _is_torch_state_dict(obj):
    return isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())


def _normalize_state_dict_keys(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _extract_generator_state(checkpoint_obj):
    if _is_torch_state_dict(checkpoint_obj):
        return _normalize_state_dict_keys(checkpoint_obj)

    if isinstance(checkpoint_obj, dict):
        if _is_torch_state_dict(checkpoint_obj.get("net_G_state")):
            return _normalize_state_dict_keys(checkpoint_obj["net_G_state"])

        state_dict = checkpoint_obj.get("state_dict")
        if _is_torch_state_dict(state_dict):
            if any(k.startswith("net_G.") for k in state_dict):
                state_dict = {
                    k[len("net_G."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("net_G.")
                }
            return _normalize_state_dict_keys(state_dict)

    raise ValueError("Unsupported checkpoint format: cannot extract generator state dict.")


def _export_generator_pth_from_ckpt(ckpt_path, pth_path):
    checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_generator_state(checkpoint_obj)
    os.makedirs(os.path.dirname(pth_path) or ".", exist_ok=True)
    torch.save(state_dict, pth_path)
    print(f"Auto-generated .pth from .ckpt: {ckpt_path} -> {pth_path}")
    return pth_path


def _resolve_or_export_checkpoint(path):
    base, ext = os.path.splitext(path)

    if ext == ".ckpt":
        pth_path = f"{base}.pth"
        if os.path.exists(pth_path):
            return pth_path
        if os.path.exists(path):
            return _export_generator_pth_from_ckpt(path, pth_path)
        return None

    if os.path.exists(path):
        return path

    if ext == ".pth":
        ckpt_path = f"{base}.ckpt"
        if os.path.exists(ckpt_path):
            return _export_generator_pth_from_ckpt(ckpt_path, path)
        return None

    pth_path = f"{path}.pth"
    if os.path.exists(pth_path):
        return pth_path

    ckpt_path = f"{path}.ckpt"
    if os.path.exists(ckpt_path):
        return _export_generator_pth_from_ckpt(ckpt_path, pth_path)
    return None


def _resolve_checkpoint(model=None, infer_train_realizations=None, infer_epochs=None):
    infer_train_realizations = (
        cfg.train_realizations if infer_train_realizations is None else infer_train_realizations
    )
    infer_epochs = cfg.infer_epochs if infer_epochs is None else infer_epochs

    if model is not None:
        base = str(model)
        candidates = [base]
        if not base.endswith((".pth", ".ckpt")):
            candidates.append(f"{base}.pth")
            candidates.append(f"{base}-e{infer_epochs}.pth")
        for path in candidates:
            resolved = _resolve_or_export_checkpoint(path)
            if resolved is not None:
                return resolved
        raise FileNotFoundError(
            "Model checkpoint not found (.pth/.ckpt). Tried: " + ", ".join(candidates)
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

    resolved = _resolve_or_export_checkpoint(str(checkpoint_path))
    if resolved is None:
        raise FileNotFoundError(f"Model checkpoint not found (.pth/.ckpt): {checkpoint_path}")
    return resolved


def _load_model(checkpoint_path, device):
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_generator_state(checkpoint_obj)
    num_pools = _infer_num_pools(state_dict)

    inc_key = "inc.double_conv.0.weight"
    if inc_key in state_dict:
        in_channels = state_dict[inc_key].shape[1]
        base_channels = state_dict[inc_key].shape[0]
    else:
        in_channels = 3
        base_channels = cfg.vec_unet_base_channels

    model = nnmodel.UNet3D(
        n_classes=in_channels,
        in_channels=in_channels,
        trilinear=True,
        base_channels=base_channels,
        num_pools=num_pools,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_pools


def _load_two_lpt_module() -> ModuleType:
    global _TWO_LPT_RUNTIME_MODULE
    if _TWO_LPT_RUNTIME_MODULE is not None:
        return _TWO_LPT_RUNTIME_MODULE

    module_path = Path(__file__).resolve().with_name("2LPT.py")
    if not module_path.exists():
        raise FileNotFoundError(f"2LPT helper file not found: {module_path}")

    module_name = "_dddf_two_lpt_runtime"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _TWO_LPT_RUNTIME_MODULE = module
    return module


def _seed_root_from_cfg(seed: int, nmesh: int) -> Path:
    wn_dir = Path(cfg.wn_dir_path(seed, nmesh))
    return wn_dir.parent


def _prepare_seed_products(
    seed: int,
    nmesh: int,
    force_rebuild: bool,
    force_nmesh: Optional[int] = None,
    lptic_dir: Optional[Path] = None,
    quijote_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    two_lpt = _load_two_lpt_module()
    seed_root = _seed_root_from_cfg(seed, nmesh)
    quijote_root_use = (
        quijote_root if quijote_root is not None else getattr(two_lpt, "DEFAULT_QUIJOTE_ROOT", None)
    )

    # Always refresh wn param from ICs/2LPT.param so inference uses path-fixed 2LPT_wn.param.
    two_lpt.write_wn_param_file(
        seed_root=seed_root,
        quijote_root=quijote_root_use,
        output_name="2LPT_wn.param",
    )

    two_lpt.generate_white_noise_for_root(
        seed_root=seed_root,
        lptic_dir=lptic_dir,
        quijote_root=quijote_root_use,
        force_overwrite=force_rebuild,
        skip_existing=not force_rebuild,
        reset_wn_dir=False,
    )

    expected_psi1_path = Path(cfg.wn_psi1_path(seed, nmesh))
    expected_qinit_path = Path(cfg.wn_qinit_path(seed, nmesh))

    need_psi_rebuild = force_rebuild or (not expected_psi1_path.exists()) or (not expected_qinit_path.exists())
    if need_psi_rebuild:
        tag = f"seed{seed:03d}"
        paths = two_lpt.discover_seed_paths(preferred_root=seed_root, fallback_root=seed_root)
        psi1_file, qinit_file = two_lpt.save_psi1_and_qinit_for_seed(
            paths,
            force_nmesh=force_nmesh,
            output_tag=tag,
        )
        expected_psi1_path = Path(psi1_file)
        expected_qinit_path = Path(qinit_file)

    if not expected_psi1_path.exists() or not expected_qinit_path.exists():
        raise FileNotFoundError(
            f"Missing psi1/q_init products for seed={seed}: {expected_psi1_path}, {expected_qinit_path}"
        )

    return expected_psi1_path, expected_qinit_path


def final_psi_cache_path(seed: int) -> Path:
    return Path(cfg.final_psi_path(seed, cfg.data_dir, cfg.L, cfg.N_p))


def _load_cached_final_psi(
    final_cache: Path,
    checkpoint_path: Optional[str],
    k_cut: Optional[float],
    k_width: Optional[float],
) -> Optional[np.ndarray]:
    if not final_cache.exists():
        return None

    with np.load(final_cache) as cached:
        if "final_psi" not in cached:
            return None

        if checkpoint_path is None:
            if k_cut is not None and "k_cut" in cached:
                cached_k_cut = float(np.asarray(cached["k_cut"]).ravel()[0])
                if not np.isclose(cached_k_cut, float(k_cut)):
                    return None
            if k_width is not None and "k_width" in cached:
                cached_k_width = float(np.asarray(cached["k_width"]).ravel()[0])
                if not np.isclose(cached_k_width, float(k_width)):
                    return None
            return np.array(cached["final_psi"])

        cached_ckpt = None
        if "checkpoint_path" in cached:
            cached_ckpt = str(np.asarray(cached["checkpoint_path"]).ravel()[0])

        if cached_ckpt is None:
            return None

        if str(Path(cached_ckpt)) == str(Path(checkpoint_path)):
            if k_cut is not None and "k_cut" in cached:
                cached_k_cut = float(np.asarray(cached["k_cut"]).ravel()[0])
                if not np.isclose(cached_k_cut, float(k_cut)):
                    return None
            if k_width is not None and "k_width" in cached:
                cached_k_width = float(np.asarray(cached["k_width"]).ravel()[0])
                if not np.isclose(cached_k_width, float(k_width)):
                    return None
            return np.array(cached["final_psi"])

    return None


def infer_final_psi_from_seed(
    seed: int,
    force_rebuild: bool = False,
    model: Optional[str] = None,
    infer_train_realizations: Optional[Sequence[int]] = None,
    infer_epochs: Optional[int] = None,
    k_cut: Optional[float] = None,
    k_width: Optional[float] = None,
    coef_file: Optional[str] = None,
    lptic_dir: Optional[Path] = None,
    quijote_root: Optional[Path] = None,
    force_nmesh: Optional[int] = None,
) -> np.ndarray:
    """Run the full seed inference chain and return only ``final_psi``.

    Parameters
    ----------
    seed : int
        Required realization/seed index.
    force_rebuild : bool
        True  -> force rebuild/overwrite wn, psi1, q_init, baseline cache, and final_psi cache.
        False -> reuse existing outputs and only generate missing artifacts.
    model, infer_train_realizations, infer_epochs : optional
        Checkpoint selection options, same semantics as previous verify pipeline.
    k_cut, k_width : optional
        Optional high-pass filter overrides. Defaults follow config values.
    coef_file : optional
        Forwarded to baseline computation.
    lptic_dir, quijote_root : optional
        Paths forwarded to the migrated 2LPT helper module.
    force_nmesh : optional
        Optional override passed to psi1/q_init generation.
    """
    importlib.reload(cfg)

    nmesh = cfg.N_p
    if force_nmesh is not None and int(force_nmesh) != int(nmesh):
        raise ValueError(
            f"force_nmesh={force_nmesh} must match cfg.N_p={nmesh} for this pipeline"
        )

    k_cut_use = float(cfg.k_cut if k_cut is None else k_cut)
    k_width_use = float(cfg.k_width if k_width is None else k_width)

    checkpoint_path = _resolve_checkpoint(
        model=model,
        infer_train_realizations=infer_train_realizations,
        infer_epochs=infer_epochs,
    )

    final_cache = final_psi_cache_path(seed)
    if not force_rebuild:
        cached_final = _load_cached_final_psi(
            final_cache=final_cache,
            checkpoint_path=checkpoint_path if model is not None else None,
            k_cut=k_cut_use,
            k_width=k_width_use,
        )
        if cached_final is not None:
            print(f"Using cached final_psi: {final_cache}")
            return cached_final

    psi1_path, qinit_path = _prepare_seed_products(
        seed=seed,
        nmesh=nmesh,
        force_rebuild=force_rebuild,
        force_nmesh=force_nmesh,
        lptic_dir=lptic_dir,
        quijote_root=quijote_root,
    )

    boxsize = cfg.boxsize
    grid_size = cfg.N_p
    mas = cfg.MAS

    dl = DDDF(cfg.Omega_m, cfg.threads)
    veck_main = dl.Veck(dl, cfg.N_p, boxsize, padding=0)

    wn_info = dl.get_snapshot_wn(psi1_path, qinit_path, boxsize, grid_size)
    q_init = wn_info["q_init"]
    init_delta = wn_info["delta"]

    final_info = dl.get_snapshot(
        cfg.final_snapshot_path(seed, cfg.N_p),
        cfg.snapshot_format(cfg.N_p),
        boxsize,
        grid_size,
    )
    target_psi_div, target_psi = dl.compute_target_psi_wn(
        q_init,
        final_info["pos"],
        cfg.N_p,
        boxsize,
        veck_main,
    )

    _, baseline_psi, _, _ = compute_baseline(
        dl,
        init_delta,
        target_psi_div,
        q_init,
        final_info["delta"],
        veck_main,
        cfg.N_p,
        boxsize,
        mas,
        seed,
        cfg.data_dir,
        cfg.L,
        coef_file=coef_file,
        overwrite=force_rebuild,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model, num_pools = _load_model(checkpoint_path, device)
    print(f"Loaded generator checkpoint: {checkpoint_path} (pools={num_pools}, device={device})")

    residual_pred = apply_model_to_field(
        baseline_psi,
        loaded_model,
        cfg.infer_patch_size,
        cfg.infer_padding,
        cfg.infer_overlap,
        device,
        batch_size=cfg.infer_batch_size,
    )
    residual_pred = highpass_vector_field(
        residual_pred,
        k_cut_use,
        boxsize,
        width=k_width_use,
    )

    final_psi = baseline_psi + residual_pred

    os.makedirs(final_cache.parent, exist_ok=True)
    np.savez_compressed(
        final_cache,
        final_psi=final_psi,
        realization=np.array([seed], dtype=np.int64),
        checkpoint_path=np.array([str(checkpoint_path)]),
        psi1_file=np.array([str(psi1_path)]),
        qinit_file=np.array([str(qinit_path)]),
        force_rebuild=np.array([int(force_rebuild)], dtype=np.int8),
        k_cut=np.array([k_cut_use], dtype=np.float64),
        k_width=np.array([k_width_use], dtype=np.float64),
        target_psi_mean=np.array([float(np.mean(target_psi))], dtype=np.float64),
    )
    print(f"Saved final_psi cache: {final_cache}")

    return final_psi
