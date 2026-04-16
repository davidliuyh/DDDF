from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import readgadget


@dataclass
class SeedPaths:
    preferred_root: Path
    wn_dir: Path
    white_noise_file: Path
    param_file: Path
    glass_file: Path
    pk_file: Path
    ic_ref: Path
    snapshot_z0: Path


def _strip_comment(line: str) -> str:
    if "%" in line:
        line = line[: line.index("%")]
    return line.strip()


def parse_param_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]
    return values


def load_param_config(path: Path) -> Dict[str, float]:
    p = parse_param_file(path)
    redshift = float(p.get("Redshift", "99"))
    return {
        "nmesh": float(p["Nmesh"]),
        "nsample": float(p["Nsample"]),
        "box": float(p["Box"]),
        "omega_m": float(p["Omega"]),
        "omega_l": float(p["OmegaLambda"]),
        "sigma8": float(p.get("Sigma8", "0.834")),
        "init_time": float(p.get("InitTime", 1.0 / (1.0 + redshift))),
    }


def discover_seed_paths(preferred_root: Path, fallback_root: Optional[Path] = None) -> SeedPaths:
    root = preferred_root
    wn_dir = root / "wn"

    white_noise_file = wn_dir / "white_noise.npz"
    if not white_noise_file.exists() and fallback_root is not None:
        cand = fallback_root / "wn" / "white_noise.npz"
        if cand.exists():
            root = fallback_root
            wn_dir = root / "wn"
            white_noise_file = cand

    if (wn_dir / "2LPT_wn.param").exists():
        param_file = wn_dir / "2LPT_wn.param"
    elif (wn_dir / "2LPT_wn_rund256.param").exists():
        param_file = wn_dir / "2LPT_wn_rund256.param"
    else:
        param_file = root / "ICs" / "2LPT.param"

    raw = parse_param_file(param_file)
    glass_file = Path(raw["GlassFile"]).expanduser()
    pk_file = Path(raw["FileWithInputSpectrum"]).expanduser()

    ic_ref = root / "ICs"
    snapshot_z0 = root / "snapdir_004"

    return SeedPaths(
        preferred_root=root,
        wn_dir=wn_dir,
        white_noise_file=white_noise_file,
        param_file=param_file,
        glass_file=glass_file,
        pk_file=pk_file,
        ic_ref=ic_ref,
        snapshot_z0=snapshot_z0,
    )


def _growth_factor(a: float, omega_m: float, omega_l: float) -> float:
    aa = np.linspace(1e-4, a, 4096)
    e = np.sqrt(omega_m / aa**3 + omega_l)
    integrand = 1.0 / (aa * e) ** 3
    integral = np.trapz(integrand, aa)
    return 2.5 * omega_m * np.sqrt(omega_m / a**3 + omega_l) * integral


def _read_power_table(pk_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(pk_file)
    k_h_mpc = data[:, 0].astype(np.float64)
    p_mpc3_h3 = data[:, 1].astype(np.float64)
    log10_k = np.log10(k_h_mpc)
    log10_delta2 = np.log10(4.0 * np.pi * k_h_mpc**3 * p_mpc3_h3)
    order = np.argsort(log10_k)
    return log10_k[order], log10_delta2[order]


def _power_spec_tabulated(
    k_internal: np.ndarray,
    log10_k_tab: np.ndarray,
    log10_d2_tab: np.ndarray,
    norm: float,
    unit_length_cm: float = 3.085678e21,
    input_spectrum_unit_cm: float = 3.085678e24,
) -> np.ndarray:
    k_hmpc = k_internal * (input_spectrum_unit_cm / unit_length_cm)
    logk = np.log10(np.maximum(k_hmpc, 1e-30))
    in_range = (logk >= log10_k_tab[0]) & (logk <= log10_k_tab[-1])

    log_delta2 = np.where(in_range, np.interp(logk, log10_k_tab, log10_d2_tab), -300.0)
    delta2 = np.power(10.0, log_delta2)

    k_safe = np.maximum(k_internal, 1e-30)
    return np.where(
        in_range & (k_internal > 0),
        norm * delta2 / (4.0 * np.pi * k_safe**3) / (8.0 * np.pi**3),
        0.0,
    )


def _sigma8_norm(log10_k_tab: np.ndarray, log10_d2_tab: np.ndarray, sigma8: float) -> float:
    unit_length_cm = 3.085678e21
    input_spectrum_unit_cm = 3.085678e24
    r8 = 8.0 * (input_spectrum_unit_cm / unit_length_cm)

    k_max = 500.0 / r8
    k_arr = np.linspace(1e-8, k_max, 100000)
    p_arr = _power_spec_tabulated(k_arr, log10_k_tab, log10_d2_tab, 1.0)

    kr = k_arr * r8
    kr2 = kr * kr
    kr3 = kr2 * kr
    w = np.where(kr > 1e-8, 3.0 * (np.sin(kr) / kr3 - np.cos(kr) / kr2), 0.0)
    sigma2 = np.trapz(4.0 * np.pi * k_arr**2 * w**2 * p_arr, k_arr)
    return sigma8**2 / sigma2


def _crop_rfft_cube(arr: np.ndarray, n_out: int) -> np.ndarray:
    n_in = arr.shape[0]
    if n_out == n_in:
        return arr.copy()

    hi = n_out // 2
    out = np.zeros((n_out, n_out, n_out // 2 + 1), dtype=arr.dtype)
    out[:hi, :hi, :] = arr[:hi, :hi, : n_out // 2 + 1]
    out[:hi, hi:, :] = arr[:hi, n_in - hi :, : n_out // 2 + 1]
    out[hi:, :hi, :] = arr[n_in - hi :, :hi, : n_out // 2 + 1]
    out[hi:, hi:, :] = arr[n_in - hi :, n_in - hi :, : n_out // 2 + 1]
    return out


def _kgrid(box: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = box / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    return np.meshgrid(kx, ky, kz, indexing="ij")


def _build_psi1_from_white_noise(
    white_noise_k: np.ndarray,
    box: float,
    log10_k_tab: np.ndarray,
    log10_d2_tab: np.ndarray,
    norm: float,
    dplus: float,
    nsample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = white_noise_k.shape[0]
    kx, ky, kz = _kgrid(box, n)
    k2 = kx * kx + ky * ky + kz * kz
    kval = np.sqrt(k2)

    pk = _power_spec_tabulated(kval.ravel(), log10_k_tab, log10_d2_tab, norm).reshape(kval.shape)
    fac = (2.0 * np.pi / box) ** 1.5
    delta = fac * np.sqrt(np.maximum(pk, 0.0)) / dplus

    k_nyq = nsample / 2
    mask_ns = (
        (np.abs(kx) * box / (2.0 * np.pi) <= k_nyq)
        & (np.abs(ky) * box / (2.0 * np.pi) <= k_nyq)
        & (np.abs(kz) * box / (2.0 * np.pi) <= k_nyq)
    )

    ikx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    ikz = np.fft.rfftfreq(n, d=1.0 / n).astype(int)
    ikx3, iky3, ikz3 = np.meshgrid(ikx, ikx, ikz, indexing="ij")
    valid = mask_ns & (k2 > 0)
    valid &= (ikx3 != n // 2) & (iky3 != n // 2) & (ikz3 != n // 2)
    valid &= ~((ikx3 == 0) & (iky3 == 0) & (ikz3 == 0))

    cdisp = [None, None, None]
    for ax, kv in enumerate((kx, ky, kz)):
        cd = np.zeros_like(white_noise_k, dtype=np.complex128)
        cd[valid] = 1j * kv[valid] / k2[valid] * delta[valid] * white_noise_k[valid]
        cdisp[ax] = cd

    n3 = float(n * n * n)
    psi1_x = (np.fft.irfftn(cdisp[0], s=(n, n, n)).real * n3).astype(np.float32)
    psi1_y = (np.fft.irfftn(cdisp[1], s=(n, n, n)).real * n3).astype(np.float32)
    psi1_z = (np.fft.irfftn(cdisp[2], s=(n, n, n)).real * n3).astype(np.float32)
    return psi1_x, psi1_y, psi1_z


def _load_glass_positions(glass_file: Path) -> Tuple[np.ndarray, float]:
    header = readgadget.header(str(glass_file))
    ptype = int(np.argmax(np.asarray(header.nall)))
    pos = readgadget.read_block(str(glass_file), "POS ", [ptype], verbose=False)
    return np.asarray(pos, dtype=np.float32), float(header.boxsize)


def _tile_glass_to_box(glass_pos: np.ndarray, glass_box: float, nsample: int, box: float) -> np.ndarray:
    n_glass = int(round(glass_pos.shape[0] ** (1.0 / 3.0)))
    tile = nsample // n_glass

    shifted = []
    for ix in range(tile):
        for iy in range(tile):
            for iz in range(tile):
                shift = np.array([ix, iy, iz], dtype=np.float32) * np.float32(glass_box)
                shifted.append(glass_pos + shift)

    tiled = np.concatenate(shifted, axis=0)
    scale = box / (glass_box * tile)
    return np.mod(tiled * np.float32(scale), np.float32(box)).astype(np.float32)


def save_psi1_and_qinit_for_seed(
    paths: SeedPaths,
    force_nmesh: Optional[int] = None,
    output_tag: str = "seed",
) -> Tuple[Path, Path]:
    cfg = load_param_config(paths.param_file)
    nsample_use = int(cfg["nsample"] if force_nmesh is None else force_nmesh)

    wn = np.load(paths.white_noise_file)
    wn_k = wn["white_noise"]
    wn_k_use = _crop_rfft_cube(wn_k, nsample_use)

    log10_k_tab, log10_d2_tab = _read_power_table(paths.pk_file)
    dplus = _growth_factor(1.0, cfg["omega_m"], cfg["omega_l"]) / _growth_factor(cfg["init_time"], cfg["omega_m"], cfg["omega_l"])
    norm = _sigma8_norm(log10_k_tab, log10_d2_tab, cfg["sigma8"])

    psi1_x, psi1_y, psi1_z = _build_psi1_from_white_noise(
        wn_k_use,
        cfg["box"],
        log10_k_tab,
        log10_d2_tab,
        norm=norm,
        dplus=dplus,
        nsample=nsample_use,
    )

    glass_pos, glass_box = _load_glass_positions(paths.glass_file)
    q_init = _tile_glass_to_box(glass_pos, glass_box, nsample=nsample_use, box=cfg["box"])

    psi1_file = paths.wn_dir / f"psi1_grid_z0_{output_tag}_n{nsample_use}.npz"
    qinit_file = paths.wn_dir / f"q_init_{output_tag}_n{nsample_use}.npz"

    np.savez_compressed(
        psi1_file,
        psi1_x=psi1_x,
        psi1_y=psi1_y,
        psi1_z=psi1_z,
        box=np.array([cfg["box"]], dtype=np.float64),
        dplus=np.array([dplus], dtype=np.float64),
    )
    np.savez_compressed(qinit_file, q_init=q_init, box=np.array([cfg["box"]], dtype=np.float64))

    return psi1_file, qinit_file


def _derivative(field: np.ndarray, axis: int, box: float) -> np.ndarray:
    n = field.shape[0]
    dx = box / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    if axis == 0:
        phase = (1j * k)[:, None, None]
    elif axis == 1:
        phase = (1j * k)[None, :, None]
    else:
        phase = (1j * k)[None, None, :]
    return np.fft.ifftn(phase * np.fft.fftn(field)).real.astype(np.float32)


def _build_psi2_from_psi1(psi1x: np.ndarray, psi1y: np.ndarray, psi1z: np.ndarray, box: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dxx = _derivative(psi1x, 0, box)
    dyy = _derivative(psi1y, 1, box)
    dzz = _derivative(psi1z, 2, box)
    dxy = _derivative(psi1x, 1, box)
    dxz = _derivative(psi1x, 2, box)
    dyz = _derivative(psi1y, 2, box)

    src = dxx * dyy + dxx * dzz + dyy * dzz - dxy * dxy - dxz * dxz - dyz * dyz
    n = src.shape[0]
    kx, ky, kz = _kgrid(box, n)
    k2 = kx * kx + ky * ky + kz * kz

    src_k = np.fft.rfftn(src)
    phi2_k = np.zeros_like(src_k, dtype=np.complex128)
    m = k2 > 0
    phi2_k[m] = -src_k[m] / k2[m]

    psi2x = np.fft.irfftn((-3.0 / 7.0) * 1j * kx * phi2_k, s=(n, n, n)).real.astype(np.float32)
    psi2y = np.fft.irfftn((-3.0 / 7.0) * 1j * ky * phi2_k, s=(n, n, n)).real.astype(np.float32)
    psi2z = np.fft.irfftn((-3.0 / 7.0) * 1j * kz * phi2_k, s=(n, n, n)).real.astype(np.float32)
    return psi2x, psi2y, psi2z


def _trilinear_interp_vector(q: np.ndarray, disp_x: np.ndarray, disp_y: np.ndarray, disp_z: np.ndarray, box: float) -> np.ndarray:
    n = disp_x.shape[0]
    cell = box / n

    gx = (q[:, 0] / cell) % n
    gy = (q[:, 1] / cell) % n
    gz = (q[:, 2] / cell) % n

    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    k0 = np.floor(gz).astype(np.int64)

    tx = gx - i0
    ty = gy - j0
    tz = gz - k0

    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    k1 = (k0 + 1) % n

    w000 = (1 - tx) * (1 - ty) * (1 - tz)
    w100 = tx * (1 - ty) * (1 - tz)
    w010 = (1 - tx) * ty * (1 - tz)
    w001 = (1 - tx) * (1 - ty) * tz
    w110 = tx * ty * (1 - tz)
    w101 = tx * (1 - ty) * tz
    w011 = (1 - tx) * ty * tz
    w111 = tx * ty * tz

    def interp(field: np.ndarray) -> np.ndarray:
        return (
            w000 * field[i0, j0, k0]
            + w100 * field[i1, j0, k0]
            + w010 * field[i0, j1, k0]
            + w001 * field[i0, j0, k1]
            + w110 * field[i1, j1, k0]
            + w101 * field[i1, j0, k1]
            + w011 * field[i0, j1, k1]
            + w111 * field[i1, j1, k1]
        )

    out = np.empty((q.shape[0], 3), dtype=np.float32)
    out[:, 0] = interp(disp_x)
    out[:, 1] = interp(disp_y)
    out[:, 2] = interp(disp_z)
    return out


def reconstruct_z127_displacement_only(psi1_file: Path, qinit_file: Path, param_file: Path) -> Dict[str, np.ndarray]:
    _ = param_file
    psi = np.load(psi1_file)
    qd = np.load(qinit_file)

    psi1x = psi["psi1_x"].astype(np.float32)
    psi1y = psi["psi1_y"].astype(np.float32)
    psi1z = psi["psi1_z"].astype(np.float32)
    q_init = qd["q_init"].astype(np.float32)
    box = float(np.asarray(psi["box"]).ravel()[0])

    psi2x, psi2y, psi2z = _build_psi2_from_psi1(psi1x, psi1y, psi1z, box)
    disp = _trilinear_interp_vector(q_init, psi1x + psi2x, psi1y + psi2y, psi1z + psi2z, box)
    x_rec = np.mod(q_init + disp, np.float32(box)).astype(np.float32)
    ids = np.arange(1, q_init.shape[0] + 1, dtype=np.uint32)

    return {"x_rec": x_rec, "ids": ids, "box": np.array([box], dtype=np.float64)}


def _load_hdf5_particles(path: Path, max_files: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    files = [path] if path.is_file() else sorted(path.glob("*.hdf5"))
    if max_files is not None:
        files = files[:max_files]

    pos_all = []
    ids_all = []
    for fp in files:
        with h5py.File(fp, "r") as h5:
            g = "PartType1" if "PartType1" in h5 else "PartType0"
            pos_all.append(h5[f"{g}/Coordinates"][:].astype(np.float32))
            ids_all.append(h5[f"{g}/ParticleIDs"][:].astype(np.uint32))

    return np.concatenate(pos_all, axis=0), np.concatenate(ids_all, axis=0)


def _match_particle_ids(query_ids: np.ndarray, ref_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ref_ids, kind="mergesort")
    ref_sorted = ref_ids[order]
    loc = np.searchsorted(ref_sorted, query_ids, side="left")
    valid = (loc < ref_sorted.size) & (ref_sorted[np.clip(loc, 0, ref_sorted.size - 1)] == query_ids)
    return np.nonzero(valid)[0].astype(np.int64), order[loc[valid]].astype(np.int64)


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = a - b
    d = np.where(d > box / 2.0, d - box, d)
    d = np.where(d < -box / 2.0, d + box, d)
    return d


def compare_z127_positions(recon_positions: np.ndarray, recon_ids: np.ndarray, ic_ref: Path, box: float, max_files: Optional[int] = 1) -> Dict[str, float]:
    ref_pos, ref_ids = _load_hdf5_particles(ic_ref, max_files=max_files)
    sel_rec, sel_ref = _match_particle_ids(recon_ids, ref_ids)

    d = _periodic_delta(recon_positions[sel_rec], ref_pos[sel_ref], box)
    dnorm = np.sqrt(np.sum(d * d, axis=1))
    return {
        "n_match": float(sel_rec.size),
        "pos_rms": float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
        "pos_max": float(np.max(dnorm)),
        "p50": float(np.percentile(dnorm, 50.0)),
        "p90": float(np.percentile(dnorm, 90.0)),
        "p99": float(np.percentile(dnorm, 99.0)),
    }


def compare_with_snapshot_z0(recon_positions: np.ndarray, recon_ids: np.ndarray, snapshot_dir: Path, box: float, max_files: Optional[int] = 1) -> Dict[str, float]:
    snap_pos, snap_ids = _load_hdf5_particles(snapshot_dir, max_files=max_files)
    sel_rec, sel_snap = _match_particle_ids(recon_ids, snap_ids)

    d = _periodic_delta(snap_pos[sel_snap], recon_positions[sel_rec], box)
    dnorm = np.sqrt(np.sum(d * d, axis=1))
    return {
        "n_match": float(sel_rec.size),
        "disp_rms": float(np.sqrt(np.mean(dnorm * dnorm))),
        "disp_max": float(np.max(dnorm)),
        "p50": float(np.percentile(dnorm, 50.0)),
        "p90": float(np.percentile(dnorm, 90.0)),
        "p99": float(np.percentile(dnorm, 99.0)),
    }


def _paint_cic_density(positions: np.ndarray, box: float, grid: int) -> np.ndarray:
    delta = np.zeros((grid, grid, grid), dtype=np.float32)
    try:
        import MAS_library as MASL  # type: ignore

        MASL.MA(np.ascontiguousarray(positions.astype(np.float32)), delta, box, "CIC", verbose=False)
    except Exception:
        cell = box / grid
        idx = np.floor(positions / cell).astype(np.int64) % grid
        np.add.at(delta, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0
    return delta


def _pk_numpy(delta: np.ndarray, box: float) -> Tuple[np.ndarray, np.ndarray]:
    n = delta.shape[0]
    dk = np.fft.rfftn(delta)
    pk3d = (np.abs(dk) ** 2) / float(n**6)

    dx = box / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    kval = np.sqrt(kx3 * kx3 + ky3 * ky3 + kz3 * kz3)

    kflat = kval.ravel()
    pflat = pk3d.ravel()
    kbins = np.linspace(0.0, np.max(kflat), n // 2 + 1)
    num, _ = np.histogram(kflat, bins=kbins, weights=pflat)
    den, _ = np.histogram(kflat, bins=kbins)
    p1d = np.where(den > 0, num / den, np.nan)
    k1d = 0.5 * (kbins[:-1] + kbins[1:])

    m = np.isfinite(p1d) & (k1d > 0)
    return k1d[m], p1d[m]


def compare_power_spectrum_z127(
    recon_positions: np.ndarray,
    recon_ids: np.ndarray,
    ic_ref: Path,
    box: float,
    max_files: Optional[int] = 1,
    grid: int = 256,
) -> Dict[str, np.ndarray]:
    ref_pos, ref_ids = _load_hdf5_particles(ic_ref, max_files=max_files)
    sel_rec, sel_ref = _match_particle_ids(recon_ids, ref_ids)

    rec = recon_positions[sel_rec]
    ref = ref_pos[sel_ref]
    delta_rec = _paint_cic_density(rec, box=box, grid=grid)
    delta_ref = _paint_cic_density(ref, box=box, grid=grid)

    try:
        import Pk_library as PKL  # type: ignore

        pk_rec_obj = PKL.Pk(delta_rec, box, axis=0, MAS="CIC", threads=16, verbose=False)
        pk_ref_obj = PKL.Pk(delta_ref, box, axis=0, MAS="CIC", threads=16, verbose=False)
        k = np.asarray(pk_ref_obj.k3D, dtype=np.float64)
        pk_ref = np.asarray(pk_ref_obj.Pk[:, 0], dtype=np.float64)
        pk_rec = np.asarray(pk_rec_obj.Pk[:, 0], dtype=np.float64)
    except Exception:
        k, pk_rec = _pk_numpy(delta_rec, box)
        k_ref, pk_ref = _pk_numpy(delta_ref, box)
        n = min(k.size, k_ref.size)
        k = k[:n]
        pk_rec = pk_rec[:n]
        pk_ref = pk_ref[:n]

    ratio = np.where(pk_ref > 1e-30, pk_rec / pk_ref, np.nan)
    return {
        "k_h_mpc": k * 1000.0,
        "pk_ratio": ratio,
        "pk_rec": pk_rec / 1e9,
        "pk_ref": pk_ref / 1e9,
    }
