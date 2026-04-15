from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import readgadget

try:
    import scipy.fft as _spfft  # type: ignore
except Exception:
    _spfft = None

try:
    import pyfftw.interfaces.numpy_fft as _fftw_fft  # type: ignore
    import pyfftw.interfaces.cache as _fftw_cache  # type: ignore

    _fftw_cache.enable()
except Exception:
    _fftw_fft = None


def _resolve_fft_threads() -> int:
    raw = os.environ.get("DDDF_FFT_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
            return max(1, n)
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


FFT_THREADS = _resolve_fft_threads()


def _fftn(a: np.ndarray) -> np.ndarray:
    if _spfft is not None:
        return _spfft.fftn(a, workers=FFT_THREADS)
    if _fftw_fft is not None:
        return _fftw_fft.fftn(a, threads=FFT_THREADS)
    return np.fft.fftn(a)


def _ifftn(a: np.ndarray) -> np.ndarray:
    if _spfft is not None:
        return _spfft.ifftn(a, workers=FFT_THREADS)
    if _fftw_fft is not None:
        return _fftw_fft.ifftn(a, threads=FFT_THREADS)
    return np.fft.ifftn(a)


def _rfftn(a: np.ndarray) -> np.ndarray:
    if _spfft is not None:
        return _spfft.rfftn(a, workers=FFT_THREADS)
    if _fftw_fft is not None:
        return _fftw_fft.rfftn(a, threads=FFT_THREADS)
    return np.fft.rfftn(a)


def _irfftn(a: np.ndarray, s: Tuple[int, int, int]) -> np.ndarray:
    if _spfft is not None:
        return _spfft.irfftn(a, s=s, workers=FFT_THREADS)
    if _fftw_fft is not None:
        return _fftw_fft.irfftn(a, s=s, threads=FFT_THREADS)
    return np.fft.irfftn(a, s=s)


@dataclass
class SeedPaths:
    preferred_root: Path
    wn_root: Path
    wn_dir: Path
    white_noise_file: Path
    param_file: Path
    glass_file: Path
    pk_file: Path
    ic_ref: Path
    snapshot_z0: Path


@dataclass
class ParamConfig:
    nmesh: int
    nsample: int
    box: float
    glass_tile_fac: int
    omega_m: float
    omega_l: float
    redshift: float
    init_time: float
    sigma8: float


@dataclass
class UnitAudit:
    position_unit: str
    box_size_raw: float
    box_size_mpc_h: float
    box_size_kpc_h: float
    camb_k_unit: str
    camb_p_unit: str
    internal_k_unit: str
    internal_p_unit: str


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
            if len(parts) < 2:
                continue
            key = parts[0]
            values[key] = parts[1]
    return values


def load_param_config(path: Path) -> ParamConfig:
    p = parse_param_file(path)

    def _f(key: str, default: Optional[float] = None) -> float:
        if key in p:
            return float(p[key])
        if default is None:
            raise KeyError(f"Missing key {key} in {path}")
        return float(default)

    return ParamConfig(
        nmesh=int(_f("Nmesh")),
        nsample=int(_f("Nsample")),
        box=_f("Box"),
        glass_tile_fac=int(_f("GlassTileFac")),
        omega_m=_f("Omega"),
        omega_l=_f("OmegaLambda"),
        redshift=_f("Redshift"),
        init_time=_f("InitTime", 1.0 / (1.0 + _f("Redshift"))),
        sigma8=_f("Sigma8", 0.834),
    )


def discover_seed_paths(
    preferred_root: Path = Path("/pscratch/sd/l/liuyh15/Quijote/fiducial_LR/0"),
    fallback_root: Path = Path("/pscratch/sd/l/liuyh15/Quijote/fiducial/0"),
) -> SeedPaths:
    roots = [preferred_root, fallback_root]

    wn_root = next((r for r in roots if (r / "wn" / "white_noise.npz").exists()), None)
    if wn_root is None:
        raise FileNotFoundError("Cannot locate white_noise.npz under fiducial_LR/0 or fiducial/0")

    wn_dir = wn_root / "wn"
    param_candidates = [wn_dir / "2LPT_wn_rund256.param", wn_dir / "2LPT_wn.param"]
    param_file = next((p for p in param_candidates if p.exists()), None)
    if param_file is None:
        raise FileNotFoundError(f"Cannot locate 2LPT_wn*.param in {wn_dir}")

    raw = parse_param_file(param_file)
    glass_file = Path(raw["GlassFile"]).expanduser()
    pk_file = Path(raw["FileWithInputSpectrum"]).expanduser()

    ic_ref_file = wn_dir / "ics_seed000_rund_n256.hdf5"
    if ic_ref_file.exists():
        ic_ref: Path = ic_ref_file
    elif (preferred_root / "ICs").exists():
        ic_ref = preferred_root / "ICs"
    elif (wn_root / "ICs").exists():
        ic_ref = wn_root / "ICs"
    else:
        raise FileNotFoundError("Cannot locate z=127 reference IC output")

    snapshot_z0 = preferred_root / "snapdir_004"
    if not snapshot_z0.exists():
        snapshot_z0 = wn_root / "snapdir_004"
    if not snapshot_z0.exists():
        raise FileNotFoundError("Cannot locate z=0 snapshot directory snapdir_004")

    return SeedPaths(
        preferred_root=preferred_root,
        wn_root=wn_root,
        wn_dir=wn_dir,
        white_noise_file=wn_dir / "white_noise.npz",
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


def growth_ratio(a_target: float, a_ref: float, omega_m: float, omega_l: float) -> float:
    return _growth_factor(a_target, omega_m, omega_l) / _growth_factor(a_ref, omega_m, omega_l)


def _read_power_table(pk_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read CAMB P(k) table, return (log10_k_hmpc, log10_Delta2).

    Matches C read_power_table: stores log10(k) in h/Mpc and
    log10(4 pi k^3 P(k)) for later interpolation.
    """
    data = np.loadtxt(pk_file)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid power spectrum table: {pk_file}")
    k_h_mpc = data[:, 0].astype(np.float64)
    p_mpc3_h3 = data[:, 1].astype(np.float64)
    log10_k = np.log10(k_h_mpc)
    log10_delta2 = np.log10(4.0 * np.pi * k_h_mpc**3 * p_mpc3_h3)
    order = np.argsort(log10_k)
    return log10_k[order], log10_delta2[order]


def _sigma8_norm(log10_k_tab: np.ndarray, log10_d2_tab: np.ndarray,
                 sigma8: float, unit_length_cm: float = 3.085678e21,
                 input_spectrum_unit_cm: float = 3.085678e24) -> float:
    """Compute Norm so that sigma(R=8 Mpc/h) = sigma8.

    Matches C initialize_powerspectrum + TopHatSigma2.
    """
    # R8 in internal units (kpc/h)
    R8 = 8.0 * (input_spectrum_unit_cm / unit_length_cm)  # 8 Mpc/h -> kpc/h

    # Integrate sigma^2(R8) with Norm=1.
    k_max = 500.0 / R8
    nk = 100000
    k_arr = np.linspace(1e-8, k_max, nk)
    # PowerSpec with Norm=1 in internal units
    p_arr = _power_spec_tabulated(k_arr, log10_k_tab, log10_d2_tab, 1.0,
                                   unit_length_cm, input_spectrum_unit_cm)
    kr = k_arr * R8
    kr2 = kr * kr
    kr3 = kr2 * kr
    w = np.where(kr > 1e-8, 3.0 * (np.sin(kr) / kr3 - np.cos(kr) / kr2), 0.0)
    integrand = 4.0 * np.pi * k_arr**2 * w**2 * p_arr
    sigma2 = np.trapz(integrand, k_arr)
    return sigma8**2 / sigma2


def _power_spec_tabulated(k_internal: np.ndarray,
                          log10_k_tab: np.ndarray,
                          log10_d2_tab: np.ndarray,
                          norm: float,
                          unit_length_cm: float = 3.085678e21,
                          input_spectrum_unit_cm: float = 3.085678e24,
                          ) -> np.ndarray:
    """Evaluate P(k) matching C PowerSpec_Tabulated.

    k_internal is in internal units (h/kpc).
    Returns P(k) in internal units (kpc/h)^3.
    """
    # convert k from internal (h/kpc) to h/Mpc for table lookup
    k_hmpc = k_internal * (input_spectrum_unit_cm / unit_length_cm)
    logk = np.log10(np.maximum(k_hmpc, 1e-30))

    in_range = (logk >= log10_k_tab[0]) & (logk <= log10_k_tab[-1])
    log_delta2 = np.where(
        in_range,
        np.interp(logk, log10_k_tab, log10_d2_tab),
        -300.0,
    )
    delta2 = np.power(10.0, log_delta2)

    # P = Norm * Delta2 / (4*pi*k^3) / (8*pi^3)  -- k in *internal* units
    k_safe = np.maximum(k_internal, 1e-30)
    P = np.where(
        in_range & (k_internal > 0),
        norm * delta2 / (4.0 * np.pi * k_safe**3) / (8.0 * np.pi**3),
        0.0,
    )
    return P


def audit_units(param_file: Path) -> UnitAudit:
    cfg = load_param_config(param_file)
    # Box in Quijote params is typically in kpc/h (e.g. 1e6 for 1 Gpc/h).
    box_kpc_h = float(cfg.box)
    box_mpc_h = box_kpc_h / 1000.0
    return UnitAudit(
        position_unit="kpc/h",
        box_size_raw=box_kpc_h,
        box_size_mpc_h=box_mpc_h,
        box_size_kpc_h=box_kpc_h,
        camb_k_unit="h/Mpc",
        camb_p_unit="(Mpc/h)^3",
        internal_k_unit="h/kpc",
        internal_p_unit="(kpc/h)^3",
    )


def _interp_pk_log(k: np.ndarray, k_tab: np.ndarray, p_tab: np.ndarray) -> np.ndarray:
    """Legacy log-log interpolation (unused in fixed pipeline)."""
    out = np.zeros_like(k, dtype=np.float64)
    m = k > 0
    if np.any(m):
        out[m] = np.exp(
            np.interp(np.log(k[m]), np.log(k_tab), np.log(p_tab), left=np.log(p_tab[0]), right=np.log(p_tab[-1]))
        )
    return out


def _crop_rfft_cube(arr: np.ndarray, n_out: int) -> np.ndarray:
    n_in = arr.shape[0]
    if n_out > n_in:
        raise ValueError(f"n_out={n_out} cannot exceed n_in={n_in}")
    if n_out == n_in:
        return arr.copy()

    if n_out % 2 != 0:
        raise ValueError("n_out must be even for symmetric crop")

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
    sphere_mode: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ZA displacement field from white noise, matching C 2LPTic.

    Returns psi1 at z=init (i.e. divided by Dplus, matching C convention).
    The output is the displacement field that can be used directly.
    """
    n = white_noise_k.shape[0]  # this is Nmesh
    kx, ky, kz = _kgrid(box, n)
    k2 = kx * kx + ky * ky + kz * kz
    kval = np.sqrt(k2)

    # P(k) in internal units, with sigma8 normalization
    pk = _power_spec_tabulated(kval.ravel(), log10_k_tab, log10_d2_tab, norm).reshape(kval.shape)

    # C amplitude: fac = (2*pi/Box)^{3/2}, delta = fac * sqrt(P(k)) / Dplus
    fac = (2.0 * np.pi / box) ** 1.5
    delta = fac * np.sqrt(np.maximum(pk, 0.0)) / dplus

    # Nsample cutoff: match C's cube-mode k-space truncation
    k_nyq = nsample / 2  # in units of 2π/Box
    if sphere_mode:
        mask_ns = (kval * box / (2.0 * np.pi)) <= k_nyq
    else:
        mask_ns = (
            (np.abs(kx) * box / (2.0 * np.pi) <= k_nyq)
            & (np.abs(ky) * box / (2.0 * np.pi) <= k_nyq)
            & (np.abs(kz) * box / (2.0 * np.pi) <= k_nyq)
        )
    # Also skip Nyquist planes and DC
    ikx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    ikz = np.fft.rfftfreq(n, d=1.0 / n).astype(int)
    IKX, IKY, IKZ = np.meshgrid(ikx, ikx, ikz, indexing="ij")
    nyquist_mask = (IKX == n // 2) | (IKY == n // 2) | (IKZ == n // 2)
    dc_mask = (IKX == 0) & (IKY == 0) & (IKZ == 0)
    valid = mask_ns & ~nyquist_mask & ~dc_mask & (k2 > 0)

    # Build displacement in k-space: cdisp[axes] = i * kvec/k² * delta * wn
    # (matches C sign convention for ZA displacement)
    # The white noise wn = re + i*im, and C computes:
    #   cdisp.re = -kvec/k² * delta * im
    #   cdisp.im =  kvec/k² * delta * re
    # which equals i * kvec/k² * delta * wn
    cdisp = [None, None, None]
    kvecs = [kx, ky, kz]
    for axes in range(3):
        cd = np.zeros_like(white_noise_k, dtype=np.complex128)
        cd[valid] = 1j * kvecs[axes][valid] / k2[valid] * delta[valid] * white_noise_k[valid]
        cdisp[axes] = cd

    # FFTW inverse does NOT divide by N³, but numpy irfftn does.
    # Multiply by N³ to match C convention.
    nmesh3 = float(n * n * n)
    psi1_x = (_irfftn(cdisp[0], s=(n, n, n)).real * nmesh3).astype(np.float32)
    psi1_y = (_irfftn(cdisp[1], s=(n, n, n)).real * nmesh3).astype(np.float32)
    psi1_z = (_irfftn(cdisp[2], s=(n, n, n)).real * nmesh3).astype(np.float32)
    return psi1_x, psi1_y, psi1_z


def _load_glass_positions(glass_file: Path) -> Tuple[np.ndarray, float]:
    header = readgadget.header(str(glass_file))
    nall = np.asarray(header.nall)
    ptype = int(np.argmax(nall))
    if nall[ptype] <= 0:
        raise ValueError(f"No particles found in glass file {glass_file}")
    pos = readgadget.read_block(str(glass_file), "POS ", [ptype], verbose=False)
    return np.asarray(pos, dtype=np.float32), float(header.boxsize)


def _tile_glass_to_box(glass_pos: np.ndarray, glass_box: float, nsample: int, box: float) -> np.ndarray:
    n_glass = int(round(glass_pos.shape[0] ** (1.0 / 3.0)))
    if n_glass**3 != glass_pos.shape[0]:
        raise ValueError("Glass particle count is not a perfect cube")
    if nsample % n_glass != 0:
        raise ValueError(f"nsample={nsample} is not divisible by glass resolution {n_glass}")

    tile = nsample // n_glass
    shifted: List[np.ndarray] = []
    for ix in range(tile):
        for iy in range(tile):
            for iz in range(tile):
                shift = np.array([ix, iy, iz], dtype=np.float32) * np.float32(glass_box)
                shifted.append(glass_pos + shift)
    tiled = np.concatenate(shifted, axis=0)

    scale = box / (glass_box * tile)
    q = tiled * np.float32(scale)
    return np.mod(q, np.float32(box)).astype(np.float32)


def _hash_array(arr: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def save_psi1_and_qinit_for_seed(
    paths: SeedPaths,
    force_nmesh: Optional[int] = None,
    output_tag: str = "seed",
) -> Tuple[Path, Path]:
    cfg = load_param_config(paths.param_file)

    wn = np.load(paths.white_noise_file)
    wn_k = wn["white_noise"]
    n_in = int(np.asarray(wn["nmesh"]).ravel()[0])
    wn_nsample = int(np.asarray(wn["nsample"]).ravel()[0])

    if force_nmesh is not None:
        if int(force_nmesh) != cfg.nsample:
            raise ValueError(
                f"force_nmesh={force_nmesh} is outside param-allowed Nsample={cfg.nsample}. "
                "Custom downsampling is disabled."
            )

    # Match C: k-space truncation at Nsample/2.
    # C runs FFT on Nmesh grid; for memory, we crop white noise to Nsample
    # and run FFT on the smaller grid. The Nsample cutoff in
    # _build_psi1_from_white_noise ensures identical k-mode content.
    if cfg.nmesh != n_in:
        raise ValueError(f"Param Nmesh={cfg.nmesh} but white_noise nmesh={n_in}; refusing implicit resampling")

    nsample_use = cfg.nsample
    nmesh_use = cfg.nsample  # FFT grid = Nsample for memory efficiency

    # Crop white noise from Nmesh to Nsample in k-space
    wn_k_use = _crop_rfft_cube(wn_k, nsample_use)

    # Read power spectrum table (C convention)
    log10_k_tab, log10_d2_tab = _read_power_table(paths.pk_file)

    # Compute Dplus = GrowthFactor(InitTime, 1.0) = growth(1.0)/growth(InitTime)
    # matching C convention: Dplus >> 1, used as divisor in amplitude
    dplus = _growth_factor(1.0, cfg.omega_m, cfg.omega_l) / _growth_factor(cfg.init_time, cfg.omega_m, cfg.omega_l)

    # Sigma8 normalization matching C
    norm = _sigma8_norm(log10_k_tab, log10_d2_tab, cfg.sigma8)
    print(f"[psi1] Norm={norm:.6g}, Dplus={dplus:.6g}, Nmesh={cfg.nmesh}, Nsample={nsample_use}")

    psi1_x, psi1_y, psi1_z = _build_psi1_from_white_noise(
        wn_k_use,
        cfg.box,
        log10_k_tab,
        log10_d2_tab,
        norm=norm,
        dplus=dplus,
        nsample=nsample_use,
    )

    glass_pos, glass_box = _load_glass_positions(paths.glass_file)
    q_init = _tile_glass_to_box(glass_pos, glass_box, nsample=nsample_use, box=cfg.box)

    psi1_file = paths.wn_dir / f"psi1_grid_z0_{output_tag}_n{nmesh_use}.npz"
    qinit_file = paths.wn_dir / f"q_init_{output_tag}_n{nsample_use}.npz"

    np.savez_compressed(
        psi1_file,
        psi1_x=psi1_x,
        psi1_y=psi1_y,
        psi1_z=psi1_z,
        box=np.array([cfg.box], dtype=np.float64),
        nmesh=np.array([nmesh_use], dtype=np.int32),
        nsample=np.array([nsample_use], dtype=np.int32),
        omega_m=np.array([cfg.omega_m], dtype=np.float64),
        omega_l=np.array([cfg.omega_l], dtype=np.float64),
        init_time=np.array([cfg.init_time], dtype=np.float64),
        dplus=np.array([dplus], dtype=np.float64),
        norm=np.array([norm], dtype=np.float64),
        source_white_noise=str(paths.white_noise_file),
        source_param=str(paths.param_file),
    )

    np.savez_compressed(
        qinit_file,
        q_init=q_init,
        box=np.array([cfg.box], dtype=np.float64),
        nsample=np.array([nsample_use], dtype=np.int32),
        sha256_q_init=np.array([_hash_array(q_init)]),
        source_glass=str(paths.glass_file),
        source_param=str(paths.param_file),
    )

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
    fk = _fftn(field)
    return _ifftn(phase * fk).real.astype(np.float32)


def _build_psi2_from_psi1(psi1x: np.ndarray, psi1y: np.ndarray, psi1z: np.ndarray, box: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dxx = _derivative(psi1x, 0, box)
    dyy = _derivative(psi1y, 1, box)
    dzz = _derivative(psi1z, 2, box)
    dxy = _derivative(psi1x, 1, box)
    dxz = _derivative(psi1x, 2, box)
    dyz = _derivative(psi1y, 2, box)

    source = dxx * dyy + dxx * dzz + dyy * dzz - dxy * dxy - dxz * dxz - dyz * dyz

    n = source.shape[0]
    kx, ky, kz = _kgrid(box, n)
    k2 = kx * kx + ky * ky + kz * kz

    src_k = _rfftn(source)
    phi2_k = np.zeros_like(src_k, dtype=np.complex128)
    m = k2 > 0
    phi2_k[m] = -src_k[m] / k2[m]

    psi2x_k = (-3.0 / 7.0) * 1j * kx * phi2_k
    psi2y_k = (-3.0 / 7.0) * 1j * ky * phi2_k
    psi2z_k = (-3.0 / 7.0) * 1j * kz * phi2_k

    psi2x = _irfftn(psi2x_k, s=(n, n, n)).real.astype(np.float32)
    psi2y = _irfftn(psi2y_k, s=(n, n, n)).real.astype(np.float32)
    psi2z = _irfftn(psi2z_k, s=(n, n, n)).real.astype(np.float32)
    return psi2x, psi2y, psi2z


def _trilinear_interp_vector(
    q: np.ndarray,
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    disp_z: np.ndarray,
    box: float,
) -> np.ndarray:
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


def reconstruct_z127_displacement_only(
    psi1_file: Path,
    qinit_file: Path,
    param_file: Path,
) -> Dict[str, np.ndarray]:
    """Reconstruct z=127 positions from stored psi1 and q_init.

    psi1 is stored at z=init scale (matching C convention),
    so no growth factor rescaling is needed.
    psi2 is computed from psi1 using the standard 2LPT formula.
    """
    cfg = load_param_config(param_file)

    psi = np.load(psi1_file)
    qd = np.load(qinit_file)

    psi1x = psi["psi1_x"].astype(np.float32)
    psi1y = psi["psi1_y"].astype(np.float32)
    psi1z = psi["psi1_z"].astype(np.float32)
    q_init = qd["q_init"].astype(np.float32)

    box = float(np.asarray(psi["box"]).ravel()[0])

    # psi1 is already at z=init scale, compute 2LPT correction
    psi2x, psi2y, psi2z = _build_psi2_from_psi1(psi1x, psi1y, psi1z, box)

    # Total displacement: psi1 + psi2 (psi2 already includes -3/7 factor)
    disp_particles = _trilinear_interp_vector(
        q_init,
        psi1x + psi2x,
        psi1y + psi2y,
        psi1z + psi2z,
        box,
    )
    x_rec = np.mod(q_init + disp_particles, np.float32(box)).astype(np.float32)

    ids = np.arange(1, q_init.shape[0] + 1, dtype=np.uint32)
    return {
        "q_init": q_init,
        "disp": disp_particles,
        "x_rec": x_rec,
        "ids": ids,
        "box": np.array([box], dtype=np.float64),
    }


def _load_hdf5_particles(path: Path, max_files: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.hdf5"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No hdf5 files found in {path}")

    pos_all: List[np.ndarray] = []
    ids_all: List[np.ndarray] = []
    for fp in files:
        with h5py.File(fp, "r") as h5:
            g = "PartType1" if "PartType1" in h5 else "PartType0"
            pos_all.append(h5[f"{g}/Coordinates"][:].astype(np.float32))
            ids_all.append(h5[f"{g}/ParticleIDs"][:].astype(np.uint32))
    return np.concatenate(pos_all, axis=0), np.concatenate(ids_all, axis=0)


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = a - b
    d = np.where(d > box / 2.0, d - box, d)
    d = np.where(d < -box / 2.0, d + box, d)
    return d


def compare_z127_positions(
    recon_positions: np.ndarray,
    recon_ids: np.ndarray,
    ic_ref: Path,
    box: float,
    max_files: Optional[int] = 1,
) -> Dict[str, float]:
    ref_pos, ref_ids = _load_hdf5_particles(ic_ref, max_files=max_files)

    map_ref = {int(pid): i for i, pid in enumerate(ref_ids.tolist())}
    sel_rec: List[int] = []
    sel_ref: List[int] = []
    for i, pid in enumerate(recon_ids.tolist()):
        j = map_ref.get(int(pid))
        if j is not None:
            sel_rec.append(i)
            sel_ref.append(j)
    if not sel_rec:
        return {
            "n_match": 0,
            "pos_rms": np.nan,
            "pos_max": np.nan,
        }

    r = recon_positions[np.asarray(sel_rec, dtype=np.int64)]
    t = ref_pos[np.asarray(sel_ref, dtype=np.int64)]
    d = _periodic_delta(r, t, box)

    rms = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    dnorm = np.sqrt(np.sum(d * d, axis=1))
    return {
        "n_match": float(len(sel_rec)),
        "pos_rms": rms,
        "pos_max": float(np.max(dnorm)),
        "p50": float(np.percentile(dnorm, 50.0)),
        "p90": float(np.percentile(dnorm, 90.0)),
        "p99": float(np.percentile(dnorm, 99.0)),
    }


def compare_with_snapshot_z0(
    recon_positions: np.ndarray,
    recon_ids: np.ndarray,
    snapshot_dir: Path,
    box: float,
    max_files: Optional[int] = 1,
) -> Dict[str, float]:
    snap_pos, snap_ids = _load_hdf5_particles(snapshot_dir, max_files=max_files)

    map_snap = {int(pid): i for i, pid in enumerate(snap_ids.tolist())}
    sel_rec: List[int] = []
    sel_snap: List[int] = []
    for i, pid in enumerate(recon_ids.tolist()):
        j = map_snap.get(int(pid))
        if j is not None:
            sel_rec.append(i)
            sel_snap.append(j)

    if not sel_rec:
        return {
            "n_match": 0,
            "disp_rms": np.nan,
            "disp_max": np.nan,
        }

    r = recon_positions[np.asarray(sel_rec, dtype=np.int64)]
    s = snap_pos[np.asarray(sel_snap, dtype=np.int64)]
    d = _periodic_delta(s, r, box)
    dnorm = np.sqrt(np.sum(d * d, axis=1))
    return {
        "n_match": float(len(sel_rec)),
        "disp_rms": float(np.sqrt(np.mean(dnorm * dnorm))),
        "disp_max": float(np.max(dnorm)),
        "p50": float(np.percentile(dnorm, 50.0)),
        "p90": float(np.percentile(dnorm, 90.0)),
        "p99": float(np.percentile(dnorm, 99.0)),
    }


def run_seed_pipeline(
    preferred_root: Path = Path("/pscratch/sd/l/liuyh15/Quijote/fiducial_LR/0"),
    force_nmesh: Optional[int] = None,
    output_tag: str = "seed",
    max_compare_files: int = 1,
) -> Dict[str, object]:
    paths = discover_seed_paths(preferred_root=preferred_root)
    psi1_file, qinit_file = save_psi1_and_qinit_for_seed(paths, force_nmesh=force_nmesh, output_tag=output_tag)
    recon = reconstruct_z127_displacement_only(psi1_file, qinit_file, paths.param_file)

    box = float(recon["box"][0])
    z127 = compare_z127_positions(recon["x_rec"], recon["ids"], paths.ic_ref, box=box, max_files=max_compare_files)
    z0 = compare_with_snapshot_z0(recon["x_rec"], recon["ids"], paths.snapshot_z0, box=box, max_files=max_compare_files)

    return {
        "paths": paths,
        "psi1_file": psi1_file,
        "qinit_file": qinit_file,
        "recon": recon,
        "z127_metrics": z127,
        "z0_metrics": z0,
    }


def _paint_cic_density(positions: np.ndarray, box: float, grid: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    delta = np.zeros((grid, grid, grid), dtype=np.float32)

    try:
        import MAS_library as MASL  # type: ignore

        MASL.MA(np.ascontiguousarray(positions), delta, box, "CIC", verbose=False)
        delta /= np.mean(delta, dtype=np.float64)
        delta -= 1.0
        return delta
    except Exception:
        # Fallback NGP assignment if MAS_library is unavailable.
        cell = box / grid
        idx = np.floor(positions / cell).astype(np.int64) % grid
        np.add.at(delta, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        delta /= np.mean(delta, dtype=np.float64)
        delta -= 1.0
        return delta


def _pk_numpy(delta: np.ndarray, box: float) -> Tuple[np.ndarray, np.ndarray]:
    n = delta.shape[0]
    dk = _rfftn(delta)
    pk3d = (np.abs(dk) ** 2) / float(n**6)

    dx = box / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    kval = np.sqrt(kx3 * kx3 + ky3 * ky3 + kz3 * kz3)

    kflat = kval.ravel()
    pflat = pk3d.ravel()
    nb = n // 2
    kbins = np.linspace(0.0, np.max(kflat), nb + 1)
    num, _ = np.histogram(kflat, bins=kbins, weights=pflat)
    den, _ = np.histogram(kflat, bins=kbins)

    with np.errstate(divide="ignore", invalid="ignore"):
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

    map_ref = {int(pid): i for i, pid in enumerate(ref_ids.tolist())}
    sel_rec: List[int] = []
    sel_ref: List[int] = []
    for i, pid in enumerate(recon_ids.tolist()):
        j = map_ref.get(int(pid))
        if j is not None:
            sel_rec.append(i)
            sel_ref.append(j)

    if not sel_rec:
        return {
            "k_h_mpc": np.array([], dtype=np.float64),
            "pk_ratio": np.array([], dtype=np.float64),
            "pk_rec": np.array([], dtype=np.float64),
            "pk_ref": np.array([], dtype=np.float64),
        }

    rec = recon_positions[np.asarray(sel_rec, dtype=np.int64)]
    ref = ref_pos[np.asarray(sel_ref, dtype=np.int64)]

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

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pk_ref > 1e-30, pk_rec / pk_ref, np.nan)

    # Convert k from h/kpc to h/Mpc for plotting readability.
    return {
        "k_h_mpc": k * 1000.0,
        "pk_ratio": ratio,
        "pk_rec": pk_rec / 1e9,
        "pk_ref": pk_ref / 1e9,
    }
