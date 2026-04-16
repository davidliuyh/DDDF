"""
NewDDDF: white-noise-based variant of DDDF.

Instead of reading Quijote ICs (z=127 2LPT snapshots), this class reads
white-noise-derived psi1 (z=127 scale) files and computes the linear density
field at z=0 by multiplying by dplus = D(z=0)/D(z=127).
growth_factor_D = 1 because the dplus scaling is handled inside get_snapshot_wn.
"""

import numpy as np
import pyfftw
import MAS_library as MASL
from dddf import DDDF


class NewDDDF(DDDF):
    """DDDF variant that works with white-noise-derived z=0 density fields."""

    def __init__(self, Omega_m, thread=16):
        # Skip the parent __init__ which needs zi/zf; set growth_factor_D=1
        # because psi1 is already scaled to z=0.
        self.growth_factor_D = 1.0
        self.threads = thread

    def get_snapshot_wn(self, psi1_path, qinit_path, boxsize, grid_size):
        """Load white-noise psi1 and q_init, return q_init, ids, and delta(z=0).

        Parameters
        ----------
        psi1_path : str or Path
            Path to ``psi1_grid_z127_seedXXX_nYYY.npz``.
        qinit_path : str or Path
            Path to ``q_init_seedXXX_nYYY.npz``.
        boxsize : float
            Box size in Mpc/h.
        grid_size : int
            Number of grid cells per side for the density field.

        Returns
        -------
        dict with keys ``q_init``, ``ids``, ``delta``
            - q_init : (N_part, 3) float32 – initial particle positions
            - ids    : (N_part,) int64 – particle IDs (0-based)
            - delta  : (grid_size, grid_size, grid_size) float32 – δ(z=0)
        """
        psi1_path = str(psi1_path)
        qinit_path = str(qinit_path)
        print(f'Loading psi1 from {psi1_path}')
        print(f'Loading q_init from {qinit_path}')

        psi = np.load(psi1_path)
        psi1_x = psi['psi1_x'].astype(np.float32)
        psi1_y = psi['psi1_y'].astype(np.float32)
        psi1_z = psi['psi1_z'].astype(np.float32)
        dplus = float(np.asarray(psi['dplus']).ravel()[0])
        box_file = float(np.asarray(psi['box']).ravel()[0])

        qd = np.load(qinit_path)
        q_init = qd['q_init'].astype(np.float32)
        ids = np.arange(q_init.shape[0], dtype=np.int64)

        N = psi1_x.shape[0]  # grid resolution of psi1 field
        print(f'psi1 grid: {N}^3,  dplus={dplus:.6f},  box_file={box_file} (kpc/h)')

        # Convert q_init from kpc/h to Mpc/h (pipeline convention)
        q_init = q_init / 1.0e3

        # Compute delta(z=0) = -div(dplus * psi1) via FFT
        # psi1 is at z=127 scale (in kpc/h); dplus = D(z=0)/D(z=127) ≈ 101
        # rescales to z=0.  box_file is in kpc/h — use it for consistent
        # units so the divergence is dimensionless.
        dx_kpc = box_file / N
        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx_kpc)
        ky = kx.copy()
        kz = kx.copy()
        kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing='ij')

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        fft_psi1_x = pyfftw.builders.fftn(dplus * psi1_x, threads=self.threads)()
        fft_psi1_y = pyfftw.builders.fftn(dplus * psi1_y, threads=self.threads)()
        fft_psi1_z = pyfftw.builders.fftn(dplus * psi1_z, threads=self.threads)()

        div_k = 1.0j * (kx3 * fft_psi1_x + ky3 * fft_psi1_y + kz3 * fft_psi1_z)
        delta = -pyfftw.builders.ifftn(div_k, threads=self.threads)().real.astype(self.real_dtype)

        print(f'delta(z=0): mean={delta.mean():.6e}, std={delta.std():.6f}')

        return dict(q_init=q_init, ids=ids, delta=delta)

    def compute_target_psi_div_wn(self, q_init, final_pos, N_p, boxsize, veck_main):
        """Compute target divergence of displacement Ψ from q_init → final_pos.

        Parameters
        ----------
        q_init : (N_part, 3) float32
            Initial (Lagrangian) particle positions.
        final_pos : (N_part, 3) float32
            Final (Eulerian) particle positions from N-body snapshot.
        N_p : int
            Grid size.
        boxsize : float
            Box size in Mpc/h.
        veck_main : Veck
            Pre-computed k-vector object.

        Returns
        -------
        ndarray (N_p, N_p, N_p) – divergence of displacement field
        """
        par_disp = np.asarray(final_pos - q_init, dtype=self.real_dtype)
        # Periodic wrapping
        par_disp = np.where(par_disp < -boxsize / 2, par_disp + boxsize, par_disp)
        par_disp = np.where(par_disp >  boxsize / 2, par_disp - boxsize, par_disp)

        disp_field = np.zeros((N_p, N_p, N_p, 3), dtype=self.real_dtype)
        self.disp_from_par(disp_field, q_init, par_disp, N_p, boxsize)
        return self.divergence(disp_field, veck_main)
