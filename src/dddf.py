import numpy as np
import readgadget
import bigfile
import MAS_library as MASL
import itertools
import pyfftw
from pylab import *
from scipy import integrate


class DDDF:

    real_dtype = np.float32
    complex_dtype = np.complex64

    def __init__(self, zi, zf, Omega_m, thread=16):
        self.growth_factor_D = self.linear_growth_factor(Omega_m, zf) / self.linear_growth_factor(Omega_m, zi)
        self.threads = thread

    class Veck:
        def __init__(self, dddf_obj, N=256, boxsize=1000.0, padding=0):
            self.N = N
            self.boxsize = boxsize
            self.padding = padding
            self.fft_freq = (
                np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
            )  # physical frequency
            self.Nyquist_freq = np.pi * N / boxsize  # Nyquist frequency
            self.boxsize_freq = 2 * np.pi / boxsize  # minimum valid frequency
            self.vec_k = np.array(
                list(itertools.product(self.fft_freq, self.fft_freq, self.fft_freq))
            ).reshape((N, N, N, 3))
            self.k_square = np.sum(self.vec_k**2, axis=3)
            self.k_mag = np.zeros((N,) * 3, dtype=dddf_obj.real_dtype)
            self.inv_k_square = np.zeros((N,) * 3, dtype=dddf_obj.real_dtype)
            non_zero_mask = self.k_square != 0
            self.k_mag[non_zero_mask] = np.sqrt(self.k_square[non_zero_mask])
            self.inv_k_square[non_zero_mask] = 1 / self.k_square[non_zero_mask]
            self.mask = np.zeros((N, N, N), dtype=bool)
            self.mask[
                padding : N - padding, padding : N - padding, padding : N - padding
            ] = True
            self.masked_shape = (N - 2 * padding,) * 3

    def get_snapshot(self, snapshot, filetype, boxsize, grid_size):
        print("Reading snapshot...")

        ptype = [1]  # [1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

        if filetype == "gadget":
            # read header
            header = readgadget.header(snapshot)
            boxsize = header.boxsize / 1e3  # Mpc/h

            # read positions, velocities and IDs of the parts
            pos = (
                np.array(
                    readgadget.read_block(
                        # positions in Mpc/h
                        snapshot,
                        "POS ",
                        ptype,
                    )
                    / 1e3
                )
                % boxsize
            ).astype(self.real_dtype)

            ids = np.array(
                readgadget.read_block(snapshot, "ID  ", ptype) - 1
            )  # start from 0
        elif filetype == "bigfile":
            # Open a BigFile snapshot directory
            bf = bigfile.File(snapshot)
            # Access one dataset
            pos = bf["Position"][:]
            ids = bf["ID"][:]

        # sort
        correct_ind = np.argsort(ids)
        pos = pos[correct_ind]
        ids = ids[correct_ind]

        delta = np.zeros((grid_size, grid_size, grid_size), dtype=self.real_dtype)

        # construct 3D density field
        MASL.MA(pos, delta, boxsize, "CIC", verbose=False)
        delta = delta / np.mean(delta) - 1.0

        return dict(pos=pos, ids=ids, delta=delta)  # return pos with shape(3,N)

    def divergence(self, x_field: np.typing.NDArray, veck: Veck) -> np.typing.NDArray:
        assert x_field.shape == veck.vec_k.shape
        print("Computing divergence...")

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        k_field = np.empty_like(x_field, dtype=self.complex_dtype)  # (N, N, N, 3)
        for i in range(3):
            k_field[:, :, :, i] = pyfftw.builders.fftn(
                x_field[:, :, :, i], threads=self.threads
            )()

        div = pyfftw.builders.ifftn(
            1.0j * np.einsum("ijkl,ijkl->ijk", veck.vec_k, k_field),
            threads=self.threads,
        )().real

        return div

    def linear_growth_factor(self, Omega_m: float, z: float):
        Omega_Lambda = 1 - Omega_m

        def Hubble(a):
            return (Omega_m / a**3 + Omega_Lambda) ** 0.5

        def integrand(a):
            return 1 / (a * Hubble(a)) ** 3

        integral, error = integrate.quad(integrand, 0, 1 / (1 + z))

        return 2.5 * Omega_m * Hubble(1 / (1 + z)) * integral

    def div_psi_1(self, x_delta: np.typing.NDArray) -> np.typing.NDArray:
        # return -self.growth_factor_D * x_delta
        return -self.growth_factor_D * x_delta

    def div_psi_2(self, x_delta: np.typing.NDArray, veck: Veck) -> np.typing.NDArray:
        print("Computing sec order displacement...")

        def _k_phi(x_delta: np.typing.NDArray) -> np.typing.NDArray:

            k_delta = pyfftw.builders.fftn(x_delta, threads=self.threads)()
            k_phi = -veck.inv_k_square * k_delta

            return k_phi

        def _grad_phi_ij(
            k_phi: np.typing.NDArray, i: int8, j: int8
        ) -> np.typing.NDArray:

            x_grad = pyfftw.builders.ifftn(
                -veck.vec_k[:, :, :, i] * veck.vec_k[:, :, :, j] * k_phi,
                threads=self.threads,
            )().real

            return x_grad

        k_phi = _k_phi(x_delta)
        sec_order_source = 0
        for i in range(3):
            for j in range(i):
                sec_order_source += (
                    _grad_phi_ij(k_phi, i, i) * _grad_phi_ij(k_phi, j, j)
                    - _grad_phi_ij(k_phi, i, j) ** 2
                )

        div_psi_2 = -3 / 7 * self.growth_factor_D**2 * sec_order_source

        return div_psi_2

    def disp_from_par(self, disp_field, par_init_pos, par_disp, N_p, boxsize):
        # compute the displacement field from particle displacements using CIC
        # par_init_pos: (N, 3)-like, par_disp: (N, 3)-like
        print("Computing displacement field using CIC...")
        grid_sep = boxsize / N_p
        par_grid_ind = (par_init_pos // grid_sep).astype(np.int16)
        disp_frac_from_grid = par_init_pos / grid_sep - par_grid_ind

        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    current_par_grid_ind = par_grid_ind.copy()
                    current_par_grid_ind[:, 0] += dx
                    current_par_grid_ind[:, 1] += dy
                    current_par_grid_ind[:, 2] += dz
                    current_par_grid_ind %= N_p
                    weight_x = (
                        1 - disp_frac_from_grid[:, 0]
                        if dx == 0
                        else disp_frac_from_grid[:, 0]
                    )
                    weight_y = (
                        1 - disp_frac_from_grid[:, 1]
                        if dy == 0
                        else disp_frac_from_grid[:, 1]
                    )
                    weight_z = (
                        1 - disp_frac_from_grid[:, 2]
                        if dz == 0
                        else disp_frac_from_grid[:, 2]
                    )
                    frac_prod = weight_x * weight_y * weight_z
                    assert np.all(frac_prod >= 0) and np.all(frac_prod <= 1)
                    np.add.at(
                        disp_field,
                        tuple(current_par_grid_ind.T),
                        par_disp * frac_prod[:, np.newaxis],
                    )

    def assign_disp(
        self,
        par_disp: np.typing.NDArray,
        par_pos: np.typing.NDArray,
        disp: np.typing.NDArray,
        N_p,
        boxsize,
    ):
        # assign displacements for particles knowing their initial positions and the displacement field using CIC
        # par_disp: particle displacement (N,3), initialized to be zero; par_pos: particle positions (N, 3); disp: displacement field (N_p, N_p, N_p, 3)
        print("Assigning displacements to particles using CIC...")
        grid_sep = boxsize / N_p
        par_grid_ind = (par_pos // grid_sep).astype(np.int16)
        disp_frac_from_grid = par_pos / grid_sep - par_grid_ind

        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    current_par_grid_ind = par_grid_ind.copy()
                    current_par_grid_ind[:, 0] += dx
                    current_par_grid_ind[:, 1] += dy
                    current_par_grid_ind[:, 2] += dz
                    current_par_grid_ind %= N_p
                    weight_x = (
                        1 - disp_frac_from_grid[:, 0]
                        if dx == 0
                        else disp_frac_from_grid[:, 0]
                    )
                    weight_y = (
                        1 - disp_frac_from_grid[:, 1]
                        if dy == 0
                        else disp_frac_from_grid[:, 1]
                    )
                    weight_z = (
                        1 - disp_frac_from_grid[:, 2]
                        if dz == 0
                        else disp_frac_from_grid[:, 2]
                    )
                    frac_prod = weight_x * weight_y * weight_z
                    assert np.all(frac_prod >= 0) and np.all(frac_prod <= 1)
                    par_disp += (
                        frac_prod[:, np.newaxis] * disp[tuple(current_par_grid_ind.T)]
                    )

    def disp_from_psi_div(self, psi_div: np.typing.NDArray, veck: Veck, N_p):
        # psi_div: divergence of displacement field (N_p, N_p, N_p)
        disp = np.empty((N_p, N_p, N_p, 3), dtype=self.real_dtype)
        for i in range(3):
            disp[:, :, :, i] = pyfftw.builders.ifftn(
                -1.0j
                * veck.vec_k[:, :, :, i]
                * veck.inv_k_square
                * pyfftw.builders.fftn(psi_div, threads=self.threads)(),
                threads=self.threads,
            )().real
        return disp

    def par_pos_from_psi_div(
        self,
        psi_div: np.typing.NDArray,
        par_init_pos: np.typing.NDArray,
        veck: Veck,
        N_p,
        boxsize,
    ):
        # psi_div: divergence of displacement field (N_p, N_p, N_p); par_init_pos: particles initial positions (N,3)
        disp = self.disp_from_psi_div(psi_div, veck, N_p)
        par_disp = np.zeros((N_p**3, 3), dtype=self.real_dtype)
        self.assign_disp(par_disp, par_init_pos, disp, N_p, boxsize)
        par_pos = par_disp + par_init_pos
        par_pos %= boxsize
        par_pos = np.where(par_pos>=boxsize, 0.0, par_pos)
        return par_pos

    def div_exp(self, x_field: np.typing.NDArray, veck: Veck, kc) -> np.typing.NDArray:
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        k_field = np.empty_like(x_field, dtype=self.complex_dtype)  # (N, N, N, 3)
        for i in range(3):
            k_field[:, :, :, i] = pyfftw.builders.fftn(
                x_field[:, :, :, i], threads=self.threads
            )()

        div_exp = pyfftw.builders.ifftn(
            1.0j
            * np.einsum(
                "ijkl,ijkl->ijk",
                veck.vec_k * np.exp(-veck.k_square / (2 * kc**2))[..., None],
                k_field,
            ),
            threads=self.threads,
        )().real

        return div_exp

    def div_butterworth(self, x_field, veck: Veck, n, kc):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        k_field = np.empty_like(x_field, dtype=self.complex_dtype)  # (N, N, N, 3)
        for i in range(3):
            k_field[:, :, :, i] = pyfftw.builders.fftn(
                x_field[:, :, :, i], threads=self.threads
            )()

        div_butterworth = pyfftw.builders.ifftn(
            1.0j
            * np.einsum(
                "ijkl,ijkl->ijk",
                veck.vec_k / (1 + (veck.vec_k / kc) ** (2 * n)) ** 0.5,
                k_field,
            ),
            threads=self.threads,
        )().real

        return div_butterworth

    def div_complement_butterworth(self, x_field, veck: Veck, n, kc, a):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        k_field = np.empty_like(x_field, dtype=self.complex_dtype)  # (N, N, N, 3)
        for i in range(3):
            k_field[:, :, :, i] = pyfftw.builders.fftn(
                a - x_field[:, :, :, i], threads=self.threads
            )()

        div_butterworth = pyfftw.builders.ifftn(
            1.0j
            * np.einsum(
                "ijkl,ijkl->ijk",
                veck.vec_k / (1 + (veck.vec_k / kc) ** (2 * n)) ** 0.5,
                k_field,
            ),
            threads=self.threads,
        )().real

        return div_butterworth

    def div_ALPT(self, x_field_2LPT, veck: Veck, rs, delta1):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)

        k_field = np.empty_like(x_field_2LPT, dtype=self.complex_dtype)  # (N, N, N, 3)
        for i in range(3):
            k_field[:, :, :, i] = pyfftw.builders.fftn(
                x_field_2LPT[:, :, :, i], threads=self.threads
            )()

        # k_kernel = (2 * np.pi * rs ** 2) ** 1.5 * np.exp(-rs ** 2 * k_square / 2)
        k_kernel = np.exp(-(rs**2) * veck.k_square / 2)
        div_psi_L = pyfftw.builders.ifftn(
            1.0j
            * np.einsum("ijkl,ijkl->ijk", veck.vec_k * k_kernel[..., None], k_field),
            threads=self.threads,
        )().real
        div_psi_SC = np.where(
            (1 - 2 * self.growth_factor_D * delta1 / 3) > 0,
            3 * ((1 - 2 * self.growth_factor_D * delta1 / 3) ** 0.5 - 1),
            -3,
        )
        k_div_psi_SC = pyfftw.builders.fftn(div_psi_SC, threads=self.threads)()
        div_psi_S = (
            div_psi_SC
            - pyfftw.builders.ifftn(
                k_kernel * k_div_psi_SC, threads=self.threads
            )().real
        )

        return div_psi_L + div_psi_S

    def div_SC(self, delta):
        div_psi_SC = np.where(
            (1 - 2 * self.growth_factor_D * delta / 3) > 0,
            3 * ((1 - 2 * self.growth_factor_D * delta / 3) ** 0.5 - 1),
            -3,
        )
        return div_psi_SC

    def div_nexp(self, delta, n):
        return (np.exp(delta) - 1) ** n

    def solve_best_fit(self, psi_div_list):
        N = psi_div_list[0].shape[0]
        field_num = len(psi_div_list)
        fit_coef = np.zeros((field_num, field_num))
        for i in range(field_num):
            for j in range(i + 1):
                fit_coef[i, j] = (
                    np.einsum("ijk,ijk->", psi_div_list[i], psi_div_list[j]) / N**3
                )
                fit_coef[j, i] = fit_coef[i, j]
        return np.linalg.solve(fit_coef[1:, 1:], fit_coef[0, 1:])

    def solve_best_fit_with_weight(self, psi_div_list, weight):
        N = psi_div_list[0].shape[0]
        field_num = len(psi_div_list)
        fit_coef = np.zeros((field_num, field_num))
        for i in range(field_num):
            for j in range(i + 1):
                fit_coef[i, j] = (
                    np.einsum("ijk,ijk,ijk->", psi_div_list[i], psi_div_list[j], weight)
                    / N**3
                )
                fit_coef[j, i] = fit_coef[i, j]
        return np.linalg.solve(fit_coef[1:, 1:], fit_coef[0, 1:])

    def cut_field(self, field, ind=(0, 0, 0), layer=0, padding=0):
        """
        cut the field into 2 ** (3n) subfields
        field: (N, N, N, M)
        """
        ind = np.array(ind)
        if layer == 0:
            return field
        N = field.shape[0]
        sub_N = int(N / 2**layer)
        start_grid = ind * sub_N - padding
        end_grid = (ind + 1) * sub_N + padding
        slices = np.array([np.arange(start_grid[i], end_grid[i]) for i in range(3)])
        slices[(slices >= N) | (slices < -N)] = (
            slices[(slices >= N) | (slices < -N)] % N
        )
        mesh = np.ix_(*slices)
        return field[mesh]

    def add_funcs(self, sub_delta, veck: Veck):
        # return func list without padding
        func_list = []
        psi_div_1 = self.div_psi_1(sub_delta)
        psi_div_2 = self.div_psi_2(sub_delta, veck)
        func_list.append(psi_div_1)
        func_list.append(psi_div_2)
        ZA_disp = self.disp_from_psi_div(psi_div_1, veck, veck.mask.shape[0])
        # LPT2_disp = self.disp_from_psi_div(
        #     psi_div_1 + psi_div_2, veck, veck.mask.shape[0]
        # )
        for kc in np.logspace(-3, 2, 9):
            func_list.append(self.div_exp(ZA_disp, veck, kc))
            # for n in np.arange(1, 4):
            #     func_list.append(self.div_butterworth(ZA_disp, veck, n, kc))
            #     func_list.append(self.div_complement_butterworth(
            #         ZA_disp, veck, n, kc, 1))
        # for rs in np.linspace(0.1, 10, 10):
        #     func_list.append(div_ALPT(LPT2_disp, veck, rs, sub_delta))
        # func_list.append(div_SC(sub_delta))
        # for n in np.arange(2, 8):
        #     func_list.append(self.div_nexp(sub_delta, n))
        return [func[veck.mask].reshape(*veck.masked_shape) for func in func_list]

    def stack_layer(self, avg_best_fit_coef, psi_div_dict, layer):
        print(f"Stacking layer {layer}...")
        subN = psi_div_dict[(0, 0, 0)][0].shape[0]
        stacked = np.zeros((subN * 2**layer,) * 3)
        for i, j, k in itertools.product(range(2**layer), repeat=3):
            current_best_fit_psi_div = np.einsum(
                "l,lijk->ijk", avg_best_fit_coef, psi_div_dict[(i, j, k)]
            )
            stacked[
                i * subN : (i + 1) * subN,
                j * subN : (j + 1) * subN,
                k * subN : (k + 1) * subN,
            ] = current_best_fit_psi_div
        return stacked

    def k_gaussian_filter(self, k0, r0, veck: Veck):
        if r0 == 0:
            return 1
        return np.exp(-((veck.k_mag - k0) ** 2) * r0**2 / 2)

    def k_log_gaussian_filter(self, k0, r0, veck: Veck):
        assert k0 != 0
        return np.exp(-((np.log(veck.k_mag) - np.log(k0)) ** 2) * r0**2 / 2)

    def k_trapezoid_filter(self, p1, p2, p3, p4, h, veck: Veck):
        assert (h > 0) & (p4 > p3) & (p3 >= p2) & (p2 > p1)
        trapz_filter = np.zeros_like(veck.k_mag)
        m1 = (veck.k_mag > p1) & (veck.k_mag < p2)
        m2 = (veck.k_mag >= p2) & (veck.k_mag <= p3)
        m3 = (veck.k_mag > p3) & (veck.k_mag < p4)
        trapz_filter[m1] = h * (veck.k_mag[m1] - p1) / (p2 - p1)
        trapz_filter[m2] = h
        trapz_filter[m3] = h * (p4 - veck.k_mag[m3]) / (p4 - p3)
        return trapz_filter

    def k_tophat_filter(self, p1, p2, veck: Veck):
        assert p1 < p2
        return np.asarray((veck.k_mag > p1) & (veck.k_mag < p2), dtype=self.real_dtype)

    def k_end_filter(self, veck: Veck):
        return np.ones_like(veck.k_mag)

    def kx2x_convolve(self, kf1, xf2):
        kf2 = pyfftw.builders.fftn(xf2, threads=self.threads)()
        return pyfftw.builders.ifftn(kf1 * kf2, threads=self.threads)().real

    def kx2k_convolve(self, kf1, xf2):
        kf2 = pyfftw.builders.fftn(xf2, threads=self.threads)()
        return kf1 * kf2

    def k_solve_best_fit(self, k_psi_div_list):
        """
        fit amplitude
        """
        N = k_psi_div_list[0].shape[0]
        field_num = len(k_psi_div_list)
        fit_coef = np.zeros((field_num, field_num))
        for i in range(field_num):
            for j in range(i + 1):
                fit_coef[i, j] = (
                    np.einsum(
                        "ijk,ijk->", k_psi_div_list[i], np.conjugate(k_psi_div_list[j])
                    ).real
                    / N**3
                )
                fit_coef[j, i] = fit_coef[i, j]
        return np.linalg.solve(fit_coef[1:, 1:], fit_coef[0, 1:])

    def to_k(self, x_field):
        assert len(x_field.shape) == 3
        return pyfftw.builders.fftn(x_field)()
