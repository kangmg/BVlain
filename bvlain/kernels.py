""" Compiled kernels for the BVSE energy grid.

The energy of a mobile ion at each grid point is a sum of pair terms over all
framework ions (and their periodic images) closer than r_cut. The grid is
regular and periodic, so instead of searching neighbors of each grid point
every framework ion of the unit cell adds its contribution to the grid points
within r_cut around it. No supercell and no (N_grid, k) arrays are needed.
"""

import math
import threading
import numpy as np
from numba import njit, prange
from scipy.special import erfc


KIND_NONE, KIND_MORSE, KIND_COULOMB = 0, 1, 2

# distances are not allowed to be smaller than this value (Angstrom),
# keeps the energy finite if a grid point coincides with an ion
R_FLOOR = 1e-12

# tabulated pair energies are used above this distance (Angstrom),
# below it the pair energy is calculated explicitly
R_EXACT = 0.5

# parallel kernels should not be launched from several python threads at once
_LOCK = threading.Lock()



def cached_njit(**options):

    """ numba.njit with on-disk caching if a cache directory is available """

    def decorator(function):
        try:
            return njit(cache = True, **options)(function)
        except RuntimeError:
            return njit(**options)(function)
    return decorator



def pair_params(q, r_min, alpha, d0, r_c, n, q_mi, rc_mi, n_mi, f = 0.74):

    """ Collect parameters of pair interactions between mobile and framework ions.

    Morse-type interaction is calculated for the ions with tabulated
    r_min, alpha, d0 (ions of opposite sign compared to mobile ion),
    Coulombic interaction for the ions of the same sign as mobile ion.

    Parameters
    ----------
    q, r_min, alpha, d0, r_c, n: np.array of floats
        oxidation states, Morse parameters, covalent radii
        and principle quantum numbers of framework ions

    q_mi, rc_mi, n_mi: float
        the same for a mobile ion

    f: float, 0.74 by default
        screening factor

    Returns
    ----------
    kind: np.array of int
        type of interaction, 0 - none, 1 - Morse, 2 - Coulomb

    p1, p2, p3: np.array of floats
        Morse: d0 / 2, alpha, r_min
        Coulomb: 14.4 * q_mi * q / sqrt(n_mi * n), 1 / (f * (rc_mi + r_c)), 0
    """

    q = np.asarray(q, dtype = np.float64)
    r_min = np.asarray(r_min, dtype = np.float64)
    alpha = np.asarray(alpha, dtype = np.float64)
    d0 = np.asarray(d0, dtype = np.float64)
    r_c = np.asarray(r_c, dtype = np.float64)
    n = np.asarray(n, dtype = np.float64)

    morse = ~(np.isnan(r_min) | np.isnan(alpha) | np.isnan(d0))
    coulomb = (q * q_mi > 0) & ~morse
    kind = np.where(morse, KIND_MORSE, np.where(coulomb, KIND_COULOMB, KIND_NONE)).astype(np.int64)
    with np.errstate(all = 'ignore'):
        p1 = np.where(morse, d0 / 2, np.where(coulomb, 14.4 * q_mi * q / np.sqrt(n_mi * n), 0.0))
        p2 = np.where(morse, alpha, np.where(coulomb, 1 / (f * (rc_mi + r_c)), 0.0))
        p3 = np.where(morse, r_min, 0.0)
    return kind, np.nan_to_num(p1), np.nan_to_num(p2), np.nan_to_num(p3)



def pair_energy(r, kind, p1, p2, p3):

    """ Pair energy as a function of distance, see BVSEPotential """

    r = np.asarray(r, dtype = np.float64)
    if kind == KIND_MORSE:
        return p1 * ((np.exp(p2 * (p3 - r)) - 1) ** 2 - 1)
    if kind == KIND_COULOMB:
        return p1 / r * erfc(p2 * r)
    return np.zeros_like(r)



def pair_energy_derivative(r, kind, p1, p2, p3):

    """ Derivative of the pair energy with respect to distance """

    r = np.asarray(r, dtype = np.float64)
    if kind == KIND_MORSE:
        x = np.exp(p2 * (p3 - r))
        return -2 * p1 * p2 * x * (x - 1)
    if kind == KIND_COULOMB:
        return -p1 * (2 * p2 / np.sqrt(np.pi) * np.exp(-(p2 * r) ** 2) / r + erfc(p2 * r) / r ** 2)
    return np.zeros_like(r)



def build_tables(kind, p1, p2, p3, r_cut, size = 4096):

    """ Tabulate pair energies on a uniform grid of distances from 0 to r_cut.

    Cubic Hermite interpolation is used within each interval:
    E(r) = c0 + t * (c1 + t * (c2 + t * c3)), t = r / dr - i

    Parameters
    ----------
    kind, p1, p2, p3: np.array
        parameters of pair interactions for each framework ion, see pair_params

    r_cut: float
        cutoff radius

    size: int, 4096 by default
        number of intervals

    Returns
    ----------
    species: np.array of int
        index of the table for each framework ion

    tables: np.array (n_species, size + 1, 4)
        interpolation coefficients

    inv_dr: float
        inverse of the table spacing
    """

    rows = np.stack([kind.astype(np.float64), p1, p2, p3], axis = 1)
    unique, species = np.unique(rows, axis = 0, return_inverse = True)
    species = np.asarray(species).reshape(-1).astype(np.int64)
    dr = r_cut / size
    # nodes below R_EXACT are never used, r is shifted there to avoid division by zero
    r = np.maximum(np.arange(size + 2) * dr, dr)
    tables = np.zeros((len(unique), size + 1, 4))
    for s, (k, a1, a2, a3) in enumerate(unique):
        with np.errstate(all = 'ignore'):
            v = pair_energy(r, int(k), a1, a2, a3)
            m = pair_energy_derivative(r, int(k), a1, a2, a3) * dr
        v0, v1, m0, m1 = v[:-1], v[1:], m[:-1], m[1:]
        tables[s, :, 0] = v0
        tables[s, :, 1] = m0
        tables[s, :, 2] = 3 * (v1 - v0) - 2 * m0 - m1
        tables[s, :, 3] = 2 * (v0 - v1) + m0 + m1
    return species, np.nan_to_num(tables), 1 / dr



@njit(inline = 'always')
def _morse(r, p1, p2, p3):
    x = math.exp(p2 * (p3 - r))
    return p1 * (x * (x - 2.0))



@njit(inline = 'always')
def _coulomb(r, p1, p2):
    return p1 / r * math.erfc(p2 * r)



@cached_njit(parallel = True, nogil = True)
def _grid_exact(out, frac, kind, p1, p2, p3, G, R, r_cut, r_floor):

    """ out[ix, iy, iz] += sum of pair energies over all ions within r_cut.

    The loop over x planes is parallel: a plane is modified by one thread only.
    Along y and z the grid indexes are not wrapped until the array is accessed,
    periodic images along x are enumerated explicitly.
    """

    NX, NY, NZ = out.shape
    rc2 = r_cut * r_cut
    G11 = G[0, 0]; G22 = G[1, 1]; G33 = G[2, 2]
    G12 = G[0, 1]; G13 = G[0, 2]; G23 = G[1, 2]
    Ra = R[0]; Rb = R[1]
    for ix in prange(NX):
        gx = ix / NX
        for a in range(frac.shape[0]):
            ka = kind[a]
            if ka == 0:
                continue
            fax = frac[a, 0]; fay = frac[a, 1]; faz = frac[a, 2]
            a1 = p1[a]; a2 = p2[a]; a3 = p3[a]
            smin = int(math.ceil(gx - fax - Ra)); smax = int(math.floor(gx - fax + Ra))
            jlo = int(math.ceil((fay - Rb) * NY)); jhi = int(math.floor((fay + Rb) * NY))
            for sx in range(smin, smax + 1):
                dx = gx - fax - sx
                for j in range(jlo, jhi + 1):
                    dy = j / NY - fay
                    # r^2 = C + 2 * B * dz + G33 * dz^2
                    C = G11 * dx * dx + 2.0 * G12 * dx * dy + G22 * dy * dy
                    B = G13 * dx + G23 * dy
                    r2min = C - B * B / G33
                    if r2min >= rc2:
                        continue
                    w = math.sqrt((rc2 - r2min) / G33)
                    zc = faz - B / G33
                    klo = int(math.ceil((zc - w) * NZ)); khi = int(math.floor((zc + w) * NZ))
                    row = out[ix, j % NY]
                    k = klo
                    kw = klo % NZ
                    while k <= khi:
                        seg = min(khi - k + 1, NZ - kw)
                        base = k / NZ - faz
                        for m in range(seg):
                            dz = base + m / NZ
                            r2 = C + dz * (2.0 * B + G33 * dz)
                            if r2 < rc2:
                                r = math.sqrt(max(r2, 0.0))
                                if ka == 1:
                                    row[kw + m] += _morse(r, a1, a2, a3)
                                else:
                                    row[kw + m] += _coulomb(max(r, r_floor), a1, a2)
                        k += seg
                        kw = 0
    return out



@cached_njit(parallel = True, nogil = True)
def _grid_table(out, frac, kind, p1, p2, p3, species, tables, inv_dr, r_exact, G, R, r_cut, r_floor):

    """ The same as _grid_exact, but pair energies are interpolated
    from the tables for the distances larger than r_exact. """

    NX, NY, NZ = out.shape
    rc2 = r_cut * r_cut
    G11 = G[0, 0]; G22 = G[1, 1]; G33 = G[2, 2]
    G12 = G[0, 1]; G13 = G[0, 2]; G23 = G[1, 2]
    Ra = R[0]; Rb = R[1]
    for ix in prange(NX):
        gx = ix / NX
        for a in range(frac.shape[0]):
            ka = kind[a]
            if ka == 0:
                continue
            fax = frac[a, 0]; fay = frac[a, 1]; faz = frac[a, 2]
            a1 = p1[a]; a2 = p2[a]; a3 = p3[a]
            ts = tables[species[a]]
            smin = int(math.ceil(gx - fax - Ra)); smax = int(math.floor(gx - fax + Ra))
            jlo = int(math.ceil((fay - Rb) * NY)); jhi = int(math.floor((fay + Rb) * NY))
            for sx in range(smin, smax + 1):
                dx = gx - fax - sx
                for j in range(jlo, jhi + 1):
                    dy = j / NY - fay
                    C = G11 * dx * dx + 2.0 * G12 * dx * dy + G22 * dy * dy
                    B = G13 * dx + G23 * dy
                    r2min = C - B * B / G33
                    if r2min >= rc2:
                        continue
                    w = math.sqrt((rc2 - r2min) / G33)
                    zc = faz - B / G33
                    klo = int(math.ceil((zc - w) * NZ)); khi = int(math.floor((zc + w) * NZ))
                    row = out[ix, j % NY]
                    k = klo
                    kw = klo % NZ
                    while k <= khi:
                        seg = min(khi - k + 1, NZ - kw)
                        base = k / NZ - faz
                        for m in range(seg):
                            dz = base + m / NZ
                            r2 = C + dz * (2.0 * B + G33 * dz)
                            if r2 < rc2:
                                r = math.sqrt(max(r2, 0.0))
                                if r < r_exact:
                                    if ka == 1:
                                        row[kw + m] += _morse(r, a1, a2, a3)
                                    else:
                                        row[kw + m] += _coulomb(max(r, r_floor), a1, a2)
                                else:
                                    t = r * inv_dr
                                    i = int(t)
                                    t -= i
                                    row[kw + m] += ts[i, 0] + t * (ts[i, 1] + t * (ts[i, 2] + t * ts[i, 3]))
                        k += seg
                        kw = 0
    return out



def bvse_grid(cell, frac, shape, kind, p1, p2, p3, r_cut = 10.0, exact = False, table_size = 4096):

    """ Calculate BVSE on a periodic grid.

    Parameters
    ----------
    cell: np.array (3, 3)
        lattice vectors (rows)

    frac: np.array (n_ions, 3)
        fractional coordinates of framework ions

    shape: tuple of 3 int
        number of grid points along lattice vectors,
        the point (i, j, k) has fractional coordinates (i / nx, j / ny, k / nz)

    kind, p1, p2, p3: np.array
        parameters of pair interactions for each framework ion, see pair_params

    r_cut: float, 10.0 by default
        maximum distance for mobile ion - framework interaction

    exact: boolean, False by default
        if False pair energies are interpolated from the tables (faster),
        otherwise exp and erfc are evaluated for each pair

    table_size: int, 4096 by default
        number of intervals in the tables of pair energies

    Returns
    ----------
    energy: np.array of the given shape
        BVSE distribution over the periodic grid
    """

    cell = np.ascontiguousarray(cell, dtype = np.float64)
    frac = np.ascontiguousarray(frac, dtype = np.float64)
    frac = frac - np.floor(frac)
    kind = np.ascontiguousarray(kind, dtype = np.int64)
    p1 = np.ascontiguousarray(p1, dtype = np.float64)
    p2 = np.ascontiguousarray(p2, dtype = np.float64)
    p3 = np.ascontiguousarray(p3, dtype = np.float64)
    r_cut = float(r_cut)

    G = cell @ cell.T
    # the largest fractional displacement of a point within r_cut along each lattice vector
    R = r_cut * np.linalg.norm(np.linalg.inv(cell), axis = 0)
    out = np.zeros(tuple(int(s) for s in shape), dtype = np.float64)
    if out.size == 0 or len(frac) == 0 or r_cut <= 0:
        return out
    if exact:
        with _LOCK:
            _grid_exact(out, frac, kind, p1, p2, p3, G, R, r_cut, R_FLOOR)
    else:
        species, tables, inv_dr = build_tables(kind, p1, p2, p3, r_cut, size = table_size)
        with _LOCK:
            _grid_table(out, frac, kind, p1, p2, p3, species, tables, inv_dr, R_EXACT,
                        G, R, r_cut, R_FLOOR)
    return out
