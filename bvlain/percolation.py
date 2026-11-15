""" Percolation analysis of periodic 3D scalar fields.

The percolation thresholds are found in a single pass: grid points are added
in the order of their values and connected with a union-find structure that
keeps, for every point, the lattice translation ("winding") accumulated on the
way to the root of its component. A component percolates when it gets
connected to one of its own periodic images.

Conventions (same as the labelling of a 2x2x2 tiled grid used before):

* the grid includes both end points of each axis (see Lain._mesh), so the
  first and the last planes are the same physical plane; the last index of
  an axis is a neighbor of the first index of the next cell
* points are 26-connected
* the "dimension" of a component is the number of its 8 periodic images within
  a 2x2x2 supercell that belong to the same component: 2 -> 1D, 4 -> 2D,
  8 -> 3D percolation
"""

import itertools
import numpy as np
from .kernels import cached_njit


_OFFSETS = np.array([o for o in itertools.product((-1, 0, 1), repeat = 3) if o != (0, 0, 0)],
                    dtype = np.int64)
_LEVEL = {2: 0, 4: 1, 8: 2}



def _z2_tables():

    """ Lookup tables for subgroups of Z2^3 encoded as 8-bit masks
    (bit x is set if the element x belongs to the subgroup).

    Returns
    ----------
    add: np.array (256, 8)
        add[s, v] is the subgroup spanned by s and the element v
    pop: np.array (256)
        number of elements in the subgroup
    """

    add = np.zeros((256, 8), np.uint8)
    for s in range(256):
        for v in range(8):
            t = s
            for x in range(8):
                if (s >> x) & 1:
                    t |= 1 << (x ^ v)
            add[s, v] = t
    pop = np.array([bin(s).count('1') for s in range(256)], np.uint8)
    return add, pop


_Z2ADD, _Z2POP = _z2_tables()



@cached_njit(nogil = True)
def _find(parent, ox, oy, oz, x):

    """ Find the root of x with path compression.
    Returns the root and the translation of x with respect to the root. """

    sx = 0; sy = 0; sz = 0; r = x
    while parent[r] != r:
        sx += ox[r]; sy += oy[r]; sz += oz[r]; r = parent[r]
    y = x; cx = sx; cy = sy; cz = sz
    while y != r and parent[y] != r:
        nxt = parent[y]; tx = ox[y]; ty = oy[y]; tz = oz[y]
        parent[y] = r; ox[y] = cx; oy[y] = cy; oz[y] = cz
        cx -= tx; cy -= ty; cz -= tz; y = nxt
    return r, sx, sy, sz



@cached_njit(nogil = True)
def _sweep(order, nx, ny, nz, offsets, z2add, z2pop):

    """ Add grid points one by one and track the largest percolation dimension.

    Parameters
    ----------
    order: np.array of int
        flat (C order) indexes of the grid points in the order of insertion

    nx, ny, nz: int
        shape of the grid

    Returns
    ----------
    steps: np.array (3)
        positions in order at which the dimension reached 2, 4 and 8
        (-1 if never reached)
    """

    N = nx * ny * nz
    parent = np.arange(N).astype(np.int32)
    size = np.ones(N, np.int32)
    ox = np.zeros(N, np.int32); oy = np.zeros(N, np.int32); oz = np.zeros(N, np.int32)
    active = np.zeros(N, np.bool_)
    z2 = np.ones(N, np.uint8)
    steps = np.full(3, -1, np.int64)
    best = 1
    nyz = ny * nz
    for step in range(order.size):
        u = order[step]
        active[u] = True
        i = u // nyz; rem = u - i * nyz; j = rem // nz; k = rem - j * nz
        for o in range(offsets.shape[0]):
            ii = i + offsets[o, 0]; jj = j + offsets[o, 1]; kk = k + offsets[o, 2]
            sx = 0; sy = 0; sz = 0
            if ii < 0:
                ii += nx; sx = -1
            elif ii >= nx:
                ii -= nx; sx = 1
            if jj < 0:
                jj += ny; sy = -1
            elif jj >= ny:
                jj -= ny; sy = 1
            if kk < 0:
                kk += nz; sz = -1
            elif kk >= nz:
                kk -= nz; sz = 1
            w = (ii * ny + jj) * nz + kk
            if not active[w]:
                continue
            ru, ux, uy, uz = _find(parent, ox, oy, oz, u)
            rw, wx, wy, wz = _find(parent, ox, oy, oz, w)
            gx = ux + sx - wx; gy = uy + sy - wy; gz = uz + sz - wz
            if ru == rw:
                # closed loop: nonzero translation connects the component to its image
                if gx != 0 or gy != 0 or gz != 0:
                    z2[ru] = z2add[z2[ru], (gx & 1) | ((gy & 1) << 1) | ((gz & 1) << 2)]
            else:
                if size[ru] < size[rw]:
                    a = ru; b = rw; gx = -gx; gy = -gy; gz = -gz
                else:
                    a = rw; b = ru
                parent[a] = b; ox[a] = gx; oy[a] = gy; oz[a] = gz; size[b] += size[a]
                s = z2[b]; t = z2[a]
                for x in range(8):
                    if (t >> x) & 1:
                        s = z2add[s, x]
                z2[b] = s
        r, _, _, _ = _find(parent, ox, oy, oz, u)
        d = z2pop[z2[r]]
        if d > best:
            if d >= 2 and steps[0] < 0:
                steps[0] = step
            if d >= 4 and steps[1] < 0:
                steps[1] = step
            if d >= 8 and steps[2] < 0:
                steps[2] = step
            best = d
        if best == 8:
            break
    return steps



def critical_values(region, ascending = True, cut = None):

    """ Exact percolation thresholds of a periodic scalar field.

    Parameters
    ----------
    region: np.array (nx, ny, nz)
        scalar field, both end points of each axis included

    ascending: boolean, True by default
        if True the set {region < threshold} is analyzed,
        otherwise the set {region > threshold}

    cut: float, None by default
        points beyond this value are ignored

    Returns
    ----------
    values: list of 3 floats
        the set {region < t} (ascending) percolates in 1D, 2D, 3D
        if and only if t > values[0], values[1], values[2].
        For descending order the condition is t < values[i].
        +np.inf (-np.inf for descending order) if there is no percolation.
    """

    region = np.ascontiguousarray(region, dtype = np.float64)
    flat = region.ravel()
    if ascending:
        ids = np.flatnonzero(flat < cut) if cut is not None else np.arange(flat.size)
        order = ids[np.argsort(flat[ids], kind = 'stable')]
    else:
        ids = np.flatnonzero(flat > cut) if cut is not None else np.arange(flat.size)
        order = ids[np.argsort(-flat[ids], kind = 'stable')]
    nx, ny, nz = region.shape
    steps = _sweep(order.astype(np.int64), nx, ny, nz, _OFFSETS, _Z2ADD, _Z2POP)
    missing = np.inf if ascending else -np.inf
    return [float(flat[order[s]]) if s >= 0 else missing for s in steps]



def percolation_energy(data, dim, encut = 10.0, values = None):

    """ Percolation energy for a given dimensionality of percolation.

    Parameters
    ----------
    data: np.array (nx, ny, nz)
        BVSE distribution

    dim: int
        dimensionality of percolation. 2 -> 1D, 4 -> 2D, 8 -> 3D percolation

    encut: float, 10.0 by default
        stop criterion for the search of percolation energy

    values: list of 3 floats, None by default
        precomputed output of critical_values

    Returns
    ----------
    barrier: float
        percolation energy or np.inf if no percolation found
    """

    data = np.asarray(data, dtype = np.float64)
    data = data - data.min()
    if values is None:
        values = critical_values(data - data.min(), ascending = True, cut = encut)
    value = values[_LEVEL[dim]]
    emin = data.min()
    emax = emin + encut
    barrier = np.inf
    # the bisection is kept to return the same (rounded) numbers as before
    while (emax - emin) > 0.01:
        probe = (emin + emax) / 2
        if probe > value:
            emax = probe
            barrier = round(emax, 4)
        else:
            emin = probe
    return barrier



def percolation_radius(void_data, dim, values = None):

    """ Percolation radius for a given dimensionality of percolation.

    Parameters
    ----------
    void_data: np.array (nx, ny, nz)
        void space distribution

    dim: int
        dimensionality of percolation. 2 -> 1D, 4 -> 2D, 8 -> 3D percolation

    values: list of 3 floats, None by default
        precomputed output of critical_values

    Returns
    ----------
    radius: float
        percolation radius or 0 if no percolation found
    """

    void_data = np.asarray(void_data, dtype = np.float64)
    data = np.where(void_data > 0, void_data, 0)
    if values is None:
        values = critical_values(data - data.min(), ascending = False, cut = 0.0)
    value = values[_LEVEL[dim]]
    emax = data.max()
    emin = 0
    radius = 0
    while (emax - emin) > 0.1:
        probe = (emin + emax) / 2
        if probe < value:
            emin = probe
            radius = round(emin, 4)
        else:
            emax = probe
    return radius



def percolation_barriers(data, encut = 10.0):

    """ Percolation energies of the BVSE distribution.

    Parameters
    ----------
    data: np.array (nx, ny, nz)
        BVSE distribution

    encut: float, 10.0 by default
        cutoff energy above which barriers supposed to be np.inf

    Returns
    ----------
    energies: dict
        {'E_1D': float, 'E_2D': float, 'E_3D': float}
    """

    data = np.asarray(data, dtype = np.float64)
    shifted = data - data.min()
    values = critical_values(shifted - shifted.min(), ascending = True, cut = encut)
    energies = {}
    for i, dim in enumerate([2, 4, 8]):
        energies.update({f'E_{i+1}D': percolation_energy(data, dim, encut = encut, values = values)})
    return energies



def percolation_radii(void_data):

    """ Percolation radii of the void space distribution.

    Parameters
    ----------
    void_data: np.array (nx, ny, nz)
        void space distribution

    Returns
    ----------
    radii: dict
        {'r_1D': float, 'r_2D': float, 'r_3D': float}
    """

    void_data = np.asarray(void_data, dtype = np.float64)
    data = np.where(void_data > 0, void_data, 0)
    values = critical_values(data - data.min(), ascending = False, cut = 0.0)
    radii = {}
    for i, dim in enumerate([2, 4, 8]):
        radii.update({f'r_{i+1}D': percolation_radius(void_data, dim, values = values)})
    return radii
