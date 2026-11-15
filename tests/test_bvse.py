""" Tests of the compiled BVSE kernel and of the percolation analysis.

Run with: pytest tests
"""

import numpy as np
import pytest
from scipy import ndimage
from ase.spacegroup import crystal
from ase.build import make_supercell

from bvlain import Lain
from bvlain import percolation



def lifepo4():
    return crystal(['Li', 'Fe', 'P', 'O', 'O', 'O'],
                   basis = [(0, 0, 0), (0.2822, 0.25, 0.9747), (0.0948, 0.25, 0.4182),
                            (0.0968, 0.25, 0.7428), (0.4571, 0.25, 0.2060), (0.1656, 0.0465, 0.2848)],
                   spacegroup = 62, cellpar = [10.332, 6.010, 4.692, 90, 90, 90])



CELLS = {'orthorhombic': np.eye(3),
         'triclinic': [[1, 0, 0], [1, 1, 0], [-1, 1, 1]],
         'skewed': [[1, 0, 0], [0, 1, 0], [0, 3, 1]]}



def calculator(cell = 'orthorhombic'):
    calc = Lain(verbose = False)
    calc.read_atoms(make_supercell(lifepo4(), CELLS[cell]))
    return calc



@pytest.mark.parametrize('cell', list(CELLS))
def test_kernel_vs_neighbor_search(cell):

    """ The kernel should give the same energies as the KDTree based
    summation if all the neighbors within r_cut are collected """

    calc = calculator(cell)
    params = {'mobile_ion': 'Li1+', 'r_cut': 10.0, 'resolution': 0.5}
    reference = calc.bvse_distribution(k = 600, **params).copy()
    exact = calc.bvse_distribution(exact = True, **params).copy()
    table = calc.bvse_distribution(**params).copy()

    assert exact.shape == reference.shape == table.shape
    scale = np.maximum(1.0, np.abs(reference))
    assert np.max(np.abs(exact - reference) / scale) < 1e-11
    assert np.max(np.abs(table - exact) / scale) < 1e-8
    assert not np.array_equal(table, exact)
    for data in (exact, table):
        assert np.all(np.isfinite(data))
        assert np.array_equal(data[0], data[-1])
        assert np.array_equal(data[:, 0], data[:, -1])
        assert np.array_equal(data[:, :, 0], data[:, :, -1])



def test_kernel_parameters():

    """ Screening factor and softBV covalent radii are passed to the kernel """

    calc = calculator()
    params = {'mobile_ion': 'Li1+', 'r_cut': 8.0, 'resolution': 0.5}
    default = calc.bvse_distribution(exact = True, **params).copy()
    for options in ({'f': 0.6}, {'use_softbv_covalent_radii': True}):
        reference = calc.bvse_distribution(k = 400, **params, **options).copy()
        exact = calc.bvse_distribution(exact = True, **params, **options).copy()
        scale = np.maximum(1.0, np.abs(reference))
        assert np.max(np.abs(exact - reference) / scale) < 1e-11
        assert np.max(np.abs(exact - default)) > 1e-3



def test_volumetric_files(tmp_path):

    """ The last (duplicated) planes are not written """

    from ase.io.cube import read_cube

    calc = calculator()
    data = calc.bvse_distribution(mobile_ion = 'Li1+', resolution = 0.5)
    nx, ny, nz = data.shape
    unique = data[:nx - 1, :ny - 1, :nz - 1]

    calc.write_grd(str(tmp_path / 'bvse'))
    with open(tmp_path / 'bvse.grd') as f:
        lines = f.read().split('\n')
    assert lines[0].startswith('#')
    assert np.allclose([float(x) for x in lines[1].split()], calc.cell.cellpar())
    assert [int(x) for x in lines[2].split()] == [nx - 1, ny - 1, nz - 1]
    assert lines[-1] == ''
    values = np.array([float(x) for x in lines[3:-1]])
    assert np.array_equal(values, unique.ravel())

    calc.write_cube(str(tmp_path / 'bvse'))
    with open(tmp_path / 'bvse.cube') as f:
        cube = read_cube(f)
    assert cube['data'].shape == unique.shape
    assert np.allclose(cube['data'], unique, rtol = 1e-5)
    assert len(cube['atoms']) == len(calc.atoms_copy)



def test_threads():

    """ The kernel can be called from several threads """

    from concurrent.futures import ThreadPoolExecutor

    def run(_):
        calc = calculator()
        return calc.bvse_distribution(mobile_ion = 'Li1+', resolution = 0.5)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(4)))
    for data in results[1:]:
        assert np.array_equal(data, results[0])



def test_neighbors_truncation():

    """ k nearest neighbors are not enough to converge the energy """

    calc = calculator()
    params = {'mobile_ion': 'Li1+', 'r_cut': 10.0, 'resolution': 0.5}
    full = calc.bvse_distribution(**params).copy()
    truncated = calc.bvse_distribution(k = 100, **params).copy()
    difference = np.abs(full - truncated)
    assert 1e-4 < difference.max() < 1e-2



def test_grid_point_on_ion():

    """ The energy should be finite if a grid point coincides with an ion """

    calc = Lain(verbose = False)
    atoms = lifepo4()
    atoms.positions -= atoms.positions[atoms.numbers == 26][0]    # Fe at the origin
    atoms.wrap()
    calc.read_atoms(atoms)
    for exact in (True, False):
        data = calc.bvse_distribution(mobile_ion = 'Li1+', resolution = 0.5, exact = exact)
        assert np.all(np.isfinite(data))
        assert data.max() > 1e6



def test_lifepo4_barriers():

    """ Values obtained with bvlain 0.25.1 (k = 100) """

    calc = calculator()
    calc.bvse_distribution(mobile_ion = 'Li1+', r_cut = 10.0, resolution = 0.2)
    assert calc.data.shape == (51, 30, 23)
    energies = calc.percolation_barriers(encut = 5.0)
    assert energies == {'E_1D': 0.4785, 'E_2D': 3.5938, 'E_3D': 3.6133}

    calc.void_distribution(mobile_ion = 'Li1+', r_cut = 10.0, resolution = 0.35)
    radii = calc.percolation_radii()
    assert radii == {'r_1D': 0.2914, 'r_2D': 0.1943, 'r_3D': 0.0971}



# reference percolation analysis: labelling of the 2x2x2 tiled grid

def max_dimension(mask):

    """ The largest number of periodic images of a point within a 2x2x2
    supercell that belong to the same connected component """

    shape = np.array(mask.shape)
    labels, n = ndimage.label(np.tile(mask, (2, 2, 2)), structure = np.ones((3, 3, 3)))
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # the opposite faces of the supercell are the same physical planes
    for axis in range(3):
        left = np.take(labels, -1, axis = axis).ravel()
        right = np.take(labels, 0, axis = axis).ravel()
        both = (left > 0) & (right > 0)
        for a, b in set(zip(left[both].tolist(), right[both].tolist())):
            parent[find(a)] = find(b)
    labels = np.array([find(x) for x in range(n + 1)])[labels]

    best = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        point = np.argwhere(labels == label)[0] % shape
        images = [labels[tuple(point + np.array(t) * shape)] == label
                  for t in np.ndindex(2, 2, 2)]
        best = max(best, sum(images))
    return best



def reference_energy(data, dim, encut):
    data = data - data.min()
    emin, emax, barrier = data.min(), data.min() + encut, np.inf
    while (emax - emin) > 0.01:
        probe = (emin + emax) / 2
        if max_dimension(data < probe) >= dim:
            emax = probe
            barrier = round(emax, 4)
        else:
            emin = probe
    return barrier



def reference_radius(void_data, dim):
    data = np.where(void_data > 0, void_data, 0)
    emin, emax, radius = 0, data.max(), 0
    while (emax - emin) > 0.1:
        probe = (emin + emax) / 2
        if max_dimension(data > probe) >= dim:
            emin = probe
            radius = round(emin, 4)
        else:
            emax = probe
    return radius



def random_field(shape, seed):

    """ Smooth periodic field, the last plane of each axis repeats the first one """

    rng = np.random.default_rng(seed)
    periods = [max(n - 1, 1) for n in shape]
    x, y, z = np.meshgrid(*[np.arange(p) / p for p in periods], indexing = 'ij')
    field = np.zeros(periods)
    for _ in range(12):
        h, k, l = rng.integers(-2, 3, size = 3)
        field += rng.normal() * np.cos(2 * np.pi * (h * x + k * y + l * z) + rng.uniform(0, 2 * np.pi))
    ids = [np.arange(n) % p for n, p in zip(shape, periods)]
    return field[np.ix_(*ids)]



@pytest.mark.parametrize('seed', range(12))
def test_percolation_vs_labelling(seed):
    shapes = [(9, 9, 9), (12, 7, 10), (6, 15, 8), (10, 10, 3), (2, 9, 11), (14, 5, 5)]
    shape = shapes[seed % len(shapes)]
    data = 3 * random_field(shape, seed)
    for encut in (2.0, 10.0):
        energies = percolation.percolation_barriers(data, encut = encut)
        for i, dim in enumerate([2, 4, 8]):
            assert energies[f'E_{i+1}D'] == reference_energy(data, dim, encut)

    void_data = random_field(shape, seed + 100)
    radii = percolation.percolation_radii(void_data)
    for i, dim in enumerate([2, 4, 8]):
        assert radii[f'r_{i+1}D'] == reference_radius(void_data, dim)



def test_percolation_simple_cases():

    """ A straight channel, a plane and the whole space """

    shape = (9, 9, 9)
    channel = np.ones(shape); channel[4, 4, :] = 0
    plane = np.ones(shape); plane[4, :, :] = 0
    space = np.zeros(shape); space[0, 0, 0] = -1.0     # data.min() is subtracted
    assert percolation.critical_values(channel, cut = 0.5)[1:] == [np.inf, np.inf]
    assert percolation.critical_values(channel, cut = 0.5)[0] == 0
    assert percolation.critical_values(plane, cut = 0.5) == [0, 0, np.inf]
    energies = percolation.percolation_barriers(space, encut = 5.0)
    assert energies['E_1D'] == energies['E_2D'] == energies['E_3D'] < 1.02
    energies = percolation.percolation_barriers(channel, encut = 0.5)
    assert energies['E_1D'] < 0.02 and energies['E_2D'] == energies['E_3D'] == np.inf



def test_percolation_merged_channels():

    """ Two perpendicular 1D channels connected by a bridge form a 2D network """

    data = np.full((9, 9, 9), 20.0)
    data[:, 2, 2] = 0.0
    data[5, :, 6] = 0.0
    data[5, 2, 3:6] = 1.0
    energies = percolation.percolation_barriers(data, encut = 5.0)
    assert energies['E_1D'] < 0.02
    assert energies['E_2D'] == 1.0059
    assert energies['E_3D'] == np.inf
    for i, dim in enumerate([2, 4, 8]):
        assert energies[f'E_{i+1}D'] == reference_energy(data, dim, 5.0)
