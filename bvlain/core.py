""" Lain - main class of BVlain library. """

import os 
import re
import sys
import json
import pickle
import numpy as np
from scipy.spatial import cKDTree
from ase.neighborlist import neighbor_list
from ase.io import read, cube
from ase.build import make_supercell
from ase.data import atomic_numbers
from .potentials import BVSEPotential
from .data.covalent_radii import get_covalent_radii
from . import kernels
from . import percolation



class Lain:
    """ 
    The class is used to perform BVSE calculations and related tasks.
    
    Parameters
    ----------

    verbose: boolean, True by default
        Will print progress steps if True
    """   

    def __init__(self, verbose = True):

        self.verbose = verbose
        self.params_path = self._resource_path('data')
        self.cation_file = os.path.join(self.params_path, 'cation.pkl')
        self.anion_file = os.path.join(self.params_path, 'anion.pkl')
        self.quantum_file = os.path.join(self.params_path, 'quantum_n.pkl')
        self.radii_file = os.path.join(self.params_path, 'shannon-radii.json')
    


    def read_structure(self, st, oxi_check = True, forbidden_species = ['O-', 'P3-']):

        """
        Read structure from pymatgen's Structure.
        Note: Works only with ordered structures

        Parameters
        ----------

        st: pymatgen's Structure object
            Should be ordered
            
        oxi_check: boolean, False by default)
            If true will try to assign oxi states by pymategen's BVAnalyzer

        forbidden_species: list of str, ['O-', 'P3-'] by default
            list of forbidden ions to be checked during structure decoration
            used when oxi_check is True

        Returns
        -------
        pymatgen's Structure object
        stores ase.atoms in self.atoms_copy

        """
        
        from pymatgen.io.ase import AseAtomsAdaptor

        self.st = st
        if oxi_check:
            from pymatgen.analysis.bond_valence import BVAnalyzer
            bva = BVAnalyzer(forbidden_species = forbidden_species)
            self.st = bva.get_oxi_state_decorated_structure(self.st)
        self.atoms_copy = AseAtomsAdaptor.get_atoms(self.st)
        self.cell = self.atoms_copy.cell
        return self.st



    def read_atoms(self, atoms, oxi_check = True, forbidden_species = ['O-', 'P3-']):

        """
        Read ase' atoms object
        Note: Works only with ordered structures

        Parameters
        ----------

        atoms: ase's Atoms object
            Should be ordered
            
        oxi_check: boolean, False by default)
            If true will try to assign oxi states by pymategen's BVAnalyzer

        forbidden_species: list of str, ['O-', 'P3-'] by default
            list of forbidden ions to be checked during structure decoration
            used when oxi_check is True

        Returns
        -------
        ase's Atoms object
        stores ase.atoms in self.atoms_copy

        """
        from pymatgen.io.ase import AseAtomsAdaptor

        st = AseAtomsAdaptor().get_structure(atoms)
        self.read_structure(st, oxi_check=oxi_check, forbidden_species=forbidden_species)
        return self.atoms_copy
    


    def read_file(self, file, oxi_check = True, forbidden_species = ['O-', 'P3-']):
        """ 
        Structure reader. Possible formats are .cif, POSCAR. 
        It is a bit modified pymatgen's function Structure.from_file.
        Note: Works only with ordered structures

        Parameters
        ----------

        file: str
            pathway to CIF or POSCAR
            
        oxi_check: boolean, False by default
            If true will try to assign oxi states by pymategen's BVAnalyzer

        forbidden_species: list of str, ['O-', 'P3-'] by default
            list of forbidden ions to be checked during structure decoration
            used when oxi_check is True


        Returns
        -------
        ase's Atoms object
        stores ase.atoms in self.atoms_copy
        """

        atoms = read(file)
        return self.read_atoms(atoms, oxi_check=oxi_check, forbidden_species=forbidden_species)


        
    def _mesh(self, resolution = 0.1, shift = [0, 0, 0]):
        
        """ 
        This method creates grid of equidistant points in 3D
        with respect to the input resolution. 

        Parameters
        ----------

        resolution: float, 0.2 by default
            spacing between points (in Angstroms)
            Note: Number of points ~(lattice_parameter/resolution)^3
            
        shift: array [x, y, z]
            Used when invoked from self.bvse_distribution function

        Returns
        -------
        mesh_: np.array
            meshgrid
        
        """

        a, b, c, _, _, _ = self.cell.cellpar()
        nx, ny, nz = int(a // resolution), int(b // resolution), int(c // resolution)
        x = np.linspace(0, 1, nx) + shift[0]
        y = np.linspace(0, 1, ny) + shift[1]
        z = np.linspace(0, 1, nz) + shift[2]
        mesh_ = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        self.mesh_ = mesh_
        self.shift = shift
        self.size = [nx, ny, nz]
        return mesh_
    
    
    
    def _scale_cell(self, cell, r_cut):
        """ 
        
        Scaling of the unit cell for the search of neighbors

        Parameters
        ----------

        r_cut: float
            cutoff distance for interaction between tracer ion and framework
            
        cell: ase.atoms.cell
            Unit cell parameters

        Returns
        -------
        scale: np.array (3, 3)
            Matrix of the unit cell transformation
        
        """
        # distances between the opposite faces of the unit cell
        heights = 1 / np.linalg.norm(np.linalg.inv(np.array(cell)), axis = 0)
        scale_a, scale_b, scale_c = 2 * np.ceil(r_cut / heights) + 1
        scale = np.vstack([[scale_a, 0, 0], [0, scale_b, 0], [0, 0, scale_c]])
        return scale



    def _get_params(self, mobile_ion = None, use_softbv_covalent_radii = False):
        
        """ 
        
        Collect parameters required for the calculations

        Parameters
        ----------
        mobile_ion: str,
            ion, e.g. Li1+, F1-

        use_softbv_covalent_radii: boolean, False by default
            use ASE or softBV covalent radii data
        """

        with open(self.quantum_file, 'rb') as f:
            quantum_number = pickle.load(f) 
        self.num_mi, self.q_mi = self._decompose(mobile_ion)
        self.framework = self.atoms_copy.copy()[self.atoms_copy.numbers != self.num_mi]
        self.atoms = self.framework
        self.n_mi = quantum_number[self.num_mi]
        self.rc_mi = get_covalent_radii(self.num_mi, self.q_mi if use_softbv_covalent_radii else None)
        numbers, oxi_states = self.atoms.numbers, self.atoms.arrays['oxi_states']
        rc = np.array([get_covalent_radii(num, q if use_softbv_covalent_radii else None) for num, q in zip(numbers, oxi_states)])
        self.atoms.set_array('r_c', rc)
        self.atoms.set_array('n', np.array([quantum_number[num] for num in self.atoms.numbers]))
        charges = self.atoms.get_array('oxi_states')
        r_min = list()
        alpha = list()
        d0 = list()

        if self.q_mi > 0:
            with open(self.cation_file, 'rb') as f:
                data = pickle.load(f) 
                data = data[self.num_mi][self.q_mi]
            try:    
                for num, charge in zip(self.atoms.numbers, charges):
                    if charge < 0:
                        params = data[num][charge]
                        r_min.append(params['r_min'])
                        alpha.append(params['alpha'])
                        d0.append(params['d0'])
                    else:
                        r_min.append(np.nan)
                        alpha.append(np.nan)
                        d0.append(np.nan)
            except KeyError:
                print('Oops. No BVSE data for a given combination of oxidation states.')
                raise
        else:
            with open(self.anion_file, 'rb') as f:
                data = pickle.load(f)
                data = data[self.num_mi][self.q_mi]
            try:    
                for num, charge in zip(self.atoms.numbers, charges):
                    if charge > 0:
                        params = data[num][charge]
                        r_min.append(params['r_min'])
                        alpha.append(params['alpha'])
                        d0.append(params['d0'])
                    else:
                        r_min.append(np.nan)
                        alpha.append(np.nan)
                        d0.append(np.nan)
            except KeyError:
                print('Oops. No BVSE data for a given combination of oxidation states.')
                raise

        r_min = np.hstack(r_min)
        alpha = np.hstack(alpha)
        d0 = np.hstack(d0)
        self.atoms.set_array('r_min', r_min)
        self.atoms.set_array('alpha', alpha)
        self.atoms.set_array('d0', d0)



    def _get_ionic_radius(self, symbol, charge, table):
        """ 
        Get Shannon radius of ion. 
        Note: radius is averaged over all possible coordination numbers


        Parameters
        ----------
        symbol: str,
            atomic symbol, e.g. Li, F
        charge: int,
            oxidation state of an ion
        table: dict,
            data read from json file shannon-radii.json
            see: https://github.com/prtkm/ionic-radii

        Returns
        ----------
        radii: float
            average ionic radii for the provided ion

        """

        d = table[symbol][str(charge)]
        radii = []
        for CN in d.keys():
            radii.append(d[CN]['r_ionic'])
        return np.array(radii).mean()


    
    def _get_params_voids(self, mobile_ion = None, ionic = True):
        
        """
        Collect parameters required for the calculations

        Parameters
        ----------
        mobile_ion: str,
            ion, e.g. Li1+, F1-

        """
        file = self.radii_file
        with open(file) as f:
            radii_data = f.read()
        table = json.loads(radii_data)

        self.num_mi, self.q_mi = self._decompose(mobile_ion)
        self.framework = self.atoms_copy.copy()[self.atoms_copy.numbers != self.num_mi]
        self.atoms = self.framework
        if ionic:
            self.ri_mi = self._get_ionic_radius(self.element_mi, self.q_mi, table)
            charges = np.array(self.atoms.get_array('oxi_states'), dtype = int)
            r_i = [self._get_ionic_radius(s, c, table) for s,c in zip(self.atoms.symbols, charges)]
            self.atoms.set_array('r_i', np.array(r_i))



    def _ionic_dist(self, R, r_i):
        
        """ Calculate distances between mobile ion and 
        framework's ions considering them hard spheres
            Note: radius of a mobile ion is 0


        Parameters
        ----------
        R: np.array of floats
            distance between mobile ion and framework ions' centers
            
        r_i: np.array of floats
            ionic radii of framework ions
            
        Returns
        ----------
        dists: np.array
            distances between mesh points and framework ions
            considering their ionic radii
        """

        dists = R - r_i
        return dists



    def void_distribution(self, mobile_ion = None, r_cut = 10.0,
                          resolution = 0.2, k = 2, ionic = True):
        
        """ 
        Calculate void space distribution for a given mobile ion.
        Note! It is a vectorized method. Works fast, but memory expensive.
        Never ever set resolution parameter lower then 0.05.


        Parameters
        ----------

        mobile_ion: str
            ion, e.g. 'Li1+', 'F-'
            
        resolution: float, 0.2 by default
            distance between grid points
            
        r_cut: float, 10.0 by default
            maximum distances for neighbor search
            Note: do not set the parameter < minimum mobile ion to framework distance
            
        k: int, 2 by default
            maximum number of neighbours (used for KDTree search of neighbors)
            adjusting this number should not effect the final result. used for tests.
        
        Returns
        ----------
        
        void_data: np.array
            void distribution
        """

        self.resolution = resolution
        
        if self.verbose:
            print('getting void distribution...')
        
        self._get_params_voids(mobile_ion, ionic = ionic)
        _, distances, ids, numbers =  self._neighbors(r_cut = r_cut,
                                                     resolution = resolution,
                                                     k = k)
        if ionic:
            r_i = np.take(self._get_array('r_i'), ids, axis = -1)
        else:
            r_i = np.zeros(ids.shape)
        min_dists = np.nan_to_num(self._ionic_dist(distances, r_i),
                                copy = False,
                                nan = 1000.0).min(axis = 1)
        #self.void_dist = np.where(min_dists > 0, min_dists, 0)
        self.void_dist = min_dists
        self.void_data = self.void_dist.reshape(self.size)
        
        if self.verbose:
            print('distribution is ready\n')
        
        return self.void_data



    def _percolation_radius(self, dim):

        """ Get percolation radius for a given dimensionality of percolation

        Parameters
        ----------

        dim: int
            dimensionality of percolation. 2 -> 1D, 4 -> 2D, 8 - 3D percolation
            
        Returns
        ----------
        percolation_radius: float
            percolation radius or 0 if no percolation found
        """
        
        return percolation.percolation_radius(self.void_data, dim)



    def _decompose(self, mobile_ion):

        """ Decompose input string into chemical element and oxidation state

        Parameters
        ----------
        mobile_ion: str,
            ion, e.g. Li1+, F1-
        

        Returns
        ----------
        tuple(atomic_number, oxidation_state)

        """

        element = re.sub('\d', '', mobile_ion).replace("+","").replace("-","")
        oxi_state = re.sub('\D', '', mobile_ion)

        if '-' in mobile_ion:
            sign = -1
        else:
            sign = 1
        if len(oxi_state) > 0:
            if sign > 0:
                oxi_state = float(oxi_state)
            else:
                oxi_state = -float(oxi_state)
        else:
            oxi_state = sign

        if self.verbose:
            print(f'\tcollecting force field parameters...',
                  f'{element} | charge: {oxi_state}')
        
        self.mi_atom = atomic_numbers[element]
        self.mi_charge = int(oxi_state)
        self.element_mi = element
        return atomic_numbers[element], int(oxi_state)
        
        
    
    def _cartesian_sites(self, mesh):
        
        """ Helper function"""
        
        sites = self.cell.cartesian_positions(mesh)
        self.sites = sites
        return sites
        
        
        
    def _neighbors(self, r_cut = 10.0, resolution = 0.1, k = 100): # modify considering kdtree bug!
        
        """ 
        Search of the neighbors using scipy's cKDTree
        Parameters
        ----------
 
        r_cut: float
           cutoff radius of tracer ion - framework interaction
            
        resolution: float
            distance between grid points (in Angstroms)
            
        k: int
            maximum number of neighbors
        
        Returns
        ----------
        
        tuple of neigbors parameters
            
        """
        
        if self.verbose:
            print('\tcollecting neighbors...')
        
        a, b, c, _, _, _ = self.cell.cellpar()
        scale = self._scale_cell(self.atoms.cell, r_cut = r_cut)
        shift = [np.median(np.arange(0, scale[0,0])),
                np.median(np.arange(0, scale[1,1])),
                np.median(np.arange(0, scale[2,2])),
                ]
        supercell = make_supercell(self.atoms, scale)
        self.supercell = supercell
        
        sites = self._cartesian_sites(self._mesh(resolution = resolution,
                                               shift = shift))
        self.sites = sites
        KDTree = cKDTree(supercell.positions)
        distances, indexes = KDTree.query(sites,
                                          workers=-1,
                                          k=k,
                                          distance_upper_bound = r_cut)            
        return sites, distances, indexes, supercell.numbers
    
    

    def _get_array(self, name):
        """ Helper function"""
        arr = self.supercell.get_array(name)
        arr = np.concatenate([arr, [np.nan]]) # np.nan is added to deal with kDTree upper bound
        return arr
    
    
    
    def bvse_distribution(self, mobile_ion = None, r_cut = 10,
                          resolution = 0.2, k = None, f = 0.74, use_softbv_covalent_radii = False,
                          exact = False):
        
        """ 
        Calculate BVSE distribution for a given mobile ion.

        By default all framework ions within r_cut are taken into account.
        The calculation is done by the compiled kernel (see bvlain.kernels),
        its memory consumption does not depend on the number of neighbors.


        Parameters
        ----------

        mobile_ion: str
            ion, e.g. 'Li1+', 'F-'
            
        resolution: float, 0.2 by default
            distance between grid points
            
        r_cut: float, 10.0 by default
            maximum distances for mobile ion - framework interaction
            
        k: int, None by default
            if None, all framework ions within r_cut are considered.
            if int, only k nearest framework ions within r_cut are considered
            (KDTree search of neighbors, the behavior of the versions up to 0.25.1, k = 100).
            Note! This option is vectorized over (grid points, k). Works slower and it is
            memory expensive. Never ever set resolution parameter lower then 0.05 with it.

        f: float, 0.74 by default
            screening factor

        use_softbv_covalent_radii: boolean, False by default
            use ASE or softBV covalent radii data

        exact: boolean, False by default
            used when k is None.
            if False, pair energies are interpolated from the tables (cubic interpolation,
            relative error is below 1e-8), otherwise they are calculated explicitly (slower)
        
        Returns
        ----------
        
        bvse_data: np.array
            BVSE distribution
        """
        self.k = k
        self.r_cut = r_cut
        self.resolution = resolution
        if self.verbose:
            print('getting BVSE distribution...')
        
        self._get_params(mobile_ion, use_softbv_covalent_radii = use_softbv_covalent_radii)
        if k is None:
            energy = self._bvse_kernel(r_cut = r_cut, resolution = resolution, f = f, exact = exact)
        else:
            energy = self._bvse_neighbors(r_cut = r_cut, resolution = resolution, k = k, f = f)
        self.distribution = energy.ravel()
        self.data = energy.reshape(self.size)
        if self.verbose:
            print('distribution is ready\n')
        return self.data



    def _bvse_kernel(self, r_cut = 10.0, resolution = 0.2, f = 0.74, exact = False):

        """ 
        Calculate BVSE distribution considering all framework ions within r_cut.

        Parameters
        ----------

        r_cut: float
            maximum distances for mobile ion - framework interaction

        resolution: float
            distance between grid points

        f: float
            screening factor

        exact: boolean
            interpolate (False) or calculate explicitly (True) pair energies

        Returns
        ----------

        energy: np.array of self.size shape
            BVSE distribution
        """

        if self.verbose:
            print('\tsumming up pair interactions...')

        # grid points within the unit cell, no supercell is constructed
        self._cartesian_sites(self._mesh(resolution = resolution, shift = [0, 0, 0]))
        self.supercell = None

        atoms = self.atoms
        kind, p1, p2, p3 = kernels.pair_params(atoms.get_array('oxi_states'),
                                               atoms.get_array('r_min'),
                                               atoms.get_array('alpha'),
                                               atoms.get_array('d0'),
                                               atoms.get_array('r_c'),
                                               atoms.get_array('n'),
                                               self.q_mi, self.rc_mi, self.n_mi, f = f)
        # The mesh includes both end points of each axis (see self._mesh), i.e. the last
        # plane is a periodic copy of the first one. Only unique points are calculated.
        periods = [max(n - 1, 1) for n in self.size]
        energy = kernels.bvse_grid(np.array(self.cell),
                                   atoms.get_scaled_positions(),
                                   periods, kind, p1, p2, p3,
                                   r_cut = r_cut, exact = exact)
        ids = [np.arange(n) % period for n, period in zip(self.size, periods)]
        return energy[np.ix_(*ids)]



    def _bvse_neighbors(self, r_cut = 10.0, resolution = 0.2, k = 100, f = 0.74):

        """ 
        Calculate BVSE distribution considering k nearest framework ions within r_cut.
        Note! It is a vectorized method. Memory expensive.

        Parameters
        ----------

        r_cut: float
            maximum distances for mobile ion - framework interaction

        resolution: float
            distance between grid points

        k: int
            maximum number of neighbours (used for KDTree search of neighbors)

        f: float
            screening factor

        Returns
        ----------

        energy: np.array
            BVSE distribution
        """

        _, distances, ids, numbers =  self._neighbors(r_cut = r_cut,
                                                     resolution = resolution,
                                                     k = k)
        r_min = np.take(self._get_array('r_min'), ids, axis = -1)
        alpha = np.take(self._get_array('alpha'), ids, axis = -1)
        r_c = np.take(self._get_array('r_c'), ids, axis = -1)
        d0 = np.take(self._get_array('d0'), ids, axis = -1)
        q = np.take(self._get_array('oxi_states'), ids, axis = -1)
        q = np.where(q * self.q_mi > 0, q, 0)
        n = np.take(self._get_array('n'), ids, axis = -1)
        
        morse = np.nan_to_num(BVSEPotential.Morse(distances, r_min, d0, alpha),
                              copy = False,
                              nan = 0.0).sum(axis = 1)
        
        coulomb = np.nan_to_num(BVSEPotential.Coulomb(distances,
                                                      self.q_mi, q,
                                                      self.rc_mi, r_c,
                                                      self.n_mi, n, f = f),
                                 copy = False,
                                 nan = 0.0).sum(axis = 1)
        return morse + coulomb
    


    def _percolation_energy(self, dim, encut = 10.0):

        """
        Get percolation energy fofr a given dimensionality of percolation

        Parameters
        ----------

        dim: int
            dimensionality of percolation. 2 -> 1D, 4 -> 2D, 8 - 3D percolation
            
        encut: float, 10.0 by default
            stop criterion for the search of percolation energy
        
        Returns
        ----------
        barrier: float
            percolation energy or np.inf if no percolation found
        """
        
        data = self.data.reshape(self.size)
        return percolation.percolation_energy(data, dim, encut = encut)



    def percolation_barriers(self, encut = 10.0, n_jobs = 1, backend = 'threading'):

        """
        Find percolation energy and dimensionality of a migration network.

        Parameters
        ----------

        encut: float, 5.0 by default
            cutoff energy above which barriers supposed to be np.inf

        n_jobs: int, 1 by default
            not used, left for compatibility with the previous versions

        backend: str, 'threading' by default
            not used, left for compatibility with the previous versions

        Returns
        ----------
        
        energies: dict
            infromation about percolation {'E_1D': float, 'E_2D': float, 'E_3D': float}

        """

        self.n_jobs = n_jobs
        self.backend = backend

        data = self.data.reshape(self.size)
        return percolation.percolation_barriers(data, encut = encut)
    


    def percolation_radii(self, n_jobs = 1, backend = 'threading'):

        """
        Find the largest percolation radius of the free sphere 
        w.r.t. the dimensionality of a migration network.

        Parameters
        ----------

        n_jobs: int, 1 by default
            not used, left for compatibility with the previous versions

        backend: str, 'threading' by default
            not used, left for compatibility with the previous versions

        Returns
        ----------
        
        energies: dict
            infromation about percolation {'r_1D': float, 'r_2D': float, 'r_3D': float}

        """
        self.n_jobs = n_jobs
        self.backend = backend

        return percolation.percolation_radii(self.void_data)
    


    def write_cube(self, filename, task = 'bvse'):

        """
        Write .cube file containing structural and BVSE data.
        Note! Run it after self.bvse_distribution method

        Parameters
        ----------

        filename: str
            file name to write .cube
        task: str, "bvse" by default
            which data to write, allowed values are "void" and "bvse"
        """

        if task == 'bvse':
            data = self.data
        else:
            data = self.void_data
        nx, ny, nz = data.shape
        with open(f'{filename}.cube', 'w') as f:
            cube.write_cube(f, self.atoms_copy, data = data[:nx-1, :ny-1, :nz-1])
    


    def write_grd(self, filename, task = 'bvse'):
        
        """
        Write BVSE distribution volumetric file for VESTA 3.0.
        Note! Run it after self.bvse_distribution method

        Parameters
        ----------

        filename: str
            file name to write .grd
        task: str, "bvse" by default
            which data to write, allowed values are "void" and "bvse"
        """

        if task == 'bvse':
            data = self.data.reshape(self.size)
        else:
            data = self.void_data
        voxels = data.shape[0] - 1, data.shape[1] - 1, data.shape[2] - 1
        cellpars = self.cell.cellpar()
        values = np.asarray(data[:voxels[0], :voxels[1], :voxels[2]], dtype = np.float64)
        with open(f'{filename}.grd' , 'w') as report:
            comment = '# BVSE data made with bvlain package: https://github.com/dembart/BVlain'
            report.write(comment + '\n')
            report.write(''.join(str(p) + ' ' for p in cellpars).strip() + '\n')
            report.write(''.join(str(v) + ' ' for v in voxels).strip() + '\n')
            if values.size > 0:
                report.write('\n'.join(map(repr, values.ravel().tolist())) + '\n')
    

    
    def mismatch(self, r_cut = 3.0):
        
        """
        Calculate bond valence sum mismatch for each site.

        Parameters
        ----------
            
        r_cut: float, 3.0 by default
            cutoff radius for nearest neighbors 

        Returns
        ----------
        pd.DataFrame
            structure data and misamtches
        """

        import pandas as pd

        with open(self.cation_file, 'rb') as f:
            data_cation = pickle.load(f) 

        with open(self.anion_file, 'rb') as f:
            data_anion = pickle.load(f)

        atoms = self.atoms_copy
        centers, neighbors, distances = neighbor_list('ijd', atoms, r_cut)


        mismatch = []
        bvs_list = []
        for i, n in enumerate(atoms.numbers):

            ids = np.argwhere(centers == i).ravel()
            env = neighbors[ids]
            r = distances[ids]
            q1 = atoms.get_array('oxi_states')[i]
            n_env = atoms.numbers[env]
            q2 = atoms.get_array('oxi_states')[env]
            alpha = np.zeros(q2.shape)
            r0 = np.zeros(q2.shape)

            if q1 > 0:
                q1q2 = np.where(q1*q2 < 0, 1, 0)
                for index in np.argwhere(q1q2 == 1).ravel():
                    alpha[index] = data_cation[n][q1][n_env[index]][q2[index]]['alpha']
                    r0[index] = data_cation[n][q1][n_env[index]][q2[index]]['r0']
                bvs = np.exp(alpha * (r0 - r)) * q1q2

            if q1 < 0:
                q1q2 = np.where(q1*q2 < 0, 1, 0)
                for index in np.argwhere(q1q2 == 1).ravel():
                    alpha[index] = data_anion[n][q1][n_env[index]][q2[index]]['alpha']
                    r0[index] = data_anion[n][q1][n_env[index]][q2[index]]['r0']
                bvs = np.exp(alpha * (r0 - r)) * q1q2

            bvs_list.append(bvs.sum())
            pos = np.round(atoms.get_scaled_positions(), 4)
            mismatch.append(bvs.sum() - abs(q1))

        df = pd.DataFrame(pos, columns = ['x/a', 'y/b', 'z/c'])
        df['mismatch'] = mismatch
        df['atom'] = atoms.get_chemical_symbols()
        df['formal_charge'] = atoms.get_array('oxi_states')
        df['bvs'] = bvs_list
        return df[['atom', 'x/a', 'y/b', 'z/c', 'formal_charge', 'bvs', 'mismatch']]

    
    
    def _resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_path, relative_path)
        return path
