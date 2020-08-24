import os
import h5py
import numpy as np
import constants as ct
from scipy.spatial import cKDTree

""" Class to read required datasets of entire snapshot """

class entire_snapshot_read:
    def __init__(self, mpi, sim='/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output', snap=99):

        if not mpi.Rank: print(' > Reading entire snapshot...', flush=True)

        self.path = sim
        self.snap = snap

        self.find_split_files(mpi, sim, snap)

        self.read_basics()
        return

    def find_split_files(self, mpi, sim, snap):
        """
        Find all files associated with a snapshot

        Arguments:
          -mpi  : Instance of the mpi class
          -sim  : Path to the simulation of interest
          -snap : Snapshot of interest
        """

        # Find files
        files = []
        for x in os.listdir('{0}/snapdir_{1:03d}/'.format(sim, snap)):
            if x.startswith('snap_'):
                files.append('{0}/snapdir_{1:03d}/{2}'.format(sim, snap, x))

        # Sort files
        if len(files) > 1:
            sort_order = np.argsort(np.array([x.split('.', 2)[1] for x in files], dtype=np.int))
            files      = list(np.array(files)[sort_order])
            del sort_order

        # Read particles per file here -- for subhalo removal
        self.Fpart = np.zeros((len(files), 6), dtype=np.int)
        for j in range(0, len(files), 1):
            f             = h5py.File(files[j], 'r')
            self.Fpart[j] = f['Header'].attrs['NumPart_ThisFile']
            f.close()

        # Split files
        num_files = int(len(files) / mpi.NProcs)
        remainder = len(files) % mpi.NProcs
        start     = mpi.Rank * num_files
        finish    = (mpi.Rank + 1) * num_files
        if mpi.Rank < remainder:
            start  += mpi.Rank
            finish += mpi.Rank + 1
        else:
            start  += remainder
            finish += remainder

        self.files  = files[start:finish]
        self.start  = start
        self.finish = finish
        del files
        return

    def read_basics(self):
        """
        Read basics -- cosmology, units and particle numbers
        """

        # Read basics
        if len(self.files) > 0:
            f             = h5py.File(self.files[0], 'r')
            self.hub      = f['Header'].attrs['HubbleParam']
            self.axp      = f['Header'].attrs['Time']
            self.mass_tab = f['Header'].attrs['MassTable']
            self.redshift = f['Header'].attrs['Redshift']
            self.omega_b  = f['Header'].attrs['OmegaBaryon']
            self.omega_m  = f['Header'].attrs['Omega0']
            self.omega_l  = f['Header'].attrs['OmegaLambda']
            self.Umass    = f['Header'].attrs['UnitMass_in_g']
            self.Ulength  = f['Header'].attrs['UnitLength_in_cm']
            self.Uvelc    = f['Header'].attrs['UnitVelocity_in_cm_per_s']
            self.BoxSize  = f['Header'].attrs['BoxSize'] * self.Ulength * self.axp / self.hub
            f.close()

        # Sum particles for dataset reads
        self.npt = 0
        for j in range(0, len(self.files), 1):
            f         = h5py.File(self.files[j], 'r')
            self.npt += f['Header'].attrs['NumPart_ThisFile']
            f.close()
        return

    def read_datasets(self, mpi, rqd_dsets):
        """
        Reads all specifed datasets

        Arguments:
          -mpi       : MPI class instance
          -rqd_dsets : List of required datasets
        """

        for x in rqd_dsets:
            ptype = int(x[8])

            if not mpi.Rank: print('  -{0}'.format(x), flush=True)
            self.read_dataset(x, ptype)
        return

    def read_dataset(self, dataset, ptype, cgs=True, physical=True):
        """
        Does all the hardlifting of actually reading a dataset and converting units
        to cgs (if required).

        Arguments:
          -dataset  : HDF5 tag name of dataset to be read
          -ptype    : Particle type of dataset -- for memory allocation
          -cgs      : Convert units to CGS
          -physical : Convert units to physical -- i.e. remove a and h
        """

        # Skip if a task without files to read
        if len(self.files) <= 0: return

        # First pass for datatype, shape and units
        if dataset != 'PartType1/Masses':
            f     = h5py.File(self.files[0], 'r')
            # Check if dataset is present -- full v. mini snapshot issues
            if dataset.split('/')[1] not in list(f[dataset.split('/')[0]].keys()):
                f.close()
                return

            dtype = f[dataset].dtype
            shape = f[dataset].shape
            a_scl = f[dataset].attrs['a_scaling']
            h_scl = f[dataset].attrs['h_scaling']
            u2cgs = f[dataset].attrs['to_cgs']
            f.close()

            # Allocate memory -- note assumes at most 2D dataset
            if dtype == 'float32': dtype = np.float64

            if len(shape) > 1:
                dset = np.zeros((self.npt[ptype], shape[1]), dtype=dtype)
            else:
                dset = np.zeros(self.npt[ptype], dtype=dtype)
            del dtype, shape

            # Loop over files, read dataset
            offset = 0
            for j in range(0, len(self.files), 1):
                f                          = h5py.File(self.files[j], 'r')
                npt                        = f['Header'].attrs['NumPart_ThisFile'][ptype]
                dset[offset:offset + npt]  = f[dataset][:]
                offset                    += npt
            del offset
        else:
            a_scl = 0
            h_scl = -1
            u2cgs = self.Umass
            dset  = np.zeros(self.npt[ptype], dtype=np.float64) + self.mass_tab[ptype]

        # Convert to cgs if required
        if cgs: 
            if u2cgs != 0: dset *= u2cgs

        # Convert to physical units
        if physical: dset *= (self.axp ** a_scl) * (self.hub ** h_scl)

        # Store
        if dataset == 'PartType0/Coordinates':
            self.pos    = dset
        elif dataset == 'PartType0/Density':
            self.rho    = dset
        elif dataset == 'PartType0/ElectronAbundance':
            self.ne_nh  = dset
        elif dataset == 'PartType0/GFM_Metallicity':
            self.zmet   = dset / ct.Ztng_solar
        elif dataset == 'PartType0/InternalEnergy':
            self.inte   = dset
        elif dataset == 'PartType0/Masses':
            self.mass   = dset
        elif dataset == 'PartType0/Potential':
            self.gpot   = dset
        elif dataset == 'PartType0/Velocities':
            self.velc   = dset
        elif dataset == 'PartType1/Coordinates':
            self.DMpos  = dset
        elif dataset == 'PartType1/Masses':
            self.DMmass = dset
        elif dataset == 'PartType1/Potential':
            self.DMgpot = dset
        elif dataset == 'PartType1/Velocities':
            self.DMvelc = dset
        elif dataset == 'PartType4/Coordinates':
            self.STpos  = dset
        elif dataset == 'PartType4/Masses':
            self.STmass = dset
        elif dataset == 'PartType4/GFM_Metallicity':
            self.STzmet = dset / ct.Ztng_solar
        elif dataset == 'PartType4/GFM_StellarFormationTime':
            self.STsft  = dset
        elif dataset == 'PartType4/Potential':
            self.STgpot = dset
        elif dataset == 'PartType4/Velocities':
            self.STvelc= dset
        else:
            print(' DATASET STORE NOT SET FOR {0}!!!'.format(dataset), flush=True)
            quit()
        del dset
        return

    def remove_subhalos(self, mpi, subfind_table):
        """
        Using subfind table to remove particles bound to subhalos
        """

        if mpi.Rank == 0: print(' > Removing particles bound to subhalos', flush=True)
        # Compute my offset in the particle distribution
        Psumf     = np.zeros((len(self.Fpart) + 1, 6), dtype=np.int)
        Psumf[1:] = np.cumsum(self.Fpart, axis=0)
        offsets   = Psumf[self.start]
        my_nparts = Psumf[self.finish] - offsets
        del Psumf

        # Get central and unbound particle indices on task
        gdx = self.return_central_indices(0, offsets[0], my_nparts[0], subfind_table)
        ddx = self.return_central_indices(1, offsets[1], my_nparts[1], subfind_table)
        sdx = self.return_central_indices(4, offsets[4], my_nparts[4], subfind_table)
        bdx = self.return_central_indices(5, offsets[5], my_nparts[5], subfind_table)

        # Now remove particles bound to subhalos
        keys = sorted(self.__dict__.keys())
        for x in keys:
            if x in ['BoxSize', 'Fpart', 'Ulength', 'Umass', 'Uvelc', 'axp', 'files',
                     'finish', 'hub', 'mass_tab', 'npt', 'omega_b', 'omega_l', 'omega_m',
                     'path', 'redshift', 'start']: continue

            if x == 'pos':
                self.pos    = self.pos[gdx]
            elif x == 'rho':
                self.rho    = self.rho[gdx]
            elif x == 'ne_nh':
                self.ne_nh  = self.ne_nh[gdx]
            elif x == 'inte':
                self.inte   = self.inte[gdx]
            elif x == 'mass':
                self.mass   = self.mass[gdx]
            elif x == 'DMpos':
                self.DMpos  = self.DMpos[ddx]
            elif x == 'DMmass':
                self.DMmass = self.DMmass[ddx]
            elif x == 'STpos':
                self.STpos  = self.STpos[sdx]
            elif x == 'STmass':
                self.STmass = self.STmass[sdx]
            elif x == 'STsft':
                self.STsft  = self.STsft[sdx]
            else:
                print(' SUBHALO REMOVAL NOT SET FOR {0}!!!'.format(x), flush=True)
        del gdx, ddx, sdx, bdx
        return

    def return_central_indices(self, ptype, offset, my_npart, subfind_table):
        """
        Return indices of bound particles
        """

        # First find all subhalo bound indices
        shx = []
        for j in range(0, len(subfind_table.SubLenType), 1):
            if len(subfind_table.SubLenType[j]) <= 1: continue
            st  = subfind_table.SubLenType[j][1]
            fh  = subfind_table.SubLenType[j][-1]

            if fh[ptype] + subfind_table.OffType[j,ptype] < offset or \
               st[ptype] + subfind_table.OffType[j,ptype] > offset + my_npart: continue

            shx.append(np.arange(fh[ptype] - st[ptype]) + st[ptype] \
                       + subfind_table.OffType[j,ptype])

        if len(shx) > 0:
            shx = np.hstack(shx) - offset
            shx = np.where((shx >= 0) & (shx < my_npart))[0]
            idx = np.arange(my_npart)
            idx = np.setdiff1d(idx, shx)
            del shx
        else:
            idx = np.arange(my_npart)
        return idx

    def calculate_gas_temperatures(self, mpi):
        """ 
        Calculate temperature from internal energy & electron density
        """

        if mpi.Rank == 0: print(' > Computing gas temperatures', flush=True)
        mu        = (4.0 * ct.mp_g) / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * self.ne_nh)
        self.temp = (2.0 / 3.0) * (self.inte / ct.kB_erg_K) * mu
        return

    def remove_wind_particles(self, mpi, dsets):
        """
        Remove PartType4 wind particles
        """

        # Check we have formation time for removal
        if 'STsft' not in self.__dict__.keys():
            if mpi.Rank == 0: print('NOTE: Cannot remove wind particles', flush=True)
            return
        # Now remove wind particles
        if mpi.Rank == 0: print(' > Removing wind particles', flush=True)
        idx = np.where(self.STsft >= 0)[0]
        for x in dsets:
            if x == 'PartType4/Coordinates':
                self.STpos   = self.STpos[idx]
            elif x == 'PartType4/Masses':
                self.STmass  = self.STmass[idx]
            elif x == 'PartType4/GFM_Metallicity':
                self.STzmet  = self.STzmet[idx]
            elif x == 'PartType4/GFM_StellarFormationTime':
                self.STsft   = self.STsft[idx]
            elif x == 'PartType4/GFM_StellarPhotometrics':
                self.STphoto = self.STphoto[idx]
        del idx
        return
