import h5py
import numpy as np
import constants as ct
import shape as sh
import potential as pt
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM

# Dictionary of file output names
fname = {
    "L205n2500TNG": "TNG300_L1",
    "L205n1250TNG": "TNG300_L2",
    "L205n625TNG": "TNG300_L3",
    "L205n2500TNG_DM": "TNG300DM_L1",
    "L205n1250TNG_DM": "TNG300DM_L2",
    "L205n625TNG_DM": "TNG300DM_L3",
    "L75n1820TNG": "TNG100_L1",
    "L75n910TNG": "TNG100_L2",
    "L75n455TNG": "TNG100_L3",
    "L75n1820TNG_DM": "TNG100DM_L1",
    "L75n910TNG_DM": "TNG100DM_L2",
    "L75n455TNG_DM": "TNG100DM_L3",
}

# Dictionary of gravitational softening values used -- [DM or STAR, GAS / h]
# NOTE: I have converted the GAS values to NO h values, the values were not
#       consistent in the TNG tables (for some reason!)
# NOTE: Values here are in kpc
soften = {
    "L205n2500TNG": [0.15, 0.369],
    "L205n1250TNG": [2.95, 0.738],
    "L205n625TNG": [5.90, 1.476],
    "L75n1820TNG": [0.74, 0.185],
    "L75n910TNG": [0.15, 0.369],
    "L75n455TNG": [2.95, 0.738],
}

"""
This class stores computes which simulation is associated with a
given halo, stores it and then compute various quantities. There
are a couple of external routines, but most is consolidated here

Takes the volume class as input, which has read a snapshot  
"""


class halo:
    def __init__(self, volume, Nbins=25):
        """
        Take cosmology from the volume class instance

        Arguments:
          -volume : An instance of the entire_snapshot_read class
          -Nbins  : Number of bins in the radial profile [INTEGER]
        """

        # Boxsize, cosmology, simulation to cgs unit conversions
        self.boxsize = volume.BoxSize
        self.axp = volume.axp
        self.hubp = volume.hub
        self.redshift = volume.redshift
        self.OmegaB = volume.omega_b
        self.OmegaM = volume.omega_m
        self.OmegaL = volume.omega_l
        self.Ulength = volume.Ulength
        self.Umass = volume.Umass
        self.Uvelc = volume.Uvelc

        # Set tags for output
        self.path = volume.path
        self.simtag = self.path.split("/")[-2]
        self.fname = fname[self.simtag]
        self.snap = volume.snap

        # Create radial bins
        self.Nbins = Nbins
        self.set_up_radial_profile()
        return

    def halo_data_store(self, mpi, subfind_table, volume, Extent=5.0, R200scale=False):
        """
        Find all particles within given sphere for every halo of interest
        then send particles to desired task and store

        Arguments:
          -mpi           : An instance of the mpi class
          -subfind_table : An instance of the build_table class
          -volume        : An instance of the entire_snapshot_read class
          -Extent        : Halocentric radial extent to extract particles to [FLOAT]
          -R200scale     : BOOLEAN, if TRUE rescale the extent by halo's R200 value
        """

        if not mpi.Rank:
            print(" > Distributing particles", flush=True)

        # Set Extent of cut sphere
        self.Extent = Extent

        self.halo_data = {}
        self.Nhalos = len(subfind_table.tags)
        dims = np.array([self.boxsize, self.boxsize, self.boxsize])
        # Loop over haloes of interest
        Ntask_per_node = int(np.rint(mpi.NProcs / mpi.NNodes))
        offset = 0
        for j in range(0, self.Nhalos, 1):
            # Scale extraction range
            if R200scale:
                Extent = self.Extent * subfind_table.R200[j]
            else:
                Extent = self.Extent * ct.Mpc_cm

            # Select task to send particle data to
            destination = (j % mpi.NNodes) * Ntask_per_node + (offset % Ntask_per_node)
            if destination >= mpi.NProcs:
                destination -= mpi.NProcs

            if j > 0 and j % mpi.NNodes == mpi.NNodes - 1:
                offset += 1
            if not mpi.Rank:
                print("  -{0:04d} {1:03d}".format(j, destination), flush=True)
            if destination == mpi.Rank:
                htag = subfind_table.tags[j]
                self.halo_data[htag] = {}

            # Find contributing cells/particles -- centering on halo
            vkey = sorted(volume.__dict__.keys())
            if "pos" in vkey:
                Grad = volume.pos - subfind_table.CoP[j]
                Grad = np.where(Grad > 0.5 * dims, Grad - dims, Grad)
                Grad = np.where(Grad < -0.5 * dims, Grad + dims, Grad)
                Grad = np.sqrt((Grad ** 2.0).sum(axis=-1))
                gdx = np.where(Grad <= Extent)[0]
                del Grad
            if "DMpos" in vkey:
                DMrad = volume.DMpos - subfind_table.CoP[j]
                DMrad = np.where(DMrad > 0.5 * dims, DMrad - dims, DMrad)
                DMrad = np.where(DMrad < -0.5 * dims, DMrad + dims, DMrad)
                DMrad = np.sqrt((DMrad ** 2.0).sum(axis=-1))
                ddx = np.where(DMrad <= Extent)[0]
                del DMrad
            if "STpos" in vkey:
                STrad = volume.STpos - subfind_table.CoP[j]
                STrad = np.where(STrad > 0.5 * dims, STrad - dims, STrad)
                STrad = np.where(STrad < -0.5 * dims, STrad + dims, STrad)
                STrad = np.sqrt((STrad ** 2.0).sum(axis=-1))
                sdx = np.where(STrad <= Extent)[0]
                del STrad
            del vkey

            # Gather particles/cells on destination task
            # NOTE: No idea why I need to recast the positions arrays for this to work
            #       but after two days of testing I got bored and this fixes it.
            keys = sorted(list(volume.__dict__.keys()))
            # --- GAS
            if "pos" in keys:
                array = mpi.gatherv_single(
                    np.zeros(volume.pos[gdx].shape) + volume.pos[gdx], root=destination
                )
                if mpi.Rank == destination:
                    pos = array - subfind_table.CoP[j]
                    pos = np.where(pos > 0.5 * dims, pos - dims, pos)
                    pos = np.where(pos < -0.5 * dims, pos + dims, pos)
                    self.halo_data[subfind_table.tags[j]]["Pos"] = pos
                    del pos
            if "rho" in keys:
                array = mpi.gatherv_single(volume.rho[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Rho"] = array
            if "ne_nh" in keys:
                array = mpi.gatherv_single(volume.ne_nh[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Ne_Nh"] = array
            if "zmet" in keys:
                array = mpi.gatherv_single(volume.zmet[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Zmet"] = array
            if "mass" in keys:
                array = mpi.gatherv_single(volume.mass[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Mass"] = array
            if "sub" in keys:
                array = mpi.gatherv_single(volume.sub[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Sub"] = array
            if "temp" in keys:
                array = mpi.gatherv_single(volume.temp[gdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Temp"] = array
            if "velc" in keys:
                array = mpi.gatherv_single(
                    np.zeros(volume.velc[gdx].shape) + volume.velc[gdx],
                    root=destination,
                )
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["Velc"] = array

            # --- DM
            if "DMpos" in keys:
                array = mpi.gatherv_single(
                    np.zeros(volume.DMpos[ddx].shape) + volume.DMpos[ddx],
                    root=destination,
                )
                if mpi.Rank == destination:
                    pos = array - subfind_table.CoP[j]
                    pos = np.where(pos > 0.5 * dims, pos - dims, pos)
                    pos = np.where(pos < -0.5 * dims, pos + dims, pos)
                    self.halo_data[subfind_table.tags[j]]["DMPos"] = pos
                    del pos
            if "DMmass" in keys:
                array = mpi.gatherv_single(volume.DMmass[ddx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["DMMass"] = array
            if "DMsub" in keys:
                array = mpi.gatherv_single(volume.DMsub[ddx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["DMSub"] = array
            if "DMvelc" in keys:
                array = mpi.gatherv_single(
                    np.zeros(volume.DMvelc[ddx].shape) + volume.DMvelc[ddx],
                    root=destination,
                )
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["DMVelc"] = array

            # --- STARS
            if "STpos" in keys:
                array = mpi.gatherv_single(
                    np.zeros(volume.STpos[sdx].shape, dtype=np.float)
                    + volume.STpos[sdx],
                    root=destination,
                )
                if mpi.Rank == destination:
                    pos = array - subfind_table.CoP[j]
                    pos = np.where(pos > 0.5 * dims, pos - dims, pos)
                    pos = np.where(pos < -0.5 * dims, pos + dims, pos)
                    self.halo_data[subfind_table.tags[j]]["STPos"] = pos
                    del pos
            if "STmass" in keys:
                array = mpi.gatherv_single(volume.STmass[sdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["STMass"] = array
            if "STsub" in keys:
                array = mpi.gatherv_single(volume.STsub[sdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["STSub"] = array
            if "STvelc" in keys:
                array = mpi.gatherv_single(
                    np.zeros((len(sdx), 3), dtype=np.float) + volume.STvelc[sdx],
                    root=destination,
                )
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["STVelc"] = array
            if "STzmet" in keys:
                array = mpi.gatherv_single(volume.STzmet[sdx], root=destination)
                if mpi.Rank == destination:
                    self.halo_data[subfind_table.tags[j]]["STZmet"] = array

            # Add key SUBFIND quanitites
            if mpi.Rank == destination:
                self.halo_data[subfind_table.tags[j]]["tag"] = subfind_table.tags[j]
                self.halo_data[subfind_table.tags[j]]["hub"] = subfind_table.hub
                self.halo_data[subfind_table.tags[j]]["axp"] = subfind_table.axp
                self.halo_data[subfind_table.tags[j]][
                    "redshift"
                ] = subfind_table.redshift
                self.halo_data[subfind_table.tags[j]][
                    "rho_crit"
                ] = subfind_table.rho_crit
                self.halo_data[subfind_table.tags[j]]["CoP"] = subfind_table.CoP[j]
                self.halo_data[subfind_table.tags[j]]["M200"] = subfind_table.M200[j]
                self.halo_data[subfind_table.tags[j]]["M500"] = subfind_table.M500[j]
                self.halo_data[subfind_table.tags[j]]["Mvir"] = subfind_table.Mvir[j]
                self.halo_data[subfind_table.tags[j]]["R200"] = subfind_table.R200[j]
                self.halo_data[subfind_table.tags[j]]["R500"] = subfind_table.R500[j]
                self.halo_data[subfind_table.tags[j]]["Rvir"] = subfind_table.Rvir[j]
                self.halo_data[subfind_table.tags[j]]["Vbulk"] = subfind_table.Vbulk[j]
                self.halo_data[subfind_table.tags[j]][
                    "LenType"
                ] = subfind_table.GrLenType[j]
                self.halo_data[subfind_table.tags[j]][
                    "OffType"
                ] = subfind_table.OffType[j]
                self.halo_data[subfind_table.tags[j]]["Nsubs"] = subfind_table.Nsubs[j]
                self.halo_data[subfind_table.tags[j]][
                    "SubLenType"
                ] = subfind_table.SubLenType[j]
            mpi.comm.Barrier()
        return

    def set_up_halo(self, mpi, tag):
        """
        Set up the required arrays for this halo as other classes expect

        Arguments:
          -mpi : An instance of the mpi class
          -tag : STRING labelling this halo for property storage
        """

        # Basic halo propertie
        self.tag = tag
        self.hub = self.halo_data[tag]["hub"]
        self.axp = self.halo_data[tag]["axp"]
        self.redshift = self.halo_data[tag]["redshift"]
        self.rho_crit = self.halo_data[tag]["rho_crit"]
        self.CoP = self.halo_data[tag]["CoP"]
        self.M200 = self.halo_data[tag]["M200"]
        self.M500 = self.halo_data[tag]["M500"]
        self.Mvir = self.halo_data[tag]["Mvir"]
        self.R200 = self.halo_data[tag]["R200"]
        self.R500 = self.halo_data[tag]["R500"]
        self.Rvir = self.halo_data[tag]["Rvir"]
        self.Vbulk = self.halo_data[tag]["Vbulk"]
        self.LenType = self.halo_data[tag]["LenType"]
        self.OffType = self.halo_data[tag]["OffType"]
        self.Nsubs = self.halo_data[tag]["Nsubs"]
        self.SubLenType = self.halo_data[tag]["SubLenType"]

        # Datasets
        keys = sorted(list(self.halo_data[tag].keys()))
        if "Mass" in keys:
            self.mass = self.halo_data[tag]["Mass"]
        if "Ne_Nh" in keys:
            self.ne_nh = self.halo_data[tag]["Ne_Nh"]
        if "Pos" in keys:
            self.pos = self.halo_data[tag]["Pos"]
        if "Rho" in keys:
            self.rho = self.halo_data[tag]["Rho"]
        if "Sub" in keys:
            self.sub = self.halo_data[tag]["Sub"]
        if "Temp" in keys:
            self.temp = self.halo_data[tag]["Temp"]
        if "Velc" in keys:
            self.velc = self.halo_data[tag]["Velc"]
        if "Zmet" in keys:
            self.zmet = self.halo_data[tag]["Zmet"]
        if "DMMass" in keys:
            self.DMmass = self.halo_data[tag]["DMMass"]
        if "DMPos" in keys:
            self.DMpos = self.halo_data[tag]["DMPos"]
        if "DMSub" in keys:
            self.DMsub = self.halo_data[tag]["DMSub"]
        if "DMVelc" in keys:
            self.DMvelc = self.halo_data[tag]["DMVelc"]
        if "STMass" in keys:
            self.STmass = self.halo_data[tag]["STMass"]
        if "STPos" in keys:
            self.STpos = self.halo_data[tag]["STPos"]
        if "STSub" in keys:
            self.STsub = self.halo_data[tag]["STSub"]
        if "STVelc" in keys:
            self.STvelc = self.halo_data[tag]["STVelc"]
        if "STZmet" in keys:
            self.STzmet = self.halo_data[tag]["STZmet"]
        del keys, self.halo_data[tag]

        # Compute radii
        if "pos" in self.__dict__.keys():
            self.rad = np.sqrt((self.pos ** 2.0).sum(axis=-1)) / self.R200
        if "DMpos" in self.__dict__.keys():
            self.DMrad = np.sqrt((self.DMpos ** 2.0).sum(axis=-1)) / self.R200
        if "STpos" in self.__dict__.keys():
            self.STrad = np.sqrt((self.STpos ** 2.0).sum(axis=-1)) / self.R200
        return

    def compute_gravitational_potentials(self, mpi):
        """
        Compute the gravitational potential of all particles of a given halo

        Arguments:
          -mpi : An instance of the mpi class
        """

        if not mpi.Rank:
            print(" > Computing gravitational potential", flush=True)
        # First build a KD tree for all particles
        DMoff = len(self.mass)
        SToff = DMoff + len(self.DMmass)

        pos = np.concatenate((self.pos, self.DMpos, self.STpos), axis=0)
        mass = np.concatenate((self.mass, self.DMmass, self.STmass), axis=0)
        soft = np.concatenate(
            (
                np.zeros(len(self.pos), dtype=np.float) + soften[self.simtag][1],
                np.zeros(len(self.DMpos), dtype=np.float) + soften[self.simtag][0],
                np.zeros(len(self.STpos), dtype=np.float) + soften[self.simtag][0],
            ),
            axis=0,
        )

        # Check all particle positions are unique -- if not, minor shift overlapping
        pos = _check_points_unique(pos)

        if not mpi.Rank:
            print("  -Building tree...", flush=True)
        tree = pt.construct_tree(pos, mass, soft)

        # Now compute potential
        if not mpi.Rank:
            print("  -Computing potential...", flush=True)
        pot = pt.compute_potential_via_tree(pos, tree)
        del pos, mass, soft, tree

        # Set potentials for particles
        self.gpot = pot[:DMoff]
        self.DMgpot = pot[DMoff:SToff]
        self.STgpot = pot[SToff:]
        del DMoff, SToff
        return

    def compute_shape(self, mpi, aperture="500", ptype="GAS", remove_subs=True):
        """
        Compute shape within aperture

        Arguments:
          -mpi         : An instance of the mpi class
          -aperture    : STRING defining the radial aperture of interest
          -ptype       : STRING defining the particle type (e.g. dark matter) of interest
          -remove_subs : BOOLEAN, if TRUE remove particles bound to substructures
        """

        if not mpi.Rank:
            print(
                " > Computing {0} shape - aperture: {1}".format(ptype, aperture),
                flush=True,
            )

        # Compute aperture
        if aperture == "500":
            ap = self.R500 / ct.Mpc_cm
        elif aperture == "200":
            ap = self.R200 / ct.Mpc_cm
        elif aperture == "Vir":
            ap = self.Rvir / ct.Mpc_cm
        else:
            print(
                "ERROR:\n --> {0} aperture not implemented!\nEXITING...".format(
                    aperture
                )
            )
            quit()

        # Check particles type, select those with aperture
        if ptype == "GAS":
            pos = np.copy(self.pos) / ct.Mpc_cm
            mass = np.copy(self.mass) / ct.Msun_g
        elif ptype == "DM":
            pos = np.copy(self.DMpos) / ct.Mpc_cm
            mass = np.copy(self.DMmass) / ct.Msun_g
        elif ptype == "STAR":
            pos = np.copy(self.STpos) / ct.Mpc_cm
            mass = np.copy(self.STmass) / ct.Msun_g
        else:
            print(
                "ERROR:\n --> {0} particle type not implemented!\nEXITING...".format(
                    ptype
                )
            )
            quit()

        # Remove those in substuctures -- if required
        if remove_subs:
            if ptype == "GAS":
                sdx = np.where(self.sub == 0)[0]
            elif ptype == "DM":
                sdx = np.where(self.DMsub == 0)[0]
            elif ptype == "STAR":
                sdx = np.where(self.STsub == 0)[0]
        else:
            sdx = np.arange(len(mass))

        # Actual shape calculation -- check for empty aperture
        if len(sdx) <= 0:
            q = 0.0
            s = 0.0
            Ivectors = np.zeros((3, 3), dtype=np.float)
        else:
            try:
                q, s, Ivectors = sh.iterative_cumulative_shape_measure(
                    pos[sdx], mass[sdx], rmax=ap
                )
            except:
                q = 0.0
                s = 0.0
                Ivectors = np.zeros((3, 3), dtype=np.float)

        # Store and return
        if ptype == "GAS":
            if aperture == "500":
                self.s_gas_500 = s
                self.q_gas_500 = q
                self.Iv_gas_500 = Ivectors
            elif aperture == "200":
                self.s_gas_200 = s
                self.q_gas_200 = q
                self.Iv_gas_200 = Ivectors
            elif aperture == "Vir":
                self.s_gas_vir = s
                self.q_gas_vir = q
                self.Iv_gas_vir = Ivectors
        elif ptype == "DM":
            if aperture == "500":
                self.s_dm_500 = s
                self.q_dm_500 = q
                self.Iv_dm_500 = Ivectors
            elif aperture == "200":
                self.s_dm_200 = s
                self.q_dm_200 = q
                self.Iv_dm_200 = Ivectors
            elif aperture == "Vir":
                self.s_dm_vir = s
                self.q_dm_vir = q
                self.Iv_dm_vir = Ivectors
        elif ptype == "STAR":
            if aperture == "500":
                self.s_st_500 = s
                self.q_st_500 = q
                self.Iv_st_500 = Ivectors
            elif aperture == "200":
                self.s_st_200 = s
                self.q_st_200 = q
                self.Iv_st_200 = Ivectors
            elif aperture == "Vir":
                self.s_st_vir = s
                self.q_st_vir = q
                self.Iv_st_vir = Ivectors
        del s, q, Ivectors
        return

    def compute_shape_profile(self, mpi, ptype="GAS", remove_subs=True):
        """
        Iteratively compute the shape profile of the gas cells

        Arguments:
          -mpi         : An instance of the mpi class
          -pytpe       : STRING defining the particle type (e.g. dark matter) of interest
          -remove_subs : BOOLEAN, if TRUE remove particles bound to substructures
        """

        if not mpi.Rank:
            print(
                " > Computing {0} iterative inertial tensor".format(ptype), flush=True
            )

        # Bin relevant particle type
        if ptype in ["GAS", "GAS_HOT", "GAS_TEMP", "GAS_PRES"]:
            pos = np.copy(self.pos) / ct.Mpc_cm
            mass = np.copy(self.mass) / ct.Msun_g  # [Msun]
            if ptype in ["GAS_HOT", "GAS_TEMP", "GAS_PRES"]:
                hdx = np.where(
                    (self.temp > 1.0e6) & (self.rho * 0.752 / ct.mp_g < 0.1)
                )[0]
                pos = pos[hdx]
            if ptype == "GAS_HOT":
                mass = mass[hdx]  # [Msun]
            elif ptype == "GAS_TEMP":
                mass = np.copy(self.temp)[hdx]  # [K]
            elif ptype == "GAS_PRES":
                mass = (
                    self.rho[hdx]
                    / (ct.mu * ct.mp_g)
                    * self.temp[hdx]
                    * ct.kB_erg_K
                    / ct.kev_2_erg
                )  # [keV/cm^3]
        elif ptype == "DM":
            pos = np.copy(self.DMpos) / ct.Mpc_cm
            mass = np.copy(self.DMmass) / ct.Msun_g
        elif ptype == "STAR":
            pos = np.copy(self.STpos) / ct.Mpc_cm
            mass = np.copy(self.STmass) / ct.Msun_g
        else:
            print(
                "ERROR:\n --> {0} particle type not implemented!\nEXITING...".format(
                    ptype
                )
            )
            quit()

        # Remove those in substuctures -- if required
        if remove_subs:
            if ptype == "GAS":
                sdx = np.where(self.sub == 0)[0]
            elif ptype == "DM":
                sdx = np.where(self.DMsub == 0)[0]
            elif ptype == "STAR":
                sdx = np.where(self.STsub == 0)[0]
        else:
            sdx = np.arange(len(mass))

        # Actual shape profile measurement -- check for no particles
        if len(sdx) <= 0:
            q = np.zeros(self.Nbins, dtype=np.float)
            s = np.zeros(self.Nbins, dtype=np.float)
            Ivectors = np.zeros((self.Nbins, 3, 3), dtype=np.float)
        else:
            try:
                q, s, Ivectors = sh.iterative_radial_shape_profile(
                    pos, mass, self.R200 / ct.Mpc_cm
                )
            except:
                q = np.zeros(self.Nbins, dtype=np.float)
                s = np.zeros(self.Nbins, dtype=np.float)
                Ivectors = np.zeros((self.Nbins, 3, 3), dtype=np.float)

        # Store and return
        if ptype == "GAS":
            self.s_gas = s
            self.q_gas = q
            self.Iv_gas = Ivectors
        elif ptype == "GAS_HOT":
            self.s_ghot = s
            self.q_ghot = q
            self.Iv_ghot = Ivectors
        elif ptype == "GAS_TEMP":
            self.s_temp = s
            self.q_temp = q
            self.Iv_temp = Ivectors
        elif ptype == "GAS_PRES":
            self.s_pres = s
            self.q_pres = q
            self.Iv_pres = Ivectors
        elif ptype == "DM":
            self.s_dm = s
            self.q_dm = q
            self.Iv_dm = Ivectors
        elif ptype == "STAR":
            self.s_star = s
            self.q_star = q
            self.Iv_star = Ivectors
        del s, q, Ivectors
        return

    def compute_mass_profiles(self, mpi, Nb=25):
        """
        Compute mass profiles and mass-weighted gas temperature profile

        Arguemnts:
          -mpi : An instance of the mpi class
          -Nb  : Number of bins in the radial profile [INTEGER]
        """

        if not mpi.Rank:
            print(" > Computing mass profiles", flush=True)

        if "rad" in self.__dict__.keys():
            self.GASpro = np.histogram(self.rad, bins=self.bins, weights=self.mass)[0]

            if "temp" in self.__dict__.keys():
                hdx = np.where(
                    (self.temp > 1.0e6) & (self.rho * 0.752 / ct.mp_g < 0.1)
                )[0]
                self.GHOTpro = np.histogram(
                    self.rad[hdx], bins=self.bins, weights=self.mass[hdx]
                )[0]
        if "DMrad" in self.__dict__.keys():
            self.DMpro = np.histogram(self.DMrad, bins=self.bins, weights=self.DMmass)[
                0
            ]
        if "STrad" in self.__dict__.keys():
            self.STpro = np.histogram(self.STrad, bins=self.bins, weights=self.STmass)[
                0
            ]

        # Temperature profile
        if "rad" in self.__dict__.keys():
            self.TEMPpro = np.histogram(
                self.rad, bins=self.bins, weights=self.mass * self.temp
            )[0]
            idx = np.where(self.GASpro != 0.0)[0]
            self.TEMPpro[idx] /= self.GASpro[idx]
            del idx

        # Total mass profile for NFW fit
        self.TOTALpro = np.zeros(Nb, dtype=np.float)
        if "rad" in self.__dict__.keys():
            self.TOTALpro += self.GASpro
        if "DMrad" in self.__dict__.keys():
            self.TOTALpro += self.DMpro
        if "STrad" in self.__dict__.keys():
            self.TOTALpro += self.STpro
        if self.TOTALpro.sum() <= 0.0:
            del self.TOTALpro
        return

    def compute_velocities_and_non_thermal_pressure(self, mpi, Nb=25):
        """
        Compute the non-thermal pressure profile

        Arguments:
          -mpi : An instance of the mpi class
          -Nb  : Number of bins in the radial profile [INTEGER]
        """

        if not mpi.Rank:
            print(" > Computing non-thermal pressure profile", flush=True)

        # Remove bulk, add Hubble flow -- !!! TNG COSMOLOGY HARD WIRED !!!
        H_z = (
            np.sqrt(0.3089 * (1.0 + self.redshift) ** 3.0 + 0.6911)
            * 100.0
            * self.hub
            * ct.km_cm
        )

        if "rad" in self.__dict__.keys():
            ghub = (sh.vnorm_rp(self.pos).T * H_z * self.rad * self.R200 / ct.Mpc_cm).T
            Gvel = self.velc - self.Vbulk + ghub
            vmag = np.sqrt(((Gvel - ghub) ** 2.0).sum(axis=-1))
            # del ghub
        if "DMrad" in self.__dict__.keys():
            dhub = (
                sh.vnorm_rp(self.DMpos).T * H_z * self.DMrad * self.R200 / ct.Mpc_cm
            ).T
            Dvel = self.DMvelc - self.Vbulk + dhub
            del dhub
        if "STrad" in self.__dict__.keys():
            shub = (
                sh.vnorm_rp(self.STpos).T * H_z * self.STrad * self.R200 / ct.Mpc_cm
            ).T
            Svel = self.STvelc - self.Vbulk + shub
            del shub
        del H_z

        # --- Compute r, theta, phi velocities
        # GAS
        if "rad" in self.__dict__.keys():
            vr_g = (self.pos * Gvel).sum(axis=-1) / np.sqrt(
                (self.pos ** 2.0).sum(axis=-1)
            )
            vt_g = (
                Gvel[:, 0] * self.pos[:, 1] - self.pos[:, 0] * Gvel[:, 1]
            ) / np.sqrt((self.pos[:, 0:2] ** 2.0).sum(axis=-1))
            vp_g = (
                self.pos[:, 2]
                * (self.pos[:, 0] * Gvel[:, 0] + self.pos[:, 1] * Gvel[:, 1])
                - Gvel[:, 2] * (self.pos[:, 0:2] ** 2.0).sum(axis=-1)
            ) / (
                np.sqrt((self.pos ** 2.0).sum(axis=-1))
                * np.sqrt((self.pos[:, 0:2] ** 2.0).sum(axis=-1))
            )
        # DM
        if "DMrad" in self.__dict__.keys():
            vr_d = (self.DMpos * Dvel).sum(axis=-1) / np.sqrt(
                (self.DMpos ** 2.0).sum(axis=-1)
            )
            vt_d = (
                Dvel[:, 0] * self.DMpos[:, 1] - self.DMpos[:, 0] * Dvel[:, 1]
            ) / np.sqrt((self.DMpos[:, 0:2] ** 2.0).sum(axis=-1))
            vp_d = (
                self.DMpos[:, 2]
                * (self.DMpos[:, 0] * Dvel[:, 0] + self.DMpos[:, 1] * Dvel[:, 1])
                - Dvel[:, 2] * (self.DMpos[:, 0:2] ** 2.0).sum(axis=-1)
            ) / (
                np.sqrt((self.DMpos ** 2.0).sum(axis=-1))
                * np.sqrt((self.DMpos[:, 0:2] ** 2.0).sum(axis=-1))
            )
        # STARS
        if "STrad" in self.__dict__.keys():
            vr_s = (self.STpos * Svel).sum(axis=-1) / np.sqrt(
                (self.STpos ** 2.0).sum(axis=-1)
            )
            vt_s = (
                Svel[:, 0] * self.STpos[:, 1] - self.STpos[:, 0] * Svel[:, 1]
            ) / np.sqrt((self.STpos[:, 0:2] ** 2.0).sum(axis=-1))
            vp_s = (
                self.STpos[:, 2]
                * (self.STpos[:, 0] * Svel[:, 0] + self.STpos[:, 1] * Svel[:, 1])
                - Svel[:, 2] * (self.STpos[:, 0:2] ** 2.0).sum(axis=-1)
            ) / (
                np.sqrt((self.STpos ** 2.0).sum(axis=-1))
                * np.sqrt((self.STpos[:, 0:2] ** 2.0).sum(axis=-1))
            )

        # --- Compute mass-weighted velocity profiles
        # GAS
        if "rad" in self.__dict__.keys():
            mass = np.histogram(self.rad, bins=self.bins, weights=self.mass)[0]
            self.vr_gas = np.histogram(
                self.rad, bins=self.bins, weights=self.mass * vr_g
            )[0]
            self.vt_gas = np.histogram(
                self.rad, bins=self.bins, weights=self.mass * vt_g
            )[0]
            self.vp_gas = np.histogram(
                self.rad, bins=self.bins, weights=self.mass * vp_g
            )[0]

            idx = np.where(mass > 0.0)[0]
            self.vr_gas[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vt_gas[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vp_gas[idx] /= mass[idx] * ct.km_cm  # [km/s]
            del mass, idx
        # DM
        if "DMrad" in self.__dict__.keys():
            mass = np.histogram(self.DMrad, bins=self.bins, weights=self.DMmass)[0]
            self.vr_dm = np.histogram(
                self.DMrad, bins=self.bins, weights=self.DMmass * vr_d
            )[0]
            self.vt_dm = np.histogram(
                self.DMrad, bins=self.bins, weights=self.DMmass * vt_d
            )[0]
            self.vp_dm = np.histogram(
                self.DMrad, bins=self.bins, weights=self.DMmass * vp_d
            )[0]

            idx = np.where(mass > 0.0)[0]
            self.vr_dm[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vt_dm[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vp_dm[idx] /= mass[idx] * ct.km_cm  # [km/s]
            del mass, idx
        # STARS
        if "STrad" in self.__dict__.keys():
            mass = np.histogram(self.STrad, bins=self.bins, weights=self.STmass)[0]
            self.vr_star = np.histogram(
                self.STrad, bins=self.bins, weights=self.STmass * vr_s
            )[0]
            self.vt_star = np.histogram(
                self.STrad, bins=self.bins, weights=self.STmass * vt_s
            )[0]
            self.vp_star = np.histogram(
                self.STrad, bins=self.bins, weights=self.STmass * vp_s
            )[0]

            idx = np.where(mass > 0.0)[0]
            self.vr_star[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vt_star[idx] /= mass[idx] * ct.km_cm  # [km/s]
            self.vp_star[idx] /= mass[idx] * ct.km_cm  # [km/s]
            del mass, idx

        # --- Velocity dispersion
        # GAS
        if "rad" in self.__dict__.keys():
            self.sigr_gas = np.zeros(Nb, dtype=np.float)
            self.sigt_gas = np.zeros(Nb, dtype=np.float)
            self.sigp_gas = np.zeros(Nb, dtype=np.float)

            gdx = np.digitize(self.rad, self.bins) - 1
            for j in np.unique(gdx):
                if j >= 0 and j < Nb:
                    self.sigr_gas[j] = (
                        np.sqrt(
                            np.average(
                                (vr_g[gdx == j] - self.vr_gas[j]) ** 2.0,
                                weights=self.mass[gdx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigt_gas[j] = (
                        np.sqrt(
                            np.average(
                                (vt_g[gdx == j] - self.vt_gas[j]) ** 2.0,
                                weights=self.mass[gdx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigp_gas[j] = (
                        np.sqrt(
                            np.average(
                                (vp_g[gdx == j] - self.vp_gas[j]) ** 2.0,
                                weights=self.mass[gdx == j],
                            )
                        )
                        / ct.km_cm
                    )
            del gdx
        # DM
        if "DMrad" in self.__dict__.keys():
            self.sigr_dm = np.zeros(Nb, dtype=np.float)
            self.sigt_dm = np.zeros(Nb, dtype=np.float)
            self.sigp_dm = np.zeros(Nb, dtype=np.float)

            ddx = np.digitize(self.DMrad, self.bins) - 1
            for j in np.unique(ddx):
                if j >= 0 and j < Nb:
                    self.sigr_dm[j] = (
                        np.sqrt(
                            np.average(
                                (vr_d[ddx == j] - self.vr_dm[j]) ** 2.0,
                                weights=self.DMmass[ddx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigt_dm[j] = (
                        np.sqrt(
                            np.average(
                                (vt_d[ddx == j] - self.vt_dm[j]) ** 2.0,
                                weights=self.DMmass[ddx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigp_dm[j] = (
                        np.sqrt(
                            np.average(
                                (vp_d[ddx == j] - self.vp_dm[j]) ** 2.0,
                                weights=self.DMmass[ddx == j],
                            )
                        )
                        / ct.km_cm
                    )
            del ddx
        # STARS
        if "STrad" in self.__dict__.keys():
            self.sigr_star = np.zeros(Nb, dtype=np.float)
            self.sigt_star = np.zeros(Nb, dtype=np.float)
            self.sigp_star = np.zeros(Nb, dtype=np.float)

            sdx = np.digitize(self.STrad, self.bins) - 1
            for j in np.unique(sdx):
                if j >= 0 and j < Nb:
                    self.sigr_star[j] = (
                        np.sqrt(
                            np.average(
                                (vr_s[sdx == j] - self.vr_star[j]) ** 2.0,
                                weights=self.STmass[sdx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigt_star[j] = (
                        np.sqrt(
                            np.average(
                                (vt_s[sdx == j] - self.vt_star[j]) ** 2.0,
                                weights=self.STmass[sdx == j],
                            )
                        )
                        / ct.km_cm
                    )
                    self.sigp_star[j] = (
                        np.sqrt(
                            np.average(
                                (vp_s[sdx == j] - self.vp_star[j]) ** 2.0,
                                weights=self.STmass[sdx == j],
                            )
                        )
                        / ct.km_cm
                    )
            del sdx

        # Non-thermal pressure
        self.vols = (
            (4.0 / 3.0)
            * np.pi
            * ((self.bins[1:] * self.R200) ** 3.0 - (self.bins[:-1] * self.R200) ** 3.0)
        )
        if "rad" in self.__dict__.keys():
            rho = self.GASpro / self.vols
            vmag2 = (
                self.sigr_gas ** 2.0 + self.sigt_gas ** 2.0 + self.sigp_gas ** 2.0
            ) * 1.0e10

            self.Pkin = rho * vmag2
            del rho, vmag2
        return

    def compute_thermo_profiles(self, mpi, Nb=25):
        """
        Compute various thermodynamic profiles

        Arguments:
          -mpi : An instance of the mpi class
          -Nb  : Number of bins in the radial profile [INTEGER]
        """

        if not mpi.Rank:
            print(" > Computing thermodynamic profiles", flush=True)

        # Find hot, non-star forming (density cut) gas
        idx = np.where((self.temp > 1.0e6) & (self.rho * 0.752 / ct.mp_g < 0.1))[0]

        # Hot gas density profile
        mass_hot = np.histogram(self.rad[idx], bins=self.bins, weights=self.mass[idx])[
            0
        ]
        hdx = np.where(mass_hot > 0.0)[0]
        self.Rho_hot = (mass_hot / self.vols) * (
            ct.Mpc_cm ** 3.0 / ct.Msun_g
        )  # [Msun / Mpc^3]

        # Spectroscopic-like temperature profile
        self.Tsl = np.zeros(len(self.cens), dtype=np.float64)
        wgts1 = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=((self.rho[idx] * (ct.Mpc_cm ** 3.0) / ct.Msun_g) ** 2.0)
            / (self.temp[idx] ** 0.5),
        )[0]
        wgts2 = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=((self.rho[idx] * (ct.Mpc_cm ** 3.0) / ct.Msun_g) ** 2.0)
            / (self.temp[idx] ** 1.5),
        )[0]
        self.Tsl[hdx] = (ct.kB_erg_K / ct.kev_2_erg) * (
            wgts1[hdx] / wgts2[hdx]
        )  # [keV]

        # Clumping density profiles
        mass = np.histogram(self.rad, bins=self.bins, weights=self.mass)[0]
        mdx = np.where(mass > 0.0)[0]

        self.rho_sq = np.histogram(
            self.rad, bins=self.bins, weights=self.mass * self.rho * self.rho
        )[0]
        self.rho_sq[mdx] /= mass[mdx] * (
            ct.Msun_g ** 2.0 / ct.Mpc_cm ** 6.0
        )  # [Msun^2 / Mpc^6]

        self.rho_sq_hot = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=self.mass[idx] * self.rho[idx] * self.rho[idx],
        )[0]
        self.rho_sq_hot[hdx] /= mass_hot[hdx] * (
            ct.Msun_g ** 2.0 / ct.Mpc_cm ** 6.0
        )  # [Msun^2 / Mpc^6]

        # Clumping pressure profiles
        pres = ct.kB_erg_K * self.temp * self.rho / (ct.mu * ct.mp_g)

        self.pres_sq = np.histogram(
            self.rad, bins=self.bins, weights=self.mass * pres * pres
        )[0]
        self.pres_sq[mdx] /= mass[mdx]  # [erg^2 / cm^6]

        self.pres_sq_hot = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=self.mass[idx] * pres[idx] * pres[idx],
        )[0]
        self.pres_sq_hot[mdx] /= mass[mdx]  # [erg^2 / cm^6]
        del pres

        # Emission measure profile (hot gas only)
        ne = (self.ne_nh[idx] * 0.76 * self.rho[idx] / ct.mp_g) ** 2.0

        sp_wgt = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=(self.rho[idx] * (ct.Mpc_cm ** 3.0 / ct.Msun_g) ** 2.0)
            * (self.temp[idx] ** 0.5),
        )[0]
        self.emm = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=(self.rho[idx] * (ct.Mpc_cm ** 3.0 / ct.Msun_g) ** 2.0)
            * (self.temp[idx] ** 0.5)
            * ne,
        )[0]
        edx = np.where(sp_wgt > 0.0)[0]
        self.emm[edx] /= sp_wgt[edx]  # [cm^-6]
        del edx, sp_wgt, ne
        return

    def compute_metallicity_profiles(self, mpi, Nb=25):
        """
        Compute gaseous and stellar metallcity profiles

        Arguments:
          -mpi : An instance of the mpi class
          -Nb  : Number of bins in the radial profile [INTEGER]
        """

        if not mpi.Rank:
            print(" > Computing metallicity profiles", flush=True)

        # Find hot, non-star forming (density cut) gas
        idx = np.where((self.temp > 1.0e6) & (self.rho * 0.752 / ct.mp_g < 0.1))[0]

        # Gas metallicity profile
        sp_wgt = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=(self.rho[idx] * (ct.Mpc_cm ** 3.0 / ct.Msun_g) ** 2.0)
            * (self.temp[idx] ** 0.5),
        )[0]
        self.Zgas = np.histogram(
            self.rad[idx],
            bins=self.bins,
            weights=(self.rho[idx] * (ct.Mpc_cm ** 3.0 / ct.Msun_g) ** 2.0)
            * (self.temp[idx] ** 0.5)
            * self.zmet[idx],
        )[0]
        edx = np.where(sp_wgt > 0.0)[0]
        self.Zgas[edx] /= sp_wgt[edx]
        del sp_wgt, edx

        # Stellar metallicity profile
        smass = np.histogram(self.STrad, bins=self.bins, weights=self.STmass)[0]
        self.Zstar = np.histogram(
            self.STrad, bins=self.bins, weights=self.STmass * self.STzmet
        )[0]
        idx = np.where(smass > 0.0)[0]
        self.Zstar[idx] /= smass[idx]
        del smass, idx
        return

    def compute_observable_properties(self, mpi, Emin=0.5, Emax=2.0, Enorm=65.0):
        """
        Compute the X-ray luminossity and SZ signal - try do everything in place

        Arguments:
          -mpi   : An instance of the mpi class
          -Emin  : Minimum value in the X-ray band [FLOAT]
          -Emax  : Maximum value in the X-ray band [FLOAT]
          -Enorm : Emission measure normalization value [FLOAT]
        """

        if not mpi.Rank:
            print(" > Computing observable properties", flush=True)

        self.Enorm = Enorm
        # We only use "hot" particles that are non-star-forming
        hdx = np.where((self.temp >= 1.0e6) & (self.rho * 0.752 / ct.mp_g < 0.1))[0]

        # --- Compton-y signal
        Ysz = (
            (ct.sigT / (ct.me_keV * ct.kev_2_erg))
            * ct.kB_erg_K
            * self.temp[hdx]
            * (self.mass[hdx] * 0.752 * self.ne_nh[hdx] / ct.mp_g)
            / ct.Mpc_cm ** 2.0
        )

        # Ysz profile
        self.Ysz_pro = np.histogram(self.rad[hdx], bins=self.bins, weights=Ysz)[0]

        # Ysz aperture values
        self.Ysz_500 = Ysz[self.rad[hdx] <= self.R500 / self.R200].sum()
        self.Ysz_200 = Ysz[self.rad[hdx] <= 1.0].sum()
        self.Ysz_5r500 = Ysz[self.rad[hdx] <= 5.0 * self.R500 / self.R200].sum()
        del Ysz

        # --- X-ray luminosity
        self._read_APEC_table()
        itemp = _locate(self.APEC_temperatures, np.log10(self.temp[hdx]))
        ne2dV = (self.mass[hdx] / self.rho[hdx]) * (
            self.ne_nh[hdx] * 0.76 * self.rho[hdx] / ct.mp_g
        ) ** 2.0

        Xspec = np.zeros((len(itemp), len(self.APEC_energies)), dtype=np.float)
        Xspec += self.APEC_H[itemp]
        Xspec += self.APEC_He[itemp]
        Xspec += (self.APEC_C[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_N[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_O[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_Ne[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_Mg[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_Si[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_S[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_Ca[itemp].T * self.zmet[hdx]).T
        Xspec += (self.APEC_Fe[itemp].T * self.zmet[hdx]).T
        Xspec = (Xspec.T * ne2dV).T
        del itemp

        # We want a soft-band luminosity
        edx = np.where((self.APEC_energies >= Emin) & (self.APEC_energies <= Emax))[0]
        Lx_sft = Xspec[:, edx].sum(axis=-1)

        # Soft-band Lx profile
        self.Lx_pro = np.histogram(self.rad[hdx], bins=self.bins, weights=Lx_sft)[0]

        # Lx aperture values
        self.Lx_500 = Lx_sft[self.rad[hdx] <= self.R500 / self.R200].sum()
        self.Lx_500ce = Lx_sft[
            (self.rad[hdx] > 0.15 * self.R500 / self.R200)
            & (self.rad[hdx] <= self.R500 / self.R200)
        ].sum()
        self.Lx_200 = Lx_sft[self.rad[hdx] <= 1.0].sum()
        del edx, Lx_sft

        # --- X-ray temperature
        self._calculate_photon_conversion_factor()

        idx = np.digitize(self.rad[hdx], self.bins) - 1
        self.Tx_pro = np.zeros(self.Nbins, dtype=np.float)
        for j in np.unique(idx):
            # X-ray spectrum in annulus
            Xspec_ann = Xspec[idx == j].sum(axis=0) * self.photon_conv

            # Initial fit guesses - temperature, density, metallicity
            T = np.log10(
                np.sum((self.mass[hdx] * self.temp[hdx])[idx == j])
                / np.sum((self.mass[hdx])[idx == j])
            )
            D = (
                np.log10(
                    np.sum((self.mass[hdx] * ne2dV)[idx == j])
                    / np.sum((self.mass[hdx])[idx == j])
                )
                - self.Enorm
            )
            Z = np.sum((self.mass[hdx] * self.zmet[hdx])[idx == j]) / np.sum(
                (self.mass[hdx])[idx == j]
            )

            # Fit spectrum for temperature
            try:
                limits = ([T - 0.5, -6.0, 1.0e-6], [T + 0.5, 6.0, 10.0])
                fit = least_squares(
                    self._spectrum_model,
                    [T, D, Z],
                    args=(Xspec_ann, "FIT_LS"),
                    bounds=limits,
                    method="trf",
                )
            except:
                limits = [(T - 0.5, T + 0.5), (-6.0, 6.0), (1.0e-6, 10.0)]
                fit = minimize(
                    self._spectrum_model,
                    [T, D, Z],
                    args=(Xspec_ann, "FIT_MN"),
                    bounds=limits,
                    method="TNC",
                    options={"maxiter": 200},
                )
            self.Tx_pro[j] = (ct.kB_erg_K / ct.kev_2_erg) * 10.0 ** fit.x[0]

        # --- Tx apertures
        # R500
        Xspec_ann = (
            Xspec[self.rad[hdx] <= self.R500 / self.R200].sum(axis=0) * self.photon_conv
        )

        T = np.log10(
            np.sum(
                (self.mass[hdx] * self.temp[hdx])[
                    self.rad[hdx] <= self.R500 / self.R200
                ]
            )
            / np.sum((self.mass[hdx])[self.rad[hdx] <= self.R500 / self.R200])
        )
        D = (
            np.log10(
                np.sum((self.mass[hdx] * ne2dV)[self.rad[hdx] <= self.R500 / self.R200])
                / np.sum((self.mass[hdx])[self.rad[hdx] <= self.R500 / self.R200])
            )
            - self.Enorm
        )
        Z = np.sum(
            (self.mass[hdx] * self.zmet[hdx])[self.rad[hdx] <= self.R500 / self.R200]
        ) / np.sum((self.mass[hdx])[self.rad[hdx] <= self.R500 / self.R200])

        try:
            limits = ([T - 0.5, -6.0, 1.0e-6], [T + 0.5, 6.0, 10.0])
            fit = least_squares(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_LS"),
                bounds=limits,
                method="trf",
            )
        except:
            limits = [(T - 0.5, T + 0.5), (-6.0, 6.0), (1.0e-6, 10.0)]
            fit = minimize(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_MN"),
                bounds=limits,
                method="TNC",
                options={"maxiter": 200},
            )
        self.Tx_500 = (ct.kB_erg_K / ct.kev_2_erg) * 10.0 ** fit.x[0]

        # R500 CE
        Xspec_ann = (
            Xspec[
                (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                & (self.rad[hdx] <= self.R500 / self.R200)
            ].sum(axis=0)
            * self.photon_conv
        )

        T = np.log10(
            np.sum(
                (self.mass[hdx] * self.temp[hdx])[
                    (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                    & (self.rad[hdx] <= self.R500 / self.R200)
                ]
            )
            / np.sum(
                (self.mass[hdx])[
                    (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                    & (self.rad[hdx] <= self.R500 / self.R200)
                ]
            )
        )
        D = (
            np.log10(
                np.sum(
                    (self.mass[hdx] * ne2dV)[
                        (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                        & (self.rad[hdx] <= self.R500 / self.R200)
                    ]
                )
                / np.sum(
                    (self.mass[hdx])[
                        (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                        & (self.rad[hdx] <= self.R500 / self.R200)
                    ]
                )
            )
            - self.Enorm
        )
        Z = np.sum(
            (self.mass[hdx] * self.zmet[hdx])[
                (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                & (self.rad[hdx] <= self.R500 / self.R200)
            ]
        ) / np.sum(
            (self.mass[hdx])[
                (self.rad[hdx] <= 0.15 * self.R500 / self.R200)
                & (self.rad[hdx] <= self.R500 / self.R200)
            ]
        )

        try:
            limits = ([T - 0.5, -6.0, 1.0e-6], [T + 0.5, 6.0, 10.0])
            fit = least_squares(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_LS"),
                bounds=limits,
                method="trf",
            )
        except:
            limits = [(T - 0.5, T + 0.5), (-6.0, 6.0), (1.0e-6, 10.0)]
            fit = minimize(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_MN"),
                bounds=limits,
                method="TNC",
                options={"maxiter": 200},
            )
        self.Tx_500ce = (ct.kB_erg_K / ct.kev_2_erg) * 10.0 ** fit.x[0]

        # R200
        Xspec_ann = Xspec[self.rad[hdx] <= 1.0].sum(axis=0) * self.photon_conv

        T = np.log10(
            np.sum((self.mass[hdx] * self.temp[hdx])[self.rad[hdx] <= 1.0])
            / np.sum((self.mass[hdx])[self.rad[hdx] <= 1.0])
        )
        D = (
            np.log10(
                np.sum((self.mass[hdx] * ne2dV)[self.rad[hdx] <= 1.0])
                / np.sum((self.mass[hdx])[self.rad[hdx] <= 1.0])
            )
            - self.Enorm
        )
        Z = np.sum((self.mass[hdx] * self.zmet[hdx])[self.rad[hdx] <= 1.0]) / np.sum(
            (self.mass[hdx])[self.rad[hdx] <= 1.0]
        )

        try:
            limits = ([T - 0.5, -6.0, 1.0e-6], [T + 0.5, 6.0, 10.0])
            fit = least_squares(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_LS"),
                bounds=limits,
                method="trf",
            )
        except:
            limits = [(T - 0.5, T + 0.5), (-6.0, 6.0), (1.0e-6, 10.0)]
            fit = minimize(
                self._spectrum_model,
                [T, D, Z],
                args=(Xspec_ann, "FIT_MN"),
                bounds=limits,
                method="TNC",
                options={"maxiter": 200},
            )
        self.Tx_200 = (ct.kB_erg_K / ct.kev_2_erg) * 10.0 ** fit.x[0]
        del Xspec, ne2dV
        return

    def compute_centre_of_mass_offset(self, mpi):
        """
        Compute the centre of mass offset inside Rvir

        Arguments:
          -mpi : An instance of the mpi class
        """

        if not mpi.Rank:
            print(" > Computing centre of mass offset", flush=True)
        # Initialise
        self.Xoff = 0.0
        gmass = 0.0
        dmass = 0.0
        smass = 0.0

        # Gas
        if "rad" in self.__dict__.keys():
            idx = np.where(self.rad * self.R200 <= self.Rvir)[0]
            self.Xoff += np.sum(self.mass[idx] * self.pos[idx].T, axis=-1)
            gmass = np.sum(self.mass[idx])

        # DM
        if "DMrad" in self.__dict__.keys():
            idx = np.where(self.DMrad * self.R200 <= self.Rvir)[0]
            self.Xoff += np.sum(self.DMmass[idx] * self.DMpos[idx].T, axis=-1)
            dmass = np.sum(self.DMmass[idx])

        # Stars
        if "STrad" in self.__dict__.keys():
            idx = np.where(self.STrad * self.R200 <= self.Rvir)[0]
            self.Xoff += np.sum(self.STmass[idx] * self.STpos[idx].T, axis=-1)
            smass = np.sum(self.STmass[idx])

        if gmass + dmass + smass == 0.0:
            del self.Xoff
        else:
            self.Xoff /= gmass + dmass + smass
            self.Xoff = np.sqrt((self.Xoff ** 2.0).sum(axis=-1)) / self.Rvir
            del gmass, dmass, smass
        return

    def compute_concentration(self, mpi, Nb=25):
        """
        Fit total matter profile to measure concentration

        Arguments:
          -mpi : An instance of the mpi class
          -Nb  : Number of bins in the radial profile
        """

        if not mpi.Rank:
            print(" > Computing concentration", flush=True)

        if "TOTALpro" not in self.__dict__.keys():
            self.compute_mass_profiles(mpi, Nb=Nb)

        # Fit NFW profile
        p0 = [6.0]
        try:
            fit = least_squares(
                self.fit_NFW,
                p0,
                args=(0, "FIT_LS"),
                bounds=([0.1, np.inf]),
                method="trf",
            )
        except:
            fit = minimize(self.fit_NFW, p0, args=(0, "FIT_MN"), method="TNC")

        self.Cvir = fit.x
        return

    def fit_NFW(self, p0, dum, mode):
        """
        Fit NFW profile

        Arguments:
          -p0   : Current parameter values
          -dum  : Dummy variable to solve argument passing issue
          -mode : STRING, if FIT then fit model to data, else return model
        """

        c = p0

        x = self.cens * self.R200 / self.Rvir

        rho_halo = (3.0 * self.Mvir) / (4.0 * np.pi * self.Rvir ** 3.0)

        A_nfw = np.log(1.0 + c) - c / (1.0 + c)

        idx = np.where((x <= 1.0) & (self.TOTALpro != 0.0))[0]

        model = rho_halo / (3.0 * A_nfw * x[idx] * (1.0 / c + x[idx]) ** 2.0)

        if mode == "MODEL":
            return model
        elif mode == "FIT_LS":
            return np.log10(model[idx]) - np.log10(self.TOTALpro[idx] / self.vols[idx])
        elif mode == "FIT_MN":
            return np.sum(
                (np.log10(model[idx]) - np.log10(self.TOTALpro[idx] / self.vols[idx]))
                ** 2.0
            )
        return

    def compute_DM_surface_pressure(self, mpi):
        """
        Compute the surface pressure of the dark matter

        Arguments:
          -mpi : An instance of the mpi class
        """

        if not mpi.Rank:
            print(" > Computing dark matter surface pressure", flush=True)
        # Hubble flow -- !!! TNG COSMOLOGY HARD WIRED !!!
        H_z = (
            np.sqrt(0.3089 * (1.0 + self.redshift) ** 3.0 + 0.6911)
            * 100.0
            * self.hub
            * ct.km_cm
        )

        # Surface pressure
        idx = np.where(
            (self.DMrad >= 0.8 * self.Rvir / self.R200)
            & (self.DMrad < self.Rvir / self.R200)
        )[0]
        vol = (4.0 / 3.0) * np.pi * (self.Rvir ** 3.0 - (0.8 * self.Rvir) ** 3.0)
        dhub = (sh.vnorm_rp(self.DMpos[idx]).T * H_z * self.DMrad[idx] / ct.Mpc_cm).T
        Dvel = self.DMvelc[idx] - self.Vbulk + dhub
        self.Pdm = (
            (1.0 / 3.0) * np.sum(self.DMmass[idx] * (Dvel ** 2.0).sum(axis=-1)) / vol
        )
        del idx, vol, dhub, Dvel
        return

    def compute_kinetic_and_potential_energies(self, mpi):
        """
        Compute the kinetic and potential energies of the halo with virial radius

        Arguments:
          -mpi : An instance of the mpi class
        """

        if not mpi.Rank:
            print(" > Computing halo kinetic and potential energies", flush=True)

        # --- Kinetic energy
        self.Tkin = 0.0
        # Hubble flow -- !!! TNG COSMOLOGY HARD WIRED !!!
        H_z = (
            np.sqrt(0.3089 * (1.0 + self.redshift) ** 3.0 + 0.6911)
            * 100.0
            * self.hub
            * ct.km_cm
        )

        # GAS
        idx = np.where(self.rad <= self.Rvir)[0]
        ghub = (sh.vnorm_rp(self.pos[idx]).T * H_z * self.rad[idx] / ct.Mpc_cm).T
        Gvel = self.velc[idx] - self.Vbulk + ghub
        self.Tkin += np.sum(0.5 * self.mass[idx] * (Gvel ** 2.0).sum(axis=-1))
        del ghub, Gvel
        # DM
        idx = np.where(self.DMrad <= self.Rvir)[0]
        dhub = (sh.vnorm_rp(self.DMpos[idx]).T * H_z * self.DMrad[idx] / ct.Mpc_cm).T
        Dvel = self.DMvelc[idx] - self.Vbulk + dhub
        self.Tkin += np.sum(0.5 * self.DMmass[idx] * (Dvel ** 2.0).sum(axis=-1))
        del dhub, Dvel
        # STARS
        idx = np.where(self.STrad <= self.Rvir)[0]
        shub = (sh.vnorm_rp(self.STpos[idx]).T * H_z * self.STrad[idx] / ct.Mpc_cm).T
        Svel = self.STvelc[idx] - self.Vbulk + shub
        self.Tkin += np.sum(0.5 * self.STmass[idx] * (Svel ** 2.0).sum(axis=-1))
        del shub, Svel

        # --- Potential energy
        self.Upot = 0.0
        # GAS
        idx = np.where(self.rad <= self.Rvir)[0]
        self.Upot += np.sum(self.mass[idx] * self.gpot[idx])
        # DM
        idx = np.where(self.DMrad <= self.Rvir)[0]
        self.Upot += np.sum(self.DMmass[idx] * self.DMgpot[idx])
        # STAR
        idx = np.where(self.STrad <= self.Rvir)[0]
        self.Upot += np.sum(self.STmass[idx] * self.STgpot[idx])
        del idx
        return

    def set_up_radial_profile(self, rmin=0.05, rmax=5.0, logb=True):
        """
        Set up radial bins for profiles

        Arguments:
          -rmin : Minimum extent of the radial profile
          -rmax : Maximum extent of the radial profile
          -logb : BOOLEAN, if TRUE radial bins are logarithmically spaced
        """

        if logb:
            self.bins = np.logspace(np.log10(rmin), np.log10(rmax), num=self.Nbins + 1)
            self.cens = 10.0 ** (
                0.5 * (np.log10(self.bins[1:]) + np.log10(self.bins[:-1]))
            )
        else:
            self.bins = np.linspace(rmin, rmax, num=self.Nbins + 1)
            self.cens = 0.5 * (self.bins[1:] + self.bins[:-1])

        self.bins[0] = 0.0
        return

    def save(self, mpi):
        """
        Save all computed quantities for a given halo

        Arguments:
          -mpi : An instance of the mpi class
        """

        keys = sorted(self.__dict__.keys())

        if not mpi.Rank:
            print(" > Saving halo", flush=True)
        ftag = "output/{0}_Snap{1:03d}_z{2:d}p{3:02d}_{4:03d}.hdf5".format(
            self.fname,
            self.snap,
            int(self.redshift),
            int(100.0 * (self.redshift - int(self.redshift))),
            mpi.Rank,
        )

        f = h5py.File(ftag, "a")
        if self.tag in list(f.keys()):
            fk = sorted(list(f[self.tag].keys()))
        else:
            fk = []

        if self.tag not in f.keys():
            # --- Aperture quantities
            grp = f.create_group(self.tag)
            # Halo properties
            if "M500" in keys:
                grp.attrs["M500_Msun"] = self.M500 / ct.Msun_g
            if "R500" in keys:
                grp.attrs["R500_Mpc"] = self.R500 / ct.Mpc_cm
            if "M200" in keys:
                grp.attrs["M200_Msun"] = self.M200 / ct.Msun_g
            if "R200" in keys:
                grp.attrs["R200_Mpc"] = self.R200 / ct.Mpc_cm
            if "Rvir" in keys:
                grp.attrs["Rvir_Mpc"] = self.Rvir / ct.Mpc_cm
            if "redshift" in keys:
                grp.attrs["Redshift"] = self.redshift
            # Morphology statistics
            if "Cvir" in keys:
                grp.attrs["Cvir"] = self.Cvir
            if "Xoff" in keys:
                grp.attrs["Xoff"] = self.Xoff
            if "Tkin" in keys:
                grp.attrs["Tkin_erg"] = self.Tkin
            if "Upot" in keys:
                grp.attrs["Upot_erg"] = self.Upot
            if "Pdm" in keys:
                grp.attrs["Pdm_erg_cm^-3"] = self.Pdm
            # Observable properties
            if "Lx_500" in keys:
                grp.attrs["Lx500_erg_s^-1"] = self.Lx_500
            if "Lx_500ce" in keys:
                grp.attrs["Lx500ce_erg_s^-1"] = self.Lx_500ce
            if "Lx_200" in keys:
                grp.attrs["Lx200_erg_s^-1"] = self.Lx_200
            if "Tx_500" in keys:
                grp.attrs["Tx500_keV"] = self.Tx_500
            if "Tx_500ce" in keys:
                grp.attrs["Tx500ce_keV"] = self.Tx_500ce
            if "Tx_200" in keys:
                grp.attrs["Tx200_keV"] = self.Tx_200
            if "Ysz_500" in keys:
                grp.attrs["Ysz500_Mpc^2"] = self.Ysz_500
            if "Ysz_200" in keys:
                grp.attrs["Ysz200_Mpc^2"] = self.Ysz_200
            if "Ysz_5r500" in keys:
                grp.attrs["Ysz5r500_Mpc^2"] = self.Ysz_5r500
            # Cumulative shape measures
            if "q_dm_200" in keys:
                grp.attrs["Qdm_200"] = self.q_dm_200
            if "q_dm_500" in keys:
                grp.attrs["Qdm_500"] = self.q_dm_500
            if "q_dm_vir" in keys:
                grp.attrs["Qdm_vir"] = self.q_dm_vir
            if "q_gas_200" in keys:
                grp.attrs["Qgas_200"] = self.q_gas_200
            if "q_gas_500" in keys:
                grp.attrs["Qgas_500"] = self.q_gas_500
            if "q_gas_vir" in keys:
                grp.attrs["Qgas_vir"] = self.q_gas_vir
            if "q_st_200" in keys:
                grp.attrs["Qstar_200"] = self.q_st_200
            if "q_st_500" in keys:
                grp.attrs["Qstar_500"] = self.q_st_500
            if "q_st_vir" in keys:
                grp.attrs["Qstar_vir"] = self.q_st_vir
            if "s_dm_200" in keys:
                grp.attrs["Sdm_200"] = self.s_dm_200
            if "s_dm_500" in keys:
                grp.attrs["Sdm_500"] = self.s_dm_500
            if "s_dm_vir" in keys:
                grp.attrs["Sdm_vir"] = self.s_dm_vir
            if "s_gas_200" in keys:
                grp.attrs["Sgas_200"] = self.s_gas_200
            if "s_gas_500" in keys:
                grp.attrs["Sgas_500"] = self.s_gas_500
            if "s_gas_vir" in keys:
                grp.attrs["Sgas_vir"] = self.s_gas_vir
            if "s_st_200" in keys:
                grp.attrs["Sstar_200"] = self.s_st_200
            if "s_st_500" in keys:
                grp.attrs["Sstar_500"] = self.s_st_500
            if "s_st_vir" in keys:
                grp.attrs["Sstar_vir"] = self.s_st_vir

        # --- Profiles
        # Basics
        if "cens" in keys:
            if "Radii_Mpc" in fk:
                del f["{0}/Radii_Mpc".format(self.tag)]
            f.create_dataset(
                "{0}/Radii_Mpc".format(self.tag), data=self.cens * self.R200 / ct.Mpc_cm
            )
        if "vols" in keys:
            if "Volumes_Mpc^3" in fk:
                del f["{0}/Volumes_Mpc^3".format(self.tag)]
            f.create_dataset(
                "{0}/Volumes_Mpc^3".format(self.tag), data=self.vols / ct.Mpc_cm ** 3.0
            )
        # Mass profiles
        if "GASpro" in keys:
            if "Mgas_Msun" in fk:
                del f["{0}/Mgas_Msun".format(self.tag)]
            f.create_dataset(
                "{0}/Mgas_Msun".format(self.tag), data=self.GASpro / ct.Msun_g
            )
        if "GHOTpro" in keys:
            if "Mgas_hot_Msun" in fk:
                del f["{0}/Mgas_hot_Msun".format(self.tag)]
            f.create_dataset(
                "{0}/Mgas_hot_Msun".format(self.tag), data=self.GHOTpro / ct.Msun_g
            )
        if "DMpro" in keys:
            if "Mdm_Msun" in fk:
                del f["{0}/Mdm_Msun".format(self.tag)]
            f.create_dataset(
                "{0}/Mdm_Msun".format(self.tag), data=self.DMpro / ct.Msun_g
            )
        if "STpro" in keys:
            if "Mstars_Msun" in fk:
                del f["{0}/Mstars_Msun".format(self.tag)]
            f.create_dataset(
                "{0}/Mstars_Msun".format(self.tag), data=self.STpro / ct.Msun_g
            )
        # Thermodynamic profiles
        if "TEMPpro" in keys:
            if "Tmw_keV" in fk:
                del f["{0}/Tmw_keV".format(self.tag)]
            f.create_dataset(
                "{0}/Tmw_keV".format(self.tag),
                data=(ct.kB_erg_K / ct.kev_2_erg) * self.TEMPpro,
            )
        if "Tsl" in keys:
            if "Tsl_keV" in fk:
                del f["{0}/Tsl_keV".format(self.tag)]
            f.create_dataset("{0}/Tsl_keV".format(self.tag), data=self.Tsl)
        if "rho_sq" in keys:
            if "RHO_sq_Msun^2_Mpc^-6" in fk:
                del f["{0}/RHO_sq_Msun^2_Mpc^-6".format(self.tag)]
            f.create_dataset(
                "{0}/RHO_sq_Msun^2_Mpc^-6".format(self.tag), data=self.rho_sq
            )
        if "rho_sq_hot" in keys:
            if "RHO_sq_hot_Msun^2_Mpc^-6" in fk:
                del f["{0}/RHO_sq_hot_Msun^2_Mpc^-6".format(self.tag)]
            f.create_dataset(
                "{0}/RHO_sq_hot_Msun^2-Mpc^-6".format(self.tag), data=self.rho_sq_hot
            )
        if "pres_sq" in keys:
            if "P_sq_erg^2_cm^-6" in fk:
                del f["{0}/P_sq_erg^2_cm^-6".format(self.tag)]
            f.create_dataset("{0}/P_sq_erg^2_cm^-6".format(self.tag), data=self.pres_sq)
        if "pres_sq_hot" in keys:
            if "P_sq_hot_erg^2_cm^-6" in fk:
                del f["{0}/P_sq_hot_erg^2_cm^-6".format(self.tag)]
            f.create_dataset(
                "{0}/P_sq_hot_erg^2_cm^-6".format(self.tag), data=self.pres_sq_hot
            )
        if "Pkin" in keys:
            if "Pnth_erg_cm^-3" in fk:
                del f["{0}/Pnth_erg_cm^-3".format(self.tag)]
            f.create_dataset("{0}/Pnth_erg_cm^-3".format(self.tag), data=self.Pkin)
        # Emission measure profile
        if "emm" in keys:
            if "EMM_cm^-6" in fk:
                del f["{0}/EMM_cm^-6".format(self.tag)]
            f.create_dataset("{0}/EMM_cm^-6".format(self.tag), data=self.emm)
        # Observable properties
        if "Lx_pro" in keys:
            if "Lx_pro_erg_s^-1" in fk:
                del f["{0}/Lx_pro_erg_s^-1".format(self.tag)]
            f.create_dataset("{0}/Lx_pro_erg_s^-1".format(self.tag), data=self.Lx_pro)
        if "Tx_pro" in keys:
            if "Tx_pro_keV" in fk:
                del f["{0}/Tx_pro_keV".format(self.tag)]
            f.create_dataset("{0}/Tx_pro_keV".format(self.tag), data=self.Tx_pro)
        if "Ysz_pro" in keys:
            if "Ysz_pro_Mpc^2" in fk:
                del f["{0}/Ysz_pro_Mpc^2".format(self.tag)]
            f.create_dataset("{0}/Ysz_pro_Mpc^2".format(self.tag), data=self.Ysz_pro)
        # Metallicity profiles
        if "Zgas" in keys:
            if "Zmet_gas_solar" in fk:
                del f["{0}/Zmet_gas_solar".format(self.tag)]
            f.create_dataset("{0}/Zmet_gas_solar".format(self.tag), data=self.Zgas)
        if "Zstar" in keys:
            if "Zmet_star_solar" in fk:
                del f["{0}/Zmet_star_solar".format(self.tag)]
            f.create_dataset("{0}/Zmet_star_solar".format(self.tag), data=self.Zstar)
        # Velocity profiles
        if "vr_gas" in keys:
            if "Vr_gas_km_s^-1" in fk:
                del f["{0}/Vr_gas_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vr_gas_km_s^-1".format(self.tag), data=self.vr_gas)
        if "vt_gas" in keys:
            if "Vt_gas_km_s^-1" in fk:
                del f["{0}/Vt_gas_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vt_gas_km_s^-1".format(self.tag), data=self.vt_gas)
        if "vp_gas" in keys:
            if "Vp_gas_km_s^-1" in fk:
                del f["{0}/Vp_gas_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vp_gas_km_s^-1".format(self.tag), data=self.vp_gas)
        if "vr_dm" in keys:
            if "Vr_dm_km_s^-1" in fk:
                del f["{0}/Vr_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vr_dm_km_s^-1".format(self.tag), data=self.vr_dm)
        if "vt_dm" in keys:
            if "Vt_dm_km_s^-1" in fk:
                del f["{0}/Vt_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vt_dm_km_s^-1".format(self.tag), data=self.vt_dm)
        if "vp_dm" in keys:
            if "Vp_dm_km_s^-1" in fk:
                del f["{0}/Vp_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vp_dm_km_s^-1".format(self.tag), data=self.vp_dm)
        if "vr_star" in keys:
            if "Vr_star_km_s^-1" in fk:
                del f["{0}/Vr_star_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vr_star_km_s^-1".format(self.tag), data=self.vr_star)
        if "vt_star" in keys:
            if "Vt_star_km_s^-1" in fk:
                del f["{0}/Vt_star_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vt_star_km_s^-1".format(self.tag), data=self.vt_star)
        if "vp_star" in keys:
            if "Vp_star_km_s^-1" in fk:
                del f["{0}/Vp_star_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/Vp_star_km_s^-1".format(self.tag), data=self.vp_star)
        # Dispersion profiles
        if "sigr_gas" in keys:
            if "SIGr_gas_km_s^-1" in fk:
                del f["{0}/SIGr_gas_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGr_gas_km_s^-1".format(self.tag), data=self.sigr_gas
            )
        if "sigt_gas" in keys:
            if "SIGt_gas_km_s^-1" in fk:
                del f["{0}/SIGt_gas_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGt_gas_km_s^-1".format(self.tag), data=self.sigt_gas
            )
        if "sigp_gas" in keys:
            if "SIGp_gas_km_s^-1" in fk:
                del f["{0}/SIGp_gas_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGp_gas_km_s^-1".format(self.tag), data=self.sigp_gas
            )
        if "sigr_dm" in keys:
            if "SIGr_dm_km_s^-1" in fk:
                del f["{0}/SIGr_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/SIGr_dm_km_s^-1".format(self.tag), data=self.sigr_dm)
        if "sigt_dm" in keys:
            if "SIGt_dm_km_s^-1" in fk:
                del f["{0}/SIGt_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/SIGt_dm_km_s^-1".format(self.tag), data=self.sigt_dm)
        if "sigp_dm" in keys:
            if "SIGp_dm_km_s^-1" in fk:
                del f["{0}/SIGp_dm_km_s^-1".format(self.tag)]
            f.create_dataset("{0}/SIGp_dm_km_s^-1".format(self.tag), data=self.sigp_dm)
        if "sigr_star" in keys:
            if "SIGr_star_km_s^-1" in fk:
                del f["{0}/SIGr_star_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGr_star_km_s^-1".format(self.tag), data=self.sigr_star
            )
        if "sigt_star" in keys:
            if "SIGt_star_km_s^-1" in fk:
                del f["{0}/SIGt_star_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGt_star_km_s^-1".format(self.tag), data=self.sigt_star
            )
        if "sigp_star" in keys:
            if "SIGp_star_km_s^-1" in fk:
                del f["{0}/SIGp_star_km_s^-1".format(self.tag)]
            f.create_dataset(
                "{0}/SIGp_star_km_s^-1".format(self.tag), data=self.sigp_star
            )
        # Shape profiles
        if "q_gas" in keys:
            if "Qgas" in fk:
                del f["{0}/Qgas".format(self.tag)]
            f.create_dataset("{0}/Qgas".format(self.tag), data=self.q_gas)
        if "s_gas" in keys:
            if "Sgas" in fk:
                del f["{0}/Sgas".format(self.tag)]
            f.create_dataset("{0}/Sgas".format(self.tag), data=self.s_gas)
        if "q_ghot" in keys:
            if "Qgas_hot" in fk:
                del f["{0}/Qgas_hot".format(self.tag)]
            f.create_dataset("{0}/Qgas_hot".format(self.tag), data=self.q_ghot)
        if "s_ghot" in keys:
            if "Sgas_hot" in fk:
                del f["{0}/Sgas_hot".format(self.tag)]
            f.create_dataset("{0}/Sgas_hot".format(self.tag), data=self.s_ghot)
        if "q_temp" in keys:
            if "Qtemp" in fk:
                del f["{0}/Qtemp".format(self.tag)]
            f.create_dataset("{0}/Qtemp".format(self.tag), data=self.q_temp)
        if "s_temp" in keys:
            if "Stemp" in fk:
                del f["{0}/Stemp".format(self.tag)]
            f.create_dataset("{0}/Stemp".format(self.tag), data=self.s_temp)
        if "q_pres" in keys:
            if "Qpres" in fk:
                del f["{0}/Qpres".format(self.tag)]
            f.create_dataset("{0}/Qpres".format(self.tag), data=self.q_pres)
        if "s_pres" in keys:
            if "Spres" in fk:
                del f["{0}/Spres".format(self.tag)]
            f.create_dataset("{0}/Spres".format(self.tag), data=self.s_pres)
        if "q_dm" in keys:
            if "Qdm" in fk:
                del f["{0}/Qdm".format(self.tag)]
            f.create_dataset("{0}/Qdm".format(self.tag), data=self.q_dm)
        if "s_dm" in keys:
            if "Sdm" in fk:
                del f["{0}/Sdm".format(self.tag)]
            f.create_dataset("{0}/Sdm".format(self.tag), data=self.s_dm)
        if "q_star" in keys:
            if "Qstar" in fk:
                del f["{0}/Qstar".format(self.tag)]
            f.create_dataset("{0}/Qstar".format(self.tag), data=self.q_star)
        if "s_star" in keys:
            if "Sstar" in fk:
                del f["{0}/Sstar".format(self.tag)]
            f.create_dataset("{0}/Sstar".format(self.tag), data=self.s_star)
        # Eigenvector profiles
        if "Iv_gas" in keys:
            if "Veigen_gas_pro" in fk:
                del f["{0}/Veigen_gas_pro".format(self.tag)]
            f.create_dataset("{0}/Veigen_gas_pro".format(self.tag), data=self.Iv_gas)
        if "Iv_ghot" in keys:
            if "Veigen_gas_hot_pro" in fk:
                del f["{0}/Veigen_gas_hot_pro".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_gas_hot_pro".format(self.tag), data=self.Iv_ghot
            )
        if "Iv_temp" in keys:
            if "Veigen_temp_pro" in fk:
                del f["{0}/Veigen_temp_pro".format(self.tag)]
            f.create_dataset("{0}/Veigen_temp_pro".format(self.tag), data=self.Iv_temp)
        if "Iv_pres" in keys:
            if "Veigen_pres_pro" in fk:
                del f["{0}/Veigen_pres_pro".format(self.tag)]
            f.create_dataset("{0}/Veigen_pres_pro".format(self.tag), data=self.Iv_pres)
        if "Iv_dm" in keys:
            if "Veigen_dm_pro" in fk:
                del f["{0}/Veigen_dm_pro".format(self.tag)]
            f.create_dataset("{0}/Veigen_dm_pro".format(self.tag), data=self.Iv_dm)
        if "Iv_star" in keys:
            if "Veigen_star_pro" in fk:
                del f["{0}/Veigen_star_pro".format(self.tag)]
            f.create_dataset("{0}/Veigen_star_pro".format(self.tag), data=self.Iv_star)
        # Cumulative eigenvectors
        if "Iv_gas_200" in keys:
            if "Veigen_gas_200" in fk:
                del f["{0}/Veigen_gas_200".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_gas_200".format(self.tag), data=self.Iv_gas_200
            )
        if "Iv_gas_500" in keys:
            if "Veigen_gas_500" in fk:
                del f["{0}/Veigen_gas_500".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_gas_500".format(self.tag), data=self.Iv_gas_500
            )
        if "Iv_gas_vir" in keys:
            if "Veigen_gas_vir" in fk:
                del f["{0}/Veigen_gas_vir".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_gas_vir".format(self.tag), data=self.Iv_gas_vir
            )
        if "Iv_dm_200" in keys:
            if "Veigen_dm_200" in fk:
                del f["{0}/Veigen_dm_200".format(self.tag)]
            f.create_dataset("{0}/Veigen_dm_200".format(self.tag), data=self.Iv_dm_200)
        if "Iv_dm_500" in keys:
            if "Veigen_dm_500" in fk:
                del f["{0}/Veigen_dm_500".format(self.tag)]
            f.create_dataset("{0}/Veigen_dm_500".format(self.tag), data=self.Iv_dm_500)
        if "Iv_dm_vir" in keys:
            if "Veigen_dm_vir" in fk:
                del f["{0}/Veigen_dm_vir".format(self.tag)]
            f.create_dataset("{0}/Veigen_dm_vir".format(self.tag), data=self.Iv_dm_vir)
        if "Iv_st_200" in keys:
            if "Veigen_star_200" in fk:
                del f["{0}/Veigen_star_200".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_star_200".format(self.tag), data=self.Iv_st_200
            )
        if "Iv_st_500" in keys:
            if "Veigen_star_500" in fk:
                del f["{0}/Veigen_star_500".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_star_500".format(self.tag), data=self.Iv_st_500
            )
        if "Iv_st_vir" in keys:
            if "Veigen_star_vir" in fk:
                del f["{0}/Veigen_star_vir".format(self.tag)]
            f.create_dataset(
                "{0}/Veigen_star_vir".format(self.tag), data=self.Iv_st_vir
            )
        f.close()
        return

    def _read_APEC_table(self, APECpath="src/APECtable.hdf5"):
        """
        Read a precomputed APEC table

        Arguments:
          -APECpath : Path to precomputed APEC table
        """

        f = h5py.File(APECpath, "r")
        self.APEC_temperatures = f["Log_Plasma_Temp"][:]
        self.APEC_energies = f["Energies"][:]
        self.APEC_H = f["Hydrogen"][:]
        self.APEC_He = f["Helium"][:]
        self.APEC_C = f["Carbon"][:]
        self.APEC_N = f["Nitrogen"][:]
        self.APEC_O = f["Oxygen"][:]
        self.APEC_Ne = f["Neon"][:]
        self.APEC_Mg = f["Magnesium"][:]
        self.APEC_Si = f["Silicon"][:]
        self.APEC_S = f["Sulphur"][:]
        self.APEC_Ca = f["Calcium"][:]
        self.APEC_Fe = f["Iron"][:]
        if "Effective_Area" in f.keys():
            self.APEC_area = f["Effective_Area"][:]
        else:
            self.APEC_area = None
        f.close()
        return

    def _spectrum_model(self, p0, spectrum, MODE):
        """
        Calculate model spectrum given guess p0 and compare to
        SPECTRUM. Mode determines whether to return a value of
        difference between model and data or a model spectrum.

        Arguments:
          -p0       : Model parameter guesses [LIST]
          -spectrum : Spectrum to be fit [ARRAY]
          -MODE     : If FIT return (model-spectrum), else
                      if MODEL returns the spectrum model [STRING]

        Returns:
          -Returns either model-spectrum difference or model [ARRAY/FLOAT/ARRAY]
        """

        T, D, Z = p0

        # Locate temperature in APEC lookup table
        itemp = _locate(self.APEC_temperatures, T)
        if T < self.APEC_temperatures[itemp]:
            itemp -= 1
        if itemp < 0:
            itemp = 0
        elif itemp > len(self.APEC_temperatures) - 2:
            itemp = len(self.APEC_temperatures) - 2
        dlogT = np.median(self.APEC_temperatures[1:] - self.APEC_temperatures[:-1])

        # Contribution of each element to the model
        H = self._model_element_contribution(itemp, T, dlogT, "Hydrogen")
        He = self._model_element_contribution(itemp, T, dlogT, "Helium")
        C = self._model_element_contribution(itemp, T, dlogT, "Carbon")
        N = self._model_element_contribution(itemp, T, dlogT, "Nitrogen")
        O = self._model_element_contribution(itemp, T, dlogT, "Oxygen")
        Ne = self._model_element_contribution(itemp, T, dlogT, "Neon")
        Mg = self._model_element_contribution(itemp, T, dlogT, "Magnesium")
        Si = self._model_element_contribution(itemp, T, dlogT, "Silicon")
        S = self._model_element_contribution(itemp, T, dlogT, "Sulphur")
        Ca = self._model_element_contribution(itemp, T, dlogT, "Calcium")
        Fe = self._model_element_contribution(itemp, T, dlogT, "Iron")

        # Calculate final model
        model = 10.0 ** (self.Enorm + D) * (
            H + He + Z * (C + N + O + Ne + Mg + Si + S + Ca + Fe)
        )
        model *= self.photon_conv

        # Return - only fit model where we have at least 1 photon
        edx = np.where(spectrum >= 1)[0]
        if MODE == "FIT_LS":
            return model[edx] - spectrum[edx]
        elif MODE == "FIT_MN":
            return np.sum((model[edx] - spectrum[edx]) ** 2.0)
        else:
            return model
        return

    def _model_element_contribution(self, itemp, T, dlogT, element):
        """
        Calculate the element contribution to a spectrum model

        Arguments:
          -itemp   : temperature index in APEC lookup table
          -T       : Log gas temperature
          -dlogT   : APEC lookup temperature spacing
          -element : Chemical element of interest

        Returns:
          -X-ray spectrum for given element at given temperature
        """

        if element == "Hydrogen":
            tmp = self.APEC_H
        elif element == "Helium":
            tmp = self.APEC_He
        elif element == "Carbon":
            tmp = self.APEC_C
        elif element == "Nitrogen":
            tmp = self.APEC_N
        elif element == "Oxygen":
            tmp = self.APEC_O
        elif element == "Neon":
            tmp = self.APEC_Ne
        elif element == "Magnesium":
            tmp = self.APEC_Mg
        elif element == "Silicon":
            tmp = self.APEC_Si
        elif element == "Sulphur":
            tmp = self.APEC_S
        elif element == "Calcium":
            tmp = self.APEC_Ca
        elif element == "Iron":
            tmp = self.APEC_Fe

        m = (np.log10(tmp[itemp + 1]) - np.log10(tmp[itemp])) / dlogT
        b = np.log10(tmp[itemp]) - m * self.APEC_temperatures[itemp]
        del tmp
        return 10.0 ** (m * T + b)

    def _calculate_photon_conversion_factor(self, Tint=1.0e6):
        """
        Calculate the factor required to convert erg/s to photons, via
        an assumed integration time.

        Arguments:
          -xcube : An instance of the datacube read class
          -Tint  : Assumed integration time in Megaseconds
        """

        # Obtain effective area if required
        if "APEC_area" not in self.__dict__.keys():
            Efficency, Eff_Area = np.loadtxt("src/chandra_acis-i_.area", unpack=True)
            edx = self.locate(Efficency, self.APEC_energies)
            self.APEC_area = Eff_Area[edx]
            del edx, Efficency, Eff_Area

        # Get absorption from lookup table
        Absorption = self._wabs(self.APEC_energies, 2.0e20)

        # Luminosity distance
        cosmo = FlatLambdaCDM(H0=self.hubp * 100.0, Om0=self.OmegaM)
        if self.redshift <= 0.005:
            Dlum = cosmo.luminosity_distance(0.005).value * ct.Mpc_cm
        else:
            Dlum = cosmo.luminosity_distance(self.redshift).value * ct.Mpc_cm

        self.photon_conv = (
            Absorption
            * self.APEC_area
            * Tint
            / (4.0 * np.pi * Dlum ** 2.0)
            / ct.kev_2_erg
        )
        self.photon_conv /= self.APEC_energies
        del Absorption, cosmo, Dlum
        return

    def _wabs(self, E, col):
        """
        Calculate absorption for energy bins via table interpolation

        Arguments:
          -E   : APEC spectrum energies
          -col : Assumed column density

        Returns:
          -The expected absorbtion array
        """

        EX = np.array(
            [
                0.100,
                0.284,
                0.400,
                0.532,
                0.707,
                0.867,
                1.303,
                1.840,
                2.471,
                3.210,
                4.038,
                7.111,
                8.331,
                10.000,
            ]
        )
        C0 = np.array(
            [
                17.30,
                34.60,
                78.10,
                71.40,
                95.50,
                308.9,
                120.6,
                141.3,
                202.7,
                342.7,
                352.2,
                433.9,
                629.0,
                701.2,
            ]
        )
        C1 = np.array(
            [
                608.1,
                267.9,
                18.80,
                66.80,
                145.8,
                -380.6,
                169.3,
                146.8,
                104.7,
                18.70,
                18.70,
                -2.40,
                30.90,
                25.20,
            ]
        )
        C2 = np.array(
            [
                -2150.0,
                -476.1,
                4.3,
                -51.4,
                -61.1,
                294.0,
                -47.7,
                -31.5,
                -17.0,
                0.000,
                0.000,
                0.750,
                0.000,
                0.000,
            ]
        )
        ii = _locate(EX, E)
        E3 = E * E * E
        AXS = C0[ii] + C1[ii] * E + C2[ii] * E * E
        BGDIF = AXS / E3
        BGDIF = -BGDIF * 1.0e-4
        arg = col / 1.0e20 * BGDIF
        return np.exp(arg)


def _unique_positions_test(x):
    """
    Check if postions are unique

    Arguments:
      -X : 2D ARRAY of particle positions

    Returns:
      -Those indices whose positions are not unique
    """

    idx = np.arange(len(x))
    pdx = np.unique(x, return_index=True)
    return np.setdiff1d(idx, pdx)


def _check_points_unique(p):
    """
    Check if postions are unique, nudge and repeat if required

    Arguments:
      -p : 2D ARRAY of particle positions

    Returns:
      -p : Positional 2D ARRAY where all postions are unique
    """

    x_non_unique = _unique_positions_test(p[:, 0])
    y_non_unique = _unique_positions_test(p[:, 1])
    z_non_unique = _unique_positions_test(p[:, 2])

    if len(x_non_unique) > 0 and len(y_non_unique) > 0 and len(z_non_unique) > 0:
        idx = np.intersect1d(x_non_unique, y_non_unique)
        idx = np.intersect1d(idx, z_non_unique)
        if len(idx) > 0:
            for i in idx:
                p[i] += np.random.rand(3) * 1.0e-6 * ct.Mpc_cm
            _check_points_unique(p)
    return p


def _locate(A, X):
    """
    Returns the locations of values from array X in array A

    Arguments:
      -A : ARRAY in which values will be found
      -X : ARRAY whose values should be located

    Returns:
      -The locations of X values in array A
    """

    fdx = A.searchsorted(X)
    fdx = np.clip(fdx, 1, len(A) - 1)
    lt = A[fdx - 1]
    rt = A[fdx]
    fdx -= X - lt < rt - X
    return fdx
