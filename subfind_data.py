import os
import h5py
import numpy as np
import constants as ct

""" 
Class to rebuild the tables stored in the subfind files. Utilises
MPI4Py to split the read work.

Arguments:
  -mpi  : Instance of the MPI environment class
  -sim  : Path to the simulation of interest [STRING]
  -snap : Snapshot of interest [INTEGER]
"""


class build_table:
    def __init__(
        self, mpi, sim="/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output", snap=99
    ):

        if mpi.Rank == 0:
            print(" > Building subfind tables...", flush=True)

        self.path = "{0}/groups_{1:03d}".format(sim, snap)

        if not os.path.exists:
            self.error(mpi, "PATH")

        self.find_files()

        self.read_basics(mpi)

        self.read_tables(mpi)
        return

    def find_files(self):
        """
        Finds all files for a given path
        """

        # Find
        subfind_files = []
        for x in os.listdir(self.path):
            if x.startswith("fof_subhalo_tab_"):
                subfind_files.append("{0}/{1}".format(self.path, x))

        # Sort
        if len(subfind_files) > 1:
            sort_order = np.argsort(
                np.array([x.split(".", 2)[1] for x in subfind_files], dtype=np.int)
            )
            subfind_files = list(np.array(subfind_files)[sort_order])
            del sort_order

        self.files = subfind_files
        del subfind_files
        return

    def read_basics(self, mpi):
        """
        Read basic info. from the header, calculate other basic props.

        Arguments:
          -mpi : Instance of the MPI environment class
        """

        f = h5py.File(self.files[mpi.Rank % len(self.files)], "r")
        self.hub = f["Header"].attrs["HubbleParam"]
        self.axp = f["Header"].attrs["Time"]
        self.redshift = f["Header"].attrs["Redshift"]
        self.omega_m = f["Header"].attrs["Omega0"]
        self.omega_L = f["Header"].attrs["OmegaLambda"]
        f.close()

        self.rho_crit = (
            1.878e-29
            * self.hub
            * self.hub
            * (self.omega_m * (1.0 + self.redshift) ** 3.0 + self.omega_L)
        )
        return

    def read_tables(self, mpi):
        """
        Reads quantities of interest

        Arguments:
          -mpi : Instance of the MPI environment class
        """

        self.qoi = [
            "Group/GroupPos",
            "Group/Group_M_Crit200",
            "Group/Group_M_Crit500",
            "Group/Group_M_TopHat200",
            "Group/Group_R_Crit200",
            "Group/Group_R_Crit500",
            "Group/Group_R_TopHat200",
            "Group/GroupLenType",
            "Group/GroupFirstSub",
            "Group/GroupNsubs",
            "Group/GroupVel",
            "Subhalo/SubhaloLenType",
        ]

        self.split_files(mpi)

        my_groups = 0
        my_subgroups = 0
        for j in range(self.start, self.finish, 1):
            f = h5py.File(self.files[j], "r")
            my_groups += f["Header"].attrs["Ngroups_ThisFile"]
            my_subgroups += f["Header"].attrs["Nsubgroups_ThisFile"]
            f.close()

        for x in self.qoi:
            self.read_quantity(mpi, my_groups, my_subgroups, x)
        return

    def split_files(self, mpi):
        """
        Split files over tasks available

        Arguments:
          -mpi : Instance of the MPI environment class
        """

        num_files = int(len(self.files) / mpi.NProcs)
        remainder = len(self.files) % mpi.NProcs
        self.start = mpi.Rank * num_files
        self.finish = (mpi.Rank + 1) * num_files
        if mpi.Rank < remainder:
            self.start += mpi.Rank
            self.finish += mpi.Rank + 1
        else:
            self.start += remainder
            self.finish += remainder
        return

    def read_quantity(self, mpi, groups, subgroups, x):
        """
        Read a given quantity from the Subfind file and stores it

        Arguments:
          -mpi      : Instance of the MPI environment class
          -groups   : Number of Friends-of-Friends groups [INTEGER]
          -subgroup : Number of SUBFIND subgroups [INTEGER]
          -x        : Subfind table quantity of interest [STRING]
        """

        if mpi.Rank == 0:
            print("  -{0}".format(x), flush=True)

        if x.split("/")[0] == "Group":
            ng = groups
            tag = "Ngroups_ThisFile"
        elif x.split("/")[0] == "Subhalo":
            ng = subgroups
            tag = "Nsubgroups_ThisFile"

        f = h5py.File(self.files[0], "r")
        tab_type = f[x].dtype
        if tab_type == "float32":
            tab_type = "float64"
        tab_shape = f[x].shape
        f.close()

        if len(tab_shape) > 1:
            ng = (ng, tab_shape[1])
            table = np.zeros(ng, dtype=tab_type)
        else:
            table = np.zeros(ng, dtype=tab_type)

        off = 0
        for j in range(self.start, self.finish, 1):
            f = h5py.File(self.files[j], "r")
            grps = f["Header"].attrs[tag]
            if grps <= 0:
                f.close()
                continue
            table[off : off + grps] = f[x][:]
            f.close()
            off += grps
            del grps
        del off

        if mpi.NProcs > 1:
            table = mpi.gather(table)

        if x == "Group/GroupPos":
            self.CoP = table
        elif x == "Group/Group_M_Crit200":
            self.M200 = table
        elif x == "Group/Group_M_Crit500":
            self.M500 = table
        elif x == "Group/Group_M_TopHat200":
            self.Mvir = table
        elif x == "Group/Group_R_Crit200":
            self.R200 = table
        elif x == "Group/Group_R_Crit500":
            self.R500 = table
        elif x == "Group/Group_R_TopHat200":
            self.Rvir = table
        elif x == "Group/GroupLenType":
            self.GrLenType = table
        elif x == "Group/GroupFirstSub":
            self.FirstSub = table
        elif x == "Group/GroupNsubs":
            self.Nsubs = table
        elif x == "Group/GroupVel":
            self.Vbulk = table
        elif x == "Subhalo/SubhaloLenType":
            self.SubLenType = table
        else:
            self.error(mpi, "TABLE_STORE")
        del table
        return

    def select_halos(self, mpi, cut=1.0e14):
        """
        Pick out halos that are interesting, work out start/finish

        Arguments:
          -mpi : Instance of the MPI environment class
          -cut : Variable determining haloes to select - [STRING, LIST or FLOAT]
        """

        if cut == "MAX":
            idx = np.where(self.M200 == np.max(self.M200))[0]
        elif isinstance(cut, list):
            idx = np.array(cut)
        else:
            idx = np.where(self.M200 * ct.Mtng_Msun / self.hub >= cut)[0]

        tmp = np.arange(len(self.M200), dtype=np.int)
        tags = []
        for x in idx:
            tags.append("halo_{0:06d}".format(tmp[x]))
        self.tags = np.array(tags)
        del tmp, tags

        for x in self.qoi:
            if x == "Group/GroupPos":
                self.CoP = self.CoP[idx] * ct.kpc_cm * self.axp / self.hub
            elif x == "Group/Group_M_Crit200":
                self.M200 = self.M200[idx] * ct.Mtng_Msun * ct.Msun_g / self.hub
            elif x == "Group/Group_M_Crit500":
                self.M500 = self.M500[idx] * ct.Mtng_Msun * ct.Msun_g / self.hub
            elif x == "Group/Group_M_TopHat200":
                self.Mvir = self.Mvir[idx] * ct.Mtng_Msun * ct.Msun_g / self.hub
            elif x == "Group/Group_R_Crit200":
                self.R200 = self.R200[idx] * ct.kpc_cm * self.axp / self.hub
            elif x == "Group/Group_R_Crit500":
                self.R500 = self.R500[idx] * ct.kpc_cm * self.axp / self.hub
            elif x == "Group/Group_R_TopHat200":
                self.Rvir = self.Rvir[idx] * ct.kpc_cm * self.axp / self.hub
            elif x == "Group/GroupLenType":
                offsets = np.zeros(self.GrLenType.shape, dtype=np.int)
                offsets[1:] = np.cumsum(self.GrLenType[:-1], axis=0)
                self.GrLenType = self.GrLenType[idx]
                self.OffType = offsets[idx]
                del offsets
            elif x == "Group/GroupFirstSub":
                self.FirstSub = self.FirstSub[idx]
            elif x == "Group/GroupNsubs":
                self.Nsubs = self.Nsubs[idx]
            elif x == "Group/GroupVel":
                self.Vbulk = self.Vbulk[idx] * ct.km_cm / self.axp
            elif x == "Subhalo/SubhaloLenType":
                store = []
                for j in range(0, len(idx), 1):
                    tmp = self.SubLenType[
                        self.FirstSub[j] : self.FirstSub[j] + self.Nsubs[j]
                    ]
                    offsets = np.zeros(tmp.shape, dtype=np.int)
                    offsets[1:] = np.cumsum(tmp[:-1], axis=0)
                    store.append(offsets)
                    del tmp, offsets
                self.SubLenType = np.array(store)
                del store
        return

    def error(self, mpi, code):
        """
        Deals with any errors that crop up

        Arguments:
          -mpi  : Instance of the MPI environment class
          -code : Error code
        """

        if mpi.Rank == 0:
            print("ERROR:", flush=True)
            if code is "PATH":
                print("   Path supplied does not exist!", flush=True)
            elif code is "FILES":
                print("   No subfind files found in specificed path!", flush=True)
            elif code is "TABLE_STORE":
                print("   Subfind table is not specified for storage!", flush=True)
            print("--> EXITING!", flush=True)
        mpi.comm.Barrier()
        quit()
        return
