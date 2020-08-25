import argparse
import h5py
import merge
import numpy as np
import mpi_init
import subfind_data
import volume_read
import halo_properties as hp


def measure_profile_properties(mpi, path, snap, extent=5.0, R200scale=True, \
                               Nbins=25):
    """
    Compute various profiles and quantities for halos above given mass

    Arguments:
      -mpi       : MPI environment class instance
      -path      : Path to simulation of interest
      -snap      : Snapshot of interest
      -extent    : Extent of the radial profiles
      -R200scale : BOOLEAN - if True: profiles extend EXTENT * R200, else [Mpc]
      -Nbins     : NUmber of bins in the radial profile
    """

    # First, rebuild the Subfind table for this snapshot
    subfind_table = subfind_data.build_table(mpi, sim=path, snap=snap)

    # Load the entire volume
    volume = volume_read.entire_snapshot_read(mpi, sim=path, snap=snap)

    # Read specified datasets, compute desired quantities, clean up
    required_datasets = ['PartType1/Coordinates',
                         'PartType1/Masses',
                         'PartType1/Velocities']
    volume.read_datasets(mpi, required_datasets)
    volume.tag_subhalo_particles(mpi, subfind_table)

    # Now select halos of interest
    #subfind_table.select_halos(mpi, cut=1.0e12)
    subfind_table.select_halos(mpi, cut='MAX')
    if not mpi.Rank: print(' > Found {0:d} halo(s)'.format(len(subfind_table.tags)), flush=True)

    # Initiate halo class, distribute required particles to tasks, store
    h = hp.halo(volume, Nbins=Nbins)

    h.halo_data_store(mpi, subfind_table, volume, Extent=extent, \
                      R200scale=R200scale)
    del subfind_table, volume

    # Now loop over haloes that were sent to my task
    hk = sorted(h.halo_data.keys())
    if not mpi.Rank: print(' > Computing halo properties', flush=True)
    for j in range(0, len(hk), 1):
        if not mpi.Rank: print('  -{0}'.format(hk[j]), flush=True)

        h.set_up_halo(mpi, hk[j])

        # Compute mass profiles
        h.compute_mass_profiles(mpi)

        # Velocities and non-thermal pressure profiles
        h.compute_velocities_and_non_thermal_pressure(mpi)

        # Centre of mass offset
        h.compute_centre_of_mass_offset(mpi)

        # Compute NFW concentration - 1 parameter fit
        h.compute_concentration(mpi)

        # DM surface pressure
        h.compute_DM_surface_pressure(mpi)

        # Cumulative shape measurements
        h.compute_shape(mpi, aperture='500', ptype='DM')
        h.compute_shape(mpi, aperture='200', ptype='DM')
        h.compute_shape(mpi, aperture='Vir', ptype='DM')

        # Dark matter shape profile
        try:
            h.compute_shape_profile(mpi, ptype='DM')
        except:
            h.q_dm  = np.zeros(Nbins, dtype=np.float)
            h.s_dm  = np.zeros(Nbins, dtype=np.float)
            h.Iv_dm = np.zeros((Nbins, 3, 3), dtype=np.float)

        # Store halo properties in HDF5 file
        h.save(mpi)

    # Now merge files
    mpi.comm.Barrier()
    if not mpi.Rank:
        ftag = '{0}_Snap{1:03d}_z{2:d}p{3:02d}'.format(h.fname, h.snap, int(h.redshift), int(100.0 * (h.redshift - int(h.redshift))))
        merge.merge_outputs(ftag)
    mpi.comm.Barrier()
    del h
    return

if __name__ == "__main__":

    # Parse command-line options -- deals with missing arguments!
    parser = argparse.ArgumentParser(description='TNG analysis script to produce radial profiles')
    parser.add_argument('start', action='store', type=int, help='First snapshot to process')
    parser.add_argument('final', action='store', type=int, help='Final snapshot to process')
    inputs = parser.parse_args()

    # Simulation of interest
    paths = ['/n/hernquistfs3/IllustrisTNG/Runs/L75n455TNG_DM/output']

    # Snapshots of interest
    snapshots = np.arange(inputs.final - inputs.start + 1) + inputs.start

    # Initialize MPI environment
    mpi = mpi_init.mpi()

    # Loop over sims and snapshots measuring profiles
    for x in paths:
        for y in snapshots:
            if not mpi.Rank:
                print('--- SNAPSHOT {0:03d} ---'.format(y), flush=True)
            measure_profile_properties(mpi, x, y)
