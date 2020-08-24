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

    # Find halos of interest
    subfind_table = subfind_data.build_table(mpi, sim=path, snap=snap)
    #subfind_table.select_halos(mpi, cut=1.0e12)
    subfind_table.select_halos(mpi, cut='MAX')

    if not mpi.Rank: print(' > Found {0:d} halo(s)'.format(len(subfind_table.tags)), flush=True)

    # Load the entire volume
    volume = volume_read.entire_snapshot_read(mpi, sim=path, snap=snap)

    # Read specified datasets, compute desired quantities, clean up
    required_datasets = ['PartType0/Coordinates',
                         'PartType0/Density',
                         'PartType0/ElectronAbundance',
                         'PartType0/GFM_Metallicity',
                         'PartType0/InternalEnergy',
                         'PartType0/Masses',
                         'PartType0/Velocities',
                         'PartType1/Coordinates',
                         'PartType1/Masses',
                         'PartType1/Velocities',
                         'PartType4/Coordinates',
                         'PartType4/GFM_Metallicity',
                         'PartType4/GFM_StellarFormationTime',
                         'PartType4/Masses',
                         'PartType4/Velocities',
                         ]
    volume.read_datasets(mpi, required_datasets)
    volume.remove_wind_particles(mpi, required_datasets)
    volume.calculate_gas_temperatures(mpi)
    del volume.inte, volume.STsft

    # Initiate halo class, distribute required particles to tasks, store
    h = read_simulation.halo(volume, Nbins=Nbins)

    hp.halo_data_store(mpi, subfind_table, volume, Extent=extent, \
                      R200scale=R200scale)
    del subfind_table, volume

    # Now loop over haloes that were sent to my task
    hk = sorted(hp.halo_data.keys())
    if not mpi.Rank: print(' > Computing halo properties', flush=True)
    for j in range(0, len(hk), 1):
        if not mpi.Rank: print('  -{0}'.format(hk[j]), flush=True)

        hp.set_up_halo(mpi, hk[j])

        # Compute mass profiles
        hp.compute_mass_profiles(mpi)

        # Velocities and non-thermal pressure profiles
        hp.compute_velocities_and_non_thermal_pressure(mpi)

        # Thermodynamic profiles
        hp.compute_thermo_profiles(mpi)

        # Metallicity profiles
        hp.compute_metallicity_profiles(mpi)

        # Observable properties
        hp.compute_observable_properties(mpi)

        # Centre of mass offset
        hp.compute_centre_of_mass_offset(mpi)

        # Compute NFW concentration - 1 parameter fit
        hp.compute_concentration(mpi)

        # DM surface pressure
        hp.compute_DM_surface_pressure(mpi)

        # Cumulative shape measurements
        hp.compute_shape(mpi, aperture='500', ptype='GAS')
        hp.compute_shape(mpi, aperture='200', ptype='GAS')
        hp.compute_shape(mpi, aperture='Vir', ptype='GAS')

        hp.compute_shape(mpi, aperture='500', ptype='DM')
        hp.compute_shape(mpi, aperture='200', ptype='DM')
        hp.compute_shape(mpi, aperture='Vir', ptype='DM')

        hp.compute_shape(mpi, aperture='500', ptype='STAR')
        hp.compute_shape(mpi, aperture='200', ptype='STAR')
        hp.compute_shape(mpi, aperture='Vir', ptype='STAR')

        # Dark matter shape profile
        try:
            hp.compute_shape_profile(mpi, ptype='DM')
        except:
            hp.q_dm  = np.zeros(Nbins, dtype=np.float)
            hp.s_dm  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_dm = np.zeros((Nbins, 3, 3), dtype=np.float)

        # Gas shape profile
        try:
            hp.compute_shape_profile(mpi, ptype='GAS')
        except:
            hp.q_gas  = np.zeros(Nbins, dtype=np.float)
            hp.s_gas  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_gas = np.zeros((Nbins, 3, 3), dtype=np.float)
        # Hot gas shape profile
        try:
            hp.compute_shape_profile(mpi, ptype='GAS_HOT')
        except:
            hp.q_ghot  = np.zeros(Nbins, dtype=np.float)
            hp.s_ghot  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_ghot = np.zeros((Nbins, 3, 3), dtype=np.float)
        # Gas temperature
        try:
            hp.compute_shape_profile(mpi, ptype='GAS_TEMP')
        except:
            hp.q_temp  = np.zeros(Nbins, dtype=np.float)
            hp.s_temp  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_temp = np.zeros((Nbins, 3, 3), dtype=np.float)
        # Gas pressure
        try:
            hp.compute_shape_profile(mpi, ptype='GAS_PRES')
        except:
            hp.q_pres  = np.zeros(Nbins, dtype=np.float)
            hp.s_pres  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_pres = np.zeros((Nbins, 3, 3), dtype=np.float)
        # Stellar shape profile
        try:
            hp.compute_shape_profile(mpi, ptype='STAR')
        except:
            hp.q_star  = np.zeros(Nbins, dtype=np.float)
            hp.s_star  = np.zeros(Nbins, dtype=np.float)
            hp.Iv_star = np.zeros((Nbins, 3, 3), dtype=np.float)

        hp.save(mpi)

    # Now merge files
    mpi.comm.Barrier()
    if not mpi.Rank:
        ftag = '{0}_Snap{1:03d}_z{2:d}p{3:02d}'.format(hp.fname, hp.snap, int(hp.redshift), int(100.0 * (hp.redshift - int(hp.redshift))))
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
    paths = ['/n/hernquistfs3/IllustrisTNG/Runs/L75n455TNG/output']

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
