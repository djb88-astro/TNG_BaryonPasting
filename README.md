# TNG_BaryonPasting
This is the repository for the IllustrisTNG data extract code used to provide the required quantities for various Baryon Pasting projects.
Due to the need to for properties that extend well beyond what is traditionally considered a halo (i.e. beyond various spherical overdensity definitions) there is a class for reading all particles in a given snapshot.
This is due to TNG's data store approach, which organises particles into groups defined by a Friend-of-Friend's percolation algorithm and then places particles outside of groups together at the end.
Reading the entire snapshot requires a substantial amount (TBs) of memory due to the numerical resolution and size of the TNG simulations.
Therefore, feel free to use this code, but make sure you have a HPC facilty handy.
