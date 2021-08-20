# Example data used for fitting in the paper

This data was generated using in-house T-Matrix code for dipolar polarisabilites arranged in a square grid (fit-data.hdf5) and rectangular grid (fit-anisotropic-data.hdf5).

Each file is organized in groups for different lattice constants, and in each group there are:
 * Dataset `k0`, giving the free-space wavenumbers for the frequency sweep (240 frequencies)
 * Datasets `r` and `t` giving the complex transmission and reflection coefficients, for the 240 frequencies and 100 incidence angles (from 0 to pi/2).

