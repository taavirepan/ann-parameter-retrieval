# Artificial neural networks used to retrieve effective properties of metamaterials

## Introduction

Codes to accompany the paper "Artificial neural networks used to retrieve effective properties of metamaterials" (to be published). 

## Requirements

tensorflow 2.x, h5py, [progress](https://pypi.org/project/progress/)

## Example usage

    # (1) Subdivide the "narrow" parameter space 15 times and save the results to data/narrow.hdf5.
	python generate-dataset.py narrow 15
	# or, to run it in parallel with 8 generating threads (-j8) and to run clustering stage in separate threads (-p)
	python generate-dataset.py narrow 15 -j8 -p

	# (2) Train the networks for each subset in the generated data. Multiple networks can be trained in parallel with -p option (in here 8 trainings in parallel---each individual training will only use one CPU core)
	python train-networks.py data/anns-for-narrow.hdf5 data/narrow.hdf5 -p8

	# (3) Use the trained ANNs to fit the data (here only for 150nm lattice spacing dataset)
	python fit.py -o data/fit-ann.hdf5 ann data/anns-for-narrow.hdf5 h=150

	# (4*) If needed, run the least-squares fitting for comparison
	python fit.py -o data/fit-lstsqr.hdf5 lstsqr h=150

For more usage details use `-h`/`--help` option of the scripts.