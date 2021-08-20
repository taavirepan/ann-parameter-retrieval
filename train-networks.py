import argparse
import h5py
import subprocess
import shutil
import sys
import os
from collections import namedtuple, defaultdict
from pathlib import Path

Training = namedtuple("Training", "data level index")
def get_trainings(file, *, all_runs=True, max_level=None):
	levels = defaultdict(lambda: 0)
	for key, data in file.items():
		if max_level is not None and data.attrs["level"] > max_level: continue
		if all_runs or data.attrs["leaf"] or data.attrs["level"] == max_level:
			level = data.attrs["level"]
			yield Training(key, level, levels[level])
			levels[level] += 1


parser = argparse.ArgumentParser(description="Driver script to run all trainings for a dataset", epilog="Other arguments are passed on to train-network.py")
parser.add_argument("output", help="Output file to save the ANNs to")
parser.add_argument("data", help="Dataset file")
parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of trainings to run in parallel")
parser.add_argument("-l", "--max-level", type=int, help="Only use first L levels from the dataset", metavar="L")
args, other_args = parser.parse_known_args()

running = []
training_files = []
with h5py.File(args.data, "r") as data, h5py.File(args.output, "w") as results:
	max_level = args.max_level or data.attrs["splits"]
	for i,training in enumerate(get_trainings(data, max_level=max_level)):
		run_name = f"level{training.level:02d}.{training.index:03d}"
		if len(running) == args.parallel:
			running.pop(0).wait()
		temporary_file = f"training-{run_name}.hdf5"
		process = subprocess.Popen([sys.executable, "train-network.py", temporary_file, args.data, training.data, run_name] + other_args)
		running.append(process)
		training_files.append(temporary_file)
	for p in running:
		p.wait()
	for result_file in training_files:
		with h5py.File(result_file, "r") as result:
			for k,v in result.items():
				results.copy(v, k)
		os.unlink(result_file)

