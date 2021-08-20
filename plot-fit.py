import matplotlib.pyplot as mpl
import numpy as np
import h5py
import argparse
from matplotlib import gridspec

class Plot:
	def __init__(self, anisotropic):
		self.axes = dict()
		self.figure = mpl.figure()
		gs = gridspec.GridSpec(2, 4 if anisotropic else 3, height_ratios=[2,1])

		if not anisotropic:
			self.axes["epsilon"] = ax = self.figure.add_subplot(gs[0,0])
			ax.set_xlabel("k0")
			ax.set_ylabel("epsilon")
			n = 1
		else:
			self.axes["epsilonx"] = ax = self.figure.add_subplot(gs[0,0])
			ax.set_xlabel("k0")
			ax.set_ylabel("epsilonx")
			self.axes["epsilonz"] = ax = self.figure.add_subplot(gs[0,1])
			ax.set_xlabel("k0")
			ax.set_ylabel("epsilonz")
			n = 2

		self.axes["mu"] = ax = self.figure.add_subplot(gs[0,n])
		ax.set_xlabel("k0")
		ax.set_ylabel("mu")

		self.error = self.figure.add_subplot(gs[0,n+1])
		self.error.set_xlabel("k0")
		self.error.set_ylabel("Error")
		self.error.set_yscale("log")
		
		self.nfev = self.figure.add_subplot(gs[1,0])
		self.nfev.set_ylabel("Function evaluations")
		self.time = self.figure.add_subplot(gs[1,1])
		self.time.set_ylabel("Time (sec)")

	def __call__(self, label, k0, error, nfev, time, run, *, imag):
		for key,ax in self.axes.items():
			if key in run:
				data = np.array(run[key])
			elif key.startswith("epsilon"):
				data = np.array(run["epsilon"])
			l, = ax.plot(k0, data.real)
			if imag:
				ax.plot(k0, data.imag, ls="-.", color=l.get_color())
		self.error.plot(k0, error, label=label)
		self.nfev.plot(k0, nfev, ".")
		self.time.plot(k0, time, ".")
	
	def finish(self):
		# self.error.legend()
		self.nfev.set_ylim(0)
		self.time.set_ylim(0)

def plot_run(run, plot, **kwargs):
	k0 = np.array(run["k0"])
	epsilon = np.array(run["epsilon"])
	mu = np.array(run["mu"])
	error = np.array(run["error"])
	nfev = np.array(run["nfev"])
	time = np.array(run["time"])
	plot(run.file.filename, k0, error, nfev, time, run, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+")
parser.add_argument("-s", "--save")
parser.add_argument("-r", "--runs", nargs="+")
parser.add_argument("-a", "--anisotropic", action="store_true")
parser.add_argument("-i", "--imag", action="store_true")
args = parser.parse_args()

plot = Plot(args.anisotropic)
for filename in args.file:
	with h5py.File(filename, "r") as file:
		for name in args.runs or file.keys():
			run = file[name]
			plot_run(run, plot, imag=args.imag)
			print(filename, run.name, file.attrs.get("models"), np.sum(run["time"]))

plot.finish()
if args.save:
	mpl.savefig(args.save)
else:
	mpl.show()