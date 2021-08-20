import h5py
import numpy as np
from code import ANNFitter, LstSqrFitter
from time import monotonic
from progress import bar
from os.path import exists

def run_fit(fitter, data, output):
	h = data.attrs["h"]
	r = np.array(data["r"])
	t = np.array(data["t"])

	output.attrs["h"] = h
	output.create_dataset("k0", data=data["k0"])

	N = len(data["k0"])
	epsilonx = output.create_dataset("epsilonx", (N,), "c8")
	epsilonz = output.create_dataset("epsilonz", (N,), "c8")
	output["epsilon"] = h5py.SoftLink(output.name + "/epsilonx")
	mu = output.create_dataset("mu", (N,), "c8")
	error = output.create_dataset("error", (N,), "f4")
	nfev = output.create_dataset("nfev", (N,), "i4")
	time = output.create_dataset("time", (N,), "f4")
	with bar.Bar(data.name, max=N) as b:
		for i,k0 in enumerate(data["k0"]):
			t1 = monotonic()
			res = fitter(r[i], t[i], k0, h)
			t2 = monotonic()
			epsilonx[i] = res.epsilonx
			epsilonz[i] = res.epsilonz
			mu[i] = res.mu
			error[i] = res.error
			nfev[i] = res.nfev
			time[i] = t2-t1
			b.next()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", default="data/fit-data.hdf5")
	parser.add_argument("-o", "--output", required=True)
	method = parser.add_subparsers(dest="method", required=True)
	
	lstsq = method.add_parser("lstsqr")
	lstsq.add_argument("-n", "--nguess", type=int, default=10, help="Number of initial guesses")
	lstsq.add_argument("--anisotropic", action="store_true")
	
	ann = method.add_parser("ann")
	ann.add_argument("--refine", choices=["all", "best", "best-3", "best-7"], help="Refine results (all, best guess only, best-n guesses only)")
	ann.add_argument("--anisotropic", action="store_true")
	ann.add_argument("ann")
	
	lstsq.add_argument("data", nargs="*")
	ann.add_argument("data", nargs="*")

	args = parser.parse_args()

	if args.method == "lstsqr":
		fitter = LstSqrFitter(args.nguess, args.anisotropic)
	else:
		fitter = ANNFitter(args.ann, args.refine)
		

	with h5py.File(args.output, "w") as f_out, h5py.File(args.input, "r") as f_in:
		if args.method == "lstsqr":
			f_out.attrs["bounds:epsilon"] = fitter.epsilon
			f_out.attrs["bounds:mu"] = fitter.mu
			f_out.attrs["bounds:loss"] = fitter.loss
		keys = args.data or f_in.keys()
		if isinstance(fitter, list):
			for level,make_fitter in fitter:
				fitter_i = make_fitter(level)
				g = f_out.create_group(f"level={level:03d}")
				g.attrs["models"] = len(fitter_i.models)
				g.attrs["level"] = level
				for key in keys:
					run_fit(fitter_i, f_in[key], g.create_group(key))
		else:
			if hasattr(fitter, "models"):
				f_out.attrs["models"] = len(fitter.models)
			for key in keys:
				run_fit(fitter, f_in[key], f_out.create_group(key))

