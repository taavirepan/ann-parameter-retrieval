from .analytic import local_model
from collections import namedtuple
import numpy as np
import h5py
from scipy import optimize

def load_model(data):
	import tensorflow as tf
	model = tf.keras.models.model_from_json(data.attrs["json"])
	for layer in model.layers:
		weights = [np.array(w) for w in data[layer.name].values()]
		layer.set_weights(weights)
	return model

def convert_data(r, t, k0, h):
	f1 = np.linspace(0, 1, 100)
	f2 = np.linspace(0, 1, 8)
	x = np.zeros((1, 2+32))
	x[:,0] = k0
	x[:,1] = h
	for j,a in enumerate([r.real, r.imag, t.real, t.imag]):
		x[0,2+j*8:10+j*8] = np.interp(f2, f1, a)
	return x

FitResult = namedtuple("FitResult", "epsilonx epsilonz mu error nfev model")
class Fitter:
	def __init__(self, network, refine):
		self.refine = refine
		self.models = dict()
		with h5py.File(network, "r") as file:
			for key,run in file.items():
				if run["data"].attrs["leaf"]:
					self.models[key] = load_model(run["model"])
		print(f"Loaded {len(self.models)} networks")
		
	def _refine(self, r0, t0, k0, h, guess):
		kx = np.linspace(0, 1-1e-6, 100)
		def fn(parameters):
			r, t = local_model(parameters, k0, h, kx, full_output=True)
			dr, dt = r-r0, t-t0
			return np.concatenate([dr.real, dr.imag, dt.real, dt.imag])
		
		x0 = np.array([guess.epsilonx.real, guess.epsilonx.imag, guess.epsilonz.real, guess.epsilonz.imag, guess.mu.real, guess.mu.imag])
		bounds = x0 - 0.5, x0 + 0.5
		res = optimize.least_squares(fn, x0, bounds=bounds)
		return FitResult(res.x[0]+1j*res.x[1], res.x[2]+1j*res.x[3], res.x[4]+1j*res.x[5], res.cost, res.nfev+guess.nfev, guess.model)

	def _try_fit(self, r0, t0, k0, h, model_name, nfev):
		model = self.models[model_name]
		model_x = convert_data(r0, t0, k0, h)
		model_y = model.predict(model_x)
		parameters = np.array(model_y[0,:])
		kx = np.linspace(0, 1-1e-6, 100)
		r, t = local_model(parameters, k0, h, kx, full_output=True)
		dr, dt = r-r0, t-t0
		error = np.concatenate([dr.real, dr.imag, dt.real, dt.imag])
		error = 0.5*np.sum(error**2)

		ret = FitResult(parameters[0]+parameters[1]*1j, parameters[2]+parameters[3]*1j, parameters[4]+parameters[5]*1j, error, nfev, model_name)
		if self.refine == "all":
			return self._refine(r0, t0, k0, h, ret)
		else:
			return ret


	def __call__(self, r, t, k0, h):
		models = list(self.models.keys())
		results = list(map(lambda model: self._try_fit(r, t, k0, h, model, len(models)), models))
		results.sort(key=lambda result: result.error)
		best = results[0]

		if self.refine == "best":
			to_refine = results[0:1]
		elif self.refine == "best-3":
			to_refine = results[0:3]
		elif self.refine == "best-7":
			to_refine = results[0:7]
		else:
			to_refine = []

		if len(to_refine) > 0:
			refined = list(map(lambda result: self._refine(r, t, k0, h, result), to_refine))
			refined.sort(key=lambda result: result.error)
			best = refined[0]
		return best
