from collections import namedtuple
from .analytic import local_model
from scipy import optimize
import numpy as np

FitResult = namedtuple("FitResult", "epsilonx epsilonz mu error nfev")

class Fitter:
	def __init__(self, tries, anisotropic):
		self.tries = tries
		self.anisotropic = anisotropic
		self.epsilon = -10, 10
		self.mu = -10, 10
		self.loss = -2, 2
		self.rng = np.random.RandomState(seed=1)

	@property
	def bounds(self):
		if self.anisotropic:
			return ([self.epsilon[0], self.loss[0], self.epsilon[0], self.loss[0], self.mu[0], self.loss[0]], 
				[self.epsilon[1], self.loss[1], self.epsilon[1], self.loss[1], self.mu[1], self.loss[1]])
		else:
			return [self.epsilon[0], self.loss[0], self.mu[0], self.loss[0]], [self.epsilon[1], self.loss[1], self.mu[1], self.loss[1]]
	
	def x0(self):
		if self.anisotropic:
			return [
				self.rng.uniform(*self.epsilon), self.rng.uniform(*self.loss),
				self.rng.uniform(*self.epsilon), self.rng.uniform(*self.loss),
				self.rng.uniform(*self.mu), self.rng.uniform(*self.loss)]
		else:
			return [
				self.rng.uniform(*self.epsilon), self.rng.uniform(*self.loss),
				self.rng.uniform(*self.mu), self.rng.uniform(*self.loss)]

	def _try_fit(self, r0, t0, k0, h, x0):
		kx = np.linspace(0, 1-1e-6, 100)
		def fn(x):
			parameters = x if self.anisotropic else [x[0], x[1], x[0], x[1], x[2], x[3]]
			r, t = local_model(parameters, k0, h, kx, full_output=True)
			dr, dt = r-r0, t-t0
			return np.concatenate([dr.real, dr.imag, dt.real, dt.imag])
		res = optimize.least_squares(fn, x0, bounds=self.bounds)
		if self.anisotropic:
			return FitResult(res.x[0]+1j*res.x[1], res.x[2]+1j*res.x[3], res.x[4]+1j*res.x[5], res.cost, res.nfev)
		else:
			return FitResult(res.x[0]+1j*res.x[1], res.x[0]+1j*res.x[1], res.x[2]+1j*res.x[3], res.cost, res.nfev)


	def __call__(self, r, t, k0, h):
		best = self._try_fit(r, t, k0, h, self.x0())
		for i in range(self.tries-1):
			current = self._try_fit(r, t, k0, h, self.x0())
			if current.error < best.error:
				best = FitResult(current.epsilonx, current.epsilonz, current.mu, current.error, best.nfev + current.nfev)
			else:
				best = FitResult(best.epsilonx, best.epsilonz, best.mu, best.error, best.nfev + current.nfev)
		return best
