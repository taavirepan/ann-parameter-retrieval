import numpy as np

def calculate_spectra(parameters, *, k0=None, h=None, kx=None, Nsamples=8):
	from .analytic import local_model
	N = len(parameters)
	rt = np.empty((N, 2+4*Nsamples))

	p = lambda x: x
	for i in range(N):
		k0_ = 4.0 if k0 is None else k0[i]
		h_ = 0.15 if h is None else h[i]
		rt[i,0] = k0_
		rt[i,1] = h_
		rt[i,2:] = local_model(p(parameters[i]), k0_, h_, kx=kx, N=Nsamples)
	return rt

class Data:
	__slots__ = ["x", "y", "coverage"]
	def __init__(self, x, y, coverage=None):
		self.x = x
		self.y = y
		self.coverage = coverage

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return Data(self.x[idx,:], self.y[idx,:], self.coverage)

	def __add__(self, other):
		n1 = len(self)
		n2 = len(other)
		coverage = (self.coverage*n1 + other.coverage*n2) / (n1 + n2)
		return Data(np.concatenate([self.x, other.x], axis=0), np.concatenate([self.y, other.y], axis=0), coverage)

	@staticmethod
	def from_parameters(parameters, *, k0=None, h=None, kx=None, **kwargs):
		return Data(calculate_spectra(parameters, k0=k0, h=h, kx=kx), parameters, **kwargs)
