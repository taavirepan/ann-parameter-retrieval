import numpy as np

def local_model(parameters, k0, h=1.0, kx=None, full_output=False, N=8):
	epsilonx = complex(parameters[0], parameters[1])
	epsilonz = complex(parameters[2], parameters[3])
	muy = complex(parameters[4], parameters[5])

	if kx is None:
		theta = np.arcsin(np.linspace(0, 1-1e-6, N))
	elif isinstance(kx, np.ndarray) and len(kx) > 2:
		theta = np.arcsin(kx)
	else:
		theta = np.arcsin(np.linspace(kx[0], kx[1], N))
	

	Complex = complex
	Power = lambda a,b: a**b
	Sin = np.sin
	Cos = np.cos
	
	def Sqrt(x):
		y = np.sqrt(x)
		return np.where(y.imag<0, -y, y)

	R = (1j*(-(k0**2*Power(Cos(theta),2)) + (k0**2*(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/Power(epsilonx,2))*Sin(h*k0*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz)))/((2*k0**2*Cos(theta)*Cos(h*k0*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/epsilonx - 1j*(k0**2*Power(Cos(theta),2) + (k0**2*(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/Power(epsilonx,2))*Sin(h*k0*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz)))
	T = (2*k0**2*Cos(theta)*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/(epsilonx*((2*k0**2*Cos(theta)*Cos(h*k0*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/epsilonx - 1j*(k0**2*Power(Cos(theta),2) + (k0**2*(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))/Power(epsilonx,2))*Sin(h*k0*Sqrt(epsilonx*muy - (epsilonx*Power(Sin(theta),2))/epsilonz))))

	if full_output:
		return -R, T

	ret = np.concatenate([-R.real, -R.imag, T.real, T.imag])
	return ret.reshape(1, 4*len(theta))

