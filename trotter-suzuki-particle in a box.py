# simulate a particle in a 2D box with QuTip (periodic boundary condition)

from matplotlib.colors import LogNorm
from pylab import *

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

from qutip import *

# Hilbert spaces
npx = 2
npy = 2

# box
lbox = 10

# particle
mass = 1

# operators
Px_p = 2*np.pi/lbox * tensor(num(npx), sigmaz(), qeye(npy), sigmaz())
Py_p = 2*np.pi/lbox * tensor(qeye(npx), sigmaz(), num(npy), sigmaz())

H_p = (Px_p*Px_p + Py_p*Py_p)*0.5/mass

# state
#px = 1
#sign_x = 0
#py = 1
#sign_y = 0
#psi0_p = tensor(basis(npx, px), basis(2,sign_x), basis(npy, py), basis(2,sign_y)) *(-1.j)
#psi0_p += tensor(basis(npx, px), basis(2,1), basis(npy, py), basis(2,1)) *(1.j)

# state sin(x)sin(y) : #|1,1> + |-1,-1> -|1,-1> -|-1,1>
psi0_p = tensor(basis(npx, 1), basis(2,0), basis(npy, 1), basis(2,0))
psi0_p += tensor(basis(npx, 1), basis(2,1), basis(npy, 1), basis(2,1))
psi0_p -= tensor(basis(npx, 1), basis(2,0), basis(npy, 1), basis(2,1))
psi0_p -= tensor(basis(npx, 1), basis(2,1), basis(npy, 1), basis(2,0))
psi0_p = psi0_p * (-0.25)

# simulation
tlist = np.linspace(0,10,100)
e_ops = [H_p, Px_p, Py_p]
result = mesolve(H_p, psi0_p, tlist, [], e_ops)

plot_expectation_values(result);
# show results
#fig, axes = plt.subplots(1,1)

#axes.plot(tlist, expect(Px_p, result.states))

#axes.set_xlabel(r'$t$', fontsize=20)
#axes.set_ylabel(r'$P_x$', fontsize=20);


def p_to_x_basis2D(psi_p, npx, npy, nx, ny):

	#const = 1/(2*np.pi)
	const = 1.
	psi_x = tensor(basis(nx,0), basis(ny,0)) * 0.
	
	for x in range(0,nx):
		for y in range(0,ny):
			var = 0.+0.j
			for px in range(0,npx):
				for sign_x in range(0,2):
					for py in range(0,npy):
						for sign_y in range(0,2):
							ket = tensor(basis(npx, px), basis(2,sign_x), basis(npy, py), basis(2,sign_y))
							var += const * np.exp(( np.power(-1.,sign_x)*x*px/float(nx) + np.power(-1.,sign_y)*y*py/float(ny) )*2j*np.pi) * ket.overlap(psi_p)
			psi_x += tensor(basis(nx,x), basis(ny,y)) * var
			
	return psi_x

psi0_x = p_to_x_basis2D(psi0_p, npx, npy, lbox, lbox)
#print(psi0_x)
#Qobj(psi0_x)
#psi0_x = psi0_x.unit()

def plot_psi_coord2D(psi, nx, ny):
	arr = psi.dag().full()
	data = arr[0,0:ny]
	for x in range(1,nx):
		data = np.vstack((data,arr[0,x*ny:(x+1)*ny]))
	data_real = real(data)
	fig = plt.figure()	
	ax = fig.add_subplot(111)
	cax = ax.matshow(data_real, interpolation='nearest')
	fig.colorbar(cax)
	plt.show()
	
	return None

plot_psi_coord2D(psi0_x, lbox, lbox)
