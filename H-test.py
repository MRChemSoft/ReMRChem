from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
import numpy as np
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar


def u(r):
    u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
    #erf(r) is an error function that is supposed to stop the potential well from going to inf.
    #if i remember correctly
    return u

def V(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    c = 0.0435
    c = 0.000435 # ten times tighter nuclear potential
    f_bar = u(r/c)/c
    return f_bar

c = 137   # NOT A GOOD WAY. MUST BE FIXED!!!
alpha = 1/c
k = -1
l = 0
n = 1
m = 0.5
Z = 1

mra = vp.MultiResolutionAnalysis(box=[-20,20], order=7)
prec = 1.0e-5
origin = [0.1, 0.2, 0.3]  # origin moved to avoid placing the nuclar charge on a node

orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra

a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
gauss_tree = vp.FunctionTree(mra)
vp.advanced.build_grid(out=gauss_tree, inp=gauss)
vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
gauss_tree.normalize()

spinor_H = orb.orbital4c()
La_comp = cf.complex_fcn()
La_comp.copy_fcns(real = gauss_tree)
spinor_H.copy_components(La = La_comp)
spinor_H.init_small_components(prec/10)
spinor_H.normalize()

Peps = vp.ScalingProjector(mra,prec)
f = lambda x: V([x[0]-origin[0],x[1]-origin[1],x[2]-origin[2]])
V_tree = Z*Peps(f)

orbital_error = 1
while orbital_error > prec:
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec)
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
    add_psi = hd_psi + v_psi
    energy, imag = spinor_H.dot(add_psi)
    print('Energy',energy,imag)
    tmp = orb.apply_helmholtz(v_psi, energy, c, prec)
    new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy)
    new_orbital.normalize()
    delta_psi = new_orbital - spinor_H
    orbital_error, imag = delta_psi.dot(delta_psi)
    print('Error',orbital_error, imag)
    spinor_H = new_orbital
    
hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec)
v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
add_psi = hd_psi + v_psi
energy, imag = spinor_H.dot(add_psi)
print('Final Energy',energy)

exact_orbital = orb.orbital4c()
orb.init_1s_orbital(exact_orbital,k,Z,n,alpha,origin,prec)
exact_orbital.normalize()

hd_psi = orb.apply_dirac_hamiltonian(exact_orbital, prec)
v_psi = orb.apply_potential(-1.0, V_tree, exact_orbital, prec)
add_psi = hd_psi + v_psi
energy, imag = exact_orbital.dot(add_psi)
print('Exact Energy',energy)

