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

c = 137.035999084   # NOT A GOOD WAY. MUST BE FIXED!!!

alpha = 1/c
k = -1
l = 0
n = 1
m = 0.5
Z = 1

mra = vp.MultiResolutionAnalysis(box=[-20,20], order=7)
prec = 1.0e-5
origin1 = [0.0, 0.0, -1.0]  # origin moved to avoid placing the nuclar charge on a node
origin2 = [0.0, 0.0,  1.0]  # origin moved to avoid placing the nuclar charge on a node

def VH2(x,origin1,origin2):
    V1 = V([x[0]-origin1[0],x[1]-origin1[1],x[2]-origin1[2]])
    V2 = V([x[0]-origin2[0],x[1]-origin2[1],x[2]-origin2[2]])
    return V1 + V2

orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra

a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss1 = vp.GaussFunc(b_coeff, a_coeff, origin1)
gauss2 = vp.GaussFunc(b_coeff, a_coeff, origin2)
gauss1_tree = vp.FunctionTree(mra)
gauss2_tree = vp.FunctionTree(mra)
vp.advanced.build_grid(out=gauss1_tree, inp=gauss1)
vp.advanced.project(prec=prec, out=gauss1_tree, inp=gauss1)
vp.advanced.build_grid(out=gauss2_tree, inp=gauss2)
vp.advanced.project(prec=prec, out=gauss2_tree, inp=gauss2)

h2p_orb = gauss1_tree + gauss2_tree
h2p_orb.normalize()

spinor_H = orb.orbital4c()
La_comp = cf.complex_fcn()
La_comp.copy_fcns(real = h2p_orb)
spinor_H.copy_components(La = La_comp)
spinor_H.init_small_components(prec/10)
spinor_H.normalize()

Peps = vp.ScalingProjector(mra,prec)
f = lambda x: VH2(x, origin1, origin2)
V_tree = Peps(f)

orbital_error = 1
while orbital_error > prec:
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec)
    
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
    add_psi = hd_psi + v_psi
    energy, imag = spinor_H.dot(add_psi)
    print('Energy',energy-c**2,imag)
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
print('Final Energy',energy-c**2)

