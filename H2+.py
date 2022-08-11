from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import NuclearPotential as nucpot
from orbital4c import complex_fcn as cf
import numpy as np
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar

light_speed = 137.035999084
Z = 1

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=13)
prec = 1.0e-8
origin1 = [0.1, 0.2, -0.7]  # origin moved to avoid placing the nuclar charge on a node
origin2 = [0.1, 0.2,  1.3]  # origin moved to avoid placing the nuclar charge on a node

def VH2(x, origin1, origin2, Z1, Z2, prec):
    V1 = nucpot.coulomb_HFYGB(x, origin1, Z1, prec)
    V2 = nucpot.coulomb_HFYGB(x, origin2, Z2, prec)
    return V1 + V2

orb.orbital4c.light_speed = light_speed
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
f = lambda x: VH2(x, origin1, origin2, Z, Z, prec)
V_tree = Peps(f)

default_der = 'PH'
orbital_error = 1
while orbital_error > prec:
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = default_der)
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
    add_psi = hd_psi + v_psi
    energy, imag = spinor_H.dot(add_psi)
    print('Energy',energy - light_speed**2,imag)
    tmp = orb.apply_helmholtz(v_psi, energy, prec)
    tmp.crop(prec/10)
    new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = default_der)
    new_orbital.crop(prec/10)
    new_orbital.normalize()
    delta_psi = new_orbital - spinor_H
    orbital_error, imag = delta_psi.dot(delta_psi)
    print('Error',orbital_error, imag)
    spinor_H = new_orbital
    
hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = default_der)
v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
add_psi = hd_psi + v_psi
energy, imag = spinor_H.dot(add_psi)
print('Final Energy',energy - light_speed**2)

