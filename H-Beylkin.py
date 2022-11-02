from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import NuclearPotential as nucpot
from orbital4c import complex_fcn as cf
import numpy as np
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar

def analytic_1s(light_speed, n, k, Z):
    alpha = 1/light_speed
    gamma = orb.compute_gamma(k,Z,alpha)
    tmp1 = n - np.abs(k) + gamma
    tmp2 = Z * alpha / tmp1
    tmp3 = 1 + tmp2**2
    return light_speed**2 / np.sqrt(tmp3)

light_speed = 137.035999084
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = 1
atom = "H"

energy_1s = analytic_1s(light_speed, n, k, Z)
print('Exact Energy',energy_1s - light_speed**2, flush = True)

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=9)
prec = 1.0e-6
origin = [0.1, 0.2, 0.3]  # origin moved to avoid placing the nuclar charge on a node
#origin = [0.0, 0.0, 0.0]

orb.orbital4c.light_speed = light_speed
orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra

def slater_1s(point,origin):
    d0 = point[0]-origin[0]
    d1 = point[1]-origin[1]
    d2 = point[2]-origin[2]
    d = np.sqrt(d0**2 + d1**2 + d2**2)
    return np.exp(-d)

a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
gauss_tree = vp.FunctionTree(mra)

vp.advanced.build_grid(out=gauss_tree, inp=gauss)

f = lambda x: slater_1s(x, origin)
vp.advanced.project(prec=prec, out=gauss_tree, inp=f)
gauss_tree.normalize()

spinor_H2 = orb.orbital4c()
La_comp = cf.complex_fcn()
La_comp.copy_fcns(real = gauss_tree)
spinor_H2.copy_components(La = La_comp)
spinor_H2.init_small_components(prec/10)
spinor_H2.normalize()

spinor_H = orb.orbital4c()
spinor_H.copy_components(
    La = spinor_H2['Sa'],
    Lb = spinor_H2['Sb'],
    Sa = spinor_H2['La'],
    Sb = spinor_H2['Lb'],
)

Peps = vp.ScalingProjector(mra,prec)
print("before", origin, Z, prec)
#f = lambda x: nucpot.point_charge(x, origin, Z)
f = lambda x: nucpot.coulomb_HFYGB(x, origin, Z, prec)
#f = lambda x: nucpot.homogeneus_charge_sphere(x, origin, Z, atom)
V_tree = Peps(f)

default_der = 'PH'

orbital_error = 1
#while orbital_error > prec:
for idx in range(3):
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = default_der)
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec) 
    add_psi = hd_psi + v_psi
    energy, imag = spinor_H.dot(add_psi)
    print('Energy',energy-light_speed**2,imag)
#    tmp = orb.apply_dirac_hamiltonian(v_psi, prec, energy)
    tmp = orb.apply_helmholtz(v_psi, energy, prec)
    tmp.crop(prec/10)
#    new_orbital = orb.apply_helmholtz(tmp, energy, prec)
    new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = default_der) 
    new_orbital.crop(prec/10)
    new_orbital.normalize()
    delta_psi = new_orbital - spinor_H
    orbital_error, imag = delta_psi.dot(delta_psi)
    print('Error',orbital_error, imag, flush = True)
    spinor_H = new_orbital
    
hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = default_der)
v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
add_psi = hd_psi + v_psi
energy, imag = spinor_H.dot(add_psi)
print('Final Energy',energy - light_speed**2)

#exact_orbital = orb.orbital4c()
#orb.init_1s_orbital(exact_orbital,k,Z,n,alpha,origin,prec)
#exact_orbital.normalize()

energy_1s = analytic_1s(light_speed, n, k, Z)

#hd_psi = orb.apply_dirac_hamiltonian(exact_orbital, prec)
#v_psi = orb.apply_potential(-1.0, V_tree, exact_orbital, prec)
#add_psi = hd_psi + v_psi
#energy, imag = exact_orbital.dot(add_psi)
print('Exact Energy',energy_1s - light_speed**2)
print('Difference',energy_1s - energy)
