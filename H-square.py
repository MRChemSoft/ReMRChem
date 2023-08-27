from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import nuclear_potential as nucpot
from orbital4c import complex_fcn as cf
import numpy as np
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar

import argparse
import numpy as np
import numpy.linalg as LA
import sys, getopt


import importlib
importlib.reload(orb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-a', '--atype', dest='atype', type=str, default='H',
                        help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='BS',
                        help='put the type of derivative')
    parser.add_argument('-z', '--charge', dest='charge', type=float, default=1.0,
                        help='put the atom charge')
    parser.add_argument('-b', '--box', dest='box', type=int, default=30,
                        help='put the box dimension')
    parser.add_argument('-cx', '--center_x', dest='cx', type=float, default=0.1,
                        help='position of nucleus in x')
    parser.add_argument('-cy', '--center_y', dest='cy', type=float, default=0.2,
                        help='position of nucleus in y')
    parser.add_argument('-cz', '--center_z', dest='cz', type=float, default=0.3,
                        help='position of nucleus in z')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=6,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-4,
                        help='put the precision')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    parser.add_argument('-s', '--save', dest='saveOrbitals', type=str, default='No',
                        help='Save the FunctionTree for the alpha spinorbital called as Z')
    parser.add_argument('-r', '--read', dest='readOrbitals', type=str, default='No',
                        help='Read the FunctionTree for the aplha spinorbital called as Z')
    args = parser.parse_args()

    assert args.potential in ['point_charge', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS', 'ABGV'], 'Please, specify the type of derivative'

    assert args.readOrbitals in ['Yes', 'No'], 'Please, specify if you want (Yes) or not (No) to read the FunctionTree of spinorbital'
    
    assert args.saveOrbitals in ['Yes', 'No'], 'Please, specify if you want (Yes) or not (No) to save the FunctionTree of spinorbital'

def analytic_1s(light_speed, n, k, Z):
    alpha = 1/light_speed
    gamma = orb.compute_gamma(k,Z,alpha)
    tmp1 = n - np.abs(k) + gamma
    tmp2 = Z * alpha / tmp1
    tmp3 = 1 + tmp2**2
    return light_speed**2 / np.sqrt(tmp3)

light_speed = args.lux_speed
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = args.charge
atom = args.atype
der = args.deriv

energy_1s = analytic_1s(light_speed, n, k, Z)
print('Exact Energy',energy_1s - light_speed**2, flush = True)

mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order)
prec = args.prec
origin = [args.cx, args.cy, args.cz] 

orb.orbital4c.light_speed = light_speed
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

################### Define V potential ######################
if args.potential == 'point_charge':
    Peps = vp.ScalingProjector(mra,prec/10)
    f = lambda x: nucpot.point_charge(x, origin, Z)
    V_tree = Peps(f)
elif args.potential == 'coulomb_HFYGB':
    Peps = vp.ScalingProjector(mra,prec/10)
    f = lambda x: nucpot.coulomb_HFYGB(x, origin, Z, prec)
    V_tree = Peps(f)
elif args.potential == 'homogeneus_charge_sphere':
    Peps = vp.ScalingProjector(mra,prec/10)
    f = lambda x: nucpot.homogeneus_charge_sphere(x, origin, Z, atom)
    V_tree = Peps(f)
elif args.potential == 'gaussian':
    Peps = vp.ScalingProjector(mra,prec/10)
    f = lambda x: nucpot.gaussian(x, origin, Z, atom)
    V_tree = Peps(f)
print('Define V Potential', args.potential, Z, 'DONE')

orbital_error = 1
mc2 = light_speed * light_speed
while orbital_error > prec:
#for idx in range(10):
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec) 
    vv_psi = orb.apply_potential(-0.5/mc2, V_tree, v_psi, prec)
    beta_v_psi = v_psi.beta2()
    apV_psi = v_psi.alpha_p(prec, der)
    ap_psi = spinor_H.alpha_p(prec, der)
    Vap_psi = orb.apply_potential(-1.0, V_tree, ap_psi, prec)
    anticom = apV_psi + Vap_psi
    RHS = beta_v_psi + vv_psi + anticom * (0.5/light_speed)
    cke = spinor_H.classicT()
    cpe,imag = spinor_H.dot(RHS)
    print('classic', cke,cpe,cpe+cke)
    mu = orb.calc_non_rel_mu(cke+cpe)
    print("mu", mu)
    new_orbital = orb.apply_helmholtz(RHS, mu, prec)
    new_orbital.normalize()
    delta_psi = new_orbital - spinor_H
    orbital_error, imag = delta_psi.dot(delta_psi)
    print('Error',orbital_error, imag, flush = True)
    spinor_H = new_orbital
    
hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = 'BS')
v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
add_psi = hd_psi + v_psi
energy, imag = spinor_H.dot(add_psi)

cke = spinor_H.classicT()
beta_v_psi = v_psi.beta2()
beta_pot,imag = beta_v_psi.dot(spinor_H)
pot_sq, imag = v_psi.dot(v_psi)
ap_psi = spinor_H.alpha_p(prec, der)
anticom, imag = ap_psi.dot(v_psi)
energy_kutzelnigg = cke + beta_pot + pot_sq/(2*mc2) + anticom/light_speed

print('Kutzelnigg',cke, beta_pot, pot_sq/(2*mc2), anticom/light_speed, energy_kutzelnigg)
print('Quadratic approx',energy_kutzelnigg - energy_kutzelnigg**2/(2*mc2))
print('Correct from Kutzelnigg', mc2*(np.sqrt(1+2*energy_kutzelnigg/mc2)-1))
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
print('Difference 1',energy_1s - energy)
print('Difference 2',energy_1s - energy_kutzelnigg - light_speed**2)
