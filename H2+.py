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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-d', '--dtype', dest='dtype', type=str, default='dirac',
                        help='Dirac or Dirac-square operators')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'
    assert args.dtype in ['dirac', 'dirac2'], 'Please, specify Dirac-type operator'

def analytic_1s(light_speed, n, k, Z):
    alpha = 1/light_speed
    gamma = orb.compute_gamma(k,Z,alpha)
    tmp1 = n - np.abs(k) + gamma
    tmp2 = Z * alpha / tmp1
    tmp3 = 1 + tmp2**2
    return light_speed**2 / np.sqrt(tmp3)


light_speed = 137.03599913900001
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = 1
atom = 'H'

energy_1s = analytic_1s(light_speed, n, k, Z)
print('Exact Energy',energy_1s - light_speed**2)

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=6)
prec = 1.0e-4
origin1 = [0.1, 0.2, -0.7]  # origin moved to avoid placing the nuclar charge on a node
origin2 = [0.1, 0.2,  1.3]  # origin moved to avoid placing the nuclar charge on a node

################### Define V potential ######################
if args.potential == 'point_charge':
    def VH2(x, origin1, origin2, Z1, Z2):
        V1 = nucpot.point_charge(x, origin1, Z1)
        V2 = nucpot.point_charge(x, origin2, Z2)
        return V1 + V2
    f = lambda x: VH2(x, origin1, origin2, Z, Z)
elif args.potential == 'coulomb_HFYGB':
    def VH2(x, origin1, origin2, Z1, Z2, prec):
        V1 = nucpot.coulomb_HFYGB(x, origin1, Z1, prec)
        V2 = nucpot.coulomb_HFYGB(x, origin2, Z2, prec)
        return V1 + V2
    f = lambda x: VH2(x, origin1, origin2, Z, Z, prec)
elif args.potential == 'homogeneus_charge_sphere':
    def VH2(x, origin1, origin2, Z1, Z2, atom):
        V1 = nucpot.homogeneus_charge_sphere(x, origin1, Z1, atom)
        V2 = nucpot.homogeneus_charge_sphere(x, origin2, Z2, atom)
        return V1 + V2
    f = lambda x: VH2(x, origin1, origin2, Z, Z, atom)
elif args.potential == 'gaussian':
    def VH2(x, origin1, origin2, Z1, Z2):
        V1 = nucpot.gaussian(x, origin1, Z1, atom)
        V2 = nucpot.gaussian(x, origin2, Z2, atom)
        return V1 + V2
    f = lambda x: VH2(x, origin1, origin2, Z, Z, atom)

Peps = vp.ScalingProjector(mra,prec/10)
V_tree = Peps(f)
print('V_tree', V_tree)
print('Define V Potential', args.potential, 'DONE')

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


defult_der = 'BS'

error_norm = 1
c2 = light_speed * light_speed

if args.dtype == 'dirac':
    while error_norm > prec:
        hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = defult_der)
        v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
        add_psi = hd_psi + v_psi
        energy = spinor_H.dot(add_psi).real
        print('Energy =',energy - light_speed**2)
        mu = orb.calc_dirac_mu(energy, light_speed)
        tmp = orb.apply_helmholtz(v_psi, mu, prec)
        tmp.crop(prec/10)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = defult_der)
        new_orbital.crop(prec/10)
        new_orbital.normalize()
        delta_psi = new_orbital - spinor_H
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Error =', error_norm)
        spinor_H = new_orbital
        
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der = defult_der)
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
    add_psi = hd_psi + v_psi
    energy = spinor_H.dot(add_psi).real
    print('Final Energy =',energy - light_speed**2)

elif args.dtype == 'dirac2':
    while error_norm > prec:
        v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec) 
        vv_psi = orb.apply_potential(-0.5/c2, V_tree, v_psi, prec)
        beta_v_psi = v_psi.beta2()
        apV_psi = v_psi.alpha_p(prec, defult_der)
        ap_psi = spinor_H.alpha_p(prec, defult_der)
        Vap_psi = orb.apply_potential(-1.0, V_tree, ap_psi, prec)
        anticom = apV_psi + Vap_psi
        RHS = beta_v_psi + vv_psi + anticom * (0.5/light_speed)
        cke = spinor_H.classicT()
        cpe = (spinor_H.dot(RHS)).real
        print("Classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
        mu = orb.calc_non_rel_mu(cke+cpe)
        print("mu =", mu)
        new_orbital = orb.apply_helmholtz(RHS, mu, prec)
        new_orbital.normalize()
        delta_psi = new_orbital - spinor_H
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print("Error =", error_norm)
        spinor_H = new_orbital
    
    hd_psi = orb.apply_dirac_hamiltonian(spinor_H, prec, der)
    v_psi = orb.apply_potential(-1.0, V_tree, spinor_H, prec)
    add_psi = hd_psi + v_psi
    energy = spinor_H.dot(add_psi).real
    
    cke = spinor_H.classicT()
    beta_v_psi = v_psi.beta2()
    beta_pot = (beta_v_psi.dot(spinor_H)).real
    pot_sq  = (v_psi.dot(v_psi)).real
    ap_psi = spinor_H.alpha_p(prec, der)
    anticom = (ap_psi.dot(v_psi)).real
    energy_kutzelnigg = cke + beta_pot + pot_sq/(2*mc2) + anticom/light_speed
    
    print('Kutzelnigg =',cke, beta_pot, pot_sq/(2*c2), anticom/light_speed, energy_kutzelnigg)
    print('Quadratic approx =',energy_kutzelnigg - energy_kutzelnigg**2/(2*c2))
    print('Correct from Kutzelnigg =', c2*(np.sqrt(1+2*energy_kutzelnigg/c2)-1))
    print('Final Energy =',energy - light_speed**2)
    
    energy_1s = analytic_1s(light_speed, n, k, Z)
    
    print('Exact Energy =',energy_1s - light_speed**2)
    print('Difference 1 =',energy_1s - energy)
    print('Difference 2 =',energy_1s - energy_kutzelnigg - light_speed**2)
