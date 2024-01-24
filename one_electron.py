from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import operators as oper
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from vampyr import vampyr3d as vp
import numpy as np
import numpy.linalg as LA
import sys, getopt

def analytic_1s(light_speed, n, k, Z):
    alpha = 1/light_speed
    gamma = orb.compute_gamma(k,Z,alpha)
    tmp1 = n - np.abs(k) + gamma
    tmp2 = Z * alpha / tmp1
    tmp3 = 1 + tmp2**2
    return light_speed**2 / np.sqrt(tmp3)

def gs_D_1e(spinorb1, potential, mra, prec, thr, derivative):
    print('One-electron calculations')
    
    error_norm = 1
    #compute_last_energy = False

    light_speed = spinorb1.light_speed
    old_energy = 0
    delta_e = 1
    while (error_norm > thr and delta_e > thr/1000):
        hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
        add_psi = hd_psi + v_psi
        energy = spinorb1.dot(add_psi).real
        mu = orb.calc_dirac_mu(energy, light_speed)
        tmp = orb.apply_helmholtz(v_psi, mu, prec)
#        tmp = orb.apply_dirac_hamiltonian(v_psi, prec, energy, der = derivative)
        tmp.cropLargeSmall(prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = derivative)
#        new_orbital =  orb.apply_helmholtz(tmp, mu, prec)
#        new_orbital.cropLargeSmall(prec)
        new_orbital.cropLargeSmall(prec)
        new_orbital.normalize()
        delta_psi = new_orbital - spinorb1
        #orbital_error = delta_psi.dot(delta_psi).real
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Error', error_norm)
        delta_e = np.abs(energy - old_energy)
        print('Delta E', delta_e)
        print('Energy',energy - light_speed**2)
        old_energy = energy
        spinorb1 = new_orbital
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, 1)
    print("Exact energy: ", energy_1s - light_speed**2)
    print('Final Energy:',energy - light_speed**2)
    print('Delta Energy:',energy - old_energy)
    print('Error Energy:',energy - energy_1s)
    return spinorb1


def gs_D2_1e(spinorb1, potential, mra, prec, thr, derivative):
    print('Hartree-Fock 1e D2')
    error_norm = 1
    delta_e = 1
    light_speed = spinorb1.light_speed
    c2 = light_speed * light_speed
    old_energy = 0
    while (error_norm > thr  and delta_e > thr/1000):
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec) 
        vv_psi = orb.apply_potential(-0.5/c2, potential, v_psi, prec*c2)
        beta_v_psi = v_psi.beta2()
        apV_psi = v_psi.alpha_p(prec, derivative)
        ap_psi = spinorb1.alpha_p(prec*light_speed, derivative)
        Vap_psi = orb.apply_potential(-1.0, potential, ap_psi, prec*light_speed)
        anticom = apV_psi + Vap_psi
#        anticom.cropLargeSmall(prec)
#        beta_v_psi.cropLargeSmall(prec)
#        vv_psi.cropLargeSmall(prec)
        RHS = beta_v_psi + vv_psi + anticom * (0.5/light_speed)
        RHS.cropLargeSmall(prec)
        cke = spinorb1.classicT()
        cpe = (spinorb1.dot(RHS)).real
        print("Classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
        classic_energy = cke + cpe
        energy = c2*(np.sqrt(1+2*classic_energy/c2)-1)
        mu = orb.calc_non_rel_mu(cke+cpe)
        print("mu =", mu)
        new_orbital = orb.apply_helmholtz(RHS, mu, prec)
        new_orbital.cropLargeSmall(prec)
        new_orbital.normalize()
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print("Error =", error_norm)
        delta_e = np.abs(energy - old_energy)
        print('Delta E', delta_e)
        print('Energy',energy, old_energy)
        old_energy = energy
        spinorb1 = new_orbital
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy_dirac = spinorb1.dot(add_psi).real
    
#    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec) 
    vv_psi = orb.apply_potential(-0.5/c2, potential, v_psi, prec*c2)
    beta_v_psi = v_psi.beta2()
    apV_psi = v_psi.alpha_p(prec*light_speed, derivative)
    ap_psi = spinorb1.alpha_p(prec*light_speed, derivative)
    Vap_psi = orb.apply_potential(-1.0, potential, ap_psi, prec)
    anticom = apV_psi + Vap_psi
#    anticom.cropLargeSmall(prec)
#    beta_v_psi.cropLargeSmall(prec)
#    vv_psi.cropLargeSmall(prec)
    RHS = beta_v_psi + vv_psi + anticom * (0.5/light_speed)
    RHS.cropLargeSmall(prec)
    cke = spinorb1.classicT()
    cpe = (spinorb1.dot(RHS)).real
    classic_energy = cke + cpe
    energy = c2*(np.sqrt(1+2*classic_energy/c2)-1)
    energy_1s = analytic_1s(light_speed, 1, -1, 1)
    
#    print('Kutzelnigg =',cke, cpe, energy_kutzelnigg)
#    print('Quadratic approx =',energy_kutzelnigg - energy_kutzelnigg**2/(2*c2))
    print('Exact Energy = ', energy_1s - light_speed**2)
    print('Dirac Energy = ', energy_dirac - light_speed**2)
    print('Kutze Energy = ', energy)
    print('Error Kutze  = ', energy - energy_1s + light_speed**2)
    print('Error Dirac  = ', energy_dirac - energy_1s)
    print('Delta Energy = ', energy - old_energy)
    print('Dirac - Kutzelnigg = ', energy_dirac - energy - light_speed**2)   
    return spinorb1
