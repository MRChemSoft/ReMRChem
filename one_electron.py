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

def gs_D_1e(spinorb1, potential, mra, prec, derivative):
    print('Hartree-Fock 1e')
    
    error_norm = 1
    #compute_last_energy = False

    light_speed = spinorb1.light_speed

    while error_norm > prec:
        hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
        add_psi = hd_psi + v_psi
        energy = spinorb1.dot(add_psi).real
        print('Energy',energy - light_speed**2)
        mu = orb.calc_dirac_mu(energy, light_speed)
        tmp = orb.apply_helmholtz(v_psi, mu, prec)
        tmp.crop(prec/10)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = derivative)
        new_orbital.crop(prec/10)
        new_orbital.normalize()
        delta_psi = new_orbital - spinorb1
        #orbital_error = delta_psi.dot(delta_psi).real
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Error', error_norm)
        spinorb1 = new_orbital
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    print('Final Energy',energy - light_speed**2)
    return spinorb1


def gs_D2_1e(spinorb1, potential, mra, prec, derivative):
    print('Hartree-Fock 1e D2')

    error_norm = 1.0
    light_speed = spinorb1.light_speed
    mc2 = light_speed**2
    P = vp.PoissonOperator(mra, prec)
    Vop = oper.PotentialOperator(mra, prec, potential)

    while error_norm > prec:
#        Vpsi = orb.apply_potential(-1.0, potential, spinorb1, prec)
#        VVpsi = orb.apply_potential(-0.5/mc2, potential, Vpsi, prec)
#        beta_Vpsi = Vpsi.beta2()
#        apV_psi = Vpsi.alpha_p(prec, derivative)
#        ap_psi = spinorb1.alpha_p(prec, derivative)
#        Vap_psi = orb.apply_potential(-1.0, potential, ap_psi, prec)
#        anticom = apV_psi + Vap_psi        
        RHS = build_RHS_D2_1e(Vop, spinorb1, prec, light_speed)
        cke = spinorb1.classicT()
        cpe = (spinorb1.dot(RHS)).real
        
#        VVpsi.cropLargeSmall(prec)
#        beta_Vpsi.cropLargeSmall(prec)
#        anticom.cropLargeSmall(prec)
#        RHS.cropLargeSmall(prec)

        print("Classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
        #print("Orbital energy =", c2 * ( -1.0 + np.sqrt(1 + 2 * (cpe + cke) / c2)))
        
        mu = orb.calc_non_rel_mu(cke+cpe)
        print("mu =", mu)
        
        new_spinorb1 = orb.apply_helmholtz(RHS, mu, prec)
        #print("normalization")
        new_spinorb1.normalize()
        #print("crop")
#        new_spinorb1.cropLargeSmall(prec)

        # Compute orbital error
        delta_psi = new_spinorb1 - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print("Error =", error_norm)
        spinorb1 = new_spinorb1

    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    Vpsi = orb.apply_potential(-1.0, potential, spinor, prec)
    add_psi = hd_psi + Vpsi
    energy = (spinor.dot(add_psi)).real

    cke = spinorb1.classicT()
    beta_Vpsi = Vpsi.beta2()
    beta_pot = (beta_Vpsi.dot(spinorb1)).real
    pot_sq  = (Vpsi.dot(Vpsi)).real
    ap_psi = spinorb1.alpha_p(prec, der)
    anticom = (ap_psi.dot(Vpsi)).real
    energy_kutzelnigg = cke + beta_pot + pot_sq/(2*mc2) + anticom/light_speed

    print('Kutzelnigg',cke, beta_pot, pot_sq/(2*mc2), anticom/light_speed, energy_kutzelnigg)
    print('Quadratic approx',energy_kutzelnigg - energy_kutzelnigg**2/(2*mc2))
    print('Correct from Kutzelnigg', mc2*(np.sqrt(1+2*energy_kutzelnigg/mc2)-1))
    print('Final Energy',energy - light_speed**2)

    energy_1s = analytic_1s(light_speed, n, k, Z)

    print('Exact Energy',energy_1s - light_speed**2)
    print('Difference 1',energy_1s - energy)
    print('Difference 2',energy_1s - energy_kutzelnigg - light_speed**2)
    return spinorb1

def build_RHS_D2_1e(Vop, spinor, prec, light_speed):
    c2 = light_speed**2
    Vpsi = Vop(spinor)
    VT_psi = -1.0 *  Vpsi

    beta_VT_psi = VT_psi.beta2()
#    beta_VT_psi.cropLargeSmall(prec)

    ap_VT_psi = VT_psi.alpha_p(prec)
    ap_psi = spinor.alpha_p(prec)
    VT_ap_psi =  -1.0 *  Vop(ap_psi)
    anticom = VT_ap_psi + ap_VT_psi
    anticom *= 1.0 / (2.0 * light_speed)
#    anticom.cropLargeSmall(prec)

    VT_VT_psi = -1.0 * Vop(VT_psi)
    VT_VT_psi *= 1.0 / (2.0 * c2)
#    VT_VT_psi.cropLargeSmall(prec)

    RHS = beta_VT_psi + anticom + VT_VT_psi
#    RHS.cropLargeSmall(prec)
    return RHS 
