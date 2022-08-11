########## Define Enviroment #################
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import NuclearPotential as nucpot
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp

import argparse
import numpy as np
import numpy.linalg as LA
import sys, getopt

import importlib
importlib.reload(orb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-a', '--atype', dest='atype', type=str,
                        help='put the atom type')
    parser.add_argument('-z', '--charge', dest='charge', type=float,
                        help='put the atom charge')
    parser.add_argument('-b', '--box', dest='box', type=int, default=30,
                        help='put the box dimension')
    parser.add_argument('-o', '--order', dest='order', type=int, default=8,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-5,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str,
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='PoCh',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, fermi_two_parameters_charge_distribution, gaussian_charge_distribution')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electran'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt'], 'Please, specify coulgau in a rigth way – coloumb or gaunt'

    assert args.potential in ['point_charge', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'fermi_two_parameters_charge_distribution', 'gaussian_charge_distribution'], 'Please, specify V'

################# Define Paramters ###########################
light_speed = 137.03604 
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = args.charge
atom = args.atype
################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order)
prec = args.prec
origin = [0.1, 0.2, 0.3]  # origin moved to avoid placing the nuclar charge on a node
print('call MRA DONE')

################# Define Gaussian function ########## 
a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
gauss_tree = vp.FunctionTree(mra)
vp.advanced.build_grid(out=gauss_tree, inp=gauss)
vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
gauss_tree.normalize()
print('Define Gaussian Function DONE')

################ Define orbital as complex function ######################
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
complexfc = cf.complex_fcn()
complexfc.copy_fcns(real=gauss_tree)
print('Define orbital as a complex function DONE')

################ Define spinorbitals ########## 
spinorb1 = orb.orbital4c()
spinorb1.copy_components(La=complexfc)
spinorb1.init_small_components(prec/10)
spinorb1.normalize()

spinorb2 = orb.orbital4c()
spinorb2.copy_components(Lb=complexfc)
spinorb2.init_small_components(prec/10)
spinorb2.normalize()
print('Define spinorbitals DONE')

################### Define V potential ######################
if args.potential == 'point_charge':
    Peps = vp.ScalingProjector(mra,prec)
    f = lambda x: nucpot.point_charge(x, origin, Z)
    V_tree = Peps(f)
elif args.potential == 'coulomb_HFYGB':
    Peps = vp.ScalingProjector(mra,prec)
    f = lambda x: nucpot.coulomb_HFYGB(x, origin, Z, prec)
    V_tree = Peps(f)
elif args.potential == 'homogeneus_charge_sphere':
    Peps = vp.ScalingProjector(mra,prec)
    f = lambda x: nucpot.homogeneus_charge_sphere(x, origin, Z, atom)
    V_tree = Peps(f)
elif args.potential == 'fermi_two_parameters_charge_distribution':
    Peps = vp.ScalingProjector(mra,prec)
    Pua = vp.PoissonOperator(mra, prec)
    f = lambda x: nucpot.fermi_two_parameters_charge_distribution(x, origin, Z, atom)
    rho_tree = Peps(f)
    V_tree = Pua(rho_tree) * (4 * np.pi)
elif args.potential == 'gaussian_charge_distribution':
    print("FIXME!")
    exit(-1)
    Peps = vp.ScalingProjector(mra,prec)
    f = lambda x: nucpot.gaussian_charge_distribution(x, origin, Z, atom)
    V_tree = Peps(f)

default_der = 'PH'
print('Define V Potential', args.potential, 'DONE')


#############################START WITH CALCULATION###################################
if args.coulgau == 'coulomb':
    print('Hartræ-Føck (Cøulømbic bielectric interåctiøn)')
    error_norm = 1

    while error_norm > prec:
        # Definition of different densities
        n_11 = spinorb1.density(prec)
        n_12 = spinorb1.exchange(spinorb2, prec)
        n_21 = spinorb2.exchange(spinorb1, prec)
        n_22 = spinorb2.density(prec)


        # Definition of Poisson operator
        Pua = vp.PoissonOperator(mra, prec)

        # Defintion of Jx
        J11 = Pua(n_11) * (4 * np.pi)
        J12 = Pua(n_12) * (4 * np.pi)
        J21 = Pua(n_21) * (4 * np.pi)
        J22 = Pua(n_22) * (4 * np.pi)


        # Definition of Kx
        K1 = Pua(n_12) * (4 * np.pi)
        K2 = Pua(n_21) * (4 * np.pi)


        # Definition of Energy Hartree of Fock matrix
        E_H11 = vp.dot(n_11, J11)
        E_H12 = vp.dot(n_12, J12)
        E_H21 = vp.dot(n_21, J21)
        E_H22 = vp.dot(n_22, J22)


        # Definition of Energy Exchange of Fock matrix
        E_xc11 = vp.dot(n_12, K2)
        E_xc12 = vp.dot(n_12, K1)
        E_xc21 = vp.dot(n_21, K2)
        E_xc22 = vp.dot(n_21, K1)


        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)


        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + v_psi_1
        add_psi_2 = hd_psi_2 + v_psi_2


        # Calculate Fij Fock matrix
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_21, imag_21 = spinorb2.dot(add_psi_1)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)


        # Orbital Energy calculation
        energy_11 = energy_11 + E_H11 - E_xc11
        energy_12 = energy_12 + E_H12 - E_xc12
        energy_21 = energy_21 + E_H21 - E_xc21
        energy_22 = energy_22 + E_H22 - E_xc22
        print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


        # Total Energy with J = K approximation
        E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22)
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))


        # Calculation of necessary potential contributions to Hellmotz
        J_spinorb1   = orb.apply_potential(1.0, J11, spinorb1, prec)
        K_spinorb1   = orb.apply_potential(1.0, K2, spinorb2, prec)
        F12_spinorb2 =  energy_12 * spinorb2


        J_spinorb2   = orb.apply_potential(1.0, J22, spinorb2, prec)
        K_spinorb2   = orb.apply_potential(1.0, K1, spinorb1, prec)
        F21_spinorb1 = energy_21 * spinorb1


        V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - F12_spinorb2
        V_J_K_spinorb2 = v_psi_2 + J_spinorb2 - K_spinorb2 - F21_spinorb1


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, energy_11, prec)
        tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, energy_22, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_11, der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_22, der = default_der)
        new_orbital_2 *= 0.5/light_speed**2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)
        

        # Compute overlap
        dot_11 = new_orbital_1.dot(new_orbital_1)
        dot_12 = new_orbital_1.dot(new_orbital_2)
        dot_21 = new_orbital_2.dot(new_orbital_1)
        dot_22 = new_orbital_2.dot(new_orbital_2)
        s_11 = dot_11[0] + 1j * dot_11[1]
        s_12 = dot_12[0] + 1j * dot_12[1]
        s_21 = dot_21[0] + 1j * dot_21[1]
        s_22 = dot_22[0] + 1j * dot_22[1]


        # Compute Overlap Matrix
        S_tilde = np.array([[s_11, s_12], [s_21, s_22]])


        # Compute U matrix
        sigma, U = LA.eig(S_tilde)


        # Compute matrix S^-1/2
        Sm5 = U @ np.diag(sigma ** (-0.5)) @ U.transpose()


        # Compute the new orthogonalized orbitals
        spinorb1 = Sm5[0, 0] * new_orbital_1 + Sm5[0, 1] * new_orbital_2
        spinorb2 = Sm5[1, 0] * new_orbital_1 + Sm5[1, 1] * new_orbital_2
        spinorb1.crop(prec)
        spinorb2.crop(prec)


    ##########

    # Definition of different densities
    n_11 = spinorb1.density(prec)
    n_12 = spinorb1.exchange(spinorb2, prec)
    n_21 = spinorb2.exchange(spinorb1, prec)
    n_22 = spinorb2.density(prec)


    # Definition of Poisson operator
    Pua = vp.PoissonOperator(mra, prec)


    # Defintion of Jx
    J11 = Pua(n_11) * (4 * np.pi)
    J12 = Pua(n_12) * (4 * np.pi)
    J21 = Pua(n_21) * (4 * np.pi)
    J22 = Pua(n_22) * (4 * np.pi)


    # Definition of Kx
    K1 = Pua(n_12) * (4 * np.pi)
    K2 = Pua(n_21) * (4 * np.pi)
    

    # Definition of Energy Hartree of Fock matrix
    E_H11 = vp.dot(n_11, J11)
    E_H12 = vp.dot(n_12, J12)
    E_H21 = vp.dot(n_21, J21)
    E_H22 = vp.dot(n_22, J22)
    

    # Definition of Energy Exchange of Fock matrix
    E_xc11 = vp.dot(n_12, K2)
    E_xc12 = vp.dot(n_12, K1)
    E_xc21 = vp.dot(n_21, K2)
    E_xc22 = vp.dot(n_21, K1)


    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)


    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2
    

    # Calculate Fij Fock matrix
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_12, imag_12 = spinorb1.dot(add_psi_2)
    energy_21, imag_21 = spinorb2.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)
    

    # Orbital Energy calculation
    energy_11 = energy_11 + E_H11 - E_xc11
    energy_12 = energy_12 + E_H12 - E_xc12
    energy_21 = energy_21 + E_H21 - E_xc21
    energy_22 = energy_22 + E_H22 - E_xc22
    print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


    # Total Energy with J = K approximation
    E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22)
    print('E_total(Coulomb) approximiation', E_tot_JK - 2.0 *(light_speed**2))


#########################################################END###########################################################################
elif args.coulgau == 'gaunt':
    print('Hartræ-Føck (Cøulømbic-Gåunt bielectric interåctiøn)')
    error_norm = 1
    while error_norm > prec:

        # Definition of different densities
        n_11 = spinorb1.density(prec)
        n_12 = spinorb1.exchange(spinorb2, prec)
        n_21 = spinorb2.exchange(spinorb1, prec)
        n_22 = spinorb2.density(prec)


        # Definition of Poisson operator
        Pua = vp.PoissonOperator(mra, prec)


        # Defintion of Jx
        J11 = Pua(n_11) * (4 * np.pi)
        J12 = Pua(n_12) * (4 * np.pi)
        J21 = Pua(n_21) * (4 * np.pi)
        J22 = Pua(n_22) * (4 * np.pi)


        # Definition of Kx
        K1 = Pua(n_12) * (4 * np.pi)
        K2 = Pua(n_21) * (4 * np.pi)


        # Definition of Energy Hartree of Fock matrix
        E_H11 = vp.dot(n_11, J11)
        E_H12 = vp.dot(n_12, J12)
        E_H21 = vp.dot(n_21, J21)
        E_H22 = vp.dot(n_22, J22)


        # Definition of Energy Exchange of Fock matrix
        E_xc11 = vp.dot(n_12, K2)
        E_xc12 = vp.dot(n_12, K1)
        E_xc21 = vp.dot(n_21, K2)
        E_xc22 = vp.dot(n_21, K1)


        # Gaunt: Direct (GJ) and Exchange (GK)
        # Definition of alpha(orbital)
        alpha_1 = orb.alpha(spinorb1, 0) + orb.alpha(spinorb1, 1) + orb.alpha(spinorb1, 2)
        alpha_2 = orb.alpha(spinorb1, 0) + orb.alpha(spinorb1, 1) + orb.alpha(spinorb1, 2)
        

        # Definition of orbital * alpha(orbital)
        spinorb1_alpha1 = spinorb1.exchange(alpha_1, prec)
        spinorb2_alpha1 = spinorb2.exchange(alpha_1, prec)
        spinorb1_alpha2 = spinorb1.exchange(alpha_2, prec)
        spinorb2_alpha2 = spinorb2.exchange(alpha_2, prec)


        # Defintion of GJx
        GJ11 = Pua(spinorb1_alpha1) * (4 * np.pi)
        GJ12 = Pua(spinorb1_alpha2) * (4 * np.pi)
        GJ21 = Pua(spinorb2_alpha1) * (4 * np.pi)
        GJ22 = Pua(spinorb2_alpha2) * (4 * np.pi)


        # Definition of GJ energy term for each spin orbital
        E_GJ11 = vp.dot(spinorb2_alpha2, GJ11)
        E_GJ12 = vp.dot(spinorb2_alpha2, GJ12)
        E_GJ21 = vp.dot(spinorb2_alpha1, GJ21)
        E_GJ22 = vp.dot(spinorb1_alpha1, GJ22)


        # Defintion of GKx
        GK1 = Pua(spinorb1_alpha2) * (4 * np.pi)
        GK2 = Pua(spinorb2_alpha1) * (4 * np.pi)


        # Definition of GK energy term for each spin orbital
        E_Gxc11 = vp.dot(spinorb1_alpha2, GK2)
        E_Gxc12 = vp.dot(spinorb1_alpha2, GK1)
        E_Gxc21 = vp.dot(spinorb2_alpha1, GK2)
        E_Gxc22 = vp.dot(spinorb2_alpha1, GK1)


        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0)


        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
        

        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + v_psi_1
        add_psi_2 = hd_psi_2 + v_psi_2


        # Calculate Fij Fock matrix
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_21, imag_21 = spinorb2.dot(add_psi_1)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)


        # Orbital Energy calculation
        energy_11 = energy_11 + E_H11 - E_xc11 - E_GJ11 + E_Gxc11
        energy_12 = energy_12 + E_H12 - E_xc12 - E_GJ12 + E_Gxc12
        energy_21 = energy_21 + E_H21 - E_xc21 - E_GJ21 + E_Gxc21
        energy_22 = energy_22 + E_H22 - E_xc22 - E_GJ22 + E_Gxc22
        print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


        # Total Energy with J = K approximation
        E_tot = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_GJ11 - E_GJ22 + E_Gxc11 + E_Gxc22)
        print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2))

        # Calculation of necessary potential contributions to Hellmotz
        CJ_spinorb1 = orb.apply_potential(1.0, J11, spinorb1, prec)
        GJ_spinorb1 = orb.apply_potential(1.0, GJ11, spinorb1, prec)
        CK_spinorb1 = orb.apply_potential(1.0, K2, spinorb2, prec)
        GK_spinorb1 = orb.apply_potential(1.0, GK2, spinorb2, prec)
        F12_spinorb2 = energy_12 * spinorb2


        CJ_spinorb2 = orb.apply_potential(1.0, J22, spinorb2, prec)
        GJ_spinorb2 = orb.apply_potential(1.0, GJ22, spinorb2, prec)
        CK_spinorb2 = orb.apply_potential(1.0, K1, spinorb1, prec)
        GK_spinorb2 = orb.apply_potential(1.0, GK1, spinorb1, prec)
        F21_spinorb1 = energy_21 * spinorb1


        V_J_K_spinorb1 = v_psi_1 + CJ_spinorb1 - CK_spinorb1 - GJ_spinorb1 + GK_spinorb1 - F12_spinorb2
        V_J_K_spinorb2 = v_psi_2 + CJ_spinorb2 - CK_spinorb2 - GJ_spinorb2 + GK_spinorb2 - F21_spinorb1


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, energy_11, prec)
        tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, energy_22, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_11)
        new_orbital_1 *= 0.5 / light_speed ** 2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_22)
        new_orbital_2 *= 0.5 / light_speed ** 2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)


        # Compute overlap
        dot_11 = new_orbital_1.dot(new_orbital_1)
        dot_12 = new_orbital_1.dot(new_orbital_2)
        dot_21 = new_orbital_2.dot(new_orbital_1)
        dot_22 = new_orbital_2.dot(new_orbital_2)
        s_11 = dot_11[0] + 1j * dot_11[1]
        s_12 = dot_12[0] + 1j * dot_12[1]
        s_21 = dot_21[0] + 1j * dot_21[1]
        s_22 = dot_22[0] + 1j * dot_22[1]
        

        # Compute Overlap Matrix
        S_tilde = np.array([[s_11, s_12], [s_21, s_22]])


        # Compute U matrix
        sigma, U = LA.eig(S_tilde)


        # Compute matrix S^-1/2
        Sm5 = U @ np.diag(sigma ** (-0.5)) @ U.transpose()


        # Compute the new orthogonalized orbitals
        spinorb1 = Sm5[0, 0] * new_orbital_1 + Sm5[0, 1] * new_orbital_2
        spinorb2 = Sm5[1, 0] * new_orbital_1 + Sm5[1, 1] * new_orbital_2
        spinorb1.crop(prec)
        spinorb2.crop(prec)

    ##########
    # Definition of different densities
    n_11 = spinorb1.density(prec)
    n_12 = spinorb1.exchange(spinorb2, prec)
    n_21 = spinorb2.exchange(spinorb1, prec)
    n_22 = spinorb2.density(prec)


    # Definition of Poisson operator
    Pua = vp.PoissonOperator(mra, prec)


    # Defintion of Jx
    J11 = Pua(n_11) * (4 * np.pi)
    J12 = Pua(n_12) * (4 * np.pi)
    J21 = Pua(n_21) * (4 * np.pi)
    J22 = Pua(n_22) * (4 * np.pi)


    # Definition of Kx
    K1 = Pua(n_12) * (4 * np.pi)
    K2 = Pua(n_21) * (4 * np.pi)


    # Definition of Energy Hartree of Fock matrix
    E_H11 = vp.dot(n_11, J11)
    E_H12 = vp.dot(n_12, J12)
    E_H21 = vp.dot(n_21, J21)
    E_H22 = vp.dot(n_22, J22)


    # Definition of Energy Exchange of Fock matrix
    E_xc11 = vp.dot(n_12, K2)
    E_xc12 = vp.dot(n_12, K1)
    E_xc21 = vp.dot(n_21, K2)
    E_xc22 = vp.dot(n_21, K1)


    # Gaunt: Direct (GJ) and Exchange (GK)
    # Definition of alpha(orbital)
    alpha_1 = orb.alpha(spinorb1, 0) + orb.alpha(spinorb1, 1) + orb.alpha(spinorb1, 2)
    alpha_2 = orb.alpha(spinorb1, 0) + orb.alpha(spinorb1, 1) + orb.alpha(spinorb1, 2)


    # Definition of orbital * alpha(orbital)
    spinorb1_alpha1 = spinorb1.exchange(alpha_1, prec)
    spinorb2_alpha1 = spinorb2.exchange(alpha_1, prec)
    spinorb1_alpha2 = spinorb1.exchange(alpha_2, prec)
    spinorb2_alpha2 = spinorb2.exchange(alpha_2, prec)


    # Defintion of GJx
    GJ11 = Pua(spinorb1_alpha1) * (4 * np.pi)
    GJ12 = Pua(spinorb1_alpha2) * (4 * np.pi)
    GJ21 = Pua(spinorb2_alpha1) * (4 * np.pi)
    GJ22 = Pua(spinorb2_alpha2) * (4 * np.pi)


    # Definition of GJ energy term for each spin orbital
    E_GJ11 = vp.dot(spinorb2_alpha2, GJ11)
    E_GJ12 = vp.dot(spinorb2_alpha2, GJ12)
    E_GJ21 = vp.dot(spinorb2_alpha1, GJ21)
    E_GJ22 = vp.dot(spinorb1_alpha1, GJ22)


    # Defintion of GKx
    GK1 = Pua(spinorb1_alpha2) * (4 * np.pi)
    GK2 = Pua(spinorb2_alpha1) * (4 * np.pi)


    # Definition of GK energy term for each spin orbital
    E_Gxc11 = vp.dot(spinorb1_alpha2, GK2)
    E_Gxc12 = vp.dot(spinorb1_alpha2, GK1)
    E_Gxc21 = vp.dot(spinorb2_alpha1, GK2)
    E_Gxc22 = vp.dot(spinorb2_alpha1, GK1)


    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0)


    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2


    # Calculate Fij Fock matrix
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_12, imag_12 = spinorb1.dot(add_psi_2)
    energy_21, imag_21 = spinorb2.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)


    # Orbital Energy calculation
    energy_11 = energy_11 + E_H11 - E_xc11 - E_GJ11 + E_Gxc11
    energy_12 = energy_12 + E_H12 - E_xc12 - E_GJ12 + E_Gxc12
    energy_21 = energy_21 + E_H21 - E_xc21 - E_GJ21 + E_Gxc21
    energy_22 = energy_22 + E_H22 - E_xc22 - E_GJ22 + E_Gxc22
    print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


    # Total Energy with J = K approximation
    E_tot = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_GJ11 - E_GJ22 + E_Gxc11 + E_Gxc22)
    print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2))
#########################################################END######################################################
