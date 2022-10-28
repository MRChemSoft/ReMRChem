########## Define Enviroment #################
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import nuclear_potential as nucpot
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
    parser.add_argument('-a', '--atype', dest='atype', type=str, default='He',
                        help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='PH',
                        help='put the type of derivative')
    parser.add_argument('-z', '--charge', dest='charge', type=float, default=2.0,
                        help='put the atom charge')
    parser.add_argument('-b', '--box', dest='box', type=int, default=60,
                        help='put the box dimension')
    parser.add_argument('-o', '--order', dest='order', type=int, default=8,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-5,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, fermi_dirac, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electran'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt'], 'Please, specify coulgau in a rigth way – coulomb or gaunt'

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'fermi_dirac', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS'], 'Please, specify the type of derivative'

################# Define Paramters ###########################
light_speed = 137.03599913900001
#light_speed = 2000000
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
#origin = [0.0, 0.0, 0.0]
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
cspinorb1 = spinorb1.complex_conj()
#print('spinorb1',spinorb1)
#print('cspinorb1',cspinorb1)

spinorb2 = orb.orbital4c()
spinorb2.copy_components(Lb=complexfc)
spinorb2.init_small_components(prec/10)
spinorb2.normalize()
cspinorb2 = spinorb2.complex_conj()
#print('spinorb2',spinorb2)
#print('cspinorb2',cspinorb2)

print('Define spinorbitals DONE')

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
elif args.potential == 'fermi_dirac':
   Peps = vp.ScalingProjector(mra,prec/10)
   Pua = vp.PoissonOperator(mra, prec/10)
   f = lambda x: nucpot.fermi_dirac(x, origin, Z, atom)
   rho_tree = Peps(f)
   V_tree = Pua(rho_tree) * (4 * np.pi)
elif args.potential == 'gaussian':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.gaussian(x, origin, Z, atom)
   V_tree = Peps(f)

default_der = args.deriv
print('Define V Potetintal', args.potential, 'DONE')

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
        J1 = Pua(n_11) * (4 * np.pi)
        J2 = Pua(n_22) * (4 * np.pi)


        # Definition of Kx
        K1 = Pua(n_12) * (4 * np.pi)
        K2 = Pua(n_21) * (4 * np.pi)


        # Definition of Energy Hartree of Fock matrix
        E_H1 = vp.dot(n_11, J1)
        E_H2 = vp.dot(n_22, J2)


        # Definition of Energy Exchange of Fock matrix
        E_xc1 = vp.dot(n_12, K2)
        E_xc2 = vp.dot(n_21, K1)


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
        energy_11 = energy_11 + E_H1 - E_xc1
        energy_22 = energy_22 + E_H2 - E_xc2
        print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


        # Total Energy with J = K approximation
        E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H1 + E_H2 - E_xc1 - E_xc2)
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))


        # Calculation of necessary potential contributions to Hellmotz
        J_spinorb1   = orb.apply_potential(1.0, J1, spinorb1, prec)
        K_spinorb1   = orb.apply_potential(1.0, K2, spinorb2, prec)
        F12_spinorb2 =  energy_12 * spinorb2


        J_spinorb2   = orb.apply_potential(1.0, J2, spinorb2, prec)
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
    J1 = Pua(n_11) * (4 * np.pi)
    J2 = Pua(n_22) * (4 * np.pi)


    # Definition of Kx
    K1 = Pua(n_12) * (4 * np.pi)
    K2 = Pua(n_21) * (4 * np.pi)


    # Definition of Energy Hartree of Fock matrix
    E_H1 = vp.dot(n_11, J1)
    E_H2 = vp.dot(n_22, J2)


    # Definition of Energy Exchange of Fock matrix
    E_xc1 = vp.dot(n_12, K2)
    E_xc2 = vp.dot(n_21, K1)


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
    energy_11 = energy_11 + E_H1 - E_xc1
    energy_22 = energy_22 + E_H2 - E_xc2
    print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)


    # Total Energy with J = K approximation
    E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H1 + E_H2 - E_xc1 - E_xc2)
    print('E_total(Coulomb) approximiation', E_tot_JK - 2.0 *(light_speed**2))

    #x = np.arange(-59.999, 59.999, 0.001)
    #y = [f_tree([x,  59.999,  59.999]) for x in x]

    #plt.plt(x, y)

#####################################################END COULOMB & START GAUNT#######################################################################
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


        #GAUNT: Direct (GJ) and Exchange (GK)
        #Definition of alpha(orbital)
        alpha_10 =  spinorb1.alpha(0)
        alpha_11 =  spinorb1.alpha(1)
        alpha_12 =  spinorb1.alpha(2)
       
        alpha_20 =  spinorb2.alpha(0)
        alpha_21 =  spinorb2.alpha(1)
        alpha_22 =  spinorb2.alpha(2)    
         
        #Defintion of orbital * alpha(orbital)
        cspinorb1_alpha10 = cspinorb1.overlap_density(alpha_10, prec)
        cspinorb1_alpha11 = cspinorb1.overlap_density(alpha_11, prec)
        cspinorb1_alpha12 = cspinorb1.overlap_density(alpha_12, prec)
       
        cspinorb1_alpha20 = cspinorb1.overlap_density(alpha_20, prec)
        cspinorb1_alpha21 = cspinorb1.overlap_density(alpha_21, prec)
        cspinorb1_alpha22 = cspinorb1.overlap_density(alpha_22, prec)
       
        cspinorb2_alpha10 = cspinorb2.overlap_density(alpha_10, prec)
        cspinorb2_alpha11 = cspinorb2.overlap_density(alpha_11, prec)
        cspinorb2_alpha12 = cspinorb2.overlap_density(alpha_12, prec)
       
        cspinorb2_alpha20 = cspinorb2.overlap_density(alpha_20, prec)
        cspinorb2_alpha21 = cspinorb2.overlap_density(alpha_21, prec)
        cspinorb2_alpha22 = cspinorb2.overlap_density(alpha_22, prec)
        
              
        #Definition of GJx
        GJ11_Re0 = Pua(cspinorb1_alpha10.real) * (2.0 * np.pi)
        GJ11_Re1 = Pua(cspinorb1_alpha11.real) * (2.0 * np.pi)
        GJ11_Re2 = Pua(cspinorb1_alpha12.real) * (2.0 * np.pi)
        GJ22_Re0 = Pua(cspinorb2_alpha20.real) * (2.0 * np.pi)
        GJ22_Re1 = Pua(cspinorb2_alpha21.real) * (2.0 * np.pi)
        GJ22_Re2 = Pua(cspinorb2_alpha22.real) * (2.0 * np.pi)
        GJ11_Im0 = Pua(cspinorb1_alpha10.imag) * (2.0 * np.pi)
        GJ11_Im1 = Pua(cspinorb1_alpha11.imag) * (2.0 * np.pi)
        GJ11_Im2 = Pua(cspinorb1_alpha12.imag) * (2.0 * np.pi)
        GJ22_Im0 = Pua(cspinorb2_alpha20.imag) * (2.0 * np.pi)
        GJ22_Im1 = Pua(cspinorb2_alpha21.imag) * (2.0 * np.pi)
        GJ22_Im2 = Pua(cspinorb2_alpha22.imag) * (2.0 * np.pi)
        
        
        GJ11_0 = cf.complex_fcn()
        GJ11_0.real = GJ11_Re0
        GJ11_0.imag = GJ11_Im0
        GJ11_1 = cf.complex_fcn()
        GJ11_1.real = GJ11_Re1
        GJ11_1.imag = GJ11_Im1
        GJ11_2 = cf.complex_fcn()
        GJ11_2.real = GJ11_Re2
        GJ11_2.imag = GJ11_Im2
    
        GJ22_0 = cf.complex_fcn()
        GJ22_0.real = GJ22_Re0
        GJ22_0.imag = GJ22_Im0
        GJ22_1 = cf.complex_fcn()
        GJ22_1.real = GJ22_Re1
        GJ22_1.imag = GJ22_Im1
        GJ22_2 = cf.complex_fcn()
        GJ22_2.real = GJ22_Re2
        GJ22_2.imag = GJ22_Im2
           
    
        #Definition of GKx
        GK12_Re0 = Pua(cspinorb1_alpha20.real) * (2.0 * np.pi)
        GK12_Re1 = Pua(cspinorb1_alpha21.real) * (2.0 * np.pi)
        GK12_Re2 = Pua(cspinorb1_alpha22.real) * (2.0 * np.pi)
        GK21_Re0 = Pua(cspinorb2_alpha10.real) * (2.0 * np.pi)
        GK21_Re1 = Pua(cspinorb2_alpha11.real) * (2.0 * np.pi)
        GK21_Re2 = Pua(cspinorb2_alpha12.real) * (2.0 * np.pi)
        GK12_Im0 = Pua(cspinorb1_alpha20.imag) * (2.0 * np.pi)
        GK12_Im1 = Pua(cspinorb1_alpha21.imag) * (2.0 * np.pi)
        GK12_Im2 = Pua(cspinorb1_alpha22.imag) * (2.0 * np.pi)
        GK21_Im0 = Pua(cspinorb2_alpha10.imag) * (2.0 * np.pi)
        GK21_Im1 = Pua(cspinorb2_alpha11.imag) * (2.0 * np.pi)
        GK21_Im2 = Pua(cspinorb2_alpha12.imag) * (2.0 * np.pi)
        
           
        GK12_0 = cf.complex_fcn()
        GK12_0.real = GK12_Re0
        GK12_0.imag = GK12_Im0
        GK12_1 = cf.complex_fcn()
        GK12_1.real = GK12_Re1
        GK12_1.imag = GK12_Im1
        GK12_2 = cf.complex_fcn()
        GK12_2.real = GK12_Re2
        GK12_2.imag = GK12_Im2
    
        GK21_0 = cf.complex_fcn()
        GK21_0.real = GK21_Re0
        GK21_0.imag = GK21_Im0
        GK21_1 = cf.complex_fcn()
        GK21_1.real = GK21_Re1
        GK21_1.imag = GK21_Im1
        GK21_2 = cf.complex_fcn()
        GK21_2.real = GK21_Re2
        GK21_2.imag = GK21_Im2
 
 
        # Calculation of necessary potential contributions to Helmotz and Energy 
        CJ_spinorb1   = orb.apply_potential(1.0, J11, spinorb1, prec)
        CK_spinorb1   = orb.apply_potential(1.0, K2, spinorb2, prec)
 
        CJ_spinorb2   = orb.apply_potential(1.0, J22, spinorb2, prec)
        CK_spinorb2   = orb.apply_potential(1.0, K1, spinorb1, prec)
 
        GJ11_0_alpha10 = orb.apply_complex_potential(1.0, GJ11_0, alpha_10, prec)
        GJ11_1_alpha11 = orb.apply_complex_potential(1.0, GJ11_1, alpha_11, prec)
        GJ11_2_alpha12 = orb.apply_complex_potential(1.0, GJ11_2, alpha_12, prec)
        GJ22_0_alpha20 = orb.apply_complex_potential(1.0, GJ22_0, alpha_20, prec)
        GJ22_1_alpha21 = orb.apply_complex_potential(1.0, GJ22_1, alpha_21, prec)
        GJ22_2_alpha22 = orb.apply_complex_potential(1.0, GJ22_2, alpha_22, prec)
        
        GK12_0_alpha10 = orb.apply_complex_potential(1.0, GK12_0, alpha_10, prec)
        GK12_1_alpha11 = orb.apply_complex_potential(1.0, GK12_1, alpha_11, prec)
        GK12_2_alpha12 = orb.apply_complex_potential(1.0, GK12_2, alpha_12, prec)
        GK21_0_alpha20 = orb.apply_complex_potential(1.0, GK21_0, alpha_20, prec)
        GK21_1_alpha21 = orb.apply_complex_potential(1.0, GK21_1, alpha_21, prec)
        GK21_2_alpha22 = orb.apply_complex_potential(1.0, GK21_2, alpha_22, prec)
 
 
        # Definition of Energy Hartree of Fock matrix
        E_GJ110, imag_E_GJ110 = spinorb1.dot(GJ11_0_alpha10)
        E_GJ111, imag_E_GJ111 = spinorb1.dot(GJ11_1_alpha11)
        E_GJ112, imag_E_GJ112 = spinorb1.dot(GJ11_2_alpha12)
        E_GJ220, imag_E_GJ220 = spinorb2.dot(GJ22_0_alpha20)
        E_GJ221, imag_E_GJ221 = spinorb2.dot(GJ22_1_alpha21)
        E_GJ222, imag_E_GJ222 = spinorb2.dot(GJ22_2_alpha22)
     
 
        E_Gxc120, imag_E_Gxc120 = spinorb2.dot(GK12_0_alpha10)
        E_Gxc121, imag_E_Gxc121 = spinorb2.dot(GK12_1_alpha11)
        E_Gxc122, imag_E_Gxc122 = spinorb2.dot(GK12_2_alpha12)
        E_Gxc210, imag_E_Gxc210 = spinorb1.dot(GK21_0_alpha20)
        E_Gxc211, imag_E_Gxc211 = spinorb1.dot(GK21_1_alpha21)
        E_Gxc212, imag_E_Gxc212 = spinorb1.dot(GK21_2_alpha22)
 
 
        E_GJ = E_GJ110 + E_GJ111 + E_GJ112 + E_GJ220 + E_GJ221 + E_GJ222
     
        E_GK = E_Gxc120 + E_Gxc121 + E_Gxc122 + E_Gxc210 + E_Gxc211 + E_Gxc212
 
        E_Gaunt = E_GJ - E_GK
 
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
        energy_11 = energy_11 + E_H11 - E_xc11 - (E_GJ110 + E_GJ111 + E_GJ112 - E_Gxc120 - E_Gxc121 - E_Gxc122)
        energy_12 = energy_12 + E_H12 - E_xc12 
        energy_21 = energy_21 + E_H21 - E_xc21 
        energy_22 = energy_22 + E_H22 - E_xc22 - (E_GJ220 + E_GJ221 + E_GJ222 - E_Gxc210 - E_Gxc211 - E_Gxc212)
        print('Energy_Spin_Orbit_1', energy_11 - light_speed**2.0)
        print('Energy_Spin_Orbit_2', energy_22 - light_speed**2.0)
 
        #Total Energy
        E_tot = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_Gaunt)
        print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2.0))
        print('E_Gåunt_cørræctiøn =', E_Gaunt * 0.5 )
 
        # Calculation of necessary potential contributions to Hellmotz
        F12_spinorb2 = energy_12 * spinorb2
        F21_spinorb1 = energy_21 * spinorb1
 
        V_J_K_spinorb1 = v_psi_1 + CJ_spinorb1 - CK_spinorb1 - GJ11_0_alpha10 - GJ11_1_alpha11 - GJ11_2_alpha12 + GK12_0_alpha10 - GK12_1_alpha11 - GK12_2_alpha12 - F12_spinorb2
        V_J_K_spinorb2 = v_psi_2 + CJ_spinorb2 - CK_spinorb2 - GJ22_0_alpha20 - GJ22_1_alpha21 - GJ22_2_alpha22 + GK21_0_alpha20 - GK21_1_alpha21 - GK21_2_alpha22 - F21_spinorb1
 
 
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
 
 
    #Definition of GJx
    GJ11_Re0 = Pua(cspinorb1_alpha10.real) * (2.0 * np.pi)
    GJ11_Re1 = Pua(cspinorb1_alpha11.real) * (2.0 * np.pi)
    GJ11_Re2 = Pua(cspinorb1_alpha12.real) * (2.0 * np.pi)
    GJ22_Re0 = Pua(cspinorb2_alpha20.real) * (2.0 * np.pi)
    GJ22_Re1 = Pua(cspinorb2_alpha21.real) * (2.0 * np.pi)
    GJ22_Re2 = Pua(cspinorb2_alpha22.real) * (2.0 * np.pi)
    GJ11_Im0 = Pua(cspinorb1_alpha10.imag) * (2.0 * np.pi)
    GJ11_Im1 = Pua(cspinorb1_alpha11.imag) * (2.0 * np.pi)
    GJ11_Im2 = Pua(cspinorb1_alpha12.imag) * (2.0 * np.pi)
    GJ22_Im0 = Pua(cspinorb2_alpha20.imag) * (2.0 * np.pi)
    GJ22_Im1 = Pua(cspinorb2_alpha21.imag) * (2.0 * np.pi)
    GJ22_Im2 = Pua(cspinorb2_alpha22.imag) * (2.0 * np.pi)
    
    
    GJ11_0 = cf.complex_fcn()
    GJ11_0.real = GJ11_Re0
    GJ11_0.imag = GJ11_Im0
    GJ11_1 = cf.complex_fcn()
    GJ11_1.real = GJ11_Re1
    GJ11_1.imag = GJ11_Im1
    GJ11_2 = cf.complex_fcn()
    GJ11_2.real = GJ11_Re2
    GJ11_2.imag = GJ11_Im2
    GJ22_0 = cf.complex_fcn()
    GJ22_0.real = GJ22_Re0
    GJ22_0.imag = GJ22_Im0
    GJ22_1 = cf.complex_fcn()
    GJ22_1.real = GJ22_Re1
    GJ22_1.imag = GJ22_Im1
    GJ22_2 = cf.complex_fcn()
    GJ22_2.real = GJ22_Re2
    GJ22_2.imag = GJ22_Im2
       
    #Definition of GKx
    GK12_Re0 = Pua(cspinorb1_alpha20.real) * (2.0 * np.pi)
    GK12_Re1 = Pua(cspinorb1_alpha21.real) * (2.0 * np.pi)
    GK12_Re2 = Pua(cspinorb1_alpha22.real) * (2.0 * np.pi)
    GK21_Re0 = Pua(cspinorb2_alpha10.real) * (2.0 * np.pi)
    GK21_Re1 = Pua(cspinorb2_alpha11.real) * (2.0 * np.pi)
    GK21_Re2 = Pua(cspinorb2_alpha12.real) * (2.0 * np.pi)
    GK12_Im0 = Pua(cspinorb1_alpha20.imag) * (2.0 * np.pi)
    GK12_Im1 = Pua(cspinorb1_alpha21.imag) * (2.0 * np.pi)
    GK12_Im2 = Pua(cspinorb1_alpha22.imag) * (2.0 * np.pi)
    GK21_Im0 = Pua(cspinorb2_alpha10.imag) * (2.0 * np.pi)
    GK21_Im1 = Pua(cspinorb2_alpha11.imag) * (2.0 * np.pi)
    GK21_Im2 = Pua(cspinorb2_alpha12.imag) * (2.0 * np.pi)
    
       
    GK12_0 = cf.complex_fcn()
    GK12_0.real = GK12_Re0
    GK12_0.imag = GK12_Im0
    GK12_1 = cf.complex_fcn()
    GK12_1.real = GK12_Re1
    GK12_1.imag = GK12_Im1
    GK12_2 = cf.complex_fcn()
    GK12_2.real = GK12_Re2
    GK12_2.imag = GK12_Im2
    GK21_0 = cf.complex_fcn()
    GK21_0.real = GK21_Re0
    GK21_0.imag = GK21_Im0
    GK21_1 = cf.complex_fcn()
    GK21_1.real = GK21_Re1
    GK21_1.imag = GK21_Im1
    GK21_2 = cf.complex_fcn()
    GK21_2.real = GK21_Re2
    GK21_2.imag = GK21_Im2
 
    # Calculation of necessary potential contributions to Energy 
    GJ11_0_alpha10 = orb.apply_complex_potential(1.0, GJ11_0, alpha_10, prec)
    GJ11_1_alpha11 = orb.apply_complex_potential(1.0, GJ11_1, alpha_11, prec)
    GJ11_2_alpha12 = orb.apply_complex_potential(1.0, GJ11_2, alpha_12, prec)
    GJ22_0_alpha20 = orb.apply_complex_potential(1.0, GJ22_0, alpha_20, prec)
    GJ22_1_alpha21 = orb.apply_complex_potential(1.0, GJ22_1, alpha_21, prec)
    GJ22_2_alpha22 = orb.apply_complex_potential(1.0, GJ22_2, alpha_22, prec)
    
    GK12_0_alpha10 = orb.apply_complex_potential(1.0, GK12_0, alpha_10, prec)
    GK12_1_alpha11 = orb.apply_complex_potential(1.0, GK12_1, alpha_11, prec)
    GK12_2_alpha12 = orb.apply_complex_potential(1.0, GK12_2, alpha_12, prec)
    GK21_0_alpha20 = orb.apply_complex_potential(1.0, GK21_0, alpha_20, prec)
    GK21_1_alpha21 = orb.apply_complex_potential(1.0, GK21_1, alpha_21, prec)
    GK21_2_alpha22 = orb.apply_complex_potential(1.0, GK21_2, alpha_22, prec)
 
    # Definition of Energy Hartree of Fock matrix
    E_GJ110, imag_E_GJ110 = spinorb1.dot(GJ11_0_alpha10)
    E_GJ111, imag_E_GJ111 = spinorb1.dot(GJ11_1_alpha11)
    E_GJ112, imag_E_GJ112 = spinorb1.dot(GJ11_2_alpha12)
    E_GJ220, imag_E_GJ220 = spinorb2.dot(GJ22_0_alpha20)
    E_GJ221, imag_E_GJ221 = spinorb2.dot(GJ22_1_alpha21)
    E_GJ222, imag_E_GJ222 = spinorb2.dot(GJ22_2_alpha22)
 
    E_Gxc120, imag_E_Gxc120 = spinorb2.dot(GK12_0_alpha10)
    E_Gxc121, imag_E_Gxc121 = spinorb2.dot(GK12_1_alpha11)
    E_Gxc122, imag_E_Gxc122 = spinorb2.dot(GK12_2_alpha12)
    E_Gxc210, imag_E_Gxc210 = spinorb1.dot(GK21_0_alpha20)
    E_Gxc211, imag_E_Gxc211 = spinorb1.dot(GK21_1_alpha21)
    E_Gxc212, imag_E_Gxc212 = spinorb1.dot(GK21_2_alpha22)
 
    E_GJ = E_GJ110 + E_GJ111 + E_GJ112 + E_GJ220 + E_GJ221 + E_GJ222
    
    E_GK = E_Gxc120 + E_Gxc121 + E_Gxc122 + E_Gxc210 + E_Gxc211 + E_Gxc212
    
    E_Gaunt = E_GJ - E_GK
 
 
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
    energy_11 = energy_11 + E_H11 - E_xc11 - (E_GJ110 + E_GJ111 + E_GJ112 - E_Gxc120 - E_Gxc121 - E_Gxc122)
    energy_12 = energy_12 + E_H12 - E_xc12 
    energy_21 = energy_21 + E_H21 - E_xc21 
    energy_22 = energy_22 + E_H22 - E_xc22 - (E_GJ220 + E_GJ221 + E_GJ222 - E_Gxc210 - E_Gxc211 - E_Gxc212)
    print('Energy_Spin_Orbit_1', energy_11 - light_speed**2.0)
    print('Energy_Spin_Orbit_2', energy_22 - light_speed**2.0)
 
 
    # Total Energy 
    E_tot = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_Gaunt)
    print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2.0))
    print('E_Gåunt_cørræctiøn =', E_Gaunt * 0.5 )
#########################################################END###########################################################################
