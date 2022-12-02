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
    parser.add_argument('-cx', '--center_x', dest='cx', type=float, default=0.0,
                        help='position of nucleus in x')
    parser.add_argument('-cy', '--center_y', dest='cy', type=float, default=0.0,
                        help='position of nucleus in y')
    parser.add_argument('-cz', '--center_z', dest='cz', type=float, default=0.0,
                        help='position of nucleus in z')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=8,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-5,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt'], 'Please, specify coulgau in a rigth way – coulomb or gaunt'

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS'], 'Please, specify the type of derivative'


################# Define Paramters ###########################
light_speed = args.lux_speed
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
origin = [args.cx, args.cy, args.cz]
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
spinorb2 = cspinorb1.ktrs()

print('Define spinorbital alpha DONE')

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

        # Defintion of J
        J = Pua(n_11 + n_22) * (4 * np.pi)
        #print('J', J)

        # Definition of Kx
        K1a = Pua(n_21) * (4 * np.pi)
        K1b = Pua(n_11) * (4 * np.pi)
        K2a = Pua(n_12) * (4 * np.pi)
        K2b = Pua(n_22) * (4 * np.pi)
        #print('K1a', K1a)
        #print('K2b', K2b)


        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)


        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + v_psi_1
        add_psi_2 = hd_psi_2 + v_psi_2


        # Calculation of necessary potential contributions to Hellmotz
        J_spinorb1  = orb.apply_potential(1.0, J, spinorb1, prec)
        J_spinorb2  = orb.apply_potential(1.0, J, spinorb2, prec)        


        Ka_spinorb1  = orb.apply_potential(1.0, K1a, spinorb2, prec)
        Kb_spinorb1  = orb.apply_potential(1.0, K1b, spinorb1, prec)
        Ka_spinorb2  = orb.apply_potential(1.0, K2a, spinorb1, prec)
        Kb_spinorb2  = orb.apply_potential(1.0, K2b, spinorb2, prec)


        K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
        K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
        #print('K_spinorb1', K_spinorb1)


        E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
        E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
             
             
        E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
        E_K22, imag_K22 = spinorb2.dot(K_spinorb2)


        # Orbital Energy calculation
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)


        # Calculate Fij Fock matrix
        F_11 = energy_11 + E_H11 - E_K11
        F_12 = energy_12
        F_22 = energy_22 + E_H22 - E_K22
        

        # Orbital Energy
        print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', F_22 - light_speed**2)


        # Total Energy 
        E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22)
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))


        V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - (F_12 * spinorb2)


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, F_11, prec)

        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F_11, der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        
        cnew_orbital_1 = new_orbital_1.complex_conj()
        new_orbital_2 = cnew_orbital_1.ktrs() 


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

    # Defintion of J
    J = Pua(n_11 + n_22) * (4 * np.pi)
    #print('J', J)

    # Definition of Kx
    K1a = Pua(n_21) * (4 * np.pi)
    K1b = Pua(n_11) * (4 * np.pi)
    K2a = Pua(n_12) * (4 * np.pi)
    K2b = Pua(n_22) * (4 * np.pi)
    #print('K1a', K1a)
    #print('K2b', K2b)


    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)


    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2


    # Calculation of necessary potential contributions to Hellmotz
    J_spinorb1  = orb.apply_potential(1.0, J, spinorb1, prec)
    J_spinorb2  = orb.apply_potential(1.0, J, spinorb2, prec)        


    Ka_spinorb1  = orb.apply_potential(1.0, K1a, spinorb2, prec)
    Kb_spinorb1  = orb.apply_potential(1.0, K1b, spinorb1, prec)
    Ka_spinorb2  = orb.apply_potential(1.0, K2a, spinorb1, prec)
    Kb_spinorb2  = orb.apply_potential(1.0, K2b, spinorb2, prec)


    K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
    K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
    #print('K_spinorb1', K_spinorb1)


    E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
    E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
         
         
    E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
    E_K22, imag_K22 = spinorb2.dot(K_spinorb2)


    # Orbital Energy calculation
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)


    # Calculate Fij Fock matrix
    F_11 = energy_11 + E_H11 - E_K11
    F_22 = energy_22 + E_H22 - E_K22
        

    # Orbital Energy
    print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', F_22 - light_speed**2)


    # Total Energy 
    E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22)
    print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

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

        # Defintion of J
        J = Pua(n_11 + n_22) * (4 * np.pi)
        #print('J', J)

        # Definition of Kx
        K1a = Pua(n_21) * (4 * np.pi)
        K1b = Pua(n_11) * (4 * np.pi)
        K2a = Pua(n_12) * (4 * np.pi)
        K2b = Pua(n_22) * (4 * np.pi)
        #print('K1a', K1a)
        #print('K2b', K2b)


        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)


        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)


        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + v_psi_1
        add_psi_2 = hd_psi_2 + v_psi_2


        # Calculation of necessary potential contributions to Hellmotz
        J_spinorb1  = orb.apply_potential(1.0, J, spinorb1, prec)
        J_spinorb2  = orb.apply_potential(1.0, J, spinorb2, prec)        


        Ka_spinorb1  = orb.apply_potential(1.0, K1a, spinorb2, prec)
        Kb_spinorb1  = orb.apply_potential(1.0, K1b, spinorb1, prec)
        Ka_spinorb2  = orb.apply_potential(1.0, K2a, spinorb1, prec)
        Kb_spinorb2  = orb.apply_potential(1.0, K2b, spinorb2, prec)


        K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
        K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
        #print('K_spinorb1', K_spinorb1)


        E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
        E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
             
             
        E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
        E_K22, imag_K22 = spinorb2.dot(K_spinorb2)


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
        GJ_Re0 = Pua(cspinorb1_alpha10.real + cspinorb2_alpha20.real) * (2.0 * np.pi)
        
        GJ_Re1 = Pua(cspinorb1_alpha11.real + cspinorb2_alpha21.real) * (2.0 * np.pi)
        
        GJ_Re2 = Pua(cspinorb1_alpha12.real + cspinorb2_alpha22.real) * (2.0 * np.pi)
        
        
        GJ_Im0 = Pua(cspinorb1_alpha10.imag + cspinorb2_alpha20.imag) * (2.0 * np.pi)
        
        GJ_Im1 = Pua(cspinorb1_alpha11.imag + cspinorb2_alpha21.imag) * (2.0 * np.pi)
        
        GJ_Im2 = Pua(cspinorb1_alpha12.imag + cspinorb2_alpha22.imag) * (2.0 * np.pi)
        
        
        
        GJ_0 = cf.complex_fcn()
        GJ_0.real = GJ_Re0
        GJ_0.imag = GJ_Im0
        
        GJ_1 = cf.complex_fcn()
        GJ_1.real = GJ_Re1
        GJ_1.imag = GJ_Im1

        GJ_2 = cf.complex_fcn()
        GJ_2.real = GJ_Re2
        GJ_2.imag = GJ_Im2
    

        #Definition of GKx
        GK1a_Re0 = Pua(cspinorb2_alpha10.real) * (2.0 * np.pi)
        GK1a_Re1 = Pua(cspinorb2_alpha11.real) * (2.0 * np.pi)
        GK1a_Re2 = Pua(cspinorb2_alpha12.real) * (2.0 * np.pi)

        GK1b_Re0 = Pua(cspinorb1_alpha10.real) * (2.0 * np.pi)
        GK1b_Re1 = Pua(cspinorb1_alpha11.real) * (2.0 * np.pi)
        GK1b_Re2 = Pua(cspinorb1_alpha12.real) * (2.0 * np.pi)

        GK2a_Re0 = Pua(cspinorb1_alpha20.real) * (2.0 * np.pi)
        GK2a_Re1 = Pua(cspinorb1_alpha21.real) * (2.0 * np.pi)
        GK2a_Re2 = Pua(cspinorb1_alpha22.real) * (2.0 * np.pi)

        GK2b_Re0 = Pua(cspinorb2_alpha20.real) * (2.0 * np.pi)
        GK2b_Re1 = Pua(cspinorb2_alpha21.real) * (2.0 * np.pi)
        GK2b_Re2 = Pua(cspinorb2_alpha22.real) * (2.0 * np.pi)

        
        GK1a_Im0 = Pua(cspinorb2_alpha10.imag) * (2.0 * np.pi)
        GK1a_Im1 = Pua(cspinorb2_alpha11.imag) * (2.0 * np.pi)
        GK1a_Im2 = Pua(cspinorb2_alpha12.imag) * (2.0 * np.pi)

        GK1b_Im0 = Pua(cspinorb1_alpha10.imag) * (2.0 * np.pi)
        GK1b_Im1 = Pua(cspinorb1_alpha11.imag) * (2.0 * np.pi)
        GK1b_Im2 = Pua(cspinorb1_alpha12.imag) * (2.0 * np.pi)
        
        GK2a_Im0 = Pua(cspinorb1_alpha20.imag) * (2.0 * np.pi)
        GK2a_Im1 = Pua(cspinorb1_alpha21.imag) * (2.0 * np.pi)
        GK2a_Im2 = Pua(cspinorb1_alpha22.imag) * (2.0 * np.pi)
        
        GK2b_Im0 = Pua(cspinorb2_alpha20.imag) * (2.0 * np.pi)
        GK2b_Im1 = Pua(cspinorb2_alpha21.imag) * (2.0 * np.pi)
        GK2b_Im2 = Pua(cspinorb2_alpha22.imag) * (2.0 * np.pi)
        
        

        GK1a_0 = cf.complex_fcn()
        GK1a_0.real = GK1a_Re0
        GK1a_0.imag = GK1a_Im0
        GK1b_0 = cf.complex_fcn()
        GK1b_0.real = GK1b_Re0
        GK1b_0.imag = GK1b_Im0
 
        GK2a_0 = cf.complex_fcn()
        GK2a_0.real = GK2a_Re0
        GK2a_0.imag = GK2a_Im0
        GK2b_0 = cf.complex_fcn()
        GK2b_0.real = GK2b_Re0
        GK2b_0.imag = GK2b_Im0


        GK1a_1 = cf.complex_fcn()
        GK1a_1.real = GK1a_Re1
        GK1a_1.imag = GK1a_Im1
        GK1b_1 = cf.complex_fcn()
        GK1b_1.real = GK1b_Re1
        GK1b_1.imag = GK1b_Im1
 
        GK2a_1 = cf.complex_fcn()
        GK2a_1.real = GK2a_Re1
        GK2a_1.imag = GK2a_Im1
        GK2b_1 = cf.complex_fcn()
        GK2b_1.real = GK2b_Re1
        GK2b_1.imag = GK2b_Im1


        GK1a_2 = cf.complex_fcn()
        GK1a_2.real = GK1a_Re2
        GK1a_2.imag = GK1a_Im2
        GK1b_2 = cf.complex_fcn()
        GK1b_2.real = GK1b_Re2
        GK1b_2.imag = GK1b_Im2
 
        GK2a_2 = cf.complex_fcn()
        GK2a_2.real = GK2a_Re2
        GK2a_2.imag = GK2a_Im2
        GK2b_2 = cf.complex_fcn()
        GK2b_2.real = GK2b_Re2
        GK2b_2.imag = GK2b_Im2


        # Calculation of necessary potential contributions to Hellmotz
        VG10 = orb.apply_complex_potential(1.0, GJ_0, alpha_10, prec)
        VG11 = orb.apply_complex_potential(1.0, GJ_1, alpha_11, prec)
        VG12 = orb.apply_complex_potential(1.0, GJ_2, alpha_12, prec)
        GJ_spinorb1 = VG10 + VG11 + VG12


        VG20 = orb.apply_complex_potential(1.0, GJ_0, alpha_20, prec)
        VG21 = orb.apply_complex_potential(1.0, GJ_1, alpha_21, prec)
        VG22 = orb.apply_complex_potential(1.0, GJ_2, alpha_22, prec)
        GJ_spinorb2 = VG20 + VG21 + VG22


        GKa_spinorb1_0  = orb.apply_complex_potential(1.0, GK1a_0, alpha_20, prec)
        GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, alpha_10, prec)
        GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, alpha_21, prec)
        GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, alpha_11, prec)
        GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, alpha_22, prec)
        GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, alpha_12, prec)
    

        GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, alpha_10, prec)
        GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, alpha_20, prec)
        GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, alpha_11, prec)
        GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, alpha_21, prec)
        GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, alpha_12, prec)
        GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, alpha_22, prec)



        GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
        GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
        #print('K_spinorb1', K_spinorb1)


        E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
        E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)
             
             
        E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
        E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)


        # Orbital Energy calculation
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)


        # Calculate Fij Fock matrix
        F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
        F_12 = energy_12
        F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
        

        # Orbital Energy
        print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', F_22 - light_speed**2)


        # Total Energy 
        E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 + E_GK11 - E_GH22 + E_GK22)
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

 
        V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - GJ_spinorb1 + GK_spinorb1 - (F_12 * spinorb2)
    
 
        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, F_11, prec)
 
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F_11)
        new_orbital_1 *= 0.5 / light_speed ** 2
        new_orbital_1.normalize()
        cnew_orbital_1 = new_orbital_1.complex_conj()
        new_orbital_2 = cnew_orbital_1.ktrs() 
 
 
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

    # Defintion of J
    J = Pua(n_11 + n_22) * (4 * np.pi)
    
    #print('J', J)
    # Definition of Kx
    K1a = Pua(n_21) * (4 * np.pi)
    K1b = Pua(n_11) * (4 * np.pi)
    K2a = Pua(n_12) * (4 * np.pi)
    K2b = Pua(n_22) * (4 * np.pi)
    #print('K1a', K1a)
    #print('K2b', K2b)

    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)

    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)

    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2

    # Calculation of necessary potential contributions to Hellmotz
    J_spinorb1  = orb.apply_potential(1.0, J, spinorb1, prec)
    J_spinorb2  = orb.apply_potential(1.0, J, spinorb2, prec) 

    Ka_spinorb1  = orb.apply_potential(1.0, K1a, spinorb2, prec)
    Kb_spinorb1  = orb.apply_potential(1.0, K1b, spinorb1, prec)
    Ka_spinorb2  = orb.apply_potential(1.0, K2a, spinorb1, prec)
    Kb_spinorb2  = orb.apply_potential(1.0, K2b, spinorb2, prec)
    K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
    K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
    #print('K_spinorb1', K_spinorb1)
    

    E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
    E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
         
         
    E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
    E_K22, imag_K22 = spinorb2.dot(K_spinorb2)
    

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
    GJ_Re0 = Pua(cspinorb1_alpha10.real + cspinorb2_alpha20.real) * (2.0 * np.pi)
    
    GJ_Re1 = Pua(cspinorb1_alpha11.real + cspinorb2_alpha21.real) * (2.0 * np.pi)
    
    GJ_Re2 = Pua(cspinorb1_alpha12.real + cspinorb2_alpha22.real) * (2.0 * np.pi)
    
    
    GJ_Im0 = Pua(cspinorb1_alpha10.imag + cspinorb2_alpha20.imag) * (2.0 * np.pi)
    
    GJ_Im1 = Pua(cspinorb1_alpha11.imag + cspinorb2_alpha21.imag) * (2.0 * np.pi)
    
    GJ_Im2 = Pua(cspinorb1_alpha12.imag + cspinorb2_alpha22.imag) * (2.0 * np.pi)
    
    
    
    GJ_0 = cf.complex_fcn()
    GJ_0.real = GJ_Re0
    GJ_0.imag = GJ_Im0
    
    GJ_1 = cf.complex_fcn()
    GJ_1.real = GJ_Re1
    GJ_1.imag = GJ_Im1
    GJ_2 = cf.complex_fcn()
    GJ_2.real = GJ_Re2
    GJ_2.imag = GJ_Im2

    #Definition of GKx
    GK1a_Re0 = Pua(cspinorb2_alpha10.real) * (2.0 * np.pi)
    GK1a_Re1 = Pua(cspinorb2_alpha11.real) * (2.0 * np.pi)
    GK1a_Re2 = Pua(cspinorb2_alpha12.real) * (2.0 * np.pi)
    GK1b_Re0 = Pua(cspinorb1_alpha10.real) * (2.0 * np.pi)
    GK1b_Re1 = Pua(cspinorb1_alpha11.real) * (2.0 * np.pi)
    GK1b_Re2 = Pua(cspinorb1_alpha12.real) * (2.0 * np.pi)
    
    GK2a_Re0 = Pua(cspinorb1_alpha20.real) * (2.0 * np.pi)
    GK2a_Re1 = Pua(cspinorb1_alpha21.real) * (2.0 * np.pi)
    GK2a_Re2 = Pua(cspinorb1_alpha22.real) * (2.0 * np.pi)
    GK2b_Re0 = Pua(cspinorb2_alpha20.real) * (2.0 * np.pi)
    GK2b_Re1 = Pua(cspinorb2_alpha21.real) * (2.0 * np.pi)
    GK2b_Re2 = Pua(cspinorb2_alpha22.real) * (2.0 * np.pi)
    
    GK1a_Im0 = Pua(cspinorb2_alpha10.imag) * (2.0 * np.pi)
    GK1a_Im1 = Pua(cspinorb2_alpha11.imag) * (2.0 * np.pi)
    GK1a_Im2 = Pua(cspinorb2_alpha12.imag) * (2.0 * np.pi)
    GK1b_Im0 = Pua(cspinorb1_alpha10.imag) * (2.0 * np.pi)
    GK1b_Im1 = Pua(cspinorb1_alpha11.imag) * (2.0 * np.pi)
    GK1b_Im2 = Pua(cspinorb1_alpha12.imag) * (2.0 * np.pi)
    
    GK2a_Im0 = Pua(cspinorb1_alpha20.imag) * (2.0 * np.pi)
    GK2a_Im1 = Pua(cspinorb1_alpha21.imag) * (2.0 * np.pi)
    GK2a_Im2 = Pua(cspinorb1_alpha22.imag) * (2.0 * np.pi)
    
    GK2b_Im0 = Pua(cspinorb2_alpha20.imag) * (2.0 * np.pi)
    GK2b_Im1 = Pua(cspinorb2_alpha21.imag) * (2.0 * np.pi)
    GK2b_Im2 = Pua(cspinorb2_alpha22.imag) * (2.0 * np.pi)
    
    
    GK1a_0 = cf.complex_fcn()
    GK1a_0.real = GK1a_Re0
    GK1a_0.imag = GK1a_Im0
    GK1b_0 = cf.complex_fcn()
    GK1b_0.real = GK1b_Re0
    GK1b_0.imag = GK1b_Im0

    GK2a_0 = cf.complex_fcn()
    GK2a_0.real = GK2a_Re0
    GK2a_0.imag = GK2a_Im0
    GK2b_0 = cf.complex_fcn()
    GK2b_0.real = GK2b_Re0
    GK2b_0.imag = GK2b_Im0
    
    GK1a_1 = cf.complex_fcn()
    GK1a_1.real = GK1a_Re1
    GK1a_1.imag = GK1a_Im1
    GK1b_1 = cf.complex_fcn()
    GK1b_1.real = GK1b_Re1
    GK1b_1.imag = GK1b_Im1

    GK2a_1 = cf.complex_fcn()
    GK2a_1.real = GK2a_Re1
    GK2a_1.imag = GK2a_Im1
    GK2b_1 = cf.complex_fcn()
    GK2b_1.real = GK2b_Re1
    GK2b_1.imag = GK2b_Im1
    
    GK1a_2 = cf.complex_fcn()
    GK1a_2.real = GK1a_Re2
    GK1a_2.imag = GK1a_Im2
    GK1b_2 = cf.complex_fcn()
    GK1b_2.real = GK1b_Re2
    GK1b_2.imag = GK1b_Im2

    GK2a_2 = cf.complex_fcn()
    GK2a_2.real = GK2a_Re2
    GK2a_2.imag = GK2a_Im2
    GK2b_2 = cf.complex_fcn()
    GK2b_2.real = GK2b_Re2
    GK2b_2.imag = GK2b_Im2
    


    VG10 = orb.apply_complex_potential(1.0, GJ_0, alpha_10, prec)
    VG11 = orb.apply_complex_potential(1.0, GJ_1, alpha_11, prec)
    VG12 = orb.apply_complex_potential(1.0, GJ_2, alpha_12, prec)
    GJ_spinorb1 = VG10 + VG11 + VG12
    

    VG20 = orb.apply_complex_potential(1.0, GJ_0, alpha_20, prec)
    VG21 = orb.apply_complex_potential(1.0, GJ_1, alpha_21, prec)
    VG22 = orb.apply_complex_potential(1.0, GJ_2, alpha_22, prec)
    GJ_spinorb2 = VG20 + VG21 + VG22
    

    GKa_spinorb1_0  = orb.apply_complex_potential(1.0, GK1a_0, alpha_20, prec)
    GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, alpha_10, prec)
    GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, alpha_21, prec)
    GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, alpha_11, prec)
    GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, alpha_22, prec)
    GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, alpha_12, prec)
    

    GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, alpha_10, prec)
    GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, alpha_20, prec)
    GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, alpha_11, prec)
    GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, alpha_21, prec)
    GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, alpha_12, prec)
    GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, alpha_22, prec)
    
    GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
    GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
    

    E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
    E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)
         
         
    E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
    E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)
    

    # Orbital Energy calculation
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)
    

    # Calculate Fij Fock matrix
    F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
    F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
    

    # Orbital Energy
    print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', F_22 - light_speed**2)
    

    # Total Energy 
    E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 + E_GK11 - E_GH22 + E_GK22)
    print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))
#########################################################END###########################################################################
