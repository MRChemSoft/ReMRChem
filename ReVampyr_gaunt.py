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

    assert args.coulgau in ['coulomb', 'gaunt', 'gaunt-test'], 'Please, specify coulgau in a rigth way – coulomb or gaunt'

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

################# Definecs Gaussian function ########## 
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
#print('spinorb1',spinorb1)
#print('cspinorb1',cspinorb1)

spinorb2 = orb.orbital4c()
spinorb2.copy_components(Lb=complexfc)
spinorb2.init_small_components(prec/10)
spinorb2.normalize()
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
elif args.potential == 'gaussian':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.gaussian(x, origin, Z, atom)
   V_tree = Peps(f)

default_der = args.deriv
print('Define V Potetintal', args.potential, 'DONE')

P = vp.PoissonOperator(mra, prec)

#############################START WITH CALCULATION###################################
if args.coulgau == 'coulomb':
    print('Hartree-Fock (Coulombic bielectric interaction)')
    error_norm = 1

    while error_norm > prec:

        n_11 = spinorb1.overlap_density(spinorb1, prec)
        n_12 = spinorb1.overlap_density(spinorb2, prec)
        n_21 = spinorb2.overlap_density(spinorb1, prec)
        n_22 = spinorb2.overlap_density(spinorb2, prec)

        # Definition of two electron operators
        B11    = P(n_11.real) * (4 * np.pi)
        B22    = P(n_22.real) * (4 * np.pi)
        B12_Re = P(n_12.real) * (4 * np.pi)
        B12_Im = P(n_12.imag) * (4 * np.pi)
        B21_Re = P(n_21.real) * (4 * np.pi)
        B21_Im = P(n_21.imag) * (4 * np.pi)

        B12 = cf.complex_fcn()
        B12.real = K12_Re
        B12.imag = K12_Im 
        
        B21 = cf.complex_fcn()
        B21.real = K21_Re
        B21.imag = K21_Im
        
        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)

        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)

        # Add Dirac and ext potential parts
        add_psi_1 = hd_psi_1 + v_psi_1
        add_psi_2 = hd_psi_2 + v_psi_2

        # Calculation of two electron terms
        J2_phi1 = orb.apply_potential(1.0, B22, spinorb1, prec)
        J1_phi2 = orb.apply_potential(1.0, B11, spinorb2, prec)
        K2_phi1 = orb.apply_complex_potential(1.0, B21, spinorb2, prec)
        K1_phi2 = orb.apply_complex_potential(1.0, B12, spinorb1, prec)

        JmK_phi1 = J2_phi1 - K2_phi1
        JmK_phi2 = J1_phi2 - K1_phi2

        JmK_11_r, JmK_11_i = spinorb1.dot(JmK_phi1)
        JmK_12_r, JmK_12_i = spinorb1.dot(JmK_phi2)
        JmK_21_r, JmK_21_i = spinorb2.dot(JmK_phi1)
        JmK_22_r, JmK_22_i = spinorb2.dot(JmK_phi2)

        JmK = np.array([ JmK_11_r + JmK_11_i * 1j , JmK_12_r + JmK_12_i * 1j ],
                       [ JmK_21_r + JmK_21_i * 1j , JmK_22_r + JmK_22_i * 1j])
        # Orbital Energy calculation
        hd_V_11_r, hd_V_11_i = Spinorb1.dot(add_psi_1)
        hd_V_12_r, hd_V_12_i = spinorb1.dot(add_psi_2)
        hd_V_21_r, hd_V_21_i = spinorb2.dot(add_psi_1)
        hd_V_22_r, hd_V_22_i = spinorb2.dot(add_psi_2)

        hd_V = np.array([ hd_V_11_r + hd_V_11_i * 1j , hd_V_12_r + hd_V_12_i * 1j ],
                        [ hd_V_21_r + hd_V_21_i * 1j , hd_V_22_r + hd_V_22_i * 1j])

        # Calculate Fij Fock matrix
        Fmat = hd_V + JmK
        eps1 = Fmat[0,0]
        eps2 = Fmat[1,1]
        
        # Orbital Energy
        print('Energy_Spin_Orbit_1', eps1 - light_speed**2)
        print('Energy_Spin_Orbit_2', eps2 - light_speed**2)

        # Total Energy 
        E_tot_JK = numpy.trace(Fmat) - 0.5 * (numpy.trace(JmK))
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1 - (F_[0,1] * spinorb2)
        V_J_K_spinorb2 = v_psi_2 + JmK_phi2 - (F_[1,0] * spinorb1)

        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, eps1, prec)
        tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, eps2, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, eps1, der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, eps2, der = default_der)
        new_orbital_2 *= 0.5/light_speed**2
        new_orbital_2.normalize()

        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        deltasq1 = delta_psi_1.squaredNorm()
        deltasq2 = delta_psi_2.squaredNorm()
        orbital_error_sq = deltasq1 + deltasq2
        error_norm = np.sqrt(orbital_error_sq)
        print('Orbital_Error norm', error_norm)

        # Compute overlap
        dot_11 = new_orbital_1.dot(new_orbital_1)
        dot_12 = new_orbital_1.dot(new_orbital_2)
        dot_21 = new_orbital_2.dot(new_orbital_1)
        dot_22 = new_orbital_2.dot(new_orbital_2)

        S_tilde = np.array([dot_11[0] + 1j * dot_11[1], dot_12[0] + 1j * dot_12[1]],
                           [dot_21[0] + 1j * dot_21[1], dot_22[0] + 1j * dot_22[1]])

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

   
    n_11 = spinorb1.overlap_density(spinorb1, prec)
    n_12 = spinorb1.overlap_density(spinorb2, prec)
    n_21 = spinorb2.overlap_density(spinorb1, prec)
    n_22 = spinorb2.overlap_density(spinorb2, prec)

    # Definition of two electron operators
    B11    = P(n_11.real) * (4 * np.pi)
    B22    = P(n_22.real) * (4 * np.pi)
    B12_Re = P(n_12.real) * (4 * np.pi)
    B12_Im = P(n_12.imag) * (4 * np.pi)
    B21_Re = P(n_21.real) * (4 * np.pi)
    B21_Im = P(n_21.imag) * (4 * np.pi)

    B12 = cf.complex_fcn()
    B12.real = K12_Re
    B12.imag = K12_Im 
    
    B21 = cf.complex_fcn()
    B21.real = K21_Re
    B21.imag = K21_Im
    
    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)

    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)

    # Add Dirac and ext potential parts
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2

    # Calculation of two electron terms
    J2_phi1 = orb.apply_potential(1.0, B22, spinorb1, prec)
    J1_phi2 = orb.apply_potential(1.0, B11, spinorb2, prec)
    K2_phi1 = orb.apply_complex_potential(1.0, B21, spinorb2, prec)
    K1_phi2 = orb.apply_complex_potential(1.0, B12, spinorb1, prec)

    JmK_phi1 = J2_phi1 - K2_phi1
    JmK_phi2 = J1_phi2 - K1_phi2

    JmK_11_r, JmK_11_i = spinorb1.dot(JmK_phi1)
    JmK_12_r, JmK_12_i = spinorb1.dot(JmK_phi2)
    JmK_21_r, JmK_21_i = spinorb2.dot(JmK_phi1)
    JmK_22_r, JmK_22_i = spinorb2.dot(JmK_phi2)

    JmK = np.array([ JmK_11_r + JmK_11_i * 1j , JmK_12_r + JmK_12_i * 1j ],
                   [ JmK_21_r + JmK_21_i * 1j , JmK_22_r + JmK_22_i * 1j]
    # Orbital Energy calculation
    hd_V_11_r, hd_V_11_i = Spinorb1.dot(add_psi_1)
    hd_V_12_r, hd_V_12_i = spinorb1.dot(add_psi_2)
    hd_V_21_r, hd_V_21_i = spinorb2.dot(add_psi_1)
    hd_V_22_r, hd_V_22_i = spinorb2.dot(add_psi_2)

    hd_V = np.array([ hd_V_11_r + hd_V_11_i * 1j , hd_V_12_r + hd_V_12_i * 1j ],
                    [ hd_V_21_r + hd_V_21_i * 1j , hd_V_22_r + hd_V_22_i * 1j])

    # Calculate Fij Fock matrix
    Fmat = hd_V + JmK
    eps1 = Fmat[0,0]
    eps2 = Fmat[1,1]
    
    # Orbital Energy
    print('Energy_Spin_Orbit_1', eps1 - light_speed**2)
    print('Energy_Spin_Orbit_2', eps2 - light_speed**2)

    # Total Energy 
    E_tot_JK = numpy.trace(Fmat) - 0.5 * (numpy.trace(JmK))
    print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 * light_speed**2))
 

#####################################################END COULOMB & START GAUNT#######################################################################
elif args.coulgau == 'gaunt':
    print('Hartræ-Føck (Cøulømbic-Gåunt bielectric interåctiøn)')
    error_norm = 1
    while error_norm > prec:


        cspinorb1 = spinorb1.complex_conj()
        cspinorb2 = spinorb2.complex_conj()
        
        # Definition of different densities
        n_11 = cspinorb1.overlap_density(spinorb1, prec)
        n_12 = cspinorb1.overlap_density(spinorb2, prec)
        n_21 = cspinorb2.overlap_density(spinorb1, prec)
        n_22 = cspinorb2.overlap_density(spinorb2, prec)

        # Definition of Poisson operator
        Pua = vp.PoissonOperator(mra, prec)

        # Defintion of J
        J_Re = Pua(n_11.real + n_22.real) * (4 * np.pi)
        J_Im = Pua(n_11.imag + n_22.imag) * (4 * np.pi)

        J = cf.complex_fcn()
        J.real = J_Re
        J.imag = J_Im
        #print('J', J)

        # Definition of Kx
        K1a_Re = Pua(n_21.real) * (4 * np.pi)
        K1a_Im = Pua(n_21.imag) * (4 * np.pi)
        K1b_Re = Pua(n_11.real) * (4 * np.pi)
        K1b_Im = Pua(n_11.imag) * (4 * np.pi)
        K2a_Re = Pua(n_12.real) * (4 * np.pi)
        K2a_Im = Pua(n_12.imag) * (4 * np.pi)
        K2b_Re = Pua(n_22.real) * (4 * np.pi)
        K2b_Im = Pua(n_22.imag) * (4 * np.pi)
        #print('K1a', K1a)
        #print('K2b', K2b)

        K1a = cf.complex_fcn()
        K1a.real = K1a_Re
        K1a.imag = K1a_Im
        
        K1b = cf.complex_fcn()
        K1b.real = K1b_Re
        K1b.imag = K1b_Im 
        
        K2a = cf.complex_fcn()
        K2a.real = K2a_Re
        K2a.imag = K2a_Im
        
        K2b = cf.complex_fcn()
        K2b.real = K2b_Re
        K2b.imag = K2b_Im


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
        J_spinorb1  = orb.apply_complex_potential(1.0, J, spinorb1, prec)
        J_spinorb2  = orb.apply_complex_potential(1.0, J, spinorb2, prec)        


        Ka_spinorb1  = orb.apply_complex_potential(1.0, K1a, spinorb2, prec)
        Kb_spinorb1  = orb.apply_complex_potential(1.0, K1b, spinorb1, prec)
        Ka_spinorb2  = orb.apply_complex_potential(1.0, K2a, spinorb1, prec)
        Kb_spinorb2  = orb.apply_complex_potential(1.0, K2b, spinorb2, prec)


        K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
        K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
        #print('K_spinorb1', K_spinorb1)


        E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
        E_H12, imag_H12 = spinorb1.dot(J_spinorb2)
        E_H21, imag_H21 = spinorb2.dot(J_spinorb1)
        E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
             
             
        E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
        E_K12, imag_K12 = spinorb1.dot(K_spinorb2)
        E_K21, imag_K21 = spinorb2.dot(K_spinorb1)
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
        GJ_Re0 = Pua(cspinorb1_alpha10.real + cspinorb2_alpha20.real) * (4.0 * np.pi)
        GJ_Re1 = Pua(cspinorb1_alpha11.real + cspinorb2_alpha21.real) * (4.0 * np.pi)
        GJ_Re2 = Pua(cspinorb1_alpha12.real + cspinorb2_alpha22.real) * (4.0 * np.pi)
        
        GJ_Im0 = Pua(cspinorb1_alpha10.imag + cspinorb2_alpha20.imag) * (4.0 * np.pi)
        GJ_Im1 = Pua(cspinorb1_alpha11.imag + cspinorb2_alpha21.imag) * (4.0 * np.pi)
        GJ_Im2 = Pua(cspinorb1_alpha12.imag + cspinorb2_alpha22.imag) * (4.0 * np.pi)

        #J gaunt vector
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
        GK1a_Re0 = Pua(cspinorb2_alpha10.real) * (4.0 * np.pi)
        GK1a_Re1 = Pua(cspinorb2_alpha11.real) * (4.0 * np.pi)
        GK1a_Re2 = Pua(cspinorb2_alpha12.real) * (4.0 * np.pi)

        GK1b_Re0 = Pua(cspinorb1_alpha10.real) * (4.0 * np.pi)
        GK1b_Re1 = Pua(cspinorb1_alpha11.real) * (4.0 * np.pi)
        GK1b_Re2 = Pua(cspinorb1_alpha12.real) * (4.0 * np.pi)

        GK2a_Re0 = Pua(cspinorb1_alpha20.real) * (4.0 * np.pi)
        GK2a_Re1 = Pua(cspinorb1_alpha21.real) * (4.0 * np.pi)
        GK2a_Re2 = Pua(cspinorb1_alpha22.real) * (4.0 * np.pi)

        GK2b_Re0 = Pua(cspinorb2_alpha20.real) * (4.0 * np.pi)
        GK2b_Re1 = Pua(cspinorb2_alpha21.real) * (4.0 * np.pi)
        GK2b_Re2 = Pua(cspinorb2_alpha22.real) * (4.0 * np.pi)

        
        GK1a_Im0 = Pua(cspinorb2_alpha10.imag) * (4.0 * np.pi)
        GK1a_Im1 = Pua(cspinorb2_alpha11.imag) * (4.0 * np.pi)
        GK1a_Im2 = Pua(cspinorb2_alpha12.imag) * (4.0 * np.pi)

        GK1b_Im0 = Pua(cspinorb1_alpha10.imag) * (4.0 * np.pi)
        GK1b_Im1 = Pua(cspinorb1_alpha11.imag) * (4.0 * np.pi)
        GK1b_Im2 = Pua(cspinorb1_alpha12.imag) * (4.0 * np.pi)
        
        GK2a_Im0 = Pua(cspinorb1_alpha20.imag) * (4.0 * np.pi)
        GK2a_Im1 = Pua(cspinorb1_alpha21.imag) * (4.0 * np.pi)
        GK2a_Im2 = Pua(cspinorb1_alpha22.imag) * (4.0 * np.pi)
        
        GK2b_Im0 = Pua(cspinorb2_alpha20.imag) * (4.0 * np.pi)
        GK2b_Im1 = Pua(cspinorb2_alpha21.imag) * (4.0 * np.pi)
        GK2b_Im2 = Pua(cspinorb2_alpha22.imag) * (4.0 * np.pi)
        
        

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


        # Calculation of necessary potential contributions to Helmholtz
        VG10 = orb.apply_complex_potential(1.0, GJ_0, alpha_10, prec)
        VG11 = orb.apply_complex_potential(1.0, GJ_1, alpha_11, prec)
        VG12 = orb.apply_complex_potential(1.0, GJ_2, alpha_12, prec)
        GJ_spinorb1 = VG10 + VG11 + VG12

        VG20 = orb.apply_complex_potential(1.0, GJ_0, alpha_20, prec)
        VG21 = orb.apply_complex_potential(1.0, GJ_1, alpha_21, prec)
        VG22 = orb.apply_complex_potential(1.0, GJ_2, alpha_22, prec)
        GJ_spinorb2 = VG20 + VG21 + VG22

        GKa_spinorb1_0  = orb.apply_complex_potential(1.0, GK1a_0, alpha_20, prec)
        GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, alpha_21, prec)
        GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, alpha_22, prec)
        GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, alpha_10, prec)
        GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, alpha_11, prec)
        GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, alpha_12, prec)
    
        GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, alpha_10, prec)
        GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, alpha_11, prec)
        GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, alpha_12, prec)
        GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, alpha_20, prec)
        GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, alpha_21, prec)
        GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, alpha_22, prec)

        GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
        GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
        #print('K_spinorb1', K_spinorb1)

        #Jij Gaunt
        E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
        E_GH12, imag_GH12 = spinorb1.dot(GJ_spinorb2)
        E_GH21, imag_GH21 = spinorb2.dot(GJ_spinorb1)
        E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)

        #Kij Gaunt
        E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
        E_GK12, imag_GK12 = spinorb1.dot(GK_spinorb2)
        E_GK21, imag_GK21 = spinorb2.dot(GK_spinorb1)
        E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)

        # (hd+V)ij
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_21, imag_21 = spinorb2.dot(add_psi_1)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)

        # (hd + V + J - K + JG - KG)ij
        F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
        F_12 = energy_12 + E_H12 - E_K12 - E_GH12 + E_GK12
        F_21 = energy_21 + E_H21 - E_K21 - E_GH21 + E_GK21
        F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
        
        # Orbital Energy
        print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', F_22 - light_speed**2)

        # Total Energy 
        E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 - E_GH22 + E_GK11 + E_GK22)
        print('E_total(Coulomb-Gaunt) approximiation', E_tot_JK - (2.0 *light_speed**2))

        #Right Hand Side
        V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - GJ_spinorb1 + GK_spinorb1 - (F_12 * spinorb2)
        V_J_K_spinorb2 = v_psi_2 + J_spinorb2 - K_spinorb2 - GJ_spinorb1 + GK_spinorb1 - (F_21 * spinorb1)
 
        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, F_11, prec)
        tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, F_22, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F_11)
        new_orbital_1 *= 0.5 / light_speed ** 2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, F_22)
        new_orbital_2 *= 0.5 / light_speed ** 2
        new_orbital_2.normalize()
 
        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        deltasq1 = delta_psi_1.squaredNorm()
        deltasq2 = delta_psi_2.squaredNorm()
        orbital_error_sq = deltasq1 + deltasq2
        error_norm = np.sqrt(orbital_error_sq)
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
    cspinorb1 = spinorb1.complex_conj()
    cspinorb2 = spinorb2.complex_conj()
        
    # Definition of different densities
    n_11 = cspinorb1.overlap_density(spinorb1, prec)
    n_12 = cspinorb1.overlap_density(spinorb2, prec)
    n_21 = cspinorb2.overlap_density(spinorb1, prec)
    n_22 = cspinorb2.overlap_density(spinorb2, prec)

    # Definition of Poisson operator
    Pua = vp.PoissonOperator(mra, prec)

    # Defintion of J
    J_Re = Pua(n_11.real + n_22.real) * (4 * np.pi)
    J_Im = Pua(n_11.imag + n_22.imag) * (4 * np.pi)

    J = cf.complex_fcn()
    J.real = J_Re
    J.imag = J_Im
    #print('J', J)

    # Definition of Kx
    K1a_Re = Pua(n_21.real) * (4 * np.pi)
    K1a_Im = Pua(n_21.imag) * (4 * np.pi)
    K1b_Re = Pua(n_11.real) * (4 * np.pi)
    K1b_Im = Pua(n_11.imag) * (4 * np.pi)
    K2a_Re = Pua(n_12.real) * (4 * np.pi)
    K2a_Im = Pua(n_12.imag) * (4 * np.pi)
    K2b_Re = Pua(n_22.real) * (4 * np.pi)
    K2b_Im = Pua(n_22.imag) * (4 * np.pi)
    #print('K1a', K1a)
    #print('K2b', K2b)

    K1a = cf.complex_fcn()
    K1a.real = K1a_Re
    K1a.imag = K1a_Im
        
    K1b = cf.complex_fcn()
    K1b.real = K1b_Re
    K1b.imag = K1b_Im 
        
    K2a = cf.complex_fcn()
    K2a.real = K2a_Re
    K2a.imag = K2a_Im
        
    K2b = cf.complex_fcn()
    K2b.real = K2b_Re
    K2b.imag = K2b_Im


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
    J_spinorb1  = orb.apply_complex_potential(1.0, J, spinorb1, prec)
    J_spinorb2  = orb.apply_complex_potential(1.0, J, spinorb2, prec) 

    Ka_spinorb1  = orb.apply_complex_potential(1.0, K1a, spinorb2, prec)
    Kb_spinorb1  = orb.apply_complex_potential(1.0, K1b, spinorb1, prec)
    Ka_spinorb2  = orb.apply_complex_potential(1.0, K2a, spinorb1, prec)
    Kb_spinorb2  = orb.apply_complex_potential(1.0, K2b, spinorb2, prec)
    K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
    K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
    #print('K_spinorb1', K_spinorb1)
    

    E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
    E_H12, imag_H12 = spinorb1.dot(J_spinorb2)
    E_H21, imag_H21 = spinorb2.dot(J_spinorb1)
    E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
         
         
    E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
    E_K12, imag_K12 = spinorb1.dot(K_spinorb2)
    E_K21, imag_K21 = spinorb2.dot(K_spinorb1)
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
    GJ_Re0 = Pua(cspinorb1_alpha10.real + cspinorb2_alpha20.real) * (4.0 * np.pi)    
    GJ_Re1 = Pua(cspinorb1_alpha11.real + cspinorb2_alpha21.real) * (4.0 * np.pi)
    GJ_Re2 = Pua(cspinorb1_alpha12.real + cspinorb2_alpha22.real) * (4.0 * np.pi)
    
    GJ_Im0 = Pua(cspinorb1_alpha10.imag + cspinorb2_alpha20.imag) * (4.0 * np.pi)
    GJ_Im1 = Pua(cspinorb1_alpha11.imag + cspinorb2_alpha21.imag) * (4.0 * np.pi)
    GJ_Im2 = Pua(cspinorb1_alpha12.imag + cspinorb2_alpha22.imag) * (4.0 * np.pi)
    
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
    GK1a_Re0 = Pua(cspinorb2_alpha10.real) * (4.0 * np.pi)
    GK1a_Re1 = Pua(cspinorb2_alpha11.real) * (4.0 * np.pi)
    GK1a_Re2 = Pua(cspinorb2_alpha12.real) * (4.0 * np.pi)

    GK1b_Re0 = Pua(cspinorb1_alpha10.real) * (4.0 * np.pi)
    GK1b_Re1 = Pua(cspinorb1_alpha11.real) * (4.0 * np.pi)
    GK1b_Re2 = Pua(cspinorb1_alpha12.real) * (4.0 * np.pi)

    GK2a_Re0 = Pua(cspinorb1_alpha20.real) * (4.0 * np.pi)
    GK2a_Re1 = Pua(cspinorb1_alpha21.real) * (4.0 * np.pi)
    GK2a_Re2 = Pua(cspinorb1_alpha22.real) * (4.0 * np.pi)

    GK2b_Re0 = Pua(cspinorb2_alpha20.real) * (4.0 * np.pi)
    GK2b_Re1 = Pua(cspinorb2_alpha21.real) * (4.0 * np.pi)
    GK2b_Re2 = Pua(cspinorb2_alpha22.real) * (4.0 * np.pi)

    
    GK1a_Im0 = Pua(cspinorb2_alpha10.imag) * (4.0 * np.pi)
    GK1a_Im1 = Pua(cspinorb2_alpha11.imag) * (4.0 * np.pi)
    GK1a_Im2 = Pua(cspinorb2_alpha12.imag) * (4.0 * np.pi)

    GK1b_Im0 = Pua(cspinorb1_alpha10.imag) * (4.0 * np.pi)
    GK1b_Im1 = Pua(cspinorb1_alpha11.imag) * (4.0 * np.pi)
    GK1b_Im2 = Pua(cspinorb1_alpha12.imag) * (4.0 * np.pi)
    
    GK2a_Im0 = Pua(cspinorb1_alpha20.imag) * (4.0 * np.pi)
    GK2a_Im1 = Pua(cspinorb1_alpha21.imag) * (4.0 * np.pi)
    GK2a_Im2 = Pua(cspinorb1_alpha22.imag) * (4.0 * np.pi)
    
    GK2b_Im0 = Pua(cspinorb2_alpha20.imag) * (4.0 * np.pi)
    GK2b_Im1 = Pua(cspinorb2_alpha21.imag) * (4.0 * np.pi)
    GK2b_Im2 = Pua(cspinorb2_alpha22.imag) * (4.0 * np.pi)

        
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
    GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, alpha_21, prec)
    GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, alpha_22, prec)
    GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, alpha_10, prec)
    GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, alpha_11, prec)
    GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, alpha_12, prec)

    GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, alpha_10, prec)
    GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, alpha_11, prec)
    GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, alpha_12, prec)
    GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, alpha_20, prec)
    GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, alpha_21, prec)
    GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, alpha_22, prec)
    
    GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
    GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
    
    E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
    E_GH12, imag_GH12 = spinorb1.dot(GJ_spinorb2)
    E_GH21, imag_GH21 = spinorb2.dot(GJ_spinorb1)
    E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)
         
    E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
    E_GK12, imag_GK12 = spinorb1.dot(GK_spinorb2)
    E_GK21, imag_GK21 = spinorb2.dot(GK_spinorb1)
    E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)
    

    # Orbital Energy calculation
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_12, imag_12 = spinorb1.dot(add_psi_2)
    energy_21, imag_21 = spinorb2.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)
    

    # Calculate Fij Fock matrix
    F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
    F_12 = energy_12 + E_H12 - E_K12 - E_GH12 + E_GK12
    F_21 = energy_21 + E_H21 - E_K21 - E_GH21 + E_GK21
    F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
    

    # Orbital Energy
    print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', F_22 - light_speed**2)
    

    # Total Energy 
    E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 - E_GH22 + E_GK11 + E_GK22)
    print('E_total(Coulomb-Gaunt) approximiation', E_tot_JK - (2.0 *light_speed**2))
#########################################################END###########################################################################
elif args.coulgau == 'gaunt-test':
    print('Hartræ-Føck (Cøulømbic-Gåunt bielectric interåctiøn)')
    error_norm = 1
    while error_norm > prec:

        cspinorb1 = spinorb1.complex_conj()
        cspinorb2 = spinorb2.complex_conj()
        
        # Definition of different densities
        n_11 = cspinorb1.overlap_density(spinorb1, prec)
        n_12 = cspinorb1.overlap_density(spinorb2, prec)
        n_21 = cspinorb2.overlap_density(spinorb1, prec)
        n_22 = cspinorb2.overlap_density(spinorb2, prec)

        # Definition of Poisson operator
        Pua = vp.PoissonOperator(mra, prec)

        # Defintion of J
        J_Re = Pua(n_11.real + n_22.real) * (4 * np.pi)
        J_Im = Pua(n_11.imag + n_22.imag) * (4 * np.pi)

        J = cf.complex_fcn()
        J.real = J_Re
        J.imag = J_Im
        #print('J', J)

        # Definition of Kx
        K1a_Re = Pua(n_21.real) * (4 * np.pi)
        K1a_Im = Pua(n_21.imag) * (4 * np.pi)
        K1b_Re = Pua(n_11.real) * (4 * np.pi)
        K1b_Im = Pua(n_11.imag) * (4 * np.pi)
        K2a_Re = Pua(n_12.real) * (4 * np.pi)
        K2a_Im = Pua(n_12.imag) * (4 * np.pi)
        K2b_Re = Pua(n_22.real) * (4 * np.pi)
        K2b_Im = Pua(n_22.imag) * (4 * np.pi)
        #print('K1a', K1a)
        #print('K2b', K2b)

        K1a = cf.complex_fcn()
        K1a.real = K1a_Re
        K1a.imag = K1a_Im
        
        K1b = cf.complex_fcn()
        K1b.real = K1b_Re
        K1b.imag = K1b_Im 
        
        K2a = cf.complex_fcn()
        K2a.real = K2a_Re
        K2a.imag = K2a_Im
        
        K2b = cf.complex_fcn()
        K2b.real = K2b_Re
        K2b.imag = K2b_Im


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
        J_spinorb1  = orb.apply_complex_potential(1.0, J, spinorb1, prec)
        J_spinorb2  = orb.apply_complex_potential(1.0, J, spinorb2, prec)        


        Ka_spinorb1  = orb.apply_complex_potential(1.0, K1a, spinorb2, prec)
        Kb_spinorb1  = orb.apply_complex_potential(1.0, K1b, spinorb1, prec)
        Ka_spinorb2  = orb.apply_complex_potential(1.0, K2a, spinorb1, prec)
        Kb_spinorb2  = orb.apply_complex_potential(1.0, K2b, spinorb2, prec)


        K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
        K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
        #print('K_spinorb1', K_spinorb1)


        E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
        E_H12, imag_H12 = spinorb1.dot(J_spinorb2)
        E_H21, imag_H21 = spinorb2.dot(J_spinorb1)
        E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
             
             
        E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
        E_K12, imag_K12 = spinorb1.dot(K_spinorb2)
        E_K21, imag_K21 = spinorb2.dot(K_spinorb1)
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
        GJ_Re0 = Pua(cspinorb1_alpha10.real * cspinorb2_alpha20.real) * (4.0 * np.pi)
        GJ_Re1 = Pua(cspinorb1_alpha11.real * cspinorb2_alpha21.real) * (4.0 * np.pi)
        GJ_Re2 = Pua(cspinorb1_alpha12.real * cspinorb2_alpha22.real) * (4.0 * np.pi)
        
        GJ_Im0 = Pua(cspinorb1_alpha10.imag * cspinorb2_alpha20.imag) * (4.0 * np.pi)
        GJ_Im1 = Pua(cspinorb1_alpha11.imag * cspinorb2_alpha21.imag) * (4.0 * np.pi)
        GJ_Im2 = Pua(cspinorb1_alpha12.imag * cspinorb2_alpha22.imag) * (4.0 * np.pi)

        #J gaunt vector
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
        GK1a_Re0 = Pua(cspinorb2_alpha10.real * cspinorb1_alpha20.real) * (4.0 * np.pi)
        GK1a_Re1 = Pua(cspinorb2_alpha11.real * cspinorb1_alpha21.real) * (4.0 * np.pi)
        GK1a_Re2 = Pua(cspinorb2_alpha12.real * cspinorb1_alpha22.real) * (4.0 * np.pi) 

        GK1b_Re0 = Pua(cspinorb1_alpha10.real * cspinorb2_alpha20.real) * (4.0 * np.pi)
        GK1b_Re1 = Pua(cspinorb1_alpha11.real * cspinorb2_alpha21.real) * (4.0 * np.pi)
        GK1b_Re2 = Pua(cspinorb1_alpha12.real * cspinorb2_alpha22.real) * (4.0 * np.pi)

        GK2a_Re0 = Pua(cspinorb1_alpha20.real * cspinorb2_alpha10.real) * (4.0 * np.pi)
        GK2a_Re1 = Pua(cspinorb1_alpha21.real * cspinorb2_alpha11.real) * (4.0 * np.pi)
        GK2a_Re2 = Pua(cspinorb1_alpha22.real * cspinorb2_alpha12.real) * (4.0 * np.pi)

        GK2b_Re0 = Pua(cspinorb2_alpha20.real * cspinorb2_alpha10.real) * (4.0 * np.pi)
        GK2b_Re1 = Pua(cspinorb2_alpha21.real * cspinorb2_alpha11.real) * (4.0 * np.pi)
        GK2b_Re2 = Pua(cspinorb2_alpha22.real * cspinorb2_alpha12.real) * (4.0 * np.pi)

        
        GK1a_Im0 = Pua(cspinorb2_alpha10.imag * cspinorb1_alpha20.imag) * (4.0 * np.pi)
        GK1a_Im1 = Pua(cspinorb2_alpha11.imag * cspinorb1_alpha21.imag) * (4.0 * np.pi)
        GK1a_Im2 = Pua(cspinorb2_alpha12.imag * cspinorb1_alpha22.imag) * (4.0 * np.pi)

        GK1b_Im0 = Pua(cspinorb1_alpha10.imag * cspinorb2_alpha20.imag) * (4.0 * np.pi)
        GK1b_Im1 = Pua(cspinorb1_alpha11.imag * cspinorb2_alpha21.imag) * (4.0 * np.pi)
        GK1b_Im2 = Pua(cspinorb1_alpha12.imag * cspinorb2_alpha22.imag) * (4.0 * np.pi)
        
        GK2a_Im0 = Pua(cspinorb1_alpha20.imag * cspinorb2_alpha10.imag) * (4.0 * np.pi)
        GK2a_Im1 = Pua(cspinorb1_alpha21.imag * cspinorb2_alpha11.imag) * (4.0 * np.pi)
        GK2a_Im2 = Pua(cspinorb1_alpha22.imag * cspinorb2_alpha12.imag) * (4.0 * np.pi)
        
        GK2b_Im0 = Pua(cspinorb2_alpha20.imag * cspinorb2_alpha10.imag) * (4.0 * np.pi)
        GK2b_Im1 = Pua(cspinorb2_alpha21.imag * cspinorb2_alpha11.imag) * (4.0 * np.pi)
        GK2b_Im2 = Pua(cspinorb2_alpha22.imag * cspinorb2_alpha12.imag) * (4.0 * np.pi)
        
        

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


        # Calculation of necessary potential contributions to Helmholtz
        VG10 = orb.apply_complex_potential(1.0, GJ_0, spinorb1, prec)
        VG11 = orb.apply_complex_potential(1.0, GJ_1, spinorb1, prec)
        VG12 = orb.apply_complex_potential(1.0, GJ_2, spinorb1, prec)
        GJ_spinorb1 = VG10 + VG11 + VG12

        VG20 = orb.apply_complex_potential(1.0, GJ_0, spinorb2, prec)
        VG21 = orb.apply_complex_potential(1.0, GJ_1, spinorb2, prec)
        VG22 = orb.apply_complex_potential(1.0, GJ_2, spinorb2, prec)
        GJ_spinorb2 = VG20 + VG21 + VG22

        GKa_spinorb1_0  = orb.apply_complex_potential(1.0, GK1a_0, spinorb2, prec)
        GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, spinorb2, prec)
        GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, spinorb2, prec)
        GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, spinorb1, prec)
        GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, spinorb1, prec)
        GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, spinorb1, prec)
    
        GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, spinorb1, prec)
        GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, spinorb1, prec)
        GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, spinorb1, prec)
        GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, spinorb2, prec)
        GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, spinorb2, prec)
        GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, spinorb2, prec)

        GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
        GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
        #print('K_spinorb1', K_spinorb1)

        #Jij Gaunt
        E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
        E_GH12, imag_GH12 = spinorb1.dot(GJ_spinorb2)
        E_GH21, imag_GH21 = spinorb2.dot(GJ_spinorb1)
        E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)

        #Kij Gaunt
        E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
        E_GK12, imag_GK12 = spinorb1.dot(GK_spinorb2)
        E_GK21, imag_GK21 = spinorb2.dot(GK_spinorb1)
        E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)

        # (hd+V)ij
        energy_11, imag_11 = spinorb1.dot(add_psi_1)
        energy_12, imag_12 = spinorb1.dot(add_psi_2)
        energy_21, imag_21 = spinorb2.dot(add_psi_1)
        energy_22, imag_22 = spinorb2.dot(add_psi_2)

        # (hd + V + J - K + JG - KG)ij
        F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
        F_12 = energy_12 + E_H12 - E_K12 - E_GH12 + E_GK12
        F_21 = energy_21 + E_H21 - E_K21 - E_GH21 + E_GK21
        F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
        
        # Orbital Energy
        print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
        print('Energy_Spin_Orbit_2', F_22 - light_speed**2)

        # Total Energy 
        E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 - E_GH22 + E_GK11 + E_GK22)
        print('E_total(Coulomb-Gaunt) approximiation', E_tot_JK - (2.0 *light_speed**2))

        #Right Hand Side
        V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - GJ_spinorb1 + GK_spinorb1 - (F_12 * spinorb2)
        V_J_K_spinorb2 = v_psi_2 + J_spinorb2 - K_spinorb2 - GJ_spinorb1 + GK_spinorb1 - (F_21 * spinorb1)
 
        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, F_11, prec)
        tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, F_22, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F_11)
        new_orbital_1 *= 0.5 / light_speed ** 2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, F_22)
        new_orbital_2 *= 0.5 / light_speed ** 2
        new_orbital_2.normalize()
 
        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        deltasq1 = delta_psi_1.squaredNorm()
        deltasq2 = delta_psi_2.squaredNorm()
        orbital_error_sq = deltasq1 + deltasq2
        error_norm = np.sqrt(orbital_error_sq)
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
    cspinorb1 = spinorb1.complex_conj()
    cspinorb2 = spinorb2.complex_conj()
        
    # Definition of different densities
    n_11 = cspinorb1.overlap_density(spinorb1, prec)
    n_12 = cspinorb1.overlap_density(spinorb2, prec)
    n_21 = cspinorb2.overlap_density(spinorb1, prec)
    n_22 = cspinorb2.overlap_density(spinorb2, prec)

    # Definition of Poisson operator
    Pua = vp.PoissonOperator(mra, prec)

    # Defintion of J
    J_Re = Pua(n_11.real + n_22.real) * (4 * np.pi)
    J_Im = Pua(n_11.imag + n_22.imag) * (4 * np.pi)

    J = cf.complex_fcn()
    J.real = J_Re
    J.imag = J_Im
    #print('J', J)

    # Definition of Kx
    K1a_Re = Pua(n_21.real) * (4 * np.pi)
    K1a_Im = Pua(n_21.imag) * (4 * np.pi)
    K1b_Re = Pua(n_11.real) * (4 * np.pi)
    K1b_Im = Pua(n_11.imag) * (4 * np.pi)
    K2a_Re = Pua(n_12.real) * (4 * np.pi)
    K2a_Im = Pua(n_12.imag) * (4 * np.pi)
    K2b_Re = Pua(n_22.real) * (4 * np.pi)
    K2b_Im = Pua(n_22.imag) * (4 * np.pi)
    #print('K1a', K1a)
    #print('K2b', K2b)

    K1a = cf.complex_fcn()
    K1a.real = K1a_Re
    K1a.imag = K1a_Im
        
    K1b = cf.complex_fcn()
    K1b.real = K1b_Re
    K1b.imag = K1b_Im 
        
    K2a = cf.complex_fcn()
    K2a.real = K2a_Re
    K2a.imag = K2a_Im
        
    K2b = cf.complex_fcn()
    K2b.real = K2b_Re
    K2b.imag = K2b_Im


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
    J_spinorb1  = orb.apply_complex_potential(1.0, J, spinorb1, prec)
    J_spinorb2  = orb.apply_complex_potential(1.0, J, spinorb2, prec) 

    Ka_spinorb1  = orb.apply_complex_potential(1.0, K1a, spinorb2, prec)
    Kb_spinorb1  = orb.apply_complex_potential(1.0, K1b, spinorb1, prec)
    Ka_spinorb2  = orb.apply_complex_potential(1.0, K2a, spinorb1, prec)
    Kb_spinorb2  = orb.apply_complex_potential(1.0, K2b, spinorb2, prec)
    K_spinorb1 = Ka_spinorb1 + Kb_spinorb1
    K_spinorb2 = Ka_spinorb2 + Kb_spinorb2
    #print('K_spinorb1', K_spinorb1)
    

    E_H11, imag_H11 = spinorb1.dot(J_spinorb1)
    E_H12, imag_H12 = spinorb1.dot(J_spinorb2)
    E_H21, imag_H21 = spinorb2.dot(J_spinorb1)
    E_H22, imag_H22 = spinorb2.dot(J_spinorb2)
         
         
    E_K11, imag_K11 = spinorb1.dot(K_spinorb1)
    E_K12, imag_K12 = spinorb1.dot(K_spinorb2)
    E_K21, imag_K21 = spinorb2.dot(K_spinorb1)
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
    GJ_Re0 = Pua(cspinorb1_alpha10.real * cspinorb2_alpha20.real) * (4.0 * np.pi)    
    GJ_Re1 = Pua(cspinorb1_alpha11.real * cspinorb2_alpha21.real) * (4.0 * np.pi)
    GJ_Re2 = Pua(cspinorb1_alpha12.real * cspinorb2_alpha22.real) * (4.0 * np.pi)
    
    GJ_Im0 = Pua(cspinorb1_alpha10.imag * cspinorb2_alpha20.imag) * (4.0 * np.pi)
    GJ_Im1 = Pua(cspinorb1_alpha11.imag * cspinorb2_alpha21.imag) * (4.0 * np.pi)
    GJ_Im2 = Pua(cspinorb1_alpha12.imag * cspinorb2_alpha22.imag) * (4.0 * np.pi)
    
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
    GK1a_Re0 = Pua(cspinorb2_alpha10.real * cspinorb1_alpha20.real) * (4.0 * np.pi)
    GK1a_Re1 = Pua(cspinorb2_alpha11.real * cspinorb1_alpha21.real) * (4.0 * np.pi)
    GK1a_Re2 = Pua(cspinorb2_alpha12.real * cspinorb1_alpha22.real) * (4.0 * np.pi)
    
    GK1b_Re0 = Pua(cspinorb1_alpha10.real * cspinorb2_alpha20.real) * (4.0 * np.pi)
    GK1b_Re1 = Pua(cspinorb1_alpha11.real * cspinorb2_alpha21.real) * (4.0 * np.pi)
    GK1b_Re2 = Pua(cspinorb1_alpha12.real * cspinorb2_alpha22.real) * (4.0 * np.pi)
    
    GK2a_Re0 = Pua(cspinorb1_alpha20.real * cspinorb2_alpha10.real) * (4.0 * np.pi)
    GK2a_Re1 = Pua(cspinorb1_alpha21.real * cspinorb2_alpha11.real) * (4.0 * np.pi)
    GK2a_Re2 = Pua(cspinorb1_alpha22.real * cspinorb2_alpha12.real) * (4.0 * np.pi)
    
    GK2b_Re0 = Pua(cspinorb2_alpha20.real * cspinorb2_alpha10.real) * (4.0 * np.pi)
    GK2b_Re1 = Pua(cspinorb2_alpha21.real * cspinorb2_alpha11.real) * (4.0 * np.pi)
    GK2b_Re2 = Pua(cspinorb2_alpha22.real * cspinorb2_alpha12.real) * (4.0 * np.pi)
    
    GK1a_Im0 = Pua(cspinorb2_alpha10.imag * cspinorb1_alpha20.imag) * (4.0 * np.pi)
    GK1a_Im1 = Pua(cspinorb2_alpha11.imag * cspinorb1_alpha21.imag) * (4.0 * np.pi)
    GK1a_Im2 = Pua(cspinorb2_alpha12.imag * cspinorb1_alpha22.imag) * (4.0 * np.pi)
    
    GK1b_Im0 = Pua(cspinorb1_alpha10.imag * cspinorb2_alpha20.imag) * (4.0 * np.pi)
    GK1b_Im1 = Pua(cspinorb1_alpha11.imag * cspinorb2_alpha21.imag) * (4.0 * np.pi)
    GK1b_Im2 = Pua(cspinorb1_alpha12.imag * cspinorb2_alpha22.imag) * (4.0 * np.pi)
    
    GK2a_Im0 = Pua(cspinorb1_alpha20.imag * cspinorb2_alpha10.imag) * (4.0 * np.pi)
    GK2a_Im1 = Pua(cspinorb1_alpha21.imag * cspinorb2_alpha11.imag) * (4.0 * np.pi)
    GK2a_Im2 = Pua(cspinorb1_alpha22.imag * cspinorb2_alpha12.imag) * (4.0 * np.pi)
    
    GK2b_Im0 = Pua(cspinorb2_alpha20.imag * cspinorb2_alpha10.imag) * (4.0 * np.pi)
    GK2b_Im1 = Pua(cspinorb2_alpha21.imag * cspinorb2_alpha11.imag) * (4.0 * np.pi)
    GK2b_Im2 = Pua(cspinorb2_alpha22.imag * cspinorb2_alpha12.imag) * (4.0 * np.pi)

        
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
    VG10 = orb.apply_complex_potential(1.0, GJ_0, spinorb1, prec)
    VG11 = orb.apply_complex_potential(1.0, GJ_1, spinorb1, prec)
    VG12 = orb.apply_complex_potential(1.0, GJ_2, spinorb1, prec)
    GJ_spinorb1 = VG10 + VG11 + VG12
    

    VG20 = orb.apply_complex_potential(1.0, GJ_0, spinorb2, prec)
    VG21 = orb.apply_complex_potential(1.0, GJ_1, spinorb2, prec)
    VG22 = orb.apply_complex_potential(1.0, GJ_2, spinorb2, prec)
    GJ_spinorb2 = VG20 + VG21 + VG22
    
    GKa_spinorb1_0  = orb.apply_complex_potential(1.0, GK1a_0, spinorb2, prec)
    GKa_spinorb1_1  = orb.apply_complex_potential(1.0, GK1a_1, spinorb2, prec)
    GKa_spinorb1_2  = orb.apply_complex_potential(1.0, GK1a_2, spinorb2, prec)
    GKb_spinorb1_0  = orb.apply_complex_potential(1.0, GK1b_0, spinorb1, prec)
    GKb_spinorb1_1  = orb.apply_complex_potential(1.0, GK1b_1, spinorb1, prec)
    GKb_spinorb1_2  = orb.apply_complex_potential(1.0, GK1b_2, spinorb1, prec)

    GKa_spinorb2_0  = orb.apply_complex_potential(1.0, GK2a_0, spinorb1, prec)
    GKa_spinorb2_1  = orb.apply_complex_potential(1.0, GK2a_1, spinorb1, prec)
    GKa_spinorb2_2  = orb.apply_complex_potential(1.0, GK2a_2, spinorb1, prec)
    GKb_spinorb2_0  = orb.apply_complex_potential(1.0, GK2b_0, spinorb2, prec)
    GKb_spinorb2_1  = orb.apply_complex_potential(1.0, GK2b_1, spinorb2, prec)
    GKb_spinorb2_2  = orb.apply_complex_potential(1.0, GK2b_2, spinorb2, prec)
    
    GK_spinorb1 = GKa_spinorb1_0 + GKb_spinorb1_0 + GKa_spinorb1_1 + GKb_spinorb1_1 + GKa_spinorb1_2 + GKb_spinorb1_2
    GK_spinorb2 = GKa_spinorb2_0 + GKb_spinorb2_0 + GKa_spinorb2_1 + GKb_spinorb2_1 + GKa_spinorb2_2 + GKb_spinorb2_2
    
    E_GH11, imag_GH11 = spinorb1.dot(GJ_spinorb1)
    E_GH12, imag_GH12 = spinorb1.dot(GJ_spinorb2)
    E_GH21, imag_GH21 = spinorb2.dot(GJ_spinorb1)
    E_GH22, imag_GH22 = spinorb2.dot(GJ_spinorb2)
         
    E_GK11, imag_GK11 = spinorb1.dot(GK_spinorb1)
    E_GK12, imag_GK12 = spinorb1.dot(GK_spinorb2)
    E_GK21, imag_GK21 = spinorb2.dot(GK_spinorb1)
    E_GK22, imag_GK22 = spinorb2.dot(GK_spinorb2)
    

    # Orbital Energy calculation
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_12, imag_12 = spinorb1.dot(add_psi_2)
    energy_21, imag_21 = spinorb2.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)
    

    # Calculate Fij Fock matrix
    F_11 = energy_11 + E_H11 - E_K11 - E_GH11 + E_GK11
    F_12 = energy_12 + E_H12 - E_K12 - E_GH12 + E_GK12
    F_21 = energy_21 + E_H21 - E_K21 - E_GH21 + E_GK21
    F_22 = energy_22 + E_H22 - E_K22 - E_GH22 + E_GK22
    

    # Orbital Energy
    print('Energy_Spin_Orbit_1', F_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', F_22 - light_speed**2)
    

    # Total Energy 
    E_tot_JK = F_11 + F_22 - 0.5 * (E_H11 + E_H22 - E_K11 - E_K22 - E_GH11 - E_GH22 + E_GK11 + E_GK22)
    print('E_total(Coulomb-Gaunt) approximiation', E_tot_JK - (2.0 *light_speed**2))
################################################################################################################################################
