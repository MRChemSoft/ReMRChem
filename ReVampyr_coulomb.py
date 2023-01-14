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
 
