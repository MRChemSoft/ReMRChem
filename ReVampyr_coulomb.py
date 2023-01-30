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
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-6,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='coulomb_HFYGB',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt', 'gaunt-test'], 'Please, specify coulgau in a rigth way: coulomb or gaunt'

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
print('Define V Potential', args.potential, 'DONE')

P = vp.PoissonOperator(mra, prec)

#############################START WITH CALCULATION###################################
if args.coulgau == 'coulomb':
    print('Hartree-Fock (Coulomb interaction)')
    error_norm = 1
    compute_last_energy = False
    
    while (error_norm > prec or compute_last_energy):
        n_11 = spinorb1.overlap_density(spinorb1, prec)
        spinorb2 = spinorb1.ktrs()

        # Definition of two electron operators
        B11    = P(n_11.real) * (4 * np.pi)

        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
        hd_11_r, hd_11_i = spinorb1.dot(hd_psi_1)
        hd_12_r, hd_12_i = spinorb1.dot(hd_psi_2)
        hd_21_r, hd_21_i = spinorb2.dot(hd_psi_1)
        hd_22_r, hd_22_i = spinorb2.dot(hd_psi_2)
        hd_mat = np.array([[hd_11_r + hd_11_i * 1j, hd_12_r + hd_12_i * 1j],
                           [hd_21_r + hd_21_i * 1j, hd_22_r + hd_22_i * 1j]])

        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        V_11_r, V_11_i = spinorb1.dot(v_psi_1)
        v_mat = np.array([[ V_11_r + V_11_i * 1j, 0],
                          [ 0,                    V_11_r + V_11_i * 1j]])
        # Calculation of two electron terms
        J2_phi1 = orb.apply_potential(1.0, B11, spinorb1, prec)
        JmK_phi1 = J2_phi1 # K part is zero for 2e system in GS
        JmK_11_r, JmK_11_i = spinorb1.dot(JmK_phi1)
        JmK = np.array([[ JmK_11_r + JmK_11_i * 1j, 0],
                        [ 0,                        JmK_11_r + JmK_11_i * 1j]])

        hd_V_mat = hd_mat + v_mat 

        print('HD_V MATRIX\n', hd_V_mat)
         # Calculate Fij Fock matrix
        Fmat = hd_V_mat + JmK
        print('FOCK MATRIX\n', Fmat)
        eps1 = Fmat[0,0].real
        eps2 = Fmat[1,1].real
        
        # Orbital Energy
        print('Energy_Spin_Orbit_1', eps1 - light_speed**2)
 #       print('Energy_Spin_Orbit_2', eps2 - light_speed**2)

        # Total Energy 
        E_tot_JK = np.trace(Fmat) - 0.5 * (np.trace(JmK))
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

        if(compute_last_energy):
            break

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1

        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, eps1, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, eps1, der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_1.crop(prec)       

        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        deltasq1 = delta_psi_1.squaredNorm()
        error_norm = np.sqrt(deltasq1)
        print('Orbital_Error norm', error_norm)
        spinorb1 = new_orbital_1
        if(error_norm < prec):
            compute_last_energy = True
        print('ORBITAL\n', spinorb1)

    ##########

    
    if(False):
        #Definition of alpha vectors for each orbital
        alpha1_0 =  spinorb1.alpha(0)
        alpha1_1 =  spinorb1.alpha(1)
        alpha1_2 =  spinorb1.alpha(2)

        alpha2_0 =  spinorb2.alpha(0)
        alpha2_1 =  spinorb2.alpha(1)
        alpha2_2 =  spinorb2.alpha(2)    
        #Defintion of orbital * alpha(orbital)
        cphi1_alpha1_0 = spinorb1.overlap_density(alpha1_0, prec)
        cphi1_alpha1_1 = spinorb1.overlap_density(alpha1_1, prec)
        cphi1_alpha1_2 = spinorb1.overlap_density(alpha1_2, prec)
   
        cphi1_alpha2_0 = spinorb1.overlap_density(alpha2_0, prec)
        cphi1_alpha2_1 = spinorb1.overlap_density(alpha2_1, prec)
        cphi1_alpha2_2 = spinorb1.overlap_density(alpha2_2, prec)
        
        cphi2_alpha1_0 = spinorb2.overlap_density(alpha1_0, prec)
        cphi2_alpha1_1 = spinorb2.overlap_density(alpha1_1, prec)
        cphi2_alpha1_2 = spinorb2.overlap_density(alpha1_2, prec)
        
        cphi2_alpha2_0 = spinorb2.overlap_density(alpha2_0, prec)
        cphi2_alpha2_1 = spinorb2.overlap_density(alpha2_1, prec)
        cphi2_alpha2_2 = spinorb2.overlap_density(alpha2_2, prec)
        
        #Definition of Gaunt two electron operators      
        BG11_Re0 = P(cphi1_alpha1_0.real) * (4.0 * np.pi)
        BG11_Re1 = P(cphi1_alpha1_1.real) * (4.0 * np.pi)
        BG11_Re2 = P(cphi1_alpha1_2.real) * (4.0 * np.pi)
        BG11_Im0 = P(cphi1_alpha1_0.imag) * (4.0 * np.pi)
        BG11_Im1 = P(cphi1_alpha1_1.imag) * (4.0 * np.pi)
        BG11_Im2 = P(cphi1_alpha1_2.imag) * (4.0 * np.pi)
        
        BG22_Re0 = P(cphi2_alpha2_0.real) * (4.0 * np.pi)
        BG22_Re1 = P(cphi2_alpha2_1.real) * (4.0 * np.pi)
        BG22_Re2 = P(cphi2_alpha2_2.real) * (4.0 * np.pi)
        BG22_Im0 = P(cphi2_alpha2_0.imag) * (4.0 * np.pi)
        BG22_Im1 = P(cphi2_alpha2_1.imag) * (4.0 * np.pi)
        BG22_Im2 = P(cphi2_alpha2_2.imag) * (4.0 * np.pi)
        
        BG12_Re0 = P(cphi1_alpha2_0.real) * (4.0 * np.pi)
        BG12_Re1 = P(cphi1_alpha2_1.real) * (4.0 * np.pi)
        BG12_Re2 = P(cphi1_alpha2_2.real) * (4.0 * np.pi) 
        BG12_Im0 = P(cphi1_alpha2_0.imag) * (4.0 * np.pi)
        BG12_Im1 = P(cphi1_alpha2_1.imag) * (4.0 * np.pi)
        BG12_Im2 = P(cphi1_alpha2_2.imag) * (4.0 * np.pi)
        
        BG21_Re0 = P(cphi2_alpha1_0.real) * (4.0 * np.pi)
        BG21_Re1 = P(cphi2_alpha1_1.real) * (4.0 * np.pi)
        BG21_Re2 = P(cphi2_alpha1_2.real) * (4.0 * np.pi)
        BG21_Im0 = P(cphi2_alpha1_0.imag) * (4.0 * np.pi)
        BG21_Im1 = P(cphi2_alpha1_1.imag) * (4.0 * np.pi)
        BG21_Im2 = P(cphi2_alpha1_2.imag) * (4.0 * np.pi)
        
        BG11_0 = cf.complex_fcn()
        BG11_0.real = BG11_Re0
        BG11_0.imag = BG11_Im0
        
        BG11_1 = cf.complex_fcn()
        BG11_1.real = BG11_Re1
        BG11_1.imag = BG11_Im1
        
        BG11_2 = cf.complex_fcn()
        BG11_2.real = BG11_Re2
        BG11_2.imag = BG11_Im2
        
        BG22_0 = cf.complex_fcn()
        BG22_0.real = BG22_Re0
        BG22_0.imag = BG22_Im0
        
        BG22_1 = cf.complex_fcn()
        BG22_1.real = BG22_Re1
        BG22_1.imag = BG22_Im1
        
        BG22_2 = cf.complex_fcn()
        BG22_2.real = BG22_Re2
        BG22_2.imag = BG22_Im2
        
        BG12_0 = cf.complex_fcn()
        BG12_0.real = BG12_Re0
        BG12_0.imag = BG12_Im0
        
        BG12_1 = cf.complex_fcn()
        BG12_1.real = BG12_Re1
        BG12_1.imag = BG12_Im1
        
        BG12_2 = cf.complex_fcn()
        BG12_2.real = BG12_Re2
        BG12_2.imag = BG12_Im2
        
        BG21_0 = cf.complex_fcn()
        BG21_0.real = BG21_Re0
        BG21_0.imag = BG21_Im0
        
        BG21_1 = cf.complex_fcn()
        BG21_1.real = BG21_Re1
        BG21_1.imag = BG21_Im1
        
        BG21_2 = cf.complex_fcn()
        BG21_2.real = BG21_Re2
        BG21_2.imag = BG21_Im2
        
        # Calculation of Gaunt two electron terms
        VGJ1_0 = orb.apply_complex_potential(1.0, BG11_0, alpha2_0, prec)
        VGJ1_1 = orb.apply_complex_potential(1.0, BG11_1, alpha2_1, prec)
        VGJ1_2 = orb.apply_complex_potential(1.0, BG11_2, alpha2_2, prec)
        GJ1_alpha2 = VGJ1_0 + VGJ1_1 + VGJ1_2
        
        VGJ2_0 = orb.apply_complex_potential(1.0, BG22_0, alpha1_0, prec)
        VGJ2_1 = orb.apply_complex_potential(1.0, BG22_1, alpha1_1, prec)
        VGJ2_2 = orb.apply_complex_potential(1.0, BG22_2, alpha1_2, prec)
        GJ2_alpha1 = VGJ2_0 + VGJ2_1 + VGJ2_2
        
        VGK1_0 = orb.apply_complex_potential(1.0, BG12_0, alpha1_0, prec)
        VGK1_1 = orb.apply_complex_potential(1.0, BG12_1, alpha1_1, prec)
        VGK1_2 = orb.apply_complex_potential(1.0, BG12_2, alpha1_2, prec)
        GK1_alpha2 = VGK1_0 + VGK1_1 + VGK1_2
        
        VGK2_0 = orb.apply_complex_potential(1.0, BG21_0, alpha2_0, prec)
        VGK2_1 = orb.apply_complex_potential(1.0, BG21_1, alpha2_1, prec)
        VGK2_2 = orb.apply_complex_potential(1.0, BG21_2, alpha2_2, prec)
        GK2_alpha1 = VGK2_0 + VGK2_1 + VGK2_2
        
        GJmK_phi1 = GJ2_alpha1 - GK2_alpha1
        GJmK_phi2 = GJ1_alpha2 - GK1_alpha2
        
        GJmK_11_r, GJmK_11_i = spinorb1.dot(GJmK_phi1)
        GJmK_12_r, GJmK_12_i = spinorb1.dot(GJmK_phi2)
        GJmK_21_r, GJmK_21_i = spinorb2.dot(GJmK_phi1)
        GJmK_22_r, GJmK_22_i = spinorb2.dot(GJmK_phi2)
        
        print('GJmK_11_r', GJmK_11_r)
        print('GJmK_12_r', GJmK_12_r)
        print('GJmK_21_r', GJmK_21_r)
        print('GJmK_22_r', GJmK_22_r)
        print('GJmK_11_i', GJmK_11_i)
        print('GJmK_12_i', GJmK_12_i)
        print('GJmK_21_i', GJmK_21_i)
        print('GJmK_22_i', GJmK_22_i)
        print('E_C_G', E_tot_JK - GJmK_11_r - (2.0 *light_speed**2))
        
