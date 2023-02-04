########## Define Enviroment #################
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import nuclear_potential as nucpot
from orbital4c import r3m as r3m
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1 

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
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt', 'gaunt-test'], 'Please, specify coulgau in a rigth way: coulomb or gaunt'

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['ABGV', 'PH', 'BS'], 'Please, specify the type of derivative'


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

#############################START WITH COULOMB CALCULATION###################################
if args.coulgau == 'coulomb':
    print('Hartree-Fock (Coulomb interaction)')
    error_norm = 1
    compute_last_energy = False

    while (error_norm > prec or compute_last_energy):
        spinorb2 = spinorb1.ktrs()
        n_22 = spinorb2.overlap_density(spinorb2, prec)

        # Definition of two electron operators
        B22    = P(n_22.real) * (4 * np.pi)

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

        JmK_phi1 = J2_phi1 

        JmK_11_r, JmK_11_i = spinorb1.dot(JmK_phi1)

        JmK = np.array([[ JmK_11_r + JmK_11_i * 1j , 0.0],
                        [ 0.0 , JmK_11_r + JmK_11_i * 1j]])

        # Orbital Energy calculation
        hd_V_11_r, hd_V_11_i = spinorb1.dot(add_psi_1)
        hd_V_12_r, hd_V_12_i = spinorb1.dot(add_psi_2)
        hd_V_21_r, hd_V_21_i = spinorb2.dot(add_psi_1)
        hd_V_22_r, hd_V_22_i = spinorb2.dot(add_psi_2)

        hd_V = np.array([[ hd_V_11_r + hd_V_11_i * 1j , hd_V_12_r + hd_V_12_i * 1j],
                         [ hd_V_21_r + hd_V_21_i * 1j , hd_V_22_r + hd_V_22_i * 1j]])

        # Calculate Fij Fock matrix
        Fmat = hd_V + JmK
        eps1 = Fmat[0,0].real
        eps2 = Fmat[1,1].real

        # Orbital Energy
        print('Energy_Spin_Orbit_1', eps1 - light_speed**2)

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

        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        deltasq1 = delta_psi_1.squaredNorm()
        error_norm = np.sqrt(deltasq1)
        print('Orbital_Error norm', error_norm)
        spinorb1 = new_orbital_1
        if (error_norm < prec):
            compute_last_energy = True

        spinorb1.crop(prec)

#############################END COULOMB & START WITH GAUNT###################################
if args.coulgau == 'gaunt':
    spinorb2 = spinorb1.ktrs() 
    
    #Definition of alpha vectors for each orbital
    alpha1_0 =  spinorb1.alpha(0)
    alpha1_1 =  spinorb1.alpha(1)
    alpha1_2 =  spinorb1.alpha(2)

    alpha2_0 =  spinorb2.alpha(0)
    alpha2_1 =  spinorb2.alpha(1)
    alpha2_2 =  spinorb2.alpha(2)

    #Defintion of orbital * alpha(orbital)
    cphi2_alpha1_0 = spinorb2.overlap_density(alpha1_0, prec)
    cphi2_alpha1_1 = spinorb2.overlap_density(alpha1_1, prec)
    cphi2_alpha1_2 = spinorb2.overlap_density(alpha1_2, prec)
   
    cphi2_alpha2_0 = spinorb2.overlap_density(alpha2_0, prec)
    cphi2_alpha2_1 = spinorb2.overlap_density(alpha2_1, prec)
    cphi2_alpha2_2 = spinorb2.overlap_density(alpha2_2, prec)
    
    #Definition of Gaunt two electron operators       
    BG22_Re0 = P(cphi2_alpha2_0.real) * (4.0 * np.pi)
    BG22_Re1 = P(cphi2_alpha2_1.real) * (4.0 * np.pi)
    BG22_Re2 = P(cphi2_alpha2_2.real) * (4.0 * np.pi)
    BG22_Im0 = P(cphi2_alpha2_0.imag) * (4.0 * np.pi)
    BG22_Im1 = P(cphi2_alpha2_1.imag) * (4.0 * np.pi)
    BG22_Im2 = P(cphi2_alpha2_2.imag) * (4.0 * np.pi)
    
    BG21_Re0 = P(cphi2_alpha1_0.real) * (4.0 * np.pi)
    BG21_Re1 = P(cphi2_alpha1_1.real) * (4.0 * np.pi)
    BG21_Re2 = P(cphi2_alpha1_2.real) * (4.0 * np.pi)
    BG21_Im0 = P(cphi2_alpha1_0.imag) * (4.0 * np.pi)
    BG21_Im1 = P(cphi2_alpha1_1.imag) * (4.0 * np.pi)
    BG21_Im2 = P(cphi2_alpha1_2.imag) * (4.0 * np.pi)
    
    
    BG22_0 = cf.complex_fcn()
    BG22_0.real = BG22_Re0
    BG22_0.imag = BG22_Im0
    
    BG22_1 = cf.complex_fcn()
    BG22_1.real = BG22_Re1
    BG22_1.imag = BG22_Im1
    
    BG22_2 = cf.complex_fcn()
    BG22_2.real = BG22_Re2
    BG22_2.imag = BG22_Im2
    
    
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
    VGJ2_0 = orb.apply_complex_potential(1.0, BG22_0, alpha1_0, prec)
    VGJ2_1 = orb.apply_complex_potential(1.0, BG22_1, alpha1_1, prec)
    VGJ2_2 = orb.apply_complex_potential(1.0, BG22_2, alpha1_2, prec)
    GJ2_alpha1 = VGJ2_0 + VGJ2_1 + VGJ2_2
    
    VGK2_0 = orb.apply_complex_potential(1.0, BG21_0, alpha2_0, prec)
    VGK2_1 = orb.apply_complex_potential(1.0, BG21_1, alpha2_1, prec)
    VGK2_2 = orb.apply_complex_potential(1.0, BG21_2, alpha2_2, prec)
    GK2_alpha1 = VGK2_0 + VGK2_1 + VGK2_2
    
    GJmK_phi1 = GJ2_alpha1 - GK2_alpha1
    
    GJmK_11_r, GJmK_11_i = spinorb1.dot(GJmK_phi1)

    print('GJmK_11_r', GJmK_11_r)
    #print('E_C_G', E_tot_JK - GJmK_11_r - (2.0 *light_speed**2))

#############################END GAUNT & START WITH GAUGE###################################
if args.coulgau == 'gauge': 
    spinorb2 = spinorb1.ktrs()
    
    #Definition of alpha vectors for each orbital
    alpha1_0 =  spinorb1.alpha(0)
    alpha1_1 =  spinorb1.alpha(1)
    alpha1_2 =  spinorb1.alpha(2)

    alpha2_0 =  spinorb2.alpha(0)
    alpha2_1 =  spinorb2.alpha(1)
    alpha2_2 =  spinorb2.alpha(2)

    #Defintion of orbital * alpha(orbital)
    cphi2_alpha1_0 = spinorb2.overlap_density(alpha1_0, prec)
    cphi2_alpha1_1 = spinorb2.overlap_density(alpha1_1, prec)
    cphi2_alpha1_2 = spinorb2.overlap_density(alpha1_2, prec)
   
    cphi2_alpha2_0 = spinorb2.overlap_density(alpha2_0, prec)
    cphi2_alpha2_1 = spinorb2.overlap_density(alpha2_1, prec)
    cphi2_alpha2_2 = spinorb2.overlap_density(alpha2_2, prec)    
    
    #Definitioin of Gauge operator 
    O = r3m.GaugeOperator(mra, 0.0001, args.box, prec)

    #Definition of Gaunt two electron operators       
    Bgauge22_Re0_xy = O(cphi2_alpha2_0.real, 1, 1, 0) 
    Bgauge22_Re1_xy = O(cphi2_alpha2_1.real, 1, 1, 0) 
    Bgauge22_Re2_xy = O(cphi2_alpha2_2.real, 1, 1, 0) 
    
    Bgauge22_Re0_xz = O(cphi2_alpha2_0.real, 1, 0, 1) 
    Bgauge22_Re1_xz = O(cphi2_alpha2_1.real, 1, 0, 1) 
    Bgauge22_Re2_xz = O(cphi2_alpha2_2.real, 1, 0, 1) 
    
    Bgauge22_Re0_yz = O(cphi2_alpha2_0.real, 0 ,1 ,1) 
    Bgauge22_Re1_yz = O(cphi2_alpha2_1.real, 0 ,1 ,1) 
    Bgauge22_Re2_yz = O(cphi2_alpha2_2.real, 0 ,1 ,1) 
     
    Bgauge22_Im0_xy = O(cphi2_alpha2_0.imag, 1, 1, 0) 
    Bgauge22_Im1_xy = O(cphi2_alpha2_1.imag, 1, 1, 0) 
    Bgauge22_Im2_xy = O(cphi2_alpha2_2.imag, 1, 1, 0) 
  
    Bgauge22_Im0_xz = O(cphi2_alpha2_0.imag, 1, 0, 1) 
    Bgauge22_Im1_xz = O(cphi2_alpha2_1.imag, 1, 0, 1) 
    Bgauge22_Im2_xz = O(cphi2_alpha2_2.imag, 1, 0, 1) 
  
    Bgauge22_Im0_yz = O(cphi2_alpha2_0.imag, 0 ,1 ,1) 
    Bgauge22_Im1_yz = O(cphi2_alpha2_1.imag, 0 ,1 ,1) 
    Bgauge22_Im2_yz = O(cphi2_alpha2_2.imag, 0 ,1 ,1) 
 


    Bgauge21_Re0_xy = O(cphi2_alpha1_0.real, 1, 1, 0) 
    Bgauge21_Re1_xy = O(cphi2_alpha1_1.real, 1, 1, 0) 
    Bgauge21_Re2_xy = O(cphi2_alpha1_2.real, 1, 1, 0) 
   
    Bgauge21_Re0_xz = O(cphi2_alpha1_0.real, 1, 0, 1) 
    Bgauge21_Re1_xz = O(cphi2_alpha1_1.real, 1, 0, 1) 
    Bgauge21_Re2_xz = O(cphi2_alpha1_2.real, 1, 0, 1) 
   
    Bgauge21_Re0_yz = O(cphi2_alpha1_0.real, 0 ,1 ,1) 
    Bgauge21_Re1_yz = O(cphi2_alpha1_1.real, 0 ,1 ,1) 
    Bgauge21_Re2_yz = O(cphi2_alpha1_2.real, 0 ,1 ,1) 
  
    Bgauge21_Im0_xy = O(cphi2_alpha1_0.imag, 1, 1, 0) 
    Bgauge21_Im1_xy = O(cphi2_alpha1_1.imag, 1, 1, 0) 
    Bgauge21_Im2_xy = O(cphi2_alpha1_2.imag, 1, 1, 0) 
 
    Bgauge21_Im0_xz = O(cphi2_alpha1_0.imag, 1, 0, 1) 
    Bgauge21_Im1_xz = O(cphi2_alpha1_1.imag, 1, 0, 1) 
    Bgauge21_Im2_xz = O(cphi2_alpha1_2.imag, 1, 0, 1) 
 
    Bgauge21_Im0_yz = O(cphi2_alpha1_0.imag, 0 ,1 ,1) 
    Bgauge21_Im1_yz = O(cphi2_alpha1_1.imag, 0 ,1 ,1) 
    Bgauge21_Im2_yz = O(cphi2_alpha1_2.imag, 0 ,1 ,1)     
   

    Bgauge22_0_xy = cf.complex_fcn()
    Bgauge22_0_xy.real = Bgauge22_Re0_xy
    Bgauge22_0_xy.imag = Bgauge22_Im0_xy
 
    Bgauge22_1_xy = cf.complex_fcn()
    Bgauge22_1_xy.real = Bgauge22_Re1_xy
    Bgauge22_1_xy.imag = Bgauge22_Im1_xy   

    Bgauge22_2_xy = cf.complex_fcn()
    Bgauge22_2_xy.real = Bgauge22_Re2_xy
    Bgauge22_2_xy.imag = Bgauge22_Im2_xy    
   

    Bgauge22_0_xz = cf.complex_fcn()
    Bgauge22_0_xz.real = Bgauge22_Re0_xz
    Bgauge22_0_xz.imag = Bgauge22_Im0_xz
 
    Bgauge22_1_xz = cf.complex_fcn()
    Bgauge22_1_xz.real = Bgauge22_Re1_xz
    Bgauge22_1_xz.imag = Bgauge22_Im1_xz   

    Bgauge22_2_xz = cf.complex_fcn()
    Bgauge22_2_xz.real = Bgauge22_Re2_xz
    Bgauge22_2_xz.imag = Bgauge22_Im2_xz


    Bgauge22_0_yz = cf.complex_fcn()
    Bgauge22_0_yz.real = Bgauge22_Re0_yz
    Bgauge22_0_yz.imag = Bgauge22_Im0_yz
 
    Bgauge22_1_yz = cf.complex_fcn()
    Bgauge22_1_yz.real = Bgauge22_Re1_yz
    Bgauge22_1_yz.imag = Bgauge22_Im1_yz   

    Bgauge22_2_yz = cf.complex_fcn()
    Bgauge22_2_yz.real = Bgauge22_Re2_yz
    Bgauge22_2_yz.imag = Bgauge22_Im2_yz


    Bgauge21_0_xy = cf.complex_fcn()
    Bgauge21_0_xy.real = Bgauge21_Re0_xy
    Bgauge21_0_xy.imag = Bgauge21_Im0_xy
 
    Bgauge21_1_xy = cf.complex_fcn()
    Bgauge21_1_xy.real = Bgauge21_Re1_xy
    Bgauge21_1_xy.imag = Bgauge21_Im1_xy   

    Bgauge21_2_xy = cf.complex_fcn()
    Bgauge21_2_xy.real = Bgauge21_Re2_xy
    Bgauge21_2_xy.imag = Bgauge21_Im2_xy    
   

    Bgauge21_0_xz = cf.complex_fcn()
    Bgauge21_0_xz.real = Bgauge21_Re0_xz
    Bgauge21_0_xz.imag = Bgauge21_Im0_xz
 
    Bgauge21_1_xz = cf.complex_fcn()
    Bgauge21_1_xz.real = Bgauge21_Re1_xz
    Bgauge21_1_xz.imag = Bgauge21_Im1_xz   

    Bgauge21_2_xz = cf.complex_fcn()
    Bgauge21_2_xz.real = Bgauge21_Re2_xz
    Bgauge21_2_xz.imag = Bgauge21_Im2_xz


    Bgauge21_0_yz = cf.complex_fcn()
    Bgauge21_0_yz.real = Bgauge21_Re0_yz
    Bgauge21_0_yz.imag = Bgauge21_Im0_yz
 
    Bgauge21_1_yz = cf.complex_fcn()
    Bgauge21_1_yz.real = Bgauge21_Re1_yz
    Bgauge21_1_yz.imag = Bgauge21_Im1_yz   

    Bgauge21_2_yz = cf.complex_fcn()
    Bgauge21_2_yz.real = Bgauge21_Re2_yz
    Bgauge21_2_yz.imag = Bgauge21_Im2_yz


    # Calculation of Gaunt two electron terms 
    VgaugeJ2_0_xy = orb.apply_complex_potential(1.0, Bgauge22_0_xy, alpha1_0, prec)
    VgaugeJ2_1_xy = orb.apply_complex_potential(1.0, Bgauge22_1_xy, alpha1_1, prec)
    VgaugeJ2_2_xy = orb.apply_complex_potential(1.0, Bgauge22_2_xy, alpha1_2, prec)
    GaugeJ2_alpha1_xy = VgaugeJ2_0_xy + VgaugeJ2_1_xy + VgaugeJ2_2_xy


    VgaugeJ2_0_xz = orb.apply_complex_potential(1.0, Bgauge22_0_xz, alpha1_0, prec)
    VgaugeJ2_1_xz = orb.apply_complex_potential(1.0, Bgauge22_1_xz, alpha1_1, prec)
    VgaugeJ2_2_xz = orb.apply_complex_potential(1.0, Bgauge22_2_xz, alpha1_2, prec)
    GaugeJ2_alpha1_xz = VgaugeJ2_0_xz + VgaugeJ2_1_xz + VgaugeJ2_2_xz


    VgaugeJ2_0_yz = orb.apply_complex_potential(1.0, Bgauge22_0_yz, alpha1_0, prec)
    VgaugeJ2_1_yz = orb.apply_complex_potential(1.0, Bgauge22_1_yz, alpha1_1, prec)
    VgaugeJ2_2_yz = orb.apply_complex_potential(1.0, Bgauge22_2_yz, alpha1_2, prec)
    GaugeJ2_alpha1_yz = VgaugeJ2_0_yz + VgaugeJ2_1_yz + VgaugeJ2_2_yz


    GaugeJ2_alpha1 = GaugeJ2_alpha1_xy + GaugeJ2_alpha1_xz + GaugeJ2_alpha1_yz


    VgaugeK2_0_xy = orb.apply_complex_potential(1.0, Bgauge21_0_xy, alpha2_0, prec)
    VgaugeK2_1_xy = orb.apply_complex_potential(1.0, Bgauge21_1_xy, alpha2_1, prec)
    VgaugeK2_2_xy = orb.apply_complex_potential(1.0, Bgauge21_2_xy, alpha2_2, prec)
    GaugeK2_alpha1_xy = VgaugeK2_0_xy + VgaugeK2_1_xy + VgaugeK2_2_xy


    VgaugeK2_0_xz = orb.apply_complex_potential(1.0, Bgauge21_0_xz, alpha2_0, prec)
    VgaugeK2_1_xz = orb.apply_complex_potential(1.0, Bgauge21_1_xz, alpha2_1, prec)
    VgaugeK2_2_xz = orb.apply_complex_potential(1.0, Bgauge21_2_xz, alpha2_2, prec)
    GaugeK2_alpha1_xz = VgaugeK2_0_xz + VgaugeK2_1_xz + VgaugeK2_2_xz


    VgaugeK2_0_yz = orb.apply_complex_potential(1.0, Bgauge21_0_yz, alpha2_0, prec)
    VgaugeK2_1_yz = orb.apply_complex_potential(1.0, Bgauge21_1_yz, alpha2_1, prec)
    VgaugeK2_2_yz = orb.apply_complex_potential(1.0, Bgauge21_2_yz, alpha2_2, prec)
    GaugeK2_alpha1_yz = VgaugeK2_0_yz + VgaugeK2_1_yz + VgaugeK2_2_yz


    GaugeK2_alpha1 = GaugeK2_alpha1_xy + GaugeK2_alpha1_xz + GaugeK2_alpha1_yz


    GaugeJmK_phi1 = GaugeJ2_alpha1 - GaugeK2_alpha1
    
    GaugeJmK_11_r, GJmK_11_i = spinorb1.dot(GJmK_phi1)

    print('GaugeJmK_11_r', GaugeJmK_11_r)
#############################END GAUGE ################################### 