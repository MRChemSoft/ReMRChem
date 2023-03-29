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

import two_electron

import importlib
importlib.reload(orb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-a', '--atype', dest='atype', type=str, default='He',
                        help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='BS',
                        help='put the type of derivative')
    parser.add_argument('-z', '--charge', dest='charge', type=float, default=2.0,
                        help='put the atom charge')
    parser.add_argument('-b', '--box', dest='box', type=int, default=30,
                        help='put the box dimension')
    parser.add_argument('-cx', '--center_x', dest='cx', type=float, default=0.1,
                        help='position of nucleus in x')
    parser.add_argument('-cy', '--center_y', dest='cy', type=float, default=0.2,
                        help='position of nucleus in y')
    parser.add_argument('-cz', '--center_z', dest='cz', type=float, default=0.3,
                        help='position of nucleus in z')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=10,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-8,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt', 'gauge'], 'Please, specify coulgau in a rigth way: coulomb or gaunt'

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

    ################ Define spinorbitals ########## 
    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La=complexfc)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    #spinorb1 = orb.orbital4c()
    #spinorb1.read('He_1s')
    print('Define spin orbital DONE')
    
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
        J2_phi1 = orb.apply_potential(1.0, B22, spinorb1, prec)
        JmK_phi1 = J2_phi1  # K part is zero for 2e system in GS
        JmK_11_r, JmK_11_i = spinorb1.dot(JmK_phi1)
        JmK_mat = np.array([[ JmK_11_r + JmK_11_i * 1j, 0],
                        [ 0,                        JmK_11_r + JmK_11_i * 1j]])

        hd_V_mat = hd_mat + v_mat

        # Calculate Fij Fock matrix
        F_mat = hd_V_mat + JmK_mat
        eps1 = F_mat[0,0].real
        eps2 = F_mat[1,1].real
        E_tot_JK = np.trace(F_mat) - 0.5 * (np.trace(JmK_mat))

        #print('h_d matrix\n', hd_mat)
        #print('v matrix\n', v_mat)
        #print('JmK matrix\n', JmK_mat)
        #print('F matrix\n', F_mat)
        print('Spinor Energy', eps1 - light_speed**2)
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

        if(compute_last_energy):
            break

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1

        # Calculation of Helmotz
        tmp = orb.apply_helmholtz(V_J_K_spinorb1, eps1, prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, eps1, der = default_der)
        new_orbital *= 0.5/light_speed**2
        new_orbital.normalize()
        new_orbital.cropLargeSmall(prec)

        # Compute orbital error
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Orbital_Error norm', error_norm)
        spinorb1 = new_orbital
        if (error_norm < prec):
            compute_last_energy = True

    spinorb1.save('Ne_1s')
#############################END COULOMB & START WITH GAUNT###################################
if args.coulgau == 'gaunt':
    spinorb1 = orb.orbital4c()
    spinorb1.read('Ne_1s')
    print('Read spin orbital DONE')
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
    spinorb1 = orb.orbital4c()
    spinorb1.read('Ne_1s')
    print('Read spin orbital DONE')
    spinorb2 = spinorb1.ktrs()


    print("alpha1")
    alpha1 =  spinorb1.alpha_vector(prec)
    print("n21")
    n21_x = spinorb2.overlap_density(alpha1[0], prec)
    n21_y = spinorb2.overlap_density(alpha1[1], prec)
    n21_z = spinorb2.overlap_density(alpha1[2], prec)
    del alpha1

    print("alpha2")
    alpha2 =  spinorb2.alpha_vector(prec)
    print("n22")
    n22_x = spinorb2.overlap_density(alpha2[0], prec)
    n22_y = spinorb2.overlap_density(alpha2[1], prec)
    n22_z = spinorb2.overlap_density(alpha2[2], prec)
    del alpha2


    length = 2 * args.box

    R3O = r3m.GaugeOperator(mra, 1e-5, length, prec)
    print('Gauge operator DONE')
     
    Bgauge22_xx = cf.complex_fcn()
    Bgauge22_xy = cf.complex_fcn()
    Bgauge22_xz = cf.complex_fcn()

    Bgauge22_xy.real = R3O(n22_x.real, 2, 0, 0)
    Bgauge22_xz.real = R3O(n22_y.real, 1, 1, 0)
    Bgauge22_xx.real = R3O(n22_z.real, 1 ,0 ,1)

    Bgauge22_xx.imag = R3O(n22_x.imag, 2, 0, 0)
    Bgauge22_xy.imag = R3O(n22_y.imag, 1, 1, 0)
    Bgauge22_xz.imag = R3O(n22_z.imag, 1 ,0 ,1)

    Bgauge22_x = Bgauge22_xx + Bgauge22_xy + Bgauge22_xz
    del Bgauge22_xx
    del Bgauge22_xy
    del Bgauge22_xz
    
    
    Bgauge22_yx = cf.complex_fcn()
    Bgauge22_yy = cf.complex_fcn()
    Bgauge22_yz = cf.complex_fcn()

    Bgauge22_Re_yx = R3O(n22_x.real, 1, 1, 0) 
    Bgauge22_Re_yy = R3O(n22_y.real, 0, 2, 0)    
    Bgauge22_Re_yz = R3O(n22_z.real, 0, 1, 1) 

    Bgauge22_Im_yx = R3O(n22_x.imag, 1, 1, 0) 
    Bgauge22_Im_yy = R3O(n22_y.imag, 0, 2, 0)    
    Bgauge22_Im_yz = R3O(n22_z.imag, 0, 1, 1) 

    Bgauge22_y = Bgauge22_yx + Bgauge22_yy + Bgauge22_yz
    del Bgauge22_yx
    del Bgauge22_yy
    del Bgauge22_yz


    Bgauge22_zx = cf.complex_fcn()
    Bgauge22_zy = cf.complex_fcn()
    Bgauge22_zz = cf.complex_fcn()

    Bgauge22_Re_zx = R3O(n22_x.real, 1, 0, 1) 
    Bgauge22_Re_zy = R3O(n22_y.real, 0, 1, 1)    
    Bgauge22_Re_zz = R3O(n22_z.real, 0 ,0 ,2) 

    Bgauge22_Im_zx = R3O(n22_x.imag, 1, 0, 1) 
    Bgauge22_Im_zy = R3O(n22_y.imag, 0, 1, 1)    
    Bgauge22_Im_zz = R3O(n22_z.imag, 0 ,0 ,2) 

    Bgauge22_z = Bgauge22_zx + Bgauge22_zy + Bgauge22_zz
    del Bgauge22_zx
    del Bgauge22_zy
    del Bgauge22_zz



    Bgauge21_xx = cf.complex_fcn()
    Bgauge21_xy = cf.complex_fcn()
    Bgauge21_xz = cf.complex_fcn()

    Bgauge21_xy.real = R3O(n21_x.real, 2, 0, 0)
    Bgauge21_xz.real = R3O(n21_y.real, 1, 1, 0)
    Bgauge21_xx.real = R3O(n21_z.real, 1 ,0 ,1)

    Bgauge21_xx.imag = R3O(n21_x.imag, 2, 0, 0)
    Bgauge21_xy.imag = R3O(n21_y.imag, 1, 1, 0)
    Bgauge21_xz.imag = R3O(n21_z.imag, 1 ,0 ,1)

    Bgauge21_x = Bgauge21_xx + Bgauge21_xy + Bgauge21_xz
    del Bgauge21_xx
    del Bgauge21_xy
    del Bgauge21_xz
    
    
    Bgauge21_yx = cf.complex_fcn()
    Bgauge21_yy = cf.complex_fcn()
    Bgauge21_yz = cf.complex_fcn()

    Bgauge21_Re_yx = R3O(n21_x.real, 1, 1, 0) 
    Bgauge21_Re_yy = R3O(n21_y.real, 0, 2, 0)    
    Bgauge21_Re_yz = R3O(n21_z.real, 0, 1, 1) 

    Bgauge21_Im_yx = R3O(n21_x.imag, 1, 1, 0) 
    Bgauge21_Im_yy = R3O(n21_y.imag, 0, 2, 0)    
    Bgauge21_Im_yz = R3O(n21_z.imag, 0, 1, 1) 

    Bgauge21_y = Bgauge21_yx + Bgauge21_yy + Bgauge21_yz
    del Bgauge21_yx
    del Bgauge21_yy
    del Bgauge21_yz


    Bgauge21_zx = cf.complex_fcn()
    Bgauge21_zy = cf.complex_fcn()
    Bgauge21_zz = cf.complex_fcn()

    Bgauge21_Re_zx = R3O(n21_x.real, 1, 0, 1) 
    Bgauge21_Re_zy = R3O(n21_y.real, 0, 1, 1)    
    Bgauge21_Re_zz = R3O(n21_z.real, 0 ,0 ,2) 

    Bgauge21_Im_zx = R3O(n21_x.imag, 1, 0, 1) 
    Bgauge21_Im_zy = R3O(n21_y.imag, 0, 1, 1)    
    Bgauge21_Im_zz = R3O(n21_z.imag, 0 ,0 ,2) 

    Bgauge21_z = Bgauge21_zx + Bgauge21_zy + Bgauge21_zz
    del Bgauge21_zx
    del Bgauge21_zy
    del Bgauge21_zz


    gaugeJr_0, gaugeJi_0 = n22_x.complex_conj().dot(Bgauge22_x)
    gaugeJr_1, gaugeJi_1 = n22_y.complex_conj().dot(Bgauge22_y)
    gaugeJr_2, gaugeJi_2 = n22_z.complex_conj().dot(Bgauge22_z)

    gaugeJr = gaugeJr_0 + gaugeJr_1 + gaugeJr_2
    gaugeJi = gaugeJi_0 + gaugeJi_1 + gaugeJi_2
    del gaugeJr_0
    del gaugeJr_1
    del gaugeJr_2
    del gaugeJi_0
    del gaugeJi_1
    del gaugeJi_2

    print("Direct   ", gaugeJr, gaugeJi)

    gaugeKr_0, gaugeKi_0 = n21_x.dot(Bgauge21_x)
    gaugeKr_1, gaugeKi_1 = n21_y.dot(Bgauge21_y)
    gaugeKr_2, gaugeKi_2 = n21_z.dot(Bgauge21_z)

    gaugeKr = gaugeKr_0 + gaugeKr_1 + gaugeKr_2
    gaugeKi = gaugeKi_0 + gaugeKi_1 + gaugeKi_2
    del gaugeKr_0
    del gaugeKr_1
    del gaugeKr_2
    del gaugeKi_0
    del gaugeKi_1
    del gaugeKi_2

    print("Exchange ", gaugeKr, gaugeKi)

    gaugeEnergy = gaugeJr - gaugeKr
    print("Gauge energy correction ", 0.5 * gaugeEnergy)


    #length = 2 * args.box
    #two_electron.gaugePert(spinorb1, spinorb2, mra, length, prec)
#############################END GAUGE ################################### 
