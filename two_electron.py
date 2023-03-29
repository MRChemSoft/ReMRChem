from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb 
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from vampyr import vampyr3d as vp
from orbital4c import r3m as r3m
from vampyr import vampyr1d as vp1 
import numpy as np
import numpy.linalg as LA
import sys, getopt


def coulomb_gs_2e(spinorb1, potential, mra, prec, der = 'ABGV'):
    print('Hartree-Fock (Coulomb interaction)')
    error_norm = 1
    compute_last_energy = False
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinorb1.light_speed


    while (error_norm > prec or compute_last_energy):
        n_11 = spinorb1.overlap_density(spinorb1, prec)
        spinorb2 = spinorb1.ktrs()
        spinorb2.cropLargeSmall(prec)
        spinorb2.normalize()

        print("Spinorb 1")
        print(spinorb1)
        print("Spinorb 2")
        print(spinorb2)
        
        return spinorb1, spinorb2

    # Definition of two electron operators
        B11    = P(n_11.real) * (4 * np.pi)

        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der)
        hd_11_r, hd_11_i = spinorb1.dot(hd_psi_1)
        hd_12_r, hd_12_i = spinorb1.dot(hd_psi_2)
        hd_21_r, hd_21_i = spinorb2.dot(hd_psi_1)
        hd_22_r, hd_22_i = spinorb2.dot(hd_psi_2)
        hd_mat = np.array([[hd_11_r + hd_11_i * 1j, hd_12_r + hd_12_i * 1j],
                           [hd_21_r + hd_21_i * 1j, hd_22_r + hd_22_i * 1j]])

        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, potential, spinorb1, prec)
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
        eps = Fmat[0,0].real
        
        print('Orbital energy', eps - light_speed**2)
        E_tot_JK = np.trace(Fmat) - 0.5 * (np.trace(JmK))
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

        if(compute_last_energy):
            break

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1

        # Calculation of Helmholtz
        tmp = orb.apply_helmholtz(V_J_K_spinorb1, eps, prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, eps, der)
        new_orbital *= 0.5/light_speed**2
        print('Norm new orbital ', np.sqrt(new_orbital.squaredNorm()))
        new_orbital.normalize()
        new_orbital.cropLargeSmall(prec)       

        # Compute orbital error
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Orbital_Error norm', error_norm)
        spinorb1 = new_orbital
        if(error_norm < prec):
            compute_last_energy = True
    return(spinorb1, spinorb2)

def gauntPert(spinorb1, spinorb2, mra, prec):
    
    P = vp.PoissonOperator(mra, prec)
    Light_speed = spinorb1.light_speed

    #Definition of alpha vectors for each orbital
    print("calculating alpha phi")
    alpha1_0 =  spinorb1.alpha(0)
    alpha1_1 =  spinorb1.alpha(1)
    alpha1_2 =  spinorb1.alpha(2)

    alpha2_0 =  spinorb2.alpha(0)
    alpha2_1 =  spinorb2.alpha(1)
    alpha2_2 =  spinorb2.alpha(2)

    #Defintion of orbital * alpha(orbital)
    n21_0 = spinorb2.overlap_density(alpha1_0, prec)
    n21_1 = spinorb2.overlap_density(alpha1_1, prec)
    n21_2 = spinorb2.overlap_density(alpha1_2, prec)
   
    n22_0 = spinorb2.overlap_density(alpha2_0, prec)
    n22_1 = spinorb2.overlap_density(alpha2_1, prec)
    n22_2 = spinorb2.overlap_density(alpha2_2, prec)
    
    n11_0 = spinorb1.overlap_density(alpha1_0, prec)
    n11_1 = spinorb1.overlap_density(alpha1_1, prec)
    n11_2 = spinorb1.overlap_density(alpha1_2, prec)
   
    n12_0 = spinorb1.overlap_density(alpha2_0, prec)
    n12_1 = spinorb1.overlap_density(alpha2_1, prec)
    n12_2 = spinorb1.overlap_density(alpha2_2, prec)
    
    #Definition of Gaunt two electron operators       
    print("calculating potentials")
    BG22_Re0 = P(n22_0.real) * (4.0 * np.pi)
    BG22_Re1 = P(n22_1.real) * (4.0 * np.pi)
    BG22_Re2 = P(n22_2.real) * (4.0 * np.pi)
    BG22_Im0 = P(n22_0.imag) * (4.0 * np.pi)
    BG22_Im1 = P(n22_1.imag) * (4.0 * np.pi)
    BG22_Im2 = P(n22_2.imag) * (4.0 * np.pi)
    
    BG21_Re0 = P(n21_0.real) * (4.0 * np.pi)
    BG21_Re1 = P(n21_1.real) * (4.0 * np.pi)
    BG21_Re2 = P(n21_2.real) * (4.0 * np.pi)
    BG21_Im0 = P(n21_0.imag) * (4.0 * np.pi)
    BG21_Im1 = P(n21_1.imag) * (4.0 * np.pi)
    BG21_Im2 = P(n21_2.imag) * (4.0 * np.pi)
    
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
    print("applying potentials")
    #    VGJ2_0 = orb.apply_complex_potential(1.0, BG22_0, alpha1_0, prec)
    #    VGJ2_1 = orb.apply_complex_potential(1.0, BG22_1, alpha1_1, prec)
    #    VGJ2_2 = orb.apply_complex_potential(1.0, BG22_2, alpha1_2, prec)
    #    GJ2_alpha1 = VGJ2_0 + VGJ2_1 + VGJ2_2

    EJ_0 = (n11_0.complex_conj()).dot(BG22_0)
    EJ_1 = (n11_1.complex_conj()).dot(BG22_1)
    EJ_2 = (n11_2.complex_conj()).dot(BG22_2)
    
    EK_0 = (n12_0.complex_conj()).dot(BG21_0)
    EK_1 = (n12_1.complex_conj()).dot(BG21_1)
    EK_2 = (n12_2.complex_conj()).dot(BG21_2)

    print("Direct part   ", EJ_0, EJ_1, EJ_2)
    print("Exchange part ", EK_0, EK_1, EK_2)
    
    EJ = EJ_0[0] + EJ_1[0] + EJ_2[0]
    EK = EK_0[0] + EK_1[0] + EK_2[0]

#    Vgk2_0 = orb.apply_complex_potential(1.0, BG21_0, alpha2_0, prec)
#    VGK2_1 = orb.apply_complex_potential(1.0, BG21_1, alpha2_1, prec)
#    VGK2_2 = orb.apply_complex_potential(1.0, BG21_2, alpha2_2, prec)
#    GK2_alpha1 = VGK2_0 + VGK2_1 + VGK2_2

    # change of sign!
#    GJmK_phi1 = GK2_alpha1 - GJ2_alpha1
    
#    print("computing exp value")
#    GJmK_11_r, GJmK_11_i = spinorb1.dot(GJmK_phi1)

    print('GJmK_11_r', EK - EJ)


def calcGaugePotential(density, operator, direction, poisson): # direction is i index
    
    Bgauge = [cf.complex_fcn(), cf.complex_fcn(), cf.complex_fcn()]
    index = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    index[0][direction] += 1
    index[1][direction] += 1
    index[2][direction] += 1
    for i in range(3): # j index
        Bgauge[i].real = operator(density[i].real, index[i][0], index[i][1], index[i][2])
        Bgauge[i].imag = operator(density[i].imag, index[i][0], index[i][1], index[i][2])
#        Bgauge[i].real = poisson(density[i].real)
#        Bgauge[i].imag = poisson(density[i].imag)
        

    return Bgauge[0] + Bgauge[1] + Bgauge[2]

def gaugePert(spinorb1, spinorb2, mra, length, prec):
    print("gaugePert")
    
    #P = vp.PoissonOperator(mra, prec)       Non serve
    #light_speed = spinorb1.light_speed    Non serve

    #Definition of alpha vectors for each orbital

    print("alpha1")
    alpha1 =  spinorb1.alpha_vector(prec)
    print("n21")
    n21 = [spinorb2.overlap_density(alpha1[0], prec),
           spinorb2.overlap_density(alpha1[1], prec),
           spinorb2.overlap_density(alpha1[2], prec)]
    del alpha1

    print("alpha2")
    alpha2 =  spinorb2.alpha_vector(prec)
    print("n22")
    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]
    del alpha2

    print("densities")
    print("n22 x y z")
    print(n22[0], n22[1], n22[2])
    print("n21 x y z")
    print(n21[0], n21[1], n21[2])



    #Definition of Gauge operator
    R3O = r3m.GaugeOperator(mra, 1e-5, length, prec)
    print('Gauge operator DONE')

    Bgauge22 = [calcGaugePotential(n22, R3O, 0), calcGaugePotential(n22, R3O, 1), calcGaugePotential(n22, R3O, 2)]
    print("Bgauge22")
    print(Bgauge22[0], Bgauge22[1], Bgauge22[2])

    Bgauge21 = [calcGaugePotential(n21, R3O, 0), calcGaugePotential(n21, R3O, 1), calcGaugePotential(n21, R3O, 2)]
    print("Bgauge21")
    print(Bgauge21[0], Bgauge21[1], Bgauge21[2])

    # the following idientites hold for two orbitals connected by KTRS
    # n_11[i] == -n22[i]
    # n_12[i] ==  n21[i].complex_conj()

    gaugeEnergy = 0

    for i in range(3): 
        Bgauge22 = calcGaugePotential(n22, R3O, i, P)
        Bgauge21 = calcGaugePotential(n21, R3O, i, P)
        gaugeJr, gaugeJi = n22[i].complex_conj().dot(Bgauge22)
        gaugeKr, gaugeKi = n21[i].dot(Bgauge21)
        print("Direct   ", gaugeJr, gaugeJi)
        print("Exchange ", gaugeKr, gaugeKi)
        gaugeEnergy = gaugeEnergy - 0.5 * (gaugeJr + gaugeKr)
        del Bgauge22
        del Bgauge21
    print("Gauge energy correction ", gaugeEnergy)
    return gaugeEnergy
