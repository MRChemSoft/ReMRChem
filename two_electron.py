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
    alpha1_0 =  spinorb1.alpha(0, prec)
    alpha1_1 =  spinorb1.alpha(1, prec)
    alpha1_2 =  spinorb1.alpha(2, prec)

    alpha2_0 =  spinorb2.alpha(0, prec)
    alpha2_1 =  spinorb2.alpha(1, prec)
    alpha2_2 =  spinorb2.alpha(2, prec)

    #Defintion of orbital * alpha(orbital)
    n21_0 = spinorb2.overlap_density(alpha1_0, prec)
    n21_1 = spinorb2.overlap_density(alpha1_1, prec)
    n21_2 = spinorb2.overlap_density(alpha1_2, prec)
   
    n22_0 = spinorb2.overlap_density(alpha2_0, prec)
    n22_1 = spinorb2.overlap_density(alpha2_1, prec)
    n22_2 = spinorb2.overlap_density(alpha2_2, prec)

    norm22 = [np.sqrt(n22_0.squaredNorm()),
              np.sqrt(n22_1.squaredNorm()),
              np.sqrt(n22_2.squaredNorm())
            ]

    norm21 = [np.sqrt(n21_0.squaredNorm()),
              np.sqrt(n21_1.squaredNorm()),
              np.sqrt(n21_2.squaredNorm())
            ]

    print(norm22)
    print(norm21)
    
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

    print("Pot norm BG22_0", BG22_0.squaredNorm(), n22_0.squaredNorm(), EJ_0)
    print("Pot norm BG21_0", BG21_0.squaredNorm(), n21_0.squaredNorm(), EK_0)
    print("Pot norm BG22_1", BG22_1.squaredNorm(), n22_1.squaredNorm(), EJ_1)
    print("Pot norm BG21_1", BG21_1.squaredNorm(), n21_1.squaredNorm(), EK_1)
    print("Pot norm BG22_2", BG22_2.squaredNorm(), n22_2.squaredNorm(), EJ_2)
    print("Pot norm BG21_2", BG21_2.squaredNorm(), n21_2.squaredNorm(), EK_2)

    print(n22_0)
    print(BG22_0)
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
    print("Gauge Perturbation")
    #P = vp.PoissonOperator(mra, prec)       Non serve
    #light_speed = spinorb1.light_speed    Non serve

    #Definition of alpha vectors for each orbital

    n21 = [spinorb2.overlap_density(alpha1[0], prec),
           spinorb2.overlap_density(alpha1[1], prec),
           spinorb2.overlap_density(alpha1[2], prec)]
    del alpha1
    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]
    del alpha2

    #Definition of Gauge operator
    R3O = r3m.GaugeOperator(mra, 1e-5, length, prec)

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

def calcGaugePert(spinorb1, spinorb2, mra, prec):
    print("Gauge Perturbation (Xiaosong Li version)")
    projection_operator = vp.ScalingProjector(mra, prec)
    alpha1 =  spinorb1.alpha_vector(prec)
    n21 = [spinorb2.overlap_density(alpha1[0], prec),
           spinorb2.overlap_density(alpha1[1], prec),
           spinorb2.overlap_density(alpha1[2], prec)]
    del alpha1
    alpha2 =  spinorb2.alpha_vector(prec)
    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]
    del alpha2
    
    n21[0].cropRealImag(prec)
    n21[1].cropRealImag(prec)
    n21[2].cropRealImag(prec)
    n22[0].cropRealImag(prec)
    n22[1].cropRealImag(prec)
    n22[2].cropRealImag(prec)
    
    P = vp.PoissonOperator(mra, prec)
    
    div_n22 = cf.divergence(n22, prec)
    div_n21 = cf.divergence(n21, prec)
    n22_dot_r = cf.vector_dot_r(n22, prec)
    n21_dot_r = cf.vector_dot_r(n21, prec)
    div_n22_r = cf.scalar_times_r(div_n22, prec)
    div_n21_r = cf.scalar_times_r(div_n21, prec)
    
    norm22 = {
        "nx":np.sqrt(n22[0].squaredNorm()),
        "ny":np.sqrt(n22[1].squaredNorm()),
        "nz":np.sqrt(n22[2].squaredNorm()),
        "divn":np.sqrt(div_n22.squaredNorm()),
        "ndotr":np.sqrt(n22_dot_r.squaredNorm()),
        "divnx":np.sqrt(div_n22_r[0].squaredNorm()),
        "divny":np.sqrt(div_n22_r[1].squaredNorm()),
        "divnz":np.sqrt(div_n22_r[2].squaredNorm())
    }
    
    norm21 = {
        "nx":np.sqrt(n21[0].squaredNorm()),
        "ny":np.sqrt(n21[1].squaredNorm()),
        "nz":np.sqrt(n21[2].squaredNorm()),
        "divn":np.sqrt(div_n21.squaredNorm()),
        "ndotr":np.sqrt(n21_dot_r.squaredNorm()),
        "divnx":np.sqrt(div_n21_r[0].squaredNorm()),
        "divny":np.sqrt(div_n21_r[1].squaredNorm()),
        "divnz":np.sqrt(div_n21_r[2].squaredNorm())
    }
    
    print(norm22)
    print(norm21)
    
    val22 = norm22["nx"] * norm22["divnx"] + norm22["ny"] * norm22["divny"] + norm22["nz"] * norm22["divnz"] + norm22["ndotr"] * norm22["divn"]
    val21 = norm21["nx"] * norm21["divnx"] + norm21["ny"] * norm21["divny"] + norm21["nz"] * norm21["divnz"] + norm21["ndotr"] * norm21["divn"]
    val = val21 + val22
    
    print("val", val)
    # the following idientites hold for two orbitals connected by KTRS
    # n_11[i] == -n22[i]
    # n_12[i] ==  n21[i].complex_conj()
    
    print("Potentials")
    print("n11 term")
    threshold = 0.01 * prec * val / norm22["ndotr"]
    print("threshold 22", threshold)
    V_div_n11 = cf.apply_poisson(div_n22, div_n22.mra, P, prec, thresholdNorm = threshold, factor = -1.0)
    
    print("n12 term")
    threshold = 0.01 * prec * val / norm21["ndotr"] 
    print("threshold 21", threshold)
    V_div_n12 = (cf.apply_poisson(div_n21, div_n21.mra, P, prec, thresholdNorm = threshold, factor = -1.0)).complex_conj()
    
    V_div_n11_r = {}
    V_div_n12_r = {}
    
    n1label = ["nx", "ny", "nz"]
    
    for i in range(3):
        print("Potentials",i)
        print("n11 term")
        threshold = 0.01 * prec * val / norm22[n1label[i]]
        print("threshold 22", threshold)
        V_div_n11_r[i] = cf.apply_poisson(div_n22_r[i], div_n22_r[i].mra, P, prec, thresholdNorm = threshold, factor = 1.0)
    
        print("n12 term")
        threshold = 0.01 * prec * val / norm21[n1label[i]]
        print("threshold 21", threshold)
        V_div_n12_r[i] = (cf.apply_poisson(div_n21_r[i], div_n21_r[i].mra, P, prec, thresholdNorm = threshold, factor = 1.0)).complex_conj()
    
    print("scalar products")
    n22_dot_r_V_div_n11r, n22_dot_r_V_div_n11i = n22_dot_r.dot(V_div_n11, False)
    n21_dot_r_V_div_n12r, n21_dot_r_V_div_n12i = n21_dot_r.dot(V_div_n12, False)    

    result_22_11 = n22_dot_r_V_div_n11r + 1j * n22_dot_r_V_div_n11i
    result_21_12 = n21_dot_r_V_div_n12r + 1j * n21_dot_r_V_div_n12i
    print("result before loop")
    print(result_22_11)
    print(result_21_12)

    for i in range(3):
        result_22_11r, result_22_11i = n22[i].dot(V_div_n11_r[i], False)
        result_21_12r, result_21_12i = n21[i].dot(V_div_n12_r[i], False)
        print("in loop",i)
        print(result_22_11r, result_22_11i)
        print(result_21_12r, result_21_12i)
        result_22_11 += result_22_11r + 1j * result_22_11i
        result_21_12 += result_21_12r + 1j * result_21_12i
    print("Final direct ", result_22_11)
    print("Final exchange ", result_21_12)
    print("Total ",  result_22_11 - result_21_12)

def calcGauntPert(spinorb1, spinorb2, mra, prec):
    print ("Gaunt Perturbation")
    P = vp.PoissonOperator(mra, prec)
    Light_speed = spinorb1.light_speed
    alpha1 =  spinorb1.alpha_vector(prec)
    n21 = [spinorb2.overlap_density(alpha1[0], prec),
           spinorb2.overlap_density(alpha1[1], prec),
           spinorb2.overlap_density(alpha1[2], prec)]
    del alpha1
    
    alpha2 =  spinorb2.alpha_vector(prec)
    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]
    del alpha2
    
    n21[0].cropRealImag(prec)
    n21[1].cropRealImag(prec)
    n21[2].cropRealImag(prec)
    n22[0].cropRealImag(prec)
    n22[1].cropRealImag(prec)
    n22[2].cropRealImag(prec)

    norm22 = [np.sqrt(n22[0].squaredNorm()),
              np.sqrt(n22[1].squaredNorm()),
              np.sqrt(n22[2].squaredNorm())
            ]

    norm21 = [np.sqrt(n21[0].squaredNorm()),
              np.sqrt(n21[1].squaredNorm()),
              np.sqrt(n21[2].squaredNorm())
            ]

    print(norm22)
    print(norm21)
    
    val22 = 0
    val21 = 0
    for i in range(3):
        val22 += norm22[i]**2
        val21 += norm21[i]**2
    val = val21 + val22
    
    print("calculating potentials")
    BG22 = []
    BG21 = []
    EJ = []
    EK = []
    EJt = 0
    EKt = 0
    for i in range(3):
        threshold = 0.01 * prec * val / norm22[i]
        pot = cf.apply_poisson(n22[i], n22[i].mra, P, prec, thresholdNorm = threshold, factor = 1)
        BG22.append(pot)
        EJ.append(n22[i].dot(BG22[i], False))
        EJt += EJ[i][0] + 1j * EJ[i][1]
        print("pot norm 22", i, BG22[i].squaredNorm(), n22[i].squaredNorm(), EJ[i])

        threshold = 0.01 * prec * val / norm21[i]
        pot = (cf.apply_poisson(n21[i], n21[i].mra, P, prec, thresholdNorm = threshold, factor = -1)).complex_conj()
        BG21.append(pot)
        EK.append(n21[i].dot(BG21[i], False))
        EKt += EK[i][0] + 1j * EK[i][1]
        print("pot norm 21", i, BG21[i].squaredNorm(), n21[i].squaredNorm(), EK[i])

    print("Direct part   ", EJ)
    print("Exchange part ", EK)

    print('GJmK_11_r', EJt - EKt)
