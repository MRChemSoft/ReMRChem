from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import operators as oper
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from vampyr import vampyr3d as vp
import numpy as np
import numpy.linalg as LA
import sys, getopt

#
# Generic coulomb Dirac HF solver. Now works for 2e connected by KTRS.
# It should be easy to extend it to more complicated cases
#
def coulomb_gs_gen(spinors, potential, mra, prec, derivative):
    print('Hartree-Fock (Coulomb interaction) Generic 2e')
    error_norm = 1.0
    compute_last_energy = False
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinors[0].light_speed
    # while (error_norm > prec or compute_last_energy):
    Jop = oper.CoulombDirectOperator(mra, prec, spinors)
    Kop = oper.CoulombExchangeOperator(mra, prec, spinors)
    Vop = oper.PotentialOperator(mra, prec, potential)
    Dop = oper.FockOperator(mra, prec, [], []) # "empty" fock operator
    Jmat = Jop.matrix(spinors)
    Kmat = Kop.matrix(spinors)
    Vmat = Vop.matrix(spinors)
    Dmat = Dop.matrix(spinors)
    Fmat = Dmat - Vmat + Jmat - Kmat
    for i in range(5):
        new_spinors = []
        # for j in range(len(spinors)):   # limited to two spinors linked by KTRS for now
        # print("j iter ", j)
        print("Applying J")
        Jpsi = Jop(spinors[0])
        print("Applying K")
        Kpsi = Kop(spinors[0])
        print("Applying V")
        Vpsi = Vop(spinors[0])
        for k in range(len(spinors)):
            print("k iter ", k)
            orbital_array = []
            coeff_array = []
            if(k != 0):
                orbital_array.append(spinors[k])
                coeff_array.append(Fmat[0][k])
        Fijpsij = orb.add_vector(orbital_array, coeff_array, prec)
        RHS = Jpsi - Kpsi - Vpsi - Fijpsij
        mu = orb.calc_dirac_mu(Fmat[0][0].real, light_speed)
        tmp = orb.apply_helmholtz(RHS, mu, prec)

        new_spinor = orb.apply_dirac_hamiltonian(tmp, prec, Fmat[0][0].real, der = derivative)
        new_spinor *= 0.5/light_speed**2
        new_spinor.normalize()
        new_spinor.cropLargeSmall(prec)
        new_spinors.append(new_spinor)
        new_spinors.append(new_spinor.ktrs(prec))
        spinors = new_spinors

        Jop = oper.CoulombDirectOperator(mra, prec, spinors)
        Kop = oper.CoulombExchangeOperator(mra, prec, spinors)
        Vop = oper.PotentialOperator(mra, prec, potential)
        Dop = oper.FockOperator(mra, prec, [], []) # "empty" fock operator
        print("Compute Jmat")
        Jmat = Jop.matrix(spinors)
        print("Compute Kmat")
        Kmat = Kop.matrix(spinors)
        print("Compute Vmat")
        Vmat = Vop.matrix(spinors)
        print("Compute Dmat")
        Dmat = Dop.matrix(spinors)
        Fmat = Dmat - Vmat + Jmat - Kmat
        print(Fmat)
        print(Fmat[0][0] - light_speed ** 2)
    return spinors[0], spinors[1]

def coulomb_2e_D2(spinors, potential, mra, prec, derivative):
    print('Hartree-Fock (Coulomb interaction) 2e D2')
    error_norm = 1.0
    compute_last_energy = False
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinors[0].light_speed
    # while (error_norm > prec or compute_last_energy):
    Jop = oper.CoulombDirectOperator(mra, prec, spinors)
#    Kop = oper.CoulombExchangeOperator(mra, prec, spinors)
    Vop = oper.PotentialOperator(mra, prec, potential)
    Dop = oper.FockOperator(mra, prec, [], []) # "empty" fock operator
    Jmat = Jop.matrix(spinors)
#    Kmat = Kop.matrix(spinors)
    Vmat = Vop.matrix(spinors)
    Dmat = Dop.matrix(spinors)
    Fmat = Dmat - Vmat + 0.5 * Jmat
    F2mat = Fmat @ Fmat
    print("F2mat")
    print(F2mat)
    c2 = light_speed**2
#    for i in range(5):
    while(error_norm > prec):
        new_spinors = []
        print("Applying J")
        Jpsi = Jop(spinors[0])
#        print("Applying K")
#        Kpsi = Kop(spinors[0])
        print("Applying V")
        Vpsi = Vop(spinors[0])
        
#        VT_psi = Jpsi - Kpsi - Vpsi
        VT_psi = 0.5 * Jpsi - Vpsi
        ap_VT_psi = VT_psi.alpha_p(prec)
        beta_VT_psi = VT_psi.beta2()
        ap_psi = spinors[0].alpha_p(prec)
#        VT_ap_psi = Jop(ap_psi) - Kop(ap_psi) - Vop(ap_psi)
#        VT_VT_psi = Jop(VT_psi) - Kop(VT_psi) - Vop(VT_psi)
        VT_ap_psi = 0.5 * Jop(ap_psi) - Vop(ap_psi)
        VT_VT_psi = 0.5 * Jop(VT_psi) - Vop(VT_psi)
        anticom = VT_ap_psi + ap_VT_psi
        anticom *= 1.0 / (2.0 * light_speed)
        VT_VT_psi *= 1.0 / (2.0 * c2)

#        for k in range(len(spinors)):
#            print("k iter ", k)
#            orbital_array = []
#            coeff_array = []
#            if(k != 0):
#                orbital_array.append(spinors[k])
#                coeff_array.append(F2mat[0][k])
#        Fijpsij = orb.add_vector(orbital_array, coeff_array, prec)
#        RHS = beta_VT_psi + anticom + VT_VT_psi - Fijpsij
        anticom.cropLargeSmall(prec)
        beta_VT_psi.cropLargeSmall(prec)
        VT_VT_psi.cropLargeSmall(prec)
        print("anticom")
        print(anticom)
        print("beta_VT_psi")
        print(beta_VT_psi)
        print("VT_VT_psi")
        print(VT_VT_psi)
        RHS = beta_VT_psi + anticom + VT_VT_psi
        RHS.cropLargeSmall(prec)
        print("RHS")
        print(RHS)

        cke = spinors[0].classicT()
        cpe = (spinors[0].dot(RHS)).real
        print("Classic-like energies: ", cke, cpe, cke + cpe)
        print("Orbital energy: ", c2 * ( -1.0 + np.sqrt(1 + 2 * (cpe + cke) / c2)))
        mu = orb.calc_non_rel_mu(cke+cpe)

#        mu = orb.calc_kutzelnigg_mu(F2mat[0][0], light_speed)
        print("this is mu: ", mu)
        new_spinor = orb.apply_helmholtz(RHS, mu, prec)
        print("normalization")
        new_spinor.normalize()
        print("crop")
        new_spinor.cropLargeSmall(prec)
        new_spinors.append(new_spinor)
        new_spinors.append(new_spinor.ktrs(prec))
        # Compute orbital error
        delta_psi = new_spinor - spinors[0]
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Orbital_Error norm', error_norm)
        spinors = new_spinors

        Jop = oper.CoulombDirectOperator(mra, prec, spinors)
#        Kop = oper.CoulombExchangeOperator(mra, prec, spinors)
        Vop = oper.PotentialOperator(mra, prec, potential)
        Dop = oper.FockOperator(mra, prec, [], []) # "empty" fock operator
        print("Compute Jmat")
        Jmat = Jop.matrix(spinors)
        print(Jmat)
#        print("Compute Kmat")
#        Kmat = Kop.matrix(spinors)
#        print(Kmat)
        print("Compute Vmat")
        Vmat = Vop.matrix(spinors)
        print(Vmat)
        print("Compute Dmat")
        Dmat = Dop.matrix(spinors)
        print(Dmat)
        Fmat = Dmat - Vmat + 0.5 * Jmat
        F2mat = Fmat @ Fmat
        print(Fmat)
        print("orbital energy: ", Fmat[0][0] - light_speed ** 2)
        total_energy = 2.0 * Fmat[0][0] - Jmat[0][0] - 2 * light_speed ** 2
        print("total energy: ", total_energy)
    return spinors[0], spinors[1]

def coulomb_2e_D2_J(spinors, potential, mra, prec, derivative):
    print('Hartree-Fock (Coulomb interaction) 2e D2 J only')
    error_norm = 1.0
    compute_last_energy = False
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinors[0].light_speed
    c2 = light_speed**2
    Vop = oper.PotentialOperator(mra, prec, potential)
    while(error_norm > prec):
        Jop = oper.CoulombDirectOperator(mra, prec, spinors)
        RHS = build_RHS_D2(Jop, Vop, spinors[0], prec, light_speed)
        cke = spinors[0].classicT()
        cpe = (spinors[0].dot(RHS)).real
        print("Classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
        print("Orbital energy: ", c2 * ( -1.0 + np.sqrt(1 + 2 * (cpe + cke) / c2)))
        mu = orb.calc_non_rel_mu(cke+cpe)
        new_spinor = orb.apply_helmholtz(RHS, mu, prec)
        #print("============= Spinor before Helmholtz =============")
        #print(spinors[0])
        #print("============= RHS before Helmh#oltz    =============")
        #print(RHS)
        #print("============= New spinor before crop  =============")
        #print(new_spinor)
        new_spinor.cropLargeSmall(prec)
        new_spinor.normalize()
        delta_psi = new_spinor - spinors[0]
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Orbital_Error norm', error_norm)
        spinors[0] = new_spinor
        spinors[1] = new_spinor.ktrs(prec)
    Jop = oper.CoulombDirectOperator(mra, prec, spinors)
    RHS = build_RHS_D2(Jop, Vop, spinors[0], prec, light_speed)
    cke = spinors[0].classicT()
    cpe = (spinors[0].dot(RHS)).real
    final_orbital_energy = c2 * ( -1.0 + np.sqrt(1 + 2 * (cpe + cke) / c2))
    Jorb = Jop(spinors[0])
    Jenergy = (spinors[0].dot(Jorb)).real
    final_total_energy = 2.0 * final_orbital_energy - 0.5 * Jenergy
    print("Final classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
    print("Final orbital energy: ", final_orbital_energy)
    print("Final Total energy: ", final_total_energy)
    return spinors[0], spinors[1]

def coulomb_gs_2e(spinorb1, potential, mra, prec, derivative):
    print('Hartree-Fock (Coulomb interaction)')
    error_norm = 1
    compute_last_energy = False
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinorb1.light_speed
#    for i in range(10):
    while (error_norm > prec or compute_last_energy):
        n_22 = spinorb1.overlap_density(spinorb1, prec)

        # Definition of two electron operators
        B22    = P(n_22.real) * (4 * np.pi)

        # Definiton of Dirac Hamiltonian for spinorbit 1 that due to TRS is equal spinorbit 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = derivative)
        hd_11 = spinorb1.dot(hd_psi_1)
        print("hd_11", hd_11)
        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, potential, spinorb1, prec)
        V1 = spinorb1.dot(v_psi_1)

        hd_V_11 = hd_11 + V1

        # Calculation of two electron terms
        J2_phi1 = orb.apply_potential(1.0, B22, spinorb1, prec)

        JmK_phi1 = J2_phi1  # K part is zero for 2e system in GS
        JmK = spinorb1.dot(JmK_phi1)

        # Calculate Fij Fock matrix
        print("J contribution", JmK.real)
        eps = hd_V_11.real + JmK.real
        E_tot_JK =  2*eps - JmK.real

        print('orbital energy', eps - light_speed**2)
        print('total energy', E_tot_JK - (2.0 *light_speed**2))
        if(compute_last_energy):
            break

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1

        mu = orb.calc_dirac_mu(eps, light_speed)
        tmp = orb.apply_helmholtz(V_J_K_spinorb1, mu, prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, eps, der = derivative)
        new_orbital *= 0.5/light_speed**2
        #print("============= Spinor before Helmholtz =============")
        #print(spinorb1)
        #print("============= RHS before Helmholtz    =============")
        #print(V_J_K_spinorb1)
        #print("============= New spinor before crop  =============")
        #print(new_orbital)
        new_orbital.normalize()
        new_orbital.cropLargeSmall(prec)       

        # Compute orbital error
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Orbital_Error norm', error_norm)
        spinorb1 = new_orbital
        spinorb2 = spinorb1.ktrs(prec)
        if(error_norm < prec):
            compute_last_energy = True
    return spinorb1, spinorb2

#def coulomb_gs(potential, spinors, mra, prec, der = 'ABGV'):
#    print("Dirac Hartree Fock iteration")
#    error_norm = 1
#    n_spinors = len(spinors)
#    while (error_norm > prec):
#        J = operators.CoulombDirectOperator(mra, prec, spinors)
#        K = operators.CoulombExchangeOperator(mra, prec, spinors)
#        F = np.zeros((n_spinors, n_spinors))
#        for i in range(n_spinors):
#            si = spinors[i]
#            Jsi = J(spinor)
#            Ksi = K(spinor)
#            Vsi = orbital.apply_potential(-1.0, potential, si, prec)
#            Dsi = orbital.apply_dirac_hamiltonian(si, prec, shift = 0, der = 'ABGV')
#            RHS = Vsi + Jsi - Ksi
#            for j in range(i, n_spinors) in spinors:
#                sj = spinors[j]
#                F[j][i] = sj.dot(RHS + Ds)
#                F[i][j]  = F[j][]
#    

def calcAlphaDensityVector(spinorb1, spinorb2, prec):
    alphaOrbital =  spinorb2.alpha_vector(prec)
    alphaDensity = [spinorb1.overlap_density(alphaOrbital[0], prec),
                    spinorb1.overlap_density(alphaOrbital[1], prec),
                    spinorb1.overlap_density(alphaOrbital[2], prec)]
    del alphaOrbital
    alphaDensity[0].cropRealImag(prec)
    alphaDensity[1].cropRealImag(prec)
    alphaDensity[2].cropRealImag(prec)
    return alphaDensity

#
# currently without the -1/2 factor
# correct gaun term: multiply by -1/2
# gaunt and delta-term from gauge: multiply by -1
#
def calcGauntPert(spinorb1, spinorb2, mra, prec):
    print ("Gaunt Perturbation")
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    norm22 = [np.sqrt(n22[0].squaredNorm()),
              np.sqrt(n22[1].squaredNorm()),
              np.sqrt(n22[2].squaredNorm())
            ]
    norm21 = [np.sqrt(n21[0].squaredNorm()),
              np.sqrt(n21[1].squaredNorm()),
              np.sqrt(n21[2].squaredNorm())
            ]
    norm11 = [np.sqrt(n11[0].squaredNorm()),
              np.sqrt(n11[1].squaredNorm()),
              np.sqrt(n11[2].squaredNorm())
            ]
    norm12 = [np.sqrt(n12[0].squaredNorm()),
              np.sqrt(n12[1].squaredNorm()),
              np.sqrt(n12[2].squaredNorm())
            ]
    
    val22 = 0
    val21 = 0
    for i in range(3):
        val22 += norm22[i]**2
        val21 += norm21[i]**2
    val = val21 + val22
    
    EJ = []
    EK = []
    EJt = 0
    EKt = 0
    for i in range(3):
        threshold = 0.0001 * prec * val / norm22[i]
        pot = cf.apply_poisson(n11[i], n11[i].mra, P, prec, thresholdNorm = threshold, factor = 1)
        EJ.append(n22[i].dot(pot, False))
        EJt += EJ[i]

        threshold = 0.0001 * prec * val / norm21[i]
        pot = (cf.apply_poisson(n12[i], n12[i].mra, P, prec, thresholdNorm = threshold, factor = 1))
        EK.append(n21[i].dot(pot, False))
        EKt += EK[i]

    gaunt = EJt - EKt
    return gaunt.real

def computeTestNorm(contributions):
    val = 0
    for i in range(len(contributions["density"])):
        squaredNormDensity = contributions["density"][i].squaredNorm()
        squaredNormPotential = contributions["potential"][i].squaredNorm()
        val += np.sqrt(squaredNormDensity * squaredNormPotential)
    return val
                   
def calcPerturbationValues(contributions, P, prec, testNorm):
    val = 0
    conjugate = False
    for i in range(len(contributions["density"])):
        sign = contributions["sign"][i]
        density = contributions["density"][i]
        auxDensity = contributions["potential"][i]
        normDensity = np.sqrt(density.squaredNorm())
        normAuxDensity = np.sqrt(auxDensity.squaredNorm())
        threshold = 0.00001 * prec * testNorm / normDensity
        potential = (cf.apply_poisson(auxDensity, auxDensity.mra, P, prec, thresholdNorm = threshold, factor = sign))
        spr, spi = density.dot(potential, conjugate)
        val += spr + 1j * spi
    return val
    
#
# no delta terms from this expression
#
def calcGaugePertA(spinorb1, spinorb2, mra, prec, derivative):
    print("Gauge Perturbation Version A")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)


    div_n22 = cf.divergence(n22, prec, derivative)
    div_n21 = cf.divergence(n21, prec, derivative)
    n11_dot_r = cf.vector_dot_r(n11, prec)
    n12_dot_r = cf.vector_dot_r(n12, prec)

    n22_r_mat = cf.vector_tensor_r(n22, prec)
    n21_r_mat = cf.vector_tensor_r(n21, prec)

    grad_n11 = cf.vector_gradient(n11, derivative)
    grad_n12 = cf.vector_gradient(n12, derivative)

    contributions = {
        "density":[n11_dot_r,
                   grad_n11[0][0], grad_n11[1][0], grad_n11[2][0],
                   grad_n11[0][1], grad_n11[1][1], grad_n11[2][1],
                   grad_n11[0][2], grad_n11[1][2], grad_n11[2][2],
                   n12_dot_r,
                   grad_n12[0][0], grad_n12[1][0], grad_n12[2][0],
                   grad_n12[0][1], grad_n12[1][1], grad_n12[2][1],
                   grad_n12[0][2], grad_n12[1][2], grad_n12[2][2]],
        "potential":[div_n22,
                     n22_r_mat[0][0], n22_r_mat[0][1], n22_r_mat[0][2],
                     n22_r_mat[1][0], n22_r_mat[1][1], n22_r_mat[1][2],
                     n22_r_mat[2][0], n22_r_mat[2][1], n22_r_mat[2][2],
                     div_n21,
                     n21_r_mat[0][0], n21_r_mat[0][1], n21_r_mat[0][2],
                     n21_r_mat[1][0], n21_r_mat[1][1], n21_r_mat[1][2],
                     n21_r_mat[2][0], n21_r_mat[2][1], n21_r_mat[2][2]],
        "sign":[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    } 

    testNorm = computeTestNorm(contributions)
    print("Test Norm A", testNorm)
    result = calcPerturbationValues(contributions, P, prec, testNorm)
    print("final Gauge A", result)
    return -0.5 * result.real

def calcGaugePertB(spinorb1, spinorb2, mra, prec, derivative):
    print("Gauge Perturbation Version B")

    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    div_n22 = cf.divergence(n22, prec, derivative)
    div_n21 = cf.divergence(n21, prec, derivative)
    div_n22_r = cf.scalar_times_r(div_n22, prec)
    div_n21_r = cf.scalar_times_r(div_n21, prec)
    n11_dot_r = cf.vector_dot_r(n11, prec)
    n12_dot_r = cf.vector_dot_r(n12, prec)

    contributions = {
        "potential":[n11_dot_r,
                   n11[0], n11[1], n11[2],
                   n12_dot_r,
                   n12[0], n12[1], n12[2]],
        "density":[div_n22,
                     div_n22_r[0], div_n22_r[1], div_n22_r[2],
                     div_n21,
                     div_n21_r[0], div_n21_r[1], div_n21_r[2]],
        "sign":[-1,  1,  1,  1,
                 1, -1, -1, -1]
    } 

    testNorm = computeTestNorm(contributions)
    print("Test Norm B", testNorm)
    result = calcPerturbationValues(contributions, P, prec, testNorm)
    print("final Gauge B", result)
    return -0.5 * result.real

#
# one delta term excluded
# same structure as Sun 2022
# least efficeint method
#
def calcGaugePertC(spinorb1, spinorb2, mra, prec, derivative):
    print("Gauge Perturbation Version C (Sun 2022)")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    grad_n11 = cf.vector_gradient(n11, derivative)
    grad_n12 = cf.vector_gradient(n12, derivative)

    grad_n11_r = [cf.vector_dot_r([grad_n11[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][2] for i in range(3)], prec)]
    grad_n12_r = [cf.vector_dot_r([grad_n12[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][2] for i in range(3)], prec)]

    n22_r_mat = cf.vector_tensor_r(n22, prec)
    n21_r_mat = cf.vector_tensor_r(n21, prec)
    
    
    div_n22 = cf.divergence(n22, prec, derivative)
    div_n21 = cf.divergence(n21, prec, derivative)
    div_n22_r = cf.scalar_times_r(div_n22, prec)
    div_n21_r = cf.scalar_times_r(div_n21, prec)
    n11_dot_r = cf.vector_dot_r(n11, prec)
    n12_dot_r = cf.vector_dot_r(n12, prec)

    contributions = {
        "density":[n22[0], n22[1], n22[2],
                   n22_r_mat[0][0], n22_r_mat[1][0], n22_r_mat[2][0],
                   n22_r_mat[0][1], n22_r_mat[1][1], n22_r_mat[2][1],
                   n22_r_mat[0][2], n22_r_mat[1][2], n22_r_mat[2][2],
                   n21[0], n21[1], n21[2],
                   n21_r_mat[0][0], n21_r_mat[1][0], n21_r_mat[2][0],
                   n21_r_mat[0][1], n21_r_mat[1][1], n21_r_mat[2][1],
                   n21_r_mat[0][2], n21_r_mat[1][2], n21_r_mat[2][2]],

        "potential":[grad_n11_r[0], grad_n11_r[1], grad_n11_r[2],
                     grad_n11[0][0], grad_n11[0][1], grad_n11[0][2],
                     grad_n11[1][0], grad_n11[1][1], grad_n11[1][2],
                     grad_n11[2][0], grad_n11[2][1], grad_n11[2][2],
                     grad_n12_r[0], grad_n12_r[1], grad_n12_r[2],
                     grad_n12[0][0], grad_n12[0][1], grad_n12[0][2],
                     grad_n12[1][0], grad_n12[1][1], grad_n12[1][2],
                     grad_n12[2][0], grad_n12[2][1], grad_n12[2][2]],

        "sign":[ 1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    } 

    testNorm = computeTestNorm(contributions)
    print("Test Norm C", testNorm)
    result = calcPerturbationValues(contributions, P, prec, testNorm)
    print("final Gauge C", result)
    return -0.5 * result.real

#
# double delta term excluded
#
def calcGaugePertD(spinorb1, spinorb2, mra, prec, derivative):
    print("Gauge Perturbation Version D")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    grad_n11 = cf.vector_gradient(n11, derivative)
    grad_n12 = cf.vector_gradient(n12, derivative)
        
    grad_n11_r = [cf.vector_dot_r([grad_n11[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][2] for i in range(3)], prec)]

    grad_n12_r = [cf.vector_dot_r([grad_n12[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][2] for i in range(3)], prec)]

    div_n22 = cf.divergence(n22, prec, derivative)
    div_n21 = cf.divergence(n21, prec, derivative)

    div_n22_r = cf.scalar_times_r(div_n22, prec)
    div_n21_r = cf.scalar_times_r(div_n21, prec)

    n22_r_mat = cf.vector_tensor_r(n22, prec)
    n21_r_mat = cf.vector_tensor_r(n21, prec)
    
    n11_dot_r = cf.vector_dot_r(n11, prec)
    n12_dot_r = cf.vector_dot_r(n12, prec)

    contributions = {
        "potential":[n22[0], n22[1], n22[2],
                     div_n22_r[0], div_n22_r[1], div_n22_r[2],
                     n21[0], n21[1], n21[2],
                     div_n21_r[0], div_n21_r[1], div_n21_r[2]],
        "density":[grad_n11_r[0], grad_n11_r[1], grad_n11_r[2],
                   n11[0], n11[1], n11[2],
                   grad_n12_r[0], grad_n12_r[1], grad_n12_r[2],
                   n12[0], n12[1], n12[2]],
        "sign":[ 1,  1,  1,  1,  1,  1,
                -1, -1, -1, -1, -1, -1]
    } 

    testNorm = computeTestNorm(contributions)
    print("Test Norm D", testNorm)
    result = calcPerturbationValues(contributions, P, prec, testNorm)
    print("final Gauge D", result)
    return -0.5 * result.real

def calcGaugeDelta(spinorb1, spinorb2, mra, prec):
    print("Gauge Perturbation Delta")

    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    contributions = {
        "potential":[n22[0], n22[1], n22[2],
                     n21[0], n21[1], n21[2]],
        "density":[n11[0], n11[1], n11[2],
                   n12[0], n12[1], n12[2]],
        "sign":[1, 1, 1, -1, -1, -1]
    } 

    testNorm = computeTestNorm(contributions)
    print("Test Norm Delta", testNorm)
    result = calcPerturbationValues(contributions, P, prec, testNorm)
    print("final Gauge Delta", result)
    return 0.5 * result.real


def build_RHS_D2(Jop, Vop, spinor, prec, light_speed):
    c2 = light_speed**2
    Jpsi = Jop(spinor)
    Vpsi = Vop(spinor)
    VT_psi = 0.5 * Jpsi - Vpsi

    beta_VT_psi = VT_psi.beta2()
    beta_VT_psi.cropLargeSmall(prec)

    ap_VT_psi = VT_psi.alpha_p(prec)
    ap_psi = spinor.alpha_p(prec)
    VT_ap_psi = 0.5 * Jop(ap_psi) - Vop(ap_psi)
    anticom = VT_ap_psi + ap_VT_psi
    anticom *= 1.0 / (2.0 * light_speed)
    anticom.cropLargeSmall(prec)

    VT_VT_psi = 0.5 * Jop(VT_psi) - Vop(VT_psi)
    VT_VT_psi *= 1.0 / (2.0 * c2)
    VT_VT_psi.cropLargeSmall(prec)

    RHS = beta_VT_psi + anticom + VT_VT_psi
    RHS.cropLargeSmall(prec)
    return RHS 
