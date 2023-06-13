from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbitla4c import operators as oper
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from vampyr import vampyr3d as vp
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
        spinorb2 = spinorb1.ktrs()
        spinorb2.cropLargeSmall(prec)
        spinorb2.normalize()
        n_22 = spinorb2.overlap_density(spinorb2, prec)

        # Definition of two electron operators
        B22    = P(n_22.real) * (4 * np.pi)

        # Definiton of Dirac Hamiltonian for spinorbit 1 that due to TRS is equal spinorbit 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der)
        hd_11 = spinorb1.dot(hd_psi_1)

        # Applying nuclear potential to spin orbit 1 and 2
        v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        V1 = spinorb1.dot(v_psi_1)

        hd_V_11 = hd_11 + V1

        # Calculation of two electron terms
        J2_phi1 = orb.apply_potential(1.0, B22, spinorb1, prec)

        JmK_phi1 = J2_phi1  # K part is zero for 2e system in GS
        JmK = spinorb1.dot(JmK_phi1)

        # Calculate Fij Fock matrix
        eps = hd_V_11.real + JmK.real
        E_tot_JK =  2*eps - JmK.real

        print('Spinor Energy', eps - light_speed**2)
        print('E_total(Dirac-Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))
        if(compute_last_energy):
            break

        V_J_K_spinorb1 = v_psi_1 + JmK_phi1

        mu = calc_dirac_mu(eps, light_speed)
        tmp = orb.apply_helmholtz(V_J_K_spinorb1, mu, prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, eps, der)
        new_orbital *= 0.5/light_speed**2
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
    return(spinorb1, spinorb2, E_tot_JK)

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
def calcGaugePertA(spinorb1, spinorb2, mra, prec, der):
    print("Gauge Perturbation Version A")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)


    div_n22 = cf.divergence(n22, prec, der)
    div_n21 = cf.divergence(n21, prec, der)
    n11_dot_r = cf.vector_dot_r(n11, prec)
    n12_dot_r = cf.vector_dot_r(n12, prec)

    n22_r_mat = cf.vector_tensor_r(n22, prec)
    n21_r_mat = cf.vector_tensor_r(n21, prec)

    grad_n11 = cf.vector_gradient(n11, der)
    grad_n12 = cf.vector_gradient(n12, der)

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

def calcGaugePertB(spinorb1, spinorb2, mra, prec, der):
    print("Gauge Perturbation Version B")

    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    div_n22 = cf.divergence(n22, prec, der)
    div_n21 = cf.divergence(n21, prec, der)
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
def calcGaugePertC(spinorb1, spinorb2, mra, prec, der):
    print("Gauge Perturbation Version C (Sun 2022)")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    grad_n11 = cf.vector_gradient(n11, der)
    grad_n12 = cf.vector_gradient(n12, der)

    grad_n11_r = [cf.vector_dot_r([grad_n11[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][2] for i in range(3)], prec)]
    grad_n12_r = [cf.vector_dot_r([grad_n12[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][2] for i in range(3)], prec)]

    n22_r_mat = cf.vector_tensor_r(n22, prec)
    n21_r_mat = cf.vector_tensor_r(n21, prec)
    
    
    div_n22 = cf.divergence(n22, prec, der)
    div_n21 = cf.divergence(n21, prec, der)
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
def calcGaugePertD(spinorb1, spinorb2, mra, prec, der):
    print("Gauge Perturbation Version D")
    projection_operator = vp.ScalingProjector(mra, prec)
    P = vp.PoissonOperator(mra, prec)
    n11 = calcAlphaDensityVector(spinorb1, spinorb1, prec)
    n12 = calcAlphaDensityVector(spinorb1, spinorb2, prec)
    n21 = calcAlphaDensityVector(spinorb2, spinorb1, prec)
    n22 = calcAlphaDensityVector(spinorb2, spinorb2, prec)

    grad_n11 = cf.vector_gradient(n11, der)
    grad_n12 = cf.vector_gradient(n12, der)
        
    grad_n11_r = [cf.vector_dot_r([grad_n11[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n11[i][2] for i in range(3)], prec)]

    grad_n12_r = [cf.vector_dot_r([grad_n12[i][0] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][1] for i in range(3)], prec),
                  cf.vector_dot_r([grad_n12[i][2] for i in range(3)], prec)]

    div_n22 = cf.divergence(n22, prec, der)
    div_n21 = cf.divergence(n21, prec, der)

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
