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

#@profile
def calcGaugePotential(density, operator, direction, P): # direction is i index

    print("calc gauge pot ", direction)

    Bgauge = [cf.complex_fcn(), cf.complex_fcn(), cf.complex_fcn()]
    index = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    index[0][direction] += 1
    index[1][direction] += 1
    index[2][direction] += 1
    for idx in range(3):
        Bgauge[idx].real = operator(density[idx].real, index[idx][0], index[idx][1], index[idx][2])
        Bgauge[idx].imag = operator(density[idx].imag, index[idx][0], index[idx][1], index[idx][2])
        #    Bgauge[i].real = P(density[i].real)
        #    Bgauge[i].imag = P(density[i].imag)
        
    out = Bgauge[0] + Bgauge[1] + Bgauge[2]
#    out = Bgauge[idx]
    del Bgauge
    return out

#@profile
def gaugePert(spinorb1, spinorb2, mra, length, prec):
    
    P = vp.PoissonOperator(mra, prec)
    light_speed = spinorb1.light_speed

    #Definition of alpha vectors for each orbital
    alpha1 =  spinorb1.alpha_vector(prec)
    alpha2 =  spinorb2.alpha_vector(prec)

    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]

    n21 = [spinorb2.overlap_density(alpha1[0], prec),
           spinorb2.overlap_density(alpha1[1], prec),
           spinorb2.overlap_density(alpha1[2], prec)]

    for i in range(3):
        n22[i].crop(prec)
        n21[i].crop(prec)
    
    
    print("densities")
    print("n22 x y z")
    print(n22[0], n22[1], n22[2])
    print("n21 x y z")
    print(n21[0], n21[1], n21[2])


    del alpha1
    del alpha2
    R3O = r3m.GaugeOperator(mra, 1e-5, length, prec)
    print('Gauge operator DONE')

    Bgauge22 = [calcGaugePotential(n22, R3O, 0), calcGaugePotential(n22, R3O, 1), calcGaugePotential(n22, R3O, 2)]
    Bgauge21 = [calcGaugePotential(n21, R3O, 0), calcGaugePotential(n21, R3O, 1), calcGaugePotential(n21, R3O, 2)]

    print("Operators")
    print("b22 x y z")
    print(Bgauge22[0], Bgauge22[1], Bgauge22[2])
    print("b21 x y z")
    print(Bgauge21[0], Bgauge21[1], Bgauge21[2])
    # the following idientites hold for two orbitals connected by KTRS
    # n_11[i] == -n22[i]
    # n_12[i] ==  n21[i].complex_conj()

    gaugeEnergy = 0
    for i in range(3): 
#        Bgauge22 = calcGaugePotential_xx(n22, R3O, i, P)
#        Bgauge21 = calcGaugePotential_xx(n21, R3O, i, P)
        gaugeJr, gaugeJi = n22[i].complex_conj().dot(Bgauge22[i])
        gaugeKr, gaugeKi = n21[i].dot(Bgauge21[i])
        print("Direct   ", gaugeJr, gaugeJi)
        print("Exchange ", gaugeKr, gaugeKi)
        gaugeEnergy = gaugeEnergy - gaugeJr - gaugeKr
    print("Gauge energy correction ", gaugeEnergy)
    return gaugeEnergy


#@profile
def testConv(spinorb1, spinorb2, mra, length, prec):

    P = vp.PoissonOperator(mra, prec)
    alpha2 =  spinorb2.alpha_vector(prec)

    n22 = [spinorb2.overlap_density(alpha2[0], prec),
           spinorb2.overlap_density(alpha2[1], prec),
           spinorb2.overlap_density(alpha2[2], prec)]

    for i in range(3):
        n22[i].crop(prec)
    
    print("densities")
    print("n22 x y z")
    print(n22[0], n22[1], n22[2])

    del alpha2
    R3O = r3m.GaugeOperator(mra, 1e-5, length, prec)
    print('Gauge operator DONE')

    gaugeEnergy = 0
#    index = 0
    for index in range(3):
        Bgauge22 = calcGaugePotential(n22, R3O, index, P)
        print("Bgauge22 in loop")
        print(Bgauge22)
        gaugeJr, gaugeJi = n22[index].complex_conj().dot(Bgauge22)
        print("Direct   ", gaugeJr, gaugeJi)
        gaugeEnergy = gaugeEnergy - gaugeJr
    print("Gauge energy correction ", gaugeEnergy)
    return gaugeEnergy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-a', '--atype', dest='atype', type=str, default='He',
                        help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='ABGV',
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
    parser.add_argument('-o', '--order', dest='order', type=int, default=6,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-4,
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

    assert args.deriv in ['PH', 'BS', 'ABGV'], 'Please, specify the type of derivative'
    

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
    
    orb.orbital4c.mra = mra
    orb.orbital4c.light_speed = light_speed
    cf.complex_fcn.mra = mra
    print('call MRA DONE')
    
    computeNuclearPotential = False
    readOrbitals = True
    runCoulomb = False
    saveOrbitals = False
    runGaunt = False
    runGauge = False
    runGaugeA = True
    runTest = False
    default_der = args.deriv
    
    ################### Define V potential ######################
    if(computeNuclearPotential):
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
        print('Define V Potential', args.potential, 'DONE')
    
    #############################START WITH CALCULATION###################################
    
    spinorb1 = orb.orbital4c()
    spinorb2 = orb.orbital4c()
    if readOrbitals:
        spinorb1.read("spinorb1")
        spinorb2.read("spinorb2")
    else:
        a_coeff = 3.0
        b_coeff = np.sqrt(a_coeff/np.pi)**3
        gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
        gauss_tree = vp.FunctionTree(mra)
        vp.advanced.build_grid(out=gauss_tree, inp=gauss)
        vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
        gauss_tree.normalize()
        complexfc = cf.complex_fcn()
        complexfc.copy_fcns(real=gauss_tree)
        spinorb1.copy_components(La=complexfc)
        spinorb1.init_small_components(prec/10)
        spinorb1.normalize()
        spinorb2 = spinorb1.ktrs() #does this go out of scope?

    length = 2 * args.box

    if runCoulomb:
        spinorb1, spinorb2 = two_electron.coulomb_gs_2e(spinorb, V_tree, mra, prec)
    
    if runGaunt:
        two_electron.calcGauntPert(spinorb1, spinorb2, mra, prec)
    
    if runGauge:
        two_electron.calcGaugePert(spinorb1, spinorb2, mra, prec)
        
    if runGaugeA:
        two_electron.calcGaugePertA(spinorb1, spinorb2, mra, prec)
        
    if runTest:
        testConv(spinorb1, spinorb2, mra, length, prec)

    if saveOrbitals:
        spinorb1.save("spinorb1")
        spinorb2.save("spinorb2")
    
