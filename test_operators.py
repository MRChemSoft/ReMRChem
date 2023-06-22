########## Define Enviroment #################
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import operators as oper
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
    readOrbitals = False
    runCoulomb = False
    saveOrbitals = False
    runGaunt = True
    runGaugeA = True
    runGaugeB = True
    runGaugeC = True
    runGaugeD = True
    runGaugeDelta = True
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

#    Vop = oper.PotentialOperator(mra, prec, V_tree)
#    Fop = oper.FockOperator(mra, prec, [Jop, Kop, Vop], [1.0, -1.0, -1.0])
#    Dop = oper.FockOperator(mra, prec, [], [])

#    Fmat = Fop.matrix([spinorb1, spinorb2])
#    print("Fmat")
#    print(Fmat)
    

#    print("Kmat")

#    Vmat = Vop.matrix([spinorb1, spinorb2])
#    print("Vmat")
#    print(Vmat)

#    Dmat = Dop.matrix([spinorb1, spinorb2])
#    print("Dmat")
#    print(Dmat)

    Jop = oper.CoulombDirectOperator(mra, prec, [spinorb1, spinorb2])
    Jmat = Jop.matrix([spinorb1, spinorb2])
    Kop = oper.CoulombExchangeOperator(mra, prec, [spinorb1, spinorb2])
    Kmat = Kop.matrix([spinorb1, spinorb2])
    print("Jmat")
    print(Jmat)
    
    P = vp.PoissonOperator(mra, prec)
    n11 = spinorb1.overlap_density(spinorb1, prec)
    n22 = spinorb2.overlap_density(spinorb2, prec)
    print("Kmat")
    print(Kmat)
#    print("density outside")
#    n = n11 + n22
#    print ("rho outside")
#    print(n.real)
#    pot    = P(n.real) * (4 * np.pi)
#    J2_phi1 = orb.apply_potential(1.0, pot, spinorb1, prec)
#    Jval = spinorb1.dot(J2_phi1)
#    print(Jval)
    #print(Jmat - Kmat + Dmat - Vmat)
