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
                        help='put the coulomb, gaunt or breit to have one of them')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    parser.add_argument('-s', '--save', dest='saveOrbitals', type=str, default='No',
                        help='Save the FunctionTree for the alpha spinorbital called as Z')
    parser.add_argument('-r', '--read', dest='readOrbitals', type=str, default='No',
                        help='Read the FunctionTree for the aplha spinorbital called as Z')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt', 'breit'], 'Please, specify coulgau in a rigth way: coulomb, gaunt or breit'

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS', 'ABGV'], 'Please, specify the type of derivative'

    assert args.readOrbitals in ['Yes', 'No'], 'Please, specify if you want (Yes) or not (No) to read the FunctionTree of spinorbital'
    
    assert args.saveOrbitals in ['Yes', 'No'], 'Please, specify if you want (Yes) or not (No) to save the FunctionTree of spinorbital'

################# Define Paramters ###########################
light_speed = args.lux_speed
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = args.charge
atom = args.atype
der = args.deriv

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order)
prec = args.prec
origin = [args.cx, args.cy, args.cz]
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
print('call MRA DONE')

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
print('Define V Potential', args.potential, Z, 'DONE')

############################# Define Spinorbitals ###################################
    
if args.readOrbitals == 'Yes':
    spinorb1 = orb.orbital4c()
    spinorb1.read(args.atype)
    spinorb2 = spinorb1.ktrs()
    spinorb2.cropLargeSmall(prec)
    spinorb2.normalize()
    print('Read alpha spinorbital and defined beta spinorbital using KTRS DONE')
elif args.readOrbitals == 'No':
    a_coeff = 3.0
    b_coeff = np.sqrt(a_coeff/np.pi)**3
    gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
    gauss_tree = vp.FunctionTree(mra)
    vp.advanced.build_grid(out=gauss_tree, inp=gauss)
    vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
    gauss_tree.normalize()
    print('Define Gaussian Function DONE')

    spinorb1 = orb.orbital4c()
    complexfc = cf.complex_fcn()
    complexfc.copy_fcns(real=gauss_tree)
    spinorb1.copy_components(La=complexfc)
    spinorb1.init_small_components(prec/10, der)
    spinorb1.normalize()
    spinorb2 = spinorb1.ktrs()
    spinorb2.cropLargeSmall(prec)
    spinorb2.normalize()
    print('Define alpha and beta spinorbitals DONE')

#############################START WITH CALCULATION###################################
if args.coulgau == 'coulomb':
    E_tot_JK = 0
    spinorb1, spinorb2, E_tot_JK = two_electron.coulomb_gs_2e(spinorb1, spinorb2, V_tree, mra, prec, der)

elif args.coulgau == 'gaunt':
    E_tot_JK = 0
    spinorb1, spinorb2, E_tot_JK = two_electron.coulomb_gs_2e(spinorb1, spinorb2, V_tree, mra, prec, der)
    gaunt = 0
    gaunt = two_electron.calcGauntPert(spinorb1, spinorb2, mra, prec, gaunt)
    print('Gaunt term =', gaunt)
    print('E_total(Dirac-Coulomb-Gaunt) =', E_tot_JK - gaunt - (2.0 *light_speed**2))

elif args.coulgau == 'breit':
    E_tot_JK = 0
    spinorb1, spinorb2, E_tot_JK = two_electron.coulomb_gs_2e(spinorb1, spinorb2, V_tree, mra, prec, der)
    gaunt = 0
    gaunt = two_electron.calcGauntPert(spinorb1, spinorb2, mra, prec)
    print('Magnetic term =', 0.5 * gaunt)
    gauge1 = 0
    gauge1 = two_electron.calcGaugeDelta(spinorb1, spinorb2, mra, prec)
    gauge2 = 0
    gauge2 = two_electron.calcGaugePertB(spinorb1, spinorb2, mra, prec, der)
    print('Gauge =', gauge1 + gauge2)
    print('E_total(Dirac-Coulomb-Breit) =', E_tot_JK - 0.5 * gaunt - gauge1 - gauge2 - (2.0 *light_speed**2))
############################# Save Spinorbitals ###################################
    
if args.saveOrbitals == 'Yes':
    spinorb1.save(args.atype)
    print('Calculation is finished saving the FunctionTree of spinorbital alpha', args.atype)
elif args.saveOrbitals == 'No':
    print('Calculation is finished')
