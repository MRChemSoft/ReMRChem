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

import one_electron
import two_electron

import importlib
importlib.reload(orb)

def read_step_file(filename):
    steps = {}
    with open(filename, 'r') as file:
        for line in file:
            terms = line.strip().split()
            option = terms[0]
            value = terms[1]
            steps[option] = value
    return steps
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='ABGV',
                        help='put the type of derivative')
    parser.add_argument('-b', '--box', dest='box', type=int, default=100,
                        help='put the box dimension')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=10,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-4,
                        help='put the precision')
    parser.add_argument('-t', '--threshold', dest='thr', type=float, default=1e-4,
                        help='put the orbital threshold')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='coulomb_HFYGB',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.coulgau in ['coulomb', 'gaunt', 'gaunt-test'], 'Please, specify coulgau in a rigth way: coulomb or gaunt'

    assert args.potential in ['point_charge', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS', 'ABGV'], 'Please, specify the type of derivative'

    ################# Define Paramters ###########################
    light_speed = args.lux_speed
    alpha = 1/light_speed
    k = -1
    l = 0
    n = 1
    m = 0.5

    ################# Call MRA #######################
    mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order, max_depth = 30)
    prec = args.prec
    thr = args.thr
    orb.orbital4c.mra = mra
    orb.orbital4c.light_speed = light_speed
    cf.complex_fcn.mra = mra
    derivative = args.deriv
    print('call MRA DONE')

    steplist = 'step_list.txt'
    atomlist = 'atom_list.txt'
    steps = read_step_file(steplist)
    coordinates, number = nucpot.read_file_with_named_lists(atomlist)

    ################## Jobs ##########################
    computeNuclearPotential = False
    readOrbitals            = False
    readPotential           = False
    runD_1e                 = False
    runD2_1e                = False
    runCoulombGen           = False
    runCoulomb2e            = False
    runKutzelnigg           = False
    runKutzSimple           = False
    saveOrbitals            = False
    savePotential           = False
    runGaunt                = False
    runGaugeA               = False
    runGaugeB               = False
    runGaugeC               = False
    runGaugeD               = False
    runGaugeDelta           = False

    for key in steps:
        flag = (steps[key] == 'True')
        locals()[key] = flag
        print(key, steps[key], locals()[key])

    if(computeNuclearPotential): print("hello")

    ################### Reading Atoms #########################
    steplist = 'step_list.txt'
    atomlist = 'atom_list.txt'
    coordinates, number = nucpot.read_file_with_named_lists(atomlist)


    ################### Define V potential ######################
    V_tree = vp.FunctionTree(mra)
    if(computeNuclearPotential):
        Peps = vp.ScalingProjector(mra, prec/10)
        typenuc = args.potential
        f = lambda x: nucpot.nuclear_potential(x, coordinates, typenuc, mra, prec, derivative)
        V_tree = Peps(f)
        print("Define V", args.potential, "DONE")
        com_coordinates = nucpot.calculate_center_of_mass(coordinates)
    if(savePotential):
        V_tree.saveTree(f"potential")
        
    if(readPotential):
        V_tree.loadTree(f"potential")

    print("Number of Atoms = ", number)
    print(coordinates)

    #############################START WITH CALCULATION###################################
    spinorb1 = orb.orbital4c()
    spinorb2 = orb.orbital4c()
    if readOrbitals:
        spinorb1.read("spinorb1")
        spinorb2 = spinorb1.ktrs(prec)
    else:
        gauss_tree_tot = vp.FunctionTree(mra)
        gauss_tree_tot.setZero()
        a_coeff = 3.0
        b_coeff = np.sqrt(a_coeff/np.pi)**3
        AO_list = []
        for atom in coordinates.values():
            gauss = vp.GaussFunc(b_coeff, a_coeff, [atom[2], atom[3], atom[4]])
            gauss_tree = vp.FunctionTree(mra)
            vp.advanced.build_grid(out=gauss_tree, inp=gauss)
            vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
            AO_list.append(gauss_tree)
        if number == 1:
            gauss_tree_tot = AO_list[0]
        elif number == 2:
            gauss_tree_tot = AO_list[0] + AO_list[1]
        gauss_tree_tot.normalize()

        La_comp = cf.complex_fcn()
        La_comp.copy_fcns(real = gauss_tree_tot)
        spinorb1.copy_components(La = La_comp)
        spinorb1.init_small_components(prec/10)
        spinorb1.normalize()
        spinorb1.cropLargeSmall(prec)
        spinorb2 = spinorb1.ktrs(prec)

    length = 2 * args.box
    print("Using derivative ", derivative)

    if runD_1e:
        spinorb1 = one_electron.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative)

    if runD2_1e:
        spinorb1 = one_electron.gs_D2_1e(spinorb1, V_tree, mra, prec, thr, derivative)

    if runCoulombGen:
        spinorb1, spinorb2 = two_electron.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec, derivative)

    if runCoulomb2e:
        spinorb1, spinorb2 = two_electron.coulomb_gs_2e(spinorb1, V_tree, mra, prec, derivative)

    if runKutzelnigg:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, derivative)

    if runKutzSimple:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, derivative)

    if runGaunt:
        two_electron.calcGauntPert(spinorb1, spinorb2, mra, prec)

    if runGaugeA:
        two_electron.calcGaugePertA(spinorb1, spinorb2, mra, prec)

    if runGaugeB:
        two_electron.calcGaugePertB(spinorb1, spinorb2, mra, prec)

    if runGaugeC:
        two_electron.calcGaugePertC(spinorb1, spinorb2, mra, prec)

    if runGaugeD:
        two_electron.calcGaugePertD(spinorb1, spinorb2, mra, prec)

    if runGaugeDelta:
        two_electron.calcGaugeDelta(spinorb1, spinorb2, mra, prec)

    if saveOrbitals:
        spinorb1.save("spinorb1")
        #spinorb2.save("spinorb2")

