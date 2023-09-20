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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='ABGV',
                        help='put the type of derivative')
    parser.add_argument('-b', '--box', dest='box', type=int, default=60,
                        help='put the box dimension')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=6,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-4,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

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
    
    ################# Call MRA #######################
    mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order, max_depth = 30)
    prec = args.prec
    orb.orbital4c.mra = mra
    orb.orbital4c.light_speed = light_speed
    cf.complex_fcn.mra = mra
    default_der = args.deriv
    print('call MRA DONE')

    ################## Jobs ##########################
    computeNuclearPotential = True
    readOrbitals            = False
    runD_1e                 = True
    runD2_1e                = False    
    runCoulombGen           = False
    runCoulomb2e            = False    
    runKutzelnigg           = False
    runKutzSimple           = False
    saveOrbitals            = False
    runGaunt                = False 
    runGaugeA               = False 
    runGaugeB               = False 
    runGaugeC               = False 
    runGaugeD               = False 
    runGaugeDelta           = False
    print('Jobs chosen') 


    ################### Reading Atoms #########################
    atomlist = 'atom_list.txt'  # Replace with the actual file name
    coordinates, total_atom_lists = nucpot.read_file_with_named_lists(atomlist)
 
    ################### Define V potential ######################
    V_tree = vp.FunctionTree(mra)
    if(computeNuclearPotential):
        typenuc = args.potential
        V_tree = nucpot.pot(coordinates, typenuc, mra, prec, default_der)
        print('V_tree', V_tree)

    ################### Define Center of Mass ###################
    if total_atom_lists >= 2:
        # Calculate the center of mass
        com_coordinates = nucpot.calculate_center_of_mass(coordinates)
        print("Center of Mass (x, y, z):", com_coordinates)
    else:
        print("There are not enough atoms (less than 2) to calculate the center of mass.")
        com_coordinates = coordinates
    
    #############################START WITH CALCULATION###################################
    
    spinorb1 = orb.orbital4c()
    spinorb2 = orb.orbital4c()
    if readOrbitals:
        spinorb1.read("spinorb1")
        spinorb2.read("spinorb2")
    else:
        gauss_tree_tot = vp.FunctionTree(mra)
        gauss_tree_tot.setZero()
        a_coeff = 3.0
        b_coeff = np.sqrt(a_coeff/np.pi)**3
        for atom, origin in coordinates.items():
            gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
            gauss_tree = vp.FunctionTree(mra)
            vp.advanced.build_grid(out=gauss_tree, inp=gauss)
            vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
            gauss_tree_tot += gauss_tree
            gauss_tree_tot.normalize()

        La_comp = cf.complex_fcn()
        La_comp.copy_fcns(real = gauss_tree_tot)
        spinorb1.copy_components(La = La_comp)
        spinorb1.init_small_components(prec/10)
        spinorb1.normalize()
        spinorb1.cropLargeSmall(prec)
        # print('Spin1', spinorb1)
        spinorb2 = spinorb1.ktrs(prec) #does this go out of scope?

    length = 2 * args.box

    if runD_1e:
        spinorb1 = one_electron.gs_D_1e(spinorb1, V_tree, mra, prec, default_der)

    if runD2_1e:
        spinorb1 = one_electron.gs_D2_1e(spinorb1, V_tree, mra, prec, default_der)

    if runCoulombGen:
        spinorb1, spinorb2 = two_electron.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec)

    if runCoulomb2e:
        spinorb1, spinorb2 = two_electron.coulomb_gs_2e(spinorb1, V_tree, mra, prec)

    if runKutzelnigg:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, default_der)

    if runKutzSimple:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, default_der)

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
        spinorb2.save("spinorb2")
    
