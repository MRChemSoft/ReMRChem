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
    #parser.add_argument('-a', '--atype', dest='atype', type=str, default='He',
    #                    help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='ABGV',
                        help='put the type of derivative')
    #parser.add_argument('-z', '--charge', dest='charge', type=float, default=2.0,
    #                    help='put the atom charge')
    parser.add_argument('-b', '--box', dest='box', type=int, default=60,
                        help='put the box dimension')
    #parser.add_argument('-cx', '--center_x', dest='cx', type=float, default=0.0,
    #                    help='position of nucleus in x')
    #parser.add_argument('-cy', '--center_y', dest='cy', type=float, default=0.0,
    #                    help='position of nucleus in y')
    #parser.add_argument('-cz', '--center_z', dest='cz', type=float, default=0.0,
    #                    help='position of nucleus in z')
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

    #assert args.atype != 'H', 'Please consider only atoms with more than one electron'

    #assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

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
    
    def read_file_with_named_lists(atomlist):
        atom_lists = {}
    
        with open(atomlist, 'r') as file:
            for line in file:
                terms = line.strip().split()
                atom = terms[0]
                origin = terms[1:]  
                origin = [float(element) for element in origin]   
    
                if atom in atom_lists:
                    # Append an identifier to make the key unique
                    identifier = len(atom_lists[atom]) + 1
                    unique_key = f"{atom}_{identifier}"
                    atom_lists[unique_key] = origin
                else:
                    atom_lists[atom] = origin
    
        return atom_lists 
    
    def get_original_list_name(key):
        return key.split('_')[0]
    
    atomlist = './atom_list.txt'  # Replace with the actual file name
    coordinates = read_file_with_named_lists(atomlist)

    #origin = [args.cx, args.cy, args.cz]
    #Z = args.charge
    #atom = args.atype
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
    readOrbitals            = True
    runCoulomb              = False
    runCoulombGen           = False
    runKutzelnigg           = False
    runKutzSimple           = True
    saveOrbitals            = False
    runGaunt                = False 
    runGaugeA               = False 
    runGaugeB               = False 
    runGaugeC               = False 
    runGaugeD               = False 
    runGaugeDelta           = False
    print('Jobs chosen') 
    
    ################### Define V potential ######################
    if(computeNuclearPotential):
        V_tot = vp.FunctionTree(mra)
        V_tot.setZero()
        for atom, origin in coordinates.items():
            atom = get_original_list_name(atom)
            print("Atom:", atom)
            fileObj = open("./Z.txt", "r")
            charge = ""
            for line in fileObj:
                if not line.startswith("#"):
                    line = line.strip().split()
                    if len(line) == 2:
                        if line[0] == atom:
                            charge = float(line[1])
                            print("Charge:", charge)
            fileObj.close()
            print("Origin:", origin)
            print()  # Print an empty line for separation
            f = lambda x: nucpot.point_charge(x, origin, charge)
            Peps = vp.ScalingProjector(mra,prec/10)
            V_tree = Peps(f)
            V_tot += V_tree
        
            #Peps = vp.ScalingProjector(mra,prec/10)
            #f = lambda x: nucpot.point_charge(x, origin, charge)
            #V_tree = Peps(f)
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
        spinorb1.cropLargeSmall(prec)
        spinorb2 = spinorb1.ktrs(prec) #does this go out of scope?

    length = 2 * args.box

    if runCoulombGen:
        spinorb1, spinorb2 = two_electron.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec)
    
    if runKutzelnigg:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, 'ABGV')

    if runKutzSimple:
        spinorb1, spinorb2 = two_electron.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, der = 'ABGV')

    if runCoulomb:
        spinorb1, spinorb2 = two_electron.coulomb_gs_2e(spinorb1, V_tree, mra, prec)

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
    
