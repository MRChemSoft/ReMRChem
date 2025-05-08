from vampyr import vampyr3d as vp
import numpy as np
from scipy.special import eval_genlaguerre
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb

def make_starting_guess(mra, prec):
    gauss_tree_tot = vp.FunctionTree(mra)
    gauss_tree_tot.setZero()
    a_coeff = 3.0
    b_coeff = np.sqrt(a_coeff/np.pi)**3
    AO_list = []
    for atom in coordinates:
        gauss = vp.GaussFunc(b_coeff, a_coeff, [atom[2], atom[3], atom[4]])
        gauss_tree = vp.FunctionTree(mra)
        vp.advanced.build_grid(out=gauss_tree, inp=gauss)
        vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
        AO_list.append(gauss_tree)
    gauss_tree_sum = vp.sum(AO_list)

    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = gauss_tree_sum)
    spinorb1.copy_components(La = La_comp)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1.cropLargeSmall(prec)
    return spinorb1

def make_NR_starting_guess(position, charge, mra, prec):
    nr_wf_tree = vp.FunctionTree(mra)
    nr_wf_tree.setZero()
    n = 1
    l = 0
    Peps = vp.ScalingProjector(mra, prec)
    guess = lambda x : wf_hydrogenionic_atom(n,l,[x[0]-position[0], x[1]-position[1], x[2]-position[2]],charge)
    nr_wf_tree = Peps(guess)
    
    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = nr_wf_tree)

    spinorb1 = orb.orbital4c()
    spinorb2 = orb.orbital4c()
    spinorb1.copy_components(La = La_comp)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1.cropLargeSmall(prec)
    return spinorb1

#returns the value of the radial WF in the point r
# 1. the nucleus is assumed infintely heavy (mass of electron and Bohr radius used)
# 2. the nucleus is placed in the origin
# 3. atomic units are assumed a0 = 1  hbar = 1  me = 1  4pie0 = 1
def radial_wf_hydrogenionic_atom(n,l,r,Z):
    rho = 2 * Z * r
    slater = np.exp(-rho/2)
    polynomial = eval_genlaguerre(n-l-1, 2*l+1, rho)
    f1 = np.math.factorial(n-l-1)
    f2 = np.math.factorial(n+l)
    norm = np.sqrt((2*Z/n)**3 * f1 / (2 * n * f2))
    value = norm * rho**l * polynomial * slater
    return value

def wf_hydrogenionic_atom(n,l,position,Z):
    if(l != 0):
        print("only s orbitals for now")
        exit(-1)
    distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
    value = radial_wf_hydrogenionic_atom(n, l, distance, Z)
    return value


