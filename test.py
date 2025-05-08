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

import one_electron as oneel
import two_electron as twoel
import starting_guess as sg

import importlib
importlib.reload(orb)

import fileinput

input_blob = ""
for line in fileinput.input():
    input_blob += line

print(input_blob)
exec(input_blob)

#
# 1. This code works now only for atoms and up to two electrons with KTRS
# 2. Orbital guess obtained by using NR hydrogenionic 1s orbital
# 3. The input file is a Python code mostly containing variable allocation
# 4. Input parsing is executing that python input after reading it
# 5. Nuclear potential selected manually (Gaussian now to reproduce Harrison's results)
#

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-box, box], order=order, max_depth = 20)
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
charge = molecule[0][1]
position = [molecule[0][2],molecule[0][3],molecule[0][4]]
radius = molecule[0][5]
epsilon = molecule[0][6]

print(charge,position)

################### Define V potential ######################
Peps = vp.ScalingProjector(mra, prec/10)
V_tree = vp.FunctionTree(mra)

if(computePotential):
    typenuc = potential
    f = 0
    if(potential == "gaussian"):
        print("Gaussian potential")
        f = lambda x: nucpot.gaussian_potential(x, position, charge, epsilon)
    elif(potential == "coulomb_HFYGB"):
        print("Harrison potential")
        f = lambda x: nucpot.coulomb_HFYGB(x, position, charge, prec/10)
    elif(potential == "point_charge"):
        print("point charge potential")
        f = lambda x: nucpot.point_charge(x, position, charge)
    else:
        exit(-1)
    V_tree = Peps(f)
elif(readPotential):
    V_tree.loadTree(f"potential")

if(savePotential):
    V_tree.saveTree(f"potential")

print("Potential")
print(V_tree)
        
print("Number of Atoms = ", len(molecule))
print(molecule)

#############################START WITH CALCULATION###################################
spinorb1 = orb.orbital4c()
spinorb2 = orb.orbital4c()
if readOrbitals:
    spinorb1.read(orbitalName)
else:
    spinorb1 = sg.make_NR_starting_guess(position, charge, mra, prec)
spinorb2 = spinorb1.ktrs(prec)

if saveGuess:
    spinorb1.save("guess1")

run_D_1e       = scf and not D2 and not two_electrons
run_D2_1e      = scf and     D2 and not two_electrons
run_D_2e       = scf and not D2 and     two_electrons and not ktrs
run_D2_2e      = scf and     D2 and     two_electrons and not ktrs
run_D_2e_ktrs  = scf and not D2 and     two_electrons and     ktrs
run_D2_2e_ktrs = scf and     D2 and     two_electrons and     ktrs


length = 2 * box
print("Using derivative ", derivative)

if run_D_1e:
    spinorb1 = oneel.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge)

if run_D2_1e:
    spinorb1 = oneel.gs_D2_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge)

if run_D_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = twoel.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D2_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = twoel.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D_2e_ktrs:
    spinorb1, spinorb2 = twoel.coulomb_gs_2e(spinorb1, V_tree, mra, prec, thr, derivative)

if run_D2_2e_ktrs:
    spinorb1, spinorb2 = twoel.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, thr, derivative)

if runGaunt:
    twoel.calcGauntPert(spinorb1, spinorb2, mra, prec)

if runGaugeA:
    twoel.calcGaugePertA(spinorb1, spinorb2, mra, prec)

if runGaugeB:
    twoel.calcGaugePertB(spinorb1, spinorb2, mra, prec)

if runGaugeC:
    twoel.calcGaugePertC(spinorb1, spinorb2, mra, prec)

if runGaugeD:
    twoel.calcGaugePertD(spinorb1, spinorb2, mra, prec)

if runGaugeDelta:
    twoel.calcGaugeDelta(spinorb1, spinorb2, mra, prec)

if saveOrbitals:
    spinorb1.save("spinorb1")

