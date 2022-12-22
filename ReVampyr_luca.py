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

import importlib
importlib.reload(orb)


################# Define Paramters ###########################
light_speed = 137
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = 1
atom = "H"
################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-60,60], order=8)
prec = 1.0e-4
origin = [0.0, 0.0,  0.0]
print('call MRA DONE')

################# Definecs Gaussian function ########## 
a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
gauss_tree = vp.FunctionTree(mra)
vp.advanced.build_grid(out=gauss_tree, inp=gauss)
vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
gauss_tree.normalize()
print('Define Gaussian Function DONE')

################ Define orbital as complex function ######################
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
complexfc = cf.complex_fcn()
complexfc.copy_fcns(imag=gauss_tree)
print('Define orbital as a complex function DONE')

################ Define spinorbitals ########## 
spinorb1 = orb.orbital4c()
spinorb1.copy_components(La=complexfc)
spinorb1.init_small_components(prec/10)
spinorb1.normalize()
#print('spinorb1',spinorb1)
#print('cspinorb1',cspinorb1)

spinorb2 = orb.orbital4c()
spinorb2.copy_components(Lb=complexfc)
spinorb2.init_small_components(prec/10)
spinorb2.normalize()
#print('spinorb2',spinorb2)
#print('cspinorb2',cspinorb2)

print('Define spinorbitals DONE')

cspinorb1 = spinorb1.complex_conj()
cspinorb2 = spinorb2.complex_conj()

#print(spinorb1)
#print(cspinorb1)
print(spinorb1+cspinorb1)

