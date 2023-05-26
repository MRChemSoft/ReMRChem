from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
import numpy as np
import pytest
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar
from orbital4c import nuclear_potential as nucpot
import two_electron as te

c = 137

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=4)
prec = 1.0e-3
orb.orbital4c.light_speed = c
orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra

def make_gauss_tree(a = 3.0, b = 0, o = [0.1, 0.2, 0.3]):
    a = 3.0
    if (b == 0):
        b = np.sqrt(a/np.pi)**3
    gauss = vp.GaussFunc(b, a, o)
    gauss_tree = vp.FunctionTree(mra)
    vp.advanced.build_grid(out=gauss_tree, inp=gauss)
    vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
    gauss_tree.normalize()
    return gauss_tree

def make_two_spinors():    
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=4.0, b=2, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=4.0, b=2, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=4.0, b=2),
                    imag = make_gauss_tree(a=4.0, b=2))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    spinorb2 = orb.orbital4c()
    spinorb2.copy_components(La = comp2, Lb = comp2)
    spinorb2.init_small_components(prec/10)

    return spinorb1, spinorb2
    
def test_gaunt():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGauntPert(spinorb1, spinorb2, mra, prec)
    print(val)
    return

