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

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=6)
prec = 1.0e-4
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
    comp1.copy_fcns(real = make_gauss_tree(a=4.0, b=2, o=[0.1, 0.1, 0.1]),
                    imag = make_gauss_tree(a=4.0, b=2, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=4.0, b=2),
                    imag = make_gauss_tree(a=4.0, b=2))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp2)

    spinorb2 = orb.orbital4c()
    spinorb2.copy_components(La = comp1, Lb=comp2, Sa = comp2, Sb = comp2)

    return spinorb1, spinorb2
    
def test_gaunt():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGauntPert(spinorb1, spinorb2, mra, prec)
    assert val == pytest.approx(-0.15280516826011337)

def test_gauge_delta():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGaugeDelta(spinorb1, spinorb2, mra, prec)
    assert val == pytest.approx( -0.07640411954534392)

def test_gaugePertA():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGaugePertA(spinorb1, spinorb2, mra, prec, 'ABGV')
    assert val == pytest.approx(  0.03479948721660901)

def test_gaugePertB():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGaugePertB(spinorb1, spinorb2, mra, prec, 'ABGV')
    assert val == pytest.approx( -0.04160707840790333)

def test_gaugePertC():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGaugePertC(spinorb1, spinorb2, mra, prec, 'ABGV')
    assert val == pytest.approx( -0.041603919995141604)

def test_gaugePertD():
    spinorb1, spinorb2 = make_two_spinors()
    val = te.calcGaugePertD(spinorb1, spinorb2, mra, prec, 'ABGV')
    assert val == pytest.approx( -0.11800712803427525)
