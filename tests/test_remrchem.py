from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
import numpy as np
import pytest
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar

c = 137   # NOT A GOOD WAY. MUST BE FIXED!!!

mra = vp.MultiResolutionAnalysis(box=[-60,60], order=4)
prec = 1.0e-3
orb.orbital4c.light_speed = c
orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra

def make_gauss_tree():
    origin = [0.1, 0.2, 0.3]  # origin moved to avoid placing the nuclar charge on a node
    a_coeff = 3.0
    b_coeff = np.sqrt(a_coeff/np.pi)**3
    gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
    gauss_tree = vp.FunctionTree(mra)
    vp.advanced.build_grid(out=gauss_tree, inp=gauss)
    vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
    gauss_tree.normalize()
    return gauss_tree;

def test_spinor():
    print("test_spinor")
    spinor_H = orb.orbital4c()
    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = make_gauss_tree())
    spinor_H.copy_components(La = La_comp)
    spinor_H.init_small_components(prec/10)
    spinor_H.normalize()
    val = spinor_H.comp_array[0].real([0.0, 0.0, 0.0])
    print(val)
    assert val == pytest.approx(0.5937902746013326)

#def test_read():
#    print("test_read")
#    spinorb1 = orb.orbital4c()
#    spinorb2 = orb.orbital4c()
#    spinorb1.read("trees/spinorb1")
#    spinorb2.read("trees/spinorb2")
#    val1 = spinorb1.comp_array[0].real([0.0, 0.0, 0.0])
#    val2 = spinorb2.comp_array[3].imag([0.0, 0.0, 0.0])
#   print(val1, val2)
#   assert val1 == pytest.approx(1.3767534073967547)
#    assert val2 == pytest.approx(-0.012619848367561309)

def test_mul():
    print("test_mul")
    spinorb1 = orb.orbital4c()
    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(La = La_comp)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1 *= 2.0
    n1 = spinorb1.squaredNorm()
    spinorb1 = spinorb1 * 2.0
    n2 = spinorb1.squaredNorm()
    spinorb1 = 2.0 * spinorb1
    n3 = spinorb1.squaredNorm()
    print(n1,n2,n3)
    assert n1 == pytest.approx(4.0)
    assert n2 == pytest.approx(16.0)
    assert n3 == pytest.approx(64.0)

def test_normalize():
    print("test_normalize")
    spinorb1 = orb.orbital4c()

    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    comp2 = cf.complex_fcn()
    comp2.copy_fcns(real = make_gauss_tree())

    spinorb1.copy_components(La = comp1, Sb = comp2)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    n1 = spinorb1.squaredNorm()
    assert n1 == pytest.approx(1.0)

def test_init_small():
    print("test_init_small")
    spinorb1 = orb.orbital4c()

    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(Lb = comp1)
    spinorb1.init_small_components(prec/10)

    val1 = spinorb1.comp_array[0].real([0.0, 0.0, 0.0])
    val2 = spinorb1.comp_array[0].imag([0.0, 0.0, 0.0])
    val3 = spinorb1.comp_array[1].real([0.0, 0.0, 0.0])
    val4 = spinorb1.comp_array[1].imag([0.0, 0.0, 0.0])
    val5 = spinorb1.comp_array[2].real([0.0, 0.0, 0.0])
    val6 = spinorb1.comp_array[2].imag([0.0, 0.0, 0.0])
    val7 = spinorb1.comp_array[3].real([0.0, 0.0, 0.0])
    val8 = spinorb1.comp_array[3].imag([0.0, 0.0, 0.0])

    assert val1 == pytest.approx(0.0)
    assert val2 == pytest.approx(0.0)                  
    assert val3 == pytest.approx(0.5938013445576377)    
    assert val4 == pytest.approx(0.0)                   
    assert val5 == pytest.approx(-0.0008088481634795703)
    assert val6 == pytest.approx(-0.0004053688109512743)
    assert val7 == pytest.approx(0.0)                   
    assert val8 == pytest.approx(0.0012122540667913475) 


