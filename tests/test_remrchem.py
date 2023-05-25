from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
import numpy as np
import pytest
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar
from orbital4c import nuclear_potential as nucpot


c = 137   # NOT A GOOD WAY. MUST BE FIXED!!!

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

def test_spinor():
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

def test_orb_der():
    spinorb1 = orb.orbital4c()
    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(Lb = comp1)
    spinorb1.init_small_components(prec/10)

    spinorb_x = spinorb1.derivative(0,'ABGV')
    spinorb_y = spinorb1.derivative(1,'PH')
    spinorb_z = spinorb1.derivative(2,'BS')

    val_x = spinorb_x.comp_array[1].real([0.0, 0.0, 0.0])
    val_y = spinorb_y.comp_array[1].real([0.0, 0.0, 0.0])
    val_z = spinorb_z.comp_array[1].real([0.0, 0.0, 0.0])

    print(val_x, val_y, val_z)

    assert val_x == pytest.approx(0.11107105420064928)
    assert val_y == pytest.approx(0.2215088864001861)
    assert val_z == pytest.approx(0.3324683072038884)
    
def test_gradient():
    spinorb1 = orb.orbital4c()
    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(Lb = comp1)
    spinorb1.init_small_components(prec/10)

    grad1 = spinorb1.gradient('ABGV')

    val_x = grad1[0].comp_array[1].real([0.0, 0.0, 0.0])
    val_y = grad1[1].comp_array[1].real([0.0, 0.0, 0.0])
    val_z = grad1[2].comp_array[1].real([0.0, 0.0, 0.0])

    print(val_x, val_y, val_z)

    assert val_x == pytest.approx(0.11107105420064928)
    assert val_y == pytest.approx(0.2216243967934017)
    assert val_z == pytest.approx(0.33215761430082913)

def test_commplex_conj():
    spinorb1 = orb.orbital4c()
    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(Lb = comp1)
    spinorb1.init_small_components(prec/10)
    spinorb2 = spinorb1.complex_conj()

    val1 = []
    val1.append(spinorb1.comp_array[0].real([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[0].imag([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[1].real([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[1].imag([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[2].real([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[2].imag([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[3].real([0.0, 0.0, 0.0]))
    val1.append(spinorb1.comp_array[3].imag([0.0, 0.0, 0.0]))
    
    val2 = []
    val2.append(spinorb2.comp_array[0].real([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[0].imag([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[1].real([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[1].imag([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[2].real([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[2].imag([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[3].real([0.0, 0.0, 0.0]))
    val2.append(spinorb2.comp_array[3].imag([0.0, 0.0, 0.0]))
    
    for i in range(4):
        assert val1[2*i] == pytest.approx(val2[2*i])
        assert val1[2*i+1] == pytest.approx(-val2[2*i+1])
        print(val1[2*i], val2[2*i], val1[2*i+1], val2[2*i+1])

def test_density():
    spinorb1 = orb.orbital4c()
    comp1 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree())
    spinorb1.copy_components(Lb = comp1)
    spinorb1.init_small_components(prec/10)

    dens = spinorb1.density(prec)

    val = dens([0.0, 0.0, 0.0])
    print(val)
    
    assert val == pytest.approx(0.3526038291701747)

def test_dot():    
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    spinorb2 = orb.orbital4c()
    spinorb2.copy_components(La = comp2, Lb = comp1)
    spinorb2.init_small_components(prec/10)

    rval, ival = spinorb1.dot(spinorb2)
    assert rval == pytest.approx(2.5307429558268524)
    assert ival == pytest.approx(-0.41650020718445097)
    print(rval, ival)
    
def test_overlap_density():    
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    spinorb2 = orb.orbital4c()
    spinorb2.copy_components(La = comp2, Lb = comp1)
    spinorb2.init_small_components(prec/10)

    overlap = spinorb1.overlap_density(spinorb2, prec)
    val = overlap([0.0, 0.0, 0.0])
    assert np.real(val) == pytest.approx(0.0006138637025866606)
    assert np.imag(val) == pytest.approx(0.0005440086610743796)
    print(np.real(val), np.imag(val))
    
def test_alhpa():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)
    alphaorb = spinorb1.alpha_vector(prec)

    alpharef = [[(-8.79837474224956e-05  + 5.16840633678686e-05j),
                 (-9.888532586901115e-05 - 0.008155058903998369j),
                 (0.05586501146881823    + 6.573755692841616e-07j),
                 (0.05586501146881823    + 6.573755692841616e-07j)],
                [(5.16840633678686e-05   + 8.79837474224956e-05j),
                 (0.008155058903998369   - 9.888532586901115e-05j),
                 (6.573755692841616e-07  - 0.05586501146881823j),
                 (-6.573755692841616e-07 + 0.05586501146881823j)],
                [(-9.888532586901115e-05 - 0.008155058903998369j),
                 (8.79837474224956e-05   - 5.16840633678686e-05j),
                 (0.05586501146881823    + 6.573755692841616e-07j),
                 (-0.05586501146881823   - 6.573755692841616e-07j)]]

    alphaval = []
    for i in range(3):
        alphaval.append(alphaorb[i]([0.0, 0.0, 0.0]))
        assert alphaval[i] == pytest.approx(alpharef[i])

def test_ktrf():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    spinorb2 = spinorb1.ktrs()

    ref = [(-0.05586501146881823  + 6.573755692775872e-07j),
           (0.05586501146881823   - 6.573755692775872e-07j),
           (-0.008157636079677332 + 1.0267337790164823e-07j),
           (-0.008157442583912674 + 0.008155058903998369j)]
    
    val = spinorb2([0.0, 0.0, 0.0])
    
    assert val == pytest.approx(ref)
    
def test_beta():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    spinorb2 = spinorb1.beta(-10000)

    ref = [( 489.88028557006885 +   0.005764526367003977j),
           ( 489.88028557006885 +   0.005764526367003977j),
           ( 234.6814656965828  + 234.61288960912916j),
           (-234.68703237623416 -   0.0029538104088736587j)]
    val = spinorb2([0.0, 0.0, 0.0])

    assert val == pytest.approx(ref)

def test_dirac_hamiltonian():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)
    spinorb2 = orb.apply_dirac_hamiltonian(spinorb1, prec, shift = 0.0, der = 'ABGV')

    ref = [(1037.4502661754984  -  22.342943384772816j),
           (1037.4382651830165  +  22.37666158042346j),
           (149.39508249498584  - 154.86998194884805j),
           (-156.41340549020896 +   1.9381892940766117j)]

    val = spinorb2([0.0, 0.0, 0.0])
#    print(val)

    assert val == pytest.approx(ref)

def test_helmholtz():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)
    spinorb2 = orb.apply_helmholtz(spinorb1, -1.0, prec)

    ref = [(-6.293490117272163e-06 - 8.965435259478884e-09j),
           (-6.293490117272163e-06 - 8.965435259478884e-09j),
           ( 9.143310240291369e-07 + 9.208886070849977e-07j),
           (-9.142891378936467e-07 - 1.275274465630999e-10j)]

    val = spinorb2([0.0, 0.0, 0.0])
#    print(val)

    assert val == pytest.approx(ref)

def test_potential():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    Z = 1
    origin = [0.0, 0.0, 0.0]
    Peps = vp.ScalingProjector(mra, prec/10)
    f = lambda x: nucpot.coulomb_HFYGB(x, origin, Z, prec)
    V_tree = Peps(f)

    spinorb2 = orb.apply_potential(1.0, V_tree, spinorb1, prec)

    ref = [( 14.787758416216949  + 0.0002419106108011777j),
           ( 14.787758416216949  + 0.0002419106108011777j),
           ( -2.1592751577438762 - 2.1586264932674926j),
           (  2.1593264201323197 + 2.5033498645473497e-05j)]

    val = spinorb2([0.0, 0.0, 0.0])
#    print(val)

    assert val == pytest.approx(ref)

def test_complex_potential():
    comp1 = cf.complex_fcn()
    comp2 = cf.complex_fcn()
    comp1.copy_fcns(real = make_gauss_tree(a=1.3, b=100, o=[0.1, 0.2, 0.1]),
                    imag = make_gauss_tree(a=1.0, b=200, o=[0.1, 0.2, 0.2]))
    comp2.copy_fcns(real = make_gauss_tree(a=1.0, b=90),
                    imag = make_gauss_tree(a=1.0, b=200))

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La = comp1, Lb=comp1)
    spinorb1.init_small_components(prec/10)

    radius = nucpot.get_param_homogeneous_charge_sphere("Ne")
    Peps = vp.ScalingProjector(mra, prec/10)
    f = lambda x: nucpot.point_charge(x, [0.0, 0.0, 0.0], 1)
    g = lambda x: nucpot.homogeneus_charge_sphere(x, [0.1, 0.1, 0.1], 2, radius)

    potential = cf.complex_fcn()
    potential.real = Peps(f)
    potential.imag = Peps(g)

    spinorb2 = orb.apply_complex_potential(1.0, potential, spinorb1, prec)

    ref = [( 30418384.270450525 +     358.58475214030517j),
           ( 30418384.270450525 +     358.58475214030517j),
           ( -4441710.637901596 - 4440412.880598812j),
           (  4441816.090120285 +      55.99966995055429j)]

    val = spinorb2([0.0, 0.0, 0.0])
#    print(val)

    assert val == pytest.approx(ref)


    
