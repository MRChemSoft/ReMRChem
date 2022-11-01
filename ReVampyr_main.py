#################################### RELATIVISTIC VAMPYR ####################################
from vampyr import vampyr3d as vp

####################################     BASIC MODULES   ####################################
import numpy as np
import numpy.linalg as LA
import argparse
#####
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma

####################################  DEVELOPED MODULES  ####################################
#####
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import nuclear_potential as nucpot
from orbital4c import operators as opr
import importlib
importlib.reload(orb)


####################################       INPUT        ####################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collecting all data tostart the program.')
    parser.add_argument('-a', '--atype', dest='atype', type=str, default='He',
                        help='put the atom type')
    parser.add_argument('-d', '--derivative', dest='deriv', type=str, default='PH',
                        help='put the type of derivative')
    parser.add_argument('-z', '--charge', dest='charge', type=float, default=2.0,
                        help='put the atom charge')
    parser.add_argument('-g', '--guessorb', dest='guessorb', type=str, default= 'gaussian',
                        help='guess function for the spinorbital')
    parser.add_argument('-b', '--box', dest='box', type=int, default=60,
                        help='put the box dimension')
    parser.add_argument('-cx', '--center_x', dest='cx', type=float, default=0.0,
                        help='position of nucleus in x')
    parser.add_argument('-cy', '--center_y', dest='cy', type=float, default=0.0,
                        help='position of nucleus in y')
    parser.add_argument('-cz', '--center_z', dest='cz', type=float, default=0.0,
                        help='position of nucleus in z')
    parser.add_argument('-l', '--light_speed', dest='lux_speed', type=float, default=137.03599913900001,
                        help='light of speed')
    parser.add_argument('-o', '--order', dest='order', type=int, default=8,
                        help='put the order of Polinomial')
    parser.add_argument('-p', '--prec', dest='prec', type=float, default=1e-5,
                        help='put the precision')
    parser.add_argument('-e', '--coulgau', dest='coulgau', type=str, default='coulomb',
                        help='put the coulomb or gaunt')
    parser.add_argument('-v', '--potential', dest='potential', type=str, default='point_charge',
                        help='tell me wich model for V you want to use point_charge, coulomb_HFYGB, homogeneus_charge_sphere, gaussian')
    args = parser.parse_args()

    assert args.atype != 'H', 'Please consider only atoms with more than one electran'

    assert args.charge > 1.0, 'Please consider only atoms with more than one electron'

    assert args.coulgau in ['coulomb', 'gaunt'], 'Please, specify coulgau in a rigth way – coulomb or gaunt'

    assert args.potential in ['point_charge', 'smoothing_HFYGB', 'coulomb_HFYGB', 'homogeneus_charge_sphere', 'gaussian'], 'Please, specify V'

    assert args.deriv in ['PH', 'BS'], 'Please, specify the type of derivative'

    assert args.guessorb in ['slater', 'gaussian'], 'Please, specify the type of function as starting guess for spinorbitals'

####################################     PARAMETERS     ####################################
light_speed = args.lux_speed
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = args.charge
atom = args.atype

####################################   MULTI RESOLUTION ANALYSIS  ####################################
mra = vp.MultiResolutionAnalysis(box=[-args.box,args.box], order=args.order)
prec = args.prec
origin = [args.cx, args.cy, args.cz]
print('call MRA DONE')

####################################      DEFINE SPINORBITALS     ####################################
guessorb = args.guessorb
SG  = opr.SpinorbGenerator(mra, guessorb, light_speed, origin, prec)
spinorb1 = SG('La')
cspinorb1 = spinorb1.complex_conj()
spinorb2  = SG('Lb')
cspinorb2 = spinorb2.complex_conj()
spinorbv = [spinorb1, spinorb2]
cspinorbv = [cspinorb1, cspinorb2]
print('Define spinorbitals DONE')

####################################    DEFINE NUCLEAR POTENTIAL  ####################################
if args.potential == 'point_charge':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.point_charge(x, origin, Z)
   V_tree = Peps(f)
elif args.potential == 'coulomb_HFYGB':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.coulomb_HFYGB(x, origin, Z, prec)
   V_tree = Peps(f)
elif args.potential == 'homogeneus_charge_sphere':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.homogeneus_charge_sphere(x, origin, Z, atom)
   V_tree = Peps(f)
elif args.potential == 'fermi_dirac':
   Peps = vp.ScalingProjector(mra,prec/10)
   Pua = vp.PoissonOperator(mra, prec/10)
   f = lambda x: nucpot.fermi_dirac(x, origin, Z, atom)
   rho_tree = Peps(f)
   V_tree = Pua(rho_tree) * (4 * np.pi)
elif args.potential == 'gaussian':
   Peps = vp.ScalingProjector(mra,prec/10)
   f = lambda x: nucpot.gaussian(x, origin, Z, atom)
   V_tree = Peps(f)

default_der = args.deriv
print('Define V Potetintal', args.potential, 'DONE')
#
#
##############################START WITH CALCULATION###################################
if args.coulgau == 'coulomb':
    print('Hartræ-Føck (Cøulømbic bielectric interåctiøn)')
    error_norm = 1

    while error_norm > prec:

        # Initialize operators for first iteration
        J = opr.CoulombDirectOperator(mra, prec, spinorbv)
        #print('ready J', J)


        K = opr.CoulombExchangeOperator(mra, prec, spinorbv)
        #print('ready K', K)


        # Applying nuclear potential to spin orbit 1 and 2
        v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
        v_spinorbv = [v_spinorb1, v_spinorb2]
        #print('v_spinorb1', v_spinorb1)
        #print('v_spinorb2', v_spinorb2)


        #Calculate the Fock matrix (Fij)
        F = opr.FockMatrix1(prec, default_der, J, K, v_spinorbv, spinorbv)

        # Orbital Energy
        print('Energy_Spin_Orbit_1', F('orb1') - light_speed**2)
        print('Energy_Spin_Orbit_2', F('orb2') - light_speed**2)


        # Total Energy 
        print('E_total(Coulomb) approximiation', F('tot') - (2.0 *light_speed**2))


        # Apply potential operator to all orbitals
        V1 = v_spinorb1 + J(spinorb1) - K(spinorb1) - F('F12')*spinorb2
        V2 = v_spinorb2 + J(spinorb2) - K(spinorb2) - F('F21')*spinorb1
        #print('V1', V1)
        #print('V2', V2)


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V1, F('orb1'), prec)
        tmp_2 = orb.apply_helmholtz(V2, F('orb2'), prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F('orb1'), der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, F('orb2'), der = default_der)
        new_orbital_2 *= 0.5/light_speed**2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)
 
 
        # Compute overlap
        O = opr.Orthogonalize(prec, new_orbital_1, new_orbital_2)

        spinorb1 = O('spinorb1')
        spinorb2 = O('spinorb2')
        spinorbv = [spinorb1, spinorb2]
#
#   ##########
# Initialize operators for first iteration
    J = opr.CoulombDirectOperator(mra, prec, spinorbv)
    #print('ready J', J)
    
    K = opr.CoulombExchangeOperator(mra, prec, spinorbv)
    #print('ready K', K)
    
    # Applying nuclear potential to spin orbit 1 and 2
    v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    v_spinorbv = [v_spinorb1, v_spinorb2]
    #print('v_spinorb1', v_spinorb1)
    #print('v_spinorb2', v_spinorb2)
    #Calculate the Fock matrix (Fij)
    
    F = opr.FockMatrix1(prec, default_der, J, K, v_spinorbv, spinorbv)
    
    # Orbital Energy
    print('Energy_Spin_Orbit_1', F('orb1') - light_speed**2)
    print('Energy_Spin_Orbit_2', F('orb2') - light_speed**2)
    
    # Total Energy 
    print('E_total(Coulomb) approximiation', F('tot') - (2.0 *light_speed**2))
#
######################################################END COULOMB & START GAUNT#######################################################################
elif args.coulgau == 'gaunt':
    print('Hartræ-Føck (Cøulømbic-Gåunt bielectric interåctiøn)')
    error_norm = 1
    while error_norm > prec:


        alpha_10 =  spinorb1.alpha(0)
        alpha_11 =  spinorb1.alpha(1)
        alpha_12 =  spinorb1.alpha(2)

        alpha_20 =  spinorb2.alpha(0)
        alpha_21 =  spinorb2.alpha(1)
        alpha_22 =  spinorb2.alpha(2)


        alphav1 = [alpha_10, alpha_11, alpha_12]
        alphav2 = [alpha_20, alpha_21, alpha_22]

        alphav = np.matrix([[*alphav1], [*alphav2]])
        #print('alphav', alphav.shape)
        #print("alphav matrix", alphav)
        #print("alphb colomn", alphav[:,0]) #all in the column
        #print("alphav1", alphav[0,:]) # all in the row


        # Initialize operators for first iteration
        J = opr.CoulombDirectOperator(mra, prec, spinorbv)
        #print('ready J', J)


        K = opr.CoulombExchangeOperator(mra, prec, spinorbv)
        #print('ready K', K)


        GJ = opr.GauntDirectOperator(mra, prec, cspinorbv, alphav)
        print('ready GJ', GJ)
        #GJ1 = opr.GauntDirectOperator(mra, prec, spinorbv, cspinorbv, alphav1)
        #print('ready GJ1', GJ1)
        #GJ2 = opr.GauntDirectOperator(mra, prec, spinorbv, cspinorbv, alphav2)
        #print('ready GJ2', GJ2)


        GK = opr.GauntExchangeOperator(mra, prec, cspinorbv)
        print('ready GK', GK)
        #GK1 = opr.GauntExchangeOperator(mra, prec, spinorbv, cspinorbv, alphav1)
        #print('ready GK1', GK1)
        #GK2 = opr.GauntExchangeOperator(mra, prec, spinorbv, cspinorbv, alphav2)
        #print('ready GK2', GK2)


        #GJ0_1 = GJ0(alpha_10)
        #GK0_1 = GK0(alpha_10)
        ##GJ0_1 = orb.apply_complex_potential(1.0, GJ0, alpha_10, prec)
        #print('GJ0_1', GJ0_1)
        #print('GK0_1', GK0_1)


        # Applying nuclear potential to spin orbit 1 and 2
        v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
        v_spinorbv = [v_spinorb1, v_spinorb2]
        #print('v_spinorb1', v_spinorb1)
        #print('v_spinorb2', v_spinorb2)


        #Calculate the Fock matrix (Fij)
        F = opr.FockMatrix2(prec, default_der, J, K, GJ, GK, v_spinorbv, spinorbv, alphav1, alphav2)

        # Orbital Energy
        print('Energy_Spin_Orbit_1', F('orb1') - light_speed**2)
        print('Energy_Spin_Orbit_2', F('orb2') - light_speed**2)


        # Total Energy 
        print('E_total(Coulomb & Gaunt) approximiation', F('tot') - (2.0 *light_speed**2))


        # Apply potential operator to all orbitals
        #GJ_1 = GJ0(alpha_10) + GJ1(alpha_11) + GJ2(alpha_12)
        #GJ_2 = GJ0(alpha_20) + GJ1(alpha_21) + GJ2(alpha_22)
        #print('ready GJ_1', GJ_1)

        #GK_1 = GK0(alpha_10) + GK1(alpha_11) + GK2(alpha_12)
        #GK_2 = GK0(alpha_20) + GK1(alpha_21) + GK2(alpha_22)
        #print('ready GK_1', GK_1)


        V1 = v_spinorb1 + J(spinorb1) - K(spinorb1) - GJ(alphav1) + GK(alphav1) - F('F12')*spinorb2
        V2 = v_spinorb2 + J(spinorb2) - K(spinorb2) - GJ(alphav2) + GK(alphav2) - F('F21')*spinorb1
        #print('V1', V1)
        #print('V2', V2)


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V1, F('orb1'), prec)
        tmp_2 = orb.apply_helmholtz(V2, F('orb2'), prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, F('orb1'), der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, F('orb2'), der = default_der)
        new_orbital_2 *= 0.5/light_speed**2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)
 
 
        # Compute overlap
        O = opr.Orthogonalize(prec, new_orbital_1, new_orbital_2)

        spinorb1 = O('spinorb1')
        spinorb2 = O('spinorb2')
        spinorbv = [spinorb1, spinorb2]

    
    ##########
    alpha_10 =  spinorb1.alpha(0)
    alpha_11 =  spinorb1.alpha(1)
    alpha_12 =  spinorb1.alpha(2)

    alpha_20 =  spinorb2.alpha(0)
    alpha_21 =  spinorb2.alpha(1)
    alpha_22 =  spinorb2.alpha(2)


    alphav1 = [alpha_10, alpha_11, alpha_12]
    alphav2 = [alpha_20, alpha_21, alpha_22]

    alphav = np.matrix([[*alphav1], [*alphav2]])


    # Initialize operators for first iteration
    J = opr.CoulombDirectOperator(mra, prec, spinorbv)
    #print('ready J', J)


    K = opr.CoulombExchangeOperator(mra, prec, spinorbv)
    #print('ready K', K)


    GJ = opr.GauntDirectOperator(mra, prec, cspinorbv, alphav)
    print('ready GJ', GJ)



    GK = opr.GauntExchangeOperator(mra, prec, cspinorbv)
    print('ready GK', GK)


    # Applying nuclear potential to spin orbit 1 and 2
    v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    v_spinorbv = [v_spinorb1, v_spinorb2]
    #print('v_spinorb1', v_spinorb1)
    #print('v_spinorb2', v_spinorb2)


    #Calculate the Fock matrix (Fij)
    F = opr.FockMatrix2(prec, default_der, J, K, GJ, GK, v_spinorbv, spinorbv, alphav1, alphav2)


    # Orbital Energy
    print('Energy_Spin_Orbit_1', F('orb1') - light_speed**2)
    print('Energy_Spin_Orbit_2', F('orb2') - light_speed**2)


    # Total Energy 
    print('E_total(Coulomb & Gaunt) approximiation', F('tot') - (2.0 *light_speed**2))

##########################################################END###########################################################################
