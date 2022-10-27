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
        print('ready J', J)


        K = opr.CoulombExchangeOperator(mra, prec, spinorbv)
        print('ready K', K)


        # Applying nuclear potential to spin orbit 1 and 2
        v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
        v_spinorbv = [v_spinorb1, v_spinorb2]
        #print('v_spinorb1', v_spinorb1)
        #print('v_spinorb2', v_spinorb2)


        #Calculate the Fock matrix (Fij)

        F = opr.FockMatrix1(prec, default_der, J, K, v_spinorbv, spinorbv)

        E1 = F('orb1')
        E2 = F('orb2')

        print('Energy_Spin_Orbit_1', E1 - light_speed**2)
        print('Energy_Spin_Orbit_2', E2 - light_speed**2)


        # Total Energy with J = K approximation
        E_tot_JK = F('tot')
        print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))


        # Apply potential operator to all orbitals
        V1 = v_spinorb1 + J(spinorb1) - K(spinorb1) - F('F12')*spinorb2
        V2 = v_spinorb2 + J(spinorb2) - K(spinorb2) - F('F21')*spinorb2
        print('V1', V1)
        print('V2', V2)




        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V1, energy_1, prec)
        tmp_2 = orb.apply_helmholtz(V2, energy_2, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_1, der = default_der)
        new_orbital_1 *= 0.5/light_speed**2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_2, der = default_der)
        new_orbital_2 *= 0.5/light_speed**2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)
 
 
        # Compute overlap
        dot_11 = new_orbital_1.dot(new_orbital_1)
        dot_12 = new_orbital_1.dot(new_orbital_2)
        dot_21 = new_orbital_2.dot(new_orbital_1)
        dot_22 = new_orbital_2.dot(new_orbital_2)

        s_11 = dot_11[0] + 1j * dot_11[1]
        s_12 = dot_12[0] + 1j * dot_12[1]
        s_21 = dot_21[0] + 1j * dot_21[1]
        s_22 = dot_22[0] + 1j * dot_22[1]

        # Compute Overlap Matrix
        S_tilde = np.array([[s_11, s_12], [s_21, s_22]])

        # Compute U matrix
        sigma, U = LA.eig(S_tilde)

        # Compute matrix S^-1/2
        Sm5 = U @ np.diag(sigma ** (-0.5)) @ U.transpose()
        spinorb1 = Sm5[0, 0] * new_orbital_1 + Sm5[0, 1] * new_orbital_2
        spinorb2 = Sm5[1, 0] * new_orbital_1 + Sm5[1, 1] * new_orbital_2
        spinorb1.crop(prec)
        spinorb2.crop(prec)    

#
#   ##########
    # Initialize operators for first iteration
    J = opr.CouloumbOperator(mra, prec)
    #print('ready J', J)

     
    K = opr.ExchangeOperator(mra, prec)
    #print('ready K', K)
    

    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
    #print('hd_psi_1', hd_psi_1)
    #print('hd_psi_2', hd_psi_2)
    

    # Applying nuclear potential to spin orbit 1 and 2
    v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec) 
    #print('v_spinorb1', v_spinorb1)
    #print('v_spinorb2', v_spinorb2)
    

    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_spinorb1
    add_psi_2 = hd_psi_2 + v_spinorb2
    #print('add_psi_1', add_psi_1)
    #print('add_psi_2', add_psi_2)
    

    # Calculate Fij Fock matrix
    F_11, imag_F_11 = spinorb1.dot(add_psi_1)
    F_12, imag_F_12 = spinorb1.dot(add_psi_2)
    F_21, imag_F_21 = spinorb2.dot(add_psi_1)
    F_22, imag_F_22 = spinorb2.dot(add_psi_2)
    #print('energy_11', energy_11)
    #print('energy_12', energy_12) 
    #print('energy_21', energy_21)
    #print('energy_22', energy_22)       
    

    # Apply potential operator to all orbitals
    V1 = v_spinorb1 + J(spinorb1) - K(spinorb1, spinorb2) - F_12*spinorb2
    V2 = v_spinorb2 + J(spinorb2) - K(spinorb2, spinorb1) - F_21*spinorb1
    #print('V1', V1)
    #print('V2', V2)
    

    E_H1,  imag_H1 = spinorb1.dot(J(spinorb1))
    E_xc1, imag_K1 = spinorb1.dot(K(spinorb1, spinorb2))
    

    E_H2,  imag_H2 = spinorb2.dot(J(spinorb2))
    E_xc2, imag_K2 = spinorb2.dot(K(spinorb2, spinorb1))
    

    energy_1 = F_11 + E_H1 - E_xc1
    energy_2 = F_22 + E_H2 - E_xc2
    print('Energy_Spin_Orbit_1', energy_1 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_2 - light_speed**2)
      
      
    # Total Energy with J = K approximation
    E_tot_JK = energy_1 + energy_2 - 0.5 * (E_H1 + E_H2 - E_xc1 - E_xc2)
    print('E_total(Coulomb) approximiation', E_tot_JK - (2.0 *light_speed**2))

#
######################################################END COULOMB & START GAUNT#######################################################################
elif args.coulgau == 'gaunt':
    print('Hartræ-Føck (Cøulømbic-Gåunt bielectric interåctiøn)')
    error_norm = 1
    while error_norm > prec:

        # Initialize operators for first iteration
        J = opr.CouloumbOperator(mra, prec)

        #print('ready J', J)
        K = opr.ExchangeOperator(mra, prec)
        #print('ready K', K)

        
        # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
        #print('hd_psi_1', hd_psi_1)
        #print('hd_psi_2', hd_psi_2)
        
        # Applying nuclear potential to spin orbit 1 and 2
        v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
        v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec) 
        #print('v_spinorb1', v_spinorb1)
        #print('v_spinorb2', v_spinorb2)
        
        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + v_spinorb1
        add_psi_2 = hd_psi_2 + v_spinorb2
        #print('add_psi_1', add_psi_1)
        #print('add_psi_2', add_psi_2)


        #GAUNT: Direct (GJ) and Exchange (GK)
        #Definition of alpha(orbital)
        alpha10 =  spinorb1.alpha(0)
        alpha11 =  spinorb1.alpha(1)
        alpha12 =  spinorb1.alpha(2)
 
        
        alpha20 =  spinorb2.alpha(0)
        alpha21 =  spinorb2.alpha(1)
        alpha22 =  spinorb2.alpha(2)    


        GCO = opr.GauntCouloumbOperator(mra, prec)
        GEO = opr.GauntExchangeOperator(mra, prec)

        GJ1_0_alpha10 = GCO(alpha10, cspinorb1)
        GJ1_1_alpha11 = GCO(alpha11, cspinorb1)
        GJ1_2_alpha12 = GCO(alpha12, cspinorb1)
        GJ1 = GJ1_0_alpha10 + GJ1_1_alpha11 + GJ1_2_alpha12
        
        GJ2_0_alpha20 = GCO(alpha20, cspinorb2)
        GJ2_1_alpha21 = GCO(alpha21, cspinorb2)
        GJ2_2_alpha22 = GCO(alpha22, cspinorb2)
        GJ2 = GJ2_0_alpha20 + GJ2_1_alpha21 + GJ2_2_alpha22

        GK12_0_alpha10 = GEO(alpha10, alpha20, cspinorb1)
        GK12_1_alpha11 = GEO(alpha11, alpha21, cspinorb1)
        GK12_2_alpha12 = GEO(alpha12, alpha22, cspinorb1)
        GK12 = GK12_0_alpha10 + GK12_1_alpha11 + GK12_2_alpha12

        GK21_0_alpha20 = GEO(alpha20, alpha10, cspinorb2)
        GK21_1_alpha21 = GEO(alpha20, alpha10, cspinorb2)
        GK21_2_alpha22 = GEO(alpha20, alpha10, cspinorb2)
        GK21 = GK21_0_alpha20 + GK12_1_alpha11 + GK12_2_alpha12

        # Calculate Fij Fock matrix
        F_11, imag_F_11 = spinorb1.dot(add_psi_1)
        F_12, imag_F_12 = spinorb1.dot(add_psi_2)
        F_21, imag_F_21 = spinorb2.dot(add_psi_1)
        F_22, imag_F_22 = spinorb2.dot(add_psi_2)
        #print('energy_11', energy_11)
        #print('energy_12', energy_12) 
        #print('energy_21', energy_21)
        #print('energy_22', energy_22)

        
        E_H1,  imag_H1 = spinorb1.dot(J(spinorb1))
        E_xc1, imag_K1 = spinorb1.dot(K(spinorb1, spinorb2))
        E_H2,  imag_H2 = spinorb2.dot(J(spinorb2))
        E_xc2, imag_K1 = spinorb2.dot(K(spinorb2, spinorb1))
        

        # Definition of Energy Hartree of Fock matrix
        E_GJ110, imag_E_GJ110 = spinorb1.dot(GJ1_0_alpha10)
        E_GJ111, imag_E_GJ111 = spinorb1.dot(GJ1_1_alpha11)
        E_GJ112, imag_E_GJ112 = spinorb1.dot(GJ1_2_alpha12)
        

        E_GJ220, imag_E_GJ220 = spinorb2.dot(GJ2_0_alpha20)
        E_GJ221, imag_E_GJ221 = spinorb2.dot(GJ2_1_alpha21)
        E_GJ222, imag_E_GJ222 = spinorb2.dot(GJ2_2_alpha22)
     

        E_Gxc120, imag_E_Gxc120 = spinorb2.dot(GK12_0_alpha10)
        E_Gxc121, imag_E_Gxc121 = spinorb2.dot(GK12_1_alpha11)
        E_Gxc122, imag_E_Gxc122 = spinorb2.dot(GK12_2_alpha12)
        

        E_Gxc210, imag_E_Gxc210 = spinorb1.dot(GK21_0_alpha20)
        E_Gxc211, imag_E_Gxc211 = spinorb1.dot(GK21_1_alpha21)
        E_Gxc212, imag_E_Gxc212 = spinorb1.dot(GK21_2_alpha22)


        E_GJ1 = E_GJ110 + E_GJ111 + E_GJ112
        E_GJ2 = E_GJ220 + E_GJ221 + E_GJ222
        
        E_GK12 = E_Gxc120 + E_Gxc121 + E_Gxc122
        E_GK21 = E_Gxc210 + E_Gxc211 + E_Gxc212


        energy_1 = F_11 + E_H1 - E_xc1 - E_GJ1 + E_GK12
        energy_2 = F_22 + E_H2 - E_xc2 - E_GJ2 + E_GK21
        print('Energy_Spin_Orbit_1', energy_1 - light_speed**2)
        print('Energy_Spin_Orbit_2', energy_2 - light_speed**2)


        #Total Energy
        E_tot = energy_1 + energy_2 - 0.5 * (E_H1 + E_H2 - E_xc1 - E_xc2 - E_GJ1 - E_GJ2 + E_GK12 + E_GK21)
        print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2.0))


        # Apply potential operator to all orbitals
        V1 = v_spinorb1 + J(spinorb1) - GJ1 - K(spinorb1, spinorb2) + GK12 - F_12*spinorb2
        V2 = v_spinorb2 + J(spinorb2) - GJ1 - K(spinorb2, spinorb1) + GK21 - F_21*spinorb1
        #print('V1', V1)
        #print('V2', V2)


        # Calculation of Helmotz
        tmp_1 = orb.apply_helmholtz(V1, energy_1, prec)
        tmp_2 = orb.apply_helmholtz(V2, energy_2, prec)
        new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_1)
        new_orbital_1 *= 0.5 / light_speed ** 2
        new_orbital_1.normalize()
        new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_2)
        new_orbital_2 *= 0.5 / light_speed ** 2
        new_orbital_2.normalize()


        # Compute orbital error
        delta_psi_1 = new_orbital_1 - spinorb1
        delta_psi_2 = new_orbital_2 - spinorb2
        orbital_error = delta_psi_1 + delta_psi_2
        error_norm = np.sqrt(orbital_error.squaredNorm())
        print('Orbital_Error norm', error_norm)


        # Compute overlap
        dot_11 = new_orbital_1.dot(new_orbital_1)
        dot_12 = new_orbital_1.dot(new_orbital_2)
        dot_21 = new_orbital_2.dot(new_orbital_1)
        dot_22 = new_orbital_2.dot(new_orbital_2)
        s_11 = dot_11[0] + 1j * dot_11[1]
        s_12 = dot_12[0] + 1j * dot_12[1]
        s_21 = dot_21[0] + 1j * dot_21[1]
        s_22 = dot_22[0] + 1j * dot_22[1]


        # Compute Overlap Matrix
        S_tilde = np.array([[s_11, s_12], [s_21, s_22]])


        # Compute U matrix
        sigma, U = LA.eig(S_tilde)


        # Compute matrix S^-1/2
        Sm5 = U @ np.diag(sigma ** (-0.5)) @ U.transpose()


        # Compute the new orthogonalized orbitals
        spinorb1 = Sm5[0, 0] * new_orbital_1 + Sm5[0, 1] * new_orbital_2
        spinorb2 = Sm5[1, 0] * new_orbital_1 + Sm5[1, 1] * new_orbital_2
        spinorb1.crop(prec)
        spinorb2.crop(prec)
    
    ##########
         
    # Initialize operators for first iteration
    J = opr.CouloumbOperator(mra, prec)
        
    #print('ready J', J)
    K = opr.ExchangeOperator(mra, prec)
    #print('ready K', K)
         
        
    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
    #print('hd_psi_1', hd_psi_1)
    #print('hd_psi_2', hd_psi_2)
    
    # Applying nuclear potential to spin orbit 1 and 2
    v_spinorb1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_spinorb2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec) 
    #print('v_spinorb1', v_spinorb1)
    #print('v_spinorb2', v_spinorb2)
                    
    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_spinorb1
    add_psi_2 = hd_psi_2 + v_spinorb2
    #print('add_psi_1', add_psi_1)
    #print('add_psi_2', add_psi_2)
        
        
    #GAUNT: Direct (GJ) and Exchange (GK)
    #Definition of alpha(orbital)
    alpha10 =  spinorb1.alpha(0)
    alpha11 =  spinorb1.alpha(1)
    alpha12 =  spinorb1.alpha(2)
            
                    
    alpha20 =  spinorb2.alpha(0)
    alpha21 =  spinorb2.alpha(1)
    alpha22 =  spinorb2.alpha(2)    
         
              
    GCO = opr.GauntCouloumbOperator(mra, prec)
    GEO = opr.GauntExchangeOperator(mra, prec)
                  
    GJ1_0_alpha10 = GCO(alpha10, cspinorb1)
    GJ1_1_alpha11 = GCO(alpha11, cspinorb1)
    GJ1_2_alpha12 = GCO(alpha12, cspinorb1)
    GJ1 = GJ1_0_alpha10 + GJ1_1_alpha11 + GJ1_2_alpha12
           
    GJ2_0_alpha20 = GCO(alpha20, cspinorb2)
    GJ2_1_alpha21 = GCO(alpha21, cspinorb2)
    GJ2_2_alpha22 = GCO(alpha22, cspinorb2)
    GJ2 = GJ2_0_alpha20 + GJ2_1_alpha21 + GJ2_2_alpha22
            
         
    GK12_0_alpha10 = GEO(alpha10, alpha20, cspinorb1)
    GK12_1_alpha11 = GEO(alpha11, alpha21, cspinorb1)
    GK12_2_alpha12 = GEO(alpha12, alpha22, cspinorb1)
    GK12 = GK12_0_alpha10 + GK12_1_alpha11 + GK12_2_alpha12
        
       
    GK21_0_alpha20 = GEO(alpha20, alpha10, cspinorb2)
    GK21_1_alpha21 = GEO(alpha20, alpha10, cspinorb2)
    GK21_2_alpha22 = GEO(alpha20, alpha10, cspinorb2)
    GK21 = GK21_0_alpha20 + GK12_1_alpha11 + GK12_2_alpha12
        
           
    # Calculate Fij Fock matrix
    F_11, imag_F_11 = spinorb1.dot(add_psi_1)
    F_12, imag_F_12 = spinorb1.dot(add_psi_2)
    F_21, imag_F_21 = spinorb2.dot(add_psi_1)
    F_22, imag_F_22 = spinorb2.dot(add_psi_2)
    #print('energy_11', energy_11)
    #print('energy_12', energy_12) 
    #print('energy_21', energy_21)
    #print('energy_22', energy_22)
         
    
    E_H1,  imag_H1 = spinorb1.dot(J(spinorb1))
    E_xc1, imag_K1 = spinorb1.dot(K(spinorb1, spinorb2))
    E_H2,  imag_H2 = spinorb2.dot(J(spinorb2))
    E_xc2, imag_K1 = spinorb2.dot(K(spinorb2, spinorb1))
    
        
    # Definition of Energy Hartree of Fock matrix
    E_GJ110, imag_E_GJ110 = spinorb1.dot(GJ1_0_alpha10)
    E_GJ111, imag_E_GJ111 = spinorb1.dot(GJ1_1_alpha11)
    E_GJ112, imag_E_GJ112 = spinorb1.dot(GJ1_2_alpha12)
    
         
    E_GJ220, imag_E_GJ220 = spinorb2.dot(GJ2_0_alpha20)
    E_GJ221, imag_E_GJ221 = spinorb2.dot(GJ2_1_alpha21)
    E_GJ222, imag_E_GJ222 = spinorb2.dot(GJ2_2_alpha22)
         
       
    E_Gxc120, imag_E_Gxc120 = spinorb2.dot(GK12_0_alpha10)
    E_Gxc121, imag_E_Gxc121 = spinorb2.dot(GK12_1_alpha11)
    E_Gxc122, imag_E_Gxc122 = spinorb2.dot(GK12_2_alpha12)
    
       
    E_Gxc210, imag_E_Gxc210 = spinorb1.dot(GK21_0_alpha20)
    E_Gxc211, imag_E_Gxc211 = spinorb1.dot(GK21_1_alpha21)
    E_Gxc212, imag_E_Gxc212 = spinorb1.dot(GK21_2_alpha22)
      
        
    E_GJ1 = E_GJ110 + E_GJ111 + E_GJ112
    E_GJ2 = E_GJ220 + E_GJ221 + E_GJ222
     
    E_GK12 = E_Gxc120 + E_Gxc121 + E_Gxc122
    E_GK21 = E_Gxc210 + E_Gxc211 + E_Gxc212
      
      
    energy_1 = F_11 + E_H1 - E_xc1 - E_GJ1 + E_GK12
    energy_2 = F_22 + E_H2 - E_xc2 - E_GJ2 + E_GK21
    print('Energy_Spin_Orbit_1', energy_1 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_2 - light_speed**2)
      
     
    #Total Energy
    E_tot = energy_1 + energy_2 - 0.5 * (E_H1 + E_H2 - E_xc1 - E_xc2 - E_GJ1 - E_GJ2 + E_GK12 + E_GK21)
    print('E_total (Coulomb & Gaunt) approximiation', E_tot - 2.0 * (light_speed**2.0))
##########################################################END###########################################################################
#
