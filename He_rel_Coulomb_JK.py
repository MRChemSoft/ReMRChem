########## Define Enviroment #################
from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import NuclearPotential as nucpot
from orbital4c import complex_fcn as cf
import numpy as np
from scipy.linalg import eig, inv
import importlib
importlib.reload(orb)
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from scipy.constants import hbar
import numpy.linalg as LA

################# Define Paramters ###########################
light_speed = 137.03604 
alpha = 1/light_speed
k = -1
l = 0
n = 1
m = 0.5
Z = 10
atom = "Ne"

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-60,60], order=8)
prec = 1.0e-5
origin = [0.1, 0.2, 0.3]  # origin moved to avoid placing the nuclar charge on a node

################# Define Gaussian function ########## 
a_coeff = 3.0
b_coeff = np.sqrt(a_coeff/np.pi)**3
gauss = vp.GaussFunc(b_coeff, a_coeff, origin)
gauss_tree = vp.FunctionTree(mra)
vp.advanced.build_grid(out=gauss_tree, inp=gauss)
vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
gauss_tree.normalize()

################ Define orbital as complex function ######################
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
complexfc = cf.complex_fcn()
complexfc.copy_fcns(real=gauss_tree)

################ Define spinorbitals ########## 
spinorb1 = orb.orbital4c()
spinorb1.copy_components(La=complexfc)
spinorb1.init_small_components(prec/10)
spinorb1.normalize()

spinorb2 = orb.orbital4c()
spinorb2.copy_components(Lb=complexfc)
spinorb2.init_small_components(prec/10)
spinorb2.normalize()

################### Define V potential ######################
Peps = vp.ScalingProjector(mra,prec)
f = lambda x: nucpot.GausChD(x, origin, Z, atom)
V_tree = Peps(f)

default_der = 'PH'

################# Working on Helium with Coulomb direct (CJ) & exchange (CK) ################

error_norm = 1

while error_norm > prec:
    # Definition of different densities
    n_11 = spinorb1.density(prec)
    n_12 = spinorb1.exchange(spinorb2, prec)
    n_21 = spinorb2.exchange(spinorb1, prec)
    n_22 = spinorb2.density(prec)
    

    # Definition of Poisson operator
    Pua = vp.PoissonOperator(mra, prec)
    
    # Defintion of Jx
    J11 = Pua(n_11) * (4 * np.pi)
    J12 = Pua(n_12) * (4 * np.pi)
    J21 = Pua(n_21) * (4 * np.pi)
    J22 = Pua(n_22) * (4 * np.pi)


    # Definition of Kx
    K1 = Pua(n_12) * (4 * np.pi)
    K2 = Pua(n_21) * (4 * np.pi)


    # Definition of Energy Hartree of Fock matrix
    E_H11 = vp.dot(n_11, J11)
    E_H12 = vp.dot(n_12, J12)
    E_H21 = vp.dot(n_21, J21)
    E_H22 = vp.dot(n_22, J22)

    
    # Definition of Energy Exchange of Fock matrix
    E_xc11 = vp.dot(n_12, K2)
    E_xc12 = vp.dot(n_12, K1)
    E_xc21 = vp.dot(n_21, K2)
    E_xc22 = vp.dot(n_21, K1)


    # Definiton of Dirac Hamiltonian for spin orbit 1 and 2
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
    

    # Applying nuclear potential to spin orbit 1 and 2
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    

    # Definition of full 4c hamitoninan
    add_psi_1 = hd_psi_1 + v_psi_1
    add_psi_2 = hd_psi_2 + v_psi_2
    

    # Calculate Fij Fock matrix
    energy_11, imag_11 = spinorb1.dot(add_psi_1)
    energy_12, imag_12 = spinorb1.dot(add_psi_2)
    energy_21, imag_21 = spinorb2.dot(add_psi_1)
    energy_22, imag_22 = spinorb2.dot(add_psi_2)
    
    # Orbital Energy calculation
    energy_11 = energy_11 + E_H11 - E_xc11
    energy_12 = energy_12 + E_H12 - E_xc12
    energy_21 = energy_21 + E_H21 - E_xc21
    energy_22 = energy_22 + E_H22 - E_xc22
    print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
    print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)
    

    # Total Energy with J = K approximation
    E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22)
    print("E_total(Coulomb) approximiation", E_tot_JK - (2.0 *light_speed**2))
    
    
    # Calculation of necessary potential contributions to Hellmotz
    J_spinorb1   = orb.apply_potential(1.0, J11, spinorb1, prec)
    K_spinorb1   = orb.apply_potential(1.0, K2, spinorb2, prec)
    F12_spinorb2 =  energy_12 * spinorb2
    
    
    J_spinorb2   = orb.apply_potential(1.0, J22, spinorb2, prec)
    K_spinorb2   = orb.apply_potential(1.0, K1, spinorb1, prec)
    F21_spinorb1 = energy_21 * spinorb1
    
    
    V_J_K_spinorb1 = v_psi_1 + J_spinorb1 - K_spinorb1 - F12_spinorb2
    V_J_K_spinorb2 = v_psi_2 + J_spinorb2 - K_spinorb2 - F21_spinorb1
    

    # Calculation of Helmotz
    tmp_1 = orb.apply_helmholtz(V_J_K_spinorb1, energy_11, prec)
    tmp_2 = orb.apply_helmholtz(V_J_K_spinorb2, energy_22, prec)
    new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_11, der = default_der)
    new_orbital_1 *= 0.5/light_speed**2
    new_orbital_1.normalize()
    new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_22, der = default_der)
    new_orbital_2 *= 0.5/light_speed**2
    new_orbital_2.normalize()
    

    # Compute orbital error
    delta_psi_1 = new_orbital_1 - spinorb1
    delta_psi_2 = new_orbital_2 - spinorb2
    orbital_error = delta_psi_1 + delta_psi_2
    error_norm = np.sqrt(orbital_error.squaredNorm())
    print("Orbital_Error norm", error_norm)
    

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

# Definition of different densities
n_11 = spinorb1.density(prec)
n_12 = spinorb1.exchange(spinorb2, prec)
n_21 = spinorb2.exchange(spinorb1, prec)
n_22 = spinorb2.density(prec)


# Definition of Poisson operator
Pua = vp.PoissonOperator(mra, prec)


# Defintion of Jx
J11 = Pua(n_11) * (4 * np.pi)
J12 = Pua(n_12) * (4 * np.pi)
J21 = Pua(n_21) * (4 * np.pi)
J22 = Pua(n_22) * (4 * np.pi)


# Definition of Kx
K1 = Pua(n_12) * (4 * np.pi)
K2 = Pua(n_21) * (4 * np.pi)


# Definition of Energy Hartree of Fock matrix
E_H11 = vp.dot(n_11, J11)
E_H12 = vp.dot(n_12, J12)
E_H21 = vp.dot(n_21, J21)
E_H22 = vp.dot(n_22, J22)

    
# Definition of Energy Exchange of Fock matrix
E_xc11 = vp.dot(n_12, K2)
E_xc12 = vp.dot(n_12, K1)
E_xc21 = vp.dot(n_21, K2)
E_xc22 = vp.dot(n_21, K1)

    
# Definiton of Dirac Hamiltonian for spin orbit 1 and 2
hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0, der = default_der)
hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0, der = default_der)
    

# Applying nuclear potential to spin orbit 1 and 2
v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    

# Definition of full 4c hamitoninan
add_psi_1 = hd_psi_1 + v_psi_1
add_psi_2 = hd_psi_2 + v_psi_2
    

# Calculate Fij Fock matrix
energy_11, imag_11 = spinorb1.dot(add_psi_1)
energy_12, imag_12 = spinorb1.dot(add_psi_2)
energy_21, imag_21 = spinorb2.dot(add_psi_1)
energy_22, imag_22 = spinorb2.dot(add_psi_2)
    

# Orbital Energy calculation
energy_11 = energy_11 + E_H11 - E_xc11
energy_12 = energy_12 + E_H12 - E_xc12
energy_21 = energy_21 + E_H21 - E_xc21
energy_22 = energy_22 + E_H22 - E_xc22
print('Energy_Spin_Orbit_1', energy_11 - light_speed**2)
print('Energy_Spin_Orbit_2', energy_22 - light_speed**2)
    

# Total Energy with J = K approximation
E_tot_JK = energy_11 + energy_22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22)
print("E_total(Coulomb) approximiation", E_tot_JK - 2.0 *(light_speed**2))
    

#########################################################END###########################################################################
