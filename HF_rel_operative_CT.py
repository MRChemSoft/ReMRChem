########## Define Enviroment #################
from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
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
c = 137   # NOT A GOOD WAY. MUST BE FIXED!!!
alpha = 1/c
k = -1
l = 0
n = 1
m = 0.5
Z = 2

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-20,20], order=6)
prec = 1.0e-3
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
def u(r):
    u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
    #erf(r) is an error function that is supposed to stop the potential well from going to inf.
    #if i remember correctly
    return u

def V(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    c = 0.0435
    c = 0.000435 # ten times tighter nuclear potential
    f_bar = u(r/c)/c
    return f_bar

Peps = vp.ScalingProjector(mra,prec)
f = lambda x: V([x[0]-origin[0],x[1]-origin[1],x[2]-origin[2]])

print(f([0.5,0.5,0.5]))
V_tree = Z*Peps(f)

################# Working on Helium with Coulomb direct (CJ) & exchange (CK) ################

# Definition of Poisson operator 
Pua = vp.PoissonOperator(mra,prec)

error_norm = 1
#for idx in range(1):
while error_norm > prec:
    # 1# Definition of alpha and beta densities
    n_alpha = spinorb1.density(prec)
    n_beta = spinorb2.density(prec)
    #print("n_alpha", n_alpha)
    #print("n_beta", n_beta)
    
    # 2# Definition of total density
    n_tot = n_alpha + n_beta
    #print("n_tot", n_tot)
    
    # 3# Definition of J_psi
    J_psi1 = (4*np.pi) * Pua(n_tot)*spinorb1
    J_psi2 = (4*np.pi) * Pua(n_tot)*spinorb2
    #print(J_psi1)
    #print(J_psi2)
    
    # 4# Definition of pseudo density made by differnt spin orbitals
    n_21 = spinorb2.exchange(spinorb1, prec)
    n_12 = spinorb1.exchange(spinorb2, prec)
    #print(n_21)
    #print(n_12)
    
    # 5# Definition K_psi
    K_11 = (4*np.pi) * Pua(n_alpha) * spinorb1
    K_12 = (4*np.pi) * Pua(n_12) * spinorb2
    K_21 = (4*np.pi) * Pua(n_21) * spinorb1
    K_22 = (4*np.pi) * Pua(n_beta) * spinorb2
    K_psi1 = K_11 + K_12
    K_psi2 = K_21 + K_22
    #print(K_psi1)
    #print(K_psi2)
    
    
    # 6# Define v_psi
    v_psi1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    #print(v_psi1)
    #print(v_psi2)
    

    # 7# Definiton of Dirac Hamiltonian for alpha and beta components   
    hd_psi1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0)
    hd_psi2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0)
    #print(hd_psi1)
    #print(hd_psi2)


    # 8# hd_psi + v_psi + J_psi - K_psi
    add_psi1 = hd_psi1 + v_psi1 + J_psi1 - K_psi1
    add_psi2 = hd_psi2 + v_psi2 + J_psi2 - K_psi2
    #print(add_psi1)
    #print(add_psi2)   
    

    # 9# Calculate Fij Fock matrix
    energy_11, imag_11 = spinorb1.dot(add_psi1)
    energy_12, imag_12 = spinorb1.dot(add_psi2)
    energy_21, imag_21 = spinorb2.dot(add_psi1)
    energy_22, imag_22 = spinorb2.dot(add_psi2)
    #print(energy_11, imag_11)
    #print(energy_12, imag_12)
    #print(energy_21, imag_21)
    #print(energy_22, imag_22)
    
    
    # 10# Print orbital energy
    print("Orb_Energy_spin1", energy_11)
    print("Orb_Energy_spin2", energy_22)
    
    
    # 11# Preparation to calculate Total Energy step 1
    e_J1, imag_e_J1 = spinorb1.dot(J_psi1)
    e_K1, imag_e_K1 = spinorb1.dot(K_psi1)
    e_J2, imag_e_J2 = spinorb2.dot(J_psi2)
    e_K2, imag_e_K2 = spinorb2.dot(K_psi2)
    #print(e_J1)
    #print(e_J2)
    #print(e_K1)
    #print(e_K2)
    
    
    # 12# Preparation to calculate Total Energy step 2
    e_J = (e_J1 + e_J2) * 0.5
    e_K = (e_K1 + e_K2) * 0.5
    #print(e_J)
    #print(e_K)
    

    # 13# Preparation to calculate Total Energy step 3
    e_v1, imag_v1 = spinorb1.dot(v_psi1)
    e_v2, imag_v2 = spinorb2.dot(v_psi2)
    #print(e_v1)
    #print(e_v2)
    e_hd1, imag_hd1 = spinorb1.dot(hd_psi1)
    e_hd2, imag_hd2 = spinorb2.dot(hd_psi2)
    #print(e_hd1)
    #print(e_hd2)
    
    
    # 14# Print Total Energy
    tot_energy = e_hd1 + e_hd2 + e_v1 +e_v2 + e_J - e_K
    print("Total_Energy", tot_energy)
    
    
    # 15# Calculation of necessary potential contributions to Helmotz g
    g1 = v_psi1 + J_psi1 - K_psi1 - energy_12*spinorb2 - energy_21*spinorb1
    g2 = v_psi2 + J_psi2 - K_psi2 - energy_21*spinorb1 - energy_12*spinorb2 
    #print("g1", tmp_g1)
    #print("g2", tmp_g2)


    # 16# Calculation of non relativistic Helmotz 
    print("applying helmholtz kernel")
    tmp1 = orb.apply_helmholtz(g1, energy_11, c, prec)
    tmp2 = orb.apply_helmholtz(g2, energy_22, c, prec)


    # 17# Calculation of necessary potential contributions to Helmotz (hd + epsilon) on g 
    new_orbital_1 = orb.apply_dirac_hamiltonian(tmp1, energy_11, prec)
    new_orbital_1 *= 0.5/c**2
    new_orbital_1.normalize()
    #print("new_orbital_1", new_orbital_1)
    new_orbital_2 = orb.apply_dirac_hamiltonian(tmp2, energy_22, prec)
    new_orbital_2 *= 0.5/c**2
    new_orbital_2.normalize()
    #print("new_orbital_2", new_orbital_2)


    # 18# Compute orbital error 
    delta_psi_1 = new_orbital_1 - spinorb1
    delta_psi_2 = new_orbital_2 - spinorb2
    orbital_error = delta_psi_1 + delta_psi_2
    error_norm = np.sqrt(orbital_error.squaredNorm())
    print("Orbital_Error norm", error_norm)

    
    # 19# Compute overlap 
    dot_11 = new_orbital_1.dot(new_orbital_1)
    dot_12 = new_orbital_1.dot(new_orbital_2)
    dot_21 = new_orbital_2.dot(new_orbital_1)
    dot_22 = new_orbital_2.dot(new_orbital_2)
    s_11 = dot_11[0] + 1j * dot_11[1]
    s_12 = dot_12[0] + 1j * dot_12[1]
    s_21 = dot_21[0] + 1j * dot_21[1]
    s_22 = dot_22[0] + 1j * dot_22[1]
    
    
    # 20# Compute Overlap Matrix
    S_tilde = np.array([[s_11, s_12], [s_21, s_22]])
    print("S_tilde", S_tilde)
    
    # 21# Compute U matrix
    sigma, U = LA.eig(S_tilde)
    
    
    # 22# Compute matrix S^-1/2
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
    
    
    # 23# Compute the new orthogonalized orbitals
    spinorb1 = Sm5[0,0] * new_orbital_1 + Sm5[0,1] * new_orbital_2
    spinorb2 = Sm5[1,0] * new_orbital_1 + Sm5[1,1] * new_orbital_2
    
    spinorb1.crop(prec)
    spinorb2.crop(prec)


##########

# 1# Definition of alpha and beta densities
n_alpha = spinorb1.density(prec)
n_beta = spinorb2.density(prec)
#print("n_alpha", n_alpha)
#print("n_beta", n_beta)
    

# 2# Definition of total density
n_tot = n_alpha + n_beta
#print("n_tot", n_tot)
    

# 3# Definition of J_psi
J_psi1 = (4*np.pi) * Pua(n_tot)*spinorb1
J_psi2 = (4*np.pi) * Pua(n_tot)*spinorb2
#print(J_psi1)
#print(J_psi2)
    

# 4# Definition of pseudo density made by differnt spin orbitals
n_21 = spinorb2.exchange(spinorb1, prec)
n_12 = spinorb1.exchange(spinorb2, prec)
#print(n_21)
#print(n_12)


# 5# Definition K_psi
K_11 = (4*np.pi) * Pua(n_alpha) * spinorb1
K_12 = (4*np.pi) * Pua(n_12) * spinorb2
K_21 = (4*np.pi) * Pua(n_21) * spinorb1
K_22 = (4*np.pi) * Pua(n_beta) * spinorb2
K_psi1 = K_11 + K_12
K_psi2 = K_21 + K_22
#print(K_psi1)
#print(K_psi2)    
    

# 6# Define v_psi
v_psi1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
v_psi2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
#print(v_psi1)
#print(v_psi2)
    

# 7# Definiton of Dirac Hamiltonian for alpha and beta components   
hd_psi1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0)
hd_psi2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0)
#print(hd_psi1)
#print(hd_psi2)


# 8# hd_psi + v_psi + J_psi - K_psi
add_psi1 = hd_psi1 + v_psi1 + J_psi1 - K_psi1
add_psi2 = hd_psi2 + v_psi2 + J_psi2 - K_psi2
#print(add_psi1)
#print(add_psi2)   
    

# 9# Calculate Fij Fock matrix
energy_11, imag_11 = spinorb1.dot(add_psi1)
energy_12, imag_12 = spinorb1.dot(add_psi2)
energy_21, imag_21 = spinorb2.dot(add_psi1)
energy_22, imag_22 = spinorb2.dot(add_psi2)
#print(energy_11, imag_11)
#print(energy_12, imag_12)
#print(energy_21, imag_21)
#print(energy_22, imag_22)
    
    
# 10# Print orbital energy
print("Orb_Energy_spin1", energy_11)
print("Orb_Energy_spin2", energy_22)
    
    
# 11# Preparation to calculate Total Energy step 1
e_J1, imag_e_J1 = spinorb1.dot(J_psi1)
e_K1, imag_e_K1 = spinorb1.dot(K_psi1)
e_J2, imag_e_J2 = spinorb2.dot(J_psi2)
e_K2, imag_e_K2 = spinorb2.dot(K_psi2)
#print(e_J1)
#print(e_J2)
#print(e_K1)
#print(e_K2)
    
    
# 12# Preparation to calculate Total Energy step 2
e_J = (e_J1 + e_J2) * 0.5
e_K = (e_K1 + e_K2) * 0.5
#print(e_J)
#print(e_K)
    

# 13# Preparation to calculate Total Energy step 3
e_v1, imag_v1 = spinorb1.dot(v_psi1)
e_v2, imag_v2 = spinorb2.dot(v_psi2)
#print(e_v1)
#print(e_v2)
e_hd1, imag_hd1 = spinorb1.dot(hd_psi1)
e_hd2, imag_hd2 = spinorb2.dot(hd_psi2)
#print(e_hd1)
#print(e_hd2)
    
    
# 14# Print Total Energy
tot_energy = e_hd1 + e_hd2 + e_v1 +e_v2 + e_J - e_K
print("Total_Energy", tot_energy)
    

#########################################################END###########################################################################





