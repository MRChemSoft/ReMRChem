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
mra = vp.MultiResolutionAnalysis(box=[-20,20], order=7)
prec = 1.0e-4
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

# Working on Helium with direct (J) = exchange (K)

orbital_error = 1
while orbital_error > prec:

    #Definiton of Dirac Hamiltonian for alpha component   
    hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec, 0.0)
    #Definiton of Dirac Hamiltonian for beta component 
    hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, 0.0)


    #Definition of alpha density
    n_alpha = spinorb1.density(prec)
    #Definition of beta density
    n_beta = spinorb2.density(prec)


    #Definition of Poisson operator 
    Pua = vp.PoissonOperator(mra,prec=0.000001)
    #Defintion of J_alpha
    Puatree_alpha = Pua(n_alpha)*(4*np.pi)
    #Definiton of J_beta
    Puatree_beta = Pua(n_beta)*(4*np.pi)
    #Definition of Energy Hartree for alpha component
    E_H_alpha = vp.dot(n_alpha,Puatree_beta)
    #Definition of Energy Hartree for beta component
    E_H_beta = vp.dot(n_beta,Puatree_alpha)
    
    print("E_H_alpha",E_H_alpha)
    print("E_H_beta",E_H_beta)


    #Applying nuclear potential to spin orbital 
    v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
    v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    

    #Definition of full 4c hamitoninan 
    add_psi_1 = hd_psi_1 + v_psi_1 
    add_psi_2 = hd_psi_2 + v_psi_2
    

    #Orbital Energy calculation
    energy_1, imag_1 = spinorb1.dot(add_psi_1)
    energy_2, imag_2 = spinorb2.dot(add_psi_2)
    energy_1 = energy_1 + E_H_alpha 
    energy_2 = energy_2 + E_H_beta 
    print('Energy_1', energy_1, imag_1)
    print('Energy_2', energy_2, imag_2)
    
    #Total Energy with J = K approximation 
    E_tot_J = energy_1 + energy_2 - E_H_alpha
    print("E_total(Coulomb) approximiation", E_tot_J)


    #Calculation of necessary potential contributions to Hellmotz 
    J_spinorb1 = orb.apply_potential(1.0, Puatree_alpha, spinorb1, prec)

    V_J_spinorb1 = v_psi_1 + J_spinorb1 

    J_spinorb2 = orb.apply_potential(1.0, Puatree_beta, spinorb2, prec)

    V_J_spinorb2 = v_psi_2 + J_spinorb2

    #Calculation of Helmotz
    tmp_1 = orb.apply_helmholtz(V_J_spinorb1, energy_1, c, prec)
    tmp_2 = orb.apply_helmholtz(V_J_spinorb2, energy_2, c, prec)

    new_orbital_1 = orb.apply_dirac_hamiltonian(tmp_1, prec, energy_1)
    new_orbital_1.normalize()
    new_orbital_2 = orb.apply_dirac_hamiltonian(tmp_2, prec, energy_2)
    new_orbital_2.normalize()

    delta_psi_1 = new_orbital_1 - spinorb1
    orbital_error, imag_1 = delta_psi_1.dot(delta_psi_1)
    print('Error_1',orbital_error, imag_1)
    spinorb1 = new_orbital_1

    delta_psi_2 = new_orbital_2 - spinorb2
    orbital_error, imag_2 = delta_psi_2.dot(delta_psi_2)
    print('Error_2',orbital_error, imag_2)
    spinorb2 = new_orbital_2


#Definiton of Dirac Hamiltonian for alpha component   
hd_psi_1 = orb.apply_dirac_hamiltonian(spinorb1, prec)
#Definiton of Dirac Hamiltonian for beta component 
hd_psi_2 = orb.apply_dirac_hamiltonian(spinorb2, prec, shift=0)


#Definition of alpha density
n_alpha = spinorb1.density(prec)
#Definition of beta density
n_beta = spinorb2.density(prec)


#Definition of Poisson operator 
Pua = vp.PoissonOperator(mra,prec=0.000001)
#Defintion of J_alpha
Puatree_alpha = Pua(n_alpha)*(4*np.pi)
#Definiton of J_beta
Puatree_beta = Pua(n_beta)*(4*np.pi)
#Definition of Energy Hartree for alpha component
E_H_alpha = vp.dot(n_alpha,Puatree_beta)
#Definition of Energy Hartree for beta component
E_H_beta = vp.dot(n_beta,Puatree_alpha)
    
print("E_H_alpha",E_H_alpha)
print("E_H_beta",E_H_beta)


#Applying nuclear potential to spin orbital 
v_psi_1 = orb.apply_potential(-1.0, V_tree, spinorb1, prec)
v_psi_2 = orb.apply_potential(-1.0, V_tree, spinorb2, prec)
    

#Definition of full 4c hamitoninan 
add_psi_1 = hd_psi_1 + v_psi_1
add_psi_2 = hd_psi_2 + v_psi_2
    

#Orbital Energy calculation
energy_1, imag_1 = spinorb1.dot(add_psi_1)
energy_2, imag_2 = spinorb2.dot(add_psi_2)
energy_1 = energy_1 + E_H_alpha 
energy_2 = energy_2 + E_H_beta 
print('Energy_1', energy_1, imag_1)
print('Energy_2', energy_2, imag_2)
    
#Total Energy with J = K approximation 
E_tot_J = energy_1 + energy_2 - E_H_alpha
print("E_total(Coulomb) approximiation", E_tot_J)

#########################################################END###########################################################################





