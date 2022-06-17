###################### Define Enviroment ######################
from vampyr import vampyr3d as vp
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
from orbital4c import operators as opr
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

################## TEST STEP BY STEP ###############################
################# Working on Helium with Coulomb direct (CJ) & exchange (CK) ################


# Working on Helium with Coulomb direct (CJ) & exchange (CK)
J_n = opr.CouloumbOperator(mra, spinorb_vec, prec)
K_n = opr.ExchangeOperator(mra, spinorb_vec, prec)
print("J_n", J_n)
print("K_n", K_n)


def calc_overlap(Bra,Ket):
     S_m = np.empty((len(Bra), len(Ket)))
     for i in range(len(Bra)):
        for j in range(len(Ket)):            
            S_m[i, j] = (Bra[i].dot(Ket[j])[0])
     return S_m



hd_psi_vec = [] 
v_psi_vec = [] 
orb_energy_vec = []
tmp = []
new_orbital_vec = []


######## Calculate hd_psi and v_psi #########################################
for i in range(spinorb_vec.shape[0]):
    hd_psi = (orb.apply_dirac_hamiltonian(spinorb_vec[i], prec, 0.0))
    v_psi = (orb.apply_potential(-1.0, V_tree, spinorb_vec[i], prec))
    hd_psi.crop(prec)
    v_psi.crop(prec)
    hd_psi_vec.append(hd_psi)
    v_psi_vec.append(v_psi)
hd_psi_vec = np.array(hd_psi_vec)
v_psi_vec = np.array(v_psi_vec)
print(hd_psi_vec)
print(v_psi_vec)


############ Calculate J_psi and K_psi #############################
J_psi_vec = J_n(spinorb_vec)
K_psi_vec = K_n(spinorb_vec)
print("J_psi_vec", J_psi_vec)
print("K_psi_vec", K_psi_vec)


################ Calculation of hd_psi + v_psi + J_psi - K_psi  #####################
add_psi_vec = hd_psi_vec + v_psi_vec + J_psi_vec - K_psi_vec
print(add_psi_vec)


##################### Calculating of Fij with i /= j  and Lamda matrix ##########################
Fij_matrix = calc_overlap(spinorb_vec, add_psi_vec)
print("Fock_Matrix", Fij_matrix)
Lambda_n = np.diag(np.diag(Fij_matrix))
print("Lambda_Matrix", Lambda_n)


################# Calculation of Orbital Energies ############################################
orb_energy_vec = np.diag(Fij_matrix)
print('Orb_Energy', orb_energy_vec)


################ Preparation to calculate the Total Energy ###############################
V_mat = calc_overlap(spinorb_vec, vPhi_vec)
J_mat = calc_overlap(spinorb_vec, JPhi_vec)
K_mat = calc_overlap(spinorb_vec, KPhi_vec)
print("V_mat", V_mat)
print("J_mat", J_mat)
print("K_mat", K_mat)


################# Calculation of Total Energy ###################################
E_orb  =  Fij_matrix.trace()
print("E_orb", E_orb)
E_en   =  V_mat.trace()
print("E_en", E_en)
E_coul =  0.5*J_mat.trace()
print("E_coul", E_coul)
E_ex   =  -0.5*K_mat.trace()
print("E_ex", E_ex)
E_tot  = E_orb - E_coul - E_ex
print("E_total", E_tot)


################### Calculation of necessary potential contributions to Helmotz (hd + epsilon) ####################
for i in range(spinorb_vec.shape[0]):
    tmp.append(orb.apply_dirac_hamiltonian(spinorb_vec[i], prec, orb_energy_vec[i]))
    tmp[i].normalize()    
tmp = np.array(tmp)
print("[hd+epsilon]psi", tmp)


####################  Calculation of necessary potential contributions to Helmotz g ###############################
g = add_psi_vec + (Lambda_n - Fij_matrix) @ spinorb_vec
print("g", g)


######################  Calculation of Helmotz  ###############################################
for i in range(0, tmp.shape[0]): 
    new_orbital_vec.append(orb.apply_helmholtz(g[i], tmp[i], c, prec))
new_orbital_vec = np.array(new_orbital_vec)
print("new_orbital_vec", new_orbital_vec)


######################  Compute orbital error  #######################################
delta_psi = new_orbital_vec - spinorb_vec
orbital_error = np.array([delta_psi.norm() for delta_psi in delta_psi])
print(orbital_error)


################## Compute Overlap Matrix ###################################
S_tilde = calc_overlap(new_orbital_vec, new_orbital_vec)
print(S_tilde)    


################################## Löwdin orthonormalization S^{-1/2} = U * Sigma^{-1/2} * U^T
sigma, U = LA.eig(S_tilde)
print("sigma", sigma)
Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
print("Sm5", Sm5)
spinorb_vec = Sm5 @  new_orbital_vec
print("spinorb_vec", spinorb_vec)


########################## Crop Orbital ###############################################
for orbital in spinorb_vec:
    orbital.crop(prec)



################## LOOP ###############################
################# Working on Helium with Coulomb direct (CJ) & exchange (CK) ################
J_n = opr.CouloumbOperator(mra, spinorb_vec, prec)
K_n = opr.ExchangeOperator(mra, spinorb_vec, prec)
print("J_n", J_n)
print("K_n", K_n)
    
    
def calc_overlap(Bra,Ket):
    S_m = np.empty((len(Bra), len(Ket)))
    for i in range(len(Bra)):
        for j in range(len(Ket)):            
            S_m[i, j] = (Bra[i].dot(Ket[j])[0])
    return S_m
    
#SCF parameters
orbital_error = np.ones(len(spinorb_vec))
print("Orbital_Error", orbital_error)

# SCF loop
while (max(orbital_error)) > prec:
    
    
    hd_psi_vec = [] 
    v_psi_vec = [] 
    orb_energy_vec = []
    tmp = []
    new_orbital_vec = []
    
    
    ######## Calculate hd_psi and v_psi #########################################
    for i in range(spinorb_vec.shape[0]):
        hd_psi = (orb.apply_dirac_hamiltonian(spinorb_vec[i], prec, 0.0))
        v_psi = (orb.apply_potential(-1.0, V_tree, spinorb_vec[i], prec))
        hd_psi.crop(prec)
        v_psi.crop(prec)
        hd_psi_vec.append(hd_psi)
        v_psi_vec.append(v_psi)
    hd_psi_vec = np.array(hd_psi_vec)
    v_psi_vec = np.array(v_psi_vec)
    
    
    ############ Calculate J_psi and K_psi #############################
    J_psi_vec = J_n(spinorb_vec)
    K_psi_vec = K_n(spinorb_vec)
    
    
    ################ Calculation of hd_psi + v_psi + J_psi - K_psi  #####################
    add_psi_vec = hd_psi_vec + v_psi_vec + J_psi_vec - K_psi_vec
    
    
    ##################### Calculating of Fij with i /= j  and Lamda matrix ##########################
    Fij_matrix = calc_overlap(spinorb_vec, add_psi_vec)
    Lambda_n = np.diag(np.diag(Fij_matrix))
    
    
    ################# Calculation of Orbital Energies ############################################
    orb_energy_vec = np.diag(Fij_matrix)
    print('Orb_Energy', orb_energy_vec)
    
    
    ################ Preparation to calculate the Total Energy ###############################
    V_mat = calc_overlap(spinorb_vec, vPhi_vec)
    J_mat = calc_overlap(spinorb_vec, JPhi_vec)
    K_mat = calc_overlap(spinorb_vec, KPhi_vec)
    
    
    ################# Calculation of Total Energy ###################################
    E_orb  =  Fij_matrix.trace()
    E_en   =  V_mat.trace()
    E_coul =  0.5*J_mat.trace()
    E_ex   =  -0.5*K_mat.trace()
    E_tot  = E_orb - E_coul - E_ex
    print("E_total", E_tot)
    
    
    ################### Calculation of necessary potential contributions to Helmotz (hd + epsilon) ####################
    for i in range(spinorb_vec.shape[0]):
        tmp.append(orb.apply_dirac_hamiltonian(spinorb_vec[i], prec, orb_energy_vec[i]))
        tmp[i].normalize()    
    tmp = np.array(tmp)
    
    
    ####################  Calculation of necessary potential contributions to Helmotz g ###############################
    g = add_psi_vec + (Lambda_n - Fij_matrix) @ spinorb_vec
    
    
    ######################  Calculation of Helmotz  ###############################################
    for i in range(0, tmp.shape[0]): 
        new_orbital_vec.append(orb.apply_helmholtz(g[i], tmp[i], c, prec))
    new_orbital_vec = np.array(new_orbital_vec)
    
    
    ######################  Compute orbital error  #######################################
    delta_psi = new_orbital_vec - spinorb_vec
    orbital_error = np.array([delta_psi.norm() for delta_psi in delta_psi])
    
    
    ################## Compute Overlap Matrix ###################################
    S_tilde = calc_overlap(new_orbital_vec, new_orbital_vec)
        

    ################################## Löwdin orthonormalization S^{-1/2} = U * Sigma^{-1/2} * U^T
    sigma, U = LA.eig(S_tilde)
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
    spinorb_vec = Sm5 @  new_orbital_vec
    
    
    ########################## Crop Orbital ###############################################
    for orbital in spinorb_vec:
        orbital.crop(prec)

#########

hd_psi_vec = [] 
v_psi_vec = [] 
orb_energy_vec = []
tmp = []
new_orbital_vec = []


######## Calculate hd_psi and v_psi #########################################
for i in range(spinorb_vec.shape[0]):
    hd_psi = (orb.apply_dirac_hamiltonian(spinorb_vec[i], prec, 0.0))
    v_psi = (orb.apply_potential(-1.0, V_tree, spinorb_vec[i], prec))
    hd_psi.crop(prec)
    v_psi.crop(prec)
    hd_psi_vec.append(hd_psi)
    v_psi_vec.append(v_psi)
hd_psi_vec = np.array(hd_psi_vec)
v_psi_vec = np.array(v_psi_vec)


############ Calculate J_psi and K_psi #############################
J_psi_vec = J_n(spinorb_vec)
K_psi_vec = K_n(spinorb_vec)


################ Calculation of hd_psi + v_psi + J_psi - K_psi  #####################
add_psi_vec = hd_psi_vec + v_psi_vec + J_psi_vec - K_psi_vec


##################### Calculating of Fij with i /= j  and Lamda matrix ##########################
Fij_matrix = calc_overlap(spinorb_vec, add_psi_vec)
Lambda_n = np.diag(np.diag(Fij_matrix))


################# Calculation of Orbital Energies ############################################
orb_energy_vec = np.diag(Fij_matrix)
print('Orb_Energy', orb_energy_vec)


################ Preparation to calculate the Total Energy ###############################
V_mat = calc_overlap(spinorb_vec, vPhi_vec)
J_mat = calc_overlap(spinorb_vec, JPhi_vec)
K_mat = calc_overlap(spinorb_vec, KPhi_vec)


################# Calculation of Total Energy ###################################
E_orb  =  Fij_matrix.trace()
E_en   =  V_mat.trace()
E_coul =  0.5*J_mat.trace()
E_ex   =  -0.5*K_mat.trace()
E_tot  = E_orb - E_coul - E_ex
print("E_total", E_tot)  


########################################################END#####################################################################
