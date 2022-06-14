import numpy as np
import numpy.linalg as LA
from  vampyr import vampyr3d as vp
from Operators import NuclearOperator, CouloumbOperator, ExchangeOperator, HelmholtzOperator

def scf_solver(mra, atoms, Phi_n, F_n, precision, threshold, max_iter=30):
    """Kinetric-free Hartree-Fock SCF solver
    Parameters:
    mra : The Multiresolution analysis to work on
    atoms : List of dicts containing charge and coordinates of the atoms
    Phi_n : Starting guess orbitals
    F_n : Starting guess for Fock matrix
    precision : Precision requirement
    threshold : Usually set to the same as precision, set to -1 to limit iterations by max_iter
    max_iter : Set maximum iterations
    Returns:
    Updates : Vector of orbital residual norms at each iteration
    Energies : List of energy contributions at each iteration
    Phi_n : Converged orbital vector
    """

    # Setup nuclear potential
    V_nuc = NuclearOperator(mra, atoms, precision)

    # Loop parameters
    iteration = 0                # Iteration counter
    update = np.ones(len(Phi_n)) # Initialize error measure (norm of orbital updates)
    updates = []                 # Will capture wavefunction updates for visusualization
    energies = []                # Will capture energies for visualization

    # SCF loop
    while (max(update) > threshold):
        if iteration > max_iter-1:
            break

        # Initialize operators for first iteration
        J_n = CouloumbOperator(mra, Phi_n, precision)
        K_n = ExchangeOperator(mra, Phi_n, precision)

        # Initialize vector of Helmholtz operators based on Fock matrix diagonal
        Lambda_n = np.diag(np.diag(F_n))
        G = HelmholtzOperator(mra, np.diag(Lambda_n), precision)

        # Apply potential operator to all orbitals
        VPhi = V_nuc(Phi_n) + 2*J_n(Phi_n) - K_n(Phi_n)

        # Apply Helmholtz operators to all orbitals
        Phi_np1 = -2*G(VPhi + (Lambda_n - F_n) @ Phi_n)
        dPhi_n = Phi_np1 - Phi_n
        update = np.array([phi.norm() for phi in dPhi_n])

        # Compute overlap matrices
        S_tilde = calc_overlap(Phi_np1, Phi_np1)
        dS_1 = calc_overlap(dPhi_n, Phi_n)
        dS_2 = calc_overlap(Phi_np1, dPhi_n)

        # Löwdin orthonormalization S^{-1/2} = U * Sigma^{-1/2} * U^T
        sigma, U = LA.eig(S_tilde)
        Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
        Phi_bar = Sm5 @ Phi_np1

        # Initialize n+1 operators
        J_np1 = CouloumbOperator(mra, Phi_bar, precision)
        K_np1 = ExchangeOperator(mra, Phi_bar, precision)

        # Compute Fock matrix updates
        V_dPhi = V_nuc(dPhi_n) + 2*J_n(dPhi_n) - K_n(dPhi_n)
        dJ_Phi = J_np1(Phi_np1) - J_n(Phi_np1)
        dK_Phi = K_np1(Phi_np1) - K_n(Phi_np1)
        dF_pot = calc_overlap(Phi_np1, V_dPhi)
        dF_pot += 2*calc_overlap(Phi_np1, dJ_Phi)
        dF_pot -= calc_overlap(Phi_np1, dK_Phi)

        # Update Fock matrix, symmetrize to average out numerical errors
        dF_n = (dS_1 @ F_n) + (dS_2 @ Lambda_n) + dF_pot
        F_tilde = F_n + 0.5*(dF_n + dF_n.transpose())

        # Prepare for next iteration
        F_n = Sm5.transpose() @ F_tilde @ Sm5
        Phi_n = Phi_bar

        # Compute energy contributions
        energy = calc_energies(F_n, Phi_n, V_nuc, J_np1, K_np1)

        # Collect output
        updates.append(update)
        energies.append(energy)
        print(iteration, " |  E_tot:", energy["$E_{tot}$"], " |  dPhi:", max(update))
        iteration += 1


    return np.array(updates), energies, Phi_n


def calc_energies(F_mat, Phi, V, J, K):
    """"Calcuate all energy contributions"""

    V_mat = calc_overlap(Phi, V(Phi))
    J_mat = calc_overlap(Phi, J(Phi))
    K_mat = calc_overlap(Phi, K(Phi))

    E_orb  = 2.0*F_mat.trace()
    E_en   = 2.0*V_mat.trace()
    E_coul = 2.0*J_mat.trace()
    E_ex   = -K_mat.trace()
    E_tot  = E_orb - E_coul - E_ex
    E_kin  = E_tot - E_en - E_coul - E_ex

    return {
        "$E_{orb}$": E_orb,
        "$E_{en}$": E_en,
        "$E_{coul}$": E_coul,
        "$E_{ex}$": E_ex,
        "$E_{kin}$": E_kin,
        "$E_{tot}$": E_tot
    }


def calc_overlap(Bra, Ket):
    """Calculate the overlap matrix between the orbitals <Bra| and |Ket>
    Parameters:
    Bra : bra vector <Phi|
    Ket : ket vector |Phi>
    Returns:
    Overlap matrix
    """

    S = np.empty((len(Bra), len(Ket)))
    for i in range(len(Bra)):
        for j in range(len(Ket)):
            S[i, j] = vp.dot(Bra[i], Ket[j])
    return S

def starting_guess(mra, atom, n_orbs, prec):
    """Primitive starting guess, works for Be"""

    # Define projector onto the MRA basis
    P_mra = vp.ScalingProjector(mra=mra, prec=prec)

    Phi = []
    for i in range(1, n_orbs+1):
        R0 = atom["R"]
        def f_gauss(r):
            R2 = (r[0]-R0[0])**2 + (r[1]-R0[1])**2 + (r[2]-R0[2])**2
            return np.exp(-R2/i)

        phi = P_mra(f_gauss)
        phi.normalize()
        Phi.append(phi)
    Phi = np.array(Phi)

    # Löwdin orthonormalization S^{-1/2} = U * Sigma^{-1/2} * U^T
    S = calc_overlap(Phi, Phi)
    sigma, U = LA.eig(S)
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
    return Sm5 @ Phi