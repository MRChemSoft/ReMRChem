import numpy as np
import copy as cp
from scipy.special import gamma
from vampyr import vampyr3d as vp
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb

class HelmholtzOperator():
    """
    Vectorized Helmholtz operator
    Parameters
    ----------
    mra : The multiresolution analysis we work on
    lamb : vector of lambda parameters, mu_i = sqrt(-2*lambda_i)
    prec : Precision requirement
    Attributes
    ----------
    operators : list containing HelmholtzOperators for each orbital
    """
    def __init__(self, mra, lamb, prec):
        self.mra = mra
        self.lamb = lamb
        self.prec = prec
        self.operators = []
        self.setup()

    def setup(self):
        mu = [np.sqrt(-2.0*l) if l < 0 else 1.0 for l in self.lamb]
        for m in mu:
            self.operators.append(vp.HelmholtzOperator(mra=self.mra, exp=m, prec=self.prec))

    def __call__(self, Psi):
        """Operate the Helmholtz operator onto an orbital vector"""
        return np.array([self.operators[i](Psi[i]) for i in range(len(Psi))])

class ExchangeOperator():
    """
    Vectorized Exchange operator
    K(phi) = \sum_i P[phi * psi_i] * psi_i
    Parameters
    ----------
    mra : The multiresolution analysis we work on
    Psi : Orbital vector defining the operator
    prec : Precision requirement
    Attributes
    ----------
    poisson : Poisson Operator
    """
    def __init__(self, mra, Psi, prec):
        self.mra = mra
        self.Psi = Psi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)

    def __call__(self, Phi):
        """Apply the exchange operator onto an orbital vector Phi"""

        Phi_out = []
        for j in range(len(Phi)):
            V_j0 = self.poisson(Phi[j].exchange(self.Psi[0],self.prec))
            tmp = (4.0*np.pi)*(V_j0 * self.Psi[0])
            for i in range(1, len(self.Psi)):
                V_ji = self.poisson(Phi[j].exchange(self.Psi[i],self.prec))
                tmp += (4.0*np.pi)*(V_ji * self.Psi[i])
            Phi_out.append(tmp)
        return np.array([phi for phi in Phi_out])


class CouloumbOperator():
    """
    Vectorized Couloumb operator
    J(phi) = \sum_i P[psi_i * psi_i] * phi
    Parameters
    ----------
    mra : The multiresolution analysis we work on
    Psi : Orbital vector defining the operator
    prec : Precision requirement
    Attributes
    ----------
    poisson : Poisson operator
    potential : Coulomb potential
    """
    def __init__(self, mra, Psi, prec):
        self.mra = mra
        self.Psi = Psi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)
        self.potential = None
        self.setup()

    def setup(self):
        rho = self.Psi[0].density(self.prec)
        for i in range(1, len(self.Psi)):
            rho += self.Psi[i].density(self.prec)
        rho.crop(self.prec)
        self.potential = (4.0*np.pi)*self.poisson(rho)

    def __call__(self, Phi):
        """Apply Couloumb operator onto an orbital vector Phi"""
        return np.array([(self.potential*phi) for phi in Phi])



class NuclearOperator():
    """
    Vectorized Nuclear potential operator
    Parameters
    ----------
    mra : The multiresolution analysis we work on
    atoms : List of dicts containing charge and coordinates of the atoms
    prec : Precision requirement
    Attributes
    ----------
    potential : Nuclear potential
    """
    def __init__(self, mra, atoms, prec):
        self.mra = mra
        self.prec= prec
        self.atoms = atoms
        self.potential = None
        self.setup()

    def setup(self):
        # Project analytic function onto MRA basis
        P_mra = vp.ScalingProjector(mra=self.mra, prec=self.prec)
        f_nuc = NuclearFunction(self.atoms)
        self.potential = P_mra(f_nuc)

    def __call__(self, Phi):
        """Apply nuclear potential operator onto an orbital vector"""
        return np.array([(self.potential*phi).crop(self.prec) for phi in Phi])

class NuclearFunction():
    """
    Vectorized Nuclear potential operator
    Parameters
    ----------
    atoms : List of dicts containing charge and coordinates of the atoms
    """


    def __init__(self, atoms):
        self.atoms = atoms

    def __call__(self, r):
        "Returns the nuclear potential value in R"
        tmp = 0.0
        for atom in self.atoms:
            Z = atom["Z"]
            R = atom["R"]
            R2 = (R[0]-r[0])**2 + (R[1]-r[1])**2 + (R[2]-r[2])**2
            tmp += -Z / np.sqrt(R2)
        return tmp