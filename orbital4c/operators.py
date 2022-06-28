import numpy as np
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from vampyr import vampyr3d as vp

class ExchangeOperator():

    def __init__(self, mra, Psi, prec):
        self.mra = mra
        self.Psi = Psi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)

    def __call__(self, Phi):
        Phi_out = []
        for j in range(len(Phi)):
            V_j0 = self.poisson(Phi[j].exchange(self.Psi[0],self.prec))
            tmp = (self.Psi[0]  * V_j0)
            for i in range(1, len(self.Psi)):
                V_ji = self.poisson(Phi[j].exchange(self.Psi[i],self.prec))
                tmp += (self.Psi[i]  * V_ji)
            tmp *= (4.0*np.pi)
            Phi_out.append(tmp)
        return np.array([phi for phi in Phi_out])


class CouloumbOperator():
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
        self.potential = (4.0*np.pi)*self.poisson(rho).crop(self.prec)

    def __call__(self, Phi):
        return np.array([(self.potential*phi) for phi in Phi])


class GauntExchange():
    def __init__(self, mra, Psi, prec):
        self.mra = mra
        self.Psi = Psi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)

    def __call__(self, Phi):
        Phi_out = []
        for j in range(len(Phi)):
            gamma = self.Psi[0].alpha(self.prec)
            V_j0 = self.poisson(Phi[j].exchange(gamma[0],self.prec))
            tmp = (self.Psi[0] * V_j0)
            for i in range(1, len(self.Psi)):
                V_ji = self.poisson(Phi[j].exchange(gamma[i],self.prec))
                tmp += (gamma[i] * V_ji)
            tmp *= (4.0*np.pi)
            Phi_out.append(tmp)
        return np.array([phi for phi in Phi_out]) 
 


class GauntDirect():
    def __init__(self, mra, Psi, prec):
        self.mra = mra
        self.Psi = Psi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)
        self.potential = None
        self.setup()

    def setup(self):
        gamma = self.Psi[0].alpha(self.prec)
        rho = self.Psi[0].exchange(gamma[0],self.prec)
        for i in range(1, len(self.Psi)):
            rho += self.Psi[i].exchange(gamma[0],self.prec)
        self.potential = (4.0*np.pi)*self.poisson(rho)

    def __call__(self, Phi):
        return np.array([(self.potential*phi) for phi in Phi])

