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
            tmp = (4.0*np.pi)*(V_j0 * self.Psi[0])
            for i in range(1, len(self.Psi)):
                V_ji = self.poisson(Phi[j].exchange(self.Psi[i],self.prec))
                tmp += (4.0*np.pi)*(V_ji * self.Psi[i])
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
        self.potential = (4.0*np.pi)*self.poisson(rho)

    def __call__(self, Phi):
        return np.array([(self.potential*phi) for phi in Phi])

