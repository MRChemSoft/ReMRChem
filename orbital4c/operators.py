import numpy as np
from orbital4c import complex_fcn as cf
from orbital4c import orbital     as orb
from vampyr    import vampyr3d    as vp


class SpinorbGenerator():

    def __init__(self, mra, guessorb, c, origin, prec):
        self.prec   = prec
        self.mra = mra 
        self.guessorb = guessorb
        self.c = c
        self.origin = origin
        self.complexfc = None

        if self.guessorb == 'slater':
            print('cazzo')
        elif guessorb == 'gaussian':
################################   DEFINE GAUSSIAN FUNCTION AS GUESS  ################################
            a_coeff = 3.0
            b_coeff = np.sqrt(a_coeff/np.pi)**3
            gauss = vp.GaussFunc(b_coeff, a_coeff, self.origin)
            gauss_tree = vp.FunctionTree(self.mra)
            vp.advanced.build_grid(out=gauss_tree, inp=gauss)
            vp.advanced.project(prec=self.prec, out=gauss_tree, inp=gauss)
            gauss_tree.normalize()
#################################### DEFINE ORBITALS (C FUNCTION) ####################################
            orb.orbital4c.mra = self.mra
            orb.orbital4c.light_speed = self.c
            cf.complex_fcn.mra = self.mra
            self.complexfc = cf.complex_fcn()
            self.complexfc.copy_fcns(real=gauss_tree)


    def __call__(self, component):
        phi = orb.orbital4c()
        if component == "La":
            phi.copy_components(La=self.complexfc)
        elif component == "Lb":
            phi.copy_components(Lb=self.complexfc)
        else:
            "Invalid component"
        phi.init_small_components(self.prec/10)
        phi.normalize()
        return phi

class CouloumbOperator():
    def __init__(self, mra, Psi, Phi, prec):
        self.mra = mra
        self.Psi = Psi
        self.Phi = Phi
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None
        self.setup()

    def setup(self):
        rho1 = self.Psi.density(self.prec)
        rho1.crop(self.prec)
        rho2 = self.Phi.density(self.prec)
        rho2.crop(self.prec)
        rho = rho1 + rho2 
        self.potential = (4.0*np.pi)*self.poisson(rho).crop(self.prec)
        
    def __call__(self, so):
        return self.potential*so

class ExchangeOperator():

    def __init__(self, mra, prec):
        self.mra = mra
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)

    def __call__(self, Phi, Psi):
        V_ij = self.poisson(Phi.exchange(Psi,self.prec))
        V_ij *= (4.0*np.pi)
        tmp = V_ij * Psi 
        return tmp

#class GauntExchange():
#    def __init__(self, mra, Psi, prec):
#        self.mra = mra
#        self.Psi = Psi
#        self.prec = prec
#        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)
#
#    def __call__(self, Phi):
#        Phi_out = []
#        for j in range(len(Phi)):
#            gamma = self.Psi[0].alpha(self.prec)
#            V_j0 = self.poisson(Phi[j].exchange(gamma[0],self.prec))
#            tmp = (self.Psi[0] * V_j0)
#            for i in range(1, len(self.Psi)):
#                V_ji = self.poisson(Phi[j].exchange(gamma[i],self.prec))
#                tmp += (gamma[i] * V_ji)
#            tmp *= (4.0*np.pi)
#            Phi_out.append(tmp)
#        return np.array([phi for phi in Phi_out]) 
# 
#
#
#class GauntDirect():
#    def __init__(self, mra, Psi, prec):
#        self.mra = mra
#        self.Psi = Psi
#        self.prec = prec
#        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)
#        self.potential = None
#        self.setup()
#
#    def setup(self):
#        gamma = self.Psi[0].alpha(self.prec)
#        rho = self.Psi[0].exchange(gamma[0],self.prec)
#        for i in range(1, len(self.Psi)):
#            rho += self.Psi[i].exchange(gamma[0],self.prec)
#        self.potential = (4.0*np.pi)*self.poisson(rho)
#
#    def __call__(self, Phi):
#        return np.array([(self.potential*phi) for phi in Phi])

