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


class CoulombDirectOperator():
    def __init__(self, mra, prec, Psi):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None
        self.setup()

    def setup(self):
        rho = self.Psi[0].density(self.prec)
        for i in range(1, len(self.Psi)):
            rho += self.Psi[i].density(self.prec)
        rho.crop(self.prec)
        self.potential = (4.0*np.pi)*self.poisson(rho).crop(self.prec)
        
    def __call__(self, Phi):
        return np.array([(self.potential*phi).crop(self.prec) for phi in Phi])


class CoulombExchangeOperator():

    def __init__(self, mra, prec, Psi):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
    
    def __call__(self, Phi):
        
        Phi_out = []
        for j in range(len(Phi)):
            V_j0 = self.poisson(Phi[j] * self.Psi[0])
            tmp = (self.Psi[0] * V_j0).crop(self.prec)
            for i in range(1, len(self.Psi)):
                V_ji = self.poisson(Phi[j] * self.Psi[i])
                tmp += (self.Psi[i] * V_ji).crop(self.prec)
            tmp *= 4.0*np.pi
            Phi_out.append(tmp)
        return np.array([phi.crop(self.prec) for phi in Phi_out])


class GauntDirectOperator():
    def __init__(self, mra, prec):
        self.mra = mra
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None
        
    def __call__(self, alpha, cPhi):
        cPhi_alpha  = cPhi.overlap_density(alpha, prec)
     
        GJ_Re = self.poisson(cPhi_alpha.real) * (2.0 * np.pi)

        GJ_Im = self.poisson(cPhi_alpha.imag) * (2.0 * np.pi)

        GJ = cf.complex_fcn()
        GJ.real = GJ_Re0
        GJ.imag = GJ_Im0

        self.potential = orb.apply_complex_potential(1.0, GJ, alpha, prec)

        return self.potential



class GauntExchangeOperator():
    def __init__(self, mra, prec):
        self.mra = mra
        self.prec = prec
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None
        
    def __call__(self, alpha1, alpha2, cPhi1):
        cPhi1_alpha2 = cPhi1.overlap_density(alpha2, prec)

        GK12_Re0 = self.poisson(cPhi1_alpha2.real) * (2.0 * np.pi)  
        
        GK12_Im0 = self.poisson(cPhi1_alpha2.imag) * (2.0 * np.pi)

        GK12 = cf.complex_fcn()
        GK12.real = GK12_Re0
        GK12.imag = GK12_Im0  

        self.potential = orb.apply_complex_potential(1.0, GK12, alpha1, prec)    

        return self.potential
