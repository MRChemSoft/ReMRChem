import numpy as np
import numpy.linalg as LA
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
        if component == 'La':
            phi.copy_components(La=self.complexfc)
        elif component == 'Lb':
            phi.copy_components(Lb=self.complexfc)
        else:
            'Invalid component'
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
        return self.potential * Phi


class CoulombExchangeOperator():
    def __init__(self, mra, prec, Psi):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.poisson = vp.PoissonOperator(mra=mra, prec=self.prec)
        self.potential = None


    def __call__(self, Phi):
        V_j0 = self.poisson(self.Psi[0].exchange(Phi, self.prec))
        self.potential = (V_j0 * self.Psi[0])
        for i in range(1, len(self.Psi)):
            V_ji = self.poisson(self.Psi[i].exchange(Phi, self.prec))
            self.potential += (V_ji * self.Psi[i])
        self.potential *= 4.0*np.pi
        return self.potential


class FockMatrix1():
    def __init__(self, prec, default_der, J, K, v_spinorbv, Psi):
        self.prec = prec
        self.default_der = default_der
        self.v_spinorbv = v_spinorbv
        self.J = J 
        self.K = K
        self.Psi = Psi
        self.energy11 = None
        self.energy12 = None
        self.energy21 = None
        self.energy22 = None
        self.energytot = None
        self.energy = None
        self.setup()

    def setup(self):
        #Definiton of Dirac Hamiltonian for spin orbit 1 and 2
        hd_psi_1 = orb.apply_dirac_hamiltonian(self.Psi[0], self.prec, 0.0, der = self.default_der)
        hd_psi_2 = orb.apply_dirac_hamiltonian(self.Psi[1], self.prec, 0.0, der = self.default_der)


        # Definition of full 4c hamitoninan
        add_psi_1 = hd_psi_1 + self.v_spinorbv[0]
        add_psi_2 = hd_psi_2 + self.v_spinorbv[1]


        energy_11, imag_11 = self.Psi[0].dot(add_psi_1)
        energy_12, imag_12 = self.Psi[0].dot(add_psi_2)
        energy_21, imag_21 = self.Psi[1].dot(add_psi_1)
        energy_22, imag_22 = self.Psi[1].dot(add_psi_2)


        E_H11,  imag_H1 = self.Psi[0].dot(self.J(self.Psi[0]))
        E_H12,  imag_H1 = self.Psi[0].dot(self.J(self.Psi[1]))
        E_H21,  imag_H1 = self.Psi[1].dot(self.J(self.Psi[0]))
        E_H22,  imag_H2 = self.Psi[1].dot(self.J(self.Psi[1]))


        E_xc11, imag_xc11 = self.Psi[0].dot(self.K(self.Psi[0]))
        E_xc12, imag_xc12 = self.Psi[0].dot(self.K(self.Psi[1]))
        E_xc21, imag_xc21 = self.Psi[1].dot(self.K(self.Psi[0]))
        E_xc22, imag_xc22 = self.Psi[1].dot(self.K(self.Psi[1]))


        self.energy11 = energy_11 + E_H11 - E_xc11
        self.energy12 = energy_12 + E_H12 - E_xc12
        self.energy21 = energy_21 + E_H21 - E_xc21
        self.energy22 = energy_22 + E_H22 - E_xc22

        self.energytot = self.energy11 + self.energy22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22)

        
    def __call__(self, label):
        if label == 'orb1':
            self.energy = self.energy11
        elif label == 'orb2':
            self.energy = self.energy22
        elif label == 'F_12':
            self.energy = self.energy12
        elif label == 'F_21':
            self.energy = self.energy21
        elif label == 'tot':
            self.energy = self.energytot
        else:
            'Invalid component'
        return self.energy


class Orthogonalize():
    def __init__(self, prec, Psi, Phi):
        self.prec = prec
        self.Psi = Psi
        self.Phi = Phi
        self.Psio = None
        self.Phio = None
        self.xi = None
        self.setup()

    def setup(self):
        dot_11 = self.Psi.dot(self.Psi)
        dot_12 = self.Psi.dot(self.Phi)
        dot_21 = self.Phi.dot(self.Psi)
        dot_22 = self.Phi.dot(self.Phi)

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
        
        self.Psio = Sm5[0, 0] * self.Psi + Sm5[0, 1] * self.Phi
        self.Phio = Sm5[1, 0] * self.Psi + Sm5[1, 1] * self.Phi
        #self.Psio.crop(self.prec)
        #self.Phio.crop(self.prec)    

    def __call__(self, label):
        if label == 'spinorb1':
            self.xi = self.Psio
        elif label == 'spinorb2':
            self.xi = self.Phio
        else:
            'Invalid component'
        return self.xi


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
