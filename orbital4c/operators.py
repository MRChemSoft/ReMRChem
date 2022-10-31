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
    def __init__(self, mra, prec, Psi, cPsi, alpha):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.cPsi = cPsi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.alpha = alpha
        self.GJ = None
        self.potential = None
        self.setup()

    def setup(self):       
        cPsi_alpha  = self.cPsi[0].overlap_density(self.alpha[0], self.prec)
        for i in range(1, len(self.cPsi)):
            cPsi_alpha +=  self.cPsi[i].overlap_density(self.alpha[i], self.prec)
        cPsi_alpha = cPsi_alpha
 

        self.GJ = cf.complex_fcn()
        GJ_Re = self.poisson(cPsi_alpha.real) * (2.0 * np.pi)
        GJ_Re.crop(self.prec)

        GJ_Im = self.poisson(cPsi_alpha.imag) * (2.0 * np.pi)
        GJ_Im.crop(self.prec)                      


    def __call__(self, alpha):
        self.potential = orb.apply_complex_potential(1.0, self.GJ, alpha, self.prec)
        return self.potential


class GauntExchangeOperator():
    def __init__(self, mra, prec, Psi, cPsi, alpha):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.cPsi = cPsi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.alpha = alpha
        self.GK = None
        self.potential = None
        
        

    def __call__(self, label):
        if label == 'spinorb1':
            
            cPsi_alpha  = self.cPsi[0].overlap_density(self.alpha[0], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha +=  self.cPsi[i].overlap_density(self.alpha[0], self.prec)
            cPsi_alpha = cPsi_alpha


        elif label == 'spinorb2':

            cPsi_alpha  = self.cPsi[0].overlap_density(self.alpha[1], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha +=  self.cPsi[i].overlap_density(self.alpha[1], self.prec)
            cPsi_alpha = cPsi_alpha


        else:


            'Invalid component'

        self.GK = cf.complex_fcn()
        GK_Re = self.poisson(cPsi_alpha.real) * (2.0 * np.pi)
        GK_Re.crop(self.prec)


        GK_Im = self.poisson(cPsi_alpha.imag) * (2.0 * np.pi)
        GK_Im.crop(self.prec)
                        


        if label == 'spinorb1':
            self.potential = orb.apply_complex_potential(1.0, self.GK, self.alpha[0], self.prec)
    
        elif label == 'spinorb2':
            self.potential = orb.apply_complex_potential(1.0, self.GK, self.alpha[1], self.prec)

        return self.potential


class FockMatrix2():
    def __init__(self, prec, default_der, J, K, GJ0, GJ1, GJ2, GK0, GK1, GK2, v_spinorbv, Psi, alpha0, alpha1, alpha2):
        self.prec = prec
        self.default_der = default_der
        self.v_spinorbv = v_spinorbv
        self.J = J 
        self.K = K
        self.GJ0 = GJ0
        self.GJ1 = GJ1
        self.GJ2 = GJ2
        self.GK0 = GK0
        self.GK1 = GK1
        self.GK2 = GK2
        self.Psi = Psi
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
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


        GJ0_1 = orb.apply_complex_potential(1.0, self.GJ0, self.alpha0[0], self.prec) 
        GJ1_1 = orb.apply_complex_potential(1.0, self.GJ1, self.alpha1[0], self.prec) 
        GJ2_1 = orb.apply_complex_potential(1.0, self.GJ2, self.alpha2[0], self.prec)


        GJ0_2 = orb.apply_complex_potential(1.0, self.GJ0, self.alpha0[1], self.prec)
        GJ1_2 = orb.apply_complex_potential(1.0, self.GJ1, self.alpha1[1], self.prec)
        GJ2_2 = orb.apply_complex_potential(1.0, self.GJ2, self.alpha2[1], self.prec)



        E_GH0_11,  imag_GH0_11 = self.Psi[0].dot(GJ0_1)
        E_GH1_11,  imag_GH1_11 = self.Psi[0].dot(GJ1_1)
        E_GH2_11,  imag_GH2_11 = self.Psi[0].dot(GJ2_1)

        E_GH11 = E_GH0_11 + E_GH1_11 + E_GH2_11
        
        E_GH0_12,  imag_GH0_12 = self.Psi[0].dot(GJ0_2)
        E_GH1_12,  imag_GH1_12 = self.Psi[0].dot(GJ1_2)
        E_GH2_12,  imag_GH2_12 = self.Psi[0].dot(GJ2_2)

        E_GH12 = E_GH0_12 + E_GH1_12 + E_GH2_12

        E_GH0_21,  imag_GH0_21 = self.Psi[1].dot(GJ0_1)
        E_GH1_21,  imag_GH1_21 = self.Psi[1].dot(GJ1_1)
        E_GH2_21,  imag_GH2_21 = self.Psi[1].dot(GJ2_1)

        E_GH21 = E_GH0_21 + E_GH1_21 + E_GH2_21

        E_GH0_22,  imag_GH0_22 = self.Psi[1].dot(GJ0_2)
        E_GH1_22,  imag_GH1_22 = self.Psi[1].dot(GJ1_2)
        E_GH2_22,  imag_GH2_22 = self.Psi[1].dot(GJ2_2)
        
        E_GH22 = E_GH0_22 + E_GH1_22 + E_GH2_22


        ###############################################


        E_GK0_11,  imag_GK0_11 = self.Psi[0].dot(GJ10)
        E_GK1_11,  imag_GK1_11 = self.Psi[0].dot(GJ11)
        E_GK2_11,  imag_GK2_11 = self.Psi[0].dot(GJ12)

        E_GK11 = E_GK0_11 + E_GK1_11 + E_GK2_11
        
        E_GK0_12,  imag_GK0_12 = self.Psi[0].dot(GJ20)
        E_GK1_12,  imag_GK1_12 = self.Psi[0].dot(GJ21)
        E_GK2_12,  imag_GK2_12 = self.Psi[0].dot(GJ22)

        E_GK12 = E_GK0_12 + E_GK1_12 + E_GK2_12

        E_GK0_21,  imag_GK0_21 = self.Psi[1].dot(GJ10)
        E_GK1_21,  imag_GK1_21 = self.Psi[1].dot(GJ11)
        E_GK2_21,  imag_GK2_21 = self.Psi[1].dot(GJ12)

        E_GK21 = E_GK0_21 + E_GK1_21 + E_GK2_21

        E_GK0_22,  imag_GK0_22 = self.Psi[1].dot(GJ20)
        E_GK1_22,  imag_GK1_22 = self.Psi[1].dot(GJ21)
        E_GK2_22,  imag_GK2_22 = self.Psi[1].dot(GJ22)

        E_GK22 = E_GK0_22 + E_GK1_22 + E_GK2_22
     

        self.energy11 = energy_11 + E_H11 - E_xc11 - E_GH11 + E_GK11 
        self.energy12 = energy_12 + E_H12 - E_xc12 - E_GH12 + E_GK12
        self.energy21 = energy_21 + E_H21 - E_xc21 - E_GH21 + E_GK21
        self.energy22 = energy_22 + E_H22 - E_xc22 - E_GH22 + E_GK22

        self.energytot = self.energy11 + self.energy22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_GH11 - E_GH22 + E_GK11 + E_GK22)

        
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

