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
    def __init__(self, mra, prec, Psi, cPsi, alpha0, alpha1, alpha2):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.cPsi = cPsi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.GJ = None
        self.potential = None
        self.setup()

    def setup(self):       
        cPsi_alpha0  = self.cPsi[0].overlap_density(self.alpha0[0], self.prec)
        for i in range(1, len(self.cPsi)):
            cPsi_alpha0 +=  self.cPsi[i].overlap_density(self.alpha0[i], self.prec)
        cPsi_alpha0 = cPsi_alpha0


        cPsi_alpha1  = self.cPsi[0].overlap_density(self.alpha1[0], self.prec)
        for i in range(1, len(self.cPsi)):
            cPsi_alpha1 +=  self.cPsi[i].overlap_density(self.alpha1[i], self.prec)
        cPsi_alpha1 = cPsi_alpha1


        cPsi_alpha2  = self.cPsi[0].overlap_density(self.alpha2[0], self.prec)
        for i in range(1, len(self.cPsi)):
            cPsi_alpha2 +=  self.cPsi[i].overlap_density(self.alpha2[i], self.prec)
        cPsi_alpha2 = cPsi_alpha2 
        

        GJ0_Re = self.poisson(cPsi_alpha0.real) * (2.0 * np.pi)
        GJ0_Re.crop(self.prec)

        GJ1_Re = self.poisson(cPsi_alpha1.real) * (2.0 * np.pi)
        GJ1_Re.crop(self.prec)

        GJ2_Re = self.poisson(cPsi_alpha2.real) * (2.0 * np.pi)
        GJ2_Re.crop(self.prec)                


        GJ0_Im = self.poisson(cPsi_alpha0.imag) * (2.0 * np.pi)
        GJ0_Im.crop(self.prec)

        GJ1_Im = self.poisson(cPsi_alpha1.imag) * (2.0 * np.pi)
        GJ1_Im.crop(self.prec)

        GJ2_Im = self.poisson(cPsi_alpha2.imag) * (2.0 * np.pi)
        GJ2_Im.crop(self.prec)                       

        
        self.GJ = cf.complex_fcn()

        self.GJ.real = GJ0_Re + GJ1_Re + GJ2_Re                               
        self.GJ.imag = GJ0_Im + GJ1_Im + GJ2_Im


    def __call__(self, label):
        if label == 'spinorb1':
            a = orb.apply_complex_potential(1.0, self.GJ, self.alpha0[0], prec)
            b = orb.apply_complex_potential(1.0, self.GJ, self.alpha1[0], prec)
            c = orb.apply_complex_potential(1.0, self.GJ, self.alpha2[0], prec)
            self.potential = a + b + c 
        elif label == 'spinorb2':
            a = orb.apply_complex_potential(1.0, self.GJ, self.alpha0[1], prec)
            b = orb.apply_complex_potential(1.0, self.GJ, self.alpha1[1], prec)
            c = orb.apply_complex_potential(1.0, self.GJ, self.alpha2[1], prec)
            self.potential = a + b + c 
        else:
            'Invalid component'
        return self.potential


class GauntExchangeOperator():
    def __init__(self, mra, prec, Psi, cPsi, alpha0, alpha1, alpha2):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi
        self.cPsi = cPsi
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.GK = None
        self.potential = None
        
        

    def __call__(self, label):
        if label == 'spinorb1':
            
            cPsi_alpha0  = self.cPsi[0].overlap_density(self.alpha0[0], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha0 +=  self.cPsi[i].overlap_density(self.alpha0[0], self.prec)
            cPsi_alpha0 = cPsi_alpha0


            cPsi_alpha1  = self.cPsi[0].overlap_density(self.alpha1[0], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha1 +=  self.cPsi[i].overlap_density(self.alpha1[0], self.prec)
            cPsi_alpha1 = cPsi_alpha1


            cPsi_alpha2  = self.cPsi[0].overlap_density(self.alpha2[0], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha2 +=  self.cPsi[i].overlap_density(self.alpha2[0], self.prec)
            cPsi_alpha2 = cPsi_alpha2


        elif label == 'spinorb2':

            cPsi_alpha0  = self.cPsi[0].overlap_density(self.alpha0[1], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha0 +=  self.cPsi[i].overlap_density(self.alpha0[1], self.prec)
            cPsi_alpha0 = cPsi_alpha0


            cPsi_alpha1  = self.cPsi[0].overlap_density(self.alpha1[1], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha1 +=  self.cPsi[i].overlap_density(self.alpha1[1], self.prec)
            cPsi_alpha1 = cPsi_alpha1


            cPsi_alpha2  = self.cPsi[0].overlap_density(self.alpha2[1], self.prec)
            for i in range(1, len(self.cPsi)):
                cPsi_alpha2 +=  self.cPsi[i].overlap_density(self.alpha2[1], self.prec)
            cPsi_alpha2 = cPsi_alpha2


        else:


            'Invalid component'


        GK0_Re = self.poisson(cPsi_alpha0.real) * (2.0 * np.pi)
        GK0_Re.crop(self.prec)

        GK1_Re = self.poisson(cPsi_alpha1.real) * (2.0 * np.pi)
        GK1_Re.crop(self.prec)

        GK2_Re = self.poisson(cPsi_alpha2.real) * (2.0 * np.pi)
        GK2_Re.crop(self.prec)                


        GK0_Im = self.poisson(cPsi_alpha0.imag) * (2.0 * np.pi)
        GK0_Im.crop(self.prec)

        GK1_Im = self.poisson(cPsi_alpha1.imag) * (2.0 * np.pi)
        GK1_Im.crop(self.prec)

        GK2_Im = self.poisson(cPsi_alpha2.imag) * (2.0 * np.pi)
        GK2_Im.crop(self.prec)                       

        
        self.GK = cf.complex_fcn()

        self.GK.real = GK0_Re + GK1_Re + GK2_Re                               
        self.GK.imag = GK0_Im + GK1_Im + GK2_Im    


        if label == 'spinorb1':
            
            a = orb.apply_complex_potential(1.0, self.GJ, self.alpha0[0], self.prec)
            b = orb.apply_complex_potential(1.0, self.GJ, self.alpha1[0], self.prec)
            c = orb.apply_complex_potential(1.0, self.GJ, self.alpha2[0], self.prec)
            self.potential = a + b + c 
        
        elif label == 'spinorb2':
            
            a = orb.apply_complex_potential(1.0, self.GJ, self.alpha0[1], self.prec)
            b = orb.apply_complex_potential(1.0, self.GJ, self.alpha1[1], self.prec)
            c = orb.apply_complex_potential(1.0, self.GJ, self.alpha2[1], self.prec)
            self.potential = a + b + c 

        return self.potential

class FockMatrix2():
    def __init__(self, prec, default_der, J, K, GJ, GK, v_spinorbv, Psi):
        self.prec = prec
        self.default_der = default_der
        self.v_spinorbv = v_spinorbv
        self.J = J 
        self.K = K
        self.GJ = GJ
        self.GK = GK
        self.Psi = Psi
        self.alpha0 = None
        self.alpha1 = None
        self.alpha2 = None
        self.energy11 = None
        self.energy12 = None
        self.energy21 = None
        self.energy22 = None
        self.energytot = None
        self.energy = None
        self.setup()

    def setup(self):
        self.alpha0 = []
        self.alpha0[0] = self.Psi[0].alpha(0) 
        for i in range(1, len(self.Psi)):
            self.alpha0[i] =  self.Psi[i].alpha(0)
        
        self.alpha1 = []
        self.alpha1[0] = self.Psi[0].alpha(1)
        for i in range(1, len(self.Psi)):
            self.alpha1[i] =  self.Psi[i].alpha(1)
        
        self.alpha2 = []
        self.alpha2[0] = self.Psi[0].alpha(2)
        for i in range(1, len(self.Psi)):
            self.alpha2[i] =  self.Psi[i].alpha(2)


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


        E_GH110,  imag_GH110 = self.Psi[0].dot(self.GJ(self.alpha0[0]))
        E_GH111,  imag_GH111 = self.Psi[0].dot(self.GJ(self.alpha1[0]))
        E_GH112,  imag_GH111 = self.Psi[0].dot(self.GJ(self.alpha2[0]))

        E_GH11 = E_GH110 + E_GH111 + E_GH112
        
        E_GH120,  imag_GH120 = self.Psi[0].dot(self.GJ(self.alpha0[1]))
        E_GH121,  imag_GH121 = self.Psi[0].dot(self.GJ(self.alpha1[1]))
        E_GH122,  imag_GH122 = self.Psi[0].dot(self.GJ(self.alpha2[1]))

        E_GH12 = E_GH120 + E_GH121 + E_GH122

        E_GH210,  imag_GH210 = self.Psi[1].dot(self.GJ(self.alpha0[0]))
        E_GH211,  imag_GH211 = self.Psi[1].dot(self.GJ(self.alpha1[0]))
        E_GH212,  imag_GH212 = self.Psi[1].dot(self.GJ(self.alpha2[0]))

        E_GH21 = E_GH210 + E_GH211 + E_GH212

        E_GH220,  imag_GH220 = self.Psi[1].dot(self.GJ(self.alpha0[1]))
        E_GH221,  imag_GH221 = self.Psi[1].dot(self.GJ(self.alpha1[1]))
        E_GH222,  imag_GH222 = self.Psi[1].dot(self.GJ(self.alpha2[1]))

        
        E_GH22 = E_GH220 + E_GH221 + E_GH222


        E_GK110,  imag_GK110 = self.Psi[0].dot(self.GK(self.alpha0[0]))
        E_GK111,  imag_GK111 = self.Psi[0].dot(self.GK(self.alpha1[0]))
        E_GK112,  imag_GK111 = self.Psi[0].dot(self.GK(self.alpha2[0]))

        E_GK11 = E_GK110 + E_GK111 + E_GK112
        
        E_GK120,  imag_GK120 = self.Psi[0].dot(self.GK(self.alpha0[1]))
        E_GK121,  imag_GK121 = self.Psi[0].dot(self.GK(self.alpha1[1]))
        E_GK122,  imag_GK122 = self.Psi[0].dot(self.GK(self.alpha2[1]))

        E_GK12 = E_GK120 + E_GK121 + E_GK122

        E_GK210,  imag_GK210 = self.Psi[1].dot(self.GK(self.alpha0[0]))
        E_GK211,  imag_GK211 = self.Psi[1].dot(self.GK(self.alpha1[0]))
        E_GK212,  imag_GK212 = self.Psi[1].dot(self.GK(self.alpha2[0]))

        E_GK21 = E_GK210 + E_GK211 + E_GK212

        E_GK220,  imag_GK220 = self.Psi[1].dot(self.GK(self.alpha0[1]))
        E_GK221,  imag_GK221 = self.Psi[1].dot(self.GK(self.alpha1[1]))
        E_GK222,  imag_GK222 = self.Psi[1].dot(self.GK(self.alpha2[1]))

        
        E_GK22 = E_GK220 + E_GK221 + E_GK222
     

        self.energy11 = energy_11 + E_H11 - E_xc11 - E_GH11 + E_Gxc11 
        self.energy12 = energy_12 + E_H12 - E_xc12 - E_GH12 + E_Gxc12
        self.energy21 = energy_21 + E_H21 - E_xc21 - E_GH21 + E_Gxc21
        self.energy22 = energy_22 + E_H22 - E_xc22 - E_GH22 + E_Gxc22

        self.energytot = self.energy11 + self.energy22 - 0.5 * (E_H11 + E_H22 - E_xc11 - E_xc22 - E_GH11 - E_GH22 + E_Gxc11 + E_Gxc22)

        
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

