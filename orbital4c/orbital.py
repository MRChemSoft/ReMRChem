from vampyr import vampyr3d as vp
import numpy as np
import copy as cp
from scipy.special import gamma

c = 137

class orbital4c:
    """Four components orbital."""
    def __init__(self, name, mra):
        self.name = name
        self.mra = mra
        self.components = {'Lar': None,
                          'Lai': None,
                          'Lbr': None,
                          'Lbi': None,
                          'Sar': None,
                          'Sai': None,
                          'Sbr': None,
                          'Sbi': None}
        self.initComponents()
        self.setZero()
        
    def __getitem__(self, key):
        return self.components[key]
    
    def __setitem__(self, key, val):
        self.components[key] = val
        
    def print_components(self):
        print("Large components")
        print("alpha real")
        print(self["Lar"])
        print("alpha imaginary")
        print(self["Lai"])
        print("beta real")
        print(self["Lbr"])
        print("beta imaginary")
        print(self["Lbi"])
        print("Small components")
        print("alpha real")
        print(self["Sar"])
        print("alpha imaginary")
        print(self["Sai"])
        print("beta real")
        print(self["Sbr"])
        print("beta imaginary")
        print(self["Sbi"])

    def initComponents(self):
        for key in self.components.keys():
            self.components[key] = vp.FunctionTree(self.mra)
            
    def setZero(self):
        for key in self.components.keys():
            self.components[key].setZero()

    def rescale(self, factor):
        for comp,func in self.components.items():
            if(func.squaredNorm() > 0):
                func *= factor
            
    def init_function(self, function, component):
        vp.advanced.copy_grid(self[component], function)
        vp.advanced.copy_func(self[component], function)
        
    def normalize(self):
        norm_sq, imag1 = scalar_product(self, self)
        norm = np.sqrt(norm_sq)
        self.rescale(1.0/norm)
    
    def init_large_components(self, Lar=None, Lai=None, Lbr=None, Lbi=None):
        nr_of_functions = 0
        if(Lar != None):
            nr_of_functions += 1
            self.init_function(Lar, 'Lar')
        if(Lai != None):
            nr_of_functions += 1
            self.init_function(Lai, 'Lai')
        if(Lbr != None):
            nr_of_functions += 1
            self.init_function(Lbr, 'Lbr')
        if(Lbi != None):
            nr_of_functions += 1
            self.init_function(Lbi, 'Lbi')
        if(nr_of_functions == 0):
            print("WARNING: Large components not initialized!")
        
    def init_small_components(self,prec):
    # initalize the small components based on the kinetic balance
        mra = self.mra
        D = vp.ABGVDerivative(mra, 0.0, 0.0)
        grad_ar = vp.gradient(D, self['Lar'])
        print(grad_ar[0],grad_ar[1],grad_ar[2])
        grad_ai = vp.gradient(D, self['Lai'])
        grad_br = vp.gradient(D, self['Lbr'])
        grad_bi = vp.gradient(D, self['Lbi'])
        sar_tree = vp.FunctionTree(mra)    
        sai_tree = vp.FunctionTree(mra)    
        sbr_tree = vp.FunctionTree(mra)    
        sbi_tree = vp.FunctionTree(mra)    
        sum_ar = []
        sum_ar.append(tuple([ 0.5/c, grad_bi[0]]))
        sum_ar.append(tuple([-0.5/c, grad_br[1]]))
        sum_ar.append(tuple([ 0.5/c, grad_ai[2]]))
        vp.advanced.add(prec/10, sar_tree, sum_ar)
        sum_ai = []
        sum_ai.append(tuple([-0.5/c, grad_br[0]]))
        sum_ai.append(tuple([-0.5/c, grad_bi[1]]))
        sum_ai.append(tuple([-0.5/c, grad_ar[2]]))
        vp.advanced.add(prec/10, sai_tree, sum_ai)
        sum_br = []
        sum_br.append(tuple([ 0.5/c, grad_ai[0]]))
        sum_br.append(tuple([ 0.5/c, grad_ar[1]]))
        sum_br.append(tuple([-0.5/c, grad_bi[2]]))
        vp.advanced.add(prec/10, sbr_tree, sum_br)
        sum_bi = []
        sum_bi.append(tuple([-0.5/c, grad_ar[0]]))
        sum_bi.append(tuple([ 0.5/c, grad_ai[1]]))
        sum_bi.append(tuple([ 0.5/c, grad_br[2]]))
        vp.advanced.add(prec/10, sbi_tree, sum_bi)
        self.init_function(sar_tree, 'Sar')
        self.init_function(sai_tree, 'Sai')
        self.init_function(sbr_tree, 'Sbr')
        self.init_function(sbi_tree, 'Sbi')
        
    def gradient(self):
        D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        orb_grad = {}
        for comp, func in self.components.items():
            orb_grad[comp] = vp.gradient(D, self.components[comp]) 
        return orb_grad
    
    def density(self, prec):
        density = vp.FunctionTree(self.mra)
        add_vector = []
        for comp, func in self.components.items():
            if(func.squaredNorm() > 0):
                temp = vp.FunctionTree(self.mra)
                vp.advanced.multiply(prec, temp, 1.0, func, func)
                add_vector.append((1.0,temp))
        vp.advanced.add(prec/10, density, add_vector)
        return density
    
def grab_sign(comp, derivative):
    grab_table = {
        'Lar': ( 1, -1,  1), 
        'Lai': (-1, -1, -1), 
        'Lbr': ( 1,  1, -1), 
        'Lbi': (-1,  1,  1), 
        'Sar': ( 1, -1,  1), 
        'Sai': (-1, -1, -1), 
        'Sbr': ( 1,  1, -1), 
        'Sbi': (-1,  1,  1), 
    }
    return grab_table[comp][derivative]
    
def grab_coefficient(comp, derivative, global_factor = 1.0):
    grab_table = {
        'Lar': (c**2,c**2,c**2), 
        'Lai': (c**2,c**2,c**2),
        'Lbr': (c**2,c**2,c**2),
        'Lbi': (c**2,c**2,c**2),
        'Sar': (1, 1, 1), 
        'Sai': (1, 1, 1), 
        'Sbr': (1, 1, 1), 
        'Sbi': (1, 1, 1)
    }
    #return grab_table[comp][derivative] * global_factor
    return c
    
def grab_component(comp, derivative):
    grab_table = {
        'Lar': ('Sbi', 'Sbr', 'Sai'), 
        'Lai': ('Sbr', 'Sbi', 'Sar'), 
        'Lbr': ('Sai', 'Sar', 'Sbi'), 
        'Lbi': ('Sar', 'Sai', 'Sbr'), 
        'Sar': ('Lbi', 'Lbr', 'Lai'), 
        'Sai': ('Lbr', 'Lbi', 'Lar'), 
        'Sbr': ('Lai', 'Lar', 'Lbi'), 
        'Sbi': ('Lar', 'Lai', 'Lbr'), 
    }
    return grab_table[comp][derivative]
    
def assemble_vectors(orb, orb_grad, shift = 0.0):
    add_orbitals = {}
    for comp, func in orb.components.items():
        add_orbitals[comp] = []
        if('L' in comp):
            beta_factor = c**2 + shift
        else:
            beta_factor = -c**2 + shift            
        if(func.squaredNorm() > 0):
            add_orbitals[comp].append((beta_factor, func))
        for idx in range(3):
            comp_der = grab_component(comp, idx)
            comp_sign = grab_sign(comp, idx)
            comp_coeff = grab_coefficient(comp, idx)
            if(orb_grad[comp_der][idx].squaredNorm() > 0):
                tmp_tuple = (comp_sign * comp_coeff,orb_grad[comp_der][idx]) 
                add_orbitals[comp].append(tmp_tuple)
    return add_orbitals

def apply_dirac_hamiltonian(orbital, prec, shift = 0.0):
    out_orbital = orbital4c("Hpsi",orbital.mra)
    orb_grad = orbital.gradient()
    add_vectors = assemble_vectors(orbital, orb_grad, shift)
    for comp, func in out_orbital.components.items():
        vp.advanced.add(prec/10, func, add_vectors[comp])
    return out_orbital

def apply_potential(nuclear_potential, orbital, prec):
    out_orbital = orbital4c("Vpsi",orbital.mra)
    for comp, func in orbital.components.items():
        if func.squaredNorm() > 0:
            vp.advanced.multiply(prec, out_orbital[comp], -1.0, nuclear_potential, func)
    return out_orbital

def add_orbitals(a, orb_a, b, orb_b, prec):
    out_orb = orbital4c("a_plus_b",orb_a.mra)
    for comp, func in out_orb.components.items():        
        func_a = orb_a[comp]
        func_b = orb_b[comp]
        if (func_a.squaredNorm() > 0 and func_b.squaredNorm() > 0):
            vp.advanced.add(prec/10, func, a, func_a, b, func_b)
        elif(func_a.squaredNorm() > 0):
            out_orb.init_function(func_a, comp)
            func *= a
        elif(func_b.squaredNorm() > 0):
            out_orb.init_function(func_b, comp)
            func *= b
        else:
            print('Warning: adding two empty trees')
    return out_orb

def scalar_product(orb_a, orb_b):
    out_real = 0
    out_imag = 0
    for comp in ['La','Lb','Sa','Sb']:
        factor = 1
#        if('S' in comp):
#            factor = c**2
        real_comp = comp + 'r'
        imag_comp = comp + 'i'
        ac = 0
        bd = 0
        ad = 0
        bc = 0
        func_a = orb_a[real_comp]
        func_b = orb_a[imag_comp]
        func_c = orb_b[real_comp]
        func_d = orb_b[imag_comp]
        if(func_a.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           ac = vp.dot(func_a, func_c)
        if(func_b.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           bd = vp.dot(func_b, func_d)
        if(func_a.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           ad = vp.dot(func_a, func_d)
        if(func_b.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           bc = vp.dot(func_b, func_c)
        out_real += (ac + bd) / factor
        out_imag += (ad - bc) / factor
    return out_real, out_imag

def apply_helmholtz(orbital, energy, c, prec):
    out_orbital = orbital4c("apply_helmholz",orbital.mra)
    mu = np.sqrt((c**4-energy**2)/c**2)
    print("mu",mu)
    H = vp.HelmholtzOperator(orbital.mra, mu, prec)
    for comp, func in orbital.components.items():
        if func.squaredNorm() > 0:
            vp.advanced.apply(prec, out_orbital[comp], H, func)
            out_orbital[comp] *= (-1.0/(2*np.pi))
    return out_orbital

def init_1s_orbital(orbital,k,Z,n,alpha,origin,prec):
    gamma_factor = compute_gamma(k,Z,alpha)
    norm_const = compute_norm_const(n, gamma_factor)
    idx = 0
    for comp, func in orbital.components.items():
        print('Now projecting component ',comp,idx,alpha,gamma_factor,norm_const)
        analytic_func = lambda x: one_s_alpha_comp([x[0]-origin[0],x[1]-origin[1],x[2]-origin[2]],Z,alpha,gamma_factor,norm_const,idx)
        vp.advanced.project(prec, func, analytic_func)
        idx += 1
    orbital.normalize()
    return orbital

def compute_gamma(k,Z,alpha):
    return np.sqrt(k**2 - Z**2 * alpha**2)

def compute_norm_const(n, gamma_factor):
# THIS NORMALIZATION CONSTANT IS FROM WIKIPEDIA BUT IT DOES NOT AGREE WITH Bethe&Salpeter
    tmp1 = 2 * n * (n + gamma_factor)
    tmp2 = 1 / (gamma_factor * gamma(2 * gamma_factor))
    return np.sqrt(tmp2/tmp1)

def one_s_alpha(x,Z,alpha,gamma_factor):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    tmp1 = 1.0 + gamma_factor
    tmp4 = Z * alpha
    u = x/r
    lar =   tmp1
    sai =   tmp4 * u[2]
    sbr = - tmp4 * u[1]
    sbi =   tmp4 * u[0]
    return lar, 0, 0, 0, 0, sai, sbr, sbi

def one_s_alpha_comp(x,Z,alpha,gamma_factor,norm_const,comp):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    tmp2 = r ** (gamma_factor - 1)
    tmp3 = np.exp(-Z*r)
    values = one_s_alpha(x,Z,alpha,gamma_factor)
    return values[comp] * tmp2 * tmp3 * norm_const / np.sqrt(2*np.pi)
                