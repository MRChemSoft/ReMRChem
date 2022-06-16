from vampyr import vampyr3d as vp
import numpy as np
import copy as cp
from scipy.special import gamma
from orbital4c import complex_fcn as cf

c = 137

class orbital4c:
    """Four components orbital."""
    mra = None
    comp_dict = {'La': 0, 'Lb': 1, 'Sa': 2, 'Sb': 3}
    def __init__(self):
        self.comp_array = np.array([cf.complex_fcn(),
                                    cf.complex_fcn(),
                                    cf.complex_fcn(),
                                    cf.complex_fcn()])
        
    def __getitem__(self, key):
        return self.comp_array[self.comp_dict[key]]
    
    def __setitem__(self, key, val):
        self.comp_array[self.comp_dict[key]] = val

    def __str__(self):
        return ('Large components\n alpha\n{} beta\n{} Small components\n alpha\n{} beta\n{}'.format(self["La"],
                                  self["Lb"],
                                  self["Sa"],
                                  self["Sb"]))
    
    def __add__(self, other):
        output = orbital4c()
        output.comp_array = self.comp_array + other.comp_array
        return output

    def __sub__(self, other):
        output = orbital4c()
        output.comp_array = self.comp_array - other.comp_array
        return output

    def __rmul__(self, factor):
        output = orbital4c()
        output.comp_array =  factor * self.comp_array
        return output

    def __mul__(self, factor):
        output = orbital4c()
        output.comp_array =  factor * self.comp_array 
        return output   

    def crop(self, prec):
        for func in self.comp_array:
            func.crop(prec)

    def setZero(self):
        for func in self.comp_array:
            func.setZero()

    def rescale(self, factor):
        self.comp_array *= factor
            
    def copy_component(self, func, component='La'):
        self[component].copy_fcns(func.real, func.imag)
        
    def normalize(self):
        norm_sq = 0
        for comp in self.comp_array:
            norm_sq += comp.squaredNorm()
        norm = np.sqrt(norm_sq)
        self.rescale(1.0/norm)
    
    def copy_components(self, La=None, Lb=None, Sa=None, Sb=None):
        nr_of_functions = 0
        if(La != None):
            nr_of_functions += 1
            self.copy_component(La, 'La')
        if(Lb != None):
            nr_of_functions += 1
            self.copy_component(Lb, 'Lb')
        if(Sa != None):
            nr_of_functions += 1
            self.copy_component(Sa, 'Sa')
        if(Sb != None):
            nr_of_functions += 1
            self.copy_component(Sb, 'Sb')
        if(nr_of_functions == 0):
            print("WARNING: No component copied!")
        
    def init_small_components(self,prec):
    # initalize the small components based on the kinetic balance
        grad_a = self['La'].gradient()
        grad_b = self['Lb'].gradient()
        plx = np.array([grad_a[0],grad_b[0]])
        ply = np.array([grad_a[1],grad_b[1]])
        plz = np.array([grad_a[2],grad_b[2]])
        sigma_x = np.array([[0,1],  
                            [1,0]])
        sigma_y = np.array([[0,-1j],
                            [1j,0]])
        sigma_z = np.array([[1,0],  
                            [0,-1]])

        sLx = sigma_x@plx
        sLy = sigma_y@ply
        sLz = sigma_z@plz
        
        sigma_p_L = sLx + sLy + sLz

        sigma_p_L *= -0.5j/c
        self['Sa'] = sigma_p_L[0]
        self['Sb'] = sigma_p_L[1]
        
    def derivative(self, dir=0):
        orb_der = orbital4c("derivative", orbital.mra)
        for comp,func in self.components.items():
            orb_der[comp] = func.derivative(dir) 
        return orb_der
    
    def gradient(self):
        orb_grad = {}
        for key in self.comp_dict.keys():
            orb_grad[key] = self[key].gradient()
        grad = []
        for i in range(3):
            comp = orbital4c()
            comp.copy_components(La = orb_grad['La'][i], 
                          Lb = orb_grad['Lb'][i], 
                          Sa = orb_grad['Sa'][i], 
                          Sb = orb_grad['Sb'][i])
            grad.append(comp)
        return grad
    
    def density(self, prec):
        density = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_array:
            temp = comp.density(prec).crop(prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))
        vp.advanced.add(prec/10, density, add_vector)
        return density

    #CT
    def exchange(self, other, prec):
        exchange = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_array:
            temp = comp.density(prec).crop(prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))    
        vp.advanced.add(prec/10, exchange, add_vector)
        return exchange

    def alpha(self,index):
        out_orb = orbital4c()
        alpha = np.array([[[0, 0, 0, 1],  
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [1, 0, 0, 0]],
                          [[0,  0,  0,  -1j],
                           [0,  0,  1j,  0],
                           [0, -1j, 0,   0],
                           [1j, 0,  0,   0]],
                          [[0, 0, 1, 0],
                           [0, 0, 0,-1],
                           [1, 0, 0, 0],
                           [0,-1, 0, 0]]])
        out_orb.comp_array = alpha[index]@self.comp_array
        return out_orb
    
    def beta(self, shift = 0):
        out_orb = orbital4c()
        beta = np.array([[c**2 + shift, 0, 0, 0  ],
                         [0, c**2 + shift, 0, 0  ],
                         [0, 0, -c**2 + shift, 0 ],
                         [0, 0,  0, -c**2 + shift]])
        out_orb.comp_array = beta@self.comp_array
        return out_orb
    
    def dot(self, other):
        out_real = 0
        out_imag = 0
        for comp in self.comp_dict.keys():
            factor = 1
#            if('S' in comp) factor = c**2
            cr, ci = self[comp].dot(other[comp])
            out_real += cr
            out_imag += ci
        return out_real, out_imag

    def norm(self):
        out = 0
        for comp in self.comp_dict.keys():
            comp_norm = self[comp].squaredNorm()
            out += comp_norm
        out = np.sqrt(out)
        return out

    #CT
    def div(self, other):
        for comp in self.comp_dict.keys():
            factor = 1
            z = self[comp]
            w = other[comp]
            a = z.real
            b = z.imag
            c = w.real
            d = w.imag
            numreal = a*c + b*d
            numimag = b*c - a*d
            denom = c*c + d*d
            out_real = np.divide(numreal,denom)
            out_imag = np.divide(numimag,denom)
        return complex(out_real, out_imag)

def grab_sign(comp, derivative):
    grab_table = {
        'Lar': ( 1, -1,  1), 
        'Lai': (-1, -1, -1), 
        'Lbr': ( 1,  1, -1), 
        'Lbi': (-1,  1,  1), 
        'Sar': ( 1, -1,  1), 
        'Sai': (-1, -1, -1), 
        'Sbr': ( 1,  1, -1), 
        'Sbi': (-1,  1,  1)}
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
        'Sbi': ('Lar', 'Lai', 'Lbr')
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
    beta_phi = orbital.beta(shift)
    grad_phi = orbital.gradient()
    alpx_phi = -1j * c * grad_phi[0].alpha(0)
    alpy_phi = -1j * c * grad_phi[1].alpha(1)
    alpz_phi = -1j * c * grad_phi[2].alpha(2)
    return beta_phi + alpx_phi + alpy_phi + alpz_phi

def apply_potential(factor, potential, orbital, prec):
    out_orbital = orbital4c()
    for comp in orbital.comp_dict:
        if orbital[comp].squaredNorm() > 0:
            out_orbital[comp] = cf.apply_potential(factor, potential, orbital[comp], prec)
    return out_orbital

def apply_helmholtz(orbital, energy, c, prec):
    out_orbital = orbital4c()
    for comp in orbital.comp_dict.keys():
        out_orbital[comp] = cf.apply_helmholtz(orbital[comp], energy, c, prec)
    out_orbital.rescale(-1.0/(2*np.pi))
    return out_orbital

def init_1s_orbital(orbital,k,Z,n,alpha,origin,prec):
    gamma_factor = compute_gamma(k,Z,alpha)
    norm_const = compute_norm_const(n, gamma_factor)
    idx = 0
    for comp in orbital.comp_array:
        print('Now projecting component ',comp,idx,alpha,gamma_factor,norm_const)
        func_real = lambda x: one_s_alpha_comp([x[0]-origin[0], x[1]-origin[1], x[2]-origin[2]],
                                                Z, alpha, gamma_factor, norm_const, idx)
        func_imag = lambda x: one_s_alpha_comp([x[0]-origin[0], x[1]-origin[1], x[2]-origin[2]],
                                                Z, alpha, gamma_factor, norm_const, idx+1 )
        vp.advanced.project(prec, comp.real, func_real)
        vp.advanced.project(prec, comp.imag, func_imag)
        idx += 2
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
