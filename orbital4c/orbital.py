from vampyr import vampyr3d as vp
import numpy as np
import copy as cp
from scipy.special import gamma
from orbital4c import complex_fcn as cf

class orbital4c:
    """Four components orbital."""
    mra = None
    light_speed = -1.0
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
        
    def __len__(self):
        return 4

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

    def save(self, name):
        self.comp_array[0].save(f"{name}_Large_alpha")
        self.comp_array[1].save(f"{name}_Large_beta")
        self.comp_array[2].save(f"{name}_Small_alpha")
        self.comp_array[3].save(f"{name}_Small_beta")

    def read(self, name):
        self.comp_array[0].read(f"{name}_Large_alpha")
        self.comp_array[1].read(f"{name}_Large_beta")
        self.comp_array[2].read(f"{name}_Small_alpha")
        self.comp_array[3].read(f"{name}_Small_beta")

    def __rmul__(self, factor):
        output = orbital4c()
        output.comp_array =  factor * self.comp_array
        return output

    def __mul__(self, factor):
        output = orbital4c()
        output.comp_array =  factor * self.comp_array 
        return output   

    def norm(self):
        out = 0
        for comp in self.comp_dict.keys():
            comp_norm = self[comp].squaredNorm()
            out += comp_norm
        out = np.sqrt(out)
        return out

    def squaredNorm(self):
        out = 0
        for comp in self.comp_dict.keys():
            comp_norm = self[comp].squaredNorm()
            out += comp_norm
        return out

    def squaredLargeNorm(self):
        alpha_ns = self.squaredNormComp('La')
        beta_ns = self.squaredNormComp('Lb')
        return alpha_ns + beta_ns

    def squaredSmallNorm(self):
        alpha_ns = self.squaredNormComp('Sa')
        beta_ns = self.squaredNormComp('Sb')
        return alpha_ns + beta_ns
    
    def squaredNormComp(self, comp):
        return self[comp].squaredNorm()

    def crop(self, prec):
        for func in self.comp_array:
            func.crop(prec)

    def cropLargeSmall(self, prec):
        largeNorm = np.sqrt(self.squaredLargeNorm())
        smallNorm = np.sqrt(self.squaredSmallNorm())
        precLarge = prec * largeNorm
        precSmall = prec * smallNorm
        print('precisions', precLarge, precSmall)
        self['La'].crop(precLarge, True)
        self['Lb'].crop(precLarge, True)
        self['Sa'].crop(precSmall, True)
        self['Sb'].crop(precSmall, True)        
        
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

        sigma_p_L *= -0.5j/orbital4c.light_speed
        self['Sa'] = sigma_p_L[0]
        self['Sb'] = sigma_p_L[1]
        
    def derivative(self, dir = 0, der = 'ABGV'):
        orb_der = orbital4c()
        for comp,func in self.comp_array.items():
            orb_der[comp] = func.derivative(dir, der) 
        return orb_der
    
    def gradient(self, der = 'ABGV'):
        orb_grad = {}
        for key in self.comp_dict.keys():
            orb_grad[key] = self[key].gradient(der)
        grad = []
        for i in range(3):
            comp = orbital4c()
            comp.copy_components(La = orb_grad['La'][i], 
                          Lb = orb_grad['Lb'][i], 
                          Sa = orb_grad['Sa'][i], 
                          Sb = orb_grad['Sb'][i])
            grad.append(comp)
        return grad
    
    def complex_conj(self):
        orb_out = orbital4c()
        for key in self.comp_dict.keys():
            orb_out[key] = self[key].complex_conj() 
        return orb_out

    def density(self, prec):
        density = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_array:
            temp = comp.density(prec).crop(prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))
        vp.advanced.add(prec/10, density, add_vector)
        return density    

    def exchange(self, other, prec):
        exchange = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_dict.keys():
            func_i = self[comp]
            func_j = other[comp]
            temp = func_i.exchange(func_j, prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))    
        vp.advanced.add(prec/10, exchange, add_vector)
        return exchange

    def alpha_exchange(self, other, prec):
        alpha_exchange = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_dict.keys():
            func_i = self[comp]
            func_j = other[comp]
            temp = func_i.alpha_exchange(func_j, prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))    
        vp.advanced.add(prec/10, alpha_exchange, add_vector)
        return alpha_exchange    

    def overlap_density(self, other, prec):
        density = cf.complex_fcn()
        add_vector_real = []
        add_vector_imag = []
        for comp in self.comp_dict.keys():
            func_i = self[comp]
            func_j = other[comp]
            temp = func_i.complex_conj() * func_j
            if(temp.real.squaredNorm() > 0):
                add_vector_real.append((1.0,temp.real))
            if(temp.imag.squaredNorm() > 0):
                add_vector_imag.append((1.0,temp.imag))                    
        vp.advanced.add(prec/10, density.real, add_vector_real)
        vp.advanced.add(prec/10, density.imag, add_vector_imag)
        return density

    def alpha(self,direction):
        out_orb = orbital4c()
        alpha_order = np.array([[3, 2, 1, 0],
                                [3, 2, 1, 0],
                                [2, 3, 0, 1]])
        
        alpha_coeff = np.array([[ 1,  1,   1,  1],
                                [-1j, 1j, -1j, 1j],
                                [ 1, -1,   1, -1]])
#        Alpha = np.array([[[0, 0, 0, 1],
#                           [0, 0, 1, 0],
#                           [0, 1, 0, 0],
#                           [1, 0, 0, 0]],
#                          [[0,  0,  0,  -1j],
#                           [0,  0,  1j,  0],
#                           [0, -1j, 0,   0],
#                           [1j, 0,  0,   0]],
#                          [[0, 0, 1, 0],
#                           [0, 0, 0,-1],
#                           [1, 0, 0, 0],
#                           [0,-1, 0, 0]]])
#        out_orb.comp_array = alpha[direction]@self.comp_array
        for idx in range(4):
            coeff = alpha_coeff[direction][idx]
            comp = alpha_order[direction][idx]
            out_orb.comp_array[idx] = coeff * self.comp_array[comp]
        return out_orb

    def ktrs(self):   #Kramers´ Time Reversal Symmetry
        out_orb = orbital4c()
        tmp = self.complex_conj()
#        ktrs = np.array([[ 0,  -1,  0,    0,],
#                         [ 1,   0,  0,    0,],
#                         [ 0,   0,  0,   -1,],
#                         [ 0,   0,  1,    0,]])
        ktrs_order = np.array([1, 0, 3, 2])
        ktrs_coeff = np.array([-1,  1,  -1,  1])
#        out_orb.comp_array = ktrs@tmp.comp_array
        for idx in range(4):
            coeff = ktrs_coeff[idx]
            comp = ktrs_order[idx]
            out_orb.comp_array[idx] = coeff * tmp.comp_array[comp]
        return out_orb

#Beta c**2
    def beta(self, shift = 0):
        out_orb = orbital4c()
#        beta = np.array([[orbital4c.light_speed**2 + shift, 0, 0, 0  ],
#                         [0, orbital4c.light_speed**2 + shift, 0, 0  ],
#                         [0, 0, -orbital4c.light_speed**2 + shift, 0 ],
#                         [0, 0,  0, -orbital4c.light_speed**2 + shift]])
#        out_orb.comp_array = beta@self.comp_array
        beta = np.array([orbital4c.light_speed**2 + shift,
                         orbital4c.light_speed**2 + shift,
                        -orbital4c.light_speed**2 + shift,
                        -orbital4c.light_speed**2 + shift])
        for idx in range(4):
            out_orb.comp_array[idx] = beta[idx] * self.comp_array[idx]
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

    def pota(self, other):
        out_real = 0
        out_imag = 0
        for comp in self.comp_dict.keys():
            factor = 1
#            if('S' in comp) factor = c**2
            cr, ci = self[comp].pota(other[comp])
            out_real += cr
            out_imag += ci
        return out_real, out_imag
#    
# here we should consider emulating the behavior of MRChem operators
#
def matrix_element(bra, operator, ket):
    Opsi = operator(ket)
    return bra.dot(Opsi)
                   
def apply_dirac_hamiltonian(orbital, prec, shift = 0.0, der = 'ABGV'):
    beta_phi = orbital.beta(shift)
    grad_phi = orbital.gradient(der)
    alpx_phi = -1j * orbital4c.light_speed * grad_phi[0].alpha(0)
    alpy_phi = -1j * orbital4c.light_speed * grad_phi[1].alpha(1)
    alpz_phi = -1j * orbital4c.light_speed * grad_phi[2].alpha(2)
    return beta_phi + alpx_phi + alpy_phi + alpz_phi

def apply_potential(factor, potential, orbital, prec):
    out_orbital = orbital4c()
    for comp in orbital.comp_dict:
        if orbital[comp].squaredNorm() > 0:
            out_orbital[comp] = cf.apply_potential(factor, potential, orbital[comp], prec)
    return out_orbital

def apply_complex_potential(factor, potential, orbital, prec):
    out_orbital = orbital4c()
    for comp in orbital.comp_dict:
        if orbital[comp].squaredNorm() > 0:
            out_orbital[comp] = potential * orbital[comp] 
    return out_orbital

#
# Keep this for now to maybe enable precise addition later
#
#def add_orbitals(a, orb_a, b, orb_b, prec):
#    out_orb = orbital4c("a_plus_b",orb_a.mra)
#    for comp, func in out_orb.components.items():        
#        func_a = orb_a[comp]
#        func_b = orb_b[comp]
#        if (func_a.squaredNorm() > 0 and func_b.squaredNorm() > 0):
#            vp.advanced.add(prec/10, func, a, func_a, b, func_b)
#        elif(func_a.squaredNorm() > 0):
#            out_orb.init_function(func_a, comp)
#            func *= a
#        elif(func_b.squaredNorm() > 0):
#            out_orb.init_function(func_b, comp)
#            func *= b
#        else:
#            print('Warning: adding two empty trees')
#    return out_orb

def apply_helmholtz(orbital, energy, prec):
    out_orbital = orbital4c()
    for comp in orbital.comp_dict.keys():
        out_orbital[comp] = cf.apply_helmholtz(orbital[comp], energy, orbital4c.light_speed, prec)
    out_orbital.rescale(-1.0/(2*np.pi))
    return out_orbital

def init_1s_orbital(orbital,k,Z,n,alpha,origin,prec):
    gamma_factor = compute_gamma(k,Z,alpha)
    norm_const = compute_norm_const(n, gamma_factor)
    idx = 0
    for comp in orbital.comp_array:
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
# and most importantly, it is wrong :-)
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
