from vampyr import vampyr3d as vp
import numpy as np

c = 137

class complex_fcn:
    """Complex function trees as pairs of real and imaginary trees"""
    mra = None
    def __init__(self):
        self.real = vp.FunctionTree(self.mra)
        self.imag = vp.FunctionTree(self.mra)
        self.setZero()
        
    def copy_fcns(self, real=None, imag=None):
        if not real == None:
            vp.advanced.copy_grid(self.real, real)
            vp.advanced.copy_func(self.real, real)
        if not imag == None:
            vp.advanced.copy_grid(self.imag, imag)
            vp.advanced.copy_func(self.imag, imag)
            
    def squaredNorm(self):
        re = self.real.squaredNorm()
        im = self.imag.squaredNorm()
        return re + im
    
    def normalize(self):
        norm = np.sqrt(self.squaredNorm())
        factor = 1.0 / norm 
        self.real *= factor
        self.imag *= factor
    
    def setZero(self):
        self.real.setZero()
        self.imag.setZero()

    def crop(self, prec):
        self.real.crop(prec)
        self.imag.crop(prec)

    def __add__(self, other):
        output = complex_fcn()
        output.real = self.real + other.real
        output.imag = self.imag + other.imag
        return output

    def __sub__(self, other):
        output = complex_fcn()
        output.real = self.real - other.real
        output.imag = self.imag - other.imag
        return output

    def __rmul__(self, other):
        output = complex_fcn()
        output.real = self.real * np.real(other) - self.imag * np.imag(other)
        output.imag = self.real * np.imag(other) + self.imag * np.real(other)
        return output
        
    def __mul__(self, other):
        output = complex_fcn()
        output.real = self.real * other.real - self.imag * other.imag
        output.imag = self.real * other.imag + self.imag * other.real
        return output
    
    def __str__(self):
        return ('Real part {}\n Imag part {}'.format(self.real, self.imag))
    
    def dot(self, other):
        re = vp.dot(self.real, other.real) - vp.dot(self.imag, other.imag)
        im = vp.dot(self.real, other.imag) + vp.dot(self.imag, other.real)
        return re + 1j * im
      
    def gradient(self):
        D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        grad_re = vp.gradient(D, self.real)
        grad_im = vp.gradient(D, self.imag)
        grad = []
        for i in range(3):
            comp = complex_fcn()
            comp.copy_fcns(real=grad_re[i], imag=grad_im[i])
            grad.append(comp)
        return grad

    def derivative(self, dir=0):
        D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        re_der = D(self.real, dir)
        im_der = D(self.imag, dir)
        der_func = complex_fcn()
        der_func.init_fcn(re_der, im_der)
        return der_func
        
    def density(self, prec):
        density = vp.FunctionTree(self.mra)
        add_vector = []
        temp_r = vp.FunctionTree(self.mra)
        temp_r.setZero()
        temp_i = vp.FunctionTree(self.mra)
        temp_i.setZero()
        if(self.real.squaredNorm() > 0):
            vp.advanced.multiply(prec, temp_r, 1.0, self.real, self.real)
        if(self.imag.squaredNorm() > 0):
            vp.advanced.multiply(prec, temp_i, 1.0, self.imag, self.imag)
        vp.advanced.add(prec/10, density, [temp_r, temp_i])
        return density

#
# Other is complex conjugate
#
   
    def exchange(self, other, prec):
        exchange = vp.FunctionTree(self.mra)
        add_vector = []
        a_ = vp.FunctionTree(self.mra)
        a_.setZero()
        b_ = vp.FunctionTree(self.mra)
        b_.setZero()
        c_ = vp.FunctionTree(other.mra)
        c_.setZero()
        d_ = vp.FunctionTree(other.mra)
        d_.setZero()        
        if(self.real.squaredNorm() > 0 and other.real.squaredNorm() > 0):
            vp.advanced.multiply(prec, a_, 1.0, self.real, other.real)
        if(self.imag.squaredNorm() > 0 and other.imag.squaredNorm() > 0):
            vp.advanced.multiply(prec, b_, 1.0, self.imag, other.imag)
        if(self.imag.squaredNorm() > 0 and other.imag.squaredNorm() > 0):
            vp.advanced.multiply(prec, c_, 1.0, self.real, other.imag)
        if(self.imag.squaredNorm() > 0 and other.real.squaredNorm() > 0):
            vp.advanced.multiply(prec, d_, -1.0, self.imag, other.real)        
        vp.advanced.add(prec/10, exchange, [a_, b_, c_, d_])
        return exchange
    
    def dot(self, other):
        out_real = 0
        out_imag = 0
        func_a = self.real
        func_b = self.imag
        func_c = other.real
        func_d = other.imag
        if(func_a.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_real += vp.dot(func_a, func_c)
        if(func_b.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_real += vp.dot(func_b, func_d)
        if(func_a.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_imag += vp.dot(func_a, func_d)
        if(func_b.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_imag -= vp.dot(func_b, func_c)
        return out_real, out_imag


#Not too happy about this design. Potential is only a real FunctionTree...
def apply_potential(factor, potential, func, prec):
    output = complex_fcn()
    vp.advanced.multiply(prec, output.real, factor, potential, func.real)
    vp.advanced.multiply(prec, output.imag, factor, potential, func.imag)
    return output

def apply_helmholtz(func, energy, c, prec):
    out_func = complex_fcn()
    print("Energy and c**2", energy, c)
    mu = np.sqrt((c**4-energy**2)/c**2)
    H = vp.HelmholtzOperator(func.mra, mu, prec)
    if(func.real.squaredNorm() > 0):
        vp.advanced.apply(prec, out_func.real, H, func.real)
    if(func.imag.squaredNorm() > 0):
        vp.advanced.apply(prec, out_func.imag, H, func.imag)
    return out_func

