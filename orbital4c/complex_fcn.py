from vampyr import vampyr3d as vp
import numpy as np

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

    def save(self, name):
        self.real.saveTree(f"{name}_real")
        self.imag.saveTree(f"{name}_imag")

    def read(self, name):
        self.real.loadTree(f"{name}_real")
        self.imag.loadTree(f"{name}_imag")

    def cropRealImag(self, prec):
        norm = np.sqrt(self.squaredNorm())
        self.crop(prec * norm, True)

    def crop(self, prec, abs = False):
        self.real.crop(prec, abs)
        self.imag.crop(prec, abs)

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
        
    def __str__(self):
        return ('Real part {}\n Imag part {}'.format(self.real, self.imag))
    
    def gradient(self, der = 'ABGV'):
        if(der == 'ABGV'):
            D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        elif(der == 'PH'):
            D = vp.PHDerivative(self.mra)
        elif(der == 'BS'):
            D = vp.BSDerivative(self.mra)
        else:
            exit("Derivative operator not found")
        grad_re = vp.gradient(D, self.real)
        grad_im = vp.gradient(D, self.imag)
        grad = []
        for i in range(3):
            comp = complex_fcn()
            comp.copy_fcns(real=grad_re[i], imag=grad_im[i])
            grad.append(comp)
        return grad

    def derivative(self, dir = 0, der = 'ABGV'):
        if(der == 'ABGV'):
            D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        elif(der == 'PH'):
            D = vp.PHDerivative(self.mra)
        elif(der == 'BS'):
            D = vp.BSDerivative(self.mra)
        else:
            exit("Derivative operator not found")
        re_der = D(self.real, dir)
        im_der = D(self.imag, dir)
        der_func = complex_fcn()
        der_func.real = re_der
        der_func.imag = im_der
        return der_func

    def complex_conj(self):
        output = complex_fcn()
        output.real = self.real 
        output.imag = -1.0 * self.imag
        return output
       
        
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

#    def exchange(self, other, prec):
#        exchange = vp.FunctionTree(self.mra)
#        add_vector = []
#        a_ = vp.FunctionTree(self.mra)
#        a_.setZero()
#        b_ = vp.FunctionTree(self.mra)
#        b_.setZero()
#        c_ = vp.FunctionTree(other.mra)
#        c_.setZero()
#        d_ = vp.FunctionTree(other.mra)
#        d_.setZero()        
#        if(self.real.squaredNorm() > 0 and other.real.squaredNorm() > 0):
#            vp.advanced.multiply(prec, a_, 1.0, self.real, other.real)
#        if(self.imag.squaredNorm() > 0 and other.imag.squaredNorm() > 0):
#            vp.advanced.multiply(prec, b_, 1.0, self.imag, other.imag)
#        if(self.real.squaredNorm() > 0 and other.imag.squaredNorm() > 0):
#            vp.advanced.multiply(prec, c_, 1.0, self.real, other.imag)
#        if(self.imag.squaredNorm() > 0 and other.real.squaredNorm() > 0):
#            vp.advanced.multiply(prec, d_, -1.0, self.imag, other.real)        
#        vp.advanced.add(prec/10, exchange, [a_, b_, c_, d_])
#        return exchange

    def alpha_exchange(self, other, prec):
        alpha_exchange = vp.FunctionTree(self.mra)
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
        if(self.real.squaredNorm() > 0 and other.imag.squaredNorm() > 0):
            vp.advanced.multiply(prec, c_, 1.0, self.real, other.imag)
        if(self.imag.squaredNorm() > 0 and other.real.squaredNorm() > 0):
            vp.advanced.multiply(prec, d_, -1.0, self.imag, other.real)        
        vp.advanced.add(prec/10, alpha_exchange, [a_, b_])
        return alpha_exchange
    
    def dot(self, other, cc_first = True):
        out_real = 0
        out_imag = 0
        func_a = self.real
        func_b = self.imag
        func_c = other.real
        func_d = other.imag

        fbd = 1.0
        fbc = -1.0
        if(not cc_first):
            fbd = -1.0
            fbc = 1.0
        
        if(func_a.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_real = out_real + vp.dot(func_a, func_c)
        if(func_b.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_real = out_real + fbd * vp.dot(func_b, func_d)
        if(func_a.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_imag = out_imag + vp.dot(func_a, func_d)
        if(func_b.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_imag = out_imag + fbc * vp.dot(func_b, func_c)

        return out_real, out_imag

    def advanced_overlap_density(self, other, prec):
        func_a = self.real
        func_b = self.imag
        func_c = other.real
        func_d = other.imag

        rr = vp.FunctionTree(self.mra)
        ii = vp.FunctionTree(self.mra)
        ri = vp.FunctionTree(self.mra)
        ir = vp.FunctionTree(self.mra)

        if(func_a.squaredNorm() > 0 and func_c.squaredNorm() > 0):
            vp.advanced.multiply(prec, rr, 1.0, func_a, func_c)
        if(func_b.squaredNorm() > 0 and func_d.squaredNorm() > 0):
            vp.advanced.multiply(prec, ii, 1.0, func_b, func_d)
        if(func_a.squaredNorm() > 0 and func_d.squaredNorm() > 0):
            vp.advanced.multiply(prec, ri, 1.0, func_a, func_d)
        if(func_b.squaredNorm() > 0 and func_c.squaredNorm() > 0):
            vp.advanced.multiply(prec, ir, -1.0, func_b, func_c)

        output = complex_fcn()
        output.real = rr + ii
        output.imag = ri + ir

        return output




    #Not too happy about this design. Potential is only a real FunctionTree...
def apply_potential(factor, potential, func, prec):
    output = complex_fcn()
    vp.advanced.multiply(prec, output.real, factor, potential, func.real)
    vp.advanced.multiply(prec, output.imag, factor, potential, func.imag)
    return output

def apply_helmholtz(func, energy, light_speed, prec):
    out_func = complex_fcn()
    mu = np.sqrt((light_speed**4-energy**2)/light_speed**2)
    H = vp.HelmholtzOperator(func.mra, mu, prec)
    if(func.real.squaredNorm() > 0):
        vp.advanced.apply(prec, out_func.real, H, func.real)
    if(func.imag.squaredNorm() > 0):
        vp.advanced.apply(prec, out_func.imag, H, func.imag)
    return out_func

def apply_poisson(func, mra, P, prec, thresholdNorm = 0, factor = 1.0):
    out_func = complex_fcn()
    if(func.real.squaredNorm() > thresholdNorm):
        vp.advanced.apply(prec, out_func.real, P, func.real)
        out_func.real *= factor
    if(func.imag.squaredNorm() > thresholdNorm):
        vp.advanced.apply(prec, out_func.imag, P, func.imag)
        out_func.imag *= factor
    out_func.cropRealImag(prec)
    return out_func

def multiply(prec, lhs, rhs):
    rr = vp.FunctionTree(lhs.mra)
    ri = vp.FunctionTree(lhs.mra)
    ir = vp.FunctionTree(lhs.mra)
    ii = vp.FunctionTree(lhs.mra)
    vp.advanced.multiply(prec, rr, 1.0, lhs.real, rhs.real, -1, True)
    vp.advanced.multiply(prec, ri, 1.0, lhs.real, rhs.imag, -1, True)
    vp.advanced.multiply(prec, ir, 1.0, lhs.imag, rhs.real, -1, True)
    vp.advanced.multiply(prec, ii, 1.0, lhs.imag, rhs.imag, -1, True)
    output = complex_fcn()
    output.real = rr - ii
    output.imag = ri + ri
    output.crop(prec)
    return output

def divergence(vector, prec, der = "BS"):
    out = complex_fcn()
    der_vec = {}
    for i in range(3):
        der_vec[i] = vector[i].derivative(i, der)
    out = der_vec[0] + der_vec[1] + der_vec[2]
    out.cropRealImag(prec)
    return out

def vector_dot_r(vector, prec):
    out = complex_fcn()
    components = {}
    projection_operator = vp.ScalingProjector(out.mra, prec)
    for i in range(3):
        r_i = projection_operator(lambda x : x[i])
        components[i] = apply_potential(1.0, r_i, vector[i], prec)
    out = components[0] + components[1] + components[2]
    out.cropRealImag(prec)
    return out

def scalar_times_r(function, prec):
    components = {}
    projection_operator = vp.ScalingProjector(function.mra, prec)
    for i in range(3):
        r_i = projection_operator(lambda x : x[i])
        components[i] = apply_potential(1.0, r_i, function, prec)
        components[i].cropRealImag(prec)
    return components
