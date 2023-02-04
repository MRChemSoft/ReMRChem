########## Define Enviroment #################
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1
import numpy as np

class GaugeOperator():
    def __init__(self, mra, r_min, r_max, prec):
        iexp = self.fill_separated_expansion(r_min, r_max, prec)
        self.gauge = vp.CartesianConvolution(mra, iexp, prec)   

    def alpha_beta_rm3(t):
        c = np.cosh(t)
        s = np.sinh(t)
        e = np.exp(-s)
        alpha = (4 / np.sqrt(np.pi)) * c / (1+e)
        beta = (np.log(1+e)+s)**2
        return alpha * beta, beta

    def integrand(self, r,t):
        alpha, beta = self.alpha_beta_rm3(t)
        return alpha * np.exp(-beta * r**2)
        
    def calc_min_max_rm3(r_min, prec):
        t1 = 1.0
        t2 = 1.0
        while ((2 * t1 * np.exp(-t1)) > prec): t1 *= 1.1
        while ((np.sqrt(t2) * np.exp(-t2) / r_min) > prec): t2 *= 1.1
        t_min = - np.log(2 * t1)
        t_max =   np.log(t2 / (r_min * r_min))/1.8
        return t_min, t_max
        
    def fill_separated_expansion(self, r_min, r_max, prec):
        r_0 = r_min / r_max
        t_min, t_max = self.calc_min_max_rm3(r_0, prec)
        h = 1 / (0.7 -  0.49 * np.log10(prec))
        n_exp = np.int(np.ceil((t_max - t_min) / h) + 1)
        ab = np.zeros(shape=(2,n_exp))
        for i in range(n_exp):
            t = t_min + h * i
            ab[0][i], ab[1][i] = self.alpha_beta_rm3(t)
        ab[0] *= h / r_max**3
        ab[1] /= r_max**2
        ab[0][0] /= 2
        ab[0][n_exp-1] /= 2
        iexp = vp1.GaussExp()
        for i in range(n_exp):
            a = ab[0][i]
            b = ab[1][i] 
            ifunc = vp1.GaussFunc(a, b)
            iexp.append(ifunc)       
        return iexp
        
    def __call__(self, Phi, x, y, z):
        self.gauge.setCartesianComponents(x, y, z)
        return  self.gauge(Phi)