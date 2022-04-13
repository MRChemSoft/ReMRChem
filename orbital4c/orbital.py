from vampyr import vampyr3d as vp
import numpy as np
from .utils import *

c = 137

class orbital4c:
    """Four components orbital."""
    name: str
    components = {'Lar': None,
                  'Lai': None,
                  'Lbr': None,
                  'Lbi': None,
                  'Sar': None,
                  'Sai': None,
                  'Sbr': None,
                  'Sbi': None}
    
    def __init__(self, name, mra):
        self.name = name
        self.mra = mra
        self.initComponents()
        self.setZero()
        
    def __getitem__(self, key):
        return self.components[key]
    
    def __setitem__(self, key, val):
        self.components[key] = val

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
            orb_grad[comp] = vp.gradient(D, func) 
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