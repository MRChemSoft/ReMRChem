from vampyr import vampyr3d as vp
import numpy as np
from .orbital import *

c = 137

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
    for comp, func in orb.items():
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

def apply_dirac_hamiltonian(orbital, shift = 0.0):
    out_orbital = orb.orbital4c("Hpsi",orbital.mra)
    orb_grad = out_orbital.gradient()
    add_vectors = assemble_vectors(orbital, orb_grad, shift)
    for comp, func in out_orbital.items():
        vp.advanced.add(prec/10, func, add_vectors[comp])
    return out_orbital

def apply_potential(nuclear_potential, orbital):
    out_orbital = init_empty_orbital()
    for comp, func in orbital.items():
        if func.squaredNorm() > 0:
            vp.advanced.multiply(prec, out_orbital[comp], -1.0, nuclear_potential, func)
    return out_orbital

def add_orbitals(a, orb_a, b, orb_b):
    out_orb = init_empty_orbital()
    for comp, func in out_orb.items():
        func_a = orb_a[comp]
        func_b = orb_b[comp]
        if (func_a.squaredNorm() > 0 and func_b.squaredNorm() > 0):
            vp.advanced.add(prec/10, func, a, func_a, b, func_b)
        elif(func_a.squaredNorm() > 0):
            init_function(out_orb, func_a, comp)
            func *= a
        elif(func_b.squaredNorm() > 0):
            init_function(out_orb, func_b, comp)
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

def init_1s_orbital(k,Z,alpha,origin):
    gamma_factor = compute_gamma(k,Z,alpha)
    norm_const = compute_norm_const(n, gamma_factor)
    idx = 0
    for comp, func in orbital.items():
        print('Now projecting component ',comp,idx,alpha,gamma_factor,norm_const)
        analytic_func = lambda x: one_s_alpha_comp([x[0]-origin[0],x[1]-origin[1],x[2]-origin[2]],Z,alpha,gamma_factor,norm_const,idx)
        vp.advanced.project(prec, func, analytic_func)
        idx += 1
    return orbital

#def normalize_orbital(orbital):
#    norm_sq, imag1 = scalar_product(orbital, orbital)
#    norm = np.sqrt(norm_sq)
#    rescale_orbital(orbital, 1.0/norm)

def apply_helmholtz(orbital, energy, c, prec):
    out_orbital = init_empty_orbital()
    mu = np.sqrt((c**4-energy**2)/c**2)
    H = vp.HelmholtzOperator(mra, mu, prec)
    for comp, func in orbital.items():
        if func.squaredNorm() > 0:
            vp.advanced.apply(prec, out_orbital[comp], H, func)
            out_orbital[comp] *= (-1.0/(2*np.pi))
    return out_orbital

#def rescale_orbital(orbital, factor):
#    for key in orbital.components.keys():
#        if(orbital.components[key].squaredNorm() > 0):
#            orbital.components[key] *= factor
            
#def compute_density(orbital):
#    density = vp.FunctionTree(mra)
#    add_vector = []
#    for comp, func in orbital.items():
#        if(func.squaredNorm() > 0):
#            temp = vp.FunctionTree(mra)
#            vp.advanced.multiply(prec, temp, 1.0, func, func)
#            add_vector.append((1.0,temp))
#    vp.advanced.add(prec/10, density, add_vector)
#    return density
