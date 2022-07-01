import numpy as np
import copy as cp

def CoulombPotential(position, center, charge):
    distance = np.linalg.norm(positon - center)
    potential = charge / distance

def SmoothingHFYGB(Z, prec):
    factor = 0.00435 * prec / Z**5
    return factor**(1./3.)
    
def CoulombHFYGB(position, center, charge, precison):
    distance = np.linalg.norm(positon - center)
    factor = SmoothingHFYGB(charge, precision)
    value = uHFYGB(distance / factor)
    return charge * value / factor

def uHFYGB(r):
    u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
    return u

