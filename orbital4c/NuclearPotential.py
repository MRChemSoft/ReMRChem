import numpy as np
import copy as cp
from scipy.special import erf

def CoulombPotential(position, center, charge):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    potential = charge / distance

def SmoothingHFYGB(Z, prec):
    factor = 0.00435 * prec / Z**5
    return factor**(1./3.)
    
def CoulombHFYGB(position, center, charge, precision):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    factor = SmoothingHFYGB(charge, precision)
    value = uHFYGB(distance / factor)
    return charge * value / factor

def uHFYGB(r):
    u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
    return u

