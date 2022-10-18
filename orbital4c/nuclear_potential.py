import numpy as np
import copy as cp
from scipy.special import erf


def point_charge(position, center , charge):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    return charge / distance

def coulomb_HFYGB(position, center, charge, precision):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    def smoothing_HFYGB(charge, prec):
        factor = 0.00435 * prec / charge**5
        return factor**(1./3.)
    def uHFYGB(r):
        u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
        return u
    factor = smoothing_HFYGB(charge, precision)
    value = uHFYGB(distance/factor)
    return charge * value / factor

def homogeneus_charge_sphere(position, center, charge, atom):
    fileObj = open("./orbital4c/param_V.txt", "r")
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 4:
               if line[0] == atom:
                   RMS = line[1]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    RMS = float(RMS)
    RMS2 = RMS**2.0
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    R0 = (RMS2*(5.0/3.0))**0.5
    if distance <= R0:
          prec = charge / (2.0*R0)
          factor = 3.0 - (distance**2.0)/(R0**2.0) 
    else:
          prec = charge / distance
          factor = 1.0  
    return prec * factor


def gaussian(position, center, charge, atom):
    fileObj = open("./orbital4c/param_V.txt", "r")
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 4:
               if line[0] == atom:
                  epsilon = line[2]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    epsilon = float(epsilon)
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    prec = charge / distance 
    u = erf(np.sqrt(epsilon) * distance)
    return prec * u
