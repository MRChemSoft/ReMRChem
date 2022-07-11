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

def PoCh(position, center , charge):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    return charge / distance

def HomChSph(position, center, charge, atom):
    fileObj = open("./param_V.txt", "r")
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

def FTwoPaChDi(position, center, charge, atom):
    fileObj = open("./param_V.txt", "r")
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 4:
               if line[0] == atom:
                   C = line[2]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    C = float(C)
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    k = np.log(81)
    T = 2.30
    Fermi = np.exp(k * ((distance - C)/T))
    return charge / (1.0 + Fermi)

def GausChD(position, center, charge, atom):
    fileObj = open("./param_V.txt", "r")
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 4:
               if line[0] == atom:
                   epsilon = line[3]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    epsilon = float(epsilon)
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    u_func = erf((epsilon**1.5) * distance)
    return (charge / distance) * u_func
