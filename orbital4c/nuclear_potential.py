from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1

import argparse
import numpy as np
import numpy.linalg as LA
import sys, getopt

def read_file_with_named_lists(atomlist):
    atom_lists = {}

    with open(atomlist, 'r') as file:
        for line in file:
            terms = line.strip().split()
            atom = terms[0]
            origin = terms[1:]  
            origin = [float(element) for element in origin]   

            if atom in atom_lists:
                # Append an identifier to make the key unique
                identifier = len(atom_lists[atom]) + 1
                unique_key = f"{atom}_{identifier}"
                atom_lists[unique_key] = origin
            else:
                atom_lists[atom] = origin
    total_atom_lists = len(atom_lists)
    return atom_lists, total_atom_lists

def get_original_list_name(key):
    return key.split('_')[0]


def calculate_center_of_mass(coordinates):
    total_mass = 0.0
    center_of_mass = [0.0, 0.0, 0.0]

    for atom, origin in coordinates.items():
        # Assuming each atom has mass 1.0 (modify if necessary)
        mass = 1.0
        total_mass += mass

        # Update the center of mass coordinates
        for i in range(3):
            center_of_mass[i] += origin[i] * mass

    # Calculate the weighted average to get the center of mass
    for i in range(3):
        center_of_mass[i] /= total_mass
    
    return center_of_mass
    

def pot(coordinates, typenuc, mra, prec, der):
    V_tree = vp.FunctionTree(mra)
    V_tree.setZero()
    for atom, origin in coordinates.items():
        atom = get_original_list_name(atom)
        print("Atom:", atom)
        fileObj = open("Z.txt", "r")
        charge = ""
        for line in fileObj:
            if not line.startswith("#"):
                line = line.strip().split()
                if len(line) == 2:
                    if line[0] == atom:
                        charge = float(line[1])
                        print("Charge:", charge)
        fileObj.close()
        print("Origin:", origin)
        print()  # Print an empty line for separation
        if typenuc == 'point_charge':
            Peps = vp.ScalingProjector(mra,prec/10)
            f = lambda x: point_charge(x, origin, charge)
            V = Peps(f)
        elif typenuc == 'coulomb_HFYGB':
            Peps = vp.ScalingProjector(mra,prec/10)
            f = lambda x: coulomb_HFYGB(x, origin, charge, prec)
            V = Peps(f)
        elif typenuc == 'homogeneus_charge_sphere':
            Peps = vp.ScalingProjector(mra,prec/10)
            f = lambda x: homogeneus_charge_sphere(x, origin, charge, atom)
            V = Peps(f)
        elif typenuc == 'gaussian':
            Peps = vp.ScalingProjector(mra,prec/10)
            f = lambda x: gaussian(x, origin, charge, atom)
            V = Peps(f)
        V_tree += V
    print('Define V Potential', typenuc, 'DONE')
    return V_tree

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

def get_param_homogeneous_charge_sphere(atom):
    fileObj = open("./orbital4c/param_V.txt", "r")
    RMS = ""
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 3:
               if line[0] == atom:
                   RMS = line[1]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    return float(RMS)

def homogeneus_charge_sphere(position, center, charge, RMS):
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
            if len(line) == 3:
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
