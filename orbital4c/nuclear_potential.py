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

def read_file_with_named_lists(atomlist, number):
    charge_list = {"H" : 1, "He": 2, "Pu": 94}
    atom_list = {}
    index = 0
    with open(atomlist, 'r') as file:
        for line in file:
            terms = line.strip().split()
            charge = charge_list[terms[0]]
            atom_list[index] = [terms[0], charge, float(terms[1]), float(terms[2]), float(terms[3])]
            index += 1
        number = len(atom_list)
    return atom_list, number


def calculate_center_of_mass(atoms_list):
    total_mass = 0.0
    center_of_mass = [0.0, 0.0, 0.0]

    for atom in atoms_list.values():
        # Assuming each atom has mass 1.0 (modify if necessary)
        mass = 1.0
        total_mass += mass

        # Update the center of mass coordinates
        for i in range(3):
            center_of_mass[i] += atom[i+2] * mass

    # Calculate the weighted average to get the center of mass
    for i in range(3):
        center_of_mass[i] /= total_mass
    
    return center_of_mass
    

#def pot(coordinates, typenuc, mra, prec, der):
#    atomic_potentials = []
#    V_tree = vp.FunctionTree(mra)
#    V_tree.setZero()
#    for atom, origin in coordinates.items():
#        atom = get_original_list_name(atom)
#        print("Atom:", atom)
#        fileObj = open("Z.txt", "r")
#        charge = ""
#        for line in fileObj:
#            if not line.startswith("#"):
#                line = line.strip().split()
#                if len(line) == 2:
#                    if line[0] == atom:
#                        charge = float(line[1])
#                        print("Charge:", charge)
#        fileObj.close()
#        print("Origin:", origin)
#        print()  # Print an empty line for separation
#        
#        if typenuc == 'point_charge':
#            Peps = vp.ScalingProjector(mra,prec/10)
#            f = lambda x: point_charge(x, origin, charge)
#            V = Peps(f)
#        elif typenuc == 'coulomb_HFYGB':
#            Peps = vp.ScalingProjector(mra,prec/10)
#            f = lambda x: coulomb_HFYGB(x, origin, charge, prec)
#            V = Peps(f)
#        elif typenuc == 'homogeneus_charge_sphere':
#            Peps = vp.ScalingProjector(mra,prec/10)
#            f = lambda x: homogeneus_charge_sphere(x, origin, charge, atom)
#            V = Peps(f)
#        elif typenuc == 'gaussian':
#            Peps = vp.ScalingProjector(mra,prec/10)
#            f = lambda x: gaussian(x, origin, charge, atom)
#            V = Peps(f)
#        print("Potential for atom ", atom)
#        print(V)
#        atomic_potentials.append(V)
##    vp.advanced.add(prec, V_tree, atomic_potentials)
#    V_tree = atomic_potentials[0] + atomic_potentials[1]
#    print('Define V Potential', typenuc, 'DONE')
#    return V_tree
#




def nuclear_potential(position, atoms_list, typenuc, mra, prec, der):
    potential = 0
    for atom in atoms_list.values():
        charge = atom[1]
        atom_coordinates = [atom[2], atom[3], atom[4]]
        if typenuc == 'point_charge':
            atomic_potential = point_charge(position, atom_coordinates, charge)
        elif typenuc == 'coulomb_HFYGB':
            atomic_potential = coulomb_HFYGB(position, atom_coordinates, charge, prec)
        else:
            print("Potential not defined")
            exit(-1)
        potential += atomic_potential
    return potential

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
