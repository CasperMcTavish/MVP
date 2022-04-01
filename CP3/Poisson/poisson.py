import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to zero except central point source
def init_array(lattice_size, gam):

    # set array of zeros except for central point, this is rho array
    rho = np.zeros((lattice_size,lattice_size,lattice_size), dtype=float)
    # set central point to 1
    mid_point = int(lattice_size/2)
    rho[mid_point,mid_point,mid_point] = 1

    array = np.zeros((lattice_size,lattice_size,lattice_size), dtype=float)
    return array, rho

# QUIVER MATPLOTLIB

def update_gam(array, lattice_size, dt, dx):
    # update the gamma values

    # complete the discretised calculation
    newarray = 1/6 * (np.roll(array,1,axis=0) + np.roll(array,-1,axis=0) + np.roll(array,1,axis=1) + np.roll(array,-1,axis=1) + np.roll(array,1,axis=2) + np.roll(array,-1,axis=2) + rho)
    return newarray

def iterator(lattice_size, iterations):

    # iterate 3D array?
    
