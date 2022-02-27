import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


from SIRS_sim import *

sys.path.append('../GameOfLife')

from gol_sim import read_file, pos_write

#############
# Altering energy list to be accurate
# This was just due to my mistakes in dividing our infection results by correct values
# Here we apply another division of lattice_size to our data, as we only did it once in the initial calculation.
#############

lattice_size = 50
N = lattice_size

# read and clean data
data = read_file("Average_energy_lists")
data = [data[i].rstrip().lstrip() for i in range(len(data))]
data = (np.array(data)).astype(float)

new_data = [data[i]/N for i in range(len(data))]

pos_write(new_data, "Proper_Average_energy_lists")
