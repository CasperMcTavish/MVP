import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy

# Import all functions from gol_sim.py
# Not recommended generally, but will work here
from gol_sim import *

##################################################################################################
# COLLECT EQUILIBRIUM VALUES FOR RANDOM GAME OF LIFE SIMULATIONS
##################################################################################################

# create list
equilibrium_list = []

# 150 simulations, find equilibrium iteration for each.
for i in range(150):
    print("Processing Simulation {}...".format(i))
    equilibrium_list.append(gol_sim_run(50, 0, 100, 1))
    print("Equilibrium found at {} iterations\n".format(equilibrium_list[i]))

# Plot Histogram, save figure and write equilibrium_list to file
hist = plt.hist(equilibrium_list, bins = 50)
plt.xlabel("Number of Iterations")
plt.ylabel("Counts")
plt.title("Equilibrium Time for 150 Game of Life Simulations")
plt.savefig("Equilibrium_Histogram.png")
plt.show()

pos_write(equilibrium_list, "equilibrium_list.txt")
