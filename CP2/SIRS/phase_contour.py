import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import matplotlib as mpl

from SIRS_sim import *

sys.path.append('../GameOfLife')

from gol_sim import pos_write

# Create function that scans across all values of p1 and p3, where p2 = 0.5
def skimmer():

    lattice_size = 50
    p2 = 0.5
    # create lists for p1 and p3 resolution
    i_av_list = []
    coord_list = []
    p_list = np.linspace(0,1.05,num=21, endpoint=False)

    # p1
    for i in range(len(p_list)):
        # p3
        for j in range(len(p_list)):
            print("p1: {:.5f}   p3: {:.5f}".format(p_list[i],p_list[j]))
            # return list of I/N values from run
            i_list = run_code(lattice_size, 1100, p_list[i], p2, p_list[j], 2)
            # add true average to list
            i_av_list.append(sum(i_list)/len(i_list))
            coord_list.append([p_list[i],p_list[j]])

    # Plot 2D contour of av_list values and positions
    #plt.contourf(coord_list, i_av_list, cmap='RdGy')
    #plt.show()
    pos_write(i_av_list, "Average_energy_lists")
    pos_write(coord_list, "Coordinate_lists")

# run skimmer
skimmer()
