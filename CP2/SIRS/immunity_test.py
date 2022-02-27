import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import matplotlib as mpl
from scipy.stats import sem

from SIRS_sim import *

sys.path.append('../GameOfLife')

from gol_sim import pos_write



######
# Create a graph of the required fraction of immunity to stop infection, across 5 runs.
######

def immune():
    # preset values
    lattice_size = 50
    iterations = 1100
    p1 = p2 = p3 = 0.5
    vis = 3
    N = lattice_size*lattice_size
    i_frac_list = []
    i_error_list = []
    # immunity list from 0.1 to 0.6 in steps of 0.05
    imm_list = np.linspace(0.1,0.5,num=11, endpoint=True)
    # iterate over immunity fractions
    for j in range(len(imm_list)):
        print("Calculating Immunity {}/{}...".format(j,len(imm_list)))
        # 5 iterations per each scan
        inf_frac_temp = []
        for i in range(5):
            print("Repeating: {}/{}".format(i,5))
            # collect list of total infections
            inf_total = run_code(lattice_size, iterations, p1, p2, p3, vis, imm_list[j])
            # calculate averages over iterations
            inf_mean = np.mean(inf_total)/N
            inf_frac_temp.append(inf_mean)

        # Calculate mean and error for each point, then append to frac list
        i_frac_list.append(np.mean(inf_frac_temp))
        i_error_list.append(sem(inf_frac_temp))
        print("Average infection at {} immunity: {:.3f}({:.5f})".format(imm_list[j],i_frac_list[j],i_error_list[j]))


    # Plot it in here, then Write
    #plt.errorbar(imm_list, i_frac_list, yerr=i_error_list, fmt = 'o')
    plt.errorbar(imm_list, i_frac_list, yerr=i_error_list)
    plt.xlabel("Immunity fraction")
    plt.ylabel("Average Infected fraction of the population")
    plt.title("Infected fraction of the population against immunity rate (p1=p2=p3=0.5)")
    plt.savefig("immunity_fraction.png")
    plt.show()


    pos_write(i_frac_list,"immunity_inf_list")
    pos_write(i_error_list,"immunity_inf_error_list")
    pos_write(imm_list,"immunity_fraction_list")


immune()
