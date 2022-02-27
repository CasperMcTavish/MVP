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

# calculate variance
def variance(i_list, i_av, L, N):
    # Squared
    i_squared_list = [i**2 for i in i_list]
    i_squared_av = (sum(i_squared_list)/L)

    #variance
    return (i_squared_av-(i_av)**2)/N


# Create function that scans across all values of p1 and p3, where p2 = 0.5
def skimmer():

    lattice_size = 50
    N = lattice_size*lattice_size
    p2 = 0.5
    # create lists for p1 and p3 resolution
    i_av_list = []
    i_var_list = []
    coord_list = []
    p_list = np.linspace(0,1.05,num=21, endpoint=False)

    # p1
    for i in range(len(p_list)):
        # p3
        for j in range(len(p_list)):
            print("p1: {:.5f}   p3: {:.5f}".format(p_list[i],p_list[j]))
            # return list of I/N values and variance from run
            i_list = run_code(lattice_size, 1100, p_list[i], p2, p_list[j], 2)
            L = len(i_list)


            # Squared
            #i_squared_list = [i**2 for i in i_list]
            #i_squared_av = (sum(i_squared_list)/L)
            # Average
            i_av = (sum(i_list)/L)
            i_var_list.append(variance(i_list, i_av, L, N))

            #variance
            #i_var_list.append((i_squared_av-(i_av)**2)/(N))
            # add true average to list, mean I over N

            i_av_list.append(i_av/N)
            coord_list.append([p_list[i],p_list[j]])

    # Plot 2D contour of av_list values and positions
    #plt.contourf(coord_list, i_av_list, cmap='RdGy')
    #plt.show()
    pos_write(i_av_list, "Average_infection_lists_p1p3")
    pos_write(i_var_list, "Variance_lists_p1p3")
    pos_write(coord_list, "Coordinate_lists_p1p3")

# run skimmer
#skimmer()

# Calculate variance for p1 = 0.2->0.5
def variator():
    p2 = p3 = 0.5
    lattice_size = 50
    N = lattice_size*lattice_size
    # linear spaced points between 0.2 and 0.5, with 0.0125 resolution
    p_list = np.linspace(0.2,0.5125,num=25, endpoint=False)

    # Lists of var and error
    i_var_list = []
    var_err_list = []


    # take i, i^2 lists from each for variance calculation
    for i in range(len(p_list)):
        print("")
        print("Run p1: {}".format(p_list[i]))
        i_list = run_code(lattice_size, 10100, p_list[i], p2, p3, 2)
        L = len(i_list)


        # Squared
        i_squared_list = [i**2 for i in i_list]
        i_squared_av = (sum(i_squared_list)/L)
        # Average
        i_av = (sum(i_list)/L)

        # variance
        i_var_list.append((i_squared_av-(i_av)**2)/(N))


        # error - bootstrap method
        var_list = []
        for j in range(len(i_list)):
            i_list_new = []
            for k in range(len(i_list)):
                # collect random variables from the list and use that to give new i_list
                r = np.random.randint(0, len(i_list))
                i_list_new.append(i_list[r])
            # Take the new variance with this list then add to list for error calculation
            L = len(i_list_new)
            i_new_av = sum(i_list_new)/L
            var_list.append(variance(i_list_new, i_new_av, L, N))

        # Take these new variances and calculate the error
        squared_var = np.mean(np.square(var_list))
        var_squared = (np.mean(var_list)**2)
        var_err_list.append(np.sqrt(squared_var - var_squared))

    # Write
    pos_write(i_var_list, "Variance_lists_p1")
    pos_write(var_err_list, "Variance_error_lists_p1")
    pos_write(p_list, "Coordinate_lists_p1")

# CALL FUNCTION
# check if not imported
if __name__ == "__main__":
    # Check to make sure enough arguments
    if len(sys.argv) == 2:
        if int(sys.argv[1]) == 0:
            skimmer()
        elif int(sys.argv[1]) == 1:
            variator()
    else:
        print("\nScript takes exactly 1 argument, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\n 0 - Scan across p1 and p3 with p2 = 0.5, from 0->1 with resolution 0.05, 1000 sweeps\n 1 - Scan across p1 = 0.2->0.5 with p2 = p3 = 0.5 and collect errors, 10000 sweeps")
