import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time
import scipy.optimize

# Import from the correct place all the scripts
from analysis_scripts import *


def main():

    # load phi array
    array = np.load("phiarray.npy")

    # define lattice size
    lattice_size = len(array)
    print("lattice size: {}".format(lattice_size))

    # collect E field (dx hard coded to 1 here)
    E_Fz, E_Fx, E_Fy, E_F = e_field(array, 1)

    # Normalise e field
    norm_E_F = E_F/np.max(E_F)
    # normalise main array
    norm_array = array/np.max(array)

    # Create meshgrid of x and y for plotting purposes
    x = y = np.linspace(0,len(array)-1,len(array))
    X,Y = np.meshgrid(x,y)

    # Show electric across slice of array
    imsh = plt.imshow(E_F[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Electric field at centre of array")
    plt.show()


    # Determind E field gradient along x and y
    e_f_grad = np.gradient(E_F[int(lattice_size/2)])
    e_fx = (e_f_grad[0]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))
    e_fy = (e_f_grad[1]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))

    # flip to get accurate e-field lines
    e_fx = e_fx * -1
    e_fy = e_fy * -1

    # plot quivers of e field gradient (step skips positions to make lines more visible)
    step = 2

    plt.quiver(X[::step,::step],Y[::step,::step], e_fy[::step,::step],e_fx[::step,::step])
    plt.title("Electric field from central slice")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("efield.png")
    plt.show()


    # plot potential across sclice of array
    imsh = plt.imshow(array[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()
    #quit()

    # Find midpoint
    midpoint = int(lattice_size/2)

    rad_array = create_radial_array_point(lattice_size, midpoint)


    # sort through both arrays to create tuples that can be plotted on scatter diagrams
    # this takes a while! lots of data points
    zipped_potential = []
    zipped_norm_potential = []
    zipped_log_norm_potential = []
    zipped_log_R_norm_potential = []
    zipped_e_field = []
    zipped_log_R_log_E = []
    R_V_E_array = []
    zipped_logR_logV = []

    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # make main array
                R_V_E_array.append((rad_array[i,j,k], norm_array[i,j,k], norm_E_F[i,j,k]))

                # add tuples
                zipped_potential.append((rad_array[i,j,k],array[i,j,k]))
                zipped_norm_potential.append((rad_array[i,j,k], norm_array[i,j,k]))
                zipped_log_norm_potential.append((rad_array[i,j,k], np.log(norm_array[i,j,k])))
                zipped_logR_logV.append((np.log(rad_array[i,j,k]),np.log(array[i,j,k])))
                zipped_log_R_norm_potential.append((np.log(rad_array[i,j,k]), norm_array[i,j,k]))
                zipped_e_field.append((rad_array[i,j,k], E_F[i,j,k]))
                zipped_log_R_log_E.append((np.log(rad_array[i,j,k]),np.log(E_F[i,j,k])))



    # plot tuple array
    plot_tuple_array(zipped_potential, "R against Potential", "R", "Potential")

    plot_tuple_array(zipped_logR_logV, "Log R against Potential", "LogR", "LogV")

    # FIT ACROSS RANGE THAT WILL GIVE 1/R GRADIENT, about R= 1.2 -> 2.2
    plot_tuple_array_fit(zipped_logR_logV, lin_func, 0.5, 2, "Fitted LogR LogV plot", "logR", "logV")





    # Plotting E field contents
    plot_tuple_array(zipped_e_field, "R against E field", "R", "E_field")

    plot_tuple_array(zipped_log_R_log_E, "logR against logE field", "R", "E_field")

    # FIT ACROSS LOG R LOG V
    plot_tuple_array_fit(zipped_log_R_log_E, lin_func, 0.5, 1.5, "Fitted LogR, LogE plot", "logR", "logE")


    # save R_V_E array
    np.savetxt("RVE.txt", R_V_E_array)



main()
