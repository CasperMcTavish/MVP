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
sys.path.insert(0,"../")
from analysis_scripts import *


def main():

    # load phi array
    array = np.load("phiarray.npy")

    # define lattice size
    lattice_size = len(array)
    print("lattice size: {}".format(lattice_size))

    # collect E field (dx hard coded to 1 here)
    B_Fx, B_Fy, B_Fz, B_F = b_field(array, 1)

    # Normalise e field
    norm_B_F = B_F/np.max(B_F)
    # normalise main array
    norm_array = array/np.max(array)

    # Create meshgrid of x and y for plotting purposes
    x = y = np.linspace(0,len(array)-1,len(array))
    X,Y = np.meshgrid(x,y)

    # Show electric across slice of array
    imsh = plt.imshow(B_F[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Magnetic field at centre of array")
    plt.show()


    # split of lattice for use later
    split = int(lattice_size/2)

    # produce array that contains slice of format: (X,Y, VECTOR_POT, E_X,E_Y)
    full_array_slice = []
    for i in range(split):
        for j in range(split):
            full_array_slice.append((i,j,array[split][i][j],B_Fx[split][i][j],B_Fy[split][i][j]))
    # save array (required for checkpoint)
    np.savetxt("XYPOTBXBY.txt", full_array_slice)


    # sadly gradient trick doesnt work here, have to use x and y magnetic components explicitly.
    b_fy = B_Fy[split]/np.sqrt(np.square(B_Fy[split])+np.square(B_Fx[split]))
    b_fx = B_Fx[split]/np.sqrt(np.square(B_Fy[split])+np.square(B_Fx[split]))

    # plot quivers of e field gradient (step skips positions to make lines more visible)
    step = 2

    plt.quiver(X[::step,::step],Y[::step,::step], b_fy[::step,::step],b_fx[::step,::step])
    plt.title("Magnetic field from central slice")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("bfield.png")
    plt.show()


    # plot potential across sclice of array
    imsh = plt.imshow(array[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()


    # Find midpoint
    midpoint = int(lattice_size/2)

    # create 3D radius array of equal size to our phi array, with each index being distance from midpoint.
    # BUT since this is now a wire, set it up to be for each layer
    rad_array = create_radial_array_wire(lattice_size, midpoint)

    # sort through both arrays to create tuples that can be plotted on scatter diagrams
    # this takes a while! lots of data points


    zipped_linR_linA = []
    zipped_logR_linA = []
    zipped_logR_logB = []


    R_A_B_array = []
    # This, but we want Z to be ignored when producing array, so removing i range
    #for i in range(lattice_size):
    for j in range(lattice_size):
        for k in range(lattice_size):

            # make main array
            R_A_B_array.append((rad_array[split,j,k], norm_array[split,j,k], norm_B_F[split,j,k]))

            # lin R, lin A
            zipped_linR_linA.append((rad_array[split,j,k],norm_array[split,j,k]))

            # log R, linear A
            zipped_logR_linA.append((np.log(rad_array[split,j,k]),norm_array[split,j,k]))

            # log R, log B
            zipped_logR_logB.append((np.log(rad_array[split,j,k]),np.log(norm_B_F[split,j,k])))
            '''
            # add tuples
            zipped_potential.append((rad_array[i,j,k],array[i,j,k]))
            zipped_norm_potential.append((rad_array[i,j,k], norm_array[i,j,k]))
            zipped_log_norm_potential.append((rad_array[i,j,k], np.log(norm_array[i,j,k])))
            zipped_logR_logV.append((np.log(rad_array[i,j,k]),np.log(array[i,j,k])))
            zipped_log_R_norm_potential.append((np.log(rad_array[i,j,k]), norm_array[i,j,k]))
            zipped_e_field.append((rad_array[i,j,k], B_F[i,j,k]))
            zipped_log_R_log_E.append((np.log(rad_array[i,j,k]),np.log(B_F[i,j,k])))
            '''

                # LIN R -> LIN A
    plot_tuple_array(zipped_linR_linA, "R against Vector Potential", "R", "A")


                # LOG R -> LIN A
    plot_tuple_array(zipped_logR_linA, "Log R against Vector Potential", "LogR", "A")

    # FIT ACROSS RANGE THAT WILL GIVE 1/R GRADIENT
    plot_tuple_array_fit(zipped_logR_linA, lin_func, 1.5, 2, "Fitted LogR linA plot", "logR", "A")

                # LOG R -> LOG B
    plot_tuple_array(zipped_logR_logB, "Log R against Log B", "LogR", "LogB")

    plot_tuple_array_fit(zipped_logR_logB, lin_func, 1, 3, "Fitted LogR LogB plot", "logR", "A")


    # save R_V_E array
    np.savetxt("RAB.txt", R_A_B_array)



main()
