import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time
import scipy.optimize

# Plot the massive tuples
def plot_tuple_array(tuple_array, name, xlabel, ylabel):
    # Splits array of tuples, plots it and then saves it

    x_val = [x[0] for x in tuple_array]
    y_val = [x[1] for x in tuple_array]

    save_name = name + ".png"
    plt.scatter(x_val, y_val, s=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(save_name)
    plt.show()


def plot_tuple_array_fit(tuple_array, func, xlower_lim, xupper_lim, name, xlabel, ylabel):

    # Take slice of the array for fitting
    cut_arrayx = []
    cut_arrayy = []
    for i in range(len(tuple_array)):
        if (tuple_array[i][0] > xlower_lim) and (tuple_array[i][0] < xupper_lim):
            cut_arrayx.append(float(tuple_array[i][0]))
            cut_arrayy.append(float(tuple_array[i][1]))


    # collect as x and y (for scatter plot)
    x_val = [x[0] for x in tuple_array]
    y_val = [x[1] for x in tuple_array]


    # fit
    popt_x, pcov_x = scipy.optimize.curve_fit(func, cut_arrayx,cut_arrayy)
    print("Standard error:")
    print(np.sqrt(np.diag(pcov_x)))

    # plot
    plt.plot(cut_arrayx, lin_func(cut_arrayx, *popt_x), 'g--', label='fit: m=%5.3f, b=%5.3f, ' % tuple(popt_x))
    save_name = name + "fitted.png"
    plt.scatter(x_val, y_val, s=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.savefig(save_name)
    plt.show()

# linear function for fitting
def lin_func(x,m,b):
    return m*np.array(x) + b

def e_field(array, dx):

    # Calculate e field across each element
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    e_zfield = -(1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))

    # find magnitude of e field at each element
    e_field = np.sqrt(np.square(e_xfield) + np.square(e_yfield) + np.square(e_zfield))
    return (e_xfield,e_yfield,e_zfield, e_field)

def b_field(array, dx):
    # produce magnetic field

    # dz = 0 so ignore those components, Ax = Ay = 0
    dyAz = (1/(2*dx)) * (np.roll(array,1,axis=2) - np.roll(array,-1,axis=2))
    dzAy = 0
    dzAx = 0
    dxAz = (1/(2*dx)) * (np.roll(array,1,axis=1) - np.roll(array,-1,axis=1))
    dxAy = 0
    dyAx = 0
    b_xfield = dyAz - dzAy
    b_yfield = dzAx - dxAz
    b_zfield = dxAy - dyAx

    b_field = np.sqrt(np.square(b_xfield) + np.square(b_yfield) + np.square(b_zfield))
    return (b_xfield, b_yfield, b_zfield, b_field)



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
    rad_array = np.zeros((lattice_size,lattice_size,lattice_size), dtype= float)
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # create radial array centred around [0,0], along Z
                rad_array[i,j,k] = np.sqrt((j-midpoint)**2 + (k-midpoint)**2)


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
    plot_tuple_array_fit(zipped_logR_linA, lin_func, 0.5, 2, "Fitted LogR linA plot", "logR", "A")

                # LOG R -> LOG B
    plot_tuple_array(zipped_logR_logB, "Log R against Log B", "LogR", "LogB")

    plot_tuple_array_fit(zipped_logR_logB, lin_func, 0.5, 2, "Fitted LogR LogB plot", "logR", "A")


    # save R_V_E array
    np.savetxt("RAB.txt", R_A_B_array)



main()
