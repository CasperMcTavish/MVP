import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time
import scipy.optimize


# FILE THAT WILL CENTRALISE ALL THE ANALYSIS MODULES

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


# calculate the e-field
def e_field(array, dx):

    # Calculate e field across each element
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    e_zfield = -(1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))

    # find magnitude of e field at each element
    e_field = np.sqrt(np.square(e_xfield) + np.square(e_yfield) + np.square(e_zfield))
    return (e_xfield,e_yfield,e_zfield, e_field)




# linear function for fitting
def lin_func(x,m,b):
    return m*np.array(x) + b


def create_radial_array_point(lattice_size, midpoint):
    # create 3D radius array of equal size to our phi array, with each index being distance from midpoint.
    # BUT since this is now a wire, set it up to be for each layer
    rad_array = np.zeros((lattice_size,lattice_size,lattice_size), dtype= float)
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # create radial array centred around [0,0], along Z
                rad_array[i,j,k] = np.sqrt((i-midpoint)**2 + (j-midpoint)**2 + (k-midpoint)**2)

    return rad_array

def create_radial_array_wire(lattice_size, midpoint):
    # create 3D radius array of equal size to our phi array, with each index being distance from midpoint.
    # BUT since this is now a wire, set it up to be for each layer
    rad_array = np.zeros((lattice_size,lattice_size,lattice_size), dtype= float)
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # create radial array centred around [0,0], along Z
                rad_array[i,j,k] = np.sqrt((j-midpoint)**2 + (k-midpoint)**2)

    return rad_array

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



# Fit across the tuple array
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
    plt.plot(x_val[::10], lin_func(x_val[::10], *popt_x), 'g--', label='fit: m=%5.3f, b=%5.3f, ' % tuple(popt_x))
    save_name = name + "fitted.png"
    plt.scatter(x_val, y_val, s=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.savefig(save_name)
    plt.show()
