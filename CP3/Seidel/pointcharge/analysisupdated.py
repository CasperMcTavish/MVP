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


def main():

    # load phi array
    array = np.load("phiarray.npy")

    # define lattice size
    lattice_size = len(array)
    print("lattice size: {}".format(lattice_size))

    # collect E field (dx hard coded to 1 here)
    E_Fx, E_Fy, E_Fz, E_F = e_field(array, 1)

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
    step = 5

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

    # create 3D radius array of equal size to our phi array, with each index being distance from midpoint.
    rad_array = np.zeros((lattice_size,lattice_size,lattice_size), dtype= float)
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # create radial array centred around [0,0]
                rad_array[i,j,k] = np.sqrt((i-midpoint)**2 + (j-midpoint)**2 + (k-midpoint)**2)



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


    # plot normalised tuple array
    ##plot_tuple_array(zipped_norm_potential, "R against normalised Potential", "R", "Potential")

    # plot normalised log tuple array
    ##plot_tuple_array(zipped_log_norm_potential, "R against log Potential", "R", "Log Potential")


    # plot normalised tuple with log R
    #plot_tuple_array(zipped_log_R_norm_potential, "LogR against Normalised Potential", "logR", "Potential")


    plot_tuple_array(zipped_logR_logV, "Log R against Potential", "LogR", "LogV")

    # FIT ACROSS RANGE THAT WILL GIVE 1/R GRADIENT, about R= 1.2 -> 2.2
    plot_tuple_array_fit(zipped_logR_logV, lin_func, 0.5, 1.5, "Fitted LogR LogV plot", "logR", "logV")





    # Plotting E field contents
    plot_tuple_array(zipped_e_field, "R against E field", "R", "E_field")

    plot_tuple_array(zipped_log_R_log_E, "logR against logE field", "R", "E_field")

    # FIT ACROSS LOG R LOG V
    plot_tuple_array_fit(zipped_log_R_log_E, lin_func, 0.5, 1.5, "Fitted LogR, LogE plot", "logR", "logE")


    # save R_V_E array
    np.savetxt("RVE.txt", R_V_E_array)



main()
