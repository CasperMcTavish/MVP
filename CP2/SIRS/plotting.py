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

from gol_sim import read_file


#####################
# CODE FOR PLOTTING PHASE TRANSITIONS
#####################


# plotting for p1p3 altering
def p1p3_plotting():

    positions = read_file("Coordinate_lists")
    positions = [positions[i].rstrip().lstrip() for i in range(len(positions))]
    positions = [positions[i].split() for i in range(len(positions))]
    positions = (np.array(positions)).astype(float)
    data = read_file("Average_infection_lists")
    data = [data[i].rstrip().lstrip() for i in range(len(data))]
    data = (np.array(data)).astype(float)
    var = read_file("Variance_lists")
    var = [var[i].rstrip().lstrip() for i in range(len(var))]
    var = (np.array(var)).astype(float)

    # Nice to look at, not saved
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:,0], positions[:,1], data, c='r', marker='o')
    plt.show()

    # reshape param
    shape = int(np.sqrt(len(positions[:,0])))


    # Reshaping data to be acceptable
    POS0 = positions[:,0].reshape(shape,shape)
    POS1 = positions[:,1].reshape(shape,shape)
    DATA = data.reshape(shape,shape)
    VAR = var.reshape(shape,shape)

    # p1p3 contour inf plot
    cont = plt.contourf(POS0, POS1, DATA, cmap='RdBu')
    plt.colorbar(cont)
    plt.title("p1-p3 contour plot for average infected")
    plt.xlabel("p1")
    plt.ylabel("p3")
    plt.savefig("p1p3infcontourplt.png")
    plt.show()

    #p1p3 imshow inf plot
    imsh = plt.imshow(DATA, extent=[POS0.min(), POS0.max(), POS1.max(), POS1.min()])
    plt.colorbar(imsh)
    plt.xlabel("p3")
    plt.ylabel("p1")
    plt.title("p1-p3 imshow plot for average infected")
    plt.savefig("p1p3infimshowplt.png")
    plt.show()

    #p1p3 contour var plot
    contv = plt.contourf(POS0, POS1, VAR, cmap='RdBu')
    plt.colorbar(contv)
    plt.title("p1-p3 contour plot for variance")
    plt.xlabel("p1")
    plt.ylabel("p3")
    plt.savefig("p1p3varcontourplt.png")
    plt.show()

    #p1p3 imshow inf plot
    imshv = plt.imshow(VAR, extent=[POS0.min(), POS0.max(), POS1.max(), POS1.min()])
    plt.colorbar(imshv)
    plt.xlabel("p3")
    plt.ylabel("p1")
    plt.title("p1-p3 imshow plot for variance")
    plt.savefig("p1p3infimshowplt.png")
    plt.show()


# plot p1 variation plots
def p1var_plotting():
    # read in relevant data

    # coords
    positions = read_file("Coordinate_lists_p1")
    positions = [positions[i].rstrip().lstrip() for i in range(len(positions))]
    positions = (np.array(positions)).astype(float)

    # variance
    var = read_file("Variance_lists_p1")
    var = [var[i].rstrip().lstrip() for i in range(len(var))]
    var = (np.array(var)).astype(float)

    # error
    err = read_file("Variance_error_lists_p1")
    err = [err[i].rstrip().lstrip() for i in range(len(err))]
    err = (np.array(err)).astype(float)

    #plot
    plt.errorbar(positions, var, yerr = err, fmt = 'o')
    plt.errorbar(positions, var, yerr = err)
    plt.xlabel("p1 values")
    plt.ylabel("Variance in infection rate")
    plt.title("p1 variance with errors")
    plt.savefig("p1_variance_error.png")
    plt.show()

# start the code up with these
p1p3_plotting()
#p1var_plotting()
