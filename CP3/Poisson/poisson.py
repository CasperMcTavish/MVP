import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to zero except central point source
def init_array(lattice_size, gam):

    # set array of zeros except for central point, this is rho array
    array = np.random.normal(gam,1/10, size = (lattice_size,lattice_size,lattice_size))
    rho = np.zeros((lattice_size,lattice_size,lattice_size), dtype=float)
    # set central point to 1
    mid_point = int(lattice_size/2)
    rho[mid_point,mid_point,mid_point] = 1


    return array, rho

# QUIVER MATPLOTLIB

def update_gam(array, lattice_size, dx, rho):
    # update the gamma values

    # complete the discretised calculation
    newarray = 1/6 * (np.roll(array,1,axis=0) + np.roll(array,-1,axis=0) + np.roll(array,1,axis=1) + np.roll(array,-1,axis=1) + np.roll(array,1,axis=2) + np.roll(array,-1,axis=2) + rho)

    # set the edges to zero
    newarray[:,[0,-1],:] = newarray[[0,-1]] = newarray[:,:,[0,-1]] = 0
    return newarray

def e_field(array, dx):
    # define electric field
    e_field = -1/(2*dx) * ((np.roll(array,1,axis=0) - array) +  ((np.roll(array,1,axis=1) - array)) + (np.roll(array,1,axis=2) - array))
    return e_field

def checker(array, newarray):
    # check the difference between the phi values of each array
    value = array - newarray
    return value



def iterator(lattice_size, dx):

    # initialise array
    array, rho = init_array(lattice_size, 0.001)

    time_list = []

    i = 0
    # iterate 3D array?
    while(True):

        # update array
        newarray = update_gam(array, lattice_size, dx, rho)

        # check if the value between these two arrays is zero, if so then break
        # if not, then continue
        value = checker(array, newarray)

        if (i%100==0):
            # print the sum of the differences
            print("Convergence factor: {0:.6f}".format(np.sum(value)))

        if (abs(np.sum(value)) < 0.01):
            array = newarray
            break;
        else:
            array = newarray
            i += 1


    # once out of the loop, find the E field for the space
    E_F = e_field(array, dx)
    # normalise e field and potential
    E_Fnorm = E_F/(np.max(E_F))
    arraynorm = array/(np.max(array))

    # print the potential here
    #plotting central slice
    imsh = plt.imshow(arraynorm[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()

    # plot e-field QUIVER
    x = y = np.linspace(0,99,100)
    X,Y = np.meshgrid(x,y)
    dx,dy = np.gradient(E_F[int(lattice_size/2)])
    plt.quiver(X,Y,dx,dy)
    plt.show()

    # SAVE EFIELD AND GRAD FIELD


# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 3:
        # run code, force as ints/floats
        iterator(int(sys.argv[1]), float(sys.argv[2]))
    else:
        print("\nScript takes exactly 2 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nLattice Size (square)\n\ndx")
