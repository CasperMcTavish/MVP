import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

def plot_quiver2D(x,y,u,v):
    plt.quiver(x,y,u,v)

# JACOBIAN, BUT 3D
# THIS IS THE POINT CHARGE CASE

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

# Seidel updater

def update_gam(array, lattice_size, rho, w):

    for i in range(len(array)):
        for j in range(len(array)):
            for k in range(len(array)):
                # set edges to 0
                if (i == 0) or (j == 0) or (k == 0) or (i==lattice_size-1) or (j==lattice_size-1) or (k==lattice_size-1):
                    array[i,j,k] = 0
                else:
                    # APPLY SOR in here as well
                    array[i,j,k] = w*(1/6 * (array[i+1,j,k] + array[i-1,j,k] + array[i,j+1,k] + array[i,j-1,k] + array[i,j,k+1] + array[i,j,k-1] + rho[i,j,k])) + (1-w)*array[i,j,k]

    return array


def e_field(array, dx):
    # define electric field
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    e_zfield = -(1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))
    #print(np.sum(e_xfield))
    #print(np.sum(e_yfield))
    #print(np.sum(e_zfield))
    e_field = np.sqrt(np.square(e_xfield) + np.square(e_yfield) + np.square(e_zfield))
    return (e_xfield,e_yfield,e_zfield, e_field)

def checker(array, newarray):
    # check the difference between the phi values of each array
    value = np.sum(np.abs(array - newarray))
    return value


def iterator(lattice_size, dx, accuracy):


    # create time list
    time_list = []
    #create SOR list from 1.5 -> 2
    SOR_list = np.linspace(1.5,1.9,50)
    eq_list = []

    for s in range(len(SOR_list)):
        # initialise array every time
        array, rho = init_array(lattice_size, 0.001)



        w = SOR_list[s]
        print("W value: {}".format(w))
        i = 0
        # iterate 3D array?
        while(True):

            # update array
            savearray = np.copy(array)
            array = update_gam(array, lattice_size, rho, w)

            # check if the value between these two arrays is zero, if so then break
            # if not, then continue
            value = checker(savearray, array)

            if (i%100==0):
                # print the sum of the differences
                print("Convergence factor: {0:.6f}".format(value))

            # when convergence is low enough, break out the loop
            # standard accuracy -> 0.01 or 0.001
            if (value < accuracy):
                # add the value of i to a new list
                print("Best value found! Moving on...")
                eq_list.append(i)
                break;
            else:
                #array = newarray
                i += 1

    # plot SOR values against eq_list to see best equilibrium value for overrelaxations
    plt.scatter(SOR_list, eq_list)
    plt.plot(SOR_list, eq_list)
    plt.title("W against iteration values")
    plt.xlabel("w")
    plt.ylabel("Iterations")
    plt.savefig("w_against_iterations.png")
    plt.show()



    # Rerun the SOR for the best w value and save it
    w = SOR_list[np.argmin(eq_list)]
    i = 0
    # iterate 3D array?
    while(True):

        # update array
        savearray = np.copy(array)
        array = update_gam(array, lattice_size, rho, w)

        # check if the value between these two arrays is zero, if so then break
        # if not, then continue
        value = checker(savearray, array)

        if (i%100==0):
            # print the sum of the differences
            print("Convergence factor: {0:.6f}".format(value))

        # when convergence is low enough, break out the loop
        # standard accuracy -> 0.01 or 0.001
        if (value < accuracy):
            # add the value of i to a new list
            break;
        else:
            #array = newarray
            i += 1


    # save xy of 3D array
    np.save("phiarray.npy", array)


# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 4:
        # run code, force as ints/floats
        iterator(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    else:
        print("\nScript takes exactly 3 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nLattice Size (square)\n\ndx\n\nAccuracy of discretisation")
