import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

# Best to run with 50x50 and 3000 iterations. dx, dt = 1 is also okay

# phi0 = 0 is spinodel decomposition
# phi0 = +- 0.5 is droplet nucleation

# Convergence of algorithm

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to phi_zero plus some random noise.
def init_array(lattice_size, phi):
    # Array of phi value with random noise of stdev 1/100
    array = np.random.normal(phi,1/10, size = (lattice_size,lattice_size))
    return array


def chem_pot(array, lattice_size, a, k, dx):
    # apply mu calculation
    mu = -a*array + a*np.power(array,3)-(k/dx**2)*(np.roll(array,1,axis=0) + np.roll(array,-1,axis=0) + np.roll(array,1,axis=1) + np.roll(array,-1,axis=1) - 4*array)

    return mu

def chem_pot_iterations(array, lattice_size, a, k, dx):
    # slower method

    #time_start = time.time()

    # compute chemical potential across the array
    # create mu array
    mu = np.zeros((lattice_size, lattice_size), dtype=float)

    for i in range(len(array)):
        for j in range(len(array)):
            # Cardinal directions
            left = (i, j-1)
            right = (i, (j+1) % lattice_size)
            top = (i-1, j)
            bottom = ((i+1) % lattice_size, j)
            # apply mu calculation
            comp1 = -a*array[i,j] + a*array[i,j]**3
            comp2 = -(k/dx**2)*(array[bottom[0], bottom[1]] + array[top[0], top[1]] + array[right[0],right[1]] + array[left[0], left[1]] - 4*array[i,j])
            # apply to mu
            mu[i,j] = comp1 + comp2

    # print time
    #print("Time taken oldschool: {0:.6g}s".format(time.time()-time_start))
    return mu


def update_phi(array, lattice_size, m, dt, dx, mu):
    # returns a new array based on the previous array and mu array

    # complete the discretised calculation
    newarray = array + ((m*dt)/dx**2) * ((np.roll(mu,1,axis=0)) + (np.roll(mu,-1,axis=0)) + (np.roll(mu,1,axis=1)) + (np.roll(mu,-1,axis=1)) - 4*mu)
    # return array
    return newarray

def free_energy(array, a, k, dx):

    # calculate free energy density
    gradphisq = 1/(dx**2)*(np.power((np.roll(array,1,axis=0) - array),2) + np.power((np.roll(array,1, axis=1)-array),2))
    f = -(a/2) * np.power(array,2) + (a/4) * np.power(array,4) + (k/2) * gradphisq
    return f

def plot_free(time_list, free_list):
    # clear plot
    plt.clf()
    # plot the free energy

    plt.scatter(time_list, free_list, s = 0.01)
    plt.plot(time_list, free_list)
    plt.show()


# main iterator here
def main(lattice_size, iterations, phi, dx, dt):
    # initialise array
    array = init_array(lattice_size, phi)

    # set inbuilt parameters
    a = 0.1
    m = 0.1
    k = 0.1
    # plotting parameter
    n = 100

    # set lists up
    mu_list = []
    free_list = []
    time_list = []

    for q in range(iterations):
        # find mu array
        #mu_list = chem_pot_iterations(array, lattice_size, a, k, dx)

        mu_list = chem_pot(array, lattice_size, a, k, dx)

        # comput phi(n+1)
        newphi = update_phi(array, lattice_size, m, dt, dx, mu_list)

        # plot every nth, after the first
        if (q%n==0) and q>101:
            print("{}/{}".format(q,iterations))
            #print("Average of mu: {:.4f}\nTotal phi: {:.4f}".format(np.mean(mu_list), np.sum(array)))
            print()
            plt.clf()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.colorbar(im)
            plt.pause(0.00005)
            # append to time list
            time_list.append(q)

            # also find free energy from the array
            free_list.append(np.sum(free_energy(array, a, k, dx)))

            #print(sum(array))

        # then update
        array = np.copy(newphi)

    # save data
    np.savetxt("time_free.csv", np.array([time_list, free_list]).T, delimiter=",")
    # plot free energy against iterations
    plot_free(time_list, free_list)


# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 6:
        # run code, force as integers
        main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    else:
        print("\nScript takes exactly 5 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nLattice Size (square)\n\nIterations\n\nphi0\n\ndx\n\ndt")
