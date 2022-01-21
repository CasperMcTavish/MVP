import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random

# Create a grid of spins i-rows,j-columns, limited to 1 and -1
def spin_array(rows, cols):
    array = np.ones((rows, cols))
    
    # create loop that goes through each row, flips a coin and makes that value negative dependent on this
    for i in range(cols):
        for j in range(rows):
            coinflip = round(random.uniform(0,1))
            if (coinflip == 1):
                array[i,j] = array[i,j]*-1

    return array

# Find nearest neighbours based on position, array itself and lattice size
# WARNING! ASSUMES A SQUARE LATTICE!
def nearest_neighbours(array, lattice_size, i, j ):
    # find the equivalent index positions for our array position, based on lattice size (will loop around the back if too large)
    left = (i, j-1)
    right = (i, (j+1) % lattice_size)
    top = (i-1, j)
    bottom = ((i+1) % lattice_size, j)

    # return values from array for these positions. Return as integers
    nn_array = [array[left[0], left[1]],
               array[right[0], right[1]],
               array[top[0], top[1]],
               array[bottom[0], bottom[1]]]
    nn_array = [int(item) for item in nn_array]

    return nn_array



# Calculate energy of position with relation to its nearest neighbouts
def energy_calc(array, J, lattice_size, i, j):

    # take position, collect its nearest neighbours and multiply it by each of those and sum. Then multiply by 2
    energy_sum = 2 * array[i,j] * sum(nearest_neighbours(array, lattice_size, i, j))

    energy = - J * energy_sum

    return energy


# Do glauber flip at random, assuming square matrix, give back index that changed
def glauber_flip(array):
    
    i_flip = np.random.randint(0, len(array))
    j_flip = np.random.randint(0, len(array))
    #print("flipping spin at " + str(i_flip) + ", " + str(j_flip))
    array_new = np.copy(array)
    array_new[i_flip,j_flip] = array_new[i_flip, j_flip] * -1
    return (array_new, i_flip, j_flip)

# calculate delta_E and return new matrix
def delta_E(array, lattice_size, T):
    # take the two arrays
    array1 = np.copy(array)
    (array2, i, j) = glauber_flip(array1)
    
    #energy1 = energy_calc(array1, 1, lattice_size, i, j)
    delta_E = energy_calc(array2, 1, lattice_size, i, j)
    
    # check for matrix return
    if (delta_E <= 0):
        return array2
    else:
        # return with exponential probability
        p = np.exp(-(1/T)*delta_E)
        r = random.uniform(0,1)
        if (r <= p):
            return array2
        else:
            return array1


# Check delta E. If deltaE <= 0. Keep new array
# If deltaE > 0, use boltzmann probability to determine flip

def iteration(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    

    for i in range(iterations):
        # find new matrix, 
        for _ in range(lattice_size**2):
            array = delta_E(array, lattice_size, T)
     # plot every 10th
        if (i%1==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.0001)



iteration(100, 10, 100)



