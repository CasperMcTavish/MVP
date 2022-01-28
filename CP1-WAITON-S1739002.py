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

# Calculate kawasaki energy at certain neareast neighbours
# given two sets of coordinates, check that they're not nearest neighbours/same values
def energy_calc_kawasaki(array, J, lattice_size, flip1, flip2):
    # check each nearest neighbour position. the kawasaki flips are nearest neighbours, -4 from result
    if (abs(flip1[0] - flip2[0])%lattice_size + abs(flip1[1] - flip2[1])%lattice_size == 1):
        #Calculate energy -4
        delta_E = energy_calc(array, J, lattice_size, flip1[0], flip1[1]) + energy_calc(array, J, lattice_size, flip2[0], flip2[1]) +4
        return delta_E
    # normal case
    else:
    # add up the two new energy changes for the nearest neighbours
        delta_E = energy_calc(array, J, lattice_size, flip1[0], flip1[1]) + energy_calc(array, J, lattice_size, flip2[0], flip2[1]) 
        #print("Not Neighbour")
        return delta_E

    
    

# Do glauber flip at random, assuming square matrix, give back index that changed
def glauber_flip(array):
    
    i_flip = np.random.randint(0, len(array))
    j_flip = np.random.randint(0, len(array))
    #print("flipping spin at " + str(i_flip) + ", " + str(j_flip))
    array_new = np.copy(array)
    array_new[i_flip,j_flip] = array_new[i_flip, j_flip] * -1
    return (array_new, i_flip, j_flip)


# Flip two spins at random
# Ensure they aren't the same ones
def kawasaki_flip(array):
    
    # Initialising arrays
    (i_flip1, j_flip1, i_flip2, j_flip2) = [0, 0, 0, 0]
    array1 = np.copy(array)
    array2 = np.copy(array)
    # A check to make sure you're not just flipping the same spins
    while (i_flip1 == i_flip2 and j_flip1 == j_flip2):
 
        # produce two random positions
        i_flip1 = np.random.randint(0, len(array))
        j_flip1 = np.random.randint(0, len(array))
        
        i_flip2 = np.random.randint(0, len(array)) 
        j_flip2 = np.random.randint(0, len(array)) 
        
        # Flip
        array2[i_flip1, j_flip1] = array1[i_flip2, j_flip2]
        array2[i_flip2, j_flip2] = array1[i_flip1, j_flip1]

    # Then return array, and flipped coordinates
    return  (array2, [i_flip1, j_flip1], [i_flip2, j_flip2])

# Calculate delta E and then 
def delta_E_kawasaki(array1, array2, lattice_size, T, flip1, flip2):
    # take the two arrays and calculate the new delta_E
    delta_E = energy_calc_kawasaki(array2, 1, lattice_size, flip1, flip2)
    # Check delta E. If deltaE <= 0. Keep new array
    # If deltaE > 0, use boltzmann probability to determine flip

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




# Give it the flipped matrix and the first matrix, return the correct matrix based on delta_E value
def delta_E_glauber(array1, array2, lattice_size, T, i, j):
    
    # Calculate delta E
    delta_E = energy_calc(array2, 1, lattice_size, i, j)
    
    # check for matrix return
    
    # Check delta E. If deltaE <= 0. Keep new array
    # If deltaE > 0, use boltzmann probability to determine flip

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






def iteration_glauber(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    

    for i in range(iterations):
        # find new matrix, 
        for _ in range(lattice_size**2):
            # glauber flip and calculate delta_E then update spins
            (array2, i, j) = glauber_flip(array)
            array = delta_E_glauber(array, array2, lattice_size, T, i, j)
     # plot every 10th
        if (i%1==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.0001)


def iteration_kawasaki(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    for i in range(iterations):
        # find new matrix within n^2 loop
        for _ in range(lattice_size**2):
            # kawasaki flip
            (array2, flip1, flip2) = kawasaki_flip(array)
            # update array
            array = delta_E_kawasaki(array, array2, lattice_size, T, flip1, flip2)
        # plot every 10th
        if (i%10==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.0001)





iteration_kawasaki(10000, 50, 1)



