import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys



#######################################################################################
# CONTAINS THE BASICS TO GET THE FULL 7 MARKS FROM THE CHECKPOINT (run glauber/kawasaki dynamics at different T, lattice size, etc)
#######################################################################################


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
# WARNING ASSUMES A SQUARE LATTICE!
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

def kawasaki(array, lattice_size, T):
    # Performs kawasaki flip and determines if array should be updated

    # Create two base flips
    flip1 = (0,0)
    flip2 = (0,0)
    # Create two flips, ensure they aren't the same coordinates, and dont have same spin
    while ((flip1 == flip2) or (array[flip1[0], flip1[1]] == array[flip2[0], flip2[1]])):
        flip1 = (np.random.randint(0, len(array)),np.random.randint(0, len(array)))
        flip2 = (np.random.randint(0, len(array)),np.random.randint(0, len(array)))

    # Flip positions, because they cant be same spin just multiply both points by -1
    array[flip1] *= -1
    array[flip2] *= -1

    # Now delta E calculation for new array

    # Check that they aren't nearest neighbours, if they are, apply energy calculation differently (with correction)
    if (abs(flip1[0] - flip2[0])%lattice_size + abs(flip1[1] - flip2[1])%lattice_size == 1):
        delta_E = energy_calc(array, 1, lattice_size, flip1[0], flip1[1]) + energy_calc(array, 1, lattice_size, flip2[0], flip2[1]) + 4
    # otherwise, calculate as expected (delta_E_1 + delta_E_2)
    else:
        delta_E = energy_calc(array, 1, lattice_size, flip1[0], flip1[1]) + energy_calc(array, 1, lattice_size, flip2[0], flip2[1])

    if (delta_E <= 0):
        # return flipped array
        return array
    # otherwise, return with exponential probability
    else:
        # return with exponential probability
        p = np.exp(-(1/T)*delta_E)
        r = random.uniform(0,1)
        if (r <= p):
            # return flipped array
            return array
        else:
            # flip array back
            array[flip1] *= -1
            array[flip2] *= -1
            return array

def glauber(array, lattice_size, T):
    # Performs glauber flip, and determines if array should be updated

    # take new coordinates within array to flip
    i_flip = np.random.randint(0, len(array))
    j_flip = np.random.randint(0, len(array))

    # Check if flip is beneficial, this is delta E for glauber

    # flip array
    array[i_flip,j_flip] *= -1
    # calculte delta E
    delta_E = energy_calc(array, 1, lattice_size, i_flip, j_flip)
    # so if delta E <= 0, flip

    if (delta_E <= 0):
        # return flipped array
        return array
    # otherwise, return with exponential probability
    else:
        # return with exponential probability
        p = np.exp(-(1/T)*delta_E)
        r = random.uniform(0,1)
        if (r <= p):
            # return flipped array
            return array
        else:
            # flip array back
            array[i_flip,j_flip] *= -1
            return array



def iteration_kawasaki(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    for i in range(iterations):
        # find new matrix within n^2 loop
        for _ in range(lattice_size**2):
            # run kawasaki function to update array
            array = kawasaki(array, lattice_size, T)
        # plot every 5th
        if (i%5==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.0001)


def iteration_glauber(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    for i in range(iterations):
        # find new matrix,
        for _ in range(lattice_size**2):
            # glauber flip and calculate delta_E then update spins
            array = glauber(array, lattice_size, T)
     # plot every 5th
        if (i%5==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.0001)


#iteration_kawasaki(1000, 50, 1)
#iteration_glauber(1000, 50, 1)


# MAIN FUNCTION
def run_code(model, lattice, T, iterations):
    # convert sys arguments from string to floats/ints
    model = int(model)
    lattice = int(lattice)
    T = float(T)
    iterations = int(iterations)
    # choose which model
    if (model == 0):
        print("Running Glauber...\nTemperature: {:.2f}\nLattice Size: {:.2f}\nSweeps: {:.2f}".format(T, lattice, iterations))
        iteration_glauber(iterations, lattice, T)
    else:
        print("Running Kawasaki...\nTemperature: {:.2f}\nLattice Size: {:.2f}\nSweeps: {:.2f}".format(T, lattice, iterations))
        iteration_kawasaki(iterations, lattice, T)


# Check to make sure enough arguments
if len(sys.argv) == 5:
    run_code(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
else:
    print("\nScript takes exactly 4 arguments, " + str(len(sys.argv)-1) + " were given")
    print("\nPlease input:\n\n DYNAMIC MODEL\n  0 - Glauber\n  1 - Kawasaki\n\n LATTICE SIZE\n\n TEMPERATURE\n\n ITERATIONS")
