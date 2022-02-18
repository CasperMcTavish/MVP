import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy



#################################################
# GAME OF LIFE BASIC FUNCTIONALITY
# INCLUDES ANIMATION (TOGGLE-ABLE) AND SIMULATION
#################################################

# Create a grid of spins i-rows,j-columns, limited to 1 and 0.
# True/1 = Alive, False/0 = Dead
def gol_array(lattice_size):
    # Setting up boolean array of zeros and ones
    # As python takes boolean True and False as 0 and 1, they can be used in mathematical operations (useful!) while being clamped to [0,1] range
    array = np.zeros((lattice_size, lattice_size), dtype=bool)

    # create loop that goes through each row, flips a coin and makes that value negative dependent on this
    for i in range(lattice_size):
        for j in range(lattice_size):
            coinflip = round(random.uniform(0,1))
            if (coinflip == 1):
                array[i,j] = True

    return array


def gol_nn_check(array, lattice_size, i, j):
    # Because we only need to consider number of nearest neighbours, can just sum the array at the end.

    # Only need to consider boundaries when adding, as array index should immediately loop back on itself if it becomes negative

    # Cardinal directions
    left = (i, j-1)
    right = (i, (j+1) % lattice_size)
    top = (i-1, j)
    bottom = ((i+1) % lattice_size, j)

    # Diagonal directions
    top_left = (i-1, j-1)
    top_right = (i-1, (j+1) % lattice_size)
    bottom_left = ((i+1) % lattice_size , j-1)
    bottom_right = ((i+1) % lattice_size , (j+1) % lattice_size)

    # Return sum of nearest neighbours.
    nn_sum = sum([
               array[left[0], left[1]],
               array[right[0], right[1]],
               array[top[0], top[1]],
               array[bottom[0], bottom[1]],
               array[top_left[0], top_left[1]],
               array[top_right[0], top_right[1]],
               array[bottom_left[0], bottom_left[1]],
               array[bottom_right[0], bottom_right[1]]
               ])

    return nn_sum


# Updated the array based on the game of life rules.
def array_update(array, lattice_size):

    # Alive cell checks:
    #       < 2 alive neighbours = DEATH
    #       2 or 3 alive neighbours = LIFE
    #       > 3 alive neighbours = DEATH

    # Dead cell checks:
    #       3 alive neighbours = LIFE

    # Copy to new array (slow but preserves old array format). You need to remember previous format to ensure iteration works
    new_array = np.copy(array)

    # scan across entire array
    for i in range(lattice_size):
        for j in range(lattice_size):

            # Collect nearest neighbours
            nn_count = gol_nn_check(array, lattice_size, i, j)

            # IF DEAD AND 3 ALIVE NEIGHBOURS
            if ((array[i, j] == False) and (nn_count == 3)):
                # BRING TO LIFE
                new_array[i, j] = True
            # IF ALIVE
            elif ((array[i, j] == True)):
                # AND 2 OR 3 NEIGHBOURS
                if (nn_count == 2 or nn_count == 3):
                    # STAY ALIVE
                    continue;
                # OTHERWISE DEATH
                else:
                    new_array[i, j] = False


    return new_array

# Create an array with a glider right in the middle if there is space
def glider_array(lattice_size):
    # If there is space, make glider
    if (lattice_size > 4):
        array = np.zeros((lattice_size, lattice_size), dtype=bool)
        # Choose central point and create glider around it
        m = int(lattice_size//2)

        array[m-1, m] = True
        array[m, m+1] = True
        array[m+1, m+1] = True
        array[m+1, m] = True

        array[m+1, m-1] = True
    return array


def main(lattice_size, sim_type, iterations):

    # Define array
    array = gol_array(lattice_size)
    #array = glider_array(lattice_size)
    # Iterate over array
    for i in range(iterations):

        # plot every nth
        n = 1
        if (i%n==0):
            plt.cla()
            im=plt.imshow(array, animated=True)
            plt.draw()
            plt.pause(0.01)


        # Update array
        array = array_update(array, lattice_size)





main(50, 0, 100)
# Testing apparatus
#array = np.zeros((5, 5), dtype=bool)
#array[3, 3] = True
#array[4, 4] = True
#array[2, 2] = True
#print(array)
#nn_check = gol_nn_check(array, 5, 3, 3)
#print(nn_check)







# Dead code, ideas that didnt make it:
'''
def gol_nn_check(array, lattice_size, i, j):
    # Because we only need to consider number of nearest neighbours, can just sum the array at the end.

    # Only need to consider boundaries when adding, as array index should immediately loop back on itself if it becomes negative

    # Create new array that represents nearest neighbours of each XY

    nn_array = np.zeros((lattice_size, lattice_size))

    for i in range(lattice_size):
        for j in range(lattice_size):
            # Cardinal directions
            left = (i, j-1)
            right = (i, (j+1) % lattice_size)
            top = (i-1, j)
            bottom = ((i+1) % lattice_size, j)


            # Diagonal directions
            top_left = (i-1, j-1)
            top_right = (i-1, (j+1) % lattice_size)
            bottom_left = ((i+1) % lattice_size , j-1)
            bottom_right = ((i+1) % lattice_size , (j+1) % lattice_size)


            # Return sum of nearest neighbours.
            nn_sum = sum([
                       array[left[0], left[1]],
                       array[right[0], right[1]],
                       array[top[0], top[1]],
                       array[bottom[0], bottom[1]],
                       array[top_left[0], top_left[1]],
                       array[top_right[0], top_right[1]],
                       array[bottom_left[0], bottom_left[1]],
                       array[bottom_right[0], bottom_right[1]]
                       ])

            # Fill in array
            nn_array[i,j] = nn_sum

    return nn_array
'''
