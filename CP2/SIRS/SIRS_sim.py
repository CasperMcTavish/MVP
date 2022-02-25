import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import matplotlib as mpl

###################
# PAREMETERS FOR CERTAIN CONDITIONS
###################

#### ABSORBING STATE

## NORMAL
# p1 -> 0.7, p2 -> 0.7, p3 -> 0.1
# lattice_size -> 50


#### DYNAMIC EQ

## NORMAL
# p1 -> 0.7, p2 -> 0.7, p3 -> 0.7
# lattice_size -> 50


#### CYCLIC WAVES

## BIG, SLOW BUT CORRECT
# p1 -> 0.8, p2 -> 0.1, p3 -> 0.01
# lattice_size -> 100

## SMALL, BIT NOISY BUT ALSO CORRECT
# p1 -> 0.7, p2 -> 0.175, p3 -> 0.025
# lattice_size -> 50


######################
# ARRAY CREATION
######################
# LAYOUT OF DIFFERENT WAYS TO CREATE INITIAL ARRAYS

####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, limited to 0, 1 and 2
# 0 = Susceptible, 1 = Infected, 2 = Recovered
def SIRS_array(lattice_size):
    # Setting up boolean array of zeros and ones
    # As python takes boolean True and False as 0 and 1, they can be used in mathematical operations (useful!) while being clamped to [0,1] range
    array = np.zeros((lattice_size, lattice_size), dtype=int)

    # create loop that goes through each row, flips a coin and makes that value negative dependent on this
    for i in range(lattice_size):
        for j in range(lattice_size):
            # set the array values randomly to 0, 1 or 2
            randomval = round(random.uniform(0,2))
            array[i,j] = int(randomval)

    return array

######################
# LOGIC
######################

def roll(probability):
    return random.random() < probability

# Search for infected nearest neighbours, returns True if nearest neighbour is infected
# WARNING ASSUMES A SQUARE LATTICE!
def nearest_neighbours(array, lattice_size, i, j):
    # find the equivalent index positions for our array position, based on lattice size (will loop around the back if too large)
    left = (i, j-1)
    right = (i, (j+1) % lattice_size)
    top = (i-1, j)
    bottom = ((i+1) % lattice_size, j)

    # return values from array for these positions.
    nn_array = [array[left[0], left[1]],
               array[right[0], right[1]],
               array[top[0], top[1]],
               array[bottom[0], bottom[1]]]

    # If infected, pass back true, otherwise false.
    if 1 in nn_array:
        infection=True
    else:
        infection=False

    return infection

# UPDATER
# Updated the array based on the SIRS rules.
def array_update(array, lattice_size, p1, p2, p3):

    # S -> I reliant on p1 if one NN is infected, otherwise unchanged
    # I -> R reliant on p2
    # R -> S reliant on p3

    # Copy to new array (slow but preserves old array format). You need to remember previous format to ensure iteration works
    #new_array = np.copy(array)

    # take new coordinates within array to flip
    i = np.random.randint(0, len(array))
    j = np.random.randint(0, len(array))

    # Update state of array position

    # S
    if array[i,j] == 0:
        # check nn, if one is infected
        if nearest_neighbours(array, lattice_size, i, j):
            # Convert based on probability p1
            if roll(p1):
                array[i,j] = 1
    # I, advance to R based on p2
    elif array[i,j] == 1:
        if roll(p2):
            array[i,j] = 2
    # R, return to S based on p3
    elif array[i,j] == 2:
        if roll(p3):
            array[i,j] = 0


    # Returns nothing as of now, as array should update regardless. If needed, will add more here.
    return True


def iteration_SIRS(lattice_size, iterations, p1,p2,p3, vis):
    # make initial array
    array = SIRS_array(lattice_size)
    # set up
    inf_av = []
    # include redundancy count, N defines how long before quitting
    N = 10
    redundancy = 0
    i_n = 0
    # Set custom cmap and its range
    cmap = mpl.cm.get_cmap("inferno", 3)

    for i in range(iterations):
        # find new matrix,
        for _ in range(lattice_size**2):
            # Update array
            array_update(array, lattice_size, p1, p2, p3)

     # plot every 5th
        if (i%1==0) and (vis==1):
            plt.clf()
            im=plt.imshow(array, animated=True, cmap=cmap)
            plt.colorbar(im)
            plt.pause(0.0005)

     # notify of sweep number, and determine <I>/N after 100 sweeps
        if (i%10==0) and (i>100):
            # Find number of infected in array
            i_n_prev = i_n
            i_n = np.count_nonzero(array == 1)/len(array)
            # append average infected to list
            inf_av.append(i_n)
            # if i_n doesnt change by more than 0.001 for N iterations, break loop
            if (math.fabs(i_n_prev - i_n) < 0.001):
                redundancy += 1
            else:
                redundancy = 0

        # print sweeps every 100
        if (i%100==0):
            i_n_visual = np.count_nonzero(array == 1)/len(array)
            print("Sweep {}/{}\nAverage Infection: {:.4f}".format(i,iterations,i_n_visual))


        # redundancy checks, if no change for 10 runs, break
        if redundancy == N:
            print("No changes after {} sweeps, quitting...".format(N))
            break

    return inf_av


# MAIN FUNCTION
def run_code(lattice, iterations, p1, p2, p3, vis):
    # convert sys arguments from string to floats/ints
    lattice = int(lattice)
    iterations = int(iterations)
    p1 = float(p1)
    p2 = float(p2)
    p3 = float(p3)
    vis = int(vis)
    # run the iterator, collecting average I values
    inf_av = iteration_SIRS(lattice, iterations, p1, p2, p3, vis)

    # plot average I by time
    time_list = np.linspace(0,len(inf_av),num=len(inf_av), endpoint=False)

    if (vis!=2):
        plt.clf()
        plt.plot(time_list, inf_av)
        plt.scatter(time_list, inf_av)
        plt.show()

    # return for the phase contour calculations
    return inf_av


# CALL FUNCTION
# check if not imported
if __name__ == "__main__":
    # Check to make sure enough arguments
    if len(sys.argv) == 7:
        run_code(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        print("\nScript takes exactly 5 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\n LATTICE SIZE\n\n ITERATIONS\n\n Probability of Infection\n\n Probability of Recovery\n\n Probability of Re-susceptibility\n\n Mode\n 0 - I/N plot only\n 1 - Visualisation & Data Collection\n 2 - Contour plot only")
