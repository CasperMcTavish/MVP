import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy

# Import all the unique GOL objects
from templates import *

##################################################################################################
# GAME OF LIFE BASIC FUNCTIONALITY
# INCLUDES ANIMATION (TOGGLE-ABLE) AND SIMULATION
##################################################################################################



######################
# ARRAY CREATION
######################
# LAYOUT OF DIFFERENT WAYS TO CREATE INITIAL ARRAYS

####
# RANDOM ARRAY
####
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


####
# MANY GLIDER ARRAY
####

# Create an array with a glider right in the middle if there is space
def glider_array(lattice_size):
    # If there is space, make glider
    # Choose central point and create glider around it
    m = int(lattice_size//2)
    array = np.zeros((lattice_size, lattice_size), dtype=bool)

    # check lattice size for what type of glider you should make
    if (lattice_size > 4) and lattice_size != 50:
        glider_creator(m, m, 0, array)
    # If 50x50, make a brigade of little gliders
    elif lattice_size == 50:

        glider_creator(m, m, 0, array)
        glider_creator(m+10, m, 1, array)
        glider_creator(m+20, m, 0, array)
        glider_creator(m-10, m, 1, array)
        glider_creator(m-20, m, 0, array)
        glider_creator(m, m+10, 1, array)
        glider_creator(m, m+20, 0, array)
        glider_creator(m, m-10, 1, array)
        glider_creator(m, m-20, 0, array)

    else:
        print("WARNING: not enough space to make glider, returning empty array instead")
        array = np.zeros((lattice_size, lattice_size), dtype=bool)

    return array

####
# OSCILLATOR ARRAY
####

def oscil_array(lattice_size):
    # only works for arrays larger than 10
    if (lattice_size >= 10):

        # Produce initial array
        array = np.zeros((lattice_size, lattice_size), dtype=bool)

        # Choose central point and create oscillator around it
        m = int(lattice_size//2)

        # Construct initial conditions for a nice oscillator
        oscil_creator(m, m, array)

        # For a larger lattice, add more! This is a lot of code, but it all does the same thing
        if lattice_size == 50:
            n = m + 10
            l = m - 10
            oscil_creator(n, n, array)
            oscil_creator(n, l, array)
            oscil_creator(l, l, array)
            # Add a little explosion in with sinks
            expl_oscil_creator(l, n+10, array)

        return array



    else:
        print("WARNING: not enough space to make oscillator, returning empty array instead")
        array = np.zeros((lattice_size, lattice_size), dtype=bool)



####
# SINGULAR GLIDER ARRAY
####


# Create an array with a glider right in the middle if there is space
def sing_glider_array(lattice_size):
    # If there is space, make glider
    # Choose central point and create glider around it
    m = int(lattice_size//2)
    array = np.zeros((lattice_size, lattice_size), dtype=bool)

    # check lattice size for what type of glider you should make
    if (lattice_size > 4) and lattice_size != 50:
        glider_creator(m, m, 0, array)
    # If 50x50, make a brigade of little gliders
    elif lattice_size == 50:

        # create glider initial conditions here.
        single_glider_creator(m, m, array)

    else:
        print("WARNING: not enough space to make glider, returning empty array instead")
        array = np.zeros((lattice_size, lattice_size), dtype=bool)

    return array



######################
# LOGIC
######################
# THE FUNCTIONAL LOGIC FOR OUR GOL SIMULATION

# NEIGHBOUR CHECKER
# Counts up the number of nearest alive neighbours.
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

# UPDATER
# Updated the array based on the game of life rules.
# Returns number of active sites of new array also
def array_update(array, lattice_size):

    # Alive cell checks:
    #       < 2 alive neighbours = DEATH
    #       2 or 3 alive neighbours = LIFE
    #       > 3 alive neighbours = DEATH

    # Dead cell checks:
    #       3 alive neighbours = LIFE

    # Copy to new array (slow but preserves old array format). You need to remember previous format to ensure iteration works
    new_array = np.copy(array)

    # start tally for 'active sites'. Will add up whenever a state changes
    a_sites = 0
    # scan across entire array
    for i in range(lattice_size):
        for j in range(lattice_size):

            # Collect nearest neighbours
            nn_count = gol_nn_check(array, lattice_size, i, j)

            # IF DEAD AND 3 ALIVE NEIGHBOURS
            if ((array[i, j] == False) and (nn_count == 3)):
                # BRING TO LIFE
                new_array[i, j] = True
                a_sites += 1
            # IF ALIVE
            elif ((array[i, j] == True)):
                # AND 2 OR 3 NEIGHBOURS
                if (nn_count == 2 or nn_count == 3):
                    # STAY ALIVE
                    continue;
                # OTHERWISE DEATH
                else:
                    new_array[i, j] = False
                    a_sites += 1

    # return new array as well as total number of active sites
    return new_array, a_sites


# INITIALISER/ITERATOR
# initialises and iterates over the array, updating it via above function
# equilibrium is True of False. If True, will shut down and output i, otherwise will just run until iterations are finished.
def gol_sim_run(lattice_size, sim_type, iterations, mode):
    # sim_type
    # 0 = random array
    # 1 = glider in centre array
    # 2 = oscillator

    # mode
    # 0 - visualisation
    # 1 - equilibrium hunting
    # 2 - glider COM

    # Define array
    if sim_type == 0:
        print("Producing random simulation...")
        array = gol_array(lattice_size)
    elif sim_type == 1:
        print("Producing glider simulation...")
        array = glider_array(lattice_size)
    elif sim_type == 2:
        print("Producing oscillator simulation...")
        array = oscil_array(lattice_size)
    elif sim_type == 3:
        print("Producing singular glider simulation...")
        array = sing_glider_array(lattice_size)

    # If only visualising, MODE = 0
    if mode == 0:

        # Iterate over array
        for i in range(iterations):

            # plot every nth
            n = 1
            if (i%n==0):
                plt.cla()
                im=plt.imshow(array, animated=True)
                plt.draw()
                plt.pause(0.05)


            # Update array
            array, n_a_sites = array_update(array, lattice_size)

        # at end, return 0. This is to be consistent for the equilibrium testing component
        return 0

    # For finding equilibrium, MODE = 1
    elif mode == 1:
        print("Searching for equilibrium...")
        i = 0
        eq_val = 0
        # set active sites to 0 initially
        a_sites = 0

        # May need to put a break on this, but we shall see
        # Loop until equilibrium is achieved for 10 updates
        while (eq_val < 10):

            # REMOVING PLOTTING AS JUST WANT EQ VALUE OUT

            # plot every nth
            #n = 1
            #if (i%n==0):
            #    plt.cla()
            #    im=plt.imshow(array, animated=True)
            #    plt.draw()
            #    plt.pause(0.001)


            # Update array
            array, n_a_sites = array_update(array, lattice_size)
            # If number of active sights doesn't change, take a tally
            if n_a_sites == a_sites:
                eq_val += 1
            else:
                eq_val = 0

            # Update a_sites
            a_sites = n_a_sites


            i += 1

        # Once equilibrium has been found, return i (iteration number)
        return i

    # Glider calculations
    elif mode == 2:
        print("Glider speed calculations...")

        # create COM array and time list
        com_list = []
        iter_list = []

        for i in range(iterations):
            # Update array
            array, n_a_sites = array_update(array, lattice_size)

            # after 50 iterations (to let state become singular glider), every 10 iterations, find COM of glider
            if ((i>50) and i%10==0):
                print("Sweep {}/{}...".format(i,iterations))
                # find COM and write to array
                # take max and min values in x and y,


                # Until scanned through to find first live cell, keep as False
                # will do screendoor scanning across X and Y

                # x_min

                check = False
                p = 0

                while(check==False):
                    # If a living cell is in column p, change check to true
                    if True in array[:, p]:
                        check = True
                        x_min = p
                    p += 1


                # x_max, starting from last column and scanning backwards

                check = False
                p = lattice_size-1

                while(check==False):
                    # If a living cell is in column p, change check to true
                    if True in array[:, p]:
                        check = True
                        x_max = p
                    p -= 1


                # y_min

                check = False
                p = 0

                while(check==False):
                    # If a living cell is in row p, change check to true
                    if True in array[p, :]:
                        check = True
                        y_min = p
                    p += 1

                # y_max, starting from last column and scanning backwards

                p = lattice_size-1
                check = False

                while(check==False):
                    # If a living cell is in row p, change check to true
                    if True in array[p, :]:
                        check = True
                        y_max = p
                    p -= 1

                # ABOVE CODE CAN BE MADE SHORTER VIA ITERATION

                # if xmax/ymax - xmin/ymin > lattice_size/2 assume boundary crossing and ignore
                if (x_max - x_min > lattice_size/2) or (y_max - y_min > lattice_size/2):
                    continue;
                # if not boundary crossing, calculate COM
                else:
                    x_numer = 0
                    x_denom = 0
                    y_numer = 0
                    y_denom = 0
                    for j in range(lattice_size):
                        for k in range(lattice_size):
                            # Collect sums
                            x_numer += j * array[j,k]
                            x_denom += array[j,k]

                            y_numer += k * array[j,k]
                            y_denom += array[j,k]
                    # After summation, divide and collect
                    x_mass = (x_numer/x_denom)
                    y_mass = (y_numer/y_denom)
                    iter_list.append(i)
                    com_list.append([x_mass,y_mass])




        # plot COM XY plot
        com_list = np.array(com_list)
        plt.scatter(com_list[:,0],com_list[:,1])
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("Flight of glider across XY coordinates, with each point taken 10 iterations apart")
        plt.savefig("glider_path.png")
        plt.show()

        # plot X COM wrt time
        plt.scatter(iter_list,com_list[:,0])
        plt.xlabel("Iteration number/Time")
        plt.ylabel("X position")
        plt.title("Flight of glider across X coordinates and time ")
        plt.savefig("glider_path_Xtime.png")
        plt.show()

        # plot Y COM wrt time
        plt.scatter(iter_list,com_list[:,1])
        plt.xlabel("Iteration number/Time")
        plt.ylabel("Y position")
        plt.title("Flight of glider across Y coordinates and time ")
        plt.savefig("glider_path_Ytime.png")
        plt.show()


        # save COM list
        pos_write(com_list, "com_list.txt")
        pos_write(iter_list, "iter_list.txt")
        # call function that fits straight line to COM list (when there are 10 points in a row with no interruptions, away from start and end.)

        # at end, return 0. This is to be consistent for the equilibrium testing component
        return 0





################################
# MISC FUNCTIONS
################################
# May not be used explicitly here, but are used in scripts that import this file

def pos_write(data, file_name):
    '''
    Writes list/array to file
    eg.
    3002 2200
    3232 2400
    ...
    :param positions:       The list of positions to be written to the file
    '''

    # creates a file if it doesnt already exist
    with open(file_name, "w") as f:
        for i in range(len(data)):
            # reformatting for the format our automation system uses
            pos = str(data[i])
            pos = pos.strip("[]")
            pos = pos.replace(",", "")
            f.write(pos + "\n")


def read_file(filename):
    with open(filename) as f:
        contents = f.readlines()

    if (len(contents) > 1):
        print("File read of length: " + str(len(contents)))
    else:
        print("File length is too short to process (please include more positions)")

    # convert to floats in this case
    return contents






# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 5:
        # run code, force as integers
        gol_sim_run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        print("\nScript takes exactly 4 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\n LATTICE SIZE\n\n INITIAL CONDITIONS\n  0 - Random\n  1 - Many Gliders\n  2 - Oscillator\n  3 - Single Glider\n\n ITERATIONS\n\n MODE\n  0 - Run for visualisation purposes\n  1 - Run to collect iteration at which equilibrium is reached\n      WARNING: Will run UNTIL equilibrium is reached, may be longer than iterations inputted.\n  2 - Run to find glider speed and COM over time\n      WARNING: For initial conditions choose 'Single Glider', or expect inaccurate results.")
