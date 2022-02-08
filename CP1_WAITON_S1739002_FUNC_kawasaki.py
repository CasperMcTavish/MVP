import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import time


###########################

# THIS IS THE AUTOMATION CODE FOR KAWASAKI
# OUTPUTS THE CORRECT FILETYPES, ACROSS 10000 ITERATIONS FOR EACH TEMPERATURES 1.0 -> 3.0

###########################


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

def total_energy_calc(array, J, lattice_size):
    # takes the total energy of the current array, while considering overcounting of the energy
    energy = 0
    for i in range(lattice_size):
        for j in range(lattice_size):
            # calculate the energy using the previous NN calculation
            energy += (energy_calc(array, J, lattice_size, i, j))
    # division by 4 to account for overcounting
    return (energy/4)

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



def iteration_kawasaki(iterations, lattice_size, T, array=None):
    # Start timer for easy tracking
    start = time.time()
    print("Running sweeps at T: {:.2f}".format(T))
    # if array is undefined in function, make new array
    if array is None:
        array = spin_array(lattice_size, lattice_size)
    # otherwise use array

    # Pull out the normalisation components
    N = len(array)
    # create energy and magnetism arrays
    enlist = []
    maglist = []
    # Normalisation value

    for i in range(iterations):
        # find new matrix,
        for _ in range(lattice_size**2):
            # glauber flip and calculate delta_E then update spins
            array = kawasaki(array, lattice_size, T)

        # collect energy values every 10 sweeps for calculation of susceptibility/heap capacity
        if (i>100) and (i%10==0):

            energy = total_energy_calc(array, 1, lattice_size)
            enlist.append(energy)


        # calculate average magnetism and susceptibility


    # make an intensive quality via averaging wrt array size and nSweeps
    #av_sus, av_mag = susceptibility(maglist, T)
    av_cap, av_en = heat_capacity(enlist, T)

    # Setup bootstrap to collect error on sus and heat capacity
    # collect k measurements of heat capacity and get error between them
    caplist = []
    for k in range(len(enlist)):
        # collect list of l length of uniform random variables from enlist and maglist to give susceptibility values
        enlist_new = []
        for l in range(len(enlist)):
            # collect random variables from the list and use that to get energy list
            r = np.random.randint(0, len(enlist))
            enlist_new.append(enlist[r])
    # Take the new heat capacity with this list and then add to list for error calculation
        caplist.append(heat_capacity(enlist_new, T)[0])
    # Calculate error on capacity here

    # Collect squared mean and mean squared
    squared_cap = np.mean(np.square(caplist))
    cap_squared = (np.mean(caplist))**2
    # calculate error on heat capacity
    er_cap = np.sqrt(squared_cap - cap_squared)

    # time tracking
    end = time.time()
    timer = end-start
    print("Time elapsed for T={:.1f}: {:.3f}s ".format(T,timer))

    return av_cap, av_en, er_cap, array



def mag_calc(array):
    # calculate the total magnetism
    M = np.sum(array)
    return M

def heat_capacity(enlist,  T):
    # Calculate heat capacity and Energy, then return them
    heat_capacity = (np.mean(np.square(enlist))/2500 - ((np.mean(enlist))**2)/2500)*1/(T**2)
    E = np.mean(enlist)/2500
    return heat_capacity, E


def susceptibility(maglist, T):
    # Find magnetism and susceptibility
    # Divide by array size (50->2500, hardcoded here)
    M = np.mean(maglist)/2500
    susceptibility = (np.mean(np.square(maglist))/2500 - ((np.mean(maglist))**2)/2500)*1/T

    return susceptibility, M


def collate_mXEC_results_kawasaki(iterations, lattice_size):
    # Function that will pass over multiple values of T (1->3, 0.1 increments)
    # Collate mag and susceptibility values from these values

    # Create storage arrays for energy and heat capacity with errors

    av_cap = []
    er_cap = []
    av_en = []
    T_list = []
    # Create first loop, T = 1
    T = 1
    # Create initial array, set to half is spin up, half is spin down (ground state)
    array1 = np.ones((lattice_size, int(lattice_size/2)))
    array2 = np.zeros((lattice_size, int(lattice_size/2)))
    array = np.concatenate((array1, array2), axis=1)
    for i in range(21):
        # Update new info based on now T
        cap, en, cap_er, array = iteration_kawasaki(iterations, lattice_size, T, array)
        av_cap.append(cap)
        av_en.append(en)
        er_cap.append(cap_er)
        T_list.append(T)
        print("Temperature: {:.2f}".format(T))
        print("Energy: {:.2f}\nHeat Capacity: {:.4f}\n".format(av_en[i], av_cap[i]))
        T += 0.1

    # Plot and save
    plt.errorbar(T_list, av_cap, yerr = er_cap, fmt='o')
    plt.errorbar(T_list, av_cap, yerr = er_cap)
    plt.xlabel("Temperature")
    plt.ylabel("Heat Capacity")
    plt.title("Kawasaki Heat Capacity against Temperature (with errors)")
    plt.savefig("Kawasaki_Cap.png")
    plt.show()
    plt.scatter(T_list, av_en)
    plt.plot(T_list, av_en)
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.title("Kawasaki Energy against Temperature")
    plt.savefig("Kawasaki_En.png")
    plt.show()

    # Then save as text files

    pos_write(av_cap, "av_cap.txt")
    pos_write(av_en, "av_en.txt")
    pos_write(er_cap, "er_cap.txt")




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




# set to 10000 for real runs
collate_mXEC_results_kawasaki(10000, 50)
