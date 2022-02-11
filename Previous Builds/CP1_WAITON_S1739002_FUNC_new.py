import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys

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
            energy += -(energy_calc(array, J, lattice_size, i, j))
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



def iteration_kawasaki(iterations, lattice_size, T):
    # make initial array
    array = spin_array(lattice_size, lattice_size)

    for i in range(iterations):
        # find new matrix within n^2 loop
        for _ in range(lattice_size**2):
            # run kawasaki function to update array
            array = kawasaki(array, lattice_size, T)
        ## plot every 5th
        # commented out currently
        #if (i%5==0):
        #    plt.cla()
        #    im=plt.imshow(array, animated=True)
        #    plt.draw()
        #    plt.pause(0.0001)


def iteration_glauber(iterations, lattice_size, T, array=None):
    # if array is undefined in function, make new array
    if array is None:
        array = spin_array(lattice_size, lattice_size)
    # otherwise use array

    # Pull out the normalisation components
    N = len(array)
    # create energy and magnetism arrays
    en1 = 0
    en2 = 0
    enlist = []

    mag1 = 0
    mag2 = 0
    maglist = []
    # Normalisation value
    sweep_no = 0
    for i in range(iterations):
        # find new matrix,
        for _ in range(lattice_size**2):
            # glauber flip and calculate delta_E then update spins
            array = glauber(array, lattice_size, T)
     ## plot every 5th
     # commented out currently
    #    if (i%5==0):
    #        plt.cla()
    #        im=plt.imshow(array, animated=True)
    #        plt.draw()
    #        plt.pause(0.0001)
        # collect magnetism/energy values every 10 sweeps for calculation of susceptibility/heap capacity
        if (i>100) and (i%10==0):
            magg = mag_calc(array)

            # collect in a list also for error calculation
            maglist.append(magg)
            #mag1 = mag1 + magg
            #mag2 = mag2 + magg*magg

            energy = total_energy_calc(array, 1, lattice_size)
            enlist.append(energy)
            #en1 = en1 + energy
            #en2 = en2 + energy*energy

            # Can collect mag and energy lists here for each iteration of T, to get an error. But it cant be used for susceptibility and heat capacity, so why would you do that?
            # update sweep number for normalisation
            sweep_no += 1
        # calculate average magnetism and susceptibility

    norm1, norm2 = 1/(N*N*sweep_no), 1/(N*N*sweep_no*sweep_no)
    # make an intensive quality via averaging wrt array size and nSweeps
    av_sus, av_mag = susceptibility(maglist, T, norm1, norm2)
    av_cap, av_en = heat_capacity(enlist, T, norm1, norm2)

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
        caplist.append(heat_capacity(enlist_new, T, norm1, norm2)[0])
    # Calculate error on capacity here

    # take average of squared and normal components
    cap1 = 0
    cap2 = 0
    # add up all components, and squared components
    for i in range(len(caplist)):
        # Add C up and Csquared up, then find average by multiplying by the normalisations.
        cap1 = cap1 + caplist[i]
        cap2 = cap2 + caplist[i]*caplist[i]
    # get average, as already normalised
    squared_cap = cap2 * 1/(len(caplist))
    # Need to square average here
    cap_squared = 1/(len(caplist))*1/(len(caplist)) * cap1 * cap1
    # caluclate error by taking square root
    er_cap = np.sqrt(squared_cap - cap_squared)

    return av_sus, av_mag, av_cap, av_en, er_cap, array
    #return av_sus, av_mag, av_cap, av_en, er_cap, er_sus, array



#iteration_kawasaki(1000, 50, 1)
#iteration_glauber(1000, 50, 1)
def mag_calc(array):
    # calculate the total magnetism
    M = np.sum(array)
    return M

def heat_capacity(enlist,  T, norm1, norm2):
    # Format energy as expected to find heat capacity
    en1 = 0
    en2 = 0
    for i in range(len(enlist)):
        en1 = en1 + enlist[i]
        en2 = en2 + enlist[i]*enlist[i]
    # calculates heat capacity and energy (intensive/normalised)
    E = norm1*en1
    heat_capacity = (norm1 * en2 - norm2*en1*en1)*1/(T**2)
    return heat_capacity, E


def susceptibility(maglist, T, norm1, norm2):
    # Format magnetism correctly
    M1 = 0
    M2 = 0
    for i in range(len(maglist)):
        M1 = M1 + maglist[i]
        M2 = M2 + maglist[i]*maglist[i]
    # give back intensive mag and susceptibility
    susceptibility = (norm1*M2 - norm2*M1*M1)*1/T
    M = norm1*M1
    return susceptibility, M


def collate_mXEC_results_glauber(iterations, lattice_size):
    # Function that will pass over multiple values of T (1->3, 0.1 increments)
    # Collate mag and susceptibility values from these values

    # Create storage arrays for mag, susc, energy, and heat capacity, and their errors
    av_mag = []
    av_sus = []
    er_sus = []
    av_cap = []
    er_cap = []
    av_en = []
    T_list = []
    # Create first loop, T = 1
    T = 1
    # Create initial array
    array = spin_array(lattice_size, lattice_size)
    for i in range(21):
        # Update new info based on now T
        #sus, mag, cap, en, cap_er, sus_er, array = iteration_glauber(iterations, lattice_size, T, array)
        sus, mag, cap, en, cap_er, array = iteration_glauber(iterations, lattice_size, T, array)
        # Append new info to array, SETTING TO ABSOLUTE RIGHT NOW BECAUSE OF FLIPS
        av_mag.append(abs(mag))
        av_sus.append(sus)
        av_cap.append(cap)
        av_en.append(en)
        er_cap.append(cap_er)
        T_list.append(T)
        print("Temperature: {:.2f}\nMagnetism: {:.4f}\nSusceptibility: {:.4f}".format(T, av_mag[i], av_sus[i]))
        print("Energy: {:.2f}\nHeat Capacity: {:.4f}\n".format(av_en[i], av_cap[i]))
        T += 0.1
    # close the animation
    plt.close()
    # Plot everything else
    plt.plot(T_list, av_mag)
    plt.show()
    plt.plot(T_list, av_sus)
    plt.show()
    plt.errorbar(T_list, av_cap, yerr = er_cap)
    plt.show()
    plt.plot(T_list, av_en)
    plt.show()
    # Then save
    pos_write(av_mag, "av_mag.txt")
    pos_write(av_sus, "av_sus.txt")
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
collate_mXEC_results_glauber(10000, 50)



'''
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

# Check if main script.
if __name__ == "__main__":
    # Check to make sure enough arguments
    if len(sys.argv) == 5:
        run_code(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("\nScript takes exactly 4 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\n DYNAMIC MODEL\n  0 - Glauber\n  1 - Kawasaki\n\n LATTICE SIZE\n\n TEMPERATURE\n\n ITERATIONS")
else:
    print("Ising model functions imported successfully...")
'''
