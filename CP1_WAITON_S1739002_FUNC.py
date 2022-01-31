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


def iteration_glauber(iterations, lattice_size, T, array=None):
    # if array is undefined in function, make new array
    if array is None:
        array = spin_array(lattice_size, lattice_size)
    # otherwise use array

    # create susceptibility and magnetism arrays
    sus = []
    mag = []
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
        # calculate susceptibility and average magnetism after 100 sweeps, every 10 sweeps
        if (i>100) and (i%10==0):
            suss,magg = susceptibility_mag(array, T)
            #print(suss, magg, T)
            # append
            sus.append(suss)
            mag.append(magg)

    # Spit out list of sus and mag, then find the average value after said sweeps, plot and average
    average_sus = np.mean(sus)
    average_mag = np.mean(mag)
    # plot and show current stabilisation, if needed, need to close animation beforehand
    #plt.close()
    #plt.plot(sus)
    #plt.plot(mag)
    #plt.show()
    return average_sus, average_mag, array



#iteration_kawasaki(1000, 50, 1)
#iteration_glauber(1000, 50, 1)

def susceptibility_mag(array, T):
    # takes an array and estimates the average value of total magnetisation in equilibrium state and susceptibility
    # Average M is sum of all spins divided by the number of spins
    # THIS IS WRONG N I THINK, WILL ASK ON FRIDAY CHECK MINUTE 50 OF LECTURE 2
    N = len(array)
    M = np.sum(array)/(N**2)
    # average M^2 is the square of the sum of signs divided by number of spins. Because our len(array) squared is the total number of spins, the squares cancel.
    # so, easier calculation (I THINK)
    M_squared = np.sum(array)/N
    susceptibility = (M_squared - M**2)*1/((N**2)*T)
    # currently giving back raw magnetism
    return susceptibility, np.sum(array)


def collate_mag_susc_results_glauber(iterations, lattice_size):
    # Function that will pass over multiple values of T (1->3, 0.1 increments)
    # Collate mag and susceptibility values from these values

    # Create storage arrays for mag and susc
    av_mag = []
    av_sus = []
    T_list = []
    # Create first loop, T = 1
    T = 1
    T_list.append(T)
    sus, mag, array = iteration_glauber(iterations, lattice_size, T)
    av_mag.append(abs(mag))
    av_sus.append(abs(sus))
    print("Temperature: {:.2f}\nMagnetism: {:.4f}\nSusceptibility: {:.4f}\n".format(T, av_mag[0], av_sus[0]))
    for i in range(19):
        T += 0.1
        # Update new info based on now T
        sus, mag, array = iteration_glauber(iterations, lattice_size, T, array)
        # Append new info to array, SETTING TO ABSOLUTE RIGHT NOW BECAUSE OF FLIPS
        av_mag.append(abs(mag))
        av_sus.append(abs(sus))
        T_list.append(T)
        print("Temperature: {:.2f}\nMagnetism: {:.4f}\nSusceptibility: {:.4f}\n".format(T, av_mag[i+1], av_sus[i+1]))
    # close the animation
    print(av_mag)
    print(av_sus)
    plt.close()
    plt.plot(T_list, av_mag)
    plt.plot(T_list, av_sus)
    pos_write(av_mag, "av_mag.txt")
    pos_write(av_sus, "av_sus.txt")
    plt.show()




def pos_write(data, file_name):
    '''
    Writes list/array to file
    eg.
    3002 2200
    3232 2400
    ...
    :param positions:       The list of positions to be written to the file
    '''
    # write file in the correct format
    # eg:
    # 3430 2323
    # 3232 0202
    # ...

    # creates a file if it doesnt already exist
    with open(file_name, "w") as f:
        for i in range(len(data)):
            # reformatting for the format our automation system uses
            pos = str(data[i])
            pos = pos.strip("[]")
            pos = pos.replace(",", "")
            f.write(pos + "\n")





collate_mag_susc_results_glauber(1000, 50)



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
