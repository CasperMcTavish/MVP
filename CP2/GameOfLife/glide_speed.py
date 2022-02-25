import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import scipy.optimize

from gol_sim import *

# TAKE DATA FROM GLIDER AND CALCULATE SPEED OF GLIDER FROM IT

# Read the DATA
positions = read_file("com_list.txt")
times = read_file("iter_list.txt")

# scrub ugly data
# positions
positions = [positions[i].rstrip().lstrip() for i in range(len(positions))]
positions = [positions[i].split() for i in range(len(positions))]
positions = (np.array(positions)).astype(float)
print("Typical positions format:")
print(" x    y ")
print(positions[0])

# times
times = [times[i].rstrip().lstrip() for i in range(len(times))]
times = (np.array(times)).astype(float)


# Find concurrent points (dont cross lattice boundary) to average over.

    # Scan through list, segment into sections based on if value of x and y keep getting bigger (when they shrink, its because lattice boundary has been crossed)
    # select largest section of list to keep for analysis
speed_list = []
speed_list_swap = []
redundancy = 0

for i in range(len(positions)):


    # Visualiser
    if (i%10) == 0:
        print("Sweep {}/{}".format(i,len(positions)))


    # Ensure reading back in index doesnt break code
    if (i !=0):

        # X value decreases, and Y value increases, so check boundary by making sure next value of X is smaller than previous, and opposite for Y
        if ((positions[i][0] < positions[i-1][0]) and (positions[i][1] > positions[i-1][1])):

            # add to the speed list
            speed_list_swap.append([positions[i][0],positions[i][1]])

        # Once it meets a boundary, check if its longer than the original speed_list.
        else:
            print("Length of new COM list: {}".format(len(speed_list_swap)))
            # If it is, rewrite and erase swap. If not, erase swap
            if (len(speed_list_swap) > len(speed_list)):
                print("Swapping...")
                speed_list = speed_list_swap
                speed_list_swap = []
                # reset redundancy
                redundancy = 0
            else:
                print("Continuing...")
                speed_list_swap = []
                # count up redundancy
                redundancy += 1

    # If its assumed list will get not longer, break loop
    # Can be set to higher threshold if needed, but here we only have one loop so its fine.
    if redundancy == 1:
        print("Maximum list found. Exiting...")
        break;

# Testing index
#print(speed_list[0])
#print(speed_list[0][0])
#print(speed_list[0][1])
#print(speed_list)

# X and Y list:
#print(np.array(speed_list)[:,0])
#print(np.array(speed_list)[:,1])

#print("")
#print(speed_list[-1:])
#print(speed_list[-1:][0][0])
#print(speed_list[-1:][0][1])

# Find speed ROUGH (take first and last points), x2-x1, y2-y1 to find distance travelled. Magnitude via sqrt(x^2+y^2), then div by number of iterations

# Indexing because slicing a list like this creates its own list...lets not talk about it
deltaX = speed_list[0][0] - speed_list[-1:][0][0]
deltaY = speed_list[0][1] - speed_list[-1:][0][1]
mag_delta = np.sqrt(deltaX**2 + deltaY**2)

# data taken every 10 iterations, so multiply list size by 10
# -1 as first component of the list counts as the 'start'
iter_no = len(speed_list)*10

speed = mag_delta/iter_no

print("Total Speed of Glider, Rough calculation: {:.5f} cells per iteration".format(speed))


# Now calculate it by fitting, which seems a bit derivative but may give different answers

# def linear function
def lin_func(x, m, b):
    return m*x + b

# Apply fitting
# Can take any values from time list, as equally spaced
# Just need to ensure correct length

time_list = times[:len(speed_list)]
x_list = np.array(speed_list)[:,0]
y_list = np.array(speed_list)[:,1]

# X fitting
popt_x, pcov_x = scipy.optimize.curve_fit(lin_func, time_list , x_list)


# Y fitting
popt_y, pcov_y = scipy.optimize.curve_fit(lin_func, time_list , y_list)

# Plot X fit
plt.plot(time_list, lin_func(time_list, *popt_x), 'g--', label='fit: m=%5.3f, b=%5.3f, ' % tuple(popt_x))
plt.scatter(time_list, x_list, label= "Data")
plt.xlabel("Iteration number/Time")
plt.ylabel("X position")
plt.title("Flight of glider across X coordinates and time with fitting")
plt.legend()
plt.savefig("Xtime_fitting.png")
plt.show()

# Plot Y fit
plt.plot(time_list, lin_func(time_list, *popt_y), 'g--', label='fit: m=%5.3f, b=%5.3f, ' % tuple(popt_y))
plt.scatter(time_list, y_list, label= "Data")
plt.xlabel("Iteration number/Time")
plt.ylabel("Y position")
plt.title("Flight of glider across Y coordinates and time with fitting")
plt.legend()
plt.savefig("Ytime_fitting.png")
plt.show()

# Average speed via magnitude
av_speed = np.sqrt(popt_x[0]**2 + popt_y[0]**2)
print("Total Speed of Glider, via fitting: {:.5f} cells per iteration".format(av_speed))
