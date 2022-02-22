import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy

from gol_sim import *

# TAKE DATA FROM GLIDER AND CALCULATE SPEED OF GLIDER FROM IT

# Read the DATA
positions = read_file("com_list.txt")

# scrub ugly data
positions = [positions[i].rstrip().lstrip() for i in range(len(positions))]
positions = [positions[i].split() for i in range(len(positions))]
positions = (np.array(positions)).astype(float)
print("Typical positions format:")
print(" x    y ")
print(positions[0])


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
#print("")
#print(speed_list[-1:])
#print(speed_list[-1:][0][0])
#print(speed_list[-1:][0][1])

# Find speed (take first and last points), x2-x1, y2-y1 to find distance travelled. Magnitude via sqrt(x^2+y^2), then div by number of iterations

# Indexing because slicing a list like this creates its own list...lets not talk about it
deltaX = speed_list[0][0] - speed_list[-1:][0][0]
deltaY = speed_list[0][1] - speed_list[-1:][0][1]
mag_delta = np.sqrt(deltaX**2 + deltaY**2)

# data taken every 10 iterations, so multiply list size by 10
iter_no = len(speed_list)*10

speed = mag_delta/iter_no

print("Total Speed of Glider: {:.2f} cells per iteration".format(speed))
