import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time


# read in the slice of the 3D phi array


def main():
    array = np.loadtxt("phiarray.txt")
    imsh = plt.imshow(array)
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()


    '''
    lat4 = int(len(array)/4)

    print(array[lat4:-lat4,lat4:-lat4])

    imsh = plt.imshow(array[lat4:-lat4,lat4:-lat4])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()
    '''

    # plot phi against R along X,

    # take X slice
    slice = array[:,int(len(array)/2):len(array)]
    negative_slice = array[:,0:int(len(array)/2)]
    negative_slice = np.flip(negative_slice)

    pos_list = np.linspace(0,int(len(slice)/2),int(len(slice)/2))
    print(len(pos_list),len(slice[0]))
    print(len(pos_list), len(negative_slice[0]))

    # full range
    plt.figure(figsize=(5,5))
    plt.title("MESSY")
    for i in range(len(slice)):

        plt.scatter(pos_list,slice[i])
        plt.scatter(pos_list,negative_slice[-i])

    plt.show()

    # V - R plot
    plt.figure(figsize=(5,5))
    plt.scatter(pos_list,slice[int(len(slice)/2)])
    plt.scatter(pos_list,negative_slice[int(len(negative_slice)/2)])
    #
    plt.title("CLEAN - R, V")
    plt.show()

    # LOG V - R plot
    # turn all zeros to small values
    slice[slice==0] = 0.1
    negative_slice[negative_slice==0] = 0.1
    log_nslice = np.log(negative_slice)
    log_slice = np.log(slice)
    pos_list[pos_list==0] = 0.1
    log_list = np.log(pos_list)

    plt.figure(figsize=(5,5))
    plt.scatter(pos_list,log_slice[int(len(log_slice)/2)])
    plt.scatter(pos_list,log_nslice[int(len(log_nslice)/2)])
    plt.title("CLEAN - R, logV")
    plt.show()


    # slice off components of position list that are too small
    # LOG LOG
    plt.figure(figsize=(5,5))
    plt.scatter(log_list,slice[int(len(slice)/2)])
    plt.scatter(log_list,negative_slice[int(len(negative_slice)/2)])
    plt.title("CLEAN - logR, V")
    plt.show()


main()
