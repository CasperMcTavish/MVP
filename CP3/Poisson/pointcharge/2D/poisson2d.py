import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to zero except central point source
def init_array(lattice_size, gam):

    # set array of zeros except for central point, this is rho array
    array = np.random.normal(gam,1/10, size = (lattice_size,lattice_size))
    rho = np.zeros((lattice_size,lattice_size), dtype=float)
    # set central point to 1
    mid_point = int(lattice_size/2)
    rho[mid_point,mid_point] = 1


    return array, rho

# QUIVER MATPLOTLIB

def update_gam(array, lattice_size, dx, rho):
    # update the gamma values

    # complete the discretised calculation
    newarray = 1/4 * (np.roll(array,1,axis=0) + np.roll(array,-1,axis=0) + np.roll(array,1,axis=1) + np.roll(array,-1,axis=1) + rho)

    # set the edges to zero
    newarray[:,[0,-1]] = newarray[[0,-1],:] = 0
    return newarray

def e_field(array, dx):
    # define electric field
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    #print(np.sum(e_xfield))
    #print(np.sum(e_yfield))
    #print(np.sum(e_zfield))

    # set edges to zero

    e_xfield[:,[0,-1]] = e_xfield[[0,-1],:] = e_yfield[:,[0,-1]] = e_yfield[[0,-1],:] = 0
    # normalise each component
    norm_e_xfield = [x/(x**2+y**2) for x,y in zip(e_xfield,e_yfield)]
    norm_e_yfield = [y/(x**2+y**2) for x,y in zip(e_xfield,e_yfield)]

    e_field = e_xfield + e_yfield
    return (norm_e_xfield,norm_e_yfield, e_field)

def checker(array, newarray):
    # check the difference between the phi values of each array
    value = np.sum(abs(newarray - array))
    return value



def iterator(lattice_size, dx):

    # initialise array
    array, rho = init_array(lattice_size, 0.001)

    time_list = []

    i = 0
    # iterate 3D array?
    while(True):

        # update array
        newarray = update_gam(array, lattice_size, dx, rho)

        # check if the value between these two arrays is zero, if so then break
        # if not, then continue
        value = checker(array, newarray)

        if (i%100==0):
            # print the sum of the differences
            print("Convergence factor: {0:.6f}".format(np.sum(value)))

        # when convergence is low enough, break out the loop
        if (value < 0.001):
            array = newarray
            break;
        else:
            array = newarray
            i += 1


    # once out of the loop, find the E field for the space
    arraynorm = array/(np.max(array))
    #E_Fx, E_Fy, E_Fz, E_F = e_field(arraynorm, dx)
    E_Fx, E_Fy, E_F = e_field(array, dx)

    # plot 2D histo of E_F
    x = y = np.linspace(0,lattice_size-1,lattice_size)
    X,Y = np.meshgrid(x,y)
    cs = plt.contourf(X, Y, E_F, cmap="bone")
    cbar = plt.colorbar(cs)
    plt.show()
    # normalise e field and potential
    #E_Fnorm = E_F/(np.max(E_F))
    #arraynorm = array/(np.max(array))

    # print the potential here
    #plotting central slice
    imsh = plt.imshow(arraynorm)
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()

    # plot e-field QUIVER

    #DX,DY = np.gradient(E_F[int(lattice_size/2)])

    E_Fyslice = E_Fy
    E_Fxslice = E_Fx
    E_Fynorm = E_Fy/(np.power(E_Fy,2)+(np.power(E_Fx,2)))
    E_Fxnorm = E_Fx/(np.power(E_Fy,2)+(np.power(E_Fx,2)))
    # set edges to zero
    #E_Fynorm[:,[0,-1],:] = E_Fynorm[[0,-1]] = E_Fynorm[:,:,[0,-1]] = E_Fxnorm[:,[0,-1],:] = E_Fxnorm[[0,-1]] = E_Fxnorm[:,:,[0,-1]] = 0


    plt.quiver(Y,X,E_Fy,E_Fx)
    plt.title("YX - Ey,Ex")
    plt.show()

    plt.quiver(X,Y,E_F,E_Fx)
    #  THIS ONE
    plt.title("XY - Ey,Ex")
    plt.show()
    '''
    plt.quiver(X,Y,E_Fxslice,E_Fyslice)
    plt.title("XY - Ex,Ey")
    plt.show()


    plt.quiver(X,Y,E_Fxnorm,E_Fynorm)
    plt.title("XYnorm - Ex,Ey")
    plt.show()

    plt.quiver(Y,X,E_Fynorm,E_Fxnorm)
    plt.title("YXnorm - Ey,Ex")
    plt.show()
    '''
    # SAVE EFIELD AND GRAD FIELD
    # save xy slice of 3D array
    np.savetxt("phiarray2D.txt", array)

# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 3:
        # run code, force as ints/floats
        iterator(int(sys.argv[1]), float(sys.argv[2]))
    else:
        print("\nScript takes exactly 2 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nLattice Size (square)\n\ndx")
