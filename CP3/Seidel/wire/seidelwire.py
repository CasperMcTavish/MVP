import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

# FOR CONTEXT
# rho -> J (current)
# phi array -> A array



# JACOBIAN, BUT 3D FOR MAGNETIC WIRE PASSING MIDPOINT

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to zero except central point source
def init_array(lattice_size, gam):

    # set array of zeros except for central point, this is J(current) array
    array = np.random.normal(gam,1/10, size = (lattice_size,lattice_size,lattice_size))
    rho = np.zeros((lattice_size,lattice_size,lattice_size), dtype=float)
    # set central point to 1 across all layers
    mid_point = int(lattice_size/2)
    for i in range(lattice_size):
        rho[i][mid_point][mid_point] = 1

    return array, rho

# QUIVER MATPLOTLIB

def update_gam(array, lattice_size, rho):

    for i in range(len(array)):
        for j in range(len(array)):
            for k in range(len(array)):
                # set edges to 0
                if (i == 0) or (j == 0) or (k == 0) or (i==lattice_size-1) or (j==lattice_size-1) or (k==lattice_size-1):
                    array[i,j,k] = 0
                else:
                    array[i,j,k] = 1/6 * (array[i+1,j,k] + array[i-1,j,k] + array[i,j+1,k] + array[i,j-1,k] + array[i,j,k+1] + array[i,j,k-1] + rho[i,j,k])

    return array


def e_field(array, dx):
    # define electric field
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    e_zfield = -(1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))

    e_field = e_xfield + e_yfield + e_zfield
    return (e_xfield,e_yfield,e_zfield, e_field)

def b_field(array, dx):
    # produce magnetic field

    # dz = 0 so ignore those components, Ax = Ay = 0
    dyAz = (1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))
    dzAy = 0
    dzAx = 0
    dxAz = (1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    dxAy = 0
    dyAx = 0
    b_xfield = dyAz - dzAy
    b_yfield = dzAx - dxAz
    b_zfield = dxAy - dyAx

    b_field = np.sqrt(np.square(b_xfield) + np.square(b_yfield) + np.square(b_zfield))
    return (b_xfield, b_yfield, b_zfield, b_field)

def checker(array, newarray):
    # check the difference between the phi values of each array
    value = np.sum(np.abs(newarray - array))
    return value



def iterator(lattice_size, dx, accuracy):

    # initialise array
    array, rho = init_array(lattice_size, 0.001)
    time_list = []

    i = 0
    # iterate 3D array?
    while(True):

        # update array
        savearray = np.copy(array)
        array = update_gam(array, lattice_size, rho)

        # check if the value between these two arrays is zero, if so then break
        # if not, then continue
        value = checker(savearray, array)

        if (i%100==0):
            # print the sum of the differences
            print("Convergence factor: {0:.6f}".format(value))

        # when convergence is low enough, break out the loop
        # standard accuracy -> 0.01 or 0.001
        if (value < accuracy):
            #array = newarray
            break;
        else:
            #array = newarray
            i += 1


    # once out of the loop, find the E field for the space
    arraynorm = array/(np.max(array))
    #E_Fx, E_Fy, E_Fz, E_F = e_field(arraynorm, dx)
    #E_Fx, E_Fy, E_Fz, E_F = e_field(array, dx)

    # calculate B field
    B_Fx, B_Fy, B_Fz, B_F = b_field(array, dx)

    # normalise e field and potential
    #E_Fnorm = E_F/(np.max(E_F))
    #arraynorm = array/(np.max(array))

    # print the potential here
    #plotting central slice
    imsh = plt.imshow(arraynorm[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()

    # plot e-field QUIVER
    x = y = np.linspace(0,lattice_size-1,lattice_size)
    X,Y = np.meshgrid(x,y)
    #DX,DY = np.gradient(E_F[int(lattice_size/2)])

    imsh = plt.imshow(B_F[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Magnetic field at centre of array")
    plt.savefig("magnetic.png")
    plt.show()


    # SAVE EFIELD AND GRAD FIELD
    # save xy slice of 3D array
    np.save("phiarray.npy", array)

# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 4:
        # run code, force as ints/floats
        iterator(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    else:
        print("\nScript takes exactly 3 arguments, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nLattice Size (square)\n\ndx\n\nAccuracy of discretisation")
