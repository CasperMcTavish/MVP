import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time

def plot_quiver2D(x,y,u,v):
    plt.quiver(x,y,u,v)

# JACOBIAN, BUT 3D

######################
# ARRAY CREATION
######################
####
# RANDOM ARRAY
####
# Create a grid of spins i-rows,j-columns, will set everything to zero except central point source
def init_array(lattice_size, gam):

    # set array of zeros except for central point, this is rho array
    array = np.random.normal(gam,1/10, size = (lattice_size,lattice_size,lattice_size))
    rho = np.zeros((lattice_size,lattice_size,lattice_size), dtype=float)
    # set central point to 1
    mid_point = int(lattice_size/2)
    rho[mid_point,mid_point,mid_point] = 1


    return array, rho

# Seidel updater

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
    #print(np.sum(e_xfield))
    #print(np.sum(e_yfield))
    #print(np.sum(e_zfield))
    e_field = e_xfield + e_yfield + e_zfield
    return (e_xfield,e_yfield,e_zfield, e_field)

def checker(array, newarray):
    # check the difference between the phi values of each array
    value = np.sum(np.abs(array - newarray))
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
    E_Fx, E_Fy, E_Fz, E_F = e_field(array, dx)

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
    '''
    E_Fyslice = E_Fy[int(lattice_size/2)]
    E_Fxslice = E_Fx[int(lattice_size/2)]
    E_Fynorm = E_Fy/(np.power(E_Fy,2)+(np.power(E_Fx,2)))
    E_Fxnorm = E_Fx/(np.power(E_Fy,2)+(np.power(E_Fx,2)))
    # set edges to zero
    E_Fynorm[:,[0,-1],:] = E_Fynorm[[0,-1]] = E_Fynorm[:,:,[0,-1]] = E_Fxnorm[:,[0,-1],:] = E_Fxnorm[[0,-1]] = E_Fxnorm[:,:,[0,-1]] = 0

    plt.quiver(Y,X,E_Fyslice,E_Fxslice)
    plt.title("YX - Ey,Ex")
    plt.show()

    plt.quiver(X,Y,E_Fyslice,E_Fxslice)
    plt.title("XY - Ey,Ex")
    plt.show()

    plt.quiver(X,Y,E_Fxslice,E_Fyslice)
    plt.title("XY - Ex,Ey")
    plt.show()


    plt.quiver(X,Y,E_Fxnorm[int(lattice_size/2)],E_Fynorm[int(lattice_size/2)])
    plt.title("XYnorm - Ex,Ey")
    plt.show()

    plt.quiver(Y,X,E_Fynorm[int(lattice_size/2)],E_Fxnorm[int(lattice_size/2)])
    plt.title("YXnorm - Ey,Ex")
    plt.show()
    '''
    # SAVE EFIELD AND GRAD FIELD
    # save xy slice of 3D array
    np.save("phiarray.npy", array)

    # collect e field plot and multiply vectors by 5 to be made visible
    e_f_grad = np.gradient(E_F[int(lattice_size/2)+1])
    e_fx = (e_f_grad[0]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))
    e_fy = (e_f_grad[1]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))

    # plot quivers
    step = 5
    plot_quiver2D(X[::step,::step],Y[::step,::step], e_fy[::step,::step],e_fx[::step,::step])
    plt.title("Electric field from central slice")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("efield.png")
    plt.show()

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
