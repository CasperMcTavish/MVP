import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import time
import scipy.optimize

def plot_tuple_array(tuple_array, name, xlabel, ylabel):
    # Splits array of tuples, plots it and then saves it

    x_val = [x[0] for x in tuple_array]
    y_val = [x[1] for x in tuple_array]

    save_name = name + ".png"
    plt.scatter(x_val, y_val, s=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(save_name)
    plt.show()

#print(zipped_potential[0])
##(43.30127018922193, 0.0) first element is R, second is V
#print(zipped_potential[0][0])
##43.30127018922193

def plot_tuple_array_fit(tuple_array, func, xlower_lim, xupper_lim, name, xlabel, ylabel):
    cut_arrayx = []
    cut_arrayy = []
    for i in range(len(tuple_array)):
        if (tuple_array[i][0] > xlower_lim) and (tuple_array[i][0] < xupper_lim):
            cut_arrayx.append(float(tuple_array[i][0]))
            cut_arrayy.append(float(tuple_array[i][1]))

    #print(type(cut_arrayx[0]))

    # collect as x and y (for scatter plot)
    x_val = [x[0] for x in tuple_array]
    y_val = [x[1] for x in tuple_array]


    # fit
    popt_x, _ = scipy.optimize.curve_fit(func, cut_arrayx,cut_arrayy)

    plt.plot(cut_arrayx, lin_func(cut_arrayx, *popt_x), 'g--', label='fit: m=%5.3f, b=%5.3f, ' % tuple(popt_x))
    save_name = name + "fitted.png"
    plt.scatter(x_val, y_val, s=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.savefig(save_name)
    plt.show()

# linear function for fitting
def lin_func(x,m,b):
    return m*np.array(x) + b

def e_field_old(array, dx):

    # define electric field
    lattice_size = len(array)

    # produce 3D vector consisting with each point having 3 coordinates, XYZ
    #e_field_array = np.empty((lattice_size,lattice_size,lattice_size), dtype=object)
    e_field_array = np.empty((lattice_size,lattice_size), dtype=object)

    # trying the old school way
    e_xfield = np.zeros((lattice_size,lattice_size,lattice_size))
    e_yfield = np.zeros((lattice_size,lattice_size,lattice_size))
    e_zfield = np.zeros((lattice_size,lattice_size,lattice_size))
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):
                # apply limits to the edges
                if (i==0) or (j==0) or (k==0) or (i==lattice_size) or (j==lattice_size) or (k==lattice_size) or (k==lattice_size-1) or (j==lattice_size-1) or (i==lattice_size-1):
                    e_xfield[i,j,k] = 0
                    e_yfield[i,j,k] = 0
                    e_zfield[i,j,k] = 0

                    # take middle slice
                    #if k==int(lattice_size/2):
                    ##e_field_array[i,j,k] = [e_xfield[i,j,k],e_yfield[i,j,k], e_zfield[i,j,k]]
                    #    e_field_array[i,j] = [e_xfield[i,j],e_yfield[i,j,k]]

                else:
                    e_xfield[i,j,k] = -(1/(2*dx) * (array[i+1,j,k]) - array[i-1,j,k])
                    e_yfield[i,j,k] = -(1/(2*dx) * (array[i,j+1,k]) - array[i,j-1,k])
                    e_zfield[i,j,k] = -(1/(2*dx) * (array[i,j,k+1]) - array[i,j,k-1])

                    # take middle slice
                    #if k==int(lattice_size/2):
                    ##e_field_array[i,j,k] = [e_xfield[i,j,k],e_yfield[i,j,k], e_zfield[i,j,k]]
                    #    e_field_array[i,j] = [e_xfield[i,j],e_yfield[i,j,k]]




def e_field(array, dx):

    # New school way (didnt work)
    e_xfield = -(1/(2*dx)) * (np.roll(array,-1,axis=0) - np.roll(array,1,axis=0))
    e_yfield = -(1/(2*dx)) * (np.roll(array,-1,axis=1) - np.roll(array,1,axis=1))
    e_zfield = -(1/(2*dx)) * (np.roll(array,-1,axis=2) - np.roll(array,1,axis=2))


    #print(np.sum(e_xfield))
    #print(np.sum(e_yfield))
    #print(np.sum(e_zfield))
    e_field = np.sqrt(np.square(e_xfield) + np.square(e_yfield) + np.square(e_zfield))
    return (e_xfield,e_yfield,e_zfield, e_field)

def plot_quiver2D(x,y,u,v):
    plt.quiver(x,y,u,v)

def main():

    # load phi array
    array = np.load("phiarray.npy")
    # load e_f array SHOULD PRODUCE E_F ARRAY INSTEAD
    #e_f = np.load("e_farray.npy")

    # define lattice size
    lattice_size = len(array)

    # collect E field
    E_Fx, E_Fy, E_Fz, E_F = e_field(array, 1)

    # Normalise e field
    norm_E_F = E_F/np.max(E_F)


    x = y = np.linspace(0,len(array)-1,len(array))
    X,Y = np.meshgrid(x,y)
    #print(len(X))
    #print(len(Y))
    #print(len(E_Fx[int(lattice_size/2)]))
    # take slice of E FIELD
    #print(E_F[int(lattice_size/2)][int(lattice_size/2)])
    #print(E_Fx[int(lattice_size/2)][int(lattice_size/2)])
    #print(E_Fy[int(lattice_size/2)][int(lattice_size/2)])
    #print(E_Fz[int(lattice_size/2)][int(lattice_size/2)])


    imsh = plt.imshow(E_F[int(lattice_size/2)+1])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Electric field at centre of array")

    plt.show()

    # collect e field plot and multiply vectors by 5 to be made visible
    e_f_grad = np.gradient(E_F[int(lattice_size/2)+1])
    e_fx = (e_f_grad[0]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))
    e_fy = (e_f_grad[1]/(np.sqrt(np.square(e_f_grad[0]) + np.square(e_f_grad[1]))))

    #print(e_fx[50][50])

    # plot quivers
    step = 2

    # This is borked sadly
    #plot_quiver2D(X[::step,::step],Y[::step,::step],(e_f_grad[0])[::step,::step], (e_f_grad[1])[::step,::step])
    #plt.show()
    plot_quiver2D(X[::step,::step],Y[::step,::step], e_fy[::step,::step],e_fx[::step,::step])
    plt.title("Electric field from central slice")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("efield.png")
    plt.show()

    # normalise main array
    norm_array = array/np.max(array)



    imsh = plt.imshow(array[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential at centre of array")
    plt.savefig("potential.png")
    plt.show()
    #quit()

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

    # plot phi against R across entire 3D array (how?)

    lattice_size = len(array)
    print("lattice size: {}".format(lattice_size))

    # Find midpoint
    midpoint = int(lattice_size/2)
    # this is R = 0

    # create 3D radius array of equal size to our phi array, with each index being distance from midpoint.
    rad_array = np.zeros((lattice_size,lattice_size,lattice_size), dtype= float)
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # create radial array centred around [0,0]
                rad_array[i,j,k] = np.sqrt((i-midpoint)**2 + (j-midpoint)**2 + (k-midpoint)**2)

    # plot shows that R is correct
    #plt.scatter(x, rad_array[int(lattice_size/2)][int(lattice_size/2)])
    #plt.show()

    # sort through both arrays to create tuples that can be plotted on scatter diagrams
    zipped_potential = []
    zipped_norm_potential = []
    zipped_log_norm_potential = []
    zipped_log_R_norm_potential = []
    zipped_e_field = []
    zipped_log_R_log_E = []
    R_V_E_array = []
    zipped_logR_logV = []

    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):

                # make main array
                R_V_E_array.append((rad_array[i,j,k], norm_array[i,j,k], norm_E_F[i,j,k]))

                # add tuples
                zipped_potential.append((rad_array[i,j,k],array[i,j,k]))
                zipped_norm_potential.append((rad_array[i,j,k], norm_array[i,j,k]))
                zipped_log_norm_potential.append((rad_array[i,j,k], np.log(norm_array[i,j,k])))
                zipped_logR_logV.append((np.log(rad_array[i,j,k]),np.log(array[i,j,k])))
                zipped_log_R_norm_potential.append((np.log(rad_array[i,j,k]), norm_array[i,j,k]))
                zipped_e_field.append((rad_array[i,j,k], E_F[i,j,k]))
                zipped_log_R_log_E.append((np.log(rad_array[i,j,k]),np.log(E_F[i,j,k])))



    # plot tuple array
    plot_tuple_array(zipped_potential, "R against Potential", "R", "Potential")


    # plot normalised tuple array
    ##plot_tuple_array(zipped_norm_potential, "R against normalised Potential", "R", "Potential")

    # plot normalised log tuple array
    ##plot_tuple_array(zipped_log_norm_potential, "R against log Potential", "R", "Log Potential")

    plot_tuple_array(zipped_logR_logV, "Log R against Potential", "LogR", "LogV")

    # FIT ACROSS RANGE THAT WILL GIVE 1/R GRADIENT, about R= 1.2 -> 2.2
    plot_tuple_array_fit(zipped_logR_logV, lin_func, 0.5, 1.5, "Fitted LogR LogV plot", "logR", "logV")


    # plot normalised tuple with log R
    #plot_tuple_array(zipped_log_R_norm_potential, "LogR against Normalised Potential", "logR", "Potential")





    plot_tuple_array(zipped_e_field, "R against E field", "R", "E_field")

    plot_tuple_array(zipped_log_R_log_E, "logR against logE field", "R", "E_field")

    # FIT ACROSS LOG R LOG V
    plot_tuple_array_fit(zipped_log_R_log_E, lin_func, 0.5, 1.5, "Fitted LogR, LogE plot", "logR", "logE")


    # save R_V_E array
    np.savetxt("RVE.txt", R_V_E_array)





    '''
    # demonstrating that radial array is correct
    print(rad_array.shape)
    print(rad_array[int(lattice_size/2)][int(lattice_size/2)].shape)

    # plotting to demonstrate that 3D radius array is accuracy
    slice = rad_array[int(lattice_size/2)][int(lattice_size/2)]
    pos_list = np.linspace(0,int(len(slice)),int(len(slice)))
    print(pos_list.shape)
    # plot across X plane
    plt.figure(figsize=(5,5))
    plt.title("MESSY")
    plt.scatter(pos_list,slice)
    plt.show()

    imsh = plt.imshow(rad_array[int(lattice_size/2)])
    plt.colorbar(imsh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Radius values across central slice")
    plt.show()
    '''

    # zip the 2 3d arrays together







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
    '''

main()
