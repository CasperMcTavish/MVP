import numpy as np

######################
# OBJECT CREATION
######################
# PRODUCES SPECIFIC SHAPES THAT CAN BE APPLIED DIRECTLY TO ARRAY FOR INITIAL CONDITIONS

# orientation
# graph is plotted with top left being 0,0. So m-1 goes 'up', and n+1 goes 'right'

def glider_creator(m, n, orientation, array):
    # orientation determines which way they will go
    if orientation == 0:
        array[m-1, n] = True
        array[m, n+1] = True
        array[m+1, n+1] = True
        array[m+1, n] = True
        array[m+1, n-1] = True
    elif orientation == 1:
        array[m+1, n] = True
        array[m, n-1] = True
        array[m-1, n-1] = True
        array[m-1, n] = True
        array[m-1, n+1] = True


def oscil_creator(m, n, array):
    # Coordinates for m oscillator
    array[m-1, n] = True
    array[m-2, n] = True
    array[m-1, n-1] = True
    array[m-1, n+1] = True
    array[m, n-1] = True
    array[m, n+1] = True
    array[m+1, n-1] = True
    array[m+1, n+1] = True


# Produces a little explosion that ends with an oscillation
def expl_oscil_creator(m, n, array):
    array[m+1, n] = True
    array[m-1, n] = True
    array[m, n+1] = True
    array[m, n-1] = True
    array[m+1, n+2] = True
    array[m-1, n+2] = True
    array[m+1, n-2] = True
    array[m-1, n-2] = True
    array[m, n-2] = True


def single_glider_creator(m, n, array):


    for i in range(6):
        array[m, n-1-i] = True
        array[m, n+i] = True

    for i in range(4):
        array[m-2, n-1-i] = True
        array[m-2, n+i] = True

        array[m+2, n-1-i] = True
        array[m+2, n+i] = True


    for i in range(2):
        array[m-4, n-1-i] = True
        array[m-4, n+i] = True

        array[m+4, n-1-i] = True
        array[m+4, n+i] = True


    # 3 absorbing blocks to remove extra Gliders
    array[m-8, n-7] = True
    array[m-8, n-8] = True
    array[m-9, n-7] = True
    array[m-9, n-8] = True

    array[m+8, n-7] = True
    array[m+8, n-8] = True
    array[m+9, n-7] = True
    array[m+9, n-8] = True

    array[m+8, n+7] = True
    array[m+8, n+8] = True
    array[m+9, n+7] = True
    array[m+9, n+8] = True
