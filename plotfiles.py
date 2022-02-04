# this file will plot the relevant values
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import math
import random
import sys


# Take multiple data files from the folder and take average for each point, then standard error on mean
def read_file(filename):
    with open(filename) as f:
        contents = f.readlines()

    if (len(contents) > 1):
        print("File read of length: " + str(len(contents)))
    else:
        print("File length is too short to process (please include more positions)")

    # Specialised case, dealing with floats here. So convert from string to float
    contents = [float(contents[i]) for i in range(len(contents))]
    return contents

# create function that takes folder name (FULL GLAUBER 1) and takes the av_cap.txt, av_en.txt etc and pushes to list
# THIS ONLY WORKS HERE IN THIS FOLDER STRUCTURE, SO DONT CHANGE IT
def pull_data(foldername):

    # Generic file path for all files
    file_path = "Previous RUNS/" + str(foldername) + "/"

    heat_capacity_data = read_file(file_path+str("av_cap.txt"))
    energy_data = read_file(file_path+str("av_en.txt"))
    suscept_data = read_file(file_path+str("av_sus.txt"))
    mag_data = read_file(file_path+str("av_mag.txt"))

    return mag_data, energy_data, suscept_data, heat_capacity_data

#def av_std():


# Collect the relevant data from each folder, once they exist
#mag_1, en_1, sus_1, hc_1 = pull_data("FULL GLAUBER 1")
run = []
run.append(pull_data("FULL GLAUBER 1"))
run.append(pull_data("FULL GLAUBER 2"))


# Take the average and standard deviation of each point and pass it back
#av = [[],[],[],[]]
#std = [[],[],[],[]]

#for i in range(len(run)):
    # average each point for each value, first we'll do magnetism
    # first [] defines the run number
    # second [] defines the value we're considering
    # 0 -> mag, 1 -> energy, 2 -> sus, 3 -> heat_cap
    # third [] defines which point within this list

    # This loop runs over mag, energy, etc
    #for j in range(len(run[i])):
    #    for k in range(len([i][j])):
    #        av[j].append()


    #run[i][0]
print(run[0][0][0])



#for i in range(len(run[0][0])):
    #point = []
    # set inner list to number of full runs
    #for j in range(2):
        # taking nth element of each list (mag, en, sus, heat_cap, etc)
#        point.append([item[i] for item in run[j]])



#df = pd.DataFrame( , columns = list("mag, en, sus, heat_cap"), index = ["R1","R2"])
#df
