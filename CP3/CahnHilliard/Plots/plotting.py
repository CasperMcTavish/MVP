import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_free_energy(file_name):
    data = np.loadtxt(file_name, dtype = float, delimiter=',')
    plt.scatter(data[:,0], data[:,1], s = 0.01)
    plt.plot(data[:,0], data[:,1])
    name = file_name + " free energy plotting"
    plt.title(name)
    plt.xlabel("Iterations")
    plt.ylabel("Free Energy")
    name = file_name + ".png"
    plt.savefig(name)
    plt.show()


# CALL FUNCTION
# check if not imported
if __name__ == "__main__":

    # Check to make sure enough arguments
    if len(sys.argv) == 2:
        # run code, force as integers
        plot_free_energy(sys.argv[1])
    else:
        print("\nScript takes exactly 1 argument, " + str(len(sys.argv)-1) + " were given")
        print("\nPlease input:\n\nFile name")
