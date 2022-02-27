import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import sys
import scipy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


from SIRS_sim import *

sys.path.append('../GameOfLife')

from gol_sim import read_file

positions = read_file("Coordinate lists")
positions = [positions[i].rstrip().lstrip() for i in range(len(positions))]
positions = [positions[i].split() for i in range(len(positions))]
positions = (np.array(positions)).astype(float)
data = read_file("Average energy lists")
data = [data[i].rstrip().lstrip() for i in range(len(data))]
data = (np.array(data)).astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.trisurf(positions[:,0], positions[:,1], data, c='r', marker='o')

#plt.show()

# Reshaping data to be acceptable
POS0 = positions[:,0].reshape(6,6)
POS1 = positions[:,1].reshape(6,6)
DATA = data.reshape(6,6)

cont = plt.contourf(POS0, POS1, DATA, cmap='RdBu')
plt.colorbar(cont)
plt.title("p1-p3 contour plot for average infected")
plt.xlabel("p1")
plt.ylabel("p3")
plt.savefig("p1p3contourplt.png")
plt.show()

imsh = plt.imshow(DATA, extent=[POS0.min(), POS0.max(), POS1.max(), POS1.min()])
plt.colorbar(imsh)
plt.xlabel("p1")
plt.ylabel("p3")
plt.title("p1-p3 imshow plot for average infected")
plt.savefig("p1p3imshowplt.png")
plt.show()
