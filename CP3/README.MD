README FOR CHECKPOINT 3

CahnHilliard
============

Contains CahnHilliard.py, which produces the different spinodel and droplet behaviours for differing values of phi0, also outputs a csv file and then plots the free energy.

Poisson
=======

Contains the jacobi algorithm for a point charge and wire, with correct plots for each

	pointcharge
	===========
	
	contains poisson.py which produces the Jacobi simulation and outputs an .npy file of the 3D potential array.
	contains analysisupdated.py which takes this .npy file and produces the relevant graphs and fits for E-field and potential. Also produces the relevant text files (Radius-Potential-ElectricField & X,Y,Potential, Ex, Ey saved in each). 
	
	wire
	====
	
	same as above, but uses B-field around wire instead
	
Seidel
======

Contains the Gauss-Seidel algorithm but otherwise identical to the Poisson folder

SOR
===

Uses Gauss-Seidel with over-relaxation to produce a plot of the iterations against omega
	
	
