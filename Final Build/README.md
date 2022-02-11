# MVP

CP1_WAITON_S1739002 is the prefix to all programs.

=============================
Simulation and visualisation:
=============================

_BOILERPLATE is the basic program that allows you to define all the relevant variables, and see the simulations individually for Glauber and Kawasaki.
Run the code once and you will see the output:

==============================================
Script takes exactly 4 arguments, 0 were given

Please input:

 DYNAMIC MODEL
  0 - Glauber
  1 - Kawasaki

 LATTICE SIZE

 TEMPERATURE

 ITERATIONS
 ==============================================

 This tells you exactly how to manipulate the code as you need.

=======================
Iteration and Plotting:
=======================

Python Iterator for glauber and kawasaki are labelled FUNC_glauber and FUNC_kawasaki. These run across T = 1.0 to T = 3.0 with 0.1 steps, 10000 iterations per temperature.
They then output text files and graphs of the results.
Glauber includes Energy, Heat Capacity (with errors), Magnetism and Susceptibility.
Kawasaki includes Energy and Heat Capacity (with errors).

These can be found separately in the 'GLAUBER FILES' and 'KAWASAKI FILES' sections.
