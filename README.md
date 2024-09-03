# OpenRANet: Neuralized Spectrum Access by Joint Subcarrier and Power Allocation with Optimization-based Deep Learning

## Introdunction

This repository addresses a non-convex problem involving joint subcarrier and power control, aiming to minimize total power while satisfying rate requirements. The complexity stems from non-convexity, coupled constraints, and implicit resource uncertainties. The code primarily implements the reweighted primal-dual algorithm for achieving local optimality and the OpenRANet algorithm for approximating global optimal solutions under varying transmission rate constraints.

Experiments are conducted on a 64-bit workstation running Windows 10, equipped with an Intel(R) Core(TM) i7-8700K CPU at 3.70GHz and 32.00GB of RAM, utilizing Python 3.7 and PyTorch 1.11.0.

## Explaination

Toy examples of problem instances, along with code for generating data for training and validating the algorithms, are located in the “data_generation” folder.

The file “iteration_solver.py” implements the reweighted primal-dual algorithm for local optimality, while “OpenRANet.py” contains the implementation for constructing the OpenRANet.


