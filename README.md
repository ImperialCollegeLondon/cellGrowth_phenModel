# cellGrowth_phenModel
Modelling and optimization of cell growth, based on imaging data

Script produced and employed in the PhD Thesis:
Title:      "Phenomenological modelling of the fission yeast cell cycle based on multi-dimensional single-cell phenotypic data across growth conditions" 
Author:       Lorenzo Ficorella
Institution:  Imperial College London
date:         October 2018

Currently, the optimization function exports all matrices produced during optimization. While it is helpful for analysing how the process worked, it can become very costly in terms of allocated memory (up to the point the system might crash).
The alpha folder contains a newer (but still untested) version of the script that only employs and exports the last two rounds of the optimization process.
