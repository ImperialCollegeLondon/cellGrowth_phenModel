# cellGrowth_phenModel
Modelling and optimization of cell growth, based on imaging data

## source of the material
Script produced and employed in the PhD Thesis: \
Title:      "Phenomenological modelling of the fission yeast cell cycle based on multi-dimensional single-cell phenotypic data across growth conditions" \
Author:       Lorenzo Ficorella \
Institution:  Imperial College London \
date:         October 2018

## functioning
The functioning of the scripts is described in the aforementioned thesis. A link to the text will be provided as soon as it is available.\
The code works in Julia 0.6.2. I'm currently in the process of updating it to the newest stable Julia release.

## notes
Currently, the optimization function exports all matrices produced during optimization. While it is helpful for analysing how the process worked, it can become very costly in terms of allocated memory (up to the point the system might crash).
The alpha folder contains a newer (but still untested) version of the script that only employs and exports the last two rounds of the optimization process.
