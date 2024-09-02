# Fate-of-methane
A water-column model for the fate of methane from seafloor seeps. The code has been used to run simulations and generate plots for the manuscript ``Fate of dissolved methane from ocean floor seeps'', by Nordam, Dissanayake, Brakstad, Hakvåg, Øverjordet, Litzler, Nepstad, Drew & Röhrs.


## Data
The repository includes water column data for temperature, salinity and diffusivity, stored as output from GOTM. Note that the GOTM output NetCDF-files have been reduced to a manageable size by deleting superfluous variables, converting to NetCDF version 4, and adding some compression. The GOTM input files we used are also found in the repo, but it is not necessary to re-run the GOTM cases to reproduce the plots from the paper, since the GOTM output is included. Note that we used GOTM version 6.0.

The repository also contains an Excel file with raw data from methane oxidation experiments based on tritium-labelled methane.

## Code
The basic diffusion-reaction model described in the paper is found in the file ```DiffusionSolver.py```. The other files contain various wrappers used to call this model in different ways, to run ensembles and parameter scans. Additionally, the file ```TamocWrapper.py``` contains code used to call the Single Bubble Model (SBM) from Tamoc. It is necessary to install TAMOC to run the code. Note that we used TAMOC 2.4.0 for the paper.

## Notebooks
The notebooks in the repo are hopefully self-explanatory, and will generate the results presented in the paper.
