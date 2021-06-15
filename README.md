# uncertaintyIndicators

uncertaintyIndicators.py provides several functions to analyse the variability among an ensemble of voxets. It allows the computation of local measures of uncertainty (cardinality and entropy) and of global dissimilarity measures (e.g. based on multiple-point statistical analysis or wavelet decomposition among other possibilities).

Examples on how to use the functions are provided in a series of python notebooks. 
A pre-requisites to run the nbk-*.ipynb is to unzip models.zip in the working folder and then run example.ipynb to load an ensemble of voxets.

In order to compare the various indicator via the viewing nbk-all-indicators-comparison.ipynb notebook, a prerequisite is to run the cardinality and entropy notebooks (other pre-computed indicators are saved in the pickledata folder, but the indicators can be computed by running the appropriate notebook).
