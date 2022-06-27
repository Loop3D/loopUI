# loopUI
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Loop3D/uncertaintyIndicators/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/377036385.svg)](https://zenodo.org/badge/latestdoi/377036385)

The *loopUI Python package* provides several functions to analyse the variability among an ensemble of voxets. It allows the computation of local measures of uncertainty (cardinality and entropy) and of global dissimilarity measures (e.g. based on multiple-point statistical analysis or wavelet decomposition among other possibilities).


## Installation
*Note: package for python >=3.7*

To install the package: `python -m pip install loopui`

To uninstall the package: `python -m pip uninstall -y loopui`


## Requirements
The following python packages are used by 'geone':
   - matplotlib
   - mpl_toolkits
   - numpy
   - gzip
   - scipy
   - sklearn
   - pywt
   - pandas


## Examples
A series of *Python* notebooks demonstrate how to use the different functions. 
First, run `ui-0-data.ipynb` to load an ensemble of voxets. It is a pre-requisites prior running the other python notebooks. 

In order to compare the various indicator via the viewing `ui-9-comparison.ipynb` notebook, a prerequisite is to run the cardinality and entropy notebooks (other pre-computed indicators are saved in the pickledata folder, but the indicators can be computed by running the appropriate notebook).
