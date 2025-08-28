# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume
"""

#%% INIT

# import modules
from loopUI import load_ls_gocad_voxets,plot_voxet
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import pandas as pd
# import numpy as np
from numpy.random import default_rng
from datetime import datetime
import pickle
picklefilename = "./pickledata/synthetic-case-data.pickle"
import os #,glob
if not os.path.exists('pickledata'):
    os.makedirs('pickledata')



print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"INITIALIZATION")
# initialization
myseed=12345 # repeatable demo
# filePrefix = "voxet/simpHamUncf-perturbed-100-my" # + f'{i:06d}'
clblab = 'lithocode' # colorbar label prefix
clblabsf = 'scalar-field'
slice_ix=0 # slice index for plotting
slice_iy=0 # slice index for plotting
slice_iz=7 # slice index for plotting
# nbsamples = 10 # fixed for the demonstration of indicator computations
# [nz,ny,nx]=[12,84,88]

rng = default_rng(myseed)

aspectratio = 1
#%% DOWNLOAD DATA
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"DOWNLOADING VOXETS.")
lithocode_100,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-100-my","*.vop1") 
lithocode_50A,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-50A-my","*.vop1") 
lithocode_50B,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-50B-my","*.vop1") 
scalarfield_100,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-100-sc","*.vop1") 
scalarfield_50A,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-50A-sc","*.vop1") 
scalarfield_50B,nx,ny,nz,nbsamples = load_ls_gocad_voxets("voxet/simpHamUncf-perturbed-50B-sc","*.vop1") 
print(str(nbsamples)+' models downloaded')
print('Dimensions:')
print('   nx: ',str(nx))
print('   ny: ',str(ny))
print('   nz: ',str(nz))

# CONTROL PLOT
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" lithocode_100 MODELS.")
for s in [0]: #range(nbsamples): #
    plot_voxet(lithocode_100,s,clblab,slice_ix,slice_iy,slice_iz,aspectratio)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" lithocode_50A MODELS.")
for s in [0]: #range(nbsamples): #[0]: #
    plot_voxet(lithocode_50A,s,clblab,slice_ix,slice_iy,slice_iz,aspectratio)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" lithocode_50B MODELS.")
for s in [0]: #range(nbsamples): #[0]: #
    plot_voxet(lithocode_50B,s,clblab,slice_ix,slice_iy,slice_iz,aspectratio)

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" scalarfield_100 MODELS.")
for s in [0]: #range(nbsamples): #
    plot_voxet(scalarfield_100,s,clblabsf,slice_ix,slice_iy,slice_iz,aspectratio)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" scalarfield_50A MODELS.")
for s in [0]: #range(nbsamples): #[0]: #
    plot_voxet(scalarfield_50A,s,clblabsf,slice_ix,slice_iy,slice_iz,aspectratio)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"CONTROL PLOTs OF "+str(nbsamples)+" scalarfield_50B MODELS.")
for s in [0]: #range(nbsamples): #[0]: #
    plot_voxet(scalarfield_50B,s,clblabsf,slice_ix,slice_iy,slice_iz,aspectratio)
    

#%% SAVE OUTPUT
with open(picklefilename, 'wb') as f:
    pickle.dump([lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf], f)