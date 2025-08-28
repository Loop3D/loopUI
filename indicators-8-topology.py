# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume PIROT
"""

# import modules
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import manifold
import pickle
from loopUI import topo_dist

picklefilenametopology = "./pickledata/synthetic-case-topology.pickle"
picklefilenamedata = "./pickledata/synthetic-case-data.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!
sample_num1 = 0
sample_num2 = 10

# WVT based dissimilarity parameters
seed = 65432
pnorm = 2 
# number of classes for discretization of continuous fields
npctiles = 10

# import data
with open(picklefilenamedata, 'rb') as f:
    [lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,
     scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf] = pickle.load(f)

yyy,zzz,xxx=np.meshgrid(np.arange(1,ny+1),np.flip(np.arange(1,nz+1)),np.arange(1,nx+1))
xx=xxx[slice_iz,:,:]
yy=yyy[slice_iz,:,:]
zz=zzz[slice_iz,:,:]
maxh3D = np.sqrt(nx**2+ny**2+nz**2)/3
maxh2D = np.sqrt(nx**2+ny**2)/3

lithocode_all = np.reshape(np.stack((lithocode_100,lithocode_50A,lithocode_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')
scalarfield_all = np.reshape(np.stack((scalarfield_100,scalarfield_50A,scalarfield_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')

# load classes
categval = np.unique(lithocode_all)

# Hamming distance (low-scale):
# Part IV Chapter 15 of the Encyplopediae of Distances (Deza & Deza 2016) p 301
# https://doi.org/10.1613/jair.1061 [Acid & de Campos, 2003]
# https://doi.org/10.1214/18-AOAS1176 [Donnat & Holmes, 2018]

# Graph edit distance: NP-complete class of problems, not computed here
# https://en.wikipedia.org/wiki/Graph_edit_distance 
# Part IV Chapter 15 of the Encyplopediae of Distances (Deza & Deza 2016) p 301
# existin packages and implementations (not checked):
#    https://github.com/Jacobe2169/GMatch4py
#    https://pypi.org/project/graphkit-learn/
# or heuristics: 
#    https://hal.archives-ouvertes.fr/hal-01717709/document

# HIM(meso-scale) (hamming IM combination) [Donnat & Holmes, 2018]

# spectral distances (high-level)
# https://doi.org/10.1214/18-AOAS1176 [Donnat & Holmes, 2018] IM, Laplacian..
# https://doi.org/10.1038/srep34944 [Shimada et al., 2016]

#%% test functions

#%% 2D CATEGORICAL VARIABLE 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D 1st order TOPOLOGY IND. LITHOCODE START")
img1=lithocode_all[slice_iz,:,:,sample_num1]
img2=lithocode_all[slice_iz,:,:,sample_num2]
# we assume that we are not comparing carrots and potatoes, 
# i.e., that lithocodes values overlap between img1 and img2
# we also assume edge (in 2D) or face (in 3D) type neighborhood
shd_2Dlitho,lsgd2Dlitho = topo_dist(img1, img2, npctiles=0, verb=1, plot=1, leg="litho")
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D 1st order TOPOLOGY IND. LITHOCODE END")

#%% 2D CONTINUOUS VARIABLE 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D 1st order TOPOLOGY IND. MAG START")
img1=scalarfield_all[slice_iz,:,:,sample_num1]
img2=scalarfield_all[slice_iz,:,:,sample_num2]
shd_2Dscal,lsgd2Dscal = topo_dist(img1, img2, npctiles, verb=1, plot=1, leg="scalar-field")
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D 1st order TOPOLOGY IND. MAG END")

#%% 3D CATEGORICAL VARIABLE 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 3D 1st order TOPOLOGY IND. LITHOCODE START")
img1=np.reshape(lithocode_all[slice_iz:,:,:,sample_num1].astype(int),(nz,ny,nx))
img2=np.reshape(lithocode_all[slice_iz:,:,:,sample_num2].astype(int),(nz,ny,nx))
shd_3Dlitho,lsgd3Dlitho = topo_dist(img1, img2, npctiles=0, verb=1, plot=1, leg="litho")
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 3D 1st order TOPOLOGY IND. LITHOCODE END")

#%% 3D CONTINUOUS VARIABLE 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 3D 1st order TOPOLOGY IND. DENSITY START")
img1=scalarfield_all[slice_iz:,:,:,sample_num1]
img2=scalarfield_all[slice_iz:,:,:,sample_num2]
shd_3Dscal,lsgd3Dscal = topo_dist(img1, img2, npctiles, verb=1, plot=1, leg="scalar-field")
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 3D 1st order TOPOLOGY IND. DENSITY END")


#%% compute for all data and sample pairs
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING TOPOLOGY BASED DIST ALL START")

dist_tpl_shd_lc = np.zeros((3*nbsamples,3*nbsamples))
dist_tpl_shd_sf = np.zeros((3*nbsamples,3*nbsamples))
dist_tpl_lsgd_lc = np.zeros((3*nbsamples,3*nbsamples))
dist_tpl_lsgd_sf = np.zeros((3*nbsamples,3*nbsamples))

k=0
for i in range(3*nbsamples):
    for j in range(i):
        k+=1
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+'k = '+str(k)+' - i = '+str(i)+' j = ',str(j))
        dist_tpl_shd_lc[i,j],dist_tpl_lsgd_lc[i,j] = topo_dist(lithocode_all[slice_iz:,:,:,i],lithocode_all[slice_iz:,:,:,j], npctiles=0)
        dist_tpl_shd_sf[i,j],dist_tpl_lsgd_sf[i,j] = topo_dist(scalarfield_all[slice_iz:,:,:,i],scalarfield_all[slice_iz:,:,:,j], npctiles)
        dist_tpl_shd_lc[j,i] = dist_tpl_shd_lc[i,j]
        dist_tpl_shd_sf[j,i] = dist_tpl_shd_sf[i,j]
        dist_tpl_lsgd_lc[j,i] = dist_tpl_lsgd_lc[i,j]
        dist_tpl_lsgd_sf[j,i] = dist_tpl_lsgd_sf[i,j]

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING TOPOLOGY BASED DIST ALL END")

#%% SAVE COMPUTATIONS
with open(picklefilenametopology, 'wb') as f:
    pickle.dump([dist_tpl_shd_lc,dist_tpl_shd_sf,dist_tpl_lsgd_lc,dist_tpl_lsgd_sf], f)

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_tpl_shd_lc).embedding_
mdspos_sf = mds.fit(dist_tpl_shd_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_tpl_shd':dist_tpl_shd_lc[ix], 'scalarfield_tpl_shd':dist_tpl_shd_sf[ix]})

lcmin = np.amin(dist_tpl_shd_lc[ix]) 
lcmax = np.amax(dist_tpl_shd_lc[ix])
sfmin = np.amin(dist_tpl_shd_sf[ix]) 
sfmax = np.amax(dist_tpl_shd_sf[ix])

lcMDSxmin = np.min(mdspos_lc[:,0])
lcMDSxmax = np.max(mdspos_lc[:,0])
lcMDSymin = np.min(mdspos_lc[:,1])
lcMDSymax = np.max(mdspos_lc[:,1])

sfMDSxmin = np.min(mdspos_sf[:,0])
sfMDSxmax = np.max(mdspos_sf[:,0])
sfMDSymin = np.min(mdspos_sf[:,1])
sfMDSymax = np.max(mdspos_sf[:,1])

s = 100
fig = plt.figure()
plt.subplot(231)
plt.title('2D MDS Representtaion of tpl_shd dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode tpl_shd', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of tpl_shd dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield tpl_shd', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_tpl_shd)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_tpl_shd,y=df.scalarfield_tpl_shd)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_tpl_shd,y=df.scalarfield_tpl_shd)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_tpl_shd)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION INTERMEDIATE")

mdspos_lc = mds.fit(dist_tpl_lsgd_lc).embedding_
mdspos_sf = mds.fit(dist_tpl_lsgd_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_tpl_lsgd':dist_tpl_lsgd_lc[ix], 'scalarfield_tpl_lsgd':dist_tpl_lsgd_sf[ix]})

lcmin = np.amin(dist_tpl_lsgd_lc[ix]) 
lcmax = np.amax(dist_tpl_lsgd_lc[ix])
sfmin = np.amin(dist_tpl_lsgd_sf[ix]) 
sfmax = np.amax(dist_tpl_lsgd_sf[ix])

lcMDSxmin = np.min(mdspos_lc[:,0])
lcMDSxmax = np.max(mdspos_lc[:,0])
lcMDSymin = np.min(mdspos_lc[:,1])
lcMDSymax = np.max(mdspos_lc[:,1])

sfMDSxmin = np.min(mdspos_sf[:,0])
sfMDSxmax = np.max(mdspos_sf[:,0])
sfMDSymin = np.min(mdspos_sf[:,1])
sfMDSymax = np.max(mdspos_sf[:,1])

s = 100
fig = plt.figure()
plt.subplot(231)
plt.title('2D MDS Representtaion of tpl_lsgd dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode tpl_lsgd', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of tpl_lsgd dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield tpl_lsgd', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_tpl_lsgd)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_tpl_lsgd,y=df.scalarfield_tpl_lsgd)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_tpl_lsgd,y=df.scalarfield_tpl_lsgd)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_tpl_lsgd)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")