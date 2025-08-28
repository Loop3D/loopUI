# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume Pirot
"""

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import manifold
import pickle
from loopUI import dist_wavelet

picklefilenamewavelet = "./pickledata/synthetic-case-wavelet.pickle"
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
max3Dnbsamples = int(0.6E3)
max2Dnbsamples = int(0.3E3)
pnorm = 2 
n_levels=4
n_bins=20

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

#%% test wavelet based distance on 2D continous

img1 = scalarfield_all[slice_iz,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = scalarfield_all[slice_iz,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 2D continuous START")
d = dist_wavelet(img1,img2,n_levels,n_bins,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 2D continuous END")

#%% test wavelet based distance on 2D categorical
img1 = lithocode_all[slice_iz,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = lithocode_all[slice_iz,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 2D categ START")
d = dist_wavelet(img1,img2,n_levels,n_bins,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 2D categ END")

#%% test wavelet based distance on 3D continuous

img1 = scalarfield_all[slice_iz:,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = scalarfield_all[slice_iz:,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

# ADD section plot of 3D images

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 3D continuous START")
d = dist_wavelet(img1,img2,n_levels,n_bins,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 3D continuous END")

#%% test wavelet based distance on 3D categorical
img1 = lithocode_all[slice_iz:,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = lithocode_all[slice_iz:,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 3D categ START")
d = dist_wavelet(img1,img2,n_levels,n_bins,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST 3D categ END")

#%% Compute wvt-based distances for all continuous and categorical variables for the subset of models
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST ALL START")

dist_wvt_lc = np.zeros((3*nbsamples,3*nbsamples))
dist_wvt_sf = np.zeros((3*nbsamples,3*nbsamples))

k=0
for i in range(3*nbsamples):
    for j in range(i):
        k+=1
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+'k = '+str(k)+' - i = '+str(i)+' j = ',str(j))
        dist_wvt_lc[i,j] = dist_wavelet(lithocode_all[slice_iz:,:,:,i],lithocode_all[slice_iz:,:,:,j],n_levels,n_bins)
        dist_wvt_sf[i,j] = dist_wavelet(scalarfield_all[slice_iz:,:,:,i],scalarfield_all[slice_iz:,:,:,j],n_levels,n_bins)
        dist_wvt_lc[j,i] = dist_wvt_lc[i,j]
        dist_wvt_sf[j,i] = dist_wvt_sf[i,j]

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING WAVELET BASED DIST ALL END")

#%% SAVE COMPUTATIONS
with open(picklefilenamewavelet, 'wb') as f:
    pickle.dump([dist_wvt_lc,dist_wvt_sf], f)

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_wvt_lc).embedding_
mdspos_sf = mds.fit(dist_wvt_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_wvt':dist_wvt_lc[ix], 'scalarfield_wvt':dist_wvt_sf[ix]})

lcmin = np.amin(dist_wvt_lc[ix]) 
lcmax = np.amax(dist_wvt_lc[ix])
sfmin = np.amin(dist_wvt_sf[ix]) 
sfmax = np.amax(dist_wvt_sf[ix])

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
plt.title('2D MDS Representtaion of wvt dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode wvt', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of wvt dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield wvt', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_wvt)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_wvt,y=df.scalarfield_wvt)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_wvt,y=df.scalarfield_wvt)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_wvt)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")
