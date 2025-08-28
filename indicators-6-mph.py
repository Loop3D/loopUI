# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume Pirot
"""
#%% Multiple-Point Histogram, multi-level and depth level
# The idea is to use pattern representatives or clusters, which enables assessing 
# the dissimilaruty of these representatives based on patterns, size and homogeneity/heterogeneity
# This probably requires subsampling to contain the nb of pattern-pairs used to compute dissimilarities
# possible clustering algorithms (from https://www.analytixlabs.co.in/blog/types-of-clustering-algorithms/):
# k-Means Clustering - scikit-learn cluster module provides the KMeans function. (sklearn.cluster.KMeans)
# Fuzzy C Means Algorithm - fuzzy clustering can be performed using the cmeans() function from skfuzzy module. (skfuzzy.cmeans) and further, it can be adapted to be applied on new data using the predictor function (skfuzzy.cmeans_predict)
# Gaussian Mixed Models (GMM) - implenteded via the GaussianMixture() function from scikit-learn. (sklearn.mixture.GaussianMixture) 

# import modules
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import manifold
import pickle
from loopUI import dist_kmeans_mph

picklefilenamemph = "./pickledata/synthetic-case-mph.pickle"
picklefilenamedata = "./pickledata/synthetic-case-data.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!
sample_num1 = 0
sample_num2 = 10

# MPH based dissimilarity parameters
seed = 65432
n_clusters=10
nmax_patterns = int(1E4)
pattern2Dsize = np.asarray([3,4])
pattern3Dsize = np.asarray([1,2,3])

# import data
with open(picklefilenamedata, 'rb') as f:
    [lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,
     scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf] = pickle.load(f)

# initialization
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

# maximum number of upscaling levels
max2Dlevels = np.min([np.floor(np.log(ny/(pattern2Dsize[0]+n_clusters**(1/2))) / np.log(2) ),
                      np.floor(np.log(nx/(pattern2Dsize[1]+n_clusters**(1/2))) / np.log(2) )]).astype(int)
freedomX = nx/2**(max2Dlevels+1)-pattern2Dsize[1]
freedomY = ny/2**(max2Dlevels+1)-pattern2Dsize[0]
if freedomX*freedomY>=n_clusters:
    max2Dlevels +=1
    
max3Dlevels = np.min([np.floor(np.log(nz/(pattern3Dsize[0]+n_clusters**(1/3)))/np.log(2)),
                      np.floor(np.log(ny/(pattern3Dsize[1]+n_clusters**(1/3)))/np.log(2)),
                      np.floor(np.log(nx/(pattern3Dsize[2]+n_clusters**(1/3)))/np.log(2))]).astype(int)
freedomX = nx/2**(max3Dlevels+1)-pattern3Dsize[2]
freedomY = ny/2**(max3Dlevels+1)-pattern3Dsize[1]
freedomZ = nz/2**(max3Dlevels+1)-pattern3Dsize[0]
if freedomX*freedomY*freedomZ>=n_clusters:
    max3Dlevels +=1


#%% JUST TO CHECK 2D REAL SHARING SOME RANGE
# tmp=np.reshape(grv_all,(ny*nx,nbsamples))
# plt.figure()
# plt.hist(tmp, density=True, histtype='bar',label=['0','1','2','3','4','5','6','7','8','9'])
# plt.legend()
# plt.show()

#%% test MPH based distance on 2D continous

img1 = scalarfield_all[slice_iz,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = scalarfield_all[slice_iz,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

n_levels=max2Dlevels+1
patternsize=pattern2Dsize

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 2D START")
d = dist_kmeans_mph(img1,img2,n_levels,patternsize,n_clusters,nmax_patterns,seed,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 2D END")

#%% test MPH based distance on 2D categorical

img1 = lithocode_all[slice_iz,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = lithocode_all[slice_iz,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

n_levels=max2Dlevels+1
patternsize=pattern2Dsize

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 2D categ START")
d = dist_kmeans_mph(img1,img2,n_levels,patternsize,n_clusters,nmax_patterns,seed,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 2D categ END")


#%% test MPH based distance on 3D continous

img1 = scalarfield_all[slice_iz:,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = scalarfield_all[slice_iz:,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

n_levels=max3Dlevels
patternsize=pattern3Dsize

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 3D START")
d = dist_kmeans_mph(img1,img2,n_levels,patternsize,n_clusters,nmax_patterns,seed,plot=False,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 3D END")

#%% test MPH based distance on 3D categorical

img1 = lithocode_all[slice_iz:,:,:,sample_num1] # for illustration, we consider the magnetic response of random sample #9
img2 = lithocode_all[slice_iz:,:,:,sample_num2] # for illustration, we consider the magnetic response of random sample #2

n_levels=max3Dlevels
patternsize=pattern3Dsize

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 3D categ START")
d = dist_kmeans_mph(img1,img2,n_levels,patternsize,n_clusters,nmax_patterns,seed,plot=False,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING k-means clustered MPH BASED DIST 3D categ END")


#%% compute for all data and sample pairs
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING MULTIPLE-POINT HISTOGRAM BASED DIST ALL START")

dist_mph_lc = np.zeros((3*nbsamples,3*nbsamples))
dist_mph_sf = np.zeros((3*nbsamples,3*nbsamples))

k=0
for i in range(3*nbsamples):
    for j in range(i):
        k+=1
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+'k = '+str(k)+' - i = '+str(i)+' j = ',str(j))
        dist_mph_lc[i,j] = dist_kmeans_mph(lithocode_all[slice_iz:,:,:,i],lithocode_all[slice_iz:,:,:,j],max3Dlevels,pattern3Dsize,n_clusters,nmax_patterns,seed)
        dist_mph_sf[i,j] = dist_kmeans_mph(scalarfield_all[slice_iz:,:,:,i],scalarfield_all[slice_iz:,:,:,j],max3Dlevels,pattern3Dsize,n_clusters,nmax_patterns,seed)
        dist_mph_lc[j,i] = dist_mph_lc[i,j]
        dist_mph_sf[j,i] = dist_mph_sf[i,j]

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING MULTIPLE-POINT HISTOGRAM BASED DIST ALL END")

#%% SAVE COMPUTATIONS
with open(picklefilenamemph, 'wb') as f:
    pickle.dump([dist_mph_lc,dist_mph_sf], f)

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_mph_lc).embedding_
mdspos_sf = mds.fit(dist_mph_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_mph':dist_mph_lc[ix], 'scalarfield_mph':dist_mph_sf[ix]})

lcmin = np.amin(dist_mph_lc[ix]) 
lcmax = np.amax(dist_mph_lc[ix])
sfmin = np.amin(dist_mph_sf[ix]) 
sfmax = np.amax(dist_mph_sf[ix])

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
plt.title('2D MDS Representtaion of mph dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode mph', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of mph dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield mph', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_mph)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_mph,y=df.scalarfield_mph)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_mph,y=df.scalarfield_mph)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_mph)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")


