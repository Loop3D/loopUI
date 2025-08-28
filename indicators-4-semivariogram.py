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
from loopUI import dist_experimental_variogram, mxdist_experimental_variogram

picklefilenamesemivariogram = "./pickledata/synthetic-case-semivariogram.pickle"
picklefilenamedata = "./pickledata/synthetic-case-data.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!
sample_num1 = 0
sample_num2 = 10

# 2PS based dissimilarity parameters
seed = 65432
max3Dnbsamples = int(0.6E3)
max2Dnbsamples = int(0.3E3)
pnorm = 2 
nblags=12

# import data
with open(picklefilenamedata, 'rb') as f:
    [lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,
     scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf] = pickle.load(f)

yyy,zzz,xxx=np.meshgrid(np.arange(1,ny+1),np.flip(np.arange(1,nz+1)),np.arange(1,nx+1))
xx=xxx[0,:,:]
yy=yyy[0,:,:]
zz=zzz[0,:,:]
maxh3D = np.sqrt(nx**2+ny**2+nz**2)/3
maxh2D = np.sqrt(nx**2+ny**2)/3

lithocode_all = np.reshape(np.stack((lithocode_100,lithocode_50A,lithocode_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')
scalarfield_all = np.reshape(np.stack((scalarfield_100,scalarfield_50A,scalarfield_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')
 
#%% 3D continuous test
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE DENSITY START")
img1 = scalarfield_all[:,:,:,sample_num1] # for illustration, we consider the density field of random sample #9
img2 = scalarfield_all[:,:,:,sample_num2] # for illustration, we consider the density field of random sample #2
verb=True
plot=True
label="scalar field"
dist3Dconti = dist_experimental_variogram(img1,img2,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,categ=False,label=label,verb=verb,plot=plot,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE DENSITY END")

#%% 2D continuous test
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE DENSITY START")
img1 = np.reshape(scalarfield_all[slice_iz,:,:,sample_num1],(ny,nx)) # for illustration, we consider the density field of random sample 1
img2 = np.reshape(scalarfield_all[slice_iz,:,:,sample_num2],(ny,nx)) # for illustration, we consider the density field of random sample 2
verb=True
plot=True
label="scalar field"
dist2Dconti = dist_experimental_variogram(img1,img2,xx,yy,zz,nblags,maxh2D,max2Dnbsamples,pnorm,seed,categ=False,label=label,verb=verb,plot=plot)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE DENSITY END")


#%% 3D categorical test
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE LITHOCODE START")
img1 = lithocode_all[:,:,:,sample_num1] # for illustration, we consider the density field of random sample #9
img2 = lithocode_all[:,:,:,sample_num2] # for illustration, we consider the density field of random sample #2
verb=True
plot=True
label="lithocode$"
dist3Dcateg = dist_experimental_variogram(img1,img2,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,categ=True,label=label,verb=verb,plot=plot,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE LITHOCODE END")
#%% 2D categorical test
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE LITHOCODE START")
img1 = np.reshape(lithocode_all[slice_iz,:,:,sample_num1],(ny,nx)) # for illustration, we consider the density field of random sample 1
img2 = np.reshape(lithocode_all[slice_iz,:,:,sample_num2],(ny,nx)) # for illustration, we consider the density field of random sample 2
verb=True
plot=True
label="lithocode$"
dist2Dcateg = dist_experimental_variogram(img1,img2,xx,yy,zz,nblags,maxh2D,max2Dnbsamples,pnorm,seed,categ=True,label=label,verb=verb,plot=plot)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM DISTANCE LITHOCODE END")

#%% compute for all data and sample pairs
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM BASED DIST ALL START")

# check index order 10 first _100, 10 next _50A, 10 last _50B
# from uncertaintyIndicators import plot_voxet
# for ixm in range(nbsamples*3):
#     plot_voxet(lithocode_all,ixm,'model '+str(ixm),0,0,7,1,cmap='viridis')


dist_2ps_lc = np.zeros((nbsamples*3,nbsamples*3))
dist_2ps_sf = np.zeros((nbsamples*3,nbsamples*3))

categval = np.unique(lithocode_all)
# dist_2ps_lc =  mxdist_experimental_variogram_categorical(lithocode_all,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,categval=categval,verb=True)
dist_2ps_lc =  mxdist_experimental_variogram(lithocode_all,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,categ=True,categval=categval,verb=False)
# dist_2ps_sf = mxdist_experimental_variogram_continuous(scalarfield_all,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,verb=False)
dist_2ps_sf = mxdist_experimental_variogram(scalarfield_all,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,seed,categ=False,verb=False)

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING SEMI-VARIOGRAM BASED DIST ALL END")


#%% SAVE COMPUTATIONS IN PICKLE FILE

with open(picklefilenamesemivariogram, 'wb') as f:
    pickle.dump([dist_2ps_lc,dist_2ps_sf], f)

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_2ps_lc).embedding_
mdspos_sf = mds.fit(dist_2ps_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_2ps':dist_2ps_lc[ix], 'scalarfield_2ps':dist_2ps_sf[ix]})

lcmin = np.amin(dist_2ps_lc[ix]) 
lcmax = np.amax(dist_2ps_lc[ix])
sfmin = np.amin(dist_2ps_sf[ix]) 
sfmax = np.amax(dist_2ps_sf[ix])

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
plt.title('2D MDS Representtaion of 2PS dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode 2ps', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of 2PS dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield 2ps', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_2ps)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_2ps,y=df.scalarfield_2ps)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_2ps,y=df.scalarfield_2ps)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_2ps)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")
