# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume Pirot
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
from loopUI import jsdist_hist #, plot_voxet

picklefilenamehistogram = "./pickledata/synthetic-case-histogram.pickle"
picklefilenamedata = "./pickledata/synthetic-case-data.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!
sample_num1 = 9
sample_num2 = 2

# hist based dissimilarity parameters
myseed = 65432
nbins = 20
base = np.e


# import data
with open(picklefilenamedata, 'rb') as f:
    [lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,
     scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf] = pickle.load(f)


#%% histogram based distance

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING HISTOGRAM BASED DENSITY START")
img1 = lithocode_100[:,:,:,sample_num1] # for illustration, we consider the density field of random sample #9
img2 = lithocode_100[:,:,:,sample_num2] # for illustration, we consider the density field of random sample #2
dist_hist_lc = jsdist_hist(img1,img2,-1,base,plot=True,title="lithocodes",lab1="model 1",lab2="model 2",iz_section=7)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING HISTOGRAM BASED DENSITY END")

#%% compute for all data and sample pairs
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING HISTOGRAM BASED DIST ALL START")

lithocode_all = np.reshape(np.stack((lithocode_100,lithocode_50A,lithocode_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')
scalarfield_all = np.reshape(np.stack((scalarfield_100,scalarfield_50A,scalarfield_50B),axis=4),(nz,ny,nx,nbsamples*3),order='F')

# check index order 10 first _100, 10 next _50A, 10 last _50B
# for ixm in range(nbsamples*3):
#     plot_voxet(lithocode_all,ixm,'model '+str(ixm),0,0,7,1,cmap='viridis')


dist_hist_lc = np.zeros((nbsamples*3,nbsamples*3))
dist_hist_sf = np.zeros((nbsamples*3,nbsamples*3))

k=0
for i in range(nbsamples*3):
    for j in range(i):
        k+=1
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+'k = '+str(k)+' - i = '+str(i)+' j = ',str(j))
        dist_hist_lc[i,j] = jsdist_hist(lithocode_all[:,:,:,i],lithocode_all[:,:,:,j],-1,base)
        dist_hist_sf[i,j] = jsdist_hist(scalarfield_all[:,:,:,i],scalarfield_all[:,:,:,j],nbins,base)
        dist_hist_lc[j,i] = dist_hist_lc[i,j]
        dist_hist_sf[j,i] = dist_hist_sf[i,j]
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING HISTOGRAM BASED DIST ALL END")

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=myseed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_hist_lc).embedding_
mdspos_sf = mds.fit(dist_hist_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_hist':dist_hist_lc[ix], 'scalarfield_hist':dist_hist_sf[ix]})

lcmin = np.amin(dist_hist_lc[ix]) 
lcmax = np.amax(dist_hist_lc[ix])
sfmin = np.amin(dist_hist_sf[ix]) 
sfmax = np.amax(dist_hist_sf[ix])

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
plt.title('2D MDS Representation of hist. dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode hist', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representation of hist. dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield hist', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_hist)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_hist,y=df.scalarfield_hist)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_hist,y=df.scalarfield_hist)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_hist)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")
#%% save computations as pickle file

with open(picklefilenamehistogram, 'wb') as f:
    pickle.dump([dist_hist_lc,dist_hist_sf], f)
