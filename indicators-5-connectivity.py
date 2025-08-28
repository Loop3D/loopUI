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
from loopUI import dist_lpnorm_percentile_global_connectivity, dist_lpnorm_categorical_lag_connectivity, dist_lpnorm_percentile_lag_connectivity 
from loopUI import mxdist_lpnorm_categorical_lag_connectivity,mxdist_lpnorm_percentile_global_connectivity

picklefilenameconnectivity = "./pickledata/synthetic-case-connectivity.pickle"
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
npctiles=20
nblags=10

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

#%% Figures for paper
from loopUI import continuous_pct_connectivity,indicator_lag_connectivity #, plot_voxet 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

lithocode = 6  # 1 or 4

img_litho = lithocode_all[slice_iz,:,:,sample_num1].astype(int)
img_categ = (1*(img_litho==lithocode)).astype(int) # 1 or 4
img_conti = (scalarfield_all[slice_iz,:,:,sample_num1])
# plot_voxet(img_categ,-1,"lithocode",slice_ix,slice_iy,slice_iz)
# # 2D image
# ax = plt.subplot()
# im = ax.imshow(img_conti)
# plt.title("magnetic field anomalies")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()

lag_center,lag_count,lag_proba = indicator_lag_connectivity(img_categ,xx,yy,zz,nblags,maxh2D,max2Dnbsamples,clblab='lithocode',verb=True)

npctiles=50
low_connect,hig_connect,pctiles = continuous_pct_connectivity(img_conti,npctiles,verb=True)

# Figure indicator lag connectivity tau
fig = plt.figure()
gs = fig.add_gridspec(1,5)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2:])
axins = inset_axes(ax01,
                    width="10%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 50%
                    loc='center left'
                    )
ax00.axis('off')
ax01.axis('off')
pos00 = ax00.imshow(img_categ[:,:])
ax00.set_title('Map view')
fig.colorbar(pos00,cax=axins,label="lithocode "+str(lithocode))
ax02.plot(lag_center,lag_proba,'-b')
ax02.set_xlabel("lag distance [px]")
ax02.set_ylabel("$\\tau (h)$ probability")
ax02.set_title("Lithocode "+str(lithocode)+ " connectivity")
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.6, wspace=0.1, hspace=0.5)
plt.show()

# Figure continuous global Gamma connectivity

th_val = np.nanpercentile(img_conti,[10,30,50,70,90])
img10 = (1*(img_conti>=th_val[0])).astype(int)
img30 = (1*(img_conti>=th_val[1])).astype(int)
img50 = (1*(img_conti>=th_val[2])).astype(int)
img70 = (1*(img_conti>=th_val[3])).astype(int)
img90 = (1*(img_conti>=th_val[4])).astype(int)

fig = plt.figure()
gs = fig.add_gridspec(2,5)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2:])
axins = inset_axes(ax01,
                    width="10%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 50%
                    loc='center left'
                    )
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax12 = fig.add_subplot(gs[1, 2])
ax13 = fig.add_subplot(gs[1, 3])
ax14 = fig.add_subplot(gs[1, 4])
ax00.axis('off')
ax01.axis('off')
ax10.axis('off')
ax11.axis('off')
ax12.axis('off')
ax13.axis('off')
ax14.axis('off')
pos00 = ax00.imshow(img_conti[:,:])
ax00.set_title('Map view')
fig.colorbar(pos00,cax=axins,label="scalar-field")
ax02.plot(pctiles,hig_connect,'-b')
ax02.plot(pctiles,low_connect,'--r')
ax02.set_xlabel("percentile threshold [p]")
ax02.set_ylabel("$\\Gamma (p)$ probability")
ax02.set_title("Thresholded scalar-field $\Gamma$ connectivity")
ax02.legend(('$\\Gamma (p)$ above threshold', '$\\Gamma^c (p)$ below threshold'),loc='best')
ax10.imshow(img10[:,:]),ax10.set_title("$p=10%$")
ax11.imshow(img30[:,:]),ax11.set_title("$p=30%$")
ax12.imshow(img50[:,:]),ax12.set_title("$p=50%$")
ax13.imshow(img70[:,:]),ax13.set_title("$p=70%$")
ax14.imshow(img90[:,:]),ax14.set_title("$p=90%$")
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.2, wspace=0.1, hspace=0.5)
plt.show()


#%% 3D INDICATOR CONNECTIVITY

# Categorical connectivity: probability that 2 points of the same class are connected, as a function of lag distance
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING INDICATOR CONNECTIVITY LITHOCODE 3D START")

img1=lithocode_all[:,:,:,sample_num1].astype(int)
img2=lithocode_all[:,:,:,sample_num2].astype(int)
clblab='lithocode'
dist_lpnorm_categorical_lag_connectivity(img1,img2,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,clblab=clblab,plot=True,verb=True,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING INDICATOR CONNECTIVITY LITHOCODE 3D END")


#%% 2D INDICATOR CONNECTIVITY
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING INDICATOR CONNECTIVITY LITHOCODE 2D START")
img1=np.reshape(lithocode_all[slice_iz,:,:,sample_num1].astype(int),(ny,nx))
img2=np.reshape(lithocode_all[slice_iz,:,:,sample_num2].astype(int),(ny,nx))
clblab='lithocode'
dist_lpnorm_categorical_lag_connectivity(img1,img2,xx,yy,zz,nblags,maxh2D,max2Dnbsamples,pnorm,clblab=clblab,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING INDICATOR CONNECTIVITY LITHOCODE 2D END")


#%% Continuous - discretization e.g. using 3 classes and 30th-70th percentile boundaries
# then applying indicator_lag_connectivity to each class

#%% Continuous - threshold connectivity as percentile of empirical distribution
# the threshold defines higher and lower subset of points, connectivity as func of lag dist. 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING percentile lag continuous CONNECTIVITY 2D START")
img1=np.reshape(scalarfield_all[slice_iz,:,:,sample_num1],(ny,nx))
img2=np.reshape(scalarfield_all[slice_iz,:,:,sample_num2],(ny,nx))
# xxx=xx
# yyy=yy
# zzz=zz
# maxh=maxh2D
# maxnbsamples=max2Dnbsamples
clblab='scalarfield'
dist_lpnorm_percentile_lag_connectivity(img1,img2,xx,yy,zz,npctiles,nblags,maxh2D,max2Dnbsamples,pnorm,clblab=clblab,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING percentile lag continuous CONNECTIVITY 2D END")

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING percentile lag continuous CONNECTIVITY 3D START")
img1=scalarfield_all[:,:,:,sample_num1]#.astype(int)
img2=scalarfield_all[:,:,:,sample_num2]#.astype(int)
clblab='scalarfield'
# maxh=maxh3D
# maxnbsamples=max3Dnbsamples
dist_lpnorm_percentile_lag_connectivity(img1,img2,xxx,yyy,zzz,npctiles,nblags,maxh3D,max3Dnbsamples,pnorm,clblab=clblab,plot=True,verb=True,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING percentile lag continuous CONNECTIVITY 3D END")

#%% Continuous global connectivity

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING Global CONTINUOUS CONNECTIVITY 2D")
img1=np.reshape(scalarfield_all[slice_iz,:,:,sample_num1],(ny,nx))
img2=np.reshape(scalarfield_all[slice_iz,:,:,sample_num2],(ny,nx))
clblab='scalarfield'

dist_lpnorm_percentile_global_connectivity(img1,img2,npctiles,pnorm,clblab=clblab,plot=True,verb=True)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING Global CONTINUOUS CONNECTIVITY 2D")


print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING Global CONTINUOUS CONNECTIVITY 3D")
img1=scalarfield_all[:,:,:,sample_num1]#.astype(int)
img2=scalarfield_all[:,:,:,sample_num2]#.astype(int)
clblab="scalarfield"

dist_lpnorm_percentile_global_connectivity(img1,img2,npctiles,pnorm,clblab=clblab,plot=True,verb=True,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING Global CONTINUOUS CONNECTIVITY 3D")

#%% compute for all data and sample pairs
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING CONNECTIVITY BASED DIST ALL START")

dist_cty_lc = np.zeros((nbsamples,nbsamples))
dist_cty_sf = np.zeros((nbsamples,nbsamples))

dist_cty_lc = mxdist_lpnorm_categorical_lag_connectivity(lithocode_all,categval,xxx,yyy,zzz,nblags,maxh3D,max3Dnbsamples,pnorm,verb=True)
dist_cty_sf = mxdist_lpnorm_percentile_global_connectivity(scalarfield_all,npctiles,pnorm,verb=True)
        
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING CONNECTIVITY BASED DIST ALL END")

#%% SAVE COMPUTATIONS
with open(picklefilenameconnectivity, 'wb') as f:
    pickle.dump([dist_cty_lc,dist_cty_sf], f)

#%% compute MDS representation for all variables
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION START")
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

mdspos_lc = mds.fit(dist_cty_lc).embedding_
mdspos_sf = mds.fit(dist_cty_sf).embedding_

s_id = np.arange(nbsamples*3)
# Plot concentric circle dataset
colors1 = plt.cm.Blues(np.linspace(0., 1, 512))
colors2 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 512)))
colors3 = plt.cm.Reds(np.linspace(0, 1, 512))
colors = np.vstack((colors1, colors2, colors3))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

ix=np.tril_indices(nbsamples*3,k=-1)
df= pd.DataFrame({'lithocodes_cty':dist_cty_lc[ix], 'scalarfield_cty':dist_cty_sf[ix]})

lcmin = np.amin(dist_cty_lc[ix]) 
lcmax = np.amax(dist_cty_lc[ix])
sfmin = np.amin(dist_cty_sf[ix]) 
sfmax = np.amax(dist_cty_sf[ix])

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
plt.title('2D MDS Representtaion of CTY dissimilarities')
plt.scatter(mdspos_lc[:, 0], mdspos_lc[:, 1], c=s_id,cmap=mycmap, s=s, label='lithocode cty', marker='+')
plt.xlim(lcMDSxmin,lcMDSxmax)
plt.ylim(lcMDSymin,lcMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(234)
plt.title('2D MDS Representtaion of CTY dissimilarities')
plt.scatter(mdspos_sf[:, 0], mdspos_sf[:, 1], c=np.arange(nbsamples*3),cmap=mycmap, s=s, label='scalarfield cty', marker='x')
plt.xlim(sfMDSxmin,sfMDSxmax)
plt.ylim(sfMDSymin,sfMDSymax)
plt.legend(scatterpoints=1, loc='best', shadow=False)
cbar = plt.colorbar()
cbar.set_label('sample #')
plt.subplot(232)
sns.histplot(df.lithocodes_cty)
plt.subplot(233)
sns.scatterplot(x=df.lithocodes_cty,y=df.scalarfield_cty)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(235)
sns.kdeplot(x=df.lithocodes_cty,y=df.scalarfield_cty)
plt.xlim(lcmin,lcmax)
plt.ylim(sfmin,sfmax)
plt.subplot(236)
sns.histplot(df.scalarfield_cty)
fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.6, wspace=0.3, hspace=0.25)
plt.show()

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING 2D MDS REPRESENTATION END")
