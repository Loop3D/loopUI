# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:14:48 2021

@author: Guillaume Pirot
"""

# import modules
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import manifold
import pickle

picklefilenamedata = "./pickledata/synthetic-case-data.pickle"
picklefilenamecardinality = "./pickledata/synthetic-case-cardinality.pickle"
picklefilenameentropy = "./pickledata/synthetic-case-entropy.pickle"
picklefilenamehistogram = "./pickledata/synthetic-case-histogram.pickle"
picklefilenamesemivariogram = "./pickledata/synthetic-case-semivariogram.pickle"
picklefilenameconnectivity = "./pickledata/synthetic-case-connectivity.pickle"
picklefilenamemph = "./pickledata/synthetic-case-mph.pickle"
picklefilenamewavelet = "./pickledata/synthetic-case-wavelet.pickle"
picklefilenametopology = "./pickledata/synthetic-case-topology.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!
sample_num1 = 0
sample_num2 = 10
myseed=65432

myfs = 24

def plot_comparison_dissimilarities(dist_hist,dist_2ps,dist_mph,dist_wvt,dist_cty,dist_tpl_shd,dist_tpl_lsgd,title_spec):
    nsamples = dist_hist.shape[0]
    # SNS plot
    ix=np.tril_indices(nsamples,k=-1)
    df= pd.DataFrame({'his':dist_hist[ix], '2ps':dist_2ps[ix], 'mph':dist_mph[ix], 'cty':dist_cty[ix], 'wvt':dist_wvt[ix],
                      'shd':dist_tpl_shd[ix], 'lsg':dist_tpl_lsgd[ix]})
    sns.set(font_scale=3)
    g = sns.PairGrid(df)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    plt.show()
    return


def plot_2D_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec):
    cmin = np.amin(cardinality)
    cmax = np.amax(cardinality)
    emin = np.amin(entropy)
    emax = np.amax(entropy)
    nrm01_card = (cardinality.flatten()-cmin)/(cmax-cmin)
    nrm01_ent = (entropy.flatten()-emin)/(emax-emin)
    fig = plt.figure()
    gs = fig.add_gridspec(1,8)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0leg = fig.add_subplot(gs[0, 2])
    ax1 = fig.add_subplot(gs[0, 3:5])
    ax1leg = fig.add_subplot(gs[0, 5])
    ax2 = fig.add_subplot(gs[0, 6:])
    pos0=ax0.imshow(cardinality,cmap='viridis')
    ax0.axis('off'),ax0.set_title(lgd_card),ax0leg.axis('off')
    axins0 = inset_axes(ax0leg,
                       width="10%",  # width = 5% of parent_bbox width
                       height="90%",  # height : 50%
                       loc='center left'
                       )
    fig.colorbar(pos0,cax=axins0,ticks=[]) 
    ax1.imshow(entropy,cmap='viridis')
    ax1.axis('off'),ax1.set_title(lgd_ent),ax1leg.axis('off')
    ax2.scatter(nrm01_card,nrm01_ent,c='blue',marker='+')
    ax2.set_xlabel('norm. cardinality'),ax2.set_ylabel('norm. entropy')
    ax2.set_title(title_spec)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.35, wspace=0.3, hspace=0.5)
    plt.show()
    return

def plot_3D_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec,slice_ix=0,slice_iy=0,slice_iz=0):
    cmin = np.nanmin(cardinality.flatten())
    cmax = np.nanmax(cardinality.flatten())
    emin = np.nanmin(entropy.flatten())
    emax = np.nanmax(entropy.flatten())
    nrm01_card = (cardinality.flatten()-cmin)/(cmax-cmin)
    nrm01_ent = (entropy.flatten()-emin)/(emax-emin)
    fig = plt.figure()
    gs = fig.add_gridspec(2,12)
    ax00leg = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1:3])
    ax02 = fig.add_subplot(gs[0, 3:5])
    ax03 = fig.add_subplot(gs[0, 5:7])
    ax10leg = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1:3])
    ax12 = fig.add_subplot(gs[1, 3:5])
    ax13 = fig.add_subplot(gs[1, 5:7])
    ax4 = fig.add_subplot(gs[:, 8:])
    ax01.axis('off'),ax02.axis('off'),ax03.axis('off')
    ax11.axis('off'),ax12.axis('off'),ax13.axis('off')
    ax01.set_title('Map'), ax02.set_title('W (N) E'), ax03.set_title('N (W) S')
    ax11.set_title('Map'), ax12.set_title('W (N) E'), ax13.set_title('N (W) S')
    pos01=ax01.imshow(cardinality[slice_iz,:,:],vmin=cmin,vmax=cmax,cmap='viridis')
    ax02.imshow(cardinality[:,slice_iy,:],vmin=cmin,vmax=cmax,cmap='viridis')
    ax03.imshow(cardinality[:,:,slice_ix],vmin=cmin,vmax=cmax,cmap='viridis')
    ax00leg.axis('off') #,ax00leg.set_title(lgd_card)
    axins00 = inset_axes(ax00leg,
                       width="10%",  # width = 5% of parent_bbox width
                       height="90%",  # height : 50%
                       loc='center left'
                       )
    fig.colorbar(pos01,cax=axins00,label=lgd_card) 
    pos11=ax11.imshow(entropy[slice_iz,:,:],vmin=emin,vmax=emax,cmap='viridis')
    ax12.imshow(entropy[:,slice_iy,:],vmin=emin,vmax=emax,cmap='viridis')
    ax13.imshow(entropy[:,:,slice_ix],vmin=emin,vmax=emax,cmap='viridis')
    ax10leg.axis('off') #,ax10leg.set_title(lgd_ent)
    axins10 = inset_axes(ax10leg,
                       width="10%",  # width = 5% of parent_bbox width
                       height="90%",  # height : 50%
                       loc='center left'
                       )
    fig.colorbar(pos11,cax=axins10,label=lgd_ent) 
    ax4.scatter(nrm01_card,nrm01_ent,c='blue',marker='+')
    ax4.set_xlabel('norm. cardinality'),ax4.set_ylabel('norm. entropy')
    ax4.set_title(title_spec)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.15, top=0.55, wspace=0.1, hspace=0.25)
    plt.show()
    return

def plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec,slice_ix=0,slice_iy=0,slice_iz=0):
    dim = len(entropy.shape)
    if dim ==2:
        plot_2D_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec)
    elif dim==3:
        plot_3D_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec,slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)
    else:
        print("NA for dimensions>3 or dimensions<2")
    return

#%% load computed measures
with open(picklefilenamecardinality, 'rb') as f:
    [crd_lc_100,crd_lc_50A,crd_lc_50B,
     sf_100_crdeq_rng,sf_100_crdeq_std,sf_100_crdeq_rngstd,
     sf_50A_crdeq_rng,sf_50A_crdeq_std,sf_50A_crdeq_rngstd,
     sf_50B_crdeq_rng,sf_50B_crdeq_std,sf_50B_crdeq_rngstd] = pickle.load(f)
with open(picklefilenameentropy, 'rb') as f:
    [ent_lc_100,ent_lc_50A,ent_lc_50B,ent_sf_100,ent_sf_50A,ent_sf_50B] = pickle.load(f)
with open(picklefilenamehistogram, 'rb') as f:
    [dist_hist_lc,dist_hist_sf] = pickle.load(f)
with open(picklefilenamesemivariogram, 'rb') as f:
    [dist_2ps_lc,dist_2ps_sf] = pickle.load(f)
with open(picklefilenameconnectivity, 'rb') as f:
    [dist_cty_lc,dist_cty_sf] = pickle.load(f)
with open(picklefilenamemph, 'rb') as f:
    [dist_mph_lc,dist_mph_sf] = pickle.load(f)
with open(picklefilenamewavelet, 'rb') as f:
    [dist_wvt_lc,dist_wvt_sf] = pickle.load(f)
with open(picklefilenametopology, 'rb') as f:
    [dist_tpl_shd_lc,dist_tpl_shd_sf,dist_tpl_lsgd_lc,dist_tpl_lsgd_sf] = pickle.load(f)


#%% Lithology code
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"LITHOCODE DISSIMILARITY COMPARISON START")
# divide by maximum distance for normalization between [0-1]
dist_hist = dist_hist_lc/np.amax(dist_hist_lc)
dist_mph = dist_mph_lc/np.amax(dist_mph_lc)
dist_2ps = dist_2ps_lc/np.amax(dist_2ps_lc)
dist_cty = dist_cty_lc/np.amax(dist_cty_lc)
dist_wvt = dist_wvt_lc/np.amax(dist_wvt_lc)
dist_tpl_shd = dist_tpl_shd_lc/np.amax(dist_tpl_shd_lc)
dist_tpl_lsgd = dist_tpl_lsgd_lc/np.amax(dist_tpl_lsgd_lc)
title_spec = 'lithocode'
plot_comparison_dissimilarities(dist_hist,dist_2ps,dist_mph,dist_cty,dist_wvt,dist_tpl_shd,dist_tpl_lsgd,title_spec)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"LITHOCODE DISSIMILARITY COMPARISON END")

# Cardinality/Entropy comparison 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"LITHOCODE UNCERTAINTY VOXETS COMPARISON START")
sns.set(font_scale=1)
entropy = ent_lc_100
lgd_ent = 'Shanon Entropy'
cardinality = crd_lc_100
lgd_card = 'Cardinality'
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S1',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)

entropy = ent_lc_50A
cardinality = crd_lc_50A
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S2',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)

entropy = ent_lc_50B
cardinality = crd_lc_50B
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S3',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"LITHOCODE UNCERTAINTY VOXETS COMPARISON END")

#%% scalar field
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"SCALARFIELD DISSIMILARITY COMPARISON START")
# divide by maximum distance for normalization between [0-1]
dist_hist = dist_hist_sf/np.amax(dist_hist_sf)
dist_mph = dist_mph_sf/np.amax(dist_mph_sf)
dist_2ps = dist_2ps_sf/np.amax(dist_2ps_sf)
dist_cty = dist_cty_sf/np.amax(dist_cty_sf)
dist_wvt = dist_wvt_sf/np.amax(dist_wvt_sf)
dist_tpl_shd = dist_tpl_shd_sf/np.amax(dist_tpl_shd_sf)
dist_tpl_lsgd = dist_tpl_lsgd_sf/np.amax(dist_tpl_lsgd_sf)
title_spec = 'scalarfierld'
plot_comparison_dissimilarities(dist_hist,dist_2ps,dist_mph,dist_cty,dist_wvt,dist_tpl_shd,dist_tpl_lsgd,title_spec)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"SCALARFIELD DISSIMILARITY COMPARISON END")

# Cardinality/Entropy comparison 
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"SCALARFIELD UNCERTAINTY VOXETS COMPARISON START")
sns.set(font_scale=1)
entropy = ent_sf_100
lgd_ent = 'Cont. Entropy'
cardinality = sf_100_crdeq_rngstd
lgd_card = 'Range & Standard Dev.'
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S1',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)

entropy = ent_sf_50A
cardinality = sf_50A_crdeq_rngstd
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S2',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)

entropy = ent_sf_50B
cardinality = sf_50B_crdeq_rngstd
plot_uncertainty_voxets(cardinality,entropy,lgd_card,lgd_ent,title_spec+' S3',slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"SCALARFIELD UNCERTAINTY VOXETS COMPARISON END")

#%% Cardinality
fig, ax = plt.subplots(4,4) #,figsize=(13,13)
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[0,3].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,2].axis('off')
ax[1,3].axis('off')
ax[2,0].axis('off')
ax[2,1].axis('off')
ax[2,2].axis('off')
ax[2,3].axis('off')
ax[3,0].axis('off')
ax[3,1].axis('off')
ax[3,2].axis('off')
ax[3,3].axis('off')
axins03 = inset_axes(ax[0,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
axins13 = inset_axes(ax[1,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
axins23 = inset_axes(ax[2,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
axins33 = inset_axes(ax[3,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
ax[0,0].set_title('Cardinality Map S1')
ax[0,1].set_title('Cardinality Map S2') # 'Cardinality W (N) E'
ax[0,2].set_title('Cardinality Map S3') # 'Cardinality N (W) S'
ax[0,3].set_title('Lithocode card.')
ax[1,0].set_title('Range $R$ Map S1')
ax[1,1].set_title('Range $R$ Map S2') # 'Range $R$ W (N) E'
ax[1,2].set_title('Range $R$ Map S3') # 'Range $R$ N (W) S'
ax[1,3].set_title('Scalar-field $R$')
ax[2,0].set_title('Std. dev. $\\sigma$ Map S1')
ax[2,1].set_title('Std. dev. $\\sigma$ Map S2') # 'Std. dev. $\\sigma$ W (N) E'
ax[2,2].set_title('Std. dev. $\\sigma$ Map S3') # 'Std. dev. $\\sigma$ N (W) S'
ax[2,3].set_title('Scalar-field $\\sigma$') # ('Density ($kg/m^3$)')
ax[3,0].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S1')
ax[3,1].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S2') # ('$(R_0^1 + \\sigma_0^1)/2$ W (N) E')
ax[3,2].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S3') # ('$(R_0^1 + \\sigma_0^1)/2$ N (W) S')
ax[3,3].set_title('Scalar-field $(R_0^1 + \\sigma_0^1)/2$') # ('Density ($kg/m^3$)')
vmin = np.min([np.min(crd_lc_100),np.min(crd_lc_50A),np.min(crd_lc_50B)])
vmax = np.max([np.max(crd_lc_100),np.max(crd_lc_50A),np.max(crd_lc_50B)])
pos00 = ax[0,0].imshow(crd_lc_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[0,1].imshow(crd_lc_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(crd[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[0,2].imshow(crd_lc_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(crd[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos00,cax=axins03)  #,label=clblab
vmin = np.nanmin([np.nanmin(sf_100_crdeq_rng),np.nanmin(sf_50A_crdeq_rng),np.nanmin(sf_50B_crdeq_rng)])
vmax = np.nanmax([np.nanmax(sf_100_crdeq_rng),np.nanmax(sf_50A_crdeq_rng),np.nanmax(sf_50B_crdeq_rng)])
pos10 = ax[1,0].imshow(sf_100_crdeq_rng[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[1,1].imshow(sf_50A_crdeq_rng[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[1,2].imshow(sf_50B_crdeq_rng[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos10,cax=axins13)  #,label=clblab
vmin = np.nanmin([np.nanmin(sf_100_crdeq_std),np.nanmin(sf_50A_crdeq_std),np.nanmin(sf_50B_crdeq_std)])
vmax = np.nanmax([np.nanmax(sf_100_crdeq_std),np.nanmax(sf_50A_crdeq_std),np.nanmax(sf_50B_crdeq_std)])
pos20 = ax[2,0].imshow(sf_100_crdeq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[2,1].imshow(sf_50A_crdeq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_std[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[2,2].imshow(sf_50B_crdeq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_std[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos20,cax=axins23)  #,label=clblab
vmin = np.nanmin([np.nanmin(sf_100_crdeq_rngstd),np.nanmin(sf_50A_crdeq_rngstd),np.nanmin(sf_50B_crdeq_rngstd)])
vmax = np.nanmax([np.nanmax(sf_100_crdeq_rngstd),np.nanmax(sf_50A_crdeq_rngstd),np.nanmax(sf_50B_crdeq_rngstd)])
pos30 = ax[3,0].imshow(sf_100_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[3,1].imshow(sf_50A_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[3,2].imshow(sf_50B_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos30,cax=axins33)  #,label=clblab
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.6, top=2.5, wspace=0.1, hspace=0.2)
plt.show()

#%% Entropy
fig, ax = plt.subplots(2,4) #,figsize=(13,13)
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[0,3].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,2].axis('off')
ax[1,3].axis('off')
axins03 = inset_axes(ax[0,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
axins13 = inset_axes(ax[1,3],
                   width="10%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left'
                   )
ax[0,0].set_title('Shannon\'s entropy Map S1')
ax[0,1].set_title('Shannon\'s entropy Map S2') # 'entropy W (N) E'
ax[0,2].set_title('Shannon\'s entropy Map S3') # 'entropy N (W) S'
ax[0,3].set_title('Lithocode Sh. ent.')
ax[1,0].set_title('Continuous entropy Map S1')
ax[1,1].set_title('Continuous entropy Map S2') # 'Range $R$ W (N) E'
ax[1,2].set_title('Continuous entropy Map S3') # 'Range $R$ N (W) S'
ax[1,3].set_title('Scalar-field cont. ent.')
vmin = np.min([np.min(ent_lc_100),np.min(ent_lc_50A),np.min(ent_lc_50B)])
vmax = np.max([np.max(ent_lc_100),np.max(ent_lc_50A),np.max(ent_lc_50B)])
pos00 = ax[0,0].imshow(ent_lc_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[0,1].imshow(ent_lc_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(ent[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[0,2].imshow(ent_lc_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(ent[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos00,cax=axins03)  #,label=clblab
vmin = np.nanmin([np.nanmin(ent_sf_100),np.nanmin(ent_sf_50A),np.nanmin(ent_sf_50B)])
vmax = np.nanmax([np.nanmax(ent_sf_100),np.nanmax(ent_sf_50A),np.nanmax(ent_sf_50B)])
pos10 = ax[1,0].imshow(ent_sf_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
ax[1,1].imshow(ent_sf_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
ax[1,2].imshow(ent_sf_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
fig.colorbar(pos10,cax=axins13)  #,label=clblab
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.6, top=1.25, wspace=0.1, hspace=0.2)
plt.show()

#%% histogram

#%% semi-variogram

#%% connectivity

#%% multiple-point histogram

#%% wavelet decomposition

#%% topology


# #%% Cardinality and Entropy CET members day POSTER
# fig, ax = plt.subplots(4,2) #,figsize=(13,13)
# ax[0,0].axis('off')
# ax[0,1].axis('off')
# ax[1,0].axis('off')
# ax[1,1].axis('off')
# ax[2,0].axis('off')
# ax[2,1].axis('off')
# ax[3,0].axis('off')
# ax[3,1].axis('off')
# axins30 = inset_axes(ax[3,0],
#                    width="90%",  # width = 5% of parent_bbox width
#                    height="10%",  # height : 50%
#                    loc='upper center'
#                    )
# axins31 = inset_axes(ax[3,1],
#                    width="90%",  # width = 5% of parent_bbox width
#                    height="10%",  # height : 50%
#                    loc='upper center'
#                    )
# ax[0,0].set_title('Cardinality Map S1')
# ax[1,0].set_title('Cardinality Map S2') # 'Cardinality W (N) E'
# ax[2,0].set_title('Cardinality Map S3') # 'Cardinality N (W) S'
# ax[3,0].set_title('Lithocode card.')
# ax[0,1].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S1')
# ax[1,1].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S2') # ('$(R_0^1 + \\sigma_0^1)/2$ W (N) E')
# ax[2,1].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S3') # ('$(R_0^1 + \\sigma_0^1)/2$ N (W) S')
# ax[3,1].set_title('Scalar-field $(R_0^1 + \\sigma_0^1)/2$') # ('Density ($kg/m^3$)')
# vmin = np.min([np.min(crd_lc_100),np.min(crd_lc_50A),np.min(crd_lc_50B)])
# vmax = np.max([np.max(crd_lc_100),np.max(crd_lc_50A),np.max(crd_lc_50B)])
# pos00 = ax[0,0].imshow(crd_lc_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[1,0].imshow(crd_lc_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(crd[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,0].imshow(crd_lc_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(crd[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos00,cax=axins30,orientation="horizontal")  #,label=clblab
# vmin = np.nanmin([np.nanmin(sf_100_crdeq_rngstd),np.nanmin(sf_50A_crdeq_rngstd),np.nanmin(sf_50B_crdeq_rngstd)])
# vmax = np.nanmax([np.nanmax(sf_100_crdeq_rngstd),np.nanmax(sf_50A_crdeq_rngstd),np.nanmax(sf_50B_crdeq_rngstd)])
# pos30 = ax[0,1].imshow(sf_100_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[1,1].imshow(sf_50A_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,1].imshow(sf_50B_crdeq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos30,cax=axins31,orientation="horizontal")  #,label=clblab
# fig.subplots_adjust(left=0.0, bottom=0.0, right=0.8, top=2.5, wspace=0.1, hspace=0.2)
# plt.show()

# fig, ax = plt.subplots(4,2) #,figsize=(13,13)
# ax[0,0].axis('off')
# ax[0,1].axis('off')
# ax[1,0].axis('off')
# ax[1,1].axis('off')
# ax[2,0].axis('off')
# ax[2,1].axis('off')
# ax[3,0].axis('off')
# ax[3,1].axis('off')
# axins30 = inset_axes(ax[3,0],
#                    width="90%",  # width = 5% of parent_bbox width
#                    height="10%",  # height : 50%
#                    loc='upper center'
#                    )
# axins31 = inset_axes(ax[3,1],
#                    width="90%",  # width = 5% of parent_bbox width
#                    height="10%",  # height : 50%
#                    loc='upper center'
#                    )
# ax[0,0].set_title('Shannon\'s entropy Map S1')
# ax[1,0].set_title('Shannon\'s entropy Map S2') # 'entropy W (N) E'
# ax[2,0].set_title('Shannon\'s entropy Map S3') # 'entropy N (W) S'
# ax[3,0].set_title('Lithocode Sh. ent.')
# ax[0,1].set_title('Continuous entropy Map S1')
# ax[1,1].set_title('Continuous entropy Map S2') # 'Range $R$ W (N) E'
# ax[2,1].set_title('Continuous entropy Map S3') # 'Range $R$ N (W) S'
# ax[3,1].set_title('Scalar-field cont. ent.')
# vmin = np.min([np.min(ent_lc_100),np.min(ent_lc_50A),np.min(ent_lc_50B)])
# vmax = np.max([np.max(ent_lc_100),np.max(ent_lc_50A),np.max(ent_lc_50B)])
# pos00 = ax[0,0].imshow(ent_lc_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[1,0].imshow(ent_lc_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(ent[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,0].imshow(ent_lc_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(ent[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos00,cax=axins30,orientation="horizontal")  #,label=clblab
# vmin = np.nanmin([np.nanmin(ent_sf_100),np.nanmin(ent_sf_50A),np.nanmin(ent_sf_50B)])
# vmax = np.nanmax([np.nanmax(ent_sf_100),np.nanmax(ent_sf_50A),np.nanmax(ent_sf_50B)])
# pos10 = ax[0,1].imshow(ent_sf_100[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[1,1].imshow(ent_sf_50A[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,1].imshow(ent_sf_50B[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rng[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos10,cax=axins31,orientation="horizontal")  #,label=clblab
# fig.subplots_adjust(left=0.0, bottom=0.0, right=0.8, top=2.5, wspace=0.1, hspace=0.2)
# plt.show()
