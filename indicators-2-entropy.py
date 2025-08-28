# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:36:51 2021

@author: Guillaume Pirot
"""
# import modules
import numpy as np
from datetime import datetime
import pickle
from loopUI import entropy,continuous_entropy, plot_voxet

picklefilenameentropy = "./pickledata/synthetic-case-entropy.pickle"
picklefilenamedata = "./pickledata/synthetic-case-data.pickle"

# plotting parameters
slice_ix=0
slice_iy=0
slice_iz=7
aspectratio = 1 # !! in pixels !!

# continuous entropy parameters
nbins=50 # continuous entropy discretization

# import data
with open(picklefilenamedata, 'rb') as f:
    [lithocode_100,lithocode_50A,lithocode_50B,scalarfield_100,scalarfield_50A,
     scalarfield_50B,nx,ny,nz,nbsamples,clblab,clblabsf] = pickle.load(f)


#%% entropy
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_100 entropy START ")
ent_lc_100 = entropy(lithocode_100)   
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_100 entropy END")
plot_voxet(ent_lc_100,-1,clblab+' entropy S1',slice_ix,slice_iy,slice_iz,aspectratio,'viridis')

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_50A entropy START ")
ent_lc_50A = entropy(lithocode_50A)   
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_50A entropy END")
plot_voxet(ent_lc_50A,-1,clblab+' entropy S2',slice_ix,slice_iy,slice_iz,aspectratio,'viridis')

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_50B entropy START ")
ent_lc_50B = entropy(lithocode_50B)   
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING lithocode_50B entropy END")
plot_voxet(ent_lc_50B,-1,clblab+' entropy S3',slice_ix,slice_iy,slice_iz,aspectratio,'viridis')

#%%
# continuous entropy
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_100 continuous entropy START")
ent_sf_100 = continuous_entropy(scalarfield_100,nbins)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_100 continuous entropy END")
plot_voxet(ent_sf_100,-1,clblabsf+' cont. ent. S1',slice_ix,slice_iy,slice_iz,aspectratio)

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_50A continuous entropy START")
ent_sf_50A = continuous_entropy(scalarfield_50A,nbins)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_50A continuous entropy END")
plot_voxet(ent_lc_50A,-1,clblabsf+' cont. ent. S2',slice_ix,slice_iy,slice_iz,aspectratio)

print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_50B continuous entropy START")
ent_sf_50B = continuous_entropy(scalarfield_50B,nbins)
print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - "+"COMPUTING scalarfield_50B continuous entropy END")
plot_voxet(ent_lc_50B,-1,clblabsf+' cont. ent. S3',slice_ix,slice_iy,slice_iz,aspectratio)


#%% SAVE COMPUTED DATA
with open(picklefilenameentropy, 'wb') as f:
    pickle.dump([ent_lc_100,ent_lc_50A,ent_lc_50B,ent_sf_100,ent_sf_50A,ent_sf_50B], f)

#%% plot for paper
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(2,4) #,figsize=(13,13)
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[0,3].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,2].axis('off')
ax[1,3].axis('off')
# ax[2,0].axis('off')
# ax[2,1].axis('off')
# ax[2,2].axis('off')
# ax[2,3].axis('off')
# ax[3,0].axis('off')
# ax[3,1].axis('off')
# ax[3,2].axis('off')
# ax[3,3].axis('off')
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
# axins23 = inset_axes(ax[2,3],
#                    width="10%",  # width = 5% of parent_bbox width
#                    height="90%",  # height : 50%
#                    loc='center left'
#                    )
# axins33 = inset_axes(ax[3,3],
#                    width="10%",  # width = 5% of parent_bbox width
#                    height="90%",  # height : 50%
#                    loc='center left'
#                    )
ax[0,0].set_title('Shannon\'s entropy Map S1')
ax[0,1].set_title('Shannon\'s entropy Map S2') # 'entropy W (N) E'
ax[0,2].set_title('Shannon\'s entropy Map S3') # 'entropy N (W) S'
ax[0,3].set_title('Lithocode Sh. ent.')
ax[1,0].set_title('Continuous entropy Map S1')
ax[1,1].set_title('Continuous entropy Map S2') # 'Range $R$ W (N) E'
ax[1,2].set_title('Continuous entropy Map S3') # 'Range $R$ N (W) S'
ax[1,3].set_title('Scalar-field cont. ent.')
# ax[2,0].set_title('Std. dev. $\\sigma$ Map S1')
# ax[2,1].set_title('Std. dev. $\\sigma$ Map S2') # 'Std. dev. $\\sigma$ W (N) E'
# ax[2,2].set_title('Std. dev. $\\sigma$ Map S3') # 'Std. dev. $\\sigma$ N (W) S'
# ax[2,3].set_title('Scalar-field $\\sigma$') # ('Density ($kg/m^3$)')
# ax[3,0].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S1')
# ax[3,1].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S2') # ('$(R_0^1 + \\sigma_0^1)/2$ W (N) E')
# ax[3,2].set_title('$(R_0^1 + \\sigma_0^1)/2$ Map S3') # ('$(R_0^1 + \\sigma_0^1)/2$ N (W) S')
# ax[3,3].set_title('Scalar-field $(R_0^1 + \\sigma_0^1)/2$') # ('Density ($kg/m^3$)')
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
# vmin = np.nanmin([np.nanmin(sf_100_enteq_std),np.nanmin(sf_50A_enteq_std),np.nanmin(sf_50B_enteq_std)])
# vmax = np.nanmax([np.nanmax(sf_100_enteq_std),np.nanmax(sf_50A_enteq_std),np.nanmax(sf_50B_enteq_std)])
# pos20 = ax[2,0].imshow(sf_100_enteq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,1].imshow(sf_50A_enteq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_std[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[2,2].imshow(sf_50B_enteq_std[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_std[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos20,cax=axins23)  #,label=clblab
# vmin = np.nanmin([np.nanmin(sf_100_enteq_rngstd),np.nanmin(sf_50A_enteq_rngstd),np.nanmin(sf_50B_enteq_rngstd)])
# vmax = np.nanmax([np.nanmax(sf_100_enteq_rngstd),np.nanmax(sf_50A_enteq_rngstd),np.nanmax(sf_50B_enteq_rngstd)])
# pos30 = ax[3,0].imshow(sf_100_enteq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
# ax[3,1].imshow(sf_50A_enteq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,0,:],cmap='viridis',vmin=vmin,vmax=vmax)
# ax[3,2].imshow(sf_50B_enteq_rngstd[slice_iz,:,:],origin='lower',cmap='viridis',vmin=vmin,vmax=vmax) # .imshow(rho_rngstd[:,:,0],cmap='viridis',vmin=vmin,vmax=vmax)
# fig.colorbar(pos30,cax=axins33)  #,label=clblab
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.6, top=1.25, wspace=0.1, hspace=0.2)
plt.show()
