# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:13:56 2021

@author: Guillaume Pirot
"""

# import modules
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import label
from numpy.random import default_rng
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import KMeans
import pywt

base = np.e

# from noddyverse: define colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    Thanks to https://gist.github.com/delestro/54d5a34676a8cef7477e
    
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    np.random.seed(seed=0)
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

# from noddyverse: download file, ungzip and stuff into numpy array
from urllib.request import urlopen
import gzip
def get_gz_array(url,skiprows):
    my_gzip_stream = urlopen(url)
    my_stream = gzip.open(my_gzip_stream, 'r')
    return(np.loadtxt(my_stream,skiprows=skiprows))


def cardinality(array):
    nbsamples = array.shape[-1]
    voxetdim = array.shape[0:-1]
    tmp = np.reshape(array,(np.prod(voxetdim),nbsamples))
    crd = np.zeros(len(tmp))
    classes = np.unique(tmp)
    for c in range(len(classes)):
        ixc =  1.0*(tmp==classes[c])    
        p = np.sum(ixc,axis=1)/nbsamples
        crd += 1.0*(p>0)
    crd=crd.reshape(voxetdim)    
    return crd

def entropy(array):
    nbsamples = array.shape[-1]
    voxetdim = array.shape[0:-1]
    tmp = np.reshape(array,(np.prod(voxetdim),nbsamples))
    ent = np.zeros(len(tmp))
    classes = np.unique(tmp)
    for c in range(len(classes)):
        ixc =  1.0*(tmp==classes[c])    
        p = np.sum(ixc,axis=1)/nbsamples
        ixp = np.where(p>0)
        logpbase = np.zeros(len(tmp))
        logpbase[ixp] = np.log(p[ixp])/np.log(base)
        ent -= p * logpbase
    ent=ent.reshape(voxetdim)    
    return ent

def entropyNcardinality(array):
    nbsamples = array.shape[-1]
    voxetdim = array.shape[0:-1]
    tmp = np.reshape(array,(np.prod(voxetdim),nbsamples))
    ent = np.zeros(len(tmp))
    crd = np.zeros(len(tmp))
    classes = np.unique(tmp)
    for c in range(len(classes)):
        ixc =  1.0*(tmp==classes[c])    
        p = np.sum(ixc,axis=1)/nbsamples
        ixp = np.where(p>0)
        logpbase = np.zeros(len(tmp))
        logpbase[ixp] = np.log(p[ixp])/np.log(base)
        ent -= p * logpbase
        crd += 1.0*(p>0)
    ent=ent.reshape(voxetdim)    
    crd=crd.reshape(voxetdim)    
    return ent,crd

def vec_hist(a, bins):
    i = np.repeat(np.arange(np.product(a.shape[:-1])), a.shape[-1])
    return np.histogram2d(i, a.flatten(), (a.shape[0], bins))[0] #.reshape(a.shape[:-1], -1)

def continuous_entropy(array,nbins):
    nbsamples = array.shape[-1]
    voxetdim = array.shape[0:-1]
    array_min =np.amin(array)
    array_max =np.amax(array)
    ent=np.zeros(np.prod(voxetdim))
    tmp = np.copy(np.reshape(array,(np.prod(voxetdim),nbsamples)))
    tmp.sort(axis=1)
    binlim = np.linspace(array_min,array_max,nbins+1)
    histcount = vec_hist(tmp, binlim)
    logpbase = np.zeros(histcount.shape)
    p = histcount/nbsamples
    ixp = np.where(p>0)
    logpbase[ixp] = np.log(p[ixp])/np.log(base)
    ent = -1*np.sum(p*logpbase,axis=1)
    ent = np.reshape(ent,voxetdim)
    return ent

def stochastic_upscale(mx,seed):
    rng = default_rng(seed)
    ndim = len(mx.shape)
    ux_shape = tuple(np.floor(np.asarray(mx.shape)/2).astype(int))
    reductionfactor = 2**ndim
    tmp_shape = list(np.floor( np.asarray(list(mx.shape))/2 ).astype(int))
    tmp_shape.append(reductionfactor)
    tmp_shape=tuple(tmp_shape)
    tmp = np.ones(tmp_shape)*np.nan
    v = np.array([0,1])
    if ndim == 2:
        ny,nx = mx.shape
        [dx,dy]=np.meshgrid(v,v)
        dx=dx.flatten().astype(int)
        dy=dy.flatten().astype(int)
    elif ndim ==3:
        nz,ny,nx = mx.shape
        [dx,dy,dz]=np.meshgrid(v,v,v)
        dx=dx.flatten().astype(int)
        dy=dy.flatten().astype(int)
        dz=dz.flatten().astype(int)
    else:
        return -1
    for i in range(reductionfactor):
        if ndim == 2:
            tmp2 = mx[dy[i]:ny+dy[i]:2,dx[i]:nx+dx[i]:2]
            tmp[:,:,i] = tmp2[0:ux_shape[0],0:ux_shape[1]]
            del tmp2
        elif ndim==3:
            tmp2 = mx[dz[i]:nz+dz[i]:2,dy[i]:ny+dy[i]:2,dx[i]:nx+dx[i]:2]
            tmp[:,:,:,i] = tmp2[0:ux_shape[0],0:ux_shape[1],0:ux_shape[2]]
            del tmp2
    ix2 = np.reshape(np.floor(rng.uniform(0,reductionfactor-1e-12,np.prod(ux_shape))).astype(int),ux_shape).flatten()
    ix1 = np.arange(np.prod(ux_shape)).flatten()
    tmp = np.reshape(tmp,(np.prod(ux_shape),reductionfactor))
    upscaled_mx = np.reshape(tmp[ix1,ix2],ux_shape)
    return upscaled_mx

def dist_kmeans_mph(img1,img2,n_levels,patternsize,n_clusters,nmax_patterns,seed,plot=False,verb=False):
    # initialize distance value for incrementation
    d=0.0
    # get ndim
    ndim = len(img1.shape)
    for l in range(n_levels+1):
        rng = np.random.default_rng(2*seed+l)
        if verb:
            print('Level '+str(l))
        # get pattern matrix shape
        tmp_shape = list(np.asarray(list(img1.shape)) - patternsize +1)
        tmp_shape.append(np.prod(patternsize))
        tmp_shape=tuple(tmp_shape)
        # get nb patterns
        npat = np.prod(tmp_shape[:-1])
        # get patterns and sample from ing1 and img2
        img1_all_patterns = np.ones(tmp_shape)*np.nan
        img2_all_patterns = np.ones(tmp_shape)*np.nan
        if verb:
            print('Number of possible patterns: '+str(npat))
        if ndim==2:
            [ppdim1,ppdim0] = np.meshgrid(np.arange(patternsize[1]),np.arange(patternsize[0]))
            ppdim0 = ppdim0.flatten()
            ppdim1 = ppdim1.flatten()
            for pp in range(np.prod(patternsize)):
                img1_all_patterns[:,:,pp] = img1[ppdim0[pp]:tmp_shape[0]+ppdim0[pp],
                                                 ppdim1[pp]:tmp_shape[1]+ppdim1[pp]]
                img2_all_patterns[:,:,pp] = img2[ppdim0[pp]:tmp_shape[0]+ppdim0[pp],
                                                 ppdim1[pp]:tmp_shape[1]+ppdim1[pp]]
        elif ndim==3:
            [ppdim2,ppdim1,ppdim0] = np.meshgrid(np.arange(patternsize[2]),np.arange(patternsize[1]),np.arange(patternsize[0]))
            ppdim0 = ppdim0.flatten()
            ppdim1 = ppdim1.flatten()
            ppdim2 = ppdim2.flatten()
            for pp in range(np.prod(patternsize)):
                img1_all_patterns[:,:,:,pp] = img1[ppdim0[pp]:tmp_shape[0]+ppdim0[pp],
                                                   ppdim1[pp]:tmp_shape[1]+ppdim1[pp],
                                                   ppdim2[pp]:tmp_shape[2]+ppdim2[pp]]
                img2_all_patterns[:,:,:,pp] = img2[ppdim0[pp]:tmp_shape[0]+ppdim0[pp],
                                                   ppdim1[pp]:tmp_shape[1]+ppdim1[pp],
                                                   ppdim2[pp]:tmp_shape[2]+ppdim2[pp]]           
        img1_all_patterns = np.reshape(img1_all_patterns,(npat,np.prod(patternsize)))
        img2_all_patterns = np.reshape(img2_all_patterns,(npat,np.prod(patternsize)))
        # subsamling the patterns
        if npat>nmax_patterns:
            ix_sub1 = (np.floor(rng.uniform(0,1,nmax_patterns)*(npat-1))).astype(int)
            ix_sub2 = (np.floor(rng.uniform(0,1,nmax_patterns)*(npat-1))).astype(int)
        else:
            ix_sub1 = np.arange(npat)
            ix_sub2 = np.arange(npat)
        img1_patterns = img1_all_patterns[ix_sub1,:]
        img2_patterns = img2_all_patterns[ix_sub2,:]
        if verb:
            print('Number of sub-sampled patterns: '+str(len(ix_sub1)))
        del img1_all_patterns,img2_all_patterns,ix_sub1,ix_sub2
            
        # kmeans clustering of patterns
        kmeans_img1 = KMeans(n_clusters=n_clusters, random_state=0).fit(img1_patterns)
        img1_cluster_id,img1_cluster_size = np.unique(kmeans_img1.labels_,return_counts=True)
        kmeans_img2 = KMeans(n_clusters=n_clusters, random_state=0).fit(img2_patterns)
        img2_cluster_id,img2_cluster_size = np.unique(kmeans_img2.labels_,return_counts=True)
        # find best cluster pairing for mph dist computation
        img_cluster_id_pairs_dist = np.ones((n_clusters,3))*np.nan # cluster_id1, cluster_id2, distance
        cpy_img2_cluster_id = img2_cluster_id + 0
        for c in range(n_clusters):
            tmp_cluster = kmeans_img1.cluster_centers_[img1_cluster_id[c],:]
            tmp_dist = (np.sum((kmeans_img2.cluster_centers_[cpy_img2_cluster_id,:] - tmp_cluster)**2,axis=1))**0.5
            img_cluster_id_pairs_dist[c,0] = img1_cluster_id[c]
            img_cluster_id_pairs_dist[c,1] = cpy_img2_cluster_id[np.argmin(tmp_dist)]
            img_cluster_id_pairs_dist[c,2] = tmp_dist[np.argmin(tmp_dist)]
            cpy_img2_cluster_id = np.delete(cpy_img2_cluster_id,np.argmin(tmp_dist))
        # compute distance contribution as density weighted distance between closest best paired clusters
        weights_sum = np.sum(img1_cluster_size+img2_cluster_size)
        weights = img1_cluster_size[(img_cluster_id_pairs_dist[:,0]).astype(int)] + img2_cluster_size[(img_cluster_id_pairs_dist[:,1]).astype(int)]
        dist_mphc = np.sum(img_cluster_id_pairs_dist[:,2]*weights) / weights_sum 
        d += dist_mphc/n_levels
        if verb:
            print('Distance component: '+str(dist_mphc/n_levels))
        if plot:
            plot_kmeans_mph(img1,img2,l,kmeans_img1,kmeans_img2,img_cluster_id_pairs_dist,patternsize,uniqueColorScale=False)
            plot_kmeans_mph(img1,img2,l,kmeans_img1,kmeans_img2,img_cluster_id_pairs_dist,patternsize,uniqueColorScale=True)
        if l<=n_levels:
            img1 = stochastic_upscale(img1,seed+l)
            img2 = stochastic_upscale(img2,seed+l)
            del img_cluster_id_pairs_dist,tmp_cluster,tmp_dist,cpy_img2_cluster_id,kmeans_img1,kmeans_img2
            del img1_cluster_id,img1_cluster_size,img2_cluster_id,img2_cluster_size
            del tmp_shape,npat,img1_patterns,img2_patterns
    return d

def plot_kmeans_mph(img1,img2,l,kmeans_img1,kmeans_img2,img_cluster_id_pairs_dist,patternsize,uniqueColorScale=False):
    c10 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[0,0]).astype(int),:],tuple(patternsize))
    c11 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[1,0]).astype(int),:],tuple(patternsize))
    c12 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[2,0]).astype(int),:],tuple(patternsize))
    c13 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[3,0]).astype(int),:],tuple(patternsize))
    c14 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[4,0]).astype(int),:],tuple(patternsize))
    c15 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[5,0]).astype(int),:],tuple(patternsize))
    c16 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[6,0]).astype(int),:],tuple(patternsize))
    c17 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[7,0]).astype(int),:],tuple(patternsize))
    c18 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[8,0]).astype(int),:],tuple(patternsize))
    c19 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[9,0]).astype(int),:],tuple(patternsize))    
    c20 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[0,1]).astype(int),:],tuple(patternsize))
    c21 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[1,1]).astype(int),:],tuple(patternsize))
    c22 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[2,1]).astype(int),:],tuple(patternsize))
    c23 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[3,1]).astype(int),:],tuple(patternsize))
    c24 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[4,1]).astype(int),:],tuple(patternsize))
    c25 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[5,1]).astype(int),:],tuple(patternsize))
    c26 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[6,1]).astype(int),:],tuple(patternsize))
    c27 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[7,1]).astype(int),:],tuple(patternsize))
    c28 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[8,1]).astype(int),:],tuple(patternsize))
    c29 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[9,1]).astype(int),:],tuple(patternsize))    
    bc1 = np.bincount(kmeans_img1.labels_)
    bc2 = np.bincount(kmeans_img2.labels_)
    bc2 = bc2[(img_cluster_id_pairs_dist[:,1]).astype(int)]    
    tfs = 8
    fig_m = plt.figure(constrained_layout=True)
    gs = fig_m.add_gridspec(4, 7)
    fm_ax1 = fig_m.add_subplot(gs[:2,:2])
    fm_ax1.set_title('img1 level '+str(l)),fm_ax1.axis('off')
    fm_ax2 = fig_m.add_subplot(gs[2:,:2])
    fm_ax2.set_title('img2 level '+str(l)),fm_ax2.axis('off')
    fm_ax10 = fig_m.add_subplot(gs[0,2])
    fm_ax10.axis('off'),fm_ax10.set_title(str(bc1[1]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax11 = fig_m.add_subplot(gs[0,3])
    fm_ax11.axis('off'),fm_ax11.set_title(str(bc1[1]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[1,0]).astype(int))+' - '+
    fm_ax12 = fig_m.add_subplot(gs[0,4])
    fm_ax12.axis('off'),fm_ax12.set_title(str(bc1[2]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[2,0]).astype(int))+' - '+
    fm_ax13 = fig_m.add_subplot(gs[0,5])
    fm_ax13.axis('off'),fm_ax13.set_title(str(bc1[3]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[3,0]).astype(int))+' - '+
    fm_ax14 = fig_m.add_subplot(gs[0,6])
    fm_ax14.axis('off'),fm_ax14.set_title(str(bc1[4]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[4,0]).astype(int))+' - '+
    fm_ax15 = fig_m.add_subplot(gs[1,2])
    fm_ax15.axis('off'),fm_ax15.set_title(str(bc1[5]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[5,0]).astype(int))+' - '+
    fm_ax16 = fig_m.add_subplot(gs[1,3])
    fm_ax16.axis('off'),fm_ax16.set_title(str(bc1[6]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[6,0]).astype(int))+' - '+
    fm_ax17 = fig_m.add_subplot(gs[1,4])
    fm_ax17.axis('off'),fm_ax17.set_title(str(bc1[7]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[7,0]).astype(int))+' - '+
    fm_ax18 = fig_m.add_subplot(gs[1,5])
    fm_ax18.axis('off'),fm_ax18.set_title(str(bc1[8]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[8,0]).astype(int))+' - '+
    fm_ax19 = fig_m.add_subplot(gs[1,6])
    fm_ax19.axis('off'),fm_ax19.set_title(str(bc1[9]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[9,0]).astype(int))+' - '+    
    fm_ax20 = fig_m.add_subplot(gs[2,2])
    fm_ax20.axis('off'),fm_ax20.set_title(str(bc2[0]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,1]).astype(int))+' - '+
    fm_ax21 = fig_m.add_subplot(gs[2,3])
    fm_ax21.axis('off'),fm_ax21.set_title(str(bc2[1]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[1,1]).astype(int))+' - '+
    fm_ax22 = fig_m.add_subplot(gs[2,4])
    fm_ax22.axis('off'),fm_ax22.set_title(str(bc2[2]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[2,1]).astype(int))+' - '+
    fm_ax23 = fig_m.add_subplot(gs[2,5])
    fm_ax23.axis('off'),fm_ax23.set_title(str(bc2[3]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[3,1]).astype(int))+' - '+
    fm_ax24 = fig_m.add_subplot(gs[2,6])
    fm_ax24.axis('off'),fm_ax24.set_title(str(bc2[4]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[4,1]).astype(int))+' - '+
    fm_ax25 = fig_m.add_subplot(gs[3,2])
    fm_ax25.axis('off'),fm_ax25.set_title(str(bc2[5]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[5,1]).astype(int))+' - '+
    fm_ax26 = fig_m.add_subplot(gs[3,3])
    fm_ax26.axis('off'),fm_ax26.set_title(str(bc2[6]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[6,1]).astype(int))+' - '+
    fm_ax27 = fig_m.add_subplot(gs[3,4])
    fm_ax27.axis('off'),fm_ax27.set_title(str(bc2[7]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[7,1]).astype(int))+' - '+
    fm_ax28 = fig_m.add_subplot(gs[3,5])
    fm_ax28.axis('off'),fm_ax28.set_title(str(bc2[8]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[8,1]).astype(int))+' - '+
    fm_ax29 = fig_m.add_subplot(gs[3,6])
    fm_ax29.axis('off'),fm_ax29.set_title(str(bc2[9]),fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[9,1]).astype(int))+' - '+   
    if uniqueColorScale==True:
        vmin1 = np.amin(img1)
        vmin2 = np.amin(img2)
        vmax1 = np.amax(img1)
        vmax2 = np.amax(img2)
        vmin = np.min([vmin1,vmin2])
        vmax = np.min([vmax1,vmax2])
        fm_ax1.imshow(img1,vmin=vmin,vmax=vmax) 
        fm_ax2.imshow(img2,vmin=vmin,vmax=vmax)
        fm_ax10.imshow(c10,vmin=vmin,vmax=vmax)
        fm_ax11.imshow(c11,vmin=vmin,vmax=vmax)
        fm_ax12.imshow(c12,vmin=vmin,vmax=vmax)
        fm_ax13.imshow(c13,vmin=vmin,vmax=vmax)
        fm_ax14.imshow(c14,vmin=vmin,vmax=vmax)
        fm_ax15.imshow(c15,vmin=vmin,vmax=vmax)
        fm_ax16.imshow(c16,vmin=vmin,vmax=vmax)
        fm_ax17.imshow(c17,vmin=vmin,vmax=vmax)
        fm_ax18.imshow(c18,vmin=vmin,vmax=vmax)
        fm_ax19.imshow(c19,vmin=vmin,vmax=vmax)
        fm_ax20.imshow(c20,vmin=vmin,vmax=vmax)
        fm_ax21.imshow(c21,vmin=vmin,vmax=vmax)
        fm_ax22.imshow(c22,vmin=vmin,vmax=vmax)
        fm_ax23.imshow(c23,vmin=vmin,vmax=vmax)
        fm_ax24.imshow(c24,vmin=vmin,vmax=vmax)
        fm_ax25.imshow(c25,vmin=vmin,vmax=vmax)
        fm_ax26.imshow(c26,vmin=vmin,vmax=vmax)
        fm_ax27.imshow(c27,vmin=vmin,vmax=vmax)
        fm_ax28.imshow(c28,vmin=vmin,vmax=vmax)
        fm_ax29.imshow(c29,vmin=vmin,vmax=vmax)
    else:
        fm_ax1.imshow(img1) #,vmin=vmin,vmax=vmax
        fm_ax2.imshow(img2) #,vmin=vmin,vmax=vmax 
        fm_ax10.imshow(c10)
        fm_ax11.imshow(c11)
        fm_ax12.imshow(c12)
        fm_ax13.imshow(c13)
        fm_ax14.imshow(c14)
        fm_ax15.imshow(c15)
        fm_ax16.imshow(c16)
        fm_ax17.imshow(c17)
        fm_ax18.imshow(c18)
        fm_ax19.imshow(c19)    
        fm_ax20.imshow(c20)
        fm_ax21.imshow(c21)
        fm_ax22.imshow(c22)
        fm_ax23.imshow(c23)
        fm_ax24.imshow(c24)
        fm_ax25.imshow(c25)
        fm_ax26.imshow(c26)
        fm_ax27.imshow(c27)
        fm_ax28.imshow(c28)
        fm_ax29.imshow(c29)
    plt.show()
    return

def kldiv(pVec1,pVec2,base,divtype):
    eps2 = np.finfo('float').eps**2
    pVec1 = pVec1 + eps2 
    pVec2 = pVec2 + eps2
    if divtype=='kl':
        KL = np.sum(pVec1 * np.log(pVec1 / pVec2) / np.log(base))
    elif divtype=='js':
        pM = (pVec1 + pVec2)/2
        KL = 0.5 * np.sum(pVec1 * np.log(pVec1 / pM) / np.log(base)) + 0.5 * np.sum(pVec2 * np.log(pVec2 / pM) / np.log(base))
    elif divtype=='sym':
        KL = ( np.sum(pVec1 * np.log(pVec1 / pVec2) / np.log(base)) + np.sum(pVec2 * np.log(pVec2 / pVec1) / np.log(base)) ) /2
    return KL

def jsdist_hist(img1,img2,nbins,base,plot=False):
    tmp_min = np.min([np.amin(img1),np.amin(img2)])
    tmp_max = np.max([np.amax(img1),np.amax(img2)])
    binedges = np.linspace(tmp_min,tmp_max,num=int(nbins+1))
    p1,_ = np.histogram(img1,bins=binedges)/np.prod(img1.shape)
    p2,_ = np.histogram(img2,bins=binedges)/np.prod(img2.shape)
    if plot:
        ix = np.where((p1>0) | (p2>0))
        X = np.round((binedges[1:]+binedges[:-1])/2,2)
        X_axis = np.arange(len(X[ix]))
        plt.bar(X_axis - 0.2, p1[ix], 0.4, label = 'img1')
        plt.bar(X_axis + 0.2, p2[ix], 0.4, label = 'img2')  
        plt.xticks(X_axis, X[ix])
        plt.xlabel("Property Values")
        plt.ylabel("Proportion")
        plt.title("Density histogram")
        plt.legend()
        plt.show()        
    return kldiv(p1,p2,base,'js')

def indicator_lag_connectivity(array,xxx,yyy,zzz,nblags,maxh,maxnbsamples,clblab='',verb=False):
    lag_count = np.zeros(nblags)+np.nan # lag center
    lag_proba = np.zeros(nblags)+np.nan # connectivity probability
    lag_center = (np.arange(nblags)+1)*maxh/nblags # count per lag  
    if np.sum(array)==0:
        return lag_center,lag_count,lag_proba
    array_size = np.prod(array.shape)
    laglim = np.linspace(0,maxh,nblags+1)
    clblabed_array, num_features = label(array) # clblab array
    clblabed_array = np.reshape(clblabed_array,(array_size,1)).flatten()
    ix_c = (np.asarray(np.where(clblabed_array>0))).flatten()
    ix_rn = (np.round(np.random.uniform(0,1,int(min(maxnbsamples,np.sum(array),len(ix_c)))) * (np.sum(array)-1))).astype(int)
    samples_ix = ix_c[ix_rn]
    samples_val = clblabed_array[samples_ix]
    samples_xxx = np.reshape(xxx,(array_size,1)).flatten()[samples_ix]
    samples_yyy = np.reshape(yyy,(array_size,1)).flatten()[samples_ix]
    samples_zzz = np.reshape(zzz,(array_size,1)).flatten()[samples_ix]  
    # compute distance and square diff between sampled pair of points 
    dist = np.zeros(np.round(len(samples_ix)*(len(samples_ix)-1)/2).astype(int))+np.nan
    conn = np.zeros(np.round(len(samples_ix)*(len(samples_ix)-1)/2).astype(int))+np.nan
    k=0
    if verb:
        print('computing distance and connexion for each sampled pair of point')
    for i in range(len(samples_ix)):
        for j  in np.arange(i):
            dist[k] = ( (samples_xxx[i]-samples_xxx[j])**2 + (samples_yyy[i]-samples_yyy[j])**2 + (samples_zzz[i]-samples_zzz[j])**2 )**0.5
            conn[k] = 1 - ( samples_val[i] != samples_val[j] )**2
            k += 1
    # for each lag
    if verb:
        print('computing connexion probability per lag')
    for l in range(nblags):
        # identify sampled pairs belonging to the lag
        lag_lb = laglim[l]
        lag_ub = laglim[l+1]
        ix = np.where((dist>=lag_lb) & (dist<lag_ub))
        # count, experimental semi vario value and center of lag cloud
        lag_count[l]=len(ix[0])
        if len(ix[0])>0:
            lag_center[l]=np.mean(dist[ix])
            lag_proba[l]=np.mean(conn[ix])       
    return lag_center,lag_count,lag_proba

def weighted_lpnorm(array1,array2,p,weights=np.array([]),verb=False):
    if weights.shape!=array1.shape:
        weights=np.ones(array1.shape)
    if verb:
        print('weights: '+np.array2string(weights, precision=2, separator=','))
    ix2keep=np.where((np.isnan(array1) | np.isnan(array2))==False)
    w=weights[ix2keep]/np.sum(weights[ix2keep])
    L=(np.sum(w*(np.abs(array1[ix2keep]-array2[ix2keep]))**p))**(1/p)
    return L

def plot_ind_cty(img1,img2,lag_xc1,lag_cp1,lag_xc2,lag_cp2,classcode,clblab=""):
    ndim = len(img1.shape)
    vmin = np.min([np.min(img1),np.min(img2)])
    vmax = np.max([np.max(img1),np.max(img2)])
    if ndim==3:
        fig = plt.figure()
        gs = fig.add_gridspec(2,7)
        ax00 = fig.add_subplot(gs[0:, 3])
        ax01 = fig.add_subplot(gs[0, 0])
        ax02 = fig.add_subplot(gs[0, 1])
        ax03 = fig.add_subplot(gs[0, 2])
        ax11 = fig.add_subplot(gs[1, 0])
        ax12 = fig.add_subplot(gs[1, 1])
        ax13 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[0:, 4:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')
        ax03.axis('off')
        ax01.set_title('img1 Map')
        ax02.set_title('img1 W (N) E')
        ax03.set_title('img1 N (W) S')
        ax11.axis('off')
        ax12.axis('off')
        ax13.axis('off')
        ax11.set_title('img2 Map')
        ax12.set_title('img2 W (N) E')
        ax13.set_title('img2 N (W) S')
        ax4.set_title("img code "+str(classcode)+" connectivity")
        pos01=ax01.imshow(img1[0,:,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax02.imshow(img1[:,0,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax03.imshow(img1[:,:,0],cmap='rainbow',vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=clblab) 
        ax11.imshow(img2[0,:,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax12.imshow(img2[:,0,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax13.imshow(img2[:,:,0],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_cp1, 'ro-')
        ax4.plot(lag_xc2, lag_cp2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("lag distance [px]",fontsize=14)
        ax4.set_ylabel("Connectivity probability") #,fontsize=14
    if ndim==2:
        fig = plt.figure()
        gs = fig.add_gridspec(1,5)
        ax00 = fig.add_subplot(gs[0, 2])
        ax01 = fig.add_subplot(gs[0, 0])
        ax11 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[0, 3:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax01.set_title('img1')
        ax11.axis('off')
        ax11.set_title('img2')
        ax4.set_title("img code "+str(classcode)+" connectivity")
        pos01=ax01.imshow(img1,cmap='rainbow',vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=clblab) 
        ax11.imshow(img2,cmap='rainbow',vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_cp1, 'ro-')
        ax4.plot(lag_xc2, lag_cp2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("lag distance [px]",fontsize=14)
        ax4.set_ylabel("Connectivity probability") #,fontsize=14
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.55, wspace=0.1, hspace=0.5)
    plt.show()
    return       

def plot_pct_lag_cty(img1,img2,extent,low_cp_pct1,low_cp_pct2,hig_cp_pct1,hig_cp_pct2,clblab='',verb=False):
    ndim = len(img1.shape)
    if ndim==3:
        fig = plt.figure()
        gs = fig.add_gridspec(2,7)
        ax3 = fig.add_subplot(gs[0:, 3])
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax02 = fig.add_subplot(gs[0, 2])
        ax04 = fig.add_subplot(gs[0, 4])
        ax05 = fig.add_subplot(gs[0, 5])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        ax12 = fig.add_subplot(gs[1, 2])
        ax14 = fig.add_subplot(gs[1, 4])
        ax15 = fig.add_subplot(gs[1, 5])
        ax6 = fig.add_subplot(gs[0:, 6])
        axins3 = inset_axes(ax3,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        axins6 = inset_axes(ax6,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')
        ax3.axis('off')
        # ax04.axis('off')
        # ax05.axis('off')
        ax6.axis('off')
        ax00.set_title('img1 Map')
        ax01.set_title('img1 W (N) E')
        ax02.set_title('img1 N (W) S')
        ax04.set_title('img1 l-conn.')
        ax05.set_title('img1 h-conn.')
        ax10.axis('off')
        ax11.axis('off')
        ax12.axis('off')
        # ax14.axis('off')
        # ax15.axis('off')
        ax10.set_title('img2 Map')
        ax11.set_title('img2 W (N) E')
        ax12.set_title('img2 N (W) S')
        ax14.set_title('img2 l-conn.')
        ax15.set_title('img2h-conn.')
        pos00=ax00.imshow(img1[0,:,:],cmap='rainbow')
        ax01.imshow(img1[:,0,:],cmap='rainbow')
        ax02.imshow(img1[:,:,0],cmap='rainbow')
        fig.colorbar(pos00,cax=axins3,label=clblab) 
        ax10.imshow(img2[0,:,:],cmap='rainbow')
        ax11.imshow(img2[:,0,:],cmap='rainbow')
        ax12.imshow(img2[:,:,0],cmap='rainbow')
        pos04=ax04.imshow(low_cp_pct1,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax05.imshow(hig_cp_pct1,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        fig.colorbar(pos04,cax=axins6,label=clblab) 
        ax04.set_ylabel("percentile") #,fontsize=14
        ax04.set_xticks([])
        ax04.set_xticklabels([])
        ax05.set_xticks([])
        ax05.set_xticklabels([])
        ax05.set_yticks([])
        ax05.set_yticklabels([])
        ax14.imshow(low_cp_pct2,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax15.imshow(hig_cp_pct2,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax14.set_xlabel("lag distance [px]")
        ax14.set_ylabel("percentile") #,fontsize=14
        ax15.set_xlabel("lag distance [px]")
        ax15.set_yticks([])
        ax15.set_yticklabels([])
        
    if ndim==2:
        fig = plt.figure()
        gs = fig.add_gridspec(2,5)
        ax1 = fig.add_subplot(gs[0:, 1])
        ax00 = fig.add_subplot(gs[0, 0])
        ax02 = fig.add_subplot(gs[0, 2])
        ax03 = fig.add_subplot(gs[0, 3])
        ax10 = fig.add_subplot(gs[1, 0])
        ax12 = fig.add_subplot(gs[1, 2])
        ax13 = fig.add_subplot(gs[1, 3])
        ax2 = fig.add_subplot(gs[0:, 4])
        ax02.set_xticks([])
        ax02.set_xticklabels([])
        ax03.set_xticks([])
        ax03.set_xticklabels([])
        ax03.set_yticks([])
        ax03.set_yticklabels([])
        ax13.set_yticks([])
        ax13.set_yticklabels([])
        axins1 = inset_axes(ax1,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        axins2 = inset_axes(ax2,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax1.axis('off')
        ax2.axis('off')
        ax00.axis('off')
        # ax02.axis('off')
        # ax03.axis('off')
        ax10.axis('off')
        # ax12.axis('off')
        # ax13.axis('off')
        ax00.set_title('img1 Map')
        ax02.set_title('img1 l-conn.')
        ax03.set_title('img1 h-conn.')
        ax10.set_title('img2 Map')
        ax12.set_title('img2 l-conn.')
        ax13.set_title('img2h-conn.')
        pos00=ax00.imshow(img1,cmap='rainbow')
        fig.colorbar(pos00,cax=axins1,label=clblab) 
        ax10.imshow(img2,cmap='rainbow')
        pos02=ax02.imshow(low_cp_pct1,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax03.imshow(hig_cp_pct1,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        fig.colorbar(pos02,cax=axins2,label=clblab) 
        # ax02.set_xlabel("lag distance [px]")
        ax02.set_ylabel("percentile") #,fontsize=14
        # ax03.set_xlabel("lag distance [px]")
        ax03.set_ylabel("percentile") #,fontsize=14
        ax12.imshow(low_cp_pct2,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax13.imshow(hig_cp_pct2,origin='lower',extent=extent,cmap='rainbow',vmin=0,vmax=1)
        ax12.set_xlabel("lag distance [px]")
        ax12.set_ylabel("percentile") #,fontsize=14
        ax13.set_xlabel("lag distance [px]")
        ax13.set_ylabel("percentile") #,fontsize=14   
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.55, wspace=0.01, hspace=0.5)
    plt.show()
    
    return

def dist_lpnorm_categorical_lag_connectivity(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,clblab='',plot=False,verb=False):
    d=0
    # identify all indicators
    indicators = np.unique(np.hstack((img1.flatten(),img2.flatten())))
    nbind = len(indicators)
    d_ind = np.zeros(nbind)
    # for all indicators
    for i in range(nbind):
        classcode = indicators[i]
        if verb:
            print('indicator '+str(i))
        img1bin = ((img1==classcode)*1).astype(int)
        img2bin = ((img2==classcode)*1).astype(int)
        img1cnt=np.sum(img1bin)
        img2cnt=np.sum(img2bin)
        if img1cnt+img2cnt==0:
            d_ind[i] = 0
        elif img1cnt*img2cnt==0:
            d_ind[i] = 1/nbind
        else:
            if verb:
                print('img1 compute indicator_lag_connectivity')
            [lag_xc1,lag_ct1,lag_cp1] = indicator_lag_connectivity(img1bin,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
            if verb:
                print('img2 compute indicator_lag_connectivity')
            [lag_xc2,lag_ct2,lag_cp2] = indicator_lag_connectivity(img2bin,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
            d_ind[i] = weighted_lpnorm(lag_cp1,lag_cp2,pnorm,verb=verb)/nbind
        d += d_ind[i]
        if verb:
            print('distance contribution: '+str(d_ind[i]))
        if plot:
            plot_ind_cty(img1,img2,lag_xc1,lag_cp1,lag_xc2,lag_cp2,classcode,clblab=clblab)
    return d #, d_ind, id_ind

def dist_lpnorm_percentile_lag_connectivity(img1,img2,xxx,yyy,zzz,npctiles,nblags,maxh,maxnbsamples,pnorm,clblab='',plot=False,verb=False):
    d=0
    d_pct=np.zeros(npctiles)
    pctiles = (np.arange(npctiles)+1)*100/npctiles
    lag_center = (np.arange(nblags)+1)*maxh/nblags
    th_pct1 = np.percentile(img1,pctiles)
    th_pct2 = np.percentile(img2,pctiles)
    low_xc_pct1 = np.ones((npctiles,nblags))*np.nan
    low_ct_pct1 = np.ones((npctiles,nblags))*np.nan
    low_cp_pct1 = np.ones((npctiles,nblags))*np.nan
    hig_xc_pct1 = np.ones((npctiles,nblags))*np.nan
    hig_ct_pct1 = np.ones((npctiles,nblags))*np.nan
    hig_cp_pct1 = np.ones((npctiles,nblags))*np.nan
    low_xc_pct2 = np.ones((npctiles,nblags))*np.nan
    low_ct_pct2 = np.ones((npctiles,nblags))*np.nan
    low_cp_pct2 = np.ones((npctiles,nblags))*np.nan
    hig_xc_pct2 = np.ones((npctiles,nblags))*np.nan
    hig_ct_pct2 = np.ones((npctiles,nblags))*np.nan
    hig_cp_pct2 = np.ones((npctiles,nblags))*np.nan
    for i in range(npctiles):
        # lower parts
        img1_low = ((img1<=th_pct1[i])*1.0).astype(int)
        img2_low = ((img2<=th_pct2[i])*1.0).astype(int)
        img1cntl=np.sum(img1_low)
        img2cntl=np.sum(img2_low)
        if verb: 
            print(str(pctiles[i])+"th percentile connectivity - lower img1")
        if img1cntl==0:
            low_xc1 = lag_center
            low_ct1 = np.ones(nblags)*np.nan
            low_cp1 = np.ones(nblags)*np.nan
        else:
            [low_xc1,low_ct1,low_cp1] = indicator_lag_connectivity(img1_low,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
        if verb: 
            print(str(pctiles[i])+"th percentile connectivity - lower img2")
        if img2cntl==0:
            low_xc2 = lag_center
            low_ct2 = np.ones(nblags)*np.nan
            low_cp2 = np.ones(nblags)*np.nan
        else:
            [low_xc2,low_ct2,low_cp2] = indicator_lag_connectivity(img2_low,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
        low_xc_pct1[i,:]=low_xc1
        low_ct_pct1[i,:]=low_ct1
        low_cp_pct1[i,:]=low_cp1
        low_xc_pct2[i,:]=low_xc2
        low_ct_pct2[i,:]=low_ct2
        low_cp_pct2[i,:]=low_cp2
        # upper parts
        img1_hig = ((img1>th_pct1[i])*1.0).astype(int)
        img2_hig = ((img2>th_pct2[i])*1.0).astype(int)
        img1cnth=np.sum(img1_hig)
        img2cnth=np.sum(img2_hig)
        if verb: 
            print(str(pctiles[i])+"th percentile connectivity - upper img1")
        if img1cnth==0:
            hig_xc1 = lag_center
            hig_ct1 = np.ones(nblags)*np.nan
            hig_cp1 = np.ones(nblags)*np.nan
        else:
            [hig_xc1,hig_ct1,hig_cp1] = indicator_lag_connectivity(img1_hig,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
        if verb: 
            print(str(pctiles[i])+"th percentile connectivity - upper img2")
        if img2cnth==0:
            hig_xc2 = lag_center
            hig_ct2 = np.ones(nblags)*np.nan
            hig_cp2 = np.ones(nblags)*np.nan
        else:
            [hig_xc2,hig_ct2,hig_cp2] = indicator_lag_connectivity(img2_hig,xxx,yyy,zzz,nblags,maxh,maxnbsamples,verb=verb)
        hig_xc_pct1[i,:]=hig_xc1
        hig_ct_pct1[i,:]=hig_ct1
        hig_cp_pct1[i,:]=hig_cp1
        hig_xc_pct2[i,:]=hig_xc2
        hig_ct_pct2[i,:]=hig_ct2
        hig_cp_pct2[i,:]=hig_cp2
        # compute distance
        if img1cntl+img2cntl==0:
            d_low = 0
        elif img1cntl*img2cntl==0:
            d_low = 1
        else:
            d_low = weighted_lpnorm(low_cp1,low_cp2,pnorm,verb=verb)
        if img1cnth+img2cnth==0:
            d_hig = 0
        elif img1cnth*img2cnth==0:
            d_hig = 1
        else:
            d_hig = weighted_lpnorm(hig_cp1,hig_cp2,pnorm,verb=verb)
        d_pct[i] = (d_low + d_hig )*0.5/npctiles
        d += d_pct[i]
        if verb:
            print('distance contribution: '+str(d_pct[i]))
    if verb:
        print('total distance: '+str(d))
    # plot option
    if plot:
        extent = 0,maxh,0,100
        plot_pct_lag_cty(img1,img2,extent,low_cp_pct1,low_cp_pct2,hig_cp_pct1,hig_cp_pct2,clblab=clblab,verb=verb)
    return d #, d_ind, id_ind

def continuous_pct_connectivity(array,npctiles,verb=False):
    if verb:
        print('Computing global percentile connectivity')
    pctiles = np.linspace(100/npctiles,100,npctiles)
    low_connect = np.zeros(npctiles)+np.nan
    hig_connect = np.zeros(npctiles)+np.nan
    th_pct = np.percentile(array,pctiles)
    for i in range(npctiles):
        if verb:
            print(str(pctiles[i])+'th percentile - global connectivity')
        array_low = array<=th_pct[i]
        lab_low, num_features_low = label(array_low)
        cnt_low = np.zeros(num_features_low)
        for j in range(num_features_low):
            cnt_low[j]=np.sum((lab_low==j+1)*1.0)
        array_hig = array>th_pct[i]
        lab_hig, num_features_hig = label(array_hig)
        cnt_hig = np.zeros(num_features_hig)
        for j in range(num_features_hig):
            cnt_hig[j]=np.sum((lab_hig==j+1)*1.0)
        nb_low = np.sum(array_low)
        nb_hig = np.sum(array_hig)
        low_connect[i] = np.sum((cnt_low/nb_low)**2)
        hig_connect[i] = np.sum((cnt_hig/nb_hig)**2)
    return low_connect,hig_connect,pctiles

def plot_pct_cty(img1,img2,pctiles,low_connect1,low_connect2,hig_connect1,hig_connect2,clblab=''):
    ndim = len(img1.shape)
    if ndim==3:
        fig = plt.figure()
        gs = fig.add_gridspec(2,11)       
        ax00 = fig.add_subplot(gs[0, 0:2])
        ax01 = fig.add_subplot(gs[0, 2:4])
        ax02 = fig.add_subplot(gs[0, 4:6])
        ax03 = fig.add_subplot(gs[0, 6])
        ax04 = fig.add_subplot(gs[0, 8:])
        ax10 = fig.add_subplot(gs[1, 0:2])
        ax11 = fig.add_subplot(gs[1, 2:4])
        ax12 = fig.add_subplot(gs[1, 4:6])
        ax13 = fig.add_subplot(gs[1, 6])
        ax14 = fig.add_subplot(gs[1, 8:])
        axins03 = inset_axes(ax03,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        axins13 = inset_axes(ax13,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')
        ax03.axis('off')
        ax10.axis('off')
        ax11.axis('off')
        ax12.axis('off')
        ax13.axis('off')
        ax00.set_title("Map img1")
        ax01.set_title("W (N) E img1")
        ax02.set_title("N (W) S img1")
        ax10.set_title("Map img2")
        ax11.set_title("W (N) E img2")
        ax12.set_title("N (W) S img2")
        ax04.set_title("Continuous global connectivity")
        pos00=ax00.imshow(img1[0,:,:])
        ax01.imshow(img1[:,0,:])
        ax02.imshow(img1[:,:,0])
        fig.colorbar(pos00,cax=axins03,label=clblab) 
        pos10=ax10.imshow(img2[0,:,:])
        ax11.imshow(img2[:,0,:])
        ax12.imshow(img2[:,:,0])
        fig.colorbar(pos10,cax=axins13,label=clblab) 
        ax04.plot(pctiles,low_connect1,'b-')
        ax04.plot(pctiles,hig_connect1,'r--')
        ax04.set_xticklabels([])
        ax04.set_xticks([])
        ax04.set_ylabel('of connexion')
        ax04.legend(('lower img1', 'upper img1'),loc='best')
        ax14.plot(pctiles,low_connect2,'b-')
        ax14.plot(pctiles,hig_connect2,'r--')
        ax14.set_xlabel('Percentile threshold')
        ax14.set_ylabel('Probability')
        ax14.legend(('lower img2', 'upper img2'),loc='best')
        fig.subplots_adjust(left=0.0, bottom=0.0, right=2.05, top=0.75, wspace=0, hspace=0.25)
        plt.show()
    if ndim==2:
        fig = plt.figure()
        gs = fig.add_gridspec(2,7)       
        ax00 = fig.add_subplot(gs[0, 0:2])
        ax01 = fig.add_subplot(gs[0, 2])
        ax02 = fig.add_subplot(gs[0, 4:])
        ax10 = fig.add_subplot(gs[1, 0:2])
        ax11 = fig.add_subplot(gs[1, 2])
        ax12 = fig.add_subplot(gs[1, 4:])
        axins01 = inset_axes(ax01,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        axins11 = inset_axes(ax11,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax10.axis('off')
        ax11.axis('off')
        ax00.set_title("Map img1")
        ax10.set_title("Map img2")
        ax02.set_title("Continuous global connectivity")
        pos00=ax00.imshow(img1)
        fig.colorbar(pos00,cax=axins01,label=clblab) 
        pos10=ax10.imshow(img2)
        fig.colorbar(pos10,cax=axins11,label=clblab) 
        ax02.plot(pctiles,low_connect1,'b-')
        ax02.plot(pctiles,hig_connect1,'r--')
        ax02.set_xticklabels([])
        ax02.set_xticks([])
        ax02.set_ylabel('of connexion')
        ax02.legend(('lower img1', 'upper img1'),loc='best')
        ax12.plot(pctiles,low_connect2,'b-')
        ax12.plot(pctiles,hig_connect2,'r--')
        ax12.set_xlabel('Percentile threshold')
        ax12.set_ylabel('Probability')
        ax12.legend(('lower img2', 'upper img2'),loc='best')
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.55, top=0.75, wspace=0, hspace=0.25)
        plt.show() 
    return

def dist_lpnorm_percentile_global_connectivity(img1,img2,npctiles,pnorm,clblab='',plot=False,verb=False):
    d=0
    if verb: 
        print('global connectivity img1')
    [low_connect1,hig_connect1,pctiles] = continuous_pct_connectivity(img1,npctiles,verb=verb)
    if verb: 
        print('global connectivity img2')
    [low_connect2,hig_connect2,pctiles] = continuous_pct_connectivity(img2,npctiles,verb=verb)
    d_low = weighted_lpnorm(low_connect1,low_connect2,pnorm,verb=verb)
    d_hig = weighted_lpnorm(hig_connect1,hig_connect2,pnorm,verb=verb)
    d = (d_low + d_hig)/2
    if verb: 
        print('d = '+str(d))
    if plot:
        plot_pct_cty(img1,img2,pctiles,low_connect1,low_connect2,hig_connect1,hig_connect2,clblab=clblab)
    return d #, d_ind, indicators


def experimental_variogram(array,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=False):
    rng = default_rng(seed)
    ndim = len(array.shape)
    if ndim==3:
        [nz,ny,nx]=array.shape
    elif ndim==2:
        [ny,nx]=array.shape
        nz=1
    if verb:
        print(str(ndim)+'D data - experimental semi-variogram computation')
    laglim = np.linspace(0,maxh,nblags+1)
    lag_xc = np.ones(nblags)*np.nan
    lag_sv = np.ones(nblags)*np.nan
    lag_ct = np.ones(nblags)*np.nan
    if nz*ny*nx<=maxnbsamples:
        sv_samples_ix = np.arange(nz*ny*nx)
    else:
        sv_samples_ix =(np.round(rng.uniform(0,1,maxnbsamples) * (nz*ny*nx-1) )).astype(int)

    sv_samples_val = np.reshape(array,(nz*ny*nx,1))[sv_samples_ix]
    sv_samples_xxx = np.reshape(xxx,(nz*ny*nx,1))[sv_samples_ix]
    sv_samples_yyy = np.reshape(yyy,(nz*ny*nx,1))[sv_samples_ix]
    sv_samples_zzz = np.reshape(zzz,(nz*ny*nx,1))[sv_samples_ix]
    
    # compute distance and square diff between sampled pair of points 
    sv_dist = np.ones(np.round(len(sv_samples_ix)*(len(sv_samples_ix)-1)/2).astype(int))*np.nan
    sv_sqdf = np.ones(np.round(len(sv_samples_ix)*(len(sv_samples_ix)-1)/2).astype(int))*np.nan
    k=0
    for i in range(len(sv_samples_ix)):
        for j  in np.arange(i):
            sv_dist[k] = ( (sv_samples_xxx[i]-sv_samples_xxx[j])**2 + (sv_samples_yyy[i]-sv_samples_yyy[j])**2 + (sv_samples_zzz[i]-sv_samples_zzz[j])**2 )**0.5
            sv_sqdf[k] = ( sv_samples_val[i] - sv_samples_val[j] )**2
            k += 1
    # for each lag
    for l in range(nblags):
        # identify sampled pairs belonging to the lag
        lag_lb = laglim[l]
        lag_ub = laglim[l+1]
        ix = np.where((sv_dist>=lag_lb) & (sv_dist<lag_ub))
        # count, experimental semi vario value and center of lag cloud
        lag_ct[l]=len(ix[0])
        lag_xc[l]=np.mean(sv_dist[ix])
        lag_sv[l]=np.mean(sv_sqdf[ix])*0.5    
    return lag_xc, lag_sv, lag_ct

def plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,label):
    ndim = len(img1.shape)
    vmin = np.min([np.min(img1),np.min(img2)])
    vmax = np.max([np.max(img1),np.max(img2)])
    if ndim==3:
        fig = plt.figure()
        gs = fig.add_gridspec(2,7)
        ax00 = fig.add_subplot(gs[0:, 3])
        ax01 = fig.add_subplot(gs[0, 0])
        ax02 = fig.add_subplot(gs[0, 1])
        ax03 = fig.add_subplot(gs[0, 2])
        ax11 = fig.add_subplot(gs[1, 0])
        ax12 = fig.add_subplot(gs[1, 1])
        ax13 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[0:, 4:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')
        ax03.axis('off')
        ax01.set_title('img1 Map')
        ax02.set_title('img1 W (N) E')
        ax03.set_title('img1 N (W) S')
        ax11.axis('off')
        ax12.axis('off')
        ax13.axis('off')
        ax11.set_title('img2 Map')
        ax12.set_title('img2 W (N) E')
        ax13.set_title('img2 N (W) S')
        ax4.set_title("experimental variogram")
        pos01=ax01.imshow(img1[0,:,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax02.imshow(img1[:,0,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax03.imshow(img1[:,:,0],cmap='rainbow',vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=label) 
        ax11.imshow(img2[0,:,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax12.imshow(img2[:,0,:],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax13.imshow(img2[:,:,0],cmap='rainbow',vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_sv1, 'ro-')
        ax4.plot(lag_xc2, lag_sv2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("$h$ - lag distance [px]",fontsize=14)
        ax4.set_ylabel("$\gamma(h)$") #,fontsize=14
    if ndim==2:
        fig = plt.figure()
        gs = fig.add_gridspec(1,5)
        ax00 = fig.add_subplot(gs[0, 2])
        ax01 = fig.add_subplot(gs[0, 0])
        ax11 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[0, 3:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax01.set_title('img1')
        ax11.axis('off')
        ax11.set_title('img2')
        ax4.set_title("experimental variogram")
        pos01=ax01.imshow(img1,cmap='rainbow',vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=label) 
        ax11.imshow(img2,cmap='rainbow',vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_sv1, 'ro-')
        ax4.plot(lag_xc2, lag_sv2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("$h$ - lag distance [px]",fontsize=14)
        ax4.set_ylabel("$\gamma(h)$") #,fontsize=14
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.55, wspace=0.1, hspace=0.5)
    plt.show()
    return       

def dist_experimental_variogram_continuous(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label="",verb=False,plot=False):
    d=0
    if verb:
        print('img1')
    [lag_xc1, lag_sv1, lag_ct1] = experimental_variogram(img1,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
    if verb:
        print('img2')
    [lag_xc2, lag_sv2, lag_ct2] = experimental_variogram(img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
    if verb:
        print('distance computation')
    w = 2/(lag_xc1+lag_xc2)
    d = weighted_lpnorm(lag_sv1,lag_sv2,pnorm,weights=w,verb=verb)
    if plot==True:
        plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,label)
    return d

def dist_experimental_variogram_categorical(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label="",verb=False,plot=False):
    classvalues = np.unique(np.hstack((img1,img2)))
    nbclasses = len(classvalues)
    d=0
    dbyclass = np.zeros(nbclasses)
    for c in range(nbclasses):
        current_class = classvalues[c]
        tmp1 = ((img1==current_class)*1).astype(int)
        tmp2 = ((img2==current_class)*1).astype(int)
        curr_label = label + " " + str(current_class)
        if verb:
            print('img1 '+curr_label)
        [lag_xc1, lag_sv1, lag_ct1] = experimental_variogram(tmp1,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
        if verb:
            print('img2 '+curr_label)
        [lag_xc2, lag_sv2, lag_ct2] = experimental_variogram(tmp2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
        w = 2/(lag_xc1+lag_xc2)
        dbyclass[c] = weighted_lpnorm(lag_sv1,lag_sv2,pnorm,weights=w,verb=verb)
        d+=dbyclass[c]
        if verb:
            print('distance '+ curr_label +": "+str(dbyclass))
        if plot==True:
            plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,curr_label)        
    return d

def dist_experimental_variogram(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,categ=False,label="",verb=False,plot=False):
    d=-1
    if categ==False:
        d = dist_experimental_variogram_continuous(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label=label,verb=verb,plot=plot)
    elif categ==True:
        d = dist_experimental_variogram_categorical(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label=label,verb=verb,plot=plot)
    return d

def kldiv(pVec1,pVec2,base,divtype):
    eps2 = np.finfo('float').eps**2
    pVec1 = pVec1 + eps2 
    pVec2 = pVec2 + eps2
    if divtype=='kl':
        KL = np.sum(pVec1 * np.log(pVec1 / pVec2) / np.log(base))
    elif divtype=='js':
        pM = (pVec1 + pVec2)/2
        KL = 0.5 * np.sum(pVec1 * np.log(pVec1 / pM) / np.log(base)) + 0.5 * np.sum(pVec2 * np.log(pVec2 / pM) / np.log(base))
    elif divtype=='sym':
        KL = ( np.sum(pVec1 * np.log(pVec1 / pVec2) / np.log(base)) + np.sum(pVec2 * np.log(pVec2 / pVec1) / np.log(base)) ) /2
    return KL

def dist_wavelet2D(img1,img2,n_levels,n_bins,plot=False,verb=False):
    w = 1/(0+n_levels*4) # uniform weight per histogram  in the JS divergence
    DIV = 0
    if plot:
        img1old=img1
        img2old=img2
    for l in range(n_levels):
        if verb:
            print('Level '+str(l)+' img1 size: '+str(img1.shape[0])+'*'+str(img1.shape[1]))
        cA1, [cH1, cV1, cD1] = pywt.dwt2(img1,'haar')
        cA2, [cH2, cV2, cD2] = pywt.dwt2(img2,'haar')
        tmp_min = np.min([np.amin(cA1),np.amin(cA2)])
        tmp_max = np.max([np.amax(cA1),np.amax(cA2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pA1,_ = np.histogram(cA1,bins=binedges)/np.prod(cA1.shape)
        pA2,_ = np.histogram(cA2,bins=binedges)/np.prod(cA2.shape)
        tmp_min = np.min([np.amin(cH1),np.amin(cH2)])
        tmp_max = np.max([np.amax(cH1),np.amax(cH2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pH1,_ = np.histogram(cH1,bins=binedges)/np.prod(cH1.shape)
        pH2,_ = np.histogram(cH2,bins=binedges)/np.prod(cH2.shape)
        tmp_min = np.min([np.amin(cV1),np.amin(cV2)])
        tmp_max = np.max([np.amax(cV1),np.amax(cV2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pV1,_ = np.histogram(cV1,bins=binedges)/np.prod(cV1.shape)
        pV2,_ = np.histogram(cV2,bins=binedges)/np.prod(cV2.shape)
        tmp_min = np.min([np.amin(cD1),np.amin(cD2)])
        tmp_max = np.max([np.amax(cD1),np.amax(cD2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pD1,_ = np.histogram(cD1,bins=binedges)/np.prod(cD1.shape)
        pD2,_ = np.histogram(cD2,bins=binedges)/np.prod(cD2.shape)
        DIV +=  w * ( kldiv(pA1,pA2,base,'js') + kldiv(pH1,pH2,base,'js') + kldiv(pV1,pV2,base,'js') + kldiv(pD1,pD2,base,'js') )
        if verb:
            print('Distance component: '+str(w * ( kldiv(pA1,pA2,base,'js') + kldiv(pH1,pH2,base,'js') + kldiv(pV1,pV2,base,'js') + kldiv(pD1,pD2,base,'js') )))
        if plot:
            plot_wvt2Ddec(img1old,img1,cA1,cH1,cV1,cD1,img2old,img2,cA2,cH2,cV2,cD2,l)
        del cH1, cV1, cD1,cH2, cV2, cD2,tmp_min,tmp_max,binedges #,img1loc,img2loc
        img1 = cA1
        img2 = cA2
        del cA1,cA2
    return DIV

def dist_wavelet3D(img1,img2,n_levels,n_bins,plot=False,verb=False):
    w = 1/(0+n_levels*8) # uniform weight per histogram  in the JS divergence
    DIV = 0
    for l in range(n_levels):
        if verb:
            print('Level '+str(l)+' img1 size: '+str(img1.shape[0])+'*'+str(img1.shape[1])+'*'+str(img1.shape[2]))
        # {'aad', 'ada', 'daa', 'add', 'dad', 'dda', 'ddd'}
        coeffs1 = pywt.dwtn(img1,'haar') #coeffs1['aaa']
        if verb:
            print('Level '+str(l)+' coeffs size: '+str(coeffs1['aaa'].shape[0])+'*'+str(coeffs1['aaa'].shape[1])+'*'+str(coeffs1['aaa'].shape[2]))
        cA1 = coeffs1['aaa']
        caad1 = coeffs1['aad']
        cada1 = coeffs1['ada']
        cdaa1 = coeffs1['daa']
        cadd1 = coeffs1['add']
        cdad1 = coeffs1['dad']
        cdda1 = coeffs1['dda']
        cddd1 = coeffs1['ddd']
        coeffs2 = pywt.dwtn(img2,'haar')
        cA2 = coeffs2['aaa']
        caad2 = coeffs2['aad']
        cada2 = coeffs2['ada']
        cdaa2 = coeffs2['daa']
        cadd2 = coeffs2['add']
        cdad2 = coeffs2['dad']
        cdda2 = coeffs2['dda']
        cddd2 = coeffs2['ddd']
        tmp_min = np.min([np.amin(cA1),np.amin(cA2)])
        tmp_max = np.max([np.amax(cA1),np.amax(cA2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pA1,_ = np.histogram(cA1,bins=binedges)/np.prod(cA1.shape)
        pA2,_ = np.histogram(cA2,bins=binedges)/np.prod(cA2.shape)
        tmp_min = np.min([np.amin(caad1),np.amin(caad2)])
        tmp_max = np.max([np.amax(caad1),np.amax(caad2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        paad1,_ = np.histogram(caad1,bins=binedges)/np.prod(caad1.shape)
        paad2,_ = np.histogram(caad2,bins=binedges)/np.prod(caad2.shape)
        tmp_min = np.min([np.amin(cada1),np.amin(cada2)])
        tmp_max = np.max([np.amax(cada1),np.amax(cada2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pada1,_ = np.histogram(cada1,bins=binedges)/np.prod(cada1.shape)
        pada2,_ = np.histogram(cada2,bins=binedges)/np.prod(cada2.shape)
        tmp_min = np.min([np.amin(cdaa1),np.amin(cdaa2)])
        tmp_max = np.max([np.amax(cdaa1),np.amax(cdaa2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pdaa1,_ = np.histogram(cdaa1,bins=binedges)/np.prod(cdaa1.shape)
        pdaa2,_ = np.histogram(cdaa2,bins=binedges)/np.prod(cdaa2.shape)
        tmp_min = np.min([np.amin(cadd1),np.amin(cadd2)])
        tmp_max = np.max([np.amax(cadd1),np.amax(cadd2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        padd1,_ = np.histogram(cadd1,bins=binedges)/np.prod(cadd1.shape)
        padd2,_ = np.histogram(cadd2,bins=binedges)/np.prod(cadd2.shape)
        tmp_min = np.min([np.amin(cdad1),np.amin(cdad2)])
        tmp_max = np.max([np.amax(cdad1),np.amax(cdad2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pdad1,_ = np.histogram(cdad1,bins=binedges)/np.prod(cdad1.shape)
        pdad2,_ = np.histogram(cdad2,bins=binedges)/np.prod(cdad2.shape)
        tmp_min = np.min([np.amin(cdda1),np.amin(cdda2)])
        tmp_max = np.max([np.amax(cdda1),np.amax(cdda2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pdda1,_ = np.histogram(cdda1,bins=binedges)/np.prod(cdda1.shape)
        pdda2,_ = np.histogram(cdda2,bins=binedges)/np.prod(cdda2.shape)
        tmp_min = np.min([np.amin(cddd1),np.amin(cddd2)])
        tmp_max = np.max([np.amax(cddd1),np.amax(cddd2)])
        binedges = np.linspace(tmp_min,tmp_max,num=int(n_bins+1))
        pddd1,_ = np.histogram(cddd1,bins=binedges)/np.prod(cddd1.shape)
        pddd2,_ = np.histogram(cddd2,bins=binedges)/np.prod(cddd2.shape)
        DIV +=  w * kldiv(pA1,pA2,base,'js')
        DIV +=  w * kldiv(paad1,paad2,base,'js')
        DIV +=  w * kldiv(pada1,pada2,base,'js')
        DIV +=  w * kldiv(pdaa1,pdaa2,base,'js')
        DIV +=  w * kldiv(padd1,padd2,base,'js')
        DIV +=  w * kldiv(pdad1,pdad2,base,'js')
        DIV +=  w * kldiv(pdda1,pdda2,base,'js')
        DIV +=  w * kldiv(pddd1,pddd2,base,'js')
        if verb:
            print('Distance component: '+str(w*(kldiv(pA1,pA2,base,'js')+kldiv(paad1,paad2,base,'js')+kldiv(pada1,pada2,base,'js')+kldiv(pdaa1,pdaa2,base,'js')+kldiv(padd1,padd2,base,'js')+kldiv(pdad1,pdad2,base,'js')+kldiv(pdda1,pdda2,base,'js')+kldiv(pddd1,pddd2,base,'js'))))
        if plot:
            plot_wvt3Ddec(coeffs1,coeffs2,l)
        del cA1, cA2, caad1, cada1, cdaa1, cadd1, cdad1, cdda1, cddd1, caad2,
        cada2, cdaa2, cadd2, cdad2, cdda2, cddd2, tmp_min, tmp_max, binedges
        img1 = coeffs1['aaa']
        img2 = coeffs2['aaa']
        del coeffs1,coeffs2
    return DIV

def dist_wavelet(img1,img2,n_levels,n_bins,plot=False,verb=False):
    dim = len(img1.shape)
    if dim==2:
        DIV = dist_wavelet2D(img1,img2,n_levels,n_bins,plot=plot,verb=verb)
    elif dim==3:
        DIV = dist_wavelet3D(img1,img2,n_levels,n_bins,plot=plot,verb=verb)
    else:
        DIV = np.nan
    return DIV


def plot_wvt2Ddec(img1old,img1,cA1,cH1,cV1,cD1,img2old,img2,cA2,cH2,cV2,cD2,l):
    coeffs1 = pywt.wavedec2(img1, 'haar',level=l+1) # [cAn,(cH3, cV3, cD3),(cH2, cV2, cD2),(cH1, cV1, cD1)]
    coeffs2 = pywt.wavedec2(img2, 'haar',level=l+1) # [cAn,(cH3, cV3, cD3),(cH2, cV2, cD2),(cH1, cV1, cD1)]
    array1 = pywt.coeffs_to_array(coeffs1, padding=0, axes=None)
    array2 = pywt.coeffs_to_array(coeffs2, padding=0, axes=None)
    tfs = 8
    fig, ax = plt.subplots(2,4)
    ax[0,0].imshow(img1old),ax[0,0].axis('off'),ax[0,0].set_title('img1-ini',fontsize=tfs)
    ax[1,0].imshow(img2old),ax[1,0].axis('off'),ax[1,0].set_title('img2-ini',fontsize=tfs)
    ax[0,1].imshow(array1[0]),ax[0,1].axis('off'),ax[0,1].set_title('img1-level '+str(l+1),fontsize=tfs) #,cmap='rainbow'
    ax[1,1].imshow(array2[0]),ax[1,1].axis('off'),ax[1,1].set_title('img2-level '+str(l+1),fontsize=tfs) #,cmap='rainbow'
    cAs = np.reshape(np.transpose(np.array([cA1.flatten(),cA2.flatten()])),(np.prod(cA1.shape),2))
    cHs = np.reshape(np.transpose(np.array([cH1.flatten(),cH2.flatten()])),(np.prod(cH1.shape),2))
    cVs = np.reshape(np.transpose(np.array([cV1.flatten(),cV2.flatten()])),(np.prod(cV1.shape),2))
    cDs = np.reshape(np.transpose(np.array([cD1.flatten(),cD2.flatten()])),(np.prod(cD1.shape),2))
    mylegend=['img1','img2']
    ax[0,2].axis('off'),ax[0,2].hist(cAs, density=True, histtype='bar',label=mylegend),ax[0,2].legend(prop={'size': tfs}),ax[0,2].set_title('approx-coeffs',fontsize=tfs)
    ax[0,3].axis('off'),ax[0,3].hist(cHs, density=True, histtype='bar',label=mylegend),ax[0,3].legend(prop={'size': tfs}),ax[0,3].set_title('horizontal-coeffs',fontsize=tfs)
    ax[1,2].axis('off'),ax[1,2].hist(cVs, density=True, histtype='bar',label=mylegend),ax[1,2].legend(prop={'size': tfs}),ax[1,2].set_title('vertical-coeffs',fontsize=tfs)
    ax[1,3].axis('off'),ax[1,3].hist(cDs, density=True, histtype='bar',label=mylegend),ax[1,3].legend(prop={'size': tfs}),ax[1,3].set_title('diagonal-coeffs',fontsize=tfs)
    plt.show()
    return

def plot_wvt3Ddec(coeffs1,coeffs2,l):
    # img1 = img1[0,:,:]
    cA1 = coeffs1['aaa'][0,:,:]
    caad1 = coeffs1['aad'][0,:,:]
    cada1 = coeffs1['ada'][0,:,:]
    cdaa1 = coeffs1['daa'][0,:,:]
    cadd1 = coeffs1['add'][0,:,:]
    cdad1 = coeffs1['dad'][0,:,:]
    cdda1 = coeffs1['dda'][0,:,:]
    cddd1 = coeffs1['ddd'][0,:,:]
    arli1 = np.hstack((cA1,caad1,cada1,cdaa1))
    arli2 = np.hstack((cadd1,cdad1,cdda1,cddd1))
    array1 = np.vstack((arli1,arli2))
    # img2map = img2[0,:,:]
    cA2 = coeffs2['aaa'][0,:,:]
    caad2 = coeffs2['aad'][0,:,:]
    cada2 = coeffs2['ada'][0,:,:]
    cdaa2 = coeffs2['daa'][0,:,:]
    cadd2 = coeffs2['add'][0,:,:]
    cdad2 = coeffs2['dad'][0,:,:]
    cdda2 = coeffs2['dda'][0,:,:]
    cddd2 = coeffs2['ddd'][0,:,:]
    arli1 = np.hstack((cA2,caad2,cada2,cdaa2))
    arli2 = np.hstack((cadd2,cdad2,cdda2,cddd2))
    array2 = np.vstack((arli1,arli2))
    Caaa = np.reshape(np.transpose(np.array([coeffs1['aaa'].flatten(),coeffs2['aaa'].flatten()])),(np.prod(coeffs1['aaa'].shape),2))
    Caad = np.reshape(np.transpose(np.array([coeffs1['aad'].flatten(),coeffs2['aad'].flatten()])),(np.prod(coeffs1['aad'].shape),2))
    Cada = np.reshape(np.transpose(np.array([coeffs1['ada'].flatten(),coeffs2['ada'].flatten()])),(np.prod(coeffs1['ada'].shape),2))
    Cdaa = np.reshape(np.transpose(np.array([coeffs1['daa'].flatten(),coeffs2['daa'].flatten()])),(np.prod(coeffs1['daa'].shape),2))
    Cadd = np.reshape(np.transpose(np.array([coeffs1['add'].flatten(),coeffs2['add'].flatten()])),(np.prod(coeffs1['add'].shape),2))
    Cdad = np.reshape(np.transpose(np.array([coeffs1['dad'].flatten(),coeffs2['dad'].flatten()])),(np.prod(coeffs1['dad'].shape),2))
    Cdda = np.reshape(np.transpose(np.array([coeffs1['dda'].flatten(),coeffs2['dda'].flatten()])),(np.prod(coeffs1['dda'].shape),2))
    Cddd = np.reshape(np.transpose(np.array([coeffs1['ddd'].flatten(),coeffs2['ddd'].flatten()])),(np.prod(coeffs1['ddd'].shape),2))
    tfs = 8
    mylegend=['img1','img2']
    fig_m = plt.figure(constrained_layout=True)
    gs = fig_m.add_gridspec(3, 8)
    fm_ax1 = fig_m.add_subplot(gs[:2,:4])
    fm_ax1.set_title('img1 level '+str(l) +' decomposition')#,fm_ax1.axis('off')
    fm_ax1.imshow(array1)
    fm_ax2 = fig_m.add_subplot(gs[:2:,4:])
    fm_ax2.set_title('img2 level '+str(l) +' decomposition')#,fm_ax2.axis('off')
    fm_ax2.imshow(array2)
    fm_ax30 = fig_m.add_subplot(gs[2,0])
    fm_ax30.axis('off'),fm_ax30.set_title('aaa',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax30.hist(Caaa, density=True, histtype='bar',label=mylegend),fm_ax30.legend(prop={'size': tfs*3/4})

    fm_ax31 = fig_m.add_subplot(gs[2,1])
    fm_ax31.axis('off'),fm_ax31.set_title('aad',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax31.hist(Caad, density=True, histtype='bar',label=mylegend),fm_ax31.legend(prop={'size': tfs*3/4})

    fm_ax32 = fig_m.add_subplot(gs[2,2])
    fm_ax32.axis('off'),fm_ax32.set_title('ada',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax32.hist(Cada, density=True, histtype='bar',label=mylegend),fm_ax32.legend(prop={'size': tfs*3/4})

    fm_ax33 = fig_m.add_subplot(gs[2,3])
    fm_ax33.axis('off'),fm_ax33.set_title('daa',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax33.hist(Cdaa, density=True, histtype='bar',label=mylegend),fm_ax33.legend(prop={'size': tfs*3/4})

    fm_ax34 = fig_m.add_subplot(gs[2,4])
    fm_ax34.axis('off'),fm_ax34.set_title('add',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax34.hist(Cadd, density=True, histtype='bar',label=mylegend),fm_ax34.legend(prop={'size': tfs*3/4})

    fm_ax35 = fig_m.add_subplot(gs[2,5])
    fm_ax35.axis('off'),fm_ax35.set_title('dad',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax35.hist(Cdad, density=True, histtype='bar',label=mylegend),fm_ax35.legend(prop={'size': tfs*3/4})

    fm_ax36 = fig_m.add_subplot(gs[2,6])
    fm_ax36.axis('off'),fm_ax36.set_title('dda',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax36.hist(Cdda, density=True, histtype='bar',label=mylegend),fm_ax36.legend(prop={'size': tfs*3/4})

    fm_ax37 = fig_m.add_subplot(gs[2,7])
    fm_ax37.axis('off'),fm_ax37.set_title('ddd',fontsize=tfs) #'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax37.hist(Cddd, density=True, histtype='bar',label=mylegend),fm_ax37.legend(prop={'size': tfs*3/4})

    plt.show()
    print('plotting coeffs size ='+str(coeffs1['aaa'].shape[0])+'*'+str(coeffs1['aaa'].shape[1])+'*'+str(coeffs1['aaa'].shape[2]))
    del array1,array2
    return
