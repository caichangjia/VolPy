#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:02:18 2019

@author: Changjia
"""
#%%
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo

#%% First setup some parameters for data and motion correction
    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (1., 1.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (10., 10.)       # maximum shift in um
    patch_motion_um = (64., 64.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }

    opts = params.CNMFParams(params_dict=mc_dict)

#%%
dr = '/home/nel/Code/Voltage_imaging/exampledata/Un-aligned image files/403106_3min/'
total = len([f for f in os.listdir(dr) if f.endswith('.tif')])

n_blocks = 10
n_files = np.int(np.floor(total/10))

c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=16, single_thread=False)

for n in range(n_blocks):
    file_list = [dr + 'cameraTube051_{:05n}.tif'.format(i+1) for i in np.arange(n_files*n, n_files*(n+1), 1)]
    mv = cm.load(file_list)
    fnames = dr+'datablock_{:02n}.hdf5'.format(n+1)
    mv.save(fnames)    
    
    mc_dict['fnames'] = fnames
    opts = params.CNMFParams(params_dict=mc_dict)    
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)



#%%
#%% Preprocessing
import caiman as cm
cellN = 0
fname = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/datasetblock1.hdf5'
#%%
dr = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min'
fns = {1:'datasetblock1.mat'}
fn_ix = 1
print('Loading data batch: ', fns[fn_ix])
f = h5py.File(dr+'/'+fns[fn_ix],'r')
#for k, v in f.items():
#    arrays[k] = np.array(v)
dataAll = np.array(f.get('data'))
sampleRate = np.array(f.get('sampleRate'))
sampleRate = sampleRate[0][0]
print('sampleRate:',np.int(sampleRate))
#%%
cm.movie(dataAll).save(fname)
#%%
dataAll = cm.load(fname)
#%%
dataAll[:1000].play(fr=100)
#%%
cm.cluster.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=8, single_thread=False)
#%%
border_to_0 = 0
# memory map the file in order 'C'
fname_new = cm.save_memmap([fname], base_name='memmap_', order='C',
                               border_to_0=border_to_0, dview=dview)  # exclude borders
#%%
fname_new
#%% now load the file
fname_new = '/media/nel/ssd/data/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
#%%
images
Yr.shape
type(Yr)
