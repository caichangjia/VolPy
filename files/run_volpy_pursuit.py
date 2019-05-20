#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:09:08 2019

@author: Changjia
"""
#%% Running spike pursuit function for voltage imaging
import sys
sys.path.append('/home/nel/Code/VolPy')

from volpy_function import *
import caiman as cm
from caiman.base.movies import movie

n_CPUs = 8
n_cells = 8

rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
f = io.loadmat(rois_path)
ROIs = f['roi'].T
sampleRate = 400

#fname_new = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'

fname_new = '/media/nel/ssd/data/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'
args = []

for i in range(n_cells):
    args.append([fname_new, i, ROIs[i,:,:], sampleRate])

#%% Process multiple neurons parallelly with multiple CPUs
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=n_CPUs, single_thread=False)

results = dview.map_async(spikePursuit, args).get()
dview.close()


#%% Process one single neuron
pars = args[3]
spikePursuit(pars)
