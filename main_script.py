#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:20:41 2019

@author: Changjia
"""

import numpy as np
import scipy.io
import os
import h5py
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import dilation
from skimage.morphology import disk
from scipy import signal
#%%
# opts
opts = {'doCrossVal':False, #cross-validate to optimize regression regularization parameters?
        'contextSize':50,  #65; #number of pixels surrounding the ROI to use as context
        'censorSize':12, #number of pixels surrounding the ROI to censor from the background PCA; roughly the spatial scale of scattered/dendritic neural signals, in pixels.
        'nPC_bg':8, #number of principle components used for background subtraction
        'tau_lp':3, #time window for lowpass filter (seconds); signals slower than this will be ignored
        'tau_pred':1, #time window in seconds for high pass filtering to make predictor for regression
        'sigmas':np.array([1,1.5,2]), #spatial smoothing radius imposed on spatial filter;
        'nIter':5, #number of iterations alternating between estimating temporal and spatial filters.
        'localAlign':False, 
        'globalAlign':True,
        'highPassRegression':False #regress on a high-passed version of the data. Slightly improves detection of spikes, but makes subthreshold unreliable.
       }
output = {}


#%%
dr = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min'
fns = {1:'datasetblock1.mat'}
rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
fn_ix = 1
cellN = 0
#%%
print('Loading data batch: ', fns[fn_ix])
arrays = {}
f = h5py.File(dr+'/'+fns[fn_ix],'r')
#for k, v in f.items():
#    arrays[k] = np.array(v)
dataAll = np.array(f.get('data'))
dataAll = dataAll.transpose()

sampleRate = np.array(f.get('sampleRate'))
sampleRate = sampleRate[0][0]
print('sampleRate:',np.int(sampleRate))
opts['windowLength'] = sampleRate*0.02 #window length for spike templates
#%%
# Can not create same disk matrix as matlab, so load the matrix from matlab instead
g = scipy.io.loadmat('/home/nel/Code/Voltage_imaging/disk.mat')
disk_matrix = g['a']
#%%
# Compute global PCs with ROIs masked out 
# To do
#%%
f = scipy.io.loadmat(rois_path)
ROIs = f['roi']

bw = ROIs[:,:,cellN]

# extract relevant region and align
bwexp = dilation(bw,np.ones([opts['contextSize'],opts['contextSize']]), shift_x=True, shift_y=True)
Xinds = np.arange(np.where(np.any(bwexp>0,axis=0)>0)[0][0],np.where(np.any(bwexp>0,axis=0)>0)[0][-1]+1)
Yinds = np.arange(np.where(np.any(bwexp>0,axis=1)>0)[0][0],np.where(np.any(bwexp>0,axis=1)>0)[0][-1]+1)
bw = bw[np.ix_(Yinds,Xinds)]
notbw = 1-dilation(bw, disk_matrix)
# notbw = 1-dilation(bw, disk(opts['censorSize']))

data = dataAll[Yinds[:,np.newaxis],Xinds, :]
bw = (bw>0)
notbw = (notbw>0)



print('processing cell:', cellN)

#%%
# Notice:ROI selection is not the same as matlab
ref = np.median(data[:,:,:500],axis=2)
fig = plt.figure()
plt.subplot(131);plt.imshow(ref);plt.axis('image');plt.xlabel('mean Intensity')
plt.subplot(132);plt.imshow(bw);plt.axis('image');plt.xlabel('initial ROI')
plt.subplot(133);plt.imshow(notbw);plt.axis('image');plt.xlabel('background')
fig.suptitle('ROI selection')
plt.show()

#%%
# local Align
# todo

#%%
output['meanIM'] = np.mean(data, axis=2)
data = np.reshape(data, (-1, data.shape[2]), order='F')

data = np.double(data)
data = np.double(data-np.mean(data,1)[:,np.newaxis])
data = np.double(data-np.mean(data,1)[:,np.newaxis])

#%%
def highpassVideo(video, freq, sampleRate):
    normFreq = freq/(sampleRate/2)
    b, a = signal.butter(3, normFreq, 'high')
    videoFilt = signal.filtfilt(b, a, video, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return videoFilt

#%%
# remove low frequency components
data_hp = highpassVideo(data, 1/opts['tau_lp'], sampleRate)
data_lp = data-data_hp

if opts['highPassRegression']:
    data_pred = highpassVideo(data, 1/opts['tau_pred'], sampleRate)
else:
    data_pred = data_hp    

#%%
t = np.nanmean(np.double(data_hp[bw.T.ravel(),:]),0)
t = t-np.mean(t)
plt.plot(t[0:200])




