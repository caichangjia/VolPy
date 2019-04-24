#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:20:41 2019

@author: Changjia Cai
"""
import os
os.environ["MKL_NUM_THREADS"] = "8" 

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import dilation
from skimage.morphology import disk
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from scipy import ndimage
import scipy.io
from scipy.sparse.linalg import svds
from volpy_function import *
import h5py
import time

#%%
# opts
opts = {'doCrossVal':False, #cross-validate to optimize regression regularization parameters?
        'doGlobalSubtract':False,
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
output = {'rawROI':{}}


#%%
dr = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min'
fns = {1:'datasetblock1.mat'}
rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
fn_ix = 1
cellN = 0
#%%
# too slow, need refined
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
#notbw = 1-dilation(bw, disk(opts['censorSize']))

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


#%% remove low frequency components
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

#%% remove any variance in trace that can be predicted from the background PCs
Ub, Sb, Vb = svds(np.double(data_hp[notbw.T.ravel(),:]), opts['nPC_bg'])
reg = LinearRegression().fit(Vb.T,t)
reg.coef_
t = t - np.matmul(Vb.T,reg.coef_)

# data, windowLength, sampleRate, doPlot, doClip = [-t, opts['windowLength'], sampleRate, True, 100]

#%%
# May need modification here
Xspikes, spikeTimes, guessData, output['rawROI']['falsePosRate'], output['rawROI']['detectionRate'], output['rawROI']['templates'], low_spk = denoiseSpikes(-t, opts['windowLength'], sampleRate, True, 100)

#%%
Xspikes = -Xspikes
output['rawROI']['X'] = t
output['rawROI']['Xspikes'] = Xspikes
output['rawROI']['spikeTimes'] = spikeTimes
output['rawROI']['spatialFilter'] = bw
output['rawROI']['X'] = output['rawROI']['X']*np.mean(t[output['rawROI']['spikeTimes']])/np.mean(output['rawROI']['X'][output['rawROI']['spikeTimes']]) # correct shrinkage

selectSpikes = np.zeros(Xspikes.shape)
selectSpikes[spikeTimes] = 1
sgn = np.mean(Xspikes[selectSpikes>0])
noise = np.std(Xspikes[selectSpikes==0])
snr = sgn/noise

#%% prebuild the regression matrix
# generate a predictor for ridge regression
pred = np.transpose(np.vstack((np.ones((1,data_pred.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_pred, (ref.shape[0], ref.shape[1], data_hp.shape[1]), order='F'), sigma=(1.5,1.5,0), truncate=2, mode='nearest'),data.shape, order='F'))))

#%% To do: if not enough spikes, take spatial filter from previous block
#%% Cross-validation of regularized regression parameters
lambdamax = np.linalg.norm(pred[1:,:], ord='fro') ** 2
lambdas = lambdamax * np.logspace(-4, -2, 3)
I0 = np.eye(pred.shape[1])
I0[0,0] = 0

if opts['doCrossVal']:
    # need to add
    print('doing cross validation')
else:
    s_max = 1
    l_max = 2
    opts['lambda'] = lambdas[l_max]
    opts['sigma'] = opts['sigmas'][s_max]
    opts['lambda_ix'] = l_max
    
selectPred = np.ones(data.shape[1])
if opts['highPassRegression']:
    selectPred[:np.int16(sampleRate/2+1)] = 0
    selectPred[-1-np.int16(sampleRate/2):] = 0

sigma = opts['sigmas'][s_max]

pred = np.transpose(np.vstack((np.ones((1,data_pred.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_pred, (ref.shape[0], ref.shape[1], data_hp.shape[1]), order='F'), sigma=(sigma,sigma,0), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest'),data.shape, order='F'))))
recon = np.transpose(np.vstack((np.ones((1,data_hp.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_hp, (ref.shape[0], ref.shape[1], data_hp.shape[1]), order='F'), sigma=(sigma,sigma,0), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest'),data.shape, order='F'))))
temp = np.linalg.inv(np.matmul(np.transpose(pred[selectPred>0,:]), pred[selectPred>0,:]) + lambdas[l_max] * I0)
kk = np.matmul(temp, np.transpose(pred[selectPred>0,:]))
#%% Identify spatial filters with regularized regression
for iteration in range(opts['nIter']):
    doPlot = False
    if iteration == opts['nIter'] - 1:
        doPlot = True
        
    print('Identifying spatial filters')
    print(iteration)
    gD = guessData[selectPred>0]
    select = (gD!=0)
    weights = np.matmul(kk[:,select], gD[select])
    
    X = np.double(np.matmul(recon, weights))
    X = X - np.mean(X)
    
    a=np.reshape(weights[1:], ref.shape, order='F')
    
    spatialFilter = ndimage.gaussian_filter(a, sigma=(sigma,sigma), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest')
    plt.imshow(spatialFilter)
    plt.show()
    
    if iteration < opts['nIter'] - 1:
        b = LinearRegression().fit(Vb.T,X).coef_
        if doPlot:
            plt.figure()
            plt.plot(X)
            plt.plot(np.matmul(Vb.T,b))
            plt.title('Denoised trace vs background')
            plt.show()
        X = X - np.matmul(Vb.T,b)
    else:
        if opts['doGlobalSubtract']:
            print('do global subtract')
            # need to add
            
    # correct shrinkage
    X = X * np.mean(t[spikeTimes]) / np.mean(X[spikeTimes])
    
    # generate the new trace and the new denoised trace
    Xspikes, spikeTimes, guessData, falsePosRate, detectionRate, templates, _ = denoiseSpikes(-X, opts['windowLength'], sampleRate, doPlot)
    selectSpikes = np.zeros(Xspikes.shape)
    selectSpikes[spikeTimes] = 1
    sgn = np.mean(Xspikes[selectSpikes>0])
    noise = np.std(Xspikes[selectSpikes==0])
    snr = sgn/noise

#%% ensure that the maximum of the spatial filter is within the ROI
matrix = np.matmul(np.transpose(pred[:, 1:]), -guessData)  
sigmax = np.sqrt(np.sum(np.multiply(pred[:, 1:], pred[:, 1:]), axis=0))
sigmay = np.sqrt(np.dot(guessData, guessData))
IMcorr = matrix/sigmax/sigmay
maxCorrInROI = np.max(IMcorr[bw.T.ravel()])
if np.any(IMcorr[notbw.ravel()]>maxCorrInROI):
    output['passedLocalityTest'] = False
else:
    output['passedLocalityTest'] = True
    
#%% compute SNR
selectSpikes = np.zeros(Xspikes.shape)
selectSpikes[spikeTimes] = 1
sgn = np.mean(Xspikes[selectSpikes>0])
noise = np.std(Xspikes[selectSpikes==0])
snr = sgn/noise
output['snr'] = snr


#%% output
output['y'] = X
output['yFilt'] = -Xspikes
output['ROI'] = np.transpose(np.vstack((Xinds[[0,-1]], Yinds[[0,-1]])))
output['ROIbw'] = bw
output['spatialFilter'] = spatialFilter
output['falsePosRate'] = falsePosRate
output['detectionRate'] = detectionRate
output['templates'] = templates
output['spikeTimes'] = spikeTimes
output['opts'] = opts
output['F0'] = np.nanmean(np.double(data_lp[bw.T.flatten(), :]) + output['meanIM'][bw][:,np.newaxis], 0) 
output['dFF'] = X / output['F0']
output['rawROI']['dFF'] = output['rawROI']['X'] / output['F0']
output['Vb'] = Vb    # background components
output['low_spk'] = low_spk

#%% save







#%% denoiseSpikes

#%%
tic = time.time()
elapse = time.time() - tic



