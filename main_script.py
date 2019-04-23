#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:20:41 2019

@author: Changjia Cai
"""

import numpy as np
import scipy.io
from scipy.sparse.linalg import svds
import os
import h5py
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import dilation
from skimage.morphology import disk
from scipy import signal
import time
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy import fftpack
import cv2
from scipy import ndimage
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
output = {'rawROI':{}}


#%%
dr = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min'
fns = {1:'datasetblock1.mat'}
rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
fn_ix = 1
cellN = 0
iteration = 1
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

#%%
def highpassVideo(video, freq, sampleRate):
    normFreq = freq/(sampleRate/2)
    b, a = signal.butter(3, normFreq, 'high')
    videoFilt = signal.filtfilt(b, a, video, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return videoFilt

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
pred = np.transpose(np.vstack((np.ones((1,data_pred.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_pred, (ref.shape[0], ref.shape[1], data.shape[1]), order='F'), sigma=(1.5,1.5,0), truncate=2, mode='nearest'),data.shape, order='F'))))

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


pred = np.transpose(np.vstack((np.ones((1,data_pred.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_pred, (ref.shape[0], ref.shape[1], data.shape[1]), order='F'), sigma=(sigma,sigma,0), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest'),data.shape, order='F'))))
recon = np.transpose(np.vstack((np.ones((1,data_hp.shape[1])), np.reshape(ndimage.gaussian_filter(np.reshape(data_hp, (ref.shape[0], ref.shape[1], data.shape[1]), order='F'), sigma=(sigma,sigma,0), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest'),data.shape, order='F'))))
temp = np.linalg.inv(np.matmul(np.transpose(pred[selectPred>0,:]), pred[selectPred>0,:]) + lambdas[l_max] * I0)
kk = np.matmul(temp, np.transpose(pred[selectPred>0,:]))

# Matrix multiplication and matrix inverse too slow
# Need to be modified
'''
tic = time.time()
elapse = time.time() - tic
np.linalg.inv(np.matmul(np.transpose(pred[selectPred>0,:]), pred[selectPred>0,:]) + lambdas[l_max] * I0)
kk = np.matmul(temp, np.transpose(pred[selectPred>0,:]))
tic = time.time()
np.matmul(np.transpose(pred[selectPred>0,:]), pred[selectPred>0,:])
elapse = time.time() - tic
'''

#%% Identify spatial filters with regularized regression
doPlot = False
if iteration == opts['nIter']:
    doPlot = True
    
print('Identifying spatial filters')
gD = guessData[selectPred>0]
select = (gD!=0)
weights = np.matmul(kk[:,select], gD[select])

X = np.double(np.matmul(recon, weights))
X = X - np.mean(X)

a=np.reshape(weights[1:], ref.shape, order='F')

spatialFilter = ndimage.gaussian_filter(a, sigma=(sigma,sigma), truncate=np.ceil((2*sigma-0.5)/sigma), mode='nearest')
plt.imshow(spatialFilter)
plt.show()

if iteration < opts['nIter']:
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
def denoiseSpikes(data, windowLength, sampleRate=500, doPlot=False, doClip=150):
    #%% highpass filter and threshold
    bb, aa = signal.butter(1, 1/(sampleRate/2), 'high') # 1Hz filter
    dataHP = signal.filtfilt(bb, aa, data).flatten()
    
    pks = dataHP[signal.find_peaks(dataHP, height=None)[0]]
    
    thresh, _, _, low_spk = getThresh(pks, doClip, 0.25)
    
    locs = signal.find_peaks(dataHP, height=thresh)[0]
    
    #%% peak-traiggered average
    window = np.int64(np.arange(-windowLength, windowLength+1, 1))
    locs = locs[np.logical_and(locs>(-window[0]), locs<(len(data)-window[-1]))]
    PTD = data[(locs[:,np.newaxis]+window)]
    PTA = np.mean(PTD, 0)
    
    # matched filter
    datafilt = whitenedMatchedFilter(data, locs, window)
    
    #%% spikes detected after filter
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    
    thresh2, falsePosRate, detectionRate, _ = getThresh(pks2, doClip, 0.5)
    spikeTimes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    guessData = np.zeros(data.shape)
    guessData[spikeTimes] = 1
    guessData = np.convolve(guessData, PTA, 'same')
    
    # filtering shrinks the data;
    # rescale so that the mean value at the peaks is same as in the input
    datafilt = datafilt * np.mean(data[spikeTimes]) / np.mean(datafilt[spikeTimes])
    
    # output templates
    templates = PTA
    
    #%% plot three graphs
    if doPlot:
       
       fig = plt.figure()
       ax1 = fig.add_subplot(2,1,1)
       ax1.hist(pks, 500)
       ax1.axvline(x=thresh, c='r')
       ax1.set_title('raw data')
       ax2 = fig.add_subplot(2,1,2)
       ax2.hist(pks2, 500)
       ax2.axvline(x=thresh2, c='r')
       ax2.set_title('after matched filter')
       plt.tight_layout()
       plt.show()
       
       fig = plt.plot()
       plt.plot(np.transpose(PTD), c=[0.5,0.5,0.5])
       plt.plot(PTA, c='black', linewidth=2)
       plt.title('Peak-triggered average')
       plt.show()
       
       fig = plt.figure()
       ax1 = fig.add_subplot(2,1,1)
       ax1.plot(data)
       ax1.plot(locs, np.max(datafilt)*1.1*np.ones(locs.shape), color='r', marker='o', fillstyle='none', linestyle='none')
       ax1.plot(spikeTimes, np.max(datafilt)*1*np.ones(spikeTimes.shape), color='g', marker='o', fillstyle='none', linestyle='none')
       ax2 = fig.add_subplot(2,1,2)
       ax2.plot(datafilt)
       ax2.plot(locs, np.max(datafilt)*1.1*np.ones(locs.shape), color='r', marker='o', fillstyle='none', linestyle='none')
       ax2.plot(spikeTimes, np.max(datafilt)*1*np.ones(spikeTimes.shape), color='g', marker='o', fillstyle='none', linestyle='none')
       plt.show()     
    return datafilt, spikeTimes, guessData, falsePosRate, detectionRate, templates, low_spk
  
#%% Get threshold
#g = scipy.io.loadmat('/home/nel/Code/Voltage_imaging/pks.mat')
#pks = g['pks']
def getThresh(pks, doClip, pnorm=0.5):    
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spk = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kernel = stats.gaussian_kde(pks,bw_method='silverman')
    f = kernel.evaluate(pts)
    xi = pts
    center = np.where(xi>np.median(pks))[0][0]
    #%%
    fmodel = np.concatenate([f[0:center+1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f)-len(fmodel))*min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]
    #%% adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1]>csmodel[0:-1]+np.spacing(1), csf[1:]<csmodel[1:]))[0]
     
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
        
    fmodel[0:lastpt+1] = f[0:lastpt+1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:],f[lastpt:])
    
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    
    maxind = np.argmax(obj)
    thresh = xi[maxind]
    
    if np.sum(pks>thresh)<30:
        low_spk = True
        print('Very few spikes were detected at the desired sensitivity/specificity tradeoff. Adjusting threshold to take 30 largest spikes')
        thresh = np.percentile(pks, 100*(1-30/len(pks)))
    elif np.sum(pks>thresh)>doClip:
        print('Selecting top',doClip,'spikes for template')
        thresh = np.percentile(pks, 100*(1-doClip/len(pks)))
        
    ix = np.argmin(np.abs(xi-thresh))
    falsePosRate = csmodel2[ix]/csf2[ix]
    detectionRate = (csf2[ix]-csmodel2[ix])/np.max(csf2-csmodel2)

    return thresh, falsePosRate, detectionRate, low_spk

#%% whitened Matched Filter
def whitenedMatchedFilter(data, locs, window):
    N = 2 * len(data) - 1
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor<0.5)
    
    noise = data[censor]
    _,pxx = signal.welch(noise, fs=2*np.pi,window=signal.get_window('hamming',1000),nfft=N, detrend=False)
    Nf2 = np.concatenate([pxx,np.flipud(pxx[:-1])])
    scaling = 1 / np.sqrt(Nf2)
    
    # need to be optimized for fft
    dataScaled = np.real(fftpack.ifft(fftpack.fft(data, N) * scaling))
    
    PTDscaled = dataScaled[(locs[:,np.newaxis]+window)]
    
    PTAscaled = np.mean(PTDscaled, 0)
    
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)] 
    
    return datafilt
    
#%%
A = np.array([1,2,3])
signal.find_peaks(A, 6)


#%%
tic = time.time()
elapse = time.time() - tic



