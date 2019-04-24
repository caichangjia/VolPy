#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:50:09 2019

@author: Changjia Cai
"""

import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import pyfftw

def denoiseSpikes(data, windowLength, sampleRate=500, doPlot=False, doClip=150):
    # highpass filter and threshold
    bb, aa = signal.butter(1, 1/(sampleRate/2), 'high') # 1Hz filter
    dataHP = signal.filtfilt(bb, aa, data).flatten()
    
    pks = dataHP[signal.find_peaks(dataHP, height=None)[0]]
    
    thresh, _, _, low_spk = getThresh(pks, doClip, 0.25)
    
    locs = signal.find_peaks(dataHP, height=thresh)[0]
    
    # peak-traiggered average
    window = np.int64(np.arange(-windowLength, windowLength+1, 1))
    locs = locs[np.logical_and(locs>(-window[0]), locs<(len(data)-window[-1]))]
    PTD = data[(locs[:,np.newaxis]+window)]
    PTA = np.mean(PTD, 0)
    
    # matched filter
    datafilt = whitenedMatchedFilter(data, locs, window)
    
    # spikes detected after filter
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
    
    # plot three graphs
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
    #
    fmodel = np.concatenate([f[0:center+1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f)-len(fmodel))*min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]
    # adjust the model so it doesn't exceed the data:
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
    
    # Use pyfftw
    a = pyfftw.empty_aligned(data.shape[0], dtype='float64')
    a[:] = data
    dataScaled = np.real(pyfftw.interfaces.scipy_fftpack.ifft(pyfftw.interfaces.scipy_fftpack.fft(a,N) * scaling))
    PTDscaled = dataScaled[(locs[:,np.newaxis]+window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)] 
    
    return datafilt    
#%%
def highpassVideo(video, freq, sampleRate):
    normFreq = freq/(sampleRate/2)
    b, a = signal.butter(3, normFreq, 'high')
    videoFilt = signal.filtfilt(b, a, video, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return videoFilt