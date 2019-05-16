#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:14:37 2019

@author: Changjia Cai
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
sys.path.append('/home/nel/Code/VolPy')

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import dilation
from skimage.morphology import disk
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from scipy import ndimage
from scipy import stats
from scipy import io
from scipy.sparse.linalg import svds
import h5py
import time
import cv2
import caiman as cm
from volpy_function import *
import multiprocessing as mp
from caiman.base.movies import movie



#%%
rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
f = io.loadmat(rois_path)
ROIs = f['roi'].T
sampleRate = 400

#fname_new = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'

fname_new = '/media/nel/ssd/data/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'
args = []

for i in range(6):
    args.append([fname_new, i, ROIs[i,:,:]])

#pars = args[3]

c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=6, single_thread=False)

results = dview.map_async(spikePursuit_parallel, args).get()
dview.close()

#results = spikePursuit_parallel(args[0])
#df_single_1 = results
df_double_6 = results
#df_single_6 = results
#df_single_12 = results


#%%
d6 = [i['time'] for i in df_double_6]
s1 = df_single_1['time']
s6 = [i['time'] for i in df_single_6]
s12 = [i['time'] for i in df_single_12]

d6 = sorted(d6, key=lambda x: x['total'], reverse=True)[0]
s6 = sorted(s6, key=lambda x: x['total'], reverse=True)[0]
s12 = sorted(s12, key=lambda x: x['total'], reverse=True)[0]

l = [d6, s1, s6, s12]

N = 4
p1 = [i['process1'] for i in l]
p2 = [i['process2'] for i in l]
p3 = [i['process3'] for i in l]
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

f1 = plt.bar(ind, p1, width)
f2 = plt.bar(ind, p2, width,
             bottom=p1)
f3 = plt.bar(ind, p3, width, bottom=[a + b for a, b in zip(p1, p2)])

plt.ylabel('Total Time')
plt.title('Total time by different cpu and number of cells')
plt.xticks(ind, ('double_6', 'single_1', 'single_6', 'single_12'))
plt.legend(('Process1', 'Process2', 'Process3'))

plt.savefig('Total time vs CPU and number of cells.jpg')



#%%
d6 = [i['time'] for i in df_double_6]
s1 = df_single_1['time']
s6 = [i['time'] for i in df_single_6]
s12 = [i['time'] for i in df_single_12]

d6 = sorted(d6, key=lambda x: x['total'], reverse=True)[0]
s6 = sorted(s6, key=lambda x: x['total'], reverse=True)[0]
s12 = sorted(s12, key=lambda x: x['total'], reverse=True)[0]

d6 = {k:v/6 for k, v in d6.items()}
s6 = {k:v/6 for k, v in s6.items()}
s12 = {k:v/12 for k, v in s12.items()}


l = [d6, s1, s6, s12]

N = 4
p1 = [i['process1'] for i in l]
p2 = [i['process2'] for i in l]
p3 = [i['process3'] for i in l]
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

f1 = plt.bar(ind, p1, width)
f2 = plt.bar(ind, p2, width,
             bottom=p1)
f3 = plt.bar(ind, p3, width, bottom=[a + b for a, b in zip(p1, p2)])

plt.ylabel('Average time')
plt.title('Average time by different cpu and number of cells')
plt.xticks(ind, ('double_6', 'single_1', 'single_6', 'single_12'))
plt.legend(('Process1', 'Process2', 'Process3'))

plt.savefig('Average time vs CPU and number of cells.jpg')
plt.show()




#%%
tic_total = time.time()
results = spikePursuit_parallel(args[0])
elapse = time.time() - tic_total
print('Use', elapse, 's IN TOTAL')

#%%
cellN = 0
fname = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/datasetblock1.hdf5'
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
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
#%%
images
Yr.shape
type(Yr)
#%%
%timeit np.array(Yr[:60*60])

#%%
'''
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
'''

#%%
@profile
def spikePursuit_parallel(pars):
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
    sampleRate = 400
    opts['windowLength'] = sampleRate*0.02 #window length for spike templates    
    #%%
    # Can not create same disk matrix as matlab, so load the matrix from matlab instead
    #g = io.loadmat('/home/nel/Code/Voltage_imaging/disk.mat')
    #disk_matrix = g['a']
    #
    # Compute global PCs with ROIs masked out 
    # To do
    #%%
    fname_new, cellN, bw = pars
    print('processing cell:', cellN)        
        
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    
    # extract relevant region and align
    bwexp = dilation(bw,np.ones([opts['contextSize'],opts['contextSize']]), shift_x=True, shift_y=True)
    
    Xinds = np.where(np.any(bwexp>0,axis=1)>0)[0]
    Yinds = np.where(np.any(bwexp>0,axis=0)>0)[0]       
    
    bw = bw[Xinds[0]:Xinds[-1]+1, Yinds[0]:Yinds[-1]+1]
    #notbw = 1-dilation(bw, disk_matrix)
    notbw = 1-dilation(bw, disk(opts['censorSize']))   
    
    tic = time.time()    
        
    data = np.array(images[:, Xinds[0]:Xinds[-1]+1, Yinds[0]:Yinds[-1]+1])
    
    elapse = time.time() - tic
    print('Use', elapse, 's')
    
    bw = (bw>0)
    notbw = (notbw>0)
    
    #%%
    # Notice:ROI selection is not the same as matlab
    ref = np.median(data[:500,:,:],axis=0)
    #fig = plt.figure()
    #plt.subplot(131);plt.imshow(ref);plt.axis('image');plt.xlabel('mean Intensity')
    #plt.subplot(132);plt.imshow(bw);plt.axis('image');plt.xlabel('initial ROI')
    #plt.subplot(133);plt.imshow(notbw);plt.axis('image');plt.xlabel('background')
    #fig.suptitle('ROI selection')
    #plt.show()  
    #%%
    # local Align
    # todo
    #%%
    output['meanIM'] = np.mean(data, axis=0)
    data = np.reshape(data, (data.shape[0],-1))  
    data = np.double(data)
    data = np.double(data-np.mean(data,0))
    data = np.double(data-np.mean(data,0))  
  
    #%% remove low frequency components
    data_hp = highpassVideo(data.T, 1/opts['tau_lp'], sampleRate).T
    data_lp = data-data_hp
    
    if opts['highPassRegression']:
        data_pred = highpassVideo(data, 1/opts['tau_pred'], sampleRate)
    else:
        data_pred = data_hp    
 
    #%%
    t = np.nanmean(np.double(data_hp[:,bw.ravel()]),1)
    t = t-np.mean(t)
    #plt.plot(t[0:200])        
    #%% remove any variance in trace that can be predicted from the background PCs
    Ub, Sb, Vb = svds(np.double(data_hp[:,notbw.ravel()]), opts['nPC_bg'])
    reg = LinearRegression(fit_intercept=False).fit(Ub,t)
    t = t - np.matmul(Ub,reg.coef_)
  
   

    # data, windowLength, sampleRate, doPlot, doClip = [-t, opts['windowLength'], sampleRate, True, 100]
    
    #%%
    # May need modification here
    tic = time.time()
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
    pred = np.hstack((np.ones((data_pred.shape[0], 1)), np.reshape
                     (movie.gaussian_blur_2D(np.reshape(data_pred, 
                       (data_hp.shape[0], ref.shape[0], ref.shape[1])),
                       kernel_size_x=7, kernel_size_y=7,kernel_std_x=1.5, 
                       kernel_std_y=1.5, borderType=cv2.BORDER_REPLICATE),data_hp.shape)))
     

    #%% To do: if not enough spikes, take spatial filter from previous block
    #%% Cross-validation of regularized regression parameters
    lambdamax = np.linalg.norm(pred[:,1:], ord='fro') ** 2
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
        
    selectPred = np.ones(data_hp.shape[0])
    if opts['highPassRegression']:
        selectPred[:np.int16(sampleRate/2+1)] = 0
        selectPred[-1-np.int16(sampleRate/2):] = 0
    
    sigma = opts['sigmas'][s_max]         
    
    pred = np.hstack((np.ones((data_pred.shape[0], 1)), np.reshape
                     (movie.gaussian_blur_2D(np.reshape(data_pred, 
                       (data_hp.shape[0], ref.shape[0], ref.shape[1])),
                       kernel_size_x=np.int(2*np.ceil(2*sigma)+1), 
                       kernel_size_y=np.int(2*np.ceil(2*sigma)+1),
                       kernel_std_x=sigma, kernel_std_y=sigma, 
                       borderType=cv2.BORDER_REPLICATE),data_hp.shape)))
        
    recon = np.hstack((np.ones((data_hp.shape[0], 1)), np.reshape
                     (movie.gaussian_blur_2D(np.reshape(data_hp, 
                       (data_hp.shape[0], ref.shape[0], ref.shape[1])),
                       kernel_size_x=np.int(2*np.ceil(2*sigma)+1), 
                       kernel_size_y=np.int(2*np.ceil(2*sigma)+1),
                       kernel_std_x=sigma, kernel_std_y=sigma, 
                       borderType=cv2.BORDER_REPLICATE),data_hp.shape)))
       
    temp = np.linalg.inv(np.matmul(np.transpose(pred[selectPred>0,:]), pred[selectPred>0,:]) + lambdas[l_max] * I0)
    kk = np.matmul(temp, np.transpose(pred[selectPred>0,:]))
    elapse = time.time() - tic
    print('Use', elapse, 's')


    #%% Identify spatial filters with regularized regression
    tic = time.time()
    for iteration in range(opts['nIter']):
        doPlot = False
        if iteration == opts['nIter'] - 1:
            doPlot = False
            
        print('Identifying spatial filters')
        print(iteration)
                    
        gD = guessData[selectPred>0]
        select = (gD!=0)
        weights = np.matmul(kk[:,select], gD[select])
                   
        X = np.double(np.matmul(recon, weights))
        X = X - np.mean(X)
        
        a=np.reshape(weights[1:], ref.shape, order='C')            
        spatialFilter = movie.gaussian_blur_2D(a[np.newaxis,:,:],
                       kernel_size_x=np.int(2*np.ceil(2*sigma)+1), 
                       kernel_size_y=np.int(2*np.ceil(2*sigma)+1),
                       kernel_std_x=sigma, kernel_std_y=sigma, 
                       borderType=cv2.BORDER_REPLICATE)[0]
        
        
        #plt.imshow(spatialFilter)
        #plt.show()
        
        if iteration < opts['nIter'] - 1:
            b = LinearRegression(fit_intercept=False).fit(Ub,X).coef_
            
            if doPlot:
                plt.figure()
                plt.plot(X)
                plt.plot(np.matmul(Ub,b))
                plt.title('Denoised trace vs background')
                plt.show()
            X = X - np.matmul(Ub,b)
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
    maxCorrInROI = np.max(IMcorr[bw.ravel()])
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
    output['F0'] = np.nanmean(np.double(data_lp[:,bw.flatten()]) + output['meanIM'][bw][np.newaxis,:], 1) 
    output['dFF'] = X / output['F0']
    output['rawROI']['dFF'] = output['rawROI']['X'] / output['F0']
    output['Vb'] = Vb    # background components
    output['low_spk'] = low_spk
    elapse = time.time() - tic
    print('Use', elapse, 's')

    
    return output
        
        
   
        
        
    #%% save
    
# Run the data

    
    
    
    
    
    #%% denoiseSpikes

#%%
tic = time.time()
elapse = time.time() - tic
print('Use', elapse, 's')










