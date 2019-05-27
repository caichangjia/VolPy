#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:42:38 2019

@author: Changjia
"""

#%% Detecting neurons
import matplotlib
matplotlib.rcParams['interactive'] == True
rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
f = io.loadmat(rois_path)
ROIs = f['roi'].T
roi = np.sum(ROIs, axis=0)
plt.imshow(roi.transpose())


fname_new = '/media/nel/ssd/data/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
df = np.array(images[:3000])
data = df.reshape((df.shape[0], -1))
data = highpassVideo(data.T, 1/opts['tau_lp'], sampleRate).T
df = data.reshape(df.shape)




corr = np.zeros((3, 3, df.shape[1], df.shape[2]))
count = np.zeros((df.shape[1], df.shape[2]))
for j in np.arange(1, df.shape[1]-1):
    for k in np.arange(1, df.shape[2]-1):
        matrix = df[:, (j-1):(j+2), (k-1):(k+2)]
        matrix = np.divide((matrix-np.mean(matrix,axis=0)),np.sqrt(np.var(matrix,axis=0)))
        vector = df[:, j, k]
        vector = (vector-np.mean(vector))/np.sqrt(np.var(vector))
        corr[:, :, j, k] = np.einsum('i,ijk->jk',vector,matrix)/10000

for j in np.arange(1, df.shape[1]-1):
    for k in np.arange(1, df.shape[2]-1):
        count[j, k] = np.sum(corr[:,:,j,k]>0.1) - 1   
count1 = count>2        
plt.imshow(count.transpose())

#%%
mu = np.zeros(df.shape[0])
sigma_square = 0
nb = 1
n = 1

epsilon = np.zeros((df.shape[1], df.shape[2]))

for j in np.arange(1, df.shape[1]-1):
    for k in np.arange(1, df.shape[2]-1):
        mu = mu * (n-1) / n + df[:, j, k] / n
        if n > 1:
            sigma_square = sigma_square * (n-1) / n + np.dot(df[:, j, k] - mu, df[:, j, k] - mu) / (n-1)  
        if n > 20:
            epsilon[j, k] = 1  + np.dot(df[:, j, k] - mu, df[:, j, k] - mu) / (sigma_square)
        n = n + 1
        
prob =  1 / (epsilon)
prob[np.isinf(prob)] = 0
prob[prob>0.1] = 1
prob[prob<0.1] = 0
plt.imshow(prob)



#%%
'''A class for computing the eccentricity for a 2D image
Written by Hoss Eybposh, UNC-CH, NEL'''

import numpy as np

def get_mu_k(x_k, mu_k_1, k):
    """
    Get the mean value at k.
    Arguments:
    :param x_k: current value in the timeseries
    :param mu_k_1: mean from previous step
    :param k: step or frame number
    :return: mu_k, the next value for mean
    """

    return ((k - 1) / k) * mu_k_1 + x_k / k


def get_sigma_k(x_k, mu_k, sigma_k_1, k):
    '''

    :param x_k: current value in the timeseries
    :param mu_k: current mean value
    :param sigma_k_1: variance from previous step
    :param k: frame number
    :return: current variance
    '''
    return ((float(k) - 1.) / k) * sigma_k_1 + ((x_k - mu_k)**2) / (k - 1)


def get_ecc(x_k, mu_k, sigma_k, k):
    '''
    get current eccentricity value.
    :param x_k: current mean value
    :param mu_k: current mean value
    :param sigma_k: variance at current step
    :param k: step number
    :return: current eccentricity value
    '''
    #print('k is: ', k)
    #print('sigma k is: ', sigma_k)
    s = (((x_k - mu_k)**2) / sigma_k)
    (s>10).astype('int')
    
    return (s>10).astype('int')


def get_ecc_batch(x):
    '''
    :param x: a n-D numpy array that stors the timeseries. The first dimension is assumed to be time.
    :return: the eccentricity values in a matrix with similar dimension as the input
    '''
    #x -= x.mean()
    eccs = np.zeros_like(x)
    mu_k = x[0]
    sigma_k = np.ones_like(x[0])
    eccs[0] = 0
    for k, frame in enumerate(x[1:]):
        #print('initial muk is: ', mu_k)
        mu_k = get_mu_k(frame, mu_k, k + 2)
        #print('new muk is: ', mu_k)
        #print('initial sigmak is: ', sigma_k)
        sigma_k = get_sigma_k(frame, mu_k, sigma_k, k + 2)
        #print('new sigmak is: ', sigma_k)
        #print('initial ecck is: ', eccs[k])
        eccs[k + 1] = get_ecc(frame, mu_k, sigma_k, k + 2)
        #print('new ecck is: ', eccs[k + 1])
    return eccs

#%%






#%%

x = np.random.rand(10000, 128, 120)
x[:, 50:70, 50:70] = 0

#%%
eccs = get_ecc_batch(x)

#%%
import matplotlib.pyplot as plt
plt.imshow(x[9999])
plt.show()
plt.imshow(eccs[9999])
plt.show()




#%%
fname_new = '/media/nel/ssd/data/memmap__d1_512_d2_128_d3_1_order_C_frames_36000_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
df = np.array(images[:1000])
mv = cm.movie(df)
(np.mean(mv, axis=0)-mv).transpose([0,2,1]).play(fr=60, magnification=3)


df1 = movie.gaussian_blur_2D(df, 
                       kernel_size_x=np.int(2*np.ceil(2*1.5)+1), 
                       kernel_size_y=np.int(2*np.ceil(2*1.5)+1),
                       kernel_std_x=1.5, kernel_std_y=1.5, 
                       borderType=cv2.BORDER_REPLICATE)

df2 = get_ecc_batch(df)
mv = cm.movie(df2)
mv.transpose([0,2,1]).play(fr=10, magnification=3, gain=2)

#%% Create a random signal
x = np.random.random((1000))
sp = np.random.randint(2, high=8, size=50)
lc = np.random.permutation(1000)
y = np.zeros(1000)
for i in range(1000):
    if i in lc[:50]:
        y[i] = x[i] + 


            

            
        
        
