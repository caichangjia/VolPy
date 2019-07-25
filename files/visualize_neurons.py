#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:26:26 2019

@author: Changjia
"""

#%% Visualization of voltage imaging
from scipy.ndimage.measurements import center_of_mass
from caiman.base.rois import com
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10

colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

#%%
del1 = np.std(np.array(output['num_spikes'])[:,-4:], axis=1)
del2 = np.array(output['passedLocalityTest'])
del3 = np.array(output['low_spk'])
select = np.multiply(np.multiply((del1<20), (del2>0)), (del3<1))

A = ROIs[select]
C = np.array(output['trace'])[select]
#%%
N = select.sum()
n = N
n_group = np.int(np.ceil(N/n))
n = np.int(np.ceil(N/n_group))
li = np.random.permutation(N)


#%% Seperate components
l = []
for i in range(A.shape[0]):
    temp = A[i]
    l.append(center_of_mass(temp))
#l = np.square(np.array(l)[:,1]
l = np.array(l)[:,0]
np.sort(l)
li = np.argsort(l)
if N != n*n_group:
    li = np.append(li, np.ones((1,n*n_group-N),dtype=np.int8)*(-1))

mat = li.reshape((np.int(np.ceil(N/n)), np.int(np.ceil(N/n_group))), order='F')

#%%
fnames = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/datasetblock1.hdf5'
m=cm.load(fnames)

from skimage import measure
i = 0
j = 0      
number=0
number1=0
#Cn = np.mean(np.array(m), axis=0)     
Cn = img
A = A.astype(np.float64)   
for i in range(n_group):
    plt.figure()    
    vmax = np.percentile(Cn, 99)
    vmin = np.percentile(Cn, 5)
    plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin) #cmap=plt.cm.gray,
    plt.title('Neurons location')
    d1, d2 = np.shape(Cn)
    #d, nr = np.shape(A)
    cm1 = com(A.reshape((N,-1), order='F').transpose(), d1, d2)
    max_number = n
    colors='yellow'
    for j in range(np.int(np.ceil(N/n_group))):
        index = mat[i,j]
        print(index) 
        img = A[index]
        img1 = img.copy()
        #img1[img1<np.percentile(img1[img1>0],15)] = 0
        #img1 = connected_components(img1)
        img2 = np.multiply(img, img1)
        contours = measure.find_contours(img2, 0.5)[0]
        #img2=img.copy()
        img2[img2 == 0] = np.nan
        if index !=-1:
            plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(number,9)])
            plt.text(cm1[index, 1]+0, cm1[index, 0]-0, str(number), color=colors)
            #print(number)
            number=number+1 
    plt.savefig('/home/nel/Code/VolPy/403106-Neurons-corr{}-{}.pdf'.format(number1,number-1))
    number1=number
    
#%%
#cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
#n = np.int(np.ceil(N/n_group))
CZ = C[:,23000:33000]
CZ = (CZ-CZ.mean(axis=1)[:,np.newaxis])/CZ.std(axis=1)[:,np.newaxis]


number=0
number1=0 
for i in range(n_group):
    fig,ax = plt.subplots((mat[i,:]>-1).sum(),1)
    length = (mat[i,:]>-1).sum()
    for j in range(np.int(np.ceil(N/n_group))):
        if j==0:
            ax[j].set_title('Signals')
        #Y_r = cnm2.estimates.YrA + cnm2.estimates.C
        if mat[i,j]>-1:
            #Y_r = cnm2.estimates.F_dff[select,:]
            index = mat[i,j]
            T = C.shape[1]
            ax[j].plot(np.arange(10000), -CZ[index], 'c', linewidth=1, color=colorsets[np.mod(number,9)])
            #ax[j].plot(np.arange(T), cnm2.estimates.S[index, :][:], 'r', linewidth=2)
            ax[j].text(-30, 0, f'{number}', horizontalalignment='center',
                 verticalalignment='center')
            ax[j].set_ylim([(-CZ).min(),(-CZ).max()])
            if j==0:
                #ax[j].legend(labels=['Filtered raw data', 'Inferred trace'], frameon=False)
                ax[j].text(-30, 3000, 'neuron', horizontalalignment='center',
                 verticalalignment='center')
            if j<length-1:
                ax[j].axis('off')
            if j==length-1:
                ax[j].spines['right'].set_visible(False)
                ax[j].spines['top'].set_visible(False)  
                ax[j].spines['left'].set_visible(True) 
                ax[j].get_yaxis().set_visible(True)
                ax[j].set_xlabel('Frames')
            number = number + 1
    fig.savefig('/home/nel/Code/DendriticData/Output/Horst-85500-signal{}-{}.pdf'.format(number1,number-1))
    number1=number    
    
    
#%%
from scipy import signal
def highpassVideo(video, freq, sampleRate):
    """
    Function for passing signals with frequency higher than freq
    """
    normFreq = freq / (sampleRate / 2)
    b, a = signal.butter(3, normFreq, 'high')
    videoFilt = np.single(signal.filtfilt(b, a, video, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return videoFilt

#%%
dims=m.shape
dims

#%%
data=m[13000:33000,:,:].reshape((10000,-1), order='F')
datahp = highpassVideo(data.T, 1/3, 400).T    
datahp = datahp.reshape((20000,dims[1],dims[2]),order='F')
from caiman.summary_images import local_correlations
img_corr = local_correlations(datahp, swap_dim=False)
plt.figure();plt.imshow(img_corr)
plt.savefig('/home/nel/Code/VolPy/403106-corr-hp.pdf')

#%%
i = 0
j = 0      
number=0
number1=0
Cn = img_corr
A = A.astype(np.float64)   
for i in range(n_group):
    plt.figure()    
    vmax = np.percentile(Cn, 99)
    vmin = np.percentile(Cn, 5)
    plt.imshow(Cn, interpolation='None', cmap=plt.cm.gray, vmax=vmax, vmin=vmin) 
    plt.title('Neurons location')
    d1, d2 = np.shape(Cn)
    #d, nr = np.shape(A)
    cm1 = com(A.reshape((N,-1), order='F').transpose(), d1, d2)
    max_number = n
    colors='yellow'
    for j in range(np.int(np.ceil(N/n_group))):
        index = mat[i,j]
        print(index) 
        img = A[index]
        img1 = img.copy()
        #img1[img1<np.percentile(img1[img1>0],15)] = 0
        #img1 = connected_components(img1)
        img2 = np.multiply(img, img1)
        contours = measure.find_contours(img2, 0.5)[0]
        #img2=img.copy()
        img2[img2 == 0] = np.nan
        if index !=-1:
            plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(number,9)])
            plt.text(cm1[index, 1]+0, cm1[index, 0]-0, str(number), color=colors)
            #print(number)
            number=number+1 
    plt.savefig('/home/nel/Code/VolPy/403106-Neurons-corr{}-{}.pdf'.format(number1,number-1))
    number1=number
 
