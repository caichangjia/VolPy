#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:21:05 2019

@author: Changjia
"""
# Folder information of preparing files for UNet
import matplotlib.pyplot as plt
import caiman as cm
import numpy as np
import cv2
import scipy.io as io

#%%
# Johaness data
# Original data
    dr = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    ds_list = ['06152017Fish1-2', '10192017Fish2-1','10192017Fish3-1']
    ds = ds_list[2]
    fnames = dr + ds + '/registered.tif'
    m = cm.load(fnames,subindices=slice(2000,12000))
# Read
    dr = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    ds_list = ['06152017Fish1-2', '10192017Fish2-1','10192017Fish3-1']
    ds = ds_list[]
    #fnames = dr + ds + '/registered.tif'
    #m = cm.load(fnames,subindices=slice(0, 10000))
    #m.save(dr + ds + '/' + ds + '.hdf5')    
    fnames = dr + ds + '/' + ds + '.hdf5'
    m = cm.load(fnames)
    images = m
# Write
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Johannes/'
    sname = npz_dir+ds+'.npz'
    np.savez(sname, img_2c)
    
# Save to tif file for ImageJ
    sname = npz_dir+ds+'.npz'
    X = np.load(sname)['arr_0']
    iname = npz_dir+ds+'.tif'
    cm.movie(X).transpose([2,0,1]).save(iname)
    
# Plot two figures
    plt.figure();plt.imshow(X[:,:,0])
    plt.figure();plt.imshow(X[:,:,1])

# Johannes's ROI
    roi_dir = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    rname = roi_dir+ds+'/ROI_info.npy'
    y = np.load(rname, allow_pickle=True)
    dims = X.shape
    img = np.zeros((dims[0], dims[1]))
    for i in range(len(y)):
        img[y[i]['pixel_yx_list'][:,0],y[i]['pixel_yx_list'][:,1]] = 1
    plt.figure();plt.imshow(img)
    
# Read my ROI
    dims = (508, 288)#(360, 256)#(364,320) 
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Johannes'
    rname = roi_dir+ds+'_RoiSet.zip'
    from caiman.base.rois import nf_read_roi_zip
    img = nf_read_roi_zip(rname,dims)
    plt.figure();plt.imshow(img.sum(axis=0))
    
# Form training data
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Johannes/'
    X_j = []
    dimension = []
    for index, file in enumerate(sorted(os.listdir(npz_dir))):
        if file[-3:] == 'npz':
            temp = np.load(npz_dir+ file)['arr_0']
            dimension.append([temp.shape[0], temp.shape[1]])
            temp = Mirror(temp)
            X_j.append(temp)
    X_j = np.array(X_j)

    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Johannes/'    
    Y_j = []
    for index, file in enumerate(sorted(os.listdir(roi_dir))):
        print(file)
        from caiman.base.rois import nf_read_roi_zip
        name = roi_dir + file
        img = nf_read_roi_zip(name,dims=dimension[index]).sum(axis=0)
        print(img.shape)
        img = Mirror(img)
        Y_j.append(img)
        
    Y_j = np.array(Y_j)
    Y_j[Y_j>0] = 1
    Y_j = Y_j[:,:,:,np.newaxis]            
    
    


    
#%%
###############################################################################
# Adam's data
# Read hdf5/mmap file after motion correction
    import os
    dr = '/home/nel/Code/Voltage_imaging/exampledata/toshare_CA1/Data'
    #ds_list = [i[:-5] for i in os.listdir(dr) if '.hdf5' in i]
    #ds_list = ['IVQ32_S2_FOV1', 'IVQ38_S2_FOV3', 'IVQ48_S7_FOV7','IVQ38_S1_FOV5',
    #           'IVQ48_S7_FOV5', 'IVQ48_S7_FOV8', 'IVQ29_S5_FOV4','IVQ29_S5_FOV6']
    ds_list = [i[:-5] for i in os.listdir(dr) if 'order_F' in i]
    ds = ds_list[8]
    fnames = dr +  '/' + ds + '.mmap'
    m = cm.load(fnames)[2000:]
    
# Write
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Adam/'
    sname = npz_dir+ds+'.npz'
    np.savez(sname, img_2c)
    
# Save to tif file for ImageJ
    sname = npz_dir+ds+'.npz'
    X = np.load(sname)['arr_0']
    iname = npz_dir+ds+'.tif'
    cm.movie(X).transpose([2,0,1]).save(iname)

# Read my ROIs
    dims = (280,96)#(164,96)#(128,88)#(284,96)#(212,96)#(176,92)#(256, 96) 
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Adam/'
    rname = roi_dir+ds[:13]+'_RoiSet.zip'
    from caiman.base.rois import nf_read_roi_zip
    img = nf_read_roi_zip(rname,dims)
    plt.figure();plt.imshow(img.sum(axis=0))
    
#%%
###############################################################################
# Kaspar's Data
    import numpy as np
    import matplotlib.pyplot as plt
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Kaspar/'
    X = []
    import os
    for index, file in enumerate(sorted(os.listdir(npz_dir))):
        temp = np.load(npz_dir + file)['arr_0']
        X.append(Mirror(temp))

#    for index, file in enumerate(sorted(os.listdir(npz_dir))):
#        cm.movie(X[index]).save(npz_dir+'/'+file[:-4]+'.tif')
    X = np.array(X)
    
    fig,ax = plt.subplots(1,X.shape[0])
    for i in range(X.shape[0]):
        ax[i].imshow(X[i,:,:,0])
    
# ROIs
    import scipy.io as io
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Kaspar/'
    Y = []
    for index, file in enumerate(sorted(os.listdir(roi_dir))):
        print(file)
        from caiman.base.rois import nf_read_roi_zip
        name = roi_dir + file
        img = nf_read_roi_zip(name,dims=(512,128)).sum(axis=0)
        Y.append(Mirror(img))
        
    Y = np.array(Y)
    Y[Y>0] = 1
    Y = Y[:,:,:,np.newaxis]

    fig,ax = plt.subplots(1,len(Y))
    for i in range(Y.shape[0]):
        ax[i].imshow(Y[i,:,:,0])
        
    X = X[np.array([0,1,2,3,5,6,4,7]),:,:,:]    
    Y = Y[np.array([0,2,1,4,6,5,3,7]),:,:,:]  
        
    Xtrain = np.concatenate([X[np.array([0,1,2,3,4,5]),:,:,:],X_j[np.array([0,1])]])
    Ytrain = np.concatenate([Y[np.array([0,1,2,3,4,5]),:,:,:], Y_j[np.array([0,1])]])
    Xtest = np.concatenate([X[[6,7],:,:,:], X_j[np.array([2])]])
    Ytest = np.concatenate([Y[[6,7],:,:,:], Y_j[np.array([2])]])
    
    
    
#%%
    def Mirror(temp,size=512):
        shape = temp.shape
        temp = cv2.copyMakeBorder(temp, size-shape[0], 0, 0, size-shape[1], cv2.BORDER_REFLECT)
        return temp
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



