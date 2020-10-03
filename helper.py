import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['axes.grid'] = False
#%%
def get_slice(data_3d, view_index, slice_index):
    if view_index == 0:
        return copy.deepcopy(data_3d[slice_index,:,:])
    if view_index == 1:
        return copy.deepcopy(data_3d[:,slice_index,:])
    if view_index == 2:
        return copy.deepcopy(data_3d[:,:,slice_index])

def show(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    
def get_X(data, view_index):
    """
    get coordinates on 2D plane
    """
    if len(data.shape) == 2:
        ny, nx = (data.shape[0], data.shape[1])
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(x, y)
        X = np.vstack((xv.flatten(), yv.flatten())).T
        return X        
    if view_index == 0:
        ny, nx = (data.shape[1], data.shape[2])
    if view_index == 1:
        ny, nx = (data.shape[0], data.shape[2])
    if view_index == 2:
        ny, nx = (data.shape[0], data.shape[1])
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    X = np.vstack((xv.flatten(), yv.flatten())).T
    return X

def plot_cut2(data):
    line_dense = sns.distplot(data.flatten(), 
                              hist=False,
                              kde_kws={"color": 'k',"alpha":0.8}).get_lines()[-1].get_data()    
    density_percentage = 0.01
    density_cut = density_percentage*max(line_dense[1])
    inten_cut = max(line_dense[0][line_dense[1]>density_cut])
    return inten_cut

def get_data_normed(data, inten_cut):
    print('>>> min/max before norm: ',data.min(),data.max())
    print('>>> intensity cut:', inten_cut)
    data[data>inten_cut] = inten_cut
    ## data = (data-data.mean()) / data.std() # do we need this?
    print('>>> min/max after norm: ',data.min(),data.max())
    data = (data-data.min()) / (data.max()-data.min())
    return data
    
def norm_slice(image):
    return (image-image.min()) / (image.max()-image.min())
#%%
import copy
def get_score_map(view_index, slice_index, class_index):
    if view_index == 0:
        if class_index == 0:
            return copy.deepcopy(score_map0_0[slice_index,:,:])
        if class_index == 1:
            return copy.deepcopy(score_map0_1[slice_index,:,:])
        if class_index == 2:
            return copy.deepcopy(score_map0_2[slice_index,:,:])
    if view_index == 1:
        if class_index == 0:
            return copy.deepcopy(score_map1_0[:,slice_index,:])
        if class_index == 1:
            return copy.deepcopy(score_map1_1[:,slice_index,:])
        if class_index == 2:
            return copy.deepcopy(score_map1_2[:,slice_index,:])
    if view_index == 2:
        if class_index == 0:
            return copy.deepcopy(score_map2_0[:,:,slice_index])
        if class_index == 1:
            return copy.deepcopy(score_map2_1[:,:,slice_index])
        if class_index == 2:
            return copy.deepcopy(score_map2_2[:,:,slice_index])

def get_score_map_add(view_index, slice_index, class_index):
    """
    add mode
    """
    if view_index == 0:
        if class_index == 0:
            return copy.deepcopy(score_map_add_0[slice_index,:,:])
        if class_index == 1:
            return copy.deepcopy(score_map_add_1[slice_index,:,:])
        if class_index == 2:
            return copy.deepcopy(score_map_add_2[slice_index,:,:])
    if view_index == 1:
        if class_index == 0:
            return copy.deepcopy(score_map_add_0[:,slice_index,:])
        if class_index == 1:
            return copy.deepcopy(score_map_add_1[:,slice_index,:])
        if class_index == 2:
            return copy.deepcopy(score_map_add_2[:,slice_index,:])
    if view_index == 2:
        if class_index == 0:
            return copy.deepcopy(score_map_add_0[:,:,slice_index])
        if class_index == 1:
            return copy.deepcopy(score_map_add_1[:,:,slice_index])
        if class_index == 2:
            return copy.deepcopy(score_map_add_2[:,:,slice_index])
                
#%%
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
def show_raw_img(intensity, pat_index,view_index,slice_index,corr=None,show=True):
    
    # show raw image with patch box
    
    img = intensity

    if show:
        fig,ax = plt.subplots(1)
        ax.imshow(img)
    
        if corr is not None:
            rect = patches.Rectangle((corr[0],corr[1]),corr[-2],corr[-1],
                             linewidth=1,edgecolor='r',facecolor='none')
        
            ax.add_patch(rect)
        plt.show()  
#%%
def show_subplots(sub_list):
    """
    show several subplots 
    """
    L = len(sub_list)
    fig = plt.figure(figsize=(4*L, 4))
    for i in range(L):
        plt.subplot(1,L,i+1)
        plt.imshow(sub_list[i])
        plt.colorbar()
    
#%%
def Sfun(array):
    y = (1 / (1 + np.exp(-array*6)) - 0.5) * 2
    return y

def Sfun_backup(array):
    """
    this might be right
    """
    return (1 / (1 + np.exp(-array*6)) - 0.5) * 2

def show_Sfun():
    x = np.linspace(0, 1, num=50)
    plt.figure()
    plt.plot(x, Sfun(x))
    plt.grid()
    plt.title('S function')
    plt.xlabel('x')
    plt.ylabel('y')
#%%
inten_cut = {
        0: 2121.4978945974694,
        1: 1577.0722181546216,
        2: 2179.357768170086,
        3: 1144.9356572412464,
        4: 1642.8813878339804,
        5: 1978.4297697927975,
        6: 1954.6497703789728,
        7: 1995.6486857980428,
        8: 1409.5082685065245,
        9: 2108.6989414349923,
        
        10: 975.421704067435,
        11: 1890.8242649674153,
        12: 1333.9869781783125,
        13: 1502.899109327496,
        14: 1537.8815429635738,
        15: 1454.23099043816,
        16: 1240.4368168986562,
        17: 361.41670132471705,
        18: 478.9789280065819,
        19: 531.4438227653775
        }
#%%
def show_3view3label(view_index, slice_index):
    fig = plt.figure(figsize=(13, 9))
    
    # view 0: ---------- ---------- ----------
    plt.subplot(4,3,1)
    if view_index == 2:
        plt.imshow(score_map0_0[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map0_0[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map0_0[slice_index,:,:])
    plt.title('view 0: blood')
    
    plt.subplot(4,3,2)
    if view_index == 2:
        plt.imshow(score_map0_1[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map0_1[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map0_1[slice_index,:,:])
    plt.title('view 0: muscle')
    
    plt.subplot(4,3,3)
    if view_index == 2:
        plt.imshow(score_map0_2[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map0_2[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map0_2[slice_index,:,:])
    plt.title('view 0: background')    
    
    # view 1: ---------- ---------- ----------
    plt.subplot(4,3,4)
    if view_index == 2:
        plt.imshow(score_map1_0[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map1_0[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map1_0[slice_index,:,:])
    plt.title('view 1: blood')
    
    plt.subplot(4,3,5)
    if view_index == 2:
        plt.imshow(score_map1_1[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map1_1[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map1_1[slice_index,:,:])
    plt.title('view 1: muscle')
    
    plt.subplot(4,3,6)
    if view_index == 2:
        plt.imshow(score_map1_2[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map1_2[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map1_2[slice_index,:,:])
    plt.title('view 1: background')          
    
    # view 2: ---------- ---------- ----------
    plt.subplot(4,3,7)
    if view_index == 2:
        plt.imshow(score_map2_0[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map2_0[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map2_0[slice_index,:,:])
    plt.title('view 2: blood')
    
    plt.subplot(4,3,8)
    if view_index == 2:
        plt.imshow(score_map2_1[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map2_1[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map2_1[slice_index,:,:])
    plt.title('view 2: muscle')
    
    plt.subplot(4,3,9)
    if view_index == 2:
        plt.imshow(score_map2_2[:,:,slice_index])
    if view_index == 1:
        plt.imshow(score_map2_2[:,slice_index,:])
    if view_index == 0:
        plt.imshow(score_map2_2[slice_index,:,:])
    plt.title('view 2: background')       
    
    # view add: ---------- ---------- ----------
    if view_index == 2:
        score_map_add_0 = score_map0_0[:,:,slice_index] + \
                          score_map1_0[:,:,slice_index] + \
                          score_map2_0[:,:,slice_index]
        score_map_add_1 = score_map0_1[:,:,slice_index] + \
                          score_map1_1[:,:,slice_index] + \
                          score_map2_1[:,:,slice_index]
        score_map_add_2 = score_map0_2[:,:,slice_index] + \
                          score_map1_2[:,:,slice_index] + \
                          score_map2_2[:,:,slice_index]
    if view_index == 1:
        score_map_add_0 = score_map0_0[:,slice_index,:] + \
                          score_map1_0[:,slice_index,:] + \
                          score_map2_0[:,slice_index,:]  
        score_map_add_1 = score_map0_1[:,slice_index,:] + \
                          score_map1_1[:,slice_index,:] + \
                          score_map2_1[:,slice_index,:]  
        score_map_add_2 = score_map0_2[:,slice_index,:] + \
                          score_map1_2[:,slice_index,:] + \
                          score_map2_2[:,slice_index,:]    
    if view_index == 0:
        score_map_add_0 = score_map0_0[slice_index,:,:] + \
                          score_map1_0[slice_index,:,:] + \
                          score_map2_0[slice_index,:,:]
        score_map_add_1 = score_map0_1[slice_index,:,:] + \
                          score_map1_1[slice_index,:,:] + \
                          score_map2_1[slice_index,:,:]
        score_map_add_2 = score_map0_2[slice_index,:,:] + \
                          score_map1_2[slice_index,:,:] + \
                          score_map2_2[slice_index,:,:]
                
    plt.subplot(4,3,10)
    plt.imshow(score_map_add_0)
    plt.title('view add: blood')
    
    plt.subplot(4,3,11)
    plt.imshow(score_map_add_1)
    plt.title('view add: muscle')
    
    plt.subplot(4,3,12)
    plt.imshow(score_map_add_2)
    plt.title('view add: background')           
    
    plt.suptitle('this view: '+str(view_index)+', slice: '+str(slice_index),fontsize=11, fontweight='bold')
#%%
def generate_predict(view_index, slice_index, threshold=0, blood_threshold=0, muscle_threshold=0):
    '''
    Note: we only consider this view
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1],3))
        score_slice[:,:,0] = score_map2_0[:,:,slice_index]-blood_threshold
        score_slice[:,:,1] = score_map2_1[:,:,slice_index]+muscle_threshold
        score_slice[:,:,2] = score_map2_2[:,:,slice_index]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2],3))
        score_slice[:,:,0] = score_map1_0[:,slice_index,:]-blood_threshold
        score_slice[:,:,1] = score_map1_1[:,slice_index,:]+muscle_threshold
        score_slice[:,:,2] = score_map1_2[:,slice_index,:]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2],3))
        score_slice[:,:,0] = score_map0_0[slice_index,:,:]-blood_threshold
        score_slice[:,:,1] = score_map0_1[slice_index,:,:]+muscle_threshold
        score_slice[:,:,2] = score_map0_2[slice_index,:,:]-threshold
        predict = np.argmax(score_slice, axis=2)
        
    return predict
#%%
def generate_predict2(view_index, slice_index, threshold=0, blood_threshold=0, muscle_threshold=0):
    '''
    Note: we consider the added view
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1],3))
        score_slice[:,:,0] = score_map_add_0[:,:,slice_index]-blood_threshold
        score_slice[:,:,1] = score_map_add_1[:,:,slice_index]+muscle_threshold
        score_slice[:,:,2] = score_map_add_2[:,:,slice_index]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2],3))
        score_slice[:,:,0] = score_map_add_0[:,slice_index,:]-blood_threshold
        score_slice[:,:,1] = score_map_add_1[:,slice_index,:]+muscle_threshold
        score_slice[:,:,2] = score_map_add_2[:,slice_index,:]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2],3))
        score_slice[:,:,0] = score_map_add_0[slice_index,:,:]-blood_threshold
        score_slice[:,:,1] = score_map_add_1[slice_index,:,:]+muscle_threshold
        score_slice[:,:,2] = score_map_add_2[slice_index,:,:]-threshold
        predict = np.argmax(score_slice, axis=2)
    
    return predict
#%%
import copy
def show_mul_prob(prob_list, class_name="", figure=False, factor=1):
    """
    here we use this function by different orders
    that is, we always fix the first element in the list
    """
    if figure:
        fig = plt.figure(figsize=(12, 4))
    
    prob_c0_norm = norm_slice(prob_list[0])
    prob_c1_norm = norm_slice(prob_list[1])
    prob_c2_norm = norm_slice(prob_list[2])

    prob_0 = (1-prob_c0_norm)**factor
    prob_1 = (1-prob_c1_norm)**factor
    prob_2 = (1-prob_c2_norm)**factor
    
    if figure:    
        plt.subplot(1,4,1)
        plt.imshow(prob_list[0])
        
        plt.subplot(1,4,2)
        plt.imshow(prob_list[0]*prob_1)
    
        plt.subplot(1,4,3)
        plt.imshow(prob_list[0]*prob_2)
    
        plt.subplot(1,4,4)
        plt.imshow(prob_list[0]*prob_1*prob_2)
        
        plt.suptitle(class_name)
    '''
    # default
    return copy.deepcopy(prob_list[0]*prob_1*prob_2), \
           copy.deepcopy(prob_list[1]*prob_0*prob_2), \
           copy.deepcopy(prob_list[2]*prob_1*prob_0)    
    '''
    return copy.deepcopy(prob_list[0]*prob_1*prob_2), \
           copy.deepcopy(prob_list[1]*prob_0*prob_2), \
           copy.deepcopy(prob_list[2]*prob_1*prob_0)

def generate_predict_mul(view_index, 
                         slice_index, 
                         threshold=0, 
                         blood_threshold=0, 
                         muscle_threshold=0,
                         factor=1):
    '''
    Note: in this function, we use adjusted score maps
    Note: we only consider this view
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1],3))
        score_slice2 = np.zeros((data.shape[0],data.shape[1],3))
        
        score_slice[:,:,0] = score_map2_0[:,:,slice_index]-blood_threshold
        score_slice[:,:,1] = score_map2_1[:,:,slice_index]+muscle_threshold
        score_slice[:,:,2] = score_map2_2[:,:,slice_index]-threshold
        
        score_slice2[:,:,0],score_slice2[:,:,1],score_slice2[:,:,2]=show_mul_prob([score_slice[:,:,0], \
                                                                                   score_slice[:,:,1], \
                                                                                   score_slice[:,:,2]],
                                                                                   factor=factor)
        predict = np.argmax(score_slice2, axis=2)

    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2],3))
        score_slice2 = np.zeros((data.shape[0],data.shape[2],3))
        
        score_slice[:,:,0] = score_map1_0[:,slice_index,:]-blood_threshold
        score_slice[:,:,1] = score_map1_1[:,slice_index,:]+muscle_threshold
        score_slice[:,:,2] = score_map1_2[:,slice_index,:]-threshold
        
        score_slice2[:,:,0],score_slice2[:,:,1],score_slice2[:,:,2]=show_mul_prob([score_slice[:,:,0], \
                                                                                   score_slice[:,:,1], \
                                                                                   score_slice[:,:,2]],
                                                                                   factor=factor)        
        predict = np.argmax(score_slice2, axis=2)

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2],3))
        score_slice2 = np.zeros((data.shape[1],data.shape[2],3))
        
        score_slice[:,:,0] = score_map0_0[slice_index,:,:]-blood_threshold
        score_slice[:,:,1] = score_map0_1[slice_index,:,:]+muscle_threshold
        score_slice[:,:,2] = score_map0_2[slice_index,:,:]-threshold
        
        score_slice2[:,:,0],score_slice2[:,:,1],score_slice2[:,:,2]=show_mul_prob([score_slice[:,:,0], \
                                                                                   score_slice[:,:,1], \
                                                                                   score_slice[:,:,2]],
                                                                                   factor=factor)        
        predict = np.argmax(score_slice2, axis=2)
        
    return predict
#%%
def show_argmax(view_index, slice_index, thresholds, blood_threshold=0,muscle_threshold=0):
    '''
    eg:
    thresholds = [0,4,8]
    '''
    fig = plt.figure(figsize=(15, 4))
    
    for i in range(len(thresholds)):
        plt.subplot(1,len(thresholds),i+1)
        predict = generate_predict(view_index,slice_index,thresholds[i],blood_threshold,muscle_threshold)
        plt.imshow(predict)
        plt.title('threshold: '+str(thresholds[i]))
    plt.suptitle('show_argmax: view: '+str(view_index)+', slice:'+str(slice_index))

def show_argmax2(view_index, slice_index, thresholds, blood_threshold=0,muscle_threshold=0):
    '''
    eg:
    thresholds = [0,4,8]
    '''
    fig = plt.figure(figsize=(15, 4))
    
    for i in range(len(thresholds)):
        plt.subplot(1,len(thresholds),i+1)
        predict = generate_predict2(view_index,slice_index,thresholds[i],blood_threshold,muscle_threshold)
        plt.imshow(predict)
        plt.title('threshold: '+str(thresholds[i]))
    plt.suptitle('show_argmax2: view: '+str(view_index)+', slice:'+str(slice_index))
#%%
def crop_edge_pixel(predict):
    """
    minor refine
    """
    pix_num = 2
    predict[:,-pix_num:] = 2
    predict[-pix_num:,:] = 2
    
    return predict

#%%
def dist(array, label=""):
    """
    show dist plot
    """    
    if len(array.shape) != 1:
        flatten = array.flatten()
    else:
        flatten = array
    plt.figure()
    sns.distplot(flatten, hist=False, color="r", label=label)
#%%    
def dist_multi(array):
    """
    show dist plot
    """ 
    plt.figure()       
    num = len(array)
    for i in range(num):
        if len(array[i].shape) != 1:
            flatten = array[i].flatten()
        else:
            flatten = array[i]        
        sns.distplot(flatten, hist=False, label=str(i)) # , color="r"
#%%
def generate_prediction_from_slices(slice_0, slice_1, slice_2):
    """
    here we use three slices to generate label
    """
    score_slice = np.zeros((slice_0.shape[0], slice_0.shape[1],3))
    score_slice[:,:,0] = slice_0
    score_slice[:,:,1] = slice_1
    score_slice[:,:,2] = slice_2
    predict = np.argmax(score_slice, axis=2)
    return predict 

def generate_prediction_from_slices_dev(slice_0, slice_1, slice_2,show=False,factor=1):
    """
    here we use three slices to generate label
    """
    if show:
        plt.figure(figsize=(15, 8))
        plt.subplot(2,3,1)
        plt.imshow(slice_0)
        plt.subplot(2,3,2)
        plt.imshow(slice_1)
        plt.subplot(2,3,3)
        plt.imshow(slice_2)        
    
    slice_0_ = copy.deepcopy(slice_0)
    slice_1_ = copy.deepcopy(slice_1)
    slice_2_ = copy.deepcopy(slice_2)
    
    slice_0_ = norm_slice(slice_0_)
    slice_1_ = norm_slice(slice_1_)
    slice_2_ = norm_slice(slice_2_)
    
    slice_0_ = (1-slice_0_)**factor
    slice_1_ = (1-slice_1_)**factor
    slice_2_ = (1-slice_2_)**factor
    
    score_slice = np.zeros((slice_0.shape[0], slice_0.shape[1],3))
    score_slice[:,:,0] = slice_0*slice_1_*slice_2_
    score_slice[:,:,1] = slice_1*slice_0_*slice_2_
    score_slice[:,:,2] = slice_2*slice_0_*slice_1_
    predict = np.argmax(score_slice, axis=2)

    if show:
        plt.subplot(2,3,4)
        plt.imshow(score_slice[:,:,0])
        plt.subplot(2,3,5)
        plt.imshow(score_slice[:,:,1])
        plt.subplot(2,3,6)
        plt.imshow(score_slice[:,:,2])
    
    return predict 
#%%
def generate_prediction_from_slices_dev_dev(slice_0, 
                                            slice_1, 
                                            slice_2,
                                            show=False,
                                            factor=1,
                                            threshold=0,
                                            muscle_threshold=0,
                                            blood_threshold=0):
    """
    here we use three slices to generate label
    """
    if show:
        plt.figure(figsize=(15, 8))
        plt.subplot(2,3,1)
        plt.imshow(slice_0)
        plt.subplot(2,3,2)
        plt.imshow(slice_1)
        plt.subplot(2,3,3)
        plt.imshow(slice_2)        
    
    slice_0_ = copy.deepcopy(slice_0)
    slice_1_ = copy.deepcopy(slice_1)
    slice_2_ = copy.deepcopy(slice_2)
    
    slice_0_ = norm_slice(slice_0_)
    slice_1_ = norm_slice(slice_1_)
    slice_2_ = norm_slice(slice_2_)
    '''
    min_value = np.min((score_map_0,score_map_1,score_map_2))
    slice_add_ = slice_0_ + slice_1_ + slice_2_
    slice_0_ = (slice_0_-min_value)/slice_add_
    slice_1_ = (slice_1_-min_value)/slice_add_
    slice_2_ = (slice_2_-min_value)/slice_add_
    '''
    slice_0__ = (1-slice_0_)**factor
    slice_1__ = (1-slice_1_)**factor
    slice_2__ = (1-slice_2_)**factor
    
    score_slice = np.zeros((slice_0.shape[0], slice_0.shape[1],3))
    score_slice[:,:,0] = slice_1__*slice_2__*(slice_0-blood_threshold)
    score_slice[:,:,1] = slice_0__*slice_2__*(slice_1+muscle_threshold)
    score_slice[:,:,2] = slice_0__*slice_1__*(slice_2-threshold)
    predict = np.argmax(score_slice, axis=2)

    if show:
        plt.subplot(2,3,4)
        plt.imshow(score_slice[:,:,0])
        plt.subplot(2,3,5)
        plt.imshow(score_slice[:,:,1])
        plt.subplot(2,3,6)
        plt.imshow(score_slice[:,:,2])
    
    return predict, score_slice
#%%
from matplotlib import cm   
import imageio

def gen_image(image,path,inten_cut=0):
    """
    generate one image
    """
    a = copy.deepcopy(image)
    if inten_cut != 0:
        a[a>inten_cut] = inten_cut
    a = (a-a.mean()) / a.std()
    a = (a-a.min()) / (a.max()-a.min())
    c = cm.viridis(a,bytes=True)[:,:,0:3]
    imageio.imwrite(path, c)
    
def image_norm(image,inten_cut):
    a = copy.deepcopy(image)
    if inten_cut != 0:
        a[a>inten_cut] = inten_cut
    a = (a-a.mean()) / a.std()
    a = (a-a.min()) / (a.max()-a.min())
    return a    
#%%
def gaussian2(x, y, mu=0, sig=1):
    """
    gaussian kernel
    """
    return (1 / (sig * sig * 2 * np.pi)) * np.exp(-(x*x + y*y) / (2 * np.power(sig, 2.)))

def get_kernel(length):
    """
    get gaussian kernel
    """
    kernel = np.zeros((length,length))
    for j in range(length):
        kernel[j,:] = [gaussian2(i-(length-1)/2,j-(length-1)/2) for i in range(length)]
    return kernel

#%%      
def get_point(prob_c0,prob_c1,prob_c2,center):
    """
    given an image patch and the center point, get the point of interest
    """
    point_cors = np.vstack((prob_c0.flatten(),
                            prob_c1.flatten(),
                            prob_c2.flatten())).T
    
    num_points = 2*center + 1                            
    center_cors = np.array([[prob_c0.flatten()[center],
                             prob_c1.flatten()[center],
                             prob_c2.flatten()[center]]]*num_points)
    
    Edist = np.linalg.norm(point_cors-center_cors,axis=1)    
    rank = np.argsort(Edist) # eg. [24, 23, 28, 25, 32, 39, 45, ...]
    
    num_cut = -1 # we use 8 points
    return rank[:num_cut]
    
def Conv(intensity, kernel, score_map_0, score_map_1, score_map_2):
    """
    perform simple convolution
    """
    select_point = False
    conved = np.zeros(intensity.shape)
    center = int(kernel.size / 2) # eg.: 7*7 -> 49 -> 24
    
    L = int((kernel.shape[0] - 1) / 2)
    length = kernel.shape[0]
        
    intensity_pad = np.pad(intensity, ((L, L), (L, L)), 'constant', constant_values=((0, 0), (0, 0)))
    score_pad_value = -5
    score_map_0_pad = np.pad(score_map_0, ((L, L), (L, L)), 'constant', constant_values=((score_pad_value, score_pad_value), (score_pad_value, score_pad_value)))
    score_map_1_pad = np.pad(score_map_1, ((L, L), (L, L)), 'constant', constant_values=((score_pad_value, score_pad_value), (score_pad_value, score_pad_value)))
    score_map_2_pad = np.pad(score_map_2, ((L, L), (L, L)), 'constant', constant_values=((score_pad_value, score_pad_value), (score_pad_value, score_pad_value)))
    
    for i in range(conved.shape[0]):
        for j in range(conved.shape[1]):
            intensity_crop = intensity_pad[i:i+length,j:j+length]
            prob_c0 = score_map_0_pad[i:i+length,j:j+length]
            prob_c1 = score_map_1_pad[i:i+length,j:j+length]
            prob_c2 = score_map_2_pad[i:i+length,j:j+length]
            
            raw_conv = intensity_crop * kernel
            if select_point:
                points_list = get_point(prob_c0,prob_c1,prob_c2,center)
                
                conv = raw_conv.flatten()[points_list]
                conv = conv.mean()
                conved[i,j] = conv
            else:
                conved[i,j] = raw_conv.flatten().mean()
    return conved    

#%%
from mpl_toolkits.mplot3d import Axes3D
    
def plot3d(arr1, arr2, arr3):
    """
    3D scatter plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = arr1.flatten()
    y = arr2.flatten()
    z = arr3.flatten()
    
    ax.scatter(x, y, z, c='r', marker='o')
    
    center = int(len(x) / 2)
    print('center point:', center, x[center], y[center], z[center])
    ax.scatter(x[center], y[center], z[center], c='b', marker='o')
    
    ax.set_xlabel('arr1')
    ax.set_ylabel('arr2')
    ax.set_zlabel('arr3')
    
    plt.show()
    
#%%
# this is for label propagation 
from sklearn.semi_supervised import label_propagation

#%%
from sklearn import mixture    
def generate_predict_blood_gaussian(view_index, slice_index, threshold=0, blood_threshold=0):
    '''
    Note: we consider the added view
    And: only see the blood, with gaussian mixture
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1]))
        score_slice[:,:] = score_map_add_0[:,:,slice_index] # 0
        
    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2]))
        score_slice[:,:] = score_map_add_0[:,slice_index,:] # 0

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2]))
        score_slice[:,:] = score_map_add_0[slice_index,:,:] # 0

    # gaussian mixture
    flattened = score_slice.flatten()
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    # threshold to intensity
    smaller_mean = clf.means_.min()
    threshold_inten = smaller_mean*2 + 0
    print(slice_index, threshold_inten)
    predict = np.zeros(score_slice.shape)
    predict[:,:] = 2 # all background
    
    predict[score_slice>threshold_inten]=0 # 0
    
    return predict
#%%
def generate_predict_blood_gaussian_dev(view_index, slice_index, threshold=0, blood_threshold=0):
    '''
    Note: we consider the added view
    And: only see the blood, with gaussian mixture
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1]))
        score_slice[:,:] = score_map2_1[:,:,slice_index] # 0
        
    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2]))
        score_slice[:,:] = score_map1_1[:,slice_index,:] # 0

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2]))
        score_slice[:,:] = score_map0_1[slice_index,:,:] # 0

    # gaussian mixture
    flattened = score_slice.flatten()
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    # threshold to intensity
    smaller_mean = clf.means_.min()
    threshold_inten = smaller_mean*2 - 3
    print(slice_index, threshold_inten)
    predict = np.zeros(score_slice.shape)
    predict[:,:] = 2 # all background
    
    predict[score_slice>threshold_inten]=1 # 0
    
    # ----- ----- ----- ----- -----
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1]))
        score_slice[:,:] = score_map2_0[:,:,slice_index] # 0
        
    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2]))
        score_slice[:,:] = score_map1_0[:,slice_index,:] # 0

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2]))
        score_slice[:,:] = score_map0_0[slice_index,:,:] # 0
    # gaussian mixture
    flattened = score_slice.flatten()
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    # threshold to intensity
    threshold_inten = smaller_mean*2 - 1.5
    print(slice_index, threshold_inten)
    
    predict[score_slice>threshold_inten]=0 # 0
    
    return predict
    
#%%
def change_label(label,from_indicator,to_indicator):
    """
    label: 3d label
    indicator: [1,2,0] etc.
    """
    label_ = np.zeros(label.shape)
    for i in range(3):
        label_[label==from_indicator[i]] = to_indicator[i]
    return label_
    
def generate_1_predicted_labels(mode='single',
                                view_index=0,
                                background_threshold=0,
                                muscle_threshold=0,
                                blood_threshold=0,
                                factor=0):
    label = np.zeros(data.shape, dtype=np.int16)

    for i in range(data.shape[view_index]):
        #% add mode
        slice_index = i
        if mode == 'single':
            prob_c0 = get_score_map(view_index, slice_index, 0)#[y0:y0+y_l,x0:x0+x_l]
            prob_c1 = get_score_map(view_index, slice_index, 1)#[y0:y0+y_l,x0:x0+x_l]
            prob_c2 = get_score_map(view_index, slice_index, 2)#[y0:y0+y_l,x0:x0+x_l]        
        elif mode == 'add':
            prob_c0 = get_score_map_add(view_index, slice_index, 0)#[y0:y0+y_l,x0:x0+x_l]
            prob_c1 = get_score_map_add(view_index, slice_index, 1)#[y0:y0+y_l,x0:x0+x_l]
            prob_c2 = get_score_map_add(view_index, slice_index, 2)#[y0:y0+y_l,x0:x0+x_l]   
        else:
            print('Error mode')
            return 0              
        #%
        pred, _ = generate_prediction_from_slices_dev_dev(prob_c0, 
                                                          prob_c1, 
                                                          prob_c2,
                                                          show=False,
                                                          factor=factor,
                                                          threshold=background_threshold,
                                                          muscle_threshold=muscle_threshold,
                                                          blood_threshold=blood_threshold)
        pred = crop_edge_pixel(pred)
        
        if view_index == 2:
            label[:,:,i] = pred # 2 - pred
        if view_index == 1:
            label[:,i,:] = pred # 2 - pred
        if view_index == 0:
            label[i,:,:] = pred # 2 - pred
            
    return label
    
#%%
def generate_3_predicted_labels(label_all,mode='single', # single view or add view
                                background_threshold=0,
                                muscle_threshold=0,
                                blood_threshold=0,
                                factor=0):
    """
    append 3 predicted labels to current label list
    we treat 3 view equally
    """
    for i in range(3):
        label = generate_1_predicted_labels(mode=mode,
                                            view_index=i,
                                            background_threshold=background_threshold,
                                            muscle_threshold=muscle_threshold,
                                            blood_threshold=blood_threshold,
                                            factor=factor)
        # label = change_label(label,[2,1,0],[2,0,1])
        label_all.append(copy.deepcopy(label))
    print('current ensemble number is:', len(label_all))
    
#%%
import cv2
def label_smoothing(label,view_index=0):
    """
    label is a 3D ndarray
    """
    k_size = 5
    
    label_ = np.zeros(label.shape)
    label__ = label.astype(np.uint8)
    if view_index == 0:
        for i in range(label.shape[0]):
            label_[i,:,:] = cv2.medianBlur(label__[i,:,:], k_size)
    if view_index == 1:
        for i in range(label.shape[1]):
            label_[:,i,:] = cv2.medianBlur(label__[:,i,:], k_size)
    if view_index == 2:
        for i in range(label.shape[2]):
            label_[:,:,i] = cv2.medianBlur(label__[:,:,i], k_size)
    return label_
    
#%%
def label_voting(label_all):
    """
    simple counting based voting
    """
    print('voting number', len(label_all))
    
    shape = label_all[0].shape
    score_label = np.zeros((3,shape[0],shape[1],shape[2]))
    label_all_ = np.array(label_all)
    
    score_label[2,:,:,:] = np.sum(label_all_==2,axis=0)
    score_label[1,:,:,:] = np.sum(label_all_==1,axis=0)
    score_label[0,:,:,:] = np.sum(label_all_==0,axis=0)
    
    predict = np.argmax(score_label, axis=0)    
    return predict

#%%
from scipy.ndimage import measurements # label

#%%
def get_score_map_dev(view_index, slice_index, class_index):
    if view_index == 0:
        slice_max = score_map0_0.shape[0]-1
        
        if class_index == 0:
            if slice_index == 0:
                current_map = score_map0_0[slice_index,:,:]+score_map0_0[slice_index+1,:,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map0_0[slice_index,:,:]+score_map0_0[slice_index-1,:,:]
                current_map = current_map/2
            else:
                current_map = score_map0_0[slice_index,:,:]+score_map0_0[slice_index-1,:,:]+score_map0_0[slice_index+1,:,:]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
        if class_index == 1:
            if slice_index == 0:
                current_map = score_map0_1[slice_index,:,:]+score_map0_1[slice_index+1,:,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map0_1[slice_index,:,:]+score_map0_1[slice_index-1,:,:]
                current_map = current_map/2
            else:
                current_map = score_map0_1[slice_index,:,:]+score_map0_1[slice_index-1,:,:]+score_map0_1[slice_index+1,:,:]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
        if class_index == 2:
            if slice_index == 0:
                current_map = score_map0_2[slice_index,:,:]+score_map0_2[slice_index+1,:,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map0_2[slice_index,:,:]+score_map0_2[slice_index-1,:,:]
                current_map = current_map/2
            else:
                current_map = score_map0_2[slice_index,:,:]+score_map0_2[slice_index-1,:,:]+score_map0_2[slice_index+1,:,:]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
    if view_index == 1:
        slice_max = score_map1_0.shape[1]-1
        
        if class_index == 0:
            if slice_index == 0:
                current_map = score_map1_0[:,slice_index,:]+score_map1_0[:,slice_index+1,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map1_0[:,slice_index,:]+score_map1_0[:,slice_index-1,:]
                current_map = current_map/2
            else:
                current_map = score_map1_0[:,slice_index,:]+score_map1_0[:,slice_index-1,:]+score_map1_0[:,slice_index+1,:]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
        if class_index == 1:
            if slice_index == 0:
                current_map = score_map1_1[:,slice_index,:]+score_map1_1[:,slice_index+1,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map1_1[:,slice_index,:]+score_map1_1[:,slice_index-1,:]
                current_map = current_map/2
            else:
                current_map = score_map1_1[:,slice_index,:]+score_map1_1[:,slice_index-1,:]+score_map1_1[:,slice_index+1,:]
                current_map = current_map/3
            
            current_map = score_map1_1[:,slice_index,:]
            return copy.deepcopy(current_map)
        
        if class_index == 2:
            if slice_index == 0:
                current_map = score_map1_2[:,slice_index,:]+score_map1_2[:,slice_index+1,:]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map1_2[:,slice_index,:]+score_map1_2[:,slice_index-1,:]
                current_map = current_map/2
            else:
                current_map = score_map1_2[:,slice_index,:]+score_map1_2[:,slice_index-1,:]+score_map1_2[:,slice_index+1,:]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
    if view_index == 2:
        slice_max = score_map2_0.shape[2]-1
        
        if class_index == 0:
            if slice_index == 0:
                current_map = score_map2_0[:,:,slice_index]+score_map2_0[:,:,slice_index+1]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map2_0[:,:,slice_index]+score_map2_0[:,:,slice_index-1]
                current_map = current_map/2
            else:
                current_map = score_map2_0[:,:,slice_index]+score_map2_0[:,:,slice_index-1]+score_map2_0[:,:,slice_index+1]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
        if class_index == 1:
            if slice_index == 0:
                current_map = score_map2_1[:,:,slice_index]+score_map2_1[:,:,slice_index+1]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map2_1[:,:,slice_index]+score_map2_1[:,:,slice_index-1]
                current_map = current_map/2
            else:
                current_map = score_map2_1[:,:,slice_index]+score_map2_1[:,:,slice_index-1]+score_map2_1[:,:,slice_index+1]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)
        
        if class_index == 2:
            if slice_index == 0:
                current_map = score_map2_2[:,:,slice_index]+score_map2_2[:,:,slice_index+1]
                current_map = current_map/2
            elif slice_index == slice_max:
                current_map = score_map2_2[:,:,slice_index]+score_map2_2[:,:,slice_index-1]
                current_map = current_map/2
            else:
                current_map = score_map2_2[:,:,slice_index]+score_map2_2[:,:,slice_index-1]+score_map2_2[:,:,slice_index+1]
                current_map = current_map/3
            
            return copy.deepcopy(current_map)

def generate_1_predicted_labels_dev(mode='single',
                                view_index=0,
                                background_threshold=0,
                                muscle_threshold=0,
                                blood_threshold=0,
                                factor=0):
    label = np.zeros(data.shape, dtype=np.int16)

    for i in range(data.shape[view_index]):
        #% add mode
        slice_index = i
        if mode == 'single':
            prob_c0 = get_score_map_dev(view_index, slice_index, 0)#[y0:y0+y_l,x0:x0+x_l]
            prob_c1 = get_score_map_dev(view_index, slice_index, 1)#[y0:y0+y_l,x0:x0+x_l]
            prob_c2 = get_score_map_dev(view_index, slice_index, 2)#[y0:y0+y_l,x0:x0+x_l]        
        elif mode == 'add':
            prob_c0 = get_score_map_add(view_index, slice_index, 0)#[y0:y0+y_l,x0:x0+x_l]
            prob_c1 = get_score_map_add(view_index, slice_index, 1)#[y0:y0+y_l,x0:x0+x_l]
            prob_c2 = get_score_map_add(view_index, slice_index, 2)#[y0:y0+y_l,x0:x0+x_l]   
        else:
            print('Error mode')
            return 0              
        #%
        pred, _ = generate_prediction_from_slices_dev_dev(prob_c0, 
                                                          prob_c1, 
                                                          prob_c2,
                                                          show=False,
                                                          factor=factor,
                                                          threshold=background_threshold,
                                                          muscle_threshold=muscle_threshold,
                                                          blood_threshold=blood_threshold)
        pred = crop_edge_pixel(pred)
        
        if view_index == 2:
            label[:,:,i] = pred # 2 - pred
        if view_index == 1:
            label[:,i,:] = pred # 2 - pred
        if view_index == 0:
            label[i,:,:] = pred # 2 - pred
            
    return label

def generate_3_predicted_labels_dev(label_all,mode='single', # single view or add view
                                background_threshold=0,
                                muscle_threshold=0,
                                blood_threshold=0,
                                factor=0):
    """
    append 3 predicted labels to current label list
    we treat 3 view equally
    """
    for i in range(3):
        label = generate_1_predicted_labels_dev(mode=mode,
                                                view_index=i,
                                                background_threshold=background_threshold,
                                                muscle_threshold=muscle_threshold,
                                                blood_threshold=blood_threshold,
                                                factor=factor)
        # label = change_label(label,[2,1,0],[2,0,1])
        label_all.append(copy.deepcopy(label))
    print('current ensemble number is:', len(label_all))

#%%

#%%

#%%

#%%


































