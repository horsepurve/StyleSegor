import os
os.chdir(r'C:\Users\horsepurve\Dropbox\UBR\Analysis\SemiSegor')

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
#%%
import nibabel as nib
nii_data_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Training dataset/'
gt_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg\gluon/Axial-cropped/Ground truth/'

sample_no = 1

nii_data = r'training_axial_crop_pat'+str(sample_no)+r'.nii.gz'
gt = r'training_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'

img = nib.load(os.path.join(nii_data_dir, nii_data))
data = img.get_fdata()
print(data.shape)

img_gt = nib.load(os.path.join(gt_dir, gt))
data_gt = img_gt.get_fdata()
print(data_gt.shape)

#%%
import pickle

slice_no = 63

score_map_dir = r'C:\Users\horsepurve\Desktop\scoredir' # <- test set
# score_map_dir = r'C:\Users\horsepurve\Desktop\result_train_set\scoredir' # <- train set
score_map_name = 'training_axial_crop_pat'+str(sample_no)+r'_'+str(slice_no)+r'.pkl'
score_map_path = os.path.join(score_map_dir, score_map_name)

with open(score_map_path, 'rb') as fi:
    score_map = pickle.load(fi)
print(score_map.shape)
#%%
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
#%%
raw_image = data[:,:,slice_no]
# plt.imshow(raw_image)
#%%
markers = np.zeros(raw_image.shape, dtype=np.uint)

background_ratio = 0.6
musule_ratio = 0.1
blood_ratio = 0.2

pixle_number = raw_image.shape[0]*raw_image.shape[1]

background_cut = np.sort(score_map[2,:,:], axis=None)[::-1][int(background_ratio*pixle_number)]
musule_cut = np.sort(score_map[1,:,:], axis=None)[::-1][int(musule_ratio*pixle_number)]
blood_cut = np.sort(score_map[2,:,:], axis=None)[::-1][int(blood_ratio*pixle_number)]

markers[score_map[2,:,:]<background_cut] = 1
markers[score_map[1,:,:]>musule_cut] = 2
markers[score_map[2,:,:]>blood_cut] = 3

plt.figure()
plt.imshow(markers)
#%%
raw_image_ = rescale_intensity(raw_image, out_range=(-1, 1))
labels = random_walker(raw_image_, markers, beta=0.1, mode='cg_mg',tol=0.01)

plt.figure()
plt.imshow(labels)
#%%
from PIL import Image

image = Image.open('C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/RW/random_walker_matlab_code/training_axial_crop_pat0_77.jpg')

plt.imshow(image)

#%%
# try chan vese method
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese

image = img_as_float(data.camera())
image = score_map_1
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()

#%%
# try watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

image = img_as_ubyte(data.camera())
image = score_map_1 # img.get_fdata()[:,:,77]

image = rescale_intensity(image, out_range=(-1, 1))
# denoise image
# denoised = rank.median(image, disk(2))
denoised = image

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
#%%
# try Edge operators
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt

# image = camera()
image = score_map_1 # img.get_fdata()[:,:,71]

edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
#%%
# write Edge operators function
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt

def edge_operators(image):
    edge_roberts = roberts(image)
    edge_sobel = sobel(image)
    
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))
    
    ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
    ax[0].set_title('Roberts Edge Detection')
    
    ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    ax[1].set_title('Sobel Edge Detection')
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()
    plt.show()

from scipy import ndimage as ndi
from skimage import feature

def canny(im):
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=3)
    
    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)
    
    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
    
    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)
    
    fig.tight_layout()
    
    plt.show()    

#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% 3 views integration
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import pickle
from PIL import Image

# load raw data
test_index = 14
nii_data = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Test dataset/testing_axial_crop_pat'+str(test_index)+'.nii.gz'

img = nib.load(nii_data)
data = img.get_fdata()
print(data.shape)

#%%

score_dir0 = r'I:\UB\HVSMR\scoredir_no_style_v0_1scale'
score_dir1 = r'I:\UB\HVSMR\scoredir_no_style_v1_1scale'
score_dir2 = r'I:\UB\HVSMR\scoredir_no_style_v2_1scale'

sample_name = nii_data.split('/')[-1].split('.')[0]
'''

score_dir0 = r'G:\HVSMR\SemiSegor\scoredir_3v'
score_dir1 = r'G:\HVSMR\SemiSegor\scoredir_3v'
score_dir2 = r'G:\HVSMR\SemiSegor\scoredir_3v'

sample_name = nii_data.split('/')[-1].split('.')[0]
'''
#%%
# view index: 0
score_map0_0 = np.zeros(data.shape)
score_map0_1 = np.zeros(data.shape)
score_map0_2 = np.zeros(data.shape)
for i in range(data.shape[0]):
    slice_name = sample_name+'_v0_'+str(i)+'.pkl' # _styled_
    with open(os.path.join(score_dir0,slice_name), 'rb') as fi:
        predict = pickle.load(fi)
    # i to indicate the number of slices
    predict = np.log(predict)
    score_map0_0[i,:,:]=predict[0,:,:] 
    score_map0_1[i,:,:]=predict[1,:,:] 
    score_map0_2[i,:,:]=predict[2,:,:] 

# plt.imshow(score_map0_0[127,:,:]) # slice no. 127

#%
# view index: 1
score_map1_0 = np.zeros(data.shape)
score_map1_1 = np.zeros(data.shape)
score_map1_2 = np.zeros(data.shape)
for i in range(data.shape[1]):
    slice_name = sample_name+'_v1_'+str(i)+'.pkl' # _styled_
    with open(os.path.join(score_dir1,slice_name), 'rb') as fi:
        predict = pickle.load(fi)
    # i to indicate the number of slices
    predict = np.log(predict)
    score_map1_0[:,i,:]=predict[0,:,:] 
    score_map1_1[:,i,:]=predict[1,:,:] 
    score_map1_2[:,i,:]=predict[2,:,:] 

# plt.imshow(score_map1_0[:,58,:]) # slice no. 58

#%
# view index: 2
score_map2_0 = np.zeros(data.shape)
score_map2_1 = np.zeros(data.shape)
score_map2_2 = np.zeros(data.shape)
for i in range(data.shape[2]):
    slice_name = sample_name+'_v2_'+str(i)+'.pkl' # _styled_
    with open(os.path.join(score_dir2,slice_name), 'rb') as fi:
        predict = pickle.load(fi)
    # i to indicate the number of slices
    predict = np.log(predict)
    score_map2_0[:,:,i]=predict[0,:,:] 
    score_map2_1[:,:,i]=predict[1,:,:] 
    score_map2_2[:,:,i]=predict[2,:,:] 

# plt.imshow(score_map1_0[:,:,151]) # slice no. 151
#%%
# save pkl    
'''
for one sample, we have
score_map0_0
score_map0_1
score_map0_2
score_map1_0
score_map1_1
score_map1_2
score_map2_0
score_map2_1
score_map2_2
'''
print('test sample:', test_index)
pkl_dir = r'D:\Project\UB\Projects\HVSMR\no_style_1scale_pkl' # r'G:\HVSMR\SemiSegor\pkl'

with open(os.path.join(pkl_dir, str(test_index)+'.pkl'), 'wb') as fo:
    pickle.dump(score_map0_0, fo)
    pickle.dump(score_map0_1, fo)
    pickle.dump(score_map0_2, fo)
    pickle.dump(score_map1_0, fo)
    pickle.dump(score_map1_1, fo)
    pickle.dump(score_map1_2, fo)
    pickle.dump(score_map2_0, fo)
    pickle.dump(score_map2_1, fo)
    pickle.dump(score_map2_2, fo)

#%%
# load pkl
print('test sample:', test_index)
pkl_dir = r'G:\HVSMR\SemiSegor\pkl'

with open(os.path.join(pkl_dir, str(test_index)+'.pkl'), 'rb') as fi:
    score_map0_0 = pickle.load(fi)
    score_map0_1 = pickle.load(fi)
    score_map0_2 = pickle.load(fi)
    score_map1_0 = pickle.load(fi)
    score_map1_1 = pickle.load(fi)
    score_map1_2 = pickle.load(fi)
    score_map2_0 = pickle.load(fi)
    score_map2_1 = pickle.load(fi)
    score_map2_2 = pickle.load(fi)    

score_map_add_0 = np.zeros(score_map0_0.shape)
score_map_add_1 = np.zeros(score_map0_0.shape)
score_map_add_2 = np.zeros(score_map0_0.shape)
 
score_map_add_0 = score_map0_0 + score_map1_0 + score_map2_0
score_map_add_1 = score_map0_1 + score_map1_1 + score_map2_1
score_map_add_2 = score_map0_2 + score_map1_2 + score_map2_2

#%%
def get_slice_score_map(view_index, from_view_index, slice_index):
    """
    from_view_index means we fetch the score from what view prediction
    """    
    if view_index == 2:
        if from_view_index == 0:
            return score_map0_0[:,:,slice_index], score_map0_1[:,:,slice_index], score_map0_2[:,:,slice_index]
        if from_view_index == 1:
            return score_map1_0[:,:,slice_index], score_map1_1[:,:,slice_index], score_map1_2[:,:,slice_index]
        if from_view_index == 2:
            return score_map2_0[:,:,slice_index], score_map2_1[:,:,slice_index], score_map2_2[:,:,slice_index]
        if from_view_index == 3:
            return score_map_add_0[:,:,slice_index], score_map_add_1[:,:,slice_index], score_map_add_2[:,:,slice_index]
        
    if view_index == 1:
        if from_view_index == 0:
            return score_map0_0[:,slice_index,:], score_map0_1[:,slice_index,:], score_map0_2[:,slice_index,:]
        if from_view_index == 1:
            return score_map1_0[:,slice_index,:], score_map1_1[:,slice_index,:], score_map1_2[:,slice_index,:]
        if from_view_index == 2:
            return score_map2_0[:,slice_index,:], score_map2_1[:,slice_index,:], score_map2_2[:,slice_index,:]
        if from_view_index == 3:
            return score_map_add_0[:,slice_index,:], score_map_add_1[:,slice_index,:], score_map_add_2[:,slice_index,:]

    if view_index == 0:
        if from_view_index == 0:
            return score_map0_0[slice_index,:,:], score_map0_1[slice_index,:,:], score_map0_2[slice_index,:,:]
        if from_view_index == 1:
            return score_map1_0[slice_index,:,:], score_map1_1[slice_index,:,:], score_map1_2[slice_index,:,:]
        if from_view_index == 2:
            return score_map2_0[slice_index,:,:], score_map2_1[slice_index,:,:], score_map2_2[slice_index,:,:]
        if from_view_index == 3:
            return score_map_add_0[slice_index,:,:], score_map_add_1[slice_index,:,:], score_map_add_2[slice_index,:,:]
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
def show_image_dense(pat_index, view_index, slice_index):
    image_dir = r'G:\HVSMR\HVSMR\JPEGImages'
    image_name = r'testing_axial_crop_pat'+str(pat_index)+'_v'+str(view_index)+'_'+str(slice_index)+'.jpg'
    image = Image.open(os.path.join(image_dir,image_name))
    
    dense_dir = r'G:\HVSMR\3ddsn\test_dense_png_v'+str(view_index)
    dense_name = r'testing_axial_crop_pat'+str(pat_index)+'_'+str(slice_index)+'_.png'
    dense = Image.open(os.path.join(dense_dir,dense_name))
    
    fig = plt.figure(figsize=(5, 8))
    
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.title('raw image')
    
    plt.subplot(2,1,2)
    plt.imshow(dense)    
    plt.title('prediction from VoxDenseNet')
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
def norm_slice(image):
    return (image-image.min()) / (image.max()-image.min())

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
def generate_predict_delta_score(view_index, slice_index, threshold=0, blood_threshold=0, muscle_threshold=0):
    '''
    Note: we only consider this view,
    but we will return two matrix: 1st score - 2nd score, 1st score - 3rd score
    '''
    if view_index == 2:
        score_slice = np.zeros((data.shape[0],data.shape[1],3))
        score_slice[:,:,0] = score_map2_0[:,:,slice_index]
        score_slice[:,:,1] = score_map2_1[:,:,slice_index]+muscle_threshold
        score_slice[:,:,2] = score_map2_2[:,:,slice_index]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 1:
        score_slice = np.zeros((data.shape[0],data.shape[2],3))
        score_slice[:,:,0] = score_map1_0[:,slice_index,:]
        score_slice[:,:,1] = score_map1_1[:,slice_index,:]+muscle_threshold
        score_slice[:,:,2] = score_map1_2[:,slice_index,:]-threshold
        predict = np.argmax(score_slice, axis=2)

    if view_index == 0:
        score_slice = np.zeros((data.shape[1],data.shape[2],3))
        score_slice[:,:,0] = score_map0_0[slice_index,:,:]
        score_slice[:,:,1] = score_map0_1[slice_index,:,:]+muscle_threshold
        score_slice[:,:,2] = score_map0_2[slice_index,:,:]-threshold
        predict = np.argmax(score_slice, axis=2)
        
    # now we have score_slice
    score_delta1 = np.zeros((data.shape[0],data.shape[1]))
    score_delta2 = np.zeros((data.shape[0],data.shape[1]))
    score_slice.sort() # sort along the last axis
    score_delta1 = score_slice[:,:,2] - score_slice[:,:,1]
    score_delta2 = score_slice[:,:,1] - score_slice[:,:,0]
    return predict, score_delta1, score_delta2
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
pat_index = test_index
view_index = 2
slice_index = 60

print(data.shape[view_index])

# show deeplab output
show_3view3label(view_index, slice_index)

# show raw image & dense output
show_image_dense(pat_index, view_index, slice_index)

#%
# plot to see with different threshold
thresholds = [0,2]
blood_threshold = 0
muscle_threshold = 3

show_argmax(view_index,slice_index,thresholds,blood_threshold,muscle_threshold)

## show_argmax2(view_index,slice_index,thresholds,blood_threshold)

#%%
# minor refine
def crop_edge_pixel(predict):
    pix_num = 2
    predict[:,-pix_num:] = 2
    predict[-pix_num:,:] = 2
    
    return predict

#%% ========== ========== ========== ========== ========== ==========
# generate nii.gz label with gaussian mixture
label = np.zeros(data.shape, dtype=np.int16)

background_threshold = 0
for i in range(data.shape[view_index]):
    predict = generate_predict_blood_gaussian(view_index,i)
    predict = crop_edge_pixel(predict)
    if view_index == 2:
        label[:,:,i] = 2 - predict
    if view_index == 1:
        label[:,i,:] = 2 - predict
    if view_index == 0:
        label[i,:,:] = 2 - predict        
#%% ========== ========== ========== ========== ========== ==========
# generate nii.gz label 
label = np.zeros(data.shape, dtype=np.int16)

background_threshold = 0 # 0
for i in range(data.shape[view_index]):
    predict = generate_predict(view_index,
                               i,
                               background_threshold,
                               blood_threshold,
                               muscle_threshold)
    predict = crop_edge_pixel(predict)
    if view_index == 2:
        label[:,:,i] = 2 - predict
    if view_index == 1:
        label[:,i,:] = 2 - predict
    if view_index == 0:
        label[i,:,:] = 2 - predict
#%%        
# generate nii.gz label using add mode
label = np.zeros(data.shape, dtype=np.int16)

background_threshold = 0 # 0
for i in range(data.shape[view_index]):
    predict = generate_predict2(view_index,
                               i,
                               background_threshold,
                               blood_threshold,
                               muscle_threshold)
    predict = crop_edge_pixel(predict)
    if view_index == 2:
        label[:,:,i] = 2 - predict
    if view_index == 1:
        label[:,i,:] = 2 - predict
    if view_index == 0:
        label[i,:,:] = 2 - predict
#%%
label_out = nib.Nifti1Image(label, np.eye(4))

label_name = 'testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
nib.save(label_out, label_name)


#%% ========== ========== ========== ========== ========== ========== 
# nii information
label_name = 'D:/Project/UB/Medical Image/HVSMR 2016/test/test_dense/testing_axial_crop_pat10-label.nii.gz'
img = nib.load(label_name)
hdr = img.header

label_name2 = 'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/testing_axial_crop_pat10-label.nii.gz'
img2 = nib.load(label_name2)
hdr2 = img2.header

#%% ========== ========== ========== ========== ========== ========== 
# change affine and header
# right_nii_dir = 'D:/Project/UB/Medical Image/HVSMR 2016/test/test_dense'
right_nii_dir = 'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/test_dense'
## wrong_nii_dir = 'C:/Users/horsepurve/Dropbox/UBR/Analysis/SemiSegor'
## out_nii_dir = 'C:/Users/horsepurve/Dropbox/UBR/Analysis/SemiSegor/submit'
wrong_nii_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor'
out_nii_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit'

for i in range(10):
    nii_name = 'testing_axial_crop_pat'+str(10+i)+'-label.nii.gz'
    
    img = nib.load(os.path.join(right_nii_dir, nii_name))
    # data = img.get_data()
    
    img2 = nib.load(os.path.join(wrong_nii_dir, nii_name))
    new_data = img2.get_data()
    
    submit_img = nib.Nifti1Image(new_data, img.affine, img.header)
    nib.save(submit_img, os.path.join(out_nii_dir,nii_name))

#%% ========== ========== ========== ========== ========== ========== 
# refinement analysis
import seaborn as sns
plt.style.use('seaborn-white')
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['axes.grid'] = False

#%% 
if view_index == 2:
    slice_score_map = score_map_add_0[:,:,slice_index]
if view_index == 1:
    slice_score_map = score_map_add_0[:,slice_index,:]
if view_index == 0:
    slice_score_map = score_map_add_0[slice_index,:,:]
flattened = slice_score_map.flatten()

plt.figure()
sns.distplot(flattened, kde=False, color="b")
plt.title('blood intensity distribution of slice: '+str(slice_index))

#%% ========== ========== ========== ========== ========== ========== 
# refinement: try gaussian mixture
# ref: https://stackoverflow.com/questions/43386493/what-is-the-correct-way-to-fit-a-gaussian-mixture-model-to-single-feature-data
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
from sklearn import mixture

data_ = np.array(flattened).reshape(-1,1)
clf = mixture.GaussianMixture(n_components=2, 
                              covariance_type='full')
clf.fit(data_)

x = np.array(np.linspace(min(flattened),max(flattened),1000)).reshape(-1,1)
y = clf.score_samples(x)

plt.figure()
plt.plot(x, y*100)
# plt.show()

# print(clf.means_)
# print(clf.covariances_)
threshold_inten = clf.means_[0,0]*2 + 10
print('threshold_inten:', threshold_inten)
#%%
from sklearn import mixture

def show_mixture(flattened):
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    
    x = np.array(np.linspace(min(flattened),max(flattened),1000)).reshape(-1,1)
    y = clf.score_samples(x)
    
    plt.figure()
    plt.plot(x, y)    
    plt.title("means: %.2f, %.2f covariances: %.2f, %.2f" % (clf.means_[0], 
                                              clf.means_[1], 
                                              clf.covariances_[0],
                                              clf.covariances_[1]))

def mixture_anchor(flattened):
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    return np.mean(clf.means_)

def mixture_means(flattened):
    data_ = np.array(flattened).reshape(-1,1)
    clf = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full')
    clf.fit(data_)
    return np.sort(clf.means_.flatten())
#%%
import copy
a_map = copy.deepcopy(score_map_add_0[:,:,slice_index]) # label: 0, view: 2, slice: index
a_map[a_map<=threshold_inten]=0

plt.figure()
plt.imshow(a_map)


#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# generate nii.gz label (3 view labels) as well as delta scores
background_threshold = 0
muscle_threshold = 1
def predict_3D(view_index):
    label = np.zeros(data.shape, dtype=np.int16)
    delta_score_M1 = np.zeros(data.shape, dtype=np.int16)
    delta_score_M2 = np.zeros(data.shape, dtype=np.int16)
    
    for i in range(data.shape[view_index]):
        predict, score_delta1, score_delta2 = generate_predict_delta_score(view_index, 
                                                                           i,
                                                                           threshold=background_threshold,
                                                                           muscle_threshold=muscle_threshold)
        # predict = crop_edge_pixel(predict)
        if view_index == 2:
            label[:,:,i] = 2 - predict
            delta_score_M1[:,:,i] = score_delta1
            delta_score_M2[:,:,i] = score_delta2
        if view_index == 1:
            label[:,i,:] = 2 - predict
            delta_score_M1[:,i,:] = score_delta1
            delta_score_M2[:,i,:] = score_delta2
        if view_index == 0:
            label[i,:,:] = 2 - predict
            delta_score_M1[i,:,:] = score_delta1
            delta_score_M2[i,:,:] = score_delta2
    
    return label, delta_score_M1, delta_score_M2
#% %
print('test sample:', test_index)

pred_v0, deltas_v0_M1, deltas_v0_M2 = predict_3D(0)
pred_v1, deltas_v1_M1, deltas_v1_M2 = predict_3D(1)
pred_v2, deltas_v2_M1, deltas_v2_M2 = predict_3D(2)

#%%
def show_delta(view_index, slice_index):
    fig = plt.figure(figsize=(13, 9))
    
    # view 2: ---------- ---------- ----------    
    if view_index == 2:
        plt.subplot(3,3,1)        
        plt.imshow(pred_v0[:,:,slice_index])
        plt.title('view 0 argmax')
        plt.subplot(3,3,2)        
        plt.imshow(deltas_v0_M1[:,:,slice_index])
        plt.title('delta score 1')
        plt.colorbar()
        plt.subplot(3,3,3)     
        plt.imshow(deltas_v0_M2[:,:,slice_index])
        plt.title('delta score 2')
        plt.colorbar()
        
        plt.subplot(3,3,4)        
        plt.imshow(pred_v1[:,:,slice_index])
        plt.title('view 1 argmax')
        plt.subplot(3,3,5)        
        plt.imshow(deltas_v1_M1[:,:,slice_index])
        plt.title('delta score 1')
        plt.colorbar()
        plt.subplot(3,3,6)     
        plt.imshow(deltas_v1_M2[:,:,slice_index])
        plt.title('delta score 2')
        plt.colorbar()

        plt.subplot(3,3,7)        
        plt.imshow(pred_v2[:,:,slice_index])
        plt.title('view 2 argmax')
        plt.subplot(3,3,8)        
        plt.imshow(deltas_v2_M1[:,:,slice_index])
        plt.title('delta score 1')
        plt.colorbar()
        plt.subplot(3,3,9)     
        plt.imshow(deltas_v2_M2[:,:,slice_index])
        plt.title('delta score 2')
        plt.colorbar()        
    
    plt.suptitle('this view: '+str(view_index)+', slice: '+str(slice_index))

#%%
show_delta(view_index, slice_index)
#%%
def slice_voting(view_index, slice_index):
    """
    simplest voting
    """
    # we asume view_index is 2
    pred_0 = pred_v0[:,:,slice_index]
    pred_1 = pred_v1[:,:,slice_index]
    pred_2 = pred_v2[:,:,slice_index]

    # vot3 = np.logical_and(pred_0 == pred_1, pred_1 == pred_2)
    
    vote_collection = np.zeros((data.shape[0],data.shape[1],3))
    for i in range(3):
        # collect class i
        vote_collection[:,:,i][pred_0 == i] += 1
        vote_collection[:,:,i][pred_1 == i] += 1
        vote_collection[:,:,i][pred_2 == i] += 1
    
    predict = np.argmax(vote_collection, axis=2)    
    return predict

def crop_edge_pixel(predict):
    pix_num = 2
    predict[:,-pix_num:] = 0
    predict[-pix_num:,:] = 0
    
    return predict

#%%
# generate nii.gz label after voting
label = np.zeros(data.shape, dtype=np.int16)

for i in range(data.shape[view_index]):
    predict = slice_voting(view_index, i)
    predict = crop_edge_pixel(predict)
    label[:,:,i] = predict
#%%
label_out = nib.Nifti1Image(label, np.eye(4))

label_name = 'testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
nib.save(label_out, label_name)


#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# ref: http://scipy-lectures.org/packages/scikit-image/auto_examples/plot_threshold.html
# try: image self-adaptive binarization
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure

camera = data.camera()
val = filters.threshold_otsu(camera)

hist, bins_center = exposure.histogram(camera)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(camera, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(camera < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure

def show_otsu(prob_c0_,prob_c1_,prob_c2_,show=True):
    nbins=256
    if show:
        fig = plt.figure(figsize=(10, 4))
    
    val_0 = filters.threshold_otsu(prob_c0_,nbins=nbins)
    if show:
        plt.subplot(1,4,1)
        plt.imshow(prob_c0_ < val_0, cmap='gray', interpolation='nearest')
    
    val_1 = filters.threshold_otsu(prob_c1_,nbins=nbins)
    if show:
        plt.subplot(1,4,2)
        plt.imshow(prob_c1_ < val_1, cmap='gray', interpolation='nearest')

    val_2 = filters.threshold_otsu(prob_c2_,nbins=nbins)
    if show:
        plt.subplot(1,4,3)
        plt.imshow(prob_c2_ < val_2, cmap='gray', interpolation='nearest')
    
    label = np.zeros(prob_c0_.shape)
    label[:,:] = 2
    label[prob_c0_ > val_0] = 0
    label[prob_c1_ > val_1] = 1
    # label[prob_c2_ > val_2] = 2
    
    if show:
        plt.subplot(1,4,4)
        plt.imshow(label)
    return label
    
#%%    









#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
    




































