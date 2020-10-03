import os
os.chdir(r'C:\Users\horsepurve\Dropbox\UBR\Analysis\StyleSegor')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['axes.grid'] = True
from PIL import Image
#%%
# percentage analysis
import nibabel as nib

train_label_dir = r'C:\Users\horsepurve\Dropbox\UBR\Analysis\TopoSeg\gluon\Axial-cropped\Ground truth'
test_label_dir = r'C:\Users\horsepurve\Dropbox\UBR\Analysis\SemiSegor\submit\ablation_study_ad3_mt'

#%%

sample_no = 0

if sample_no < 10:
    gt = r'training_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'
    gt_dir = train_label_dir
else:
    gt = r'testing_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'
    gt_dir = test_label_dir

img_gt = nib.load(os.path.join(gt_dir, gt))
data_gt = img_gt.get_fdata()
print(data_gt.shape)

#%%
def percentage_along(view_index):
    """
    label 0: background
    label 1: muscle
    label 2: blood
    """
    if view_index == 0:
        x = [i for i in range(data_gt.shape[0])]
        area = data_gt.shape[1] * data_gt.shape[2]
        y_background = np.zeros(len(x))
        y_blood = np.zeros(len(x))
        y_muscle = np.zeros(len(x))
        y_muscleOblood = np.zeros(len(x))
        for j in range(len(x)):
            img = data_gt[j,:,:]
            y_background[j] = (img==0).flatten().sum() / area
            y_muscle[j] = (img==1).flatten().sum() / area
            y_blood[j] = (img==2).flatten().sum() / area
            if y_blood[j] != 0:
                y_muscleOblood[j] = y_muscle[j] / y_blood[j]
    if view_index == 1:
        x = [i for i in range(data_gt.shape[1])]
        area = data_gt.shape[0] * data_gt.shape[2]
        y_background = np.zeros(len(x))
        y_blood = np.zeros(len(x))
        y_muscle = np.zeros(len(x))
        y_muscleOblood = np.zeros(len(x))
        for j in range(len(x)):
            img = data_gt[:,j,:]
            y_background[j] = (img==0).flatten().sum() / area
            y_muscle[j] = (img==1).flatten().sum() / area
            y_blood[j] = (img==2).flatten().sum() / area
            if y_blood[j] != 0:
                y_muscleOblood[j] = y_muscle[j] / y_blood[j]
    if view_index == 2:
        x = [i for i in range(data_gt.shape[2])]
        area = data_gt.shape[0] * data_gt.shape[1]
        y_background = np.zeros(len(x))
        y_blood = np.zeros(len(x))
        y_muscle = np.zeros(len(x))
        y_muscleOblood = np.zeros(len(x))
        for j in range(len(x)):
            img = data_gt[:,:,j]
            y_background[j] = (img==0).flatten().sum() / area
            y_muscle[j] = (img==1).flatten().sum() / area
            y_blood[j] = (img==2).flatten().sum() / area
            if y_blood[j] != 0:
                y_muscleOblood[j] = y_muscle[j] / y_blood[j]
    return x, y_background, y_blood, y_muscle, y_muscleOblood
#%%
plt.plot(x, y_background, label='background')
plt.plot(x, y_muscle, label='muscle')
plt.plot(x, y_blood, label='blood')
# plt.plot(x, y_muscleOblood, label='muscle over blood')
plt.legend()
plt.title('sample: '+str(sample_no)+', view: '+str(view_index))
#%%
def plot_one_view(view_index, x, y_background, y_muscle, y_blood):
    plt.plot(x, y_background, label='background')
    plt.plot(x, y_muscle, label='muscle')
    plt.plot(x, y_blood, label='blood')
    # plt.plot(x, y_muscleOblood, label='muscle over blood')
    plt.legend()
    plt.title('sample: '+str(sample_no)+', view: '+str(view_index))     
    
def plot_one_sample():
    plt.figure(figsize=(10,3))
    plt.tight_layout()
    
    x, y_background, y_blood, y_muscle, y_muscleOblood = percentage_along(0)
    plt.subplot(1,3,1)
    plot_one_view(0, x, y_background, y_muscle, y_blood)
    
    x, y_background, y_blood, y_muscle, y_muscleOblood = percentage_along(1)
    plt.subplot(1,3,2)
    plot_one_view(1, x, y_background, y_muscle, y_blood)
    
    x, y_background, y_blood, y_muscle, y_muscleOblood = percentage_along(2)
    plt.subplot(1,3,3)
    plot_one_view(2, x, y_background, y_muscle, y_blood)
    
#%%
plot_one_sample()
plt.savefig(str(sample_no)+'.png')
#%% ========== ========== ========== ========== ========== ========== 
# collect all the data:
id2y_background = {}
id2y_blood = {}
id2y_muscle = {}

for i in range(20):
    sample_no = i
    
    if sample_no < 10:
        gt = r'training_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'
        gt_dir = train_label_dir
    else:
        gt = r'testing_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'
        gt_dir = test_label_dir
    
    img_gt = nib.load(os.path.join(gt_dir, gt))
    data_gt = img_gt.get_fdata()
    print(data_gt.shape)
        
    id2y_background[sample_no] = {} # 0: view 0 list; 1: view 1 list; 2: view 2 list
    
    for i in range(3): # 3 views
        x, y_background, y_blood, y_muscle, y_muscleOblood = percentage_along(i)
        id2y_background[sample_no][i] = (1 - y_background) / (1 - y_background).max()
#%%
import pickle
with open('id2y_background.pkl',"wb") as fo:
    pickle.dump(id2y_background, fo)
#%%
# get the corresponding location for data augmentation step
sample_name = 'training_axial_crop_pat0_v0_53'

sample_no = int(sample_name.split('_')[3][3:])
view_index = int(sample_name.split('_')[4][1:])
slice_index = int(sample_name.split('_')[-1])

slice_percentage = id2y_background[sample_no][view_index][slice_index]
peak_x_this_sample = np.argmax(id2y_background[sample_no][view_index])

for i in range(10): # for each training slice, we only consider 10 testing style
    sample_no = i + 10 
    fixed_array = id2y_background[sample_no][view_index]
    peak_x = np.argmax(fixed_array)
    mapped_x_1 = np.argmin(np.abs(fixed_array[:peak_x] - slice_percentage))
    mapped_x_2 = np.argmin(np.abs(fixed_array[peak_x:] - slice_percentage)) + peak_x
    if slice_index > peak_x_this_sample:
        mapped_x = mapped_x_2
    else:
        mapped_x = mapped_x_1
    print(mapped_x)
#%%
# get the corresponding location for test stage
sample_name = 'testing_axial_crop_pat15_v0_63'

sample_no = int(sample_name.split('_')[3][3:])
view_index = int(sample_name.split('_')[4][1:])
slice_index = int(sample_name.split('_')[-1])

slice_percentage = id2y_background[sample_no][view_index][slice_index]
peak_x_this_sample = np.argmax(id2y_background[sample_no][view_index])

for i in range(10): # for each training slice, we only consider 10 testing style
    sample_no = i
    fixed_array = id2y_background[sample_no][view_index]
    peak_x = np.argmax(fixed_array)
    mapped_x_1 = np.argmin(np.abs(fixed_array[:peak_x] - slice_percentage))
    mapped_x_2 = np.argmin(np.abs(fixed_array[peak_x:] - slice_percentage)) + peak_x
    if slice_index > peak_x_this_sample:
        mapped_x = mapped_x_2
    else:
        mapped_x = mapped_x_1
    print(mapped_x)

#%% ========== ========== ========== ========== ========== ========== 
# prediction diagnosis
import pickle
score_dir = r'D:\Project\UB\Projects\StyleSegor\scoredir_style'
score_file = r'testing_axial_crop_pat16_v0_24_6.pkl'

with open(os.path.join(score_dir, score_file), "rb") as fi:
    score_map = pickle.load(fi)
#%% ========== ========== ========== ========== ========== ========== 
# size recovering analysis
os.chdir(r'C:\Users\horsepurve\Dropbox\UBR\Analysis\StyleSegor')

test_shape = [(192, 269, 190), # 10
                         (229, 300, 110), # 11
                         (172, 257, 205), # 12
                         (191, 295, 190), # 13
                         (151, 218, 176), # 14
                         (260, 228, 215), # 15
                         (151, 195, 156), # 16
                         (185, 279, 214), # 17
                         (214, 279, 169), # 18
                         (154, 206, 216)] # 19

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from utils import *
from models import *

import numpy as np

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
image_dir = r'C:\Users\horsepurve\Desktop\TMP\JPEGImages_test_styled'
output_dir = r'C:\Users\horsepurve\Desktop\TMP\JPEGImages_test_styled_recovered'

image_2 = r'testing_axial_crop_pat10_v0_100_styled_.jpg'

width = (test_shape[0][1],test_shape[0][2]) 

content_img = read_image(os.path.join(image_dir,image_2), target_width=width).to(device)

#%%
import imageio
img = content_img.cpu().numpy()[0,:,:,:]
img = img.transpose((1, 2, 0))
imageio.imwrite(os.path.join(output_dir,image_2),img)

#%%
# do it all
from os import listdir
from os.path import isfile, join
raw_images = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

for k in raw_images:
    image_2 = k
    
    sample_no = int(k.split('_')[3][3:]) - 10
    view_index = int(k.split('_')[4][1:])
    if view_index == 0:
        width = (test_shape[sample_no][1],test_shape[sample_no][2]) 
    if view_index == 1:
        width = (test_shape[sample_no][0],test_shape[sample_no][2]) 
    if view_index == 2:
        width = (test_shape[sample_no][0],test_shape[sample_no][1])
    
    content_img = read_image(os.path.join(image_dir,image_2), target_width=width).to(device)

    img = content_img.cpu().numpy()[0,:,:,:]
    img = img.transpose((1, 2, 0))
    imageio.imwrite(os.path.join(output_dir,image_2),img)
    
    print('->', end='')
#%% ========== ========== ========== ========== ========== ========== 
# integration analysis
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
# load pkl
print('test sample:', test_index)
pkl_dir = r'D:\Project\UB\Projects\HVSMR\no_style_pkl'
## pkl_dir = r'C:\Users\horsepurve\Desktop\TMP\no_1scale_pkl'
## pkl_dir = r'H:\UB\HVSMR\LCloss\no_1scale_pkl'
'''
[we have:]

no_style_pkl

styled_pkl

no_style_1scale_pkl

styled_1scale_pkl
'''
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
pat_index = test_index
view_index = 1
slice_index = 154

print(data.shape[view_index])
#%%
# show deeplab output
show_3view3label(view_index, slice_index)

# show raw image & dense output
## show_image_dense(pat_index, view_index, slice_index)

#%
# plot to see with different threshold
thresholds = [0,2,4,6,8,10]
blood_threshold = 0
muscle_threshold = 2

show_argmax(view_index,slice_index,thresholds,blood_threshold,muscle_threshold)

show_argmax2(view_index,slice_index,thresholds,blood_threshold,muscle_threshold)

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
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# now, we want to do new LP analysis

# 1: we need label
print(label.shape)

# 2: we need intensity
print(data.shape)

pat_index = test_index
view_index = 0
slice_index = 92

intensity = get_slice(data, view_index, slice_index)
show(intensity)

predict_all = get_slice(label, view_index, slice_index)
show(predict_all)

# 3: we need score maps
score_map_0 = get_score_map(view_index, slice_index, 0)
score_map_1 = get_score_map(view_index, slice_index, 1)
score_map_2 = get_score_map(view_index, slice_index, 2)

#%%
# 4. crop patch
x0,y0,x_l,y_l = 85,104,  50,50

show_raw_img(intensity, pat_index,view_index,slice_index,(x0,y0,x_l,y_l))

img_crop = intensity[y0:y0+y_l,x0:x0+x_l]
## show(img_crop)
## plt.title('cropped image')

predict_all_crop = predict_all[y0:y0+y_l,x0:x0+x_l]
## show(predict_all_crop)
## plt.title('cropped predict')

score_map_crop_0 = score_map_0[y0:y0+y_l,x0:x0+x_l]
score_map_crop_1 = score_map_1[y0:y0+y_l,x0:x0+x_l]
score_map_crop_2 = score_map_2[y0:y0+y_l,x0:x0+x_l]

score_map_crop_0_norm = norm_slice(score_map_crop_0)
score_map_crop_1_norm = norm_slice(score_map_crop_1)
score_map_crop_2_norm = norm_slice(score_map_crop_2)
#%%
factor = 1
predict_crop = generate_prediction_from_slices_dev(score_map_crop_0,
                                    score_map_crop_1,
                                    score_map_crop_2,
                                    show=True,
                                    factor=factor)
show(predict_crop)

#%%
# so now we have:
# 1. img_crop
# 2. predict_all_crop
# 3. score_map_crop_0, score_map_crop_1, score_map_crop_2

# ========== ========== ========== ========== ========== ========== 
# try LP
X = get_X(img_crop, view_index)

labels = np.full(len(X), -1.)
labels[:] = predict_all_crop.flatten()

img_crop_norm = norm_slice(img_crop)
#%
size = img_crop.size
X2 = np.hstack((X, 
                (img_crop*(1-(1-score_map_crop_0_norm)*(1-score_map_crop_1_norm))).reshape(size,1),
                
               ))

#%
# Learn with LabelSpreading
label_spread = label_propagation.LabelSpreading(kernel='rbf', gamma=0.01,
                                                alpha=0.9, # 0.2
                                                max_iter=50, # 30
                                                n_neighbors=5, # 7
                                                tol=0.0001 # 0.001
                                                )
label_spread.fit(X2, labels)
#%
output_labels = label_spread.transduction_

if True:
    plt.figure(figsize=(10, 3))
    plt.subplot(1,3,1)
    plt.imshow(img_crop)
    plt.subplot(1,3,2)
    plt.imshow(predict_all_crop)
    plt.subplot(1,3,3)
    plt.imshow(output_labels.reshape(img_crop.shape))    
#%%
from scipy import ndimage
from scipy import misc

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
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# here we want to try image enhancement
pat_index = test_index
view_index = 1
slice_index = 110
#%%
intensity = get_slice(data, view_index, slice_index)

intensity_norm = image_norm(intensity, inten_cut[pat_index])

score_map_0 = get_score_map(view_index, slice_index, 0) # get_score_map_add
score_map_1 = get_score_map(view_index, slice_index, 1) # get_score_map_add
score_map_2 = get_score_map(view_index, slice_index, 2) # get_score_map_add


score_map_0_norm = norm_slice(score_map_0)
score_map_1_norm = norm_slice(score_map_1)
score_map_2_norm = norm_slice(score_map_2)

show_subplots([intensity,score_map_0,score_map_1,score_map_2])
#%%
factor = 1
prob = 1 - (1-score_map_0_norm)*(1-score_map_1_norm)
prob = prob**factor
intensity_mul = intensity * prob

show(intensity_mul)
#%% ========== ========== ========== ========== ========== ========== 
x0,y0,x_l,y_l = 0,0,  700,700

intensity_crop = intensity_norm[y0:y0+y_l,x0:x0+x_l]

prob_c0 = score_map_0[y0:y0+y_l,x0:x0+x_l] # new_map_0
prob_c1 = score_map_1[y0:y0+y_l,x0:x0+x_l] # new_map_1 
prob_c2 = score_map_2[y0:y0+y_l,x0:x0+x_l] # new_map_2
'''

prob_c0 = new_map_0[y0:y0+y_l,x0:x0+x_l] # new_map_0
prob_c1 = new_map_1[y0:y0+y_l,x0:x0+x_l] # new_map_1 
prob_c2 = new_map_2[y0:y0+y_l,x0:x0+x_l] # new_map_2
s
'''

prob_c0_ = norm_slice(prob_c0)
prob_c1_ = norm_slice(prob_c1)
prob_c2_ = norm_slice(prob_c2)
#%%
show_raw_img(intensity, pat_index,view_index,slice_index,(x0,y0,x_l,y_l))

show_subplots([intensity_crop,prob_c0,prob_c1,prob_c2])

#%%
factor = 0

predict, score_slice = generate_prediction_from_slices_dev_dev(score_map_0,
                                                               score_map_1,
                                                               score_map_2,
                                                               show=True,
                                                               factor=factor)

show(predict)

#%%
new_map_0 = score_slice[:,:,0]
new_map_1 = score_slice[:,:,1]
new_map_2 = score_slice[:,:,2]

show_subplots([new_map_0,new_map_1,new_map_2])
#%% naive score cut
mask = np.zeros(new_map_1.shape)
mask[new_map_1>2.5]=1

plt.figure()
plt.imshow(new_map_1)
plt.imshow(mask,alpha=0.3)
#%% ===== ===== ===== ===== ===== =====
a = (Sfun(prob_c1_)+prob_c0_)*intensity_crop
# a = (1 - (1-prob_c0_)*(1-prob_c1_))*intensity_crop
show(a)
# a = intensity_crop
gen_image(a,'run_omen/enhanced.jpg')
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
# now we want to try image guided smoothing
plot3d(prob_c0,prob_c1,prob_c2)

#%%
center = 24
point_cors = np.vstack((prob_c0.flatten(),prob_c1.flatten(),prob_c2.flatten())).T
center_cors = np.array([[prob_c0.flatten()[center],prob_c1.flatten()[center],prob_c2.flatten()[center]]]*49)
Edist = np.linalg.norm(point_cors-center_cors,axis=1)
rank = np.argsort(Edist)
# plt.plot([i for i in range(49)],Edist[rank])
#%% ========== ========== ========== ========== ========== ========== 

kernel = get_kernel(15)
conved = Conv(a, kernel, score_map_0, score_map_1, score_map_2)

show(conved)

gen_image(conved,'run_omen/enhanced.jpg')

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
#%% this is for prediction on score map
a=score_map_0

a = score_map_0+score_map_1*0.2

show(a)

gen_image(a,'run_omen/enhanced.jpg')
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
# here we want to try image synthesis
nii_data_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Training dataset/'
gt_dir = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg\gluon/Axial-cropped/Ground truth/'

sample_no = 1

nii_data = r'training_axial_crop_pat'+str(sample_no)+r'.nii.gz'
gt = r'training_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'

img = nib.load(os.path.join(nii_data_dir, nii_data))
data_tr = img.get_fdata()
print(data_tr.shape)

img_gt = nib.load(os.path.join(gt_dir, gt))
data_gt = img_gt.get_fdata()
print(data_gt.shape)
#%%
pat_index = test_index
view_index = 0
slice_index = 55

image = get_slice(data_tr,view_index, slice_index)
label = get_slice(data_gt,view_index, slice_index)

#%%
# we are focusing on label 1 -> muscle and 2 -> blood
inten_muscle = image[label==1]
print('number of muscle pixels:', len(inten_muscle))

inten_blood = image[label==2]
print('number of blood pixels:', len(inten_blood))
#%%
dist_multi([inten_muscle, inten_blood])

#%%
score_map_0_part = score_map_0 / (score_map_1 + score_map_0)
score_map_1_part = score_map_1 / (score_map_1 + score_map_0)

dist_multi([score_map_1_part,score_map_0_part])

syn_image = score_map_0 / 21 * 1117

show(syn_image)

#%%
a = syn_image
gen_image(a,'run_omen/enhanced.jpg')

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
# integration analysis
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import pickle
from PIL import Image

# load raw data
test_index = 19
nii_data = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Test dataset/testing_axial_crop_pat'+str(test_index)+'.nii.gz'

img = nib.load(nii_data)
data = img.get_fdata()
print(data.shape)
#%%
# load pkl
print('test sample:', test_index)
pkl_dir = r'D:\Project\UB\Projects\HVSMR\no_style_pkl'
## pkl_dir = r'H:\UB\HVSMR\LCloss\no_pkl' # no_pkl no_1scale_pkl with_pkl with_1scale_pkl
'''
[we have:]

no_style_pkl

styled_pkl

no_style_1scale_pkl

styled_1scale_pkl
'''
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
pat_index = test_index
view_index = 2
slice_index = 97

print(data.shape[view_index])
#%%
# show deeplab output
show_3view3label(view_index, slice_index)

# show raw image & dense output
## show_image_dense(pat_index, view_index, slice_index)

#%
# plot to see with different threshold
thresholds = [0,2,4,6,8,10]
blood_threshold = 0
muscle_threshold = 0

show_argmax(view_index,slice_index,thresholds,blood_threshold,muscle_threshold)

show_argmax2(view_index,slice_index,thresholds,blood_threshold,muscle_threshold)

#%%
'''
# generate nii.gz label with gaussian mixture
label = np.zeros(data.shape, dtype=np.int16)

background_threshold = 0
for i in range(data.shape[view_index]):
    predict = generate_predict_blood_gaussian_dev(view_index,i)
    predict = crop_edge_pixel(predict)
    if view_index == 2:
        label[:,:,i] = 2 - predict
    if view_index == 1:
        label[:,i,:] = 2 - predict
    if view_index == 0:
        label[i,:,:] = 2 - predict      
'''        
#%% ========== ========== ========== ========== ========== ==========
# ##### generate nii.gz label #####
view_index = 0
label = np.zeros(data.shape, dtype=np.int16)

background_threshold = 0 # 0
muscle_threshold= 0
for i in range(data.shape[view_index]):
    predict = generate_predict(view_index,
                               i,
                               background_threshold,
                               blood_threshold,
                               muscle_threshold=muscle_threshold)
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
'''
view_index = 2
if view_index == 0:
    label_all = []
# generate nii.gz label using add mode with gaussian
label = np.zeros(data.shape, dtype=np.int16)

for i in range(data.shape[view_index]):
    pred = generate_predict_blood_gaussian_dev(view_index,i)
    
    if view_index == 2:
        label[:,:,i] = 2 - pred
    if view_index == 1:
        label[:,i,:] = 2 - pred
    if view_index == 0:
        label[i,:,:] = 2 - pred

label_all.append(copy.deepcopy(label))
'''
#%% ========== ========== ========== ========== ========== ==========
view_index = 2
if view_index == 0:
    label_all = []
# ##### generate nii.gz label using add mode #####
label = np.zeros(data.shape, dtype=np.int16)

for i in range(data.shape[view_index]):
    #% add mode
    slice_index = i
    prob_c0 = get_score_map_add(view_index, slice_index, 0)#[y0:y0+y_l,x0:x0+x_l]
    prob_c1 = get_score_map_add(view_index, slice_index, 1)#[y0:y0+y_l,x0:x0+x_l]
    prob_c2 = get_score_map_add(view_index, slice_index, 2)#[y0:y0+y_l,x0:x0+x_l]
    #% no normalization
    prob_c0_ = prob_c0
    prob_c1_ = prob_c1
    prob_c2_ = prob_c2    
    #%
    pred, _ = generate_prediction_from_slices_dev_dev(prob_c0_, 
                                                      prob_c1_, 
                                                      prob_c2_,
                                                      show=False,
                                                      factor=0.5,
                                                      threshold=0,
                                                      muscle_threshold=0)
    
    pred = crop_edge_pixel(pred)
    
    if view_index == 2:
        label[:,:,i] = 2 - pred
    if view_index == 1:
        label[:,i,:] = 2 - pred
    if view_index == 0:
        label[i,:,:] = 2 - pred

label_all.append(copy.deepcopy(label))
#%% ========== ========== ========== ========== ========== ==========
label = np.zeros(data.shape, dtype=np.int16)
label[:,:,:] = label_voting(label_all)

#%% ========== ========== ========== ========== ========== ==========
label_v0 = label_all[0] # copy.deepcopy(label)
label_v1 = label_all[1] # copy.deepcopy(label)
label_v2 = label_all[2] # copy.deepcopy(label)

#%%
label = np.zeros(data.shape, dtype=np.int16)
union = np.logical_and(label_v0 == label_v1,label_v1 == label_v2)
label[union] = label_v0[union]
#%% ========== ========== ========== ========== ========== ==========
label_out = nib.Nifti1Image(label, np.eye(4))

label_name = 'testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
nib.save(label_out, label_name)

#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
label = np.zeros(data.shape, dtype=np.int16)
label = (label_v0 + label_v1 + label_v2)/3
label = np.round(label)

show(label[:,154,:])

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
# integration analysis
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import pickle
from PIL import Image

# load raw data
test_index = 10
nii_data = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Test dataset/testing_axial_crop_pat'+str(test_index)+'.nii.gz'

img = nib.load(nii_data)
data = img.get_fdata()
print(data.shape)
#%
# load pkl
print('test sample:', test_index)
pkl_dir = r'F:\Project\UB\Projects\HVSMR\no_style_pkl'
## pkl_dir = r'D:\UB\HVSMR\LCloss\no_1scale_pkl' # no_pkl no_1scale_pkl with_pkl with_1scale_pkl
'''
[we have:]

no_style_pkl

styled_pkl

no_style_1scale_pkl

styled_1scale_pkl
'''
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
#%% ========== ========== ========== ========== ========== ========== 
# large scale ensemble
label_all = []
#%%
mode='single'
background_threshold = 1
muscle_threshold = 2
blood_threshold = -4
factor = 0.2

generate_3_predicted_labels_dev(label_all,mode=mode,
                            background_threshold=background_threshold,
                            muscle_threshold=muscle_threshold,
                            blood_threshold=blood_threshold,factor=factor)


#%%
slice_index = 88
print('ensemble number:', len(label_all))

label = np.zeros(data.shape, dtype=np.int16)

label = np.mean(np.array(label_all), axis=0)
# label2 = np.round(label+0.0)
label2 = label_voting(label_all)

show_subplots([label[slice_index,:,:], label2[slice_index,:,:]])
#%%
## label_a = change_label(label3, [2,1,0], [0,1,2])
## label_b = change_label(label3, [2,0,1], [0,1,2])
'''
label3 = label_smoothing(label2)
label3 = label_smoothing(label3,view_index=1)
label3 = label_smoothing(label3,view_index=2)
'''
label = change_label(label2, [2,1,0], [0,1,2])
#%%
label_out = nib.Nifti1Image(label, np.eye(4))

label_name = 'testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
nib.save(label_out, label_name)
print('saved:',label_name)
#%% ========== ========== ========== ========== ========== ========== 
label_tmp = label
#%%
label_new = label_tmp
label_new[np.logical_and(label==0,label_tmp==2)] = 0


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
# label smoothing
label_img = label2[slice_index,:,:]

label_img = label_img.astype(np.uint8)

img = get_slice(data, view_index, slice_index)
img = norm_slice(img)
img = (img*255).astype(np.uint8)

# label_img = img
#%%
import cv2

img = intensity

k_size = 5
mean_img = cv2.blur(img, (k_size,k_size))
median_img = cv2.medianBlur(img.astype(np.uint8), k_size)

gaussian_img = cv2.GaussianBlur(img, (k_size,k_size), 75)
bilateral_img = cv2.bilateralFilter(img.astype(np.uint8), k_size, 75, 75)

show_subplots([label_img,mean_img,median_img,gaussian_img,bilateral_img])
#%% ========== ========== ========== ========== ========== ========== 
label_all_14 = label_all

#%% ========== ========== ========== ========== ========== ========== 
# find components
from scipy.ndimage import measurements # label

labeled, ncomponents = measurements.label(label)

show(labeled[100,:,:])

label[labeled!=1] = 0 # index: 4
#%%
label_1 = copy.deepcopy(label)
label_2 = copy.deepcopy(label)

label_1[label!=1] = 0
label_2[label!=2] = 0

labeled, ncomponents = measurements.label(label_1)
show(labeled[100,:,:])
label_1[labeled!=1] = 0 # index: 1

labeled, ncomponents = measurements.label(label_2)
show(labeled[100,:,:])
label_2[labeled!=1] = 0 # index: 1

label[np.logical_and(label==1,label_1!=1)] = 0
label[np.logical_and(label==2,label_2!=2)] = 0
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


#%%







