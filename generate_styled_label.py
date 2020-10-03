raw_image_dir = r'/media/MyDataStor2/chunweim/data/HVSMR/JPEGImages_no_style'
label_dir = r'/media/MyDataStor2/chunweim/data/HVSMR/SegmentationClass'
output_dir = r'/home/chunweim/chunweim/projects/StyleSegor/JPEGImages_styled'
output_label_dir = r'/home/chunweim/chunweim/projects/StyleSegor/SegmentationClass_styled'
mode = 'train'
print('mode:', mode)

from os import listdir
from os.path import isfile, join
raw_images = [f for f in listdir(raw_image_dir) if isfile(join(raw_image_dir, f))]

import timeit
import imageio
import os

#%% ========== ========== ========== ========== ========== ========== 
# import library

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
print('device:', device)

def generate_styled_label(image_1, image_1_styled_label):
    width = 512
    style_img = read_image(os.path.join(label_dir, image_1), target_width=width).to(device)
    img = style_img.cpu().detach().numpy()[0,:,:,:]
    img = img.transpose((1, 2, 0))
    imageio.imwrite(image_1_styled_label,img)    

#%% ========== ========== ========== ========== ========== ========== 
def generate_styled_image(image_1, image_2, image_2_styled):
    """
    parameters: full path
    image_1 = 'training_axial_crop_pat0_v0_36.jpg'
    image_2 = 'testing_axial_crop_pat15_v0_63.jpg'
    image_2_styles = 'testing_axial_crop_pat15_v0_63_0.jpg' # using pat0 style
    """
    width = 512
    style_img = read_image(os.path.join(raw_image_dir, image_1), target_width=width).to(device)
    content_img = read_image(os.path.join(raw_image_dir,image_2), target_width=width).to(device)
    #%% ========== ========== ========== ========== ========== ==========
    # build model
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = VGG(vgg16.features[:23]).to(device).eval()

    # compute feature
    style_features = vgg16(style_img)
    content_features = vgg16(content_img)
    #%% ========== ========== ========== ========== ========== ==========
    # compute Gram matrix
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    style_grams = [gram_matrix(x) for x in style_features]
    #%% ========== ========== ========== ========== ========== ==========
    # trining image
    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    style_weight = 1e6 # 1e6
    content_weight = 1 # 1

    run = [0]
    while run[0] <= 50: # 300
        def f():
            optimizer.zero_grad()
            features = vgg16(input_img)
            
            content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
            style_loss = 0
            grams = [gram_matrix(x) for x in features]
            for a, b in zip(grams, style_grams):
                style_loss += F.mse_loss(a, b) * style_weight
            
            loss = style_loss + content_loss
            
            if run[0] % 50 == 0:
                print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                    run[0], style_loss.item(), content_loss.item()))
            run[0] += 1
            
            loss.backward()
            return loss
        
        optimizer.step(f)
    #%% ========== ========== ========== ========== ========== ==========
    img = input_img.cpu().detach().numpy()[0,:,:,:]
    img = img.transpose((1, 2, 0))
    imageio.imwrite(image_2_styled,img)    
    #%% ========== ========== ========== ========== ========== ==========

#%% ========== ========== ========== ========== ========== ========== 
# load background lines:
import pickle
with open('id2y_background.pkl', "rb") as fi:
    id2y_background = pickle.load(fi)

#%% ========== ========== ========== ========== ========== ========== 
for k in range(len(raw_images)):
    print('\n')
    image_path = os.path.join(raw_image_dir, raw_images[k])
    sample_name = raw_images[k].split('.')[0]
    if mode == 'train':
        if sample_name.startswith('test'):
            print('skip:', raw_images[k])
            continue
    if mode == 'test':
        if sample_name.startswith('train'):
            print('skip:', raw_images[k])
            continue        

    print('processing:', raw_images[k])
    #%% ========== ========== ========== ========== ========== ==========
    # we want to find 10 positions:
    sample_no = int(sample_name.split('_')[3][3:])
    view_index = int(sample_name.split('_')[4][1:])
    slice_index = int(sample_name.split('_')[-1])

    slice_percentage = id2y_background[sample_no][view_index][slice_index]
    peak_x_this_sample = np.argmax(id2y_background[sample_no][view_index])

    for i in range(10): # for each training slice, we only consider 10 testing style
        # note: below is the new sample_no
        if mode == 'train':
            sample_no = i + 10 
        if mode == 'test':
            sample_no = i
        fixed_array = id2y_background[sample_no][view_index]
        peak_x = np.argmax(fixed_array)
        mapped_x_1 = np.argmin(np.abs(fixed_array[:peak_x] - slice_percentage))
        mapped_x_2 = np.argmin(np.abs(fixed_array[peak_x:] - slice_percentage)) + peak_x
        if slice_index > peak_x_this_sample:
            mapped_x = mapped_x_2
        else:
            mapped_x = mapped_x_1
        
        #%% ========== ========== ========== ========== ========== ==========
        # now, we have this position: mapped_x
        image_2 = image_path # current image is the content image
        if mode == 'train':
            # this is test image style, used for train set:
            image_1_name = 'testing_axial_crop_pat'+str(sample_no)+'_v'+str(view_index)+'_'+str(mapped_x)+'.png' # png here
        if mode == 'test':
            image_1_name = 'training_axial_crop_pat'+str(sample_no)+'_v'+str(view_index)+'_'+str(mapped_x)+'.jpg'
        image_1 = os.path.join(label_dir, image_1_name) 
        image_2_styled_name = sample_name.split('.')[0]+'_'+str(sample_no)+'.png' # png here
        image_2_styled = os.path.join(output_label_dir, image_2_styled_name)

        print('content image:', image_2)
        print('style image:', image_1)
        print('content image output:', image_2_styled)

        #%% ========== ========== ========== ========== ========== ==========
        '''
        tic = timeit.default_timer()

        generate_styled_image(image_1, image_2, image_2_styled)

        toc = timeit.default_timer()
        print('time:', toc-tic)
        '''
        generate_styled_label(image_1, image_2_styled)

    # break # only one image to see
    #%% ========== ========== ========== ========== ========== ==========


