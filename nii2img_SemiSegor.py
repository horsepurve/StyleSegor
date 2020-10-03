windows = False # True

if windows:
    training_nii_dir = r'D:\Project\UB\Medical Image\HVSMR 2016\axial_crop\Training dataset'
    gt_dir = r'D:\Project\UB\Medical Image\HVSMR 2016\axial_crop\Ground truth'    
    
    jpeg_dir = r'C:\Users\horsepurve\.mxnet\datasets\voc\VOC2012\JPEGImages'
    png_dir = r'C:\Users\horsepurve\.mxnet\datasets\voc\VOC2012\SegmentationClass'
    train_txt = r'C:\Users\horsepurve\.mxnet\datasets\voc\VOC2012\ImageSets\Segmentation\train.txt'
    val_txt = r'C:\Users\horsepurve\.mxnet\datasets\voc\VOC2012\ImageSets\Segmentation\val.txt'
else:
    training_nii_dir = r'/home/chunweim/chunweim/data/Axial-cropped/Training dataset'
    testing_nii_dir = r'/home/chunweim/chunweim/data/Axial-cropped/Test dataset'

    gt_dir = r'/home/chunweim/chunweim/data/Axial-cropped/Ground truth'
    
    jpeg_dir = r'/home/chunweim/chunweim/data/HVSMR/JPEGImages'
    png_dir = r'/home/chunweim/chunweim/data/HVSMR/SegmentationClass'
    train_txt = r'/home/chunweim/chunweim/data/HVSMR/ImageSets/Segmentation/train.txt'
    val_txt = r'/home/chunweim/chunweim/data/HVSMR/ImageSets/Segmentation/val.txt' 
    test_txt = r'/home/chunweim/chunweim/data/HVSMR/ImageSets/Segmentation/test.txt' # [horse changed here]
    
output_mask = False
RGB = True
only_test = True

train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
val_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

import os
import numpy as np
import nibabel as nib
import imageio
from matplotlib import cm   
from PIL import Image
import pickle
import scipy.io as sio
import seaborn as sns

def plot_cut2(data):
    line_dense = sns.distplot(data.flatten(), 
                              hist=False,
                              kde_kws={"color": 'k',"alpha":0.8}).get_lines()[-1].get_data()    
    density_percentage = 0.01
    density_cut = density_percentage*max(line_dense[1])
    inten_cut = max(line_dense[0][line_dense[1]>density_cut])
    return inten_cut

def nii2view(fo,
             sample_index = 0,
             view=2):
    """
    mode: train or test
    list: train_list or test_list
    """
    if sample_index >= 10:
        mode = 'test'
    else:
        mode = 'train'

    index = sample_index

    if mode == 'train':
        nii_data = os.path.join(training_nii_dir, 'training_axial_crop_pat'+str(index)+'.nii.gz')
        print('>>> sample: ',str(index))
        print('>>> file:', nii_data)            
        gt = os.path.join(gt_dir, 'training_axial_crop_pat'+str(index)+'-label.nii.gz')
        print(gt) 
    if mode == 'test':
        nii_data = os.path.join(testing_nii_dir, 'testing_axial_crop_pat'+str(index)+'.nii.gz')
        print('>>> sample: ',str(index))
        print('>>> file:', nii_data)              

    img = nib.load(nii_data)
    data = img.get_fdata()
    print('>>> data shape', data.shape)
    print('>>> min/max before norm: ',data.min(),data.max())

    inten_cut = plot_cut2(data)
    print('>>> intensity cut:', inten_cut)
    data[data>inten_cut] = inten_cut
    data = (data-data.mean()) / data.std()
    print('>>> min/max after norm: ',data.min(),data.max())
    data = (data-data.min()) / (data.max()-data.min())

    if view == 2:
        num_slice = data.shape[2]
    if view == 1:
        num_slice = data.shape[1]
    if view == 0:
        num_slice = data.shape[0]

    for i in range(num_slice):

        if view == 2:
            a = data[:,:,i]
        if view == 1:
            a = data[:,i,:]
        if view == 0:
            a = data[i,:,:]
        
        if RGB:
            c = cm.viridis(a,bytes=True)[:,:,0:3]
        
        if output_mask:
            ga = data_gt[:,:,i]
            gc = (2-ga).astype(np.uint8) # ga*100 | 2-ga | ga


        if mode == 'train':
            output_name = 'training_axial_crop_pat'+str(index)+'_v'+str(view)+'_'+str(i)
        if mode == 'test':
            output_name = 'testing_axial_crop_pat'+str(index)+'_v'+str(view)+'_'+str(i)
        fo.write(output_name+'\n')
        
        imageio.imwrite(os.path.join(jpeg_dir,output_name+'.jpg'), c) 
            
        if output_mask:
            imageio.imwrite(os.path.join(png_dir,output_name+'.png'), gc)
                    
with open(val_txt, "w") as fo:
    for i in val_list:        
        nii2view(fo, sample_index=i, view=2)
        nii2view(fo, sample_index=i, view=1)
        nii2view(fo, sample_index=i, view=0)
       
import sys
if only_test:
    print('exit')
    sys.exit()

with open(train_txt, "w") as fo:
    for i in train_list:        
        nii2view(fo, sample_index=i, view=2)
        nii2view(fo, sample_index=i, view=1)
        nii2view(fo, sample_index=i, view=0)

print('\ndone.\n')