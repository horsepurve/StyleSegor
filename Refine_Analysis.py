#%%
# try refinement
view_index = 0
slice_index = 30
#%%
# this is our material
intensity = get_slice(data, view_index, slice_index)

label_img = get_slice(label, view_index, slice_index)

show_subplots([intensity, label_img])

#%%
intensity_blood = intensity[label_img==2]
intensity_muscle = intensity[label_img==1]

dist_multi([intensity_blood, intensity_muscle])
#%%
kernel = np.ones((2,1), np.uint8)  # note this is a horizontal kernel
d_im = cv2.dilate(e_im, kernel, iterations=2)
e_im = cv2.erode(d_im, kernel, iterations=2) 

show_subplots([l,d_im,e_im])
#%%
label_img_ = copy.deepcopy(e_im)

muscle_int_cut = 640

label_img_[np.logical_and(label_img==1,intensity>muscle_int_cut)] = 2

show(label_img_)

#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# so now we have our tentative label: label
view_index = 0
slice_index = 112

intensity = get_slice(data, view_index, slice_index)

label_img = get_slice(label, view_index, slice_index)

show_subplots([intensity, label_img])
#%%
# we are focusing on label 1 -> muscle and 2 -> blood
inten_muscle = data[label==1]
print('number of muscle pixels:', len(inten_muscle))

inten_blood = data[label==2]
print('number of blood pixels:', len(inten_blood))

dist_multi([inten_muscle, inten_blood])
#%%
label_img = get_slice(label, view_index, slice_index)

blood_cut = 750
blood_cut2 = 2000
label_img[np.logical_and(label_img==2,intensity<blood_cut)] = 0
label_img[np.logical_and(label_img==2,intensity>blood_cut2)] = 0
#%%
muscle_cut = 500
label_img[np.logical_and(label_img==1,intensity>muscle_cut)] = 2

show_subplots([intensity, label_img])

#%%
blood_cut = 750
blood_cut2 = 2000
label[np.logical_and(label==2,data<blood_cut)] = 0
label[np.logical_and(label==2,data>blood_cut2)] = 0

muscle_cut = 500
label[np.logical_and(label==1,data>muscle_cut)] = 2

label = label_smoothing(label)
label = label_smoothing(label,view_index=1)
label = label_smoothing(label,view_index=2)

#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
# try inter-slice smoothing
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
view_index = 0
slice_index = 55

#%%
image = get_slice(data_tr,view_index, slice_index)
label = get_slice(data_gt,view_index, slice_index)

label_a = get_slice(data_gt,view_index, slice_index+1)
label_b = get_slice(data_gt,view_index, slice_index-1)

show_subplots([image, label_b, label, label_a])


#%%

#%%

#%%

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














