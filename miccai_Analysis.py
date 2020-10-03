from help import *
#%%
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
# load raw data
test_index = 14
nii_data = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Test dataset/testing_axial_crop_pat'+str(test_index)+'.nii.gz'

img = nib.load(nii_data)
data = img.get_fdata()
print(data.shape)
#%%
data_all = []
label_all = []
for i in range(10):
    sample_no = i
    nii_data = r'training_axial_crop_pat'+str(sample_no)+r'.nii.gz'
    img = nib.load(os.path.join(nii_data_dir, nii_data))
    data_tr = img.get_fdata()
    data_all.append(data_tr)
    
    gt = r'training_axial_crop_pat'+str(sample_no)+r'-label.nii.gz'
    img_gt = nib.load(os.path.join(gt_dir, gt))
    data_gt = img_gt.get_fdata()
    label_all.append(data_gt)

for i in range(10):
    test_index = i+10
    nii_data = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/TopoSeg/gluon/Axial-cropped/Test dataset/testing_axial_crop_pat'+str(test_index)+'.nii.gz'
    
    img = nib.load(nii_data)
    data = img.get_fdata()
    data_all.append(data)
    
    gt = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/result_0.319/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
    img_gt = nib.load(os.path.join(gt_dir, gt))
    data_gt = img_gt.get_fdata()
    label_all.append(data_gt)
#%%
from scipy.stats import wasserstein_distance
# wasserstein_distance([0, 1, 3], [5, 6, 8])

wd_matrix = np.zeros((20,20))

for i in range(20):
    for j in range(i,20):
        print(i,j)
        dist_1 = image_norm(data_all[i].flatten(),0)#,inten_cut[i])
        dist_2 = image_norm(data_all[j].flatten(),0)#,inten_cut[j])
        
        dist_1 = np.sort(dist_1)
        dist_2 = np.sort(dist_2)
        
        wd = wasserstein_distance(dist_1,dist_2)
        wd_matrix[i,j] = wd
        wd_matrix[j,i] = wd
    # break
#%%
#%%
#%%
with open('figure/wd_matrix.pkl','wb') as fo:
    pickle.dump(wd_matrix, fo)
#%%
with open('figure/wd_matrix_nocut.pkl','rb') as fi:
    wd_matrix_nocut = pickle.load(fi)
#%%
plt.imshow(wd_matrix*0+wd_matrix_nocut, cmap='bwr')

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
## corr = np.corrcoef(wd_matrix_nocut) # Pearson conduct 
#%%
ax_cluster = sns.clustermap(wd_matrix_nocut, figsize=(6,6),
               cmap='bwr',#'RdYlGn_r', 
               linewidths=.3,
               cbar_kws={'label': 'Wasserstein distance'},                  
               metric="canberra") # correlation
# .fig.suptitle('Cluster of 20 samples based on Wasserstein metric') # 
# metric: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

# plt.tight_layout()

#%%
# remove cluster: ref: https://stackoverflow.com/questions/42417798/how-to-suppress-drawing-dendrograms-but-still-cluster-in-seaborn
ax_cluster.ax_row_dendrogram.set_visible(False)
ax_cluster.ax_col_dendrogram.set_visible(False)

ax_cluster.cax.set_visible(False)
#%%
'''
ax_cluster.fig.suptitle('(A) Cluster of 20 samples based on Wasserstein metric', 
                        verticalalignment='baseline',
                        horizontalalignment='right')
'''
#%% ========== ========== ========== ========== ========== ==========
intensity = data_tr[:,:,70] 
label = data_gt[:,:,70]

show_subplots([intensity, label])

intensity_blood = intensity[label==2]
intensity_muscle = intensity[label==1]
intensity_background = intensity[label==0]

dist_multi([intensity,intensity_blood,intensity_muscle])

#%% ========== ========== ========== ========== ========== ==========
img_pred = nib.load('C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/result_0.319/testing_axial_crop_pat14-label.nii.gz')
data_pred = img_pred.get_fdata()
print(data_pred.shape)
#%% ========== ========== ========== ========== ========== ==========
intensity = data[:,154,:] 
label = data_pred[:,154,:]

show_subplots([intensity, label])

intensity_blood = intensity[label==2]
intensity_muscle = intensity[label==1]
intensity_background = intensity[label==0]

dist_multi([intensity,intensity_blood,intensity_muscle])

#%%
pat_index = 14 # test_index
view_index = 1
slice_index = 154
#%%
test_styled_dir = r'G:\Project\UB\Projects\StyleSegor\JPEGImages_test_styled_cover'            
image = Image.open(os.path.join(test_styled_dir, 
                                'testing_axial_crop_pat'+str(pat_index)+'_v'+str(view_index)+'_'+str(slice_index)+'_styled_.jpg'))
img = np.array(image)[:,:,1]

show(img)
#%%
image = Image.open('testing_axial_crop_pat'+str(pat_index)+'_v'+str(view_index)+'_'+str(slice_index)+'_styled_.jpg')
img = np.array(image)[:,:,1]

show(img)
#%%
intensity = img 
label = data_pred[:,154,:]

show_subplots([intensity, label])

intensity_blood = intensity[label==2]
intensity_muscle = intensity[label==1]
intensity_background = intensity[label==0]

dist_multi([intensity,intensity_blood,intensity_muscle])

#%%
# then we want to output three files
'''
intensity = data_tr[:,:,70] 
label = data_gt[:,:,70]
'''

'''
intensity = data[:,154,:] 
label = data_pred[:,154,:]

label_smo = label_smoothing(np.array([label]))
label = label_smo[0,:,:]
'''

intensity = img 
label = data_pred[:,154,:]

with open('figure/inten_style.csv', "w") as fo:
    fo.write('intensity,label\n')
    for i in range(intensity.size):
        value = intensity.flatten()[i]
        mark = label.flatten()[i]
        if mark == 0:
            mark_ = 'background'
        if mark == 1:
            mark_ = 'myocardium'
        if mark == 2:
            mark_ = 'blood'
        fo.write(str(value)+','+mark_+'\n')
#%%
plt.imshow(intensity, cmap='gray')
# https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
#%%
# plt.imshow(label, alpha=0.3)
# follow: https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph

x0, y0, x1, y1 = 0, 0, label.shape[1], label.shape[0]
label_index = 2
color_list = ["#FF8B8B", "#00AFBB"]

import numpy as np
import matplotlib.pyplot as plt

# our image with the numbers 1-3 is in array maskimg
# create a boolean image map which has trues only where maskimg[x,y] == 3
mapimg = (label == label_index)

# a vertical line segment is needed, when the pixels next to each other horizontally
#   belong to diffferent groups (one is part of the mask, the other isn't)
# after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

# the same is repeated for horizontal segments
hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

# if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
#   (2,7) and (2,8), i.e. from (2,8)..(3,8)
# in order to draw a discountinuous line, we add Nones in between segments
l = []
for p in zip(*hor_seg):
    l.append((p[1], p[0]+1))
    l.append((p[1]+1, p[0]+1))
    l.append((np.nan,np.nan))

# and the same for vertical segments
for p in zip(*ver_seg):
    l.append((p[1]+1, p[0]))
    l.append((p[1]+1, p[0]+1))
    l.append((np.nan, np.nan))

# now we transform the list into a numpy array of Nx2 shape
segments = np.array(l)

# now we need to know something about the image which is shown
#   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
#   drawn with origin='lower'
# with this information we can rescale our points
segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
segments[:,1] = y0 + (y1-x0) * segments[:,1] / mapimg.shape[0]

# and now there isn't anything else to do than plot it
plt.plot(segments[:,0], segments[:,1], 
         color=color_list[label_index-1], alpha=1, # 0.5, # 
         linewidth=0.8)
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
# plot mean intensity
three_mean = []

for i in range(20):
    # we have data_all and label_all
    mean_0 = np.mean(data_all[i][label_all[i]==0])    
    mean_1 = np.mean(data_all[i][label_all[i]==1])    
    mean_2 = np.mean(data_all[i][label_all[i]==2])    
    three_mean.append([mean_0, mean_1, mean_2])
#%%
three_mean_ = np.array(three_mean_[index])
plt.imshow(three_mean, cmap='coolwarm')
plt.colorbar()
#%%
index = [3,4,0,9,1,8,6,7,2,5,14,16,17,10,19,15,12,13,11,18]
#%% ========== ========== ========== ========== ========== ==========
test_index = 16
gt_ensemble = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/result_0.319/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_deeplabv3 = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/no_factor_0/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_deeplabv3_adj = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/no_factor_multi/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_stylesegor = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/with_factor_0/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_stylesegor_adj = r'C:/Users/horsepurve/Dropbox/UBR/Analysis/StyleSegor/submit/with_factor_multi/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_3ddsn = r'D:/Project/UB/Projects/TopoSeg/DenseVoxNet-tensorflow/matlab/code/test_dsn/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'
gt_densevoxnet = r'D:/Project/UB/Projects/TopoSeg/DenseVoxNet-tensorflow/matlab/code/test_dense/testing_axial_crop_pat'+str(test_index)+'-label.nii.gz'

gt_string_list = [gt_ensemble, gt_deeplabv3, gt_deeplabv3_adj,
                  gt_stylesegor, gt_stylesegor_adj, gt_3ddsn, gt_densevoxnet]

sample_all = []
for i in range(7):
    gt = gt_string_list[i]
    img_gt = nib.load(os.path.join(gt_dir, gt))
    data_gt = img_gt.get_fdata()
    sample = np.zeros(data_gt[94,:,:].shape)
    sample[data_gt[94,:,:]==2]=-1
    sample[data_gt[94,:,:]==1]=1
    sample_all.append(sample)
#%%
plt.figure(figsize=(6,7))
f_size = 11
plt.subplot(3,3,1)
plt.imshow(data[94,:,:], cmap='gray')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(B) Testing slice',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,6)
plt.imshow(img, cmap='gray')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(G) Transfered slice',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,2)
plt.imshow(sample_all[-2], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(C) 3D-DSN',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,3)
plt.imshow(sample_all[-1], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(D) DenseVoxNet',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,4)
plt.imshow(sample_all[1], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(E) DeepLabv3',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,7)
plt.imshow(sample_all[3], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(H) StyleSegor',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,5)
plt.imshow(sample_all[2], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(F) DeepLabv3 (adjusted)',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,8)
plt.imshow(sample_all[-3], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(I) StyleSegor (adjusted)',fontweight = 'bold', fontsize=f_size)

plt.subplot(3,3,9)
plt.imshow(sample_all[0], cmap='bwr')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.title('(J) StyleSegor (ensemble)',fontweight = 'bold', fontsize=f_size)

plt.tight_layout(w_pad=0.3,h_pad=0.1) # pad=0.3, w_pad=0.3, h_pad=0.5

#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========













