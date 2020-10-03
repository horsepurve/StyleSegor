import os
os.chdir(r'C:\Users\horsepurve\Dropbox\UBR\Analysis\StyleSegor')

#%%
#***************************************************************************  
#*   
#* Description: label propagation  
#* Author: Zou Xiaoyi (zouxy09@qq.com)  
#* Date:   2015-10-15  
#* HomePage: http://blog.csdn.net/zouxy09  
#*   
#**************************************************************************  
  
import time  
import numpy as np  
  
# return k neighbors index  
def navie_knn(dataSet, query, k):  
    numSamples = dataSet.shape[0]  
  
    ## step 1: calculate Euclidean distance  
    diff = np.tile(query, (numSamples, 1)) - dataSet # calculate delta_x and delta_y
    squaredDiff = diff ** 2  
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row  
  
    ## step 2: sort the distance  
    sortedDistIndices = np.argsort(squaredDist)  
    if k > len(sortedDistIndices):  
        k = len(sortedDistIndices)  
  
    return sortedDistIndices[0:k]  
  
  
# build a big graph (normalized weight matrix)  
def buildGraph(MatX, kernel_type, rbf_sigma = None, knn_num_neighbors = None):  
    num_samples = MatX.shape[0]  
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)  
    if kernel_type == 'rbf':  
        if rbf_sigma == None:  
            raise ValueError('You should input a sigma of rbf kernel!')  
        for i in range(num_samples):  
            row_sum = 0.0  
            for j in range(num_samples):  
                diff = MatX[i, :] - MatX[j, :]  
                affinity_matrix[i][j] = np.exp(sum(diff**2) / (-2.0 * rbf_sigma**2))  
                row_sum += affinity_matrix[i][j]  
            affinity_matrix[i][:] /= row_sum  
    elif kernel_type == 'knn':  
        if knn_num_neighbors == None:  
            raise ValueError('You should input a k of knn kernel!')  
        for i in range(num_samples):  
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)  # return the indexes of data points
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors  
    else:  
        raise NameError('Not support kernel type! You can use knn or rbf!')  
      
    return affinity_matrix  
  
  
# label propagation  
def labelPropagation(Mat_Label, 
                     Mat_Unlabel, 
                     labels, 
                     kernel_type = 'rbf', 
                     rbf_sigma = 1.5,
                     knn_num_neighbors = 10, 
                     max_iter = 500, 
                     tol = 1e-3):  
    # initialize  
    num_label_samples = Mat_Label.shape[0]  
    num_unlabel_samples = Mat_Unlabel.shape[0]  
    num_samples = num_label_samples + num_unlabel_samples  
    labels_list = np.unique(labels)  
    num_classes = len(labels_list)  
    
    MatX = np.vstack((Mat_Label, Mat_Unlabel)) # all data points
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)  
    for i in range(num_label_samples):  
        clamp_data_label[i][labels[i]] = 1.0 # error here?
      
    label_function = np.zeros((num_samples, num_classes), np.float32)  
    label_function[0 : num_label_samples] = clamp_data_label  
    label_function[num_label_samples : num_samples] = -1  # all dummy values
      
    # graph construction  
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)  
      
    # start to propagation  
    iter = 0; pre_label_function = np.zeros((num_samples, num_classes), np.float32)  # all zeros
    changed = np.abs(pre_label_function - label_function).sum()  # label_function: current labels
    while iter < max_iter and changed > tol:  
        if iter % 1 == 0:  
            print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))
        pre_label_function = label_function  
        iter += 1  
          
        # propagation  
        label_function = np.dot(affinity_matrix, label_function)  
          
        # clamp  
        label_function[0 : num_label_samples] = clamp_data_label  
          
        # check converge  
        changed = np.abs(pre_label_function - label_function).sum()  
      
    # get terminate label of unlabeled data  
    unlabel_data_labels = np.zeros(num_unlabel_samples)  
    for i in range(num_unlabel_samples):  
        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])  
      
    return unlabel_data_labels  
#%%
#***************************************************************************  
#*   
#* Description: label propagation  
#* Author: Zou Xiaoyi (zouxy09@qq.com)  
#* Date:   2015-10-15  
#* HomePage: http://blog.csdn.net/zouxy09  
#*   
#**************************************************************************  
  
import time  
import math  
import numpy as np  
# from label_propagation import labelPropagation  
  
# show  
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):   
    import matplotlib.pyplot as plt   
      
    for i in range(Mat_Label.shape[0]):  
        if int(labels[i]) == 0:    
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')    
        elif int(labels[i]) == 1:    
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')  
        else:  
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')  
      
    for i in range(Mat_Unlabel.shape[0]):  
        if int(unlabel_data_labels[i]) == 0:    
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')    
        elif int(unlabel_data_labels[i]) == 1:    
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')  
        else:  
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')  
      
    plt.xlabel('X1'); plt.ylabel('X2')   
    plt.xlim(0.0, 12.)  
    plt.ylim(0.0, 12.)  
    plt.show()    
  
  
def loadCircleData(num_data):  
    center = np.array([5.0, 5.0])  
    radiu_inner = 2  
    radiu_outer = 4  
    num_inner = num_data / 3  
    num_outer = num_data - num_inner  
      
    data = []  
    theta = 0.0  
    for i in range(int(num_inner)):  
        pho = (theta % 360) * math.pi / 180  
        tmp = np.zeros(2, np.float32)  
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]  
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]  
        data.append(tmp)  
        theta += 2  
      
    theta = 0.0  
    for i in range(int(num_outer)):  
        pho = (theta % 360) * math.pi / 180  
        tmp = np.zeros(2, np.float32)  
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]  
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]  
        data.append(tmp)  
        theta += 1  
      
    Mat_Label = np.zeros((2, 2), np.float32)  
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])  
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])  
    labels = [0, 1]  
    Mat_Unlabel = np.vstack(data)  
    return Mat_Label, labels, Mat_Unlabel  
  
  
def loadBandData(num_unlabel_samples):  
    #Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])  
    #labels = [0, 1]  
    #Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])  
      
    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])  
    labels = [0, 1]  
    num_dim = Mat_Label.shape[1]  

    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)  

    x = int(num_unlabel_samples/2)
    Mat_Unlabel[:x, :] = (np.random.rand(x, num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[0]  

    Mat_Unlabel[x : num_unlabel_samples, :] = (np.random.rand(x, num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[1]  

    return Mat_Label, labels, Mat_Unlabel  
  
#%% ========== ========== ========== ========== ========== ==========
# main function  
if __name__ == "__main__":  
    num_unlabel_samples = 800  
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)  
    Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)  
      
    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be  
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that  
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be  
    ## 'knn' kernel is better!  
    #unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)  
    unlabel_data_labels = labelPropagation(Mat_Label, 
                                           Mat_Unlabel, 
                                           labels, 
                                           kernel_type = 'knn', 
                                           knn_num_neighbors = 10, 
                                           max_iter = 400)  # 400
    
    show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)  
#%% ========== ========== ========== ========== ========== ==========
num_unlabel_samples = 800
# Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)  
Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples) 

unlabel_data_labels = labelPropagation(Mat_Label, 
                                       Mat_Unlabel, 
                                       labels, 
                                       kernel_type = 'knn', 
                                       knn_num_neighbors = 10, 
                                       max_iter = 400)  # 400

show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)   
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
# Label Propagation learning a complex structure

# print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles
#%%
# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False) # X: coordinates at 2D plane [-1, 1]
# plt.scatter(X[:,0],X[:,1])
outer, inner = 0, 1
labels = np.full(n_samples, -1.) # original labels
labels[0] = outer # first point
labels[-1] = inner # last point

# #############################################################################
# Learn with LabelSpreading
label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

# #############################################################################
# Plot output labels
output_labels = label_spread.transduction_
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
            marker='s', lw=0, label="outer labeled", s=10)
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
            marker='s', lw=0, label='inner labeled', s=10)
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
            marker='.', label='unlabeled')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Raw data (2 classes=outer and inner)")

plt.subplot(1, 2, 2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
            marker='s', lw=0, s=10, label="outer learned")
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
            marker='s', lw=0, s=10, label="inner learned")
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
plt.show()

#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
# Label Propagation digits: Demonstrating performance

print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn import datasets
from sklearn.semi_supervised import label_propagation

from sklearn.metrics import confusion_matrix, classification_report

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:330]] # clip this amount of data: (330, 64)
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_total_samples = len(y) # we only use this amount of data 
n_labeled_points = 30

indices = np.arange(n_total_samples)

unlabeled_set = indices[n_labeled_points:]

# #############################################################################
# Shuffle everything around
y_train = np.copy(y)
y_train[unlabeled_set] = -1 # unlabeled class is -1

# #############################################################################
# Learn with LabelSpreading
lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]

cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
      (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

print(classification_report(true_labels, predicted_labels))

print("Confusion matrix")
print(cm)

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

# #############################################################################
# Pick the top 10 most uncertain labels
uncertainty_index = np.argsort(pred_entropies)[-10:]

# #############################################################################
# Plot
f = plt.figure(figsize=(7, 5))
for index, image_index in enumerate(uncertainty_index):
    image = images[image_index]

    sub = f.add_subplot(2, 5, index + 1)
    sub.imshow(image, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    sub.set_title('predict: %i\ntrue: %i' % (
        lp_model.transduction_[image_index], y[image_index]))

f.suptitle('Learning with small amount of labeled data')
plt.show()
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
# Label Propagation digits active learning

# print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 10
max_iterations = 5

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()

for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1

    lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(
        lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.in1d(uncertainty_index, unlabeled_indices)][:5]

    # keep track of indices that we get labels for
    delete_indices = np.array([])

    # for more than 5 iterations, visualize the gain only on the first 5
    if i < 5:
        f.text(.05, (1 - (i + 1) * .183),
               "model %d\n\nfit with\n%d labels" %
               ((i + 1), i * 5 + 10), size=10)
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        # for more than 5 iterations, visualize the gain only on the first 5
        if i < 5:
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
            sub.set_title("predict: %i\ntrue: %i" % (
                lp_model.transduction_[image_index], y[image_index]), size=10)
            sub.axis('off')

        # labeling 5 points, remote from labeled set
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)

f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
           "uncertain labels to learn with the next model.", y=1.15)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                    hspace=0.85)
plt.show()
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 














