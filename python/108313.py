# # CNN Based Image Segmentation of DICOM Images - Testing
# ### **```Author : BlackViper42```**
# ______
# 

# Overview
# ======
# 
# ------
#  - This notebook consists of HeadRest Segmentation of CT Scan DICOM images of head patients downloaded from `gemsvnc server-3.204.27.254` of anonymous patients. 
#  
#  
#  - Currently this notebook predicts the segmented image of original `input_image` and this can be further taken to get `output_image` without headrest by multiplying `predicted_outcome` (binary labeled image) with `input_image`.
#  
#  
#  - Prerequisites for this notebook:
#     - Graphics Card -------------------------------  NVIDIA  Quadro M5000 - 8GB 
#     - `CUDA` toolkit 7.0 or above -------------------  installed  CUDA-8.0
#     - `cuDNN` 5.1 or above --------------------------  installed  cuDNN-5.1 
#     - `tensorflow-gpu` library
#  
#  - [please go through this link for installation of tensorflow-gpu](https://www.tensorflow.org/install/install_linux)
#  - [Tensorflow implementation of Image Segmentation on Python](https://github.com/jakeret/tf_unet)
#  
#  
#  - If you want to run this notebook without GPUs then install `tensorflow` library and comment these lines:
#  
#  ```python
#  config = tf.ConfigProto()
#  config.gpu_options.allow_growth = True``` 
#   and change
#   ```python
#   tf.Session(config=config)```
#   by 
#   ```python
#   tf.Session()```
#  - In order to check whether GPU is correctly called by tensorflow run this line:
#  ```python
#  from tensorflow.python.client import device_lib
# local_device_protos = device_lib.list_local_devices()```
# and check the output you got on `terminal` where you initialized `Jupyter notebook`. If your output shows `(/gpu:0)` in last line then it is correct. If you are using Non-GPU tensorflow then your output should be `(/cpu:0)`.
# 
# 
# Project Description
# ======
# 
# ______
# 
# ### Data
#  - Input data consists of 2D dicom format images of ** 512*512 ** pixels of various patients. Input dataset reffered here as **`train_images_input`** contains 2490 images with grayscale values. 
#  - Labelled data consists of binary indicated images of **0s** and **1s** where **1s** tells presence of headrest and **0s** tells background without headrest.
# 
# ### Model
#  - Model built here is 7 layer Convolutional nueral networks with pooling and upsampling. Below is the structure of network:
#  
#  - Parameters which are used in this model:
#  
#  
# | Parameter        | Value           | 
# | ------------- |:-------------:| 
# | Filter size   | **`3*3`**   |
# | Pool size    | **`2*2`**   |
# | Zero Padding  | **`1`**  |
# | Padding    | **`VALID`**   |
# | Cost Function      | **`Softmax Cross Entropy`** | 
# | Optimizer      |  **`Momentum`**    |  
# | Epocs | **`150`**      | 
# | Training Iterations     | **`50`** |
# | Learning Rate        |  **`0.2`**  |
# | Decay Rate         |  **`0.95`**   |
# 
# 
# 
#  - Accuracy is calculated on pixel wise correct classification of an image. 
#  - Predictions after each epoc is stored in **`prediction_model_50_500_3`** folder.
#  - Model outputs are stored in pickel format in **`unet_trained_50_500_3`** folder.
# 

# ______
# ## ``` Importing Libraries```
# 

from __future__ import division, print_function,absolute_import, unicode_literals
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt                                    ##...Plotting libraries
import matplotlib                                                  
import numpy as np                                                 ##...for mathematical operations
import glob                                                        ##...for importing directories and paths
from PIL import Image                                              ##...for saving images
import os
import shutil
from glob import glob
from collections import OrderedDict
import logging                                                     ##...for making logs on progress based on real time
import tensorflow as tf                                            ##...for building Convolutional Neural Networks 
import dicom
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from matplotlib.patches import Ellipse
from skimage import io, color, measure, draw, img_as_bool
from scipy import optimize
import pandas as pd
from skimage import color
import scipy.misc
import sys
import zipfile


from tensorflow.python.client import device_lib                    ##...to check whether GPU is called properly or not
local_device_protos = device_lib.list_local_devices()


config = tf.ConfigProto()                                          ##...to allow bfc::Allocator provide more than default 
config.gpu_options.allow_growth = True                             ##...memory  to GPU


# ______
# ## ``` Importing Data :: Preprocessing```
# 

# > Importing Data from the given **```input_folder```**
# 

input_path = "/home/ctuser/images/input/input.zip"
data_path = "/home/ctuser/images/input/"
output_data_path = "/home/ctuser/images/output/"
a=[]
z = zipfile.ZipFile(input_path, "r")
z.extractall(data_path)
for filename in z.namelist():
    a.append(filename.split("/")[1] +"-out")





# > Loading CT Scan Images and calculate HU units for each pixels
# 
# > **```load_scan```** : loading all the image slices for each patient folder
# 
# > **```get_pixels_hu```** : Storing HU_units pixel array in **```numpy```** format for image processing
# 

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in z.namelist()]
    #a=[]
    #for s in slices:
    #    a.append(s.InstanceNumber)
    #slices.sort(key = lambda x: int(x.InstanceNumber))
    #try:
    #    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    #except:
    #    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    #for s in slices:
    #    s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#def store_dcm_format(output_data_path, patient_out, out_test, a):
#    #nx = out_test.shape[1]
#    #ny = out_test.shape[2]
#    #out_test = out_test.reshape((-1,nx,ny))
#    out_test = out_test*4000.
#    out_test = out_test.astype(np.int16)
#    out_test_zip = zipfile.ZipFile(output_data_path+"out_test.zip", 'w')
#    for i in range(0,len(patient_out)):
#        #print(np.unique(patient_out[0].pixel_array))
#        #outo = out_test[i,...,0].flat
#        #print(np.unique(out_test[i,...,0]))
#        patient_out[i].pixel_array = out_test[i,...,0]
#        #print(np.unique(patient_out[0].pixel_array))
#        #patient_out[i].pixel_array
#        patient_out[i].PixelData = patient_out[i].pixel_array.tostring()
#        storing_path = a[i]
#        #patient_out[i].save_as(output_data_path+"/"+storing_path)
#        dicom.write_file(output_data_path+"/"+storing_path, patient_out[i])    
#        #out_test_zip.write(storing_path, compress_type=zipfile.ZIP_DEFLATED)
#    
#    out_test_zip.close()
        

id=1000000
patient = load_scan(data_path)
patient_out = load_scan(data_path)
imgs = get_pixels_hu(patient)





offset = np.ones_like(imgs,dtype=np.float32)
imgs = imgs.astype("float32")
offset = offset*1024.
imgs+=offset
offset=1


# > Creating Class - **```BaseDataProvider```** for importing data and making **```4D Tensors```** and giving batch-wise output when called.
# 
# > Parameter here is **```n```** - Number of randomly sample data you want to call in **```4D Tensor```** format.
# 

class BaseDataProvider(object):
    """
    This class is used to import, preprocessing of data
    before feeding into Convolutional neural networks.
    It also create labels into same fashion.
    """
    
    #channels = 1
    #n_class = 2
    def __init__(self,data,a_min=None,a_max=None,channels =1,n_class =2):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_max is not None else np.inf
        self.data = data
        self.file_count = data.shape[0]
        self.n_class=n_class
        self.channels = channels
    
    def _next_data(self,i):
        #idx = np.random.choice(self.file_count)
        return self.data[i]
    
    def _load_data_and_label(self,i):
        data = self._next_data(i)
        
        train_data, min_data_i, max_data_i = self._process_data(data)
        
        train_data = self._postprocess_data(train_data)
        
        nx = data.shape[1]
        ny = data.shape[0]
        
        return train_data.reshape(1, ny, nx, self.channels), min_data_i, max_data_i
    
    def _process_data(self, data):
        data = np.clip(data, self.a_min, self.a_max)
        min_val = np.amin(data)
        max_val = np.amax(data)
        data = data-np.amin(data)    
        data = data/np.amax(data)
        return data, min_val, max_val
    
    def _postprocess_data(self,data):
        """
        Post processing can be done to make it more easier 
        for CNN to work and give better accuracy.
        
        """
        return data
    
    def __call__(self,n):
        i=0
        train_data, min_data_i, max_data_i = self._load_data_and_label(i)
        nx = train_data.shape[1]
        ny = train_data.shape[2]
        
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, 2))
        X[0] = train_data
        Y[0, 0] = min_data_i
        Y[0, 1] = max_data_i
        for i in range(1,n):
            train_data, min_data_i, max_data_i = self._load_data_and_label(i)
            X[i] = train_data
            Y[i, 0] = min_data_i
            Y[i, 1] = max_data_i
            
        
        return X, Y


mydata = BaseDataProvider(data=imgs,channels=1,n_class=2)


m = imgs.shape[0]


X_test, minmax = mydata(m)


# ______
# ## **```Building Convolutional Networks```**
# 

# > Defining some functions which we will use in later part of notebook:
#   - **```plot_prediction```** : takes ```test_data, labels, predictions``` and plot the images; can be used to save the output as well.
#   - **```crop_to_shape```** : crop ```initial_tensor``` to ```final_tensor``` shape
#   - **```to_rgb```** : convert given image to RGB format; not used here
#   - **```error_rate```** : error of prediction in **%** format
#   - **```get_image_summary```** : gives image summary 
#   - **```combine_img_prediction ```** : Combines the data, grouth thruth and the prediction into one rgb image; amethod to visualize
#   - **```save_image ```** : to save the image
# 

def plot_prediction(x_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 2, figsize=(20,15), sharey=True, sharex=True)
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i,...,0])
        plt.colorbar(cax, ax=ax[i,0])
        #cax = ax[i, 1].imshow(y_test[i, ..., 1])
        #plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        #pred -= np.amin(pred)                           ## recheck this :: might create some errors later.
        #pred /= np.amax(pred)
        cax = ax[i, 1].imshow(pred)
        plt.colorbar(cax, ax=ax[i,1])
        if i==0:
            ax[i, 0].set_title("x")
            #ax[i, 1].set_title("y")
            ax[i, 1].set_title("pred")
    #fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
    plt.show()


def storing_dicom(x_test, prediction):
    out_test = np.zeros_like(x_test,dtype="float32")
    test_size = x_test.shape[0]
    for i in range(test_size):
        ini = x_test[i,...,0]
        pred = prediction[i, ..., 1]
        mask = np.zeros_like(pred,dtype="float32")
        mask_2 = np.ones_like(pred,dtype="float32")
        mask[pred>=0.5]=1.0
        dilation = morphology.dilation(mask,np.ones([4,4]))
        mask = mask_2-dilation
        out = mask*ini
        out_test[i,...,0]=out
    #for i in range(test_size):
    #    pred = out_test[i,...,0]
    #    path = "%s/%s.jpg"%(output_path, "slice_%s"%i)
    #    scipy.misc.imsave(path, pred)
    return out_test       


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if data.shape[1] == shape[1]:
        return data
    else:
        return data[:, offset0:(-offset0), offset1:(-offset1)]


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
                    (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img


def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)


# > **```layers initialization functions```**
# 

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  ## to get log of real time and progress


# ______
# ## **```Model Training```**
# 

# > **``` Model_3 layer ```** : layers = 7 with convolutional varying : 2,4,8 each 3,3,1 layers
# 

def create_conv_net_3(x, keep_prob, channels, n_class, layers=7, features_root=16, filter_size=3, pool_size=2, summaries=False):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size},pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights1 = []
    weights2= []
    weights3 = []
    biases1 = []
    biases2 = []
    biases3 = []
    convs1 = []
    convs2 = []
    convs3 = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    paddings=[[0,0],[1,1],[1,1],[0,0]]
    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        in_node = tf.pad(in_node,paddings,"CONSTANT")
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        tmp_h_conv = tf.pad(tmp_h_conv,paddings,"CONSTANT")
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        if layer>=3:
            tmp_h_conv2 = tf.nn.relu(conv2 + b2)
        else:
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        ####
        if layer>=3:
            w3 = weight_variable([filter_size, filter_size, features, features], stddev)
            b3 = bias_variable([features])
            w4 = weight_variable([filter_size, filter_size, features, features], stddev)
            b4 = bias_variable([features])
            tmp_h_conv2 = tf.pad(tmp_h_conv2,paddings,"CONSTANT")
            conv3 = conv2d(tmp_h_conv2, w3, keep_prob)
            tmp_h_conv3 = tf.nn.relu(conv3 +b3)
            tmp_h_conv3 = tf.pad(tmp_h_conv3,paddings,"CONSTANT")
            conv4 = conv2d(tmp_h_conv3, w4, keep_prob)
            if layer>=6:
                tmp_h_conv4 = tf.nn.relu(conv4 + b4)
            else:
                dw_h_convs[layer] = tf.nn.relu(conv4 + b4)
        ####
        ####
        if layer>=6:
            w5 = weight_variable([filter_size, filter_size, features, features], stddev)
            b5 = bias_variable([features])
            w6 = weight_variable([filter_size, filter_size, features, features], stddev)
            b6 = bias_variable([features])
            w7 = weight_variable([filter_size, filter_size, features, features], stddev)
            b7 = bias_variable([features])
            w8 = weight_variable([filter_size, filter_size, features, features], stddev)
            b8 = bias_variable([features])
            tmp_h_conv4 = tf.pad(tmp_h_conv4,paddings,"CONSTANT")
            conv5 = conv2d(tmp_h_conv4, w5, keep_prob)
            tmp_h_conv5 = tf.nn.relu(conv5 +b5)
            tmp_h_conv5 = tf.pad(tmp_h_conv5,paddings,"CONSTANT")
            conv6 = conv2d(tmp_h_conv5, w6, keep_prob)
            tmp_h_conv6 = tf.nn.relu(conv6 +b6)
            tmp_h_conv6 = tf.pad(tmp_h_conv6,paddings,"CONSTANT")
            conv7 = conv2d(tmp_h_conv6, w7, keep_prob)
            tmp_h_conv7 = tf.nn.relu(conv7 +b7)
            tmp_h_conv7 = tf.pad(tmp_h_conv7,paddings,"CONSTANT")
            conv8 = conv2d(tmp_h_conv7, w8, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv8 + b8)
        ####
        if layer<3:
            weights1.append((w1, w2))
            biases1.append((b1, b2))
            convs1.append((conv1, conv2))
        ####
        if layer>=3 and layer<6:
            weights2.append((w1, w2, w3, w4))
            biases2.append((b1, b2, b3, b4))
            convs2.append((conv1, conv2, conv3, conv4))
        ####
        ####
        if layer>=6:
            weights3.append((w1, w2, w3, w4, w5, w6, w7, w8))
            biases3.append((b1, b2, b3, b4, b5, b6, b7,  b8))
            convs3.append((conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8))
        ####
        
        size -= 4
        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        h_deconv_concat = tf.pad(h_deconv_concat,paddings,"CONSTANT")
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        h_conv = tf.pad(h_conv,paddings,"CONSTANT")
        conv2 = conv2d(h_conv, w2, keep_prob)
        if layer>=3:
            w3 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
            b3 = bias_variable([features//2])
            w4 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
            b4 = bias_variable([features//2])
            h_conv2 = tf.nn.relu(conv2 + b2)
            h_conv2 = tf.pad(h_conv2,paddings,"CONSTANT")
            conv3 = conv2d(h_conv2, w3, keep_prob)
            h_conv3 = tf.nn.relu(conv3 + b3)
            h_conv3 = tf.pad(h_conv3,paddings,"CONSTANT")
            conv4 = conv2d(h_conv3, w4, keep_prob)
            in_node = tf.nn.relu(conv4 + b4)
            up_h_convs[layer] = in_node
        else:
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

        ####
        if layer<3:
            weights1.append((w1, w2))
            biases1.append((b1, b2))
            convs1.append((conv1, conv2))
        ####
        if layer>=3:
            weights2.append((w1, w2, w3, w4))
            biases2.append((b1, b2, b3, b4))
            convs2.append((conv1, conv2, conv3, conv4))
        
        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map
    
    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
            
        for k in pools.keys():
            tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
        
        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
            
        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])
            
    variables = []
    for w1,w2 in weights1:
        variables.append(w1)
        variables.append(w2)
    
    for w1,w2,w3,w4 in weights2:
        variables.append(w1)
        variables.append(w2)
        variables.append(w3)
        variables.append(w4)
    
    for w1,w2,w3,w4,w5,w6,w7,w8 in weights3:
        variables.append(w1)
        variables.append(w2)
        variables.append(w3)
        variables.append(w4)
        variables.append(w5)
        variables.append(w6)
        variables.append(w7)
        variables.append(w8)
    
    for b1,b2 in biases1:
        variables.append(b1)
        variables.append(b2)
        
    for b1,b2,b3,b4 in biases2:
        variables.append(b1)
        variables.append(b2)
        variables.append(b3)
        variables.append(b4)
    
    for b1,b2,b3,b4,b5,b6,b7,b8 in biases3:
        variables.append(b1)
        variables.append(b2)
        variables.append(b3)
        variables.append(b4)
        variables.append(b5)
        variables.append(b6)
        variables.append(b7)
        variables.append(b8)

    
    return output_map, variables, int(in_size - size)


class Unet(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=1, n_class=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        logits, self.variables, self.offset = create_conv_net_3(self.x, self.keep_prob, channels, n_class, **kwargs)
        
        self.cost = self._get_cost(logits, cost, cost_kwargs)
        
        self.gradients_node = tf.gradients(self.cost, self.variables)
         
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))
        
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(flat_logits, flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
            
        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


net = Unet(channels=mydata.channels, n_class=mydata.n_class, layers=7, features_root=16)


# >  Making predictions on **```K```** data samples using saved model
# 

prediction = net.predict("/home/ctuser/Desktop/Image_Seg/unet_trained_50_500_3/model_50_500_3.cpkt", X_test)


out_test = storing_dicom(x_test=X_test,prediction=prediction)


n = out_test.shape[0]
for i in range(0,n):
    out_test[i,...,0] = out_test[i,...,0]*minmax[i,1]
    out_test[i,...,0] = out_test[i,...,0]+minmax[i,0]

out_test = out_test.astype(np.int16)


np.unique(out_test[0,...,0])


def store_in_input(path,out_test):
    out_test_zip = zipfile.ZipFile(output_data_path+"out_test.zip", 'w')
    slices = [dicom.read_file(path + '/' + s) for s in z.namelist()]
    #slices.sort(key = lambda x: int(x.InstanceNumber))
    for i in range(0,len(slices)):
        s = slices[i]
        storing_path = a[i]
        out_test_flat = out_test[i].flat
        #for n,val in enumerate(s.pixel_array.flat):
        s.pixel_array.flat[:]=out_test_flat[:]
        #s.pixel_array = copy(out_test[i,...,0])
        s.PixelData = s.pixel_array.tostring()
        s.save_as(output_data_path+'/'+a[i])
        out_test_zip.write(output_data_path+storing_path, compress_type=zipfile.ZIP_DEFLATED)
        print(i)
    out_test_zip.close()


store_in_input(data_path,out_test)














# # Author : Harshit Saxena
# _______
# 

# ### Importing Libraries
# 

import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from matplotlib.patches import Ellipse
from skimage import io, color, measure, draw, img_as_bool
from scipy import optimize
import pandas as pd


# ### Importing Data from a folder which contains all DCM files
# 

data_path = "/home/ctuser/myData/s1735"
output_path = working_path = "/home/ctuser/Desktop/Image_Seg/"
g = glob(data_path + "/*")

print("Total no. of CT images are: %d \nFirst 5 elements:" %len(g))
print '\n'.join(g[:5])


# #### Loading CT Scan Images and calculate HU units for each pixels
# 

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=1
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)


# #### saving output images in Hounsfield Units
# 

np.save(output_path + "fullimages_%d.npy" % (id), imgs)


file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))


def sample_stack(stack, rows=8, cols=4):
    fig,ax = plt.subplots(rows,cols,figsize=[12,16])
    for i in range(rows*cols):
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % i)
        ax[int(i/cols),int(i % cols)].imshow(stack[i],cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()

sample_stack(imgs_to_process)


# #### A Static Elliptic filter to remove head rest
# 

def filter_ellipse(imgs_array):
    for a in imgs_array:
        o,p=256,246
        arr_x,arr_y=512.0,512.0
        y,x = np.ogrid[0.0:512.0,0.0:512.0]
        mask = ((x-256.0)*(x-256.0))/(190.0*190.0) + ((y-246.0)*(y-246.0))/(215.0*215.0) > 1.0
        a[mask] = -1000
    return(imgs_array,mask)
    
ad = filter_ellipse(imgs_to_process)


# #### A Linear Elliptic filter to remove head rest
# 

def filter_linear(imgs_array):
    for a in imgs_array:
        for j in range(100,450):
            for i in range(400,512):
                c=np.mean(np.array(a[j,i-20:i]))
                if c<=-999:
                    a[j,i:512]=-1000
        for j in range(100,450):
            for i in range(0,100):
                c=np.mean(np.array(a[j,i:i+20]))
                if c<=-999:
                    a[j,0:i]=-1000
            
        for i in range(0,512):
            for j in range(450,512):
                c = np.mean(np.array(a[j:j+20,i]))
                if c<=-999:
                    a[j:512,i]=-1000
                else:
                    d = np.mean(np.array(a[j-20:j,i]))
                    if d<=-999:
                        a[j:512,i]=-1000
    return(imgs_array)
imgs_to_process_1inear_filter = filter_linear(imgs_to_process)


def sample_stack(stack, rows=8, cols=4):
    fig,ax = plt.subplots(rows,cols,figsize=[12,16])
    for i in range(rows*cols):
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % i)
        ax[int(i/cols),int(i % cols)].imshow(stack[i],cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
        ellipse = Ellipse(xy=(256,246), width=380, height=430, edgecolor='r',fill=False, lw=1)
        ax[int(i/cols),int(i % cols)].add_patch(ellipse)
    plt.show()

sample_stack(imgs_to_process)


a = imgs_to_process[4]


fig,ax = plt.subplots(1)
ax.imshow(a,cmap=plt.cm.gray)
ellipse = Ellipse(xy=(256,246), width=380, height=430, edgecolor='r',fill=False, lw=1)
ax.add_patch(ellipse)
plt.show()


# #### Optimal Ellipse fitting 
# 

a=imgs_to_process[16]


image = np.int16(dilated)
regions = measure.regionprops(label_image=image)
bubble = regions[0]

r, c = bubble.centroid
r_radius = bubble.major_axis_length / 2.
c_radius = bubble.minor_axis_length / 2.

def cost(params):
    r,c,r_radius,c_radius = params
    coords = draw.ellipse(r, c, r_radius,c_radius, shape=image.shape)
    template = np.zeros_like(image)
    template[coords] = 0
    return -np.sum(template == image)

r,c,r_radius,c_radius = optimize.fmin(cost, (r,c,r_radius,c_radius))

import matplotlib.pyplot as plt

f, ax = plt.subplots()
ellipse = Ellipse(xy=(r,c), width=2*c_radius, height=2*r_radius,fill=True,lw=5,edgecolor='r')
ax.imshow(image, cmap='gray', interpolation='nearest')
ax.add_artist(ellipse)
plt.show()


fig,ax = plt.subplots(1)
ax.imshow(a,cmap=plt.cm.gray)
plt.show()


image = a
regions = measure.regionprops(image)
bubble = regions[0]

y0, x0 = bubble.centroid
r = bubble.major_axis_length / 2.

def cost(params):
    x0, y0, r = params
    coords = draw.circle(y0, x0, r, shape=image.shape)
    template = np.zeros_like(image)
    template[coords] = 1
    return -np.sum(template == image)

x0, y0, r = optimize.fmin(cost, (x0, y0, r))

import matplotlib.pyplot as plt

f, ax = plt.subplots()
circle = plt.Circle((x0, y0), r)
ax.imshow(image, cmap='gray', interpolation='nearest')
ax.add_artist(circle)
plt.show()


id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    print(spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print "Shape before resampling\t", imgs_to_process.shape
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print "Shape after resampling\t", imgs_after_resamp.shape


def make_mesh(image, threshold=-300, step_size=1):

    print "Transposing surface"
    p = image.transpose(2,1,0)
    
    print "Calculating surface"
    verts, faces, norm, val = measure.marching_cubes(image, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print "Drawing"
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    f = FF()
    fig = f.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    print "Drawing"
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()


v, f = make_mesh(imgs_after_resamp,700)
plt_3d(v, f)


v, f = make_mesh(imgs_after_resamp, 350, 2)
plotly_3d(v, f)


#Standardize the pixel values
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))
    return dilation


img = imgs_to_process[2]
dilated = make_lungmask(img, display=True)


#Standardize the pixel values
def min_distance(lst):
    lst = sorted(lst)
    index=-1
    dst = max(lst)-min(lst)
    for i in range(len(lst)-1):
        if lst[i+1] - lst[i] < distance:
            distance = lst[i+1] - lst[i] 
            index = i
    for i in range(len(lst)-1):
        if lst[i+1] - lst[i] == distance:
            id_1 = lst[i]
            id_2 = lst[i+1]
            print lst[i],lst[i+1]
    
def make_lungmask(img, display=False):
    img_2 = img
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    #img[img==max]=mean
    #img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(img,[np.prod(img.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,0,1)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    #eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    #dilation = morphology.dilation(thresh_img,np.ones([1,1]))

    labels = measure.label(thresh_img,neighbors=8,connectivity=2) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    print(label_vals)
    print('\n -------------------------------------------')
    labels_flat = labels.reshape(np.prod(labels.shape),1)
    labels_flat = pd.DataFrame(labels_flat)
    labels_flat.columns=['label']
    #print(labels_flat)
    #print(labels_flat.dtype)
    img_2_flat = np.array(np.reshape(img_2,[np.prod(img_2.shape),1]))
    img_2_flat = pd.DataFrame(img_2_flat)
    img_2_flat.columns = ['HU_Value']
    #print(img_2_flat)
    df=pd.DataFrame.join(labels_flat,img_2_flat)
    #print(df.head())
    std = pd.DataFrame(df.groupby('label').std())
    print (std)
    print (df.groupby('label').mean())
    
    
    
    
    
    
    
    
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 1
    mask[labels==0]= 0
    #print(good_labels)
    
    for N in labels:
        mask = mask + np.where(labels==N,1,0)
    #print(np.unique(mask))
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(thresh_img, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
       
        plt.show()
    return dilation


img = imgs_to_process[31]
#for img in imgs_to_process:
dilated = make_lungmask(img, display=True)


