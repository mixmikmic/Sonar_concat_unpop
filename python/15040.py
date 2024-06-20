# # Multilabel classification on PASCAL using python data-layers
# 

# In this tutorial we will do multilabel classification on PASCAL VOC 2012.
# 
# Multilabel classification is a generalization of multiclass classification, where each instance (image) can belong to many classes. For example, an image may both belong to a "beach" category and a "vacation pictures" category. In multiclass classification, on the other hand, each image belongs to a single class.
# 
# Caffe supports multilabel classification through the SigmoidCrossEntropyLoss layer, and we will load data using a Python data layer. Data could also be provided through HDF5 or LMDB data layers, but the python data layer provides endless flexibility, so that's what we will use.
# 

# ### 1. Preliminaries
# 
# * First, make sure you compile caffe using
# WITH_PYTHON_LAYER := 1
# 
# * Second, download PASCAL VOC 2012. It's available here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
# 
# * Third, import modules:
# 

import sys 
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from copy import copy

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (6, 6)

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

import tools #this contains some tools that we need


# * Fourth, set data directories and initialize caffe
# 

# set data root directory, e.g:
pascal_root = osp.join(caffe_root, 'data/pascal/VOC2012')

# these are the PASCAL classes, we'll need them later.
classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

# make sure we have the caffenet weight downloaded.
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    get_ipython().system('../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)


# ### 2. Define network prototxts
# 
# * Let's start by defining the nets using caffe.NetSpec. Note how we used the SigmoidCrossEntropyLoss layer. This is the right loss for multilabel classification. Also note how the data layer is defined.
# 

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=20)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())


# ### 3. Write nets and solver files
# 
# * Now we can crete net and solver prototxts. For the solver, we use the CaffeSolver class from the "tools" module
# 

workdir = './pascal_multilabel_with_datalayer'
if not os.path.isdir(workdir):
    os.makedirs(workdir)

solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
solverprototxt.sp['display'] = "1"
solverprototxt.sp['base_lr'] = "0.0001"
solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

# write train net.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train', pascal_root = pascal_root)
    f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))

# write validation net.
with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val', pascal_root = pascal_root)
    f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))


# * This net uses a python datalayer: 'PascalMultilabelDataLayerSync', which is defined in './pycaffe/layers/pascal_multilabel_datalayers.py'. 
# 
# * Take a look at the code. It's quite straight-forward, and gives you full control over data and labels.
# 
# * Now we can load the caffe solver as usual.
# 

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
solver.test_nets[0].share_with(solver.net)
solver.step(1)


# * Let's check the data we have loaded.
# 

transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
image_index = 0 # First image in the batch.
plt.figure()
plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
plt.title('GT: {}'.format(classes[np.where(gtlist)]))
plt.axis('off');


# * NOTE: we are readin the image from the data layer, so the resolution is lower than the original PASCAL image.
# 

# ### 4. Train a net.
# 
# * Let's train the net. First, though, we need some way to measure the accuracy. Hamming distance is commonly used in multilabel problems. We also need a simple test loop. Let's write that down. 
# 

def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


# * Alright, now let's train for a while
# 

for itt in range(6):
    solver.step(100)
    print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))


# * Great, the accuracy is increasing, and it seems to converge rather quickly. It may seem strange that it starts off so high but it is because the ground truth is sparse. There are 20 classes in PASCAL, and usually only one or two is present. So predicting all zeros yields rather high accuracy. Let's check to make sure.
# 

def check_baseline_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, len(gts)))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 5823/128))


# ### 6. Look at some prediction results
# 

test_net = solver.test_nets[0]
for image_index in range(5):
    plt.figure()
    plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
    estlist = test_net.blobs['score'].data[image_index, ...] > 0
    plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
    plt.axis('off')


# # Brewing Logistic Regression then Going Deeper
# 
# While Caffe is made for deep networks it can likewise represent "shallow" models like logistic regression for classification. We'll do simple logistic regression on synthetic data that we'll generate and save to HDF5 to feed vectors to Caffe. Once that model is done, we'll add layers to improve accuracy. That's what Caffe is about: define a model, experiment, and then deploy.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
os.chdir('..')

import sys
sys.path.insert(0, './python')
import caffe


import os
import h5py
import shutil
import tempfile

import sklearn
import sklearn.datasets
import sklearn.linear_model

import pandas as pd


# Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative features and 2 noise features.
# 

X, y = sklearn.datasets.make_classification(
    n_samples=10000, n_features=4, n_redundant=0, n_informative=2, 
    n_clusters_per_class=2, hypercube=False, random_state=0
)

# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

# Visualize sample of the data
ind = np.random.permutation(X.shape[0])[:1000]
df = pd.DataFrame(X[ind])
_ = pd.scatter_matrix(df, figsize=(9, 9), diagonal='kde', marker='o', s=40, alpha=.4, c=y[ind])


# Learn and evaluate scikit-learn's logistic regression with stochastic gradient descent (SGD) training. Time and check the classifier's accuracy.
# 

get_ipython().run_cell_magic('timeit', '', "# Train and test the scikit-learn SGD logistic regression.\nclf = sklearn.linear_model.SGDClassifier(\n    loss='log', n_iter=1000, penalty='l2', alpha=5e-4, class_weight='auto')\n\nclf.fit(X, y)\nyt_pred = clf.predict(Xt)\nprint('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))")


# Save the dataset to HDF5 for loading in Caffe.
# 

# Write out the data to HDF5 files in a temp directory.
# This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
dirname = os.path.abspath('./examples/hdf5_classification/data')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname, 'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')


# Let's define logistic regression in Caffe through Python net specification. This is a quick and natural way to define nets that sidesteps manually editing the protobuf model.
# 

from caffe import layers as L
from caffe import params as P

def logreg(hdf5, batch_size):
    # logistic regression: data, matrix multiplication, and 2-class softmax loss
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip1, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()

train_net_path = 'examples/hdf5_classification/logreg_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(logreg('examples/hdf5_classification/data/train.txt', 10)))

test_net_path = 'examples/hdf5_classification/logreg_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(logreg('examples/hdf5_classification/data/test.txt', 10)))


# Now, we'll define our "solver" which trains the network by specifying the locations of the train and test nets we defined above, as well as setting values for various parameters used for learning, display, and "snapshotting".
# 

from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 1000  # Test after every 1000 training iterations.
    s.test_iter.append(250) # Test 250 "batches" each time we test.

    s.max_iter = 10000      # # of times to update the net (training iterations)

    # Set the initial learning rate for stochastic gradient descent (SGD).
    s.base_lr = 0.01        

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000

    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = 10000
    s.snapshot_prefix = 'examples/hdf5_classification/data/train'

    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    return s

solver_path = 'examples/hdf5_classification/logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))


# Time to learn and evaluate our Caffeinated logistic regression in Python.
# 

get_ipython().run_cell_magic('timeit', '', 'caffe.set_mode_cpu()\nsolver = caffe.get_solver(solver_path)\nsolver.solve()\n\naccuracy = 0\nbatch_size = solver.test_nets[0].blobs[\'data\'].num\ntest_iters = int(len(Xt) / batch_size)\nfor i in range(test_iters):\n    solver.test_nets[0].forward()\n    accuracy += solver.test_nets[0].blobs[\'accuracy\'].data\naccuracy /= test_iters\n\nprint("Accuracy: {:.3f}".format(accuracy))')


# Do the same through the command line interface for detailed output on the model and solving.
# 

get_ipython().system('./build/tools/caffe train -solver examples/hdf5_classification/logreg_solver.prototxt')


# If you look at output or the `logreg_auto_train.prototxt`, you'll see that the model is simple logistic regression.
# We can make it a little more advanced by introducing a non-linearity between weights that take the input and weights that give the output -- now we have a two-layer network.
# That network is given in `nonlinear_auto_train.prototxt`, and that's the only change made in `nonlinear_logreg_solver.prototxt` which we will now use.
# 
# The final accuracy of the new network should be higher than logistic regression!
# 

from caffe import layers as L
from caffe import params as P

def nonlinear_net(hdf5, batch_size):
    # one small nonlinearity, one leap for model kind
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    # define a hidden layer of dimension 40
    n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
    # transform the output through the ReLU (rectified linear) non-linearity
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # score the (now non-linear) features
    n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
    # same accuracy and loss as before
    n.accuracy = L.Accuracy(n.ip2, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

train_net_path = 'examples/hdf5_classification/nonlinear_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(nonlinear_net('examples/hdf5_classification/data/train.txt', 10)))

test_net_path = 'examples/hdf5_classification/nonlinear_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(nonlinear_net('examples/hdf5_classification/data/test.txt', 10)))

solver_path = 'examples/hdf5_classification/nonlinear_logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))


get_ipython().run_cell_magic('timeit', '', 'caffe.set_mode_cpu()\nsolver = caffe.get_solver(solver_path)\nsolver.solve()\n\naccuracy = 0\nbatch_size = solver.test_nets[0].blobs[\'data\'].num\ntest_iters = int(len(Xt) / batch_size)\nfor i in range(test_iters):\n    solver.test_nets[0].forward()\n    accuracy += solver.test_nets[0].blobs[\'accuracy\'].data\naccuracy /= test_iters\n\nprint("Accuracy: {:.3f}".format(accuracy))')


# Do the same through the command line interface for detailed output on the model and solving.
# 

get_ipython().system('./build/tools/caffe train -solver examples/hdf5_classification/nonlinear_logreg_solver.prototxt')


# Clean up (comment this out if you want to examine the hdf5_classification/data directory).
shutil.rmtree(dirname)


