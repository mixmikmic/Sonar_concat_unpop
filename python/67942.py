# # Adding visualization to our network
# 
# When training real models, it's crucial to see what's happening as the network learns. If you're using TensorFlow or Keras with the TF backend, you can use a visualization tool called TensorBoard. This is a basic demo...
# 

# for DSX, need to switch to the right directory. Detect using path name.
s = get_ipython().magic('pwd')
if s.startswith('/gpfs'):
    get_ipython().magic('cd ~/deep-learning-workshop/')


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# for making plots prettier
import seaborn as sns 
sns.set_style('white')


from __future__ import print_function
np.random.seed(1337)  # for reproducibility


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


import notmnist
notmnist_path = "~/data/notmnist/notMNIST.pickle"


from display import visualize_keras_model, plot_training_curves


batch_size = 128
nb_classes = 10
nb_epoch = 10


# the data, shuffled and split between train, validation, and test sets
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notmnist.load_data(notmnist_path)


# Reshape inputs to be flat.
# Convert labels to 1-hot encoding.
# 

x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


model = Sequential()
model.add(Dense(128, input_shape=(784,), name="hidden"))
model.add(Activation('relu', name="ReLU"))
model.add(Dense(10, name="output"))
model.add(Activation('softmax', name="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# # Understanding the computation graph
# 
# Keras has built-in ways to view network. Let's look at them again.
# 

model.summary()


# The `summary()` lets us check that the tensors flowing through the network have the expected shape.
# 
# The graphical visualization below is useful for more complex architectures. Here, it's just pretty :)
# 

visualize_keras_model(model)


# # Adding TensorBoard
# 
# (Ref: https://keras.io/callbacks/#tensorboard)
# 
# To use TensorBoard in Keras, we need to tell the optimizer to log info as the network learns.
# 

from keras.callbacks import TensorBoard

def make_tb_callback(run):
    """
    Make a callback function to be called during training.
    
    Args:
        run: folder name to save log in. 
    
    Made this a function since we need to recreate it when
    resetting the session. 
    (See https://github.com/fchollet/keras/issues/4499)
    """
    return TensorBoard(
            # where to save log file
            log_dir='./graph-tb-demo/' + run,
            # how often (in epochs) to compute activation histograms
            # (more frequently slows down training)
            histogram_freq=1, 
            # whether to visualize the network graph.
            # This now works reasonably in Keras 2.01!
            write_graph=True,
            # if true, write layer weights as images
            write_images=False)

tb_callback = make_tb_callback('1')


# and add it to our model.fit call
history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid),
                    callbacks=[tb_callback])


# Let's look at our manually visualized learning curves first
plot_training_curves(history.history);


# # TensorBoard visualizations
# 
# During (or after) running the above, we can run
# 
# ```tensorboard --logdir=graph-tb-demo/```
# 
# And then open 0.0.0.0:6006 in a browser to take a look at the results. We can see learning curves in the Scalars tab:
# 
# ## Accuracy:
# 
# <img src="img/tb-demo-acc.png" width="400">
# 
# ## Loss:
# <img src="img/tb-demo-loss.png" width="400">
# 
# ## Validation accuracy
# <img src="img/tb-demo-val-acc.png" width="400">
# 
# Much the same -- not very interesting so far. But it gets better -- we can e.g. look at histograms of our variables over time. Here's the output distribution: 
# 
# <img src="img/tb-demo-output-hist.png" width="400">
# 
# We can see that the distribution is shifting to lower values and is getting wider over time.
# 
# ## Graph
# 
# We can also see a much more detailed view of the computation graph. Here's is a partial screenshot:
# 
# <img src="img/tb-demo-graph.png" width="400">
# 
# (Read the TensorBoard docs for many more details).
# 

# We'll use TensorBoard to eval our models later in the workshop.
# 

# Let's try our final model with more layers and dropout.
# 

# Uncomment if getting a "Invalid argument: You must feed a value
# for placeholder tensor ..." when rerunning training. 
# https://github.com/fchollet/keras/issues/4499
from keras.layers.core import K
K.clear_session() 
### 

model3 = Sequential()
model3.add(Dense(512, input_shape=(784,), name="hidden1"))
model3.add(Activation('relu', name="ReLU1"))
model3.add(Dropout(0.5))
model3.add(Dense(512, input_shape=(784,), name="hidden2"))
model3.add(Activation('relu', name="ReLU2"))
model3.add(Dropout(0.5))
model3.add(Dense(10, name="output"))
model3.add(Activation('softmax', name="softmax"))

model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


tb_callback = make_tb_callback('complex')
history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid),
                     # Don't forget to include the callback!
                    callbacks=[tb_callback])


plot_training_curves(history.history);


# # Comparing runs
# 
# TensorBoard lets us easily compare different runs (when saved to different subfolders). Here are the training and validation accuracy of the two above models:
# 
# <img src="img/tb-demo-acc-complex.png" width="400">
# <img src="img/tb-demo-val-acc-complex.png" width="400">
# 
# We can see that the more complex model generalizes better -- lower training accuracy, but higher validation.
# 

# # IMDB classification with MLPs and CNNs
# 
# The dataset is originally from http://ai.stanford.edu/~amaas/data/sentiment/
# 
# Code based on 
# https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
# 
# Here's our plan:
# 
# * load the dataset
# * Keep only the most frequently occuring 20000 words
# * Pad or truncate all reviews to fixed length (300)
# * Look up an embedding for each word
# * Build our networks: a simple MLP and then a 1D CNN
# 

from __future__ import print_function

import os
import os.path
import sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns


from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer


from display import visualize_keras_model, plot_training_curves


get_ipython().magic('matplotlib inline')
sns.set_style('white')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# To work around bug in latest version of the dataset in Keras,
# load older version manually, downloaded from 
# https://s3.amazonaws.com/text-datasets/imdb_full.pkl
print('Loading data...')
path = os.path.expanduser('~/.keras/datasets/imdb_full.pkl')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


word2index = imdb.get_word_index()


# we want the other direction
index2word = dict([(i,w) for (w,i) in word2index.items()])


def totext(review):
    return ' '.join(index2word[i] for i in review)


GLOVE_DIR = os.path.expanduser('~/data/glove')
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {} # word -> coefs
# We'll use the 100-dimensional version
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# Shuffle training and test data
# (copying from imdb.py because we had to load data manually)
seed = 113
np.random.seed(seed)
np.random.shuffle(x_train)
np.random.seed(seed) # same shuffle for labels!
np.random.shuffle(y_train)

np.random.seed(seed * 2)
np.random.shuffle(x_test)
np.random.seed(seed * 2)
np.random.shuffle(y_test)


print('Pad sequences')
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# Recall: words are indexed in descending order of frequency. Remove the 
# less frequent ones -- replace with constant value
x_train[x_train >= MAX_NB_WORDS] = MAX_NB_WORDS-1
x_test[x_test >= MAX_NB_WORDS] = MAX_NB_WORDS-1


print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word2index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# # MLP
# 
# Let's try a simple two-layer dense network, based on the embeddings...
# 

# train a regular MLP
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
x = Flatten()(embedded_sequences)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model_mlp = Model(sequence_input, preds)
model_mlp.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


model_mlp.summary()


print('Training model.')
history = model_mlp.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=25, batch_size=128)


plot_training_curves(history.history);


score, acc = model_mlp.evaluate(x_test, y_test,
                            batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)


# Overfitting! In practice, would use the weights after ~5 epochs. Get 70% accuracy. Let's see if we can do better with a CNN...
# 

# # 1D CNN
# 
# It's not a crazy assumption that adjacent words are more important than far away ones, and that the same groups of words in different positions mean roughly the same thing, so we can use our CNN tricks. Sequences are 1D, so we'll use a 1-D sliding window of weights.
# 

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(7)(x)  # modified from example since our seq len is 300 
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


model.summary()


print('Training model.')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=10, batch_size=128)


plot_training_curves(history.history);


score, acc = model.evaluate(x_test, y_test,
                            batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)


# Still overfitting on the training data, but looks like we have about 84% validation accuracy. Not too bad for a first attempt.
# 

# # LSTM
# Just for fun -- dataset isn't really big enough. Using fchollet's parameters..
# 

print('Build model...')
batch_size = 32
maxlen = 80

x_train_short = pad_sequences(x_train, maxlen=maxlen)
x_test_short = pad_sequences(x_train, maxlen=maxlen)

model_lstm = Sequential()
model_lstm.add(Embedding(MAX_NB_WORDS, 128))
model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model_lstm.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model_lstm.fit(x_train_short, y_train, batch_size=batch_size, epochs=15,
          validation_data=(x_test_short, y_test))


score, acc = model_lstm.evaluate(x_test_short, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# Perhaps due to different data prep because of the imdb.py bug, this didn't work at all... That's part of the fun of neural networks. To try:
# 
# * completely remove low frequency words from training data before training network. That way the 80 words we do keep will have more signal. (Can check how much it matters).
# * try the example unchanged, using older version of TF/Keras.
# 

# # Building a car classifier
# 
# Continuing as we were...
# 
# The plan going in:
# 
# - Try some ways to deal with our class imbalance
# - Try transfer learning by fine-tuning squeezenet or VGG16
# - See how much we can gain via data augmentation
# - Perhaps go back and redo our pre-processing to use bounding boxes.
# 

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# system
import os
import glob
import itertools as it
import operator
from collections import defaultdict
from StringIO import StringIO

# other libraries
import cPickle as pickle
import numpy as np 
import pandas as pd
import scipy.io  # for loading .mat files
import scipy.misc # for imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
import requests


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, GlobalAveragePooling2D
from keras.utils import np_utils

# https://github.com/fchollet/keras/issues/4499
from keras.layers.core import K
from keras.callbacks import TensorBoard

# for name scopes to make TensorBoard look prettier (doesn't work well yet)
import tensorflow as tf 


# my code
from display import (visualize_keras_model, plot_training_curves,
                     plot_confusion_matrix)
from helpers import combine_histories


get_ipython().magic('matplotlib inline')
sns.set_style("white")
p = sns.color_palette()

# repeatability:
np.random.seed(42)


data_root = os.path.expanduser("~/data/cars")


# # Load saved metadata
# 
# (the images are too big when stored raw -- faster to just reload from jpg)
# 

from collections import namedtuple
Example = namedtuple('Example',
                     ['rel_path', 'x1', 'y1', 'x2','y2','cls','test'])


# Load data we saved in 10-cars.ipynb
with open('class_details.pkl') as f:
    loaded = pickle.load(f)
    macro_classes = loaded['macro_classes']
    macro_class_map = loaded['macro_class_map']
    cls_tuples = loaded['cls_tuples']
    classes = loaded['classes']
    examples = loaded['examples']
    by_class = loaded['by_class']
    by_car_type = loaded['by_car_type']

macro_class_map


resized_path = os.path.join(data_root,'resized_car_ims') 


# ## Load the data again.
# 

def gray_to_rgb(im):
    """
    Noticed (due to array projection error in code below) that there is at least
    one grayscale image in the dataset.
    We'll use this to convert.
    """
    w, h = im.shape
    ret = np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,0] = im
    ret[:,:,1] = im
    ret[:,:,2] = im
    return ret


def load_examples(by_class, cls, limit=None):
    """
    Load examples for a class. Ignores test/train distinction -- 
    we'll do our own train/validation/test split later.
    
    Args:
        by_class: our above dict -- class_id -> [Example()]
        cls: which class to load
        limit: if not None, only load this many images.
        
    Returns:
        list of (X,y) tuples, one for each image.
            X: 3x227x227 ndarray of type uint8
            Y: class_id (will be equal to cls)
    """
    res = []
    to_load = by_class[cls]
    if limit:
        to_load = to_load[:limit]

    for ex in to_load:
        # load the resized image!
        img_path = os.path.join(data_root, 
                        ex.rel_path.replace('car_ims', 'resized_car_ims'))
        img = mpimg.imread(img_path)
        # handle any grayscale images
        if len(img.shape) == 2:
            img = gray_to_rgb(img)
        res.append((img, cls))
    return res


def split_examples(xs, valid_frac, test_frac):
    """
    Randomly splits the xs array into train, valid, test, with specified 
    percentages. Rounds down.
    
    Returns:
        (train, valid, test)
    """
    assert valid_frac + test_frac < 1
    
    n = len(xs)
    valid = int(valid_frac * n)
    test = int(test_frac * n)
    train = n - valid - test
    
    # don't change passed-in list
    shuffled = xs[:]
    np.random.shuffle(shuffled)

    return (shuffled[:train], 
            shuffled[train:train + valid], 
            shuffled[train + valid:])

# quick test
split_examples(range(10), 0.2, 0.4)


# Look at training data -- there's so little we can look at all of it

def plot_data(xs, ys, predicts):
    """Plot the images in xs, with corresponding correct labels
    and predictions.
    
    Args:
        xs: RGB or grayscale images with float32 values in [0,1].
        ys: one-hot encoded labels
        predicts: probability vectors (same dim as ys, normalized e.g. via softmax)
    """
    
    # sort all 3 by ys
    xs, ys, ps = zip(*sorted(zip(xs, ys, predicts), 
                             key=lambda tpl: tpl[1][0]))
    n = len(xs)
    rows = (n+9)/10
    fig, plots = plt.subplots(rows,10, sharex='all', sharey='all',
                             figsize=(20,2*rows), squeeze=False)
    for i in range(n):
        # read the image
        ax = plots[i // 10, i % 10]
        ax.axis('off')
        img = xs[i].reshape(227,227,-1) 

        if img.shape[-1] == 1: # Grayscale
            # Get rid of the unneeded dimension
            img = img.squeeze()
            # flip grayscale:
            img = 1-img 
            
        ax.imshow(img)
        # dot with one-hot vector picks out right element
        pcorrect = np.dot(ps[i], ys[i]) 
        if pcorrect > 0.8:
            color = "blue"
        else:
            color = "red"
        ax.set_title("{}   p={:.2f}".format(int(ys[i][0]), pcorrect),
                     loc='center', fontsize=18, color=color)
    return fig


# normalize the data, this time leaving it in color
def normalize_for_cnn(xs):
    ret = (xs / 255.0)
    return ret


def image_from_url(url):
    response = requests.get(url)
    img = Image.open(StringIO(response.content))
    return img


# Load images
IMG_PER_CAR = None # 20 # None to use all
valid_frac = 0.2
test_frac = 0.2

train = []
valid = []
test = []
for car_type, model_tuples in by_car_type.items():
    macro_class_id = macro_class_map[car_type]
    
    for model_tpl in model_tuples:
        cls = model_tpl[0]
        examples = load_examples(by_class, cls, limit=IMG_PER_CAR)
        # replace class labels with the id of the macro class
        examples = [(X, macro_class_id) for (X,y) in examples]
        # split each class separately, so all have same fractions of 
        # train/valid/test
        (cls_train, cls_valid, cls_test) = split_examples(
            examples,
            valid_frac, test_frac)
        # and add them to the overall train/valid/test sets
        train.extend(cls_train)
        valid.extend(cls_valid)
        test.extend(cls_test)

# ...and shuffle to make training work better.
np.random.shuffle(train)
np.random.shuffle(valid)
np.random.shuffle(test)


# We have lists of (X,Y) tuples. Let's unzip into lists of Xs and Ys.
X_train, Y_train = zip(*train)
X_valid, Y_valid = zip(*valid)
X_test, Y_test = zip(*test)

# and turn into np arrays of the right dimension.
def convert_X(xs):
    '''
    Take list of (w,h,3) images.
    Turn into an np array, change type to float32.
    '''
    return np.array(xs).astype('float32')
    
X_train = convert_X(X_train)
X_valid = convert_X(X_valid)
X_test = convert_X(X_test)


X_train.shape


def convert_Y(ys, macro_classes):
    '''
    Convert to np array, make one-hot.
    Already ensured they're sequential from zero.
    '''
    n_classes = len(macro_classes)
    return np_utils.to_categorical(ys, n_classes)

Y_train = convert_Y(Y_train, macro_classes)
Y_valid = convert_Y(Y_valid, macro_classes)
Y_test = convert_Y(Y_test, macro_classes)


Y_train.shape


# normalize the data, this time leaving it in color
X_train_norm = normalize_for_cnn(X_train)
X_valid_norm = normalize_for_cnn(X_valid)
X_test_norm = normalize_for_cnn(X_test)


# Let's use more or less the same model to start (num classes changes)
def cnn_model2(use_dropout=True):
    model = Sequential()
    nb_filters = 16
    pool_size = (2,2)
    filter_size = 3
    nb_classes = len(macro_classes)
    
    with tf.name_scope("conv1") as scope:
        model.add(Convolution2D(nb_filters, filter_size, 
                            input_shape=(227, 227, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("conv2") as scope:
        model.add(Convolution2D(nb_filters, filter_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("conv3") as scope:
        model.add(Convolution2D(nb_filters, filter_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("dense1") as scope:
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("softmax") as scope:
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
    return model

# Uncomment if getting a "Invalid argument: You must feed a value
# for placeholder tensor ..." when rerunning training. 
# K.clear_session() # https://github.com/fchollet/keras/issues/4499
    

model3 = cnn_model2()
model3.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# This model will train slowly, so let's checkpoint it periodically
from keras.callbacks import ModelCheckpoint


recompute = False

if recompute:
#     # Save info during computation so we can see what's happening
#     tbCallback = TensorBoard(
#         log_dir='./graph', histogram_freq=1, 
#         write_graph=False, write_images=False)

    checkpoint = ModelCheckpoint('macro_class_cnn_checkpoint.5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max',
                                 save_weights_only=True)

    # Fit the model! Using a bigger batch size and fewer epochs
    # because we have ~10K training images now instead of 100.
    history = model3.fit(
        X_train_norm, Y_train,
        batch_size=64, nb_epoch=50, verbose=1,
        validation_data=(X_valid_norm, Y_valid),
        callbacks=[checkpoint]
    )
else:
    model3.load_weights('macro_class_cnn.5')


# change to True to save
if False:
    model3.save('macro_class_cnn.h5')


# ## Diagnosing what's going wrong...
# 
# As we saw before, the model is starting to overfit. Let's try to diagnose what's going on, then decide what to do. Let's start by looking at the confusion matrices again.
# 

# Get the predictions
predict_train = model3.predict(X_train_norm)
predict_valid = model3.predict(X_valid_norm)
predict_test = model3.predict(X_test_norm)


plot_confusion_matrix(Y_test, predict_test, macro_classes,
                      normalize=False,
                      title="Test confusion matrix");


plot_confusion_matrix(Y_train, predict_train, macro_classes,
                      title="Train confusion matrix");


# **Note: normalized confusion matrices can be helpful...**
# 

# Normalized to see per-class behavior better
plot_confusion_matrix(Y_train, predict_train, macro_classes,                      
                      title="Train confusion matrix", normalize=True);


# Well, it seems that most car types are classified as sedan. Not too surprising, especially given that sedans are overrepresented. It's starting to learn that SUVs and pickups are different from sedans, and occasionally manages to distinguish coupes from sedans. 
# 
# So far, it doesn't use minivan, van, or wagon labels at all.
# 
# **Things to check / do:**
# - Is it correctly getting coupe images from the side, incorrectly from the front or back?
# - Look at class probabilities for some images, not just the maximal one
# - Count how many training images we have for each class. May want to oversample the low prob classes.
# - Try fine-tuning an off-the-shelf model.

# What's our class balance
xs, counts = np.unique(np.argmax(Y_train, axis=1),return_counts=True)
plt.bar(xs, counts, tick_label=macro_classes, align='center')


# ## Let's look at some mistakes on the train set
# 

predict_train_labels = np.argmax(predict_train, axis=1)
correct_labels = np.argmax(Y_train, axis=1)


correct_train = np.where(predict_train_labels==correct_labels)[0]
wrong_train = np.where(predict_train_labels!=correct_labels)[0]
percent = 100 * len(correct_train)/float(len(correct_labels))
print("Training: {:.2f}% correct".format(percent))


n_to_view = 20
subset = np.random.choice(correct_train, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Correct predictions")


# Note that plot_data uses red whenever the probability of the correct class is <0.8. This doesn't make much sense with 8 classes -- would be nice to change it. These were all in fact "correct" -- true class had maximum prob. 
# 

n_to_view = 20
subset = np.random.choice(wrong_train, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Wrong predictions")


# ## Zooming in on the coupe class in particular
# 
# Is there a pattern for when it's confusing coupes and sedans?

correct_coupe = np.where((predict_train_labels==correct_labels) & (correct_labels==macro_class_map['Coupe']))[0]
wrong_coupe = np.where((predict_train_labels!=correct_labels) & (correct_labels==macro_class_map['Coupe']))[0]

n_to_view = 40
subset = np.random.choice(correct_coupe, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Correct coupe predictions")

subset = np.random.choice(wrong_coupe, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Wrong coupe predictions", fontsize=18)


# Should make some functions to make this kind of analysis easier...
# 
# I don't see a clear pattern--side and front and back views in both sets of labels. One option is to keep training our own network, then come back and fight overfitting somehow. Instead, lets try transfer learning using squeezenet. 
# 

# # Try transfer learning
# 
# Let's try to use a model that's been trained on the imagenet dataset (1 million images!) That's generally a better place to start for this kind of vision problem than training from scratch on a small dataset. 
# 
# ## First, get SqueezeNet set up
# 
# https://github.com/rcmalli/keras-squeezenet
# 

get_ipython().system('pip install keras_squeezenet')


from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


model = SqueezeNet()


# Oops -- screwed up the first training image (see note about preprocess_input below)
X_train[0]


img = X_train[1]
plt.imshow(img/255.0)
x = img# image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# preprocess_input modifies its argument!
x = preprocess_input(x.copy())

preds = model.predict(x)

print('Predicted:', decode_predicktions(preds))


# screwed up X_train[0] earlier. Rather than rerun, I tried to hack/fix it manually 
# (didn't really work, but it's just one image, so decided not to care). Now it looks funny, and is a good
# reminder to check your data throughout your pipeline, not just once at the beginning...
plt.imshow(X_train[0])


# # Adapting Squeezenet
# 
# Let's replace the output layer with a smaller classifier.
# 
# Following some comments at https://github.com/fchollet/keras/issues/2371 and our code in 07-transfer.
# 

model.summary()


# ## Modify the model to compute bottleneck features...
# 
# Leave out the final classification layers.
# 

# We want to pull out the activations before conv10...

from keras.models import Model

# Get input
new_input = model.input
# Find the layer to connect
hidden_layer = model.get_layer('drop9').output
# Build a new model
bottleneck_model = Model(new_input, hidden_layer)
bottleneck_model.summary()


16000 * 13 * 13 * 512 * 4 / 2**20


# This will generate 5G of saved features! I guess I can save 20% by not pre-computing the test ones, but doesn't seem worth it. Instead, let's first train on a subset. We'll ignore the test set entirely, and see if we can get reasonable 
# validation performance by using 2000 training images and 1000 validation.
# 

train_subset = 2000
valid_subset = 1000


def save_bottlebeck_features(bottleneck_model, xs, name):
    # don't change the param!
    xs = preprocess_input(xs.copy())
    bottleneck_features = bottleneck_model.predict(xs)
    
    with open('cars_bottleneck_features_{}.npy'.format(name), 'w') as f:
        np.save(f, bottleneck_features)

def save_labels(ys, name):
    with open('cars_bottleneck_labels_{}.npy'.format(name), 'w') as f:
        np.save(f, ys)

        
if False: # change to True to recompute  
    save_bottlebeck_features(bottleneck_model, X_train[:train_subset], 'train_subset')
    save_bottlebeck_features(bottleneck_model, X_valid[:valid_subset], 'valid_subset')
    # save_bottlebeck_features(bottleneck_model, X_test, 'test')
    
    save_labels(Y_train[:train_subset], 'train_subset')
    save_labels(Y_valid[:valid_subset], 'valid_subset')


get_ipython().system('ls -lh cars*')


def load_features(name):
    with open('cars_bottleneck_features_{}.npy'.format(name), 'r') as f:
        return np.load(f)

def load_labels(name):
    with open('cars_bottleneck_labels_{}.npy'.format(name)) as f:
        return np.load(f)


top_model_weights_path = 'cars_bottleneck_fc_model.h5'    
    
# Now let's train the model -- we'll put the same squeezenet structure, just with fewer classes
def make_top_model():
    inputs = Input((13,13,512))
    x = Convolution2D(len(macro_classes), (1, 1), padding='valid', name='new_conv10')(inputs)
    x = Activation('relu', name='new_relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs, out, name='squeezed_top')
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

top_model = make_top_model()
print(top_model.summary())


train_data = load_features('train_subset')
train_labels = load_labels('train_subset')

valid_data = load_features('valid_subset')
valid_labels = load_labels('valid_subset')


epochs = 50
batch_size = 128
history = top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(valid_data, valid_labels))

top_model.save_weights(top_model_weights_path)


plot_training_curves(history.history);


# Almost 90% validation accuracy! Clearly we could have stopped earlier. Let's take a quick look at the confusion matrix.
# 

predict_train = top_model.predict(train_data)


plot_confusion_matrix(train_labels, predict_train, macro_classes,                      
                      title="Train confusion matrix");
plt.figure()
plot_confusion_matrix(train_labels, predict_train, macro_classes,                      
                      title="Train confusion matrix",
                     normalize=True);


# So we're making relatively few mistakes with pickups and vans and sedans, somewhat more with SUVs, and confusing wagons, convertibles, and coupes for sedans. Makes sense. Perhaps we should combine all those classes together anyway.
# 
# We could fine-tune the network and do data augmentation too. For now, let's just train on the rest of our training data.
# 

def compute_bottleneck_features(xs):
    xs = preprocess_input(xs.copy())
    return bottleneck_model.predict(xs)

rest_train_data = compute_bottleneck_features(X_train[train_subset:])
rest_train_labels = Y_train[train_subset:]


epochs = 50
batch_size = 128
history2 = top_model.fit(rest_train_data, rest_train_labels,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(valid_data, valid_labels))


from helpers import combine_histories
plot_training_curves(combine_histories(history.history, history2.history));


# Ok, that got us about 2 more percent. Let's save these weights.
# 

top_model.save_weights(top_model_weights_path)


# ## Look at validation confusion matrix
# 
# So, what's the confusion matrix for our validation data like?

predict_valid = top_model.predict(valid_data)


top_model.evaluate(valid_data, valid_labels)


plot_confusion_matrix(valid_labels, predict_valid, macro_classes,                      
                      title="Validation confusion matrix");
plt.figure()
plot_confusion_matrix(valid_labels, predict_valid, macro_classes,                      
                      title="Validation confusion matrix",
                     normalize=True);


# # Next steps...
# 
# We'll stop for now. If we wanted to continue, here are some things to do:
# 
# * Continue to improve the classifier by using data augmentation
# * Fine-tune several layers of squeezenet
# * Go back to our raw data and use the bounding box info, or resize/crop differently.
# * Add more than one new layer on top of squeezenet.
# * Try a different pre-trained network.
# * To test out our final output, combine the really confusing classes as discussed above, and make a handy function to
# take an image and run it through the combined network to 
# give a classification...
# 

# Roughly based on the keras mnist-mlp example, but using the [notmnist](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset. The dataset consists of images of letters from A-J, in lots of different fonts. We'll be using a subset of the whole dataset.
# 

# for DSX, need to switch to the right directory. Detect using path name.
s = get_ipython().magic('pwd')
if s.startswith('/gpfs'):
    get_ipython().magic('cd ~/deep-learning-workshop/')


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# for making plots prettier
import seaborn as sns 
sns.set_style('white')


from __future__ import print_function
np.random.seed(1337)  # for reproducibility


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


import notmnist
notmnist_path = "~/data/notmnist/notMNIST.pickle"


from display import visualize_keras_model, plot_training_curves
from helpers import combine_histories


# # Load the data
# 

# the data, shuffled and split between train, validation, and test sets
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notmnist.load_data(notmnist_path)


# # Look at the data
# 
# Always a good idea to know what you're working with.
# 

len(x_train), len(x_valid), len(x_test)


# expect images...
plt.imshow(x_train[0])


# Confirm that labels are in order, with 'a' == 0
y_train[0], ord('g')-ord('a')


# Look at a bunch of examples
fig, axs = plt.subplots(20,20, sharex=True, sharey=True, figsize=(10,10))
for i, idx in enumerate(np.random.choice(len(x_train), 400, replace=False)):
    img = x_train[idx]
    ax = axs[i//20, i%20]
    ax.imshow(img)
    ax.axis('off')
sns.despine(fig, left=True, bottom=True)


# Many look pretty straightforward, but some are clearly going to be challenging. e.g. what letters are the various stars supposed to be? Not to mention the totally blank images.
# 

# look at the distribution of values being used
fig, ax = plt.subplots(figsize=(3,2))
ax.hist(x_train[0].flatten(), bins=20);
sns.despine(fig)


# So values are normalized to be between -0.5 and 0.5. This is handy for feeding into neural networks, because of the useful ranges of our activation functions.
# 

# # Prepare the data
# 
# Reshape inputs to flat vectors, convert labels to one-hot.
# 

x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')


batch_size = 128
nb_classes = 10
nb_epoch = 10


# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# ## Make our first multi-layer model
# 
# We'll have one "hidden" layer, then our output layer with 10 classes.
# 

model = Sequential()
model.add(Dense(128, input_shape=(784,), name="hidden"))
model.add(Activation('relu', name="ReLU"))
model.add(Dense(10, name="output"))
model.add(Activation('softmax', name="softmax"))

model.summary()

# for multi-class classification, we'll use cross-entropy as the loss.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              # Here, we tell Keras that we care about accuracy in addition to loss
              metrics=['accuracy'])


visualize_keras_model(model)


history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


plot_training_curves(history.history);


# Both training and validation accuracy still going up. Let's train some more.
# 

history2 = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_valid, y_valid))


plot_training_curves(combine_histories(history.history, history2.history));


# Training accuracy still going up, but doesn't help validation accuracy anymore. We can see this in the test score too (note that this is cheating a bit -- best practice is to not look at your test data till you're pretty happy with your model).
# 

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Let's try a bigger model, with an extra layer, and dropout to reduce overfitting.
# 

model2 = Sequential()
model2.add(Dense(512, input_shape=(784,), name="hidden1"))
model2.add(Activation('relu', name="ReLU1"))
model2.add(Dropout(0.2))
model2.add(Dense(512, input_shape=(784,), name="hidden2"))
model2.add(Activation('relu', name="ReLU2"))
model2.add(Dropout(0.2))
model2.add(Dense(10, name="output"))
model2.add(Activation('softmax', name="softmax"))

model2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


visualize_keras_model(model2)


# let's train for 20 epochs right away
history = model2.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))


plot_training_curves(history.history);


# So, validation accuracy plateaus very quickly. It's somewhat strange that it starts out higher than training accuracy too. Likely noise...
# 
# Let's add more regularization to the model -- higher dropout first.
# 

model3 = Sequential()
model3.add(Dense(512, input_shape=(784,), name="hidden1"))
model3.add(Activation('relu', name="ReLU1"))
model3.add(Dropout(0.5))
model3.add(Dense(512, input_shape=(784,), name="hidden2"))
model3.add(Activation('relu', name="ReLU2"))
model3.add(Dropout(0.5))
model3.add(Dense(10, name="output"))
model3.add(Activation('softmax', name="softmax"))

model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))


plot_training_curves(history.history);


# Aha. Much better curves. Still going up, so let's train some more. 
# 

history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))


plot_training_curves(history.history);


# Ok, not gaining much anymore (note new scale). Could fiddle further (e.g. reducing learning rate now that performance has plateaued), but let's stop for now. We'll try more sophisticated approaches later to get better performance. Now we can look at test performance.
# 

score = model3.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# You should be surprised that the test accuracy is significantly better than the validation accuracy. The reason is that the test set is "clean"--manually curated, so practically all the labels are correct. The training and validation sets are noisy, and have some wrong labels. 
# 

# ## Questions you should have
# 
# This is an intro, so there were many "magical" decisions in the above code. I suggest making your own list, then comparing to this partial one:
# 
# - why start with dropout 0.2, then switch to 0.5. Why not something else?
# - why two hidden layers?
# - why not a different batch size?
# - why use the RMSprop optimizer?
# - why use 512 hidden units in the hidden layers? Why not more or less?
# - Are there other ways of reducing overfitting besides dropout? Would they work as well?
# 
# As a preview, the primary answer to all of the above is "try initial values based on your own and others' experience, then tweak from there."

# # Transfer learning with Keras
# 
# Based on https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d,
# as discussed on the [Keras blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) ("Building powerful image classification models using very little data")
# 

# It uses data that can be downloaded at:
# https://www.kaggle.com/c/dogs-vs-cats/data
# 
# In our setup, we:
# - created a data/ folder
# - created train/ and validation/ subfolders inside data/
# - created cats/ and dogs/ subfolders inside train/ and validation/
# - put the cat pictures index 0-999 in data/train/cats
# - put the cat pictures index 1000-1400 in data/validation/cats
# - put the dogs pictures index 12500-13499 in data/train/dogs
# - put the dog pictures index 13500-13900 in data/validation/dogs
# So that we have 1000 training examples for each class, and 400 validation examples for each class.
# In summary, this is our directory structure:
# 
# ```
# data/
#     train/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...
#     validation/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...
# ```
# 

# for DSX, need to switch to the right directory. Detect using path name.
s = get_ipython().magic('pwd')
if s.startswith('/gpfs'):
    get_ipython().magic('cd ~/deep-learning-workshop/')
    


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


import os
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# for making plots prettier
import seaborn as sns 
sns.set_style('white')


from __future__ import print_function
np.random.seed(1331)  # for reproducibility


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers


from display import visualize_keras_model, plot_training_curves


data_root = os.path.expanduser("~/data/cats_dogs")
train_data_dir = os.path.join(data_root, 'train')
validation_data_dir = os.path.join(data_root, 'validation')


# Make sure we have the expected numbers of images
for d in [train_data_dir, validation_data_dir]:
    for category in ['cats', 'dogs']:
        print("{}/{}: {}".format(
                d, category, len(os.listdir(os.path.join(d, category)))))


# Let's look at a few of the images...
# 

train_cats = os.path.join(train_data_dir,'cats')
train_dogs = os.path.join(train_data_dir,'dogs')

def viz_dir(dirpath):
    image_paths = os.listdir(dirpath)

    fig, axs = plt.subplots(2,5, figsize=(11,2.5))
    for i, img_path in enumerate(np.random.choice(image_paths, 10, replace=False)):
        img = Image.open(os.path.join(dirpath, img_path))
        ax = axs[i//5, i%5]
        ax.imshow(img)
        ax.axis('off')
        
viz_dir(train_cats)
viz_dir(train_dogs)


# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


visualize_keras_model(model)


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')


plot_training_curves(history.history);


# # Using a pre-trained model
# 
# Based on https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# 

from keras.preprocessing.image import ImageDataGenerator
from keras import applications

top_model_weights_path = 'bottleneck_fc_model.h5'
nb_train_samples = 2000
nb_validation_samples = 800

epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    # Note: train_data.shape = (2000, 4, 4, 512)
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    return history


save_bottlebeck_features()


history = train_top_model()


plot_training_curves(history.history);


# 90% accuracy in just a couple of minutes! Nice.
# 

# # Fine-tuning the whole network
# 
# Based on https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
# 

# Reset things...
from keras.layers.core import K
K.clear_session() 


# path to the model weights files.
weights_path = os.path.expanduser('~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

epochs = 50
batch_size = 16


# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', 
                           include_top=False,
 # Add below line to gist to fix error:
 # ValueError: The shape of the input to "Flatten" is not fully
 # defined (got (None, None, 512). We can use our own
 # width and height because we're only keeping the convolutional layers
                           input_shape=(img_height,img_width,3))
print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))


# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)


# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


# Now we have the VGG model with our own layer on top
visualize_keras_model(model)


# Easier to count layers in this form. We want to freeze the first 
# 4 conv->pool blocks, which works out to be the first 15 layers
# (note that Keras blog post says 25--seems like a typo!)
model.layers


# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# fine-tune the model
epochs = 20
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/batch_size)


plot_training_curves(history.history);


full_model_weights_path = 'full_model_weights.h5'
model.save_weights(full_model_weights_path)


# We got a couple of percent lift on validation accuracy. We'll stop here for now. To improve further, we could try:
# 
# - increasing dropout or other regularization and train more
# - using data augmentation for the pre-trained model as well
# - fine tuning more of the VGG model (though we'd want to fight overfitting more)
# 

# # Exploration of the IMDB dataset
# 
# The dataset is originally from http://ai.stanford.edu/~amaas/data/sentiment/
# 

get_ipython().magic('matplotlib inline')


from __future__ import print_function

from keras.datasets import imdb

import cPickle as pickle
import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


max_features = 20000

# To work around bug in latest version of the dataset in Keras,
# load older version manually, downloaded from 
# https://s3.amazonaws.com/text-datasets/imdb_full.pkl
print('Loading data...')
path = os.path.expanduser('~/.keras/datasets/imdb_full.pkl')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# what does the data look like?
x_train[0]


# Hmm. That doesn't look like a movie review! The words are replaced by indexes. Let's convert that review back to text.
# 

word2index = imdb.get_word_index()


word2index.items()[:5]


# we want the other direction
index2word = dict([(i,w) for (w,i) in word2index.items()])


print('\n'.join([index2word[i] for i in range(1,20)]))


# Note that words are indexed in descending order of frequency. This means we can remove words with high index if we want to reduce the vocab size.
# 

def totext(review):
    return ' '.join(index2word[i] for i in review)

for review in x_train[:10]:
    # let's look at the first 30 words
    print(totext(review[:30]))
    print('\n')


# what about labels?
y_train[1]


# how many labels?
np.unique(y_train)


# label balance
np.unique(y_train, return_counts=True)


# Evenly balanced between positive (we could also have read this in the docs, but it's good to confirm :)
# 

# How long are the reviews?

lengths = map(len, x_train)
fig, axs = plt.subplots(2,1, figsize=(3,5))
axs[0].hist(lengths, bins=30)
axs[1].hist(lengths, bins=30, cumulative=True, normed=True)
axs[1].set_xlim([0,800])
axs[1].set_xticks(range(0,800,150))

sns.despine(fig)


# We'll want to truncate long reviews and pad short ones to the same length. 300-500 seems like a reasonable option to keep most of the data.
# 

