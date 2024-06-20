# Data source: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
# 
# > some publicly available fonts and extracted glyphs from them to make a dataset similar to MNIST. There are 10 classes, with letters A-J taken from different fonts.
# 
# > Approaching 0.5% error rate on notMNIST_small would be very impressive. If you run your algorithm on this dataset, please let me know your results.
# 
# This is a starter 
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# try using other layers
from keras.layers import Conv2D, MaxPool2D, Dropout


# optionally
# install with: 
from keras_sequential_ascii import sequential_model_to_ascii_printout


# load data
data = io.loadmat("notMNIST_small.mat")


# transform data
X = data['images']
y = data['labels']
resolution = 28
classes = 10

X = np.transpose(X, (2, 0, 1))

y = y.astype('int32')
X = X.astype('float32') / 255.

# channel for X
X = X.reshape((-1, resolution, resolution, 1))

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
Y = np_utils.to_categorical(y, 10)


# looking at data; some fonts are strange
i = 42
print("It is:", "ABCDEFGHIJ"[y[i]])
plt.imshow(X[i,:,:,0]);


# splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# creating a simple netural network
# in this case - just logistic regression
model = Sequential()

# add Conv2D and MaxPool2D layers

model.add(Flatten(input_shape=(resolution, resolution, 1)))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


sequential_model_to_ascii_printout(model)


model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test))





# # Live loss plots in Jupyter Notebook for Keras
# 
# by [Piotr Migdał](http://p.migdal.pl/)
# 
# * inspired by a Reddit discussion [Live loss plots inside Jupyter Notebook for Keras? - r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/65jelb/d_live_loss_plots_inside_jupyter_notebook_for/)
# * my other Keras add-on: [Sequential model in Keras -> parameter ASCII diagram](https://github.com/stared/keras-sequential-ascii)
# 

get_ipython().magic('matplotlib inline')

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output


keras.__version__


# data loading
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# data preprocessing
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.


# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()


# just logistic regression, to keep it simple and fast

model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# in this static viewer it is not obvious,
# but this plot grows step by step

model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test),
          callbacks=[plot_losses])


# ## Further ideas
# 
# * loss and accuracy side by side, as two plots
# * time per epoch (plot title?)
# 




# It is a Jupyter Notebook showing (not explaining!) basics of neural networks with Keras, using [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
# 
# It assumes:
# 
# * Python 3.5+ with Jupyter Notebook (e.g. [Anaconda distribution](https://www.continuum.io/downloads))
# * [Keras](https://keras.io/) 2.x with [TensorFlow](https://www.tensorflow.org/) 1.x backend
# 
# Optionally, install [stared/keras-sequential-ascii](https://github.com/stared/keras-sequential-ascii) to see architecture visualizaitons:
# 
# * `pip install git+git://github.com/stared/keras-sequential-ascii.git`
# 

get_ipython().magic('matplotlib inline')
# import seaborn as sns

from keras.datasets import mnist
from keras import utils
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPool2D


from keras_sequential_ascii import sequential_model_to_ascii_printout


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# (number of examples, x, y)
X_train.shape


X_test.shape


# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
Y_train = utils.to_categorical(y_train)
Y_test = utils.to_categorical(y_test)


# we need to add channel dimension (for convolutions)

# TensorFlow backend
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# for Theano backend, it would be:
# X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.
# X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.


# ## Logistic Regression
# 
# Multi-class [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), expressed in Keras.
# 

model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))  # for Theano: (1, 28, 28)
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


sequential_model_to_ascii_printout(model)


# look at validation scores
model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))


# ## One hidden layer
# 
# Old-school neural networks.
# 

model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


sequential_model_to_ascii_printout(model)


model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))


# ## Convolutional network
# 
# Not yet that deep.
# 

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


sequential_model_to_ascii_printout(model)


model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))


# ## Ideas
# 
# * Add one more `Conv2D` and `MaxPool2D` layer.
# * Add one more dense layer (with `relu` activation) before.
# 

