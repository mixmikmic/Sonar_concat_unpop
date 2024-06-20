# In this demo, we will try to model time series using LSTMs. We will try two kind of time series, a periodic time series (generated using sine function) and a transcendental time series (generated using exponential function)
# 

import numpy as np
import matplotlib.pyplot as plt
# import math
# from sklearn.metrics import mean_squared_error


# function to be modelled
f = np.sin


number_of_period = 10
number_of_points_per_period = 20


x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, number_of_points_per_period)
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, np.sin(x), 'ro')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()


# We will try to predict the series corresponding to sine values of X, given the previous sine values.
# 

np.random.seed(42)
dataset = np.sin(X)


# train-test split
train_size = int(0.8*len(dataset))
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]


len(train)


# convert a time series array into matrix of previous values
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        data_x.append(a)
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


train[0], train[1], train[2]


trainX[0], trainX[1], trainX[2]


trainY[0], trainY[1], trainY[2]


trainX.shape


# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


from keras.layers import Dense, LSTM
from keras.models import Sequential


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


len(trainPredict)


x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)[:len(trainPredict)]
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, trainPredict, 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()


# Let us increase the lookback size
# 

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)[:len(trainPredict)]
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, trainPredict, 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()


f = np.exp


f(100)


number_of_points = 20


plt.close()
x = np.linspace(-10, 20, number_of_points_per_period)
X = np.linspace(-10, 20, 1000)
plt.plot(X, f(X), 'b')
plt.plot(x, f(x), 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()


dataset = f(X)
# train-test split
train_size = int(0.8*len(dataset))
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]


len(dataset)


look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=100, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=10000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))


from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=1000000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))





# MNIST Classification task is the "hello world" of Computer Vision. The task is to identify (or classify) the greyscale images of hand-written digits. The dataset can be easily loaded using Keras.
# 

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


n_train, height, width = X_train.shape
n_test, _, _ = X_test.shape
print("Number of samples in training data = {}.\nDimensions of each sample are {}X{}".format(n_train, height, width))
print("Number of samples in test data = {}".format(n_test))


print(X_train.shape)
print(X_train[0].shape)


import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()


print(y_train)


# We need to process the data before feeding to Keras. 
# 
# * Reshape data. Keras layers for 2-d convolution expects another dimension corresponding to pixel. For greyscale image, we have only one pixel dimension. For RGB, there would be 3 pixel dimensions.
# * Normalize image data so that all pixel values are in range [0, 1] instead of [0, 255]. 
# * Transform output values to be one-hot vectors instead of single values.
# 

from keras.utils.np_utils import to_categorical
from keras import backend as K

# Reshaping data

# this if condition is needed because the input shape is specified differently for theano and tensorflow backend.
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
    X_test = X_test.reshape(n_test, 1, height, width).astype('float32')
    input_shape = (1, height, width)
else:
    X_train = X_train.reshape(n_train, height, width, 1).astype('float32')
    X_test = X_test.reshape(n_test, height, width, 1).astype('float32')
    input_shape = (height, width, 1)
    

# Normalizing data
X_train /= 255
X_test /= 255

# Transforming output variables
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


print(X_train.shape)
print(X_train[0].shape)


print(y_train)


print(y_train[0])
print(y_train[1])


# We will create a sequential Keras model where different layers would be added in sequence.
# 

from keras.models import Sequential
model = Sequential()


# Architecture:
#  * Convolution layers
#      * Convolution layer would have 32 feature maps.
#      * Each feature map will be of size 5x5.
#      * This will be followed by RELU activation.
#  * Pooling layer
#      * Pool Size of 2x2
#  * Regularization layer (Dropout)
#      * Exclude 25% of neurons in the layer to reduce overfitting.
#  * Flatten layer to convert the matrix data to vectors.
#  * Fully connected (dense) layer
#      * 128 neurons and RELU activation.
#  * Regularization layer (Dropout)
#      * Exclude 25% of neurons in the layer to reduce overfitting.
#  * Fully connected (dense) layer
#      * 10 neurons (1 for each class)
#      * Softmax classifier convolution filters).
# 

from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

# Convolution Layer

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, activation='relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Regularization Layer
model.add(Dropout(0.25))

# Flatten Layer
model.add(Flatten())

# Fully Connected
model.add(Dense(128))
model.add(Activation('relu'))

# Regularization Layer
model.add(Dropout(0.5))

# Fully Connected with softmax
model.add(Dense(n_classes))
model.add(Activation('softmax'))


# Specify the loss function, the optimizer to use and the metric to track.
# 

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Now we can start the training. For demo purpose, we will train it only for 2 epochs. Feel free to train for more epochs.
# 

model.fit(X_train, y_train, batch_size=128, nb_epoch=2, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)





