# ## Predicting level values from last 300 sec data using tflearn
# 

from __future__ import division, print_function, absolute_import
import re, collections
import numpy as np
import pandas as pd

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.estimator import regression

from IPython import display
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.set_option('max_rows', 100)
# np.set_printoptions(threshold=np.nan)


# ### Creating the new csv file with missing
# Don't run the below cell as it takes a long time to generate new dataframe
# The below lines read our csv and adds missing time second data to df2 as described in 
# 

# df = pd.read_csv('irage_ds_trim.csv',parse_dates = ['date'], 
#                    infer_datetime_format = True)
# 
# df.insert(1,'time_diff',0)
# df['time_diff'] = (df['date']-df['date'].shift()).fillna(0).apply(lambda x: x  / np.timedelta64(1,'s')).astype('int64') % (24*60*60)
# df.head(n=50)
# 
# df2 = pd.DataFrame(columns = df.columns)
# rows = len(df.index)
# c=0
# for i in range(rows-1):
#     time_diff = df.iloc[i+1,1] 
#     date_diff = df.iloc[i+1,0].day - df.iloc[i,0].day
#     if time_diff >1 and date_diff==0:
#         for t in range(0,time_diff):
#             new_time = df.iloc[i,0] + pd.DateOffset(seconds=t)
#             df2 = pd.concat([df2, df.iloc[[i]]], ignore_index=True)
#             df2.iloc[len(df2.index)-1,0] = new_time
#     else:
#         df2 = pd.concat([df2, df.iloc[[i]]], ignore_index=True)
#       
#     
#     if c==1000:
#         display.clear_output(wait=True)
#         print('At row: %s and %0.2f completed'%(i,i/rows*100))
#         c=0
#     c+=1
# 
# 
# df2.drop(['time_diff'])
# df2.to_csv('modified.csv', index = False)
# 

df = pd.read_csv('modified.csv',parse_dates = ['date'], 
                 infer_datetime_format = True)

df.drop(['date','b0','a0','fut_direction'], 1, inplace = True)
df.sort_index(axis=1, inplace = True)
df


# Bringing the data in the shape as shown in 'How data looks.xlsx excel file'
X = df.values

print('Original X')
print(X.shape)

seq_length = 300
img_height = 20
num_images = X.shape[0]//seq_length

X = X[:seq_length*num_images,:]
print('\nX after calculating num_images')
print(X.shape)

X = X.reshape(num_images,seq_length, img_height).astype("float32").transpose((0,2,1))

Y = np.array([X[i+1,:,0] for i,x in enumerate(X) if i<X.shape[0]-1]).astype("int64")
print('\nChecking Y \nFirst time series value of X[1]')
print(X[1,:,0])
print('Value of Y corresponding to X[0]')
print(Y[0])
print('\nX and Y after converting to 20x300 time-series images')
X = X[:-1,:,:]
print(X.shape)
print(Y.shape)

train_split = 0.85
split_val = round(num_images*train_split)

X_train = X[:split_val,:,:,np.newaxis]
X_test = X[split_val:,:,:,np.newaxis]
Y_train = Y[:split_val,:]
Y_test = Y[split_val:,:]

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_train, axis=0)
X_test /= np.std(X_train, axis=0)

print('\nX and Y after train-test split, normalisation and channel dimension insertion')
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### Making the neural network
# 

tf.reset_default_graph()
filter_size = 3
stride_y = 5
epochs = 50

# Convolutional network with 2 conv and 2 fully connected layers
network = input_data(shape=[None, 20, 300, 1])
conv1 = conv_2d(network, 32, filter_size, activation='relu', padding = 'same', strides= [1,1,stride_y,1], name = 'conv1')
pool1 = max_pool_2d(conv1, 2)
conv2 = conv_2d(pool1, 64, filter_size, activation='relu', padding = 'same', name ='conv2')
pool2 = max_pool_2d(conv2, 2)
fc1 = fully_connected(pool2, 512, activation='relu',name ='fc1')
drop1 = dropout(fc1, 0.5)
fc2 = fully_connected(drop1, 20, name ='fc2')
network = regression(fc2, optimizer='adam', loss='mean_square',
                     learning_rate=0.001, metric='R2')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, Y_train, n_epoch=epochs, shuffle=False, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=26,run_id='irage')

conv1 = model.get_weights(conv1.W)
conv2 = model.get_weights(conv2.W)
fc1 = model.get_weights(fc1.W)
fc2 = model.get_weights(fc2.W)


# ### Tried different parameters
# 
# filter_size = 3 stride y = 5 epoch = 50
# Training Step: 3949  | total loss: 14.60604 | time: 2.270s
# | Adam | epoch: 050 | loss: 14.60604 - R2: 1.0658 -- iter: 2028/2054
# Training Step: 3950  | total loss: 14.87758 | time: 3.300s
# | Adam | epoch: 050 | loss: 14.87758 - R2: 1.0415 | val_loss: 18890.15718 - val_acc: 54.9775 -- iter: 2054/2054
# 
# filter_size = 5 stride y = 5 epoch = 50
# Training Step: 3949  | total loss: 16.56158 | time: 2.267s
# | Adam | epoch: 050 | loss: 16.56158 - R2: 1.0335 -- iter: 2028/2054
# Training Step: 3950  | total loss: 17.20631 | time: 3.305s
# | Adam | epoch: 050 | loss: 17.20631 - R2: 1.0086 | val_loss: 23708.56095 - val_acc: 67.8062 -- iter: 2054/2054
# 
# filter_size = 5 stride y = 5 epoch = 30
# Training Step: 2369  | total loss: 21.36651 | time: 2.831s
# | Adam | epoch: 030 | loss: 21.36651 - R2: 0.9475 -- iter: 2028/2054
# Training Step: 2370  | total loss: 20.76781 | time: 3.872s
# | Adam | epoch: 030 | loss: 20.76781 - R2: 0.9431 | val_loss: 62821.49411 - val_acc: 163.0235 -- iter: 2054/2054
# 
# filter_size = 5 stride y = 5 epoch = 20
# Training Step: 1579  | total loss: 21.93146 | time: 2.857s
# | Adam | epoch: 020 | loss: 21.93146 - R2: 0.9790 -- iter: 2028/2054
# Training Step: 1580  | total loss: 21.09883 | time: 3.910s
# | Adam | epoch: 020 | loss: 21.09883 - R2: 0.9611 | val_loss: 60197.73860 - val_acc: 155.9296 -- iter: 2054/2054
# 
# filter_size = 5 stride y = 5 epoch = 10
# Training Step: 789  | total loss: 27.39871 | time: 2.850s
# | Adam | epoch: 010 | loss: 27.39871 - R2: 0.9602 -- iter: 2028/2054
# Training Step: 790  | total loss: 29.12849 | time: 3.912s
# | Adam | epoch: 010 | loss: 29.12849 - R2: 0.9374 | val_loss: 71309.16242 - val_acc: 183.2223 -- iter: 2054/2054
# 

tf.trainable_variables()


# ### Tried saving array as image but coudn't
# 

from PIL import Image

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')


# ### Tried to do VisualBackProp 
# 

t = np.mean(conv2, axis=3, keepdims = True)
t = np.mean(t, axis=2, keepdims = True)

conv2_avg = tf.constant(t)
conv2_decon = conv_2d_transpose(conv2_avg, 1, filter_size, output_shape = [2, 2, 1])
conv1_avg = tf.constant(np.mean(conv1, axis=3))
conv1_conv2 = tf.multiply(conv1_avg,conv2_decon)
mask = conv_2d_transpose(conv1_conv2, 1, filter_size, [20,300,1])

sess = tf.Session()
init = tf.global_variables_initializer()
result = sess.run(mask)
print(result)





# # Cats Vs Dogs using Keras with VGG
# This script is refactored version of the one available on Keras's blog. <br>
# It uses data that can be downloaded from:
# https://www.kaggle.com/c/dogs-vs-cats/data
# 
# **In our setup, we:**
# - created a data/ folder
# - created train/ and validation/ subfolders inside data/
# - created cats/ and dogs/ subfolders inside train/ and validation/
# - put the cat pictures index 0-999 in data/train/cats
# - put the cat pictures index 1000-1400 in data/validation/cats
# - put the dogs pictures index 12500-13499 in data/train/dogs
# - put the dog pictures index 13500-13900 in data/validation/dogs
# 
# So that we have 1000 training examples for each class, and 400 validation examples for each class.
# In summary, this is our directory structure:
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

import os
import numpy as np
from keras import applications, optimizers
from keras.layers import Input,Dense, Dropout, Flatten
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator


# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = os.path.join(os.getcwd(), 'fc_model.h5')
train_data_dir = os.path.join(os.getcwd(), 'data', 'cats_and_dogs_small', 'train')
validation_data_dir = os.path.join(os.getcwd(), 'data', 'cats_and_dogs_small', 'validation')
img_width, img_height = 150, 150
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 4 #more than enough to get a good result
batch_size = 16


datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
print('Reading vgg')
model = applications.VGG16(include_top=False, weights='imagenet',input_shape=(img_width, img_height, 3))
model.summary()


# Our strategy will be as follow: we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. We will then run this model on our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.
# 
# The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base and running the whole thing, is computational effiency. Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. Note that this prevents us from using data augmentation.
# 
# <img src ="https://github.com/bhavsarpratik/Deep_Learning_Notebooks/raw/master/data/images/vgg16_original.png" width="40%">
# 

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)

np.save('bottleneck_features_train.npy', bottleneck_features_train)
print('\nSaved bottleneck_features_train\n')

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // batch_size)

np.save('bottleneck_features_validation.npy',  bottleneck_features_validation)
print('\n--Saved bottleneck_features_validation--')


train_data = np.load('bottleneck_features_train.npy')

train_labels = np.array([0] * int(nb_train_samples/2 ) + [1] * int(nb_train_samples/2 ))

validation_data = np.load('bottleneck_features_validation.npy')

validation_labels = np.array([0] * int(nb_validation_samples/2 ) + [1] * int(nb_validation_samples/2))

print(train_data.shape)
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)


# To further improve our previous result, we can try to "fine-tune" the last convolutional block of the VGG16 model alongside the top-level classifier. Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates. In our case, this can be done in 3 steps:
# 
# -instantiate the convolutional base of VGG16 and load its weights
# -add our previously defined fully-connected model on top, and load its weights
# -freeze the layers of the VGG16 model up to the last convolutional block
# 
# <img src ="https://github.com/bhavsarpratik/Deep_Learning_Notebooks/raw/master/data/images/vgg16_modified.png" width="40%">
# 

# **Note that:**
# 
# -In order to perform fine-tuning, all layers should start with properly trained weights: for instance you should not slap a randomly initialized fully-connected network on top of a pre-trained convolutional base. This is because the large gradient updates triggered by the randomly initialized weights would wreck the learned weights in the convolutional base. In our case this is why we first train the top-level classifier, and only then start fine-tuning convolutional weights alongside it.
# 
# -We choose to only fine-tune the last convolutional block rather than the entire network in order to prevent overfitting, since the entire network would have a very large entropic capacity and thus a strong tendency to overfit. The features learned by low-level convolutional blocks are more general, less abstract than those found higher-up, so it is sensible to keep the first few blocks fixed (more general features) and only fine-tune the last one (more specialized features).
# 
# -Fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the magnitude of the updates stays very small, so as not to wreck the previously learned features.
# 

#Using generated model with layers of vgg

input_tensor = Input(shape=(150,150,3))
base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
print('VGG model')
base_model.summary()


top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights('fc_model.h5') #optional - to load the already saved weights
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set the first 15 layers (up to the conv block 4) to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

print('\nAugmenting train data')
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


print('\nScaling test data')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save_weights('VGG_cats_Vs_dogs.h5')


# # Amazon stock 'Close' value prediction
# 

import math, time
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np

import keras
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


get_ipython().run_cell_magic('javascript', '', "//Creating shortcut 'r' to run all the below cells\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('r', {\n    help : 'run below cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_cells_below();\n        return false;\n    }}\n);\n\n//Creating shortcut 'l' to run all the cells\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('l', {\n    help : 'run all cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);")


# ## Info on pandas_datareader
# 
# def DataReader(name, data_source=None, start=None, end=None,
#                retry_count=3, pause=0.001, session=None, access_key=None):
#     <br><br>
#     
#     Imports data from a number of online sources.
#     Currently supports Yahoo! Finance, Google Finance, St. Louis FED (FRED),
#     Kenneth French's data library, and the SEC's EDGAR Index.
#     Parameters
#     ----------
#     name : str or list of strs
#         the name of the dataset. Some data sources (yahoo, google, fred) will
#         accept a list of names.
#     data_source: {str, None}
#         the data source ("yahoo", "yahoo-actions", "yahoo-dividends",
#         "google", "fred", "ff", or "edgar-index")
#     start : {datetime, None}
#         left boundary for range (defaults to 1/1/2010)
#     end : {datetime, None}
#         right boundary for range (defaults to today)
#     retry_count : {int, 3}
#         Number of times to retry query request.
#     pause : {numeric, 0.001}
#         Time, in seconds, to pause between consecutive queries of chunks. If
#         single value given for symbol, represents the pause between retries.
#     session : Session, default None
#             requests.sessions.Session instance to be used
#     Examples
#     ----------
#     
#     Data from Yahoo! Finance
#     gs = DataReader("GS", "yahoo")
#     
#     Corporate Actions (Dividend and Split Data)
#     with ex-dates from Yahoo! Finance
#     gs = DataReader("GS", "yahoo-actions")
#     
#     Data from Google Finance
#     aapl = DataReader("AAPL", "google")
#     
#     Data from FRED
#     vix = DataReader("VIXCLS", "fred")
#     
#     Data from Fama/French
#     ff = DataReader("F-F_Research_Data_Factors", "famafrench")
#     ff = DataReader("F-F_Research_Data_Factors_weekly", "famafrench")
#     ff = DataReader("6_Portfolios_2x3", "famafrench")
#     ff = DataReader("F-F_ST_Reversal_Factor", "famafrench")
#     
#     Data from EDGAR index
#     ed = DataReader("full", "edgar-index")
#     ed2 = DataReader("daily", "edgar-index")
# 

# ## Loading Amazon stock data from google.com
# 

stock_name = 'AMZN'
start = dt.datetime(1995,1,1)
end   = dt.date.today()
df = web.DataReader(stock_name, 'google', start, end)
df.to_csv('%s_data.csv'%stock_name, header=True, index=False)
df = pd.read_csv('%s_data.csv'%stock_name)

# Dropping all columns except 'Open','High' and 'Close'
df.drop(['Low','Volume'], axis = 1, inplace=True)
df.head()


# ## Normalisation
# 

#Method1 - Division by 10
df = df/(10^(len(str(df.iloc[0,0]).split('.')[0])-1))
#Method2 - Division by 1st value
# df = df/df.iloc[0,0]
df.head()


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# ## Building model functions
# 

def build_model(layers):
    d = 0.2
    init = glorot_uniform(seed = 69)

    model = Sequential()
    model.add(LSTM(32, input_shape=(layers[0], layers[1]), return_sequences=True, kernel_initializer = init))
    model.add(Dropout(d))
    model.add(LSTM(32, return_sequences=False, kernel_initializer = init))
    model.add(Dropout(d))
    model.add(Dense(8,kernel_initializer=init ,activation='relu'))        
    model.add(Dense(1,kernel_initializer= init ,activation='linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model


# ## Setting X and Y for training and testing
# 

window = 22
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)


# ## Loading the model sequence structure
# 

model = build_model([window, 3])


# ## Executing the model & RMS/RMSE results
# 

stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta= 0.5 , patience=3, verbose=2, mode='auto')
start_time = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
#     validation_split=0.1,
#     callbacks = [stop],
    verbose = 2)

print('\nTime taken for training: %.2f minutes'%((time.time()-start_time)/60))


trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))


# ## Predictions vs Real results
# 

plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()


# ## Play with the layer sizes to get more accuracy and also check for training time
# 

