import os
import sys
import pandas as pd
import numpy as np
import PIL

seed = 16
np.random.seed(seed)

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# Purpose of this notebook is to build/test working model using a smaller subset of classes and data to minimize iteration time and to test CovNets of varying size before running on broader dataset.
# 

#check using system GPU for processing

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())


# copied over the train, validate and test sets for 5 randomly selected breeds


# Given the subset, I copied over their respesctive train/validate/test image folders from the broader image data set.  I maintained the full size of each train, val and test set.  
# 

os.chdir('C:\\Users\\Garrick\Documents\\Springboard\\Capstone Project 2\\datasets_subset1')


train_datagen = ImageDataGenerator(rotation_range=15, shear_range=0.1, channel_shift_range=20,
                                    width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                    fill_mode='nearest', rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 25

train_generator = train_datagen.flow_from_directory('subset_train', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)

validation_generator = validation_datagen.flow_from_directory('subset_val', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)


test_generator = test_datagen.flow_from_directory('subset_test', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)

# reminder to self... flow_from_directory infers the class labels


# importing keras modules and setting up a few parameters, instantiating early stopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras.utils
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)


import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
# tf_config.gpu_options.allow_growth = True **this causes python to crash, error: las.cc:444] failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
sess = tf.Session(config=tf_config)


input_shape = (224,224, 3)
num_classes = 5

# will create a few different models.... initial base model 

base_model = Sequential()
base_model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
base_model.add(Flatten())

base_model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
base_model.add(Dropout(0.2))
base_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
base_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(base_model.summary())


# train base_model

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=epochs, callbacks=[early_stopping])


# same model, more epochs (10 -> 50) and fewere steps per epoch, prior model params saw train and validate accuracies double, expecting the model to be within 80% acc

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=25, callbacks=[early_stopping])


# Looks like we are increasing in accuracy with each successive epoch, perhaps need to train for more epochs. Let's test a deeper network with the same amount of epochs and see if we can begin with a better first epoch accuracy of 15%.
# 

# taking the base model and adding more hidden layers

deep_model = Sequential()

deep_model = Sequential()
deep_model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(Flatten())

deep_model.add(Dense(288, activation='relu', kernel_constraint=maxnorm(3)))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
deep_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(deep_model.summary())


# train deeper model

deep_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=epochs, callbacks=[early_stopping])


# looks like the deeper model overfits the training data, but performs better on the validation data... let's train for more epochs

deep_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=50, callbacks=[early_stopping])


# deeper model ran into early stopping on the validation set

# lets try base model with more epochs

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=25, callbacks=[early_stopping])


# deep model with Adam optimizer

# taking the base model and adding more hidden layers



deep_model_Adam = Sequential()

deep_model_Adam = Sequential()
deep_model_Adam.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(Flatten())

deep_model_Adam.add(Dense(288, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(Dropout(0.2))
deep_model_Adam.add(Dense(num_classes, activation='softmax'))
    
# Compile model
adam_op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam.summary())


deep_model_Adam.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=25, callbacks=[early_stopping])


# tweaked deep model w/ Adam optimizer.  Deeper network topology near the input (less convolution than prior models), more FC nodes

deep_model_Adam_2 = Sequential()

deep_model_Adam_2 = Sequential()
deep_model_Adam_2.add(Conv2D(64, (8, 8), strides=2, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(Flatten())

deep_model_Adam_2.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(Dropout(0.2))
deep_model_Adam_2.add(Dense(num_classes, activation='softmax'))
    
# Compile model
adam_op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam_2.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam_2.summary())


deep_model_Adam_2.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])


# Deeper topology not necessarily better and is over-fitting.  
# 

# tweaked deep model w/ RMSProp optimizer again with Deeper network topology near the input (less convolution than prior models), more FC nodes

deep_model_RMS = Sequential()

deep_model_RMS = Sequential()
deep_model_RMS.add(Conv2D(64, (8, 8), strides=2, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(Flatten())

deep_model_RMS.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(Dropout(0.2))
deep_model_RMS.add(Dense(num_classes, activation='softmax'))
    
# Compile model
deep_model_RMS.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(deep_model_RMS.summary())


deep_model_RMS.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=50, callbacks=[early_stopping])


# so more layers doesn't work.... let us keep the standard 3 CONV layers and widen the toplogy

wide_model = Sequential()
wide_model.add(Conv2D(32, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))

wide_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))

wide_model.add(Conv2D(32, (3, 3), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))
wide_model.add(Flatten())

wide_model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
wide_model.add(Dropout(0.2))
wide_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
wide_model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(wide_model.summary())


wide_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])


# wider doesn't necessarily work... however, slowing the learning rate seems to having a positive impact. same model as above, decrease LR

wide_model_slow_learn = Sequential()
wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(Flatten())

wide_model_slow_learn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(Dropout(0.2))
wide_model_slow_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_slow_learn.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(wide_model_slow_learn.summary())


wide_model_slow_learn.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])


# it appears a slower learning rate might be key in allowing prior models to train for more epochs... 
# let's try a few earlier models with a decreased learning rate


adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam.summary())


deep_model_Adam.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])


# let's try tye base model w/ decreased learning rate and Adam optimizer (vs. SGD)

base_model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(base_model.summary())


base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])


# So far the wider model (with less stride, smaller Convolution filter) w/ more trainable parameters and the simple base model appear to perform the best on validation data.  
# 

# let's test on these iterations of the base, deep and wide models

base_scores = base_model.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (base_scores[1]*100))


deep_model_Adam_scores = deep_model_Adam.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (deep_model_Adam_scores[1]*100))


wide_model_slow_learn_scores = wide_model_slow_learn.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (wide_model_slow_learn_scores[1]*100))


# Next steps... pick top 2 or 3 models and test and note which performs best.  use top 2-3 on broader image data set (simple_CNN notebook)
# 

# saving models and weights just in case... will need to retrain on broader image sets anyways
base_model.save('subset_base_model.h5')
deep_model_Adam.save('subset_deep_model_Adam.h5')
wide_model_slow_learn.save('subset_wide_model_slow_learn.h5')








import os
import sys
import pandas as pd
import numpy as np

seed = 16
np.random.seed(seed)

from keras.utils.np_utils import to_categorical

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())


from scipy.io import loadmat
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')


# reading in the test/train data for images and labels.

train_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''')['train_data']
train_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']
test_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_data.mat''')['test_data']
test_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_list.mat''')['labels']


from sklearn.model_selection import train_test_split


print(train_data.shape)
print(train_labels.shape)


df = pd.DataFrame(train_data)
df.head()


labels = [item for label in train_labels for item in label] 
df2 = pd.DataFrame({'label':labels})

pre_split = pd.concat([df, df2], axis=1)
pre_split.head()


train, validate = train_test_split(df, test_size = 0.2, stratify=train_labels, random_state=16)


train.head()


#might have finally figured out how to stratify the image data...

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, stratify=train_labels, random_state=16)


# Note on the data.... X_train X_val are already resized from 0-255 to 0-1. No need to transform these arrays. 
# 

# transform the y_train and y_val to categoricals  ***not sure if this is needed**
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)
num_classes = y_val.shape[0]
num_classes


X_train.shape


# using a simple CNN to start

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')


input_shape = (9600, 12000, 0)

# create the model

model = Sequential()
model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(32, (11, 11), strides=4, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, (11, 11), strides=4, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
     

# Compile model
epochs = 25
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())





# Fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)
scores = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# I decided to go the route of converting images into numpy arrays for training and testing as opposed to using the Keras built-in ImageDataGenerator and FlowFromDirectory (opting instead for a single file of raw image data and discrete file of labels).  In this notebook, I will be building a CNN from scratch an leveraing the different data pipeline methodology
# 

import os
import sys
import pandas as pd
import numpy as np
import PIL

seed = 16
np.random.seed(seed)

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

import keras.utils
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)


#check using system GPU for processing and declaring system/GPU parameters

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

# configure tensorflow before fitting model
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=tf_config)


# changing directory to access data (as numpy arrays)
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')


# define functions to load data

def load_array(fname):
    return np.load(open(fname,'rb'))


# load in labels and data (as tensors)

train_labels=load_array('train_labels.npy')
valid_labels=load_array('valid_labels.npy')


train_tensor=load_array('train_dataset.npy')


def Normalize_Input(X):
    minimum=0
    maximum=255
    X-minimum/(maximum-minimum)
    return X  


train_tensor=Normalize_Input(train_tensor)


valid_tensor=load_array('valid_dataset.npy')


valid_tensor=Normalize_Input(valid_tensor)


# feeding the training data through an Image Augmentation process (including resizing and shifting tolerance)

num_classes = 120
batch_size = 12
input_shape = (224, 224, 3)

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
                             zoom_range=0.1, horizontal_flip=True)


# note to self... perhaps the imagedatagenerator parameters I had before were root cause of low accuracy...



train_generator = datagen.flow(x=train_tensor, y=train_labels, batch_size=batch_size)
validation_generator = datagen.flow(x=valid_tensor, y=valid_labels, batch_size=batch_size)


# fit the ImageDataGenerator 
datagen.fit(train_tensor)


wide_model_slow_learn = Sequential()

wide_model_slow_learn.add(BatchNormalization(input_shape=input_shape))
wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(Dropout(0.2))
wide_model_slow_learn.add(GlobalAveragePooling2D())

wide_model_slow_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_slow_learn.compile(loss='sparse_categorical_crossentropy', optimizer=adam_op, metrics=['accuracy']) 
#loss changed to sparse for new label data
print(wide_model_slow_learn.summary())


from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_v3.hdf5', 
                               verbose=1, save_best_only=True)


wide_model_slow_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=10, callbacks=[checkpointer, early_stopping])


# try faster learning rate, see if we can speed up the improvement

wide_model_fast_learn = Sequential()

wide_model_fast_learn.add(BatchNormalization(input_shape=input_shape))
wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(Dropout(0.2))
wide_model_fast_learn.add(GlobalAveragePooling2D())

wide_model_fast_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_fast_learn.compile(loss='sparse_categorical_crossentropy', optimizer=adam_op, metrics=['accuracy']) 
#loss changed to sparse for new label data
print(wide_model_fast_learn.summary())


checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5', 
                               verbose=1, save_best_only=True)

history_wmfl = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=20, callbacks=[checkpointer, early_stopping])


wide_model_fast_learn.save('saved_models/wide_model_fast_learn.h5')


# lets plot/visualize the model training progress

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')

font = {'family' : 'sans-serif',
        'weight' : 'medium',
        'size'   : 16}

plt.rc('font', **font)


print(history_wmfl.history.keys())


def plot_history(history, figsize=(8,8)):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1, figsize=figsize)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    
    ## Accuracy
    plt.figure(2, figsize=figsize)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()


plot_history(history_wmfl, figsize=(10,6))


plot_history(history_wmfl_2, figsize=(10,6))


# let's continue training and see if we can improve accuracy and loss

wide_model_fast_learn.load_weights('saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5')

history_wmfl_2 = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=20, callbacks=[checkpointer, early_stopping])


wide_model_fast_learn.save('saved_models/wide_model_fast_learn.h5')





batch_size = 64

history_wmfl_3 = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=10, callbacks=[checkpointer])


# appears this first model is going to top out in accuracy and bottom out in loss here... let's plot our progress so far

plot_history(history_wmfl_3, figsize=(12,6))


# trying new model with increase # of filters and "he_normal" kernel initializer.  "glorot_uniform" is default
# also adding dropout layer since it seems that our prior model experience overfitting 

new_model = Sequential()

new_model.add(BatchNormalization(input_shape=input_shape))
new_model.add(Conv2D(16, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())


new_model.add(Conv2D(32, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())

new_model.add(Conv2D(64, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())

new_model.add(Conv2D(128, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Dropout(0.4))
new_model.add(BatchNormalization())

new_model.add(Conv2D(256, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Dropout(0.4))
new_model.add(BatchNormalization())

new_model.add(GlobalAveragePooling2D())

new_model.add(Dense(num_classes, activation='softmax'))

new_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(new_model.summary())


checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_saksham789.hdf5', 
                               verbose=1, save_best_only=True)

batch_size = 64

history = new_model.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=100, callbacks=[checkpointer])


plot_history(history,figsize=(12,6))


new_model.save('saved_models/new_model.h5')





# continue training?


# appears dropout might be too aggresive... 

new_model_2 = Sequential()

new_model_2.add(BatchNormalization(input_shape=input_shape))
new_model_2.add(Conv2D(16, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())


new_model_2.add(Conv2D(32, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(64, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(128, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(Dropout(0.2))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(256, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(Dropout(0.2))

new_model_2.add(GlobalAveragePooling2D())

new_model_2.add(Dense(num_classes, activation='softmax'))

new_model_2.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(new_model_2.summary())


from keras.callbacks import TensorBoard


tensorboard = TensorBoard(log_dir='C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\tbLogs\\CNN_from_scratch', 
                          histogram_freq=0, batch_size=64, write_graph=True, write_grads=False, write_images=False, 
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)




batch_size = 64

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_saksham789_v3.hdf5', 
                               verbose=1, save_best_only=True)

history = new_model_2.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=100, callbacks=[checkpointer, tensorboard])


new_model_2.save('saved_models/new_model_2_v2.h5')


plot_history(history, figsize=(12,6))


# Looks like a slight improvement with a smaller dropout rate. Let's move on to using a pre-trained model and see if we can improve for additional training.... 
# 
# For now, let's evaluate how the model performs on test data.
# 

# load the  model

from keras.models import load_model

model = load_model('saved_models/wide_model_fast_learn.h5')
model.load_weights('saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5')








test_tensor=load_array('test_dataset.npy')


test_tensor = Normalize_Input(test_tensor)


test_labels =load_array('test_labels.npy')


datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
                             zoom_range=0.1, horizontal_flip=True)

test_generator = datagen.flow(x=test_tensor, y=test_labels, batch_size=batch_size)


# load the model
from keras.models import load_model

model = load_model('saved_models/new_model_2.h5')


accuracy = model.evaluate_generator(test_generator, max_queue_size=10)


print(accuracy)





# In this notebook, using VGG16 with Batch Normalization and adding own fully-connected layers.  Also, in lieu of pre-processed image data loaded into single tensors (one for train, validation and test, respectively), will revert back to the flow_from_directory method from Keras (applying to pre-cropped images as per the annotation data provided by the Stanford dataset). 
# 

import numpy as np
from keras.models import Sequential
from utils_channels_last import Vgg16BN
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K


import os

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

# configure tensorflow before fitting model
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=tf_config)


# changing directory for flow_from_directory method
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')


# ## Classifying the Dog Breed Using (Transfer Leraning)
# Here we are using  pretrained VGG16BN network, which is a VGG16 architecture with Batch Normalization.  This addition of Batch Normalization is useful in my case as we have such few samples per class and aids in overfitting.  However, this doesn't meant that overfitting is avoidable. 
# 

batch_size=12
num_classes = 120
image_size = 224
num_channels = 3


# ## IMAGE DATA  AUGMENTATION 
# Here we use data Augmentation as Described in the previous post.
# 

train_datagen = ImageDataGenerator(rotation_range=15, shear_range=0.1, channel_shift_range=20,
                                    width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                  validation_split=0.2)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory('cropped/train', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=batch_size, subset='training')

validation_generator = train_datagen.flow_from_directory('cropped/train', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=batch_size, subset='validation')

test_generator = test_datagen.flow_from_directory('cropped/test', target_size=(224,224),
            class_mode='categorical', shuffle=False, batch_size=batch_size)


base_model = Vgg16BN(include_top=False).model
x = base_model.output
x = Flatten()(x)
x = Dropout(0.4)(x)
# let's add two fully-connected layer
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
# and a final FC layer with 'softmax' activation since we are doing a multi-class problem 
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


from keras.callbacks import ModelCheckpoint


model.load_weights('saved_models/weights.vgg16_BN_finetuned.h5')


checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.pre_trained_vgg16_v3.hdf5', 
                               verbose=1, save_best_only=True)


history = model.fit_generator(train_generator, steps_per_epoch=800, epochs=25, 
                              validation_data=validation_generator,
                              callbacks=[checkpointer])


# saving what we have so far
model.save('saved_models/vgg16_BN_finetuned.h5')
model.save_weights('saved_models/weights.vgg16_BN_finetuned.h5')


# not sure if I need this later but saving the classes from the generator object and the respective indices
label_dict = train_generator.class_indices


# lets plot/visualize the model training progress

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')

font = {'family' : 'sans-serif',
        'weight' : 'medium',
        'size'   : 16}

plt.rc('font', **font)


def plot_history(history, figsize=(8,8)):
    '''
    Args: the history attribute of model.fit, figure size (defaults to (8,8))
    Description: Takes the history object and plots the loss and accuracy metrics (both train and validation)
    Returns: Plots of Loss and Accuracy from model training
    '''
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1, figsize=figsize)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    
    ## Accuracy
    plt.figure(2, figsize=figsize)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()


plot_history(history, figsize=(10,6))


# looks like we're starting to overfit, let's see if we can improve by continuing to train with a slower learning rate 
# as gains in val_loss are becoming less frequent

model.optimizer.lr = 1e6
history = model.fit_generator(train_generator, steps_per_epoch=800, epochs=15, 
                        validation_data=validation_generator,
                       callbacks=[checkpointer])





plot_history(history, figsize=(10,6))





from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import itertools


model.load_weights('saved_models/weights.vgg16_BN_finetuned.h5')


# now lets evaluate the model on our unseen test data

y_pred = model.predict_generator(test_generator, max_queue_size =10)


import pandas as pd


# create a dataframe of the predictions to find the most hallmark example of a class in the eyes of the model

results_df = pd.DataFrame(y_pred)
results_df.columns = list(label_dict.values())
results_df.head()


# find most pugly

image = results_df['pug'].idxmax()
image


# need index of all test images

folders = [x[0] for x in os.walk('test')][1:]
files = [os.listdir(f) for f in folders]


flattened_list = [y for x in files for y in x]





files = pd.DataFrame(flattened_list)
files.head()


# ok lets find the pug image

pug = files.iloc[image]
print(pug)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110958-pug\n02110958_609.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# corgi
image = results_df['Pembroke'].idxmax()
image


corgi = files.iloc[image]
print(corgi)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02113023-Pembroke\n02113023_3913.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# let's look at the most misclassified pairs siberian husky and eskimo dog

image = results_df['Siberian_husky'].idxmax()

husky = files.iloc[image]
print(husky)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110185-Siberian_husky\n02110185_13187.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


#... now eskimo dog

image = results_df['Eskimo_dog'].idxmax()

eskimo = files.iloc[image]
print(eskimo)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110185-Siberian_husky\n02110185_699.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

#interestingly the model thinks that an image from the Siberian husky folder is the most "Eskimo Dog" even out of the eskimo dog images


# entlebucher

image = results_df['EntleBucher'].idxmax()

EntleBucher = files.iloc[image]
print(EntleBucher)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108000-EntleBucher\n02108000_1462.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# Greater_Swiss_Mountain_dog
image = results_df['Greater_Swiss_Mountain_dog'].idxmax()

greater_swiss = files.iloc[image]
print(greater_swiss)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02107574-Greater_Swiss_Mountain_dog\n02107574_2665.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# Labrador_retriever, 2017 most popular breed

image = results_df['Labrador_retriever'].idxmax()

Labrador_retriever = files.iloc[image]
print(Labrador_retriever)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02099712-Labrador_retriever\n02099712_7866.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# Number 2, German_shepherd

image = results_df['German_shepherd'].idxmax()

German_shepherd = files.iloc[image]
print(German_shepherd)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02106662-German_shepherd\n02106662_16817.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# 3, golden_retriever

image = results_df['golden_retriever'].idxmax()

golden_retriever = files.iloc[image]
print(golden_retriever)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02099601-golden_retriever\n02099601_2994.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# 4, French_bulldog

image = results_df['French_bulldog'].idxmax()

French_bulldog = files.iloc[image]
print(French_bulldog)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108915-French_bulldog\n02108915_3702.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# 6, beagle (#5 english bulldog is not in database)

image = results_df['beagle'].idxmax()

beagle = files.iloc[image]
print(beagle)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02088364-beagle\n02088364_959.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# malinois

image = results_df['malinois'].idxmax()

malinois = files.iloc[image]
print(malinois)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02105162-malinois\n02105162_6596.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# cairn

image = results_df['cairn'].idxmax()

cairn = files.iloc[image]
print(cairn)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02096177-cairn\n02096177_2842.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


# vizsla

image = results_df['vizsla'].idxmax()

vizsla = files.iloc[image]
print(vizsla)


display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02100583-vizsla\n02100583_7522.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)


y_pred = np.argmax(y_pred, axis=1).astype(int)
y_pred





test_labels = np.load(open('test_labels.npy','rb'))


cm = confusion_matrix(test_labels, y_pred)

#normalize the confusion matrix

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


breeds = list(label_dict.values())


import seaborn as sns
import pandas as pd


fig, ax = plt.subplots(figsize=(30, 26))
plt.title('Confusion Matrix on Test Images')
_ = sns.heatmap(cm, ax=ax, yticklabels=breeds, xticklabels=breeds, robust=True)


accuracy = model.evaluate_generator(test_generator, max_queue_size=10)


print(accuracy)


'''

# credit to: https://gist.github.com/nickynicolson/202fe765c99af49acb20ea9f77b6255e

def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df

df = cm2df(cm, breeds)
'''


df = pd.DataFrame(y_pred, columns=['predicted'], dtype='int')


df.head()


df['actual'] = pd.Series(test_labels).astype('int')
df['count'] = 1
df['count'] = df['count'].astype('int')


df.info()


# need to create dictionary to assign the numeric labels to string labels 
# first let's reverse the order of the label_dict (again this was datagen.class_indices)

label_dict = {y:x for x,y in label_dict.items()}


for key, value in label_dict.items():
    label_dict[key] = value[10:]


df.replace({"actual": label_dict}, inplace=True)
df.replace({"predicted": label_dict}, inplace=True)


df.head(25)


# let's take a look at the top 30 most confused pairs

misclass_df = df[df['actual'] != df['predicted']].groupby(['actual', 'predicted']).sum().sort_values(['count'], ascending=False).reset_index()
misclass_df['pair'] = misclass_df['actual'] + ' / ' + misclass_df['predicted']
misclass_df = misclass_df[['pair', 'count']].take(range(30))

misclass_df.sort_values(['count'], ascending=False).plot(kind='barh', figsize=(8, 10), x=misclass_df['pair'])
plt.title('Top 30 Misclassified Breed Pairs')


new_misclass_df = df[df['actual'] != df['predicted']].groupby(['actual', 'predicted']).sum().sort_values(['count'], ascending=True).reset_index()
new_misclass_df['pair'] = new_misclass_df['actual'] + ' / ' + new_misclass_df['predicted']

new_misclass_df.tail()



new_misclass_df.tail(30).plot(kind='barh', figsize=(8, 10), x='pair', y='count', color='red')
plt.title('Top 30 Misclassified Breed Pairs')


# https://stackoverflow.com/questions/41695844/keras-showing-images-from-data-generator?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
'''
x,y = train_generator.next()
for i in range(0,5):
    image = x[i]
    plt.imshow(image.transpose(2,1,0))
    plt.show()
'''


# finally let's get the classification report in case we need it

print(classification_report(y_true=test_labels, y_pred=y_pred, target_names=list(label_dict.values())))


# Highest precision rate = Komodor
# 
# Lowest Precision rate = briard  
# 
# Highest Recall = Norwegian_elkhound   
# 
# Lowest Recall = collie 
# 
# Highest F1 = Komodor
# 
# Lowest F1 = collie
# 

# now let's visualize whats going on in the network itself as we pass images through
# adopted from Keras documentation: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

layer_dict = dict([(layer.name, layer) for layer in model.layers])


from keras import backend as K

from __future__ import print_function

from scipy.misc import imsave
import time


img_width = 224
img_height = 224


layer_name = 'conv2d_5'


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


input_img = model.input


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


get_ipython().run_cell_magic('capture', '', "\nkept_filters = []\nfor filter_index in range(200):\n\n    # we only scan through the first 200 filters,\n    # but there are actually 512 of them\n    print('Processing filter %d' % filter_index)\n    start_time = time.time()\n\n    # we build a loss function that maximizes the activation\n    # of the nth filter of the layer considered\n    layer_output = layer_dict[layer_name].output\n    if K.image_data_format() == 'channels_first':\n        loss = K.mean(layer_output[:, filter_index, :, :])\n    else:\n        loss = K.mean(layer_output[:, :, :, filter_index])\n\n    # we compute the gradient of the input picture wrt this loss\n    grads = K.gradients(loss, input_img)[0]\n\n    # normalization trick: we normalize the gradient\n    grads = normalize(grads)\n\n    # this function returns the loss and grads given the input picture\n    iterate = K.function([input_img], [loss, grads])\n\n    # step size for gradient ascent\n    step = 1.\n\n    # we start from a gray image with some random noise\n    if K.image_data_format() == 'channels_first':\n        input_img_data = np.random.random((1, 3, img_width, img_height))\n    else:\n        input_img_data = np.random.random((1, img_width, img_height, 3))\n    input_img_data = (input_img_data - 0.5) * 20 + 128\n\n    # we run gradient ascent for 20 steps\n    for i in range(20):\n        loss_value, grads_value = iterate([input_img_data])\n        input_img_data += grads_value * step\n\n        print('Current loss value:', loss_value)\n        if loss_value <= 0.:\n            # some filters get stuck to 0, we can skip them\n            break\n\n    # decode the resulting input image\n    if loss_value > 0:\n        img = deprocess_image(input_img_data[0])\n        kept_filters.append((img, loss_value))\n    end_time = time.time()\n    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))")


# we will stich the best 64 filters on a 8 x 8 grid.
n = 8


# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]


# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))



# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img


# save the result to disk
imsave('stitched_filters_%dx%d_2.png' % (n, n), stitched_filters)

# end keras tutorial


# let's visualize how the model interprets the image data. let's start using a picture that we're familiar with, Pepsi.
# adopted from https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb

import cv2


pepsi = cv2.imread(r'''C:\Users\Garrick\Downloads\pepsi.jpg''')
pepsi = cv2.cvtColor(pepsi, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(pepsi)


pepsi.shape


pepsi = cv2.resize(pepsi,(224,224))


pepsi = np.reshape(pepsi,[1,224,224,3])


pepsi.shape


pepsi = np.rollaxis(pepsi, 3, 1)


pepsi.shape

# image is now in shape appropriate for VGG16BN


pepsi_predict_proba = model.predict(pepsi)


pepsi_predict_class = np.argmax(pepsi_predict_proba)
label_dict[pepsi_predict_class]


# bummer not a good prediction, let's see how confident the model was

pepsi_top_prob = np.max(pepsi_predict_proba)
pepsi_top_prob


#let's create a function to apply these same steps over other images

def predict_image(filepath, model=model):
    '''
    Takes a single image and returns class/breed prediction and accuracy
    Dependencies: needs os, cv2 and keras, dependency file or perhaps annex to docker
    Args: filepath of image (no quotes), optional: specify the model, will default to current model in environment
    '''
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
    print('Uploaded Image...')
    plt.imshow(img)
    
    #resize and reshape for model input
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    img = np.rollaxis(img, 3, 1)
    
    #predict
    prediction = model.predict(img)
    
    #print results
    class_predict = np.argmax(prediction)
    breed = label_dict[class_predict]
    print('Woof! The model predicted the breed as...{}!'.format(breed))
    top_prob = np.max(prediction)
    print('...with a confidence of {0:.2f}%.'.format(top_prob*100))
    
          


predict_image('C:\\Users\\Garrick\\Downloads\\IMG_20180221_073001.jpg')


predict_image('C:\\Users\\Garrick\\Downloads\\pug.jpg')

#yaaaaaaaaas
# this is especially encouraging as pug/bull mastiff was the 2nd most common misclassification


predict_image('C:\\Users\\Garrick\\Downloads\\maggie.jpg')

# trick question, even my parents don't know what kind of dog Maggie is


predict_image('C:\\Users\\Garrick\\Downloads\\marnie.jpg')


predict_image('C:\\Users\\Garrick\\Downloads\\barkley.jpg')


# ok so which breed do I look like?

predict_image('C:\\Users\\Garrick\\Downloads\\1510749_671062136947_5867944268483557830_n.jpg')


# in conjunction with the human-performance study, passing in the images for the model
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02086910-papillon\n02086910_1775.jpg''')


# 2 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02109961-Eskimo_dog\n02109961_18381.jpg''')


# 3 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108000-EntleBucher\n02108000_1462.jpg''')


# 4 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02096177-cairn\n02096177_1390.jpg''')


# 5 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110063-malamute\n02110063_566.jpg''')


# 6 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02101006-Gordon_setter\n02101006_3379.jpg''')


# 7 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02106166-Border_collie\n02106166_152.jpg''')


# 8 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02086240-Shih-Tzu\n02086240_3424.jpg''')


# 10 of 10 (the 9th was Sir Charles Barkely)
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108551-Tibetan_mastiff\n02108551_1543.jpg''')





import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import toimage
from  PIL import Image

plt.rcParams['figure.figsize'] = (20.0, 10.0)

get_ipython().run_line_magic('matplotlib', 'inline')


# Image data set obtained from: http://vision.stanford.edu/aditya86/ImageNetDogs
# 
# Data formats and sizes:
# Images (757MB)
# Annotations (21MB)
# Lists, with train/test splits (0.5MB)
# Train Features (1.2GB), Test Features (850MB)
# 
# Image format is .png
# Annotations, lists, train/test features are in matlab file format
# 
# ---------
# 
# Purpose of this notebook is to explore the available stanford dog breed data set to later train and test Image Classification model.
# 
# 

# list the image folders

path = 'C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\images'

img_folder = os.listdir(path)

print('{} folders in img_folder'.format(len(img_folder)))
print('\n'.join(img_folder))


pug_folder = os.path.join(path,'n02110958-pug')

print('{} images in pug_folder'.format(len(pug_folder)))
pug_images = os.listdir(pug_folder)

os.chdir(pug_folder)

im = Image.open(pug_images[12],'r')
plt.imshow(im)


from __future__ import print_function

print(im.format, im.size, im.mode)


im = Image.open(pug_images[13],'r')
plt.imshow(im)
print(im.format, im.size, im.mode)


im = Image.open(pug_images[68],'r')
plt.imshow(im)
print(im.format, im.size, im.mode)


# check for class imbalance of labels and data

file_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\file_list.mat''')
print(file_list.keys())


df = pd.DataFrame(file_list['labels'], columns=['breed_label'], index=file_list['file_list'], dtype='int64')
df.head()


file_list['annotation_list'][0][0]
#annotation simply provides the image file name per label breed..same as file_list


df.shape


df.info()


# checking for missing labels
df['breed_label'].isnull().any()


missing_values_count = df.isnull().sum()
missing_values_count


# I conclude the train/test labels is clean and not missing any data.  The input data does not require any further examination as input data will be images (and by nature, will not have missing data).  
# 

df.nunique()


avg_num_images = 20580/120
median_num_images = df['breed_label'].value_counts().median()


# given 20,580 labels, does this correspond to the amount of images?

image_count = sum([len(files) for r, d, files in os.walk(path)])

print(image_count)


# Exploring further into the image labels...
# 


df.hist(bins=120, grid=False, figsize=(15,5))
plt.axhline(avg_num_images, color='r', linestyle='dashed', linewidth=2, label='Average # of Images')
plt.axhline(median_num_images, color='g', linestyle='dashed', linewidth=2, label='Median # of Images')
plt.legend()
plt.title('Count of Images per Breed Label')

plt.show()


# The list/annotation data is for the entire dataset (has not been split into train, validate or test partitions).  Will need to take correct proportions of each breed (can't shuffle).  Also given the distribution of images per breed_label, does not seem we have a major class imbalance.  However, the data seems a bit light at just 150-250 images per breed_label.  
# 

#repeat the process with loading and referencing images and display in subplot/grid
# currently getting Errno13, permission denied. Skip for now...
"""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        for img in os.listdir(filename):
            img = Image.open(os.path.join(folder, filename), mode='r')
            images.append(img)
    return images

image_set = load_images_from_folder(path)
"""
#root_folder = '[whatever]/data/train'
# use path variable
#folders = [os.path.join(path, x) for x in img_folder]
#all_images = [img for folder in folders for img in load_images_from_folder(folder)]


train_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''')
print(train_data.keys())


train_data





train_data['train_fg_data'][0]
print(len(train_data['train_fg_data'][0]))


train_data['train_data'][0]
print(len(train_data['train_data'][0]))


#what exactly is the train_info values comprised of?
train_data['train_info']


len(train_data['train_info'])
#so it's a single array... of multiple arrays


file_list = train_data['train_info'][0][0][0] #array of array of arrays...
annotation_list = train_data['train_info'][0][0][1]
labels = train_data['train_info'][0][0][2]
fg_ids = train_data['train_info'][0][0][3]

print(file_list.shape)
print(annotation_list.shape)
print(labels.shape)
print(fg_ids.shape)


print(labels[0][0].shape)


train_info = pd.DataFrame(train_data['train_info'][0])
train_info.head()





#define NP arrays to construct DF will need to do this for the test data too

def mat_to_df(filepath, struc_label):
    matfile = loadmat(filepath)
    file_list_arr = matfile[struc_label][0][0][0]
    annot_list_arr = matfile[struc_label][0][0][1]
    labels_arr = matfile[struc_label][0][0][2]
    fg_id_arr = matfile[struc_label][0][0][3]
    
    data = np.array([annot_list_arr, labels_arr, fg_id_arr])
    
    df = pd.DataFrame(data)
    return df


#train_data = mat_to_df(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''', 'train_info')
#test_data = mat_to_df(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_data.mat''', 'test_info')


'''
def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files   
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """
    # Subset dictionary to limit keys to print.  Only works on first level
   
    if nkeys>0:
        d = {k: d[k] for k in d.keys()[:nkeys]} # Dictionary comprehension: limit to first nkeys keys.
    
    if isinstance(d, dict): 
        for key, value in d.iteritems(): # iteritems loops through key, value pairs
            print('\t' * indent + 'Key: ' + str(key)) 
            print_mat_nested(value, indent+1)
            
    if isinstance(d,np.ndarray) and d.dtype.names is not None:
        for n in d.dtype.names:  
            print('\t' * indent + 'Field: ' + str(n))
            print_mat_nested(d[n], indent+1)
''' 
#not used


# What we need for model is X_train, y_train, X_test and y_test.
# 
# X_train: images as defined by train_data (need to obtain this as list), fed through image pre-processing into array data object.
# y_train: labels, obtained from the matlab train_data file
# X_test: images as defined by test_data (need to obtain this as list), fed through image pre-processing into array data object.
# y_test: labels, obtained from the matlab test_data file
# 

#I realized I did not unzip/unpack a file which contains the list of stratified train/test splits.

train_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['file_list']
test_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_list.mat''')['file_list']


len(train_list)


len(test_list)


#creating mock variables for TF


train_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']
test_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']



