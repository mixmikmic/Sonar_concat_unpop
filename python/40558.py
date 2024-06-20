from __future__ import print_function
import re
import string
import collections
import math
import numpy as np
import os
import nltk
import random
from nltk.tokenize import RegexpTokenizer
import json
import pandas as pd

from keras.preprocessing import sequence
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, Merge, Highway
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import keras.optimizers
from keras.regularizers import l2, l1
from sklearn.cross_validation import train_test_split, KFold


# ### Loading pretrained character embeddings
# 

header = ['Char']
for i in range(64):
    header.append('X' + str(i+1))


embeddings = pd.read_csv('char_embeddings_d150_tr1e6_w2_softmax_adagrad_spaces.csv', names=header)
embeddings_dictionary = {}


for i in xrange(len(embeddings)):
    vec = []
    for j in xrange (64):
        vec += [embeddings['X' + str(j+1)][i]]
    embeddings_dictionary[unicode(embeddings['Char'][i], 'utf8')] = vec
embeddings_dictionary[' '] = embeddings_dictionary['_']

class Embeddings_Reader(dict):
         def __missing__(self, key):
            return embeddings_dictionary[u'UNK']
        
embeddings_lookup = Embeddings_Reader(embeddings_dictionary)


# ### Loading training data and extracting features and labels
# 

def stops(char):
    stop = "\?\!\."
    m = re.search(r'^[{0}]$'.format(stop), char)
    return m != None


yandex_corpus = pd.read_csv('./1mcorpus/corpus.en_ru.1m.ru' , sep='##%##', names = ['sentence'])


first_sentences = list(yandex_corpus['sentence'])
stops_data = collections.deque([])
pointer = 0
radius = 7
window_size = 2*radius+1
sliding_window = collections.deque([], maxlen = window_size)
dot_features = []

for i in xrange(len(first_sentences)):
    
    initial_pointer = 0    
    sentence = [' '] + list(unicode (first_sentences[i], 'utf8'))    
    
    if len(sliding_window) < window_size:
        for charnum in range(len(sentence)):
            if (charnum == len(sentence) - 1) & stops(sentence[charnum]):
                sliding_window.append(sentence[charnum] + u'#')
            else:
                sliding_window.append(sentence[charnum])
            pointer += 1
            initial_pointer += 1
            if pointer == window_size:
                break
    
    if pointer < window_size:
        continue
    
    for charnum in range (initial_pointer, len(sentence)):
        if stops(sliding_window[radius][0]):                        
            dot_features = list(sliding_window)[:radius] + list(sliding_window)[-radius:]
            if (len (sliding_window[radius]) == 2):
                label = 0
            else:                
                label = 1
            vec_features = map (lambda x: embeddings_lookup[x[0]], dot_features)            
            stops_data.append((label, vec_features))
        if (charnum == len(sentence) - 1) & stops(sentence[charnum]):
            sliding_window.append(sentence[charnum] + u'#')
        else:
            sliding_window.append(sentence[charnum])    
    if i % 100000 == 0:        
        print('Iteration %d : Length of the data set is %d' % (i, len(stops_data)))        


#Number of nonbreaking stop characters in the dataset
counter = 0
for i in range (len(stops_data)):
    if stops_data[i][0] == 1:
        counter +=1
print (counter)


# ### Initial Validation run with random train/tast split
# 

labels, features = zip (*stops_data)


data_train, data_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.10, random_state=42)


X_train = np.array(data_train, dtype='float32')
X_test = np.array(data_test, dtype='float32')

y_train = np.array(labels_train)
y_test = np.array(labels_test)


model = Sequential()
model.add(Flatten(input_shape = X_train[0].shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


batch_size = 100
stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, callbacks= [stop], shuffle=True,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print ('\n')
print('Validation score :', score)
print('Validation accuracy :', acc)


# early stop monitored value should be changed to the validation loss and the patience should be decreased
# 

# ### 10-fold Cross-Validation
# 

splitpoints = range (0,len(stops_data),len(stops_data)/10)


batches = []
stops_data = list (stops_data)
random.shuffle(stops_data)
i_prev = 0
for i in splitpoints[1:]:
    batches.append (stops_data[i_prev:i])
    i_prev = i


validation_training = []
validation_test = []
indices = range (len(batches))
for i in indices:
    test = batches[i]
    validation_test.append(test)
    
    training = []
    training_indices = list (indices)
    training_indices.remove(i)
    
    for j in training_indices:        
        training += batches[j]
    
    validation_training.append(training)


cv_results = []
for i in range (len(validation_test)):
    test_data = validation_test[i]
    train_data = validation_training[i]
    
    print ('Training step:', i, '...' )    
    
    labels_train, features_train = zip (*train_data)
    labels_test, features_test = zip (*test_data)

    X_train = np.array(features_train, dtype='float32')
    X_test = np.array(features_test, dtype='float32')

    y_train = np.array(labels_train)
    y_test = np.array(labels_test)    
    
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    batch_size = 100
    
    model = Sequential()
    model.add(Flatten(input_shape = X_train[0].shape))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, callbacks= [stop], shuffle=True,  verbose=0,
              validation_data=(X_test, y_test))
    
    print ('Done.')

    _, acc = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)

    cv_results.append(acc)
    print ('Step', i, 'accuracy:', acc)
    print ('-----------------------------------\n')
    del model

cross_val = np.mean(cv_results)
print ('10-Fold Cross-Validation accuracy is:', cross_val)


# ### Final model training
# 

random.shuffle(stops_data)
labels, features = zip (*stops_data)
data_train, data_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.01, random_state=42)
X_train = np.array(data_train, dtype='float32')
X_test = np.array(data_test, dtype='float32')

y_train = np.array(labels_train)
y_test = np.array(labels_test)


model = Sequential()
model.add(Flatten(input_shape = X_train[0].shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


batch_size = 100
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, callbacks= [stop], shuffle=True,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print ('\n')
print('Validation score :', score)
print('Validation accuracy :', acc)


json_string = model.to_json()
name_ = './Models/Keras_boundary_nn_model_r7_l40_l10_l1'
model_name = name_ + '.json'
open (model_name, 'w').write(json_string)
weights_name = name_ + '_weights.h5'
model.save_weights(weights_name)


# ### Testing on the Opencorpora data
# 

corpus = pd.read_csv('opencorpora.csv')


first_sentences = list(corpus['sentence'])
stops_opencorp_data = collections.deque([])
pointer = 0
radius = 7
window_size = 2*radius+1
sliding_window = collections.deque([], maxlen = window_size)
dot_features = []

for i in xrange(len(first_sentences)-1):
    
    initial_pointer = 0    
    sentence = [' '] + list(unicode (first_sentences[i], 'utf8'))    
    
    if len(sliding_window) < window_size:
        for charnum in range(len(sentence)):
            if (charnum == len(sentence) - 1) & stops(sentence[charnum]):
                sliding_window.append(sentence[charnum] + u'#')
            else:
                sliding_window.append(sentence[charnum])
            pointer += 1
            initial_pointer += 1
            if pointer == window_size:
                break
    
    if pointer < window_size:
        continue
    
    for charnum in range (initial_pointer, len(sentence)):
        if stops(sliding_window[radius][0]):
            dot_features = list(sliding_window)[:radius] + list(sliding_window)[-radius:]
            if (len (sliding_window[radius]) == 2):
                label = 0
            else:
                label = 1
            vec_features = map (lambda x: embeddings_lookup[x[0]], dot_features)                            
            stops_opencorp_data.append((label, vec_features))
        if (charnum == len(sentence) - 1) & stops(sentence[charnum]):
            sliding_window.append(sentence[charnum] + u'#')
        else:
            sliding_window.append(sentence[charnum])    
    if i % 10000 == 0:
        print('Iteration %d : Length of the data set is %d' % (i, len(stops_opencorp_data)))


counter = 0
for i in range (len(stops_opencorp_data)):
    if stops_opencorp_data[i][0] == 1:
        counter +=1
print (counter)


labels_op, features_op = zip (*stops_opencorp_data)


X_test_op = np.array(features_op, dtype='float32')

y_test_op = np.array(labels_op)


score_op, acc_op = model.evaluate(X_test_op, y_test_op, batch_size=1000)
print ('\n')
print('Test score :', score_op)
print('Test accuracy :', acc_op)


model = model_from_json(open('/home/mithfin/anaconda2/docs/Wikiproject/Models/Keras_boundary_nn_model_r7_l40_l10_l1.json').read())
model.load_weights('/home/mithfin/anaconda2/docs/Wikiproject/Models/Keras_boundary_nn_model_r7_l40_l10_l1_weights.h5')


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])








