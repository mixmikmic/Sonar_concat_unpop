# # Predictive Maintenance
# 
# ## Step 3: Predict the results
# 
# ### Prerequisites
# 
# For this notebook you need to install:
# 
# - `pandas`
# - `numpy`
# - `matplotlib`
# - `keras`
# - `os`
# 
# ### What does this file do ?
# 
# This file constitues the final step of teh process. It predicts the RUL for the test set, based on the previously trained Neural Network.
# 
# ### When do I need to run it ?
# 
# This file extracts the test data from the `..\input\` folder. It requires a `test.csv` file, that can be created with the `preprocessing.py` file. It also requires a model, found in `output_path` and that can be created with the `trainNN.py` file.
# 
# Thus, this script is necessary to predict the results. In case a new test set is available, you need to run first `preprocessing.py` and then this file. It is not required to train the Neural Network (i.e. `trainNN.py` file) if the train set doesn't change.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import load_model
import keras.backend as K


# Path from where to retrieve the model output file
output_path = 'model/regression_model_v0.h5'
sequence_length = 50


test_data = pd.read_csv("input/test.csv")


# ## Evaluate on Test Data
# 
# We predict the RUL with the previously trained Neural Network.
# 
# ### 1. Shape the features
# 
# For the same reasons as for the train set, the test set needs to be reshaped.
# 

n_turb = test_data['id'].unique().max()

# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# We pick the last sequence for each id in the test data
seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

print("This is the shape of the test set: {} turbines, {} cycles and {} features.".format(
    seq_array_test_last.shape[0], seq_array_test_last.shape[1], seq_array_test_last.shape[2]))

print("There is only {} turbines out of {} as {} turbines didn't have more than {} cycles.".format(
    seq_array_test_last.shape[0], n_turb, n_turb - seq_array_test_last.shape[0], sequence_length))


# ### 2. Selecting and reshaping the labels
# 
# Similarly, we pick the labels.
# 

y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)


# ### 3. Predicting the RUL for test data
# 

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# if best iteration's model was saved then load and use it
if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last
    
    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    print('\nMSE: {}'.format(scores_test[0]))
    print('\nMSE: {}'.format(scores_test[1]))
    print('\nMAE: {}'.format(scores_test[2]))
    
    s1 = ((y_pred_test - y_true_test)**2).sum()
    moy = y_pred_test.mean()
    s2 = ((y_pred_test - moy)**2).sum()
    s = 1 - s1/s2
    print('\nEfficiency: {}%'.format(s * 100))

    test_set = pd.DataFrame(y_pred_test)
    test_set.to_csv('output/submit_test.csv', index = None)


if os.path.isfile(output_path):
    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(60, 30))
    # plt.plot(y_pred_test, 'ro', color="red", lw=3.0)
    # plt.plot(y_true_test, 'ro', color="blue")
    X = np.arange(1, 94)
    width = 0.35
    plt.bar(X, np.array(y_pred_test).reshape(93,), width, color='r')
    plt.bar(X + width, np.array(y_true_test).reshape(93,), width, color='b')
    plt.xticks(X)
    plt.title('Remaining Useful Life for each turbine')
    plt.ylabel('RUL')
    plt.xlabel('Turbine')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    fig_verify.savefig("output/model_regression_verify.png")





# # Predictive Maintenance
# 
# ## Step 1: Preprocessing
# 
# ### Prerequisites
# 
# For this notebook you need to install:
# 
# - `pandas`
# - `numpy`
# - `sklearn`
# 
# The easiest way is to install these libraries with `pip`, which is the python package installation tool.
# You can simply use
# 
#     pip install pandas
#     pip install numpy
#     pip install sklearn
# 
# which should install everything easily.
# 
# ### What does this file do ?
# 
# This file constitues the first step of the predicitive maintenance process for the NASA turbines dataset.
# 
# ### When do I need to run it ?
# 
# This file extracts the raw data from the `..\input\` folder and preprocess it in order to create useable data for the rest of the process.
# 
# Thus, this script is necessary each time raw data is changed (new train set, new test set or new real-world-based test failure observation set).
# 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ## 1. Data Ingestion
# 
# In this step, data is extracted from txt files.
# 
# This dataset contains records of NASA turbines. The train set holds the engine run-to-failure data. The test set holds the engine operating data without failure events recorded. Finally, the truth set contains the information of true remaining cycles for each engine in the testing data.
# 

names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read training data
train_data = pd.read_csv('input/TrainSet.txt', sep=" ", header=None)
train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
train_data.columns = names

train_data = train_data.sort_values(['id','cycle'])

# read test data
test_data = pd.read_csv('input/TestSet.txt', sep=" ", header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.columns = names

# read ground truth data
truth_df = pd.read_csv('input/TestSet_RUL.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)


print("This is the size of the train dataset: {} entries and {} features".format(train_data.shape[0], 
                                                                                 train_data.shape[1]))
print("This is the size of the test dataset: {} entries and {} features".format(test_data.shape[0],
                                                                                test_data.shape[1]))
print("This is the size of the truth dataset: {} entries and {} features".format(truth_df.shape[0],
                                                                                 truth_df.shape[1]))


n_turb = train_data["id"].unique().max()
n_train, n_features = train_data.shape
print("There is {} turbines in each dataset".format(n_turb))


# ## 2. Data Preprocessing
# 
# This step adds new features to train and test set, which will constitutes the labels for the coming prediction algorithms.
# 
# ### 2.1 Train Set
# 
# For this train set, we calculate the Remaining Useful Life (RUL) for each cycle of each turbine.
# 
# Then, we generate labels for a hypothetical binary classification, while trying to answer the question: is a specific engine going to fail within n cycles ? These labels aren't used in the following learning algorithms, but could be useful for a (future ?) binary classification step.
# 

# Data Labeling - generate column RUL
rul = pd.DataFrame(train_data.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_data = train_data.merge(rul, on=['id'], how='left')
train_data['RUL'] = train_data['max'] - train_data['cycle']
train_data.drop('max', axis=1, inplace=True)

# generate label columns
w1 = 30
w0 = 15
train_data['label1'] = np.where(train_data['RUL'] <= w1, 1, 0 )
train_data['label2'] = train_data['label1']
train_data.loc[train_data['RUL'] <= w0, 'label2'] = 2


# As the values of the different features are widely scattered, it is interesting to normalize them. Here, we use the min-max normalisation to perform it.
# 
# Only the settings and the parameters are normalized (in place), as well as the cycle's number (in an other column). The other variables are left untouched.
# 

# MinMax normalization (from 0 to 1)
train_data['cycle_norm'] = train_data['cycle']
cols_normalize = train_data.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = MinMaxScaler()
norm_train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data[cols_normalize]),
                               columns=cols_normalize, index=train_data.index)
join_data = train_data[train_data.columns.difference(cols_normalize)].join(norm_train_data)
train_data = join_data.reindex(columns = train_data.columns)

print("The size of the train data set is now: {} entries and {} features.".format(train_data.shape[0],
                                                                                  train_data.shape[1]))

train_data.to_csv('input/train.csv', encoding='utf-8',index = None)
print("Train Data saved as input/train.csv")


# ## 2.2 Test Set
# 
# The process is similar to the train set one.
# 
# However, the RUL is calculated based on the values in the truth data set.
# 

# MinMax normalization (from 0 to 1)
test_data['cycle_norm'] = test_data['cycle']
norm_test_data = pd.DataFrame(min_max_scaler.transform(test_data[cols_normalize]),
                              columns=cols_normalize, index=test_data.index)
test_join_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
test_data = test_join_data.reindex(columns = test_data.columns)
test_data = test_data.reset_index(drop=True)

# generate RUL
rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)
test_data = test_data.merge(truth_df, on=['id'], how='left')
test_data['RUL'] = test_data['max'] - test_data['cycle']
test_data.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_data['label1'] = np.where(test_data['RUL'] <= w1, 1, 0 )
test_data['label2'] = test_data['label1']
test_data.loc[test_data['RUL'] <= w0, 'label2'] = 2

print("The size of the test data set is now: {} entries and {} features.".format(test_data.shape[0],
                                                                                 test_data.shape[1]))

test_data.to_csv('input/test.csv', encoding='utf-8',index = None)
print("Test Data saved as input/test.csv")





# # Predictive Maintenance
# 
# ## Step 2: Train Neural Network
# 
# ### Prerequisites
# 
# For this notebook you need to install:
# 
# - `pandas`
# - `numpy`
# - `keras`
# - `matplotlib.pyplot`
# 
# ### What does this file do ?
# 
# This file builds and fit the Neural Network algorithm to the train data set. It is imperative that a `train.csv` file has been built with the `preprocess.py` script.
# 
# ### When do I need to run it ?
# 
# This file extracts preprocessed train data from the `..\input\` folder and fit the Neural Network with it.
# 
# Thus, this script is needed if the train set is modified. Be careful, it requires computer ressources and time to process.
# 

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# where to save the model output file
output_path = 'model/regression_model_v0.h5'


train_data = pd.read_csv('input/train.csv')
n_turb = train_data['id'].unique().max()


# ## 1. Shape the train set
# 
# In order to correctly train the Neural Network, we need to reshape the data.
# 
# The point is to obtain a train set that has shape (n_turbines, n_cycles, n_features).
# 

# pick a large window size of 50 cycles
sequence_length = 50

# function to reshape features into (samples, time steps, features) 
def reshapeFeatures(id_df, seq_length, seq_cols):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    An alternative would be to pad sequences so that
    we can use shorter ones.
    
    :param id_df: the data set to modify
    :param seq_length: the length of the window
    :param seq_cols: the columns concerned by the step
    :return: a generator of the sequences
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator for the sequences
feat_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, sequence_cols)) 
           for id in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)

print("The data set has now shape: {} entries, {} cycles and {} features.".format(feat_array.shape[0],
                                                                                  feat_array.shape[1],
                                                                                  feat_array.shape[2]))


# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)


# ## 2. Modeling
# 
# Now that the train data set is corrected, we can build and fit the Recurrent Neural Network.
# 
# ### 2.1 Create a Recurrent Neural Network
# 
# In order to take into account the dependancy of a given Time Serie (i.e. for a turbine), we use LSTM (Long Short Term Memory) Recurrent Neural Network.
# 
# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units and a LSTM layer with 25 units. 
# 
# Dropout is also applied after each LSTM layer to control overfitting. 
# 
# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
# 
# The model is refined using `mean_squared_error` loss function.
# 
# The metrics used are `root_mean_squared_error` as well as `mae` (i.e. Mean Absolute Error). The `root_mean_squared_error` is easier to understand the `mean_squared_error`, as it is closer to the real value of the data.
# 

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=100, return_sequences=True, name="lstm_0"))
model.add(Dropout(0.2, name="dropout_0"))
model.add(LSTM(units=50, return_sequences=True, name="lstm_1"))
model.add(Dropout(0.2, name="dropout_1"))
model.add(LSTM(units=25, return_sequences=False, name="lstm_2"))
model.add(Dropout(0.2, name="dropout_2"))
model.add(Dense(units=nb_out, name="dense_0"))
model.add(Activation("linear", name="activation_0"))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae'])

print(model.summary())


# ### 2.1 Train the Recurrent Neural Network
# 
# In this step, we fit the Neural Network to the train set.
# 
# The `epochs` and `batch_size` are hyper-parameters of the Neural Network.
# 
# The `callbacks` parameter in the `fit` function allows to speed up the process and register the model.
# 

epochs = 100
batch_size = 200

# fit the network
history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                     verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',
                                                       save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))


# ## Plot the results
# 
# We then plot the results of the simulation.
# 

# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_mae.png")

# summarize history for RMSE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_rmse.png")

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_regression_loss.png")





