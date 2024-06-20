# ## MNIST Dataset Introduction
# Most examples are using MNIST dataset of handwritten digits. It has 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image, so each sample is represented as a matrix of size 28x28 with values from 0 to 1.
# ### Overview
# MNIST Digits
# <img src="https://camo.githubusercontent.com/b06741b45df8ffe29c7de999ab2ec4ff6b2965ba/687474703a2f2f6e657572616c6e6574776f726b73616e64646565706c6561726e696e672e636f6d2f696d616765732f6d6e6973745f3130305f6469676974732e706e67" />
# ### Usage
# In our examples, we are using TensorFlow input_data.py script to load that dataset. It is quite useful for managing our data, and handle:
# - Dataset downloading
# - Loading the entire dataset into numpy array:
# 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)


# # Basic Classification Example with TensorFlow
# Supervised problem
# 

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Import or generate data.
# All of our machine learning algorithms will depend on data. In this book we will either generate data or use an outside source of data. Sometimes it is better to rely on generated data because we will want to know the expected outcome.
# 

dataframe = pd.read_csv('input/data.csv')

dataframe = dataframe.drop(['index', 'price', 'sq_price'],axis=1)
dataframe = dataframe[0:10]
print (dataframe)


# Add labels
# 1 is good buy and 0 is bad buy
dataframe.loc[:, ('y1')]  = [1,0,1,1,0,1,1,0,0,1]

#y2 is the negation of the y1
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0

# convert true/false value to 1s and 0s
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)
dataframe


inputX = dataframe.loc[:, ['area','bathrooms']].as_matrix()
inputY = dataframe.loc[:, ['y1','y2']].as_matrix()
inputX


# ### Set algorithm parameters.
# Our algorithms usually have a set of parameters that we hold constant throughout the procedure. For example, this can be the number of iterations, the learning rate, or other fixed parameters of our choosing. It is considered good form to initialize these together so the reader or user can easily find them.
# learning_rate = 0.01  iterations = 1000
# 

# Hyperparameters
learning_rate = 0.00001
training_epochs = 2000 #iterations
display_steps = 50
n_samples = inputY.size


# ### Initialize variables and placeholders.
# Tensorflow depends on us telling it what it can and cannot modify. Tensorflow will modify the variables during optimization to minimize a loss function. To accomplish this, we feed in data through placeholders. We need to initialize both of these, variables and placeholders with size and type, so that Tensorflow knows what to expect.
# a_var = tf.constant(42)  x_input = tf.placeholder(tf.float32, [None, input_size])  y_input = tf.placeholder(tf.fload32, [None, num_classes])
# 

# Create our computation graph/ Neural Network
# for feature input tensors, none means any numbers of examples
# placgeholder are gateways for data into our computation
x = tf.placeholder(tf.float32, [None,2])

#create weights
# 2 X 2 float matrix, that we'll keep updateing through the training process
w = tf.Variable(tf.zeros([2,2]))

#create bias , we have 2 bias since we have two features
b = tf.Variable(tf.zeros([2]))

y_values = tf.add(tf.matmul(x,w),b)

y = tf.nn.softmax(y_values)

# For trainign purpose, we'll also feed a matrix of labels
y_ = tf.placeholder(tf.float32, [None,2]) 


# ### Define the model structure.
# After we have the data, and initialized our variables and placeholders, we have to define the model. This is done by building a computational graph. We tell Tensorflow what operations must be done on the variables and placeholders to arrive at our model predictions. We talk more in depth about computational graphs in chapter two, section one of this book.
# y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)
# ### Declare the loss functions.
# After defining the model, we must be able to evaluate the output. This is where we declare the loss function. The loss function is very important as it tells us how far off our predictions are from the actual values. The different types of loss functions are explored in greater detail in chapter two, section five.
# loss = tf.reduce_mean(tf.square(y_actual – y_pred))
# 

#Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# ### Initialize and train the model.
# Now that we have everything in place, we create an instance or our graph and feed in the data through the placeholders and let Tensorflow change the variables to better predict our training data. Here is one way to initialize the computational graph.
# with tf.Session(graph=graph) as session:
#  ...
#  session.run(...)
#  ...
# Note that we can also initiate our graph with
# session = tf.Session(graph=graph)  session.run(…)
# 

#Initialize variables and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# ### Evaluate the model.
# Once we have built and trained the model, we should evaluate the model by looking at how well it does on new data through some specified criteria.
# 

# Now for the actual training 
for i in range(training_epochs):
    #Take a gradient descent step using our input and labels 
    sess.run(optimizer, feed_dict={x: inputX, y_:inputY})
     
    # Display logs per epoch step   
    if (i)  % display_steps == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print("Training step:", '%04d' % (i), "Cost: :", "{:.9f}".format(cc) )
    
print("Optimization Done!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')


# ### Predict new outcomes.
# It is also important to know how to make predictions on new, unseen, data. We can do this with all of our models, once we have them trained.
# 

sess.run(y, feed_dict={x:inputX})


# # Decision Tree
# 
# ### Introduction
# In this we are considering simple Supervises learning example in which we show implementation of the Decision tree in which we differentiate **Orange** and **Apple** from the data set. We are using **sklearn** which provide us with ML alogorithms. 
# 

from  sklearn import tree


# ### Data
# Features here are **Length** and **Texture** of the Apple / Orange, these features values are define by us. Just to show how decision tree classifier works. Moreover, there are two types of Labels **Apple** And **Orange**
# 

features = [[140, 1] , [130, 1] , [150, 0] , [170, 0]];
labels = [1, 1 , 0 , 0];


# ### Classifying
# After initializing the custom data, we are using Decision Tree Classifier to train on **Features** and **Labels**.
# 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)


# In the end, we use one example and predict it using the train classifier. 
# 


print (clf.predict([[160, 0]]))


# # Titanic: Machine Learning from Disaster
# ## Introduction
# This example is taken from the Kaggle to get deeper knowledge of Machine Learning and Data Science
# https://www.kaggle.com/c/titanic
# 
# ## Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


train = pd.read_csv('http://localhost:8888/files/input/train.csv')
test = pd.read_csv('http://localhost:8888/files/input/test.csv')


print ("Dimension of train data {}".format(train.shape))
print ("Dimension of test data{}".format(test.shape))


print ("Basic statistical description:")
train.describe()


# ### Comparison between test and train data
# From following cells, we could know that train and test data are split by PassengerId.
# 

train.tail()


test.head()


# Let's look at data graphically. We could see that all the distribution of features are similar.
# 

## Plotting data here 
## https://www.kaggle.com/amanullahtariq/titanic/exploratory-tutorial-titanic-disaster/editnb

plt.rc('font', size=13)
fig = plt.figure(figsize=(18, 8))
fig.show()


# # Decision Tree Regression
# Implementation of http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
# 

from sklearn import tree
X = [[0 , 0], [2 , 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y)
clf.predict([[1, 6]])


print(__doc__)
# Import the necessary modules and libraries 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2);
regr_2 = DecisionTreeRegressor(max_depth=5);
regr_1.fit(X, y);
regr_2.fit(X, y);

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# # Linear Regression Using TensorFlow
# 

import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


x_data = np.random.rand(100).astype(np.float32)


# #### Equation for Y = 3X + 2
# 

y_data = 3 * x_data + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0,scale=0.1))(y_data)


zipped = zip(x_data,y_data)

print(zipped)


w = tf.Variable(1.0)
b = tf.Variable(0.2)
y = w * x_data + b


loss = tf.reduce_mean(tf.square(y - y_data))


optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


train_data = []
for step in range(1000):
    evals = sess.run([train, w, b])[1:]
    if step % 50 == 0:
        print(step,evals)
        train_data.append(evals)


converter = plt.colors 
cr,cg, cb = (1.0,1.0,0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0 : cb = 1.0
    if cg < 0.0 : cg = 0.0
    [w,b] = f
    
    f_y = np.vectorize(lambda x: w*x + b) (x_data)
    line = plt.plot(x_data,f_y)
    plt.setp(line,color=(cr,cg,cb))
    
plt.plot(x_data,y_data,'ro')

green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()


# #### Import dataset 
# Importing the iris dataset and then partitioning it into test and training dataset
# 

from sklearn import datasets
iris = datasets.load_iris()

#print(iris)
# X = inputs for the classifier
X = iris.data

# y = ouput 
y = iris.target
print(y.size)


# We can either manually partition dataset into test and training dataset or either use cross validation
from sklearn.cross_validation import train_test_split

#help(train_test_split)
# Using half of the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# ### Decision Tree Classifier
# First using decision tree as our classifier to train the data and then predict the output by using the test data
# 

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)



# Checking accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# ## K-Neighbors Classifier
# 

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


# Checking accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# ## Reference:
# 1) Let’s Write a Pipeline - Machine Learning Recipes
# https://www.youtube.com/watch?v=84gqSbLcBFE
# 

# # Deep MNIST for Experts
# 
# ## Load MNIST Data
# 
# If you are copying and pasting in the code from this tutorial, start here with these two lines of code which will download and read in the data automatically:
# 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ## Start TensorFlow InteractiveSession
# 

import tensorflow as tf
sess = tf.InteractiveSession()


# ## Build a Softmax Regression Model
# 
# In this section we will build a softmax regression model with a single linear layer. In the next section, we will extend this to the case of softmax regression with a multilayer convolutional network.
# 

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# ### Variables
# We now define weights **W** and **b** of our model.
# 

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


sess.run(tf.global_variables_initializer())


y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# ## Train the Model
# 

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# ## Improving
# 91% accuracy is not too good. Now, we will try to add some convolutional neural network to improve accuracy to 99.2%.
# 

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# ## Convolution and Pooling
# 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# ## First Convolutional Layer
# 

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1,28,28,1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# ## Second Convolutional Layer
# 

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# ## Densely Connected Layer
# 

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# ## Dropout
# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.
# 
# For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks.
# 

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# ## Readout Layer
# 

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# ## Train and Evaluate the Model
# 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

# By default, the Saver handles every Variables related to the default graph
all_saver = tf.train.Saver() 

for i in range (20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        
    train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})

all_saver.save(sess, "model")
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# ## Reference
# * https://www.tensorflow.org/get_started/mnist/pros
# 

# # Sentiment Analyzer For Movie Reviews
# 
# The scraping process took 2 hours to finish. In the end, I was able to obtain all needed 28 variables for 5043 movies and 4906 posters (998MB), spanning across 100 years in 66 countries. There are 2399 unique director names, and thousands of actors/actresses. Below are the 28 variables:
# 
# 

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


help(tflearn.input_data)


#IMDB Dataset loading
#load_data in module tflearn.datasets.imdb:
#extension is picke which means a bit steam, so it make it easier to convert to other 
#python objects like list, tuple later 
#param n_words: The number of word to keep in the vocabulary.All extra words are set to unknow (1).
#param valid_portion: The proportion of the full train set used for the validation set.

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test


# Data preprocessing
# Sequence padding
#function pad_sequences in module tflearn.data_utils:
# padding is necessary to have consistency in our input dimensionality
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#Now we can convert output to binaries where 0 means Negative and 1 means Positive
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


print(trainX[1:10,:])
print(trainX[0,:].shape)
type(trainX[0,0])


# Network building
#function input_data in module tflearn.layers.core:
#`input_data` is used as a data entry (placeholder) of a network.
#This placeholder will be feeded with data when training
# shape: list of `int`. An array or tuple representing input data shape.
#            It is required if no placeholder provided. First element should
#            be 'None' (representing batch size), if not provided, it will be
#            added automatically.
net = tflearn.input_data([None, 100])

# LAYER-1
# input_dim = 10000, since that how many words we loaded from our dataset
# output_dim = 128, number of outputs of resulting embedding

# LAYER-2
net = tflearn.embedding(net, input_dim=10000, output_dim=128)

# LSTM = long short term memory
# this layer allow us to remember our data from the beginning of the sequences which 
# allow us to improve our prediction
# dropout = 0.8, is the technique to prevent overfitting, which randomly switch on and off different 
# pathways in our network
net = tflearn.lstm(net, 128, dropout=0.8)

# LAYER-3: Our next layer is fully connected, this means every neuron in the previous layer is connected to 
# the every neuron in this layer
# Good way of learning non-linear combination
# activation: take in vector of input and squash it into vector of output probability between 
# 0 and 1.
net = tflearn.fully_connected(net, 2, activation='softmax')

# LAYER-4: 
# adam : perform gradient descent
# categorical_crossentropy: find error between predicted output and original output 
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')


# Training
# Initialize with tflearn Deep Neural Network Model.
# tensorboard_verbose: `int`. Summary verbose level, it accepts
#          different levels of tensorboard logs:
#          0: Loss, Accuracy (Best Speed).
#          1: Loss, Accuracy, Gradients.
#          2: Loss, Accuracy, Gradients, Weights.
#          3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
#              (Best visualization)
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32)


help(tflearn.DNN)


testY


predictionsY = model.predict(testX)


import matplotlib.pyplot as plt

plt.scatter(testX[:,1], testX[:,2])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Imdb Film Dataset')
plt.show()


# # Citation
# 
# @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#   title     = {Learning Word Vectors for Sentiment Analysis},
#   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
#   month     = {June},
#   year      = {2011},
#   address   = {Portland, Oregon, USA},
#   publisher = {Association for Computational Linguistics},
#   pages     = {142--150},
#   url       = {http://www.aclweb.org/anthology/P11-1015}
# }
# 
# # References
# 
# Potts, Christopher. 2011. On the negativity of negation. In Nan Li and
# David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20,
# 636-659.
# 

# ## Challenge
# 
# The challenge for this [video](https://www.youtube.com/watch?v=vOppzHpvTiQ) is to use scikit-learn to create a line of best fit for the included 'challenge_dataset'. Then, make a prediction for an existing data point and see how close it matches up to the actual value. Print out the error you get. You can use scikit-learn's [documentation](http://scikit-learn.org/stable/documentation.html) for more help. These weekly challenges are not related to the Udacity nanodegree projects, those are additional.
# 

#Import all the libraries
import pandas as pd
import numpy as np
from sklearn import linear_model as model
import matplotlib.pyplot as plt 


#read data from the challenge_dataset
dataframe = pd.read_csv('input/challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
regr = model.LinearRegression()
regr.fit(x_values, y_values)

# The coefficients
print('Coefficients: ', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((regr.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_values, y_values))

#Visualize Results
plt.scatter(x_values, y_values)
plt.plot(x_values, regr.predict(x_values))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Challenge Dataset')
plt.show()


# ## Bonus
# Bonus points if you perform linear regression on a dataset with 3 different variables
# 

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from sklearn import linear_model as model
from sklearn import datasets
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
# Imports
import matplotlib as mpl


# Loading data
#read data from the challenge_dataset
iris = datasets.load_iris()
#print(iris.data.shape)
# for the above bonus, consider only 3 different vaiables 
# i.e. two are input variable and one is output variable
x_values = iris.data[:,1:3]
print(x_values.shape)
y_values = iris.target


#train model on data
linearmodel = model.LinearRegression()
linearmodel.fit(x_values, y_values)


# The coefficients
print('Coefficients: ', linearmodel.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((linearmodel.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linearmodel.score(x_values, y_values))



#Visualize Results
fig = plt.figure()
fig.set_size_inches(12.5,7.5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values[:,0],x_values[:,1], y_values, c='g', marker= 'o')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Orignal Dataset')
ax.view_init(10, -45)

fig1 = plt.figure()
fig1.set_size_inches(12.5,7.5)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x_values[:,0],x_values[:,1], linearmodel.predict(x_values), c='r', marker= 'o')
#ax.plot_surface(x_values[:,0],x_values[:,1], linearmodel.predict(x_values), cmap=cm.hot, color='b', alpha=0.2); 
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Predicted Dataset')
ax.view_init(10, -45)


# # MNIST For ML Beginners
# 
# MNIST is a simple computer vision dataset. . It consists of images of handwritten digits like these:
# 
# <img width="300" height="250" src="https://www.tensorflow.org/versions/r0.11/images/MNIST.png">
# 
# It also includes labels for each image, telling us which digit it is. For example, the labels for the above images are 5, 0, 4, and 1.
# 
# <img width="300" height="250" src="https://www.tensorflow.org/versions/r0.11/images/softmax-regression-vectorequation.png">
# 

import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


mnist.train.labels


# ### Initialization
# 
# We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, as long as we're consistent between images. From this perspective, the MNIST images are just a bunch of points in a 784-dimensional vector space, with a very rich structure (warning: computationally intensive visualizations).
# 
# **x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation.** We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)
# 
# 
# **y = softmax(Wx + b)**
# 

x = tf.placeholder(tf.float32, [None, 784])


# A **Variable** is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation. For machine learning applications, one generally has the model parameters be Variables.
# 

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(x,W) + b)


# ### Training
# 
# In order to train our model, we need to define what it means for the model to be good. Well, actually, in machine learning we typically define what it means for a model to be bad. We call this the cost, or the loss, and it represents how far off our model is from our desired outcome. We try to minimize that error, and the smaller the error margin, the better our model is.
# 
# One very common, very nice function to determine the loss of a model is called "cross-entropy." Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning. It's defined as:
# 
# **Hy′(y)=−∑iyi′log⁡(yi)**
# 
# 
# 

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
# 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    


# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
# 
# ### Evaluating Our Model
# 

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# ## Reference
# 
# - https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html
# - https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/
# 

get_ipython().magic('matplotlib inline')

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#import seaborn as sns
import sklearn
import numpy as np


# Import data
co2_df = pd.read_csv('input/global_co2.csv')
temp_df = pd.read_csv('input/annual_temp.csv')
print(co2_df.head())
print(temp_df.head())
#print(co2_df.columns=['Year','CO2'])


# Clean data
co2_df = co2_df.ix[:,:2]                     # Keep only total CO2
co2_df = co2_df.ix[co2_df['Year'] >= 1960]   # Keep only 1960 - 2010
co2_df.columns=['Year','CO2']                # Rename columns
co2_df = co2_df.reset_index(drop=True)                # Reset index
print(co2_df.tail())


temp_df =temp_df[temp_df.Source != 'GISTEMP'] # Keep only one source 
temp_df.drop('Source', inplace = True, axis = 1) #Drop name of source
temp_df = temp_df.reindex(index=temp_df.index[::-1]) #Reset Index


temp_df = temp_df.ix[temp_df['Year'] >= 1960]   # Keep only 1960 - 2010
temp_df.columns=['Year', 'Temperature']# Rename columns
temp_df = temp_df.reset_index(drop=True)     
print(temp_df.tail())


# Concatenate
climate_change_df = pd.concat([co2_df, temp_df.Temperature], axis=1)
print(climate_change_df.head())


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(12.5,7.5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(climate_change_df['Year'], climate_change_df['Temperature'] , climate_change_df['CO2'])
ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.view_init(10, -45)


# ### Projected 2D plots
# 

f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5,7.5)
#help(plt.subplots)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')


# ### 3. Linear Regression
# 

from sklearn.model_selection import train_test_split

climate_change_df = climate_change_df.dropna()

X = climate_change_df.as_matrix(['Year'])
Y = climate_change_df.as_matrix(['CO2', 'Temperature']).astype('float32')

indexes = ~np.isnan(X)
X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X,Y, test_size=0.1)) 
#print(X_train)


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, Y_train)
print('Score: ', reg.score(X_test.reshape(-1, 1), Y_test))


# ### 4. Plot regression and visualize results
# 

x_line = np.arange(1960,2011).reshape(-1,1)
p = reg.predict(x_line).T
print(x_line)



fig2 = plt.figure()
fig2.set_size_inches(12.5, 7.5)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.plot(x_line, p[1], p[0], color='g')
ax.view_init(10, -45)


f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5, 7.5)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[0].plot(x_line, p[0])
axarr[0].set_ylabel('CO2 Emissions')
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].plot(x_line, p[1])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')


