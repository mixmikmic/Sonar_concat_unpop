# # Deep Learning with TensorFlow - Creating the Neural Network Model
# 
# 
# TensorFlow allows us to perform specific machine learning number-crunching operations like derivatives on huge matricies with large efficiency. We can also easily distribute this processing across our CPU cores, GPU cores, or even multiple devices like multiple GPUs. Tensor, in TensorFlow is an array-like object, and, similar to an array it can hold matrix, vector, and even a scalar. In this tutorial we'll work with MNIST dataset. MNIST is a simple computer vision dataset. It consists of images of handwritten digits like the image below. We will then train a deep neural network on the training set using TensorFlow and make predictions on a test set.
# 
# ![](https://www.tensorflow.org/images/MNIST.png)
# 
# Resources:
# 
# - [Tutorial by PythonProgramming.net](https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/)
# - [About MNIST](https://www.tensorflow.org/get_started/mnist/beginners)
# 

# ## Understanding and Loading the Data
# 
# We're going to be working first with the MNIST dataset, which is a dataset that contains 60,000 training samples and 10,000 testing samples of hand-written and labeled digits, 0 through 9, so ten total "classes." 
# 
# The MNIST dataset has the images (see example above), which we'll be working with as purely black and white, thresholded, images, of size 28 x 28, or 784 pixels total. 
# 
# ![](https://www.tensorflow.org/images/MNIST-Matrix.png)
# 
# Our features will be the pixel values for each pixel, thresholded. Either the pixel is "blank" (nothing there, a 0), or there is something there (1). Those are our features. We're going to attempt to just use this extremely rudimentary data, and predict the number we're looking at (a 0,1,2,3,4,5,6,7,8, or 9).
# 

import tensorflow as tf
# loading the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


# The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). 
# 
# For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.
# 

# ## Building the model: Setting up the Computation Model
# 
# In case of neural network,
# 
# We have feature data, X,  value in each pixel, weights (w), and thresholds or biases (t). 
# 
# TensorFlow works by first defining and describing our model in abstract, and then, when we are ready, we make it a reality in the session. The description of the model is what is known as your "Computation Graph" in TensorFlow terms. Here is the algorithm:
# 
# - We begin by specifying how many nodes each hidden layer will have, how many classes our dataset has, and what our batch size will be. 
# - First, we take our input data, and we need to send it to hidden layer 1. 
#     - We weight the input data, and send it to layer 1, where it will undergo the activation function, 
#     - The neuron can decide whether or not to output data to either output layer, or another hidden layer. 
#     
# - We will have three hidden layers in this example, making this a Deep Neural Network.     
# - From the output we get, we will start training.
# 

# defining number of hidden layers, nodes in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100


# placeholders for variables x and y
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


# We have used [None,784] as a 2nd parameter in the first placeholder. This is an optional parameter. It can be useful, however, to be explicit like this.
# We're now complete with our constants and starting values. Now we can actually build the Neural Network Model
# 

def neural_network_model(data):
    
    """Layers definitions"""
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1])) }

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2])) }

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3])) }

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])) }
    
    """Feed Forward"""
    # input_data*weights + biases
    # relu (rectified linear) activation function
    # layer 1
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']) , output_layer['biases'])
    
    return output


# The bias is a value that is added to our sums, before being passed through the activation function, not to be confused with a bias node, which is just a node that is always on. The purpose of the bias here is mainly to handle for scenarios where all neurons fired a 0 into the layer. A bias makes it possible that a neuron still fires out of that layer. A bias is as unique as the weights, and will need to be optimized too. 
# 
# ## Training the model
# 
# Under a new function, train_neural_network, we will pass our output data.
# 
# - We then produce a prediction based on the output of that data through our neural_network_model. 
# - Next, we create a cost variable. This measures how wrong we are, and is the variable we desire to minimize by manipulating our weights. The cost function is synonymous with a loss function. 
# - To optimize our cost, we will use the AdamOptimizer, which is a popular optimizer along with others like Stochastic Gradient Descent and AdaGrad, for example.
# - Within AdamOptimizer(), we can optionally specify the learning_rate as a parameter. The default is 0.001, which is fine for most circumstances. 
# - Now that we have these things defined, we begin the session.
# 

def train_neural_network(x):
    # predictions from one feedforward epoch
    prediction = neural_network_model(x)
    
    # Minimizing the cost
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # total number of epochs
    hm_epochs = 10
    
    # Begin the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


# Somewhere between 10 and 20 epochs should give us ~95% accuracy. 95% accuracy, sounds great, but is actually considered to be very bad compared to more popular methods. Consider that the only information we gave to our network was pixel values, that's it. We did not tell it about looking for patterns, or how to tell a 4 from a 9, or a 1 from a 8. The network simply figured it out with an inner model, based purely on pixel values to start, and achieved 95% accuracy in just 4 epochs. In next coming posts I will use this same dataset with more complex neural networks such as Convolution Neural Networks and Recurring Neural Networks.
# 

# # Naive Bayes to classify movie reviews based on sentiment
# 
# ## From scratch and with Scikit-Learn
# 
# We want to predict whether a review is negative or positive, based on the text of the review. We'll use Naive Bayes for our classification algorithm. A Naive Bayes classifier works by figuring out how likely data attributes are to be associated with a certain class. Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.
# 
# ![](http://psychonerds.com/wp-content/uploads/2017/03/defining-movie-genres-horror-movies.png)
# 
# 
# Key Results:
# 
# - I first coded the Naive Bayes in python from scratch to understand the algorithm in depth. 
# - Then I had my model trained on a training set containing 1418 reviews, half of which were positive. 
# - I used a test set containing 197 reviews to make predictions and obtaining an AUC of 0.68
# - Then I repeated the analysis in scikit-learn and got a similar accuracy
# 
# 
# Resources:
# 
# [View the project and data-files on DataQuest](https://www.dataquest.io/m/27/naive-bayes-for-sentiment-analysis)
# [Scikit-learn Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
# 
# Organization:
# 
# [What is Naive Bayes?](#what-is-naive-bayes)
# 
# [Loading the Data](#loading-the-data)
# 
# [Training a Model](#training-a-model)
# 
# [Making Predictions](#making-predictions)
# 
# [Naive Bayes implementation in scikit-learn](naive-bayes-implementation-in-scikit-learn)

# ## What is Naive Bayes?
# Naive Bayes classifier is based on Bayes' theorem, which is:
# 
# $$P(H \mid x) = \frac{P(H) P(x \mid H) }{P(x)}$$
# 
# - P(H|x) is the posterior probability of hypothesis (H or target) given predictor (x or attribute).
# - P(H) is the prior probability of hypothesis
# - P(x|H) is the likelihood which is the probability of predictor given hypothesis.
# - P(x) is the prior probability of predictor.
# 
# Naive Bayes extends Bayes' theorem to handle multiple evidences by assuming that each evidence is independent.
# 
# $$ P(H \mid x_1, \dots, x_n) = \frac{P(H) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \dots, x_n)}$$
# 
# 
# In most of these problems we will compare the probabilities H being true or false and the denominator will not affect the outcome so we can simply calculate the numerator.  
# 

# ## Loading the data
# 
# We'll be working with a CSV file containing movie reviews. Each row contains the text of the review, as well as a number indicating whether the tone of the review is positive(1) or negative(-1).
# 
# We want to predict whether a review is negative or positive, based on the text alone. To do this, we'll train an algorithm using the reviews and classifications in train.csv, and then make predictions on the reviews in test.csv. We'll be able to calculate our error using the actual classifications in test.csv to see how good our predictions were.
# 

import csv
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))
    
print(reviews[0]) 
print(reviews[1]) 
print(len(reviews))


# ## Training a Model
# 
# 
# ### Obtaining P(H) : the prior probability of hypothesis (positive review)
# 
# Now that we have the word counts, we just need to convert them to probabilities and multiply them out to predict the classifications.
# 
# Let's start by obtaining the prior probabilities as follows:
# 

# Computing the prior(H=positive reviews) according to the Naive Bayes' equation
def get_H_count(score):
    # Compute the count of each classification occurring in the data
    return len([r for r in reviews if r[1] == str(score)])

# We'll use these counts for smoothing when computing the prediction
positive_review_count = get_H_count(1)
negative_review_count = get_H_count(-1)

# These are the prior probabilities (we saw them in the formula as P(H))
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)
print("P(H) or the prior is:", prob_positive)


# ### Obtaining P(xi|H) : the likelihood
# 
# #### Finding Word Counts
# 
# We're trying to determine if we should classify a data row as negative or positive. The easiest way to generate features from text is to split the text up into words. Each word in a movie review will then be a feature that we can work with. To do this, we'll split the reviews based on whitespace.
# 
# Afterwards, we'll count up how many times each word occurs in the negative reviews, and how many times each word occurs in the positive reviews. Eventually, we'll use the counts to compute the probability that a new review will belong to one class versus the other.
# 
# [We will use the following python library](https://docs.python.org/2/library/collections.html)
# ***class collections.Counter([iterable-or-mapping])***
# 
# Where, a Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. The Counter class is similar to bags or multisets in other languages.
# 

# Python class that lets us count how many times items occur in a list
from collections import Counter
import re

def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone
    # Lowercase the text so that the algorithm doesn't see "Not" and "not" as different words, for example
    return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(text):
    # Split text into words based on whitespace -- simple but effective
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)

negative_text = get_text(reviews, -1)
positive_text = get_text(reviews, 1)

# Generate word counts(WC) dictionary for negative tone
negative_WC_dict = count_text(negative_text)

# Generate word counts(WC) dictionary for positive tone
positive_WC_dict = count_text(positive_text)

print("Negative text sample: {0}".format(negative_text[:100]))
print("Positive text sample: {0}".format(positive_text[:100]))


# example
print("count of word 'bad' in negative reviews", negative_WC_dict.get("bad"))
print("count of word 'good' in negative reviews", negative_WC_dict.get("good"))


# #### Obtaining P(xi|H) the likelyhood or the probability of predictor (a word) given hypothesis (positive review)
# 
# - For every word in the text, we get the number of times that word occurred in the text, text_WC_dict.get(word) 
# - Multiply it with the probability of that word in Hypothesis (e.g. positive review) = (H_WC_dict.get(word) / (sum(H_WC_dict.values())
# 
# 
# ```python
# prediction =  text_WC_dict.get(word) * (H_WC_dict.get(word) / (sum(H_WC_dict.values())
# ```
# 
# We add 1 to smooth the value, smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data and correspondingly smooth the denominator counts to keep things even.
# 
# After smoothing above equation becomes:
# 
# ```python
# prediction =  text_WC_dict.get(word) * (H_WC_dict.get(word)+1) / (sum(H_WC_dict.values()+ H_count)
# ```
# 

# H = positive review or negative review
def make_class_prediction(text, H_WC_dict, H_prob, H_count):
    prediction = 1
    text_WC_dict = count_text(text)
    
    for word in text_WC_dict:       
        prediction *=  text_WC_dict.get(word,0) * ((H_WC_dict.get(word, 0) + 1) / (sum(H_WC_dict.values()) + H_count))

        # Now we multiply by the probability of the class existing in the documents
    return prediction * H_prob


# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater
def make_decision(text):
    
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return -1
    return 1

print("For this review: {0}".format(reviews[0][0]))
print("")
print("The predicted label is ", make_decision(reviews[0][0]))
print("The actual label is ", reviews[0][1])


# ## Making Predictions
# 
# Now that we can make predictions, let's predict the probabilities for the reviews in test.csv
# 

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0]) for r in test]


# ### Error analysis on predictions
# 

actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))


# ## Naive Bayes implementation in scikit-learn
# 
# There are a lot of extensions we could add to this algorithm to make it perform better. We could remove punctuation and other non-characters. We could remove stopwords, or perform stemming or lemmatization.
# 
# We don't want to have to code the entire algorithm out every time, though. An easier way to use Naive Bayes is to use the implementation in scikit-learn. Scikit-learn is a Python machine learning library that contains implementations of all the common machine learning algorithms
# 

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# Generate counts from text using a vectorizer  
# We can choose from other available vectorizers, and set many different options
# This code performs our step of computing word counts
vectorizer = CountVectorizer(stop_words='english', max_df=.05)
train_features = vectorizer.fit_transform([r[0] for r in reviews])
test_features = vectorizer.transform([r[0] for r in test])

# Fit a Naive Bayes model to the training data
# This will train the model using the word counts we computed and the existing classifications in the training set
nb = MultinomialNB()
nb.fit(train_features, [int(r[1]) for r in reviews])

# Now we can use the model to predict classifications for our test features
predictions = nb.predict(test_features)

# Compute the error
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomal naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))


# It's slightly different from our model because the internals of this process work differently from our implementation.
# 

# # Using Decision Trees With Scikit-Learn
# 
# The decision tree algorithm is a supervised learning algorithm -- we first construct the tree with historical data, and then use it to predict an outcome. One of the major advantages of decision trees is that they can pick up nonlinear interactions between variables in the data that linear regression can't.
# 
# 
# The data is income data from the 1994 census, and contains information on an individual's marital status, age, type of work, and more. The target column, or what we want to predict, is whether individuals make less than or equal to 50k a year, or more than 50k a year.
# 
# ![](https://1.bp.blogspot.com/_iJyjQ6GZMcE/R8TzdU148XI/AAAAAAAAABY/bkjLU0VOouY/S660/dec1.bmp)
# 
# 
# Resources:
# - [Download the data from the University of California, Irvine's website](http://archive.ics.uci.edu/ml/datasets/Adult).
# 
# - [Check out my project in Jupyter Notebook in which I build Decision Trees from scratch.](https://github.com/SuruchiFialoke/MachineLearning/blob/master/RandomForest/Intro_DecisionTree.ipynb)
# 
# - [View this project on DataQuest](https://www.dataquest.io/m/92/applying-decision-trees/2/using-decision-trees-with-scikit-learn)
# 

# ## Understanding the data
# 
# Before we get started with decision trees, we need to convert the categorical variables in our data set to numeric variables. This involves assigning a number to each category label, then converting all of the labels in a column to the corresponding numbers.
# One strategy is to convert the columns to a categorical type. Under this approach, pandas will display the labels as strings, but internally store them as numbers so we can do computations with them.
# 
# Here are some of the columns from the [dataset](http://archive.ics.uci.edu/ml/datasets/Adult)
# 
# - age: continuous. 
# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
# - fnlwgt: continuous. 
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
# - education-num: continuous. 
# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, 
# 
# Steps:
# 
# **Step 1: Identify the columns that could be useful **   
# 
# ```python 
# columns_of_interest = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country", 'high_income']
# ```
# 
# **Step 2: Convert categorical columns to their numeric values**
# 
# ```python 
# categorical_columns = [ "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income"]
# ```
# 
# ** Step 3: Identify features and target **. Store the names of columns to train in list called 'features' and the name of the target column 'high_income' in variable called target.
# 

## 1. Step 1: Identify the columns that could be useful ##
import pandas as pd

# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age).
income = pd.read_csv("income.csv", index_col=False)
columns_of_interest = ["age", "workclass", "education_num", "marital_status", "occupation",                       "relationship", "race", "sex", "hours_per_week", "native_country", 'high_income']

income = income.loc[:, columns_of_interest]
income.head(2)


## Step 2: Convert categorical columns to their numeric values ##

features_categorical = [ "workclass", "education_num", "marital_status", "occupation",                        "relationship", "race", "sex","native_country", "high_income"]
for c in features_categorical:
    income[c] = pd.Categorical(income[c]).codes

income.head(2)


## Step 3: Identify features and target ##

features = ["age", "workclass", "education_num", "marital_status", "occupation",            "relationship", "race", "sex", "hours_per_week", "native_country"]
target = 'high_income'


# ## Decision Trees With Scikit-Learn
# 
# We can use the scikit-learn package to fit a decision tree. We use the DecisionTreeClassifier class for classification problems, and DecisionTreeRegressor for regression problems. The sklearn.tree package includes both of these classes.
# 
# In this case, we're predicting a binary outcome, so we'll use a classifier.
# 
# step 1: Instantiate the classifier
# 
# step 2: Create training and testing sets
# 
# step 3: Fit the training data to the classifier
# 
# step 4: Make predictions
# 
# step 5: Check accuracy on test set using AUC
# 
# step 6: Check for overfitting 
# 
# ### Step 1: Instantiate the classifier
# 

from sklearn.tree import DecisionTreeClassifier

# Instantiate the classifier
clf = DecisionTreeClassifier(random_state=1)


# ### Step 2: Train-Test Split
# 

import numpy as np
import math

# Set a random seed so the shuffle is the same every time
np.random.seed(1)

# Shuffle the rows  
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(np.random.permutation(income.index))

# 80% to train and 20% to test
train_max_row = math.floor(income.shape[0] * .8)

train = income.iloc[:train_max_row, :]
test = income.iloc[train_max_row:, :]


# ### Step 3: Fit the Model
# 

# Fit the model

clf.fit(train[features], train[target])


# ### Step 4: Make predictions on the Test Set
# 

# Making predictions
predictions = clf.predict(test[features])
predictions[:2]


# ### Step 5: Check Accuracy on test set using AUC
# 
# While there are many methods for evaluating error with classification, we'll use [AUC obtained from ROC curve](http://gim.unmc.edu/dxtests/roc3.htm). AUC ranges from 0 to 1, so it's ideal for binary classification. The higher the AUC, the more accurate our predictions. We can compute AUC with the roc_auc_score function from sklearn.metrics. This function takes in two parameters:
# 
# y_true: true labels
# 
# y_score: predicted labels
# 

from sklearn.metrics import roc_auc_score

test_auc = roc_auc_score(test[target], predictions)

print(test_auc)


# ### Step 6: Check for overfitting
# The AUC for the predictions on the testing set is about .694 which is not very good. Let's compare this against the AUC for predictions on the training set to see if the model is overfitting.
# 

train_predictions = clf.predict(train[columns])

train_auc = roc_auc_score(train[target], train_predictions)

print(train_auc)


# Our AUC on the training set was .947, and the AUC on the test set was .694. There's no hard and fast rule on when overfitting is occurring, but our model is predicting the training set much better than the test set. Splitting the data into training and testing sets helps us detect and fix it. Trees overfit when they have too much depth and make overly complex rules that match the training data, but aren't able to generalize well to new data. The deeper a tree is, the worse it typically performs on new data.
# 
# 
# 

# ## Optimize Overfitting With A Shallower Tree
# 
# There are three main ways to combat overfitting:
# 
# 1. Restrict the depth of the tree while we're building it.
# 2. "Prune" the tree after we build it to remove unnecessary leaves.
# 3. Use ensembling to blend the predictions of many trees.
# 
# In this project we will focus on the first method
# 
# ### Restrict the depth of the tree
# 
# We can restrict tree depth by adding a few parameters when we initialize the DecisionTreeClassifier class:
# 
# - max_depth - Globally restricts how deep the tree can go
# - min_samples_split - The minimum number of rows a node should have before it can be split; if this is set to 2, for example, then nodes with 2 rows won't be split, and will become leaves instead
# - min_samples_leaf - The minimum number of rows a leaf must have
# - min_weight_fraction_leaf - The fraction of input rows a leaf must have
# - max_leaf_nodes - The maximum number of total leaves; this will cap the count of leaf nodes as the tree is being built
# Some of these parameters aren't compatible, however. For example, we can't use max_depth and max_leaf_nodes together.
# 

def get_aucs(max_depth):
    # Decision trees model with max_depth 
    clf = DecisionTreeClassifier(random_state=1, max_depth=max_depth)

    clf.fit(train[columns], train[target])

    # Test AUC
    predictions = clf.predict(test[columns])
    test_auc = roc_auc_score(test[target], predictions)

    # Train AUC
    predictions_train = clf.predict(train[columns])
    train_auc = roc_auc_score(train[target], predictions_train)
    
    return test_auc, train_auc

depth_values = np.arange(2, 40)
auc_values = np.zeros((len(depth_values), 3))
for i, val in enumerate(depth_values):
    test_auc, train_auc = get_aucs(val)
    auc_values[i, 0]  = val
    auc_values[i, 1]  = test_auc
    auc_values[i, 2]  = train_auc


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(8,4))
plt.plot(auc_values[:,0], auc_values[:,1], label='Test AUC') 
plt.plot(auc_values[:,0], auc_values[:,2], color='b', label='Train AUC')
plt.legend()
plt.xlabel('Maximum Tree Depth')
plt.ylabel('AUC')

plt.show()


# For smaller values of tree depth (< 6), we have comparable values of AUC for both test and training, smaller the value the worse the accuracy. This is underfitting, suggesting that our model is not complex enough for the features. However for larger values the models does very well on the training set but not as well on the test set due to over-fitting. Around tree-depth of 10, we have optimal performance, largest value of AUC where the performance of test and training data are similar.  
# 

# #### The Bias-Variance Tradeoff
# 
# By artificially restricting the depth of our tree, we prevent it from creating a model that's complex enough to correctly categorize some of the rows. If we don't perform the artificial restrictions, however, the tree becomes too complex, fits quirks in the data that only exist in the training set, and doesn't generalize to new data.
# 
# Imagine that we take a random sample of the training data and create many models.
# 
# - High bias can cause underfitting -- if a model is consistently failing to predict the correct value, it may be that it's too simple to model the data faithfully.
# 
# - High variance can cause overfitting. If a model varies its predictions significantly based on small changes in the input data, then it's likely fitting itself to quirks in the training data, rather than making a generalizable model.
# 
# We call this the bias-variance tradeoff because decreasing one characteristic will usually increase the other. 
# 
# ## Advantages and Disadvantages of using Decision Trees
# 
# Let's go over the main advantages and disadvantages of using decision trees. The main advantages of using decision trees is that they're:
# 
# - Easy to interpret
# - Relatively fast to fit and make predictions
# - Able to handle multiple types of data
# - Able to pick up nonlinearities in data, and usually fairly accurate
# 
# The main disadvantage of using decision trees is their tendency to overfit.
# 
# Decision trees are a good choice for tasks where it's important to be able to interpret and convey why the algorithm is doing what it's doing.
# 
# The most powerful way to reduce decision tree overfitting is to create ensembles of trees. The random forest algorithm is a popular choice for doing this. In cases where prediction accuracy is the most important consideration, random forests usually perform better.
# 
# 

# # Deep Learning with TFLearn -  Convolutional Neural Network Model
# 
# In this tutorial we'll continue working with MNIST dataset. MNIST is a simple computer vision dataset and consists of images of handwritten digits like the image below. In [my first project](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/) on this series, I trained a basic neural network to distinguish digits from 0 to 9 on this dataset. Then, in [the second tutorial](http://suruchifialoke.com/2017-06-17-predicting-digits-cnn-tensorflow/), I implemented a Convolutional Neural Network (ConvNet) using TensorFlow. In this tutorial I inplement the ConvNet using TFLearn, which is a high-level/abstraction layer for TensorFlow. The reason for using TFLearn or similar abstractions such as Keras, SKFlow, and TFSlim is to bypass the verbosity of the TensorFlow code. So let's begin.
# 
# 
# Resources:
# - [About TFLearn](http://tflearn.org/)
# - [About MNIST](https://www.tensorflow.org/get_started/mnist/beginners)
# - [My basic Neural Network project on this dataset](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/)
# - [My TensorFlow implementation of ConvNet on this dataset](http://suruchifialoke.com/2017-06-17-predicting-digits-cnn-tensorflow/)
# - [Tutorial by PythonProgramming.net](https://pythonprogramming.net/convolutional-neural-network-cnn-machine-learning-tutorial/?completed=/rnn-tensorflow-python-machine-learning-tutorial/)
# 

# ## Importing the MNIST Data
# 
# We're going to be working first with the MNIST dataset, which is a dataset that contains 60,000 training samples and 10,000 testing samples of hand-written and labeled digits, 0 through 9, so ten total "classes." 
# 
# The MNIST dataset has the images as purely black and white, of size 28 x 28, or 784 pixels total. 
# 
# Our features will be the pixel values for each pixel, thresholded. Either the pixel is "blank" (nothing there, a 0), or there is something there (1). Those are our features. We're going to attempt to just use this extremely rudimentary data, and predict the number we're looking at (a 0,1,2,3,4,5,6,7,8, or 9).
# 

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])


# ## Setting up the Computation Model
# 
# We have imported a function for the convolution and pooling.  We have also imported fully_connected and regression. Then, we load in the data, and reshape the data. Now we are going to begin building the convolutional neural network, starting with the input layer:
# 
# ** Definitions of the layers **
# 
# - Each input image had a 28x28 pixels - > Reshape input to a 4D tensor 
# - Take 5x5 convolutions on the initial image, and get 32 outputs and apply ReLU
# - Next, take 5x5 convolutions of the 32 inputs and make 64 output and apply RelU. 
# - We're left with 64 of 7x7 sized images (It went through two max pooling process. And each time, it reduces the dimension by half because of its stride of 2 and size of 2. Hence, 28/2/2 = 7)
# - then we're outputting to 1024 nodes in the fully connected layer and applying ReLu
# - Then, the output layer is 1024 layers, to 10, which are the final 10 possible outputs for the actual label itself (0-9).
# 
# * The ReLU rectifier is an activation function defined as f(x)=max(0,x) 
# 

# input layer
convnet = input_data(shape=[None, 28, 28, 1], name='input')

# 2 layers of convolution and pooling:

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# Fully connected layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# Output Layer
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='targets')


# 
# ## Training the model
# 
# 

# Create the model 

model = tflearn.DNN(convnet)

model.fit({'input': X}, {'targets': Y}, n_epoch=10, 
          validation_set=({'input': test_x}, {'targets': test_y}), 
          snapshot_step=500, show_metric=True, run_id='mnist')


# Our [basic neural network model](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/) gave an accuracy of ~95% for 10 epochs. [The ConvNet model implemented with TensorFlow]() increased it to ~98% for the same number of epochs. 
# 




# 
# # Machine Learning on Iris Dataset
# 
# _Iris_  might be more polular in the data science community as a machine learning classification problem than as a decorative flower. Three _Iris_ varieties were used in the Iris flower data set outlined by Ronald Fisher in his famous 1936 paper _"The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis"_ [PDF](http://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x/epdf). Since then his paper has been cited over 2000 times and the data set has been used by almost every data science beginner. 
# 
# In this project, I proudly join the bandwagon and use this popular Iris dataset to make predictions on the three types of iris. The prediction is based on shape of an iris leaf represented by its sepal length, sepal width, petal length and petal width as shown in the image.
# 
# ![Iris](./iris.png)
# 
# ## Understanding and loading the data
# 
# Since it is such a common data set it's  built in scikit learn as a module [find here.](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) 
# This data is saved as Dictionary-like object, the interesting attributes are: **data**, the data to learn, **target**, the classification labels, **target_names**, the names of the labels, **feature_names**, the names of the features, and **DESCR**, the full description of the dataset.
# 
# The data set consists of:
# * **150 samples** 
# * **3 labels: species of Iris (_Iris setosa, Iris virginica_ and _Iris versicolor_)** 
# * **4 features: length and the width of the sepals and petals**, in centimetres.
# 
# Scikit learn only works if data is stored as numeric data, irrespective of it being a regression or a classeification problem. It also requires the arrays to be stored at numpy arrays for optimization. Since, this dataset is loaded from scikit learn, everything is appropriately formatted.

# install relevant modules
import numpy as np

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt
# allow plots to appear within the notebook
get_ipython().magic('matplotlib inline')
import seaborn as sns


# save "bunch" object containing iris dataset and its attributes into iris_df
iris_df = load_iris()
type(iris_df)


# Look into the features 
print (iris_df.feature_names)
print (iris_df.data[0:3, :])
print(type(iris_df.data))


# Look into the labels
print (iris_df.target_names)
print (iris_df.target[:3])
print(type(iris_df.target))


# store feature matrix in X and label vector in y
X = iris_df.data
y = iris_df.target
# print and check shapes of X and y
print("shape of X: ", X.shape, "& shape of y: ", y.shape)


# ## Training a machine learning model with scikit-learn
# 
# ### K-nearest neighbors (KNN) classification
# 
# This method searches for the K observations in the training data that are "nearest" to the measurements of the new observation. Then it uses the most popular response value from the K nearest neighbors as the predicted response value for the new observation. Following steps:
# 
# - Provide a value of K
# - "Instantiate" (make an instance of) the "estimator" (scikit-learn's term for model)
# - Train the model with data (Model learns the relationship between X and y, Occurs in-place)
# - Predict the response for a new observation
# 

# KNN classification 
# Instantiate the estimator 
knn1 = KNeighborsClassifier(n_neighbors=1)
knn5 = KNeighborsClassifier(n_neighbors=5)

# Train the model
# output displays the default values
knn1.fit(X, y)
knn5.fit(X, y)


# Predict the response
X_new = [[3, 4, 5, 2], [5, 2, 3, 2]]
print("n_neighbors=1 predicts: ", knn1.predict(X_new))
print("n_neighbors=5 predicts: ", knn5.predict(X_new))


# ### Logistic Regression Classification
# Logistic regression is another very common way of classification. Logistic regression was developed by statistician David Cox in 1958. The binary logistic model is used to estimate the probability of a binary response based on one or more features. And for classifying more than two labels, it uses "one versus the rest" technique. In scikit-learn the implementation for all models are very similar, making it very easy for begineers.
# 

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
# output displays the default values
logreg.fit(X, y)


# predict the response for new observations
logreg.predict(X_new)


# ## Evaluating the models
# One Common evaluation metric for classification accuracy in classification problems is proportion of correct predictions, _**accuracy_score**_, on a given set. We can get _accuracy_score_ for the training data or a new set of observations. 
# 
# 
# ### 1. Train and test on the entire dataset
# 
# Training and testing on the same data is not recommended as our goal is to estimate likely performance of a model on out-of-sample data. But, maximizing training accuracy rewards overly complex models that won't necessarily generalize and result in overfitting the training data.
# 

# store the predicted response values
y_pred_knn1 = knn1.predict(X)
y_pred_knn5 = knn5.predict(X)
y_pred_logreg = logreg.predict(X)

# compute classification accuracy for the logistic regression model
print("Accuracy of KNN with n_neighbors=1: ", metrics.accuracy_score(y, y_pred_knn1))
print("Accuracy of KNN with n_neighbors=5: ", metrics.accuracy_score(y, y_pred_knn5))
print("Accuracy of logistic regression: ", metrics.accuracy_score(y, y_pred_logreg))


# ### 2. Train / test Split Method
# 
# It is one of the most common way to test the accuracy of a model. Its fairly intuitive to understand, split the dataset into a training set and a testing set in any proportion.
# Train the model on the training set. Test the model on the testing set.
# 
# Note tha I use **random_state=some_number**, to guarantee that my split is always identica;s. This is useful to get reproducible results, and compare across models. 
# 

# Splitting the data in 75% training data and 25% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
print("shape of X_train: ", X_train.shape, "& shape of y_train: ", y_train.shape)
print("shape of X_test: ", X_test.shape, "& shape of y_test: ", y_test.shape)


# Instantiate the estimators 
knn1 = KNeighborsClassifier(n_neighbors=1)
knn5 = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression()

# Train the models
# output displays the default values
logreg.fit(X_train, y_train)
knn1.fit(X_train, y_train)
knn5.fit(X_test, y_test)
print('\n')

# Predictions
y_pred_knn1 = knn1.predict(X_test)
y_pred_knn5 = knn5.predict(X_test)
y_pred_logreg = logreg.predict(X_test)

# compute classification accuracy for the logistic regression model
print("Accuracy of KNN with n_neighbors=1: ", metrics.accuracy_score(y_test, y_pred_knn1))
print("Accuracy of KNN with n_neighbors=5: ", metrics.accuracy_score(y_test, y_pred_knn5))
print("Accuracy of logistic regression: ", metrics.accuracy_score(y_test, y_pred_logreg))


# ### 3. Best estimate of K for KNN-classification
# 

# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26, 2))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# plot the relationship between K and testing accuracy
plt.rcParams.update({'font.size': 18})
plt.plot(k_range, scores, 'ro', linewidth=2.0, linestyle="-")
plt.xlabel('K')
plt.ylabel('Accuracy')


# ## Let's Talk Linear Regression 
# 

# ## Loading and Understanding Data
# 

# import relevant modules
import pandas as pd
import numpy as np
import quandl, math

# Machine Learning
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Visualization
import matplotlib 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

import datetime


# Get unique quandl key by creating a free account with quandl 
# And directly load financial data from GOOGL

quandl.ApiConfig.api_key = 'q-UWpMLYsWKFejy5y-4a'
df = quandl.get('WIKI/GOOGL')

# Getting a peek into data 
# I am using round function to see only upto 2 decimal digits
print(df.head(2).round(1))
print('\n')

# Also print columns and index
print(df.columns)
print(df.index)


# ## Feature Engineering
# As you would notice the data has very strongly dependent features such as 'Open' and 'Adj. Open'. Let's deal with only adjusted data as they are largely self contained before we even get into feature engineering. We can also discard any other column that are irrelevant for or prediction.We can refine our features even further based on our general understanding of financial data.
# For instance, instead of dealing with High and Low separately, we could create volatility percentage as a new feature.
# 
# $$HL\_PCT = \frac{high - low}{low*100}$$
# Similarly, 
# $$PCT\_CHNG = \frac{close - open}{open*100}$$
# 

# Discarding features that aren't useful
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# define a new feature, HL_PCT
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/(df['Adj. Low']*100)

# define a new feature percentage change
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open'])/(df['Adj. Open']*100)

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHNG', 'Adj. Volume']]

print(df.head(1))


# Visualization

df['Adj. Close'].plot(figsize=(15,6), color="green")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

df['HL_PCT'].plot(figsize=(15,6), color="red")
plt.xlabel('Date')
plt.ylabel('High Low Percentage')
plt.show()

df['PCT_CHNG'].plot(figsize=(15,6), color="blue")
plt.xlabel('Date')
plt.ylabel('Percent Change')
plt.show()


# ## Machine Learning
# 
# ### Creating Features and Label
# 

# pick a forecast column
forecast_col = 'Adj. Close'

# Chosing 30 days as number of forecast days
forecast_out = int(30)
print('length =',len(df), "and forecast_out =", forecast_out)


# Creating label by shifting 'Adj. Close' according to 'forecast_out'
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head(2))
print('\n')
# If we look at the tail, it consists of n(=forecast_out) rows with NAN in Label column 
print(df.tail(2))


# Define features (X) by excluding the label column which we just created 
X = np.array(df.drop(['label'], 1))
# Using a feature in sklearn, preposessing to scale features
X = preprocessing.scale(X)
print(X[1,:])


# X contains last 'n= forecast_out' rows for which we don't have label data
# Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

X_forecast_out = X[-forecast_out:]
X = X[:-forecast_out]
print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X))


# Similarly Define Label y for the data we have prediction for
# A good test is to make sure length of X and y are identical
y = np.array(df['label'])
y = y[:-forecast_out]
print('Length of y: ',len(y))


# ### Creating Training and Test Sets
# 
# Using cross validation basically shuffles the data and according to our test_size criteria, splits the data into test and training data.
# 

# Cross validation (split into test and train data)
# test_size = 0.2 ==> 20% data is test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

print('length of X_train and x_test: ', len(X_train), len(X_test))


# ### Using Linear Regression
# Now time to use linear regression. I first Split the data into 80% of training data and 20% of test data. Used Linear regression to train and test data. Finally, I tested the accuracy of our model on the test data.
# 

# Train
clf = LinearRegression()
clf.fit(X_train,y_train)
# Test
accuracy = clf.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)


# Predict using our Model
forecast_prediction = clf.predict(X_forecast_out)
print(forecast_prediction)


# Plotting data
df.dropna(inplace=True)
df['forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_prediction:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj. Close'].plot(figsize=(15,6), color="green")
df['forecast'].plot(figsize=(15,6), color="orange")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Zoomed In to a year
df['Adj. Close'].plot(figsize=(15,6), color="green")
df['forecast'].plot(figsize=(15,6), color="orange")
plt.xlim(xmin=datetime.date(2015, 4, 26))
plt.ylim(ymin=500)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





# # Natural Language Processing on Hacker News Data
# 
# 
# 
# Natural language processing (NLP) is the study of enabling computers to understand human languages. This field may involve teaching computers to automatically score essays, infer grammatical rules, or determine the emotions associated with text.  In this project we will employ NLP on Hacker News data. Hacker News is a community where users can submit articles, and other users can upvote those articles. The articles with the most upvotes make it to the front page, where they're more visible to the community. We'll be predicting the number of upvotes the articles received, based on their headlines. Because upvotes are an indicator of popularity, we'll discover which types of articles tend to be the most popular.
# 
# 
# Resources:
# 
# [View the project and data-files on DataQuest](https://www.dataquest.io/m/67/introduction-to-natural-language-processing/)
# [Hacker News](https://news.ycombinator.com/)
# [Scikit-learn Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
# 
# Organization:
# 
# 
# [Loading the Data](#loading-the-data)
# 
# [Bag of Words Model to Tokenize the Data](#bag-of-words-model-to-tokenize-the-data)
# 
# [Linear Regression](#linear-regression)
# 

# ## Loading the Data
# 
# This set consists of submissions users made to Hacker News from 2006 to 2015. Developer Arnaud Drizard used the Hacker News API to scrape the data ([hosted on his github repository](https://github.com/arnauddri/hn)). DataQuest has randomly sampled 3000 rows from the [data, and removed all but 4 of the columns](https://www.dataquest.io/m/67/introduction-to-natural-language-processing/2/overview-of-the-data). 
# 
# Our data only has four columns:
# 
# - submission_time - When the article was submitted
# - upvotes - The number of upvotes the article received
# - url - The base URL of the article
# - headline - The article's headline
# 

import pandas as pd

submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()


submissions.head()


# ## Bag of Words Model to Tokenize the Data
# 
# Our final goal is to train a linear regression algorithm that predicts the number of upvotes a headline would receive. To do this, we'll need to convert each headline to a numerical representation.
# 
# While there are several ways to accomplish this, we'll use a [bag of words model](https://en.wikipedia.org/wiki/Bag-of-words_model). A bag of words model represents each piece of text as a numerical vector as shown below.
# 
# ![](https://image.slidesharecdn.com/wordembedings-whythehype-151203065649-lva1-app6891/95/word-embeddings-why-the-hype-4-638.jpg?cb=1449126428)
# 
# 
# Step 1: Ttokenization or breaking a sentence up into disconnected words.
# 
# Step 2: Preprocessing Tokens To Increase Accuracy: Lowercasing and removing punctuation 
# 
# Step 3: Retrieve all of the unique words from all of the headlines
# 
# Step 4: Counting Token Occurrences

# Step 1: Ttokenization
"""Split each headline into individual words on the space character(" "), 
and append the resulting list to tokenized_headlines."""

tokenized_headlines = []
for item in submissions["headline"]:
    tokenized_headlines.append(item.split(" "))
    
print(tokenized_headlines[:2])    


# Step 2: Lowercasing and removing punctuation
"""For each list of tokens: Convert each individual token to lowercase
and remove all of the items from the punctuation list"""

punctuations_list = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []
for item in tokenized_headlines:
    tokens = []
    for token in item:
        token = token.lower()
        for punc in punctuations_list:
            token = token.replace(punc, "")
        tokens.append(token)
    clean_tokenized.append(tokens)

print(clean_tokenized[:2])     


# Step 3: Retrieve all of the unique words from all of the headlines
# unique_tokens contains any tokens that occur more than once across all of the headlines.

import numpy as np
unique_tokens = []
single_tokens = []
for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)


# Step 4: Counting Token Occurrences
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1


counts.shape
counts.head(5)


# ### Removing Columns To Increase Accuracy
# 
# We have 2309 columns in our matrix, and too many columns will cause the model to fit to noise instead of the signal in the data. There are two kinds of features that will reduce prediction accuracy. Features that occur only a few times will cause overfitting, because the model doesn't have enough information to accurately decide whether they're important. These features will probably correlate differently with upvotes in the test set and the training set.
# 
# Features that occur too many times can also cause issues. These are words like and and to, which occur in nearly every headline. These words don't add any information, because they don't necessarily correlate with upvotes. These types of words are sometimes called stopwords.
# 
# To reduce the number of features and enable the linear regression model to make better predictions, we'll remove any words that occur fewer than 5 times or more than 100 times.
# 

word_counts = counts.sum(axis=0)
counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]


# ## Linear Regression
# 
# We'll train our algorithm on a training set, then test its performance on a test set. The train_test_split() function from scikit-learn will help us randomly select 20% of the rows for our test set, and 80% for our training set.
# 

# Train-test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)


# Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate an instance
clf = LinearRegression()

# Fit the training data
clf.fit(X_train, y_train)

# Make predictions
y_predict = clf.predict(X_test)


# ### Error in Prediction
# 
# We'll use mean squared error (MSE), which is a common error metric. With MSE, we subtract the predictions from the actual values, square the results, and find the mean. Because the errors are squared, MSE penalizes errors further away from the actual value more than those close to the actual value. We have high error (rmse of ~50 upvotes) in predicting upvotes as we have used a very small data set. With larger training sets, this should decrease dramatically.
# 

mse = sum((y_predict - y_test) ** 2) / len(y_predict)
rmse = (mse)**0.5
print(rmse)


# How to make better predictions?
# 
# - Using more data will ensure that the model will find more occurrences of the same features in the test and training sets, which will help the model make better predictions.
# 
# - Add "meta" features like headline length and average word length.
# 
# - Use a random forest, or another more powerful machine learning technique.
# 
# - Explore different thresholds for removing extraneous columns.

# #  Desicion Trees from Scratch
# 
# The decision tree algorithm is a supervised learning algorithm -- we first construct the tree with historical data, and then use it to predict an outcome. One of the major advantages of decision trees is that they can pick up nonlinear interactions between variables in the data that linear regression can't.
# 
# 
# The data is income data from the 1994 census, and contains information on an individual's marital status, age, type of work, and more. The target column, or what we want to predict, is whether individuals make less than or equal to 50k a year, or more than 50k a year.
# 
# Download the data from the [University of California, Irvine's website](http://archive.ics.uci.edu/ml/datasets/Adult).
# 

## Loading dataset ##

import pandas as pd

# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age).
income = pd.read_csv("income.csv", index_col=False)
print(income.head(2))


# Before we get started with decision trees, we need to convert the categorical variables in our data set to numeric variables. This involves assigning a number to each category label, then converting all of the labels in a column to the corresponding numbers.
# 
# One strategy is to convert the columns to a categorical type. Under this approach, pandas will display the labels as strings, but internally store them as numbers so we can do computations with them.
# 

## Converting categorical variables ##

# Convert a single column from text categories into numbers.
col = pd.Categorical(income["workclass"])
income["workclass"] = col.codes
print(income["workclass"].head(5))

cols = ['education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'high_income']
for c in cols:
    income[c] = pd.Categorical(income[c]).codes

print(income.head(2))


# ## Creating Splits
# 
# We can split the data set into two portions based on whether the individual works in the private sector or not based on the value of the 'workclass' column.
# 
# - private_incomes should contain all rows where workclass is 4.
# - public_incomes should contain all rows where workclass is not 4.
# 
# 
# When we performed the split, 9865 rows went to the left, where workclass does not equal 4, and 22696 rows went to the right, where workclass equals 4.
# 
# 

private_incomes = income[income['workclass'] == 4]
public_incomes = income[income['workclass'] != 4]
print(private_incomes.shape, public_incomes.shape)


# ## Information Entropy
# 
# Entropy refers to disorder. The more "mixed together" 1s and 0s are, the higher the entropy. A data set consisting entirely of 1s in the high_income column would have low entropy.
# 
# The formula for entropy looks like this:
# 
# $$-\sum_{i=1}^{c} {\mathrm{P}(x_i) \log_b \mathrm{P}(x_i)}$$
# 
# We iterate through each unique value in a single column (in this case, high_income), and assign it to i. We then compute the probability of that value occurring in the data (P(xi)). 
# 

import math

length = income.shape[0]
prob_high = sum(income['high_income'] == 1) / length
prob_low = sum(income['high_income'] == 0) / length
income_entropy = -(prob_high * math.log(prob_high, 2) + prob_low * math.log(prob_low, 2))
print(income_entropy)


# General Function to get entropy
import numpy as np
def calc_entropy(column):
    """
    Calculate entropy given a pandas Series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column.
    counts = np.bincount(column)
    # Divide by the total column length to get a probability.
    probabilities = counts / len(column)
    
    # Initialize the entropy to 0.
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy.
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy


calc_entropy(income['high_income'])


# ## Information Gain
# 
# We'll need a way to go from computing entropy to figuring out which variable to split on. We can do this using information gain, which tells us which split will reduce entropy the most.
# 
# Here's the formula for information gain:
# 
# $$IG(T,A) = Entropy(T)-\sum_{v\in A}\frac{|T_{v}|}{|T|} \cdot Entropy(T_{v})$$
# 
# We're computing information gain (IG) for a given target variable (T), as well as a given variable we want to split on (A).
# 
# To compute it, we first calculate the entropy for T. Then, for each unique value v in the variable A, we compute the number of rows in which A takes on the value v, and divide it by the total number of rows. Next, we multiply the results by the entropy of the rows where A is v. We add all of these subset entropies together, then subtract from the overall entropy to get information gain.
# 
# We first compute the information gain for splitting on the age column of income.
# 
# - First, compute the median of age.
# - Then, assign anything less than or equal to the median to the left branch, and anything greater than the median to the right branch.
# - Compute the information gain and assign it to age_information_gain
# 

# Get the median
age_median = income['age'].median()

# left and right split
left_split = income[income['age'] <= age_median]['high_income']
right_split = income[income['age'] > age_median]['high_income']

# obtain the entropy
income_entropy = calc_entropy(income['high_income'])

# Information gain
age_information_gain = income_entropy - (left_split.shape[0] / income.shape[0] * calc_entropy(left_split) + right_split.shape[0] / income.shape[0] * calc_entropy(right_split))
print(age_information_gain)


# General function to get information gain
def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a dataset, column to split on, and target.
    """
    # Calculate original entropy.
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting.
    column = data[split_name]
    median = column.median()
    
    # Make two subsets of the data based on the median.
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits, and calculate the subset entropy.
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain.
    return original_entropy - to_subtract

print(calc_information_gain(income, "age", "high_income"))


# ## Finding The Best Split
# 
# Now that we know how to compute information gain, we can determine the best variable to split a node on. When we start our tree, we want to make an initial split. We'll find the variable to split on by calculating which split would have the highest information gain.
# 
# Now we need a function that returns the name of the column we should use to split a data set. The function should take the name of the data set, the target column, and a list of columns we might want to split on as input.
# 
# - Write a function named find_best_column() that returns the name of a column to split the data on. 
# - Use find_best_column() to find the best column on which to split income.
# - The target is the high_income column, and the potential columns to split with are in the list columns below.
# - Assign the result to income_split.
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import re
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})


# Function to find the column in columns to split on
def find_best_column(data, target_name, columns, iplot=0):
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    information_gains = [calc_information_gain(data, c, target_name) for c in columns]
    
    # plot data optional
    if iplot==1:
        plt.figure(figsize=(25,5))
        x_pos = np.arange(len(columns))
        plt.bar(x_pos, information_gains,align='center', alpha=0.5)
        plt.xticks(x_pos, columns)
        plt.ylabel('information gains')
        plt.show()
    
    # return column name with highest gain
    highest_gain = columns[information_gains.index(max(information_gains))] 
    return highest_gain

# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

income_split = find_best_column(income, 'high_income', columns, iplot=1)
print(income_split, "has highest gain")


# ##  Storing the Trees (ID3) 
# 
# Let's build up to making the full id3() function by creating a recursive algorithm.We'll use nested dictionaries to do this. We can represent the root node with a dictionary, and branches with the keys left and right. We'll store the column we're splitting on as the key column, and the median value as the key median. Finally, we can store the label for a leaf as the key label. We'll also number each node as we go along using the number key.
# 
# Here's what that algorithm looks like in pseudocode:
# ```python
# def id3(data, target, columns, tree)
#     1 Create a node for the tree
#     2 Number the node
#     3 If all of the values of the target attribute are 1, assign 1 to the label key in tree
#     4 If all of the values of the target attribute are 0, assign 0 to the label key in tree
#     5 Using information gain, find A, the column that splits the data best
#     6 Find the median value in column A
#     7 Assign the column and median keys in tree
#     8 Split A into values less than or equal to the median (0), and values above the median (1)
#     9 For each possible value (0 or 1), vi, of A,
#     10 Add a new tree branch below Root that corresponds to rows of data where A = vi
#     11 Let Examples(vi) be the subset of examples that have the value vi for A
#     12 Create a new key with the name corresponding to the side of the split (0=left, 1=right).  The value of this key should be an empty dictionary.
#     13 Below this new branch, add the subtree id3(data[A==vi], target, columns, tree[split_side])
#     14 Return Root
# ```
# 
# Under this approach, we're now passing the tree dictionary into our id3 function and setting some keys on it. One complexity is in how we're creating the nested dictionary. For the left split, we're adding a key to the tree dictionary that looks like this:
# 
# tree["left"] = {}
# 
# For the right side, we're adding:
# 
# tree["right"] = {}
# 

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pd.unique(data[target])
    
    # Assign the number key to the node dictionary
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        # assign "label" field to  node dictionary
        if unique_targets[0]==1:
            tree["label"] = 1
        else: 
            tree["label"] = 0
        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    # assign "column", "median" to node dictionary
    tree["column"] = best_column
    tree["median"] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

# Create the data set that we used in the example on the last screen
data = pd.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"], tree)


print(tree)


def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
        
    # recursively call print_node on each branch, increment depth
    print_node(branches[0], depth+1)
    print_node(branches[1], depth+1)
print_node(tree, 0)


# The left branch prints out first, then the right branch. Each node prints the criteria on which it was split. 
# 
# Let's say we want to predict the following row:
# 
# 1. Example 1
# 
# ```python 
# age    marital_status
# 50     1
# ```
# 
#     First, we'd split on age > 37.5 and go to the right. Then, we'd split on age > 55.0 and go to the left. Then, we'd split on age > 47.5 and go to the right. We'd end up predicting a 1 for high_income.
#     
# 
# 2. Example 2
# ```python 
# age    marital_status
# 20     1
# ```
#     
#     First, we'd split on age > 37.5 and go to the left. Then, we'd split on age > 25.0 and go to the left again. Then, we'd split on age > 22.5 and got to left and predict a 0 for high_income.
#     
# 
# Making predictions with such a small tree is fairly straightforward, but what if we want to use the entire income dataframe? We wouldn't be able to eyeball predictions; we'd want an automated way to do this instead.
# 
# 
# 

# ## Making Predictions Automatically
# 
# Let's write a function that makes predictions automatically. All we need to do is follow the split points we've already defined with a new row.
# 
# Here's the pseudocode:
# 
# ```python
# def predict(tree, row):
#     1 Check for the presence of "label" in the tree dictionary
#     2    If found, return tree["label"]
#     3 Extract tree["column"] and tree["median"]
#     4 Check whether row[tree["column"]] is less than or equal to tree["median"]
#     5    If it's less than or equal, call predict(tree["left"], row) and return the result
#     6    If it's greater, call predict(tree["right"], row) and return the result
# ```
# 
# The major difference here is that we're returning values. Because we're only calling the function recursively once in each iteration (we only go "down" a single branch), we can return a single value up the chain of recursion. This will let us get a value back when we call the function.
# 

def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    # If row[column] is <= median,
    # return the result of prediction on left branch of tree
    # else right
    
    if row[column] <= median:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)

print(predict(tree, data.iloc[0]))


# Making Multiple Predictions using pandas apply method
# df.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)

new_data = pd.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    # Insert your code here
    return df.apply(lambda x: predict(tree, x), axis=1)
    

predictions = batch_predict(tree, new_data)
print(predictions)


# In this mission, we learned how to create a full decision tree model, print the results, and use the tree to make predictions. We applied a modified version of the ID3 algorithm on a small data set for clarity.
# 
# 

# # Deep Learning with TensorFlow - Creating the Recurrent Neural Network (RNN) Model
# 
# Recurrent Neural Networks (RNNs) are popular models that have shown great promise in many NLP tasks. The Recurrent Neural Network attempts to address the necessity of understanding data in sequences. In this tutorial we'll work with MNIST dataset. MNIST is a simple computer vision dataset. It consists of images of handwritten digits like the image below. We will then train a RNN model on the training set using TensorFlow and make predictions on a test set.
# 
# 
# ![](https://www.tensorflow.org/images/MNIST.png)
# 
# Resources:
# 
# - [About MNIST](https://www.tensorflow.org/get_started/mnist/beginners)
# - [My basic Neural Network project on this dataset](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/)
# - [Tutorial by PythonProgramming.net](https://pythonprogramming.net/recurrent-neural-network-rnn-lstm-machine-learning-tutorial/)
# - [Tutorial by WildML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
# - [A research paper on RNN, that I found useful](http://www.aclweb.org/anthology/P14-1140.pdf)
# 

# ## What are RNNs
# 
# 
# 
# The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that is very limiting. If we want to predict the next word in a sentence we can perform better by knowing which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. 
# 
# RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps. Here is what a typical RNN looks like:
# ![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg)
# 
# 
# The above diagram shows a RNN being unrolled (or unfolded) into a full network. By unrolling we  mean that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network, one layer for each word. The formulas that govern the computation happening in a RNN are as follows:
# 
# - x_t is the input at time step t. For example, x_1 could be a one-hot vector corresponding to the second word of a sentence.
# - s_t is the hidden state at time step t. It’s the “memory” of the network. s_t is calculated based on the previous hidden state and the input at the current step: s_t=f(Ux_t + Ws_{t-1}). The function f usually is a nonlinearity such as tanh or ReLU.  s_{-1}, which is required to calculate the first hidden state, is typically initialized to all zeroes.
# - o_t is the output at step t. For example, if we wanted to predict the next word in a sentence it would be a vector of probabilities across our vocabulary. o_t = \mathrm{softmax}(Vs_t).

# ## Understanding and Loading the Data
# 
# We're going to be working first with the MNIST dataset, which is a dataset that contains 60,000 training samples and 10,000 testing samples of hand-written and labeled digits, 0 through 9, so ten total "classes." 
# 
# The MNIST dataset has the images (see example above), which we'll be working with as purely black and white, thresholded, images, of size 28 x 28, or 784 pixels total. 
# 
# ![](https://www.tensorflow.org/images/MNIST-Matrix.png)
# 
# Our features will be the pixel values for each pixel, thresholded. Either the pixel is "blank" (nothing there, a 0), or there is something there (1). Those are our features. We're going to attempt to just use this extremely rudimentary data, and predict the number we're looking at (a 0,1,2,3,4,5,6,7,8, or 9).
# 

import tensorflow as tf
# loading the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


# The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). 
# 
# For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.
# 

# ## Building the model: Setting up the Computation Model
# 
# - Here, we're importing the rnn model/cell code from TensorFlow. 
# - We're also defining the chunk size, number of chunks, and rnn size as new variables.
# - Also, the shape of the x variable to include the chunks. 
# 
# In the basic neural network, we were sending in the entire image of pixel data all at once. With the Recurrent Neural Network, we're treating inputs now as sequential inputs of chunks instead.
# 

# import rnn from tensorFlow
from tensorflow.contrib import rnn 

# define number of classes = 10 for digits 0 through 9 
n_classes = 10

# defining the chunk size, number of chunks, and rnn size as new variables
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# placeholders for variables x and y
x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


# We have a weights/biases dictionary like before, but then we get to some modifications to our input data, x. We're doing this is purely to satisfy the structure that TensorFlow wants of us to fit their rnn_cell model. 
# 

def recurrent_neural_network(x):

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


# 
# ## Training the model
# 
# Under a new function, train_neural_network, we will pass our output data.
# 
# - We then produce a prediction based on the output of that data through our neural_network_model. 
# - Next, we create a cost variable. This measures how wrong we are, and is the variable we desire to minimize by manipulating our weights. The cost function is synonymous with a loss function. 
# - To optimize our cost, we will use the AdamOptimizer, which is a popular optimizer along with others like Stochastic Gradient Descent and AdaGrad, for example.
# - Within AdamOptimizer(), we can optionally specify the learning_rate as a parameter. The default is 0.001, which is fine for most circumstances. 
# - Now that we have these things defined, we begin the session.
# 

hm_epochs = 4

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    #optimizer is learning rate but in this casde the default s fine
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #_ is variable we dont care about
            #we have total number items and divide by batch_size for number of batches
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                #c is cost
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c

            print('Epoch', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)

        #we come here after optimizing the weights
        #tf.argmax is going to return the index of max values
        #in this training module we are checking if the prediction is matching the value
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #accuracy is the float value of correct
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #this is just evaluation
        print('Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)


# Somewhere between 10 and 20 epochs should give us ~95% accuracy. 95% accuracy, sounds great, but is actually considered to be very bad compared to more popular methods. Consider that the only information we gave to our network was pixel values, that's it. We did not tell it about looking for patterns, or how to tell a 4 from a 9, or a 1 from a 8. The network simply figured it out with an inner model, based purely on pixel values to start, and achieved 93% accuracy in just 4 epochs.
# 

# ## Let's Talk Linear Regression 
# 

# ## Loading and Understanding Data
# 

# import relevant modules
import pandas as pd
import numpy as np
import quandl, math

# Machine Learning
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Visualization
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')


# Get unique quandl key by creating a free account with quandl 
# And directly load financial data from GOOGL

quandl.ApiConfig.api_key = 'q-UWpMLYsWKFejy5y-4a'
df = quandl.get('WIKI/GOOGL')


# Getting a peek into data
print(df.columns)
print(df.head(2))


# As you would notice the data has very strongly dependent features such as 'Open' and 'Adj. Open'. Let's deal with only adjusted data as they are largely self contained before we even get into feature engineering. We can also discard any other column that are irrelevant for or prediction.
# 

# Discarding features that aren't useful
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print(df.head(2))


# ## Feature Engineering
# We can refine our features even further based on our general understanding of financial data.
# For instance, instead of dealing with High and Low separately, we could create volatility percentage as a new feature.
# $$HL\_PCT = \frac{high - low}{low*100}$$
# Similarly, 
# $$PCT\_CHNG = \frac{close - open}{open*100}$$
# 

# define a new feature, HL_PCT
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/(df['Adj. Low']*100)

# define a new feature percentage change
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open'])/(df['Adj. Open']*100)

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHNG', 'Adj. Volume']]

print(df.head(3))


# Check which columns have missing data
for column in df.columns:
    if np.any(pd.isnull(df[column])) == True:
        print(column)


# It seems like we don't have any missing or NaN data, in case we had that we need to replace them with something to be able to run our machine learning algorithm.
# 
# One common way to do so is as follows:  
# ``` python 
# df.fillna(-99999, inplace = True) 
# ```
# 

# pick a forecast column
forecast_col = 'Adj. Close'


# Plot features

df.plot( x = 'HL_PCT', y = 'PCT_CHNG', style = 'o')


# ## Machine Learning
# 
# ### Creating Features and Label
# 

# Chosing 1% of total days as forecast, so length of forecast data is 0.01*length
print('length = ',len(df))
forecast_out = math.ceil(0.01*len(df))


# Creating label and shifting data as per 'forecast_out'
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head(2))


# If we look at the tail, it consists of forecast_out rows with NAN in Label column 
print(df.tail(2))
print('\n')
# We can simply drop those rows
df.dropna(inplace=True)
print(df.tail(2))


# Define features (X) and Label (y)
# For X drop label and index
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
print('X[1,:] = ', X[1,:])
print('y[1] = ',y[1])
print('length of X and y: ', len(X), len(y))


# ### Scaling the features
# 

# Use skalearn, preposessing to scale features
X = preprocessing.scale(X)
print(X[1,:])


# ### Creating Training and Test Sets
# 
# Using cross validation basically shuffles the data and according to our test_size criteria, splits the data into test and training data.
# 

# Cross validation (split into test and train data)
# test_size = 0.2 ==> 20% data is test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

print('length of X_train and x_test: ', len(X_train), len(X_test))


# ### Using Linear Regression
# Now time to use linear regression. I first Split the data into 80% of training data and 20% of test data. Used Linear regression to train and test data. Finally, I tested the accuracy of our model on the test data.
# 

# Train
clf = LinearRegression()
clf.fit(X_train,y_train)
# Test
accuracy = clf.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)


# ### Using SVM Regression
# 
# It seems like Linear regression did fairly well on the test data set. I also wanted to check another regression algorithm (support vector regression) just out of curiosity. This doesn't do as well as the linear regression, but is a lot more versatile and would be useful in other more complex cases.
# 

# Train
clf2 = svm.SVR()
clf2.fit(X_train,y_train)
# Test
accuracy = clf2.score(X_test, y_test)
print("Accuracy of SVM: ", accuracy)


# # Deep Learning with TensorFlow -  Convolutional Neural Network Model
# 
# The Convolutional Neural Network or "ConvNets" gained popularity through its use with image data, and is currently the state of the art for detecting what an image is, or what is contained in the image. ConvNets have been successful in identifying faces, objects and traffic signs apart from powering vision in robots and self driving cars. In this tutorial we'll work with MNIST dataset. MNIST is a simple computer vision dataset. It consists of images of handwritten digits like the image below. We will then train a ConvNets model on the training set using TensorFlow and make predictions on a test set.
# 
# 
# ![](https://www.tensorflow.org/images/MNIST.png)
# 
# Resources:
# 
# - [About MNIST](https://www.tensorflow.org/get_started/mnist/beginners)
# - [My basic Neural Network project on this dataset](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/)
# - [TensorFlow Tutorial on ConvNets](https://www.tensorflow.org/tutorials/deep_cnn)
# - [Tutorial by PythonProgramming.net](https://pythonprogramming.net/convolutional-neural-network-cnn-machine-learning-tutorial/?completed=/rnn-tensorflow-python-machine-learning-tutorial/)

# ## What are ConvNets
# 
# The basic CNN structure is as follows: Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output
# 
# ![](http://niharsarangi.com/wp-content/uploads/2014/10/convnet-1024x521.png)
# 
# There are three main operations in the ConvNet shown in Figure 3 above:
# 
# 1. Convolution: Convolution is the act of taking the original data, and creating feature maps from it. The convolutional layers are not fully connected like a traditional neural network.
# 2. Pooling or Sub Sampling: Pooling is sub-sampling, most often in the form of "max-pooling," where we select a region, and then take the maximum value in that region, and that becomes the new value for the entire region.
# 3. Classification (Fully Connected Layer): Fully Connected Layers are typical neural networks, where all nodes are "fully connected."
# 
# These operations are the basic building blocks of every Convolutional Neural Network, so understanding how these work is an important step to developing a sound understanding of ConvNets.

# ## Understanding and Importing the MNIST Data
# 
# We're going to be working first with the MNIST dataset, which is a dataset that contains 60,000 training samples and 10,000 testing samples of hand-written and labeled digits, 0 through 9, so ten total "classes." 
# 
# The MNIST dataset has the images (see example above), which we'll be working with as purely black and white, thresholded, images, of size 28 x 28, or 784 pixels total. 
# 
# ![](https://www.tensorflow.org/images/MNIST-Matrix.png)
# 
# Our features will be the pixel values for each pixel, thresholded. Either the pixel is "blank" (nothing there, a 0), or there is something there (1). Those are our features. We're going to attempt to just use this extremely rudimentary data, and predict the number we're looking at (a 0,1,2,3,4,5,6,7,8, or 9).
# 

import tensorflow as tf
# loading the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


# The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). 
# 
# For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.
# 

# ## Building the model: Setting up the Computation Model
# 
# ### Defining Variables
# To begin, we're mostly simply defining some starting variables, the class-size, and then we're defining the batch size to be 128.
# 

# import rnn from tensorFlow

batch_size = 128

# define number of classes = 10 for digits 0 through 9 
n_classes = 10 

# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])


# ### Functions for Convolution and Pooling
# 
# Next, we're going to define a couple simple functions that will help us with our convolutions and pooling. The functions here are the exact same as the ones from the offical TensorFlow CNN tutorial. 
# 
# - The strides parameter dictates the movement of the window. In this case, we just move 1 pixel at a time for the conv2d function.
# - The ksize parameter is the size of the pooling window and we move 2 pixel at a time.
# - Padding refers to operations on the windows at the edges, it is like a reflective boundary condition. 
# 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ### Convolutional Neural Network model
# 
# We have a weights/biases dictionary like basic deep nets, but they depend on specific convolutions algorithm as follows:  
# 
# ** Definitions of the layers **
# 
# - Each input image had a 28x28 pixels
# - We're taking 5x5 convolutions on the initial image, and producing 32 outputs. (Not set in stone, pick any similar number) 
# - Next, we take 5x5 convolutions of the 32 inputs and make 64 outputs. 
# - From here, we're left with 64 of 7x7 sized images (It went through two max pooling process. And each time, it reduces the dimension by half because of its stride of 2 and size of 2. Hence, 28/2/2 = 7)
# - then we're outputting to 1024 nodes in the fully connected layer. 
# - Then, the output layer is 1024 layers, to 10, which are the final 10 possible outputs for the actual label itself (0-9).
# 
# ** Apply appropriate functions to the layers **
# 
# - Reshape input to a 4D tensor 
# - Apply Non Linearity (ReLU) on the convoluted output (conv1)
# - The ReLU rectifier is an activation function defined as f(x)=max(0,x) 
# - Apply pooling and repeat the previous two steps for layer 2 (conv2)
# - Reshape conv2 output to fit fully connected layer
# - Apply ReLU on the fully connected layer
# - Obtain the output layer
# 

def convolutional_neural_network(x):#, keep_rate):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        
        # 5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        # Biases only depend on the outputs of a layer from above
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    # Convolution Layer, using our function and pass it through ReLU
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
   
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    
    # Output Layer
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


# 
# ## Training the model
# 
# Under a new function, train_neural_network, we will pass our output data.
# 
# - We then produce a prediction based on the output of that data through our convolutional_neural_network(). 
# - Next, we create a cost variable. This measures how wrong we are, and is the variable we desire to minimize by manipulating our weights. The cost function is synonymous with a loss function. 
# - To optimize our cost, we will use the AdamOptimizer, which is a popular optimizer along with others like Stochastic Gradient Descent and AdaGrad, for example.
# - Within AdamOptimizer(), we can optionally specify the learning_rate as a parameter. The default is 0.001, which is fine for most circumstances. 
# - Now that we have these things defined, we begin the session.
# 

hm_epochs = 4

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


# Our [basic neural network model](http://suruchifialoke.com/2017-06-15-predicting-digits_tensorflow/) gave an accuracy of ~95% for 10 epochs. This ConvNet model certainly gives better accuracy of ~98% for the same number of epochs. 
# 




# 
# # Introduction
# 
# This is my first project with Kaggle:
# [_Titanic: Machine Learning from Disaster
# Start here! Predict survival on the Titanic and get familiar with ML basics_](https://www.kaggle.com/c/titanic)
# 
# ![](https://s-media-cache-ak0.pinimg.com/originals/76/ab/d9/76abd9aa85d89cd53c5297129ea57cee.jpg)
# 
# This was a great project for me to start as the data is fairly clean and the calculations are relatively simple. 
# 
# My project has following parts
# * Feature engineering
# * Missing value imputation
# * Prediction!
# 
# ## Load and understand data
# 

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# I started off with the packages that I needed right away such as **numpy** and **pandas** 
# and added more as and when I needed more packages.
# Now lets take a look at our data that I have loaded in a variable called titanic_DF.
# 
# I have used the following command to see the first two rows of the data. 
# ```python
# titanic_DF.head(2)
# ```
# 

# Loading data and printing first few rows
titanic_DF = pd.read_csv('train.csv')
test_DF = pd.read_csv('test.csv')

titanic_DF.head(2)


# Similarly look into test data 
test_DF.head(2)


# Another way to understand the data and find out if there are missing data is to use 
# ```python 
# data.describe()
# ```
# or 
# ```python 
# data.info()
# ```
# 
# This command will print all the statistical information of the data, including how many data points each column have. For instance you can see the age column only has 714 non NULL data  as opposed to PassengerId that has 891.
# 
# Similarly the test data also has missing values in several columns. I have commented the test_DF.info() out, but you can uncomment and check. 
# 

# Previewing the statistics of training data and test data
titanic_DF.info()
# print('')
# test_DF.info()


# Data Visualization 
plt.rc('font', size=24)
fig = plt.figure(figsize=(18, 8))
alpha = 0.6

# Plot pclass distribution
ax1 = plt.subplot2grid((2,3), (0,0))
titanic_DF.Pclass.value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Pclass.value_counts().plot(kind='barh',color='magenta', label='test', alpha=alpha)
ax1.set_ylabel('Pclass')
ax1.set_xlabel('Frequency')
ax1.set_title("Distribution of Pclass" )
plt.legend(loc='best')

# Plot sex distribution
ax2 = plt.subplot2grid((2,3), (0,1))
titanic_DF.Sex.value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Sex.value_counts().plot(kind='barh', color='magenta', label='test', alpha=alpha)
ax2.set_ylabel('Sex')
ax2.set_xlabel('Frequency')
ax2.set_title("Distribution of Sex" )
plt.legend(loc='best')


# Plot Embarked Distribution
ax5 = plt.subplot2grid((2,3), (0,2))
titanic_DF.Embarked.fillna('S').value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Embarked.fillna('S').value_counts().plot(kind='barh',color='magenta', label='test', alpha=alpha)
ax5.set_ylabel('Embarked')
ax5.set_xlabel('Frequency')
ax5.set_title("Distribution of Embarked" )
plt.legend(loc='best')

# Plot Age distribution
ax3 = plt.subplot2grid((2,3), (1,0))
titanic_DF.Age.fillna(titanic_DF.Age.median()).plot(kind='kde', color='blue', label='train', alpha=alpha)
test_DF.Age.fillna(test_DF.Age.median()).plot(kind='kde',color='magenta', label='test', alpha=alpha)
ax3.set_xlabel('Age')
ax3.set_title("Distribution of age" )
plt.legend(loc='best')

# Plot fare distribution
ax4 = plt.subplot2grid((2,3), (1,1))
titanic_DF.Fare.fillna(titanic_DF.Fare.median()).plot(kind='kde', color='blue', label='train', alpha=alpha)
test_DF.Fare.fillna(test_DF.Fare.median()).plot(kind='kde',color='magenta', label='test', alpha=alpha)
ax4.set_xlabel('Fare')
ax4.set_title("Distribution of Fare" )
plt.legend(loc='best')

plt.tight_layout()



# We’ve got a good sense of our 12 variables in our training data frame, variable types, and the variables that have missing data. The next section now is feature engineering!
# 

# # Feature Engineering
# 
# Feature engineering is the process of using our knowledge of the data to create features that make machine learning algorithms work. And as per Andrew Ng, 
# 
# > Coming up with features is difficult, time-consuming, requires expert knowledge. 
# > "Applied machine learning" is basically feature engineering. 
# 
# So the challenge for me as a beginner was to pay a lot of attention to the various variables that could be potential features and with an open mind. Lets list our potential features one more time and the data dictionary to decide which ones we can use.
# 
# Data Dictionary
# 
# 
# | Variable      | Definition    | Key  |
# | ------------- |:-------------:| -----:|
# | survival      | _Survival_ | 0 = No, 1 = Yes |
# | pclass      | _Ticket class_      |   1 = 1st, 2 = 2nd, 3 = 3rd |
# | sex |  _Sex_      |     |
# | Age | _Age in years_| |
# | sibsp| _# of siblings / spouses aboard_| |
# | parch|_# of parents / children aboard_ | |
# | ticket|	_Ticket number_|	|
# | fare	|_Passenger fare_|	|
# | cabin	| _Cabin number_ |	|
# | embarked|	_Port of Embarkation_|	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# 

# print the names of the columns in the data frame
titanic_DF.columns
# Check which columns have missing data
for column in titanic_DF.columns:
    if np.any(pd.isnull(titanic_DF[column])) == True:
        print(column)


# ## Visualizing Features
# First I generated a distribution of various features for both training and test data of system to understand which factors are important. [Credit](https://www.kaggle.com/arthurlu/titanic/exploratory-tutorial-titanic)
# 
# Then, I started plotting the survival frequency to understand my features better and in the process play around with plotting styles (I used Seaborn for most of these plots).
# 
# [Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics](https://seaborn.pydata.org/introduction.html#introduction)
# 

# Plot pclass distribution
fig = plt.figure(figsize=(6, 6))

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_DF, size=4,color="green")
plt.ylabel('Fraction Survived')
plt.xlabel('Pclass')
plt.title("Survival according to Class" )


# Plot Gender Survival
fig = plt.figure(figsize=(6,6))
sns.factorplot('Sex','Survived', data=titanic_DF, size=4,color="green")
plt.ylabel('Fraction Survived')
plt.xlabel('Gender')
plt.title("Survival according to Gender" )


# Plot Fare
fig = plt.figure(figsize=(15, 6))
titanic_DF[titanic_DF.Survived==0].Fare.plot(kind='density', color='red', label='Died', alpha=alpha)
titanic_DF[titanic_DF.Survived==1].Fare.plot(kind='density',color='green', label='Survived', alpha=alpha)
plt.ylabel('Density')
plt.xlabel('Fare')
plt.xlim([-100,200])
plt.title("Distribution of Fare for Survived and Did not survive" )

plt.legend(loc='best')
plt.grid()


# 
# - Passenger Class (pclass): Lets start with Passenger Class, Here I have plotted factors survived as a function of passenger class. This seems like a no-brainer, passengers in better classes were certainly evacuated first. There is near linear correlation!
# 
# - Sex of passenger: Again there is a strong correlation between the sex and survival.
# - sibsp and parch
# 
# 
#  sibsp =  _# of siblings / spouses aboard_
# 
#  parch = _# of parents / children aboard_ 
#  
#  These features are obviously not linearly correleted with survival, but seem to have some complex dependence 
#  - ticket: There shouldn't be any use to ticket data as it seems they are some unique number generated per person. At this point we might as well drop ticket from our dataframe.
# 
# - Fare: Lets see how it fares! On its own it doesnt have any striking correlation with survival frequency.
# 

# # Fix Missing Data
# 
# Now that we have broadly looked at single features for all the columns that didn't have missing data. Time to fix the columns with missing data. The missing data are in columns Age, Embarked and Cabin so lets figure out of fix these. For age it makes sense to simply fill the data by median age.
# 

# Filling missing age data with median values
titanic_DF["Age"] = titanic_DF["Age"].fillna(titanic_DF["Age"].median())
titanic_DF.describe()


# Plot age
fig = plt.figure(figsize=(15, 6))
titanic_DF[titanic_DF.Survived==0].Age.plot(kind='density', color='red', label='Died', alpha=alpha)
titanic_DF[titanic_DF.Survived==1].Age.plot(kind='density',color='green', label='Survived', alpha=alpha)
plt.ylabel('Density')
plt.xlabel('Age')
plt.xlim([-10,90])
plt.title("Distribution of Age for Survived and Did not survive" )
plt.legend(loc='best')
plt.grid()


# For embarked there are multiple choices:
# 
# 1. Fill it using the most frequent option 'S'.
# 2. Use the fare as the fare price might be dependent on the port embarked.
# 
# Here I have used the simpler option 1, but there are many Notebooks that describe option2 on Kaggle.
# I have further converted type 'S', 'C' and 'Q' to 0, 1 and 2 respectively to be able to train data.
# 

# data cleaning for Embarked
print (titanic_DF["Embarked"].unique())
print (titanic_DF.Embarked.value_counts())


# filling Embarked data with most frequent 'S'
titanic_DF["Embarked"] = titanic_DF["Embarked"].fillna('S')
titanic_DF.loc[titanic_DF["Embarked"] == 'S', "Embarked"] = 0
titanic_DF.loc[titanic_DF["Embarked"] == 'C', "Embarked"] = 1
titanic_DF.loc[titanic_DF["Embarked"] == 'Q', "Embarked"] = 2


# convert female/male to numeric values (male=0, female=1)
titanic_DF.loc[titanic_DF["Sex"]=="male","Sex"]=0
titanic_DF.loc[titanic_DF["Sex"]=="female","Sex"]=1
titanic_DF.head(5)


# # Train a model: Logistic Regression
# 
# For our titanic dataset, our prediction is a binary variable, which is discontinuous. So using a logistic regression model makes more sense than using a linear regression model. So in the following snippet I have used python library to perform logistic regression using the featured defined in predictors.
# 

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

# columns we'll use to predict outcome
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]


# instantiate the model
logreg = LogisticRegression()

# perform cross-validation
print(cross_val_score(logreg, titanic_DF[predictors], titanic_DF['Survived'], cv=10, scoring='accuracy').mean())


# # Kaggle Submission
# 
# Now we need to run our prediction on the test data set and Submit to Kaggle.
# 
# 

# print the names of the columns in the data frame
test_DF.columns
# Check which columns have missing data
for column in test_DF.columns:
    if np.any(pd.isnull(test_DF[column])) == True:
        print(column)


# Filling missing age data with median values
test_DF["Age"] = test_DF["Age"].fillna(titanic_DF["Age"].median())

# filling Embarked data with most frequent 'S'
test_DF["Embarked"] = test_DF["Embarked"].fillna('S')
test_DF.loc[test_DF["Embarked"] == 'S', "Embarked"] = 0
test_DF.loc[test_DF["Embarked"] == 'C', "Embarked"] = 1
test_DF.loc[test_DF["Embarked"] == 'Q', "Embarked"] = 2

# convert female/male to numeric values (male=0, female=1)
test_DF.loc[test_DF["Sex"]=="male","Sex"]=0
test_DF.loc[test_DF["Sex"]=="female","Sex"]=1

test_DF.describe()


# Test also has empty fare columns
test_DF["Fare"] = test_DF["Fare"].fillna(test_DF["Fare"].median())


# Apply our prediction to test data
logreg.fit(titanic_DF[predictors], titanic_DF["Survived"])
prediction = logreg.predict(test_DF[predictors])


# Create a new dataframe with only the columns Kaggle wants from the dataset
submission_DF = pd.DataFrame({ 
    "PassengerId" : test_DF["PassengerId"],
    "Survived" : prediction
    })
print(submission_DF.head(5))


# prepare file for submission
submission_DF.to_csv("submission.csv", index=False)





# 
# # Machine Learning on Iris Dataset Part 2: Unsupervised Machine Learning Using Python
# 
# In my previous blogpost, I used supervised machine learning to classify the variety of Iris flower. The prediction was based on shape of an iris leaf represented by its sepal length, sepal width, petal length and petal width as shown in the image. However mush such classification problems come without labels and unsupervised machine learning is required to find clusters in the data, based on the features. 
# 
# ![Iris](./iris.png)
# 
# 
# ## Understanding and loading the data
# 
# I used the dataset from the [DataQuest website](https://www.dataquest.io/m/51/introduction-to-neural-networks) that only contains two species of iris.
# The data set consists of:
# * **110 samples** 
# * **2 labels: species of Iris (Iris virginica_ and _Iris versicolor_)** 
# * **4 features: length and the width of the sepals and petals**, in centimetres.
# 
# 

# install relevant modules
import numpy as np
import pandas as pd

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})

import seaborn as sns
sns.set(style="white", color_codes=True)


# Read in dataset
iris = pd.read_csv("iris.csv")
# shuffle rows
shuffled_rows = np.random.permutation(iris.index)
iris = iris.loc[shuffled_rows,:]
print(iris.head())


iris.hist(sharey=True, sharex=True, figsize=(12,8), ylabelsize=20, xlabelsize=20)
plt.show()


# Its clear from the histograms above that the petal length has the largest variability as opposed to the sepal width which seems to have a much smaller variance.
# 

# ## Neural Networks
# 
# ### Define a Neuron
# 
# Neural networks are very loosely inspired by the structure of neurons in the human brain. These models are built by using a series of activation units, known as neurons, to make predictions of some outcome. Neurons take in some input, apply a transformation function, and return an output. In the following code cell, we 
# 
# 1. Write a function called sigmoid_activation with inputs 'x' a feature vector and 'theta' a parameter vector of the same length to implement the sigmoid activation function.
# 
# $$ h_\theta = \frac{1}{1+e^{-\theta^T X}}$$
# 
# 2. Assign the value of sigmoid_activation(x0, theta_init) to a1. a1 should be a vector
# 

# Variables to test sigmoid_activation
iris["ones"] = np.ones(iris.shape[0])
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = (iris.species == 'Iris-versicolor').values.astype(int)

# The first observation
x0 = X[0]

# Initialize thetas randomly 
theta_init = np.random.normal(0,0.01,size=(5,1))

print(x0, theta_init)

def sigmoid_activation(x, theta):
    sigmoid = 1 / (1 + np.exp(-np.dot(theta.T, x)))
    return(sigmoid)

a1 = sigmoid_activation(x0, theta_init)
print(a1)


# ### Cost function
# 
# We can train a single neuron as a two layer network using gradient descent. We need to minimize a cost function which measures the error in our model. The cost function measures the difference between the desired output and actual output.
# 
# - We first write a function, single_cost(), that can compute the cost from just a single observation.
#  $$ J_\theta = - y \log h - (1-y) log (1-h)$$
#  
# - This function should use input features X, targets y, and parameters theta to compute the cost function.
# - Assign the cost of variables x0, y0, and theta_init to variable first_cost.
# 

# First observation's features and target
x0 = X[0]
y0 = y[0]

# Initialize parameters, we have 5 units and just 1 layer
theta_init = np.random.normal(0,0.01,size=(5,1))

# This function should use input features X, targets y, 
# and parameters theta to compute the cost function.
def single_cost(x, y, theta):
    h = sigmoid_activation(x.T, theta)
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return(cost)

first_cost = single_cost(x0, y0, theta_init)
print(first_cost)


# ### Compute the Gradients
# 
# Calculating derivatives are more complicated in neural networks than in linear regression. Here we compute the overall error and then distribute that error to each parameter. 
# For single layer, we use the quardratic cost function for error estimates:
# 
# $$ J_\theta = -\frac{1}{2} (y-h)^2 \to \frac{\partial J_\theta}{\partial h} = y-h$$
# 
# and the derivative of the logit function is given by:
# 
# $$\frac{\partial h}{\partial \theta} = h(1-h)X$$
# 
# hence by using chain rule, 
# 
# $$\frac{\partial J_\theta}{\partial \theta}  = (y-h) h (1-h) X$$
# 
# 
# In the following code we:
# 
# - Compute the average gradients over each observation in X and corresponding target y with the initialized parameters theta_init.
# - Assign the average gradients to variable grads
# 

## Compute the Gradients ##

# Initialize parameters
theta_init = np.random.normal(0,0.01,size=(5,1))

# Store the updates into this array
grads = np.zeros(theta_init.shape)

# Number of observations 
n = X.shape[0]

for j, obs in enumerate(X):
    h = sigmoid_activation(obs, theta_init)
    delta = (y[j]-h) * h * (1-h) * obs
    grads += delta[:,np.newaxis]/X.shape[0]

print(grads)


# ### Gradient Descent
# 
# Now that we can compute the gradients, we use gradient descent to learn the parameters and predict the species of iris flower given the 4 features. Gradient descent minimizes the cost function by adjusting the parameters accordingly. We adjust the parameters by substracting the product of the gradients and the learning rate from the previous parameters. Finally, we repeat until the cost function coverges or a maximum number of iterations is reached.
# 
# #### Pseudo-Code
# 
# ```python
# while (number_of_iterations < max_iterations 
#     and (prev_cost - cost) > convergence_thres ):
#     update paramaters
#     get new cost
#     repeat
# ```
# 

## Gradient descent ##
theta_init = np.random.normal(0,0.01,size=(5,1))

# set a learning rate
learning_rate = 0.1

# maximum number of iterations for gradient descent
maxepochs = 10000     

# costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.0001  

def learn(X, y, theta, learning_rate, maxepochs, convergence_thres):
    costs = []
    cost = single_cost(X, y, theta)  # compute initial cost
    costprev = cost + convergence_thres + 0.01  # set an inital costprev to past while loop
    counter = 0  # add a counter
    # Loop through until convergence
    for counter in range(maxepochs):
        grads = np.zeros(theta.shape)
        for j, obs in enumerate(X):
            h = sigmoid_activation(obs, theta)   # Compute activation
            delta = (y[j]-h) * h * (1-h) * obs   # Get delta
            grads += delta[:,np.newaxis]/X.shape[0]  # accumulate
        
        # update parameters 
        theta += grads * learning_rate
        counter += 1  # count
        costprev = cost  # store prev cost
        cost = single_cost(X, y, theta) # compute new cost
        costs.append(cost)
        if np.abs(costprev-cost) < convergence_thres:
            break
        
    plt.figure(figsize=(4,4))
    plt.plot(costs) 
    plt.title("Convergence of the Cost Function")
    plt.ylabel("J($\Theta$)")
    plt.xlabel("Iteration")
    plt.show()
    return theta
        
theta = learn(X, y, theta_init, learning_rate, maxepochs, convergence_thres)


# ## 2-layer Network
# Neural networks are usually built using mulitple layers of neurons. Adding more layers into the network allows to learn more complex functions.
# 

# ### Feed Forward
# 
# Here we are organizing multiple logistic regression models to create a more complex function.
# - Write a function feedforward() that will take in an input X and two sets of parameters theta0 and theta1 to compute the output hΘ(X).
# - Assign the output to variable h using features X and parameters theta0_init and theta1_init
# 

# two layer neural nets

theta0_init = np.random.normal(0,0.01,size=(5,4))
theta1_init = np.random.normal(0,0.01,size=(5,1))

def feedforward(X, theta0, theta1):
    # feedforward to the first layer
    a1 = sigmoid_activation(X.T, theta0).T
    
    # add a column of ones for bias term
    a1 = np.column_stack([np.ones(a1.shape[0]), a1])
    
    # activation units are then input to the output layer
    out = sigmoid_activation(a1.T, theta1)
    return out

h = feedforward(X, theta0_init, theta1_init)

#print(h)


# ### Multiple neural network cost function
# 
# The cost function to multiple layer neural networks is identical to the cost function we used for two networks, but hΘ(x) is more complicated.
# 

## Multiple neural network cost function ##

theta0_init = np.random.normal(0,0.01,size=(5,4))
theta1_init = np.random.normal(0,0.01,size=(5,1))

# X and y are in memory and should be used as inputs to multiplecost()

def multiplecost(X, y, theta0, theta1):
    # feed through network
    h = feedforward(X, theta0, theta1) 
    
    # compute error
    inner = y * np.log(h) + (1-y) * np.log(1-h)
    
    # negative of average error
    return -np.mean(inner)

c = multiplecost(X, y, theta0_init, theta1_init)
print(c)


# ## Multiple-layer Network and Backpropagation
# 
# Backpropagation focuses on updating parameters starting at the last layer and circling back through each layer, updating accordingly. We have reused feedforward() and multiplecost() but in more condensed forms. During initialization, we set attributes like the learning rate, maximum number of iterations to convergence, and number of units in the hidden layer. In learn() we have the backpropagation algorithm, which computes the gradients and updates the parameters. We then test the class by using the features and the species of the flower.
# 

# Use a class for this model
class NNet3:
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)
        
    def _multiplecost(self, X, y):
        # feed through network
        l1, l2 = self._feedforward(X) 
        # compute error
        inner = y * np.log(l2) + (1-y) * np.log(1-l2)
        # negative of average error
        return -np.mean(inner)
    
    def _feedforward(self, X):
        # feedforward to the first layer
        l1 = sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2
    
    def predict(self, X):
        _, y = self._feedforward(X)
        return y
    
    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
        
        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1  # set an inital costprev to past while loop
        counter = 0  # intialize a counter

        # Loop through until convergence
        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self._feedforward(X)

            # Start Backpropagation
            # Compute gradients
            l2_delta = (y-l2) * l2 * (1-l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate
            
            # Store costs and check for convergence
            counter += 1  # Count
            costprev = cost  # Store prev cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break


# ### Set a Learning Rate
# 

# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001  
# Number of hidden units
hidden_units = 4

# Initialize model 
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
model.learn(X, y)

# Plot costs
plt.figure(figsize=(4,4))
plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()


# ## Train a Model
# 
# Now that we have learned about neural networks, learned about backpropagation, and have code which will train a 3-layer neural network, we will split the data into training and test datasets and run the model.
# 
# ### Test-Train Split
# 

# First 70 rows to X_train and y_train
# Last 30 rows to X_train and y_train

X_train = X[:70]
y_train = y[:70]

X_test = X[-30:]
y_test = y[-30:]

#print(X_train, y_train, X_test, y_test)


# ## Make predictions
# 
# To benchmark how well a three layer neural network performs when predicting the species of iris flowers, we compute the AUC, area under the curve, score of the receiver operating characteristic. 
# 
# The function NNet3 not only trains the model but also returns the predictions. The method predict() will return a 2D matrix of probabilities. Since there is only one target variable in this neural network, we select the first row of this matrix, which corresponds to the type of flower.
# 

## Predicting iris flowers ##

from sklearn.metrics import roc_auc_score
# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000       
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001  
# Number of hidden units
hidden_units = 4

# Initialize model 
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)

model.learn(X_train, y_train)

yhat = model.predict(X_test)[0]
auc = roc_auc_score(y_test, yhat)
print(auc)


# # PCA on Iris Varieties
# 
# 
# The sheer size of data in the modern age is not only a challenge for computer hardware but also a main bottleneck for the performance of many machine learning algorithms. The main goal of a PCA analysis is to identify patterns in data; PCA aims to detect the correlation between variables. If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense. In a nutshell, this is what PCA is all about: Finding the directions of maximum variance in high-dimensional data and project it onto a smaller dimensional subspace while retaining most of the information.
# 
# ![Iris](./iris.png)
# 
# A Summary of the PCA Approach
# 
# - Standardize the data.
# - Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
# - Sort eigenvalues in descending order and choose the **k** eigenvectors that correspond to the **k** largest eigenvalues where **k** is the number of dimensions of the new feature subspace (**k**≤**d**)/.
# - Construct the projection matrix W from the selected **k** eigenvectors.
# - Transform the original dataset X via W to obtain a **k**-dimensional feature subspace Y.
# 
# ## Understanding and loading the data
# 
# The data set consists of:
# * **150 samples** 
# * **3 labels: species of Iris (_Iris setosa, Iris virginica_ and _Iris versicolor_)** 
# * **4 features: length and the width of the sepals and petals**, in centimetres.
# 
# 

# install relevant modules
import numpy as np
import pandas as pd

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})

import seaborn as sns
sns.set(style="white", color_codes=True)


# Read file
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.head(2)


# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values

print(X[:1, :])
print(y[:1])


# ## Exploratory Visualization
# 
# 
# Our iris dataset is now stored in form of a 150 x 4 matrix where the columns are the different features, and every row represents a separate flower sample. Each sample row X can be pictured as a 4-dimensional vect:
# 
# $$ \mathbf{x^T} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} 
# = \begin{pmatrix} \text{sepal length} \\ \text{sepal width} \\\text{petal length} \\ \text{petal width} \end{pmatrix}$$
# 

import seaborn as sns

#g = sns.FacetGrid(df, hue="class", col=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'], col_wrap=4, sharex=False)

g = sns.FacetGrid(df, col="class", size=5, aspect = 1.2)
g.map(sns.kdeplot, "sepal_len", shade=True).fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "petal_len", shade=True, color = "r").fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "petal_wid", shade=True, color = "g").fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "sepal_wid", shade=True, color = "m").fig.subplots_adjust(wspace=.3)
sns.set(font_scale=2)


# ## Standardizing
# 
# Whether to standardize the data prior to a PCA on the covariance matrix depends on the measurement scales of the original features. Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms.
# 

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

X_std[:5]


# ## Eigendecomposition - Computing Eigenvectors and Eigenvalues
# 
# The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the "core" of a PCA: The eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.
# 
# ### Covariance Matrix
# The classic approach to PCA is to perform the eigendecomposition on the covariance matrix ΣΣ, which is a dXd matrix where each element represents the covariance between two features. The covariance between two features is calculated as follows:
# 
# $$\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^{N}\left(  x_{ij}-\bar{x}_j \right)  \left( x_{ik}-\bar{x}_k \right)$$
# 
# We can summarize the calculation of the covariance matrix via the following matrix equation:
# 
# $$\Sigma = \frac{1}{n-1} \left( (\mathbf{X} - \mathbf{\bar{x}})^T\;(\mathbf{X} - \mathbf{\bar{x}}) \right)$$
# 
# where $\bar{x}$ is the mean vector $$\mathbf{\bar{x}} = \sum\limits_{k=1}^n x_{i}.$$
# 
# The mean vector is a dd-dimensional vector where each value in this vector represents the sample mean of a feature column in the dataset.
# 

# Calculate covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

# Could alternatively use:
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# Eigendecomposition on the covariance matrix:

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# ### Correlation Matrix
# 
# The correlation matrix typically used instead of the covariance matrix. However, the eigendecomposition of the covariance matrix (if the input data was standardized) yields the same results as a eigendecomposition on the correlation matrix, since the correlation matrix can be understood as the normalized covariance matrix. 
# 

# Eigendecomposition on the Correlation matrix:

cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# ## Singular Vector Decomposition
# 
# While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. So, let us perform an SVD to confirm that the result are indeed the same:
# 

u,s,v = np.linalg.svd(X_std.T)
u


# ## Selecting Principal Components
# 
# The typical goal of a PCA is to reduce the dimensionality of the original feature space by projecting it onto a smaller subspace, where the eigenvectors will form the axes.
# 
# In order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.
# 
# In order to do so, the common approach is to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors.
# 

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# After sorting the eigenpairs, the next question is "how many principal components are we going to choose for our new feature subspace?" A useful measure is the so-called "explained variance," which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.
# 

# Explained variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
#pc_col = ['PC1', 'PC2', 'PC3', 'PC4']
index = [1, 2, 3, 4]
plt.bar(pc_col, var_exp)
plt.xticks(index , ('PC1', 'PC2', 'PC3', 'PC4'))
plt.plot(index , cum_var_exp, c = 'r')
plt.ylabel('Explained variance')
plt.show()


# The plot above clearly shows that most of the variance (72.77% of the variance to be precise) can be explained by the first principal component alone. The second principal component still bears some information (23.03%) while the third and fourth principal components can safely be dropped without losing to much information. Together, the first two principal components contain 95.8% of the information.
# 
# ### Construction of the projection matrix 
# 
# Projection matrix  will be used to transform the Iris data onto the new feature subspace.Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the "top 2" eigenvectors with the highest eigenvalues to construct our dxk-dimensional eigenvector matrix W.
# 

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


# ## Projection Onto the New Feature Space
# 
# In this last step we will use the 4×2-dimensional projection matrix W to transform our samples onto the new subspace via the equation
# Y=X×W, where Y is a 150×2 matrix of our transformed samples.
# 

Y = X_std.dot(matrix_w)
Y[:3]


y[:5]


# ## PCA in scikit-learn
# 
# Simple implementation of PCA in scikit-learn is as follows:
# 

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

Y_sklearn[:3]
new_list = []
for row in range(len(y)):
    if y[row]=='Iris-setosa':
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-setosa'])
    elif y[row]=='Iris-versicolor':
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-versicolor'])
    else:
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-verginica'])
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1])
plt.show()





# # Naive Bayes to classify movie reviews based on sentiment
# 
# ## From scratch and with Scikit-Learn
# 
# We want to predict whether a review is negative or positive, based on the text of the review. We'll use Naive Bayes for our classification algorithm. A Naive Bayes classifier works by figuring out how likely data attributes are to be associated with a certain class. Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.
# 
# 

# ## What is Naive Bayes?
# Naive Bayes classifier is based on Bayes' theorem, which is:
# 
# $$P(H \mid x) = \frac{P(H) P(x \mid H) }{P(x)}$$
# 
# - P(H|x) is the posterior probability of hypothesis (H or target) given predictor (x or attribute).
# - P(H) is the prior probability of hypothesis
# - P(x|H) is the likelihood which is the probability of predictor given hypothesis.
# - P(x) is the prior probability of predictor.
# 
# Naive Bayes extends Bayes' theorem to handle multiple evidences by assuming that each evidence is independent.
# 
# $$ P(H \mid x_1, \dots, x_n) = \frac{P(H) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \dots, x_n)}$$
# 
# 
# In most of these problems we will compare the probabilities H being true or false and the denominator will not affect the outcome so we can simply calculate the numerator.  
# 
# **Example**
# ![](movie_review.png)

# ## Loading the data
# 
# We'll be working with a CSV file containing movie reviews. Each row contains the text of the review, as well as a number indicating whether the tone of the review is positive(1) or negative(-1).
# 
# We want to predict whether a review is negative or positive, based on the text alone. To do this, we'll train an algorithm using the reviews and classifications in train.csv, and then make predictions on the reviews in test.csv. We'll be able to calculate our error using the actual classifications in test.csv to see how good our predictions were.
# 

import csv
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))
    
print(len(reviews))


# ## Training a Model
# 
# 
# ### Obtaining P(H) : the prior probability of hypothesis (positive review)
# 
# Now that we have the word counts, we just need to convert them to probabilities and multiply them out to predict the classifications.
# 
# Let's start by obtaining the prior probabilities as follows:
# 

# Computing the prior(H=positive reviews) according to the Naive Bayes' equation
def get_H_count(score):
    # Compute the count of each classification occurring in the data
    return len([r for r in reviews if r[1] == str(score)])

# We'll use these counts for smoothing when computing the prediction
positive_review_count = get_H_count(1)
negative_review_count = get_H_count(-1)

# These are the prior probabilities (we saw them in the formula as P(H))
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)
print("P(H) or the prior is:", prob_positive)


# ### Obtaining P(xi|H) : the likelihood
# 
# #### Finding Word Counts
# 
# We're trying to determine if we should classify a data row as negative or positive. The easiest way to generate features from text is to split the text up into words. Each word in a movie review will then be a feature that we can work with. To do this, we'll split the reviews based on whitespace.
# 
# Afterwards, we'll count up how many times each word occurs in the negative reviews, and how many times each word occurs in the positive reviews. Eventually, we'll use the counts to compute the probability that a new review will belong to one class versus the other.
# 
# [We will use the following python library](https://docs.python.org/2/library/collections.html)
# ***class collections.Counter([iterable-or-mapping])***
# 
# Where, a Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. The Counter class is similar to bags or multisets in other languages.
# 

# Python class that lets us count how many times items occur in a list
from collections import Counter
import re

def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone
    # Lowercase the text so that the algorithm doesn't see "Not" and "not" as different words, for example
    return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(text):
    # Split text into words based on whitespace -- simple but effective
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)

negative_text = get_text(reviews, -1)
positive_text = get_text(reviews, 1)

# Generate word counts(WC) dictionary for negative tone
negative_WC_dict = count_text(negative_text)

# Generate word counts(WC) dictionary for positive tone
positive_WC_dict = count_text(positive_text)


# example
print("count of word 'awesome' in positive reviews", positive_WC_dict.get("awesome"))
print("count of word 'movie' in positive reviews", positive_WC_dict.get("movie"))

print("count of word 'awesome' in negative reviews", negative_WC_dict.get("awesome"))
print("count of word 'movie' in negative reviews", negative_WC_dict.get("movie"))


# #### Obtaining P(xi|H) the likelyhood or the probability of predictor (a word) given hypothesis (positive review)
# 
# - For every word in the text, we get the number of times that word occurred in the text, text_WC_dict.get(word) 
# - Multiply it with the probability of that word in Hypothesis (e.g. positive review) = (H_WC_dict.get(word) / (sum(H_WC_dict.values())
# 
# 
# ```python
# prediction =  text_WC_dict.get(word) * (H_WC_dict.get(word) / (sum(H_WC_dict.values())
# ```
# 
# We add 1 to smooth the value, smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data and correspondingly smooth the denominator counts to keep things even.
# 
# After smoothing above equation becomes:
# 
# ```python
# prediction =  text_WC_dict.get(word) * (H_WC_dict.get(word)+1) / (sum(H_WC_dict.values()+ H_count)
# ```
# 

# H = positive review or negative review
def make_class_prediction(text, H_WC_dict, H_prob, H_count):
    prediction = 1
    text_WC_dict = count_text(text)
    
    for word in text_WC_dict:       
        prediction *=  text_WC_dict.get(word,0) * ((H_WC_dict.get(word, 0) + 1) / (sum(H_WC_dict.values()) + H_count))

        # Now we multiply by the probability of the class existing in the documents
    return prediction * H_prob


# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater
def make_decision(text):
    
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_WC_dict, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_WC_dict, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return -1
    return 1

print("For this review: {0}".format(reviews[0][0]))
print("")
print("The predicted label is ", make_decision(reviews[0][0]))
print("The actual label is ", reviews[0][1])


make_decision("movie")


make_decision("best movie ever")


# ## Making Predictions
# 
# Now that we can make predictions, let's predict the probabilities for the reviews in test.csv
# 

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0]) for r in test]


# ### Error analysis on predictions
# 

actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))


# ## Naive Bayes implementation in scikit-learn
# 
# There are a lot of extensions we could add to this algorithm to make it perform better. We could remove punctuation and other non-characters. We could remove stopwords, or perform stemming or lemmatization.
# 
# We don't want to have to code the entire algorithm out every time, though. An easier way to use Naive Bayes is to use the implementation in scikit-learn. Scikit-learn is a Python machine learning library that contains implementations of all the common machine learning algorithms
# 

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# Generate counts from text using a vectorizer  
# We can choose from other available vectorizers, and set many different options
# This code performs our step of computing word counts
vectorizer = CountVectorizer(stop_words='english', max_df=.05)
train_features = vectorizer.fit_transform([r[0] for r in reviews])
test_features = vectorizer.transform([r[0] for r in test])

# Fit a Naive Bayes model to the training data
# This will train the model using the word counts we computed and the existing classifications in the training set
nb = MultinomialNB()
nb.fit(train_features, [int(r[1]) for r in reviews])

# Now we can use the model to predict classifications for our test features
predictions = nb.predict(test_features)

# Compute the error
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomal naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))


