# ## Building a Neural Network with TensorFlow
# 
# Using TensorFlow's API to build a neural network as a tensor graph.
# 

import tensorflow as tf#Reset the tf graph
import numpy as np


#reset the TF graph

def reset_graph(seed=42): 
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


'''
Construction phase
----------------------------------------
Here we define the network in terms of: 
- input, hidden, output layers
- activation function: ReLu
- loss function: cross-entropy
- optimizer: mini-batch GD
- performacne evaluation: accuracy
'''

#specify the NN architecture

n_inputs = 28*28 # MNIST 
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


#We use placeholders for the input/output of the network. 
#For X, we know the number of features each instance will have (784) but we don't know
#how many instances each batch of the mini-batch GD will contain.

#Same for y, we know it's a single label per training instance but still rely on the
#number of instances in our mini-batch. Each Yi will contain the error of example Xi

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


#This function creates a layer in the NN. It takes an input layer X (units connected to this layer),
#number of neurons (units in the layer), name and an activation function.

#This has a TF implementation that does it- see TF's fully_connected function

def neuron_layer(X, n_neurons, name, activation=None): 
    #add namescope for better visualization in tensorboard
    with tf.name_scope(name):
        
        #get number of features from input
        n_inputs = int(X.get_shape()[1])
        
        #initialize W matrix with random weight.
        #W is the shape (n_features_input, n_neurons_nextlayer)
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev) 
        W = tf.Variable(init, name="weights")
        
        #bias edges in hidden layer
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        
        #hidden layer- net
        z = tf.matmul(X, W) + b
        
        #hidden layer- out
        if activation=="relu":
            return tf.nn.relu(z) 
        else:
            return z


#Create the DNN:
    #input layer: a training instance
    #hidden layer 1: 300
    #hidden layer 2: 100
    #output layer: 10

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu") 
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu") 
    logits = neuron_layer(hidden2, n_outputs, "outputs")


from tensorflow.contrib.layers import fully_connected

#The TF implementation for our 'neuron_layer' function

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1", reuse=True) 
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", reuse=True) 
    logits = fully_connected(hidden2, n_outputs, scope="outputs",
                                 activation_fn=None, reuse=True)


#define cross-entropy loss function for the output layer.

with tf.name_scope("loss"):
    
    #computes the cross-entropy loss of each output node
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    
    #computes the total error as a mean over the cross-entropies
    loss = tf.reduce_mean(xentropy, name="loss")


#define a gradient-descent optimizer to minimize the loss function

learning_rate = 0.01

with tf.name_scope("train"):
    #this will minimize the loss, a TF node that contains the rest 
    #of our nodes: X,W and y
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
    training_op = optimizer.minimize(loss)


#Model Evaluation

with tf.name_scope("eval"):
    
    #foreach prediction i (logits) determine wether the highest logit 
    #(amonge the topk predictions where k=1) corresponds to the target class
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


#initializer and saver

init = tf.global_variables_initializer()
saver = tf.train.Saver()


'''
Execution Phase
-------------------
- Load, scale and split the MNIST data: done by TF read_data_sets method
- Define execution parameters: ephocs and batch size
- Run training, evaluate every epoch
- Save network to disk
'''
from tensorflow.examples.tutorials.mnist import input_data

#load data using TF's API. Loads, scales and shuffles the data
mnist = input_data.read_data_sets("/tmp/data/")

#define mini-batch GD parameters
n_epochs = 30
batch_size = 50


with tf.Session() as sess: 
    #initialize all nodes in the graph
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            
            #get batch i - examples and targets
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            
            #feed batch to network - all examples in a batch will run in parallel.
            #populate the place holders in the graph
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        #train/test sets accuracy measurments
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels}) 
        
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        
    save_path = saver.save(sess, "./my_model_final.ckpt")


#Use the network on a "new" examples

from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Get 2 digits from the dataset
mnist = fetch_mldata('MNIST original')
target = mnist.target.reshape(mnist.target.shape[0], 1)
data = mnist.data

digit_5 = scaler.fit_transform(data[35000].reshape(28, 28))
digit_4 = scaler.fit_transform(data[29999].reshape(28, 28))
digits = np.c_[digit_5, digit_4]

plt.imshow(digits, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.show()




with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    
    digit4_new_scaled = digit_4.reshape(1,784)
    Z = logits.eval(feed_dict={X: digit4_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    
    print ('prediction: ', y_pred)


# <h1>Classification - Single Class</h1>
# 
# This is a notebook that demonstrates a single class classification. It uses the MNIST data set - a collection of 70000 hand written digit images. Each image is 28 * 28 pixels and labeled with a single digit. 
# 
# In this notebook we train a model to distinguish between the number 5 to the rest of the digits. We explore the metrics (precision, recall, etc...) of our trained model and try out 2 different models. 
# 

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


#fetch the data
mnist = fetch_mldata('MNIST original')


#get data and targets
target = mnist.target.reshape(mnist.target.shape[0], 1)
data = mnist.data

print ('shape of data and targets')
print ('\ndata shape: ', data.shape)
print ('target shape: ', target.shape)


#target proportions
def get_proportions(data):
    target_df = pd.DataFrame(data)
    value_counts = target_df[0].value_counts()
    num_targets = target_df.shape[0]

    return value_counts / len(data)

print ('target proportions')
get_proportions(target)


#see what an image from data looks like
print ('See arbitrary examples from the data')

digit_5 = data[35000].reshape(28, 28)
digit_4 = data[29999].reshape(28, 28)
digits = np.c_[digit_5, digit_4]
plt.imshow(digits, cmap = matplotlib.cm.binary, interpolation="nearest")


#train/test split (MNIST data set is already ready for split)
print ('splitting data to train and test sets')
X_train, X_test, y_train, y_test = data[:60000], data[60000:], target[:60000], target[60000:]

#shuffle training set so that same numbers don't appear in a row
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#check train targets proportions
print ('\ntrain targets proportions')
get_proportions(y_train)


from sklearn.linear_model import SGDClassifier

print ('training SGD Classifier...')

#Binary classifier for the digit 5- if 5 true, if not 5 false
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits. 
y_test_5 = (y_test == 5)

#stochastic gradient descent classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5.ravel())

#predict on our some_digit from before
print ('\ntesting predictions on arbitrary examples')
print ('testing 5: ', sgd_clf.predict([digit_5.ravel()]))
print ('testing 4: ', sgd_clf.predict([digit_4.ravel()]))


from sklearn.model_selection import StratifiedKFold 
from sklearn.base import clone

#k-fold cv implementation
def cross_validation(classifier, X, y, n_splits = 3):
    skfolds = StratifiedKFold(n_splits=n_splits, random_state=42)

    #need to reshape y to (y,) for cv
    y = y.reshape(y.shape[0],)
    
    print ('Cross validation results')
    
    for train_index, test_index in skfolds.split(X, y): 
        clone_clf = clone(classifier)
        X_train_folds = X[train_index]
        y_train_folds = (y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (y[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)

        print(n_correct / len(y_pred)) 

#testing our 5-only classifier
cross_validation(sgd_clf, X_train, y_train_5)


from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

#what accuracy will we get if we predict not-5 all the time?
#we're getting 90% (since only 10% of the examples are 5 - see proportions above).
#This means that accuracy measurement does not tell us much here
class Never5Classifier(BaseEstimator): 
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()

print ('Accuracy for predicting not-5 all the time')
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.model_selection import cross_val_predict

#Creating a confusion matrix for performance measuremence

#returns the predictions (instead of the performance scores)
y_train_5_reshaped = y_train_5.reshape(y_train_5.shape[0],)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5_reshaped, cv=3)
y_train_actuals = y_train_5

confusion_mat = confusion_matrix(y_train_actuals, y_train_pred)
print ('Confusion matrix')
print (confusion_mat)

tp = confusion_mat[1][1]
fp = confusion_mat[0][1]
tn = confusion_mat[0][0]
fn = confusion_mat[1][0]

print ('TP: {}, FP: {}, TN: {}, FN: {} \n'.format(tp, fp, tn, fn))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * ((precision * recall) / (precision + recall))
print ('precision: ', precision)
print ('recall: ', recall)
print ('f1: ', f1)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print ('metrics- SGD classifier')

#precision/recall tradeoff
#useful for when the positive examples are sparse

#by default the SGD classifier's threshold for positive example is above 0.
#get the confidence score of the classifier for true and false examples
print ('\nconfidence for a single positive example: ', sgd_clf.decision_function([digit_5.ravel()]))
print ('confidence for a single negative example: ', sgd_clf.decision_function([digit_4.ravel()]))

#get confidence level for each example in training set
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5_reshaped, cv=3, method="decision_function")
y_positive = y_scores[y_scores > 0.0]
y_negative = y_scores[y_scores < 0.0]

print ('number of positives: ', len(y_positive))
print ('number of positives: ', len(y_negative))
print ('avreage positive confidence: ', np.mean(y_positive))
print ('avreage negative confidence: ', np.mean(y_negative))

#plot precision and recall for increasing thresholds
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, x_range = [-600000, 600000]): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlim(x_range)
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

#get the precision and recall pairs for increasing threshold values
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

print ('\nPlot of precision-recall pairs for increasing threshold values')
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#example: setting a threshold that produces 90% precision
y_train_pred_90 = (y_scores > 70000)
print ('precision score for threshold 70000: ', precision_score(y_train_actuals, y_train_pred_90))


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#ROC Curve
#plots true positive rates (recall) against false positive rates for increasing threshold values

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
print ('ROC Curve- SGD classifier')
plot_roc_curve(fpr, tpr)

print ('Area under the curve: ', roc_auc_score(y_train_5, y_scores))


from sklearn.ensemble import RandomForestClassifier

print ('Training Random Forest classifier...')

#Changing our model - random forest classifier

#get the probabilities for True and False classifications
#column 1 - prob of negative, column 2 - prob of positive
#default cutoff - 0.5
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train_5_reshaped)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5_reshaped, cv=3, method="predict_proba")


print ('metrics- Random Forest classifier')

y_positive= y_probas_forest[y_probas_forest[:,1] >= 0.5]
y_negative = y_probas_forest[y_probas_forest[:,0] < 0.5]

print ('\nconfidence (probability) for a single positive example: ', forest_clf.predict_proba([digit_5.ravel()]))
print ('confidence (probability) for a single negative example: ', forest_clf.predict_proba([digit_4.ravel()]))
print ('Number of positive predictions: ', len(y_positive))
print ('Number of negative predictions: ', len(y_negative))
print ('avreage positive confidence (probability): ', np.mean(y_positive))
print ('avreage negative confidence (probability): ', np.mean(y_negative))

#get the precision and recall pairs for increasing threshold values
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_probas_forest[:,1])

print ('\nPlot of precision-recall pairs for increasing threshold values')
plot_precision_recall_vs_threshold(precisions, recalls, thresholds, x_range = [0, 1])

#Example: getting recall of ~95% and precision of ~84%
threshold = 0.30
y_recall_95 = (y_probas_forest[:,1] >= threshold)
print ('precision score for {} threshold : {} '.format(threshold, precision_score(y_train_actuals, y_recall_95)))
print ('recall score for {} threshold : {}'.format(threshold, recall_score(y_train_actuals, y_recall_95)))


print ('\nPlot ROC Curve or SGD classifier against random forest')
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")

print ('Area under the curve for the random forest classifers ROC curve: ', roc_auc_score(y_train_5, y_scores_forest))


