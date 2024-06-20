# # _MLP_
# ## Multi Layer Perceptron Model (Feed forward Neural Networks)
# 

# An implementation of a Neural Network used for ATR (Automatic Target Recognition)
# 
# --------
# 

# dependencies
import tensorflow as tf
import numpy as np
import pickle


# load data
pickle_file = 'final_dataset.pickle'
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Now lets test if the file really matches or is corrupted
# train_labels[0] => 2
# so the first image is a BTR70 LETS test this out
print(train_labels[0])
with open('TRAIN_BTR70.pickle','rb') as f:
    s = pickle.load(f)
    btr_train = s
    del s
    for image in btr_train:
        if (image - train_dataset[0]).any() == 0:
            print('no problem')
            break
    print('done')


# **Reformat Data - **
# Flatten arrays and make labels 1-hot encoded arrays
# 

image_size = 128
num_labels = 3

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# hyper parameters
num_steps = 551
batch_size = 30
num_labels = 3
h_nodes = 200
beta = 0.01    


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


train_subset = 30
graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  # in tensorflow you create a bunch of nodes or operations - some are constant (do not require tensor input)
  # and some are not constant example matrix multilication -the end node that you want as output is supposed 
  # to be passed as a parameter to the session variable
  #placing inside constant means that you have do not perform any computation on these tensors
  # everything is an operation the below one produces a matrix  
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  # y = (W*x) + b 
  logits = tf.matmul(tf_train_dataset, weights) + biases
  # S(y)-> will be reduced to one hot encoded values then cross entropy will be calculated
  # the log function D(S,L) that is the loss
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # lets add the l2 regularization layer
  regularization = tf.nn.l2_loss(weights)
  loss = tf.reduce_mean(loss + beta*regularization)  
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# deeper network
graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Why will weigth1 be of the size 784*h_nodes
    
  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, h_nodes]))
  biases1 = tf.Variable(tf.zeros([h_nodes]))
  
  weights2 = tf.Variable(
  tf.truncated_normal([h_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
       

  # Training computation.  
  logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
  # now send these logits to relu
  relu_output = tf.nn.relu(logits1)
  # introduce dropout to outputs from the relu layer
  keep_prob = 0.5
  relu_output = tf.nn.dropout(relu_output,keep_prob)  
  final_logits = tf.matmul(relu_output,weights2) + biases2
  
    
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=final_logits))
  # now add regularization to it
  regularization = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
  loss = tf.reduce_mean(loss + beta*regularization)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(final_logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(
    tf.matmul(tf_valid_dataset, weights1) + biases1),weights2)+biases2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(
    tf.matmul(tf_test_dataset, weights1) + biases1),weights2)+biases2)


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ** We conclude that an MLP is not able to perform well on this task **
# 




# # Morphological Associative memories
# 

# dependencies
import matplotlib.pyplot as plt
import pickle
import numpy as np


f = open('final_dataset.pickle','rb')
dataset = pickle.load(f)


sample_image = dataset['train_dataset'][0]
sample_label = dataset['train_labels'][0]
print(sample_label)
plt.figure()
plt.imshow(sample_image)
plt.show()


# lets make Wxx and Mxx for this images
# Wxx
x = np.array([0,0,0])
y = np.array([0,1,0])
final = np.subtract.outer(y,x)
print(final)


# 1. flatten the whole image into n-pixels
x_vectors = sample_image.flatten()
# dimensions must be of the form img_len,1
# for this x_vector the weights must be of the order 1,num_perceptrons
# but this gives me a sparse matrix of the order img_len,num_perceptrons
# what we do here is take sum row wise we will get like  [1,0,0,1] will become [2]
# and therefore we will have outputs as img_len,1

# so here to get the weights for x_vectors we have to multiply matrices of the order
# img_len,1 and 1,img_len 


#x_vectors = np.array([1,2,5,1])
weights = np.subtract.outer(x_vectors, x_vectors)
print(weights)


add_individual = np.add(weights, x_vectors)
# pg-6 now perform row wise max
result = [max(row) for row in add_individual]
np.testing.assert_array_almost_equal(x_vectors, result)
print('done')
# for k=1 dimesions of Mxx and Wxx are same


# Now lets add some erosive noise to the image and then lets see the recall
# 

import cv2

erode_img = sample_image
# kernel is a pixel set like a cross( or any shape) which convolves and erodes
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(erode_img,kernel,iterations = 1)
plt.figure()
plt.imshow(erosion)
plt.show()


# Now lets try to do some recall
x_eroded = erosion
x_eroded_vector = x_eroded.flatten()

add_individual = np.add(weights, x_eroded_vector)
result = np.array([max(row) for row in add_individual])
# now lets reshape the result to 128 x 128
result.shape = (128, 128)
plt.figure()
plt.imshow(result)
plt.show()


# now lets see the amount of recall error
result = result.flatten()
np.testing.assert_array_almost_equal(result, x_vectors)
print('done 0%')


# **Further investigation will be done on obtaining kernel matrix and creating a Neural Network**
# 




