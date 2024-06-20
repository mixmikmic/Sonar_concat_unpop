# < Convolutional Neural Network - Restore >
# =================================================
# Loading an image with matplotlib.image
# ----------------------
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import os

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np


n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32, name = 'prob')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name='conv2d')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' , name = 'maxpool2d')


# load an image(28 x 28)
img1 = mpimg.imread( "/Users/moonsooyoung/Downloads/5-2.png")
plt.imshow(img1, cmap='Greys', interpolation='nearest')
plt.show()

image1=np.array(img1,np.float32)
image1[image1[:,:]<1]=0

data1=image1[:,:,2]
data1[data1[:,:]==0]=255
data1[data1[:,:]==1]=0

plt.imshow(data1)
plt.show()

data=data1.reshape(-1,784)
data= data/255

x=tf.cast(data, 'float')


weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32]), name = 'w_c1'),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64]), name = 'w_c2'),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024]), name = 'w_f'),
               'out':tf.Variable(tf.random_normal([1024, n_classes]), name ='w_out')}

biases = {'b_conv1':tf.Variable(tf.random_normal([32]), name = 'b_c1'),
               'b_conv2':tf.Variable(tf.random_normal([64]), name = 'b_c2'),
               'b_fc':tf.Variable(tf.random_normal([1024]), name = 'b_f'),
               'out':tf.Variable(tf.random_normal([n_classes]), name = 'b_out')}


save_path = 'pyhelp/'
model_name = 'sy2'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

sess= tf.Session()
saver = tf.train.Saver()
saver = tf.train.import_meta_graph('/Users/moonsooyoung/Desktop/pyhelp/sy2.meta')
saver.restore(sess,save_path_full)


x = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'] , name = 'c1_')
conv1 = maxpool2d(conv1)
    
conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'], name = 'c2_')
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1, 7*7*64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'], name = 'fc_')
fc = tf.nn.dropout(fc, keep_rate, name = 'fc')

output = tf.matmul(fc, weights['out'])+biases['out']


print(sess.run(output))
print('=========================================')
print(sess.run(tf.argmax(output,1)))


# < Convolutional Neural Network - Restore >
# =================================================
# Loading an image with cv2
# ----------------------
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import os
import cv2
import numpy as np


n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32, name = 'prob')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name='conv2d')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' , name = 'maxpool2d')


# load an image as gray scale and resize the image
img1 = cv2.imread("/Users/moonsooyoung/Downloads/5-2.png",0)    # 0 = gray

rsz_img1 = cv2.resize(img1, (28,28), interpolation=cv2.INTER_AREA)

cv2.imshow("resize1", rsz_img1)
cv2.waitKey(0)

image1 = cv2.adaptiveThreshold(rsz_img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('inv1',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

data1=image1.reshape(1,784)
data1 = data1/255    # 255 -> 1

x= tf.cast(data1, 'float')


weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32]), name = 'w_c1'),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64]), name = 'w_c2'),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024]), name = 'w_f'),
               'out':tf.Variable(tf.random_normal([1024, n_classes]), name ='w_out')}

biases = {'b_conv1':tf.Variable(tf.random_normal([32]), name = 'b_c1'),
               'b_conv2':tf.Variable(tf.random_normal([64]), name = 'b_c2'),
               'b_fc':tf.Variable(tf.random_normal([1024]), name = 'b_f'),
               'out':tf.Variable(tf.random_normal([n_classes]), name = 'b_out')}


save_path = 'pyhelp/'
model_name = 'sy2'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

sess= tf.Session()
saver = tf.train.Saver()
saver = tf.train.import_meta_graph('/Users/moonsooyoung/Desktop/pyhelp/sy2.meta')
saver.restore(sess,save_path_full)


x = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'] , name = 'c1_')
conv1 = maxpool2d(conv1)
    
conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'], name = 'c2_')
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1, 7*7*64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'], name = 'fc_')
fc = tf.nn.dropout(fc, keep_rate, name = 'fc')

output = tf.matmul(fc, weights['out'])+biases['out']


print(sess.run(output))
print('=========================================')
print(sess.run(tf.argmax(output,1)))


# < How to Save and Restore the model >
# =================================================
# Deep Neural Network - Restore
# ----------------------
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import matplotlib.pylab as plt

import sys, os
sys.path.append(os.pardir)

from mnist import load_mnist


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

a=x_train[5500]
aa = a.reshape(28,28)
plt.imshow(aa)
plt.show()

x = x_train[5500]
x = x.reshape(-1,784)
x = tf.cast(x, 'float')


hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1]), name = 'w1'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]), name = 'b1')}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name = 'w2'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]), name = 'b2')}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name = 'w3'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]), name = 'b3')}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]), name = 'w4'),
                    'biases':tf.Variable(tf.random_normal([n_classes]), name = 'b4')}


save_path = 'pyhelp/'
model_name = 'sy'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_full = os.path.join(save_path, model_name)

sess= tf.Session()
saver = tf.train.Saver()
saver = tf.train.import_meta_graph('/Users/moonsooyoung/Desktop/pyhelp/pyhelp/sy.meta')
saver.restore(sess,save_path_full)


l1 = tf.add(tf.matmul(x,hidden_1_layer['weights']), hidden_1_layer['biases'],name = 'l1_')
l1 = tf.nn.relu(l1, name = 'l1')

l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'],name = 'l2_')
l2 = tf.nn.relu(l2, name = 'l2')

l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'],name = 'l3_')
l3 = tf.nn.relu(l3, name = 'l3')

output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']


print('*****************************************')
print(sess.run(output))
print('=========================================')
print(sess.run(tf.argmax(output,1)))


# < How to Save and Restore the model >
# =================================================
# Deep Neural Network - Save
# ----------------------
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

import sys, os
sys.path.append(os.pardir)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')


hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1]), name = 'w1'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]), name = 'b1')}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name = 'w2'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]), name = 'b2')}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name = 'w3'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]), name = 'b3')}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]), name = 'w4'),
                    'biases':tf.Variable(tf.random_normal([n_classes]), name = 'b4')}

sess=tf.Session()
saver = tf.train.Saver()


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'],name = 'l1_')
    l1 = tf.nn.relu(l1, name = 'l1')

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'],name = 'l2_')
    l2 = tf.nn.relu(l2, name = 'l2')

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'],name = 'l3_')
    l3 = tf.nn.relu(l3, name = 'l3')

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output


save_path = 'pyhelp/'
model_name = 'sy'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)


def train_neural_network(x):
    prediction = neural_network_model(x)
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

        saver.save(sess,save_path_full)


train_neural_network(x)


# Recurrent Neural Network 
# =================================================
# 

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell


mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size], name = 'x_')
y = tf.placeholder('float', name = 'y_')


layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes]), name = 'w1'),
             'biases':tf.Variable(tf.random_normal([n_classes]), name = 'b1')}
lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
sess=tf.Session()
saver = tf.train.Saver()


def recurrent_neural_network(x):
    x = tf.transpose(x, [1,0,2], name = 'trp')
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0, name = 'x')
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32 )
    output = tf.matmul(outputs[-1],layer['weights'], name = 'output') + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='cost' )
    optimizer = tf.train.AdamOptimizer(name='beta_power').minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))       
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))


train_neural_network(x)


# Convolutional Neural Network 
# =================================================
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import os
import cv2


n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32, name = 'prob')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name='conv2d')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' , name = 'maxpool2d')


weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32]), name = 'w_c1'),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64]), name = 'w_c2'),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024]), name = 'w_f'),
               'out':tf.Variable(tf.random_normal([1024, n_classes]), name ='w_out')}

biases = {'b_conv1':tf.Variable(tf.random_normal([32]), name = 'b_c1'),
               'b_conv2':tf.Variable(tf.random_normal([64]), name = 'b_c2'),
               'b_fc':tf.Variable(tf.random_normal([1024]), name = 'b_f'),
               'out':tf.Variable(tf.random_normal([n_classes]), name = 'b_out')}

sess=tf.Session()
saver = tf.train.Saver()


def convolutional_neural_network(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'] , name = 'c1_')
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'], name = 'c2_')
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'], name = 'fc_')
    fc = tf.nn.dropout(fc, keep_rate, name = 'fc')

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


save_path = 'pyhelp/'
model_name = 'sy2'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 30
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
    
        saver.save(sess,save_path_full)


train_neural_network(x)


