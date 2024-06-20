# Create a new Markdown cell and answer: Above we mentioned that convolutional
# filters are applied to local image regions, with weights shared across regions.
# How does this compare to fully-connected neural networks?
# 
# Answer: TBD

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

#Prepare the data.
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from numpy import linalg as LA

dtype = torch.FloatTensor

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

images = images - images.mean()
images = images/images.std() 

train_seqs = images[0:40000]
val_seqs = images[40000:50000]

train_labels = labels[0:40000]
cv_labels = labels[40000:50000]


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS = 26, 26, 5, 5000
learning_rate = 0.01

class TooSimpleConvNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 convolution that takes in an image with one channel
        # and outputs an image with 8 channels.
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3)
        # 3x3 convolution that takes in an image with 8 channels
        # and outputs an image with 16 channels. The output image
        # has approximately half the height and half the width
        # because of the stride of 2.
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2)
        # 1x1 convolution that takes in an image with 16 channels and
        # produces an image with 5 channels. Here, the 5 channels
        # will correspond to class scores.
        self.final_conv = torch.nn.Conv2d(16, 5, kernel_size=1)
    
    def forward(self, x):
        # Convolutions work with images of shape
        # [batch_size, num_channels, height, width]
        x = x.view(-1, HEIGHT, WIDTH).unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        n, c, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=[h, w])
        x = self.final_conv(x).view(-1, NUM_CLASSES)
        return x
        


model = TooSimpleConvNN()


optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


def train(batch_size):
    model.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]


def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]


import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = train_labels[i].astype(np.int)
    
    output = model(x)
    y_hat = np.argmax(output.data.numpy(), axis =1)
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(val_seqs[i].astype(np.float32)))
    y = cv_labels[i].astype(np.int) 
    
    output = model(x)
    y_hat = np.argmax(output.data.numpy(), axis =1)
    acc = accuracy(y,y_hat)

    return acc


for m in model.children():
    m.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
train_accs, val_accs = [], []
batch_size = 10
for i in range(2000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


for m in model.children():
    m.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.05)
train_accs, val_accs = [], []
batch_size = 300
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

#Prepare the data.
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from numpy import linalg as LA

dtype = torch.FloatTensor

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

images = images - images.mean()
images = images/images.std() 

train_seqs = images[0:40000]
val_seqs = images[40000:50000]

train_labels = labels[0:40000]
cv_labels = labels[40000:50000]


# A nn model with 2 hidden layers
HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.001

class TwoLayerNN(torch.nn.Module):
    def __init__(self, D_in, D_out, layers):
        super(TwoLayerNN, self).__init__()
        #self.Linear = torch.nn.Linear(D_in, D_out)
        self.hidden_layer_count = layers
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        for i in range(self.hidden_layer_count):
            h_middle = self.middleLinear(h_relu)
            h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle_relu)
        return y_pred
        


model = TwoLayerNN(HEIGHT * WIDTH, NUM_CLASSES, 2)


optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


def train(batch_size):
    model.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]


def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]


import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = train_seqs[i].astype(np.float32)
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model.parameters())
        
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = val_seqs[i].astype(np.float32)
    y = cv_labels[i].astype(np.int)
    
    
    lst = list(model.parameters())
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 300
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


# # 4 Layer Feed Forward Neural Network
# 


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 100
learning_rate = 0.001
    
model2 = TwoLayerNN(HEIGHT * WIDTH, NUM_CLASSES, 3)
optimizer = torch.optim.Adam(model2.parameters(), lr= learning_rate)
def train(batch_size):
    model2.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model2(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = train_seqs[i].astype(np.float32)
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model2.parameters())
        
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = val_seqs[i].astype(np.float32)
    y = cv_labels[i].astype(np.int)
    
    
    lst = list(model2.parameters())
    for i in range(1000):
        
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 100
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


# # Choosing 3 layer (3 hidden layer) Neural Network to implement dropout.
# 


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.0001

class FeedForwardNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(FeedForwardNN, self).__init__()
        #self.Linear = torch.nn.Linear(D_in, D_out)
        #self.hidden_layer_count = layers
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.drop = torch.nn.Dropout(p=0.5, inplace=False)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        #x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        h_middle = self.middleLinear(self.drop(h_relu))
        h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle)
        return y_pred
        
    
model3 = FeedForwardNN(HEIGHT * WIDTH, NUM_CLASSES)
optimizer = torch.optim.Adam(model3.parameters(), lr= learning_rate)
def train(batch_size):
    model3.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model3(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

import random
def approx_train_accuracy():
    model3.eval()
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model3.parameters())
        
    for i in range(1000):
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    model3.eval()
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(val_seqs[i].astype(np.float32)))
    y = cv_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    
    lst = list(model3.parameters())
    for i in range(1000):
        
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

train_accs, val_accs = [], []
batch_size = 100
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


# # 3 Layer Neural Network with multi class hinge loss
# 

HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.0001

class FeedForwardNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(FeedForwardNN, self).__init__()
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        h_drop = F.dropout(h_relu, training = self.training)
        h_middle = self.middleLinear(h_drop)
        h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle_relu)
        return y_pred
        
    
model3 = FeedForwardNN(HEIGHT * WIDTH, NUM_CLASSES)
optimizer = torch.optim.Adam(model3.parameters(), lr= learning_rate)
def train(batch_size):
    model3.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model3(x)
    loss = F.multi_margin_loss(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]

import random
def approx_train_accuracy():
    model3.eval()
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model3.parameters())
        
    for i in range(1000):
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    model3.eval()
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(val_seqs[i].astype(np.float32)))
    y = cv_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    
    lst = list(model3.parameters())
    for i in range(1000):
        
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 200
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        #model3.eval()
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


# What was the best validation accuracy
# achieved? What configuration led to this performance? What was the corresponding
# batch size and learning rate? How many optimization steps did you need to take to
# reach that accuracy? How long did training take?
# 
# Best Validation accuracy is achieved using Cross entropy loss and drop out. 
# The highest validation accuracy achieved is 90% with significantly reduced overfitting trend.
# 
# Batch size = 250
# Learning rate = 0.0001
# Optimization steps = 5000
# time taken < 1 minute

import torch
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from torch.autograd import Variable


images = np.load("./images.npy")
labels = np.load("./labels.npy")

input_labels = (labels > 0).astype(dtype=np.float)
images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

#normalize the input data.
images = images/255
images = images.astype(dtype=np.float)


train_images = torch.Tensor(images[0:40000, : ])
cv_images = torch.Tensor(images[40000:45000, : ])
test_images = torch.Tensor(images[45000:50000, : ])

#split the labels in to train, validation and test data
train_labels = torch.FloatTensor(input_labels[0:40000])
cv_labels = torch.FloatTensor(input_labels[40000:45000])
test_labels = torch .FloatTensor(input_labels[45000:50000])


def accuracy(y,y_hat):
    count = 0;
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]

def accuracy_random_train():

    import random
    import numpy as np


    x =random.sample(range(0,40000),1000)
    y = np.empty(1000)
    y_hat = np.empty(1000)


    index=0
    
    for i in x:

        images_random = Variable(train_images[i].view(1,676), requires_grad=False)
        y[index] = train_labels[i]

        y_pred = torch.sigmoid(torch.mm(images_random, W))

        res = y_pred.data[0][0]
    

        if res > 0.5:
            act_label = 1
        else:
            act_label = 0

        y_hat[index] = act_label
        index += 1

    acc = accuracy(y,y_hat)
   
  
    return acc


    

def accuracy_random_validation():

    import random
    import numpy as np


    x =random.sample(range(0,5000),1000)
    y = np.empty(1000)
    y_hat = np.empty(1000)


    index=0
    
    for i in x:

        images_random = Variable(cv_images[i].view(1,676), requires_grad=False)
        y[index] = cv_labels[i]

       
        y_pred = torch.sigmoid(torch.mm(images_random, W))

        res = y_pred.data[0][0]
    

        if res > 0.5:
            act_label = 1
        else:
            act_label = 0

        y_hat[index] = act_label
        index += 1

    acc = accuracy(y,y_hat)
   
  
    return acc



dtype = torch.FloatTensor
input_size = 40000
D_in = train_images.shape[1]
online_learning_rate = 0.001
online_training_iterations = 1



size = (online_training_iterations * input_size) / 100
index = 0

# N dimensional arrays to store training and validation accuracies for every 100 steps
accuracy_train = []
accuracy_valid = []

W = Variable(torch.randn(D_in, 1).type(dtype), requires_grad = True)

while (online_training_iterations >= 0):
    for t in range(input_size):
        
        x = Variable(train_images[t].view(1,676), requires_grad=False)
        y = train_labels[t]
        
        y_pred = torch.sigmoid(torch.mm(x, W))
        
        #loss = -(y * torch.log(torch.sigmoid(torch.mm(x, W))) + 
         #        (1-y) * torch.log(torch.sigmoid(-torch.mm(x, W))))
        
        loss = -(y * torch.log(y_pred) +
                 (1-y) * torch.log(1 - y_pred))
        
        if t%100 == 0:
            accuracy_train.append(accuracy_random_train())
            accuracy_valid.append(accuracy_random_validation())
            #index += 1
    
        loss.backward()
        W.data -= online_learning_rate * W.grad.data
        W.grad.data.zero_()
        W.data = W.data/torch.norm(W.data)
    online_training_iterations -= 1


import matplotlib.pyplot as plt


t = np.arange(0,400,1)

s = accuracy_train[0:400]
k = accuracy_valid[0:400]
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)

plt.show()


# Is this over fitting:
# 
#     From the plot above there is no over fitting here.The accuracies are similar both cases although there are minor fluctuations in the trend
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

#Prepare the data.
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from numpy import linalg as LA

dtype = torch.FloatTensor

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

images = images - images.mean()
images = images/images.std() 

train_seqs = images[0:40000]
val_seqs = images[40000:50000]

train_labels = labels[0:40000]
cv_labels = labels[40000:50000]


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 100
learning_rate = 0.001

class TwoLayerNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(TwoLayerNN, self).__init__()
        #self.Linear = torch.nn.Linear(D_in, D_out)
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        y_pred = self.Linear2(h_relu)
        return y_pred
        


model = TwoLayerNN(HEIGHT * WIDTH, NUM_CLASSES)


optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


def train(batch_size):
    model.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]


def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]


import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = train_seqs[i].astype(np.float32)
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model.parameters())
    w1 = lst[0].data.numpy()
    b1 = lst[1].data.numpy()
    w2 = lst[2].data.numpy()
    b2 = lst[3].data.numpy()
    
    for i in range(1000):
        h = x[i].dot(w1.transpose()) + b1
        h_relu = np.maximum(0.0, h)
        y_pred = h_relu.dot(w2.transpose()) + b2
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = val_seqs[i].astype(np.float32)
    y = cv_labels[i].astype(np.int)
    
    
    lst = list(model.parameters())
    w1 = lst[0].data.numpy()
    b1 = lst[1].data.numpy()
    w2 = lst[2].data.numpy()
    b2 = lst[3].data.numpy()
    
    for i in range(1000):
        h = x[i].dot(w1.transpose()) + b1
        h_relu = np.maximum(0.0, h)
        y_pred = h_relu.dot(w2.transpose()) + b2
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 1
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


for m in model.children():
    m.reset_parameters()


train_accs, val_accs = [], []
batch_size = 10
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))


import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


for m in model.children():
    m.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

train_accs, val_accs = [], []
batch_size = 500
for i in range(10000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))



import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


# # Best Validation Accuracy:
# 
# The best validation accuracy reached = 88% at,
# 
# learning rate = 0.0001
# 
# Optimizer = Adam
# 
# Batch size = 500
# 
# Number of optimization steps = 10000
# 
# Time taken to complete : 1 minute
# 
# The model clearly overfits here. I'm able to reach 87% validation accuracy (at lr = 0.001, batch size = 20, Optimization steps = 5000), anything beyond this, the model clearly overfits with high training accuracy.
# 

