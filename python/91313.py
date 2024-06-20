# # Serialize Trained Hybrid Keras/TF Model for Tensorflow Serving
# 
# Adapted from code in [mnist\_saved\_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py) in tensorflow\_serving/examples.
# 

from __future__ import division, print_function
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy
import os
import shutil
import tensorflow as tf


DATA_DIR = "../../data"
TEST_FILE = os.path.join(DATA_DIR, "mnist_test.csv")

IMG_SIZE = 28
BATCH_SIZE = 128
NUM_CLASSES = 10

MODEL_DIR = os.path.join(DATA_DIR, "01-tf-serving")
TF_MODEL_NAME = "model-5"

EXPORT_DIR = os.path.join(DATA_DIR, "tf-export")
MODEL_NAME = "ktf-mnist-cnn"
MODEL_VERSION = 1


# ## Restore trained model
# 
# This [blog post](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) contains lots of good information on how to save and restore Tensorflow models.
# 

tf.contrib.keras.backend.set_learning_phase(0)
sess = tf.contrib.keras.backend.get_session()
with sess.as_default():
    saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, TF_MODEL_NAME + ".meta"))
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))


# ## Export Model to form suitable by TF-Serving
# 
# TF-Serving needs its models to be serialized to the [SavedModel format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md). The following code is largely adapted from the [mnist_saved_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py). More information on this [Tensorflow Serving documentation page](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md).
# 
# The resulting exported model directory structure is as follows:
# 
#     .
#     └── ktf-mnist-cnn
#         └── 1
#             ├── saved_model.pb
#             └── variables
#                 ├── variables.data-00000-of-00001
#                 └── variables.index
# 

shutil.rmtree(EXPORT_DIR, True)


serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {"x": tf.FixedLenFeature(shape=[IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
X = tf.identity(tf_example["x"], name="X")
Y = tf.placeholder("int32", shape=[None, 10], name="Y")
Y_ = tf.placeholder("int32", shape=[None, 10], name="Y_")


export_dir = os.path.join(EXPORT_DIR, MODEL_NAME)
export_path = os.path.join(export_dir, str(MODEL_VERSION))
print("Exporting model to {:s}".format(export_path))


builder = tf.saved_model.builder.SavedModelBuilder(export_path)


tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
tensor_info_y = tf.saved_model.utils.build_tensor_info(Y)
tensor_info_y_ = tf.saved_model.utils.build_tensor_info(Y_)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"images": tensor_info_x,
                "labels": tensor_info_y},
        outputs={"scores": tensor_info_y_},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
print(prediction_signature)


legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map = {
        "predict": prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature
    },
    legacy_init_op=legacy_init_op)
builder.save()





# # Model AND gate with single neuron
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


Xvalue = np.array([[0, 0], 
                   [0, 1],
                   [1, 0],
                   [1, 1]])
yvalue = np.array([[0], 
                   [0], 
                   [0], 
                   [1]])
# yvalue = np.array([[0], 
#                    [1], 
#                    [1], 
#                    [1]])


X = tf.placeholder(tf.float32, [4, 2], name="X")
y = tf.placeholder(tf.float32, [4, 1], name="y")


W = tf.Variable(tf.random_normal([2, 1]), name="W")
b = tf.Variable(tf.random_normal([1, 1]), name="b")


y_ = tf.nn.sigmoid(tf.matmul(X, W) + b, name="y_")
loss = tf.reduce_mean(0.5 * tf.pow(y_ - y, 2), name="loss")


optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        _, yval_, lossval = sess.run([optimizer, y_, loss], 
                                     feed_dict={X: Xvalue, y: yvalue})
        losses.append(lossval)
    print("predictions")
    print(yval_)
    print("weights")
    Wval, bval = sess.run([W, b])
    print(Wval)
    print(bval)


plt.plot(np.arange(len(losses)), losses)
plt.xlabel("epochs")
plt.ylabel("loss")





# # Serialize native Keras model for Tensorflow Serving
# 
# Code adapted from discussion about this in [Tensorflow Serving Issue 310](https://github.com/tensorflow/serving/issues/310), specifically the recipe suggested by @tspthomas.
# 

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

import keras.backend as K
from keras.models import load_model
import os
import shutil

K.set_learning_phase(0)


# ## Load model
# 
# 
# The model we will use is the best model produced by [this Keras model](https://github.com/sujitpal/polydlot/blob/master/src/keras/01-mnist-fcn.ipynb).
# 

DATA_DIR = "../../data"
EXPORT_DIR = os.path.join(DATA_DIR, "tf-export")

MODEL_NAME = "keras-mnist-fcn"
MODEL_VERSION = 1

MODEL_BIN = os.path.join(DATA_DIR, "{:s}-best.h5".format(MODEL_NAME))
EXPORT_PATH = os.path.join(EXPORT_DIR, MODEL_NAME)


model = load_model(MODEL_BIN)


# ## Export model
# 
# Resulting exported model should be as follows, under the export directory given by `EXPORT_DIR`.
# 
#     .
#     └── keras-mnist-fcn
#         └── 1
#             ├── saved_model.pb
#             └── variables
#                 ├── variables.data-00000-of-00001
#                 └── variables.index
# 

shutil.rmtree(EXPORT_PATH, True)


full_export_path = os.path.join(EXPORT_PATH, str(MODEL_VERSION))
builder = saved_model_builder.SavedModelBuilder(full_export_path)
signature = predict_signature_def(inputs={"images": model.input},
                                  outputs={"scores": model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})
    builder.save()





# # Moving Square Video Prediction
# 
# This is the third toy example from Jason Brownlee's [Long Short Term Memory Networks with Python](https://machinelearningmastery.com/lstms-with-python/). It illustrates using a CNN LSTM, ie, an LSTM with input from CNN. Per section 8.2 of the book:
# 
# > The moving square video prediction problem is contrived to demonstrate the CNN LSTM. The
# problem involves the generation of a sequence of frames. In each image a line is drawn from left to right or right to left. Each frame shows the extension of the line by one pixel. The task is for the model to classify whether the line moved left or right in the sequence of frames. Technically, the problem is a sequence classification problem framed with a many-to-one prediction model.
# 

from __future__ import division, print_function
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
get_ipython().magic('matplotlib inline')


DATA_DIR = "../../data"
MODEL_FILE = os.path.join(DATA_DIR, "torch-08-moving-square-{:d}.model")

TRAINING_SIZE = 5000
VALIDATION_SIZE = 100
TEST_SIZE = 500

SEQUENCE_LENGTH = 50
FRAME_SIZE = 50

BATCH_SIZE = 32

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3


# ## Prepare Data
# 
# Our data is going to be batches of sequences of images. Each image will need to be in channel-first format, since Pytorch only supports that format. So our output data will be in the (batch_size, sequence_length, num_channels, height, width) format.
# 

def next_frame(frame, x, y, move_right, upd_int):
    frame_size = frame.shape[0]
    if x is None and y is None:
        x = 0 if (move_right == 1) else (frame_size - 1)
        y = np.random.randint(0, frame_size, 1)[0]
    else:
        if y == 0:
            y = np.random.randint(y, y + 1, 1)[0]
        elif y == frame_size - 1:
            y = np.random.randint(y - 1, y, 1)[0]
        else:
            y = np.random.randint(y - 1, y + 1, 1)[0]
        if move_right:
            x = x + 1
        else:
            x = x - 1
    new_frame = frame.copy()
    new_frame[y, x] = upd_int
    return new_frame, x, y

row, col = None, None
frame = np.ones((5, 5))
move_right = 1 if np.random.random() < 0.5 else 0
for i in range(5):
    frame, col, row = next_frame(frame, col, row, move_right, 0)
    plt.subplot(1, 5, (i+1))
    plt.xticks([])
    plt.yticks([])
    plt.title((col, row, "R" if (move_right==1) else "L"))
    plt.imshow(frame, cmap="gray")
plt.tight_layout()
plt.show()


def generate_data(frame_size, sequence_length, num_samples):
    assert(frame_size == sequence_length)
    xs, ys = [], []
    for bid in range(num_samples):
        frame_seq = []
        row, col = None, None
        frame = np.ones((frame_size, frame_size))
        move_right = 1 if np.random.random() < 0.5 else 0
        for sid in range(sequence_length):
            frm, col, row = next_frame(frame, col, row, move_right, 0)
            frm = frm.reshape((1, frame_size, frame_size))
            frame_seq.append(frm)
        xs.append(np.array(frame_seq))
        ys.append(move_right)
    return np.array(xs), np.array(ys)

X, y = generate_data(FRAME_SIZE, SEQUENCE_LENGTH, 10)
print(X.shape, y.shape)


Xtrain, ytrain = generate_data(FRAME_SIZE, SEQUENCE_LENGTH, TRAINING_SIZE)
Xval, yval = generate_data(FRAME_SIZE, SEQUENCE_LENGTH, VALIDATION_SIZE)
Xtest, ytest = generate_data(FRAME_SIZE, SEQUENCE_LENGTH, TEST_SIZE)
print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)


# ## Define Network
# 
# We want to build a CNN-LSTM network. Each image in the sequence will be fed to a CNN which will learn to produce a feature vector for the image. The sequence of vectors will be fed into an LSTM and the LSTM will learn to generate a context vector that will be then fed into a FCN that will predict if the square is moving left or right.
# 
# <img src="08-network-design.png"/>
# 

class CNN(nn.Module):
    
    def __init__(self, input_height, input_width, input_channels, 
                 output_channels, 
                 conv_kernel_size, conv_stride, conv_padding,
                 pool_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 
                               kernel_size=conv_kernel_size, 
                               stride=conv_stride, 
                               padding=conv_padding)
        self.relu1 = nn.ReLU()
        self.output_height = input_height // pool_size
        self.output_width = input_width // pool_size
        self.output_channels = output_channels
        self.pool_size = pool_size
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, self.pool_size)
        x = x.view(x.size(0), self.output_channels * self.output_height * self.output_width)
        return x

cnn = CNN(FRAME_SIZE, FRAME_SIZE, 1, 2, 2, 1, 1, 2)
print(cnn)

# size debugging
print("--- size debugging ---")
inp = Variable(torch.randn(BATCH_SIZE, 1, FRAME_SIZE, FRAME_SIZE))
out = cnn(inp)
print(out.size())


class CNNLSTM(nn.Module):
    
    def __init__(self, image_size, input_channels, output_channels, 
                 conv_kernel_size, conv_stride, conv_padding, pool_size, 
                 seq_length, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        # capture variables
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.image_size = image_size
        self.output_channels = output_channels
        self.hidden_size = hidden_size
        self.lstm_input_size = output_channels * (image_size // pool_size) ** 2
        # define network layers
        self.cnn = CNN(image_size, image_size, input_channels, output_channels, 
                       conv_kernel_size, conv_stride, conv_padding, pool_size)
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = (Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size).cuda()),
                  Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size).cuda()))
        else:
            h0 = (Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)),
                  Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)))
        
        cnn_out = []
        for i in range(self.seq_length):
            cnn_out.append(self.cnn(x[:, i, :, :, :]))
        x = torch.cat(cnn_out, dim=1).view(-1, self.seq_length, self.lstm_input_size)
        x, h0 = self.lstm(x, h0)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x        

model = CNNLSTM(FRAME_SIZE, 1, 2, 2, 1, 1, 2, SEQUENCE_LENGTH, 50, 1, 2)
if torch.cuda.is_available():
    model.cuda()
print(model)

# size debugging
print("--- size debugging ---")
inp = Variable(torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 1, FRAME_SIZE, FRAME_SIZE))
out = model(inp)
print(out.size())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ## Train Network
# 
# Training on GPU is probably preferable for this example, takes a long time on CPU. During some runs, the training and validation accuracies get stuck, possibly because of bad initializations, the fix appears to be to just retry the training until it results in good training and validation accuracies and use the resulting model.
# 

def compute_accuracy(pred_var, true_var):
    if torch.cuda.is_available():
        ypred = pred_var.cpu().data.numpy()
        ytrue = true_var.cpu().data.numpy()
    else:
        ypred = pred_var.data.numpy()
        ytrue = true_var.data.numpy()
    return accuracy_score(ypred, ytrue)
    
history = []
for epoch in range(NUM_EPOCHS):
    num_batches = Xtrain.shape[0] // BATCH_SIZE
    
    shuffled_indices = np.random.permutation(np.arange(Xtrain.shape[0]))
    train_loss, train_acc = 0., 0.
    for bid in range(num_batches):
        Xbatch_data = Xtrain[shuffled_indices[bid * BATCH_SIZE : (bid+1) * BATCH_SIZE]]
        ybatch_data = ytrain[shuffled_indices[bid * BATCH_SIZE : (bid+1) * BATCH_SIZE]]
        Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
        ybatch = Variable(torch.from_numpy(ybatch_data).long())
        if torch.cuda.is_available():
            Xbatch = Xbatch.cuda()
            ybatch = ybatch.cuda()
            
        # initialize gradients
        optimizer.zero_grad()
        
        # forward
        Ybatch_ = model(Xbatch)
        loss = loss_fn(Ybatch_, ybatch)
        
        # backward
        loss.backward()

        train_loss += loss.data[0]
        
        _, ybatch_ = Ybatch_.max(1)
        train_acc += compute_accuracy(ybatch_, ybatch)
        
        optimizer.step()
        
    # compute training loss and accuracy
    train_loss /= num_batches
    train_acc /= num_batches
    
    # compute validation loss and accuracy
    val_loss, val_acc = 0., 0.
    num_val_batches = Xval.shape[0] // BATCH_SIZE
    for bid in range(num_val_batches):
        # data
        Xbatch_data = Xval[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
        ybatch_data = yval[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
        Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
        ybatch = Variable(torch.from_numpy(ybatch_data).long())
        if torch.cuda.is_available():
            Xbatch = Xbatch.cuda()
            ybatch = ybatch.cuda()

        Ybatch_ = model(Xbatch)
        loss = loss_fn(Ybatch_, ybatch)
        val_loss += loss.data[0]

        _, ybatch_ = Ybatch_.max(1)
        val_acc += compute_accuracy(ybatch_, ybatch)
        
    val_loss /= num_val_batches
    val_acc /= num_val_batches
    
    torch.save(model.state_dict(), MODEL_FILE.format(epoch+1))
    print("Epoch {:2d}/{:d}: loss={:.3f}, acc={:.3f}, val_loss={:.3f}, val_acc={:.3f}"
          .format((epoch+1), NUM_EPOCHS, train_loss, train_acc, val_loss, val_acc))
    
    history.append((train_loss, val_loss, train_acc, val_acc))


losses = [x[0] for x in history]
val_losses = [x[1] for x in history]
accs = [x[2] for x in history]
val_accs = [x[3] for x in history]

plt.subplot(211)
plt.title("Accuracy")
plt.plot(accs, color="r", label="train")
plt.plot(val_accs, color="b", label="valid")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(losses, color="r", label="train")
plt.plot(val_losses, color="b", label="valid")
plt.legend(loc="best")

plt.tight_layout()
plt.show()


# ## Test/Evaluate Network
# 

saved_model = CNNLSTM(FRAME_SIZE, 1, 2, 2, 1, 1, 2, SEQUENCE_LENGTH, 50, 1, 2)
saved_model.load_state_dict(torch.load(MODEL_FILE.format(5)))
if torch.cuda.is_available():
    saved_model.cuda()


ylabels, ypreds = [], []
num_test_batches = Xtest.shape[0] // BATCH_SIZE
for bid in range(num_test_batches):
    Xbatch_data = Xtest[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
    ybatch_data = ytest[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
    Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
    ybatch = Variable(torch.from_numpy(ybatch_data).long())
    if torch.cuda.is_available():
        Xbatch = Xbatch.cuda()
        ybatch = ybatch.cuda()

    Ybatch_ = saved_model(Xbatch)
    _, ybatch_ = Ybatch_.max(1)
    if torch.cuda.is_available():
        ylabels.extend(ybatch.cpu().data.numpy())
        ypreds.extend(ybatch_.cpu().data.numpy())
    else:
        ylabels.extend(ybatch.data.numpy())
        ypreds.extend(ybatch_.data.numpy())

print("Test accuracy: {:.3f}".format(accuracy_score(ylabels, ypreds)))
print("Confusion matrix")
print(confusion_matrix(ylabels, ypreds))


for i in range(NUM_EPOCHS):
    os.remove(MODEL_FILE.format(i + 1))





# # Consume native Keras model served by TF-Serving
# 
# This notebook shows client code needed to consume a native Keras model served by Tensorflow serving. The Tensorflow serving model needs to be started using the following command:
# 
#     bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server         --port=9000 --model_name=keras-mnist-fcn         --model_base_path=/home/sujit/Projects/polydlot/data/tf-export/keras-mnist-fcn
# 

from __future__ import division, print_function
from google.protobuf import json_format
from grpc.beta import implementations
from sklearn.preprocessing import OneHotEncoder
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os
import sys
import threading
import time
import numpy as np
import tensorflow as tf


SERVER_HOST = "localhost"
SERVER_PORT = 9000

DATA_DIR = "../../data"
TEST_FILE = os.path.join(DATA_DIR, "mnist_test.csv")

IMG_SIZE = 28

MODEL_NAME = "keras-mnist-fcn"


# ## Load Test Data
# 

def parse_file(filename):
    xdata, ydata = [], []
    fin = open(filename, "rb")
    i = 0
    for line in fin:
        if i % 10000 == 0:
            print("{:s}: {:d} lines read".format(os.path.basename(filename), i))
        cols = line.strip().split(",")
        ydata.append(int(cols[0]))
        xdata.append(np.reshape(np.array([float(x) / 255. for x in cols[1:]]), 
                     (IMG_SIZE * IMG_SIZE, )))
        i += 1
    fin.close()
    print("{:s}: {:d} lines read".format(os.path.basename(filename), i))
    X = np.array(xdata, dtype="float32")
    y = np.array(ydata, dtype="int32")
    return X, y

Xtest, ytest = parse_file(TEST_FILE)
print(Xtest.shape, ytest.shape)


# ## Make Predictions
# 

channel = implementations.insecure_channel(SERVER_HOST, SERVER_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
labels, predictions = [], []
for i in range(Xtest.shape[0]):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = "predict"

    Xbatch, ybatch = Xtest[i], ytest[i]
    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(Xbatch, shape=[1, Xbatch.size]))

    result = stub.Predict(request, 10.0)
    result_json = json.loads(json_format.MessageToJson(result))
    y_ = np.array(result_json["outputs"]["scores"]["floatVal"], dtype="float32")
    labels.append(ybatch)
    predictions.append(np.argmax(y_))


print("Test accuracy: {:.3f}".format(accuracy_score(labels, predictions)))
print("Confusion Matrix")
print(confusion_matrix(labels, predictions))





# # Addition Prediction
# 
# This is the fourth toy example from Jason Brownlee's [Long Short Term Memory Networks with Python](https://machinelearningmastery.com/lstms-with-python/). It demonstrates the solution to a sequence-to-sequence (aka seq2seq) prediction problem. Per section 9.3 of the book:
# 
# > The problem is defined as calculating the sum output of two input numbers. This is
# challenging as each digit and mathematical symbol is provided as a character and the expected
# output is also expected as characters. For example, the input 10+6 with the output 16 would
# be represented by the sequences ['1', '0', '+', '6'] and ['1', '6'] respectively.
# 
# > The model must learn not only the integer nature of the characters, but also the nature
# of the mathematical operation to perform. Notice how sequence is now important, and that
# randomly shuffling the input will create a nonsense sequence that could not be related to the
# output sequence. Also notice how the number of digits could vary in both the input and output
# sequences. Technically this makes the addition prediction problem a sequence-to-sequence
# problem that requires a many-to-many model to address.
# 

from __future__ import division, print_function
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
get_ipython().magic('matplotlib inline')


DATA_DIR = "../../data"
MODEL_FILE = os.path.join(DATA_DIR, "torch-09-addition-predict-{:d}.model")

TRAIN_SIZE = 7500
VAL_SIZE = 100
TEST_SIZE = 500

ENC_SEQ_LENGTH = 8
DEC_SEQ_LENGTH = 2
EMBEDDING_SIZE = 12

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


# ## Prepare Data
# 
# We first generate a number of random math addition problems of the form a + b + c = d, then stringify them into the sequence of characters as needed by our network. We then left pad these sequences with space. Finally, we one-hot encode these padded strings so they can be fed into an LSTM. The blocks below show these transformations on a small set of random triples.
# 
# We then apply these sequence of transformations to create our training and test sets. Each row of input to the network is represented by a sequence of one-hot vectors corresponding to the characters in the alphabet, and each row of output is a sequence of ids corresponding to the characters in the result.
# 

def generate_random_addition_problems(max_num, num_probs):
    lhs_1 = np.random.randint(0, max_num + 1, num_probs)
    lhs_2 = np.random.randint(0, max_num + 1, num_probs)
    lhs_3 = np.random.randint(0, max_num + 1, num_probs)
    rhs = lhs_1 + lhs_2 + lhs_3
    in_seqs, out_seqs = [], []
    for i in range(num_probs):
        in_seqs.append([c for c in "".join([str(lhs_1[i]), "+", str(lhs_2[i]), 
                                            "+", str(lhs_3[i])])])
        out_seqs.append([c for c in str(rhs[i])])
    return in_seqs, out_seqs

input_seq, output_seq = generate_random_addition_problems(10, 5)
for i, o in zip(input_seq, output_seq):
    print(i, o)


def left_pad(chr_list, pad_len, pad_char=' '):
    len_to_pad = pad_len - len(chr_list)
    padded_list = []
    for i in range(len_to_pad):
        padded_list.append(pad_char)
    padded_list.extend(chr_list)
    return padded_list

for i, o in zip(input_seq, output_seq):
    print(left_pad(i, 8), left_pad(o, 2))


def one_hot_encode(padded_chr_list, char2idx):
    encodeds = []
    for c in padded_chr_list:
        v = np.zeros(len(char2idx))
        v[char2idx[c]] = 1
        encodeds.append(v)
    return np.array(encodeds)

def one_hot_decode(enc_matrix, idx2char):
    decodeds = []
    for i in range(enc_matrix.shape[0]):
        v = enc_matrix[i]
        j = np.where(v == 1)[0][0]
        decodeds.append(idx2char[j])
    return decodeds

chrs = [str(x) for x in range(10)] + ['+', ' ']
char2idx, idx2char = {}, {}
for i, c in enumerate(chrs):
    char2idx[c] = i
    idx2char[i] = c
for i, o in zip(input_seq, output_seq):
    X = one_hot_encode(left_pad(i, 8), char2idx)
    Y = np.array([char2idx[x] for x in left_pad(o, 2)])
    x_dec = one_hot_decode(X, idx2char)
    print(x_dec, X.shape, Y)


def generate_data(data_size, enc_seqlen, dec_seqlen):
    input_seq, output_seq = generate_random_addition_problems(10, data_size)
    Xgen = np.zeros((data_size, enc_seqlen, EMBEDDING_SIZE))
    Ygen = np.zeros((data_size, dec_seqlen))
    for idx, (inp, outp) in enumerate(zip(input_seq, output_seq)):
        Xgen[idx] = one_hot_encode(left_pad(inp, ENC_SEQ_LENGTH), char2idx)
        Ygen[idx] = np.array([char2idx[x] for x in left_pad(outp, DEC_SEQ_LENGTH)])
    return Xgen, Ygen

Xtrain, Ytrain = generate_data(TRAIN_SIZE, ENC_SEQ_LENGTH, DEC_SEQ_LENGTH)
Xval, Yval = generate_data(VAL_SIZE, ENC_SEQ_LENGTH, DEC_SEQ_LENGTH)
Xtest, Ytest = generate_data(TEST_SIZE, ENC_SEQ_LENGTH, DEC_SEQ_LENGTH)

print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape, Xtest.shape, Ytest.shape)


# ## Define Network
# 

class AdditionPredictor(nn.Module):
    
    def __init__(self, enc_seqlen, enc_embed_dim, enc_hidden_dim,
                 dec_seqlen, dec_hidden_dim, output_dim):
        super(AdditionPredictor, self).__init__()
        # capture variables needed in forward
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_seqlen = dec_seqlen
        self.output_dim = output_dim
        # define network layers
        self.enc_lstm = nn.LSTM(enc_embed_dim, enc_hidden_dim, 1, batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_dim, dec_hidden_dim, 1, batch_first=True)
        self.dec_fcn = nn.Linear(dec_hidden_dim, output_dim)
        self.dec_softmax = nn.Softmax()
    
    def forward(self, x):
        if torch.cuda.is_available():
            he = (Variable(torch.randn(1, x.size(0), self.enc_hidden_dim).cuda()),
                  Variable(torch.randn(1, x.size(0), self.enc_hidden_dim).cuda()))
            hd = (Variable(torch.randn(1, x.size(0), self.dec_hidden_dim).cuda()),
                  Variable(torch.randn(1, x.size(0), self.dec_hidden_dim).cuda()))
        else:
            he = (Variable(torch.randn(1, x.size(0), self.enc_hidden_dim)),
                  Variable(torch.randn(1, x.size(0), self.enc_hidden_dim)))
            hd = (Variable(torch.randn(1, x.size(0), self.dec_hidden_dim)),
                  Variable(torch.randn(1, x.size(0), self.dec_hidden_dim)))

        x, he = self.enc_lstm(x, he)         # encoder LSTM
        x = x[:, -1, :].unsqueeze(1)         # encoder context vector
        x = x.repeat(1, self.dec_seqlen, 1)  # repeat vector decoder seqlen times
        x, hd = self.dec_lstm(x, hd)         # decoder LSTM
        x_fcn = Variable(torch.zeros(x.size(0), self.dec_seqlen, self.output_dim))
        for i in range(self.dec_seqlen):     # decoder LSTM -> fcn for each timestep
            x_fcn[:, i, :] = self.dec_softmax(self.dec_fcn(x[:, i, :]))
        x = x_fcn
        return x

model = AdditionPredictor(ENC_SEQ_LENGTH, EMBEDDING_SIZE, 75,
                          DEC_SEQ_LENGTH, 50, len(chrs))
if torch.cuda.is_available():
    model.cuda()
print(model)

# size debugging
print("--- size debugging ---")
inp = Variable(torch.randn(BATCH_SIZE, ENC_SEQ_LENGTH, EMBEDDING_SIZE))
outp = model(inp)
print(outp.size())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ## Train Network
# 

def compute_accuracy(pred_var, true_var, idx2char):
    if torch.cuda.is_available():
        ypred = pred_var.cpu().data.numpy()
        ytrue = true_var.cpu().data.numpy()
    else:
        ypred = pred_var.data.numpy()
        ytrue = true_var.data.numpy()
    pred_nums, true_nums = [], []
    total_correct = 0
    for i in range(ypred.shape[0]):
        true_num = int("".join([idx2char[x] for x in ytrue[i].tolist()]).lstrip())
        true_nums.append(true_num)
        try:
            pred_num = int("".join([idx2char[x] for x in ypred[i].tolist()]).lstrip())
            pred_nums.append(pred_num)
        except ValueError:
            pred_nums.append(true_num + 1)
            continue
    return pred_nums, true_nums, accuracy_score(pred_nums, true_nums)

history = []
for epoch in range(NUM_EPOCHS):
    
    num_batches = Xtrain.shape[0] // BATCH_SIZE
    shuffled_indices = np.random.permutation(np.arange(Xtrain.shape[0]))
    train_loss, train_acc = 0., 0.
    
    for bid in range(num_batches):
        
        # extract one batch of data
        Xbatch_data = Xtrain[shuffled_indices[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]]
        Ybatch_data = Ytrain[shuffled_indices[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]]
        Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
        Ybatch = Variable(torch.from_numpy(Ybatch_data).long())
        if torch.cuda.is_available():
            Xbatch = Xbatch.cuda()
            Ybatch = Ybatch.cuda()
            
        # initialize gradients
        optimizer.zero_grad()

        # forward
        loss = 0.
        Ybatch_ = model(Xbatch)
        for i in range(Ybatch.size(1)):
            loss += loss_fn(Ybatch_[:, i, :], Ybatch[:, i])
        
        # backward
        loss.backward()

        train_loss += loss.data[0]
        
        _, ybatch_ = Ybatch_.max(2)
        _, _, acc = compute_accuracy(ybatch_, Ybatch, idx2char)
        train_acc += acc
        
        optimizer.step()
        
    # compute training loss and accuracy
    train_loss /= num_batches
    train_acc /= num_batches
    
    # compute validation loss and accuracy
    val_loss, val_acc = 0., 0.
    num_val_batches = Xval.shape[0] // BATCH_SIZE
    for bid in range(num_val_batches):
        # data
        Xbatch_data = Xval[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
        Ybatch_data = Yval[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
        Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
        Ybatch = Variable(torch.from_numpy(Ybatch_data).long())
        if torch.cuda.is_available():
            Xbatch = Xbatch.cuda()
            Ybatch = Ybatch.cuda()

        loss = 0.
        Ybatch_ = model(Xbatch)
        for i in range(Ybatch.size(1)):
            loss += loss_fn(Ybatch_[:, i, :], Ybatch[:, i])
        val_loss += loss.data[0]

        _, ybatch_ = Ybatch_.max(2)
        _, _, acc = compute_accuracy(ybatch_, Ybatch, idx2char)
        val_acc += acc
        
    val_loss /= num_val_batches
    val_acc /= num_val_batches
    
    torch.save(model.state_dict(), MODEL_FILE.format(epoch+1))
    print("Epoch {:2d}/{:d}: loss={:.3f}, acc={:.3f}, val_loss={:.3f}, val_acc={:.3f}"
          .format((epoch+1), NUM_EPOCHS, train_loss, train_acc, val_loss, val_acc))
    
    history.append((train_loss, val_loss, train_acc, val_acc))


losses = [x[0] for x in history]
val_losses = [x[1] for x in history]
accs = [x[2] for x in history]
val_accs = [x[3] for x in history]

plt.subplot(211)
plt.title("Accuracy")
plt.plot(accs, color="r", label="train")
plt.plot(val_accs, color="b", label="valid")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(losses, color="r", label="train")
plt.plot(val_losses, color="b", label="valid")
plt.legend(loc="best")

plt.tight_layout()
plt.show()


# ## Evaluate Network
# 
# Our first test takes the (presumably) best trained model and evaluates it against the test set. The second test takes a slice of the test set and displays the predicted and the true result for each. From the second, it appears that the model has learned to always return a single value.
# 

saved_model = AdditionPredictor(ENC_SEQ_LENGTH, EMBEDDING_SIZE, 75,
                                DEC_SEQ_LENGTH, 50, len(chrs))
saved_model.load_state_dict(torch.load(MODEL_FILE.format(NUM_EPOCHS)))
if torch.cuda.is_available():
    saved_model.cuda()


ylabels, ypreds = [], []
num_test_batches = Xtest.shape[0] // BATCH_SIZE
for bid in range(num_test_batches):
    Xbatch_data = Xtest[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
    Ybatch_data = Ytest[bid * BATCH_SIZE : (bid + 1) * BATCH_SIZE]
    Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
    Ybatch = Variable(torch.from_numpy(Ybatch_data).long())
    if torch.cuda.is_available():
        Xbatch = Xbatch.cuda()
        Ybatch = Ybatch.cuda()

    Ybatch_ = saved_model(Xbatch)
    _, ybatch_ = Ybatch_.max(2)

    pred_nums, true_nums, _ = compute_accuracy(ybatch_, Ybatch, idx2char)
    ylabels.extend(true_nums)
    ypreds.extend(pred_nums)

print("Test accuracy: {:.3f}".format(accuracy_score(ylabels, ypreds)))


Xbatch_data = Xtest[0:10]
Ybatch_data = Ytest[0:10]
Xbatch = Variable(torch.from_numpy(Xbatch_data).float())
Ybatch = Variable(torch.from_numpy(Ybatch_data).long())
if torch.cuda.is_available():
    Xbatch = Xbatch.cuda()
    Ybatch = Ybatch.cuda()

Ybatch_ = saved_model(Xbatch)
_, ybatch_ = Ybatch_.max(2)

pred_nums, true_nums, _ = compute_accuracy(ybatch_, Ybatch, idx2char)
Xbatch_var = Xbatch.data.numpy()

for i in range(Xbatch_var.shape[0]):
    problem = "".join(one_hot_decode(Xbatch_var[i], idx2char)).lstrip()
    print("{:>8s} = {:2d} (expected {:2d})".format(problem, pred_nums[i], true_nums[i]))


for i in range(NUM_EPOCHS):
    os.remove(MODEL_FILE.format(i + 1))


