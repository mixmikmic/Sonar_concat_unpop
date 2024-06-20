import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

IMAGE_HEIGHT, IMAGE_WIDTH = 240, 240
TUMOR_IMAGES = 155
AGE_CLASSES = 100
MAX_SURVIVAL = 2000
df = pd.read_csv("data/survival_data.csv")
df.head()


X = np.load('data/tumors_nz.npy')
X = X[:, 14:143, :, :, :] # All images from 0 to 15 and 144 to the end are totally black
Y = df['Survival']


plt.imshow(X[100, 77, 0, :, :], cmap='gray')
np.unique(X[100, 77, 0, :, :])


def get_splitted_data(X, ages, labels):
    assert len(X) == len(labels) == len(ages)
    # OneHotEncoding for ages
    enc_table = np.eye(AGE_CLASSES)
    ages_ohe = np.array([enc_table[int(round(x))] for x in ages])
    # Normalize labels
    labels /= MAX_SURVIVAL
    # Split data into: 70% train, 15% test, 15% validation
    cuts = [int(.70*len(X)), int(.85*len(X))]
    X1_train, X1_test, X1_val = np.split(X, cuts)
    X2_train, X2_test, X2_val = np.split(ages_ohe, cuts)
    Y_train, Y_test, Y_val = np.split(labels, cuts)
    return X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, X1_val, X2_val, Y_val

X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, X1_val, X2_val, Y_val = get_splitted_data(X, df['Age'], df['Survival'])


with tf.device('/gpu:0'):
    
    main_input = Input(shape=X.shape[1:], dtype='float32', name='main_input')
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(main_input)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    cnn_out = Dense(64, activation='relu')(x)
    # auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(cnn_out)
    
    auxiliary_input = Input(shape=(AGE_CLASSES,), name='aux_input', dtype='float32')
    x = keras.layers.concatenate([cnn_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    
    
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])  # , auxiliary_output
    
    # RMSprop uses:
    # - Momentum taking knowledge from previous steps into account about where
    #   we should be heading (prevents gradient descent to oscillate)
    # - Uses recent gradients to adjust alpha
    #   (when the gradient is very large, alpha is reduced and vice-versa)
    # Later we should test if AdaDelta or Adam are improving our results (quite similar to RMSprop)
    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss={'main_output': 'mean_squared_error'},  # 'aux_output': 'mean_squared_error'
                  loss_weights={'main_output': 1.})  # , 'aux_output': 0.2

    # And trained it via:
    model.fit({'main_input': X1_train, 'aux_input': X2_train},
              {'main_output': Y_train},  # 'aux_output': Y
              epochs=50, batch_size=32, verbose=1,
              validation_data=({'main_input': X1_test, 'aux_input': X2_test}, Y_test))


for i in range(5):
    prediction = int(model.predict({'main_input': X[i:i+1], 'aux_input': ages_ohe[i:i+1]}) * MAX_SURVIVAL)
    print('Patient: {}, Age: {}, GT: {}, Prediction: {}'.format(df.values[i, 0], df.values[i, 1], df.values[i, 2], prediction))


# ### 163 patients with tumor images of sizes 240x240 of which we have 155
# 

# ### Start
# Implementing a concatenated model as shown in the Docs: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
# The model uses a small common CNN for processing one patient (155 slices x 128 width x 128 height) and just inserts the age at a certain point. With this simple model we get really bad results (main_output_loss: 263381.5156 [not getting better]).
# 
# 
# ### Possible improvements
# 1. ~~Conv3D (in CNN) which gets a 4D input - therefore add the tumor region as a dimension (transform input image into three images containing each tumor region seperately~~)
# 2. ~~One-hot encoder (in sklearn or implement manually)
#   [ I tried Keras Lambda layer but after fixing the shapes keras told me that I'm not allowed to concatenate with a non input layer ]~~
# 3. Group ages into few classes changing the regression task into a classification task (with softmax).
# 4. Evaluate if LSTM is useful for this task (before concatenating the two branches or as a concatenator?)
# 5. Evaluate our optimizer (RMSprp vs Adam vs AdaDelta)
#     -> RMSprop doesn't learn at all (Adam does!)
#     -> First run was without normalization but no improvement (After normalization quite good results)
# 6. I red full batch learning is the best way for training on few training data - check if this is helpful.
# 7. Use for a patient nx, ny and nz at the same time
# 8. Data augmentation?

# ### This notebook creates 4 CSV files with $n$ rows and  $(h * w) + 1$ columns each, where $n$ is the number of pictures per image set and $h$ and $w$ are height and width of a picture. The first $h * w$ cells of the $n$th row contain the mean of the RGB value of a pixel of the $n$th picture of the respective image set. The last column contains the labels of the respective images.
# 

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

WIDTH, HEIGHT = 124, 124


def convert_images_to_csv(images_description_path):
    CSV_PATH = "data/" + images_description_path + ".csv"
    
    # Load file with image paths
    with open("data/" + images_description_path) as f:
        data_file = f.read().splitlines() 

    # Split in paths and labels
    image_paths = []
    labels = []
    for entry in data_file:
        path, label = entry.split(' ')
        image_paths.append(path)
        labels.append(label)

    # Sanitize paths
    if not os.path.isfile(image_paths[0]):
        image_paths = [x.replace('Train/', 'Train/Train/') for x in image_paths]
    assert os.path.isfile(image_paths[0])

    # Convert image to matrix
    def png_to_matrix(path):
        def convert_rgba_to_rgb(path):
            png = Image.open(path)
            png.load() # required for png.split()
            if len(png.split()) > 3:
                background = Image.new("RGB", png.size, (255, 255, 255))
                background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
                return background
            else:
                return png
        img = convert_rgba_to_rgb(path)
        img_mean = np.mean(img, axis=2)
        return np.asarray(img_mean)
    
    
    # Test png_to_matrix with first picture
    img = png_to_matrix(image_paths[0])
    print("Image matrix shape:", img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()

    # Build dataframe with X and y
    columns = [str(x) for x in range(WIDTH * HEIGHT)] + ["label"]
    df = pd.DataFrame([], columns=columns)
    
    for path, y in zip(image_paths, labels):
        img_id = '.'.join(path.split('/')[-1].split('.')[:-1])
        try:
            img = png_to_matrix(path).reshape([-1])
            img = np.append(img, y)
            s = pd.Series(img, index=columns)
            
            if not os.path.isfile(CSV_PATH):
                df = pd.DataFrame([], columns=columns)
                df.to_csv(CSV_PATH, index=False)
            
            with open(CSV_PATH, "a") as f:
                features = ",".join([str(x) for x in s]) + "\n"
                f.write(features)
            print(".", end='')
        except Exceptiontio as e:
            print(img_id + ": " + e)


for path in tqdm(['subset-1-HnxTny.txt', 'subset-2-HnyTnz.txt', 'subset-3-HnzTnx.txt', 'subset-4-All.txt']):
    convert_images_to_csv(path)


# I aborted after adding one picture to the CSV file
pd.read_csv("data/subset-1-HnxTny.txt.csv", index_col=False)





# ### This notebook creates 4 CSV files with $n$ rows and  $(h * w) + 1$ columns each, where $n$ is the number of pictures per image set and $h$ and $w$ are height and width of a picture. The first $h * w$ cells of the $n$th row contain the mean of the RGB value of a pixel of the $n$th picture of the respective image set. The last column contains the labels of the respective images.
# 

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from PIL import Image

WIDTH, HEIGHT = 124, 124
DATA_DIR = os.path.join("..", "data")
UNHEALTHY_DIR = '/mnt/naruto/mina_workspace/Brain_5Classes/Data/HT/'
HEALTHY_LIST_FILE = 'subset-3-HnzTnx.txt'
OUTPUT_CSV = 'scull_set.txt.csv'


def get_images(directory):
    imagefiles = (os.path.join(directory, f) for f in os.listdir(directory))
    imagefiles = (f for f in imagefiles if os.path.isfile(f))
    imagefiles = ((f, 1) for f in imagefiles if f[-4:] == ".png")
    return imagefiles

def get_unhealthy_paths():
    return get_images(UNHEALTHY_DIR)

def get_healthy_paths():
    csv_file = os.path.join(DATA_DIR, HEALTHY_LIST_FILE)
    with open(csv_file) as f:
        data_file = f.read().splitlines()
    # Split in paths and labels
    imagefiles = (entry.split(' ') for entry in data_file)
    imagefiles = ((x[0], int(x[1])) for x in imagefiles if len(x) == 2)
    # Get only healthy images
    imagefiles = (x for x in imagefiles if x[1] == 0)
    # Sanitize paths
    imagefiles = ((x[0].replace('Train/', 'Train/Train/'), x[1]) for x in imagefiles)
    return imagefiles


healthy_set = list(get_healthy_paths())
unhealthy_set = list(get_unhealthy_paths())
for p, l in healthy_set:
    assert os.path.isfile(p)
    assert l == 0
for p, l in unhealthy_set:
    assert os.path.isfile(p)
    assert l == 1
print('{} healthy images'.format(len(healthy_set)))
print('{} unhealthy images'.format(len(unhealthy_set)))


def get_all_images():
    healthy_set = list(get_healthy_paths())
    unhealthy_set = list(get_unhealthy_paths())[:len(healthy_set)]
    return healthy_set + unhealthy_set

def convert_rgba_to_rgb(path):
    png = Image.open(path)
    png.load() # required for png.split()
    if len(png.split()) > 3:
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        png = background
    return png

def png_to_matrix(path):
    # Convert image to matrix
    img = np.array(convert_rgba_to_rgb(path))
    if len(img.shape) == 3:
        # Turn RGB into grayscaled image
        img = np.mean(img, axis=2)
    # Fix the first line of each healthy image (which already have the correct size)
    if img.shape == (WIDTH, HEIGHT):
        # Before there was a white row: 255
        img[0] = 0
        # Healthy images are previously rotated by -90°
        img = np.rot90(img)
        # Last row contains almost black values: 3
        img[-1, img[-1] == 3] = 0
    else:
        # Resize unhealthy images to match the healthy ones
        img = scipy.misc.imresize(img, (WIDTH, HEIGHT))
    return img.astype(np.uint8)

def show_image(path):
    print(path)
    img = png_to_matrix(path)
    print("Image matrix shape:", img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()

def prepare_output_file(file):
    columns = [str(x) for x in range(WIDTH * HEIGHT)] + ['label']
    df = pd.DataFrame([], columns=columns, dtype=np.uint8)
    df.to_csv(file, index=False)
    
def convert_images_to_csv():
    csv_output = os.path.join(DATA_DIR, OUTPUT_CSV)
    prepare_output_file(csv_output)
    image_paths, labels = zip(*get_all_images())
    # Test png_to_matrix with first (healthy) and last (unhealthy) picture
    show_image(image_paths[0])
    show_image(image_paths[-1])
    # Build dataframe with X and y
    print('Preprocessing', end='')
    content = ''
    for i, path, y in zip(range(len(labels)), image_paths, labels):
        try:
            img = png_to_matrix(path).reshape([-1])
            content += ','.join([str(x) for x in img]) + ',{}\n'.format(y)
            if (i % 500 == 0 and i != 0) or i == len(labels) - 1:
                # Only open stream after a few hundred steps due to performance
                with open(csv_output, 'a') as f:
                    f.write(content)
                content = ''
                print('.', end='')
        except Exception as e:
            img_id = '.'.join(path.split('/')[-1].split('.')[:-1])
            print(img_id + ': ' + str(e))
            print('Failed')
            return -1
    print('Done')
    return csv_output


output_file = convert_images_to_csv()


pd.read_csv(os.path.join(DATA_DIR, OUTPUT_CSV), index_col=False)


import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices() 


def get_shuffled_splitted_data(path):
    df = pd.read_csv(path)

    # Shuffle and split data
    X_train, X_test, X_val = np.split(df.sample(frac=1), [int(.7*len(df)), int(.8*len(df))])
    
    # Pop labels and transform them to vectors
    y_train, y_test, y_val = X_train.pop("label"), X_test.pop("label"), X_val.pop("label")
    y_train, y_test, y_val = y_train.values.reshape((-1, 1)), y_test.values.reshape((-1, 1)), y_val.values.reshape((-1, 1))
    
    # Reshape the features for CNN
    X_train = X_train.as_matrix().reshape(X_train.shape[0], 1, 124, 124)
    X_test = X_test.as_matrix().reshape(X_test.shape[0], 1, 124, 124)
    X_val = X_val.as_matrix().reshape(X_val.shape[0], 1, 124, 124)
    
    # Norm data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_test /= 255
    X_val /= 255
    
    # Convert labels to categorical values
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    y_val = keras.utils.to_categorical(y_val, 2)
    
    return X_train, y_train, X_test, y_test, X_val, y_val
    
X_train, y_train, X_test, y_test, X_val, y_val = get_shuffled_splitted_data('../data/subset-3-HnzTnx.txt.csv')


with tf.device('/gpu:0'):

    import keras
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Conv2D, MaxPooling2D, Cropping2D
    from keras import backend as K
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adadelta
    K.set_image_dim_ordering('th')

    batch_size = 40
    num_classes = 2
    epochs = 2

    # The data, shuffled and split between train and test sets:
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_val.shape[0], 'validation samples')

    model = Sequential()
    model.add(MaxPooling2D(pool_size=(20, 5), input_shape=(1, 124, 124)))
    model.add(Conv2D(16, kernel_size=(1, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Dense(6))
    
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_val, y_val))

    test_score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])
    
    val_score = model.evaluate(X_val, y_val, verbose=0)
    print('Val loss:', val_score[0])
    print('Val accuracy:', val_score[1])


# model.summary()
# 

model.summary()


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
# y_test_pred = model.predict(X_test, batch_size=32, verbose=0)
# y_test_pred = np.round(y_test_pred).astype(int)

def plot_sample(ax, sample, title):
    # The first line contains 65000 values for any reason
    img = sample.reshape(124, 124)[1:, 1:]
    ax.imshow(img, cmap='gray',  interpolation='nearest')
    ax.axis('off')
    ax.set_title(title)

def has_tumor(one_hot_vector):
    return one_hot_vector.argmax()
    
def plot_samples(count, samples, labels, predicted, main_title):
    # Shuffle datapoints
    idx = np.random.choice(np.arange(samples.shape[0]), count, replace=False)
    samples, labels, predicted = (samples[idx], labels[idx], predicted[idx])
    cols = 4
    rows = count // cols
    assert rows * cols == count, 'Number of samples must be a multiple of 4'
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    for i, ax in enumerate(axes.flat):
        plot_sample(ax, samples[i], '#{}, Tumor: {}, Predicted: {}'.format(
            idx[i], has_tumor(labels[i]), has_tumor(predicted[i])))
    fig.suptitle(main_title)

# Always the same results
np.random.seed(0)
plot_samples(4, X_test, y_test, y_test, 'Testing set')
no_tumors = y_test.argmax(axis=1) == 0
plot_samples(4, X_test[no_tumors], y_test[no_tumors],
             y_test[no_tumors], 'Testing set - No tumor')
plot_samples(4, X_test[no_tumors == False], y_test[no_tumors == False],
             y_test[no_tumors == False], 'Testing set - Tumor')

# keras.utils.plot_model(model, show_shapes=True, to_file='model-Small-tk.png')
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


