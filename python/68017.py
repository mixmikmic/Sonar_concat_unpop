# # Invariance check
# This notebook loads the architectures :
# - baseline
# - invariant
# 
# And check that the invariant is really invariant under the dihedral transformation of the input
# 

import tensorflow as tf
import numpy as np
from astropy.io import fits
import importlib.util
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Declare utility functions to load the architectures
# 

def load_module(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_arch(arch_path, bands):
    arch = load_module(arch_path)
    nn = arch.CNN()

    g = tf.Graph()
    with g.as_default():
        nn.create_architecture(bands=bands)
    return g, nn


# In the next cell, the achitectures are loaded and the NN graph is built
# 

graph_baseline, nn_baseline = load_arch("arch_baseline.py", 1)
graph_invariant, nn_invariant = load_arch("arch_invariant.py", 1)


# Then two session are created. A session contain the values of the variables of a graph.
# 

sess_baseline = tf.Session(graph=graph_baseline)
sess_invariant = tf.Session(graph=graph_invariant)


# Now all the variables of the sessions are initialized according to how they where created (see arch code).
# 

sess_baseline.run(tf.variables_initializer(graph_baseline.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
sess_invariant.run(tf.variables_initializer(graph_invariant.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))


# Load an image and normalize it
# 

image = fits.open('samples/space_based/lens/imageEUC_VIS-100002.fits')[0].data
plt.imshow(image)
image = nn_baseline.prepare(image.reshape(101, 101, 1))


# The following function can apply all the transformation of the dihedral group on an image (for i = 0 to 7)
# 

def dihedral(x, i):
    x = x.copy()
    if i & 4:
        x = np.transpose(x, (1, 0, 2))  # tau[4]
    if i & 1:
        x = x[:, ::-1, :]  # tau[1]
    if i & 2:
        x = x[::-1, :, :]  # tau[2]
    return x


# The image x is transformed in all the different manner with the previous function
# 

images = np.array([dihedral(image, i) for i in range(8)])


# The architecture baseline, at least when it is not trained, is not expected to give the same result for all the xs.
# 

ps_baseline = sess_baseline.run(nn_baseline.tfp, feed_dict={nn_baseline.tfx: images})
plt.plot(ps_baseline)
print(ps_baseline)


# On the other hand, the invariant architecture is expected to give the same result up to rounding error
# 

ps_invariant = sess_invariant.run(nn_invariant.tfp, feed_dict={nn_invariant.tfx: images})
plt.plot(ps_invariant)
print(ps_invariant)


# Furthermore, the regular representation of the intermediate tensor can be obseved. (nn_invariant.test is a tensor close to the end of the graph, see the code)
# 

test = sess_invariant.run(nn_invariant.test, feed_dict={nn_invariant.tfx: images})
test = np.reshape(test, (8, 8, -1))


# The following graph shows the value of the test tensor for the different transformation of the image
# 

step = test[0].max() - test[0].min()
for i in range(8):
    plt.plot(test[i].flatten() + step * i)


# Up to a specific permutation that can be obtained with the help of the mutiplication table of the group, the test tensor is expected to not depend on the transformation of the input image.
# 

mt = np.array([ [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
                [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
# tau[mt[a,b]] = tau[a] o tau[b]

iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
# tau[iv[a]] is the inverse of tau[a]

for i in range(8):
    plt.plot(test[i][mt[i]].flatten() + step * i)





import tensorflow as tf
import numpy as np
import glob
import os
from astropy.io import fits
import importlib.util
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def load_module(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_arch(arch_path, bands):
    arch = load_module(arch_path)
    nn = arch.CNN()

    g = tf.Graph()
    with g.as_default():
        nn.create_architecture(bands=bands)
    return g, nn


graph, nn = load_arch("arch_views.py", 4)


def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)


sess = tf.Session(graph=graph)
os.system('cd trained_variables/ground_based/ && 7za e views.7z.001')
load_backup(sess, graph, 'trained_variables/ground_based/views')


# download the training set on [this page](http://metcalf1.bo.astro.it/blf-portal/gg_challenge.html)
# then run the `fits_to_npz.py` script to generate npz files
# 

files = sorted(glob.glob('../LE-data/GroundBasedTraining/npz/*.npz'))[:3000]


concat = []
for i in range(0, len(files), 50):
    images = [np.load(x)['image'] for x in files[i: i + 50]]
    images = nn.prepare(np.array(images))
    res = sess.run(nn.embedding_input, feed_dict={nn.tfx: images})
    concat.append(res)
    print(i, end=' ')
concat = np.array(concat).reshape((3000, 4))


plt.hist(concat[:,0], bins=150);


plt.hist(concat[:,1], bins=150);


plt.hist(concat[:,2], bins=150);


plt.hist(concat[:,3], bins=150);





# # Making perdictions from fits images
# 

import os
import tensorflow as tf
import numpy as np
from astropy.io import fits
import importlib.util


def load_module(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_arch(arch_path, bands):
    arch = load_module(arch_path)
    nn = arch.CNN()

    g = tf.Graph()
    with g.as_default():
        nn.create_architecture(bands=bands)
    return g, nn

def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)


# ## load invariant architecture and reload trained variables from a file
# 

graph, nn = load_arch("arch_invariant.py", 1)
sess = tf.Session(graph=graph)
load_backup(sess, graph, 'trained_variables/space_based/invariant')


# ## load fits images
# 

path = 'samples/space_based/lens/'
images = [fits.open(os.path.join(path, file))[0].data for file in os.listdir(path)]

path = 'samples/space_based/nolens/'
images += [fits.open(os.path.join(path, file))[0].data for file in os.listdir(path)]

images = nn.prepare(np.array(images).reshape((-1, 101, 101, 1)))


# ## make predictions
# 

predictions = nn.predict(sess, images)
predictions


# ### plot the images with the predictions
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


fig = plt.figure(figsize=(10,3))

for i in range(len(images)):
    a = fig.add_subplot(2, len(images) // 2,i+1)
    img = a.imshow(images[i, :, :, 0])
    img.set_cmap('hot')
    a.axis('off')
    a.annotate('p={:.3f}'.format(predictions[i]), xy=(10,80), color='white')

plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)





# # Get started script
# 
# This script shows how to 
# - load fits images
# - load architecture
# - initialize architecture
# - restore variables architecture from a file
# - make a prediction
# 

import tensorflow as tf
import numpy as np
from astropy.io import fits
import importlib.util
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# The ground based images of the challenge contains 4 bands 
# 

lens_band1 = fits.open('samples/ground_based/lens/Band1/imageSDSS_R-100002.fits')[0].data
lens_band2 = fits.open('samples/ground_based/lens/Band2/imageSDSS_I-100002.fits')[0].data
lens_band3 = fits.open('samples/ground_based/lens/Band3/imageSDSS_G-100002.fits')[0].data
lens_band4 = fits.open('samples/ground_based/lens/Band4/imageSDSS_U-100002.fits')[0].data
lens = np.stack([lens_band1, lens_band2, lens_band3, lens_band4], 2)

# show the green band
plt.imshow(lens[:,:,2])


nolens_band1 = fits.open('samples/ground_based/nolens/Band1/imageSDSS_R-100004.fits')[0].data
nolens_band2 = fits.open('samples/ground_based/nolens/Band2/imageSDSS_I-100004.fits')[0].data
nolens_band3 = fits.open('samples/ground_based/nolens/Band3/imageSDSS_G-100004.fits')[0].data
nolens_band4 = fits.open('samples/ground_based/nolens/Band4/imageSDSS_U-100004.fits')[0].data
nolens = np.stack([nolens_band1, nolens_band2, nolens_band3, nolens_band4], 2)

# show the green band
plt.imshow(nolens[:,:,2])


# Now let's create a neural network with the architecture defined in the file `arch_baseline.py`
# 

def load_module(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_arch(arch_path, bands):
    arch = load_module(arch_path)
    nn = arch.CNN()

    g = tf.Graph()
    with g.as_default():
        nn.create_architecture(bands=bands)
    return g, nn


graph, nn = load_arch("arch_baseline.py", 4)


# `graph` is an object from the tensorflow libraray that contain the graph of the NN (operation = verticies, tensors = edges)
# 
# The NN has been trained with images normalized with the statistics of the full dataset. The function `nn.prepare` simply subtract the mean and divide by the std. mean and std that has been computed on the full dataset.
# 

images = nn.prepare(np.array([lens, nolens]))


sess = tf.Session(graph=graph)


# Let make a prediction with the NN randomly initialized
# 

sess.run(tf.variables_initializer(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))


nn.predict(sess, images)


# Now lets load the backup of a training from a backup file.
# 

def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)


load_backup(sess, graph, 'trained_variables/ground_based/baseline')


# Now the prediction should give better results
# 

nn.predict(sess, images)


# If the normalisation is done only with the statistics of these two images it gives less good results
# 

images = (images - images.mean()) / images.std()


nn.predict(sess, images)





