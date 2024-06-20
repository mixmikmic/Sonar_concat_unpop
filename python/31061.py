# As a warm-up project, the mathematical formula for the model is already provided: 
# 
# Price = -5,269 + 8,413 x Carat + 158.1 x Cut + 454 x Clarity
# 
# So what I do in this project is to verify the equation. 
# 

# ## Step 1: Understanding the Model 
# ### 1. According to the linear model provided, if a diamond is 1 carat heavier than another with the same cut and clarity, how much more should we expect to pay? Why?
# 
# Ans: We expect to pay 8413 more, because this is the coefficient for Carat
# 
# ### 2. If you were interested in a 1.5 carat diamond with a Very Good cut (represented by a 3 in the model) and a VS2 clarity rating (represented by a 5 in the model), how much would the model predict you should pay for it?
# 

carat, cut, clarity = 1.5, 3, 5
price = -5269 + 8413 * carat + 158.1 * cut + 454 * clarity
print("The price is ", price)


# ## Step 2: Visualize the Data 
# ### 1 - Plot the data for the diamonds in the database, with carat on the x-axis and price on the y-axis. 
# ### 2 - Plot the data for the diamonds for which you are predicting prices with carat on the x-axis and predicted price on the y-axis. 
# 

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
data = pd.read_csv("diamonds.csv")
test = pd.read_csv("new-diamonds.csv")


data.head()


data.median()


test.info()


test['predict'] = -5269 + 8413 * test['carat'] + 158.1 * test['cut_ord'] + 454.0 * test['clarity_ord']

plot1, = plt.plot(data["carat"],data["price"],'.r',label ="known price")
plot2, = plt.plot(test["carat"],test["predict"],'.b',label = "predicted price")
plt.legend(handles=[plot1, plot2],fontsize = 15)
plt.xlabel("carat",fontsize = 20)
plt.ylabel("price",fontsize = 20)
plt.show()


test[test['predict']>19000]


# ### 3. What strikes you about this comparison? After seeing this plot, do you feel confident in the model’s ability to predict prices? 
# 
# Ans: The predicted prices increase linearly with the carat, and has good overlap with the known price. However, it seems the predicted prices increase much faster than known prices and cause some outliers at the high end. So I will apply some discount to correct the inflated slope.
# 

# ## Step 3: Make a Recommendation
# ### What price do you recommend the jewelry company to bid? Please explain how you arrived at that number. HINT: The number should be 7 digits.
# 

total = sum(test['predict'])*0.8
print("The bid price for 3000 diamonds:", total)


# Ans: I would bid 9.4 million for the whole set. I use a discount factor 0.8 to suppress the inflated price by the model. This is just an ad hoc strategy. 
# 

# ## Task 1: Store Format for Existing Stores
# 
# - Determine the optimal number of store formats based on sales data.
# - Use percentage sales per category per store for clustering (category sales as a percentage of total store sales).
# - Use only 2015 sales data.
# - Use a K-means clustering model.
# - Segment the 85 current stores into the different store formats.
# - Use the StoreSalesData.csv and StoreInformation.csv files.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


store = pd.read_csv("storedemographicdata.csv")
info = pd.read_csv("storeinformation.csv")
sales = pd.read_csv("storesalesdata.csv")


store.head()


store.shape


info.head()


info.shape


sales.head()


sales.info()


sales.columns.values


fields = ['Dry_Grocery', 'Dairy','Frozen_Food', 'Meat','Produce','Floral', 'Deli', 'Bakery','General_Merchandise']
data = sales[sales["Year"]== 2015].loc[:,fields]


data.shape


data.plot.box(vert=False)


# get the feature correlations
corr = data.corr()

# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True

# plot the heatmap
with sns.axes_style("white"):
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu', fmt='+.2f', cbar=False)


data.describe()


# ## data scale by normalization and transform by PCA
# 

from sklearn.preprocessing import Normalizer
scale = Normalizer()
for field in fields:
    data = pd.DataFrame(scale.fit_transform(data))


data.describe()


data.columns = fields
data.plot.box(vert=False)


# get the feature correlations
corr = data.corr()

# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True

# plot the heatmap
with sns.axes_style("white"):
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu', fmt='+.2f', cbar=False)


# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca_data = pca.fit_transform(data)


np.cumsum(pca.explained_variance_ratio_)


dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

# PCA components
components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys())
components.index = dimensions

# PCA explained variance
ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
variance_ratios.index = dimensions

# Create a bar plot visualization
fig, ax = plt.subplots(figsize = (14,8))

# Plot the feature weights as a function of the components
components.plot(ax = ax, kind = 'bar');
ax.set_ylabel("Feature Weights")
ax.set_xticklabels(dimensions, rotation=0)

# Display the explained variance ratios
for i, ev in enumerate(pca.explained_variance_ratio_):
    ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))


from sklearn.cluster import KMeans
from time import time
t0 = time()
model = KMeans(n_clusters=2, random_state=0)
model.fit(pca_data)
centers = model.cluster_centers_

# sample_preds = clusterer.predict(data.iloc[0:5,:])

# TODO: Calculate the mean silhouette coefficient 
from sklearn.metrics import silhouette_score
preds = model.predict(pca_data)
score = silhouette_score(data,preds)
print(time()-t0,score)


def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.

    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)

    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'],
        facecolors='b', edgecolors='b', s=70, alpha=0.5)

    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1],
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black',
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
biplot(data,pca_data,pca)


# I realize I do pca use the wrong scale, I should not use normaliztion across the same category, I should do it accross the same store, or percentage.
# 

# ## Task 2: Determine the Store Format for New Stores
# 
# - Develop a model that predicts which segment a store falls into based on the **demographic and socioeconomic characteristics of the population** that resides in the area around each new store.
# - Use a 20% validation sample with Random Seed = 3 when creating samples with which to compare the accuracy of the models. Make sure to compare a decision tree, forest, and boosted model.
# - Use the model to predict the best store format for each of the 10 new stores.
# 




# # Language Translation
# In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
# ## Get the Data
# Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)


# ## Explore the Data
# Play around with view_sentence_range to view different parts of the data.
# 

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Function
# ### Text to Word Ids
# As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.
# 
# You can get the `<EOS>` word id by doing:
# ```python
# target_vocab_to_int['<EOS>']
# ```
# You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.
# 

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    '''
    source_ids = []
    sentences = source_text.split('\n')
    for sentence in sentences:
        sentence_words = []
        for word in sentence.split():
            sentence_words.append(source_vocab_to_int[word])
        #sentence_words.append(source_vocab_to_int['<EOS>'])
        source_ids.append(sentence_words)
    target_ids = []
    tsentences = target_text.split('\n')
    for sentence in tsentences:
        sentence_words = []
        for word in sentence.split():
            sentence_words.append(target_vocab_to_int[word])
        sentence_words.append(target_vocab_to_int['<EOS>'])  
        target_ids.append(sentence_words)
    '''    
    # list comprehension
    source_ids = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_text.split('\n')]
    target_ids = [[target_vocab_to_int[word] for word in sentence.split()] + [source_vocab_to_int['<EOS>']] for sentence in target_text.split('\n')]
        
    return source_ids,target_ids

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)


# ### Preprocess all the data and save it
# Running the code cell below will preprocess all the data and save it to file.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)


# # Check Point
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()


# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ## Build the Neural Network
# You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
# - `model_inputs`
# - `process_decoding_input`
# - `encoding_layer`
# - `decoding_layer_train`
# - `decoding_layer_infer`
# - `decoding_layer`
# - `seq2seq_model`
# 
# ### Input
# Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# 
# - Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
# - Targets placeholder with rank 2.
# - Learning rate placeholder with rank 0.
# - Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
# 
# Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)
# 

def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input_data = tf.placeholder(tf.int32, [None, None], name = "input")
    targets = tf.placeholder(tf.int32, [None, None], name = "target")
    lr = tf.placeholder(tf.float32, name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    return input_data, targets, lr, keep_prob

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


# ### Process Decoding Input
# Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the beginning of each batch.
# 

def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoding
    :param target_data: Target Placeholder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    
    return dec_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)


# ### Encoding
# Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).
# 

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    _, enc_state = tf.nn.dynamic_rnn(enc_cell, rnn_inputs, dtype=tf.float32)
    return enc_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)


# ### Decoding - Training
# Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.
# 

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
    dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
    train_logits =  output_fn(train_pred)
    return train_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)


# ### Decoding - Inference
# Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder). 
# 

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
    output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, 
    maximum_length - 1, vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
    return inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)


# ### Build the Decoding Layer
# Implement `decoding_layer()` to create a Decoder RNN layer.
# 
# - Create RNN cell for decoding using `rnn_size` and `num_layers`.
# - Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
# - Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
# - Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.
# 
# Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.
# 

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    with tf.variable_scope("decoding") as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob)
    with tf.variable_scope("decoding", reuse = True) as decoding_scope:
        inference_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'],
                         sequence_length, vocab_size, decoding_scope, output_fn, keep_prob)
    return train_logits, inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)


# ### Build the Neural Network
# Apply the functions you implemented above to:
# 
# - Apply embedding to the input data for the encoder.
# - Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
# - Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
# - Apply embedding to the target data for the decoder.
# - Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.
# 

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    return decoding_layer(dec_embed_input, dec_embeddings, enc_state, target_vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)


# ## Neural Network Training
# ### Hyperparameters
# Tune the following parameters:
# 
# - Set `epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `num_layers` to the number of layers.
# - Set `encoding_embedding_size` to the size of the embedding for the encoder.
# - Set `decoding_embedding_size` to the size of the embedding for the decoder.
# - Set `learning_rate` to the learning rate.
# - Set `keep_probability` to the Dropout keep probability
# 

# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256 #20
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256 #13
decoding_embedding_size = 256 #13
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5


# ### Build the Graph
# Build the graph using the neural network you implemented.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# ### Train
# Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forums to see if anyone is having the same problem.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        #start_time = time.time()
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
        #end_time = time.time()
        #print('time:{0:2f}, Epoch:{1}, Validation Accuracy: {2:.3f}, Loss: {3:.3f}'.format(end_time-start_time,epoch_i, valid_acc, loss))
        # Batch number: 1077
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
    # rnn size 20, embedding size 13, learning rate 0.01 , 5 epoch, accuracy: 0.6-0.808, 180 s per epoch
    # rnn size 256, embedding size 128, learning rate 0.002 , 2 epoch, accuracy: 0.8-0.932, 1000 s per epoch
    # rnn size 256, embedding size 128, learning rate 0.002 , 10 epoch, accuracy: 0.893-0.982, 164 s per epoch, cloud gpu


import os
os.system('say "your program has finished"')


# ## continue training if needed
# 

'''
with tf.Session(graph=train_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess,save_path)

    for epoch_i in range(5):
        start_time = time.time()
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: 0.001,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            
            #print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
            #      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
        end_time = time.time()
        print('time:{0}, Epoch:{1}, Validation Accuracy: {2:.3f}, Loss: {3:.3f}'
                  .format(end_time-start_time,epoch_i, valid_acc, loss))
        # Batch number: 1077
    # Save Model
    #saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
    # learning rate:0.01 , 5 epoch: 0.80~0.85
    # learning rate:0.002 , 5 epoch: 0.85~0.88
    # learning rate:0.001 , 5 epoch: 0.881~0.895
    # learning rate:0.001 , 5 epoch: 0.895~0.906
'''


# ### Save Parameters
# Save the `batch_size` and `save_path` parameters for inference.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)


# # Checkpoint
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()


# ## Sentence to Sequence
# To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.
# 
# - Convert the sentence to lowercase
# - Convert words into ids using `vocab_to_int`
# - Convert words not in the vocabulary, to the `<UNK>` word id.
# 

def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    sentence = sentence.lower()
    ids = []
    for word in sentence.split():
        if word in vocab_to_int:
            ids.append(vocab_to_int[word])
        else:
            ids.append(vocab_to_int["<UNK>"])
    return ids

    # list comprehension
    # [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.lower().split()]

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)


# ## Translate
# This will translate `translate_sentence` from English to French.
# 

translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))


# ## Imperfect Translation
# You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  Additionally, the translations in this data set were made by Google translate, so the translations themselves aren't particularly good.  (We apologize to the French speakers out there!) Thankfully, for this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.
# 
# You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
# ## Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


movie = pd.read_csv("movies.csv")


movie.head(2)


movie.info()


sum(movie["release_year"]>=2000) 


# ### Movie industry changes over time. Among the 10866 movies since 1960,  7168 was produced since 2000, or 66%. I will be focused on the data in 2000-2015.
# 

movie = movie [movie["release_year"]>=2000]


sns.boxplot(movie["vote_average"])


plt.figure (figsize=(10,5))
ax = sns.countplot(movie["release_year"])
plt.xticks(rotation=90)
# ax.set_xticklabels(movie["release_year"], rotation=90,fontsize=15)
plt.show()


num = movie.shape[0]


# ## **Question 1:** How have movie genres changed over time?
# 
# **It turns out Tableau is much better at aggregation plot than seaborn/matplotlib.**
# 

genres = movie['genres'].unique() # len: 2040, due to remix
movie['genres'].value_counts()[0:10]


data = movie.loc[:,['id','release_year','genres']]


data['genres']=data['genres'].apply(lambda s: str(s).split("|")[0])


data.head()


sns.countplot(x = "release_year", hue = "genres", data = data)


gb = data.groupby(["release_year", "genres"]).count()


gb.head()


# ##  Q 2: How do the attributes differ between Universal Pictures and Paramount Pictures?
# 

attributes = ["budget_adj", "revenue_adj", "tagline", "keywords","genres","production_companies", "vote_average","release_year"]
data = movie[attributes]


data.info()


data["production_companies"]=data["production_companies"].apply(lambda s: str(s).split("|")[0])


data["production_companies"].value_counts()[0:10]


two = data[(data["production_companies"] == "Paramount Pictures")| 
           (data["production_companies"] =="Universal Pictures") ]


two1 = two[(two["budget_adj"]>0) & (two["revenue_adj"]>0)]  # remove missing data


two1.info()


two1.groupby("production_companies").median()


two1["production_companies"].value_counts()


# ## Q3: How have movies based on novels performed relative to movies not based on novels?
# 

data = movie[movie["keywords"].notnull()]  # 5904
cnt = 0
total = data.shape[0]
for i,keyword in data['keywords'].iteritems():
    if "based" in keyword:
        cnt += 1
        #print(cnt,i,keyword)
print(cnt,total)
#  since 1960 /2000
# "novel" 295 /193
# "true" 58
# "based" 495 /344
# "nudity" 283/151


attributes = ["budget_adj", "revenue_adj", "tagline", "keywords","genres","production_companies", "vote_average","release_year"]
data = data[attributes]


data ['novel'] = data['keywords'].apply(lambda x: "novel" in x)


data1 = data[(data["budget_adj"]>0) & (data["revenue_adj"]>0)] 


data1.groupby('novel').median()


data1['novel'].value_counts()


130/2260


# ## Q4: What are the best-selling movies
# 

movie.info()


movie[movie["original_title"]== "Avatar"]


data['genres']=data['genres'].apply(lambda s: str(s).split("|")[0])


data [data['genres']== "Animation"].info()





# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*
# 

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
# 

# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.
# 

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.
# 

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
#data.tail(n=3)


#data.info()  # , no missing data
data.shape


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.
# 
# **Hint:** You may need to look at the table above to understand how the `'income'` entries are formatted. 
# 

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = sum(data['income']=='>50K')

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = sum(data['income']=='<=50K')

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k*100.0/n_records

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)


# ----
# ## Preparing the Data
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.
# 

# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.
# 

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 
# 

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.
# 

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
features_raw[0:1]


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
#  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_raw'` data.
#  - Convert the target label `'income_raw'` to numerical entries.
#    - Set records with "<=50K" to `0` and records with ">50K" to `1`.
# 

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
income = pd.get_dummies(income_raw)['>50K']
#income = pd.DataFrame(income)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
#print encoded
#income.info()


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.
# 

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])





# ----
# ## Evaluating Model Performance
# In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.
# 

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *UdacityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would is appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *UdacityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *UdacityML* would identify no one as donors. 
# 

# ### Question 1 - Naive Predictor Performace
# *If we chose a model that always predicted an individual made more than \$50,000, what would that model's accuracy and F-score be on this dataset?*  
# **Note:** You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.
# 

# TODO: Calculate accuracy
accuracy = greater_percent/100.0
recall = 1

# TODO: Calculate F-score using the formula above for beta = 0.5
beta = 0.5
fscore = (1+ beta**2)*accuracy*recall/(beta**2*accuracy+recall)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)


# ###  Supverised Learning Models
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression
# 

# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# - *Describe one real-world application in industry where the model can be applied.* (You may need to do research for this — give references!)
# - *What are the strengths of the model; when does it perform well?*
# - *What are the weaknesses of the model; when does it perform poorly?*
# - *What makes this model a good candidate for the problem, given what you know about the data?*
# 

# **Answer: ** 
# ### Naive Bayes classifier
# - used in spam email detection. https://en.wikipedia.org/wiki/Naive_Bayes_classifier
# http://www.dataschool.io/comparing-supervised-learning-algorithms/
# - strength: highly scalable, has fast training and prediction speed
# performs well in simple supervised classification.
# - weakness: accuracy is relatively lower
# performs poorly when features have correlation.
# - reason for choosing: highly scalable, easy to apply.
# ### Decision Trees
# - used in many classification problems, including Medical diagnosis, Manufacturing, and Insurance.
# - strength: white box, simple to understand and interpret
# - weakness: doesn't generalize well and may overfit
# - reason for choosing: easy to interpret the model
# ### Surport vector machine
# - used in text and image categorizations, biological science.
# - strength: training is easy
# - weakness: need a good kernel function
# - reason for choosing: relatively high accuracy
# 

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!
# 

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train[0:sample_size],y_train[0:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[0:300],predictions_train )
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[0:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Dependent on which algorithms you chose, the following implementation may take some time to run!
# 

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = SVC(kernel="linear",random_state=0)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 =  X_train.shape[0]/100
samples_10 = X_train.shape[0]/10
samples_100 =X_train.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 
# 

# ### Question 3 - Choosing the Best Model
# *Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000.*  
# **Hint:** Your answer should include discussion of the metrics, prediction/training time, and the algorithm's suitability for the data.
# 

# **Answer: ** In terms of the Accuracy Score and F-socre on the Testing Set,**SVC** is the best model amonge the three. The only downside is that SVC has significantly longer training/predicting time. But this can be improved by optimizing the model kernel or reducing training data size, which still keep high prediction score.
# 

# ### Question 4 - Describing the Model in Layman's Terms
# *In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*
# 

# **Answer: ** 
# 1. The SVM takes data about individuals whose census data is known (e.g., age, gender, etc) and uses them to create a function that draws a boundary between individuals with income over and under 50k. The boundary should be drawn so as to maximize the difference to either group
# 
# 2. Often, though, it's not easy to draw a decision boundary in low dimensions, so the SVM separates the high & low income individuals by multiple dimensions.
# 
# 3. Using this function created with individuals we already know earn over or under 50k, the SVM can look at new potential donors' data and predict their incomes by the decision boundary.
# 

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Initialize the classifier you've chosen and store it in `clf`.
#  - Set a `random_state` if one is available to the same state you set before.
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
#  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# 
# **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!
# 

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = SVC(random_state=0)

# TODO: Create the parameters list you wish to tune
param_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10,100]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, param_grid,scoring = scorer)


# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))


best_clf


# ### Question 5 - Final Model Evaluation
# _What is your optimized model's accuracy and F-score on the testing data? Are these scores better or worse than the unoptimized model? How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.
# 

# #### Results:
# 
# |     Metric     | Benchmark Predictor | Unoptimized Model | Optimized Model |
# | :------------: | :-----------------: | :---------------: | :-------------: | 
# | Accuracy Score |       0.24              |    0.8301               | 0.8516                |
# | F-score        |       0.29              |    0.6592               |   0.7109       |
# 

# **Answer: ** Optimized Model is better than Unoptimized Model and Benchmark Predictor
# 

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.
# 

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data.  
# _Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?_
# 

data['income']=income
#data[0:1]


#data['age'].describe()
bins =range(10,91,10)
group = pd.cut(data['age'],bins,labels=range(10,90,10))
data['AgeGroup']=group


import seaborn
seaborn.factorplot('AgeGroup','income',data=data,hue='sex',aspect=3)


# **Answer:** marital-status, education-num, sex, age, race
# because only enough money can support a marriage life, education usually means the skill and knowledge level, male usually earn more than female, older people usually have more experience, race is due to the historical and cultural reasons.
# 

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.
# 

# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
# TODO: Train the supervised model on the training set 
model = DecisionTreeClassifier()
#model = AdaBoostClassifier()
#model = RandomForestClassifier()
model.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# _How do these five features compare to the five features you discussed in **Question 6**? If you were close to the same answer, how does this visualization confirm your thoughts? If you were not close, why do you think these features are more relevant?_
# 

# **Answer:** 
# DecisionTree:
# marital-status, age, education-num, capital-gain, hours-per-week
# AdaBoost ranking:
# capital-loss,age,capital-gain, hours-per-week, education-num
# RandomForest ranking: 
# age, hours-per-week, captial-gain, relationship, education-num
# 
# It's very interesting to me that different classifier actually rank the features differently while they can all achieve decently good accuracy. 
# I think these are correlation instead of causal relation, so each feature make somewhat sense but not a single one feature is dominant. 
# 

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 
# 

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))


# ### Question 8 - Effects of Feature Selection
# *How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?*  
# *If training time was a factor, would you consider using the reduced data as your training set?*
# 

# **Answer:** The F-score and accuracy score are reduced a little bit. 
# If we have time constrain and can tolerate some sacrifice of the accuracy, I would use the reduced data. 
# 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 

# http://docs.h5py.org/en/latest/quick.html
# HDF stands for Hierarchical Data Format. 
# 
# An HDF5 file is a container for two kinds of objects:
# 
# 1. **datasets**. array-like collection of data. work like numpy arrays.
# 2. **groups**. folder-like containers that hold dataset and other group. work like dictionaries. 
# 

import h5py, numpy as np
f = h5py.File("mytestfile.hdf5", "w") # create hdf5 file
dset = f.create_dataset("mydataset", (100,), dtype='i') # create dataset
dset[...] = np.arange(100)  # assign value
dset.name  # u'/mydataset'
f.name   # u'/', root group

grp = f.create_group("subgroup")   # create group
dset2 = grp.create_dataset("another_dataset", (50,), dtype='f') # create dataset from grp
dset2.name  # u'/subgroup/another_dataset'
dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i') # create dataset and group

# print all keys
for name in f:
  print name
f.keys() # another to access all keys  

# print all keys and subgroups
def printname(name):
  print name
f.visit(printname)

# add attribute to dataset
dset.attrs['temperature'] = 99.5
dset.attrs['temperature']
for name in dset.attrs:
  print name

# add attribute to group  
grp.attrs['hello'] = 9
for i in grp.attrs:
  print i


# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
# 

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.
# 

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


data.head()


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.
# 

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.
# 

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# _Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each._  
# **Hint:** Would you expect a home that has an `'RM'` value of 6 be worth more or less than a home that has an `'RM'` value of 7?
# 

data.corr()


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
for i, col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    plt.plot(data[col], prices, 'o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')


# **Answer: **
# 1. 'RM': the larger this value, the higher housing price. Because it means larger house.
# 2. 'LSTAT': the larger this value, the smaller housing price. Because it means less developed community.
# 3. 'PTRATIO': the larger this value, the smaller housing price. Because it means fewer education resources per person.
# 

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.
# 

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.
# 

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true,y_predict)
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# *Would you consider this model to have successfully captured the variation of the target variable? Why or why not?* 
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.
# 

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)


# **Answer:** Yes, it predicts the correct trend and the values don't deviate too much.
# 

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.
# 

# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)

# Success
print "Training and testing split was successful."


# ### Question 3 - Training and Testing
# *What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?*  
# **Hint:** What could go wrong with not having a way to test your model?

# **Answer: ** The training set can be used to fit the model parameters and the testing set can be use to see the model accuracy:
# 1. Gives estimate of performance on an indepdent dataset;
# 2. Serves as check on overfitting
# 

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.
# 

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.
# 

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# *Choose one of the graphs above and state the maximum depth for the model. What happens to the score of the training curve as more training points are added? What about the testing curve? Would having more training points benefit the model?*  
# **Hint:** Are the learning curves converging to particular scores?

# **Answer: ** For max_depth=10, the training socre is close to one, but the testing curve is only about 0.7. More training points don't improve the model.
# 

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# Run the code cell below and use this graph to answer the following two questions.
# 

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# *When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?*  
# **Hint:** How do you know when a model is suffering from high bias or high variance?

# **Answer: ** For max_depth=1, the model suffer from high bias, because it pay little attention to data and has high error on trainning set. For max_depth=10, the model suffer from high variance, because the training score is quite different from the validation score.
# 

# ### Question 6 - Best-Guess Optimal Model
# *Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?*
# 

# **Answer: ** Max_depth= 4 will make a good model because its validation score is the highest among all.
# 

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.
# 

# ### Question 7 - Grid Search
# *What is the grid search technique and how it can be applied to optimize a learning algorithm?*
# 

# **Answer: ** Grid search is an algorithm for hyperparameter opimization. It's also called parameter sweep, because it just scans through a manually specified subset of the hyperparameter space.
# 
# To apply it to a learning algorithm, we first make a dictionary of the setting parameters, then pass it to the classifier. For example:
# ```python
# dict = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clf = grid_search.GridSearchCV(clf_default, dict)
# ```
# where clf_default is the classifier with its defaulst setting or empty setting, and dict is used to generate different combinations of the parameter. In the above case, there are 4 possible combinations:
# ('rbf', 1)	('rbf', 10)
# ('linear', 1)	('linear', 10)
# which forms a 2*2 grids.
# 

# ### Question 8 - Cross-Validation
# *What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?*  
# **Hint:** Much like the reasoning behind having a testing set, what could go wrong with using grid search without a cross-validated set?

# **Answer: ** k-fold cross-validation is to divide the dataset into k pieces and run k times, each time using 1 piece as the validations set and the other k-1 pieces as the training set.
# 
# This technique maximimize the use of dataset and can improve the accuracy of the model.
# 
# If the grid search is without cross-validated set, the model may be overfitting. Because we don't know how the model performs on an "unseen" dataset.
# 

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_iter'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.
# 

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor,param_grid=params,scoring= scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.
# 

# ### Question 9 - Optimal Model
# _What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?_  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.
# 

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# **Answer: ** max_depth = 4, the same with my previous guess. 
# 

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# *What price would you recommend each client sell his/her home at? Do these prices seem reasonable given the values for the respective features?*  
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  
# 
# Run the code block below to have your optimized model make predictions for each client's home.
# 

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


data.corr()


# **Answer: ** The predicted prices for 3 clients are 391k, 189k, 943k dollars, respectively. They seem reasonable because the worse features lead to cheaper prices. Client 3 has the best features and it's no surprising its price is much higher than other two. 
# 

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with the data it's trained on.
# 

vs.PredictTrials(features, prices, fit_model, client_data)


import matplotlib.pyplot as plt
plt.hist(prices, bins = 20)
for price in reg.predict(client_data):
    plt.axvline(price, lw = 5, c = 'r')


# ### Question 11 - Applicability
# *In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.*  
# **Hint:** Some questions to answering:
# - *How relevant today is data that was collected from 1978?*
# - *Are the features present in the data sufficient to describe a home?*
# - *Is the model robust enough to make consistent predictions?*
# - *Would data collected in an urban city like Boston be applicable in a rural city?*
# 

# **Answer: ** This model is not enough to make good prediction, because:
# 1. the data is almost 40 years ago and should be updated instead of simple correction of inflation.
# 2. Other features such as occupating area, crime rate, age of house. The original dataset has included 13 features beside the price.
# 3. As shown in the PredictTrials, the variance range is about 10% of the housing price. So the model is not robust enough.
# 4. No. There are many difference between urban and rural city, which usually depends on the development of the civilization. 
# 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 

import pandas as pd
import json


df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                   index=['row 1', 'row 2'],
                   columns=['col 1', 'col 2'])


df


# ## By split
# 

a = df.to_json(orient='split')


print json.dumps(json.loads(a), indent=4, sort_keys=True) 


pd.read_json(a, orient='split')


# ## by index
# 

b = df.to_json(orient='index')  ## b is a string, need to be parsed


di = json.loads(b) # parsed into a dictionary


j = json.dumps(di, indent=4, sort_keys=True)  # stringify to a string with indent and order


import pprint ## pprint is not able to parsed, it takes advantage of the existing data structure
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(b)


pp.pprint(di)


print j


for row in di:
    print di[row]


pd.read_json(b, orient='index')


pd.read_json(j, orient='index')


# ## by record
# 

c = df.to_json(orient='records')


print json.dumps(json.loads(c), indent=4, sort_keys=True) 


pd.read_json(c, orient='records')





# # Face Generation
# In this project, you'll use generative adversarial networks to generate new images of faces.
# ### Get the Data
# You'll be using two datasets in this project:
# - MNIST
# - CelebA
# 
# Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.
# 
# If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".
# 

data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)


# ## Explore the Data
# ### MNIST
# As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 
# 

show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
get_ipython().magic('matplotlib inline')
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# ### CelebA
# The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.
# 

show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ## Preprocess the Data
# Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.
# 
# The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
# ## Build the Neural Network
# You'll build the components necessary to build a GANs by implementing the following functions below:
# - `model_inputs`
# - `discriminator`
# - `generator`
# - `model_loss`
# - `model_opt`
# - `train`
# 
# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# - Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
# - Z input placeholder with rank 2 using `z_dim`.
# - Learning rate placeholder with rank 0.
# 
# Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)
# 

import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    lr = tf.placeholder(tf.float32, name = "learning_rate")
    return inputs_real, inputs_z, lr


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


# ### Discriminator
# Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).
# 

def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function
    alpha=0.2
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 28x28xN
        x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        # 16x16x64
        
        x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)
        # 8x8x128
        
        x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        # 4x4x256

        # Flatten it
        flat = tf.reshape(relu3, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        
        return out, logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)


# ### Generator
# Implement `generator` to generate an image using `z`. This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.
# 

def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function
    alpha=0.2
    with tf.variable_scope('generator', reuse = (not is_train)):
        # First fully connected layer
        x1 = tf.layers.dense(z, 7*7*256)
        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 7, 7, 256))
        x1 = tf.layers.batch_normalization(x1, training = is_train)
        x1 = tf.maximum(alpha * x1, x1)
        # 7x7x256 now
        
        x2 = tf.layers.conv2d_transpose(x1, 128, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training = is_train)
        x2 = tf.maximum(alpha * x2, x2)
        # 14x14x128 now
        
        #x3 = tf.layers.conv2d_transpose(x2, 112, 5, strides=2, padding='same')
        #x3 = tf.layers.batch_normalization(x3, training = is_train)
        #x3 = tf.maximum(alpha * x3, x3)
        
        
        # Output layer
        logits = tf.layers.conv2d_transpose(x2, out_channel_dim, 5, strides=2, padding='same')
        # 28x28x dim now
        
        out = tf.tanh(logits)
        
        return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)


# ### Loss
# Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
# - `discriminator(images, reuse=False)`
# - `generator(z, out_channel_dim, is_train=True)`
# 

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    alpha=0.2
    
    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)


# ### Optimization
# Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).
# 

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)


# ## Neural Network Training
# ### Show Output
# Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.
# 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


# ### Train
# Implement `train` to build and train the GANs.  Use the following functions you implemented:
# - `model_inputs(image_width, image_height, image_channels, z_dim)`
# - `model_loss(input_real, input_z, out_channel_dim)`
# - `model_opt(d_loss, g_loss, learning_rate, beta1)`
# 
# Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.
# 

from time import time
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = time()
        for epoch_i in range(epoch_count):
            i = 0
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                i+=1
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                #print(batch_images.shape)
                d, g = sess.run([d_train_opt,g_train_opt], feed_dict={input_real: batch_images, input_z: batch_z})
                if i%100==0:
                    print("epoch:{}, batch:{}, time:{}".format(epoch_i,i, time()-t0))
                    dl,gl = sess.run([d_loss, g_loss], feed_dict={input_real: batch_images, input_z: batch_z})
                    print("d_loss:{}, g_loss:{}".format(dl,gl))
                    show_generator_output(sess, 10, input_z, data_shape[3], data_image_mode)
                


# ### MNIST
# Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.
# 

mnist_dataset.shape


batch_size = 128
z_dim = 100
learning_rate = 0.001
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


# ### CelebA
# Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.
# 

batch_size = 128
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5
# take 3000 seconds
# learning_rate 0.001, d_loss: 0.005, g_loss: 5.74757
# learning_rate 0.0002, d_loss:0.0359, g_loss:11.715
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)


# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
# 

# change theme of ipython notebooks
from IPython.core.display import HTML
import urllib2
HTML(urllib2.urlopen('http://bit.ly/1Bf5Hft').read())


# ## Format 1, work flow
# 1. use h5py to read mat file, pickle dump as "_matadata.pickle"
# 2. use cv2 to imread png file, crop into single digits, greyscale, resize , data scale and pickle dump as " _dataset_labels.pickle"
# 

# step 1
import numpy as np
import cPickle as pickle
import h5py

#f = h5py.File('train/digitStruct.mat')
f = h5py.File('test/digitStruct.mat')

metadata= {}
metadata['height'] = []
metadata['label'] = []
metadata['left'] = []
metadata['top'] = []
metadata['width'] = []

def print_attrs(name, obj):
    vals = []
    if obj.shape[0] == 1:
        vals.append(int(obj[0][0]))
    else:
        for k in range(obj.shape[0]):
            vals.append(int(f[obj[k][0]][0][0]))
    metadata[name].append(vals)

for item in f['/digitStruct/bbox']:
    f[item[0]].visititems(print_attrs)
    
#with open('train_metadata.pickle','wb') as pf:
with open('test_metadata.pickle','wb') as pf:
    pickle.dump(metadata, pf, pickle.HIGHEST_PROTOCOL)    


# check the number of digits
image_num = 33402 # 33402 for train, 13068 for test
count = 0
for i in range(image_num):                
    digit_num += len(metadata['width'][i])  
print count


# step 2 

import cPickle as pickle
#with open('train_metadata.pickle', 'rb') as f:
with open('test_metadata.pickle', 'rb') as f:  
    metadata = pickle.load(f)

import numpy as np 
import cv2
#image_num = 33402
#sample_num = 73257
image_num =  13068
sample_num = 26032

dataset = np.ndarray(shape=(sample_num, 28, 28),dtype=np.float32)
lables = np.ndarray(shape=(sample_num, ),dtype=np.int)

def crop(image, i,j):
    top = metadata['top'][i][j]
    height = metadata['height'][i][j]
    left = metadata['left'][i][j]
    width = metadata['width'][i][j]
    if left < 0:
        left, width = 0, width+left
  
    return image[top:top+height, left:left+width]

depth = 255.0  # pixel depth
for i in range(image_num):
    #path = 'train/{0}.png'.format(i+1)
    path = 'test/{0}.png'.format(i+1)
    image = cv2.imread(path)
    num = len(metadata['width'][i])  
    for j in range(num):
        crop_image = crop(image,i,j)
        gray_image = rgb2gray(crop_image)
        #print i,j  # find (250,0) has left value of -1
        resize_image = cv2.resize(gray_image,(28,28))
        normal_image = resize_image/depth -0.5

        dataset[count,:,:] = normal_image
        lables[count] = metadata['label'][i][j] % 10

#with open('train_dataset_labels.pickle','wb') as pf:
with open('test_dataset_labels.pickle','wb') as pf:  
    pickle.dump((dataset,lables), pf, pickle.HIGHEST_PROTOCOL)  
  
del metadata # clean cache


# ## format 2
# use scipy to load mat file, greyscale, data scale and pickle dump into 'train_test_32x32.pickle'
# 

import numpy as np
import scipy.io
# train_mat = scipy.io.loadmat('train_32x32.mat')  # dict.key() ['y', 'X', '__version__', '__header__', '__globals__']

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def mat2data(matfile):
    mat = scipy.io.loadmat(matfile)
    Xdata = mat['X']
    ydata = mat['y']
    size = Xdata.shape   # (32,32,3,73257)
    print 'size = {0}'.format(size)
    image_size_x = size[0]
    image_size_y = size[0]
    num_samples = size[3]

    depth = 255.0  # pixel depth
    dataset = np.ndarray(shape=(num_samples, image_size_x, image_size_y),dtype=np.float32)
    labels = np.ndarray(shape=(num_samples,),dtype=np.int8)
    for i in range(num_samples):
        dataset[i,:,:] = rgb2gray(Xdata[:,:,:,i]) /depth - 0.5  # 3D-2D and normalize
        labels[i] = ydata[i][0] % 10
        return dataset, labels

X_train,y_train = mat2data('train_32x32.mat')
X_test, y_test  = mat2data('test_32x32.mat') 
with open('train_test_32x32.pickle','wb') as pf:
    pickle.dump(((X_train,y_train),(X_test, y_test)), pf, pickle.HIGHEST_PROTOCOL)  


# works for python 3.0
from IPython.core.display import HTML
import urllib.request
request = urllib.request.Request('http://bit.ly/1Bf5Hft')
response = urllib.request.urlopen(request)
HTML(response.read().decode('utf-8'))


# [project link](https://docs.google.com/document/d/1-OkpZLjG_kX9J6LIQ5IltsqMzVWjh36QpnP2RYpVdPU/pub?embedded=True)
# 
# https://review.udacity.com/#!/rubrics/71/view
# 
# https://en.wikipedia.org/wiki/Stroop_effect
# 
# 
# # Background Information
# 
# In a Stroop task, participants are presented with a list of words, with each word displayed in a color of ink. The participant’s task is to say out loud the color of the ink in which the word is printed. The task has two conditions: a congruent words condition, and an incongruent words condition. In the congruent words condition, the words being displayed are color words whose names match the colors in which they are printed: for example RED, BLUE. In the incongruent words condition, the words displayed are color words whose names do not match the colors in which they are printed: for example PURPLE, ORANGE. In each case, we measure the time it takes to name the ink colors in equally-sized lists. Each participant will go through and record a time from each condition.
# 

# # Questions For Investigation
# 
# As a general note, be sure to keep a record of any resources that you use or refer to in the creation of your project. You will need to report your sources as part of the project submission.
# 
# 1. What is our independent variable? What is our dependent variable?
# 
# **Ans:**  independent variable is the two different conditions, whether it is congruent or incongruent.
# 
# dependent variable is the reading time.

# \2. What is an appropriate set of hypotheses for this task? What kind of statistical test do you expect to perform? Justify your choices.
# 
# Ans: 
# 
# - Null hypothesis: the congruent group and the incongruent group has the same mean value, with a significant value of 5%. $H_0: \mu_1=\mu_2$. Here $\mu$ is the mean value for each group.
# - Alternative hypothesis:The mean of the incongruent group is outside the critical region of the mean of congruent group. $H_a: \mu_1 \neq \mu_2$
# 
# The kind of statistical test is **dependent sample t-test** on 2 conditions. Because we have relative few samples (<30), we don't know the mean and standard deviation of the population, and we assume the distribution are Gaussian
# 
# 

# \3. Report some descriptive statistics regarding [this dataset](https://www.google.com/url?q=https://drive.google.com/file/d/0B9Yf01UaIbUgQXpYb2NhZ29yX1U/view?usp%3Dsharing&sa=D&ust=1487887532881000&usg=AFQjCNF6W3Dm_zTuLyhH2jZJBFsMnET7ng) . Include at least one measure of central tendency and at least one measure of variability.
# 
# **Ans:** As shown below, the congruent group has a mean of 14.05 and std of 3.56, the incongruent group has a mean of 22.02 and std of 4.80
# 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

data = pd.read_csv('stroopdata.csv')


data.info()


data.describe()


# \4. Provide one or two visualizations that show the distribution of the sample data. Write one or two sentences noting what you observe about the plot or plots.
# **Ans:** As shown below, the incongruent group has signficantly longer time.
# 

data.plot(kind='kde')


data.boxplot()


# \5. Now, perform the statistical test and report your results. What is your confidence level and your critical statistic value? Do you reject the null hypothesis or fail to reject it? Come to a conclusion in terms of the experiment task. Did the results match up with your expectations?
# 
# **Ans:** This is a dependent sample test. From t-table, for 95% confident level with 23 degree of freedom, the t critical value is 2.069. So t=8.02 means the incongruent task take much more time than the congruent task. It is **statistically significant**. So we reject the null hypothesis, which is what I expect. 

s1 = data['Congruent']
s2 = data['Incongruent']
mean_diff = s2.mean()-s1.mean()
length = len(s1)
sample_sd = np.sqrt(np.var(s2-s1)*length/(length-1))
t = mean_diff/sample_sd*np.sqrt(length)
print('mean_diff={0:.2f}\n sample_sd={1:.3f}\n t={2:.3f}'.format(mean_diff,sample_sd,t))


# \6. Optional: What do you think is responsible for the effects observed? Can you think of an alternative or similar task that would result in a similar effect? Some research about the problem will be helpful for thinking about these two questions!
# 
# **ans:** The effect is due to the inconsistency between the name tag and the actual property of a given object. In psychology, there is a theory called cognitive dissonance, which desribes the conflicts between 2 cognitive systems. The Stroop effect is real life is actually **more hidden and subtle**, and  can be used to affect customer's behavior. Like the background laughing in a comedian show, the exaggerated ad of a big brand product (i.e., Coke, or Pizza)
# 




# ## Step 1: Business and Data Understanding
# 
# Key Decisions:
#  
# ### 1.     What decisions needs to be made?
# Ans: Whether to send out $6.50 catalog to the 250 new customers which are the potential buyers.
#  
# ### 2.     What data is needed to inform those decisions?
# Ans: Whether the total profit from these 250 new customers will meet expected value of 10,000. To break it down, we will need to predict ave_sale_amount for each customer.
# 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
customer = pd.read_excel("p1-customers.xlsx")
mail = pd.read_excel("p1-mailinglist.xlsx")


customer.head()


#mail.head()


customer.info()


mail.info()


# ## Step 2: Analysis, Modeling, and Validation
# ### 1 How and why did you select the predictor variables in your model? 
# Ans: The first thing is the target variable. My origional thought is `Responded_to_Last_Catalog`, but it turns out the majority of customers doesn't response (2204 vs 171), these customers actually has a higher `Avg_Sale_Amount` (308 vs 162). Another confusing thing is in `p1-mailinglist.xlsx` : `Score_Yes`. How is this probability calculated? This is important but however beyond this question.
# 
# With some commom sense and some data exploration, the predictor variables are Customer_Segment (multi-class) and Avg_Num_Products_Purchased (continuous). 
# 

customer['Responded_to_Last_Catalog'].value_counts()


sns.boxplot(x='Responded_to_Last_Catalog', y = "Avg_Sale_Amount",data = customer)


sns.regplot(x='Avg_Num_Products_Purchased', y = "Avg_Sale_Amount",data = customer)


sns.boxplot(x="Customer_Segment", y = "Avg_Sale_Amount",data = customer)


sns.boxplot(x="City", y = "Avg_Sale_Amount",data = customer)


sns.jointplot(x='#_Years_as_Customer', y ='Avg_Sale_Amount', data = customer)


dummies = pd.get_dummies(customer['Customer_Segment'])
X = pd.concat([customer['Avg_Num_Products_Purchased'], dummies], axis=1)
X = X.drop("Credit Card Only",axis=1)  # use "Credit Card Only" as default
y = customer["Avg_Sale_Amount"]
X.head()


import statsmodels.api as sm
#from scipy import stats
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# ### 2. Explain why you believe your linear model is a good model.
# 
# Ans: The R value is 0.837, indicating a good fit. The p value is almost zero for each variable, indicating good choices of variables.
# 
# ### 3.What is the best linear regression equation based on the available data?
# 
# Ans: y = 303.46 + 66.98 x Avg_Num_Products_Purchased - 149.36 x Loyalty Club Only +
#                                               281.84 x Loyalty Club and Credit Card - 245.42 x Store Mailing List 
# 

# ## Step 3: Presentation/Visualization
# ### 1.  What is your recommendation? Should the company send the catalog to these 250 customers?
# 
# 
# Ans: Yes
# 
# ### 2.   How did you come up with your recommendation?
# 
# Ans: I used the linear model to predict how much individual cusotemer will pay, times its chance to buy, times gross margin rate, minus cost and compare with expected profit. If it exceeds expectation, it is a buy. 
# 
# ### 3.   What is the expected profit from the new catalog
# 
# Ans: According to the following analysis, the average customer will pay \$ 504 with 26% chance. And the total profit after 50% gross margin and 6.50 cost will be \$ 21,986. This exceeds the threshold profit of \$10,000. So I would recommend to do this catalog ad campaign. 
# 
# In addition, in the `p1-customers.xlsx`, the customers who don't respond may also contribute to "Avg_Sale_Amount". I don't have a clear explanation at this moment.
# 

from sklearn.linear_model import LinearRegression, Ridge
#model = LinearRegression() # ridicularly larege intercept
model = Ridge()
model.fit(X, y)
model.coef_, model.intercept_


model.score(X,y)


dummies = pd.get_dummies(mail['Customer_Segment'])
X_test = pd.concat([mail['Avg_Num_Products_Purchased'], dummies], axis=1)
X_test = X_test.drop("Credit Card Only", axis = 1)
X_test.head()


y_test = model.predict(X_test)


import numpy as np
print(np.median(y_test))
print(mail["Score_Yes"].median())


sum(mail["Score_Yes"]*y_test*0.5) - 6.5 *250


(mail["Score_Yes"]*y_test)[0]  # first record of revenue


# # Your first neural network
# 
# In this project, you'll build your first neural network and use it to predict daily bike rental ridership. We've provided some of the code, but left the implementation of the neural network up to you (for the most part). After you've submitted this project, feel free to explore the data and the model more.
# 
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# change theme of ipython notebooks
from IPython.core.display import HTML
import urllib.request
request = urllib.request.Request('http://bit.ly/1Bf5Hft')
response = urllib.request.urlopen(request)
HTML(response.read().decode('utf-8'))


# ## Load and prepare the data
# 
# A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!
# 

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)


#rides.info()
rides.describe()


#rides.head()
sum(rides['cnt'])


# ## Checking out the data
# 
# This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the `cnt` column. You can see the first few rows of the data above.
# 
# Below is a plot showing the number of bike riders over the first 10 days in the data set. You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.
# 

rides[:24*10].plot(x='dteday', y='cnt')


# ### Dummy variables
# Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to `get_dummies()`.
# 

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# ### Scaling target variables
# To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
# 
# The scaling factors are saved so we can go backwards when we use the network for predictions.
# 

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# ### Splitting the data into training, testing, and validation sets
# 
# We'll save the last 21 days of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.
# 

# Save the last 21 days 
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).
# 

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# ## Time to build the network
# 
# Below you'll build your network. We've built out the structure and the backwards pass. You'll implement the forward pass through the network. You'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes.
# 
# The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.
# 
# We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.
# 
# > **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation $y = x$. What is the slope of that equation? That is the derivative of $f(x)$.
# 
# Below, you have these tasks:
# 1. Implement the sigmoid function to use as the activation function. Set `self.activation_function` in `__init__` to your sigmoid function.
# 2. Implement the forward pass in the `train` method.
# 3. Implement the backpropagation algorithm in the `train` method, including calculating the output error.
# 4. Implement the forward pass in the `run` method.
#   
# 

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))  #(2,56)

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes)) #(1,2)
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1+np.exp(-x))

    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T  #(56,1)
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        #print(inputs.shape)  #(56,1)
        #print(self.weights_input_to_hidden.shape) # (2,56)
        hidden_inputs = np.dot(self.weights_input_to_hidden , inputs) #(2,1)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) #(1,1)
        final_outputs = final_inputs 
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error
        output_errors = targets - final_outputs  #(1,1)
        
        # TODO: Backpropagated error
        hidden_errors = output_errors  #(1,1)
        hidden_grad = hidden_errors * self.weights_hidden_to_output.T * hidden_outputs * (1-hidden_outputs)  #(2,1)
        
        # TODO: Update the weights
        self.weights_hidden_to_output +=  self.lr * np.dot(hidden_errors , hidden_outputs.T) #(1,2)
        self.weights_input_to_hidden += self.lr * np.dot(hidden_grad , inputs.T)
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden , inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs 
        
        return final_outputs


def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## Training the network
# 
# Here you'll set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.
# 
# You'll also be using a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. You use many more training passes than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.
# 
# ### Choose the number of epochs
# This is the number of times the dataset will pass through the network, each time updating the weights. As the number of epochs increases, the network becomes better and better at predicting the targets in the training set. You'll need to choose enough epochs to train the network well but not too many or you'll be overfitting.
# 
# ### Choose the learning rate
# This scales the size of weight updates. If this is too big, the weights tend to explode and the network fails to fit the data. A good choice to start at is 0.1. If the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.
# 
# ### Choose the number of hidden nodes
# The more hidden nodes you have, the more accurate predictions the model will make. Try a few different numbers and see how it affects the performance. You can look at the losses dictionary for a metric of the network performance. If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.
# 

import sys

### Set the hyperparameters here ###
epochs = 300
learning_rate = 0.1
hidden_nodes = 30
output_nodes = 1

N_i = train_features.shape[1]  # 56
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4]                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)
plt.show()


# ## Check out your predictions
# 
# Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.
# 

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# ## Thinking about your results
#  
# Answer these questions about your results. How well does the model predict the data? Where does it fail? Why does it fail where it does?
# 
# > **Note:** You can edit the text in this cell by double clicking on it. When you want to render the text, press control + enter
# 
# #### Your answer below
# - Overall,the neural network model predict pretty well, the peaks and valleys have amazing matches.
# - It seem the model has much larger error for the peak values during Dec.22~31. The reason, I guess, is probably in the holiday season, people tend to change their behaviors more than usual.

# ## Unit tests
# 
# Run these unit tests to check the correctness of your network implementation. These tests must all be successful to pass the project.
# 

import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)





# works for python 3.0
from IPython.core.display import HTML
import urllib.request
request = urllib.request.Request('http://bit.ly/1Bf5Hft')
response = urllib.request.urlopen(request)
HTML(response.read().decode('utf-8'))


# # Bay Area Bike Share Analysis
# 
# ## Introduction
# 
# [Bay Area Bike Share](http://www.bayareabikeshare.com/) is a company that provides on-demand bike rentals for customers in San Francisco, Redwood City, Palo Alto, Mountain View, and San Jose. Users can unlock bikes from a variety of stations throughout each city, and return them to any station within the same city. Users pay for the service either through a yearly subscription or by purchasing 3-day or 24-hour passes. Users can make an unlimited number of trips, with trips under thirty minutes in length having no additional charge; longer trips will incur overtime fees.
# 
# In this project, you will put yourself in the shoes of a data analyst performing an exploratory analysis on the data. You will take a look at two of the major parts of the data analysis process: data wrangling and exploratory data analysis. But before you even start looking at data, think about some questions you might want to understand about the bike share data. Consider, for example, if you were working for Bay Area Bike Share: what kinds of information would you want to know about in order to make smarter business decisions? Or you might think about if you were a user of the bike share service. What factors might influence how you would want to use the service?
# 
# **Question 1**: Write at least two questions you think could be answered by data.
# 
# **Answer**: From business's perspective, I want to know which group of people is my major customer, what is the market size, where should I set the renting stations, what is the reasonable price to attract users, How many bikes should I put in each station.
# 
# From user's perspective, I want to know whether the shared bike will save me time and money, and whether it feels good to ride.
# 

# ## Using Visualizations to Communicate Findings in Data
# 
# As a data analyst, the ability to effectively communicate findings is a key part of the job. After all, your best analysis is only as good as your ability to communicate it.
# 
# In 2014, Bay Area Bike Share held an [Open Data Challenge](http://www.bayareabikeshare.com/datachallenge-2014) to encourage data analysts to create visualizations based on their open data set. You’ll create your own visualizations in this project, but first, take a look at the [submission winner for Best Analysis](http://thfield.github.io/babs/index.html) from Tyler Field. Read through the entire report to answer the following question:
# 
# **Question 2**: What visualizations do you think provide the most interesting insights? Are you able to answer either of the questions you identified above based on Tyler’s analysis? Why or why not?
# 
# **Answer**: The most interesting insight is Tyler successfully characterize two groups of users: commuters and tourist. They have quite different bike renting habits.
# 
# His analysis helped me answer some of my questions, like the major customer.
# 
# But the problem is still the business goal. As far as I know, the renting system is government funded project. It is unclear how to measure the bike sharing has benefit the public transport system. What can we do to make it more cost-efficient?

# ## Data Wrangling
# 
# Now it's time to explore the data for yourself. Year 1 and Year 2 data from the Bay Area Bike Share's [Open Data](http://www.bayareabikeshare.com/open-data) page have already been provided with the project materials; you don't need to download anything extra. The data comes in three parts: the first half of Year 1 (files starting `201402`), the second half of Year 1 (files starting `201408`), and all of Year 2 (files starting `201508`). There are three main datafiles associated with each part: trip data showing information about each trip taken in the system (`*_trip_data.csv`), information about the stations in the system (`*_station_data.csv`), and daily weather data for each city in the system (`*_weather_data.csv`).
# 
# When dealing with a lot of data, it can be useful to start by working with only a sample of the data. This way, it will be much easier to check that our data wrangling steps are working since our code will take less time to complete. Once we are satisfied with the way things are working, we can then set things up to work on the dataset as a whole.
# 
# Since the bulk of the data is contained in the trip information, we should target looking at a subset of the trip data to help us get our bearings. You'll start by looking at **only the first month of the bike trip data, from 2013-08-29 to 2013-09-30**. The code below will take the data from the first half of the first year, then write the first month's worth of data to an output file. This code exploits the fact that the data is sorted by date (though it should be noted that the first two days are sorted by trip time, rather than being completely chronological).
# 
# First, load all of the packages and functions that you'll be using in your analysis by running the first code cell below. Then, run the second code cell to read a subset of the first trip data file, and write a new file containing just the subset we are initially interested in.
# 

# import all necessary packages and functions.
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from babs_datacheck import question_3
from babs_visualizations import usage_stats, usage_plot
from IPython.display import display
get_ipython().magic('matplotlib inline')


# file locations
file_in  = '201402_trip_data.csv'
file_out = '201309_trip_data.csv'

with open(file_out, 'w') as f_out, open(file_in, 'r') as f_in:
    # set up csv reader and writer objects
    in_reader = csv.reader(f_in)
    out_writer = csv.writer(f_out)

    # write rows from in-file to out-file until specified date reached
    while True:
        datarow = next(in_reader)
        # trip start dates in 3rd column, m/d/yyyy HH:MM formats
        if datarow[2][:9] == '10/1/2013':
            break
        out_writer.writerow(datarow)


# ### Condensing the Trip Data
# 
# The first step is to look at the structure of the dataset to see if there's any data wrangling we should perform. The below cell will read in the sampled data file that you created in the previous cell, and print out the first few rows of the table.
# 

sample_data = pd.read_csv('201309_trip_data.csv')

display(sample_data.head())


sample_data.info()


# In this exploration, we're going to concentrate on factors in the trip data that affect the number of trips that are taken. Let's focus down on a few selected columns: **the trip duration, start time, start terminal, end terminal, and subscription type**. Start time will be divided into year, month, and hour components. We will also add a column for the day of the week and abstract the start and end terminal to be the start and end _city_.
# 
# Let's tackle the lattermost part of the wrangling process first. Run the below code cell to see how the station information is structured, then observe how the code will create the station-city mapping. Note that the station mapping is set up as a function, `create_station_mapping()`. Since it is possible that more stations are added or dropped over time, this function will allow us to combine the station information across all three parts of our data when we are ready to explore everything.
# 

# Display the first few rows of the station data file.
station_info = pd.read_csv('201402_station_data.csv')
display(station_info.head())


# This function will be called by another function later on to create the mapping.
def create_station_mapping(station_data):
    """
    Create a mapping from station IDs to cities, returning the
    result as a dictionary.
    """
    station_map = {}
    for data_file in station_data:
        with open(data_file, 'r') as f_in:
            # set up csv reader object - note that we are using DictReader, which
            # takes the first row of the file as a header row for each row's
            # dictionary keys
            weather_reader = csv.DictReader(f_in)  # csv.DictReader object

            for row in weather_reader:
                station_map[row['station_id']] = row['landmark']
    return station_map


# You can now use the mapping to condense the trip data to the selected columns noted above. This will be performed in the `summarise_data()` function below. As part of this function, the `datetime` module is used to **p**arse the timestamp strings from the original data file as datetime objects (`strptime`), which can then be output in a different string **f**ormat (`strftime`). The parsed objects also have a variety of attributes and methods to quickly obtain
# 
# There are two tasks that you will need to complete to finish the `summarise_data()` function. First, you should perform an operation to convert the trip durations from being in terms of seconds to being in terms of minutes. (There are 60 seconds in a minute.) Secondly, you will need to create the columns for the year, month, hour, and day of the week. Take a look at the [documentation for datetime objects in the datetime module](https://docs.python.org/2/library/datetime.html#datetime-objects). **Find the appropriate attributes and method to complete the below code.**
# 

create_station_mapping(['201402_station_data.csv']).keys()


create_station_mapping(['201402_station_data.csv']).values()


def summarise_data(trip_in, station_data, trip_out):
    """
    This function takes trip and station information and outputs a new
    data file with a condensed summary of major trip information. The
    trip_in and station_data arguments will be lists of data files for
    the trip and station information, respectively, while trip_out
    specifies the location to which the summarized data will be written.
    """
    # generate dictionary of station - city mapping
    station_map = create_station_mapping(station_data)
    
    with open(trip_out, 'w') as f_out:
        # set up csv writer object        
        out_colnames = ['duration', 'start_date', 'start_year',
                        'start_month', 'start_hour', 'weekday',
                        'start_city', 'end_city', 'subscription_type']        
        trip_writer = csv.DictWriter(f_out, fieldnames = out_colnames)
        trip_writer.writeheader()
        
        for data_file in trip_in:
            with open(data_file, 'r') as f_in:
                # set up csv reader object
                trip_reader = csv.DictReader(f_in)

                # collect data from and process each row
                for row in trip_reader:
                    new_point = {}
                    
                    # convert duration units from seconds to minutes
                    ### Question 3a: Add a mathematical operation below   ###
                    ### to convert durations from seconds to minutes.     ###
                    new_point['duration'] = float(row['Duration'])/60
                    
                    # reformat datestrings into multiple columns
                    ### Question 3b: Fill in the blanks below to generate ###
                    ### the expected time values.                         ###
                    trip_date = datetime.strptime(row['Start Date'], '%m/%d/%Y %H:%M')
                    new_point['start_date']  = trip_date.strftime('%Y-%m-%d')
                    new_point['start_year']  = trip_date.strftime('%Y')
                    new_point['start_month'] = trip_date.strftime('%m')
                    new_point['start_hour']  = trip_date.strftime('%H')
                    new_point['weekday']     = trip_date.strftime('%w')
                    
                    # remap start and end terminal with start and end city
                    new_point['start_city'] = station_map[row['Start Terminal']]
                    new_point['end_city'] = station_map[row['End Terminal']]
                    # two different column names for subscribers depending on file
                    if 'Subscription Type' in row:
                        new_point['subscription_type'] = row['Subscription Type']
                    else:
                        new_point['subscription_type'] = row['Subscriber Type']

                    # write the processed information to the output file.
                    trip_writer.writerow(new_point)


# **Question 3**: Run the below code block to call the `summarise_data()` function you finished in the above cell. It will take the data contained in the files listed in the `trip_in` and `station_data` variables, and write a new file at the location specified in the `trip_out` variable. If you've performed the data wrangling correctly, the below code block will print out the first few lines of the dataframe and a message verifying that the data point counts are correct.
# 

# Process the data by running the function we wrote above.
station_data = ['201402_station_data.csv']
trip_in = ['201309_trip_data.csv']
trip_out = '201309_trip_summary.csv'
summarise_data(trip_in, station_data, trip_out)

# Load in the data file and print out the first few rows
sample_data = pd.read_csv(trip_out)
display(sample_data.head())

# Verify the dataframe by counting data points matching each of the time features.
question_3(sample_data)


# > **Tip**: If you save a jupyter Notebook, the output from running code blocks will also be saved. However, the state of your workspace will be reset once a new session is started. Make sure that you run all of the necessary code blocks from your previous session to reestablish variables and functions before picking up where you last left off.
# 
# ## Exploratory Data Analysis
# 
# Now that you have some data saved to a file, let's look at some initial trends in the data. Some code has already been written for you in the `babs_visualizations.py` script to help summarize and visualize the data; this has been imported as the functions `usage_stats()` and `usage_plot()`. In this section we'll walk through some of the things you can do with the functions, and you'll use the functions for yourself in the last part of the project. First, run the following cell to load the data, then use the `usage_stats()` function to see the total number of trips made in the first month of operations, along with some statistics regarding how long trips took.
# 

trip_data = pd.read_csv('201309_trip_summary.csv')

usage_stats(trip_data)


trip_data['duration'].describe()


# You should see that there are over 27,000 trips in the first month, and that the average trip duration is larger than the median trip duration (the point where 50% of trips are shorter, and 50% are longer). In fact, the mean is larger than the 75% shortest durations. This will be interesting to look at later on.
# 
# Let's start looking at how those trips are divided by subscription type. One easy way to build an intuition about the data is to plot it. We'll use the `usage_plot()` function for this. The second argument of the function allows us to count up the trips across a selected variable, displaying the information in a plot. The expression below will show how many customer and how many subscriber trips were made. Try it out!
# 

usage_plot(trip_data, 'subscription_type')


# Seems like there's about 50% more trips made by subscribers in the first month than customers. Let's try a different variable now. What does the distribution of trip durations look like?

usage_plot(trip_data, 'duration')


# Looks pretty strange, doesn't it? Take a look at the duration values on the x-axis. Most rides are expected to be 30 minutes or less, since there are overage charges for taking extra time in a single trip. The first bar spans durations up to about 1000 minutes, or over 16 hours. Based on the statistics we got out of `usage_stats()`, we should have expected some trips with very long durations that bring the average to be so much higher than the median: the plot shows this in a dramatic, but unhelpful way.
# 
# When exploring the data, you will often need to work with visualization function parameters in order to make the data easier to understand. Here's where the third argument of the `usage_plot()` function comes in. Filters can be set for data points as a list of conditions. Let's start by limiting things to trips of less than 60 minutes.
# 

usage_plot(trip_data, 'duration', ['duration < 60'])


# This is looking better! You can see that most trips are indeed less than 30 minutes in length, but there's more that you can do to improve the presentation. Since the minimum duration is not 0, the left hand bar is slighly above 0. We want to be able to tell where there is a clear boundary at 30 minutes, so it will look nicer if we have bin sizes and bin boundaries that correspond to some number of minutes. Fortunately, you can use the optional "boundary" and "bin_width" parameters to adjust the plot. By setting "boundary" to 0, one of the bin edges (in this case the left-most bin) will start at 0 rather than the minimum trip duration. And by setting "bin_width" to 5, each bar will count up data points in five-minute intervals.
# 

usage_plot(trip_data, 'duration', ['duration < 60'], boundary = 0, bin_width = 5)


# **Question 4**: Which five-minute trip duration shows the most number of trips? Approximately how many trips were made in this range?
# 
# **Answer**: 5~10 minutes trip are the most frequent with about 9000 trips

# Visual adjustments like this might be small, but they can go a long way in helping you understand the data and convey your findings to others.
# 
# ## Performing Your Own Analysis
# 
# Now that you've done some exploration on a small sample of the dataset, it's time to go ahead and put together all of the data in a single file and see what trends you can find. The code below will use the same `summarise_data()` function as before to process data. After running the cell below, you'll have processed all the data into a single data file. Note that the function will not display any output while it runs, and this can take a while to complete since you have much more data than the sample you worked with above.
# 

from time import time
station_data = ['201402_station_data.csv',
                '201408_station_data.csv',
                '201508_station_data.csv' ]
trip_in = ['201402_trip_data.csv',
           '201408_trip_data.csv',
           '201508_trip_data.csv' ]
trip_out = 'babs_y1_y2_summary.csv'

# This function will take in the station data and trip data and
# write out a new data file to the name listed above in trip_out.
t0 = time()
summarise_data(trip_in, station_data, trip_out)
print('time cost ={0:.2}'.format(time()-t0))


# Since the `summarise_data()` function has created a standalone file, the above cell will not need to be run a second time, even if you close the notebook and start a new session. You can just load in the dataset and then explore things from there.
# 

trip_data = pd.read_csv('babs_y1_y2_summary.csv')
display(trip_data.head())


# #### Now it's your turn to explore the new dataset with `usage_stats()` and `usage_plot()` and report your findings! Here's a refresher on how to use the `usage_plot()` function:
# - first argument (required): loaded dataframe from which data will be analyzed.
# - second argument (required): variable on which trip counts will be divided.
# - third argument (optional): data filters limiting the data points that will be counted. Filters should be given as a list of conditions, each element should be a string in the following format: `'<field> <op> <value>'` using one of the following operations: >, <, >=, <=, ==, !=. Data points must satisfy all conditions to be counted or visualized. For example, `["duration < 15", "start_city == 'San Francisco'"]` retains only trips that originated in San Francisco and are less than 15 minutes long.
# 
# If data is being split on a numeric variable (thus creating a histogram), some additional parameters may be set by keyword.
# - "n_bins" specifies the number of bars in the resultant plot (default is 10).
# - "bin_width" specifies the width of each bar (default divides the range of the data by number of bins). "n_bins" and "bin_width" cannot be used simultaneously.
# - "boundary" specifies where one of the bar edges will be placed; other bar edges will be placed around that value (this may result in an additional bar being plotted). This argument may be used alongside the "n_bins" and "bin_width" arguments.
# 
# You can also add some customization to the `usage_stats()` function as well. The second argument of the function can be used to set up filter conditions, just like how they are set up in `usage_plot()`.
# 

usage_stats(trip_data)


usage_plot(trip_data, 'duration', ['duration < 60'], boundary = 0, bin_width = 5)


# One you're done with your explorations, copy the two visualizations you found most interesting into the cells below, then answer the following questions with a few sentences describing what you found and why you selected the figures. Make sure that you adjust the number of bins or the bin limits so that they effectively convey data findings. Feel free to supplement this with any additional numbers generated from `usage_stats()` or place multiple visualizations to support your observations.
# 

trip_data.groupby('subscription_type').count()


import seaborn as sns
from matplotlib import pyplot as plt
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='subscription_type', data=trip_data, ax=axis1)
sns.countplot(x='weekday', hue="subscription_type", data=trip_data, ax=axis2)


# **Question 5a**: What is interesting about the above visualization? Why did you select it?
# 
# **Answer**: It shows that Subscriber is the dominante user group and they use shared bike for commute to work. On weekends, they are no different from Customer.

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='start_month', hue="subscription_type", data=trip_data, ax=axis1)
sns.countplot(x='start_hour', hue="subscription_type", data=trip_data, ax=axis2)


# **Question 5b**: What is interesting about the above visualization? Why did you select it?
# 
# **Answer**: It seems the seasons won't affect the user too much, but the time shows clear peak hours for Subsribers.

# ## Conclusions
# 
# Congratulations on completing the project! This is only a sampling of the data analysis process: from generating questions, wrangling the data, and to exploring the data. Normally, at this point in the data analysis process, you might want to draw conclusions about our data by performing a statistical test or fitting the data to a model for making predictions. There are also a lot of potential analyses that could be performed on the data which are not possible with only the code given. Instead of just looking at number of trips on the outcome axis, you could see what features affect things like trip duration. We also haven't looked at how the weather data ties into bike usage.
# 
# **Question 6**: Think of a topic or field of interest where you would like to be able to apply the techniques of data science. What would you like to be able to learn from your chosen subject?
# 
# **Answer**: I want to look into the relation between educational level and househeld income level. A very important goal is what is the minimal househeld income to fully support a child's education, how much can government provdie loan or subsidy
# 

