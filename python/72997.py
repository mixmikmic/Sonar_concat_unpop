# # Predicting Airline Data using a Generalized Linear Model (GLM) in Keras
# 
# In particular, we will predict the probability that a flight is late based on its departure date/time, the expected flight time and distance, the origin and destitation airports.
# 
# Most part of this notebooks are identical to what has been done in Airline Delay with a GLM in python3.ipynb
# The main difference is that we will use the [Keras](https://keras.io/) high-level library with a tensorflow backend (theano backend is also available) to perform the machine learning operations instead of scikit-learn.
# 
# The core library for the dataframe part is [pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).<br>
# The core library for the machine learning part is [Keras](https://keras.io/).  This library is mostly used for deeplearning/neural-network machine learning, but it can also be used to implement most of Generalized Linear Models.  It is also quite easy to add new types of model layers into the Keras API if new functionalities would be needed.
# 
# The other main advantage of Keras is that it a high level API on top of either tensorflow/theano.  Writting new complex model is in Keras is much more simple than in tensorflow/theano.  But keep the benefits of these low-level library for what concerns the computing performances on CPU/GPU.
# 
# ### Considerations
# 
# The objective of this notebook is to define a simple model offerring a point of comparison in terms of computing performances across datascience language and libraries.  In otherwords, this notebook is not for you if you are looking for the most accurate model in airline predictions.  
# 

# ## Install and Load useful libraries
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ## Load the data (identical to python3 scikit-learn)
# 
# - The dataset is taken from [http://stat-computing.org](http://stat-computing.org/dataexpo/2009/the-data.html).  We take the data corresponding to year 2008.
# - We restrict the dataset to the first million rows
# - We print all column names and the first 5 rows of the dataset
# 

df = pd.read_csv("2008.csv")
df.shape[0]


df = df[0:1000000]


df.columns


df[0:5]


# ## Data preparation for training (identical to python3 scikit-learn)
# 
# - We turn origin/destination categorical data to a "one-hot" encoding representation
# - We create a new "binary" column indicating if the flight was delayed or not.
# - We show the first 5 rows of the modified dataset
# - We split the dataset in two parts:  a training dataset and a testing dataset containing 80% and 20% of the rows, respectively.
# 

df = pd.concat([df, pd.get_dummies(df["Origin"], prefix="Origin")], axis=1);
df = pd.concat([df, pd.get_dummies(df["Dest"  ], prefix="Dest"  )], axis=1);
df = df.dropna(subset=["ArrDelay"]) 
df["IsArrDelayed" ] = (df["ArrDelay"]>0).astype(int)
df[0:5]


train = df.sample(frac=0.8)
test  = df.drop(train.index)


# ## Model building
# 
# - We define the generalized linear model using a binomial function --> Logistic regression.
#    - The model has linear logits = (X*W)+B = (Features * Coefficients) + Bias
#    - The Loss function is a logistic function (binary_cross_entropy)
#    - A L2 regularization is added to mimic what is done in scikit learn
#    - Specific callbacks are defined (one for logging and one for early stopping the training)
# - We train the model and measure the training time --> ~55sec on an intel i7-6700K (4.0 GHz) with a GTX970 4GB GPU for 800K rows 
#    - The model is trained using a minibatch strategy (that can be tune for further performance increase)
# - We show the model coefficients
# - We show the 10 most important variables
# 

#get the list of one hot encoding columns
OriginFeatCols = [col for col in df.columns if ("Origin_" in col)]
DestFeatCols   = [col for col in df.columns if ("Dest_"   in col)]
features = train[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
labels   = train["IsArrDelayed"]
featuresMatrix = features.as_matrix()
labelsMatrix   = labels  .as_matrix().reshape(-1,1)


featureSize     = features.shape[1]
labelSize       = 1
training_epochs = 25
batch_size      = 2500

from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.regularizers import l2, activity_l2
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

#DEFINE A CUSTOM CALLBACK
class IntervalEvaluation(Callback):
    def __init__(self): super(Callback, self).__init__()
    def on_epoch_end(self, epoch, logs={}): print("interval evaluation - epoch: %03d - loss:%8.6f" % (epoch, logs['loss']))

#DEFINE AN EARLY STOPPING FOR THE MODEL
earlyStopping = EarlyStopping(monitor='loss', patience=1, verbose=0, mode='auto')
        
#DEFINE THE MODEL
model = Sequential() 
model.add(Dense(labelSize, input_dim=featureSize, activation='sigmoid', W_regularizer=l2(1e-5))) 
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 

#FIT THE MODEL
model.fit(featuresMatrix, labelsMatrix, batch_size=batch_size, nb_epoch=training_epochs,verbose=0,callbacks=[IntervalEvaluation(),earlyStopping]);


coef = pd.DataFrame(data=model.layers[0].get_weights()[0], index=features.columns, columns=["Coef"])
coef = coef.reindex( coef["Coef"].abs().sort_values(axis=0,ascending=False).index )  #order by absolute coefficient magnitude
coef[ coef["Coef"].abs()>0 ] #keep only non-null coefficients
coef[ 0:10 ] #keep only the 10 most important coefficients


# ## Model testing (identical to python3 scikit-learn)
# 
# - We add a model prediction column to the testing dataset
# - We show the first 10 rows of the test dataset (with the new column)
# - We show the model ROC curve
# - We measure the model Area Under Curve (AUC) to be 0.689 on the testing dataset.  
# 
# This is telling us that our model is not super accurate  (we generally assume that a model is raisonable at predicting when it has an AUC above 0.8).  But, since we are not trying to build the best possible model, but just show comparison of data science code/performance accross languages/libraries.
# If none the less you are willing to improve this result, you should try adding more feature column into the model.
# 

testFeature = test[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
pred = model.predict( testFeature.as_matrix() )
test["IsArrDelayedPred"] = pred
test[0:10]


fpr, tpr, _ = roc_curve(test["IsArrDelayed"], test["IsArrDelayedPred"])
AUC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.3f)' % AUC)
plt.legend(loc=4)


AUC


# ## Key takeaways
# 
# - We built a GLM model predicting airline delay probability in tensorflow
# - We train it on 800K rows in ~55sec on an intel i7-6700K (4.0 GHz) with a GTX970 GPU
# - We measure an AUC of 0.689, which is almost identical to python-3 scikit learn results
# - We demonstrated a typical workflow in python+keras in a Jupyter notebook
# - We can easilly customize the model using the several type of layers available in Keras.  That would make our model much more accurarte and sophisticated with no additional pain in either complexity or computing performance.
# 
# [Keras](https://keras.io/) documentation is quite complete and contains several examples from linear algebra to advance deep learning techniques.
# 




# # Predicting Airline Data using a Generalized Linear Model (GLM) in Python3
# 
# In particular, we will predict the probability that a flight is late based on its departure date/time, the expected flight time and distance, the origin and destitation airports.
# 
# The core library for the dataframe part is [pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).<br>
# The core library for the machine learning part is [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).
# 
# ### Considerations
# 
# The objective of this notebook is to define a simple model offerring a point of comparison in terms of computing performances across datascience language and libraries.  In otherwords, this notebook is not for you if you are looking for the most accurate model in airline predictions.  
# 

# ## Install and Load useful libraries
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ## Load the data
# 
# - The dataset is taken from [http://stat-computing.org](http://stat-computing.org/dataexpo/2009/the-data.html).  We take the data corresponding to year 2008.
# - We restrict the dataset to the first million rows
# - We print all column names and the first 5 rows of the dataset
# 

df = pd.read_csv("2008.csv")
df.shape[0]


df = df[0:1000000]


df.columns


df[0:5]


# ## Data preparation for training
# 
# - We turn origin/destination categorical data to a "one-hot" encoding representation
# - We create a new "binary" column indicating if the flight was delayed or not.
# - We show the first 5 rows of the modified dataset
# - We split the dataset in two parts:  a training dataset and a testing dataset containing 80% and 20% of the rows, respectively.
# 

df = pd.concat([df, pd.get_dummies(df["Origin"], prefix="Origin")], axis=1);
df = pd.concat([df, pd.get_dummies(df["Dest"  ], prefix="Dest"  )], axis=1);
df = df.dropna(subset=["ArrDelay"]) 
df["IsArrDelayed" ] = (df["ArrDelay"]>0).astype(int)
df[0:5]


train = df.sample(frac=0.8)
test  = df.drop(train.index)


# ## Model building
# 
# - We define the generalized linear model using a binomial function --> Logistic regression.
# - We train the model and measure the training time --> ~15sec on an intel i7-6700K (4.0 GHz) for 800K rows 	
# - We show the model coefficients
# - We show the 10 most important variables
# 

#get the list of one hot encoding columns
OriginFeatCols = [col for col in df.columns if ("Origin_" in col)]
DestFeatCols   = [col for col in df.columns if ("Dest_"   in col)]
features = train[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
labels   = train["IsArrDelayed"]


model = LogisticRegression(C=1E5, max_iter=10000)
model.fit(features, labels)
model


coef = pd.DataFrame(data=np.transpose(model.coef_), index=features.columns, columns=["Coef"])
coef = coef.reindex( coef["Coef"].abs().sort_values(axis=0,ascending=False).index )  #order by absolute coefficient magnitude
coef[ coef["Coef"].abs()>0 ] #keep only non-null coefficients
coef[ 0:10 ] #keep only the 10 most important coefficients


# ## Model testing
# 
# - We add a model prediction column to the testing dataset
# - We show the first 10 rows of the test dataset (with the new column)
# - We show the model ROC curve
# - We measure the model Area Under Curve (AUC) to be 0.706 on the testing dataset.  
# 
# This is telling us that our model is not super accurate  (we generally assume that a model is raisonable at predicting when it has an AUC above 0.8).  But, since we are not trying to build the best possible model, but just show comparison of data science code/performance accross languages/libraries.
# If none the less you are willing to improve this result, you should try adding more feature column into the model.
# 

testFeature = test[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
test["IsArrDelayedPred"] = model.predict_proba( testFeature )[:,1]
test[0:10]


fpr, tpr, _ = roc_curve(test["IsArrDelayed"], test["IsArrDelayedPred"])
AUC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.3f)' % AUC)
plt.legend(loc=4)


AUC


# ## Key takeaways
# 
# - We built a GLM model predicting airline delay probability
# - We train it on 800K rows in ~15sec on an intel i7-6700K (4.0 GHz)
# - We measure an AUC of 0.702, which is not super accurate but reasonable
# - We demonstrated a typical workflow in python language in a Jupyter notebook
# 
# I might be biased, but I find the [pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)/[scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) documentation particularly complete and easy to read.  In addition they are thousdands of recent examples/tutorials all over the web.
# 




# # Predicting Airline Data using a Generalized Linear Model (GLM) in Tensorflow
# 
# In particular, we will predict the probability that a flight is late based on its departure date/time, the expected flight time and distance, the origin and destitation airports.
# 
# Most part of this notebooks are identical to what has been done in Airline Delay with a GLM in python3.ipynb
# The main difference is that we will use the google tensorflow framework to perform the machine learning operations instead of scikit-learn.
# 
# The core library for the dataframe part is [pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).<br>
# The core library for the machine learning part is [google tensorflow](https://www.tensorflow.org/).  This library is mostly used for deeplearning/neural-network machine learning, but it also provides many low level functions that can be used for all sorts of matrix based operations.  Generalized Linear Model is one of those.
# 
# The other main advantage of tensorflow is that it easilly allows to run computation on the GPU of the graphic card and therefore allows to speed up considerably long computations.
# 
# ### Considerations
# 
# The objective of this notebook is to define a simple model offerring a point of comparison in terms of computing performances across datascience language and libraries.  In otherwords, this notebook is not for you if you are looking for the most accurate model in airline predictions.  
# 

# ## Install and Load useful libraries
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ## Load the data (identical to python3 scikit-learn)
# 
# - The dataset is taken from [http://stat-computing.org](http://stat-computing.org/dataexpo/2009/the-data.html).  We take the data corresponding to year 2008.
# - We restrict the dataset to the first million rows
# - We print all column names and the first 5 rows of the dataset
# 

df = pd.read_csv("2008.csv")
df.shape[0]


df = df[0:1000000]


df.columns


df[0:5]


# ## Data preparation for training (identical to python3 scikit-learn)
# 
# - We turn origin/destination categorical data to a "one-hot" encoding representation
# - We create a new "binary" column indicating if the flight was delayed or not.
# - We show the first 5 rows of the modified dataset
# - We split the dataset in two parts:  a training dataset and a testing dataset containing 80% and 20% of the rows, respectively.
# 

df = pd.concat([df, pd.get_dummies(df["Origin"], prefix="Origin")], axis=1);
df = pd.concat([df, pd.get_dummies(df["Dest"  ], prefix="Dest"  )], axis=1);
df = df.dropna(subset=["ArrDelay"]) 
df["IsArrDelayed" ] = (df["ArrDelay"]>0).astype(int)
df[0:5]


train = df.sample(frac=0.8)
test  = df.drop(train.index)


# ## Model building
# 
# - We define the generalized linear model using a binomial function --> Logistic regression.
#    - The model has linear logits = (X*W)+B = (Features * Coefficients) + Bias
#    - Predictions are define as the sigmoid of the logits
#    - The Loss function is a logistic function (defined in tensorflow has sigmoid_cross_entropy_with_logits)
#    - A L2 regularization is added to mimic what is done in scikit learn
#    - We define a tensorflow graph with all these details
#    - All types of model can be implemented --> so we can go for more fancy models with no pain
# - We train the model and measure the training time --> ~10.5sec on an intel i7-6700K (4.0 GHz) with a GTX970 4GB GPU for 800K rows 	
#    - The model is trained using a minibatch strategy (that can be tune for further performance increase)
# - We show the model coefficients
# - We show the 10 most important variables
# - We show the tensorflow computation graph as seen by tensorboard
# 

#get the list of one hot encoding columns
OriginFeatCols = [col for col in df.columns if ("Origin_" in col)]
DestFeatCols   = [col for col in df.columns if ("Dest_"   in col)]
features = train[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
#features = train[["DepTime", "AirTime", "Distance"]]
labels   = train["IsArrDelayed"]

#convert it to numpy array to feed in tensorflow
featuresMatrix = features.as_matrix()
labelsMatrix   = labels  .as_matrix().reshape(-1,1)


features.shape[0]


featureSize = features.shape[1]
labelSize   = 1

training_epochs = 25
batch_size = 2500

graph = tf.Graph()
with graph.as_default():   
    # tf Graph Input
    LR = tf.placeholder(tf.float32 , name = 'LearningRate')
    X = tf.placeholder(tf.float32, [None, featureSize], name="features") # features
    Y = tf.placeholder(tf.float32, [None, labelSize], name="labels")   # training label

    with tf.name_scope("model") as scope:    
        # Set model weights
        W = tf.Variable(tf.random_normal([featureSize, labelSize],stddev=0.001), name="coefficients")
        B = tf.Variable(tf.random_normal([labelSize], stddev=0.001), name="bias")
        
        # Construct model
        logits = tf.matmul(X, W) + B                        
        with tf.name_scope("prediction") as scope:    
            P      = tf.nn.sigmoid(logits)

    with tf.name_scope("loss") as scope:
        with tf.name_scope("L2") as scope:             
           L2  = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                
        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=Y, logits=logits) ) + 1E-5*L2
            
    with tf.name_scope("optimizer") as scope:
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

    #used to make training plot on tensorboard
    with tf.name_scope("summary") as scope:
        tf.scalar_summary('cost', cost)
        tf.scalar_summary('L2', L2)
        SUMMARY = tf.merge_all_summaries()
    
        
    # Initializing the variables
    init = tf.initialize_all_variables()
      
sess = tf.Session(graph=graph)
tfTrainWriter = tf.train.SummaryWriter("./tfsummary/train", graph)        
sess.run(init)


# Training cycle
avg_cost_prev = -1
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(features.shape[0]/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = featuresMatrix[i*batch_size:(i+1)*batch_size]#features[i*batch_size:(i+1)*batch_size].as_matrix()
        batch_ys = labelsMatrix[i*batch_size:(i+1)*batch_size]#labels  [i*batch_size:(i+1)*batch_size].as_matrix().reshape(-1,1)

        #set learning rate
        learning_rate = 0.1 * pow(0.2, (epoch + float(i)/total_batch))
        
        # Fit training using batch data
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})

        # Compute average loss
        avg_cost += c / total_batch
        
        #uncomment to send tensorflow summaries to tensorboard
        #summary = sess.run([SUMMARY], feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})
        #tfTrainWriter.add_summary(summary, (epoch + float(i)/total_batch))
        
    # Display logs per epoch step
    print("Epoch: %04d, LearningRate=%.9f, cost=%.9f" % (epoch+1, learning_rate, avg_cost) )
               
    #check for early stopping
    if(avg_cost_prev>=0 and (abs(avg_cost-avg_cost_prev))<1e-4):
        print("early stopping")
        break
    else: avg_cost_prev = avg_cost        
print("Optimization Finished!")


w = sess.run(W, feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})
coef = pd.DataFrame(data=w, index=features.columns, columns=["Coef"])
coef = coef.reindex( coef["Coef"].abs().sort_values(axis=0,ascending=False).index )  #order by absolute coefficient magnitude
coef[ coef["Coef"].abs()>0 ] #keep only non-null coefficients
coef[ 0:10 ] #keep only the 10 most important coefficients


# ### Display of the computation graph as seen by tensorboard
# 
# ![image](Airline Delay with a GLM in Tensorflow.png)

# ## Model testing (identical to python3 scikit-learn)
# 
# - We add a model prediction column to the testing dataset
# - We show the first 10 rows of the test dataset (with the new column)
# - We show the model ROC curve
# - We measure the model Area Under Curve (AUC) to be 0.699 on the testing dataset.  
# 
# This is telling us that our model is not super accurate  (we generally assume that a model is raisonable at predicting when it has an AUC above 0.8).  But, since we are not trying to build the best possible model, but just show comparison of data science code/performance accross languages/libraries.
# If none the less you are willing to improve this result, you should try adding more feature column into the model.
# 

testFeature = test[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
pred = sess.run(P, feed_dict={X: testFeature.as_matrix()})
test["IsArrDelayedPred"] = pred
test[0:10]


fpr, tpr, _ = roc_curve(test["IsArrDelayed"], test["IsArrDelayedPred"])
AUC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.3f)' % AUC)
plt.legend(loc=4)


AUC


# ## Key takeaways
# 
# - We built a GLM model predicting airline delay probability in tensorflow
# - We train it on 800K rows in ~10sec on an intel i7-6700K (4.0 GHz) with a GTX970 GPU
# - We measure an AUC of 0.699, which is almost identical to python-3 scikit learn results
# - We demonstrated a typical workflow in python+tensorflow in a Jupyter notebook
# - The boost in performance with respect to scikit-learn is minimal, but this is because the logistic regression model is so simple that the majority of time is spent in transfering the data from/to the graphic card.
# - We can easilly customize the model using tensorflow graph operation with no additional pain in either complexity or computing performance.
# 
# 
# Tensorflow documentation is quite complete and contains several examples from linear algebra to advance deep learning techniques.
# 




