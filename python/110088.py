# # Text Analysis
# 
# This notebook focuses on analysing the Text we are using for our model. This includes counting the total words, counting the unique words, counting total number of posts, and the total number of users, and determining the number of words removed by Gensim
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

model_name = "model6"


# ### Load our Data
# 
# After selecting our model, we next load the data
# 

posts = load_object('objects/',model_name+"-posts")
df    = load_object('objects/',model_name+"-df")


# ### Posts Analysis
# 
# We first analyze the posts, and their users
# 

num_posts = len(df['cleantext'])


#get the number of users (minus [deleted])
user_list= df["author"].tolist()
user_dict = {}
for user in user_list:
    if user in user_dict.keys() and user != "[deleted]":
        user_dict[user] =1+user_dict[user]
    else:
        user_dict[user] =1
num_users = len(list(user_dict.keys()))


num_posts


num_users


# ### Word/Phrase Analysis
# 
# We now analyze the words and phrases used by our model
# 

plain_words = list(df['cleantext'].apply(lambda x: x.split()))


total_phrases =0
for post in posts:
    for phrase in post:
        total_phrases +=1


total_words =0
for post in plain_words:
    for word in post:
        total_words +=1


phrase_dict = {}
for post in posts:
    for phrase in post:
        if phrase in phrase_dict.keys():
            phrase_dict[phrase] =1+phrase_dict[phrase]
        else:
            phrase_dict[phrase] =1


word_dict= {}
for post in plain_words:
    for word in post:
        if word in word_dict.keys():
            word_dict[word] =1+word_dict[word]
        else:
            word_dict[word] =1


# Total words in the corpus
total_words


# Total phrases in the corpus
total_phrases


# Total vocabulary of words
len(list(word_dict))


# Total vocabulary of phrases
len(list(phrase_dict))


phrases = list(phrase_dict.keys())
phrase_freq_count            = 0
filtered_phrase_freq_count   = 0
phrase_unique_count          = 0
filtered_phrase_unique_count = 0
for phrase in phrases:
    count = phrase_dict[phrase]
    phrase_freq_count            += count
    filtered_phrase_freq_count   += count if count >= 10 else 0
    phrase_unique_count          += 1
    filtered_phrase_unique_count += 1 if count >= 10 else 0


words = list(word_dict.keys())
word_freq_count            = 0
filtered_word_freq_count   = 0
word_unique_count          = 0
filtered_word_unique_count = 0
for word in words:
    count = word_dict[word]
    word_freq_count            += count
    filtered_word_freq_count   += count if count >= 10 else 0
    word_unique_count          += 1
    filtered_word_unique_count += 1 if count >= 10 else 0


# Total number of tokens, including phrases and words
phrase_freq_count


# Total number of words, not including phrases
word_freq_count


# Number of words removed by including them in phrases
word_freq_count-phrase_freq_count


# Total number of tokens after filtering, including phrases and words
filtered_phrase_freq_count


# Total number of tokens after filtering, including just words
filtered_word_freq_count


# Check that unique count was calculated correctlly
phrase_unique_count == len(phrase_dict) and word_unique_count == len(word_dict)


# the size of the vocabulary after filtering phrases
filtered_phrase_unique_count


# the number of unique tokens removed by filtering
phrase_unique_count - filtered_phrase_unique_count


# The percent of total tokens removed
str((phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count*100) + str("%")


# The percent of total tokens preserved
str(100 -100*(phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count) + str("%")


# ### Check model
# 
# We now will analyze the model, to ensure that it has a word count that coresponds to the posts word count.
# 

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')


vocab_list = sorted(list(model.wv.vocab))


# Ensure model has correct number of unique words
len(vocab_list)==filtered_phrase_unique_count


model_freq_count = 0
for word in vocab_list:
    model_freq_count += model.wv.vocab[word].count


# Ensure that the total count of the model's words is the total count of the filtered words
model_freq_count==filtered_phrase_freq_count


# # Association analysis
# 
# This notebook focuses on creating and analysing assocation rules found in posts.
# 
# We first load all the libraries we may use throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects

from orangecontrib.associate.fpgrowth import *


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

# Set the model we are going to be analyzing
model_name = "example_model"


# ### Prepare data
# 
# We now load and process the data we will need for the rest of this project
# 

df = load_object('objects/', model_name + '-df')

scores = list(df['score'])
num_comments_list = list(df['num_comments'])

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
len(PostsByFeatures)
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

# Initialize a word clustering to use
num_word_clusters = 100
kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# Ensure consistency
len(WordsByFeatures) == ClustersByWords.shape[1]

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)


PostsByClusters


# ### Create Association rules
# 
# Now that our data has been prepared, we create our itemsets, and then analyze them by creating association rules.
# 

itemsets = dict(frequent_itemsets(PostsByClusters > 0, .40))


assoc_rules = association_rules(itemsets,0.8)


rules = [(P, Q, supp, conf, conf/(itemsets[P]/PostsByClusters.shape[0]))
         for P, Q, supp, conf in association_rules(itemsets, .95)]


for lhs, rhs, support, confidence,lift in rules:
    print(", ".join([str(i) for i in lhs]), "-->",", ".join([str(i) for i in rhs]), "support: ",
          support, " confidence: ",confidence, "lift: ", lift)


len(rules)


rule_clusters =[]
for i in range(100):
    for lhs, rhs, support, confidence,lift in rules:
        if (i in lhs) or (i in rhs): 
            rule_clusters.append(i)
            break


rule_clusters


len(rule_clusters)


# ### Save results
# 
# After running the lengthy computation of finding the association rules, we save the results so we will not need to run the same computation again in the future.
# 

save_object(rules,'objects/',model_name+'-assoc_rules')
save_object(itemsets,'objects/',model_name+'-itemset')


# # Correlation Analysis
# 
# In this notebook we will attempt to analyze the correlations between different clusters in the posts.
# 
# We first load all necesary libraries
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects

from orangecontrib.associate.fpgrowth import *


# Set the model we are going to be analyzing
model_name = "PTSD_model"


# ### Make Correlation Matrix
# 

# Initialize a word clustering to use
num_word_clusters = 100
# Initialize the threshold to count a correlation
correlation_threshold = 0.65


df = load_object('objects/', model_name + '-df')

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)


X = np.array(PostsByClusters.todense())


cluster_df = pd.DataFrame(data = X)


correlations = cluster_df.corr().values


# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)


correlations_list = []
for i in range(len(correlations)):
    for j in range(i+1,len(correlations[0])):
        corr_val = correlations[i][j]
        if corr_val > correlation_threshold:
            correlations_list.append([i,j,corr_val,clusters[i]["word_list"][:5],clusters[j]["word_list"][:5]])


len(correlations_list)


correlations_list


import os
directories = ['correlation-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


import csv
heading = ["cluster 1 number", "cluster 2 number", "correlation values","cluster 1","cluster 2"]
with open("correlation-analysis/"+model_name+"-correlations.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(heading)
    [writer.writerow(r) for r in correlations_list]


# # Post Cluster Analysis
# 
# This notebook focuses on analysing the posts clusters for a model. We will do this by first generating 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects


# #### Setup directories
# 
# If this is the first time doing this analysis, 
# we first will set up all the directories we need
# to save and load the models we will be using
# 

import os
directories = ['post-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

# Set the model we are going to be analyzing
model_name = "model6"


# ### Prepare data
# 
# We now load and process the data we will need for the rest of this project
# 

df = load_object('objects/', model_name + '-df')

scores = list(df['score'])
num_comments_list = list(df['num_comments'])


# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
len(PostsByFeatures)


model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

# Initialize a word clustering to use
num_word_clusters = 100
kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# Ensure consistency
len(WordsByFeatures) == ClustersByWords.shape[1]


# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)


PostsByClusters = PostsByClusters.todense() * 1.0


row_min = PostsByFeatures.min(axis = 1)
row_max = PostsByFeatures.max(axis = 1)
row_diff_normed = (row_max - row_min == 0) + (row_max - row_min)
PostsByFeaturesNormed = (PostsByFeatures - row_min) / row_diff_normed

row_min = PostsByClusters.min(axis = 1)
row_max = PostsByClusters.max(axis = 1)
row_diff_normed = (row_max - row_min == 0) + (row_max - row_min)
PostsByClustersNormed = (PostsByClusters - row_min) / row_diff_normed


# ### Make Correlation matrix
# 

a = np.array(PostsByClusters)


len(a[0])


posts_df = pd.DataFrame(a)


rows, columns = posts_df.shape


import scipy
correlation_table =[]
for i in range(columns): # rows are the number of rows in the matrix. 
    correlation_row = []
    for j in range(columns):
        r = scipy.stats.pearsonr(a[:,i], a[:,j])
        correlation_row.append(r[0])
    correlation_table.append(correlation_row)


scipy.stats.pearsonr(a[:,19], a[:,18])


len(a[:,19])


# Print Correlation table
import csv
header = ["Cluster "+ str(i) for i in range(1,columns+1)]
with open('cluster-analysis/' + "correlation-"+model_name + "-" + str(num_word_clusters) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([""]+header)
    for i in range(len(correlation_table)):
        writer.writerow([header[i]]+correlation_table[i])


# ### Generate Post Clusters
# 
# We now will generate post clusters, and then save them in a format conducive to analysis.
# 

num_posts_clusters =10
matricies = [PostsByFeatures, PostsByClusters, PostsByFeaturesNormed, PostsByClustersNormed]
names     = ["byFeatures", "byClusters", "byFeatures-Normed", "byClusters-Normed"]
mat_names = list(zip(matricies, names))
post_dfs  = []


for mat,name in mat_names:
    #initialize kmeans model
    kmeans = KMeans(n_clusters = num_posts_clusters, random_state = 42).fit(mat)
    # Save the clusters directory
    save_object(kmeans, 'clusters/', model_name + "-posts-" + name + "-" + str(num_posts_clusters))
    del kmeans


# Setup the header for the CSV files
header = ['total_posts', 'score_mean', 'score_median', 'score_range', 'comments_mean', 'comments_median', 'comments_range']
# Loop over all matricies
for mat,name in mat_names:
    # Load Clusters
    kmeans= load_object('clusters/', model_name + "-posts-" + name + "-" + str(num_posts_clusters))
    # Generate Post_clusters
    post_clusters = make_post_clusters(kmeans,mat,scores,num_comments_list)
    temp_header =header+list(map(lambda x:"element "+str(x),range(1,mat.shape[1]+1)))
    temp_table = list(map(lambda x: list(map(lambda y: x[1][y],header))+
                          list(map(lambda z: z[0],post_clusters[x[0]]['center'])),enumerate(post_clusters)))
    #post_dfs.append(pd.DataFrame.from_records(temp_table,columns =temp_header))

    import csv
    with open('post-analysis/' + model_name + '-' + str(num_posts_clusters) + '-' + name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(temp_header)
        [writer.writerow(r) for r in temp_table]


# # Word2Vec Analysis
# 
# This notebook focuses on analysing the word2vec model we are using. This will mostly involve testing the functions given by gensim.
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

model_name = "model6"


# ### Examine word similarities
# 
# We first examine word similarities
# 

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')


model.most_similar(positive=["heartbreak"])


model.most_similar(positive=["pills"])


model.most_similar(positive=["knife"])


model.most_similar(positive=["kitten"])


model.most_similar(positive=["puppy"])


# ### Examine word relationships
# 
# We now examine information contained in word vectors relative locations
# 

model.most_similar(positive=["abusive","words"],negative =["physical"])


model.most_similar(positive=["suicide","self"])


model.most_similar(positive=["family","obligation"],negative = ["love"])


model.most_similar(positive=["father","woman"],negative=["man"])


model.most_similar(positive=["kitten","dog"],negative=["cat"])


model.most_similar(positive=["i"])


# # Text Analysis
# 
# This notebook focuses on analysing the Text we are using for our model. This includes counting the total words, counting the unique words, counting total number of posts, and the total number of users, and determining the number of words removed by Gensim
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

model_name = "PTSD_model"


# ### Load our Data
# 
# After selecting our model, we next load the data
# 

posts = load_object('objects/',model_name+"-posts")
df    = load_object('objects/',model_name+"-df")


# ### Posts Analysis
# 
# We first analyze the posts, and their users
# 

num_posts = len(df['cleantext'])


#get the number of users (minus [deleted])
user_list= df["author"].tolist()
user_dict = {}
for user in user_list:
    if user in user_dict.keys() and user != "[deleted]":
        user_dict[user] =1+user_dict[user]
    else:
        user_dict[user] =1
num_users = len(list(user_dict.keys()))


num_posts


num_users


# ### Word/Phrase Analysis
# 
# We now analyze the words and phrases used by our model
# 

plain_words = list(df['cleantext'].apply(lambda x: x.split()))


total_phrases =0
for post in posts:
    for phrase in post:
        total_phrases +=1


total_words =0
for post in plain_words:
    for word in post:
        total_words +=1


phrase_dict = {}
for post in posts:
    for phrase in post:
        if phrase in phrase_dict.keys():
            phrase_dict[phrase] =1+phrase_dict[phrase]
        else:
            phrase_dict[phrase] =1


word_dict= {}
for post in plain_words:
    for word in post:
        if word in word_dict.keys():
            word_dict[word] =1+word_dict[word]
        else:
            word_dict[word] =1


# Total words in the corpus
total_words


# Total phrases in the corpus
total_phrases


# Total vocabulary of words
len(list(word_dict))


# Total vocabulary of phrases
len(list(phrase_dict))


phrases = list(phrase_dict.keys())
phrase_freq_count            = 0
filtered_phrase_freq_count   = 0
phrase_unique_count          = 0
filtered_phrase_unique_count = 0
for phrase in phrases:
    count = phrase_dict[phrase]
    phrase_freq_count            += count
    filtered_phrase_freq_count   += count if count >= 10 else 0
    phrase_unique_count          += 1
    filtered_phrase_unique_count += 1 if count >= 10 else 0


words = list(word_dict.keys())
word_freq_count            = 0
filtered_word_freq_count   = 0
word_unique_count          = 0
filtered_word_unique_count = 0
for word in words:
    count = word_dict[word]
    word_freq_count            += count
    filtered_word_freq_count   += count if count >= 10 else 0
    word_unique_count          += 1
    filtered_word_unique_count += 1 if count >= 10 else 0


# Total number of tokens, including phrases and words
phrase_freq_count


# Total number of words, not including phrases
word_freq_count


# Number of words removed by including them in phrases
word_freq_count-phrase_freq_count


# Total number of tokens after filtering, including phrases and words
filtered_phrase_freq_count


# Total number of tokens after filtering, including just words
filtered_word_freq_count


# Get the unique count for phrases
phrase_unique_count


# Check that unique count was calculated correctlly
phrase_unique_count == len(phrase_dict) and word_unique_count == len(word_dict)


# the size of the vocabulary after filtering phrases
filtered_phrase_unique_count


# the number of unique tokens removed by filtering
phrase_unique_count - filtered_phrase_unique_count


# The percent of total tokens removed
str((phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count*100) + str("%")


# The percent of total tokens preserved
str(100 -100*(phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count) + str("%")


# ### Check model
# 
# We now will analyze the model, to ensure that it has a word count that coresponds to the posts word count.
# 

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')


vocab_list = sorted(list(model.wv.vocab))


# Ensure model has correct number of unique words
len(vocab_list)==filtered_phrase_unique_count


model_freq_count = 0
for word in vocab_list:
    model_freq_count += model.wv.vocab[word].count


# Ensure that the total count of the model's words is the total count of the filtered words
model_freq_count==filtered_phrase_freq_count


# # Cluster Analysis
# 
# This notebook focuses on analysing the word clusters for a model. This includes visualizing fit of the clusters, formating them for manual inspection, and visualizing them using Multi Dimensional Scaling (MDS).
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object, make_clustering_objects


# #### Setup directories
# 
# If this is the first time doing this analysis, 
# we first will set up all the directories we need
# to save and load the models we will be using
# 

import os
directories = ['cluster-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

# Set the model we are going to be analyzing
model_name = "example_model"


# ### Measure fit
# 
# Now that we have initialized all we need for our analysis, we can procceed to examine the fit of each clustering.
# 

# Load the fit and test point values
fit = load_object('objects/', model_name + "-words" + "-fit")
test_points = load_object('objects/', model_name + "-words" + "-test_points")


# Plot the fit for each size
plt.plot(test_points, fit, 'ro')
plt.axis([0, 400, 0, np.ceil(fit[0] + (1/10)*fit[0])])
plt.show()


# ### Format for inspection
# 
# After measuring the fit of each clustering, we can decide the number of clusters to use, and further focus on this clustering. To better examine this clustering, we convert the clustering into an readable csv here.
# 

# Set the number of clusters to analyze
num_clusters = 100 


# load the models
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')
kmeans = load_object('clusters/', model_name + "-words-cluster_model-" + str(num_clusters))
WordsByFeatures = load_object('matricies/', model_name + '-' + 'WordsByFeatures')


vocab_list = sorted(list(model.wv.vocab))


clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)


# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)


# Set the number of words to display. The table with contain the top size_words_list words
size_words_list = 100
table = []
for i in range(len(clusters)):
    row = []
    row.append("cluster " + str(i+1))
    row.append(clusters[i]["total_freq"])
    row.append(clusters[i]["unique_words"])
    for j in range(size_words_list):
        try:
            row.append(clusters[i]["word_list"][j])
        except:
            break
    table.append(row)


import csv
with open('cluster-analysis/' + model_name + "-" + str(num_clusters) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    [writer.writerow(r) for r in table]


# #### Display Clusters Using MDS
# 
# Produce a visualization of our clusters in a low dimensional space
# 

# Fit the model to the clusters
from sklearn.manifold import MDS
mds = MDS().fit(kmeans.cluster_centers_)


# Get the embeddings
embedding = mds.embedding_.tolist()
x = list(map(lambda x:x[0], embedding))
y = list(map(lambda x:x[1], embedding))


top_words= list(map(lambda x: x[0][0], map(lambda x: x["word_list"], clusters)))


# Plot the Graph with top words
plt.figure(figsize = (20, 10))
plt.plot(x, y, 'bo')
for i in range(len(top_words)):
    plt.annotate(top_words[i], (x[i], y[i]))
plt.show()


# # Correlation Analysis
# 
# In this notebook we will attempt to analyze the correlations between different clusters in the posts.
# 
# We first load all necesary libraries
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects

from orangecontrib.associate.fpgrowth import *


# Set the model we are going to be analyzing
model_name = "example_model"


# ### Make Correlation Matrix
# 

# Initialize a word clustering to use
num_word_clusters     = 100
# Initialize the threshold to count a correlation
correlation_threshold = 0.65


df = load_object('objects/', model_name + '-df')

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)


X = np.array(PostsByClusters.todense())


cluster_df = pd.DataFrame(data = X)


correlations = cluster_df.corr().values


# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)


correlations_list = []
for i in range(len(correlations)):
    for j in range(i+1,len(correlations[0])):
        corr_val = correlations[i][j]
        if corr_val > correlation_threshold:
            correlations_list.append([i+1,j+1,corr_val,clusters[i]["word_list"][:5],clusters[j]["word_list"][:5]])


len(correlations_list)


correlations_list


import os
directories = ['correlation-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


import csv
heading = ["cluster 1 number", "cluster 2 number", "correlation values","cluster 1","cluster 2"]
with open("correlation-analysis/"+model_name+"-correlations.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(heading)
    [writer.writerow(r) for r in correlations_list]


# # Initialize Models
# This notebook will walk you through building and saving the most basic 
# models we used for analyzing our text data.
# 
# We first import the libraries and utility files we are going to be using.
# 

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import read_df, remove_links, clean_sentence, save_object, load_object


# #### Setup directories
# 
# If this is the first time doing this analysis, 
# we first will set up all the directories we need
# to save and load the models we will be using
# 

import os
directories = ['objects', 'models', 'clusters', 'matricies']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# #### Name Model
# 
# Before begining the rest of our project, we select a name for our model.
# This name will be used to save and load the files for this model
# 

model_name = "PTSD_model4"


# #### Parse and Clean Data
# 
# We first parse and clean our data. Our data is assumed to be in csv format, 
# in a directory labeled 'data'.
# 

# Get the data from the csv
df = read_df('data')


# Do an inspection of our data to ensure nothing went wrong
df.info()


df.head()


# Clean the text in the dataframe
df = df.replace(np.nan, '', regex = True)
df = df.replace("\[deleted\]", '', regex = True)
df["rawtext"] = df["title"] + " " + df["selftext"]
df["cleantext"] = df["rawtext"].apply(remove_links).apply(clean_sentence)


# Check that the cleaning was successful
df.info()


df.head()


# ### Phrase Analysis
# 
# After parsing and cleaning the data we run the gensim phraser
# tool on our text data to join phrases like "new york city" 
# together to form the word "new_york_city"
# 

# Get a stream of tokens
posts = df["cleantext"].apply(lambda str: str.split()).tolist()


# Train a phraseDetector to join two word phrases together
two_word_phrases = gensim.models.Phrases(posts)
two_word_phraser = gensim.models.phrases.Phraser(two_word_phrases)


# Train a phraseDetector to join three word phrases together
three_word_phrases = gensim.models.Phrases(two_word_phraser[posts])
three_word_phraser = gensim.models.phrases.Phraser(three_word_phrases)
posts = list(three_word_phraser[two_word_phraser[posts]])


# Update Data frame
df["phrasetext"] = df["cleantext"].apply(lambda str: " ".join(three_word_phraser[two_word_phraser[str.split()]]))


# Ensure posts contain same number of elements
len(posts) == len(df)


# Check that the dataframe was updated correctly
for i in range(len(posts)):
    if not " ".join(posts[i]) == list(df["phrasetext"])[i]:
        print("index :" + str(i) + " is incorrect")


# ### Data Saving
# 
# After cleaning and parsing all of our data, we can now
# save it, so that we can analysis it later without having
# to go through lengthy computations
# 

save_object(posts, 'objects/', model_name + "-posts")
save_object(df, 'objects/', model_name + "-df")


# ### Initialize Word2Vec Model
# 
# After all of our data has been parsed and saved, 
# we generate our Word2Vec Model
# 

# Set the minimum word count to 10. This removes all words that appear less than 10 times in the data
minimum_word_count = 10
# Set skip gram to 1. This sets gensim to use the skip gram model instead of the Continuous Bag of Words model
skip_gram = 1
# Set Hidden layer size to 300.
hidden_layer_size = 300
# Set the window size to 5. 
window_size = 5
# Set hierarchical softmax to 1. This sets gensim to use hierarchical softmax
hierarchical_softmax = 1
# Set negative sampling to 20. This is good for relatively small data sets, but becomes harder for larger datasets
negative_sampling = 20
# number of iterations to run default 5
iterations =80


# Build the model
model = gensim.models.Word2Vec(posts, min_count = minimum_word_count, sg = skip_gram, size = hidden_layer_size,
                               window = window_size, hs = hierarchical_softmax, negative = negative_sampling,iter=iterations)


# ### Basic Model test
# 
# After generating our model, we run some basic tests
# to ensure that it has captured some semantic information results
# 

model.most_similar(positive = ["kitten"])


model.most_similar(positive = ["her"])


model.most_similar(positive = ["my"])


model.most_similar(positive = ["father", "woman"], negative = ["man"])


model.most_similar(positive = ["family", "obligation"], negative = ["love"])


# ### Save Model
# 
# After generating our model, and runing some basic tests,
# we now save it so that we can analysis it later without having
# to go through lengthy computations. We also delete and then reload
# the model, as an example of how to do so.
# 

model.save('models/' + model_name + '.model')
del model


model = gensim.models.Word2Vec.load('models/' + model_name + '.model')


# ### Generate Matricies
# 
# After generating our Word2Vec Model, we generate 
# a collection of matricies that will be useful for
# analysis. This includes a Words By feature matrix,
# and a Post By Words Matrix. Note, we will use camelCase 
# for matrix names, and only matrix names
# 

# Initialize the list of words used
vocab_list = sorted(list(model.wv.vocab))


# Extract the word vectors
vecs = []
for word in vocab_list:
    vecs.append(model.wv[word].tolist())


# change array format into numpy array
WordsByFeatures = np.array(vecs)


from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s:s), lst))), min_df = 0)


# Make Posts By Words Matrix
PostsByWords = countvec.fit_transform(posts)


# ### Basic Matrix tests
# 
# After generating our matricies, we run some basic tests
# to ensure that they seem resaonable later without having
# to go through lengthy computations
# 

# Check that PostsByWords is the number of Posts by the number of words
PostsByWords.shape[0] == len(posts)


# check that the number of words is consistant for all matricies
PostsByWords.shape[1] == len(WordsByFeatures)


# ### Save Matricies
# 
# After generating our matricies, we save them so we can 
# analyze them later without having to go through lengthy
# computations.
# 

save_object(PostsByWords,'matricies/', model_name + "-PostsByWords")
save_object(WordsByFeatures,'matricies/', model_name + "-WordsByFeatures")


# ### Generate Word Clusters
# 
# Now that we have generated and saved our matricies,
# we will proceed to generate word clusters using 
# kmeans clustering, and save them for later analysis.
# 

from sklearn.cluster import KMeans
# get the fit for different values of K
test_points = [12] + list(range(25, 401, 25))
fit = []
for point in test_points:
    kmeans = KMeans(n_clusters = point, random_state = 42).fit(WordsByFeatures)
    save_object(kmeans, 'clusters/', model_name + "-words-cluster_model-" + str(point))
    fit.append(kmeans.inertia_)


save_object(fit, 'objects/', model_name + "-words" + "-fit")
save_object(test_points, 'objects/', model_name + "-words" + "-test_points")
del fit
del test_points


# # Cluster Analysis
# 
# This notebook focuses on analysing the word clusters for a model. This includes visualizing fit of the clusters, formating them for manual inspection, and visualizing them using Multi Dimensional Scaling (MDS).
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object, make_clustering_objects


# #### Setup directories
# 
# If this is the first time doing this analysis, 
# we first will set up all the directories we need
# to save and load the models we will be using
# 

import os
directories = ['cluster-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

# Set the model we are going to be analyzing
model_name = "PTSD_model"


# ### Measure fit
# 
# Now that we have initialized all we need for our analysis, we can procceed to examine the fit of each clustering.
# 

# Load the fit and test point values
fit = load_object('objects/', model_name + "-words" + "-fit")
test_points = load_object('objects/', model_name + "-words" + "-test_points")


# Plot the fit for each size
plt.plot(test_points, fit, 'ro')
plt.axis([0, 400, 0, np.ceil(fit[0] + (1/10)*fit[0])])
plt.show()


# ### Format for inspection
# 
# After measuring the fit of each clustering, we can decide the number of clusters to use, and further focus on this clustering. To better examine this clustering, we convert the clustering into an readable csv here.
# 

# Set the number of clusters to analyze
num_clusters = 100


# load the models
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')
kmeans = load_object('clusters/', model_name + "-words-cluster_model-" + str(num_clusters))
WordsByFeatures = load_object('matricies/', model_name + '-' + 'WordsByFeatures')


vocab_list = sorted(list(model.wv.vocab))


clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)


# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)


# Set the number of words to display. The table with contain the top size_words_list words
size_words_list = 100
table = []
for i in range(len(clusters)):
    row = []
    row.append("cluster " + str(i+1))
    row.append(clusters[i]["total_freq"])
    row.append(clusters[i]["unique_words"])
    for j in range(size_words_list):
        try:
            row.append(clusters[i]["word_list"][j])
        except:
            break
    table.append(row)


import csv
with open('cluster-analysis/' + model_name + "-" + str(num_clusters) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    [writer.writerow(r) for r in table]


# #### Display Clusters Using MDS
# 
# Produce a visualization of our clusters in a low dimensional space
# 

# Fit the model to the clusters
from sklearn.manifold import MDS
mds = MDS().fit(kmeans.cluster_centers_)


# Get the embeddings
embedding = mds.embedding_.tolist()
x = list(map(lambda x:x[0], embedding))
y = list(map(lambda x:x[1], embedding))


top_words= list(map(lambda x: x[0][0], map(lambda x: x["word_list"], clusters)))


# Plot the Graph with top words
plt.figure(figsize = (20, 10))
plt.plot(x, y, 'bo')
for i in range(len(top_words)):
    plt.annotate(top_words[i], (x[i], y[i]))
plt.show()


# # Association analysis
# 
# This notebook focuses on creating and analysing assocation rules found in posts.
# 
# We first load all the libraries we may use throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects

from orangecontrib.associate.fpgrowth import *


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

# Set the model we are going to be analyzing
model_name = "PTSD_model"


# ### Prepare data
# 
# We now load and process the data we will need for the rest of this project
# 

# Initialize a word clustering to use
num_word_clusters = 100


df = load_object('objects/', model_name + '-df')

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)


PostsByClusters





sorted_clusters = sorted(list(zip(clusters,range(len(clusters)))),key = (lambda x : x[0]['total_freq']))

large_indicies = list(map(lambda x: x[1],sorted_clusters[-20:]))

sorted_large_indicies = sorted(large_indicies, reverse =True)

X = np.array(PostsByClusters.todense())
index_mapping = list(range(100))

for index in sorted_large_indicies:
    X = np.delete(X,index,1)
    del index_mapping[index]


# ### Generate Rules
# 
# Test rule generation on a subset of the data, before moving on to the entirety of the data
# 

assoc_confidence = 50
itemset_support  = 10


X_test = X[:700]


X_test


len(X_test)


itemsets = dict(frequent_itemsets(X_test > 0, itemset_support/
100))
assoc_rules = association_rules(itemsets, assoc_confidence/100)
rules = [(P, Q, supp, conf, conf/(itemsets[P]/X_test.shape[0]))
             for P, Q, supp, conf in assoc_rules
             if len(Q) == 1 and len(P)==1]


rules


# ### Load results
# 
# We now load our results from running the association miner so we can analyze them
# 

rules    = load_object('association_rules/',model_name+'-assoc_rules-'+str(itemset_support)+
                       '-'+str(assoc_confidence)+'-'+str(num_word_clusters))
itemsets = load_object('itemsets/',model_name+'-itemset-'+str(itemset_support)+'-'+str(num_word_clusters))


# ### Analyze results
# 
# after loading our results we analyze them
# 

len(rules)


len(itemsets)


len(rules)/len(itemsets)


rule_clusters =[]
for i in range(num_word_clusters):
    for lhs, rhs, support, confidence,lift in rules:
        if (i in lhs) or (i in rhs): 
            rule_clusters.append(i)
            break


len(rule_clusters)


rules.sort(key = lambda x : x[4],reverse = True)


filtered_rules = list(filter(lambda x: len(x[0])==1 and len(x[1])==1,rules ))


# load the models
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')
kmeans = load_object('clusters/', model_name + "-words-cluster_model-" + str(num_word_clusters))
WordsByFeatures = load_object('matricies/', model_name + '-' + 'WordsByFeatures')


vocab_list = sorted(list(model.wv.vocab))


clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)


# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)


len(filtered_rules)


import csv
top_num = min(10000,len(filtered_rules))
header = ["lhs","rhs","support","confidence","lift"]
with open('association-analysis/'+ model_name + "-filtered-lift-supp"+str(itemset_support) +
          "-conf-"+str(assoc_confidence)+'-'+ str(top_num) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for i in range(top_num):
        rule = filtered_rules[i]
        lhs_top = clusters[index_mapping[next(iter(rule[0]))]]["word_list"][:5]
        rhs_top = clusters[index_mapping[next(iter(rule[1]))]]["word_list"][:5]
        writer.writerow([lhs_top,rhs_top ,rule[2],rule[3],rule[4]])





# # Word2Vec Analysis
# 
# This notebook focuses on analysing the word2vec model we are using. This will mostly involve testing the functions given by gensim.
# 
# We first import the libaries we will need throughout the project
# 

#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object


# ### Set model name
# 
# Before begining the rest of this project, we select a name for our model. This name will be used to save and load the files for this model
# 

model_name = "PTSD_model"


# ### Examine word similarities
# 
# We first examine word similarities
# 

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')


model.most_similar(positive=["abuse"])


model.most_similar(positive=["military"])


model.most_similar(positive=["medication"])


model.most_similar(positive=["victim"])


model.most_similar(positive=["ptsd"])


# ### Examine word relationships
# 
# We now examine information contained in word vectors relative locations
# 

model.most_similar(positive=["his"])


model.most_similar(positive=["suicide","self"])


model.most_similar(positive=["family","obligation"],negative = ["love"])


model.most_similar(positive=["brother","girl"],negative = ["boy"])


model.most_similar(positive=["father","woman"],negative=["man"])


model.most_similar(positive=["kitten","dog"],negative=["cat"])


model.most_similar(positive=["veteran","trauma"])


model.most_similar(positive=["law","family"],negative=["love"])


# # Initialize Models
# This notebook will walk you through building and saving the most basic 
# models we used for analyzing our text data.
# 
# We first import the libraries and utility files we are going to be using.
# 

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import read_df, remove_links, clean_sentence, save_object, load_object


# #### Setup directories
# 
# If this is the first time doing this analysis, 
# we first will set up all the directories we need
# to save and load the models we will be using
# 

import os
directories = ['objects', 'models', 'clusters', 'matricies']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# #### Name Model
# 
# Before begining the rest of our project, we select a name for our model.
# This name will be used to save and load the files for this model
# 

model_name = "example_model"


# #### Parse and Clean Data
# 
# We first parse and clean our data. Our data is assumed to be in csv format, 
# in a directory labeled 'data'.
# 

# Get the data from the csv
df = read_df('data',extension = "/*.csv")


# Do an inspection of our data to ensure nothing went wrong
df.info()


df.head()


# Clean the text in the dataframe
df = df.replace(np.nan, '', regex = True)
df = df.replace("\[deleted\]", '', regex = True)
df["rawtext"] = df["title"] + " " + df["selftext"]
df["cleantext"] = df["rawtext"].apply(remove_links).apply(clean_sentence)


# Check that the cleaning was successful
df.info()


df.head()


# ### Phrase Analysis
# 
# After parsing and cleaning the data we run the gensim phraser
# tool on our text data to join phrases like "new york city" 
# together to form the word "new_york_city"
# 

# Get a stream of tokens
posts = df["cleantext"].apply(lambda str: str.split()).tolist()


# Train a phraseDetector to join two word phrases together
two_word_phrases = gensim.models.Phrases(posts)
two_word_phraser = gensim.models.phrases.Phraser(two_word_phrases)


# Train a phraseDetector to join three word phrases together
three_word_phrases = gensim.models.Phrases(two_word_phraser[posts])
three_word_phraser = gensim.models.phrases.Phraser(three_word_phrases)
posts = list(three_word_phraser[two_word_phraser[posts]])


# Update Data frame
df["phrasetext"] = df["cleantext"].apply(lambda str: " ".join(three_word_phraser[two_word_phraser[str.split()]]))


# Ensure posts contain same number of elements
len(posts) == len(df)


# Check that the dataframe was updated correctly
for i in range(len(posts)):
    if not " ".join(posts[i]) == list(df["phrasetext"])[i]:
        print("index :" + str(i) + " is incorrect")


# ### Data Saving
# 
# After cleaning and parsing all of our data, we can now
# save it, so that we can analysis it later without having
# to go through lengthy computations
# 

save_object(posts, 'objects/', model_name + "-posts")
save_object(df, 'objects/', model_name + "-df")


# ### Initialize Word2Vec Model
# 
# After all of our data has been parsed and saved, 
# we generate our Word2Vec Model
# 

# Set the minimum word count to 10. This removes all words that appear less than 10 times in the data
minimum_word_count = 10
# Set skip gram to 1. This sets gensim to use the skip gram model instead of the Continuous Bag of Words model
skip_gram = 1
# Set Hidden layer size to 300.
hidden_layer_size = 300
# Set the window size to 5. 
window_size = 5
# Set hierarchical softmax to 1. This sets gensim to use hierarchical softmax
hierarchical_softmax = 1
# Set negative sampling to 20. This is good for relatively small data sets, but becomes harder for larger datasets
negative_sampling = 20


# Build the model
model = gensim.models.Word2Vec(posts, min_count = minimum_word_count, sg = skip_gram, size = hidden_layer_size,
                               window = window_size, hs = hierarchical_softmax, negative = negative_sampling)


# ### Basic Model test
# 
# After generating our model, we run some basic tests
# to ensure that it has captured some semantic information results
# 

model.most_similar(positive = ["kitten"])


model.most_similar(positive = ["father", "woman"], negative = ["man"])


model.most_similar(positive = ["family", "obligation"], negative = ["love"])


# ### Save Model
# 
# After generating our model, and runing some basic tests,
# we now save it so that we can analysis it later without having
# to go through lengthy computations. We also delete and then reload
# the model, as an example of how to do so.
# 

model.save('models/' + model_name + '.model')
del model


model = gensim.models.Word2Vec.load('models/' + model_name + '.model')


# ### Generate Matricies
# 
# After generating our Word2Vec Model, we generate 
# a collection of matricies that will be useful for
# analysis. This includes a Words By feature matrix,
# and a Post By Words Matrix. Note, we will use camelCase 
# for matrix names, and only matrix names
# 

# Initialize the list of words used
vocab_list = sorted(list(model.wv.vocab))


# Extract the word vectors
vecs = []
for word in vocab_list:
    vecs.append(model.wv[word].tolist())


# change array format into numpy array
WordsByFeatures = np.array(vecs)


from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s:s), lst))), min_df = 0)


# Make Posts By Words Matrix
PostsByWords = countvec.fit_transform(posts)


# ### Basic Matrix tests
# 
# After generating our matricies, we run some basic tests
# to ensure that they seem resaonable later without having
# to go through lengthy computations
# 

# Check that PostsByWords is the number of Posts by the number of words
PostsByWords.shape[0] == len(posts)


# check that the number of words is consistant for all matricies
PostsByWords.shape[1] == len(WordsByFeatures)


# ### Save Matricies
# 
# After generating our matricies, we save them so we can 
# analyze them later without having to go through lengthy
# computations.
# 

save_object(PostsByWords,'matricies/', model_name + "-PostsByWords")
save_object(WordsByFeatures,'matricies/', model_name + "-WordsByFeatures")


# ### Generate Word Clusters
# 
# Now that we have generated and saved our matricies,
# we will proceed to generate word clusters using 
# kmeans clustering, and save them for later analysis.
# 

from sklearn.cluster import KMeans
# get the fit for different values of K
test_points = [12] + list(range(25, 401, 25))
fit = []
for point in test_points:
    kmeans = KMeans(n_clusters = point, random_state = 42).fit(WordsByFeatures)
    save_object(kmeans, 'clusters/', model_name + "-words-cluster_model-" + str(point))
    fit.append(kmeans.inertia_)


save_object(fit, 'objects/', model_name + "-words" + "-fit")
save_object(test_points, 'objects/', model_name + "-words" + "-test_points")
del fit
del test_points


