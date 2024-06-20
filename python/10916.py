import pandas as pd
import csv
import os
import re
home = os.getenv('HOME')
downloads = os.path.join(home, 'Downloads', 'dmps')
mem_MB = 2000
top_tlds = {        # top 20 in Google searches per day
    '.com': ('Commercial', 4860000000),
    '.org': ('Noncomercial', 1950000000),
    '.edu': ('US accredited postsecondary institutions', 1550000000),
    '.gov': ('United States Government', 1060000000),
    '.uk':  ('United Kingdom', 473000000),
    '.net': ('Network services', 206000000),
    '.ca': ('Canada', 165000000),
    '.de': ('Germany', 145000000),
    '.jp': ('Japan', 139000000),
    '.fr': ('France', 96700000),
    '.au': ('Australia', 91000000),
    '.us': ('United States', 68300000),
    '.ru': ('Russian Federation', 67900000),
    '.ch': ('Switzerland', 62100000),
    '.it': ('Italy', 55200000),
    '.nl': ('Netherlands', 45700000),
    '.se': ('Sweden', 39000000),
    '.no': ('Norway', 32300000),
    '.es': ('Spain', 31000000),
    '.mil': ('US Military', 28400000)
    }


# # Be careful loading 1 GB  CSV with Pandas
# Use python csv row iterator to extract a single column   
# Lot's of room to improve the efficiency but this works... just barely  
# 

def extract_emails(top_tlds=top_tlds, colnum=1,
                   dest=os.path.join(downloads, extracted_emails)):
    tlds = set(top_tlds)
    email_regex = re.compile('[a-zA-Z0-9-.!#$%&*+-/=?^_`{|}~]+@[a-zA-Z0-9-.]+(' + '|'.join(tlds) + ')')
    emails = ''
    with open(os.path.join(downloads, 'aminno_member_email.csv')) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            em = email_regex.search(row[colnum])
            if em:
                emails += em.group().replace(',', '\n') + '\n'
            if len(emails) > mem_MB * 1000000:
                break
            if not (i % 100000):
                print("{}M emails read".format(i / 1000000.))
    # should just stream them to the outfile simultaneously
    with open(dest, 'w') as f:
        f.write(emails)
    # or at least just return them as a sequence/list/array
    # return pd.Series(emails.split('\n'))

extract_emails()


# # Pandas struggles to load & index 30M emails?
# 

emails = pd.DataFrame.from_csv(extracted_emails).index
# $ free -h
#              total       used       free     shared    buffers     cached
# Mem:          7.7G       6.4G       1.3G       139M       108M       2.6G
# -/+ buffers/cache:       3.6G       4.1G
# Swap:           0B         0B         0B
# (hackor)hobs@hplap:~/src/totalgood/webapps/hackor$  master
# $ free -h
#              total       used       free     shared    buffers     cached
# Mem:          7.7G       6.5G       1.2G       139M       108M       2.6G
# -/+ buffers/cache:       3.7G       4.0G
# Swap:           0B         0B         0B
# (hackor)hobs@hplap:~/src/totalgood/webapps/hackor$  master
# $ free -h
#              total       used       free     shared    buffers     cached
# Mem:          7.7G       7.5G       171M       132M       107M       1.8G
# -/+ buffers/cache:       5.6G       2.1G
# Swap:           0B         0B         0B
# (hackor)hobs@hplap:~/src/totalgood/webapps/hackor$  master
# $ free -h
#              total       used       free     shared    buffers     cached
# Mem:          7.7G       6.8G       937M       134M       107M       1.8G
# -/+ buffers/cache:       4.9G       2.8G
# Swap:           0B         0B         0B
# (hackor)hobs@hplap:~/src/totalgood/webapps/hackor$  master
# $ free -h
#              total       used       free     shared    buffers     cached
# Mem:          7.7G       5.1G       2.6G       134M       115M       1.9G
# -/+ buffers/cache:       3.1G       4.6G
# Swap:           0B         0B         0B


# # Who's been naughty or nice?
# 

naughty, nice = [], []
candidates = pd.DataFrame.from_csv(
    os.path.join('..', 'data', 'public.raw_candidate_filings.csv'))
candidates = set(candidates['email'].unique())
for i, em in enumerate(candidates):
    if str(em).lower().strip() in emails:
        naughty += [em]
        print('{}'.format(em[-10:]))
    else:
        nice += [em]
# @yahoo.com
# @yahoo.com
# @gmail.com
# ue@msn.com
# eurlaw.com
# etmail.com
# m1@msn.com
# @gmail.com

# print(nice)


# # Pandas as a Database
# Rather than installing postrgres and Django and configuring all that
# You can just use a Pandas collection of DataFrame tables as your database
# Lets see if we can find a "primary key" that we can use to connect a couple of these tables
# 

import pandas as pd
pacs_scraped = pd.DataFrame.from_csv('public.raw_committees_scraped.csv')  # id
pacs = pd.DataFrame.from_csv('public.raw_committees.csv')  # no ID that I can find
candidates = pd.DataFrame.from_csv('public.raw_candidate_filings.csv')  # id_nmbr
print(pacs_scraped.info())
print(candidates.info())


# # Primary Keys?
# Find PK and foreign key fields to be able to join across tables
# 

import re
from itertools import product
regex = re.compile(r'(\b|_|^)[Ii][Dd](\b|_|$)')
pac_id_cols = [col for col in pacs.columns if regex.search(col)]
print(pac_id_cols)
pac_scraped_id_cols = [col for col in pacs_scraped.columns if regex.search(col)]
print(pac_scraped_id_cols)
candidate_id_cols = [col for col in candidates.columns if regex.search(col)]
print(candidate_id_cols)
trans = pd.DataFrame.from_csv('public.raw_committee_transactions_ammended_transactions.csv')
trans_id_cols = [col for col in trans.columns if regex.search(col)]
print(trans_id_cols)
tables = [('pac', pacs, pac_id_cols), ('pac_scraped', pacs_scraped, pac_scraped_id_cols), ('candidate', candidates, candidate_id_cols), ('trans', trans, trans_id_cols)]
graph = []
for ((n1, df1, cols1), (n2, df2, cols2)) in product(tables, tables):
    if n1 == n2:
        continue
    for col1 in cols1:
        for col2 in cols2:
            s1 = set(df1[col1].unique())
            s2 = set(df2[col2].unique())
            similarity = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
            print('{}.{} -- {:.3} -- {}.{}'.format(n1, col1, similarity, n2, col2 ))
            graph += [(n1, col1, similarity, n2, col2)]
graph = pd.DataFrame(sorted(graph, key=lambda x:x[2]), columns=['table1', 'column1', 'similarity', 'table2', 'column2'])
print(graph)


print(pacs_scraped.index.dtype)
print(pacs.index.dtype)


trans = pd.DataFrame.from_csv('public.raw_committee_transactions_ammended_transactions.csv')
trans.describe()


# # Original_ID?
# 
# So it looks like there are multiple revisions for many of the "unique" original_id. So to consolidate those revisions into unqiue records with unique IDs (take the most recent revision as the official record):
# 

filtered_trans = []
for id in trans.original_id.unique():
    rows = sorted(trans[trans.original_id == id].iterrows(), key=lambda x:x[1].attest_date, reverse=True)
    filtered_trans += [rows[0][1]]
filtered_trans = pd.DataFrame(filtered_trans)
print(len(trans) / float(len(filtered_trans)))
print(filtered_trans.describe())


df = filtered_trans
filer_sums = df.groupby('filer_id').amount.sum()
print(pacs_scraped.columns)
print(df.columns)
for (filer_id, amount) in sorted(filer_sums.iteritems(), key=lambda x:x[1], reverse=True):
    names = pacs_scraped[pacs_scraped.id == filer_id].index.values
    print('{}\t{}\t{}'.format(filer_id, names[0][:40] if len(names) else '', amount))


# # NLP
# Let's build a graph of the similarity between PACs based on the wording of their committee names
# 

import matplotlib
get_ipython().magic('matplotlib inline')
np = pd.np
np.norm = np.linalg.norm
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


df = pacs_scraped
names = df.index.values
corpus = [' '.join(str(f) for f in fields) for fields in zip(*[df[col] for col in df.columns if df[col].dtype == pd.np.dtype('O')])]
print(corpus[:3])
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')
tfidf = vectorizer.fit_transform(corpus)
cov = tfidf * tfidf.T
cov[0:]


import re
import pandas as pd
pd.options.display.max_columns = 9
pd.options.display.max_rows = 3
np = pd.np
np.norm = np.linalg.norm
from datetime import datetime, date
import json
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
from sklearn.feature_extraction.text import TfidfVectorizer  # equivalent to TFIDFTransformer(CountVectorizer())
from django.db.models import Sum
from pacs.models import CampaignDetail, WorkingTransactions
import django
django.setup()
CampaignDetail.objects.count(), WorkingTransactions.objects.count()


# # Django
# This is how you join CampaignDetail & WorkingTransactions  
# and aggregate the WorkingTransactions.amount
# 

qs = CampaignDetail.objects.annotate(net_amount=Sum('workingtransactions__amount')).values().all()
print('Net transactions: {:,}M'.format(round(sum(qs.values_list('net_amount', flat=True)) / 1e6)))


# Convert a Django Queryset into a Pandas DataFrame
# 

df = pd.DataFrame.from_records(qs)
df.columns


df = df[df.committee_name.astype(bool)].copy()
df


# # Pandas DataFrame.join
# What if you only want positive transactions?

qs_pos = CampaignDetail.objects.filter(committee_name__isnull=False, workingtransactions__amount__gt=0)
qs_pos = qs_pos.annotate(pos_amount=Sum('workingtransactions__amount'))
df_pos = df.join(pd.DataFrame.from_records(qs_pos.values('pos_amount').all())['pos_amount'])
df_pos


# ### What if I just insert a new column with the values?
# 

df = pd.DataFrame.from_records(qs)
df = pd.DataFrame(df[df.committee_name.astype(bool)])
df['pos_amount'] = pd.DataFrame.from_records(qs_pos.values('pos_amount').all())['pos_amount']
df


# # Pandas indices are tricky
# Did all the rows get inserted in the right place (are the indices still alligned)
# 

df == df_pos


pd.options.display.max_rows = 6
(df == df_pos).mean()


# # A NaN is not equal to a NaN!
# Any operation involving a NaN returns a NaN  
# And NaN (like None) always evaluates to False  
# 

(df == df_pos).mean() + df.isnull().mean()


# # Negative transaction amounts?
# 

qs_neg = CampaignDetail.objects.filter(workingtransactions__amount__lt=0)
qs_neg = qs_neg.annotate(neg_amount=Sum('workingtransactions__amount'))
df = df.join(pd.DataFrame.from_records(qs_neg.values('neg_amount').all())['neg_amount'])
df


print('Positve transactions: {:,} M'.format(round(sum(qs_pos.values_list('pos_amount', flat=True)) / 1e6)))


print('Negative transactions: {:,} M'.format(round(sum(qs_neg.values_list('neg_amount', flat=True)) / 1.e6, 2)))


print('Net net transactions: {:,} M'.format(round(sum(qs.values_list('net_amount', flat=True)) / 1.e6)))


# ## Something's fishy in Denmark ^
# 

df.sum()


print('Net amount: ${:} M'.format(round(df.sum()[['pos_amount', 'neg_amount']].sum()/1e6, 2)))


print('Volume: ${:} M'.format(round(np.abs(df.sum()[['pos_amount', 'neg_amount']]).sum()/1e6, 2)))


# # Directed graph of financial transactions
# Are the payee_committee_ids the same as "filer_id"?
# 

filer_id = set(pd.DataFrame.from_records(WorkingTransactions.objects.values(
               'filer_id').all()).dropna().values.T[0])
payee_id = set(pd.DataFrame.from_records(WorkingTransactions.objects.values(
               'contributor_payee_committee_id').all()).dropna().values.T[0])
com_id = set()
len(payee_id.intersection(filer_id)) * 1. / len(filer_id)


# # Good enough for Government Work
# 53% of payee_ids were found in the filer_id of the same Table
# filer_id -> payee_id
# 

qs = WorkingTransactions.objects.filter(filer_id__isnull=False, 
                                        contributor_payee_committee_id__isnull=False,
                                        amount__gt=0)
df_trans = pd.DataFrame.from_records(qs.values().all())
_, trans = df_trans.iterrows().next()
print(trans)
print(trans.index.values)


# # Lets compute a similarity matrix from positive transactions
# 

ids = [int(i) for i in payee_id.intersection(filer_id)]
id_set = set(ids)
id_str = [str(int(i)) for i in ids]
N = len(ids)
cov = pd.DataFrame(np.zeros((N, N)),
                   index=pd.Index(id_str, name='payee'),
                   columns=pd.Index(id_str, name='filer'))
print(cov)
for rownum, trans in df_trans.iterrows():
    fid = trans['filer_id_id']
    # print(trans.index.values)
    cid = trans['contributor_payee_committee_id']
    if fid in id_set and cid in id_set:
#         if not (fid % 100):
#             print(cov[str(fid)][str(cid)])
        #only populate the upper
        if fid > cid:
            fid, cid = cid, fid
        amount = abs(trans['amount'])
        if amount > 0:
            cov[str(fid)][str(cid)] += amount
cov.describe()
    


cov


cov.sum()


# # My Older Self Say Wha?
# 

# ## Stolen Shamelessly From [I Am Trask](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)
# ### I HIGHLY recommend his blog for basic intros to Neural Network hacking.
# 

# RNN's are for adding the dimension of sequence to the network, just as CNN's added spatial dimensionality.
# This is why they are so powerful for NLP, specifically text generation.
# 

# But let's start small and teach an algorithm to add.  The "sequence" here is the sum of each bit of the binary representation.
# 

""" Literal cut and paste from Andrew Trask's blog above """

import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

        


# Note: "Unrolling" is the term for how far to extend the network in time before computing backprop.  8 in the case of the binary digits above.
# 

# ## Text Generation
# 
# ### Let's defer to [Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
# Text (of Moby Dick) is from [Project Guttenberg](https://www.gutenberg.org/) with the license information stripped out (sorry!).
# 

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('moby_dick.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 


# ## The Heavy Lifting - LSTM
# Discussion of Long Short Term Memory from [Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
# 

# Notes from the above blog:
# - Cell state
# - Logic gates with learned parameters to update state (All sigmoids)
# - Normal Activation is tanh to cell state and output of tne node
# - Forget Gate
# - Input Gate
# - Output Gate
# 
# Not discussed here:
# - How do we update the weights of the gates?  Same as always, just change the math based on the activation function it when through.
# - Vannishing error gradients - prevented by Error Carusel
# - Error Carousel (propogate error, until network learns the proper layers to diminish it)
# 

# 
# # Pandas as a Database
# Rather than installing postrgres and Django and configuring all that You can just use a Pandas collection of DataFrame tables as your database Lets see if we can find a "primary key" that we can use to connect a couple of these tables
# Thanks [Jeremy Tanner](https://twitter.com/Penguin) for the max_rows trick
# 

import pandas as pd
pd.set_option('display.max_rows', 6)


candidates = pd.DataFrame.from_csv('public.raw_candidate_filings.csv').sort_values(by='last_name')  # id_nmbr
candidates[[col for col in candidates.columns if 'name' in col or 'id' in col]]


pacs = pd.DataFrame.from_csv('public.raw_committees.csv').sort_values(by='committee_name')  # no ID that I can find
pacs[[col for col in pacs.columns if 'name' in col or 'id' in col or 'type' in col]]


pacs_scraped = pd.DataFrame.from_csv('public.raw_committees_scraped.csv', index_col='id').sort_values(by='name')  # id
pacs_scraped


candidates


# Thanks
pd.set_option("display.max_rows",101)


# # Vectorizing
# Thanks [@larsmans](http://stackoverflow.com/users/166749/larsmans) 
# for a speedy [stemming vectorizer](http://stackoverflow.com/a/26212970/623735)
# 

get_ipython().system('pip install pystemmer')
import Stemmer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))
     
tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1, 2))


# # K-Means
# (undirected learning technique)
# 
# first bit pulled from [my snippet](https://github.com/TheGrimmScientist/Snippets/blob/master/OutlierDetectionWithClustering/OutlierDetectionWithClustering.ipynb)
# 

get_ipython().magic('pylab inline')
import sklearn


# generate data
from sklearn.datasets.samples_generator import make_blobs
data = make_blobs(n_samples=100, n_features=2, centers=3,cluster_std=2.5)[0]


# compute centers
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(data)
[center_1, center_2, center_3] = km.cluster_centers_


figure(1,figsize=(10,6))
plot([row[0] for row in data],[row[1] for row in data],'b.')
plot(center_1[0],center_1[1], 'g*',ms=15)
plot(center_2[0],center_2[1], 'g*',ms=15)
plot(center_3[0],center_3[1], 'g*',ms=15)


# ## Now let's look under the hood
# 

#re-init things
data = make_blobs(n_samples=100, n_features=2, centers=3,cluster_std=1)[0]
km = KMeans(n_clusters=3, max_iter=1, init='random', n_init=1)
figure(2,figsize=(10,6))
plot([row[0] for row in data],[row[1] for row in data],'b.')


# ctrl-enter on this one to keep re-running it to watch the centers move around

km.fit(data)
[center_1, center_2, center_3] = km.cluster_centers_


figure(3,figsize=(10,6))
plot([row[0] for row in data],[row[1] for row in data],'b.')

plot(center_1[0],center_1[1], 'g*',ms=15)
plot(center_2[0],center_2[1], 'g*',ms=15)
plot(center_3[0],center_3[1], 'g*',ms=15)


# # Scree plot
# (well, something like it)
# 

X = []
Y = []
for n_clusters in range(1,10):
    X.append(n_clusters)
    n_repeats = 10
    y = 0.
    for i in range(n_repeats):
        data = make_blobs(n_samples=100, n_features=2, centers=3,cluster_std=2.5)[0]
        km = KMeans(n_clusters=n_clusters)
        km.fit(data)
        score = km.score(data)
        y += score
    Y.append(y/n_repeats)

figure(4,figsize=(10,6))
plot(X,Y)
title('Average Score for a given K')
xlabel('K')
ylabel('Average Score')


# fyi:  mini-batch k-means is better at scale.
# 

# ### Do this first
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ### Got Data?
# Unzip some crime data downloaded from the Portland Civic Data website
# 

get_ipython().system('unzip -u -d ../../data ../../data/crime_incident_data_2013.zip')
get_ipython().system('mv ../../data/crime_incident_data.csv ../../data/crime_incident_data_2013.csv')
get_ipython().system('unzip -u -d ../../data ../../data/crime_incident_data_2014.zip')
get_ipython().system('mv ../../data/crime_incident_data.csv ../../data/crime_incident_data_2014.csv')


# ### Your First DataFrame
# Load some crime data into a dictionary of pandas data frames
# 

import pandas as pd
crime = {}
for yr in range(13, 15):
    crime[yr] = pd.DataFrame.from_csv('crime_incident_data_20{}.csv'.format(yr))
df = crime[13]
df


# ### Explore the Data
# `df.describe()` works well for numerical data, but there are only 2 columns with numerical data
# 

df.describe()


# **Better than `describe`** (but poorly named)
# The pandas-profiling package works better for text and categorical data.
# `pandas_profiling.describe(df)`  
# But it's another package you have to install separately".
# Side Note: In development "profiling" usually means checking the memory and CPU usage of each element of a program. In statistics it just means to compute statistics about a subpopulation (as-in "racial profiling" by police) 
# 

get_ipython().system('pip install pandas-profiling')
import pandas_profiling as prof
stats = prof.describe(df)
stats


# Whenever you "profile" a dataset, should you be worried about any ethics or legal issues?  
# Why or why not?  
# Hint: I took the trouble to ask the question...  
# 

df = crime[13].copy()
labels = list(df.columns)
for i in range(-1, -3, -1):
    labels[i] = labels[i].strip().lower()[0]
df = pd.DataFrame(df.values, columns=labels)
plt.scatter(df.x, df.y)


# see: http://github.com/totalgood/python-rtmbot
# fork of official slack RTM-bot
# 

from slackclient import SlackClient
client = SlackClient()



