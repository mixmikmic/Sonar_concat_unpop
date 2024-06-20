# %matplotlib notebook
get_ipython().magic('matplotlib inline')

import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from collections import deque
from _converter_v2 import Event2Dict
from _evaluation import print_metrics

_true = "true"
_false = "false"
path = "../offline/"


# Quality_OK is mapped to Faultiness
# * 'False' -> 1 (Faulty)
# * 'True'  -> 0 (Good)
# 

def load(prod_id, mx=0):
    filename = path+"layouts/"+prod_id+".json"
    print("Loading layout... {}".format(filename))
    with open(filename) as j:    
        layout = json.load(j)

    filename = prod_id+".txt"
    data = []
    times = []
    bad = 0
    
    print("Loading rows... {}".format(filename))
    with open(path+filename) as f:
        i = 0
        for line in f:
            i = i+1
            if mx>0 and i>mx:
                break
                
            try:
                features, timestamps = Event2Dict(json.loads(line), layout)
                data.append(features.values())
                times.append(timestamps.values())
            except Exception, e:
                print(e)
                break
                bad+=1

    print("Errors: {}".format(bad))
    print("Loaded: {}".format(len(data)))

    """ random split seed """
    data, times = np.asarray(data), np.asarray(times)
    # print(data.shape, times.shape)
    mask = np.random.rand(len(data)) < 0.9
    print("Total: {} Good: {} Faulty: {} Ratio: {}".format(len(data), len(data[data[:,-1]==_true]), len(data[data[:,-1]==_false]), len(data[data[:,-1]==_false])/float(len(data[data[:,-1]==_true]))))
    return data, times, mask, layout


""" Logistic Sub-Sample """
from scipy.stats import logistic

# Take random numbers from a logistic probability density function
def logistic_choice(total, sample_size, replace=False):
    p = logistic.pdf(np.arange(0,total), loc=0, scale=total/5)
    p /= np.sum(p)
    return np.random.choice(total, size=sample_size, replace=replace, p=p)

def subsample(train, quiet=False):
    faulty = train[train[:,-1]==_false]
    not_faulty = train[train[:,-1]==_true]
    fr = len(faulty)/float(len(train))
    if not quiet:
        print("Train Total: {} Good: {} Faulty: {} Ratio: {}".format(len(train), len(not_faulty), len(faulty), fr))
        print("Test  Total: {} Good: {} Faulty: {} Ratio: {}".format(len(test), len(test[test[:,-1]==_true]), len(test[test[:,-1]==_false]), float(len(test[test[:,-1]==_false]))/len(test)))

        print("Re-sampling...")
        
    sample_size = np.min([5000, len(not_faulty)])
    samples = logistic_choice(len(not_faulty), sample_size)
    # TODO: Upsample faulties with logistic_choice(replace=True)
    f_sample_size = np.min([1000, len(faulty)])
    f_samples = logistic_choice(len(faulty), f_sample_size)
    # Put samples together and shuffle
    train = np.concatenate((not_faulty[samples], faulty[f_samples]))
    train = np.random.permutation(train)

    fr = len(train[train[:,-1]==_false])/float(len(train))
    if not quiet:
        print("Train Total: {} Good: {} Faulty: {} Ratio: {}".format(len(train), len(train[train[:,-1]==_true]), len(train[train[:,-1]==_false]), fr))

    return train


# ## Accuracy
# 

""" load data """
data, _, mask, _ = load("ABU6")
train = data[mask]
test = data[~mask]

test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)
train_data = train[:,2:-1].astype(np.float32)
train_labels = np.array(train[:,-1]==_false).astype(np.int32)
# sub-sample
train_s = subsample(train)
train_data_s = train_s[:,2:-1].astype(np.float32)
train_labels_s = np.array(train_s[:,-1]==_false).astype(np.int32)


# balance the train set (for accuracy metric)
balanced = True
if balanced:
    faulty = test[test[:,-1]==_false]
    not_faulty = test[test[:,-1]==_true]
    test = np.concatenate((not_faulty[:len(faulty)], faulty))
    test_data = test[:,2:-1].astype(np.float32)
    test_labels = np.array(test[:,-1]==_false).astype(np.int32)


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier

# ALL
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data, train_labels)')
# print_metrics(full_train_labels, clf.predict(full_train_data))
print_metrics(test_labels, clf.predict(test_data))

# SUBSAMPLE
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data_s, train_labels_s)')
# print_metrics(train_labels_s, clf.predict(train_data_s))
print_metrics(test_labels, clf.predict(test_data))


""" Gradient Boosting """
from sklearn.ensemble import GradientBoostingClassifier
losses = ['deviance', 'exponential']

# ALL
clf = GradientBoostingClassifier(loss=losses[0], n_estimators=100,  max_depth=15, learning_rate=0.1)
get_ipython().magic('time clf = clf.fit(train_data, train_labels)')
# print_metrics(full_train_labels, clf.predict(full_train_data))
print_metrics(test_labels, clf.predict(test_data))

# SUBSAMPLE
clf = GradientBoostingClassifier(loss=losses[0], n_estimators=100,  max_depth=25, learning_rate=0.1)
get_ipython().magic('time clf = clf.fit(train_data_s, train_labels_s)')
# print_metrics(train_labels_s, clf.predict(train_data_s))
print_metrics(test_labels, clf.predict(test_data))


# ## Accuacy vs. Number of Datapoints
# 

""" load data """
# mx = 50000
# D = {}
# for n in range(1,6+1):
#     prod_id = "ABU{}".format(n)
#     data, _, mask = load(prod_id, mx)
#     D[prod_id] = {
#         'data': data, 
#         'mast': mask
#     }

data, _, mask, _ = load("ABU4")
test = data[~mask]
test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)

train = data[mask]
faulty = train[train[:,-1]==_false]
not_faulty = train[train[:,-1]==_true]

ratio = 0.95/0.05


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
MCCs = []
HLs = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,100):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    MCCs.append(mcc)
    HLs.append(hl)

# with sub-sample    
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
MCCs_s = []
HLs_s = []
subs_size = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,100):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    # sub-sample
    train_s = subsample(small, quiet=True)
    faulty_s = train_s[train_s[:,-1]==_false]
    not_faulty_s = train_s[train_s[:,-1]==_true]
    
    small = np.concatenate((not_faulty_s, faulty_s))
    subs_size.append(len(small))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    MCCs_s.append(mcc)
    HLs_s.append(hl)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

clf = GradientBoostingClassifier(loss='deviance', n_estimators=100,  max_depth=15, learning_rate=0.1)
GBMCCs = []
GBHLs = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,100):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    GBMCCs.append(mcc)
    GBHLs.append(hl)
    
# with sub-sample
clf = GradientBoostingClassifier(loss='deviance', n_estimators=100,  max_depth=25, learning_rate=0.1)
GBMCCs_s = []
GBHLs_s = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,100):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    # sub-sample
    train_s = subsample(small, quiet=True)
    faulty_s = train_s[train_s[:,-1]==_false]
    not_faulty_s = train_s[train_s[:,-1]==_true]
    
    small = np.concatenate((not_faulty_s, faulty_s))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    GBMCCs_s.append(mcc)
    GBHLs_s.append(hl)


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/17)))
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_ylabel('Matthew\'s Correlation Coefficient (MCC)')


ax1.set_xlabel("Total Datapoints")
p = ax1.plot(rrange, MCCs, label="Random Forests", color="black")
ax1.axhline(y=0.9, color='g', linestyle='dashed', alpha=.5)
ax1.grid(True)
ax1.legend(loc=4)
# for xy in zip(rrange, MCCs):  
#     ax1.annotate("%.2f"%xy[1], xy=xy, textcoords='data')

ax3 = fig.add_subplot(111)
ax3.plot(rrange, GBMCCs, label="Gradient Boosting", color="black", linestyle='--')
ax3.legend(loc=4)

ax5 = fig.add_subplot(111)
ax5.plot(rrange, MCCs_s, label="Random Forests (subs.)", color="black", linestyle=':')
ax5.legend(loc=4)

ax5 = fig.add_subplot(111)
ax5.plot(rrange, GBMCCs_s, label="Gradient Boosting (subs.)", color="black", linestyle='-.')
ax5.legend(loc=4)

# ax2 = ax1.twiny()
# ax2.cla()
# ax2.set_xlabel('Sub-sample Datapoints')
# ax2.plot(subs_size,np.ones(18), alpha=.0)
# print subs_size

plt.show()


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/17)))
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_ylabel('Hamming Loss')


ax1.set_xlabel("Total Datapoints")
p = ax1.plot(rrange, HLs, label="Random Forests", color="black")
# ax1.axhline(y=0.9, color='g', linestyle='dashed', alpha=.5)
ax1.grid(True)
ax1.legend(loc=1)
# ax1.set_autoscaley_on(False)
# ax1.set_ylim([0.0,0.06])

ax3 = fig.add_subplot(111)
ax3.plot(rrange, GBHLs, label="Gradient Boosting", color="black", linestyle='--')
ax3.legend(loc=1)

ax5 = fig.add_subplot(111)
ax5.plot(rrange, HLs_s, label="Random Forests (subs.)", color="black", linestyle=':')
ax5.legend(loc=1)

ax5 = fig.add_subplot(111)
ax5.plot(rrange, GBHLs_s, label="Gradient Boosting (subs.)", color="black", linestyle='-.')
ax5.legend(loc=1)

plt.show()


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/17)))
arange = xrange(0,18,1)
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_ylabel('Datapoints')
ax1.set_xlabel('Available Chunks')
p = ax1.plot(arange, rrange, label="Train Set", color="black")
ax1.grid(True)
ax1.legend(loc=4)

ax3 = fig.add_subplot(111)
ax3.plot(arange, subs_size, label="Sub-sampled Train Set", color="black", linestyle='--')
ax3.legend(loc=4)

plt.show()


# ## Early Detection Rate
# 

""" load data """
data, times, mask, layout = load("ABU3")
train = data[mask]
test = data[~mask]

test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)
train_data = train[:,2:-1].astype(np.float32)
train_labels = np.array(train[:,-1]==_false).astype(np.int32)

faulty = train[train[:,-1]==_false]
not_faulty = train[train[:,-1]==_true]
means = np.mean(not_faulty[:,2:-1].astype(np.float32), axis=0)

# sub-sample
train_s = subsample(train)
train_data_s = train_s[:,2:-1].astype(np.float32)
train_labels_s = np.array(train_s[:,-1]==_false).astype(np.int32)


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier

# Use subsample
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data_s, train_labels_s)')
# print_metrics(train_labels_s, clf.predict(train_data_s))
print_metrics(test_labels, clf.predict(test_data))


from sklearn import metrics
# test with partial feature vectors
# check_at = ["PasteInspection/PosY6", "AOI1/PosY6", "AOI2/PosY6"] # ABU1-3, ABU5-6
check_at = ["PasteInspection/PosX5", "AOI1/PosX5", "AOI2/PosX5"] # ABU4
check_at_index = []
for name in check_at:
    check_at_index.append(layout["measurements"].index(name))

CMs = []
for i in check_at_index:
    snapshot = copy.deepcopy(test_data)
    for r in snapshot:
        r[i:] = means[i:]
    
    predictions = clf.predict(snapshot)
    cm = metrics.confusion_matrix(test_labels, predictions)
    print np.where(predictions==1)[0].shape
    print cm
    CMs.append(cm)
    
CMs = np.asarray(CMs)
N = CMs[:,1,0]+CMs[:,1,1]
TN = CMs[:,1,1]
print TN/N.astype(np.float32)


# ## Early Detection - Saving Production Time
# 

from sklearn import metrics
# test with partial feature vectors
# check_at = ["PasteInspection/PosY6", "AOI1/PosY6", "AOI2/PosY6"] # ABU1-3, ABU5-6
check_at = ["PasteInspection/PosX5", "AOI1/PosX5", "AOI2/PosX5"] # ABU4
check_at_index = []
for name in check_at:
    check_at_index.append(layout["measurements"].index(name))

CMs = []
total_TN = 0
total_FP = 0
total_saved = 0
total_wasted = 0
for i in check_at_index:
    snapshot = copy.deepcopy(test_data)
    for r in snapshot:
        r[i:] = means[i:]
    
    predictions = clf.predict(snapshot)
    cm = metrics.confusion_matrix(test_labels, predictions)
    print cm
    ####
    TN = cm[1,1] - total_TN
    total_TN += TN
    saved = times[0][-1]-times[0][i+2]
#     print TN, saved
    total_saved += TN*saved
    ####
    FP = cm[0,1] - total_FP
    total_FP += FP
    wasted = times[0][i+2]-times[0][0]
#     print FP, wasted
    total_wasted += FP*wasted
    
total = times[0][-1]-times[0][0]
total *= len(test_data)

print "Total Saved:", total_saved/3600.0
print "Total Wasted:", total_wasted/3600.0
print "Total Spent (if not detected):", total/len(check_at_index)/3600.








# %matplotlib notebook
get_ipython().magic('matplotlib inline')

import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from collections import deque
from _converter_v2 import Event2Dict
from _evaluation import print_metrics

_true = "true"
_false = "false"
path = ""


# Quality_OK is mapped to Faultiness
# * 'False' -> 1 (Faulty)
# * 'True'  -> 0 (Good)
# 

def load(prod_id, mx=0):
    filename = path+"layouts/"+prod_id+".json"
    print("Loading layout... {}".format(filename))
    with open(filename) as j:    
        layout = json.load(j)

    filename = prod_id+".txt"
    data = []
    times = []
    bad = 0
    
    print("Loading rows... {}".format(filename))
    with open(path+filename) as f:
        i = 0
        for line in f:
            i = i+1
            if mx>0 and i>mx:
                break
                
            try:
                features, timestamps = Event2Dict(json.loads(line), layout)
                data.append(features.values())
                times.append(timestamps.values())
            except Exception, e:
                print(e)
                break
                bad+=1

    print("Errors: {}".format(bad))
    print("Loaded: {}".format(len(data)))

    """ random split seed """
    data, times = np.asarray(data), np.asarray(times)
    # print(data.shape, times.shape)
    mask = np.random.rand(len(data)) < 0.9
    print("Total: {} Good: {} Faulty: {} Ratio: {}".format(len(data), len(data[data[:,-1]==_true]), len(data[data[:,-1]==_false]), len(data[data[:,-1]==_false])/float(len(data[data[:,-1]==_true]))))
    return data, times, mask, layout


""" Logistic Sub-Sample """
from scipy.stats import logistic

# Take random numbers from a logistic probability density function
def logistic_choice(total, sample_size, replace=False):
    p = logistic.pdf(np.arange(0,total), loc=0, scale=total/5)
    p /= np.sum(p)
    return np.random.choice(total, size=sample_size, replace=replace, p=p)

def subsample(train, quiet=False):
    faulty = train[train[:,-1]==_false]
    not_faulty = train[train[:,-1]==_true]
    fr = len(faulty)/float(len(train))
    if not quiet:
        print("Train Total: {} Good: {} Faulty: {} Ratio: {}".format(len(train), len(not_faulty), len(faulty), fr))
        print("Test  Total: {} Good: {} Faulty: {} Ratio: {}".format(len(test), len(test[test[:,-1]==_true]), len(test[test[:,-1]==_false]), float(len(test[test[:,-1]==_false]))/len(test)))

        print("Re-sampling...")
        
    sample_size = np.min([5000, len(not_faulty)])
    samples = logistic_choice(len(not_faulty), sample_size)
    # TODO: Upsample faulties with logistic_choice(replace=True)
    f_sample_size = np.min([1000, len(faulty)])
    f_samples = logistic_choice(len(faulty), f_sample_size)
    # Put samples together and shuffle
    train = np.concatenate((not_faulty[samples], faulty[f_samples]))
    train = np.random.permutation(train)

    fr = len(train[train[:,-1]==_false])/float(len(train))
    if not quiet:
        print("Train Total: {} Good: {} Faulty: {} Ratio: {}".format(len(train), len(train[train[:,-1]==_true]), len(train[train[:,-1]==_false]), fr))

    return train


# ## Accuracy
# 

""" load data """
data, _, mask, _ = load("ABU1")
train = data[mask]
test = data[~mask]

test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)
train_data = train[:,2:-1].astype(np.float32)
train_labels = np.array(train[:,-1]==_false).astype(np.int32)
# sub-sample
train_s = subsample(train)
train_data_s = train_s[:,2:-1].astype(np.float32)
train_labels_s = np.array(train_s[:,-1]==_false).astype(np.int32)


# balance the train set (for accuracy metric)
balanced = True
if balanced:
    faulty = test[test[:,-1]==_false]
    not_faulty = test[test[:,-1]==_true]
    test = np.concatenate((not_faulty[:len(faulty)], faulty))
    test_data = test[:,2:-1].astype(np.float32)
    test_labels = np.array(test[:,-1]==_false).astype(np.int32)


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier

# ALL
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data, train_labels)')
# print_metrics(full_train_labels, clf.predict(full_train_data))
print_metrics(test_labels, clf.predict(test_data))

# SUBSAMPLE
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data_s, train_labels_s)')
# print_metrics(train_labels_s, clf.predict(train_data_s))
print_metrics(test_labels, clf.predict(test_data))


# ## Accuacy vs. Number of Datapoints
# 

""" load data """
# mx = 50000
# D = {}
# for n in range(1,6+1):
#     prod_id = "ABU{}".format(n)
#     data, _, mask = load(prod_id, mx)
#     D[prod_id] = {
#         'data': data, 
#         'mast': mask
#     }

data, _, mask, _ = load("ABU1")
test = data[~mask]
test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)

train = data[mask]
faulty = train[train[:,-1]==_false]
not_faulty = train[train[:,-1]==_true]

ratio = 0.95/0.05


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
MCCs = []
HLs = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,50):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    MCCs.append(mcc)
    HLs.append(hl)


""" Gradient Boosting """
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

clf = GradientBoostingClassifier(loss='deviance', n_estimators=100,  max_depth=15, learning_rate=0.1)
GBMCCs = []
GBHLs = []
print("Total\tMCC\tHL")
for tr in xrange(10,1720,50):
    small = np.concatenate((not_faulty[:int(tr*ratio)], faulty[:tr]))
    small_data = small[:,2:-1].astype(np.float32)
    small_labels = np.array(small[:,-1]==_false).astype(np.int32)
    clf = clf.fit(small_data, small_labels)
    predicted_labels = clf.predict(test_data)
    mcc = metrics.matthews_corrcoef(test_labels, predicted_labels)
    hl = metrics.hamming_loss(test_labels, predicted_labels)
    print("{}\t{}\t{}".format(len(small), mcc, hl))
    GBMCCs.append(mcc)
    GBHLs.append(hl)


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/34)))
fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(111)
ax1.set_ylabel('MCC')
ax1.set_xlabel("Feature Vectors")

# RF
p = ax1.plot(rrange, MCCs, label="Random Forest", color='b')
ax1.axhline(y=0.9, color='gray', linestyle='dashed', alpha=.5)
ax1.grid(True)
ax1.legend(loc=4)
# GB
ax3 = fig.add_subplot(111)
ax3.plot(rrange, GBMCCs, label="Gradient Boosting", color="g")
ax3.legend(loc=4)

plt.show()


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/34)))
fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(111)
ax1.set_ylabel('Hamming Loss')
ax1.set_xlabel("Feature Vectors")

# RF
p = ax1.fill_between(rrange, HLs, label="Random Forest", alpha=0.2, color='b')
ax1.grid(True)
ax1.axhline(y=0.01, color='gray', linestyle='dashed', alpha=.5)
ax1.legend(loc=1)
# GB
ax2 = fig.add_subplot(111)
ax2.fill_between(rrange, GBHLs, label="Gradient Boosting", alpha=0.5, color='g')
ax2.grid(True)
ax2.legend(loc=1)

plt.show()


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', quality=100)

rrange = xrange(0,30000,int(np.ceil(30000/17)))
arange = xrange(0,18,1)
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_ylabel('Datapoints')
ax1.set_xlabel('Available Chunks')
p = ax1.plot(arange, rrange, label="Train Set", color="black")
ax1.grid(True)
ax1.legend(loc=4)

ax3 = fig.add_subplot(111)
ax3.plot(arange, subs_size, label="Sub-sampled Train Set", color="black", linestyle='--')
ax3.legend(loc=4)

plt.show()


# ## Early Detection Rate
# 

""" load data """
data, times, mask, layout = load("ABU5")
train = data[mask]
test = data[~mask]

test_data = test[:,2:-1].astype(np.float32)
test_labels = np.array(test[:,-1]==_false).astype(np.int32)
train_data = train[:,2:-1].astype(np.float32)
train_labels = np.array(train[:,-1]==_false).astype(np.int32)

faulty = train[train[:,-1]==_false]
not_faulty = train[train[:,-1]==_true]
means = np.mean(not_faulty[:,2:-1].astype(np.float32), axis=0)

# sub-sample
train_s = subsample(train)
train_data_s = train_s[:,2:-1].astype(np.float32)
train_labels_s = np.array(train_s[:,-1]==_false).astype(np.int32)


""" Random Forest """
from sklearn.ensemble import RandomForestClassifier

# Use subsample
clf = RandomForestClassifier(n_estimators=100, max_depth=90, n_jobs=4)
get_ipython().magic('time clf = clf.fit(train_data_s, train_labels_s)')
# print_metrics(train_labels_s, clf.predict(train_data_s))
print_metrics(test_labels, clf.predict(test_data))


from sklearn import metrics
# test with partial feature vectors
# check_at = ["PasteInspection/PosY6", "AOI1/PosY6", "AOI2/PosY6"] # ABU1-3, ABU5-6
check_at = ["PasteInspection/PosX5", "AOI1/PosX5", "AOI2/PosX5"] # ABU4
check_at_index = []
for name in check_at:
    check_at_index.append(layout["measurements"].index(name))

CMs = []
for i in check_at_index:
    snapshot = copy.deepcopy(test_data)
    for r in snapshot:
        r[i:] = means[i:]
    
    predictions = clf.predict(snapshot)
    cm = metrics.confusion_matrix(test_labels, predictions)
    print np.where(predictions==1)[0].shape
    print cm
    CMs.append(cm)
    
CMs = np.asarray(CMs)
N = CMs[:,1,0]+CMs[:,1,1]
TN = CMs[:,1,1]
print TN/N.astype(np.float32)


# ## Early Detection - Saving Production Time
# 

from sklearn import metrics
# test with partial feature vectors
# check_at = ["PasteInspection/PosY6", "AOI1/PosY6", "AOI2/PosY6"] # ABU1-3, ABU5-6
check_at = ["PasteInspection/PosX5", "AOI1/PosX5", "AOI2/PosX5"] # ABU4
check_at_index = []
for name in check_at:
    check_at_index.append(layout["measurements"].index(name))

CMs = []
total_TN = 0
total_FP = 0
total_saved = 0
total_wasted = 0
for i in check_at_index:
    snapshot = copy.deepcopy(test_data)
    for r in snapshot:
        r[i:] = means[i:]
    
    predictions = clf.predict(snapshot)
    cm = metrics.confusion_matrix(test_labels, predictions)
    print cm
    ####
    TN = cm[1,1] - total_TN
    total_TN += TN
    saved = times[0][-1]-times[0][i+2]
#     print TN, saved
    total_saved += TN*saved
    ####
    FP = cm[0,1] - total_FP
    total_FP += FP
    wasted = times[0][i+2]-times[0][0]
#     print FP, wasted
    total_wasted += FP*wasted
    
total = times[0][-1]-times[0][0] # assuming they all take same time
print "Production duration:", total/3600.
total *= len(test_data)

print "Total Saved:", total_saved/3600.0
print "Total Wasted:", total_wasted/3600.0
print "Total Spent (if not detected):", total/3600.0





