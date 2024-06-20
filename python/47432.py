import numpy as np
import deepdish.io as io
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


yhat = np.load('yhat.npy')
test = io.load('data/test_dict.hdf5')
y = test['y']
w = test['w']
X = test['X']


y_true = np.argmax(y, axis=1)
yhat_top1 = np.argmax(yhat, axis=1)


bins = np.linspace(0, max(y_true), max(y_true)+1)
h1 = plt.hist(y_true, bins=bins) #weights=w
plt.title('Total number of best jet pairs per position') # unweighted

for k in range(max(y_true)):
    plt.text((h1[1][k] + h1[1][k+1])/2, 100+h1[0][k],
                '%d' % int(h1[0][k]),
                ha='center', va='bottom')
plt.ylim(0, 90000)


h2 = plt.hist(y_true[yhat_top1 == y_true], bins=bins, color='green')
plt.title('Number of correctly identified best jet pairs by position')
for k in range(max(y_true)):
    plt.text((h2[1][k] + h2[1][k+1])/2, 100+h2[0][k],
                '%d' % int(h2[0][k]),
                ha='center', va='bottom')
plt.ylim(0, 90000)


h1 = plt.hist(y_true, bins=bins, histtype='step', color='blue',
              label='Total # of best jet pairs by position')
h2 = plt.hist(y_true[yhat_top1 == y_true], bins=bins, histtype='step', color='green',
              label='# of correctly identified best jet pairs')
_ = plt.ylim(0, 90000)
_ = plt.legend()
# -- this is the same as above, but with the 2 histograms on the same plot for comparison


_ = plt.scatter(h1[1][:-1], h2[0] / h1[0])
_ = plt.ylim(0, 1)
plt.grid('on')
plt.title('Accuracy per best jet pair position')


from sklearn.metrics import confusion_matrix
def _plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.tight_layout()
    plt.ylabel('True position of best jet pair')
    plt.xlabel('Predicted position of best jet pair')

cm = confusion_matrix(y_true, yhat_top1)
# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
_plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# import cPickle as pickle
# d = pickle.load(open(
#     '../James/bbyy_jet_classifier/data/eventLevel_Oct15/SM_bkg_photon_jet.root---events_1tag---0.7---test.pkl', 'rb')
# )


np.where(np.array(test['vars']) == 'm_jb')


test['vars']


scaler = test['scalers'][7]


m_jb = X[:, :, 7]


m_jb = scaler.inverse_transform(m_jb)


m_jb_true = np.array(
    [m_jb[ev][y_true[ev]-1] if (y_true[ev]-1 >= 0) else -99 for ev in range(len(y_true))]
)


m_jb_pred = np.array(
    [m_jb[ev][yhat_top1[ev]-1] if (yhat_top1[ev]-1 >= 0) else -99 for ev in range(len(yhat_top1))]
)


y_mHmatch = np.array([
        np.argmin(ev[ev != -999]) if (ev[ev != -999].size != 0) else -1 for ev in X[:, :, 4]
])
m_jb_mHmatch = np.array(
    [m_jb[ev][y_mHmatch[ev]] if (y_mHmatch[ev] >= 0) else -99 for ev in range(len(y_mHmatch))]
)


y_pThigh = np.array([
        np.argmin(ev[ev != -999]) if (ev[ev != -999].size != 0) else -1 for ev in X[:, :, 5]
])
m_jb_pThigh = np.array(
    [m_jb[ev][y_pThigh[ev]] if (y_pThigh[ev] >= 0) else -99 for ev in range(len(y_pThigh))]
)


y_pTjbhigh = np.array([
        np.argmin(ev[ev != -999]) if (ev[ev != -999].size != 0) else -1 for ev in X[:, :, 6]
])
m_jb_pTjbhigh = np.array(
    [m_jb[ev][y_pTjbhigh[ev]] if (y_pTjbhigh[ev] >= 0) else -99 for ev in range(len(y_pTjbhigh))]
)


matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
bins = np.linspace(0, 800, 100)
_ = plt.hist(m_jb_true, bins=bins, histtype='step', label='true')
_ = plt.hist(m_jb_pred, bins=bins, histtype='step', label='RNN')
_ = plt.hist(m_jb_mHmatch, bins=bins, histtype='step', label='mHmatch')
_ = plt.hist(m_jb_pThigh, bins=bins, histtype='step', label='pThigh')
_ = plt.hist(m_jb_pTjbhigh, bins=bins, histtype='step', label='pTjbhigh')
_ = plt.xlabel(r'$m_{jb} \ \mathrm{GeV}$')
_ = plt.ylabel('Number of events')
_ = plt.title(r'$m_jb$ of jet pairs selected by different strategies')
_ = plt.legend()


# The distributions for 'true' and 'RNN' have lower area than the other three because these strategies are allowed to tell us that there was no correct jet pair in an event, hence automatically removing those events. On the other hand, the other three strategies are forced to pick a correct jet pair in any case, even when there isn't a good one.
# 

# To make the comparison fairer, we can condition on the existance of a correct jet pair, but the strategy would have never known this and would have still made a choice.


# matplotlib.rcParams.update({'font.size': 16})
# fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
# bins = np.linspace(0, 800, 100)
# _ = plt.hist(m_jb_true[y_true != 0], bins=bins, histtype='step', label='true')
# _ = plt.hist(m_jb_pred[y_true != 0], bins=bins, histtype='step', label='RNN')
# _ = plt.hist(m_jb_mHmatch[y_true != 0], bins=bins, histtype='step', label='mHmatch')
# _ = plt.hist(m_jb_pThigh[y_true != 0], bins=bins, histtype='step', label='pThigh')
# _ = plt.hist(m_jb_pTjbhigh[y_true != 0], bins=bins, histtype='step', label='pTjbhigh')
# plt.legend()
# # -- this doesn't make much sense I don't think


bkg_flag = np.array([True if 'bkg' in sample else False for sample in test['sample']])


# Let's divide the m_jb plot into components from the signal samples and the background samples. In theory, this should show us that mHmatch brings in lots of background in the signal region, while that effect is minor for the RNN.
# 

matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
bins = np.linspace(-100, 400, 100)
_ = plt.hist(m_jb_true[-bkg_flag], bins=bins, histtype='step', label='true signal', color='blue')
_ = plt.hist(m_jb_pred[-bkg_flag], bins=bins, histtype='step', label='RNN signal', color='limegreen')
_ = plt.hist(m_jb_mHmatch[-bkg_flag], bins=bins, histtype='step', label='mHmatch signal', color='red')
# _ = plt.hist(m_jb_pThigh[-bkg_flag], bins=bins, histtype='step', label='pThigh signal', color='cyan')
# _ = plt.hist(m_jb_pTjbhigh[-bkg_flag], bins=bins, histtype='step', label='pTjbhigh signal', color='magenta')

#_ = plt.hist(m_jb_true[bkg_flag], bins=bins, histtype='step', label='true bkg', color='blue', linestyle='--')
_ = plt.hist(m_jb_pred[bkg_flag], bins=bins, histtype='step', label='RNN bkg', color='limegreen', linestyle='--')
_ = plt.hist(m_jb_mHmatch[bkg_flag], bins=bins, histtype='step', label='mHmatch bkg', color='red', linestyle='--')
# _ = plt.hist(m_jb_pThigh[bkg_flag], bins=bins, histtype='step', label='pThigh bkg', color='cyan', linestyle='--')
# _ = plt.hist(m_jb_pTjbhigh[bkg_flag], bins=bins, histtype='step', label='pTjbhigh bkg', color='magenta', linestyle='--')

plt.xlim(0, 400)
plt.ylim(0, 6000)
#plt.vlines(125, 0, 6000, alpha=0.5)
plt.legend()


# Do we need to fit these curves to get the mass resolution?

# To do: add the BDT!
# 




