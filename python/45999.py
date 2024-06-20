# # Ideas for analysis
# 
# ## Neural data only
# 
# - raster plot of events
# - distribution of events/sec (across neurons)
# - distribution of events/sec (across time)
#     - chunk into windows? 
# 
# 
# ## Neural data + behavior
# - align neural data to specific behavior switch points
# 
# 

# # Combing data from multiple days / mice / conditions to create bigger data sets
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
from sklearn import preprocessing
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/new_master_8020_df.csv',index_col=0)


data.head(5)


data['Session ID'][0]


# ## Do one-hot encoding for mouse ID and session ID
# 

def encode_categorical(array):
    if not (array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64')) :
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array


categorical = (data.dtypes.values != np.dtype('float64'))
data_1hot = data.apply(encode_categorical)


# Apply one hot endcoing
encoder = preprocessing.OneHotEncoder(categorical_features=categorical, sparse=False)  # Last value in mask is y
x = encoder.fit_transform(data_1hot.values)


encoder.feature_indices_


encoder.active_features_


np.unique(data['Session ID'].values)


x.shape


# ## Try function
# 

data_encoded = bp.OneHotEncode(data)


master_matrix.to_csv(os.path.join(root_dir,'new_master_8020_df.csv'))


master_matrix


import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import bandit_preprocessing as bp
import sys
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


record = pd.read_csv('/Users/celiaberon/GitHub/mouse_bandit/session_record.csv',index_col=0)
ca_data = scio.loadmat('/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat',squeeze_me = True, struct_as_record = False)
neuron = ca_data['neuron_results'] 


record.head(5)


session_name  = '03142017_K7'
mouse_id = 'K7'

record[record['Session ID'] == session_name]


# # Extract data from specific session
# 

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame']

root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

full_name = session_name + '_trials.csv'

path_name = os.path.join(root_dir,full_name)

trial_df = pd.read_csv(path_name,names=columns)


trial_df.head(11)


# # convert to feature matrix
# 

feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)


feature_matrix.head(2)


feature_matrix[['10_Port','10_ITI','10_trialDuration']].head(5)


decisions = [0,1]
frames = ['center_frame','decision_frame']
imaging_frames = [] #initialize empty list
for decision in decisions:
    for frame in frames:
        imaging_frames.append(feature_matrix[((feature_matrix['Switch'] == 0) 
                                      & (feature_matrix['Decision'] == decision))][frame])


d_right = {'center_frame_right':imaging_frames[0],
     'decision_frame_right':imaging_frames[1],
    }
d_left = {'center_frame_left':imaging_frames[2],
     'decision_frame_left':imaging_frames[3]}

df_right = pd.DataFrame(data=d_right) #df_right.values[:,0]
df_left = pd.DataFrame(data=d_left)


switch_decision = feature_matrix[feature_matrix['Switch'] == 1]['decision_frame']
switch_center = feature_matrix[feature_matrix['Switch'] == 1]['center_frame']

preStart = stay_center_port0 -10
postDecision = stay_decision_port0 +10
trialDecision = stay_decision_port0


# ## function to get frames based on one or two conditions
# 

def extract_frames(df, cond1_name, cond1=False, cond2_name=False, cond2=False, frame_type='decision_frame'):
    if type(cond2_name)==str:
        frames = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))][frame_type])
        return frames
    else:
        frames =(df[(df[cond1_name] == cond1)][frame_type])
        return frames


cond1_name = 'Switch'
cond1_a = 1
cond1_b = 0
cond2_name = 'Decision'
cond2 = 0


frames_center_a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2, 'center_frame')
frames_decision_a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2, 'decision_frame')

frames_center_b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2, 'center_frame')
frames_decision_b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2, 'decision_frame')

#preStart_a = frames_center_a - 10 # start 10 frames before center poke
#postDecision_a = frames_decision_a + 10 # end 10 frames after decision poke

start_stop_times_a = [[frames_center_a - 10], [frames_decision_a + 10]] # start 10 frames before center poke
start_stop_times_b = [[frames_center_b - 10], [frames_decision_b + 10]] # start 10 frames before center poke




#plt.plot(neuron.C_raw[0, preStart:trialDecision])
nNeurons = neuron.C.shape[0]

# remove neurons that have NaNs
nan_neurons = np.where(np.isnan(neuron.C_raw))[0]
nan_neurons = np.unique(nan_neurons)
good_neurons = [x for x in range(0, nNeurons) if x not in nan_neurons]

nNeurons = len(good_neurons) # redefine number of neurons
nTrials = [len(start_stop_times_a[0][0]), len(start_stop_times_b[0][0])] # number of trials

# iterate through to determine duration between preStart and postDecision for each trial
window_length_a = []
window_length_b = []
for i in range(0,nTrials[0]):
    window_length_a.append((start_stop_times_a[1][0].iloc[i] - start_stop_times_a[0][0].iloc[i]))
for i in range(0,nTrials[1]):
    window_length_b.append((start_stop_times_b[1][0].iloc[i] - start_stop_times_b[0][0].iloc[i]))
window_length = [window_length_a, window_length_b]

# find longest window between preStart and postDecision and set as length for all trials
max_window = int(np.max((np.max(window_length[:]))))


start_stop_times = [start_stop_times_a, start_stop_times_b]
aligned_start = np.zeros((np.max(nTrials), max_window, nNeurons, 2))
mean_center_poke = np.zeros((max_window, nNeurons, 2))
norm_mean_center = np.zeros((mean_center_poke.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    count = 0
    for iNeuron in good_neurons:
        for iTrial in range(0,nTrials[i]):
            aligned_start[iTrial,0:max_window, count, i] = neuron.C_raw[iNeuron, int(start_stop_times[i][0][0].iloc[iTrial]):(int(start_stop_times[i][0][0].iloc[iTrial])+max_window)]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_center_poke[:,:,i]= np.mean(aligned_start[0:nTrials[i],:,:,i], axis=0)

    for iNeuron in range(0,nNeurons):
        norm_mean_center[:,iNeuron, i] = (mean_center_poke[:,iNeuron, i] - np.min(np.min(mean_center_poke, axis=0)[iNeuron][i]))/(np.max(np.max(mean_center_poke,axis=0)[iNeuron][i]) - np.min(np.min(mean_center_poke,axis=0)[iNeuron][i]))
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,i+1)  
    plt.imshow(np.transpose(mean_center_poke[:,:,i])), plt.colorbar()
    
    


# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
for i in [0,1]:
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,i+1)  
    plt.imshow(np.transpose(norm_mean_center[:,:,i])), plt.colorbar()
    plt.axvline(x=10, color='k', linestyle = '--', linewidth=.9)


# heatmap for calcium traces of a single neuron across all trials
# white dashed line for center poke time
# white vertical lines for decision poke time -- need something more subtle
sample_neuron = 10

plt.figure(figsize=(8,8))
plt.imshow(aligned_start[:,:,sample_neuron, 0])
plt.axvline(x=10, color='white', linestyle = '--', linewidth=.9)
#plt.scatter(trialDecision-preStart,range(0,nTrials), color='white', marker = '|', s=8)


# ## aligned to decision poke
# 

aligned_decision = np.zeros((np.max(nTrials), max_window, nNeurons, 2))
mean_decision = np.zeros((max_window, nNeurons, 2))
norm_mean_decision = np.zeros((mean_decision.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to decision poke
    count = 0
    for iNeuron in good_neurons:
        for iTrial in range(0,nTrials[i]):
            aligned_decision[iTrial,0:max_window, count, i] = neuron.C_raw[iNeuron, int(start_stop_times[i][0][0].iloc[iTrial])-max_window:(int(start_stop_times[i][0][0].iloc[iTrial]))]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_decision[:,:,i]= np.mean(aligned_decision[0:nTrials[i],:,:,i], axis=0)

    for iNeuron in range(0,nNeurons):
        norm_mean_decision[:,iNeuron, i] = (mean_decision[:,iNeuron, i] - np.min(np.min(mean_decision, axis=0)[iNeuron][i]))/(np.max(np.max(mean_decision,axis=0)[iNeuron][i]) - np.min(np.min(mean_decision,axis=0)[iNeuron][i]))
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,i+1)  
    plt.imshow(np.transpose(mean_decision[:,:,i])), plt.colorbar()


# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
for i in [0,1]:
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,i+1) 
    plt.imshow(np.transpose(norm_mean_decision[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window-10, color='k', linestyle = '--', linewidth=.9)


# ## plot the difference between two different conditions
# 

# plot the difference between two conditions for aligned to center poke
plt.imshow(np.transpose(norm_mean_center[:,:,0] - norm_mean_center[:,:,1])), plt.colorbar()
plt.axvline(x=10, color='white', linestyle = '--', linewidth=.9)


# plot the difference between two conditions for aligned to decision poke
plt.imshow(np.transpose(norm_mean_decision[:,:,0] - norm_mean_decision[:,:,1])), plt.colorbar()
plt.axvline(x=max_window-10, color='white', linestyle = '--', linewidth=.9)


import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import bandit_preprocessing as bp
import support_functions as sf
get_ipython().magic('matplotlib inline')


# ### predict belief using HMM on feature matrix
# #### input variables:
#         data = data in the format of feature matrix
#         n_plays = number of trials (up to 10) that will make up the memory of model (int)
# #### output variables:
#         master_beliefs = value of belief (range 0 to 1) that the system is in the right port state (p(z=0)) (nx1 numpy array, n=number of trials)
# 

def predictBeliefFeatureMat(data, n_plays, p=0.9, duration=60):
    
    #data = pd.read_csv(data_path, index_col=0) 
    
    """
    Initialize port and reward identities
    """
    
    port_features = []
    reward_features = []

    #change right port to -1 instead of 0
    for col in data:
        if '_Port' in col:
            port_features.append(col)
        elif '_Reward' in col:
            reward_features.append(col)

    '''
    Tuned parameters
    '''
    #duration = 60
    #p = 0.9 # prob of reward if choose the correct side
    q = 1.0-p # prob of reward if choose the incorrect side
    
    '''
    Set up outcome & transition matrices T such that T[i,j] is the probability of transitioning
    from state i to state j. 
    If the true number of trials before switching is 'duration', then set the
    probability of switching to be 1 / duration, and the probability of 
    staying to 1 - 1 / duration
    '''
    
    s = 1 - 1./duration
    T = np.array([[s, 1.0-s],
                 [1.0-s,s]])

    #observation array
    '''
    set up array such that O[r,z,a] = Pr(reward=r | state=z,action=a)

    eg. when action = L, observation matrix should be:
    O[:,:,1]    = [P(r=0 | z=0,a=0),  P(r=0 | z=1,a=0)
                   P(r=1 | z=0,a=0),  P(r=1 | z=1,a=0)]
                = [1-p, 1-q
                    p,   q]
    '''
    O = np.zeros((2,2,2))
    # let a 'right' choice be represented by '0'
    O[:,:,0] = np.array([[1.0-p, 1.0-q],
                         [p,q]])
    O[:,:,1] = np.array([[1.0-q, 1.0-p],
                         [q,p]])

    #TEST: All conditional probability distributions must sum to one
    assert np.allclose(O.sum(0),1), "All conditional probability distributions must sum to one!"
    
    """
    Run model on data and output predicted belief state based on evidence from past 10 trials
    """
    
    #set test data
    data_test = data.copy()

    n_trials = data_test.shape[0]

    #initialize prediction array
    y_predict = np.zeros(n_trials)
    likeli = []
    master_beliefs = np.zeros(n_trials)

    for trial in range(data_test.shape[0]):
        curr_trial = data_test.iloc[trial]
        actions = curr_trial[port_features].values
        rewards = curr_trial[reward_features].values
        beliefs = np.nan*np.ones((n_plays+1,2))
        beliefs[0] = [0.5,0.5] #initialize both sides with equal probability
        #run the algorithm
        for play in range(n_plays):

            assert np.allclose(beliefs[play].sum(), 1.0), "Beliefs must sum to one!"

            #update neg log likelihood
            likeli.append(-1*np.log(beliefs[play,int(actions[play])]))

            #update beliefs for next play
            #step 1: multiply by p(r_t | z_t = k, a_t)
            belief_temp = O[int(rewards[play]),:,int(actions[play])] * beliefs[play]

            #step 2: sum over z_t, weighting by transition matrix
            beliefs[play+1] = T.dot(belief_temp)

            #step 3: normalize
            beliefs[play+1] /= beliefs[play+1].sum()

        #predict action
        y_predict[trial] = np.where(beliefs[-1] == beliefs[-1].max())[0][0]
        master_beliefs[trial] = beliefs[-1][0]
        
    return master_beliefs


# # Visualizing the mouse behavior
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sklearn.linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import tree as Tree
import sklearn.tree
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/new_master_8020_df.csv',index_col=0)


data.head(2)


np.unique(data['Reward Streak'])


data[data['Reward Streak'] == 0]['Switch'].mean()


# ## separate back into individual days
# 

data_dumble = data[data['Mouse ID'] == 'dumble']
data_harry = data[data['Mouse ID'] == 'harry']
datas = []

for s in np.unique(data['Session ID'].values):
    datas.append(data_dumble[data_dumble['Session ID'] == s])
    datas.append(data_harry[data_harry['Session ID'] == s])


# # p(switch) | switched in ith previous trial
# 

p_switch = np.zeros(20)
for i in np.arange(0,20):
    p_switch[i] = data.iloc[np.where(data['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


p_switchy = np.zeros((len(datas),20))

for s,d in enumerate(datas):
    for i in np.arange(0,20):
        p_switchy[s,i] = d.iloc[np.where(d['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


errors = p_switchy.std(axis=0) / np.sqrt(len(p_switchy))


sns.set_style('white')
plt.figure(figsize=(5,5))
for s in range(20):
    if (s%2 == 0):
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='blue')
    else:
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='green')
        
plt.hlines(y=data['Switch'].mean(axis=0),xmin=0,xmax=20,color='black',alpha=1,linewidth=2,linestyles='dotted',label='average')
plt.plot(np.arange(1,20),p_switch[1:],color='black',linewidth=1.5)
plt.fill_between(np.arange(1,20),p_switch[1:]+errors[1:],p_switch[1:]-errors[1:],color='grey')
plt.xlim(0.5,19)
plt.ylim(0,0.5)
plt.xlabel('# trials from switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


# # Looking more closely at the switch trials
# 

switches = data['Switch'].values


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_a = np.zeros_like(streak)*0.0
p_switch_b = np.zeros_like(streak)*0.0

for i,s in enumerate(streak): 
        p_switch_a[i] = data[(data['Port Streak'] >= 5) & (data['Reward Streak'] == s)]['Switch'].mean()
        p_switch_b[i] = data[(data['Port Streak'] < 5) & (data['Reward Streak'] == s)]['Switch'].mean()


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_indi_a = np.zeros((len(datas),streak.shape[0]))
p_switch_indi_b = np.zeros_like(p_switch_indi_a)

for j,d in enumerate(datas):
    for i,s in enumerate(streak): 
            p_switch_indi_a[j,i] = d[(d['Port Streak'] >= 5) & (d['Reward Streak'] == s)]['Switch'].mean()
            p_switch_indi_b[j,i] = d[(d['Port Streak'] < 5) & (d['Reward Streak'] == s)]['Switch'].mean()


errors_a = np.nanstd(p_switch_indi_a,axis=0) / np.sqrt(p_switch_indi_a.shape[0])
errors_b = np.nanstd(p_switch_indi_b,axis=0) / np.sqrt(p_switch_indi_a.shape[0])


plt.figure(figsize=(5,5))
#plt.vlines(x=0,ymin=0,ymax=1,color='white',linewidth=60,zorder=3)
plt.plot(streak,p_switch_a,label='Port Streak >=5',linewidth=3,zorder=1,color='purple')
plt.fill_between(streak,p_switch_a+errors_a,p_switch_a-errors_a,color='purple',alpha=0.2)
plt.plot(streak,p_switch_b,label='Port Streak <5',linewidth=3,zorder=2,color='green')
plt.fill_between(streak,p_switch_b+errors_b,p_switch_b-errors_b,color='green',alpha=0.2)
plt.xticks(np.arange(3,-8,-1),streak,fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(3,-8)
plt.ylim(-0.1,1.1)
plt.legend(loc='upper left',fontsize=15)
sns.despine()
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('Reward Streak',fontsize=20)


# # Switches when 1_Reward = 0
# 

plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 1)]['Port Streak'],color='black',alpha=0.4,normed=True,label='Switch Trials')
plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 0)]['Port Streak'],color='red',alpha=0.4,normed=True,label='Stay Trials')
plt.title('Distribution of Port Streaks\nWhen the last reward = 0')
plt.ylabel('Frequency')
plt.xlabel('Port Streak')
plt.legend(loc='upper left')
plt.xlim(0,9)
plt.ylim(0,0.5)


plt.hist(data[data['Switch']==1]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


plt.hist(data[data['Port Streak'] <= 3]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


data[data['Port Streak'] > 5]['Switch'].mean()


p_switch


p_switch = np.zeros(10)*0.0
avg = data['Switch'].mean()

for i,s in enumerate(np.arange(1,11)):
    p_switch[i] = data[data['Port Streak'] == s]['Switch'].mean()

p_switches = np.zeros((20,10))*0.0
p_switches_R = np.zeros((20,10))*0.0
p_switches_nR = np.zeros((20,10))*0.0
for j,d in enumerate(datas):
    for i,s in enumerate(np.arange(1,11)):
        p_switches[j,i] = d[d['Port Streak'] == s]['Switch'].mean()
        p_switches_R[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==1)]['Switch'].mean()
        p_switches_nR[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==0)]['Switch'].mean()

errors = p_switches.std(axis=0) / np.sqrt(p_switches.shape[0])
errors_R = np.nanstd(p_switches_R,axis=0) / np.sqrt(p_switches.shape[0])
errors_nR = np.nanstd(p_switches_nR,axis=0) / np.sqrt(p_switches.shape[0])
p_switch_R = np.nanmean(p_switches_R,axis=0)
p_switch_nR = np.nanmean(p_switches_nR,axis=0)


plt.figure(figsize=(5,5))

plt.plot(np.arange(1,11),p_switch,color='black',linewidth=3,label='All Trials')
plt.fill_between(np.arange(1,11),p_switch+errors,p_switch-errors,color='black',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_R,color='green',linewidth=3,label='Previous Trial Rewarded')
plt.fill_between(np.arange(1,11),p_switch_R+errors_R,p_switch_R-errors_R,color='green',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_nR,color='blue',linewidth=3,label='Previous Trial Not Rewarded')
plt.fill_between(np.arange(1,11),p_switch_nR+errors_nR,p_switch_nR-errors_nR,color='blue',alpha=0.5)

plt.hlines(y=avg,xmin=1,xmax=10,linestyle='dotted')
plt.ylim(0,0.7)
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('# trials since previous switch',fontsize=20)
plt.legend(loc='upper right',fontsize=15)
plt.xticks(np.arange(1,11),[0,1,2,3,4,5,6,7,8,'>8','>9'],fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


c = 0
cs = np.zeros(10)
for j,i in enumerate(np.arange(1,11)):
    c +=  data[data['Port Streak'] == i]['Switch'].sum()/data['Switch'].sum()
    cs[j] = c


plt.figure(figsize=(5,5))
plt.plot(np.arange(1,11),cs)
plt.ylim(0,1)
plt.ylabel('% of switches')
plt.xlabel('Port Streak')


data.shape


t_block_unique = np.unique(data['Block Trial'].values)
p_switch_block = np.zeros((t_block_unique.shape[0],2))
high_p_port = np.zeros_like(p_switch_block)
trial_block_count = np.zeros_like(t_block_unique)

for t in t_block_unique:
    p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
    trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
    p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])
    
    high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
    high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])
    


x_end=65
plt.figure(figsize=(15,5))
plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

plt.subplot(131)
plt.plot(t_block_unique,p_switch_block[:,0],color='black')
plt.fill_between(t_block_unique,p_switch_block[:,0]+p_switch_block[:,1],p_switch_block[:,0]-p_switch_block[:,1],color='grey',alpha=0.5)
plt.hlines(data['Switch'].mean(axis=0),xmin=0,xmax=x_end,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.legend(loc='upper right')
plt.xlim(0,x_end)
plt.ylim(0,0.25)
plt.xlabel('block trial #',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) vs block trial',fontsize=20)

plt.subplot(132)
plt.hist(data.iloc[np.where(data['Block Trial']==0)[0]-1]['Block Trial'],bins=20,color='grey')
plt.title('distribution of block lengths',fontsize=20)
plt.xlabel('# of trials taken to get 50 rewards',fontsize=20)
plt.ylabel('count',fontsize=20)

plt.subplot(133)
plt.plot(t_block_unique,trial_block_count,color='black')
plt.title('# of data points for each trial #',fontsize=20)
plt.ylabel('# of data points',fontsize=20)
plt.xlabel('block trial #',fontsize=20)
plt.xlim(0,x_end)

plt.tight_layout()
print('total # of blocks in dataset: ~%.0f' % (np.sum(data['Block Trial']==0)))



data.index = np.arange(data.shape[0])


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


switch_points = data[data['Block Trial'] == 0 ].index.values

L = 15
paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
paraswitch_port = np.zeros_like(paraswitch)

for i,point in enumerate(switch_points):
    paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
    paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port'] 


u = paraswitch.mean(axis=0)
s = paraswitch.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=0.5,color='black',linestyle='dotted')
plt.hlines(data['Switch'].mean(axis=0),xmin=-1*L,xmax=L+1,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L)
plt.ylim(0,0.25)

plt.subplot(122)
u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)

plt.tight_layout()


u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(5,5))
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


# little note on the for loop below. 
# 
# took me little while because I had the order of the else-if statements wrong. 
# 
# when block_trial == 0 needs to come BEFORE whether the block trial incremented by 1 or not (which is my hokey way of detecting when a new session started where block_trial does not equal 0. 
# 
# I suppose a better way would be to detect when the block trial is 11 AND the previous block trial != 10. that would work. okay. switched it to that now. 
# 

block_reward = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    trial = data.iloc[i]
    
    #lets deal with weird cases first:
    #1) the first row
    if(i == 0):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    
    #3) the first trial of a new block
    elif (trial['Block Trial'] == 0):
        block_reward[i] = 0
    
    #2) the first trial of a new session
    elif (((trial['Block Trial'] - trial_prev['Block Trial']) != 1) and (trial['Block Trial'] == 11)):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    else:
        block_reward[i] = block_reward[i-1] + trial['Reward']
    
    trial_prev = trial


reward_switches = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterR = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterNoR = np.zeros(np.unique(block_reward).shape[0])
for i,r_block in enumerate(np.unique(block_reward)):
    reward_switches[i] = data[block_reward == r_block]['Switch'].mean()
    reward_switches_afterR[i] = data[((block_reward == r_block) & (data['1_Reward']==1))]['Switch'].mean()
    reward_switches_afterNoR[i] = data[((block_reward == r_block) & (data['1_Reward']==0))]['Switch'].mean()


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(np.unique(block_reward),reward_switches,color='black',label='all trials')
plt.plot(np.unique(block_reward),reward_switches_afterR,color='green',label='after rewarded trials')
plt.plot(np.unique(block_reward),reward_switches_afterNoR,color='purple',label='after non-rewarded trials')
plt.xlabel('Reward number')
plt.ylabel('p(switch)')
plt.legend(loc='upper right')
plt.xlim(-1,51)
plt.ylim(-0.01,0.5)
sns.despine()

plt.subplot(122)
plt.hist(block_reward,bins=51,color='grey')
plt.title('Histogram of reward numbers within a block')
plt.xlabel('Reward Number')
plt.ylabel('Count')


# # Chapter 4 Figure 0
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().magic('matplotlib inline')


record = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)


mice = ['K7','K13','Q43','Q45','dumble','harry']
mice = np.unique(record['Mouse ID'].values)
#mice = np.delete(mice,np.where(mice=='dumble'))
#mice = np.delete(mice,np.where(mice=='harry'))
#mice = np.delete(mice,np.where(mice=='K4'))
#mice = np.delete(mice,np.where(mice=='K10'))
#mice = np.delete(mice,np.where(mice=='K11'))
mice = np.delete(mice,np.where(mice=='q45'))
mice = np.delete(mice,np.where(mice=='q43'))
#mice = np.delete(mice,np.where(mice=='quirrel'))
mice = np.delete(mice,np.where(mice=='sprout'))
mice = np.delete(mice,np.where(mice=='tom'))
#mice = np.delete(mice,np.where(mice=='tonks'))
#mice = np.delete(mice,np.where(mice=='volde'))
#mice = np.delete(mice,np.where(mice=='K9'))
#mice = np.delete(mice,np.where(mice=='K7'))
#mice = np.delete(mice,np.where(mice=='myrtle'))

print(mice.shape[0])
sns.set_style('white')
plt.figure(figsize=(5,4))
y = np.zeros((mice.shape[0]-9,10))
x = np.arange(0,10)
k = 0 
for i,mouse in enumerate(mice):
    y_temp = record[((record['Mouse ID'] == mouse) & ((record['Left Reward Prob'] == 0.8) | (record['Left Reward Prob'] == 0.2)))]['p(high Port)'].values
    try:
        y[k,:] = y_temp[-10:]
        plt.plot(x,y[k,:],label=mouse,alpha=0.3,linewidth=3)
        plt.scatter(x,y[k,:],label=mouse,alpha=0.3,s=100,color='black')
        k+=1
    except:
        print(mouse)

plt.plot(x,y.mean(axis=0),color='black',linewidth=3)
err = y.std(axis=0)/np.sqrt(7)
plt.fill_between(x,y1=y.mean(axis=0)+err,y2=y.mean(axis=0)-err,color='black',alpha=0.3)
plt.ylim(0,1)
plt.xlim(0,7)
plt.xticks(fontsize=20)
plt.yticks([0,0.5,1],fontsize=20)
plt.xlabel('Day',fontsize=20)
plt.ylabel('fraction higher prob\nport chosen',fontsize=20)
#plt.legend(loc='best')
sns.despine(top='True')


columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward']
data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data/07132016_harry_trials.csv',names=columns)


block_start_trials = np.where(np.abs(np.diff(data['Right Reward Prob'].values))!=0)
block_start_times = data['Elapsed Time (s)'].values[block_start_trials[0]]

num_trials = 1600
sns.set_style('white')
plt.figure(figsize=(15,4))
plt.vlines(block_start_times,ymin=0,ymax=3,linestyle='dotted')
plt.scatter(data[data['Reward'] == 0]['Elapsed Time (s)'].values[:num_trials],
            data[data['Reward'] == 0]['Port Poked'].values[:num_trials],color='black',s=200,alpha=0.7)
plt.scatter(data[data['Reward'] == 1]['Elapsed Time (s)'].values[:num_trials],
            data[data['Reward'] == 1]['Port Poked'].values[:num_trials],color='green',s=200,alpha=0.7)
plt.xticks(np.arange(0,1700,60),list(map(int,np.arange(0,1700/60))),fontsize=20)
plt.yticks([1,2],['Right Port','Left Port'],fontsize=20)
plt.xlim(-1,1201)
plt.xlabel('Time (min)',fontsize=20)
plt.ylim(0.8,2.2)
sns.despine(left=True)
#fig_name = '/Users/shayneufeld/Dropbox/Thesis/CHPT4/Figures/singlesession.eps'
#plt.savefig(fig_name, format='eps', dpi=1000)


data90 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/full_9010_02192017.csv',index_col=0)
data80 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/full_8020_02192017.csv',index_col=0)
data70 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/full_7030_02192017.csv',index_col=0)
data90['Condition'] = '90-10'
data80['Condition'] = '80-20'
data70['Condition'] = '70-30'
datas = data90.append(data80)
datas = datas.append(data70)


datas = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/master_data.csv',index_col=0)


datas.head(100)


#ax2 = sns.barplot(x='Condition',y='Higher p port',data=datas)
plt.figure(figsize=(5,4))
ax1 = sns.barplot(x='Condition',y='Higher p port',data=datas[datas['Condition'] != '100-0'],hue='Mouse ID')
plt.yticks([0,0.5,1.0],fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('')
plt.xlabel('')
ax1.legend_.remove()
sns.despine()


session_ids = np.unique(datas['Session ID'].values)
#session_ids = ['01182017_q43']
block_trials_ = np.array([])
for session in session_ids:
    data = datas[datas['Session ID']==session].copy()
    block_ends = data['Trial'].values[np.where(data['Block Trial'].values==0)[0]]
    
    for trial in data.iterrows():
        try:
            block_trial = block_ends[np.where(block_ends > trial[1]['Trial'])[0][0]] - trial[1]['Block Trial']
        except:
            block_trial = 0
        block_trials_ = np.append(block_trials_,block_trial)


datas['Block Trial Rev'] = block_trials_


block_trials = np.unique(datas['Block Trial'].values)
conditions = np.unique(datas['Condition'].values)
mice = np.unique(datas['Mouse ID'].values)
p = np.zeros((block_trials.shape[0],2))
trial_stats = pd.DataFrame(data=None)
for condition in conditions:
        for i,trial in enumerate(block_trials):
            d = datas[datas['Condition']==condition]
            #d = d[d['Mouse ID']==mouse]
            d = d[d['Block Trial']==trial]
            '''
            boolean = (((datas['Block Trial'] == trial) 
                    & (datas['Condition']==condition))
                    & (datas['Mouse ID'] == mouse))
            
            t = datas[boolean]['Higher p port'].values
            '''
            t = d['Higher p port'].values
            p[i,0] = t.mean()
            p[i,1] = t.std()/np.sqrt(t.shape[0])

            d = {'Condition':condition,'trial':trial,
                 'mean':t.mean(),'sem':t.std()/np.sqrt(t.shape[0]),'n':t.shape[0]}
            trial_stats = trial_stats.append(pd.DataFrame(data=d,index=[0]))


block_trials_rev = np.unique(datas['Block Trial Rev'].values)
conditions = np.unique(datas['Condition'].values)
p = np.zeros((block_trials_rev.shape[0],2))
trial_stats_ = pd.DataFrame(data=None)

for condition in conditions:
    for i,trial in enumerate(block_trials_rev):
        t = datas[((datas['Block Trial Rev'] == trial) 
                   & (datas['Condition']==condition))]['Higher p port'].values
        p[i,0] = t.mean()
        p[i,1] = t.std()/np.sqrt(t.shape[0])
    
        d = {'Condition':condition,'trial':trial,'mean':t.mean(),'sem':t.std()/np.sqrt(t.shape[0]),'n':t.shape[0]}
        trial_stats_ = trial_stats_.append(pd.DataFrame(data=d,index=[0]))
        
trial_stats_['trial'] = trial_stats_['trial']*-1


trial_stats = trial_stats[trial_stats['n'] > 30]
trial_stats_ = trial_stats_[trial_stats_['n'] > 30]

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.swarmplot(x='trial',y='n',hue='Condition',data=trial_stats)
plt.xlim(0,100)
plt.xticks([0,50,100],[0,50,100])

plt.subplot(122)
sns.swarmplot(x='trial',y='n',hue='Condition',data=trial_stats_)
plt.xlim(-100,0)
plt.xticks([0,-50,-100],[0,-50,-100])


colors = ['red','green','blue','purple']
conditions=['90-10','80-20','70-30']

for i,condition in enumerate(conditions):
    trial_stat = trial_stats[trial_stats['Condition']==condition]
    trial_stat_ = trial_stats_[trial_stats_['Condition']==condition]
    u = trial_stat['mean'].values
    e = trial_stat['sem'].values
    x = trial_stat['trial'].values
    
    u_ = trial_stat_['mean'].values
    e_ = trial_stat_['sem'].values
    x_ = trial_stat_['trial'].values
    
    plt.plot(x,u,color=colors[i],alpha=0.5,label=condition)
    plt.fill_between(x,y1=u-e,y2=u+e,color=colors[i],alpha=0.3)
    
    plt.plot(x_,u_,color=colors[i],alpha=0.5)
    plt.fill_between(x_,y1=u_-e_,y2=u_+e_,color=colors[i],alpha=0.3)
    
plt.vlines(x=0,ymin=0,ymax=1,linestyle='dotted',label='p(reward) switched')

plt.xlim(-20,50)
plt.legend(loc='lower right')


conditions = ['90-10','80-20','70-30']
u_avg_ = np.zeros(3)
e_avg_ = np.zeros(3)
for i,condition in enumerate(conditions):
    trials = trial_stats_[((trial_stats_['trial'] > -15) 
                           & (trial_stats_['trial'] < 0)
                           & (trial_stats_['Condition']==condition))]
    
    u_avg_[i] = trials['mean'].mean()
    e_avg_[i] = np.sqrt(np.sum(trials['sem'].values**2) / (trials.shape[0]-1))


np.where(u >= u_avg_[2])


stats = pd.DataFrame()
for c in conditions:
    for mouse in mice:
        d = datas[((datas['Condition'] == c) & (datas['Mouse ID'] == mouse))]
        if (d['Session ID'].unique().shape[0] > 1):
            s = bp.extract_session_stats(d)
            s['mouse'] = mouse
            s['condition'] = c
            s['n_trials'] = d.shape[0]
            s['n_sessions'] = d['Session ID'].unique().shape[0]
            stats = stats.append(s)


stats


plt.figure(figsize=(10,5))

plt.subplot(121)
sns.pointplot(x='condition',y='stable_phigh',hue='mouse',data=stats)
plt.legend(bbox_to_anchor=(1.5,1))
plt.ylim(0.5,1.05)
plt.title('stable fraction better port chosen')

plt.subplot(122)
sns.boxplot(x='condition',y='stable_phigh',data=stats)
sns.swarmplot(x='condition',y='stable_phigh',data=stats,color='.25')
plt.legend(bbox_to_anchor=(1.5,1))
plt.ylim(0.5,1.05)
plt.title('stable fraction better port chosen')


plt.figure(figsize=(10,5))

plt.subplot(121)
sns.pointplot(x='condition',y='peak_pswitch',hue='mouse',data=stats)
plt.legend(bbox_to_anchor=(1.5,1))
plt.ylim(0,.5)
plt.title('peak fraction switch trials following block switch')

plt.subplot(122)
sns.boxplot(x='condition',y='peak_pswitch',data=stats)
sns.swarmplot(x='condition',y='peak_pswitch',data=stats,color='.25')
plt.legend(bbox_to_anchor=(1.5,1))
plt.ylim(0,.5)
plt.title('peak fraction switch trials following block switch')


plt.figure(figsize=(10,5))

plt.subplot(121)
sns.pointplot(x='condition',y='rebias_tau',hue='mouse',data=stats)
plt.legend(bbox_to_anchor=(1.5,1))
#plt.ylim(0.5,1.05)
plt.title('peak fraction switch trials following block switch')

plt.subplot(122)
sns.boxplot(x='condition',y='rebias_tau',data=stats)
sns.swarmplot(x='condition',y='rebias_tau',data=stats,color='0.25')
plt.legend(bbox_to_anchor=(1.5,1))
#plt.ylim(0.5,1.05)
plt.title('peak fraction switch trials following block switch')


# # Comparing mouse behavior to HMM with Thompson Sampling
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data_100 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/reduced_1000_02132017.csv',index_col=0)
data_90 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/reduced_9010_02132017.csv',index_col=0)
data_80 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/reduced_8020_02132017.csv',index_col=0)
data_70 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/reduced_7030_02132017.csv',index_col=0)


datas = [data_90]


# # Looking more closely at the switch trials
# 

sns.set_style('white')
fig = plt.figure(figsize=(10,5))
plt.suptitle('70-30 reward probabilities',x=0.5,y=1.2,fontsize=30)

labels = ['90-10','80-20','70-30']
colors = ['red','green','navy']

for c,data in enumerate(datas):

    switches = data['Switch'].values

    t_block_unique = np.unique(data['Block Trial'].values)
    p_switch_block = np.zeros((t_block_unique.shape[0],2))
    high_p_port = np.zeros_like(p_switch_block)
    trial_block_count = np.zeros_like(t_block_unique)

    for t in t_block_unique:
        p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
        trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
        p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])

        high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
        high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])


    data.index = np.arange(data.shape[0]) # <-- this is important
    switch_points = data[data['Block Trial'] == 0].index.values

    L = 15
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        try:
            paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
            paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port']
        except:
            pass

    u = paraswitch.mean(axis=0)
    s = paraswitch.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])

    #plt.figure(figsize=(15,5))
    #plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

    plt.subplot(122)
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.2)
    plt.vlines(x=0,ymin=0,ymax=0.5,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
    plt.yticks([0,0.1,0.2,0.3],fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim(-1*L,L)
    plt.ylim(0,0.305)

    plt.subplot(121)
    u = paraswitch_port.mean(axis=0)
    s = paraswitch_port.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c],label=labels[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.2)
    #plt.vlines(x=0,ymin=0,ymax=1,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(high reward port)',fontsize=20)
    plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L+10)
    plt.xticks(fontsize=15)
    plt.yticks([0,0.25,0.5,0.75,1],fontsize=15)
    plt.ylim(0,1)
    #plt.legend(bbox_to_anchor=(1.05,0.5),fontsize=20)

    plt.tight_layout()


fig.savefig('temp.png', transparent=True)


# # Comparing mouse behavior to HMM
# 
# ## This notebook extracts the following behavioral features to compare with the HMM model:
# 
# 1. probability/rate of choosing 'correct' port at the end of a block (ie at its most stable point)
# 2. time constant of rate/probability of choosing the 'correct' port following a block switch
# 3. peak probability/rate of switching at the beginning of a block
# 
# Approach:
# 
# 1. load in feature matrix for each session
# 2. Try and extract the 3 features and append columns to a copy of the record DB
# 3. Calculate averages for each condition & mouse
# 4. store in dataframe <-- maybe want to do some filtering here to take out really bad performances?
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import os
from scipy import optimize
get_ipython().magic('matplotlib inline')


# ## function *extract_session_stats(data)*
# Takes in a feature matrix and returns the 3 stats we want to compare to the hmm
# 

def extract_session_stats(data):
    '''
    Inputs:
        data - (pandas dataframe) feature matrix (reduced or not)
    
    Outputs:
        dataframe with 3 columns:
            stable_phigh- prob/rate of choosing high p port at end of block (last 10 trials)
            peak_pswitch- prob/rate of switching at beginning of block (first 10 trials)
            rebias_tau-   time constant of exponential function fit to p(high p port) after block switch
    '''
    
    
    #all the block numbers
    t_block_unique = np.unique(data['Block Trial'].values)

    # initialize matrix for p(switch) at every trial number in block. 2nd column for SEM
    p_switch_block = np.zeros((t_block_unique.shape[0],2))

    # initialize matrix for p(high_p_port)
    high_p_port = np.zeros_like(p_switch_block)

    '''
    calculate p(switch) for each trial # in block (from 0 -> end)
    '''
    for t in t_block_unique:
        switches = data[data['Block Trial'] == t]['Switch']
        p_switch_block[t,0] = switches.mean(axis=0)
        p_switch_block[t,1] = switches.std(axis=0) / np.sqrt(switches.shape[0])
        
        highport = data[data['Block Trial']==t]['Higher p port']
        high_p_port[t,0] = highport.mean(axis=0)
        high_p_port[t,1] = highport.std(axis=0) / np.sqrt(highport.shape[0])


    '''
    calculate p(switch) and p(high port) for trial #s in block (from -L to +L)
    '''

    data.index = np.arange(data.shape[0]) # <-- this is important
    switch_points = data[data['Block Trial'] == 0].index.values

    L = 30
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 1))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        try:
            paraswitch[i,:] = data.iloc[point-L:point+L+1]['Switch']
            paraswitch_port[i,:] = data.iloc[point-L:point+L+1]['Higher p port']
        except:
            pass

    '''
    calculate exponential fit for p(high port) after block switch
    '''
    #first define exponential function to be optimized
    def exp_func(x,a,b,c):
        return a*np.exp(-b*x) + c
    
    #fit curve
    popt,pcov = optimize.curve_fit(exp_func,np.arange(L+1),paraswitch_port.mean(axis=0)[L:])
    
    #calc peak_switch, stable_phigh, and tau
    peak_switch = paraswitch[:,L:L+10].mean(axis=0).max()
    stable_phigh = paraswitch_port[:,L-10:L].mean()
    rebias_tau = 1./popt[1]
    
    d = {'stable_phigh':stable_phigh,'peak_pswitch':peak_switch,'rebias_tau':rebias_tau}
    
    return pd.DataFrame(data=d,index=[0])


# ## Load in data database
# 
# We don't want to consider *every* session. Let's make the following requirements:
# 1. It's in Phase 2 (not phase 1, which does not require switches)
# 2. No. Blocks >= 4
# 3. No. Rewards > 100
# 4. p(high Port) > 0.5
# 

#load in data base
db = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)

'''
select the sessions we want
'''

#only phase 2
db_s = db[(db['Phase'] == 2)].copy()
# No. Blocks >= 4
#db_s = db_s[db_s['No. Blocks'] >= 4].copy()
# No. Rewards > 100
#db_s = db_s[db_s['No. Rewards'] > 100].copy()
# p(high Port) 0.5
#db_s = db_s[db_s['p(high Port)'] > 0.5].copy()
# only 50 reward blocks
db_s = db_s[db_s['Block Range Min'] == 50].copy()

session_names = db_s['Session ID'].values
mouse_ids = db_s['Mouse ID'].values
session_ids = db_s['Session ID'].values
dates = db_s['Date'].values
conditions = db_s['Left Reward Prob'].values
phigh = db_s['p(high Port)'].values
nrewards = db_s['No. Rewards'].values


root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/feature_data'
failures = []
for i,session in enumerate(session_ids):
    
    #load in the feature matrix
    full_name = session + '_features.csv'
    path_name = os.path.join(root_dir,full_name)
    session_matrix = pd.read_csv(path_name,index_col=0)
    #try calculating the stats with extract_stats
    try:
        stats = extract_session_stats(session_matrix)

        #append them to a master dataframe
        stats['Mouse ID'] = mouse_ids[i]
        stats['Date'] = dates[i]
        stats['Condition'] = conditions[i]
        stats['p(high Port)'] = phigh[i]
        stats['No. Rewards'] = nrewards[i]

        if i == 0:
            stats_df = stats.copy()
        else:
            stats_df = stats_df.append(stats)
    
    except:
        failures.append(session)
        


#clean up conditions
stats_df['Condition'] = stats_df['Condition'].replace(to_replace=0,value=1.0)
stats_df['Condition'] = stats_df['Condition'].replace(to_replace=0.1,value=0.9)
stats_df['Condition'] = stats_df['Condition'].replace(to_replace=0.2,value=0.8)
stats_df['Condition'] = stats_df['Condition'].replace(to_replace=0.3,value=0.7)


# ## Data visualization
# 
# start with Dumble. 
# track his performance over days, 
# 

sns.set_style('white')
mice = np.unique(stats_df['Mouse ID'].values)

for mouse in mice:
    d = stats_df[stats_df['Mouse ID'] == mouse].copy()
    d.index = np.arange(d.shape[0])
    plt.figure(figsize=(15,3))
    plt.suptitle(mouse,fontsize=20,x=0.5,y=1.1)
    colors = ['black','green','purple','orange']
    for i,c in enumerate([1.0,0.9,0.8,0.7]):
        dates = d[d['Condition']==c]['Date'].values
        dates = dates.astype('str')
        x = d[d['Condition']==c].index.values

        h1 = plt.subplot(131)
        plt.plot(x,d[d['Condition']==c]['p(high Port)'].values,color=colors[i])
        plt.scatter(x,d[d['Condition']==c]['p(high Port)'].values,label=c,color=colors[i])
        plt.ylim(0,1)
        plt.title('Stable p(high)')

        h2 = plt.subplot(132)
        plt.plot(x,d[d['Condition']==c]['peak_pswitch'].values,color=colors[i])
        plt.scatter(x,d[d['Condition']==c]['peak_pswitch'].values,label=c,color=colors[i])
        plt.ylim(0,1)
        plt.title('Peak p(switch)')

        h3 = plt.subplot(133)
        plt.plot(x,d[d['Condition']==c]['rebias_tau'].values,color=colors[i])
        plt.scatter(x,d[d['Condition']==c]['rebias_tau'].values,label=c,color=colors[i])
        plt.ylim(0,30)
        plt.title('Rebias Tau')

    #h1.set_xticklabels(q43['Date'].values,rotation='vertical')
    #h2.set_xticklabels(q43['Date'].values,rotation='vertical')
    #h3.set_xticklabels(q43['Date'].values,rotation='vertical')
    plt.legend(loc='best')
plt.tight_layout()


q43['Date'].values


stats_90 = stats_df[stats_df['Condition']==0.9].copy()
stats_80 = stats_df[stats_df['Condition']==0.8].copy()
stats_70 = stats_df[stats_df['Condition']==0.7].copy()


plt.figure(figsize=(15,5))
plt.suptitle('prob choose high port',fontsize=20,x=0.5,y=1.1)
for i,s in enumerate([stats_90,stats_80,stats_70]):
    plt.subplot(1,3,i+1)
    sns.boxplot(x='Mouse ID',y='p(high Port)',data=s)
    plt.ylim(0.5,1)
plt.tight_layout()


# # Creating a feature matrix for each session
# 
# This is useful to a number of things. In general, easier to create a folder where each session has a feature matrix. Then we can append different feature matrices (by mouse / date etc...) 
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
from sklearn import preprocessing
import sys
import os
get_ipython().magic('matplotlib inline')


# ## Load in record & and get names of each session
# 

#load in data base
db = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)
session_names = db['Session ID'].values
mouse_ids = db['Mouse ID'].values
session_ids = db['Session ID'].values


# ## load in trial csv files (from running exportTrials.m)
# 

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data'
save_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/feature_data'

for i,session in enumerate(session_names):
    #get name of trial csv
    full_name = session + '_trials.csv'
    #get name of full patch to trial data
    path_name = os.path.join(root_dir,full_name)
    #read in trial csv file
    trial_df = pd.read_csv(path_name,names=columns)
    #create reduced feature matrix
    feature_matrix = bp.create_reduced_feature_matrix(trial_df,mouse_ids[i],session_ids[i],feature_names='Default')
    #create file name to be saved
    save_name = session + '_features.csv'
    #save feature matrix for this session
    feature_matrix.to_csv(os.path.join(save_dir,save_name))


# # Visualizing the mouse behavior
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sklearn.linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import tree as Tree
import sklearn.tree
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

record = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)

r_8020 = record[((record['Left Reward Prob'] == 0.8) |  (record['Right Reward Prob'] == 0.8))].copy()
r_8020 = r_8020[r_8020['p(high Port)'] > 0.85].copy()
r_8020 = r_8020[r_8020['No. Blocks'] > 3].copy()
r_8020 = r_8020[r_8020['Block Range Min'] == 50].copy()
r_8020 = r_8020[r_8020['Mouse ID'] == 'harry'].copy()

r_8020


columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward']


data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data/07132016_harry_trials.csv',names=columns)


data.head(100)





block_start_trials = np.where(np.abs(np.diff(data['Right Reward Prob'].values))!=0)
block_start_times = data['Elapsed Time (s)'].values[block_start_trials[0]]
block_start_times


time_mins = data['Elapsed Time (s)'].values / 60.0


num_trials = 1600
sns.set_style('white')
plt.figure(figsize=(22,5))
plt.vlines(block_start_times,ymin=0,ymax=3,linestyle='dotted')
plt.scatter(data[data['Reward'] == 0]['Elapsed Time (s)'].values[:num_trials],
            data[data['Reward'] == 0]['Port Poked'].values[:num_trials],color='black',s=200,alpha=0.7)
plt.scatter(data[data['Reward'] == 1]['Elapsed Time (s)'].values[:num_trials],
            data[data['Reward'] == 1]['Port Poked'].values[:num_trials],color='green',s=200,alpha=0.7)
plt.xticks(np.arange(0,1700,60),list(map(int,np.arange(0,1700/60))),fontsize=30)
plt.yticks([1,2],['Right Port','Left Port'],fontsize=30)
plt.xlim(-1,1201)
plt.xlabel('Time (s)',fontsize=30)
plt.ylim(0.8,2.2)
sns.despine(left=True)
fig_name = '/Users/shayneufeld/Dropbox/Thesis/CHPT4/Figures/singlesession.eps'
plt.savefig(fig_name, format='eps', dpi=1000)


data[58:100]


# # Visualizing the mouse behavior
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sklearn.linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import tree as Tree
import sklearn.tree
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/new_master_8020_df.csv',index_col=0)


data.head(2)


np.unique(data['Reward Streak'])


data[data['Reward Streak'] == 0]['Switch'].mean()


# ## separate back into individual days
# 

data_dumble = data[data['Mouse ID'] == 'dumble']
data_harry = data[data['Mouse ID'] == 'harry']
datas = []

for s in np.unique(data['Session ID'].values):
    datas.append(data_dumble[data_dumble['Session ID'] == s])
    datas.append(data_harry[data_harry['Session ID'] == s])


# # p(switch) | switched in ith previous trial
# 

p_switch = np.zeros(20)
for i in np.arange(0,20):
    p_switch[i] = data.iloc[np.where(data['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


p_switchy = np.zeros((len(datas),20))

for s,d in enumerate(datas):
    for i in np.arange(0,20):
        p_switchy[s,i] = d.iloc[np.where(d['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


errors = p_switchy.std(axis=0) / np.sqrt(len(p_switchy))


sns.set_style('white')
plt.figure(figsize=(5,5))
for s in range(20):
    if (s%2 == 0):
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='blue')
    else:
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='green')
        
plt.hlines(y=data['Switch'].mean(axis=0),xmin=0,xmax=20,color='black',alpha=1,linewidth=2,linestyles='dotted',label='average')
plt.plot(np.arange(1,20),p_switch[1:],color='black',linewidth=1.5)
plt.fill_between(np.arange(1,20),p_switch[1:]+errors[1:],p_switch[1:]-errors[1:],color='grey')
plt.xlim(0.5,19)
plt.ylim(0,0.5)
plt.xlabel('# trials from switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


# # Looking more closely at the switch trials
# 

switches = data['Switch'].values


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_a = np.zeros_like(streak)*0.0
p_switch_b = np.zeros_like(streak)*0.0

for i,s in enumerate(streak): 
        p_switch_a[i] = data[(data['Port Streak'] >= 5) & (data['Reward Streak'] == s)]['Switch'].mean()
        p_switch_b[i] = data[(data['Port Streak'] < 5) & (data['Reward Streak'] == s)]['Switch'].mean()


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_indi_a = np.zeros((len(datas),streak.shape[0]))
p_switch_indi_b = np.zeros_like(p_switch_indi_a)

for j,d in enumerate(datas):
    for i,s in enumerate(streak): 
            p_switch_indi_a[j,i] = d[(d['Port Streak'] >= 5) & (d['Reward Streak'] == s)]['Switch'].mean()
            p_switch_indi_b[j,i] = d[(d['Port Streak'] < 5) & (d['Reward Streak'] == s)]['Switch'].mean()


errors_a = np.nanstd(p_switch_indi_a,axis=0) / np.sqrt(p_switch_indi_a.shape[0])
errors_b = np.nanstd(p_switch_indi_b,axis=0) / np.sqrt(p_switch_indi_a.shape[0])


plt.figure(figsize=(5,5))
#plt.vlines(x=0,ymin=0,ymax=1,color='white',linewidth=60,zorder=3)
plt.plot(streak,p_switch_a,label='Port Streak >=5',linewidth=3,zorder=1,color='purple')
plt.fill_between(streak,p_switch_a+errors_a,p_switch_a-errors_a,color='purple',alpha=0.2)
plt.plot(streak,p_switch_b,label='Port Streak <5',linewidth=3,zorder=2,color='green')
plt.fill_between(streak,p_switch_b+errors_b,p_switch_b-errors_b,color='green',alpha=0.2)
plt.xticks(np.arange(3,-8,-1),streak,fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(3,-8)
plt.ylim(-0.1,1.1)
plt.legend(loc='upper left',fontsize=15)
sns.despine()
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('Reward Streak',fontsize=20)


# # Switches when 1_Reward = 0
# 

plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 1)]['Port Streak'],color='black',alpha=0.4,normed=True,label='Switch Trials')
plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 0)]['Port Streak'],color='red',alpha=0.4,normed=True,label='Stay Trials')
plt.title('Distribution of Port Streaks\nWhen the last reward = 0')
plt.ylabel('Frequency')
plt.xlabel('Port Streak')
plt.legend(loc='upper left')
plt.xlim(0,9)
plt.ylim(0,0.5)


plt.hist(data[data['Switch']==1]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


plt.hist(data[data['Port Streak'] <= 3]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


data[data['Port Streak'] > 5]['Switch'].mean()


p_switch


p_switch = np.zeros(10)*0.0
avg = data['Switch'].mean()

for i,s in enumerate(np.arange(1,11)):
    p_switch[i] = data[data['Port Streak'] == s]['Switch'].mean()

p_switches = np.zeros((20,10))*0.0
p_switches_R = np.zeros((20,10))*0.0
p_switches_nR = np.zeros((20,10))*0.0
for j,d in enumerate(datas):
    for i,s in enumerate(np.arange(1,11)):
        p_switches[j,i] = d[d['Port Streak'] == s]['Switch'].mean()
        p_switches_R[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==1)]['Switch'].mean()
        p_switches_nR[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==0)]['Switch'].mean()

errors = p_switches.std(axis=0) / np.sqrt(p_switches.shape[0])
errors_R = np.nanstd(p_switches_R,axis=0) / np.sqrt(p_switches.shape[0])
errors_nR = np.nanstd(p_switches_nR,axis=0) / np.sqrt(p_switches.shape[0])
p_switch_R = np.nanmean(p_switches_R,axis=0)
p_switch_nR = np.nanmean(p_switches_nR,axis=0)


plt.figure(figsize=(5,5))

plt.plot(np.arange(1,11),p_switch,color='black',linewidth=3,label='All Trials')
plt.fill_between(np.arange(1,11),p_switch+errors,p_switch-errors,color='black',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_R,color='green',linewidth=3,label='Previous Trial Rewarded')
plt.fill_between(np.arange(1,11),p_switch_R+errors_R,p_switch_R-errors_R,color='green',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_nR,color='blue',linewidth=3,label='Previous Trial Not Rewarded')
plt.fill_between(np.arange(1,11),p_switch_nR+errors_nR,p_switch_nR-errors_nR,color='blue',alpha=0.5)

plt.hlines(y=avg,xmin=1,xmax=10,linestyle='dotted')
plt.ylim(0,0.7)
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('# trials since previous switch',fontsize=20)
plt.legend(loc='upper right',fontsize=15)
plt.xticks(np.arange(1,11),[0,1,2,3,4,5,6,7,8,'>8','>9'],fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


c = 0
cs = np.zeros(10)
for j,i in enumerate(np.arange(1,11)):
    c +=  data[data['Port Streak'] == i]['Switch'].sum()/data['Switch'].sum()
    cs[j] = c


plt.figure(figsize=(5,5))
plt.plot(np.arange(1,11),cs)
plt.ylim(0,1)
plt.ylabel('% of switches')
plt.xlabel('Port Streak')


data.shape


t_block_unique = np.unique(data['Block Trial'].values)
p_switch_block = np.zeros((t_block_unique.shape[0],2))
high_p_port = np.zeros_like(p_switch_block)
trial_block_count = np.zeros_like(t_block_unique)

for t in t_block_unique:
    p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
    trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
    p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])
    
    high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
    high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])
    


x_end=65
plt.figure(figsize=(15,5))
plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

plt.subplot(131)
plt.plot(t_block_unique,p_switch_block[:,0],color='black')
plt.fill_between(t_block_unique,p_switch_block[:,0]+p_switch_block[:,1],p_switch_block[:,0]-p_switch_block[:,1],color='grey',alpha=0.5)
plt.hlines(data['Switch'].mean(axis=0),xmin=0,xmax=x_end,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.legend(loc='upper right')
plt.xlim(0,x_end)
plt.ylim(0,0.25)
plt.xlabel('block trial #',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) vs block trial',fontsize=20)

plt.subplot(132)
plt.hist(data.iloc[np.where(data['Block Trial']==0)[0]-1]['Block Trial'],bins=20,color='grey')
plt.title('distribution of block lengths',fontsize=20)
plt.xlabel('# of trials taken to get 50 rewards',fontsize=20)
plt.ylabel('count',fontsize=20)

plt.subplot(133)
plt.plot(t_block_unique,trial_block_count,color='black')
plt.title('# of data points for each trial #',fontsize=20)
plt.ylabel('# of data points',fontsize=20)
plt.xlabel('block trial #',fontsize=20)
plt.xlim(0,x_end)

plt.tight_layout()
print('total # of blocks in dataset: ~%.0f' % (np.sum(data['Block Trial']==0)))


data.index = np.arange(data.shape[0])


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


switch_points = data[data['Block Trial'] == 0 ].index.values

L = 15
paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
paraswitch_port = np.zeros_like(paraswitch)

for i,point in enumerate(switch_points):
    paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
    paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port'] 


u = paraswitch.mean(axis=0)
s = paraswitch.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=0.5,color='black',linestyle='dotted')
plt.hlines(data['Switch'].mean(axis=0),xmin=-1*L,xmax=L+1,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L)
plt.ylim(0,0.25)

plt.subplot(122)
u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)

plt.tight_layout()


u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(5,5))
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


# little note on the for loop below. 
# 
# took me little while because I had the order of the else-if statements wrong. 
# 
# when block_trial == 0 needs to come BEFORE whether the block trial incremented by 1 or not (which is my hokey way of detecting when a new session started where block_trial does not equal 0. 
# 
# I suppose a better way would be to detect when the block trial is 11 AND the previous block trial != 10. that would work. okay. switched it to that now. 
# 

block_reward = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    trial = data.iloc[i]
    
    #lets deal with weird cases first:
    #1) the first row
    if(i == 0):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    
    #3) the first trial of a new block
    elif (trial['Block Trial'] == 0):
        block_reward[i] = 0
    
    #2) the first trial of a new session
    elif (((trial['Block Trial'] - trial_prev['Block Trial']) != 1) and (trial['Block Trial'] == 11)):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    else:
        block_reward[i] = block_reward[i-1] + trial['Reward']
    
    trial_prev = trial


reward_switches = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterR = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterNoR = np.zeros(np.unique(block_reward).shape[0])
for i,r_block in enumerate(np.unique(block_reward)):
    reward_switches[i] = data[block_reward == r_block]['Switch'].mean()
    reward_switches_afterR[i] = data[((block_reward == r_block) & (data['1_Reward']==1))]['Switch'].mean()
    reward_switches_afterNoR[i] = data[((block_reward == r_block) & (data['1_Reward']==0))]['Switch'].mean()


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(np.unique(block_reward),reward_switches,color='black',label='all trials')
plt.plot(np.unique(block_reward),reward_switches_afterR,color='green',label='after rewarded trials')
plt.plot(np.unique(block_reward),reward_switches_afterNoR,color='purple',label='after non-rewarded trials')
plt.xlabel('Reward number')
plt.ylabel('p(switch)')
plt.legend(loc='upper right')
plt.xlim(-1,51)
plt.ylim(-0.01,0.5)
sns.despine()

plt.subplot(122)
plt.hist(block_reward,bins=51,color='grey')
plt.title('Histogram of reward numbers within a block')
plt.xlabel('Reward Number')
plt.ylabel('Count')


data


# # Visualizing the mouse behavior
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sklearn.linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import tree as Tree
import sklearn.tree
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/hmm_matrix_full_8020_greedy.csv',index_col=0)


data.head(2)


np.unique(data['Reward Streak'])


data[data['Reward Streak'] == -1]['Switch'].mean()


# ## separate back into individual days
# 

datas = data


# # p(switch) | switched in ith previous trial
# 

p_switch = np.zeros(20)
for i in np.arange(0,20):
    p_switch[i] = data.iloc[np.where(data['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


p_switchy = np.zeros(20)

for i in np.arange(0,20):
    p_switchy[i] = data.iloc[np.where(data['Switch'].values == 1)[0]-i]['Switch'].mean(axis=0)


errors = p_switchy.std(axis=0) / np.sqrt(len(p_switchy))


sns.set_style('white')
plt.figure(figsize=(5,5))
for s in range(20):
    if (s%2 == 0):
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='blue')
    else:
        plt.plot(np.arange(1,20),p_switchy[s,1:],alpha=0.05,linewidth=7,color='green')
        
plt.hlines(y=data['Switch'].mean(axis=0),xmin=0,xmax=20,color='black',alpha=1,linewidth=2,linestyles='dotted',label='average')
plt.plot(np.arange(1,20),p_switch[1:],color='black',linewidth=1.5)
plt.fill_between(np.arange(1,20),p_switch[1:]+errors[1:],p_switch[1:]-errors[1:],color='grey')
plt.xlim(0.5,19)
plt.ylim(0,0.5)
plt.xlabel('# trials from switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


# # Looking more closely at the switch trials
# 

switches = data['Switch'].values


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_a = np.zeros_like(streak)*0.0
p_switch_b = np.zeros_like(streak)*0.0

for i,s in enumerate(streak): 
        p_switch_a[i] = data[(data['Port Streak'] >= 5) & (data['Reward Streak'] == s)]['Switch'].mean()
        p_switch_b[i] = data[(data['Port Streak'] < 5) & (data['Reward Streak'] == s)]['Switch'].mean()


streak = np.array([3,2,1,-1,-2,-3,-4,-5,-6,-7,-8])
port_streaks = np.arange(0,6)
p_switch_indi_a = np.zeros((len(datas),streak.shape[0]))
p_switch_indi_b = np.zeros_like(p_switch_indi_a)

for j,d in enumerate(datas):
    for i,s in enumerate(streak): 
            p_switch_indi_a[j,i] = d[(d['Port Streak'] >= 5) & (d['Reward Streak'] == s)]['Switch'].mean()
            p_switch_indi_b[j,i] = d[(d['Port Streak'] < 5) & (d['Reward Streak'] == s)]['Switch'].mean()


errors_a = np.nanstd(p_switch_indi_a,axis=0) / np.sqrt(p_switch_indi_a.shape[0])
errors_b = np.nanstd(p_switch_indi_b,axis=0) / np.sqrt(p_switch_indi_a.shape[0])


plt.figure(figsize=(5,5))
#plt.vlines(x=0,ymin=0,ymax=1,color='white',linewidth=60,zorder=3)
plt.plot(streak,p_switch_a,label='Port Streak >=5',linewidth=3,zorder=1,color='purple')
plt.fill_between(streak,p_switch_a+errors_a,p_switch_a-errors_a,color='purple',alpha=0.2)
plt.plot(streak,p_switch_b,label='Port Streak <5',linewidth=3,zorder=2,color='green')
plt.fill_between(streak,p_switch_b+errors_b,p_switch_b-errors_b,color='green',alpha=0.2)
plt.xticks(np.arange(3,-8,-1),streak,fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(3,-8)
plt.ylim(-0.1,1.1)
plt.legend(loc='upper left',fontsize=15)
sns.despine()
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('Reward Streak',fontsize=20)


# # Switches when 1_Reward = 0
# 

plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 1)]['Port Streak'],color='black',alpha=0.4,normed=True,label='Switch Trials')
plt.hist(data[(data['1_Reward'] == 0) & (data['Switch'] == 0)]['Port Streak'],color='red',alpha=0.4,normed=True,label='Stay Trials')
plt.title('Distribution of Port Streaks\nWhen the last reward = 0')
plt.ylabel('Frequency')
plt.xlabel('Port Streak')
plt.legend(loc='upper left')
plt.xlim(0,9)
plt.ylim(0,0.5)


plt.hist(data[data['Switch']==1]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


plt.hist(data[data['Port Streak'] <= 3]['Port Streak'],normed=True,alpha=0.4)
plt.hist(data[data['Switch']==0]['Port Streak'],normed=True,alpha=0.4)


data[data['Port Streak'] > 5]['Switch'].mean()


p_switch


p_switch = np.zeros(10)*0.0
avg = data['Switch'].mean()

for i,s in enumerate(np.arange(1,11)):
    p_switch[i] = data[data['Port Streak'] == s]['Switch'].mean()

p_switches = np.zeros((20,10))*0.0
p_switches_R = np.zeros((20,10))*0.0
p_switches_nR = np.zeros((20,10))*0.0
for j,d in enumerate(datas):
    for i,s in enumerate(np.arange(1,11)):
        p_switches[j,i] = d[d['Port Streak'] == s]['Switch'].mean()
        p_switches_R[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==1)]['Switch'].mean()
        p_switches_nR[j,i] = d[(d['Port Streak'] == s) & (d['1_Reward']==0)]['Switch'].mean()

errors = p_switches.std(axis=0) / np.sqrt(p_switches.shape[0])
errors_R = np.nanstd(p_switches_R,axis=0) / np.sqrt(p_switches.shape[0])
errors_nR = np.nanstd(p_switches_nR,axis=0) / np.sqrt(p_switches.shape[0])
p_switch_R = np.nanmean(p_switches_R,axis=0)
p_switch_nR = np.nanmean(p_switches_nR,axis=0)


plt.figure(figsize=(5,5))

plt.plot(np.arange(1,11),p_switch,color='black',linewidth=3,label='All Trials')
plt.fill_between(np.arange(1,11),p_switch+errors,p_switch-errors,color='black',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_R,color='green',linewidth=3,label='Previous Trial Rewarded')
plt.fill_between(np.arange(1,11),p_switch_R+errors_R,p_switch_R-errors_R,color='green',alpha=0.5)

plt.plot(np.arange(1,11),p_switch_nR,color='blue',linewidth=3,label='Previous Trial Not Rewarded')
plt.fill_between(np.arange(1,11),p_switch_nR+errors_nR,p_switch_nR-errors_nR,color='blue',alpha=0.5)

plt.hlines(y=avg,xmin=1,xmax=10,linestyle='dotted')
plt.ylim(0,0.7)
plt.ylabel('p(switch)',fontsize=20)
plt.xlabel('# trials since previous switch',fontsize=20)
plt.legend(loc='upper right',fontsize=15)
plt.xticks(np.arange(1,11),[0,1,2,3,4,5,6,7,8,'>8','>9'],fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


c = 0
cs = np.zeros(10)
for j,i in enumerate(np.arange(1,11)):
    c +=  data[data['Port Streak'] == i]['Switch'].sum()/data['Switch'].sum()
    cs[j] = c


plt.figure(figsize=(5,5))
plt.plot(np.arange(1,11),cs)
plt.ylim(0,1)
plt.ylabel('% of switches')
plt.xlabel('Port Streak')


data.shape


t_block_unique = np.unique(data['Block Trial'].values)
p_switch_block = np.zeros((t_block_unique.shape[0],2))
high_p_port = np.zeros_like(p_switch_block)
trial_block_count = np.zeros_like(t_block_unique)

for t in t_block_unique:
    p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
    trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
    p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])
    
    high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
    high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])
    


x_end=65
plt.figure(figsize=(15,5))
plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

plt.subplot(131)
plt.plot(t_block_unique,p_switch_block[:,0],color='black')
plt.fill_between(t_block_unique,p_switch_block[:,0]+p_switch_block[:,1],p_switch_block[:,0]-p_switch_block[:,1],color='grey',alpha=0.5)
plt.hlines(data['Switch'].mean(axis=0),xmin=0,xmax=x_end,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.legend(loc='upper right')
plt.xlim(0,x_end)
plt.ylim(0,0.25)
plt.xlabel('block trial #',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) vs block trial',fontsize=20)

plt.subplot(132)
plt.hist(data.iloc[np.where(data['Block Trial']==0)[0]-1]['Block Trial'],bins=20,color='grey')
plt.title('distribution of block lengths',fontsize=20)
plt.xlabel('# of trials taken to get 50 rewards',fontsize=20)
plt.ylabel('count',fontsize=20)

plt.subplot(133)
plt.plot(t_block_unique,trial_block_count,color='black')
plt.title('# of data points for each trial #',fontsize=20)
plt.ylabel('# of data points',fontsize=20)
plt.xlabel('block trial #',fontsize=20)
plt.xlim(0,x_end)

plt.tight_layout()
print('total # of blocks in dataset: ~%.0f' % (np.sum(data['Block Trial']==0)))


data.index = np.arange(data.shape[0])


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


switch_points = data[data['Block Trial'] == 0 ].index.values

L = 15
paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
paraswitch_port = np.zeros_like(paraswitch)

for i,point in enumerate(switch_points):
    paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
    paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port'] 


u = paraswitch.mean(axis=0)
s = paraswitch.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=0.5,color='black',linestyle='dotted')
plt.hlines(data['Switch'].mean(axis=0),xmin=-1*L,xmax=L+1,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(switch)',fontsize=20)
plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L)
plt.ylim(0,0.25)

plt.subplot(122)
u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)

plt.tight_layout()


u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(5,5))
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


# little note on the for loop below. 
# 
# took me little while because I had the order of the else-if statements wrong. 
# 
# when block_trial == 0 needs to come BEFORE whether the block trial incremented by 1 or not (which is my hokey way of detecting when a new session started where block_trial does not equal 0. 
# 
# I suppose a better way would be to detect when the block trial is 11 AND the previous block trial != 10. that would work. okay. switched it to that now. 
# 

block_reward = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    trial = data.iloc[i]
    
    #lets deal with weird cases first:
    #1) the first row
    if(i == 0):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    
    #3) the first trial of a new block
    elif (trial['Block Trial'] == 0):
        block_reward[i] = 0
    
    #2) the first trial of a new session
    elif (((trial['Block Trial'] - trial_prev['Block Trial']) != 1) and (trial['Block Trial'] == 11)):
        block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
    else:
        block_reward[i] = block_reward[i-1] + trial['Reward']
    
    trial_prev = trial


reward_switches = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterR = np.zeros(np.unique(block_reward).shape[0])
reward_switches_afterNoR = np.zeros(np.unique(block_reward).shape[0])
for i,r_block in enumerate(np.unique(block_reward)):
    reward_switches[i] = data[block_reward == r_block]['Switch'].mean()
    reward_switches_afterR[i] = data[((block_reward == r_block) & (data['1_Reward']==1))]['Switch'].mean()
    reward_switches_afterNoR[i] = data[((block_reward == r_block) & (data['1_Reward']==0))]['Switch'].mean()


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(np.unique(block_reward),reward_switches,color='black',label='all trials')
plt.plot(np.unique(block_reward),reward_switches_afterR,color='green',label='after rewarded trials')
plt.plot(np.unique(block_reward),reward_switches_afterNoR,color='purple',label='after non-rewarded trials')
plt.xlabel('Reward number')
plt.ylabel('p(switch)')
plt.legend(loc='upper right')
plt.xlim(-1,51)
plt.ylim(-0.01,0.5)
sns.despine()

plt.subplot(122)
plt.hist(block_reward,bins=51,color='grey')
plt.title('Histogram of reward numbers within a block')
plt.xlabel('Reward Number')
plt.ylabel('Count')


data


# # Visualizing the mouse behavior
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import sklearn.linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import tree as Tree
import sklearn.tree
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data90 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/master_9010_df.csv',index_col=0)
data80 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/new_master_8020_df.csv',index_col=0)
data70 = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/master_7030_df.csv',index_col=0)


data90.head(2)


data80.head(2)


data70.head(2)


sns.set_style('white')


datas = [data80,data70]

for data in datas:
    t_block_unique = np.unique(data['Block Trial'].values)
    p_switch_block = np.zeros((t_block_unique.shape[0],2))
    high_p_port = np.zeros_like(p_switch_block)
    trial_block_count = np.zeros_like(t_block_unique)

    for t in t_block_unique:
        p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
        trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
        p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])

        high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
        high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])


    x_end=90
    plt.figure(figsize=(15,5))
    plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

    plt.subplot(131)
    plt.scatter(t_block_unique,p_switch_block[:,0],color='black')
    plt.fill_between(t_block_unique,p_switch_block[:,0]+p_switch_block[:,1],p_switch_block[:,0]-p_switch_block[:,1],color='grey',alpha=0.5)
    plt.hlines(data['Switch'].mean(axis=0),xmin=0,xmax=x_end,color='red',linestyle='dotted',label='avg',linewidth=2)
    plt.legend(loc='upper right')
    plt.xlim(0,x_end)
    plt.ylim(0,0.25)
    plt.xlabel('block trial #',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    plt.title('p(switch) vs block trial',fontsize=20)

    plt.subplot(132)
    plt.hist(data.iloc[np.where(data['Block Trial']==0)[0]-1]['Block Trial'],bins=20,color='grey')
    plt.title('distribution of block lengths',fontsize=20)
    plt.xlabel('# of trials taken to get 50 rewards',fontsize=20)
    plt.ylabel('count',fontsize=20)
    plt.xlim(0,100)

    plt.subplot(133)
    plt.plot(t_block_unique,trial_block_count,color='black')
    plt.title('# of data points for each trial #',fontsize=20)
    plt.ylabel('# of data points',fontsize=20)
    plt.xlabel('block trial #',fontsize=20)
    plt.xlim(0,x_end)

    plt.tight_layout()
    print('total # of blocks in dataset: ~%.0f' % (np.sum(data['Block Trial']==0)))


for data in datas:

    data.index = np.arange(data.shape[0])

    switch_points = data[data['Block Trial'] == 0 ].index.values

    L = 15
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
        paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port'] 

    u = paraswitch.mean(axis=0)
    s = paraswitch.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])
    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.scatter(np.arange(-1*L,L+10),u,color='black')
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
    plt.vlines(x=0,ymin=0,ymax=0.5,color='black',linestyle='dotted')
    plt.hlines(data['Switch'].mean(axis=0),xmin=-1*L,xmax=L+1,color='red',linestyle='dotted',label='avg',linewidth=2)
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L)
    plt.ylim(0,0.25)

    plt.subplot(122)
    u = paraswitch_port.mean(axis=0)
    s = paraswitch_port.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])
    plt.scatter(np.arange(-1*L,L+10),u,color='black')
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
    plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
    plt.hlines(0.82,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(high reward port)',fontsize=20)
    plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L+10)
    plt.ylim(0,1)

    plt.tight_layout()


u = paraswitch_port.mean(axis=0)
s = paraswitch_port.std(axis=0)
SE = s/np.sqrt(paraswitch.shape[0])
plt.figure(figsize=(5,5))
plt.plot(np.arange(-1*L,L+10),u,color='black')
plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color='grey',alpha=0.5)
plt.vlines(x=0,ymin=0,ymax=1,color='black',linestyle='dotted')
plt.hlines(0.92,xmin=-1*L,xmax=L+10,color='red',linestyle='dotted',label='avg',linewidth=2)
plt.xlabel('Trial # from block switch',fontsize=20)
plt.ylabel('p(high reward port)',fontsize=20)
plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
plt.xlim(-1*L,L+10)
plt.ylim(0,1)


switch_points = data[data['Block Trial'] == 0 ].index.values
switch_points


# little note on the for loop below. 
# 
# took me little while because I had the order of the else-if statements wrong. 
# 
# when block_trial == 0 needs to come BEFORE whether the block trial incremented by 1 or not (which is my hokey way of detecting when a new session started where block_trial does not equal 0. 
# 
# I suppose a better way would be to detect when the block trial is 11 AND the previous block trial != 10. that would work. okay. switched it to that now. 
# 

for data in datas:

    block_reward = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        trial = data.iloc[i]

        #lets deal with weird cases first:
        #1) the first row
        if(i == 0):
            block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                    '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()

        #3) the first trial of a new block
        elif (trial['Block Trial'] == 0):
            block_reward[i] = 0

        #2) the first trial of a new session
        elif (((trial['Block Trial'] - trial_prev['Block Trial']) != 1) and (trial['Block Trial'] == 11)):
            block_reward[i] = trial[['10_Reward','9_Reward','8_Reward','7_Reward','6_Reward',
                                    '5_Reward','4_Reward','3_Reward','2_Reward','1_Reward','Reward']].sum()
        else:
            block_reward[i] = block_reward[i-1] + trial['Reward']

        trial_prev = trial

    reward_switches = np.zeros(np.unique(block_reward).shape[0])
    reward_switches_afterR = np.zeros(np.unique(block_reward).shape[0])
    reward_switches_afterNoR = np.zeros(np.unique(block_reward).shape[0])
    for i,r_block in enumerate(np.unique(block_reward)):
        reward_switches[i] = data[block_reward == r_block]['Switch'].mean()
        reward_switches_afterR[i] = data[((block_reward == r_block) & (data['1_Reward']==1))]['Switch'].mean()
        reward_switches_afterNoR[i] = data[((block_reward == r_block) & (data['1_Reward']==0))]['Switch'].mean()

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(np.unique(block_reward),reward_switches,color='black',label='all trials')
    plt.plot(np.unique(block_reward),reward_switches_afterR,color='green',label='after rewarded trials')
    plt.plot(np.unique(block_reward),reward_switches_afterNoR,color='purple',label='after non-rewarded trials')
    plt.xlabel('Reward number')
    plt.ylabel('p(switch)')
    plt.legend(loc='upper right')
    plt.xlim(-1,51)
    plt.ylim(-0.01,0.5)
    sns.despine()

    plt.subplot(122)
    plt.hist(block_reward,bins=51,color='grey')
    plt.title('Histogram of reward numbers within a block')
    plt.xlabel('Reward Number')
    plt.ylabel('Count')


# # Creating a feature matrix from a DB Query
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
from sklearn import preprocessing
import sys
import os
get_ipython().magic('matplotlib inline')


# ## Retrieve names of sessions you want from the DB
# 

#load in data base
db = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)


db[db['Session ID']=='03262017_K13']


# ### Query all 80-20 sessions where performance > 0.7 and block structure was 50
# 

conditions = ['100-0','90-10','80-20','70-30']
probs = [1,0.9,0.8,0.7]

for condition,prob in zip(conditions,probs):
    print(condition)
    print(prob)


master_matrix = np.zeros(1)

conditions = ['100-0','90-10','80-20','70-30']
probs = [1,0.9,0.8,0.7]

for condition,prob in zip(conditions,probs):
    r = db[((db['Left Reward Prob'] == prob) |  (db['Right Reward Prob'] == prob))].copy()
    r = r[r['p(high Port)'] > prob-0.1].copy()
    r = r[r['Block Range Min'] == 50].copy()
    r['Condition'] = condition
    session_names = r['Session ID'].values
    conditions_ = r['Condition'].values
    
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

    root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data'

    trial_df = []

    #load in trials
    for session in session_names:
        full_name = session + '_trials.csv'

        path_name = os.path.join(root_dir,full_name)

        trial_df.append(pd.read_csv(path_name,names=columns))

    mouse_ids = r['Mouse ID'].values
    
    
    #create feature matrix
    for i,df in enumerate(trial_df):
    
        curr_feature_matrix = bp.create_feature_matrix(df,10,mouse_ids[i],session_names[i],feature_names='Default')
        curr_feature_matrix['Condition'] = condition

        if master_matrix.shape[0]==1:
            master_matrix = curr_feature_matrix.copy()
        else:
            master_matrix = master_matrix.append(curr_feature_matrix)


master_matrix.index = np.arange(master_matrix.shape[0])

master_matrix[master_matrix['Session ID']=='03242017_K13'].index.values


m=master_matrix.copy()


m = m.drop(master_matrix[master_matrix['Session ID']=='03242017_K13'].index.values)


master_matrix = m.copy()


# ## Save combined feature matrix  
# 

master_matrix.to_csv(os.path.join(root_dir,'master_data.csv'))


root_dir


# # Combing data from multiple days / mice / conditions to create bigger data sets
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

'''
load in trial data
'''
columns = ['Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

trial_df = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/trials_hmm_full_7030_greedy.csv',names=columns)


trial_df['Since last trial (s)'] = 0
trial_df['Trial Duration (s)'] = 0


trial_df.head(2)


feature_matrix = bp.create_reduced_feature_matrix(trial_df,'hmm_7030_greedy','02272017_2')


feature_matrix.head(5)


feature_matrix.to_csv('hmm_matrix_full_7030_greedy.csv')


# # Combing data from multiple days / mice / conditions to create bigger data sets
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
from sklearn import preprocessing
import sys
import os
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/70_30_trial_data'

trial_df = []
mouse_ids = []
session_ids = []
for file in os.listdir(root_dir):
    if not file[0] == '.':
        file_name = os.path.join(root_dir,file)
        trial_df.append(pd.read_csv(file_name,names=columns))
        mouse_ids.append(file[:file.index('_')])
        session_ids.append(file[file.index('_')+1:file.index('_')+7])


# ## convert into 1 feature matrix
# 

for i,df in enumerate(trial_df):
    
    curr_feature_matrix = bp.create_feature_matrix(df,10,mouse_ids[i],session_ids[i],feature_names='Default')
    
    if i == 0:
        master_matrix = curr_feature_matrix.copy()
    else:
        master_matrix = master_matrix.append(curr_feature_matrix)
    


master_matrix.shape


master_matrix.head(30)


# ## Do one-hot encoding for mouse ID and session ID
# 

def encode_categorical(array):
    if not (array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64')) :
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array


categorical = (master_matrix.dtypes.values != np.dtype('float64'))
master_matrix_1hot = master_matrix.apply(encode_categorical)


# Apply one hot endcoing
encoder = OneHotEncoder(categorical_features=categorical, sparse=False)  # Last value in mask is y
x = encoder.fit_transform(master_matrix.values)


# ## Save combined feature matrix  
# 

master_matrix.to_csv(os.path.join(root_dir,'master_7030_df.csv'))


master_matrix


# # Comparing mouse behavior to HMM with Thompson Sampling
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
get_ipython().magic('matplotlib inline')


# ## load in csv files (from running exportTrials.m)
# 

data_hmm_g = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/hmm_matrix_full_7030_greedy.csv',index_col=0)
data_mouse = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/reduced_7030_02132017.csv',index_col=0)
data_hmm_t = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/hmm_matrix_7030.csv',index_col=0)

datas = [data_hmm_g,data_hmm_t,data_mouse]


# # Looking more closely at the switch trials
# 

sns.set_style('white')
plt.figure(figsize=(15,5))
plt.suptitle('80-20 reward probabilities',x=0.5,y=1.2,fontsize=30)

labels = ['mouse','HMM [greedy]','HMM [Thompson]']
colors = ['navy','black','grey']

for c,data in enumerate(datas):

    switches = data['Switch'].values

    t_block_unique = np.unique(data['Block Trial'].values)
    p_switch_block = np.zeros((t_block_unique.shape[0],2))
    high_p_port = np.zeros_like(p_switch_block)
    trial_block_count = np.zeros_like(t_block_unique)

    for t in t_block_unique:
        p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
        trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
        p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])

        high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
        high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])


    data.index = np.arange(data.shape[0]) # <-- this is important
    switch_points = data[data['Block Trial'] == 0].index.values

    L = 15
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        try:
            paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
            paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port']
        except:
            pass

    u = paraswitch.mean(axis=0)
    s = paraswitch.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])

    #plt.figure(figsize=(15,5))
    #plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

    plt.subplot(131)
    plt.plot(t_block_unique,p_switch_block[:,0],color=colors[c])
    plt.fill_between(t_block_unique,p_switch_block[:,0]+p_switch_block[:,1],p_switch_block[:,0]-p_switch_block[:,1],color=colors[c],alpha=0.5)
    plt.legend(loc='upper right')
    plt.ylim(0,0.5)
    plt.xlim(0,60)
    plt.xlabel('block trial #',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    plt.title('p(switch) vs block trial',fontsize=20)

    plt.tight_layout()
    print('total # of blocks in dataset: ~%.0f' % (np.sum(data['Block Trial']==0)))

    plt.subplot(132)
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.5)
    plt.vlines(x=0,ymin=0,ymax=0.5,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L)
    plt.ylim(0,0.5)

    plt.subplot(133)
    u = paraswitch_port.mean(axis=0)
    s = paraswitch_port.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c],label=labels[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.5)
    plt.vlines(x=0,ymin=0,ymax=1,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(high reward port)',fontsize=20)
    plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L+10)
    plt.ylim(0,1.1)
    plt.legend(bbox_to_anchor=(2,0.5),fontsize=20)

    plt.tight_layout()


sns.set_style('white')
plt.figure(figsize=(5,10))
plt.suptitle('80-20 reward probabilities',x=0.5,y=1.2,fontsize=30)

labels = ['HMM [greedy]','HMM [Thompson]','mouse']
colors = ['black','grey','navy']

for c,data in enumerate(datas):

    switches = data['Switch'].values

    t_block_unique = np.unique(data['Block Trial'].values)
    p_switch_block = np.zeros((t_block_unique.shape[0],2))
    high_p_port = np.zeros_like(p_switch_block)
    trial_block_count = np.zeros_like(t_block_unique)

    for t in t_block_unique:
        p_switch_block[t,0] = data[data['Block Trial'] == t]['Switch'].mean(axis=0)
        trial_block_count[t] = data[data['Block Trial'] == t].shape[0]
        p_switch_block[t,1] = data[data['Block Trial'] == t]['Switch'].std(axis=0) / np.sqrt(trial_block_count[t])

        high_p_port[t,0] = data[data['Block Trial']==t]['Higher p port'].mean(axis=0)
        high_p_port[t,1] = data[data['Block Trial']==t]['Higher p port'].std(axis=0) / np.sqrt(trial_block_count[t])


    data.index = np.arange(data.shape[0]) # <-- this is important
    switch_points = data[data['Block Trial'] == 0].index.values

    L = 15
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 10))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        try:
            paraswitch[i,:] = data.iloc[point-L:point+L+10]['Switch']
            paraswitch_port[i,:] = data.iloc[point-L:point+L+10]['Higher p port']
        except:
            pass

    u = paraswitch.mean(axis=0)
    s = paraswitch.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])

    #plt.figure(figsize=(15,5))
    #plt.suptitle('analysis of blocks where probabilities switched every 50 rewards',x=0.5,y=1.1,fontsize=20)

    plt.subplot(212)
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.2)
    plt.vlines(x=0,ymin=0,ymax=1.0,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(switch)',fontsize=20)
    #plt.title('p(switch) around the block switch',fontsize=20,x=0.5,y=1.1)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim(-1*L,L)
    plt.ylim(0,0.705)

    plt.subplot(211)
    u = paraswitch_port.mean(axis=0)
    s = paraswitch_port.std(axis=0)
    SE = s/np.sqrt(paraswitch.shape[0])
    plt.plot(np.arange(-1*L,L+10),u,color=colors[c],label=labels[c])
    plt.fill_between(np.arange(-1*L,L+10),u+SE,u-SE,color=colors[c],alpha=0.2)
    plt.vlines(x=0,ymin=0,ymax=1,color=colors[c],linestyle='dotted')
    plt.xlabel('Trial # from block switch',fontsize=20)
    plt.ylabel('p(high reward port)',fontsize=20)
    #plt.title('probability of choosing high reward port \naround the block switch',fontsize=20,x=0.5,y=1.1)
    plt.xlim(-1*L,L+10)
    plt.xticks(fontsize=15)
    plt.yticks([0,0.25,0.5,0.75,1],fontsize=15)
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(2.45,0.5),fontsize=20)

    plt.tight_layout()


import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import bandit_preprocessing as bp
import support_functions as sf
get_ipython().magic('matplotlib inline')


# ### predict belief using HMM on full session
# #### input variables:
#         record_path = string containing path to .csv file of session record
#         session_name = name of session to extract from trial matrix (string, only 1)
#         mouse_id = ID of mouse to extract from trial matrix (string, only 1)
# #### output variables:
#         master_beliefs = value of belief (range 0 to 1) that the system is in the right port state (p(z=0)) (nx1 numpy array, n=number of trials)
# 

def predictBeliefBySession(record_path, session_name, mouse_id, p=0.9, duration=60,
                          root_dir='/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'):

    record = pd.read_csv(record_path,index_col=0)

    record[record['Session ID'] == session_name]

    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
               'Right Reward Prob','Left Reward Prob','Reward Given',
              'center_frame','decision_frame']

    #root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

    full_name = session_name + '_trials.csv'

    path_name = os.path.join(root_dir,full_name)

    trial_df = pd.read_csv(path_name,names=columns)

    data = trial_df.copy()

    '''
    Tuned parameters
    '''
    #duration = 60 # number steps until switch reward probability
    #p = 0.8 # prob of reward if choose the correct side
    q = 1.0-p # prob of reward if choose the incorrect side

    '''
    Set up outcome & transition matrices T such that T[i,j] is the probability of transitioning
    from state i to state j. 
    If the true number of trials before switching is 'duration', then set the
    probability of switching to be 1 / duration, and the probability of 
    staying to 1 - 1 / duration
    '''

    s = 1 - 1./duration
    T = np.array([[s, 1.0-s],
                  [1.0-s,s]])

    #observation array
    '''
    set up array such that O[r,z,a] = Pr(reward=r | state=z,action=a)

    eg. when action = L, observation matrix should be:
    O[:,:,1]    = [P(r=0 | z=0,a=0),  P(r=0 | z=1,a=0)
                    P(r=1 | z=0,a=0),  P(r=1 | z=1,a=0)]
                = [1-p, 1-q
                    p,   q]
    '''
    O = np.zeros((2,2,2))
    # let a 'right' choice be represented by '0'
    O[:,:,0] = np.array([[1.0-p, 1.0-q],
                        [p,q]])
    O[:,:,1] = np.array([[1.0-q, 1.0-p],
                        [q,p]])

    #TEST: All conditional probability distributions must sum to one
    assert np.allclose(O.sum(0),1), "All conditional probability distributions must sum to one!"

    """
    Run model on data and output predicted belief state based on evidence from entire session
    """

    #set test data
    data_test = data.copy()

    n_trials = data_test.shape[0]

    #initialize prediction array
    y_predict = np.zeros(n_trials)
    likeli = []
    master_beliefs = np.zeros(n_trials)

    for trial in range(data_test.shape[0]):
        curr_trial = data_test.iloc[trial]
        n_plays = trial
        actions = data['Port Poked'].values - 1
            # originally Port 2 = left (Decision 1), port 1 = right...switch to zeros and ones (right = 0)
        rewards = data['Reward Given'].values
        beliefs = np.nan*np.ones((n_plays+1,2))
        beliefs[0] = [0.5,0.5] #initialize both sides with equal probability
        #run the algorithm


        for play in range(n_plays):

            assert np.allclose(beliefs[play].sum(), 1.0), "Beliefs must sum to one!"

            #update neg log likelihood
            likeli.append(-1*np.log(beliefs[play,actions[play]]))

            #update beliefs for next play
            #step 1: multiply by p(r_t | z_t = k, a_t)
            belief_temp = O[rewards[play],:,actions[play]] * beliefs[play]

            #step 2: sum over z_t, weighting by transition matrix
            beliefs[play+1] = T.dot(belief_temp)

            #step 3: normalize
            beliefs[play+1] /= beliefs[play+1].sum()

        #predict action
        y_predict[trial] = np.where(beliefs[-1] == beliefs[-1].max())[0][0]
        master_beliefs[trial] = beliefs[-1][0]
    return master_beliefs


# # Trial durations new bigger box vs old boxes
# 

import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import os
get_ipython().magic('matplotlib inline')


# ## Load in record
# 

record = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)


q43_ = record[(((record['Mouse ID']=='q43') |(record['Mouse ID']=='Q43')) & (record['p(high Port)'] > 0))].copy()
q45_ = record[(((record['Mouse ID']=='q45') |(record['Mouse ID']=='Q45')) & (record['p(high Port)'] > 0))].copy()
q43 = q43_.loc[623:].copy()
q45 = q45_.loc[644:].copy()


data = q45.append(q43)


# ## Load in trial matrices
# 

'''
load in trial data
'''

#let's just start with 03312017
session_names = ['03152017_Q43','03152017_Q45','04032017_Q43','04032017_Q45']

columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data'

trial_df = []

for session in session_names:
    full_name = session + '_trials.csv'
    
    path_name = os.path.join(root_dir,full_name)
    
    trial_df.append(pd.read_csv(path_name,names=columns))

mouse_ids = ['q43','q45','q43','q45']


for i,df in enumerate(trial_df):
    
    curr_feature_matrix = bp.create_reduced_feature_matrix(df,mouse_ids[i],session_names[i],feature_names='Default',curr_trial_duration=True)
    
    if i == 0:
        master_matrix = curr_feature_matrix.copy()
    else:
        master_matrix = master_matrix.append(curr_feature_matrix)


master_matrix


# # Add column for which box the mouse was in
# 

master_matrix.index = np.arange(master_matrix.shape[0])

inds = master_matrix[master_matrix['Session ID'] == '03152017_Q43'].index.values
master_matrix.loc[inds,'Box'] = 'small'

inds = master_matrix[master_matrix['Session ID'] == '03152017_Q45'].index.values
master_matrix.loc[inds,'Box'] = 'small'

inds = master_matrix[master_matrix['Session ID'] == '04032017_Q43'].index.values
master_matrix.loc[inds,'Box'] = 'big'

inds = master_matrix[master_matrix['Session ID'] == '04032017_Q45'].index.values
master_matrix.loc[inds,'Box'] = 'big'


master_matrix.head(5)


# # Hist of trial durations
# 

plt.figure(figsize=(10,10))
plt.hist(master_matrix[master_matrix['Box']=='small']['Trial Duration'].values,bins=30,color='purple',label = 'small box',alpha=0.5,normed=True)
plt.hist(master_matrix[master_matrix['Box']=='big']['Trial Duration'].values,bins=30,color='green',label = 'big box',alpha=0.5,normed=True)
plt.title('Distribution of trial durations')
plt.xlabel('Trial Duration (s)')
plt.legend(loc = 'upper right')


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 0))]['Trial Duration'].values,bins=30,color='purple',label = 'stays',alpha=0.5,normed=True)
plt.hist(master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 1))]['Trial Duration'].values,bins=30,color='green',label = 'switches',alpha=0.5,normed=True)
plt.title('Small box')
plt.xlabel('Trial Duration (s)')
plt.legend(loc = 'upper right')
plt.xlim(0,2)

plt.subplot(122)
plt.hist(master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 0))]['Trial Duration'].values,bins=30,color='purple',label = 'stays',alpha=0.5,normed=True)
plt.hist(master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 1))]['Trial Duration'].values,bins=30,color='green',label = 'switches',alpha=0.5,normed=True)
plt.title('Big box')
plt.xlabel('Trial Duration (s)')
plt.legend(loc = 'upper right')
plt.xlim(0,2)

print('Number of stays:switches in small box: %.0f : %.0f' % 
      (master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 0))].shape[0],
       master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 1))].shape[0]))

print('Number of stays:switches in big box: %.0f : %.0f' % 
      (master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 0))].shape[0],
       master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 1))].shape[0]))


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 0))]['Trial Duration'].values,bins=30,color='purple',label = 'small',alpha=0.5,normed=True)
plt.hist(master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 0))]['Trial Duration'].values,bins=30,color='green',label = 'big',alpha=0.5,normed=True)
plt.title('Stays')
plt.xlabel('Trial Duration (s)')
plt.legend(loc = 'upper right')
plt.xlim(0,2)

plt.subplot(122)
plt.hist(master_matrix[((master_matrix['Box']=='small') & (master_matrix['Switch'] == 1))]['Trial Duration'].values,bins=30,color='purple',label = 'small',alpha=0.5,normed=True)
plt.hist(master_matrix[((master_matrix['Box']=='big') & (master_matrix['Switch'] == 1))]['Trial Duration'].values,bins=30,color='green',label = 'big',alpha=0.5,normed=True)
plt.title('Switches')
plt.xlabel('Trial Duration (s)')
plt.legend(loc = 'upper right')
plt.xlim(0,2)


import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import bandit_preprocessing as bp
import sys
import os
import matplotlib.pyplot as plt
import calcium_codes as cc
import hmm_on_behavior as hob
get_ipython().magic('matplotlib inline')


record_path = '/Users/celiaberon/GitHub/mouse_bandit/session_record.csv'
ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat'
#ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/q43_03202017_bandit_8020/q43_03202017_neuron_master.mat'
#ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/cnmfe_test/neuron_results.mat'

record = pd.read_csv(record_path,index_col=0)
ca_data = scio.loadmat(ca_data_path,squeeze_me = True, struct_as_record = False)
neuron = ca_data['neuron_results'] 


neuron.C_raw.shape


session_name  = '03142017_K7'
mouse_id = 'K7'

#session_name = '03202017_Q43'
#mouse_id = 'Q43'

record[record['Session ID'] == session_name]


# # Extract data from specific session
# 

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame']

root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

full_name = session_name + '_trials.csv'

path_name = os.path.join(root_dir,full_name)

trial_df = pd.read_csv(path_name,names=columns)


# ### Add column for estimated by HMM using entire session as memory
# 

beliefs = hob.predictBeliefBySession(record_path, session_name, mouse_id)

columns.append('Belief')
trial_df['Belief'] = beliefs
trial_df.head(15)


trial_df.iloc[-1]['decision_frame'] - trial_df.iloc[0]['center_frame']


neuron.C_raw.shape


# # convert to feature matrix
# ### Add column for beliefs using past (n) trials as memory (n <= 10)
# 

feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)

beliefs_feat_mat = hob.predictBeliefFeatureMat(feature_matrix, 10)

feature_matrix['Belief'] = beliefs_feat_mat


feature_matrix.head(5)


feature_matrix[feature_matrix['Switch']==1]


feature_matrix[feature_matrix['Switch']==1]['0_ITI'].values.mean()


plt.plot(trial_df['Belief'])
plt.plot(feature_matrix[feature_matrix['Reward']==1]['Trial'], feature_matrix[feature_matrix['Reward']==1]['Belief'], alpha=0.5)
#plt.scatter(feature_matrix[feature_matrix['Reward']==1]['Trial'],temp[:,1,0], alpha=0.3)


#plt.scatter(temp[:,0,0], feature_matrix[feature_matrix['Reward']==1]['Belief'])
#temp = aligned_start.sum(axis=1)


plt.scatter(feature_matrix[feature_matrix['Decision']==1]['Port Streak'],
            feature_matrix[feature_matrix['Decision']==1]['Belief'])


# ## function to get frames based on one to three conditions
#     First define function so it can take multiple conditions:
#     Input variables:
#         df = feature matrix
#         cond(n)_name = string containing column name (i.e. Reward, Switch) 
#         cond(n) = desired identity (0,1)
#         
#     Output variables:
#         frames = (num_Trials x 2) matrix containing frame # for center poke and decision poke for each trial
# 

def extract_frames(df, cond1_name, cond1=False, cond2_name=False,cond2=False, cond3_name=False,
                   cond3=False, cond1_ops= '=', cond2_ops = '=', cond3_ops = '='):
    
    import operator
    
    # set up operator dictionary
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '=': operator.eq}
    
    if type(cond3_name)==str:
        frames_c = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['center_frame'])
        frames_d = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    elif type(cond2_name)==str:
        frames_c = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['center_frame'])
        frames_d = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    else:
        frames_c =(df[(df[cond1_name] == cond1)]['center_frame'])
        frames_d =(df[(df[cond1_name] == cond1)]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames


# ### Set the parameters to input into extract_frames function
#     The notebook is set to automatically run for both conditions of each condition called.
#     Setting n_variables to a value between 1 and 3 will run the corresponding number of conditions through the rest of the script.
#     (This may be confusing because here you can always define up to 3 condition identities, but if n_variables is set to a value less than 3, only the first conditions will be included)
# 

"""
Decision: 0=Right, 1=Left
Reward: 0=unrewarded, 1=rewarded
Switch: 0=last trial at same port, 1=last trial at different port-->switched
Belief: 0-1 value where 0 represents right port and 1 represents left port
"""

cond1_name = 'Switch'
cond1_a = 0
cond1_b = 1
cond1_ops = '='
cond2_name = 'Decision'
cond2_a = 0
cond2_b = 1
cond3_name = 'Reward'
cond3_a = 0
cond3_b = 1

conditions = [cond1_name, cond2_name, cond3_name]
n_variables = 2
extension = 30

cond1_ops_b='='
if cond1_ops != '=':
    if cond1_ops == '>':
        cond1_ops_b = '<='
        print(cond1_ops_b)
    elif cond1_ops == '>=':
        cond1_ops_b = '<'
        print(cond1_ops_b)


# ### Extract the frames for the specified conditions and create arrays containing frames for beginning and end of window of interest
# 
#     Set the value of 'extension' to the # of frames added before the center poke and after the decision poke
#     fr_1x2x3x = center poke, decision poke frame #s for all trials with the corresponding conditions.
# 
#     At the moment this is not set up in the most efficient way...the extract_frames function operates by calling all 3 variables (i.e. Switch, Reward) across both conditions (a=0, b=1) to generate the maximum number of combinations (8 for 3 variables). If the number of variables is less than 3, it will later use the 'groupings_x' variables to concatenate these arrays into the appropriate final arrays (for example, to go from 3 variables to 2, all beginning with 1a2a get combined)
# 

# center frames in first column, decision frames in second
fr_1a2a3a = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_a, cond3_name, cond3_a, cond1_ops=cond1_ops)

fr_1b2a3a = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_a, cond3_name, cond3_a, cond1_ops=cond1_ops_b)

fr_1a2b3a = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_b, cond3_name, cond3_a, cond1_ops=cond1_ops)

fr_1b2b3a = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_b, cond3_name, cond3_a, cond1_ops=cond1_ops_b)

fr_1a2b3b = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_b, cond3_name, cond3_b, cond1_ops=cond1_ops)

fr_1a2a3b = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_a, cond3_name, cond3_b, cond1_ops=cond1_ops)

fr_1b2a3b = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_a, cond3_name, cond3_b, cond1_ops=cond1_ops_b)

fr_1b2b3b = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_b, cond3_name, cond3_b, cond1_ops=cond1_ops_b)

var_keys = '1a2a3a', '1b2a3a', '1a2b3a', '1b2b3a', '1a2b3b', '1a2a3b', '1b2a3b', '1b2b3b'
groupings_2 = np.stack(((0,5), (1,6), (2,4), (3,7)))
groupings_1 = np.stack(((0,2,4,5), (1,3,6,7)))

#start_stop_frames = {var_keys[0]:fr_1a2a3a, var_keys[1]:fr_1b2a3a, var_keys[2]:fr_1a2b3a, var_keys[3]:fr_1b2b3a, 
#          var_keys[4]:fr_1a2b3b, var_keys[5]:fr_1a2a3b, var_keys[6]:fr_1b2a3b, var_keys[7]:fr_1b2b3b}


# ### Create a dictionary referencing every array of start/stop frames
#     Start-stop_frames = frame starting however long the extension value was before the center poke frame, and frame that long after the decision poke
#     Dictionary is designed to construct each array based on the number of variables (conditions) specified in the beginning...this is where the groupings variables are used to condense arrays for fewer conditions.
# 

n_combos = 2**n_variables

for i in range(n_combos):
    if n_variables == 3:
        if i == 0:
            start_stop_frames = {var_keys[i]:eval('fr_%s' %var_keys[i])}
        if i > 0:
            start_stop_frames.update({var_keys[i]:eval('fr_%s' %var_keys[i])})
    if n_variables == 2:
        if i == 0:
            start_stop_frames = {var_keys[i][0:4]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][1]])))))}
        if i > 0:
            start_stop_frames.update({var_keys[i][0:4]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][1]])))))})
            if i == np.max(n_combos)-1:
                var_keys = list(start_stop_frames.keys())  

    if n_variables == 1:
        if i == 0:
            start_stop_frames = {var_keys[i][0:2]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][1]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][2]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][3]])))))}
        if i > 0:
            start_stop_frames.update({var_keys[i][0:2]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][1]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][2]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][3]])))))})
            if i == np.max(n_combos)-1:
                var_keys = list(start_stop_frames.keys())            

                
#if n_variables == 2:
#    var_keys = list(start_stop_frames.keys())     
        
start_stop_frames.keys()


for i in start_stop_frames:
    start_stop_frames[i][:,0] = start_stop_frames[i][:,0] - extension
    start_stop_frames[i][:,1] = start_stop_frames[i][:,1] + extension


# ### Set up some other parameters to be used in the rest of the script
#     detectEvents() returns binary matrix of 0s and 1s representing frames where events occurred. Replace raw trace file with this processed file and apply a Gaussian filter over events.
#     
#     --nNeurons = number of neurons output by CNMF-e
#         Redefined as number of neurons after some initial processing:
# 
#     Working on a function to combine the above with previous execution of this script on the raw trace file using a flag for 'events'. Once that's complete, will need to add back in NaN cleansing (this is already incorporated into the detectEvents() function). The following arrays will then be used.
#     
#     --nan_neurons = any neurons containing NaNs in their calcium traces
#     --good_neurons = neurons not containing NaNs -- used to redefine nNeurons
# 

events = cc.detectEvents(ca_data_path)

neuron.C_raw = np.copy(events)
nNeurons = neuron.C_raw.shape[0]
nFrames = neuron.C_raw.shape[1]

#Create Gaussian filter and apply to raw trace
sigma = 3;
sz = 10; # total width 

x = np.linspace(-sz / 2, sz / 2, sz);
gaussFilter = np.exp(-x**2 / (2*sigma**2));
gaussFilter = gaussFilter / np.sum(gaussFilter);

smoothed = np.zeros((nNeurons, neuron.C_raw.shape[1]+sz-1));

for i in range(0, nNeurons):
    smoothed[i,:] = np.convolve(neuron.C_raw[i,:], gaussFilter);
    
neuron.C_raw = smoothed[:,0:nFrames]


# This is just used to visualize the effect of the Gaussian filter on each event
plt.plot(neuron.C_raw[0,:])
plt.plot(events[0,:])


# ### Calculate the number of frames required to span the longest trial in all conditions, and use this for all trials
#     --nTrials = list of number of trials in each condition (length = n_combos)
# 
#     --max_window = find window length (number of frames) required to capture center poke to decision poke for all trials (with extensions) and then just take maximum length trial across all conditions)
# 

nTrials = [start_stop_frames[var_keys[i]].shape[0] for i in range(n_combos)]
max_window = np.zeros(n_combos) 
window_length= np.zeros((np.max(nTrials), n_combos))

    
for i in range(n_combos):
    for iTrial in range(nTrials[i]):
        window_length[iTrial, i] = int(((start_stop_frames[var_keys[i]][iTrial][1]-
                                 start_stop_frames[var_keys[i]][iTrial][0])))
    max_window[i] = np.max(window_length)
    
max_window = int(max_window.max())


med_trial_length = [np.median(window_length[0:nTrials[i],:]) for i in range(n_combos)]
med_trial_length = np.median(med_trial_length) - 2*extension


# ### Aligned to center poke
#     Pull out segements of calcium traces in designated window of frames. Calculate mean fluorescence for each neuron across all trials. 
#     -- aligned_start = (number of trials x number frames x number of neurons x number of combinations)
#     -- mean_center_poke = mean fluorescence (or mean number of events) for each neuron across all trials aligned to center poke
#     
#     If using raw traces, will add back in normalization here.
# 

print(int(start_stop_frames[var_keys[i]][iTrial][0]))
start_stop_frames[var_keys[i]][iTrial][0]+max_window


aligned_start = np.zeros((np.max(nTrials), max_window, nNeurons, n_combos))
mean_center_poke = np.zeros((max_window, nNeurons, n_combos))
during_trial = np.zeros_like((aligned_start))

for i in range(n_combos):

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    for iNeuron in range(nNeurons): # for each neuron
        for iTrial in range(0,nTrials[i]): # and for each trial
            aligned_start[iTrial,:, iNeuron, i] = neuron.C_raw[iNeuron,
                int(start_stop_frames[var_keys[i]][iTrial][0]):
                (int(start_stop_frames[var_keys[i]][iTrial][0])+max_window)]
            during_trial[iTrial,0:int(window_length[iTrial,i]-extension),iNeuron,i] = neuron.C_raw[iNeuron,
                int(start_stop_frames[var_keys[i]][iTrial][0]+extension):
                (int(start_stop_frames[var_keys[i]][iTrial][0] +window_length[iTrial,i]))]
  
    # take mean of fluorescent traces across all trials for each neuron, then normalize
    # for each neuron
    mean_center_poke[:,:,i]= np.mean(aligned_start[0:nTrials[i],:,:,i], axis=0)


pre_trial = aligned_start[:, int(extension-med_trial_length):extension, :, :].sum(axis=1)
during_trial = during_trial.sum(axis=1)


plt.hist(neuron.C_raw.sum(axis=1)/32)


# ### Plot heatmap of average events per trial (intensity) for each neuron (y) over time (x)
# ### Aligned to center poke
# 

ydim = n_combos/2
plt.figure(figsize=(8,ydim*4))
for i in range(n_combos):

    plt.subplot(ydim,2,i+1)  
    plt.imshow(np.transpose(mean_center_poke[:,:,i]))#, plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.axis('tight')
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    if n_variables == 3:
        plt.title("%s = %s\n %s = %s\n%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], 
                    conditions[int(var_keys[i][4])-1],
                    var_keys[i][5], nTrials[i])) 
    if n_variables == 2:
        plt.title("%s = %s\n %s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], nTrials[i]))
    if n_variables == 1:
        plt.title("%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1], nTrials[i]))
plt.tight_layout()


# ### heatmap for calcium traces of a single neuron across all trials
#     white dashed line for center poke time
#     white vertical lines for decision poke time -- need something more subtle
# 

sample_neuron = 10

#plt.figure(figsize=(10,10))
plt.imshow(aligned_start[0:nTrials[0],:,sample_neuron, 0])
plt.axvline(x=extension, color='white', linestyle = '--', linewidth=.9)
plt.ylabel('Trial Number')
plt.xlabel('Frame (center poke at %s)' %extension)
#plt.scatter((start_stop_times[var_keys[0][:]])-(start_stop_frames[var_keys[0]])+extension,range(nTrials[0]), color='white', marker = '|', s=10)
plt.title('%s = %s\n%s = %s\nNeuron ID = %s' % (cond1_name, conditions[0], cond2_name, cond2_a, sample_neuron))
plt.axis('tight')


# ## aligned to decision poke
#     Same process as for aligned to center poke
# 

aligned_decision = np.zeros((np.max(nTrials), max_window, nNeurons, n_combos))
mean_decision = np.zeros((max_window, nNeurons, n_combos))

for i in range(n_combos):

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to decision poke
    for iNeuron in range(nNeurons):
        for iTrial in range(nTrials[i]):
            aligned_decision[iTrial,:, iNeuron, i] = neuron.C_raw[iNeuron, 
                int(start_stop_frames[var_keys[i]][iTrial][1])-max_window:
                (int(start_stop_frames[var_keys[i]][iTrial][1]))]

    # take mean of fluorescent traces across all trials for each neuron
    mean_decision[:,:,i]= np.mean(aligned_decision[0:nTrials[i],:,:,i], axis=0)
   


post_trial = aligned_decision[:, int(max_window-extension):int(max_window-extension + 
             med_trial_length), :, :].sum(axis=1)


[plt.scatter(window_length[:,c], during_trial[:,n,c]) for n in range(nNeurons) 
 for c in range(n_combos)]
print('c')


temp = mean_decision.sum(axis=1)
[plt.plot(np.transpose(temp[:,c])) for c in range(n_combos)]
plt.axvline(x=max_window-extension, linestyle='--', color='k', linewidth=.9)
plt.axvline(x=max_window-(med_trial_length+extension), linestyle='--', color='k', linewidth=.9)


# ### Plot heatmap aligned to decision poke
# 

plt.figure(figsize=(8,ydim*4))
for i in range(n_combos):
    plt.subplot(ydim,2,i+1)  
    plt.imshow(np.transpose(mean_decision[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window-extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frames (decision poke at %s)' % (max_window-extension))
    plt.ylabel('Neuron ID')
    plt.axis('tight')
    if n_variables == 3:
        plt.title("%s = %s\n %s = %s\n%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], 
                    conditions[int(var_keys[i][4])-1],
                    var_keys[i][5], nTrials[i])) 
    if n_variables == 2:
        plt.title("%s = %s\n %s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], nTrials[i]))
    if n_variables == 1:
        plt.title("%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1], nTrials[i]))
plt.tight_layout()


