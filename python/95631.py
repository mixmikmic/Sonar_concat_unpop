#  Dependencies
import requests
from elasticsearch import Elasticsearch,helpers
import numpy as np
import uuid
import random
import json


es = Elasticsearch(['atlas-kibana.mwt2.org:9200'],
                                 timeout=10000)


es.ping()


print(es.info(), '\n', es.cluster.health())


# checking indices

indices = es.indices.get_aliases().keys()
print('Total No. of Indices: ',len(indices),'\n')
# print('\n', indices)
rucio=(index for index in indices if('rucio-events' in index))
network_weather = (index for index in indices if('network_weather-2017' in index))
rucio_indices = []
nws_indices = []
for event in network_weather:
    nws_indices.append(event)
for event in rucio:
    rucio_indices.append(event)
print('total NWS indices:',len(nws_indices),'\n')
print(nws_indices[0:5], '\n')

print('total rucio indices:',len(rucio_indices),'\n')
print(rucio_indices[0:5])


count=es.count(index='network_weather-2017*')
print('total documents : {}'.format( count['count']) )


nws_indices_dict = {}
for event in nws_indices:
    i = es.count(index=event)
    nws_indices_dict[event] = i['count']
# print('total data points:',sum(int(list(indices_dict.values()))))

print(nws_indices_dict)


def extract_data(index, query, scan_size, scan_step):
    resp = es.search(
    index = index,
    scroll = '20m',
    body = query,
    size = scan_step)

    sid = resp['_scroll_id']
    scroll_size = resp['hits']['total']
    print('total hits in {} : {}'.format(index,scroll_size))
    results=[]
    for hit in resp['hits']['hits']:
        results.append(hit)
    #plot_data_stats(results)
    steps = int((scan_size-scan_step)/ scan_step)

    # Start scrolling

    for i in range(steps):
        if i%10==0:
            print("Scrolling index : {} ; step : {} ...\n ".format(index,i))
        resp = es.scroll(scroll_id = sid, scroll = '20m')
        # Update the scroll ID
        sid = resp['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(resp['hits']['hits'])
        if i%10==0:
            print("scroll size: " + str(scroll_size))
        for hit in resp['hits']['hits']:
            results.append(hit)
    
    print("\n Done Scrolling through {} !! \n".format(index))
    results = pd.DataFrame(results)
    print(results.info(), '\n')
    return results


import numpy as np
import pandas as pd
nws= extract_data(index='network_weather-2017.8.10',query={}, scan_size=1000000, scan_step=10000)


nws.to_csv('nws.csv')


nws['_type'].unique()


nws['_type'].value_counts()


myquery = {
    "query": {
        "term": {
            '_type': 'throughput'
            }
        }
    }
es.search(index='network_weather-2017.8.10',body={}, size = 1)


query2={
    "query": {
        "term": {
            '_type': 'throughput'
            }
        }
    }
import pandas as pd
throughput= extract_data(index='network_weather-2017.8.10', query=query2, scan_size=15033, scan_step=10000)


throughput.head()


cond1 = throughput['destSite'] ==float('nan')
a= throughput[not(cond1)]
# cond2 = a['srcSite'] !=float('nan')
# a= a[cond2]


a.info()


a.head()
# type(a['destSite'][2])


# # INDEX- network-weather-2017*
# 
# ## event format - 
# 

# ```json
# {
#     "_index": "jobs_archive_2017-06-05",
#     "_type": "jobs_data",
#     "_id": "3413570266",
#     "_version": 1,
#     "_score": null,
#     "_source": {
#         "pandaid": 3413570266,
#         "jobdefinitionid": 146514542,
#         "schedulerid": "aCT-atlact1-ai403",
#         "pilotid": "http://aipanda403.cern.ch/data/jobs/20170605/jost.arnes.si/YWRKDmXP0bqnmmR0Xox1SiGmABFKDmABFKDmSMIKDmGBGKDmUdp8Lo|NEWMOVER-ON|SLURM|PR|PICARD 68.3",
#         "creationtime": "2017-06-05T06:02:41",
#         "creationhost": "hammercloud-ai-74.ipv6.cern.ch",
#         "modificationtime": "2017-06-05T06:59:50",
#         "modificationhost": "wn095.arnes.si",
#         "atlasrelease": "Atlas-20.7.6",
#         "transformation": "http://pandaserver.cern.ch:25080/trf/user/runAthena-00-00-11",
#         "homepackage": "AnalysisTransforms-AtlasDerivation_20.7.6.4",
#         "prodserieslabel": null,
#         "prodsourcelabel": "user",
#         "produserid": "/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=gangarbt/CN=601592/CN=Robot: Ganga Robot/CN=proxy",
#         "assignedpriority": 0,
#         "currentpriority": 4000,
#         "attemptnr": 0,
#         "maxattempt": 2,
#         "jobstatus": "finished",
#         "jobname": "90084d6f-22ef-462f-8fb0-b383e7e4629b_67487",
#         "maxcpucount": 0,
#         "maxdiskcount": 0,
#         "minramcount": 0,
#         "starttime": "2017-06-05T06:29:31",
#         "endtime": "2017-06-05T06:49:49",
#         "cpuconsumptiontime": 834,
#         "cpuconsumptionunit": "s+Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz 20480 KB",
#         "commandtopilot": null,
#         "transexitcode": null,
#         "piloterrorcode": 0,
#         "piloterrordiag": null,
#         "exeerrorcode": 0,
#         "exeerrordiag": "OK",
#         "superrorcode": 0,
#         "superrordiag": null,
#         "ddmerrorcode": 0,
#         "ddmerrordiag": null,
#         "brokerageerrorcode": 0,
#         "brokerageerrordiag": null,
#         "jobdispatchererrorcode": 0,
#         "jobdispatchererrordiag": null,
#         "taskbuffererrorcode": 0,
#         "taskbuffererrordiag": null,
#         "computingsite": "ANALY_ARNES_DIRECT",
#         "computingelement": "jost.arnes.si",
#         "proddblock": "data15_13TeV.00276336.physics_Main.merge.AOD.r7562_p2521_tid07709524_00",
#         "dispatchdblock": null,
#         "destinationdblock": "user.gangarbt.hc20107254.tid845.ANALY_ARNES_DIRECT.6",
#         "destinationse": "ANALY_ARNES_DIRECT",
#         "nevents": 3903,
#         "grid": null,
#         "cloud": "ND",
#         "cpuconversion": "1",
#         "sourcesite": null,
#         "destinationsite": null,
#         "transfertype": null,
#         "taskid": 2,
#         "cmtconfig": "x86_64-slc6-gcc49-opt",
#         "statechangetime": "2017-06-05T06:59:50",
#         "lockedby": null,
#         "relocationflag": 1,
#         "jobexecutionid": 0,
#         "vo": "atlas",
#         "workinggroup": null,
#         "processingtype": "gangarobot",
#         "produsername": "gangarbt",
#         "countrygroup": null,
#         "batchid": "12405832",
#         "parentid": null,
#         "specialhandling": null,
#         "jobsetid": null,
#         "corecount": 1,
#         "ninputdatafiles": 1,
#         "inputfiletype": "AOD",
#         "inputfileproject": "data15_13TeV",
#         "inputfilebytes": 824476423,
#         "noutputdatafiles": 1,
#         "outputfilebytes": 4025868,
#         "dbTime": 29.64,
#         "dbData": 8090396,
#         "workDirSize": 1650688,
#         "jobmetrics": null,
#         "workqueue_id": null,
#         "jeditaskid": null,
#         "jobsubstatus": null,
#         "actualcorecount": 1,
#         "reqid": null,
#         "maxrss": 1483408,
#         "maxvmem": 3293680,
#         "maxpss": 1411801,
#         "avgrss": 1102786,
#         "avgvmem": 2744469,
#         "avgswap": 6041,
#         "avgpss": 1042013,
#         "maxwalltime": null,
#         "wall_time": 1218,
#         "cpu_eff": 0.68472904,
#         "queue_time": 1610,
#         "timeGetJob": 0,
#         "timeStageIn": 17,
#         "timeExe": 1038,
#         "timeStageOut": 15,
#         "timeSetup": 0,
#         "nucleus": null,
#         "eventservice": null,
#         "failedattempt": null,
#         "hs06sec": null,
#         "hs06": null,
#         "gShare": "Analysis",
#         "IOcharRead": 753904,
#         "IOcharWritten": 17778,
#         "IObytesRead": 3151948,
#         "IObytesWritten": 23136,
#         "IOcharReadRate": 766631,
#         "IOcharWriteRate": 18078,
#         "IObytesReadRate": 3205158,
#         "IObytesWriteRate": 23526
#     },
#     "fields": {
#         "Average RSS": [
#             1129252864
#         ],
#         "Average PSS": [
#             1067021312
#         ],
#         "IObytesWritten_Bytes": [
#             23691264
#         ],
#         "modificationtime": [
#             1496645990000
#         ],
#         "wall_time_cores_h": [
#             0.3383333333333333
#         ],
#         "Max_PSS_per_core": [
#             1445684224
#         ],
#         "starttime": [
#             1496644171000
#         ],
#         "TotalJobTime": [
#             2828
#         ],
#         "IOcharRead_Bytes": [
#             771997696
#         ],
#         "dbRate": [
#             0.26031049371244186
#         ],
#         "Wall time per event times core": [
#             0.31206764027671025
#         ],
#         "trf_scripted": [
#             "runAthena-00-00-11"
#         ],
#         "Average VMEM": [
#             2810336256
#         ],
#         "cpueff_per_core_painless": [
#             0.6847290640394089
#         ],
#         "Max VMEM": [
#             3372728320
#         ],
#         "infile_proj": [
#             "data15_13TeV"
#         ],
#         "CPU eff per core * 100": [
#             68.47290396690369
#         ],
#         "Max PSS": [
#             1445684224
#         ],
#         "statechangetime": [
#             1496645990000
#         ],
#         "Max RSS": [
#             1519009792
#         ],
#         "endtime": [
#             1496645389000
#         ],
#         "temp_timeOther": [
#             148
#         ],
#         "CPU eff per core": [
#             0.6847290396690369
#         ],
#         "sharedmem": [
#             71607
#         ],
#         "CPU eff of 40 core jobs": [
#             0.01711822599172592
#         ],
#         "CPU eff of 8 core jobs": [
#             0.08559112995862961
#         ],
#         "IOcharWritten_Bytes": [
#             18204672
#         ],
#         "Wall time per event * 10": [
#             3.1206764027671023
#         ],
#         "cputimeperevent": [
#             0.21368178324365872
#         ],
#         "IObytesRead_Bytes": [
#             3227594752
#         ],
#         "cpueff_per_core_over_timeExe": [
#             80.34682080924856
#         ],
#         "inputRate": [
#             48498613.11764706
#         ],
#         "cputimeperevent seconds": [
#             0.21368178324365872
#         ],
#         "cpu_eff_wen": [
#             0.6847290640394089
#         ],
#         "Walltime times core": [
#             1218
#         ],
#         "creationtime": [
#             1496642561000
#         ],
#         "Wall time per event": [
#             0.31206764027671025
#         ],
#         "top_country": [
#             null
#         ],
#         "CPU eff of 6 core jobs": [
#             11.41214609629035
#         ],
#         "timeExe_Core_Events": [
#             0.26594926979246736
#         ]
#     },
#     "sort": [
#         1496645990000
#     ]
# }
# ```
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import datetime 
import seaborn as sns
import os
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
import tensorflow as tf
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
import keras.callbacks as cb
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM
from keras_tqdm import TQDMNotebookCallback
from multi_gpu import to_multi_gpu
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model


def load_encoders():
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    t_endpoint_encoder = LabelEncoder()
    
    src_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    dst_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    type_encoder.classes_ = np.load('encoders/type.npy')
    activity_encoder.classes_ = np.load('encoders/activity.npy')
    protocol_encoder.classes_ = np.load('encoders/protocol.npy')
    t_endpoint_encoder.classes_ = np.load('encoders/endpoint.npy')
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def train_encoders(rucio_data, use_cache=True):
    
    if use_cache:
        if os.path.isfile('encoders/ddm_rse_endpoints.npy') and os.path.isfile('encoders/activity.npy'):
            print('using cached LabelEncoders for encoding data.....')
            src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder=load_encoders()
        else:
            print('NO cache found')
    else:
        print('No cached encoders found ! Training Some New Ones using input data!')
        src_encoder = LabelEncoder()
        dst_encoder = LabelEncoder()
        type_encoder = LabelEncoder()
        activity_encoder = LabelEncoder()
        protocol_encoder = LabelEncoder()
        t_endpoint_encoder = LabelEncoder()

        src_encoder.fit(rucio_data['src-rse'].unique())
        dst_encoder.fit(rucio_data['dst-rse'].unique())
        type_encoder.fit(rucio_data['src-type'].unique())
        activity_encoder.fit(rucio_data['activity'].unique())
        protocol_encoder.fit(rucio_data['protocol'].unique())
        t_endpoint_encoder.fit(rucio_data['transfer-endpoint'].unique())

        np.save('encoders/src.npy', src_encoder.classes_)
        np.save('encoders/dst.npy', dst_encoder.classes_)
        np.save('encoders/type.npy', type_encoder.classes_)
        np.save('encoders/activity.npy', activity_encoder.classes_)
        np.save('encoders/protocol.npy', protocol_encoder.classes_)
        np.save('encoders/endpoint.npy', t_endpoint_encoder.classes_)
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)


def preprocess_data(rucio_data, use_cache=True):
    
    fields_to_drop = ['account','reason','checksum-adler','checksum-md5','guid','request-id','transfer-id','tool-id',
                      'transfer-link','name','previous-request-id','scope','src-url','dst-url', 'Unnamed: 0']
    timestamps = ['started_at', 'submitted_at','transferred_at']

    #DROP FIELDS , CHANGE TIME FORMAT, add dataetime index
    rucio_data = rucio_data.drop(fields_to_drop, axis=1)
    for timestamp in timestamps:
        rucio_data[timestamp]= pd.to_datetime(rucio_data[timestamp], infer_datetime_format=True)
    rucio_data['delay'] = rucio_data['started_at'] - rucio_data['submitted_at']
    rucio_data['delay'] = rucio_data['delay'].astype('timedelta64[s]')
    
    rucio_data = rucio_data.sort_values(by='submitted_at')
    
    # Reindex data with 'submitted_at timestamp'
    rucio_data.index = pd.DatetimeIndex(rucio_data['submitted_at'])
    
    #remove all timestamp columns
    rucio_data = rucio_data.drop(timestamps, axis=1)
    
    # encode categorical data
 
    if use_cache==True:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=True)
    else:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=False)

    rucio_data['src-rse'] = src_encoder.transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.transform(rucio_data['transfer-endpoint'])
    
    return rucio_data


def rescale_data(rucio_data, durations):
    # Normalization
    # using custom scaling parameters (based on trends of the following variables)

#     durations = durations / 1e3
    rucio_data['bytes'] = rucio_data['bytes'] / 1e8
    rucio_data['delay'] = rucio_data['delay'] / 1e3
#     rucio_data['src-rse'] = rucio_data['src-rse'] / 1e2
#     rucio_data['dst-rse'] = rucio_data['dst-rse'] / 1e2
    
    return rucio_data, durations

def plot_graphs_and_rescale(data):
    
    durations = data['duration']
    durations.plot()
    plt.ylabel('durations(seconds)')
    plt.show()

    filesize = data['bytes']
    filesize.plot(label='filesize(bytes)')
    plt.ylabel('bytes')
    plt.show()

    delays = data['delay']
    delays.plot(label='delay(seconds)')
    plt.ylabel('delay')
    plt.show()
    
    print('rescaling input continuous variables : filesizes, queue-times, transfer-durations')
    data, byte_scaler, delay_scaler, duration_scaler = rescale_data(data)

    plt.plot(data['bytes'], 'r', label='filesize')
    plt.plot(data['duration'], 'y', label='durations')
    plt.plot(data['delay'],'g', label='queue-time')
    plt.legend()
    plt.xticks(rotation=20)
    plt.show()
    
    return data, byte_scaler, delay_scaler, duration_scaler


def prepare_model_inputs(rucio_data,durations, num_timesteps=50):
    
    #slice_size = batch_size*num_timesteps
    print(rucio_data.shape[0], durations.shape)
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps +1)
    print('Total Data points for training/testing : {} of {} timesteps each.'.format(n_batches, num_timesteps))
    
    inputs=[]
    outputs=[]
    for i in range(0,n_batches):
        v = rucio_data[i:i+num_timesteps]
        w = durations[i+num_timesteps-1]
        inputs.append(v)
        outputs.append(w)
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)
    
    return inputs, outputs


path = '../' # Change this as you need.

def get_rucio_files(path='../', n_files =100):
    abspaths = []
    for fn in os.listdir(path):
        if 'atlas_rucio' in fn:
            abspaths.append(os.path.abspath(os.path.join(path, fn)))
    print("\n Found : ".join(abspaths))
    print('\n total files found = {}'.format(len(abspaths)))
    return abspaths

def load_rucio_data(file, use_cache = True, limit=None):
    print('reading : {}'.format(file))
    data = pd.read_csv(file)
    if limit != None:
        n_size = data.shape[0]
        cut = n_size-1000000
#         cut = n_size - int(n_size * limit)
        data= data[cut:]
        print('Limiting data size to {} '.format(int(1000000)))
#     print(data)
    print('preprocessing data... ')
    data = preprocess_data(data)
    print('Saving indices for later..')
    indices = data.index
    durations = data['duration']
    data = data.drop(['duration'], axis=1)
    data = data[['bytes', 'delay', 'activity', 'dst-rse', 'dst-type',
                 'protocol', 'src-rse', 'src-type', 'transfer-endpoint']]
    data, durations = rescale_data(data, durations)
    data = data.as_matrix()
    durations = durations.as_matrix()
    return data, durations, indices


# path='data/'
# a= get_rucio_files(path=path)
# x, y, indices = load_rucio_data(a[1], limit=5)

# print(x ,'\n', y, '\n', indices)


# x,y = prepare_model_inputs(x,y,num_timesteps=2)


def return_to_original(x, y, preds, index=None):
    #print(x.shape, y.shape)
    #print(x[0,1])
    n_steps = x.shape[1]
    #print(index[:n_steps])
    #print(index[n_steps-1:])
    index = index[n_steps-1:]
    
    cols = ['bytes', 'delay', 'activity', 'dst-rse', 'dst-type','protocol', 'src-rse', 'src-type', 'transfer-endpoint']
    data = list(x[0])
    for i in range(1,x.shape[0]):
        data.append(x[i,n_steps-1,:])
    
    data = data[n_steps-1:]
    #print(len(data))
    data = pd.DataFrame(data, index=index, columns=cols)
    data['bytes'] = data['bytes'] * 1e10
    data['delay'] = data['delay'] * 1e5
    data['src-rse'] = data['src-rse'] * 1e2
    data['dst-rse'] = data['dst-rse'] * 1e2
    
    data = data.round().astype(int)
    print(data.shape)
    data = decode_labels(data)
    data['duration'] = y
    data['prediction'] = pred
    data['duration'] = data['duration'] * 1e3
    data['prediction'] = data['prediction'] * 1e3
    
    return data

# return_to_original(x,y, index=indices)


def decode_labels(rucio_data):
    src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = load_encoders()
    
    rucio_data['src-rse'] = src_encoder.inverse_transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.inverse_transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.inverse_transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.inverse_transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.inverse_transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.inverse_transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.inverse_transform(rucio_data['transfer-endpoint'])
    
    return rucio_data

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def build_model(num_timesteps=50, batch_size = 512, parallel=False):

    model = Sequential()
    layers = [512, 512, 512, 512, 128, 1]
    
    model.add(LSTM(layers[0], input_shape=(num_timesteps, 9), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(layers[3]))
    model.add(Activation("relu"))
    
    model.add(Dense(layers[4]))
    model.add(Activation("relu"))
    
    model.add(Dense(layers[5]))
    model.add(Activation("linear"))
    
    start = time.time()
    
    if parallel:
        model = to_multi_gpu(model,4)
    
    model.compile(loss="mse", optimizer="adam")
    print ("Compilation Time : ", time.time() - start)
    return model


a=['ffv', 'sfdf', 'dsdfd']
a.remove('ffv')
a

a= get_rucio_files(path='data/')
a= list(reversed(a))
a.remove('/home/carnd/DeepAnomaly/data/atlas_rucio-events-2017.07.12.csv')
a


def plot_losses(losses):
    sns.set_context('poster')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    print(len(losses))
    fig.show()

get_ipython().magic('matplotlib inline')
sns.set_context('poster')

def train_network(model=None,limit=None, data=None, epochs=1,n_timesteps=100, batch=128, path="data/",parallel=True):
    
    if model is None:
        #model = build_model(num_timesteps=n_timesteps, parallel=parallel)
        model = load_lstm()
        history = LossHistory()
            
        checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
        print('model built and compiled !')
    
    print('\n Locating training data files...')
    a= get_rucio_files(path=path)
    a= list(reversed(a))
    a.remove('/home/carnd/DeepAnomaly/data/atlas_rucio-events-2017.07.12.csv')
    a.remove('/home/carnd/DeepAnomaly/data/atlas_rucio-events-2017.07.19.csv')
    try:
        for i,file in enumerate(a):
            print("Training on file :{}".format(file))
            x, y, indices = load_rucio_data(file, limit=limit)
            print('\n Data Loaded and preprocessed !!....')
            x, y = prepare_model_inputs(x, y, num_timesteps=n_timesteps)
            print('Data ready for training.')

            start_time = time.time()

            print('Training model...')
            if parallel:
                training = model.fit(x, y, epochs=epochs, batch_size=batch*4,
                                     validation_split=0.05, callbacks=[history,TQDMNotebookCallback(leave_inner=True), checkpointer],
                                     verbose=0)
            else:
                training = model.fit(x, y, epochs=epochs, batch_size=batch,
                                     validation_split=0.05, callbacks=[history,TQDMNotebookCallback(leave_inner=True), checkpointer],
                                     verbose=0)

            print("Training duration : {0}".format(time.time() - start_time))
#             score = model.evaluate(x, y, verbose=0)
#             print("Network's Residual training score [MSE]: {0} ; [in seconds]: {1}".format(score,np.sqrt(score)))
            print("Training on {} finished !!".format(file))
            print('\n Saving model to disk..')
            # serialize model to JSON
            model_json = model.to_json()
            with open("models/lstm_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("models/lstm_model.h5")
            print("Saved model to disk")
            print('plotting losses..')
            plot_losses(history.losses)

        print('Training Complete !!')
        
        return training, model, indices, history.losses

    except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return model, history.losses


train_network(n_timesteps=100, batch=256, parallel =False, limit=0.5)


# # Model performance and error evaluation on a different unseen test set
# 
# ## Sample size = 200,000
# 

model = load_lstm()

data = pd.read_csv('data/atlas_rucio-events-2017.07.06.csv')
data = data[:200000]
data = preprocess_data(data)
indices = data.index
durations = data['duration']
data = data.drop(['duration'], axis=1)
data = data[['bytes', 'delay', 'activity', 'dst-rse', 'dst-type',
             'protocol', 'src-rse', 'src-type', 'transfer-endpoint']]
data, durations = rescale_data(data, durations)
data = data.as_matrix()
durations = durations.as_matrix()
data, durations = prepare_model_inputs(data, durations, num_timesteps=100)

print('DONE')


# pred = model.predict(data)

get_ipython().magic('matplotlib inline')
import seaborn as sns

sns.set_context('poster')

plt.figure(figsize=(50,20))
plt.plot(durations, 'g', label='original-duration')
plt.plot(pred, 'y', label='predicted-duration')
plt.title('Model predictions on Unseen data')
plt.legend()
plt.xlabel(' # sequential samples')
plt.ylabel('transfer duration in seconds')
plt.show()

plt.figure(figsize=(30,20))
plt.plot(durations[:500], 'g', label='original-duration')
plt.plot(pred[:500], 'y', label='predicted-duration')
plt.show()


sns.set_context('poster')
plt.figure(figsize=(20,10))
durations = np.reshape(durations, (durations.shape[0],1))
errors = durations - pred
plt.plot(errors, 'r')
plt.title('Errors fn')
plt.ylabel('errors in seconds')
plt.show()

plt.figure(figsize=(20,10))
plt.plot(errors, 'ro')
plt.title('Errors ScatterPlot')
plt.ylabel('errors in seconds')
plt.show()

errs = errors/60

print('Maximum error value:',np.max(errs))
j=0
k=0
l=0
m=0
n=0
o=0
p=0
for err in errs:
    if err<=3:
        j+=1
    if err<=5:
        k+=1
    if err<=8:
        l+=1
    if err<=10:
        m+=1
    if err<=12:
        n+=1
    if err<=15:
        o+=1
    if err<=20:
        p+=1

print('Total points with errors less than 3 minutes:{} , percetage={}'.format(j, 100*(j/200000)))
print('Total points with errors less than 5 minutes:{} , percetage={}'.format(k, 100*(k/200000)))
print('Total points with errors less than 8 minutes:{} , percetage={}'.format(l, 100*(l/200000)))
print('Total points with errors less than 10 minutes:{} , percetage={}'.format(m, 100*(m/200000)))
print('Total points with errors less than 12 minutes:{} , percetage={}'.format(n, 100*(n/200000)))
print('Total points with errors less than 15 minutes:{} , percetage={}'.format(o, 100*(o/200000)))
print('Total points with errors less than 20 minutes:{} , percetage={}'.format(p, 100*(p/200000)))
    

# durations = np.reshape(durations, (durations.shape[0],1))
# errors = durations - pred
plt.figure(figsize=(20,10))
plt.plot(errs, 'ro')
plt.title('Errors ScatterPlot')
plt.ylabel('errors in minutes')
plt.show()

plt.figure(figsize=(20,10))
plt.plot(np.absolute(errs), 'ro')
plt.title('Absolute Errors-ScatterPlot')
plt.ylabel('errors in minutes')
plt.show()

# bins=[-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60]
plt.figure(figsize=(20,10))
arr= plt.hist(errs, bins=15)
plt.title('Histogram: Errors')
for i in range(15):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
plt.show()

plt.figure(figsize=(20,10))
arr2 = plt.hist(np.absolute(errs), bins=15)
plt.title('Histogram: Absolute errors')
for i in range(15):
    plt.text(arr2[1][i],arr2[0][i],str(arr2[0][i]))
plt.show()


score = model.evaluate(data, durations, verbose=0)
print('MSE = {} ; RMSE = {}'.format(score, np.sqrt(score)))


def load_lstm():
    json_file = open('models/lstm_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/lstm_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss="mse", optimizer="adam")
    print('Model model compiled!!')
    return loaded_model

def evaluate_network(limit=None, n_timesteps=100, path="data/",model=None):
    
    print('\n Locating training data files...')
    a= get_rucio_files(path=path)
    

    for i,file in enumerate(a):
        print("Training on file :{}".format(file))
        x, y, indices = load_rucio_data(file, limit=limit)
        print('\n Data Loaded and preprocessed !!....')
        x, y = prepare_model_inputs(x, y, num_timesteps=n_timesteps)
        print('Data ready for Evaluation')
        
        with tf.device('/gpu:0'):
            start_time = time.time()
            print('making predictions...')
            model = load_lstm()
            predictions = model.predict(x)
            end = time.time - start_time
            print('Done !! in {} min'.format(end/60))
            print('plotting graphs')

            plt.plot(y, 'g')
            plt.plot(predictions, 'y')
            plt.show()

            data = return_to_original(x, y, predictions, index=indices)
            plt.plot(data['duration'], 'g')
            plt.plot(data['prediction'], 'y')
            plt.title('Network predictions')
            plt.ylabel('durations in seconds')
            plt.show()

            data['mae'] = data['duration'] - data['prediction']
            data['mae'].plot()
            plt.show()


train_network(path='data/', limit=0.6, )
# evaluate_network(path='data/', limit = 50000)


# a= get_rucio_files(path=path)
# x, y, indices = load_rucio_data(a[2], limit=1000)
# print('\n Data Loaded and preprocessed !!....')
# x, y = prepare_model_inputs(x, y, num_timesteps=100)
# with tf.device('/gpu:0'):
#     model = load_lstm()
#     pred= model.predict(x)
#     print('done')


# plt.plot(pred)





import datetime as datetime  
import numpy as np
import seaborn as sns
import pandas as pd  
import statsmodels.api as sm  
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import os
get_ipython().magic('matplotlib inline')
sns.set_context('poster')


def load_encoders():
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    t_endpoint_encoder = LabelEncoder()
    
    src_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    dst_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    type_encoder.classes_ = np.load('encoders/type.npy')
    activity_encoder.classes_ = np.load('encoders/activity.npy')
    protocol_encoder.classes_ = np.load('encoders/protocol.npy')
    t_endpoint_encoder.classes_ = np.load('encoders/endpoint.npy')
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def train_encoders(rucio_data, use_cache=True):
    
    if use_cache:
        if os.path.isfile('encoders/ddm_rse_endpoints.npy') and os.path.isfile('encoders/activity.npy'):
            print('using cached LabelEncoders for encoding data.....')
            src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder=load_encoders()
        else:
            print('NO cache found')
    else:
        print('No cached encoders found ! Training Some New Ones using input data!')
        src_encoder = LabelEncoder()
        dst_encoder = LabelEncoder()
        type_encoder = LabelEncoder()
        activity_encoder = LabelEncoder()
        protocol_encoder = LabelEncoder()
        t_endpoint_encoder = LabelEncoder()

        src_encoder.fit(rucio_data['src-rse'].unique())
        dst_encoder.fit(rucio_data['dst-rse'].unique())
        type_encoder.fit(rucio_data['src-type'].unique())
        activity_encoder.fit(rucio_data['activity'].unique())
        protocol_encoder.fit(rucio_data['protocol'].unique())
        t_endpoint_encoder.fit(rucio_data['transfer-endpoint'].unique())

        np.save('encoders/src.npy', src_encoder.classes_)
        np.save('encoders/dst.npy', dst_encoder.classes_)
        np.save('encoders/type.npy', type_encoder.classes_)
        np.save('encoders/activity.npy', activity_encoder.classes_)
        np.save('encoders/protocol.npy', protocol_encoder.classes_)
        np.save('encoders/endpoint.npy', t_endpoint_encoder.classes_)
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def preprocess_data(rucio_data, use_cache=True):
    
    fields_to_drop = ['account','reason','checksum-adler','checksum-md5','guid','request-id','transfer-id','tool-id',
                      'transfer-link','name','previous-request-id','scope','src-url','dst-url', 'Unnamed: 0']
    timestamps = ['started_at', 'submitted_at','transferred_at']

    #DROP FIELDS , CHANGE TIME FORMAT, add dataetime index
    rucio_data = rucio_data.drop(fields_to_drop, axis=1)
    for timestamp in timestamps:
        rucio_data[timestamp]= pd.to_datetime(rucio_data[timestamp], infer_datetime_format=True)
    rucio_data['delay'] = rucio_data['started_at'] - rucio_data['submitted_at']
    rucio_data['delay'] = rucio_data['delay'].astype('timedelta64[s]')
    
    rucio_data = rucio_data.sort_values(by='submitted_at')
    
    # Reindex data with 'submittedd at timestamp'
    rucio_data.index = pd.DatetimeIndex(rucio_data['submitted_at'])
    
    #remove all timestamp columns
    rucio_data = rucio_data.drop(timestamps, axis=1)
    
    # encode categorical data
 
    if use_cache==True:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=True)
    else:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=False)

    rucio_data['src-rse'] = src_encoder.transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.transform(rucio_data['transfer-endpoint'])
    
    return rucio_data

def rescale_data(rucio_data):
    
    # Normalization
    
    byte_scaler = MinMaxScaler(feature_range=(0, 1))
    delay_scaler = MinMaxScaler(feature_range=(0, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1))
    
    byte_scaler = byte_scaler.fit(rucio_data['bytes'])
    delay_scaler = delay_scaler.fit(rucio_data['delay'])
    duration_scaler = duration_scaler.fit(rucio_data['duration'])
    
    rucio_data['bytes'] = byte_scaler.transform(rucio_data['bytes'])
    rucio_data['delay'] = delay_scaler.transform(rucio_data['delay'])
    rucio_data['duration'] = duration_scaler.transform(rucio_data['duration'])
    
    return rucio_data, byte_scaler, delay_scaler, duration_scaler
    
    
def split_data(rucio_data,durations, num_timesteps=50, split_frac=0.9):
    
#     slice_size = batch_size*num_timesteps
    print(rucio_data.shape[0])
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps )
    print('Total Batches : {}'.format(n_batches))
    
    inputs=[]
    outputs=[]
    for i in range(0,n_batches):
        v = rucio_data[i:i+num_timesteps]
        w = durations[i+num_timesteps]
        inputs.append(v)
        outputs.append(w)
    
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)
    
    split_idx = int(inputs.shape[0]*split_frac)
    trainX, trainY = inputs[:split_idx], outputs[:split_idx]
    testX, testY = inputs[split_idx:], outputs[split_idx:]
    print('Training Data shape:',trainX.shape, trainY.shape)
    print('Test Data shape: ',testX.shape, testY.shape)
    return trainX, trainY, testX, testY

def plot_graphs_and_rescale(data):
    
    durations = data['duration']
    durations.plot()
    plt.ylabel('durations(seconds)')
    plt.show()

    filesize = data['bytes']
    filesize.plot(label='filesize(bytes)')
    plt.ylabel('bytes')
    plt.show()

    delays = data['delay']
    delays.plot(label='delay(seconds)')
    plt.ylabel('delay')
    plt.show()
    
    print('rescaling input continuous variables : filesizes, queue-times, transfer-durations')
    data, byte_scaler, delay_scaler, duration_scaler = rescale_data(data)

    plt.plot(data['bytes'], 'r', label='filesize')
    plt.plot(data['duration'], 'y', label='durations')
    plt.plot(data['delay'],'g', label='queue-time')
    plt.legend()
    plt.xticks(rotation=20)
    plt.show()
    
    return data, byte_scaler, delay_scaler, duration_scaler
# def 
# get_and_preprocess_data():
    





path = '../' # Change this as you need.

def plot_rucio(path='../'):
    abspaths = []
    for fn in os.listdir(path):
        if 'atlas_rucio' in fn:
            abspaths.append(os.path.abspath(os.path.join(path, fn)))
    print("\n".join(abspaths))
    
    for path in abspaths:
        print('reading : ',path)
        data = pd.read_csv(path)
        print('shape :', data.shape)
        data  = preprocess_data(data)
        data, byte_scaler, delay_scaler, duration_scaler = plot_graphs_and_rescale(data)


plot_rucio(path='../data/')


data.shape


data.head()


# # Applying Seasonal Trend Decomposition
# 

duration = data['duration']
filesizes = data['bytes']
queue_times = data['delay']

duration.plot()
plt.show()

filesizes.plot()
plt.show()

queue_times.plot()
plt.show()


# resampling data by the Hour
duration_per_min = duration.resample('H', how='mean').ffill()

res = sm.tsa.seasonal_decompose(duration_per_min)  
res.plot()
plt.show()

# duration.head()


filesizes_per_hour = filesizes.resample('H', how='mean').ffill()

res = sm.tsa.seasonal_decompose(filesizes_per_hour)  
res.plot()
plt.show()


delay_per_hour = queue_times.resample('H', how='mean').ffill()

res = sm.tsa.seasonal_decompose(delay_per_hour)  
res.plot()
plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_context('poster')


# # Current way to label anomalies
# 
# ```python
# threshold=600
# 
# data['err'] = data['duration']-data['prediction']
# 
# def f(x):
#     if x<=threshold:
#         return 'normal'
#     else:
#         return 'anomaly'
# 
# data['correct_label']= data['err'].apply(lambda x : f(x))
# data=data.drop('err', axis=1)
# ```
# 

file='data/rucio_transfer-events-2017.08.06.csv'
data = pd.read_csv(file)
data = data.drop('Unnamed: 0', axis=1)
# data=data.set_index(['submitted_at'])
print(data.head(5), '\n --------------------- \n')
data.info()


plt.plot(data['duration'] / 60, 'g')
plt.plot(data['prediction']/ 60, 'y')
# data['duration'].plot()
# data['prediction'].plot()
plt.show()


errors= data['duration'] - data['prediction']
print(errors.shape)
errors=np.reshape(errors, [errors.shape[0],1])
print(errors.shape)
plt.plot(errors, 'or')

plt.show()


errs = errors/60
print(np.max(errs))

bins=15
# bins=[-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60]
arr= plt.hist(errs, bins=bins)
for i in range(bins):
    plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
plt.show()

arr2 = plt.hist(np.absolute(errs), bins=bins)
for i in range(bins):
    plt.text(arr2[1][i],arr2[0][i],str(int(arr2[0][i])))
plt.show()


i_1=0
i_2=0
i_3 = 0
i_4 = 0
i_5 = 0
i_10 = 0
i_20 = 0
i_30=0
i_40 =0
i_50 = 0
i_60=0
j=0
k=0
for err in errors:
    if err<=0:
        k+=1
    if np.absolute(err/ 60)<=1:
        i+=1
    if np.absolute(err/ 60)<=2:
        i_2+=1
    if np.absolute(err/ 60)<=3:
        i_3+=1
    if np.absolute(err/ 60)<=4:
        i_4+=1
    if np.absolute(err/ 60)<=5:
        i_5+=1
    if np.absolute(err/ 60)<=10:
        i_10+=1
    if np.absolute(err/ 60)<=20:
        i_20+=1
    if np.absolute(err/ 60)<=30:
        i_30+=1
    if np.absolute(err/ 60)<=40:
        i_40+=1
    if np.absolute(err/ 60)<=50:
        i_50+=1
    if np.absolute(err/ 60)<=60:
        i_60+=1
    else:
        j+=1
print('total values with error less than 1 minutes : {}  percentage :{} %'.format(i, (i/len(errors) *100)))
print('total values with error less than 2 minutes : {}  percentage :{} %'.format(i_2, (i_2/len(errors) *100)))
print('total values with error less than 3 minutes : {}  percentage :{} %'.format(i_3, (i_3/len(errors) *100)))
print('total values with error less than 5 minutes : {}  percentage :{} %'.format(i_5, (i_5/len(errors) *100)))
print('total values with error less than 10 minutes : {}  percentage :{} %'.format(i_10, (i_10/len(errors) *100)))
print('total values with error less than 30 minutes : {}  percentage :{} %'.format(i_30, (i_30/len(errors) *100)))
print('total values with error more than an hour : {}  percentage :{} %'.format(j, (j/len(errors) *100)))
print('total values with negative errors i.e transfers faster tha predicted by the model(positive anomalies) : {}  percentage :{} %'.format(k, (k/len(errors) *100)))
max_err = np.max(np.absolute(errors))
print('max error :{}  minutes'.format(max_err/60))


# sns.barplot(x='submitted_at', y=err_min, data=data)


# # Scatterplots
# 

cond = data['label']=='anomaly'
anomalies= data[cond]
normal_data = data[cond!=True]
assert len(normal_data)+len(anomalies)==len(data)


fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
# x_norm = []
# y_norm = []
# x_anom = []
# y_anom = []
ax.scatter(normal_data['duration'], normal_data['prediction'],c='green', s=200.0, label='normal', alpha=0.3, edgecolors='none')
ax.scatter(anomalies['duration'], anomalies['prediction'],c='red', s=200.0, label='anomaly', alpha=0.3, edgecolors='none')
ax.plot(normal_data['duration'],normal_data['duration'], 'b', label='ideal')
# ax.plot(data['duration'], data['duration'], 'y', label='reality')
ax.legend()
plt.xlabel('duration', fontsize=16)
plt.ylabel('prediction', fontsize=16)


plt.scatter(normal_data['duration'], normal_data['prediction'], c='green')


plt.scatter(anomalies['duration'], anomalies['prediction'], c='red')


anomalies.shape


normal_data.shape


print('% of anomalies from {} events = {:.3f} % ; ({}) '.format(data.shape[0], (anomalies.shape[0]/data.shape[0])*100,anomalies.shape[0]))


anomalies.info()


delta= anomalies['duration']-anomalies['prediction']
plt.plot(delta, 'ro')
plt.show()


size_gb=anomalies['bytes']/1073741824
# print(size_gb.value_counts())

count, division = np.histogram(size_gb, bins = range(0,60))
size_gb.hist(bins=division)
count,division


c


data.info()


# # Label anomalies --- using a simple threshold..improved version --ignore "positive anomalies"
# 

threshold=600

data['err'] = data['duration']-data['prediction']

def f(x):
    if x<=threshold:
        return 'normal'
    else:
        return 'anomaly'

data['correct_label']= data['err'].apply(lambda x : f(x))
data=data.drop('err', axis=1)
data.head()


cond = data['correct_label']=='anomaly'
anomalies= data[cond]
normal_data = data[cond!=True]
assert len(normal_data)+len(anomalies)==len(data)


len(anomalies)


delta= anomalies['duration']-anomalies['prediction']
delta= delta/60
plt.plot(delta, 'ro')
plt.xlabel('#events')
plt.ylabel('duration errors in minutes')
plt.show()


plt.scatter(anomalies['bytes'], anomalies['prediction'], c='yellow', alpha=0.3,  s=100, edgecolors=None, label='predicted')
plt.scatter(anomalies['bytes'], anomalies['duration'], c='green',alpha=0.3, s=100, edgecolors=None, label='actual')
plt.xlabel('bytes')
plt.ylabel('transfer duration in seconds')
plt.title('Duration vs Filesizes')
plt.legend()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(normal_data['duration'], normal_data['prediction'],c='green', s=200.0, label='normal', alpha=0.3, edgecolors='none')
ax.scatter(anomalies['duration'], anomalies['prediction'],c='red', s=200.0, label='anomaly', alpha=0.3, edgecolors='none')
ax.plot(normal_data['duration'],normal_data['duration'], 'b', label='ideal')
# ax.plot(data['duration'], data['duration'], 'y', label='reality')
ax.legend()
plt.xlabel('duration', fontsize=16)
plt.ylabel('prediction', fontsize=16)


size_gb=anomalies['bytes']/1073741824
# print(size_gb.value_counts())

count, division = np.histogram(size_gb, bins = range(0,60))
size_gb.hist(bins=division)
count,division


c= data['bytes']>=10*1073741824
v= data[c]
v.head()


v.shape


v


data['activity'].unique()


a=data['activity']=='Data Rebalancing'
a=data[a]


duration_minutes=data['duration']/60
bins=range(0,int(np.max(duration_minutes)), 5)
count, division = np.histogram(duration_minutes, bins = range(0,int(np.max(duration_minutes)), 5))
duration_minutes.hist(bins=division)
print(count,division)
for i in range(0,len(bins)-1):
    plt.text(division[i],count[i]+150000,str(int(count[i])), rotation=90)
plt.show()

duration_minutes=data['prediction']/60
bins=range(0,int(np.max(duration_minutes)), 5)
count, division = np.histogram(duration_minutes, bins = range(0,int(np.max(duration_minutes)), 5))
duration_minutes.hist(bins=division)
print(count,division)
for i in range(0,len(bins)-1):
    plt.text(division[i],count[i]+150000,str(int(count[i])), rotation=90)
    





import datetime as datetime  
import numpy as np
import seaborn as sns
import pandas as pd  
import statsmodels.api as sm  
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import os
get_ipython().magic('matplotlib inline')
sns.set_context('poster')


data = pd.read_csv('../atlas_rucio-events-2017.06.21.csv')
print(data.info())
data.shape


def load_encoders():
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    t_endpoint_encoder = LabelEncoder()
    
    src_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    dst_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    type_encoder.classes_ = np.load('encoders/type.npy')
    activity_encoder.classes_ = np.load('encoders/activity.npy')
    protocol_encoder.classes_ = np.load('encoders/protocol.npy')
    t_endpoint_encoder.classes_ = np.load('encoders/endpoint.npy')
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def train_encoders(rucio_data, use_cache=True):
    
    if use_cache:
        if os.path.isfile('encoders/ddm_rse_endpoints.npy') and os.path.isfile('encoders/activity.npy'):
            print('using cached LabelEncoders for encoding data.....')
            src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder=load_encoders()
        else:
            print('NO cache found')
    else:
        print('No cached encoders found ! Training Some New Ones using input data!')
        src_encoder = LabelEncoder()
        dst_encoder = LabelEncoder()
        type_encoder = LabelEncoder()
        activity_encoder = LabelEncoder()
        protocol_encoder = LabelEncoder()
        t_endpoint_encoder = LabelEncoder()

        src_encoder.fit(rucio_data['src-rse'].unique())
        dst_encoder.fit(rucio_data['dst-rse'].unique())
        type_encoder.fit(rucio_data['src-type'].unique())
        activity_encoder.fit(rucio_data['activity'].unique())
        protocol_encoder.fit(rucio_data['protocol'].unique())
        t_endpoint_encoder.fit(rucio_data['transfer-endpoint'].unique())

        np.save('encoders/src.npy', src_encoder.classes_)
        np.save('encoders/dst.npy', dst_encoder.classes_)
        np.save('encoders/type.npy', type_encoder.classes_)
        np.save('encoders/activity.npy', activity_encoder.classes_)
        np.save('encoders/protocol.npy', protocol_encoder.classes_)
        np.save('encoders/endpoint.npy', t_endpoint_encoder.classes_)
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def preprocess_data(rucio_data, use_cache=True):
    
    fields_to_drop = ['account','reason','checksum-adler','checksum-md5','guid','request-id','transfer-id','tool-id',
                      'transfer-link','name','previous-request-id','scope','src-url','dst-url', 'Unnamed: 0']
    timestamps = ['started_at', 'submitted_at','transferred_at']

    #DROP FIELDS , CHANGE TIME FORMAT, add dataetime index
    rucio_data = rucio_data.drop(fields_to_drop, axis=1)
    for timestamp in timestamps:
        rucio_data[timestamp]= pd.to_datetime(rucio_data[timestamp], infer_datetime_format=True)
    rucio_data['delay'] = rucio_data['started_at'] - rucio_data['submitted_at']
    rucio_data['delay'] = rucio_data['delay'].astype('timedelta64[s]')
    
    rucio_data = rucio_data.sort_values(by='submitted_at')
    rucio_data.index = pd.DatetimeIndex(rucio_data['submitted_at'])

    rucio_data = rucio_data.drop(timestamps, axis=1)
    
    # Normalization
    
    byte_scaler = MinMaxScaler(feature_range=(0, 1))
    delay_scaler = MinMaxScaler(feature_range=(0, 1))
    duration_scaler = MinMaxScaler(feature_range=(0, 1))
    
    byte_scaler = byte_scaler.fit(rucio_data['bytes'])
    delay_scaler = delay_scaler.fit(rucio_data['delay'])
    duration_scaler = duration_scaler.fit(rucio_data['duration'])
    
    rucio_data['bytes'] = byte_scaler.transform(rucio_data['bytes'])
    rucio_data['delay'] = delay_scaler.transform(rucio_data['delay'])
    rucio_data['duration'] = duration_scaler.transform(rucio_data['duration'])
    
    # encode categorical data
 
    if use_cache==True:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=True)
    else:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=False)

    rucio_data['src-rse'] = src_encoder.transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.transform(rucio_data['transfer-endpoint'])
    
    return rucio_data, byte_scaler, delay_scaler, duration_scaler

def split_data(rucio_data,durations, num_timesteps=50, split_frac=0.9):
    
#     slice_size = batch_size*num_timesteps
    print(rucio_data.shape[0])
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps )
    print('Total Batches : {}'.format(n_batches))
    
    inputs=[]
    outputs=[]
    for i in range(0,n_batches):
        v = rucio_data[i:i+num_timesteps]
        w = durations[i+num_timesteps]
        inputs.append(v)
        outputs.append(w)
    
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)
    
    split_idx = int(inputs.shape[0]*split_frac)
    trainX, trainY = inputs[:split_idx], outputs[:split_idx]
    testX, testY = inputs[split_idx:], outputs[split_idx:]
    print('Training Data shape:',trainX.shape, trainY.shape)
    print('Test Data shape: ',testX.shape, testY.shape)
    return trainX, trainY, testX, testY

def plot_graphs(data):
    
    durations = data['duration']
    durations.plot()
    plt.ylabel('durations')
    plt.show()

    filesize = data['bytes']
    filesize.plot(label='filesize')
    plt.ylabel('bytes')
    plt.show()

    delays = data['delay']
    delays.plot(label='delay')
    plt.ylabel('delay')
    plt.show()

    plt.plot(filesize, 'r', label='filesize')
    plt.plot(durations, 'y', label='durations')
    plt.plot(delays,'g', label='queue-time')
    plt.legend()
    plt.show()


# data = data[:10000]


data, byte_scaler, delay_scaler, duration_scaler  = preprocess_data(data)


data.head(20)


def plot_graphs(data):
    
    durations = data['duration']
    durations.plot()
    plt.ylabel('durations')
    plt.show()

    filesize = data['bytes']
    filesize.plot(label='filesize')
    plt.ylabel('bytes')
    plt.show()

    delays = data['delay']
    delays.plot(label='delay')
    plt.ylabel('delay')
    plt.show()

    plt.plot(filesize, 'r', label='filesize')
    plt.plot(durations, 'y', label='durations')
    plt.plot(delays,'g', label='queue-time')
    plt.legend()
    plt.xticks(rotation=20)
    plt.show()
    
plot_graphs(data)


path = '../' # Change this as you need.

def plot_rucio(path='../'):
    abspaths = []
    for fn in os.listdir(path):
        if 'atlas_rucio' in fn:
            abspaths.append(os.path.abspath(os.path.join(path, fn)))
    print("\n".join(abspaths))
    
    for path in abspaths:
        print('reading : ',path)
        data = pd.read_csv(path)
        print('shape :', data.shape)
        data, byte_scaler, delay_scaler, duration_scaler  = preprocess_data(data)
        plot_graphs(data)


plot_rucio()


# # Found New "Activity" label ....Added this to the cached encoder classes
# 

act_encoder = LabelEncoder()
act_encoder.classes_ = np.load('encoders/activity.npy')
print(act_encoder.classes_)
a = act_encoder.classes_
type(a)
# a=np.append(a, 'default')
a


# # Total Two faulty/Uncommon activity labels were found:
# 
# * 'default'
# * 'Debug'
# 
# ## Both were added and stored in the cache files
# 




