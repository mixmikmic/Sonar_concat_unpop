# ### Notebook name: MarkRecaptureSim.ipynb
# #### Author: Sreejith Menon (smenon8@uic.edu)
# 
# #### Mark Recapture Simulation Notebook
# 
# Recognize individuals that appeared on day 1 and then on day 2
# Individuals that appear on day 1 are **marks**.    
# If the same individuals appear on day 2 then these are **recaptures**
# 
# *Appeared means the individuals who were photographed on day 1 as well as day 2*
# 
# To change the behavior of the script only change the values of the dictionary days. Changing days dict can filter out the images to the days the images were clicked. 
# 
# The first level calculations are based on what pictures were clicked and by applying the Pertersen-Lincoln Index calculations
# 
# The second level calculations will filter out only the images that were shared (only highly shared images with proportion >= 80).
# 

import json
from datetime import datetime
import DataStructsHelperAPI as DS
import JobsMapResultsFilesToContainerObjs as J
import importlib
importlib.reload(J)
import pandas as pd
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_online()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import MarkRecapHelper as MR
import importlib
importlib.reload(MR)
import DeriveFinalResultSet as DRS
from collections import Counter


days = {'2015-02-18' : '2015-02-18',
 '2015-02-19' : '2015-02-19',
 '2015-02-20' : '2015-02-20',
 '2015-02-25' : '2015-02-25',
 '2015-02-26' : '2015-02-26',
 '2015-03-01' : '2015-03-01',
 '2015-03-02' : '2015-03-02'}

nidMarkRecapSet = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json",days)


# #### Visualizations on how pictures were taken.
# Visualizations on how individuals were identified across different days of the Great Zebra Count (GZC) rally. There are visuals which show how many individuals were identified on the first day, how many individuals were seen only on that day and how many individuals were first seen on that day.
# 

# How many individuals were identified on each day, 
# i.e. how many different individuals did we see each day?

indsPerDay = {}
for nid in nidMarkRecapSet:
    for day in nidMarkRecapSet[nid]:
        indsPerDay[day] = indsPerDay.get(day,0) + 1
        
df1 = pd.DataFrame(indsPerDay,index=['IndsIdentified']).transpose()

fig1 = df1.iplot(kind='bar',filename='Individuals seen per day',title='Individuals seen per day')
iframe1 = fig1.embed_code


# How many individuals did we see only on that day, 
# i.e. how many individuals were only seen that day and not any other day.

uniqIndsPerDay = {}
for nid in nidMarkRecapSet:
    if len(nidMarkRecapSet[nid]) == 1:
        uniqIndsPerDay[nidMarkRecapSet[nid][0]] = uniqIndsPerDay.get(nidMarkRecapSet[nid][0],0) + 1
        
df2 = pd.DataFrame(uniqIndsPerDay,index=['IndsIdentifiedOnlyOnce']).transpose()

fig2 = df2.iplot(kind='bar',filename='Individuals seen only that day',title='Individuals seen only that day')
iframe2 = fig2.embed_code


# How many individuals were first seen on that day, i.e. the unique number of animals that were identified on that day.
# The total number of individuals across all the days is indeed equal to all the unique individuals in the database. We have 1997 identified individuals.
indsSeenFirst = {}
for nid in nidMarkRecapSet:
    indsSeenFirst[min(nidMarkRecapSet[nid])] = indsSeenFirst.get(min(nidMarkRecapSet[nid]),0) + 1
    
df3 = pd.DataFrame(indsSeenFirst,index=['FirstTimeInds']).transpose()

fig3 = df3.iplot(kind='bar',filename='Individuals first seen on that day',title='Individuals first seen on that day')
iframe3 = fig3.embed_code


df1['IndsIdentifiedOnlyOnce'] = df2['IndsIdentifiedOnlyOnce']
df1['FirstTimeInds'] = df3['FirstTimeInds']

df1.columns = ['Total inds seen today','Inds seen only today','Inds first seen today']
fig4 = df1.iplot(kind='bar',filename='Distribution of sightings',title='Distribution of sightings')
iframe4 = fig4.embed_code


# ### Actual Mark-Recapture Calculations
# 
# #### The below snippets only consider photos clicked and _NOT_ shared data.
# 

days = {'2015-03-01' : 1,
        '2015-03-02' : 2 }


# Entire population estimate (includes giraffes and zebras)
nidMarkRecapSet = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet)
print("Population of all animals = %f" %population)
marks,recaptures


nidMarkRecapSet_Zebras = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,'zebra_plains',shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet_Zebras)
print("Population of zebras = %f" %population)
marks,recaptures


nidMarkRecapSet_Giraffes = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,'giraffe_masai',shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet_Giraffes)
print("Population of giraffes = %f" %population)
marks,recaptures


# #### The below snippets consider the share data
# 

nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       None,
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of all animals = %f" %population)
marks,recaptures


nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       'zebra_plains',
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of zebras = %f" %population)
marks,recaptures


nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       'giraffe_masai',
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of giraffes = %f" %population)
marks,recaptures


days = [{'2004' : 1, '2005' : 2 },{'2005' : 1, '2006' : 2 }, {'2006' : 1, '2007' : 2 }, {'2007' : 1, '2008' : 2 }, {'2008' : 1, '2009' : 2 }, {'2009' : 1, '2010' : 2 }, {'2010' : 1, '2011' : 2 }, {'2014' : 1, '2015' : 2 }, {'2015' : 1, '2016' : 2}, {'2016' : 1, '2017' : 2}] 
for i in range(len(days)):
    nidMarkRecapSet = MR.genNidMarkRecapDict("/tmp/gir_new_exif.json",
                                         "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json",
                                         "../data/Flickr_IBEIS_Giraffe_Ftrs_aid_features.json",
                                         "../FinalResults/rankListImages_expt2.csv", # this is useless
                                         days[i],
                                         shareData='other',
                                        filterBySpecies='giraffe_reticulated')
    marks, recaps, population, confidence = MR.applyMarkRecap(nidMarkRecapSet)
      
    print("Estimate for the year : "  + ' & '.join(list(days[i].keys())))
    print("Number of marks : %i" %marks)
    print("Number of recaptures : %i" %recaps)
    print("Estimated population : %f" %population)
    print()


inGidAidMapFl, inAidFtrFl = "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json", "../data/Flickr_IBEIS_Ftrs_aid_features.json",

gidNid = DRS.getCountingLogic(inGidAidMapFl,inAidFtrFl,"NID",False)
flickr_nids = list(gidNid.values())
flickr_nids = [item for sublist in flickr_nids for item in sublist]

print("Number of unique individuals identified : %i" %len(set(flickr_nids)))


occurence = Counter(flickr_nids)


inExifFl = "../data/Flickr_EXIF_full.json"
with open(inExifFl, "r") as fl:
    obj = json.load(fl)


'''
lat in between -1.50278 and 1.504953
long in between 35.174045 and 38.192836
'''

gids_geoTagged = [gid for gid in obj.keys() if int(gid) < 1702 and obj[gid]['lat'] != 0 ]
gids_nairobi = [gid for gid in obj.keys() if int(gid) <1702 and obj[gid]['lat'] >= -1.50278 and obj[gid]['lat'] <= 1.504953 and obj[gid]['long'] >= 35.174045 and obj[gid]['long'] <= 38.192836 ]
gids_zoo = list(set(gids_geoTagged) - set(gids_nairobi))


import DeriveFinalResultSet as DRS, DataStructsHelperAPI as DS


inGidAidMapFl, inAidFtrFl = "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json", "../data/Flickr_IBEIS_Ftrs_aid_features.json",

gidNid = DRS.getCountingLogic(inGidAidMapFl,inAidFtrFl,"NID",False)


locs = []
for gid in gidNid.keys():
    if gid in gids:
        for nid in gidNid[gid]:
            locs.append((obj[gid]['lat'], obj[gid]['long']))


nid_gid = DS.flipKeyValue(gidNid)


nids_zoo = []

for gid in gidNid.keys():
    if gid in gids_zoo:
        nids_zoo.extend(gidNid[gid])


len(gids_zoo), len(nids_zoo)


# removing all nids that are in zoos, with it you will also remove the other occurences of images in which that individual occurs.
nids_only_wild_gid =  {nid : nid_gid[nid] for nid in nid_gid.keys() if nid not in nids_zoo}
nids_zoo_wild_gid = {nid : nid_gid[nid] for nid in nid_gid.keys() if nid in nids_zoo}


len(list(nids_only_wild_gid.values())), len(nids_zoo_wild_gid.values())


len({gid for sublist in list(nids_only_wild_gid.values()) for gid in sublist})


len({gid for sublist in list(nids_zoo_wild_gid.values()) for gid in sublist})


max(list(map(int, list(gidNid.keys()))))


gidNid['110']


l =[12,12,12,12,12]
l.extend([1,2,3])


a = 5

print("a = %d" %a)


MR.genNidMarkRecapDict("../data/Flickr_Giraffe_EXIF.json",
                                         "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json",
                                         "../data/Flickr_IBEIS_Giraffe_Ftrs_aid_features.json",
                                         "../FinalResults/rankListImages_expt2.csv", # this is useless
                                         days[i],
                                         shareData='other',
                                        filterBySpecies='giraffe_reticulated')


gidSpecies


{ gid : gidsDayNumFull[gid] for gid in gidsDayNumFull  if gid in gidSpecies.keys() and 'giraffe_reticulated' in gidSpecies[gid]}


gidsDayNumFull


gir_exif= DS.json_loader("../data/Flickr_Giraffe_EXIF.json")
gid_fl = DS.json_loader("../data/Flickr_Giraffes_imgs_gid_flnm_map.json")


fl_gid = DS.flipKeyValue(gid_fl)

new_exif = {}
for fl in gir_exif.keys():
    new_exif[fl_gid[fl]] = gir_exif[fl]


with open("/tmp/gir_new_exif.json", "w") as fl:
    json.dump(new_exif, fl, indent=4)


df = pd.DataFrame(new_exif).transpose()


df['date'] = pd.to_datetime(df['date'])


df['year'] = df.date.apply(lambda x : x.year)


year_num = dict(df.groupby('year').count()['date'])


year_num = [(key,year_num[key]) for key in year_num.keys()]

X = [year_num[i][0] for i in range(len(year_num)) if year_num[i][0] > 1999]
Y = [year_num[i][1] for i in range(len(year_num)) if year_num[i][0] > 1999]


import plotly.graph_objs as go

data = [go.Bar(
            x=X,
            y=Y
    )]
layout = go.Layout(
    annotations=[
        dict(x=xi,y=yi,
             text=str(yi),
             xanchor='center',
             yanchor='bottom',
             showarrow=False,
        ) for xi, yi in zip(X, Y)]
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic-bar')


year_num[0][1]





# Image processing 
# 

from skimage import io, color, feature, transform
import numpy as np
from skimage.filters import rank
import plotly.plotly as py
import cufflinks as cf # this is necessary to link pandas to plotly
import plotly.graph_objs as go
cf.go_offline()
from plotly.offline import plot, iplot
from skimage.morphology import disk
from matplotlib import pyplot as PLT


rgbImg = io.imread("/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/8598.jpeg")


hsvImg = color.rgb2hsv(rgbImg)


red = np.array([pix[0] for row in rgbImg for pix in row])
green = np.array([pix[1] for row in rgbImg for pix in row])
blue = np.array([pix[2] for row in rgbImg for pix in row])

hue = np.array([pix[0] for row in hsvImg for pix in row])
saturation = np.array([pix[1] for row in hsvImg for pix in row])
value = np.array([pix[2] for row in hsvImg for pix in row])


y = 0.299 * red + 0.587 * green + 0.114 * blue
(np.max(y) - np.min(y))/np.mean(y)


a = np.array(rgbImg)


a


hue


data = [
    go.Histogram(
    x = hue,
    nbinsx = 12
    )
]

plot(data)


a, _ = np.histogram(saturation, bins=5)
np.std(a)


gray = color.rgb2gray(rgbImg)


gray.shape


glcm = feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
glcm.shape


entropy = rank.entropy(gray, disk(5))


entropy.shape


feature.hog(resized_img, orientations=8)


resized_img = transform.resize(gray, (600,600))


resized_img.shape


PLT.imshow(flipped)
PLT.show()


left = resized_img.transpose()[:300].transpose()
right = resized_img.transpose()[300:].transpose()


left.shape, right.shape


I = np.identity(600)[::-1]


flipped = I.dot(right)


inner = feature.hog(left) - feature.hog(flipped)


np.linalg.norm(inner)


inner


a =dict(a=1, b = 2, c=3) 
a.update(dict(d = 4))


imgFlNms = ["/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/%i.jpeg" %i for i in range(1,9405)]


imgFlNms





# ### Notebook name: VisualizeResults.ipynb
# #### Author: Sreejith Menon (smenon8@uic.edu)
# 
# Contains code for generating various visuals and plots that are currently hosted in [Results home page](https://compbio.cs.uic.edu/~sreejith/ResultsDashboard.html).     
# 
# Highlights:
# * Rank list of all images with results
# * Share proportion comparisons based on single features
# * Share proportion compariosns based on two features (all-pairs)
# * Position Bias - Share proportion changes bases on the position of the image in the mechanical turk album
# * Responses to general questions asked in the mechanical turk jobs.
# 

import csv
import htmltag as HT
import DeriveFinalResultSet as drs
import JobsMapResultsFilesToContainerObjs as ImageMap
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import statistics as s
importlib.reload(drs)
importlib.reload(ImageMap)
pd.set_option('display.max_colwidth', -1)
from collections import Counter
import plotly.plotly as py
import cufflinks as cf
cf.go_offline()
import GetPropertiesAPI as GP
importlib.reload(drs)


def genHTMLTableFiles(shrCntsObj):
    shrPropDict = drs.getShrProp(shrCntsObj)
    
    df = pd.DataFrame(shrPropDict,index = ['Share Proportion']).transpose()
    
    return df,df.to_html(bold_rows = False)


# Generate rank list of all images by share proportion
rnkFlLst = []
with open("../FinalResults/rankListImages_expt2.csv","r") as rnkFl:
    rnkFlCsv = csv.reader(rnkFl)
    header = rnkFlCsv.__next__()
    for row in rnkFlCsv:
        rnkFlLst.append(row)
        
thTgs = []
trTgs = []

trTgs.append(HT.tr(HT.th("GID"),HT.th("Share count"),HT.th("Not Share count"),
                  HT.th("Total Count"),HT.th("Share Proportion"),HT.th("Image")))

for tup in rnkFlLst:
    tdGid = HT.td(tup[0])
    tdShare = HT.td(tup[1])
    tdNotShare = HT.td(tup[2])
    tdTot = HT.td(tup[3])
    tdProp = HT.td(tup[4])
    url = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/" + tup[0] + ".jpeg"
    tdImg = HT.td(HT.img(src = url,alt = "Unavailable",width = "300"))
    trTgs.append(HT.tr(tdGid,tdShare,tdNotShare,tdTot,tdProp,tdImg))
    
fullFile = HT.html(HT.body(HT.table(HT.HTML('  \n'.join(trTgs)),border="1")))

outputFile = open("../data/resultsExpt2RankList1.html","w")
outputFile.write(fullFile)
outputFile.close()


# Generate the share prortion tables for pair wise features.

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","AGE",drs.imgJobMap,1,100)
h3_1 = HT.h3("Data-Frame by SPECIES-AGE")
df1,tb1 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","SEX",drs.imgJobMap,1,100)
h3_2 = HT.h3("Data-Frame by SPECIES-SEX")
df2,tb2 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","VIEW_POINT",drs.imgJobMap,1,100)
h3_3 = HT.h3("Data-Frame by SPECIES-VIEW_POINT")
df3,tb3 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","QUALITY",drs.imgJobMap,1,100)
h3_4 = HT.h3("Data-Frame by SPECIES-QUALITY")
df4,tb4 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_5 = HT.h3("Data-Frame by SPECIES-EXEMPLAR_FLAG")
df5,tb5 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","SPECIES",drs.imgJobMap,1,100)
h3_6 = HT.h3("Data-Frame by VIEW_POINT-SPECIES")
df6,tb6 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","AGE",drs.imgJobMap,1,100)
h3_7 = HT.h3("Data-Frame by VIEW_POINT-AGE")
df7,tb7 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","SEX",drs.imgJobMap,1,100)
h3_8 = HT.h3("Data-Frame by VIEW_POINT-SEX")
df8,tb8 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","QUALITY",drs.imgJobMap,1,100)
h3_9 = HT.h3("Data-Frame by VIEW_POINT-QUALITY")
df9,tb9 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_10 = HT.h3("Data-Frame by VIEW_POINT-EXEMPLAR_FLAG")
df10,tb10 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******
d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","SPECIES",drs.imgJobMap,1,100)
h3_11 = HT.h3("Data-Frame by SEX-SPECIES")
df11,tb11 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","AGE",drs.imgJobMap,1,100)
h3_12 = HT.h3("Data-Frame by SEX-AGE")
df12,tb12 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","QUALITY",drs.imgJobMap,1,100)
h3_13 = HT.h3("Data-Frame by SEX-QUALITY")
df13,tb13 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_14 = HT.h3("Data-Frame by SEX-EXEMPLAR_FLAG")
df14,tb14 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","VIEW_POINT",drs.imgJobMap,1,100)
h3_15 = HT.h3("Data-Frame by SEX-VIEW_POINT")
df15,tb15 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******
d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","SPECIES",drs.imgJobMap,1,100)
h3_16 = HT.h3("Data-Frame by AGE-SPECIES")
df16,tb16 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","SEX",drs.imgJobMap,1,100)
h3_17 = HT.h3("Data-Frame by AGE-SEX")
df17,tb17 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","QUALITY",drs.imgJobMap,1,100)
h3_18 = HT.h3("Data-Frame by AGE-QUALITY")
df18,tb18 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_19 = HT.h3("Data-Frame by AGE-EXEMPLAR_FLAG")
df19,tb19 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","VIEW_POINT",drs.imgJobMap,1,100)
h3_20 = HT.h3("Data-Frame by AGE-VIEW_POINT")
df20,tb20 = genHTMLTableFiles(d)

fullFl = HT.html(HT.body(HT.HTML(h3_1),HT.HTML(tb1),
                        HT.HTML(h3_2),HT.HTML(tb2),
                        HT.HTML(h3_3),HT.HTML(tb3),
                        HT.HTML(h3_4),HT.HTML(tb4),
                        HT.HTML(h3_5),HT.HTML(tb5),
                        HT.HTML(h3_6),HT.HTML(tb6),
                        HT.HTML(h3_7),HT.HTML(tb7),
                        HT.HTML(h3_8),HT.HTML(tb8),
                        HT.HTML(h3_9),HT.HTML(tb9),
                        HT.HTML(h3_10),HT.HTML(tb10),
                        HT.HTML(h3_11),HT.HTML(tb11),
                        HT.HTML(h3_12),HT.HTML(tb12),
                        HT.HTML(h3_13),HT.HTML(tb13),
                        HT.HTML(h3_14),HT.HTML(tb14),
                        HT.HTML(h3_15),HT.HTML(tb15),
                        HT.HTML(h3_16),HT.HTML(tb16),
                        HT.HTML(h3_17),HT.HTML(tb17),
                        HT.HTML(h3_18),HT.HTML(tb18),
                        HT.HTML(h3_19),HT.HTML(tb19),
                        HT.HTML(h3_20),HT.HTML(tb20)
                        ))

outputFile = open("../FinalResults/twoFeatures.html","w")
outputFile.write(fullFl)
outputFile.close()


def getHTMLTabForFtr(ftr):
    d = drs.ovrallShrCntsByFtr(drs.gidAidMapFl,drs.aidFeatureMapFl,ftr,drs.imgJobMap,1,100)
    head3 = HT.h3("Data-Frame by " + ftr)
    df1,tb1 = genHTMLTableFiles(d)
    df1.sort_values(by=['Share Proportion'],ascending=False,inplace=True)
    fig = df1.iplot(kind='bar',filename=str(ftr + '_expt2' ))
    iframe = fig.embed_code

    df1.reset_index(inplace=True)
    df1.columns = [ftr,'Share Proportion']
    
    a,b,c = drs.genObjsForConsistency(drs.gidAidMapFl,drs.aidFeatureMapFl,ftr,drs.imgJobMap)    
    consistency = drs.getConsistencyDict(a,b,c)
    
    df2 = pd.DataFrame(drs.genVarStddevShrPropAcrsAlbms(consistency)).transpose()
    df2.reset_index(inplace=True)
    df2.columns = [ftr,'mean','standard_deviation','variance']
    
    df = pd.merge(df1,df2,left_on=ftr,right_on=ftr,how='left')
    
    return df,head3,df.to_html(bold_rows = False,index=False),iframe


# Generate the share prortion tables and visuals (bar diagrams) for single features.
df1,h3_1,tb1,img1 = getHTMLTabForFtr("SEX")
df2,h3_2,tb2,img2 = getHTMLTabForFtr("AGE")
df3,h3_3,tb3,img3 = getHTMLTabForFtr("SEX")
df4,h3_4,tb4,img4 = getHTMLTabForFtr("VIEW_POINT")
df5,h3_5,tb5,img5 = getHTMLTabForFtr("QUALITY")
df6,h3_6,tb6,img6 = getHTMLTabForFtr("EXEMPLAR_FLAG")
df7,h3_7,tb7,img7 = getHTMLTabForFtr("CONTRIBUTOR")
fullFl = HT.html(HT.body(HT.HTML(h3_1),HT.HTML(tb1), HT.html(img1),
                        HT.HTML(h3_2),HT.HTML(tb2), HT.html(img2),
                        HT.HTML(h3_3),HT.HTML(tb3), HT.html(img3),
                        HT.HTML(h3_4),HT.HTML(tb4), HT.html(img4),
                        HT.HTML(h3_5),HT.HTML(tb5), HT.html(img5),
                        HT.HTML(h3_6),HT.HTML(tb6), HT.html(img6),
                        HT.HTML(h3_7),HT.HTML(tb7), HT.html(img7)
                         ))

outputFile = open("../FinalResults/oneFeature.html","w")
outputFile.write(fullFl)
outputFile.close()

plt.close('all')


df = pd.DataFrame(drs.genAlbmFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,drs.imgJobMap,['SPECIES','AGE','SEX'])).transpose()
fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))))

df.to_csv("../FinalResults/albumProperties.csv",index=False)
outputFile = open("../FinalResults/albumProperties.html","w")
outputFile.write(fullFl)
outputFile.close()


# DO NOT RUN AGAIN WITHOUT TAKING BACKUP OF THE HTML FILE
imgAlbmShrs,consistency = drs.getShrPropImgsAcrossAlbms(drs.imgJobMap,1,100,"../FinalResults/shareRateSameImgsAcrossAlbums.json")
df = pd.DataFrame(imgAlbmShrs,index=["Share Proportion"]).transpose()
gidShrVarStdDevDict = drs.genVarStddevShrPropAcrsAlbms(consistency)
df2 = pd.DataFrame(gidShrVarStdDevDict).transpose()
df2.reset_index(inplace=True)
df2.columns = ['GID','Standard Deviation','Variance']

subindex = df.groupby(level=0).head(1).index
subindex2 = df2.groupby(level=0).head(1)['Standard Deviation']
subindex3 = df2.groupby(level=0).head(1)['Variance']
df.loc[subindex, 'Standard Deviation'] = subindex2.get_values()
df.loc[subindex, 'Variance'] = subindex3.get_values()

df.to_csv("../FinalResults/shareRateSameImgsAcrossAlbums.csv")

df.loc[subindex, 'URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + subindex.get_level_values(0) + '.jpeg" width = "350">'
df = df.fillna("")

fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))))
outputFile = open("../FinalResults/shareRateSameImgsAcrossAlbums.html","w")
outputFile.write(fullFl)
outputFile.close()

df = pd.DataFrame(sorted(gidShrVarStdDevDict.items(),key = lambda x : x[1]['standard_deviation'],reverse=True),columns=['GID','Stats'])
df.to_csv("../FinalResults/ImgsStdDevDesc.csv",index=False)


summaryPosCnt = drs.getPosShrProptn(drs.imgJobMap,1,100)

df = pd.DataFrame(summaryPosCnt).transpose()
cols = ['share','not_share','total','proportion']
df = df[cols]
# imgTg = HT.img(src="images/PositionBias.png")
# df.plot()
# plt.savefig("../FinalResults/PositionBias.png",bbox_inches='tight')

fig = df.iplot(kind='line',filename=str('Position_bias' + '_expt2' ))
iframe = fig.embed_code

fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))),HT.HTML(iframe))
outputFile = open("../FinalResults/PositionBias1.html","w")
outputFile.write(fullFl)
outputFile.close()


df[['share','not_share','proportion']].plot()
plt.show()


# Overall share statistics ranked by shared proportion of images along with features and tags
df = ImageMap.createMstrFl("../data/resultsFeaturesComb_expt2.csv",[ 'GID', 'AID','Album', 'AGE','EXEMPLAR_FLAG', 'INDIVIDUAL_NAME', 'NID', 'QUALITY', 'SEX', 'SPECIES','VIEW_POINT','CONTRIBUTOR'])

dfRes = pd.DataFrame.from_csv("../FinalResults/resultsExpt2RankList_Tags.csv")
dfRes.reset_index(inplace=True)
dfRes.GID = dfRes.GID.astype(str)
dfRes['URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + dfRes['GID'] + '.jpeg" width = "350">'

dfSummary = pd.merge(df,dfRes,left_on ='GID',right_on='GID')
dfSummary.sort_values(by='Proportion',ascending=False,inplace=True)
dfSummary = dfSummary[['GID','Album','AGE','INDIVIDUAL_NAME','QUALITY','SEX','SPECIES','VIEW_POINT','CONTRIBUTOR','tags','Shared','Not Shared','Total','Proportion','URL']]
dfSummary.to_csv("/tmp/ImgShrRnkListWithTags.csv",index=False)

fullFl= HT.html(HT.body(HT.HTML(dfSummary.to_html(bold_rows = False,index=False))))
outputFile = open("../FinalResults/ImgShrRnkListWithTags.html","w")
outputFile.write(fullFl)
outputFile.close()


# Visualizations for general questions asked in the mechanical turk
ans = ImageMap.genCntrsGenQues(1,100,['Answer.q1','Answer.q2'])

q1 = ans['Answer.q1']
q1 = {key : q1[key] for key in q1 if key != ''}
dfQ1 = pd.DataFrame(q1,index=['Counts']).transpose()
dfQ1.sort_values(by='Counts',ascending=False,inplace=True)

fig = dfQ1.iplot(kind='bar',filename="Frequency of posting pictures",title="How frequently do you share pictures on social media")
iframe = fig.embed_code
iframe

mapVal = {'A' : 'None',
'B' : '1 to 5',
'C' : '5 to 10',
'D' : '10 to 50',
'E' : '50 or more'}

q2 = ans['Answer.q2']
q2 = {mapVal[key] : q2[key] for key in q2 if key != ''}
dfQ2 = pd.DataFrame(q2,index=['Counts']).transpose()
dfQ2.sort_values(by='Counts',ascending=False,inplace=True)

fig = dfQ2.iplot(kind='bar',filename="Number of photos people share after safari",title="How many photos will you share on social media after a safari")
iframe2 = fig.embed_code
iframe2





import ClassiferHelperAPI as CH
import importlib
import numpy as np
import pandas as pd
importlib.reload(CH)
from ast import literal_eval
import plotly.plotly as py
import htmltag as HT
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_online()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
import csv
import plotly.graph_objs as go
import RegressionCapsuleClass as RgrCls


def plot(rgrObj,arr,arr_min,title,flNm,range,errorBar = True):
    trace1 = go.Scatter(
        x = list(rgrObj.preds),
        y = list(rgrObj.test_y),
        error_y = dict(
            type='data',
            symmetric = False,
            array = arr,
            arrayminus = arr_min,
            visible=errorBar
        ),
        mode = 'markers'
    )

    layout= go.Layout(
        title= title,
        xaxis= dict(
            title= 'Predicted Share Rate',
            ticklen= 5,
            zeroline= False,
            gridwidth= 2,
            range=[-5,110]
        ),
        yaxis=dict(
            title= 'Actual Share rate',
            ticklen= 5,
            gridwidth= 2,
            #range=range
        )
    )
    
    data = [trace1]

    fig = dict(data=data,layout=layout)
    a = py.iplot(fig,filename=flNm)
    
    return a

def runRgr(methodName,attribType):
    if attribType == "beauty":
        inpData = pd.DataFrame.from_csv("../data/BeautyFtrVector_GZC.csv")
        inpData.reindex(np.random.permutation(inpData.index))
        y = inpData['Proportion']
        inpData.drop(['Proportion'],1,inplace=True)
        train_x, test_x, train_y, test_y = CH.train_test_split(inpData, y, test_size = 0.4)
        rgr = CH.getLearningAlgo(methodName,{'fit_intercept':True})
    
        rgrObj = RgrCls.RegressionCapsule(rgr,methodName,0.8,train_x,train_y,test_x,test_y)
        
    else:
        train_data_fl = "../FinalResults/ImgShrRnkListWithTags.csv"
        infoGainFl = "../data/infoGainsExpt2.csv"
        allAttribs = CH.genAllAttribs(train_data_fl,attribType,infoGainFl)

        rgrObj = CH.buildRegrMod(train_data_fl,allAttribs,0.6,methodName,kwargs={'fit_intercept':True})
    
    rgrObj.runRgr(computeMetrics=True,removeOutliers=True)
    
    x = [i for i in range(len(rgrObj.preds))]
    errors = [list(rgrObj.test_y)[i] - list(rgrObj.preds)[i] for i in range(len(rgrObj.preds))]
    arr = [-1 * errors[i] if errors[i] < 0 else 0 for i in range(len(errors)) ]
    arr_min = [errors[i] if errors[i] > 0 else 0 for i in range(len(errors)) ]
    return rgrObj,arr,arr_min


# ## Linear Regression accuracy plots with error bars.
# 

attribTypes = ['beauty']
rgrAlgoTypes = ['linear','ridge','lasso','svr']

embedCodes = []
for attrib in attribTypes:
    code = []
    for alg in rgrAlgoTypes:
        rgrObj, arr,arr_min = runRgr(alg,attrib)
        title = "%s regression results using %s attributes" %(alg,attrib)
        flNm = "%s_regession_%s_attributes_%s" %(alg,attrib,str(True))
        # a = plot(rgrObj,arr,arr_min,title,flNm,[-100,200],errorBar = True)
        # code.append(a.embed_code)    
    embedCodes.append(code)


# ## Linear Regression plots without error bars
# 

attribTypes = ['sparse','non_sparse','non_zero','abv_mean']
rgrAlgoTypes = ['linear','ridge','lasso']

embedCodes = []
for attrib in attribTypes:
    code = []
    for alg in rgrAlgoTypes:
        rgrObj, _, _ = runRgr(alg,attrib)
        title = "%s regression results using %s attributes" %(alg,attrib)
        flNm = "%s_regession_%s_attributes_%s" %(alg,attrib,str(False))
        a = plot(rgrObj,errs,[],title,flNm,[-10,110],errorBar = False)
        code.append(a.embed_code)
    embedCodes.append(code)


for c in embedCodes[3]:
    print(c)


rgrObj, arr,arr_min = runRgr('dtree_regressor','non_sparse')


title = "%s regression results using %s attributes" %(alg,attrib)
flNm = "%s_regession_%s_attributes_%s" %(alg,attrib,str(True))
a = plot(rgrObj,arr,arr_min,title,flNm,[-100,200],errorBar = True)


attribTypes = ['sparse','non_sparse','non_zero','abv_mean']
rgrAlgoTypes = ['linear_svr','svr','dtree_regressor']
rgrAlgoTypes = ['linear']
attribTypes = ['beauty']
embedCodes = []
for attrib in attribTypes:
    code = []
    for alg in rgrAlgoTypes:
        rgrObj, _, _ = runRgr(alg,attrib)
        print("Absolute error for %s using %s : %f" %(alg,attrib,rgrObj.abserr))
        print("Mean Squared error for %s using %s : %f" %(alg,attrib,rgrObj.sqerr))
        title = "%s regression results using %s attributes" %(alg,attrib)
        flNm = "%s_regession_%s_attributes_%s" %(alg,attrib,str(False))
        a = plot(rgrObj,[],[],title,flNm,[-10,110],errorBar = False)
        code.append(a.embed_code)
    embedCodes.append(code)


len(embedCodes)


attribTypes = ['sparse','non_sparse','non_zero','abv_mean']
rgrAlgoTypes = ['linear_svr','svr','dtree_regressor']
rgrAlgoTypes = ['linear']
attribTypes = ['beauty']
embedCodes = []
for attrib in attribTypes:
    code = []
    for alg in rgrAlgoTypes:
        rgrObj, arr,arr_min = runRgr(alg,attrib)
        title = "%s regression results using %s attributes" %(alg,attrib)
        flNm = "%s_regession_%s_attributes_%s" %(alg,attrib,str(True))
        a = plot(rgrObj,arr,arr_min,title,flNm,[-100,200],errorBar = True)
        code.append(a.embed_code)    
    embedCodes.append(code)


import PopulationEstimatorFromClf as PE
import importlib
importlib.reload(PE)
import ClassiferHelperAPI as CH


train_data_fl = "../FinalResults/ImgShrRnkListWithTags.csv"
test_data_fl = "../data/full_gid_aid_ftr_agg.csv"
methodName = "linear"
attribType = "non_sparse"
infoGainFl=None
regrArgs=kwargsDict
obj, pred_results = PE.trainTestRgrs(train_data_fl,
                test_data_fl,
                methodName,
                attribType,
                 None,
                regrArgs=kwargsDict)


embedCodes





# ### Contains code for performing feature selection using Information Gain
# #### Author: Sreejith Menon(smenon8@uic.edu)
# 

import ClassiferHelperAPI as CH
import pandas as pd
import sys
import FeatureSelectionAPI as FS
import importlib
importlib.reload(FS)
from numpy import mean
import csv


# Generating attributes, converting categorical attributes into discrete binary output.
# For instance - SPECIES : Zebra will be converted into (Zebra: 1, Giraffe: 0 .. )
hasSparse = False
data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")
if hasSparse:   
    ftrList = ['SPECIES','SEX','AGE','QUALITY','VIEW_POINT','INDIVIDUAL_NAME','CONTRIBUTOR','tags'] 
else:
    ftrList = ['SPECIES','SEX','AGE','QUALITY','VIEW_POINT'] #,'tags']
    
allAttribs = CH.genAttribsHead(data,ftrList)


ftrList = ['INDIVIDUAL_NAME','CONTRIBUTOR','tags'] 
allAttribs = CH.genAttribsHead(data,ftrList)


gidAttribDict = CH.createDataFlDict(data,allAttribs,80,'Train') # binaryClf attribute in createDataFlDict will be True here


df = pd.DataFrame.from_dict(gidAttribDict).transpose()
df = df[allAttribs+["TARGET"]]
df.head(5)


infoGains = [(col,FS.infoGain(df[col],df.TARGET)) for col in df.columns]


for col in df.columns:
    infoGains.append((col,FS.infoGain(df[col],df.TARGET)))
infoGains = sorted(infoGains,key = lambda x : x[1],reverse=True)
infoGains = infoGains[2:]
infoGains

with open("../data/infoGainsExpt2.csv","w") as infGainFl:
    csvWrite = csv.writer(infGainFl)
    
    for row in infoGains:
        csvWrite.writerow(row)


len(infoGains)


import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
import pandas as pd


d = [(1,2),(2,3),(3,4)]

pd.DataFrame(d).iplot()





# ### Script for creating the m-turk files in bulk
# 
# There are 58 unique contributors and each contributor has a contiguous set of image contributions
# 
# Get contributed images per user:
# http://pachy.cs.uic.edu:5000/api/contributor/gids/?contrib_rowid_list=[1]
# 
# Get all contributor IDs
# http://pachy.cs.uic.edu:5000/api/contributor/valid_rowids/
# 

import csv
import GetPropertiesAPI as GP
import GenerateMTurkFileAPI as GM
import importlib
import random

# un-comment if there are any changes made to API
importlib.reload(GP) 
importlib.reload(GM) 


contributorImages = {}
for contributor in range(1,59):
     contributorImages[contributor] = CB.getContributorGID(contributor)

# Contributors with 0 images
contributorImages.pop(52)
contributorImages.pop(57)
contributorImages.pop(8)
contributorImages.pop(11)
contributorImages.pop(17)
contributorImages.pop(32)
contributorImages.pop(34)
contributorImages.pop(41)

contributors = list(contributorImages.keys())

selectedImgContributors = []
for i in range(100):
    selectedImgContributors.append(contributors[random.randrange(0,50)])


argToAPI = []
for index in selectedImgContributors:
    imgList = contributorImages[index]
    print(len(imgList))
    minGID = min(imgList)
    maxGID = max(imgList)
    argToAPI.append([index,minGID,maxGID])


jobImageMap= {}


for i in range(0,100):
    flName = str("photo_album_%d.question" %(i+1))
    tup = argToAPI[i]
    slctdImgs = GM.generateMTurkFile(tup[1],tup[2],str("/tmp/files/" + flName))
    jobImageMap[flName] = slctdImgs
    i += 1
    print(flName)


len(jobImageMap.keys())





import skimage.measure as sk
from skimage import io, color
import matplotlib.pyplot as plt


path = "/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/"

img1 = io.imread(path+"167.jpeg")
img2 = io.imread(path+"168.jpeg")


# STRUCTURAL SIMILARITY; higher means similar 

IMG1 = color.rgb2gray(img1)
IMG2 = color.rgb2gray(img2)
print(sk.compare_ssim(IMG1, IMG2))
 


## SQUARED ERRORS; lower means similar

# difference in colored images
print(sk.simple_metrics.compare_nrmse(img1, img2))
print(sk.simple_metrics.compare_mse(img1, img2))

# difference in gray scale images
print(sk.simple_metrics.compare_nrmse(IMG1, IMG2))
print(sk.simple_metrics.compare_mse(IMG1, IMG2))


fig = plt.figure("Images")

ax = fig.add_subplot(1,2,1)
plt.imshow(img1, cmap=plt.cm.gray)

ax = fig.add_subplot(1,2,2)
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()


import PopulationEstimatorAPI as PE, ClassiferHelperAPI as CH

regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}


train_fl = "../data/BeautyFtrVector_GZC_Expt2.csv"
test_fl = "../data/GZC_exifs_beauty_full.csv"

methObj,predResults = CH.trainTestRgrs(train_fl,
                                test_fl,
                                'linear',
                                'beauty',
                                infoGainFl="../data/infoGainsExpt2.csv",
                                methArgs = regrArgs
                                )


predResults['1'], predResults['2']


# ## Adding logic to filter out human images
# 
# *The idea is that, human images are far more easily shared than animal images, in total there are ~459 images and it could bring in a greater accuracy or in turn a lower number of images that are required for convergence.*
# 

import pandas as pd, numpy as np
import ClassifierCapsuleClass as ClfClass, ClassiferHelperAPI as CH
clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}

methodName = 'logistic'


train_data_fl = "../data/BeautyFtrVector_GZC_Expt2.csv"
train_x = pd.DataFrame.from_csv(train_data_fl)
        
train_x = train_x[(train_x['Proportion'] >= 80.0) | (train_x['Proportion'] <= 20.0)]
train_x['TARGET'] = np.where(train_x['Proportion'] >= 80.0, 1, 0)

train_y = train_x['TARGET']
train_x.drop(['Proportion','TARGET'],1,inplace=True)        
clf = CH.getLearningAlgo(methodName,clfArgs.get(methodName,None))
lObj = ClfClass.ClassifierCapsule(clf,methodName,0.0,train_x,train_y,None,None)


test_data_fl = "../data/GZC_exifs_beauty_full.csv"
testDf = pd.DataFrame.from_csv(test_data_fl)

testDataFeatures = testDf[lObj.train_x.columns]


with open("../data/HumanImagesException.csv", "r") as HImgs:
    h_img_list = HImgs.read().split("\n")

h_img_list = list(map(int, h_img_list))


len(set(testDataFeatures.index) - set(h_img_list))


count = 0
for i in h_img_list:
    if i in testDataFeatures.index:
        count += 1
print(count)


len(testDf)


testDataFeatures.index = set(testDataFeatures.index) - set(h_img_list)


obj


# ### Notebook name: ImageShareabilityClassifiers.ipynb
# #### Author: Sreejith Menon (smenon8@uic.edu)
# 
# ### General Description:
# Multiple features are extracted per image.    
# The features are majorly classified as:
# * Bilogical features like age, species, sex
# * Ecological features like yaw, view_point
# * Image EXIF/Quality data: unixtime, latitude, longitude, quality
# * Tags generated by Microsoft Image tagging API
# * Image Contributor - Sparse attribute
# * Individual animals (NID)
# 
# Based on these features mutliple classification algorithms are implemented and the metrics are evaluated. The aim of the classification algorithms is to predict given features, will a certain image be shared/not shared on a social media platform.    
# The ClassifierHelperAPI has *off-the-shelf* implementations from `sk-learn` library and uses a Classifier Object to store the metrics of each classifier.    
# The performance metrics evaluated are:
# * Accuracy - Number of correct predictions in the test data
# * Precision 
# * Recall
# * F1 score
# * Absolute Error
# * AUC
# * Squared Error - Not displayed currently
# * Zero One Hinge Loss - Not displayed currently
# 

import ClassiferHelperAPI as CH
import importlib
import numpy as np
import pandas as pd
importlib.reload(CH)
from ast import literal_eval
import plotly.plotly as py
import htmltag as HT
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
import csv
import plotly.graph_objs as go


# ### Building data for the Classifier
# * Discretizing non-binary data using the bag-of-words model
# * Building and running the classifer for all train-test splits starting from 10% upto 90%
# * Computing the performance metrics for each of the classifier.
# 

allAttribs = CH.genAllAttribs("../FinalResults/ImgShrRnkListWithTags.csv","sparse","../data/infoGains.csv")
data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")


# Block of code for building and running the classifier
# Will generate custom warnings, setting scores to 0, if there are no valid predictions
methods = ['dummy', 'bayesian', 'logistic','svm','dtree','random_forests','ada_boost']
# methods = ['ada_boost']
clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}
classifiers = []
for method in methods:
    for i in np.arange(0.4,0.5,0.1):
        clfObj = CH.buildBinClassifier(data,allAttribs,1-i,80,method,clfArgs)
        clfObj.runClf()
        classifiers.append(clfObj)


# Writing all the scores into a pandas data-frame and then into a CSV file
printableClfs = []

for clf in classifiers:
    printableClfs.append(dict(literal_eval(clf.__str__())))
    
df = pd.DataFrame(printableClfs)
df = df[['methodName','splitPercent','accScore','precision','recall','f1Score','auc','sqerr']]
df.columns = ['Classifier','Train-Test Split','Accuracy','Precision','Recall','F1 score','AUC','Squared Error']
# df.to_csv("../ClassifierResults/extrmClfMetrics_abv_mean.csv",index=False)


# Will take up valuable Plot.ly plots per day. Limited to 50 plots per day.
# changes to file name important
iFrameBlock = []
for i in np.arange(0.4,0.5,0.1):
    df1 = df[(df['Train-Test Split']==1-i)]
    df1.index = df1['Classifier']
    df1 = df1[['Accuracy','Precision','Recall','F1 score','AUC','Squared Error']].transpose()
    df1.iplot(kind='bar',filename=str('Train-Test_Split_Ratio_abv_mean %f' %i),title=str('Train-Test Split Ratio: %f' %i))
    # iFrameBlock.append(fig.embed_code)

# with open("../ClassifierResults/performanceComparisonsparse.html","w") as perf:
#     perf.write(HT.h1("Performance Comparisons of Classifiers with non_sparse Attributes."))
#     for row in iFrameBlock:
#         perf.write(HT.HTML(row))


# ### Calculating weights of features in the classifiers
# 

clfWeights = []
for clf in classifiers:
    clfAttribs = list(clf.test_x.columns)
    if clf.methodName == 'logistic':
        clfAttribWgts = list(clf.clfObj.coef_[0])
    elif clf.methodName == 'dtree' or clf.methodName == 'random_forests':
        clfAttribWgts = list(clf.clfObj.feature_importances_)
    else:
        continue
        
        
    attribWgt = {clfAttribs[i] : clfAttribWgts[i] for i in range(len(clfAttribs))}
    attribWgt['Method'] = clf.methodName
    attribWgt['Split_Percent'] = clf.splitPercent
        
    clfWeights.append(attribWgt)


clfDf = pd.DataFrame(clfWeights)


indDF = clfDf[(clfDf['Method']=='logistic')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/LogisiticWeights.csv")

indDF = clfDf[(clfDf['Method']=='dtree')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/DecisionTreeWeights.csv")

indDF = clfDf[(clfDf['Method']=='random_forests')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/RandomForestsWeights.csv")


logisticDf = clfDf[(clfDf['Method']=='logistic')]
del logisticDf['Method']
del logisticDf['Split_Percent']
dtreeDf = clfDf[(clfDf['Method']=='dtree')]
del dtreeDf['Method']
del dtreeDf['Split_Percent']
randomForestDf = clfDf[(clfDf['Method']=='random_forests')]
del randomForestDf['Method']
del randomForestDf['Split_Percent']


logisticDf = logisticDf.transpose()
logisticDf.reset_index(inplace=True)
logisticDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_logistic = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    logisticDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = logisticDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_logistic.append(df)
    
concatdf_logisitc = pd.concat([dfs_logistic[0],dfs_logistic[1],dfs_logistic[2],dfs_logistic[3],dfs_logistic[4],dfs_logistic[5],dfs_logistic[6],dfs_logistic[7],dfs_logistic[8]],axis=1)
concatdf_logisitc.to_csv("../ClassifierResults/Top15_Weights_Logisitic.csv")


dtreeDf = dtreeDf.transpose()
dtreeDf.reset_index(inplace=True)
dtreeDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_tree = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    dtreeDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = dtreeDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_tree.append(df)
    
concatdf_dtree = pd.concat([dfs_tree[0],dfs_tree[1],dfs_tree[2],dfs_tree[3],dfs_tree[4],dfs_tree[5],dfs_tree[6],dfs_tree[7],dfs_tree[8]],axis=1)
concatdf_dtree.to_csv("../ClassifierResults/Top15_Weights_Dtree.csv")


randomForestDf = randomForestDf.transpose()
randomForestDf.reset_index(inplace=True)
randomForestDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_rndf = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    randomForestDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = randomForestDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_rndf.append(df)
    
concatdf_rndf = pd.concat([dfs_rndf[0],dfs_rndf[1],dfs_rndf[2],dfs_rndf[3],dfs_rndf[4],dfs_rndf[5],dfs_rndf[6],dfs_rndf[7],dfs_rndf[8]],axis=1)
concatdf_rndf.to_csv("../ClassifierResults/Top15_Weights_Rndf.csv")


attribs = [list(dfs_logistic[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)


attribs = [list(dfs_tree[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)


attribs = [list(dfs_rndf[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)


attribs = [list(dfs_logistic[i]['Feature']) for i in range(0,9)]
attribs += [list(dfs_tree[i]['Feature']) for i in range(0,9)]
attribs += [list(dfs_rndf[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)


logisticDf.sort_values(by='10%',inplace=True,ascending=False)
fig = {
    'data' : [
        {'x' : logisticDf.Feature.head(15),'y' : logisticDf['10%'].head(15), 'mode' : 'markers', 'name' : '10%'}
    ]
}
iplot(fig)


obj1.precision


classifiers[0].preds


data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")  
methods = ['dummy','bayesian','logistic','svm','dtree','random_forests','ada_boost']
kwargsDict = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}


allAttribs = CH.genAllAttribs("../FinalResults/ImgShrRnkListWithTags.csv",'non_sparse',"../data/infoGainsExpt2.csv")
clfObj = CH.buildBinClassifier(data,allAttribs,1-0.5,80,'dtree',kwargsDict['dtree'])
clfObj.runClf()


clfObj.precision,clfObj.recall,clfObj.methodName


fpr,tpr,_ = clfObj.roccurve
rocCurve = {}
for i in range(len(fpr)):
    rocCurve[fpr[i]] = tpr[i]
    
pd.DataFrame(rocCurve,index=['tpr']).transpose().iplot()


CH.getLearningAlgo('random_forests', clfArgs)


classifiers[0].clfObj


df = df1.transpose().reset_index()
df


layout= go.Layout(
                    showlegend=True,
                    legend=dict(
                        x=1,
                        y=1,
                        font=dict(size=20)
                    ),
                    xaxis= dict(
                        title= 'Classification Quality Metrics',
                        ticklen= 5,
                        zeroline= True,
                        titlefont=dict(size=20),
                        tickfont=dict(size=20),
          # tickangle=45
                    ),
                    yaxis=dict(
                        ticklen= 5,
                        titlefont=dict(size=20),
                        tickfont=dict(size=20),
                        title="Percentage (%)"
                        #range=range
                    ),
        barmode='grouped'
                )

trace1 = go.Bar(
                    x = df1.index,
                    name = "Dummy",
                    y = df1['dummy']*100,
                    opacity = 0.5,
                    marker=dict(color='red')
                    
            )

trace2 = go.Bar(
                   x = df1.index,
                    name = "Bayesian",
                    y = df1['bayesian']*100,
                    opacity = 0.5,
                    marker=dict(color='green')
                    
            )

trace3 = go.Bar(
                   x = df1.index,
                    name = "Logistic",
                    y = df1['logistic']*100,
                    opacity = 0.5,
                    marker=dict(color='blue')
                    
            )

trace4 = go.Bar(
                   x = df1.index,
                    name = "SVM",
                    y = df1['svm']*100,
                    opacity = 1,
                    marker=dict(color='pink')
                    
            )

trace5 = go.Bar(
                   x = df1.index,
                    name = "Decision Tree",
                    y = df1['dtree']*100,
                    opacity = 1,
                    marker=dict(color='orange')
                    
            )

trace6 = go.Bar(
                   x = df1.index,
                    name = "Random Forests",
                    y = df1['random_forests']*100,
                    opacity = 0.5,
                    marker=dict(color='brown')
                    
            )

trace7 = go.Bar(
                   x = df1.index,
                    name = "Ada Boost",
                    y = df1['ada_boost']*100,
                    opacity = 1,
                    marker=dict(color='yellow')
                    
            )



data = [trace1, trace2, trace3,trace4,trace5,trace6, trace7]
fig = dict(data=data,layout=layout)
iplot(fig,filename="Expt2 Training data distributions")


df = df1.reset_index()


df1.index


df1


classifiers[0].roccurve[0]


layout= go.Layout(
                    showlegend=True,
                    legend=dict(
                        x=1,
                        y=1,
                        font=dict(size=15)
                    ),
                    xaxis= dict(
                        title= 'False Positive Rate (FPR)',
                        ticklen= 5,
                        zeroline= True,
                        titlefont=dict(size=15),
                        tickfont=dict(size=15),
          # tickangle=45
                    ),
                    yaxis=dict(
                        ticklen= 5,
                        titlefont=dict(size=15),
                        tickfont=dict(size=15),
                        title="True Positive Rate (TPR)"
                        #range=range
                    ),
        barmode='grouped'
                )

trace1 = go.Scatter(
                    x = classifiers[0].roccurve[0],
                    name = "Dummy",
                    y = classifiers[0].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='red')
                    
            )

trace2 = go.Scatter(
                    x = classifiers[1].roccurve[0],
                    name = "Bayesian",
                    y = classifiers[1].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='green')
                    
            )

trace3 = go.Scatter(
                    x = classifiers[2].roccurve[0],
                    name = "Logistic",
                    y = classifiers[2].roccurve[1],
                    opacity = 1,
                    marker=dict(color='blue')
                    
            )

trace4 = go.Scatter(
                    x = classifiers[3].roccurve[0],
                    name = "SVM",
                    y = classifiers[3].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='pink')
                    
            )

trace5 = go.Scatter(
                    x = classifiers[4].roccurve[0],
                    name = "Decision Tree",
                    y = classifiers[4].roccurve[1],
                    opacity = 1,
                    marker=dict(color='orange')
                    
            )

trace6 = go.Scatter(
                    x = classifiers[5].roccurve[0],
                    name = "Random Forests",
                    y = classifiers[5].roccurve[1],
                    opacity = 1,
                    marker=dict(color='brown')
                    
            )

trace7 = go.Scatter(
                    x = classifiers[6].roccurve[0],
                    name = "Ada Boost",
                    y = classifiers[6].roccurve[1],
                    opacity = 1,
                    marker=dict(color='yellow')
                    
            )



data = [trace1, trace2, trace3,trace4,trace5,trace6, trace7]
fig = dict(data=data,layout=layout)
iplot(fig,filename="Expt2 Training data distributions")





from plotly.offline import plot, iplot
import cufflinks as cf, plotly.plotly as py, json, pandas as pd, numpy as np
import ClassifierCapsuleClass as ClfClass, ClassiferHelperAPI as CH
import RegressionCapsuleClass as RgrClass
import plotly.graph_objs as go
from collections import Counter
cf.go_offline()
import folium
from folium.plugins import MarkerCluster
from folium import plugins
import importlib
importlib.reload(CH)


with open("../data/Flickr_EXIF_full.json" , "r") as exif_fl:
    exif_d = json.load(exif_fl)

df = pd.DataFrame.from_dict(exif_d).transpose()
df['datetime'] = pd.to_datetime(df['date'])
df['date'] = df['datetime'].dt.date

df['date'] = pd.to_datetime(df.date)
df['year_month'] = df.date.dt.strftime("%m-%Y")
df['year'] = df.date.dt.strftime("%Y")
df['month'] = df.date.dt.strftime("%m")
df['week'] = df.date.dt.week

df = df[(df['year'] > '1970')]
df.sort_values(by='date', inplace=True)


# ## Month-wise distribution over the years
# 

df['year_month'].iplot(kind='histogram')


# ## Annual Distribution of images
# 

df['year'].iplot(kind='histogram')


# ## General trend - month wise distribution
# 

df['month'].iplot(kind='histogram')


df_non_zero = df[(df['lat'] != 0) & (df['long'] != 0)][['lat', 'long']]

map_loc = folium.Map(location=[38.64264252590279, -51.622090714285676],tiles='Stamen Terrain',zoom_start=2)

recds = df_non_zero.to_records()
for i in range(0,len(recds)):
    folium.Marker([recds[i][1],recds[i][2]],
              icon=folium.Icon(color='green',icon='info-sign'), popup=recds[i][0]
              ).add_to(map_loc)
    
map_loc.save(outfile='../FinalResults/FlickrLocations.html')


locs = [(recd[1],recd[2]) for recd in recds]
heatmap_map = folium.Map(location=[38.64264252590279, -51.622090714285676],tiles='Stamen Terrain', zoom_start=2)
hm = plugins.HeatMap(locs)
heatmap_map.add_children(hm)

heatmap_map.save("../FinalResults/heatMap_Flickr.html")


# ## Weekly distribution of Flickr Images
# 

df_new = df.groupby(['year','week']).count()['date']
df_dict = df_new.to_dict()
df_tups = [(' wk#'.join(map(str,key)), df_dict[key]) for key in df_dict.keys()]
df_tups = sorted(df_tups, key=lambda x : (x[0], x[1]))
x = ["'"+tup[0][2:] for tup in df_tups]
y = [tup[1] for tup in df_tups]
trace1 = go.Bar(
            x = x,
            y = y
        )

data = [trace1]
layout = go.Layout(
    xaxis=dict(tickangle=45)
)
fig = dict(data=data, layout=layout)
py.iplot(fig)


df_train.iplot(kind='histogram',histnorm='probability')


clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}

regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}
# ['dummy', 'bayesian', 'logistic', 'svm', 'dtree', 'random_forests', 'ada_boost']:
# ['linear', 'ridge', 'lasso', 'svr', 'dtree_regressor', 'elastic_net']:
for rgrMeth in ['ada_boost']:
    train_data_fl = "/tmp/training_fl.csv"
    test_data_fl = "/tmp/training_fl.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)

    
    df_train = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])
    df_train.index = df_train.GID
    df_train.drop(['GID'],1,inplace=True)
    
    test_data_fl = "/tmp/testing_fl.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)
    df_test = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])

    df_test.index = df_test.GID
    df_test.drop(['GID'],1,inplace=True)
    
    test_data_fl = "/tmp/testing_fl_bing.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)
    df_test2 = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])

    df_test2.index = df_test2.GID
    df_test2.drop(['GID'],1,inplace=True)
    
    trace1 = go.Histogram(
        x=df_train['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Training',
        marker=dict(
            color='grey')
    )
    trace2 = go.Histogram(
        x=df_test['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Flickr',
        marker=dict(
            color='blue')
    )
    trace3 = go.Histogram(
        x=df_test2['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Bing',
        marker=dict(
            color='lightgreen')
    )

    data = [trace1, trace3, trace2]

    layout = go.Layout(
        title='PDF %s' %rgrMeth,
        xaxis=dict(
            title='Share rate'
        ),
        yaxis=dict(
            title='P(X)'
        ),
        barmode='overlay'
    )

    fig = go.Figure(data=data, layout=layout)
    f = py.iplot(fig)
    print(f.embed_code)


'''
'symmetry',
 'hsv_itten_std_v',
 'arousal',
 'contrast',
 'pleasure',
 'hsv_itten_std_h',
 'hsv_itten_std_s',
 'dominance'

'''


with open("../data/GZC_beauty_features.json", "r") as fl1:
    gzc_bty = json.load(fl1)
    
with open("../data/ggr_beauty_features.json") as fl2:
    ggr_bty = json.load(fl2)
    
with open("../data/Flickr_Beauty_features.json") as fl3:
    flickr_zbra_bty = json.load(fl3)
    
with open("../data/Flickr_Bty_Giraffe.json") as fl4:
    flickr_giraffe_bty = json.load(fl4)


def build_box_plot(ftr, names, *datasets):
    traces = []
    i = 0
    for dataset in datasets:
        ftrset = [dataset[img][ftr] for img in dataset.keys()]
        traces.append(go.Box(x=ftrset,name=names[i]))
        i += 1
    return traces


layout = go.Layout(title = "Symmetry")
data = build_box_plot('symmetry',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "contrast")
data = build_box_plot('contrast',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "hsv_itten_std_h")
data = build_box_plot('hsv_itten_std_h',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "hsv_itten_std_s")
data = build_box_plot('hsv_itten_std_s',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "hsv_itten_std_v")
data = build_box_plot('hsv_itten_std_v',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "arousal")
data = build_box_plot('arousal',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "dominance")
data = build_box_plot('dominance',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


layout = go.Layout(title = "pleasure")
data = build_box_plot('pleasure',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)





# ### Notebook Name: AppendMicrosoftAIData
# #### Author: Sreejith Menon (smenon8@uic.edu)
# 
# ### General Description:
# Microsoft Image Tagging API generates a bag of words that can be used to describe a image.    
# Think of it as, the words (nouns) you will use to describe the image to a person who cannot see the image. Each word that is returned has an associated __confidence__ associated with the prediction. Tags with __low confidence__ will be not considered(or ignored). For the purpose of experiment 2, the confidence level has hardcoded to 0.5.
# 
# This notebook has code that will take the API data which has been already parsed into a JSON file and joins it with the share proportion results from Amazon Mechanical Turk albums.     
# 
# The idea is to check if occurence of a certain word influnces the share rate in any way. 
# 

import csv
import json
import JobsMapResultsFilesToContainerObjs as ImageMap
import DeriveFinalResultSet as drs
import DataStructsHelperAPI as DS
import importlib
import pandas as pd
import htmltag as HT
from collections import OrderedDict
#import matplotlib.pyplot as plt
import plotly.plotly as py
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_online()
flName = "../data/All_Zebra_Count_Tag_Output_Results.txt"
pd.set_option('display.max_colwidth', -1)
imgAlbumDict = ImageMap.genImgAlbumDictFromMap(drs.imgJobMap)
master = ImageMap.createResultDict(1,100)
imgShareNotShareList,noResponse = ImageMap.imgShareCountsPerAlbum(imgAlbumDict,master)
importlib.reload(ImageMap)
importlib.reload(DS)


# ### Rank list of images by share rates with Microsoft Image Tagging API output
# Block of code for building rank list of images shared in the descending order of their share rates 
# Appended with Microsoft Image Tagging API results
# 
# The output is a rank list of all the images by their share rates along with the tags against every image.
# There is a capability to display the actual images as well alongside the rank-list.
# 
# Known issue - The '<' and '>' characters in the HTML tags in URL are often intepreted as is. 
# Future - make sure to add escape logic for these characters in HTML tags. There are opportunities to convert some of these code blocks into methods.
# 

header,rnkFlLst = DS.genlstTupFrmCsv("../FinalResults/rankListImages_expt2.csv")
rnkListDf = pd.DataFrame(rnkFlLst,columns=header)
rnkListDf['Proportion'] = rnkListDf['Proportion'].astype('float')
rnkListDf.sort_values(by="Proportion",ascending=False,inplace=True)

# create an overall giant csv
gidFtrs = ImageMap.genMSAIDataHighConfidenceTags("../data/GZC_data_tagged.json",0.5)
        
gidFtrsLst = DS.cnvrtDictToLstTup(gidFtrs)
df = pd.DataFrame(gidFtrsLst,columns=['GID','tags'])

shrPropsTags = pd.merge(rnkListDf,df,left_on='GID',right_on='GID')

# shrPropsTags.to_csv("../FinalResults/resultsExpt2RankList_Tags.csv",index=False)
shrPropsTags['URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + shrPropsTags['GID'] + '.jpeg" width = "350">'

shrPropsTags.sort_values(by=['Proportion','GID'],ascending=False,inplace=True)
fullFl = HT.html(HT.body(HT.HTML(shrPropsTags.to_html(bold_rows = False,index=False))))

fullFl
# outputFile = open("../FinalResults/resultsExpt2RankList_Tags.html","w")
# outputFile.write(fullFl)
# outputFile.close()


# ### Generate rank list of tags by share rate.
# 

tgsShrNoShrCount = {}
for lst in rnkFlLst:
    tgs = gidFtrs[lst[0]]
    tmpDict = {'share': int(lst[1]), 'not_share': int(lst[2]), 'total' : int(lst[3])}
    for tag in tgs:
        oldDict ={}
        oldDict =  tgsShrNoShrCount.get(tag,{'share' : 0,'not_share' : 0,'total' : 0})
        oldDict['share'] = oldDict.get('share',0) + tmpDict['share']
        oldDict['not_share'] = oldDict.get('not_share',0) + tmpDict['not_share']
        oldDict['total'] = oldDict.get('total',0) + tmpDict['total']

        tgsShrNoShrCount[tag] = oldDict


## Append data into data frames and build visualizations
tgsShrCntDf = pd.DataFrame(tgsShrNoShrCount).transpose()
tgsShrCntDf['proportion'] = tgsShrCntDf['share'] * 100 / tgsShrCntDf['total']
tgsShrCntDf.sort_values(by=['proportion','share'],ascending=False,inplace=True)
tgsShrCntDf = tgsShrCntDf[['share','not_share','total','proportion']]
tgsShrCntDf.to_csv("../FinalResults/RankListTags.csv")

fullFl = HT.html(HT.body(HT.HTML(tgsShrCntDf.to_html(bold_rows = False))))

outputFile = open("../FinalResults/RankListTags.html","w")
outputFile.write(fullFl)
outputFile.close()


iFrameBlock = []
fig = tgsShrCntDf['proportion'].iplot(kind='line',filename="All_Tags",title="Distribution of Tags")
iFrameBlock.append(fig.embed_code)
#plt.savefig("../FinalResults/RankListTags.png",bbox_inches='tight')


gidFtrs = ImageMap.genMSAIDataHighConfidenceTags("../data/GZC_data_tagged.json",0.5)
        
gidFtrsLst = DS.cnvrtDictToLstTup(gidFtrs)
df = pd.DataFrame(gidFtrsLst,columns=['GID','tags'])


df





import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
import json
import plotly.graph_objs as go
import pandas as pd
import htmltag as HT
import PopulationEstimatorAPI as PE, ClassiferHelperAPI as CH
import importlib
import MarkRecapHelper as MR
importlib.reload(PE)
import random
import DataStructsHelperAPI as DS
from plotly.offline import plot, iplot


# ### Logic for creating the comma seperated aggregate data file
# Not needed to run every time
# 

attribs = [ 'GID', 'AID', 'AGE',
       'EXEMPLAR_FLAG', 'INDIVIDUAL_NAME', 'NID', 'QUALITY', 'SEX', 'SPECIES',
       'VIEW_POINT','CONTRIBUTOR']

df = ImageMap.genGidAidFtrDf("../data/full_gid_aid_map.json","../data/full_aid_features.json",'../data/full_gid_aid_ftr.csv')
df_comb = ImageMap.createMstrFl("../data/full_gid_aid_ftr.csv","../data/GZC_data_tagged.json",attribs,"../data/full_gid_aid_ftr_agg.csv")


# ## Visuals for accuracies of predictions
# 

with open("../FinalResults/PopulationEstimate.json","r") as jsonFl:
    resObj = json.load(jsonFl)


df = pd.DataFrame(resObj)
df['Axes Name'] = df['Classifier'] + " " + df['Attribute']

df = df[['Axes Name', 'all','giraffes','zebras','shared_images_count']]
df['Error_total_pop'] = df['all'] - 3620
df['Error_zebra_pop'] = df['zebras'] - 3468
df['Error_giraffe_pop'] = df['giraffes'] - 177
df['Predicted_Shared_proportion'] = df['shared_images_count'] * 100 / 6523
dfFull = df[['Axes Name','all','Error_total_pop','zebras','Error_zebra_pop','giraffes','Error_giraffe_pop','shared_images_count','Predicted_Shared_proportion']]
dfFull['norm_error_total_pop'] = dfFull['Error_total_pop'] / 3620
dfFull['norm_error_zebra_pop'] = dfFull['Error_zebra_pop'] / 3468
dfFull['norm_error_giraffe_pop'] = dfFull['Error_giraffe_pop'] / 177
dfFull.head()


dfErrors= dfFull[['Axes Name','Error_total_pop','Error_zebra_pop','Error_giraffe_pop']]
dfErrors.index = df['Axes Name']
dfErrors.drop(['Axes Name'],1,inplace=True)


layout = go.Layout(
    title="Estimation absolute-errors using predict-shared data",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Absolute Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig1 = dfErrors.iplot(kind='bar',filename="Absolute_Errors",layout=layout)


dfNormErrors= dfFull[['Axes Name','norm_error_total_pop','norm_error_zebra_pop','norm_error_giraffe_pop']]
dfNormErrors.index = df['Axes Name']
dfNormErrors.drop(['Axes Name'],1,inplace=True)


layout = go.Layout(
    title="Estimation normalized-errors using predict-shared data",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Normalized Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig2 = dfNormErrors.iplot(kind='bar',filename="Norm_Errors",layout=layout)
# Error = (predicted population - actual population)
# Normalized error formula =  Error / actual population


dfNoOutliers = dfErrors[(abs(dfErrors['Error_total_pop']) <= 2750 )][(abs(dfErrors['Error_total_pop']) > 10)]


layout = go.Layout(
    title="Estimation errors using predict-shared data -no outliers",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Absolute Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig3 = dfNoOutliers.iplot(kind='bar',filename="errors_noOutliers",layout=layout)


# predicted shared proportion (x) vs normalized error zebra (y1) and giraffe (y2)? thanks!
dfNewPlot = dfFull[['Predicted_Shared_proportion','norm_error_zebra_pop','norm_error_giraffe_pop']]
dfNewPlot.index = dfNewPlot['Predicted_Shared_proportion']/100
dfNewPlot.drop(['Predicted_Shared_proportion'],1,inplace=True)
dfNewPlot.head()


layout = go.Layout(
    title="Predicted Shared Proportion versus Norm Error",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Predicted Share Proportion",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Normalized Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    )
    )
fig4 = dfNewPlot.iplot(kind='bar',filename="predictedSharedVsError",layout=layout)


fullFl = HT.HTML(HT.body(HT.h2("Population Estimates using predicted shared data - master table"),
                HT.HTML(dfFull.to_html(index=False)),
                HT.HTML(fig1.embed_code),
                HT.HTML(fig2.embed_code),
                HT.HTML(fig3.embed_code),
                HT.HTML(fig4.embed_code)
               ))


outputFile = open("../FinalResults/PopulationEstimationUsingClf.html","w")
outputFile.write(fullFl)
outputFile.close()


# ## Synthetic Experiments
# 
# ### Synthetic Experiment #1
# #### Assign a score to each image (here probability) and select the top 'k' images for each contributor and share them
# #### Calculate the population estimate
# 
# 
# ### Synthetic Experiment #2
# #### Assign a score to each image (here probability) and select the top 'x' images for each contributor where x is a random number and share them
# #### Calculate the population estimate
# 

appearanceDays = {}
for card in sdCards.keys():
    pred_results = {gid : predResults[gid] for gid in sdCards[card] if gid != '3644'}
    dfPredRes = pd.DataFrame(pred_results,index=['share']).transpose().reset_index()
    dfPredRes.columns = ['GID','share']
    appearanceDays[card] = set(pd.DataFrame.merge(dfPredRes,dfGidDays,on='GID').to_dict()['day'].values())


appearanceDays


import PopulationEstimatorAPI as PE
import importlib
importlib.reload(PE)


l = PE.buildErrPlots('clf')
for ifrm in l:
    print(ifrm)
    print("<p>X-axis : k <br>Y axis = Percentage Error</p>")
    print()


import pandas as pd


def buildErrPlots(clfOrRgr, thresholdMeth=False, randomShare=False):
    if clfOrRgr == 'clf':
        algTypes = ['bayesian','logistic','svm','dtree','random_forests','ada_boost']
    else:
        algTypes = ['linear','ridge','lasso','svr','dtree_regressor','elastic_net']
    attribTypes = ['sparse','non_sparse','non_zero','abv_mean', 'beauty']
    
    flNms = [str(alg + "_" + attrib) for alg in algTypes for attrib in attribTypes]

    if thresholdMeth:
        suffix = "_thresholded.csv"
        hdr = "threshold"
        if clfOrRgr == 'clf':
            titleSuffix = "classifiers thresholded"
        else:
            titleSuffix = "regressors thresholded"
    else:
        hdr = "num_images"
        if randomShare:
            suffix = "_kSharesRandom.csv"
            if clfOrRgr == 'clf':
                titleSuffix = "classifiers random choices"
            else:
                titleSuffix = "regressors random choices"
        else:
            suffix = "_kShares.csv"
            if clfOrRgr == 'clf':
                titleSuffix = "classifiers top k choices"
            else:
                titleSuffix = "regressors top k choices"

    df = pd.DataFrame.from_csv(str("../FinalResults/"+flNms[0]+suffix)).reset_index()
    df.columns = list(map(lambda x : str(x + "_" + flNms[0]) if x != hdr else x,list(df.columns)))
    for i in range(1,len(flNms)):
        df1 = pd.DataFrame.from_csv(str("../FinalResults/"+flNms[i]+suffix)).reset_index()
        df1.columns = list(map(lambda x : str(x + "_" + flNms[i]) if x != hdr else x,list(df1.columns)))
        df = pd.DataFrame.merge(df,df1,on=hdr)

    df.index = df[hdr]
    df.drop([hdr],1,inplace=True)
    

    # calculate errors in estimation
    # % error = (predicted - actual) * 100 / actual
    for col in df.columns:
        if 'all' in col:
            df[str(col+'_err')] = (df[col] - 3620) / 36.20
        elif 'zebras' in col:
            df[str(col+'_err')] = (df[col] - 3468) / 34.68
        elif 'giraffes' in col:
            df[str(col+'_err')] = (df[col] - 177) / 1.77

    figs=[]
    errorCols = [col for col in df.columns if 'err' in col]
    # df = df[errorCols]
    return df

    for alg in algTypes:
        algCol = [col for col in df.columns if alg in col]
        algDf = df[algCol]
        titleAlg = "All %s %s" %(alg,titleSuffix)
        figs.append(algDf.iplot(kind='line',title=titleAlg))

    for attrib in attribTypes:
        attribCol = [col for col in df.columns if attrib in col]
        attribDf = df[attribCol]
        titleAttrib = "All %s %s" %(attrib,titleSuffix)
        figs.append(attribDf.iplot(kind='line',title=titleAttrib))

    figCodes = [fig.embed_code for fig in figs]
    return figCodes


df = buildErrPlots('clf', randomShare=True)


df.to_csv("/tmp/test.csv")





cols = list(filter(lambda x : 'zebra' in x and 'beauty' in x, list(df.columns)))
df[cols].to_csv("/tmp/zebras_bty_rgr.csv")


import PopulationEstimatorAPI as PE
import importlib
importlib.reload(PE)


l = PE.buildErrPlots('rgr', thresholdMeth=True)


for i in l:
    print(i)
    print("<p>X-axis : k <br>Y axis = Percentage Error</p>")
    print()


train_fl, test_fl = "../data/BeautyFtrVector_GZC_Expt2.csv", "../data/GZC_exifs_beauty_full.csv"
inExifFl,inGidAidMapFl,inAidFtrFl = "../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json"
meth = 'linear'
attrib = 'beauty'
regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}

methObj,predResults = CH.trainTestRgrs(train_fl,
                                test_fl,
                                meth,
                                attrib,
                                infoGainFl="../data/infoGainsExpt2.csv",
                                methArgs = regrArgs
                                )


PE.kSharesPerContribAfterCoinFlip(predResults, inExifFl, inGidAidMapFl, inAidFtrFl, lambda : 2)


res = [{'all': 1320.0, 'giraffes': None, 'zebras': 817.0},
{'all': 2000.0, 'giraffes': 120, 'zebras': 817.0},
{'all': 2220.0, 'giraffes': None, 'zebras': None},
{'all': 3220.0, 'giraffes': 180, 'zebras': 2000},
{'all': 3220.0, 'giraffes': 180, 'zebras': 2500}]

df1 = pd.DataFrame(res)


df1.iplot(kind='line')


df = PE.runSyntheticExptsRgr(inExifFl, inGidAidMapFl, inAidFtrFl, range(2,30), thresholdMeth=False, randomShare=False, beautyFtrs = True)


df['gnd_truth_zebra'] = 3468
df['gnd_truth_girrafe'] = 177
df['gnd_truth_all'] = 3628


df.plot(kind='line')


import matplotlib.pyplot as plt


plt.show()


df.drop(['all', 'giraffes', 'gnd_truth_girrafe', 'gnd_truth_all'],1,inplace=True)


df.head()





# Notebook Name: BuildConsolidatedFeaturesFile.ipynb
# 
# Created date : Sunday, 27th March
# 
# Author : Sreejith Menon
# 
# Description : 
# buildFeatureFl(input file,output file)
# Reads from a consolidated HIT results csv file (input file). 
# 
# Extracts the below features from the IBEIS dataset:
# 1. species_texts
# 2. sex_texts
# 3. age_months_est
# 4. exemplar_flags
# 5. quality_texts
# 
# Consolidated HIT results contain number of shares and not shares per image in the mechanical turk jobs.
# Expects an input file in the following format:
# [GID,SHARE,NOT_SHARE,TOTAL]
# 

import csv
import GetPropertiesAPI as GP
import importlib
importlib.reload(GP) # un-comment if there are any changes made to API


# Logic for reading data from the consolidatedHITResults file
# 

def buildFeatureFl(inFL,outFL):    
    reader = csv.reader(open(inFL,"r"))
    head = reader.__next__()

    data = {}
    for row in reader:
        data[row[0]] = row[1:]

    # Extracts all the annotation ID's from IBEIS
    aidList = []
    for gid in data.keys():
        aid = GP.getAnnotID(int(gid))
        data[gid].append(aid)

    # Extracts all feature info based on annotation ID's from IBEIS
    for gid in data.keys():
        if data[gid][3] != None:
            aid = data[gid][3]
            spec_text = GP.getImageFeature(aid,"species_texts")
            data[gid].append(spec_text)
            sex_text = GP.getImageFeature(aid,"sex_texts")
            data[gid].append(sex_text)
            est_age = GP.getImageFeature(aid,"age_months_est")
            data[gid].append(est_age)
            exemplar = GP.getImageFeature(aid,"exemplar_flags")
            data[gid].append(exemplar)
            qual_text = GP.getImageFeature(aid,"quality_texts")
            data[gid].append(qual_text)
        else:
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')
            data[gid].append('NULL')

    # Write all the extracted info to a CSV file
    head += ['ANNOTATION_ID','SPECIES','SEX','AGE_MONTHS','EXEMPLAR_FLAG','IMAGE_QUALITY']
    writeFL = open(outFL,"w")
    writer = csv.writer(writeFL)
    writer.writerow(head)
    for row in data.keys():
        writer.writerow([row] + data[row])
    writeFL.close()


def __main__():
    buildFeatureFl("../data/consolidatedHITResults.csv","../data/consolidatedHITResultsWithInfo1.csv")
    
if __name__ == __main__:
    __main__()


GP.getAnnotID(5381)


gid_aid_map = {}
for gid in range(1,5384):
    gid_aid_map[gid] = GP.getAnnotID(gid)


import json


with open("../data/flickr_zebra_gid_aid_map.json","w") as fl:
    json.dump(gid_aid_map, fl, indent=4)


list(gid_aid_map.values())


aids = [aid for lst in list(gid_aid_map.values()) for aid in lst if len(lst)]


aid_species_map = {aids[i] : features[i] for i in range(len(aids))}


features = GP.getImageFeature(aids, 'species/text')


with open("../data/flickr_zebra_aid_species_map.json", "w") as fl:
    json.dump(aid_species_map, fl, indent = 4)


import UploadAndDetectIBEIS as UD


UD.check_job_status('jobid-5388')


data_dict = {
        'jobid': 'jobid-5388',
    }
response = UD.get('api/engine/job/status', data_dict)


response





