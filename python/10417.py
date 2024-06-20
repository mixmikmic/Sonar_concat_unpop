# # WALK	: Web Archives for Longitudinal Research
# 
# "WALK will enable present and future humanities and social science scholars to access, interpret and curate born-digital primary resources that document our recent past."
# 
# This is a demo/tutorial for the **University of Toronto** on our up-to-now explorations on Web Archives.
# 
# Over the summer, I have had the pleasure of working with Dr. Ian Milligan and Nick Ruest, building a tool for conducting analytics on the Web Archive collections. This has involved working with [Warcbase](http://lintool.github.io/warcbase-docs/) in order to extract important data from the collections and using Python including the pandas / numpy libraries via Anaconda, a framework intended for analytics-style work.
# 
# I have called this tool / library *Compare* because it uses Warcbase derivative data from the archives / WARC files.
# 
# Compare uses Multiple Correspondence Analysis (MCA) to explore relationships among different collections of data. It is like factor analysis.  I have a demo of MCA [using parliamentary committees and MPs as a sample.](https://github.com/web-archive-group/WALK/blob/master/Scripts/compare_collection/UnderstandingMCA.ipynb). 
# 

#First let's import the necessary libraries.
get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries
import matplotlib.pyplot as plt #plotting library
from mpl_toolkits.mplot3d import Axes3D #for 3d graphs
import copy #need deepcopy() for working with the dictionaries.

####   Uncomment the items below if you want 
####   to use D3 for output.

#import mpld3
#mpld3.enable_notebook()


# While many questions could be answered using MCA, we have decided to focus on a simple problem statement: ***Can we conduct an evaluation of a collection of web archives using data analytics?*** For this demo, I have included a number of items:
# 
# - Two 'dummy' collections (HUMANITIES COMPUTING & UNIVERSITY OF TORONTO T-SPACE)
# - Two 'external' collections (UofA prairie provinces & UVIC environmental organizations)
# - Three Twitter crawls.
# - University of Toronto collections (4 in all)
# 
# While there are a number of ways we could compare the collections, this example will use web domain names (urls). Often MCA tries to compare both factors (collections *and* urls) but including the urls would be too difficult. (Besides, do we really need to understand why www.youtube.com is in multiple collections?)  The urls will be represented by off-white squares and no labels to keep them from confusing the analysis.
# 
# We assume that libraries will continue not to count format as a selection criteria, meaning that web archives are selected on the same criteria as books, journals or any other source. We have decided to focus on the following principles of collection management, however:
# 
# - Coverage / Scope
#   + What are the best ways to evaluate coverage?
# - Accuracy
#   + Can we detect problems (eg. ineffective crawls)
# - Relevance
#   + Is a historian likely to find something unique or interesting in the collection?
# - Dynamics
#   + How has the collection changed from crawl to crawl?
# 

# The output below takes the derivative files from the folder "UFT/" and puts them into a python dictionary (array)
# for later use.  I have included two of these. One including the Heritage Community Foundation's collection and one
# not.

path = "UFT/"

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:10], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1]) for line in file.readlines()}))
    return(urls)

#newdict = defaultdict(dict)
newdict = defaultdict(lambda: defaultdict(list))
newdict2 = defaultdict(lambda: defaultdict(list))
PC = processCollection(path)
#print(list(zip(PC[0])))
#print(list(zip(PC[0][0])))
#print (**collect)
for collect in PC:
    for coll, date, url in collect:
        newdict[date][coll].append(url)

# newdict will provide all the data like so:

#{'DATE': {'COLLECTION': ['url1.com', 'url2.com', 'etc']}}
#

 


## Produce a dictionary output that creates a list of outputs suitable for analysis by date.
##
## collection_var[-1] would analyze all the links together until the latest year (2016). collection_var[-2]
## would analyze everything up to t-1 (2015).
##
## Our hope for the future is that the data could be used in an animation, showing changes over time. But for now, 
## we will just show the progress.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

collection_var = reduce_collections (copy.deepcopy(newdict))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.


# First we will start with the earliest date (2011). Two collections have data listed for these dates.
# 
# - Canadian Labor Unions
# - Prairie Provinces (UofA)
# 
# When there are fewer than 3 collections, you get a venn diagram showing the cross-connections. You can also extract data using V2_AB object or V3_ABC (if there are three items).
# 

UFT = Compare(collection_var[-6])
print("There are "+ str(len(UFT.V2_AB)) + " members in common. They are : " + str(UFT.V2_AB))


# In 2012, both collections grew, but the Alberta provinces grew more. Perhaps some content analysis or a Q-sort could be done to analyse what changed?

UFT = Compare(collection_var[-5])
print("There are "+ str(len(UFT.V2_AB)) + " members in common. They are : " + str(UFT.V2_AB))


# In 2013, University of Alberta's humanities collection started.  Now we have three collections and can switch to a size-three venn digram.
# 
# 

UFT = Compare(collection_var[-4])
print("There are "+ str(len(UFT.V3_ABC)) + " members in common. They are : " + str(UFT.V3_ABC))


# In 2014, we have four collections and move to correspondence analysis.  (Scroll down for more descriptions, as 2015 is much more interesting).  So far the main item of interest is that the Prairie Provinces and Canadian Labor Unions collections have more in common with each other than with the other two collections.
# 
# To analyse, I suggest a three pronged approach.
# 
# - HA: What does the horizontal axis mean?
# - VA: What does the vertical axis mean?
# - QUA: Can we name the quadrants?
# - CLUS: What sources have clustered together
# 
# HA-VA-QUA-CLUS!

compare1 = Compare(collection_var[-3]) #2014


# In 2015, most of the new collections have been introduced. The factor differences are quite low, suggesting that the differences in terms of link sharing are not very pronounced.  Consider that most websites in this era will include the usual adobe, youtube, twitter, facebook and other social media links. While this information may be obvious to us now, a historian 50 years from now may be able to use this to confirm the importance (or the extent of the importance) of sites like YouTube on web collections. 
# 
# HA: Maybe global (snowden) to local (Toronto pan-am-games)?  Also the dummy collections on are the left.
# VA: Maybe humanities to social sciences?
# QUA: 
# 
# |               |               |
# | :------------ |:-------------:|
# | T-SPACE       | POLITICS      |
# | INDIV. RIGHTS | ORGANIZATIONS |
# 
# CLUS: Definitely a political cluster, but the others are not very explanatory yet.

Compare(collection_var[-2]) #2015


# Now that some Twitter collections have been included, the analysis is a bit more meaningful.  The way to look at this graph is to examine the horizontal axis, which accounts for 30% of the difference among the collections.  There is a clear deliniation between the Twitter collections (+ TSpace) and the web archives.  On the vertical axis (17% of the explanation) there is a shift between the Fort McMurray Fires and Tspace.
# 
# HA: Twitter to archives.
# VA: Local (bottom) to global (top)
# QUA: Toronto collections
# 

Compare(collection_var[-1]) #2016


TestDict1 = {'2009': {'c1': {'lk1', 'lk2', 'lk3'},
                     'c2': {'lk1', 'lk10', 'lk20', 'lk2'},
                     'c3': {'lk3', 'lk10', 'lk33', 'lk4'}},
            '2010': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c3': {'lk10', 'lk9', 'lk7'}},
            '2011': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c4': {'lk1', 'lk2', 'lk3'}},
            '2012': {'c1': {'lk1', 'lk99', 'lk6'}}
           }

#print(list(zip(*zip(TestDict['2009'])))


# # Using Simple Correspondence Analysis to evaluate collections of Web Archives
# 
# Evaluating any knowledge collection requires some attention to a variety of principles.  These can include:
# - *Coverage* : Does the collection provide sufficient scope to be useful to historians of the future?
# - *Relevance* : Will the collection be meaningful for historians answering important questions about the current era?
# - *Accuracy* : Is the collection providing an accurate account of the current era?
# 
# Relevance and accuracy requires some interpretation by the curators of the collection. Much of the evaluation will depend on why the web sites are being archived, and for whom. While correspondence analysis can help with interpretive approaches, **coverage** will be the focus of this tutorial. Thus we will start with a simple research question:  * How diverse are the collections being stored by members of the WALK project? *
# 
# 

get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries will eventually be moved to the Compare class.
import matplotlib.pyplot as plt #plotting library
import copy #need deepcopy() for working with the dictionaries.
import pandas as pd
from adjustText import adjustText as AT


# The methods we will use will follow this approach:
# 
# - Identify outliers and explain why they are different.
# - Hypothesize based on dimensions
# - Hypothesize based on clusters
# - Choose greatest outliers and produce Venn Diagrams to show common areas.
# - Include a dummy collections to explain variance.
# 
# For the most part, we will use comparisons based on the urls mentioned in the collection.  However, we will also include another example using word counts.
# 

path = "ALL/"

##  INCLUDE ANY DICTIONARIES YOU WISH TO EXCLUDE in the following list.  The excluded libraries will be removed from
##  newdict2 for comparison purposes:

exclude1 = ['DAL_Mikmaq', 'WINNIPEG_truth_', 'SFU_BC_Local_go', 'SFU_NGOS', 'SFU_PKP',  'DAL_NS_Municipl', 'DAL_HRM_Docs', 
            'WINNIPEG_oral_h',  'WINNIPEG_websit', 'WINNIPEG_digiti'] 

dummies = ['DUMMY_OVERALL', 'DUMMY_MEDIA', 'DUMMY_GOVERNMEN', 'DUMMY_ORGANIZAT',
           'DUMMY_TECHNOLOG', 'DUMMY_SOCIALMED']

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

newdict = defaultdict(lambda: defaultdict(list))

PC = processCollection(path)
for collect in PC:
    for coll, date, url in collect:
        if coll in dummies:
            pass
        else:
            newdict[date][coll].append(url)
            
## add_two_collections merges two dictionaries and is used by reduce_collections to show the accumulation of 
## the collections over dates.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

## reduce_collections takes the newdict dictionaries in {date: collection : [list of urls]} form
## and returns a list of the dictionaries as they accumulated by date. [2009 : collection [list of urls], 
## 2010+2009 : collection etc.]
def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

collection_var = reduce_collections(copy.deepcopy(newdict))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.


# ### Identify Outliers 
# First we can run a Compare script to see if there are any outliers. Since collection_var is a list of accumulated collections from oldest to most recent, -1 will include all of the available collections for all years.
# 

dummy = Compare(collection_var[-1])


# We have found our outliers!  Two collections: the Heritage Alberta online encyclopedia and the Heritage Community Foundation web archive collection have more in common with each other than they do to the rest of the collections in the list. We can ask why later on, but for now, let's eliminate them from the collection and see if we can find out more about the collections.
# 
# ### Create hypotheses based on dimensions
# 
# The purpose of exploratory analysis is to create hypotheses that can later be tested.  The first way we can do this is by examining the axes, each of which represents a dimension for which the collections have been differentiated. Unfortunately, in the first analysis, the outliers were so great (a Chi-squared distance of about) it's too difficult to see what is happening with the other collections.  So we can remove them.
# 

# Create a new defaultdict and a list of collections to exclude

exclude = ['ALBERTA_heritag', 'ALBERTA_hcf_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var2 = reduce_collections (copy.deepcopy(newdict2))
dummy2 = Compare(collection_var2[-1])


d2points = dummy2.result['rows']
d2urls = dummy2.result['columns']
d2rlabels = dummy2.response.index
d2labels = dummy2.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*d2points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*d2urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
texts=[]
for x, y, label in zip(*d2points, d2labels):
    texts.append(plt.text(x, y, label))

AT.adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()


# As it turns out, we have the Twitter collections created by the WAHR program on the left and most of the others on the right.  Alberta's Idle No More collection is also among the Twitter groups.  This implies that the main differences may have to do with what people link to on Twitter versus on other regular sites. Later, we can apply some dummy collections with the main social media and perhaps media sites to see if this is true.  Nonetheless, we can suggest that the 33.2% differentiation calculated on the horizontal axis may have something to do with the differences between the Twitter collections and other collections.
# 
# On the vertical scale there is a clear delineation between collections that are focussed on Alberta versus collections of a more general nature (The exln42 and ymmfire Twitter collections are both events that focussed on Alberta). 
# 
# ### Generate Hypotheses based on Quadrants
# 
# The quadrants are fairly clear in this example.  Given the differentiation between the local Alberta collections and more general collections plus the separation between Twitter collections and non-Twitter collections, it's pretty easy to develop a description of the quadrants.
# 
# |           |            |
# | :-------- | :--------- |
# |Twitter General | Web General |
# |Twitter Alberta | Web Alberta |
# 
# However, on the top left the UfT T-Space is not a Twitter collection and unlike the Idle No More collection (which was a social movement characterized as having a large Twitter presence), it is not clear why it falls in the upper left quadrant.
# 
# A few other observations here:
# 
# - There is no observable differentiation between UfT and UVic collections currently.  This could change if some collections were removed from the analysis (eg. the Twitter collections)
# - The Alberta in the upper quadrants are of a general nature (Humanities and Web Archives)
# - We also have new outliers, albeit with a much smaller differentiation than the Heritage Community collections.  The UVIC Calendar and the Alberta Education sites are important, for example.
# 
# ### Generate Hypotheses based on Clusters
# 
# It seems like we have three major clusters.  They do not necessarily correspond with the quadrants.  First there is a large cluster in the upper right quadrant. Then there is a medium cluster in the lower right area.  And finally a small cluster consisting of three collections:  Alberta Floods, Alberta Energy and Alberta University.
# 
# Since it is easiest, let's focus on the smallest cluster.
# 

Cluster = Compare([collection_var2[-1]['ALBERTA_floods_'], collection_var2[-1]['ALBERTA_energy_'], 
                  collection_var2[-1]['ALBERTA_univers']], names=["floods", "energy", "university"])


# Because there are only three collections used in the analysis, Compare goes to a Venn diagram to show the cross-correlations. While the quantity of common links is interesting, it is also important to look at why each item has the correlations.
# 

print("These sites were common to all three collections." + str(Cluster.V3_ABC) + "\n")
print("These sites were common to floods and energy only" + str(Cluster.V3_AB) + "\n")
print("These sites were common to energy and university" + str(Cluster.V3_AC) + "\n")
print("These sites were common to energy and floods" + str(Cluster.V3_BC) + "\n")


# The links common to all three collections include social media sites like You Tube, common technology sites like Adobe, Alberta media sites like the Edmonton Journal.  Between energy and floods are a primarily public policy and government related sites. Between energy and university are mixture of national and international media sites and a number of university departments.  Between energy and floods are a number of Google videos. 
# 
# Unfortunately, because Google Video is no longer active, it is unclear that we can find out what these videos contained. However, it is possible to go to the collections and find the videos.
# 
# ### Choose greatest outliers and produce Venn Diagrams to show common areas.
# 
# Let's do the same thing with the Heritage collections and see if we can explain why they are such an outlier.
# 

Cluster2 = Compare([collection_var[-1]['ALBERTA_heritag'], collection_var[-1]['ALBERTA_hcf_onl'], collection_var[-1]['ALBERTA_prairie']], names=["HERITAGE", "HCF_ENCYCL", "PRAIRIES"])


print("These sites were common to all three collections." + str(Cluster2.V3_ABC) + "\n")
print("These sites were common to heritage and hfc encyclopedia only" + str(Cluster2.V3_AB) + "\n")
print("These sites were common to heritage and prairies" + str(Cluster2.V3_AC) + "\n")
print("These sites were common to hcf encyclopedia and prairies" + str(Cluster2.V3_BC) + "\n")


# It appears like the main reason the Heritage collections are not really outliers in the traditional sense. Instead, they just simply have more in common with each other than with everything else.
# 

# ### Include a dummy collections to explain variance.
# 
# The last technique we can use is to apply a dummy collection to the analysis.  We created an "OVERALL" dummies by finding approximately the 200 most commonly matched web urls. Then, using content analysis, we categorized the urls by one of five types:  media (eg. The Globe and Mail), social media (Twitter), technology (Apple.com), government (gc.ca) or other organizations (eg. University of Toronto).  Let's start with what collections have the most commonly linked urls, using the "OVERALL" dummy collection.  
# 

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies[1:6]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])


# It's hard to see, but "DUMMY OVERALL" appears in the upper left quadrant.  Let's try social media.
# 

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies[0]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])


d3points = dummy3.result['rows']
d3urls = dummy3.result['columns']
d3rlabels = dummy3.response.index
d3labels = dummy3.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*d3points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*d3urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
texts=[]
for x, y, label in zip(*d3points, d3labels):
    texts.append(plt.text(x, y, label))

AT.adjust_text(texts, arrowprops=dict(arrowstyle="-", color='r', lw=0.5))

plt.show()


# The social media dummy site stayed close to the Overall dummy, suggesting that it is not a strong decider among the sites. Perhaps Government is more informative.  
# 

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in [x for i,x in enumerate(dummies) if i!=2]: #Government is the 3rd in the dummies list.
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])


# The government websites appear closer to the Alberta sites, suggesting that they play a role in differentiating between the Alberta-specific and more general sites.
# 

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in [x for i,x in enumerate(dummies) if i!=1]: #Government is the 3rd in the dummies list.
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])


# Using the media dummy, we can see that media sites partly explain the difference between the Twitter collections and the non-Twitter collections.
# 
# While this analysis focussed primarily on urls as a source of identifying differences, similar analyses could be done using other topics.  For instance, text-based analyses are also possible.
# 

newpath = "frequencies/"

textdict = dict()

def processText (path):
    #initialise vars:
    text = []
    names = []
    lines= []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            lines = [line.strip().split(" ")[1] if len(line.strip().split(" ")[1]) > 4 else "none" for line in file.readlines()[0:250] if len(line.strip().split(" ")) == 2]
        text.append(lines)
        names.append(filename[0:25])
                
            #text.append(list({(filename[0:25], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()[0:5]}))
    return([text, names])

TC = processText(newpath)
#print(TC)

CP = Compare(TC[0], names=TC[1], LABEL_BOTH_FACTORS=True)



points = CP.result['rows']
urls = CP.result['columns']
rlabels = CP.response.index
labels = CP.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*urls,  s=120, marker='s', c='b', alpha=.5, linewidths=0)
texts=[]
rtexts=[]
for x, y, label in zip(*points, labels):
    texts.append(plt.text(x, y, label, color='b'))

for rx, ry, rlabel in zip(*urls, rlabels):
    rtexts.append(plt.text(rx, ry, rlabel))
AT.adjust_text(rtexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()


newpath = "frequencies/"

textdict = dict()

def processText (path):
    #initialise vars:
    text = []
    names = []
    lines= []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            lines = [line.strip().split(" ")[1] if len(line.strip().split(" ")[1]) > 4 else "none" for line in file.readlines()[0:250] if len(line.strip().split(" ")) == 2]
        text.append(lines)
        names.append(filename[0:25])
                
            #text.append(list({(filename[0:25], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()[0:5]}))
    return([text, names])

TC = processText(newpath)
#print(TC)

CP = Compare(TC[0], names=TC[1], LABEL_BOTH_FACTORS=True)

points = CP.result['rows']
urls = CP.result['columns']
rlabels = CP.response.index
labels = CP.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*urls,  s=120, marker='s', c='b', alpha=.5, linewidths=0)
texts=[]
rtexts=[]
for x, y, label in zip(*points, labels):
    texts.append(plt.text(x, y, label, color='b'))

for rx, ry, rlabel in zip(*urls, rlabels):
    rtexts.append(plt.text(rx, ry, rlabel))
AT.adjust_text(rtexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()


# ## Conclusions:
# 
# The University of Alberta Web Archive collections are the most exhaustive.  For example, Alberta collections occur in all four quadrants. That these collections are the oldest among the three libraries studied may explain part of this difference.
# 
# The Toronto and Victoria collections appear to be more general in nature, although further research could determine whether this is the case. Some assessment about whether these collections are achieving organizational objectives could be considered. 
# 
# Collecting and archiving Twitter collections imply a preference for news and event-based information. While regional collections, such as energy and education, will have more links related to government and policy web sites. 
# 
# Conducting the analysis for the most commonly used words provides a similar result to the urls for the limited collections provided.  More research is required for further analysis.
# 

# Create a new defaultdict and a list of collections to exclude

exclude = ['ALBERTA_heritag', 'ALBERTA_hcf_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in exclude1 or coll in dummies[0]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var2 = reduce_collections (copy.deepcopy(newdict2))
dummy2 = Compare(collection_var2[-1])





# # WALK	: Web Archives for Longitudinal Research
# 
# "WALK will enable present and future humanities and social science scholars to access, interpret and curate born-digital primary resources that document our recent past."
# 
# This is a demo/tutorial for **Simon Fraser University** on our up-to-now explorations on Web Archives.
# 
# I have called this tool / library *Compare* because it uses Warcbase derivative data from the archives / WARC files.
# 
# Compare uses either Simple (CA) or Multiple Correspondence Analysis (MCA) to explore relationships among different collections of data. It is like factor analysis.  I have a demo of Simple CA [using parliamentary committees and MPs as a sample.](https://github.com/web-archive-group/WALK/blob/master/Scripts/compare_collection/UnderstandingMCA.ipynb). 
# 

#First let's import the necessary libraries.
get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries
import matplotlib.pyplot as plt #plotting library
from mpl_toolkits.mplot3d import Axes3D #for 3d graphs
import copy #need deepcopy() for working with the dictionaries.

####   Uncomment the items below if you want 
####   to use D3 for output.

#import mpld3
#mpld3.enable_notebook()


# While many questions could be answered using MCA, we have decided to focus on a simple problem statement: ***Can we conduct an evaluation of a collection of web archives using data analytics?*** For this demo, I have included a number of collections:
# 
# - Three Twitter crawls.
# - SFU collections (6 in all)
# 
# We have also included some dummy collections based on the most popular websites, categorized by media, social media, government, tech industry and NGOs.
# 
# While there are a number of ways we could compare the collections, this example will use web domain names (urls). Often MCA tries to compare both factors (collections *and* urls) but including the urls would be too difficult. (Besides, do we really need to understand why www.youtube.com is in multiple collections?)  The urls will be represented by off-white squares and no labels to keep them from confusing the analysis.
# 
# We assume that libraries will continue not to count format as a selection criteria, meaning that web archives are selected on the same criteria as books, journals or any other source. We have decided to focus on the following principles of collection management, however:
# 
# - Coverage / Scope
#   + What are the best ways to evaluate coverage?
# - Accuracy
#   + Can we detect problems (eg. ineffective crawls)
# - Relevance
#   + Is a historian likely to find something unique or interesting in the collection?
# - Dynamics
#   + How has the collection changed from crawl to crawl?
# 

# The output below takes the derivative files from the folder "UFT/" and puts them into a python dictionary (array)
# for later use.  I have included two of these. One including the Heritage Community Foundation's collection and one
# not.

path = "SFU/"

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

#newdict = defaultdict(dict)
newdict = defaultdict(lambda: defaultdict(list))
newdict2 = defaultdict(lambda: defaultdict(list))
PC = processCollection(path)
#print(list(zip(PC[0])))
#print(list(zip(PC[0][0])))
#print (**collect)
for collect in PC:
    for coll, date, url in collect:
        newdict[date][coll].append(url)

# newdict will provide all the data like so:

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll == 'ALBERTA_heritag' or coll=='DUMMY_GOVERNMEN' or coll == 'ALBERTA_hcf_onl' or coll=="DUMMY_MEDIA" or coll=="DUMMY" or coll== "ORGDUMMY" or coll== "TECHDUMMY" or coll== "SOCIALMEDIADUMM" or coll=="GOVDUMMY":
            pass
        else:
            newdict2[date][coll].append(url)

#{'DATE': {'COLLECTION': ['url1.com', 'url2.com', 'etc']}}
#

 


## Produce a dictionary output that creates a list of outputs suitable for analysis by date.
##
## collection_var[-1] would analyze all the links together until the latest year (2016). collection_var[-2]
## would analyze everything up to t-1 (2015).
##
## Our hope for the future is that the data could be used in an animation, showing changes over time. But for now, 
## we will just show the progress.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

collection_var = reduce_collections (copy.deepcopy(newdict2))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.


# First we will start with the earliest date (2013). Two collections have data listed for these dates.
# 
# When there are fewer than 3 collections, you get a venn diagram showing the cross-connections. You can also extract data using V2_AB object or V3_ABC (if there are three items). 
# 

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["down", "depressed", "nostalgic"]}, LABEL_BOTH_FACTORS=True)


# By 2015, we have four collections and move to correspondence analysis.  
# 
# To analyse, I suggest a three pronged approach.
# 
# - HA: What does the horizontal axis mean?
# - VA: What does the vertical axis mean?
# - QUA: Can we name the quadrants?
# - CLUS: What sources have clustered together
# 
# HA-VA-QUA-CLUS!
# 
# Below, try changing -4 to -3 to -2 to -1 to move forward in time from 2013.

SFU = Compare(collection_var[-1])


# According to this graph, a strong distinction exists for the VI News collection. Adding a dummy collection puts VI NEWS and trans_web on the same vertical axis but both collections have few connections to the MEDIA DUMMY.  This probably means that the "news" sites do not link frequently to the more popular news media sites.  
# 

#collection_var[-3] = removekey(collection_var[-3], 'SOCIALMEDIA')
collection_var[-1]['MEDIADUMMY'] = newdict['2015']['DUMMY_MEDIA']

SFU = Compare(collection_var[-1])


SFU = Compare(collection_var[-1])


# Moving to 2016, we now see the WAHR Twitter collections included, which greatly expand the difference among the 
# collections overall. So the collections are distinct from Twitter collections.
# 

SFU = Compare(collection_var[-1])


# Removing the Twitter examples, again we see that the collections are not very distinct in terms of url exchange.
# 

#collection_var[-1] = removekey(collection_var[-1], 'WAHR_ymmfire-ur')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_panamapape')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_exln42-all')
#collection_var[-1] = removekey(collection_var[-1], 'MEDIADUMMY')
Compare(collection_var[-1])


# *Conclusion*: The web archive collections are quite similar in terms of url content. More research is necessary to determine why, but analysis of collection policies regarding web archives may be beneficial to decide whether the current approach is meeting organizational needs. The lack of distinction among the collections is an area of curiosity, due to a lack of internal diversity among the sources.
# 

TestDict1 = {'2009': {'c1': {'lk1', 'lk2', 'lk3'},
                     'c2': {'lk1', 'lk10', 'lk20', 'lk2'},
                     'c3': {'lk3', 'lk10', 'lk33', 'lk4'}},
            '2010': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c3': {'lk10', 'lk9', 'lk7'}},
            '2011': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c4': {'lk1', 'lk2', 'lk3'}},
            '2012': {'c1': {'lk1', 'lk99', 'lk6'}}
           }

#print(list(zip(*zip(TestDict['2009'])))


# ## Using Canadian Parliamentary Committees to Understand Correspondence Analysis
# 
# Large datasets can be confusing, especially when you have very little idea about what's actually inside the data.
# 
# Correspondence analysis (CA) and Multiple Correspondence Analysis (MCA) are ways to visualize data in two dimensional space.  The Compare class we have in development uses [MCA](https://pypi.python.org/pypi/mca/1.0) to quickly turn datasets into data plots.  
# 
# In this case, the dataset uses [parliamentary committees](http://www.parl.gc.ca/Committees/en/List) to show CA in action. When you use CA to describe the membership of various committees, we call this Bivariate or Affiliation Networks. Bi-variate means you have two difference groups you are studying. In this case, we are observing 1. parliamentary committees and 2. The MPs who belong to them.
# 

#First let's import the necessary library.
get_ipython().magic('matplotlib inline')
from Compare import Compare
from json import dumps

# Let's test the class with the new dictionary.  When you have a dictionary with two keys, 
# the Compare class will
# generate a simple venn diagram.

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["morose", "depressed", "nostalgic"]},
            LABEL_BOTH_FACTORS=True)

print (x.LABEL_BOTH_FACTORS)


# ### Collecting the Data
# 
# This time I collected the data manually.  Eventually, perhaps this information will be available as an [Open Data source](http://docs.ckan.org/en/latest/api/).  Perhaps it already is.  Throw me a tweet [@ryandeschamps](http://www.twitter.com/ryandeschamps) if you think you know sometime more about this. I am always interested in Open Data.
# 
# So the following is a dictionary containing a number of Committees, followed by a list of tuples containing the name of a member/MP and the party they belong to.  You will notice some members belong to more than one committee (for instance A Vandenbeld belongs to both the Pay Equity (ESPE) and Status of Women (FEWO) committees. All the committees have more than one person (obviously). However, since we are most interested in how the committees are related, our focus will be on the MPs who belong to more than one.
# 

# Here are Committees of the current (2015-2016) sitting parliament.
# 
# There are two sets of variables.  1. MPs and 2. Committees that the MPs belong to.
# Or worded in another way, we are examining how MPs are affiliated with different groups.


Committees = dict({"ESPE": #Pay equity
                   [("Lib", "A Vandenbeld"), ("Con", "S Stubbs"), ("NDP", "S Benson"), ("Con", "D Albas"), 
                    ("Lib", "M DeCourcey"), ("Lib", "J Dzerowicz"), ("Con", "M Gladu"), ("Lib", "T Sheehan"), 
                    ("Lib", "S Sidhu")],
                   
                   "FEWO": #Status of Women
                   [("Con", "M Gladu"), ("Lib", "P Damoff"), ("NDP", "S Malcomson"), ("Lib", "S Fraser"), 
                    ("Con", "R Harder"), ("Lib", "K Ludwig"), ("Lib", "E Nassif"), ("Lib", "R Sahota"), 
                    ("Lib", "A Vandenbeld"), ("Con", "K Vecchio")],
                   
                   "HESA": #Health
                   [("Lib", "B Casey"), ("Con", "L Webber"), ("NDP", "D Davies"), ("Lib", "R Ayoub"), 
                    ("Con", "C Carrie"), ("Lib", "D Eyolfson"), ("Con", "R Harder"), ("Lib", "DS Kang"), 
                    ("Lib", "J Oliver"), ("Lib", "S Sidhu")],
                  
                   "BILI": #Library of Parliament
                   [("Con", "G Brown"), ("Con", "K Diotte"), ("Con", "T Doherty"), ("Lib", "A Iacono"),
                   ("Con", "M Lake"), ("Lib", "M Levitt"), ("Lib", "E Nassif"), ("NDP", "AMT Quach"), 
                   ("Lib", "D Rusnak"), ("Lib", "M Serré"), ("Lib", "G Sikand"), ("Lib", "S Simms")],
                   
                   "RNNR": #Natural Resources
                   [("Lib", "J Maloney"), ("Con", "J Barlow"), ("NDP", "R Cannings"), ("Con", "C Bergen"),
                   ("Lib", "TJ Harvey"), ("Lib", "D Lemieux"), ("Lib", "MV Mcleod"), ("Lib", "M Serré"), 
                   ("Con", "S Stubbs"), ("Lib", "G Tan")],
                   
                   "ACVA": #Veteran's Affairs
                   [("Lib", "NR Ellis"), ("Con", "R Kitchen"), ("NDP", "I Mathyssen"), ("Lib", "B Bratina"),
                   ("Con", "A Clarke"), ("Lib", "D Eyolfson"), ("Lib", "C Fraser"), ("Lib", "A Lockhart"), 
                   ("Lib", "S Romando"), ("Con", "C Wagantall")],
                   
                   "JUST": #Justice and Human Rights
                   [("Lib", "A Housefather"), ("Con", "T Falk"), ("NDP", "M Rankin"), ("Lib", "C Bittle"), 
                    ("Con", "M Cooper"), ("Lib", "C Fraser"), ("Lib", "A Hussen"), ("Lib", "I Khalid"), 
                    ("Lib", "R McKinnon"), ("Con", "R Nicholson")],
                   
                   "TRAN": #Transport, Infrastructure and Communities
                   [("Lib", "JA Sgro"), ("Con", "L Berthold"), ("NDP", "L Duncan"), ("Lib", "V Badawey"), 
                    ("Con", "K Block"), ("Lib", "S Fraser"), ("Lib", "K Hardie"), ("Lib", "A Iacono"), 
                    ("Lib", "G Sikand"), ("Con", "DL Watts")],
                   
                   "AGRI": #Agriculture and Agri-food
                   [("Lib", "P Finnigan"), ("Con", "B Shipley"), ("NDP", "RE Brosseau"), ("Con", "D Anderson"),
                   ("Lib", "P Breton"), ("Lib", "F Drouin"), ("Con", "J Gourde"), ("Lib", "A Lockhart"), 
                    ("Lib", "L Longfield"), ("Lib", "J Peschisolido")],
                   
                   "FOPO": #Fisheries and Oceans
                   [("Lib", "S Simms"), ("Con", "R Sopuck"), ("NDP", "F Donnelly"), ("Con", "M Arnold"), 
                    ("Con", "T Doherty"), ("Con", "P Finnigan"), ("Lib", "K Hardie"), ("Lib", "B Jordan"),
                   ("Lib", "K McDonald"), ("Lib", "RJ Morrissey")]
                  })


# Let's just start with a case of two committees and focus just on the people (not their parties)
EquityStatus = {"ESPE": [y for x, y in Committees["ESPE"]], "FEWO": [y for x, y in Committees["FEWO"]]}
equityStatus = Compare(EquityStatus)

print("There are "+ str(len(equityStatus.V2_AB)) + " members in common. They are : " + str(equityStatus.V2_AB))


# Now for the Correspondence Analysis:
#  
# 
# First, we can examine the factors separately.  Along the horizontal axis, the committees appear in a spectrum from Transport on the left to Justice on the right.
# 
# Along the vertical axis, a spectrum runs from Fisheries and Oceans to Pay Equity.
# We could also name the quadrants.  They aren't perfect (Natural Resources is connected to Status of Women?) but they 
# offer a rough outline of how the committees connect.
# 
# |               |               |
# | :------------ |:-------------:|
# | INDUSTRY      | COMPENSATION  |
# | EQUAL RIGHTS  | HEALTH        |
# 
# Three clusters also appear.  One with Fisheries and Oceans, Transport and Library of Parliament, another with Justice and Veteran's Affairs (with Agriculture and Agri-food not far away) and the largest one with Health, Status of Women and Natural Resources.
# 
# 

# Now let's see what ca does:
LABEL_FOR_TWO = False
Committee = dict()

for com, members in Committees.items():
    Committee[com] = [y for x, y in members]
committees_ca = Compare(Committee, LABEL_BOTH_FACTORS=True)






# # COMPARING THE WALK COLLECTIONS
# 
# **_WALK_** will make web archives usable for historical research now and in the future. 
# 
# Web archives store websites by periodically crawling websites on different dates. Your blog probably has a lot of new information today than it did five years ago.  A web archive would store your blog with today's data (with a crawl date 2016) and in 2011.   
# 
# But what if we had a collection of all history blogs?  That's a lot of information! And
# files that would store that information would be huge. 
# 
# Imagine being a graduate-level historian in 2050, writing a paper about  amateur historians during the period of 2011-2016. You ask for the collection of history blogs and you get a web archives file (also called a **.WARC** file) that's 500 Gigabytes or more. 
# 
# You might want to give up already! Fortunately, [Warcbase](http://lintool.github.io/warcbase-docs/) will pull useful information like links, text and images from the files.  Maybe you could use links from the websites to do a network analysis. 
# 
# Derivative files like these run in the 100s of MBs. Using Warcbase, you can also do Name-entity Recognition (NER) on the text and get a summary of the people, locations and subjects mentioned in the file.
# 
# Let us go with a research question. Some of the Social media are full of filter bubbles.  Maybe our historian bloggers are biased towards the same subjects. What if we could compare what those historians say and link to 
# against other related bloggers - say archeologists, economists, political scientists, sociologists and anthropologists? 
# 
# Then we could have a more complete view of what this world is all about. But how?
# 
# This Compare class is a first crack at solving this problem using (soon to be multiple!) correspondence analysis and
# venn diagrams. Instead of bloggers, we are looking at some web archive collections collected by the University of Alberta
# and the University of Toronto.
#  

"""
First we get our script ready with all the libraries we need. If you know Python, you are probably familiar with
most of these files.
"""

### For these import to work, you should have a copy of Anaconda available to you.  We are using Python 3 in this case.
### Also, you will need to install mca and matplotlib_venn.  See the README file for more information.

get_ipython().magic('matplotlib inline')
import os
import csv
import pandas as pd
import numpy as np
from matplotlib_venn import venn2, venn3
import mca
import matplotlib.pyplot as plt
from collections import defaultdict
#import mpld3
#mpld3.enable_notebook()


###################################
## The main class
###################################

class Compare:
    """ 
    Compare -- plot collections for comparison purposes.
    
    Description:
        The purpose of this class is to take a series of collections and plot them so to show how they match.
        If the series is a dictionary, the keys will be used to create plot names.
        If the series contains two or three collections, then the plot will show venn diagrams and return a venn object
        that can be used for other purposes.
        If the series is greater than three, the plot will show the collections in a scatter plot based on correspondence
        scores.
    
    Parameters: 
        @collections (required):  A list of lists or a dict() of size 2 or greater for comparison purposes. 
        @names:  An optional list of names for the collections.  Must be equal in size to collections. If collections is 
            a dict, this parameter will be overwritten.
        @index:  A list of index keys to create a sublist for the collections.
        @var: An optional list for further categorization of the collection data (not fully implemented yet).
        @REMOVE_SINGLES: (Default:True) For 4 collections or more, remove from the analysis any data points that are
            members of only one collection. This reduces the chance that a disproportionately large collection
            will be seen as an outlier merely because it is disproportionately large.
        @DIM3: either 2 or 3 - # of dimensions to visualize (2D or 3D)
        LABEL_BOTH_FACTORS: whether to use two factors in results (default to just one).
            
    """
    
    def __init__ (self, collections, names=[], index=[], var=[], REMOVE_SINGLES=True, DIM3=False, LABEL_BOTH_FACTORS=False):
        self.collection_names = names
        self.index = index
        self.collections = collections
        self.REMOVE_SINGLES = REMOVE_SINGLES
        if DIM3 == True:
            self.DIMS = 3
        else:
            self.DIMS = 2
        self.LABEL_BOTH_FACTORS = LABEL_BOTH_FACTORS
        self.dimensions = None
        self.result = {}
        self.clabels = []
        self.rlabels = []
        self.plabels = []
        #self.cur_depth = self.recur_len(self.collections)
        if isinstance(self.collections, dict):
            # print("dict passed")
            self.collection_names = [x for x in self.collections.keys()]
            self.collections = [x for x in self.collections.values()]
        #print(type([y[0] for y in self.collections][0]))
        # if a dictionary is inputed, then get names from dictionary
        if type([y[0] for y in self.collections][0]) is tuple: #will need to include checks for size of sample 
            print ("yay mca")
            self.collection_names = list(set([x[0] for y in self.collections for x in y]))            
            if self.index:
                self.collections = self.sublist(self.collections, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
            self.mca(self.collections, self.collection_names)
        else:            
            #self.collections = dict([(x[0], x[1]) for y in self.collections for x in y])
            if not self.collection_names:
                self.collection_names = range(1, len(self.collections)+1)
            # if index var is provided, use index to filter collection list
            if self.index:
                self.collections = self.sublist(self.collections, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
        #two sample venn
            if len(self.collections) == 2:
                self.response = self.two_venn(self.collections)
        #three sample venn
            elif len(self.collections) == 3:
                self.response = self.three_venn(self.collections)
        #use mca for greater than three
            elif len(self.collections) >3:
                if var:
                    self.var = var
                else: 
                    self.var = []
                self.ca = self.ca(self.collections, self.collection_names)
            else:
                self.no_compare()
                
    def recur_len(self, L):
        return sum(L + recur_len(item) if isinstance(item, list) else L for item in L)   
    def no_compare(self):
        return ("Need at least two collections to compare results.")
    #get a sublist from a list of indices
    def sublist (self, list1, list2):
        return([list1[x] for x in list2]) 
    
    ## Data processing functions
    
    def processCollection (self, path):
        #initialise vars:
        urls = []
        #establish the data folder
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print (filename) #see the filenames available.
                urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1]) for line in file.readlines()}))
        return(urls)
    
    def convert_full_urls (self, path):
        collection = dict()
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print(filename)
        # result:  {'www.url1.suf', 'www.biglovely.url2.suf', 'education.url3.suf'}
                collect2 = [x.split(".")[-2]+"."+x.split(".")[-1] for x in collect]
                collection[filename[0:15]] = (collect2) #convert collect2 to a dict {truncatedFILENAME: [url1.suf, url2.suf, url3.suf]}
        return (collection)
            
    def convert_subdomain_urls (self, path):
        collection = dict()
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print(filename)
        #split the data by comma and lose the closing url. Put it in a set to remove duplicates.
                collect = {line.translate(str.maketrans(')'," ")).split(",")[1] for line in file.readlines()}
                collect4 = [x for x in collect]
        # result merely converts each set to a big list of urls. (Full scope of analysis)
                collection[filename[0:15]] = (collect4) #convert collect4 to a dict.
        return(collection)

    def removekey(self, d, key):
        r = dict(d)
        del r[key]
        return r
    
    #get set of all items (unduplicated)
    def unionize (self, sets_list):
        return set().union(*sets_list)
    def two_venn (self, collections):
        self.V2_AB = set(collections[0]).intersection(set(collections[1]))
        return  (venn2([set(x) for x in collections], set_labels=self.collection_names))
    def three_venn (self, collections):
        self.V3_ABC = set(collections[0]) & set(collections[1]) & set(collections[2]) 
        self.V3_AB = set(collections[0]) & set(collections[1]) - self.V3_ABC
        self.V3_BC = set(collections[1]) & set(collections[2]) - self.V3_ABC
        self.V3_AC = set(collections[0]) & set(collections[2]) - self.V3_ABC
        self.V3_A = set(collections[0]) - (self.V3_ABC | self.V3_AB | self.V3_AC )
        self.V3_B = set(collections[1]) - (self.V3_ABC | self.V3_AB | self.V3_BC )
        self.V3_C = set(collections[2]) - (self.V3_ABC | self.V3_BC | self.V3_AC )
        return  (venn3([set(x) for x in collections], set_labels=self.collection_names))
    def ca(self, collections, names):
        # use dd to create a list of all websites in the collections
        print (names)
        dd = self.unionize(collections)
        d = [] #create
        e = [] #labels
        fs, cos, cont = 'Factor Score', 'Squared cosines', 'Contributions x 1000'
        #populate table with matches for actors (weblists)
        for y in collections:
            d.append({x: x in y for x in dd})
        #if self.var:
        #    e = ({x.split(".")[0]: x.split(".")[1] for x in dd })
        df = pd.DataFrame(d, index=names)       
        if self.REMOVE_SINGLES:
            df = df.loc[:, df.sum(0) >1 ]
            df.fillna(False)
        #if self.var:
        #    df.loc[:,"SUFFIX"] = pd.Series(e, index=df.index)
        self.response = df.T
        counts = mca.mca(df)
        self.dimensions = counts.L
        print(self.dimensions)
        data = pd.DataFrame(columns=df.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))
        self.result["rows"] = counts.fs_r(N=self.DIMS).T
        self.result["columns"] = counts.fs_c(N=self.DIMS).T
            #self.result["df"] = data.T[fs].add(noise).groupby(level=['Collection'])
        #data.loc[fs,    :] = counts.fs_r(N=self.DIMS).T
        points = self.result["rows"]
        urls = self.result["columns"]
        if self.DIMS == 3:
            clabels = data.columns.values
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')

            plt.margins(0.1)
            plt.axhline(0, color='gray')
            plt.axvline(0, color='gray')
            ax.set_xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)') 
            ax.set_ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
            ax.set_zlabel('Factor 3 (' + str(round(float(self.dimensions[2]), 3)*100) + '%)')
        
            ax.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
            ax.scatter(*urls, s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
            for clabel, x, y, z in zip(clabels, *points):
                ax.text(x,y,z,  '%s' % (clabel), size=20, zorder=1, color='k') 
        else:
            self.clabels = data.columns.values
            plt.figure(figsize=(25,25))
            plt.margins(0.1)
            plt.axhline(0, color='gray')
            plt.axvline(0, color='gray')
            plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)') 
            plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
            plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
            plt.scatter(*urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
            for label, x, y in zip(self.clabels, *points):
                plt.annotate(label, xy=(x, y), xytext=(x + .03, y + .03))
            if self.LABEL_BOTH_FACTORS:
                self.rlabels = df.T.index
                for label, x, y in zip(self.rlabels, *urls):
                    plt.annotate(label, xy=(x, y), xytext=(x + .03, y + .03))
            plt.show()
        return(data.T)
    
    def mca(self, collections, names):
        #print ([x[2] for y in collections for x in y][0:3])
        default = defaultdict(list)
        coll = defaultdict(list)
        src_index, var_index, d = [], [], []
        for x in collections:
            for y,k,v in x:
                default[y+'%'+k].append(v)
        #print(list(default)[0:3])
        dd = self.unionize([j for y, j in default.items()])
        #print (dd)
        for key, val in default.items():
            #print (key)
            keypair = key.split("%")
            collect, year = keypair[0], keypair[1]
            coll[collect].append(year)
            d.append({url: url in val for url in dd})
        for happy, sad in coll.items():
            src_index = (src_index + [happy] * len(sad))
        #src_index = (happy * len(sad) for happy, sad in coll.items())
            var_index = (var_index + sad)
        col_index = pd.MultiIndex.from_arrays([src_index, var_index], names=["Collection", "Date"])
        #X = {x for x in (self.unionize(collections))}
        table1 = pd.DataFrame(data=d, index=col_index, columns=dd)
        if self.REMOVE_SINGLES:
            table1 = table1.loc[:, table1.sum(0) >1 ]
        table2 = mca.mca(table1)
        #print (table2.index)
        self.response = table1
        self.dimensions = table2.L 
        #print(table2.inertia)
        fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
        data = pd.DataFrame(columns=table1.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, self.DIMS+1)]))
        #print(data)
        noise = 0.07 * (np.random.rand(*data.T[fs].shape) - 0.5)
        if self.DIMS > 2:
            data.loc[fs, :] = table2.fs_r(N=self.DIMS).T
            self.result["rows"] = table2.fs_r(N=self.DIMS).T
            self.result["columns"] = table2.fs_c(N=self.DIMS).T
            self.result["df"] = data.T[fs].add(noise).groupby(level=['Collection'])
            
        data.loc[fs,    :] = table2.fs_r(N=self.DIMS).T
 #       print(data.loc[fs, :])

        #print(points)
        urls = table2.fs_c(N=self.DIMS).T
        self.plabels = var_index        

        fs_by_source = data.T[fs].add(noise).groupby(level=['Collection'])

        fs_by_date = data.T[fs]
        self.dpoints = data.loc[fs].values
        print(self.dpoints[1:3])
        fig, ax = plt.subplots(figsize=(10,10))
        plt.margins(0.1)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)')
        plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
        ax.margins(0.1)
        markers = '^', 's', 'o', 'o', 'v', "<", ">", "p", "8", "h"
        colors = 'r', 'g', 'b', 'y', 'orange', 'peachpuff', 'm', 'c', 'k', 'navy'
        for fscore, marker, color in zip(fs_by_source, markers, colors):
            #print(type(fscore))
            label, points = fscore
            ax.plot(*points.T.values[0:1], marker=marker, color=color, label=label, linestyle='', alpha=.5, mew=0, ms=12)
            for plabel, x, y in zip(self.plabels, *self.dpoints[1:3]):
                print(plabel)
                #print(xy)
                plt.annotate(plabel, xy=(x, y), xytext=(x + .15, y + .15))
        ax.legend(numpoints=1, loc=4)
        plt.show()
        


# ## Processing the data
# 
# We requested dates, links and the counts (this tells us the most popular links) from WARC files. We've stored the data in a folder called "assembled."
# 
# These archives include examples of tweets from the Alberta Provincial Election \#42. This election was interesting because it represented a shift from long-standing Alberta Conservative governments to a more left-leaning NDP.  Other important events were Idle No More and the Ottawa Shooting on October 2014.
# 
# There are also archives that
# contain grey literature from the Humanities, Health Sciences and Canadian Business. 
# 
# And then there are some more general archives.  For instance, Canadian Political Parties, Energy and the Environment
# and Humanities Computing.
# 
# The first three results from each output looks something like this:
# 
# (([CRAWL DATE], link), count)
#  
# ((201601,linkis.com),11102) 
# ((201601,m.youtube.com),8764)
# ((201601,www.youtube.com),7481)
# 
# This gives us the ability to compare the different archive collections through links.  (Later this description will
# contain examples that use the dates as well.)
# 

#initialise vars:
collection = dict()
collection_2 = dict()
var = dict()

#establish the data folder
path = "assembled/"

#get the files

print ("These are the files that have been accessed through this script:\n\n")
for filename in os.listdir(path):
    with open(path+filename, "r") as file:
        print(filename)
        #split the data by comma and lose the closing url. Put it in a set to remove duplicates.
        collect = {line.translate(str.maketrans(')'," ")).split(",")[1] for line in file.readlines()}
        # result:  {'www.url1.suf', 'www.biglovely.url2.suf', 'education.url3.suf'}
        collect2 = [x.split(".")[-2]+"."+x.split(".")[-1] for x in collect]
        # result: ['url1.suf', 'url2.suf', 'url3.suf']  (this decreases scope of analysis - removes "education" 
        # in education.ab.ca), for example.
        collect4 = [x for x in collect]
        # result merely converts each set to a big list of urls. (Full scope of analysis)
        collection[filename[0:10]] = (collect2) #convert collect2 to a dict {truncatedFILENAME: [url1.suf, url2.suf, url3.suf]}
        collection_2[filename[0:10]] = (collect4) #convert collect4 to a dict.

#Just separate the names and values for now.
comparit = [x for x in collection_2.values()]
names = [x.upper() for x in collection_2.keys()]


""" 
Since the variable "collection" has 16 different archives, we can use the index variable to choose two. When you 
have two collections, then Compare will provide you with a venn diagram with two variables.

You can find out the content inside the circles by using V2_[A, B or AB].


(V2_A (V2_AB) V2_B)

"""

#Two collections will produce a two-way Venn diagram showing the cross over in terms of links.
#Since collection is a dict() no need to include names.
compare1 = Compare(collection, index=[4,7])
print("Links in Common (there are "+ str(len(compare1.V2_AB)) + ") : /n" )
for x in compare1.V2_AB:
    print (str(x) + "\n")


#Although you can add your own names if you want to ...  (recall "names" is x.upper()))

compare1 = Compare(collection, names=names, index=[4,7] )
print("Links that both collections have in common: " + ", ".join(compare1.V2_AB))


# What happens with three collections
compare2 = Compare(comparit, names, [2,0,5])
print("Links that all collections have in common: " + ', '.join(compare2.V3_ABC))


# With four or more collections, Compare switches to correspondence analysis. Correspondence analysis visualizes relationships under two categories.  In our case, collections and the
# links contained (or not contained) in them. 
# 
# In this case, the collections have been named and are represented by red circles.  The blue squares represent links that
# two or more of the collections have in common, I have not included names here because there are too many.
# 
# Focussing on the red circles, when two collections are close together in the analysis they tend to have more links
# in common. 
# 
# You can also consider the dimensions. In theory, there are as many dimensions of analysis as there are collections.
# In practice, you can only show the top two most explanatory dimensions (see inertia scores below).  The dimensions 
# sometimes can demonstrate factors in real life.  For example, in the example below, we have the Ottawa shootings on 
# one side of the horizontal axis, and heritage community on the other so it appears that there is a dimension that 
# ranges from federal focus (left of the y axis) to Alberta focus (right of the y axis).
# 
# On the vertical axis, it seems like you have prairie province related materials. More study would be necessary, but 
# perhaps this is a dimension for grassroots related collections versus more organizational focussed ones.
# 
# Because you have these axes, one can also examine the quadrants as representing some aspect of the collections. 
# Top left is empty.  Top right is prairie province related.  Bottom right is Alberta-focussed.  Bottom left is federal /
# Canada wide.
# 
# There are also some percentage scores associated with each factor.  Given that we are seeing only 10.9 & 11%
# respectively, perhaps some other form of analysis would be more effective than links.
# 
# From a collection evaluation perspective, it might be worth examining the prairie provinces and idle no more 
# collections for significant cross-over, or perhaps to ensure that they have some cross-references. However, some
# caution is necessary to ensure that the cross-over occurs due to meaningful reasons rather than some accident of 
# Internet behavior (eg. maybe YouTube is used more frequently by those collections).
# 

# With more than three collections, the output switches to correspondence analysis.
# Katherine Faust offers a great overview of the method here:
# www.socsci.uci.edu/~kfaust/faust/research/articles/articles.htm

# In this case, we've eliminated a few items from the analysis.

compare3 = Compare(collection, names, [i for i, x in enumerate(collection) if i not in [5,6,7,14,16,17,18,19,21,23,24]])


"""
Inertia is somewhat like an R-squared score for a correspondence graph.  The overall inertia is only 40%, so quite low.
However, the third and fourth dimensions of analysis also seem about as relevant and the first and second.

Later developments will show how to look at the correspondence from these perspectives as well.
"""

# Compare.dimensions store factor values for all the dimensions.  The top two dimensions are shown in the graph above.
# Inertia or the total explantory value of the graph is the sum of all these values.
print ([round(x,3) for x in compare3.dimensions])
print ("Inertia: " + str(sum(compare3.dimensions)))


# Compare.response stores the table that these scores are based on (I've selected only 5 items for clarity)
#If REMOVE_SINGLES is not FALSE, then no row should contain fewer than two "TRUES".

print (compare3.response[10:15])


"""
Let's do the same thing with a different dataset:
 
 Name Entity Recognition (NER) offers a way of identifying locations named within the collections.  
 So we have data in csv format like this:
 
CPP 201508,Sum of Frequency
Canada ,283162
Ontario ,34197
United States,32008
Ottawa ,30233
Toronto ,18787
Alberta,17015
British Columbia ,13181
Manitoba,11332
Hawaiian ,10110
QUEBEC,9633
Vancouver,9442

"""


#initialise vars:
loc_collection = dict()
loc_collection_2 = dict()
loc_var = dict()

#establish the data folder
loc_path = "../../NER/"

#get the files
for loc_filename in os.listdir(loc_path):
    with open(loc_path+loc_filename, "r", encoding="utf-8", errors="ignore") as loc_file:
        print(loc_filename)
        loc_collect = [row[0] for row in csv.reader(loc_file)]
        loc_collection[loc_filename[0:10]] = (loc_collect)


#Just separate the names and values for now.
loc_comparit = [x for x in loc_collection.values()]
loc_names = [x.upper() for x in loc_collection.keys()]


"""  This provides a graph that shows the dates (labels) in action by collection (shape/color).  It is also possible to show the 
collections by date if desired.
"""

compare4 = Compare(urls, index=[0,1,2,3], DIMENSIONS_OP=3)
#print(compare4.dimensions)
#print(compare4.response[0:10])


loc_compare_1 = Compare(loc_collection)
#print ("Locations in all Three collections: \n\n" + ", ".join(loc_compare_1.V3_ABC) + "\n ... \n")
#print ("in Just the First (top Left): \n\n" + ", ".join(loc_compare_1.V3_A) + "\n ... \n")
#print ("in top left and bottom \(purple section\): \n\n" + ", ".join(loc_compare_1.V3_AC) + "\n ... \n")
#print ("in top right and bottom \(light blue\): \n\n" + ", ".join(loc_compare_1.V3_BC) + "\n ... \n")


""" 
Now let\'s try to use the dates in our analysis.  This brings up the power a little to use multiple correspondence 
analysis.

((201601,linkis.com),11102), 
((201601,m.youtube.com),8764),
((201601,www.youtube.com),7481)

"""

#initialise vars:
dat_collection = dict()
dat_collection_2 = dict()
dat_var = dict()
dat_collect = dict()
#establish the data folder
dat_path = "assembled/"
urls = []

#get the files
for dat_filename in os.listdir(dat_path):
    with open(dat_path+dat_filename, "r") as dat_file:
        print (dat_filename)
        urls.append(list({(dat_filename[0:10], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1]) for line in dat_file.readlines()}))

""" This produces something like this:
[ { (COLLECTION_NAME, DATE, URL)}]  (the {} is to make it unique.)

"""


#Unit tests to be removed later.

import unittest

class CompareTests(unittest.TestCase):
    
    collection1 = ["google", "apple", "microsoft", "msn", "napster", "oracle", "amazon", "ibm"]
    collection2 = ["google", "pear", "thebeatles", "thepogues", "napster", "apple", "cow"]
    collection3 = ["google", "apple", "msn", "skunk", "beaver", "wolf", "cow"]
    collection4 = ["apple", "jump", "walk", "run", "saunter", "skunk", "napster"]
    collection5 = ["pear", "wolf", "jive", "tango"]
    collection6 = ["google", "apple", "msn", "thepogues", "napster", "wolf", "amazon", "tango"]
    one_collect = [collection1]
    two_collect = [collection1, collection2]
    three_collect = [collection1, collection2, collection3]
    all_collect = [collection1, collection2, collection3, collection4, collection5, collection6]
    
    def test_one (self):
        print("test error with one collection")
        self.assertTrue(Compare(self.one_collect), "Need at least two collections to compare results.")
        
    def test_two (self):
        print ("test results for two collections")
        self.assertTrue(Compare(self.two_collect).response.subset_labels[1].get_text(), 4)
        self.assertTrue(Compare(self.two_collect).response.subset_labels[0].get_text(), 5)
        self.assertTrue(Compare(self.two_collect).response.subset_labels[2].get_text(), 3)
    
    def test_three (self):
        print("test results for three")
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 4)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[1].get_text(), 3)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 3)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 2)
    
    def test_all (self):
        print("test results for more than three")
        test=Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False)
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.iloc[1].values), 
                        [True, True, True, True, False, True])
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.ix['amazon'].values),
                        [True, False, False, False, False, True])
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.iloc[5].values),
                        [True, False, False, False, False, False])
        
        
        

suite = unittest.TestLoader().loadTestsFromTestCase(CompareTests)
unittest.TextTestRunner().run(suite)
        



compare3.ca.loc[:,'SUFFIX'] = pd.Series(var, index=compare3.ca.index)
print (compare3.ca)
compare3.ca.to_csv("output.csv",  encoding='utf-8')





# # WALK	: Web Archives for Longitudinal Research
# 
# "WALK will enable present and future humanities and social science scholars to access, interpret and curate born-digital primary resources that document our recent past."
# 
# This is a demo/tutorial for the **University of Victoria** on our up-to-now explorations on Web Archives.
# 
# Over the summer, I have had the pleasure of working with Dr. Ian Milligan and Nick Ruest, building a tool for conducting analytics on the Web Archive collections. This has involved working with [Warcbase](http://lintool.github.io/warcbase-docs/) in order to extract important data from the collections and using Python including the pandas / numpy libraries via Anaconda, a framework intended for analytics-style work.
# 
# I have called this tool / library *Compare* because it uses Warcbase derivative data from the archives / WARC files.
# 
# Compare uses either Simple (CA) or Multiple Correspondence Analysis (MCA) to explore relationships among different collections of data. It is like factor analysis.  I have a demo of Simple CA [using parliamentary committees and MPs as a sample.](https://github.com/web-archive-group/WALK/blob/master/Scripts/compare_collection/UnderstandingMCA.ipynb). 
# 

#First let's import the necessary libraries.
get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries
import matplotlib.pyplot as plt #plotting library
from mpl_toolkits.mplot3d import Axes3D #for 3d graphs
import copy #need deepcopy() for working with the dictionaries.

####   Uncomment the items below if you want 
####   to use D3 for output.

#import mpld3
#mpld3.enable_notebook()


# While many questions could be answered using MCA, we have decided to focus on a simple problem statement: ***Can we conduct an evaluation of a collection of web archives using data analytics?*** For this demo, I have included a number of collections:
# 
# - Two 'external' collections (UofA prairie provinces & UFT Canadian Labour Unions)
# - Three Twitter crawls.
# - University of Victoria collections (10 in all)
# 
# We have also included some dummy collections based on the most popular websites, categorized by media, social media, government, tech industry and NGOs.
# 
# While there are a number of ways we could compare the collections, this example will use web domain names (urls). Often MCA tries to compare both factors (collections *and* urls) but including the urls would be too difficult. (Besides, do we really need to understand why www.youtube.com is in multiple collections?)  The urls will be represented by off-white squares and no labels to keep them from confusing the analysis.
# 
# We assume that libraries will continue not to count format as a selection criteria, meaning that web archives are selected on the same criteria as books, journals or any other source. We have decided to focus on the following principles of collection management, however:
# 
# - Coverage / Scope
#   + What are the best ways to evaluate coverage?
# - Accuracy
#   + Can we detect problems (eg. ineffective crawls)
# - Relevance
#   + Is a historian likely to find something unique or interesting in the collection?
# - Dynamics
#   + How has the collection changed from crawl to crawl?
# 

# The output below takes the derivative files from the folder "UFT/" and puts them into a python dictionary (array)
# for later use.  I have included two of these. One including the Heritage Community Foundation's collection and one
# not.

path = "UVIC/"

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

#newdict = defaultdict(dict)
newdict = defaultdict(lambda: defaultdict(list))
newdict2 = defaultdict(lambda: defaultdict(list))
PC = processCollection(path)
#print(list(zip(PC[0])))
#print(list(zip(PC[0][0])))
#print (**collect)
for collect in PC:
    for coll, date, url in collect:
        newdict[date][coll].append(url)

# newdict will provide all the data like so:

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll == 'ALBERTA_heritag' or coll=='DUMMY_GOVERNMEN' or coll == 'ALBERTA_hcf_onl' or coll=="DUMMY_MEDIA" or coll=="DUMMY" or coll== "ORGDUMMY" or coll== "TECHDUMMY" or coll== "SOCIALMEDIADUMM" or coll=="GOVDUMMY":
            pass
        else:
            newdict2[date][coll].append(url)

#{'DATE': {'COLLECTION': ['url1.com', 'url2.com', 'etc']}}
#

 


## Produce a dictionary output that creates a list of outputs suitable for analysis by date.
##
## collection_var[-1] would analyze all the links together until the latest year (2016). collection_var[-2]
## would analyze everything up to t-1 (2015).
##
## Our hope for the future is that the data could be used in an animation, showing changes over time. But for now, 
## we will just show the progress.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

collection_var = reduce_collections (copy.deepcopy(newdict2))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.


# First we will start with the earliest date (2014). Eleven collections have data listed for these dates.
# 
# When there are fewer than 3 collections, you get a venn diagram showing the cross-connections. You can also extract data using V2_AB object or V3_ABC (if there are three items). 
# 

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["down", "depressed", "nostalgic"]}, LABEL_BOTH_FACTORS=True)


# In 2014, we have eleven collections and move to correspondence analysis.  
# 
# To analyse, I suggest a three pronged approach.
# 
# - HA: What does the horizontal axis mean?
# - VA: What does the vertical axis mean?
# - QUA: Can we name the quadrants?
# - CLUS: What sources have clustered together
# 
# HA-VA-QUA-CLUS!
# 
# We can speculate, for instance, that the horizontal axis represents collections specific to UVIC and collections curated by UVIC about other issues.  The vertical axis, may have something to do with the differences between Vancouver Island local news and the other collections.

UVIC = Compare(collection_var[-3])


# According to this graph, a strong distinction exists for the VI News collection. Adding a dummy collection puts VI NEWS and trans_web on the same vertical axis but both collections have few connections to the MEDIA DUMMY.  This probably means that the "news" sites do not link frequently to the more popular news media sites.  
# 

#collection_var[-3] = removekey(collection_var[-3], 'SOCIALMEDIA')
collection_var[-3]['MEDIADUMMY'] = newdict['2015']['DUMMY_MEDIA']

UVIC = Compare(collection_var[-3])


# Going to 2015 (which is a cumulative look at the collections, so includes both 2014 & 2015) the level of difference among the collections reduces by a lot.
# 

UVIC = Compare(collection_var[-2])


# Moving to 2016, we now see the WAHR Twitter collections included, which greatly expand the difference among the 
# collections overall. So the collections are distinct from Twitter collections.
# 

UVIC = Compare(collection_var[-1])
#print("There are "+ str(len(UVIC.V2_AB)) + " members in common. They are : " + str(UVIC.V2_AB))


# Removing the Twitter examples, again we see that the collections are not very distinct in terms of url exchange.
# 

collection_var[-1] = removekey(collection_var[-1], 'WAHR_ymmfire-ur')
collection_var[-1] = removekey(collection_var[-1], 'WAHR_panamapape')
collection_var[-1] = removekey(collection_var[-1], 'WAHR_exln42-all')
Compare(collection_var[-1])
#print("There are "+ str(len(UFT.V2_AB)) + " members in common. They are : " + str(UFT.V2_AB))


# *Conclusion*: The web archive collections are quite similar in terms of url content. More research is necessary to determine why, but analysis of collection policies regarding web archives may be beneficial to decide whether the current approach is meeting organizational needs. The lack of distinction among the collections is an area of curiosity, due to a lack of internal diversity among the sources.
# 

TestDict1 = {'2009': {'c1': {'lk1', 'lk2', 'lk3'},
                     'c2': {'lk1', 'lk10', 'lk20', 'lk2'},
                     'c3': {'lk3', 'lk10', 'lk33', 'lk4'}},
            '2010': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c3': {'lk10', 'lk9', 'lk7'}},
            '2011': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c4': {'lk1', 'lk2', 'lk3'}},
            '2012': {'c1': {'lk1', 'lk99', 'lk6'}}
           }

#print(list(zip(*zip(TestDict['2009'])))


# # WALK	: Web Archives for Longitudinal Research
# 
# "WALK will enable present and future humanities and social science scholars to access, interpret and curate born-digital primary resources that document our recent past."
# 
# This is a demo/tutorial for the **University of Winnipeg** on our up-to-now explorations on Web Archives.
# 
# I have called this tool / library *Compare* because it uses Warcbase derivative data from the archives / WARC files.
# 
# Compare uses either Simple (CA) or Multiple Correspondence Analysis (MCA) to explore relationships among different collections of data. It is like factor analysis.  I have a demo of Simple CA [using parliamentary committees and MPs as a sample.](https://github.com/web-archive-group/WALK/blob/master/Scripts/compare_collection/UnderstandingMCA.ipynb). 
# 

#First let's import the necessary libraries.
get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries
import matplotlib.pyplot as plt #plotting library
from mpl_toolkits.mplot3d import Axes3D #for 3d graphs
import copy #need deepcopy() for working with the dictionaries.

####   Uncomment the items below if you want 
####   to use D3 for output.

#import mpld3
#mpld3.enable_notebook()


# While many questions could be answered using MCA, we have decided to focus on a simple problem statement: ***Can we conduct an evaluation of a collection of web archives using data analytics?*** For this demo, I have included a number of collections:
# 
# - Three Twitter crawls.
# - University of Winnipeg collections (5 in all)
# 
# We have also included some dummy collections based on the most popular websites, categorized by media, social media, government, tech industry and NGOs.
# 
# While there are a number of ways we could compare the collections, this example will use web domain names (urls). Often MCA tries to compare both factors (collections *and* urls) but including the urls would be too difficult. (Besides, do we really need to understand why www.youtube.com is in multiple collections?)  The urls will be represented by off-white squares and no labels to keep them from confusing the analysis.
# 
# We assume that libraries will continue not to count format as a selection criteria, meaning that web archives are selected on the same criteria as books, journals or any other source. We have decided to focus on the following principles of collection management, however:
# 
# - Coverage / Scope
#   + What are the best ways to evaluate coverage?
# - Accuracy
#   + Can we detect problems (eg. ineffective crawls)
# - Relevance
#   + Is a historian likely to find something unique or interesting in the collection?
# - Dynamics
#   + How has the collection changed from crawl to crawl?
# 

# The output below takes the derivative files from the folder "UFT/" and puts them into a python dictionary (array)
# for later use.  I have included two of these. One including the Heritage Community Foundation's collection and one
# not.

path = "WINNIPEG/"

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

#newdict = defaultdict(dict)
newdict = defaultdict(lambda: defaultdict(list))
newdict2 = defaultdict(lambda: defaultdict(list))
PC = processCollection(path)
#print(list(zip(PC[0])))
#print(list(zip(PC[0][0])))
#print (**collect)
for collect in PC:
    for coll, date, url in collect:
        newdict[date][coll].append(url)

# newdict will provide all the data like so:

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll == 'ALBERTA_heritag' or coll=='DUMMY_GOVERNMEN' or coll == 'ALBERTA_hcf_onl' or coll=="DUMMY_MEDIA" or coll=="DUMMY" or coll== "ORGDUMMY" or coll== "TECHDUMMY" or coll== "SOCIALMEDIADUMM" or coll=="GOVDUMMY":
            pass
        else:
            newdict2[date][coll].append(url)

#{'DATE': {'COLLECTION': ['url1.com', 'url2.com', 'etc']}}
#

 


## Produce a dictionary output that creates a list of outputs suitable for analysis by date.
##
## collection_var[-1] would analyze all the links together until the latest year (2016). collection_var[-2]
## would analyze everything up to t-1 (2015).
##
## Our hope for the future is that the data could be used in an animation, showing changes over time. But for now, 
## we will just show the progress.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

collection_var = reduce_collections (copy.deepcopy(newdict2))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.


# First we will start with the earliest date (2013). Two collections have data listed for these dates.
# 
# When there are fewer than 3 collections, you get a venn diagram showing the cross-connections. You can also extract data using V2_AB object or V3_ABC (if there are three items). 
# 

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["down", "depressed", "nostalgic"]}, LABEL_BOTH_FACTORS=True)


# By 2015, we have four collections and move to correspondence analysis.  
# 
# To analyse, I suggest a three pronged approach.
# 
# - HA: What does the horizontal axis mean?
# - VA: What does the vertical axis mean?
# - QUA: Can we name the quadrants?
# - CLUS: What sources have clustered together
# 
# HA-VA-QUA-CLUS!
# 
# Below, try changing -4 to -3 to -2 to -1 to move forward in time from 2013.

UVIC = Compare(collection_var[-1])


# According to this graph, a strong distinction exists for the VI News collection. Adding a dummy collection puts VI NEWS and trans_web on the same vertical axis but both collections have few connections to the MEDIA DUMMY.  This probably means that the "news" sites do not link frequently to the more popular news media sites.  
# 

#collection_var[-3] = removekey(collection_var[-3], 'SOCIALMEDIA')
collection_var[-1]['MEDIADUMMY'] = newdict['2015']['DUMMY_MEDIA']

UVIC = Compare(collection_var[-1])


WINNIPEG = Compare(collection_var[-1])


# Moving to 2016, we now see the WAHR Twitter collections included, which greatly expand the difference among the 
# collections overall. So the collections are distinct from Twitter collections.
# 

WINNIPEG = Compare(collection_var[-1])


# Removing the Twitter examples, again we see that the collections are not very distinct in terms of url exchange.
# 

#collection_var[-1] = removekey(collection_var[-1], 'WAHR_ymmfire-urls')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_panamapapers-urls')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_exln42-all')
collection_var[-1] = removekey(collection_var[-1], 'MEDIADUMMY')
Compare(collection_var[-1])


# *Conclusion*: The web archive collections are quite similar in terms of url content. More research is necessary to determine why, but analysis of collection policies regarding web archives may be beneficial to decide whether the current approach is meeting organizational needs. The lack of distinction among the collections is an area of curiosity, due to a lack of internal diversity among the sources.
# 

TestDict1 = {'2009': {'c1': {'lk1', 'lk2', 'lk3'},
                     'c2': {'lk1', 'lk10', 'lk20', 'lk2'},
                     'c3': {'lk3', 'lk10', 'lk33', 'lk4'}},
            '2010': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c3': {'lk10', 'lk9', 'lk7'}},
            '2011': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c4': {'lk1', 'lk2', 'lk3'}},
            '2012': {'c1': {'lk1', 'lk99', 'lk6'}}
           }

#print(list(zip(*zip(TestDict['2009'])))


