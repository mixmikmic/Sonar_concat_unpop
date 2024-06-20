# # Generating Fire Safety Complaint Features
# ## Introduction
# This notebook documents the process of generating feature data from the file `matched_Fire_Safety_Complaints.csv`
# 

# ## Loading the File
# 

import pandas

PATH_TO_CSV = "./data/matched_Fire_Safety_Complaints.csv"
complaints = pandas.read_csv(PATH_TO_CSV)


# ## Examining and Grouping the Complaint Item Types
# Using Python's `Counter`, accumulate all the types of complaints, and the number of each type, in this dadaset.
# 

from collections import Counter

complaint_types = Counter(complaints["Complaint Item Type Description"])
complaint_types


# By looking at the total types, it seems that there are two complaints that have a type "nan", which appear to be complaints that weren't specified any type. We can drop these two.
# Several other complaint types seem to be more related to the safety of evacuation in case of fire (e.g. "blocked exits"), which are not related to the risk of fire happening. (We may drop these?)
# Furthermore, we may consider grouping similar complaint types.
# First, we may group many of the complaints into three categories: "potential fire cause", "potential fire control", and "fire emergency safety".
# * Potential Fire Cause: 
#     * general hazardous materials
#     * leaking underground tanks
#     * hoarding
#     * combustible materials
#     * weeds and grass
#     * open vacant building
#     * refused hood + duct service
#     * electrical systems
#     * unlicensed auto repair
# * Potential Fire Control:
#     * alarm systems
#     * sprinkler/standpipe systems
#     * extinguishers
# * Fire Emergency Safety
#     * roof access
#     * unapproved place of assembly
#     * fire escape
#     * blocked exits
#     * exit maintenance
#     * street numbering
#     * overcrowded place of assembly
#     
# In addition, I'm not exactly sure how to categorize the following complaint types:
# * ul cert verification
# * crisp complaint inspection
# * uncategorized complaint
# * multiple fire code violations
# * illegal occupancy
# * operating without a permit
# 

# ## The Disposition Column
# 
# The possible values in this column can be acquired with the following step:
# 

disposition_types = Counter(complaints["Disposition"])
disposition_types


# There are 11 types of disposition values. The type 'no merit' and the type 'duplicate complaint' may indicate that the complaint is not valid, therefore we may ignore such complaints in the final output. 
# There are also 2355 complaints where the disposition value is NaN. These complaints are preserved and counted towards the final output at this moment.
# 

# ## Generating Output
# 
# First, define a few functions to check the values of each complaint:
# 

from datetime import date

def is_valid_complaint(row):
    disposition = row["Disposition"]
    return not (disposition == "no merit" or disposition == "duplicate complaint")

def is_corrected(row):
    disposition = row["Disposition"]
    return disposition == "condition corrected"

def parse_date(date_str):
    """For a string in the format of YYYY-MM-DD, 
    return (YYYY, MM, DD)"""
    return tuple(map(int, date_str.split('-')))

def is_within_date_range(row, min_date_str, max_date_str):
    """checks if beg <= row["Received Date"] <= end
    row: a row in the dataset, representing one complaint
    min_date_str: a str representing the beginning of the date range
    max_date_str: a str representing the end of the date range
    """
    complaint_date = date(*parse_date(row["Received Date"]))
    min_date = date(*parse_date(min_date_str))
    max_date = date(*parse_date(max_date_str))
    
    return min_date <= complaint_date and max_date >= complaint_date


# Next, define a mapping from a complaint description to a more general complaint category, following the previous observation. 
# 

# get the mappting from Complaint Item Type Description to Complaint Item Type
complaint_id_mapping = {}

for i, r in complaints.iterrows():
    dsc = r["Complaint Item Type Description"]
    complaint_id = r["Complaint Item Type"]
    if dsc in complaint_id_mapping:
        if complaint_id_mapping[dsc] != complaint_id:
            raise Exception("Complaint Type has different IDs")
    else:
        complaint_id_mapping[dsc] = complaint_id

complaint_id_mapping


# define mapping from complaint item type to category
potential_fire_cause = "potential fire cause"
potential_fire_control = "potential fire control"
fire_emergency_safety = "fire emergency safety"
multiple_violations = "multiple violations"

complaint_category_mapping = {"potential fire cause":['15', '13', '18', '10', '01', '07', '12', '20', '04'],
                              "potential fire control":['05', '19', '06'], 
                              "fire emergency safety": ['03', '24', '22', '02', '23', '21', '11']}
# reverse the mapping to get id -> category mappings
complaint_category_mapping = {d:c for c, d_list in complaint_category_mapping.items()
                                  for d in d_list}


# Now, we're able to generate the output dataset.
# 

from collections import defaultdict
from math import isnan

eas_to_features = defaultdict(lambda :defaultdict(float))

for d, r in complaints.iterrows():
    eas = r["EAS"]
    complaint_type = r["Complaint Item Type"]
    if not isnan(eas) and is_within_date_range(r, "2005-01-01", "2016-12-31"):
        features = eas_to_features[int(eas)]
        # increment count features for generalized complaint types
        if complaint_type in complaint_category_mapping and is_valid_complaint(r):
            feature_name = "count {}".format(complaint_category_mapping[complaint_type])
            features[feature_name] += 1
            features["count all complaints"] += 1
        
            # increment count features for generalized complaint types not corrected:
            if not is_corrected(r):
                feature_name = "count {} not corrected".format(complaint_category_mapping[complaint_type])
                features[feature_name] += 1
                features["count all complaints not corrected"] += 1
        
        # count for each complaint type, maybe remove this?
        #complaint_type_dsc = r["Complaint Item Type Description"]
        #if is_valid_complaint(r):
        #    feature_name = "count {}".format(complaint_type_dsc)
        #    features[feature_name] += 1
        #    
        #    if not is_corrected(r):
        #        feature_name = "count {} not corrected".format(complaint_type_dsc)
        #        features[feature_name] += 1


df = pandas.DataFrame.from_dict(eas_to_features, orient='index', dtype=float)
df.fillna(0, inplace=True)
df


# # Generate tax features
# 

# ### Introduction
# 

# This notebook documents the process of generating feature data from the file matched_Fire_Incidents.csv. These features will be used as the target variables in modeling.
# 

# ### Load libraries and CSV
# 

import numpy as np
import pandas as pd
from datetime import datetime

path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'

#This will take a while to load.  Very large file...
tax_df = pd.read_csv(path + '\\' + 'matched_EAS_Tax_Data.csv', 
              low_memory=False)[[
                 'EAS BaseID',
                 'Neighborhoods - Analysis Boundaries',
                 'Property Class Code',
                 'Property_Class_Code_Desc',
                 'Location_y',
                 'Address',
                 'Year Property Built',
                 'Number of Bathrooms',
                 'Number of Bedrooms',
                 'Number of Rooms',
                 'Number of Stories',
                 'Number of Units',
                 'Percent of Ownership',
                 'Closed Roll Assessed Land Value',
                 'Property Area in Square Feet',
                 'Closed Roll Assessed Improvement Value'
                 ]].dropna()

#Create land value per square foot var
tax_df['landval_psqft'] = tax_df['Closed Roll Assessed Land Value'] / tax_df['Property Area in Square Feet']

tax_df.rename(columns = {'EAS BaseID': 'EAS'}, inplace=True)
tax_df.rename(columns = {'Neighborhoods - Analysis Boundaries': 'Neighborhood'}, inplace=True)


tax_df.head()


# ### Remove outlier observations, then collapse by EAS
# 

def removal(var, low, high):
    tax_df[(tax_df[var]<=low) & (tax_df[var]<=high)]
    return tax_df

#Remove if 0 stories, remove if > 30 stories
tax_df = removal('Number of Stories',1,30)

#Remove if landvalue/sq_foot = 1 or > 1000
tax_df = removal('landval_psqft',1,1000)

#Remove if num. bathrooms, bedrooms, extra rooms > 100
tax_df = removal('Number of Bathrooms',0,100)
tax_df = removal('Number of Bedrooms',0,100)
tax_df = removal('Number of Rooms',0,100)

#Remove if year_built < 1880 or > 2017
tax_df = removal('Year Property Built',1880,2017)

#Remove num units > 250
tax_df = removal('Number of Units',0,250)

#Remove percent ownership < 0, > 1
tax_df = removal('Percent of Ownership',0,1)

#Create Tot_rooms var
tax_df['Tot_Rooms'] = tax_df['Number of Bathrooms'] +                     tax_df['Number of Bedrooms']  +                     tax_df['Number of Rooms']
        
#Subset to numeric vars only, group by EAS average          
tax_df_num = tax_df[[
                 'EAS',
                 'Year Property Built',
                 'Number of Bathrooms',
                 'Number of Bedrooms',
                 'Number of Rooms',
                 'Number of Stories',
                 'Number of Units',
                 'Percent of Ownership',
                 'Closed Roll Assessed Land Value',
                 'Property Area in Square Feet',
                 'Closed Roll Assessed Improvement Value',
                 'Tot_Rooms',
                 'landval_psqft'
                 ]].groupby(by='EAS').mean().reset_index()


pd.options.display.float_format = '{:.2f}'.format
tax_df_num.describe()


# ### Create subset of string vars
# 

tax_df_str = tax_df[[
                 'EAS',
                 'Neighborhood',
                 'Property Class Code',
                 'Property_Class_Code_Desc',
                 'Location_y',
                 'Address',
                 ]].groupby(by='EAS').max().reset_index()

tax_df_str['Property_Class_Code_Desc'] = tax_df_str['Property_Class_Code_Desc'].apply(lambda x: x.upper())
tax_df_str['Neighborhood'] = tax_df_str['Neighborhood'].apply(lambda x: x.upper())


tax_df_str.head()


# ### Create more generalized grouping for Property Class
# 

pd.set_option("display.max_rows",999)
tax_df_str.groupby(['Property Class Code', 'Property_Class_Code_Desc']).count()


# Somewhat difficult to group.  I think we should seperate some of the large categories, and roll all of the smaller categories into "Other".  For example:
# 
# APARTMENTS -  A, AC, DA, TIA  
# DWELLING - D  
# FLATS AND DUPLEX - F, F2, FA, TIF  
# CONDO - Z Condominium  
# COMMERCIAL - C, CD, B, C1, CD, CM, CZ  
# INDUSTRIAL - I, IDC, IW, IX, IZ  
# OFFICE - O, OA, OAH, OBH, OBM, OC, OCH, OCL, OCM, OMD, OZ  
# OTHER - All other codes
# 

di = {'APARTMENT': ['A', 'AC', 'DA', 'TIA'], 
      'DWELLING': ['D'], 
      'FLATS AND DUPLEX': ['F','F2','FA','TIF'], 
      'CONDO, ETC.': ['Z'],
      'COMMERCIAL USE': ['C','CD','B','C1','CD','CM','CZ'],
      'INDUSTRIAL USE': ['I','IDC','IW','IX','IZ'],
      'OFFICE' : ['O', 'OA','OAH', 'OBH', 'OBM', 'OC', 'OCH', 'OCL', 'OCM', 'OMD', 'OZ']}

# reverse the mapping
di = {d:c for c, d_list in di.items()
        for d in d_list}

#Map to 'Building_Cat' groupings var
tax_df_str['Building_Cat'] = tax_df_str['Property Class Code'].map(di)

#Remainders placed in "OTHER" category
x = ['APARTMENT', 'DWELLING', 'FLATS AND DUPLEX', 'CONDO, ETC.', 'COMMERCIAL USE', 'INDUSTRIAL USE', 'OFFICE']
tax_df_str.loc[~tax_df_str['Building_Cat'].isin(x), 'Building_Cat'] = 'OTHER'


tax_df_str['Building_Cat'].value_counts()


# ### Merge DF back, clean up, export 
# 

exp_df = pd.merge(tax_df_str, tax_df_num, how='left', on='EAS')
exp_df.drop(['Property Class Code', 'Property_Class_Code_Desc'], inplace=True, axis=1)


#Rename
exp_df.rename(columns = {'Year Property Built': 'Yr_Property_Built'}, inplace=True)
exp_df.rename(columns = {'Number of Bathrooms': 'Num_Bathrooms'}, inplace=True)
exp_df.rename(columns = {'Number of Bedrooms': 'Num_Bedrooms'}, inplace=True)
exp_df.rename(columns = {'Number of Rooms': 'Num_Rooms'}, inplace=True)
exp_df.rename(columns = {'Number of Stories': 'Num_Stories'}, inplace=True)
exp_df.rename(columns = {'Number of Units': 'Num_Units'}, inplace=True)
exp_df.rename(columns = {'Percent of Ownership': 'Perc_Ownership'}, inplace=True)
exp_df.rename(columns = {'Closed Roll Assessed Land Value': 'Land_Value'}, inplace=True)
exp_df.rename(columns = {'Property Area in Square Feet': 'Property_Area'}, inplace=True)
exp_df.rename(columns = {'Closed Roll Assessed Improvement Value': 'Assessed_Improvement_Val'}, inplace=True)


exp_df.info()


#Export data
exp_df.to_csv(path_or_buf= path + '\\' + 'tax_data_formerge_20170917.csv', index=False)





# # Complaint / Incident Data Exploration
# 
# ### Summary
# This Jupyter notebook performs preliminary exploratory analysis of the relationship between fire safety-related complaints and fire incidents.
# 
# ### Preliminary Findings
# At the EAS level, there appears to be a fairly significant negative relationship between the number of fire safety complaints issued at a certain location, and the incident of a fire at any point between 2006-2016.  This could be because addressing fire safety complaints actually works as a preemptive measure to avoid a fire incident.
# 
# However, these are very preliminary findings and it could be that results change significantly when we filter out some of the types of complaints (like "uncategorized complaint") or subset the data in other ways.
# 
# ### Reccomendations
# Consider using the number of fire safety complaints (or, more simply, a safety complaints dummy) at a given EAS location as a predictor in the model.  We could also modify this variable to filter out "uncategorized" complaints or complaints found to be without merit.
# 

# ## Data preperation
# For now, I am keeping the complaint and incident data simple.  Both datasets are subset to 2006-2016, and I am only considering building fire/cooking fire/trash fire incidents for analysis. 
# 

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import statsmodels.api as sm
import patsy

get_ipython().magic('matplotlib inline')

#Read in data
path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'
complaint_df = pd.read_csv(path + '\\' + 'matched_Fire_Safety_Complaints.csv', 
              low_memory=False)
incident_df = pd.read_csv(path + '\\' + 'matched_Fire_Incidents.csv', 
              low_memory=False)

#Functions
def topcom(data, strvar):
    t = pd.DataFrame(data.groupby(strvar)[strvar].count().sort_values(ascending=False))
    t.rename(columns = {strvar: 'Count'}, inplace=True)
    t[strvar] = t.index
    return t

def my_barplot(xaxis, yaxis, dta, title):
    g1 = sns.barplot(x=xaxis, y=yaxis, data=dta, palette='coolwarm_r')
    g1.set_xticklabels(labels=dta[xaxis], rotation=90)
    g1.set_title(title)
    sns.despine()
    return g1

#Subset complaint data, keep only 2006-2016 to maintain comparability
complaint_df = complaint_df[['Complaint Item Type Description',
               'Received Date',
               'Disposition',
               'Neighborhood  District',  #Note two spaces b/w words
               'Closest Address',
               'EAS']].dropna(subset=['EAS'])  #Drop where EAS nan
           
complaint_df = complaint_df[(complaint_df['Received Date'] >= '2006-01-01') & 
              (complaint_df['Received Date'] <= '2016-12-31')]

               
#Subset fire incident data, keep only 2006-2016 to maintain comparability            
#Also, keep only building fire/cooking fire/trash fire                                    
incident_df = incident_df[['Incident Date',
                           'Closest Address',
                           'Primary Situation',
                           'Neighborhood  District',  #Note two spaces b/w words
                           'Property Use',
                           'EAS']].dropna(subset=['EAS'])  #Drop where EAS nan

incident_df.rename(columns = {'Primary Situation': 'Situation'}, inplace=True)

incident_df = incident_df[(incident_df.Situation.str.contains("111")) |
                          (incident_df.Situation.str.contains("113")) |
                          (incident_df.Situation.str.contains("118")) |
                          (incident_df.Situation.str.contains("150")) |
                          (incident_df.Situation.str.contains("151")) |
                          (incident_df.Situation.str.contains("154"))]

incident_df['Situation'] = incident_df['Situation'].str.replace('111 building fire', '111 - building fire')
incident_df['Situation'] = incident_df['Situation'].str.replace('113 cooking fire, confined to container', '113 - cooking fire, confined to container')
incident_df['Situation'] = incident_df['Situation'].str.replace('118 trash or rubbish fire, contained', '118 - trash or rubbish fire, contained')
incident_df['Situation'] = incident_df['Situation'].str.replace('150 outside rubbish fire, other', '150 - outside rubbish fire, other')
incident_df['Situation'] = incident_df['Situation'].str.replace('151 outside rubbish, trash or waste fire', '151 - outside rubbish, trash or waste fire')
incident_df['Situation'] = incident_df['Situation'].str.replace('154 dumpster or other outside trash receptacle fire', '154 - dumpster/outside trash receptacle fire')
                          
incident_df = incident_df[(incident_df['Incident Date'] >= '2006-01-01') & 
              (incident_df['Incident Date'] <= '2016-12-31')]


# ## Types of complaints and fire incidents
# 

#Types of complaints
temp = topcom(complaint_df, 'Complaint Item Type Description')
my_barplot('Complaint Item Type Description', 'Count', temp , 'Types of Complaints, 2006-2016')


# The most prevelant type of complaint in the data is "uncategorized", which is not very helpful.  However, other prevelant complaints such as alarm systems (not functioning?) and combustible materials would appear to pose a fire risk.  
# 

#Dispositions
temp = topcom(complaint_df, 'Disposition')
my_barplot('Disposition', 'Count', temp, 'Types of Complaint Dispositions, 2006-2016')


# One aspect of the complaints data to consider (if we end up using this) is the "dispositions" variable.  For example, would we want to include complaints that have "no merit" in our model?  (Probably not...)
# 

#Types of fire incidents (remember we removed a few of these)
temp = topcom(incident_df, 'Situation')
my_barplot('Situation', 'Count', temp, 'Types of Fire Incidents, 2006-2016')


# ## Relationship between complaints and fire incidents (at neighborhood level)
# Aggregating the data at the neighborhood level would imply that there is a strong positive relationship between the number of complaints between 2006-2016 and the total number of (subset) fire incidents. However, there could be a number of collinear factors.  For example, complaints and incidents could both be correlated with neighborhood population, density, or per capita income.  Lets see what happens when we try to analyze the relationship at the EAS level...
# 

#Top complaint districts
t1 = topcom(complaint_df, 'Neighborhood  District')
t1.rename(columns = {'Count': 'Complaint Count'}, inplace=True)
my_barplot('Neighborhood  District', 'Complaint Count', t1, 'Complaints by Neighborhood, 2006-2016')


#Top incident districts
t2 = topcom(incident_df, 'Neighborhood  District')
t2.rename(columns = {'Count': 'Incident Count'}, inplace=True)
my_barplot('Neighborhood  District', 'Incident Count', t2, 'Fire Incidents by Neighborhood, 2006-2016')


#Relationship b/w incidents and complaints
mrg_dta = pd.merge(t1, t2, on='Neighborhood  District')
sns.jointplot(x='Incident Count',y='Complaint Count',data=mrg_dta,kind='reg')


# ## Relationship between incidents and complaints at EAS level
# Instead of grouping incidents/complaints at the neighborhood level, I create another dataset that pairs incidents to complaints (b/w 2006-2016) at the EAS level.  I also create simple "incident_dummy" and "complaint_dummy" variables to represent whether an EAS has had one (or more) incidents/complaints in the last 10 years.
# 
# The takeaway here appears to be the opposite of what we saw in the neighborhood-level data.  There is actually a fairly strong negative correlation between complaints and incidents (especially when looking at dummy var correlation).
# 
# I also run a simple probit model with incident_dummy as the dependent var and complaint count on the RHS.  The resulting coefficient on complaint counts is negative and significant, implying a diminishing probability of a fire incident if a complaint is issued.  Likely, the reasoning is that fire complaints are preemptive and actually prevent fires from occuring! 
# 

#Now, does this relationship hold at EAS level?  Lets see...
t3 = topcom(incident_df, 'EAS')
t3.rename(columns = {'Count': 'incident_count'}, inplace=True)
t3['incident_dummy'] = 1

t4 = topcom(complaint_df, 'EAS')
t4.rename(columns = {'Count': 'complaint_count'}, inplace=True)
t4['complaint_dummy'] = 1

mrg_dta2 = pd.merge(t3, t4, on='EAS', how='outer')
mrg_dta2 = mrg_dta2.fillna(0)
mrg_dta2.corr()


#Probit model fire incident dummy on number of complaints
f = 'incident_dummy ~ complaint_count'
incident_dummy, X = patsy.dmatrices(f, mrg_dta2, return_type='dataframe')
sm.Probit(incident_dummy, X).fit().summary()


# # Generate Fire Incident Features
# 

# ### Introduction
# 

# This notebook documents the process of generating feature data from the file matched_Fire_Incidents.csv.  These features will be used as the target variables in modeling.
# 

# ### Load libraries and CSV
# 

import numpy as np
import pandas as pd
from datetime import datetime

path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'
incident_df = pd.read_csv(path + '\\' + 'matched_Fire_Incidents.csv', 
              low_memory=False)[['Incident Date','Primary Situation','EAS']].dropna()  #Drop obs  where any variable NAN

incident_df['Incident Date'] = pd.to_datetime(incident_df['Incident Date'])
incident_df['Incident_Year'] = incident_df['Incident Date'].dt.year


incident_df.head()


# ### Create larger incident category groupings 
# 

incident_df['code'] = incident_df['Primary Situation'].apply(lambda s: s[0:3])
pd.set_option("display.max_rows",999)
incident_df.groupby(['Primary Situation', 'code']).count()


# A bit difficult to make clear groupings, but I see the following:
# 
# 1, 10, 11, 100, 112  "OTHER FIRE"  
# 111                  "BUILDING FIRE"  
# 113                  "COOKING FIRE"  
# 114-118              "TRASH FIRE (INDOOR)"  
# 120-138              "VEHICLE FIRE"  
# 140-173              "OUTDOOR FIRE"  
# 
# Therefore, I implement as such.
# 

di = {'FIRE OTHER': ['1 -', '10', '100', '11', '112'], 
      'BUILDING FIRE': ['111'], 
      'COOKING FIRE': ['113'], 
      'TRASH FIRE (INDOOR)': ['114','115','116','117','118'],
      'VEHICLE FIRE': ['120', '121', '122', '123', '130', '131', '132', '133', '134', '135', '136', '137', '138'],
      'OUTDOOR FIRE': ['140', '141', '142', '143', '150', '151', '152', '153', '154', '155', '160', '161', '162', '163', '164', '170', '173']}
# reverse the mapping
di = {d:c for c, d_list in di.items()
        for d in d_list}
#Map to 'Incident_Cat' groupings var
incident_df['Incident_Cat'] = incident_df['code'].map(di)


incident_df['Incident_Cat'].value_counts()


# ### Clean up and save data
# 

incident_df['Incident_Dummy'] = 1
incident_df = incident_df[['Incident Date', 
                           'EAS', 
                           'Incident_Year', 
                           'Incident_Cat', 
                           'Incident_Dummy']] 


incident_df.head()


#Export data
incident_df.to_csv(path_or_buf= path + '\\' + 'fireincident_data_formerge_20170917.csv', index=False)





