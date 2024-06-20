# `ApJdataFrames` 002: Luhman2004b
# ---
# `Title`: The First Discovery of a Wide Binary Brown Dwarf
# `Authors`: Luhman K.L.
# 
# Data is from this paper:
# http://iopscience.iop.org/0004-637X/614/1/398
# 

import warnings
warnings.filterwarnings("ignore")


# ## Table 1- Composite photometry
# 

# Since this table is formatted in the `CDS` format, it is easiest to take advantage of the `Astropy` functionality, `ascii.read`, which automatically detects the header formatting.  We will also take advantage of the fact that `ascii.read` can accept a url as an argument, so we won't have to actually save the data locally.  Let's hope ApJ doesn't change their web link though.
# 

import pandas as pd


names = ["component", "RA", "Dec", "Spectral Type", "Teff", "AJ", "Lbol", "R-I","I", "J-H","H-Ks", "Ks", "Mass"]
tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/614/1/398/fulltext/60660.tb1.txt", sep='\t', names=names)
tbl1


# **The end.**
# 

# # `ApJdataFrames` 
# Devor et al. 2008
# ---
# `Title`: IDENTIFICATION, CLASSIFICATIONS, AND ABSOLUTE PROPERTIES OF 773 ECLIPSING BINARIES FOUND IN THE TRANS-ATLANTIC EXOPLANET SURVEY  
# `Authors`: Jonathan Devor, David Charbonneau, Francis T O'Donovan, Georgi Mandushev, and Guillermo Torres  
# 
# 
# Data is from this paper:  
# http://iopscience.iop.org/article/10.1088/0004-6256/135/3/850/
# 

import pandas as pd


from astropy.io import ascii, votable, misc


# ### Download Data
# 

#! mkdir ../data/Devor2008


#! curl http://iopscience.iop.org/1538-3881/135/3/850/suppdata/aj259648_mrt7.txt >> ../data/Devor2008/aj259648_mrt7.txt


get_ipython().system(' du -hs ../data/Devor2008/aj259648_mrt7.txt')


# Not too big at all.
# 

# ### Data wrangle-- read in the data
# 

dat = ascii.read('../data/Devor2008/aj259648_mrt7.txt')


get_ipython().system(' head ../data/Devor2008/aj259648_mrt7.txt')


dat.info


df = dat.to_pandas()


df.head()


df.columns


sns.distplot(df.Per, norm_hist=False, kde=False)


# ## Look for LkCa 4
# 

gi = (df.RAh == 4) & (df.RAm == 16) & (df.DEd == 28) & (df.DEm == 7)


gi.sum()


df[gi].T


# The source is named `T-Tau0-01262`
# 

# ## Get the raw lightcurve
# http://jdevor.droppages.com/Catalog.html
# 

# >The light curve files have the following 3-column format:  
# Column 1 - the Heliocentric Julian date (HJD), minus 2400000  
# Column 2 - normalized r-band magnitude  
# Column 3 - magnitude uncertainty  
# 

get_ipython().system(' head ../data/Devor2008/T-Tau0-01262.lc')


cols = ['HJD-2400000', 'r_band', 'r_unc']
lc_raw = pd.read_csv('../data/Devor2008/T-Tau0-01262.lc', names=cols, delim_whitespace=True)


lc_raw.head()


lc_raw.count()


sns.set_context('talk')


plt.plot(lc_raw['HJD-2400000'], lc_raw.r_band, '.')
plt.ylim(0.6, -0.6)


plt.plot(np.mod(lc_raw['HJD-2400000'], 3.375)/3.375, lc_raw.r_band, '.', alpha=0.5)
plt.xlabel('phase')
plt.ylabel('$\Delta \;\; r$')
plt.ylim(0.6, -0.6)


plt.plot(np.mod(lc_raw['HJD-2400000'], 6.74215), lc_raw.r_band, '.')
plt.ylim(0.6, -0.6)


# The Devor et al. period is just twice the photometric period of 3.375 days.  
# Are those large vertical drops flares?

get_ipython().system(' ls /Users/gully/Downloads/catalog/T-Tau0-* | head -n 10')


lc2 = pd.read_csv('/Users/gully/Downloads/catalog/T-Tau0-00397.lc', names=cols, delim_whitespace=True)
plt.plot(lc2['HJD-2400000'], lc2.r_band, '.')
plt.ylim(0.6, -0.6)


this_p = df.Per[df.Name == 'T-Tau0-00397']
plt.plot(np.mod(lc2['HJD-2400000'], this_p), lc2.r_band, '.', alpha=0.5)
plt.xlabel('phase')
plt.ylabel('$\Delta \;\; r$')
plt.ylim(0.6, -0.6)





# `ApJdataFrames` Malo et al. 2014
# ---
# `Title`: BANYAN. III. Radial velocity, Rotation and X-ray emission of low-mass star candidates in nearby young kinematic groups  
# `Authors`: Malo L., Artigau E., Doyon R., Lafreniere D., Albert L., Gagne J.
# 
# Data is from this paper:  
# http://iopscience.iop.org/article/10.1088/0004-637X/722/1/311/
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 1 - Target Information for Ophiuchus Sources
# 

#! mkdir ../data/Malo2014
#! wget http://iopscience.iop.org/0004-637X/788/1/81/suppdata/apj494919t7_mrt.txt


get_ipython().system(' head ../data/Malo2014/apj494919t7_mrt.txt')


from astropy.table import Table, Column


t1 = Table.read("../data/Malo2014/apj494919t7_mrt.txt", format='ascii')  


sns.distplot(t1['Jmag'].data.data)


t1





# The end.
# 

# `ApJdataFrames` Cottaar_2014
# ---
# `Title`: IN-SYNC I: Homogeneous Stellar Parameters from High-resolution APOGEE Spectra for Thousands of Pre-main Sequence Stars  
# `Authors`: Michiel Cottaar, Kevin R Covey, Michael R Meyer, David L Nidever, Keivan G. Stassun, Jonathan B Foster, Jonathan C Tan, S Drew Chojnowski, Nicola da Rio, Kevin M Flaherty, Peter M Frinchaboy, Michael Skrutskie, Steven R Majewski, John C Wilson, and Gail Zasowski  
# 
# Data is from this paper:  
# http://iopscience.iop.org/article/10.1088/0004-637X/794/2/125/meta
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.max_columns = 150


import astropy


from astropy.io import fits
hdulist1 = fits.open('../data/Cottaar2014/per_epoch.fit')
hdulist2 = fits.open('../data/Cottaar2014/per_star.fit')


table1 = hdulist1[1]


table1.columns


table1.data.shape


table1.size


table1


from astropy.table import Table


tt = Table(data=table1.data)

tt.write('../data/Cottaar2014/per_epoch.csv', format='csv')

dat = pd.read_csv('../data/Cottaar2014/per_epoch.csv')


dat.head()


dat.columns


sns.set_context('talk', font_scale=1.0)


get_ipython().magic("config InlineBackend.figure_format = 'svg'")


dat.columns


cmap = sns.cubehelix_palette(light=1, as_cmap=True)


plt.figure(figsize=[10, 6])
sc = plt.scatter(dat['Teff'], dat['log(g)'], c=dat['R_H'], vmin=0, vmax=2, s=35, cmap=cmap, alpha=0.5)
plt.colorbar(sc)
plt.xlabel('$T_{eff}$')
plt.ylabel('$\log{g}$')
plt.title('Cottaar et al. 2014 APOGEE/INSYNC data')
plt.xlim(7000, 2500)
plt.ylim(5.5, 2.7)


import numpy as np


plt.figure(figsize=[6, 4])
plt.hist(dat['R_H'], bins=np.arange(0, 3, 0.15));
plt.xlabel('$R-H$')
plt.ylabel('$N$')


# Find a source with specific properties.
# 

Tfi = (dat.Teff > 4000) & (dat.Teff < 4200) 
lgi = (dat['log(g)'] > 3.5) & (dat['log(g)'] < 4.0)

gi = Tfi & lgi


dat.shape


gi.sum()


dat['2MASS'][gi]


# # Per star observations (this is actually what we wanted).
# 

table2 = hdulist2[1]


table2.columns


table2.data.shape


t2 = Table(data=table2.data)

t2.write('../data/Cottaar2014/per_star.csv', format='csv')

data = pd.read_csv('../data/Cottaar2014/per_star.csv')


data.head()


data.Cluster.unique()


data.columns


ic = data.Cluster == 'IC 348'
pl = data.Cluster == 'Pleiades'


# Read in the Chabrier and Baraffe models
# 

bcah = pd.read_csv('../data/BCAH2002/BCAH2002_isochrones.csv', sep = '\t')
groups =bcah.groupby(by='Age')





plt.figure(figsize=[10, 6])
plt.scatter(data['Teff'][ic], data['log(g)'][ic], label='IC 348', c='r')
plt.scatter(data['Teff'][pl], data['log(g)'][pl], label='Pleiades')
plt.xlabel('$T_{eff}$')
plt.ylabel('$\log{g}$')
plt.title('Cottaar et al. 2014 APOGEE/INSYNC data')
plt.legend(loc='best')

for age, group in groups:
    no_decimal = np.abs(np.mod(age, 1)) <0.001
    if no_decimal:
        plt.plot(group.Teff, group.logg, 'k-', alpha=0.5, label='{:0.1f} Myr'.format(age))

plt.xlim(7000, 2500)
plt.ylim(5.5, 2.7)


data.columns


data.shape


sns.distplot(data['R_H'][ic], hist=False, label='IC 348')
sns.distplot(data['R_H'][pl], hist=False, label='Pleiades')





# `ApJdataFrames` Erickson2011
# ---
# `Title`: THE INITIAL MASS FUNCTION AND DISK FREQUENCY OF THE Rho OPHIUCHI CLOUD: AN EXTINCTION-LIMITED SAMPLE 
# `Authors`: Erickson et al.
# 
# Data is from this paper:  
# http://iopscience.iop.org/1538-3881/142/4/140/
# 

get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context("notebook", font_scale=1.5)

#import warnings
#warnings.filterwarnings("ignore")


import pandas as pd


# ## Table 2- Optical Properties of Candidate Young Stellar Objects
# 

addr = "http://iopscience.iop.org/1538-3881/142/4/140/suppdata/aj403656t2_ascii.txt"
names = ['F', 'Ap', 'Alt_Names', 'X-Ray ID', 'RA', 'DEC', 'Li', 'EW_Ha', 'I', 'R-I',
       'SpT_Lit', 'Spectral_Type', 'Adopt', 'Notes', 'blank']
tbl2 = pd.read_csv(addr, sep='\t', skiprows=[0,1,2,3,4], skipfooter=7, engine='python', na_values=" ... ", 
                   index_col=False, names = names, usecols=range(len(names)-1))
tbl2.head()


# ## Table 3 - Association Members with Optical Spectra
# 

addr = "http://iopscience.iop.org/1538-3881/142/4/140/suppdata/aj403656t3_ascii.txt"
names = ['F', 'Ap', 'Alt_Names', 'WMR', 'Spectral_Type', 'A_v', 'M_I',
       'log_T_eff', 'log_L_bol', 'Mass', 'log_age', 'Criteria', 'Notes', 'blank']
tbl3 = pd.read_csv(addr, sep='\t', skiprows=[0,1,2,3,4], skipfooter=9, engine='python', na_values=" ... ", 
                   index_col=False, names = names, usecols=range(len(names)-1))
tbl3.head()


get_ipython().system(' mkdir ../data/Erickson2011')


# ###The code to merge the tables isn't working
# 

# ```python
# on_F_ap = ["F", "Ap"]
# on_name = "Alt_Names"
# erickson2011 = pd.merge(tbl2, tbl3, on=on_F_ap, how="right")
# erickson2011 = pd.merge(tbl2, erickson2011, on="Alt_Names", how="right")
# message = "Table 2: {} entries \nTable 3: {} entries \nMerge: {} entries"
# print message.format(len(tbl2), len(tbl3), len(erickson2011))
# ```
# 

plt.plot(10**tbl3.log_T_eff, 10**tbl3.log_L_bol, '.')
plt.yscale("log")
plt.xlim(5000, 2000)
plt.ylim(1.0E-4, 1.0E1)
plt.xlabel(r"$T_{eff}$")
plt.ylabel(r"$L/L_{sun}$")
plt.title("Erickson et al. 2011 Table 3 HR Diagram")


# Another thing to do would be to filter out the "Possible dwarfs", etc...  
# Save the data tables locally.
# 

tbl2.to_csv("../data/Erickson2011/tbl2.csv", sep="\t", index=False)
tbl3.to_csv("../data/Erickson2011/tbl3.csv", sep="\t", index=False)


# *Script finished.*
# 

# `ApJdataFrames` 009: Luhman2009
# ---
# `Title`: An Infrared/X-Ray Survey for New Members of the Taurus Star-Forming Region  
# `Authors`: Kevin L Luhman, E. E. Mamajek, P R Allen, and Kelle L Cruz
# 
# Data is from this paper:  
# http://iopscience.iop.org/0004-637X/703/1/399/article#apj319072t2
# 

get_ipython().magic('pylab inline')
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


import pandas as pd


# ## Table 2- Members of Taurus in Spectroscopic Sample
# 

tbl2 = pd.read_csv("http://iopscience.iop.org/0004-637X/703/1/399/suppdata/apj319072t2_ascii.txt", 
                   nrows=43, sep='\t', skiprows=2, na_values=[" sdotsdotsdot"])
tbl2.drop("Unnamed: 10",axis=1, inplace=True)


# Clean the column names.
# 

new_names = ['2MASS', 'Other_Names', 'Spectral_Type', 'T_eff', 'A_J','L_bol','Membership',
       'EW_Halpha', 'Basis of Selection', 'Night']
old_names = tbl2.columns.values
tbl2.rename(columns=dict(zip(old_names, new_names)), inplace=True)


tbl2.head()


sns.set_context("notebook", font_scale=1.5)


plt.plot(tbl2.T_eff, tbl2.L_bol, '.')
plt.ylabel(r"$L/L_{sun}$")
plt.xlabel(r"$T_{eff} (K)$")
plt.yscale("log")
plt.title("Luhman et al. 2009 Taurus Members")
plt.xlim(5000,2000)


# # Save the data tables locally.
# 

get_ipython().system(' mkdir ../data/Luhman2009')


tbl2.to_csv("../data/Luhman2009/tbl2.csv", sep="\t")


# **The end.**
# 

# `ApJdataFrames` Wilking2005
# ---
# `Title`: Optical Spectroscopy of the Surface Population of the ρ Ophiuchi Molecular Cloud: The First Wave of Star Formation  
# `Authors`: Wilking et al.
# 
# Data is from this paper:  
# http://iopscience.iop.org/1538-3881/130/4/1733/fulltext/
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 2 - Optical Properties of Candidate T Tauri Stars
# 

tbl2_vo = ascii.read("http://iopscience.iop.org/1538-3881/130/4/1733/fulltext/datafile2.txt")
tbl2_vo[0:3]


# ##Table 4- Association Members
# 

tbl4_vo = ascii.read("http://iopscience.iop.org/1538-3881/130/4/1733/fulltext/datafile4.txt")
tbl4_vo[0:4]


# ##Join the tables
# 

tbl2 = tbl2_vo.to_pandas()
del tbl2["Name"]
tbl2.rename(columns={'Note':"Flag"}, inplace=True)
tbl4 = tbl4_vo.to_pandas()


wilking2005 = pd.merge(tbl2, tbl4, how="right", on=["Field", "Aper"])


wilking2005


wilking2005["RA"] = wilking2005.RAh.astype(str) + wilking2005.RAm.astype(str) + wilking2005.RAs.astype(str)


wilking2005.RA


# ## Save the data
# 

get_ipython().system(' mkdir ../data/Wilking2005')


wilking2005.to_csv("../data/Wilking2005/Wilking2005.csv", index=False, sep='\t')


# *The end*
# 

# `ApJdataFrames` 008: Luhman2012
# ---
# `Title`: THE DISK POPULATION OF THE UPPER SCORPIUS ASSOCIATION  
# `Authors`: K. L. Luhman and E. E. Mamajek  
# 
# Data is from this paper:  
# http://iopscience.iop.org/0004-637X/758/1/31/article#apj443828t1
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 1 - VOTable with all source properties
# 

tbl1 = ascii.read("http://iopscience.iop.org/0004-637X/758/1/31/suppdata/apj443828t1_mrt.txt")


tbl1.columns


tbl1[0:5]


len(tbl1)


# #Cross match with SIMBAD
# 

from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u


customSimbad = Simbad()
customSimbad.add_votable_fields('otype', 'sptype')


query_list = tbl1["Name"].data.data
result = customSimbad.query_objects(query_list, verbose=True)


result[0:3]


print "There were {} sources queried, and {} sources found.".format(len(query_list), len(result))
if len(query_list) == len(result):
    print "Hooray!  Everything matched"
else:
    print "Which ones were not found?"


def add_input_column_to_simbad_result(self, input_list, verbose=False):
    """
    Adds 'INPUT' column to the result of a Simbad query

    Parameters
    ----------
    object_names : sequence of strs
            names of objects from most recent query
    verbose : boolean, optional
        When `True`, verbose output is printed

    Returns
    -------
    table : `~astropy.table.Table`
        Query results table
    """
    error_string = self.last_parsed_result.error_raw
    fails = []

    for error in error_string.split("\n"):
        start_loc = error.rfind(":")+2
        fail = error[start_loc:]
        fails.append(fail)

    successes = [s for s in input_list if s not in fails]
    if verbose:
        out_message = "There were {} successful Simbad matches and {} failures."
        print out_message.format(len(successes), len(fails))

    self.last_parsed_result.table["INPUT"] = successes

    return self.last_parsed_result.table


result_fix = add_input_column_to_simbad_result(customSimbad, query_list, verbose=True)


tbl1_pd = tbl1.to_pandas()
result_pd = result_fix.to_pandas()
tbl1_plusSimbad = pd.merge(tbl1_pd, result_pd, how="left", left_on="Name", right_on="INPUT")


# # Save the data table locally.
# 

tbl1_plusSimbad.head()


get_ipython().system(' mkdir ../data/Luhman2012/')


tbl1_plusSimbad.to_csv("../data/Luhman2012/tbl1_plusSimbad.csv", index=False)


# *The end*
# 

# `ApJdataFrames` Hernandez2014
# ---
# `Title`: A SPECTROSCOPIC CENSUS IN YOUNG STELLAR REGIONS: THE σ ORIONIS CLUSTER  
# `Authors`: Jesus Hernandez, Nuria Calvet, Alice Perez, Cesar Briceno, Lorenzo Olguin, Maria E Contreras, Lee Hartmann, Lori E Allen, Catherine Espaillat, and Ramírez Hernan  
# 
# Data is from this paper:  
# http://iopscience.iop.org/0004-637X/794/1/36/article
# 

get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context("notebook", font_scale=1.5)

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


# ## Table 4 -  Low Resolution Analysis
# 

tbl4 = ascii.read("http://iopscience.iop.org/0004-637X/794/1/36/suppdata/apj500669t4_mrt.txt")


tbl4[0:4]


Na_mask = ((tbl4["f_EWNaI"] == "Y") | (tbl4["f_EWNaI"] == "N"))
print "There are {} sources with Na I line detections out of {} sources in the catalog".format(Na_mask.sum(), len(tbl4))


tbl4_late = tbl4[['Name', '2MASS', 'SpType', 'e_SpType','EWHa', 'f_EWHa', 'EWNaI', 'e_EWNaI', 'f_EWNaI']][Na_mask]


tbl4_late.pprint(max_lines=100, )


# ### Meh... not a lot of late type sources... M5.5 is the latest.  Oh well.
# 

# *Script finished.*
# 

# `ApJdataFrames` 007: Patten2006
# ---
# `Title`: Spitzer IRAC Photometry of M, L, and T Dwarfs  
# `Authors`: Brian M Patten, John R Stauffer, Adam S Burrows, Massimo Marengo, Joseph L Hora, Kevin L Luhman, Sarah M Sonnett, Todd J Henry, Deepak Raghavan, S Thomas Megeath, James Liebert, and Giovanni G Fazio  
# 
# Data is from this paper:  
# 

get_ipython().magic('pylab inline')
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


import pandas as pd


# The tables define the value and error as a string:  
# `val (err)`   
# which is a pain in the ass because now I have to parse the strings, which always takes much longer than it should because data wrangling is hard sometimes.
# 
# I define a function that takes a column name and a data frame and strips the output.
# 

def strip_parentheses(col, df):
    '''
    splits single column strings of "value (error)" into two columns of value and error
    
    input:
    -string name of column to split in two
    -dataframe to apply to
    
    returns dataframe
    '''
    
    out1 = df[col].str.replace(")","").str.split(pat="(")
    df_out = out1.apply(pd.Series)
    
    # Split the string on the whitespace 
    base, sufx =  col.split(" ")
    df[base] = df_out[0].copy()
    df[base+"_e"] = df_out[1].copy()
    del df[col]
    
    return df
    


# ## Table 1 - Basic data on sources
# 

names = ["Name","R.A. (J2000.0)","Decl. (J2000.0)","Spectral Type","SpectralType Ref.","Parallax (error)(arcsec)",
         "Parallax Ref.","J (error)","H (error)","Ks (error)","JHKRef.","PhotSys"]

tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/651/1/502/fulltext/64991.tb1.txt", 
                   sep='\t', names=names, na_values='\ldots')


cols_to_fix = [col for col in tbl1.columns.values if "(error)" in col]
for col in cols_to_fix:
    print col
    tbl1 = strip_parentheses(col, tbl1)


tbl1.head()


# ## Table 3- IRAC photometry
# 

names = ["Name","Spectral Type","[3.6] (error)","n1","[4.5] (error)","n2",
         "[5.8] (error)","n3","[8.0] (error)","n4","[3.6]-[4.5]","[4.5]-[5.8]","[5.8]-[8.0]","Notes"]

tbl3 = pd.read_csv("http://iopscience.iop.org/0004-637X/651/1/502/fulltext/64991.tb3.txt", 
                   sep='\t', names=names, na_values='\ldots')


cols_to_fix = [col for col in tbl3.columns.values if "(error)" in col]
cols_to_fix
for col in cols_to_fix:
    print col
    tbl3 = strip_parentheses(col, tbl3)


tbl3.head()


pd.options.display.max_columns = 50


del tbl3["Spectral Type"] #This is repeated


patten2006 = pd.merge(tbl1, tbl3, how="outer", on="Name")
patten2006.head()


# Convert spectral type to number
# 

import gully_custom


patten2006["SpT_num"], _1, _2, _3= gully_custom.specTypePlus(patten2006["Spectral Type"])


# Make a plot of mid-IR colors as a function of spectral type.
# 

sns.set_context("notebook", font_scale=1.5)


for color in ["[3.6]-[4.5]", "[4.5]-[5.8]", "[5.8]-[8.0]"]:
    plt.plot(patten2006["SpT_num"], patten2006[color], '.', label=color)
    
plt.xlabel(r'Spectral Type (M0 = 0)')
plt.ylabel(r'$[3.6]-[4.5]$')
plt.title("IRAC colors as a function of spectral type")
plt.legend(loc='best')


# ## Save the cleaned data.
# 

patten2006.to_csv('../data/Patten2006/patten2006.csv', index=False)


# `ApJdataFrames` Liu et al. 2010
# ---
# `Title`: Discovery of a Highly Unequal-mass Binary T Dwarf with Keck Laser Guide Star Adaptive Optics: A Coevality Test of Substellar Theoretical Models and Effective Temperatures  
# `Authors`: Liu, Dupuy, Leggett
# 
# Data is from this paper:  
# http://iopscience.iop.org/article/10.1088/0004-637X/722/1/311/
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 1 - Target Information for Ophiuchus Sources
# 

#! mkdir ../data/Liu2010
#! curl http://iopscience.iop.org/0004-637X/722/1/311/suppdata/apj336343t6_ascii.txt > ../data/Liu2010/apj336343t6_ascii.txt


tbl6 = pd.read_csv("../data/Liu2010/apj336343t6_ascii.txt",
                   sep="\t", na_values=" ... ", skiprows=[0,1,2], skipfooter=1, usecols=range(9))
tbl6.head()


# We want the J-Band coefficients, **in order of highest order coefficient to lowest order coefficient**.
# 

J_s = tbl6.loc[0]


coeffs = J_s[["c_"+str(i) for i in range(6, -1, -1)]].values


func = np.poly1d(coeffs)


print(func)


# *The end*
# 

# `ApJdataFrames` Rayner et al. 2009
# ---
# `Title`: THE INFRARED TELESCOPE FACILITY (IRTF) SPECTRAL LIBRARY: COOL STARS  
# `Authors`: John T. Rayner, Michael C. Cushing, and William D. Vacca
# 
# Data is from this paper:  
# http://iopscience.iop.org/article/10.1088/0067-0049/185/2/289/meta
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 7 - Strong metal lines in the Arcturus spectrum
# 

#! curl http://iopscience.iop.org/0067-0049/185/2/289/suppdata/apjs311476t7_ascii.txt > ../data/Rayner2009/apjs311476t7_ascii.txt


#! head ../data/Rayner2009/apjs311476t7_ascii.txt


nn = ['wl1', 'id1', 'wl2', 'id2', 'wl3', 'id3', 'wl4', 'id4']


tbl7 = pd.read_csv("../data/Rayner2009/apjs311476t7_ascii.txt", index_col=False,
                   sep="\t", skiprows=[0,1,2,3], names= nn)


# This is a verbose way to do this, but whatever, it works:
# 

line_list_unsorted = pd.concat([tbl7[[nn[0], nn[1]]].rename(columns={"wl1":"wl", "id1":"id"}),
           tbl7[[nn[2], nn[3]]].rename(columns={"wl2":"wl", "id2":"id"}),
           tbl7[[nn[4], nn[5]]].rename(columns={"wl3":"wl", "id3":"id"}),
           tbl7[[nn[6], nn[7]]].rename(columns={"wl4":"wl", "id4":"id"})], ignore_index=True, axis=0)


line_list = line_list_unsorted.sort_values('wl').dropna().reset_index(drop=True)


# Finally:
# 

#line_list.tail()


sns.distplot(line_list.wl)


# The lines drop off towards $K-$band.
# 

# Save the file:
# 

line_list.to_csv('../data/Rayner2009/tbl7_clean.csv', index=False)


# ## *The end*
# 

# # `ApJdataFrames` 
# Shetrone et al. 2015  
# ---
# `Title`: THE SDSS-III APOGEE SPECTRAL LINE LIST FOR H-BAND SPECTROSCOPY  
# `Authors`: M Shetrone, D Bizyaev, J E Lawler, C Allende Prieto, J A Johnson, V V Smith, K Cunha, J. Holtzman, A E García Pérez, Sz Mészáros, J Sobeck, O Zamora, D A Garcia Hernandez, D Souto, D Chojnowski, L Koesterke, S Majewski, and G Zasowski
# 
# Data is from this paper:  
# http://iopscience.iop.org/0067-0049/221/2/24/
# 
# 

import pandas as pd


from astropy.io import ascii, votable, misc


# ### Download Data
# 

#! mkdir ../data/Shetrone2015
#! wget http://iopscience.iop.org/0067-0049/221/2/24/suppdata/apjs521087t7_mrt.txt
#! mv apjs521087t7_mrt.txt ../data/Shetrone2015/
#! du -hs ../data/Shetrone2015/apjs521087t7_mrt.txt


# The file is about 24 MB.
# 

# ### Data wrangle-- read in the data
# 

dat = ascii.read('../data/Shetrone2015/apjs521087t7_mrt.txt')


get_ipython().system(' head ../data/Shetrone2015/apjs521087t7_mrt.txt')


dat.info


df = dat.to_pandas()


df.head()


df.columns


sns.distplot(df.Wave, norm_hist=False, kde=False)


df.count()


sns.lmplot('orggf', 'newgf', df, fit_reg=False)


from astropy import units as u


u.cm


EP1 = df.EP1.values*1.0/u.cm
EP2 = df.EP2.values*1.0/u.cm


EP1_eV = EP1.to(u.eV, equivalencies=u.equivalencies.spectral())
EP2_eV = EP2.to(u.eV, equivalencies=u.equivalencies.spectral())


deV = EP1_eV - EP2_eV


sns.distplot(deV)


plt.plot(df.Wave, deV, '.', alpha=0.05)
plt.xlabel('$\lambda (\AA)$')
plt.ylabel('$\Delta E \;(\mathrm{eV})$')


# There are finite differences between the difference in the energy levels and the emitted wavelength based on other properties of the transition.
# 

# ## The end.
# 

# `ApJdataFrames` 004: Hartmann2005
# ---
# `Title`: IRAC Observations of Taurus Pre–Main-Sequence Stars  
# `Authors`: Lee Hartmann, S. T. Megeath, Lori Allen, Kevin Luhman, Nuria Calvet, Paola D'Alessio, Ramiro Franco-Hernandez, and Giovanni Fazio
# 
# Data is from this paper:
# http://iopscience.iop.org/0004-637X/629/2/881
# 

import warnings
warnings.filterwarnings("ignore")


# ## Table 1- IRAC Photometry
# 

import pandas as pd


names = ["Name","2MASS ID","R.A.(J2000.0)(deg)","Decl. (J2000.0)(deg)",
         "J(mag)","H(mag)","Ks(mag)","[3.6](mag)","[4.5](mag)",
         "[5.8](mag)","[8](mag)","JD-53000","IRAC Type"]
tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/629/2/881/fulltext/61849.tb1.txt", 
                   na_values='\ldots',sep='\t', names=names)
tbl1.head()


# **The end.**
# 

# `ApJdataFrames` Chapman 2009
# ---
# `Title`: THE MID-INFRARED EXTINCTION LAW IN THE OPHIUCHUS, PERSEUS, AND SERPENS MOLECULAR CLOUDS  
# `Authors`: Nicholas L. Chapman, Lee G Mundy, Shih-Ping Lai, and Neal J Evans  
# 
# Data is from this paper:
# http://iopscience.iop.org/0004-637X/690/1/496/
# 

import warnings
warnings.filterwarnings("ignore")


# ## Table 1- Measured Quantities for PMS Candidates with Observed Spectra
# 

import pandas as pd


tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/690/1/496/suppdata/apj291883t3_ascii.txt",
                   skiprows=[0,1,2,4], sep='\t', header=0, na_values=' ... ', skipfooter=6)
del tbl1["Unnamed: 6"]
tbl1


# **The end.**
# 

# `ApJdataFrames` Geers2011
# ---
# `Title`: SUBSTELLAR OBJECTS IN NEARBY YOUNG CLUSTERS (SONYC). II. THE BROWN DWARF POPULATION OF ρ OPHIUCHI  
# `Authors`: Geers et al.
# 
# Data is from this paper:  
# http://iopscience.iop.org/0004-637X/726/1/23/
# 

import warnings
warnings.filterwarnings("ignore")


from astropy.io import ascii


import pandas as pd


# ## Table 2 - Probable Low Mass and Substellar Mass Members of rho Oph, with MOIRCS Spectroscopy Follow-up
# 

names = ["No.","R.A. (J2000)","Decl. (J2000)","i (mag)","J (mag)","K_s (mag)","T_eff (K)","A_V","Notes"]
tbl2 = pd.read_csv("http://iopscience.iop.org/0004-637X/726/1/23/suppdata/apj373191t2_ascii.txt",
                   sep="\t", skiprows=[0,1,2,3], na_values="sat", names = names)
tbl2.dropna(how="all", inplace=True)
tbl2.head()


# ## Save the data
# 

get_ipython().system(' mkdir ../data/Geers2011')


tbl2.to_csv("../data/Geers2011/tb2.csv", index=False, sep='\t')


# *The end*
# 

