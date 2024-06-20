get_ipython().magic('matplotlib inline')
from bs4 import BeautifulSoup
import urllib2
import requests
import pandas as pd
import re
import time
import numpy as np
import json
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import matplotlib.pyplot as plt
from pyquery import PyQuery as pq


# This notebook initiates our work. We scrape the election commission website at myneta.info, focusing first on Lok Sabha 2014, but we will extend it to all years and all state at later stages. We can use this initial page to do all of that. Note that this will be the "clean" programming sheet.
# 

# Our source page: Lok Sabha 2014#
all_cand = "http://myneta.info/ls2014/index.php?action=summary&subAction=candidates_analyzed&sort=candidate#summary"
source=requests.get(all_cand) # Modify here: winners works with smaller sample, all_cand with the whole set
tree= BeautifulSoup(source.text,"html.parser")
table_pol = tree.findAll('table')[2]
rows = table_pol.findAll("tr")[2:]


# We build one dictionary per candidate #
# We give each candidates to idenfiers: its order in the list and its url address

def id2(r):
    url_string = str(r.find("a").get("href"))
    id2 = int(re.search(r'\d+', url_string).group())
    return id2

def c_link(r):
    return r.find("a").get("href")
def name(r):
    return r.find("a").get_text()
def cols(r):
    return r.findAll("td")
def assets(r):
    col = cols(r)
    ass1 = col[6].get_text().split("~")[0].encode('ascii', 'ignore').replace("Rs","").replace(",","")
    if ass1 == "Nil":
        ass2 = 0
    else:
        ass2=int(ass1)
    return ass2

def liab(r):
    col = cols(r)
    liab1 = col[7].get_text().split("~")[0].encode('ascii', 'ignore').replace("Rs","").replace(",","")
    if liab1 == "Nil":
        liab2 = 0
    else:
        liab2 = int(liab1)
    return liab2

info_candidate = lambda r: [int(cols(r)[0].get_text()),id2(r), r.find("a").get("href"),r.find("a").get_text(),
                            cols(r)[2].get_text(),cols(r)[3].get_text(),cols(r)[5].get_text(),
                            int(cols(r)[4].get_text()),assets(r), liab(r)]

title = ['id','id2','url','name','district','party','education','nr_crime','assets','liabilities']
dict_candidates = [dict(zip(title,info_candidate(r))) for r in rows]
print len(dict_candidates)


# Now we create a really big dictionary which stores url and page for each candidate
# Work in progress...
# First transform dict_candidate into a dataframe

df_pol = pd.DataFrame(dict_candidates)
df_pol.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\df_pol_LS2014.csv", index = True)
order_cols = ['id2','name','district','party','education','assets','liabilities','nr_crime','url']
df_pol = df_pol[order_cols].sort(['assets'],ascending=0)


urlcache={}


def get_page(url):
    # Check if URL has already been visited.
    url_error = []
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(1)
        steps = len(urlcache)
        if 100*int(steps/100)==steps:
            print steps # This counter tells us how many links were downloaded at every 100 mark
        # try/except blocks are used whenever the code could generate an exception (e.g. division by zero).
        # In this case we don't know if the page really exists, or even if it does, if we'll be able to reach it.
        try:
            r = requests.get("http://myneta.info/ls2014/%s" % url)
            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
            url_error.append(url)
            print url
    return urlcache[url]


# retry downloading missing pages:
for r in url_error:
    urlcache[r] = requests.get("http://myneta.info/ls2014/%s" % r).text()


#df_pol["url"].apply(get_page) # This is a very long call (~4.5 hours on full dataset)
                              # I am saving it in order to run it only once


print np.sum([(urlcache[k]==1) or (urlcache[k]==2) for k in urlcache])# no one or 0's
print len(df_pol.url.unique())==len(urlcache)#we got all of the urls


with open("tempdata/polinfo.json","w") as fd:
    json.dump(pol_pages, fd)
del urlcache


get_ipython().magic('matplotlib inline')
import itertools
from bs4 import BeautifulSoup
import urllib2
import requests
import pandas as pd
import re
import time
import numpy as np
import json
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import matplotlib.pyplot as plt
from pyquery import PyQuery as pq
dropbox = "C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables"


# #This notebook takes all the candidates that appear in regional elections in 2012-2015 and takes their personal info #
# 

#First load the dataframe
df_fin = pd.read_csv("df_all_regional_elections2.csv")
print df_fin.shape
df_fin.head()
df_fin.columns


def url_tsform(el,c_id):
    url_link = "http://myneta.info/%s/candidate.php?candidate_id=%s" %(el,c_id)
    return url_link
def url_split(url_link):
    return url_link.split("/")[3], url_link.split("=")[1]


url_split("http://myneta.info/mp2013/candidate.php?candidate_id=666")


def get_page(el,c_id):
    # Check if URL has already been visited.
    url = (el,c_id)
    url_error = []
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(.5)
        try:
            r = requests.get(url_tsform(el,c_id))
            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
            url_error.append(url)
            print "error with:", url
    return urlcache[url]


#Re open the dict
with open('tempdata/largedict.json', 'r') as f:
     dict_start = json.load(f)


dict_id = {url_split(k):v for k,v in dict_start.iteritems()}
print len(dict_id)
urlcache = {}


#Creates a giant dictionary
#dict_id2 = {}
steps = len(dict_id2)
print "initial length: ", steps
for row in df_fin.itertuples():
    if (row[19],row[20]) in dict_id2:
        pass
    else:
        dict_id2[(row[19],row[20])] = get_page(row[19],row[20])
        steps = len(dict_id2)
        if steps % 1000 ==0:
            print steps,


dict_clean = {url_tsform(k[0],k[1]):v for k,v in dict_id2.iteritems() if v!=2}
print len(dict_clean), len(dict_id2)


with open('tempdata/largedict.json', 'w') as f:
     json.dump(dict_clean, f)


listerror=[]
for k,v in dict_id2.iteritems():
    if isinstance(v, basestring):
        listerror.append(v)
len(listerror)


page_ex = dict_clean['http://myneta.info/mp2013/candidate.php?candidate_id=258']


bs = BeautifulSoup(page_ex,"html.parser")
bs.findAll("table")[0]


ROOT_LINK = "http://myneta.info"
def get_otherelec_link(page_text, url):
    for a in pq(page_text)("a"):
        if a.text == "Click here for more details":
            other_elec_link = pq(a).attr.href
            return ROOT_LINK+other_elec_link
    return False

test_index = "http://myneta.info/mp2013/candidate.php?candidate_id=258"
page_data = dict_clean[test_index]
other_link = get_otherelec_link(page_data,test_index)


def get_otherelec_data(otherelec_link):
    
    otherelec_dict = {'common_link': otherelec_link}
    
    html = requests.get(otherelec_link)
    doc = pq(html.content)
    
    columns = []
    all_dicts = []
    add = 0
    trs = doc('tr')
    for tr in trs:
        elec_dict = otherelec_dict.copy()
        for th in pq(tr)('th'):
            columns.append(pq(th).text().replace(" ","_"))
            add = 0
        for i,td in enumerate(pq(tr)('td')):
            a = pq(td)('a')
            if a:
                elec_dict['elec_link'] = ROOT_LINK+a.attr.href
                elec_dict[columns[i]] = a.text()
            else:
                try:
                    if pq(td)('span') and i < 6:
                        elec_dict[columns[i]] = pq(td)('span').text()
                    else:
                        elec_dict[columns[i]] = str(pq(td).contents()[0]).encode('utf-8').strip().replace(',','')
                except:
                    print ""
                    print "Skipping col %s for %s" % (columns[i], elec_dict['common_link'])
            add = 1
            
        if add == 1:
            all_dicts.append(elec_dict)
    
    return all_dicts

get_otherelec_data(other_link)


def get_all_elec():
    all_elec_data = []   
    counter = 0.0
    for key, val in dict_clean.iteritems():
        thelink = get_otherelec_link(val,key)
        counter += 1
        if counter%100 == 0.0:
            print ".",
        if thelink:
            all_elec_data = all_elec_data + get_otherelec_data(thelink)
    
    df = pd.DataFrame(all_elec_data)
    return df.drop_duplicates()

all_elecs_df = get_all_elec()
all_elecs_df.head(3)['common_link']


all_elecs_df.shape


all_elecs_df.to_excel("tempdata/all_elecs_statelevel.xls")





get_ipython().magic('matplotlib inline')
from bs4 import BeautifulSoup
import urllib2
import requests
import pandas as pd
import re
import time
import numpy as np
import json
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import matplotlib.pyplot as plt
from pyquery import PyQuery as pq
dropbox = "C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables"


with open(dropbox + "\candidate_pages.json") as json_file:
    pol_pages = json.load(json_file) # Next iterations will start from here!


def find_year(link):
    year = re.findall('\d+', link)
    return int(year[0])


find_year("http://www.myneta.info/andhra2014/index.php?action=affidavitComparison&myneta_folder2=ap09&id1=2150&id2=2415")


# ### First we start with the income data for each candidate, collecting all info in a csv file. ###
# 

cleaner = lambda e: int(re.findall('\d+', e.replace(',', '').split(" ~ ")[0])[0])
income_cols = ["Relation","PAN","Year","Income"]

def income_table(candidate_id):
    page_candidate = pol_pages[candidate_id]
    c_soup = BeautifulSoup(page_candidate,"html.parser")
    table_titles =[x.get_text().strip() for x in c_soup.findAll("h3")]
    tables = [x.find_next() for x in c_soup.findAll("h3")]
    dict_tab = dict(zip(table_titles,tables))
    income_tab = dict_tab['Details of PAN and status of Income Tax return']
    income_rows = income_tab.find_all("tr")
    dict_income = {}
    df_inc = pd.DataFrame([])
    if income_cols==[]:
        dict_income = {'HH':{"Year":np.nan,"PAN":"N","Relation":np.nan,"Income":np.nan}}
    else:
        for r in income_rows[1:]:
            list_items = [x.get_text() for x in r.findAll("td")]
            if len(list_items)==4 and list_items[3]!="Nil":
                list_items[3] = cleaner(list_items[3])
            if len(list_items)==4 and list_items[3]=="Nil":
                list_items[3] = 0
            dict_income[list_items[0]] = dict(zip(income_cols,list_items))
        df_inc = df_inc.from_dict(dict_income,orient = "index")
    try:
        df_inc = df_inc[df_inc.PAN=="Y"]
        HHinc = np.sum(df_inc['Income'])
        HHDeclarations = np.count_nonzero(df_inc['PAN'])
        self_income = dict_income['self']['Income']
        self_declare = dict_income['self']['PAN']
    except AttributeError:
        df_inc=df_inc
        HHinc = np.nan
        HHDeclarations = 0
        self_income = np.nan
        self_declare = np.nan
    newdict = {'self_inc':self_income,'self_declare':self_declare,'HHinc':HHinc,"HHDeclarations":HHDeclarations}
    return newdict


candidate_id = pol_pages.keys()[5]
print candidate_id
page_candidate = pol_pages[candidate_id]
c_soup = BeautifulSoup(page_candidate,"html.parser")
table_titles =[x.get_text().strip() for x in c_soup.findAll("h3")]
tables = [x.find_next() for x in c_soup.findAll("h3")]
dict_tab = dict(zip(table_titles,tables))
dict_tab.keys()
#    income_tab = dict_tab['Details of PAN and status of Income Tax return']
#    income_rows = income_tab.find_all("tr")
#    dict_income = {}
#    df_inc = pd.DataFrame([])
#    if income_cols==[]:
#        dict_income = {'HH':{"Year":np.nan,"PAN":"N","Relation":np.nan,"Income":np.nan}}
#    else:
#        for r in income_rows[1:]:
#            list_items = [x.get_text() for x in r.findAll("td")]
#            if len(list_items)==4 and list_items[3]!="Nil":
#                list_items[3] = cleaner(list_items[3])
#            if len(list_items)==4 and list_items[3]=="Nil":
#                list_items[3] = 0
#            dict_income[list_items[0]] = dict(zip(income_cols,list_items))
#        df_inc = df_inc.from_dict(dict_income,orient = "index")
#    try:
   #     df_inc = df_inc[df_inc.PAN=="Y"]
    #    HHinc = np.sum(df_inc['Income'])
     #   HHDeclarations = np.count_nonzero(df_inc['PAN'])
      #  self_income = dict_income['self']['Income']
       # self_declare = dict_income['self']['PAN']
   # except AttributeError:
    #    df_inc=df_inc
     #   HHinc = np.nan
      #  HHDeclarations = 0
       # self_income = np.nan
        #self_declare = np.nan
    #newdict = {'self_inc':self_income,'self_declare':self_declare,'HHinc':HHinc,"HHDeclarations":HHDeclarations}


get_ipython().run_cell_magic('time', '', 'counterror = 0\ndict_allinc = {}\nif any("Details of PAN and status of Income Tax return" in dict_tab.keys())==False:\n    break\nelse:\n    for k,cid in enumerate(pol_pages.keys()):\n        year = find_year(cid)\n        try:\n            dict_allinc[cid] = income_table(cid)\n        except TypeError:\n            counterror = counterror+1\n            print "Error with this page: ", cid\n        except KeyError:\n            counterror = counterror+1\n            print "Error with this page: ", cid\n        if k%100==0:\n            print k,\nprint "\\n Number of errors: ", counterror')


d_inc_HH=pd.DataFrame([])
d_inc_HH = d_inc_HH.from_dict(dict_allinc,orient = "index") #d_inc_HH associates income to all candidates
                                                             # as well as for the whole family
                                                             # and the number of declarations
d_inc_HH.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\incomes.csv", index=True)
d_inc_HH.head()


#This part is just to play a bit with incomes
d_inc = d_inc_HH.copy()
d_inc['ln_HHinc'] = np.log(d_inc['HHinc'])
d_inc['ln_selfinc'] = np.log(d_inc['self_inc'])
d_inc['sh_self'] = d_inc['self_inc']/d_inc['HHinc']
sns.kdeplot(d_inc.ln_HHinc)
sns.kdeplot(d_inc.ln_selfinc)


d_inc[d_inc.HHinc!=0].describe()


# ### In this second part we look at the personal information for each candidate. ###
# 

cols = ['year','cid','full_name','district','state','party_full','address','self_profession','spouse_profession','age']
def personal_info(candidate_id):
    year = find_year(candidate_id)
    page_candidate = pol_pages[candidate_id]
    c_soup = BeautifulSoup(page_candidate,"html.parser")
    personal = c_soup.findAll(attrs={"class": "grid_3 alpha"})[0]
    full_name = personal.find("h2").get_text().strip().title()
    district1 = personal.find("h5").get_text().strip()
    district = district1.title()
    state = district1[district1.find("(")+1:district1.find(")")].title()
    grid2 = personal.findAll(attrs={"class":"grid_2 alpha"})
    party_full = grid2[0].get_text().split(":")[1].split("\n")[0]
    age = grid2[2].get_text().split(":")[1].split("\n")[0]
    try:
        age = float(age)
    except ValueError:
        age = np.nan
    address = grid2[3].get_text().split(":")[1].split("\n")[1].strip() # Careful this one changes
    if personal.find("p").get_text()=="":
        self_profession = ""
        spouse_profession = ""
    else:
        self_profession = personal.find("p").get_text().split('\n')[0].split(":")[1].capitalize()
        spouse_profession = personal.find("p").get_text().split('\n')[1].split(":")[1].capitalize()
    list_info = [candidate_id,full_name,district,state,party_full,address,self_profession,spouse_profession]
    list_encode = [year]+[x.encode('utf-8') for x in list_info]+[age]
    dict_info = dict(zip(cols,list_encode))
    return dict_info


get_ipython().run_cell_magic('time', '', 'counterror = 0\ndict_allcand = {}\nfor k,cid in enumerate(pol_pages.keys()):\n    try:\n        dict_allcand[cid] = personal_info(cid)\n    except TypeError:\n        counterror = counterror+1\n        print "Error with this page: ", cid\n    if k%100==0:\n        print k,\nprint "Number of errors: ", counterror')


d_perso_info = pd.DataFrame([])
d_perso_info = d_perso_info.from_dict(dict_allcand, orient="index") # Dumping into a dataframe
d_perso_info.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\info_perso_LS.csv",index = True)


d_perso_info[d_perso_info.self_profession!=""].head(10)


d_perso_info.describe()


sns.kdeplot(d_perso_info.age.dropna(),cumulative=True)


# # Scraping the Liabilities
# 

# Loading libraries and the big json file
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from bs4 import BeautifulSoup
import unicodedata
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
import requests


import json
with open('candidate_pages.json') as data_file:    
    data = json.load(data_file)


dic2014 = {}
for k in data.keys():
    if 'http://myneta.info/ls2014/' in k:
        dic2014[k] = data[k]


dic2009 = {}
for k in data.keys():
    if 'http://myneta.info/ls2009/' in k:
        dic2009[k] = data[k]


dic2004 = {}
for k in data.keys():
    if 'http://myneta.info/loksabha2004/' in k:
        dic2004[k] = data[k]


# Creating a toy example
# 

ex = data['http://myneta.info/ls2014/candidate.php?candidate_id=7120']


# The following function gets the information from the text. Returns de information in each text as a dictionary
# 

def get_info_2014(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll("a"):
        try:
            if r['name'] == "liabilities":
                table = r
                for r in table.findAll('td'):
                    try:

                        if r.get_text()=="Loans from Banks / FIs":
                            dic["Loans from Banks / FIs"] = r.findNext('b').get_text()

                        if r.get_text()=="Loans due to Individual / Entity":
                            dic["Loans due to Individual / Entity"] = r.findNext('b').get_text()

                        if r.get_text()=="Any other Liability":
                            dic["Any other Liability"] = r.findNext('b').get_text()

                        if r.get_text()=="Grand Total of Liabilities (as per affidavit)":
                            dic["Grand Total of Liabilities (as per affidavit)"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with government accommodation":
                            dic["Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of water":
                            dic["Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of electricity":
                            dic["Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with telephones":
                            dic["Dues to departments dealing with telephones"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of transport":
                            dic["Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

                        if r.get_text()=="Income Tax Dues":
                            dic["Income Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Wealth Tax Dues":
                            dic["Wealth Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Service Tax Dues":
                            dic["Service Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Property Tax Dues":
                            dic["Property Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Sales Tax Dues":
                            dic["Sales Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Any Other Dues":
                            dic["Any Other Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Grand Total of all Govt Dues (as per affidavit)":
                            dic["Grand Total of all Govt Dues (as per affidavit)"] = r.findNext('b').get_text()

                        if r.get_text()=="Whether any other liabilities are in dispute, if so, mention the amount involved and the authority before which it is pending":
                            dic["Whether any other liabilities are in dispute, if so, mention the amount involved and the authority before which it is pending"] = r.findNext('b').get_text()

                        if r.get_text()=="Totals (Calculated as Sum of Values)":
                            dic["Totals (Calculated as Sum of Values)"] = r.findNext('b').get_text()
                    except:
                        pass
        except:
            pass
    return dic


# Testing the example
# 

get_info_2014(ex)


# The following function cleans the dictionary. Numbers are in weird format, this function converts them from string into numbers
# 

def clean_dic_2014(dic):
    for k in dic.keys():
        if dic[k]==' Nil ':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic


# This is an example on dictionary cleaning
# 

clean_dic_2014(get_info(ex))


# This overall loop runs through the whole text for each of the candidates. The loops takes approximately 7 minutes to run in my computer
# 

get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2014.keys():\n    dic = get_info_2014(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2014(dic))\nlist2014 = tmplist")


# Now we get info for year 2009
# 

def get_info_2009(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll('td'):
        if r.get_text()=="Loans from Banks":
            dic["Loans from Banks"] = r.findNext('b').get_text()
                            
        if r.get_text()=="Loans from Financial Institutions":
            dic["Loans from Financial Institutions"] = r.findNext('b').get_text()

        if r.get_text()=="(a) Dues to departments dealing with government accommodation":
            dic["(a) Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

        if r.get_text()=="(b) Dues to departments dealing with supply of water":
            dic["(b) Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

        if r.get_text()=="(c) Dues to departments dealing with supply of electricity":
            dic["(c) Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

        if r.get_text()=="(d) Dues to departments dealing with telephones":
            dic["(d) Dues to departments dealing with telephones"] = r.findNext('b').get_text()

        if r.get_text()=="(e) Dues to departments dealing with supply of transport":
            dic["(e) Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

        if r.get_text()=="(f) Other Dues if any":
            dic["(f) Other Dues if any"] = r.findNext('b').get_text()

        if r.get_text()=="(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]":
            dic["(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]"] = r.findNext('b').get_text()

        if '(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]' in r.get_text():
            dic["(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iii) Sales Tax [Only in case proprietary business]":
            dic["(iii) Sales Tax [Only in case proprietary business]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iv) Property Tax":
            dic["(iv) Property Tax"] = r.findNext('b').get_text()
        
        if r.get_text()=="Totals":
            dic["Totals"] = r.findNext('b').get_text()
    return dic


def clean_dic_2009(dic):
    for k in dic.keys():
        if dic[k]=='Nil':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic


ex = dic2009['http://myneta.info/ls2009/candidate.php?candidate_id=306']
clean_dic_2009(get_info_2009(ex))


get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2009.keys():\n    dic = get_info_2009(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2009(dic))\nlist2009 = tmplist")


# Now we do the same for year 2004
# 

def get_info_2004(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll('td'):
        if r.get_text()=="Loans from Banks":
            dic["Loans from Banks"] = r.findNext('b').get_text()
                            
        if r.get_text()=="Loans from Financial Institutions":
            dic["Loans from Financial Institutions"] = r.findNext('b').get_text()

        if r.get_text()=="(a) Dues to departments dealing with government accommodation":
            dic["(a) Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

        if r.get_text()=="(b) Dues to departments dealing with supply of water":
            dic["(b) Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

        if r.get_text()=="(c) Dues to departments dealing with supply of electricity":
            dic["(c) Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

        if r.get_text()=="(d) Dues to departments dealing with telephones":
            dic["(d) Dues to departments dealing with telephones"] = r.findNext('b').get_text()

        if r.get_text()=="(e) Dues to departments dealing with supply of transport":
            dic["(e) Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

        if r.get_text()=="(f) Other Dues if any":
            dic["(f) Other Dues if any"] = r.findNext('b').get_text()

        if r.get_text()=="(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]":
            dic["(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]"] = r.findNext('b').get_text()

        if '(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]' in r.get_text():
            dic["(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iii) Sales Tax [Only in case proprietary business]":
            dic["(iii) Sales Tax [Only in case proprietary business]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iv) Property Tax":
            dic["(iv) Property Tax"] = r.findNext('b').get_text()
        
        if r.get_text()=="Totals":
            dic["Totals"] = r.findNext('b').get_text()
    return dic


def clean_dic_2004(dic):
    for k in dic.keys():
        if dic[k]=='Nil':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic


ex = dic2004['http://myneta.info/loksabha2004/candidate.php?candidate_id=3453']
clean_dic_2004(get_info_2004(ex))


get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2004.keys():\n    dic = get_info_2004(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2004(dic))\nlist2004 = tmplist")


print len(data)
print len(list2004) + len(list2009) + len(list2014)


def change_dic2014(orig):
    dic = {}
    dic['url'] = orig['url']
    dic['liab_banks/fis'] = orig['Loans from Banks / FIs']
    dic['liab_accom'] = orig['Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['Any other Liability']
    dic['liab_tax_income'] = orig['Income Tax Dues']
    dic['liab_tax_wealth'] = orig['Wealth Tax Dues']
    dic['liab_tax_sales'] = orig['Sales Tax Dues']
    dic['liab_tax_prop'] = orig['Property Tax Dues']
    dic['liab_total'] = orig['Totals (Calculated as Sum of Values)']
    return dic    


def change_dic2009(orig):
    dic = {}
    dic['url'] = orig['url']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = np.nan
    if np.isnan(orig['Loans from Banks'])==False and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = orig['Loans from Banks']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions'])==False:
        dic['liab_banks/fis'] = orig['Loans from Financial Institutions']
    dic['liab_accom'] = orig['(a) Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['(b) Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['(c) Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['(d) Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['(e) Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['(f) Other Dues if any']
    dic['liab_tax_income'] = orig['(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]']
    dic['liab_tax_wealth'] = orig['(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]']
    dic['liab_tax_sales'] = orig['(iii) Sales Tax [Only in case proprietary business]']
    dic['liab_tax_prop'] = orig['(iv) Property Tax']
    dic['liab_total'] = orig['Totals']
    return dic 


def change_dic2004(orig):
    dic = {}
    dic['url'] = orig['url']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = np.nan
    if np.isnan(orig['Loans from Banks'])==False and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = orig['Loans from Banks']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions'])==False:
        dic['liab_banks/fis'] = orig['Loans from Financial Institutions']
    dic['liab_accom'] = orig['(a) Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['(b) Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['(c) Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['(d) Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['(e) Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['(f) Other Dues if any']
    dic['liab_tax_income'] = orig['(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]']
    dic['liab_tax_wealth'] = orig['(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]']
    dic['liab_tax_sales'] = orig['(iii) Sales Tax [Only in case proprietary business]']
    dic['liab_tax_prop'] = orig['(iv) Property Tax']
    dic['liab_total'] = orig['Totals']
    return dic 


change_dic2009(list2009[1000])


change_dic2014(list2014[1000])


change_dic2004(list2004[1000])


# Now we merge the three years into one data frame
# 

allyears = []
for dic in list2004:
    allyears.append(change_dic2004(dic))
for dic in list2009:
    allyears.append(change_dic2009(dic))
for dic in list2014:
    allyears.append(change_dic2014(dic))


len(allyears)


liabilities = pd.DataFrame(allyears)
liabilities.to_csv("liabilities.csv", header=True, index=False)





# # Random Forests Model
# 
# The one uses the basic model and splits the test and training set by:
# * Year
# * Entire constituencies
# 

from pyquery import PyQuery as pq
import urllib2
import requests
import pandas as pd
import re
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from time import sleep
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 400)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook")
from sklearn.ensemble import RandomForestClassifier
get_ipython().magic('matplotlib inline')

from sklearn.metrics import roc_curve, auc
from scipy import interp


# ### Load dataset and make variables categorical
# 

ls_2014 = "http://myneta.info/ls2014/index.php?action=show_winners&sort=default"
ls_2009 = "http://myneta.info/ls2009/index.php?action=show_winners&sort=default"
ls_2004 = "http://myneta.info/loksabha2004/index.php?action=show_winners&sort=default"
base_2014 = "http://myneta.info/ls2014/"
base_2009 = "http://myneta.info/ls2009/"
base_2004 = "http://myneta.info/loksabha2004/"

url_list={'2004':ls_2004,'2009':ls_2009,'2014':ls_2014}
base_urls = {'2004':base_2004,'2009':base_2009,'2014':base_2014}


candidates = pd.read_csv('../candidates.csv')   
test_size = 0.2
train_size = 0.8

candidates.Party = candidates['Party'].astype('category')
candidates.Constituency = candidates['Constituency'].astype('category')
candidates.State = candidates['State'].astype('category')
candidates.Education = candidates['Education'].astype('category')
candidates.Assets_Rs = candidates.Assets_Rs.convert_objects(convert_numeric=True)
## Some null values in Assets that needed to be cleaned up
candidates.loc[candidates.Assets_Rs.isnull(),'Assets_Rs'] = 0

candidates.Constituency = candidates.Constituency.cat.rename_categories(range(0,len(candidates.Constituency.unique())))
candidates.Party = candidates.Party.cat.rename_categories(range(0,len(candidates.Party.unique())))
candidates.State = candidates.State.cat.rename_categories(range(0,len(candidates.State.unique())))
candidates.Education = candidates.Education.cat.rename_categories(range(0,len(candidates.Education.unique())))




# Merging 'Type of Crime'
# 

crim_data = pd.read_csv('../criminal_record.csv')

gby_type = crim_data.groupby(['url','type']).count()
gby_type = gby_type.reset_index()
gby_type_piv = gby_type.pivot(index='url',columns='type',values='IPC_sections')
candidates['full_link'] = candidates.Year.apply(lambda x: base_urls[str(x)]) + candidates.Link

candidates = candidates.merge(gby_type_piv, how='left', left_on='full_link', right_index=True)
candidates.loc[candidates.Cogniz.isnull(),'Cogniz'] = 0
candidates.loc[candidates.Framed.isnull(),'Framed'] = 0
candidates.loc[candidates.Convict.isnull(),'Convict'] = 0


# Incorporating incumbency
# 

def find_year(yr_string):
    found = re.findall(r'\d+',yr_string)
    if len(found):
        return found[0]
    else :
        return "0"


multi_elect = pd.read_csv('../multiple_elections.csv')
multi_elect['year'] = multi_elect.elec_link.apply(lambda x: find_year(x.split('/')[3]))
#print multi_elect.head(3)


prev_list = []
for link, df in multi_elect.groupby('common_link'):
        df.ite = df.sort_values('year')
        prev_elec = {'common_link':link}
        for x,y in df[['elec_link','year']].iteritems():                    
            prev_elec[x] = list(y)
        win = []
        for y in prev_elec['elec_link']:
            winner = candidates.loc[candidates.full_link == y, 'Winner']
            if len(winner > 0):
                win.append(int(winner.iloc[0]))
            else :
                win.append(0)
        prev_elec['win'] = win
        prev_list.append(prev_elec)


# Following take a while to run. You may want to load the csv file instead.
# 

# Adding recent race information

def greater_years(year):
    year_list = []
    year_int = int(year)
    for y in d['year']:
        if int(y) < year_int:
            year_list.append(y)
    return len(year_list)

def recent_years(year):
    year_list = []
    y_small = 0
    year_int = int(year)
    for y in d['year']:
        y_int = int(y)
        if (y_int < year_int) and (y_int > y_small) :
            y_small = y_int
    
    if y_small == 0:
        return 0
    else:
        return (year_int - y_small)

def total_wins(year):
    win_list = []
    year_int = int(year)
    for i,y in enumerate(d['year']):
        if int(y) < year_int:
            win_list.append(d['win'][i])
    return np.sum(win_list)
    
for d in prev_list:
    for i,link in enumerate(d['elec_link']):
        candidates.loc[candidates.full_link == link, 'Other_years'] = str(d['year'])
        candidates.loc[candidates.full_link == link, 'Total_Wins'] = candidates.Year.apply(total_wins)
        candidates.loc[candidates.full_link == link, 'History'] = candidates.Year.apply(greater_years)
        candidates.loc[candidates.full_link == link, 'Years_since'] = candidates.Year.apply(recent_years)
        candidates.loc[candidates.full_link == link, 'Incumbent'] = candidates.Years_since.apply(lambda x: (x<=5) and (x>0))
        candidates.loc[candidates.full_link == link, 'Other_elec'] = len(d['year'])
        candidates.loc[candidates.full_link == link, 'Common_link'] = d['common_link']
        if float(i%100.0) == 0.0:
            print ".",
            
candidates.head(3)


# Following code saves/loads the CSV, you can just start from here
# 

candidates.to_csv('../candidates_incumbency_crim.csv')


candidates = pd.read_csv('../candidates_incumbency_crim.csv')


# Merging 'Code of Crime'
# 

def concat_IPC(IPC):
    list = []
    for x in IPC:
        list = list + x.strip('[').strip(']').split(', ')
    return list

IPCs = crim_data[['url','IPC_sections']].groupby('url').agg(concat_IPC)
IPCs.head(3)


# Collect all the lists
y=[]
for x in crim_data.IPC_sections:
    x = x.strip('[').strip(']').split(', ')
    type(y)
    y = y + x

# Get a set of unique codes
z=[]
for code in y:
    if len(code) > 0:
        z.append(code)
unique_codes = list(set(z))

for col in unique_codes:
    IPCs[col] = IPCs.IPC_sections.apply(lambda x: int(col in x))

candidates = candidates.merge(IPCs,how='left',left_on='full_link',right_index=True)


for cols in candidates.columns:
    candidates.loc[candidates[cols].isnull(),cols] = 0


# Save/Load the final dataset
# 

candidates.to_csv('../candidates_final.csv')


candidates = pd.read_csv('../candidates_final.csv')


# ## 1. With a very simple dataset
# 

# ###Create test and train set
# 

def gen_test_train(df, RATIO, TYPE):

    if TYPE == 'YEAR':
        train_df = df[df.Year.isin([2009,2004])]
        test_df = df[df.Year == 2014]
        print "Using 'YEAR' split"
    
    elif TYPE == 'CONST':
        print "Using 'CONSTITUENCY' split"
        df['year_const'] = df.Year.apply(str) + df.Constituency.apply(str)
        all_year_const = df.year_const.drop_duplicates()
        test_index=np.random.choice(all_year_const, size=RATIO*all_year_const.count(), replace=False)

        train_df = df[~df.year_const.isin(test_index)]
        test_df = df[df.year_const.isin(test_index)]
    
    else:
        test_index=np.random.choice(range(df.count()[0]), size=RATIO*df.count()[0], replace=False)
        test_mask = df.index.isin(list(test_index))
        test_df = df[test_mask]
        train_df = df[~test_mask]
    
    print "Train set: %s, Test set: %s" % (train_df.count()[0], test_df.count()[0])
    
    cols = ['Age', 'Constituency', 'Criminal_Cases', 'Assets_Rs','Year',
                                  'Education', 'Liabilities_Rs', 'Party', 'State','Cogniz','Convict','Framed',
                                  'Total_Wins','History','Years_since','Incumbent']
                       
                       
#     cols = list(train_df.columns)
#     cols.remove('Winner')
#     cols.remove('Unnamed: 0')
#     cols.remove('IPC_sections')
#     cols.remove('full_link')    
#     cols.remove('Name')
#     cols.remove('Link')
    
    X_train = train_df[cols]
    X_test = test_df[cols]

    Y_train = train_df.Winner
    Y_test = test_df.Winner
    
    return X_train, Y_train, X_test, Y_test
   


# ###A poor effort at balancing - creating weights
# 

def generate_weights(Y, X_train):
    
    weight = [0,0]
    number_of_wins = np.sum(Y)
    weight[1] = float(number_of_wins)/len(Y)
    weight[0] = 1.0-weight[1]
    
    sample_weights = [weight[x] for x in Y]
    
    return np.array(sample_weights)


# ###Balancing the dataset
# 

def balance_me(X_train, y_train, prop=1.0):
    
    X_train['y_temp'] = y_train
    print np.sum(y_train)
    
    winners = X_train[X_train.y_temp == 1]
    losers = X_train[X_train.y_temp == 0]
    
    bal_index=np.random.choice(range(losers.count()[0]), size=len(winners), replace=False)
    print(len(bal_index))
    bal_mask = losers.index.isin(list(bal_index))
    
    loser_df = losers[bal_mask]
    print(len(loser_df))
    out = loser_df.append(winners)

    out_Y = list(out.y_temp)
    out_X = out.drop('y_temp',axis=1)
    
    return out_X, out_Y
    
X, Y = balance_me(X_train, Y_train)
len(X), len(X)


# ###Define a few functions for a random forest classfier
# 

def run_rforest(x_train, y_train, x_test, y_test, weight, est_type, n_est=100):
    
    rf = RandomForestClassifier(n_estimators=n_est)
    rf.fit(x_train, y_train, sample_weight=weight)

    prob = [x[1] for x in rf.predict_proba(x_test)]
    x_test_cp = x_test.copy()
    x_test_cp['y_hat'] = prob
    x_test_cp['y'] = y_test
    all_dfs = []
    if (est_type == 'YEAR') or (est_type == 'CONST'):
        big_df = pd.DataFrame()
        for const, df in x_test_cp.groupby('Constituency'):
            df = df.sort_values(['y_hat'],ascending=False)
            if len(df) > 0 :
                df['out'] = [0] * len(df)
                df.out.iloc[0] = 1
                if len(df) > 1:
                    df.out.iloc[1] = 2
                df = df.reset_index()
                all_dfs.append(df)
        big_df = pd.concat(all_dfs)
        return rf, big_df.out, big_df.y
    else:
        y_hat = []
        for val in prob:
            if val > est_type:
                y_hat.append(1)
            else:
                y_hat.append(0)
        return rf, y_hat, y_test
        


def confusion_matrix(y_hat, y, val=[1]):
    y_hat = list(y_hat)
    y = list(y)
    conf = [[0,0],[0,0]]
    score = 0.0
    for i,x in enumerate(y_hat):
        if x in val :
            if y[i] == 1:
                conf[1][1] += 1
                score += 1
            if y[i] == 0:
                conf[0][1] += 1
        if x not in val:
            if y[i] == 1:
                conf[1][0] += 1
            if y[i] == 0:
                conf[0][0] += 1    
                score += 1
    return conf, score/len(y_hat)


def make_roc(clf, testset, y, save=False, ax=False):
    probas_ = clf.predict_proba(testset)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,8))
    if not(ax):
        ax = plt.axes()
        
    ax.plot(fpr, tpr, lw=2, label='RF (area = %0.2f)' % (roc_auc))
    ax.plot([0,1],[0,1],ls='--')
    ax.legend(loc=0)

    if save :
        plt.savefig('roc_rf1.png', format='png')
    
    return ax


# ###Run the functions above splitting training and test sets by constituency
# 

# get dataset
X_train, Y_train, X_test, Y_test = gen_test_train(candidates, 0.2, 'CONST')
weights = generate_weights(Y_train, X_train)
(rf, Y_hat, Y) = run_rforest(X_train, Y_train, X_test, Y_test, weights, 'CONST', n_est=100)
conf, score = confusion_matrix(Y_hat, Y)
all_zeroes = [0]*len(Y_hat)

print "If you guessed all as losing: %s" % (1.0 - float(np.sum(Y_test))/len(Y_test))
print "Current model: %s" % score
print "Confusion matrix [0,1],[0,1]: %s" % conf

ax = make_roc(rf, X_test, Y_test, save=False)


# ### Now run by splitting by year
# 

# get dataset
X_train, Y_train, X_test, Y_test = gen_test_train(candidates, 0.2, 'YEAR')
weights = generate_weights(Y_train, X_train)
(rf, Y_hat, Y) = run_rforest(X_train, Y_train, X_test, Y_test, weights, 'YEAR', n_est=100)
conf, score = confusion_matrix(Y_hat, Y)
all_zeroes = [0]*len(Y_hat)

print "If you guessed all as losing: %s" % (1.0 - float(np.sum(Y_test))/len(Y_test))
print "Current model: %s" % score
print "Confusion matrix [0,1],[0,1]: %s" % conf

make_roc(rf, X_test, Y_test, save=False)


# ### Finally, the 'classic' version ###
# 

# get dataset
X_train, Y_train, X_test, Y_test = gen_test_train(candidates, 0.2, 0.40)
weights = generate_weights(Y_train, X_train)
(rf, Y_hat, Y) = run_rforest(X_train, Y_train, X_test, Y_test, weights, 0.40, n_est=100)
conf, score = confusion_matrix(Y_hat, Y, val=[1,2])
all_zeroes = [0]*len(Y_hat)

print "If you guessed all as losing: %s" % (1.0 - float(np.sum(Y_test))/len(Y_test))
print "Current model: %s" % score
print "Confusion matrix [0,1],[0,1]: %s" % conf

make_roc(rf, X_test, Y_test, save=False)





