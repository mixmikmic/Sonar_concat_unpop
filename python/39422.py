import multiprocess as mp

from glob import glob
import re
import pandas as pd
import numpy as np
import dill
import os
import warnings
import h5py

import cv2

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise
#from scipy.sparse import csr_matrix, vstack
#from sklearn.feature_extraction.text import TfidfTransformer


# # Clustering with KMeans to form a bag of words
# 

# The data set of the higher resolution images
large_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')
large_images.shape


# All remaining images
all_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_all_labels.pkl')
mask = all_images['basename'].isin(large_images['basename']).pipe(np.invert)
all_images = all_images.loc[mask]
all_images.shape


# Read in SIFT keypoints and perform k-means analysis to create codebook.
# 

is_trial = True

st = pd.HDFStore('../priv/data/features.h5', 'r')

mask = st['basename'].isin(large_images.basename)

print('Total images: {}'.format(st['basename'].shape[0]))
print('Total features: {}'.format(st['index']['end'].max()))

if is_trial:
    max_index = st['index'].loc[mask,'end'].max()
else:
    max_index = st['index']['end'].max()
    
print('Maximum index: {}'.format(max_index))

st.close()


def mini_batch_kmeans(data_path, out_path, max_index, n_clusters, frac_points=0.5):
    
    print n_clusters
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Select randomized indexes for data and read it in
    st = pd.HDFStore(data_path, 'r')
    n_points = int(frac_points * max_index)
    indexes = np.random.choice(np.arange(max_index), n_points, replace=False)
    data = st['features'].loc[indexes].values
    
    
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, init_size=3*n_clusters)
    model.fit(data)
    
    st.close()
    
    # Write the resulting model clusters out to a file
    st = h5py.File(out_path, 'a')
    if str(n_clusters) in st.keys():
        st.pop(str(n_clusters))
        
    mod = st.create_dataset(str(n_clusters), model.cluster_centers_.shape)
    mod[:,:] = model.cluster_centers_
    
    st.close()
    
    with open('../priv/models/minibatch_kmeans_clusters_{}.pkl'.format(n_clusters),'wb') as fh:
        dill.dump(model.cluster_centers_, fh)
        
    return


nclusters = [1500, 1536, 2000, 2500, 3000, 5000]

for cluster in nclusters:
    mini_batch_kmeans('../priv/data/features.h5', '../data/kmeans.h5', max_index, cluster)


get_ipython().system(' echo "pushover \'kmeans clustering finished\'" | /usr/bin/zsh')


# # Parse CellarTracker.com reviews
# 
# Parse the text file I have of amateur wine reviews from [CellarTracker.com](http://cellartracker.com). Note that this data file was downloaded from the web, and this is several years old. The website was scraped in a later notebook.
# 

import pandas as pd
import numpy as np

import requests
import json
from bs4 import BeautifulSoup

import dill
import re
import time


# Import the data. Note that it isn't in a csv format so it must be parsed.
# 

get_ipython().system(' head -n 10 ../priv/csv/cellartracker.txt')


with open('../priv/csv/cellartracker.txt','r') as fh:
    data_str = fh.read()


data_list = re.split(r"""\n\n""", data_str)


# ## Parse reviews
# Convert each entry to a Pandas Series and store it in a list.
# 

series_list = list()

for dat in data_list:
    
    dat_list = [x.strip() for x in dat.split('\n') 
                if (x.startswith('wine') or x.startswith('review'))]
    
    series_list.append(pd.Series(dict([re.search(r"""((?:wine|review)\/.+?): (.+)""", 
                                       x.strip()).groups() for x in dat_list])).T)
    


data_df = pd.concat(series_list, axis=1).T


data_df = data_df.rename_axis(lambda x: x.replace('/', '_'), axis=1)


data_df['wine_name'] = data_df.wine_name.apply(lambda x: x.replace('&#226;','a'))
data_df['review_text'] = data_df.review_text.apply(lambda x: x.replace('&#226;','a'))
# data_df['review_points'] = data_df.review_points.replace('N/A', np.NaN)
data_df = data_df.replace('N/A',np.NaN)


data_df.head()


data_df.isnull().sum()


data_df.to_pickle('../priv/pkl/03_cellartracker_dot_com_data.pkl')


import pandas as pd
import numpy as np
import os
import h5py
import warnings

from sklearn.metrics.pairwise import euclidean_distances
#from scipy.sparse import csr_matrix, vstack
# from sklearn.feature_extraction.text import TfidfTransformer


# # Convert clusters to bag of words
# 
# Also calculate the inverse document frequency (IDF) matrix.
# 

# The data set of the higher resolution images
large_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')
large_images.shape


is_trial = True

warnings.filterwarnings('ignore')

kmeans_file = '../priv/data/kmeans.h5'
km = h5py.File(kmeans_file, 'r')

features_file = '../priv/data/features.h5'
ft = pd.HDFStore(features_file, 'r')

hist_file = '../priv/data/hist.h5'
hs = pd.HDFStore(hist_file, 'w')

if is_trial:
    mask = ft['basename'].isin(large_images.basename)
    max_index = ft['index'].loc[mask,'end'].max()
    nimages = mask.sum()
else:
    max_index = ft['index']['end'].max()
    nimages = ft['index'].shape[0]
    
    
# for ncluster in ['1500']:
for ncluster in km.keys():
    print(ncluster)
    
    km_matrix = km[ncluster].value

    hist_list = list()
    
    for im in range(nimages):

        indexes = ft['index'].iloc[im]
        image_path = ft['image_path'].iloc[im]
        
        # This is a much faster and lower memory way of accessing a subset
        # of a dataframe
        features = ft.select('features', start=indexes.beg, stop=indexes.end).values
        
        # Pairwise euclidean distances
        ec = euclidean_distances(features, km_matrix)
        
        # Closest cluster id and count
        closest_clust_id = np.argmin(ec, axis=1)
        cluster_id, word_count = np.unique(closest_clust_id, return_counts=True)
        
        # Dense matrix of word counts
        bag_of_nums = np.zeros(int(ncluster), dtype=np.int)
        bag_of_nums[cluster_id] = word_count            
        
        # Store the histogram in the proper row
        hist_list.append(pd.Series(bag_of_nums, name=image_path))
        
    hist_df = pd.concat(hist_list, axis=1).T
    hist_df = hist_df.reset_index().rename(columns={'index':'image_path'})
    hs.append(ncluster, hist_df)
        
        
km.close()
ft.close()
hs.close()


# # Combine and clean expert reviews from WineEnthusiast.com
# 

import pandas as pd
import dill
from glob import glob
import re
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ## Aggregate the reviews
# 

# # Read in the entire list and sort it numerically
file_list = glob('../priv/pkl/06_wine_enthusiast_dot_com_data_*.pkl')
int_sorter = lambda x: int(re.search(r"""06_wine_enthusiast_dot_com_data_(.+).pkl""", x).group(1))
file_list = sorted(file_list, key=int_sorter)

full_list = np.arange(1,6530)
num_list = np.array([int_sorter(x) for x in file_list])

mask = np.invert(np.in1d(full_list, num_list))
print(sum(mask))
full_list[mask]


# Read in just the last 32 files to check them
# file_list = ! ls -tr ../pkl/06_wine_enthusiast_dot_com_data* | tail -n 32

# Load and combine the data for the list of files
combined_data = list()
for fil in file_list:
    
    with open(fil, 'r') as fh:
        
        data = dill.load(fh)
        
        for key in data.keys():
            
            dat = data[key]
            
            if isinstance(dat, pd.Series):
                dat['url'] = key[1]
                dat['list_url_no'] = key[0]
                combined_data.append(dat)
            else:
                print(key)
                
combined_df = pd.concat(combined_data, axis=1).T


print((combined_df.review.apply(lambda x: len(x))==0).sum())
print(combined_df.isnull().sum())


combined_df.shape


# ## Clean reviews
# 

# Drop the ones without reviews
mask = combined_df.review.apply(lambda x: len(x)==0).pipe(np.invert)
combined_df = combined_df.loc[mask]
combined_df.shape


# Convert prices to floats, have to remove the 'buy now'
replace_string = combined_df.loc[combined_df.price.str.contains('Buy Now'), 
                                 'price'].unique()[0]
combined_df.loc[combined_df.price==replace_string, 'price'] = np.NaN

combined_df['price'] = combined_df.price.astype(float)


combined_df['rating'] = combined_df.rating.astype(float)


# There are some % alcohol values that are way too high (above 100), 
# set anything above 50% to NaN
mask = combined_df.alcohol!='N/A'
combined_df.loc[mask, 'alcohol'] = combined_df.loc[mask,'alcohol'].str.replace(r"""\s*%""",'')
combined_df.loc[mask.pipe(np.invert), 'alcohol'] = np.NaN
combined_df['alcohol'] = combined_df.alcohol.astype(float)


mask = combined_df.alcohol >= 40.0
combined_df.loc[mask, 'alcohol'] = np.NaN


# Fixing bottle size requires some more extensive work
mask_L = combined_df.bottle_size.str.contains('L')
mask_ml = combined_df.bottle_size.str.contains('ml')
mask_M = combined_df.bottle_size.str.contains('ML')

combined_df.loc[mask_M, 'bottle_size'] = (combined_df
                                           .loc[mask_M, 'bottle_size']
                                           .str.replace(r"""ML""",'')
                                           .astype(float))

combined_df.loc[mask_L, 'bottle_size'] = (combined_df
                                       .loc[mask_L, 'bottle_size']
                                       .str.replace(r"""\s*L""",'')
                                       .astype(float)*1000)

combined_df.loc[mask_ml, 'bottle_size'] = (combined_df
                                           .loc[mask_ml, 'bottle_size']
                                           .str.replace(r"""\s*ml""",'')
                                           .astype(float))

combined_df['bottle_size'] = combined_df.bottle_size.astype(float)


combined_df['date_published'] = pd.to_datetime(combined_df.date_published)


# Was cleaning user ratings, but I decided to discard them as not useful
# combined_df['user_avg_rating'] = combined_df.user_avg_rating.str.replace(r""" \[Add Your Review\]""",'').head()
# combined_df.user_avg_rating.unique()
del combined_df['user_avg_rating']


# Some reviewers sign their reviews--remove these initials
mask = combined_df.review.str.contains(r"""\s+-\s?[A-Z]\.[A-Z]\.$""")
combined_df.loc[mask,'review'] = (combined_df
                                  .loc[mask,'review']
                                  .str.replace(r"""\s+-\s?[A-Z]\.[A-Z]\.$""", ''))


combined_df['title'].head()


def convert_year(ser):
    try:
        return int(ser)
    except:
        return ser
    
combined_df['year'] = combined_df['title'].str.extract(r""" ((?:19|20)[0-9]{2}) """, expand=True).apply(convert_year)


combined_df.year.isnull().sum()


# Discard blends
mask = combined_df.variety.str.contains('Blend').astype(np.bool).pipe(np.invert)
combined_df = combined_df.loc[mask]


combined_df.shape


# Discard all types except White, Red, and Rose
combined_df.category.unique()


mask = combined_df.category.isin(['White', 'Red', 'Rose'])
combined_df = combined_df.loc[mask]
combined_df.shape


# Clean up of wine variety names
# Now rename  probematic class names

#### Untouched class names ####
# Barbera                                        1091
# Cabernet Franc                                 1733
# Cabernet Sauvignon                            15830
# Chardonnay                                    17800
# Chenin Blanc                                    767
# Malbec                                         3681
# Merlot                                         6699
# Petit Verdot                                    287
# Pinot Blanc                                     634
# Pinot Noir                                    18284
# Pinotage                                        323
# Portuguese Red                                 2721
# Portuguese White                               1136
# Tempranillo                                    2976
# Viognier                                       1689
# Zinfandel                                      5216
# Gamay                                          1028
# Grenache                                        768
# Nebbiolo                                       3212
# Riesling                                       6819
# Sangiovese                                     4193
# Sauvignon Blanc                                7650

rename_dict = {'Aglianico, Italian Red'         :   'Aglianico',
 'Albariño'                                     :   'Albarino',
 'Blaufränkisch, Other Red'                     :   'Blaufrankisch',
 'Carmenère'                                    :   'Carmenere',
 'Corvina, Rondinella, Molinara, Italian Red'   :   'Corvina',
 'Dolcetto, Italian Red'                        :   'Dolcetto',
 'Garganega, Italian White'                     :   'Garganega',
 'Garnacha, Grenache'                           :   'Grenache',
 'Gewürztraminer'                               :   'Gewurztraminer',
 'Gewürztraminer, Gewürztraminer'               :   'Gewurztraminer',
 'Grüner Veltliner'                             :   'Gruner Veltliner',
 'Melon, Other White'                           :   'Melon',
 'Montepulciano, Italian Red'                   :   'Montepulciano',
 'Mourvèdre'                                    :   'Mourvedre',
 "Nero d'Avola, Italian Red"                    :   "Nero d Avola",
 'Petite Sirah'                                 :   'Petite Syrah',
 'Pinot Grigio, Pinot Grigio/Gris'              :   'Pinot Grigio',
 'Pinot Gris, Pinot Grigio/Gris'                :   'Pinot Grigio',
 'Primitivo, Zinfandel'                         :   'Zinfandel',
 'Rosé'                                         :   'Rose',
 'Sangiovese Grosso, Sangiovese'                :   'Sangiovese',
 'Sauvignon, Sauvignon Blanc'                   :   'Sauvignon Blanc',
 'Shiraz, Shiraz/Syrah'                         :   'Syrah',
 'Syrah, Shiraz/Syrah'                          :   'Syrah',
 'Tinta de Toro, Tempranillo'                   :   'Tempranillo',
 'Torrontés'                                    :   'Torrontes',
 'Verdejo, Spanish White'                       :   'Verdejo',
 'Vermentino, Italian White'                    :   'Vermentino'}

def val_rename(val):
    if val in rename_dict.keys():
        return rename_dict[val]
    else:
        return val
    
combined_df['variety'] = combined_df.variety.apply(lambda x: val_rename(x))


wine_varieties = combined_df[['variety','category']].groupby(['category','variety']).size()
wine_varieties_vc = wine_varieties.sort_values(ascending=False).reset_index().rename(columns={0:'count'})
wine_varieties_vc = wine_varieties_vc.query("count>=1200")
wine_varieties_vc


20*1500


combined_df_sampled = list()

for idx,dat in wine_varieties_vc.iterrows():
    
    mask = (combined_df.category==dat.category)&(combined_df.variety==dat.variety)
    combined_df_cat = combined_df.loc[mask]
    
    if dat['count'] < 1500:
        index = combined_df_cat.index
    else:
        index = np.random.choice(combined_df_cat.index, 1500, replace=False)
        
    combined_df_sampled.append(combined_df_cat.loc[index])
        
combined_df_sampled = pd.concat(combined_df_sampled, axis=0)        


combined_df_sampled.groupby(['variety','category']).size()


combined_df_sampled.to_pickle('../priv/pkl/07_wine_enthusiast_data_small_cleaned.pkl')


# ## Initial cut of data
# Based simply on class size
# 

# Keeping only those with at least 250 members cuts the number of varieties down to 50 but retains 93% of data
mask1 = wine_varieties>=500
wine_varieties.loc[mask1].shape, wine_varieties.loc[mask1].sum()/float(wine_varieties.sum())


print wine_varieties.loc[mask1].sum(), wine_varieties.sum(), wine_varieties.loc[mask1].nunique()
# print wine_varieties.sort_values(ascending=False).iloc[:wine_varieties.loc[mask1].nunique()]


wine_varieties = wine_varieties.loc[mask1]

mask2 = combined_df.variety.isin(wine_varieties.index.values.tolist())
combined_df_large_output = combined_df.loc[mask2]
print(combined_df_large_output.shape)


combined_df_large_output.variety.value_counts().sort_index()


combined_df.to_pickle('../priv/pkl/07_wine_enthusiast_data_cleaned.pkl')


# # Scrape vintage years
# 
# Scrape known vintage years for anticipated sentiment analysis as a function of year.
# 

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests


req = requests.get('http://www.erobertparker.com/newsearch/vintagechart1.aspx/VintageChart.aspx')


soup = BeautifulSoup(req.text, 'lxml')


charts = soup.find_all(attrs={'class':'chart'})


# Get the labels as a dataframe.
# 

labels = charts[0].find_all('tr')[1:-1]

label_list = [[(y.find('img'), y.get('rowspan')) 
               for y in x.find_all('td')] 
              for x in labels]

label_len = len(label_list)
empty_list = [np.NaN]*label_len
label_df = pd.DataFrame({'loc1':empty_list, 'loc2':empty_list, 'loc3':empty_list})

for col in range(3):
    pos = 0
    while pos < label_len:

        label = label_list[pos].pop(0)
        
        try:
            text = label[0]
            if text is None:
                text = np.NaN
            else:
                text = text.get('alt')
            
            nrows = label[1]
            if nrows is None:
                nrows = 1
            else:
                nrows = int(nrows)
            
            label_df.loc[pos:pos+nrows, 'loc'+str(col+1)] = text
            pos += nrows
        except:
            pos += 1


label_df.head()


label_df.shape


# Get the year labels for the chart.
# 

year_list = [x.text.strip() for x in charts[1].find('tr').find_all('th')]


len(year_list)


# Get the rankings for the chart and clean up the values.
# 

ranking_df = pd.DataFrame([[y.text.strip() 
               for y in x.find_all('td')] 
              for x in charts[1].find_all('tr')[1:-1]]).loc[:,1:]

ranking_df.columns = year_list

ranking_df = pd.concat([label_df, ranking_df], axis=1).set_index(['loc1', 'loc2', 'loc3'])

ranking_df = ranking_df.replace('NT',np.NaN).replace('NV',np.NaN)

for col in ranking_df.columns:
    ranking_df[col] = ranking_df[col].str.replace(r"""[A-Z]+""", '')
    ranking_df[col] = ranking_df[col].apply(lambda x: float(x))


ranking_df.head(5)


ranking_df.to_pickle('../priv/pkl/05_vintage_years.pkl')


import pandas as pd
import numpy as np
import os
import h5py
import dill
import warnings


# # Create inverted index of files with clusters
# 

warnings.filterwarnings('ignore')

hist_file = '../priv/data/hist.h5'
hs = h5py.File(hist_file, 'r')

features_file = '../priv/data/features.h5'
ft = pd.HDFStore(features_file, 'r')
files = ft['image_path'].values
ft.close()

index_file = '../priv/data/inverted_index.h5'
ix = pd.HDFStore(index_file, 'w')

for ncluster in hs.keys():
        
    print ncluster
    # Get the histograms for a given number of clusters
    dat = hs[ncluster]
    
    df_list = list()
    for clust in range(dat.shape[1]):
        
        # All files that have data at a given cluster
        file_index = np.where(dat[:, clust] !=0 )[0]
        clust_counts = dat[:, clust][file_index].astype(int)
        df = pd.DataFrame({'file':files[file_index], 
                           'count':clust_counts},
                          index = pd.Index([clust] * len(file_index)))
        df_list.append(df)
        
    cluster_df = pd.concat(df_list)
    ix.append(ncluster, cluster_df)
        
hs.close()
ix.close()


import pandas as pd
import numpy as np

import requests
import json
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from fake_useragent import UserAgent
import multiprocess as mp

from glob import glob

import dill
import re
import time


# A function to create the Selenium web driver

def make_driver(port):
    
    service_args = ['--proxy=127.0.0.1:{}'.format(port), '--proxy-type=socks5']
    
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    ua = UserAgent()
    dcap.update({'phantomjs.page.settings.userAgent':ua.random})
    
    phantom_path = '/usr/bin/phantomjs'
    
    driver = webdriver.PhantomJS(phantom_path, 
                                   desired_capabilities=dcap,
                                   service_args=service_args)
    
    return driver


ncomputers = 16
nthreads = 16

port_nos = np.array([8081+x for x in range(ncomputers)])


# Start the ssh tunnels
get_ipython().system(' ./ssh_tunnels.sh')


# # Scrape the catalog of producers
# 

def scrape_catalog(args):
    port, page_nums = args

    base_url = 'http://www.wine-searcher.com/biz/producers?s={}'
    
    driver = make_driver(port)

    table_list = list()
    for num in page_nums:
        print num
        
        full_url = base_url.format(num * 25 + 1)

        driver.get(full_url)
        time.sleep(10. + np.random.random()*5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        try:
            table = pd.read_html(html)[2]
            columns = table.iloc[0].values
            columns[2] = 'Wines'
            table = table.iloc[1:25]
            table.columns = columns

            url_list = [x.find('a').get('href') for x in soup.find_all(attrs={'class':'wlrwdt wlbdrl vtop'})]
            table['Url'] = url_list
            table['Page'] = num

            winery_urls = list()
            for url in url_list:
                try:
                    driver.get(url)
                    time.sleep(10. + np.random.random()*5)
                    html = driver.page_source
                    soup = BeautifulSoup(html, 'lxml')

                    winery_urls.append(soup.find(text=re.compile('Winery Profile')).find_parent().get('href'))
                except:
                    winery_urls.append('')

            table['Winery Url'] = winery_urls

            table.to_pickle('../pkl/13_wine_searcher_scraping_table_{}.pkl'.format(num))

            table_list.append(table)
        except:
            pass
        
    return table_list


# Load the completed data

num_list = np.arange(0,1742)

file_list = glob('../pkl/13_wine_searcher_scraping_table_*.pkl')
int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.pkl""", x).group(1))
file_nums = sorted(np.array(map(int_sorter, file_list)))

num_list = num_list[np.invert(np.in1d(num_list, file_nums))]

num_list


len(num_list)


used_threads = np.min([nthreads, len(num_list)])

used_port_nos = port_nos
if used_threads < nthreads:
    used_port_nos = port_nos[:used_threads]
    
pool = mp.Pool(processes=used_threads)
table_list = pool.map(scrape_catalog, [x for x in zip(used_port_nos, 
                                                     np.array_split(num_list, used_threads))])
pool.close()


table_df = pd.concat(sum(table_list,[]), axis=0).reset_index(drop=True)


table_df.to_pickle('../pkl/13_wine_searcher_url_table.pkl')


# ## Get the data from wineries with images
# 

# Get images from profile
driver.get(winery_url)
html = driver.page_source
soup = BeautifulSoup(html, 'lxml')


num_images = int(soup.find(attrs={'id':'img_high_t'}).text)


# iterate through each image, find "view larger" and download if it exists

pos = 1
for _ in range(num_images - 1):
    
    try:
        driver.find_element_by_id('showFullLabel1').click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        wine_info = soup.find(attrs={'id':'imgLabel'})
        wine_name = wine_info.get('alt')
        wine_url = wine_info.get('src')
        wine_height = wine_info.get('height')
        wine_width = wine_info.get('width')
        
        img = req.get(wine_url)#, proxies=req_proxy)
        time.sleep(1.2)

        filext = os.path.splitext(wine_url)[-1]
        path = 'tmp_' + str(pos) + '.' + filext

        if img.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in img:
                    f.write(chunk)
                    
        driver.find_element_by_id('okButtonModal').click()
    except:
        pass
                    
    pos += 1
    driver.find_element_by_id('nextImg').click()
        
        
    
    


import multiprocess as mp

from glob import glob
import re
import pandas as pd
import numpy as np
import os

from PIL import Image
import cv2

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# # Cleaning of wine label data set
# 
# Read in sizes of all wine labels to begin analysis.
# 

snooth_images = glob('../priv/images/snooth_dot_com_*.*')
wine_dot_com_images = glob('../priv/images/wine_dot_com_*.*')

int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.""", x).group(1))
snooth_images = sorted(snooth_images, key=int_sorter)
wine_dot_com_images = sorted(wine_dot_com_images, key=int_sorter)


def get_sizes(file_list):
    
    file_df = list()

    for fil in file_list:
        try:
            with Image.open(fil) as im:
                width, height = im.size        
        except:
            width = np.NaN
            height = np.NaN

        file_ser = pd.Series({'image_name':fil, 'width':width, 'height':height})
        
        file_df.append(file_ser)
        
    return file_df


file_list = snooth_images
file_list.extend(wine_dot_com_images)


nthreads = 48
pool = mp.Pool(processes=nthreads)
size_list = pool.map(get_sizes, np.array_split(file_list, nthreads))
pool.close()


image_size_df = pd.concat(sum(size_list,[]), axis=1).T


image_size_df['height'] = image_size_df.height.astype(int)
image_size_df['width'] = image_size_df.width.astype(int)
image_size_df['area'] = image_size_df.height * image_size_df.width


image_size_df.shape


def extract_basename(x):
    return os.path.splitext(os.path.basename(x))[0]

image_size_df['basename'] = image_size_df.image_name.apply(extract_basename)


image_size_df.area.min(), image_size_df.area.max()


# # Visualize label sizes
# Visualize the size of the images in a histogram.
# 

image_size_df.hist('area', bins=100)


image_size_df.hist('height', bins=100)


image_size_df.hist('width', bins=100)


# # Data cleaning notes
# 
# Specifically, there are many labels such at the one below that are "placeholders". There are other labels that are additionally just very small.
# 

img = Image.open('../priv/images/snooth_dot_com_47220.png')
img.size


plt.imshow(img)


# It also appears that some of the very large images will need to be removed. I believe these are only in the Wine.com data set.
# 

plt.imshow(Image.open('../priv/images/wine_dot_com_8516.jpg'))


# # Data for preliminary analysis
# Save the filenames of the high(-ish) resolution images to use for preliminary analysis. 
# 

mask = (image_size_df.area>1.0e6)&(image_size_df.area<1.5e6)
mask.sum()


image_size_df.loc[mask].to_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')


image_size_df.head()


# # Cleaning of data
# Remove small images
# 

mask = image_size_df.area>10000
image_size_df_out = image_size_df[mask]
image_size_df_out.shape


mask = np.invert((image_size_df.area>1.5e6)&(image_size_df['basename'].str.contains('wine_dot_com')))
image_size_df_out = image_size_df_out[mask]
image_size_df_out.shape


image_size_df_out.to_pickle('../priv/pkl/20_wine_label_analysis_all_labels.pkl')


# ## Average color histogram
# Histogram of nine image regions
# 

image = np.asarray(Image.open('../priv/images/wine_dot_com_8516.jpg'))
height, width = image.shape[:2]
nrows = 3
ncols = 3
w = int(width/ncols)
h = int(height/nrows)

segments = list()
for r in range(nrows):
    for c in range(ncols):
        x_beg = c*w
        y_beg = r*h
        
        if c != (ncols-1):
            x_end = (c+1)*w
        else:
            x_end = width+1
            
        if r != (nrows-1):
            y_end = (r+1)*h
        else:
            y_end = height+1
            
        segments.append((x_beg, x_end, y_beg, y_end))
        
segments


bins = (4, 6, 3)
# bins = (64, 64, 64)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = []

# loop over the segments
for (x_beg, x_end, y_beg, y_end) in segments:
    
    # construct a mask for each part of the image
    squareMask = np.zeros(image.shape[:2], dtype='uint8')
    cv2.rectangle(squareMask, (x_beg, y_beg), (x_end, y_end), 255, -1)
    
    hist = cv2.calcHist([image], [0, 1, 2], squareMask, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    
    features.extend(hist)


f, axList = plt.subplots(nrows=3, ncols=3)

for his,ax in zip(hist, axList.flatten()):
    ax.hist(his, bins=25)


# # Sentiment analysis on CellarTracker.com reviews
# 
# Preliminary sentiment analysis on amateur reviews. The sentiment by year was examined for some of the reviews. This notebook is just "scratch".
# 

import pandas as pd
from textblob import TextBlob
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('dark')
sns.set_context('talk')
sns.set_style('white')


data = pd.read_pickle('../priv/pkl/03_cellartracker_dot_com_data.pkl')


data = data.loc[data.review_text.isnull().pipe(np.invert)]


data['review_points'] = data.review_points.astype(float)


data.head(2)


data['textblob'] = data.review_text.apply(lambda x: TextBlob(x))


data2 = data[:2000].copy()


data2['polarity'] = data2.textblob.apply(lambda x: x.sentiment.polarity)
data2['subjectivity'] = data2.textblob.apply(lambda x: x.sentiment.subjectivity)


data2.dtypes


ax = data2.plot('review_points', 'polarity', marker='o',ls='')


ax = data2.plot('review_points', 'subjectivity', marker='o',ls='')


