# # Exploratory Data Analysis
# 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5,3)


data = pd.read_pickle('data/data.pickle')
data.columns = ['drug_name','symptoms','label','severity_label']


data.describe()


# The dataset contains information about around 205 drugs over 450 symptoms and labels each combination based on its characteristics (9335) and severity label (3).
# 

data.severity_label.value_counts()


# We shall remove the records containing any label other than 'doctor', 'noneed' or 'emergency'.
# 

data = data[data['severity_label'].isin(['doctor','emergency','noneed'])]


def plot(df):

    plot_df = df.value_counts(normalize = True).apply(lambda x: x*100)
    
    recs = plot_df.plot(kind='barh')
    


plot(data['severity_label'])


# ### Let's take a look at the symptoms  
#   
# As can be seen from the figure below, some frequent terms include 'tiredness', 'stomach pain', 'loss of appetite', 'skin rash' etc. These are very common symptoms, hence one of the design goals should be to avoid any false negatives.
# 

symptoms = data['symptoms'].str.cat(sep=', ')


wordcloud = WordCloud(max_font_size=50).generate(symptoms)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('A wordcloud of symptoms')
plt.show()


symp_list = ['stomach','pain','abdominal','tiredness','weakness']


data['temp1'] = data['symptoms'].apply(lambda x: 1 if any(symp in x for symp in symp_list) else 0)


data.temp1.value_counts(normalize = True)








