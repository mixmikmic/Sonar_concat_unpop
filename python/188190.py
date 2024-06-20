# ## Notebook tasks for <b>cleaning</b> and <b>exploratory data analysis</b> for our thumbnail images:
# ### - Creates categorical value for Celebrity or Not
# ### - Concatenates all Labels
# ### - Concatenates all Text
# ### - Does LDA and divides into 15 topics
# #### 00 = every day lifestyle
# #### 01 = face shot/closeup
# #### 02 = floral
# #### 03 = women's daily fashion "swimwear" "boots" "heel"
# #### 04 = speakers before a crowd, formal events
# #### 05 = interior design, lifestyle
# #### 06 = outdoors, scenic views
# #### 07 = cosmetics and glam "lipstick", "dress", "mascara", "music", "hat", "instrument"
# #### 08 = red carpet premier, "drink", "beverage"
# #### 09 = rugged lifestyle "road", "sign", "dirt", "plaid", "gravel"
# #### 10 = cuisine, food, fine dining
# #### 11 = mixed media "poster" "collage" "paper" "flier"
# #### 12 = life luxury milestones/feminine lense "hair" "afro" "newborn" "bride" "underwear" "lingerie" "linen" "kid" "bling" "ring" "yacht"
# #### 13 = artistic form "dance" "pose" "tango" "paint"
# #### 14 = sports and fitness
# #### left out dummie variable: there are images that have no labels.
# ### - Trains model so that other pictures can be put into these topics
# ### - Celebrity column T/F
# ### - Text column T/F
# 
# Brought to you by Natalie Olivo
# <a href = https://www.linkedin.com/in/natalie-olivo-82548951/>LinkedIn</a>
# <a href = https://nmolivo.github.io/NMOstatic/>Website</a>
# <a href = https://medium.com/@NatalieOlivo>Blog</a>
# <a href = https://github.com/nmolivo>GitHub</a>
# 

import pandas as pd


thumb_imgs_long = pd.read_csv("../assets/thumbnail_link_long.csv")


thumb_imgs_long.head()


df = pd.read_csv("../gitignore/newtweets_10percent.csv")


mapping_dict = pd.read_csv("../assets/mapping_dict_thumbnail.csv")


mapping_dict.columns = [["img", "link_thumbnail"]]


df.columns


df = df[["brand", "engagement", "impact", "timestamp", "favorite_count", "hashtags", "retweet_count", "link_thumbnail"]]


df.head()


thumb = thumb_imgs_long.drop("Unnamed: 0", 1)


thumb.head()


# for-loop to drop all "Human", "People", "Person" Label rows where the image 
# contains a celebrity.
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Celebrity")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['label'].isin(['Human', 'People', 'Person'])))]


# for-loop to drop all "Label" rows below 90% confidence if there is a celebrity
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Celebrity")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['type'].isin(['Label'])) & (thumb['confidence']<90))]


# for loop to drop all "Label", "Sticker", "Text" label rows where image contains text.
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Text")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['label'].isin(['Label', 'Sticker', 'Text'])))]


import numpy as np


thumb.head(10)


thumb_new = []
for img in thumb['img'].unique():
    img_dict = {'img': img}
    if len(thumb[(thumb['img']==img) & (thumb['type']=='Label')])>0:
        img_dict['label'] = ' '.join(thumb.loc[(thumb['img']==img) & (thumb['type']=='Label'), 'label'].tolist())
    else:
        img_dict['label'] = None
    if len(thumb[(thumb['img']==img) & (thumb['type']=='Text')])>0:
        text = [str(detected_text) 
                for detected_text in thumb.loc[(thumb['img']==img) & (thumb['type']=='Text'), 'label'].tolist()]
        img_dict['text'] = ' '.join(text)
    else:
        img_dict['text'] = None
    img_dict['celebrity'] = len(thumb[(thumb['img']==img) & (thumb['type']=='Celebrity')])>0
    thumb_new.append(img_dict)
thumb_new_df = pd.DataFrame(thumb_new)


thumb_new_df["text"] = [False if x == None else True for x in thumb_new_df["text"]]


thumb_new_df.to_csv("01_thumb_text_data.csv")


import pandas as pd


#thumb_new_df = pd.read_csv("../assets/02_thumb_text_data.csv")


thumb_new_df.head()


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim


tokenizer = RegexpTokenizer(r'\w+')


# create English stop words list
en_stop = get_stop_words('en')


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()  


doc_set = thumb_new_df.loc[:,["img", "label"]]


# compile sample documents into a list
doc_set.dropna(inplace=True)


texts = []

# loop through document list
for i in doc_set.label:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)


corpus = [dictionary.doc2bow(text) for text in texts]


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=14, id2word = dictionary, passes=20)


print(ldamodel.print_topics(num_topics=14, num_words=5))


#this line is commented out so we don't re-save over our LDA model on images.
#ldamodel.save('labels_lda_14.model')


doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)


ldamodel[corpus[0]]


doc_label_topic_df.head()


thumb_new_df = thumb_new_df.merge(doc_label_topic_df, on='img', how='left')
thumb_new_df.head()


vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


pyLDAvis.display(vis)


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, id2word = dictionary, passes=20)


vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)


#we picked 14 topics. 
thumb_new_df.head()


thumb_new_df.head(10)


thumb_new_df = thumb_new_df.join(pd.get_dummies(thumb_new_df["label_topic"]))


image_df = thumb_new_df.drop(["label", "label_topic"], axis =1)


image_df = image_df.drop("label_topic_prob", axis=1)


image_df.to_csv("../assets/02_thumb_data.csv")


image_df





# ## Notebook tasks for <b>cleaning</b> and <b>exploratory data analysis</b> for our thumbnail images:
# ### - Creates categorical value for Celebrity or Not
# ### - Concatenates all Labels
# ### - Concatenates all Text
# ### - Does LDA and divides into 14 topics
# #### 00 = every day lifestyle
# #### 01 = face shot/closeup
# #### 02 = floral
# #### 03 = women's daily fashion "swimwear" "boots" "heel"
# #### 04 = speakers before a crowd, formal events
# #### 05 = interior design, lifestyle
# #### 06 = outdoors, scenic views
# #### 07 = cosmetics and glam "lipstick", "dress", "mascara", "music", "hat", "instrument"
# #### 08 = red carpet premier, "drink", "beverage"
# #### 09 = rugged lifestyle "road", "sign", "dirt", "plaid", "gravel"
# #### 10 = cuisine, food, fine dining
# #### 11 = mixed media "poster" "collage" "paper" "flier"
# #### 12 = life luxury milestones/feminine lense "hair" "afro" "newborn" "bride" "underwear" "lingerie" "linen" "kid" "bling" "ring" "yacht"
# #### 13 = artistic form "dance" "pose" "tango" "paint"
# #### 14 = sports and fitness
# #### left out dummie variable: there are images that have no labels.
# ### - Trains model so that other pictures can be put into these topics
# ### - Celebrity column T/F
# ### - Text column T/F
# 
# Brought to you by Natalie Olivo
# <a href = https://www.linkedin.com/in/natalie-olivo-82548951/>LinkedIn</a>
# <a href = https://nmolivo.github.io/NMOstatic/>Website</a>
# <a href = https://medium.com/@NatalieOlivo>Blog</a>
# <a href = https://github.com/nmolivo>GitHub</a>
# 

import pandas as pd


media_imgs_long = pd.read_csv("../assets/media_url_link_long.csv")


media_imgs_long.drop("Unnamed: 0", axis =1, inplace = True)


# for-loop to drop all "Human", "People", "Person" Label rows where the image 
# contains a celebrity.
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Celebrity")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['label'].isin(['Human', 'People', 'Person'])))]


# for-loop to drop all "Label" rows below 90% confidence if there is a celebrity
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Celebrity")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['type'].isin(['Label'])) & (media_imgs_long['confidence']<90))]


# for loop to drop all "Label", "Sticker", "Text" label rows where image contains text.
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Text")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['label'].isin(['Label', 'Sticker', 'Text'])))]


import numpy as np


media_new = []
for img in media_imgs_long['img'].unique():
    img_dict = {'img': img}
    if len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Label')])>0:
        img_dict['label'] = ' '.join(media_imgs_long.loc[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Label'), 'label'].tolist())
    else:
        img_dict['label'] = None
    if len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Text')])>0:
        text = [str(detected_text) 
                for detected_text in media_imgs_long.loc[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Text'), 'label'].tolist()]
        img_dict['text'] = ' '.join(text)
    else:
        img_dict['text'] = None
    img_dict['celebrity'] = len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Celebrity')])>0
    media_new.append(img_dict)
media_new_df = pd.DataFrame(media_new)


media_new_df


media_new_df["text"] = [False if x == None else True for x in media_new_df["text"]]


media_new_df.to_csv("01_media_text_data.csv")


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim


tokenizer = RegexpTokenizer(r'\w+')


# create English stop words list
en_stop = get_stop_words('en')


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()  


doc_set = media_new_df.loc[:,["img", "label"]]


# compile sample documents into a list
doc_set.dropna(inplace=True)


texts = []

# loop through document list
for i in doc_set.label:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)


corpus = [dictionary.doc2bow(text) for text in texts]


ldamodel = models.ldamodel.LdaModel.load('labels_lda.model')


doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)


doc_label_topic_df.head()


media_new_df = media_new_df.merge(doc_label_topic_df, on='img', how='left')
media_new_df.head()


media_new_df = media_new_df.join(pd.get_dummies(media_new_df["label_topic"]))


media_new_df.drop(["label", "label_topic", "label_topic_prob"], axis =1, inplace = True)


media_new_df.to_csv("../assets/02_media_text_data.csv")


## some of those topic fitting percentages looked kind of low. Let's try to do topic modelling again. 


ldamodel2=gensim.models.ldamodel.LdaModel(corpus, num_topics=14, id2word = dictionary, passes=20)


#lda model 1
doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)


ldamodel[corpus[0]]


#might be worth while to create one LDA model on all the images.


doc_label_topic_df.head()


media_new_df = media_new_df.merge(doc_label_topic_df, on='img', how='left')
media_new_df.head()


media_new_df.to_csv("media_text_data.csv")





import pandas as pd
import numpy as np


thumb_imgs_long = pd.read_csv("../assets/02_thumb_text_data.csv")
media_imgs_long = pd.read_csv("../assets/02_media_text_data.csv")


thumb_map = pd.read_csv("../assets/mapping_dict_thumbnail.csv")
media_map = pd.read_csv("../assets/mapping_dict_mediaurl.csv")


thumb_map.columns = ["img", "url"]
media_map.columns = ["img", "url"]


thumb = thumb_imgs_long.drop("Unnamed: 0", axis =1)
thumb = thumb.merge(thumb_map, on = "img")


media = media_imgs_long.drop("Unnamed: 0", axis = 1)
media = media.merge(media_map, on="img")


thumb[thumb.isnull().any(axis=1)]


media.head()


thumb.columns = ["thumb-celeb","thumb-img","thumb-text", "0", "1", "2", "3", "4", "5", "6", "7", "8", 
                 "9", "10", "11", "12", "13", "link_thumbnail"]


media.columns = ["media-celeb","media-img","media-text", "0", "1", "2", "3", "4", "5", "6", "7", "8", 
                 "9", "10", "11", "12", "13", "media_url"]


all_data = pd.read_csv("../gitignore/newtweets_10percent.csv")


all_data.columns


all_data = all_data[["id", "brand", "link_thumbnail", "media_url", "engagement", "impact", 
                    "timestamp", "hashtags", "favorite_count", "retweet_count", "text", "tweet_url"]]


all_data = all_data.merge(thumb, on = "link_thumbnail", how = "outer")


all_data = all_data.merge(media, on = "media_url", how = "outer")


# x is thumbnail, y is media. will be working to condense this
all_data


all_data.columns


#null thumbnails
len(all_data[all_data["link_thumbnail"].isnull()])
thumbnail_i = 10000-len(all_data[all_data["link_thumbnail"].isnull()])


thumbnail_i


#null media
len(all_data[all_data["media_url"].isnull()])
media_i = 10000-len(all_data[all_data["media_url"].isnull()])


media_i


len(all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].isnull())])


#800 records have either thumbnail or media link.


len(all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())])


#^^ records have both. I inspected these and discovered they are the same image. When merging, will favor thumbnail.


#confirming that if an image has both thumbnail and media link, they are the same image seen in tweet.
list(all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())]["link_thumbnail"])


# #### Create DF with one image per row:
# 1. Create a df where both media_url and thumbnail_link are blanks and then append cosolidated parts to it.
# 2. Append thumbnail-only rows
# 3. Append media-only rows
# 4. If an image has both thumbnail and image links, favor media link.
# 

new_df = all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].isnull())]


new_df.columns


new_df = new_df[["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]


new_df.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


len(new_df)


new_df.info()


new_df.head()


thumb_only = all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].isnull())]


thumb_only = thumb_only[["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]


thumb_only.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


new_df = new_df.append(thumb_only)


len(new_df)


sum(new_df.duplicated())


#total number of tweets with no image shown despite a thumbnail link
5597-5034-200


cleanup_thumb = new_df[new_df["image_url"].notnull() & new_df["thumb_img"].isnull()][["id", "image_url"]]


cleanup_thumb.to_csv("cleanup_thumb.csv")


new_df.info()


media_only = all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].notnull())]


media_only = media_only[["id", "brand", "timestamp", "media_url", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "media-celeb", 
                 "media-text", "0_y", "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", 
                 "10_y", "11_y", "12_y", "13_y"]]


media_only.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


media_cleanup = media_only[media_only["media_img"].isnull()][["id", "image_url"]]


media_cleanup.to_csv("cleanup_media.csv")


new_df = new_df.append(media_only)


len(new_df)


sum(new_df.duplicated())


new_df = new_df.drop_duplicates()


len(new_df)


5910 - 5333


new_df.info()


both = all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())]


len(both)


list(both["link_thumbnail"])[9]


list(both["tweet_url"])[9]


len(both)


both[["tweet_url", "link_thumbnail", "media_url","media-celeb", "thumb-img", "media-img", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", "10_x", "11_x", "12_x", "13_x", "0_y", 
      "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", "10_y", "11_y", "12_y", "13_y"]]


both_thumb = both[(both["thumb-img"].notnull())&(both["media-img"].isnull())][["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "thumb-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]


both_thumb.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


both_media = both[(both["media-img"].notnull())&(both["thumb-img"].isnull())][["id", "brand", "timestamp", "media_url", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "media-img", "media-img", "media-celeb", 
                 "media-text", "0_y", "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", 
                 "10_y", "11_y", "12_y", "13_y"]]


both_media.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


reduced_both = both_thumb.append(both_media)


len(reduced_both)


reduced_both.info()


new_df = new_df.append(reduced_both)


len(new_df)





