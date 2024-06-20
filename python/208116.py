# ## NLTK POS Tagging + sklearn TF-IDF + Gensim Topic Modeling
# A new way forward
# 

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

import re
from stop_words import get_stop_words

import numpy as np
import pandas as pd

import time
import numpy as np
import matplotlib.pyplot as plt

# Some NLTK specifics
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer


import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim

from pprint import pprint

import codecs
import os
import time


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')


#CONNECTING TO THE DATASET
CORPUS_ROOT = "/Users/goodgame/desktop/Shift/match/reqs/jobs_text_meat_only/"

def load_data(root=CORPUS_ROOT):
    """
    Loads the text data into memory using the bundle dataset structure.
    Note that on larger corpora, memory safe CorpusReaders should be used.
    """

    # Open the README and store
    with open(os.path.join(root, 'README'), 'r') as readme:
        DESCR = readme.read()

    # Iterate through all the categories
    # Read the HTML into the data and store the category in target
    data      = []
    target    = []
    filenames = []

    for category in os.listdir(root):
        if category == "README": continue # Skip the README
        if category == ".DS_Store": continue # Skip the .DS_Store file
        for doc in os.listdir(os.path.join(root, category)):
            if doc == ".DS_Store": continue
            fname = os.path.join(root, category, doc)

            # Store information about document
            filenames.append(fname)
            target.append(category)
            with codecs.open(fname, 'r', 'ISO-8859-1') as f:
                data.append(f.read())
            # Read data and store in data list
            # with open(fname, 'r') as f:
            #     data.append(f.read())

    return Bunch(
        data=data,
        target=target,
        filenames=filenames,
        target_names=frozenset(target),
        DESCR=DESCR,
    )

dataset = load_data()

#print out the readme file
print(dataset.DESCR)
#Remember to create a README file and place it inside your CORPUS ROOT directory if you haven't already done so.

#print the number of records in the dataset
print("The number of instances is ", len(dataset.data), "\n")


job_str = '''Data Scientist/Machine Learning Engineer, Adaptive Authentication
Position Description:

This is an opportunity to join our fast-growing Adaptive Authentication team to develop cutting-edge risk-based adaptive authentication policies. We are looking for a Data Scientist/Machine Learning Engineer to build large-scale distributed systems while using machine learning to solve business problems. The ideal candidate has experience building models from complex systems, developing enterprise-grade software in an object-oriented language, and experience or knowledge in security, authentication or identity.

Our elite team is fast, innovative and flexible; with a weekly release cycle and individual ownership we expect great things from our engineering and reward them with stimulating new projects and emerging technologies.


Job Duties and Responsibilities:

Build and own models that identify risk associated with anomalous activity in the cloud for authentication
Build Machine Learning pipelines for training and deploying models at scale
Analyze activity data in the cloud for new behavioral patterns
Partner with product leaders to define requirements for building models
Work closely with engineering lead and management to scope and plan engineering efforts
Test-driven development, design and code reviews

Required Skills:

2+ years of Data Science/Machine Learning experience
Skilled in using machine learning algorithms for classification and regression
5+ years of software development experience in an object-oriented language building highly-reliable, mission-critical software
Excellent grasp of software engineering principles
Experience with multi-factor authentication, security, or identity is a plus

Education:

B.S, M.S, or Ph.D. in computer science, data science, machine learning, information retrieval, math or equivalent work experience

 

Okta is an Equal Opportunity Employer'''


stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    word = word.lower().replace('/','').replace('-','').replace('•','')
    # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term


# combine functions above
def noun_phrases(text):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')    
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    toks = tokenizer.tokenize(text)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    bad_words = ['opportunity', 'ideal candidate', 'team', 'year', 'knowledge','experience']
    clean_terms = []
    
    for term in terms:
        term = ' '.join(term).replace('\n','').replace(',','').replace('(','')
        term = term.replace(')','')
        term = term.strip()
        if term not in bad_words:
            clean_terms.append(term)
    return clean_terms





get_ipython().run_cell_magic('time', '', 'parsed_dataset = []\ntitles = []\n\ndef parse_input_docs(doc):\n    return \' \'.join(noun_phrases(doc))\n\nfor item in dataset.data:\n    titles.append(item.split("\\n",2)[0])\n    copy_minus_title = item.split("\\n",2)[2]\n    parsed_dataset.append(parse_input_docs(copy_minus_title))\nprint(len(parsed_dataset))\nprint(titles[0])')


# Use Gensim Phrases with POS-tagged JDs
# phrases = gensim.models.Phrases(list_pos_jd)
phrases = gensim.models.Phrases(parsed_dataset)
bigram = gensim.models.phrases.Phraser(phrases)


# ### Transform POS vectors into TF-IDF space. 
# The terms should be more useful than the terms from the entire document.
# 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# porter_stemmer = PorterStemmer()

def lemmatizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return words


stop_words = text.ENGLISH_STOP_WORDS


# TF-IDF transformation in sklearn

pos_vect = TfidfVectorizer(stop_words=stop_words, tokenizer=lemmatizer, ngram_range=(1,2), analyzer='word')  
pos_tfidf = pos_vect.fit_transform(parsed_dataset)
print("\n Here are the dimensions of our two-gram dataset: \n", pos_tfidf.shape, "\n")


true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(pos_tfidf)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = pos_vect.get_feature_names()
for i in range(true_k):
    print("\n\nCluster %d:" % i, "\n")
    for ind in order_centroids[i, :40]:
        print(' %s' % terms[ind])


# ### Gensim 
# #### (with streaming and new data input type)
# 
# Note that there's a memory-friendly way to do this by streaming docs.
# 
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Corpora_and_Vector_Spaces.ipynb
# 
# I'm not dealing with it yet because I want a POC first. But we'll need to use that if we build this corpus using all of the jobs in S3.
# 
# There are a few places where I'd need to use fixed-memory workarounds.
# 

print(parsed_dataset[0])


documents = parsed_dataset


# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]


get_ipython().system('rm /tmp/parsed.mm')


dictionary = gensim.corpora.Dictionary(texts)
dictionary.save('/tmp/parsed.dict')  # store the dictionary, for future reference


# Look at token IDs
# print(dictionary.token2id)


new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # All three words appear in the dictionary


corpus = [dictionary.doc2bow(text) for text in texts]
gensim.corpora.MmCorpus.serialize('/tmp/parsed.mm', corpus)  # store to disk, for later use


if (os.path.exists("/tmp/parsed.dict")):
    dictionary = gensim.corpora.Dictionary.load('/tmp/parsed.dict')
    corpus = gensim.corpora.MmCorpus('/tmp/parsed.mm')
    print("Used saved dictionary and corpus")
else:
    print("Please run first tutorial to generate data set")


print(dictionary[0])
print(dictionary[1])
print(dictionary[2])


tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)


# hdp = gensim.models.HdpModel(corpus, id2word=dictionary)


print(job_str)


doc = parse_input_docs(job_str)
print(doc, "\n\n")
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)


index = gensim.similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it


get_ipython().system('rm /tmp/parsed.index')
index.save('/tmp/parsed.index')


sims = index[vec_lsi]
sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
for item in sims_sorted[:5]:
    print(titles[item[0]],"\n\tIndex:",item[0],"\n\tSimilarity:",item[1])





print(documents[174])


# Word similarities in a high-scored document:
print([word for word in documents[821].split() if word in doc])


# Word similarities in a low-scored document
print([word for word in documents[174].split() if word in doc])


# This seems to work pretty well, but we lose all of the structure around the skill phrases, because we just project them into TF-IDF space.
# 
# This may work as an MVP solution, that we can hook up to the application.
# 
# For entity resolution, maybe we can look at trigram/bigram/unigram embeddings from these words, as trained on the wikipedia corpus. Then:
# 1. cluster the objects by their embeddings. Set K to be pretty high (the number of distinct skills we want represented -- perhaps 1000). In the near term, maybe we manually prune these clusters -- removing "junk" terms that we know as humans don't qualify as skills
# 2. Label each skill phrase by its cluster ID
# 3. Canonicalize the skill phrases by tagging them with a cluster ID. Input is a skill phrase, output is a genericized cluster ID that will be the same for any skill phrase that _is used in a similar context_ as the 
# 

# # try with a resume
# 

with open('sample_resume.txt', 'r') as infile:
    resume = infile.read()
doc = parse_input_docs(resume)
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space


sims = index[vec_lsi]
sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
for item in sims_sorted[:5]:
    print(titles[item[0]],"\n\tIndex:",item[0],"\n\tSimilarity:",item[1])


# Load Google's pre-trained Word2Vec model.
word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)  


word2vec.wv['product']  # numpy vector of a word


word2vec.wv.most_similar(positive=['woman', 'king'], negative=['man'])


# ## Phrase co-location
# 
# As a good preprocessing step. 
# https://radimrehurek.com/gensim/models/phrases.html
# 

phrases = gensim.models.Phrases(' '.join(parsed_dataset))


bigram = gensim.models.phrases.Phraser(phrases)


sent = ['machine','learning','is','so','hot','right','now']
print(bigram[sent])


from gensim.models import Phrases
from gensim.models.word2vec import LineSentence


# ## SpaCy
# From [this example on Github](https://github.com/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb)
# 
# Installation had some oddities; run the following:
# ```bash
# $ pip install -U spacy
# $ python -m spacy download en
# ```
# 
# I tried first without preprocessed "skill phrase" text as the corpus. Then, I ran back through with the "skill phrase" text. Both types of models and text output docs are saved in this directory.
# 

import spacy
import pandas as pd
import itertools as it
import en_core_web_sm

import spacy
nlp = spacy.load('en')


parsed_review = nlp(job_str)
print(parsed_review)


# for num, sentence in enumerate(parsed_review.sents):
#     print('Sentence {}:'.format(num + 1))
#     print(sentence)
#     print()


# token_text = [token.orth_ for token in parsed_review]
# token_pos = [token.pos_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_pos)),
#              columns=['token_text', 'part_of_speech'])


# token_lemma = [token.lemma_ for token in parsed_review]
# token_shape = [token.shape_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_lemma, token_shape)),
#              columns=['token_text', 'token_lemma', 'token_shape'])


# token_entity_type = [token.ent_type_ for token in parsed_review]
# token_entity_iob = [token.ent_iob_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_entity_type, token_entity_iob)),
#              columns=['token_text', 'entity_type', 'inside_outside_begin'])



# token_attributes = [(token.orth_,
#                      token.prob,
#                      token.is_stop,
#                      token.is_punct,
#                      token.is_space,
#                      token.like_num,
#                      token.is_oov)
#                     for token in parsed_review]

# df = pd.DataFrame(token_attributes,
#                   columns=['text',
#                            'log_probability',
#                            'stop?',
#                            'punctuation?',
#                            'whitespace?',
#                            'number?',
#                            'out of vocab.?'])

# df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']
#                                        .applymap(lambda x: u'Yes' if x else u''))
                                               
# df


from gensim.models import Phrases
from gensim.models.word2vec import LineSentence


# 1. Segment text of complete reviews into sentences & normalize text
# 2. First-order phrase modeling $\rightarrow$ apply first-order phrase 3. model to transform sentences
# 4. Second-order phrase modeling $\rightarrow$ apply second-order 5. phrase model to transform sentences
# 6. Apply text normalization and second-order phrase model to text of complete reviews
# 

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    SRG: modified for a list
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    for review in filename:
        yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])


get_ipython().run_cell_magic('time', '', "\nimport codecs\n# This is time consuming; make the if statement True to run\nif 0 == 0:\n    with codecs.open('spacy_parsed_jobs_PARSED.txt', 'w', encoding='utf_8') as f:\n        for sentence in parsed_dataset:\n            f.write(sentence + '\\n')")


unigram_sentences = LineSentence('spacy_parsed_jobs_PARSED.txt')


for unigram_sentence in it.islice(unigram_sentences, 230, 240):
    print(u' '.join(unigram_sentence))
    print(u'')


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 0:\n\n    bigram_model = Phrases(unigram_sentences)\n\n    bigram_model.save('spacy_bigram_model_all_PARSED')\n    \n# load the finished model from disk\nbigram_model = Phrases.load('spacy_bigram_model_all_PARSED')")


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_bigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for unigram_sentence in unigram_sentences:\n            \n            bigram_sentence = u' '.join(bigram_model[unigram_sentence])\n            \n            f.write(bigram_sentence + '\\n')")


bigram_sentences = LineSentence('spacy_bigram_sentences_PARSED.txt')


for bigram_sentence in it.islice(bigram_sentences, 240, 250):
    print(u' '.join(bigram_sentence))
    print(u'')


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 0:\n\n    trigram_model = Phrases(bigram_sentences)\n\n    trigram_model.save('spacy_trigram_model_all_PARSED')\n    \n# load the finished model from disk\ntrigram_model = Phrases.load('spacy_trigram_model_all_PARSED')")


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_trigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for bigram_sentence in bigram_sentences:\n            \n            trigram_sentence = u' '.join(trigram_model[bigram_sentence])\n            \n            f.write(trigram_sentence + '\\n')")


trigram_sentences = LineSentence('spacy_trigram_sentences_PARSED.txt')


for trigram_sentence in it.islice(trigram_sentences, 240, 250):
    print(u' '.join(trigram_sentence))
    print(u'')


# ### Final processing
# 

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_trigram_transformed_reviews_all_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for parsed_review in nlp.pipe(line_review(dataset.data),\n                                      batch_size=10000, n_threads=4):\n            \n            # lemmatize the text, removing punctuation and whitespace\n            unigram_review = [token.lemma_ for token in parsed_review\n                              if not punct_space(token)]\n            \n            # apply the first-order and second-order phrase models\n            bigram_review = bigram_model[unigram_review]\n            trigram_review = trigram_model[bigram_review]\n            \n            # remove any remaining stopwords\n            trigram_review = [term for term in trigram_review\n                              if term not in stopwords]\n            \n            # write the transformed review as a line in the new file\n            trigram_review = u' '.join(trigram_review)\n            f.write(trigram_review + '\\n')")


print(u'Original:' + u'\n')

for review in it.islice(line_review(dataset.data), 11, 12):
    print(review)

print(u'----' + u'\n')
print(u'Transformed:' + u'\n')

with codecs.open('spacy_trigram_transformed_reviews_all.txt', encoding='utf_8') as f:
    for review in it.islice(f, 11, 12):
        print(review)


from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to learn the dictionary yourself.\nif 0 == 0:\n\n    trigram_reviews = LineSentence('spacy_trigram_sentences_PARSED.txt')\n\n    # learn the dictionary by iterating over all of the reviews\n    trigram_dictionary = Dictionary(trigram_reviews)\n    \n    # filter tokens that are very rare or too common from\n    # the dictionary (filter_extremes) and reassign integer ids (compactify)\n    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)\n    trigram_dictionary.compactify()\n\n    trigram_dictionary.save('spacy_trigram_dict_all.dict')\n    \n# load the finished dictionary from disk\ntrigram_dictionary = Dictionary.load('spacy_trigram_dict_all.dict')")


def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """    
    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to build the bag-of-words corpus yourself.\nif 0 == 0:\n\n    # generate bag-of-words representations for\n    # all reviews and save them as a matrix\n    MmCorpus.serialize('spacy_trigram_bow_corpus_all.mm',\n                       trigram_bow_generator('spacy_trigram_sentences_PARSED.txt'))\n    \n# load the finished bag-of-words corpus from disk\ntrigram_bow_corpus = MmCorpus('spacy_trigram_bow_corpus_all.mm')")


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to train the LDA model yourself.\nif 0 == 0:\n\n    with warnings.catch_warnings():\n        warnings.simplefilter('ignore')\n        \n        # workers => sets the parallelism, and should be\n        # set to your number of physical cores minus one\n        lsi = gensim.models.LsiModel(trigram_bow_corpus, \n                                     id2word=trigram_dictionary, \n                                     num_topics=500)\n    \n    lsi.save('spacy_lsi_model_all')\n    \n# load the finished LDA model from disk\nlsi = LsiModel.load('spacy_lsi_model_all')")


def explore_topic(topic_number, topn=10):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
#     print(u'{:20} {}'.format(u'term', u'frequency') + u'')

    for term, frequency in lsi.show_topic(topic_number, topn=10):
        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))


for i in range(600):
    print("\n\nTopic %s" % str(i+1))
    explore_topic(topic_number=i)





# # Extract "Meat" from Job Descriptions
# 
# Purpose: many job descriptions have lots of text that isn't important. In fact, our methodology for pulling JD information is to take *_all_* visible text from a JD's website, which means that lots of it is useless.
# 
# This document cleans the JDs and prepares them for analysis. Here are the steps:
# 1. Look at a few JDs for each text. Identify the trends. Typically, the "meat" will start after the same line, or a series of lines. 
# 2. Remove everything that's not meat.
# 3. Make a dictionary. Key is job title, value is meat.
# 
# Process notes:
# - Lowercase e'rrything.
# - Strip each line.
# 
# ### Open Questions
# - How to handle "team" (currently: ignore, because it's organized differently by organization).
# 
# ### End State
# - Each word in the list of skills across all companies is a dimension; value is tf-idf (on post-processed skill words)
# 
# 
# #### Important Notes
# - The `.DS_Store` file that macOS autogenerates really fucks with this process. Remove it before running any of the scripts that iterate through directories.
# - Slashes in job titles mess up the file writing.
# 

# # Affirm
# Patterns:
# - 1st line is title
# - 2nd line is location
# - 3rd line is team
# - 4th and 5th lines are trash
# 
# Meat:
# - Lines after "WHAT YOU'LL DO"
# - Lines after "WHAT WE LOOK FOR
# 
# Stop trigger:
# 
# - "ABOUT AFFIRM" -- others too
# 

import os
import re


in_directory = './jobs_text_complete/affirm/'
out_directory = './jobs_text_meat_only/affirm/'

doc_num = 0

def extract_meat(in_directory, filename):
    meat = []
    bad_lines = ['what you\'ll do', 'what we look for', 'who we look for']
    abandon = ['about affirm', 'apply for this job', 'at affirm we are using technology to re-imagine and re-build core parts of financial infrastructure to enable cheaper, friendlier, and more transparent financial products and services that improve lives.']

    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'")
            counter += 1
            if counter in [1,2,3,4]:
                continue
            if counter == 0:
                title = line
            elif line in bad_lines:
                continue
            elif line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Affirm ' + title.title()
        header = 'Affirm ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ') + '.txt' 
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(directory):
    extract_meat(directory, filename)
    doc_num += 1
print("Processed", doc_num, "documents.")


# # Uber
# 
# Patterns:
# - Everything up to line 116 is junk
# - Looks like I can stop collecting at "Apply Now"
# 
# Notes:
# - Removed 'Agente-de-Atendimento' roles
# - Title name convention is different; titles of docs will include 'Uber' multiple times
# - I left in notes about the team. Too hard to sift out.
# 

about_uber = ['We’re changing the way people think about transportation. Not that long ago we were just an app to request premium black cars in a few metropolitan areas. Now we’re a part of the logistical fabric of more than 600 cities around the world. Whether it’s a ride, a sandwich, or a package, we use technology to give people what they want, when they want it.',
              ' For the people who drive with Uber, our app represents a flexible new way to earn money. For cities, we help strengthen local economies, improve access to transportation, and make streets safer.',
              ' And that’s just what we’re doing today. We’re thinking about the future, too. With teams working on autonomous trucking and self-driving cars, we’re in for the long haul. We’re reimagining how people and things move from one place to the next.',
              ' Uber is a technology company that is changing the way the world thinks about transportation. We are building technology people use everyday. Whether it\'s heading home from work, getting a meal delivered from a favorite restaurant, or a way to earn extra income, Uber is becoming part of the fabric of daily life.',
              ' We\'re making cities safer, smarter, and more connected. And we\'re doing it at a global scale-energizing local economies and bringing opportunity to millions of people around the world.',
              ' Uber\'s positive impact is tangible in the communities we operate in, and that drives us to keep moving forward.',
             'at uber, we pride ourselves on the amazing team we\'ve built. the driver behind all our growth, our bold and disruptive brand, and the game-changing technology we bring to market is the people that make uber well, uber.',
             'uber is an equal opportunity employer and enthusiastically encourages people from a wide variety of backgrounds and experiences to apply. uber does not discriminate on the basis of race, color, religion, sex (including pregnancy), gender, national origin, citizenship, age, mental or physical disability, veteran status, marital status, sexual orientation or any other basis prohibited by law.',
              'we\'re changing the way people think about transportation. not that long ago we were just an app to request premium black cars in a few metropolitan areas. now we\'re a part of the logistical fabric of more than 500 cities around the world. whether it\'s a ride, a sandwich, or a package, we use technology to give people what they want, when they want it.',
              'for the people who drive with uber, our app represents a flexible new way to earn money. for cities, we help strengthen local economies, improve access to transportation, and make streets safer.',
              'uber\'s positive impact is tangible in the communities we operate in, and that drives us to keep moving forward'
             ]

about_uber_lower = [item.strip().lower().replace('’',"'") for item in about_uber]

throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'qualifications & requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise']


uber_bad_lines = sorted(list(set(about_uber_lower + throwaway + [item + ':' for item in throwaway])))

# for item in uber_bad_lines:
#     print(item)

uber_abandon = ['perks', 'apply now', 'benefits', 'benefits:', 'perks:', 'apply now:', 'benefits (u.s.)']


in_directory = './jobs_text_complete/uber/'
out_directory = './jobs_text_meat_only/uber/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'")
            counter += 1
            if counter in range(116):
                continue
            elif line in bad_lines:
                continue
            elif line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Uber ' + title.title()
        header = 'Uber ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ') + '.txt' 
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, uber_bad_lines, uber_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


# # Salesforce
# 
# Patterns
# - About 980 lines of junk opening each file
# - Always ends with "Would you like to apply to this job?"
# 
# Notes
# - First line is a dirty title
# 


# Don't appear to be bad lines... tbd
throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role'
               ]

about_sf = ['about salesforce: salesforce, the customer success platform and world\'s #1 crm, empowers companies to connect with their customers in a whole new way. the company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model. these founding principles have taken our company to great heights, including being named one of forbes\'s “world\'s most innovative company” six years in a row and one of fortune\'s “100 best companies to work for” nine years in a row. we are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at salesforce. together, with our whole ohana (hawaiian for "family") made up of our employees, customers, partners and communities, we are working to improve the state of the world.',
           'salesforce is a critical business skill that anyone should have on their resume. by 2022, salesforce and our ecosystem of customers and partners will drive the creation of 3.3 million new jobs and more than $859 billion in new business revenues worldwide according to idc. salesforce is proud to partner with deloitte and our entire community of trailblazers to build a bridge into the salesforce ecosystem. our pathfinder program provides the training and accreditation necessary to be positioned for high paying jobs as salesforce administrators and salesforce developers.',
            'salesforce, the customer success platform and world\'s #1 crm, empowers companies to connect with their customers in a whole new way. the company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model. these founding principles have taken our company to great heights, including being named forbes\' \"world\'s most innovative company\" in 2017 and one of the \"world\'s most innovative company\" the previous five years. we have also been named one of fortune\'s \"100 best companies to work for\" nine years in a row. we are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at salesforce. together, with our whole ohana (hawaiian for \"family\") made up of our employees, customers, partners and communities, we are working to improve the state of the world.'
           ]

embedded_useless_sentences_raw = ['Salesforce, the Customer Success Platform and world\'s #1 CRM, empowers companies to connect with their customers in a whole new way.', 
                              'The company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model.',
                              'These founding principles have taken our company to great heights, including being named one of Forbes’s “World’s Most Innovative Company” five years in a row and one of Fortune’s “100 Best Companies to Work For” eight years in a row. We are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at Salesforce.',
                              'Together, with our whole Ohana (Hawaiian for \"family\") made up of our employees, customers, partners and communities, we are working to improve the state of the world.']

embedded_useless_sentences = [item.lower() for item in embedded_useless_sentences_raw]

sf_bad_lines = sorted(list(set([item.strip().lower() for item in about_sf] + 
                               throwaway + 
                               [item + ':' for item in throwaway])))

sf_abandon = ['would you like to apply for this job?', 'would you like to apply to this job?']


in_directory = './jobs_text_complete/salesforce/'
out_directory = './jobs_text_meat_only/salesforce/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if counter in range(980):
                continue
            elif line in bad_lines:
                continue
            for item in embedded_useless_sentences:
                line_fixed = line.replace(item, '')
                line = line_fixed                
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Salesforce ' + title.title()
        header = 'Salesforce ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ')
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, sf_bad_lines, sf_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


# # Okta
# 
# - Similar to Salesforce; threshold for trash at beginning is 209 lines
# - again, first line is ugly title
# 


# Don't appear to be bad lines... tbd
throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role',
             'required skills ',
             'preferred skills ',
             'position   description',
             'position description',
             'education and training',
             'minimum required knowledge, skills, and abilities',
             'nice to haves',
             'skills & qualifications',
             'preferred skills',
             'education and experience:',
             'preferred',
             'job duties and responsibilities',
             'minimum required knowledge, skills, and abilities ',
             'skills and experience',
             'what we are looking for',
             'bonus skills',
             'education and certification'
               ]


okta_bad_lines = sorted(list(set(throwaway + 
                               [item + ':' for item in throwaway])))

okta_abandon = ['okta is an equal opportunity employer', 'okta is an equal opportunity employer.',
               'okta is an equal opportunity employer', 'okta is an equal opoortunity employer']


in_directory = './jobs_text_complete/okta/'
out_directory = './jobs_text_meat_only/okta/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if counter in range(209):
                continue
            elif line in bad_lines:
                continue             
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
            if line[:10] == 'apply now':
                break
            if 'u.s. equal opportunity employment information' in line:
                break
        formatted_title = 'Okta ' + title.title()
        header = 'Okta ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ')
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, okta_bad_lines, okta_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


# # Square
# - Title is first line
# 

throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role',
             'required skills ',
             'preferred skills ',
             'position   description',
             'position description',
             'education and training',
             'minimum required knowledge, skills, and abilities',
             'nice to haves',
             'skills & qualifications',
             'preferred skills',
             'education and experience:',
             'preferred',
             'job duties and responsibilities',
             'minimum required knowledge, skills, and abilities ',
             'skills and experience',
             'what we are looking for',
             'bonus skills',
             'education and certification',
             'company description',
             'full-time',
             'san francisco, ca, usa',
             'bonus',
             'contract'
               ]

about_square = ['We believe everyone should be able to participate and thrive in the economy. So we’re building tools that make commerce easier and more accessible to all.  We started with a little white credit card reader but haven’t stopped there. Our new reader helps our sellers accept chip cards and NFC payments, and our Cash app lets people pay each other back instantly.   We’re empowering the independent electrician to send invoices, setting up the favorite food truck with a delivery option, helping the ice cream shop pay its employees, and giving the burgeoning coffee chain capital for a second, third, and fourth location.  Let’s shorten the distance between having an idea and making a living from it. We’re here to help sellers of all sizes start, run, and grow their business—and helping them grow their business is good business for everyone.',
                'we believe everyone should be able to participate and thrive in the economy. so we\'re building tools that make commerce easier and more accessible to all. we started with a little white credit card reader but haven\'t stopped there. our new reader helps our sellers accept chip cards and nfc payments, and our cash app lets people pay each other back instantly. we\'re empowering the independent electrician to send invoices, setting up the favorite food truck with a delivery option, helping the ice cream shop pay its employees, and giving the burgeoning coffee chain capital for a second, third, and fourth location. let\'s shorten the distance between having an idea and making a living from it. we\'re here to help sellers of all sizes start, run, and grow their business—and helping them grow their business is good business for everyone.'
               ]


square_bad_lines = sorted(list(set([item.lower() for item in about_square] +
                                   throwaway + 
                               [item + ':' for item in throwaway])))

square_abandon = ['additional information']


in_directory = './jobs_text_complete/square/'
out_directory = './jobs_text_meat_only/square/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = 0
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            if counter == 0:
                title = line
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if line in bad_lines:
                continue             
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Square ' + title.title()
        header = 'Square ' + title.title().replace('(','-').replace(')','-').replace('/','-') + '.txt'
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, square_bad_lines, square_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


from gensim.models import LsiModel
import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle
import gensim

from pprint import pprint

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import itertools as it

import en_core_web_sm

import spacy
nlp = spacy.load('en')

import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer


stopwords = stopwords.words('english')
# lemmatizer = nltk.WordNetLemmatizer()


# # load the finished bag-of-words corpus from disk
# trigram_bow_corpus = MmCorpus('../data/models_data_lower/spacy_trigram_bow_corpus_all.mm') # No POS preprocessing

# # load the finished dictionary from disk
# trigram_dictionary = Dictionary.load('../data/models_data_lower/spacy_trigram_dict_all.dict') # No POS preprocessing

# load the finished bag-of-words corpus from disk
trigram_bow_corpus_POS = MmCorpus('../data/spacy_trigram_bow_corpus_all_POS.mm') # With POS preprocessing

# load the finished dictionary from disk
trigram_dictionary_POS = Dictionary.load('../data/spacy_trigram_dict_all_POS.dict') # With POS preprocessing


# ## Create a new model
# 

get_ipython().run_cell_magic('time', '', "# Create LDA model\n\nwith warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n\n    # workers => sets the parallelism, and should be\n    # set to your number of physical cores minus one\n    lda_alpha_auto = LdaModel(trigram_bow_corpus_POS, \n                                 id2word=trigram_dictionary_POS, \n                                 num_topics=25,\n                             alpha='auto', eta='auto')\n    \n    lda_alpha_auto.save('../data/models_data_lower/spacy_lda_model_POS_alpha_eta_auto')")


# load the finished LDA model from disk
lda = LdaModel.load('../data/models_data_lower/spacy_lda_model_POS_alpha_eta_auto')


topic_names = {1: u'(?)Large Tech Corps (NVIDIA, Splunk, Twitch)',
               2: u'Technical Federal Contracting and Cybersecurity',
               3: u'Financial Risk and Cybersecurity',
               4: u'Web Development (More Frontend)',
               5: u'Social Media Marketing',
               6: u'Fintech, Accounting, and Investing Analysis/Data',
               7: u'(?)Students, Interns, CMS/Marketing, Benefits',
               8: u'Health Care (Data Systems)',
               9: u'Database Administrator',
               10: u'Marketing and Growth Strategy',
               11: u'Quality Assurance and Testing',
               12: u'Data Science',
               13: u'Big Data Engineering',
               14: u'Sales',
               15: u'(?)Large Tech Corps Chaff: Fiserv, Adove, SAP',
               16: u'Flight and Space (Hardware & Software)',
               17: u'Networks, Hardware, Linux',
               18: u'Supervisor, QA, and Process Improvement',
               19: u'Defense Contracting',
               20: u'Social Media Advertising Management',
               21: u'UX and Design',
               22: u'(?)Amazon Engineering/Computing/Robotics/AI',
               23: u'Mobile Developer',
               24: u'DevOps',
               25: u'Payments, Finance, and Blockchain'}


## Visualize


LDAvis_data_filepath = '../models/ldavis_prepared'


# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)


pyLDAvis.display(LDAvis_prepared)


# Original functions from processing step, in modeling.ipynb
def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    SRG: modified for a list
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    for review in filename:
        yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])


from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
# Load up the bigram and trigram models we trained earlier
bigram_model = Phrases.load('../models/spacy_bigram_model_all_PARSED_POS')
trigram_model = Phrases.load('../models/spacy_trigram_model_all_PARSED_POS')


trigram_dictionary = trigram_dictionary_POS
def vectorize_input(input_doc, bigram_model, trigram_model, trigram_dictionary):
    """
    (1) parse input doc with spaCy, apply text pre-proccessing steps, 
    (3) create a bag-of-words representation (4) create an LDA representation
    """
    
    # parse the review text with spaCy
    parsed_doc = nlp(input_doc)
    
    # lemmatize the text and remove punctuation and whitespace
    unigram_doc = [token.lemma_ for token in parsed_doc
                      if not punct_space(token)]
    
    # apply the first-order and secord-order phrase models
    bigram_doc = bigram_model[unigram_doc]
    trigram_doc = trigram_model[bigram_doc]
    
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_doc
                      if not term in stopwords]
    
    # create a bag-of-words representation
    doc_bow = trigram_dictionary.doc2bow(trigram_doc)

    # create an LDA representation
    document_lda = lda[doc_bow]
    return trigram_review, document_lda

def lda_top_topics(document_lda, topic_names, min_topic_freq=0.05):
    '''
    Print a sorted list of the top topics for a given LDA representation
    '''
    # sort with the most highly related topics first
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])
    
    for topic_number, freq in sorted_doc_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        print('*'*56) 
        print('{:50} {:.3f}'.format(topic_names[topic_number+1],
                                round(freq, 3)))
        print('*'*56)
        for term, term_freq in lda.show_topic(topic_number, topn=10):
            print(u'{:20} {:.3f}'.format(term, round(term_freq, 3)))
        print('\n\n')

def top_match_items(document_lda, topic_names, num_terms=100):
    '''
    Print a sorted list of the top topics for a given LDA representation
    '''
    # sort with the most highly related topics first
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])

    topic_number, freq = sorted_doc_lda[0][0], sorted_doc_lda[0][1]
    print('*'*56)
    print('{:50} {:.3f}'.format(topic_names[topic_number+1],
                            round(freq, 3)))
    print('*'*56)
    for term, term_freq in lda.show_topic(topic_number, topn=num_terms):
        print(u'{:20} {:.3f}'.format(term, round(term_freq, 3)))


def top_match_list(document_lda, topic_names, num_terms=500):
    # Take the above results and just save to a list of the top 500 terms in the topic
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])
    topic_number, freq = sorted_doc_lda[0][0], sorted_doc_lda[0][1]
    print('Highest probability topic:', topic_names[topic_number+1],'\t', round(freq, 3))
    top_topic_skills = []
    for term, term_freq in lda.show_topic(topic_number, topn=num_terms):
        top_topic_skills.append(term)
    return top_topic_skills

def common_skills(top_topic_skills, user_skills):
    return [item for item in top_topic_skills if item in user_skills]

def non_common_skills(top_topic_skills, user_skills):
    return [item for item in top_topic_skills if item not in user_skills]


with open('../data/sample_resume.txt', 'r') as infile:
    sample1 = infile.read()
with open('../data/sample_ds_resume2.txt', 'r') as infile:
    sample2 = infile.read()


def generate_common_skills(input_sample):
    user_skills, my_lda = vectorize_input(input_sample, bigram_model, trigram_model, trigram_dictionary)
    # top_items = top_match_items(my_lda, topic_names)
    # print(top_items)
    skills_list = top_match_list(my_lda, topic_names, num_terms=500)
    print("Top 40 skills user has in common with topic:")
    pprint(common_skills(skills_list, user_skills)[:100])
    print("\n\nTop 40 skills user DOES NOT have in common with topic:")
    pprint(non_common_skills(skills_list, user_skills)[:100])


user_skills, my_lda = vectorize_input(sample1, bigram_model, trigram_model, trigram_dictionary)
print(user_skills)


generate_common_skills(sample2)


from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import time
import requests
import random
import pandas as pd


# The logic is something like this:
# 1. Identify the types of jobs we want to get the job postings for. Let us say one of them is "backend engineer". 
# 2. Use Selenium Webdriver to load the page http://www.indeed.com/m/jobs?q=backend+engineer (You can add other query parameters here like location, which we are ignoring here).Get the html and use BeautifulSoup to parse the content. This main search page tells us 3 important things - (a) The first 10 jobs for this search and their URLs, (b) The next to the next page of 10 jobs and (c) How many pages we need to click through to load all the jobs for this search.
# 3. Write code to parse out the Job Urls in the first page. Every URL will be of this format: https://www.indeed.com/m/viewjob?jk=10bca72276277b22 . Load this page and get job title, location, salary, company name, summary and write one row per job into a CSV.
# 4. Use selenium to click the next link to go to the next page and do exactly what we did in (3). Do these N times where N is the number of pages.
# 5. Identify the next job type and repeat from (2).
# 6. Upload all these CSVs into a S3 bucket.
# 

# This is a very simple example of how to parse stuff in the main Search page.
start_url = "http://www.indeed.com/m/jobs?q=backend+engineer"
page = requests.get(start_url)
start_soup = BeautifulSoup(page.text, "html.parser")
print(start_soup.title.text)


# This is a very simple example of how to parse stuff out of an individual job page.
test_url="https://www.indeed.com/m/viewjob?jk=10bca72276277b22"
test_job_link_page = requests.get(test_url)
test_job_link_soup = BeautifulSoup(test_job_link_page.text, "html.parser")
#print(test_job_link_soup.body.p.text)
print('job title:', test_job_link_soup.body.p.b.text.strip())
print('company name:', test_job_link_soup.body.p.b.next_sibling.next_sibling.string.strip())
print('location:', test_job_link_soup.body.p.span.text.strip())
#print('summary:', test_job_link_soup.find(name="div", attrs={"id":"desc"}).text)


# Given a soup object, parse out all the job urls.
def extract_job_links(soup): 
  job_links = []
  for h in soup.find_all(name="h2", attrs={"class":"jobTitle"}):
      for a in h.find_all(name="a", attrs={"rel":"nofollow"}):
        job_links.append(a["href"])
  return(job_links)


# Given a list of job urls (links), parse out relevant content for all the job pages and store in a dataframe
def extract_job_listings(job_links):
    job_link_base_url="https://www.indeed.com/m/{}"
    job_listings=[]
    for job_link in job_links:
        j = random.randint(1000,2200)/1000.0
        time.sleep(j) #waits for a random time so that the website don't consider you as a bot
        job_link_url = job_link_base_url.format(job_link)
        #print('job_link_url:', job_link_url)
        job_link_page = requests.get(job_link_url)
        job_link_soup = BeautifulSoup(job_link_page.text, "html.parser")
        #print('job_link_text:', job_link_soup.text)
        #job_listings_df.loc[count] = extract_job_listing(job_link_url, job_link_soup)
        job_listings.append(extract_job_listing(job_link_url, job_link_soup))
    
    
    columns = ["job_url", "job_title", "company_name", "location", "summary", "salary"]
    job_listings_df = pd.DataFrame(job_listings, columns=columns)
    return job_listings_df

# Given a single job listing url and the corresponding page, parse out the relevant content to create an entry 
def extract_job_listing(job_link_url, job_link_soup):
    job_listing = []
    job_listing.append(job_link_url)
    job_listing.append(job_link_soup.body.p.b.text.strip())
    job_listing.append(job_link_soup.body.p.b.next_sibling.next_sibling.string.strip())
    job_listing.append(job_link_soup.body.p.span.text.strip())
    job_listing.append(job_link_soup.find(name="div", attrs={"id":"desc"}).text)
    job_listing.append("Not_Found")
    return job_listing
    
    #print(job_listing)        


# Given a single page with many listings, go to the individual job pages and store all the content to CSV
def parse_job_listings_to_csv(soup, fileName):
    job_links = extract_job_links(soup)
    job_posts = extract_job_listings(job_links)
    job_posts.to_csv(fileName, encoding='utf-8', index=False)


# A simple example to show how to use Selenium to go through the next links.
next_page_url_pattern="https://www.indeed.com/m/{}"
driver = webdriver.Chrome('/Users/Raghu/Downloads/chromedriver')
start_url = "http://www.indeed.com/m/jobs?q=data+analyst"
driver.set_page_load_timeout(15)
driver.get(start_url)
start_soup = BeautifulSoup(driver.page_source, "html.parser")

print('first page jobs:')
print(extract_job_links(start_soup))
#print start_soup.find(name='a', text='Next')
next_link=driver.find_elements_by_xpath("//a[text()='Next']")[0]
#print next_link
next_link.click()


# Use Selenium to go to do pagination - Click the next links (for now limit to only 5 next links)
driver = webdriver.Chrome('/Users/Raghu/Downloads/chromedriver')
start_url = "http://www.indeed.com/m/jobs?q=frontend+engineer"
driver.set_page_load_timeout(15)
driver.get(start_url)
start_soup = BeautifulSoup(driver.page_source, "html.parser")

parse_job_listings_to_csv(start_soup, "job_postings_0.csv")

for i in range(1,5):
    print('loading {} page'.format(str(i)))
    j = random.randint(1000,3300)/1000.0
    time.sleep(j) #waits for a random time so that the website don't consider you as a bot
    next_page_url = driver.find_elements_by_xpath("//a[text()='Next']")[0]
    page_loaded = True
    try:
        next_page_url.click()
    except TimeoutException:
        get_info = False
        driver.close()
    if page_loaded:
        soup=BeautifulSoup(driver.page_source)
        parse_job_listings_to_csv(soup, "job_postings_{}.csv".format(str(i)))

driver.close()





