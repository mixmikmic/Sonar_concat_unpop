# # Non-Jargon Word Removal
# 
# In our approach to removing the words we know are not jargon, we take the following steps:
# 
# * Remove the proper nouns
# * Remove the n most common words
# * Remove anything non-alphabetical
# 

import spacy
import sys
import pprint as pp
import re

from spacy.en import English
from spacy.tokenizer import Tokenizer


nlp = spacy.load('en')


def strip_non_words(tokenized_text):
    return [token for token in tokenized_text if token.is_alpha==True]

# Takes in a bag of words and spits out that same bag of words but without the proper nouns
def strip_proper_nouns(tokenized_text):
    return [token for token in tokenized_text if token.tag_ != 'NNP' and token.tag_ != 'NNPS']

# Takes in a bag of words and removes any of them that are in the top n most common words
def strip_most_common_words(tokenized_text, n_most_common=10000):
    # Build the list of most common words
    most_common_words = []
    google_most_common_words_path = sys.path[1] + '/../Texts/google-10000-english-usa.txt'
    with open(google_most_common_words_path, 'r') as f:
        for i in range(n_most_common):
            most_common_words.append(f.readline().strip())
    # Remove anything in the n most common words
    return [token for token in tokenized_text if token.text.lower() not in most_common_words]

def strip_non_jargon_words(tokenized_text):
    text_no_proper_nouns = strip_proper_nouns(tokenized_text)
    text_no_non_words = strip_non_words(text_no_proper_nouns)
    text_no_common_words = strip_most_common_words(text_no_non_words)
    return text_no_common_words



def load_doc(filepath):
    # Open and read the file
    with open(file_path) as f:
        text = f.read()
    doc = nlp(text)
    return doc

# Here's an example with Gladwell. Shout out to Gladwell!

file_path = sys.path[1] + '/../Rule3/gladwell_latebloomers.txt'
gladwell_doc = load_doc(file_path)


pp.pprint(strip_non_jargon_words(gladwell_doc))


# ## Read and parse XML
# 

# import xml.etree.ElementTree as etree
from lxml import etree, objectify

root = etree.parse('VUAMC.xml')

## Cleanup xml schema/namespaces from tags ##    
for elem in root.getiterator():
    if not hasattr(elem.tag, 'find'): continue  # (1)
    i = elem.tag.find('}')
    if i >= 0:
        elem.tag = elem.tag[i+1:]
objectify.deannotate(root, cleanup_namespaces=True)


# ## Traverse XML tree and extract sentences containing similes
# 

import pandas as pd

def extract_similes(root):
    rows = []
    for sent in root.findall('.//s'): # scan all sentences
        text = ''
        mflag = ''
        mrw = ''
        for word in sent.findall('.//w'): # for each word in sentence
            aseg = word.find('.//seg')
            if aseg is not None:
                if not aseg.text or not aseg.text.strip():
                    continue
                ft = aseg.text.strip()#.encode('UTF-8')
                if aseg.get('function') == 'mFlag': # flag for similes
                    mflag += ' ' + ft
                    text += ' ' + ft
                elif aseg.get('function') == 'mrw' and not (not mflag): # start collecting keywords only after mflag
                    mrw += ' ' + ft
                    text += ' ' + ft
            elif not (not word.text):
                text += ' ' + word.text.strip()#.encode('UTF-8')

        text = text.strip()
        mrw = mrw.strip()
        mflag = mflag.strip()
        if not (not mflag): # we are only interested in similes; for metaphors: if not mflag 
            rows.append([mflag, mrw, text])
    df = pd.DataFrame(rows)
    df.columns = ['mflag', 'mrw', 'sentence']
    return df
   


# ## Go to town 
# 

df = extract_similes(root)
df.to_csv('similes.csv')





# # Rule 4: Never use the passive where you can use the active.
# 
# We are using the dependency parser from Spacy to check for clauses, nouns and verbs which are tagged with a "passive tag". We are simply calculating the number of occurences of sentences that are in the passive voice and returning a decimal number denoting the percentage of articles in the article that are in the passive voice.
# 

from spacy.en import English

parser = English()


def rule4_ranges_in_text(article, parser):
    '''This function accepts a string of sentences and prints them out classifying them into active or passive.    It returns a list of tuples in the format (starting_char_of_passive_sentence, length_of_passive_sentence)    of sentences that are passive.''' 

    edited_article = remove_quotes_from_text(article)
    
    parse = parser(edited_article)
    
    passive_list = []
    

    for sentence in parse.sents:
        sent = str(sentence)
        hasPassive = False
        passive_indicators = []
        for word in sentence:
            if word.dep_ in ['nsubjpass', 'csubjpass', 'auxpass']:
                passive_indicators.append((word.dep_, word.text))
                hasPassive = True
        if hasPassive:
            passive_list.append((article.find(sent), len(sent)))
            print("Passive Voice Sentence: {0}.\nPassive Voice indicators: {1}".format(sentence, passive_indicators))
        else:
            continue
            
    return passive_list

def remove_quotes_from_text(text):
    # Check for all types of quotes
    import re
    quote_regex = r'"(.*?)"|“(.*?)”'
    text = re.sub(quote_regex, '', text)
    return text


##Using same article as Gabe's Rule 3 to check my code

# with open('gladwell_latebloomers.txt', 'r') as f:
#     rule4_percentage = rule4_ranges_in_text(f.read(), parser)
#     print(rule4_percentage)





# # Feature Engineering and Classification for Jargon Words
# 

from collections import Counter
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from random import shuffle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from nltk.corpus import wordnet as wn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.externals import joblib


def generate_features(candidate):
    features = Counter()
#     print(candidate)
    candidate = str(str(candidate).lower().encode("ascii", "ignore"))
    features  = get_letter_combinations(candidate, features, 1)
    features  = get_letter_combinations(candidate, features, 2)
    features  = get_letter_combinations(candidate, features, 3)
    features  = get_letter_combinations(candidate, features, 4)
    features['first'] = candidate[:1]
#     features['second'] = word[1:2] # get the 'h' in Charlie?
    features['last_three'] = candidate[-3:]
    features['last_two'] = candidate[-2:]
    features['first_three'] = candidate[:3]
    features['name_len_id'] = len(candidate)
#     features['repeating_letters'] = get_first_repeating_letters(word)
#     features['continuous_vowels'] = get_first_repeating_vowels(word)
#     features['has_letters'] = has_letters(word, 'yzwx')
    return dict(features)
generate_features('veni vidi vici') 


def get_letter_combinations(candidate, features, number):
    candidate = candidate.replace(" ", "")
    if len(candidate) < number:
        return features
    else:
        for index in range(0, len(candidate), number):
            features[candidate[index:index + number]] += 1
        return features



def get_wordnet_definition(candidate):
    words = word_tokenize(candidate)
    for word in words:
        synsets = wn.synsets(word)
        
    
    


# This function allows experimentation with different feature definitions
# items is a list of (key, value) pairs from which features are extracted and training sets are made
# Feature sets returned are dictionaries of features

# This function also optionally returns the names of the training, development, 
# and test data for the purposes of error checking

def create_training_sets (feature_function, items, return_items=False):
    # Create the features sets.  Call the function that was passed in.
    # For names data, key is the name, and value is the gender
    shuffle(items)
    featuresets = [(feature_function(key), value, key) for (key, value) in items]
    
    # Divided training and testing in thirds.  Could divide in other proportions instead.
    fifth = int(float(len(featuresets)) / 5.0)
    
    train_set, dev_set, test_set = featuresets[0:fifth*4], featuresets[fifth*4:fifth*5], featuresets[fifth*4:]
    train_items, dev_items, test_items = items[0:fifth*4], items[fifth*4:fifth*5], items[fifth*4:]
    if return_items == True:
        return train_set, dev_set, test_set, train_items, dev_items, test_items
    else:
        return train_set, dev_set, test_set


dataset_df = pd.read_csv("jargon_dataset.csv")


print(dataset_df.head())
items = []
print(dataset_df["Jargon_Terms"][0])
for index in range(len(dataset_df)):
    items.append((dataset_df["Jargon_Terms"][index], dataset_df["is_Jargon"][index]))
print(items)
    


train_set, dev_set, test_set, train_items, dev_items, test_items = create_training_sets(generate_features, items, True)
# cl4 = nltk.NaiveBayesClassifier.train(train_set4)
# This is code from the NLTK chapter
errors = []
# print ("%.3f" % nltk.classify.accuracy(cl4, dev_set4))


# print ("%.3f" % nltk.classify.accuracy(cl4, test_set4))
# print(train_set4[0][1])
# print(test_set4[:2])
test_set_features = np.asarray([item[0] for item in test_set])
train_set_features = np.asarray([item[0] for item in train_set])
test_set_names = np.asarray([item[2] for item in test_set])
train_set_names = np.asarray([item[2] for item in train_set])
test_set_labels = np.asarray([item[1] for item in test_set])
train_set_labels = np.asarray([item[1] for item in train_set])

train_set = {}
train_set["features"] = train_set_features
train_set["names"] = train_set_names
train_set["labels"] = train_set_labels

test_set = {}
test_set["features"] = test_set_features
test_set["names"] = test_set_names
test_set["labels"] = test_set_labels

print(test_set["names"][0])



def create_manual_test_set(manual_list, generate_features):
    manual_set = [(generate_features(key), value, key) for (key, value) in manual_list]
    test_set_features = np.asarray([item[0] for item in manual_set])
    test_set_labels = np.asarray([item[1] for item in manual_set])
    test_set_names = np.asarray([item[2] for item in manual_set])
    manual_set_dict = {}
    manual_set_dict["features"] = test_set_features
    manual_set_dict["names"] = test_set_names
    manual_set_dict["labels"] = test_set_labels
    return manual_set_dict



class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
#         print(data_dict)
        return data_dict[self.key]


kaggle_classifier = Pipeline([('union', FeatureUnion(
                                    transformer_list=[

                                        # Pipeline for pulling features from the post's subject line
                                        ('names', Pipeline([
                                            ('selector', ItemSelector(key='names')),
                                            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2,3), sublinear_tf=True)),
                                        ])),

                                        # Pipeline for standard bag-of-words model for body
                                        ('features', Pipeline([
                                            ('selector', ItemSelector(key='features')),
                                            ('dict', DictVectorizer(sparse='False'))
                                        ])),

                                    ],

                                    # weight components in FeatureUnion
                                    transformer_weights={
                                        'names': 0.2,
                                        'features': 0.8,
                                    },
                                )),

                                # Use a SVC classifier on the combined features
                                ('svc', LinearSVC()),
                            ])
kaggle_classifier = kaggle_classifier.fit(train_set,train_set_labels)
    
kaggle_predictions = kaggle_classifier.predict(test_set)

accuracy_score(test_set_labels, kaggle_predictions)


# kaggle_classifier.predict(["viz a viz", "tete a tete", "locker", "scrum", "scalar", "table"])
manual_list = [("dividend", True)]
manual_test_dict = create_manual_test_set(manual_list, generate_features)
print(manual_test_dict)
manual_predictions = kaggle_classifier.predict(manual_test_dict)
print(manual_predictions)


kaggle_classifier = Pipeline([('tfidfvect', TfidfVectorizer(analyzer='char', ngram_range=(2,4), sublinear_tf=True)),
#                                     ('feat',SelectKBest(chi2, 5)),
                                    ('classifier', LinearSVC())
                                   ])
kaggle_classifier = kaggle_classifier.fit(train_set_names,train_set_labels)
    
kaggle_predictions = kaggle_classifier.predict(test_set_names)

accuracy_score(test_set_labels, kaggle_predictions)


def test_manual_predictions(manual_list):
    manual_test_dict = create_manual_test_set(manual_list, generate_features)
    manual_predictions = kaggle_classifier.predict(manual_test_dict)
    print(manual_predictions)
    
    


# The True/False bit of the Tuple only needs to be accurate if you plan to test the accuracy using accuracy_score, 
# else it isn't considered.
manual_list = [("viz a viz", True), ("tete a tete",False), ("bottomline", True), ("ibuprofin", True), ("uninterested", False)]
test_manual_predictions(manual_list)


joblib.dump(kaggle_classifier, 'linear_jargon_classifier.pkl') 





# # Rule 3: If it is possible to cut a word out, always cut it out.
# 
# This list of unnecessary words comes from the Purdue Online Writing Lab articles on [eliminating words](https://owl.english.purdue.edu/owl/resource/572/02/) and [avoiding common pitfalls.](https://owl.english.purdue.edu/owl/resource/572/04/) Here, we are simply calculating the number of occurences of removable words and putting it in a nice data frame.
# 

import re
import pandas as pd
import pprint as pp


def load_csv(filename):
    try:
        f = open(filename)
    except:
        pp.pprint('Bad filename ' + filename)
        return None
    words = f.read().split(',')
    return words

def regex_for_word(word):
    return word.replace('*', '[a-zA-Z]+')

# Save the regexes to find unnecessary words as a global variable
unnecessary_regexes = load_csv('unnecessary_words.csv')


def remove_quotes_from_text(text):
    # Check for all types of quotes
    quote_regex = r'"(.*?)"|“(.*?)”'
    text = re.sub(quote_regex, '', text)
    return text

def find_phrases_in_text(text, phrases):
    phrase_list = []
    for phrase in phrases:
        phrase_count = len(re.findall(regex_for_word(phrase), text, flags=re.IGNORECASE))
        if phrase_count is not 0:
            phrase_list.append((phrase, phrase_count))
    return phrase_list

def unnecessary_phrase_count_in_text(text):
    text = remove_quotes_from_text(text)
    text_phrases = find_phrases_in_text(text, unnecessary_regexes)
    frame = pd.DataFrame(text_phrases)
    frame.columns = ['PHRASE', 'COUNT']
    return frame

# This article can be found here:
# http://www.newyorker.com/magazine/2008/10/20/late-bloomers-malcolm-gladwell
def test_on_gladwell():
    with open('gladwell_latebloomers.txt', 'r') as f:
        rule3_count = unnecessary_phrase_count_in_text(f.read())
        print(rule3_count)


def rule3_ranges_in_text(text):
    phrase_location_list = []
    for phrase in unnecessary_regexes:
        phrase_matches = re.finditer(regex_for_word(phrase), text, flags=re.IGNORECASE)
        for phrase_match in phrase_matches:            
            phrase_location_list.append(phrase_match.span())
    return [(start, end - start) for (start, end) in phrase_location_list]


# # Rule 2: Never use a long word where a short word will do
# 
# We are going to take a loose interpretation of this. Instead of word length, we will use both the number of syllables and the order of where it appears in order in a frequency distribution of words. If we just did number of syllables, we would, for example, always replace the word `therefore` with `thus`, which is not in the spirit of the problem. 
# 
# So let's get cracking on this score!
# 

import collections
import math
import nltk
import pprint as pp
import re
import spacy
import sys
import time

from google_ngram_downloader import readline_google_store
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.wsd import lesk


# Some of this code was directly copied from Rule #5
# We use it to minimize the words we need to check for because it's a computationally heavy task
def google_most_common_words(n_most_common=10000):
    google_most_common_words_path = sys.path[1] + '/../Texts/google-10000-english-usa.txt'
    most_common_words = []
    with open(google_most_common_words_path, 'r') as f:
        for i in range(n_most_common):
            most_common_words.append(f.readline().strip())
    return most_common_words

def is_non_replaceable_word(word):
    return (word.isalpha() is False) or (word in most_common_words)


# Global variables
syllable_dict = cmudict.dict()
most_common_words = google_most_common_words()


# This number comes from Google's blog
# https://research.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html
# TODO: If there's time, confirm this number
NGRAM_TOKEN_COUNT = 1024908267229

# Shout out to Quora for this snippet of code
# https://www.quora.com/Is-there-any-Google-Ngram-API-for-Python
def find_google_ngrams_word_count(word, time_function=False, verbose=False):
    if time_function == True:
        time1 = time.time()

    count = 2 # Set this to a minimum of 2 so we don't get a divide by zero error
    # TODO: Consider how we want to deal with capitalization
    fname, url, records = next(readline_google_store(ngram_len=1, indices=word[0]))
    # If we use the verbose settings, occaisionally print out the record
    verbosity_count = 1000000000
    earliest_year = 1950
    i = 0
    try:
        record = next(records)
        while record.ngram != word:
            record = next(records)
            if verbose == True and i%verbosity_count == 0:
                print(record)
            i += 1
        while record.ngram == word:
            if record.year >= earliest_year:
                count += record.match_count
                if verbose == True:
                    print(record)
            record = next(records)
    except StopIteration:
        pass
    # Default to 1 so our program doesn't crash
    if count == 0:
        count = 1
    if time_function == True:
        time2 = time.time()
    print('Total seconds for ' + word + ': ' + str(int((time2-time1))))
    return count

def find_frequency_score(word):
    unigram_count = find_google_ngrams_word_count(word, time_function=True)
    percent_occurrence = unigram_count/NGRAM_TOKEN_COUNT
    # Get the log of the frequency to make our number manageable
    freq_val = math.log(percent_occurrence)
    max_ngram_val = math.log(1/NGRAM_TOKEN_COUNT)
    relative_freq = ((freq_val - max_ngram_val)/(-max_ngram_val))
    return round(relative_freq, 5)


BIG_NUMBER = 18109831

def syllable_count(word):
    syllable_count = 0
    for word in word.split():
        if word in syllable_dict:
            # Shout out to StackOverflow for this snippet of code
            # http://stackoverflow.com/a/4103234/1031615
            syllable_count += [len(list(y for y in x if y[-1].isdigit())) for x in syllable_dict[word]][0]
            continue
        # If it's not in the dictionary count the number of vowels and ignore an e at the end not
        # preceded by another vowel. It's rough, but there will be few cases if any cases in which
        # a word is not in the CMU dictionary but in WordNet
        if word[-1] == 'e':
            word = word[:-1]
        word = re.sub(r'[^aeiou]', '', word)
        syllable_count += len(word)
    return max(syllable_count, 1)


def readability_for_word(word, ignore_common_words=False, use_ngrams=False):
    if word is None:
        return BIG_NUMBER 
    word = word.lower()
    # If it's in the top 10000 most common words, we assume it is readable enough
    if ignore_common_words is True and is_non_replaceable_word(word) is True:
        return 0
    syllables = syllable_count(word)
    if use_ngrams == False:
        return syllables
    freq_score = find_frequency_score(word)
    return syllables * freq_score


# Awesome. Now let's bust some synsets up in here!
# 

def synsets_for_tokens_in_tokenized_sentence(tokenized_sentence):
    sentence = [token.text for token in tokenized_sentence]
    synsets = [lesk(sentence, token.text, spacy_to_wordnet_pos(token.pos_)) for token in tokenized_sentence]
    for i in range(len(synsets)):
        # Get the lemmas of the word. Ignore if there is only one because it's just the root of the word
        if synsets[i] is not None and len(synsets[i].lemma_names()) > 1:
            synsets[i] = synsets[i].lemma_names()[1:]
        else:
            synsets[i] = None
    return synsets


def spacy_to_wordnet_pos(pos):
    # To see all the parts of speech spaCy uses, see the link below
    # http://polyglot.readthedocs.io/en/latest/POS.html
    if pos == 'ADJ':
        return wn.ADJ
    elif pos == 'ADV':
        return wn.ADV
    elif pos == 'NOUN':
        return wn.NOUN
    elif pos == 'VERB':
        return wn.VERB
    return None


# Returns an array of tuples. If the word cannot be replaced, the second value is the replacing word.
# If it cannot be replaced, it is None
def replaceable_word_in_tokenized_sentence(tokenized_sentence):
    sentence_words = [token.text for token in tokenized_sentence]
    sentence_alternatives = synsets_for_tokens_in_tokenized_sentence(tokenized_sentence)
    for i in range(len(sentence_alternatives)):
        alternatives_list = sentence_alternatives[i]
        if alternatives_list is not None:
            alternatives_list = [(readability_for_word(alt), alt) for alt in alternatives_list]
            sentence_alternatives[i] = min(alternatives_list)[1]
    # Get the minimum syllables among the alternatives
    
    words_and_alternatives = zip(tokenized_sentence, sentence_alternatives)
    replaceable_words = []
    
    
    for (token, alt) in words_and_alternatives:
        if readability_for_word(token.text, ignore_common_words=True) <= readability_for_word(alt):
            replaceable_words.append((token, None))
        else:
            replaceable_words.append((token, alt))
    return replaceable_words


def print_replaceable_words_marked_in_document(document, open_marker='{', close_marker='}'):
    checked_sentences = [replaceable_word_in_tokenized_sentence(sentence) for sentence in document.sents]
    new_document_text = ''
    for sent_array in checked_sentences:
        for word in sent_array:
            new_document_text += word[0].text_with_ws
            if word[1] is not None:
                new_document_text += open_marker + word[1].upper() + close_marker + ' '
    return new_document_text
                                         
def load_doc(filepath):
    # Open and read the file
    with open(filepath) as f:
        text = f.read()
    nlp = spacy.load('en')
    doc = nlp(text)
    return doc

def test_with_gladwell():
    file_path = sys.path[1] + '/../Rule3/gladwell_latebloomers.txt'
    gladwell_doc = load_doc(file_path)
    replaceable_words = print_replaceable_words_marked_in_document(gladwell_doc)
    print(replaceable_words)


def rule2_ranges_in_text(text, nlp=None):
    if nlp == None:
        nlp = spacy.load('en')
    document = nlp(text)
    checked_sentences = [replaceable_word_in_tokenized_sentence(sentence) for sentence in document.sents]
    checked_words = [word for sentence in checked_sentences for word in sentence]
    ranges = []
    character_count = 0
    for i in range(len(checked_words)):
        word, alt = checked_words[i]
        if alt is not None:
            ranges.append((character_count, len(word)))
        character_count += len(word.text_with_ws)
    return ranges


def marked_up_doc(document_str):
    replaceable_ranges = rule2_ranges_in_text(document_str)
    open_tag_indices = [(index, '<rule2>') for (index, length) in replaceable_ranges]
    close_tag_indices = [(index + length, '</rule2>') for (index, length) in replaceable_ranges]
    tag_dictionary = collections.defaultdict(list)
    for (index, tag) in  open_tag_indices + close_tag_indices:
        tag_dictionary[index].append(tag)

    new_document = ''
    character_count = 0
    for i in range(len(document_str)):
        index_tags = tag_dictionary[i]
        if len(index_tags) == 0:
            new_document += document_str[i]
            continue
        for tag in index_tags:
            new_document += tag
        new_document += document_str[i]
    return new_document

def is_closed_tag(tag):
    return tag[1] == '/'


get_ipython().system('head ./wiki/wiki-links-abridged.txt')


get_ipython().system('head ./wiki/wiki-links-simple-sorted.txt')


get_ipython().system('head ./wiki/wiki-titles-sorted.txt')


import networkx as nx


graph_dict = dict()
l = dict()

with open("./wiki/wiki-links-simple-sorted.txt", "r") as g, open("./wiki/wiki-titles-sorted.txt", "r") as f:
    for line in g:
        text = f.readline()[:-1]
#         print(line, text)
        vertex = int(line.split(": ")[0])
        edges = line.split(":")[1].split()
        edges = set([int(edge) for edge in edges])
#         print(graph_dict, edges)
        graph_dict[vertex] = edges
        l[vertex] = text
        if vertex%300000 == 0:
            print(vertex)
#         if vertex == 10:
#             break
#     print(g)
#     print(l)
    print("Done")
    G=nx.Graph(graph_dict)
    result = nx.closeness_centrality(G)


with open("../Texts/french_expressions.txt", "r") as f:
    for line in f:
        if len(line) > 1 and line[-2] in r'.]:)"' or ";" in line:
            continue
        print(line)


import pandas as pd


df = pd.read_csv("../Texts/latin_expressions.txt", sep='\t')


df.head()


# ## Rule 5 - Never use a foreign phrase, a scientific word, or a jargon word if you can think of an everyday English equivalent.
# 
# ### Algorithm
# 
# Gather domain specific articles 
# 




