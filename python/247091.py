# <center><b><font size=6>Text data pre-processing</font></b></center>
# 
# In this exercice, we shall load a database of email messages and pre-format them so that we can design automated classification methods or use off-the-shelf classifiers.
# 
# "What is there to pre-process?" you might ask. Well, actually, text data comes in a very noisy form that we, humans, have become accustomed to and filter out effortlessly to grasp the core meaning of the text. It has a lot of formatting (fonts, colors, typography...), punctuation, abbreviations, common words, grammatical rules, etc. that we might wish to discard before even starting the data analysis.
# 
# Here are some pre-processing steps that can be performed on text:
# 1. loading the data, removing attachements, merging title and body;
# 2. tokenizing - splitting the text into atomic "words";
# 3. removal of stop-words - very common words;
# 4. removal of non-words - punctuation, numbers, gibberish;
# 3. lemmatization - merge together "find", "finds", "finder".
# 
# The final goal is to be able to represent a document as a mathematical object, e.g. a vector, that our machine learning black boxes can process.
# 

# # Load the data
# 
# Let's first load the emails.
# 

import os
data_switch=1
if(data_switch==0):
    train_dir = '../data/ling-spam/train-mails/'
    email_path = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
else:
    train_dir = '../data/lingspam_public/bare/'
    email_path = []
    email_label = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir,d)
        email_path += [os.path.join(folder,f) for f in os.listdir(folder)]
        email_label += [f[0:3]=='spm' for f in os.listdir(folder)]
print("number of emails",len(email_path))
email_nb = 0 # try 8 for a spam example
print("email file:", email_path[email_nb])
print("email is a spam:", email_label[email_nb])
print(open(email_path[email_nb]).read())


# # Filtering out the noise
# 
# One nice thing about scikit-learn is that is has lots of preprocessing utilities. Like [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for instance, that converts a collection of text documents to a matrix of token counts.
# 
# - To remove stop-words, we set: `stop_words='english'`
# - To convert all words to lowercase: `lowercase=True`
# - The default tokenizer in scikit-learn removes punctuation and only keeps words of more than 2 letters.
# 

from sklearn.feature_extraction.text import CountVectorizer
countvect = CountVectorizer(input='filename', stop_words='english', lowercase=True)
word_count = countvect.fit_transform(email_path)


print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
print("Number of words:", len(words))
print("Document - words matrix:", word_count.shape)
print("First words:", words[0:100])


# # Even better filtering
# 
# That's already quite ok, but this pre-processing does not perform lemmatization, the list of stop-words could be better and we could wish to remove non-english words (misspelled, with numbers, etc.).
# 
# A slightly better preprocessing uses the [Natural Language Toolkit](https://www.nltk.org/https://www.nltk.org/). The one below:
# - tokenizes;
# - removes punctuation;
# - removes stop-words;
# - removes non-English and misspelled words (optional);
# - removes 1-character words;
# - removes non-alphabetical words (numbers and codes essentially).
# 

from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
class LemmaTokenizer(object):
    def __init__(self, remove_non_words=True):
        self.wnl = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.words = set(words.words())
        self.remove_non_words = remove_non_words
    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if(self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word)>1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.wnl.lemmatize(t) for t in word_list]

countvect = CountVectorizer(input='filename',tokenizer=LemmaTokenizer(remove_non_words=True))
word_count = countvect.fit_transform(email_path)
feat2word = {v: k for k, v in countvect.vocabulary_.items()}


print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
print("Number of words:", len(words))
print("Document - words matrix:", word_count.shape)
print("First words:", words[0:100])


# # Term frequency times inverse document frequency
# 
# After this first preprocessing, each document is summarized by a vector of size "number of words in the extracted dictionnary". For example, the first email in the list has become:
# 

mail_number = 0
text = open(email_path[mail_number]).read()
print("Original email:")
print(text)
#print(LemmaTokenizer()(text))
#print(len(set(LemmaTokenizer()(text))))
#print(len([feat2word[i] for i in word_count2[mail_number, :].nonzero()[1]]))
#print(len([word_count2[mail_number, i] for i in word_count2[mail_number, :].nonzero()[1]]))
#print(set([feat2word[i] for i in word_count2[mail_number, :].nonzero()[1]])-set(LemmaTokenizer()(text)))
emailBagOfWords = {feat2word[i]: word_count[mail_number, i] for i in word_count[mail_number, :].nonzero()[1]}
print("Bag of words representation (", len(emailBagOfWords), " words in dict):", sep='')
print(emailBagOfWords)
print("\nVector reprensentation (", word_count[mail_number, :].nonzero()[1].shape[0], " non-zero elements):", sep='')
print(word_count[mail_number, :])


# Counting words is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.
# 
# To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called `tf` for Term Frequencies.
# 
# Another refinement on top of `tf` is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.
# 
# This downscaling is called `tf–idf` for “Term Frequency times Inverse Document Frequency” and again, scikit-learn does the job for us with the [TfidfTransformer](scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) function.
# 

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit_transform(word_count)
tfidf.shape


# Now every email in the corpus has a vector representation that filters out unrelevant tokens and retains the significant information.
# 

print("email 0:")
print(tfidf[0,:])


# # Utility function
# 
# Let's put all this loading process into a separate file so that we can reuse it in other experiments.
# 

import load_spam
spam_data = load_spam.spam_data_loader()
spam_data.load_data()


spam_data.print_email(8)





