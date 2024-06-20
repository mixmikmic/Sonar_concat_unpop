# # Part 2: Phrase Learning
# 
# If you haven't complete the **Part 1: Data Preparation**, please complete it before moving forward with **Part 2: Phrase Learning**. Part 2 requires files created from Part 1.
# 
# Please make sure you have __notebook__ and __nltk__ Python packages installed in the compute context you choose as kernel. For demonstration purpose, this series of notebooks uses the `local` compute context.
# 
# **NOTE**: Python 3 kernel doesn't include Azure Machine Learning Workbench functionalities. Please switch the kernel to `local` before continuing further. 
# 
# To install __notebook__ and __nltk__, please uncomment and run the following script.
# 

# !pip install --upgrade notebook
# !pip install --upgrade nltk


# ### Import Required Python Modules
# 
# `modules.phrase_learning` contains a list of Python user-defined Python modules to learn informative phrases that are used in this examples. You can find the source code of those modules in the directory of `modules/phrase_learning.py`.
# 

import pandas as pd
import numpy as np
import re, os, requests, warnings
from collections import (namedtuple, Counter)
from modules.phrase_learning import (CleanAndSplitText, ComputeNgramStats, RankNgrams, ApplyPhraseRewrites,
                            ApplyPhraseLearning, ApplyPhraseRewritesInPlace, ReconstituteDocsFromChunks,
                            CreateVocabForTopicModeling)
from azureml.logging import get_azureml_logger
warnings.filterwarnings("ignore")


run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.part2-phrase-learning','true')


# ## Access trainQ and testQ from Part 1
# 
# As we have prepared the _trainQ_ and _testQ_ from the `Part 1: Data Preparation`, we retrieve the datasets here for the further process.
# 
# _trainQ_ contains 5,153 training examples and _testQ_ contains 1,735 test examples. Also, there are 103 unique answer classes in both datasets.
# 

# load non-content bearing function words (.txt file) into a Python dictionary. 
def LoadListAsHash(fileURL):
    response = requests.get(fileURL, stream=True)
    wordsList = response.text.split('\n')

    # Read in lines one by one and strip away extra spaces, 
    # leading spaces, and trailing spaces and inserting each
    # cleaned up line into a hash table.
    listHash = {}
    re1 = re.compile(' +')
    re2 = re.compile('^ +| +$')
    for stringIn in wordsList:
        term = re2.sub("",re1.sub(" ",stringIn.strip('\n')))
        if term != '':
            listHash[term] = 1
    return listHash


workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')

# paths to trainQ, testQ and function words.
trainQ_path = os.path.join(workfolder, 'trainQ_part1')
testQ_path = os.path.join(workfolder, 'testQ_part1')
function_words_url = 'https://bostondata.blob.core.windows.net/stackoverflow/function_words.txt'

# load the training and test data.
trainQ = pd.read_csv(trainQ_path, sep='\t', index_col='Id', encoding='latin1')
testQ = pd.read_csv(testQ_path, sep='\t', index_col='Id', encoding='latin1')

# Load the list of non-content bearing function words.
functionwordHash = LoadListAsHash(function_words_url)


# ## Clean and Split the Text
# 
# The CleanAndSplitText function from __phrase_learning__ takes as input a list where each row element is a single cohesive long string of text, i.e. a "question". The function first splits each string by various forms of punctuation into chunks of text that are likely sentences, phrases or sub-phrases. The splitting is designed to prohibit the phrase learning process from using cross-sentence or cross-phrase word strings when learning phrases.
# 
# The function returns a table where each row represents a chunk of text from the questions. The `DocID` coulmn indicates the original row index from associated question in the input from which the chunk of text originated. The `DocLine` column contains the original text excluding the punctuation marks and `HTML` markup that have been during the cleaning process. The `Lowercase Taxt` column contains a fully lower-cased version of the text in the `CleanedText` column.
# 

CleanedTrainQ = CleanAndSplitText(trainQ)
CleanedTestQ = CleanAndSplitText(testQ)


CleanedTrainQ.head(5)


# ## Learn Informative Phrases 
# The phrases can be treated as single compound word units in down-stream processes such as discriminative training. To learn the phrases, we have implemented the basic framework for key phrase learning as described in the paper entitled ["Modeling Multiword Phrases with Constrained Phrases Tree for Improved Topic Modeling of Conversational Speech"](http://people.csail.mit.edu/hazen/publications/Hazen-SLT-2012.pdf) which was originally presented in the 2012 IEEE Workshop on Spoken Language Technology. Although the paper examines the use of the technology for analyzing human-to-human conversations, the techniques are quite general and can be applied to a wide range of natural language data including news stories, legal documents, research publications, social media forum discussions, customer feedback forms, product reviews, and many more.
# 
# `ApplyPhraseLearning` module takes the following arguments:
# - `textData`: array, a list of text data.
# - `learnedPhrases`: array, a list of learned phrases. For initialization, an empty list should be given.
# - `maxNumPhrases`: int, (default=200), maximium number of phrases to learn. If you want to test the code out quickly then set this to a small value (e.g. 100) and set `verbose` to True when running the quick test.
# - `maxPhraseLength`: int, (default=7), maximum number of words allowed in the learned phrases.
# - `maxPhrasesPerIter`: int, (default=50), maximum number of phrases to learn per iteration. Increasing this number may speed up processing but will affect the ordering of the phrases learned and good phrases could be by-passed if the maxNumPhrases is set to a small number.
# - `minCount`: int, (default=5), minimum number of times a phrase must occur in the data to be considered during the phrase learning process.
# - `functionwordHash`: dict, (default={}), a precreated hash table containing the list of function words used during phrase learning. 
# - `blacklistHash`: dict, (default={}), a precreated hash table containing the list of black list words to be ignored during phrase learning.
# - `verbose`: boolean, (default=False). If verbose=True, it prints out the learned phrases to stdout buffer while its learning. This will generate a lot of text to stdout, so best to turn this off except for testing and debugging.
# 

# Initialize an empty list of learned phrases
# If you have completed a partial run of phrase learning
# and want to add more phrases, you can use the pre-learned 
# phrases as a starting point instead and the new phrases
# will be appended to the list
learnedPhrasesQ = []

# Create a copy of the original text data that will be used during learning
# The copy is needed because the algorithm does in-place replacement of learned
# phrases directly on the text data structure it is provided
phraseTextDataQ = []
for textLine in CleanedTrainQ['LowercaseText']:
    phraseTextDataQ.append(' ' + textLine + ' ')

# Run the phrase learning algorithm.
ApplyPhraseLearning(phraseTextDataQ, learnedPhrasesQ, maxNumPhrases=200, maxPhraseLength=7, maxPhrasesPerIter=50,
                    minCount=5, functionwordHash=functionwordHash)

# Add text with learned phrases back into data frame
CleanedTrainQ['TextWithPhrases'] = phraseTextDataQ

# Apply the phrase learning to test data.
CleanedTestQ['TextWithPhrases'] = ApplyPhraseRewritesInPlace(CleanedTestQ, 'LowercaseText', learnedPhrasesQ)


print("\nHere are some phrases we learned in this part of the tutorial: \n")
print(learnedPhrasesQ[:20])


# ## Reconstruct the Full Processed Text
# 
# After replacing the text with learned phrases, we reconstruct the sentences from the chunks of text and insert the sentences in the `TextWithPhrases` field.  
# 

# reconstitue the text from seperated chunks.
trainQ['TextWithPhrases'] = ReconstituteDocsFromChunks(CleanedTrainQ, 'DocID', 'TextWithPhrases')
testQ['TextWithPhrases'] = ReconstituteDocsFromChunks(CleanedTestQ, 'DocID', 'TextWithPhrases')


# ## Tokenize Text with Learned Phrases
# 
# We learn a vocabulary by considering some text exclusion criteria, such as stop words, non-alphabetic words, the words below word count threshold, etc. 
# 
# `TokenizeText` module breaks the reconstituted text into individual tokens and excludes any word that doesn't exist in the vocabulary.
# 

def TokenizeText(textData, vocabHash):
    tokenizedText = ''
    for token in textData.split():
        if token in vocabHash:
            tokenizedText += (token.strip() + ',')
    return tokenizedText.strip(',')


# create the vocabulary.
vocabHashQ = CreateVocabForTopicModeling(trainQ['TextWithPhrases'], functionwordHash)

# tokenize the text.
trainQ['Tokens'] = trainQ['TextWithPhrases'].apply(lambda x: TokenizeText(x, vocabHashQ))
testQ['Tokens'] = testQ['TextWithPhrases'].apply(lambda x: TokenizeText(x, vocabHashQ))


trainQ[['AnswerId', 'Tokens']].head(5)


# ## Save Outputs to a Share Directory in the Workbench
# 

trainQ.to_csv(os.path.join(workfolder, 'trainQ_part2'), sep='\t', header=True, index=True, index_label='Id')
testQ.to_csv(os.path.join(workfolder, 'testQ_part2'), sep='\t', header=True, index=True, index_label='Id')


# # Part 1: Data Preparation
# 
# Please make sure you have __notebook__ and __nltk__ Python packages installed in the compute context you choose as kernel. For demonstration purpose, this series of notebooks uses the `local` compute context.
# 
# **NOTE**: Python 3 kernel doesn't include Azure Machine Learning Workbench functionalities. Please switch the kernel to `local` before continuing further. 
# 
# To install __notebook__ and __nltk__, please uncomment and run the following script.
# 

# !pip install --upgrade notebook
# !pip install --upgrade nltk


# ### Import Required Python Modules
# 

import pandas as pd
import numpy as np
import re, os, gzip, requests, warnings
from azureml.logging import get_azureml_logger
warnings.filterwarnings("ignore")


run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.part1-data-preparation','true')


# ## Access Sample Data
# 
# In this example, we have collected a set of Q&A pairs from Stack Overflow site tagged as `JavaScript` questions. The data contains 1,201 original Q&A pairs as well as many duplicate questions, i.e. new questions that Stack Overflow users have linked back to pre-existing Q&A pairs that effectively provide answers to these new questions. The data schema of the original questions (Q), duplicate questions (D), and answers (A) can be found in the following table:
# 
# | Dataset | Field | Type | Description
# | ----------|------------|------------|--------
# | question (Q) | Id | String | The unique question ID (primary key)
# |  | AnswerId | String | The unique answer ID per question
# |  | Text0 | String | The raw text data including the question's title and body
# |  | CreationDate | Timestamp | The timestamp of when the question has been asked
# | dupes (D) | Id | String | The unique duplication ID (primary key)
# |  | AnswerId | String | The answer ID associated with the duplication
# |  | Text0 | String | The raw text data including the duplication's title and body
# |  | CreationDate | Timestamp | The timestamp of when the duplication has been asked
# | answers (A) | Id | String | The unique answer ID (primary key)
# |  | text0 | String | The raw text data of the answer
# 
# The datasets are compressed and stored in Azure Blob storage as `.tsv.gz` files and this section provides you the code to retreive the data in the notebook.
# 

# load raw data from a .tsv.gz file into Pandas data frame.
def read_csv_gz(url, **kwargs):
    df = pd.read_csv(gzip.open(requests.get(url, stream=True).raw, mode='rb'), sep='\t', encoding='utf8', **kwargs)
    return df.set_index('Id')


# URLs to Original questions, Duplications, and Answers.
questions_url = 'https://bostondata.blob.core.windows.net/stackoverflow/orig-q.tsv.gz'
dupes_url = 'https://bostondata.blob.core.windows.net/stackoverflow/dup-q.tsv.gz'
answers_url = 'https://bostondata.blob.core.windows.net/stackoverflow/ans.tsv.gz'

# load datasets.
questions = read_csv_gz(questions_url, names=('Id', 'AnswerId', 'Text0', 'CreationDate'))
dupes = read_csv_gz(dupes_url, names=('Id', 'AnswerId', 'Text0', 'CreationDate'))
answers = read_csv_gz(answers_url, names=('Id', 'Text0'))


# To provide some example, here are the first five rows of the __questions__ table:
# 

questions.head(5)


# Here is the full text of one __original__ question, whose is `Id` is `220231`. The `AnswerId` associated with this question is `220233`.
# 

# This text include the HTML code.
print(questions["Text0"][220231])


# Here is the full text of the __answer__ associated with the above original question:
# 

print(answers["Text0"][220233])


# __Duplicate__ questions share the same `AnswerId` as the original question they link to. Here is the first duplicate question linked to the above original question:
# 

print(dupes.query("AnswerId == 220233").iloc[0]["Text0"])


# ## Pre-process Text Data
# 
# ### Clean up text
# 
# The raw data is in `HTML` format and needs to be cleaned up for any further analysis. We exclude HTML tags, links and code snippets from the data.
# 

# remove embedded code chunks, HTML tags and links/URLs.
def clean_text(text):
    global EMPTY
    EMPTY = ''
    
    if not isinstance(text, str): 
        return text
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)
    
    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    return re.sub('<[^>]+>', EMPTY, text)


for df in (questions, dupes, answers):
    df['Text'] = df['Text0'].apply(clean_text).str.lower()
    df['NumChars'] = df['Text'].str.len()


# ### Set data selection criteria
# 
# To obtain the high quality datasets for phrase learning and model training, we requires a minimum length of characters in the text field. Different thresholds are considered for original questions, duplications, and answers, respectively. Also, each Q&A pair in our set must have a minimum of 3 additional semantically equivalent duplicate questions linked to it. 
# 

# find the AnswerIds has at least 3 dupes.
def find_answerId(answersC, dupesC, num_dupes):
       
    countHash = {}
    for i in dupesC.AnswerId:
        if i not in answersC.index.values:
            continue
        if i not in countHash.keys():
            countHash[i] = 1
        else:
            countHash[i] += 1
            
    countHash = {k: v for k, v in countHash.items() if v >= num_dupes}
    commonAnswerId = countHash.keys()
    
    return commonAnswerId

# extract data based on the selection criteria.
def select_data(questions, dupes, answers):
    # exclude the records without any text
    questions_nz = questions.query('NumChars > 0')
    dupes_nz = dupes.query('NumChars > 0')
    answers_nz = answers.query('NumChars > 0')

    # get the 10th percentile of text length as the minimum length of characters to consider in the text field
    minLenQ = questions_nz.quantile(.1)['NumChars']
    minLenD = dupes_nz.quantile(.1)['NumChars']
    minLenA = answers_nz.quantile(.1)['NumChars']
    
    # eliminate records with text less than the minimum length
    questionsC = questions.query('NumChars >' + str(int(minLenQ)))
    dupesC = dupes.query('NumChars >' + str(minLenD))
    answersC = answers.query('NumChars >' + str(minLenA))
    
    # remove the records in dupesC whose questionId has already existed in questionsC
    duplicatedIndex = list(set(questionsC.index).intersection(set(dupesC.index)))
    dupesC.drop(duplicatedIndex, inplace=True)
    
    # make sure Questions 1:1 match with Answers 
    matches = questionsC.merge(answersC, left_on = 'AnswerId', right_index = True)
    questionsC = matches[['AnswerId', 'Text0_x', 'CreationDate', 'Text_x', 'NumChars_x']]
    questionsC.columns = ['AnswerId', 'Text0', 'CreationDate', 'Text', 'NumChars']

    answersC = matches[['Text0_y', 'Text_y', 'NumChars_y']]
    answersC.index = matches['AnswerId']
    answersC.columns = ['Text0', 'Text', 'NumChars']
    
    # find the AnswerIds has at least 3 dupes
    commonAnswerId = find_answerId(answersC, dupesC, 3)
    
    # select the records with those AnswerIds
    questionsC = questionsC.loc[questionsC.AnswerId.isin(commonAnswerId)]
    dupesC = dupesC.loc[dupesC.AnswerId.isin(commonAnswerId)]
    
    return questionsC, dupesC


# some questions have been linked to multiple AnswerIds.
# we keep the first AnswerId associated with that question and remove the rest.
questions = questions.groupby(questions.index).first()
dupes = dupes.groupby(dupes.index).first()

# execute the data selection function on questions, dupes and answers.
questionsC, dupesC = select_data(questions, dupes, answers)


# ## Prepare Training and Test datasets
# 
# In this example, we retain original question and 75% of the duplicate questions for training, and hold-out the most recently posted 25% of duplicate questions as test data. The training and test data are split by `CreationDate`.
# 
# - training set = Original questions + 75% of oldest Duplications per original question
# - test set = remaining 25% of Duplications per original question
# 

# split Original questions and their Duplications into training and test sets.
def split_data(questions, dupes, frac):
    trainQ = questions
    testQ = pd.DataFrame(columns = dupes.columns.values) # create an empty data frame

    for answerId in np.unique(dupes.AnswerId):
        df = dupes.query('AnswerId == ' + str(answerId))
        totalCount = len(df)
        splitPoint = int(totalCount * frac)
        dfSort = df.sort_values(by = ['CreationDate'])
        trainQ = trainQ.append(dfSort.head(splitPoint)) # oldest N percent of duplications
        testQ = testQ.append(dfSort.tail(totalCount - splitPoint))

    # convert data type to int
    testQ[["AnswerId", "NumChars"]] = testQ[["AnswerId", "NumChars"]].astype(int) 
    # rename the index 
    testQ.index.rename("Id", inplace=True)
    
    return trainQ, testQ


trainQ, testQ = split_data(questionsC, dupesC, 0.75)


trainQ.head(5)


# ## Select Subsets with Sufficient Training Questions per Answer Class
# 
# In our past experiments, we notice that some Q&A pairs only link to a small number of duplicate questions. This means those answer classes may contain an insufficient amount of examples to train an accurate model. We examine the effect of the number of duplicate questions available for training for each Q&A pair. 
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/training_size.PNG?token=APoO9rnKXamwVdXu8luA_Dd28UUBncwrks5ZwtRowA%3D%3D">
# 
# The above Figure shows results for questions relative to the number of training examples available for the correct Q&A pair that should be returned. Most of our Q&A pairs (857 out of 1201) have 5 or fewer known duplicate questions available for training. Performance on these questions is relatively weak, with the correct Q&A pair landing in the top 10 results less than 40% of the time. However, when greater numbers of duplicate questions are available for training, performance improves dramatically; when Q&A pairs have 50 or more duplicate questions available for training, the classification model places these pairs in the top 10 of the retrieved results 98% of the time when they correctly match the query. The most duplicated question contains 962 duplications. 
# 
# For the study in this notebook, we only consider the answer classes that have more than 13 training questions (original and duplicate questions) in this notebook. This reduces the entire dataset to 5,153 training questions, 1,735 test questions, and 103 unique answer classes.
# 

countPerAns = pd.DataFrame({"NumTrain" : trainQ.groupby("AnswerId").size()})
trainQwithCount = trainQ.merge(countPerAns, left_on="AnswerId", right_index=True)
testQwithCount = testQ.merge(countPerAns, left_on="AnswerId", right_index=True)

# for each Answer class, we request more than 13 training questions.
trainQ = trainQwithCount[trainQwithCount["NumTrain"] > 13]
testQ = testQwithCount[testQwithCount["NumTrain"] > 13]


print("# of training examples: " + str(len(trainQ)))
print("# of testing examples: " + str(len(testQ)) + "\n")
print("A quick glance of the training data: \n")
trainQ[["AnswerId", "Text"]].head(5)


# ## Save Outputs to a Share Directory in the Workbench
# 
# Azure Machine Learning Workbench provides a flexible way of saving intermediate files. `os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')` retrieves a share directory where the files are stored. Those files can be accessed from other notebooks or Python files.
# 

workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')
trainQ.to_csv(os.path.join(workfolder, 'trainQ_part1'), sep='\t', header=True, index=True, index_label='Id')
testQ.to_csv(os.path.join(workfolder, 'testQ_part1'), sep='\t', header=True, index=True, index_label='Id')


# # Part 3: Model Training and Evaluation
# 
# If you haven't complete the **Part 1: Data Preparation** and **Part 2: Phrase Learning**, please complete them before moving forward with **Part 3: Model Training and Evaluation**.
# 
# **NOTE**: Python 3 kernel doesn't include Azure Machine Learning Workbench functionalities. Please switch the kernel to `local` before continuing further. 
# 
# This example is designed to score new questions against the pre-existing Q&A pairs by training text classification models where each pre-existing Q&A pair is a unique class and a subset of the duplicate questions for each Q&A pair are available as training material. 
# 
# In the Part 3, the classification model uses an ensemble method to aggregate the following three base classifiers. In each base classifier, the `AnswerId` is used as the class label and the BOWs representations is used as the features.
# 
# 1. Naive Bayes Classifier
# 2. Support Vector Machine (TF-IDF as features)
# 3. Random Forest (NB Scores as features)
# 
# Two different evaluation metrics are used to assess performance.
# 1. `Average Rank (AR)`: indicates the average position where the correct answer is found in the list of retrieved Q&A pairs (out of the full set of 103 answer classes). 
# 2. `Top 3 Percentage`: indicates the percentage of test questions that the correct answer can be retrieved in the top three choices in the returned ranked list. 
# 
# `Average Rank (AR)` and `Top 3 Percentage` on the test set are calculated using the following formula:
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/evaluation_3.PNG?token=APoO9pHTAVmmb7YsGlsyWXgMHXDUz0xkks5Zwt4ywA%3D%3D">
# 

# ### Import Required Python Modules
# 
# `modules.feature_extractor` contains a list of user-defined Python modules to extract effective features that are used in this examples. You can find the source code of those modules in the directory of `modules/feature_extractor.py`.
# 

import pandas as pd
import numpy as np
import os, warnings
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from modules.feature_extractor import (tokensToIds, countMatrix, priorProbabilityAnswer, posterioriProb, 
                               feature_selection, featureWeights, wordProbabilityInAnswer, 
                               wordProbabilityNotinAnswer, normalizeTF, getIDF, softmax)
from azureml.logging import get_azureml_logger
warnings.filterwarnings("ignore")


run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.part3-model-training-eval','true')


# ## Access trainQ and testQ from Part 2
# 
# As we have prepared the _trainQ_ and _testQ_ with learned phrases and tokens from `Part 2: Phrase Learning`, we retrieve the datasets here for the further process.
# 
# _trainQ_ contains 5,153 training examples and _testQ_ contains 1,735 test examples. Also, there are 103 unique answer classes in both datasets.
# 

workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')

# paths to trainQ and testQ.
trainQ_path = os.path.join(workfolder, 'trainQ_part2')
testQ_path = os.path.join(workfolder, 'testQ_part2')

# load the training and test data.
trainQ = pd.read_csv(trainQ_path, sep='\t', index_col='Id', encoding='latin1')
testQ = pd.read_csv(testQ_path, sep='\t', index_col='Id', encoding='latin1')


# ## Extract Features
# 
# Selecting the right set of features is very critical for the model training. In this section, we show you several feature extraction approaches that have proved to yield good performance in text classification use cases.
# 
# ### Term Frequency and Inverse Document Frequency (TF-IDF) 
# 
# TF-IDF is a commonly used feature weighting approach for text classification. 
# 
# Each question `d` is typically represented by a feature vector `x` that represents the contents of `d`. Because different questions may have different lengths, it can be useful to apply L1 normalization on the feature vector `x`. Therefore, a normalized `Term Frequency` matrix can be obtained based on the following formula.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/tf.PNG?token=APoO9vtYknxorWSIoJ-dvhbNdu-3pjSIks5ZwuKzwA%3D%3D">
# 
# Considering all tokens observed in the training questions, we compute the `Inverse Document Frequency` for each token based on the following formula.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/idf.PNG?token=APoO9gVRgPlRbg7OSaV56CO0-yj2178Iks5ZwuK-wA%3D%3D">
# 
# By knowing the `Term Frequency (TF)` matrix and `Inverse Document Frequency (IDF)` vector, we can simply compute `TF-IDF` matrix by multiplying them together.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/tfidf.PNG?token=APoO9pllkWjHQTsshFCEGIUbyknjvq8Vks5ZwuMxwA%3D%3D">
# 

token2IdHashInit = tokensToIds(trainQ['Tokens'], featureHash=None)

# get unique answerId in ascending order
uniqueAnswerId = list(np.unique(trainQ['AnswerId']))

N_wQ = countMatrix(trainQ, token2IdHashInit)
idf = getIDF(N_wQ)

x_wTrain = normalizeTF(trainQ, token2IdHashInit)
x_wTest = normalizeTF(testQ, token2IdHashInit)

tfidfTrain = (x_wTrain.T * idf).T
tfidfTest = (x_wTest.T * idf).T


# ### Naive Bayes Scores
# 
# Besides using the IDF as the word weighting mechnism, a hypothesis testing likelihood ratio approach is also implemented here. 
# 
# In this approach, the word weights are associated with the answer classes and are calculated using the following formula.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/NB_weight.PNG?token=APoO9kRUjFMeslJIVyY3wpBy8ycfyddKks5ZwuNjwA%3D%3D">
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/probability_function.PNG?token=APoO9v8kpp4bnjH00Tcr9qPA-tTs5Hezks5ZySQ8wA%3D%3D">
# 
# By knowing the `Term Frequency (TF)` matrix and `Weight` vector for each class, we can simply compute the `Naive Bayes Scores` matrix for each class by multiplying them together.
# 
# #### Feature selection
# 
# Text classification models often pre-select a set of features (i.e., tokens) which carry the most class relevant information for further processing while ignoring words that carry little to no value for identifying classes. A variety of feature selection methods have been previously explored for both text processing. In this example, we have had the most success selecting features based on the estimated class posterior probability `P(A|w)`, where `A` is a specific answer class and `w` is a specific token. The maximum a posteriori probability (MAP) estimate of `P(A|w)` is expressed as
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/feature_selection.PNG?token=APoO9uPZca25b_A2_7-I4m3v1P2K0jSrks5ZySROwA%3D%3D">
# 
# Feature selection in this example is performed by selecting the top `N` tokens which maximize for each `P(A|w)`. In order to determine the best value for the `TopN` parameter, you can simply run the `scripts/naive_bayes.py` with `local` compute context in the Azure Machine Learning Workbench and enter different integer values as `Arguments`.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/run_naive_bayes.PNG?token=APoO9pKfKs4--gnpxNfM8Pueedv5oOwAks5ZwuXpwA%3D%3D">
# 
# Based our experiments, the `TopN = 19` yields the best result and is demonstrated in this notebook. 
# 

# calculate the count matrix of all training questions.
N_wAInit = countMatrix(trainQ, token2IdHashInit, 'AnswerId', uniqueAnswerId)

P_A = priorProbabilityAnswer(trainQ['AnswerId'], uniqueAnswerId)
P_Aw = posterioriProb(N_wAInit, P_A, uniqueAnswerId)

# select top N important tokens per answer class.
featureHash = feature_selection(P_Aw, token2IdHashInit, topN=19)
token2IdHash = tokensToIds(trainQ['Tokens'], featureHash=featureHash)

N_wA = countMatrix(trainQ, token2IdHash, 'AnswerId', uniqueAnswerId)

alpha = 0.0001
P_w = featureWeights(N_wA, alpha)

beta = 0.0001
P_wA = wordProbabilityInAnswer(N_wA, P_w, beta)
P_wNotA = wordProbabilityNotinAnswer(N_wA, P_w, beta)

NBWeights = np.log(P_wA / P_wNotA)


# ## Train Classification Models and Predict on Test Data
# 

# ### Naive Bayes Classifier
# 
# We implement the _Naive Bayes Classifier_ as described in the paper entitled ["MCE Training Techniques for Topic Identification of Spoken Audio Documents"](http://ieeexplore.ieee.org/abstract/document/5742980/).
# 

beta_A = 0

x_wTest = normalizeTF(testQ, token2IdHash)
Y_test_prob1 = softmax(-beta_A + np.dot(x_wTest.T, NBWeights))


# ### Support Vector Machine (TF-IDF as features)
# 
# Traditionally, Support Vector Machine (SVM) model finds a hyperplane which maximally seperates positive and negative training tokens in a vector space. In its standard form, an SVM is a two-class classifier. To create a SVM model for a problem with multiple classes, a one-versus-rest (OVR) SVM classifier is typically learned for each answer class.
# 
# The `sklearn` Python package implement such a classifier and we use the implementation in this example. More information about this `LinearSVC` classifier can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
# 

X_train, Y_train = tfidfTrain.T, np.array(trainQ['AnswerId'])
clf = svm.LinearSVC(dual=True, multi_class='ovr', penalty='l2', C=1, loss="squared_hinge", random_state=1)
clf.fit(X_train, Y_train)

X_test = tfidfTest.T
Y_test_prob2 = softmax(clf.decision_function(X_test))


# ### Random Forest (NB Scores as features)
# 
# Similar to the above one-versus-rest SVM classifier, we also implement a one-versus-rest Random Forest classifier based on a base two-class Random Forest classifier from `sklearn`. More information about the `RandomForestClassifier` can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
# 
# In each base classifier, we dynamically compute the naive bayes scores for the positive class as the features. Since the number of negative examples is much larger than the number of positive examples, we hold all positive example and randomly select negative examples based on a negative to positive ratio to obtain a balanced training data. This is controlled by the `ratio` parameter in the `ovrClassifier` function below.
# 
# In this classifier, we need to tune two hyper-parameters: `TopN` and `n_estimators`. `TopN` is the same parameter as we learned in the _Feature Selection_ step and `n_estimators` indicates the number of trees to be constructed in the Random Forest classifier. To identify the best values for the hyper-parameters, you can run `scripts/random_forest.py` with `local` compute context in the Azure Machine Learning Workbench and enter different integer values `Arguments`. The value of `TopN` and the value of `n_estimators` should be space delimited.
# 
# <img src="https://raw.githubusercontent.com/Azure/MachineLearningSamples-QnAMatching/master/Image/run_rf.PNG?token=APoO9qTD6OH201WZFpAETKAWN3MII-Ocks5ZwumRwA%3D%3D">
# 
# Based our experiments, the `TopN = 19` and `n_estimators = 250` yields the best result, and are demonstrated in this notebook.
# 

# train one-vs-rest classifier using NB scores as features.
def ovrClassifier(trainLabels, x_wTrain, x_wTest, NBWeights, clf, ratio):
    uniqueLabel = np.unique(trainLabels)
    dummyLabels = pd.get_dummies(trainLabels)
    numTest = x_wTest.shape[1]
    Y_test_prob = np.zeros(shape=(numTest, len(uniqueLabel)))

    for i in range(len(uniqueLabel)):
        X_train_all, Y_train_all = x_wTrain.T * NBWeights[:, i], dummyLabels.iloc[:, i]
        X_test = x_wTest.T * NBWeights[:, i]
        
        # with sample selection.
        if ratio is not None:
            # ratio = # of Negative/# of Positive
            posIdx = np.where(Y_train_all == 1)[0]
            negIdx = np.random.choice(np.where(Y_train_all == 0)[0], ratio*len(posIdx))
            allIdx = np.concatenate([posIdx, negIdx])
            X_train, Y_train = X_train_all[allIdx], Y_train_all.iloc[allIdx]
        else: # without sample selection.
            X_train, Y_train = X_train_all, Y_train_all
            
        clf.fit(X_train, Y_train)
        if hasattr(clf, "decision_function"):
            Y_test_prob[:, i] = clf.decision_function(X_test)
        else:
            Y_test_prob[:, i] = clf.predict_proba(X_test)[:, 1]

    return softmax(Y_test_prob)


x_wTrain = normalizeTF(trainQ, token2IdHash)
x_wTest = normalizeTF(testQ, token2IdHash)

clf = RandomForestClassifier(n_estimators=250, criterion='entropy', random_state=1)
Y_test_prob3 = ovrClassifier(trainQ["AnswerId"], x_wTrain, x_wTest, NBWeights, clf, ratio=3)


# ### Ensemble Model
# 
# We build an ensemble model by aggregating the predicted probabilities from three previously trained classifiers. The base classifiers are equally weighted in this ensemble method. 
# 

Y_test_prob_aggr = np.mean([Y_test_prob1, Y_test_prob2, Y_test_prob3], axis=0)


# ## Evaluate Model Performance
# 
# Two different evaluation metrics are used to assess performance. 
# 1. `Average Rank (AR)`: indicates the average position where the correct answer is found in the list of retrieved Q&A pairs (out of the full set of 103 answer classes). 
# 2. `Top 3 Percentage`: indicates the percentage of test questions that the correct answer can be retrieved in the top three choices in the returned ranked list. 
# 

# get the rank of answerIds for a given question. 
def rank(frame, scores, uniqueAnswerId):
    frame['SortedAnswers'] = list(np.array(uniqueAnswerId)[np.argsort(-scores, axis=1)])
    
    rankList = []
    for i in range(len(frame)):
        rankList.append(np.where(frame['SortedAnswers'].iloc[i] == frame['AnswerId'].iloc[i])[0][0] + 1)
    frame['Rank'] = rankList
    
    return frame


testQ = rank(testQ, Y_test_prob_aggr, uniqueAnswerId)

AR = np.floor(testQ['Rank'].mean())
top3 = round(len(testQ.query('Rank <= 3'))/len(testQ), 3)
 
print('Average of rank: ' + str(AR))
print('Percentage of questions find answers in the first 3 choices: ' + str(top3))


