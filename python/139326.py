# ## Automatic Learning of Key Phrases and Topics in Document Collections
# 
# ## Part 2: Phrase Learning
# 
# ### Overview
# 
# This notebook is Part 2 of 6, in a series providing a step-by-step description of how to process and analyze the contents of a large collection of text documents in an unsupervised manner. Using Python packages and custom code examples, we have implemented the basic framework that combines key phrase learning and latent topic modeling as described in the paper entitled ["Modeling Multiword Phrases with Constrained Phrases Tree for Improved Topic Modeling of Conversational Speech"](http://people.csail.mit.edu/hazen/publications/Hazen-SLT-2012.pdf) which was originally presented in the 2012 IEEE Workshop on Spoken Language Technology.
# 
# Although the paper examines the use of the technology for analyzing human-to-human conversations, the techniques are quite general and can be applied to a wide range of natural language data including news stories, legal documents, research publications, social media forum discussions, customer feedback forms, product reviews, and many more.
# 
# Part 2 of the series shows how to learn the most salient phrases present in a large collection of documents. These phrases can be treated as single compound word units in down-stream processes such as topic modeling.
# 

# ### Import Relevant Python Packages
# 

import pandas 
import re
import math
from operator import itemgetter
from collections import namedtuple
from datetime import datetime
from multiprocessing import cpu_count
from math import log
from sys import getsizeof
import concurrent.futures
import threading
import platform
import time
import gc
import sys
import os

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook2', 'true')


# ### Load Text Data
# 

# > **NOTE** The data file is saved under the folder defined by environment variable `AZUREML_NATIVE_SHARE_DIRECTORY` in notebook 1. If you have changed it to `../Data`, please also do the change here.
# 

textFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsCleaned.tsv'), 
                            sep='\t', 
                            encoding='ISO-8859-1')


print ("Total lines in cleaned text: %d\n" % len(textFrame))

# Show the first 25 rows of the data in the frame
textFrame[0:25]


# ### Create Lowercase Version of the Text Data
# 
# Before learning phrases we lowercase the entire text corpus to ensure all casing variants for each word are collapsed into a single uniform variant used during the learning process. 
# 

# Create a lowercased version of the data and add it into the data frame
lowercaseText = []
for textLine in textFrame['CleanedText']:
    lowercaseText.append(str(textLine).lower())
textFrame['LowercaseText'] = lowercaseText;           
            
textFrame[0:25]


# ### Load the Supplemental Word Lists
# 
# Words in the black list are completely ignored by the process and cannot be used in the creation of phrases. Words in the function word list can only be used in between content words in the creation of phrases.
# 

# Define a function for loading lists into dictionary hash tables
def LoadListAsHash(filename):
    listHash = {}
    fp = open(filename, encoding='utf-8')

    # Read in lines one by one stripping away extra spaces, 
    # leading spaces, and trailing spaces and inserting each
    # cleaned up line into a hash table
    re1 = re.compile(' +')
    re2 = re.compile('^ +| +$')
    for stringIn in fp.readlines():
        term = re2.sub("",re1.sub(" ",stringIn.strip('\n')))
        if term != '':
            listHash[term] = 1

    fp.close()
    return listHash 


# Load the black list of words to ignore 
blacklistHash = LoadListAsHash(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'black_list.txt'))

# Load the list of non-content bearing function words
functionwordHash = LoadListAsHash(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'function_words.txt'))

# Add more terms to the function word list
functionwordHash["foo"] = 1


# ### Compute N-gram Statistics for Phrase Learning
# 

# This is the function that used to define how to compute N-gram stats
# This function will be executed in parallel as process pool executor
def ComputeNgramStatsJob(textList, functionwordHash, blacklistHash, reValidWord, jobId, verbose=False):
    if verbose:
        startTS = datetime.now()
        print("[%s] Starting batch execution %d" % (str(startTS), jobId+1))
    
    # Create an array to store the total count of all ngrams up to 4-grams
    # Array element 0 is unused, element 1 is unigrams, element 2 is bigrams, etc.
    ngramCounts = [0]*5;
       
    # Create a list of structures to tabulate ngram count statistics
    # Array element 0 is the array of total ngram counts,
    # Array element 1 is a hash table of individual unigram counts
    # Array element 2 is a hash table of individual bigram counts
    # Array element 3 is a hash table of individual trigram counts
    # Array element 4 is a hash table of individual 4-gram counts
    ngramStats = [ngramCounts, {}, {}, {}, {}]
    
    numLines = len(textList)
    if verbose:
        print("# Batch %d, received %d lines data" % (jobId+1, numLines))
    
    for i in range(0, numLines):
        # Split the text line into an array of words
        wordArray = textList[i].strip().split()
        numWords = len(wordArray)
        
        # Create an array marking each word as valid or invalid
        validArray = [reValidWord.match(word) != None for word in wordArray]
        
        # Tabulate total raw ngrams for this line into counts for each ngram bin
        # The total ngrams counts include the counts of all ngrams including those
        # that we won't consider as parts of phrases
        for j in range(1, 5):
            if j <= numWords:
                ngramCounts[j] += numWords - j + 1
        
        # Collect counts for viable phrase ngrams and left context sub-phrases
        for j in range(0, numWords):
            word = wordArray[j]

            # Only bother counting the ngrams that start with a valid content word
            # i.e., valid words not in the function word list or the black list
            if ( ( word not in functionwordHash ) and ( word not in blacklistHash ) and validArray[j] ):

                # Initialize ngram string with first content word and add it to unigram counts
                ngramSeq = word 
                if ngramSeq in ngramStats[1]:
                    ngramStats[1][ngramSeq] += 1
                else:
                    ngramStats[1][ngramSeq] = 1

                # Count valid ngrams from bigrams up to 4-grams
                stop = 0
                k = 1
                while (k<4) and (j+k<numWords) and not stop:
                    n = k + 1
                    nextNgramWord = wordArray[j+k]
                    # Only count ngrams with valid words not in the blacklist
                    if ( validArray[j+k] and nextNgramWord not in blacklistHash ):
                        ngramSeq += " " + nextNgramWord
                        if ngramSeq in ngramStats[n]:
                            ngramStats[n][ngramSeq] += 1
                        else:
                            ngramStats[n][ngramSeq] = 1 
                        k += 1
                        if nextNgramWord not in functionwordHash:
                            # Stop counting new ngrams after second content word in 
                            # ngram is reached and ngram is a viable full phrase
                            stop = 1
                    else:
                        stop = 1
    
    if verbose:
        endTS = datetime.now()
        delta_t = (endTS - startTS).total_seconds()
        print("[%s] Batch %d finished, time elapsed: %f seconds" % (str(endTS), jobId+1, delta_t))
    
    return ngramStats


# This is Step 1 for each iteration of phrase learning
# We count the number of occurrences of all 2-gram, 3-ngram, and 4-gram
# word sequences 
def ComputeNgramStats(textData, functionwordHash, blacklistHash, numWorkers, verbose=False):
          
    # Create a regular expression for assessing validity of words
    # for phrase modeling. The expression says words in phrases
    # must either:
    # (1) contain an alphabetic character, or 
    # (2) be the single charcater '&', or
    # (3) be a one or two digit number
    reWordIsValid = re.compile('[A-Za-z]|^&$|^\d\d?$');
    
    # Go through the text data line by line collecting count statistics
    # for all valid n-grams that could appear in a potential phrase
    numLines = len(textData)
    
    # Get the number of CPU to run the tasks
    if numWorkers > cpu_count() or numWorkers <= 0:
        worker = cpu_count()
    else:
        worker = numWorkers
    if verbose:
        print("Worker size = %d" % worker)
    
    # Get the batch size for each execution job
    # The very last job executor may received more lines of data
    batch_size = int(numLines/worker)
    batchIndexes = range(0, numLines, batch_size)
    
    batch_returns = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        jobs = set()
        
        # Map the task into multiple batch executions
        if platform.system() == "Linux" or platform.system() == "Darwin":
            for idx in range(worker):
                # The very last job executor
                if idx == (worker-1):
                    jobs.add(executor.submit(ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]: ], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx, 
                                                 verbose))
                else:
                    jobs.add(executor.submit(ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx,
                                                 verbose))
        else:
            # For Windows system, it is different to handle ProcessPoolExecutor
            from notebooks import winprocess
            
            for idx in range(worker):
                # The very last job executor
                if idx == (worker-1):
                    jobs.add(winprocess.submit(executor,
                                                 ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]: ], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx, 
                                                 verbose))
                else:
                    jobs.add(winprocess.submit(executor,
                                                 ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx,
                                                 verbose))
        
        # Get results from batch executions
        for job in concurrent.futures.as_completed(jobs):
            try:
                ret = job.result()
            except Exception as e:
                print("Generated an exception while trying to get result from a batch: %s" % e)
            else:
                batch_returns.append(ret)

    # Reduce the results from batch executions
    # Reuse the first return
    ngramStats = batch_returns[0]
    
    for batch_id in range(1, len(batch_returns)):
        result = batch_returns[batch_id]
        
        # Update the ngram counts
        ngramStats[0] = [x + y for x, y in zip(ngramStats[0], result[0])]
        
        # Update the hash table of ngram counts
        for n_gram in range(1, 5):
            for item in result[n_gram]:
                if item in ngramStats[n_gram]:
                    ngramStats[n_gram][item] += result[n_gram][item]
                else:
                    ngramStats[n_gram][item] = result[n_gram][item]
    
    return ngramStats


# ### Rank Potential Phrases by the Weighted Pointwise Mutual Information of their Constituent Words
# 

def RankNgrams(ngramStats,functionwordHash,minCount):
    # Create a hash table to store weighted pointwise mutual 
    # information scores for each viable phrase
    ngramWPMIHash = {}
        
    # Go through each of the ngram tables and compute the phrase scores
    # for the viable phrases
    for n in range(2,5):
        i = n-1
        for ngram in ngramStats[n].keys():
            ngramCount = ngramStats[n][ngram]
            if ngramCount >= minCount:
                wordArray = ngram.split()
                # If the final word in the ngram is not a function word then
                # the ngram is a valid phrase candidate we want to score
                if wordArray[i] not in functionwordHash: 
                    leftNgram = ' '.join(wordArray[:-1])
                    rightWord = wordArray[i]
                    
                    # Compute the weighted pointwise mutual information (WPMI) for the phrase
                    probNgram = float(ngramStats[n][ngram])/float(ngramStats[0][n])
                    probLeftNgram = float(ngramStats[n-1][leftNgram])/float(ngramStats[0][n-1])
                    probRightWord = float(ngramStats[1][rightWord])/float(ngramStats[0][1])
                    WPMI = probNgram * math.log(probNgram/(probLeftNgram*probRightWord));

                    # Add the phrase into the list of scored phrases only if WMPI is positive
                    if WPMI > 0:
                        ngramWPMIHash[ngram] = WPMI  
    
    # Create a sorted list of the phrase candidates
    rankedNgrams = sorted(ngramWPMIHash, key=ngramWPMIHash.__getitem__, reverse=True)

    # Force a memory clean-up
    ngramWPMIHash = None
    gc.collect()

    return rankedNgrams


# ### Apply Phrase Rewrites to Text Data
# 

def phraseRewriteJob(ngramRegex, text, ngramRewriteHash, jobId, verbose=True):
    if verbose:
        startTS = datetime.now()
        print("[%s] Starting batch execution %d" % (str(startTS), jobId+1))
    
    retList = []
    
    for i in range(len(text)):
        # The regex substitution looks up the output string rewrite
        # in the hash table for each matched input phrase regex
        retList.append(ngramRegex.sub(lambda mo: ngramRewriteHash[mo.string[mo.start():mo.end()]], text[i]))
    
    if verbose:
        endTS = datetime.now()
        delta_t = (endTS - startTS).total_seconds()
        print("[%s] Batch %d finished, batch size: %d, time elapsed: %f seconds" % (str(endTS), jobId+1, i, delta_t))
    
    return retList, jobId


def ApplyPhraseRewrites(rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, 
                        maxPhraseLength, verbose, numWorkers=cpu_count()):

    # If the number of rankedNgrams coming in is zero then
    # just return without doing anything
    numNgrams = len(rankedNgrams)
    if numNgrams == 0:
        return

    # This function will consider at most maxRewrite 
    # new phrases to be added into the learned phrase 
    # list as specified by the calling function
    maxRewrite=maxPhrasesToAdd

    # If the remaining number of proposed ngram phrases is less 
    # than the max allowed, then reset maxRewrite to the size of 
    # the proposed ngram phrases list
    if numNgrams < maxRewrite:
        maxRewrite = numNgrams

    # Create empty hash tables to keep track of phrase overlap conflicts
    leftConflictHash = {}
    rightConflictHash = {}
    
    # Create an empty hash table collecting the set of rewrite rules
    # to be applied during this iteration of phrase learning
    ngramRewriteHash = {}
    
    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')

    # Initialize some bookkeeping variables
    numLines = len(textData)  
    numPhrasesAdded = 0
    numConsidered = 0
    lastSkippedNgram = ""
    lastAddedNgram = ""
  
    # Collect list of up to maxRewrite ngram phrase rewrites
    stop = False
    index = 0
    while not stop:

        # Get the next phrase to consider adding to the phrase list
        inputNgram = rankedNgrams[index]

        # Create the output compound word version of the phrase
        # The extra space is added to make the regex rewrite easier
        outputNgram = " " + regexSpace.sub("_",inputNgram)

        # Count the total number of words in the proposed phrase
        numWords = len(outputNgram.split("_"))

        # Only add phrases that don't exceed the max phrase length
        if (numWords <= maxPhraseLength):
    
            # Keep count of phrases considered for inclusion during this iteration
            numConsidered += 1

            # Extract the left and right words in the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = inputNgram.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[-1]

            # Skip any ngram phrases that conflict with earlier phrases added
            # These ngram phrases will be reconsidered in the next iteration
            if (leftWord in leftConflictHash) or (rightWord in rightConflictHash): 
                if verbose: 
                    print ("(%d) Skipping (context conflict): %s" % (numConsidered,inputNgram))
                lastSkippedNgram = inputNgram
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                if verbose:
                    print ("(%d) Adding: %s" % (numConsidered,inputNgram))
                ngramRewriteHash[" " + inputNgram] = outputNgram
                learnedPhrases.append(inputNgram) 
                lastAddedNgram = inputNgram
                numPhrasesAdded += 1
            
            # Keep track of all context words that might conflict with upcoming
            # propose phrases (even when phrases are skipped instead of added)
            leftConflictHash[rightWord] = 1
            rightConflictHash[leftWord] = 1

            # Stop when we've considered the maximum number of phrases per iteration
            if ( numConsidered >= maxRewrite ):
                stop = True
            
        # Increment to next phrase
        index += 1
    
        # Stop if we've reached the end of the ranked ngram list
        if index >= len(rankedNgrams):
            stop = True
    
    # Now do the phrase rewrites over the entire set of text data
    # Compile a single regex rule from the collected set of phrase rewrites for this iteration
    ngramRegex = re.compile(r'%s(?= )' % "(?= )|".join(map(re.escape, ngramRewriteHash.keys())))
    
    # Get the number of CPU to run the tasks
    if numWorkers > cpu_count() or numWorkers <= 0:
        worker = cpu_count()
    else:
        worker = numWorkers
    if verbose:
        print("Worker size = %d" % worker)
        
    # Get the batch size for each execution job
    # The very last job executor may receive more lines of data
    batch_size = int(numLines/worker)
    batchIndexes = range(0, numLines, batch_size)
    
    batch_returns = [[]] * worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        jobs = set()
        
        # Map the task into multiple batch executions
        if platform.system() == "Linux" or platform.system() == "Darwin":
            for idx in range(worker):
                if idx == (worker-1):
                    jobs.add(executor.submit(phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]: ], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
                else:
                    jobs.add(executor.submit(phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
        else:
            from notebooks import winprocess
            
            for idx in range(worker):
                if idx == (worker-1):
                    jobs.add(winprocess.submit(executor,
                                             phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]: ], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
                else:
                    jobs.add(winprocess.submit(executor,
                                             phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
        
        textData.clear()
        
        # Get results from batch executions
        for job in concurrent.futures.as_completed(jobs):
            try:
                ret, idx = job.result()
            except Exception as e:
                print("Generated an exception while trying to get result from a batch: %s" % e)
            else:
                batch_returns[idx] = ret
        textData += sum(batch_returns, [])
     
    return


# ### Run the full iterative phrase learning process
# 

def ApplyPhraseLearning(textData,learnedPhrases,learningSettings):
    
    stop = False
    iterNum = 0

    # Get the learning parameters from the structure passed in by the calling function
    maxNumPhrases = learningSettings.maxNumPhrases
    maxPhraseLength = learningSettings.maxPhraseLength
    functionwordHash = learningSettings.functionwordHash
    blacklistHash = learningSettings.blacklistHash
    verbose = learningSettings.verbose
    minCount = learningSettings.minInstanceCount
    
    # Start timing the process
    functionStartTime = time.clock()
    
    numPhrasesLearned = len(learnedPhrases)
    print ("Start phrase learning with %d phrases of %d phrases learned" % (numPhrasesLearned,maxNumPhrases))

    while not stop:
        iterNum += 1
                
        # Start timing this iteration
        startTime = time.clock()
 
        # Collect ngram stats
        ngramStats = ComputeNgramStats(textData, functionwordHash, blacklistHash, cpu_count(), verbose)

        # Uncomment this for more detailed timing info
        countTime = time.clock()
        elapsedTime = countTime - startTime
        print ("--- Counting time: %.2f seconds" % elapsedTime)
        
        # Rank ngrams
        rankedNgrams = RankNgrams(ngramStats,functionwordHash,minCount)
        
        # Uncomment this for more detailed timing info
        rankTime = time.clock()
        elapsedTime = rankTime - countTime
        print ("--- Ranking time: %.2f seconds" % elapsedTime)
        
        
        # Incorporate top ranked phrases into phrase list
        # and rewrite the text to use these phrases
        if len(rankedNgrams) > 0:
            maxPhrasesToAdd = maxNumPhrases - numPhrasesLearned
            if maxPhrasesToAdd > learningSettings.maxPhrasesPerIter:
                maxPhrasesToAdd = learningSettings.maxPhrasesPerIter
            ApplyPhraseRewrites(rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, 
                                maxPhraseLength, verbose, cpu_count())
            numPhrasesAdded = len(learnedPhrases) - numPhrasesLearned
        else:
            stop = True
            
        # Uncomment this for more detailed timing info
        rewriteTime = time.clock()
        elapsedTime = rewriteTime - rankTime
        print ("--- Rewriting time: %.2f seconds" % elapsedTime)
           
        # Garbage collect
        ngramStats = None
        rankedNgrams = None
        gc.collect();
               
        elapsedTime = time.clock() - startTime

        numPhrasesLearned = len(learnedPhrases)
        print ("Iteration %d: Added %d new phrases in %.2f seconds (Learned %d of max %d)" % 
               (iterNum,numPhrasesAdded,elapsedTime,numPhrasesLearned,maxNumPhrases))
        
        if numPhrasesAdded >= maxPhrasesToAdd or numPhrasesAdded == 0:
            stop = True
        
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textData)):
        textData[i] = regexSpacePadding.sub("",textData[i])
    
    gc.collect()
 
    elapsedTime = time.clock() - functionStartTime
    elapsedTimeHours = elapsedTime/3600.0;
    print ("*** Phrase learning completed in %.2f hours ***" % elapsedTimeHours) 

    return


# -------
# ### Main top level execution of phrase learning functionality
# 

# > **NOTE:** The phrase learning step is implemented with multiprocessing. However, more CPU cores do NOT mean a faster execution time. In our tests, the performance is not improved with more than eight cores due to the overhead of multiprocessing. It took about two and a half hours to learn 25,000 phrases on a machine with eight cores (3.6 GHz). 
# 
# > If you just need to run the code and see how it works, change the variable `learningSettings.maxNumPhrases` in the cell below to a small number, by default it will try to learn 25,000 phrases.
# 

# Create a structure defining the settings and word lists used during the phrase learning
learningSettings = namedtuple('learningSettings',['maxNumPhrases','maxPhrasesPerIter',
                                                  'maxPhraseLength','minInstanceCount'
                                                  'functionwordHash','blacklistHash','verbose'])

# If true it prints out the learned phrases to stdout buffer
# while its learning. This will generate a lot of text to stdout, 
# so best to turn this off except for testing and debugging
learningSettings.verbose = False

# Maximum number of phrases to learn
# If you want to test the code out quickly then set this to a small
# value (e.g. 100) and set verbose to true when running the quick test
learningSettings.maxNumPhrases = 25000

# Maximum number of phrases to learn per iteration 
# Increasing this number may speed up processing but will affect the ordering of the phrases 
# learned and good phrases could be by-passed if the maxNumPhrases is set to a small number
learningSettings.maxPhrasesPerIter = 500

# Maximum number of words allowed in the learned phrases 
learningSettings.maxPhraseLength = 7

# Minimum number of times a phrase must occur in the data to 
# be considered during the phrase learning process
learningSettings.minInstanceCount = 5

# This is a precreated hash table containing the list 
# of function words used during phrase learning
learningSettings.functionwordHash = functionwordHash

# This is a precreated hash table containing the list 
# of black list words to be ignored during phrase learning
learningSettings.blacklistHash = blacklistHash

# Initialize an empty list of learned phrases
# If you have completed a partial run of phrase learning
# and want to add more phrases, you can use the pre-learned 
# phrases as a starting point instead and the new phrases
# will be appended to the list
learnedPhrases = []

# Create a copy of the original text data that will be used during learning
# The copy is needed because the algorithm does in-place replacement of learned
# phrases directly on the text data structure it is provided
phraseTextData = []
for textLine in textFrame['LowercaseText']:
    phraseTextData.append(' ' + textLine + ' ')

# Run the phrase learning algorithm
if True:
    ApplyPhraseLearning(phraseTextData, learnedPhrases, learningSettings)


learnedPhrasesFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsLearnedPhrases.txt")
phraseTextDataFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsPhraseTextData.txt")

writeLearnedPhrases = True

if writeLearnedPhrases:
    # Write out the learned phrases to a text file
    fp = open(learnedPhrasesFile, 'w', encoding='utf-8')
    for phrase in learnedPhrases:
        fp.write("%s\n" % phrase)
    fp.close()

    # Write out the text data containing the learned phrases to a text file
    fp = open(phraseTextDataFile, 'w', encoding='utf-8')
    for line in phraseTextData:
        fp.write("%s\n" % line)
    fp.close()
else:
    # Read in the learned phrases from a text file
    learnedPhrases = []
    fp = open(learnedPhrasesFile, 'r', encoding='utf-8')
    for line in fp:
        learnedPhrases.append(line.strip())
    fp.close()

    # Read in the learned phrases from a text file
    phraseTextData = []
    fp = open(phraseTextDataFile, 'r', encoding='utf-8')
    for line in fp:
        phraseTextData.append(line.strip())
    fp.close()


learnedPhrases[0:10]


learnedPhrases[5000:5010]


phraseTextData[0:15]


# Add text with learned phrases back into data frame
textFrame['TextWithPhrases'] = phraseTextData


textFrame[0:10]


textFrame['TextWithPhrases'][2]


# ### Find Most Common Surface Form of Each Lower-Cased Word and Phrase
# 
# The text data is lower cased in order to merge differently cased versions of the same word prior to doing topic modeling. In order to generate summaries of topics that will be learned, we would like to present the most likely surface form of a word to the user. For example, if a proper noun is converted to all lowercase characters for latent topic modeling, we want the user to see this proper name with its proper capitalization within summaries. The MapVocabToSurfaceForms() function achieves this by mapping every lowercased word and phrase used during latent topic modeling to its most common surface form in the text collection.
# 
# 

def MapVocabToSurfaceForms(textData):
    surfaceFormCountHash = {}
    vocabToSurfaceFormHash = {}
    regexUnderBar = re.compile('_')
    regexSpace = re.compile(' +')
    regexClean = re.compile('^ +| +$')
    
    # First go through every line of text, align each word/phrase with
    # it's surface form and count the number of times each surface form occurs
    for i in range(0,len(textData)):    
        origWords = regexSpace.split(regexClean.sub("",str(textData['CleanedText'][i])))
        numOrigWords = len(origWords)
        newWords = regexSpace.split(regexClean.sub("",str(textData['TextWithPhrases'][i])))
        numNewWords = len(newWords)
        origIndex = 0
        newIndex = 0
        while newIndex < numNewWords:
            # Get the next word or phrase in the lower-cased text with phrases and
            # match it to the original form of the same n-gram in the original text
            newWord = newWords[newIndex]
            phraseWords = regexUnderBar.split(newWord)
            numPhraseWords = len(phraseWords)
            matchedWords = " ".join(origWords[origIndex:(origIndex+numPhraseWords)])
            origIndex += numPhraseWords
                
            # Now do the bookkeeping for collecting the different surface form 
            # variations present for each lowercased word or phrase
            if newWord in vocabToSurfaceFormHash:
                vocabToSurfaceFormHash[newWord].add(matchedWords)
            else:
                vocabToSurfaceFormHash[newWord] = set([matchedWords])

            # Increment the counter for this surface form
            if matchedWords not in surfaceFormCountHash:
                surfaceFormCountHash[matchedWords] = 1
            else:
                surfaceFormCountHash[matchedWords] += 1
   
            if ( len(newWord) != len(matchedWords)):
                print ("##### Error #####")
                print ("Bad Match: %s ==> %s " % (newWord,matchedWords))
                print ("From line: %s" % textData['TextWithPhrases'][i])
                print ("Orig text: %s" % textData['CleanedText'][i])
                
                return False

            newIndex += 1
    # After aligning and counting, select the most common surface form for each

    # word/phrase to be the canonical example shown to the user for that word/phrase
    for ngram in vocabToSurfaceFormHash.keys():
        maxCount = 0
        bestSurfaceForm = ""
        for surfaceForm in vocabToSurfaceFormHash[ngram]:
            if surfaceFormCountHash[surfaceForm] > maxCount:
                maxCount = surfaceFormCountHash[surfaceForm]
                bestSurfaceForm = surfaceForm
        if ngram != "":
            if bestSurfaceForm == "":
                print ("Warning: NULL surface form for ngram '%s'" % ngram)
            else:
                vocabToSurfaceFormHash[ngram] = bestSurfaceForm
    
    return vocabToSurfaceFormHash


get_ipython().run_cell_magic('time', '', '\nif True:\n    vocabToSurfaceFormHash = MapVocabToSurfaceForms(textFrame)')


# Save the mapping between model vocabulary and surface form mapping
tsvFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "Vocab2SurfaceFormMapping.tsv")

saveSurfaceFormFile = True

if saveSurfaceFormFile:
    with open(tsvFile, 'w', encoding='utf-8') as fp:
        for key, val in vocabToSurfaceFormHash.items():
            if key != "":
                strOut = "%s\t%s\n" % (key, val)
                fp.write(strOut)
else:
    # Load surface form mappings here
    vocabToSurfaceFormHash = {}
    fp = open(tsvFile, encoding='utf-8')

    # Each line in the file has two tab separated fields;
    # the first is the vocabulary item used during modeling
    # and the second is its most common surface form in the 
    # original data
    for stringIn in fp.readlines():
        fields = stringIn.strip().split("\t")
        if len(fields) != 2:
            print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
        elif fields[0] == "" or fields[1] == "":
            print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
        else:
            vocabToSurfaceFormHash[fields[0]] = fields[1]
    fp.close()


print(vocabToSurfaceFormHash['security'])
print(vocabToSurfaceFormHash['declares'])
print(vocabToSurfaceFormHash['mental_health'])
print(vocabToSurfaceFormHash['el_salvador'])
print(vocabToSurfaceFormHash['department_of_the_interior'])


# ### Reconstruct the Full Processed Text of Each Document and Put it into a New Frame 
# 

def ReconstituteDocsFromChunks(textData, idColumnName, textColumnName):
    dataOut = []
    
    currentDoc = ""
    currentDocID = ""
    
    for i in range(0,len(textData)):
        textChunk = textData[textColumnName][i]
        docID = str(textData[idColumnName][i])
        if docID != currentDocID:
            if currentDocID != "":
                dataOut.append([currentDocID, currentDoc])
            currentDoc = textChunk
            currentDocID = docID
        else:
            currentDoc += " " + textChunk
    dataOut.append([currentDocID,currentDoc])
    
    frameOut = pandas.DataFrame(dataOut, columns=['DocID','ProcessedText'])
    
    return frameOut


get_ipython().run_cell_magic('time', '', "\nif True:\n    docsFrame = ReconstituteDocsFromChunks(textFrame, 'DocID', 'TextWithPhrases')")


saveProcessedText = True

# Save processed text for each document back out to a TSV file
if saveProcessedText:
    docsFrame.to_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsProcessed.tsv'),  
                        sep='\t', index=False)
else: 
    docsFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsProcessed.tsv'), 
                                    sep='\t')


docsFrame[0:5]


docsFrame['ProcessedText'][1]


# ### Apply Rules to New Documents
# 
# 

def ApplyPhraseRewritesInPlace(textFrame, textColumnName, phraseRules):
    
    # Make sure we have phrase to add
    numPhraseRules = len(phraseRules)
    if numPhraseRules == 0: 
        print ("Warning: phrase rule lise is empty - no phrases being applied to text data")
        return
    
    # Get text data column from frame
    textData = textFrame[textColumnName]
    numLines = len(textData)
    
    # Add leading and trailing spaces to make regex matching easier
    for i in range(0,numLines):
        textData[i] = " " + textData[i] + " "  

    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')
   
    # Initialize some bookkeeping variables

    # Iterate through full set of phrases to find sets of 
    # non-conflicting phrases that can be apply simultaneously
    index = 0
    outerStop = False
    while not outerStop:
       
        # Create empty hash tables to keep track of phrase overlap conflicts
        leftConflictHash = {}
        rightConflictHash = {}
        prevConflictHash = {}
    
        # Create an empty hash table collecting the next set of rewrite rules
        # to be applied during this iteration of phrase rewriting
        phraseRewriteHash = {}
    
        # Progress through phrases until the next conflicting phrase is found
        innerStop = 0
        numPhrasesAdded = 0
        while not innerStop:
        
            # Get the next phrase to consider adding to the phrase list
            nextPhrase = phraseRules[index]            
            
            # Extract the left and right sides of the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = nextPhrase.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[-1] 

            # Stop if we reach any phrases that conflicts with earlier phrases in this iteration
            # These ngram phrases will be reconsidered in the next iteration
            if ((leftWord in leftConflictHash) or (rightWord in rightConflictHash) 
                or (leftWord in prevConflictHash) or (rightWord in prevConflictHash)): 
                innerStop = True
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                # Create the output compound word version of the phrase
                                
                outputPhrase = regexSpace.sub("_",nextPhrase);
                
                # Keep track of all context words that might conflict with upcoming
                # propose phrases (even when phrases are skipped instead of added)
                leftConflictHash[rightWord] = 1
                rightConflictHash[leftWord] = 1
                prevConflictHash[outputPhrase] = 1           
                
                # Add extra space to input an output versions of the current phrase 
                # to make the regex rewrite easier
                outputPhrase = " " + outputPhrase
                lastAddedPhrase = " " + nextPhrase
                
                # Add the phrase to the rewrite hash
                phraseRewriteHash[lastAddedPhrase] = outputPhrase
                  
                # Increment to next phrase
                index += 1
                numPhrasesAdded  += 1
    
                # Stop if we've reached the end of the phrases list
                if index >= numPhraseRules:
                    innerStop = True
                    outerStop = True
                    
        # Now do the phrase rewrites over the entire set of text data
        if numPhrasesAdded == 1:
        
            # If only one phrase to add use a single regex rule to do this phrase rewrite        
            outputPhrase = phraseRewriteHash[lastAddedPhrase]
            regexPhrase = re.compile (r'%s(?= )' % re.escape(lastAddedPhrase)) 
        
            # Apply the regex over the full data set
            for j in range(0,numLines):
                textData[j] = regexPhrase.sub(outputPhrase, textData[j])
        
        elif numPhrasesAdded > 1:
            # Compile a single regex rule from the collected set of phrase rewrites for this iteration
            regexPhrase = re.compile(r'%s(?= )' % "|".join(map(re.escape, phraseRewriteHash.keys())))
            
            # Apply the regex over the full data set
            for i in range(0,numLines):
                # The regex substituion looks up the output string rewrite  
                # in the hash table for each matched input phrase regex
                textData[i] = regexPhrase.sub(lambda mo: phraseRewriteHash[mo.string[mo.start():mo.end()]], textData[i]) 
    
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textData)):
        textData[i] = regexSpacePadding.sub("",textData[i])
    
    return


testText = ["the president of the united states appoints the secretary of labor to lead the department of labor", 
            "the speaker of the house of representatives is elected each session by the members of the house",
            "the president pro tempore of the the u.s. senate resides over the senate when the vice president is absent"]

testFrame = pandas.DataFrame(testText, columns=['TestText'])      

ApplyPhraseRewritesInPlace(testFrame, 'TestText', learnedPhrases)

print(testFrame['TestText'][0])
print(testFrame['TestText'][1])
print(testFrame['TestText'][2])


# ### Next
# 
# The phrase learning step is finished. The next step will be topic modeling which will be in the third notebook of the series: [`3_Topic_Model_Training.ipynb`](./3_Topic_Model_Training.ipynb).
# 

# # Automatic Learning of Key Phrases and Topics in Document Collections
# 
# ## Part 4: Topic Model Summarization
# 
# ### Overview
# 
# This notebook is Part 4 of 6 in a series providing a step-by-step description of how to process and analyze the contents of a large collection of text documents in an unsupervised manner. Using Python packages and custom code examples, we have implemented the basic framework that combines key phrase learning and latent topic modeling as described in the paper entitled ["Modeling Multiword Phrases with Constrained Phrases Tree for Improved Topic Modeling of Conversational Speech"](http://people.csail.mit.edu/hazen/publications/Hazen-SLT-2012.pdf) which was originally presented in the 2012 IEEE Workshop on Spoken Language Technology.
# 
# Although the paper examines the use of the technology for analyzing human-to-human conversations, the techniques are quite general and can be applied to a wide range natural language data including news stories, legal documents, research publications, social media forum discussion, customer feedback forms, product reviews, and many more.
# 
# Part 4 of the series shows how to summarize the contents of the document based on a trained LDA topic model. The summarization is applied to an LDA topic model learned in Part 3.  
# 

# > **NOTE:** If you have retrained your own LDA model, you may not get the same topic model we are showing in this notebook. For the demonstration purpose, all files used in this notebook can be downloaded via the links below. You can download all files to the `AZUREML_NATIVE_SHARE_DIRECTORY` folder and you will have exactly the same results in this notebook.
# 

# 
# | File Name | Link |
# |-----------|------|
# | `CongressionalDocsLDA.pickle` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle |
# | `CongressionalDocsLDA.pickle.expElogbeta.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.expElogbeta.npy |
# | `CongressionalDocsLDA.pickle.id2word` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.id2word |
# | `CongressionalDocsLDA.pickle.state` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.state |
# | `CongressionalDocsLDA.pickle.state.sstats.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.state.sstats.npy |
# | `CongressionalDocTopicLM.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicLM.npy |
# | `CongressionalDocTopicProbs.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicProbs.npy |
# | `CongressionalDocTopicSummaries.tsv` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicSummaries.tsv |
# | `Vocab2SurfaceFormMapping.tsv` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/Vocab2SurfaceFormMapping.tsv |
# 

# Need to set the `saveFile` flag to `True` in case you do not want to download those pre-trained files, and want to re-run everything.
# 

saveFile = True


# ### Download Data Files (optional)
# 

# You can download all those data files by executing the code in the cells below.
# 

import urllib.request
import os

def download_file_from_blob(filename):
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    save_path = os.path.join(shared_path, filename)

    if not os.path.exists(save_path):
        # Base URL for anonymous read access to Blob Storage container
        STORAGE_CONTAINER = 'https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/'
        url = STORAGE_CONTAINER + filename
        urllib.request.urlretrieve(url, save_path)
        print("Downloaded file: %s" % filename)
    else:
        print("File \"%s\" already existed" % filename)


download_file_from_blob('CongressionalDocsLDA.pickle')
download_file_from_blob('CongressionalDocsLDA.pickle.expElogbeta.npy')
download_file_from_blob('CongressionalDocsLDA.pickle.id2word')
download_file_from_blob('CongressionalDocsLDA.pickle.state')
download_file_from_blob('CongressionalDocsLDA.pickle.state.sstats.npy')
download_file_from_blob('CongressionalDocTopicLM.npy')
download_file_from_blob('CongressionalDocTopicProbs.npy')
download_file_from_blob('CongressionalDocTopicSummaries.tsv')
download_file_from_blob('Vocab2SurfaceFormMapping.tsv')

# Set the saveFile flag to False since you have already downloaded those files
saveFile = False


# ### Import Relevant Python Packages
# 
# Most significantly, Part 3 relies on the use of the [Gensim Python library](http://radimrehurek.com/gensim/)  for generating a sparse bag-of-words representation of each document and then training a [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) model on the data. LDA produces a collection of latent topics learned in a completely unsupervised fashion from the text data. Each document can then be represented with a distribution of the learned topics.
# 

import numpy
import pandas 
import re
import math
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora
from gensim import models
from operator import itemgetter
from collections import namedtuple
import time
import gc
import sys
import multiprocessing
import matplotlib
matplotlib.use('Agg')

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook4', 'true')


# ### Load the Trained LDA Model Learned in Part 3 
# 

# > **NOTE** The data file is saved under the folder defined by environment variable `AZUREML_NATIVE_SHARE_DIRECTORY` in notebook 1. If you have changed it to `../Data`, please also do the change here.
# 

# Load pretrained LDA topic model
ldaFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsLDA.pickle")
lda = gensim.models.ldamodel.LdaModel.load(ldaFile)


# Get the mapping from token ID to token string
id2token = lda.id2word
print(id2token[1])


# ### Load the Mapping of Lower-Cased Vocabulary Items to Their Most Common Surface Form
# 

# Load surface form mappings here
fp = open(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "Vocab2SurfaceFormMapping.tsv"), encoding='utf-8')

vocabToSurfaceFormHash = {}

# Each line in the file has two tab separated fields;
# the first is the vocabulary item used during modeling
# and the second is its most common surface form in the 
# original data
for stringIn in fp.readlines():
    fields = stringIn.strip().split("\t")
    if len(fields) != 2:
        print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
    elif fields[0] == "" or fields[1] == "":
        print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
    else:
        vocabToSurfaceFormHash[fields[0]] = fields[1]
fp.close()


def CreateTermIDToSurfaceFormMapping(id2token, token2surfaceform):
    termIDToSurfaceFormMap = []
    for i in range(0, len(id2token)):
        if id2token[i] in token2surfaceform:
            termIDToSurfaceFormMap.append(token2surfaceform[id2token[i]])
    return termIDToSurfaceFormMap;

termIDToSurfaceFormMap = CreateTermIDToSurfaceFormMapping(id2token, vocabToSurfaceFormHash);


# print out the modeled token form and the best matching surface for the token with the index value of 18
i = 18
print('Term index:', i)
print('Modeled form:', id2token[i])
print('Surface form:', termIDToSurfaceFormMap[i])


# ### Use the Build-in <i> print_topics </i> Method to Summarize a Random Sample of 10 Topics
# 

numTopics = lda.num_topics
print ("Number of topics:", numTopics)

lda.print_topics(10)


# ### Use Word Cloud to Visualize a Topic
# 

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def _terms_to_counts(terms, multiplier=1000):
    return ' '.join([' '.join(int(multiplier * x[1]) * [x[0]]) for x in terms])


def visualizeTopic(lda, topicID=0, topn=500, multiplier=1000):
    terms = []
    tmp = lda.show_topic(topicID, topn)
    for term in tmp:
        terms.append(term)
    
    # If the version of wordcloud is higher than 1.3, then you will need to set 'collocations' to False.
    # Otherwise there will be word duplicates in the figure. 
    try:
        wordcloud = WordCloud(max_words=10000, collocations=False).generate(_terms_to_counts(terms, multiplier))
    except:
        wordcloud = WordCloud(max_words=10000).generate(_terms_to_counts(terms, multiplier))
    fig = plt.figure(figsize=(12, 16))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Topic %d" % topicID)
    plt.show()
    


get_ipython().run_line_magic('matplotlib', 'inline')


# Visualize topic \#38 using Word Cloud. This topic seems to be related to national security.
# 

visualizeTopic(lda, topicID=38, topn=1000)


# Visualize topic \#168 using Word Cloud. This topic is mainly related to health care.
# 

visualizeTopic(lda, topicID=168, topn=1000)


# ### Generate Various Required Probability Distributions
# 

# #### Load the Document Probability Score P(topic|doc) Computed by the LDA Model from File
# 
# In this section, each document from the corpus is passed into the LDA model which then infers the topic distribution for each document. The topic distributions are collected into a single numpy array.
# 

docTopicProbsFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicProbs.npy")

# docTopicProbs[docID,TopicID] --> P(topic|doc)
docTopicProbs = numpy.load(docTopicProbsFile)

# The docTopicProbs shape should be (# of docs, # of topics)
docTopicProbs.shape


# #### Compute the Global Topic Likelihood Scores P(topic)
# 

# Computing the global topic likelihoods by aggregating topic probabilities over all documents
# topicProbs[topicID] --> P(topic)
def ComputeTopicProbs(docTopicProbs):
    topicProbs = docTopicProbs.sum(axis=0) 
    topicProbs = topicProbs/sum(topicProbs)
    return topicProbs

topicProbs = ComputeTopicProbs(docTopicProbs)


# #### Convert the Topic Language Model Information P(term|topic) from the LDA Model into a NumPy Representation
# 

def ExtractTopicLMMatrix(lda):
    # Initialize the matrix
    docTopicProbs = numpy.zeros((lda.num_topics,lda.num_terms))
    for topicID in range(0,lda.num_topics):
        termProbsList = lda.get_topic_terms(topicID,lda.num_terms)
        for termProb in termProbsList:
            docTopicProbs[topicID,termProb[0]]=termProb[1]
    return docTopicProbs
    
# topicTermProbs[topicID,termID] --> P(term|topic)
topicTermProbs = ExtractTopicLMMatrix(lda)


# Set saveFile flag to true if you want to save the Topic LMs for a newly trained LDA model to file
if saveFile:
    numpy.save(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicLM.npy"), topicTermProbs)


# #### Compute P(topic,term), P(term), and P(topic|term)
# 

# Compute the joint likelihoods of topics and terms
# jointTopicTermProbs[topicID,termID] --> P(topic,term) = P(term|topic)*P(topic)
jointTopicTermProbs = numpy.diag(topicProbs).dot(topicTermProbs) 

# termProbs[termID] --> P(term)
termProbs = jointTopicTermProbs.sum(axis=0)

# topicProbsPermTerm[topicID,termID] --> P(topic|term)
topicProbsPerTerm = jointTopicTermProbs / termProbs


# Print the most frequent words in the LDA vocabulary. Compare it to Cell 11 in Notebook 3, and you will find that the most frequent words in LDA vocabulary are NOT the same as the most frequent words in corpus vocabulary. This is due to the fact that the probability derived from the LDA model do not account for document length, and therefore words common in a shorter documents carry more weight in these distributions than words common in longer documents.
# 

# Print most frequent words in LDA vocab
mostFrequentTermIDs = (-termProbs).argsort()
for i in range(0,25):
    print ("%d: %s --> %f" % (i+1, id2token[mostFrequentTermIDs[i]], termProbs[mostFrequentTermIDs[i]]))


# #### Compute WPMI
# 
# To determine which vocabulary terms are most representative of a topic, systems typically just choose a set of terms that are most likely for the topic, i.e., terms that maximize the language model expression <i>P(term|topic)</i> for the given topic. This approach is adequate for many data sets. However, for some data sets there may be common words in the corpus that are frequent terms within multiple topics, and hence not a distinguishing term for any of these topics. In this case, selecting words which have the largest weighted pointwise mutual information (WPMI) with a given topic is more appropriate. 
# 
# The expression for WPMI between a word and token is given as:
# 
# 
# $WPMI(term,topic) = P(term,topic)\log\frac{P(term,topic)}{P(term)P(topic)} = P(term,topic)\log\frac{P(topic|term)}{P(topic)}$
# 

topicTermWPMI =(jointTopicTermProbs.T * numpy.log(topicProbsPerTerm.T / topicProbs)).T
topicTermWPMI.shape


# #### Compute Topic to Document Purity measure for Each Topic
# 
# One measure of the importance or quality of a topic is its topic to document purity measure. This purity measure assumes latent topics that dominate the documents in which they appear are more semantically important than latent topics that are weakly spread across many documents. This concept was introduced in the paper ["Latent Topic Modeling for Audio Corpus Summarization"](http://people.csail.mit.edu/hazen/publications/Hazen-Interspeech11.pdf). The purity measure is expressed by the following equation:
# 
# $Purity(topic) = \exp\left (
#                  \frac{\sum_{\forall doc}P(topic|doc)\log P(topic|doc)}{\sum_{\forall doc}P(topic|doc)}
#                 \right )$
# 

topicPurity = numpy.exp(((docTopicProbs * numpy.log(docTopicProbs)).sum(axis=0))/(docTopicProbs).sum(axis=0))


# ### Create Topic Summaries 
# 

# In the code snippet below we demonstrate how the WPMI measure lowers the score of some common tokens that do not provide value in a topic summary in comparison to the standard word likely measure P(token|topic). For topic 38 below notice how the generic words <i>United States</i>, <i>including</i>, and <i>Government</i> have their position in the summaries lowered by the WPMI measure relative to the straight P(token|topic) measure, while the WMPI measure improves the ranking for the content bearing tokens <i>Security Act</i>, <i>security forces</i> and <i>counterterrorism</i>.
# 
# Again, this may not apply to your LDA model if you have retrained one.
# 

topicID = 38

highestWPMITermIDs = (-topicTermWPMI[topicID]).argsort()
highestProbTermIDs = (-topicTermProbs[topicID]).argsort()
print ("                                        WPMI                                                 Prob")
for i in range(0,15):
    print ("%2d: %35s ---> %8.6f    %35s ---> %8.6f" % (i+1, 
                                                        termIDToSurfaceFormMap[highestWPMITermIDs[i]], 
                                                        topicTermWPMI[topicID,highestWPMITermIDs[i]],
                                                        termIDToSurfaceFormMap[highestProbTermIDs[i]], 
                                                        topicTermProbs[topicID,highestProbTermIDs[i]]))                


def CreateTopicSummaries(topicTermScores, id2token, tokenid2surfaceform, maxStringLen):
    reIgnore = re.compile('^[a-z]\.$')
    reAcronym = re.compile('^[A-Z]+$')
    topicSummaries = []
    for topicID in range(0,len(topicTermScores)):
        rankedTermIDs = (-topicTermScores[topicID]).argsort()
        maxNumTerms = len(rankedTermIDs)
        termIndex = 0
        stop = 0
        outputTokens = []
        prevAcronyms = []
        topicSummary = ""
        while not stop:
            # If we've run out of tokens then stop...
            if (termIndex>=maxNumTerms):
                stop=1
            # ...otherwise consider adding next token to summary
            else:
                nextToken = id2token[rankedTermIDs[termIndex]]
                nextTokenOut = tokenid2surfaceform[rankedTermIDs[termIndex]]
                keepToken = 1
                
                # Prepare to test current word as an acronym or a string that reduces to an acronym
                nextTokenIsAcronym = 0
                nextTokenAbbrev = ""
                if reAcronym.match(nextTokenOut) != None:
                    nextTokenIsAcronym = 1
                else:
                    subTokens = nextToken.split('_')
                    if (len(subTokens)>1):
                        for subToken in subTokens:
                            nextTokenAbbrev += subToken[0]                        

                # See if we should ignore this token because it matches the regex for tokens to ignore
                if ( reIgnore.match(nextToken) != None ):
                    keepToken = 0;

                # Otherwise see if we should ignore this token because
                # it is a close match to a previously selected token
                elif len(outputTokens) > 0:          
                    for prevToken in outputTokens:
                        # Ignore token if it is a substring of a previous token
                        if nextToken in prevToken:
                            keepToken = 0
                        # Ignore token if it is a superstring of a previous token
                        elif prevToken in nextToken:
                            keepToken = 0
                        # Ignore token if it is an acronym of a previous token
                        elif nextTokenIsAcronym:
                            subTokens = prevToken.split('_')
                            if (len(subTokens)>1):
                                prevTokenAbbrev = ""
                                for subToken in subTokens:
                                    prevTokenAbbrev += subToken[0]
                                if prevTokenAbbrev == nextToken:
                                    keepToken = 0                                  
                    for prevAcronym in prevAcronyms:
                        # Ignore token if it is the long form of an earlier acronym
                        if nextTokenAbbrev == prevAcronym:
                                keepToken = 0

                # Add tokens to the summary for this topic                
                if keepToken:
                    # Always add at least one token to the summary
                    if len(topicSummary) == 0 or ( len(topicSummary) + len(nextTokenOut) + 1 < maxStringLen):
                        if len(topicSummary) == 0:
                            topicSummary = nextTokenOut
                        else: 
                            topicSummary += ", " + nextTokenOut
                        outputTokens.append(nextToken)
                        if nextTokenIsAcronym:
                            prevAcronyms.append(nextToken)
                    # If we didn't add the previous word and we're within 10 characters of 
                    # the max string length then we'll just stop here
                    elif maxStringLen - len(topicSummary) < 10 :
                        stop = 1
                    # Otherwise if the current token is too long, but we still have more than
                    # 10 characters of space left we'll just skip this one and add the next token
                    # one if it's short enough
                termIndex += 1         
        topicSummaries.append(topicSummary)
    return topicSummaries   
    
topicSummaries = CreateTopicSummaries(topicTermWPMI, id2token, termIDToSurfaceFormMap, 85)


# Rank the topics by their prominence score in the corpus
# The topic score combines the total weight of each a topic in the corpus 
# with a topic document purity score for topic 
# Topics with topicScore > 1 are generally very strong topics

topicScore = (numTopics * topicProbs) * (2 * topicPurity)
topicRanking = (-topicScore).argsort()


# ### Print Out Topical Summary of the Entire Corpus 
# 

print ("Rank  ID  Score  Prob  Purity  Summary")
for i in range(0, numTopics):
    topicID = topicRanking[i]
    print (" %3d %3d %6.3f (%5.3f, %4.3f) %s" 
           % (i+1, topicID, topicScore[topicID], 100*topicProbs[topicID], topicPurity[topicID], topicSummaries[topicID]))


# ### Save LDA Topic Summaries
# 

# If you want to save out the summaries to file makes saveFile flag True
if saveFile:
    fp = open(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicSummaries.tsv"), "w")
    i = 0
    fp.write("TopicID\tTopicSummary\n")
    for line in topicSummaries:
        fp.write("%d\t%s\n" % (i, line))
        i += 1
    fp.close()


# ### Next
# 
# The topic model summization step is finished. The next will be topic modeling analysis which will be in the fifth notebook of the series: [`5_Topic_Model_Analysis.ipynb`](5_Topic_Model_Analysis.ipynb).
# 

# ## Automatic Learning of Key Phrases and Topics in Document Collections
# 
# ## Part 5: Topic Modeling Analysis
# 
# ### Overview
# 
# This notebook is Part 5 of 6, in a series providing a step-by-step description of how to process and analyze the contents of a large collection of text documents in an unsupervised manner. Using Python packages and custom code examples, we have implemented the basic framework that combines key phrase learning and latent topic modeling as described in the paper entitled ["Modeling Multiword Phrases with Constrained Phrases Tree for Improved Topic Modeling of Conversational Speech"](http://people.csail.mit.edu/hazen/publications/Hazen-SLT-2012.pdf) which was originally presented in the 2012 IEEE Workshop on Spoken Language Technology.
# 
# Although the paper examines the use of the technology for analyzing human-to-human conversations, the techniques are quite general and can be applied to a wide range natural language data including news stories, legal documents, research publications, social media forum discussion, customer feedback forms, product reviews, and many more.
# 
# Part 5 of the series shows how to analysis the topical content of a collection of text documents and correlate topical information against other meta-data associated with the document collection. The topic model and topic summarizations were generated in Part 3 and Part 4 of the series.  
# 

# > **NOTE:** If you have retrained your own LDA model, you may not get the same topic model we are showing in this notebook. For the demonstration purpose, all files used in this notebook can be downloaded via the links below. You can download all files to the `AZUREML_NATIVE_SHARE_DIRECTORY` folder and you will have exactly the same results in this notebook.
# 

# 
# | File Name | Link |
# |-----------|------|
# | `CongressionalDocsLDA.pickle` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle |
# | `CongressionalDocsLDA.pickle.expElogbeta.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.expElogbeta.npy |
# | `CongressionalDocsLDA.pickle.id2word` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.id2word |
# | `CongressionalDocsLDA.pickle.state` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.state |
# | `CongressionalDocsLDA.pickle.state.sstats.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocsLDA.pickle.state.sstats.npy |
# | `CongressionalDocTopicLM.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicLM.npy |
# | `CongressionalDocTopicProbs.npy` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicProbs.npy |
# | `CongressionalDocTopicSummaries.tsv` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/CongressionalDocTopicSummaries.tsv |
# | `Vocab2SurfaceFormMapping.tsv` | https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/Vocab2SurfaceFormMapping.tsv |
# 

# ### Download Data Files (optional)
# 

# You can download all those data files by executing the code in the cells below.
# 

import urllib.request
import os

def download_file_from_blob(filename):
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    save_path = os.path.join(shared_path, filename)

    if not os.path.exists(save_path):
        # Base URL for anonymous read access to Blob Storage container
        STORAGE_CONTAINER = 'https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/'
        url = STORAGE_CONTAINER + filename
        urllib.request.urlretrieve(url, save_path)
        print("Downloaded file: %s" % filename)
    else:
        print("File \"%s\" already existed" % filename)


download_file_from_blob('CongressionalDocsLDA.pickle')
download_file_from_blob('CongressionalDocsLDA.pickle.expElogbeta.npy')
download_file_from_blob('CongressionalDocsLDA.pickle.id2word')
download_file_from_blob('CongressionalDocsLDA.pickle.state')
download_file_from_blob('CongressionalDocsLDA.pickle.state.sstats.npy')
download_file_from_blob('CongressionalDocTopicLM.npy')
download_file_from_blob('CongressionalDocTopicProbs.npy')
download_file_from_blob('CongressionalDocTopicSummaries.tsv')
download_file_from_blob('Vocab2SurfaceFormMapping.tsv')


# ### Import Relevant Python Packages
# 
# Part 5 primarily relies on the [matplotlib Python library](http://matplotlib.org) for generating graphs.
# 

import numpy as np
import pandas 
import re
import math
import warnings
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
from operator import itemgetter
from collections import namedtuple
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook5', 'true')


# ### Load Text Data
# 

# > **NOTE** The data file is saved under the folder defined by environment variable `AZUREML_NATIVE_SHARE_DIRECTORY` in notebook 1. If you have changed it to `../Data`, please also do the change here.
# 

# Load full TSV file including a column of text
docsFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDataAll_Jun_2017.tsv"), 
                            sep='\t')


print(docsFrame[90:100])


# ### Compute the Quarter Information for Each Document
# 
# A congressional session lasts 2 years. To summarize Congressional actions taken over the duration of a session, we break each two years session into its 8 annual quarters. We can then summarize actions over the duration of these 8 quarters for each session.
# 

# Break out the session number as a unique column in the documents frame 
# also create a column for the session quarter where we break dates down 
# into one of eight annual quarters per session, i.e., four quarters for 
# each of the two years in the session

reType = re.compile(r"^([a-z]+)[0-9]+$")

quarterArray = [] 
sessionArray = []
typeArray = []
for i in range(0,len(docsFrame)):

    dateFields = (docsFrame['Date'][i]).split('-')
    year = int(dateFields[0])
    month = int(dateFields[1])
    evenYear = int ((year % 2) == 0) 
    quarterArray.append(int((month - 1) / 3 ) + (evenYear * 4))

    idFields = (docsFrame['ID'][i]).split('-')
    
    billType = reType.match(idFields[0]).group(1)
    typeArray.append(billType)
    session = int(idFields[1])
    sessionArray.append(session)

# Add the meta-data entries into the data frame
docsFrame['Quarter'] = quarterArray
docsFrame['Session'] = sessionArray
docsFrame['Type'] = typeArray

# Extract the minimum session number in the data
minSessionNum = min(sessionArray)  


sessionQuarterIndex = []
for i in range(len(docsFrame)):
    session = docsFrame['Session'][i]
    quarter = docsFrame['Quarter'][i]
    sessionQuarterIndex.append(((session-minSessionNum)*8)+quarter)
    
docsFrame['SessionQuarterIndex'] = sessionQuarterIndex
maxSessionQuarterIndex = max(sessionQuarterIndex)
print("Total number of quarters over all sessions in data:", maxSessionQuarterIndex+1)


print(docsFrame[90:100])


# ### Plot Count of Total Congressional Actions Taken per Session Quarter
# 
# This plot shows the total number of Congressional actions proposed by congress aggregated over each of the eight annual quarters in a Congressional session over all sessions from 1973 until June 2017. Notice that the first quarter of a session after a new Congress starts, there is a large amount of activity. By the third annual quarter of the session, activity has subsided to a typical level. The final annual quarter in the second year of the session encompasses the election season and the two months of "lame duck" status for the Congress. During this time Congress typically does not engage in new legislative activity.
# 

totalDocsPerQuarter = []

for i in range(8):
    totalDocsPerQuarter.append(len(docsFrame[docsFrame['Quarter'] == i]))
print(totalDocsPerQuarter) 

N = len(totalDocsPerQuarter)
xvalues = np.arange(N)
xlabels = ['Y1/Q1', 'Y1/Q2', 'Y1/Q3', 'Y1/Q4', 'Y2/Q1', 'Y2/Q2', 'Y2/Q3', 'Y2/Q4']
plt.bar(xvalues, totalDocsPerQuarter, width=1.0, edgecolor="black")
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels)
plt.show()


totalBillsPerQuarter = []
totalResolutionsPerQuarter = []
isBill = (docsFrame['Type'] == 'hr') | (docsFrame['Type'] == 's')
isResolution = (isBill==False)

for i in range(8):  
    totalBillsPerQuarter.append(len(docsFrame[ (docsFrame['Quarter'] == i) & isBill ])) 
    totalResolutionsPerQuarter.append(len(docsFrame[ (docsFrame['Quarter'] == i) & isResolution ]))
        
totalResolutionsPerQuarter 


# This plot can be broken down into counts for bills versus resolutions.
# 

N = len(totalDocsPerQuarter)
xvalues = np.arange(N)
xlabels = ['Y1/Q1', 'Y1/Q2', 'Y1/Q3', 'Y1/Q4', 'Y2/Q1', 'Y2/Q2', 'Y2/Q3', 'Y2/Q4']
p1 = plt.bar(xvalues, totalResolutionsPerQuarter, color='g', edgecolor="black", width=1.0)
p2 = plt.bar(xvalues, totalBillsPerQuarter, width=1.0, color='b', edgecolor="black", bottom=totalResolutionsPerQuarter)
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels)
plt.legend((p1[0], p2[0]),('Resolutions', 'Bills'))
plt.show()


# ### Plot Count of Total Congressional Actions Taken per Unique Quarter
# 
# This plot shows the total number of Congressional actions proposed by congress within each unique quarter over all sessions from 1973 until 2017. 
# 
# 
# 

totalDocsPerUniqueQuarter = []
numQuarters = maxSessionQuarterIndex+1
for i in range(0,numQuarters):
    totalDocsPerUniqueQuarter.append(len(docsFrame[docsFrame['SessionQuarterIndex']==i]))
print(totalDocsPerUniqueQuarter) 


# Create label set which marks only the first quarter of each year with the year label
sessionQuarterLabels = []
for i in range(0,numQuarters):
    if ( i % 4 ) == 0:
        year = int((i/4) + 1973)
        sessionQuarterLabels.append(str(year))
    else:
        sessionQuarterLabels.append("")
        
print (sessionQuarterLabels)        


# Set the default figure size to be 15 in by 5 in
from pylab import rcParams
rcParams['figure.figsize'] = 15,5

# Create a function for plotting a topic over time
xlabels = sessionQuarterLabels
xvalues = np.arange(len(sessionQuarterLabels))
yvalues = totalDocsPerUniqueQuarter
    
plt.bar(xvalues, yvalues, width=1.0, edgecolor="black")
plt.title("Total Congressional Actions Per Quarter (1973-2017)")
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels, rotation=90)
plt.show()


# ### Plot Topics Over Time
# 
# Here we show how to aggregate the total amount of activity for any topic within each annual quarter over the entire span of the data from 1973 until 2017.
# 

# Load the topic distributions for all documents from file
ldaDocTopicProbsFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicProbs.npy")
docTopicProbs = np.load(ldaDocTopicProbsFile)
docTopicProbs.shape


# Aggregate the topic contributions for each document into topic bins for each quarter
numQuarters = maxSessionQuarterIndex + 1;
numTopics = docTopicProbs.shape[1]
numDocs = len(docsFrame)
topicQuarterRawCounts = np.zeros((numTopics, numQuarters))
for docIndex in range(0,numDocs):
    quarter = docsFrame['SessionQuarterIndex'][docIndex]
    for topicID in range(0,numTopics):
        topicQuarterRawCounts[topicID, quarter] += docTopicProbs[docIndex, topicID]


# Get the topic summaries to use as titles for each topic plot
ldaTopicSummariesFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicSummaries.tsv")
topicSummaries = pandas.read_csv(ldaTopicSummariesFile, sep='\t')


# Set the default figure size to be 15 by 5 in
from pylab import rcParams
rcParams['figure.figsize'] = 15, 5

# Create a function for plotting a topic over time
def PlotTopic(topicID, topicQuarterRawCounts, ylabel, xlabels, topicSummaries):
    xvalues = np.arange(len(xlabels))
    yvalues = topicQuarterRawCounts[topicID]
    
    plt.bar(xvalues, yvalues, width=1.0, edgecolor="black")
    plt.title(topicSummaries['TopicSummary'][topicID])
    plt.ylabel(ylabel)
    plt.xticks(xvalues+0.50, xlabels, rotation=90)
    plt.show()


# Plot topic 165 (which was the top ranked topic identified in Part 4)
PlotTopic(165, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)


# Show a plot of topic 140 which was identified as the sixth highest ranked topic in Part 4.
PlotTopic(140, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)


PlotTopic(38, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)


PlotTopic(168, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)


# ### Normalized Topic Plots
# 

# The topic plots above are based on aggregated topic distribution counts per quarter. Below we show how to nomalize those plots against an expected number of actions per quarter. This normalization, which we will refer it as anomalous activity score, hightlights quarters when there is an unusualy high amount of activity for topic.
# 

np.seterr(divide='ignore', invalid='ignore')

# This array contain the probability that a randomly selected document 
# came from a specific quarter for the data time span (1973-2017) 
probQuarter = np.array(totalDocsPerUniqueQuarter, dtype='f') / sum(totalDocsPerUniqueQuarter)

# This array contains the prior probability of a topic across the whole corpus
probTopic = docTopicProbs.sum(axis=0)[:, np.newaxis]
probTopic = probTopic / np.sum(probTopic)

# Compute the conditional probability of a topic given a specific quarter  
normTopicGivenQuarter = (np.sum(topicQuarterRawCounts, axis=0))[:, np.newaxis]
probTopicGivenQuarter = np.transpose(np.transpose(topicQuarterRawCounts) / normTopicGivenQuarter)

# Compute the conditional probability of a specific quarter given a topic
probQuarterGivenTopic = topicQuarterRawCounts / (np.sum(topicQuarterRawCounts,axis=1)[:, np.newaxis])

# Produce a "heat" indicator to highlight quarters for which a topic has higher than expected activity
topicHeatMap = 10000 * probQuarterGivenTopic * probTopicGivenQuarter * np.log((probQuarterGivenTopic / probQuarter))


# The plot below shows unusual activities for several quarters for topic 165, which contains bills related to Harmonized Tariff Schedule. During those quarters, many amendments were proposed by congressional members seeking exceptions to the Harmonized Tariff Schedule.
# 

PlotTopic(165, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)


# The plot below shows larger than expected activity in 2009 for a health care related topic. This is related to congressional actions proposed during the introduction of the Affordable Care Act.
# 

PlotTopic(168, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)


# The plot below shows larger than expected activity in 2002 for a national security topic. This is related to congressional actions proposed in the wake of the 9/11 terror attack.
# 

PlotTopic(38, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)


# ### Load Topic Language Models and Summaries from File
# 

topicTermProbs = np.load(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicLM.npy"))
topicTermProbs.shape


# ### Compute Topic Similarity and Find Similar Topics to A Reference Topic
# 

# Topic Similarity 
# First compute unit normalized vectors
normVector = np.matrix(np.sqrt(np.sum(np.array(topicTermProbs) * np.array(topicTermProbs), axis=1))).transpose()
topicTermProbsUnitNormed = np.matrix(np.array(topicTermProbs) / np.array(normVector))

# Compute topicSimilarity using cosine simlarity measure
topicSimilarity = topicTermProbsUnitNormed * topicTermProbsUnitNormed.transpose()
topicSimilarity.shape


print(topicSimilarity)


# ### Print Similar Topics to a Reference Topic
# 

def PrintSimilarTopics(topicID, topicSimilarity, topicSummaries, topN):
    sortedTopics = np.array(np.argsort(-topicSimilarity[topicID]))[0]
    for i in range(topN+1):
        print ("%4.3f %3d : %s" % (topicSimilarity[topicID,sortedTopics[i]], 
                                   sortedTopics[i], 
                                   topicSummaries['TopicSummary'][sortedTopics[i]]))


PrintSimilarTopics(38, topicSimilarity, topicSummaries, 10)


PrintSimilarTopics(168, topicSimilarity, topicSummaries, 10)


# ### Plot a Dendrogram of Hierarchically Clustered Topics
# 

topicDistances = -np.log2(topicSimilarity)
# For some reason diagonal elements are not exactly zero...so force them to zero
for i in range(0,numTopics):
    topicDistances[i,i]=0

# Extract the upper right diagonal of topicDistances into a condensed 
# distance format for clustering and pass it into the hierarchical 
# clustering algorithm using the max (or 'complete') distance metric
topicClustering=linkage(ssd.squareform(topicDistances), 'complete')


# Plot a dendrogram of the hierarchical clustering of topics
def PlotTopicDendrogram(topicClustering, topicSummaries):    
    numTopics = len(topicSummaries)
    if numTopics != len(topicClustering) + 1:
        print ("Error: Number of topics in topic label set (%d) and topic clustering (%d) are not equal"
               % (numTopics, len(topicClustering) + 1)
              )
        return
    height = int(numTopics/4)
    
    plt.figure(figsize=(10,height))
    plt.title('Topic Dendrogram')
    plt.xlabel('Topical Distance')
    dendrogram(topicClustering, leaf_font_size=12, orientation='right', labels=topicSummaries)
    plt.show()
    return
    
PlotTopicDendrogram(topicClustering,list(topicSummaries['TopicSummary']))


# ### Next
# 
# The topic model analysis step is finished. The next will be some cool interactive visualization which will be in the sixth notebook of the series: [6_Interactive_Visualization.ipynb](6_Interactive_Visualization.ipynb).
# 

# # Automatic Learning of Key Phrases and Topics in Document Collections
# 
# ## Part 1: Text Preprocessing
# 
# ### Overview
# 
# This notebook is Part 1 in a series of 6, providing a step-by-step description of how to process and analyze the contents of a large collection of text documents in an unsupervised manner. Using Python packages and custom code examples, we have implemented the basic framework that combines key phrase learning and latent topic modeling as described in the paper entitled ["Modeling Multiword Phrases with Constrained Phrases Tree for Improved Topic Modeling of Conversational Speech"](http://people.csail.mit.edu/hazen/publications/Hazen-SLT-2012.pdf) which was originally presented in the 2012 IEEE Workshop on Spoken Language Technology.
# 
# This notebook demonstrates how to preprocess the raw text from a collection of documents as precursor to applying the natural language processing techniques of unsupervised phrase learning and latent topic modeling.
# 
# 

# **These series of notebooks can be run on any compute context. Before you run those notebooks, make sure you have all dependencies installed in the compute context you choose as kernel.**
# 
# * For **local** kernel, click "_**Open Command Prompt**_" from "_**File**_" menu in Azure Machine Learning Workbench, and then manually install the following packages:
# ```
# $ conda install numpy
# $ conda install nltk
# $ conda install -c conda-forge wordcloud
# $ conda install bokeh
# $ pip install gensim
# $ pip install matplotlib
# ```
# 
# * For local or remote **Docker kernels**:
#     * ensure **notebook**, **matplotlib**, **nltk**, **gensim**, **wordcloud** are listed in your **conda_dependencies.yml** file under **aml_config** folder.
#     ```
#         name: project_environment
#         channels:
#           - conda-forge
#           - defaults
#         dependencies:
#           - python=3.5.2
#           - numpy>=1.13
#           - scikit-learn
#           - nltk
#           - pandas
#           - azure
#           - gensim
#           - scipy
#           - wordcloud
#           - bokeh
#           - pip:
#             - notebook
#             - gensim
#             - matplotlib
#     ```
# 

# ### Import Relevant Python Packages
# 

# #### Importing NLTK Model for Sentence Tokenization
# 
# 
# NLTK is a collection of Python modules, prebuilt models and corpora that provides tools for complex natural language processing tasks. Because the toolkit is large, the base installation of NLTK only installs the core skeleton of the toolkit. Installation of specific modules, corpora and pre-built models can be invoked from within Python using a download functionality provided by NLTK that can be invoked from Python. 
# 
# In this notebook, we make use of the NLTK sentence tokenization capability which takes a long string of text and splits it into sentence units. The tokenizer requires the installation of the 'punkt'  tokenizer models. After importing nltk, the nltk.download() function can be used to download specific packages such as 'punkt'.
# 
# For more information on NLTK see http://www.nltk.org/.
# 

import os
import urllib.request
import nltk
# The first time you run NLTK you will need to download the 'punkt' models 
# for breaking text strings into individual sentences
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk import tokenize

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook1', 'true')


# #### Import Other Required Packages
# The 'pandas' package is used for handling and manipulating data frames. The 're' package is used for applying regular expressions.
# 

import pandas 
import re


# ### Load Text Data
# 

# Need to download the datasets from Blob Storage.
# 

# > **NOTE**: If you are running this notebook outside of Azure Machine Learning Workbench, you will need to change the `'shared_path'` to relative path `'../Data'`.
# 

def download_file_from_blob(filename):
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    save_path = os.path.join(shared_path, filename)

    # Base URL for anonymous read access to Blob Storage container
    STORAGE_CONTAINER = 'https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/'
    url = STORAGE_CONTAINER + filename
    urllib.request.urlretrieve(url, save_path)
    
    
def getData():
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']

    data_file = os.path.join(shared_path, DATASET_FILE)
    blacklist_file = os.path.join(shared_path, BLACK_LIST_FILE)
    function_words_file = os.path.join(shared_path, FUNCTION_WORDS_FILE)

    if not os.path.exists(data_file):
        download_file_from_blob(DATASET_FILE)
    if not os.path.exists(blacklist_file):
        download_file_from_blob(BLACK_LIST_FILE)
    if not os.path.exists(function_words_file):
        download_file_from_blob(FUNCTION_WORDS_FILE)

    df = pandas.read_csv(data_file, sep='\t')
    return df


# Define constants used to get data from Blog Storage and read it as Pandas DataFrame.
# 

# The dataset file name
# DATASET_FILE = 'small_data.tsv'
DATASET_FILE = 'CongressionalDataAll_Jun_2017.tsv'

# The black list of words to ignore
BLACK_LIST_FILE = 'black_list.txt'

# The non-content bearing function words
FUNCTION_WORDS_FILE = 'function_words.txt'


# > **NOTE:** By default, this notebook will use the entire congressional dataset which contains about 290,000 bills. There is also an option to comment the line `DATASET_FILE = 'CongressionalDataAll_Jun_2017.tsv'` and uncomment line `DATASET_FILE = 'small_data.tsv'` on the above cell to run it on a dataset with 50,000 bills. Use a small dataset could significantly reduce the execution time of downstream notebooks.
# 

# Load full TSV file including a column of text
# 

frame = getData()


print("Total documents in corpus: %d\n" % len(frame))

# Show the first five rows of the data in the frame
frame[0:5]


# Print the full text of the first three documents
# 

print(frame['Text'][0])
print('---')
print(frame['Text'][1])
print('---')
print(frame['Text'][2])


# ### Preprocess Text Data
# 
# The CleanAndSplitText function below takes as input a list where each row element is a single cohesive long string of text, i.e. a "document". The function first splits each string by various forms of punctuation into chunks of text that are likely sentences, phrases or sub-phrases. The splitting is designed to prohibit the phrase learning process from using cross-sentence or cross-phrase word strings when learning phrases.
# 
# The function creates a table where each row represents a chunk of text from the original documents. The DocIndex column indicates the original row index from associated document in the input from which the chunk of text originated. The TextLine column contains the original text excluding the punctuation marks and HTML markup that have been during the cleaning process.The TextLineLower column contains a fully lower-cased version of the text in the TextLIne column.
# 

def CleanAndSplitText(textDataFrame):

    textDataOut = [] 
   
    # This regular expression is for section headers in the bill summaries that we wish to ignore
    reHeaders = re.compile(r" *TABLE OF CONTENTS:? *"
                           "| *Title [IVXLC]+:? *"
                           "| *Subtitle [A-Z]+:? *"
                           "| *\(Sec\. \d+\) *")

    # This regular expression is for punctuation that we wish to clean out
    # We also will split sentences into smaller phrase like units using this expression
    rePhraseBreaks = re.compile("[\"\!\?\)\]\}\,\:\;\*\-]*\s+\([0-9]+\)\s+[\(\[\{\"\*\-]*"                             
                                "|[\"\!\?\)\]\}\,\:\;\*\-]+\s+[\(\[\{\"\*\-]*"
                                "|\.\.+"
                                "|\s*\-\-+\s*"
                                "|\s+\-\s+"
                                "|\:\:+"
                                "|\s+[\/\(\[\{\"\-\*]+\s*"
                                "|[\,!\?\"\)\(\]\[\}\{\:\;\*](?=[a-zA-Z])"
                                "|[\"\!\?\)\]\}\,\:\;]+[\.]*$"
                             )
    
    # Regex for underbars
    regexUnderbar = re.compile('_')
    
    # Regex for space
    regexSpace = re.compile(' +')
 
    # Regex for sentence final period
    regexPeriod = re.compile("\.$")

    # Iterate through each document and do:
    #    (1) Split documents into sections based on section headers and remove section headers
    #    (2) Split the sections into sentences using NLTK sentence tokenizer
    #    (3) Further split sentences into phrasal units based on punctuation and remove punctuation
    #    (4) Remove sentence final periods when not part of an abbreviation 

    for i in range(0, len(frame)):     
        # Extract one document from frame
        docID = frame['ID'][i]
        docText = str(frame['Text'][i])

        # Set counter for output line count for this document
        lineIndex=0;

        # Split document into sections by finding sections headers and splitting on them 
        sections = reHeaders.split(docText)
        
        for section in sections:
            # Split section into sentence using NLTK tokenizer 
            sentences = tokenize.sent_tokenize(section)
            
            for sentence in sentences:
                # Split each sentence into phrase level chunks based on punctuation
                textSegs = rePhraseBreaks.split(sentence)
                numSegs = len(textSegs)
                
                for j in range(0,numSegs):
                    if len(textSegs[j])>0:
                        # Convert underbars to spaces 
                        # Underbars are reserved for building the compound word phrases                   
                        textSegs[j] = regexUnderbar.sub(" ",textSegs[j])
                    
                        # Split out the words so we can specially handle the last word
                        words = regexSpace.split(textSegs[j])
                        phraseOut = ""
                        # If the last word ends in a period then remove the period
                        words[-1] = regexPeriod.sub("", words[-1])
                        # If the last word is an abbreviation like "U.S."
                        # then add the word final period back on
                        if "\." in words[-1]:
                            words[-1] += "."
                        phraseOut = " ".join(words)  

                        textDataOut.append([docID, lineIndex, phraseOut])
                        lineIndex += 1
                        
    # Convert to pandas frame 
    frameOut = pandas.DataFrame(textDataOut, columns=['DocID', 'DocLine', 'CleanedText'])                      
    
    return frameOut


# Set this to true to run the function
writeFile = True

if writeFile:
    cleanedDataFrame = CleanAndSplitText(frame)


# #### Writing and reading text data to and from a file 
# 

# Writing the text data to file and reading it back in. If the value is 'False' it assumes we have already run the CleanAndSplitData function and written it to file.
# 

cleanedDataFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsCleaned.tsv')

if writeFile:
    # Write frame with preprocessed text out to TSV file 
    cleanedDataFrame.to_csv(cleanedDataFile, sep='\t', index=False)
else:
    # Read a cleaned data frame in from a TSV file
    cleanedDataFrame = pandas.read_csv(cleanedDataFile, sep='\t', encoding="ISO-8859-1")


# #### Examining the processed text data
# 

cleanedDataFrame[0:25]


print(cleanedDataFrame['CleanedText'][0])
print(cleanedDataFrame['CleanedText'][1])
print(cleanedDataFrame['CleanedText'][2])


# ### Next
# 
# The data preprocessing step is finished. The next step will be phrase learning which will be in the second notebook of the series: [`2_Phrase_Learning.ipynb`](./2_Phrase_Learning.ipynb).
# 

