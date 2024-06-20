from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# GLOVE Vectors
import torchtext.vocab as vocab

use_cuda = torch.cuda.is_available()
print(" I have a GPU? ", use_cuda)


# ## Installing torchtext
# 
# The torchtext package is not currently on the PIP or Conda package managers, but it's easy to install manually:
# may require sudo. 
# 
# ```
# git clone https://github.com/pytorch/text pytorch-text
# cd pytorch-text
# python setup.py install
# ```
# 

glove = vocab.GloVe(name='6B', dim=50)

print('Loaded {} words'.format(len(glove.itos)))


# The returned `GloVe` object includes attributes:
# - `stoi` _string-to-index_ returns a dictionary of words to indexes
# - `itos` _index-to-string_ returns an array of words by index
# - `vectors` returns the actual vectors. To get a word vector get the index to get the vector:
# 

glove.vectors[0][:5]


# ## Can skip the preprocessign step and just load the pickle file
# 

class Lang:
    
    def __init__(self, name):
        
        '''
        Store the string token to index token
        mapping in the word2index and index2word
        dictionaries. 
        '''
        
        self.name = name
        self.trimmed = False # gets changed to True first time Lang.trim(min_count) is called
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

    def index_sentence(self, sentence):
        '''
        Absorbs a sentence string into the token dictionary
        one word at a time using the index_word function
        increments the word count dictionary as well
        '''
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    
    def trim(self, min_count):
        '''
        Removes words from our 3 dictionaries that
        are below a certain count threshold (min_count)
        '''
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        for word in keep_words:
            self.index_word(word)
            
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
            
    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub("newlinechar", "", s)
        s = ' '.join(s.split())
        return s

    def filterPair(self, p, max_sent_len, min_sent_len):
        
        '''
        Your Preferences here
        '''

        return len(p[0].split(' ')) < max_sent_len and                len(p[1].split(' ')) < max_sent_len and                len(p[1].split(' ')) > min_sent_len and                "https://" not in p[1] 

    
    def make_pairs(self, path_to_tab_sep_dialogue, 
                   max_sent_len = 20, min_sent_len = 4):

        print("making final_pairs list ...")
        lines = open(path_to_tab_sep_dialogue).read().strip().split('\n')
        
        final_pairs = []
        i = 0
        for l in lines:
            
            pair = [self.normalize_string(sentence) for sentence in l.split('\t')]
            
            if self.filterPair(pair,max_sent_len, min_sent_len):
                
                filtered_pair = []
                
                for sentence in pair:

                    self.index_sentence(sentence)
                    filtered_pair.append(sentence)
                  
                final_pairs.append(filtered_pair)
        print("number of pairs", len(final_pairs))
        return final_pairs
    
    def tokens2glove(self, min_word_count,glove, mbed_dim = 50):
    
        print("trimming...")    
        self.trim(min_word_count)
        
        if glove is None:
            glove = vocab.GloVe(name='6B', dim=embed_dim)
            print('Loaded {} words'.format(len(glove.itos)))
        else:
            embed_dim = glove.vectors.size(1)
                    
        print("building embedding from glove...")
        embedding = np.zeros((len(self.index2word), embed_dim)).astype(np.float32)
        for i in range(self.num_nonwordtokens):
            embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)
        for i in range(self.num_nonwordtokens,len(self.index2word)):
            if self.index2word[i] in glove.stoi:
                embedding[i,:] = glove.vectors[glove.stoi[self.index2word[i]]]
            else:
                embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)
        
        return self.index2word, self.word2index, embedding, self.n_words #torch.from_numpy(embeddings).float() 
    


lang = Lang("chat")
final_pairs = lang.make_pairs("../data/input-output.txt", max_sent_len = 30, min_sent_len = 1)
index2word, word2index, embedding, vocab_size = lang.tokens2glove(10,glove)


index2word, word2index, embedding, final_pairs, vocab_size  = pickle.load( open( "saved_pickle/mx_sent_30_min_sent_1_min_wrd_10_dim50.p", "rb" ) )


mx_sent_30_min_sent_1_min_wrd_10_dim50 = (index2word, word2index, embedding, final_pairs, vocab_size)


pickle.dump(mx_sent_30_min_sent_1_min_wrd_10_dim50, open( "saved_pickle/mx_sent_30_min_sent_1_min_wrd_10_dim50.p", "wb" ) )


lang = Lang("chat")
lang.index2word = index2word
lang.word2index = word2index


# ## Makesure the GloVe vectors work the way they are supposed to
# 

np.dot(embedding[word2index["admire"]],embedding[word2index["respect"]])


np.dot(embedding[word2index["admire"]],embedding[word2index["spoon"]])


np.dot(embedding[word2index["fork"]],embedding[word2index["spoon"]])


##########  Converts [" input string ", " output string "] (pair) , appends <EOS> index and returns indices ######

def indexesFromSentence(lang, sentence):
    '''
    account for strings not in the vocabulary by using the unknown token
    '''
    sentence_as_indices = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            sentence_as_indices.append(lang.word2index[word])
        else:
            sentence_as_indices.append(lang.UNK_token)
    
    return sentence_as_indices


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair,lang):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)

######## the pair indices are returned as 2 LongTensor Variables in torch #############


######## Tells you how long youve been training and how much longer you have left ####

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################3

############### plot_losses #######################################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
####################################################################


class EncoderRNN(nn.Module):
    
    def __init__(self, hidden_size, embedding,
                 num_layers = 3, bidirectional = False, train_embedding = True):
        
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        embedding = torch.from_numpy(embedding).float()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding, requires_grad=train_embedding)
        self.gru = nn.GRU(embedding.shape[1], hidden_size, num_layers, bidirectional=bidirectional)
        
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        # make the initial hidden state learnable as well 
        hidden0 = torch.zeros(self.num_layers*num_directions, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs
            
        return output, hidden

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0


class AttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, embedding, dropout_p=0.1, max_length=30, 
                 num_layers = 3, train_embedding = True):
        
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        embedding = torch.from_numpy(embedding).float()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding, requires_grad=train_embedding)
        
        self.attn = nn.Linear(self.hidden_size + embedding.shape[1], self.max_length)
        
        self.attn_combine = nn.Linear(self.hidden_size + embedding.shape[1], self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = num_layers)
        self.out = nn.Linear(self.hidden_size, embedding.shape[0])
        
        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        attn = self.attn(cat)
        
        attn_weights = F.softmax(attn, dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0



def train(lang, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=30, teacher_forcing_ratio = 0.5, bidirectional = False):
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]

    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[lang.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:] 
        
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == lang.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, pairs, lang, n_iters =1000, print_every=100, plot_every=100,
               learning_rate=0.01, teacher_forcing_ratio = 0.5, bidirectional = False,
               name = "noname", lowest_loss =100, gamma = 0.95):
    
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    escheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=print_every, gamma=gamma) 
    dscheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=print_every, gamma=gamma) 
    
    training_pairs = [variablesFromPair(random.choice(pairs),lang)
                      for i in range(n_iters)]
    
    criterion = nn.NLLLoss()

    lowest_loss = lowest_loss
    
    for iter in range(1, n_iters + 1):
        
        escheduler.step()
        dscheduler.step()
        
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(lang, input_variable, target_variable, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion, 
                     teacher_forcing_ratio = teacher_forcing_ratio,
                     bidirectional =  bidirectional)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
            if lowest_loss > print_loss_avg:
                lowest_loss = print_loss_avg
                print("new lowest loss, saving...")
                torch.save(encoder.state_dict(), "saved_params/encoder"+name+".pth")
                torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+".pth")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


hidden_size = 64
num_layers = 2
bidirectional = False 

encoder = EncoderRNN(hidden_size, embedding, num_layers = num_layers, bidirectional = bidirectional,
                     train_embedding = True)


attn_decoder = AttnDecoderRNN(hidden_size,  embedding, dropout_p=0.1, num_layers = num_layers,
                              train_embedding = True)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()
    
#name = "_256h_4L_bi_glove"
version = 0
name = "hidden" + str(hidden_size) + "L" + str(num_layers) + "bidir"+str(bidirectional) + "v"+ str(version)


encoder.load_state_dict(torch.load("saved_params/encoder"+name+"cpu.pth"))
attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+name+"cpu.pth"))


trainIters(encoder, attn_decoder, final_pairs, lang, n_iters = 100000, print_every=1000, 
           learning_rate=0.01, teacher_forcing_ratio = 1.0,
           bidirectional = bidirectional, name = name,
           lowest_loss = 6.0, gamma = 0.95)


#name = "_256h_4L_bi_glove"
torch.save(encoder.state_dict(), "saved_params/encoder"+name+"cpu.pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+"cpu.pth")


evaluateRandomly(final_pairs, encoder, attn_decoder,  n=10, bidirectional = bidirectional)

evaluateAndShowAttention("what does this mean where are we and where are we going ?",
                         encoder, attn_decoder, bidirectional = bidirectional  )

evaluateAndShowAttention("why is this happening ?",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("tell me something stupid",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("you can make it if you try",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("what link ?",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("hello",encoder, attn_decoder, bidirectional = bidirectional)


def evaluate(encoder, decoder, sentence, max_length=30,
             bidirectional =bidirectional):
    
    
    input_variable = variableFromSentence(lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[lang.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:]

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == lang.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(pairs, encoder, decoder, n=10, bidirectional = False):
    for i in range(n):
        pair = random.choice(pairs)
        print('input from data >', pair[0])
        print('output from data=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], bidirectional = bidirectional)
        output_sentence = ' '.join(output_words)
        print('bot response <', output_sentence)
        print('')
        
def showAttention(input_sentence, output_words, attentions, bidirectional = False):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence,encoder, attn_decoder, bidirectional = False):
    
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence, bidirectional = bidirectional)
    
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)





from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print(" I have a GPU? ", use_cuda)


# # Skip the data Pre-Processing 
# 
# ### you already have the pickle files, just run the 2 cells below to collect the 3 pickle files that are the training data, they are just Lang Class instances with the word2index and index2word mappings and n_words for vocab size, the class Lang must be defined before the pickle and populate it with data. 
# 

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Maximum length of the sequences you are mapping 
MAX_LENGTH = 20 #50
MIN_LENGTH = 4 #2

# start of sentence and end of sentence indices
SOS_token = 0
EOS_token = 1


input_lang = pickle.load( open( "saved_pickle/input_lang_4_20.p", "rb" ) )
output_lang = pickle.load( open(  "saved_pickle/output_lang_4_20.p", "rb" ) )
pairs = pickle.load( open( "saved_pickle/pairs_4_20.p", "rb" ) )

# see the way it works

#input_lang.n_words, input_lang.word2index["froid"], input_lang.index2word[33]
output_lang.word2index["?"]


print("number of training examples",len(pairs))


# ## Dont have to run these, the pre-processing from ../data folder/input-output.txt
#  final outputs of the preprocessing steps, they are just Lang Class instances with the word2index and index2word mappings and n_words for vocab size, the class Lang must be defined before the pickle and populate it with data. 
# 

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub("newlinechar", "", s)
    return s



def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    '''
    Your Preferences here
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and            len(p[1].split(' ')) < MAX_LENGTH and            len(p[1].split(' ')) > MIN_LENGTH and            "https://" not in p[1] 
        


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False, Filter = False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if Filter:
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


#input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#print(random.choice(pairs))
#input_lang.index2word[15] # input is french

#pickle.dump( input_lang, open( "saved_pickle/input_lang.p", "wb" ) )
#pickle.dump( output_lang, open( "saved_pickle/output_lang.p", "wb" ) )
#pickle.dump( pairs, open( "saved_pickle/pairs.p", "wb" ) )

input_lang, output_lang, pairs = prepareData('input', 'output', reverse=False, Filter = True)


pickle.dump( input_lang, open( "saved_pickle/input_lang_4_20.p", "wb" ) )
pickle.dump( output_lang, open( "saved_pickle/output_lang_4_20.p", "wb" ) )
pickle.dump( pairs, open( "saved_pickle/pairs_4_20.p", "wb" ) )


# # Start here after you have the pickle files, skip above
# 
# ## Just some Helper functions used in the Main training loop and main model 
# 
# ### Example training output
# 
# time elapsed, estimated time remaining given, and progress %, Loss
# 
# 1m 52s (- 26m 10s) (5000 6%) 2.8937 
# 
# 3m 40s (- 23m 50s) (10000 13%) 2.3437
# 
# 5m 31s (- 22m 4s) (15000 20%) 2.0382
# 


##############  Converts [" input string ", " output string "] (pair) , appends <EOS> index and returns indices ######

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

######## the pair indices are returned as 2 LongTensor Variables in torch #############


######## Tells you how long youve been training and how much longer you have left ####

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################3

############### plot_losses #######################################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
####################################################################


# ## GRU - The Gated recurrent unit - FYI
# 
# Here is the way torch formulates each layer of the GRU
# 
# \begin{split}\begin{array}{ll}
# r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\end{array}\end{split}
# 
# where h_t is the hidden state at time t, x_t is the hidden state of the previous layer at time t or input_t for the first layer, and r_t, z_t, n_t are the reset, input, and new gates, respectively. σ is the sigmoid function
# 
# ### Instantiation Parameters:	
# 
# input_size – The number of expected features in the input x 
# 
# hidden_size – The number of features in the hidden state h
# 
# num_layers – Number of recurrent layers.
# 
# bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# 
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
# 
# dropout – If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
# p – probability of an element to be zeroed. Default: 0.5
# 
# bidirectional – If True, becomes a bidirectional RNN. Default: False
# 
# 
# #### Inputs: input, h_0
# 
# input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
# 
# h_0 (num_layers * num_directions(2 for bidirectional), batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.
# 
# #### Outputs: output, h_n
# 
# output (seq_len, batch, hidden_size * num_directions): tensor containing the output features h_t from the last layer of the RNN, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
# 
# h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
# 

# # Encoder and Decoder in next 2 cells
# 
# ### the class for these models needs to be defined to load parameters into them
# 

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers = 3, bidirectional = False):
        
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirectional)
        
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        # make the initial hidden state learnable as well 
        hidden0 = torch.zeros(self.num_layers*num_directions, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs
            
        return output, hidden

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, num_layers = 3):
        
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0


# # Training
# 
# ## trainIters() takes the encoder instance, decoder instance and trains it using the helper function train()
# 


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH, teacher_forcing_ratio = 0.5, bidirectional = False):
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:] 
        
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01, teacher_forcing_ratio = 0.5, bidirectional = False,
               name = "noname", lowest_loss =100, gamma = 0.95):
    
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    escheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=print_every, gamma=gamma) 
    dscheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=print_every, gamma=gamma) 
    
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    
    criterion = nn.NLLLoss()

    lowest_loss = lowest_loss
    
    for iter in range(1, n_iters + 1):
        
        escheduler.step()
        dscheduler.step()
        
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion, 
                     teacher_forcing_ratio = teacher_forcing_ratio,
                     bidirectional =  bidirectional)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
            if lowest_loss > print_loss_avg:
                lowest_loss = print_loss_avg
                print("new lowest loss, saving...")
                torch.save(encoder.state_dict(), "saved_params/encoder"+name+".pth")
                torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+".pth")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


# # Instantiate The Encoder and Decoder 
# 
# ### move the model to GPU 
# 

hidden_size = 128#256
num_layers = 2#3
bidirectional = False#True
encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers = num_layers, bidirectional = bidirectional)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, num_layers = num_layers)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()


# ## Run this cell if you have saved parameters you want to load into the encoder decoder
# 

#encoder.load_state_dict(torch.load("saved_params/encoder.pth"))
#attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder.pth"))
encoder.load_state_dict(torch.load("saved_params/encoder_2L_h128_uni3.pth"))
attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder_2L_h128_uni3.pth"))


# ## call to the MAIN TRAINING LOOP and cell for storing parameters
# 

trainIters(encoder, attn_decoder, n_iters = 100000, print_every=10000, 
           learning_rate=0.001, teacher_forcing_ratio = 0.9,
           bidirectional = bidirectional, name = "_2L_h128_uni3",
           lowest_loss = 4.1, gamma = 0.95) # last loss 4.3393, 16 secs per 100 iters, so ~ 22500 iters/hr


# If you want to save the results of your training
torch.save(encoder.state_dict(), "saved_params/encoder_2L_h128_uni3.pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder_2L_h128_uni3.pth")


# # Inference/Prediction/Chat with me
# 
# ### evaluate() is your main inference function to deploy the encoder-decoder as a chatbot
# 
# ### evaluateRandomly() calls evaluate() to give you a sampling of dialogue
# 

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH,
             bidirectional =bidirectional):
    
    
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:]

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10, bidirectional = False):
    for i in range(n):
        pair = random.choice(pairs)
        print('input from data >', pair[0])
        print('output from data=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], bidirectional = bidirectional)
        output_sentence = ' '.join(output_words)
        print('bot response <', output_sentence)
        print('')


evaluateRandomly(encoder, attn_decoder,  n=10, bidirectional = bidirectional)


bidirectional = False #True

def showAttention(input_sentence, output_words, attentions, bidirectional = False):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence,encoder, attn_decoder, bidirectional = False):
    
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence, bidirectional = bidirectional)
    
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("what does this mean where are we and where are we going ?",
                         encoder, attn_decoder, bidirectional = bidirectional  )

evaluateAndShowAttention("why is this happening ?",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("tell me something stupid",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("you can make it if you try",encoder, attn_decoder, bidirectional = bidirectional)


# # Reinforcement Learning
# 

#### TO DO: ####

##############


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import numpy as np
import scipy.spatial
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# GLOVE Vectors
import torchtext.vocab as vocab

use_cuda = torch.cuda.is_available()
print(" I have a GPU? ", use_cuda)


# ## Installing torchtext
# 
# The torchtext package is not currently on the PIP or Conda package managers, but it's easy to install manually:
# may require sudo. 
# 
# ```
# git clone https://github.com/pytorch/text pytorch-text
# cd pytorch-text
# python setup.py install
# ```
# 

glove = vocab.GloVe(name='6B', dim=50)

print('Loaded {} words'.format(len(glove.itos)))


# The returned `GloVe` object includes attributes:
# - `stoi` _string-to-index_ returns a dictionary of words to indexes
# - `itos` _index-to-string_ returns an array of words by index
# - `vectors` returns the actual vectors. To get a word vector get the index to get the vector:
# 

glove.vectors[0][:5]


# ## Can skip the preprocessign step and just load the pickle file
# 

class Lang:
    
    def __init__(self, name):
        
        '''
        Store the string token to index token
        mapping in the word2index and index2word
        dictionaries. 
        '''
        
        self.name = name
        self.trimmed = False # gets changed to True first time Lang.trim(min_count) is called
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

    def index_sentence(self, sentence):
        '''
        Absorbs a sentence string into the token dictionary
        one word at a time using the index_word function
        increments the word count dictionary as well
        '''
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    
    def trim(self, min_count):
        '''
        Removes words from our 3 dictionaries that
        are below a certain count threshold (min_count)
        '''
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = len(self.index2word) # Count default tokens
        self.num_nonwordtokens = len(self.index2word)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        for word in keep_words:
            self.index_word(word)
            
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
            
    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub("newlinechar", "", s)
        s = s.replace("'","")
        s = s.replace(".","")
        s = s.replace("n t ","nt ")
        s = s.replace("i m ","im ")
        s = s.replace("t s ","ts ")
        s = s.replace(" s ","s ")
        s = s.replace(" re "," are ")
        s = s.replace("i ve ","ive ")
        s = s.replace(" d ","d ")
        s = ' '.join(s.split())
        return s

    def filterPair(self, p, max_sent_len, min_sent_len):
        
        '''
        Your Preferences here
        '''

        return len(p[0].split(' ')) < max_sent_len and                len(p[1].split(' ')) < max_sent_len and                len(p[1].split(' ')) > min_sent_len and                "https://" not in p[1] 

    
    def make_pairs(self, path_to_tab_sep_dialogue, 
                   max_sent_len = 20, min_sent_len = 4):

        print("making final_pairs list ...")
        lines = open(path_to_tab_sep_dialogue).read().strip().split('\n')
        
        final_pairs = []
        i = 0
        for l in lines:
            
            pair = [self.normalize_string(sentence) for sentence in l.split('\t')]
            
            if self.filterPair(pair,max_sent_len, min_sent_len):
                
                filtered_pair = []
                
                for sentence in pair:

                    self.index_sentence(sentence)
                    filtered_pair.append(sentence)
                  
                final_pairs.append(filtered_pair)
        print("number of pairs", len(final_pairs))
        return final_pairs
    
    def tokens2glove(self, min_word_count,glove, mbed_dim = 50):
    
        print("trimming...")    
        self.trim(min_word_count)
        
        if glove is None:
            glove = vocab.GloVe(name='6B', dim=embed_dim)
            print('Loaded {} words'.format(len(glove.itos)))
        else:
            embed_dim = glove.vectors.size(1)
                    
        print("building embedding from glove...")
        embedding = np.zeros((len(self.index2word), embed_dim)).astype(np.float32)
        for i in range(self.num_nonwordtokens):
            embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)
        for i in range(self.num_nonwordtokens,len(self.index2word)):
            if self.index2word[i] in glove.stoi:
                embedding[i,:] = glove.vectors[glove.stoi[self.index2word[i]]]
            else:
                embedding[i,:] = np.random.uniform(-1,1,embed_dim).astype(np.float32)
        
        return self.index2word, self.word2index, embedding, self.n_words #torch.from_numpy(embeddings).float() 
    


MAX_SENT_LENGTH = 20
lang = Lang("chat")
final_pairs = lang.make_pairs("../data/input-output.txt", max_sent_len = MAX_SENT_LENGTH, 
                              min_sent_len = 3)
index2word, word2index, embedding, vocab_size = lang.tokens2glove(5,glove)


final_pairs


new_pairs = []
for pair in final_pairs:
    if len(pair[1].split(" ")) > 3:
        pair[1] = pair[1].replace(".","")
        new_pairs.append([pair[0],pair[1]])
        
new_pairs


mx_sent_30_min_sent_1_min_wrd_10_dim50 = (index2word, word2index, embedding, final_pairs, vocab_size)


pickle.dump(mx_sent_30_min_sent_1_min_wrd_10_dim50, open( "saved_pickle/mx_sent_30_min_sent_1_min_wrd_10_dim50.p", "wb" ) )


index2word, word2index, embedding, final_pairs, vocab_size  = pickle.load( open( "saved_pickle/mx_sent_30_min_sent_1_min_wrd_10_dim50.p", "rb" ) )


# ## Makesure the GloVe vectors work the way they are supposed to
# 

np.dot(embedding[word2index["admire"]],embedding[word2index["respect"]])


np.dot(embedding[word2index["admire"]],embedding[word2index["sad"]])


np.dot(embedding[word2index["sad"]],embedding[word2index["depressed"]])


##########  Converts [" input string ", " output string "] (pair) , appends <EOS> index and returns indices ######

def indexesFromSentence(lang, sentence):
    '''
    account for strings not in the vocabulary by using the unknown token
    '''
    sentence_as_indices = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            sentence_as_indices.append(lang.word2index[word])
        else:
            sentence_as_indices.append(lang.UNK_token)
    
    return sentence_as_indices


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair,lang):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)

######## the pair indices are returned as 2 LongTensor Variables in torch #############


######## Tells you how long youve been training and how much longer you have left ####

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################3

############### plot_losses #######################################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
####################################################################


class EncoderRNN(nn.Module):
    
    def __init__(self, hidden_size, embedding,
                 num_layers = 3, bidirectional = False, train_embedding = True):
        
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        embedding = torch.from_numpy(embedding).float()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding, requires_grad=train_embedding)
        self.gru = nn.GRU(embedding.shape[1], hidden_size, num_layers, bidirectional=bidirectional)
        
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        # make the initial hidden state learnable as well 
        hidden0 = torch.zeros(self.num_layers*num_directions, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs
            
        return output, hidden

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0


class AttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, embedding, dropout_p=0.1,  
                 num_layers = 3, train_embedding = True, max_length=30):
        
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        embedding = torch.from_numpy(embedding).float()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding, requires_grad=train_embedding)
        
        self.attn = nn.Linear(self.hidden_size + embedding.shape[1], self.max_length)
        
        self.attn_combine = nn.Linear(self.hidden_size + embedding.shape[1], self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = num_layers)
        self.out = nn.Linear(self.hidden_size, embedding.shape[0])
        
        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        attn = self.attn(cat)
        
        attn_weights = F.softmax(attn, dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        
        if use_cuda:
            return self.hidden0.cuda()
        else:
            return self.hidden0






def train(lang, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=30, teacher_forcing_ratio = 0.5, bidirectional = False):
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]

    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[lang.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:] 
        
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == lang.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, pairs, lang, n_iters =1000, print_every=100, plot_every=100,
               learning_rate=0.01, teacher_forcing_ratio = 0.5, bidirectional = False,
               name = "noname", lowest_loss =100, max_length = 30, gamma = 0.95, reverse=False):
    
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    escheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=print_every, gamma=gamma) 
    dscheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=print_every, gamma=gamma) 
    
    training_pairs = [variablesFromPair(random.choice(pairs),lang)
                      for i in range(n_iters)]
    
    criterion = nn.NLLLoss()

    lowest_loss = lowest_loss
    
    for iter in range(1, n_iters + 1):
        
        escheduler.step()
        dscheduler.step()
        
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        
        if reverse:
            input_variable = training_pair[1]
            target_variable = training_pair[0]

        loss = train(lang, input_variable, target_variable, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion, 
                     teacher_forcing_ratio = teacher_forcing_ratio,
                     max_length = max_length, bidirectional =  bidirectional)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
            if lowest_loss > print_loss_avg:
                lowest_loss = print_loss_avg
                print("new lowest loss, saving...")
                torch.save(encoder.state_dict(), "saved_params/encoder"+name+".pth")
                torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+".pth")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


hidden_size = 1024
num_layers = 2
bidirectional = True 

encoder = EncoderRNN(hidden_size, embedding, num_layers = num_layers, bidirectional = bidirectional,
                     train_embedding = True)


attn_decoder = AttnDecoderRNN(hidden_size,  embedding, dropout_p=0.1, num_layers = num_layers,
                               train_embedding = True, max_length = MAX_SENT_LENGTH)

if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()
    

version = 0
name = "hidden" + str(hidden_size) + "L" + str(num_layers) + "bidir"+str(bidirectional) + "v"+ str(version)


#encoder.load_state_dict(torch.load("saved_params/encoder"+name+".pth"))
#attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+name+".pth"))
#name = "_256h_4L_bi_glove"
encoder.load_state_dict(torch.load("saved_params/encoder"+name+".pth"))
attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+name+".pth"))

version = 3
name = "hidden" + str(hidden_size) + "L" + str(num_layers) + "bidir"+str(bidirectional) + "v"+ str(version)


trainIters(encoder, attn_decoder, final_pairs, lang, n_iters = 100000, print_every=10000, 
           learning_rate=0.01, teacher_forcing_ratio = 0.99,
           bidirectional = bidirectional, name = name,
           max_length = MAX_SENT_LENGTH, lowest_loss = 4.0, gamma = 0.5)


#name = "_256h_4L_bi_glove"
torch.save(encoder.state_dict(), "saved_params/encoder"+name+".pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+".pth")


# save cpu version

encoder.cpu()
attn_decoder.cpu()

torch.save(encoder.state_dict(), "saved_params/encoder"+name+"cpu.pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+"cpu.pth")


evaluateRandomly(final_pairs, encoder, attn_decoder,  n=10, bidirectional = bidirectional)

evaluateAndShowAttention("hello",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("what link ?",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("why is this happening ?",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("tell me something stupid",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("you can make it if you try",encoder, attn_decoder, bidirectional = bidirectional)

evaluateAndShowAttention("where are we and where are we going ?",
                         encoder, attn_decoder, bidirectional = bidirectional  )


def evaluate(encoder, decoder, sentence, max_length=MAX_SENT_LENGTH,
             bidirectional =bidirectional):
    
    
    input_variable = variableFromSentence(lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[lang.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:]

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == lang.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(pairs, encoder, decoder, n=10, bidirectional = False):
    for i in range(n):
        pair = random.choice(pairs)
        print('input from data >', pair[0])
        print('output from data=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], bidirectional = bidirectional)
        output_sentence = ' '.join(output_words)
        print('bot response <', output_sentence)
        print('')
        
def showAttention(input_sentence, output_words, attentions, bidirectional = False):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence,encoder, attn_decoder, bidirectional = False):
    
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence, bidirectional = bidirectional)
    
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


# # Reinforcement Learning
# 

# ## Backward Seq2Seq
# 

backward_encoder = EncoderRNN(hidden_size, embedding, num_layers = num_layers,
                              bidirectional = bidirectional, train_embedding=True)

backward_attn_decoder = AttnDecoderRNN(hidden_size, embedding, dropout_p=0.1, 
                                       num_layers = num_layers, train_embedding=True, max_length = MAX_SENT_LENGTH)

if use_cuda:
    backward_encoder = backward_encoder.cuda()
    backward_attn_decoder = backward_attn_decoder.cuda()

backward_name = '_backward_' + name


backward_encoder.load_state_dict(torch.load("saved_params/encoder"+backward_name+".pth"))
backward_attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+backward_name+".pth"))


trainIters(backward_encoder, backward_attn_decoder, final_pairs, lang, n_iters = 100000, print_every=1000, 
           learning_rate=0.01, teacher_forcing_ratio = 1.0,
           bidirectional = bidirectional, name = backward_name,
           max_length = MAX_SENT_LENGTH, lowest_loss = 6.0, gamma = 0.95, reverse=True)


torch.save(backward_encoder.state_dict(), "saved_params/encoder"+backward_name+".pth")
torch.save(backward_attn_decoder.state_dict(), "saved_params/attn_decoder"+backward_name+".pth")

# save cpu version

encoder.cpu()
attn_decoder.cpu()

torch.save(backward_encoder.state_dict(), "saved_params/encoder"+backward_name+"cpu.pth")
torch.save(backward_attn_decoder.state_dict(), "saved_params/attn_decoder"+backward_name+"cpu.pth")


# ## The main training loops is trainRLIters(), it calls RLStep to get the forward loss, it calls calculate_rewards to get the rewards for each forward_loss element, the two are multiplied to ge the final loss as per REINFORCE. This final Loss is backpropagated 
# 

# Forward
forward_encoder = EncoderRNN(hidden_size, embedding, num_layers = num_layers, 
                             bidirectional = bidirectional, train_embedding = True)


forward_attn_decoder = AttnDecoderRNN(hidden_size,  embedding, dropout_p=0.1, 
                                      num_layers = num_layers, train_embedding = True, max_length = MAX_SENT_LENGTH)

# Backward
backward_encoder = EncoderRNN(hidden_size, embedding, num_layers = num_layers,
                              bidirectional = bidirectional, train_embedding=False)

backward_attn_decoder = AttnDecoderRNN(hidden_size, embedding, dropout_p=0.1, 
                                       num_layers = num_layers, train_embedding=False, max_length = MAX_SENT_LENGTH)

if use_cuda:
    forward_encoder = forward_encoder.cuda()
    forward_attn_decoder = forward_attn_decoder.cuda()
    
    backward_encoder = backward_encoder.cuda()
    backward_attn_decoder = backward_attn_decoder.cuda()

rl_name = '_RL_' + name


forward_encoder.load_state_dict(torch.load("saved_params/encoder"+name+".pth"))
forward_attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+name+".pth"))
backward_encoder.load_state_dict(torch.load("saved_params/encoder"+backward_name+".pth"))
backward_attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder"+backward_name+".pth"))


def RLStep(lang, input_variable, target_variable, encoder, decoder, criterion, 
           max_length=30, teacher_forcing_ratio=0.5, bidirectional=False):
    
    encoder_hidden = encoder.initHidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    response = []
    
    for ei in range(input_length):
        
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[lang.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, : ,:] 
        
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            
            # TODO: ni or decoder_output?
            response.append(ni)
            if ni == lang.EOS_token:
                break
    # Cross entropy loss, target length, response
    return (loss, target_length, response)


def calculate_rewards(lang, input_variable, target_variable, curr_response,
                      forward_encoder, forward_decoder, 
                      backward_encoder, backward_decoder, 
                      criterion, max_length, dull_responses, 
                      teacher_forcing_ratio, bidirectional):
    ep_rewards = []
    
    # ep_num are used to bound the number of episodes
    # MAXIMUM ep = 10
    ep_num = 1
    
    responses = []
    
    ep_input = input_variable
    ep_target = target_variable
    
    ## At first, in trainRLIters, we already generated the 'first' curr_response. The loop below is to
    ## simulate one episode where two agents response back-to-back.
    while (ep_num <= 10):
        
        ## TODO: min length of response to break?
        
        ## Break once we see (1) dull response, (2) the response is less than 1, (3) repetition
        if (len(curr_response) < 1):# or (curr_response in responses) or (curr_response in dull_responses):
            break
            
        curr_response = Variable(torch.LongTensor(curr_response), requires_grad=False).view(-1, 1)
        curr_response = curr_response.cuda() if use_cuda else curr_response
        responses.append(curr_response)
        
        
        ## Ease of answering
        # Use the forward model to generate the log prob of generating dull response given ep_input.
        # Use the teacher_forcing_ratio = 1!
        r1 = 0
        for d in dull_responses:
            forward_loss, forward_len, _ = RLStep(lang, ep_input, d, 
                                                  forward_encoder, forward_decoder, 
                                                  criterion, max_length, 
                                                  teacher_forcing_ratio = 1.1,
                                                  bidirectional =  bidirectional)
            if forward_len > 0:
                # log (1/P(a|s)) = CE  --> log(P(a | s)) = - CE
                r1 -= forward_loss.data[0] / forward_len
        if len(dull_responses) > 0:
            r1 = r1 / len(dull_responses)
    
    
        ## Information flow
        # responses contains all the generated response by the forward model
        r2 = 0
        if (len(responses) > 2):
            # vec_a --> h_(i)  = responses[-3]
            # vec_b --> h_(i+1)= responses[-1]
            vec_a = responses[-3].data
            vec_b = responses[-1].data
            # length of the two vector might not match
            min_length = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:min_length]
            vec_b = vec_b[:min_length]
            cos_sim = 1 - scipy.spatial.distance.cosine(vec_a, vec_b)
            # -1 <= cos_sim <= 1
            # TODO: how to handle negative cos_sim?
            if cos_sim <= 0:
                r2 = - cos_sim
            else:
                r2 = - np.log(cos_sim)
            
        
        
        ## Semantic Coherence
        # Use the forward model to generate the log prob of generating curr_response given ep_input
        # Use the backward model to generate the log prob of generating ep_input given curr_response
        r3 = 0
        forward_loss, forward_len, _ = RLStep(lang, ep_input, curr_response, 
                                              forward_encoder, forward_decoder, 
                                              criterion, max_length, 
                                              teacher_forcing_ratio = teacher_forcing_ratio,
                                              bidirectional =  bidirectional)
        backward_loss, backward_len, _ = RLStep(lang, curr_response, ep_input, 
                                                backward_encoder, backward_decoder, 
                                                criterion, max_length, 
                                                teacher_forcing_ratio = teacher_forcing_ratio,
                                                bidirectional =  bidirectional)
        
        if forward_len > 0:
            r3 += forward_loss.data[0] / forward_len
        if backward_len > 0:
            r3 += backward_loss.data[0] / backward_len
            
            
        ## TODO: r1 is really negative, which is good. But I think it messes up the training. What to do?   
        ## Add up all the three rewards
        rewards = 0.25 * r1 + 0.25 * r2 + 0.5 * r3
        ep_rewards.append(rewards)
        
        ## Set the next input
        ep_input = curr_response
        ## TODO: what's the limit of the length? and what should we put as the dummy target?
        ep_target = Variable(torch.LongTensor([0] * max_length), requires_grad=False).view(-1, 1)
        ep_target = ep_target.cuda() if use_cuda else ep_target
        
        # Turn off the teacher forcing ration after first iteration (since we don't have a target anymore).
        teacher_forcing_ratio = 0
        ep_num += 1
        
        # First start with Forward model to generate the current response, given ep_input
        # ep_target is empty if ep_num > 1
        _, _, curr_response = RLStep(lang, ep_input, ep_target, 
                                     forward_encoder, forward_decoder, 
                                     criterion, max_length, 
                                     teacher_forcing_ratio = teacher_forcing_ratio,
                                     bidirectional =  bidirectional)

    # Take the mean of the episodic rewards
    r = 0
    if len(ep_rewards) > 0:
        r = np.mean(ep_rewards)
    else:
        ## TODO: what to do when overall reward = 0?
        ## This means the current response is too short
        r = 1
        
    ## TODO: negative reward means our model is doing well (?) but it messes up the training.
    ## For now, I'll cap the reward to a small positive value.
    r = max(0.001, r)
    return r


def trainRLIters(forward_encoder, forward_decoder, backward_encoder, backward_decoder,
                 pairs, lang, dull_responses, 
                 n_iters =1000, print_every=100, plot_every=100, 
                 learning_rate=0.01, teacher_forcing_ratio = 0.5, bidirectional = False, 
                 name = "noname", max_length = 30, lowest_loss =100, gamma = 0.95):
    
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    # Optimizer
    forward_encoder_optimizer = optim.Adam(forward_encoder.parameters(), lr=learning_rate)
    forward_decoder_optimizer = optim.Adam(forward_decoder.parameters(), lr=learning_rate)
    
    # Scheduler
    forward_escheduler = optim.lr_scheduler.StepLR(forward_encoder_optimizer, step_size=print_every, gamma=gamma) 
    forward_dscheduler = optim.lr_scheduler.StepLR(forward_decoder_optimizer, step_size=print_every, gamma=gamma) 
    
    training_pairs = [variablesFromPair(random.choice(pairs),lang)
                      for i in range(n_iters)]
    
    criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
    
    lowest_loss = lowest_loss
    
    for iter in range(1, n_iters + 1):
        
        forward_escheduler.step()
        forward_dscheduler.step()
        
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        
        ## Manually zero out the optimizer
        forward_encoder_optimizer.zero_grad()
        forward_decoder_optimizer.zero_grad()
        
        ## Do one forward step.
        forward_loss, forward_len, forward_response = RLStep(lang, input_variable, target_variable, 
                                                             forward_encoder, forward_decoder, 
                                                             criterion, max_length,
                                                             teacher_forcing_ratio = teacher_forcing_ratio,
                                                             bidirectional =  bidirectional)
        
        ## Calculate the reward
        reward = calculate_rewards(lang, input_variable, target_variable, forward_response,
                                   forward_encoder, forward_decoder, 
                                   backward_encoder, backward_decoder, 
                                   criterion, max_length, dull_responses, 
                                   teacher_forcing_ratio, bidirectional)
        
        ## Manually zero out the optimizer (Again, before we minimize)
        forward_encoder_optimizer.zero_grad()
        forward_decoder_optimizer.zero_grad()
        
        ## Update the forward seq2seq with its loss scaled by the reward
        loss = forward_loss * reward
        
        loss.backward()
        forward_encoder_optimizer.step()
        forward_decoder_optimizer.step()
        
        print_loss_total += (loss.data[0] / forward_len)
        plot_loss_total += (loss.data[0] / forward_len)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
            if lowest_loss > print_loss_avg:
                lowest_loss = print_loss_avg
                print("new lowest loss, saving...")
                torch.save(encoder.state_dict(), "saved_params/encoder"+name+".pth")
                torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+name+".pth")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


# ### Initialize dull_responses as a list of Variables
# 

# TODO
dull_set = ["I don't know what you're talking about.", "I don't know.", 
 "You don't know.", "You know what I mean.", "I know what you mean.", 
 "You know what I'm saying.", "You don't know anything."]

dull_responses = [variableFromSentence(lang, d) for d in dull_set]


trainRLIters(forward_encoder, forward_attn_decoder, backward_encoder, backward_attn_decoder, 
             final_pairs, lang, dull_responses,
             n_iters=100000, print_every=1000, learning_rate=0.01, 
             teacher_forcing_ratio = 0.75, bidirectional = bidirectional, name = rl_name, 
             max_length = MAX_SENT_LENGTH, lowest_loss=6.0, gamma = 0.95)


torch.save(encoder.state_dict(), "saved_params/encoder"+rl_name+"cpu.pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder"+rl_name+"cpu.pth")


evaluateRandomly(final_pairs, forward_encoder, forward_attn_decoder,  n=10, bidirectional = bidirectional)


