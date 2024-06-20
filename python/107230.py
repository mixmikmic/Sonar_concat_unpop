import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


from tichumania_game_scraper import GenCombWeights


# # Looking at Weights
# 

weights = pd.read_csv("./weights_12657951.csv")


weights.dtypes


weights['comb'] = weights['type'].map(str) + "_" + weights['comblength'].map(str) + "_" + weights['height'].map(str)


weights.head()


w1 = weights[weights.length == 1]
w2 = weights[weights.length == 2]
w3 = weights[weights.length == 3]
w4 = weights[weights.length == 4]
w5 = weights[weights.length == 5]
w6 = weights[weights.length == 6]
w7 = weights[weights.length == 7]
w8 = weights[weights.length == 8]
w9 = weights[weights.length == 9]
w10 = weights[weights.length == 10]
w11 = weights[weights.length == 11]
w12 = weights[weights.length == 12]
w13 = weights[weights.length == 13]
w14 = weights[weights.length == 14]


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 10), # length, height
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


def plot_all_types_for_length(length):
    plt.nipy_spectral()  # set colormap
    W = weights[weights.length == length]
    for type_ in ['Single', 'Pair', 'Trio', 'SquareBomb', 'FullHouse', 'PairSteps', 'Straight', 'StraightBomb']:
        W_type = W[W['type'] == type_]
        if len(W_type):
            plt.plot(W_type['height'], W_type['probability'], 'o', label=type_)
    plt.legend(loc='upper left')
    plt.ylabel('probability')
    plt.xlabel('combination height')
    plt.title("{} Handcards".format(length), fontsize=22)
    
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = ['\nDog', 'MahJong', '\nPhoenix', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               'J', 'Q', 'K', 'A', 'Dragon']
    xpos = list(range(len(xlabels)-1))
    xpos.insert(2, 1.5) # insert the phoenix (1.5 at position 2)

    plt.xticks(xpos, xlabels, rotation=0)

def plot_all_lengths_for_type(type_):
    W = weights[weights.type == type_]
    for l in range(1, 15):
        W_len = W[W['length'] == l]
        pp = 'o' if l <= 5 else '.'
        plt.plot(W_len['height'], W_len['probability'], pp, label=l)
    plt.legend(loc='upper left')
    plt.title(str(t), fontsize=22)
    plt.ylabel('probability')
    plt.xlabel('combination height')
    xlabels = ['\nDog', 'MahJong', '\nPhoenix', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               'J', 'Q', 'K', 'A', 'Dragon']
    xpos = list(range(len(xlabels)-1))
    xpos.insert(2, 1.5) # insert the phoenix (1.5 at position 2)
    plt.xticks(xpos, xlabels, rotation=0)


for l in range(1, 15):
    plot_all_types_for_length(l)
    plt.savefig("./figures/type_for_len_{}.png".format(l))
    plt.show()





for t in w14.type.unique():
    plot_all_lengths_for_type(t)
    plt.show()








