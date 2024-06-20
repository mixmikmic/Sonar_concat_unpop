import requests
import json
# from bs4 import BeautifulSoup
import time
import pickle


token = 'E9LQF4KUV7SE2MH0'


# # Finding Total Accounts
# 

total_accounts = 9022265 # last update
# finding total number of hon international accounts using binary search
lo, hi = total_accounts, 9100000
while lo+1 < hi:
    mid = (lo+hi)//2
    r = requests.get('http://api.heroesofnewerth.com/player_statistics/ranked/accountid/{}/?token={}'.format(mid, token))
    if r.text == 'Unauthorized!':
        print(r.text)
        break
    try:
        player_dict = r.json()
        lo = mid
    except ValueError:
        hi = mid-1
    print(lo, hi)
print('total accounts:', lo)
total_accounts = lo


# # Finding match statistics
# 

db_file = 'hon_matches.pkl'
try:
    with open(db_file, 'rb') as f:
        all_matches = pickle.load(f)
except FileNotFoundError:
    all_matches = dict()
print(len(all_matches))


def clean(accounts):
    new_accounts = []
    removed = 0
    for acc in accounts:
        secs = int(acc['secs'])
        if int(acc['actions']) <= 300: # if APM is too low, the player did not contribute to the team, (disconnection assumed)
            removed += 1
        else:
            new_acc = dict()
            new_acc['hero_id'] = int(acc['hero_id'])
            new_acc['win'] = int(acc['wins'])
            new_acc['concede'] = int(acc['concedes'])
            new_acc['match_id'] = int(acc['match_id'])
            new_acc['secs'] = secs
            new_acc['team'] = int(acc['team']) - 1 # legion 0, hellbourne 1
            new_accounts.append(new_acc)
    if removed:
        print('Removed Low APM:', removed)
    return new_accounts


# return a dict that can be accessed by [match_id]
def aggregate(accounts):
    matches = dict()
    for acc in accounts:
        matchid = acc['match_id'] 
        if matchid not in matches:
            matches[matchid] = {
                'concedes': 0,
                'secs': 0,
                'legion': list(),
                'hellbourne': list()
            }
        # use max because secs is the amount of seconds that player is in the game
        matches[matchid]['secs'] = max(matches[matchid]['secs'], acc['secs'])
        if acc['secs'] == matches[matchid]['secs']: # this guy plays until the end of the game
            matches[matchid]['winner'] = acc['team'] if acc['win'] else 1 - acc['team']
        if acc['concede']:
            matches[matchid]['concedes'] += 1
        if acc['team'] == 0:
            matches[matchid]['legion'].append(acc['hero_id'])
        else:
            matches[matchid]['hellbourne'].append(acc['hero_id'])
    return matches


matchid = 147862153 # latest match id found


import sys
dump_every = 10 # iterations
sleep_period = 0.5 # sleep to prevent API requests limit
for iter in range(2501):
    matchids = []
    for i in range(25):
        matchids.append(str(matchid))
        matchid -= 1
    url = 'http://api.heroesofnewerth.com/multi_match/statistics/matchids/{}/?token={}'.format('+'.join(matchids), token)
    print('-- Requesting...', end=' ')
    while True:
        try:
            r = requests.get(url)
            break
        except:
            print('ConnectionError, pause for {} secs'.format(sleep_period * 20), file=sys.stderr)
            time.sleep(sleep_period * 20)
    print('Sleeping...', end=' ')
    time.sleep(sleep_period)
    if r.text == 'No results.':
        print(r.text)
        continue
    try:
        print('Decoding...', end=' ')
        match_accounts = r.json()
    except ValueError:
        print('Error:', r.text, file=sys.stderr)
        time.sleep(sleep_period * 10)
        continue
    match_accounts = clean(match_accounts)
    matches = aggregate(match_accounts)
    all_matches.update(matches)
    print('Total matches in this/all batch:', len(matches), len(all_matches))
    # print('Next Unused MatchID:', matchid)
    if iter % dump_every == 0:
        print('==> Dumping at iter', iter)
        with open(db_file, 'wb') as f:
            pickle.dump(all_matches, f)


print('Next Unused MatchID:', matchid)


match_accounts


matches


with open(db_file, 'wb') as f:
    pickle.dump(all_matches, f)


# # Heroes reference
# http://api.heroesofnewerth.com/heroes/all?token=E9LQF4KUV7SE2MH0
# 

r = requests.get('http://api.heroesofnewerth.com/heroes/all?token=E9LQF4KUV7SE2MH0')


heroes_dict = r.json()
heroes_dict


len(heroes_dict)


new_heroes_dict = dict()
for value in heroes_dict.values():
    hero_id, hero = list(value.items())[0]
    hero_id = int(hero_id)
    new_heroes_dict[hero_id] = hero
new_heroes_dict


with open('heroes_name.pkl', 'wb') as f:
    pickle.dump(new_heroes_dict, f)


import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
get_ipython().magic('matplotlib notebook')
np.set_printoptions(suppress=True, precision=3)


with open('hon_matches.pkl', 'rb') as db:
    matches = pickle.load(db)
print('Total Matches:', len(matches))


# try removing matches that do not have total 10 players
# or have duplicate heroes
remove_incomplete = False
removes = []
dup_removes = []
for k, v in matches.items():
    ll = len(v['legion'])
    hl = len(v['hellbourne'])
    if remove_incomplete and ll + hl < 10:
        removes.append(k)
    # checking for duplicate heroes match
    lls = len(set(v['legion']))
    hls = len(set(v['hellbourne']))
    if lls < ll or hls < hl:
        dup_removes.append(k)
removes = set(removes)
for key in removes:
    del matches[key]
print('Incomplete Matches Removed:', len(removes))
dup_removes = set(dup_removes) & set(matches.keys())
for key in dup_removes:
    del matches[key]
print('Duplicate-Heroes Matches Removed:', len(dup_removes))
print('Remaining Matches:', len(matches))


X = []
for i, item in enumerate(matches.items()):
    if i == 5:
        break
    print(item)
    X.append(item[1])


AVAIL_HEROES = 260 # actually 134 but extra for future
def vectorize_matches(matches, include_Y=True):
    legion_vec = np.zeros([len(matches), AVAIL_HEROES])
    hellbourne_vec = np.zeros([len(matches), AVAIL_HEROES])
    if include_Y:
        winner = np.zeros([len(matches), 1])
        concede = np.zeros([len(matches), 1])
        secs = np.zeros([len(matches), 1])
    for m, match in enumerate(matches):
        for hero_id in match['legion']:
            legion_vec[m, hero_id] = 1.
        for hero_id in match['hellbourne']:
            hellbourne_vec[m, hero_id] = 1.
        if include_Y:
            if match['winner']:
                winner[m, 0] = 1.
            if match['concedes']:
                concede[m, 0] = 1.
            secs[m, 0] = match['secs']
    x = np.concatenate([legion_vec, hellbourne_vec], axis=1)
    if include_Y:
        y = np.concatenate([winner, concede, secs], axis=1)
    return (x, y) if include_Y else x


# # Serious time
# 

X, Y = vectorize_matches(matches.values())
X.shape, Y.shape


with open('heroes_name.pkl', 'rb') as f:
    heroes_dict = pickle.load(f)
heroes_dict[125]


def hero_id_to_name(hero_id):
    return heroes_dict[hero_id]['disp_name']
hero_id_to_name(125)


def hero_name_to_id(name):
    if not name:
        return None
    name = name.lower()
    for id, hero in heroes_dict.items():
        if name in hero['disp_name'].lower():
            return id, hero['disp_name']
    return None
hero_name_to_id('BUB')


from operator import itemgetter
# returns a hero that maximize win probability in a given team
# if 'optimal' is false, it will return a hero that minimize win probability
def optimal_hero_choice(model, match, hellbourne_side=False, as_list=True, as_name=True, optimal=True):
    legion = match['legion']
    hellbourne = match['hellbourne']
    team_ids = hellbourne if hellbourne_side else legion
    hypothesis = []
    for id in set(heroes_dict.keys()) - set(legion + hellbourne): # all choosable hero ids
        team_ids.append(id)
        x = vectorize_matches([match], include_Y=False)
        team_ids.pop()
        p = model.predict(x, verbose=0)[0]
        hero = id
        if as_name:
            hero = hero_id_to_name(hero)
        hypothesis.append((hero, p[0, 1 if hellbourne_side else 0]))
    extrema = max if optimal else min
    return sorted(hypothesis, key=itemgetter(1), reverse=optimal) if as_list else extrema(hypothesis, key=itemgetter(1))


def humanize(xrow):
    legion, hellbourne = [], []
    for i, el in enumerate(xrow):
        if el:
            if i < AVAIL_HEROES:
                name = hero_id_to_name(i)
                legion.append(name)
            else:
                name = hero_id_to_name(i - AVAIL_HEROES)
                hellbourne.append(name)
    return {'legion': legion, 'hellbourne': hellbourne}


# # Data Exploration
# 

# which team won more? Hellbourne!
Counter(Y[:,0]), Y.mean(axis=0)


played = [] # played heroes
for i, item in enumerate(matches.items()):
    v = item[1]
    if 0 in v['legion']:
        print(item)
    if 0 in v['hellbourne']:
        print(item)
    played.extend(v['legion'])
    played.extend(v['hellbourne'])
len(played)


Counter(played).most_common()


[(hero_id_to_name(id), freq) for (id, freq) in Counter(played).most_common() if id != 0]


# how many players are there in a game?
players = Counter(X.sum(axis=1))
players


plt.plot(list(players.keys()), list(players.values()))


# # Machine Learning
# 

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.91)
X_train.shape, Y_test.shape


from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, ELU, Reshape, Convolution2D, Flatten, Permute, BatchNormalization, Input
from keras.regularizers import l2, activity_l2
hero_input = Input(shape=[2 * AVAIL_HEROES], name='hero_input')
h = Reshape([1, 2, AVAIL_HEROES], input_shape=[AVAIL_HEROES*2,])(hero_input)
h = Permute([2, 3, 1])(h)
h = Convolution2D(135, 1, AVAIL_HEROES, border_mode='valid')(h) # learn to represent 135 heroes from 260-d vector
# h = ELU()(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Convolution2D(64, 1, 1, border_mode='valid')(h)
# h = ELU()(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Flatten()(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dropout(0.5)(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dropout(0.5)(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
logit = Dropout(0.5)(h)
out_winner = Dense(output_dim=2, activation='softmax', name='out_winner')(logit) # 0 for legion and 1 for hellbourne team
out_concede = Dense(output_dim=1, activation='sigmoid', name='out_concede')(logit) # would the loser concede ?
out_secs = Dense(output_dim=1, name='out_secs')(logit) # how many seconds would the game last ?
model = Model([hero_input], [out_winner, out_concede, out_secs])
loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error']
metrics = {
    'out_winner': 'accuracy',
    'out_concede': 'accuracy',
    'out_secs': 'mean_absolute_error'
}
loss_weights = {
    'out_winner': 2.,
    'out_concede': 1.,
    'out_secs': 1./600000 # mean squared loss is so high we have to penalize it, otherwise, it would steal all computations
}
model.compile(loss=loss, optimizer='adam', metrics=metrics, loss_weights=loss_weights)
hist = model.fit(X_train, np.split(Y_train, 3, axis=1), batch_size=32, nb_epoch=10, verbose=1, validation_split=0.075)


model.summary()
model.save('honnet_brain.h5')


loss_and_metrics = model.evaluate(X_test, np.split(Y_test, 3, axis=1), batch_size=32, verbose=1)
loss_and_metrics


from collections import Counter
Counter(Y[:, 0]), Counter(Y[:, 1]) # win team and concede counts


plt.hist(Y[:, -1], bins=50) # lasting time in secs


pred = model.predict(X_test[:10, :])
# prediction is [legion_win_prob, hellbourne_win_prob, loser_concede_prob, estimated_game_time_in_secs]
np.concatenate(pred, 1), Y_train[:10, :]


humanize(X_test[0])


# # Inference
# 

def inputName():
    name = input('Hero Name: ')
    hero_id, hero_name = hero_name_to_id(name)
    return hero_id, hero_name


legion = []
hellbourne = []
hellbourne_bool = 0
match = [{'legion': legion, 'hellbourne': hellbourne}]


hellbourne_bool = not hellbourne_bool
'Hellbourne' if hellbourne_bool else 'Legion'


hero_id, hero_name = inputName()
if hero_id is not None:
    print('Hero:', hero_name)
    if hellbourne_bool:
        hellbourne.append(hero_id)
    else:
        legion.append(hero_id)
    x = vectorize_matches(match, include_Y=False)
    print('Team:', humanize(x[0]))
    proba = model.predict(x, verbose=0)
    print('Proba:', np.concatenate(proba, axis=1))


# selecting an optimal hero for the current team
choice = optimal_hero_choice(model, match[0], hellbourne_side=hellbourne_bool, as_list=False, as_name=False, optimal=True)
print(choice, hero_id_to_name(choice[0]))
team_ids = hellbourne if hellbourne_bool else legion
team_ids.append(choice[0])
match = [{'legion': legion, 'hellbourne': hellbourne}]
x = vectorize_matches(match, include_Y=False)
print('Team:', humanize(x[0]))
proba = model.predict(x, verbose=0)
print('Proba:', np.concatenate(proba, axis=1))


