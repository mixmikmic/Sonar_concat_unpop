# ## Data preprocessing
# 

import pandas as pd
import numpy as np
import time
import datetime
import scipy.signal as sp


columns = ['sample_index', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'A1', 'A2', 'A3', 'timestamp']


sean = pd.read_table('recorded_data/Seantry4_GoodandBad/OpenBCI-RAW-Seantry4_GoodandBad.txt', delimiter=',', names=columns)


sean_time = np.loadtxt('recorded_data/Seantry4_GoodandBad/sean_good_bad_23.57_time')


sean = sean.dropna()


sec = []
yr = "2018-04-07"
for i in sean.timestamp:
    str_fixed = yr + i
    sec.append(datetime.datetime.timestamp(datetime.datetime.strptime(str_fixed, "%Y-%m-%d %H:%M:%S.%f")))


sec = pd.Series(sec, name = 'sec')


data = pd.concat([sean, sec], axis = 1)


start = sec[0]

fixed_time = [start]
for i in range(754100):
    fixed_time.append(fixed_time[-1]+(1/250))
    
fixed_time = pd.Series(fixed_time, name = 'fixed_time')
fixed_time = fixed_time[:data.sec.shape[0]]
fixed_time.shape


data = pd.concat([sean, fixed_time], axis = 1)


exp_beg = data[abs(fixed_time - sean_time[0]) <= 2*1e-3].index[0]


nonfiltered_data = data[exp_beg:].drop(['timestamp', 'A1', 'A2', 'A3'], axis = 1).reset_index(drop = True)


no_white = nonfiltered_data[2560:].reset_index(drop = True)


# load pictures:
pict = np.loadtxt('recorded_data/Seantry4_GoodandBad/sean_good_bad_23.57_pictures_only', dtype='<U9')


pict.shape


ans = []
for i in pict:
    if i[0] == 'P':
        ans.append(1) #1 = positive
    else:
        ans.append(0) #0 -negative
    ans.append(3) #3 - black
ans = np.array(ans)


ans.shape[0]


no_white.shape


#no_white.head()


lbl_data = []

for i in range(ans.shape[0]):
    if ans[i] != 3:
        lbl_data.append(no_white[i:i+8*250])
    if ans[i] == 3:
        lbl_data.append(no_white[i:i+4*250])


#lbl_data[0].head()


dataset = []

for i in lbl_data:
    feature = []
    for j in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        ch = i[j]
        f, Pxx_spec = sp.welch(ch, 250, nperseg = 250)
        alpha_ch = np.mean(Pxx_spec[(f>=8) & (f<=13)])
        beta_ch = np.mean(Pxx_spec[(f>13) & (f<=30)])
        theta_ch = np.mean(Pxx_spec[(f>=4) & (f<=7)])
        gamma_ch = np.mean(Pxx_spec[(f>=30) & (f<=50)])
        feature.append(alpha_ch)
        feature.append(beta_ch)
        feature.append(theta_ch)
        feature.append(gamma_ch)    
    dataset.append(feature)


cols = []
for j in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
    col = [i+'_' +j for i in ['alpha', 'beta', 'theta', 'gamma']]
    cols +=col


processed_data = pd.DataFrame(dataset, columns=cols)


# ## Training the model
# 

from sklearn.svm import SVC


#clf = SVC()#class_weight = 'balanced')
#clf.fit(processed_data[:400], ans[:400]) 


from sklearn.metrics import accuracy_score
#ans_pred= clf.predict(processed_data[400:])
#accuracy_score(ans[400:], ans_pred)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, max_depth=100000, random_state = 0,class_weight = 'balanced')
rf.fit(processed_data[:400], ans[:400]) 


ans_pred= rf.predict(processed_data[:400])

print('Accuracy on 18% of data:', accuracy_score(ans[:400], ans_pred))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, processed_data, ans, cv=4)
print('Cross-validation on 6 folds score:',np.mean(scores))


#DEMO OPENBCI

#import time
#def follow(thefile):
    #thefile.seek(0,2)
#    while True:
#        line = thefile.readline()
#        if not line:
#            time.sleep(0.1)
#            continue
#        yield line

#if __name__ == '__main__':
#logfile = open("sean_EEG","r")
#loglines = follow(logfile)
#current_data = []
#for line in loglines:
#    if (line[0:1] != '\n') and (line[0:1] != '--') and (line[0:1] != 'ID') and (line.count('.') != 3):
#        my_float_list = [float(x) for x in line[:-1].split(',')]
#        current_data.append(my_float_list)
#    if len(current_data) == 1:
#        print(line)
    #pass
    #print(line)


#current_data


no_white.head()


rf.predict(processed_data[1:2])[0]


def feat_extract(i):
    feature = []
    for j in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        ch = i[j]
        f, Pxx_spec = sp.welch(ch, 250, nperseg = 250)
        alpha_ch = np.mean(Pxx_spec[(f>=8) & (f<=13)])
        beta_ch = np.mean(Pxx_spec[(f>13) & (f<=30)])
        theta_ch = np.mean(Pxx_spec[(f>=4) & (f<=7)])
        gamma_ch = np.mean(Pxx_spec[(f>=30) & (f<=50)])
        feature.append(alpha_ch)
        feature.append(beta_ch)
        feature.append(theta_ch)
        feature.append(gamma_ch)  
    return np.array(feature)


def show_emotion(prediction):
    if prediction[0] == 1: #positive
        img = cv.imread('experiment/emotions/happy.jpg',-1)
    if prediction[0] == 0: #negative
        img = cv.imread('experiment/emotions/sad.png',-1)
    if prediction[0] == 3: #black
        img = cv.imread('experiment/emotions/Neutral.png',-1)
    return img


pict.shape


import cv2 as cv
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

cv.namedWindow("picture", cv.WINDOW_NORMAL)
cv.namedWindow("emotion", cv.WINDOW_NORMAL)
#cv.imshow('window',white_image)
#key = cv.waitKey(10000)
black_image = np.zeros((600, 600, 3), np.uint8)
it = 0
present_data = lbl_data[82*5:]
for name in pict[41*5:]:
    #cv.setWindowProperty("window",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
    img = cv.imread('experiment/pictures/'+name,-1)
    current = present_data[it]
    
    to_predict = [feat_extract(current)]
    prediction = rf.predict(to_predict)
    em = show_emotion(prediction)
    
    cv.imshow('picture', img)
    cv.imshow('emotion', em)
    key = cv.waitKey(3000)
    it += 1
    current = present_data[it]
    
    to_predict = [feat_extract(current)]
    prediction = rf.predict(to_predict)
    em = show_emotion(prediction)
    
    cv.imshow('picture', black_image)
    cv.imshow('emotion', em)
    key = cv.waitKey(1500)
    it += 1
    if key == 27: # exit on ESC
        break
cv.destroyAllWindows()





