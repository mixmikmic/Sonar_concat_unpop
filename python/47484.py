# # Before you run this notebook
# ```
# $ pip instal quilt
# $ quilt install examples/world100m
# ```
# to access the world100m.men dataframe
# 

get_ipython().system(' quilt install examples/world100m --force')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from quilt.data.examples import world100m


world100m.men
df = world100m.men()
df['Date'] = pd.to_datetime(df['Date'])
df


get_ipython().magic('matplotlib inline')
x = df['Time (seconds)']
y = df['Date']
plt.figure(figsize=[12, 6])



plt.subplot(122)
plt.title("Men's 100 meter world records")
plt.plot(y, x, 'crimson')

plt.subplot(121)
axes = plt.gca()
axes.set_ylim([0, 11])
plt.title("Men's 100 meter world records")
plt.plot(y, x, 'crimson')








