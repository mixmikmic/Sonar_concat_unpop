# ### Visualizing the distribution of the observations
# 
# ### Load the required libraries
# 

sc.stop()


import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math
get_ipython().run_line_magic('pylab', 'inline')

import findspark
findspark.init()

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)

import sys
sys.path.append('./lib')

import numpy as np
from numpy_pack import packArray,unpackArray
from spark_PCA import computeCov
from computeStats import computeOverAllDist, STAT_Descriptions

### Read the data frame from pickle file

data_dir='../../Data/Weather'
file_index='BSBSSSSS'

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
print 'keys from STAT=',STAT.keys()

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index

df=sqlContext.read.parquet(filename)
print df.count()
df.show(5)


# ### Select data for a particular station and measurement type
# 

sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT * FROM weather\n\tWHERE measurement='%s' and station='%s'"%('PRCP','USC00081356')#USC00198301
print Query
df1 = sqlContext.sql(Query)
print df1.count(),'rows'
df1.show(2)
rows=df1.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()
T=np.vstack(rows)
T=T/10.  # scaling to make the temperature be in centingrates
shape(T)


# ### Script for plotting yearly plots
# 

from YearPlotter import YearPlotter
fig, ax = plt.subplots(figsize=(10,7));
YP=YearPlotter()
YP.plot(T[:2,:].transpose(),fig,ax,title='PRCP')
#title('A sample of graphs');


# ### Distribution of missing observations
# The distribution of missing observations is not uniform throughout the year. We visualize it below.
# 

def plot_pair(pair,func):
    j=0
    fig,X=subplots(1,2,figsize=(16,6))
    axes=X.reshape(2)
    for m in pair:
        axis = axes[j]
        j+=1
        func(m,fig,axis)
        
def plot_valid(m,fig,axis):
    valid_m=STAT[m]['NE']
    YP.plot(valid_m,fig,axis,title='valid-counts '+m)
    


plot_pair(['TMIN','TMAX'],plot_valid)


plot_pair(['TOBS','PRCP'],plot_valid)


plot_pair(['SNOW', 'SNWD'],plot_valid)


# ### Plots of mean and std of observations
# 

def plot_mean_std(m,fig,axis):
    mean=STAT[m]['Mean']
    std=np.sqrt(STAT[m]['Var'])
    graphs=np.vstack([mean-std,mean,mean+std]).transpose()
    YP.plot(graphs,fig,axis,title='Mean+-std   '+m)


plot_pair(['TMIN','TMAX'],plot_mean_std)


plot_pair(['TOBS','PRCP'],plot_mean_std)


plot_pair(['SNOW', 'SNWD'],plot_mean_std)


# ### plotting top 3 eigenvectors
# 

def plot_eigen(m,fig,axis):
    EV=STAT[m]['eigvec']
    YP.plot(EV[:,:3],fig,axis,title='Top Eigenvectors '+m)


plot_pair(['TMIN','TMAX'],plot_eigen)


plot_pair(['TOBS','PRCP'],plot_eigen)


plot_pair(['SNOW', 'SNWD'],plot_eigen)


# ### Script for plotting percentage of variance explained
# 

def pltVarExplained(j):
    subplot(1,3,j)
    EV=STAT[m]['eigval']
    k=5
    plot(([0,]+list(cumsum(EV[:k])))/sum(EV))
    title('Percentage of Variance Explained for '+ m)
    ylabel('Percentage of Variance')
    xlabel('# Eigenvector')
    grid()
    


f=plt.figure(figsize=(15,4))
j=1
for m in ['TMIN', 'TOBS', 'TMAX']: #,
    pltVarExplained(j)
    j+=1


f=plt.figure(figsize=(15,4))
j=1
for m in ['SNOW', 'SNWD', 'PRCP']:
    pltVarExplained(j)
    j+=1 


sc.stop()





#sc.stop()


#setup
data_dir='../../Data/Weather'
file_index='BSBSSSSS'
m='TOBS'


# ## Reconstruction using top eigen-vectors
# For measurement = {{m}}
# 

# ## Load the required libraries
# 

# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command


#%matplotlib inline
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets


from pyspark import SparkContext
#sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)





# ## Read Statistics File
# 

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements


# ## Read data file into a spark DataFrame
# We focus on the snow-depth records, because the eigen-vectors for them make sense.
# 

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==m)
df.show(5)


# ### Plot Reconstructions
# 
# Construct approximations of a time series using the mean and the $k$ top eigen-vectors
# First, we plot the mean and the top $k$ eigenvectors
# 

import pylab as plt
fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=3
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])


v=[np.array(EigVec[:,i]).flatten() for i in range(np.shape(EigVec)[1])]


# ### plot the percent of residual variance on average
# 

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=np.cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');


# ## Process whole dataframe to find best and worse residuals
# 

# ### Add to each row in the dataframe a residual values 
# Residuals are after subtracting in sequence: the mean, the projection on the first eigen-vector the projection on the second eigen-vector etc.
# 
# `decompose(row)` axtracts the series from the row, computes the residuals and constructs a new row that is reassembled into a dataframe.
# 

def decompose(row):
    """compute residual and coefficients for decomposition           

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :returns: the input row with additional information from the eigen-decomposition.
    :rtype: SparkSQL Row 

    Note that Decompose is designed to run inside a spark "map()" command.
    Mean and v are sent to the workers as local variables of "Decompose"

    """
    Series=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    recon=Eigen_decomp(None,Series,Mean,v);
    total_var,residuals,reductions,coeff=recon.compute_var_explained()
    #print coeff
    residuals=[float(r) for r in residuals[1]]
    coeff=[float(r) for r in coeff[1]]
    D=row.asDict()
    D['total_var']=float(total_var[1])
    D['res_mean']=residuals[0]
    for i in range(1,len(residuals)):
        D['res_'+str(i)]=residuals[i]
        D['coeff_'+str(i)]=coeff[i-1]
    return Row(**D)


rdd2=df.rdd.map(decompose)
df2=sqlContext.createDataFrame(rdd2)
row,=df2.take(1)

#filter out vectors for which the mean is a worse approximation than zero.
print 'before filter',df2.count()
df3=df2.filter(df2.res_mean<1)
print 'after filter',df3.count()


# Sort entries by increasing values of ers_3
df3=df3.sort(df3.res_3,ascending=True)


def plot_decomp(row,Mean,v,fig=None,ax=None,Title=None,interactive=False):
    """Plot a single reconstruction with an informative title

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :param Mean: The mean vector of all measurements of a given type
    :param v: eigen-vectors for the distribution of measurements.
    :param fig: a matplotlib figure in which to place the plot
    :param ax: a matplotlib axis in which to place the plot
    :param Title: A plot title over-ride.
    :param interactive: A flag that indicates whether or not this is an interactive plot (widget-driven)
    :returns: a plotter returned by recon_plot initialization
    :rtype: recon_plot

    """
    target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    if Title is None:
        Title='%s / %d    %s'%(row['station'],row['year'],row['measurement'])
    eigen_decomp=Eigen_decomp(range(1,366),target,Mean,v)
    plotter=recon_plot(eigen_decomp,year_axis=True,fig=fig,ax=ax,interactive=interactive,Title=Title)
    return plotter

def plot_recon_grid(rows,column_n=4, row_n=3, figsize=(15,10)):
    """plot a grid of reconstruction plots

    :param rows: Data rows (as extracted from the measurements data-frame
    :param column_n: number of columns
    :param row_n:  number of rows
    :param figsize: Size of figure
    :returns: None
    :rtype: 

    """
    fig,axes=plt.subplots(row_n,column_n, sharex='col', sharey='row',figsize=figsize);
    k=0
    for i in range(row_n):
        for j in range(column_n):
            row=rows[k]
            k+=1
            #_title='%3.2f,r1=%3.2f,r2=%3.2f,r3=%3.2f'\
            #        %(row['res_mean'],row['res_1'],row['res_2'],row['res_3'])
            #print i,j,_title,axes[i,j]
            plot_decomp(row,Mean,v,fig=fig,ax=axes[i,j],interactive=False)
    return None


# #### Different things to try
# The best/worst rows in terms of res_mean,res_1, res_2, res_3
# 
# The rows with the highest lowest levels of coeff1, coeff2, coeff3, when the corresponding residue is small.
# 

df4=df3.filter(df3.res_2<0.4).sort(df3.coeff_2)
rows=df4.take(12)
df4.select('coeff_2','res_2').show(4)


plot_recon_grid(rows)


get_ipython().run_line_magic('pinfo', 'df3.sort')


df5=df3.filter(df3.res_2<0.4).sort(df3.coeff_2,ascending=False)
rows=df5.take(12)
df5.select('coeff_2','res_2').show(4)


plot_recon_grid(rows)


# ## Interactive plot of reconstruction
# 
# Following is an interactive widget which lets you change the coefficients of the eigen-vectors to see the effect on the approximation.
# The initial state of the sliders (in the middle) corresponds to the optimal setting. You can zero a positive coefficient by moving the slider all the way down, zero a negative coefficient by moving it all the way up.
# 

row=rows[0]
target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
eigen_decomp=Eigen_decomp(None,target,Mean,v)
total_var,residuals,reductions,coeff=eigen_decomp.compute_var_explained()
res=residuals[1]
print 'residual normalized norm  after mean:',res[0]
print 'residual normalized norm  after mean + top eigs:',res[1:]

plotter=recon_plot(eigen_decomp,year_axis=True,interactive=True)
display(plotter.get_Interactive())


# ### What is the distribution of the residuals and the coefficients?
# 
# To answer this question we extract all of the values of `res_3` which is the residual variance after the Mean and the 
# first two Eigen-vectors have been subtracted out. We rely here on the fact that `df3` is already sorted according to `res_3`
# 

# A function for plotting the CDF of a given feature
def plot_CDF(feat):
    rows=df3.select(feat).sort(feat).collect()
    vals=[r[feat] for r in rows]
    P=np.arange(0,1,1./(len(vals)))
    vals=[vals[0]]+vals
    vals1=vals[:-1]
    plot(vals1,P)
    title('cumulative distribution of '+feat)
    ylabel('number of instances')
    xlabel(feat)
    grid()


plot_CDF('res_2')


plot_CDF('coeff_2')


filename=data_dir+'/decon_'+file_index+'_'+m+'.parquet'
get_ipython().system('rm -rf $filename')
df3.write.parquet(filename)


get_ipython().system('du -sh $data_dir/*.parquet')





#sc.stop()


#setup
data_dir='../../Data/Weather'
file_index='BSBSSSSS'
m='PRCP'


# # Reconstruction using top eigen-vectors
# For measurement = {{m}}
# 

# ## Load the required libraries
# 

# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets


from pyspark import SparkContext
sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)


# ## Read Statistics File
# 

data_dir='../../Data/Weather'
#file_index='BBBSBBBB'
file_index='BSBSSSSS'


from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements


# ## Read data file into a spark DataFrame
# We focus on the snow-depth records, because the eigen-vectors for them make sense.
# 

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==m)
df.show(5)


# ### Create a matrix with all of the series
# 

rows=df.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()

T=np.vstack(rows)
shape(T)


# ### Plot two time series
# `SNWD` stands for `snow-depth`, which explains why it is zero during the summer
# 

fig, ax = plt.subplots(figsize=(6,4));
YP=YearPlotter()
YP.plot(T[16:18].transpose(),fig,ax,title=m)


# ### Plot Reconstructions
# 
# Construct approximations of a time series using the mean and the $k$ top eigen-vectors
# First, we plot the mean and the top $k$ eigenvectors
# 

fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=3
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])


# ### plot the percent of residual variance on average
# 

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');


# ## Interactive plot of reconstruction
# 
# Following is an interactive widget which lets you change the coefficients of the eigen-vectors to see the effect on the approximation.
# The initial state of the sliders (in the middle) corresponds to the optimal setting. You can zero a positive coefficient by moving the slider all the way down, zero a negative coefficient by moving it all the way up.
# 

i=10
v=[np.array(EigVec[:,i]).flatten() for i in range(shape(EigVec)[1])]
eigen_decomp=Eigen_decomp(None,T[i],Mean,v)
total_var,residuals,reductions,coeff=eigen_decomp.compute_var_explained()
res=residuals[1]
print 'residual normalized norm  after mean:',res[0]
print 'residual normalized norm  after mean + top eigs:',res[1:]


plotter=recon_plot(eigen_decomp,year_axis=True,interactive=True)
display(plotter.get_Interactive())


# ## Process whole dataframe to find best and worse residuals
# 

# ### Add to each row in the dataframe a residual values 
# Residuals are after subtracting in sequence: the mean, the projection on the first eigen-vector the projection on the second eigen-vector etc.
# 

# `decompose(row)` axtracts the series from the row, computes the residuals and constructs a new row that is reassembled into a dataframe.
# 

rows=df.take(3)
L=[]
for row in rows:
    row_out=decompose(row,Mean,v)
    for field in ('res_mean','res_1','res_2','res_3'):
        print field,':',row_out[field],
    print


def decompose(row):
    """compute residual and coefficients for decomposition           

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :returns: the input row with additional information from the eigen-decomposition.
    :rtype: SparkSQL Row 

    Note that Decompose is designed to run inside a spark "map()" command.
    Mean and v are sent to the workers as local variables of "Decompose"

    """
    Series=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    recon=Eigen_decomp(None,Series,Mean,v);
    total_var,residuals,reductions,coeff=recon.compute_var_explained()
    #print coeff
    residuals=[float(r) for r in residuals[1]]
    coeff=[float(r) for r in coeff[1]]
    D=row.asDict()
    D['total_var']=float(total_var[1])
    D['res_mean']=residuals[0]
    for i in range(1,len(residuals)):
        D['res_'+str(i)]=residuals[i]
        D['coeff_'+str(i)]=coeff[i-1]
    return Row(**D)


rdd2=df.rdd.map(decompose)
df2=sqlContext.createDataFrame(rdd2)
row,=df2.take(1)

#filter out vectors for which the mean is a worse approximation than zero.
print 'before filter',df2.count()
df3=df2.filter(df2.res_mean<1)
print 'after filter',df3.count()


df3=df3.sort(df3.res_3,ascending=True)
rows=df3.take(12)


def plot_decomp(row,Mean,v,fig=None,ax=None,Title=None,interactive=False):
    """Plot a single reconstruction with an informative title

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :param Mean: The mean vector of all measurements of a given type
    :param v: eigen-vectors for the distribution of measurements.
    :param fig: a matplotlib figure in which to place the plot
    :param ax: a matplotlib axis in which to place the plot
    :param Title: A plot title over-ride.
    :param interactive: A flag that indicates whether or not this is an interactive plot (widget-driven)
    :returns: a plotter returned by recon_plot initialization
    :rtype: recon_plot

    """
    target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    if Title is None:
        Title='%s / %d    %s'%(row['station'],row['year'],row['measurement'])
    eigen_decomp=Eigen_decomp(range(1,366),target,Mean,v)
    plotter=recon_plot(eigen_decomp,year_axis=True,fig=fig,ax=ax,interactive=interactive,Title=Title)
    return plotter

def plot_recon_grid(rows,column_n=4, row_n=3, figsize=(15,10)):
    """plot a grid of reconstruction plots

    :param rows: Data rows (as extracted from the measurements data-frame
    :param column_n: number of columns
    :param row_n:  number of rows
    :param figsize: Size of figure
    :returns: None
    :rtype: 

    """
    fig,axes=plt.subplots(row_n,column_n, sharex='col', sharey='row',figsize=figsize);
    k=0
    for i in range(row_n):
        for j in range(column_n):
            row=rows[k]
            k+=1
            _title='%3.2f,r1=%3.2f,r2=%3.2f,r3=%3.2f'                    %(row['res_mean'],row['res_1'],row['res_2'],row['res_3'])
            #print i,j,_title,axes[i,j]
            plot_decomp(row,Mean,v,fig=fig,ax=axes[i,j],Title=_title,interactive=False)
    return None


df3=df3.sort(df3.res_3)
rows=df3.take(12)
df3.select('res_mean','res_1','res_2','res_3').show(4)


plot_recon_grid(rows)


df3=df3.sort(df3.res_3,ascending=False)
rows=df3.take(12)
df3.select('res_mean','res_1','res_2','res_3').show(4)


plot_recon_grid(rows)


# ### How well-explained are the vectors in this collection?
# 
# To answer this question we extract all of the values of `res_3` which is the residual variance after the Mean and the 
# first two Eigen-vectors have been subtracted out. We rely here on the fact that `df3` is already sorted according to `res_3`
# 

res3=df3.select('res_3').collect()
R3=[r['res_3'] for r in res3]
plot(R3)
title('distribution of residuals after 3 vectors')
xlabel('number of instances')
ylabel('residual')
ylim([0,1])
grid()


filename=data_dir+'/decon_'+file_index+'_'+m+'.parquet'
get_ipython().system('rm -rf $filename')
df3.write.parquet(filename)


get_ipython().system('du -sh $data_dir/*.parquet')





get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
#import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
import ipywidgets as widgets

print 'version of ipwidgets=',widgets.__version__

import sys
sys.path.append('lib')
from recon_plot import recon_plot
from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter


# ### High-dimensional vectors
# To get an intuition about the working of the PCA, we used an example in the plane, or $R^2$.
# While useful for intuition, this is not the typical case in which we use PCA. Typically we are interested in vectors in a space whose dimension is in the hundreds or more.
# 
# How can we depict such vectors? If the coordinates of the vector have a natural order. For example, if the coordinates correspond to a grid of times, then a good representation is to make a plot in which the $x$-axis is the time and the $y$-axis is the value that corresponds to this time. 
# 
# Later in this class we will consider vectors that correspond to the temperature at a particular location each day of the year. These vectors will be of length 365 (we omit the extra days of leap years) and the PCA analysis will reveal the low dimensional subspace.
# 

# ### Function approximation
# For now, we will consider the vectors that are defined by sinusoidal functions.
# 

# We define a grid that extends from o to 2*pi
step=2*pi/365
x=arange(0,2*pi,step)
len(x)


# #### Define a basis
# 
# The dimension of the space is 629.
# 
# We define some functions based on $\sin()$ and $\cos()$ 
# 

c=sqrt(step/(pi))
v=[]
v.append(np.array(cos(0*x))*c/sqrt(2))
v.append(np.array(sin(x))*c)
v.append(np.array(cos(x))*c)
v.append(np.array(sin(2*x))*c)
v.append(np.array(cos(2*x))*c)
v.append(np.array(sin(3*x))*c)
v.append(np.array(cos(3*x))*c)
v.append(np.array(sin(4*x))*c)
v.append(np.array(cos(4*x))*c)

print"v contains %d vectors"%(len(v))


# plot some of the functions (plotting all of them results in a figure that is hard to read.
figure(figsize=(8,6))
for i in range(5):
    plot(x,v[i])
grid()
legend(['const','sin(x)','cos(x)','sin(2x)','cos(2x)'])


# #### Check that it is  an orthonormal basis
# This basis is not **complete** it does not span the space of all functions. It spans a 9 dimensional sub-space.
# 
# We will now check that this is an **orthonormal** basis. In other words, the length of each vector is 1 and every pair of vectors are orthogonal.
# 

for i in range(len(v)): 
    print
    for j in range(len(v)):
        a=dot(v[i],v[j])
        a=round(1000*a+0.1)/1000
        print '%1.0f'%a,


# #### Rewriting the basis as a matrix
# 
# Combining the vectors as rows in a matrix allows us use very succinct (and very fast) matrix multiplications instead of for loops with vector products.
# 

U=vstack(v)
shape(U)


# ### Approximating an arbitrary function
# We now take an unrelated function $f=|x-4|$
# and see how we can use the basis matrix `U` to approximate it. 
# 

f1=abs(x-4)
plot(x,f1);
grid()


# #### Approximations  of increasing accuracy
# To understand the workings of the basis, we create a sequence of approximations $g(i)$ such that $g(i)$ is an approximation that uses the first $i$ vectors in the basis.
# 
# The larger $i$ is, the closer $g(i)$ is to $f$. Where the distance between $f$ and $g(i)$ is defined by the euclidean norm:
# $$   \| g(i)- f \|_2
# $$
# 

# #### Plotting the approximations
# Below we show how increasing the number of vectors in the basis improves the approximation of $f$.
# 

eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),v)
recon_plot(eigen_decomp,year_axis=False,Title='Best Reconstruction',interactive=False);


eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),v)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True);
display(plotter.get_Interactive())


# ### Excercise
# Visually, it is clear that $g(i)$ is getting close to $f$ as $i$ increases. To quantify the improvement, calculate 
# $ \| g(i)- f \|_2 $ as a function of $i$
# 

# ### Recovering from Noise
# 

noise=np.random.normal(size=x.shape)
f2=2*v[1]-4*v[5] +1*noise
plot(x,f2);


eigen_decomp=Eigen_decomp(x,f2,np.zeros(len(x)),v)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True);
display(plotter.get_Interactive())











# # Montana Weather Analysis
# 
# This is a report on the historical analysis of weather patterns in an area that overlaps the area of the state of Montana.
# 
# <p><img src="r_figures/montana_map2.png" style="height:500px; width:700px" /></p>
# 
# 
# The data we will use here comes from [NOAA](https://www.ncdc.noaa.gov/). Specifically, it was downloaded from This [FTP site](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/).
# 
# We focused on six measurements:
# * **TMIN, TMAX:** the daily minimum and maximum temperature.
# * **TOBS:** The average temperature for each day.
# * **PRCP:** Daily Percipitation (in mm)
# * **SNOW:** Daily snowfall (in mm)
# * **SNWD:** The depth of accumulated snow.
# 
# <table>
#   <tr>
#     <th>Measurement</th>
#     <th>25th Quantile</th> 
#     <th>50th Quantile</th> 
#     <th>75th Quantile</th> 
#     <th>Min</th> 
#     <th>Max</th>
#     <th>NaN count</th>
#   </tr>
#   <tr>
#     <th>TMIN</th>
#     <th>-6.1</th> 
#     <th>0.0</th> 
#     <th>6.69</th>
#     <th>-25</th>
#     <th>26.7</th>
#     <th>31807</th>
#   </tr>
#   <tr>
#     <th>TMAX</th>
#     <th>6.10</th> 
#     <th>14.3</th> 
#     <th>23.9</th>
#     <th>-11.1</th>
#     <th>42.8</th>
#     <th>35077</th>
#   </tr>
#   <tr>
#     <th>TMIN</th> 
#     <th>-6.1</th> 
#     <th>0.0</th> 
#     <th>6.69</th> 
#     <th>-25</th>
#     <th>26.7</th>
#     <th>31807</th>
#   </tr>
#   <tr>
#     <th>TOBS</th> 
#     <th>0.6</th> 
#     <th>8.2</th> 
#     <th>17.7</th> 
#     <th>-17.2</th>
#     <th>39.4</th>
#     <th>30791</th>
#   </tr>
#   <tr>
#     <th>TMIN</th>
#     <th>-6.1</th> 
#     <th>0.0</th> 
#     <th>6.69</th> 
#     <th>-25</th>
#     <th>26.7</th>
#     <th>31807</th>
#   </tr>
#   <tr>
#     <th>PRCP</th>
#     <th>0</th> 
#     <th>0</th> 
#     <th>0</th>
#     <th>0</th>
#     <th>74.68</th>
#     <th>13430</th>
#   </tr>
#    <tr>
#     <th>SNOW</th>
#     <th>0</th> 
#     <th>0</th> 
#     <th>0</th>
#     <th>0</th>
#     <th>20.29</th>
#     <th>16479</th>
#   </tr>
#     <tr>
#     <th>SNWD</th>
#     <th>0</th> 
#     <th>0</th> 
#     <th>0</th>
#     <th>0</th>
#     <th>172.75</th>
#     <th>13043</th>
#   </tr>
#  </table>
# 
# ## Sanity-check: Temperature
# 
# <p>We compare  min and max temperatures with graphs that we obtained from a site called <a href="http://www.usclimatedata.com/climate/roy/montana/united-states/usmt0293" target="_blank">US Climate Data</a>
# 
# <p>&nbsp;</p>
# <div>
#     <img  src="r_figures/montana_climate_graph2.png" style="float:left;width:290px;height:320px;"/>
#     <img src="r_figures/tmin_tmax_montanna2.png" style="height:320px; width:700px" />
#     <div style="clear:both;"></div>
# </div>
# <div>
# <table style="float:left">
#   <tr>
#     <th>Average Low temperature in Celcius</th>
#     <th>Average Mean TMIN temperature in Celcius</th> 
#     <th>Average High temperature in Celcius</th> 
#     <th>Average Mean TMAX temperature in Celcius</th> 
#   </tr>
#   <tr>
#     <td>-13.33</td>
#     <td>-10.13</td>
#     <td>0</td> 
#     <td>2.68</td>
#   </tr>
#   <tr>
#     <td>-10.55</td>
#     <td>-9.26</td>
#     <td>3.33</td>
#     <td>3.78</td>
#   </tr>
#   <tr>
#     <td>-5.55</td>
#     <td>-6.19</td>
#     <td>8.33</td> 
#     <td>7.25</td>
#   </tr>
#   <tr>
#     <td>-0.55</td>
#     <td>-1.27</td>
#     <td>14.44</td> 
#     <td>13.21</td>
#   </tr>
#   <tr>
#     <td>4.44</td>
#     <td>3.84</td> 
#     <td>19.44</td>
#     <td>18.73</td>
#   </tr>
#   <tr>
#     <td>8.88</td>
#     <td>8.23</td> 
#     <td>25</td> 
#     <td>23.63</td> 
#   </tr>
#   <tr>
#     <td>11.66</td>
#     <td>11.09</td> 
#     <td>29.44</td> 
#     <td>28.9</td> 
#   </tr>
#   <tr>
#     <td>11.11</td>
#     <td>9.32</td> 
#     <td>29.44</td>
#     <td>27.25</td>
#   </tr>
#   <tr>
#     <td>5.55</td>
#     <td>3.74</td> 
#     <td>22.77</td>
#     <td>20.15</td>
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>-1.59</td> 
#     <td>16.11</td>
#     <td>12.91</td>
#   </tr>
#   <tr>
#     <td>-6.66</td>
#     <td>-6.41</td> 
#     <td>6.66</td>
#     <td>6.41</td> 
#   </tr>
#   <tr>
#     <td>-12.22</td>
#     <td>-9.23</td> 
#     <td>1.11</td>
#     <td>3.28</td> 
#   </tr>
#  </table>
#  <div style="clear:both;"></div>
#  <h3><strong>Comparison of Avg Low temperature from usclimate site with Avg Mean TMIN temperature from ncdc site</strong></h3> <br><br>
#  </div>
#  
# 
# * <strong>Null hypothesis</strong>: The distribution of average low temperature in celcius obtained from us climate data site is same as the average mean TMIN temperature in celcius obtained from ncdc site. <br/>
# 
# * <i><strong>Alternate hypothesis</strong>: The distributions are are not same.<br/></i>
# 
# * <strong>Statistical test</strong>: Two sided Kolmogorov–Smirnov(KS) test which is a statistical test that is used to compare and find the similarity between two continuous distributions that are on same scale. <br/>
# 
# 
# * <i><strong>Result</strong>: The P-Value obtained from this comparison is 0.78 > 0.05. Hence, the null hypothesis cannot be rejected and the distribution of average low temperature in celcius obtained from usclimate data site is same as the average mean TMIN temperature in celcius obtained from ncdc site.<br/></i>
# 
# 
#  <h3><strong>Comparison of Avg High temperature from usclimate site with Avg Mean TMAX temperature from ncdc site</strong></h3> <br><br>
# 
# * <strong>**Null hypothesis**</strong>: The distribution of average high temperature in celcius obtained from us climate data site is same as the average Mean TMAX temperature in celcius obtained from ncdc site.<br/>
# 
# 
# * <i><strong>**Alternate hypothesis**</strong>: The distributions are are not same.<br/></i>
# 
# 
# * <strong>**Statistical test**</strong>: Two sided Kolmogorov–Smirnov(KS) test which is a statistical test that is used to compare and find the similarity between two continuous distributions that are on same scale. <br/>
# 
# 
# * <i><strong>**Result**</strong>: The P-Value obtained from this comparison is 0.99 > 0.05. Hence, the null hypothesis cannot be rejected and the distribution of average high temperature incelcius obtained from us climate data site is same as the average mean TMAX temperature in celcius obtained from ncdc site.</i><br/>
# 
# <p></p>
# 
# 
# ## Sanity-check: Precipitation
# 
# 
# <p>We compare precipitation with graphs that we obtained from a site called <a href="http://www.usclimatedata.com/climate/roy/montana/united-states/usmt0293" target="_blank">US Climate Data</a>
# 
# <p>&nbsp;</p>
# <div>
#     <img  src="r_figures/montana_climate_graph2.png" style="float:left;width:400px;height:320px;"/>
#     <img src="r_figures/prcp_montana.png" style="height:320px; width:500px" />
#     <div style="clear:both;"></div>
# </div>
# <div>
# <table align="center">
#   <tr>
#     <th>Average Precipitation in mm from Climate data site</th>
#     <th>Average PRCP in mm from ncdc site</th> 
#   </tr>
#   <tr>
#     <td>13</td>
#     <td>17.27</td>
#   </tr>
#   <tr>
#     <td>8</td>
#     <td>13.46</td>
#   </tr>
#   <tr>
#     <td>21</td>
#     <td>22.08</td>
#   </tr>
#   <tr>
#     <td>29</td>
#     <td>35.52</td>
#   </tr>
#   <tr>
#     <td>69</td>
#     <td>69.62</td> 
#   </tr>
#   <tr>
#     <td>58</td>
#     <td>75.4</td> 
#   </tr>
#   <tr>
#     <td>50</td>
#     <td>39.49</td> 
#   </tr>
#   <tr>
#     <td>39</td>
#     <td>35.49</td> 
#   </tr>
#   <tr>
#     <td>31</td>
#     <td>35.78</td> 
#   </tr>
#   <tr>
#     <td>19</td>
#     <td>22.44</td> 
#   </tr>
#   <tr>
#     <td>11</td>
#     <td>11.99</td>  
#   </tr>
#   <tr>
#     <td>13</td>
#     <td>16.99</td> 
#   </tr>
#  </table>
#  <div style="clear:both;"></div>
#  </div>
# 
# * <strong>**Null hypothesis**</strong>: The distribution of average precipitation in mm obtained from us climate data site is same as the Average PRCP in mm obtained from ncdc site.<br/>
# 
# * <i><strong>**Alternate hypothesis**</strong></i>: The distributions are are not same.<br/>
# 
# * <strong>**Statistical test**</strong>: Two sided Kolmogorov–Smirnov(KS) test which is a statistical test that is used to compare and find the similarity between two continuous distributions that are on same scale.<br/> 
# 
# * <i><strong>**Result**</strong></i>: The P-Value obtained from this comparison is 0.78 > 0.05. Hence, the null hypothesis cannot be rejected and the distribution of average precipitation in mm obtained from us climate data site is same as the average PRCP in mm obtained from ncdc site.<br/>
# 
# <p></p>
# 
# ## PCA analysis
# 
# Out of the 365 eigen vectors that are generated, 81 of them have negative eigen values. Hence, a part of the data is corrupted, but we have tried to come up with the best analysis possible with the given dataset.
# 
# For each of the six measurement, we compute the percentate of the variance explained as a function of the number of eigen-vectors used.
# 
# 
# ![VarExplained1.png](r_figures/variance_1_montana.png)
# We see that the top 5 eigen-vectors explain 24% of variance for TMIN, 34% for TOBS and 27% for TMAX.
# 
# We conclude that of the three, TOBS is best explained by the top 5 eigenvectors. This is especially true for the first eigen-vector which, by itself, explains 25% of the variance.
# 
# ![VarExplained2.png](r_figures/variance_2_montana.png)
# 
# The top 5 eigenvectors explain 13% for SNOW and 11% for PRCP. Both are low values. On the other hand the top 5 eigenvectors explain 82% of the variance for SNWD. This means that these top 5 eigenvectors capture most of the variation in the snow depth signals. Based on that we will dig deeper into the PCA analysis for snow-depth.
# <p></p>
# 
# 
# ## Analysis of Snow Depth(SNWD)
# 
# We choose to analyze the eigen-decomposition for snow-depth because the first 4 eigen-vectors explain 80% of the variance.
# 
# First, we graph the mean and the top 4 eigen-vectors.
# 
# We observe that the snow season is from November to April, where the month of January and February marks the peak of the snow-depth.Next we interpret the eigen-functions.
# 
# <p><img src="r_figures/snowd_mean2.png" style="height:400px; width:700px" /></p>
# 
# 
# ### Interpretation of eig1
# 
# 
# The main difference between the shape of eigen-function(eig1) with that of mean function is that that the eigen-function does not peak during october-december while the mean is peaking near october-december. However,let us see if the overall shape of first eigen-function (eig1) is similar to the mean function.
# 
# 
# * **Null hypothesis**: The overall shape of first eigen-function (eig1) is similar to the mean function. <br/>
# 
# 
# * **Alternate hypothesis**: The distributions are are not similar.<br/>
# 
# 
# * **Statistical test**: The cofficient of variation test which is a statistical test that is used to compare and find the similarity between two continuous distributions that are on different scales. <br/>
# 
# 
# * **Result**: The cofficient of variation for mean distribution is 35.56/30.11 = 1.18. The cofficient of variation for eig1 distribution is 0.038/0.035 = 1.06. Since the cofficient of variation for both distributions are almost same, the two distributions are similar.
# 
# The interpretation of this shape is that eig1 represents the overall amount of snow above/below the mean, but without changing the distribution over time.
# 
# ### Interpretation of eig2,eig3 and eig4
# 
# **eig2,eig3 and eig4** are similar in the following way. They all oscilate between positive and negative values. In other words, they correspond to changing the distribution of the snow depth over the winter months, but they don't change the total (much).
# 
# They can be interpreted as follows:
# * **eig2:** more snow in Jan and Feb, less snow from mid March to end of May.
# * **eig3:** more snow in Nov and Dec, less snow in April.
# * **eig4:** more snow in Jan, less snow in Feb.

# ## Examples of SNWD reconstructions
# 
# In all of the below graphs we find that the reconstruction of the target vector gets more and more accurate with additional eigen vectors.
# 
# 
# ### Coeff1
# #### Coeff1: most positive
# <img  src="r_figures/coeff1_high_montana.png" style="width:800px;height:600px;"/>
# #### Coeff1: most negative
# <img  src="r_figures/coeff1_low_montana.png" style="width:800px;height:600px;"/>
# 
# * Large positive values of coeff1 correspond to more than average snow. 
# * Large negative values correspond to less than average snow. 
# 
# ### Coeff2
# #### Coeff2: most positive
# <img  src="r_figures/coeff2_high_montana.png" style="width:800px;height:600px;"/>
# 
# #### Coeff2: most negative
# <img  src="r_figures/coeff2_low_montana.png" style="width:800px;height:600px;"/>
# 
# * Large positive values of coeff2 correspond to an early snow season (from mid January to Febraury). 
# * Large negative values for coeff2 correspond to a late snow season(mid March to mid April). 
# ### Coeff3
# #### Coeff3: most positive
# <img  src="r_figures/coeff3_high_montana.png" style="width:800px;height:600px;"/>
# #### Coeff3: most negative
# <img  src="r_figures/coeff3_low_montana.png" style="width:800px;height:600px;"/>
# 
# * Large positive values of coeff3 correspond to a snow season with fluctuations from January to March, and a peak from November to December. 
# * Large negative values of coeff3 correspond to a snow season with  peak from February to April and a spike from October to December. 

# ## Cumulative Distributions of SNWD
# 
# 
# <p><img src="r_figures/cum_residual.png" style="height:400px; width:500px" /></p>
# 
# The above graph shows a plot of the cumulative distribution function of the first three residuals. As expected, residual3 is better than residual2 which in turn is better than residual1 since the first three eigen vectors combined capture more variance than the first two and first eigen vector alone. In other words the residual error from reconstruction using only the first eigen vector grows faster than the residual error from reconstruction using the first two eigen vectors combined which in turn grows faster than the residual error from reconstruction using the first three eigen vectors combined. 
# 
# 
# <p><img src="r_figures/cum_coeff.png" style="height:400px; width:500px" /></p>
# 
# The above graph shows a plot of the cumulative distribution function of the first three coefficients of the eigen vectors. Since the first eigen vector gives the direction of maximum variation it is natural that many data points are dominated by a large coefficient1 compared to the other coefficients.
# 
# ## Visualizing the distribution of observations for SNWD
# 
# <p><img src="r_figures/montana_map.png" style="height:400px; width:800px" /></p>
# 
# The above map shows a distribution of the different weather stations that have collected the SNOWD measurement in the Montana region. The size of the circles in the map is proportional to the number of datapoints contributed by a particular weather station. The fill color of the circles is a heatmap denoting the average value of coefficient1 which is the deviation of SNOWD from the mean. A red weather station indicates a high SNOWD value while a blue weather station indicates a relatively lower average SNOWD value.

# # Residual Analysis of SNWD
# 
# <p><img src="r_figures/histogram.png" style="height:400px; width:500px" /></p>
# 
# The above histogram shows a plot of the significance of data. The values show the correlation values between various stations. The sudden spike observed at 0.4 can be attributed to assigning a value of 0.4 to the diagonal elements (correlation between the same station). Apart from this spike, the data resembles a normal distribution. 
# 
# <p><img src="r_figures/residual2.png" style="height:400px; width:500px" /></p>
# 
# The above figure shows a heatmap of the correlation probability values. We observe that there are alernating points of dark and white. Given the objective is to cluster points with similar correlation values, we need a way to group the dark patterns together. The 54x54 matrix corresponds to correlation of 54 stations with each of the other 54 stations. In order to check spatial similarity in correlation values between weather stations, PCA is applied to the matrix shown in the above figure. We take the first 4 eigenvectors of the P_norm matrix and try to cluster data. In order to facilitate this, we sort the components of the first eigenvector of the decomposition and rearrange the stations according to the indices of the sorted eigenvector. 
# 
# The plots obtained on applying this procedure to the first 4 eigenvectors are shown below.
# 
# <p><img src="r_figures/residual3.png" style="height:400px; width:500px" /></p>
# 
# If we look at the figure at the bottom left box, we notice a (13,13) square clustered together in blue with small yellow patches among them. This is a representation of stations with similar correlation values and are potentially correlated together. We now plot the geographical spread of the stations to check for spatial similarity. The latitude and longitude data of the correlated stations are taken and plot using gmplot library.We can see that among top 13 most correlated stations, 3 of them are slighlty further away from the remaining 10 stations. This is because of the few yellow patches that we had in the (13,13) square in the bottom left corner of the previous figure. This shows that the stations which are all very close together share comparative snow depth values. 
# 
# <p><img src="r_figures/montana_map3.png" style="height:300px; width:500px" /></p>
# 
# An alternate method of visualizing the data is to plot the weather stations along with the coefficients of the first principal components. The below geo plot indicates the principal components by triangles with the size of the triangle representing the magnitude of the coefficient and the opacity of the triangle representing the sign of the coefficient (filled triangles for negative and un-filled triangles for positive coefficients). As we can see there are a few close weather stations with similar triangle structures,that is, similar principal components.
# 
# <p><img src="r_figures/montana_map5.png" style="height:300px; width:500px" /></p>
# 
#  

# ## Spatial vs Temporal variation of SNWD

# We now estimate the relative importance of location-to-location variation relative to year-by-year variation. These are measured using the fraction by which the variance is reduced when we subtract from each station/year entry the average-per-year or the average-per-station respectively.
# 
# 
# Here are the results:
# 
# 
# coeff_1<br/>
# total RMS = 1757.62414965 <br/>
# RMS removing mean-by-station= 911.871235198 Fraction explained= 48.1%<br/>
# RMS removing mean-by-year = 1510.65793281 Fraction explained= 14.05%<br/>
# 
# 
# coeff_2 <br/>
# total RMS = 756.872346905
# RMS removing mean-by-station= 578.445915616 Fraction explained= 23.5%<br/>
# RMS removing mean-by-year = 555.289152167 Fraction explained= 26.5%<br/>
# 
# 
# coeff_3<br/>
# total RMS = 530.7728529<br/>
# RMS removing mean-by-station= 499.701075446 Fraction explained= 5.8%<br/>
# RMS removing mean-by-year = 345.465554066 Fraction explained= 34.9%<br/>
# 
# 
# We see that the variation by year explains more than the variation by station. 

# ### Extra analysis - Climate change in Montana resulting in increased chances of global warming 
# 
# As mentioned in the introduction we use the data to look for evidences of global warming due to climate changes using Montana weather data. This analysis is performed on the town Flatwillow,Montana(station id: USC00243013).
# 
# <p>
# <img src="r_figures/extra1.png" style="height:300px; width:400px; float:left" />
# <img src="r_figures/extra4.png" style="height:300px; width:400px;" /></p>
# 
# As shown in the above figure the observed temperatures have been increasing over the past century steadily. This is a reflection of the global phenomenon wherein temperatures are increasing annually which tends to increase precipitation in cold areas and faster melting of snow during spring.
# 
# <img src="r_figures/extra5.png" style="height:300px; width:400px;" /></p>
# 
# From the above plot we see that the quantity of snow has been on the rise for the past 100 years on the average. 
# As expected, warmer air in a cold place like Montana has led to an increase in the mean amount of precipitation over the years.




#setup
data_dir='../../Data/Weather'
file_index='SBBBSBSS'
m='SNWD'


# ## Reconstruction using top eigen-vectors
# For measurement = {{m}}
# 

# ## Load the required libraries
# 



# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command


#%matplotlib inline
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets


from pyspark import SparkContext
#sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)





# ## Read Statistics File
# 

from pickle import load

#read statistics
print data_dir
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements


# ## Read data file into a spark DataFrame
# We focus on the snow-depth records, because the eigen-vectors for them make sense.
# 

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==m)
df.show(5)


# ### Plot Reconstructions
# 
# Construct approximations of a time series using the mean and the $k$ top eigen-vectors
# First, we plot the mean and the top $k$ eigenvectors
# 

import pylab as plt
import numpy as np
fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=4
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])
                                             


v=[np.array(EigVec[:,i]).flatten() for i in range(np.shape(EigVec)[1])]


# ### plot the percent of residual variance on average
# 

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=np.cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');


# ## Process whole dataframe to find best and worse residuals
# 

# ### Add to each row in the dataframe a residual values 
# Residuals are after subtracting in sequence: the mean, the projection on the first eigen-vector the projection on the second eigen-vector etc.
# 
# `decompose(row)` axtracts the series from the row, computes the residuals and constructs a new row that is reassembled into a dataframe.
# 

def decompose(row):
    """compute residual and coefficients for decomposition           

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :returns: the input row with additional information from the eigen-decomposition.
    :rtype: SparkSQL Row 

    Note that Decompose is designed to run inside a spark "map()" command.
    Mean and v are sent to the workers as local variables of "Decompose"

    """
    Series=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    recon=Eigen_decomp(None,Series,Mean,v);
    total_var,residuals,reductions,coeff=recon.compute_var_explained()
    #print coeff
    residuals=[float(r) for r in residuals[1]]
    coeff=[float(r) for r in coeff[1]]
    D=row.asDict()
    D['total_var']=float(total_var[1])
    D['res_mean']=residuals[0]
    for i in range(1,len(residuals)):
        D['res_'+str(i)]=residuals[i]
        D['coeff_'+str(i)]=coeff[i-1]
    return Row(**D)


rdd2=df.rdd.map(decompose)
df2=sqlContext.createDataFrame(rdd2)
row,=df2.take(1)

#filter out vectors for which the mean is a worse approximation than zero.
print 'before filter',df2.count()
df3=df2.filter(df2.res_mean<1)
print 'after filter',df3.count()


# Sort entries by increasing values of ers_3
df3=df3.sort(df3.res_3,ascending=True)


def plot_decomp(row,Mean,v,fig=None,ax=None,Title=None,interactive=False,coeff_val=1):
    """Plot a single reconstruction with an informative title

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :param Mean: The mean vector of all measurements of a given type
    :param v: eigen-vectors for the distribution of measurements.
    :param fig: a matplotlib figure in which to place the plot
    :param ax: a matplotlib axis in which to place the plot
    :param Title: A plot title over-ride.
    :param interactive: A flag that indicates whether or not this is an interactive plot (widget-driven)
    :returns: a plotter returned by recon_plot initialization
    :rtype: recon_plot

    """
    target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    if Title is None:
        Title= 'coeff %s=%s, reconst. error(res %s)=%s'%(coeff_val,row['coeff_' + str(coeff_val)],coeff_val,row['res_' + str(coeff_val)])
    eigen_decomp=Eigen_decomp(range(1,366),target,Mean,v)
    plotter=recon_plot(eigen_decomp,year_axis=True,fig=fig,ax=ax,interactive=interactive,Title=Title)
    return plotter

def plot_recon_grid(rows,column_n=4, row_n=3, figsize=(15,10),coeff_val=1):
    """plot a grid of reconstruction plots

    :param rows: Data rows (as extracted from the measurements data-frame
    :param column_n: number of columns
    :param row_n:  number of rows
    :param figsize: Size of figure
    :returns: None
    :rtype: 

    """
    fig,axes=plt.subplots(row_n,column_n, sharex='col', sharey='row',figsize=figsize);
    k=0
    for i in range(row_n):
        for j in range(column_n):
            row=rows[k]
            k+=1
            #_title='%3.2f,r1=%3.2f,r2=%3.2f,r3=%3.2f'\
            #        %(row['res_mean'],row['res_1'],row['res_2'],row['res_3'])
            #print i,j,_title,axes[i,j]
            plot_decomp(row,Mean,v,fig=fig,ax=axes[i,j],interactive=False,coeff_val=coeff_val)
    return None


# #### Different things to try
# The best/worst rows in terms of res_mean,res_1, res_2, res_3
# 
# The rows with the highest lowest levels of coeff1, coeff2, coeff3, when the corresponding residue is small.
# 

import gmplot
sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT latitude,longitude FROM weather"
df1 = sqlContext.sql(Query)
lat_long = df1.collect()
latitude = [row[0] for row in lat_long]
longitude = [row[1] for row in lat_long]
gmap = gmplot.GoogleMapPlotter(latitude[0], longitude[0], 16)
gmap.scatter(latitude, longitude, 'b', marker=True)
gmap.heatmap(latitude, longitude)
gmap.draw("Sudarshan_map.html")


df4=df3.sort(df3.coeff_1)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=1)


df4=df3.sort(df3.coeff_1,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=1)


df4=df3.sort(df3.coeff_2)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=2)


df4=df3.sort(df3.coeff_2,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=2)


df4=df3.sort(df3.coeff_3)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=3)


df4=df3.sort(df3.coeff_3,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=3)


get_ipython().run_line_magic('pinfo', 'df3.sort')


df5=df3.filter(df3.res_2<0.4).sort(df3.coeff_2,ascending=False)
rows=df5.take(12)
df5.select('coeff_2','res_2').show(4)


plot_recon_grid(rows)


# ## Interactive plot of reconstruction
# 
# Following is an interactive widget which lets you change the coefficients of the eigen-vectors to see the effect on the approximation.
# The initial state of the sliders (in the middle) corresponds to the optimal setting. You can zero a positive coefficient by moving the slider all the way down, zero a negative coefficient by moving it all the way up.
# 

row=rows[0]
target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
eigen_decomp=Eigen_decomp(None,target,Mean,v)
total_var,residuals,reductions,coeff=eigen_decomp.compute_var_explained()
res=residuals[1]
print 'residual normalized norm  after mean:',res[0]
print 'residual normalized norm  after mean + top eigs:',res[1:]

plotter=recon_plot(eigen_decomp,year_axis=True,interactive=True)
display(plotter.get_Interactive())


# ### What is the distribution of the residuals and the coefficients?
# 
# To answer this question we extract all of the values of `res_3` which is the residual variance after the Mean and the 
# first two Eigen-vectors have been subtracted out. We rely here on the fact that `df3` is already sorted according to `res_3`
# 

# A function for plotting the CDF of a given feature
def plot_CDF(feat):
    rows=df3.select(feat).sort(feat).collect()
    vals=[r[feat] for r in rows]
    P=np.arange(0,1,1./(len(vals)+1))
    vals=[vals[0]]+vals
    plot(vals,P)
    title('cumulative distribution of '+feat)
    ylabel('number of instances')
    xlabel(feat)
    grid()
    


plot_CDF('res_2')


plot_CDF('coeff_2')


# A function for plotting the CDF of a given feature
def plot_CDF(feat1,feat2,feat3,title_param):
    rows1=df3.select(feat1).sort(feat1).collect()
    vals1=[r[feat1] for r in rows1]
    P1=np.arange(0,1,1./(len(vals1)+1))
    vals1=[vals1[0]]+vals1
    rows2=df3.select(feat2).sort(feat2).collect()
    vals2=[r[feat2] for r in rows2]
    vals2=[vals2[0]]+vals2
    P2=np.arange(0,1,1./(len(vals2)))
    rows3=df3.select(feat3).sort(feat3).collect()
    vals3=[r[feat3] for r in rows3]
    vals3=[vals3[0]]+vals3
    P3=np.arange(0,1,1./(len(vals3)))
    plot(vals1,P1,label=feat1)
    plot(vals2,P2,label=feat2)
    plot(vals3,P3,label=feat3)
    title(title_param)
    ylabel('number of instances')
    xlabel('residual')
    legend()
    grid()
    
plot_CDF('res_1','res_2','res_3','cumulative distribution of residual 1,2 and 3')


plot_CDF('coeff_1','coeff_2','coeff_3','cumulative distribution of coefficients 1,2 and 3')


filename=data_dir+'/decon_'+file_index+'_'+m+'.parquet'
get_ipython().system('rm -rf $filename')
df3.write.parquet(filename)


get_ipython().system('du -sh $data_dir/*.parquet')





# ### Visualizing the distribution of the observations
# 
# ### Load the required libraries
# 

import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math
get_ipython().run_line_magic('pylab', 'inline')

import findspark
findspark.init()

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/computeStats.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)

import sys
sys.path.append('./lib')

import numpy as np
from numpy_pack import packArray,unpackArray
#from spark_PCA import computeCov
from computeStats import computeOverAllDist, STAT_Descriptions

### Read the data frame from pickle file

data_dir='../../Data/Weather'
file_index='SBBBSBSS'

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
print 'keys from STAT=',STAT.keys()

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index

df=sqlContext.read.parquet(filename)
print df.count()
df.show(50)


# ### Select data for a particular station and measurement type
# 

sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT * FROM weather\n\tWHERE measurement='%s' and station='%s'"%('TOBS','USC00242886')
print Query
df1 = sqlContext.sql(Query)
print df1.count(),'rows'
df1.show(2)
rows=df1.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()
T=np.vstack(rows)
T=T/10.  # scaling to make the temperature be in centingrates


sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT count(year) as year_cnt,station FROM weather\n\tWHERE measurement='%s' GROUP BY station ORDER BY year_cnt DESC"%('TOBS') 
print Query
df1 = sqlContext.sql(Query)
print df1.count(),'rows'
df1.show(50)


# ### Script for plotting yearly plots
# 

from YearPlotter import YearPlotter
fig, ax = plt.subplots(figsize=(10,7));
YP=YearPlotter()
YP.plot(T[:2,:].transpose(),fig,ax,title='PRCP')
#title('A sample of graphs');


# ### Distribution of missing observations
# The distribution of missing observations is not uniform throughout the year. We visualize it below.
# 

def plot_pair(pair,func):
    j=0
    fig,X=subplots(1,2,figsize=(16,6))
    axes=X.reshape(2)
    for m in pair:
        axis = axes[j]
        j+=1
        func(m,fig,axis)
        
def plot_valid(m,fig,axis):
    valid_m=STAT[m]['NE']
    YP.plot(valid_m,fig,axis,title='valid-counts '+m)
    


plot_pair(['TMIN','TMAX'],plot_valid)


plot_pair(['TOBS','PRCP'],plot_valid)


plot_pair(['SNOW', 'SNWD'],plot_valid)


# ### Plots of mean and std of observations
# 

def plot_mean_std(m,fig,axis):
    mean=STAT[m]['Mean'] / 10
    std=np.sqrt(STAT[m]['Var']) / 10
    graphs=np.vstack([mean-std,mean,mean+std]).transpose()
    YP.plot(graphs,fig,axis,title='Mean+-std   '+m)


b = plot_pair(['TMIN','TMAX'],plot_mean_std)


plot_pair(['TOBS','PRCP'],plot_mean_std)


plot_pair(['SNOW', 'SNWD'],plot_mean_std)


# ### plotting top 3 eigenvectors
# 

def plot_eigen(m,fig,axis):
    EV=STAT[m]['eigvec']
    YP.plot(EV[:,:3],fig,axis,title='Top Eigenvectors '+m)


plot_pair(['TMIN','TMAX'],plot_eigen)


plot_pair(['TOBS','PRCP'],plot_eigen)


plot_pair(['SNOW', 'SNWD'],plot_eigen)


# ### Script for plotting percentage of variance explained
# 

def pltVarExplained(j):
    subplot(1,3,j)
    EV=STAT[m]['eigval']
    k=5
    plot(([0,]+list(cumsum(EV[:k])))/sum(EV))
    title('Percentage of Variance Explained for '+ m)
    ylabel('Percentage of Variance')
    xlabel('# Eigenvector')
    grid()
    


f=plt.figure(figsize=(15,4))
j=1
for m in ['TMIN', 'TOBS', 'TMAX']: #,
    pltVarExplained(j)
    j+=1


f=plt.figure(figsize=(15,4))
j=1
for m in ['SNOW', 'SNWD', 'PRCP']:
    pltVarExplained(j)
    j+=1 


from scipy import stats
def return_temp_data(measurement):

    Query_temp = "SELECT * FROM weather\n\tWHERE measurement='%s' and station='%s'"%('TOBS','USC00243013')
    print Query_temp
    df_T = sqlContext.sql(Query_temp)
    print df_T.count(),'rows'
    df_T.show(2)
    rows=df_T.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()
    T_max = np.vstack(rows)
    return T_max

def plot_winter(T,measurement):
    winter_temp = []
    for temp in T:
        winter_temp.append(np.nanmean(temp))
    xi = np.arange(0,len(winter_temp))
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi,winter_temp)
    line = slope*xi+intercept
    print "slope is :"+str(slope)
    plt.plot(xi,winter_temp,'o', xi, line)
    plt.title("Mean "+measurement+" statistics for past "+str(len(xi))+" years")
    plt.ylabel(measurement)
    plt.xlabel("years")

T_min = return_temp_data('TOBS')
plot_winter(T_min,'TOBS')

T_min = return_temp_data('SNOW')
plot_winter(T_min,'SNOW')


#sc.stop()


#setup
data_dir='../../Data/Weather'
file_index='BSBSSSSS'
meas='PRCP'


# # Reconstruction using top eigen-vectors
# For measurement = {{meas}}
# 

# ## Load the required libraries
# 

# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets


from pyspark import SparkContext
#sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)


# ## Read Statistics File
# 

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements


# ## Read data file into a spark DataFrame
# We focus on the snow-depth records, because the eigen-vectors for them make sense.
# 

#read data
filename=data_dir+'/decon_%s_%s.parquet'%(file_index,meas)
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==meas)
df.show(5)


# ### Plot Mean and Eigenvecs
# 

m=meas
fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=3
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])


# ### plot the percent of residual variance on average
# 

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');


# ### How well-explained are the vectors in this collection?
# 
# To answer this question we extract all of the values of `res_3` which is the residual variance after the Mean and the 
# first two Eigen-vectors have been subtracted out. We rely here on the fact that `df3` is already sorted according to `res_3`
# 

# A function for plotting the CDF of a given feature
def plot_CDF(df,feat):
    rows=df.select(feat).sort(feat).collect()
    vals=[r[feat] for r in rows]
    P=np.arange(0,1,1./(len(vals)))
    while len(vals)< len(P):
        vals=[vals[0]]+vals
    plot(vals,P)
    title('cumulative distribution of '+feat)
    ylabel('fraction of instances')
    xlabel(feat)
    grid()
    


plot_CDF(df,'res_3')


rows=df.rdd.map(lambda row:(row.station,row.year,unpackArray(row['vector'],np.float16))).collect()
rows[0][:2]


days=set([r[1] for r in rows])
miny=min(days)
maxy=max(days)
record_len=int((maxy-miny+1)*365)
record_len


## combine the measurements for each station into a single long array with an entry for each day of each day
All={}  # a dictionary with a numpy array for each day of each day
i=0
for station,day,vector in rows:
    i+=1; 
    # if i%1000==0: print i,len(All)
    if not station in All:
        a=np.zeros(record_len)
        a.fill(np.nan)
        All[station]=a
    loc = int((day-miny)*365)
    All[station][loc:loc+365]=vector


from datetime import date
d=datetime.date(int(miny), month=1, day=1)
start=d.toordinal()
dates=[date.fromordinal(i) for i in range(start,start+record_len)]


for station in All:
    print station, np.count_nonzero(~np.isnan(All[station]))


Stations=sorted(All.keys())
A=[]
for station in Stations:
    A.append(All[station])

day_station_table=np.hstack([A])
print shape(day_station_table)


def RMS(Mat):
    return np.sqrt(np.nanmean(Mat**2))

mean_by_day=np.nanmean(day_station_table,axis=0)
mean_by_station=np.nanmean(day_station_table,axis=1)
tbl_minus_day = day_station_table-mean_by_day
tbl_minus_station = (day_station_table.transpose()-mean_by_station).transpose()

print 'total RMS                   = ',RMS(day_station_table)
print 'RMS removing mean-by-station= ',RMS(tbl_minus_station)
print 'RMS removing mean-by-day   = ',RMS(tbl_minus_day)


RT=day_station_table
F=RT.flatten()
NN=F[~np.isnan(F)]

NN.sort()
P=np.arange(0.,1.,1./len(NN))
plot(NN,P)
grid()
title('CDF of daily rainfall')
xlabel('daily rainfall')
ylabel('cumulative probability')


# ### Conclusions
# It is likely to be hard to find correlations between the **amount** of rain on the same day in different stations. Because amounts of rain vary a lot between even close locations. It is more reasonable to try to compare whether or not it rained on the same day in different stations. As we see from the graph above, in our region it rains in about one third of the days.
# 
# ### measuring statistical significance
# We want to find a statistical test for rejecting the null hypothesis that says that the rainfall in the two locations is independent.
# 
# Using the inner product is too noisy, because you multiply the rainfall on the same day in two locations and that product can be very large - leading to a large variance and poor ability to discriminate.
# 
# An alternative is to ignore the amount of rain, and just ask whether it rained in both locations. We can then compute the probability associated with the number of overlaps under the null hypothesis.
# 

# Fix two stations. We restrict our attention to the days for which we have measurements for both stations, and define the following notation:
# * $m$ : the total number of days (for which we have measurements for both stations).
# * $n_1$ : the number of days that it rained on station 1
# * $n_2$ : the number of days that it rained on station 2
# * $l$ : the number of days that it rained on both stations.
# 
# We want to calculate the probability that the number of overlap days is $l$ given $m,n_1,n_2$.
# 
# The answer is:
# $$
# P = {m \choose l,n_1-l,n_2-l,m-n_1-n_2+l} /{m \choose n_1}{m \choose n_2}
# $$
# 
# Where
# $$
# {m \choose l,n_1-l,n_2-l,m-n_1-n_2+l} = \frac{m!}{l! (n_1-l)! (n_2-l)! (m-n_1-n_2+l)!}
# $$
# 
# We use the fact that $\Gamma(n+1) = n!$ and denote $G(n) \doteq \log \Gamma(n+1)$
# $$
# \log P = \left[G(m) - G(l) -G(n_1-l) -G(n_2-l) -G(m-n_1-n_2+l) \right] - 
# \left[G(m)-G(n_1)-G(m-n_1)\right] - \left[G(m)-G(n_2)-G(m-n_2)\right]
# $$
# Which slightly simplifies to 
# $$
# \log P = -G(l) -G(n_1-l) -G(n_2-l) -G(m-n_1-n_2+l) - G(m)+G(n_1)+G(m-n_1) +G(n_2)+G(m-n_2)
# $$
# 
# The log probability scales with $m$ the length of the overlap. So to get a per-day significance we consider $
# \frac{1}{m} \log P $
# 

from scipy.special import gammaln,factorial
#for i in range(10):
#    print exp(gammaln(i+1))-factorial(i)
def G(n):
    return gammaln(n+1)
def LogProb(m,l,n1,n2):
    logP=-G(l)-G(n1-l)-G(n2-l)-G(m-n1-n2+l)-G(m)+G(n1)+G(m-n1)+G(n2)+G(m-n2)
    return logP/m
exp(LogProb(1000,0,500,500))


#USC00193270 21482
#USC00193702 28237
X=copy(All['USC00193270'])
Y=copy(All['USC00193702'])
print sum(~np.isnan(X))
print sum(~np.isnan(Y))
X[np.isnan(Y)]=np.nan
Y[np.isnan(X)]=np.nan
print sum(~np.isnan(X))
print sum(~np.isnan(Y))


def computeLogProb(X,Y):
    X[np.isnan(Y)]=np.nan
    Y[np.isnan(X)]=np.nan
    G=~isnan(X)
    m=sum(G)
    XG=X[G]>0
    YG=Y[G]>0
    n1=sum(XG)
    n2=sum(YG)
    l=sum(XG*YG)
    logprob=LogProb(m,l,n1,n2)
    # print 'm=%d,l=%d,n1=%d,n2=%d,LogPval=%f'%(m,l,n1,n2,logprob)
    return logprob,m
print computeLogProb(X,Y)


# ### calculate the normalized log probability for each pair of stations.
# 

L=len(Stations)
Pvals=np.zeros([L,L])
Length=np.zeros([L,L])
P_norm=np.zeros([L,L])
for i in range(L):
    print i,
    for j in range(L):
        if i==j: 
            P_norm[i,j]=-0.4
            continue
        X=copy(All[Stations[i]])
        Y=copy(All[Stations[j]])
        P_norm[i,j],Length[i,j]=computeLogProb(X,Y)
        if Length[i,j]<200:
            P_norm[i,j]=np.nan

            


print Pvals[:2,:2]
print Length[:2,:2]
print P_norm[:2,:2]


A=P_norm.flatten();
B=A[~isnan(A)]
print A.shape,B.shape
hist(-B,bins=100);
xlabel('significance')


def showmat(mat):
    fig,axes=plt.subplots(1,1,figsize=(10,10))
    axes.imshow(mat, cmap=plt.cm.gray)


showmat(P_norm)


# ### Finding structure in the rependency matrix.
# The matrix above shows, for each pair of stations, the normalized log probability that the overlap in rain days is random.
# 
# We see immediately the first 8 stations are highly correlatedwith each other. 
# 
# To find more correlations we use SVD (the term PCA is reserved for decomposition of the covariance matrix). As we shall see that the top 10 eigenvectors explain about 80% of the square magnitude of the matrix.
# 

print 'A group of very correlated stations is:',All.keys()[:8]


from sklearn.decomposition import PCA
P_norm0 = np.nan_to_num(P_norm)
n_comp=10
pca = PCA(n_components=n_comp, svd_solver='full')
pca.fit(P_norm0)     
#print(pca.explained_variance_)
Var_explained=pca.explained_variance_ratio_
plot(insert(cumsum(Var_explained),0,0))
grid()


# we will look only at the top 4 eigenvectors.
n_comp=4
pca = PCA(n_components=n_comp, svd_solver='full')
pca.fit(P_norm0)     


fig,axes=plt.subplots(1,4,figsize=(20,5),sharey='row');
L=list(pca.components_.transpose())
for i in range(4):
    X=sorted(L,key=lambda x:x[i]) 
    axes[i].plot(X);


def re_order_matrix(M,order):
    M_reord=M[order,:]
    M_reord=M_reord[:,order]
    return M_reord


fig,axes=plt.subplots(2,2,figsize=(15,15),sharex='col',sharey='row');
i=0
for r in range(2):
    for c in range(2):
        order=np.argsort(pca.components_[i,:])
        P_norm_reord=re_order_matrix(P_norm0,order)
        axes[r,c].matshow(P_norm_reord)
        i+=1


# ### Explanation and possibe extensions
# When we reorder the rows and columns of the matrix using one of the eigenvectors, the grouping of the 
# stations becomes more evident. For example, consider the upper left corner of the scond matrix (The upper left one). The stations at positions 0-22 are clearly strongly correlated with each other. Even though there are some stations, in positions 15-18 or so, which are more related to each other than to the rest of this block.
# 
# This type of organization is called **Block Diagonal** and it typically reveals important structure such as grouping or clustering.
# 
# You might want to extract the sets of stations that form blocks for your region, and then plot them on the map to see their spatial relationship.
# 

from pickle import dump
with open(data_dir+'/PRCP_residuals_PCA.pickle','wb') as file:
    dump({'stations':All.keys(),
          'eigen-vecs':pca.components_},
        file)
    


#setup
data_dir='../../Data/Weather'
file_index='SBBBSBSS'
meas='SNWD'


# # Reconstruction using top eigen-vectors
# For measurement = {{meas}}
# 

# ## Load the required libraries
# 

# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets


from pyspark import SparkContext
#sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)


# ## Read Statistics File
# 

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements


# ## Read data file into a spark DataFrame
# We focus on the snow-depth records, because the eigen-vectors for them make sense.
# 

#read data
filename=data_dir+'/decon_%s_%s.parquet'%(file_index,meas)
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==meas)
df.show(5)


# ### Plot Mean and Eigenvecs
# 

m=meas
fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=3
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])


# ### plot the percent of residual variance on average
# 

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');


# ### How well-explained are the vectors in this collection?
# 
# To answer this question we extract all of the values of `res_3` which is the residual variance after the Mean and the 
# first two Eigen-vectors have been subtracted out. We rely here on the fact that `df3` is already sorted according to `res_3`
# 

# A function for plotting the CDF of a given feature
def plot_CDF(df,feat):
    rows=df.select(feat).sort(feat).collect()
    vals=[r[feat] for r in rows]
    P=np.arange(0,1,1./(len(vals)))
    while len(vals)< len(P):
        vals=[vals[0]]+vals
    plot(vals,P)
    title('cumulative distribution of '+feat)
    ylabel('fraction of instances')
    xlabel(feat)
    grid()
    


plot_CDF(df,'res_3')


rows=df.rdd.map(lambda row:(row.station,row.year,unpackArray(row['vector'],np.float16))).collect()
rows[0][:2]


days=set([r[1] for r in rows])
miny=min(days)
maxy=max(days)
record_len=int((maxy-miny+1)*365)
record_len


## combine the measurements for each station into a single long array with an entry for each day of each day
All={}  # a dictionary with a numpy array for each day of each day
i=0
for station,day,vector in rows:
    i+=1; 
    # if i%1000==0: print i,len(All)
    if not station in All:
        a=np.zeros(record_len)
        a.fill(np.nan)
        All[station]=a
    loc = int((day-miny)*365)
    All[station][loc:loc+365]=vector


from datetime import date
d=datetime.date(int(miny), month=1, day=1)
start=d.toordinal()
dates=[date.fromordinal(i) for i in range(start,start+record_len)]


for station in All:
    print station, np.count_nonzero(~np.isnan(All[station]))


Stations=sorted(All.keys())
A=[]
for station in Stations:
    A.append(All[station])

day_station_table=np.hstack([A])
print shape(day_station_table)


def RMS(Mat):
    return np.sqrt(np.nanmean(Mat**2))

mean_by_day=np.nanmean(day_station_table,axis=0)
mean_by_station=np.nanmean(day_station_table,axis=1)
tbl_minus_day = day_station_table-mean_by_day
tbl_minus_station = (day_station_table.transpose()-mean_by_station).transpose()

print 'total RMS                   = ',RMS(day_station_table)
print 'RMS removing mean-by-station= ',RMS(tbl_minus_station)
print 'RMS removing mean-by-day   = ',RMS(tbl_minus_day)


RT=day_station_table
F=RT.flatten()
print shape(F)
NN=F[~np.isnan(F)]
print shape(NN)

NN.sort()
P=np.arange(0.,1.,1./len(NN))
plot(NN,P[:len(P)-1])
grid()
title('CDF of daily snowdepth')
xlabel('daily snowdepth')
ylabel('cumulative probability')


# ### Conclusions
# It is likely to be hard to find correlations between the **amount** of rain on the same day in different stations. Because amounts of rain vary a lot between even close locations. It is more reasonable to try to compare whether or not it rained on the same day in different stations. As we see from the graph above, in our region it rains in about one third of the days.
# 
# ### measuring statistical significance
# We want to find a statistical test for rejecting the null hypothesis that says that the rainfall in the two locations is independent.
# 
# Using the inner product is too noisy, because you multiply the rainfall on the same day in two locations and that product can be very large - leading to a large variance and poor ability to discriminate.
# 
# An alternative is to ignore the amount of rain, and just ask whether it rained in both locations. We can then compute the probability associated with the number of overlaps under the null hypothesis.
# 

# Fix two stations. We restrict our attention to the days for which we have measurements for both stations, and define the following notation:
# * $m$ : the total number of days (for which we have measurements for both stations).
# * $n_1$ : the number of days that it rained on station 1
# * $n_2$ : the number of days that it rained on station 2
# * $l$ : the number of days that it rained on both stations.
# 
# We want to calculate the probability that the number of overlap days is $l$ given $m,n_1,n_2$.
# 
# The answer is:
# $$
# P = {m \choose l,n_1-l,n_2-l,m-n_1-n_2+l} /{m \choose n_1}{m \choose n_2}
# $$
# 
# Where
# $$
# {m \choose l,n_1-l,n_2-l,m-n_1-n_2+l} = \frac{m!}{l! (n_1-l)! (n_2-l)! (m-n_1-n_2+l)!}
# $$
# 
# We use the fact that $\Gamma(n+1) = n!$ and denote $G(n) \doteq \log \Gamma(n+1)$
# $$
# \log P = \left[G(m) - G(l) -G(n_1-l) -G(n_2-l) -G(m-n_1-n_2+l) \right] - 
# \left[G(m)-G(n_1)-G(m-n_1)\right] - \left[G(m)-G(n_2)-G(m-n_2)\right]
# $$
# Which slightly simplifies to 
# $$
# \log P = -G(l) -G(n_1-l) -G(n_2-l) -G(m-n_1-n_2+l) - G(m)+G(n_1)+G(m-n_1) +G(n_2)+G(m-n_2)
# $$
# 
# The log probability scales with $m$ the length of the overlap. So to get a per-day significance we consider $
# \frac{1}{m} \log P $
# 

from scipy.special import gammaln,factorial
#for i in range(10):
#    print exp(gammaln(i+1))-factorial(i)
def G(n):
    return gammaln(n+1)
def LogProb(m,l,n1,n2):
    logP=-G(l)-G(n1-l)-G(n2-l)-G(m-n1-n2+l)-G(m)+G(n1)+G(m-n1)+G(n2)+G(m-n2)
    return logP/m
exp(LogProb(1000,0,500,500))


#USC00244386 10255
#USC00241231 6302
X=copy(All['USC00244386'])
Y=copy(All['USC00241231'])
print sum(~np.isnan(X))
print sum(~np.isnan(Y))
X[np.isnan(Y)]=np.nan
Y[np.isnan(X)]=np.nan
print sum(~np.isnan(X))
print sum(~np.isnan(Y))


def computeLogProb(X,Y):
    X[np.isnan(Y)]=np.nan
    Y[np.isnan(X)]=np.nan
    G=~isnan(X)
    m=sum(G)
    XG=X[G]>0
    YG=Y[G]>0
    n1=sum(XG)
    n2=sum(YG)
    l=sum(XG*YG)
    logprob=LogProb(m,l,n1,n2)
    # print 'm=%d,l=%d,n1=%d,n2=%d,LogPval=%f'%(m,l,n1,n2,logprob)
    return logprob,m
print computeLogProb(X,Y)


# ### calculate the normalized log probability for each pair of stations.
# 

L=len(Stations)
Pvals=np.zeros([L,L])
Length=np.zeros([L,L])
P_norm=np.zeros([L,L])
for i in range(L):
    print i,
    for j in range(L):
        if i==j: 
            P_norm[i,j]=-0.4
            continue
        X=copy(All[Stations[i]])
        Y=copy(All[Stations[j]])
        P_norm[i,j],Length[i,j]=computeLogProb(X,Y)
        if Length[i,j]<200:
            P_norm[i,j]=np.nan

            


print Pvals[:2,:2]
print Length[:2,:2]
print P_norm[:2,:2]


A=P_norm.flatten();
B=A[~isnan(A)]
print A.shape,B.shape
hist(-B,bins=100);
xlabel('significance')


def showmat(mat):
    fig,axes=plt.subplots(1,1,figsize=(10,10))
    axes.imshow(mat, cmap=plt.cm.gray)


showmat(P_norm)


# ### Finding structure in the rependency matrix.
# The matrix above shows, for each pair of stations, the normalized log probability that the overlap in rain days is random.
# 
# We see immediately the first 8 stations are highly correlatedwith each other. 
# 
# To find more correlations we use SVD (the term PCA is reserved for decomposition of the covariance matrix). As we shall see that the top 10 eigenvectors explain about 80% of the square magnitude of the matrix.
# 

print 'A group of very correlated stations is:',All.keys()[:8]


from sklearn.decomposition import PCA
P_norm0 = np.nan_to_num(P_norm)
n_comp=10
pca = PCA(n_components=n_comp, svd_solver='full')
pca.fit(P_norm0)     
#print(pca.explained_variance_)
Var_explained=pca.explained_variance_ratio_
plot(insert(cumsum(Var_explained),0,0))
grid()


# we will look only at the top 4 eigenvectors.
n_comp=4
pca = PCA(n_components=n_comp, svd_solver='full')
pca.fit(P_norm0)     


fig,axes=plt.subplots(1,4,figsize=(20,5),sharey='row');
L=list(pca.components_.transpose())
for i in range(4):
    X=sorted(L,key=lambda x:x[i]) 
    axes[i].plot(X);


def re_order_matrix(M,order):
    M_reord=M[order,:]
    M_reord=M_reord[:,order]
    return M_reord


fig,axes=plt.subplots(2,2,figsize=(10,10),sharex='col',sharey='row');
i=0
station_grp = []
for r in range(2):
    for c in range(2):
        order=np.argsort(pca.components_[i,:])
        P_norm_reord=re_order_matrix(P_norm0,order)
        station_grp.append(np.array(Stations)[order])
        axes[r,c].matshow(P_norm_reord)
        i+=1


# ### Explanation and possibe extensions
# When we reorder the rows and columns of the matrix using one of the eigenvectors, the grouping of the 
# stations becomes more evident. For example, consider the upper left corner of the scond matrix (The upper left one). The stations at positions 0-22 are clearly strongly correlated with each other. Even though there are some stations, in positions 15-18 or so, which are more related to each other than to the rest of this block.
# 
# This type of organization is called **Block Diagonal** and it typically reveals important structure such as grouping or clustering.
# 
# You might want to extract the sets of stations that form blocks for your region, and then plot them on the map to see their spatial relationship.
# 

from pickle import dump
with open(data_dir+'/PRCP_residuals_PCA.pickle','wb') as file:
    dump({'stations':All.keys(),
          'eigen-vecs':pca.components_},
        file)
    


import gmplot

sqlContext.registerDataFrameAsTable(df,'weather')

Query = "SELECT latitude,longitude,station FROM weather"
df1 = sqlContext.sql(Query)
x = df1.collect()

dict_x = {}
for latitude,longitude,station in x:
    dict_x[station] = (latitude,longitude)
grouped_location = []
for station in station_grp[2]:
    grouped_location.append(dict_x[station])
    
latitude = [x[0] for x in grouped_location[:13]]
longitude = [x[1] for x in grouped_location[:13]]
gmap = gmplot.GoogleMapPlotter(latitude[0],longitude[0],16)
gmap.scatter(latitude, longitude, 'b', marker=True)
gmap.draw("mycorrelation.html")





# # California Weather Analysis
# 
# This is a report on the historical analysis of weather patterns in an area that approximately overlaps the Central area of California.
# 
# The data we will use here comes from [NOAA](https://www.ncdc.noaa.gov/). Specifically, it was downloaded from This [FTP site](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/).
# 
# We focused on six measurements:
# * **TMIN, TMAX:** the daily minimum and maximum temperature.
# * **TOBS:** The average temperature for each day.
# * **PRCP:** Daily Percipitation (in mm)
# * **SNOW:** Daily snowfall (in mm)
# * **SNWD:** The depth of accumulated snow.
# 
# ## 1. Span of Weather Stations
# 
# By plotting the latitude,longitude of the weather stations in the dataset, it was found that all the weather stations belong to the coastal side of central California.   
# <p><img style='height:300px' src="myfigs/location_map.png" /></p>
# 
# 
# ### 1.1 Sanity-check: comparison with outside sources
# 
# <p>As a quick sanity check, I picked one of the weather stations from my dataset at Oakley, california and got it's weather statistics from the <a href='http://www.usclimatedata.com/climate/oakley/california/united-states/usca2070'>US climate data</a> website.The graph below shows the daily minimum and maximum temperatures for each month, as well as the total precipitation for each month.</p>
# 
# <p>&nbsp;</p>
# 
# <p><img style='height:300px' src="myfigs/sanity_check_oakley.png" /></p>
# 
# <p>&nbsp;</p>
# 
# <p>We see that the min and max daily&nbsp;temperature approximately agrees with the pattern we got from our data(shown in the below figure). Same is the case with the precipitation data, the mean precipitation is approximately similar to the data obtained from US climate data website. A point to be noted here is that the statistics obtained from US climate data is only pertaining to a single weather station while the mean precipitation shown in our graph is an average over the entire Central California region.</p>
# 
# <p>&nbsp;</p>
# 
# <p><img alt="TMIN,TMAX.png" src="myfigs/sanity_check_oakley2.png" style="height:300px;" /></p>
# 
# <p>&nbsp;<img alt="PRCP.png" src="myfigs/sanity_check_oakley3.png" style="height:300px;" /></p>
# 

# ## 2. PCA analysis
# 
# For each of the six measurement, we compute the percentate of the variance explained as a function of the number of eigen-vectors used.
# 
# ### Percentage of variance explained.
# ![VarExplained1.png](myfigs/pca1.png)
# We see that the top 5 eigen-vectors explain 38% of variance for TMIN, 59% for TOBS and 47% for TMAX.
# 
# We conclude that of the three, TOBS is best explained by the top 5 eigenvectors. This is especially true for the first eigen-vector which, by itself, explains 48% of the variance. TMAX and TMIN also have a pretty good approximation using the first five eigen vectors since they contribute to a significant share of the total variance. On the whole it can be inferred that the temperature statistics follow a pattern with majority of the variance along a few principal axes.
# Based on this initial promise, we will dig deeper into the PCA analysis of TOBS. However we can expect that the TOBS data is going to be noisy since the temperatures on consecutive days of the year do not follow a strictly increasing or decreasing sequence over a considerably big window. 
# <p></p>
# ![VarExplained2.png](myfigs/pca2.png)
# 
# The top 5 eigenvectors explain only 16% of the variance for PRCP. On the other hand the top 5 eigenvectors explain 72% of the variance for SNWD and 68% for SNOW. This means that these top 5 eigenvectors capture most of the variation in the snow signals. However this is not statistically significant since the majority of the weather stations are located in parts of California where it never really snows. As we can see from the below graphs the mean snow depth is close to zero for major parts of the year apart from minor spikes. Also the top three eigen vectors remain zero almost the whole year except for a few sporadic spikes. This means that any reconstruction using the top few eigen vectors will will get 70% of the data correct simply owing to the fact that the snow is zero always. 
# 
# <p>&nbsp;<img src="myfigs/snwd-stats.png" style="height:300px;" /></p>

# ### 2.1 Analysis of TOBS
# 
# We choose to analyze the eigen-decomposition for TOBS because the first 3 eigen-vectors explain more than 55% of the variance.
# First, we graph the mean and the top 3 eigen-vectors.
# 
# We observe that the average temperatures conform to the seasonal pattern with the temperature being maximum between mid june and end of october which is the Summer season. Likewise the minimum temperature is observed between December and March which is the Winter season.
# <p>&nbsp;<img src="myfigs/tobs-stats.png" style="height:300px;" /></p>
# 

# Next we interpret the eigen-functions. The first eigen-vector has a shape very similar to the mean function. The interpretation of this shape is that eig1 represents the deviation of temperature above/below the mean, but without changing the distribution over time.
# 
# **eig2 and eig3** are similar in the following way. They peak during a certain period of the year. In other words, they correspond to the deviation in temperature distribition between different months.
# 
# They can be interpreted as follows:
# * **eig2:** less temperature in june - october than the rest of the year.
# * **eig3:** more temperature in march - july, than the rest of the year.
# 

# #### 2.1.1 Examples of reconstructions
# 
# In all of the below graphs we find that the reconstruction of the target vector gets more and more accurate with additional eigen vectors. As stated in the earlier section,the average daily temperatures is not a smoothly increasing/decreasing function and the reconstruction from the eigen vectors minimizes the noise in the function.
# 
# #### Coeff1
# Coeff1: small values
# ![c1small.png](myfigs/c1small.png)
# Coeff1: large values
# ![c1large.png](myfigs/c1large.png)
# Large values of coeff1 correspond to more than average temperature and low values correspond to less than average temperature.
# 
# #### Coeff2
# Coeff2: small values
# ![c2small.png](myfigs/c2small.png)
# Coeff2: large values
# ![c2large.png](myfigs/c2large.png)
# 
# Large values of coeff2 correspond to low summer temperatures between june and october. Small values for coeff2 correspond to high summer temperatures.
# 
# #### Coeff3
# Coeff3: small values
# ![c3small.png](myfigs/c3small.png)
# Coeff3: large values
# ![c3large.png](myfigs/c3large.png)
# 
# Large values of coeff3 correspond to a high temperatures during the march-july period of the year and small values of coeff3 correspond to low temperatures during the march-july timeperiod. 
# 

# #### 2.1.2 Cumulative Distribution of residuals and coefficients
# The below graph shows a plot of the cumulative distribution function of the first three residuals. As expected the residual2 is better than residual1 since the first two eigen vectors combined capture more variance than the first eigen vector alone. In other words the residual error from reconstruction using only the first eigen vector grows faster than the residual error from reconstruction using the first two eigen vectors combined. However we can see that there is not much difference in the cumulative residual errors of res_2 and res_3. This behaviour is as expected,conforming with what we saw in the percentage variance explained plot of TOBS, where the increase in the percentage of variance explained between 2 and 3 eigen vectors is very small.<br/>
# ![residuals_tobs.png](myfigs/residuals_tobs.png)
# 
# The below graph shows a plot of the cumulative distribution function of the first three coefficients of the eigen vectors. Since the first eigen vector gives the direction of maximum variation it is natural that many data points are dominated by a large coefficient1 compared to the other coefficients. As we can see there is not much difference between the coefficients 2 and 3, for the same reason as explained above.
# ![residuals_tobs.png](myfigs/coeffs_tobs.png)

# #### 2.1.3 Visualizing data distribution for TOBS
# 
# The below map shows a distribution of the different weather stations that have collected the TOBS measurement in the central California region. The size of the circles in the map is proportional to the number of datapoints contributed by a particular weather station. The fill color of the circles is a heatmap denoting the average value of coefficient1 which is the deviation of temperature from the mean. A red weather station indicates a high average temperature while a blue weather station indicates a relatively lower average temperature. 
# 
# ![residuals_tobs.png](myfigs/5-map.png)

# ## 3. Analysis of Precipitation
# 
# There is an average rainfall of 13.31 mm/day in the given region. As shown in the below graph, most of the rain occurs during the period of November to February. The CDF plot shows that it rains for about 20% of the days in our region. The first eigen vector represents the deviation in rainfall from mean. The second and third eigen vectors represent seasonal rain. 
# <p>&nbsp;<img src="myfigs/prcp-stats.png" style="height:300px;float:left;" /><img src="myfigs/cdf-rain.png" style="height:258px;" /></p>
# 
# Since the weather stations are all close to one another there is good chance that precipitation in one station guarantees precipitation in a nearby station. To accept/reject our hypothesis we begin with plotting a correlation matrix of Log probabilities where each of the values represent the probability of a coincidental rain in two weather stations. 
# <p>&nbsp;<img src="myfigs/7-correlation.png" style="height:400px;" /></p>
# 
# We can see from the above graph that the first 30 weather stations are correlated. To find more correlations we use PCA of this correlation matrix and cluster the weather stations based on the first few principal components. As shown in the below graph the top 10 eigen vectors of the correlation matrix explain about 90% of the of the square magnitude of the matrix.
# 
# <p>&nbsp;<img src="myfigs/7-correlation-decomp.png" style="height:300px;" /></p>
# 
# For the purpose of clustering, we consider only the first four principal components. We sort the weather stations in the correlation matrix according to increasing order of the magnitude of dimensions of the first eigen vector. The resultant correlation matrix visualized as a heatmap shows clusters of weather stations that are correlated based on the first eigen vector. We repeat this process for the 2nd,3rd and 4th eigen vectors. Below are the new heatmaps obtained after sorting. From the upper left heatmap we can see that the first 40 stations are correlated and especially the first 20 are highly correlated.
# 
# <p>&nbsp;<img src="myfigs/7-heatmap.png" style="height:700px;" /></p>
# 
# Plotting the first few correlated weather stations on a geo map, we get the below plot. From our analysis, it is evident that the weather stations that are nearby have a good chance of experiencing rain on the same day of the year. 
# 
# <p>&nbsp;<img src="myfigs/7-heatmap-geo.png" style="height:300px;" /></p>
# 
# An alernate method of visualizing the data is to plot the weather stations along with the coefficients of the first principal components. The below geo plot indicates the principal components by triangles with the size of the triangle representing the magnitude of the coefficient and the opacity of the triangle representing the sign of the coefficient (filled triangles for negative and un-filled triangles for positive coefficients). As we can see there are a few close weather stations with similar triangle structures,that is, similar principal components. Both the visualizations(the above map and the below map) conform that the weather stations near the region of Concord and Pleasant Hill are correlated and experience precipitaion on the same days of the year.
# 
# <p>&nbsp;<img src="myfigs/5.5-map.png" style="height:300px;" /></p>
# 

# ## 4. Temporal Vs Spatial Analysis of Precipitation
# 
# In the previous section we see the variation of Coeff1, which corresponds to the total amount of rain, with respect to location. We now estimate the relative importance of location-to-location variation relative to year-by-year variation.
# These are measured using the fraction by which the variance is reduced when we subtract from each station/year entry the average-per-year or the average-per-station respectively. Here are the results:
# 
# coeff_1<br/> 
# total RMS                   =  194.754604183<br/>
# RMS removing mean-by-station=  173.891026199 Fraction explained= 10.71%<br/>
# RMS removing mean-by-year   =  120.264234979 Fraction explained= 38.24%<br/>
# 
# coeff_2 <br/>
# total RMS                   =  180.793723228<br/>
# RMS removing mean-by-station=  172.563345122 Fraction explained= 4.55%<br/>
# RMS removing mean-by-year   =  80.9796786501 Fraction explained= 55.20%<br/>
# 
# coeff_3<br/> 
# total RMS                   =  171.693528795<br/>
# RMS removing mean-by-station=  167.550306474 Fraction explained= 2.41%<br/>
# RMS removing mean-by-year   =  70.5968252719 Fraction explained= 58.88%<br/>
# 
# We see that the variation by year explains more than the variation by station. However this effect is weaker consider coeff_1, which has to do with the total rainfall, vs. coeff_2,3 which, as we saw above have to do with the timining of rainfall. We see that for coeff_2,3 the stations explain 2-5% of the variance while the year explaines 55-60%.
# 




