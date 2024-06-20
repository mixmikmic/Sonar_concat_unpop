# This will be an example notebook showing exploratory regression analysis with a simple, point-based hedonic house price model for Baltimore
# 

import pysal as ps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('talk')
get_ipython().magic('matplotlib inline')


ps.examples.available()


ps.examples.explain('baltim')


data = ps.pdio.read_files(ps.examples.get_path('baltim'))


data.head()


mindist = ps.min_threshold_dist_from_shapefile(ps.examples.get_path('baltim.shp'))
mindist


W = ps.threshold_binaryW_from_array(np.array([data.X.values, data.Y.values]).T, 2*mindist)


W = ps.W(W.neighbors, W.weights)
W.transform = 'r'


ycols = ['PRICE']
xcols = ['NROOM', 'DWELL', 'LOTSZ', 'SQFT']#, 'AGE']#, 'NBATH', 'PATIO', 'FIREPL', 'AC', 'BMENT', 'NSTOR', 'GAR', ]
y = data[ycols].values
X = data[xcols].values


ols_reg = ps.spreg.OLS(y, X, w=W, spat_diag=True, moran=True, name_y=ycols, 
                       name_x = xcols)


print(ols_reg.summary)


effects, errs = ols_reg.betas, ols_reg.std_err


#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,5, -12,30])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 5, linestyle='--', color='k')


resids = y - ols_reg.predy


Mresids = ps.Moran(resids.flatten(), W)


fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.2,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()


ml_lag = ps.spreg.ML_Lag(y, X, w=W)#, name_y=ycols, name_x = xcols)
effects, errs = ml_lag.betas, ml_lag.std_err


print(ml_lag.summary)


plt.title('Regression Effects plot')
plt.axis([-1,5, -38,20])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k')


resids = y - ml_lag.predy
Mresids = ps.Moran(resids.flatten(), W)


fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))



ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

XtXi = np.linalg.inv(np.dot(X.T, X))
H = np.dot(X, XtXi)
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Tangental Leverage')
ax[2].set_ylabel('Residuals')
ax[2].axis([-.01,.2,-60,120])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Tangental Leverage')


plt.show()


xcols.append('AGE')


X = data[xcols].values


reg_ommit = ps.spreg.OLS(y,X, name_y = ycols, name_x = xcols)
effects, errs = reg_ommit.betas, reg_ommit.std_err
print(reg_ommit.summary)


#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,6, -5,28])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k')


resids = y - reg_ommit.predy
Mresids = ps.Moran(resids.flatten(), W)


fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))



ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()


xcols.extend(['NBATH', 'PATIO', 'FIREPL', 'AC', 'BMENT', 'NSTOR', 'GAR', ])
X = data[xcols].values
reg_ommit = ps.spreg.OLS(y,X, name_y = ycols, name_x = xcols)
effects, errs = reg_ommit.betas, reg_ommit.std_err
resids = y - reg_ommit.predy
print(reg_ommit.summary)


plt.title('Regression Effects plot')
plt.axis([-1,13, -12,35])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k', linewidth=.9)


Mresids = ps.Moran(resids, W)


fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')


H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()


reg_ommit = ps.spreg.ML_Lag(y,X, w=W)
effects, errs = reg_ommit.betas, reg_ommit.std_err
resids = y - reg_ommit.predy
print(reg_ommit.summary)


#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,14, -10,20])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten(), fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 14, linestyle='--', color='k')


Mresids = ps.Moran(resids, W)


fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')


H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Tangental Leverage')
ax[2].set_ylabel('Residuals')

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()





import pysal as ps
import pandas as pd
import numpy as np


# A well-used functionality in PySAL is the use of PySAL to conduct exploratory spatial data analysis. This notebook will provide an overview of ways to conduct exploratory spatial analysis in Python. 
# 

# First, let's read in some data:
# 

data = ps.pdio.read_files(ps.examples.get_path('NAT.shp'))
W = ps.queen_from_shapefile(ps.examples.get_path('NAT.shp'))
W.transform = 'r'


data.head()


# In PySAL, commonly-used analysis methods are very easy to access. For example, if we were interested in examining the spatial dependence in `HR90` we could quickly compute a Moran's $I$ statistic:
# 

I_HR90 = ps.Moran(data.HR90.values, W)


I_HR90.I, I_HR90.p_sim


# Thus, the $I$ statistic is $.383$ for this data, and has a very small $p$ value. 
# 

# We can visualize the distribution of simulated $I$ statistics using the stored collection of simulated statistics:
# 

I_HR90.sim[0:5]


# A simple way to visualize this distribution is to make a KDEplot (like we've done before), and add a rug showing all of the simulated points, and a vertical line denoting the observed value of the statistic:
# 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


sns.kdeplot(I_HR90.sim, shade=True)
plt.vlines(I_HR90.sim, 0, 1)
plt.vlines(I_HR90.I, 0, 40, 'r')


# Instead, if our $I$ statistic were close to our expected value, `I_HR90.EI`, our plot might look like this:
# 

sns.kdeplot(I_HR90.sim, shade=True)
plt.vlines(I_HR90.sim, 0, 1)
plt.vlines(I_HR90.EI+.01, 0, 40, 'r')


# In addition to univariate Moran's $I$, PySAL provides many other types of autocorrelation statistics:
# 

c_HR90 = ps.Geary(data.HR90.values, W)
#ps.Gamma
#ps.Join_Counts


c_HR90.C, c_HR90.p_sim


# Since the statistic is below one with a significant $p$-value, it indicates the same thing as the Moran's $I$ above, moderate significant global spatial dependence in `HR90`. 
# 

# In addition, we can compute a global Bivariate Moran statistic, which relates an observation to the spatial lag of another observation:
# 

bv_HRBLK = ps.Moran_BV(data.HR90.values, data.BLK90.values, W)


bv_HRBLK.I, bv_HRBLK.p_sim


# ### Local Autocorrelation Statistics
# 

# In addition to the Global autocorrelation statistics, PySAL has many local autocorrelation statistics. Let's compute a local Moran statistic for the same data shown above:
# 

LMo_HR90 = ps.Moran_Local(data.HR90.values, W)


# Now, instead of a single $I$ statistic, we have an *array* of local $I_i$ statistics, stored in the `.Is` attribute, and p-values from the simulation are in `p_sim`. 
# 

LMo_HR90.Is, LMo_HR90.p_sim


# We can adjust the number of permutations used to derive every *pseudo*-$p$ value by passing a different `permutations` argument:
# 

LMo_HR90 = ps.Moran_Local(data.HR90.values, W, permutations=9999)


# In addition to the typical clustermap, a helpful visualization for LISA statistics is a Moran scatterplot with statistically significant LISA values highlighted. 
# 
# This is very simple, if we use the same strategy we used before:
# 
# First, construct the spatial lag of the covariate:
# 

Lag_HR90 = ps.lag_spatial(W, data.HR90.values)
HR90 = data.HR90.values


# Then, we want to plot the statistically-significant LISA values in a different color than the others. To do this, first find all of the statistically significant LISAs. Since the $p$-values are in the same order as the $I_i$ statistics, we can do this in the following way
# 

sigs = HR90[LMo_HR90.p_sim <= .001]
W_sigs = Lag_HR90[LMo_HR90.p_sim <= .001]
insigs = HR90[LMo_HR90.p_sim > .001]
W_insigs = Lag_HR90[LMo_HR90.p_sim > .001]


# Then, since we have a lot of points, we can plot the points with a statistically insignficant LISA value lighter using the `alpha` keyword. In addition, we would like to plot the statistically significant points in a dark red color. 
# 

b,a = np.polyfit(HR90, Lag_HR90, 1)


# Matplotlib has a list of [named colors](http://matplotlib.org/examples/color/named_colors.html) and will interpret colors that are provided in hexadecimal strings:
# 

plt.plot(sigs, W_sigs, '.', color='firebrick')
plt.plot(insigs, W_insigs, '.k', alpha=.2)
 # dashed vert at mean of the last year's PCI
plt.vlines(HR90.mean(), Lag_HR90.min(), Lag_HR90.max(), linestyle='--')
 # dashed horizontal at mean of lagged PCI
plt.hlines(Lag_HR90.mean(), HR90.min(), HR90.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(HR90, a + b*HR90, 'r')
plt.text(s='$I = %.3f$' % I_HR90.I, x=50, y=15, fontsize=18)
plt.title('Moran Scatterplot')
plt.ylabel('Spatial Lag of HR90')
plt.xlabel('HR90')


# We can also make a LISA map of the data. 
# 

# ### Simple exploratory regression
# 

# Sometimes, to check for simple spatial heterogeneity, a fixed effects model can be estimated.  If the heterogeneity has known bounds. First, though, note that `pandas` can build dummy variable matrices from categorical data very quickly:
# 

pd.get_dummies(data.SOUTH) #dummies for south (already binary)


# Where this becomes handy is if you have data you know can be turned into a dummy variable, but is not yet correctly encoded. 
# 
# For example, the same call as above can make a dummy variable matrix to encode state fixed effects using the `STATE_NAME` variable:
# 

pd.get_dummies(data.STATE_NAME) #dummies for state by name


# For now, let's estimate a spatial regimes regression on the south/not-south division. To show how a regimes effects plot may look, let's consider one covariate that is likely related and one that is very likely unrelated to $y$. That is our formal specification for the regression will be:
# 
# $$ y = \beta_{0} + x_{1a}\beta_{1a} + x_{1b}\beta_{1b} + x_{2}\beta_{2} + \epsilon $$
# 
# Where $x_{1a} = 1$ when an observation is not in the south, and zero when it is not. This is a simple spatial fixed effects setup, where each different spatial unit is treated as having a special effect on observations inside of it. 
# 
# In addition, we'll add an unrelated term to show how the regression effects visualization works:
# 

y = data[['HR90']].values
x = data[['BLK90']].values

unrelated_effect = np.random.normal(0,100, size=y.shape[0]).reshape(y.shape)

X = np.hstack((x, unrelated_effect))

regimes = data.SOUTH.values.tolist()


regime_reg = ps.spreg.OLS_Regimes(y, X, regimes)


betas = regime_reg.betas
sebetas = np.sqrt(regime_reg.vm.diagonal())


sebetas


plt.plot(betas,'ok')
plt.axis([-1,6,-1,8])
plt.hlines(0,-1,6, color='k', linestyle='--')
plt.errorbar([0,1,2,3,4,5], betas.flatten(), yerr=sebetas*3, fmt='o', ecolor='r')
plt.xticks([-.5,0.5,1.5,2.5,3.5,4.5, 5.5], ['',
                                       'Not South: Constant',
                                       'Not South: BLK90',
                                       'Not South: Not South',
                                       'South: Constant',
                                       'South: South',
                                       'South: Unrelated',''], rotation='325')
plt.title('Regime Fixed Effects')


