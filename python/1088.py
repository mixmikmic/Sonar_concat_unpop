# ## Variability Depth
# 

# This metric was written by Keaton Bell (keatonb@astro.as.utexas.edu) to assess the survey depth (magnitude) to which variability of a given amplitude can be overwhelmingly detected. This question of *whether* a target is variable is most fundamental to the study of variability with LSST and is prerequisite to the measurements of other variability parameters (e.g., pulsation periods). Analysis of survey data with an approach similar (but admittedly more sophisticated) to that outlined here will uncover the maximum number of variable sources in LSST with reasonably well understood completenesses. The magnitude limits calculated by this metric are particularly important as they represent the depth to which short-period (< ~revisit time), multi-periodic and/or low-amplitude pulsators can be detected in LSST. For the short-period case, or any time that observational epochs are assumed to be randomly distributed in phase, detection of excess photometric scatter over the calibrated photometric error can yield a measure of total pulsational power. This is of particular interest for studying the energetics of pulsating white dwarf stars through the six-filter parameter space of the instability strips.
# 

#import general things
import numpy as np 
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#import lsst things
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric


# This metric relies on the calculated 'fiveSigmaDepth' and the total number of visits in a given filter. The metric uses these values from the OpSIM output, but in demonstrating the derivation of the metric here, I will use some arbitrary example values.
# 

# 5 sigma depth
m5 = np.random.rand(200)*2.+23. #200 observations with 5sigma depths uniformily distributed between mag 23 and 25
N = len(m5) # number of visits
N


# To give an illustrative example of what the source recovery fraction would look like for stars of an arbitrary magnitude, we consider the m=18 case:
# 

mref = 18.


# At the fiveSigmaDepth, m5, the photometric signal-to-noise (S/N) is 5. The fractional standard deviation (sigma) in the measured flux is then 1/(S/N) = 0.2.
# 
# Since photometric S/N scales as sqrt(Flux), and the fractional difference in flux between m5 and mref is equal to 10^(0.4(m5-mref))
# 

sigma = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*mref)) # Flux standard error for star of magnitude mref
print np.mean(sigma)


# We're going to assume for this simulation that the photometric errors are distributed as Gaussians. This isn't such a crazy assumption; since photometric uncertainty is influenced by a large number of factors (seeing, clouds, dust, sky brightness, pixel sensitivity, etc.), the central limit theorem supports that their combined effects approach a Gaussian.
# 
# We will compare the liklihoods that a measured variance of a set of flux observations arises from either purely measurement error of a source of constant brightness versus the combined effect of measurement error and intrinsic variability.
# 
# We assess our ability to recover Normally distributed intrinsic variability with a standard deviation of 1%. This is of order the variability expected from low-amplitude pulsating white dwarfs, and the central limit theorem supports that these multi-periodic pulsators have intrinsic brightness variations that are approximately Normally distributed.
# 

signal = 0.01 # 1% intrinsic standard deviation


# I found an analytic solution for the probablility density functions for noise-only and signal-plus-noise variances to be intractable.  We proceed by directly simulating the measurements, which is not very computationally expensive.
# 

#Let's try directly simulating a bunch of standard deviations, both with and without signal.
numruns = 10000 # simulate numruns realizations of these observations

noiseonlyvar = np.zeros(numruns) # hold the measured noise-only variances 
noiseandsignalvar = np.zeros(numruns) # hold the measured noise-plus-signal variances 
signalonlyvar = np.zeros(numruns) #temporary for testing

#Simulate the measured variances
for i in np.arange(numruns):
    scatter = np.random.randn(N)*sigma # random realization of the Gaussian error distributions
    sig = np.random.randn(N)*signal #random realization of the underlying variable signal
    noiseonlyvar[i] = np.var(scatter) # store the noise-only variance
    signalonlyvar[i] = np.var(sig) # store the signal-only variance
    noiseandsignalvar[i] = np.var(sig+scatter) # store the noise-plus-signal variance
    
#plot the histograms of measured variances:
plt.hist(noiseonlyvar,bins=50,label='noise only',color='blue')
plt.hist(noiseandsignalvar,bins=50,label='signal plus noise',color='green')
plt.hist(signalonlyvar,bins=50,label='signal only',color='black')
plt.xlabel('measured variance')
plt.ylabel('# instances')
plt.xlim(0.00,0.0005)
plt.legend()
plt.show()

#Plot the cumulative density functions
plt.plot(np.sort(noiseonlyvar),np.arange(numruns)/float(numruns),label='noise only',color='blue') # noise only
plt.plot(np.sort(signalonlyvar),np.arange(numruns)/float(numruns),label='signal only',color='black') # signal only
plt.plot(np.sort(noiseandsignalvar),1.-np.arange(numruns)/float(numruns),label='1-(signal+noise)',color='green') # 1-(signal+noise)
plt.xlabel('measured variance')
plt.ylabel('cumulative density')
plt.xlim(0.00,0.0005)
plt.legend()
plt.show()


# We see that the measured variance tends to be higher for stars with underlying signal (as expected!), but there is some magnitude-dependent overlap between the distributions that will lead to confusion.
# 
# This metric employs a classification scheme that makes a cut on measured variability and identifies everything above the cut as intrinsically variable, and everything below as likely noise only.
# 
# Specifically, we will demand that we recover 95% of the variable sources (sample is 95% complete).  We will then calculate the magnitude where only 5% of the noise-only sources contaminate this sample.  The actual false alarm probability depends on the relative numbers of variable and non-variable sources, and this measurement is far more complicated in reality where there is a continuum of allowed intrinsic flux variances and probability density functions.  However, more highly variable sources will generally be recovered more easily and less variable sources will be recovered less often, and I think this scheme characterizes the magnitude limit for detecting new low-amplitude variables fairly accurately.
# 

completeness = .95 # fraction of variable sources recovered
contamination = .05 # fraction of non-variable sources that are misclassified


# To find this magnitude limit, we simulate the overlap between the variance distributions for numerous test magnitudes between the LSST bright limit (m=16) and the single visit depth (m~24.5, as determined by the calculated m5 depth).  We then interpolate to approximate where the completeness and contamination limits coincide.
# 

#%%timeit #This is the computationally expensive part, but hopefully not too bad.
#1 loops, best of 3: <300 ms per loop

mag = np.arange(16,np.mean(m5),0.5) #magnitudes to be sampled
res = np.zeros(mag.shape) #hold the distance between the completeness and contamination goals.

noiseonlyvar = np.zeros(numruns) # hold the measured noise-only variances 

#Calculate the variance at a reference magnitude and scale from that
m0=20.
sigmaref = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*m0))

#run the simulations
#Simulate the measured noise-only variances at a reference magnitude
for i in np.arange(numruns):
    scatter = np.random.randn(N)*sigmaref # random realization of the Gaussian error distributions
    noiseonlyvar[i] = np.var(scatter) # store the noise-only variance

#Since we are treating the underlying signal being representable by a fixed-width gaussian,
#its variance pdf is a Chi-squared distribution with the degrees of freedom = visits.
#Since variances add, the variance pdfs convolve.

#We'll use the cdf of the noise-only variances because it's easier to interpolate
noisesorted = np.sort(noiseonlyvar)
interpnoisecdf = UnivariateSpline(noisesorted,np.arange(numruns)/float(numruns),k=1,s=0) #linear

#We need a binned, signal-only variance probability distribution function for numerical convolution
numsignalsamples = 50
xsig = np.linspace(chi2.ppf(0.001, N),chi2.ppf(0.999, N),numsignalsamples)
signalpdf = chi2.pdf(xsig, N)
#correct x to the proper variance scale
xsig = (signal**2.)*xsig/N
pdfstepsize = xsig[1]-xsig[0]
#Since everything is going to use this stepsize down the line,
#normalize so the pdf integrates to 1 when summed (no factor of stepsize needed)
signalpdf /= np.sum(signalpdf)

#run through the sample magnitudes, calculate distance between cont and comp thresholds
for i,mref in enumerate(mag): #i counts and mref is the currently sampled magnitude
    #Scale factor from m0
    scalefact = 10.**(0.4*(mref-m0))
    
    #Calculate the desired contamination threshold
    contthresh = np.percentile(noiseonlyvar,100.-100.*contamination)*scalefact
    
    #Realize the noise CDF at the required stepsize
    xnoise = np.arange(noisesorted[0]*scalefact,noisesorted[-1]*scalefact,pdfstepsize)
    noisecdf = interpnoisecdf(xnoise/scalefact)
    noisepdf = (noisecdf[1:]-noisecdf[:-1]) #turn into a noise pdf
    noisepdf /= np.sum(noisepdf)
    xnoise = (xnoise[1:]+xnoise[:-1])/2. #from cdf to pdf conversion
    
    #calculate and plot the convolution = signal+noise variance dist.
    convolution=0
    if len(noisepdf) > len(signalpdf):
        convolution = np.convolve(noisepdf,signalpdf)
    else: 
        convolution = np.convolve(signalpdf,noisepdf)
    xconvolved = xsig[0]+xnoise[0]+np.arange(len(convolution))*pdfstepsize
    
    #calculate the completeness threshold
    combinedcdf = np.cumsum(convolution)
    findcompthresh = interp1d(combinedcdf,xconvolved)
    compthresh = findcompthresh(1.-completeness)
    
    #Plot the pdfs for demonstration purposes
    plt.plot(xsig, signalpdf, label="signal",c='b')
    plt.plot(xnoise,noisepdf,  label="noise",c='r')
    plt.plot([contthresh,contthresh],[0,noisepdf[np.argmin(np.abs(xnoise-contthresh))]],
             'r--',label='cont thresh')
    plt.plot(xconvolved,convolution, 
             'g-',label="sig+noise")
    plt.plot([compthresh,compthresh],[0,convolution[np.argmin(np.abs(xconvolved-compthresh))]],
             'g--',label='comp thresh')
    plt.xlabel('variance')
    plt.ylabel('pdf')
    plt.title('mag = '+str(mref))
    plt.legend()
    plt.show()

    res[i] = compthresh - contthresh


#Plot the results:
plt.scatter(mag,res)
#Interpolate with a cubic spline
f1 = UnivariateSpline(mag,res,k=1,s=0)
#Find the closest approach to zero to the desired resolution
magres = 0.01 #magnitude resolution
magsamples = np.arange(16,np.mean(m5),magres) #sample the magnitude range at this resolution
plt.plot(magsamples,f1(magsamples)) #Plot the interpolated values
plt.xlabel('magnitude')
plt.ylabel('completeness - contamination')
plt.ylim(-0.0005,0.0001)

#Find the closest approach to zero
vardepth = magsamples[np.argmin(np.abs(f1(magsamples)))]
plt.plot([16,np.mean(m5)],[0,0],'--',color='red') #plot zero
#that's the final result
print vardepth
plt.plot([vardepth,vardepth],[-1,1],'--',color='red') #Plot resulting variability depth

plt.show()


# So, for this made-up example pointing, we simulate a "variability depth" of ~18.4 mags. Now it's time to write this as a proper metric.
# 

#Calculate the survey depth there a variable star can be reliably identified through a comparsion
#of the measured variance to the measurement uncertainty.

class VarDepth(BaseMetric):
    def __init__(self, m5Col = 'fiveSigmaDepth', 
                 metricName='variability depth', 
                 completeness = .95, contamination = .05, 
                 numruns = 10000, signal = 0.01,
                 magres = 0.01, **kwargs):
        """
        Instantiate metric.
        
        :m5col: the column name of the individual visit m5 data.
        :completeness: fractional desired completeness of recovered variable sample.
        :contamination: fractional allowed incompleteness of recovered nonvariables.
        :numruns: number of simulated realizations of noise (most computationally espensive part).
        :signal: sqrt total pulsational power meant to be recovered.
        :magres: desired resolution of variability depth result."""
        self.m5col = m5Col
        self.completeness = completeness
        self.contamination = contamination
        self.numruns = numruns 
        self.signal = signal
        self.magres = magres
        super(VarDepth, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        #Get the visit information
        m5 = dataSlice[self.m5col]
        #Number of visits
        N = len(m5)
        
        #magnitudes to be sampled
        mag = np.arange(16,np.mean(m5),0.5) 
        #hold the distance between the completeness and contamination goals.
        res = np.zeros(mag.shape) 
        #make them nans for now
        res[:] = np.nan 

        #hold the measured noise-only variances 
        noiseonlyvar = np.zeros(self.numruns)

        #Calculate the variance at a reference magnitude and scale from that
        m0=20.
        sigmaref = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*m0))

        #run the simulations
        #Simulate the measured noise-only variances at a reference magnitude
        for i in np.arange(self.numruns):
            # random realization of the Gaussian error distributions
            scatter = np.random.randn(N)*sigmaref 
            noiseonlyvar[i] = np.var(scatter) # store the noise-only variance
            
        #Since we are treating the underlying signal being representable by a 
        #fixed-width gaussian, its variance pdf is a Chi-squared distribution 
        #with the degrees of freedom = visits. Since variances add, the variance 
        #pdfs convolve. The cumulative distribution function of the sum of two 
        #random deviates is the convolution of one pdf with a cdf. 

        #We'll consider the cdf of the noise-only variances because it's easier 
        #to interpolate
        noisesorted = np.sort(noiseonlyvar)
        #linear interpolation
        interpnoisecdf = UnivariateSpline(noisesorted,np.arange(self.numruns)/float(self.numruns),k=1,s=0)

        #We need a binned, signal-only variance probability distribution function for numerical convolution
        numsignalsamples = 100
        xsig = np.linspace(chi2.ppf(0.001, N),chi2.ppf(0.999, N),numsignalsamples)
        signalpdf = chi2.pdf(xsig, N)
        #correct x to the proper variance scale
        xsig = (self.signal**2.)*xsig/N
        pdfstepsize = xsig[1]-xsig[0]
        #Since everything is going to use this stepsize down the line,
        #normalize so the pdf integrates to 1 when summed (no factor of stepsize needed)
        signalpdf /= np.sum(signalpdf)

        #run through the sample magnitudes, calculate distance between cont 
        #and comp thresholds.
        #run until solution found.
        solutionfound=False
        
        for i,mref in enumerate(mag): 
            #i counts and mref is the currently sampled magnitude
            #Scale factor from m0
            scalefact = 10.**(0.4*(mref-m0))

            #Calculate the desired contamination threshold
            contthresh = np.percentile(noiseonlyvar,100.-100.*self.contamination)*scalefact

            #Realize the noise CDF at the required stepsize
            xnoise = np.arange(noisesorted[0]*scalefact,noisesorted[-1]*scalefact,pdfstepsize)
            
            #Only do calculation if near the solution:
            if (len(xnoise) > numsignalsamples/10) and (not solutionfound):
                noisecdf = interpnoisecdf(xnoise/scalefact)
                noisepdf = (noisecdf[1:]-noisecdf[:-1]) #turn into a noise pdf
                noisepdf /= np.sum(noisepdf)
                xnoise = (xnoise[1:]+xnoise[:-1])/2. #from cdf to pdf conversion

                #calculate and plot the convolution = signal+noise variance dist.
                convolution=0
                if len(noisepdf) > len(signalpdf):
                    convolution = np.convolve(noisepdf,signalpdf)
                else: 
                    convolution = np.convolve(signalpdf,noisepdf)
                xconvolved = xsig[0]+xnoise[0]+np.arange(len(convolution))*pdfstepsize

                #calculate the completeness threshold
                combinedcdf = np.cumsum(convolution)
                findcompthresh = UnivariateSpline(combinedcdf,xconvolved,k=1,s=0)
                compthresh = findcompthresh(1.-self.completeness)

                res[i] = compthresh - contthresh
                if res[i] < 0: solutionfound = True
        
        #interpolate for where the thresholds coincide
        #print res
        if np.sum(np.isfinite(res)) > 1:
            f1 = UnivariateSpline(mag[np.isfinite(res)],res[np.isfinite(res)],k=1,s=0)
            #sample the magnitude range at given resolution
            magsamples = np.arange(16,np.mean(m5),self.magres) 
            vardepth = magsamples[np.argmin(np.abs(f1(magsamples)))]
            return vardepth
        else:
            return min(mag)-1


#And test it out:
metric = VarDepth('fiveSigmaDepth',numruns=100) #Note: default numruns=10000 takes way too long.
slicer = slicers.HealpixSlicer(nside=64)
sqlconstraint = 'filter = "r"'
myBundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint)


#Run it:
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
bgroup = metricBundles.MetricBundleGroup({0: myBundle}, opsdb, outDir='newmetric_test', resultsDb=None)
bgroup.runAll()


myBundle.setPlotDict({'colorMin':16.1, 'colorMax':20.5})
bgroup.plotAll(closefigs=False,dpi=600,figformat='png')





# ###For a given hybrid focal plane layout how much sky do we cover as a function of time###
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler
import healpy as hp


# This is needed to avoid an error when a metric is redefined
from lsst.sims.maf.metrics import BaseMetric
try:
    del metrics.BaseMetric.registry['__main__.selectChipsMetric']
except KeyError:
    pass


class selectChipsMetric(BaseMetric):
    """
    Pass a set of sensors or rafts for a given vendor
    """

    def __init__(self, cols=None, sensors=None, **kwargs):
        if sensors == None:
            self.sensors = []
        else:
            self.sensors = sensors

        if cols is None:
            cols = []
        super(selectChipsMetric,self).__init__(col=cols, metricDtype=float, **kwargs)

    def _selectChips(self, chipName):
        """
        given a list of sensors increment count if a region of the sky covers that sensor 
        """
        count = 0
        for chip in chipName:
            for sensor in self.sensors:
                if sensor in chip:
                    count +=1
                    break
        return count
                
    def run(self, dataSlice, slicePoint=None):

        if 'chipNames' not in slicePoint.keys():
            raise ValueError('No chipname info, need to set useCamera=True with a spatial slicer.')

        result = self._selectChips(slicePoint['chipNames'])

        if result == 0:
            result = self.badval
        return result


# Set the database and query
database = 'enigma_1189_sqlite.db'
#sqlWhere = 'filter = "r" and night < 400'
opsdb = db.OpsimDatabase(database)
outDir = 'Camera'
resultsDb = db.ResultsDb(outDir=outDir)
nside=512


#define rafts for a given vendor

rafts = ['R:0,1', 'R:0,2', 'R:0,3',
         'R:1,0', 'R:1,1', 'R:1,2', 'R:1,3', 'R:1,4',
         'R:2,0', 'R:2,1', 'R:2,2', 'R:2,3', 'R:2,4',
         'R:3,0', 'R:3,1', 'R:3,2', 'R:3,3', 'R:3,4',
         'R:4,1', 'R:4,2', 'R:4,3',
        ]
rafts2 = ['R:0,1',  'R:0,3',
         'R:1,1', 'R:1,3', 
         'R:2,0', 'R:2,2', 'R:2,4',
         'R:3,1', 'R:3,3', 
         'R:4,1',  'R:4,3',
        ]
rafts1 = ['R:2,2', 'R:2,3', 'R:2,4',
         'R:3,0', 'R:3,1', 'R:3,2', 'R:3,3', 'R:3,4',
         'R:4,0', 'R:4,1', 'R:4,2', 'R:4,3', 
        ]
metric1 = metrics.CountMetric('expMJD')
metric2 = selectChipsMetric('expMJD', sensors =rafts2)
slicer = slicers.HealpixSlicer(nside=nside, useCamera=True)
summaryMetrics = [metrics.SumMetric()]


#run metric with all sensors
sqlWhere = 'filter = "r" and expMJD < 49547.36 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15.), np.radians(-15.))
#sqlWhere = 'fieldID = 2266 and expMJD < 49593.3'
bundle1 = metricBundles.MetricBundle(metric1,slicer,sqlWhere, summaryMetrics=summaryMetrics)
bundle2 = metricBundles.MetricBundle(metric2,slicer,sqlWhere, summaryMetrics=summaryMetrics)
bgFull = metricBundles.MetricBundleGroup({'Full':bundle1,'Hybrid':bundle2,},opsdb, outDir=outDir, resultsDb=resultsDb)
bgFull.runAll()
normPixels = len((np.where(bundle1.metricValues > 0))[0])


#plot camera
hp.mollview(bundle1.metricValues, title='No Camera')
hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(0,0,0))
hp.gnomview(bundle2.metricValues, xsize=800,ysize=800, rot=(0,0,0))
#hp.gnomview(bundle1.metricValues, xsize=400,ysize=400, rot=(48,-9,0), title='With Camera', unit='Vendor')
#hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(48,-9,0))


#loop over seasons and calculate the depth
nSeasons = 10
seasonLength = 50
nRepeats = [1,2,4,8,10]
numPixels = np.zeros((nSeasons,len(nRepeats)))
for i in range(nSeasons):
    sqlWhere = 'filter = "r" and night < %f and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (i*seasonLength,np.radians(15), np.radians(-15))
    bundle2 = metricBundles.MetricBundle(metric2,slicer,sqlWhere, summaryMetrics=summaryMetrics)
    bgHybrid = metricBundles.MetricBundleGroup({'Hybrid':bundle2},opsdb, outDir=outDir, resultsDb=resultsDb)
    bgHybrid.runAll() 
    for j,repeat in enumerate(nRepeats):
        numPixels[i,j] = len((np.where(bundle2.metricValues > repeat))[0])


#plot repeats as a function of time
fig = plt.figure(figsize=(10, 3.75))
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Days')
ax.set_ylabel('Fraction of sky')
for i in range(len(nRepeats)):
    plt.plot(seasonLength*np.arange(nSeasons), numPixels[:,i]*100./normPixels, label='Number of repeated observations %d'%(nRepeats[i]))
ax.legend(loc=0)


# We now have metrics that use the stellar luminosity function maps to compute the errors that will result from stellar crowding.
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.maps as maps


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'crowding_test'
resultsDb = db.ResultsDb(outDir=outDir)
nside = 32


# For the CrowdingMetric, one sets the acceptable magnitude uncertainty, and the metric finds the magnitude star that will have that uncertainty given the crowding.  
# 

bundleList = []
metric = metrics.CrowdingMetric(crowding_error=0.05)


slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
sql = 'filter="r" and night < 730'
plotDict={'colorMin':21., 'colorMax':25.}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict)
bundleList.append(bundle)

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)


bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# The second metric takes a single stellar magnitude and returns the resulting magnitude uncertainty on it, taking the standard 5-sigma depth or the crowding uncertainty, whichever is larger.
# 

bundleList = []
metric = metrics.CrowdingMagUncertMetric()
slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
sql = 'filter="r" and night < 730'
bundle = metricBundles.MetricBundle(metric,slicer,sql)
bundleList.append(bundle)
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)


bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)





# It can be useful to see how LSST performs with a mixed vendor focal plane. In this notebook, we set some chips to be from "vendor 1" and the rest from "vendor 2", and look at how the coverage of the sky looks if we only include chips from a single vendor.
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler
import healpy as hp


def drawRaft(ax, xLL, yLL, color='#CCCCCC', width = 1.0, height = 1.0, plotCCDs = 1):

    ax.add_patch(Rectangle((yLL, xLL), width, height, fill=True, color=color, ec='k'))
    # raft frame
    ax.plot([xLL, xLL], [yLL, yLL+height], 'black', linewidth=2)
    ax.plot([xLL+width, xLL+width], [yLL, yLL+height], 'black', linewidth=2)
    ax.plot([xLL, xLL+width], [yLL, yLL], 'black', linewidth=2)
    ax.plot([xLL, xLL+width], [yLL+height, yLL+height], 'black', linewidth=2)

    if plotCCDs: 
        ax.plot([xLL+width/3.0, xLL+width/3.0], [yLL, yLL+height], 'black', linewidth=1)
        ax.plot([xLL+2*width/3.0, xLL+2*width/3.0], [yLL, yLL+height], 'black', linewidth=1)
        ax.plot([xLL, xLL+width], [yLL+height/3.0, yLL+height/3.0], 'black', linewidth=1)
        ax.plot([xLL, xLL+width], [yLL+2*height/3.0, yLL+2*height/3.0], 'black', linewidth=1)

def plotHybridFOV(axes, option):

    # title 
    axes.set_title(option)

    # axes limits
    axes.set_xlim(0, 5.1)
    axes.set_ylim(0, 5.1)

    for i in (1, 2, 3): 
        for j in range(0,5):
            drawRaft(axes, i, j, color = 'red')
    for i in (0, 4): 
        for j in range(1,4):
            drawRaft(axes, i, j, color = 'red')
            
    if (option == 'A'):
        drawRaft(axes, 0, 2)
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 3)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 2, 4)
        drawRaft(axes, 3, 1)
        drawRaft(axes, 3, 3)
        drawRaft(axes, 4, 2)

    if (option == 'B'):
        drawRaft(axes, 0, 1)
        drawRaft(axes, 0, 2)
        drawRaft(axes, 0, 3)
        drawRaft(axes, 1, 0)
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 2)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 1)
        drawRaft(axes, 3, 0)

    if (option == 'C'):
        drawRaft(axes, 0, 1)
        drawRaft(axes, 0, 3)
        drawRaft(axes, 1, 0)
        drawRaft(axes, 1, 4)
        drawRaft(axes, 3, 0)
        drawRaft(axes, 3, 4)
        drawRaft(axes, 4, 1)
        drawRaft(axes, 4, 3)

    if (option == 'D'):
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 3)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 3, 1)
        drawRaft(axes, 3, 3)

    if (option == 'E'):
        drawRaft(axes, 1, 2)
        drawRaft(axes, 2, 1)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 2, 3)
        drawRaft(axes, 3, 2)

    if (option == 'F'):
        drawRaft(axes, 0, 2)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 4)
        drawRaft(axes, 4, 2)
     

### plot a 6-panel figure with hybrid focal plane realizations
def plotHybridFOVoptions(): 

    # Create figure and subplots
    fig = plt.figure(figsize=(8, 10))
    # this work well in *.py version but not so well in ipython notebook
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.9, bottom=0.05, top=0.95)

    optionsList = ('A', 'B', 'C', 'D', 'E', 'F')
    plotNo = 0
    for option in optionsList:
        plotNo += 1
        axes = plt.subplot(3, 2, plotNo, xticks=[], yticks=[], frameon=False)
        plotHybridFOV(axes, option)

    #plt.savefig('./HybridFOVoptions.png')
    plt.show() 


### 
plotHybridFOVoptions()


# Set up each configuration to return a list of chips in a way MAF understands
# Let's do this for a hybrid focal plane
def makeChipList(raftConfig):
    raftDict = {'R:1,0':1,
                'R:2,0':2 ,
                'R:3,0':3 ,
                'R:0,1':4 ,
                'R:1,1':5 ,
                'R:2,1':6 ,
                'R:3,1':7 ,
                'R:4,1':8 ,
                'R:0,2':9 ,
                'R:1,2':10,
                'R:2,2':11,
                'R:3,2':12,
                'R:4,2':13,
                'R:0,3':14,
                'R:1,3':15,
                'R:2,3':16,
                'R:3,3':17,
                'R:4,3':18,
                'R:1,4':19,
                'R:2,4':20,
                'R:3,4':21}

    sensors = ['S:0,0', 'S:0,1', 'S:0,2',
               'S:1,0', 'S:1,1', 'S:1,2',
               'S:2,0', 'S:2,1', 'S:2,2',]


    raftReverseDict = {}
    for key in raftDict:
        raftReverseDict[raftDict[key]] = key
    raftConfigs = {'A':{'rafts2':[1,3,4,6,8,10,12,14,16,18,19,21], 'rafts1':[2,5,7,9,11,13,15,17,20]},
                   'B':{'rafts2':[7,8,11,12,13,15,16,17,18,19,20,21], 'rafts1':[1,2,3,4,5,6,9,10,14]},
                   'C':{'rafts2':[2,5,6,7,9,10,11,12,13,15,16,17,20], 'rafts1':[1,3,4,8,14,18,19,21]},
                   'D':{'rafts2':[1,2,3,4,6,8,9,10,12,13,14,16,18,19,20,21], 'rafts1':[5,7,11,15,17]},
                   'E':{'rafts2':[1,2,3,4,5,7,8,9,13,14,15,17,18,19,20,21], 'rafts1':[6,10,11,12,16]},
                   'F':{'rafts2':[1,2,3,4,5,7,8,9,13,14,15,17,18,19,20,21], 'rafts1':[6,10,11,12,16]}
                  }
    rafts1 = []
    rafts2 = []
    for indx in raftConfigs[raftConfig]['rafts1']:
        rafts1.append(raftReverseDict[indx])

    for indx in raftConfigs[raftConfig]['rafts2']:
        rafts2.append(raftReverseDict[indx])

    chips1 = []
    for raft in rafts1:
        for sensor in sensors:
            chips1.append(raft+' '+sensor)

    chips2 = []
    for raft in rafts2:
        for sensor in sensors:
            chips2.append(raft+' '+sensor)
    return chips1, chips2


chips1, chips2 = makeChipList('E')


print 'chips1', chips1
print 'chips2', chips2


database = 'enigma_1189_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'Camera'
resultsDb = db.ResultsDb(outDir=outDir)
nside = 512


# Dithering off, just count the number of observations using only vendor 1, or only vendor 2 and 
# compare to the regular full focal plane.
sqlWhere = 'filter = "u" and night < 730 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15), np.radians(-15))
metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(latCol='fieldDec', lonCol='fieldRA', nside=nside)
slicer2 = slicers.HealpixSlicer(latCol='fieldDec', lonCol='fieldRA',nside=nside, useCamera=True, chipNames=chips2)
slicer3 = slicers.HealpixSlicer(latCol='fieldDec', lonCol='fieldRA',nside=nside, useCamera=True, chipNames=chips1)
bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere, metadata='No camera')
bundle2 = metricBundles.MetricBundle(metric,slicer2,sqlWhere, metadata='Chips2')
bundle3 = metricBundles.MetricBundle(metric,slicer3,sqlWhere, metadata='Chips1')
bd = metricBundles.makeBundlesDictFromList([bundle1,bundle2,bundle3])
bg = metricBundles.MetricBundleGroup(bd,opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(7,-7,0), title='No Camera', unit='Count', min=1,max=21)
hp.gnomview(bundle2.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='Chips1', unit='Count', min=1,max=21)
hp.gnomview(bundle3.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='Chips2', unit='Count', min=1,max=21)


# Dithering on, just count the number of exposures using only vendor 1, only vendor 2, and all spots.
sqlWhere = 'filter = "u" and night < 730 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15), np.radians(-15))
metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA', nside=nside)
slicer2 = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA',nside=nside, useCamera=True, chipNames=chips2)
slicer3 = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA',nside=nside, useCamera=True, chipNames=chips1)
bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere, metadata='No camera')
bundle2 = metricBundles.MetricBundle(metric,slicer2,sqlWhere, metadata='Chips2')
bundle3 = metricBundles.MetricBundle(metric,slicer3,sqlWhere, metadata='Chips1')
bd = metricBundles.makeBundlesDictFromList([bundle1,bundle2,bundle3])
bg = metricBundles.MetricBundleGroup(bd,opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(7,-7,0), title='No Camera', unit='Count', min=1,max=21)
hp.gnomview(bundle2.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='Chips1', unit='Count', min=1,max=21)
hp.gnomview(bundle3.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='Chips2', unit='Count', min=1,max=21)





# This notebook assumes sims_maf version >= 1.1 and that you have 'setup sims_maf' in your shell. 
# 

import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


# ##Writing a new metric##
# 
# MAF provides many 'stock' metrics, and there are more in the sims_maf_contrib library. 
# 

# List of provided metrics
metrics.BaseMetric.list(doc=False)


# But at some point, you're likely to want to write your own metric. We have tried to make the process for this simple. 
# 
# Here is the code for a very simple, (existing) metric which calculates the coadded depth of a set of visits. 
# 

from lsst.sims.maf.metrics import BaseMetric

class Coaddm5Metric(BaseMetric):
    """Calculate the coadded m5 value at this gridpoint."""
    def __init__(self, m5Col = 'fiveSigmaDepth', metricName='CoaddM5', **kwargs):
        """Instantiate metric.
        m5col = the column name of the individual visit m5 data."""
        self.m5col = m5col
        super(Coaddm5Metric, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5col])))


# To understand this, you need to know a little bit about "classes" and "inheritance". 
# 
# Basically, a "class" is a python object which can hold data and methods (like functions) to manipulate that data. The idea is that a class can be a self-encapsulated thing -- the class knows what its data should look like, and then the methods know how to work with that data.  
# 
# "Inheritance" means that you can create a child version of another class, that inherits all of its features - and possibly adds new data or methods or replaces data or methods of the parent. 
# 
# The point here is that the "framework" part of MAF is encapsulated in the BaseMetric. By inheriting from the BaseMetric (that's the bit where we said class Coaddm5Metric(**BaseMetric**) above), we get the column tracking so that MAF knows what columns to query the database for and we get added to the registry of existing metrics. 
# 
# By following the same API (the 'signature' of the methods), we can write a new metric that will plug into the MAF framework seamlessly. This means you write an `__init__` method that includes `(self,  **kwargs)` and whatever else your particular metric needs. And then you write a `run` method that is called as `run(self, dataSlice, slicePoint=None)`.  
# 

# `dataSlice` refers to the visits handed to the metric by the slicer. `slicePoint` refers to the metadata about the slice (such as it's ra/dec in the case of a HealpixSlicer, or it's bin information in the case of a OneDSlicer).
# 

# Let's write another example, this time to calculate the Percentile value of a given column in a set of visits.
# 

# Import BaseMetric, or have it available to inherit from
from lsst.sims.maf.metrics import BaseMetric

# Define our class, inheriting from BaseMetric
class OurPercentileMetric(BaseMetric):
    # Add a doc string to describe the metric.
    """
    Calculate the percentile value of a data column
    """
    # Add our "__init__" method to instantiate the class.
    # We will make the 'percentile' value an additional value to be set by the user.
    # **kwargs allows additional values to be passed to the BaseMetric that you 
    #     may not have been using here and don't want to bother with. 
    def __init__(self, colname, percentile, **kwargs):
        # Set the values we want to keep for our class.
        self.colname = colname
        self.percentile = percentile
        # Now we have to call the BaseMetric's __init__ method, to get the "framework" part set up.
        # We currently do this using 'super', which just calls BaseMetric's method.
        # The call to super just basically looks like this .. you must pass the columns you need, and the kwargs.
        super(OurPercentileMetric, self).__init__(col=colname, **kwargs)
        
    # Now write out "run" method, the part that does the metric calculation.
    def run(self, dataSlice, slicePoint=None):
        # for this calculation, I'll just call numpy's percentile function.
        result = np.percentile(dataSlice[self.colname], self.percentile)
        return result


# So then how do we use this new metric? Just as before, although you may have to adjust the namespace.
# 

metric = OurPercentileMetric('airmass', 20)
slicer = slicers.HealpixSlicer(nside=64)
sqlconstraint = 'filter = "r" and night<365'
myBundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint)


opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
bgroup = metricBundles.MetricBundleGroup({0: myBundle}, opsdb, outDir='newmetric_test', resultsDb=None)
bgroup.runAll()


myBundle.setPlotDict({'colorMin':1.0, 'colorMax':1.8})
bgroup.plotAll(closefigs=False)





# This notebook assumes you have sims_maf version >= 1.1 and have 'setup sims_maf' in your shell. 
# 
# #Transient metric#
# 
# This notebook demonstrates the transient metric.  For this metric, one can set the light curve shape and detection threshold (e.g., one detection, demand a detection on the rise, demand multiple filters, etc).  The metric then computes what fraction of sources would meet the detection criteria if the object was continually exploding throughout the survey.  
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db


# Let's look at how the transient metric makes light curves
peaks = {'uPeak':18, 'gPeak':19, 'rPeak':20, 'iPeak':21, 'zPeak':22,'yPeak':23}
colors = ['b','g','r','purple','y','magenta','k']
filterNames = ['u','g','r','i','z','y']

transDuration = 60. # Days
transMetric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, 
                                      transDuration=transDuration, peakTime=5., **peaks)


times = np.arange(0.,121,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([35,18])
plt.legend()


# So the transient metric basically takes a simple lightcurve, repeats it continually over the entire survey length, and then checks to see what fraction of the lightcurves meet some specified criteria.
# 

# Modify the slopes and duration a bit
transDuration = 10.
transMetric = metrics.TransientMetric(riseSlope=-1., declineSlope=1, transDuration=transDuration, 
                                 peakTime=5., **peaks)


times = np.arange(0.,121,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([30,18])
plt.legend()


# By default, the transient metric let's you make simple saw-tooth light curves.  If you want to use a more complicated light curve, one can simply sub-class the transient metric and replace the lightCurve method with a function of your own. 
# 

# Pick a slicer
slicer = slicers.HealpixSlicer(nside=64)

summaryMetrics = [metrics.MedianMetric()]
# Configure some metrics
metricList = []
# What fraction of 60-day, r=20 mag flat transients are detected at least once?
metric = metrics.TransientMetric(riseSlope=0., declineSlope=0., transDuration=60., 
                                 peakTime=5., rPeak=20., metricName='Alert')
metricList.append(metric)
# Now make the light curve shape a little more realistic. 
metric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, transDuration=60., 
                                 peakTime=5., rPeak=20., metricName='Alert, shaped LC')
metricList.append(metric)
#  Demand 2 points before tmax before counting the LC as detected
metric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, transDuration=60., 
                                 peakTime=5., rPeak=20., nPrePeak=2, metricName='Detected on rise')
metricList.append(metric)


# Set the database and query
runName = 'enigma_1189'
sqlconstraint = 'filter = "r"'
bDict={}
for i,metric in enumerate(metricList):
    bDict[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)


opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)


bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


# Compute and print summary metrics
for key in bDict:
    bDict[key].computeSummaryStats(resultsDb=resultsDb)
    print bDict[key].metric.name, bDict[key].summaryValues


# Update to use all the observations, not just the r-band
bDict={}
sqlconstraint = ''
for i,metric in enumerate(metricList):
    bDict[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, summaryMetrics=summaryMetrics)

bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)
for key in bDict:
    bDict[key].computeSummaryStats(resultsDb=resultsDb)
    print bDict[key].metric.name, bDict[key].summaryValues


bDict[0].metricValues








# Try out the ability to select only a few rafts from the camera geometry
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler
import healpy as hp


# Set the database and query
database = 'enigma_1189_sqlite.db'
#sqlWhere = 'filter = "r" and night < 400'
opsdb = db.OpsimDatabase(database)
outDir = 'Camera'
resultsDb = db.ResultsDb(outDir=outDir)
nside=512


rafts = ['R:0,1', 'R:0,2', 'R:0,3',
         'R:1,0', 'R:1,1', 'R:1,2', 'R:1,3', 'R:1,4',
         'R:2,0', 'R:2,1', 'R:2,2', 'R:2,3', 'R:2,4',
         'R:3,0', 'R:3,1', 'R:3,2', 'R:3,3', 'R:3,4',
         'R:4,1', 'R:4,2', 'R:4,3',
        ]
chips = ['S:0,0', 'S:0,1', 'S:0,2',
        'S:1,0', 'S:1,1', 'S:1,2',
        'S:2,0', 'S:2,1', 'S:2,2']
allChips =[]
for raft in rafts:
    for chip in chips:
        allChips.append(raft+' '+chip)


sqlWhere = 'filter = "r" and expMJD < 49547.36 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15.), np.radians(-15.))
metric = metrics.Coaddm5Metric()
slicer = slicers.HealpixSlicer(nside=nside, useCamera=True, chipNames=allChips)


bundle = metricBundles.MetricBundle(metric,slicer,sqlWhere)
bg = metricBundles.MetricBundleGroup({0:bundle},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)
hp.gnomview(bundle.metricValues, xsize=800,ysize=800, rot=(0,0,0))


# Now let's use every-other chip
halfChips = []
for raft in rafts[0::2]:
    for chip in chips:
        halfChips.append(raft+' '+chip)


slicer = slicers.HealpixSlicer(nside=nside, useCamera=True, chipNames=halfChips)


bundle = metricBundles.MetricBundle(metric,slicer,sqlWhere)
bg = metricBundles.MetricBundleGroup({0:bundle},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)
hp.gnomview(bundle.metricValues, xsize=800,ysize=800, rot=(0,0,0))


# I think raft 0,0 is the lower left in this image (there's a slight rotation).  Note that the chip name info is also passed to the metric with the other slicePoint info, so metrics can also use the chip name info if they want to.
# 










# ## Index ##
# 
# This notebook serves as an 'index' to the other sims_maf tutorials, providing a short description of their contents. 
# 
# ---
# 
# ### Installing and updating MAF ### 
# If you have not installed the LSST software stack and sims_maf (an LSST software package), please follow the instructions at https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF
# 
# ---
# 
# Note that all of these tutorials expect sims_maf to be setup in your shell, and it must be sims_maf version 1.1 or better.  
# 
# To check which version of sims_maf you are using, you can enter the following commands at a python prompt:
# >`import lsst.sims.maf`<br>
# >`lsst.sims.maf.__version__`
# 
# 

# ---
# 
# ## Tutorials: Start here ##
# 
# If you are new to ipython notebooks and uncertain if your environment is set up correctly, you may find it helpful to start with the [TestNotebook](./TestNotebook.ipynb). In this notebook, you can see how the basic ipython environment works. 
# 
# Start with the [Introduction](./Introduction%20Notebook.ipynb) to start using MAF with a hands-on example. If you want some overview slides, we also have an [introductory talk](https://github.com/LSST-nonproject/sims_maf_contrib/blob/master/workshops/UK_2015/Cambridge_MAFIntro.pdf). 
# 
# To learn about various slicer options, look at the [Slicers](./Slicers.ipynb) notebook.
# 

# ## More advanced capabilities##
# 
# [Stackers](./Stackers.ipynb) Example of using Stackers to generate new columns on-the-fly for each opsim visit (for example, adding "Hour Angle"). The [Dithers](./Dithers.ipynb) notebook may also be useful to understand stackers.
# 
# [Plotting Examples](./Plotting Examples.ipynb) Examples of generating metric data and then using this to create many different kinds of plots. This illustrates using the plotHandler class, as well as plotting data directly from a metricBundle.
# 
# [Dithers](./Dithers.ipynb) This notebook demonstrates the different dithering patterns available within MAF. It also shows more examples of generating plots, with the plotHandler class as well as directly with metricBundles. It also introduces the use of metric data outside of the MAF metricBundle, including read data back from disk. There are lots of concepts used in this notebook (we'll split it up eventually). 
# 
# [MAFCameraGeom](./MAFCameraGeom.ipynb) In this notebook we demonstrate using the MAF camera footprint geometry - that is, including the chipgaps and realistic focal plane in your metric calculations. 
# 
# [Complex Metrics](./Complex Metrics.ipynb) This notebook illustrates using a 'complex' metric - that is, a metric which calculates more than one value at a time. 
# 
# [Compare Runs](./Compare Runs.ipynb) This notebook looks at how you can grab summary statistic data from the results databases of multiple opsim runs (where you've already calculated metrics and generated summary statistics). With the summary statistics you could compare (for example) the number of visits or the mean proper motion accuracy, from many different simulated surveys. 
# 
# [Writing a New Metric](./Writing A New Metric.ipynb) If it's time to write your own metric, this is where you'll find a demonstration of how to do it.
# 
# [Exploring Light Curves](./PullLightCurves.ipynb) Often, one just wants to see what an LSST light curve will look like. This example pulls all the observations from a single ra,dec position and plots possible SNe light curves for taht point.
# 




# ## Example of comparing maf run results ##
# 
# This notebook provides a demonstration of using MAF to compare the summary statistics coming from multiple runs. These summary statistics are visible using 'showMaf' for a single run (for example, see the results from [enigma_1189](http://tusken.astro.washington.edu:8080/summaryStats?runId=2)), but we are starting to develop tools to compare these summary statistics from multiple runs. 
# 
# Requirements:
# To run this notebook, you need sims_maf version >= 1.3. 
# You will also need some resultsDB_sqlite.db (maf results sqlite databases) files to query. An example set (as expected in this notebook) is available for [download](http://www.astro.washington.edu/users/lynnej/opsim/results_example.tar.gz). 
# 

# import the modules needed.
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from lsst.sims.maf.db import ResultsDb


# Modify the next cell as needed to specify the location of your results database files. If you downloaded the [example set](http://www.astro.washington.edu/users/lynnej/opsim/results_example.tar.gz), set 'rootDir' below to the directory you wish to work with, and untar the 'results_example.tar.gz' file into that directory. 
# 
# Note that the connection to ResultsDB assumes the standard name for the resultsDB file (e.g. "resultsDb_sqlite.db"). If you modify these filenames, you should also modify the database name below. An example is given in the last line.
# 

rootDir = '.'
rundb = {}
rundb['enigma_1189'] = ResultsDb(database=os.path.join(rootDir, 'enigma_1189_scheduler_results.db'))
rundb['ewok_1004'] = ResultsDb(database=os.path.join(rootDir, 'ewok_1004_scheduler_results.db'))


# There are various methods on the resultsDb to help you gather information that was recorded in the sqlite file. 
# The most relevant methods, in order to compare metrics between runs, are the **getMetricId**, **getMetricDisplayInfo**, **getSummaryStats**, and **getPlotFiles** methods. These return numpy arrays with the relevant information. Generally, the first thing you'd want to do is identify the 'metricID' for a particular metric. You may know the metric name, the slicer name, and/or the metadata associated with the metric you want.  
# 

help(rundb['enigma_1189'].getMetricId)
help(rundb['enigma_1189'].getMetricDisplayInfo)
help(rundb['enigma_1189'].getSummaryStats)
help(rundb['enigma_1189'].getPlotFiles)


# For example: you might want to look at the "Nvisits" metric results. In our scheduler validation metric runs, the "Nvisits" metric counts how many visits a field receives; without any further clarification, getMetricId will return the metric IDs associated with all metrics named Nvisits.  (There are actually one of these for each filter for each of 'All props', 'DD' and 'WFD' -- the information on the metrics available can be derived from the showMaf visualization of the MAF run results). 
# 

metricName = 'NVisits'
mIds = {}
for r in rundb:
    mIds[r] = rundb[r].getMetricId(metricName=metricName)
    print r, mIds[r]
    print ''


# We can also specify a particular metadata value -- so let's look for Nvisits metric results for the r band, WFD only.
# 

# Retrieve all summary statistics for a metric + set of metric metadata + for a particular slicer.
metricName = 'NVisits'
metricMetadata = 'i band, WFD'
mIds = {}
for r in rundb:
    mIds[r] = rundb[r].getMetricId(metricName=metricName, metricMetadata=metricMetadata)
    print r, mIds[r]


# With this very specific information and the metricIds associated with this specific metric, we can then get the plots associated with this metric and the summary statistics.
# 

for r in rundb:
    plotFiles = rundb[r].getPlotFiles(mIds[r])
    summaryStats = rundb[r].getSummaryStats(mIds[r])
    print "Run %s" %r
    print plotFiles['plotFile']  # this is a numpy array with the metric information + plot file name
    print summaryStats
    print ''


# And if we had a very specific summary statistic we wanted to get, for a very specific metric, slicer and metric metadata combination: 
# 

metricName = 'NVisits'
slicerName = 'OneDSlicer'
metricMetadata  = 'Per night'  # capitalization matters!
summaryStatName = 'Median'

stats = {}
for r in rundb:
    mIds = rundb[r].getMetricId(metricName=metricName, metricMetadata=metricMetadata, slicerName=slicerName)
    stats[r] = rundb[r].getSummaryStats(mIds, summaryName=summaryStatName)   


# At this point, 'stats' is a dictionary containing a numpy array with fields "metricId", "metricMetadata, "slicerName",  "summaryName" and "summaryValue" for each statistic that matches the metricIds: (i.e. all the metric info and the summary statistic info). 
# 

# All of the values in stats
print stats['enigma_1189']
# And the relevant 'summaryValue' -- of which there is only one, because we used one metricID and one summaryStatName.
print stats['enigma_1189']['summaryValue']


# So you can easily create bigger tables or ratios:
baseline = stats['enigma_1189']['summaryValue'][0]
for r in rundb:
    print r, stats[r]['summaryValue'][0], stats[r]['summaryValue'][0]/baseline


# Or you could pull out several summary statistics, to plot together.

# Nice names for the comparisons we'll do (nice names for a plot)
metricComparisons = ['Nights in survey', 'Total NVisits', 'NVisits Per night', 'Mean slew time', 'Mean Nfilter changes', 
                     'Median Nvisits WFD', 'Median Nvisits r All']
# But we need to know how to pull this info out of the resultsDB, so get the actual metric names, metadata, summaryName.
metricInfo = [{'metricName':'Total nights in survey', 'metadata':None, 'summary':None},
              {'metricName':'TotalNVisits', 'metadata':'All Visits', 'summary':None},
              {'metricName':'NVisits', 'metadata':'Per night', 'summary':'Median'},
              {'metricName':'Mean slewTime', 'metadata':None, 'summary':None},
              {'metricName':'Filter Changes', 'metadata':'Per night', 'summary':'Mean'}, 
              {'metricName':'Nvisits, all filters', 'metadata':'All filters WFD: histogram only', 'summary':'Median'},
              {'metricName':'Nvisits', 'metadata':'r band, all props', 'summary':'Median'}]

stats = {}
for r in rundb:
    stats[r] = np.zeros(len(metricComparisons), float)
    for i, (mComparison, mInfo) in enumerate(zip(metricComparisons, metricInfo)):
        mIds = rundb[r].getMetricId(metricName=mInfo['metricName'], metricMetadata=mInfo['metadata'])
        s = rundb[r].getSummaryStats(mIds, summaryName=mInfo['summary'])
        stats[r][i] = s['summaryValue'][0]        
    print r, stats[r]


# Because the scales will be quite different (# of visits vs. # of filter changes, for example), normalize
#   both by dividing by the first set of values (or pick another baseline).

baseline = stats['ewok_1004']
xoffset = 0.8/(float(len(rundb)))
x = np.arange(len(baseline))
colors = np.random.random_sample((len(baseline), 3))
for i, r in enumerate(rundb):
    plt.bar(x+i*xoffset, stats[r]/baseline, width=xoffset, color=colors[i], label=r)
plt.xticks(x, metricComparisons, rotation=60)
plt.axhline(1.0)
plt.ylim(0.9, 1.1)
plt.legend(loc=(1.0, 0.2))


# We can also do more general comparisons. We can query for all of the metrics and summary statistics; Pandas becomes useful at this point, for slicing and dicing these bigger arrays. 
# 

import pandas as pd

metrics = {}
stats = {}
for r in rundb:
    metrics[r] = rundb[r].getMetricDisplayInfo()
    stats[r] = rundb[r].getSummaryStats()
    metrics[r] = pd.DataFrame(metrics[r])
    stats[r] = pd.DataFrame(stats[r])


# Let's pull out all the metrics for subgroup 'WFD' in the Seeing and SkyBrightness groups.

groupList = ['G: Seeing', 'H: SkyBrightness']

compareStats = {}
for r in rundb:
    m = metrics[r].query('displaySubgroup == "WFD"')
    m = m.query('displayGroup in @groupList')
    m = m.query('slicerName != "OneDSlicer"')
    m = m[m.metricName.str.contains('%ile') == False]
    mIds = m.metricId
    compareStats[r] = stats[r].query('metricId in @mIds')  #we could have done this using getSummaryStats too.


# Find stats in common. 
baseline =  'ewok_1004'
foundStat = np.ones(len(compareStats[baseline]), dtype=bool)
plotStats = {}
for r in rundb:
    plotStats[r] = np.zeros(len(compareStats[baseline]), float)
for count, (i, compStat) in enumerate(compareStats[baseline].iterrows()):
    for r in rundb:
        query = '(metricName == @compStat.metricName) and (metricMetadata == @compStat.metricMetadata)'
        query +=  ' and (summaryName == @compStat.summaryName)'
        s = compareStats[r].query(query)
        if len(s) > 0:
            s = s.iloc[0]
            plotStats[r][count] = s.summaryValue
        else:
            foundStat[count] = False
for r in rundb:
    plotStats[r] = plotStats[r][np.where(foundStat)]
print len(plotStats[baseline])
    
compareStats[baseline].loc[:,'foundCol'] = foundStat


baseline = compareStats['ewok_1004'].query('foundCol == True')
metricNames = []
for i, pStat in baseline.iterrows():
    if pStat.summaryName == 'Identity':
        metricNames.append(' '.join([pStat.metricName, pStat.metricMetadata]))
    else:
        metricNames.append(' '.join([pStat.summaryName, pStat.metricName, pStat.metricMetadata]))


baseline= 'ewok_1004'
xoffset = 0.8/(float(len(rundb)))
x = np.arange(len(plotStats[baseline]))
colors = np.random.random_sample((len(plotStats[baseline]), 3))
plt.figure(figsize=(20, 6))
for i, r in enumerate(rundb):
    plt.bar(x+i*xoffset, plotStats[r]/plotStats[baseline], width=xoffset, color=colors[i], label=r)
plt.xticks(x, metricNames, rotation=60)
plt.axhline(1.0)
plt.ylim(0.9, 1.1)
plt.legend(loc=(1.0, 0.2))





# Example of using the stellar luminosity function
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.maps as maps


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'starMap_test'
resultsDb = db.ResultsDb(outDir=outDir)


bundleList = []
sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64, useCache=False)
metric = metrics.StarDensityMetric(metricName='rmag<25')
# setup the stellar density map to use. By default, all stars in the CatSim catalog are included
mafMap = maps.StellarDensityMap()
plotDict = {'colorMin':0.001, 'colorMax':.1, 'logScale':True}
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)

metric = metrics.StarDensityMetric(rmagLimit=27.5,metricName='rmag<28')
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)


bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# Now try it again with the White Dwarf density maps
# 

bundleList = []
sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64, useCache=False)
metric = metrics.StarDensityMetric(metricName='WhiteDwarfs_rmag<25')
mafMap = maps.StellarDensityMap(startype='wdstars')
plotDict = {'colorMin':0.0001, 'colorMax':0.01, 'logScale':True}
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)

metric = metrics.StarDensityMetric(rmagLimit=27.5,metricName='WhiteDwarfs_rmag<28')
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)


bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)





# Compare the coadded depth with the extragalactic extincted depth
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np


# Set up the database connection
opsdb = db.OpsimDatabase('minion_1016_sqlite.db')
outDir = 'dustMaps'
resultsDb = db.ResultsDb(outDir=outDir)


nside=128
bundleList = []
slicer1 = slicers.HealpixSlicer(nside=nside)
slicer2 = slicers.HealpixSlicer(nside=nside, useCache=False)
metric1 = metrics.Coaddm5Metric()
metric2 = metrics.ExgalM5()
filters = ['u', 'g', 'r', 'i', 'z', 'y']
mins = {'u': 23.7, 'g':25.2, 'r':22.5, 'i': 24.6, 'z': 23.7,'y':22.8 }
maxes = {'u': 27.6, 'g':28.5, 'r': 28.5, 'i': 27.9, 'z': 27.6,'y':26.1 }

for filtername in filters:
    sql = 'filter="%s"' % filtername
    plotDict = {'colorMin': mins[filtername], 'colorMax': maxes[filtername]}
    bundleList.append(metricBundles.MetricBundle(metric1, slicer1, sql))
    bundleList.append(metricBundles.MetricBundle(metric2, slicer2, sql, plotDict=plotDict))


bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)





# ##Variability Metrics
# 

# These variabilty metrics were developed as a first attempt to understand LSST performance for discovering and characterising periodic variables. They are available in the [sims_maf_contrib](https://github.com/LSST-nonproject/sims_maf_contrib) github repo, in [varMetrics.py](https://github.com/LSST-nonproject/sims_maf_contrib/blob/master/mafContrib/varMetrics.py). 
# 
# One metric (PhaseGaps) looks at the gaps in phase coverage for a set of observations. At each point in the sky (if we use a healpix slicer), a range of periods and the times of the visits are converted to phase values: then the largest gap in these phases is calculated. The metric calculates the max gap for a range of periods at each point; thus it also provides a few reduce functions to convert this to the period with the largest gap (at each point in the sky) and the value of the largest gap in phase coverage (at each point in the sky). We can also read the metric values back into python and generate plots of phase gaps vs. period, including the data for all points.
# 
# The second metric (PeriodDeviation) generates a light curve (representing the variability of the source as a simple sine curve but neglecting photometric errors) with user-settable range of periods, then attempts to recover the period using a simple lomb-scargle calculation. Changing the period fitting routine would doubtless improve the period recovery.  Again, this should be paired with a healpix slicer to evaluate sets of visits which correspond to the same spot on the sky, and again each point is sampled at a range of different periods so the metric provides reduce methods and we can also read back in the results to evaluate true period vs. fit period across the entire sky. 
# 
# (this notebook requires sims_maf version >= 1.0)
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Import MAF modules.
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup
# Import the contributed metrics and stackers 
import mafContrib


# Connect to the opsim database and set up some metadata.
# 

runName = 'enigma_1189'
database =  runName + '_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'variability_test'


# Set up the metrics and slicer. 
# 

phaseMetric = metrics.PhaseGapMetric(nPeriods=20, periodMin=0.2, periodMax=3.5)
periodMetric = mafContrib.PeriodDeviationMetric(nPeriods=2, periodMin=2, periodMax=3.5)
phaseslicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')
periodslicer = slicers.HealpixSlicer(nside=32, lonCol='ditheredRA', latCol='ditheredDec')


summaryMetrics = [metrics.MinMetric(), metrics.MaxMetric(), metrics.MeanMetric()]


sqlconstraint = 'night<365 and (filter="r" or filter="i")'
phaseBundle = MetricBundle(phaseMetric, phaseslicer, sqlconstraint=sqlconstraint, 
                           runName=runName, summaryMetrics=summaryMetrics)
periodBundle = MetricBundle(periodMetric, periodslicer, sqlconstraint=sqlconstraint, 
                            runName=runName, summaryMetrics=summaryMetrics)


bdict = {'Phase':phaseBundle, 'Period':periodBundle}


resultsDb = db.ResultsDb(outDir=outDir)
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)


bgroup.runAll()


bgroup.plotAll(closefigs=False)


# Plot a contour plot of the periods vs. phase gaps (counting all points across the sky). 
# (We'll add this kind of plot as a plotter option into maf itself, but it just hasn't quite made it there yet.)
# 

def count_number(x, y, xbinsize=None, ybinsize=None, nxbins=None, nybins=None):
    # Set up grid for contour/density plot.
    xmin = min(x)
    ymin = min(y)
    if (xbinsize!=None) & (ybinsize!=None):
        xbins = np.arange(xmin, max(x), xbinsize)
        ybins = np.arange(ymin, max(y), ybinsize)
        nxbins = xbins.shape[0]
        nybins = ybins.shape[0]
    elif (nxbins!=None) & (nybins!=None):
        xbinsize = (max(x) - xmin)/float(nxbins)
        ybinsize = (max(y) - ymin)/float(nybins)
        xbins = np.arange(xmin, max(x), xbinsize)
        ybins = np.arange(ymin, max(y), ybinsize)
        nxbins = xbins.shape[0]
        nybins = ybins.shape[0]
    else:
        raise Exception("Must specify both of either xbinsize/ybinsize or nxbins/nybins")
    counts = np.zeros((nybins, nxbins), dtype='int')
    # Assign each data point (x/y) to a bin.
    for i in range(len(x)):
        xidx = min(int((x[i] - xmin)/xbinsize), nxbins-1)
        yidx = min(int((y[i] - ymin)/ybinsize), nybins-1)
        counts[yidx][xidx] += 1
    # Create 2D x/y arrays, to match 2D counts array.
    xi, yi = np.meshgrid(xbins, ybins)
    return xi, yi, counts


nperiods = len(phaseBundle.metricValues[-1:][0]['periods'])
periods = []
phaseGaps = []
for mval in phaseBundle.metricValues.compressed():
    for p, pGap in zip(mval['periods'], mval['maxGaps']):
            periods.append(p)
            phaseGaps.append(pGap)

periods = np.array(periods, 'float')
phaseGaps = np.array(phaseGaps, 'float')
timeGaps = phaseGaps * periods

periodi, phasegapi, counts = count_number(periods, phaseGaps, nxbins=100, nybins=100)
plt.figure()
levels = np.log10(np.arange(0.1, 200, 1))
plt.contourf(periodi, phasegapi, np.log10(counts), levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('Period (days)')
plt.ylabel('Largest Phase gap')


nperiods = len(periodBundle.metricValues[-1:][0]['periods'])
periods = []
periodsdev = []
for mval in periodBundle.metricValues.compressed():
    for p, pdev in zip(mval['periods'], mval['periodsdev']):
            periods.append(p)
            periodsdev.append(pdev)

periods = np.array(periods, 'float')
periodsdev = np.array(periodsdev, 'float')
fitperiods = periodsdev * periods + periods

periodi, periodsdevi, counts = count_number(periods, periodsdev, nxbins=100, nybins=100)
plt.figure()
levels = np.arange(0, 15, .1)
#levels = np.log10(levels)
#counts = np.log10(counts)
plt.contourf(periodi, periodsdevi, counts, levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('Period (days)')
plt.ylabel('Period deviation')


plt.figure()
plt.plot(periods, fitperiods, 'k.')
plt.xlabel('True Period (days)')
plt.ylabel('Fit Period (days)')

periodi, fitperiodsi, counts = count_number(periods, fitperiods, nxbins=100, nybins=100)
plt.figure()
#counts = np.log10(counts)
plt.contourf(periodi, fitperiodsi, counts, levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('True Period (days)')
plt.ylabel('Fit period (days)')


# Check how well we can recover a 3-day period variable
periodMetric = mafContrib.PeriodDeviationMetric(nPeriods=2, periodMin=2, periodMax=3.5, periodCheck=3.)
periodslicer = slicers.HealpixSlicer(nside=32, lonCol='ditheredRA', latCol='ditheredDec')
sqlconstraint = 'night<365 and (filter="r" or filter="i")'
periodBundle = MetricBundle(periodMetric, periodslicer, sqlconstraint=sqlconstraint, 
                            runName=runName)
bdict = {'Period':periodBundle}
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)





# An example of visulizing how well LSST can detect NEO objects and plottting things in rotating ecliptic coordinates.
# 

import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.plots import NeoDistancePlotter
import lsst.sims.maf.plots as plotters


# Set up the database connection
dbDir = '../../tutorials/'
opsdb = db.OpsimDatabase(os.path.join(dbDir,'enigma_1189_sqlite.db'))
outDir = 'NeoDistance_enigma'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.UniSlicer()
metric = metrics.PassMetric(metricName='NEODistances')
stacker = stackers.NEODistStacker()
stacker2 = stackers.EclipticStacker()
filters = ['u','g','r','i','z','y']


# Plotting the maximum distance an H=22 NEO could be detected in each LSST pointing (year 1). Sun is at the origin (0,0), Earth is at (0,1). Plotting everything in the earth-sun-neo plane (i.e., for the true 3-d distrubution, points would be rotated around the y-axis).
# 
# Note, this is a non-intuitive projection of the data. It is essentially plotting points in polar coordinates of solar elongation angle and NEO distance.  Thus, the planet orbits are illustrative, but only realistic if one restricts the observations to be in the ecliptic plane.
# 

for filterName in filters:
    bundle = metricBundles.MetricBundle(metric, slicer,
                                        'night < 365 and filter="%s"'%filterName,
                                        stackerList=[stacker,stacker2],
                                        plotDict={'title':'%s-band'%filterName},
                                        plotFuncs=[NeoDistancePlotter()])
    bgroup = metricBundles.MetricBundleGroup({filterName:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


# A perhaps slightly more intuitive way to plot the data is in rotating ecliptic coordinates.  Here, opposition is at the center of the plot and the ecliptic is the center horizontal line.
# 

metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
sql = ''
plotDict = {'rot':(180,0,0)}
bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker], plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# Loop through each filter and see the median and max neo dist
metric = metrics.MedianMetric('MaxGeoDist')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
stacker2 = stackers.NEODistStacker()
plotDict = {'rot':(180,0,0)}
for i,filterName in enumerate(filters):
    sql = 'filter ="%s"'%filterName
    bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
    bgroup = metricBundles.MetricBundleGroup({filterName:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


# All observations
metric = metrics.MedianMetric('MaxGeoDist')
metric2 = metrics.MaxMetric('MaxGeoDist')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
stacker2 = stackers.NEODistStacker()
plotDict = {'rot':(180,0,0)}
bDict = {}
sql = ''
bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
bDict[0]=bundle
bundle2 = metricBundles.MetricBundle(metric2, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
bDict[1] = bundle2
bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


metric = metrics.MaxMetric('solarElong')
metric2 = metrics.MinMetric('solarElong')
slicer = slicers.HealpixSlicer(nside=64)
sql = ''
bundle = metricBundles.MetricBundle(metric, slicer,sql)
bDict = {0:bundle}
bundle = metricBundles.MetricBundle(metric2, slicer,sql)
bDict[2] = bundle
metric = metrics.MedianMetric('solarElong')
bundle = metricBundles.MetricBundle(metric, slicer,sql)
bDict[1]= bundle
bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)





# Let's run a Monte Carlo with a new transient metric that computes some agregate of transient properties
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


class TransientMonteMetric(metrics.BaseMetric):
    """
    Generate a population of transient objects and see what fraction are detected
    """
    
    def __init__(self, metricName='TransientMonteMetric', mjdCol='expMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', duration=1., **kwargs):
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(TransientMonteMetric, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol],
                                                  units='Fraction Detected',
                                                  metricName=metricName, **kwargs)
        # Properties of the transient object
        self.duration = duration
        
    def lightCurve(self, t, t0, m_r_0):
        """
        Let's say this is a simple object that pops up and linearly decays for 10 days
        
        t0:  mjd of initial detection
        m_r_0: initial r-band brightness
        times: array of mjd
        """
        
        good = np.where( (t >= t0) & (t <= t0+self.duration))
        mags = np.zeros(t.size, dtype=float)
        mags.fill(300.) #  Set to very faint by default
        mags[good] = m_r_0
        return mags
        
    def run(self,  dataSlice, slicePoint=None):
        
        objRate = 0.7 # how many go off per day
        # Define the magnitude distribution
        timeSpan = dataSlice[self.mjdCol].max() - dataSlice[self.mjdCol].min()
        # decide how many objects we want to generate
        nObj = np.random.poisson(timeSpan*objRate)
        t0s = np.random.rand(nObj)*timeSpan
        m0s = np.random.rand(nObj)*2.+20.
        t = dataSlice[self.mjdCol] - dataSlice[self.mjdCol].min()
        detected = np.zeros(t0s.size)
        # Loop though each generated transient and decide if it was detected
        for i,t0 in enumerate(t0s):
            lc = self.lightCurve(t, t0, m0s[i])
            detectTest = dataSlice[self.m5Col] - lc
            if detectTest.max() > 0:
               detected[i] += 1
        # Return the fraction of transients detected
        return float(np.sum(detected))/float(nObj)
    


runName = 'enigma_1189'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'TransientsMonte'
resultsDb = db.ResultsDb(outDir=outDir)


metric = TransientMonteMetric()
sql = 'night < 365'
slicer = slicers.HealpixSlicer(nside=8)
bundle = metricBundles.MetricBundle(metric, slicer, sql, runName=runName)
bundleList = [bundle]
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)


bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)





# # TransientAsciiMetric
# 
# This notebook demonstrates how to use the TransientAsciiMetric, which lets you use an ascii file to specify a lightcurve template, and then either calculate the likelihood of detecting a transient with that lightcurve or generate the actual lightcurve datapoints.
# 

#    import maf packages
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots

from mafContrib import TransientAsciiMetric


# First do some bookkeeping: connect to the opsim database and set up the output directory and resultsDb.
# 

# Set the database and query
runName = 'minion_1016'
opsdb = db.OpsimDatabase('../' + runName + '_sqlite.db')

# Set the output directory
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir)


# Set up the transient ascii metric.  
# 
# Note that if `dataout` is True, the output at each slicePoint will be a dictionary of: 
#  'lcNumber', 'expMJD', 'epoch', 'filter', 'lcMag', 'SNR', 'detected' (a flag indicating whether that lightcurve met the detection criteria). 
# If `dataout` is False, the output at each slicePoint will be the likelihood of detecting a transient with this lightcurve.
# 

asciiLC = '2013ab_1.dat'
transMetric = TransientAsciiMetric(asciiLC, surveyDuration=1, 
                                   detectSNR={'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5},
                                   nPreT=0, preT=0, nFilters=0, filterT=None, nPerLC=0, peakOffset=0,
                                   dataout=True)


# Use the metric to generate a tightly sampled lightcurve, to illustrate what the lightcurve looks like.
filterNames = ['u', 'g', 'r', 'i', 'z', 'y']
colors = {'u': 'k', 'g': 'cyan', 'r': 'g', 'i': 'r', 'z': 'y', 'y': 'orange'}
times = np.arange(0, transMetric.transDuration, 0.5)
lc = {}
for f in filters:
    lc[f] = transMetric.make_lightCurve(times, np.array([f]*len(times)))

plt.figure()
for f in filterNames:
    plt.plot(times, lc[f], color=colors[f], label=f)
plt.ylim(23, 17)
plt.xlabel('Epoch (days)')
plt.ylabel('Magnitude')
plt.legend(fontsize='smaller', numpoints=1)


# Set up the slicer and sql constraint, assign them all to a metricBundle.
# 

# Slicer - we just want to look at what the full lightcurve output looks like, so choose a few representative points.
# With the UserPointsSlicer, you can set ra/dec for the places you want to evaluate.
# These ra/dec pairs are 1 DD field and 3 WFD fields.
ra = np.array([0.600278, 1.284262, 1.700932, 1.656778])
dec = np.array([-0.088843, 0.00327, -0.65815, -0.323526])
slicer = slicers.UserPointsSlicer(ra, dec)

# SQL constraint.
# select the of the survey that you want to run 
year = 9
sqlconstraint = 'night between %f and %f '% ((365.25*year,365.25*(year+1)))

lightcurve_metric = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, runName=runName)


# run the metric
bgroup = metricBundles.MetricBundleGroup({0: lightcurve_metric}, opsdb, 
                                         outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


# Plot each of the lightcurves created by setting off a set of back-to-back transients at each of the slicePoints. Note that not every slicePoint is sampled equally in time, especially since we only looked at one year. 
# 

for i, data in enumerate(lightcurve_metric.metricValues):
    for lcN in np.unique(data['lcNumber']):
        match = np.where(data['lcNumber'] == lcN)
        plt.figure()
        epoch = data['epoch'][match]
        mjd = data['expMJD'][match]
        mags = data['lcMag'][match]
        filters = data['filter'][match]
        for f in filterNames:
            filtermatch = np.where(filters == f)
            plt.plot(times - epoch[0] + mjd[0], lc[f], color=colors[f])
            plt.plot(mjd[filtermatch], mags[filtermatch], 'o', color=colors[f], label=f)
        plt.ylim(plt.ylim()[::-1])
        plt.xlim(times[0] - epoch[0] + mjd[0] - 2, times[0] - epoch[0] + mjd[0] + 192)
        plt.legend(ncol = 2, loc = (.8,.8), numpoints=1, fontsize='smaller') 
        plt.xlabel('MJD')
        plt.ylabel('Mags')
        plt.title('Field %d at %f/%f, lightcurve %d' % (i, np.degrees(slicer.slicePoints['ra'][i]), 
                                                        np.degrees(slicer.slicePoints['dec'][i]), lcN))
        plt.show()


# Now let's look at the metric, with dataout = False. In this case, it calculates the likelihood of detecting a transient at each slicePoint. We can also add some additional detection criteria.
# 

metric = TransientAsciiMetric(asciiLC, surveyDuration=1, 
                              detectSNR={'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5},
                              nPreT=3, preT=5, nFilters=3, filterT=30, nPerLC=2, peakOffset=0,
                              dataout=False)
slicer = slicers.HealpixSlicer(nside=16)
year = 8
sqlconstraint = 'night between %d and %d' % (365.25 * year, 365.25 * (year + 2))

plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
plotDict = {'colorMin': 0, 'colorMax': 1}

summaryMetrics = [metrics.MeanMetric(), metrics.RmsMetric()]

bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                    runName=runName, summaryMetrics=summaryMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)


bundlegroup = metricBundles.MetricBundleGroup({0: bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bundlegroup.runAll()


bundlegroup.plotAll(closefigs=False)


print bundle.summaryValues





# This notebook looks at detection of a simple SN Ia model in various survey strategies.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils


# We look at year=9 at most of the galactic plane etc. work is completed
# 

year = 9


# This is a simple SN Ia, z=0.5, stretch=1, colour=0 event from SiFTO. AB mAGS. These peak mags are in DES filters.
# 
# At z=0.5, the rise is $18d*(1+z)=27$ and we want to follow the event for about 20 days post max in the rest-frame. But we will never detect in the first few days so we just say we care about the 10 days prior to peak ie a tise of 15d observer frame.
# 
# We mock-up the rise as 2 mags over peak time, and the fall as 1.4 mags over 30 observer days in r at z=0.5
# 

# Trying to make a type Ia-like 
peaks = {'uPeak':25.9, 'gPeak':23.6, 'rPeak':22.6, 'iPeak':22.7, 'zPeak':22.7,'yPeak':22.8}

colors = ['b','g','r','purple','y','magenta','k']
filterNames = ['u','g','r','i','z','y']

peakTime = 15
transDuration = peakTime+30 # Days
transMetric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0, 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=1, **peaks)


times = np.arange(0.,transDuration*2,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([35,18])
plt.legend()


# ### What fraction are detected at least once in any filter?
# 

# Pick a slicer
nside = 64
slicer = slicers.HealpixSlicer(nside=nside)
pixelArea = hp.nside2pixarea(nside, degrees=True) # in sq degrees
surveyDuration = 1. # year, since we are selection only one year in the SQL

summaryMetrics = [metrics.MedianMetric(), metrics.SumMetric()]
# Configure some metrics
metricList = []
# What fraction are detected at least once?
metricList.append(transMetric)


# Set the database and query
runName = 'enigma_1189'
sqlconstraint = 'night between %f and %f' % ((365.25*year,365.25*(year+1)))
bDict={}
for i,metric in enumerate(metricList):
    bDict[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)


# #### NOTE - change your path and/or opsim database here
# 

opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)


bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


for key in bDict:
    bDict[key].computeSummaryStats(resultsDb=resultsDb)
    print bDict[key].metric.name, bDict[key].summaryValues


# Now to try and compute the total number of SN detected
snRate = 0.1 #XXX--TOTALLY MADE UP NUMBER.  SNe/yr/sq Deg
sneN = bDict[0].summaryValues['Sum']*snRate*pixelArea*surveyDuration
print 'Total number of SN detected = %f' % sneN


# ### What fraction are detected at least 6 times in one of g r i z, 3 in first half, 3 in second half
# 

transMetric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30., 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=1, 
                                      nFilters=3, nPrePeak=3, nPerLC=2, **peaks)

sqlconstraint = '(filter="r" or filter="g" or filter="i" or filter="z") and night between %f and %f' % (365.25*year,365.25*(year+1))
transBundle = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)


bgroup = metricBundles.MetricBundleGroup({0:transBundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


sneN = transBundle.summaryValues['Sum']*snRate*pixelArea*surveyDuration
print 'Total number of SN detected = %f' % sneN


# 
# Keywords that we need to add:
# 
# * Total number of points
# * Gap time to demand between points in the same filter to count them as independent
# 

# **Let's try that again, but select only the Deep Drilling Fields**
# 

transMetric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30., 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=1, 
                                      nFilters=3, nPrePeak=3, nPerLC=2, **peaks)
propids, propTags = opsdb.fetchPropInfo()
sqlDD = utils.createSQLWhere('DD', propTags)
sqlconstraint = sqlDD+' and '+ '(filter="r" or filter="g" or filter="i" or filter="z") and night between %f and %f' % (365.25*year,365.25*(year+1))
transBundle = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)
bgroup = metricBundles.MetricBundleGroup({0:transBundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


sneN = transBundle.summaryValues['Sum']*snRate*pixelArea*surveyDuration
print 'Total number of SN detected = %f' % sneN


# ### What fraction are detected at least 9 times in one of g r i z, 3 in each third
# 

transMetric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30., 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=1, 
                                      nFilters=3, nPrePeak=3, nPerLC=3, **peaks)

sqlconstraint = '(filter="r" or filter="g" or filter="i" or filter="z") and night between %f and %f'% (365.25*year,365.25*(year+1))
transBundle = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)


bgroup = metricBundles.MetricBundleGroup({0:transBundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


sneN = transBundle.summaryValues['Sum']*snRate*pixelArea*surveyDuration
print 'Total number of SN detected = %f' % sneN





# Generate a light curve that approximates a z-2.0 superluminous supernova. Use the transient metric to see how well it gets observed.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db


# Trying to make a z=2 superluminous 
peaks = {'uPeak':35., 'gPeak':35., 'rPeak':23.3, 'iPeak':22.8, 'zPeak':22.8,'yPeak':22.8}


colors = ['b','g','r','purple','y','magenta','k']
filterNames = ['u','g','r','i','z','y']

peakTime = 90.
transDuration = peakTime+90. # Days
transMetric = metrics.TransientMetric(riseSlope= -1./20., declineSlope=1./20., 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=2, 
                                      nFilters=3, nPrePeak=3, nPerLC=2, nPhaseCheck=5, **peaks)


times = np.arange(0.,transDuration*2,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([35,18])
plt.legend()


# Pick a slicer
slicer = slicers.HealpixSlicer(nside=64)

summaryMetrics = [metrics.MedianMetric()]


# Set the database and query
runName = 'enigma_1189'
sqlconstraint = 'night < %f' %(365.25*2)


opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)

bundle = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)








# ## Evaluate the number of observatories available to follow up an event##
# 
# This demonstrates the nFollowStacker, which adds a column indicating how many observatories would be available to follow up an event that occured during that visit (for a user-specified timestep). 
# You can then calculate statistics on this new column - such as the average number of observatories available for all visits to a particular region on the sky. 
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from mafContrib import NFollowStacker


# Set the database and query
runName = 'enigma_1189'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'FollowUp'
resultsDb = db.ResultsDb(outDir=outDir)


sqlconstraint = 'night < 30'

slicer = slicers.HealpixSlicer(nside=64)
metric = metrics.MeanMetric('nObservatories')
stackerList = [NFollowStacker(minSize=6.5)]
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# Change it up to be in alt,az
slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=True)
plotDict = {'rot':(0,90,0)}
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# Let's see what happens with a 24-hour follow-up window
slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=True)
plotDict = {'rot':(0,90,0)}
stackerList = [NFollowStacker(minSize=6.5, timeSteps=np.arange(0,26,2))]
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


# ## Figure of Merit Prototype for Section 4.3: Detection of pre-SN variability for Galactic Supernova science case ##
# 

# ** 2016-01-30 (WIC): ** Read in previously-computed metrics and combine their values to come up with a prototype figure of merit for the "Galactic Supernova" science case. 
# 
# Making the assumption that a core-collapse supernova would be detected directly by neutrino experiments, triggering intensive followup at all wavelengths, I adopt the detectability of an $\sim$ 8 mag nova-like outburst *before* the supernova (parameters similar to SN2010mc; Ofek et al. 2013 Nature) as a figure of merit for LSST's contribution to this science. 
# 
# I assume that the probability of a supernova going off at a given location scales with the number of stars along the line of sight (out to a fiducial distance, which I set at 80 kpc to avoid much of the halo and the Magellanic Clouds in the figure of merit). Multiplying (fraction detected) $\times$ (star count) at each position then yields a rough figure of merit for the use of LSST to detect pre-Supernova variability.
# 
# **Implementation:**
# 
# I use a slightly modified version of Mike Lund's Starcounts maf_contrib module to produce the stellar density out to some fiducial distance, then use Peter Yoachim's TransientMetric to assess the fraction of *pre*-Supernova outbursts that might be detected. The intention here is to provide a simple example using pre-computed metrics into a figure of merit to provoke the community of LSST-interested Milky Way observers. 
# 
# Imports inside subsections are needed only to run the cells in those subsections.
# 

# For reference, here are the parameters used to simulate the transients:
#peaks = {'uPeak':11, 'gPeak':9, 'rPeak':8, 'iPeak':7, 'zPeak':6,'yPeak':6}
#colors = ['b','g','r','purple','y','magenta','k']
#filterNames = ['u','g','r','i','z','y']
## Timing parameters of the outbursts
#riseSlope = -2.4
#declineSlope = 0.05  # following Ofek et al. 2013
#transDuration = 80.
#peakTime = 20.

# relevant parameter for the TransientMetric:
# nPhaseCheck=20

# Additionally, all filters were used (passed as **peaks to the TransientMetric).


get_ipython().magic('matplotlib inline')


import numpy as np
import time


# Some colormaps we might use
import matplotlib.cm as cm


# Capability to load previously-computed metrics, examine them
import lsst.sims.maf.metricBundles as mb

# plotting (to help assess the results)
import lsst.sims.maf.plots as plots


# ### Slightly modified version of the Starcounts metric ###
# 

# The example CountMetric provided by Mike Lund seems to have the column indices for coords
# hardcoded (which breaks the examples I try on my setup). This version finds the co-ordinates by 
# name instead. First the imports we need:
# import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from mafContrib import starcount 


class AsCountMetric(BaseMetric):

    """
    WIC - Lightly modified copy of Mike Lund's example StarCounts metric in sims_maf_contrib. 
    Accepts the RA, DEC column names as keyword arguments. Docstring from the original:
    
    Find the number of stars in a given field between distNear and distFar in parsecs. 
    Field centers are read from columns raCol and decCol.
    """
    
    def __init__(self,**kwargs):
        
        self.distNear=kwargs.pop('distNear', 100)
        self.distFar=kwargs.pop('distFar', 1000)
        self.raCol=kwargs.pop('raCol', 'ra')
        self.decCol=kwargs.pop('decCol', 'dec')
        super(AsCountMetric, self).__init__(col=[], **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        sliceRA = np.degrees(slicePoint[self.raCol])
        sliceDEC = np.degrees(slicePoint[self.decCol])
        return starcount.starcount(sliceRA, sliceDEC, self.distNear, self.distFar)


# ### Run this out to a fiducial distance ###
# 

# (This section can be ignored if the Counts metric was already computed.) We do not do plots or summary statistics, since we should be able to evaluate those later on.
# 

distNear=10.
distFar = 8.0e4  # Get most of the plane but not the magellanic clouds 


import lsst.sims.maf.slicers as slicers


import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db


slicer = slicers.HealpixSlicer(nside=64)


metricCount=AsCountMetric(distNear=distNear, distFar=distFar)
metricList = [metricCount]


runName1092 = 'ops2_1092'
sqlconstraintCount = 'filter = "r" & night < 1000'  # Assume everywhere visited once in three days...
bDict1092={}
for i,metric in enumerate(metricList):
    bDict1092[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                          runName=runName1092)
opsdb1092 = db.OpsimDatabase(runName1092 + '_sqlite.db')
outDir1092 = 'TestCountOnly1092'
resultsDb1092 = db.ResultsDb(outDir=outDir1092)


tStart = time.time()
bgroup1092 = metricBundles.MetricBundleGroup(bDict1092, opsdb1092, outDir=outDir1092,                                              resultsDb=resultsDb1092)
bgroup1092.runAll()
tPost1092 = time.time()
print "Time spent Counting 1092: %.3e seconds" % (tPost1092 - tStart)


# Ensure the output file actually got written...
get_ipython().system(' ls -l ./TestCountOnly1092/*npz')


# We will need the same counts information for enigma if we want to normalize by 
# total counts in the survey area. So let's run the above for enigma_1189 as well.
runName1189 = 'enigma_1189'
bDict1189={}
for i,metric in enumerate(metricList):
    bDict1189[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                          runName=runName1189)
opsdb1189 = db.OpsimDatabase(runName1189 + '_sqlite.db')
outDir1189 = 'TestCountOnly1189'
resultsDb1189 = db.ResultsDb(outDir=outDir1189)

tStart = time.time()
bgroup1189 = metricBundles.MetricBundleGroup(bDict1189, opsdb1189, outDir=outDir1189,                                              resultsDb=resultsDb1189)
bgroup1189.runAll()
tPost1189 = time.time()
print "Time spent Counting 1189: %.3e seconds" % (tPost1189 - tStart)


get_ipython().system(' ls ./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz')


# ### Loading the pre-computed metrics: Counts and Transients ###
# 

pathCount='./TestCountOnly1092/ops2_1092_AsCount_r_HEAL.npz'
pathTransient='Transients1092Like2010mc/ops2_1092_Alert_sawtooth_HEAL.npz'


#Initialize then load
bundleCount = mb.createEmptyMetricBundle()
bundleTrans = mb.createEmptyMetricBundle()

bundleCount.read(pathCount)
bundleTrans.read(pathTransient)


# Set a mask for the BAD values of the transient metric
bTrans = (np.isnan(bundleTrans.metricValues)) | (bundleTrans.metricValues <= 0.)
bundleTrans.metricValues.mask[bTrans] = True


# ### Normalizing the counts metric for OpSim runs with different total area ###
# 

# The two runs compared, enigma_1189 and ops1092, cover different total areas. Before proceeding further, we read in the counts metric for enigma_1189 in order to compare the total number of stars.
# 

# Read in the stellar density for 1189 so that we can compare the total NStars...
pathCount1189='./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz'
bundleCount1189 = mb.createEmptyMetricBundle()
bundleCount1189.read(pathCount1189)


# Do the comparison
nTot1092 = np.sum(bundleCount.metricValues)
nTot1189 = np.sum(bundleCount1189.metricValues)
print "Total NStars - ops2_1092: %.3e - enigma_1189 %.3e" % (nTot1092, nTot1189)


# There is a difference between the total NStars (out to 80kpc) but it's on the order of 1%, even though enigma_1189 covers more of the plane exterior to the Sun. This is probably an expression of the dominance of the central regions of the Milky Way galaxy in the density model.
# 
# To express the figure of merit as a fraction of detected pre-outbursts that would be detected by LSST, we divide the density metric for each run by its own total NStars. Thus the figure of merit will take the range $0 - 1$.
# 

# ### Multiply the metric value sets together for ops2_1092 ###
# 

# I think the best way to ensure that the result has all the pieces needed by maf is to create a copy of one of the metrics, then re-read the metric in. Since we don't trust negative values of the transient metric, use the mask for non-negative transient values too.
# 

bundleProc = mb.createEmptyMetricBundle()
bundleProc.read(pathTransient)


# Set the mask
bundleProc.metricValues.mask[bTrans] = True


# Multiply the two together, normalise by the total starcounts over the survey
bundleProc.metricValues = (bundleCount.metricValues * bundleTrans.metricValues) 
bundleProc.metricValues /= np.sum(bundleCount.metricValues)


bundleProc.metric.name = '(sawtooth alert) x (counts) / NStars_total'


FoM1092 = np.sum(bundleProc.metricValues)
print "FoM 1092: %.2e" % (FoM1092)


# ### Multiply the two metrics together for enigma_1189 ###
# 

pathCount1189='./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz'
pathTrans1189='./Transients1189Like2010mc/enigma_1189_Alert_sawtooth_HEAL.npz'
bundleCount1189 = mb.createEmptyMetricBundle()
bundleTrans1189 = mb.createEmptyMetricBundle()

bundleCount1189.read(pathCount1189)
bundleTrans1189.read(pathTrans1189)
bTrans1189 = (np.isnan(bundleTrans1189.metricValues)) | (bundleTrans1189.metricValues <= 0.)
bundleTrans1189.metricValues.mask[bTrans1189] = True


# Load 1189-like metric bundle and replace its values with processed values
bundleProc1189 = mb.createEmptyMetricBundle()
bundleProc1189.read(pathTrans1189)
bundleProc1189.metricValues.mask[bTrans1189] = True

bundleProc1189.metricValues = (bundleCount1189.metricValues * bundleTrans1189.metricValues) 
bundleProc1189.metricValues /= np.sum(bundleCount1189.metricValues)
bundleProc1189.metric.name = '(sawtooth alert) x (counts) / NStars_total'


FoM1189 = np.sum(bundleProc1189.metricValues)
print FoM1189


# ### Our "result:" f = (Sawtooth alert) $\times$ ($\rho_{\ast}$) for the two runs ###
# 

# Print the sum total of our f.o.m. for each run
print "FOM for ops2_1092: %.3f" % (FoM1092)
print "FOM for enigma_1189: %.3f" % (FoM1189)


# ### Plot the processed metrics ###
# 

# Same plot information as before:
plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProc.setPlotDict(plotDictProc)
bundleProc.setPlotFuncs(plotFuncs)

plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProc1189.setPlotDict(plotDictProc)
bundleProc1189.setPlotFuncs(plotFuncs)





bundleProc.plot(savefig=True)
bundleProc1189.plot(savefig=True)





# ### Plot the two input metrics ###
# 

# Plot just the spatial map and the histogram for the two. Use different colormaps for each.
#plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
bundleTrans.setPlotFuncs(plotFuncs)
bundleCount.setPlotFuncs(plotFuncs)


# Use a different colormap for each so we can tell them apart easily...
plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCount.setPlotDict(plotDictCount)
bundleTrans.setPlotDict(plotDictTrans)


plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCount1189.setPlotDict(plotDictCount)
bundleTrans1189.setPlotDict(plotDictTrans)
bundleTrans1189.setPlotFuncs(plotFuncs)
bundleCount1189.setPlotFuncs(plotFuncs)


bundleCount.plot()
bundleTrans.plot()


bundleCount1189.plot()
bundleTrans1189.plot()





# # Test many dither patterns#
# 
# This notebook is intended to demonstrate the various 'stock' dithering patterns available through MAF and then show how to use these dithering patterns to evaluate metrics under different dithering schemes.
# 
# This also provides an extended example of MAF [`Stackers`](https://confluence.lsstcorp.org/display/SIM/MAF+Stackers) which allow the addition of 'virtual' columns to the Opsim data, by generating these additional columns on the fly. 
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


import lsst.sims.maf.db as db
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots


# ---
# 
# # Demonstrate the dithering patterns#
# 
# Here is an overview of the available dithering patterns. First - a list of the 'stock' dithering patterns available from MAF, along with their docstrings.
# 

# Grab all the dither stackers from the stock stackers (rather than all stackers)
import inspect
ditherStackerList = []
for name, s in inspect.getmembers(stackers):
    if inspect.isclass(s):
        if 'Dither' in s.__name__:
            ditherStackerList.append(s)


# Print their docstrings
for s in ditherStackerList:
    print '-- ', s.__name__, ' --'
    print s.__doc__
    print '    Generates columns named', s().colsAdded
    print ' '


# Note that the 'nightly' versions of the stackers apply the same dither to all observations of a particular field taken in a single night (but may vary from one field to the next), while the versions without 'nightly' apply a separate dither offset to each visit.
# 

# **Visualize the dither options:**
# Just write a bit of code to make nice figures of the dithering patterns.
# 

import numpy as np


def plotDither(ditherStacker, nvisits=1000, addPolygon=True):
    # Set up some 'data' on a single pointing to dither
    fieldIds = np.ones(nvisits, int)
    fieldRA = np.zeros(nvisits, float) + np.radians(10.0)
    fieldDec = np.zeros(nvisits, float) 
    night = np.arange(0, nvisits/2.0, 0.5)
    night = np.floor(night) 
    simdata = np.core.records.fromarrays([fieldIds, fieldRA, fieldDec, night], 
                                         names = ['fieldID', 'fieldRA', 'fieldDec', 'night'])
    # Apply the stacker. 
    simdata = ditherStacker.run(simdata)
    
    fig = plt.figure()
    plt.axis('equal')
    # Draw a point for the center of the FOV.
    x = np.degrees(simdata['fieldRA'][0])
    y = np.degrees(simdata['fieldDec'][0])
    plt.plot(x, y, 'g+')
    # Draw a circle approximately the size of the FOV.
    stepsize = np.pi/50.
    theta = np.arange(0, np.pi*2.+stepsize, stepsize)
    radius = 1.75
    plt.plot(radius*np.cos(theta)+x, radius*np.sin(theta)+y, 'g-')
    # Add the inscribed hexagon
    nside = 6
    a = np.arange(0, nside)
    xCoords = np.sin(2*np.pi/float(nside)*a + np.pi/2.0)*radius + x
    yCoords = np.cos(2*np.pi/float(nside)*a + np.pi/2.0)*radius + y
    xCoords = np.concatenate([xCoords, np.array([xCoords[0]])])
    yCoords = np.concatenate([yCoords, np.array([yCoords[0]])])
    plt.plot(xCoords, yCoords, 'b-')
    # Draw the dithered pointings.
    x = np.degrees(simdata[s.colsAdded[0]])
    y = np.degrees(simdata[s.colsAdded[1]])
    plt.plot(x, y, 'k-', alpha=0.2)
    plt.plot(x, y, 'r.')
    plt.title(s.__class__.__name__)


# Look at the default dither patterns.
# 

for ditherStacker in ditherStackerList:
    s = ditherStacker()
    plotDither(s)


# Example comparison of spiral dither with one offset applied per field per night vs one offset per field per visit (the timescale on which the dithers are applied is different). 
# 

s = stackers.SpiralDitherFieldVisitStacker()
plotDither(s, nvisits=30)
s = stackers.SpiralDitherFieldNightStacker()
plotDither(s, nvisits=30)


# Look at the dither patterns after some additional configuration (such as setting the random seed or changing the maxDither or allowing the points to wander outside the inscribed hexagon). 
# 

s = stackers.RandomDitherFieldVisitStacker(maxDither=1.75, inHex=False)
plotDither(s)
s = stackers.RandomDitherFieldVisitStacker(maxDither=1.75, inHex=True)  #inHex is true by default
plotDither(s)
s = stackers.RandomDitherFieldVisitStacker(maxDither=0.5)
plotDither(s)


s = stackers.RandomDitherFieldVisitStacker(randomSeed=253)
plotDither(s, nvisits=200)
s = stackers.RandomDitherFieldVisitStacker(randomSeed=100)
plotDither(s, nvisits=200)


s = stackers.SpiralDitherFieldVisitStacker(nCoils=3)
plotDither(s)
s = stackers.SpiralDitherFieldVisitStacker(nCoils=6)
plotDither(s)
s = stackers.SpiralDitherFieldVisitStacker(nCoils=6, inHex=False)
plotDither(s)


# ---
# # Using the dither stackers #
# 

# We'll set up with a very simple metricbundle to just count the number of visits at each RA/Dec point in a healpix grid. To extend this to your own work, you could simply swap the metric to something more relevant. To speed up the notebook, run this example with a smaller nside (64?) to use a slightly lower resolution for the healpix slicer and only look at the first two years of visits. 
# 
# First, a metricbundle with no dithering.
# 

nside = 128
metric = metrics.CountMetric('expMJD')
slicer0 = slicers.HealpixSlicer(lonCol='fieldRA', latCol='fieldDec', nside=nside)  
#sqlconstraint = 'filter="r" and night<730' 
sqlconstraint = 'filter="r"'


myBundles = {}
myBundles['no dither'] = metricBundles.MetricBundle(metric, slicer0, sqlconstraint, 
                                                    runName='enigma_1189', metadata='no dither')


# Next, a metricBundle using the built-in opsim hex-dither dithering. This dither pattern looks like the 'sequential dither pattern' above, but each offset is applied to *all* visits during a night (for all fields). The sequence along the vertices is controlled by the night %217 (the number of vertices), so some fields may never be offset at a particular vertex.
# 

# To use a different dither pattern to evaluate our metric, we simply change the values of ra/dec that the slicer is using to match visits to healpix grid points. 
# 

# ditheredRA and ditheredDec correspond to the stock opsim dither pattern
slicer1 = slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec', nside=nside)
myBundles['hex dither'] = metricBundles.MetricBundle(metric, slicer1, sqlconstraint, runName='enigma_1189', 
                                                    metadata = 'hex dither')


# If we want to use a stacker in its 'default' state, we can let MAF handle this internally, and we don't have to explicitly instantiate our stacker. We do have to use the exact column names that each stacker adds to the simData output (which you can see in the list of stackers and their docstrings, above). 
# 

slicer2 = slicers.HealpixSlicer(lonCol='randomDitherFieldVisitRa', latCol='randomDitherFieldVisitDec', nside=nside)
myBundles['random dither'] = metricBundles.MetricBundle(metric, slicer2, sqlconstraint, runName='enigma_1189',
                                                       metadata='random dither')


# On the other hand, if we want to customize a stacker, we must explicitly instantiate it and pass it to the metricBundle.
# 

stackerList = [stackers.SpiralDitherFieldNightStacker(nCoils=7)]
slicer3 = slicers.HealpixSlicer(lonCol='spiralDitherFieldNightRa', latCol='spiralDitherFieldNightDec', nside=nside)
myBundles['spiral dither'] = metricBundles.MetricBundle(metric, slicer3, sqlconstraint, 
                                                       stackerList=stackerList, runName='enigma_1189',
                                                       metadata='spiral dither')


# Now, as before, we run this dictionary of metricBundles, using the MetricBundleGroup (note the sqlconstraint is the same for all these bundles). 
# 

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'dither_test'
resultsDb = db.ResultsDb(outDir=outDir)
bgroup = metricBundles.MetricBundleGroup(myBundles, opsdb, outDir=outDir, resultsDb=resultsDb)


# (If we had previously calculated these metrics, but wanted to read them back in here to regenerate the plots, we'd do something like the next cell.)
# 

#import os
#for b in bgroup.bundleDict.itervalues():
#    filename = os.path.join(outDir, b.fileRoot) + '.npz'
#    b.read(filename)


bgroup.runAll()


# We could just plot the individual skymap/histogram/power spectrum plots using bgroup.plotAll(), but here we'd prefer to get the individual skymaps, then a histogram and power spectrum using all the different metricBundles together.  In order to do that, we'll use the `PlotHandler` directly.
# 

ph = plots.PlotHandler(outDir=outDir, resultsDb=resultsDb)


for mB in myBundles.itervalues():
    plotDict = {'xMin':0, 'xMax':300, 'colorMin':0, 'colorMax':300}
    mB.setPlotDict(plotDict)
    mB.plot(plotFunc=plots.HealpixSkyMap, plotHandler=ph)


ph.setMetricBundles(myBundles)
# We must set a series of plotDicts: one per metricBundle. 
#  because we don't explicitly set the colors, they will be set randomly. 
plotDict = {'binsize':1, 'xMin':0, 'xMax':350}
ph.plot(plots.HealpixHistogram(), plotDicts=plotDict)


# Plot some close-ups.
import healpy as hp
for mB in myBundles.itervalues():
    hp.cartview(mB.metricValues, lonra=[70, 100], latra=[-30, -0], min=150., max=300., 
            flip='astro', title=mB.metadata)


# For some purposes, examining the angular power spectra (and the power in the power spectra) is important. In practice, this likely requires some masking of the metric data (to remove ringing near the edges of the dithering and from the lower number of visits/depth from the areas in the survey outside the WFD region). But, after this masking is done (see the science notebook [GalaxyCounts](../science/static/GalaxyCounts.ipynb) as an example), generating plots of the power spectra labelled with total power could be done as follows.
# 

summaryMetrics = [metrics.TotalPowerMetric()]
for mB in myBundles.itervalues():
    mB.setSummaryMetrics(summaryMetrics)
    mB.computeSummaryStats()
    plotDict = {'label':'%s : %g' %(mB.metadata, mB.summaryValues['TotalPower'])}
    mB.setPlotDict(plotDict)


ph.plot(plots.HealpixPowerSpectrum(), plotDicts={'legendloc':(1, 0.3)})





# A metric that checks how many nights we have color information on a potential transient
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


class NnightsWColor(metrics.BaseMetric):
    """See how many nights a spot is observed at least twice in one filter and once in another.
    
    Parameters
    ----------
    n_in_one : int (2)
       The number of observations in a single filter to require
    n_filts : int (2)
       The number of unique filters to demand
    filters : list 
       The list of acceptable filters
    
    """
    def __init__(self, metricName='', mjdCol='expMJD',
                 filterCol='filter', nightCol='night', n_in_one=2, n_filts=2, 
                 filters=None, **kwargs):
        if filters is None:
            self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        else:
            self.filters = filters
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.filterCol = filterCol
        self.n_in_one = n_in_one
        self.n_filts = n_filts
        super(NnightsWColor, self).__init__(col=[self.mjdCol, self.nightCol,
                                                 self.filterCol],
                                            units='# Nights',
                                            metricName=metricName,
                                            **kwargs)
        
    def run(self,  dataSlice, slicePoint=None):
        
        night_bins = np.arange(dataSlice[self.nightCol].min()-.5, dataSlice[self.nightCol].max()+2.5, 1)
        all_obs = np.zeros((night_bins.size-1, len(self.filters)))
        for i, filtername in enumerate(self.filters):
            good = np.where(dataSlice[self.filterCol] == filtername)
            hist, edges = np.histogram(dataSlice[self.nightCol][good], bins=night_bins)
            all_obs[:,i] += hist
        # max number of observations in a single filter per night
        max_collapse = np.max(all_obs, axis=1)
        all_obs[np.where(all_obs > 1)] = 1
        # number of unique filters per night
        n_filt = np.sum(all_obs, axis=1)
        good = np.where((max_collapse >= self.n_in_one) & (n_filt >= self.n_filts))[0]
        return np.size(good)


runName = 'minion_1016'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'TransientsUPS'
resultsDb = db.ResultsDb(outDir=outDir)


metric = NnightsWColor()
sql = ''
slicer = slicers.HealpixSlicer()
bundle = metricBundles.MetricBundle(metric, slicer, sql, runName=runName)


bundleList = [bundle]
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)





# When observing a point source (star, quasar), it can be displaced by motion due to parallax or by differential chromatic refraction (it has a slightly different color than other objects in the field, and thus diffracts through the atmosphere slightly differently).
# 
# If one is not careful, the direction of parallax offset and the direction of DCR can align, and thus it is impossible to tell which effect is responsible for shifting the position of the star.  This metric uses stackers to compute the predicted offsets due to parallax and DCR.  Then, it tries to fit the offsets and checks for any correlation between the fitted parallax amplitude and DCR amplitude.  
# 
# The metric fits the RA and Dec offsets simultaneously.
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np


# Set up the database connection
opsdb = db.OpsimDatabase('astro_lsst_01_1000_sqlite.db')
outDir = 'astrometry_dcr'
resultsDb = db.ResultsDb(outDir=outDir)


sql = 'night < %i' % (365.25*5) 
slicer = slicers.HealpixSlicer(nside=8)
metricList = []
metricList.append(metrics.ParallaxDcrDegenMetric())
metricList.append(metrics.ParallaxDcrDegenMetric(rmag=24., SedTemplate='B', metricName='DCR-Degen-faint-B'))


bundleList = []
for metric in metricList:
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql))


bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


import healpy as hp
hp.mollview(np.abs(bundleList[0].metricValues))








# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# It demonstrates calculating a metric at a series of time periods. 
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


# Here, we'll look at computing the co-added depth in a single filter after year 1, 5, and 10
# 

filterName = 'r'
years = [1.,5.,10.]
nights = np.array(years)*365.25
sqls = ['filter = "%s" and night < %f' %(filterName, night) for night in nights]
print sqls


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'depths_test'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer()
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]
metric = metrics.Coaddm5Metric()
bgroupList = []
for year,sql in zip(years,sqls):
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bundle.plotDict['label'] = '%i' % year
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)
    


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


print 'year, mean depth, median depth'
for year,bundleGroup in zip(years,bgroupList):
    print year, bundleGroup.bundleDict[0].summaryValues['Mean'], bundleGroup.bundleDict[0].summaryValues['Median']


# Let's do a zoom-in and gnomic project for all three
# 

import healpy as hp
for year,bundleGroup in zip(years,bgroupList):
    hp.gnomview(bundleGroup.bundleDict[0].metricValues, rot=(0,-30), title='year %i'%year, 
                min=25.5, max=27.8, xsize=500,ysize=500)


# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# Example of a combination of metricBundles.
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


# Let's compare the depth in the r-band after 5 years, and the depth after 5 years when the seeing is better than 0.7 arcseconds
# 

filterName = 'r'
goodSeeing = 0.7
sqls = ['filter = "%s" and night < %f' % (filterName, 5.*365.25),
        'filter = "%s" and night < %f and finSeeing < %f'% (filterName, 5.*365.25, goodSeeing)]
print sqls


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'goodseeing_test'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer(lonCol='ditheredRA',latCol='ditheredDec')
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]

bgroupList = []
names = ['All Visits', 'Good Seeing']
for name,sql in zip(names, sqls):
    metric = metrics.Coaddm5Metric(metricName=name)
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


print 'name, mean depth, median depth'
for bundleGroup in bgroupList:
    print bundleGroup.bundleDict[0].metric.name, bundleGroup.bundleDict[0].summaryValues['Mean'], bundleGroup.bundleDict[0].summaryValues['Median']


# ## Example periodic star metric including crowding in the Monte Carlo trials ##
# 

# WIC 2015-01-03 - example metric where crowding uncertainty is incorporated within the Monte Carlo runs. 
# 
# In some (many?) cases where Monte Carlo is performed on a large number of fake lightcurves at a given slicePoint, with high-amplitude brightness changes (e.g. realizations of a dwarf nova outburst in crowded region; e.g. realizations of ellipsoidal variation in crowded region) it may be preferable to apply the crowding uncertainty separately to each each trial lightcurve within the Monte Carlo loop, since the crowding uncertainty is magnitude-dependent. 
# 
# It was not clear to me how to make this work with two separate metrics (e.g. how to combined the new CrowdingMetrics with the periodicStarMetric) so I refactored one of the crowding metrics (in this case Knut Olsen's tutorials/CrowdingMetric.ipynb) into a separate module that can be imported to the Metric itself. 
# 
# This Notebook gives an example. The crowding error is in module confusion.py with an example metric that uses it, in module sineMetricWithCrowd.py . Since I use knutago's version of crowding information, this also expects the list of luminosity functions and field information in the subdirectory "./lfs/" . (This can be tweaked as an argument to confusion.CrowdingSigma on initialization.) Downloading the directory containing this notebook and then unzipping the tarball should bring everything down that is specific to this Notebook (apart from the OpSim database file itself).
# 

# **OpSIM run chosen:** All the examples in this Notebook use the PanSTARRS-like OpSim run (ops2_1092) to isolate the effects of crowding on lightcurve recovery. The enigma_1189 still has observations in the Galactic Plane that are heavily front-loaded towards the beginning of the survey, which may lead to recovery drop-outs that are not due to crowding. 
# 

get_ipython().magic('matplotlib inline')


# Some standard items
import numpy as np
import time


import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


# the mag(5sig)-to-snr metric
from lsst.sims.maf.utils import m52snr


# Bring in the periodicStarMetric
import sineMetricWithCrowd as sM


# not needed for the metric, but needed for this notebook
import matplotlib.pyplot as plt
import confusion


# ## Illustration of crowding uncertainty vs magnitude ##
# 

# Here we take a single slice and generate a large-amplitude sinusoidal lightcurve to illustrate the impact of confusion uncertainty on the lightcurve. 
# 
# We then use the confusion.CrowdSigma object to estimate errors due to crowding, and apply them to each magnitude in the "trial" lightcurve. The end result of this subsection is a plot showing crowding and photometric errors along the lightcurve.
# 

# Pick coordinates - we use a certain window towards the inner Milky Way...
raDegPlot = 270.75
deDegPlot = -30.01

# We want to pull down the OpSim exposure information for this location. 
# First set up, ensuring the seeing information is passed through...
raPlot = np.radians(raDegPlot)
decPlot = np.radians(deDegPlot)

lCols = ['filter','fiveSigmaDepth','expMJD', 'finSeeing']
metricPlot = metrics.PassMetric(cols=lCols)
slicerPlot = slicers.UserPointsSlicer(raPlot, decPlot,                                       lonCol='ditheredRA', latCol='ditheredDec')
sqlPlot = ''
outDirPlot = 'crowdPullLC_1092'

# we need to reconnect separately to pass thru to output directory and db
dbFilePlot='ops2_1092_sqlite.db'
opsimdbPlot = utils.connectOpsimDb(dbFilePlot)
resultsDbPlot = db.ResultsDb(outDir=outDirPlot)

bundlePlot = metricBundles.MetricBundle(metricPlot,slicerPlot,sqlPlot)
bgPlot =  metricBundles.MetricBundleGroup({0:bundlePlot}, opsimdbPlot,
                                      outDir=outDirPlot, resultsDb=resultsDbPlot)


# ... and run:
tStartPlot = time.time()
bgPlot.runAll()
tEndPlot = time.time()
print "Time elapsed passing data through: %.2f minutes" % ((tEndPlot - tStartPlot)/60.)


# We want to simulate a large-amplitude variation. Pick parameters for large amplitude 
# variation visible by eye.
genPeriod = 900.
genPhase = 5.
genAmp = 3.
meanMag = 20.
magList = [meanMag for i in range(6)]


# Generate this variation. (Lines borrowed from peterY's periodicStarMetric)
inSlice = bundlePlot.metricValues.data[0]
mjdCol = 'expMJD'
filterCol = 'filter'

# perform the same data preparation as in the original metric
t = np.empty(inSlice.size, dtype=zip(['time','filter'],[float,'|S1']))
t['time'] = inSlice[mjdCol]-inSlice[mjdCol].min()
t['filter'] = inSlice[filterCol]

trueLC = sM.periodicStar(t, genPeriod, genPhase, genAmp, *magList)


# Now convert OpSim info into 1-sigma photometric uncertainty.
m5Col = 'fiveSigmaDepth'
snr = m52snr(trueLC,inSlice[m5Col])
sigmaPhot = 2.5*np.log10(1.+1./snr)


# Set up to estimate the crowding error at each point. For consistency, use the RA, DEC 
# that came down with the slice. I
raFirst = inSlice['ditheredRA'][0]
decFirst = inSlice['ditheredDec'][0]
PhotConfuse = confusion.CrowdingSigma(inSlice, magInput=trueLC, ra=raFirst, dec=decFirst)
PhotConfuse.getErrorFuncAndSeeing()


# for-loop for Monte Carlo at this slicePoint might be declared here.
# For now, let's just set a view of the "true" lightcurve
trialLC = trueLC


# Estimate the error due to crowding for this lightcurve. 
PhotConfuse.magSamples = np.copy(trialLC)
PhotConfuse.calcSigmaSeeingFromInterp()
sigmaCrowd = np.copy(PhotConfuse.sigmaWithSeeing)


# We now have the 1-sigma values for each point for photometric and 
# for crowding error. Turn them into perturbations for plotting later.
perturbPhot = np.random.randn(trialLC.size) * sigmaPhot
perturbCrowd = np.random.randn(trialLC.size) * sigmaCrowd


# at the moment we only have crowding information for r-band. Select that out now.
gR = np.where(t['filter'] == "r")

# Set a few views for convenience below
rSor = np.argsort(t[gR]['time'])
figTime = t[gR]['time'][rSor]
errPhot = perturbPhot[gR][rSor]
errCrowd = perturbCrowd[gR][rSor]
errBoth = np.sqrt(errPhot**2 + errCrowd**2)

figLCOrig = trialLC[gR][rSor]
figLCPhot = figLCOrig + errPhot
figLCBoth = figLCPhot + errCrowd


# Let's plot a couple of figures.
plt.figure(1, figsize=(10,6))
plt.clf()
plt.subplots_adjust(wspace=0.3, hspace=0.4)

plt.suptitle(r'$(\alpha, \delta)_{2000} \approx (%.1f^\circ, %.1f^\circ)$'              % (raDegPlot, deDegPlot), fontsize=14)

plt.subplot(222)
plt.errorbar(figTime, figLCBoth, errBoth, ls='None', color='g')
plt.plot(figTime, figLCBoth, 'go', markersize=3)
plt.plot(figTime, figLCOrig, 'k-', alpha=0.5)
plt.xlabel('Time elapsed (days)')
plt.ylabel('rMag')
plt.title('Photometric + Crowding')
plt.grid(which='both')
yRange = np.copy(plt.ylim())

plt.subplot(221)
plt.errorbar(figTime, figLCPhot, errPhot, ls='None')
plt.plot(figTime, figLCOrig, 'k-', alpha=0.5)
plt.xlabel('Time elapsed (days)')
plt.ylabel('rMag')
plt.title('Photometric error only')
plt.grid(which='both')
plt.ylim(yRange)

plt.subplot(223)
plt.plot(figLCOrig, sigmaPhot[gR][rSor], 'bo', ms=4, label='photometric only')
plt.plot(figLCOrig, sigmaCrowd[gR][rSor], 'rs', ms=4, label='crowding only')
plt.grid(which='both')
plt.xlabel('Simulated rMag before any perturbation')
plt.ylabel(r'1$\sigma$ uncertainties')
plt.title('Uncertainties vs rMag')
plt.ylim(-0.2,1.0)
plt.legend(loc=2, fontsize=8, numpoints=1)

plt.subplot(224)
plt.semilogy(figLCOrig, sigmaPhot[gR][rSor], 'bo', ms=4, label='photometric only')
plt.semilogy(figLCOrig, sigmaCrowd[gR][rSor], 'rs', ms=4, label='crowding only')
plt.grid(which='both')
plt.xlabel('Simulated rMag before any perturbation')
plt.ylabel(r'1$\sigma$ uncertainties')
plt.title('Uncertainties vs rMag')
plt.legend(loc=4, fontsize=8, numpoints=1)
#plt.ylim(-0.2,1.0)

#plt.scatter(figTime, figLCBoth, edgecolor='0.5', s=9, c='b')


# ### Running period star metric at low-amplitude to explore crowding errors ###
# 

# Now we apply this to a metric in which Monte Carlo trials are performed inside the metric. This time we opt for a reasonably short-period, reasonably *low*-amplitude, sinusoidal variation at reasonably faint magnitudes to see a case where the recovery fraction depends strongly on location. 
# 

# Now we create a new metric, this time with a number of monte carlo trials.
nSide=32 # 16 is reasonably fast on laptop 
meanMag = 22.
period=6.3 
nMonte=100 # small number to keep this reasonably quick on my laptop...
ampl=0.05  # low enough at this magn that crowding error should wash out entirely.
LMeans = [meanMag for i in range(6)]
metricMonte = sM.SineCrowdedMetric(nMonte=nMonte, period=period, periodTol=0.1,                                      amplitude=ampl, means=LMeans,                                    beVerbose=False)
slicerMonte = slicers.HealpixSlicer(nSide, lonCol='ditheredRA', latCol='ditheredDec')

# For the moment, don't include distance modulus.
# Let's make the distance modulus vary with healpix ID.
#distMod = np.arange(0.,slicerMonte.slicePoints['dec'].size)
distMod = np.repeat(0., slicerMonte.slicePoints['dec'].size)
slicerMonte.slicePoints['distMod'] = distMod

# Test sql query. We only have r-band crowding, so limit to this. 
# (SineCrowdedMetric also contains syntax to limit to r-band, so 
# it is safe to run this without the filter condition.)
sqlMonte='night < 14000 and filter = "r"'
#sqlMonte = 'night < 14000'


# Set up the database (this and the cell below are pasted from above
# to make re-running easier)
outDir='TESTperiodicWithCrowd_32'  # Results and images will go here..
dbFile='ops2_1092_sqlite.db'


# Connect to the database
opsimdb=utils.connectOpsimDb(dbFile)
resultsDb = db.ResultsDb(outDir=outDir)


bundleMonte = metricBundles.MetricBundle(metricMonte, slicerMonte, sqlMonte)
bgMonte = metricBundles.MetricBundleGroup({0:bundleMonte}, opsimdb,
                                      outDir=outDir, resultsDb=resultsDb)


# Run it
tStarted = time.time()
bgMonte.runAll()
print "INFO - Monte Carlo finished: %.2f minutes" %     ((time.time()-tStarted)/60.)


bgMonte.plotAll(closefigs=False)





# Let's vizualize the sequence of observations from a spot in the sky.  
# 

import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db





runName = 'enigma_1189'

opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)


metric=metrics.PassMetric(cols=['expMJD', 'fiveSigmaDepth', 'filter'])
ra = [0.]
dec = [np.radians(-30.)]
slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
sqlconstraint = 'night < 365'


bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


filters = np.unique(bundle.metricValues[0].compressed()['filter'])
colors = {'u':'b','g':'g','r':'r','i':'purple',"z":'y',"y":'magenta'}


mv = bundle.metricValues[0].compressed()
for filterName in filters:
    good = np.where(mv['filter'] == filterName)
    plt.scatter(mv['expMJD'][good]-mv['expMJD'].min(), mv['fiveSigmaDepth'][good], 
                c=colors[filterName], label=filterName)
plt.xlabel('Day')
plt.ylabel('5$\sigma$ depth (mags)')
plt.xlim([0,100])
plt.legend(scatterpoints=1)








# This is just a very little test notebook. It might give you a place to play, and it ought to indicate if the ipython notebook environment is working as you expect.
# 
# You will see various 'cells' (with numbers on the left hand side) below. These cells hold executable code (executable on your own machine, after installing sims_maf, doing a git clone of the sims_maf_contrib repo, and starting 'ipython notebook' in the tutorials directory of sims_maf_contrib). 
# 
# You can run the code in each cell by clicking on the cell, then pressing "shift + return". 
# 
# If you have more questions about ipython notebooks, you might find the [documentation](http://ipython.org/ipython-doc/stable/interactive/tutorial.html) useful. I also like the [ipython tutorials](http://nbviewer.ipython.org/github/AstroHackWeek/AstroHackWeek2014/blob/master/day1/ipython/Index.ipynb) from the 2014 Astro Hack Week. 
# 

print "Hi"


import os
yourname = os.getenv('USER')


print "Hi", yourname


currentdir = os.getenv('PWD')
print "I think I'm in directory ", currentdir


filenames = get_ipython().getoutput('ls *ipynb')
print "Notebooks in this directory include: "
print filenames


# Let's try to import various modules. 
# 

import numpy as np


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# And why not get all of the maf modules loaded up. 
# 

import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plot
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metricBundles as metricBundles


# If you get an error message after running the cell above saying "ImportError: No module named lsst.sims.maf.db", please check to be sure that you have 
# > `setup sims_maf -t sims` (or whatever is your appropriate tag - if you installed via 'conda' binaries, there is no tag)
# 
# If you get an error message like "ImportError: No module named persistence" or "ImportError: No module named lsst.sims.maf.metricBundles" you have an old version of the LSST software stack installed. Please update your software by following the instructions on the [Index](./Index.ipynb) notebook.
# 




# The basic idea here is to evaluate how well we can detect RGB stars and construct colour-magnitude diagrams (CMDs) for populations at various distances after 1,5 and 10 years of survey operations. This could be relevant for studies of the resolved stellar content in known Local Volume galaxies (e.g. searching for stellar streams in the their halos and mapping their low surface brightness extents) as well as perhaps for detecting new satellites and tidal streams in the very peripheral regions of the Galactic Halo (where main sequence turn off stars will be too faint to detect).  
# 
# Let's assume that we need to reach at least 2 magnitudes below the RGB tip (TRGB) in order to detect sufficient numbers of stars. If we take the TRGB as M$_i$=-4 then this corresponds to reaching stars at M$_i$=-2. While RGB stars have a range of g-i colours, assume here a fiducial value of g-i=1.2 (appropriate for metal-poor RGB populations). Hence, assuming no extinction, we want to evaluate how we can detect point sources with M$-i$=-2 and g-i=1.2 at various distances and at various stages of
# the survey.  
# 
# LSST is likely to be able to detect resolved populations out to several Mpc so a useful range in distance to explore might be 0.1-5 Mpc. But to begin with, we could assume we are interested in a single galaxy -- NGC 300 at m-M=26.44 -- which is situated in the nearby Sculptor Group and just beyond the boundary of the Local Group.  
# 
# Constructing a CMD requires detecting stars in two filters with good significance. After [1,5,10] years, how well is this RGB star detected in the g and i bands at a distance of NGC300 (i.e. g=25.6,i=24.4)? Or, alternatively, how long do we need to wait to get a 5$\sigma$ detection of this RGB star at the distance of NGC300? 
# 
# Additional things to consider:
# - simulate this for the actual population of Local Volume galaxies, using their known distances and line-of-sight extinction. 
# - because this involves detecting resolved stellar populations, good seeing is important for optimising star-galaxy separation. Hence might want to limit to consider only those observations with seeing less than 0.7 arcsec. 
# - I have assumed g,i as the default filter set. This a commonly used filter combination because it has good sensitivity to metallicity on the RGB, but other filter combinations could be looked at for CMD construction as they may be more efficient.  
# - simulate this for a real RGB population (i.e. generate a synthetic RGB for a given star formation history and assumed metallicity distribution) as opposed to just a fiducial RGB star. 
# - there will be contaminants in the form of unresolved background galaxies and foreground Milky Way stars. Detecting a significant population of RGB stars requires detecting them against this contaminant background, and hence what is considered here is idealised. 
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


filterName1 = 'g'
filterName2 = 'i'
years=[1.,5.,10.]
#nights = np.array(years)*365.25
goodSeeing = 0.7
sqls = ['filter = "%s" and night < %f and finSeeing < %f'% (filterName1, 5.*365.25, goodSeeing),
        'filter = "%s" and night < %f and finSeeing < %f'% (filterName2, 5.*365.25, goodSeeing)]
print sqls


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'goodseeing_test'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer(lonCol='ditheredRA',latCol='ditheredDec')
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]

bgroupList = []
names = ['All Visits', 'Good Seeing']
for name,sql in zip(names, sqls):
    metric = metrics.Coaddm5Metric(metricName=name)
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


print 'name, mean depth, median depth'
for bundleGroup in bgroupList:
    print bundleGroup.bundleDict[0].metric.name, bundleGroup.bundleDict[0].summaryValues['Mean'], bundleGroup.bundleDict[0].summaryValues['Median']


b = bundleGroup.bundleDict[0]


b.metricValues.data[10000]





# One nice aspect of the HealpixSlicer is that it can support lots of coordinate systems, and the metrics can be pulled out and visualized with the standard healpy tools as well
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots


opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'coordinates'


# let's just look at the number of observations in r-band after 2 years with default kwargs
sql = 'filter="r" and night < %i' % (365.25*2)
metric = metrics.CountMetric(col='expMJD')
slicer = slicers.HealpixSlicer()
plotDict = {'colorMax': 75}  # Set the max on the color bar so DD fields don't saturate
plotFuncs = [plots.HealpixSkyMap()] # only plot the sky maps for now
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)


bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)


bg.runAll()
bg.plotAll(closefigs=False)


# Same, only now run at very low resolution
slicer = slicers.HealpixSlicer(nside=8)
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# One thing we often want to do is run with dithered positions rather than the default to get rid of the field overlap issue
# 

slicer = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# Now let's try galactic and ecliptic coordiantes
# 

slicer = slicers.HealpixSlicer(latCol='galb', lonCol='gall')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


slicer = slicers.HealpixSlicer(latCol='eclipLat', lonCol='eclipLon')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# Those coordinates are getting generated automatically by stackers. If we want to move the dithered positions to galactic coordinates, we have to set that manually like thus:
# 

stacker = stackers.GalacticStacker(raCol='ditheredRA', decCol='ditheredDec')
slicer = slicers.HealpixSlicer(latCol='galb', lonCol='gall')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs,
                                    stackerList=[stacker])
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# One can also change things by using the rot plotting kwarg to rotate the projection. I never remember how to rotate from equatorial to galactic coords, but in theory, one could do it this way rather than using the stackers.
# 

slicer = slicers.HealpixSlicer() # back the the default
plotDict = {'colorMax': 75, 'rot':(35, 26, 22.)}

bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# Once can also use the healpy display tools:
hp.mollview(bundle.metricValues, max=75)


hp.gnomview(bundle.metricValues, max=75)


hp.cartview(bundle.metricValues, max=75)


hp.orthview(bundle.metricValues, max=75)


# And my personal favorite, looking at alt,az and using a special lambertian plotter.
# 
# NOTE: This plotter requires Basemap to run, which is not included by default in the LSST stack. To install:
# 
# `conda install basemap`
# 
# or
# 
# `pip install basemap`
# 

slicer = slicers.HealpixSlicer(latCol='zenithDistance', lonCol='azimuth')
plotFuncs=[plots.LambertSkyMap()]
plotDict = {}

bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)


# and this is still a healpix array
hp.mollview(bundle.metricValues, rot=(0,90))





# ### Getting Help in MAF
# 
# This notebook is a collection of snippets of how to get help on the various bits of the **MAF** ecosystem. It shows some of the **MAF** provided help functions. It also uses the `help` function. The `help` function used below is a Python standard library function. It can be used on any module, class or function. Using `help` should give clarity to the parameters used in associated functions. It will also list functions associated with modules and classes. The notebook also uses the `dir` command which is another Python standard library function. This is useful for getting a list of names from the target object (module/class/function). 
# 

from __future__ import print_function
# Need to import everything before getting help!
import lsst.sims.maf
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots


# Show the list of metrics
metrics.BaseMetric.help(doc=False)


# Show the help of a given metric
help(metrics.TransientMetric)


# If you have an object, help works on it too!
metric = metrics.CountMetric('expMJD')
help(metric)


# Show the list of slicers
slicers.BaseSlicer.help(doc=False)


# Show help of a given slicer
help(slicers.HealpixSlicer)


stackers.BaseStacker.help(doc=False)


# Show help of a given stacker
help(stackers.RandomDitherFieldPerNightStacker)


# See the plots available.
import inspect
vals = inspect.getmembers(plots, inspect.isclass)
for v in vals:
    print(v[0])


# Show the help of a given plots class
help(plots.HealpixSkyMap)





# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# This notebook shows how MAF can use the LSST camera geometry to decide if a region has been observed.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler


# Set the database and query
database = 'enigma_1189_sqlite.db'
sqlWhere = 'filter = "r" and night < 400 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15), np.radians(-15))
opsdb = db.OpsimDatabase(database)
outDir = 'Camera'
resultsDb = db.ResultsDb(outDir=outDir)


nside=512
metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(nside=nside)
slicer2 = slicers.HealpixSlicer(nside=nside, useCamera=True, radius=1.9)
summaryMetrics = [metrics.SumMetric()]


bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere, summaryMetrics=summaryMetrics)
bundle2 = metricBundles.MetricBundle(metric,slicer2,sqlWhere, summaryMetrics=summaryMetrics)
bg = metricBundles.MetricBundleGroup({'NoCamera':bundle1, 'WithCamera':bundle2},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


import healpy as hp
hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(7,-7,0), title='No Camera', unit='Count', min=1,max=21)
hp.gnomview(bundle2.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='With Camera', unit='Count', min=1,max=21)


# Print the number of pixel observations in the 2 cases. Note that running without the camera is about 3-4% optimistic.
print bundle1.summaryValues
print bundle2.summaryValues


# Now to try it again with dithering turned on
# 

slicer = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA', nside=nside)
slicer2 = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA',nside=nside, useCamera=True, radius=1.9)
bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere, summaryMetrics=summaryMetrics)
bundle2 = metricBundles.MetricBundle(metric,slicer2,sqlWhere, summaryMetrics=summaryMetrics)
bg = metricBundles.MetricBundleGroup({'NoCamera':bundle1, 'WithCamera':bundle2},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


import healpy as hp
hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(7,-7,0), title='No Camera', unit='Count', min=1,max=21)
hp.gnomview(bundle2.metricValues, xsize=800,ysize=800, rot=(7,-7,0),title='With Camera', unit='Count', min=1,max=21)


sqlWhere = 'fieldID = 2266 and night < 500'
nside = 2048
metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(nside=nside, useCamera=True, radius=1.9)
bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere)

bg = metricBundles.MetricBundleGroup({'HighResCamera':bundle1},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


hp.gnomview(bundle1.metricValues, xsize=400,ysize=400, rot=(48,-9,0), unit='Count')





# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# This notebooks demonstrates the most commonly used slicers and their features.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler


# There are three main slicers that we use in MAF.  For all three, we'll use the same metric and sql query to see how they differ.
# 
# Each slicer subdivides and groups the visits from opsim in different ways. 
# * The Unislicer simply clumps all visits into one group. 
# * The OneDSlicer groups visits into subsets based on the value of a single parameter from the opsim data.
# * The HealpixSlicer groups visits into subsets based on whether or not they overlap a given healpixel. 
# 
# Each slicer iterates through its set of "slicePoints" - the OneDSlicer iterates through the values of the opsim parameter (in bins) and the HealpixSlicer iterates through all pixels in the healpix grid. At each slicePoint, the metric value is calculated, resulting in a metric data array with as many values as the slicer had slicePoints. 
# 

# Set the database and query
database = 'pontus_1074.db'
sqlWhere = 'filter = "r" and night < 400'
opsdb = db.OpsimDatabase(database)
outDir = 'slicers_test'
resultsDb = db.ResultsDb(outDir=outDir)

# For the count metric the col kwarg is pretty much irrelevant, so we'll just use expMJD, but any column in the database would work
metric = metrics.CountMetric(col='observationStartMJD', metricName='Count')


# First the UniSlicer--this slicer simply passes all the data directly to the metric. So in this case, we will get the total number of visits.
# 

slicer = slicers.UniSlicer()


bundles = {}
bundles['uni'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)


# Next, the oneDSlicer. Here, we say we want to bin based on the 'night' column, and use binsize of 10 (days).  
# 

slicer = slicers.OneDSlicer(sliceColName='night', binsize=10)
bundles['oneD'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)


# Finally, the healpixSlicer will calculate the metric at a series of points accross the sky, using only the pointings that overlap the given point.
# 

slicer = slicers.HealpixSlicer(nside=64)
metric2 = metrics.Coaddm5Metric()
bundles['healpix'] = metricBundles.MetricBundle(metric2,slicer,sqlWhere)


slicer = slicers.OpsimFieldSlicer()
bundles['ops'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)


bgroup = metricBundles.MetricBundleGroup(bundles,opsdb, outDir=outDir, resultsDb=resultsDb)


# Now we can run all three and see the output
# 

bgroup.runAll()
bgroup.plotAll(closefigs=False)


# Let's examine the results from each slicer a bit further. 
# 

print bundles['uni'].metricValues
bundles['uni'].plot()


# With the UniSlicer, note the metric value matches the output (number of visits) from when we executed bgroup.runAll().  Since the UniSlicer only computes a single value, the plot method returns nothing.
# 
# ---
# 

print bundles['oneD'].metricValues
bundles['oneD'].plot()


# With the OneDSlicer, we binned on the 'night' value in opsim, and counted how many visits we had in each bin. The metric values show how many visits were in each bin, and the plot method produces a plot of the metric results as a function of the slicer bin values.
# 
# ---
# 

print bundles['healpix'].metricValues
bundles['healpix'].setPlotDict({'colorMin':0, 'colorMax':50})
bundles['healpix'].plot()


# With the HealpixSlicer, we calculate the number of visits at each point in the Healpix grid, so we have a long metric data array. We also have three ways to visualize the data; the skymap, a histogram (scaled by the area of each healpixel), and the angular power spectrum of the metric values. 
# 
# ---
# 







# ### An introduction to ipython notebooks
# 
# This notebook provides a demo of the basic capabilities and use of the MAF python interface
# 
# To use this notebook, you need a version of MAF >= 1.1, available via eups installation (eups distrib [install](https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF) sims_maf -t sims, after installing the base LSST software stack requirements). 
# 
# To run this notebook you should have,
# 
# > setup sims_maf -t sims
# 
# within the terminal where you ran ipython, i.e.
# 
# >ipython notebook IntroductionNotebook.ipynb
# 
# In this directory you should have downloaded the survey simulation database 
# 
# >wget http://ops2.tuc.noao.edu/runs/ops2_1114/data/ops2_1114_sqlite.db    
# 

# Check the version of MAF - the output should be version 1.1.1 or higher.
import lsst.sims.maf
lsst.sims.maf.__version__


# import matplotlib to show plots inline.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# Import the sims_maf modules needed.
# 

# import our python modules
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles


# ### Defining what we will measure and output
# 
# A `MetricBundle` combines a particular [`Metric`](https://confluence.lsstcorp.org/display/SIM/MAF+Metrics), a [`Slicer`](https://confluence.lsstcorp.org/display/SIM/MAF+Slicers), and a SQL constraint (sqlconstraint). 
# - a `metric` calculates some quantity that you want to measure (e.g. the mean value of an attribute such as the airmass). 
# - a `slicer` takes the list of visits from a simulated survey and orders or groups them (e.g. by night or by region on the sky).
# - a sqlconstraint applies a sql selection to the visits you want to select from the simulated survey database.
# 
# After a MetricBundle has been run it stores the metric values calculated at each point of the slice data. You can also add additional provenance/metadata information to a given MetricBundle, in the form of the (opsim) runName and a metadata comment. This information will be used to generate output filenames and plot labels. Basically, a `MetricBundle` completely defines a particular measurement on the OpSim simulated survey.
# 
# To create a `MetricBundle`, we will first generate a [`Metric`](https://confluence.lsstcorp.org/display/SIM/MAF+Metrics) and [`Slicer`](https://confluence.lsstcorp.org/display/SIM/MAF+Slicers), and a SQL constraint, and then combine them in a  `MetricBundle`.
# 
# As an example, here we'll get set up to calculate the maximum airmass value, at each point in a healpix grid.
# 

# metric = the "maximum" of the "airmass" for each group of visits in the slicer
metric1 = metrics.MaxMetric('airmass')

# slicer = a grouping or subdivision of visits for the simulated survey based on their position on the sky 
# (using a Healpix grid)
slicer1 = slicers.HealpixSlicer(nside=64)

# sqlconstraint = the sql query (or 'select') that selects all visits in r band.
sqlconstraint= 'filter = "r"'

# MetricBundle = combination of the metric, slicer, and sqlconstraint
maxairmassSky = metricBundles.MetricBundle(metric1, slicer1, sqlconstraint)


# ### Choosing the simulated survey database and the directory to output the results
# The input data is queried from a database (usually a SQLite database).
# The outputs are tracked in another database ('resultsDb_sqlite.db') in the results directory 'outDir'
# 

opsdb = db.OpsimDatabase('ops2_1114_sqlite.db')
outDir = 'output_directory'
resultsDb = db.ResultsDb(outDir=outDir)


# We can combine multiple MetricBundles and run them all at once to calculate their metric values, by combining them into a dictionary and sending this dictionary into a `MetricBundleGroup`. For now we will use calculate the metricvalues for our single 'maxairmassSky' MetricBundle. 
# 

bundleDict = {'maxairmassSky':maxairmassSky}


# We generate the outputs by combining the bundle dictionary with the input database and output directories and database, then using the `MetricBundleGroup` class.
# 
# The MetricBundleGroup will query the data from the opsim database and calculate the metric values, using the 'runAll' method. Note that MAF determines what columns you need from the database for your metrics and slicers, and only queries for those. MAF calculates the metrics in an efficient manner, caching results where possible and iterating through each slicer only once for all related metrics and saves everything to disk. 
# 

group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
group.runAll()


# ### Visualizing the output
# We can now plot the figures generated by running the metrics. This by default comes as three plots
# - the maximum airmass as a function of position on the sky (in a Healpix projection)
# - a histogram of the airmass distribution for the pixels in the Healpix projection
# - the angular powerspectrum of the maximum airmass
# 

group.plotAll(closefigs=False)


# ### Extending the analysis: adding more metrics and more information
# 
# We create a new `MetricBundle` (nvisitsSky), that works with the same slicer and sql constraint but now counts the number of visits at each point in the healpix grid. To do this we use the 'CountMetric' to count the number of exposure MJD's in each Healpix pixel 
# 

metric2 = metrics.CountMetric('expMJD')
nvisitsSky = metricBundles.MetricBundle(metric2, slicer1, sqlconstraint)


# We can also add ["summary metrics"](https://confluence.lsstcorp.org/display/SIM/MAF+Summary+Statistics) to each MetricBundle. These metrics  generate statistical summaries of the metric data values (e.g. the means of the number of visits per point on the sky).
# 

summaryMetrics = [metrics.MinMetric(), metrics.MedianMetric(), metrics.MaxMetric(), metrics.RmsMetric()]
maxairmassSky.setSummaryMetrics(summaryMetrics)
nvisitsSky.setSummaryMetrics(summaryMetrics)


# We can use the same metric but change the slicer (in this case grouping the visits by night) so we plot the maximum airmass and the number of visits per night)
# 

# A slicer that will calculate a metric after grouping the visits into subsets corresponding to each night.
slicer2 = slicers.OneDSlicer(sliceColName='night', binsize=1, binMin=0, binMax=365*10)

# We can combine these slicers and metrics and generate more metricBundles
nvisitsPerNight = metricBundles.MetricBundle(metric1, slicer2, sqlconstraint, summaryMetrics=summaryMetrics)
maxairmassPerNight = metricBundles.MetricBundle(metric2, slicer2, sqlconstraint, summaryMetrics=summaryMetrics)


# ### Grouping everything together 
# We can group any metricBundles together into a dictionary, and pass this to the MetricBundleGroup, which will run them (efficiently) together. 
# 

bundleDict = {'maxairmassSky':maxairmassSky, 'maxairmassPerNight':maxairmassPerNight, 
        'nvisitsSky':nvisitsSky, 'nvisitsPerNight':nvisitsPerNight}


group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# ### Accessing the data generated by a metric
# 
# The results of the metric calculation are stored as an attribute in each metricbundle, as 'metricValues' - a numpy masked array. The results of the summary statistics are also stored in each metricbundle, in an attribute called 'summaryValues', which is a dictionary of the summarymetric name and value. 
# 
# For nvisitsSky, the metricValues are an array of the number of visits on the sky for each healpix pixel and the summaryValues are the mean, median etc of the these values
# 

print "Array with the number of visits per pixel:", nvisitsSky.metricValues


# The values of the metricValues above are shown as '--' because these happen to be values which are masked. There were no opsim visits for the healpix points which correspond to the values shown from the array. However, there actually is data in the numpy MaskedArray, and this can be seen by doing any of the "normal" numpy operations:
# 

import numpy as np
np.max(nvisitsSky.metricValues)


print "Summary of the max, median, min, and rms of the number of visits per pixel", nvisitsSky.summaryValues


# ---
# ### Hands-on example ###
# 
# To get you started with a hands-on example: try setting up a metric and slicer to calculate the mean seeing at each point across the sky in the 'i' band.   
# 
# You might find the documentation on the column names in the [opsim summary table](https://confluence.lsstcorp.org/display/SIM/Summary+Table+Column+Descriptions) useful, in particular for finding the name of the seeing column.
# 

# This is what the skymap should look like:
from IPython.display import Image
Image(filename='images/thumb.ops2_1114_Mean_finSeeing_i_HEAL_SkyMap.png')





# # Time Delay Accuracy and Precision
# 
# _Phil Marshall & Lynne Jones_

# In the first [Time Delay Challenge](http://timedelaychallenge.org) paper, [Liao et al (2015)](http://arxiv.org/pdf/1409.1254.pdf) derived the following simple model for how strongly gravitationally lensed quasar time delay accuracy (A), precision (P) and success rate (f) depend on the night-to-night cadence, season and campaign length. 
# wrote several metrics and stackers to determine time delay accuracy (A), precision (P) and success rate (f). 

# \begin{align}
# |A|_{\rm model} &\approx 0.06\% \left(\frac{\rm cad} {\rm 3 days}  \right)^{0.0}
#                           \left(\frac{\rm sea}  {\rm 4 months}\right)^{-1.0}
#                           \left(\frac{\rm camp}{\rm 5 years} \right)^{-1.1} \notag \\
#   P_{\rm model} &\approx 4.0\% \left(\frac{\rm cad} {\rm 3 days}  \right)^{ 0.7}
#                          \left(\frac{\rm sea}  {\rm 4 months}\right)^{-0.3}
#                          \left(\frac{\rm camp}{\rm 5 years} \right)^{-0.6} \notag \\
#   f_{\rm model} &\approx 30\% \left(\frac{\rm cad} {\rm 3 days}  \right)^{-0.4}
#                         \left(\frac{\rm sea}  {\rm 4 months}\right)^{ 0.8}
#                         \left(\frac{\rm camp}{\rm 5 years} \right)^{-0.2} \notag
# \end{align}

# The first two of these metrics are candidate Figure of Merit proxies, while one can imagine combining all three somehow to provide an approximate dark energy parameter Figuer of Merit. These three metrics are implemented in [`tdcMetric.py`](http://github.com/LSST-nonproject/sims_maf_contrib/tree/master/mafContrib/tdcMetric.py) of the  [sims_maf_contrib](http://github.com/LSST-nonproject/sims_maf_contrib) git repository. 

# This notebook provides a demo calculation of these metrics, using the MAF python interface (requires sims_maf version >= 1.0). 

# ## Getting Started
# 
# You'll need the `sims_maf_contrib` folder on your `PYTHONPATH`. For me, that meant doing:
# ```bash
# export PYTHONPATH=$PYTHONPATH:/Users/pjm/work/stronglensing/LSST/ObservingStrategy/MAF/sims_maf_contrib
# ```

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Import MAF modules.
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup
# Import the contributed metrics and stackers 
import mafContrib


# ## Setting up  the MAF Evaluation of the Baseline Cadence 
# 
# As usual, we need to set up a `MetricBundleGroup` object so that we can run all of the metrics it contains on a single `OpSim` output database, which we first have to connect to. The easiest way I found to do this is to link my copy of the database to the current working directory.

runName = 'minion_1016'
database = runName + '_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'tmp'


# Instantiate the metrics, stackers and slicer that we want to use. These are the TDC metrics, the season stacker, and the healpix slicer. Actually, since we'll just use the stackers in their default configuration, we don't need to explicitly instantiate the stackers -- MAF will handle that for us.  
# Note that our metric (`TdcMetric`) is actually a "complex" metric, as it calculates A, P, and f in one go (thus re-using the cadence/season/campaign values which must also be calculated for each set of visits), and then has 'reduce' methods that separate each of these individual results into separate values. 

metric = mafContrib.TdcMetric(metricName='TDC', seasonCol='season', expMJDCol='expMJD', nightCol='night')
slicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')


# Let's set the plotFuncs so that we only create the skymap and histogram for each metric result. This is an optional step - otherwise, we'd just make the angular power spectra plots too. 

plotFuncs = [plots.HealpixSkyMap, plots.HealpixHistogram]
slicer.plotFuncs = plotFuncs


# The last ingredient we need is an SQL query to define the subset of visits we are interested in. Let's do something quick, like just the _i_ band in the first three years.

sqlconstraint = 'night < %i and filter = "i"' % (3*365.25)
tdcBundle = MetricBundle(metric=metric, slicer=slicer, sqlconstraint=sqlconstraint, runName=runName)


resultsDb = db.ResultsDb(outDir=outDir)
bdict = {'tdc':tdcBundle}
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)


# ## Running the Metrics and Working with the Results
# 
# We can query the database, run the stackers, run the metric calculation, and run the reduce functions with `runAll`. This would also generate summary statistics if we had defined any.

bgroup.runAll()


# Note that we now have more bundles in our bundle dictionary. These new bundles contain the results of the reduce functions - so, the metrics A, P, f separately as well as the "cadence", "season" and "campaign" diagnostics. 

bdict.keys()


# We want to set the `plotDict` for each of these separately, so that we can get each plot to look "just right", and then we'll make the plots.

minVal = 0.01
maxVal = {'Accuracy':0.04, 'Precision':10.0, 'Rate':40, 'Cadence':14, 'Season':8.0, 'Campaign':11.0}
units = {'Accuracy':'%', 'Precision':'%', 'Rate':'%', 'Cadence':'days', 'Season':'months', 'Campaign':'years'}
for key in maxVal:
    plotDict = {'xMin':minVal, 'xMax':maxVal[key], 'colorMin':minVal, 'colorMax':maxVal[key]}
    plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
    bdict['TDC_%s' % (key)].setPlotDict(plotDict)


bgroup.plotAll(closefigs=False)


# The `tdcBundle.metricValues` array contains the `HEALPix` maps of each metric.

tdcBundle.metricValues


# Let's do some post-processing to turn the metric maps into single numbers for a table in the observing strategy white paper.

import numpy as np
x = tdcBundle.metricValues
index = np.where(x.mask == False)


f = np.array([each['rate'] for each in x[index]])
A = np.array([each['accuracy'] for each in x[index]])
P = np.array([each['precision'] for each in x[index]])
c = np.array([each['cadence'] for each in x[index]])
s = np.array([each['season'] for each in x[index]])
y = np.array([each['campaign'] for each in x[index]])
print np.mean(f), np.mean(A), np.mean(P), np.mean(c), np.mean(s), np.mean(y)


# We are only interested in lenses with high accuracy delays, i.e. the fraction of the survey area where the _A_ metric is below some threshold. We can turn this into a sky area if we know the average size of a `HEALPix` pixel.

accuracy_threshold = 0.04 # 5 times better than threshold of 0.2% set by Hojjati & Linder (2014).

high_accuracy = np.where(A < accuracy_threshold)
high_fraction = 100*(1.0*len(A[high_accuracy]))/(1.0*len(A))
print "Fraction of total survey area providing high accuracy time delays = ",np.round(high_fraction,1),'%'

high_accuracy_cadence = np.median(c[high_accuracy])
print "Median night-to-night cadence in high accuracy regions = ",np.round(high_accuracy_cadence,1),'days'

high_accuracy_season = np.median(s[high_accuracy])
print "Median season length in high accuracy regions = ",np.round(high_accuracy_season,1),'months'

high_accuracy_campaign = np.median(y[high_accuracy])
print "Median campaign length in high accuracy regions = ",int(high_accuracy_campaign),'years'


Nside = 64
Npix = 12*Nside**2
Area_per_pixel = 4*np.pi / float(Npix) # steradians
Area_per_pixel *= (180.0/np.pi)*(180.0/np.pi) # square degrees
high_accuracy_area = len(A[high_accuracy])*Area_per_pixel
print "Area of sky providing high accuracy time delays = ",int(high_accuracy_area),"sq deg"


precision_per_lens = np.array([np.mean(P[high_accuracy]),4.0])
precision_per_lens = np.sqrt(np.sum(precision_per_lens*precision_per_lens))
print "Mean precision per lens in high accuracy sample, including modeling error = ",np.round(precision_per_lens,2),'%'

fraction = np.mean(f[high_accuracy])
N_lenses = int((high_accuracy_area/18000.0) * (fraction/30.0) * 400)
print "Number of lenses in high accuracy sample = ",N_lenses

distance_precision = (precision_per_lens * (N_lenses > 0)) / (np.sqrt(N_lenses) + (N_lenses == 0))
print "Maximum combined percentage distance precision (as in Coe & Moustakas 2009) = ",np.round(distance_precision,2),'%'


# The above overall precision can be related to the cosmological parameter precision, and so is a reasonable proxy Figure of Merit. This quantity is plotted by [Treu & Marshall (2016)](http://arxiv.org/abs/1605.05333) in their recent review: their target for the LSST era is between 0.4 and 0.7%.
# 
# The above analysis can all be repeated for alternative `OpSim` runs and SQL constraints, as we now show.

# ## Investigating Multiple `OpSim` Outputs
# 
# Since we want to compare several runs, and draw conclusions about which observing strategy is best, let's make a function that does all this for us, given a specified `OpSim` run name and an SQL query to select just the filters we want (either all, or just _r_ and _i_). This is just all the code we just wrote, but packed into a `def`.
# 
# ### Method

def evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', filters='ugrizy', Nyears=10):
    '''
    Sets up and executes a MAF analysis based on the Time Delay Challenge metrics.
    
    Parameters
    ----------
    runName : string ('minion_2016')
        The name of an OpSim simulation, whose output database will be used.
    filters : string ('ugrizy')
        List of bands to be used in analysis.
    Nyears : int
        No. of years in campaign to be used in analysis, starting from night 0.
        
    Returns
    -------
    results : dict
        Various summary statistics
    
    Notes
    -----

    '''
    # Set up some of the metadata, and connect to the OpSim database. 
    database = runName + '_sqlite.db'
    opsdb = db.OpsimDatabase(database)
    
    # Instantiate the metrics, stackers and slicer that we want to use. 
    # These are the TDC metrics, the season stacker, and the healpix slicer. 
    # Actually, since we'll just use the stackers in their default configuration, we don't need to 
    # explicitly instantiate the stackers -- MAF will handle that for us.  
    # Note that the metric (TdcMetric) is actually a "complex" metric, as it calculates A, P, and f 
    # all in one go (thus re-using the cadence/season/campaign values which must also be calculated
    # for each set of visits), and then has 'reduce' methods that separate each of these individual
    # results into separate values. 
    metric = mafContrib.TdcMetric(metricName='TDC', seasonCol='season', expMJDCol='expMJD', nightCol='night')
    slicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')
    
    # Set the plotFuncs so that we only create the skymap and histogram for each metric result 
    # (we're not interested in the power spectrum). 
    plotFuncs = [plots.HealpixSkyMap, plots.HealpixHistogram]
    slicer.plotFuncs = plotFuncs
    
    # Write the SQL constraint:
    sql = 'night < %i' % (365.25*Nyears)
    sqlstring = str(Nyears)+'years-'
    if filters == 'ugrizy':
        sql += ''
        sqlstring += 'ugrizy'
    elif filters == 'ri':
        sql += ' and (filter="r" or filter="i")'
        sqlstring += 'r+i-only'
    else:
        raise ValueError('Unrecognised filter set '+filters)

    # Set the output directory name:
    outDir = 'output_'+runName+'_'+sqlstring
    
    # Now bundle everything up:
    tdcBundle = MetricBundle(metric=metric, slicer=slicer, sqlconstraint=sql, runName=runName)
    resultsDb = db.ResultsDb(outDir=outDir)
    bdict = {'tdc':tdcBundle}
    bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)

    # And run the metrics!
    bgroup.runAll()
    
    # Now to make the plots. 
    # Note that we now have more bundles in our bundle dictionary - these new bundles contain 
    # the results of the reduce functions - so, A/P/f separately:
    #     bdict.keys() => ['tdc', 'TDC_Rate', 'TDC_Precision', 'TDC_Accuracy', 'TDC_Cadence', 'TDC_Campaign', 'TDC_Season']
    # We want to set the plotDict for each of these separately, so that we can get each plot 
    # to look "just right", and then we'll make the plots.   
    minVal = 0.01
    maxVal = {'Accuracy':0.04, 'Precision':10.0, 'Rate':40, 'Cadence':14, 'Season':8.0, 'Campaign':11.0}
    units = {'Accuracy':'%', 'Precision':'%', 'Rate':'%', 'Cadence':'days', 'Season':'months', 'Campaign':'years'}
    for key in maxVal:
        plotDict = {'xMin':minVal, 'xMax':maxVal[key], 'colorMin':minVal, 'colorMax':maxVal[key]}
        plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
        bdict['TDC_%s' % (key)].setPlotDict(plotDict)
    
    bgroup.plotAll(closefigs=False)
    
    # Now pull out metric values so that we can compute some useful summaries: 
    import numpy as np
    x = tdcBundle.metricValues
    index = np.where(x.mask == False)
    f = np.array([each['rate'] for each in x[index]])
    A = np.array([each['accuracy'] for each in x[index]])
    P = np.array([each['precision'] for each in x[index]])
    c = np.array([each['cadence'] for each in x[index]])
    s = np.array([each['season'] for each in x[index]])
    y = np.array([each['campaign'] for each in x[index]])

    # Summaries:
    results = dict()
    results['runName'] = runName
    results['filters'] = filters
    results['Nyears'] = Nyears
    
    accuracy_threshold = 0.04 # 5 times better than threshold of 0.2% set by Hojjati & Linder (2014).
    high_accuracy = np.where(A < accuracy_threshold)
    results['high_accuracy_area_fraction'] = 100*(1.0*len(A[high_accuracy]))/(1.0*len(A))
    print "Fraction of total survey area providing high accuracy time delays = ",np.round(results['high_accuracy_area_fraction'],1),'%'

    results['high_accuracy_cadence'] = np.median(c[high_accuracy])
    print "Median night-to-night cadence in high accuracy regions = ",np.round(results['high_accuracy_cadence'],1),'days'

    results['high_accuracy_season'] = np.median(s[high_accuracy])
    print "Median season length in high accuracy regions = ",np.round(results['high_accuracy_season'],1),'months'

    results['high_accuracy_campaign'] = np.median(y[high_accuracy])
    print "Median campaign length in high accuracy regions = ",int(results['high_accuracy_campaign']),'years'

    Nside = 64
    Npix = 12*Nside**2
    Area_per_pixel = 4*np.pi / float(Npix) # steradians
    Area_per_pixel *= (180.0/np.pi)*(180.0/np.pi) # square degrees
    results['high_accuracy_area'] = len(A[high_accuracy])*Area_per_pixel
    print "Area of sky providing high accuracy time delays = ",int(results['high_accuracy_area']),"sq deg"

    precision_per_lens = np.array([np.mean(P[high_accuracy]),4.0])
    results['precision_per_lens'] = np.sqrt(np.sum(precision_per_lens*precision_per_lens))
    print "Mean precision per lens in high accuracy sample, including modeling error = ",np.round(results['precision_per_lens'],2),'%'

    fraction = np.mean(f[high_accuracy])
    results['N_lenses'] = int((results['high_accuracy_area']/18000.0) * (fraction/30.0) * 400)
    print "Number of lenses in high accuracy sample = ",results['N_lenses']
 
    results['distance_precision'] = results['precision_per_lens'] / np.sqrt(results['N_lenses'])
    print "Maximum combined percentage distance precision (as in Coe & Moustakas 2009) = ",np.round(results['distance_precision'],2),'%'

    return results


# ### Results
# 
# We'll save all our results in a single array, so that it can be used to make a `latex` table at the end.

results = []


# #### `minion_2016`: The Baseline Cadence - 10 years, ugrizy

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=10, filters='ugrizy'))


# #### `minion_2016`: The Baseline Cadence - 5 years, ugrizy

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=5, filters='ugrizy'))


# #### `minion_2016`: The Baseline Cadence - 10 years, r+i only

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=10, filters='ri'))


# #### `minion_2016`: The Baseline Cadence - 5 years, r+i only

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=5, filters='ri'))


# #### `kraken_1043`: No Visit Pairs Required - 10 years, ugrizy

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=10, filters='ugrizy'))


# #### `kraken_1043`: No Visit Pairs Required - 5 years, ugrizy

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=5, filters='ugrizy'))


# #### `kraken_1043`: No Visit Pairs Required - 10 years, r+i only

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=10, filters='ri'))


# #### `kraken_1043`: No Visit Pairs Required - 5 years, r+i only

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=5, filters='ri'))


# ## Reporting the Results
# 
# For this we need to write a `latex` table.

def make_latex_table(results):
    """
    Writes a latex table, with one row per test, presenting all the TDC metrics.
    
    Parameters
    ----------
    results : list(dict)
        List of results dictionaries, one per experiment.
        
    Returns
    -------
    None.
    
    Notes
    -----
    The latex code is written to a simple .tex file for \input into a document.
    
    Each element of the results list is a dictionary, like this:
    {'high_accuracy_season': 6.9118266805479518, 'high_accuracy_area': 19004.12600851645, 
     'high_accuracy_area_fraction': 70.67103620474407, 'precision_per_lens': 5.0872446140504994, 
     'N_lenses': 468, 'Nyears': 10, 'runName': 'minion_1016', 'filters': 'ugrizy', 
     'high_accuracy_cadence': 4.5279775970305893, 'distance_precision': 0.23515796547162932, 
     'high_accuracy_campaign': 10.0}
     
    The interpretation of these numbers is as follows:
    
    Fraction of total survey area providing high accuracy time delays =  70.7 %
    Median night-to-night cadence in high accuracy regions =  4.5 days
    Median season length in high accuracy regions =  6.9 months
    Median campaign length in high accuracy regions =  10 years
    Area of sky providing high accuracy time delays =  19004 sq deg
    Mean precision per lens in high accuracy sample, including modeling error =  5.09 %
    Number of lenses in high accuracy sample =  468
    Maximum combined percentage distance precision (as in Coe & Moustakas 2009) =  0.24 %

    "High accuracy" means Accuracy metric > 0.04 
    
    Which element of results is which?
    
    for k in range(len(results)):
       print k, results[k]['runName'], results[k]['filters'], results[k]['Nyears']

    0 minion_1016 ugrizy 10
    1 minion_1016 ugrizy 5
    2 minion_1016 ri 10
    3 minion_1016 ri 5
    4 kraken_1043 ugrizy 10
    5 kraken_1043 ugrizy 5
    6 kraken_1043 ri 10
    7 kraken_1043 ri 5

    """
    
    # Open file object: 
    texfile = 'table_lenstimedelays.tex'
    f = open(texfile, 'w')
    
    # Start latex:
    tex = r'''
\begin{table*}
\begin{center}
\caption{Lens Time Delay Metric Analysis Results.}
\label{tab:lenstimedelays:results}
\footnotesize
\begin{tabularx}{\linewidth}{ccccccccc}
  \hline
  \OpSim run                       % runName -> db
   & Filters                       % filters
    & Years                        % Nyears
     & \texttt{cadence}            % high_accuracy_cadence
      & \texttt{season}            % high_accuracy_season
       & \texttt{Area}             % high_accuracy_area
        & \texttt{dtPrecision}     % precision_per_lens
         & \texttt{Nlenses}        % N_lenses
          & \texttt{DPrecision} \\ % distance_precision
  \hline\hline'''
    f.write(tex)

    # Now write the table rows:
    for k in range(8):
        x = results[k]
        if x['runName'] == 'minion_1016':
            x['db'] = '\opsimdbref{db:baseCadence}'
        elif x['runName'] == 'kraken_1043':
            x['db'] = '\opsimdbref{db:NoVisitPairs}'
        else:
            raise ValueError('Unrecognized runName: '+x['runName'])
        tex = r'''
  {db}
   & ${filters}$
    & ${Nyears:.0f}$
     & ${high_accuracy_cadence:.1f}$
      & ${high_accuracy_season:.1f}$
       & ${high_accuracy_area:.0f}$
        & ${precision_per_lens:.2f}$
         & ${N_lenses:.0f}$
          & ${distance_precision:.2f}$ \\'''.format(**x)
        f.write(tex)
        
    # Now finish up the table:    
    tex = r'''
   \hline

\multicolumn{9}{p{\linewidth}}{\scriptsize Notes: see the text for
the definitions of each metric.}
\end{tabularx}
\normalsize
\medskip\\
\end{center}
\end{table*}'''
    
    # Write last part to file and close up:
    f.write(tex)
    f.close()
    
    # Report
    print "LaTeX table written to "+texfile
    
    return


make_latex_table(results)


get_ipython().system(' cat table_lenstimedelays.tex')


# # Conclusions
# 
# We no have good diagnostic and Figure of Merit metrics for assessing lens time delay measurement, based on extrapolating the TDC1 single filter catalog-level simulations. We see that we are *analysis-limited* in the sense that *if* we can combine the *ugrizy* light curves with such fidelity that they appear as if we had simply undertaken a single filter monitoring withthe same cadence, then we can achieve the TDC1 results of 400 accurate measurements and 0.25% precision in time delay distance, and hence (roughly) $H_0$ - but if we cannot, then time delay cosmography will be significantly degraded and we would need additional monitoring data.

# Look at the evolution of the sky coverage.  If a template image has to come from a same-vendor chip, it takes longer to build up the sky template archive. 
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler
import healpy as hp


def drawRaft(ax, xLL, yLL, color='#CCCCCC', width = 1.0, height = 1.0, plotCCDs = 1):

    ax.add_patch(Rectangle((yLL, xLL), width, height, fill=True, color=color, ec='k'))
    # raft frame
    ax.plot([xLL, xLL], [yLL, yLL+height], 'black', linewidth=2)
    ax.plot([xLL+width, xLL+width], [yLL, yLL+height], 'black', linewidth=2)
    ax.plot([xLL, xLL+width], [yLL, yLL], 'black', linewidth=2)
    ax.plot([xLL, xLL+width], [yLL+height, yLL+height], 'black', linewidth=2)

    if plotCCDs: 
        ax.plot([xLL+width/3.0, xLL+width/3.0], [yLL, yLL+height], 'black', linewidth=1)
        ax.plot([xLL+2*width/3.0, xLL+2*width/3.0], [yLL, yLL+height], 'black', linewidth=1)
        ax.plot([xLL, xLL+width], [yLL+height/3.0, yLL+height/3.0], 'black', linewidth=1)
        ax.plot([xLL, xLL+width], [yLL+2*height/3.0, yLL+2*height/3.0], 'black', linewidth=1)

def plotHybridFOV(axes, option):

    # title 
    axes.set_title(option)

    # axes limits
    axes.set_xlim(0, 5.1)
    axes.set_ylim(0, 5.1)

    for i in (1, 2, 3): 
        for j in range(0,5):
            drawRaft(axes, i, j, color = 'red')
    for i in (0, 4): 
        for j in range(1,4):
            drawRaft(axes, i, j, color = 'red')
            
    if (option == 'A'):
        drawRaft(axes, 0, 2)
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 3)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 2, 4)
        drawRaft(axes, 3, 1)
        drawRaft(axes, 3, 3)
        drawRaft(axes, 4, 2)

    if (option == 'B'):
        drawRaft(axes, 0, 1)
        drawRaft(axes, 0, 2)
        drawRaft(axes, 0, 3)
        drawRaft(axes, 1, 0)
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 2)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 1)
        drawRaft(axes, 3, 0)

    if (option == 'C'):
        drawRaft(axes, 0, 1)
        drawRaft(axes, 0, 3)
        drawRaft(axes, 1, 0)
        drawRaft(axes, 1, 4)
        drawRaft(axes, 3, 0)
        drawRaft(axes, 3, 4)
        drawRaft(axes, 4, 1)
        drawRaft(axes, 4, 3)

    if (option == 'D'):
        drawRaft(axes, 1, 1)
        drawRaft(axes, 1, 3)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 3, 1)
        drawRaft(axes, 3, 3)

    if (option == 'E'):
        drawRaft(axes, 1, 2)
        drawRaft(axes, 2, 1)
        drawRaft(axes, 2, 2)
        drawRaft(axes, 2, 3)
        drawRaft(axes, 3, 2)

    if (option == 'F'):
        drawRaft(axes, 0, 2)
        drawRaft(axes, 2, 0)
        drawRaft(axes, 2, 4)
        drawRaft(axes, 4, 2)
     

### plot a 6-panel figure with hybrid focal plane realizations
def plotHybridFOVoptions(): 

    # Create figure and subplots
    fig = plt.figure(figsize=(8, 10))
    # this work well in *.py version but not so well in ipython notebook
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.9, bottom=0.05, top=0.95)

    optionsList = ('A', 'B', 'C', 'D', 'E', 'F')
    plotNo = 0
    for option in optionsList:
        plotNo += 1
        axes = plt.subplot(3, 2, plotNo, xticks=[], yticks=[], frameon=False)
        plotHybridFOV(axes, option)

    #plt.savefig('./HybridFOVoptions.png')
    plt.show() 


### 
plotHybridFOVoptions()


# Set up each configuration to return a list of chips in a way MAF understands
# Let's do this for a hybrid focal plane
def makeChipList(raftConfig):
    raftDict = {'R:1,0':1,
                'R:2,0':2 ,
                'R:3,0':3 ,
                'R:0,1':4 ,
                'R:1,1':5 ,
                'R:2,1':6 ,
                'R:3,1':7 ,
                'R:4,1':8 ,
                'R:0,2':9 ,
                'R:1,2':10,
                'R:2,2':11,
                'R:3,2':12,
                'R:4,2':13,
                'R:0,3':14,
                'R:1,3':15,
                'R:2,3':16,
                'R:3,3':17,
                'R:4,3':18,
                'R:1,4':19,
                'R:2,4':20,
                'R:3,4':21}

    sensors = ['S:0,0', 'S:0,1', 'S:0,2',
               'S:1,0', 'S:1,1', 'S:1,2',
               'S:2,0', 'S:2,1', 'S:2,2',]


    raftReverseDict = {}
    for key in raftDict:
        raftReverseDict[raftDict[key]] = key
    raftConfigs = {'A':{'rafts2':[1,3,4,6,8,10,12,14,16,18,19,21], 'rafts1':[2,5,7,9,11,13,15,17,20]},
                   'B':{'rafts2':[7,8,11,12,13,15,16,17,18,19,20,21], 'rafts1':[1,2,3,4,5,6,9,10,14]},
                   'C':{'rafts2':[2,5,6,7,9,10,11,12,13,15,16,17,20], 'rafts1':[1,3,4,8,14,18,19,21]},
                   'D':{'rafts2':[1,2,3,4,6,8,9,10,12,13,14,16,18,19,20,21], 'rafts1':[5,7,11,15,17]},
                   'E':{'rafts2':[1,2,3,4,5,7,8,9,13,14,15,17,18,19,20,21], 'rafts1':[6,10,11,12,16]},
                   'F':{'rafts2':[1,2,3,4,5,7,8,9,13,14,15,17,18,19,20,21], 'rafts1':[6,10,11,12,16]}
                  }
    rafts1 = []
    rafts2 = []
    for indx in raftConfigs[raftConfig]['rafts1']:
        rafts1.append(raftReverseDict[indx])

    for indx in raftConfigs[raftConfig]['rafts2']:
        rafts2.append(raftReverseDict[indx])

    chips1 = []
    for raft in rafts1:
        for sensor in sensors:
            chips1.append(raft+' '+sensor)

    chips2 = []
    for raft in rafts2:
        for sensor in sensors:
            chips2.append(raft+' '+sensor)
    return chips1, chips2





database = 'enigma_1189_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'Template'
resultsDb = db.ResultsDb(outDir=outDir)
nside = 128


bundleList = []
raftLayout = 'E'
filter = 'u'
chips1, chips2 = makeChipList(raftLayout)
years = 5
sqlWhere = 'filter = "%s" and night < %i' %(filter, 365.25*years)
metric = metrics.AccumulateCountMetric(bins=np.arange(0,365.25*years,1))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA', useCamera=True, chipNames=chips1)
bundleList.append(metricBundles.MetricBundle(metric,slicer,sqlWhere, metadata='Chips1, %s' % raftLayout))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA', useCamera=True, chipNames=chips2)
bundleList.append(metricBundles.MetricBundle(metric,slicer,sqlWhere, metadata='Chips2, %s' % raftLayout))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA', useCamera=True)
bundleList.append(metricBundles.MetricBundle(metric,slicer,sqlWhere, metadata='SingleVendor'))
bd = metricBundles.makeBundlesDictFromList(bundleList)
bg = metricBundles.MetricBundleGroup(bd,opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()





def timeToArea(metricValues, area=20000, nside=128):
    """
    compute how many night it took to reach the area limit (sq degrees)
    """
    pixArea = hp.nside2pixarea(nside, degrees=True)
    tmp = metricValues.copy()
    tmp[np.where(tmp > 0)] = 1
    tmp = np.sum(tmp, axis=0)
    tmp *= pixArea
    overLimit = np.where(tmp > area)[0]
    if np.size(overLimit) > 0:
        return np.min(overLimit)
    else:
        return -1


for bundle in bundleList:
    print bundle.metadata, 'days to 20,000 sq deg covered = ', timeToArea(bundle.metricValues, nside=nside, area=20000)








# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# #Demonstrate the proper motion and parallax metrics.#
# 
# For both of these metrics, visits in all bands are used (unless restricted by the user via a sqlconstraint). 
# The astrometric error for each visit, for a star of the specified brightness and spectral type, is estimated from the magnitude of the star and the m5 limit of each visit (recorded by opsim). 
# The parallax error is estimated by assuming that the proper motion is perfectly fit (or zero); the proper motion error is likewise estimated by assuming the parallax is perfectly fit (or zero). 
# In both cases, the effects of refraction is not currently included in the astrometric error. 
# 
# Here is a link to the [code](https://github.com/lsst/sims_maf/blob/master/python/lsst/sims/maf/metrics/calibrationMetrics.py) for the proper motion and parallax metrics. 
# The 'ra_pi_amp' and 'dec_pi_amp' columns referred to in the parallax metric are stacker columns which calculate the parallax amplitude (in ra and dec) for each visit, using this [code](https://github.com/lsst/sims_maf/blob/master/python/lsst/sims/maf/stackers/generalStackers.py). 
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'astrometry_test'
resultsDb = db.ResultsDb(outDir=outDir)


# Define our basic sqlconstraint (we'll only look at the first three years here) and define the metrics and slicer. 
# 

sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64)
metricList = []
metricList.append(metrics.ParallaxMetric())
metricList.append(metrics.ParallaxMetric(metricName='properMotion Normed', normalize=True))
metricList.append(metrics.ProperMotionMetric())
metricList.append(metrics.ProperMotionMetric(metricName='properMotion Normed', normalize=True))


# Add a summary metric to compute for each of the metricBundles. 
# 

summaryList = [metrics.MedianMetric()]


bundleList = []
for metric in metricList:
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, summaryMetrics=summaryList))


bundleDict = metricBundles.makeBundleDict(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


# By default, the proper motion and parallax metrics assume a flat SED with an r-band magnitude of 20, however this can be specified by the user when setting up the metric.
# 

rmags = {'faint':25, 'bright':18}
specTypes = ['B', 'K']
metricList = []
for mag in rmags:
    for specType in specTypes:
        metricList.append(metrics.ParallaxMetric(rmag=rmags[mag], SedTemplate=specType, 
                                                 metricName='parallax_'+mag+'_'+specType))
        metricList.append(metrics.ProperMotionMetric(rmag=rmags[mag], SedTemplate=specType, 
                                                     metricName='properMotion'+mag+'_'+specType))


bundlesSpec = []
for metric in metricList:
    bundlesSpec.append(metricBundles.MetricBundle(metric,slicer,sql, summaryMetrics=summaryList))
bundlesSpec = metricBundles.makeBundleDict(bundlesSpec)


bgroup = metricBundles.MetricBundleGroup(bundlesSpec, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


bgroup.plotAll(closefigs=False)


print 'Flat SED:'
for bundle in bundles.values():
    print bundle.metric.name, bundle.summaryValues
print 'B and K stars:'
for bundle in bundlesSpec.values():
    print bundle.metric.name, bundle.summaryValues





# Try out MAF on SDSS data
# 
# To run this, one needs to be able to connect to fatboy.  See instructions here: https://confluence.lsstcorp.org/display/SIM/Catalog+Simulations+Documentation
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots


outDir = 'SDSSSlicer'
resultsDb = db.ResultsDb(outDir=outDir)


# For connecting when on UW campus
sdssDB = db.SdssDatabase(database='clue', driver='mssql+pymssql', port=1433, host='fatboy-private.phys.washington.edu')
# XXX-note, this is untested: For off campus, need to open an ssh tunnel and use: 
# sdssDB = db.SdssDatabase(database='clue', driver='mssql+pymssql', port=51433, host='localhost')


sqlWhere = "filter='r' and nStars > 0 and nGalaxy > 0"
slicer = slicers.HealpixSDSSSlicer(nside=64, lonCol='RA1', latCol='Dec1')
metric = metrics.MeanMetric(col='psfWidth')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, sdssDB, outDir=outDir, resultsDb=resultsDb, dbTable='clue')
bgroup.runAll()
bgroup.plotAll(closefigs=False)
bgroup.writeAll()


import healpy as hp
hp.mollview(bundle.metricValues)





# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# This notebook demonstrates the basic use of stackers. You can also look at the "Dithers" notebook for more examples, specifically pertaining to dithering. 
# 
# Stackers allow the creation of 'virtual' columns to extend the opsim database. These columns are created on-the-fly, using the algorithms in the Stackers classes and cease to exist after the data queried by MAF leaves memory. 
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots


# MAF provides a set of stock Stackers, and more are available with the sims_maf_contrib package. You can see the list of registered stackers like this:
# 

stackers.BaseStacker.list(doc=True)


# ---
# Let's look at how you might use a simple stacker to calculate the mean of the Hour Angle visited each night. By default Opsim does not provide the Hour Angle, but you can see above that we provide a stacker (HourAngleStacker) to calculate it automatically if requested. 
# 
# To use simple stackers that do not require configuration (such as the HourAngleStacker), simply reference the column name which the stacker adds to the opsim data ('simdata').
# 

metric = metrics.MeanMetric(col='HA')
slicer = slicers.OneDSlicer(sliceColName='night', binsize=1)
sqlconstraint = 'night<100'
runName = 'enigma_1189'
mB = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName)


opsdb = db.OpsimDatabase(runName+'_sqlite.db')
outDir = 'stackers_test'
resultsDb = db.ResultsDb(outDir=outDir)


# We have not set up or referenced the stacker, except by the column name, but when we get the data from the database (done during the 'runAll' step), we can see that the stacker has been called and the virtual column added to the simdata results.
# 

bgroup = metricBundles.MetricBundleGroup({'ha':mB}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
print bgroup.simData.dtype.names


bgroup.plotAll(closefigs=False)


# Now let's use a stacker to try out some different dithering options.  Rather than the standard fieldRA and fieldDec columns, we'll add columns that are slightly offset.
# 

slicer = slicers.HealpixSlicer(nside=64, lonCol='randomDitherFieldNightRa', latCol='randomDitherFieldNightDec')
sqlconstraint = 'filter="r" and night<400'


metric = metrics.CountMetric(col='night')


# We could use the stacker in its default configuration, but we want to change it so that the maxDither is much smaller. To do this, we must instantiate the stacker and configure it ourselves, then pass this to the metricBundle.
# 

maxDither = 0.1
stackerList = [stackers.RandomDitherFieldNightStacker(maxDither=maxDither)]


plotDict={'colorMax':50, 'xlabel':'Number of Visits', 'label':'max dither = %.2f' % maxDither}
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName, 
                                    stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({'dither':bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)


bgroup.runAll()


bgroup.plotAll(closefigs=False)


# Update the stacker to use a larger max dither and re-run the bundle
maxDither = 1.75
plotDict2 = {'colorMax':50, 'xlabel':'Number of Visits', 'label':'max dither = %.2f' % maxDither}
stackerList = [stackers.RandomDitherFieldNightStacker(maxDither=maxDither)]
bundle2 = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict2)
bgroup2 = metricBundles.MetricBundleGroup({'dither_large':bundle2}, opsdb, outDir=outDir, resultsDb=resultsDb)


bgroup2.runAll()


bgroup2.plotAll(closefigs=False)


# Now we can combine the results on a single plot if we want to more easily compare them.
# 

ph = plots.PlotHandler()
ph.setMetricBundles([bundle, bundle2])
ph.setPlotDicts([{'label':'max dither 0.1', 'color':'b'}, {'label':'max dither 1.75', 'color':'r'}])
ph.plot(plots.HealpixPowerSpectrum())





# Evaluate SB limit after N years
# 
# Still to do:
# * use seeing in db rather than assuming fixed seeing
# * Add and recover fake galaxies
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import math


# This is needed to avoid an error when a metric is redefined
from lsst.sims.maf.metrics import BaseMetric
try:
    del metrics.BaseMetric.registry['__main__.SB']
except KeyError:
    pass


class SB(BaseMetric):
    """Calculate the SB at this gridpoint."""
    def __init__(self, m5Col = 'fiveSigmaDepth', metricName='SB', **kwargs):
        """Instantiate metric.

        m5col = the column name of the individual visit m5 data."""
        super(SB, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        seeing = 0.7
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname])) * (math.pi*seeing**2))


# Let's compare the depth in the r-band after 5 years, and the depth after 5 years when the seeing is better than 0.7 arcseconds
# 

filterName = 'r'
years = [1, 2, 3, 5, 10]
nights = np.array(years)*365.25
sqls = ['filter = "%s" and night < %f' %(filterName, night) for night in nights]
print sqls


# Set up the database connection
dbdir = '/Users/loveday/sw/lsst/enigma_1189/'
opsdb = db.OpsimDatabase(database = os.path.join(dbdir, 'enigma_1189_sqlite.db'))
outDir = 'GoodSeeing'
resultsDb = db.ResultsDb(outDir=outDir)

# opsdb.tables['Summary'].columns


slicer = slicers.HealpixSlicer()
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]
metric = SB()
bgroupList = []
for year,sql in zip(years,sqls):
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bundle.plotDict['label'] = '%i' % year
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)


mean_depth = []
median_depth = []
print 'year, mean depth, median depth'
for year,bundleGroup in zip(years,bgroupList):
    mean_depth.append(bundleGroup.bundleDict[0].summaryValues['Mean'])
    median_depth.append(bundleGroup.bundleDict[0].summaryValues['Median'])
    print (year, bundleGroup.bundleDict[0].summaryValues['Mean'], 
           bundleGroup.bundleDict[0].summaryValues['Median'])


# Plot SB limits as fn of time
plt.clf()
plt.plot(years, mean_depth, label='mean')
plt.plot(years, median_depth, label='median')
plt.plot((years[0], years[-1]), (26, 26), ':', label='~1:10 mass ratio')
plt.xlabel('Time (years)')
plt.ylabel(r'SB (r mag arcsec$^{-2}$)')
plt.legend(loc=4)
plt.show()









# XXX-We need more documentation on the contributed photometric precision metrics.  It is currently unclear how one should use and interpret them.
# 

get_ipython().magic('matplotlib inline')


# This notebook assumes you are using sims_maf version >= 1.0, and have 'setup sims_maf' in your shell. 
# 

import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np
import mafContrib
"""
Run the PhotPrecMetrics 
"""
goodSeeing = 0.7

sqls = [' night < %f' % ( 5.*365.25), ' night < %f and finSeeing < %f'% ( 5.*365.25, goodSeeing)]


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'goodseeing_test'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer(nside=16, lonCol='ditheredRA', latCol='ditheredDec')
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric(), mafContrib.RelRmsMetric()]

bgroupList = []
names = ['All Visits', 'Good Seeing']

for name,sql in zip(names, sqls):
    bundles = {}
    cnt=0
    sed = { 'g':25, 'r': 26, 'i': 25}
    metric1 = mafContrib.SEDSNMetric(metricName='SEDSN', mags=sed)
    metric2 = mafContrib.ThreshSEDSNMetric(metricName='SEDSN', mags=sed)

    bundle1 = metricBundles.MetricBundle(metric1, slicer, sql, summaryMetrics=summaryMetrics)
    bundle2 = metricBundles.MetricBundle(metric2, slicer, sql, summaryMetrics=summaryMetrics)

    bundles={0:bundle1,1:bundle2}

    bgroup = metricBundles.MetricBundleGroup(bundles, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

if False:
    print 'name, mean PhotPrec, median PhotPrec '
    for bundleGroup in bgroupList:
        for i in range(6):
            print 'Filter %d'%i
            print bundleGroup.bundleDict[i].metric.name,                 bundleGroup.bundleDict[i].summaryValues['Mean'],                 bundleGroup.bundleDict[i].summaryValues['Median'],                bundleGroup.bundleDict[i].summaryValues['RelRms']





# Want to get rid of systematics from CCDs for weak lensing, so we want the position angle (mod 180 deg) to be as uniform as possible
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from scipy import stats


class uniformKSTest(metrics.BaseMetric):
    """
    Return the KS-test statistic. Values near zero are good, near 1 is bad.
    """
    def __init__(self, paCol = 'rotSkyPos', modVal=180., metricName='uniformKSTest', units='unitless', **kwargs):
        self.paCol = paCol
        self.modVal = modVal
        super(uniformKSTest, self).__init__(col=paCol, metricName=metricName, units=units, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        angleDist = dataSlice[self.paCol] % self.modVal
        ks_D, pVal = stats.kstest(angleDist, 'uniform')
        return ks_D


class KuiperMetric(metrics.BaseMetric):
    """
    Like the KS test, but for periodic things.
    """
    def __init__(self, col='rotSkyPos', cdf=lambda x:x/(2*np.pi), args=(), period=2*np.pi, **kwargs):
        self.cdf = cdf
        self.args = args
        self.period = period
        assert self.cdf(0) == 0.0
        assert self.cdf(self.period) == 1.0
        super(KuiperMetric, self).__init__(col=col, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        data = np.sort(dataSlice[self.colname] % self.period)
        cdfv = self.cdf(data, *self.args)
        N = len(data)
        D = np.amax(cdfv-np.arange(N)/float(N)) + np.amax((np.arange(N)+1)/float(N)-cdfv)
        return D


opsdb = db.OpsimDatabase('minion_1016_sqlite.db')
outDir = 'temp'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer()
sql = 'filter = "g"'
metric = KuiperMetric()
bundle = metricBundles.MetricBundle(metric, slicer, sql)


bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)


bundle.metricValues.max()


slicer = slicers.UniSlicer()
sql = 'fieldID=310 and filter="i"'
metric = metrics.PassMetric('rotSkyPos')


bundle = metricBundles.MetricBundle(metric, slicer, sql)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()


ack = plt.hist(np.degrees(bundle.metricValues[0]['rotSkyPos']) % 180.)


ks = uniformKSTest()
print ks.run(bundle.metricValues[0])





# These vector metrics are designed to help visualize the time evolution of the survey.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots


nside = 32

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = '2DSlicers'
resultsDb = db.ResultsDb(outDir=outDir)
plotFuncs = [plots.TwoDMap()]


# Plot the total number of visits to each healpixel as a function of time
metric = metrics.AccumulateCountMetric(bins=np.arange(366*10))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'colorMax':1000, 'xlabel':'Night (days)'}
sql=''
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# Same as above, but now only using OpSim field IDs rather than healpixels
plotFuncs = [plots.TwoDMap()]
slicer = slicers.OpsimFieldSlicer()
plotDict = {'colorMax':1000, 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# Make a histogram of the number of visits per field per night in year 1
plotFuncs = [plots.TwoDMap()]
metric = metrics.HistogramMetric(bins=np.arange(367)-0.5)
slicer = slicers.OpsimFieldSlicer()
sql = 'night < 370'
plotDict = {'colorMin':1, 'colorMax':5, 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# Now, if we want to see the number of visit pairs, tripples, quads per night, we can just use a different plotter
plotters = [plots.VisitPairsHist()]
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotFuncs=plotters)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)





# Special metrics for computing the co-added depth as a function of time
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateM5Metric(bins=np.arange(365.25*10)-0.5)
slicer = slicers.HealpixSlicer(nside=nside)
sql = 'filter="r"'
plotDict = {'cbarTitle':'mags', 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# Look at the minimum seeing as a function of time. 
# One could use this to feed a summary metric to calc when the entire sky has a good template image
# Or, what fraction of the sky has a good template after N years.
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateMetric(col='finSeeing', function=np.minimum,
                                 bins=np.arange(366*10))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'xlabel':'Night (days)', 'cbarTitle':'Minimum Seeing (arcsec)', 'colorMax':0.8}
sql='filter="r"'
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)


# Note that these are arrays of healpix maps, so it's easy to pull out a few and plot them all-sky
# I think this means it should be pretty easy to make a matplotlib animation without having to dump each plot to disk: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
import healpy as hp
for i,night in zip(np.arange(4)+1,[200,400, 800,1200]):
    hp.mollview(bundle.metricValues[:,night], title='night = %i'%night, unit='Seeing (arcsec)', sub=(2,2,i))


# Can use for just a few user-defined points
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateMetric(col='finSeeing', function=np.minimum,
                                 bins=np.arange(366*10))
ra = np.zeros(10.)+np.radians(10.)
dec = np.radians(np.arange(0,10)/9.*(-30))
slicer = slicers.UserPointsSlicer(ra,dec, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'xlabel':'Night (days)', 'cbarTitle':'Minimum Seeing (arcsec)', 'colorMax':0.8}
sql='filter="r"'
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)














# This notebook assumes you are using sims_maf version >= 1.1, and have 'setup sims_maf' in your shell. 
# 
# Demo of using a complex metric with reduce functions, the 'completeness' metric.
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils


# Connect to opsim database as usual, and get information on the proposals and the requested number of visits, in order to configure the 'completeness' metric.
# 

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')


propinfo, proptags = opsdb.fetchPropInfo()
reqvisits = opsdb.fetchRequestedNvisits(proptags['WFD'])
print reqvisits


# Set up metric and slicer, and generate a sql constraint for the WFD proposal using a MAF utility. Set the limits we want for the x/color range in the plotDict.
# 

completeness_metric = metrics.CompletenessMetric(u=reqvisits['u'], g=reqvisits['g'], r=reqvisits['r'], 
                                          i=reqvisits['i'], z=reqvisits['z'], y=reqvisits['y'])
slicer = slicers.OpsimFieldSlicer()
sqlconstraint = utils.createSQLWhere('WFD', proptags)
summaryMetric = metrics.TableFractionMetric()
plotDict = {'xMin':0, 'xMax':1.2, 'colorMin':0, 'colorMax':1.2, 'binsize':0.025}


# Instantiate the metric bundle, and turn it into a dictionary, then set up the metricbundlegroup.
# 

completeness = metricBundles.MetricBundle(metric=completeness_metric, slicer=slicer, 
                                          sqlconstraint=sqlconstraint, runName='enigma_1189', 
                                          summaryMetrics=summaryMetric, plotDict=plotDict)


bdict = {'completeness':completeness}


outDir = 'completeness_test'
resultsDb = db.ResultsDb(outDir=outDir)
bg = metricBundles.MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)


# Run it! This also runs all 'reduce' methods (via MetricBundleGroup.reduceAll()) and 'summary metrics' (via MetricBundleGroup.summaryAll()). 
# 

bg.runAll()


bg.plotAll(closefigs=False)


for b in bdict.itervalues():
    print b.metric.name, b.summaryMetrics, b.summaryValues


