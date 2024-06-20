# ### Import required packages
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as g

from neuronunit.neuron.models import *
from neuronunit.tests import *
import neuronunit.neuroelectro
from quantities import nA, pA, s, ms, mV, ohm


# ### Define tests to run
# 

# Define which cell's properties to get
# 'nlx_anat_100201' is from Olfactory bulb mitral cell, 
# obtained from http://neuroelectro.org/neuron/129/ (Details)
neuron = {'nlex_id': 'nlx_anat_100201'} 

# Define the tests to perform on the cell
testTypes = [     InputResistanceTest,     RestingPotentialTest,     InjectedCurrentAPWidthTest,     InjectedCurrentAPThresholdTest,     InjectedCurrentAPAmplitudeTest,     TimeConstantTest, ]

observations = {}


# ### Retrieve Un-Pooled Property Summary Stats from NeuroElectro.org
# 

observations["UnPooled"] = {}

# Fetch NeuroElectro property values for each test
for t in xrange(len(testTypes)):
    testType = testTypes[t]
    
    # Get the observations: property means and sds
    obs = testType.neuroelectro_summary_observation(neuron)
    
    observations["UnPooled"][testType.name] = obs
    
print("UnPooled Summary Stats")
pp(observations["UnPooled"])


# ### Retrieve Pooled Property Summary Stats
# 

observations["Pooled"] = {}

# Fetch NeuroElectro property values for each test
for t in xrange(len(testTypes)):
    testType = testTypes[t]
    
    # Get the observations: property means and sds
    obs = testType.neuroelectro_pooled_observation(neuron,quiet=False)
    
    observations["Pooled"][testType.name] = obs
    
print("Pooled Summary Stats")
pp(observations["Pooled"])


# ### Store User Defined Property Stats
# 

result = {}

# These should be the values after NeuroElectro returns N's and type of Err

result[InputResistanceTest.name] =            { 'mean': 87.398 * 1000000 * ohm, 'std': 41.42 * 1000000 * ohm, 'n': 386 }
result[RestingPotentialTest.name] =           { 'mean': -54.91 * mV, 'std': 3.5969 * mV, 'n': 295 }
result[InjectedCurrentAPWidthTest.name] =     { 'mean': 1.737 * ms, 'std': 0.145588 * ms, 'n': 64 }
result[InjectedCurrentAPThresholdTest.name] = { 'mean': -48.47 * mV, 'std': 3.932 * mV, 'n': 8 }
result[InjectedCurrentAPAmplitudeTest.name] = { 'mean': 74.529 * mV, 'std': 1.88 * mV, 'n': 52 }
result[TimeConstantTest.name] =               { 'mean': 27.78595 * ms, 'std': 11.75 * ms, 'n': 66 }

observations["UserDefined"] = result
    
print("UserDefined Summary Stats")
pp(observations["UserDefined"])


# ### Show All Observations
# 

pp(observations)


# ### Store The Observations in a File
# 

import pickle

with open("observations.dat", "wb") as file:
    pickle.dump(observations,file)


