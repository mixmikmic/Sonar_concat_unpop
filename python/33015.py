# Current tasks
# -------------
# - Set _eval_energy and eval_energy to be nogil
# - ~~Make CompiledModel a cdef class~~
# - Link CompiledModel-based energy evaluation into the main solver
# - Decide how/if to store gradients and Hessians of RedlichKisterSums
# - Write analagous eval_gradient and eval_hessian functions
# - Link gradient and Hessian for CompiledModel into main solver
# 

from pycalphad.core.compiled_model import CompiledModel
from pycalphad import calculate, Database, Model
from pycalphad.core.phase_rec import PhaseRecord
import pycalphad.variables as v
import numpy as np


dbf = Database('alcocr-sandbox.tdb')


cmpmdl = CompiledModel(dbf, ['AL', 'CO', 'CR', 'VA'], 'BCC_A2')
out = np.zeros(7)
get_ipython().magic('time cmpmdl.eval_energy_gradient(out, np.array([101325, 600, 0.3, 0.3, 0.4-1e-12, 1e-12, 1]), np.array([]))')
out


mdl = Model(dbf, ['AL', 'CO', 'CR', 'VA'], 'BCC_A2')
#mdl.models['idmix'] = 0
#mdl.models['ref'] = 0
#print(mdl.GM.diff(v.Y('BCC_A2', 0, 'VA')))
[mdl.GM.diff(x).subs({v.T: 600., v.P:101325., v.Y('BCC_A2', 0, 'AL'): 0.3, v.Y('BCC_A2', 0, 'CO'): 0.3,
                    v.Y('BCC_A2', 0, 'CR'): 0.4-1e-12, v.Y('BCC_A2', 0, 'VA'): 1e-12, v.Y('BCC_A2', 1, 'VA'): 1.})
 for x in [v.P, v.T, v.Y('BCC_A2', 0, 'AL'),v.Y('BCC_A2', 0, 'CO'),v.Y('BCC_A2', 0, 'CR'),v.Y('BCC_A2', 0, 'VA'), v.Y('BCC_A2', 1, 'VA')]]


get_ipython().magic("time res = calculate(dbf, ['AL', 'CO', 'CR', 'VA'], sorted(dbf.phases.keys()), T=2000, P=101325, output='GM', pdens=200, model=CompiledModel)")
res.GM[..., :20]


get_ipython().magic("time res2 = calculate(dbf, ['AL', 'CO', 'CR', 'VA'], sorted(dbf.phases.keys()), T=2000, P=101325, output='GM', pdens=200, model=Model)")
res2.GM[..., :20]





# # Fitting experimental data to thermodynamic models
# 

# Besides calculating and visualizing phase equilibria using known models, **pycalphad** also enables you to fit experimental data to a parameterized thermodynamic model. While pycalphad handles construction of the energy functions and equilibrium calculation, a separate module handles the numerical details of fitting disparate sources of data to the thermodynamic functions.
# 
# Here we consider the case study of the Al-Zn system. First, we use a TDB file as the setup for our model parameterizations. Note that not all of the parameters have not been filled in. These are what we will fit.
# 

get_ipython().magic('matplotlib inline')

TDB_STR = """
$ Al-Zn
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT ZN   HCP_ZN                    6.5390E+01  5.6568E+03  4.1631E+01!
 FUNCTION  GHSERAL  298.0
    -7976.15+137.0715*T-24.36720*T*LN(T)-1.884662E-3*T**2-0.877664E-6*T**3
    +74092*T**(-1);                                                   700.00 Y
    -11276.24+223.0269*T-38.58443*T*LN(T)+18.531982E-3*T**2-5.764227E-6*T**3
    +74092*T**(-1);                                                    933.6 Y
    -11277.68+188.6620*T-31.74819*T*LN(T)-1234.26E25*T**(-9);        2900.00 N ! 
 FUNCTION  GALLIQ   298.0
    +3029.403+125.2307*T-24.36720*T*LN(T)-1.884662E-3*T**2-0.877664E-6*T**3
    +74092*T**(-1)+79.401E-21*T**7;                                   700.00 Y
    -270.6860+211.1861*T-38.58443*T*LN(T)+18.53198E-3*T**2-5.764227E-6*T**3
    +74092*T**(-1)+79.401E-21*T**7;                                    933.6 Y
    -795.7090+177.4100*T-31.74819*T*LN(T);                           2900.00 N !
 FUNCTION  GALHCP   298.0  +5481-1.8*T+GHSERAL#;                      6000  N !
 FUNCTION GHSERZN    298.0  -7285.787+118.4693*T-23.70131*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3;                              692.7 Y
     -11070.60+172.3449*T-31.38*T*LN(T)+4.70657E+26*T**(-9);           1700 N !
 FUNCTION GZNLIQ     298.0  -1.285170+108.1769*T-23.70131*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3-3.585652E-19*T**7;            692.7 Y
     -11070.60+172.3449*T-31.38*T*LN(T)+4.70657E+26*T**(-9);           1700 N !
 FUNCTION GZNLIQ     298.14  +7157.213-10.29299*T-3.5896E-19*T**7+GHSERZN#;
                                                                      692.7 Y
     +7450.168-10.737066*T-4.7051E+26*T**(-9)+GHSERZN#;                 1700 N !
 FUNCTION GZNFCC     298.15  +2969.82-1.56968*T+GHSERZN#;               1700 N !
$-------------------------------------------------------------------------------
 TYPE_DEFINITION % SEQ *!
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :AL,ZN :  !
   PARAMETER G(LIQUID,AL;0)      298.15  +GALLIQ#;                      2900 N !
   PARAMETER G(LIQUID,ZN;0)      298.15  +GZNLIQ#;                      1700 N !
   PARAMETER G(LIQUID,AL,ZN;0)   298.15  LIQALZN0H-LIQALZN0S*T;         6000 N !
 PHASE FCC_A1  %  1  1.0  !
 CONSTITUENT FCC_A1  :AL,ZN :  !
   PARAMETER G(FCC_A1,AL;0)      298.15  +GHSERAL#;                     2900 N !
   PARAMETER G(FCC_A1,ZN;0)      298.15  +GZNFCC#;                      1700 N !
   PARAMETER G(FCC_A1,AL,ZN;0)   298.15  FCCALZN0H-FCCALZN0S*T;         6000 N !
   PARAMETER G(FCC_A1,AL,ZN;1)   298.15  FCCALZN1H-FCCALZN1S*T;         6000 N !
   PARAMETER G(FCC_A1,AL,ZN;2)   298.15  FCCALZN2H-FCCALZN2S*T;         6000 N !
 PHASE HCP_A3  %  1  1.0  !
 CONSTITUENT HCP_A3  :AL,ZN :  !
  PARAMETER G(HCP_A3,AL;0)       298.15  +GALHCP#;                      2900 N !
  PARAMETER G(HCP_A3,ZN;0)       298.15  +GHSERZN#;                     1700 N !
  PARAMETER G(HCP_A3,AL,ZN;0)    298.15  HCPALZN0H-HCPALZN0S*T;         6000 N !
  PARAMETER G(HCP_A3,AL,ZN;3)    298.15  HCPALZN3H-HCPALZN3S*T;         6000 N !"""


# Now we will import pycalphad and load the database.
# 

from pycalphad import Database, Model, binplot
import pycalphad.variables as v
import matplotlib.pyplot as plt
from sympy import Symbol

# Initialize database and choose active phases
db_alzn = Database(TDB_STR)
my_phases_alzn = ['LIQUID', 'FCC_A1', 'HCP_A3']


# ## Prior to fitting
# 
# First, we'll show what the phase diagram looks like without any binary interaction parameters. In this diagram only the pure component lattice stabilities of each phase relative to their standard states are considered.
# 

blank_parameters = {
    'LIQALZN0H': 0.0,
    'LIQALZN0S': 0.0,
    'FCCALZN0H': 0.0,
    'FCCALZN0S': 0.0,
    'FCCALZN1H': 0.0,
    'FCCALZN1S': 0.0,
    'FCCALZN2H': 0.0,
    'FCCALZN2S': 0.0,
    'HCPALZN0H': 0.0,
    'HCPALZN0S': 0.0,
    'HCPALZN3H': 0.0,
    'HCPALZN3S': 0.0
    }

models = {
    'LIQUID': Model(db_alzn, ['AL', 'ZN'], 'LIQUID', parameters=blank_parameters),
    'FCC_A1': Model(db_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1', parameters=blank_parameters),
    'HCP_A3': Model(db_alzn, ['AL', 'ZN', 'VA'], 'HCP_A3', parameters=blank_parameters)
    }

fig = plt.figure(figsize=(9,6))
get_ipython().magic("time ax = binplot(db_alzn, ['AL', 'ZN', 'VA'] , my_phases_alzn, 'X(ZN)', 300, 1000, ax=plt.gca(), model=models)")
ax = ax.set_title('Al-Zn with endmembers only', fontsize=20)


# Now load a test Comma-Separated Values (CSV) file with some phase equilibria data. This data could be from experiment, *ab-initio* methods or another source. Let's plot the data on top of our phase diagram to get an idea of what we're dealing with here.
# 

import pandas as pd
from pycalphad.plot.utils import phase_legend

phase_eq = pd.read_csv('alzn_test_phase_eq.csv')

fig = plt.figure(figsize=(9,6))
ax = binplot(db_alzn, ['AL', 'ZN', 'VA'] , my_phases_alzn, 'X(ZN)', 300, 1000, ax=plt.gca(), model=models)
ax.set_title('Al-Zn with experimental data', fontsize=20)
legend_handles, colors = phase_legend(my_phases_alzn)
for phase, phase_df in phase_eq.groupby('Phase'):
    ax = phase_df.plot(kind='scatter', x='X(ZN)', y='T', ax=ax, edgecolor='black',
                       s=30, xlim=(0,1), marker='s', facecolor='none')
ax = ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.6))


# ## Starting parameter values
# Let's imagine we have some way (first-principles calculations, intuition or both) to guess some starting values for our parameters.
# 

perturbed_parameters = {
    'LIQALZN0H': 10000,
    'LIQALZN0S': 3,
    'FCCALZN0H': 7000,
    'FCCALZN0S': -0.2,
    'FCCALZN1H': 5000,
    'FCCALZN1S': 4,
    'FCCALZN2H': -3000,
    'FCCALZN2S': -3,
    'HCPALZN0H': 17000,
    'HCPALZN0S': 6,
    'HCPALZN3H': -700,
    'HCPALZN3S': 1
    }
fit_models = {
    'LIQUID': Model(db_alzn, ['AL', 'ZN'], 'LIQUID', parameters=perturbed_parameters),
    'FCC_A1': Model(db_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1', parameters=perturbed_parameters),
    'HCP_A3': Model(db_alzn, ['AL', 'ZN', 'VA'], 'HCP_A3', parameters=perturbed_parameters)
    }

fig = plt.figure(figsize=(9,6))
ax = binplot(db_alzn, ['AL', 'ZN', 'VA'] , my_phases_alzn, 'X(ZN)', 300, 1000, ax=plt.gca(), model=fit_models)
ax = phase_eq.plot(kind='scatter', x='X(ZN)', y='T', ax=ax, edgecolor='black',
                       s=30, xlim=(0,1), marker='s', facecolor='none')
ax = ax.set_title('Al-Zn with parameter guesses', fontsize=20)


# ## Constructing a residual and performing a fit
# 
# We construct a residual function based on distance from the equilibrium hyperplane (driving force).
# 

from pycalphad.residuals import fit_model
import numpy as np
import pandas as pd

perturbed_parameters = {
    'LIQALZN0H': 10000,
    'LIQALZN0S': 3,
    'FCCALZN0H': 7000,
    'FCCALZN0S': -0.2,
    'FCCALZN1H': 5000,
    'FCCALZN1S': 4,
    'FCCALZN2H': -3000,
    'FCCALZN2S': -3,
    'HCPALZN0H': 17000,
    'HCPALZN0S': 6,
    'HCPALZN3H': -700,
    'HCPALZN3S': 1
    }

param_names, param_values = list(zip(*perturbed_parameters.items()))
phase_eq = pd.read_csv('alzn_test_phase_eq.csv')
param_values = np.array(param_values)

get_ipython().magic("time result, mi = fit_model(perturbed_parameters, phase_eq, db_alzn, ['AL', 'ZN', 'VA'], ['LIQUID', 'HCP_A3', 'FCC_A1'])")


fit_models = {
    'LIQUID': Model(db_alzn, ['AL', 'ZN'], 'LIQUID', parameters=result),
    'FCC_A1': Model(db_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1', parameters=result),
    'HCP_A3': Model(db_alzn, ['AL', 'ZN', 'VA'], 'HCP_A3', parameters=result)
    }

fig = plt.figure(figsize=(9,6))
ax = binplot(db_alzn, ['AL', 'ZN', 'VA'] , my_phases_alzn, 'X(ZN)', 300, 1000, ax=plt.gca(), model=fit_models)
ax = phase_eq.plot(kind='scatter', x='X(ZN)', y='T', ax=ax, edgecolor='black',
                       s=30, xlim=(0,1), marker='s', facecolor='none')
ax = ax.set_title('Al-Zn leastsq fit', fontsize=20)


# ## Getting the fitted parameters
# The **result** object contains the fitted values for the parameters.
# 

import lmfit
lmfit.report_fit(mi.params)


# ## Comparing with previous work
# 
# Next, we show values from Mey et al.'s 1993 assessment of the Al-Zn system. We have not done any fitting here; we are just showing how arbitrary values for the parameters may be specified and easily used to plot the phase diagram. Note the optional *parameters* keyword argument specified for each call to **Model**. This feature is what makes it easy to perform arbitrary variable substitutions in our thermodynamic models.
# 

# Parameters from Mey (1993)
assessed_parameters = {
    'LIQALZN0H': 10465.3,
    'LIQALZN0S': 3.39259,
    'FCCALZN0H': 7297.5,
    'FCCALZN0S': -0.47512,
    'FCCALZN1H': 6612.9,
    'FCCALZN1S': 4.5911,
    'FCCALZN2H': -3097.2,
    'FCCALZN2S': -3.306635,
    'HCPALZN0H': 18821.0,
    'HCPALZN0S': 8.95255,
    'HCPALZN3H': -702.8,
    'HCPALZN3S': 0.0
    }

# Initialize phase models
assessed_models = {
    'LIQUID': Model(db_alzn, ['AL', 'ZN'], 'LIQUID', parameters=assessed_parameters),
    'FCC_A1': Model(db_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1', parameters=assessed_parameters),
    'HCP_A3': Model(db_alzn, ['AL', 'ZN', 'VA'], 'HCP_A3', parameters=assessed_parameters)
    }

fig = plt.figure(figsize=(9,6))
get_ipython().magic("time ax = binplot(db_alzn, ['AL', 'ZN', 'VA'] , my_phases_alzn, 'X(ZN)', 300, 1000, ax=plt.gca(), model=assessed_models)")
ax = phase_eq.plot(kind='scatter', x='X(ZN)', y='T', ax=ax, edgecolor='black',
                       s=30, xlim=(0,1), marker='s', facecolor='none')
ax = ax.set_title('Al-Zn assessed by Mey (1993)', fontsize=20)


# ## Quantitative comparison of assessment values with fitted values
# 
# The difference can be partially understood by the fact that Mey's assessment also relies on thermochemical data while here we only fit to points on the phase diagram.
# 

for param_name, param_value in sorted(result.items()):
    percent_change = abs(param_value - assessed_parameters[param_name]) / abs(assessed_parameters[param_name])
    print('{0}: {1:.2%} difference'.format(param_name, percent_change))





# # Fitting experimental data to thermodynamic models
# 

# Besides calculating and visualizing phase equilibria using known models, **pycalphad** also enables you to fit experimental data to a parameterized thermodynamic model. While pycalphad handles construction of the energy functions and equilibrium calculation, a separate module handles the numerical details of fitting disparate sources of data to the thermodynamic functions.
# 
# Here we consider the case study of the **Al-Mg-Zn** system. First, we use a TDB file as the setup for our model parameterizations. Note that not all of the parameters have not been filled in. These are what we will fit.
# 
# A+BT+CTln(T)+DT2+E/T+FT3 
# 
# **Model parametrization:** Liang, H., Chen, S. L., & Chang, Y. A. (1997). A thermodynamic description of the Al-Mg-Zn system. _Metallurgical and Materials Transactions A_, 28(9), 1725-1734.
# 

get_ipython().magic('matplotlib inline')

TDB_STR = """
$=============================================================================
$
$  Al-Mg-Zn System, almgzn.tdb
$  Parameterized by R. Otis from an Al-Cu-Mg-Zn database, 04-13-2015
$  Last updated: 01-07-1997 by H. Liang
$                                             
$  Based on Al-Mg-Zn[96Liang]
$
$=============================================================================
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT MG   HCP_A3                    2.4305E+01  4.9980E+03  3.2671E+01!
 ELEMENT ZN   HCP_A3                    6.5380E+01  5.6567E+03  4.1631E+01!

 
 FUNCTION UN_ASS     2.98140E+02 0.0; 3.00000E+02  N !
 FUNCTION GHSERAL    2.98130E+02  -7976.15+137.071542*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
      -11276.24+223.02695*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
      -11277.683+188.661987*T-31.748192*T*LN(T)-1.234264E+28*T**(-9);
     2.90000E+03  N !
 FUNCTION GHSERMG    2.98130E+02  -8367.34+143.677875*T-26.1849782*T*LN(T)
     +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1);  9.23000E+02  Y
      -14130.185+204.718543*T-34.3088*T*LN(T)+1.038192E+28*T**(-9);
     3.00000E+03  N !
 FUNCTION GHSERZN    2.98140E+02  -7285.787+118.470069*T-23.701314*T*LN(T)
   -.001712034*T**2-1.264963E-06*T**3;  6.92680E+02  Y
   -11070.559+172.34566*T-31.38*T*LN(T)+4.70514E+26*T**(-9);
     1.70000E+03  N !
$
 FUNCTION GALLIQ     2.98130E+02  +11005.553-11.840873*T+7.9401E-20*T**7
     +GHSERAL#;  9.33600E+02  Y
      +10481.974-11.252014*T+1.234264E+28*T**(-9)+GHSERAL#;  
     2.90000E+03  N !
 FUNCTION GMGLIQ     2.98130E+02  +8202.24-8.83693*T-8.01759E-20*T**7
     +GHSERMG#;  9.23000E+02  Y
      +8690.32-9.39216*T-1.03819E+28*T**(-9)+GHSERMG#;  3.00000E+03  N !
 FUNCTION GZNLIQ     2.70000E+02  -128.517+108.176926*T-23.701314*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3-3.58652E-19*T**7;  6.92730E+02  Y
      -3620.474+161.608677*T-31.38*T*LN(T);  2.90000E+03  N !
$
 FUNCTION GBCCAL  2.98150E+02  +10083-4.813*T+GHSERAL#; 6.00000E+03  N !
 FUNCTION GBCCZN  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCZN  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCMG  2.98150E+02  2600-0.9*T+GHSERMG#;
		  6.00000E+03 N !

 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT SPECIE 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
 
 PHASE LIQUID:L %  1  1.0  !
    CONSTITUENT LIQUID:L :AL,MG,ZN :  !
 
   PARAMETER G(LIQUID,AL;0)  2.98130E+02  +GALLIQ#;
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,MG;0)  2.98130E+02  +GMGLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,ZN;0)  2.98130E+02  +GZNLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,MG;0)  2.98150E+02  LIQALMG0A+LIQALMG0B*T;   
				6.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,ZN;0)  2.98150E+02  LIQALZN0A+LIQALZN0B*T;
				6.00000E+03   N REF: 3 !
   PARAMETER G(LIQUID,AL,ZN;1)  2.98150E+02  LIQALZN1A+LIQALZN1B*T;
				6.00000E+03   N REF: 3 !                      
   PARAMETER G(LIQUID,MG,ZN;0) 298.15  LIQMGZN0A+LIQMGZN0B*T+LIQMGZN0C*T*LN(T); 
			       6000.0   N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;1) 298.15  LIQMGZN1A+LIQMGZN1B*T; 
			       6000.0 N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;2) 298.15  LIQMGZN2A;   
			       6000.0   N REF:   4 ! 
$
   PARAMETER G(LIQUID,AL,MG,ZN;0) 298.15 LIQALMGZN0A;  
				  6000.0  N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;1) 298.15 LIQALMGZN1A; 
				  6000.0 N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;2) 298.15 LIQALMGZN2A;  
				  6000.0  N REF:  0 !

$----------------------------------------------------------------------
 TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01 !

 PHASE FCC_A1  %'  2 1   1 !
    CONSTITUENT FCC_A1  :AL%,MG,ZN : VA% :  !
 
   PARAMETER G(FCC_A1,AL:VA;0)  2.98130E+02  +GHSERAL#;  
				2.90000E+03  N REF:  0 !
   PARAMETER G(FCC_A1,MG:VA;0)  2.98130E+02  +2600-.9*T+GHSERMG#;
				3.00000E+03  N REF: 0 !
   PARAMETER G(FCC_A1,ZN:VA;0)  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(FCC_A1,AL,MG:VA;0)  2.98150E+02  FCCALMG0A+FCCALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,MG:VA;1)  2.98150E+02  FCCALMG1A+FCCALMG1B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,ZN:VA;0)  2.98150E+02  FCCALZN0A+FCCALZN0B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;1)  2.98150E+02  FCCALZN1A+FCCALZN1B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;2)  2.98150E+02  FCCALZN2A+FCCALZN2B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,MG,ZN:VA;0)  2.98150E+02  FCCMGZN0A; 
				   6000.0        N REF: 4 !
   PARAMETER G(FCC_A1,AL,MG,ZN:VA;0) 298.15 FCCALMGZN0A;  
				     6000.0  N REF:  0 !
   
$----------------------------------------------------------------------

 TYPE_DEFINITION ( GES A_P_D HCP_A3 MAGNETIC  -3.0    2.80000E-01 !
 
 PHASE HCP_A3  %(  2 1   .5 !
    CONSTITUENT HCP_A3  :AL,MG%,ZN : VA% :  !
 
   PARAMETER G(HCP_A3,AL:VA;0)  2.98130E+02  +5481-1.8*T+GHSERAL#;
				2.90000E+03  N REF: 0 !
   PARAMETER G(HCP_A3,MG:VA;0)  2.98130E+02  +GHSERMG#;  
				3.00000E+03  N REF:  0 !
   PARAMETER G(HCP_A3,ZN:VA;0)  2.98150E+02  +GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(HCP_A3,AL,MG:VA;0)  2.98150E+02  HCPALMG0A+HCPALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(HCP_A3,AL,MG:VA;1)  2.98150E+02  HCPALMG1A;   
				   6.00000E+03   N  REF: 0 !
   PARAMETER G(HCP_A3,AL,ZN:VA;0)  2.98150E+02  HCPALZN0A;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(HCP_A3,MG,ZN:VA;0) 298.15  HCPMGZN0A+HCPMGZN0B*T; 
				  6000.0 N REF: 4 !
   PARAMETER G(HCP_A3,MG,ZN:VA;1) 298.15  HCPMGZN1A+HCPMGZN1B*T; 
				  6000.0 N REF: 4 !

$-------------------------------------------------------------------

  PHASE SIGMA  %  2 .66667    .33333 !
    CONSTITUENT SIGMA  :AL,ZN : MG :  !
 
   PARAMETER G(SIGMA,AL:MG;0)  2.98150E+02  +20133.73+6.3946*T
			       +.66667*GALLIQ#+.33333*GMGLIQ#;  
			       3.00000E+03  N REF: 0 !
   PARAMETER G(SIGMA,ZN:MG;0)  2.98150E+02  -19389.65+13.644*T
			       +.66667*GZNLIQ#+.33333*GMGLIQ#; 
			       3.00000E+03  N REF: 0 ! 
   PARAMETER G(SIGMA,AL,ZN:MG;0) 2.98150E+02  SIGALZN0A;  
				 3.00000E+03  N  REF: 0 !
   PARAMETER G(SIGMA,AL,ZN:MG;1) 2.98150E+02  SIGALZN1A;  
				 3.00000E+03  N  REF: 0 !

$----------------------------------------------------------------

 PHASE T  %  2 .605   .395 !
    CONSTITUENT T  :AL,ZN : MG :  !
 
   PARAMETER G(T,AL:MG;0)  2.98150E+02  -10910.836+8.71*T
			   +.605*GALLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,ZN:MG;0)  2.98150E+02  -15733.501+12.6746*T
			   +.605*GZNLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,AL,ZN:MG;0)  2.98150E+02  TALZN0A+TALZN0B*T;  
			      3.00000E+03  N  REF: 0 !
   PARAMETER G(T,AL,ZN:MG;1)  2.98150E+02  TALZN1A;  
			      3.00000E+03  N REF:   0 !
$----------------------------------------------------------------- 

 PHASE EPS  %  2 1   1 !
    CONSTITUENT EPS  :AL,MG,ZN : VA :  !

   PARAMETER G(EPS,AL:VA;0)  2.98150E+02   5481-1.8*T+GHSERAL#;
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,MG:VA;0)  2.98150E+02  +10+GFCCMG#;     
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,ZN:VA;0)  2.98150E+02  +GFCCZN#;
			     6.00000E+03   N REF: 0 !
   PARAMETER G(EPS,AL,ZN:VA;0)  2.98150E+02   EPSALZN0A;
				6.00000E+03   N REF: 0 !
                
$-----------------------------------------------------------------              

 TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    4.00000E-01 !
 
  PHASE BCC_A2  %  2 1   3 !
    CONSTITUENT BCC_A2  :AL,ZN : VA% :  !

   PARAMETER G(BCC_A2,AL:VA;0)  2.98150E+02  +10083-4.813*T+GHSERAL#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,ZN:VA;0)  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,AL,ZN:VA;0)  2.98150E+02   BCCALZN0A;
				   6.00000E+03  N REF: 0 !

$------------------------------------------------------------------
 
PHASE GAMMA  %  2 1   1 !
    CONSTITUENT GAMMA  :AL,ZN : VA :  !

   PARAMETER G(GAMMA,AL:VA;0)  2.98150E+02  +GHSERAL#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,ZN:VA;0)  2.98150E+02  +GHSERZN#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,AL,ZN:VA;0)  2.98150E+02  GAMALZN0A;
				  6.00000E+03  N REF: 0 !
$---------------------------------------------------------------------

 PHASE PHI  %  3  2  5  2 !
    CONSTITUENT PHI  :AL : MG : ZN :  !

  PARAMETER G(PHI,AL:MG:ZN;0)    298.15  -169985.46+136.8*T
	+2*GALLIQ#+5*GMGLIQ#+2*GZNLIQ#;  6000.0 N REF: 0 !


$====================================================================== 
$              Binary Intermetallic Phases
$======================================================================
 
$---------------------------------------------------------------------

 PHASE ALMG_BETA  %  2 .615   .385 !
    CONSTITUENT ALMG_BETA  :AL : MG :  !
 
   PARAMETER G(ALMG_BETA,AL:MG;0)  2.98150E+02  -1451.1-1.907*T
	      +.615*GHSERAL#+.385*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$---------------------------------------------------------------------
 
 PHASE ALMG_EPSILON  %  2 .56   .44 !
    CONSTITUENT ALMG_EPSILON  :AL : MG :  !
   PARAMETER G(ALMG_EPSILON,AL:MG;0)  2.98150E+02  ALMG_EPSILONALMG0A+ALMG_EPSILONALMG0B*T
	       +.56*GHSERAL#+.44*GHSERMG#;   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
 
 PHASE ALMG_GAMMA  %  3 .4483   .1379   .4138 !
    CONSTITUENT ALMG_GAMMA  :MG : AL,MG : AL,MG :  !

   PARAMETER G(ALMG_GAMMA,MG:AL:AL;0)  2.98150E+02  ALMG_GAMMAMGALAL0A+ALMG_GAMMAMGALAL0B*T
	   +.5517*GHSERAL#+.4483*GHSERMG#;  6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:AL;0)  2.98150E+02  ALMG_GAMMAMGMGAL0A+ALMG_GAMMAMGMGAL0B*T
	   +.4138*GHSERAL#+.5862*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:AL:MG;0)  2.98150E+02  ALMG_GAMMAMGALMG0A+ALMG_GAMMAMGALMG0B*T
	   +.1379*GHSERAL#+.8621*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:MG;0)  2.98150E+02  ALMG_GAMMAMGMGMG0A+GHSERMG#;
	   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
$ 
 PHASE ALMG_ZETA  %  2 .525   .475 !
    CONSTITUENT ALMG_ZETA  :AL : MG :  !
 
   PARAMETER G(ALMG_ZETA,AL:MG;0)  2.98150E+02  -837.8-3.163*T
	    +.525*GHSERAL#+.475*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$----------------------------------------------------------------------
$
 PHASE MG2ZN11  %  2 .153846   .846154 !
    CONSTITUENT MG2ZN11  :MG : ZN :  !
 
  PARAMETER G(MG2ZN11,MG:ZN;0)  298.15  -5823.05+1.94323*T
  +0.153846154*GHSERMG#+0.846153846*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG2ZN3  %  2 .4   .6 !
    CONSTITUENT MG2ZN3  :MG : ZN :  !
 
  PARAMETER G(MG2ZN3,MG:ZN;0)  298.15  -11014.5+3.67151*T+0.4*GHSERMG#
			       +0.6*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG7ZN3  %  2 .71831   .28169 !
    CONSTITUENT MG7ZN3  :MG : ZN :  !
 
  PARAMETER G(MG7ZN3,MG:ZN;0)  298.15  -4814.11+T+0.71831*GHSERMG#
			       +0.28169*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MGZN  %  2 .48   .52 !
    CONSTITUENT MGZN  :MG : ZN :  !
 
  PARAMETER G(MGZN,MG:ZN;0)  298.15  -9590.44+3.19681*T+0.48*GHSERMG#
			     +0.52*GHSERZN#;   6000.0   N REF: 4 !
$

$===========================================================================
"""

from pycalphad import Database, Model, binplot
from pycalphad.eq.energy_surf import energy_surf
import pycalphad.variables as v
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol
import numpy as np

dbf = Database(TDB_STR)


# ## Prior to fitting
# 
# First, we'll show what the phase diagram looks like without any binary interaction parameters. In this diagram only the pure component lattice stabilities of each phase relative to their standard states are considered.
# 

liang96_parameters = {
    'LIQALMG0A':   -11200,
    'LIQALMG0B':   9.578,
    'LIQALZN0A':   10288.0,
    'LIQALZN0B':   -3.035,
    'LIQALZN1A':   -810.0,
    'LIQALZN1B':   0.471,
    'LIQMGZN0A':   -81439.68,
    'LIQMGZN0B':   518.25,
    'LIQMGZN0C':   -64.7144,
    'LIQMGZN1A':   2627.54,
    'LIQMGZN1B':   2.93061,
    'LIQMGZN2A':   -1673.28,
    'LIQALMGZN0A': -4094.48,
    'LIQALMGZN1A': -39973.74,
    'LIQALMGZN2A': -11337.52,
    'FCCALMG0A':   4945.7,
    'FCCALMG0B':   -1.318,
    'FCCALMG1A':   1594.4,
    'FCCALMG1B':   -.973,
    'FCCALZN0A':   6656.0,
    'FCCALZN0B':   1.615,
    'FCCALZN1A':   6793.0,
    'FCCALZN1B':   -4.982,
    'FCCALZN2A':   -5352.0,
    'FCCALZN2B':   7.261,
    'FCCMGZN0A':   18000,
    'FCCALMGZN0A': -20000,
    'HCPALMG0A':   4063.4,
    'HCPALMG0B':   -3.243,
    'HCPALMG1A':   -1642.1,
    'HCPALZN0A':   14620,
    'HCPMGZN0A':   -1600.77,
    'HCPMGZN0B':   7.62441,
    'HCPMGZN1A':   -3823.03,
    'HCPMGZN1B':   8.02575,
    'ALMG_EPSILONALMG0A': -768.3,
    'ALMG_EPSILONALMG0B': -3.119,
    'ALMG_GAMMAMGALAL0A': -1270,
    'ALMG_GAMMAMGALAL0B': -1.75,
    'ALMG_GAMMAMGMGAL0A': -2441.4,
    'ALMG_GAMMAMGMGAL0B': .219,
    'ALMG_GAMMAMGALMG0A': 1279.6,
    'ALMG_GAMMAMGALMG0B': 1.1606,
    'ALMG_GAMMAMGMGMG0A': 5000,
    'SIGALZN0A':   -23927.13,
    'SIGALZN1A':   9335.47,
    'TALZN0A':     -25696.19,
    'TALZN0B':     25,
    'TALZN1A':     9153.84,
    'EPSALZN0A':   10000,
    'BCCALZN0A':   20000,
    'GAMALZN0A':   75000
    }

phases_alzn = ['LIQUID', 'FCC_A1', 'HCP_A3', 'EPS']
phases_mgzn = ['LIQUID', 'FCC_A1', 'HCP_A3', 'EPS', 'SIGMA', 'T', 'MG2ZN11', 'MG2ZN3', 'MG7ZN3', 'MGZN']
phases_almg = ['LIQUID', 'FCC_A1', 'HCP_A3', 'EPS', 'SIGMA', 'T', 'ALMG_BETA', 'ALMG_GAMMA',
               'ALMG_EPSILON', 'ALMG_ZETA']

alzn_models = {name: Model(dbf, ['AL', 'ZN', 'VA'], name, parameters=liang96_parameters) for name in phases_alzn}
mgzn_models = {name: Model(dbf, ['MG', 'ZN', 'VA'], name, parameters=liang96_parameters) for name in phases_mgzn}
almg_models = {name: Model(dbf, ['AL', 'MG', 'VA'], name, parameters=liang96_parameters) for name in phases_almg}
almg_enthalpies = energy_surf(dbf, ['AL', 'MG', 'VA'], ['FCC_A1', 'HCP_A3'], output='MIX_HM', T=300.0, model=almg_models)
almg_enthalpies.sort('X(MG)', inplace=True) # points need to be ordered on x axis for line plot to work
almg_enthalpies_by_phase = dict(list(almg_enthalpies.groupby('Phase')))
zhong_2005_hcp_sqs = np.array([[0.25, 525], [0.5, 1174], [0.75, 852]])


fig = plt.figure(figsize=(16, 24))
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
binplot(dbf, ['AL', 'ZN', 'VA'], phases_alzn, 'X(ZN)', 300, 1000, ax=ax1, model=alzn_models)
binplot(dbf, ['MG', 'ZN', 'VA'], phases_mgzn, 'X(MG)', 300, 1000, ax=ax2, model=mgzn_models)
binplot(dbf, ['AL', 'MG', 'VA'], phases_almg, 'X(MG)', 300, 1000, ax=ax3, model=almg_models)
ax4.plot(almg_enthalpies_by_phase['HCP_A3']['X(MG)'], almg_enthalpies_by_phase['HCP_A3']['MIX_HM'])
ax4.scatter(zhong_2005_hcp_sqs[:, 0], zhong_2005_hcp_sqs[:, 1], label='Zhong 2005 (SQS)')
ax4.set_xlim((0,1))
ax4.set_ylim(bottom=0)
ax4.set_title('Al-Mg HCP Mixing Enthalpy at 300 K', fontsize=20)
ax4.set_xlabel('X(MG)', fontsize=20)
ax4.set_ylabel('Enthalpy of Mixing (J/mol-atom)', fontsize=20)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.legend()
ax5.plot(almg_enthalpies_by_phase['FCC_A1']['X(MG)'], almg_enthalpies_by_phase['FCC_A1']['MIX_HM'])
#ax5.scatter(zhong_2005_hcp_sqs[:, 0], zhong_2005_hcp_sqs[:, 1], label='Zhong 2005 (SQS)')
ax5.set_xlim((0,1))
ax5.set_ylim(bottom=0)
ax5.set_title('Al-Mg FCC Mixing Enthalpy at 300 K', fontsize=20)
ax5.set_xlabel('X(MG)', fontsize=20)
ax5.set_ylabel('Enthalpy of Mixing (J/mol-atom)', fontsize=20)
ax5.tick_params(axis='both', which='major', labelsize=14)
#ax5.legend()
plt.show()


# ## Performing a fit
# 
# We perform the fit using the **fit_model** routine in pycalphad.
# 

import pandas as pd
from pycalphad.residuals import fit_model

liang96_alzn_parameters = {
    'LIQALZN0A':   10288.0,
    'LIQALZN0B':   -3.035,
    'LIQALZN1A':   -810.0,
    'LIQALZN1B':   0.471,
    'FCCALZN0A':   6656.0,
    'FCCALZN0B':   1.615,
    'FCCALZN1A':   6793.0,
    'FCCALZN1B':   -4.982,
    'FCCALZN2A':   -5352.0,
    'FCCALZN2B':   7.261,
    'HCPALZN0A':   14620,
    }
liang96_almg_parameters = {
    'LIQALMG0A':   -11200,
    'LIQALMG0B':   9.578,
    'FCCALMG0A':   4945.7,
    'FCCALMG0B':   -1.318,
    'FCCALMG1A':   1594.4,
    'FCCALMG1B':   -.973,
    'HCPALMG0A':   4063.4,
    'HCPALMG0B':   -3.243,
    'HCPALMG1A':   -1642.1,
    'ALMG_EPSILONALMG0A': -768.3,
    'ALMG_EPSILONALMG0B': -3.119,
    'ALMG_GAMMAMGALAL0A': -1270,
    'ALMG_GAMMAMGALAL0B': -1.75,
    'ALMG_GAMMAMGMGAL0A': -2441.4,
    'ALMG_GAMMAMGMGAL0B': .219,
    'ALMG_GAMMAMGALMG0A': 1279.6,
    'ALMG_GAMMAMGALMG0B': 1.1606,
    'ALMG_GAMMAMGMGMG0A': 5000,
}
from pycalphad.log import debug_mode
import lmfit
#debug_mode()
phase_eq = pd.read_csv('almg_test_phase_eq.csv')
phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'ALMG_EPSILON', 'ALMG_BETA', 'ALMG_GAMMA']
comps = ['AL', 'MG', 'VA']
start_models = {name: Model(dbf, comps, name, parameters=liang96_almg_parameters) for name in phases}
get_ipython().magic('time result, mi = fit_model(liang96_almg_parameters, phase_eq, dbf, comps, phases, maxfev=50)')
fit_models = {name: Model(dbf, comps, name, parameters=result) for name in phases}


print(lmfit.report_fit(mi.params))
#print(lmfit.conf_interval(mi))


from pycalphad.plot.utils import phase_legend
legend_handles, colors = phase_legend(phases)
almg_start_df = energy_surf(dbf, ['AL', 'MG', 'VA'], phases, T=550.0, model=start_models)
almg_fit_df = energy_surf(dbf, ['AL', 'MG', 'VA'], phases, T=550.0, model=fit_models)
fig = plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
#ax3 = fig.add_subplot(gs[1, 0])
#ax4 = fig.add_subplot(gs[1, 1])
#almg_enthalpies = energy_surf(dbf, ['AL', 'MG', 'VA'], phases, output='MIX_HM', T=300.0, model=fit_models)
#almg_enthalpies.sort('X(MG)', inplace=True) # points need to be ordered on x axis for line plot to work
#almg_enthalpies_by_phase = dict(list(almg_enthalpies.groupby('Phase')))
for phase in phases:
    phase_eq[phase_eq['Phase'] == phase].plot(kind='scatter', x='X(MG)', y='T', ax=ax1, edgecolor=colors[phase],
                                                    s=50, xlim=(0,1), marker='s', linewidth=2, facecolor='none',
                                                    legend=False)
    phase_eq[phase_eq['Phase'] == phase].plot(kind='scatter', x='X(MG)', y='T', ax=ax2, edgecolor=colors[phase],
                                                    s=50, xlim=(0,1), marker='s', linewidth=2, facecolor='none',
                                                    legend=False)
#    ax3 = almg_start_df[almg_start_df['Phase'] == phase].plot(kind='scatter', x='X(MG)', y='GM', ax=ax3, color=colors[phase], s=1, xlim=(0,1))
#    ax4 = almg_fit_df[almg_fit_df['Phase'] == phase].plot(kind='scatter', x='X(MG)', y='GM', ax=ax4, color=colors[phase], s=1, xlim=(0,1))

ax1 = binplot(dbf, comps, phases, 'X(MG)', 300, 1000, ax=ax1, model=start_models, pdens=5000)
ax1.set_title('Al-Mg (Liang 1996)', fontsize=20)
ax2 = binplot(dbf, comps, phases, 'X(MG)', 300, 1000, ax=ax2, model=fit_models, pdens=5000)
ax2.get_legend().set_visible(False)
ax2 = ax2.set_title('Al-Mg leastsq fit', fontsize=20)


# ## Quantitative comparison of assessment values with fitted values
# 
# The difference can be partially understood by the fact that Liang's assessment also relies on thermochemical data while here we only fit to points on the phase diagram.
# 

for param_name, param_value in sorted(result.items()):
    percent_change = abs(param_value - liang96_almg_parameters[param_name]) / abs(liang96_almg_parameters[param_name])
    print('{:<20}: {:>8.2%} difference from original'.format(param_name, percent_change))


from pycalphad.residuals import residual_thermochemical
from pycalphad.eq.utils import make_callable, generate_dof
from pycalphad import Model
import itertools
import lmfit

def fit_model_thermochemical(guess, data, dbf, comps, phases, **kwargs):
    """
    Fit model parameters to input data based on an initial guess for parameters.

    Parameters
    ----------
    guess : dict
        Parameter names to fit with initial guesses.
    data : list of DataFrames
        Input data to fit.
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list
        Names of phases to consider in the calculation.

    Returns
    -------
    (Dictionary of fit key:values), (lmfit minimize result)

    Examples
    --------
    None yet.
    """
    if 'maxfev' not in kwargs:
        kwargs['maxfev'] = 100
    fit_params = lmfit.Parameters()
    for guess_name, guess_value in guess.items():
        fit_params.add(guess_name, value=guess_value)
    param_names = fit_params.valuesdict().keys()
    fit_models = {name: Model(dbf, comps, name) for name in phases}
    fit_variables = dict()
    for name, mod in fit_models.items():
        fit_variables[name], _ = generate_dof(dbf.phases[name], mod.components)
    callables = {name: make_callable(mod.ast, itertools.chain(param_names, [v.T],
                                                              fit_variables[name])) \
                    for name, mod in fit_models.items()}
    out = lmfit.minimize(residual_thermochemical, fit_params,
                         args=(data, dbf, comps, fit_models, callables))
    return fit_params.valuesdict(), out


comps = sorted(['Fe', 'Ni', 'Cr', 'Al', 'Mg', 'Zn'])
import itertools
print(list('-'.join(sorted(x)) for x in itertools.combinations(comps, 2)))
print(list('-'.join(sorted(x)) for x in itertools.combinations(comps, 3) if set(x).issuperset(set(['Fe', 'Al'])))+['Al-Mg-Zn', 'Cr-Fe-Ni'])
print(list('-'.join(sorted(x)) for x in itertools.combinations(comps, 3)))





# # Fitting experimental data to thermodynamic models
# 

# Besides calculating and visualizing phase equilibria using known models, **pycalphad** also enables you to fit experimental data to a parameterized thermodynamic model. While pycalphad handles construction of the energy functions and equilibrium calculation, a separate module handles the numerical details of fitting disparate sources of data to the thermodynamic functions.
# 
# Here we consider the case study of the **Al-Mg-Zn** system. First, we use a TDB file as the setup for our model parameterizations. Note that not all of the parameters have not been filled in. These are what we will fit.
# 
# A+BT+CTln(T)+DT2+E/T+FT3 
# 
# **Model parametrization:** Liang, H., Chen, S. L., & Chang, Y. A. (1997). A thermodynamic description of the Al-Mg-Zn system. _Metallurgical and Materials Transactions A_, 28(9), 1725-1734.
# 

get_ipython().magic('matplotlib inline')

TDB_STR = """
$=============================================================================
$
$  Al-Mg-Zn System, almgzn.tdb
$  Parameterized by R. Otis from an Al-Cu-Mg-Zn database, 04-13-2015
$  Last updated: 01-07-1997 by H. Liang
$                                             
$  Based on Al-Mg-Zn[96Liang]
$
$=============================================================================
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT MG   HCP_A3                    2.4305E+01  4.9980E+03  3.2671E+01!
 ELEMENT ZN   HCP_A3                    6.5380E+01  5.6567E+03  4.1631E+01!

 
 FUNCTION UN_ASS     2.98140E+02 0.0; 3.00000E+02  N !
 FUNCTION GHSERAL    2.98130E+02  -7976.15+137.071542*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
      -11276.24+223.02695*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
      -11277.683+188.661987*T-31.748192*T*LN(T)-1.234264E+28*T**(-9);
     2.90000E+03  N !
 FUNCTION GHSERMG    2.98130E+02  -8367.34+143.677875*T-26.1849782*T*LN(T)
     +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1);  9.23000E+02  Y
      -14130.185+204.718543*T-34.3088*T*LN(T)+1.038192E+28*T**(-9);
     3.00000E+03  N !
 FUNCTION GHSERZN    2.98140E+02  -7285.787+118.470069*T-23.701314*T*LN(T)
   -.001712034*T**2-1.264963E-06*T**3;  6.92680E+02  Y
   -11070.559+172.34566*T-31.38*T*LN(T)+4.70514E+26*T**(-9);
     1.70000E+03  N !
$
 FUNCTION GALLIQ     2.98130E+02  +11005.553-11.840873*T+7.9401E-20*T**7
     +GHSERAL#;  9.33600E+02  Y
      +10481.974-11.252014*T+1.234264E+28*T**(-9)+GHSERAL#;  
     2.90000E+03  N !
 FUNCTION GMGLIQ     2.98130E+02  +8202.24-8.83693*T-8.01759E-20*T**7
     +GHSERMG#;  9.23000E+02  Y
      +8690.32-9.39216*T-1.03819E+28*T**(-9)+GHSERMG#;  3.00000E+03  N !
 FUNCTION GZNLIQ     2.70000E+02  -128.517+108.176926*T-23.701314*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3-3.58652E-19*T**7;  6.92730E+02  Y
      -3620.474+161.608677*T-31.38*T*LN(T);  2.90000E+03  N !
$
 FUNCTION GBCCAL  2.98150E+02  +10083-4.813*T+GHSERAL#; 6.00000E+03  N !
 FUNCTION GBCCZN  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCZN  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCMG  2.98150E+02  2600-0.9*T+GHSERMG#;
		  6.00000E+03 N !

 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT SPECIE 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
 
 PHASE LIQUID:L %  1  1.0  !
    CONSTITUENT LIQUID:L :AL,MG,ZN :  !
 
   PARAMETER G(LIQUID,AL;0)  2.98130E+02  +GALLIQ#;
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,MG;0)  2.98130E+02  +GMGLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,ZN;0)  2.98130E+02  +GZNLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,MG;0)  2.98150E+02  LIQALMG0A+LIQALMG0B*T;   
				6.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,ZN;0)  2.98150E+02  LIQALZN0A+LIQALZN0B*T;
				6.00000E+03   N REF: 3 !
   PARAMETER G(LIQUID,AL,ZN;1)  2.98150E+02  LIQALZN1A+LIQALZN1B*T;
				6.00000E+03   N REF: 3 !                      
   PARAMETER G(LIQUID,MG,ZN;0) 298.15  LIQMGZN0A+LIQMGZN0B*T+LIQMGZN0C*T*LN(T); 
			       6000.0   N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;1) 298.15  LIQMGZN1A+LIQMGZN1B*T; 
			       6000.0 N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;2) 298.15  LIQMGZN2A;   
			       6000.0   N REF:   4 ! 
$
   PARAMETER G(LIQUID,AL,MG,ZN;0) 298.15 LIQALMGZN0A;  
				  6000.0  N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;1) 298.15 LIQALMGZN1A; 
				  6000.0 N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;2) 298.15 LIQALMGZN2A;  
				  6000.0  N REF:  0 !

$----------------------------------------------------------------------
 TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01 !

 PHASE FCC_A1  %'  2 1   1 !
    CONSTITUENT FCC_A1  :AL%,MG,ZN : VA% :  !
 
   PARAMETER G(FCC_A1,AL:VA;0)  2.98130E+02  +GHSERAL#;  
				2.90000E+03  N REF:  0 !
   PARAMETER G(FCC_A1,MG:VA;0)  2.98130E+02  +2600-.9*T+GHSERMG#;
				3.00000E+03  N REF: 0 !
   PARAMETER G(FCC_A1,ZN:VA;0)  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(FCC_A1,AL,MG:VA;0)  2.98150E+02  FCCALMG0A+FCCALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,MG:VA;1)  2.98150E+02  0+0*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,ZN:VA;0)  2.98150E+02  FCCALZN0A+FCCALZN0B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;1)  2.98150E+02  FCCALZN1A+FCCALZN1B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;2)  2.98150E+02  FCCALZN2A+FCCALZN2B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,MG,ZN:VA;0)  2.98150E+02  FCCMGZN0A; 
				   6000.0        N REF: 4 !
   PARAMETER G(FCC_A1,AL,MG,ZN:VA;0) 298.15 FCCALMGZN0A;  
				     6000.0  N REF:  0 !
   
$----------------------------------------------------------------------

 TYPE_DEFINITION ( GES A_P_D HCP_A3 MAGNETIC  -3.0    2.80000E-01 !
 
 PHASE HCP_A3  %(  2 1   .5 !
    CONSTITUENT HCP_A3  :AL,MG%,ZN : VA% :  !
 
   PARAMETER G(HCP_A3,AL:VA;0)  2.98130E+02  +5481-1.8*T+GHSERAL#;
				2.90000E+03  N REF: 0 !
   PARAMETER G(HCP_A3,MG:VA;0)  2.98130E+02  +GHSERMG#;  
				3.00000E+03  N REF:  0 !
   PARAMETER G(HCP_A3,ZN:VA;0)  2.98150E+02  +GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(HCP_A3,AL,MG:VA;0)  2.98150E+02  HCPALMG0A+HCPALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(HCP_A3,AL,MG:VA;1)  2.98150E+02  HCPALMG1A;   
				   6.00000E+03   N  REF: 0 !
   PARAMETER G(HCP_A3,AL,ZN:VA;0)  2.98150E+02  HCPALZN0A;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(HCP_A3,MG,ZN:VA;0) 298.15  HCPMGZN0A+HCPMGZN0B*T; 
				  6000.0 N REF: 4 !
   PARAMETER G(HCP_A3,MG,ZN:VA;1) 298.15  HCPMGZN1A+HCPMGZN1B*T; 
				  6000.0 N REF: 4 !

$-------------------------------------------------------------------

  PHASE SIGMA  %  2 .66667    .33333 !
    CONSTITUENT SIGMA  :AL,ZN : MG :  !
 
   PARAMETER G(SIGMA,AL:MG;0)  2.98150E+02  +20133.73+6.3946*T
			       +.66667*GALLIQ#+.33333*GMGLIQ#;  
			       3.00000E+03  N REF: 0 !
   PARAMETER G(SIGMA,ZN:MG;0)  2.98150E+02  -19389.65+13.644*T
			       +.66667*GZNLIQ#+.33333*GMGLIQ#; 
			       3.00000E+03  N REF: 0 ! 
   PARAMETER G(SIGMA,AL,ZN:MG;0) 2.98150E+02  SIGALZN0A;  
				 3.00000E+03  N  REF: 0 !
   PARAMETER G(SIGMA,AL,ZN:MG;1) 2.98150E+02  SIGALZN1A;  
				 3.00000E+03  N  REF: 0 !

$----------------------------------------------------------------

 PHASE T  %  2 .605   .395 !
    CONSTITUENT T  :AL,ZN : MG :  !
 
   PARAMETER G(T,AL:MG;0)  2.98150E+02  -10910.836+8.71*T
			   +.605*GALLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,ZN:MG;0)  2.98150E+02  -15733.501+12.6746*T
			   +.605*GZNLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,AL,ZN:MG;0)  2.98150E+02  TALZN0A+TALZN0B*T;  
			      3.00000E+03  N  REF: 0 !
   PARAMETER G(T,AL,ZN:MG;1)  2.98150E+02  TALZN1A;  
			      3.00000E+03  N REF:   0 !
$----------------------------------------------------------------- 

 PHASE EPS  %  2 1   1 !
    CONSTITUENT EPS  :AL,MG,ZN : VA :  !

   PARAMETER G(EPS,AL:VA;0)  2.98150E+02   5481-1.8*T+GHSERAL#;
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,MG:VA;0)  2.98150E+02  +10+GFCCMG#;     
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,ZN:VA;0)  2.98150E+02  +GFCCZN#;
			     6.00000E+03   N REF: 0 !
   PARAMETER G(EPS,AL,ZN:VA;0)  2.98150E+02   EPSALZN0A;
				6.00000E+03   N REF: 0 !
                
$-----------------------------------------------------------------              

 TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    4.00000E-01 !
 
  PHASE BCC_A2  %  2 1   3 !
    CONSTITUENT BCC_A2  :AL,ZN : VA% :  !

   PARAMETER G(BCC_A2,AL:VA;0)  2.98150E+02  +10083-4.813*T+GHSERAL#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,ZN:VA;0)  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,AL,ZN:VA;0)  2.98150E+02   BCCALZN0A;
				   6.00000E+03  N REF: 0 !

$------------------------------------------------------------------
 
PHASE GAMMA  %  2 1   1 !
    CONSTITUENT GAMMA  :AL,ZN : VA :  !

   PARAMETER G(GAMMA,AL:VA;0)  2.98150E+02  +GHSERAL#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,ZN:VA;0)  2.98150E+02  +GHSERZN#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,AL,ZN:VA;0)  2.98150E+02  GAMALZN0A;
				  6.00000E+03  N REF: 0 !
$---------------------------------------------------------------------

 PHASE PHI  %  3  2  5  2 !
    CONSTITUENT PHI  :AL : MG : ZN :  !

  PARAMETER G(PHI,AL:MG:ZN;0)    298.15  -169985.46+136.8*T
	+2*GALLIQ#+5*GMGLIQ#+2*GZNLIQ#;  6000.0 N REF: 0 !


$====================================================================== 
$              Binary Intermetallic Phases
$======================================================================
 
$---------------------------------------------------------------------

 PHASE ALMG_BETA  %  2 .615   .385 !
    CONSTITUENT ALMG_BETA  :AL : MG :  !
 
   PARAMETER G(ALMG_BETA,AL:MG;0)  2.98150E+02  -1451.1-1.907*T
	      +.615*GHSERAL#+.385*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$---------------------------------------------------------------------
 
 PHASE ALMG_EPSILON  %  2 .56   .44 !
    CONSTITUENT ALMG_EPSILON  :AL : MG :  !
   PARAMETER G(ALMG_EPSILON,AL:MG;0)  2.98150E+02  ALMG_EPSILONALMG0A+ALMG_EPSILONALMG0B*T
	       +.56*GHSERAL#+.44*GHSERMG#;   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
 
 PHASE ALMG_GAMMA  %  3 .4483   .1379   .4138 !
    CONSTITUENT ALMG_GAMMA  :MG : AL,MG : AL,MG :  !

   PARAMETER G(ALMG_GAMMA,MG:AL:AL;0)  2.98150E+02  ALMG_GAMMAMGALAL0A+ALMG_GAMMAMGALAL0B*T
	   +.5517*GHSERAL#+.4483*GHSERMG#;  6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:AL;0)  2.98150E+02  ALMG_GAMMAMGMGAL0A+ALMG_GAMMAMGMGAL0B*T
	   +.4138*GHSERAL#+.5862*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:AL:MG;0)  2.98150E+02  ALMG_GAMMAMGALMG0A+ALMG_GAMMAMGALMG0B*T
	   +.1379*GHSERAL#+.8621*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:MG;0)  2.98150E+02  ALMG_GAMMAMGMGMG0A+GHSERMG#;
	   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
$ 
 PHASE ALMG_ZETA  %  2 .525   .475 !
    CONSTITUENT ALMG_ZETA  :AL : MG :  !
 
   PARAMETER G(ALMG_ZETA,AL:MG;0)  2.98150E+02  -837.8-3.163*T
	    +.525*GHSERAL#+.475*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$----------------------------------------------------------------------
$
 PHASE MG2ZN11  %  2 .153846   .846154 !
    CONSTITUENT MG2ZN11  :MG : ZN :  !
 
  PARAMETER G(MG2ZN11,MG:ZN;0)  298.15  -5823.05+1.94323*T
  +0.153846154*GHSERMG#+0.846153846*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG2ZN3  %  2 .4   .6 !
    CONSTITUENT MG2ZN3  :MG : ZN :  !
 
  PARAMETER G(MG2ZN3,MG:ZN;0)  298.15  -11014.5+3.67151*T+0.4*GHSERMG#
			       +0.6*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG7ZN3  %  2 .71831   .28169 !
    CONSTITUENT MG7ZN3  :MG : ZN :  !
 
  PARAMETER G(MG7ZN3,MG:ZN;0)  298.15  -4814.11+T+0.71831*GHSERMG#
			       +0.28169*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MGZN  %  2 .48   .52 !
    CONSTITUENT MGZN  :MG : ZN :  !
 
  PARAMETER G(MGZN,MG:ZN;0)  298.15  -9590.44+3.19681*T+0.48*GHSERMG#
			     +0.52*GHSERZN#;   6000.0   N REF: 4 !
$

$===========================================================================
"""

from pycalphad import Database, Model, binplot
import pycalphad.variables as v
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol
import numpy as np

dbf = Database(TDB_STR)


# ## Performing a fit
# 
# We perform the fit using the **fit_model** routine in pycalphad.
# 

import pandas as pd
from pycalphad.residuals import fit_model

liang96_almg_parameters = {
    'FCCALMG0A':   1000,
    'FCCALMG0B':   10,
}
import lmfit
phase_eq = pd.read_csv('almg_fcc_enthalpy.csv')
#phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'ALMG_EPSILON', 'ALMG_BETA', 'ALMG_GAMMA']
phases = ['FCC_A1']
comps = ['AL', 'MG', 'VA']
start_models = {name: Model(dbf, comps, name, parameters=liang96_almg_parameters) for name in phases}
get_ipython().magic('time result, mi = fit_model(liang96_almg_parameters, phase_eq, dbf, comps, phases, maxfev=50)')
fit_models = {name: Model(dbf, comps, name, parameters=result) for name in phases}


print(lmfit.report_fit(mi.params))
print(lmfit.conf_interval(mi))


# # Comparison using AlgoPy
# https://github.com/b45ch1/algopy
# 

import numpy as np
import algopy
import autograd.numpy as anp


def func(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, mod=anp):
    "Toy function (calibrated to take roughly the same time as real one)"
    return 8.3145 * x0 * (x1 * mod.log(x1) + x2 * mod.log(x2)) + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 200*x2*x3*(x1-x2)**2 + 2000*x3*x4 + 100*x5*x6*(x1-x2) + 200*x7*x8*(x8-x9)**2 + 2400*x1*x2 + 120*x1*x2*(x1-x2) + 200*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 200*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 160*x1*x2*(x1-x2) + 1000*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 500*x1*x2*(x1-x2)**2


# We repeat the same two values over and over to simulate lots of different input
C = 20000
N = 10
inp_arr = np.tile([[300,0.5,0.5, 300,0.5,0.5, 300,0.5,0.5, 1], [600, 0.4, 0.6, 600, 0.4, 0.6, 600, 0.4, 0.6, 1]], (int(C/2),1))
print(inp_arr.shape)


get_ipython().run_cell_magic('timeit', '', '\n# record computational graph\ncg = algopy.CGraph()\nfx = algopy.Function(np.ones(N))\nfy = func(fx[0], fx[1], fx[2], fx[3], fx[4], fx[5], fx[6], fx[7], fx[8], fx[9], mod=algopy)\ncg.independentFunctionList = [fx]\ncg.dependentFunctionList = [fy]')


get_ipython().run_cell_magic('timeit', '', '\n# compute obj and grad in reverse mode\nux = algopy.UTPM(np.zeros((1, C, N)))\nux.data[0, ...] = inp_arr\ncg.pushforward([ux])\nuy_bar = algopy.UTPM(np.ones((1,C)))\ncg.pullback([uy_bar])\n\nobj_cgraph = cg.dependentFunctionList[0].x.data[0]\ngrad_cgraph = cg.independentFunctionList[0].xbar.data[0,:,:]')


get_ipython().run_cell_magic('timeit', '', '\n# compute obj, grad, hess in combined forward+reverse mode\nux = algopy.UTPM(np.zeros((2, N*C, N)))\nfor i in range(N):\n    ux.data[0, i::N, :] = inp_arr\n    ux.data[1, i::N, i::N] = 1\n\ncg.pushforward([ux])\nuy_bar = algopy.UTPM(np.zeros((2,N*C)))\nuy_bar.data[0,:] = 1\ncg.pullback([uy_bar])\n\nobj_cgraph = cg.dependentFunctionList[0].x.data[0, ::N]\ngrad_cgraph = cg.independentFunctionList[0].xbar.data[0,::N]\nhess_cgraph = cg.independentFunctionList[0].xbar.data[1,:,:].reshape((C, N, N))')


def extract_hessian(N, y):
    "Vectorized version of extract_hessian"
    H = np.zeros((y.data.shape[1], N,N), dtype=y.data.dtype)
    for n in range(N):
        for m in range(n):
            a =  sum(range(n+1))
            b =  sum(range(m+1))
            k =  sum(range(n+2)) - m - 1
            #print 'k,a,b=', k,a,b
            if n!=m:
                tmp = (y.data[2, :, k] - y.data[2, :, a] - y.data[2, :, b])
                H[:, m,n]= H[:, n,m]= tmp
        a =  sum(range(n+1))
        H[:, n,n] = 2*y.data[2, :, a]
    return H


# generate directions
M = (N*(N+1))/2
L = (N*(N-1))/2
S = np.zeros((N,M))

s = 0
i = 0
for n in range(1,N+1):
    S[-n:, s:s+n] = np.eye(n)
    S[-n, s:s+n] = np.ones(n)
    s += n
    i += 1
#print(S)
S = S[::-1].T
#print(S)
x = algopy.UTPM(np.zeros((3, inp_arr.shape[0]) + S.shape))
x.data[0, :, :, :] = inp_arr[..., None, :]
x.data[1, :, :] = S


get_ipython().run_cell_magic('timeit', '', "y = func(*[x[..., i] for i in range(N)], mod=algopy)\nobj_algopy = y.data[0, :, 0]\ngrad_algopy = y.data[1, :, np.cumsum(np.arange(N), dtype=np.int)].T\nhess_algopy = extract_hessian(N, y)\n#print('OBJ SHAPE', obj_algopy.shape)\n#print('GRAD SHAPE', grad_algopy.shape)\n#print('HESS SHAPE', hess_algopy.shape)")


# # Comparison using Autograd
# 
# https://github.com/HIPS/autograd
# 

import autograd.numpy as anp
from autograd import elementwise_grad, jacobian
from itertools import chain

def elementwise_hess(fun, argnum=0):
    """
    From https://github.com/HIPS/autograd/issues/60
    """
    def sum_latter_dims(x):
        return anp.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(fun)(*args, **kwargs))
    return jacobian(sum_grad_output, argnum)


def build_functions():
    obj = func

    def argwrapper(args):
        return obj(*args)

    def grad_func(*args):
        inp = anp.array(anp.broadcast_arrays(*args))
        result = anp.atleast_2d(elementwise_grad(argwrapper)(inp))
        # Put 'gradient' axis at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[1:], [axes[0]]))
        return result

    def hess_func(*args):
        result = anp.atleast_3d(elementwise_hess(argwrapper)(anp.array(anp.broadcast_arrays(*args))))
        # Put 'hessian' axes at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[2:], axes[0:2]))
        return result

    return obj, grad_func, hess_func

autograd_obj, autograd_grad, autograd_hess = build_functions()


get_ipython().run_cell_magic('timeit', '', 'ag_o = autograd_obj(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])\nag_g = autograd_grad(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])\nag_h = autograd_hess(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])')


# Have to comment out '%%timeit' lines to actually run this
import numpy.testing

numpy.testing.assert_allclose(obj_algopy, ag_o)
numpy.testing.assert_allclose(grad_algopy, ag_g)
numpy.testing.assert_allclose(hess_algopy, ag_h)

numpy.testing.assert_allclose(obj_cgraph, ag_o)
numpy.testing.assert_allclose(grad_cgraph, ag_g)
numpy.testing.assert_allclose(hess_cgraph, ag_h)

print('equivalent')








