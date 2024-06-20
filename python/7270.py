# # Notebook Tutorials
# 
# [Using Sensitivity Analysis to Interrogate Models](./Using%20SA%20to%20Interrogate%20Models/Using%20Sensitivity%20Analysis.ipynb)
# 




# # Using Sensitivity Analysis to Interrogate Models
# 
# Will Usher, Environmental Change Institute, University of Oxford 
# 
# <img style="float: right;" src="http://www.eci.ox.ac.uk/assets/img/eci-logo-colour.png">
# 
# 8th December 2016
# 

# ## How to use this notebook
# 
# There are several alternative ways in which you can use this notebook.
# 1. Access online here [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/salib/satut) (changes won't be saved)
# 2. [Download](http://www.github.com/SALib/SATut) the notebooks, and install jupyter and python
#     1. For Mac, open terminal and type
#         ```bash
#         pip install jupyter notebook ipython
#         ```
#     2. Then
#         ```bash
#         jupyter notebook
#         ```
# 

# # Agenda
# 
# 1. [What is Sensitivity Analysis?](#What-is-Sensitivity-Analysis?)
# 1. [Sensitivity Analysis Techniques](#Sensitivity-Analysis-Techniques)
# 1. [An Interactive Example](#Sensitivity-Analysis-in-Practice:-Vehicle-to-Grid)
# 1. [Summary](#Summary)
# 

from ipywidgets import widgets, interact
from IPython.display import display
get_ipython().magic('matplotlib inline')
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
figsize(12, 10)
sbn.set_context("talk", font_scale=1)

# The model used for this seminar is contained in the file model.py
from model import (cost_of_vehicle_to_grid, compute_profit, 
                   annualized_capital_cost, battery_lifetime, 
                   max_vehicle_power)


# Uncomment the next line and run this cell to view the model code in this notebook
# %load model.py


# # Uncertainty and Modelling
# 
# We use models to encode natural phenomena, to project and forecast, to understand, to learn.
# 
# Examples of models:
# * discounted cash flow analysis
# * [Gina coefficient](https://en.wikipedia.org/wiki/Gini_coefficient#Based_on_just_two_levels_of_income) (statistical measure of inequality)
# * UKTM - energy system model of the United Kingdom
# * [MetUM](http://www.metoffice.gov.uk/research/modelling-systems/unified-model) - UK weather forecasting
# 

# ## What is Sensitivity Analysis?
# 
# “...the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be apportioned to different sources of uncertainty in its inputs.” 
# 
# There are four settings for sensitivity analysis:
# * **Factor prioritisation** - which parameters are most influential?
# * **Factor fixing** - which parameters can we ignore?
# * Factor mapping - which inputs matter for just this space in the model output?
# * (Metamodelling) - which parameters can I use to model my model?
# 
# There are two families of approaches:
# * Local approaches
# * Global approaches

# ### Local versus Global Approaches
# 
# * Local approaches
#     * e.g. one-at-a-time (OAT) approach
#     * low data requirements
#     * quick and easy to conduct
#     * do not capture interactions between inputs
#     * misleading for non-linear models
# 
# 
# * Global approaches
#     * e.g. Sobol analysis
#     * often need probabilistic data
#     * computationally demanding
#     * capture interactions between inputs
#     * handle non-linear and non-additive models
# 

# ### Sensitivity Analysis Techniques 
# 
# (adapted from Flechsig (2012), Saltelli(2008))
# 
# | Type | Morris | Variance | Factorial | Monte Carlo | Local SA |
# |:------|------|------|------|------|------|
# |Model independent? | yes | yes | yes | yes | yes|
# |Sample source | levels | distributions | levels | distributions | levels |
# |No. factors | $20-100^1$ | $<20^1$ | $>100^1$ | $<20$ | $<100$ |
# |Factor range | global | global | global | global | local |
# |Multi-factor variation | yes | yes | yes | yes | no |
# |Correlated factors? | no | no | yes | yes | no |
# |Cost (for k factors)? | $10(k+1)$ | $500(k+2)$ | $k \to 2k$ | $500+1$ | $2(k+1)$ |
# |Estimated CPU time$^2$ | 1 day | 11 days | 3 hours | ~2 days | 1 hour |
#   
# [1] using groups of factors would enable larger numbers of factors to be explored
# [2] assuming 5 minutes per simulation and 30 groups of factors
# 

# Sensitivity Analysis is strongly linked to uncertainty.
# 
# When considering the range over which a mode input should be explored, 
# you are actually considering the plausibility of this range.
# 
# Sensitivity Analysis makes you critically analyse your assumptions.
# 

# ### Global Sensitivity Analysis
# 
# Screening approaches, such as Fractional Factorial and Morris, **rank** inputs according to their influence upon the output.
# 
# Variance-based approaches, such as Saltelli and DMIM, **score** the sensitivity of each input with a numerical value, called the sensitivity index. Sensitivity indices come in several forms:
# * First-order indices: measures the contribution to the output variance by a single model input alone.
# * Second-order indices: measures the contribution to the output variance caused by the interaction of two model inputs.
# * Total-order index: measures the contribution to the output variance caused by a model input, including both its first-order effects (the input varying alone) and all higher-order interactions.
# 

# # Sensitivity Analysis in Practice: Vehicle to Grid
# 
# [Kempton (2005)](http://www.sciencedirect.com/science/article/pii/S0378775305000352) raise the prospect of using battery electric vehicles (BEVs) as mobile storage devices for electricity.  Other than pumped hydro, the electricity grid has virtually no storage devices.  Electricity storage could facilitate the integration of variable output renewable technologies such as wind turbines and solar photovoltaics.
# 
# The concept of V2G is that the owners of electric (or even hydrogen fuel cell) vehicles could be paid by the System Operator (National Grid in the UK) for the use of their cars as giant batteries.
# 
# There are three services that Vehicle-to-Grid (V2G) could provide:
# * Regulation - helps keep the grid operating at 60 Hz by exporting energy during times of extra demand or importing during times of extra supply
# * Spinning Reserve - an always-available (but not necessarily generating) backup in case a power plant drops offline
# * Peak Power - exports electricity to the grid during times of peak demand (e.g. 5-7pm)
# 

# We'll just look at **regulation** services:
# 
# 
# |Service | Revenue | Cost|
#  |---- | ---- | ----|
#  |**Regulation** | Energy Exported, Saving on Energy Imported | Cost of Energy, Degradation of Battery|
#  |Spinning Reserve | Capacity Available, Energy Exported | Cost of Energy, Degradation of Battery|
#  |Peak Power | Capacity Available, Energy Exported | Cost of Energy, Degradation of Battery|
# 

# ## Tesla Model S
# 
# ![Tesla](https://www.teslamotors.com/sites/default/files/images/model-s/gallery/exterior/hero-01.jpg?20151030)
# 
# Parameter | Min | Max
# ---| ---| ---
# Battery Size | 70 kWh | 85 kWh
# Charge Connectors | 230V 10 Amp (2.3 kW) | 400V 32 Amp (22 kW)
# Stated efficiency | ~ 5.5 km/kWh |

# ## Nissan Leaf
# 
# ![Nissan Leaf](https://www.nissan-cdn.net/content/dam/Nissan/nissan_europe/vehicles/leaf/product_code/product_version/overview/packshot_colorpicker_Leaf_QAB_small_zero-emission.png.ximg.s_12_m.smart.png)
# 
# Parameter | Min | Max
# ---| ---| ---
# Battery Size | 24 kWh | 30 kWh
# Charge Connectors | 2.3 kW | 6.6 kW
# Stated efficiency | ~5.5 km/kWh |

# ### One-at-time approach
# 
# First, we're going to try the **one-at-a-time** (OAT) approach.
# 
# Find the sliders in the example which are set up for the parameters of the Nissan Leaf.
# 

@interact(connector=widgets.FloatSlider(value=2.3, min=2.3, max=22, step=0.5), 
          battery_size=widgets.FloatSlider(value=24, min=10, max=100, step=5), 
          distance_driven=widgets.FloatSlider(value=0, min=0, max=100, step=5), 
          range_buffer=widgets.FloatSlider(value=0, min=0, max=100, step=10),
          dispatch_time=widgets.FloatSlider(value=1.4, min=0.5, max=24, step=0.5))
def plot_power(connector: float, battery_size: float, distance_driven: float, 
               range_buffer: float, dispatch_time: float) -> float :
    power = max_vehicle_power(connector,
                      battery_size,
                      distance_driven,
                      range_buffer,
                      dispatch_time
                      )
    return print("The maximum power is {} kW".format(round(power, 2)))


def monte_carlo_large(data):
    dispatch_time = 4
    y = max_vehicle_power(data[0], data[1], data[2], data[3], data[6], data[4], data[5])
    return y


# ### Scatter plots
# 
# Scatter plots can tell you quite a lot about the relationship between the model inputs and outputs.
# 
# Each of the scatter plots shows _all_ the model outputs on the y-axis, but re-ordered by the relationship to the input variable (on the x-axis).
# 
# ![Scatter Plot](./scatter.png)

number_sims = 1000

# Make some random data in the correct ranges
mc_connector = np.random.uniform(2.3, 22, number_sims)
mc_battery_size = np.random.uniform(50, 100, number_sims)
mc_distance_driven = np.random.uniform(0, 80, number_sims)
mc_range_buffer = np.random.uniform(0, 80, number_sims)
mc_driving_eff = np.random.uniform(2, 6, number_sims)
mc_inv_eff = np.random.uniform(0.87, 0.97, number_sims)
mc_dispatch_time = np.random.uniform(0.5, 24, number_sims)

data = np.array((mc_connector, 
                 mc_battery_size, 
                 mc_distance_driven, 
                 mc_range_buffer, 
                 mc_driving_eff, 
                 mc_inv_eff, 
                 mc_dispatch_time))

# Run the code
y = monte_carlo_large(data)


# Make some scatter plots to compare the results
plt.subplot(241)
plt.scatter(mc_connector, y)
plt.title("Connector size (kW)")
plt.ylabel("Max Power (kW)")
plt.subplot(242)
plt.scatter(mc_battery_size, y)
plt.title("Battery Size (kWh)")
# plt.ylabel("Max Power (kW)")
plt.subplot(243)
plt.scatter(mc_distance_driven, y)
plt.title("Distance Driven (km)")
# plt.ylabel("Max Power (kW)")
plt.subplot(244)
plt.scatter(mc_range_buffer, y)
plt.title("Range Buffer (km)")
# plt.ylabel("Max Power (kW)")
plt.subplot(245)
plt.scatter(mc_driving_eff, y)
plt.title("Driving Eff (kWh/km)")
plt.ylabel("Max Power (kW)")
plt.subplot(246)
plt.scatter(mc_inv_eff, y)
plt.title("Inverter Eff (%)")
# plt.ylabel("Max Power (kW)")
plt.subplot(247)
plt.scatter(mc_dispatch_time, y)
plt.title("Dispatch Time (hours)")
# plt.ylabel("Max Power (kW)")
plt.tight_layout()
# plt.savefig('scatter.png')


# You might be tempted to plot a histogram of the model outputs.  This shows how often a particular value occurs in the results, but given that we are only exploring the model variable ranges, don't read too much into this distribution.
# 

plt.hist(y)
plt.xlabel("Power (kW)")
plt.ylabel("Frequency")


# # Using SALib to run a Sensitivity Analysis
# 
# SALib is a **free** **open-source** **Python** library
# 
# If you use Python, you can install it by running the command
# 
# ```python
# pip install SALib
# ```
# 
# [Documentation](http://salib.readthedocs.org/) is available online and you can also view the code on [Github](http://salib.github.io/SALib/).
# 
# The library includes:
# * Sobol Sensitivity Analysis ([Sobol 2001](http://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli 2002](http://www.sciencedirect.com/science/article/pii/S0010465502002801), [Saltelli et al. 2010](http://www.sciencedirect.com/science/article/pii/S0010465509003087))
# * Method of Morris, including groups and optimal trajectories ([Morris 1991](http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804), [Campolongo et al. 2007](http://www.sciencedirect.com/science/article/pii/S1364815206002805))
# * Fourier Amplitude Sensitivity Test (FAST) ([Cukier et al. 1973](http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571), [Saltelli et al. 1999](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594))
# * Delta Moment-Independent Measure ([Borgonovo 2007](http://www.sciencedirect.com/science/article/pii/S0951832006000883), [Plischke et al. 2013](http://www.sciencedirect.com/science/article/pii/S0377221712008995))
# * Derivative-based Global Sensitivity Measure (DGSM) ([Sobol and Kucherenko 2009](http://www.sciencedirect.com/science/article/pii/S0378475409000354))
# * Fractional Factorial Sensitivity Analysis ([Saltelli et al. 2008](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html))
# 

# ### Import the package
# 

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp


# ### Define a problem file
# 
# In the code below, a problem file is used to define the variables we wish to explore
# 

morris_problem = {
    # There are six variables
    'num_vars': 7,
    # These are their names
    'names': ['conn', 'batt', 'dist', 'range', 
              'dri_eff', 'inv_eff', 'dispatch_time'],
    # Plausible ranges over which we'll move the variables
    'bounds': [[2.3, 22], # connection_power (kW)
               [50, 100], # battery size (kWh)
               [0, 80], # distance driven (km)
               [0, 80], # range buffer (km)
               [4,5.5], # driving efficiency (kWh/km)
               [0.87,0.97], # inverter efficienct (%)
               [0.5, 24] # dispatch time - hours of the day in which 
                         # the energy is dispatched
              ],
    # I don't want to group any of these variables together
    'groups': None
    }


# ### Generate a Sample
# 
# We then generate a sample using the `morris.sample()` procedure from the SALib package.
# 

number_of_trajectories = 1000
sample = ms.sample(morris_problem, number_of_trajectories, num_levels=4, grid_jump=2)


# ### Factor Prioritisation
# 
# We'll run a sensitivity analysis of the power module to see which is the most influential parameter.
# 
# The results parameters are called **mu**, **sigma** and **mu_star**.
# 
# * **Mu** is the mean effect caused by the input parameter being moved over its range.
# * **Sigma** is the standard deviation of the mean effect.
# * **Mu_star** is the mean absolute effect.
# 

# Run the sample through the monte carlo procedure of the power model
output = monte_carlo_large(sample.T)
# Store the results for plotting of the analysis
Si = ma.analyze(morris_problem, sample, output, print_to_console=False)
print("{:20s} {:>7s} {:>7s} {:>7s}".format("Name", "mu", "mu_star", "sigma"))
for name, s1, st, mean in zip(morris_problem['names'], 
                              Si['mu'], 
                              Si['mu_star'], 
                              Si['sigma']):
    print("{:20s} {:=7.2f} {:=7.2f} {:=7.2f}".format(name, s1, st, mean))


# We can plot the results
# 

fig, (ax1, ax2) = plt.subplots(1,2)
mp.horizontal_bar_plot(ax1, Si, {})
mp.covariance_plot(ax2, Si, {})


# ## A More Complicated Example
# 
# Lets look at a more complicated example.  This now integrates the previous power module into a simple cost-benefit analysis.
# 
# Trying to work out anything with all those sliders is pretty difficult.  We need to strip out the uneccesssary parameters and focus our efforts on the influential inputs.
# 

@interact(battery_size=widgets.FloatSlider(value=24, min=10, max=100, step=5), 
          battery_unit_cost=widgets.FloatSlider(value=350, min=100, max=400, step=50),
          connector_power=widgets.FloatSlider(value=2.3, min=2.3, max=22, step=0.5), 
          lifetime_cycles=widgets.FloatSlider(value=2000, min=1000, max=10000, step=1000),
          depth_of_discharge=widgets.FloatSlider(value=0.8, min=0.5, max=1.0, step=0.1),
          electricity_price=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          purchased_energy_cost=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          capacity_price=widgets.FloatSlider(value=0.007, min=0.001, max=0.01, step=0.001),
          round_trip_efficiency=widgets.FloatSlider(value=0.73, min=0.50, max=1.0, step=0.01),
          cost_of_v2g_equip=widgets.FloatSlider(value=2000, min=100, max=5000, step=100),
          discount_rate=widgets.FloatSlider(value=0.10, min=0.0, max=0.2, step=0.01),
          economic_lifetime=widgets.FloatSlider(value=10, min=3, max=25, step=1),
          ratio_dispatch_to_contract=widgets.FloatSlider(value=0.10, min=0.01, max=0.50, step=0.01),
          distance_driven=widgets.FloatSlider(value=0, min=0, max=100, step=5), 
          range_buffer=widgets.FloatSlider(value=0, min=0, max=100, step=10),
          hours_connected_per_day=widgets.FloatSlider(value=18, min=0.5, max=24, step=0.5))
def plot_profit(battery_size,
                battery_unit_cost,
                connector_power,
                lifetime_cycles,
                depth_of_discharge,
                electricity_price,
                purchased_energy_cost,
                capacity_price,
                round_trip_efficiency,
                cost_of_v2g_equip,
                discount_rate,
                economic_lifetime,
                distance_driven,
                range_buffer,
                ratio_dispatch_to_contract,
                hours_connected_per_day):
    profit, revenue, cost = compute_profit(battery_size,
                                           battery_unit_cost,
                                           connector_power,
                                           lifetime_cycles,
                                           depth_of_discharge,
                                           electricity_price,
                                           purchased_energy_cost,
                                           capacity_price,
                                           round_trip_efficiency,
                                           cost_of_v2g_equip,
                                           discount_rate,
                                           economic_lifetime,
                                           distance_driven,
                                           range_buffer,
                                           ratio_dispatch_to_contract,
                                           hours_connected_per_day
                                           )
    return print("Profit £{} = £{} - £{}".format(np.round(profit,2), np.round(revenue, 2), np.round(cost,2) ))


# ### Factor Fixing
# 
# We'll perform a **factor fixing** sensitivity analysis using a different method - that of Sobol.
# 

from SALib.sample.saltelli import sample as ss
from SALib.analyze.sobol import analyze as sa

problem = {
    # There are sixteen variables
    'num_vars': 16,
    # These are their names
    'names': ['battery_size',
              'battery_unit_cost',
              'connector_power',
              'lifetime_cycles',
              'depth_of_discharge',
              'electricity_price',
              'purchased_energy_cost',
              'capacity_price',
              'round_trip_efficiency',
              'cost_of_v2g_equip',
              'discount_rate',
              'economic_lifetime',
              'distance_driven',
              'range_buffer',
              'ratio_dispatch_to_contract',
              'hours_connected_per_day'],
    # These are their plausible ranges over which we'll move the variables
    'bounds': [       
                [10, 100],
                [100, 400],
                [2.3, 22],
                [1000, 10000],
                [0.5, 1.0],
                [0.01, 0.2], 
                [0.01, 0.2],
                [0.001, 0.01], 
                [0.65, 1.0],
                [100, 5000],
                [0.0, 0.2], 
                [3, 25],
                [0, 100], 
                [0, 100], 
                [0.01, 0.50],
                [0.5, 24],
              ],
    # I don't want to group any of these variables together
    'groups': None
    }


sample = ss(problem, 1000, calc_second_order=False)
profit, revenue, cost = compute_profit(sample[:, 0], sample[:, 1], sample[:, 2], 
                                       sample[:, 3], sample[:, 4], sample[:, 5], 
                                       sample[:, 6], sample[:, 7], sample[:, 8], 
                                       sample[:, 9], sample[:, 10], sample[:, 11], 
                                       sample[:, 12], sample[:, 13], sample[:, 14], 
                                       sample[:, 15])
SI = sa(problem, profit, calc_second_order=False, print_to_console=False)
print("{:28s} {:>5s} {:>5s} {:>12s}".format("Name", "1st", "Total", "Mean of Input"))
for name, s1, st, mean in zip(problem['names'], SI['S1'], SI['ST'], sample.mean(axis=0)):
    print("{:28s} {:=5.2f} {:=5.2f} ({:=12.2f})".format(name, s1, st, mean))


# The results should look something like this:
# 
# |Name                      |     1st |Total  |Mean of Input|
# | :--- | ---:| ---: | ---: |
# |battery_size               |  -0.01  |0.25 |       55.00|
# |battery_unit_cost          |  0.01   |0.03 |      250.10|
# |connector_power            |   0.01  |0.04 |       12.14|
# |lifetime_cycles            |   0.05  |0.09 |     5501.03|
# |depth_of_discharge         |   0.00  |0.03 |        0.75|
# |electricity_price          |   0.01  |0.06 |        0.10|
# |purchased_energy_cost      |   0.02  |0.13 |        0.10|
# |capacity_price             |  0.01   |0.03 |        0.01|
# |round_trip_efficiency      |   0.00  |0.01 |        0.82|
# |cost_of_v2g_equip          |   0.27  |0.34 |     2549.62|
# |discount_rate              |   0.05  |0.08 |        0.10|
# |economic_lifetime          |   0.13  |0.16 |       14.00|
# |distance_driven            |  -0.00  |0.03 |       49.96|
# |range_buffer               |  -0.01  |0.03 |       50.01|
# |ratio_dispatch_to_contract |   0.07  |0.27 |        0.26|
# |hours_connected_per_day    |  -0.01  |0.06 |       12.26|
# 

# The results show that the most important parameters are:
# * Capital cost of the V2G equipment
# * Ratio of dispatch to contract
# * Battery size
# * Economic lifetime
# * Purchased energy cost
# 
# Other comments:
# * __Lifetime cycles__ has a reasonably important first order effect so we can include that too.
# * __Battery size__ has much more important interaction effects than first-order effects
# * Same for __Purchased_energy_cost__
# 
# We can now fix the other parameters and revisit our slider model to perform some analysis.
# 

@interact(battery_size=widgets.FloatSlider(value=70, min=10, max=100, step=5), 
          purchased_energy_cost=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01),
          cost_of_v2g_equip=widgets.FloatSlider(value=2000, min=100, max=5000, step=100),
          economic_lifetime=widgets.FloatSlider(value=10, min=3, max=25, step=1),
          ratio_dispatch_to_contract=widgets.FloatSlider(value=0.10, min=0.01, max=0.50, step=0.01),
         lifetime_cycles=widgets.FloatSlider(value=2000, min=1000, max=10000, step=500))
def plot_profit(battery_size,
                purchased_energy_cost,
                cost_of_v2g_equip,
                economic_lifetime,
                ratio_dispatch_to_contract,
                lifetime_cycles):
    profit, revenue, cost = compute_profit(lifetime_cycles=lifetime_cycles,
                                           battery_size=battery_size,
                                           purchased_energy_cost=purchased_energy_cost,
                                           cost_of_v2g_equip=cost_of_v2g_equip,
                                           economic_lifetime=economic_lifetime,
                                           ratio_dispatch_to_contract=ratio_dispatch_to_contract
                                           )
    return print("Profit £{} = £{} - £{}".format(np.round(profit,2), 
                                                 np.round(revenue, 2), 
                                                 np.round(cost,2) ))


# # Summary
# 
# 
# Sensitivity analysis helps you:
# 
# * Think through your assumptions
# * Quantify uncertainty
# * Focus on the most influential uncertainties first
# 
# Learn [Python](https://www.python.org)
# 
# Similar packages to [SALib]() for other languages/programmes:
# 
# * [Matlab Toolbox **SAFE** for GSA](http://www.sciencedirect.com/science/article/pii/S1364815215001188)
# * [`sensitivity` package for R](https://cran.r-project.org/web/packages/sensitivity/index.html)
# 

