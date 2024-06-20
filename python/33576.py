# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#PCA" data-toc-modified-id="PCA-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>PCA</a></div><div class="lev1 toc-item"><a href="#Robust-Regressions" data-toc-modified-id="Robust-Regressions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Robust Regressions</a></div><div class="lev2 toc-item"><a href="#Figure-3-and-8" data-toc-modified-id="Figure-3-and-8-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Figure 3 and 8</a></div><div class="lev1 toc-item"><a href="#Heatmap" data-toc-modified-id="Heatmap-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Heatmap</a></div><div class="lev1 toc-item"><a href="#Figure-4A" data-toc-modified-id="Figure-4A-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Figure 4A</a></div><div class="lev1 toc-item"><a href="#Making-the-Genetic-Graph,-Figure-4b" data-toc-modified-id="Making-the-Genetic-Graph,-Figure-4b-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Making the Genetic Graph, Figure 4b</a></div>
# 

# In this notebook, we focus on developing the idea that whole-organism RNA-seq contains sufficient information to predict interactions between genes, and we will make some graphs, namely a PCA graph, that motivates the idea that epistasis can be measured genome-wide.
# 
# First, I will load a number of useful libraries. Notable libraries to load are `genpy`, a module that contains useful graphing functions tailored specifically for this project and developed by us; `morgan` a module that specifies what a Morgan object and a McClintock object are, and `gvars`, which contains globally defined variables that we used in this project.
# 

# important stuff:
import os
import pandas as pd
import numpy as np

import genpy
import gvars
import morgan as morgan

# stats
import sklearn.decomposition
import statsmodels.api as stm

# network graphics
import networkx as nx

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# mcmc
import pymc3 as pm

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


# Next, I will specify my q-value cutoff. A typical value for RNA-seq datasets is q=0.1 for statistical significance. I will also initialize a `genvar.genvars` object, which contains all of the global variables used for this project.
# 

q = 0.1
genvar = gvars.genvars()


# Next, I will prepare to initialize a Morgan project. Morgan objects have a large number of attributes. I wrote the Morgan library, but over the past year it has become deprecated and less useful. We will load it here, but it's a bit messy. I am in the process of cleaning it up.
# 
# So what does a Morgan object do?
# 
# Well, when you initialize a Morgan object, you must pass at least a set of 4 strings. These strings are, in order, the column where the isoform names (unique) reside, the name of the column that holds the regression coefficient from sleuth; the name of the column that holds the TPM values passed by Kallisto and the name of the column that holds the q-values.
# 
# Next, we can add what I call a genmap. A genmap is a file that maps read files to genotypes. A genmap file has three columns: 'project_name', 'genotype' and 'batch' in that exact order. For this project, the genotypes are coded. In other words, they are letters, 'a', 'b', 'd',... and not specific genotypes. The reason for this is that we wanted to make sure that, at least during the initial phase of the project, I could not unduly bias the results by searching the literature and what not.  Because the genotypes are coded, we need to specify which of the letters represent single mutants, and which letters represent double mutants. I also need to be able to figure out what the individual components of a double mutant are. Finally, we need to set the q-value threshold. If no q-value is specified, the threshold defaults to 0.1.
# 
# I will now initialize the object. I call it thomas.

# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
# Specify which genotypes are double mutants 
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.set_qval()


# Ok. Our Morgan object is up and running, but it doesn't have any data yet. So now, we need to specify where the object can look for the Sleuth outputs (`sleuth_loc`)and the Kallisto outputs (`kallisto_loc`). After we have specified these directories, we just let thomas loose in the directories. We will load the files into dictionaries:
# `{g1: df_beta1,..., gn: df_betan}`
# 

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'

thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')
# load all the beta dataframes:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)


# Great. Now we have loaded all the dataframes that contain TPM information (all 27 of them) and we have loaded the dataframes that contain the regression coefficients for each gene. However, the dataframes with the beta coefficients have plenty of NaNs. Applying thomas.filter_data() will drop them all. I also initialize a `genes` variable that basically holds fancy string names for each genotype in the order they exist within `thomas`.
# 

thomas.filter_data()
# labelling var:
genes = [genvar.fancy_mapping[x] for x in thomas.single_mutants]


# Finally, we will go ahead and make a tidy version of our data. Although we will be dealing with the Morgan object for some things, I have found it far easier to work with the tidied dataframe over time. In the tidy_data dataframe, each row is an observation.
# 

frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
tidy_data = pd.concat(frames)

# drop any genes that don't have a WormBase ID
tidy_data.dropna(subset=['ens_gene'], inplace=True)
# take a look at it:
tidy_data.head()


# # PCA
# 
# First, we will perform an exploratory procedure, PCA, to demonstrate that transcriptomic signatures from whole-organism RNA-seq have valuable information regarding genetic interactions.
# 
# First, I will identify the set of genes that is differentially expressed in at least one genotype. Then, for each genotype I will find what $\beta$  values have an associated q-value that is significant and which ones are not. Set all $\beta$ values that are not statistically significantly different from 0 equal to 0. Finally, we will standardize each genotype so that the collection $\beta$ values for each genotype has a mean of zero and standard deviation of 1. 
# 

max_overlap = tidy_data[tidy_data.qval < q].target_id.unique()
print('There are {0} isoforms that are DE in at least one genotype in this analysis'.format(len(max_overlap)))

grouped = tidy_data.groupby('code')
bvals = np.array([])
labels = []
for code, group in grouped:
    # find names:
    names = group.target_id.isin(max_overlap)
    # extract (b, q) for each gene
    bs = group[names].b.values
    qs = group[names].qval.values
    
    # find sig genes:
    inds = np.where(qs > q)
    # set non-sig b values to 0
    bs[inds] = 0
    #standardize bs
    bs = (bs - bs.mean())/(bs.std())
    
    # place in array
    if len(bvals) == 0:
        bvals = bs
    else:
        bvals = np.vstack((bvals, bs))
    # make a label array
    labels +=  [code]


# Next, we initialize the PCA object, specifying that we want to project the data onto two axes. Finally, we plot.
# 

# initialize the PCA object and fit to the b-values
sklearn_pca = sklearn.decomposition.PCA(n_components=2).fit(bvals)
coords = sklearn_pca.fit(bvals).transform(bvals)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', 'k']
shapes = ['D', 'D', 'v', '8', 'D', 'v', 'o']

# go through each pair of points and plot them:
for i, array in enumerate(coords):
    l = genvar.fancy_mapping[labels[i]]
    plt.plot(array[0], array[1], shapes[i], color=colors[i], label=l, ms=17)

# plot prettify:
plt.legend(loc=(1, 0.25), fontsize=16)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.savefig('../output/PCA_genotypes.svg', bbox_inches='tight')


# We can see that the diamonds all cluster together and triangles cluster together. The triangles are HIF-1$^-$ genotypes, whereas the diamonds (and purple octagon) are HIF-1$^+$ genotypes. The *fog-2* mutant is far away from genes in this pathway. The closeness of the *egl-9;hif-1(lf)* mutant to the *hif-1* double mutant suggests to me that epistasis can be measured genome-wide. 
# 

# # Robust Regressions
# 
# Next, I illustrate how we will perform robust regressions for this dataset. I will use [PyMC3](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html) to perform a robust regression. 
# 
# The recipe is as follows:
# 1. Choose two genotypes to compare, and find D.E. isoforms common to both. This is the set of isoforms to use from now on.
# 2. Rank the isoforms by $\beta$.
# 3. Perform a Student-T regression on the data.
# 4. Find the outliers to the regression, and perform a second regression on them. 
# 5. Plot the primary regression. If the secondary regression is of opposite value to the primary, plot that too.
# 
# In the cell below, I have selected two letters, `e` and `b`. I will find the isoforms that are differentially expressed in both genotypes. Then I will rank the $\beta$ coefficients for the remaining isoforms.
# 

# the genotypes to compare
letters = ['e', 'b']
sig = tidy_data[(tidy_data.code.isin(letters)) & (tidy_data.qval < q)]
grouped = sig.groupby('target_id')
genes = []

# find the intersection between the two.
for target, group in grouped:
    # make sure the group contains all desired genotypes
    all_in = (len(group.code.unique()) == 2)
    if all_in:
        genes += [target]

# extract a temporary dataframe with all the desired genes
temp = tidy_data[tidy_data.target_id.isin(genes)]

# split the dataframes and find the rank of each gene
ovx = genpy.find_rank(temp[temp.code == letters[0]])
ovy = genpy.find_rank(temp[temp.code == letters[1]])


# Having found the rank of the data, we will perform the MCMC by calling a function I wrote, called `robust regress`.
# 
# Then, we will find the outliers to that regression and we will perform a secondary regression.
# 

# Place data into dictionary:
data = dict(x=ovx.r, y=ovy.r)
x = np.linspace(ovx.r.min(), ovx.r.max())

# perform the simulation
trace_robust = genpy.robust_regress(data)

# find inliers, and outliers
outliers = genpy.find_inliers(ovx, ovy, trace_robust)

# run a second regression on the outliers
data2 = dict(x=ovx[ovx.target_id.isin(outliers)].r,
             y=ovy[ovy.target_id.isin(outliers)].r)

trace_robust2 = genpy.robust_regress(data2)
slope2 = trace_robust2.x.mean()


# ## Figure 3 and 8
# 
# Now we can plot the results. Figures 3 & 8 were generated using this code. 
# 

# draw a figure
plt.figure(figsize=(5, 5))

# plot mcmc results
label = 'posterior predictive regression lines'
pm.glm.plot_posterior_predictive(trace_robust, eval=x, 
                                 label=label, color='#357EC7')

# only plot secondary slope if it's of opposite sign to first
slope = trace_robust.x.mean()
if slope2*slope < 0:
    pm.glm.plot_posterior_predictive(trace_robust2, eval=x, 
                                     label=label, color='#FFA500')

# plot the data 
ind = ovx.target_id.isin(outliers)
x = ovx[~ind].r
y = ovy[~ind].r
plt.plot(x, y, 'go', ms = 5, alpha=0.4, label='inliers')

x = ovx[ind].r
y = ovy[ind].r
plt.plot(x, y, 'rs', ms = 6, label='outliers')

# prettify plot
plt.xlim(0, len(ovx))
plt.ylim(0, len(ovy))
plt.yticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xlabel(genvar.fancy_mapping[letters[0]] +
           r'(lf) isoforms ranked by $\beta$')
plt.ylabel(genvar.fancy_mapping[letters[1]] +
           r'(lf) isoforms ranked by $\beta$')

comp = letters[0] + letters[1]
plt.savefig('../output/multiplemodes-{0}.svg'.format(comp), bbox_inches='tight')


# the genotypes to compare
letters = ['e', 'g']
sig = tidy_data[(tidy_data.code.isin(letters)) & (tidy_data.qval < q)]
grouped = sig.groupby('target_id')
genes = []

# find the intersection between the two.
for target, group in grouped:
    # make sure the group contains all desired genotypes
    all_in = (len(group.code.unique()) == 2)
    if all_in:
        genes += [target]

# extract a temporary dataframe with all the desired genes
temp = tidy_data[tidy_data.target_id.isin(genes)]

# split the dataframes and find the rank of each gene
ovx = genpy.find_rank(temp[temp.code == letters[0]])
ovy = genpy.find_rank(temp[temp.code == letters[1]])
plt.plot(ovx.r, ovy.r, 'go', ms = 5, alpha=0.4,)

plt.xlim(0, len(ovx))
plt.ylim(0, len(ovy))
plt.yticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xlabel(genvar.fancy_mapping[letters[0]] +
           r'(lf) isoforms ranked by $\beta$')
plt.ylabel(genvar.fancy_mapping[letters[1]] +
           r'(lf) isoforms ranked by $\beta$')


# Three points are immediately noticeable about this plot. First, there's a LOT of points on it. This makes me feel like there is bound to be an interaction of some form between *fog-2* and the hypoxia pathway. Second, the cross is quite clearly visible in this diagram, and it seems like both positive and negative interactions are ocurring. This suggests that whatever the crosstalk between *fog-2* and the hypoxia pathway is, it involves more than just a single unidirectional interaction. Third, the spread in these correlations is, by eye, considerably larger than the spread we showed above, and actually it is considerably larger than the spread observed when correlating any two mutants within the hypoxia pathway. 
# 
# # Heatmap
# 
# We can automate and repeat the process above for every pairwise combination in this dataset. I automated this process and stored it the `morgan` library, which defines an object `mcclintock` that does exactly this. 
# 

barbara = morgan.mcclintock('bayesian', thomas, progress=False)


# # Figure 4A
# 
# I will plot the results in a heatmap for easy visualization. This is Figure 4A in the paper. 
# 

mat = barbara.robust_slope.as_matrix(columns=thomas.single_mutants)
labels = [genvar.fancy_mapping[x] for x in barbara.robust_slope.corr_with.values]

genpy.tri_plot(mat, labels)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig('../output/bayes_primary_single_mutants.svg',
            bbox_inches='tight')


# # Making the Genetic Graph, Figure 4b
# 
# Another (equivalent) way of displaying the data is to build a network, where the edge thickness or the edge color provide information about the strength of the correlation. We do that below.
# 

# make the graph:
G, width, weights, elarge = genpy.make_genetic_graph(barbara.robust_slope, w=3)

# paint the canvas:
with sns.axes_style('white'):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)  # positions for all nodes
    # draw the nodes:
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color='g', alpha=.5)
    # draw the edges:
    edges = nx.draw_networkx_edges(G, pos, edgelist=elarge,
                                   width=width, edge_color=weights,
                                   edge_cmap=plt.cm.RdBu,
                                   edge_vmin=-.3, 
                                   edge_vmax=.3)
    # add the labels:
    nx.draw_networkx_labels(G, pos, font_size=16,
                            font_family='sans-serif')

    # add a colorbar:
    fig.colorbar(edges)
    sns.despine()
    sns.despine(left=True, bottom=True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("../output/weighted_graph.svg") # save as png





# # Table of Contents
#  <p>
# 

# important stuff:
import os
import pandas as pd
import numpy as np
import scipy
import statsmodels.tools.numdiff as smnd


# TEA and morgan
import tissue_enrichment_analysis as tea
import morgan as morgan
import epistasis as epi
import gvars

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

from scipy.stats import gaussian_kde

# Magic function to make matplotlib inline;
# other style specs must come AFTER
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


# simulate data:
xdata = np.linspace(-10, 10, 40)
ydata = np.linspace(5, -5, 40) + np.random.normal(0, 0.5, 40)


plt.plot(xdata, ydata, '.')


import emcee

# Define our posterior using Python functions
# for clarity, I've separated-out the prior and likelihood
# but this is not necessary. Note that emcee requires log-posterior

def log_prior(theta):
    beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood(theta, x, y):
    beta, sigma = theta
    y_model = beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

# Here we'll set up the computation. emcee combines multiple "walkers",
# each of which is its own MCMC chain. The number of trace results will
# be nwalkers * nsteps

ndim = 2  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

# Here's the function call where all the work happens:
# we'll time it using IPython's %time magic

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata, ydata])
get_ipython().magic('time sampler.run_mcmc(starting_guesses, nsteps)')
print("done")


def compute_sigma_level(trace1, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins = np.histogram2d(trace1, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])

    return xbins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
#     xbins, ybins, sigma = compute_sigma_level(trace[0])
    sns.distplot(trace[0], ax=ax)
#     ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
#     if scatter:
#         ax.plot(trace[0], trace[1], ',k', alpha=0.1)
#     ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    
    
def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    beta = trace[0]
    xfit = np.linspace(-20, 20, 10)
    yfit = beta[:, None]* xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)


# sampler.chain is of shape (nwalkers, nsteps, ndim)
# we'll throw-out the burn-in points and reshape:
emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
plot_MCMC_results(xdata, ydata, emcee_trace)


emcee_trace


def neg_log_posterior(theta, x, y):
    return -log_posterior(theta, x, y)


scipy.optimize.minimize(neg_log_posterior, [-1, 1], args=(xdata, ydata), method='powell')


def log_prior(theta):
    beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood(theta, x, y):
    beta, sigma = theta
    y_model = beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

def neg_log_prob_free(theta, x, y):
    return -log_posterior(theta, x, y)


res = scipy.optimize.minimize(neg_log_prob_free, [0, 1], args=(xdata, ydata), method='Powell')


res





# # Table of Contents
#  <p>
# 

import os
import pandas as pd


# params
directory = '../sleuth/'
batch = False
# sequences:
analysis = next(os.walk(directory))[1]


def sleuth_analysis(directory, genovar, batch=False):
    """
    A function to write the differential_expression_analyzer batch command.
    """
    if not batch:
        heart = 'Rscript diff_exp_analyzer.R -d {0} --genovar {1}'.format(directory, genovar)
    else:
        heart = 'Rscript diff_exp_analyzer.R -d {0} --genovar {1} --batch'.format(directory, genovar)
    return heart

def walk_sleuth_directories(directory, batch=False):
    """
    Given a directory, walk through it,
    find all the rna-seq repository folders
    and generate kallisto commands
    """
    sleuth = ''
    #directory contains all the projects, walk through it:
    current, dirs, files = next(os.walk(directory))
    for d in dirs:
        # genovar always begins with a z:
        genovar = 'z' + d[-1:]
        message = '# Sleuth analysis command for {0}\n'.format(d)
        command = sleuth_analysis(d, genovar, batch) +'\n'
        sleuth += message
        sleuth += command
    return sleuth


with open(directory + 'sleuth_commands.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# Bash commands for diff. expression analysis using Sleuth.\n')
    sleuth_command = walk_sleuth_directories(directory, batch)
    f.write(sleuth_command)
#     print(sleuth_command)








# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Isoforms-Identified-in-all-Genotypes" data-toc-modified-id="Isoforms-Identified-in-all-Genotypes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Isoforms Identified in all Genotypes</a></div><div class="lev1 toc-item"><a href="#Number-of-Differentially-Expressed-Genes-(DEG)-in-each-genotype" data-toc-modified-id="Number-of-Differentially-Expressed-Genes-(DEG)-in-each-genotype-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Number of Differentially Expressed Genes (DEG) in each genotype</a></div><div class="lev1 toc-item"><a href="#Perturbation-Distributions" data-toc-modified-id="Perturbation-Distributions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Perturbation Distributions</a></div><div class="lev1 toc-item"><a href="#Transcriptomic-overlap-between-gene-pairs" data-toc-modified-id="Transcriptomic-overlap-between-gene-pairs-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Transcriptomic overlap between gene pairs</a></div>
# 

# In this notebook, I will go over the basic results from the RNA-seq in what is essentially a top-level view of the results. Nothing specific, mainly numbers, some histograms and that's it. 
# 
# First, I will load a number of useful libraries. Notable libraries to load are `genpy`, a module that contains useful graphing functions tailored specifically for this project and developed by us; `morgan` a module that specifies what a Morgan object and a McClintock object are, and `gvars`, which contains globally defined variables that we used in this project.
# 

# important stuff:
import os
import pandas as pd
import numpy as np

import morgan as morgan
import genpy
import gvars

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


# Next, I will specify my q-value cutoff. A typical value for RNA-seq datasets is q=0.1 for statistical significance. I will also initialize a `genvar.genvars` object, which contains all of the global variables used for this project.
# 

q = 0.1
# this loads all the labels we need
genvar = gvars.genvars()


# Now, I will prepare to initialize a Morgan project. Morgan objects have a large number of attributes. I wrote the Morgan library, but over the past year it has become deprecated and less useful. We will load it here, but it's a bit messy. I am in the process of cleaning it up.
# 
# So what does a Morgan object do?
# 
# Well, when you initialize a Morgan object, you must pass at least a set of 4 strings. These strings are, in order, the column where the isoform names (unique) reside, the name of the column that holds the regression coefficient from sleuth; the name of the column that holds the TPM values passed by Kallisto and the name of the column that holds the q-values.
# 
# Next, we can add what I call a genmap. A genmap is a file that maps read files to genotypes. A genmap file has three columns: 'project_name', 'genotype' and 'batch' in that exact order. For this project, the genotypes are coded. In other words, they are letters, 'a', 'b', 'd',... and not specific genotypes. The reason for this is that we wanted to make sure that, at least during the initial phase of the project, I could not unduly bias the results by searching the literature and what not.  Because the genotypes are coded, we need to specify which of the letters represent single mutants, and which letters represent double mutants. I also need to be able to figure out what the individual components of a double mutant are. Finally, we need to set the q-value threshold. If no q-value is specified, the threshold defaults to 0.1.
# 
# I will now initialize the object. I call it thomas. Then I will load in all the variables we will use; I will load in the genmap, and at last I will load in the datasets that contain the TPM and the Sleuth $\beta$ coefficients. After everything has been loaded, I will call `thomas.filter_data`, which drops all the rows that have a $\beta$ coefficient equal to NaN

# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']

# Specify which letters are double mutants and their genotype
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
# input the genmap file:
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
# add the names of the single mutants
thomas.add_single_mutant(single_mutants)
# add the names of the double mutants
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
# set the q-value threshold for significance to its default value, 0.1
thomas.set_qval()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'
thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')
# load all the beta values for each genotype:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)
        
thomas.filter_data()


# Finally, we will place all the data in a tidy dataframe, where each row is an observation.
# 

frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.mapping[key]
    frames += [df]
    df['sorter'] = genvar.sort_muts[key]
tidy = pd.concat(frames)

# I will make a new column, called absb where I place the absolute val of b
tidy['absb'] = tidy.b.abs()

# sort_values according to their position in the sorter column
# (makes sure single mutants are clustered and doubles are clustered)
tidy.sort_values('sorter', inplace=True)
tidy.dropna(subset=['ens_gene'], inplace=True)


# # Isoforms Identified in all Genotypes
# 

total_genes_id = tidy.target_id.unique().shape[0]
print("Total isoforms identified in all genotypes: {0}".format(total_genes_id))


# We identified 18685 isoforms using 7 million reads. Not bad considering there are ~25,000 isoforms in C. elegans. Each gene has just slightly over 1 isoform on average, so what this means is that we sampled almost 80% of the genome.
# 
# # Number of Differentially Expressed Genes (DEG) in each genotype
# 
# Next, let's figure out how many *genes* were differentially expressed in each mutant relative to the wild-type control.
# 

print('Genotype: DEG')
for x in tidy.genotype.unique():
    # select the DE isoforms in the current genotype:
    sel = (tidy.qval < q) & (tidy.genotype == x)
    # extract the number of unique genes:
    s = tidy[sel].ens_gene.unique().shape[0]
    print(
"""{0}: {1}""".format(x, s))


# From the above exploration, we can already conclude that:
#  * *hif-1(lf)* has a transcriptomic phenotype
#  * *hif-1;egl-9(lf)* has a transcriptomic phenotype
#  * The *egl-9* phenotype is stronger than the *vhl-1* or the *hif-1* phenotypes.
# 
# We should be careful is saying whether *rhy-1*, *egl-9* and *egl-9;vhl-1(lf)* are different from each other, and the same goes for *hif-1(lf)*, *vhl-1(lf)* and *egl-9;hif-1(lf)* because we set our FDR threshold at 10%. Notice that *egl-9(lf)* and *rhy-1(lf)* are barely 300 genes separated from each other. A bit of wiggle from both, and they might be identical. 
# 
# # Perturbation Distributions
# 
# Another thing we could do is plot the distribution of effect sizes for each genotype for genes that showed statistically significant perturbations. Because of the large number of points, we should expect a spread large enough that every mutant will probably overlap. This next plot is not very informative, but always useful to look at. 
# 
# In the plot below, each line within the box represents a quartile (25, 50 and 75%) and the whiskers represent the rest of the distribution, *sans* outliers. The outliers are the black dots outside the whiskers.
# 

sns.boxplot(x='genotype', y='absb', data=tidy[tidy.qval < q])
plt.yscale('log')
plt.xticks(rotation=30)


# We can see that *egl-9* genotypes have the largest median effect size of the hypoxia pathway mutants, but all genotypes have overlapping distributions. How much can we gain from this? Probably not a whole lot... At any rate, it goes to show that *egl-9(lf)* does have the most severe phenotype transcriptomically, much like its macroscopic counterparts.
# 
# # Transcriptomic overlap between gene pairs
# 
# In order to be able to assess whether two genes are interacting, we must first determine that the mutants we are studying act upon the same phenotype. What defines a phenotype in transcriptomic space? We use an operational definition -- two genotypes share the same phenotype if they regulate more than a pre-specified(and admittedly subjective) number of genes in common between the two of them, angostic of direction. Let's figure out to what extent the genes we have studied share the same phenotype.
# 
# The code below is a hack, but it does the job.
# 

sig = (tidy.qval < q)
print('pair, shared GENES, percent shared (isoforms)')
for i, g1 in enumerate(tidy.genotype.unique()):
    genes1 = tidy[sig & (tidy.genotype == g1)]
    for j, g2 in enumerate(tidy.genotype.unique()[i+1:]):
        genes2 = tidy[sig & (tidy.genotype == g2)]
        
        # find the overlap between the two:
        n = genes2[genes2.ens_gene.isin(genes1.ens_gene)].shape[0]
        OR = ((tidy.genotype == g1) | (tidy.genotype == g2)) 
        
        n_iso = genes2[genes2.target_id.isin(genes1.target_id)].shape[0]
        ntot = tidy[sig & OR].target_id.shape[0]
        print(
            "{0}-{1}, {2}, {3:.2g}%".format(g1, g2, n, 100*n_iso/ntot)
             )


# Well, we can see that the number of genes that is shared between mutants of the same pathway ranges from ~100 genes all the way to ~1,300. However, the hypoxia mutants share between ~140 and ~700 genes in common with another mutant, the *fog-2(lf)* mutant that has never been reported to act in the hypoxia pathway. What are we to make of this? My own conclusion is that *fog-2* probably does interact (probably suppresses, actually) with the hypoxia pathway and they both act on some of the same genes. Whether they are epistatic to each other, or whether the interaction is simply additive, we can't tell but it does raise questions. 
# I won't show it here, but among the genes that *fog-2* regulates is *cysl-1*, which is a powerful regulator of *egl-9*.
# 

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Finding-HIF-1-direct-target-candidates" data-toc-modified-id="Finding-HIF-1-direct-target-candidates-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Finding HIF-1 direct target candidates</a></div><div class="lev1 toc-item"><a href="#vhl-1-dependent,-hif-1-independent,-genes" data-toc-modified-id="vhl-1-dependent,-hif-1-independent,-genes-2"><span class="toc-item-num">2&nbsp;&nbsp;</span><em>vhl-1</em> dependent, <em>hif-1</em>-independent, genes</a></div><div class="lev1 toc-item"><a href="#egl-9-dependent,-non-hif-1-dependent,-genes" data-toc-modified-id="egl-9-dependent,-non-hif-1-dependent,-genes-3"><span class="toc-item-num">3&nbsp;&nbsp;</span><em>egl-9</em> dependent, non-<em>hif-1</em>-dependent, genes</a></div>
# 

# In this notebook, I will identify gene targets that are specifically regulated by each *egl-9*, *vhl-1*, and *hif-1*. I define a specific regulatory node to mean the node that is the nearest regulatory node to these targets out of the subset of genes we have mutants for. 
# 
# As usual, we first load up all the libraries
# 

# important stuff:
import os
import pandas as pd
import numpy as np

# morgan
import morgan as morgan
import tissue_enrichment_analysis as tea
import epistasis as epi
import genpy
import gvars

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


q = 0.1
genvar = gvars.genvars()
tissue_df = tea.fetch_dictionary()
phenotype_df = pd.read_csv('../input/phenotype_ontology.csv')
go_df = pd.read_csv('../input/go_dictionary.csv')


# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.set_qval()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'

thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')
# load all the beta values for each genotype:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)
thomas.filter_data()


frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
tidy_data = pd.concat(frames)
tidy_data.dropna(subset=['ens_gene'], inplace=True)
tidy_data = tidy_data[tidy_data.code != 'g']


# # Finding HIF-1 direct target candidates
# 
# We are interested in identifying gene targets of HIF-1. In order to do this, I will decouple my data into two parts:
# a `positive` dataframe, which contains all genes with $\beta$ values greater than 0
# a `negative` dataframe, which contains all genes with $\beta$ values less than 0
# 
# I will also define a function called `collate`. This function takes in a list or a numpy array and returns a boolean indicator of what genes are in a specified dataframe. It's a lot shorter to define this function than it is to write the one-liner over and over again.
# 

hif_genes = pd.read_csv('../output/hypoxia_response.csv')

n = len(hif_genes[hif_genes.b > 0].ens_gene.unique())
message = 'There are {0} unique genes that' +          ' are candidates for HIF-1 direct binding'
print(message.format(n))


# Alright! Now we're talking. As a safety check, let's make a qPCR like plot to visualize our genes, and let's make sure they have the behavior we want:
# 

ids = hif_genes[hif_genes.b > 0].target_id
hypoxia_direct_targets = tidy_data[tidy_data.target_id.isin(ids)]


names = hypoxia_direct_targets.sort_values('qval').target_id.unique()[0:10]

name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1

plot_df = tidy_data[tidy_data.target_id.isin(names)].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)


_ = tea.enrichment_analysis(hypoxia_direct_targets.ens_gene.unique(),
                            phenotype_df, show=True)


_ = tea.enrichment_analysis(hypoxia_direct_targets.ens_gene.unique(),
                            go_df, show=False)
tea.plot_enrichment_results(_, analysis='go')


# # *vhl-1* dependent, *hif-1*-independent, genes
# 
# Finally, we can gate our settings to observe only *vhl-1*-dependent genes, by selecting only those genes that were present in the *vhl-1* and *egl-9;vhl-1* genotypes.
# 

# find the genes that overlap between vhl1 and egl-9vhl-1 and change in same directiom
vhl_pos = epi.find_overlap(['d', 'a'], positive)
vhl_neg = epi.find_overlap(['d', 'a'], negative)
vhl = list(set(vhl_pos + vhl_neg))

# find genes that change in the same direction in vhl(-) and vhl(+ datasets)
same_vhl = []
for genotype in ['b', 'e', 'f', 'c']:
    same_vhl += epi.find_overlap(['d', 'a', genotype], positive)
    same_vhl += epi.find_overlap(['d', 'a', genotype], negative)

# put it all together:
ind = (collate(vhl)) & (~collate(same_vhl))
vhl_regulated = tidy_data[ind & (tidy_data.code == 'd')]

n = len(vhl_regulated.ens_gene.unique())
message = 'There are {0} genes that appear to be ' +          'regulated in a hif-1-independent, vhl-1-dependent manner.'
print(message.format(n))


# begin plotting
names = vhl_regulated.sort_values('qval').target_id.unique()[0:10]
name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1

plot_df = tidy_data[tidy_data.target_id.isin(names)].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)

# save to file
cols = ['ext_gene', 'ens_gene', 'target_id', 'b', 'qval']
vhl_regulated[cols].to_csv('../output/vhl_1_regulated_genes.csv')


# No enrichment was observed for these genes.
# 

# # *egl-9* dependent, non-*hif-1*-dependent, genes
# 
# I am also interested in knowing whether I can actually identify targets of *egl-9*. What are conditions on *egl-9* targets? Let's go ahead and see what we should do if HIF-1-OH is active.
# 
# Genotype | HIF-1| HIF-1OH| EGL-9
# ---------|------|--------|------
# *hif-1(lf)*| HIF-1$\downarrow$| HIF-1OH$\downarrow$| EGL-9$\uparrow$ (?)
# *egl-9(lf)*| HIF-1$\uparrow$| HIF-1OH$\downarrow$| EGL-9$\downarrow$
# *rhy-1(lf)*| HIF-1$\uparrow$| HIF-1OH$\downarrow$| EGL-9$\downarrow$
# *vhl-1(lf)*| HIF-1$\uparrow$| HIF-1OH$\uparrow$| EGL-9$\uparrow$
# *egl-9(lf);hif-1*| HIF-1$\downarrow$| HIF-1OH$\downarrow$| EGL-9$\downarrow$
# *egl-9(lf);vhl-1*| HIF-1$\uparrow$| HIF-1OH$\downarrow$| EGL-9$\downarrow$
# 
# OK. Sorry for the confusing table. The rule of the game is that we can combine rows as follows: For any entry, if the two arrows point in the same direction, that entry remains. If the two arrows point in different directions, they cancel. You can subtract rows, which amounts to flipping the sign.
# 

# If I add *egl-9*, *rhy-1* and *egl-9;hif-1*, I get:
# 
# Result | HIF-1| HIF-1OH| EGL-9| RHY-1
# ---------|------|--------|------|
# Sum| -| HIF-1OH$\downarrow$| EGL-9$\downarrow$| NaN
# 
# Ack. We are almost there. But we are not quite there yet. We need to get rid of that pesky HIF-1OH. If I subtract the *hif-1* row, we can get rid of it.
# 
# This suggests a solution. Find the genes that are commonly regulated (in the same directions!) between *egl-9*, *rhy-1* and *egl-9;hif-1*. Next, find the genes that are regulated by *hif-1*. If the genes appear in the same list in the same direction in both lists, remove those genes. What is left should be the *egl-9* specific transcriptome
# 

# genes that change in the same direction in egl, rhy and eglhif
egl_pos = epi.find_overlap(['e', 'b', 'f'], positive)
egl_neg = epi.find_overlap(['e', 'b', 'f'], negative)
egl = list(set(egl_pos + egl_neg))

cup = tidy_data[(tidy_data.code == 'c') & (tidy_data.b > 0)]
bdown = tidy_data[(tidy_data.code.isin(['e', 'b', 'f'])) & (tidy_data.b < 0)]
cdown = tidy_data[(tidy_data.code == 'c') & (tidy_data.b > 0)]
bup = tidy_data[tidy_data.code.isin(['e', 'b', 'f']) & (tidy_data.b < 0)]

antihif_1 = pd.concat([cup, bdown])
antihif_2 = pd.concat([cdown, bup])

antihif = []
for genotype in ['b', 'e', 'f']:
    temp = epi.find_overlap([genotype, 'c'], antihif_1)
    antihif += temp
    temp = epi.find_overlap([genotype, 'c'], antihif_2)
    antihif += temp

ind = collate(egl) & (collate(antihif)) & (~collate(same_vhl))

egl_regulated = tidy_data[ind & (tidy_data.code == 'b')]


n = egl_regulated.ens_gene.unique().shape[0]
print('There appear to be {0} egl-specific genes'.format(n))


egl_regulated[['ext_gene', 'b', 'qval']]


names = egl_regulated.sort_values('qval').target_id.unique()
name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1


plot_df = tidy_data[tidy_data.target_id.isin(egl_regulated.target_id.unique())].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  


plot_df = tidy_data[tidy_data.target_id.isin(egl_regulated.target_id.unique())].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)

plt.savefig('../output/egl9_downstream.pdf', bbox_inches='tight')





# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Defining-the-Hypoxia-Response" data-toc-modified-id="Defining-the-Hypoxia-Response-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Defining the Hypoxia Response</a></div><div class="lev1 toc-item"><a href="#Enrichment-Analysis-of-the-Global-HIF-1-response" data-toc-modified-id="Enrichment-Analysis-of-the-Global-HIF-1-response-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Enrichment Analysis of the Global HIF-1 response</a></div><div class="lev1 toc-item"><a href="#Enrichment-Analyses-of-the-egl-9-transcriptome" data-toc-modified-id="Enrichment-Analyses-of-the-egl-9-transcriptome-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Enrichment Analyses of the <em>egl-9</em> transcriptome</a></div><div class="lev1 toc-item"><a href="#Enrichment-Analysis-of-the-vhl-1-transcriptome" data-toc-modified-id="Enrichment-Analysis-of-the-vhl-1-transcriptome-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Enrichment Analysis of the <em>vhl-1</em> transcriptome</a></div><div class="lev1 toc-item"><a href="#Enrichment-Analysis-of-the-hif-1--transcriptome" data-toc-modified-id="Enrichment-Analysis-of-the-hif-1--transcriptome-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Enrichment Analysis of the <em>hif-1</em>  transcriptome</a></div>
# 

# In this notebook, we will isolate the hypoxia response (defined as the set of genes commonly regulated by *egl-9*, *rhy-1* and *vhl-1*), and we will perform enrichment analysis on the hypoxia response. We will also perform enrichment analyses on each mutant transcriptomes, to try to understand how different each transcriptome actually is.
# 

# important stuff:
import os
import pandas as pd

# TEA and morgan
import tissue_enrichment_analysis as tea
import morgan as morgan
import gvars
import epistasis as epi

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
get_ipython().magic("config InlineBackend.figure_formats = {'png','retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


q = 0.1
# this loads all the labels we need
genvar = gvars.genvars()

tissue_df = tea.fetch_dictionary()
phenotype_df = pd.read_csv('../input/phenotype_ontology.csv')
go_df = pd.read_csv('../input/go_dictionary.csv')
respiratory_complexes = pd.read_excel('../input/respiratory_complexes.xlsx')


# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.set_qval()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'

thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')
# load beta dataframes:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)
thomas.filter_data()


frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.mapping[key]
    df['code'] = key
    frames += [df]
    df['sorter'] = genvar.sort_muts[key]
tidy = pd.concat(frames)
tidy.sort_values('sorter', inplace=True)
tidy.dropna(subset=['ens_gene'], inplace=True)


# # Defining the Hypoxia Response
# 
# We defined the hypoxia response as the set of genes that change in the same direction in mutants of *egl-9*, *vhl-1* and *rhy-1*. We use `epi.find_overlap` from the `epistasis` module that we wrote. This function finds genes that are statistically significantly altered in common between $n$ genotypes. However, `find_overlap` is direction agnostic. Therefore, I will call it twice: Once with a sliced list of genes that have positive $\beta$s and once with negative $\beta$s. 
# 

hyp_response_pos = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b > 0])
hyp_response_neg = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b < 0])


either_or = ((tidy.b < 0) & (tidy.qval < q)) | (tidy.qval > q)
hyp_response_pos = tidy[(tidy.target_id.isin(hyp_response_pos)) & (tidy.code == 'f') & either_or].target_id.values.tolist()

either_or = ((tidy.b > 0) & (tidy.qval < q)) | (tidy.qval > q)
hyp_response_neg = tidy[(tidy.target_id.isin(hyp_response_neg)) & (tidy.code == 'f') & either_or].target_id.values.tolist()

hyp_response = list(set(hyp_response_neg + hyp_response_pos))


hyp = tidy[(tidy.target_id.isin(hyp_response)) &
           (tidy.code == 'b')
          ].copy().sort_values('qval')

def annotate(x):
    if x > 0:
        return 'candidate for direct regulation'
    else:
        return 'candidate for indirect regulation'
    
hyp['regulation'] = hyp.b.apply(annotate)


# save to file
cols = ['target_id', 'ens_gene', 'ext_gene', 'b', 'qval', 'regulation']
hyp[cols].to_csv('../output/hypoxia_response.csv', index=False)

# print the number of genes (not isoforms) in the hypoxia response
hyp_response = tidy[tidy.target_id.isin(hyp_response)].ens_gene.unique()
print('There are {0} genes in the predicted hypoxia response'.format(len(hyp_response)))


tea.enrichment_analysis(hyp.ens_gene.unique(), tissue_df=go_df, show=False)


# # Enrichment Analysis of the Global HIF-1 response
# 
# Now that we have found the hypoxia response, we can perform tissue, phenotype and gene ontology enrichment analysis on this gene battery.
# Note that we don't show all possibilities. When a particular analysis is not present, it is because the enrichment results were empty.
# 

teaH = tea.enrichment_analysis(hyp_response, tissue_df, show=False)
geaH = tea.enrichment_analysis(hyp_response, go_df, show=False)


tea.plot_enrichment_results(geaH, analysis='go')
plt.savefig('../output/hypoxia_response_gea.svg', bbox_inches='tight')


teaH


# These enrichment analyses show, as expected, that the HIF-dependent hypoxia response causes important changes in metabolism and physiology relating to oxygen availability. Tissue enrichment shows enrichment in the cephalic sheath cells (strange) but also in the uterine seam cells. Again, no idea what it means, although reports by [Chang and Bargmann](http://www.pnas.org/content/105/20/7321.full) suggest that the somatic gonad plays a special role in modulating behavior during hypoxia.
# 
# # Enrichment Analyses of the *egl-9* transcriptome
# 

egl = tidy[(tidy.qval < q) & (tidy.code == 'e')]
teaEgl = tea.enrichment_analysis(egl[egl.qval < q].ens_gene.unique(), tissue_df, show=False)
peaEgl = tea.enrichment_analysis(egl[egl.qval < q].ens_gene.unique(), phenotype_df, show=False)
geaEgl = tea.enrichment_analysis(egl[egl.qval < q].ens_gene.unique(), go_df, show=False)


tea.plot_enrichment_results(teaEgl)


tea.plot_enrichment_results(peaEgl, analysis='phenotype')


tea.plot_enrichment_results(geaEgl, analysis='go')


# # Enrichment Analysis of the *vhl-1* transcriptome
# 

vhl = tidy[(tidy.qval < q) & (tidy.code == 'd')]
teaVhl = tea.enrichment_analysis(vhl[vhl.qval < 0.1].ens_gene.unique(), tissue_df, show=False)
peaVhl = tea.enrichment_analysis(vhl[vhl.qval < 0.1].ens_gene.unique(), phenotype_df, show=False)
geaVhl = tea.enrichment_analysis(vhl[vhl.qval < 0.1].ens_gene.unique(), go_df, show=False)


tea.plot_enrichment_results(geaVhl, analysis='go')


teaVhl


# # Enrichment Analysis of the *hif-1*  transcriptome
# 

hif = tidy[(tidy.qval < q) & (tidy.code == 'c')]
teahif = tea.enrichment_analysis(hif[hif.qval < 0.1].ens_gene.unique(), tissue_df, show=False)
peahif = tea.enrichment_analysis(hif[hif.qval < 0.1].ens_gene.unique(), phenotype_df, show=False)
geahif = tea.enrichment_analysis(hif[hif.qval < 0.1].ens_gene.unique(), go_df, show=False)


teahif


peahif


tea.plot_enrichment_results(geahif, analysis='go')


# Many terms later, a conclusion is not to be had from enrichment analysis. The transcriptomic results reflect the known involvement of the hypoxic response in the immune response. On the other hand, the transcriptome results from *hif-1* suggest that HIF-1 regulates chaperones tightly even under normoxic conditions.
# 

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Sleuth-Prep-File" data-toc-modified-id="Sleuth-Prep-File-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Sleuth Prep File</a></div>
# 

# # Sleuth Prep File
# 
# This script does a number of things.
# 
# * Makes a file with the correct design matrix for each genotype
# * Places files into folders for sleuth processing. 
# 

import pandas as pd
import os
import shutil


batch = False
kallisto_loc = '../input/kallisto_all/'
genmap = pd.read_csv('../input/library_genotype_mapping.txt', comment='#')
genmap.genotype = genmap.genotype.apply(str)
genmap.genotype = genmap.genotype.apply(str.lower) # make sure everything is always in lowercase
# Make all the folders required for sleuth processing
sleuth_loc = '../sleuth/'


genmap.head()


# Make all possible combinations of WT, X
combs = []
for gene in genmap.genotype.unique():
    if gene != 'wt':
        combs += [['WT', gene]]



if not os.path.exists(sleuth_loc):
    os.makedirs(sleuth_loc)

# sort the groups by batches, then do the comparisons by batch
grouped = genmap.groupby('batch')

# do the comparison by batches
for name, group in grouped:
    if batch == True:
        WTnames = genmap[genmap.genotype=='wt'].project_name.values
    else:
        WTnames = group[group.genotype=='wt'].project_name.values
    print(name, )

    # For each combination, make a folder
    for comb in combs:
        current = sleuth_loc + comb[0]+'_'+comb[1]
        MTnames = group[group.genotype == comb[1]].project_name.values
        if len(MTnames) == 0:
            continue
    
        if not os.path.exists(current):
            os.makedirs(current)
    
        # copy the right files into the new directory
        # inside a folder called results
        def copy_cat(src_folder, dst_folder, names):
            """
            A function that copies a set of directories from one place to another.
            """
            for name in names:
#               print('The following file was created:', dst_folder+name)
                shutil.copytree(src_folder + name, dst_folder + name)
        
        # copy WT files into the new directory
        copy_cat(kallisto_loc, current+'/results/', WTnames)
    
        # copy the MT files into the new directory
        copy_cat(kallisto_loc, current+'/results/', MTnames)


def matrix_design(name, factor, df, a, b, directory, batch=False):
    """
    A function that makes the matrix design file for sleuth.
    
    This function can only make single factor design matrices. 
    
    This function requires a folder 'results' to exist within
    'directory', and the 'results' folder in turn must contain
    files that are named exactly the same as in the dataframe.
    
    name - a string
    factor - list of factors to list in columns
    df - a dataframe containing the list of project names and the value for each factor
    i.e. sample1, wt, pathogen_exposed.
    a, b - conditions to slice the df with, i.e: a=WT, b=MT1
    directory - the directory address to place file in folder is in.

    """
    
    with open(directory + name, 'w') as f:
        f.write('# Sleuth design matrix for {0}-{1}\n'.format(a, b))
        f.write('experiment {0}'.format(factor))
        if batch:
            f.write(' batch')
        f.write('\n')
        
        # walk through the results directory and get each folder name
        # write in the factor value by looking in the dataframe
        names = next(os.walk(directory+'/results/'))[1]
        for name in names:
            fval = df[df.project_name == name][factor].values[0]
            
            if batch:
                batchvar = df[df.project_name == name].batch.values[0]
            
            # add a if fval is WT or z otherwise
            # this is to ensure sleuth does
            # the regression as WT --> MT
            # but since sleuth only works alphabetically
            # simply stating WT --> MT doesn't work
            if fval == 'wt':
                fval = 'a' + fval
            else:
                fval = 'z' + fval
            
            if batch:
                line = name + ' ' + fval + ' ' + batchvar + '\n'
            else:
                line = name + ' ' + fval + '\n'
            f.write(line)
        

# Now make the matrix for each combination
# I made this separately from the above if loop
# because R is stupid and wants the files in this
# folder to be in the same order as they are 
# listed in the matrix design file....
for comb in combs:
    current = sleuth_loc + comb[0]+'_'+comb[1] + '/'
    
    # write a file called rna_seq_info for each combination
    matrix_design('rna_seq_info.txt', 'genotype', genmap,
                  comb[0], comb[1], current, batch=False)











# # Table of Contents
#  <p>
# 

for name, group in thomas.genmap.groupby('genotype'):
    sns.boxplot()
    print(name)
    for p1 in group.project_name:
        for p2 in group.project_name:
            s = sts.spearmanr(thomas.tpm[p1].tpm, thomas.tpm[p2].tpm)
            print('{0}, {1}, {2:2g}'.format(p1, p2, s[0]))


# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Generating-synthetic-data" data-toc-modified-id="Generating-synthetic-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Generating synthetic data</a></div><div class="lev1 toc-item"><a href="#Line-fitting-using-Bayes'-theorem" data-toc-modified-id="Line-fitting-using-Bayes'-theorem-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Line fitting using Bayes' theorem</a></div><div class="lev1 toc-item"><a href="#Quantifying-the-probability-of-a-fixed-model:" data-toc-modified-id="Quantifying-the-probability-of-a-fixed-model:-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Quantifying the probability of a fixed model:</a></div><div class="lev1 toc-item"><a href="#Selecting-between-two-models" data-toc-modified-id="Selecting-between-two-models-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Selecting between two models</a></div><div class="lev2 toc-item"><a href="#Different-datasets-will-prefer-different-models" data-toc-modified-id="Different-datasets-will-prefer-different-models-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Different datasets will prefer different models</a></div><div class="lev1 toc-item"><a href="#The-larger-the-dataset,-the-more-resolving-power" data-toc-modified-id="The-larger-the-dataset,-the-more-resolving-power-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>The larger the dataset, the more resolving power</a></div>
# 

# Welcome to our primer on Bayesian Model Selection. 
# 
# As always, we begin by loading our required libraries.
# 

# important stuff:
import os
import pandas as pd
import numpy as np
import statsmodels.tools.numdiff as smnd
import scipy

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
# other style specs must come AFTER
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


# # Generating synthetic data
# 
# First, we will generate the data. We will pick evenly spaced x-values. The y-values will be picked according to the equation $y=-\frac{1}{2}x$ but we will add Gaussian noise to each point. Each y-coordinate will have an associated error. The size of the error bar will be selected randomly.
# 
# After we have picked the data, we will plot it to visualize it. It looks like a fairly straight line.
# 

n = 50  # number of data points
x = np.linspace(-10, 10, n)
yerr = np.abs(np.random.normal(0, 2, n))
y = np.linspace(5, -5, n) + np.random.normal(0, yerr, n)
plt.scatter(x, y)


# # Line fitting using Bayes' theorem
# 
# Now that we have generated our data, we would like to find the line of best fit given our data. To do this, we will perform a Bayesian regression. Briefly, Bayes equation is,
# 
# $$
# P(\alpha~|D, M_1) \propto P(D~|\alpha, M_1)P(\alpha~|M_1).
# $$
# 
# In other words, the probability of the slope given that Model 1 (a line with unknown slope) and the data is proportional to the probability of the data given the model and alpha times the probability of alpha given the model. 
# 
# Some necessary nomenclature at this point:
#  * $P(D~|\alpha, M_1)\cdot P(\alpha|M_1)$ is called the posterior probability
#  * $P(\alpha~|M_1)$ is called the prior
#  * $P(D~|\alpha, M_1)$ is called the likelihood
#  
# 
# I claim that a functional form that will allow me to fit a line through this data is:
# 
# $$
# P(X|D) \propto \prod_{Data} \mathrm{exp}(-{\frac{(y_{Obs} - \alpha x)^2}{2\sigma_{Obs}^2}})\cdot (1 + \alpha^2)^{-3/2}
# $$
# 
# The first term in the equation measures the deviation between the observed y-coordinates and the predicted y-coordinates from a theoretical linear model, where $\alpha$ remains to be determined. We weight the result by the observed error, $\sigma_{Obs}$. Then, we multiply by a prior that tells us what values of $\alpha$ should be considered. How to pick a good prior is somewhat difficult and a bit of an artform. One way is to pick a prior that is uninformative for a given parameter. In this case, we want to make sure that we sample slopes between [0,1] as densely as we sample [1,$\infty$]. For a more thorough derivation and explanation, please see [this excellent blog post](http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/) by Jake Vanderplas.
# 
# The likelihood is the first term, and the prior is the second. We code it up in the next functions, with a minor difference. It is often computationally much more tractable to compute the natural logarithm of the posterior, and we do so here. 
# 
# We can now use this equation to find the model we are looking for. How? Well, the equation above basically tells us what model is most likely given that data and the prior information on the model. If we maximize the probability of the model, whatever parameter combination can satisfy that is a model that we are interested in!
# 

# bayes model fitting:
def log_prior(theta):
    beta = theta
    return -1.5 * np.log(1 + beta ** 2)

def log_likelihood(beta, x, y, yerr):
    sigma = yerr
    y_model = beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y, yerr):
    return log_prior(theta) + log_likelihood(theta, x, y, yerr)

def neg_log_prob_free(theta, x, y, yerr):
    return -log_posterior(theta, x, y, yerr)


# Specificity is necessary for credibility. Let's show that by optimizing the posterior function, we can fit a line.
# 
# We optimize the line by using the function `scipy.optimize.minimize`. However, minimizing the logarithm of the posterior does not achieve anything! We are looking for the place at which the equation we derived above is maximal. That's OK. We will simply multiply the logarithm of the posterior by -1 and minimize that.
# 

# calculate probability of free model:
res = scipy.optimize.minimize(neg_log_prob_free, 0, args=(x, y, yerr), method='Powell')

plt.scatter(x, y)
plt.plot(x, x*res.x, '-', color='g')
print('The probability of this model is {0:.2g}'.format(np.exp(log_posterior(res.x, x, y, yerr))))
print('The optimized probability is {0:.4g}x'.format(np.float64(res.x)))


# We can see that the model is very close to the model we drew the data from. It works! 
# 
# However, the probability of this model is not very large. Why? Well, that's because the posterior probability is spread out over a large number of parameters. Bayesians like to think that a parameter is actually a number plus or minutes some jitter. Therefore, the probability of the parameter being exactly one number is usually smaller the larger the jitter. In thise case, the jitter is not terribly a lot, but the probability of this one parameter being exactly -0.5005 is quite low, even though it is the best guess for the slope given the data. 
# 

# # Quantifying the probability of a fixed model:
# 
# Suppose now that we had a powerful theoretical tool that allowed us to make a very, very good guess as to what line the points should fall on. Suppose this powerful theory now tells us that the line should be:
# 
# $$
# y = -\frac{1}{2}x.
# $$
# 
# Using Bayes' theorem, we could quantify the probability that the model is correct, given the data. Now, the prior is simply going to be 1 when the slope is -0.5, and 0 otherwise. This makes the equation:
# 
# $$
# P(X|D) \propto \prod_{Data}\mathrm{exp}({-\frac{(y_{Obs} + 0.5x)^2}{2\sigma_{Obs}}})
# $$
# 
# Notice that this equation cannot be minimized. It is a fixed statement, and its value depends only on the data. 
# 

# bayes model fitting:
def log_likelihood_fixed(x, y, yerr):
    sigma = yerr
    y_model = -1/2*x

    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior_fixed(x, y, yerr):
    return log_likelihood_fixed(x, y, yerr)


plt.scatter(x, y)
plt.plot(x, -0.5*x, '-', color='purple')
print('The probability of this model is {0:.2g}'.format(np.exp(log_posterior_fixed(x, y, yerr))))


# We can see that the probability of this model is very similar to the probability of the alternative model we fit above. How can we pick which one to use?

# # Selecting between two models
# 
# An initial approach to selecting between these two models would be to take the probability of each model given the data and to find the quotient, like so:
# 
# $$
# OR = \frac{P(M_1~|D)}{P(M_2~|D)} = \frac{P(D~|M_1)P(M_1)}{P(D~|M_2)P(M_1)}
# $$
# 
# However, this is tricky to evaluate. First of all, the equations we derived above are not solely in terms of $M_1$ and $D$. They also include $\alpha$ for the undetermined slope model. We can get rid of this parameter via a technique known as marginalization (basically, integrating the equations over $\alpha$). Even more philosophically difficult are the terms $P(M_i)$. How is one to evaluate the probability of a model being true? The usual solution to this is to set $P(M_i) \sim 1$ and let those terms cancel out. However, in the case of models that have been tested before or where there is a powerful theoretical reason to believe one is more likely than the other, it may be entirely reasonable to specify that one model is several times more likely than the other. For now, we set the $P(M_i)$ to unity.
# 
# We can approximate the odds-ratio for our case as follows:
# 
# $$
# OR = \frac{P(D|\alpha^*)}{P(D|M_2)} \cdot \frac{P(\alpha^*|M_1) (2\pi)^{1/2} \sigma_\alpha^*}{1},
# $$
# 
# where $\alpha^*$ is the parameter we found when we minimized the probability function earlier. Here, the second term we added represents the complexity of each model. The denominator in the second term is 1 because the fixed model cannot become any simpler. On the other hand, we penalize the model with free slope by multiplying the probability of the observed slope by the square root of two pi and then multiplying all of this by the uncertainty in the parameter $\alpha$. This is akin to saying that the less likely we think $\alpha$ should be *a priori*, or the more uncertain we are that $\alpha$ is actually a given number, then we should give points to the simpler model. 
# 

def model_selection(X, Y, Yerr, **kwargs):
    guess = kwargs.pop('guess', -0.5)

    # calculate probability of free model:
    res = scipy.optimize.minimize(neg_log_prob_free, guess, args=(X, Y, Yerr), method='Powell')
    
    # Compute error bars
    second_derivative = scipy.misc.derivative(log_posterior, res.x, dx=1.0, n=2, args=(X, Y, Yerr), order=3)
    cov_free = -1/second_derivative
    alpha_free = np.float64(res.x)
    log_free = log_posterior(alpha_free, X, Y, Yerr)
    
    # log goodness of fit for fixed models
    log_MAP = log_posterior_fixed(X, Y, Yerr)

    good_fit = log_free - log_MAP

    # occam factor - only the free model has a penalty
    log_occam_factor =(-np.log(2 * np.pi) + np.log(cov_free)) / 2 + log_prior(alpha_free)

    # give more standing to simpler models. but just a little bit!
    lg = log_free - log_MAP + log_occam_factor - 2
    return lg


# We performed the Odds Ratio calculation on logarithmic space, so negative values show that the simpler (fixed slope) model is preferred, whereas if the values are positive and large, the free-slope model is preferred. 
# 
# As a guide, Bayesian statisticians usually suggest that 10^2 or above is a good ratio to abandon one model completely in favor of another. 
# 

model_selection(x, y, yerr)


# ## Different datasets will prefer different models
# 
# Let's try this again. Maybe the answer will change sign this time.
# 

n = 50  # number of data points
x = np.linspace(-10, 10, n)
yerr = np.abs(np.random.normal(0, 2, n))
y = x*-0.55 + np.random.normal(0, yerr, n)
plt.scatter(x, y)


model_selection(x, y, yerr)


# Indeed, the answer changed sign. Odds Ratios, p-values and everything else should always be interpreted conservatively. I prefer odds ratios that are very large, larger than 1,000 before stating that one model is definitively preferred. Otherwise, I tend to prefer the simpler model.
# 

# # The larger the dataset, the more resolving power
# 
# What distribution of answers would you get if you obtained five points? Ten? Fifteen? I've written a couple of short functions to help us find out.
# 
# In the functions below, I simulate two datasets. One datasets is being plucked from points that obey the model 
# 
# $$
# y = -\frac{1}{2}x,
# $$
# 
# whereas the second model is being plucked from
# 
# $$
# y = -0.46x.
# $$
# 
# Clearly, the fixed model $y=-0.5x$ should only be preferred for the first dataset, and the free model is the correct one to use for the second model. Now let us find out if this is the case.
# 
# By the way, the function below trims odds ratios to keep them from becoming too large. If an odds ratio is bigger than 10, we set it equal to 10 for plotting purposes.
# 

def simulate_many_odds_ratios(n):
    """
    Given a number `n` of data points, simulate 1,000 data points drawn from a null model and an alternative model and
    compare the odds ratio for each.
    """
    iters = 1000
    lg1 = np.zeros(iters)
    lg2 = np.zeros(iters)

    for i in range(iters):
        x = np.linspace(-10, 10, n)
        yerr = np.abs(np.random.normal(0, 2, n))

        # simulate two models: only one matches the fixed model
        y1 = -0.5*x + np.random.normal(0, yerr, n)
        y2 = -0.46*x + np.random.normal(0, yerr, n)

        lg1[i] = model_selection(x, y1, yerr)
        
        m2 = model_selection(x, y2, yerr)
        # Truncate OR for ease of plotting
        if m2 < 10:
            lg2[i] = m2
        else:
            lg2[i] = 10
            
    return lg1, lg2


def make_figures(n):
    lg1, lg2 = simulate_many_odds_ratios(n)
    
    lg1 = np.sort(lg1)
    lg2 = np.sort(lg2)
    
    fifty_point1 = lg1[int(np.floor(len(lg1)/2))]
    fifty_point2 = lg2[int(np.floor(len(lg2)/2))]
    
    fig, ax = plt.subplots(ncols=2, figsize=(15, 7), sharey=True)
    fig.suptitle('Log Odds Ratio for n={0} data points'.format(n), fontsize=20)
    sns.kdeplot(lg1, label='slope=-0.5', ax=ax[0], cumulative=False)
    ax[0].axvline(x=fifty_point1, ls='--', color='k')
    ax[0].set_title('Data drawn from null model')
    ax[0].set_ylabel('Density')

    sns.kdeplot(lg2, label='slope=-0.46', ax=ax[1], cumulative=False)
    ax[1].axvline(x=fifty_point2, ls='--', color='k')
    ax[1].set_title('Data drawn from alternative model')
    fig.text(0.5, 0.04, 'Log Odds Ratio', ha='center', size=18)

    return fig, ax


fig, ax = make_figures(n=5)


# Here we can see that with five data points, the odds ratio will tend to prefer the simpler model. We do not have too much information---why request the extra information? Note that for the second dataset in some cases the deviations are great enough that the alternative model is strongly preferred (right panel, extra bump at 10). However, this is rare.
# 

fig, ax = make_figures(n=50)


# When we increase the number of points we are using, the curve on the right stays fairly similar, but the right panel shows an important shift towards large positive numbers. Sometimes we still prefer the simpler model, but much more often we can tell the difference between a slope of -0.46 and -0.5!
# 

# # Table of Contents
#  <p>
# 

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# set to use tex, but make sure it is sans-serif fonts only
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
# other style specs must come AFTER
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


n2_1= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17434_indexN704-N517/kallisto/abundance.tsv', sep='\t')
n2_2= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17435_indexN704-N502/kallisto/abundance.tsv', sep='\t')
n2_3= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17436_indexN704-N503/kallisto/abundance.tsv', sep='\t')

egl_1= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17437_indexN704-N504/kallisto/abundance.tsv', sep='\t')
egl_2= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17438_indexN704-N505/kallisto/abundance.tsv', sep='\t')
egl_3= pd.read_csv('../../sleuth_all_adjusted/kallisto/results/Project_17439_indexN704-N506/kallisto/abundance.tsv', sep='\t')

egl9_beta = pd.read_csv('../../sleuth_all_adjusted/kallisto/betasB.csv')


frames = []

for df in [n2_1, n2_2, n2_3]:
    df['genotype'] = 'wt'
    frames += [df]

for df in [egl_1, egl_2, egl_3]:
    df['genotype'] = 'egl-9'
    frames += [df]


tidy = pd.concat(frames)


plot_up = tidy[tidy.target_id == 'R08E5.3'].copy()
plot_up['logtpm'] = plot_up.tpm.apply(np.log)
plot_up['logcounts'] = plot_up.est_counts.apply(np.log)
plot_up['estcounts'] = plot_up['est_counts']

plot_down = tidy[tidy.target_id == 'F15E11.15a'].copy()
plot_down['logtpm'] = plot_down.tpm.apply(np.log)
plot_down['logcounts'] = (plot_down.est_counts + .5).apply(np.log)
plot_down['estcounts'] = plot_down['est_counts']


bup = egl9_beta[egl9_beta.target_id == 'R08E5.3'].b.values[0]
bdown = egl9_beta[egl9_beta.target_id == 'F15E11.15a'].b.values[0]


x = np.linspace(0, 1)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(10, 10))

fig.suptitle(r'Calculation and meaning of \beta', fontsize=20)


sns.swarmplot(x='genotype', y='estcounts', data=plot_up, size=10, ax=ax[0, 0])
ax[0, 0].set_yticks([0, 12500, 25000])
ax[0, 0].set_ylabel('est.counts')
ax[0, 0].set_xlabel('')
ax[0, 0].set_title('R08E5.3')


sns.swarmplot(x='genotype', y='logcounts', data=plot_up, size=10, ax=ax[1, 0])
plt.ylim([5, 11])
ax[1, 0].set_yticks([5, 7.5, 10])
ax[1, 0].plot(x, x*bup + plot_up[plot_up.genotype == 'wt'].logcounts.mean(), 'k')
ax[1, 0].set_xlabel('')
ax[1, 0].set_ylabel(r'$\log{(\mathrm{est.counts})}$')
ax[1, 0].set_xticks([0, 1])


sns.swarmplot(x='genotype', y='estcounts', data=plot_down, size=10, ax=ax[0, 1])
ax[0, 1].set_xlabel('')
ax[0, 1].set_ylabel('')
ax[0, 1].set_title('F15E11.15a')


sns.swarmplot(x='genotype', y='logcounts', data=plot_down, size=10, ax=ax[1, 1])
plt.ylim([-2, 7])
ax[1, 1].set_yticks([-2, 2.5, 7])
ax[1, 1].plot(x, x*bdown + plot_down[plot_down.genotype == 'wt'].logcounts.mean(), 'k')
ax[1, 1].set_xlabel('')
ax[1, 1].set_ylabel('')

fig.text(0.5, 0.04, 'Genotype', ha='center', size=18)

plt.savefig('../../output/meaningofbeta.svg', bbox_inches='tight')


# \begin{cases} 
#       H_0: \beta_{i,\mathrm{egl-9}} = 0\      H_1: \beta_{i,\mathrm{egl-9}} \neq 0 \\end{cases}
# 
# $$
#       \mathrm{iff}~q_{i, \mathrm{egl-9}}<0.1,~\mathrm{reject}~ H_0 \$$
# 




# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Transcription-Factors-in-the-Hypoxic-Response" data-toc-modified-id="Transcription-Factors-in-the-Hypoxic-Response-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Transcription Factors in the Hypoxic Response</a></div><div class="lev1 toc-item"><a href="#Identify-the-transcription-factors-in-the-hypoxia-response" data-toc-modified-id="Identify-the-transcription-factors-in-the-hypoxia-response-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Identify the transcription factors in the hypoxia response</a></div><div class="lev2 toc-item"><a href="#A-slightly-less-restrictive-approach-reveals-even-more-Transcription-Factors" data-toc-modified-id="A-slightly-less-restrictive-approach-reveals-even-more-Transcription-Factors-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>A slightly less restrictive approach reveals even more Transcription Factors</a></div>
# 

# In this notebook, I dissect what transcription factors are present in the hypoxia response we defined earlier. 
# 

# important stuff:
import os
import pandas as pd
import numpy as np

import tissue_enrichment_analysis as tea
import morgan as morgan
import epistasis as epi
import genpy

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

ft = 35 #title fontsize
import genpy
import gvars

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14


tfs = pd.read_csv('../input/tf_list.csv')


q = 0.1
# this loads all the labels we need
genvar = gvars.genvars()


# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'
# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.set_qval()
thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')

# load all the beta values for each genotype:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)
thomas.filter_data()


frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
    df['sorter'] = genvar.sort_muts[key]

tidy = pd.concat(frames)
tidy.sort_values('sorter', inplace=True)
tidy.dropna(subset=['ens_gene'], inplace=True)


# # Transcription Factors in the Hypoxic Response
# 
# First, it is useful to get an overview of the number of transcription factors involved. Therefore, I will explore how many TFs were present in each mutant transcriptome. 
# 

codes = ['a', 'b', 'c', 'd', 'e', 'f']

print('Genotype, #TFs')
for c in codes:
    ind = (tidy.qval < q) & (tidy.code == c) & (tidy.target_id.isin(tfs.target_id))
    print(genvar.mapping[c], tidy[ind].shape[0])


# Next, I will extract the hypoxia response from this dataset. Remember, the hypoxia response is equivalent to the intersection of the *egl-9*, *rhy-1*, and *vhl-1* containing mutants. 
# 

# extract the hypoxia response:
hyp_response_pos = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b > 0])
hyp_response_neg = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b < 0])
hyp_response = list(set(hyp_response_neg + hyp_response_pos))


# # Identify the transcription factors in the hypoxia response
# 
# Let's extract them, and plot them.
# 

# find tfs in the hif-1 response
tfs_in_hif = tfs[tfs.target_id.isin(hyp_response)].target_id
print('There are {0} transcription factors in HIF-1+ animals'.format(tfs_in_hif.shape[0]))

# The qPCR function I wrote is quite stupid, so I always have to tidy up my dataframe a little
# bit and add a couple of columns:

# select the data to be plotted:
plotdf = tidy[tidy.target_id.isin(tfs_in_hif)].copy()
# sort by genotype
plotdf.sort_values(['genotype', 'target_id'], inplace=True)
# add an 'order' column
plot_order = {i: t+1 for t, i in enumerate(plotdf.target_id.unique())}
plotdf['order'] = plotdf.target_id.map(plot_order)
# sort by 'order'
plotdf.sort_values('order', inplace=True)
plotdf.reset_index(inplace=True)  


genpy.qPCR_plot(plotdf[plotdf.code != 'g'], genvar.plot_order, genvar.plot_color, clustering='genotype',
                plotting_group='target_id', rotation=90)


# ## A slightly less restrictive approach reveals even more Transcription Factors
# 
# *vhl-1* has the weakest transcriptomic phenotype and as a result strongly constrains what we call a hypoxia response. What TFs come out if we just find the intersection between *egl-9*, *rhy-1* and *egl-9;vhl-1* genotypes?

# extract the hypoxia response:
hyp_response_pos = epi.find_overlap(['e', 'b', 'a'], tidy[tidy.b > 0])
hyp_response_neg = epi.find_overlap(['e', 'b', 'a'], tidy[tidy.b < 0])
hyp_response = list(set(hyp_response_neg + hyp_response_pos))


print('There are {0} isoforms in the relaxed hypoxia response'.format(len(hyp_response)))
tfs_in_hif = tfs[tfs.target_id.isin(hyp_response)].target_id
print('There are {0} transcription factors in HIF-1+/HIF-1OH- animals'.format(tfs_in_hif.shape[0]))

plotdf = tidy[tidy.target_id.isin(tfs_in_hif)].copy()
plotdf.sort_values(['genotype', 'target_id'], inplace=True)
plot_order = {i: t+1 for t, i in enumerate(plotdf.target_id.unique())}
plotdf['order'] = plotdf.target_id.map(plot_order)
plotdf.sort_values('order', inplace=True)
plotdf.reset_index(inplace=True)  
plotdf = plotdf[['target_id', 'ens_gene', 'ext_gene','b', 'se_b', 'qval', 'genotype', 'order', 'code']]


genpy.qPCR_plot(plotdf[plotdf.code != 'g'], genvar.plot_order, genvar.plot_color, clustering='genotype',
                plotting_group='target_id', rotation=90)





# # Table of Contents
#  <p>
# 

import os
import pandas as pd
# params
directory = '../input/rawseq'
length = 180
sigma = 60
btstrp = 200
thrds = 6

# sequences:
seqs = next(os.walk(directory))[1]


# params
directory = '../input/rawseq'
length = 180
sigma = 60
btstrp = 200
thrds = 6

# sequences:
seqs = next(os.walk(directory))[1]


def explicit_kallisto(directory, files, res_dir):
    """
    TODO: Make a function that allows you to systematically 
    set up each parameter for each sequencing run individually.
    """
    
    if type(directory) is not str:
        raise ValueError('directory must be a str')
    if type(files) is not list:
        raise ValueError('files must be a list')
    
    print('This sequence file contains a Kallisto_Info file            and cannot be processed at the moment.')
    return '# {0} could not be processed'.format(res_dir), ''
    
def implicit_kallisto(directory, files, res_dir):
    """
    A function to write a Kallisto command with standard parameter
    setup
    """
    if type(directory) is not str:
        raise ValueError('directory must be a str')
    if type(files) is not list:
        raise ValueError('files must be a list')

    # parts of each kallisto statement
    
    # information
    info = '# kallisto command for {0}'.format(directory)
    # transcript file location:
    k_head = 'kallisto quant -i input/transcripts.idx -o '
    
    # output file location
    k_output = 'input/kallisto_all/' + res_dir + '/kallisto '
    # parameter info:
    k_params = '--single -s {0} -l {1} -b {2} -t {3} --bias --fusion'.format(sigma, length, btstrp, thrds)
    
    # what files to use:
    k_files = ''    
    # go through each file and add it to the command
    # unless it's a SampleSheet.csv file, in which
    # case you should ignore it. 
    for y in files:
        if y != 'SampleSheet.csv':
            if directory[:3] == '../':
                d = directory[3:]
            else:
                d = directory[:]
            k_files += ' '+ d + '/' + y
    # all together now:
    kallisto = k_head + k_output + k_params + k_files +';'
    return info, kallisto


def walk_seq_directories(directory):
    """
    Given a directory, walk through it,
    find all the rna-seq repository folders
    and generate kallisto commands
    """
    kallisto = ''
    #directory contains all the projects, walk through it:
    for x in os.walk(directory):
        # first directory is always parent
        # if it's not the parent, move forward:
        if x[0] != directory:
            # cut the head off and get the project name:
            res_dir = x[0][len(directory)+1:]
            
            # if this project has attributes explicitly written in
            # use those parameter specs:
            if 'Kallisto_Info.csv' in x[2]:
                info, command = explicit_kallisto(x[0], x[2], res_dir)
                continue
            
            # otherwise, best guesses:
            info, command = implicit_kallisto(x[0], x[2], res_dir)
            kallisto += info + '\n' + command + '\n'
            
            if not os.path.exists('../input/kallisto_all/' + res_dir):
                os.makedirs('../input/kallisto_all/' + res_dir)
    return kallisto

with open('../kallisto_commands.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# make transcript index\n')
    f.write('kallisto index -i input/transcripts.idx input/c_elegans_WBcel235.rel79.cdna.all.fa;\n')
    kallisto = walk_seq_directories(directory)
    f.write(kallisto)








