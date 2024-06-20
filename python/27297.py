import mutex as mex
import csv

mutationmatrix = '/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.m2'
patientFile = '/Users/jlu96/maf/new/OV_broad/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
closer_than_distance = 10000000
partition_file = '/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.ppf9'
load_pmm_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM.txt'
dna_pmm_comparison_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-dnacomp.txt'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

if COSMICFile:
    COSMICgenes = set()
    with open(COSMICFile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            COSMICgenes.add(*row)
    print "Number of COSMIC genes ", len(COSMICgenes)
    genes = (set(genes)).intersection(COSMICgenes)
    geneToCases = dict([g for g in geneToCases.items() if g[0] in genes])

print "Num COSMIC genes in this cancer", len(genes)
            


cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

print clusterToProp.keys()




# let's look at the smallest cluster
c0patients = clusterToProp[4]['Patients']
c1patients = clusterToProp[6]['Patients']

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[4]['Mean']
print list(c0genes)[0:30]
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 50
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/OV_broad-cna-jl-cpairs-min_cohort.txt'



genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[4])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


# # Below uses the bottom 15 %, least mutated patients
# # It filters to those genes mutated in 10% of those patients
# # It uses segmentation
# # OV_broad has ~5000 segments, unlike BRCA's ~2000. Let's see how well we do under multiple testing: do many end up significant?
# 

# Build this so that it takes all patients within the farthest cluster

import mutex as mex
import csv
import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
import time
import os
import scipy.stats as stats
import partition as par
import numpy as np


cancer = 'OV_broad'
suffix = '-seg-jl'
top_folder = '/Users/jlu96/maf/new/'
mutationmatrix = top_folder + cancer + '/' + cancer + suffix + '.m2'
patientFile = top_folder + cancer + '/shared_patients.plst'
partition_file = top_folder + cancer + '/' + cancer + '-cna-jl.ppf'
segment_info_file = top_folder + cancer + "/segment_info.txt"
file_prefix = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/' + cancer + suffix
cpairfile = file_prefix + '-cpairs-min_cohort.txt'
triplet_file_prefix = file_prefix + '-triplet-'
new_cpairfile = file_prefix + "-cpairs-min_cohort_filtered.txt"
geneFile = None
minFreq = 0
compute_mutex = True
closer_than_distance = 100000000
test_minFreq = 0.3
minPercentile = 15
cpairPercentile = 1
mpairPercentile = 1 # If do_max_pairs is true, this is adjusted to ensure that that maximum number of pairs is tested
max_m_pairs = 5000
max_c_pairs = 5000
do_max_m_pairs = True
do_max_c_pairs = True
pthresh = 0.05
use_whole_partition = False


numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

D = [len(patientToGenes[p]) for p in patientToGenes]
minThreshold = stats.scoreatpercentile(D, minPercentile)

c0patients = set([p for p in patientToGenes if len(patientToGenes[p]) <= minThreshold])

if use_whole_partition:
    cohort_dict, clusterToProp, min_cohort = par.load_patient_cohorts(partition_file, patientToGenes)
    max_cluster = max([c for c in cohort_dict if cohort_dict[c].intersection(c0patients)])
    print "Largest cluster containing is ", max_cluster, " with mean ", clusterToProp[max_cluster]['Mean'], "and number ", len(clusterToProp[max_cluster]['Patients'])

    c0patients = set(c0patients)
    c0patients = c0patients.union(cohort_dict[max_cluster])

print "Number of new patients is ", len(c0patients)


test_minFreq = int( test_minFreq * len(c0patients))
c0cohort_dict = {0: c0patients}
c0genes, c0geneToCases, c0patientToGenes = par.get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)

print "number genes in smallest cluster is ", len(c0genes)
print "number of genes above threashold ", len([g for g in c0genes if len(c0geneToCases[g]) >= test_minFreq])
print "number patients is ", len(c0patients)

t = time.time()
genepairs = bgbp.getgenepairs(c0geneToCases, c0genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs), " retrieved in time : ", time.time() - t

if do_max_m_pairs:
    mpairPercentile = max_m_pairs * 1.0 / len(genepairs) * 100.0
    print "We will test only the top ", max_m_pairs, " pairs, using percentile threshold of ", mpairPercentile
if do_max_c_pairs:
    cpairPercentile = max_c_pairs * 1.0 / len(genepairs) * 100.0
    print "We will test only the top ", max_c_pairs, " pairs, using percentile threshold of ", cpairPercentile


os.system('say "pairs retrieved"')

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
print "Getting cooccurrence across the whole distribution"

cpairsdict = chi.add_BinomP_cohorts_all_pairs(cpairsdict, geneToCases, patientToGenes, c0cohort_dict, c0patients)

# cpairsdict = chi.add_BinomP_all_pairs(cpairsdict, geneToCases, patientToGenes)
print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')

met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


cpvalues = np.array([cpairsdict[c]['1CBinomProb0'] for c in cpairsdict])
logcp = np.log(cpvalues)

threshold = pthresh/len(logcp)

plt.figure()
plt.hist(logcp, bins=50)
plt.title("Distribution of log p-values")
plt.show()


mpvalues = np.array([cpairsdict[c]['1MBinomProb0'] for c in cpairsdict])
logmp = np.log(mpvalues)

threshold = pthresh/len(logmp)

plt.figure()
plt.hist(logmp, bins=50)
plt.title("Distribution of log p-values")
plt.show()

cthreshold = stats.scoreatpercentile(cpvalues, cpairPercentile)
mthreshold = stats.scoreatpercentile(mpvalues, mpairPercentile)
print "Top ", cpairPercentile, "percent of cooccurring pairs: ", cthreshold
print "Top ", mpairPercentile, "percent of mutually exclusive pairs : ", mthreshold

# Let's get the top 10 percent of pairs

goodpairs = [c for c in cpairsdict if (cpairsdict[c]['1CBinomProb0'] <= cthreshold or cpairsdict[c]['1MBinomProb0'] <= mthreshold)]
print "Now number of pairs to test ", len(goodpairs)


new_cpairsdict, new_cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, goodpairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(new_cpairsdict)
print "Getting cooccurrence across the whole distribution"

new_cpairsdict = chi.add_BinomP_cohorts_all_pairs(new_cpairsdict, geneToCases, patientToGenes, c0cohort_dict, c0patients)
new_cpairsdict = chi.add_BinomP_all_pairs(new_cpairsdict, geneToCases, patientToGenes)
print "Writing to file at ", new_cpairfile

fieldnames = (new_cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')

met.writeanydict(new_cpairsdict, new_cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


# Plot the p-value distribution
pvalues = np.array([new_cpairsdict[c]['BinomProbability'] for c in new_cpairsdict])
logp = np.log(pvalues)

threshold = 0.05/len(logp)

plt.figure()
plt.hist(logp, bins=50)
plt.title("Distribution of log p-values")
plt.axvline(x=np.log(threshold), ymin=0, ymax=1000)
plt.show()

sig = [pvalue for pvalue in pvalues if pvalue < threshold]
print "Number of significant pairs ", len(sig)


# add the segment infos

bgbp.write_segment_infos(c0genes, segment_info_file)

for pair in new_cpairsdict:
    info0 = bgbp.get_segment_gene_info(new_cpairsdict[pair]['Gene0'])
    new_cpairsdict[pair]['Gene0Loc'] = str(info0['Chromosome']) + ':' + str(info0['Start'])
    info1 = bgbp.get_segment_gene_info(new_cpairsdict[pair]['Gene1'])
    new_cpairsdict[pair]['Gene1Loc'] = str(info1['Chromosome']) + ':' + str(info1['Start'])
    
fieldnames += ['Gene0Loc', 'Gene1Loc']
print "Writing to file at ", new_cpairfile
met.writeanydict(new_cpairsdict, new_cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


# # Check the co-occurrence/ mutual exclusivity. Is it good? Are they near each other?
# 
# 
# First run: uses ~5000000 pairs
# 

__author__ = 'jlu96'
import mutex as mex
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from scipy.stats import poisson
from scipy import stats
import collections
import os

def partition_EM(patientToGenes, k):
    """
    :param geneToCases:
    :param patientToGenes:
    :param k: Number of partitions
    :return: cohort_list
    """

    # partition the patients, and intersect the geneToCases
    return



def partition_gene(patientToGenes, genes):
    """
    :param geneToCases:
    :param patientToGenes:
    :param genes:
    :return: cohorts by each gene. Size 2^(#genes)
    """

    cohorts = [patientToGenes.keys()]
    for gene in genes:
        new_cohorts = []
        for cohort in cohorts:
            new_cohort_1 = [patient for patient in patientToGenes if gene not in patientToGenes[patient]]
            if new_cohort_1:
                new_cohorts.append(new_cohort_1)
            new_cohort_2 = list(set(cohort).difference(set(new_cohort_1)))
            if new_cohort_2:
                new_cohorts.append(new_cohort_2)
        cohorts = new_cohorts
    # print genes
    # print cohorts

    return cohorts

def partition_gene_list(patientToGenes, genes, binary=True):
    """
    :param patientToGenes:
    :param genes:
    :return: The cohorts, ordered from least to greatest in number of those genes they have.
    If binary = True, return just those with, those without.

    """



    gene_set = set(genes)
    cohort_dict = {}

    for patient in patientToGenes:
        num = len(set.intersection(gene_set, patientToGenes[patient]))

        # just 0 and 1
        if binary:
            if num > 0:
                num = 1

        if num not in cohort_dict:
            cohort_dict[num] = []
        cohort_dict[num].append(patient)


    return cohort_dict


def get_patients_gene_mut_num(patients, genes, patientToGenes):
    return [set.intersection(patientToGenes[p], genes) for p in patients]

def integrate_cohorts(cohort_dict, numCases, num_integrated):
    cohorts_int = {}
    start_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > numCases/num_integrated):
            cohorts_int[start_index] = new_cohort
            start_index = i+1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[start_index] = new_cohort

    return cohorts_int

def integrate_cohorts_sizes(cohort_dict, sizes):
    cohorts_int = {}
    size_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > sizes[size_index]):
            cohorts_int[size_index] = new_cohort
            size_index += 1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[size_index] = new_cohort

    return cohorts_int


def draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title=None, num_bins=50):
    # LEFT OF HERE, JLU. Finish this, then above. Make plots in parallel, compare.
    # Work with: TP53? Others?

    numGenes = len(geneToCases.keys())
    numCohorts = len(cohort_pairings)

    cohort_frequencies = [[len(patientToGenes[case]) for case in cohort_pair[1]] for cohort_pair in cohort_pairings]
    cohort_names = [cohort_pair[0] for cohort_pair in cohort_pairings]

    draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=title, num_bins=num_bins)


def draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=None, num_bins=50):

    numCohorts = len(cohort_frequencies)
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)

    plt.figure()


    for i in range(len(cohort_frequencies)):
        plt.hist(cohort_frequencies[i], bins, alpha=1.0/numCohorts, label=str(cohort_names[i]))


    plt.title(title, fontsize=20)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    plt.show()

def norm(x, height, center, std):
    return(height*np.exp(-(x - center)**2/(2*std**2)))



def partition_GMM(patientToGenes, num_components, num_bins, title=None, do_plot=True):
    g = mixture.GMM(n_components=num_components)
    mut_num_list = [len(patientToGenes[p]) for p in patientToGenes]
    obs = np.array([[entry] for entry in mut_num_list])
    g.fit(obs)

    print "***********************************"
    print "COMPONENTS: ", num_components
    print "Weights: " + str(np.round(g.weights_,2))
    print "Means: " + str(np.round(g.means_,2))
    print "Covariates: " + str(np.round(g.covars_,2))

    print "Total log probability: " + str(sum(g.score(obs)))
    print "AIC: " + str(g.aic(obs))
    print "BIC: ", g.bic(obs)

    score, respon = g.score_samples(obs)

    for i in range(num_components):
        print "Model ", np.round(g.means_, 2)[i], " explains ", np.round(len([in_w for in_w in respon if in_w[i] == max(in_w)])) * 1.0 /len(respon)


    # Simulate gaussians
    # sim_samples = g.sample(len(patientToGenes))
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
    histogram = np.histogram([len(patientToGenes[p]) for p in patientToGenes], bins=bins)

    # get the scale of the gaussians from the biggest one
    # max_comp = g.weights_.index(max(g.weights_))
    # max_mean = g.means_[max_comp]

    which_bins = [[bin for bin in bins if bin > mean][0] for mean in g.means_]
    print which_bins
    print bins
    print histogram
    print bins.index(which_bins[0]) - 1
    bin_heights = [histogram[0][bins.index(which_bin) - 1] for which_bin in which_bins]
    # max_height = max(histogram)

    if do_plot:
        plt.figure()
        plt.hist([len(patientToGenes[p]) for p in patientToGenes], bins=bins)
        for i in range(num_components):
            X = np.arange(0, max(mut_num_list), 1)
            Y = norm(X, bin_heights[i], g.means_[i], np.sqrt(g.covars_[i]))
            plt.plot(X, Y, label=str(np.round(g.weights_[i], 3)), linewidth=5)
        plt.title("GMM size " + str(num_components), fontsize=20)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.show()
        # draw_partitions(patientToGenes, ['Original', 'Simulated'], [[len(patientToGenes[p]) for p in patientToGenes], sim_samples],
        #                 num_bins=num_bins, title=title)

    data = {}
    data['Components'] = num_components
    data['Weights'] = np.round(g.weights_,2)
    data['Means'] = np.round(g.means_,2)
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    data["AIC"] = g.aic(obs)
    data["BIC"] = g.bic(obs)
    data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data

def partition_gene_kmeans(geneToCases, patientToGenes, gene_list, num_components, num_bins, title=None, do_plot=True):

    # get gene index mapping
    giv = getgiv(geneToCases.keys(), gene_list)

    # convert patients into vectors
    patientToVector = getpatientToVector(patientToGenes, giv)

    vectors = patientToVector.values()

    print vectors[0]
    print "Length of vectors is ", len(vectors[0])

    km = KMeans(num_components)

    km.fit(vectors)

    clusterToPatient = {}

    for patient in patientToVector:
        cluster = km.predict(patientToVector[patient])[0]
        if cluster not in clusterToPatient:
            clusterToPatient[cluster] = set()
        clusterToPatient[cluster].add(patient)

    # plot patients in each cluster


    if do_plot:
        bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
        plt.figure()
        for cluster in clusterToPatient:
            plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.title("Kmeans size " + str(num_components), fontsize=20)
        plt.show()



    data = {}
    data['Score'] = km.score(vectors)
    data['Number'] = num_components
    data['% Explained'] = np.round([100 * len(clusterToPatient[cluster]) * 1.0 / len(patientToGenes) for cluster in clusterToPatient], 2)
    data['Vector size'] = len(vectors[0])
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    # data["AIC"] = g.aic(obs)
    # data["BIC"] = g.bic(obs)
    # data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data


def getgiv(all_genes, gene_list):
    """
    :param all_genes:
    :param gene_list:
    :return: A list of the genes in common, the gene_index_vector.
    """
    giv = list(set(all_genes).intersection(set(gene_list)))

    return giv



def getpatientToVector(patientToGenes, gene_index_vector):
    patientToVector = {}
    for patient in patientToGenes:
        patient_genes = patientToGenes[patient]
        patientToVector[patient] = []
        for gene in gene_index_vector:
            patientToVector[patient].append(1 if gene in patient_genes else 0)

    return patientToVector


def get_cluster_gTC_pTG(geneToCases, patientToGenes, patients):
    new_pTG = dict([c for c in patientToGenes.items() if c[0] in patients])
    new_genes = set.union(*new_pTG.values())
    new_gTC = dict([g for g in geneToCases.items() if g[0] in new_genes])
    for g in new_gTC:
        new_gTC[g] = new_gTC[g].intersection(patients)
    
    for g in new_genes:
        if g in new_gTC and not new_gTC[g]:
            new_gTC.pop(g)
    
    new_genes = new_genes.intersection(set(new_gTC.keys()))
    
    return new_genes, new_gTC, new_pTG










# 3/12/16-Jlu


class PMM:

    def __init__(self, filename=None, delimiter='\t', lam=None, p_k=None, classes=None, patientToGenes=None,
                data = None, clusterToPatient = None, do_fit=True):

        if filename:
            with open(filename, 'rU') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                row = reader.next()
                print row
                self.lam = eval(row['Means'])
                self.p_k = eval(row['Probabilities'])
                self.classes = eval(row['Classes']) if 'Classes' in row else range(len(self.lam))
                self.num_components = len(self.classes)
        else:
            self.lam = lam
            self.p_k = p_k
            self.classes = classes
            if not classes:
                self.classes = range(len(self.lam))
            self.num_components = len(self.classes)


        self.data = data
        self.clusterToPatient = clusterToPatient
        print "Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()

        self.patientToGenes = patientToGenes

        if patientToGenes and do_fit:
            self.fit_to_data(patientToGenes)

    def fit_to_data(self, patientToGenes, min_cluster_size=0):
        self.patientToGenes = patientToGenes
        self.data, self.clusterToPatient = pmm_fit_to_data(patientToGenes, classes=self.classes, lam=self.lam, p_k=self.p_k,
                                                           min_cluster_size=min_cluster_size)
        return self.data, self.clusterToPatient


    def plot_clusters(self, title):
        plot_pmm_clusters(self.patientToGenes, self.clusterToPatient, self.num_components, title=title)


    def write_clusters(self, partition_file):
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Likelihood', self.data['Likelihood']])
            writer.writerow(['BIC', self.data['BIC']])
            writer.writerow(['NumComponents', self.data['Number']])
            writer.writerow(['Cluster', 'Lambda', 'Probability', 'Patients'])
            for k in self.clusterToPatient:
                if k != -1:
                    lam = self.data['Means'][k]
                    p_k = self.data['Probabilities'][k]
                else:
                    lam = None
                    p_k = None
                writer.writerow([k, lam, p_k] + list(self.clusterToPatient[k]))

    def compare_dna(self, dna_cohort_dict, do_KS=False):

        partition_stats_list = []

        sizes = [len(self.clusterToPatient[c]) for c in self.clusterToPatient]

        # partition by genes
        dna_cohorts = integrate_cohorts_sizes(dna_cohort_dict, sizes)

        pmm_cluster_list = []
        dna_cluster_list = []
        
        print "In partition stats Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()
        
        for i in range(len(self.classes)):
            partition_stats = collections.OrderedDict()
            partition_stats['Class'] = self.classes[i]
            partition_stats['Mean'] = self.lam[i]
            partition_stats['Probability'] = self.p_k[i]


            partition_stats['PMM_patients'] = self.clusterToPatient[self.classes[i]]
            partition_stats['DNA_patients'] = dna_cohorts[i]

            pmm_cluster_list.append(partition_stats['PMM_patients'])
            dna_cluster_list.append(partition_stats['DNA_patients'])
            
            dna_pmn = [len(self.patientToGenes[p]) for p in partition_stats['DNA_patients']]
            pmm_pmn = [len(self.patientToGenes[p]) for p in partition_stats['PMM_patients']]

            if do_KS:
                poisson_cdf.mu = self.lam[i]
                partition_stats['KS'] = stats.kstest(dna_pmn, poisson_cdf)

            #qq plot of the dna and then the poisson
            poisson_q = get_quantiles(dna_pmn, pmm_pmn)
            dna_q = get_quantiles(dna_pmn, dna_pmn)

            plot_pmm_clusters(self.patientToGenes, {'PMM': partition_stats['PMM_patients'], 'DNA': partition_stats['DNA_patients'] },
                              2, num_bins=100, title='DNA VS PMN')

            plt.figure()
            plt.plot(dna_q, poisson_q, 'bo')
            plt.plot([0, 100], [0,100], 'r-', label = 'y=x')
            plt.title('QQ for ' + str(self.classes[i]), fontsize=20)
            plt.xlabel('DNA_Q', fontsize=20)
            plt.ylabel('PMM_Q', fontsize=20)
            plt.legend()
            plt.show()

            partition_stats_list.append(partition_stats)

        if do_KS:
            self.data['KS_geom_mean'] = mex.prod([partition_stats['KS'][1] for partition_stats in partition_stats_list]) ** (1.0/ len(partition_stats_list))

            print "KS average is ", self.data['KS_geom_mean']
            
        self.data['CohenKappa'] = cohen_kappa(pmm_cluster_list, dna_cluster_list)


        return partition_stats_list



def cohen_kappa(cluster_list_1, cluster_list_2):
    # assume same categories each
    num_agree = 0
    prob_agree = 0
    total = len(set.union(*[set(c) for c in cluster_list_1]))
    
    num_classes = len(cluster_list_1)
    
    cluster_list_1 = [set(c) for c in cluster_list_1]
    cluster_list_2 = [set(c) for c in cluster_list_2]
    
    for k in range(num_classes):
        a = cluster_list_1[k]
        b = cluster_list_2[k]
        num_agree += len(a.intersection(b))
        prob_agree += (len(a) * len(b) * 1.0) / (total ** 2)
    

    obs_agree = num_agree * 1.0 / total
    
    ck = (obs_agree - prob_agree)/(1.0 - prob_agree)
    
    print "Number agreements ", num_agree
    print "Total ", total
    print "Prob agreements ", prob_agree
    print "Cohen kappa ", ck
    
    return ck
        
    
    




def poisson_cdf(x):
    if not hasattr(poisson_cdf, 'mu'):
        poisson_cdf.mu = 0
    print "X is ", x, "and mu is ", poisson_cdf.mu
    return poisson.cdf(x, poisson_cdf.mu)

def get_quantiles(test_dist, base_dist):
    return [stats.percentileofscore(base_dist, t) for t in test_dist]

def assign_missing(clusterToPatient, patientToGenes):
    if -1 not in clusterToPatient:
        print "No missing patients in clusters"
        return clusterToPatient
    missing_patients = clusterToPatient[-1]
    cluster_means = [(sum([len(patientToGenes[p]) for p in clusterToPatient[c]]) * 1.0 /len(clusterToPatient[c]), c) for c in clusterToPatient if c != -1]
    print cluster_means, cluster_means[0][0]
    for patient in missing_patients:
        num = len(patientToGenes[patient])
        correct_cluster = sorted(cluster_means, key=lambda entry: abs(num - entry[0]))[0][1]
        clusterToPatient[correct_cluster].add(patient)
    clusterToPatient.pop(-1)

    return clusterToPatient



def best_pmm(patientToGenes, num_components, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0,
             plot_clusters=True):

    data_record = []
    lls_record = []

    # Do normal
    first_data, lls = partition_pmm(patientToGenes, num_components,  max_iter=max_iter, min_cluster_size=min_cluster_size)

    data_record.append(first_data)
    lls_record.append(lls)

    # Do best rand init
    for i in range(rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    for i in range(far_rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, far_rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    combined_record = zip(data_record, lls_record)

    combined_record = sorted(combined_record, key=lambda entry: (-1 * entry[0]['Missing'], entry[0]['Likelihood']), reverse=True)

    data_record, lls_record = zip(*combined_record)

    best_data = data_record[0]

    if (best_data['Likelihood'] > first_data['Likelihood'] + 10):
        print "First data not best!"
        best_data['IsFirst'] = False
    else:
        best_data['IsFirst'] = True


    clusterToPatient = pmm_to_cluster(patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

    if plot_clusters:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components)

    plot_likelihoods(lls_record)

    return best_data, clusterToPatient
    # Return clusters


def pmm_to_cluster(patientToGenes, classes, lam, p_k):
    clusterToPatient = {}

    for k in classes:
        clusterToPatient[k] = set()

    clusterToPatient[-1] = set()


    for patient in patientToGenes:
        d = len(patientToGenes[patient])

        max_class = -1
        max_ll = -np.inf
        for k in classes:
            if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > -np.inf:
                if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > max_ll:
                    max_class = k
                    max_ll = (np.log(poisson(lam[k]).pmf(d)))


        clusterToPatient[max_class].add(patient)

    missing_clusters = set()
    for cluster in clusterToPatient:
        if not clusterToPatient[cluster]:
            print '**********NO PATIENTS IN CLUSTER ', lam[cluster], p_k[cluster]
            missing_clusters.add(cluster)
            #clusterToPatient[cluster].add('NO PATIENTS IN CLUSTER')
    for cluster in missing_clusters:
        clusterToPatient.pop(cluster)
            
    return clusterToPatient



def pmm_cross_validate(num_components, patientToGenes, num_folds, kf_random_state=None, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0):
    """
    :return: The average likelihood of the model when applied to a new test set, and its BIC
    """

    kf = KFold(len(patientToGenes), n_folds=num_folds, random_state=kf_random_state)

    lls = []
    missing_patients = []
    bics = []
    for train_index, test_index in kf:

        train_patientToGenes = dict([patientToGenes.items()[x] for x in train_index])
        test_patientToGenes = dict([patientToGenes.items()[x] for x in test_index])
        best_data, _ = best_pmm(train_patientToGenes, num_components, max_iter=max_iter, rand_num=rand_num,
                                               far_rand_num=far_rand_num, min_cluster_size=min_cluster_size)

        test_stats, test_cluster = pmm_fit_to_data(test_patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

        plot_pmm_clusters(test_patientToGenes, test_cluster, num_components, title='Test clusters size ' + str(num_components))

        lls.append(test_stats['Likelihood'])
        missing_patients.append(test_stats['Missing'])
        bics.append(test_stats['BIC'])

    return sum(lls) * 1.0/len(lls), sum(missing_patients) * 1.0 / len(missing_patients), sum(bics) * 1.0/ len(bics)





def pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=None, min_cluster_size=0):
    """
    :param patientToGenes:
    :param lam:
    :param p_k:
    :param data:
    :return: data, clusterToPatient
    """

    if not data:
        data = collections.OrderedDict()


    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    num_components = len(lam)

    ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])
    likelihood_sums = np.zeros(numCases)

    for i in range(numCases):
        likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

    # complete log likelihood

    ll = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

    clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    print "LL:", np.round(ll), "Missing patients: ", len(clusterToPatient[-1]) if -1 in clusterToPatient else 0

    data['Number'] = num_components
    data['OriginalNumber'] = num_components
    mp = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))

    data['Means'], data['Probabilities'] =  list(mp[0]), list(mp[1])   
    data['Likelihood'] = np.round(ll)
    data['Classes'] = classes
    data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0
    data['Merged'] = False
    data['MergeHistory'] = set()

    return data, clusterToPatient




def partition_pmm(patientToGenes, num_components, diff_thresh=0.01, num_bins=50, max_iter=100, by_iter=True,
                  rand_init=False, far_rand_init=False, do_plot=False, get_best=True, min_cluster_size=0,
                 verbose=True):


    # get the whole data distribution


    # D = [1,2,3,4,5, 100, 150, 200, 1000]
    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    data = collections.OrderedDict()

    # print "D is ", D

    # get the lambdas at equal-spaced intervals


    lam = [np.percentile(D, (i + 1) * 100.0 / (num_components + 1)) for i in range(num_components)]
    p_k = [1.0 / num_components for i in range(num_components)]
    classes = range(num_components)

    if rand_init:
        old_lam = lam
        old_p_k = p_k
        #random sample  in a range centered at the quartiles
        lam = [np.random.uniform(l - 0.5 * old_lam[0], l + 0.5 * old_lam[0]) for l in old_lam]
        rand_freq = [2**np.random.uniform(-1, 1) * pk for pk in old_p_k]
        p_k = list(np.array(rand_freq)/sum(rand_freq))
        classes = range(num_components)

    if far_rand_init:
        lam = [np.random.uniform(min(D), max(D)) for l in lam]
        rand_freq = [np.random.uniform(0, 1) for l in lam]
        p_k = list(np.array(rand_freq)/sum(rand_freq))

    if verbose:
        print "Initial Lambda is ", lam
        print "Initial p_k is", p_k

    data['Initial Means'] = np.round(lam,1)
    data['Initial p_k'] = np.round(p_k, 2)

    ll = -3e100
    num_iter = 0

    # stupid inital values
    p_k_d= np.zeros(num_components)
    lam_prev = np.zeros(num_components)
    p_k_prev = np.zeros(num_components)

    # for the best values
    ll_best = -np.inf
    p_k_best = None
    lam_best = None
    missing_best = numCases

    lls = []

    while 1:


        # We have the log-likelihood of data d and class k in matrix
        #            data 1 data 2 data 3
        # clsss 1   ll_11   ll_12
        # class 2
        ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])

        

        # Likelihood_sums: the total likelihood of each data, summed across class k
        likelihood_sums = np.zeros(numCases)

        for i in range(numCases):
            likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

            
        missing_new = len([x for x in likelihood_sums if x == 0])
        # complete log likelihood

        ll_new = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

        if num_iter == 0:
            data['Initial LL'] = np.round(ll_new)

        if verbose:
            print "ll_new is ", ll_new, "missing is ", missing_new


        if ll_new > ll_best or missing_new < missing_best:
            ll_best = ll_new
            p_k_best = p_k
            lam_best = lam
            missing_best = missing_new

        # When we break out of the loop, take previous value since it might have jumped out
        if (by_iter):
            if num_iter > max_iter:
                break
            elif abs(ll_new - ll) < diff_thresh:
                break
        else:
            if abs(ll_new - ll) < diff_thresh:

                p_k_d = p_k_d_prev
                lam = lam_prev
                p_k = p_k_prev

            break

        p_k_d_prev = p_k_d
        lam_prev = lam
        p_k_prev = p_k


        # Calculate p_k_d. This is p(data d | class k) * p(class k)/sum(p(data|class i) *p(class i);
        # i.e. prob of this class given this data

        p_k_d = np.zeros(ll_kd.shape)

        for i in range(numCases):
            # Use max class likelihood to divide all the likelihoods by
            max_val = np.amax(ll_kd, axis=0)[i]

            # sum the likekhoods for every class, make this the denominator of probability
            denom = sum([(np.exp(ll_kd[k][i] - max_val) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)])

            for k in range(num_components):
                p_k_d[k][i] = (np.exp(ll_kd[k][i] - max_val) / denom if ll_kd[k][i] > -np.inf else 0)
                # print "numerator is ", np.exp(ll_kd[k][i] - max), " prob is ", p_k_d[k][i]

        # print "p_k_d is ", p_k_d

        # sum probabilities of each data being each class over all data
        Z_k = p_k_d.sum(axis=1)


        # see derivation

        lam = [sum([p_k_d[k][i] * D[i] for i in range(numCases)]) * 1.0 / Z_k[k] for k in classes]
        p_k = Z_k * 1.0 / numCases

        p_k = p_k/p_k.sum()


        # print "New lambda is ", lam
        # print "New p_k is ", p_k


        ll = ll_new

        lls.append(ll)
        num_iter += 1



    if get_best:
        p_k = p_k_best
        lam = lam_best
        ll = ll_best





    data, clusterToPatient = pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=data, min_cluster_size=min_cluster_size)
    # plot patients in each cluster

    if do_plot:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100)


    # clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    #
    #
    #
    #
    # data['Number'] = num_components
    # data['Means'] = np.round(lam, 1)
    # data['Probabilities'] = np.round(p_k, 2)
    # data['Likelihood'] = np.round(ll)
    # data['Classes'] = classes
    # data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    # data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    # data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    # data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    # data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

    return data, lls



def sort_data_by_means(data):
    """ Sort in ascending order. Don't need to change cluster labels"""
    data_items = data.items()
    mean_indices = ((i, data['Means'][i]) for i in range(len(data['Means'])))
    mean_indices = sorted(mean_indices, key=lambda entry: min(entry[1]) if isinstance(entry[1], list)
                         else entry[1])
    
    conversion_array = [m[0] for m in mean_indices] # this should map to the correct index now. these are new clusters
    
    new_data = collections.OrderedDict()
    
    for key in data:
        value = data[key]
        if isinstance(value, np.ndarray):
            new_value = np.zeros(len(value))
            for i in range(len(conversion_array)):
                new_value[i] = value[conversion_array[i]]
            new_data[key] = new_value
        if isinstance(value, list):
            new_value = [value[conversion_array[i]] for i in range(len(conversion_array))]
            new_data[key] = new_value
            
        else:
            new_data[key] = value
    
    return new_data
    

def merge_clusters(data, clusterToPatient, patientToGenes,
                  missing_limit=0.5, min_cluster_size=30):
    """Merge adjacent clusters. Choosse to merge those clusters that
    are the most similar, as measured by the likelihood of one within
    another.
    missing_limit is the limit on number of patients that can't
    be explained by one cluster. Clusters will be sorted first
    by those who are below the minimum cluster size,
    less missing patients in their merging
    cluster, then by those that have the highest likelihood
    """
    # get the likelihood of each cluster rel. to other ones
    # only look at adjacent clusters! sort them
    
    data = sort_data_by_means(data)
    
    print "****************************************"
    print "Begin merging."
    # first go forward

    
    classes = data['Classes']
    p_k = data['Probabilities']
    lam = data['Means']
    
    
    all_list = []
    
    for i in range(len(lam) - 1):
        from_index, to_index = i, i + 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
        #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
            
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num > missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    # now go backwards
    for i in reversed(range(1, len(lam))):
        from_index, to_index = i, i - 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
                #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
        
        
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num < missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    
    # sort by the cluster that's below the min size, then byminimum missing, then by maximum likelihood ratio
    all_list = sorted(all_list, key=lambda entry: (entry[4], entry[3], entry[2]), reverse=True)
    
    print "Possible merged clusters is ", all_list
    print "Best cluster is ", all_list[0]
    

    (from_index, to_index), missing_num, ll_diffs_total, more_than_missing, cluster_too_small = all_list[0]

    # calculate the new AIC, BIC, make new cluster to patient, make new classes..new means? update probabilities
    
    # Record merge history
    new_data = data
    if 'MergeHistory' not in new_data:
        new_data['MergeHistory'] = set()
    
    new_data['MergeHistory'].add((str([lam[from_index], lam[to_index]]),
                  str([p_k[from_index], p_k[to_index]]),
                  (len(clusterToPatient[classes[from_index]]), len(clusterToPatient[classes[to_index]])),
                  missing_num, ll_diffs_total, ('Num classes befpre', len(classes), ('Cluster too small?', cluster_too_small))))
        
    new_clusterToPatient = clusterToPatient
    moved_patients = new_clusterToPatient[classes[from_index]]
    new_clusterToPatient[classes[to_index]] = new_clusterToPatient[classes[to_index]].union(moved_patients)
    new_clusterToPatient.pop(classes[from_index])

    
    print "MERGING the probs and likelihoods"
    if not isinstance(p_k[from_index], list):
        p_k[from_index] = [p_k[from_index]]
        lam[from_index] = [lam[from_index]]
    if not isinstance(p_k[to_index], list):
        p_k[to_index] = [p_k[to_index]]
        lam[to_index] = [lam[to_index]] 
    p_k[to_index].extend(p_k[from_index])
    lam[to_index].extend(lam[from_index])
    new_data['Probabilities'] = p_k
    new_data['Means'] = lam
    
    
    print "MERGING: HERE ARE OLD VALUES", new_data
    #remove all the old values
    new_data['Merged'] = True
    new_data['Number'] -= 1
    for key in new_data:
        value = new_data[key]
        if isinstance(value, np.ndarray):
            value = list(value)
            value = value[0: from_index] + value[from_index + 1 :]
            value = np.array(value)
            new_data[key] = value
        elif isinstance(value, list):
            value = value[0: from_index] + value[from_index + 1 :]
            new_data[key] = value

    print "New classe:", new_data['Classes'], "VS NEW KEYS", new_clusterToPatient.keys()
            
    # integrate the old patients to the new ones

    
    
    new_data['MinClusterSize'] = min(len(new_clusterToPatient[c]) for c in new_clusterToPatient)
    
    print "MERGING: HERE ARE NEW VALUES", new_data
    
    plot_pmm_clusters(patientToGenes, clusterToPatient, new_data['Number'], title='Merging')
    
    print "End merging."
    print "****************************************"    
    
    return new_data, new_clusterToPatient
 
    
#     data['Number'] = num_components
#     data['Means'], data['Probabilities'] = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))
#     data['Likelihood'] = np.round(ll)
#     data['Classes'] = classes
#     data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
#     data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
#     data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
#     data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
#     data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

def backward_selection(data, clusterToPatient, patientToGenes, min_cluster_size = 30,
                       max_components = 10):
    """Merge clusters until a criterion is satisfied. Missing patients are assumed to
    be assigned already.
    """
    

    merged_data = data
    merged_cluster = clusterToPatient
    
    while (merged_data['Number'] > max_components or merged_data['MinClusterSize'] < min_cluster_size):
        merged_data, merged_cluster = merge_clusters(merged_data, merged_cluster, patientToGenes,
                                                    min_cluster_size = min_cluster_size)
    
    return merged_data, merged_cluster
    







def plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100, title=None):
    D = [len(patientToGenes[p]) for p in patientToGenes]

    bins = range(0, max(list(D)), max(list(D))/num_bins)
    plt.figure()
    for cluster in clusterToPatient:
        plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    if not title:
        plt.title("Cluster size " + str(num_components), fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.show()

def plot_likelihoods(ll_record):
    plt.figure()
    for i in range(len(ll_record)):
        plt.plot(ll_record[i], label=str(i))
    plt.title("Log-likelihood change in EM", fontsize=20)
    plt.legend(loc=4)
    plt.show()

# If there are any patients that aren't assigned, i.e. in cluster -1
# Throw them out?
def load_patient_cohorts(partitionfile, patientToGenes, add_to_closest=True, delimiter='\t'):
    clusterToProp = {}

    with open(partitionfile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if (row[0] == 'Cluster'): break
        # reader = csv.DictReader(csvfile, delimiter=delimiter)
        # print "begun dict reader\n"
        for row in reader:
            c = eval(row[0])
            print c
            clusterToProp[c] = {}
            clusterToProp[c]['Mean'] = eval(row[1]) if row[1] else 0
            clusterToProp[c]['Probability'] = eval(row[2]) if row[2] else 0
            clusterToProp[c]['Patients'] = set(row[3:]) if row[3] else set()


    if -1 in clusterToProp:
        if add_to_closest:
            other_cs = clusterToProp.keys()
            other_cs.remove(-1)
            print "Removed ", clusterToProp[-1]
            for patient in clusterToProp[-1]:
                sims = [(abs(len(patientToGenes[patient]) - clusterToProp[c]['Mean']), c) for c in other_cs]
                sims = sorted(sims, key = lambda entry: entry[0])
                best_c = sims[0][1]
                clusterToProp[best_c]['Patients'].add(patient)
            print "completed"

        clusterToProp.pop(-1)

    sorted_clusters = sorted(clusterToProp.keys(), key = lambda entry: clusterToProp[entry]['Mean'])
    
    oldclusterToProp = clusterToProp.copy()
    clusterToProp = {}
    cohort_dict = {}
    
    for i in range(len(sorted_clusters)):
        cohort_dict[i] = oldclusterToProp[sorted_clusters[i]]['Patients']
        clusterToProp[i] = oldclusterToProp[sorted_clusters[i]]
    
    min_cohort = cohort_dict[0]
    
    
    
    

#     for c in clusterToProp:
#         cohort_dict[c] = clusterToProp[c]['Patients']
#     min_cohort = cohort_dict[sorted(clusterToProp.keys(), key=lambda entry: clusterToProp[entry]['Mean'])[0]]

    return cohort_dict, clusterToProp, min_cohort

# INDEX BY LOSSES
get_ipython().magic('matplotlib inline')
def run_partitions(mutationmatrix = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.m2',
        patientFile = None, #'/Users/jlu96/maf/new/OV_broad/shared_patients.plst',
        out_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-crossval.txt',
        partition_file = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.ppf',
        load_pmm_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM.txt',
        dna_pmm_comparison_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-dnacomp.txt',
        cluster_matrix = None, # '/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl-cluster.m2',
        min_cluster_size = 15,
        num_init = 9,
        minComp = 2,
        maxComp = 5,
        do_plot = True,
        do_gmm = False,
        do_dna = False,
        num_integrated = 4,
        do_kmeans = False,
        do_pmm = True,
        do_cross_val = False,
        do_pmm_dna = True,
        do_back_selection = True,
        write_cluster_matrices = True,
        rand_num = 3,
        far_rand_num = 3,
        kf_random_state = 1,
        kf_num_folds = 5,

        geneFile = None,
        minFreq = 0,
        dna_gene_file = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/DNADamageRepair_loss.txt',
       out_dir = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/',
        write_all_partitions = True):
    
    mutationmatrix_list = mutationmatrix.split('/')
    matrix_dir = '/'.join(mutationmatrix_list[:-1]) + '/'
    prefix = (mutationmatrix_list[-1]).split('.m2')[0]
    

    if not patientFile:
        patientFile = matrix_dir + 'shared_patients.plst'
        
    if not out_file:
        if do_cross_val:
            out_file = out_dir + prefix + '-PMM-crossval-kf' + str(kf_num_folds) + '.txt'
        else:
            out_file = out_dir + prefix + '-PMM-comparisons.txt'
    
    if not partition_file:
        partition_file = matrix_dir + prefix + '.ppf'
        
    
    if not load_pmm_file:
        load_pmm_file = out_dir + prefix + '-PMM.txt'
    
    if not dna_pmm_comparison_file:
        dna_pmm_comparison_file = out_dir + prefix + '-PMM-dnacomp.txt'
        
    if not cluster_matrix:
        cluster_matrix = matrix_dir + prefix + '-cluster.m2'

    
    numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

    p_gene_list = []

    with open(dna_gene_file, 'rU') as row_file:
        reader = csv.reader(row_file, delimiter='\t')
        for row in reader:
            p_gene_list.append(row[0])
        dna_cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))


    if do_kmeans:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_gene_kmeans(geneToCases, patientToGenes, p_gene_list, i, num_bins=50, title=None, do_plot=True))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_dna:
        cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))
        # Make new cohorts over this
        if num_integrated:
            cohort_dict = integrate_cohorts(cohort_dict, numCases, num_integrated)


        cohort_pairings = [(key, cohort_dict[key]) for key in cohort_dict]
        draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title='DNADamageGenes',
                        num_bins=100 if mutationmatrix[-9:] == 'cna-jl.m2' else 50)


    if do_gmm:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_GMM(patientToGenes, i, num_bins=50, title='GMM size ' + str(i), do_plot=do_plot))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_pmm:
        datas = []
        clusters = []

        partition_stats_list = []
        for num_components in np.arange(minComp, maxComp, 1):
            best_data, clusterToPatient = best_pmm(patientToGenes, num_components, rand_num=rand_num, far_rand_num=far_rand_num,
                                                   min_cluster_size=min_cluster_size)

            if do_back_selection:
                # assign the missing data
                clusterToPatient = assign_missing(clusterToPatient, patientToGenes)
                best_data, clusterToPatient = backward_selection(best_data, clusterToPatient, patientToGenes, min_cluster_size = min_cluster_size,
                       max_components = maxComp)
            
            if do_pmm_dna:
                print "cfirst lasses are ", best_data['Classes'], "clusterToPatient is ", clusterToPatient.keys()
                pmm = PMM(lam=best_data['Means'], p_k=best_data['Probabilities'], patientToGenes=patientToGenes,
                         data=best_data, clusterToPatient=clusterToPatient, classes=best_data['Classes'],
                          do_fit=False)

                partition_stats_list.extend(pmm.compare_dna(dna_cohort_dict))

                best_data = pmm.data


            if do_cross_val:
            #cross validate each of the components
                print "*******************************************************************************************************"
                print "BEGINNING CROSS VALIDATION for ", num_components
                print "*******************************************************************************************************"
                best_data['TestLL'], best_data['TestMissing'], best_data['TestBIC'] = pmm_cross_validate(num_components, patientToGenes,
                                                                                                         num_folds=kf_num_folds,
                                                                                                     kf_random_state=kf_random_state,
                                                                                   rand_num=rand_num, far_rand_num=far_rand_num,
                                                                                   min_cluster_size=min_cluster_size)
                best_data['TestFolds'] = kf_num_folds

                print "*******************************************************************************************************"
                print "EMDING CROSS VALIDATION  for ", num_components
                print "*******************************************************************************************************"

            datas.append(best_data)
            clusters.append(clusterToPatient)
            
            if write_all_partitions:
                with open(partition_file + str(num_components), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')

                    writer.writerow(['Likelihood', best_data['Likelihood']])
                    writer.writerow(['BIC', best_data['BIC']])
                    writer.writerow(['NumComponents', best_data['Number']])
                    writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
                    if 'Merged' in best_data and best_data['Merged']:
                        for k in range(len(clusterToPatient)):
                            lam = best_data['Means'][k]
                            p_k = best_data['Probabilities'][k]
                            writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
                    else:
                        for k in clusterToPatient:
                            if k != -1:
                                lam = best_data['Means'][k]
                                p_k = best_data['Probabilities'][k]
                            else:
                                lam = None
                                p_k = None
                            writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        # get the best BIC
        combined = zip(datas, clusters)
        if do_cross_val:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], np.round(entry[0]['TestMissing']), -1 * entry[0]['TestLL'], entry[0]['TestBIC'], entry[0]['BIC']))
        else:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], entry[0]['BIC']))

        datas, clusters = zip(*combined)




        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[-1].keys(), delimiter='\t', extrasaction='ignore')
            print datas
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


        best_data = datas[0]
        clusterToPatient = clusters[0]

        # code to parition by best clusters
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            writer.writerow(['Likelihood', best_data['Likelihood']])
            writer.writerow(['BIC', best_data['BIC']])
            writer.writerow(['NumComponents', best_data['Number']])
            writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
            if 'Merged' in best_data and best_data['Merged']:
                for k in range(len(clusterToPatient)):
                    lam = best_data['Means'][k]
                    p_k = best_data['Probabilities'][k]
                    writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
            else:
                for k in clusterToPatient:
                    if k != -1:
                        lam = best_data['Means'][k]
                        p_k = best_data['Probabilities'][k]
                    else:
                        lam = None
                        p_k = None
                    writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        if write_cluster_matrices:
            for cluster in clusterToPatient:
                with open(cluster_matrix + str(cluster), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    for patient in clusterToPatient[cluster]:
                        writer.writerow('\t'.join([patient] + list(patientToGenes[patient])))


        if do_pmm_dna:
            with open(dna_pmm_comparison_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=partition_stats_list[0].keys(), delimiter='\t')
                writer.writeheader()
                print "header written"
                for row in partition_stats_list:
                    writer.writerow(row)








# # This notebook will compute the bayes factors of mutual exclusivity and co-occurrence under the model:
# 
# \begin{equation}
# \begin{split}
# K = \frac{P(D | M_{A,B})}{P(D) | H_0} = \frac{\int \int \int P(\theta_A, \theta_B, \theta_{A,B}^M | M_{A,B}) P(D| \theta_A, \theta_B, \theta_{A,B}^M , M_{A,B}) d\theta_{A,B}^M d\theta_A d\theta_b }{\int \int \int P(\theta_A, \theta_B, \theta_{A,B}^M | H_0) P(D| \theta_A, \theta_B, \theta_{A,B}^M , H_0) d\theta_{A,B}^M d\theta_A d\theta_b }
# \end{split}
# \end{equation}
# 
# where we assume uniform priors on $\theta_A $ and $\theta_B$:
# $ P(\theta_A \in dp | M_{A,B}) = dp$ (since  $\int_0^1 P(\theta_A \in dp | M_{A,B}) dp = 1$)
# 
# and
# $ P(\theta_B \in dp | M_{A,B}) = dp$ (since  $\int_0^1 P(\theta_B \in dp | M_{A,B}) dp = 1$)
# 
# Then, because under $M_{A,B}$, $\theta_{A,B}^M$ must be greater than the value inferred by independence:
# 
# \[ P(\theta_{A,B}^M \in dp | \theta_A, \theta_B, M_{A,B}) = \begin{cases}
# 0 & p \leq \theta_A (1 - \theta_B) + \theta_B (1 - \theta_A) \\frac{dp}{\int_{\theta_A (1 - \theta_B) + \theta_B (1 - \theta_A)}^1 dp^*} & p > \theta_A (1 - \theta_B) + \theta_B (1 - \theta_A)\\end{cases}
# \]
# 
# Now, the likelihoods:
# 
# \begin{equation}
# P(D | \theta_A, \theta_B, \theta_{A,B}^M, M_{A,B}) = P(X_A | \theta_A, M_{A,B}) P(X_B | \theta_B, M_{A,B}) P(X_{A,B}^M | \theta_A, \theta_B, \theta_{A,B}^M, M_{A,B})
# \end{equation}
# 
# All the $X_E \sim Binom(\theta_E)$ for event $E$.
# 
# 
# So, $P(X_A | \theta_A, M_{A,B}) = {n \choose X_A} (\theta_A)^{X_A} (1 - \theta_A)^{n - X_A}$
# 
# $P(X_B | \theta_B, M_{A,B}) = {n \choose X_B} (\theta_B)^{X_B} (1 - \theta_B)^{n - X_B}$
# 
# $P(X_{A,B}^M | \theta_{A,B}^M, M_{A,B}) = {n \choose X_{A,B}^M} (\theta_{A,B}^M)^{X_{A,B}^M} (1 - \theta_{A,B}^M)^{n - X_{A,B}^M}$
# 

from scipy import special
# Let's calculate the top integral



n = 100
x_A = 20
x_B = 30
x_AB = 15

def make_model(n, x_A, x_B, x_AB):
    theta_A = pymc.DiscreteUniform('theta_A', lower=0, upper=1)
    theta_B = pymc.DiscreteUniform('theta_B', lower=0, upper=1)
    
    @pymc.stochastic(dtype=float)
    def theta_AB(value = 0.3, theta_A=0.1, theta_B=0.2):
        """Return log-likelihood of it"""
        if value < theta_A * theta_B or value > 1:
            return -np.inf
        else:
            return -np.log(1 - theta_B*theta_A)

    
    @pymc.deterministic
    def likelihood_A(theta=theta_A, n=n, x=x_A):
        return special.binom(n, x) * np.power(theta,x) * np.power((1- theta), n-x)

    @pymc.deterministic
    def likelihood_B(theta=theta_B, n=n, x=x_B):
        return special.binom(n, x) * np.power(theta,x) * np.power((1- theta), n-x)

    @pymc.deterministic
    def likelihood_AB(theta=theta_AB, n=n, x=x_AB):
        return special.binom(n, x) * np.power(theta,x) * np.power((1- theta), n-x)

# do you have to say the likelihood is a different variable?
# It's conditional. Has it been observed?
# partially, not completely. still needs theta_A, theta_B


M = pymc.MCMC(make_model(n, x_A, x_B, x_AB))


M.sample(200)
print M.value
plt.hist(M.trace('x')[:])
plt.hist(M.trace('x_A')[:])


import numpy as np
import pymc
import scipy.optimize as opt
import scipy.stats as stats

sigma = 1.0
tau = 1/ sigma**2

sigma0 = 3.0
tau0 = 1/sigma0**2

mu = pymc.Normal("mu", mu=0, tau=tau0)
x = pymc.Normal("x", mu=mu, tau=tau) # a bunch of samples for x
mcmc = pymc.MCMC([mu, x])
mcmc.sample(50000, 10000, 1) #1 is thinning. Results in autocorrelation , serial dependent
# First set burn=0, thin = 1, examine traces


x_samples = mcmc.trace("x")[:]
mu_samples = mcmc.trace("mu")[:]


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize=(5,5))
axes = fig.add_subplot(111)
axes.hist(x_samples, bins=50, normed=True, color="gray");
x_range = np.arange(-15, 15, 0.1)
x_precision = tau * tau0 / (tau + tau0)
axes.plot(x_range, stats.norm.pdf(x_range, 0, 1 / np.sqrt(x_precision)), color='k', linewidth=2);
fig.show()


x_pdf = stats.kde.gaussian_kde(x_samples)

def bayes_factor_sim(x_obs, x_pdf):
    return x_pdf.evaluate(x_obs) / stats.norm.pdf(x_obs, 0, sigma)

def bayes_factor_exact(x_obs):
    return np.sqrt(tau0 / (tau + tau0)) * np.exp(0.5 * tau**2 / (tau + tau0) * x_obs**2)


fig = plt.figure(figsize=(5,5))
axes = fig.add_subplot(111)
x_range = np.arange(0, 2, 0.1)
axes.plot(x_range, bayes_factor_sim(x_range, x_pdf), color="red", label="Simulated Bayes factor", linewidth=2)
axes.plot(x_range, bayes_factor_exact(x_range), color="blue", label="Exact Bayes factor")
axes.legend(loc=2)
fig.show()





import mutex as mex
import csv

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
closer_than_distance = 100000000
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.ppf9'
load_pmm_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/BRCA_wustl-cna-jl-PMM.txt'
dna_pmm_comparison_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/BRCA_wustl-cna-jl-PMM-dnacomp.txt'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

if COSMICFile:
    COSMICgenes = set()
    with open(COSMICFile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            COSMICgenes.add(*row)
    print "Number of COSMIC genes ", len(COSMICgenes)
    genes = (set(genes)).intersection(COSMICgenes)
    geneToCases = dict([g for g in geneToCases.items() if g[0] in genes])

print "Num COSMIC genes in this cancer", len(genes)
            


cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

print clusterToProp.keys()




# let's look at the smallest cluster
c0patients = clusterToProp[0]['Patients']
c1patients = clusterToProp[1]['Patients']

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[0]['Mean']
print list(c0genes)[0:30]
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 3
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-cpairs-min_cohort.txt'
test_minFreq = 3


genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[4])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


# Apparently some of the patients in the first cluster have these pairs mutated.
# HERPUD1loss	CBFBloss
# CDH11loss	MAFloss
# CBFBloss	MAFloss
# FANCAloss	MAFloss
# CDH1loss	HERPUD1loss
# CDH1loss	MAFloss
# HERPUD1loss	MAFloss
# CDH11loss	CYLDloss
# CDH11loss	FANCAloss
# FANCAloss	CBFBloss
# 
# HERPUD1loss	16	56932048	56944863
# CBFBloss	16	67029116	67101058
# CDH11loss	16	64943753	65126112
# MAFloss	16	79585843	79600714
# FANCAloss	16	89737549	89816657
# CDH1loss	16	68737225	68835548
# PTENloss 10	87863113	87971930
# CDKN1Bloss	12	12715058	12722371
# 
# Seems like the 10 MB filter didn't work.
# 
# 
# Is it possible that those patients just lost a big chunk of their DNA?
# 
# See cell below: seems like most of the guys with the co-occurrence are just mutated crazily.
# 
# When we look at the probabilities of the genes minus the highly mutated patients (>400),
# (i.e. the 47 smallest patients),  only one pair remains:
# FANCA Aand MAF loss.
# The others are simply not mutated there. Likely that we have some false positives?
# 
# 
# Let's look at the gene segments for BRCA_wustl:
# FANCA/CBFA2T3 are in the same
#  ZNF469_ZFPM1_MIR5189_ZC3H18_IL17C_CYBA_MVD_SNAI3_RNF166_CTU2_PIEZO1_MIR4722_CDT1_APRT_GALNS_TRAPPC2L_PABPN1L_CBFA2T3_ACSF3_LINC00304_CDH15_SLC22A31_ZNF778_ANKRD11_SPG7_RPL13_SNORD68_CPNE7_DPEP1_CHMP1A_SPATA33_CDK10_SPATA2L_VPS9D1_ZNF276_FANCA_SPIRE2_TCF25_MC1R_TUBB3_DEF8_CENPBD1_AFG3L1P_DBNDD1_GAS8_URAHP_PRDM7_TUBB8P7_FAM157Clos
# 
# 
# CDH11 and CBFB are in the same segment CDH11_LINC00922_RNA5SP428_CDH5_LINC00920_BEAN1_TK2_CKLF_CMTM1_CMTM2_CMTM3_CMTM4_DYNC1LI2_CCDC79_NAE1_CA7_PDP2_CDH16_RRAD_FAM96B_CES2_CES3_CES4A_RN7SL543P_CBFB_C16orf70_B3GNT9_TRADD_FBXL8_HSF4_NOL3_KIAA0895L_EXOC3L1_E2F4_ELMO3_MIR328_LRRC29_TMEM208_FHOD1_SLC9A5_PLEKHG4_KCTD19_RN7SKP118_LRRC36_TPPP3_ZDHHC1_HSD11B2_ATP6V0D1_AGRP_FAM65A_CTCF_RLTPR_ACD_PARD6A_ENKD1_C16orf86_GFOD2_RANBP10_TSNAXIP1_CENPT_THAP11_NUTF2_EDC4_NRN1L_PSKH1_CTRL_PSMB10_LCAT_SLC12A4_DPEP3_DPEP2_DDX28_NFATC3_ESRP2_PLA2G15_SLC7A6_SLC7A6OS_PRMT7_SMPD3_ZFP90_CDH3_CDH1_RNA5SP429loss
# 
# HERPUD loss:
# GNAO1_MIR3935_AMFR_NUDT21_OGFOD1_BBS2_MT4_MT3_MT2A_MT1L_MT1E_MT1M_MT1JP_MT1A_MT1DP_MT1B_MT1F_MT1G_MT1H_MT1X_NUP93_SLC12A3_HERPUD1_CETP_NLRC5_CPNE2_FAM192A_RSPRY1_ARL2BP_PLLP_CCL22_CX3CL1_CCL17_CIAPIN1_COQ9_POLR2C_DOK4_CCDC102Alos
# 
# 
# Let's do a 100MB filter.
# Then, let's segment.
# 
# Most genes are mutated only once.
# 
# Let's confirm that they actually lost a segment of their DNA.
# 
# There was a segment deletion at chr16:88525832-9035475. Even so, they're not contained in it...
# 
# They all happened to have deletions at that segment. See 2 cells down.

genes1 = ["CDH11loss", "CBFBloss", "FANCAloss", "CDH1loss", "CDH1loss", "HERPUD1loss", "CDH11loss", "CDH11loss", "FANCAloss"]
genes2 = ["MAFloss", "MAFloss", "MAFloss", "HERPUD1loss", "MAFloss", "MAFloss", "CYLDloss", "FANCAloss", "CBFBloss"]  

genepairs = zip(genes1, genes2)
cooccurPatients = {}
for patient in c0patients:
    for gene1, gene2 in genepairs:
        if gene1 in patientToGenes[patient] and gene2 in patientToGenes[patient]:
            if patient not in cooccurPatients:
                cooccurPatients[patient] = 0
            cooccurPatients[patient] += 1
            
cooccurPatients = dict(sorted(cooccurPatients.items(), key=lambda entry: len(patientToGenes[entry[0]]), reverse=True))
print len(cooccurPatients)
print [len(patientToGenes[p]) for p in cooccurPatients]
print cooccurPatients


import pandas as pd
df = pd.read_csv('/Users/jlu96/maf/new/BRCA_wustl/all_lesions.conf_99.txt', sep='\t')
indices = np.where(df['Unique Name'][:] == "Deletion Peak 32")[0]
print indices

keys = list(df.columns.values)
patientToKey = {}
for key in keys:
    if key[:12] in cooccurPatients:
        patientToKey[key[:12]] = key

delPatients = []
for patient in cooccurPatients:
    key = patientToKey[patient]
    if df[key][60] > 0:
        delPatients.append(patient)
print delPatients


# let's try to get the probabilities using the least mutated patients.
import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-cpairs-smallest-cohort.txt'
test_minFreq = 1

newc0patients = [p for p in c0patients if len(patientToGenes[p]) < 400]
print "Smallest cluster size", len(newc0patients)
newc0geneToCases = c0geneToCases.copy()
for g in newc0geneToCases:
    newc0geneToCases[g] = newc0geneToCases[g].intersection(set(newc0patients))

genepairs = bgbp.getgenepairs(newc0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, newc0patients)


print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)



# Now, let's take a look at the segment data.
# 
# BRCA_wustl was segmented. Here are the output from there:
# cna file size:  60246810
# time used to read in:  313.754554033
# Time used to convert genes:  73.9540939331
# 22771 genes were loaded
# 2005 genes were not loaded
# Time used to get gene positions:  6.44031000137
# Binning genes.
# Adjacent concordance threshold:  0.99
# Distance threshold:  10000000.0
# Number of position segments  1921
# Number of segments  1921
# Number of genes  24776
# Time used to bin genes:  214.709136009
# ****************
# WRITING SEGMENT MATRIX:
# -----------------------
# Total number of segment alterations:  962262
# 712 segments had equal numbers of gain and losses for  a sample were found, written as gain alterations
# 10109  had some gains and some losses, but were overridden by majority
# Average majority ratio:  0.778375217298
# Standard Dev of majority ratios:  0.137182333658
# *****************
# ********************
# WRITING SEG2GENES:
# --------------------
# Number of genes  22771
# Number of segments  1921
# Average concordance:  0.988522560372
# Standard deviation of concordance:  0.0106270392535
# Average nonzero concordances:  0.973493456503
# Standard deviation of nonzero concordances:  0.0250897363908
# Segments matrix written to:  ./BRCA_wustl-seg-gl.m2
# Gene to segments written to:  ./BRCA_wustl-seg-gl.gene2seg
# 

#Let's now look at the patient mutation distribution of BRCA, segmented. Is it broad? Does it make sense to partition over this?
get_ipython().magic('matplotlib inline')

import mutex as mex

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2'
numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)
mex.graph_mutation_distribution(numCases, genes, geneToCases, filename='Breast Cancer', top_percentile=10, bins=100)
mex.graph_patient_distribution(numGenes, patients, patientToGenes, filename='Patient Mutation Distribution', top_percentile=10, bins=100)


# A partition over this space looks good! Let's partition it.

run_partitions(mutationmatrix='/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2', min_cluster_size=70,
              do_cross_val=False, do_pmm_dna=False, maxComp=10)


# when we're ready, let's look at the cooccurring pairs here. EDIT PARTITION FILE

import mutex as mex
import csv

patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
closer_than_distance = 100000000
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.ppf9'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

print clusterToProp.keys()


c0patients = clusterToProp[0]['Patients']
c1patients = clusterToProp[1]['Patients']

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[0]['Mean']
print list(c0genes)[0:30]
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 3
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-sega-jl-cpairs-min_cohort.txt'
test_minFreq = 3


genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[0])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


__author__ = 'jlu96'
import mutex as mex
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from scipy.stats import poisson
from scipy import stats
import collections
import os

def partition_EM(patientToGenes, k):
    """
    :param geneToCases:
    :param patientToGenes:
    :param k: Number of partitions
    :return: cohort_list
    """

    # partition the patients, and intersect the geneToCases
    return



def partition_gene(patientToGenes, genes):
    """
    :param geneToCases:
    :param patientToGenes:
    :param genes:
    :return: cohorts by each gene. Size 2^(#genes)
    """

    cohorts = [patientToGenes.keys()]
    for gene in genes:
        new_cohorts = []
        for cohort in cohorts:
            new_cohort_1 = [patient for patient in patientToGenes if gene not in patientToGenes[patient]]
            if new_cohort_1:
                new_cohorts.append(new_cohort_1)
            new_cohort_2 = list(set(cohort).difference(set(new_cohort_1)))
            if new_cohort_2:
                new_cohorts.append(new_cohort_2)
        cohorts = new_cohorts
    # print genes
    # print cohorts

    return cohorts

def partition_gene_list(patientToGenes, genes, binary=True):
    """
    :param patientToGenes:
    :param genes:
    :return: The cohorts, ordered from least to greatest in number of those genes they have.
    If binary = True, return just those with, those without.

    """



    gene_set = set(genes)
    cohort_dict = {}

    for patient in patientToGenes:
        num = len(set.intersection(gene_set, patientToGenes[patient]))

        # just 0 and 1
        if binary:
            if num > 0:
                num = 1

        if num not in cohort_dict:
            cohort_dict[num] = []
        cohort_dict[num].append(patient)


    return cohort_dict


def get_patients_gene_mut_num(patients, genes, patientToGenes):
    return [set.intersection(patientToGenes[p], genes) for p in patients]

def integrate_cohorts(cohort_dict, numCases, num_integrated):
    cohorts_int = {}
    start_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > numCases/num_integrated):
            cohorts_int[start_index] = new_cohort
            start_index = i+1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[start_index] = new_cohort

    return cohorts_int

def integrate_cohorts_sizes(cohort_dict, sizes):
    cohorts_int = {}
    size_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > sizes[size_index]):
            cohorts_int[size_index] = new_cohort
            size_index += 1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[size_index] = new_cohort

    return cohorts_int


def draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title=None, num_bins=50):
    # LEFT OF HERE, JLU. Finish this, then above. Make plots in parallel, compare.
    # Work with: TP53? Others?

    numGenes = len(geneToCases.keys())
    numCohorts = len(cohort_pairings)

    cohort_frequencies = [[len(patientToGenes[case]) for case in cohort_pair[1]] for cohort_pair in cohort_pairings]
    cohort_names = [cohort_pair[0] for cohort_pair in cohort_pairings]

    draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=title, num_bins=num_bins)


def draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=None, num_bins=50):

    numCohorts = len(cohort_frequencies)
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)

    plt.figure()


    for i in range(len(cohort_frequencies)):
        plt.hist(cohort_frequencies[i], bins, alpha=1.0/numCohorts, label=str(cohort_names[i]))


    plt.title(title, fontsize=20)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    plt.show()

def norm(x, height, center, std):
    return(height*np.exp(-(x - center)**2/(2*std**2)))



def partition_GMM(patientToGenes, num_components, num_bins, title=None, do_plot=True):
    g = mixture.GMM(n_components=num_components)
    mut_num_list = [len(patientToGenes[p]) for p in patientToGenes]
    obs = np.array([[entry] for entry in mut_num_list])
    g.fit(obs)

    print "***********************************"
    print "COMPONENTS: ", num_components
    print "Weights: " + str(np.round(g.weights_,2))
    print "Means: " + str(np.round(g.means_,2))
    print "Covariates: " + str(np.round(g.covars_,2))

    print "Total log probability: " + str(sum(g.score(obs)))
    print "AIC: " + str(g.aic(obs))
    print "BIC: ", g.bic(obs)

    score, respon = g.score_samples(obs)

    for i in range(num_components):
        print "Model ", np.round(g.means_, 2)[i], " explains ", np.round(len([in_w for in_w in respon if in_w[i] == max(in_w)])) * 1.0 /len(respon)


    # Simulate gaussians
    # sim_samples = g.sample(len(patientToGenes))
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
    histogram = np.histogram([len(patientToGenes[p]) for p in patientToGenes], bins=bins)

    # get the scale of the gaussians from the biggest one
    # max_comp = g.weights_.index(max(g.weights_))
    # max_mean = g.means_[max_comp]

    which_bins = [[bin for bin in bins if bin > mean][0] for mean in g.means_]
    print which_bins
    print bins
    print histogram
    print bins.index(which_bins[0]) - 1
    bin_heights = [histogram[0][bins.index(which_bin) - 1] for which_bin in which_bins]
    # max_height = max(histogram)

    if do_plot:
        plt.figure()
        plt.hist([len(patientToGenes[p]) for p in patientToGenes], bins=bins)
        for i in range(num_components):
            X = np.arange(0, max(mut_num_list), 1)
            Y = norm(X, bin_heights[i], g.means_[i], np.sqrt(g.covars_[i]))
            plt.plot(X, Y, label=str(np.round(g.weights_[i], 3)), linewidth=5)
        plt.title("GMM size " + str(num_components), fontsize=20)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.show()
        # draw_partitions(patientToGenes, ['Original', 'Simulated'], [[len(patientToGenes[p]) for p in patientToGenes], sim_samples],
        #                 num_bins=num_bins, title=title)

    data = {}
    data['Components'] = num_components
    data['Weights'] = np.round(g.weights_,2)
    data['Means'] = np.round(g.means_,2)
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    data["AIC"] = g.aic(obs)
    data["BIC"] = g.bic(obs)
    data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data

def partition_gene_kmeans(geneToCases, patientToGenes, gene_list, num_components, num_bins, title=None, do_plot=True):

    # get gene index mapping
    giv = getgiv(geneToCases.keys(), gene_list)

    # convert patients into vectors
    patientToVector = getpatientToVector(patientToGenes, giv)

    vectors = patientToVector.values()

    print vectors[0]
    print "Length of vectors is ", len(vectors[0])

    km = KMeans(num_components)

    km.fit(vectors)

    clusterToPatient = {}

    for patient in patientToVector:
        cluster = km.predict(patientToVector[patient])[0]
        if cluster not in clusterToPatient:
            clusterToPatient[cluster] = set()
        clusterToPatient[cluster].add(patient)

    # plot patients in each cluster


    if do_plot:
        bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
        plt.figure()
        for cluster in clusterToPatient:
            plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.title("Kmeans size " + str(num_components), fontsize=20)
        plt.show()



    data = {}
    data['Score'] = km.score(vectors)
    data['Number'] = num_components
    data['% Explained'] = np.round([100 * len(clusterToPatient[cluster]) * 1.0 / len(patientToGenes) for cluster in clusterToPatient], 2)
    data['Vector size'] = len(vectors[0])
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    # data["AIC"] = g.aic(obs)
    # data["BIC"] = g.bic(obs)
    # data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data


def getgiv(all_genes, gene_list):
    """
    :param all_genes:
    :param gene_list:
    :return: A list of the genes in common, the gene_index_vector.
    """
    giv = list(set(all_genes).intersection(set(gene_list)))

    return giv



def getpatientToVector(patientToGenes, gene_index_vector):
    patientToVector = {}
    for patient in patientToGenes:
        patient_genes = patientToGenes[patient]
        patientToVector[patient] = []
        for gene in gene_index_vector:
            patientToVector[patient].append(1 if gene in patient_genes else 0)

    return patientToVector


def get_cluster_gTC_pTG(geneToCases, patientToGenes, patients):
    new_pTG = dict([c for c in patientToGenes.items() if c[0] in patients])
    new_genes = set.union(*new_pTG.values())
    new_gTC = dict([g for g in geneToCases.items() if g[0] in new_genes])
    for g in new_gTC:
        new_gTC[g] = new_gTC[g].intersection(patients)
    
    for g in new_genes:
        if g in new_gTC and not new_gTC[g]:
            new_gTC.pop(g)
    
    new_genes = new_genes.intersection(set(new_gTC.keys()))
    
    return new_genes, new_gTC, new_pTG










# 3/12/16-Jlu


class PMM:

    def __init__(self, filename=None, delimiter='\t', lam=None, p_k=None, classes=None, patientToGenes=None,
                data = None, clusterToPatient = None, do_fit=True):

        if filename:
            with open(filename, 'rU') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                row = reader.next()
                print row
                self.lam = eval(row['Means'])
                self.p_k = eval(row['Probabilities'])
                self.classes = eval(row['Classes']) if 'Classes' in row else range(len(self.lam))
                self.num_components = len(self.classes)
        else:
            self.lam = lam
            self.p_k = p_k
            self.classes = classes
            if not classes:
                self.classes = range(len(self.lam))
            self.num_components = len(self.classes)


        self.data = data
        self.clusterToPatient = clusterToPatient
        print "Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()

        self.patientToGenes = patientToGenes

        if patientToGenes and do_fit:
            self.fit_to_data(patientToGenes)

    def fit_to_data(self, patientToGenes, min_cluster_size=0):
        self.patientToGenes = patientToGenes
        self.data, self.clusterToPatient = pmm_fit_to_data(patientToGenes, classes=self.classes, lam=self.lam, p_k=self.p_k,
                                                           min_cluster_size=min_cluster_size)
        return self.data, self.clusterToPatient


    def plot_clusters(self, title):
        plot_pmm_clusters(self.patientToGenes, self.clusterToPatient, self.num_components, title=title)


    def write_clusters(self, partition_file):
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Likelihood', self.data['Likelihood']])
            writer.writerow(['BIC', self.data['BIC']])
            writer.writerow(['NumComponents', self.data['Number']])
            writer.writerow(['Cluster', 'Lambda', 'Probability', 'Patients'])
            for k in self.clusterToPatient:
                if k != -1:
                    lam = self.data['Means'][k]
                    p_k = self.data['Probabilities'][k]
                else:
                    lam = None
                    p_k = None
                writer.writerow([k, lam, p_k] + list(self.clusterToPatient[k]))

    def compare_dna(self, dna_cohort_dict, do_KS=False):

        partition_stats_list = []

        sizes = [len(self.clusterToPatient[c]) for c in self.clusterToPatient]

        # partition by genes
        dna_cohorts = integrate_cohorts_sizes(dna_cohort_dict, sizes)

        pmm_cluster_list = []
        dna_cluster_list = []
        
        print "In partition stats Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()
        
        for i in range(len(self.classes)):
            partition_stats = collections.OrderedDict()
            partition_stats['Class'] = self.classes[i]
            partition_stats['Mean'] = self.lam[i]
            partition_stats['Probability'] = self.p_k[i]


            partition_stats['PMM_patients'] = self.clusterToPatient[self.classes[i]]
            partition_stats['DNA_patients'] = dna_cohorts[i]

            pmm_cluster_list.append(partition_stats['PMM_patients'])
            dna_cluster_list.append(partition_stats['DNA_patients'])
            
            dna_pmn = [len(self.patientToGenes[p]) for p in partition_stats['DNA_patients']]
            pmm_pmn = [len(self.patientToGenes[p]) for p in partition_stats['PMM_patients']]

            if do_KS:
                poisson_cdf.mu = self.lam[i]
                partition_stats['KS'] = stats.kstest(dna_pmn, poisson_cdf)

            #qq plot of the dna and then the poisson
            poisson_q = get_quantiles(dna_pmn, pmm_pmn)
            dna_q = get_quantiles(dna_pmn, dna_pmn)

            plot_pmm_clusters(self.patientToGenes, {'PMM': partition_stats['PMM_patients'], 'DNA': partition_stats['DNA_patients'] },
                              2, num_bins=100, title='DNA VS PMN')

            plt.figure()
            plt.plot(dna_q, poisson_q, 'bo')
            plt.plot([0, 100], [0,100], 'r-', label = 'y=x')
            plt.title('QQ for ' + str(self.classes[i]), fontsize=20)
            plt.xlabel('DNA_Q', fontsize=20)
            plt.ylabel('PMM_Q', fontsize=20)
            plt.legend()
            plt.show()

            partition_stats_list.append(partition_stats)

        if do_KS:
            self.data['KS_geom_mean'] = mex.prod([partition_stats['KS'][1] for partition_stats in partition_stats_list]) ** (1.0/ len(partition_stats_list))

            print "KS average is ", self.data['KS_geom_mean']
            
        self.data['CohenKappa'] = cohen_kappa(pmm_cluster_list, dna_cluster_list)


        return partition_stats_list



def cohen_kappa(cluster_list_1, cluster_list_2):
    # assume same categories each
    num_agree = 0
    prob_agree = 0
    total = len(set.union(*[set(c) for c in cluster_list_1]))
    
    num_classes = len(cluster_list_1)
    
    cluster_list_1 = [set(c) for c in cluster_list_1]
    cluster_list_2 = [set(c) for c in cluster_list_2]
    
    for k in range(num_classes):
        a = cluster_list_1[k]
        b = cluster_list_2[k]
        num_agree += len(a.intersection(b))
        prob_agree += (len(a) * len(b) * 1.0) / (total ** 2)
    

    obs_agree = num_agree * 1.0 / total
    
    ck = (obs_agree - prob_agree)/(1.0 - prob_agree)
    
    print "Number agreements ", num_agree
    print "Total ", total
    print "Prob agreements ", prob_agree
    print "Cohen kappa ", ck
    
    return ck
        
    
    




def poisson_cdf(x):
    if not hasattr(poisson_cdf, 'mu'):
        poisson_cdf.mu = 0
    print "X is ", x, "and mu is ", poisson_cdf.mu
    return poisson.cdf(x, poisson_cdf.mu)

def get_quantiles(test_dist, base_dist):
    return [stats.percentileofscore(base_dist, t) for t in test_dist]

def assign_missing(clusterToPatient, patientToGenes):
    if -1 not in clusterToPatient:
        print "No missing patients in clusters"
        return clusterToPatient
    missing_patients = clusterToPatient[-1]
    cluster_means = [(sum([len(patientToGenes[p]) for p in clusterToPatient[c]]) * 1.0 /len(clusterToPatient[c]), c) for c in clusterToPatient if c != -1]
    print cluster_means, cluster_means[0][0]
    for patient in missing_patients:
        num = len(patientToGenes[patient])
        correct_cluster = sorted(cluster_means, key=lambda entry: abs(num - entry[0]))[0][1]
        clusterToPatient[correct_cluster].add(patient)
    clusterToPatient.pop(-1)

    return clusterToPatient



def best_pmm(patientToGenes, num_components, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0,
             plot_clusters=True):

    data_record = []
    lls_record = []

    # Do normal
    first_data, lls = partition_pmm(patientToGenes, num_components,  max_iter=max_iter, min_cluster_size=min_cluster_size)

    data_record.append(first_data)
    lls_record.append(lls)

    # Do best rand init
    for i in range(rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    for i in range(far_rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, far_rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    combined_record = zip(data_record, lls_record)

    combined_record = sorted(combined_record, key=lambda entry: (-1 * entry[0]['Missing'], entry[0]['Likelihood']), reverse=True)

    data_record, lls_record = zip(*combined_record)

    best_data = data_record[0]

    if (best_data['Likelihood'] > first_data['Likelihood'] + 10):
        print "First data not best!"
        best_data['IsFirst'] = False
    else:
        best_data['IsFirst'] = True


    clusterToPatient = pmm_to_cluster(patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

    if plot_clusters:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components)

    plot_likelihoods(lls_record)

    return best_data, clusterToPatient
    # Return clusters


def pmm_to_cluster(patientToGenes, classes, lam, p_k):
    clusterToPatient = {}

    for k in classes:
        clusterToPatient[k] = set()

    clusterToPatient[-1] = set()


    for patient in patientToGenes:
        d = len(patientToGenes[patient])

        max_class = -1
        max_ll = -np.inf
        for k in classes:
            if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > -np.inf:
                if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > max_ll:
                    max_class = k
                    max_ll = (np.log(poisson(lam[k]).pmf(d)))


        clusterToPatient[max_class].add(patient)

    missing_clusters = set()
    for cluster in clusterToPatient:
        if not clusterToPatient[cluster]:
            print '**********NO PATIENTS IN CLUSTER ', lam[cluster], p_k[cluster]
            missing_clusters.add(cluster)
            #clusterToPatient[cluster].add('NO PATIENTS IN CLUSTER')
    for cluster in missing_clusters:
        clusterToPatient.pop(cluster)
            
    return clusterToPatient



def pmm_cross_validate(num_components, patientToGenes, num_folds, kf_random_state=None, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0):
    """
    :return: The average likelihood of the model when applied to a new test set, and its BIC
    """

    kf = KFold(len(patientToGenes), n_folds=num_folds, random_state=kf_random_state)

    lls = []
    missing_patients = []
    bics = []
    for train_index, test_index in kf:

        train_patientToGenes = dict([patientToGenes.items()[x] for x in train_index])
        test_patientToGenes = dict([patientToGenes.items()[x] for x in test_index])
        best_data, _ = best_pmm(train_patientToGenes, num_components, max_iter=max_iter, rand_num=rand_num,
                                               far_rand_num=far_rand_num, min_cluster_size=min_cluster_size)

        test_stats, test_cluster = pmm_fit_to_data(test_patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

        plot_pmm_clusters(test_patientToGenes, test_cluster, num_components, title='Test clusters size ' + str(num_components))

        lls.append(test_stats['Likelihood'])
        missing_patients.append(test_stats['Missing'])
        bics.append(test_stats['BIC'])

    return sum(lls) * 1.0/len(lls), sum(missing_patients) * 1.0 / len(missing_patients), sum(bics) * 1.0/ len(bics)





def pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=None, min_cluster_size=0):
    """
    :param patientToGenes:
    :param lam:
    :param p_k:
    :param data:
    :return: data, clusterToPatient
    """

    if not data:
        data = collections.OrderedDict()


    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    num_components = len(lam)

    ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])
    likelihood_sums = np.zeros(numCases)

    for i in range(numCases):
        likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

    # complete log likelihood

    ll = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

    clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    print "LL:", np.round(ll), "Missing patients: ", len(clusterToPatient[-1]) if -1 in clusterToPatient else 0

    data['Number'] = num_components
    data['OriginalNumber'] = num_components
    mp = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))

    data['Means'], data['Probabilities'] =  list(mp[0]), list(mp[1])   
    data['Likelihood'] = np.round(ll)
    data['Classes'] = classes
    data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0
    data['Merged'] = False
    data['MergeHistory'] = set()

    return data, clusterToPatient




def partition_pmm(patientToGenes, num_components, diff_thresh=0.01, num_bins=50, max_iter=100, by_iter=True,
                  rand_init=False, far_rand_init=False, do_plot=False, get_best=True, min_cluster_size=0,
                 verbose=True):


    # get the whole data distribution


    # D = [1,2,3,4,5, 100, 150, 200, 1000]
    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    data = collections.OrderedDict()

    # print "D is ", D

    # get the lambdas at equal-spaced intervals


    lam = [np.percentile(D, (i + 1) * 100.0 / (num_components + 1)) for i in range(num_components)]
    p_k = [1.0 / num_components for i in range(num_components)]
    classes = range(num_components)

    if rand_init:
        old_lam = lam
        old_p_k = p_k
        #random sample  in a range centered at the quartiles
        lam = [np.random.uniform(l - 0.5 * old_lam[0], l + 0.5 * old_lam[0]) for l in old_lam]
        rand_freq = [2**np.random.uniform(-1, 1) * pk for pk in old_p_k]
        p_k = list(np.array(rand_freq)/sum(rand_freq))
        classes = range(num_components)

    if far_rand_init:
        lam = [np.random.uniform(min(D), max(D)) for l in lam]
        rand_freq = [np.random.uniform(0, 1) for l in lam]
        p_k = list(np.array(rand_freq)/sum(rand_freq))

    if verbose:
        print "Initial Lambda is ", lam
        print "Initial p_k is", p_k

    data['Initial Means'] = np.round(lam,1)
    data['Initial p_k'] = np.round(p_k, 2)

    ll = -3e100
    num_iter = 0

    # stupid inital values
    p_k_d= np.zeros(num_components)
    lam_prev = np.zeros(num_components)
    p_k_prev = np.zeros(num_components)

    # for the best values
    ll_best = -np.inf
    p_k_best = None
    lam_best = None
    missing_best = numCases

    lls = []

    while 1:


        # We have the log-likelihood of data d and class k in matrix
        #            data 1 data 2 data 3
        # clsss 1   ll_11   ll_12
        # class 2
        ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])

        

        # Likelihood_sums: the total likelihood of each data, summed across class k
        likelihood_sums = np.zeros(numCases)

        for i in range(numCases):
            likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

            
        missing_new = len([x for x in likelihood_sums if x == 0])
        # complete log likelihood

        ll_new = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

        if num_iter == 0:
            data['Initial LL'] = np.round(ll_new)

        if verbose:
            print "ll_new is ", ll_new, "missing is ", missing_new


        if ll_new > ll_best or missing_new < missing_best:
            ll_best = ll_new
            p_k_best = p_k
            lam_best = lam
            missing_best = missing_new

        # When we break out of the loop, take previous value since it might have jumped out
        if (by_iter):
            if num_iter > max_iter:
                break
            elif abs(ll_new - ll) < diff_thresh:
                break
        else:
            if abs(ll_new - ll) < diff_thresh:

                p_k_d = p_k_d_prev
                lam = lam_prev
                p_k = p_k_prev

            break

        p_k_d_prev = p_k_d
        lam_prev = lam
        p_k_prev = p_k


        # Calculate p_k_d. This is p(data d | class k) * p(class k)/sum(p(data|class i) *p(class i);
        # i.e. prob of this class given this data

        p_k_d = np.zeros(ll_kd.shape)

        for i in range(numCases):
            # Use max class likelihood to divide all the likelihoods by
            max_val = np.amax(ll_kd, axis=0)[i]

            # sum the likekhoods for every class, make this the denominator of probability
            denom = sum([(np.exp(ll_kd[k][i] - max_val) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)])

            for k in range(num_components):
                p_k_d[k][i] = (np.exp(ll_kd[k][i] - max_val) / denom if ll_kd[k][i] > -np.inf else 0)
                # print "numerator is ", np.exp(ll_kd[k][i] - max), " prob is ", p_k_d[k][i]

        # print "p_k_d is ", p_k_d

        # sum probabilities of each data being each class over all data
        Z_k = p_k_d.sum(axis=1)


        # see derivation

        lam = [sum([p_k_d[k][i] * D[i] for i in range(numCases)]) * 1.0 / Z_k[k] for k in classes]
        p_k = Z_k * 1.0 / numCases

        p_k = p_k/p_k.sum()


        # print "New lambda is ", lam
        # print "New p_k is ", p_k


        ll = ll_new

        lls.append(ll)
        num_iter += 1



    if get_best:
        p_k = p_k_best
        lam = lam_best
        ll = ll_best





    data, clusterToPatient = pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=data, min_cluster_size=min_cluster_size)
    # plot patients in each cluster

    if do_plot:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100)


    # clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    #
    #
    #
    #
    # data['Number'] = num_components
    # data['Means'] = np.round(lam, 1)
    # data['Probabilities'] = np.round(p_k, 2)
    # data['Likelihood'] = np.round(ll)
    # data['Classes'] = classes
    # data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    # data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    # data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    # data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    # data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

    return data, lls



def sort_data_by_means(data):
    """ Sort in ascending order. Don't need to change cluster labels"""
    data_items = data.items()
    mean_indices = ((i, data['Means'][i]) for i in range(len(data['Means'])))
    mean_indices = sorted(mean_indices, key=lambda entry: min(entry[1]) if isinstance(entry[1], list)
                         else entry[1])
    
    conversion_array = [m[0] for m in mean_indices] # this should map to the correct index now. these are new clusters
    
    new_data = collections.OrderedDict()
    
    for key in data:
        value = data[key]
        if isinstance(value, np.ndarray):
            new_value = np.zeros(len(value))
            for i in range(len(conversion_array)):
                new_value[i] = value[conversion_array[i]]
            new_data[key] = new_value
        if isinstance(value, list):
            new_value = [value[conversion_array[i]] for i in range(len(conversion_array))]
            new_data[key] = new_value
            
        else:
            new_data[key] = value
    
    return new_data
    

def merge_clusters(data, clusterToPatient, patientToGenes,
                  missing_limit=0.5, min_cluster_size=30):
    """Merge adjacent clusters. Choosse to merge those clusters that
    are the most similar, as measured by the likelihood of one within
    another.
    missing_limit is the limit on number of patients that can't
    be explained by one cluster. Clusters will be sorted first
    by those who are below the minimum cluster size,
    less missing patients in their merging
    cluster, then by those that have the highest likelihood
    """
    # get the likelihood of each cluster rel. to other ones
    # only look at adjacent clusters! sort them
    
    data = sort_data_by_means(data)
    
    print "****************************************"
    print "Begin merging."
    # first go forward

    
    classes = data['Classes']
    p_k = data['Probabilities']
    lam = data['Means']
    
    
    all_list = []
    
    for i in range(len(lam) - 1):
        from_index, to_index = i, i + 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
        #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
            
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num > missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    # now go backwards
    for i in reversed(range(1, len(lam))):
        from_index, to_index = i, i - 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
                #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
        
        
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num < missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    
    # sort by the cluster that's below the min size, then byminimum missing, then by maximum likelihood ratio
    all_list = sorted(all_list, key=lambda entry: (entry[4], entry[3], entry[2]), reverse=True)
    
    print "Possible merged clusters is ", all_list
    print "Best cluster is ", all_list[0]
    

    (from_index, to_index), missing_num, ll_diffs_total, more_than_missing, cluster_too_small = all_list[0]

    # calculate the new AIC, BIC, make new cluster to patient, make new classes..new means? update probabilities
    
    # Record merge history
    new_data = data
    if 'MergeHistory' not in new_data:
        new_data['MergeHistory'] = set()
    
    new_data['MergeHistory'].add((str([lam[from_index], lam[to_index]]),
                  str([p_k[from_index], p_k[to_index]]),
                  (len(clusterToPatient[classes[from_index]]), len(clusterToPatient[classes[to_index]])),
                  missing_num, ll_diffs_total, ('Num classes befpre', len(classes), ('Cluster too small?', cluster_too_small))))
        
    new_clusterToPatient = clusterToPatient
    moved_patients = new_clusterToPatient[classes[from_index]]
    new_clusterToPatient[classes[to_index]] = new_clusterToPatient[classes[to_index]].union(moved_patients)
    new_clusterToPatient.pop(classes[from_index])

    
    print "MERGING the probs and likelihoods"
    if not isinstance(p_k[from_index], list):
        p_k[from_index] = [p_k[from_index]]
        lam[from_index] = [lam[from_index]]
    if not isinstance(p_k[to_index], list):
        p_k[to_index] = [p_k[to_index]]
        lam[to_index] = [lam[to_index]] 
    p_k[to_index].extend(p_k[from_index])
    lam[to_index].extend(lam[from_index])
    new_data['Probabilities'] = p_k
    new_data['Means'] = lam
    
    
    print "MERGING: HERE ARE OLD VALUES", new_data
    #remove all the old values
    new_data['Merged'] = True
    new_data['Number'] -= 1
    for key in new_data:
        value = new_data[key]
        if isinstance(value, np.ndarray):
            value = list(value)
            value = value[0: from_index] + value[from_index + 1 :]
            value = np.array(value)
            new_data[key] = value
        elif isinstance(value, list):
            value = value[0: from_index] + value[from_index + 1 :]
            new_data[key] = value

    print "New classe:", new_data['Classes'], "VS NEW KEYS", new_clusterToPatient.keys()
            
    # integrate the old patients to the new ones

    
    
    new_data['MinClusterSize'] = min(len(new_clusterToPatient[c]) for c in new_clusterToPatient)
    
    print "MERGING: HERE ARE NEW VALUES", new_data
    
    plot_pmm_clusters(patientToGenes, clusterToPatient, new_data['Number'], title='Merging')
    
    print "End merging."
    print "****************************************"    
    
    return new_data, new_clusterToPatient
 
    
#     data['Number'] = num_components
#     data['Means'], data['Probabilities'] = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))
#     data['Likelihood'] = np.round(ll)
#     data['Classes'] = classes
#     data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
#     data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
#     data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
#     data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
#     data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

def backward_selection(data, clusterToPatient, patientToGenes, min_cluster_size = 30,
                       max_components = 10):
    """Merge clusters until a criterion is satisfied. Missing patients are assumed to
    be assigned already.
    """
    

    merged_data = data
    merged_cluster = clusterToPatient
    
    while (merged_data['Number'] > max_components or merged_data['MinClusterSize'] < min_cluster_size):
        merged_data, merged_cluster = merge_clusters(merged_data, merged_cluster, patientToGenes,
                                                    min_cluster_size = min_cluster_size)
    
    return merged_data, merged_cluster
    







def plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100, title=None):
    D = [len(patientToGenes[p]) for p in patientToGenes]

    bins = range(0, max(list(D)), max(list(D))/num_bins)
    plt.figure()
    for cluster in clusterToPatient:
        plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    if not title:
        plt.title("Cluster size " + str(num_components), fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.show()

def plot_likelihoods(ll_record):
    plt.figure()
    for i in range(len(ll_record)):
        plt.plot(ll_record[i], label=str(i))
    plt.title("Log-likelihood change in EM", fontsize=20)
    plt.legend(loc=4)
    plt.show()

# If there are any patients that aren't assigned, i.e. in cluster -1
# Throw them out?
def load_patient_cohorts(partitionfile, patientToGenes, add_to_closest=True, delimiter='\t'):
    clusterToProp = {}

    with open(partitionfile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if (row[0] == 'Cluster'): break
        # reader = csv.DictReader(csvfile, delimiter=delimiter)
        # print "begun dict reader\n"
        for row in reader:
            c = eval(row[0])
            print c
            clusterToProp[c] = {}
            clusterToProp[c]['Mean'] = eval(row[1]) if row[1] else 0
            clusterToProp[c]['Probability'] = eval(row[2]) if row[2] else 0
            clusterToProp[c]['Patients'] = set(row[3:]) if row[3] else set()


    if -1 in clusterToProp:
        if add_to_closest:
            other_cs = clusterToProp.keys()
            other_cs.remove(-1)
            print "Removed ", clusterToProp[-1]
            for patient in clusterToProp[-1]:
                sims = [(abs(len(patientToGenes[patient]) - clusterToProp[c]['Mean']), c) for c in other_cs]
                sims = sorted(sims, key = lambda entry: entry[0])
                best_c = sims[0][1]
                clusterToProp[best_c]['Patients'].add(patient)
            print "completed"

        clusterToProp.pop(-1)

    cohort_dict = {}

    for c in clusterToProp:
        cohort_dict[c] = clusterToProp[c]['Patients']
    min_cohort = cohort_dict[sorted(clusterToProp.keys(), key=lambda entry: clusterToProp[entry]['Mean'])[0]]

    return cohort_dict, clusterToProp, min_cohort

# INDEX BY LOSSES
get_ipython().magic('matplotlib inline')
def run_partitions(mutationmatrix = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.m2',
        patientFile = None, #'/Users/jlu96/maf/new/OV_broad/shared_patients.plst',
        out_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-crossval.txt',
        partition_file = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.ppf',
        load_pmm_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM.txt',
        dna_pmm_comparison_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-dnacomp.txt',
        cluster_matrix = None, # '/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl-cluster.m2',
        min_cluster_size = 15,
        num_init = 9,
        minComp = 2,
        maxComp = 5,
        do_plot = True,
        do_gmm = False,
        do_dna = False,
        num_integrated = 4,
        do_kmeans = False,
        do_pmm = True,
        do_cross_val = False,
        do_pmm_dna = True,
        do_back_selection = True,
        write_cluster_matrices = True,
        rand_num = 3,
        far_rand_num = 3,
        kf_random_state = 1,
        kf_num_folds = 5,

        geneFile = None,
        minFreq = 0,
        dna_gene_file = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/DNADamageRepair_loss.txt',
       out_dir = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/',
        write_all_partitions = True):
    
    mutationmatrix_list = mutationmatrix.split('/')
    matrix_dir = '/'.join(mutationmatrix_list[:-1]) + '/'
    prefix = (mutationmatrix_list[-1]).split('.m2')[0]
    

    if not patientFile:
        patientFile = matrix_dir + 'shared_patients.plst'
        
    if not out_file:
        if do_cross_val:
            out_file = out_dir + prefix + '-PMM-crossval-kf' + str(kf_num_folds) + '.txt'
        else:
            out_file = out_dir + prefix + '-PMM-comparisons.txt'
    
    if not partition_file:
        partition_file = matrix_dir + prefix + '.ppf'
        
    
    if not load_pmm_file:
        load_pmm_file = out_dir + prefix + '-PMM.txt'
    
    if not dna_pmm_comparison_file:
        dna_pmm_comparison_file = out_dir + prefix + '-PMM-dnacomp.txt'
        
    if not cluster_matrix:
        cluster_matrix = matrix_dir + prefix + '-cluster.m2'

    
    numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

    p_gene_list = []

    with open(dna_gene_file, 'rU') as row_file:
        reader = csv.reader(row_file, delimiter='\t')
        for row in reader:
            p_gene_list.append(row[0])
        dna_cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))


    if do_kmeans:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_gene_kmeans(geneToCases, patientToGenes, p_gene_list, i, num_bins=50, title=None, do_plot=True))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_dna:
        cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))
        # Make new cohorts over this
        if num_integrated:
            cohort_dict = integrate_cohorts(cohort_dict, numCases, num_integrated)


        cohort_pairings = [(key, cohort_dict[key]) for key in cohort_dict]
        draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title='DNADamageGenes',
                        num_bins=100 if mutationmatrix[-9:] == 'cna-jl.m2' else 50)


    if do_gmm:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_GMM(patientToGenes, i, num_bins=50, title='GMM size ' + str(i), do_plot=do_plot))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_pmm:
        datas = []
        clusters = []

        partition_stats_list = []
        for num_components in np.arange(minComp, maxComp, 1):
            best_data, clusterToPatient = best_pmm(patientToGenes, num_components, rand_num=rand_num, far_rand_num=far_rand_num,
                                                   min_cluster_size=min_cluster_size)

            if do_back_selection:
                # assign the missing data
                clusterToPatient = assign_missing(clusterToPatient, patientToGenes)
                best_data, clusterToPatient = backward_selection(best_data, clusterToPatient, patientToGenes, min_cluster_size = min_cluster_size,
                       max_components = maxComp)
            
            if do_pmm_dna:
                print "cfirst lasses are ", best_data['Classes'], "clusterToPatient is ", clusterToPatient.keys()
                pmm = PMM(lam=best_data['Means'], p_k=best_data['Probabilities'], patientToGenes=patientToGenes,
                         data=best_data, clusterToPatient=clusterToPatient, classes=best_data['Classes'],
                          do_fit=False)

                partition_stats_list.extend(pmm.compare_dna(dna_cohort_dict))

                best_data = pmm.data


            if do_cross_val:
            #cross validate each of the components
                print "*******************************************************************************************************"
                print "BEGINNING CROSS VALIDATION for ", num_components
                print "*******************************************************************************************************"
                best_data['TestLL'], best_data['TestMissing'], best_data['TestBIC'] = pmm_cross_validate(num_components, patientToGenes,
                                                                                                         num_folds=kf_num_folds,
                                                                                                     kf_random_state=kf_random_state,
                                                                                   rand_num=rand_num, far_rand_num=far_rand_num,
                                                                                   min_cluster_size=min_cluster_size)
                best_data['TestFolds'] = kf_num_folds

                print "*******************************************************************************************************"
                print "EMDING CROSS VALIDATION  for ", num_components
                print "*******************************************************************************************************"

            datas.append(best_data)
            clusters.append(clusterToPatient)
            
            if write_all_partitions:
                with open(partition_file + str(num_components), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')

                    writer.writerow(['Likelihood', best_data['Likelihood']])
                    writer.writerow(['BIC', best_data['BIC']])
                    writer.writerow(['NumComponents', best_data['Number']])
                    writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
                    if 'Merged' in best_data and best_data['Merged']:
                        for k in range(len(clusterToPatient)):
                            lam = best_data['Means'][k]
                            p_k = best_data['Probabilities'][k]
                            writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
                    else:
                        for k in clusterToPatient:
                            if k != -1:
                                lam = best_data['Means'][k]
                                p_k = best_data['Probabilities'][k]
                            else:
                                lam = None
                                p_k = None
                            writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        # get the best BIC
        combined = zip(datas, clusters)
        if do_cross_val:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], np.round(entry[0]['TestMissing']), -1 * entry[0]['TestLL'], entry[0]['TestBIC'], entry[0]['BIC']))
        else:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], entry[0]['BIC']))

        datas, clusters = zip(*combined)




        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[-1].keys(), delimiter='\t', extrasaction='ignore')
            print datas
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


        best_data = datas[0]
        clusterToPatient = clusters[0]

        # code to parition by best clusters
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            writer.writerow(['Likelihood', best_data['Likelihood']])
            writer.writerow(['BIC', best_data['BIC']])
            writer.writerow(['NumComponents', best_data['Number']])
            writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
            if 'Merged' in best_data and best_data['Merged']:
                for k in range(len(clusterToPatient)):
                    lam = best_data['Means'][k]
                    p_k = best_data['Probabilities'][k]
                    writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
            else:
                for k in clusterToPatient:
                    if k != -1:
                        lam = best_data['Means'][k]
                        p_k = best_data['Probabilities'][k]
                    else:
                        lam = None
                        p_k = None
                    writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        if write_cluster_matrices:
            for cluster in clusterToPatient:
                with open(cluster_matrix + str(cluster), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    for patient in clusterToPatient[cluster]:
                        writer.writerow('\t'.join([patient] + list(patientToGenes[patient])))


        if do_pmm_dna:
            with open(dna_pmm_comparison_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=partition_stats_list[0].keys(), delimiter='\t')
                writer.writeheader()
                print "header written"
                for row in partition_stats_list:
                    writer.writerow(row)











import mutex as mex
import csv

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
closer_than_distance = 100000000
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.ppf9'
load_pmm_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/BRCA_wustl-cna-jl-PMM.txt'
dna_pmm_comparison_file = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/BRCA_wustl-cna-jl-PMM-dnacomp.txt'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

if COSMICFile:
    COSMICgenes = set()
    with open(COSMICFile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            COSMICgenes.add(*row)
    print "Number of COSMIC genes ", len(COSMICgenes)
    genes = (set(genes)).intersection(COSMICgenes)
    geneToCases = dict([g for g in geneToCases.items() if g[0] in genes])

print "Num COSMIC genes in this cancer", len(genes)
            


cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

print clusterToProp.keys()




# let's look at the smallest cluster
c0patients = clusterToProp[0]['Patients']
c1patients = clusterToProp[1]['Patients']

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[0]['Mean']
print list(c0genes)[0:30]
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 3
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-cpairs-min_cohort.txt'
test_minFreq = 3


genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[4])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


# Apparently some of the patients in the first cluster have these pairs mutated.
# HERPUD1loss	CBFBloss
# CDH11loss	MAFloss
# CBFBloss	MAFloss
# FANCAloss	MAFloss
# CDH1loss	HERPUD1loss
# CDH1loss	MAFloss
# HERPUD1loss	MAFloss
# CDH11loss	CYLDloss
# CDH11loss	FANCAloss
# FANCAloss	CBFBloss
# 
# HERPUD1loss	16	56932048	56944863
# CBFBloss	16	67029116	67101058
# CDH11loss	16	64943753	65126112
# MAFloss	16	79585843	79600714
# FANCAloss	16	89737549	89816657
# CDH1loss	16	68737225	68835548
# PTENloss 10	87863113	87971930
# CDKN1Bloss	12	12715058	12722371
# 
# Seems like the 10 MB filter didn't work.
# 
# 
# Is it possible that those patients just lost a big chunk of their DNA?
# 
# See cell below: seems like most of the guys with the co-occurrence are just mutated crazily.
# 
# When we look at the probabilities of the genes minus the highly mutated patients (>400),
# (i.e. the 47 smallest patients),  only one pair remains:
# FANCA Aand MAF loss.
# The others are simply not mutated there. Likely that we have some false positives?
# 
# 
# Let's look at the gene segments for BRCA_wustl:
# FANCA/CBFA2T3 are in the same
#  ZNF469_ZFPM1_MIR5189_ZC3H18_IL17C_CYBA_MVD_SNAI3_RNF166_CTU2_PIEZO1_MIR4722_CDT1_APRT_GALNS_TRAPPC2L_PABPN1L_CBFA2T3_ACSF3_LINC00304_CDH15_SLC22A31_ZNF778_ANKRD11_SPG7_RPL13_SNORD68_CPNE7_DPEP1_CHMP1A_SPATA33_CDK10_SPATA2L_VPS9D1_ZNF276_FANCA_SPIRE2_TCF25_MC1R_TUBB3_DEF8_CENPBD1_AFG3L1P_DBNDD1_GAS8_URAHP_PRDM7_TUBB8P7_FAM157Clos
# 
# 
# CDH11 and CBFB are in the same segment CDH11_LINC00922_RNA5SP428_CDH5_LINC00920_BEAN1_TK2_CKLF_CMTM1_CMTM2_CMTM3_CMTM4_DYNC1LI2_CCDC79_NAE1_CA7_PDP2_CDH16_RRAD_FAM96B_CES2_CES3_CES4A_RN7SL543P_CBFB_C16orf70_B3GNT9_TRADD_FBXL8_HSF4_NOL3_KIAA0895L_EXOC3L1_E2F4_ELMO3_MIR328_LRRC29_TMEM208_FHOD1_SLC9A5_PLEKHG4_KCTD19_RN7SKP118_LRRC36_TPPP3_ZDHHC1_HSD11B2_ATP6V0D1_AGRP_FAM65A_CTCF_RLTPR_ACD_PARD6A_ENKD1_C16orf86_GFOD2_RANBP10_TSNAXIP1_CENPT_THAP11_NUTF2_EDC4_NRN1L_PSKH1_CTRL_PSMB10_LCAT_SLC12A4_DPEP3_DPEP2_DDX28_NFATC3_ESRP2_PLA2G15_SLC7A6_SLC7A6OS_PRMT7_SMPD3_ZFP90_CDH3_CDH1_RNA5SP429loss
# 
# HERPUD loss:
# GNAO1_MIR3935_AMFR_NUDT21_OGFOD1_BBS2_MT4_MT3_MT2A_MT1L_MT1E_MT1M_MT1JP_MT1A_MT1DP_MT1B_MT1F_MT1G_MT1H_MT1X_NUP93_SLC12A3_HERPUD1_CETP_NLRC5_CPNE2_FAM192A_RSPRY1_ARL2BP_PLLP_CCL22_CX3CL1_CCL17_CIAPIN1_COQ9_POLR2C_DOK4_CCDC102Alos
# 
# 
# Let's do a 100MB filter.
# Then, let's segment.
# 
# Most genes are mutated only once.
# 
# Let's confirm that they actually lost a segment of their DNA.
# 
# There was a segment deletion at chr16:88525832-9035475. Even so, they're not contained in it...
# 
# They all happened to have deletions at that segment. See 2 cells down.

genes1 = ["CDH11loss", "CBFBloss", "FANCAloss", "CDH1loss", "CDH1loss", "HERPUD1loss", "CDH11loss", "CDH11loss", "FANCAloss"]
genes2 = ["MAFloss", "MAFloss", "MAFloss", "HERPUD1loss", "MAFloss", "MAFloss", "CYLDloss", "FANCAloss", "CBFBloss"]  

genepairs = zip(genes1, genes2)
cooccurPatients = {}
for patient in c0patients:
    for gene1, gene2 in genepairs:
        if gene1 in patientToGenes[patient] and gene2 in patientToGenes[patient]:
            if patient not in cooccurPatients:
                cooccurPatients[patient] = 0
            cooccurPatients[patient] += 1
            
cooccurPatients = dict(sorted(cooccurPatients.items(), key=lambda entry: len(patientToGenes[entry[0]]), reverse=True))
print len(cooccurPatients)
print [len(patientToGenes[p]) for p in cooccurPatients]
print cooccurPatients


import pandas as pd
df = pd.read_csv('/Users/jlu96/maf/new/BRCA_wustl/all_lesions.conf_99.txt', sep='\t')
indices = np.where(df['Unique Name'][:] == "Deletion Peak 32")[0]
print indices

keys = list(df.columns.values)
patientToKey = {}
for key in keys:
    if key[:12] in cooccurPatients:
        patientToKey[key[:12]] = key

delPatients = []
for patient in cooccurPatients:
    key = patientToKey[patient]
    if df[key][60] > 0:
        delPatients.append(patient)
print delPatients


# let's try to get the probabilities using the least mutated patients.
import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-cpairs-smallest-cohort.txt'
test_minFreq = 1

newc0patients = [p for p in c0patients if len(patientToGenes[p]) < 400]
print "Smallest cluster size", len(newc0patients)
newc0geneToCases = c0geneToCases.copy()
for g in newc0geneToCases:
    newc0geneToCases[g] = newc0geneToCases[g].intersection(set(newc0patients))

genepairs = bgbp.getgenepairs(newc0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, newc0patients)


print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)



# Now, let's take a look at the segment data.
# 
# BRCA_wustl was segmented. Here are the output from there:
# cna file size:  60246810
# time used to read in:  313.754554033
# Time used to convert genes:  73.9540939331
# 22771 genes were loaded
# 2005 genes were not loaded
# Time used to get gene positions:  6.44031000137
# Binning genes.
# Adjacent concordance threshold:  0.99
# Distance threshold:  10000000.0
# Number of position segments  1921
# Number of segments  1921
# Number of genes  24776
# Time used to bin genes:  214.709136009
# ****************
# WRITING SEGMENT MATRIX:
# -----------------------
# Total number of segment alterations:  962262
# 712 segments had equal numbers of gain and losses for  a sample were found, written as gain alterations
# 10109  had some gains and some losses, but were overridden by majority
# Average majority ratio:  0.778375217298
# Standard Dev of majority ratios:  0.137182333658
# *****************
# ********************
# WRITING SEG2GENES:
# --------------------
# Number of genes  22771
# Number of segments  1921
# Average concordance:  0.988522560372
# Standard deviation of concordance:  0.0106270392535
# Average nonzero concordances:  0.973493456503
# Standard deviation of nonzero concordances:  0.0250897363908
# Segments matrix written to:  ./BRCA_wustl-seg-gl.m2
# Gene to segments written to:  ./BRCA_wustl-seg-gl.gene2seg
# 

#Let's now look at the patient mutation distribution of BRCA, segmented. Is it broad? Does it make sense to partition over this?
get_ipython().magic('matplotlib inline')

import mutex as mex

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2'
numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)
mex.graph_mutation_distribution(numCases, genes, geneToCases, filename='Breast Cancer', top_percentile=10, bins=100)
mex.graph_patient_distribution(numGenes, patients, patientToGenes, filename='Patient Mutation Distribution', top_percentile=10, bins=100)


# A partition over this space looks good! Let's partition it.

run_partitions(mutationmatrix='/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2', min_cluster_size=70,
              do_cross_val=False, do_pmm_dna=False, maxComp=10)


# when we're ready, let's look at the cooccurring pairs here. EDIT PARTITION FILE

import mutex as mex
import csv

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
closer_than_distance = 100000000
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.ppf9'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

print clusterToProp.keys(), cohort_dict.keys()


c0patients = cohort_dict[0]
c1patients = cohort_dict[1]

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[0]['Mean']
print list(c0genes)[0:30]
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "number of patients in first cluster ", len(c1patients)

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 10
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-sega-jl-cpairs-min_cohort.txt'
test_minFreq = 10


genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
print "Getting cooccurrence across the whole distribution"
cpairsdict = chi.add_BinomP_all_pairs(cpairsdict, geneToCases, patientToGenes)

cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[0])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


# We were able to find the genes that cooccur most significantly at a threshold of 14.
# However they seem to be false positives. We'll need to filter by COSMIC genes, possibly.
# 
# 
# Gene0	Gene1	MutationFrequency0	MutationFrequency1	BinomProbability	7SizeMin	7FreqsMin	7OverlapMin	7MinCBinomProb
# SNORD22loss	RNU4ATAC3P_MIR548ASloss	364	363	3.34E-70	95	[14, 14]	14	2.03E-08
# GPC6_RNA5SP35_DCT_TGDS_GPR180_RNA5SP36_RN7SL585P_LINC00391_SOX21_LINC00557_ABCC4_RNY3P8_RNY4P27_CLDN10_DZIP1_DNAJC3_UGGT2loss	SNORD22loss	370	364	3.82E-69	95	[14, 14]	14	2.03E-08
# SNORD22loss	GPC5loss	364	369	6.69E-69	95	[14, 15]	14	4.74E-08
# HS6ST3_RN7SL164Ploss	SNORD22loss	358	364	6.86E-68	95	[14, 14]	14	2.03E-08
# FAM230Closs	PEX26_TUBA8_USP18_GGTLC3_PI4KAP1_RN7SKP131_RIMBP3_GGT3P_DGCR6_PRODH_DGCR5_DGCR2_DGCR14_TSSK2_GSC2_SLC25A1_CLTCL1_HIRA_C22orf39_RN7SL168P_MRPL40_UFD1L_CDC45_CLDN5_GP1BB_TBX1_GNB1L_C22orf29_TXNRD2_COMT_MIR4761_ARVCF_TANGO2_MIR185_DGCR8_MIR3618_MIR1306_TRMT2A_RANBP1_ZDHHC8_RTN4R_MIR1286_DGCR6L_FAM230A_USP41_ZNF74_SCARF2_KLHL22_RN7SL812P_MED15_SMPD4P1_POM121L4P_TMEM191A_PI4KA_SERPIND1_SNAP29_CRKL_AIFM3_LZTR1_THAP7_TUBA3FP_P2RX6_SLC7A4_MIR649_P2RX6P_BCRP2_POM121L7_FAM230B_GGT2_RIMBP3B_RN7SKP63_HIC2_TMEM191C_PI4KAP2_RN7SKP221_RIMBP3C_UBE2L3_YDJC_CCDC116_SDF2L1_PPIL2_MIR301B_MIR130B_YPEL1_RN7SL280Ploss	410	413	5.27E-67	95	[31, 31]	31	6.73E-09
# SNORD22loss	LINC00359_RN7SKP7_OXGR1_MBNL2_RNA5SP37_RAP2Aloss	364	353	1.08E-66	95	[14, 14]	14	2.03E-08
# SNORD22loss	LINC00433_LINC00560_LINC00440_LINC00353_LINC00559_RNA5SP34_MIR622_LINC00410_LINC00379_MIR17HGloss	364	373	4.67E-66	95	[14, 14]	14	2.03E-08
# SNORD22loss	IPO5loss	364	350	1.95E-65	95	[14, 14]	14	2.03E-08
# SNORD22loss	FARP1_RNF113B_RN7SKP8_MIR3170_STK24_RN7SL60P_SLC15A1_DOCK9_UBAC2_RN7SKP9_GPR18_GPR183_MIR623_LINC00449_TM9SF2_CLYBL_MIR4306_ZIC5_ZIC2_LINC00554_PCCA_GGACT_TMTC4_LINC00411_NALCN_ITGBL1loss	364	362	6.11E-65	95	[14, 15]	14	4.74E-08
# MAPK1_RNA5SP493loss	FAM230Closs	405	410	3.20E-64	95	[30, 31]	30	1.30E-08
# SPRY2loss	SNORD22loss	383	364	2.64E-57	95	[15, 14]	13	3.44E-07
# 
# SNORD22loss position: 11	62854161	62854285
# RNU4ATAC3P_MIR548ASloss position: 13	92059844	92059969, 13	92490163	92490220
# GPC6_RNA5SP35_DCT_TGDS_GPR180_RNA5SP36_RN7SL585P_LINC00391_SOX21_LINC00557_ABCC4_RNY3P8_RNY4P27_CLDN10_DZIP1_DNAJC3_UGGT2loss	 position:
# 13	93226842	94407401
# IPO5 loss position: 13	97953658	98024297
# FAM230Closs position: 13	18195297	18232024
# PEX26loss_RN7S2Lloss position: 22	18077920	18131138, 22	21722895	21723182
# MAPK1loss: 22	21754500	21867680
# SPRY2loss	13	80335976	80340951
# 
# Conclusion:
# ## SNORD22 gets legitimate co-occurrence! But with many in same chromosome range: 13, 920000-980000
# ## Explanation: the 14 that cooccur happen to be very highly mutated patients
# ## Solution. Partition over CNA space first, but test on SEG genes
# 
# 
# ## Mutual Exclusivity: nothing seems mutex
# 
# 

# # Next plan: use the CNA clusters, but test the segment genes
# 

# when we're ready, let's look at the cooccurring pairs here. EDIT PARTITION FILE

import mutex as mex
import csv

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
geneFile = None
minFreq = 0
COSMICFile = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/COSMIC/COSMICGenes_OnlyLoss.txt'
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.ppf9'

numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

plot_pmm_clusters(patientToGenes, cohort_dict, len(cohort_dict), 100, 'Partitioned BRCA patients in Segments')
print clusterToProp.keys(), cohort_dict.keys()


c0patients = cohort_dict[0]
c1patients = cohort_dict[1]

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)
c1genes, c1geneToCases, c1patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c1patients)

print "number genes is ", len(c0genes)
print "number patients is ", len(c0patients)
print "Mean is ", clusterToProp[0]['Mean']
print "Number of genes in cluster 0: ", len(c0genes)



pfreq = [len(c0patientToGenes[p]) for p in c0patients]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(pfreq, 100)
plt.title("Patient Mutation Frequencies in first cluster")
plt.show()

gfreq = [len(c0geneToCases[g]) for g in c0geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in first cluster")
plt.show()

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]

gfreq = [len(c1geneToCases[g]) for g in c1geneToCases]
get_ipython().magic('matplotlib inline')
plt.figure()
plt.hist(gfreq, 100)
plt.title("Gene Mutation Frequencies in second cluster")
plt.show()

print "number of patients in first cluster ", len(c1patients)

print "Top gene frequencies are ", sorted(gfreq, reverse=True)[0:10]
# let's limit to the genes with at least 

test_minFreq = 5
test_genes = [c for c in c0genes if len(c0geneToCases[c]) >= test_minFreq]
print "numbr of genes used is ", len(test_genes)


print test_genes[0:10]


import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
compute_mutex = True

closer_than_distance = 50000000
cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-seg-jl-cpairs-min_cohort_cna_partitions.txt'
test_minFreq = 5


genepairs = bgbp.getgenepairs(c0geneToCases, test_genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs)

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
print "Getting cooccurrence across the whole distribution"
cpairsdict = chi.add_BinomP_all_pairs(cpairsdict, geneToCases, patientToGenes)

cpairsdict = chi.add_BinomP_min_cohort_all_pairs(cpairsdict, geneToCases, patientToGenes, cohort_dict, cohort_dict[0])

print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')


met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)


# # All found pairs were within 100 MB of each other
# 120 pairs, 30MB away from each other
# 388 pairs > 10 MB from each other
# 
# Found pairs:
# 
# ZNF267_HERC2P4_TP53TG3D_HERC2P5_SLC6A10P_HERC2P8_TP53TG3C_TP53TG3B_TP53TG3_ARHGAP23P1_LINC00273_RNA5SP406_RNA5SP407_RNA5SP408_RNA5SP409_RNA5SP410_RNA5SP413_RNA5SP415_RNA5SP416_RNA5SP417_RNA5SP418_RNA5SP419_RNA5SP420_RNA5SP421_RNA5SP422_RNA5SP423loss
# 16	31873758	31917357 to 16	35755177	35755285
# C16orf95_FBXO31_MAP1LC3B_ZCCHC14_JPH3_KLHDC4_SLC7A5_CA5A_BANPloss
# 16	87083562	87317420 to 16	87949244	88118422
# 
# No COSMIC genes found
# P = 0.0007
# 
# Coocurrs in all of them, p < 0.15 in most
# 

print cpairsdict.keys()

cooccurpatients = [patient for patient in c0patients if len(patientToGenes[patient].intersection(set(cpairsdict.keys()[0]))) >= 2]
missingpatients = c0patients.difference(cooccurpatients)
print len(cooccurpatients)


# _, _, _, _, _, old_cna_pTG = mex.load_mutation_data('/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.m2', patientFile, geneFile, minFreq)

seg_mutations = [len(patientToGenes[p]) for p in cooccurpatients]
seg_mutations_miss = [len(patientToGenes[p]) for p in missingpatients]
cna_mutations = [len(old_cna_pTG[p]) for p in cooccurpatients]
cna_mutations_miss = [len(old_cna_pTG[p]) for p in missingpatients]

get_ipython().magic('matplotlib inline')
plt.title("Segment Mutation number of cooccuring patients")
bins = np.linspace(0, 80, 100)
plt.hist(seg_mutations, alpha=0.5, label='Cooccurring', bins=bins)
plt.hist(seg_mutations_miss, alpha=0.5, label="Not cooccurring", bins=bins)
plt.legend()
plt.show()

bins = np.linspace(0, 600, 100)
plt.title("CNA Mutation number of cooccuring patients")
plt.hist(cna_mutations, alpha=0.5, label='Cooccurring', bins=bins)
plt.hist(cna_mutations_miss, alpha=0.5, label="Not cooccurring", bins=bins)
plt.legend()
plt.show()


only0 = {0: cohort_dict[0]}
plot_pmm_clusters(c0patientToGenes, only0, 1, 20, 'Partitioned BRCA patients in Segments')
print len(c0genes)


# #  We're going to take the mutations in the first cluster, and try testing pairs across all the clusters
# # Can we find significance?
# # How many pairs can we afford to test?
# 
# 
# 
# 
# 
# # Below uses the 70 minimum mutated patients
# 

import mutex as mex
import csv
import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
import time
import os

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.ppf9'
cpairfile = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-cpairs-min_cohort.txt'
geneFile = None
minFreq = 0
compute_mutex = True
closer_than_distance = 100000000
test_minFreq = 3


numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

cohort_dict, clusterToProp, min_cohort = load_patient_cohorts(partition_file, patientToGenes)

c0patients = cohort_dict[0]
c0cohort_dict = {0: c0patients}

c0genes, c0geneToCases, c0patientToGenes = get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)

print "number genes in smallest cluster is ", len(c0genes)
print "number above threashold ", len([g for g in c0genes if len(c0geneToCases[g]) >= test_minFreq])
print "number patients is ", len(c0patients)

t = time.time()
genepairs = bgbp.getgenepairs(c0geneToCases, c0genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs), " retrieved in time : ", time.time() - t




cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
print "Getting cooccurrence across the whole distribution"

cpairsdict = chi.add_BinomP_cohorts_all_pairs(cpairsdict, geneToCases, patientToGenes, c0cohort_dict, c0patients)
cpairsdict = chi.add_BinomP_all_pairs(cpairsdict, geneToCases, patientToGenes)
print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')

met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


# Plot the p-value distribution
pvalues = np.array([cpairsdict[c]['BinomProbability'] for c in cpairsdict])
logp = np.log(pvalues)

threshold = 0.05/len(logp)

plt.figure()
plt.hist(logp, bins=50)
plt.title("Distribution of log p-values")
plt.axvline(x=np.log(threshold), ymin=0, ymax=1000)
plt.show()

sig = [pvalue for pvalue in pvalues if pvalue < threshold]
print "Number of significant pairs ", len(sig)


# # Below uses the bottom 15 %, least mutated patients
# # It filters to those genes mutated in at least10% of those patients
# # It uses segmentation
# 

# Look at top segments. Same filters.


# Let's try limiting the significant co-occurrent pairs and see if we can increase our power
# Let's consider the 100 least mutated patients and search for co-occurrence within them

import mutex as mex
import csv
import mutex_triangles as met
import chisquared as chi
import bingenesbypairs as bgbp
import time
import os
import scipy.stats as stats
import partition as par

mutationmatrix = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-seg-jl.m2'
patientFile = '/Users/jlu96/maf/new/BRCA_wustl/shared_patients.plst'
partition_file = '/Users/jlu96/maf/new/BRCA_wustl/BRCA_wustl-cna-jl.ppf9'
file_prefix = '/Users/jlu96/conte/jlu/Analyses/CooccurImprovement/LorenzoModel/Binomial/BRCA_wustl-cna-jl-'
cpairfile = file_prefix + 'cpairs-min_cohort.txt'
triplet_file_prefix = file_prefix + 'triplet-'
new_cpairfile = file_prefix + "-cpairs-min_cohort_filtered.txt"
geneFile = None
minFreq = 0
compute_mutex = True
closer_than_distance = 100000000
test_minFreq = 0.1
minPercentile = 15
cpairPercentile = 1
mpairPercentile = 1


numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

D = [len(patientToGenes[p]) for p in patientToGenes]
minThreshold = stats.scoreatpercentile(D, minPercentile)

c0patients = [p for p in patientToGenes if len(patientToGenes[p]) <= minThreshold]
print "Number of new patients is ", len(c0patients)
test_minFreq = int( test_minFreq * len(c0patients))

c0cohort_dict = {0: c0patients}

c0genes, c0geneToCases, c0patientToGenes = par.get_cluster_gTC_pTG(geneToCases, patientToGenes, c0patients)

print "number genes in smallest cluster is ", len(c0genes)
print "number of genes above threashold ", len([g for g in c0genes if len(c0geneToCases[g]) >= test_minFreq])
print "number patients is ", len(c0patients)

t = time.time()
genepairs = bgbp.getgenepairs(c0geneToCases, c0genes, test_minFreq=test_minFreq, closer_than_distance=closer_than_distance)
print "Number of pairs is ", len(genepairs), " retrieved in time : ", time.time() - t

cpairsdict, cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, genepairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(cpairsdict)
print "Getting cooccurrence across the whole distribution"

cpairsdict = chi.add_BinomP_cohorts_all_pairs(cpairsdict, geneToCases, patientToGenes, c0cohort_dict, c0patients)

# cpairsdict = chi.add_BinomP_all_pairs(cpairsdict, geneToCases, patientToGenes)
print "Writing to file..."

fieldnames = (cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')

met.writeanydict(cpairsdict, cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


cpvalues = np.array([cpairsdict[c]['1CBinomProb0'] for c in cpairsdict])
logcp = np.log10(cpvalues)
mpvalues = np.array([cpairsdict[c]['1MBinomProb0'] for c in cpairsdict])
logmp = np.log10(mpvalues)

threshold = 0.05/len(logcp)

cthreshold = stats.scoreatpercentile(cpvalues, cpairPercentile)
mthreshold = stats.scoreatpercentile(mpvalues, mpairPercentile)
print "Top ", cpairPercentile, "percent of cooccurring pairs: ", cthreshold
print "Top ", mpairPercentile, "percent of mutually exclusive pairs : ", mthreshold

# Let's get the top 10 percent of pairs

goodpairs = [c for c in cpairsdict if (cpairsdict[c]['1CBinomProb0'] <= cthreshold or cpairsdict[c]['1MBinomProb0'] <= mthreshold)]
print "Now number of pairs to test ", len(goodpairs)


plt.figure()
plt.hist(logcp, bins=50)
plt.vline(x= np.log10(cthreshold), label="Top " + str(cpairPercentile) + "Cooccurrence Threshold ")
plt.title("Distribution of Co-occurring log p-values", fontsize=20)
plt.show()




threshold = 0.05/len(logmp)

plt.figure()
plt.hist(logmp, bins=50)
plt.vline(x= np.log10(mthreshold), label="Top " + str(mpairPercentile) + "Mutually Exclusive Threshold ")
plt.title("Distribution of Mutually Exclusive log p-values",fontsize=20)
plt.show()



new_cpairsdict, new_cgenedict = met.cooccurpairs(numCases, geneToCases, patientToGenes, goodpairs, compute_mutex=compute_mutex)

print "number of pairs is ", len(new_cpairsdict)
print "Getting cooccurrence across the whole distribution"

new_cpairsdict = chi.add_BinomP_cohorts_all_pairs(new_cpairsdict, geneToCases, patientToGenes, c0cohort_dict, c0patients)
new_cpairsdict = chi.add_BinomP_all_pairs(new_cpairsdict, geneToCases, patientToGenes)
print "Writing to file..."

fieldnames = (new_cpairsdict.values()[0]).keys()
fieldnames.remove('Type')
fieldnames.remove('MutationFrequencies')
fieldnames.remove('MutationFrequencyDifference')
fieldnames.remove('MutationFrequencyDifferenceRatio')
fieldnames.remove('CooccurrenceRatio')
fieldnames.remove('Coverage')
fieldnames.remove('SetScore')
fieldnames.remove('AverageOverlapPMN')
fieldnames.remove('CombinedScore')
fieldnames.remove('Concordance')
fieldnames.remove('Somatic')
fieldnames.remove('RoundedLogPCov')
fieldnames.remove('GeneSet')

met.writeanydict(new_cpairsdict, new_cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


# Plot the p-value distribution
pvalues = np.array([new_cpairsdict[c]['BinomProbability'] for c in new_cpairsdict])
logp = np.log(pvalues)

threshold = 0.05/len(logp)

plt.figure()
plt.hist(logp, bins=50)
plt.title("Distribution of log p-values")
plt.axvline(x=np.log(threshold), ymin=0, ymax=1000)
plt.show()

sig = [pvalue for pvalue in pvalues if pvalue < threshold]
print "Number of significant pairs ", len(sig)


# add the segment infos

bgbp.write_segment_infos(c0genes, "/Users/jlu96/maf/new/BRCA_wustl/segment_info.txt")

for pair in new_cpairsdict:
    info0 = bgbp.get_segment_gene_info(new_cpairsdict[pair]['Gene0'])
    new_cpairsdict[pair]['Gene0Loc'] = str(info0['Chromosome']) + ':' + str(info0['Start'])
    info1 = bgbp.get_segment_gene_info(new_cpairsdict[pair]['Gene1'])
    new_cpairsdict[pair]['Gene1Loc'] = str(info1['Chromosome']) + ':' + str(info1['Start'])
    
fieldnames += ['Gene0Loc', 'Gene1Loc']
met.writeanydict(new_cpairsdict, new_cpairfile, fieldnames=fieldnames)
os.system('say "finished"')


# # Results of segmentation. Min cluster size 145. 
# # Tested: 94 pairs. Found co-occurrence
# 
# 
# 31 significant pairs, below:
# Gene0	Gene1	MutationFrequency0	MutationFrequency1	BinomProbability
# FAM230Closs	PEX26_TUBA8_USP18_GGTLC3_PI4KAP1_RN7SKP131_RIMBP3_GGT3P_DGCR6_PRODH_DGCR5_DGCR2_DGCR14_TSSK2_GSC2_SLC25A1_CLTCL1_HIRA_C22orf39_RN7SL168P_MRPL40_UFD1L_CDC45_CLDN5_GP1BB_TBX1_GNB1L_C22orf29_TXNRD2_COMT_MIR4761_ARVCF_TANGO2_MIR185_DGCR8_MIR3618_MIR1306_TRMT2A_RANBP1_ZDHHC8_RTN4R_MIR1286_DGCR6L_FAM230A_USP41_ZNF74_SCARF2_KLHL22_RN7SL812P_MED15_SMPD4P1_POM121L4P_TMEM191A_PI4KA_SERPIND1_SNAP29_CRKL_AIFM3_LZTR1_THAP7_TUBA3FP_P2RX6_SLC7A4_MIR649_P2RX6P_BCRP2_POM121L7_FAM230B_GGT2_RIMBP3B_RN7SKP63_HIC2_TMEM191C_PI4KAP2_RN7SKP221_RIMBP3C_UBE2L3_YDJC_CCDC116_SDF2L1_PPIL2_MIR301B_MIR130B_YPEL1_RN7SL280Ploss	410	413	5.27E-67
# FAM230Closs	MAPK1_RNA5SP493loss	410	405	3.20E-64
# FAM230Closs	PPM1F_TOP3B_VPREB1_ZNF280B_ZNF280A_PRAME_POM121L1P_GGTLC2_IGLL5_MIR650_IGLJ1_IGLC1_IGLJ2_IGLC2_IGLJ3_IGLC3_IGLJ4_IGLJ5_IGLJ6_IGLJ7_IGLC7_GNAZ_RAB36_BCR_RN7SL263P_CES5AP1_ZDHHC8P1loss	410	411	5.54E-63
# FAM230Closs	OR11H1_POTEH_KCNMB3P1_CCT8L2_TPTEP1_XKR3_HSFY1P1_GAB4_CECR7_IL17RA_CECR6_CECR5_CECR1_CECR3_CECR9_RN7SL843P_CECR2_SLC25A18_ATP6V1E1_BCL2L13_BID_LINC00528_MICAL3_MIR648loss	410	413	2.45E-60
# FAM230Closs	IGLL1_GUSBP11_RGL4_ZNF70_VPREB3_C22orf15_CHCHD10_MMP11_SMARCB1_DERL3_SLC2A11_RN7SL268P_MIF_GSTT2B_DDTL_DDT_GSTT2_CABIN1_SUSD2_GGT5_POM121L9P_SPECC1L_ADORA2A_UPB1_GUCD1_SNRPD3_GGT1_PIWIL3_SGSM1_TMEM211_KIAA1671_CRYBB3_CRYBB2_LRP5L_CRYBB2P1_ADRBK2_RNA5SP494_MYO18B_RN7SKP169loss	410	413	2.51E-59
# FAM230Closs	SEZ6L_RNA5SP495_ASPHD2_HPS4_SRRD_TFIP11_TPST2_MIR548J_CRYBB1_CRYBA4_MIATloss	410	424	4.80E-54
# FAM230Closs	TTC28loss	410	419	1.38E-51
# FAM230Closs	RN7SL757Ploss	410	419	1.38E-51
# FAM230Closs	RN7SL162P_CHEK2_HSCB_CCDC117_XBP1loss	410	419	5.32E-51
# FAM230Closs	MN1_PITPNBloss	410	419	5.32E-51
# FAM230Closs	ZNRF3_C22orf31_KREMEN1_EMID1_RHBDD3_EWSR1_GAS2L1_RASL10A_AP1B1_SNORD125_RFPL1S_RFPL1_NEFH_THOC5_NIPSNAP1_NF2_CABP7_ZMAT5_UQCR10_ASCC2_MTMR3_HORMAD2_LIF_OSM_GATSL3_TBC1D10A_SF3A1_CCDC157_RNF215_SEC14L2_MTFP1_SEC14L3_SEC14L4_SEC14L6_GAL3ST1_PES1_TCN2_SLC35E4_DUSP18_OSBP2_MORC2_TUG1_RN7SL633P_SMTN_INPP5J_PLA2G3_MIR3928_RNF185_LIMK2_PIK3IP1_RNA5SP496_PATZ1_DRG1_EIF4ENIF1_SFI1_PISD_PRR14L_DEPDC5_RN7SL20P_C22orf24_YWHAH_RN7SL305P_SLC5A1_AP1B1P1_C22orf42_RFPL2_SLC5A4_RFPL3_RFPL3S_RTCB_BPIFC_FBXO7_SYN3_RNA5SP497_TIMP3loss	410	432	8.72E-49
# FAM230Closs	MIR4764loss	410	427	3.70E-48
# FAM230Closs	ISXloss	410	433	1.34E-47
# FAM230Closs	LARGEloss	410	429	2.52E-47
# CRK_MYO1C_INPP5K_PITPNA_SLC43A2_RN7SL105P_SCARF1_RILP_PRPF8_TLCD2_MIR22HG_WDR81_SERPINF2_SERPINF1_SMYD4_RPA1_RTN4RL1_DPH1_OVCA2_MIR132_MIR212_HIC1_SMG6_RN7SL624P_SRR_TSR1_SNORD91B_SNORD91A_SGSM2_MNT_METTL16_RN7SL33P_PAFAH1B1_RN7SL608P_CLUH_MIR1253_RAP1GAP2_OR1D5_OR1D2_OR1G1_OR1A2_OR1A1_OR1D4_OR3A2_OR3A1_OR3A4P_OR1E1_OR3A3_OR1E2_SPATA22_ASPA_TRPV3_TRPV1_SHPK_CTNS_TAX1BP3_EMC6_P2RX5_ITGAE_GSG2_C17orf85_CAMKK1_P2RX1_ATP2A3_ZZEF1_RNA5SP434_CYB5D2_ANKFY1_UBE2G1_RN7SL774P_SPNS3_SPNS2_MYBBP1A_GGT6_SMTNL2_ALOX15_PELP1_ARRB2_MED11_CXCL16_ZMYND15_TM4SF5_VMO1_GLTPD2_PSMB6_PLD2_MINK1_RN7SL784P_CHRNE_C17orf107_GP1BA_SLC25A11_RNF167_PFN1_ENO3_SPAG7_CAMTA2_INCA1_KIF1C_SLC52A1_ZFP3_ZNF232_USP6_ZNF594_SCIMP_RABEP1_NUP88_RPAIN_C1QBP_DHX33_DERL2_MIS12_NLRP1loss	RN7SL605Ploss	580	564	1.94E-46
# DOC2B_RPH3AL_C17orf97_FAM101B_VPS53_FAM57A_GEMIN4_DBIL5P_GLOD4_RNMTL1_NXN_TIMM22_ABR_MIR3183_BHLHA9_TUSC5loss	RN7SL605Ploss	565	564	3.87E-46
# YWHAEloss	RN7SL605Ploss	566	564	7.08E-46
# FAM230Closs	HMGXB4_TOM1_MIR3909_HMOX1_MCM5_RASD2_MB_APOL6_APOL5_RBFOX2_APOL3_APOL4_APOL2_APOL1_MYH9loss	410	448	3.19E-45
# WSCD1loss	RN7SL605Ploss	574	564	9.93E-43
# FAM230Closs	LINC00898loss	410	442	3.32E-41
# FAM230Closs	TBC1D22Aloss	410	449	1.90E-40
# FAM230Closs	MIR3201_FAM19A5_MIR4535_C22orf34_MIR3667_RN7SKP252_BRD1_ZBED4_ALG12_CRELD2_PIM3_IL17REL_TTLL8_MLC1_MOV10L1_PANX2_TRABD_TUBGCP6_HDAC10_MAPK12_MAPK11_PLXNB2_DENND6B_PPP6R2_RN7SL500P_SBF1_ADM2_MIOX_LMF2_NCAPH2_SCO2_TYMP_ODF3B_KLHDC7B_SYCE3_CPT1B_CHKB_MAPK8IP2_ARSA_SHANK3_ACR_RABL2Bloss	410	438	3.66E-40
# AIPL1_FAM64A_KIAA0753_RNA5SP435_TXNDC17_MED31_C17orf100_SLC13A5_XAF1_FBXO39_TEKT1_ALOX12P2_ALOX12_RNASEK_C17orf49_MIR497HG_BCL6B_SLC16A13_SLC16A11_CLEC10A_ASGR2_ASGR1_DLG4_ACADVL_MIR324_DVL2_PHF23_GABARAP_CTDNEP1_ELP5_CLDN7_SLC2A4_YBX2_EIF5A_GPS2_NEURL4_ACAP1_KCTD11_TMEM95_TNK1_TMEM256_NLGN2_SPEM1_C17orf74_TMEM102_FGF11_CHRNB1_ZBTB4_SLC35G6_POLR2A_TNFSF12_TNFSF13_SENP3_EIF4A1_SNORD10_CD68_MPDU1_SOX15_FXR2_SHBG_SAT2_ATP1B2_TP53_WRAP53_EFNB3_DNAH2_RPL29P2_KDM6B_TMEM88loss	RN7SL605Ploss	596	564	9.45E-40
# CYB5D1_CHD3_KCNAB3_TRAPPC1_CNTROB_GUCY2D_ALOX15B_ALOX12B_MIR4314_ALOXE3_HES7_PER1_VAMP2_TMEM107_SNORD118_C17orf59_AURKB_LINC00324_CTC1_PFAS_SLC25A35_RANGRF_ARHGEF15_ODF4_KRBA2_RPL26_RNF222_NDEL1_MYH10_CCDC42_SPDYE4_MFSD6L_PIK3R6_PIK3R5_NTN1loss	RN7SL605Ploss	591	564	1.05E-39
# DNAH9_ZNF18_RPL21P122loss	RN7SL605Ploss	583	564	4.32E-39
# MAP2K4_MIR744loss	RN7SL605Ploss	583	564	4.32E-39
# STX8_USP43_DHRS7C_GLP2R_RCVRN_GAS7loss	RN7SL605Ploss	591	564	1.12E-38
# FAM230Closs	TXN2_FOXRED2_EIF3D_CACNG2_IFT27_PVALB_NCF4_CSF2RB_TEX33_TST_MPST_KCTD17_RN7SKP214_TMPRSS6_IL2RB_C1QTNF6_SSTR3_RAC2_CYTH4_ELFN2_MFNG_CARD10_CDC42EP1_LGALS2_GGA1_SH3BP1_PDXP_RN7SL385P_LGALS1_NOL12_TRIOBP_H1F0_GCAT_GALR3_ANKRD54_MIR658_MIR659_EIF3L_MICALL1_C22orf23_POLR2F_SOX10_MIR4534_PICK1_SLC16A8_BAIAP2L2_PLA2G6_MAFF_TMEM184B_RN7SL704P_CSNK1E_KCNJ4_KDELR3_DDX17_DMC1_FAM227A_CBY1_TOMM22_JOSD1_GTPBP1_SUN2_DNAL4_NPTXR_CBX6_APOBEC3A_APOBEC3B_APOBEC3C_APOBEC3D_APOBEC3F_APOBEC3G_APOBEC3H_CBX7_PDGFB_RPL3_SNORD83B_SNORD83A_SYNGR1_TAB1_MGAT3_ATF4_RPS19BP1_CACNA1I_ENTHD1_RN7SKP210_GRAP2_FAM83F_TNRC6B_ADSL_SGSM3_MKL1_MCHR1_SLC25A17_MIR4766_ST13_XPNPEP3_DNAJB7_RBX1_EP300_MIR1281_L3MBTL2_CHADL_RANGAP1_ZC3H7B_TEF_TOB2_PHF5A_ACO2_POLR3H_CSDC2_PMM1_DESI1_XRCC6_NHP2L1_C22orf46_MEI1_RNU6ATAC22P_CCDC134_SREBF2_MIR33A_SHISA8_TNFRSF13C_MIR378I_CENPM_LINC00634_WBP2NL_NAGA_FAM109B_SMDT1_NDUFA6_CYP2D6_TCF20_NFAM1_SERHL_RRP7A_SERHL2_RN7SKP80_POLDIP3_RNU12_CYB5R3_ATP5L2_A4GALT_ARFGAP3_PACSIN2_TTLL1_BIK_MCAT_TSPO_TTLL12_SCUBE1_MPPED1_EFCAB6_SULT4A1_PNPLA5_PNPLA3_SAMM50_PARVB_PARVG_KIAA1644_LDOC1L_LINC00207_LINC00229_PRR5_ARHGAP8_PHF21B_NUP50_KIAA0930_MIR1249_UPK3A_FAM118A_SMC1B_RIBC2_FBLN1_ATXN10_MIR4762_WNT7B_LINC00899_MIR3619_MIRLET7A3_MIR4763_MIRLET7B_PPARA_CDPF1_PKDREJ_TTC38_GTSE1_TRMU_CELSR1_GRAMD4_CERKloss	410	476	1.13E-38
# MYH13_MYH8_MYH4_MYH1_MYH2_MYH3_SCO1_ADPRM_TMEM220_LINC00675_PIRT_RN7SL601P_SHISA6loss	RN7SL605Ploss	585	564	1.25E-38
# LINC00670_MYOCD_ARHGAP44_MIR1269B_ELAC2loss	RN7SL605Ploss	579	564	1.96E-38
# HS3ST3A1_MIR548H3_COX10_CDRT15_HS3ST3B1_CDRT7_CDRT8_PMP22_MIR4731_TEKT3_RN7SL792P_CDRT4_TVP23C_CDRT1_TRIM16_ZNF286A_TBC1D26_RNA5SP436_ADORA2B_ZSWIM7_TTC19_NCOR1_RN7SL442P_PIGL_MIR1288_CENPV_UBB_TRPV2_ZNF287_ZNF624_CCDC144A_RN7SL620P_USP32P1_KRT16P2_KRT17P1_TBC1D27_TNFRSF13Bloss	RN7SL605Ploss	575	564	6.06E-38
# 
# Locations:
# Gene0Loc	Gene1Loc
# 13:18195297	22:18077920
# 13:18195297	22:21754500
# 13:18195297	22:21919420
# 13:18195297	22:15528158
# 13:18195297	22:23573125
# 13:18195297	22:26169474
# 13:18195297	22:27978014
# 13:18195297	22:28056153
# 13:18195297	22:28642979
# 13:18195297	22:27748277
# 13:18195297	22:28883592
# 13:18195297	22:33436582
# 13:18195297	22:35066136
# 13:18195297	22:33162226
# 17:1420689	11:57528085
# 17:142789	11:57528085
# 17:1344272	11:57528085
# 13:18195297	22:35257452
# 17:5772234	11:57528085
# 13:18195297	22:47621043
# 13:18195297	22:46762617
# 13:18195297	22:48274364
# 17:6393693	11:57528085
# 17:7857746	11:57528085
# 17:11598431	11:57528085
# 17:12020824	11:57528085
# 17:9250471	11:57528085
# 13:18195297	22:36467036
# 17:10298084	11:57528085
# 17:12549782	11:57528085
# 17:13495689	11:57528085
# 
# This looks good.
# 

# Search for triplets
import mutex_triangles as met
import edgereader as edg
reload(chi)
reload(met)

pairsdict = new_cpairsdict.copy()
genedict = edg.get_gene_dict(pairsdict)
Triplets, pairsdict_Triplets, sorted_pairs, genesdict_Triplets, sorted_genes = met.getTriplets(pairsdict, genedict, pairsdict, numCases, geneToCases=geneToCases, patientToGenes=patientToGenes)
Triplet_dict = met.sort_triplets_by_type(Triplets)

for t_type in Triplet_dict:
    print len(Triplet_dict[t_type]), " of type ",  t_type
    met.writeTriplets(Triplet_dict[t_type], triplet_file_prefix + t_type + '.txt')

# Let's built a nonparametric method to test for significance

pvalues = []
pvalues_ab = []
for i in range(len(Triplets)):
    
    t_genes = Triplets[i].genes
    if Triplets[i].type == 'CooccurringMutuallyExclusiveMutuallyExclusive':
        pvalues.append(chi.get_triplet_BinomP(t_genes, geneToCases, patientToGenes,
                                     cpairs=Triplets[i].cpairs,
                                     mpairs=Triplets[i].mpairs))
        pvalues_ab.append(chi.get_triplet_BinomP_ab(t_genes, geneToCases, patientToGenes,
                                             cpairs=Triplets[i].cpairs,
                                             mpairs=Triplets[i].mpairs))

               
print pvalues
print pvalues_ab


    


with open(triplet_file_prefix + 'CooccurringMutuallyExclusiveMutuallyExclusive' + '_pvalues.txt', 'w') as csvfile:
    writer = csv.writer(csvfile,delimiter='\t')
    writer.writerow(['P(A)P(B) pvalue', 'P(AB) pvalue'])
    for pvalue, pvalue_ab in zip(pvalues, pvalues_ab):
        writer.writerow([pvalue, pvalue_ab])


import matplotlib.pyplot as plt
logpvalues = np.log10(pvalues)
plt.scatter(logpvalues, pvalues_ab)
plt.xlabel("Log PValues using P(A)P(B)", fontsize=20)
plt.ylabel("Pvalues using P(A&B)", fontsize=20)
plt.title("Triplet Pvalues", fontsize=20)
plt.show()


# ## Triplets analysis
# 
# One of the triplets:
# CMM: 151 in all, 259 have 2, 213 have only the remaining
# 

# Mutual Exclusivity/Co-occurrence: Statistical model
# =====================================================
# 
# Let D represent our data. Let P(A), P(B) be bernoulli random variables, and assume our observations (patients)are all i.i.d.
# Let X_A be the count of A in the data. Let X_B be the count of B in the data. Both are binomial random variables.
# Let X_{A&B} be the count of A & B in the data.
# Let n be the total observations (i.e. patients) we have.
# 
# 
# Define mutual exclusivity b/n A and B as
# 
# $$M_{A,B} = P(A&B) < P(A)P(B)$$
# 
# Define co-occurrence b/n A and B as 
# 
# $$C_{A,B} = P(A&B) > P(A)P(B)$$
# 
# Define triplet co-occurrence b/n A, B, and C as
# 
# $$CCC_{A, B, C} = P(A&B&C) > P(A)P(B)P(C)$$
# 
# Define CCM b/n A, B, and C (A&B are mutex) as
# 
# $$CCM_{A, B, C} = P(A&!B&C) + P(!A&B&C) > P(A)P(!B)P(C) + P(!A)P(B)P(C)$$
# 
# Define CMM b/n A, B, and C (A&B cooccur) as
# 
# $$CMM_{A, B, C} = P(A&B&!C) + P(!A&!B&C) > P(A)P(B)P(!C) + P(!A)P(!B)P(C)$$
# 
# Define triplet mutual exclusviity b/n A, B, and C as:
# 
# $$MMM_{A, B, C} = P(A&!B&!C) + P(!A&B&!C) + P(!A&!B&C) > P(A)P(!B)P(!C) + P(!A)P(B)P(!C) + P(!A)P(!B)P(C)$$.
# 
# 
# 
# ### Frequentist formulation
# We test for M_{A,B} under $H_0: P(A&B) = P(A)P(B)$.
# H_0 assumes: A & B are independent Bernoulli random variables
# 
# We estimate P(A), and P(B) as p(A)=X_A/n, and p(B)=X_B/n, the data.
# We then calculate $P(X_{A&B} | P(A&B) = P(A)P(B), P(A) = p(A), P(B) = p(B))$. If this is below some threshold, we accept.a
# 
# 
# ### Bayesian formulation
# Using the same estimates as above,
# we then calculate $P(X_{A&B} | P(A&B) = P(A)P(B), P(A) = p(A), P(B) = p(B))$.
# We can also calculate $P(X_{A&B} | P(A&B) > P(A)P(B), P(A) = p(A), P(B) = p(B))$. This is conditioning on the event of co_occurrence
# 
# 
# We can then compute Bayes Factors: 
# $P(X_{A&B} | P(A&B) > p(A)p(B))/P(X_{A&B} | P(A&B) = p(A)p(B))$
# 
# 
# Should we integrate out over events P(A) != p(A), P(B) != p(B) to get:
# P(X_{A&B} | P(A&B) = P(A)P(B))?
# 
# 
# 
# 
# We have co-occurrence
# 

# BRCA wustl: rerun


# # Results of segmentation. Min cluster size 145. 
# # Tested: 94 pairs. Found co-occurrence
# 
# 
# 31 significant pairs, below:
# Gene0	Gene1	MutationFrequency0	MutationFrequency1	BinomProbability
# FAM230Closs	PEX26_TUBA8_USP18_GGTLC3_PI4KAP1_RN7SKP131_RIMBP3_GGT3P_DGCR6_PRODH_DGCR5_DGCR2_DGCR14_TSSK2_GSC2_SLC25A1_CLTCL1_HIRA_C22orf39_RN7SL168P_MRPL40_UFD1L_CDC45_CLDN5_GP1BB_TBX1_GNB1L_C22orf29_TXNRD2_COMT_MIR4761_ARVCF_TANGO2_MIR185_DGCR8_MIR3618_MIR1306_TRMT2A_RANBP1_ZDHHC8_RTN4R_MIR1286_DGCR6L_FAM230A_USP41_ZNF74_SCARF2_KLHL22_RN7SL812P_MED15_SMPD4P1_POM121L4P_TMEM191A_PI4KA_SERPIND1_SNAP29_CRKL_AIFM3_LZTR1_THAP7_TUBA3FP_P2RX6_SLC7A4_MIR649_P2RX6P_BCRP2_POM121L7_FAM230B_GGT2_RIMBP3B_RN7SKP63_HIC2_TMEM191C_PI4KAP2_RN7SKP221_RIMBP3C_UBE2L3_YDJC_CCDC116_SDF2L1_PPIL2_MIR301B_MIR130B_YPEL1_RN7SL280Ploss	410	413	5.27E-67
# FAM230Closs	MAPK1_RNA5SP493loss	410	405	3.20E-64
# FAM230Closs	PPM1F_TOP3B_VPREB1_ZNF280B_ZNF280A_PRAME_POM121L1P_GGTLC2_IGLL5_MIR650_IGLJ1_IGLC1_IGLJ2_IGLC2_IGLJ3_IGLC3_IGLJ4_IGLJ5_IGLJ6_IGLJ7_IGLC7_GNAZ_RAB36_BCR_RN7SL263P_CES5AP1_ZDHHC8P1loss	410	411	5.54E-63
# FAM230Closs	OR11H1_POTEH_KCNMB3P1_CCT8L2_TPTEP1_XKR3_HSFY1P1_GAB4_CECR7_IL17RA_CECR6_CECR5_CECR1_CECR3_CECR9_RN7SL843P_CECR2_SLC25A18_ATP6V1E1_BCL2L13_BID_LINC00528_MICAL3_MIR648loss	410	413	2.45E-60
# FAM230Closs	IGLL1_GUSBP11_RGL4_ZNF70_VPREB3_C22orf15_CHCHD10_MMP11_SMARCB1_DERL3_SLC2A11_RN7SL268P_MIF_GSTT2B_DDTL_DDT_GSTT2_CABIN1_SUSD2_GGT5_POM121L9P_SPECC1L_ADORA2A_UPB1_GUCD1_SNRPD3_GGT1_PIWIL3_SGSM1_TMEM211_KIAA1671_CRYBB3_CRYBB2_LRP5L_CRYBB2P1_ADRBK2_RNA5SP494_MYO18B_RN7SKP169loss	410	413	2.51E-59
# FAM230Closs	SEZ6L_RNA5SP495_ASPHD2_HPS4_SRRD_TFIP11_TPST2_MIR548J_CRYBB1_CRYBA4_MIATloss	410	424	4.80E-54
# FAM230Closs	TTC28loss	410	419	1.38E-51
# FAM230Closs	RN7SL757Ploss	410	419	1.38E-51
# FAM230Closs	RN7SL162P_CHEK2_HSCB_CCDC117_XBP1loss	410	419	5.32E-51
# FAM230Closs	MN1_PITPNBloss	410	419	5.32E-51
# FAM230Closs	ZNRF3_C22orf31_KREMEN1_EMID1_RHBDD3_EWSR1_GAS2L1_RASL10A_AP1B1_SNORD125_RFPL1S_RFPL1_NEFH_THOC5_NIPSNAP1_NF2_CABP7_ZMAT5_UQCR10_ASCC2_MTMR3_HORMAD2_LIF_OSM_GATSL3_TBC1D10A_SF3A1_CCDC157_RNF215_SEC14L2_MTFP1_SEC14L3_SEC14L4_SEC14L6_GAL3ST1_PES1_TCN2_SLC35E4_DUSP18_OSBP2_MORC2_TUG1_RN7SL633P_SMTN_INPP5J_PLA2G3_MIR3928_RNF185_LIMK2_PIK3IP1_RNA5SP496_PATZ1_DRG1_EIF4ENIF1_SFI1_PISD_PRR14L_DEPDC5_RN7SL20P_C22orf24_YWHAH_RN7SL305P_SLC5A1_AP1B1P1_C22orf42_RFPL2_SLC5A4_RFPL3_RFPL3S_RTCB_BPIFC_FBXO7_SYN3_RNA5SP497_TIMP3loss	410	432	8.72E-49
# FAM230Closs	MIR4764loss	410	427	3.70E-48
# FAM230Closs	ISXloss	410	433	1.34E-47
# FAM230Closs	LARGEloss	410	429	2.52E-47
# CRK_MYO1C_INPP5K_PITPNA_SLC43A2_RN7SL105P_SCARF1_RILP_PRPF8_TLCD2_MIR22HG_WDR81_SERPINF2_SERPINF1_SMYD4_RPA1_RTN4RL1_DPH1_OVCA2_MIR132_MIR212_HIC1_SMG6_RN7SL624P_SRR_TSR1_SNORD91B_SNORD91A_SGSM2_MNT_METTL16_RN7SL33P_PAFAH1B1_RN7SL608P_CLUH_MIR1253_RAP1GAP2_OR1D5_OR1D2_OR1G1_OR1A2_OR1A1_OR1D4_OR3A2_OR3A1_OR3A4P_OR1E1_OR3A3_OR1E2_SPATA22_ASPA_TRPV3_TRPV1_SHPK_CTNS_TAX1BP3_EMC6_P2RX5_ITGAE_GSG2_C17orf85_CAMKK1_P2RX1_ATP2A3_ZZEF1_RNA5SP434_CYB5D2_ANKFY1_UBE2G1_RN7SL774P_SPNS3_SPNS2_MYBBP1A_GGT6_SMTNL2_ALOX15_PELP1_ARRB2_MED11_CXCL16_ZMYND15_TM4SF5_VMO1_GLTPD2_PSMB6_PLD2_MINK1_RN7SL784P_CHRNE_C17orf107_GP1BA_SLC25A11_RNF167_PFN1_ENO3_SPAG7_CAMTA2_INCA1_KIF1C_SLC52A1_ZFP3_ZNF232_USP6_ZNF594_SCIMP_RABEP1_NUP88_RPAIN_C1QBP_DHX33_DERL2_MIS12_NLRP1loss	RN7SL605Ploss	580	564	1.94E-46
# DOC2B_RPH3AL_C17orf97_FAM101B_VPS53_FAM57A_GEMIN4_DBIL5P_GLOD4_RNMTL1_NXN_TIMM22_ABR_MIR3183_BHLHA9_TUSC5loss	RN7SL605Ploss	565	564	3.87E-46
# YWHAEloss	RN7SL605Ploss	566	564	7.08E-46
# FAM230Closs	HMGXB4_TOM1_MIR3909_HMOX1_MCM5_RASD2_MB_APOL6_APOL5_RBFOX2_APOL3_APOL4_APOL2_APOL1_MYH9loss	410	448	3.19E-45
# WSCD1loss	RN7SL605Ploss	574	564	9.93E-43
# FAM230Closs	LINC00898loss	410	442	3.32E-41
# FAM230Closs	TBC1D22Aloss	410	449	1.90E-40
# FAM230Closs	MIR3201_FAM19A5_MIR4535_C22orf34_MIR3667_RN7SKP252_BRD1_ZBED4_ALG12_CRELD2_PIM3_IL17REL_TTLL8_MLC1_MOV10L1_PANX2_TRABD_TUBGCP6_HDAC10_MAPK12_MAPK11_PLXNB2_DENND6B_PPP6R2_RN7SL500P_SBF1_ADM2_MIOX_LMF2_NCAPH2_SCO2_TYMP_ODF3B_KLHDC7B_SYCE3_CPT1B_CHKB_MAPK8IP2_ARSA_SHANK3_ACR_RABL2Bloss	410	438	3.66E-40
# AIPL1_FAM64A_KIAA0753_RNA5SP435_TXNDC17_MED31_C17orf100_SLC13A5_XAF1_FBXO39_TEKT1_ALOX12P2_ALOX12_RNASEK_C17orf49_MIR497HG_BCL6B_SLC16A13_SLC16A11_CLEC10A_ASGR2_ASGR1_DLG4_ACADVL_MIR324_DVL2_PHF23_GABARAP_CTDNEP1_ELP5_CLDN7_SLC2A4_YBX2_EIF5A_GPS2_NEURL4_ACAP1_KCTD11_TMEM95_TNK1_TMEM256_NLGN2_SPEM1_C17orf74_TMEM102_FGF11_CHRNB1_ZBTB4_SLC35G6_POLR2A_TNFSF12_TNFSF13_SENP3_EIF4A1_SNORD10_CD68_MPDU1_SOX15_FXR2_SHBG_SAT2_ATP1B2_TP53_WRAP53_EFNB3_DNAH2_RPL29P2_KDM6B_TMEM88loss	RN7SL605Ploss	596	564	9.45E-40
# CYB5D1_CHD3_KCNAB3_TRAPPC1_CNTROB_GUCY2D_ALOX15B_ALOX12B_MIR4314_ALOXE3_HES7_PER1_VAMP2_TMEM107_SNORD118_C17orf59_AURKB_LINC00324_CTC1_PFAS_SLC25A35_RANGRF_ARHGEF15_ODF4_KRBA2_RPL26_RNF222_NDEL1_MYH10_CCDC42_SPDYE4_MFSD6L_PIK3R6_PIK3R5_NTN1loss	RN7SL605Ploss	591	564	1.05E-39
# DNAH9_ZNF18_RPL21P122loss	RN7SL605Ploss	583	564	4.32E-39
# MAP2K4_MIR744loss	RN7SL605Ploss	583	564	4.32E-39
# STX8_USP43_DHRS7C_GLP2R_RCVRN_GAS7loss	RN7SL605Ploss	591	564	1.12E-38
# FAM230Closs	TXN2_FOXRED2_EIF3D_CACNG2_IFT27_PVALB_NCF4_CSF2RB_TEX33_TST_MPST_KCTD17_RN7SKP214_TMPRSS6_IL2RB_C1QTNF6_SSTR3_RAC2_CYTH4_ELFN2_MFNG_CARD10_CDC42EP1_LGALS2_GGA1_SH3BP1_PDXP_RN7SL385P_LGALS1_NOL12_TRIOBP_H1F0_GCAT_GALR3_ANKRD54_MIR658_MIR659_EIF3L_MICALL1_C22orf23_POLR2F_SOX10_MIR4534_PICK1_SLC16A8_BAIAP2L2_PLA2G6_MAFF_TMEM184B_RN7SL704P_CSNK1E_KCNJ4_KDELR3_DDX17_DMC1_FAM227A_CBY1_TOMM22_JOSD1_GTPBP1_SUN2_DNAL4_NPTXR_CBX6_APOBEC3A_APOBEC3B_APOBEC3C_APOBEC3D_APOBEC3F_APOBEC3G_APOBEC3H_CBX7_PDGFB_RPL3_SNORD83B_SNORD83A_SYNGR1_TAB1_MGAT3_ATF4_RPS19BP1_CACNA1I_ENTHD1_RN7SKP210_GRAP2_FAM83F_TNRC6B_ADSL_SGSM3_MKL1_MCHR1_SLC25A17_MIR4766_ST13_XPNPEP3_DNAJB7_RBX1_EP300_MIR1281_L3MBTL2_CHADL_RANGAP1_ZC3H7B_TEF_TOB2_PHF5A_ACO2_POLR3H_CSDC2_PMM1_DESI1_XRCC6_NHP2L1_C22orf46_MEI1_RNU6ATAC22P_CCDC134_SREBF2_MIR33A_SHISA8_TNFRSF13C_MIR378I_CENPM_LINC00634_WBP2NL_NAGA_FAM109B_SMDT1_NDUFA6_CYP2D6_TCF20_NFAM1_SERHL_RRP7A_SERHL2_RN7SKP80_POLDIP3_RNU12_CYB5R3_ATP5L2_A4GALT_ARFGAP3_PACSIN2_TTLL1_BIK_MCAT_TSPO_TTLL12_SCUBE1_MPPED1_EFCAB6_SULT4A1_PNPLA5_PNPLA3_SAMM50_PARVB_PARVG_KIAA1644_LDOC1L_LINC00207_LINC00229_PRR5_ARHGAP8_PHF21B_NUP50_KIAA0930_MIR1249_UPK3A_FAM118A_SMC1B_RIBC2_FBLN1_ATXN10_MIR4762_WNT7B_LINC00899_MIR3619_MIRLET7A3_MIR4763_MIRLET7B_PPARA_CDPF1_PKDREJ_TTC38_GTSE1_TRMU_CELSR1_GRAMD4_CERKloss	410	476	1.13E-38
# MYH13_MYH8_MYH4_MYH1_MYH2_MYH3_SCO1_ADPRM_TMEM220_LINC00675_PIRT_RN7SL601P_SHISA6loss	RN7SL605Ploss	585	564	1.25E-38
# LINC00670_MYOCD_ARHGAP44_MIR1269B_ELAC2loss	RN7SL605Ploss	579	564	1.96E-38
# HS3ST3A1_MIR548H3_COX10_CDRT15_HS3ST3B1_CDRT7_CDRT8_PMP22_MIR4731_TEKT3_RN7SL792P_CDRT4_TVP23C_CDRT1_TRIM16_ZNF286A_TBC1D26_RNA5SP436_ADORA2B_ZSWIM7_TTC19_NCOR1_RN7SL442P_PIGL_MIR1288_CENPV_UBB_TRPV2_ZNF287_ZNF624_CCDC144A_RN7SL620P_USP32P1_KRT16P2_KRT17P1_TBC1D27_TNFRSF13Bloss	RN7SL605Ploss	575	564	6.06E-38
# 
# Locations:
# Gene0Loc	Gene1Loc
# 13:18195297	22:18077920
# 13:18195297	22:21754500
# 13:18195297	22:21919420
# 13:18195297	22:15528158
# 13:18195297	22:23573125
# 13:18195297	22:26169474
# 13:18195297	22:27978014
# 13:18195297	22:28056153
# 13:18195297	22:28642979
# 13:18195297	22:27748277
# 13:18195297	22:28883592
# 13:18195297	22:33436582
# 13:18195297	22:35066136
# 13:18195297	22:33162226
# 17:1420689	11:57528085
# 17:142789	11:57528085
# 17:1344272	11:57528085
# 13:18195297	22:35257452
# 17:5772234	11:57528085
# 13:18195297	22:47621043
# 13:18195297	22:46762617
# 13:18195297	22:48274364
# 17:6393693	11:57528085
# 17:7857746	11:57528085
# 17:11598431	11:57528085
# 17:12020824	11:57528085
# 17:9250471	11:57528085
# 13:18195297	22:36467036
# 17:10298084	11:57528085
# 17:12549782	11:57528085
# 17:13495689	11:57528085
# 
# This looks good.
# 

# Search for triplets
import mutex_triangles as met
import edgereader as edg
reload(chi)
reload(met)

pairsdict = new_cpairsdict.copy()
genedict = edg.get_gene_dict(pairsdict)
Triplets, pairsdict_Triplets, sorted_pairs, genesdict_Triplets, sorted_genes = met.getTriplets(pairsdict, genedict, pairsdict, numCases, geneToCases=geneToCases, patientToGenes=patientToGenes)
Triplet_dict = met.sort_triplets_by_type(Triplets)

for t_type in Triplet_dict:
    print len(Triplet_dict[t_type]), " of type ",  t_type
    met.writeTriplets(Triplet_dict[t_type], triplet_file_prefix + t_type + '.txt')

# Let's built a nonparametric method to test for significance

pvalues = []
pvalues_ab = []
for i in range(len(Triplets)):
    
    t_genes = Triplets[i].genes
    if Triplets[i].type == 'CooccurringMutuallyExclusiveMutuallyExclusive':
        pvalues.append(chi.get_triplet_BinomP(t_genes, geneToCases, patientToGenes,
                                     cpairs=Triplets[i].cpairs,
                                     mpairs=Triplets[i].mpairs))
        pvalues_ab.append(chi.get_triplet_BinomP_ab(t_genes, geneToCases, patientToGenes,
                                             cpairs=Triplets[i].cpairs,
                                             mpairs=Triplets[i].mpairs))

               
print pvalues
print pvalues_ab


    


# # Summary of analysis with taking in 80 least mutated patients, using CNAs
# # Result: 
# 1. dominated by co-occurrence among a few patients and between a few genes that weren't filtered out by the MB filter
# 2. not enough gene co-occurrence/mutex within the 80 least mutated
# 
# 
# Filters:
# -50 MB
# -at least 5 x in first cluster, size 80
# 
# Log:
# -running with 50MB seems to result in a bunch of pairs between the same several patients (33, 16)
# 
# TELL LORENZO: no more cooccurrence. Try a bigger partition of the data? Look into mutex?
# 
# -One significant pair at 0.005. Two genes mutated 3 times, cooccur twice
# -1276 mutated 3 times, cooccur in 1.
# 
# -plot p-value distribution
# -put on figure
# -run with bigger partitions: size 100
# 
# Gene0	Gene1	MutationFrequency0	MutationFrequency1	Overlap	1Freqs0	1Overlap0	1CBinomProb0	BinomProbability
# TMEM132Dloss	CCDC91loss	196	103	52	[3, 3]	0	1	5.62E-09
# TMEM132Dloss	BCL2L14loss	196	156	65	[3, 3]	0	1	1.05E-07
# TMEM132Dloss	LRP6loss	196	158	65	[3, 3]	0	1	2.01E-07
# TMEM132Dloss	RERGloss	196	159	65	[3, 3]	0	1	2.23E-07
# TMEM132Dloss	EMP1loss	196	170	68	[3, 3]	0	1	2.28E-07
# TMEM132Dloss	KIAA1467loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	HEBP1loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	SNORD88loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	GSG1loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	HTR7P1loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	GPRC5Dloss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	MIR614loss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	GPRC5Aloss	196	171	68	[3, 3]	0	1	3.76E-07
# TMEM132Dloss	ETV6loss	196	156	63	[3, 3]	0	1	4.82E-07
# TMEM132Dloss	MANSC1loss	196	164	65	[3, 3]	0	1	5.88E-07
# TMEM132Dloss	DDX47loss	196	172	67	[3, 3]	0	1	7.00E-07
# TMEM132Dloss	APOLD1loss	196	172	67	[3, 3]	0	1	7.00E-07
# TMEM132Dloss	STRAPloss	196	155	62	[3, 3]	0	1	7.66E-07
# TMEM132Dloss	PLBD1loss	196	165	65	[3, 3]	0	1	9.62E-07
# TMEM132Dloss	LOH12CR2loss	196	165	65	[3, 3]	0	1	9.62E-07
# 
# Significant co-occurrence
# TMEM132dLOSS, ccdc91loss
# 
# Can we test more? these were only mutated 3 each in the least mutated guy.
# 
# 
# Conclusion: 70 too small, not enough mutual exclusivity/co-occurrence
# Even when rerunning for cluster size 100. not


__author__ = 'jlu96'
import mutex as mex
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from scipy.stats import poisson
from scipy import stats
import collections
import os

def partition_EM(patientToGenes, k):
    """
    :param geneToCases:
    :param patientToGenes:
    :param k: Number of partitions
    :return: cohort_list
    """

    # partition the patients, and intersect the geneToCases
    return



def partition_gene(patientToGenes, genes):
    """
    :param geneToCases:
    :param patientToGenes:
    :param genes:
    :return: cohorts by each gene. Size 2^(#genes)
    """

    cohorts = [patientToGenes.keys()]
    for gene in genes:
        new_cohorts = []
        for cohort in cohorts:
            new_cohort_1 = [patient for patient in patientToGenes if gene not in patientToGenes[patient]]
            if new_cohort_1:
                new_cohorts.append(new_cohort_1)
            new_cohort_2 = list(set(cohort).difference(set(new_cohort_1)))
            if new_cohort_2:
                new_cohorts.append(new_cohort_2)
        cohorts = new_cohorts
    # print genes
    # print cohorts

    return cohorts

def partition_gene_list(patientToGenes, genes, binary=True):
    """
    :param patientToGenes:
    :param genes:
    :return: The cohorts, ordered from least to greatest in number of those genes they have.
    If binary = True, return just those with, those without.

    """



    gene_set = set(genes)
    cohort_dict = {}

    for patient in patientToGenes:
        num = len(set.intersection(gene_set, patientToGenes[patient]))

        # just 0 and 1
        if binary:
            if num > 0:
                num = 1

        if num not in cohort_dict:
            cohort_dict[num] = []
        cohort_dict[num].append(patient)


    return cohort_dict


def get_patients_gene_mut_num(patients, genes, patientToGenes):
    return [set.intersection(patientToGenes[p], genes) for p in patients]

def integrate_cohorts(cohort_dict, numCases, num_integrated):
    cohorts_int = {}
    start_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > numCases/num_integrated):
            cohorts_int[start_index] = new_cohort
            start_index = i+1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[start_index] = new_cohort

    return cohorts_int

def integrate_cohorts_sizes(cohort_dict, sizes):
    cohorts_int = {}
    size_index = 0
    num_in_cohort = 0
    new_cohort = []
    for i in cohort_dict.keys():
        num_in_cohort += len(cohort_dict[i])
        new_cohort.extend(cohort_dict[i])
        if (num_in_cohort > sizes[size_index]):
            cohorts_int[size_index] = new_cohort
            size_index += 1
            new_cohort = []
            num_in_cohort = 0

    if new_cohort:
        cohorts_int[size_index] = new_cohort

    return cohorts_int


def draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title=None, num_bins=50):
    # LEFT OF HERE, JLU. Finish this, then above. Make plots in parallel, compare.
    # Work with: TP53? Others?

    numGenes = len(geneToCases.keys())
    numCohorts = len(cohort_pairings)

    cohort_frequencies = [[len(patientToGenes[case]) for case in cohort_pair[1]] for cohort_pair in cohort_pairings]
    cohort_names = [cohort_pair[0] for cohort_pair in cohort_pairings]

    draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=title, num_bins=num_bins)


def draw_partitions(patientToGenes, cohort_names, cohort_frequencies, title=None, num_bins=50):

    numCohorts = len(cohort_frequencies)
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)

    plt.figure()


    for i in range(len(cohort_frequencies)):
        plt.hist(cohort_frequencies[i], bins, alpha=1.0/numCohorts, label=str(cohort_names[i]))


    plt.title(title, fontsize=20)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    plt.show()

def norm(x, height, center, std):
    return(height*np.exp(-(x - center)**2/(2*std**2)))



def partition_GMM(patientToGenes, num_components, num_bins, title=None, do_plot=True):
    g = mixture.GMM(n_components=num_components)
    mut_num_list = [len(patientToGenes[p]) for p in patientToGenes]
    obs = np.array([[entry] for entry in mut_num_list])
    g.fit(obs)

    print "***********************************"
    print "COMPONENTS: ", num_components
    print "Weights: " + str(np.round(g.weights_,2))
    print "Means: " + str(np.round(g.means_,2))
    print "Covariates: " + str(np.round(g.covars_,2))

    print "Total log probability: " + str(sum(g.score(obs)))
    print "AIC: " + str(g.aic(obs))
    print "BIC: ", g.bic(obs)

    score, respon = g.score_samples(obs)

    for i in range(num_components):
        print "Model ", np.round(g.means_, 2)[i], " explains ", np.round(len([in_w for in_w in respon if in_w[i] == max(in_w)])) * 1.0 /len(respon)


    # Simulate gaussians
    # sim_samples = g.sample(len(patientToGenes))
    bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
    histogram = np.histogram([len(patientToGenes[p]) for p in patientToGenes], bins=bins)

    # get the scale of the gaussians from the biggest one
    # max_comp = g.weights_.index(max(g.weights_))
    # max_mean = g.means_[max_comp]

    which_bins = [[bin for bin in bins if bin > mean][0] for mean in g.means_]
    print which_bins
    print bins
    print histogram
    print bins.index(which_bins[0]) - 1
    bin_heights = [histogram[0][bins.index(which_bin) - 1] for which_bin in which_bins]
    # max_height = max(histogram)

    if do_plot:
        plt.figure()
        plt.hist([len(patientToGenes[p]) for p in patientToGenes], bins=bins)
        for i in range(num_components):
            X = np.arange(0, max(mut_num_list), 1)
            Y = norm(X, bin_heights[i], g.means_[i], np.sqrt(g.covars_[i]))
            plt.plot(X, Y, label=str(np.round(g.weights_[i], 3)), linewidth=5)
        plt.title("GMM size " + str(num_components), fontsize=20)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.show()
        # draw_partitions(patientToGenes, ['Original', 'Simulated'], [[len(patientToGenes[p]) for p in patientToGenes], sim_samples],
        #                 num_bins=num_bins, title=title)

    data = {}
    data['Components'] = num_components
    data['Weights'] = np.round(g.weights_,2)
    data['Means'] = np.round(g.means_,2)
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    data["AIC"] = g.aic(obs)
    data["BIC"] = g.bic(obs)
    data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data

def partition_gene_kmeans(geneToCases, patientToGenes, gene_list, num_components, num_bins, title=None, do_plot=True):

    # get gene index mapping
    giv = getgiv(geneToCases.keys(), gene_list)

    # convert patients into vectors
    patientToVector = getpatientToVector(patientToGenes, giv)

    vectors = patientToVector.values()

    print vectors[0]
    print "Length of vectors is ", len(vectors[0])

    km = KMeans(num_components)

    km.fit(vectors)

    clusterToPatient = {}

    for patient in patientToVector:
        cluster = km.predict(patientToVector[patient])[0]
        if cluster not in clusterToPatient:
            clusterToPatient[cluster] = set()
        clusterToPatient[cluster].add(patient)

    # plot patients in each cluster


    if do_plot:
        bins = range(0, max([len(p_gene) for p_gene in patientToGenes.values()]), max([len(p_gene) for p_gene in patientToGenes.values()])/num_bins)
        plt.figure()
        for cluster in clusterToPatient:
            plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
        plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.legend()
        plt.title("Kmeans size " + str(num_components), fontsize=20)
        plt.show()



    data = {}
    data['Score'] = km.score(vectors)
    data['Number'] = num_components
    data['% Explained'] = np.round([100 * len(clusterToPatient[cluster]) * 1.0 / len(patientToGenes) for cluster in clusterToPatient], 2)
    data['Vector size'] = len(vectors[0])
    # data['Covariates'] = np.round(g.covars_,2)
    # data["Total log probability"] = sum(g.score(obs))
    # data["AIC"] = g.aic(obs)
    # data["BIC"] = g.bic(obs)
    # data['Explained'] = [np.round([len([in_w for in_w in respon if in_w[i] == max(in_w)]) * 1.0 /len(respon) for i in range(num_components)], 2)]

    return data


def getgiv(all_genes, gene_list):
    """
    :param all_genes:
    :param gene_list:
    :return: A list of the genes in common, the gene_index_vector.
    """
    giv = list(set(all_genes).intersection(set(gene_list)))

    return giv



def getpatientToVector(patientToGenes, gene_index_vector):
    patientToVector = {}
    for patient in patientToGenes:
        patient_genes = patientToGenes[patient]
        patientToVector[patient] = []
        for gene in gene_index_vector:
            patientToVector[patient].append(1 if gene in patient_genes else 0)

    return patientToVector


def get_cluster_gTC_pTG(geneToCases, patientToGenes, patients):
    new_pTG = dict([c for c in patientToGenes.items() if c[0] in patients])
    new_genes = set.union(*new_pTG.values())
    new_gTC = dict([g for g in geneToCases.items() if g[0] in new_genes])
    for g in new_gTC:
        new_gTC[g] = new_gTC[g].intersection(patients)
    
    for g in new_genes:
        if g in new_gTC and not new_gTC[g]:
            new_gTC.pop(g)
    
    new_genes = new_genes.intersection(set(new_gTC.keys()))
    
    return new_genes, new_gTC, new_pTG










# 3/12/16-Jlu


class PMM:

    def __init__(self, filename=None, delimiter='\t', lam=None, p_k=None, classes=None, patientToGenes=None,
                data = None, clusterToPatient = None, do_fit=True):

        if filename:
            with open(filename, 'rU') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                row = reader.next()
                print row
                self.lam = eval(row['Means'])
                self.p_k = eval(row['Probabilities'])
                self.classes = eval(row['Classes']) if 'Classes' in row else range(len(self.lam))
                self.num_components = len(self.classes)
        else:
            self.lam = lam
            self.p_k = p_k
            self.classes = classes
            if not classes:
                self.classes = range(len(self.lam))
            self.num_components = len(self.classes)


        self.data = data
        self.clusterToPatient = clusterToPatient
        print "Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()

        self.patientToGenes = patientToGenes

        if patientToGenes and do_fit:
            self.fit_to_data(patientToGenes)

    def fit_to_data(self, patientToGenes, min_cluster_size=0):
        self.patientToGenes = patientToGenes
        self.data, self.clusterToPatient = pmm_fit_to_data(patientToGenes, classes=self.classes, lam=self.lam, p_k=self.p_k,
                                                           min_cluster_size=min_cluster_size)
        return self.data, self.clusterToPatient


    def plot_clusters(self, title):
        plot_pmm_clusters(self.patientToGenes, self.clusterToPatient, self.num_components, title=title)


    def write_clusters(self, partition_file):
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Likelihood', self.data['Likelihood']])
            writer.writerow(['BIC', self.data['BIC']])
            writer.writerow(['NumComponents', self.data['Number']])
            writer.writerow(['Cluster', 'Lambda', 'Probability', 'Patients'])
            for k in self.clusterToPatient:
                if k != -1:
                    lam = self.data['Means'][k]
                    p_k = self.data['Probabilities'][k]
                else:
                    lam = None
                    p_k = None
                writer.writerow([k, lam, p_k] + list(self.clusterToPatient[k]))

    def compare_dna(self, dna_cohort_dict, do_KS=False):

        partition_stats_list = []

        sizes = [len(self.clusterToPatient[c]) for c in self.clusterToPatient]

        # partition by genes
        dna_cohorts = integrate_cohorts_sizes(dna_cohort_dict, sizes)

        pmm_cluster_list = []
        dna_cluster_list = []
        
        print "In partition stats Class is ", self.classes, "Keys are ", self.clusterToPatient.keys()
        
        for i in range(len(self.classes)):
            partition_stats = collections.OrderedDict()
            partition_stats['Class'] = self.classes[i]
            partition_stats['Mean'] = self.lam[i]
            partition_stats['Probability'] = self.p_k[i]


            partition_stats['PMM_patients'] = self.clusterToPatient[self.classes[i]]
            partition_stats['DNA_patients'] = dna_cohorts[i]

            pmm_cluster_list.append(partition_stats['PMM_patients'])
            dna_cluster_list.append(partition_stats['DNA_patients'])
            
            dna_pmn = [len(self.patientToGenes[p]) for p in partition_stats['DNA_patients']]
            pmm_pmn = [len(self.patientToGenes[p]) for p in partition_stats['PMM_patients']]

            if do_KS:
                poisson_cdf.mu = self.lam[i]
                partition_stats['KS'] = stats.kstest(dna_pmn, poisson_cdf)

            #qq plot of the dna and then the poisson
            poisson_q = get_quantiles(dna_pmn, pmm_pmn)
            dna_q = get_quantiles(dna_pmn, dna_pmn)

            plot_pmm_clusters(self.patientToGenes, {'PMM': partition_stats['PMM_patients'], 'DNA': partition_stats['DNA_patients'] },
                              2, num_bins=100, title='DNA VS PMN')

            plt.figure()
            plt.plot(dna_q, poisson_q, 'bo')
            plt.plot([0, 100], [0,100], 'r-', label = 'y=x')
            plt.title('QQ for ' + str(self.classes[i]), fontsize=20)
            plt.xlabel('DNA_Q', fontsize=20)
            plt.ylabel('PMM_Q', fontsize=20)
            plt.legend()
            plt.show()

            partition_stats_list.append(partition_stats)

        if do_KS:
            self.data['KS_geom_mean'] = mex.prod([partition_stats['KS'][1] for partition_stats in partition_stats_list]) ** (1.0/ len(partition_stats_list))

            print "KS average is ", self.data['KS_geom_mean']
            
        self.data['CohenKappa'] = cohen_kappa(pmm_cluster_list, dna_cluster_list)


        return partition_stats_list



def cohen_kappa(cluster_list_1, cluster_list_2):
    # assume same categories each
    num_agree = 0
    prob_agree = 0
    total = len(set.union(*[set(c) for c in cluster_list_1]))
    
    num_classes = len(cluster_list_1)
    
    cluster_list_1 = [set(c) for c in cluster_list_1]
    cluster_list_2 = [set(c) for c in cluster_list_2]
    
    for k in range(num_classes):
        a = cluster_list_1[k]
        b = cluster_list_2[k]
        num_agree += len(a.intersection(b))
        prob_agree += (len(a) * len(b) * 1.0) / (total ** 2)
    

    obs_agree = num_agree * 1.0 / total
    
    ck = (obs_agree - prob_agree)/(1.0 - prob_agree)
    
    print "Number agreements ", num_agree
    print "Total ", total
    print "Prob agreements ", prob_agree
    print "Cohen kappa ", ck
    
    return ck
        
    
    




def poisson_cdf(x):
    if not hasattr(poisson_cdf, 'mu'):
        poisson_cdf.mu = 0
    print "X is ", x, "and mu is ", poisson_cdf.mu
    return poisson.cdf(x, poisson_cdf.mu)

def get_quantiles(test_dist, base_dist):
    return [stats.percentileofscore(base_dist, t) for t in test_dist]

def assign_missing(clusterToPatient, patientToGenes):
    if -1 not in clusterToPatient:
        print "No missing patients in clusters"
        return clusterToPatient
    missing_patients = clusterToPatient[-1]
    cluster_means = [(sum([len(patientToGenes[p]) for p in clusterToPatient[c]]) * 1.0 /len(clusterToPatient[c]), c) for c in clusterToPatient if c != -1]
    print cluster_means, cluster_means[0][0]
    for patient in missing_patients:
        num = len(patientToGenes[patient])
        correct_cluster = sorted(cluster_means, key=lambda entry: abs(num - entry[0]))[0][1]
        clusterToPatient[correct_cluster].add(patient)
    clusterToPatient.pop(-1)

    return clusterToPatient



def best_pmm(patientToGenes, num_components, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0,
             plot_clusters=True):

    data_record = []
    lls_record = []

    # Do normal
    first_data, lls = partition_pmm(patientToGenes, num_components,  max_iter=max_iter, min_cluster_size=min_cluster_size)

    data_record.append(first_data)
    lls_record.append(lls)

    # Do best rand init
    for i in range(rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    for i in range(far_rand_num):
        data, lls = partition_pmm(patientToGenes, num_components, far_rand_init=True, max_iter=max_iter, min_cluster_size=min_cluster_size,
                                 verbose=False)
        data_record.append(data)
        lls_record.append(lls)

    combined_record = zip(data_record, lls_record)

    combined_record = sorted(combined_record, key=lambda entry: (-1 * entry[0]['Missing'], entry[0]['Likelihood']), reverse=True)

    data_record, lls_record = zip(*combined_record)

    best_data = data_record[0]

    if (best_data['Likelihood'] > first_data['Likelihood'] + 10):
        print "First data not best!"
        best_data['IsFirst'] = False
    else:
        best_data['IsFirst'] = True


    clusterToPatient = pmm_to_cluster(patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

    if plot_clusters:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components)

    plot_likelihoods(lls_record)

    return best_data, clusterToPatient
    # Return clusters


def pmm_to_cluster(patientToGenes, classes, lam, p_k):
    clusterToPatient = {}

    for k in classes:
        clusterToPatient[k] = set()

    clusterToPatient[-1] = set()


    for patient in patientToGenes:
        d = len(patientToGenes[patient])

        max_class = -1
        max_ll = -np.inf
        for k in classes:
            if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > -np.inf:
                if (np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d))) > max_ll:
                    max_class = k
                    max_ll = (np.log(poisson(lam[k]).pmf(d)))


        clusterToPatient[max_class].add(patient)

    missing_clusters = set()
    for cluster in clusterToPatient:
        if not clusterToPatient[cluster]:
            print '**********NO PATIENTS IN CLUSTER ', lam[cluster], p_k[cluster]
            missing_clusters.add(cluster)
            #clusterToPatient[cluster].add('NO PATIENTS IN CLUSTER')
    for cluster in missing_clusters:
        clusterToPatient.pop(cluster)
            
    return clusterToPatient



def pmm_cross_validate(num_components, patientToGenes, num_folds, kf_random_state=None, max_iter=30, rand_num=5, far_rand_num=5, min_cluster_size=0):
    """
    :return: The average likelihood of the model when applied to a new test set, and its BIC
    """

    kf = KFold(len(patientToGenes), n_folds=num_folds, random_state=kf_random_state)

    lls = []
    missing_patients = []
    bics = []
    for train_index, test_index in kf:

        train_patientToGenes = dict([patientToGenes.items()[x] for x in train_index])
        test_patientToGenes = dict([patientToGenes.items()[x] for x in test_index])
        best_data, _ = best_pmm(train_patientToGenes, num_components, max_iter=max_iter, rand_num=rand_num,
                                               far_rand_num=far_rand_num, min_cluster_size=min_cluster_size)

        test_stats, test_cluster = pmm_fit_to_data(test_patientToGenes, best_data['Classes'], best_data['Means'], best_data['Probabilities'])

        plot_pmm_clusters(test_patientToGenes, test_cluster, num_components, title='Test clusters size ' + str(num_components))

        lls.append(test_stats['Likelihood'])
        missing_patients.append(test_stats['Missing'])
        bics.append(test_stats['BIC'])

    return sum(lls) * 1.0/len(lls), sum(missing_patients) * 1.0 / len(missing_patients), sum(bics) * 1.0/ len(bics)





def pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=None, min_cluster_size=0):
    """
    :param patientToGenes:
    :param lam:
    :param p_k:
    :param data:
    :return: data, clusterToPatient
    """

    if not data:
        data = collections.OrderedDict()


    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    num_components = len(lam)

    ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])
    likelihood_sums = np.zeros(numCases)

    for i in range(numCases):
        likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

    # complete log likelihood

    ll = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

    clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    print "LL:", np.round(ll), "Missing patients: ", len(clusterToPatient[-1]) if -1 in clusterToPatient else 0

    data['Number'] = num_components
    data['OriginalNumber'] = num_components
    mp = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))

    data['Means'], data['Probabilities'] =  list(mp[0]), list(mp[1])   
    data['Likelihood'] = np.round(ll)
    data['Classes'] = classes
    data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0
    data['Merged'] = False
    data['MergeHistory'] = set()

    return data, clusterToPatient




def partition_pmm(patientToGenes, num_components, diff_thresh=0.01, num_bins=50, max_iter=100, by_iter=True,
                  rand_init=False, far_rand_init=False, do_plot=False, get_best=True, min_cluster_size=0,
                 verbose=True):


    # get the whole data distribution


    # D = [1,2,3,4,5, 100, 150, 200, 1000]
    D = [len(patientToGenes[p]) for p in patientToGenes]
    numCases = len(D)
    data = collections.OrderedDict()

    # print "D is ", D

    # get the lambdas at equal-spaced intervals


    lam = [np.percentile(D, (i + 1) * 100.0 / (num_components + 1)) for i in range(num_components)]
    p_k = [1.0 / num_components for i in range(num_components)]
    classes = range(num_components)

    if rand_init:
        old_lam = lam
        old_p_k = p_k
        #random sample  in a range centered at the quartiles
        lam = [np.random.uniform(l - 0.5 * old_lam[0], l + 0.5 * old_lam[0]) for l in old_lam]
        rand_freq = [2**np.random.uniform(-1, 1) * pk for pk in old_p_k]
        p_k = list(np.array(rand_freq)/sum(rand_freq))
        classes = range(num_components)

    if far_rand_init:
        lam = [np.random.uniform(min(D), max(D)) for l in lam]
        rand_freq = [np.random.uniform(0, 1) for l in lam]
        p_k = list(np.array(rand_freq)/sum(rand_freq))

    if verbose:
        print "Initial Lambda is ", lam
        print "Initial p_k is", p_k

    data['Initial Means'] = np.round(lam,1)
    data['Initial p_k'] = np.round(p_k, 2)

    ll = -3e100
    num_iter = 0

    # stupid inital values
    p_k_d= np.zeros(num_components)
    lam_prev = np.zeros(num_components)
    p_k_prev = np.zeros(num_components)

    # for the best values
    ll_best = -np.inf
    p_k_best = None
    lam_best = None
    missing_best = numCases

    lls = []

    while 1:


        # We have the log-likelihood of data d and class k in matrix
        #            data 1 data 2 data 3
        # clsss 1   ll_11   ll_12
        # class 2
        ll_kd = np.array([ [np.log(p_k[k]) + np.log(poisson(lam[k]).pmf(d)) for d in D] for k in classes])

        

        # Likelihood_sums: the total likelihood of each data, summed across class k
        likelihood_sums = np.zeros(numCases)

        for i in range(numCases):
            likelihood_sums[i] = sum([(np.exp(ll_kd[k][i]) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)] )

            
        missing_new = len([x for x in likelihood_sums if x == 0])
        # complete log likelihood

        ll_new = sum(np.log(np.array([ls for ls in likelihood_sums if ls > 0])))

        if num_iter == 0:
            data['Initial LL'] = np.round(ll_new)

        if verbose:
            print "ll_new is ", ll_new, "missing is ", missing_new


        if ll_new > ll_best or missing_new < missing_best:
            ll_best = ll_new
            p_k_best = p_k
            lam_best = lam
            missing_best = missing_new

        # When we break out of the loop, take previous value since it might have jumped out
        if (by_iter):
            if num_iter > max_iter:
                break
            elif abs(ll_new - ll) < diff_thresh:
                break
        else:
            if abs(ll_new - ll) < diff_thresh:

                p_k_d = p_k_d_prev
                lam = lam_prev
                p_k = p_k_prev

            break

        p_k_d_prev = p_k_d
        lam_prev = lam
        p_k_prev = p_k


        # Calculate p_k_d. This is p(data d | class k) * p(class k)/sum(p(data|class i) *p(class i);
        # i.e. prob of this class given this data

        p_k_d = np.zeros(ll_kd.shape)

        for i in range(numCases):
            # Use max class likelihood to divide all the likelihoods by
            max_val = np.amax(ll_kd, axis=0)[i]

            # sum the likekhoods for every class, make this the denominator of probability
            denom = sum([(np.exp(ll_kd[k][i] - max_val) if ll_kd[k][i] > -np.inf else 0) for k in range(num_components)])

            for k in range(num_components):
                p_k_d[k][i] = (np.exp(ll_kd[k][i] - max_val) / denom if ll_kd[k][i] > -np.inf else 0)
                # print "numerator is ", np.exp(ll_kd[k][i] - max), " prob is ", p_k_d[k][i]

        # print "p_k_d is ", p_k_d

        # sum probabilities of each data being each class over all data
        Z_k = p_k_d.sum(axis=1)


        # see derivation

        lam = [sum([p_k_d[k][i] * D[i] for i in range(numCases)]) * 1.0 / Z_k[k] for k in classes]
        p_k = Z_k * 1.0 / numCases

        p_k = p_k/p_k.sum()


        # print "New lambda is ", lam
        # print "New p_k is ", p_k


        ll = ll_new

        lls.append(ll)
        num_iter += 1



    if get_best:
        p_k = p_k_best
        lam = lam_best
        ll = ll_best





    data, clusterToPatient = pmm_fit_to_data(patientToGenes, classes, lam, p_k, data=data, min_cluster_size=min_cluster_size)
    # plot patients in each cluster

    if do_plot:
        plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100)


    # clusterToPatient = pmm_to_cluster(patientToGenes, classes, lam, p_k)

    #
    #
    #
    #
    # data['Number'] = num_components
    # data['Means'] = np.round(lam, 1)
    # data['Probabilities'] = np.round(p_k, 2)
    # data['Likelihood'] = np.round(ll)
    # data['Classes'] = classes
    # data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
    # data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
    # data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
    # data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
    # data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

    return data, lls



def sort_data_by_means(data):
    """ Sort in ascending order. Don't need to change cluster labels"""
    data_items = data.items()
    mean_indices = ((i, data['Means'][i]) for i in range(len(data['Means'])))
    mean_indices = sorted(mean_indices, key=lambda entry: min(entry[1]) if isinstance(entry[1], list)
                         else entry[1])
    
    conversion_array = [m[0] for m in mean_indices] # this should map to the correct index now. these are new clusters
    
    new_data = collections.OrderedDict()
    
    for key in data:
        value = data[key]
        if isinstance(value, np.ndarray):
            new_value = np.zeros(len(value))
            for i in range(len(conversion_array)):
                new_value[i] = value[conversion_array[i]]
            new_data[key] = new_value
        if isinstance(value, list):
            new_value = [value[conversion_array[i]] for i in range(len(conversion_array))]
            new_data[key] = new_value
            
        else:
            new_data[key] = value
    
    return new_data
    

def merge_clusters(data, clusterToPatient, patientToGenes,
                  missing_limit=0.5, min_cluster_size=30):
    """Merge adjacent clusters. Choosse to merge those clusters that
    are the most similar, as measured by the likelihood of one within
    another.
    missing_limit is the limit on number of patients that can't
    be explained by one cluster. Clusters will be sorted first
    by those who are below the minimum cluster size,
    less missing patients in their merging
    cluster, then by those that have the highest likelihood
    """
    # get the likelihood of each cluster rel. to other ones
    # only look at adjacent clusters! sort them
    
    data = sort_data_by_means(data)
    
    print "****************************************"
    print "Begin merging."
    # first go forward

    
    classes = data['Classes']
    p_k = data['Probabilities']
    lam = data['Means']
    
    
    all_list = []
    
    for i in range(len(lam) - 1):
        from_index, to_index = i, i + 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
        #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
            
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num > missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    # now go backwards
    for i in reversed(range(1, len(lam))):
        from_index, to_index = i, i - 1
        from_class, to_class = classes[from_index], classes[to_index]
        patients = clusterToPatient[from_class]
        p = [len(patientToGenes[patient]) for patient in patients]
        
                #check if we're dealing with merged clusters. if so... add the likelihoods of the individual
        # underlying poissons?
        if isinstance(p_k[from_index], list):
            clust_probs = p_k[from_index]
            clust_means = lam[from_index]
            clust_size = len(clust_means)
            
            from_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            from_ll = [np.log(p_k[from_index]) + np.log(poisson(lam[from_index]).pmf(d)) for d in p]
            
        if isinstance(p_k[to_index], list):
            clust_probs = p_k[to_index]
            clust_means = lam[to_index]
            clust_size = len(clust_means)
            
            to_ll = [max([np.log(clust_probs[x]) + 
                           np.log(poisson(clust_means[x]).pmf(d)) for x in range(clust_size)])
                          for d in p]
        else:
            to_ll = [np.log(p_k[to_index]) + np.log(poisson(lam[to_index]).pmf(d)) for d in p]
        
        
        missing = np.isinf(from_ll) ^ np.isinf(to_ll)
        
        missing_indices = np.where(missing)[0]
        good_indices = np.where(~missing)[0]
        
        missing_num = len(missing_indices)
        
        ll_diffs = [to_ll[j] - from_ll[j] for j in good_indices]
        
        ll_diffs_total = sum(ll_diffs)
        
        
        all_list.append([(from_index, to_index), missing_num, ll_diffs_total, missing_num < missing_limit * len(p),
                        len(patients) < min_cluster_size])
        
    
    # sort by the cluster that's below the min size, then byminimum missing, then by maximum likelihood ratio
    all_list = sorted(all_list, key=lambda entry: (entry[4], entry[3], entry[2]), reverse=True)
    
    print "Possible merged clusters is ", all_list
    print "Best cluster is ", all_list[0]
    

    (from_index, to_index), missing_num, ll_diffs_total, more_than_missing, cluster_too_small = all_list[0]

    # calculate the new AIC, BIC, make new cluster to patient, make new classes..new means? update probabilities
    
    # Record merge history
    new_data = data
    if 'MergeHistory' not in new_data:
        new_data['MergeHistory'] = set()
    
    new_data['MergeHistory'].add((str([lam[from_index], lam[to_index]]),
                  str([p_k[from_index], p_k[to_index]]),
                  (len(clusterToPatient[classes[from_index]]), len(clusterToPatient[classes[to_index]])),
                  missing_num, ll_diffs_total, ('Num classes befpre', len(classes), ('Cluster too small?', cluster_too_small))))
        
    new_clusterToPatient = clusterToPatient
    moved_patients = new_clusterToPatient[classes[from_index]]
    new_clusterToPatient[classes[to_index]] = new_clusterToPatient[classes[to_index]].union(moved_patients)
    new_clusterToPatient.pop(classes[from_index])

    
    print "MERGING the probs and likelihoods"
    if not isinstance(p_k[from_index], list):
        p_k[from_index] = [p_k[from_index]]
        lam[from_index] = [lam[from_index]]
    if not isinstance(p_k[to_index], list):
        p_k[to_index] = [p_k[to_index]]
        lam[to_index] = [lam[to_index]] 
    p_k[to_index].extend(p_k[from_index])
    lam[to_index].extend(lam[from_index])
    new_data['Probabilities'] = p_k
    new_data['Means'] = lam
    
    
    print "MERGING: HERE ARE OLD VALUES", new_data
    #remove all the old values
    new_data['Merged'] = True
    new_data['Number'] -= 1
    for key in new_data:
        value = new_data[key]
        if isinstance(value, np.ndarray):
            value = list(value)
            value = value[0: from_index] + value[from_index + 1 :]
            value = np.array(value)
            new_data[key] = value
        elif isinstance(value, list):
            value = value[0: from_index] + value[from_index + 1 :]
            new_data[key] = value

    print "New classe:", new_data['Classes'], "VS NEW KEYS", new_clusterToPatient.keys()
            
    # integrate the old patients to the new ones

    
    
    new_data['MinClusterSize'] = min(len(new_clusterToPatient[c]) for c in new_clusterToPatient)
    
    print "MERGING: HERE ARE NEW VALUES", new_data
    
    plot_pmm_clusters(patientToGenes, clusterToPatient, new_data['Number'], title='Merging')
    
    print "End merging."
    print "****************************************"    
    
    return new_data, new_clusterToPatient
 
    
#     data['Number'] = num_components
#     data['Means'], data['Probabilities'] = zip(*sorted(zip(list(np.round(lam, 1)), list(np.round(p_k, 2))), key = lambda entry: entry[0]))
#     data['Likelihood'] = np.round(ll)
#     data['Classes'] = classes
#     data['AIC'] = np.round(2 * (len(p_k) + len(lam)) - 2 * ll)
#     data['BIC'] = np.round(-2 * ll + (len(p_k) + len(lam)) * np.log(numCases))
#     data['Missing'] = len(clusterToPatient[-1]) if -1 in clusterToPatient else 0
#     data['MinClusterSize'] = min([len(clusterToPatient[c]) if c != -1 else np.inf  for c in clusterToPatient])
#     data['MoreThanMin'] = 1 if data['MinClusterSize'] > min_cluster_size else 0

def backward_selection(data, clusterToPatient, patientToGenes, min_cluster_size = 30,
                       max_components = 10):
    """Merge clusters until a criterion is satisfied. Missing patients are assumed to
    be assigned already.
    """
    

    merged_data = data
    merged_cluster = clusterToPatient
    
    while (merged_data['Number'] > max_components or merged_data['MinClusterSize'] < min_cluster_size):
        merged_data, merged_cluster = merge_clusters(merged_data, merged_cluster, patientToGenes,
                                                    min_cluster_size = min_cluster_size)
    
    return merged_data, merged_cluster
    







def plot_pmm_clusters(patientToGenes, clusterToPatient, num_components, num_bins=100, title=None):
    D = [len(patientToGenes[p]) for p in patientToGenes]

    bins = range(0, max(list(D)), max(list(D))/num_bins)
    plt.figure()
    for cluster in clusterToPatient:
        plt.hist([len(patientToGenes[p]) for p in clusterToPatient[cluster]], bins=bins, label=str(cluster), alpha = 1.0/num_components)
    plt.xlabel('# Somatic Mutations In Tumor', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend()
    if not title:
        plt.title("Cluster size " + str(num_components), fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.show()

def plot_likelihoods(ll_record):
    plt.figure()
    for i in range(len(ll_record)):
        plt.plot(ll_record[i], label=str(i))
    plt.title("Log-likelihood change in EM", fontsize=20)
    plt.legend(loc=4)
    plt.show()

# If there are any patients that aren't assigned, i.e. in cluster -1
# Throw them out?
def load_patient_cohorts(partitionfile, patientToGenes, add_to_closest=True, delimiter='\t'):
    clusterToProp = {}

    with open(partitionfile, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if (row[0] == 'Cluster'): break
        # reader = csv.DictReader(csvfile, delimiter=delimiter)
        # print "begun dict reader\n"
        for row in reader:
            c = eval(row[0])
            clusterToProp[c] = {}
            clusterToProp[c]['Mean'] = eval(row[1]) if row[1] else 0
            clusterToProp[c]['Probability'] = eval(row[2]) if row[2] else 0
            clusterToProp[c]['Patients'] = set(row[3:]) if row[3] else set()


    if -1 in clusterToProp:
        if add_to_closest:
            other_cs = clusterToProp.keys()
            other_cs.remove(-1)
            print "Removed ", clusterToProp[-1]
            for patient in clusterToProp[-1]:
                sims = [(abs(len(patientToGenes[patient]) - clusterToProp[c]['Mean']), c) for c in other_cs]
                sims = sorted(sims, key = lambda entry: entry[0])
                best_c = sims[0][1]
                clusterToProp[best_c]['Patients'].add(patient)
            print "completed"

        clusterToProp.pop(-1)

    sorted_clusters = sorted(clusterToProp.keys(), key = lambda entry: clusterToProp[entry]['Mean'])
    
    oldclusterToProp = clusterToProp.copy()
    clusterToProp = {}
    cohort_dict = {}
    
    for i in range(len(sorted_clusters)):
        cohort_dict[i] = oldclusterToProp[sorted_clusters[i]]['Patients']
        clusterToProp[i] = oldclusterToProp[sorted_clusters[i]]
    
    min_cohort = cohort_dict[0]
    
    
    
    return cohort_dict, clusterToProp, min_cohort

# INDEX BY LOSSES
get_ipython().magic('matplotlib inline')
def run_partitions(mutationmatrix = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.m2',
        patientFile = None, #'/Users/jlu96/maf/new/OV_broad/shared_patients.plst',
        out_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-crossval.txt',
        partition_file = None, #'/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl.ppf',
        load_pmm_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM.txt',
        dna_pmm_comparison_file = None, #'/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/OV_broad-cna-jl-PMM-dnacomp.txt',
        cluster_matrix = None, # '/Users/jlu96/maf/new/OV_broad/OV_broad-cna-jl-cluster.m2',
        min_cluster_size = 15,
        num_init = 9,
        minComp = 2,
        maxComp = 5,
        do_plot = True,
        do_gmm = False,
        do_dna = False,
        num_integrated = 4,
        do_kmeans = False,
        do_pmm = True,
        do_cross_val = False,
        do_pmm_dna = True,
        do_back_selection = True,
        write_cluster_matrices = True,
        rand_num = 3,
        far_rand_num = 3,
        kf_random_state = 1,
        kf_num_folds = 5,

        geneFile = None,
        minFreq = 0,
        dna_gene_file = '/Users/jlu96/conte/jlu/Analyses/CancerGeneAnalysis/DNADamageRepair_loss.txt',
       out_dir = '/Users/jlu96/conte/jlu/Analyses/CancerMutationDistributions/',
        write_all_partitions = True):
    
    mutationmatrix_list = mutationmatrix.split('/')
    matrix_dir = '/'.join(mutationmatrix_list[:-1]) + '/'
    prefix = (mutationmatrix_list[-1]).split('.m2')[0]
    

    if not patientFile:
        patientFile = matrix_dir + 'shared_patients.plst'
        
    if not out_file:
        if do_cross_val:
            out_file = out_dir + prefix + '-PMM-crossval-kf' + str(kf_num_folds) + '.txt'
        else:
            out_file = out_dir + prefix + '-PMM-comparisons.txt'
    
    if not partition_file:
        partition_file = matrix_dir + prefix + '.ppf'
        
    
    if not load_pmm_file:
        load_pmm_file = out_dir + prefix + '-PMM.txt'
    
    if not dna_pmm_comparison_file:
        dna_pmm_comparison_file = out_dir + prefix + '-PMM-dnacomp.txt'
        
    if not cluster_matrix:
        cluster_matrix = matrix_dir + prefix + '-cluster.m2'

    
    numGenes, numCases, genes, patients, geneToCases, patientToGenes = mex.load_mutation_data(mutationmatrix, patientFile, geneFile, minFreq)

    p_gene_list = []

    with open(dna_gene_file, 'rU') as row_file:
        reader = csv.reader(row_file, delimiter='\t')
        for row in reader:
            p_gene_list.append(row[0])
        dna_cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))


    if do_kmeans:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_gene_kmeans(geneToCases, patientToGenes, p_gene_list, i, num_bins=50, title=None, do_plot=True))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_dna:
        cohort_dict = partition_gene_list(patientToGenes, p_gene_list, binary=not bool(num_integrated))
        # Make new cohorts over this
        if num_integrated:
            cohort_dict = integrate_cohorts(cohort_dict, numCases, num_integrated)


        cohort_pairings = [(key, cohort_dict[key]) for key in cohort_dict]
        draw_partitions_cohorts(geneToCases, patientToGenes, cohort_pairings, title='DNADamageGenes',
                        num_bins=100 if mutationmatrix[-9:] == 'cna-jl.m2' else 50)


    if do_gmm:
        datas = []
        for i in np.arange(minComp, maxComp, 1):
            datas.append(partition_GMM(patientToGenes, i, num_bins=50, title='GMM size ' + str(i), do_plot=do_plot))

        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[0].keys())
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


    if do_pmm:
        datas = []
        clusters = []

        partition_stats_list = []
        for num_components in np.arange(minComp, maxComp, 1):
            best_data, clusterToPatient = best_pmm(patientToGenes, num_components, rand_num=rand_num, far_rand_num=far_rand_num,
                                                   min_cluster_size=min_cluster_size)

            if do_back_selection:
                # assign the missing data
                clusterToPatient = assign_missing(clusterToPatient, patientToGenes)
                best_data, clusterToPatient = backward_selection(best_data, clusterToPatient, patientToGenes, min_cluster_size = min_cluster_size,
                       max_components = maxComp)
            
            if do_pmm_dna:
                print "cfirst lasses are ", best_data['Classes'], "clusterToPatient is ", clusterToPatient.keys()
                pmm = PMM(lam=best_data['Means'], p_k=best_data['Probabilities'], patientToGenes=patientToGenes,
                         data=best_data, clusterToPatient=clusterToPatient, classes=best_data['Classes'],
                          do_fit=False)

                partition_stats_list.extend(pmm.compare_dna(dna_cohort_dict))

                best_data = pmm.data


            if do_cross_val:
            #cross validate each of the components
                print "*******************************************************************************************************"
                print "BEGINNING CROSS VALIDATION for ", num_components
                print "*******************************************************************************************************"
                best_data['TestLL'], best_data['TestMissing'], best_data['TestBIC'] = pmm_cross_validate(num_components, patientToGenes,
                                                                                                         num_folds=kf_num_folds,
                                                                                                     kf_random_state=kf_random_state,
                                                                                   rand_num=rand_num, far_rand_num=far_rand_num,
                                                                                   min_cluster_size=min_cluster_size)
                best_data['TestFolds'] = kf_num_folds

                print "*******************************************************************************************************"
                print "EMDING CROSS VALIDATION  for ", num_components
                print "*******************************************************************************************************"

            datas.append(best_data)
            clusters.append(clusterToPatient)
            
            if write_all_partitions:
                with open(partition_file + str(num_components), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')

                    writer.writerow(['Likelihood', best_data['Likelihood']])
                    writer.writerow(['BIC', best_data['BIC']])
                    writer.writerow(['NumComponents', best_data['Number']])
                    writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
                    if 'Merged' in best_data and best_data['Merged']:
                        for k in range(len(clusterToPatient)):
                            lam = best_data['Means'][k]
                            p_k = best_data['Probabilities'][k]
                            writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
                    else:
                        for k in clusterToPatient:
                            if k != -1:
                                lam = best_data['Means'][k]
                                p_k = best_data['Probabilities'][k]
                            else:
                                lam = None
                                p_k = None
                            writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        # get the best BIC
        combined = zip(datas, clusters)
        if do_cross_val:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], np.round(entry[0]['TestMissing']), -1 * entry[0]['TestLL'], entry[0]['TestBIC'], entry[0]['BIC']))
        else:
            combined = sorted(combined, key=lambda entry: ( -1 * entry[0]['MoreThanMin'], entry[0]['BIC']))

        datas, clusters = zip(*combined)




        with open(out_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=datas[-1].keys(), delimiter='\t', extrasaction='ignore')
            print datas
            writer.writeheader()
            for row in datas:
                writer.writerow(row)


        best_data = datas[0]
        clusterToPatient = clusters[0]

        # code to parition by best clusters
        with open(partition_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            writer.writerow(['Likelihood', best_data['Likelihood']])
            writer.writerow(['BIC', best_data['BIC']])
            writer.writerow(['NumComponents', best_data['Number']])
            writer.writerow(['Cluster', 'Mean', 'Probability', 'Patients'])
            if 'Merged' in best_data and best_data['Merged']:
                for k in range(len(clusterToPatient)):
                    lam = best_data['Means'][k]
                    p_k = best_data['Probabilities'][k]
                    writer.writerow([best_data['Classes'][k] , lam, p_k]  + list(clusterToPatient[best_data['Classes'][k]]))
                        
            else:
                for k in clusterToPatient:
                    if k != -1:
                        lam = best_data['Means'][k]
                        p_k = best_data['Probabilities'][k]
                    else:
                        lam = None
                        p_k = None
                    writer.writerow([k, lam, p_k] + list(clusterToPatient[k]))

        if write_cluster_matrices:
            for cluster in clusterToPatient:
                with open(cluster_matrix + str(cluster), 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    for patient in clusterToPatient[cluster]:
                        writer.writerow('\t'.join([patient] + list(patientToGenes[patient])))


        if do_pmm_dna:
            with open(dna_pmm_comparison_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=partition_stats_list[0].keys(), delimiter='\t')
                writer.writeheader()
                print "header written"
                for row in partition_stats_list:
                    writer.writerow(row)





