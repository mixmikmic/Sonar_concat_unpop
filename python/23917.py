get_ipython().magic('matplotlib inline')
from __future__ import division

import pandas as pd
import numpy as np

from biom import load_table

import matplotlib.pyplot as plt


# Download the open reference OTU table.
# 

get_ipython().system('scp barnacle:/home/yovazquezbaeza/research/suchodolski-dogs/open-ref-otus/otu_table_mc2_w_tax.biom  counts/open-ref-table.biom')


get_ipython().system('biom summarize-table -i counts/open-ref-table.biom -o counts/open-ref-table-summary.txt')


# Load the summaries from split libraries and the OTU table to get a proportion of the sequences assigned to an OTU. Note that the original output formats are not Pandas friendly, therefore I used vim to trimmed out the lines that were not needed at all.
# 

sl = pd.read_csv('counts/split_library_log.txt', sep='\t', index_col='#SampleID')
ot = pd.read_csv('counts/table-summary.txt', sep='\t', index_col='#SampleID')
ot_open = pd.read_csv('counts/open-ref-table-summary.txt', sep='\t', index_col='#SampleID')


tot = pd.DataFrame(index=sl.index, columns=['Sequences', 'OTU_Counts', 'OTU_Counts_open'])


tot['Sequences'] = sl.Count
tot['OTU_Counts'] = ot.Count
tot['OTU_Counts_open'] = ot_open.Count

tot['Percent'] = np.divide(tot.OTU_Counts, tot.Sequences *1.0) * 100
tot['Percent_open'] = np.divide(tot.OTU_Counts_open, tot.Sequences *1.0) * 100


# # Examine the percents of sequences assigned to an OTU, both for open and closed reference.
# 

tot['Percent'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (closed reference)')
plt.ylabel('Counts')
plt.xlim([0, 100])

plt.title('Entire Cohort')


tot['Percent_open'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (open reference)')
plt.ylabel('Counts')
plt.xlim([0, 100])

plt.title('Entire Cohort')


# # Look only at the samples that were used for analysis
# 
# Note that during our entire analysis we excluded the samples from the dogs that had accute hemorragic diarrhea. Also note that a few samples are dropped since they don't accrue the minimum of 15,000 sequences. All this can be summarized by looking at the core OTU table we used.
# 

bt = load_table('otu_table.15000.no-diarrhea.biom')

sub = tot.loc[bt.ids('sample')].copy()

sub['Percent'] = np.divide(sub.OTU_Counts, sub.Sequences *1.0) * 100
sub['Percent_open'] = np.divide(sub.OTU_Counts_open, sub.Sequences *1.0) * 100


# Now visualize this data.
# 

sub['Percent'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU')
plt.ylabel('Counts')
plt.xlim([70, 100])
plt.title('Samples used for Analysis')


sub['Percent_open'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (Open Reference)')
plt.ylabel('Counts')
plt.xlim([70, 100])
plt.title('Samples used for Analysis')


sub.Percent.mean()


sub.Percent_open.mean()


# This last bit is particularly revealing of the minor gains that we get from using the open reference OTU picking protocol. The mean percent of reads mapped to an OTU went from 94.2 % (in closed reference) to 99.14 % (in open reference). Furthermore in the original analysis performed in Gevers et al 2014, a closed reference OTU picking approach was used, therefore to directly compare co-occurrence networks between analyses would be incorrect.
# 




# This link contains information on CCREPE:
#     
#     https://www.bioconductor.org/packages/devel/bioc/vignettes/ccrepe/inst/doc/ccrepe.pdf
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rmagic')

from __future__ import division

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from biom import load_table
from scipy.stats import mannwhitneyu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


# # Dysbiosis index at level 6
# 

# We filter the table for low abundnce OTUs to only consider well represented features.
# 

get_ipython().system('summarize_taxa.py -i otu_table.15000.25percent.biom -o stats/group-significance/taxa-summaries-25pct')


get_ipython().run_cell_magic('R', '', '\nlibrary("ccrepe")\n\notus <- read.table("stats/group-significance/taxa-summaries-25pct/otu_table.15000.25percent_L6.txt",\n                   sep="\\t", header=TRUE, skip=1, comment.char=\'\')\nrownames(otus) <- otus$X.OTU.ID\notus$X.OTU.ID <- NULL\n\notus.score <- ccrepe(x=t(otus), iterations=1000, sim.score=nc.score)\n\nwrite.table(otus.score$sim.score,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$z.stat,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$p.values,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$q.values,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')')


# # Visualize these relationships as a heatmap
# 

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

plt.figure(figsize=(50, 50))

g = sns.heatmap(otus_score, mask=mask, annot=True, fmt=".2f",
            annot_kws={'fontdict': {'fontsize': 8}})

plt.savefig('stats/group-significance/no-diarrhea/ccrepe/ccrepe-otu_table.filtered.25pct_L6.pdf')


get_ipython().system('group_significance.py -i stats/group-significance/taxa-summaries-25pct/otu_table.15000.25percent_L6.biom -o stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt -m mapping-file-full.txt --category disease_stat')


# # Visualize the data as a network
# 

import networkx as nx

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 'g'
    else:
        return 'r'
colors = kw_stats.apply(color_funk, axis=1, reduce=False)

plt.figure(figsize=(20, 20))
G = nx.from_numpy_matrix(~mask.values)
G = nx.relabel_nodes(G, {i: o for i, o in enumerate(mask.index.tolist())})
G.remove_nodes_from([n for n in G.nodes_iter() if len(G.edges(n)) == 0])

for e in G.edges_iter():
    u, v = e
    weight = otus_score.loc[u][v]
    if weight > 0 :
        relation = 'coccurrence'
    else:
        relation = 'coexlcusion'

    G.add_edge(u, v, weight=weight, relation=relation)

nx.spring_layout(G)

#nx.draw(G, node_list=colors.index.tolist(), node_color=colors.tolist(), node_name=kw_stats.index.tolist())
nx.draw(G, node_list=colors.index.tolist(), node_color=colors.tolist())


# ## Create these files to make importing in Cytoscape easier
# 

node_attrs = pd.DataFrame()

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 'protective'
    else:
        return 'inflammatory'
node_attrs['role'] = kw_stats.apply(color_funk, axis=1, reduce=False)

def short_name(row):
    #f__Planococcaceae;g__
    n = row.name.split('f__')[1]
    n = n.replace(';g__', ' ')
    
    if n.strip() == '':
        n = row.name.split('o__')[1].split(';')[0]
    return n
node_attrs['short_name'] = kw_stats.apply(short_name, axis=1, reduce=False)


node_attrs.to_csv('node-attributes.txt')


nx.write_edgelist(G, 'test.edgelist.2.txt', data=True)


# # Largest connected component only
# 
# From the inferred network, filter to preserve only the largest connected component of the graph.
# 

import networkx as nx
import igraph

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))


kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 0x00ff00
    else:
        return 0xff0000
colors = kw_stats.apply(color_funk, axis=1, reduce=False)

# XXXXXX
connected_components = list(nx.connected_component_subgraphs(G))
largest_cc = max(connected_components, key=len)

nodes = {i: {'color': colors[i], 'size':1.25} for i in largest_cc.nodes_iter()}

edges = []
for edge in largest_cc.edges_iter():
    val = otus_score[edge[0]][edge[1]]
    if val > 0:
        edges.append({'source': edge[0], 'target': edge[1], 'color': 0x800080})
    elif val < 0:
        edges.append({'source': edge[0], 'target': edge[1], 'color': 0xFF8000})

graph = {
    'nodes': nodes,
    'edges': edges
}
igraph.draw(graph, directed=False)


# # Dysbiosis Index
# 
# To define the dysbiosis index, we first need to have calculated the nc-score tables with `CCREPE`, along with the z-stats and corrected p-values, once we've done this, the general steps we need to follow are:
# 
# * Remove useless and redundant data:
#     * Remove the upper diagonal of the matrix.
#     * Remove nc-scores that differ in sign with the z statistics.
#     * Remove nc-scores that have a q value above 0.05.
#     * Filter all nc-scores at a threshold so as to keep only the **strong** associations.
# * Build a weighted graph from the resulting matrix:
#     * Find the largest connected component and only keep these vertices.
# * Separate the nodes of the graph by the OTUs that had a higher mean in `healthy` samples than in `ibd` samples.
#     * We create two sets of OTUs $I$ and $H$, such that $\forall i \in I$ the abundance of every OTU $i$ was higher in the samples with IBD than in the healthy samples, conversely $\forall h \in H$ the abundance of every OTU $h$ was higher in samples of healthy dogs than in the samples of dogs with IBD.
# * Finally, for every sample, we define the microbial dysbiosis index (similarly to how it was done in Gevers et al) value as:
# 
# $md \left(sample \right) = Log \left(\frac{ \sum_{i \in I} sample[i] }{ \sum_{h \in H} sample[h] } \right)$
# 

# ### Summarize everythign as a single cell to create the appropriate columns
# 

import networkx as nx
import igraph

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt',
                       sep='\t', index_col='OTU')

G = nx.from_numpy_matrix(~mask.values, parallel_edges=False, create_using=nx.Graph())
G = nx.relabel_nodes(G, {i: o for i, o in enumerate(mask.index.tolist())})
G.remove_nodes_from([n for n in G.nodes_iter() if len(G.edges(n)) == 0])

# find the largest connected component
connected_components = list(nx.connected_component_subgraphs(G))
largest_cc = max(connected_components, key=len)

good_bad = {'healthy': [], 'ibd': []}

for n in largest_cc.nodes():
    row = kw_stats.loc[n]
    
    if row['healthy_mean'] > row['IBD_mean']:
        good_bad['healthy'].append(n)
    else:
        good_bad['ibd'].append(n)

mf = load_mf('taxonomic_summaries/no-diarrhea/mapping-file-full.alpha_L6.txt')

mf['PD_whole_tree_even_15000_alpha'] = pd.to_numeric(mf.PD_whole_tree_even_15000_alpha, errors='coerce')

prot = set(good_bad['healthy'])
infl = set(good_bad['ibd'])

mf['Protective'] = pd.Series(np.zeros_like(mf.index.values), mf.index, dtype=np.float)
mf['Inflammatory'] = pd.Series(np.zeros_like(mf.index.values), mf.index, dtype=np.float)

for column_name in mf.columns:
    if any([True for p in prot if p in column_name]):
        mf['Protective'] += mf[column_name].astype(np.float)
    elif any([True for i in infl if i in column_name]):
        mf['Inflammatory'] += mf[column_name].astype(np.float)
    else:
        continue

# calculating the dysbiosis index
mf['Dogbyosis Index'] = np.divide(mf['Inflammatory'], mf['Protective']).astype(np.float)
# drop any samples with undefined values
mf['Dogbyosis Index'].replace({0: np.nan}, inplace=True)
mf['Dogbyosis Index'] = np.log(mf['Dogbyosis Index'])
mf.dropna(0, 'any', subset=['Dogbyosis Index'], inplace=True)


serializable_mf = mf.apply(lambda x: x.astype(str), axis=0)
write_mf('mapping-file.alpha.index.dogbyosis.txt', serializable_mf)


for k, v in good_bad.iteritems():
    print k
    print '\n'.join(sorted(good_bad[k]))


plt.figure()
sns.jointplot('Dogbyosis Index', 'PD_whole_tree_even_15000_alpha',
              mf[mf['disease_stat'] == 'healthy'], kind='reg', color='#1b9e77')
plt.savefig('md-index/new-md-index.healthy.pdf')

plt.figure()
sns.jointplot('Dogbyosis Index', 'PD_whole_tree_even_15000_alpha',
              mf[mf['disease_stat'] != 'healthy'], kind='reg', color='#d95f02')
plt.savefig('md-index/new-md-index.ibd.pdf')


get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m  mapping-file.alpha.index.dogbyosis.txt -o beta/15000/unweighted-index --add_unique_columns')


get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -o beta/15000/unweighted-index-humans/ --add_unique_columns')





get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


# The nearest sequenced taxon index can be calculated as part of the `predict_metagenomes.py` script.
# 

get_ipython().system('cp /Users/yoshikivazquezbaeza/Dropbox/16s-fecal-only/predicted_metagenomes/full_table/nsti_per_sample.tab     NSTI_dogs_IBD.txt')


nsti = pd.read_csv('NSTI_dogs_IBD.txt', sep='\t', index_col='#Sample')


nsti.Value.hist(bins=60, color='#aeaeae')
plt.xlabel('Nearest Sequenced Taxon Index')
plt.ylabel('Samples')
plt.title('99.8 % of samples are below 0.15 NSTI')

plt.savefig('nsti.pdf')


(nsti.Value < 0.06).value_counts()


(nsti.Value < 0.15).value_counts()





get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr
from skbio.stats.distance import permanova, anosim
from skbio import DistanceMatrix

from IPython.display import Image

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


# # Compare the effect of antibiotics
# 

mf = load_mf('mapping-file-full.alpha.txt')


# We have some ambiguity in the antibiotics usage information, for the rest of this analysis, we will only use the **definite answers**.
# 

mf.Antibiotics.value_counts()


get_ipython().system("filter_distance_matrix.py -i beta/15000/unweighted_unifrac_dm.txt -o beta/15000/unweighted_unifrac_dm.abxs-only.txt -m mapping-file-full.alpha.txt -s 'Antibiotics:definite_no,definite_yes'")


# Load the distance matrix and mapping file:
# 

dm = DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.abxs-only.txt')


emf = mf.loc[list(dm.ids)].copy()
emf.groupby('disease_stat').Antibiotics.value_counts()


# Test for subjects **with IBD** and antibiotics and without antibiotics.
# 

permanova(dm.filter(emf[emf.disease_stat == 'IBD'].index, strict=False), mf, 'Antibiotics', permutations=10000)


# Test for subjects **without IBD** and antibiotics and without antibiotics.
# 

permanova(dm.filter(emf[emf.disease_stat == 'healthy'].index, strict=False), mf, 'Antibiotics', permutations=10000)


# Compare them on a disease state basis:
# 

permanova(DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.txt'),
          mf, 'disease_stat', permutations=10000)


# Compare them on an Antibiotic-history basis
# 

permanova(DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.abxs-only.txt'),
          mf, 'Antibiotics', permutations=10000)





# # Filter out low mean abundance features in the heatmaps and sort them by how prevalent they are in each category
# 

get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr
from biom import load_table

from IPython.display import Image

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


def exploding_panda(_bt):
    """BIOM->Pandas dataframe converter

    Parameters
    ----------
    _bt : biom.Table
        BIOM table

    Returns
    -------
    pandas.DataFrame
        The BIOM table converted into a DataFrame
        object.
        
    References
    ----------
    Based on this answer on SO:
    http://stackoverflow.com/a/17819427/379593
    """
    m = _bt.matrix_data
    data = [pd.SparseSeries(m[i].toarray().ravel()) for i in np.arange(m.shape[0])]
    out = pd.SparseDataFrame(data, index=_bt.ids('observation'),
                             columns=_bt.ids('sample'))
    
    return out.to_dense()


# We filter out OTUs that are not present in at least 10 percent of the samples and in at least 20 percent of the sample. This helps us reduce the penalization that we get from doing multiple comparisons and to only assess the statistical significance of microbes that are well represented by the samples (HT to Kyle Bittinger, as I discussed a lot about this with him at the Quebec QIIME workshop).
# 

get_ipython().run_cell_magic('bash', '', '\nmkdir -p stats/group-significance/no-diarrhea/\n\n# 5 percent\nfilter_otus_from_otu_table.py -s 8 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom\n\n# 10 percent\nfilter_otus_from_otu_table.py -s 16 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom\n\n# 20 percent\nfilter_otus_from_otu_table.py -s 32 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.20pct.biom\n\n# 40 percent\nfilter_otus_from_otu_table.py -s 64 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom')


get_ipython().run_cell_magic('bash', '-e', '\n# 5 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-5pct/\n\n# genus\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L6.tsv \\\n-s kruskal_wallis\n\n# family\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L5.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L5.tsv \\\n-s kruskal_wallis\n\n# order\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv \\\n-s kruskal_wallis\n\n# class\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L3.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L3.tsv \\\n-s kruskal_wallis\n\n# 10 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-10pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv \\\n-s kruskal_wallis\n\n# 40 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-40pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv \\\n-s kruskal_wallis\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L4.tsv \\\n-s kruskal_wallis')


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom')
mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

colors = []
for sid in bt.ids('sample'):
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


get_ipython().run_cell_magic('bash', '', "\n# You must be in CU's VPN for the following to work\n\nmkdir -p gevers\n\nscp barnacle:/home/yovazquezbaeza/research/gevers/closed-ref-13-8/trimmed-100/otu_table.biom gevers/\nscp barnacle:/home/yovazquezbaeza/research/gevers/mapping_file.shareable.txt gevers/\n\nls gevers")


# Let's filter out the diarrhea dogs from the table that hasn't yet been rarefied.
# 

get_ipython().system("filter_samples_from_otu_table.py -i otu_table.biom -o otu_table.no-diarrhea.biom -s 'disease_stat:!acute hem. diarrhea,*' -m mapping-file-full.txt")


get_ipython().run_cell_magic('bash', '-e', '\nmkdir -p combined-gevers-suchodolski\n\n# both tables were picked against 13_8\nmerge_otu_tables.py \\\n-i otu_table.no-diarrhea.biom,gevers/otu_table.biom \\\n-o combined-gevers-suchodolski/otu-table.biom\n\nmerge_mapping_files.py \\\n-m mapping-file-full.txt,gevers/mapping_file.shareable.txt \\\n-o combined-gevers-suchodolski/mapping-file.txt \\\n--case_insensitive')


# Cleaning up the mapping file:
# 

mf = load_mf('combined-gevers-suchodolski/mapping-file.txt')

def funk(row):
    if row['DIAGNOSIS'] == 'no_data':
        # we want to standardize the values of this column
        if row['DISEASE_STAT'] == 'healthy':
            return 'control'
        return row['DISEASE_STAT']
    else:
        return row['DIAGNOSIS']
mf['STATUS'] = mf.apply(funk, axis=1, reduce=True)

# clean up some other fields
repl = {'TITLE': {'no_data': 'Gevers_CCFA_RISK'},
        'HOST_COMMON_NAME': {'no_data': 'human'}}
mf.replace(repl, inplace=True)

write_mf('combined-gevers-suchodolski/mapping-file.standardized.txt',
         mf)


get_ipython().system('single_rarefaction.py -i combined-gevers-suchodolski/otu-table.biom -o combined-gevers-suchodolski/otu-table.15000.biom -d 15000')


# # Filter out low mean abundance stuff in the heatmaps and sort them by how prevalent they are in each category
# 

get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr
from biom import load_table

from IPython.display import Image

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


def exploding_panda(_bt):
    """BIOM->Pandas dataframe converter

    Parameters
    ----------
    _bt : biom.Table
        BIOM table

    Returns
    -------
    pandas.DataFrame
        The BIOM table converted into a DataFrame
        object.
        
    References
    ----------
    Based on this answer on SO:
    http://stackoverflow.com/a/17819427/379593
    """
    m = _bt.matrix_data
    data = [pd.SparseSeries(m[i].toarray().ravel()) for i in np.arange(m.shape[0])]
    out = pd.SparseDataFrame(data, index=_bt.ids('observation'),
                             columns=_bt.ids('sample'))
    
    return out.to_dense()


# We filter out OTUs that are not present in at least 10 percent of the samples and in at least 20 percent of the sample. This helps us reduce the penalization that we get from doing multiple comparisons and to only assess the statistical significance of microbes that are well represented by the samples (HT to Kyle Bittinger, as I discussed a lot about this with him at the Quebec QIIME workshop).
# 

get_ipython().run_cell_magic('bash', '', '\nmkdir -p stats/group-significance/no-diarrhea/\n\n# 5 percent\nfilter_otus_from_otu_table.py -s 8 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom\n\n# 10 percent\nfilter_otus_from_otu_table.py -s 16 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom\n\n# 20 percent\nfilter_otus_from_otu_table.py -s 32 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.20pct.biom\n\n# 25 percent\nfilter_otus_from_otu_table.py -i otu_table.15000.biom \\\n-o otu_table.15000.25percent.biom \\\n-s 40 \\\n\n# 40 percent\nfilter_otus_from_otu_table.py -s 64 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom')


get_ipython().run_cell_magic('bash', '-e', '\n# 5 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-5pct/\n\n# genus\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L6.tsv \\\n-s kruskal_wallis\n\n# family\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L5.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L5.tsv \\\n-s kruskal_wallis\n\n# order\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv \\\n-s kruskal_wallis\n\n# class\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L3.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L3.tsv \\\n-s kruskal_wallis\n\n# 10 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-10pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv \\\n-s kruskal_wallis\n\n# 40 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-40pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv \\\n-s kruskal_wallis\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L4.tsv \\\n-s kruskal_wallis')


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom')
mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

colors = []
for sid in bt.ids('sample'):
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


df = exploding_panda(bt)

# sample_order = (mf.disease_stat == 'healthy').index.tolist() + (mf.disease_stat != 'healthy').index.tolist()
# df = df[sample_order]


h = exploding_panda(bt)


h


len(sample_order)


h.reindex(columns=sample_order)


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')


get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr

from IPython.display import Image

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')


# ## Only run the following cell if you don't yet have greengenes, otherwise this will download the full database
# 

get_ipython().run_cell_magic('bash', '', '\ncurl -O ftp://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz\ntar -xzf gg_13_8_otus.tar.gz')


get_ipython().system('beta_diversity_through_plots.py -i otu_table.15000.no-diarrhea.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o beta/15000 -a -O 7 --color_by_all_fields -f')


# # *Faecalibacterium* is reduced in diseased subjects and increased in healthy subjects
# 
# Principal Coordinates Analysis plot of unweighted UniFrac distances of healthy and diseased dogs, colored by the relative abundance of *k\__Bacteria;p\__Firmicutes;c\__Clostridia;o\__Clostridiales;f\__Ruminococcaceae;g\__Faecalibacterium* as described by the genus-level taxonomic summary. The big spheres represent the subjects with IBD and the small spheres the healhty subjects, the color scale goes from white to orange and finally to red, where white is a low value and red is a high value (aproximately 0.5).
# 
# This is conflicting with the findings of Suchodolski et al 2012.
# 

Image('beta/15000/screen-shots/faecalibacterium-unweighted-diseased-are-big-spheres.png')


Image('beta/15000/screen-shots/unweighted-disease-status.png')


# # $\beta -diversity$ specific by disease state
# 

get_ipython().system('split_otu_table.py -i otu_table.15000.no-diarrhea.biom -m mapping-file-full.alpha.L6index.txt  -f disease_stat -o split-by-disease-state')


get_ipython().system('beta_diversity_through_plots.py -i split-by-disease-state/otu_table.15000.no-diarrhea__disease_stat_IBD__.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o split-by-disease-state/beta/15000/ibd --color_by_all_fields -f')

get_ipython().system('beta_diversity_through_plots.py -i split-by-disease-state/otu_table.15000.no-diarrhea__disease_stat_healthy__.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o split-by-disease-state/beta/15000/healthy --color_by_all_fields -f')


# We couldn't quite find anything too interesting in these plots, except for the fact that there doesn't seem to be any grouping withing the disease states i.e. dogs of the same country, weight or age do not seem to cluster together.
# 

# # Biplots of the $\beta$-diversity plots
# 

get_ipython().system('summarize_taxa.py -i otu_table.15000.no-diarrhea.biom -o taxonomic_summaries/no-diarrhea/summaries')


get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L3.txt -o beta/15000/unweighted_unifrac_emperor_pcoa_biplot/ --biplot_fp beta/15000/unweighted_unifrac_emperor_pcoa_biplot/biplot.txt')

get_ipython().system('make_emperor.py -i beta/15000/weighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L3.txt -o beta/15000/weighted_unifrac_emperor_pcoa_biplot/ --biplot_fp beta/15000/weighted_unifrac_emperor_pcoa_biplot/biplot.txt')


get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L6.txt -o beta/15000/unweighted_unifrac_emperor_pcoa_biplot-L6/ --biplot_fp beta/15000/unweighted_unifrac_emperor_pcoa_biplot/biplot-L6.txt')

get_ipython().system('make_emperor.py -i beta/15000/weighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L6.txt -o beta/15000/weighted_unifrac_emperor_pcoa_biplot-L6/ --biplot_fp beta/15000/weighted_unifrac_emperor_pcoa_biplot/biplot-L6.txt')


# ###Unweighted UniFrac
# 

Image('beta/15000/screen-shots/unweighted-unifrac-biplot-disease-status.png')


# ### Weighted UniFrac
# 

Image('beta/15000/screen-shots/weighted-unifrac-biplot-disease-status.png')


get_ipython().system('compare_categories.py --method permanova -i beta/15000/unweighted_unifrac_dm.txt -m mapping-file-full.alpha.txt -c disease_stat -o beta/15000/stats-unweighted/')

get_ipython().system('compare_categories.py --method permanova -i beta/15000/weighted_unifrac_dm.txt -m mapping-file-full.alpha.txt -c disease_stat -o beta/15000/stats-weighted/')


pd.read_csv('beta/15000/stats-unweighted/permanova_results.txt', sep='\t')


pd.read_csv('beta/15000/stats-weighted/permanova_results.txt', sep='\t')





