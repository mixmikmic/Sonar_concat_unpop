# # Figure. Subject Information
# 

import copy
import os
import subprocess

import cdpybio as cpb
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ciepy
import cardipspy as cpy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

dy_name = 'figure_subject_information'
    
outdir = os.path.join(ciepy.root, 'output', dy_name)
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output', dy_name)
cpy.makedir(private_outdir)


# Each figure should be able to fit on a single 8.5 x 11 inch page. Please do not send figure panels as individual files. We use three standard widths for figures: 1 column, 85 mm; 1.5 column, 114 mm; and 2 column, 174 mm (the full width of the page). Although your figure size may be reduced in the print journal, please keep these widths in mind. For Previews and other three-column formats, these widths are also applicable, though the width of a single column will be 55 mm.
# 

fn = os.path.join(ciepy.root, 'output', 'input_data', 'wgs_metadata.tsv')
wgs_meta = pd.read_table(fn, index_col=0, squeeze=True)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'rnaseq_metadata.tsv')
rna_meta = pd.read_table(fn, index_col=0)
rna_meta = rna_meta[rna_meta.in_eqtl]
fn = os.path.join(ciepy.root, 'output', 'input_data', 'subject_metadata.tsv')
subject_meta = pd.read_table(fn, index_col=0)


subject_meta = subject_meta.ix[set(rna_meta.subject_id)]
family_vc = subject_meta.family_id.value_counts()
family_vc = family_vc[family_vc > 1]
eth_vc = subject_meta.ethnicity_group.value_counts().sort_values()
sex_vc = subject_meta.sex.value_counts()
sex_vc.index = pd.Series(['Female', 'Male'], index=['F', 'M'])[sex_vc.index]


sns.set_style('whitegrid')


p = subject_meta.ethnicity_group.value_counts()['European'] / float(subject_meta.shape[0])
print('{:.2f}% of the subjects are European.'.format(p * 100))


n = subject_meta.age.median()
print('Median subject age: {}.'.format(n))


p = sex_vc['Female'] / float(sex_vc.sum())
print('{:.2f}% of subjects are female.'.format(p * 100))


bcolor = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254, 1.0)

fig = plt.figure(figsize=(6.85, 4), dpi=300)

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.text(0, 1, 'Figure S1',
        size=16, va='top')
ciepy.clean_axis(ax)
ax.set_xticks([])
ax.set_yticks([])
gs.tight_layout(fig, rect=[0, 0.90, 0.5, 1])

# Age
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
subject_meta.age.hist(bins=np.arange(5, 95, 5), ax=ax)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_xlabel('Age (years)', fontsize=8)
ax.set_ylabel('Number of subjects', fontsize=8)
ax.grid(axis='x')
gs.tight_layout(fig, rect=[0, 0.45, 0.4, 0.9])

# Ethnicity
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
eth_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_ylabel('Ethnicity', fontsize=8)
ax.set_xlabel('Number of subjects', fontsize=8)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0.4, 0.45, 1, 0.9])

# Family size
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
family_vc.plot(kind='bar', color=bcolor)
ax.set_xticks([])
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_xlabel('Family', fontsize=8)
ax.set_ylabel('Number of family members', fontsize=8)
gs.tight_layout(fig, rect=[0.4, 0, 1, 0.45])

# Sex 
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
sex_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_ylabel('Sex', fontsize=8)
ax.set_xlabel('Number of subjects', fontsize=8)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0, 0, 0.4, 0.45])

t = fig.text(0.005, 0.87, 'A', weight='bold', 
             size=12)
t = fig.text(0.4, 0.87, 'B', weight='bold', 
             size=12)
t = fig.text(0.005, 0.45, 'C', weight='bold', 
             size=12)
t = fig.text(0.4, 0.45, 'D', weight='bold', 
             size=12)

fig.savefig(os.path.join(outdir, 'subject_info.pdf'))
fig.savefig(os.path.join(outdir, 'subject_info.png'), dpi=300)


fs = 10
bcolor = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254, 1.0)

fig = plt.figure(figsize=(6, 4), dpi=300)

# Age
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
subject_meta.age.hist(bins=np.arange(5, 95, 5), ax=ax)
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_xlabel('Age (years)', fontsize=fs)
ax.set_ylabel('Number of subjects', fontsize=fs)
ax.grid(axis='x')
gs.tight_layout(fig, rect=[0, 0.45, 0.4, 1])

# Ethnicity
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
eth_vc.plot(kind='barh', color=bcolor)
ax.set_xticks(ax.get_xticks()[0::2])
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_ylabel('Ethnicity', fontsize=fs)
ax.set_xlabel('Number of subjects', fontsize=fs)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0.4, 0.45, 1, 1])

# Family size
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
family_vc.plot(kind='bar', color=bcolor)
ax.set_xticks([])
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_xlabel('Family', fontsize=fs)
ax.set_ylabel('Number of family members', fontsize=fs)
gs.tight_layout(fig, rect=[0.4, 0, 1, 0.5])

# Sex 
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
sex_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_ylabel('Sex', fontsize=fs)
ax.set_xlabel('Number of subjects', fontsize=fs)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0, 0, 0.4, 0.5])

fig.savefig(os.path.join(outdir, 'subject_info_presentation.pdf'))


# # eQTL Methods Exploration
# 
# I want to look at the effect of using different expression estimates.
# 

import cPickle
import glob
import gzip
import os
import random
import shutil
import subprocess
import sys

import cdpybio as cpb
import matplotlib.pyplot as plt
import mygene
import myvariant
import numpy as np
import pandas as pd
import pybedtools as pbt
import scipy.stats as stats
import seaborn as sns

import ciepy
import cardipspy as cpy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

dy_name = 'eqtl_methods_exploration'

import socket
if socket.gethostname() == 'fl-hn1' or socket.gethostname() == 'fl-hn2':
    dy = os.path.join(ciepy.root, 'sandbox', 'tmp', dy_name)
    cpy.makedir(dy)
    pbt.set_tempdir(dy)
    
outdir = os.path.join(ciepy.root, 'output', dy_name)
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output', dy_name)
cpy.makedir(private_outdir)


fn = os.path.join(ciepy.root, 'output', 'input_data', 'rnaseq_metadata.tsv')
rna_meta = pd.read_table(fn, index_col=0)


fn = os.path.join(ciepy.root, 'output', 'eqtl_input', 'gene_to_regions.p')
gene_to_regions = cPickle.load(open(fn, 'rb'))
gene_info = pd.read_table(cpy.gencode_gene_info, index_col=0)


def orig_dir():
    os.chdir('/raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/notebooks/')


# I've copied the methods from the `run_emmax.py` script here so I can run some
# simple analyses.
# 

# This is an example call to `run_emmax.py`:
# 
#     python /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/scripts/run_emmax.py         ENSG00000103723.8         /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/private_data/wgs/biallelic_snvs.vcf.gz         chr15:82331447-84378665         /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/vst_counts_phe.tsv         /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax.ind         /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/wgs.kin         /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/run_eqtl_analysis/test_results/ENSG00000103723.8         -c /raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax.cov
#         
# The genes ENSG00000189306.6 and ENSG00000169715.10 are significant while ENSG00000181315.6
# and ENSG00000198912.6 are not significant. I can use these for testing.
# 

# ## Different Expression Estimates
# 
# I'm considering using three different expression estimates:
# * naive counts: The number of reads overlapping a gene.
# * RSEM effective counts: The effective counts estimated by RSEM.
# * RSEM TPM: Transcript TPM summed together per gene.
# 

fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'eqtls01', 'qvalues.tsv')
qvalues = pd.read_table(fn, index_col=0)


fn = os.path.join(ciepy.root, 'output', 'input_data', 'rsem_tpm.tsv')
tpm = pd.read_table(fn, index_col=0)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'rsem_expected_counts_norm.tsv')
ec = pd.read_table(fn, index_col=0)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'gene_counts_norm.tsv')
gc = pd.read_table(fn, index_col=0)


# Transform to standard normal and change IDs to match those in VCF.
# 

tpm = tpm[rna_meta[rna_meta.in_eqtl].index]
tpm.columns = rna_meta[rna_meta.in_eqtl].wgs_id
tpm_sn = cpb.general.transform_standard_normal(tpm)
tpm_sn.to_csv(os.path.join(outdir, 'tpm_sn.tsv'), sep='\t')

ec = ec[rna_meta[rna_meta.in_eqtl].index]
ec.columns = rna_meta[rna_meta.in_eqtl].wgs_id
ec_sn = cpb.general.transform_standard_normal(ec)
ec_sn.to_csv(os.path.join(outdir, 'ec_sn.tsv'), sep='\t')

gc = gc[rna_meta[rna_meta.in_eqtl].index]
gc.columns = rna_meta[rna_meta.in_eqtl].wgs_id
gc_sn = cpb.general.transform_standard_normal(gc)
gc_sn.to_csv(os.path.join(outdir, 'gc_sn.tsv'), sep='\t')


fig,axs = plt.subplots(2, 2)
ax = axs[0, 0]
ax.scatter(tpm_sn.ix[qvalues.index[0]], ec_sn.ix[qvalues.index[0]])
ax.set_ylabel('Expected counts')
ax.set_xlabel('TPM')
ax = axs[0, 1]
ax.scatter(tpm_sn.ix[qvalues.index[0]], gc_sn.ix[qvalues.index[0]])
ax.set_ylabel('Gene counts')
ax.set_xlabel('TPM')
ax = axs[1, 0]
ax.scatter(gc_sn.ix[qvalues.index[0]], ec_sn.ix[qvalues.index[0]])
ax.set_ylabel('Expected counts')
ax.set_xlabel('Gene counts')
plt.tight_layout();


def make_emmax_sh(gene, exp, exp_name):
    out = os.path.join(private_outdir, '{}_{}'.format(gene, exp_name))
    cpy.makedir(out)
    f = open(os.path.join(out, '{}.sh'.format(gene)), 'w')
    f.write('#!/bin/bash\n\n')
    f.write('#$ -N emmax_{}_{}_test\n'.format(gene, exp_name))
    f.write('#$ -l opt\n')
    f.write('#$ -l h_vmem=2G\n')
    f.write('#$ -pe smp 4\n')
    f.write('#$ -S /bin/bash\n')
    f.write('#$ -o {}.out\n'.format(os.path.join(out, gene)))
    f.write('#$ -e {}.err\n\n'.format(os.path.join(out, gene)))

    f.write('module load cardips/1\n')
    f.write('source activate cie\n\n')

    f.write('python /frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/scripts/run_emmax.py \\\n')
    f.write('\t{} \\\n'.format(gene))
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/private_output/eqtl_input/filtered_all/0000.vcf.gz \\\n')
    f.write('\t{} \\\n'.format(gene_to_regions[gene][0][3:]))
    f.write('\t{} \\\n'.format(exp))
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax_samples.tsv \\\n')
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/wgs.kin \\\n')
    f.write('\t{} \\\n'.format(out))
    f.write('\t-c /frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax_full.tsv \\\n')
    f.write('\t-a 0\n')
    f.close()
    return os.path.join(out, '{}.sh'.format(gene))


genes = qvalues.index[0:5000:500]
names = ['tpm', 'ec', 'gc']
for gene in genes:
    for i,exp in enumerate([os.path.join(outdir, 'tpm_sn.tsv'), 
                            os.path.join(outdir, 'ec_sn.tsv'), 
                            os.path.join(outdir, 'gc_sn.tsv')]):
        exp_name = names[i]
        fn = make_emmax_sh(gene, exp, exp_name)
        get_ipython().system('qsub {fn}')


def plot_results(gene):
    gene = genes[0]
    fn = os.path.join(private_outdir, '{}_tpm'.format(gene), gene + '.tsv')
    tpm_res = ciepy.read_emmax_output(fn).dropna()
    fn = os.path.join(private_outdir, '{}_ec'.format(gene), gene + '.tsv')
    ec_res = ciepy.read_emmax_output(fn).dropna()
    fn = os.path.join(private_outdir, '{}_gc'.format(gene), gene + '.tsv')
    gc_res = ciepy.read_emmax_output(fn).dropna()
    for c in ['BETA', 'R2']:
        df = pd.DataFrame({'tpm': tpm_res[c], 'ec': ec_res[c], 'gc': gc_res[c]})
        sns.pairplot(df)
        plt.title(c)
    df = pd.DataFrame({'tpm': -np.log10(tpm_res['PVALUE']), 
                       'ec': -np.log10(ec_res['PVALUE']), 
                       'gc': -np.log10(gc_res['PVALUE'])})
    sns.pairplot(df)
    plt.title('$-\log_{10}$ $p$ value');


for g in genes:
    plot_results(g)


# We can see that expected counts and naive counts are very similar. TPM seems to have
# larger $R^2$ and $\beta$ values.
# 

# # Figure. eQTL Summary
# 

import cPickle
import glob
import os
import random
import subprocess

import cdpybio as cpb
from ipyparallel import Client
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools as pbt
from scipy.stats import fisher_exact
import seaborn as sns
import tabix
import vcf as pyvcf
import weblogolib as logo

import cardipspy as cpy
import ciepy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

dy_name = 'figure_eqtl_summary'

import socket
if socket.gethostname() == 'fl-hn1' or socket.gethostname() == 'fl-hn2':
    dy = os.path.join(ciepy.root, 'sandbox', dy_name)
    cpy.makedir(dy)
    pbt.set_tempdir(dy)
    
outdir = os.path.join(ciepy.root, 'output', dy_name)
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output', dy_name)
cpy.makedir(private_outdir)


sns.set_style('whitegrid')


gene_info = pd.read_table(cpy.gencode_gene_info, index_col=0)

dy = os.path.join(ciepy.root, 'output/eqtl_processing/eqtls01')
fn = os.path.join(dy, 'qvalues.tsv')
qvalues = pd.read_table(fn, index_col=0)

# fn = '/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_processing/no_peer_no_std_norm01/h2.tsv'
# h2 = pd.read_table(fn, index_col=0, header=None, squeeze=True)

fn = os.path.join(ciepy.root, 'output', 'input_data', 'rsem_tpm.tsv')
exp = pd.read_table(fn, index_col=0)


fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'eqtls01', 'lead_variants.tsv')
lead_vars = pd.read_table(fn, index_col=0)
sig = lead_vars[lead_vars.perm_sig]

fn = os.path.join(ciepy.root, 'output', 'gtex_analysis', 'plot_data.tsv')
plotd = pd.read_table(fn, index_col=0)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'rnaseq_metadata.tsv')
rna_meta = pd.read_table(fn, index_col=0)


fn = os.path.join(ciepy.root, 'private_output', 'eqtl_input', 
                  'filtered_all', '0000.vcf.gz')
vcf_reader = pyvcf.Reader(open(fn), compressed=True)
res_fns = glob.glob(os.path.join(ciepy.root, 'private_output', 'run_eqtl_analysis', 'eqtls01', 
                                 'gene_results', '*', 'ENS*.tsv'))
res_fns = pd.Series(res_fns,
                    index=[os.path.splitext(os.path.split(x)[1])[0] for x in res_fns])

qvalue_sig = qvalues[qvalues.perm_sig == 1]
qvalue_sig = qvalue_sig.sort_values('perm_qvalue')


fn = os.path.join(ciepy.root, 'output', 'eqtl_input', 
                  'tpm_log_filtered_phe_std_norm_peer_resid.tsv')
resid_exp = pd.read_table(fn, index_col=0)


log_exp = np.log10(exp + 1)
#log_exp = (log_exp.T - log_exp.mean(axis=1)).T
log_exp = log_exp[rna_meta[rna_meta.in_eqtl].index]
log_exp.columns = rna_meta[rna_meta.in_eqtl].wgs_id


# Example genes.
pgenes = ['IDO1', 'LCK', 'POU5F1', 'CXCL5', 'BCL9', 'FGFR1']
pgenes = ['POU5F1', 'CXCL5', 'BCL9', 'FGFR1', 'IDO1']
genes = []
for g in pgenes:
    i = gene_info[gene_info.gene_name == g].index[0]
    if i in qvalues[qvalues.perm_sig].index:
        genes.append(i)


pdf = gene_info.ix[genes]
t_h2 = []
t_r2 = []
for g in pdf.index:
    fn = os.path.join(ciepy.root, 'private_output/run_eqtl_analysis/no_peer_no_std_norm01/gene_results', 
                      g, '{}.tsv'.format(g))
    res = ciepy.read_emmax_output(fn)
    res = res.sort_values('PVALUE')
    t_r2.append(res.R2.values[0])
    fn = os.path.join(ciepy.root, 'private_output/run_eqtl_analysis/no_peer_no_std_norm01/gene_results', 
                      g, '{}.reml'.format(g))
    t_h2.append(pd.read_table(fn, index_col=0, header=None, squeeze=True)['h2'])
pdf['r2'] = t_r2
pdf['h2'] = t_h2
pdf


def eqtl_violin(gene_id, exp, ax):
    res = ciepy.read_emmax_output(res_fns[gene_id])
    res = res.sort_values('PVALUE')
    t =  vcf_reader.fetch(res.CHROM.values[0], 
                          res.BEG.values[0], 
                          res.BEG.values[0] + 1)
    r = t.next()
    tdf = pd.DataFrame(exp.ix[gene_id])
    tdf.columns = ['Expression']
    tdf['Genotype'] = 0
    hets = set(exp.columns) & set([s.sample for s in r.get_hets()])
    tdf.ix[hets, 'Genotype'] = 1
    alts = set(exp.columns) & set([s.sample for s in r.get_hom_alts()])
    tdf.ix[alts, 'Genotype'] = 2
    ax = sns.violinplot(x='Genotype', y='Expression', data=tdf, color='grey',
                        order=[0, 1, 2], scale='count', linewidth=0.5)
    ax.set_ylabel('$\\log_{10}$ TPM', fontsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(6)
    sns.regplot(x='Genotype', y='Expression', data=tdf, scatter=False, color='red', 
                ci=None, line_kws={'linewidth':0.8})
    ax.set_ylabel('$\\log_{10}$ TPM', fontsize=8)
    ya, yb = plt.ylim()
    ax.set_xlabel('')
    #ax.set_xticklabels(['Homo.\nref.', 'Het.', 'Homo.\nalt.'], fontsize=8)
    ax.set_xticklabels([r.REF + '\n' + r.REF, r.REF + '\n' + str(r.ALT[0]), str(r.ALT[0]) + '\n' + str(r.ALT[0])])
    ax.set_title(gene_info.ix[pdf.index[i], 'gene_name'], fontsize=8, style='italic')
    #ax.set_xlabel('Genotype', fontsize=8)
    return ax


a = gene_info.ix[qvalue_sig.index, 'gene_type'].value_counts()
b = gene_info.ix[qvalues.index, 'gene_type'].value_counts()
gtypes = pd.concat([a, b], axis=1)
gtypes = gtypes.fillna(0)
gtypes.columns = ['Significant', 'Tested']
gtypes['Not significant'] = gtypes.Tested - gtypes.Significant
gtypes.sort_values(by=['Tested'], inplace=True, ascending=True)
gtypes = gtypes.drop('Tested', axis=1)
gtypes.index = [x.replace('_', ' ') for x in gtypes.index]


gtypes.tail(5)


from matplotlib.colors import ListedColormap

my_cmap = ListedColormap(sns.color_palette())


fig = plt.figure(figsize=(4.48, 6.1), dpi=300)

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.text(0, 1, 'Figure 1',
        size=16, va='top')
ciepy.clean_axis(ax)
ax.set_xticks([])
ax.set_yticks([])
gs.tight_layout(fig, rect=[0, 0.95, 0.5, 1])

# eGene types.
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
ax = fig.add_subplot(gs[0, 0])
gtypes.tail().plot(kind='barh', stacked=True, ax=ax, legend=False, lw=0, colormap=my_cmap)
ax.legend(frameon=True, fancybox=True, fontsize=8, loc='lower right')
ax.set_xlim(0, 5500)
ax.grid(axis='y')
ax.xaxis.set_major_formatter(ciepy.comma_format)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_xlabel('Number of genes', fontsize=8)
sns.despine(ax=ax, top=False)
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
xbreaksize = 0.01 * (xmax - xmin)
ybreaksize = 0.1 * (ymax - ymin)
fudge = 0.001 * (xmax - xmin)
kwargs = dict(lw=1, color='0.8', solid_capstyle='butt', clip_on=False, zorder=0)
ax.plot((xmax + fudge - xbreaksize, xmax + fudge + xbreaksize), 
        (ymin - ybreaksize, ymin + ybreaksize), **kwargs) # top-left diagonal
ax.plot((xmax + fudge - xbreaksize, xmax + fudge + xbreaksize), 
        (ymax - ybreaksize, ymax + ybreaksize), **kwargs) # bottom-left diagonal

# Make top of protein-coding genes.
ax2 = fig.add_subplot(gs[0, 1])
gtypes.tail().plot(kind='barh', stacked=True, ax=ax2, legend=False, lw=0, colormap=my_cmap)
sns.despine(ax=ax2, top=False, left=True, right=False)
ax2.set_xlim(12500, 14000)
ax2.set_yticks([])
ax2.grid(axis='y')
ax2.set_xticks([13000, 14000])
ax2.xaxis.set_major_formatter(ciepy.comma_format)
for t in ax2.get_xticklabels() + ax2.get_yticklabels():
    t.set_fontsize(8)
gs.tight_layout(fig, rect=[0, 0.75, 1, 0.95])
xmin,xmax = ax2.get_xlim()
ymin,ymax = ax2.get_ylim()
ax2.plot((xmin - fudge - xbreaksize, xmin - fudge + xbreaksize), 
        (ymin - ybreaksize, ymin + ybreaksize), **kwargs) # bottom-right diagonal
ax2.plot((xmin - fudge - xbreaksize, xmin - fudge + xbreaksize), 
        (ymax - ybreaksize, ymax + ybreaksize), **kwargs) # top-right diagonal

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
i = 0
ax = eqtl_violin(pdf.index[i], log_exp, ax)
ax.set_title(gene_info.ix[pdf.index[i], 'gene_name'], fontsize=8, style='italic')
gs.tight_layout(fig, rect=[0, 0.55, 0.5, 0.78])

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
i = 1
ax = eqtl_violin(pdf.index[i], log_exp, ax)
ax.set_title(gene_info.ix[pdf.index[i], 'gene_name'], fontsize=8, style='italic')
gs.tight_layout(fig, rect=[0.5, 0.55, 1, 0.78])

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
i = 2
ax = eqtl_violin(pdf.index[i], log_exp, ax)
ax.set_title(gene_info.ix[pdf.index[i], 'gene_name'], fontsize=8, style='italic')
ax.set_yticks(ax.get_yticks()[0::2])
gs.tight_layout(fig, rect=[0, 0.35, 0.5, 0.58])

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
i = 3
ax = eqtl_violin(pdf.index[i], log_exp, ax)
gs.tight_layout(fig, rect=[0.5, 0.35, 1, 0.58])

# Number of eGenes vs. number of samples
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(plotd.ix[plotd.source == 'gtex', 'num_samples'], 
           plotd.ix[plotd.source == 'gtex', 'num_sig_genes'],
           label='GTEx', color=cpb.analysis.tableau20[0], s=25, alpha=0.75)
ax.scatter([plotd.ix['ipsc', 'num_samples']], 
           [plotd.ix['ipsc', 'num_sig_genes']],
           label='iPSC', color=cpb.analysis.tableau20[6], alpha=0.75, s=40, marker='*')
ax.scatter([plotd.ix['ipsc_unrelateds', 'num_samples']], 
           [plotd.ix['ipsc_unrelateds', 'num_sig_genes']],
           label='iPSC unrelateds', color=cpb.analysis.tableau20[6], alpha=0.75, s=25, marker='d')
ax.legend(fontsize=7, loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
ax.set_xlabel('Number of samples', fontsize=8)
ax.set_ylabel('Number of eGenes', fontsize=8)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
#ax.legend(frameon=True, fancybox=True, fontsize=7, loc='lower right')
ax.yaxis.set_major_formatter(ciepy.comma_format)
ax.set_xticks(ax.get_xticks()[1::2])
gs.tight_layout(fig, rect=[0, 0, 0.52, 0.34])

# Percent distinct eGenes
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(plotd.ix[plotd.source == 'gtex', 'num_samples'], 
           plotd.ix[plotd.source == 'gtex', 'percent_distinct_egenes'],
           label='GTEx', color=cpb.analysis.tableau20[0], s=25, alpha=0.75)
ax.scatter([plotd.ix['ipsc', 'num_samples']], 
           [plotd.ix['ipsc', 'percent_distinct_egenes']],
           label='iPSC', color=cpb.analysis.tableau20[6], alpha=0.75, s=40, marker='*')
ax.scatter([plotd.ix['ipsc_unrelateds', 'num_samples']], 
           [plotd.ix['ipsc_unrelateds', 'percent_distinct_egenes']],
           label='iPSC unrelateds', color=cpb.analysis.tableau20[6], alpha=0.75, s=25, marker='d')
ax.legend(fontsize=7, loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
ax.set_xlabel('Number of samples', fontsize=8)
ax.set_ylabel('Percent unique eGenes', fontsize=8)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
# ax.legend(frameon=True, fancybox=True, fontsize=7, loc='upper right')
ymin,ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
ax.set_ylim(0, 0.1)
ax.set_xticks(ax.get_xticks()[1::2])
gs.tight_layout(fig, rect=[0.48, 0, 1, 0.34])

t = fig.text(0.005, 0.91, 'A', weight='bold', 
             size=12)
t = fig.text(0.005, 0.74, 'B', weight='bold', 
             size=12)
t = fig.text(0.5, 0.74, 'C', weight='bold', 
             size=12)
t = fig.text(0.005, 0.54, 'D', weight='bold', 
             size=12)
t = fig.text(0.5, 0.54, 'E', weight='bold', 
             size=12)
t = fig.text(0.005, 0.34, 'F', weight='bold', 
             size=12)
t = fig.text(0.5, 0.34, 'G', weight='bold', 
             size=12)

fig.savefig(os.path.join(outdir, 'eqtl_summary.pdf'))
fig.savefig(os.path.join(outdir, 'eqtl_summary.png'), dpi=300)


fig = plt.figure(figsize=(7, 4), dpi=300)

fs = 12

gs = gridspec.GridSpec(1, 1)
# Number of eGenes vs. number of samples
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(plotd.ix[plotd.source == 'gtex', 'num_samples'], 
           plotd.ix[plotd.source == 'gtex', 'num_sig_genes'],
           label='GTEx', color=cpb.analysis.tableau20[0], s=50, alpha=0.75)
ax.scatter([plotd.ix['ipsc', 'num_samples']], 
           [plotd.ix['ipsc', 'num_sig_genes']],
           label='iPSC', color=cpb.analysis.tableau20[6], alpha=0.75, s=75, marker='*')
ax.scatter([plotd.ix['ipsc_unrelateds', 'num_samples']], 
           [plotd.ix['ipsc_unrelateds', 'num_sig_genes']],
           label='iPSC unrelateds', color=cpb.analysis.tableau20[6], alpha=0.75, s=50, marker='d')
ax.legend(fontsize=fs, loc='lower center', ncol=3, bbox_to_anchor=(1, 1))
ax.set_xlabel('Number of samples', fontsize=fs)
ax.set_ylabel('Number of eGenes', fontsize=fs)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(fs)
#ax.legend(frameon=True, fancybox=True, fontsize=7, loc='lower right')
ax.yaxis.set_major_formatter(ciepy.comma_format)
ax.set_xticks(ax.get_xticks()[1::2])
gs.tight_layout(fig, rect=[0, 0, 0.52, 0.9])

# Percent distinct eGenes
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(plotd.ix[plotd.source == 'gtex', 'num_samples'], 
           plotd.ix[plotd.source == 'gtex', 'percent_distinct_egenes'],
           label='GTEx', color=cpb.analysis.tableau20[0], s=50, alpha=0.75)
ax.scatter([plotd.ix['ipsc', 'num_samples']], 
           [plotd.ix['ipsc', 'percent_distinct_egenes']],
           label='iPSC', color=cpb.analysis.tableau20[6], alpha=0.75, s=75, marker='*')
ax.scatter([plotd.ix['ipsc_unrelateds', 'num_samples']], 
           [plotd.ix['ipsc_unrelateds', 'percent_distinct_egenes']],
           label='iPSC unrelateds', color=cpb.analysis.tableau20[6], alpha=0.75, s=50, marker='d')
#ax.legend(fontsize=fs, loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
ax.set_xlabel('Number of samples', fontsize=fs)
ax.set_ylabel('Percent unique eGenes', fontsize=fs)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(fs)
# ax.legend(frameon=True, fancybox=True, fontsize=7, loc='upper right')
ymin,ymax = ax.get_ylim()
ax.set_ylim(0, 0.1)
ax.set_xticks(ax.get_xticks()[1::2])
gs.tight_layout(fig, rect=[0.48, 0, 1, 0.9])

fig.savefig(os.path.join(outdir, 'power_presentation.pdf'))


# # Motif Search
# 
# I want to search for transcription factor binding sites that
# are disrupted by potential QTNs.
# 

import copy
import cPickle
import os

from Bio.Seq import Seq
import cdpybio as cpb
import matplotlib.pyplot as plt
import MOODS
import numpy as np
import pandas as pd
import pybedtools as pbt
import seaborn as sns
import weblogolib as logo

import cardipspy as cpy
import ciepy

from IPython.display import Image 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')


outdir = os.path.join(ciepy.root, 'output',
                      'motif_search')
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output',
                              'motif_search')
cpy.makedir(private_outdir)


# fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'qvalues.tsv')
# qvalues = pd.read_table(fn, index_col=0)
# qvalues.columns = ['{}_gene'.format(x) for x in qvalues.columns]
# fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'most_sig.tsv')
# most_sig = pd.read_table(fn, index_col=0)
# most_sig = most_sig.join(qvalues)
# sig = most_sig[most_sig.sig_gene]

fn = os.path.join(ciepy.root, 'output', 'functional_annotation_analysis',
                  'encode_stem_cell_chip_seq.tsv')
encode_chip_seq = pd.read_table(fn, index_col=0)

gene_info = pd.read_table(cpy.gencode_gene_info, index_col=0)


# I'll use [motifs from the Kheradpour et al. 2013 paper](http://compbio.mit.edu/encode-motifs/). 
# The columns in the Kheradpour file are ACGT.
# 
# I'm going to choose a representative motif for each TF. I'll preferentially choose
# the motifs from H1-hESC experiments.
# 

motif_info_full_fn = os.path.join(outdir, 'motif_info_full.tsv')
motif_info_rep_fn = os.path.join(outdir, 'motif_info_rep.tsv')
matrices_fn = os.path.join(outdir, 'matrices.pickle')

if not sum([os.path.exists(x) for x in [motif_info_full_fn, motif_info_rep_fn, matrices_fn]]) == 3:
    key = []
    tf = []
    cell_line = []
    source = []
    length = []
    with open(cpy.kheradpour_motifs) as f:
        lines = f.read()
    m = lines.split('>')[1:]
    m = [x.split('\n')[:-1] for x in m]
    matrices = {}
    for x in m:
        k = x[0].split()[0]
        key.append(k)
        if 'transfac' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('transfac')
        elif 'jolma' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('jolma')
        elif 'jaspar' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('jaspar')
        elif 'bulyk' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('bulyk')
        else:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(x[0].split()[1].split('_')[1])
            source.append('encode')
        t = pd.DataFrame([y.split() for y in x[1:]],
                         columns=['base', 'A', 'C', 'G', 'T'])
        t.index = t.base
        t = t.drop('base', axis=1)
        for c in t.columns:
            t[c] = pd.to_numeric(t[c])
        matrices[k] = t
        length.append(t.shape[0])

    motif_info = pd.DataFrame({'tf': tf, 'cell_line': cell_line, 'source': source, 
                               'length': length}, index=key)
    motif_info.to_csv(motif_info_full_fn, sep='\t')
    
    with open(matrices_fn, 'w') as f:
        cPickle.dump(matrices, f)

    a = motif_info[motif_info.tf.apply(lambda x: x in encode_chip_seq.target.values)]
    b = a[a.cell_line == 'H1-hESC']
    b = b.drop_duplicates(subset='tf')
    a = a[a.cell_line != 'H1-hESC']
    a = a[a.tf.apply(lambda x: x not in b.tf.values)]
    a['so'] = a.source.replace({'jolma': 0, 'bulyk': 1, 'transfac': 2, 
                                'jaspar': 3, 'encode': 4})
    a = a.sort_values(by='so')
    a = a.drop_duplicates(subset='tf').drop('so', axis=1)
    motif_info = pd.concat([b, a])
    motif_info.to_csv(motif_info_rep_fn, sep='\t')


encode_chip_seq = encode_chip_seq[encode_chip_seq.target.apply(lambda x: x in motif_info.tf.values)]
encode_chip_seq = encode_chip_seq.drop_duplicates(subset='target')


# There are a few TF ChIP-seq datasets for which I don't have motifs. I can't find
# them by hand in the Kheradpour data either. I'll skip these for now.
# 
# For now, I'm going to restrict to SNVs beacuse this code was written
# for SNVs. I can add indels later if I'd like.
# 

sig = sig[sig.vtype == 'snp']


lines = (sig.chrom + '\t' + sig.start.astype(str) + 
         '\t' + sig.end.astype(str) + '\t' + sig.chrom +
         ':' + sig.end.astype(str))
lines = lines.drop_duplicates()
sig_bt = pbt.BedTool('\n'.join(lines + '\n'), from_string=True)
m = max([x.shape[0] for x in matrices.values()])
sig_bt = sig_bt.slop(l=m, r=m, g=pbt.genome_registry.hg19)
seqs = sig_bt.sequence(fi=cpy.hg19)
sig_seqs = [x.strip() for x in open(seqs.seqfn).readlines()]
sig_seqs = pd.Series(sig_seqs[1::2], index=[x[1:] for x in sig_seqs[0::2]])
sig_seqs = sig_seqs.apply(lambda x: x.upper())


snvs = sig[['chrom', 'start', 'end', 'loc', 'marker_id']]
snvs.index = snvs['loc'].values
snvs = snvs.drop_duplicates()
snvs['ref'] = snvs.marker_id.apply(lambda x: x.split('_')[1].split('/')[0])
snvs['alt'] = snvs.marker_id.apply(lambda x: x.split('_')[1].split('/')[1])

snvs['interval'] = ''
snvs['seq'] = ''
snvs['alt_seq'] = ''
for i in sig_seqs.index:
    chrom, start, end = cpb.general.parse_region(i)
    k = '{}:{}'.format(chrom, int(end) - m)
    snvs.ix[k, 'interval'] = i
    snvs.ix[k, 'seq'] = sig_seqs[i]
    ref, alt = snvs.ix[k, ['ref', 'alt']]
    assert sig_seqs[i][m] == ref
    snvs.ix[k, 'alt_seq'] = sig_seqs[i][0:m] + alt + sig_seqs[i][m + 1:]


lines = (sig.chrom + '\t' + sig.start.astype(str) + 
         '\t' + sig.end.astype(str) + '\t' + sig.chrom +
         ':' + sig.end.astype(str))
lines = lines.drop_duplicates()
sig_bt = pbt.BedTool('\n'.join(lines + '\n'), from_string=True)
sig_bt = sig_bt.sort()


snvs_tf = pd.DataFrame(False, index=snvs.index, columns=encode_chip_seq.target)
for i in encode_chip_seq.index:
    c = encode_chip_seq.ix[i, 'target']
    snvs_tf[c] = False
    bt = pbt.BedTool(cpb.general.read_gzipped_text_url(encode_chip_seq.ix[i, 'narrowPeak_url']), 
                     from_string=True)
    bt = bt.sort()
    res = sig_bt.intersect(bt, sorted=True, wo=True)
    for r in res:
        snvs_tf.ix['{}:{}'.format(r.chrom, r.end), c] = True


snv_motifs = {}
for i in snvs_tf[snvs_tf.sum(axis=1) > 0].index:
    se = snvs_tf.ix[i]
    se = se[se]
    keys = motif_info[motif_info.tf.apply(lambda x: x in se.index)].index
    ms = [matrices[x].T.values.tolist() for x in keys]
    # seq_res is a dict whose keys are motif names and whose values are lists 
    # of the hits of that motif. Each hit is a tuple of (pos, score). 
    seq_res = MOODS.search(snvs.ix[i, 'seq'], ms, 0.001, both_strands=True, 
                           bg=[0.25, 0.25, 0.25, 0.25])
    seq_mres = dict(zip(keys, seq_res))
    alt_seq_res = MOODS.search(snvs.ix[i, 'alt_seq'], ms, 0.001, both_strands=True, 
                               bg=[0.25, 0.25, 0.25, 0.25])
    alt_seq_mres = dict(zip(keys, alt_seq_res))
    sp = len(snvs.ix[i, 'seq']) / 2
    if seq_mres != alt_seq_mres:
        for k in seq_mres.keys():
            # Remove motifs where all the hits have the same score.
            if seq_mres[k] == alt_seq_mres[k]:
                seq_mres.pop(k)
                alt_seq_mres.pop(k)
            else:
                # Remove individual hits that have the same score for both sequences.
                shared = set(seq_mres[k]) & set(alt_seq_mres)
                seq_mres[k] = [x for x in seq_mres[k] if x not in shared]
                alt_seq_mres[k] = [x for x in alt_seq_mres[k] if x not in shared]
                a = seq_mres[k]
                to_remove = []
                for v in a:
                    start = v[0]
                    if start < 0:
                        start = start + len(snvs.ix[i, 'seq'])
                    if not start <= sp < start + motif_info.ix[k, 'length']:
                        to_remove.append(v)
                for v in to_remove:
                    a.remove(v)
                seq_mres[k] = a
                a = alt_seq_mres[k]
                to_remove = []
                for v in a:
                    start = v[0]
                    if start < 0:
                        start = start + len(snvs.ix[i, 'seq'])
                    if not start <= sp < start + motif_info.ix[k, 'length']:
                        to_remove.append(v)
                for v in to_remove:
                    a.remove(v)
                alt_seq_mres[k] = a
        snv_motifs[i] = [seq_mres, alt_seq_mres]


def plot_tf_disruption(m, ref, alt, fn, title=None):
    """m is the PWM, ref is the ref sequence, alt is the alt sequence"""
    k = 'SIX5_disc2'
    alphabet = logo.corebio.seq.unambiguous_dna_alphabet
    prior = [0.25, 0.25, 0.25, 0.25]
    counts = m.values
    assert counts.shape[1] == 4
    assert len(ref) == len(alt) == counts.shape[0]
    ref_counts = []
    for t in ref:
        ref_counts.append([int(t.upper() == 'A'), int(t.upper() == 'C'),
                           int(t.upper() == 'G'), int(t.upper() == 'T')])
    alt_counts = []
    for t in alt:
        alt_counts.append([int(t.upper() == 'A'), int(t.upper() == 'C'),
                           int(t.upper() == 'G'), int(t.upper() == 'T')])
    counts = np.concatenate([counts, ref_counts, alt_counts])
    data = logo.LogoData.from_counts(alphabet, counts, prior=None)
    fout = open(fn, 'w')
    options = logo.LogoOptions()
    options.fineprint = ''
    if title:
        options.logo_title = title
    else:
        options.logo_title = ''
    options.stacks_per_line = m.shape[0]
    options.show_xaxis = False
    options.show_yaxis = False
    options.color_scheme = logo.ColorScheme([logo.ColorGroup("G", "orange"), 
                                             logo.ColorGroup("C", "blue"),
                                             logo.ColorGroup("A", "green"),
                                             logo.ColorGroup("T", "red")])
    logo_format = logo.LogoFormat(data, options)
    fout.write(logo.png_print_formatter(data, logo_format))
    #fout.write(logo.pdf_formatter(data, logo_format))
    fout.close()
    Image(filename=fn)


cpy.makedir(os.path.join(outdir, 'tf_plots'))
for snv in snv_motifs.keys():
    seq_mres, alt_seq_mres = snv_motifs[snv]
    for k in seq_mres.keys():
        pwm = matrices[k]
        a = seq_mres[k]
        b = alt_seq_mres[k]
        starts = set([x[0] for x in a]) | set([x[0] for x in b])
        for start in starts:
            ref_seq = snvs.ix[snv, 'seq'][start: start + motif_info.ix[k, 'length']]
            alt_seq = snvs.ix[snv, 'alt_seq'][start: start + motif_info.ix[k, 'length']]
            if start < 0:
                ref_seq = str(Seq(ref_seq).reverse_complement())
                alt_seq = str(Seq(alt_seq).reverse_complement())
            fn = os.path.join(outdir, 'tf_plots', '{}_{}_{}.png'.format(
                snv.replace(':', '_'), k, str(start).replace('-', 'neg')))
            plot_tf_disruption(pwm, ref_seq, alt_seq, fn)


