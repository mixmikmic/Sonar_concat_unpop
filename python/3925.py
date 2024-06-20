# # The ISB-CGC open-access TCGA tables in Big-Query
# 
# The goal of this notebook is to introduce you to a new publicly-available, open-access dataset in BigQuery.  This set of BigQuery tables was produced by the [ISB-CGC](http://www.isb-cgc.org) project, based on the open-access [TCGA](http://cancergenome.nih.gov/) data available at the TCGA [Data Portal](https://tcga-data.nci.nih.gov/tcga/).  You will need to have access to a Google Cloud Platform (GCP) project in order to use BigQuery.  If you don't already have one, you can sign up for a [free-trial](https://cloud.google.com/free-trial/) or contact [us](mailto://info@isb-cgc.org) and become part of the community evaluation phase of our Cancer Genomics Cloud pilot.  (You can find more information about this NCI-funded program [here](https://cbiit.nci.nih.gov/ncip/nci-cancer-genomics-cloud-pilots).)
# 
# We are not attempting to provide a thorough BigQuery or IPython tutorial here, as a wealth of such information already exists.  Here are links to some resources that you might find useful: 
# * [BigQuery](https://cloud.google.com/bigquery/what-is-bigquery), 
# * the BigQuery [web UI](https://bigquery.cloud.google.com/) where you can run queries interactively, 
# * [IPython](http://ipython.org/) (now known as [Jupyter](http://jupyter.org/)), and 
# * [Cloud Datalab](https://cloud.google.com/datalab/) the recently announced interactive cloud-based platform that this notebook is being developed on. 
# 
# There are also many tutorials and samples available on github (see, in particular, the [datalab](https://github.com/GoogleCloudPlatform/datalab) repo and the [Google Genomics](  https://github.com/googlegenomics) project).
# 
# In order to work with BigQuery, the first thing you need to do is import the [gcp.bigquery](http://googlecloudplatform.github.io/datalab/gcp.bigquery.html) package:
# 

import gcp.bigquery as bq


# The next thing you need to know is how to access the specific tables you are interested in.  BigQuery tables are organized into datasets, and datasets are owned by a specific GCP project.  The tables we are introducing in this notebook are in a dataset called **`tcga_201607_beta`**, owned by the **`isb-cgc`** project.  A full table identifier is of the form `<project_id>:<dataset_id>.<table_id>`.  Let's start by getting some basic information about the tables in this dataset:
# 

d = bq.DataSet('isb-cgc:tcga_201607_beta')
for t in d.tables():
  print '%10d rows  %12d bytes   %s'       % (t.metadata.rows, t.metadata.size, t.name.table_id)


# These tables are based on the open-access TCGA data as of July 2016.  The molecular data is all "Level 3" data, and is divided according to platform/pipeline.  See [here](https://tcga-data.nci.nih.gov/tcga/tcgaDataType.jsp) for additional details regarding the TCGA data levels and data types.
# 
# Additional notebooks go into each of these tables in more detail, but here is an overview, in the same alphabetical order that they are listed in above and in the BigQuery web UI:
# 
# 
# - **Annotations**:  This table contains the annotations that are also available from the interactive [TCGA Annotations Manager](https://tcga-data.nci.nih.gov/annotations/).  Annotations can be associated with any type of "item" (*eg* Patient, Sample, Aliquot, etc), and a single item may have more than one annotation.  Common annotations include "Item flagged DNU", "Item is noncanonical", and "Prior malignancy."  More information about this table can be found in the [TCGA Annotations](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/TCGA%20Annotations.ipynb) notebook.
# 
# 
# - **Biospecimen_data**:  This table contains information obtained from the "biospecimen" and "auxiliary" XML files in the TCGA Level-1 "bio" archives.  Each row in this table represents a single "biospecimen" or "sample".  Most participants in the TCGA project provided two samples: a "primary tumor" sample and a "blood normal" sample, but others provided normal-tissue, metastatic, or other types of samples.  This table contains metadata about all of the samples, and more information about exploring this table and using this information to create your own custom analysis cohort can be found  in the [Creating TCGA cohorts (part 1)](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%201.ipynb) and [(part 2)](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%202.ipynb) notebooks.
# 
# 
# - **Clinical_data**:  This table contains information obtained from the "clinical" XML files in the TCGA Level-1 "bio" archives.  Not all fields in the XML files are represented in this table, but any field which was found to be significantly filled-in for at least one tumor-type has been retained.  More information about exploring this table and using this information to create your own custom analysis cohort can be found in the [Creating TCGA cohorts (part 1)](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%201.ipynb) and [(part 2)](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%202.ipynb) notebooks.
# 
# 
# - **Copy_Number_segments**:  This table contains Level-3 copy-number segmentation results generated by The Broad Institute, from Genome Wide SNP 6 data using the CBS (Circular Binary Segmentation) algorithm.  The values are base2 log(copynumber/2), centered on 0.  More information about this data table can be found in the [Copy Number segments](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Copy%20Number%20segments.ipynb) notebook.
# 
# 
# - **DNA_Methylation_betas**:  This table contains Level-3 summary measures of DNA methylation for each interrogated locus (beta values: M/(M+U)).  This table contains data from two different platforms: the Illumina Infinium HumanMethylation 27k and 450k arrays.  More information about this data table can be found in the [DNA Methylation](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/DNA%20Methylation.ipynb) notebook.  Note that individual chromosome-specific DNA Methylation tables are also available to cut down on the amount of data that you may need to query (depending on yoru use case).  
# 
# 
# - **Protein_RPPA_data**:  This table contains the normalized Level-3 protein expression levels based on each antibody used to probe the sample.  More information about how this data was generated by the RPPA Core Facility at MD Anderson can be found [here](https://wiki.nci.nih.gov/display/TCGA/Protein+Array+Data+Format+Specification#ProteinArrayDataFormatSpecification-Expression-Protein), and more information about this data table can be found in the [Protein expression](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Protein%20expression.ipynb) notebook.
# 
# 
# - **Somatic_Mutation_calls**: This table contains annotated somatic mutation calls.  All current MAF (Mutation Annotation Format) files were annotated using [Oncotator](http://onlinelibrary.wiley.com/doi/10.1002/humu.22771/abstract;jsessionid=15E7960BA5FEC21EE608E6D262390C52.f01t04) v1.5.1.0, and merged into a single table.  More information about this data table can be found in the [Somatic Mutations](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Somatic%20Mutations.ipynb) notebook, including an example of how to use the [Tute Genomics annotations database in BigQuery](http://googlegenomics.readthedocs.org/en/latest/use_cases/annotate_variants/tute_annotation.html).
# 
# 
# - **mRNA_BCGSC_HiSeq_RPKM**: This table contains mRNAseq-based gene expression data produced by the [BC Cancer Agency](http://www.bcgsc.ca/).  (For details about a very similar table, take a look at a [notebook](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/UNC%20HiSeq%20mRNAseq%20gene%20expression.ipynb) describing the other mRNAseq gene expression table.)
# 
# 
# - **mRNA_UNC_HiSeq_RSEM**: This table contains mRNAseq-based gene expression data produced by [UNC Lineberger](https://unclineberger.org/).  More information about this data table can be found in the [UNC HiSeq mRNAseq gene expression](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/UNC%20HiSeq%20mRNAseq%20gene%20expression.ipynb) notebook.
# 
# 
# - **miRNA_expression**: This table contains miRNAseq-based expression data for mature microRNAs produced by the [BC Cancer Agency](http://www.bcgsc.ca/).  More information about this data table can be found in the [microRNA expression](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/BCGSC%20microRNA%20expression.ipynb) notebook.
# 

# ### Where to start?
# We suggest that you start with the two "Creating TCGA cohorts" notebooks ([part 1](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%201.ipynb) and [part 2](https://github.com/isb-cgc/examples-Python/blob/master/notebooks/Creating%20TCGA%20cohorts%20--%20part%202.ipynb)) which describe and make use of the Clinical and Biospecimen tables.  From there you can delve into the various molecular data tables as well as the Annotations table.  For now these sample notebooks are intentionally relatively simple and do not do any analysis that integrates data from multiple tables but once you have a grasp of how to use the data, developing your own more complex analyses should not be difficult.  You could even contribute an example back to our github repository!  You are also welcome to submit bug reports, comments, and feature-requests as [github issues](https://github.com/isb-cgc/examples-Python/issues).
# 

# ### A note about BigQuery tables and "tidy data"
# You may be used to thinking about a molecular data table such as a gene-expression table as a matrix where the rows are genes and the columns are samples (or *vice versa*).  These BigQuery tables instead use the [tidy data](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) approach, with each "cell" from the traditional data-matrix becoming a single row in the BigQuery table.  A 10,000 gene x 500 sample matrix would therefore become a 5,000,000 row BigQuery table.
# 

# # Copy Number segments (Broad)
# 
# The goal of this notebook is to introduce you to the Copy Number (CN) segments BigQuery table.
# 
# This table contains all available TCGA Level-3 copy number data produced by the Broad Institute using the Affymetrix Genome Wide SNP6 array, as of July 2016.  The most recent archives (*eg* ``broad.mit.edu_UCEC.Genome_Wide_SNP_6.Level_3.143.2013.0``) for each of the 33 tumor types was downloaded from the DCC, and data extracted from all files matching the pattern ``%_nocnv_hg19.seg.txt``. Each of these segmentation files has six columns: ``Sample``, ``Chromosome``, ``Start``, ``End``, ``Num_Probes``, and ``Segment_Mean``.  During ETL the sample identifer contained in the segmentation files was mapped to the TCGA aliquot barcode based on the SDRF file in the associated mage-tab archive.
# 
# In order to work with BigQuery, you need to import the python bigquery module (`gcp.bigquery`) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
cn_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Copy_Number_segments')


# From now on, we will refer to this table using this variable ($cn_BQtable), but we could just as well explicitly give the table name each time.
# 
# Let's start by taking a look at the table schema:
# 

get_ipython().magic('bigquery schema --table $cn_BQtable')


# Unlike most other molecular data types in which measurements are available for a common set of genes, CpG probes, or microRNAs, this data is produced using a data-driven approach for each aliquot independently.  As a result, the number, sizes and positions of these segments can vary widely from one sample to another.
# 
# Each copy-number segment produced using the CBS (Circular Binary Segmentation) algorithm is described by the genomic extents of the segment (chromosome, start, and end), the number of SNP6 probes contained within that segment, and the estimated mean copy-number value for that segment.  Each row in this table represents a single copy-number segment in a single sample.
# 
# The ``Segment_Mean`` is the base2 log(copynumber/2), centered at 0.  Positive values represent amplifications (CN>2), and negative values represent deletions (CN<2).  Although within each cell, the number of copies of a particular region of DNA must be an integer, these measurements are not single-cell measurements but are based on a heterogenous sample.  If 50% of the cells have 2 copies of a particular DNA segment, and 50% of the cells have 3 copies, this will result in an estimated copy number value  of 2.5, which becomes 1.32 after the log transformation.
# 

# Let's count up the number of unique patients, samples and aliquots mentioned in this table.  We will do this by defining a very simple parameterized query.  (Note that when using a variable for the table name in the FROM clause, you should not also use the square brackets that you usually would if you were specifying the table name as a string.)
# 

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')


fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = cn_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=cn_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# Unlike most other molecular data types, in addition to data being available from each tumor sample, data is also typically available from a matched "blood normal" sample.  As we can see from the previous queries, there are roughly twice as many samples as there are patients (aka participants).  The total number of rows in this table is ~2.5 million, and the average number of segments for each aliquot is ~116 (although the distribution is highly skewed as we will see shortly).
# 

# Let's count up the number of samples using the ``SampleTypeLetterCode`` field:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  SampleTypeLetterCode,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    SampleTypeLetterCode,\n    SampleBarcode\n  FROM\n    $cn_BQtable\n  GROUP BY\n    SampleTypeLetterCode,\n    SampleBarcode )\nGROUP BY\n  SampleTypeLetterCode\nORDER BY\n  n DESC')


# As shown in the results of this last query, most samples are primary tumor samples (TP), and in most cases the matched-normal sample is a "normal blood" (NB) sample, although many times it is a "normal tissue" (NT) sample.  You can find a description for each of these sample type codes in the TCGA [Code Tables Report](https://tcga-data.nci.nih.gov/datareports/codeTablesReport.htm).
# 

# In order to get a better feel for the data in this table, let's take a look at the range of values and the distributions of segment lengths, mean segment values, and number of probes contributing to each segment.
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  MIN(Length) AS minLength,\n  MAX(Length) AS maxLength,\n  AVG(Length) AS avgLength,\n  STDDEV(Length) AS stdLength,\n  MIN(Num_Probes) AS minNumProbes,\n  MAX(Num_Probes) AS maxNumProbes,\n  AVG(Num_Probes) AS avgNumProbes,\n  STDDEV(Num_Probes) AS stdNumProbes,\n  MIN(Segment_Mean) AS minCN,\n  MAX(Segment_Mean) AS maxCN,\n  AVG(Segment_Mean) AS avgCN,\n  STDDEV(Segment_Mean) AS stdCN,\nFROM (\n  SELECT\n    Start,\n    END,\n    (End-Start+1) AS Length,\n    Num_Probes,\n    Segment_Mean\n  FROM\n    $cn_BQtable )')


# Segment lengths range from just 1 bp all the way up to entire chromosome arms, and the range of segment mean values is from -8.7 to +10.5 (average = -0.28, standard deviation = 1.0)
# 

# Now we'll use matplotlib to create some simple visualizations.
# 

import numpy as np
import matplotlib.pyplot as plt


# For the segment means, let's invert the log-transform and then bin the values to see what the distribution looks like:
# 

get_ipython().run_cell_magic('sql', '--module getCNhist', '\nSELECT\n  lin_bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    Segment_Mean,\n    (2.*POW(2,Segment_Mean)) AS lin_CN,\n    INTEGER(((2.*POW(2,Segment_Mean))+0.50)/1.0) AS lin_bin\n  FROM\n    $t\n  WHERE\n    ( (End-Start+1)>1000 AND SampleTypeLetterCode="TP" ) )\nGROUP BY\n  lin_bin\nHAVING\n  ( n > 2000 )\nORDER BY\n  lin_bin ASC')


CNhist = bq.Query(getCNhist,t=cn_BQtable).results().to_dataframe()
bar_width=0.80
plt.bar(CNhist['lin_bin']+0.1,CNhist['n'],bar_width,alpha=0.8);
plt.xticks(CNhist['lin_bin']+0.5,CNhist['lin_bin']);
plt.title('Histogram of Average Copy-Number');
plt.ylabel('# of segments');
plt.xlabel('integer copy-number');


# The histogram illustrates that the vast majority of the CN segments have a copy-number value near 2, as expected, with significant tails on either side representing deletions (left) and amplifications (right).
# 

# Let's take a look at the distribution of segment lengths now.  First we'll use 1Kb bins and look at segments with lengths up to 1 Mb.  
# 

get_ipython().run_cell_magic('sql', '--module getSLhist_1k', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')


SLhist_1k = bq.Query(getSLhist_1k,t=cn_BQtable).results().to_dataframe()
plt.plot(SLhist_1k['bin'],SLhist_1k['n'],'ro:');
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (Kb)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths');


# As expected, shorter segment lengths dominate, and between 1Kb and 1Mb it appears that segment lengths follow a power-law distribution.
# 

# Let's have a closer look at the shorter segments, under 1Kb in length:
# 

get_ipython().run_cell_magic('sql', '--module getSLhist', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000 AND SampleTypeLetterCode="TP" )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')


SLhist = bq.Query(getSLhist,t=cn_BQtable).results().to_dataframe()
plt.plot(SLhist['bin'],SLhist['n'],'ro:');
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (bp)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths (<1Kb)');


# At this finer scale, we see that the most comment segment length is ~15bp.
# 

# Let's go back and take another look at the medium-length CN segments and see what happens when we separate out the amplifications and deletions.  We'll use queries similar to the ``getSLhist_1k`` query above, but add another ``WHERE`` clause to look at amplifications and deletions respectively.
# 

get_ipython().run_cell_magic('sql', '--module getSLhist_1k_del', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" AND Segment_Mean<-0.7 )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')


get_ipython().run_cell_magic('sql', '--module getSLhist_1k_amp', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" AND Segment_Mean>0.7 )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')


SLhistDel = bq.Query(getSLhist_1k_del,t=cn_BQtable).results().to_dataframe()
SLhistAmp = bq.Query(getSLhist_1k_amp,t=cn_BQtable).results().to_dataframe()


plt.plot(SLhist_1k['bin'],SLhist_1k['n'],'ro:');
plt.plot(SLhistDel['bin'],SLhistDel['n'],'bo-')
plt.plot(SLhistAmp['bin'],SLhistDel['n'],'go-',alpha=0.3)
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (Kb)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths');


# The amplification and deletion distributions are nearly identical and still seem to roughly follow a power-law distribution.  We can also infer from this graph that a majority of the segments less than 10Kb in length are either amplifications or deletions, while ~90% of the segments of lengths >100Kb are copy-number neutral.
# 

# Before we leave this dataset, let's look at how we might analyze the copy-number as it relates to a particular gene of interest.  This next parameterized query looks for all copy-number segments overlapping a specific genomic region and computes some statistics after grouping by sample.
# 

get_ipython().run_cell_magic('sql', '--module getGeneCN', '\nSELECT\n  SampleBarcode, \n  AVG(Segment_Mean) AS avgCN,\n  MIN(Segment_Mean) AS minCN,\n  MAX(Segment_Mean) AS maxCN,\nFROM\n  $t\nWHERE\n  ( SampleTypeLetterCode=$sampleType\n    AND Num_Probes > 10\n    AND Chromosome=$geneChr\n    AND ( (Start<$geneStart AND End>$geneStop)\n       OR (Start<$geneStop  AND End>$geneStop)\n       OR (Start>$geneStart AND End<$geneStop) ) )\nGROUP BY\n  SampleBarcode')


# Now we'll use this query to get copy-number statistics for three widely-studied genes: EGFR, MYC and TP53.
# 

# EGFR gene coordinates  
geneChr = "7"
geneStart = 55086725
geneStop = 55275031
egfrCN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()

# MYC gene coordinates
geneChr = "8"
geneStart = 128748315
geneStop = 128753680
mycCN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()

# TP53 gene coordinates
geneChr = "17"
geneStart = 7571720
geneStop = 7590868
tp53CN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()


# And now we'll take a look at histograms of the average copy-number for these three genes.  TP53 (in green) shows a significant number of partial deletions (CN<0), while MYC (in blue) shows some partial amplifications -- more frequently than EGFR, while EGFR (pale red) shows a few extreme amplifications (log2(CN/2) > 2). The final figure shows the same histograms on a semi-log plot to bring up the rarer events.
# 

binWidth = 0.2
binVals = np.arange(-2+(binWidth/2.), 6-(binWidth/2.), binWidth)
plt.hist(tp53CN['avgCN'],bins=binVals,normed=False,color='green',alpha=0.9,label='TP53');
plt.hist(mycCN ['avgCN'],bins=binVals,normed=False,color='blue',alpha=0.7,label='MYC');
plt.hist(egfrCN['avgCN'],bins=binVals,normed=False,color='red',alpha=0.5,label='EGFR');
plt.legend(loc='upper right');


plt.hist(tp53CN['avgCN'],bins=binVals,normed=False,color='green',alpha=0.9,label='TP53');
plt.hist(mycCN ['avgCN'],bins=binVals,normed=False,color='blue',alpha=0.7,label='MYC');
plt.hist(egfrCN['avgCN'],bins=binVals,normed=False,color='red',alpha=0.5,label='EGFR');
plt.yscale('log');
plt.legend(loc='upper right');





# # UNC HiSeq mRNAseq gene expression (RSEM)
# 
# The goal of this notebook is to introduce you to the mRNAseq gene expression BigQuery table.
# 
# This table contains all available TCGA Level-3 gene expression data produced by UNC's RNAseqV2 pipeline using the Illumina HiSeq platform, as of July 2016.  The most recent archive (*eg* ``unc.edu_BRCA.IlluminaHiSeq_RNASeqV2.Level_3.1.11.0``) for each of the 33 tumor types was downloaded from the DCC, and data extracted from all files matching the pattern ``%.rsem.genes.normalized_results``. Each of these raw “RSEM genes normalized results” files has two columns: gene_id and normalized_count.  The gene_id string contains two parts: the gene symbol, and the Entrez gene ID, separated by **|**  *eg*: **`TP53|7157`**.  During ETL, the gene_id string is split and the gene symbol is stored in the ``original_gene_symbol`` field, and the Entrez gene ID is stored in the ``gene_id`` field.  In addition, the Entrez ID is used to look up the current HGNC approved gene symbol, which is stored in the ``HGNC_gene_sybmol`` field. 
# 
# In order to work with BigQuery, you need to import the python bigquery module (`gcp.bigquery`) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
mRNAseq_BQtable = bq.Table('isb-cgc:tcga_201607_beta.mRNA_UNC_HiSeq_RSEM')


# From now on, we will refer to this table using this variable ($mRNAseq_BQtable), but we could just as well explicitly give the table name each time.
# 
# Let's start by taking a look at the table schema:
# 

get_ipython().magic('bigquery schema --table $mRNAseq_BQtable')


# Now let's count up the number of unique patients, samples and aliquots mentioned in this table.  We will do this by defining a very simple parameterized query.  (Note that when using a variable for the table name in the FROM clause, you should not also use the square brackets that you usually would if you were specifying the table name as a string.)
# 

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')


fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = mRNAseq_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=mRNAseq_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# We can do the same thing to look at how many unique gene symbols and gene ids exist in the table:
# 

fieldList = ['original_gene_symbol', 'HGNC_gene_symbol', 'gene_id']
for aField in fieldList:
  field = mRNAseq_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=mRNAseq_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# Based on the counts, we can see that there are a few instances where the original gene symbol (from the underlying TCGA data file), or the HGNC gene symbol or the gene id (also from the original TCGA data file) is missing, but for the majority of genes, all three values should be available and for the most part the original gene symbol and the HGNC gene symbol that was added during ETL should all match up.  This next query will generate the complete list of genes for which none of the identifiers are null, and where the original gene symbol and the HGNC gene symbol match.  This list has over 18000 genes in it.
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  HGNC_gene_symbol,\n  original_gene_symbol,\n  gene_id\nFROM\n  $mRNAseq_BQtable\nWHERE\n  ( original_gene_symbol IS NOT NULL\n    AND HGNC_gene_symbol IS NOT NULL\n    AND original_gene_symbol=HGNC_gene_symbol\n    AND gene_id IS NOT NULL )\nGROUP BY\n  original_gene_symbol,\n  HGNC_gene_symbol,\n  gene_id\nORDER BY\n  HGNC_gene_symbol')


# We might also want to know how often the gene symbols do not agree:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  HGNC_gene_symbol,\n  original_gene_symbol,\n  gene_id\nFROM\n  $mRNAseq_BQtable\nWHERE\n  ( original_gene_symbol IS NOT NULL\n    AND HGNC_gene_symbol IS NOT NULL\n    AND original_gene_symbol!=HGNC_gene_symbol\n    AND gene_id IS NOT NULL )\nGROUP BY\n  original_gene_symbol,\n  HGNC_gene_symbol,\n  gene_id\nORDER BY\n  HGNC_gene_symbol')


# BigQuery is not just a "look-up" service -- you can also use it to perform calculations.  In this next query, we take a look at the mean, standard deviation, and coefficient of variation for the expression of EGFR, within each tumor-type, as well as the number of primary tumor samples that went into each summary statistic.
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Study,\n  n,\n  exp_mean,\n  exp_sigma,\n  (exp_sigma/exp_mean) AS exp_cv\nFROM (\n  SELECT\n    Study,\n    AVG(LOG2(normalized_count+1)) AS exp_mean,\n    STDDEV_POP(LOG2(normalized_count+1)) AS exp_sigma,\n    COUNT(AliquotBarcode) AS n\n  FROM\n    $mRNAseq_BQtable\n  WHERE\n    ( SampleTypeLetterCode="TP"\n      AND HGNC_gene_symbol="EGFR" )\n  GROUP BY\n    Study )\nORDER BY\n  exp_sigma DESC')


# We can also easily move the gene-symbol out of the WHERE clause and into the SELECT and GROUP BY clauses and have BigQuery do this same calculation over *all* genes and all tumor types.  This time we will use the `--module` option to define the query and then call it in the next cell from python.
# 

get_ipython().run_cell_magic('sql', '--module highVar', '\nSELECT\n  Study,\n  HGNC_gene_symbol,\n  n,\n  exp_mean,\n  exp_sigma,\n  (exp_sigma/exp_mean) AS exp_cv\nFROM (\n  SELECT\n    Study,\n    HGNC_gene_symbol,\n    AVG(LOG2(normalized_count+1)) AS exp_mean,\n    STDDEV_POP(LOG2(normalized_count+1)) AS exp_sigma,\n    COUNT(AliquotBarcode) AS n\n  FROM\n    $t\n  WHERE\n    ( SampleTypeLetterCode="TP" )\n  GROUP BY\n    Study,\n    HGNC_gene_symbol )\nORDER BY\n  exp_sigma DESC')


# Once we have defined a query, we can put it into a python object and print out the SQL statement to make sure it looks as expected:
# 

q = bq.Query(highVar,t=mRNAseq_BQtable)
print q.sql


# And then we can run it and save the results in another python object:
# 

r = bq.Query(highVar,t=mRNAseq_BQtable).results()


#r.to_dataframe()


# Since the result of the previous query is quite large (over 600,000 rows representing ~20,000 genes x ~30 tumor types), we might want to put those results into one or more subsequent queries that further refine these results, for example:
# 

get_ipython().run_cell_magic('sql', '--module hv_genes', '\nSELECT *\nFROM ( $hv_result )\nHAVING\n  ( exp_mean > 6.\n    AND n >= 200\n    AND exp_cv > 0.5 )\nORDER BY\n  exp_cv DESC')


bq.Query(hv_genes,hv_result=r).results().to_dataframe()





# # Creating TCGA cohorts  (part 1)
# 
# This notebook will show you how to create a TCGA cohort using the publicly available TCGA BigQuery tables that the [ISB-CGC](http://isb-cgc.org) project has produced based on the open-access [TCGA](http://cancergenome.nih.gov/) data available at the [Data Portal](https://tcga-data.nci.nih.gov/tcga/).  You will need to have access to a Google Cloud Platform (GCP) project in order to use BigQuery.  If you don't already have one, you can sign up for a [free-trial](https://cloud.google.com/free-trial/) or contact [us](mailto://info@isb-cgc.org) and become part of the community evaluation phase of our [Cancer Genomics Cloud pilot](https://cbiit.nci.nih.gov/ncip/nci-cancer-genomics-cloud-pilots).
# 
# We are not attempting to provide a thorough BigQuery or IPython tutorial here, as a wealth of such information already exists.  Here are some links to some resources that you might find useful: 
# * [BigQuery](https://cloud.google.com/bigquery/what-is-bigquery), 
# * the BigQuery [web UI](https://bigquery.cloud.google.com/) where you can run queries interactively, 
# * [IPython](http://ipython.org/) (now known as [Jupyter](http://jupyter.org/)), and 
# * [Cloud Datalab](https://cloud.google.com/datalab/) the recently announced interactive cloud-based platform that this notebook is being developed on. 
# 
# There are also many tutorials and samples available on github (see, in particular, the [datalab](https://github.com/GoogleCloudPlatform/datalab) repo and the [Google Genomics](  https://github.com/googlegenomics) project).
# 
# OK then, let's get started!  In order to work with BigQuery, the first thing you need to do is import the bigquery module:
# 

import gcp.bigquery as bq


# The next thing you need to know is how to access the specific tables you are interested in.  BigQuery tables are organized into datasets, and datasets are owned by a specific GCP project.  The tables we will be working with in this notebook are in a dataset called **`tcga_201607_beta`**, owned by the **`isb-cgc`** project.  A full table identifier is of the form `<project_id>:<dataset_id>.<table_id>`.  Let's start by getting some basic information about the tables in this dataset:
# 

d = bq.DataSet('isb-cgc:tcga_201607_beta')
for t in d.tables():
  print '%10d rows  %12d bytes   %s'       % (t.metadata.rows, t.metadata.size, t.name.table_id)


# In this tutorial, we are going to look at a few different ways that we can use the information in these tables to create cohorts.  Now, you maybe asking what we mean by "cohort" and why you might be interested in *creating* one, or maybe what it even means to "create" a cohort.  The TCGA dataset includes clinical, biospecimen, and molecular data from over 10,000 cancer patients who agreed to be a part of this landmark research project to build [The Cancer Genome Atlas](http://cancergenome.nih.gov/).  This large dataset was originally organized and studied according to [cancer type](http://cancergenome.nih.gov/cancersselected) but now that this multi-year project is nearing completion, with over 30 types of cancer and over 10,000 tumors analyzed, **you** have the opportunity to look at this dataset from whichever angle most interests you.  Maybe you are particularly interested in early-onset cancers, or gastro-intestinal cancers, or a specific type of genetic mutation.  This is where the idea of a "cohort" comes in.  The original TCGA "cohorts" were based on cancer type (aka "study"), but now you can define a cohort based on virtually any clinical or molecular feature by querying these BigQuery tables.  A cohort is simply a list of samples, using the [TCGA barcode](https://wiki.nci.nih.gov/display/TCGA/TCGA+barcode) system.  Once you have created a cohort you can use it in any number of ways: you could further explore the data available for one cohort, or compare one cohort to another, for example.
# 
# In the rest of this tutorial, we will create several different cohorts based on different motivating research questions.  We hope that these examples will provide you with a starting point from which you can build, to answer your own research questions.
# 

# ### Exploring the Clinical data table
# Let's start by looking at the clinical data table.  The TCGA dataset contains a few very basic clinical data elements for almost all patients, and contains additional information for some tumor types only.  For example smoking history information is generally available only for lung cancer patients, and BMI (body mass index) is only available for tumor types where that is a known significant risk factor.  Let's take a look at the clinical data table and see how many different pieces of information are available to us:
# 

get_ipython().magic('bigquery schema --table isb-cgc:tcga_201607_beta.Clinical_data')


# That's a lot of fields!  We can also get at the schema programmatically:
# 

table = bq.Table('isb-cgc:tcga_201607_beta.Clinical_data')
if ( table.exists() ):
    fieldNames = map(lambda tsf: tsf.name, table.schema)
    fieldTypes = map(lambda tsf: tsf.data_type, table.schema)
    print " This table has %d fields. " % ( len(fieldNames) )
    print " The first few field names and types are: " 
    print "     ", fieldNames[:5]
    print "     ", fieldTypes[:5]
else: 
    print " There is no existing table called %s:%s.%s" % ( table.name.project_id, table.name.dataset_id, table.name.table_id )


# Let's look at these fields and see which ones might be the most "interesting", by looking at how many times they are filled-in (not NULL), or how much variation exists in the values.  If we wanted to look at just a single field, "tobacco_smoking_history" for example, we could use a very simple query to get a basic summary:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT tobacco_smoking_history, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nGROUP BY tobacco_smoking_history\nORDER BY n DESC')


# But if we want to loop over *all* fields and get a sense of which fields might provide us with useful criteria for specifying a cohort, we'll want to automate that.  We'll put a threshold on the minimum number of patients that we expect information for, and the maximum number of unique values (since fields such as the "ParticipantBarcode" will be unique for every patient and, although we will need that field later, it's probably not useful for defining a cohort).
# 

numPatients = table.metadata.rows
print " The %s table describes a total of %d patients. " % ( table.name.table_id, numPatients )

# let's set a threshold for the minimum number of values that a field should have,
# and also the maximum number of unique values
minNumPatients = int(numPatients*0.80)
maxNumValues = 50

numInteresting = 0
iList = []
for iField in range(len(fieldNames)):
  aField = fieldNames[iField]
  aType = fieldTypes[iField]
  try:
    qString = "SELECT {0} FROM [{1}]".format(aField,table)
    query = bq.Query(qString)
    df = query.to_dataframe()
    summary = df[str(aField)].describe()
    if ( aType == "STRING" ):
      topFrac = float(summary['freq'])/float(summary['count'])
      if ( summary['count'] >= minNumPatients ):
        if ( summary['unique'] <= maxNumValues and summary['unique'] > 1 ):
          if ( topFrac < 0.90 ):
            numInteresting += 1
            iList += [aField]
            print "     > %s has %d values with %d unique (%s occurs %d times) "               % (str(aField), summary['count'], summary['unique'], summary['top'], summary['freq'])
    else:
      if ( summary['count'] >= minNumPatients ):
        if ( summary['std'] > 0.1 ):
          numInteresting += 1
          iList += [aField]
          print "     > %s has %d values (mean=%.0f, sigma=%.0f) "             % (str(aField), summary['count'], summary['mean'], summary['std'])
  except:
    pass

print " "
print " Found %d potentially interesting features: " % numInteresting
print "   ", iList


# The above helps us narrow down on which fields are likely to be the most useful, but if you have a specific interest, for example in menopause or HPV status, you can still look at those in more detail very easily: 
# 

get_ipython().run_cell_magic('sql', '', 'SELECT menopause_status, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE menopause_status IS NOT NULL\nGROUP BY menopause_status\nORDER BY n DESC')


# We might wonder which specific tumor types have menopause information:
# 

get_ipython().run_cell_magic('sql', '', 'SELECT Study, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE menopause_status IS NOT NULL\nGROUP BY Study\nORDER BY n DESC')


get_ipython().run_cell_magic('sql', '', 'SELECT hpv_status, hpv_calls, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE hpv_status IS NOT NULL\nGROUP BY hpv_status, hpv_calls\nHAVING n > 20\nORDER BY n DESC')


# ### TCGA Annotations
# 
# An additional factor to consider, when creating a cohort is that there may be additional information that might lead one to exclude a particular patient from a cohort.  In certain instances, patients have been redacted or excluded from analyses for reasons such as prior treatment, etc, but since different researchers may have different criteria for using or excluding certain patients or certain samples from their analyses, in many cases the data is still available while at the same time "annotations" may have been entered into a searchable [database](https://tcga-data.nci.nih.gov/annotations/).  These annotations have also been uploaded into a BigQuery table and can be used in conjuction with the other BigQuery tables.
# 

# ### Early-onset Breast Cancer
# 
# Now that we have a better idea of what types of information is available in the Clinical data table, let's create a cohort consisting of female breast-cancer patients, diagnosed at the age of 50 or younger.
# 

# In this next code cell, we define several queries within a **`module`** which allows us to use them both individually and by reference in the final, main query.  
# + the first query, called **`select_on_annotations`**, finds all patients in the Annotations table which have either been 'redacted' or had 'unacceptable prior treatment';  
# + the second query, **`select_on_clinical`** selects all female breast-cancer patients who were diagnosed at age 50 or younger, while also pulling out a few additional fields that might be of interest;  and
# + the final query joins these two together and returns just those patients that meet the clinical-criteria and do **not** meet the exclusion-criteria.
# 

get_ipython().run_cell_magic('sql', '--module createCohort_and_checkAnnotations', '\nDEFINE QUERY select_on_annotations\nSELECT\n  ParticipantBarcode,\n  annotationCategoryName AS categoryName,\n  annotationClassification AS classificationName\nFROM\n  [isb-cgc:tcga_201607_beta.Annotations]\nWHERE\n  ( itemTypeName="Patient"\n    AND (annotationCategoryName="History of unacceptable prior treatment related to a prior/other malignancy"\n      OR annotationClassification="Redaction" ) )\nGROUP BY\n  ParticipantBarcode,\n  categoryName,\n  classificationName\n\nDEFINE QUERY select_on_clinical\nSELECT\n  ParticipantBarcode,\n  vital_status,\n  days_to_last_known_alive,\n  ethnicity,\n  histological_type,\n  menopause_status,\n  race\nFROM\n  [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE\n  ( Study="BRCA"\n    AND age_at_initial_pathologic_diagnosis<=50\n    AND gender="FEMALE" )\n\nSELECT\n  c.ParticipantBarcode AS ParticipantBarcode\nFROM (\n  SELECT\n    a.categoryName,\n    a.classificationName,\n    a.ParticipantBarcode,\n    c.ParticipantBarcode,\n  FROM ( $select_on_annotations ) AS a\n  OUTER JOIN EACH \n       ( $select_on_clinical ) AS c\n  ON\n    a.ParticipantBarcode = c.ParticipantBarcode\n  WHERE\n    (a.ParticipantBarcode IS NOT NULL\n      OR c.ParticipantBarcode IS NOT NULL)\n  ORDER BY\n    a.classificationName,\n    a.categoryName,\n    a.ParticipantBarcode,\n    c.ParticipantBarcode )\nWHERE\n  ( a.categoryName IS NULL\n    AND a.classificationName IS NULL\n    AND c.ParticipantBarcode IS NOT NULL )\nORDER BY\n  c.ParticipantBarcode')


# Here we explicitly call just the first query in the module, and we get a list of 212 patients with one of these disqualifying annotations:
# 

bq.Query(createCohort_and_checkAnnotations.select_on_annotations).results().to_dataframe()


# and here we explicitly call just the second query, resulting in 329 patients:
# 

bq.Query(createCohort_and_checkAnnotations.select_on_clinical).results().to_dataframe()


# and finally we call the main query:
# 

bq.Query(createCohort_and_checkAnnotations).results().to_dataframe()


# Note that we didn't need to call each sub-query individually, we could have just called the main query and gotten the same result.  As you can see, two patients that met the clinical select criteria (which returned 329 patients) were excluded from the final result (which returned 327 patients).
# 

# Before we leave off, here are a few useful tricks for working with BigQuery in Cloud Datalab:
# + if you want to see the raw SQL, you can just build the query and then print it out (this might be useful, for example, in debugging a query -- you can copy and paste the SQL directly into the BigQuery Web UI);
# + if you want to see how much data and which tables are going to be touched by this data, you can use the "dry run" option.  (Notice the "cacheHit" flag -- if you have recently done a particular query, you will not be charged to repeat it since it will have been cached.)
# 

q = bq.Query(createCohort_and_checkAnnotations)
q


q.execute_dry_run()





# # DNA Methylation (JHU-USC beta values)
# 
# The goal of this notebook is to introduce you to the DNA methylation BigQuery table.
# 
# This table contains all available TCGA Level-3 DNA methylation data produced by the JHU-USC methylation pipeline using the Illumina Infinium Human Methylation 27k and 450k platforms, as of July 2016.  The most recent archives (*eg* ``jhu-usc.edu_HNSC.HumanMethylation450.Level_3.18.8.0``) for each of the 33 tumor types were downloaded from the DCC, and data extracted from all files matching the pattern ``jhu-usc.edu_%.HumanMethylation%.lvl-3.%.txt``. Each of these text files has five columns.  The first two columns contain the CpG probe id and the methylation beta value.  The additional columns contain annotation information (gene symbol(s), and chromosome and genomic coordinate for the CpG probe).  Only the CpG probe id and the beta value were extracted during ETL and stored in this BigQuery table, along with the aliquot ID (which can be found both in the text filename, and in the SDRF file in the mage-tab archive).
# 
# **WARNING**: This BigQuery table contains almost **4 billion** rows of data and is over 400 GB in size.  When experimenting with new queries, be sure to put a "LIMIT" on the results to avoid accidentally launching a query that might either take a very very long time or produce a very large results table! 
# 
# **NOTE**: For convenience, individual per-chromosome tables are also available, so if your queries are specific to a single chromosome, your queries will be faster and cheaper if you use the appropriate single-chromosome table.
# 
# In order to work with BigQuery, you need to import the python bigquery module (`gcp.bigquery`) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
meth_BQtable = bq.Table('isb-cgc:tcga_201607_beta.DNA_Methylation_betas')


# From now on, we will refer to this table using this variable ($meth_BQtable), but we could just as well explicitly give the table name each time.
# 
# Let's start by taking a look at the table schema:
# 

get_ipython().magic('bigquery schema --table $meth_BQtable')


# Let's count up the number of unique patients, samples and aliquots mentioned in this table.  Using the same approach, we can count up the number of unique CpG probes.  We will do this by defining a very simple parameterized query.  (Note that when using a variable for the table name in the FROM clause, you should not also use the square brackets that you usually would if you were specifying the table name as a string.)
# 

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 500000) AS n\nFROM $t')


fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode', 'Probe_Id']
for aField in fieldList:
  field = meth_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=meth_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# As mentioned above, two different platforms were used to measure DNA methylation.  The annotations from Illumina are also available in a BigQuery table:
# 

methAnnot = bq.Table('isb-cgc:platform_reference.methylation_annotation')


get_ipython().magic('bigquery schema --table $methAnnot')


# Given the coordinates for a gene of interest, we can find the associated methylation probes.
# 

get_ipython().run_cell_magic('sql', '--module getGeneProbes', '\nSELECT\n  IlmnID, Methyl27_Loci, CHR, MAPINFO\nFROM\n  $t\nWHERE\n  ( CHR=$geneChr\n    AND ( MAPINFO>$geneStart AND MAPINFO<$geneStop ) )\nORDER BY\n  Methyl27_Loci DESC, \n  MAPINFO ASC')


# MLH1 gene coordinates (+/- 2500 bp)
geneChr = "3"
geneStart = 37034841 - 2500
geneStop  = 37092337 + 2500
mlh1Probes = bq.Query(getGeneProbes,t=methAnnot,geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results()


# There are a total of 50 methlyation probes in and near the MLH1 gene, although only 6 of them are on both the 27k and the 450k versions of the platform.
# 

mlh1Probes


# We can now use this list of CpG probes as a filter on the data table to extract all of the methylation data across all tumor types for MLH1:
# 

get_ipython().run_cell_magic('sql', '--module getMLH1methStats', '\nSELECT \n  cpg.IlmnID AS Probe_Id,\n  cpg.Methyl27_Loci AS Methyl27_Loci,\n  cpg.CHR AS Chr,\n  cpg.MAPINFO AS Position,\n  data.beta_stdev AS beta_stdev,\n  data.beta_mean AS beta_mean,\n  data.beta_min AS beta_min,\n  data.beta_max AS beta_max\nFROM (\n  SELECT *\n  FROM $mlh1Probes \n) AS cpg\nJOIN (\n  SELECT \n    Probe_Id,\n    STDDEV(beta_value) beta_stdev,\n    AVG(beta_value) beta_mean,\n    MIN(beta_value) beta_min,\n    MAX(beta_value) beta_max\n    FROM $meth_BQtable\n    WHERE ( SampleTypeLetterCode=$sampleType )\n    GROUP BY Probe_Id\n) AS data\nON \n  cpg.IlmnID = data.Probe_Id\nORDER BY\n  Position ASC')


qTP = bq.Query(getMLH1methStats,mlh1Probes=mlh1Probes,meth_BQtable=meth_BQtable,sampleType="TP")
rTP = qTP.results().to_dataframe()
rTP.describe()


qNT = bq.Query(getMLH1methStats,mlh1Probes=mlh1Probes,meth_BQtable=meth_BQtable,sampleType="NT")
rNT = qNT.results().to_dataframe()
rNT.describe()


import numpy as np
import matplotlib.pyplot as plt


bins=range(1,len(rTP)+1)
#print bins
plt.bar(bins,rTP['beta_mean'],color='red',alpha=0.8,label='Primary Tumor');
plt.bar(bins,rNT['beta_mean'],color='blue',alpha=0.4,label='Normal Tissue');
plt.legend(loc='upper left');
plt.title('MLH1 DNA methylation: average');


plt.bar(bins,rTP['beta_stdev'],color='red',alpha=0.8,label='Primary Tumor');
plt.bar(bins,rNT['beta_stdev'],color='blue',alpha=0.4,label='Normal Tissue');
plt.legend(loc='upper right');
plt.title('MLH1 DNA methylation: standard deviation');


# From the figures above, we can see that, with the exception of the CpG probes near the 3' end of MLH1, the primary tumor samples have a slightly higher average methylation, with significantly greater variability.
# 

# # TCGA Annotations
# 
# The goal of this notebook is to introduce you to the TCGA Annotations BigQuery table.  You can find more detail about [Annotations](https://wiki.nci.nih.gov/display/TCGA/Introduction+to+Annotations) on the [TCGA Wiki](https://wiki.nci.nih.gov/display/TCGA/TCGA+Home), but the key things to know are:
# * an annotation can refer to any "type" of TCGA "item" (*eg* patient, sample, portion, slide, analyte or aliquot), and
# * each annotation has a "classification" and a "category", both of which are drawn from controlled vocabularies.
# 
# The current set of annotation classifications includes: Redaction, Notification, CenterNotification, and Observation.  The authority for Redactions and Notifications is the BCR (Biospecimen Core Resource), while CenterNotifications can come from any of the data-generating centers (GSC or GCC), and Observations from any authorized TCGA personnel.  Within each classification type, there are several categories.  
# 
# We will look at these further by querying directly on the Annotations table.
# 
# Note that annotations about patients, samples, and aliquots are *separate* from the clinical, biospecimen, and molecular data, and most patients, samples, and aliquots do not in fact have any annotations associated with them.  It can be important, however, when creating a cohort or analyzing the molecular data associated with a cohort, to check for the existence of annotations.
# 
# As usual, in order to work with BigQuery, you need to import the python bigquery module (gcp.bigquery) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
annotations_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Annotations')


# ### Schema
# Let's start by looking at the schema to see what information is available from this table:
# 

get_ipython().magic('bigquery schema --table $annotations_BQtable')


# ### Item Types
# 
# Most of the schema fields come directly from the TCGA Annotations.  First and foremost, an annotation is associated with an **itemType**, as described above.  This can be a patient, an aliquot, etc.  Let's see what the breakdown is of annotations according to item-type:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT itemTypeName, COUNT(*) AS n\nFROM $annotations_BQtable\nGROUP BY itemTypeName\nORDER BY n DESC')


# The length of the barcode in the ``itemBarcode`` field will depend on the value in the ``itemTypeName`` field:  if the itemType is "Patient", then the barcode will be something like ``TCGA-E2-A15J``, whereas if the itemType is "Aliquot", the barcode will be a full-length barcode, *eg* ``TCGA-E2-A15J-10A-01D-a12N-01``.  
# 

# ### Annotation Classifications and Categories
# 
# The next most important pieces of information about an annotation are the "classification" and "category".  Each of these comes from a controlled vocabulary and each "classification" has a specific set of allowed "categories".
# 
# One important thing to understand is that if an aliquot carries some sort of disqualifying annotation, in general all other data from other samples or aliquots associated with that same patient should still be usable.  On the other hand, if a *patient* carries some sort of disqualifying annotation, then that information should be considered prior to using *any* of the samples or aliquots derived from that patient.
# 

# To illustrate this, let's look at the most frequent annotation classifications and categories when the itemType is Patient:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  annotationClassification,\n  annotationCategoryName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( itemTypeName="Patient" )\nGROUP BY\n  annotationClassification,\n  annotationCategoryName\nHAVING ( n >= 50 )\nORDER BY\n  n DESC')


# The results of the previous query indicate that the majority of patient-level annotations are "Notifications", most frequently regarding prior malignancies.  In most TCGA publications, "history of unacceptable prior treatment" and "item is noncanonical" notifications are treated as disqualifying annotations, and all data associated with those patients is not used in any analysis.
# 

# Let's make a slight modification to the last query to see what types of annotation categories and classifications we see when the item type is *not* patient:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  annotationClassification,\n  annotationCategoryName,\n  itemTypeName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( itemTypeName!="Patient" )\nGROUP BY\n  annotationClassification,\n  annotationCategoryName,\n  itemTypeName\nHAVING ( n >= 50 )\nORDER BY\n  n DESC')


# The results of the previous query indicate that the vast majority of annotations are at the aliquot level, and more specifically were submitted by one of the data-generating centers, indicating that the data derived from that aliquot is "DNU" (Do Not Use).  In general, this should not affect any other aliquots derived from the same sample or any other samples derived from the same patient.
# 

# We see in the output of the previous query that a Notification that an "Item is noncanonical" can be applied to different types of items (*eg* slides and analytes).  Let's investigate this a little bit further, for example let's count up these types of annotations by study (ie tumor-type):
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Study,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( annotationCategoryName="Item is noncanonical" )\nGROUP BY\n  Study\nORDER BY\n  n DESC')


# and now let's pick one of these tumor types, and delve a little bit further:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  itemTypeName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( annotationCategoryName="Item is noncanonical"\n    AND Study="OV" )\nGROUP BY\n  itemTypeName\nORDER BY\n  n DESC')


# ### Barcodes
# 
# As described above, an annotation is specific to a single TCGA "item" and the fields ``itemTypeName`` and ``itemBarcode`` are the most important keys to understanding which TCGA item carries the annotation.  Because we use the fields ``ParticipantBarcode``, ``SampleBarcode``, and ``AliquotBarcode`` throughout our other TCGA BigQuery tables, we have added them to this table as well, but they should be interpreted with some care:  when an annotation is specific to an aliquot (*ie* ``itemTypeName="Aliquot"``), the ``ParticipantBarcode``, ``SampleBarcode``, and ``AliquotBarcode`` fields will all be set, *but* this should not be interpreted to mean that the annotation applies to all data derived from that patient.
# 
# This will be illustrated with the following two queries which extract information pertaining to a few specific patients:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-61-1916" )\nORDER BY n ASC')


get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-GN-A261" )\nORDER BY n ASC')


# As you can see in the results returned from the previous two queries, the SampleBarcode and the AliquotBarcode fields may or may not be filled in, depending on the itemTypeName.
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n annotationNoteText,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-RS-A6TP" )\nORDER BY n ASC')


# In this example, there is just one annotation relevant to this particular patient, and one has to look at the ``annotationNoteText`` to find out what the potential issue may be with this particular analyte.  Any aliquots derived from this blood-normal analyte might need to be used with care.
# 




# # Protein expression (MDAnderson RPPA)
# 
# The goal of this notebook is to introduce you to the Protein expression BigQuery table.
# 
# This table contains all available TCGA Level-3 protein expression data produced by MD Anderson's RPPA pipeline, as of July 2016.  The most recent archives (*eg* ``mdanderson.org_COAD.MDA_RPPA_Core.Level_3.2.0.0``) for each of the 32 tumor types was downloaded from the DCC, and data extracted from all files matching the pattern ``%_RPPA_Core.protein_expression%.txt``. Each of these “protein expression” files has two columns: the ``Composite Element REF`` and the ``Protein Expression``.  In addition, each mage-tab archive contains an ``antibody_annotation`` file which is parsed in order to obtain the correct mapping between antibody name, protein name, and gene symbol.  During the ETL process, portions of the protein name and the antibody name were extracted into additional columns in the table, including ``Phospho``, ``antibodySource`` and ``validationStatus``. 
# 
# In order to work with BigQuery, you need to import the python bigquery module (`gcp.bigquery`) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
rppa_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Protein_RPPA_data')


# From now on, we will refer to this table using this variable ($rppa_BQtable), but we could just as well explicitly give the table name each time.
# 
# Let's start by taking a look at the table schema:
# 

get_ipython().magic('bigquery schema --table $rppa_BQtable')


# Let's count up the number of unique patients, samples and aliquots mentioned in this table.  We will do this by defining a very simple parameterized query.  (Note that when using a variable for the table name in the FROM clause, you should not also use the square brackets that you usually would if you were specifying the table name as a string.)
# 

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')


fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = rppa_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=rppa_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


fieldList = ['Gene_Name', 'Protein_Name', 'Protein_Basename']
for aField in fieldList:
  field = rppa_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=rppa_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# Based on the counts, we can see that there are several genes for which multiple proteins are assayed, and that overall this dataset is quite small compared to most of the other datasets.  Let's look at which genes have multiple proteins assayed:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    Gene_Name,\n    Protein_Name,\n  FROM\n    $rppa_BQtable\n  GROUP BY\n    Gene_Name,\n    Protein_Name )\nGROUP BY\n  Gene_Name\nHAVING\n  ( n > 1 )\nORDER BY\n  n DESC')


# Let's look further in the the EIF4EBP1 gene which has the most different proteins being measured:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nFROM\n  $rppa_BQtable\nWHERE\n  ( Gene_Name="EIF4EBP1" )\nGROUP BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nORDER BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus')


# Some antibodies are non-specific and bind to protein products from multiple genes in a gene family.  One example of this is the AKT1, AKT2, AKT3 gene family.  This non-specificity is indicated in the antibody-annotation file by a list of gene symbols, but in this table, we duplicate the entries (as well as the data values) on multiple rows:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nFROM\n  $rppa_BQtable\nWHERE\n  ( Gene_Name CONTAINS "AKT" )\nGROUP BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nORDER BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus')


get_ipython().run_cell_magic('sql', '', '\nSELECT\n  SampleBarcode,\n  Study,\n  Gene_Name,\n  Protein_Name,\n  Protein_Expression\nFROM\n  $rppa_BQtable\nWHERE\n  ( Protein_Name="Akt" )\nORDER BY\n  SampleBarcode,\n  Gene_Name\nLIMIT\n  9')





# # microRNA expression (BCGSC RPKM)
# 
# The goal of this notebook is to introduce you to the microRNA expression BigQuery table.
# 
# This table contains all available TCGA Level-3 microRNA expression data produced by BCGSC's microRNA pipeline using the Illumina HiSeq platform, as of July 2016.  The most recent archive (*eg* ``bcgsc.ca_THCA.IlluminaHiSeq_miRNASeq.Level_3.1.9.0``) for each of the 32 tumor types was downloaded from the DCC, and data extracted from all files matching the pattern ``%.isoform.quantification.txt``. The isoform-quantification values were then processed through a Perl script provided by BCGSC which produces normalized expression levels for *mature* microRNAs.  Each of these mature microRNAs is identified by name (*eg* hsa-mir-21) and by MIMAT accession number (*eg* MIMAT0000076).
# 
# In order to work with BigQuery, you need to import the python bigquery module (`gcp.bigquery`) and you need to know the name(s) of the table(s) you are going to be working with:
# 

import gcp.bigquery as bq
miRNA_BQtable = bq.Table('isb-cgc:tcga_201607_beta.miRNA_Expression')


# From now on, we will refer to this table using this variable ($miRNA_BQtable), but we could just as well explicitly give the table name each time.
# 
# Let's start by taking a look at the table schema:
# 

get_ipython().magic('bigquery schema --table $miRNA_BQtable')


# Now let's count up the number of unique patients, samples and aliquots mentioned in this table.  We will do this by defining a very simple parameterized query.  (Note that when using a variable for the table name in the FROM clause, you should not also use the square brackets that you usually would if you were specifying the table name as a string.)
# 

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')


fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = miRNA_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=miRNA_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


fieldList = ['mirna_id', 'mirna_accession']
for aField in fieldList:
  field = miRNA_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=miRNA_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)


# These counts show that the mirna_id field is not a unique identifier and should be used in combination with the MIMAT accession number.
# 

# Another thing to note about this table is that these expression values are obtained from two different platforms -- approximately 15% of the data is from the Illumina GA platform, and 85% from the Illumina HiSeq:
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Platform,\n  COUNT(*) AS n\nFROM\n  $miRNA_BQtable\nGROUP BY\n  Platform\nORDER BY\n  n DESC')


