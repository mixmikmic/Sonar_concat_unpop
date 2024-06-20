output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/20161126_Olivo_potential_deletions_MSP1_DBLMSP'
get_ipython().system('mkdir -p {output_dir}')


pf_6_bams_fn = '/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/pf_60_mergelanes.txt'
pf_5_bams_fn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'


get_ipython().system('grep PA0169 {pf_6_bams_fn}')


get_ipython().system('grep PA0169 {pf_5_bams_fn}')


get_ipython().system('grep PM0293 {pf_6_bams_fn}')


get_ipython().system('grep PM0293 {pf_5_bams_fn}')


get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')


get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')


get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000.bam ')


get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_10_v3:1403500-1420000 > ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_mem.bam')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_aln.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_aln.bam')


# # Map to 16 ref genomes

ftp_dir = 'ftp://ftp.sanger.ac.uk/pub/project/pathogens/Plasmodium/falciparum/PF3K/PilotReferenceGenomes/GenomeSequence/Version1/'
get_ipython().system('wget -r {ftp_dir} -P {output_dir}/')


output_dir


get_ipython().system('samtools bamshuf -uon 128 /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam tmp-prefix | samtools bam2fq -s se.fq.gz - | bwa mem -p ref.fa -')


# # Introduction
# This is details of a VR-PIPE setup of full calling pipeline of "reads" created by splitting PacBio reference genomes in 50kb chunks (don't know exact details - would need to get this from Thomas).

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


DATA_DIR


RELEASE_DIR = "%s/pacbio_1" % DATA_DIR
RESOURCES_DIR = '%s/resources' % RELEASE_DIR

# GENOME_FN = "/nfs/pathogen003/tdo/Pfalciparum/3D7/Reference/Oct2011/Pf3D7_v3.fasta" # Note this ref used by Thomas is different to other refs we have used, e.g. chromsomes aren't in numerical order
GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
SNPEFF_DIR = "/lustre/scratch109/malaria/pf3k_methods/resources/snpEff"
REGIONS_FN = "/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz"

RELEASE_METADATA_FN = "%s/pf3k_pacbio_1_sample_metadata.txt" % RELEASE_DIR
WG_VCF_FN = "%s/vcf/pf3k_pacbio_1.vcf.gz" % RELEASE_DIR

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
PICARD = 'java -jar /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/picard/picard-tools-1.137/picard.jar'

VRPIPE_FOFN = "%s/pf3k_pacbio_1.fofn" % RELEASE_DIR
HC_INPUT_FOFN = "%s/pf3k_pacbio_1_hc_input.fofn" % RELEASE_DIR


print(WG_VCF_FN)


chromosomes = ["Pf3D7_%02d_v3" % x for x in range(1, 15, 1)]
#     'Pf3D7_API_v3', 'Pf_M76611'
# ]
chromosome_vcfs = ["%s/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz" % (RELEASE_DIR, x) for x in chromosomes]
chromosome_vcfs


if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


get_ipython().system('cp {GENOME_FN}* {RESOURCES_DIR}')
get_ipython().system('cp -R {SNPEFF_DIR} {RESOURCES_DIR}')
get_ipython().system('cp -R {REGIONS_FN} {RESOURCES_DIR}')


for lustre_dir in ['temp', 'input', 'output', 'meta']:
    new_dir = "/lustre/scratch109/malaria/pf3k_pacbio/%s" % lustre_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# # Modify Thomas's bams to make them compatible
# Issues:
# 1) Multiple bams per sample (one per chromosome)
# 2) Different bams for sample have different read groups and sample names, so will be considered different sample
# 
# Approach is to create new bams replacing all read groups with something appropriate, then merging chroms to WG

test1 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_01.01.bam'
test2 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_02.02.bam'
test3 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_03.03.bam'
test_dir = '/lustre/scratch111/malaria/rp7/temp'
get_ipython().system('mkdir -p {test_dir}')


get_ipython().system('samtools view -H {test1}')


get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test1} O={test_dir}/Pf7G8_01.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')
get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test2} O={test_dir}/Pf7G8_02.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')
get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test3} O={test_dir}/Pf7G8_03.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')


get_ipython().system('{PICARD} MergeSamFiles I={test_dir}/Pf7G8_01.bam I={test_dir}/Pf7G8_02.bam I={test_dir}/Pf7G8_03.bam O={test_dir}/temp.bam MERGE_SEQUENCE_DICTIONARIES=true')


17*14


thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
                         'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSD01', 'PfSN01', 'PfTG01']


for sample_id in thomas_bam_sample_ids:
    for chrom in range(1, 15):
        sample_chrom = '%s_%02d' % (sample_id, chrom)
        print(sample_chrom)
        input_bam_fn = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.%s_%02d.%02d.bam' % (
            sample_id, chrom, chrom
        )
        get_ipython().system('{PICARD} AddOrReplaceReadGroups I={input_bam_fn} O=/lustre/scratch109/malaria/pf3k_pacbio/temp/{sample_chrom}.bam RGID={sample_id} RGSM={sample_id} RGLB={sample_id} RGPU={sample_id} RGPL=illumina')


for sample_id in thomas_bam_sample_ids:
    input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in range(1, 15)]
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
    get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')


sample_id = 'PfSD01'
input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in
              [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')


fo = open(VRPIPE_FOFN, 'w')
print('path\tsample', file=fo)
fo.close()
fo = open(VRPIPE_FOFN, 'a')
for sample_id in thomas_bam_sample_ids:
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    print('%s\t%s' % (output_bam, sample_id), file=fo)
fo.close()


# Removed PfSD01 as no reads for chromsome 4 - see email from Thomas 14/07/2016 20:17 and response 17/07/2016 11:37
thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
                         'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSN01', 'PfTG01']


fo = open(HC_INPUT_FOFN, 'w')
for sample_id in thomas_bam_sample_ids:
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    print('%s' % (output_bam), file=fo)
fo.close()


VRPIPE_FOFN


HC_INPUT_FOFN


get_ipython().system("grep -P 'Pf3D7_[0|1]' /lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta.fai > /lustre/scratch109/malaria/pf3k_pacbio/Pfalciparum.autosomes.fasta.fai")


# # VR-PIPE pipeline
# These were created using vrpipe-setup --based_on, with the setups originally created in 20160706_pf3k_mrs_1_setup

# # Create whole genome VCF

get_ipython().system('{BCFTOOLS} concat {" ".join(chromosome_vcfs)} | sed \'s/##FORMAT=<ID=AD,Number=./##FORMAT=<ID=AD,Number=R/\' | bgzip -c > {WG_VCF_FN}')
get_ipython().system('tabix -p vcf {WG_VCF_FN}')


WG_VCF_FN


# # Create README section containing variant numbers

number_of_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')

print("%s variants" % ("{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0]))))
print("%s SNPs" % ("{:,}".format(int(number_of_snps[0]))))
print("%s indels" % ("{:,}".format(int(number_of_indels[0]))))
print()
print("%s PASS variants" % ("{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0]))))
print("%s PASS SNPs" % ("{:,}".format(int(number_of_pass_snps[0]))))
print("%s PASS indels" % ("{:,}".format(int(number_of_pass_indels[0]))))
print()
print("%s PASS biallelic variants" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]) + int(number_of_pass_biallelic_indels[0]))))
print("%s PASS biallelic SNPs" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]))))
print("%s PASS biallelic indels" % ("{:,}".format(int(number_of_pass_biallelic_indels[0]))))
print()
print("%s VQSLOD>6.0 variants" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]) + int(number_of_VQSLODgt6_indels[0]))))
print("%s VQSLOD>6.0 SNPs" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]))))
print("%s VQSLOD>6.0 indels" % ("{:,}".format(int(number_of_VQSLODgt6_indels[0]))))
print()


"{number_of_pass_variants}/{number_of_variants} variants ({pct_pass}%) pass all filters".format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        pct_pass=round((
            (int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])) /
            (int(number_of_snps[0]) + int(number_of_indels[0]))
        ) * 100)
)


print('''
The VCF file contains details of {number_of_variants} discovered variants of which {number_of_snps}
are SNPs and {number_of_indels} are indels (or multi-allelic mixtures of SNPs
and indels). It is important to note that many of these variants are
considered low quality. Only the variants for which the FILTER column is set
to PASS should be considered of high quality. There are {number_of_pass_variants} such high-
quality PASS variants ({number_of_pass_snps} SNPs and {number_of_pass_indels} indels).

The FILTER column is based on two types of information. Firstly certain regions
of the genome are considered "non-core". This includes sub-telomeric regions,
centromeres and internal VAR gene clusters on chromosomes 4, 6, 7, 8 and 12.
The apicoplast and mitochondrion are also considered non-core. All variants within
non-core regions are considered to be low quality, and hence will not have the
FILTER column set to PASS. The regions which are core and non-core can be found
in the file resources/regions-20130225.bed.gz.

Secondly, variants are filtered out based on a quality score called VQSLOD. All
variants with a VQSLOD score below 0 are filtered out, i.e. will have a value of
Low_VQSLOD in the FILTER column, rather than PASS. The VQSLOD score for each
variant can be found in the INFO field of the VCF file. It is possible to use the
VQSLOD score to define a more or less stringent set of variants. For example for
a very stringent set of the highest quality variants, select only those variants
where VQSLOD >= 6. There are {number_of_VQSLODgt6_snps} such stringent SNPs and {number_of_VQSLODgt6_indels}
such stringent indels.

It is also important to note that some variants have more than two alleles. For
example, amongst the {number_of_pass_snps} high quality PASS SNPs, {number_of_pass_biallelic_snps} are biallelic. The
remaining {number_of_pass_multiallelic_snps} high quality PASS SNPs have 3 or more alleles. Similarly, amongst
the {number_of_pass_indels} high-quality PASS indels, {number_of_pass_biallelic_indels} are biallelic. The remaining
{number_of_pass_multiallelic_indels} high quality PASS indels have 3 or more alleles.
'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_snps="{:,}".format(int(number_of_snps[0])),
        number_of_indels="{:,}".format(int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        number_of_pass_snps="{:,}".format(int(number_of_pass_snps[0])),
        number_of_pass_indels="{:,}".format(int(number_of_pass_indels[0])),
        number_of_VQSLODgt6_snps="{:,}".format(int(number_of_VQSLODgt6_snps[0])),
        number_of_VQSLODgt6_indels="{:,}".format(int(number_of_VQSLODgt6_indels[0])),
        number_of_pass_biallelic_snps="{:,}".format(int(number_of_pass_biallelic_snps[0])),
        number_of_pass_biallelic_indels="{:,}".format(int(number_of_pass_biallelic_indels[0])),
        number_of_pass_multiallelic_snps="{:,}".format(int(number_of_pass_snps[0]) - int(number_of_pass_biallelic_snps[0])),
        number_of_pass_multiallelic_indels="{:,}".format(int(number_of_pass_indels[0]) - int(number_of_pass_biallelic_indels[0])),
    )
)


# # Make all files read-only

# Seems like Jim prefers team112 to have write access, so haven't made unreadable
# !chmod -R uga-w {RELEASE_DIR}


RELEASE_DIR


# # Cleanup
# vrpipe-setup --setup pf3k_pacbio_1_haplotype_caller --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_combine_gvcfs --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_genotype_gvcfs --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_variant_recalibration_snps_QD_SOR_DP --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_variant_recalibration_indels_QD_DP --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_apply_recalibration_snps --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_apply_recalibration_indels --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_annotate_snps --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_annotate_indels --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_combine_variants --deactivate
# vrpipe-setup --setup pf3k_pacbio_1_variant_filtration --deactivate
# 
# 

# # Introduction
# This is details of a VR-PIPE setup of full calling pipeline of "reads" created by splitting PacBio reference genomes in 50kb chunks (don't know exact details - would need to get this from Thomas).

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


DATA_DIR


RELEASE_DIR = "%s/pacbio_2" % DATA_DIR
RESOURCES_DIR = '%s/resources' % RELEASE_DIR

# GENOME_FN = "/nfs/pathogen003/tdo/Pfalciparum/3D7/Reference/Oct2011/Pf3D7_v3.fasta" # Note this ref used by Thomas is different to other refs we have used, e.g. chromsomes aren't in numerical order
GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
SNPEFF_DIR = "/lustre/scratch109/malaria/pf3k_methods/resources/snpEff"
REGIONS_FN = "/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz"

RELEASE_METADATA_FN = "%s/pf3k_pacbio_2_sample_metadata.txt" % RELEASE_DIR
WG_VCF_FN = "%s/vcf/pf3k_pacbio_2.vcf.gz" % RELEASE_DIR

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
PICARD = 'java -jar /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/picard/picard-tools-1.137/picard.jar'

VRPIPE_FOFN = "%s/pf3k_pacbio_2.fofn" % RELEASE_DIR
HC_INPUT_FOFN = "%s/pf3k_pacbio_2_hc_input.fofn" % RELEASE_DIR


print(WG_VCF_FN)


chromosomes = ["Pf3D7_%02d_v3" % x for x in range(1, 15, 1)]
#     'Pf3D7_API_v3', 'Pf_M76611'
# ]
chromosome_vcfs = ["%s/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz" % (RELEASE_DIR, x) for x in chromosomes]
chromosome_vcfs


if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


for lustre_dir in ['temp', 'input', 'output', 'meta']:
    new_dir = "/lustre/scratch109/malaria/pf3k_pacbio/%s" % lustre_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
                         'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSD01', 'PfSN01', 'PfTG01']


sample_chrom = 'PfSD01_04_07'
sample_id = 'PfSD01'
get_ipython().system('{PICARD} AddOrReplaceReadGroups I=/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.PfSD01_04_07.fasta.Merged.0407.bam O=/lustre/scratch109/malaria/pf3k_pacbio/temp/{sample_chrom}.bam RGID={sample_id} RGSM={sample_id} RGLB={sample_id} RGPU={sample_id} RGPL=illumina')


sample_id = 'PfSD01'
sample_chrom = 'PfSD01_04_07'
input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s.bam' % sample_chrom] + ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in
              [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]]
output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')


get_ipython().system("{PICARD} ReplaceSamHeader I={output_bam} O={output_bam.replace('.bam', 'reheader.bam')} HEADER=/lustre/scratch109/malaria/pf3k_pacbio/input/PfSD01.header.sam")


input_bams


sample_id = 'PfSD01'
output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
get_ipython().system("java -jar /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt/picard/picard-tools-1.135/picard.jar ReorderSam I={output_bam.replace('.bam', 'reheader.bam')} O={output_bam.replace('.bam', '.sorted.bam')} REFERENCE=/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta")
# CREATE_INDEX=TRUE
get_ipython().system("mv {output_bam.replace('.bam', '.sorted.bam')} {output_bam}")
get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')


output_bam


# I messed up the folowing by accident so had to recreate
for sample_id in ['PfTG01']:
    input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in range(1, 15)]
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
    get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')


# # Removed PfSD01 as no reads for chromsome 4 - see email from Thomas 14/07/2016 20:17 and response 17/07/2016 11:37
# thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
#                          'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSN01', 'PfTG01']


fo = open(HC_INPUT_FOFN, 'w')
for sample_id in thomas_bam_sample_ids:
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    print('%s' % (output_bam), file=fo)
fo.close()


HC_INPUT_FOFN


# # VR-PIPE pipeline
# These were created using vrpipe-setup --based_on, with the setups originally created in 20160716_pf3k_pacbio_1_setup

# # Create whole genome VCF

get_ipython().system('{BCFTOOLS} concat {" ".join(chromosome_vcfs)} | sed \'s/##FORMAT=<ID=AD,Number=./##FORMAT=<ID=AD,Number=R/\' | bgzip -c > {WG_VCF_FN}')
get_ipython().system('tabix -p vcf {WG_VCF_FN}')


WG_VCF_FN


# # Create README section containing variant numbers

number_of_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')

print("%s variants" % ("{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0]))))
print("%s SNPs" % ("{:,}".format(int(number_of_snps[0]))))
print("%s indels" % ("{:,}".format(int(number_of_indels[0]))))
print()
print("%s PASS variants" % ("{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0]))))
print("%s PASS SNPs" % ("{:,}".format(int(number_of_pass_snps[0]))))
print("%s PASS indels" % ("{:,}".format(int(number_of_pass_indels[0]))))
print()
print("%s PASS biallelic variants" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]) + int(number_of_pass_biallelic_indels[0]))))
print("%s PASS biallelic SNPs" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]))))
print("%s PASS biallelic indels" % ("{:,}".format(int(number_of_pass_biallelic_indels[0]))))
print()
print("%s VQSLOD>6.0 variants" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]) + int(number_of_VQSLODgt6_indels[0]))))
print("%s VQSLOD>6.0 SNPs" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]))))
print("%s VQSLOD>6.0 indels" % ("{:,}".format(int(number_of_VQSLODgt6_indels[0]))))
print()


"{number_of_pass_variants}/{number_of_variants} variants ({pct_pass}%) pass all filters".format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        pct_pass=round((
            (int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])) /
            (int(number_of_snps[0]) + int(number_of_indels[0]))
        ) * 100)
)


"{number_of_pass_variants}/{number_of_variants} variants ({pct_pass}%) pass all filters".format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        pct_pass=round((
            (int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])) /
            (int(number_of_snps[0]) + int(number_of_indels[0]))
        ) * 100)
)


print('''
The VCF file contains details of {number_of_variants} discovered variants of which {number_of_snps}
are SNPs and {number_of_indels} are indels (or multi-allelic mixtures of SNPs
and indels). It is important to note that many of these variants are
considered low quality. Only the variants for which the FILTER column is set
to PASS should be considered of high quality. There are {number_of_pass_variants} such high-
quality PASS variants ({number_of_pass_snps} SNPs and {number_of_pass_indels} indels).

The FILTER column is based on two types of information. Firstly certain regions
of the genome are considered "non-core". This includes sub-telomeric regions,
centromeres and internal VAR gene clusters on chromosomes 4, 6, 7, 8 and 12.
The apicoplast and mitochondrion are also considered non-core. All variants within
non-core regions are considered to be low quality, and hence will not have the
FILTER column set to PASS. The regions which are core and non-core can be found
in the file resources/regions-20130225.bed.gz.

Secondly, variants are filtered out based on a quality score called VQSLOD. All
variants with a VQSLOD score below 0 are filtered out, i.e. will have a value of
Low_VQSLOD in the FILTER column, rather than PASS. The VQSLOD score for each
variant can be found in the INFO field of the VCF file. It is possible to use the
VQSLOD score to define a more or less stringent set of variants. For example for
a very stringent set of the highest quality variants, select only those variants
where VQSLOD >= 6. There are {number_of_VQSLODgt6_snps} such stringent SNPs and {number_of_VQSLODgt6_indels}
such stringent indels.

It is also important to note that some variants have more than two alleles. For
example, amongst the {number_of_pass_snps} high quality PASS SNPs, {number_of_pass_biallelic_snps} are biallelic. The
remaining {number_of_pass_multiallelic_snps} high quality PASS SNPs have 3 or more alleles. Similarly, amongst
the {number_of_pass_indels} high-quality PASS indels, {number_of_pass_biallelic_indels} are biallelic. The remaining
{number_of_pass_multiallelic_indels} high quality PASS indels have 3 or more alleles.
'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_snps="{:,}".format(int(number_of_snps[0])),
        number_of_indels="{:,}".format(int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        number_of_pass_snps="{:,}".format(int(number_of_pass_snps[0])),
        number_of_pass_indels="{:,}".format(int(number_of_pass_indels[0])),
        number_of_VQSLODgt6_snps="{:,}".format(int(number_of_VQSLODgt6_snps[0])),
        number_of_VQSLODgt6_indels="{:,}".format(int(number_of_VQSLODgt6_indels[0])),
        number_of_pass_biallelic_snps="{:,}".format(int(number_of_pass_biallelic_snps[0])),
        number_of_pass_biallelic_indels="{:,}".format(int(number_of_pass_biallelic_indels[0])),
        number_of_pass_multiallelic_snps="{:,}".format(int(number_of_pass_snps[0]) - int(number_of_pass_biallelic_snps[0])),
        number_of_pass_multiallelic_indels="{:,}".format(int(number_of_pass_indels[0]) - int(number_of_pass_biallelic_indels[0])),
    )
)


print('''
The VCF file contains details of {number_of_variants} discovered variants of which {number_of_snps}
are SNPs and {number_of_indels} are indels (or multi-allelic mixtures of SNPs
and indels). It is important to note that many of these variants are
considered low quality. Only the variants for which the FILTER column is set
to PASS should be considered of high quality. There are {number_of_pass_variants} such high-
quality PASS variants ({number_of_pass_snps} SNPs and {number_of_pass_indels} indels).

The FILTER column is based on two types of information. Firstly certain regions
of the genome are considered "non-core". This includes sub-telomeric regions,
centromeres and internal VAR gene clusters on chromosomes 4, 6, 7, 8 and 12.
The apicoplast and mitochondrion are also considered non-core. All variants within
non-core regions are considered to be low quality, and hence will not have the
FILTER column set to PASS. The regions which are core and non-core can be found
in the file resources/regions-20130225.bed.gz.

Secondly, variants are filtered out based on a quality score called VQSLOD. All
variants with a VQSLOD score below 0 are filtered out, i.e. will have a value of
Low_VQSLOD in the FILTER column, rather than PASS. The VQSLOD score for each
variant can be found in the INFO field of the VCF file. It is possible to use the
VQSLOD score to define a more or less stringent set of variants. For example for
a very stringent set of the highest quality variants, select only those variants
where VQSLOD >= 6. There are {number_of_VQSLODgt6_snps} such stringent SNPs and {number_of_VQSLODgt6_indels}
such stringent indels.

It is also important to note that some variants have more than two alleles. For
example, amongst the {number_of_pass_snps} high quality PASS SNPs, {number_of_pass_biallelic_snps} are biallelic. The
remaining {number_of_pass_multiallelic_snps} high quality PASS SNPs have 3 or more alleles. Similarly, amongst
the {number_of_pass_indels} high-quality PASS indels, {number_of_pass_biallelic_indels} are biallelic. The remaining
{number_of_pass_multiallelic_indels} high quality PASS indels have 3 or more alleles.
'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_snps="{:,}".format(int(number_of_snps[0])),
        number_of_indels="{:,}".format(int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        number_of_pass_snps="{:,}".format(int(number_of_pass_snps[0])),
        number_of_pass_indels="{:,}".format(int(number_of_pass_indels[0])),
        number_of_VQSLODgt6_snps="{:,}".format(int(number_of_VQSLODgt6_snps[0])),
        number_of_VQSLODgt6_indels="{:,}".format(int(number_of_VQSLODgt6_indels[0])),
        number_of_pass_biallelic_snps="{:,}".format(int(number_of_pass_biallelic_snps[0])),
        number_of_pass_biallelic_indels="{:,}".format(int(number_of_pass_biallelic_indels[0])),
        number_of_pass_multiallelic_snps="{:,}".format(int(number_of_pass_snps[0]) - int(number_of_pass_biallelic_snps[0])),
        number_of_pass_multiallelic_indels="{:,}".format(int(number_of_pass_indels[0]) - int(number_of_pass_biallelic_indels[0])),
    )
)


# # Make all files read-only

# Seems like Jim prefers team112 to have write access, so haven't made unreadable
# !chmod -R uga-w {RELEASE_DIR}


RELEASE_DIR


# # Cleanup
# vrpipe-setup --setup pf3k_pacbio_2_haplotype_caller --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_combine_gvcfs --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_genotype_gvcfs --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_variant_recalibration_snps_QD_SOR_DP --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_variant_recalibration_indels_QD_DP --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_apply_recalibration_snps --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_apply_recalibration_indels --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_annotate_snps --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_annotate_indels --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_combine_variants --deactivate
# 
# vrpipe-setup --setup pf3k_pacbio_2_variant_filtration --deactivate
# 
# 




# This notebook must be run directly from MacBook after running ~/bin/sanger-tunneling.sh in order to connect
# to Sanger network. I haven't figured out a way to do this from Docker container

get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


get_ipython().system('mkdir -p {os.path.dirname(INTERIM5_VCF_FOFN)}')
get_ipython().system('rsync -avL malsrv2:{INTERIM5_VCF_FOFN} {os.path.dirname(INTERIM5_VCF_FOFN)}')


for release in CHROM_VCF_FNS.keys():
    for chrom in CHROM_VCF_FNS[release].keys():
        vcf_fn = CHROM_VCF_FNS[release][chrom]
        get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
        get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')


vcf_fn = WG_VCF_FNS['release3']
get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')


if not os.path.exists(RELEASE4_RESOURCES_DIR):
    get_ipython().system('mkdir -p {RELEASE4_RESOURCES_DIR}')
get_ipython().system('rsync -avL malsrv2:{RELEASE4_RESOURCES_DIR} {os.path.dirname(RELEASE4_RESOURCES_DIR)}')


# GATK executables
get_ipython().system('mkdir -p /nfs/team112_internal/production/tools/bin')
get_ipython().system('rsync -avL malsrv2:/nfs/team112_internal/production/tools/bin/gatk /nfs/team112_internal/production/tools/bin/')


# Other executables - decided to leave these for now
# !rsync -avL malsrv2:/nfs/team112_internal/production/tools/bin /nfs/team112_internal/production/tools/


# #Plan
# - Create function to create biallelic, 5/2 rule, new AF, segregating, minimal, renormalised VCF
# - Split the above into SNPs and INDELs
# - Test function on small subset of chr14
# - Run function on chrom 14
# - New function to also create npy file
# - Read in chr14 npy file, and calculate Mendelian error and genotype concordance
# - Attempt to reannotate above with STR and SNPEFF annotations
# - Rerun scripts to get breakdown by SNP/STR/nonSTR coding/noncoding

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
chrom_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160527_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')
chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3.analysis.vcf.gz" % output_dir

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'


chrom_vcf_fn


tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


all_samples = ','.join(tbl_release5_crosses_metadata.values('sample'))


all_samples


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 1).values('sample').list()


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 0).values('sample').list()


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


np.in1d(replicates_first, tbl_release5_crosses_metadata.values('sample').array())


rep_index_first = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_first)
rep_index_second = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_second)
print(np.sum(rep_index_first))
print(np.sum(rep_index_second))


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.duplicates('clone').displayall()


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


(tbl_release5_crosses_metadata
 .selecteq('study_title', '3D7xHB3 cross progeny')
 .selecteq('parent_or_progeny', 'parent')
 .values('sample')
 .array()
)


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'MLEAF': 1,
            'MLEAC': 1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        verbose=False,
        cache=True,
        cachedir=output_dir
    )


def create_analysis_vcf(input_vcf_fn=chrom_vcf_fn, output_vcf_fn=chrom_analysis_vcf_fn,
                        region='Pf3D7_14_v3:1000000-2000000', BCFTOOLS=BCFTOOLS, rewrite=False):
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['biallelic', 'regenotyped', 'new_af', 'nonref', 'pass', 'minimal', 'analysis', 'SNP', 'INDEL',
                             'SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

#     if not os.path.exists(subset_vcf_fn):
#         !{BCFTOOLS} view -Oz -o {subset_vcf_fn} -s {validation_samples} {chrom_vcf_fn}
#         !{BCFTOOLS} index --tbi {subset_vcf_fn}

    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        if region is not None:
            get_ipython().system("{BCFTOOLS} annotate --regions {region} --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        else:
            get_ipython().system("{BCFTOOLS} annotate --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['regenotyped']):
        get_ipython().system("/nfs/team112/software/htslib/vfp/vfp_tool {intermediate_fns['biallelic']} /nfs/team112/software/htslib/vfp/just_call.config |         bgzip -c > {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['regenotyped']}")

    if rewrite or not os.path.exists(intermediate_fns['new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['new_af']} {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']} {intermediate_fns[\'new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['pass']} {intermediate_fns['nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['pass']}")

    if rewrite or not os.path.exists(intermediate_fns['minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['minimal']} {intermediate_fns['pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['analysis']} {intermediate_fns['minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'SNP\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'SNP_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MIXED\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'INDEL\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_MULTIALLELIC']}")
        
    for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
        if rewrite or not os.path.exists("%s/.vcfnp_cache/variants.npy" % intermediate_fns[variant_type]):
            create_variants_npy(intermediate_fns[variant_type])
        if rewrite or not os.path.exists("%s/.vcfnp_cache/calldata_2d.npy" % intermediate_fns[variant_type]):
            create_calldata_npy(intermediate_fns[variant_type])
        


create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)





variants_SNP_BIALLELIC = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))


calldata_SNP_BIALLELIC['GT'].shape[1]


calldata_SNP_BIALLELIC['GT'][:, 1] == b'0'


np.unique(variants_SNP_BIALLELIC['SNPEFF_EFFECT'], return_counts=True)


np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)


4788/(np.sum([  1533, 196926,   4788,  95261]))


hets_per_sample = np.sum(calldata_SNP_BIALLELIC['GT'] == b'0/1', 0)
print(len(hets_per_sample))


hets_per_sample


def genotype_concordance(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0') & (parent2_calls == b'0') & (progeny_calls == b'1')) |
                ((parent1_calls == b'1') & (parent2_calls == b'1') & (progeny_calls == b'0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0') | (parent1_calls == b'1' )) &
                ((parent2_calls == b'0') | (parent2_calls == b'1' )) &
                ((progeny_calls == b'0') | (progeny_calls == b'1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0', b'1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0', b'1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0') | (calldata[:, rep_index_first] == b'1')) &
        ((calldata[:, rep_index_second] == b'0') | (calldata[:, rep_index_second] == b'1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0') & (calldata[:, rep_index_second] == b'1')) |
        ((calldata[:, rep_index_first] == b'1') & (calldata[:, rep_index_second] == b'0'))
    )
    missingness_per_sample = np.sum(calldata == b'.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    


for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
    variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
    calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
#     print(variant_type)
    mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
    print(variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
    mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
    print(variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


genotype_concordance()


calldata_SNP_BIALLELIC[:, rep_index_first]


variants_crosses = np.load('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz.vcfnp_cache/variants.npy')


variants_crosses.dtype.names


np.unique(variants_crosses['VariantType'])


del(variants_crosses)
gc.collect()


2+2





# # Introduction
# 
# The aim of this notebook is to determine a sensible threshold for creating a DUST mask. This came about after conversations with Roberto about how to remove the many variants in AT repeats (or near AT repeats) for the PPQ GWAS. In addition, I also evaluate windowmasker and tantan, but find them to be less use (especially windowmasker which is removing a large proportion of the genome)
# 
# The earlier version of the notbeook (20151027_dustmasker) used petl intervals to find overlaps between regions, but I then realised using numpy boolean arrays would be much more efficient (important when trying out lots of different thresholds)

get_ipython().run_line_magic('run', '_shared_setup.ipynb')


install_dir = '../opt_4'
REF_GENOME="/lustre/scratch110/malaria/rp7/Pf3k/GATKbuild/Pfalciparum_GeneDB_Aug2015/Pfalciparum.genome.fasta"
regions_fn = '/nfs/users/nfs_r/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'
# regions_fn = '/Users/rpearson/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'
ref_gff = "%s/snpeff/snpEff/data/Pfalciparum_GeneDB_Aug2015/genes.gff" % install_dir
ref_cds_gff = REF_GENOME.replace('.fasta', '.CDS.gff')


get_ipython().system("head -n -34 {ref_gff} | grep -P '\\tCDS\\t' > {ref_cds_gff}")


# # Download software

# !wget ftp://ftp.ncbi.nlm.nih.gov/pub/agarwala/dustmasker/dustmasker -O {install_dir}/dustmasker
# !chmod a+x {install_dir}/dustmasker


get_ipython().system('wget ftp://ftp.ncbi.nlm.nih.gov/pub/agarwala/windowmasker/windowmasker -O {install_dir}/windowmasker')
get_ipython().system('chmod a+x {install_dir}/windowmasker')


current_dir = get_ipython().getoutput('pwd')
current_dir = current_dir[0]
get_ipython().system('wget http://cbrc3.cbrc.jp/~martin/tantan/tantan-13.zip -O {install_dir}/tantan-13.zip')
get_ipython().run_line_magic('cd', '{install_dir}')
get_ipython().system('unzip tantan-13.zip')
get_ipython().run_line_magic('cd', 'tantan-13')
get_ipython().system('make')
get_ipython().run_line_magic('cd', '{current_dir}')


# # Run algorithms on ref genome

ref_dict=SeqIO.to_dict(SeqIO.parse(open(REF_GENOME), "fasta"))
chromosome_lengths = [len(ref_dict[chrom]) for chrom in ref_dict]
tbl_chromosomes=(etl.wrap(zip(ref_dict.keys(), chromosome_lengths))
    .pushheader(['chrom', 'stop'])
    .addfield('start', 0)
    .cut(['chrom', 'start', 'stop'])
    .sort('chrom')
)
tbl_chromosomes


tbl_regions = (etl
    .fromtsv(regions_fn)
    .pushheader(['chrom', 'start', 'stop', 'region'])
    .convertnumbers()
)
tbl_regions.display(10)


iscore_array = collections.OrderedDict()
for chromosomes_row in tbl_chromosomes.data():
    chrom=chromosomes_row[0]
    iscore_array[chrom] = np.zeros(chromosomes_row[2], dtype=bool)
    for regions_row in tbl_regions.selecteq('chrom', chrom).selecteq('region', 'Core').data():
        iscore_array[chrom][regions_row[1]:regions_row[2]] = True


tbl_ref_cds_gff = (
    etl.fromgff3(ref_cds_gff)
    .select(lambda rec: rec['end'] > rec['start'])
    .unpackdict('attributes')
    .select(lambda rec: rec['Parent'].endswith('1')) # Think there are alternate splicings for some genes, here just using first
    .distinct(['seqid', 'start'])
)


tbl_coding_regions = (tbl_ref_cds_gff
    .cut(['seqid', 'start', 'end'])
    .rename('end', 'stop')
    .rename('seqid', 'chrom')
    .convert('start', lambda val: val-1)
)
tbl_coding_regions                   


iscoding_array = collections.OrderedDict()
for chromosomes_row in tbl_chromosomes.data():
    chrom=chromosomes_row[0]
    iscoding_array[chrom] = np.zeros(chromosomes_row[2], dtype=bool)
    for coding_regions_row in tbl_coding_regions.selecteq('chrom', chrom).data():
        iscoding_array[chrom][coding_regions_row[1]:coding_regions_row[2]] = True


def which_lower(string):
    return np.array([str.islower(x) for x in string])
which_lower('abCDeF') 
# np.array([str.islower(x) for x in 'abCDeF'])


def find_regions(masked_pos, number_of_regions=3):
    masked_regions = list()
    start = masked_pos[0]
    stop = start
    region_number = 1
    for pos in masked_pos:
        if pos > (stop + 1):
            masked_regions.append([start, stop])
            start = stop = pos
            region_number = region_number + 1
            if region_number > number_of_regions:
                break
        else:
            stop = pos
    return(masked_regions)


def summarise_masking(
    classification_array,
    masking_description = "Dust level 20",
    number_of_regions = 10,
    max_sequence_length = 60
):
    number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
    number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
    number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
    number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
    proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
    proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
    print("%s: %4.1f%% coding and %4.1f%% non-coding masked" % (
            masking_description,
            proportion_core_coding_masked*100,
            proportion_core_noncoding_masked*100
        )
    )
    coding_masked_pos = np.where(classification_array['Core coding masked'])[0]
    noncoding_masked_pos = np.where(classification_array['Core noncoding masked'])[0]
    coding_masked_regions = find_regions(coding_masked_pos, number_of_regions)
    noncoding_masked_regions = find_regions(noncoding_masked_pos, number_of_regions)    
    
    print("    First %d Pf3D7_01_v3 coding sequences masked:" % number_of_regions)
    for region in coding_masked_regions:
        if region[1] - region[0] > max_sequence_length:
            masked_sequence = "%s[...]" % ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]][0:max_sequence_length]
        else:
            masked_sequence = ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]]
        print("        %d - %d: %s" % (
                region[0],
                region[1],
                masked_sequence
            )
        )
    print("    First %d Pf3D7_01_v3 non-coding sequences masked:" % number_of_regions)
    for region in noncoding_masked_regions:
        if region[1] - region[0] > max_sequence_length:
            masked_sequence = "%s[...]" % ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]][0:max_sequence_length]
        else:
            masked_sequence = ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]]
        print("        %d - %d: %s" % (
                region[0],
                region[1],
                masked_sequence
            )
        )
    print()
    


def evaluate_dust_threshold(
    dust_level=20,
    verbose=False
):
    masked_genome_fn = "%s.dustmasker.%d.fasta" % (REF_GENOME.replace('.fasta', ''), dust_level)
    
    if verbose:
        print("Running dustmasker %d" % dust_level)
    get_ipython().system('{install_dir}/dustmasker     -in {REF_GENOME}     -outfmt fasta     -out {masked_genome_fn}     -level {dust_level}')

    if verbose:
        print("Reading in fasta %d" % dust_level)
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array %d" % dust_level)
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        masked_ref_dict_chrom = "lcl|%s" % chrom
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[masked_ref_dict_chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "Dust level %d" % dust_level)
                      
#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("dustmasker dust_level=%d: %4.1f%% coding and %4.1f%% non-coding masked" % (
#             dust_level,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
# #             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#        )
#     )
#     non_coding_masked_pos = np.where(classification_array['Core coding masked'])[0]
#     for masked_coding_region in find_regions(non_coding_masked_pos):
#         print("\t%d - %d: %s" % (
#                 masked_coding_region[0],
#                 masked_coding_region[1],
#                 ref_dict['Pf3D7_01_v3'].seq[masked_coding_region[0]:masked_coding_region[1]]
#             )
#         )

    return(classification_array, masked_ref_dict, ismasked_array)


def evaluate_windowmasker(
    check_dup='true',
    use_dust='false',
    verbose=False
):
    ustat_fn = "%s.windowmasker.%s.%s.ustat" % (REF_GENOME.replace('.fasta', ''), check_dup, use_dust)
    masked_genome_fn = "%s.windowmasker.%s.%s.fasta" % (REF_GENOME.replace('.fasta', ''), check_dup, use_dust)
    
    if verbose:
        print("Running dustmasker check_dup=%s use_dust=%s" % (check_dup, use_dust))
    get_ipython().system('{install_dir}/windowmasker -mk_counts     -in {REF_GENOME}     -checkdup {check_dup}     -out {ustat_fn}')

    get_ipython().system('{install_dir}/windowmasker     -ustat {ustat_fn}     -in {REF_GENOME}     -outfmt fasta     -out {masked_genome_fn}     -dust {use_dust} ')
    if verbose:
        print("Reading in fasta check_dup=%s use_dust=%s" % (check_dup, use_dust))
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array check_dup=%s use_dust=%s" % (check_dup, use_dust))
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "windowmasker check_dup=%s use_dust=%s" % (check_dup, use_dust))

#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("windowmasker check_dup=%s use_dust=%s: %4.1f%% coding and %4.1f%% non-coding masked\n\t%s" % (
#             check_dup,
#             use_dust,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
#             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#         )
#     )
        
    return(classification_array, masked_ref_dict, ismasked_array)


def evaluate_tantan(
    r=0.005,
    m=None,
    verbose=False
):
    masked_genome_fn = "%s.tantan.%s.%s.fasta" % (REF_GENOME.replace('.fasta', ''), r, m)
    
    if verbose:
        print("Running tantan r=%s m=%s" % (r, m))
    if m is None:
        get_ipython().system('{install_dir}/tantan-13/src/tantan -r {r} {REF_GENOME} > {masked_genome_fn}')
    elif m == 'atMask.mat':
        get_ipython().system('{install_dir}/tantan-13/src/tantan -r {r} -m {install_dir}/tantan-13/test/atMask.mat {REF_GENOME} >             {masked_genome_fn}')
    else:
        stop("Unknown option m=%s" % m)

    if verbose:
        print("Reading in fasta r=%s m=%s" % (r, m))
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array r=%s m=%s" % (r, m))
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "tantan r=%s m=%s" % (r, m))

#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("tantan r=%s m=%s: %4.1f%% coding and %4.1f%% non-coding masked\n\t%s" % (
#             r,
#             m,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
#             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#         )
#     )
        
    return(classification_array, masked_ref_dict, ismasked_array)


dustmasker_classification_arrays = collections.OrderedDict()
for dust_level in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    dustmasker_classification_arrays[str(dust_level)] = evaluate_dust_threshold(dust_level, verbose=False)


windowmasker_classification_arrays = collections.OrderedDict()
for check_dup in ['true', 'false']:
    windowmasker_classification_arrays[check_dup] = collections.OrderedDict()
    for use_dust in ['true', 'false']:
        windowmasker_classification_arrays[check_dup][use_dust] = evaluate_windowmasker(check_dup, use_dust, verbose=False)


tantan_classification_arrays = collections.OrderedDict()
for m in ['atMask.mat', None]:
    tantan_classification_arrays[str(m)] = collections.OrderedDict()
#     for r in [0.000000000001, 0.000000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
    for r in [0.000000000001, 0.000000001, 0.000001, 0.001, 0.1]:
        tantan_classification_arrays[str(m)][str(r)] = evaluate_tantan(r, m, verbose=False)


str(ref_dict['Pf3D7_01_v3'].seq[100362:100536])


str(ref_dict['Pf3D7_01_v3'].seq[101707:101957])


for dust_level in [20, 30, 40, 50, 60, 70]:
    summarise_masking(
        dustmasker_classification_arrays[str(dust_level)][0],
        "Dust level %d" % dust_level
    )


for check_dup in ['true', 'false']:
    for use_dust in ['true', 'false']:
        summarise_masking(
            windowmasker_classification_arrays[check_dup][use_dust][0],
            "windowmasker check_dup=%s use_dust=%s" % (check_dup, use_dust)
        )


# for r in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
for r in [0.000000000001, 0.000000001, 0.000001, 0.001, 0.1]:
    for m in [None, 'atMask.mat']:
        summarise_masking(
            tantan_classification_arrays[str(m)][str(r)][0],
            "tantan r=%s m=%s" % (r, m)
        )
       








import pickle
dustmasker_classification_arrays_fn = REF_GENOME.replace('.fasta', 'dustmasker_classification_arrays.p')
pickle.dump(dustmasker_classification_arrays, open(dustmasker_classification_arrays_fn, "wb"))











# # Plan
# - Copy everything from 20161125_Pf60_final_vcfs.ipynb
# 

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pv3.0/20161130_Pv30_final_vcfs'
vrpipe_vcfs_dir = '/nfs/team112_internal/production_files/Pv/3_0/vcf'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pv/3_0_release_packages'
nfs_final_vcf_dir = '%s/vcf' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_vcf_dir}')

gff_fn = "/lustre/scratch116/malaria/pvivax/resources/snpEff/data/PvivaxP01_GeneDB_Oct2016/PvivaxP01.noseq.gff3"
cds_gff_fn = "%s/gff/PvivaxP01_GeneDB_Oct2016.PvivaxP01.noseq.gff3.cds.gz" % output_dir
annotations_header_fn = "%s/intermediate_files/annotations.hdr" % (output_dir)

run_create_multiallelics_file_job_fn = "%s/scripts/run_create_multiallelics_file_job.sh" % output_dir
submit_create_multiallelics_file_jobs_fn = "%s/scripts/submit_create_multiallelics_file_jobs.sh" % output_dir
create_study_vcf_job_fn = "%s/scripts/create_study_vcf_job.sh" % output_dir

vrpipe_metadata_fn = "%s/Pv_3.0_vrpipe_bam_summaries.txt" % output_dir
short_contigs_fofn = "%s/short_contigs.fofn" % output_dir

GENOME_FN = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
# BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

genome_fn = "%s/PvivaxP01.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/gff')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/study_vcfs')
get_ipython().system('mkdir -p {output_dir}/intermediate_files')
get_ipython().system('mkdir -p {output_dir}/tables')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')


cds_gff_fn


get_ipython().system("grep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")


fo=open(annotations_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (INDEL)">', file=fo)
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()


fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
JOB=$LSB_JOBINDEX
# JOB=19
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_FULL_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_SITES_VCF_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.txt.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pv_30_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN

bcftools view --drop-genotypes -Oz -o $INPUT_SITES_VCF_FN $INPUT_FULL_VCF_FN

python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC | \
bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\t%%CDS\t%%VARIANT_TYPE\t%%MULTIALLELIC\n' | \
bgzip -c > $ANNOTATION_FN

tabix -s1 -b2 -e2 $ANNOTATION_FN

#/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
#$ANNOTATION_FN

bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
bcftools view \
--include 'ALT!="*"' | \
bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CHROM,POS,REF,ALT,CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

bcftools index --tbi \
$NORMALISED_ANNOTATION_FN

bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_FULL_VCF_FN | \
bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AF,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC,^FORMAT/GT,FORMAT/AD,FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL,\
^FILTER/PASS,FILTER/Centromere,FILTER/InternalHypervariable,FILTER/SubtelomericHypervariable,\
FILTER/SubtelomericRepeat,FILTER/ShortContig,FILTER/Low_VQSLOD \
-Oz -o $OUTPUT_VCF_FN

bcftools index --tbi \
$OUTPUT_VCF_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        cds_gff_fn,
        annotations_header_fn,
        GENOME_FN,
        annotations_header_fn,
        cds_gff_fn,
        annotations_header_fn,
        )
        , file=fo)
fo.close()


get_ipython().system('bash {run_create_multiallelics_file_job_fn}')


fo = open(submit_create_multiallelics_file_jobs_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
LOG_DIR=%s/log
 
NUM_CHROMS=`wc -l < $FASTA_FAI_FILE`
QUEUE=long

bsub -q $QUEUE -G malaria-dk -J "ma[1-$NUM_CHROMS]" -n2 -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s
''' % (
        GENOME_FN,
        output_dir,
        "bash %s" % run_create_multiallelics_file_job_fn,
        ),
     file=fo)
fo.close()


get_ipython().system('bash {submit_create_multiallelics_file_jobs_fn}')


fo = open(short_contigs_fofn, 'w')
for chrom in sorted(list(genome.keys())):
    if chrom.startswith('Transfer'):
        vcf_fn = "%s/vcf/Pv_30_%s.final.vcf.gz" % (output_dir, chrom)
        print(vcf_fn, file=fo)
fo.close()


short_contigs_fofn


get_ipython().system('bcftools concat --file-list {short_contigs_fofn} -Oz -o {"%s/vcf/Pv_30_PvP01_00.final.vcf.gz" % output_dir}')


# # Create study-specific VCFs

get_ipython().system('cp {GENOME_FN} {genome_fn}')
genome = pyfasta.Fasta(genome_fn)
genome


sorted(list(genome.keys()))


genome_length = 0
for chrom in genome.keys():
    genome_length += len(genome[chrom])
genome_length


transfer_length = 0
for chrom in genome.keys():
    if chrom.startswith('Transfer'):
        transfer_length += len(genome[chrom])
transfer_length


4802351/29052596


vrpipe_columns = [
    'path', 'sample', 'study', 'bases_of_1X_coverage', 'bases_of_2X_coverage', 'bases_of_5X_coverage',
    'mean_coverage', 'mean_insert_size', 'sd_insert_size', 'avg_read_length', 'bases_callable_percent',
    'bases_no_coverage_percent', 'bases_low_coverage_percent', 'bases_excessive_coverage_percent',
    'bases_poor_mapping_quality_percent', 'bases_ref_n_percent', 'reads', 'reads_mapped', 'reads_mapped_and_paired',
    'reads_properly_paired', 'reads_qc_failed', 'pairs_on_different_chromosomes', 'non_primary_alignments',
    'center_name'
]
print(",".join(vrpipe_columns[1:]))


metadata_columns = [
    'sample', 'study', 'center_name', 'bases_callable_proportion', 'bases_no_coverage_proportion', 'bases_low_coverage_proportion',
#     'bases_excessive_coverage_proportion', 'bases_poor_mapping_quality_proportion', 'bases_ref_n_proportion',
    'bases_poor_mapping_quality_proportion',
    'proportion_genome_covered_at_1x', 'proportion_genome_covered_at_5x', 'mean_coverage',
    'mean_insert_size', 'sd_insert_size', 'avg_read_length', 
    'reads_mapped_proportion', 'mapped_reads_properly_paired_proportion', 'pairs_on_different_chromosomes_proportion',
    'non_primary_alignments_proportion',
]


get_ipython().system('vrpipe-fileinfo --setup pv_30_mergelanes --metadata {",".join(vrpipe_columns[1:])} > {vrpipe_metadata_fn}')

# | grep '\.summary' \


get_ipython().system('vrpipe-fileinfo --setup pf_60_mergelanes --metadata sample,study,bases_of_1X_coverage,bases_of_2X_coverage,bases_of_5X_coverage,mean_coverage,mean_insert_size,sd_insert_size,avg_read_length,bases_callable_percent,bases_no_coverage_percent,bases_low_coverage_percent,bases_excessive_coverage_percent,bases_poor_mapping_quality_percent,bases_ref_n_percent,reads,reads_mapped,reads_mapped_and_paired,reads_properly_paired,reads_qc_failed,pairs_on_different_chromosomes,non_primary_alignments,center_name > {vrpipe_metadata_fn}')

# | grep '\.summary' \
# | sort -k 2,2 \


vcf_samples = vcf.Reader(filename='%s/vcf/Pv_30_PvP01_01_v1.final.vcf.gz' % output_dir).samples
print(len(vcf_samples))
vcf_samples[0:10]


tbl_vcf_samples = etl.fromcolumns([vcf_samples]).setheader(['sample'])
print(len(tbl_vcf_samples.data()))


tbl_vcf_samples.duplicates('sample')


metadata_columns


tbl_sample_metadata = (
    etl
    .fromtsv(vrpipe_metadata_fn)
    .setheader(vrpipe_columns)
#     .select(lambda rec: 'pe' in rec['path'] or rec['sample'] == 'PN0002-C')
    .convertnumbers()
#     .convert('avg_read_length', lambda val: val+1)
    .addfield('bases_callable_proportion', lambda rec: 0.0 if rec['bases_callable_percent'] == 'unknown' else round(rec['bases_callable_percent'] / 100, 4))
    .addfield('bases_no_coverage_proportion', lambda rec: 0.0 if rec['bases_no_coverage_percent'] == 'unknown' else round(rec['bases_no_coverage_percent'] / 100, 4))
    .addfield('bases_low_coverage_proportion', lambda rec: 0.0 if rec['bases_low_coverage_percent'] == 'unknown' else round(rec['bases_low_coverage_percent'] / 100, 4))
    .addfield('bases_excessive_coverage_proportion', lambda rec: 0.0 if rec['bases_excessive_coverage_percent'] == 'unknown' else round(rec['bases_excessive_coverage_percent'] / 100, 4))
    .addfield('bases_poor_mapping_quality_proportion', lambda rec: 0.0 if rec['bases_poor_mapping_quality_percent'] == 'unknown' else round(rec['bases_poor_mapping_quality_percent'] / 100, 4))
    .addfield('bases_ref_n_proportion', lambda rec: 0.0 if rec['bases_ref_n_percent'] == 'unknown' else round(rec['bases_ref_n_percent'] / 100, 4))
    .addfield('proportion_genome_covered_at_1x', lambda rec: 0.0 if rec['bases_of_1X_coverage'] == 'unknown' else round(rec['bases_of_1X_coverage'] / genome_length, 4))
    .addfield('proportion_genome_covered_at_5x', lambda rec: 0.0 if rec['bases_of_5X_coverage'] == 'unknown' else round(rec['bases_of_5X_coverage'] / genome_length, 4))
    .addfield('reads_mapped_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_mapped'] / rec['reads'], 4))
    .addfield('mapped_reads_properly_paired_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_properly_paired'] / rec['reads_mapped'], 4))
    # Note in the following we use reads_properly_paired/2 to get numbers of pairs of reads
    .addfield('pairs_on_different_chromosomes_proportion', lambda rec: 0.0 if rec['pairs_on_different_chromosomes'] == 'unknown' or rec['pairs_on_different_chromosomes'] == 0.0 else round(rec['pairs_on_different_chromosomes'] / (rec['pairs_on_different_chromosomes'] + ( rec['reads_properly_paired'] / 2)), 4))
    .addfield('non_primary_alignments_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['non_primary_alignments'] / rec['reads_mapped'], 4))
    .addfield('reads_qc_failed_proportion', lambda rec: 0.0 if rec['reads_qc_failed'] == 'unknown' else round(rec['reads_qc_failed'] / rec['reads'], 4))
    #  .leftjoin(tbl_solaris_metadata, lkey='sample', rkey='ox_code')
    #  .convert('run_accessions', 'NULL', where=lambda rec: rec['study'] == '1156-PV-ID-PRICE') # These were wrongly accessioned and are currently being removed from ENA
    #  .cut(['sample', 'study', 'src_code', 'run_accessions', 'genome_covered_at_1x', 'genome_covered_at_5x',
    #        'mean_coverage', 'avg_read_length'])
#     .cut(metadata_columns)
    .selectin('sample', vcf_samples)
    .sort('sample')
)
print(len(tbl_sample_metadata.data()))
tbl_sample_metadata.display(index_header=True)


tbl_vcf_samples.antijoin(tbl_sample_metadata, key='sample')


tbl_sample_metadata.duplicates('sample').displayall()


print(len(tbl_sample_metadata.selectin('sample', vcf_samples).data()))


tbl_sample_metadata.valuecounts('avg_read_length').sort('avg_read_length').displayall()


tbl_sample_metadata.valuecounts('study').sort('study').displayall()


studies = tbl_sample_metadata.distinct('study').values('study').array()
studies


study_vcf_jobs_manifest = '%s/study_vcf_jobs_manifest.txt' % output_dir
fo = open(study_vcf_jobs_manifest, 'w')
for study in studies:
    sample_ids = ",".join(tbl_sample_metadata.selecteq('study', study).values('sample'))
    for chrom in sorted(genome.keys()):
        if chrom.startswith('PvP01'):
            print('%s\t%s\t%s' % (study, chrom, sample_ids), file=fo)
    print('%s\t%s\t%s' % (study, 'PvP01_00', sample_ids), file=fo)
fo.close()


fo = open(create_study_vcf_job_fn, 'w')
print('''STUDY_VCF_JOBS_FILE=%s
 
JOB=$LSB_JOBINDEX
# JOB=19
 
IN=`sed "$JOB q;d" $STUDY_VCF_JOBS_FILE`
read -a LINE <<< "$IN"
STUDY=${LINE[0]}
CHROM=${LINE[1]}
SAMPLES=${LINE[2]}

OUTPUT_DIR=%s

mkdir -p $OUTPUT_DIR/study_vcfs/$STUDY

INPUT_VCF_FN=$OUTPUT_DIR/vcf/Pv_30_$CHROM.final.vcf.gz
OUTPUT_VCF_FN=$OUTPUT_DIR/study_vcfs/$STUDY/Pv_30__$STUDY\__$CHROM.vcf.gz

echo $OUTPUT_VCF_FN
echo $STUDY

bcftools view --samples $SAMPLES --output-file $OUTPUT_VCF_FN --output-type z $INPUT_VCF_FN
bcftools index --tbi $OUTPUT_VCF_FN
md5sum $OUTPUT_VCF_FN > $OUTPUT_VCF_FN.md5

''' % (
        study_vcf_jobs_manifest,
        output_dir,
        )
        , file=fo)
fo.close()


get_ipython().system('bash {create_study_vcf_job_fn}')


QUEUE = 'normal'
wc_output = get_ipython().getoutput('wc -l {study_vcf_jobs_manifest}')
NUM_JOBS = wc_output[0].split(' ')[0]
MEMORY = 8000
LOG_DIR = "%s/log" % output_dir

print(NUM_JOBS, LOG_DIR)

get_ipython().system('bsub -q {QUEUE} -G malaria-dk -J "s_vcf[1-{NUM_JOBS}]" -n2 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {LOG_DIR}/output_%J-%I.log bash {create_study_vcf_job_fn}')


# # Copy files to /nfs

get_ipython().system('cp {output_dir}/vcf/* {nfs_final_vcf_dir}/')


get_ipython().system('cp -R {output_dir}/study_vcfs/* {nfs_release_dir}/')


for study in studies:
    get_ipython().system('cp /lustre/scratch116/malaria/pvivax/resources/gatk/pvp01_regions_20151213.bed.gz* {nfs_release_dir}/{study}/')


2+2





get_ipython().run_line_magic('run', '_standard_imports.ipynb')


scratch_dir = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161115_run_Olivo_GRC"
output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161115_run_Olivo_GRC"
get_ipython().system('mkdir -p {scratch_dir}/grc')
get_ipython().system('mkdir -p {scratch_dir}/species')
get_ipython().system('mkdir -p {scratch_dir}/log')
get_ipython().system('mkdir -p {output_dir}/grc')
get_ipython().system('mkdir -p {output_dir}/species')

bam_fn = "%s/pf_60_mergelanes.txt" % output_dir
bam_list_fn = "%s/pf_60_mergelanes_bamfiles.txt" % output_dir
chromosomeMap_fn = "%s/chromosomeMap.tab" % output_dir
grc_properties_fn = "%s/grc/grc.properties" % output_dir
species_properties_fn = "%s/species/species.properties" % output_dir
submitArray_fn = "%s/grc/submitArray.sh" % output_dir
submitSpeciesArray_fn = "%s/species/submitArray.sh" % output_dir
runArrayJob_fn = "%s/grc/runArrayJob.sh" % output_dir
runSpeciesArrayJob_fn = "%s/species/runArrayJob.sh" % output_dir
mergeGrcResults_fn = "%s/grc/mergeGrcResults.sh" % output_dir
mergeSpeciesResults_fn = "%s/species/mergeSpeciesResults.sh" % output_dir

ref_fasta_fn = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"


bam_list_fn


get_ipython().system('cp /nfs/users/nfs_r/rp7/pf_60_mergelanes.txt {bam_fn}')


# Create list of bam files in format required
tbl_bam_file = (etl
    .fromtsv(bam_fn)
    .addfield('ChrMap', 'Pf3k')
    .rename('path', 'BamFile')
    .rename('sample', 'Sample')
    .cut(['Sample', 'BamFile', 'ChrMap'])
)
tbl_bam_file.totsv(bam_list_fn)
get_ipython().system('dos2unix -o {bam_list_fn}')


fo = open(grc_properties_fn, 'w')
print('''grc.loci=crt_core,crt_ex01,crt_ex02,crt_ex03,crt_ex04,crt_ex06,crt_ex09,crt_ex10,crt_ex11,dhfr_1,dhfr_2,dhfr_3,dhps_1,dhps_2,dhps_3,dhps_4,mdr1_1,mdr1_2,mdr1_3,arps10,mdr2,fd,exo

# CRT
grc.locus.crt_core.region=Pf3D7_07_v3:403500-403800
grc.locus.crt_core.targets=crt_72-76@403612-403626
grc.locus.crt_core.anchors=403593@TATTATTTATTTAAGTGTA,403627@ATTTTTGCTAAAAGAAC

grc.locus.crt_ex01.region=Pf3D7_07_v3:403150-404420
grc.locus.crt_ex01.targets=crt_24@403291-403293
grc.locus.crt_ex01.anchors=403273@GAGCGTTATA.[AG]GAATTA...AATTTA.TACAAGAA[GA]GAA

grc.locus.crt_ex02.region=Pf3D7_07_v3:403550-403820
grc.locus.crt_ex02.targets=crt_97@403687-403689
grc.locus.crt_ex02.anchors=403657@GGTAACTATAGTTTTGT.[AT]CATC[CT]GAAAC,403690@AACTTTATTTGTATGATTA[TA]GTTCTTTATT

grc.locus.crt_ex03.region=Pf3D7_07_v3:403850-404170
grc.locus.crt_ex03.targets=crt_144@404007-404009,crt_148@404019-404021
grc.locus.crt_ex03.anchors=404022@ACAAGAACTACTGGAAA[TC]AT[CT]CA[AG]TCATTT,403977@TC[CT]AT.TTA.AT[GT]CCTGTTCA.T[CA]ATT

grc.locus.crt_ex04.region=Pf3D7_07_v3:404200-404500
grc.locus.crt_ex04.targets=crt_194@404329-404331,crt_220@404407-404409
grc.locus.crt_ex04.anchors=404304@CGGAGCA[GC]TTATTATTGTTGTAACA...GCTC,404338@GTAGAAATGAAATTATC[TA]TTTGAAACAC,404359@GAAACACAAGAAGAAAATTCTATC[AG]TATTTAATC,404382@C[AG]TATTTAATCTTGTCTTA[AT]TTAGT...TTAATTG

grc.locus.crt_ex06.region=Pf3D7_07_v3:404700-405000
grc.locus.crt_ex06.targets=crt_271@404836-404838
grc.locus.crt_ex06.anchors=404796@TTGTCTTATATT.CCTGTATACACCCTTCCATT[TC]TTAAAA...C

grc.locus.crt_ex09.region=Pf3D7_07_v3:405200-405500
grc.locus.crt_ex09.targets=crt_326@405361-405363,crt_333@405382-405384
grc.locus.crt_ex09.anchors=405334@AAAACCTT[CT]G[CT]ATTGTTTTCCTTCTTT,405364@A.TTGTGATAATTTAATA...AGCTAT

grc.locus.crt_ex10.region=Pf3D7_07_v3:405400-405750
grc.locus.crt_ex10.targets=crt_342@405557-405559,crt_356@405599-405601
grc.locus.crt_ex10.anchors=405539@ATTATCGACAAATTTTCT...[AT]TGACATATAC,405573@TTGTTAGTTGTATACAAG[GT]TCCA[GA]CA,405602@GCAATT[GT]CTTATTACTTTAAATTCTTA[GA]CC

grc.locus.crt_ex11.region=Pf3D7_07_v3:405700-406000
grc.locus.crt_ex11.targets=crt_371@405837-405839
grc.locus.crt_ex11.anchors=405825@[GT]GTGATGTT.[TA]A...G.ACCAAGATTATTAG,405840@G.ACCAAGATTATTAGATTTCGTAACTTTG

# DHFR
grc.locus.dhfr_1.region=Pf3D7_04_v3:748100-748400
grc.locus.dhfr_1.targets=dhfr_51@748238-748240,dhfr_59@748262-748264
grc.locus.dhfr_1.anchors=748200@GAGGTCTAGGAAATAAAGGAGTATTACCATGGAA,748241@TCCCTAGATATGAAATATTTT...GCAG,748265@GCAGTTACAACATATGTGAATGAATC

grc.locus.dhfr_2.region=Pf3D7_04_v3:748250-748550
grc.locus.dhfr_2.targets=dhfr_108@748409-748411
grc.locus.dhfr_2.anchors=748382@CAAAATGTTGTAGTTATGGGAAGAACA,748412@TGGGAAAGCATTCCAAAAAAATTT

grc.locus.dhfr_3.region=Pf3D7_04_v3:748400-748720
grc.locus.dhfr_3.targets=dhfr_164@748577-748579
grc.locus.dhfr_3.anchors=748382@GGGAAATTAAATTACTATAAATG,748382@CTATAAATGTTTTATT...GGAGGTTC,748412@GGAGGTTCCGTTGTTTATCAAG


# DHPS
grc.locus.dhps_1.region=Pf3D7_08_v3:549550-549750
grc.locus.dhps_1.targets=dhps_436@549681-549683,dhps_437@549684-549686
grc.locus.dhps_1.anchors=549657@GTTATAGAT[AG]TAGGTGGAGAATCC,549669@GGTGGAGAATCC..TG.TCC,549687@CCTTTTGTTAT[AG]CCTAATCCAAAAATTAGTG

grc.locus.dhps_2.region=Pf3D7_08_v3:549850-550150
grc.locus.dhps_2.targets=dhps_540@549993-549995
grc.locus.dhps_2.anchors=549949@GTGTAGTTCTAATGCATAAAAGAGG,549970@GAGGAAATCCACATACAATGGAT,549985@CAATGGAT...CTAACAAATTA[TA]GATA,549996@CTAACAAATTA[TA]GATAATCTAGT

grc.locus.dhps_3.region=Pf3D7_08_v3:549950-550250
grc.locus.dhps_3.targets=dhps_581@550116-550118
grc.locus.dhps_3.anchors=550092@CTATTTGATATTGGATTAGGATTT,550119@AAGAAACATGATCAATCT[AT]TTAAACTC

grc.locus.dhps_4.region=Pf3D7_08_v3:550050-550350
grc.locus.dhps_4.targets=dhps_613@550212-550214
grc.locus.dhps_4.anchors=550167@GATGAGTATCCACTTTTTATTGG,550188@GGATATTCAAGAAAAAGATTTATT,550215@CATTGCATGAATGATCAAAATGTTG


# MDR1
grc.locus.mdr1_1.region=Pf3D7_05_v3:957970-958280
grc.locus.mdr1_1.targets=mdr1_86@958145-958147
grc.locus.mdr1_1.anchors=958120@GTTTG[GT]TGTAATATTAAA[GA]AACATG,958141@CATG...TTAGGTGATGATATTAATCCT

grc.locus.mdr1_2.region=Pf3D7_05_v3:958300-958600
grc.locus.mdr1_2.targets=mdr1_184@958439-958441
grc.locus.mdr1_2.anchors=958413@CATATGC[CA]AGTTCCTTTTTAGG,958446@GGTC[AG]TTAATAAAAAAT[GA]CACGTTTGAC

grc.locus.mdr1_3.region=Pf3D7_05_v3:961470-961770
grc.locus.mdr1_3.targets=mdr1_1246@961625-961627
grc.locus.mdr1_3.anchors=961595@GTTATAGAT[AG]TAGGTGGAGAATCC,961628@CTTAGAAA[CT][TA]TATTTTC[AT]ATAGTTAGTC

# ARPS10
grc.locus.arps10.region=Pf3D7_14_v3:2480900-2481200
grc.locus.arps10.targets=arps10_127@2481070-2481072
grc.locus.arps10.anchors=2481045@ATTTAC[CA]TTTTTGCGATCTCCCCAT...[GC],2481079@GACAGT[AC]G[AG]GA[GA]CAATTCGAAATAAAAC

# MDR2
grc.locus.mdr2.region=Pf3D7_14_v3:1956070-1956370
grc.locus.mdr2.targets=mdr2_484@-1956224-1956226
grc.locus.mdr2.anchors=1956203@ACATGTTATTAATCCT[TC]TAT...TGCC,1956227@TGCCGGAATAAT[AG]TACATTAAAACAGAAC

# Ferredoxin
grc.locus.fd.region=Pf3D7_13_v3:748250-748550
grc.locus.fd.targets=fd_193@-748393-748395
grc.locus.fd.anchors=748396@[GA]TGTAGTTCGTCTTCCTTGTG[CT]GTTTC

# Exo
grc.locus.exo.region=Pf3D7_13_v3:2504400-2504700
grc.locus.exo.targets=exo_415@2504559-2504561
grc.locus.exo.anchors=2504526@[GC]ATGATTTTA[AG][CA]AATATGGT[TC]ATAA[CT]GATAAAA,2504562@GAA[GT]TAAA[CT][AC]ATCATTGG[GA]AAAA[TC]AATATATAC
''', file=fo)
fo.close() 





fo = open(species_properties_fn, 'w')
print('''sampleClass.classes=Pf,Pv,Pm,Pow,Poc,Pk
sampleClass.loci=mito1,mito2,mito3,mito4,mito5,mito6 

sampleClass.locus.mito1.region=M76611:520-820 
sampleClass.locus.mito1.anchors=651@CCTTACGTACTCTAGCT....ACACAA
sampleClass.locus.mito1.targets=species1@668-671&678-683
sampleClass.locus.mito1.target.species1.alleles=Pf@ATGATTGTCT|ATGATTGTTT,Pv@TTTATATTAT,Pm@TTGTATTAAT,Pow@ATTTACATAA,Poc@ATTTATATAT,Pk@TTTTTATTAT

sampleClass.locus.mito2.region=M76611:600-900 
sampleClass.locus.mito2.anchors=741@GAATAGAA...GAACTCTATAAATAACCA
sampleClass.locus.mito2.targets=species2@728-733&740-740&749-751&770-773
sampleClass.locus.mito2.target.species2.alleles=Pf@GTTCATTTAAGATT|GTTCATTTAAGACT,Pv|Pk@TATTCATAAATACA,Pm@GTTCAATTAGTACT,Pow|Poc@GTTACAATAATATT

sampleClass.locus.mito3.region=M76611:720-1020 
sampleClass.locus.mito3.anchors=842@(?:GAAAGAATTTATAA|ATATA[AG]TGAATATG)ACCAT
sampleClass.locus.mito3.targets=species3@861-869&878-881&884-887
sampleClass.locus.mito3.target.species3.alleles=Pf@TCGGTAGAATATTTATT,Pv@TCACTATTACATTAACT,Pm@TCACTATTTAATATATC,Pow@CCCTTATTTAACTAACC|TCCTTATTTAACTAACC,Poc@TCGTTATTAAACTAACC,Pk@TCACAATTAAACTTATT

sampleClass.locus.mito4.region=M76611:820-1120 
sampleClass.locus.mito4.anchors=948@CCTGTAACACAATAAAATAATGT
sampleClass.locus.mito4.targets=species4@971-982
sampleClass.locus.mito4.target.species4.alleles=Pf@AGTATATACAGT,Pv|Pow|Poc@ACCAGATATAGC,Pm@TCCTGAAACTCC,Pk@ACCTGATATAGC

sampleClass.locus.mito5.region=M76611:900-1200 
sampleClass.locus.mito5.anchors=1029@GATGCAAAACATTCTCC
sampleClass.locus.mito5.targets=species5@1025-1028&1046-1049
sampleClass.locus.mito5.target.species5.alleles=Pf@TAGATAAT,Pv|Pk@AAGTAAGT,Pm@TAATAAGT,Pow@TAATAAGA,Poc@TAATAAGG

sampleClass.locus.mito6.region=M76611:950-1250
sampleClass.locus.mito6.anchors=1077@ATTTC[AT]AAACTCAT[TA]CCTTTTTCTA
sampleClass.locus.mito6.targets=species6@1062-1066&1073-1073&1076-1076&1082-1082&1091-1091&1102-1108
sampleClass.locus.mito6.target.species6.alleles=Pf@CAAATAGATTAAATAC,Pv|Pk@AATACAATTTTAGAAA|AATATAATTTTAGAAA,Pm@AATATTTAAAAAGAAA,Pow|Poc@AATATTTTTTGAGAAA|AATATTTTTTAAGAAA
''', file=fo)
fo.close() 


fo = open(chromosomeMap_fn, 'w')
print('''default	Pf3k
Pf3D7_01_v3	Pf3D7_01_v3
Pf3D7_02_v3	Pf3D7_02_v3
Pf3D7_03_v3	Pf3D7_03_v3
Pf3D7_04_v3	Pf3D7_04_v3
Pf3D7_05_v3	Pf3D7_05_v3
Pf3D7_06_v3	Pf3D7_06_v3
Pf3D7_07_v3	Pf3D7_07_v3
Pf3D7_08_v3	Pf3D7_08_v3
Pf3D7_09_v3	Pf3D7_09_v3
Pf3D7_10_v3	Pf3D7_10_v3
Pf3D7_11_v3	Pf3D7_11_v3
Pf3D7_12_v3	Pf3D7_12_v3
Pf3D7_13_v3	Pf3D7_13_v3
Pf3D7_14_v3	Pf3D7_14_v3
M76611	Pf_M76611
PFC10_API_IRAB	Pf3D7_API_v3
''', file=fo)
fo.close()


fo = open(runArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()


fo = open(runSpeciesArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

echo
echo $SAMPLE_NAME
echo $BAM_FILE
echo $CHR_MAP_NAME
echo
echo $JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
echo

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()


fo = open(submitArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=normal
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%25" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runArrayJob_fn,
        ),
     file=fo)
fo.close()


fo = open(submitSpeciesArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/species
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=small
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%25" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runSpeciesArrayJob_fn,
        ),
     file=fo)
fo.close()


fo = open(mergeGrcResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()


fo = open(mergeSpeciesResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
OUT_DIR=%s/species
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.species.SpeciesAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()


# # Kicking off pipeline

get_ipython().system('bash {submitArray_fn}')


get_ipython().system('bash {mergeGrcResults_fn}')


get_ipython().system('bash {submitSpeciesArray_fn}')


submitSpeciesArray_fn


get_ipython().system('bash {mergeSpeciesResults_fn}')


# # Introduction
# This notebook creates PCA plots for samples in the Pf 6.0 internal release. The main purpose of this is sample QC: identifying samples that look like outliers based on the study they are in or their reported collection location.
# 

# ## Setup
# 
# Let's import the libraries we'll be using.

import numpy as np
import scipy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import h5py
import allel; print('scikit-allel', allel.__version__)
import collections


callset_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5'
callset = h5py.File(callset_fn, mode='r')
callset


variants = allel.VariantChunkedTable(callset['variants'], 
                                     names=['CHROM', 'POS', 'FILTER_PASS', 'CDS', 'MULTIALLELIC', 'VARIANT_TYPE', 'AC', 'AN', 'AF', 'VQSLOD'],
                                     index=['CHROM', 'POS'])
variants


np.unique(variants['CDS'][:], return_counts=True)


pca_selection = (
    (variants['FILTER_PASS'][:]) &
    (variants['CDS'][:]) &
    (variants['MULTIALLELIC'][:] == b'BI') &
    (variants['VARIANT_TYPE'][:] == b'SNP') &
    (variants['AF'][:,0] >= 0.05) &
    (variants['AF'][:,0] <= 0.95) &
    (variants['VQSLOD'][:] > 6.0)
)


np.unique(pca_selection, return_counts=True)


variants_pca = variants.compress(pca_selection)
variants_pca


calldata = callset['calldata']
calldata


genotypes = allel.GenotypeChunkedArray(calldata['genotype'])
genotypes


get_ipython().run_cell_magic('time', '', 'genotypes_pca = genotypes.subset(pca_selection)')


genotypes_pca


get_ipython().run_cell_magic('time', '', 'n_variants = len(variants_pca)\npc_missing = genotypes_pca.count_missing(axis=0)[:] * 100 / n_variants')


fig, ax = plt.subplots(figsize=(12, 4))
_ = ax.hist(pc_missing[pc_missing<10], bins=100)


good_samples = (pc_missing < 2.0)
genotypes_pca_good = genotypes_pca.take(np.where(good_samples)[0], axis=1)
genotypes_pca_good


gn = genotypes_pca_good.to_n_alt()[:]
gn


get_ipython().run_cell_magic('time', '', 'coords, model = allel.stats.pca(gn)')


samples_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt'
samples = pandas.DataFrame.from_csv(samples_fn, sep='\t')
samples.head()


continents = collections.OrderedDict()
continents['1001-PF-ML-DJIMDE']               = '1_WA'
continents['1004-PF-BF-OUEDRAOGO']            = '1_WA'
continents['1006-PF-GM-CONWAY']               = '1_WA'
continents['1007-PF-TZ-DUFFY']                = '2_EA'
continents['1008-PF-SEA-RINGWALD']            = '4_SEA'
continents['1009-PF-KH-PLOWE']                = '4_SEA'
continents['1010-PF-TH-ANDERSON']             = '4_SEA'
continents['1011-PF-KH-SU']                   = '4_SEA'
continents['1012-PF-KH-WHITE']                = '4_SEA'
continents['1013-PF-PEGB-BRANCH']             = '6_SA'
continents['1014-PF-SSA-SUTHERLAND']          = '3_AF'
continents['1015-PF-KE-NZILA']                = '2_EA'
continents['1016-PF-TH-NOSTEN']               = '4_SEA'
continents['1017-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1018-PF-GB-NEWBOLD']              = '9_Lab'
continents['1020-PF-VN-BONI']                 = '4_SEA'
continents['1021-PF-PG-MUELLER']              = '5_OC'
continents['1022-PF-MW-OCHOLLA']              = '2_EA'
continents['1023-PF-CO-ECHEVERRI-GARCIA']     = '6_SA'
continents['1024-PF-UG-BOUSEMA']              = '2_EA'
continents['1025-PF-KH-PLOWE']                = '4_SEA'
continents['1026-PF-GN-CONWAY']               = '1_WA'
continents['1027-PF-KE-BULL']                 = '2_EA'
continents['1031-PF-SEA-PLOWE']               = '4_SEA'
continents['1044-PF-KH-FAIRHURST']            = '4_SEA'
# continents['1052-PF-TRAC-WHITE']              = '4_SEA'
continents['1052-PF-TRAC-WHITE']              = '0_MI'
continents['1062-PF-PG-BARRY']                = '5_OC'
continents['1083-PF-GH-CONWAY']               = '1_WA'
continents['1093-PF-CM-APINJOH']              = '1_WA'
continents['1094-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1095-PF-TZ-ISHENGOMA']            = '2_EA'
continents['1096-PF-GH-GHANSAH']              = '1_WA'
continents['1097-PF-ML-MAIGA']                = '1_WA'
continents['1098-PF-ET-GOLASSA']              = '2_EA'
continents['1100-PF-CI-YAVO']                 = '1_WA'
continents['1101-PF-CD-ONYAMBOKO']            = '1_WA'
continents['1102-PF-MG-RANDRIANARIVELOJOSIA'] = '2_EA'
continents['1103-PF-PDN-GMSN-NGWA']           = '1_WA'
continents['1107-PF-KEN-KAMAU']               = '2_EA'
continents['1125-PF-TH-NOSTEN']               = '4_SEA'
continents['1127-PF-ML-SOULEYMANE']           = '1_WA'
continents['1131-PF-BJ-BERTIN']               = '1_WA'
continents['1133-PF-LAB-MERRICK']             = '9_Lab'
continents['1134-PF-ML-CONWAY']               = '1_WA'
continents['1135-PF-SN-CONWAY']               = '1_WA'
continents['1136-PF-GM-NGWA']                 = '1_WA'
continents['1137-PF-GM-DALESSANDRO']          = '1_WA'
continents['1138-PF-CD-FANELLO']              = '1_WA'
continents['1141-PF-GM-CLAESSENS']            = '1_WA'
continents['1145-PF-PE-GAMBOA']               = '6_SA'
continents['1147-PF-MR-CONWAY']               = '1_WA'
continents['1151-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1152-PF-DBS-GH-AMENGA-ETEGO']     = '1_WA'
continents['1155-PF-ID-PRICE']                = '5_OC'


samples['continent'] = pandas.Series([continents[x] for x in samples.study], index=samples.index)


samples['is_SA'] = pandas.Series(samples['continent'] == '6_SA', index=samples.index)


samples.continent.value_counts()


samples.is_SA.value_counts()


samples_subset = samples[good_samples]
samples_subset.reset_index(drop=True, inplace=True)
samples_subset.head()





def plot_pca_coords(coords, model, pc1, pc2, ax, variable='continent', exclude_values=['9_Lab', '3_AF', '0_MI']):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for value in samples_subset[variable].unique():
        if not value in exclude_values:
            flt = (samples_subset[variable] == value).values
            ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=value, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax, variable='study')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax, variable='is_SA')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 3, 4, ax, variable='continent')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 3, 4, ax, variable='is_SA')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 5, 6, ax, variable='is_SA')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 7, 8, ax, variable='is_SA')
ax.legend(loc='upper left');


fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 5, ax)
ax.legend(loc='upper left');








get_ipython().run_cell_magic('time', '', 'genotypes_subset = genotypes.subset(variant_selection, sample_selection)')














# Define a function to plot variant density in windows over the chromosome.

def plot_windowed_variant_density(pos, window_size, title=None):
    
    # setup windows 
    bins = np.arange(0, pos.max(), window_size)
    
    # use window midpoints as x coordinate
    x = (bins[1:] + bins[:-1])/2
    
    # compute variant density in each window
    h, _ = np.histogram(pos, bins=bins)
    y = h / window_size
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y)
    ax.set_xlabel('Chromosome position (bp)')
    ax.set_ylabel('Variant density (bp$^{-1}$)')
    if title:
        ax.set_title(title)


# Now make a plot with the SNP positions from our chosen chromosome.

for current_chrom in chroms:
    plot_windowed_variant_density(
        pos[chrom==current_chrom], window_size=1000, title='Raw variant density %s' % current_chrom.decode('ascii')
    )


# From this we can see that variant density is around 0.2 over much of the genome, which means the raw data contains a variant about every 5 bases of the reference genome. Variant density much higher in var gene regions, as expected, with almost every base being variant.

# ## Explore variant attributes
# 
# As I mentioned above, each variant also has a number "annotations", which are data attributes that originally came from the "INFO" field in the VCF file. These are important for data quality, so let's begin by getting to know a bit more about the numerical range and distribution of some of these attributes.
# 
# Each attribute can be loaded from the table we setup earlier into a numpy array. E.g., load the "DP" field into an array.

dp = variants['DP'][:]
dp


# Define a function to plot a frequency distribution for any variant attribute.

def plot_variant_hist(f, bins=30):
    x = variants[f][:]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')
    ax.set_title('Variant %s distribution' % f)


# "DP" is total depth of coverage across all samples.

plot_variant_hist('DP', bins=50)


# "MQ" is average mapping quality across all samples.

plot_variant_hist('MQ')


# "QD" is a slightly odd statistic but turns out to be very useful for finding poor quality variants. Roughly speaking, high numbers mean that evidence for variation is strong (concentrated), low numbers mean that evidence is weak (dilute).

plot_variant_hist('QD')


# Finally let's see how many biallelic, triallelic, quadriallelic, etc variants we have.

plot_variant_hist('num_alleles', bins=np.arange(1.5, 8.5, 1))
plt.gca().set_xticks([2, 3, 4, 5, 6, 7]);


# We can also look at the joint frequency distribution of two attributes.

def plot_variant_hist_2d(f1, f2, downsample):
    x = variants[f1][:][::downsample]
    y = variants[f2][:][::downsample]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.despine(ax=ax, offset=10)
    ax.hexbin(x, y, gridsize=40)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s joint distribution' % (f1, f2))


# To make the plotting go faster I've downsampled to use every 10th variant.

plot_variant_hist_2d('QD', 'MQ', downsample=10)


# ## Investigate variant quality
# 
# The DP, MQ and QD attributes are potentially informative about SNP quality. For example, we have a prior expectation that putative SNPs with very high or very low DP may coincide with some form of larger structural variation, and may therefore be unreliable. However, it would be great to have some empirical indicator of data quality, which could guide our choices about how to filter the data.
# 
# There are several possible quality indicators that could be used, and in general it's a good idea to use more than one if available. Here, to illustrate the general idea, let's use just one indicator, which is the number of [transitions]() divided by the number of [transversions](), which I will call Ti/Tv.
# 
# ![Transitions and transversions](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Transitions-transversions-v4.svg/500px-Transitions-transversions-v4.svg.png)
# 
# If mutations were completely random we would expect a Ti/Tv of 0.5, because there are twice as many possible transversions as transitions. However, in most species a mutation bias has been found towards transitions, and so we expect the true Ti/Tv to be higher. We can therefore look for features of the raw data that are associated with low Ti/Tv (close to 0.5) and be fairly confident that these contain a lot of noise. 
# 
# To do this, let's first set up an array of mutations, where each entry contains two characters representing the reference and alternate allele. For simplicity of presentation I'm going to ignore the fact that some SNPs are multiallelic, but if doing this for real this should be restricted to biallelic variants only.

mutations = np.char.add(variants['REF'].subset(variants['is_snp']), variants['ALT'].subset(variants['is_snp'])[:, 0])
mutations


# Define a function to locate transition mutations within a mutations array.

def locate_transitions(x):
    x = np.asarray(x)
    return (x == b'AG') | (x == b'GA') | (x == b'CT') | (x == b'TC')


# Demonstrate how the ``locate_transitions`` function generates a boolean array from a mutations array.

is_ti = locate_transitions(mutations)
is_ti


# Define a function to compute Ti/Tv.

def ti_tv(x):
    if len(x) == 0:
        return np.nan
    is_ti = locate_transitions(x)
    n_ti = np.count_nonzero(is_ti)
    n_tv = np.count_nonzero(~is_ti)
    if n_tv > 0:
        return n_ti / n_tv
    else:
        return np.nan


# Demonstrate the ``ti_tv`` function by computing Ti/Tv over all SNPs.

ti_tv(mutations)


# Define a function to plot Ti/Tv in relation to a variant attribute like DP or MQ.

def plot_ti_tv(f, downsample, bins):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    x = variants[f].subset(variants['is_snp'])[:][::downsample]
    
    # plot a histogram
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')

    # plot Ti/Tv
    ax = ax.twinx()
    sns.despine(ax=ax, bottom=True, left=True, right=False, offset=10)
    values = mutations[::downsample]
    with np.errstate(over='ignore'):
        # binned_statistic generates an annoying overflow warning which we can ignore
        y1, _, _ = scipy.stats.binned_statistic(x, values, statistic=ti_tv, bins=bins)
    bx = (bins[1:] + bins[:-1]) / 2
    ax.plot(bx, y1, color='k')
    ax.set_ylabel('Ti/Tv')
    ax.set_ylim(0.6, 1.3)

    ax.set_title('Variant %s and Ti/Tv' % f)


# Example the relationship between the QD, MQ and DP attributes and Ti/Tv. 

plot_ti_tv('QD', downsample=5, bins=np.arange(0, 40, 1))


plot_ti_tv('MQ', downsample=5, bins=np.arange(0, 60, 1))


plot_ti_tv('DP', downsample=5, bins=np.linspace(0, 50000, 50))


# Ti/Tv is not a simple variable and so some care is required when interpreting these plots. However, we can see that there is a trend towards low Ti/Tv for low values of QD, MQ and DP.
# 
# To investigate further, let's look at Ti/Tv in two dimensions. 

def plot_joint_ti_tv(f1, f2, downsample, gridsize=20, mincnt=20, vmin=0.6, vmax=1.4, extent=None):
    fig, ax = plt.subplots()
    sns.despine(ax=ax, offset=10)
    x = variants[f1].subset(variants['is_snp'])[:][::downsample]
    y = variants[f2].subset(variants['is_snp'])[:][::downsample]
    C = mutations[::downsample]
    im = ax.hexbin(x, y, C=C, reduce_C_function=ti_tv, mincnt=mincnt, extent=extent,
                   gridsize=gridsize, cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s and Ti/Tv' % (f1, f2))


plot_joint_ti_tv('QD', 'MQ', downsample=5, mincnt=400, extent=(0, 40, 0, 80))


plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400, extent=(0, 40, 0, 8e+5))
# plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400)


plot_joint_ti_tv('MQ', 'DP', downsample=5, mincnt=400, extent=(0, 80, 0, 8e+5))


# This information may be useful when designing a variant filtering strategy. If you have other data that could be used as a quality indicator, such as Mendelian errors in a trio or cross, and/or data on genotype discordances between replicate samples, a similar analysis could be performed.

# ## Filtering variants
# 
# There are many possible approaches to filtering variants. The simplest approach is define thresholds on variant attributes like DP, MQ and QD, and exclude SNPs that fall outside of a defined range (a.k.a. "hard filtering"). This is crude but simple to implement and in many cases may suffice, at least for an initial exploration of the data. 
# 
# Let's implement a simple hard filter. First, a reminder that we have a table containing all these variant attributes.

variants


# Define the hard filter using an expression. This is just a string of Python code, which we will evaluate in a moment.

filter_expression = '(QD > 5) & (MQ > 40) & (DP > 3e+5) & (DP < 8e+5)'


# Now evaluate the filter using the columns from the table

variant_selection = variants.eval(filter_expression)[:]
variant_selection


# How many variants to we keep?

np.count_nonzero(variant_selection)


# How many variants do we filter out?

np.count_nonzero(~variant_selection)


# Now that we have our variant filter, let's make a new variants table with only rows for variants that pass our filter.

variants_pass = variants.compress(variant_selection)
variants_pass


# ## Cleanup to reduce RAM usage

import gc
gc.collect()


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))


del(variants)
del(mutations)
del(pos)
del(dp)
del(chrom)
del(_3)
del(_31)
del(_18)
del(_7)
del(_10)
del(_5)
gc.collect()


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))


# 
# 
# ## Subset genotypes
# 
# Now that we have some idea of variant quality, let's look at our samples and at the genotype calls.
# 
# All data relating to the genotype calls is stored in the HDF5.

calldata = callset['calldata']
calldata


list(calldata.keys())


# Each of these is a separate dataset in the HDF5 file. To make it easier to work with the genotype dataset, let's wrap it using a class from scikit-allel.

genotypes = allel.GenotypeChunkedArray(calldata['genotype'])
genotypes


# N.B., at this point we have not loaded any data into memory, it is still in the HDF5 file. From the representation above we have some diagnostic information about the genotypes, for example, we have calls for 6,051,695 variants in 7,182 samples with ploidy 2 (i.e., diploid). Uncompressed these data would be 81.0G but the data are compressed and so actually use 6.4G on disk.
# 
# We can also see genotype calls for the last 3 variants in the first and last 5 samples, which are all missing ("./.").
# 
# Before we go any furter, let's also pull in some data about the 7,182 samples we've genotyped.

samples_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt'
samples = pandas.DataFrame.from_csv(samples_fn, sep='\t')
samples.head()


# The "study" column defines which of 54 studies the parasites came from. How many parasites come from each of these studies?

samples.study.value_counts()


# These don't tell us much about geography, so let's assign each study to an approximate continental grouping

continents = collections.OrderedDict()
continents['1001-PF-ML-DJIMDE']               = '1_WA'
continents['1004-PF-BF-OUEDRAOGO']            = '1_WA'
continents['1006-PF-GM-CONWAY']               = '1_WA'
continents['1007-PF-TZ-DUFFY']                = '2_EA'
continents['1008-PF-SEA-RINGWALD']            = '4_SEA'
continents['1009-PF-KH-PLOWE']                = '4_SEA'
continents['1010-PF-TH-ANDERSON']             = '4_SEA'
continents['1011-PF-KH-SU']                   = '4_SEA'
continents['1012-PF-KH-WHITE']                = '4_SEA'
continents['1013-PF-PEGB-BRANCH']             = '6_SA'
continents['1014-PF-SSA-SUTHERLAND']          = '3_AF'
continents['1015-PF-KE-NZILA']                = '2_EA'
continents['1016-PF-TH-NOSTEN']               = '4_SEA'
continents['1017-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1018-PF-GB-NEWBOLD']              = '9_Lab'
continents['1020-PF-VN-BONI']                 = '4_SEA'
continents['1021-PF-PG-MUELLER']              = '5_OC'
continents['1022-PF-MW-OCHOLLA']              = '2_EA'
continents['1023-PF-CO-ECHEVERRI-GARCIA']     = '6_SA'
continents['1024-PF-UG-BOUSEMA']              = '2_EA'
continents['1025-PF-KH-PLOWE']                = '4_SEA'
continents['1026-PF-GN-CONWAY']               = '1_WA'
continents['1027-PF-KE-BULL']                 = '2_EA'
continents['1031-PF-SEA-PLOWE']               = '4_SEA'
continents['1044-PF-KH-FAIRHURST']            = '4_SEA'
continents['1052-PF-TRAC-WHITE']              = '4_SEA'
continents['1062-PF-PG-BARRY']                = '5_OC'
continents['1083-PF-GH-CONWAY']               = '1_WA'
continents['1093-PF-CM-APINJOH']              = '1_WA'
continents['1094-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1095-PF-TZ-ISHENGOMA']            = '2_EA'
continents['1096-PF-GH-GHANSAH']              = '1_WA'
continents['1097-PF-ML-MAIGA']                = '1_WA'
continents['1098-PF-ET-GOLASSA']              = '2_EA'
continents['1100-PF-CI-YAVO']                 = '1_WA'
continents['1101-PF-CD-ONYAMBOKO']            = '1_WA'
continents['1102-PF-MG-RANDRIANARIVELOJOSIA'] = '2_EA'
continents['1103-PF-PDN-GMSN-NGWA']           = '1_WA'
continents['1107-PF-KEN-KAMAU']               = '2_EA'
continents['1125-PF-TH-NOSTEN']               = '4_SEA'
continents['1127-PF-ML-SOULEYMANE']           = '1_WA'
continents['1131-PF-BJ-BERTIN']               = '1_WA'
continents['1133-PF-LAB-MERRICK']             = '9_Lab'
continents['1134-PF-ML-CONWAY']               = '1_WA'
continents['1135-PF-SN-CONWAY']               = '1_WA'
continents['1136-PF-GM-NGWA']                 = '1_WA'
continents['1137-PF-GM-DALESSANDRO']          = '1_WA'
continents['1138-PF-CD-FANELLO']              = '1_WA'
continents['1141-PF-GM-CLAESSENS']            = '1_WA'
continents['1145-PF-PE-GAMBOA']               = '6_SA'
continents['1147-PF-MR-CONWAY']               = '1_WA'
continents['1151-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1152-PF-DBS-GH-AMENGA-ETEGO']     = '1_WA'
continents['1155-PF-ID-PRICE']                = '5_OC'


samples['continent'] = pandas.Series([continents[x] for x in samples.study], index=samples.index)


samples.continent.value_counts()


samples


# Let's work with two populations only for simplicity. These are *Plasmodium falciparum* populations from Oceania (5_OC) and South America (6_SA).

sample_selection = samples.continent.isin({'5_OC', '6_SA'}).values
sample_selection[:5]


sample_selection = samples.study.isin(
    {'1010-PF-TH-ANDERSON', '1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE'}
).values
sample_selection[:5]


# Now restrict the samples table to only these two populations.

samples_subset = samples[sample_selection]
samples_subset.reset_index(drop=True, inplace=True)
samples_subset.head()


samples_subset.continent.value_counts()


# Now let's subset the genotype calls to keep only variants that pass our quality filters and only samples in our two populations of interest.

get_ipython().run_cell_magic('time', '', 'genotypes_subset = genotypes.subset(variant_selection, sample_selection)')


# This takes a few minutes, so time for a quick tea break.

genotypes_subset


# The new genotype array we've made has 1,816,619 variants and 98 samples, as expected.

# ## Sample QC
# 
# Before we go any further, let's do some sample QC. This is just to check if any of the 98 samples we're working with have major quality issues that might confound an analysis. 
# 
# Compute the percent of missing and heterozygous genotype calls for each sample.

get_ipython().run_cell_magic('time', '', 'n_variants = len(variants_pass)\npc_missing = genotypes_subset.count_missing(axis=0)[:] * 100 / n_variants\npc_het = genotypes_subset.count_het(axis=0)[:] * 100 / n_variants')


# Define a function to plot genotype frequencies for each sample.

def plot_genotype_frequency(pc, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    left = np.arange(len(pc))
    palette = sns.color_palette()
    pop2color = {'1_WA': palette[0], '6_SA': palette[1], '4_SEA': palette[2]}
    colors = [pop2color[p] for p in samples_subset.continent]
    ax.bar(left, pc, color=colors)
    ax.set_xlim(0, len(pc))
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title(title)
    handles = [mpl.patches.Patch(color=palette[0]),
               mpl.patches.Patch(color=palette[1])]
    ax.legend(handles=handles, labels=['1_WA', '6_SA', '4_SEA'], title='Population',
              bbox_to_anchor=(1, 1), loc='upper left')


# Let's look at missingness first.

plot_genotype_frequency(pc_missing, 'Missing')


# All samples have pretty low missingness, though generally slightly higher in South America than in West Africa, as might be expected given the 3D7 reference is thought to originate from West Africa. Just for comparison with Alistair's original notebook on which this is based, let's look at the sample with the highest missingness.

np.argsort(pc_missing)[-1]


# Let's dig a little more into this sample. Is the higher missingness spread over the whole genome, or only in a specific region? Choose two other samples to compare with.

g_strange = genotypes_subset.take([30, 62, 63], axis=1)
g_strange


# Locate missing calls.

is_missing = g_strange.is_missing()[:]
is_missing


# Plot missingness for each sample over the chromosome.

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_pass['CHROM'][:]==current_chrom
    pos = variants_pass['POS'][:][this_chrom_variant]
    window_size = 10000
    y1, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 0], statistic=np.count_nonzero, size=window_size)
    y2, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 1], statistic=np.count_nonzero, size=window_size)
    y3, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 2], statistic=np.count_nonzero, size=window_size)
    x = windows.mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1 * 100 / window_size, lw=1)
    ax.plot(x, y2 * 100 / window_size, lw=1)
    ax.plot(x, y3 * 100 / window_size, lw=1)
    ax.set_title(current_chrom.decode('ascii'))
    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Percent calls');


# The sample with higher missingness (in red) has generally higher missingness in the same places as other samples (i.e. in var gene regions)
# 
# Let's look at heterozygosity.

plot_genotype_frequency(pc_het, 'Heterozygous')


# No samples stand out, although it looks like there is a general trend for lower heterozogysity in the South American population.
# 

# ## Allele count
# 
# As a first step into doing some population genetic analyses, let's perform an allele count within each of the two populations we've selected. This just means, for each SNP, counting how many copies of the reference allele (0) and each of the alternate alleles (1, 2, 3) are observed.
# 
# To set this up, define a dictionary mapping population names onto the indices of samples within them.

subpops = {
    'all': list(range(len(samples_subset))),
    'WA': samples_subset[samples_subset.continent == '1_WA'].index.tolist(),
    'SA': samples_subset[samples_subset.continent == '6_SA'].index.tolist(),
    'SEA': samples_subset[samples_subset.continent == '4_SEA'].index.tolist(),
}
subpops['WA'][:5]


# Now perform the allele count.

get_ipython().run_cell_magic('time', '', 'ac_subpops = genotypes_subset.count_alleles_subpops(subpops, max_allele=6)')


ac_subpops


# Each column in the table above has allele counts for a population, where "all" means the union of both populations. We can pull out a single column, e.g.:

ac_subpops['SA'][:5]


# So in the SA population, at the first variant (index 0) we observe 82 copies of the reference allele (0) and 2 copies of the first alternate allele (1).

# ## Locate segregating variants
# 
# There are lots of SNPs which do not segregate in either of these populations are so are not interesting for any analysis of these populations. We might as well get rid of them.
# 
# How many segregating SNPs are there in each population?

for pop in 'all', 'WA', 'SA', 'SEA':
    print(pop, ac_subpops[pop].count_segregating())


# Locate SNPs that are segregating in the union of our two selected populations.

is_seg = ac_subpops['all'].is_segregating()[:]
is_seg


# Subset genotypes again to keep only the segregating SNPs.

genotypes_seg = genotypes_subset.compress(is_seg, axis=0)
genotypes_seg


# Subset the variants and allele counts too.

variants_seg = variants_pass.compress(is_seg)
variants_seg


ac_seg = ac_subpops.compress(is_seg)
ac_seg


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, int(eval("%s.nbytes" % object_name) / 1e+6))


# ## Population differentiation
# 
# Are these two populations genetically different? To get a first impression, let's plot the alternate allele counts from each population.

jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SA');


jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SEA');
jsfs


jsfs = allel.stats.joint_sfs(ac_seg['SA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, SA')
ax.set_ylabel('Alternate allele count, SEA');


jsfs


# So the alternate allele counts are correlated, meaning there is some relationship between these two populations, however there are plenty of SNPs off the diagonal, suggesting there is also some differentiation.
# 
# Let's compute average Fst, a statistic which summarises the difference in allele frequencies averaged over all SNPs. This also includes an estimate of standard error via jacknifing in blocks of 100,000 SNPs.

fst, fst_se, _, _ = allel.stats.blockwise_hudson_fst(ac_seg['WA'], ac_seg['SA'], blen=100000)
print("Hudson's Fst: %.3f +/- %.3f" % (fst, fst_se))


# Define a function to plot Fst in windows over the chromosome.

def plot_fst(ac1, ac2, pos, blen=2000, current_chrom=b'Pf3D7_01_v3'):
    
    fst, se, vb, _ = allel.stats.blockwise_hudson_fst(ac1, ac2, blen=blen)
    
    # use the per-block average Fst as the Y coordinate
    y = vb
    
    # use the block centres as the X coordinate
    x = allel.stats.moving_statistic(pos, statistic=lambda v: (v[0] + v[-1]) / 2, size=blen)
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y, 'k-', lw=.5)
    ax.set_ylabel('$F_{ST}$')
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())


# Are any chromosome regions particularly differentiated?

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    plot_fst(
        ac_seg['WA'].subset(this_chrom_variant),
        ac_seg['SA'].subset(this_chrom_variant),
        variants_seg['POS'][:][this_chrom_variant],
        100,
        current_chrom
    )


# Maybe some interesting signals of differentiation here.
# 
# There are a number of subtleties to Fst analysis which I haven't mentioned here, but you can read more about [estimating Fst](http://alimanfoo.github.io/2015/09/21/estimating-fst.html) on my blog.
# 
# ## Site frequency spectra
# 
# While we're looking at allele counts, let's also plot a site frequency spectrum for each population, which gives another summary of the data and is also informative about demographic history.
# 
# To do this we really do need to restrict to biallelic variants, so let's do that first.

is_biallelic_01 = ac_seg['all'].is_biallelic_01()[:]
ac1 = ac_seg['WA'].compress(is_biallelic_01, axis=0)[:, :2]
ac2 = ac_seg['SA'].compress(is_biallelic_01, axis=0)[:, :2]
ac3 = ac_seg['SEA'].compress(is_biallelic_01, axis=0)[:, :2]
ac1


# OK, now plot folded site frequency spectra, scaled such that populations with constant size should have a spectrum close to horizontal (constant across allele frequencies).

fig, ax = plt.subplots(figsize=(8, 5))
sns.despine(ax=ax, offset=10)
sfs1 = allel.stats.sfs_folded_scaled(ac1)
allel.stats.plot_sfs_folded_scaled(sfs1, ax=ax, label='WA', n=ac1.sum(axis=1).max())
sfs2 = allel.stats.sfs_folded_scaled(ac2)
allel.stats.plot_sfs_folded_scaled(sfs2, ax=ax, label='SA', n=ac2.sum(axis=1).max())
sfs3 = allel.stats.sfs_folded_scaled(ac3)
allel.stats.plot_sfs_folded_scaled(sfs3, ax=ax, label='SEA', n=ac3.sum(axis=1).max())
ax.legend()
ax.set_title('Scaled folded site frequency spectra')
# workaround bug in scikit-allel re axis naming
ax.set_xlabel('minor allele frequency');


# The spectra are very different for the three populations. WA has an excess of rare variants, suggesting a population expansion, while SA and SEA are closer to neutral expectation, suggesting a more stable population size.
# 
# We can also plot Tajima's D, which is a summary of the site frequency spectrum, over the chromosome, to see if there are any interesting localised variations in this trend.

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    # compute windows with equal numbers of SNPs
    pos = variants_seg['POS'][:][this_chrom_variant]
    windows = allel.stats.moving_statistic(pos, statistic=lambda v: [v[0], v[-1]], size=100)
    x = np.asarray(windows).mean(axis=1)

    # compute Tajima's D
    y1, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['WA'].subset(this_chrom_variant), windows=windows)
    y2, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SA'].subset(this_chrom_variant), windows=windows)
    y3, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SEA'].subset(this_chrom_variant), windows=windows)

    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1, lw=.5, label='WA')
    ax.plot(x, y2, lw=.5, label='SA')
    ax.plot(x, y3, lw=.5, label='SEA')
    ax.set_ylabel("Tajima's $D$")
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1));


# Not really sure what to make of the above!

# ## Principal components analysis
# 
# Finally, let's to a quick-and-dirty PCA to confirm our evidence for differentiation between these two populations and check if there is any other genetic structure within populations that we might have missed.
# 
# First grab the allele counts for the union of the two populations.

ac = ac_seg['all'][:]
ac


# Select the variants to use for the PCA, including only biallelic SNPs with a minor allele count above 2.

pca_selection = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 2)
pca_selection


np.count_nonzero(pca_selection)


# Now randomly downsample these SNPs purely for speed.

indices = np.nonzero(pca_selection)[0]
indices


len(indices)


indices_ds = np.random.choice(indices, size=50000, replace=False)
indices_ds.sort()
indices_ds


# Subset the genotypes to keep only our selected SNPs for PCA.

genotypes_pca = genotypes_seg.take(indices_ds, axis=0)
genotypes_pca


# Transform the genotypes into an array of alternate allele counts per call. 

gn = genotypes_pca.to_n_alt()[:]
gn


# Run the PCA.

coords, model = allel.stats.pca(gn)


coords


coords.shape


# Plot the results.

def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for pop in ['1_WA', '6_SA', '4_SEA']:
        flt = (samples_subset.continent == pop).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=pop, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))


fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();


def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for study in ['1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE', '1010-PF-TH-ANDERSON']:
        flt = (samples_subset.study == study).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=study, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))


fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();


samples.index[sample_selection][(coords[:, 0] > -50) & (coords[:, 0] < -20)]


samples.index[sample_selection][coords[:, 0] > 0]


samples.index[sample_selection][coords[:, 1] > 0]


coords[(samples.index == 'PP0012-C')[sample_selection]]


coords[(samples.index == 'PP0022-C')[sample_selection]]


coords[(samples.index == 'PP0022-Cx')[sample_selection]]


coords[(samples.index == 'PD0047-C')[sample_selection]]


coords[(samples.index == 'PP0018-C')[sample_selection]]


fig, ax = plt.subplots(figsize=(5, 4))
sns.despine(ax=ax, offset=10)
y = 100 * model.explained_variance_ratio_
x = np.arange(len(y))
ax.set_xticks(x + .4)
ax.set_xticklabels(x + 1)
ax.bar(x, y)
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance explained (%)');


# From this PCA we can see that PC1 and PC2 separate the three populations. Some of the South American samples are clustering together with South-East Asia - so maybe these are the lab sample (Dd2) contaminants mentioned by Julian?
# 
# For running PCA with more populations there are a number of subtleties which I haven't covered here, for all the gory details see the [fast PCA](http://alimanfoo.github.io/2015/09/28/fast-pca.html) article on my blog.

# ## Under the hood
# 
# Here's a few notes on what's going on under the hood. If you want to know more, the best place to look is the [scikit-allel source code](https://github.com/cggh/scikit-allel).

# ### NumPy arrays
# 
# NumPy is the foundation for everything in scikit-allel. A NumPy array is an N-dimensional container for binary data.

x = np.array([0, 4, 7])
x


x.ndim


x.shape


x.dtype


# item access
x[1]


# slicing
x[0:2]


# NumPy support array-oriented programming, which is both convenient and efficient, because looping is implemented internally in C code. 

y = np.array([1, 6, 9])
x + y


# Scikit-allel defines a number of conventions for storing variant call data using NumPy arrays. For example, a set of diploid genotype calls over *m* variants in *n* samples is stored as a NumPy array of integers with shape (m, n, 2). 

g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[0, 2], [-1, -1]]], dtype='i1')
g


# The ``allel.GenotypeArray`` class is a sub-class of ``np.ndarray``.

isinstance(g, np.ndarray)


# All the usual properties and methods of an ndarray are inherited.

g.ndim


g.shape


# obtain calls for the second variant in all samples
g[1, :]


# obtain calls for the second sample in all variants
g[:, 1]


# obtain the genotype call for the second variant, second sample
g[1, 1]


# make a subset with only the first and third variants
g.take([0, 2], axis=0)


# find missing calls
np.any(g < 0, axis=2)


# Instances of ``allel.GenotypeArray`` also have some extra properties and methods. 

g.n_variants, g.n_samples, g.ploidy


g.count_alleles()


# ### Chunked, compressed arrays
# 
# The ``scikit-allel`` genotype array convention is flexible, allowing for multiallelic and polyploid genotype calls. However, it is not very compact, requiring 2 bytes of memory for each call. A set of calls for 10,000,000 SNPs in 1,000 samples thus requires 20G of memory.
# 
# One option to work with large arrays is to use bit-packing, i.e., to pack two or more items of data into a single byte. E.g., this is what the [plink BED format](http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml) does. If you have have diploid calls that are only ever biallelic, then it is possible to fit 4 genotype calls into a single byte. This is 8 times smaller than the NumPy unpacked representation.
# 
# However, coding against bit-packed data is not very convenient. Also, there are several libraries available for Python which allow N-dimensional arrays to be stored using **compression**: [h5py](http://www.h5py.org/), [bcolz](http://bcolz.blosc.org/en/latest/) and [zarr](http://zarr.readthedocs.io). Genotype data is usually extremely compressible due to sparsity - most calls are homozygous ref, i.e., (0, 0), so there are a lot of zeros. 
# 
# For example, the ``genotypes`` data we used above has calls for 16 million variants in 765 samples, yet requires only 1.2G of storage. In other words, there are more than 9 genotype calls per byte, which means that each genotype call requires less than a single bit on average.

genotypes


# The data for this array are stored in an HDF5 file on disk and compressed using zlib, and achieve a compression ratio of 19.1 over an equivalent uncompressed NumPy array.
# 
# To avoid having to decompress the entire dataset every time you want to access any part of it, the data are divided into chunks and each chunk is compressed. You have to choose the chunk shape, and there are some trade-offs regarding both the shape and size of a chunk. 
# 
# Here is the chunk shape for the ``genotypes`` dataset.

genotypes.chunks


# This means that the dataset is broken into chunks where each chunk has data for 6553 variants and 10 samples.
# 
# This gives a chunk size of ~128K (6553 \* 10 \* 2) which we have since found is not optimal - better performance is usually achieved with chunks that are at least 1M. However, performance is not bad and the data are publicly released so I haven't bothered to rechunk them.
# 
# Chunked, compressed arrays can be stored either on disk (as for the ``genotypes`` dataset) or in main memory. E.g., in the tour above, I stored all the intermediate genotype arrays in memory, such as the ``genotypes_subset`` array, which can speed things up a bit.

genotypes_subset


# To perform some operation over a chunked arrays, the best way is to compute the result for each chunk separately then combine the results for each chunk if needed. All functions in ``scikit-allel`` try to use a chunked implementation wherever possible, to avoid having to load large data uncompressed into memory.

# ## Further reading
# 
# 
# * [scikit-allel reference documentation](http://scikit-allel.readthedocs.io/)
# * [Introducing scikit-allel](http://alimanfoo.github.io/2015/09/15/introducing-scikit-allel.html)
# * [Estimating Fst](http://alimanfoo.github.io/2015/09/21/estimating-fst.html)
# * [Fast PCA](http://alimanfoo.github.io/2015/09/28/fast-pca.html)
# * [To HDF5 and beyond](http://alimanfoo.github.io/2016/04/14/to-hdf5-and-beyond.html)
# * [CPU blues](http://alimanfoo.github.io/2016/05/16/cpu-blues.html)
# * [vcfnp](https://github.com/alimanfoo/vcfnp)
# * [numpy](http://www.numpy.org/)
# * [matplotlib](http://matplotlib.org/)
# * [pandas](http://pandas.pydata.org/)
# 

import datetime
print(datetime.datetime.now().isoformat())


# # Intro
# Aim here is to find breakpoint for samples that appear to be duplicated around plasmepsin I-III, but not with the same breakpoints as most other samples. See email from Rob 21/09/2016 17:15.
# 
# I installed IGV on maslrv and ran with:
# 
# ssh -C -X -c blowfish malsrv2
# 
# /nfs/team112_internal/rp7/opt/igv/IGV_2.3.81/igv.sh
# 
# Load genome (Genomes/Load Genome from File) with /lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta
# 
# Load bam using manifest below to find correct file
# 
# Set "Show soft-clipped bases" on Alignments tab (View/Preferences)
# 
# Zoom to Pf3D7_14_v3:280,000-310,000
# 
# Later decided that running IGV over -X was too slow so copied bam files to MacBook
# 

# # Plan
# - Try searching for "1-sided" reads near long homopolymer-A
# - Also search for reads spanning homopolymer, but with variable length polyA, and 1-5 bases downstream
# - pysam code to find number and proportion of pair where one end in one region, with particular orientation, and mate in second region with expected orientation
# - generalise the above code to find all peaks of faceaway reads in larger region
# - HMM code to search for coverage changes in whole region
# - combine evidence from all above approaches
# 
# 
# - pysam code to pull out all soft-clipped reads in a region?
# 

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


from Bio.Seq import Seq


diff_breakpoint_samples = ['PH0243-C', 'PH0247-C', 'PH0484-C', 'PH0906-C', 'PH0912-C']
mdr1_samples = ['PH0254-C']


# Note we used same samples as were previously used for HRP deletions. See 20160621_HRP_sample_metadata.ipynb
manifest_5_0_fn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'
# Note the following file created whilst 6.0 build still in progress, so don't have final sample bams
manifest_6_0_fn = '/nfs/team112_internal/rp7/data/Pf/6_0/metadata/plasmepsin_manifest_6_0_20160922.txt'
scratch_dir = '/lustre/scratch109/malaria/rp7/data/ppq'

genome_fn = '/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta'


genome = SeqIO.to_dict(SeqIO.parse(genome_fn, "fasta"))
# , key_function=get_accession)


genome.keys()


genome['Pf3D7_14_v3'].count()


tbl_manifest = etl.fromtsv(manifest_fn)
print(len(tbl_manifest.data()))
tbl_manifest


tbl_bams_5_0 = tbl_manifest.selectin('sample', diff_breakpoint_samples)
tbl_bams_5_0.display()


tbl_manifest.selectin('sample', mdr1_samples)


# # Results
# PH0243-C: polyA (36bp) at 283,034 to polyA (30bp) at 300,493 (17kb)
# Think PH0247-C and PH0484-C have the same breakpoints
# 
# PH0906-C: polyA (36bp) at 283,034 to polyA (30bp) at 362,990 (80kb)
# Think PH0912-C has same breakpoints
# 

genome['Pf3D7_14_v3'][283032:283070].seq


genome['Pf3D7_14_v3'].seq.count(genome['Pf3D7_14_v3'][283020:283069].seq)


temp = list(genome.keys())
temp.sort()
temp


sorted(genome.keys())


import re
for chrom in sorted(genome.keys()):
#     print(chrom, genome[chrom].seq.count(genome['Pf3D7_14_v3'][283027:283069].seq))
#     print(chrom, genome[chrom].seq.find('agaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(chrom, [m.start() for m in re.finditer('agaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

print()

for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaaaaaagaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

print()

for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaaaagaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

    





for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaatgaagggaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])


breakpoint_sequences = collections.OrderedDict()
breakpoint_sequences['0bp'] = 'GATAATCACAC'
breakpoint_sequences['1bp'] = 'CGATAATCACACT'
breakpoint_sequences['5bp'] = 'ATTACGATAATCACACTGTTG'
breakpoint_sequences['10bp'] = 'TTATGATTACGATAATCACACTGTTGGTTTC'
breakpoint_sequences['15bp'] = 'ACCGTTTATGATTACGATAATCACACTGTTGGTTTCGCCCT'
breakpoint_sequences['20bp'] = 'ATTTTACCGTTTATGATTACGATAATCACACTGTTGGTTTCGCCCTTGCCA'


breakpoint_sequences['0bp'].lower()


for breakpoint_sequence in [x.lower() for x in breakpoint_sequences.values()]:
    print(breakpoint_sequence)
    for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])


for breakpoint_sequence in ['cgataatcacact', 'cgataatcacac', 'gataatcacact', 'gataatcacac']:
    print(breakpoint_sequence)
    for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])


genome['Pf3D7_14_v3'][283027:283069].seq





for breakpoint_sequence in ['aaaaaaaaaaaaaaaaaaaa']:
    print(breakpoint_sequence)
    for chrom in ['Pf3D7_05_v3']:
#     for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])


# # Copy bams to MacBook
# Decided to copy both Pf 5.0 bams (mapped with bwa aln) and 6.0 bams (mapped with bwa mem)
# 

print("mkdir -p /lustre/scratch109/malaria/rp7/data/ppq/bams/5_0/")
for rec in tbl_bams_5_0.data():
    original_bam = rec[0]
    macbook_bam = "%s/bams/5_0/%s.bam" % (scratch_dir, rec[1])
    print("scp malsrv2:%s %s" % (original_bam.replace('.bam', '.bam.bai'), macbook_bam.replace('.bam', '.bam.bai')))
    print("scp malsrv2:%s %s" % (original_bam, macbook_bam))


get_ipython().system('mkdir -p {os.path.dirname(manifest_6_0_fn)}')
get_ipython().system('vrpipe-fileinfo --setup pf_60_bqsr --metadata sample > {manifest_6_0_fn}')


tbl_manifest_6_0 = etl.fromtsv(manifest_6_0_fn).select(lambda rec: rec[0][-3:] == 'bam')
print(len(tbl_manifest_6_0.data()))
tbl_manifest_6_0


tbl_bams_6_0 = tbl_manifest_6_0.selectin('sample', diff_breakpoint_samples).sort('sample')
tbl_bams_6_0.display()


print("mkdir -p /lustre/scratch109/malaria/rp7/data/ppq/bams/6_0/")
for rec in tbl_bams_6_0.data():
    original_bam = rec[0]
    macbook_bam = "%s/bams/6_0/%s.bam" % (scratch_dir, rec[1])
    print("scp malsrv2:%s %s" % (original_bam.replace('.bam', '.bai'), macbook_bam.replace('.bam', '.bai')))
    print("scp malsrv2:%s %s" % (original_bam, macbook_bam))


tbl_manifest_6_0.selectin('sample', mdr1_samples)





import stat
bsub = sh.Command('bsub')

for breakpoint_sequence_name in breakpoint_sequences:
# for breakpoint_sequence_name in ['10bp']:
    breakpoint_reads_dir = '%s/plasmepsin_1_3_7979_samples/%s' % (output_dir, breakpoint_sequence_name)
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/results')
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/scripts')
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/logs')

    breakpoint_sequence = "'%s|%s'" % (
        breakpoint_sequences[breakpoint_sequence_name],
        Seq(breakpoint_sequences[breakpoint_sequence_name]).reverse_complement()
    )
    for rec in tbl_manifest:
        bam_fn = rec[0]
        ox_code = rec[1]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        if not os.path.exists(num_breakpoint_reads_fn):
            script_fn = "%s/scripts/nbpr_%s.sh" % (breakpoint_reads_dir, ox_code)
            fo = open(script_fn, 'w')
            print('samtools view %s | grep -E %s | wc -l > %s' % (bam_fn, breakpoint_sequence, num_breakpoint_reads_fn), file = fo)
            fo.close()
            st = os.stat(script_fn)
            os.chmod(script_fn, st.st_mode | stat.S_IEXEC)
            bsub(
                '-G', 'malaria-dk',
                '-P', 'malaria-dk',
                '-q', 'normal',
                '-o', '%s/logs/nbpr_%s.out' % (breakpoint_reads_dir, ox_code),
                '-e', '%s/logs/nbpr_%s.err' % (breakpoint_reads_dir, ox_code),
                '-J', 'nbpr_%s' % (ox_code),
                '-R', "'select[mem>4000] rusage[mem=4000]'",
                '-M', '4000',
                script_fn
            )


tbl_manifest





pf_5_0_breakpoint_reads = collections.OrderedDict()

# for breakpoint_sequence_name in breakpoint_sequences:
for breakpoint_sequence_name in ['10bp']:
    pf_5_0_breakpoint_reads[breakpoint_sequence_name] = collections.OrderedDict()
    breakpoint_reads_dir = '%s/plasmepsin_1_3_pf_5_0/%s' % (output_dir, breakpoint_sequence_name)
    for rec in tbl_5_0_manifest:
        bam_fn = rec[0]
        ox_code = rec[2]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        fi = open(num_breakpoint_reads_fn, 'r')
        pf_5_0_breakpoint_reads[breakpoint_sequence_name][ox_code] = int(fi.read())


breakpoint_reads_dict = collections.OrderedDict()

for breakpoint_sequence_name in breakpoint_sequences:
    print(breakpoint_sequence_name)
# for breakpoint_sequence_name in ['10bp']:
    breakpoint_reads_dict[breakpoint_sequence_name] = collections.OrderedDict()
    breakpoint_reads_dir = '%s/plasmepsin_1_3_7979_samples/%s' % (output_dir, breakpoint_sequence_name)
    for rec in tbl_manifest.data():
        bam_fn = rec[0]
        ox_code = rec[1]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        fi = open(num_breakpoint_reads_fn, 'r')
        breakpoint_reads_dict[breakpoint_sequence_name][ox_code] = int(fi.read())


tbl_breakpoint_reads = (etl.wrap(zip(breakpoint_reads_dict['0bp'].keys(), breakpoint_reads_dict['0bp'].values())).pushheader(['ox_code', 'bp_reads_0bp'])
 .join(
        etl.wrap(zip(breakpoint_reads_dict['1bp'].keys(), breakpoint_reads_dict['1bp'].values())).pushheader(['ox_code', 'bp_reads_1bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['5bp'].keys(), breakpoint_reads_dict['5bp'].values())).pushheader(['ox_code', 'bp_reads_5bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['10bp'].keys(), breakpoint_reads_dict['10bp'].values())).pushheader(['ox_code', 'bp_reads_10bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['15bp'].keys(), breakpoint_reads_dict['15bp'].values())).pushheader(['ox_code', 'bp_reads_15bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['20bp'].keys(), breakpoint_reads_dict['20bp'].values())).pushheader(['ox_code', 'bp_reads_20bp']),
        key='ox_code')
)
# tbl_breakpoint_reads.displayall()


tbl_breakpoint_reads.selectgt('bp_reads_10bp', 0).displayall()


tbl_breakpoint_reads.totsv("%s/plasmepsin_1_3_7979_samples.tsv" % output_dir)


tbl_breakpoint_reads.toxlsx("%s/plasmepsin_1_3_7979_samples.xlsx" % output_dir)


2+2





# This notebook must be run directly from MacBook after running ~/bin/sanger-tunneling.sh in order to connect
# to Sanger network. I haven't figured out a way to do this from Docker container

get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


get_ipython().system('rsync -avL malsrv2:/nfs/team112_internal/rp7/data/pf3k/analysis /nfs/team112_internal/rp7/data/pf3k/')





# # Introduction
# Was planning on comparing different defintions of the core, i.e. crosses vs pacbio vs a new defintion based on positions of genes, but decided the positions of genes approach wasn't as straightforward as I had hoped (think just using last VAR/RIF/STEVOR would probably not be as good as crosses definition), so decided to abandon, at least for now. Will come back to this if there is an appetite for a gene-to-gene based defintion of the core (or at least the short read mappable core).

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')
import pandas as pd


output_dir = '/nfs/team112_internal/rp7/data/methods-dev/pf3k_techbm/20170216_core_defintion_comparisons'
gff_fn = '/lustre/scratch118/malaria/team112/pipelines/resources/pf3k_methods/resources/snpEff/data/Pfalciparum_GeneDB_Aug2015/genes.gff'


get_ipython().system("grep 'product=term%3Dstevor%3B' {gff_fn}")


get_ipython().system("grep 'Name=VAR' {gff_fn}")





get_ipython().run_line_magic('run', '_standard_imports.ipynb')


panoptes_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_20170124.txt.gz"
panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
oxford_table_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/oxford.txt"
pf3k_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata.txt"
crosses_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_crosses_metadata.txt"
new_pf3k_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata_20170210.txt.gz"


tbl_pf_60_metadata = (
    etl
    .fromtsv(panoptes_metadata_fn)
#     .cut(['Sample', 'pf60_OxfordSrcCode', 'pf60_OxfordDonorCode', 'Individual_ID', 'pf60_AlfrescoStudyCode',
#           'pf60_HasDuplicate', 'pf60_DiscardAsDuplicate', 'pc_pass_missing', 'pc_genome_covered_at_1x'])
)
print(len(tbl_pf_60_metadata.data()))
tbl_pf_60_metadata


(
    tbl_pf_60_metadata
    .valuecounts('AlfrescoStudyCode')
    .displayall()
)


(
    tbl_pf_60_metadata
    .selectin('AlfrescoStudyCode', ['1155-PF-ID-PRICE', '1106-PV-MULTI-PRICE'])
    .valuecounts('pf60_Location')
    .displayall()
)


tbl_pf_60_metadata = (
    etl
    .fromtsv(panoptes_final_metadata_fn)
#     .cut(['Sample', 'OxfordSrcCode', 'OxfordDonorCode', 'Individual_ID', 'IndividualGroup', 'AlfrescoStudyCode',
#           'HasDuplicate', 'DiscardAsDuplicate', 'pc_pass_missing', 'pc_genome_covered_at_1x'])
    .cut(['Sample', 'OxfordSrcCode', 'OxfordDonorCode', 'Individual_ID'])
)
print(len(tbl_pf_60_metadata.data()))
tbl_pf_60_metadata


tbl_pf3k_metadata = (
    etl
    .fromtsv(pf3k_metadata_fn)
)
print(len(tbl_pf3k_metadata.data()))
tbl_pf3k_metadata


tbl_oxford = (
    etl
    .fromtsv(oxford_table_fn)
    .distinct('oxford_code')
    .cut(['oxford_code', 'oxford_source_code', 'oxford_donor_source_code'])
)
print(len(tbl_oxford.distinct('oxford_code').data()))
print(len(tbl_oxford.data()))
tbl_oxford


tbl_crosses = (
    etl
    .fromtsv(crosses_metadata_fn)
    .cut(['sample', 'study', 'clone'])
)
print(len(tbl_crosses.distinct('sample').data()))
print(len(tbl_crosses.data()))
tbl_crosses


def is_duplicate(prv, cur, nxt):
    if prv is None and (nxt['Individual_ID'] == cur['Individual_ID']):
        return(True)
    elif prv is None and (nxt['Individual_ID'] != cur['Individual_ID']):
        return(False)
    elif nxt is None and (prv['Individual_ID'] == cur['Individual_ID']):
        return(True)
    elif nxt is None and (prv['Individual_ID'] != cur['Individual_ID']):
        return(False)
    elif(
            (prv['Individual_ID'] == cur['Individual_ID']) or
            (nxt['Individual_ID'] == cur['Individual_ID'])
        ):
        return(True)
    else:
        return(False)
    
def discard_as_duplicate(prv, cur, nxt):
    if prv is None:
        return(False)
    elif (prv['Individual_ID'] == cur['Individual_ID']):
        return(True)
    else:
        return(False)
    


# final_columns = list(tbl_pf3k_metadata.header()) + ['IsFieldSample', 'PreferredSample', 'AllSamplesThisIndividual', 'DiscardAsDuplicate', 'HasDuplicate', 'Individual_ID']
final_columns = list(tbl_pf3k_metadata.header()) + ['IsFieldSample', 'PreferredSample', 'AllSamplesThisIndividual']
final_columns


tbl_temp = (
    tbl_pf3k_metadata
    .leftjoin(tbl_pf_60_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_oxford, lkey='sample', rkey='oxford_code')
    .leftjoin(tbl_crosses, key='sample', rprefix='crosses_')
    .convert('Individual_ID', lambda v, r: r['crosses_clone'], where=lambda r: r['study'] in ['1041', '1042', '1043'], pass_row=True)
    .convert('Individual_ID', lambda v, r: r['sample'], where=lambda r: r['study'] in ['Broad Senegal', '1104', ''], pass_row=True)
    .convert('Individual_ID', lambda v, r: r['oxford_donor_source_code'], where=lambda r: r['Individual_ID'] is None, pass_row=True)
    .convert('Individual_ID', lambda v, r: 'PF955', where=lambda r: r['Individual_ID'] == 'PF955_MACS', pass_row=True)
    .addfield('IsFieldSample', lambda r: r['country'] != '')
    .sort(['Individual_ID', 'bases_of_5X_coverage'], reverse=True)
    .addfieldusingcontext('HasDuplicate', is_duplicate)
    .addfieldusingcontext('DiscardAsDuplicate', discard_as_duplicate)
    .addfield('PreferredSample', lambda rec: rec['DiscardAsDuplicate'] == False)
)

# tbl_temp.totsv('temp.txt')
# tbl_temp = etl.fromtsv('temp.txt')

tbl_duplicates_sample = (
    tbl_temp
    .aggregate('Individual_ID', etl.strjoin(','), 'sample')
    .rename('value', 'AllSamplesThisIndividual')
)

tbl_new_pf3k_metadata = (
    tbl_temp
    .leftjoin(tbl_duplicates_sample, key='Individual_ID')
    .cut(final_columns)
    .sort('sample')
    .sort('IsFieldSample', reverse=True)
)

print(len(tbl_new_pf3k_metadata.distinct('sample').data()))
print(len(tbl_new_pf3k_metadata.data()))
tbl_new_pf3k_metadata


tbl_new_pf3k_metadata.totsv(new_pf3k_metadata_fn, lineterminator='\n')
# tbl_new_pf3k_metadata.totsv(new_pf3k_metadata_fn)


# Sanity check new file is same as old but with extra_columns
get_ipython().system('zcat {new_pf3k_metadata_fn} | cut -f 1-23 > /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt')


get_ipython().system('sed \'s/"//g\' /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata.txt > /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')


get_ipython().system('diff /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')


get_ipython().system('rm /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt')
get_ipython().system('rm /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')





# # Sanity checks

tbl_new_pf3k_metadata.selecteq('Individual_ID', 'PF955')


tbl_new_pf3k_metadata.valuecounts('IsFieldSample')


tbl_new_pf3k_metadata.valuecounts('PreferredSample')


tbl_new_pf3k_metadata.valuecounts('DiscardAsDuplicate')


tbl_new_pf3k_metadata.selecteq('PreferredSample', False).displayall()


tbl_new_pf3k_metadata.selecteq('Individual_ID', 'EIMK002').displayall()


tbl_temp.valuecounts('DiscardAsDuplicate')


tbl_new_pf3k_metadata.valuecounts('HasDuplicate')


tbl_new_pf3k_metadata.selectnone('HasDuplicate').valuecounts('study').displayall()


tbl_new_pf3k_metadata.selectnone('HasDuplicate')


tbl_new_pf3k_metadata.selecteq('Individual_ID', 'PFD140')


tbl_new_pf3k_metadata.selecteq('Individual_ID', '')


tbl_new_pf3k_metadata.selectnone('Individual_ID')


len(tbl_new_pf3k_metadata.selecteq('study', 'Broad Senegal').data())


print(len(tbl_new_pf3k_metadata.distinct('AllSamplesThisIndividual').data()))
print(len(tbl_new_pf3k_metadata.distinct(('study', 'AllSamplesThisIndividual')).data()))








tbl_new_pf3k_metadata.selecteq('Individual_ID', 'HB3')


tbl_new_pf3k_metadata.selecteq('Individual_ID', 'C02')


tbl_new_pf3k_metadata.selecteq('Individual_ID', 'A4')








# # Earlier work to determine duplicates
# Note this is no longer relevant after final file created

tbl_new_pf3k_metadata.valuecounts('study').displayall()


(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .valuecounts('study')
    .displayall()
)


(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .valuecounts('study')
    .displayall()
)


(
    tbl_new_pf3k_metadata
    .selecteq('Individual_ID', '')
    .valuecounts('study')
    .displayall()
)


(
    tbl_new_pf3k_metadata
    .selectnotin('study', ['Broad Senegal', '', '1041', '1042', '1043', '1104'])
    .selecteq('AlfrescoStudyCode', None)
    .valuecounts('study', 'AlfrescoStudyCode')
    .displayall()
)


(
    tbl_new_pf3k_metadata
    .selectnotin('study', ['Broad Senegal', '', '1041', '1042', '1043', '1104'])
    .selecteq('AlfrescoStudyCode', None)
    .displayall()
)


(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .selecteq('AlfrescoStudyCode', None)
    .displayall()
)


# # Manual checking
# After manual checking, all samples from study 1017 are unique.
# 
# 1052 has duplicates, but PH0979-C/KH002-110 is unique
# 
# For 1022: H183, H185, H196, H198, H220, H224, H229, H237, H250 have duplicates. All can be matched on source code
# 
# 1083 contains duplicates that can be matched on source code
# 
# 1044 has one duplicate that needs manual matching




get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161212_sample_level_summaries'
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pf60')
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pv30')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

GENOME_FN = collections.OrderedDict()
genome_fn = collections.OrderedDict()
genome = collections.OrderedDict()
GENOME_FN['Pf60'] = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
GENOME_FN['Pv30'] = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
genome_fn['Pf60'] = "%s/Pfalciparum.genome.fasta" % output_dir
genome_fn['Pv30'] = "%s/PvivaxP01.genome.fasta" % output_dir

run_create_sample_summary_job_fn = "%s/scripts/run_create_sample_summary_job.sh" % output_dir
submit_create_sample_summary_jobs_fn = "%s/scripts/submit_create_sample_summary_jobs.sh" % output_dir


# sites_annotation_pf60_fn = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161125_Pf60_final_vcfs/vcf/SNP_INDEL_WG.combined.filtered.annotation.vcf.gz'
hdf_fn = collections.OrderedDict()
hdf_fn['Pf60'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'
hdf_fn['Pv30'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161201_Pv_30_HDF5_build/hdf5/Pv_30.h5'


for release in GENOME_FN:
    get_ipython().system('cp {GENOME_FN[release]} {genome_fn[release]}')
    genome[release] = pyfasta.Fasta(genome_fn[release])
    print(sorted(genome[release].keys())[0])


hdf = collections.OrderedDict()
for release in hdf_fn:
    hdf[release] = h5py.File(hdf_fn[release], 'r')
    print(release, len(hdf[release]['samples']))
    


get_ipython().run_cell_magic('time', '', 'import pickle\nimport allel\ncalldata_subset = collections.OrderedDict()\nfor release in hdf_fn:\n    calldata_subset[release] = collections.OrderedDict()\n    for variable in [\'genotype\', \'GQ\', \'DP\', \'PGT\']:\n        calldata_subset[release][variable] = collections.OrderedDict()\n        calldata = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][variable])\n        \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first_pass.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_pass.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_all.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n            \n#             print(release, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n#             print(release, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n        pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))')


# %%time
# import pickle
# genotypes_subset_fn = "%s/genotypes_subset.p" % output_dir
# if os.path.exists(genotypes_subset_fn):
#     genotypes_subset = pickle.load(open(genotypes_subset_fn, "rb"))
# else:
#     genotypes = collections.OrderedDict()
#     genotypes_subset = collections.OrderedDict()
#     import allel
#     for release in hdf_fn:
#     # for release in ['Pv30']:
#         genotypes[release] = allel.GenotypeChunkedArray(hdf[release]['calldata']['genotype'])
#         genotypes_subset[release] = collections.OrderedDict()
#         print(release, 'first')
#         genotypes_subset[release]['first'] = genotypes[release].subset(
#             (hdf[release]['variants']['FILTER_PASS'][:]) &
#             (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
#         )
#         print(release, 'pass')
#         genotypes_subset[release]['pass'] = genotypes[release].subset(hdf[release]['variants']['FILTER_PASS'][:])
#         print(release, 'all')
#         genotypes_subset[release]['all'] = genotypes[release].subset()
#     pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))


get_ipython().run_cell_magic('time', '', 'import pickle\nGQ_subset_fn = "%s/GQ_subset.p" % output_dir\nif os.path.exists(GQ_subset_fn):\n    genotypes_subset = pickle.load(GQ_subset_fn)\nelse:\n    GQ = collections.OrderedDict()\n    GQ_subset = collections.OrderedDict()\n    for release in hdf_fn:\n        GQ[release] = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][\'GQ\'])\n        GQ_subset[release] = collections.OrderedDict()\n        print(release, \'first\')\n        GQ_subset[release][\'first\'] = GQ[release].subset(\n            (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n            (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n        )\n        print(release, \'pass\')\n        GQ_subset[release][\'pass\'] = GQ[release].subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n        print(release, \'all\')\n        GQ_subset[release][\'all\'] = GQ[release].subset()\n    pickle.dump(GQ_subset, open(GQ_subset_fn, "wb"))')


pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))


temp = (
    (hdf[release]['variants']['FILTER_PASS'][:]) &
    (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
)
pd.value_counts(temp)


sorted(genome[release].keys())[0].encode('ascii')


pd.value_counts(hdf[release]['variants']['CHROM'][:])


hdf[release]['samples'][:]


genotypes_subset['Pf60']


genotypes_subset['Pv30']


import allel
def create_sample_summary(hdf5_fn=hdf_fn['Pf60'], index=0, output_filestem="%s/sample_summaries/Pf60" % output_dir):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
    output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
    fo = open(output_fn, 'w')
    print(0)
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
#     genotypes = allel.GenotypeChunkedArray(hdf['calldata']['genotype'])
    print(1)
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    print(2)
    genotypes_pass = genotypes[is_pass]
    
    print(3)
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    print(4)
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    
    print(5)
    results['num_variants']      = genotypes.shape[0]
    results['num_pass_variants'] = np.count_nonzero(is_pass)
    results['num_missing']       = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']  = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']        = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']   = (results['num_pass_variants'] - results['num_pass_missing'])
    print(6)
    results['num_het']           = genotypes.count_het(axis=0)[0]
    results['num_pass_het']      = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']       = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']  = genotypes_pass.count_hom_alt(axis=0)[0]
    print(7)
    results['num_snp_hom_ref']   = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']       = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']   = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref'] = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']     = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt'] = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]
    print(8)    
    results['num_ins_hom_ref']   = genotypes_pass.subset(is_ins).count_hom_ref(axis=0)[0]
    results['num_ins_het']       = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins']           = (results['num_ins_hom_ref'] + results['num_ins_het'])
    results['num_del_hom_ref']   = genotypes_pass.subset(is_del).count_hom_ref(axis=0)[0]
    results['num_del_het']       = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del']           = (results['num_del_hom_ref'] + results['num_del_het'])
    
    print(9)
    results['pc_pass']           = results['num_pass_called'] / results['num_called']
    results['pc_missing']        = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']   = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']            = results['num_het'] / results['num_called']
    results['pc_pass_het']       = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']        = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']   = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']            = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_ins']            = (results['num_ins'] / (results['num_ins'] + results['num_del']))

    print(10)
    
    print('\t'.join([str(x) for x in list(results.keys())]), file=fo)
    print('\t'.join([str(x) for x in list(results.values())]), file=fo)
    fo.close()
    
    df_sample_summary = pd.DataFrame(
            {
                'Sample': pd.Series(results['sample_id']),
                'Variants called': pd.Series(results['num_called']),
                'Variants missing': pd.Series(results['num_missing']),
                'Proportion missing': pd.Series(results['pc_missing']),
                'Proportion pass missing': pd.Series(results['pc_pass_missing']),
                'Proportion heterozygous': pd.Series(results['pc_het']),
                'Proportion pass heterozygous': pd.Series(results['pc_pass_het']),
                'Proportion homozygous alternative': pd.Series(results['pc_hom_alt']),
                'Proportion pass homozygous alternative': pd.Series(results['pc_pass_hom_alt']),
                'Proportion variants SNPs': pd.Series(results['pc_snp']),
                'Proportion indels insertions': pd.Series(results['pc_ins']),
            }
        )  
    return(df_sample_summary, results)


hdf_fn['Pf60']


import allel
def create_sample_summary(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:][is_pass])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:][is_pass] >= 6.0)
    is_vhq_snp = (is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'SYNONYMOUS_CODING')
    is_frameshift = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'FRAME_SHIFT')
    is_inframe = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:][is_pass], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )[is_pass]
    
    is_pass_nonref = (is_pass & ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0)))
    is_biallelic_snp_nonref = (is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    is_biallelic_indel_nonref = (~is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    
    GQ = hdf['calldata']['GQ'][:, [index]][is_pass]
    DP = hdf['calldata']['DP'][:, [index]][is_pass]
    PGT = hdf['calldata']['PGT'][:, [index]][is_pass]
    
    mutations = np.char.add(hdf['variants']['REF'][:][is_pass][is_biallelic_snp_nonref], hdf['variants']['ALT'][:, 0][is_pass][is_biallelic_snp_nonref])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']         = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_het']                  = genotypes.count_het(axis=0)[0]
    results['num_pass_het']             = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']              = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']         = genotypes_pass.count_hom_alt(axis=0)[0]
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass_nonref)
    
    results['num_biallelic_het']        = genotypes_pass.subset(is_bi).count_het(axis=0)[0]
    results['num_biallelic_hom_alt']    = genotypes_pass.subset(is_bi).count_hom_alt(axis=0)[0]
    results['num_spanning_del_het']     = genotypes_pass.subset(is_sd).count_het(axis=0)[0]
    results['num_spanning_del_hom_alt'] = genotypes_pass.subset(is_sd).count_hom_alt(axis=0)[0]
    results['num_multiallelic_het']     = genotypes_pass.subset(is_mu).count_het(axis=0)[0]
    results['num_multiallelic_hom_alt'] = genotypes_pass.subset(is_mu).count_hom_alt(axis=0)[0]
    
    results['num_snp_hom_ref']          = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']              = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']          = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt']        = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]

    results['num_ins_het']              = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins_hom_alt']          = genotypes_pass.subset(is_ins).count_hom_alt(axis=0)[0]
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del_hom_alt']          = genotypes_pass.subset(is_del).count_hom_alt(axis=0)[0]
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = genotypes_pass.subset(is_coding).count_het(axis=0)[0]
    results['num_coding_hom_alt']       = genotypes_pass.subset(is_coding).count_hom_alt(axis=0)[0]
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_vhq_snp_hom_ref']      = genotypes_pass.subset(is_vhq_snp).count_hom_ref(axis=0)[0]
    results['num_vhq_snp_het']          = genotypes_pass.subset(is_vhq_snp).count_het(axis=0)[0]
    results['num_vhq_snp_hom_alt']      = genotypes_pass.subset(is_vhq_snp).count_hom_alt(axis=0)[0]
    
    results['num_singleton']            = np.count_nonzero(is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_bi & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_biallelic_snp_nonref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_biallelic_snp_nonref & is_synonymous)
    results['num_bi_frameshift']        = np.count_nonzero(is_biallelic_indel_nonref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_biallelic_indel_nonref & is_inframe)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['pc_pass']                  = results['num_pass_called'] / results['num_called']
    results['pc_missing']               = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = results['num_het'] / results['num_called']
    results['pc_pass_het']              = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']               = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_bi_frameshift']         = results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_bi_transition']         = results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    
    results['mean_GQ']                  = np.mean(GQ)
    results['mean_GQ_2']                = np.nanmean(GQ)
    results['mean_DP']                  = np.mean(DP)
    results['mean_DP_2']                = np.nanmean(DP)
    
    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

    return(results, PGT)


import allel
def create_sample_summary_2(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
#     genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    indel_len = svlen1
    het_indels = (svlen1 != svlen2)
    indel_len[het_indels] = svlen1[het_indels] + svlen2[het_indels]
    
    is_indel = (indel_len != 0)
    is_inframe = ((indel_len != 0) & (indel_len%3 == 0))
    is_frameshift = ((indel_len != 0) & (indel_len%3 != 0))
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))
    is_del = ((svlen1 < 0) | (svlen2 < 0))
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0))) # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:] >= 6.0)
    is_hq_snp = (is_pass & is_snp & is_bi & is_coding)
    is_vhq_snp = (is_pass & is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'SYNONYMOUS_CODING')
    is_frameshift_snpeff = (hdf['variants']['SNPEFF_EFFECT'][:] == b'FRAME_SHIFT')
    is_inframe_snpeff = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )
    
    is_hom_ref = ((genotypes[:, 0, 0] == 0) & (genotypes[:, 0, 1] == 0))
    is_het = ((genotypes[:, 0, 0] != genotypes[:, 0, 1]))
    is_hom_alt = ((genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 0] == genotypes[:, 0, 1]))
    is_non_ref = ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0))
    is_missing = ((genotypes[:, 0, 0] == -1))
    is_called = ((genotypes[:, 0, 0] >= 0))
    
    GQ = hdf['calldata']['GQ'][:, index]
    is_GQ_30 = (GQ >= 30)
    is_GQ_99 = (GQ >= 99)
    DP = hdf['calldata']['DP'][:, index]
    PGT = hdf['calldata']['PGT'][:, index]
    is_phased = np.in1d(PGT, [b'.', b''], invert=True)
    
    mutations = np.char.add(hdf['variants']['REF'][:][(is_pass & is_snp & is_bi & is_non_ref)], hdf['variants']['ALT'][:, 0][(is_pass & is_snp & is_bi & is_non_ref)])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = np.count_nonzero(is_missing)
    results['num_pass_missing']         = np.count_nonzero(is_pass & is_missing)
    results['num_called']               = np.count_nonzero(~is_missing)
#     results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = np.count_nonzero(is_pass & is_called)
#     results['num_pass_called_2']        = np.count_nonzero(is_pass & ~is_missing)
#     results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_hom_ref']              = np.count_nonzero(is_hom_ref)
    results['num_het']                  = np.count_nonzero(is_het)
    results['num_pass_het']             = np.count_nonzero(is_pass & is_het)
    results['num_hom_alt']              = np.count_nonzero(is_hom_alt)
    results['num_pass_hom_alt']         = np.count_nonzero(is_pass & is_hom_alt)
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass & is_non_ref)
#     results['num_variants_2']           = results['num_hom_ref'] + results['num_het'] + results['num_hom_alt'] + results['num_missing']
    
    results['num_biallelic_het']        = np.count_nonzero(is_pass & is_bi & is_het)
    results['num_biallelic_hom_alt']    = np.count_nonzero(is_pass & is_bi & is_hom_alt)
    results['num_spanning_del_het']     = np.count_nonzero(is_pass & is_sd & is_het)
    results['num_spanning_del_hom_alt'] = np.count_nonzero(is_pass & is_sd & is_hom_alt)
    results['num_multiallelic_het']     = np.count_nonzero(is_pass & is_mu & is_het)
    results['num_multiallelic_hom_alt'] = np.count_nonzero(is_pass & is_mu & is_hom_alt)
    
#     results['num_snp_hom_ref']          = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_het']              = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_hom_alt']          = np.count_nonzero(is_pass & is_snp & is_hom_alt)
    results['num_snp']                  = (results['num_snp_het'] + results['num_snp_hom_alt'])
#     results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel_hom_alt']        = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel']                  = (results['num_indel_het'] + results['num_indel_hom_alt'])

    results['num_ins_het']              = np.count_nonzero(is_pass & is_ins & is_het)
    results['num_ins_hom_alt']          = np.count_nonzero(is_pass & is_ins & is_hom_alt)
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = np.count_nonzero(is_pass & is_del & is_het)
    results['num_del_hom_alt']          = np.count_nonzero(is_pass & is_del & is_hom_alt)
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = np.count_nonzero(is_pass & is_coding & is_het)
    results['num_coding_hom_alt']       = np.count_nonzero(is_pass & is_coding & is_hom_alt)
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_hq_snp_called']        = np.count_nonzero(is_hq_snp & ~is_missing)
    results['num_hq_snp_hom_ref']       = np.count_nonzero(is_hq_snp & is_hom_ref)
    results['num_hq_snp_het']           = np.count_nonzero(is_hq_snp & is_het)
    results['num_hq_snp_hom_alt']       = np.count_nonzero(is_hq_snp & is_hom_alt)
    results['num_vhq_snp_called']       = np.count_nonzero(is_vhq_snp & ~is_missing)
    results['num_vhq_snp_hom_ref']      = np.count_nonzero(is_vhq_snp & is_hom_ref)
    results['num_vhq_snp_het']          = np.count_nonzero(is_vhq_snp & is_het)
    results['num_vhq_snp_hom_alt']      = np.count_nonzero(is_vhq_snp & is_hom_alt)
    
    results['num_singleton']            = np.count_nonzero(is_pass & is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_pass & is_bi & is_singleton)
    results['num_hq_snp_singleton']     = np.count_nonzero(is_hq_snp & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_synonymous)
#     results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_frameshift)
#     results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_inframe)
    results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_coding & is_frameshift)
    results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_coding & is_inframe)
    results['num_bi_frameshift']        = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_hq_frameshift']        = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_hq_inframe']           = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_bi_frameshift_snpeff'] = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_frameshift_snpeff)
    results['num_bi_inframe_snpeff']    = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_inframe_snpeff)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['num_phased']               = np.count_nonzero(is_pass & is_phased)
    results['num_phased_non_ref']       = np.count_nonzero(is_pass & is_phased & is_non_ref)
    results['num_phased_hom_ref']       = np.count_nonzero(is_pass & is_phased & is_hom_ref)
    results['num_phased_missing']       = np.count_nonzero(is_pass & is_phased & is_missing)
    
    results['num_GQ_30']                = np.count_nonzero(is_pass & is_called & is_GQ_30)
    results['num_het_GQ_30']            = np.count_nonzero(is_pass & is_het & is_GQ_30)
    results['num_hom_alt_GQ_30']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_30)
    results['num_GQ_99']                = np.count_nonzero(is_pass & is_called & is_GQ_99)
    results['num_het_GQ_99']            = np.count_nonzero(is_pass & is_het & is_GQ_99)
    results['num_hom_alt_GQ_99']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_99)

    results['pc_pass']                  = 0.0 if results['num_called'] == 0 else         results['num_pass_called'] / results['num_called']
    results['pc_missing']               = 0.0 if results['num_variants'] == 0 else         results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = 0.0 if results['num_pass_variants'] == 0 else         results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = 0.0 if results['num_called'] == 0 else         results['num_het'] / results['num_called']
    results['pc_pass_het']              = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_het'] / results['num_pass_called']
    results['pc_hq_snp_het']            = 0.0 if results['num_hq_snp_called'] == 0 else         results['num_hq_snp_het'] / results['num_hq_snp_called']
    results['pc_vhq_snp_het']           = 0.0 if results['num_vhq_snp_called'] == 0 else         results['num_vhq_snp_het'] / results['num_vhq_snp_called']
    results['pc_hom_alt']               = 0.0 if results['num_called'] == 0 else         results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = 0.0 if (results['num_ins'] + results['num_del']) == 0 else         (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = 0.0 if (results['num_bi_nonsynonymous'] + results['num_bi_synonymous']) == 0 else         results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_frameshift']            = 0.0 if (results['num_frameshift'] + results['num_inframe']) == 0 else         results['num_frameshift'] / (results['num_frameshift'] + results['num_inframe'])
    results['pc_bi_frameshift']         = 0.0 if (results['num_bi_frameshift'] + results['num_bi_inframe']) == 0 else         results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_hq_frameshift']         = 0.0 if (results['num_hq_frameshift'] + results['num_hq_inframe']) == 0 else         results['num_hq_frameshift'] / (results['num_hq_frameshift'] + results['num_hq_inframe'])
    results['pc_bi_frameshift_snpeff']  = 0.0 if (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff']) == 0 else         results['num_bi_frameshift_snpeff'] / (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff'])
    results['pc_bi_transition']         = 0.0 if (results['num_bi_transition'] + results['num_bi_transversion']) == 0 else         results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_phased']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_phased_non_ref'] / results['num_pass_non_ref']
    results['pc_phased_hom_ref']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_hom_ref'] / results['num_phased']
    results['pc_phased_missing']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_missing'] / results['num_phased']
    results['pc_GQ_30']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_30'] / results['num_pass_called']
    results['pc_het_GQ_30']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_30'] / results['num_pass_het']
    results['pc_hom_alt_GQ_30']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_30'] / results['num_pass_hom_alt']
    results['pc_GQ_99']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_99'] / results['num_pass_called']
    results['pc_het_GQ_99']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_99'] / results['num_pass_het']
    results['pc_hom_alt_GQ_99']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_99'] / results['num_pass_hom_alt']
     
    results['mean_GQ']                  = np.mean(GQ[is_pass])
    results['mean_GQ_hom_ref']          = np.mean(GQ[is_pass & is_hom_ref])
    results['mean_GQ_het']              = np.mean(GQ[is_pass & is_het])
    results['mean_GQ_hom_alt']          = np.mean(GQ[is_pass & is_hom_alt])
    results['mean_DP']                  = np.mean(DP[is_pass])
    results['mean_DP_hom_ref']          = np.mean(DP[is_pass & is_hom_ref])
    results['mean_DP_het']              = np.mean(DP[is_pass & is_het])
    results['mean_DP_hom_alt']          = np.mean(DP[is_pass & is_hom_alt])
#     results['mean_GQ_2']                = np.nanmean(GQ[is_pass])
#     results['mean_DP_2']                = np.nanmean(DP[is_pass])
#     results['mean_DP']                  = np.mean(DP)
#     results['mean_DP_2']                = np.nanmean(DP)

    results['mean_indel_len']           = np.mean(indel_len[is_pass])
    results['total_indel_len']          = np.sum(indel_len[is_pass])

    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

#     return(results, is_pass, is_phased, is_non_ref, is_hom_ref, is_missing)
#     return(results, svlen, svlen1, svlen2, indel_len, is_indel, is_inframe, is_frameshift, is_pass, is_bi, is_non_ref, is_frameshift_snpeff, is_inframe_snpeff, is_coding, is_vqslod6)
#     return(results, is_pass, is_called, is_GQ_30)
    return(results)


# - Create sample level summaries of genotypes
#     - #Pass/Fail SNP/INDEL BI/MU/SD het/hom
#     - Pass/Fail
#     - SNP/INDEL
#     - INS/DEL
#     - Coding/non-coding
#     - singletons
#     - singleton mean VQSLOD
#     - het/hom
#     - NS/S
#     - mean GQ
#     - mean DP
#     - %phased
#     - num_pass_biallelic_coding_snp_het, num_VQSLOD6_biallelic_coding_snp_het
#     - mean indel size, total_indel_size
#     - %coding indel mod3
#     - #transitions/transversions, Ti/Tv
#     

results = create_sample_summary_2(index=275)


results, is_pass, is_called, is_GQ_30 = create_sample_summary_2()


print('pc_GQ_30', results['pc_GQ_30'])
print('pc_het_GQ_30', results['pc_het_GQ_30'])
print('pc_hom_alt_GQ_30', results['pc_hom_alt_GQ_30'])
print('pc_GQ_99', results['pc_GQ_99'])
print('pc_het_GQ_99', results['pc_het_GQ_99'])
print('pc_hom_alt_GQ_99', results['pc_hom_alt_GQ_99'])


results = create_sample_summary_2()


print('num_frameshift', results['num_frameshift'])
print('num_inframe', results['num_inframe'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('num_hq_frameshift', results['num_bi_frameshift'])
print('num_hq_inframe', results['num_bi_inframe'])
print('num_bi_frameshift_snpeff', results['num_bi_frameshift_snpeff'])
print('num_bi_inframe_snpeff', results['num_bi_inframe_snpeff'])
print()
print('pc_frameshift', results['pc_frameshift'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('pc_hq_frameshift', results['pc_bi_frameshift'])
print('pc_bi_frameshift_snpeff', results['pc_bi_frameshift_snpeff'])


np.count_nonzero(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)


np.where(is_pass & is_vqslod6 & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]


np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]


np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]


np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]
)


np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]
)


index=140202
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['CDS'][index])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=61186
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


is_phased_hom_ref = (is_pass & is_phased & is_hom_ref)
is_phased_missing = (is_pass & is_phased & is_missing)
print(np.count_nonzero(is_phased_hom_ref))
print(np.count_nonzero(is_phased_missing))


np.where(is_phased_hom_ref)


index=71926
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=79318
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=79318
print(hdf['Pf60']['variants']['REF'][71926])
print(hdf['Pf60']['variants']['ALT'][71926, :])
print(hdf['Pf60']['calldata']['genotype'][71926, 0, :])
print(hdf['Pf60']['calldata']['AD'][71926, 0, :])
print(hdf['Pf60']['calldata']['PL'][71926, 0, :])
print(hdf['Pf60']['calldata']['PID'][71926, 0])
print(hdf['Pf60']['calldata']['PGT'][71926, 0])


hdf['Pf60']['variants']['ALT'][:][71926, :]


hdf['Pf60']['calldata']['genotype'][:, 0, :][71926, :]


hdf['Pf60']['calldata']['genotype'][71926, 0, :]


hdf['Pf60']['calldata']['AD'][71926, 0, :]


hdf['Pf60']['calldata']['PL'][71926, 0, :]


hdf['Pf60']['calldata']['PID'][71926, 0]


hdf['Pf60']['calldata']['PGT'][71926, 0]


np.where(is_phased_missing)


pd.value_counts(PGT[:,0])


results, mutations = create_sample_summary()


is_pass = hdf['Pf60']['variants']['FILTER_PASS'][:]
is_snp = (hdf['Pf60']['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
is_bi = (hdf['Pf60']['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
temp = hdf['Pf60']['variants']['REF'][:][is_pass][is_snp]


mutations = np.char.add(
    hdf['Pf60']['variants']['REF'][:][is_pass][(is_snp & is_bi)],
    hdf['Pf60']['variants']['ALT'][:, 0][is_pass][(is_snp & is_bi)]
)


pd.value_counts(mutations)


pd.value_counts(mutations)


list(hdf['Pf60'].keys())


pd.value_counts(hdf['Pf60']['variants']['AC'][:,0] == 0)


pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:])


7182/20


pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:][(
            hdf['Pf60']['variants']['FILTER_PASS'][:] &
            (hdf['Pf60']['variants']['MULTIALLELIC'][:] == b'BI') &
            (hdf['Pf60']['variants']['AC'][:, 0] > 359) &
            (hdf['Pf60']['variants']['AC'][:, 0] < (7182-359))
        )])


print('num_pass_non_ref', results['num_pass_non_ref'])
print('num_pass_non_ref_2', results['num_pass_non_ref_2'])


print('pc_bi_transition', results['pc_bi_transition'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('pc_biallelic', results['pc_biallelic'])
print('pc_spanning_del', results['pc_spanning_del'])
print('pc_mutliallelic', results['pc_mutliallelic'])
print('pc_bi_nonsynonymous', results['pc_bi_nonsynonymous'])
print('pc_bi_nonsynonymous_2', results['pc_bi_nonsynonymous_2'])


pd.value_counts(mutations)


print('num_snp_hom_ref', results['num_snp_hom_ref'])
print('num_snp_het', results['num_snp_het'])
print('num_snp_hom_alt', results['num_snp_hom_alt'])
print('num_indel_hom_ref', results['num_indel_hom_ref'])
print('num_indel_het', results['num_indel_het'])
print('num_indel_hom_alt', results['num_indel_hom_alt'])
print()
print('pc_snp', results['pc_snp'])
# print('pc_snp_v2', results['pc_snp_v2'])


create_sample_summary()


get_ipython().system('/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\\ 6.0/scripts/create_sample_summary.py')


fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX

''', file=fo)
fo.close()


fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
RELEASE=$2
# LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX
OUTPUT_FN=%s/sample_summaries/$RELEASE/results_$INDEX.txt

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX > $OUTPUT_FN

''' % (output_dir), file=fo)
fo.close()


get_ipython().system('LSB_JOBINDEX=2 && bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')


MEMORY=8000
# Kick off Pf 6.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-7182]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')


MEMORY=8000
# Kick off Pv 3.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-1001]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} {hdf_fn[\'Pv30\']} Pv30')


get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pf60/results_0.txt &   cat {output_dir}/sample_summaries/Pf60/results_*.txt | grep -v '^sample_id') > {output_dir}/pf_60_summaries.txt")


get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pv30/results_0.txt &   cat {output_dir}/sample_summaries/Pv30/results_*.txt | grep -v '^sample_id') > {output_dir}/pv_30_summaries.txt")


output_dir





create_sample_summary(1)


df_sample_summary, results = create_sample_summary()


df_sample_summary, results = create_sample_summary(index=100)


b'FP0008-C'.decode('ascii')


'\t'.join([str(x) for x in list(results.values())])


'\t'.join([str(x) for x in list(results.values())])


'\t'.join(list(results.values()))


list(results.values())


list(results.values())[0]


hdf = h5py.File(hdf_fn['Pv30'], 'r')
hdf['calldata']['genotype'].shape


svlen = hdf['variants']['svlen'][:]
svlen


genotype = hdf['calldata']['genotype'][:, 0, :]
genotype


pd.value_counts(genotype[:,0])


print(genotype.shape)
print(svlen.shape)


svlen1 = svlen[np.arange(svlen.shape[0]), genotype[:, 0] - 1]
svlen1[np.in1d(genotype[:, 0], [-1, 0])] = 0


pd.Series(svlen1).describe()


pd.value_counts(svlen1)


svlen2 = svlen[genotype]


svlen


genotype[:,0]


svlen[:, genotype[:,0]]


np.take(svlen[0:100000], genotype[0:100000,0]-1, axis=1)


alt_indexes = genotype[:, 0] - 1
alt_indexes[alt_indexes < 0] = 0


pd.value_counts(alt_indexes)


np.take(svlen[0:10000], alt_indexes[0:10000], axis=0).shape


svlen[0:1002]


alt_indexes[0:1002]


np.take(svlen[0:1002], alt_indexes[0:1002], axis=0)


svlen[0:1002][np.arange(1002), alt_indexes[0:1002]].shape


alt_indexes[0:10000].shape


svlen[0:10000].shape


svlen


print(svlen2.shape)
svlen2


temp = hdf['calldata']['genotype'][:, [0], :]


temp2=allel.GenotypeArray(temp)
temp2


temp.shape


get_ipython().run_cell_magic('time', '', "df_sample_summary = collections.OrderedDict()\n# for release in genotypes_subset:\nfor release in ['Pv30', 'Pf60']:\n    print(release)\n    samples = hdf[release]['samples'][:]\n    pass_variants = hdf[release]['variants']['FILTER_PASS'][:]\n    \n    print(0)\n    is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')\n    is_bi = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'BI')\n    is_sd = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'SD')\n    is_mu = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'MU')\n    is_ins = (hdf[release]['variants']['svlen'][:][pass_variants] > 0)\n    is_del = (hdf[release]['variants']['svlen'][:][pass_variants] < 0)\n    \n    print(1)\n    num_variants = genotypes_subset[release]['all'].shape[0]\n    num_pass_variants = genotypes_subset[release]['pass'].shape[0]\n    num_missing = genotypes_subset[release]['all'].count_missing(axis=0)[:]\n    num_pass_missing = genotypes_subset[release]['pass'].count_missing(axis=0)[:]\n    num_called = (num_variants - num_missing)\n    num_pass_called = (num_pass_variants - num_pass_missing)\n    print(2)\n    num_het = genotypes_subset[release]['all'].count_het(axis=0)[:]\n    num_pass_het = genotypes_subset[release]['pass'].count_het(axis=0)[:]\n    num_hom_alt = genotypes_subset[release]['all'].count_hom_alt(axis=0)[:]\n    num_pass_hom_alt = genotypes_subset[release]['pass'].count_hom_alt(axis=0)[:]\n    print(3)\n    num_snp_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_snp_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_snp_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    num_indel_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_indel_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_indel_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    print(4)    \n    num_ins_hom_ref = genotypes_subset[release]['pass'].subset(is_ins).count_hom_ref(axis=0)[:]\n    num_ins_het = genotypes_subset[release]['pass'].subset(is_ins).count_het(axis=0)[:]\n    num_ins = (num_ins_hom_ref + num_ins_het)\n    num_del_hom_ref = genotypes_subset[release]['pass'].subset(is_del).count_hom_ref(axis=0)[:]\n    num_del_het = genotypes_subset[release]['pass'].subset(is_del).count_het(axis=0)[:]\n    num_del = (num_del_hom_ref + num_del_het)\n    \n    print(5)\n    pc_pass = num_pass_called / num_called\n    pc_missing = num_missing / num_variants\n    pc_pass_missing = num_pass_missing / num_pass_variants\n    pc_het = num_het / num_called\n    pc_pass_het = num_pass_het / num_pass_called\n    pc_hom_alt = num_hom_alt / num_called\n    pc_pass_hom_alt = num_pass_hom_alt / num_pass_called\n    pc_snp = (num_snp_het + num_snp_homalt) / (num_snp_het + num_snp_homalt + num_indel_het + num_indel_homalt)\n    pc_ins = (num_ins / (num_ins + num_del))\n\n    print(6)\n    df_sample_summary[release] = pd.DataFrame(\n            {\n                'Sample': pd.Series(samples),\n                'Variants called': pd.Series(num_called),\n                'Variants missing': pd.Series(num_called),\n                'Proportion missing': pd.Series(pc_missing),\n                'Proportion pass missing': pd.Series(pc_pass_missing),\n                'Proportion heterozygous': pd.Series(pc_het),\n                'Proportion pass heterozygous': pd.Series(pc_pass_het),\n                'Proportion homozygous alternative': pd.Series(pc_hom_alt),\n                'Proportion pass homozygous alternative': pd.Series(pc_pass_hom_alt),\n                'Proportion variants SNPs': pd.Series(pc_snp),\n                'Proportion indels insertions': pd.Series(pc_ins),\n            }\n        )")


is_ins


num_snp_hom_ref = genotypes_subset['Pv30']['pass'][is_snp, :, :].count_hom_ref(axis=0)[:]


genotypes_subset['Pv30']['pass'].subset(is_snp)


pd.value_counts(is_snp)


len(is_snp)


is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')


2+2


df_sample_summary['Pv30']


df_sample_summary['Pf60']





get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf 6.0/20161110_K13_double_mutants"
get_ipython().system('mkdir -p {output_dir}')


def create_variants_npy(vcf_fn='/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz',
                        region='Pf3D7_13_v3:1724817-1726997'):
    cache_dir = '%s/vcfnp_cache' % output_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    variants = vcfnp.variants(
        vcf_fn,
        region,
#         fields=['CHROM', 'POS', 'REF', 'ALT',
#                 'AC', 'AN', 'FILTER', 'VQSLOD'],
        dtypes={
            'REF':                      'a200',
            'ALT':                      'a200',
        },
#         arities={
#             'ALT':   1,
#             'AC':    1,
#         },
        flatten_filter=True,
        progress=100,
        verbose=True,
        cache=True,
        cachedir=cache_dir
    )
    return(variants)

def create_calldata_npy(vcf_fn='/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz',
                        region='Pf3D7_13_v3:1724817-1726997'):
    cache_dir = '%s/vcfnp_cache' % output_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    calldata = vcfnp.calldata_2d(
        vcf_fn,
        region,
        fields=['GT', 'AD', 'GQ', 'PGT', 'PID'],
#         dtypes={
#             'AD': 'u2',
#         },
#         arities={
#             'AD': max_alleles,
#         },
        progress=100,
        verbose=True,
        cache=True,
        cachedir=cache_dir
    )
    return(calldata)


variants = create_variants_npy()
calldata = create_calldata_npy()


calldata['GT'].shape


def num_nonref(gt):
    return(np.count_nonzero(np.logical_not(np.in1d(gt, [b'0/0', b'./.']))))
    


def num_nonref_hom(gt):
    return(np.count_nonzero(np.logical_not(np.in1d(gt, [b'0/0', b'./.', b'0/1', b'0/2', b'1/2']))))
    


nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'])


len(nonref_per_sample)


variant_set = collections.OrderedDict()
variant_set['all'] = (variants['CHROM'] == b'Pf3D7_13_v3')
variant_set['pass'] = (variants['FILTER_PASS'])
variant_set['non_synonymous'] = (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO')
variant_set['non_synonymous pass'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['FILTER_PASS']) )
variant_set['BTBPOZ_propeller'] = (variants['POS'] <= 1725953) # from amino acid 349
variant_set['non_synonymous BTBPOZ_propeller'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['POS'] <= 1725953) )
variant_set['non_synonymous pass BTBPOZ_propeller'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['FILTER_PASS']) & (variants['POS'] <= 1725953) )


for set_name in variant_set:
    print(set_name, np.count_nonzero(variant_set[set_name]))
    nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'][variant_set[set_name], :])
    print(np.unique(nonref_per_sample, return_counts=True))
    print()


for set_name in variant_set:
    print(set_name, np.count_nonzero(variant_set[set_name]))
    nonref_hom_per_sample = np.apply_along_axis(num_nonref_hom, 0, calldata['GT'][variant_set[set_name], :])
    print(np.unique(nonref_hom_per_sample, return_counts=True))
    print()


nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :])


vcf_fn = '/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz'
samples = np.array(vcf.Reader(filename=vcf_fn).samples)
samples


print(list(samples[nonref_per_sample>2]))
nonref_per_sample[nonref_per_sample>2]





nonref_per_variant_dodgy = np.apply_along_axis(
    lambda x: np.count_nonzero(np.logical_not(np.in1d(x, [b'0/0', b'./.']))),
    1,
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]
)
nonref_per_variant_dodgy


nonref_per_variant_all = np.apply_along_axis(
    lambda x: np.count_nonzero(np.logical_not(np.in1d(x, [b'0/0', b'./.']))),
    1,
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :]
)
nonref_per_variant_all


print(nonref_per_variant_dodgy[nonref_per_variant_dodgy >= 4])
print(nonref_per_variant_all[nonref_per_variant_dodgy >= 4])


print(list(samples[nonref_per_sample==2]))


calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PC0172-C'][:, 0].transpose()


get_ipython().system('grep PC0172 ~/pf_60_mergelanes.txt')


# This one has 3 K13 mutants, but Olivo has down as Pf only
get_ipython().system('grep PH0714 ~/pf_60_mergelanes.txt')


# This one has 9 K13 mutants, but Olivo has down as Pf/Pm/Po - can we see any Po reads?
get_ipython().system('grep QP0097 ~/pf_60_mergelanes.txt')


# This one has 3 K13 mutants, but Olivo has down as Pf/Pv - can we see any Pv reads?
get_ipython().system('grep PH0914 ~/pf_60_mergelanes.txt')


# This one has 4 K13 mutants, but Olivo has down as Pf/Pv - can we see any Pv reads?
get_ipython().system('grep QC0280 ~/pf_60_mergelanes.txt')


# This one looks like a mixture of two different K13 mutants in proportions ~1:2
calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PD0464-C'][:, 0].transpose()


for sample in samples[nonref_per_sample==2]:
    print(sample)
    print(calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples==sample][:, 0].transpose())
    print()


np.unique(nonref_per_sample, return_counts=True)


calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]


calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0190-C'][:, 0]


calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PF0345-C'][:, 0]


calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0]


calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0]


calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0][
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0] == b'1/1'
]


calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0][
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0] == b'1/1'
]


calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0].transpose()


calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PF0345-C'][:, 0]


calldata['AD'].shape


calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0190-C'][:, 0]


calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]


get_ipython().system('grep PA0271 ~/pf_60_mergelanes.txt')
get_ipython().system('grep PF0345 ~/pf_60_mergelanes.txt')
# Then copied these over to macbook to look at in IGV


get_ipython().system('grep PA0271 ~/pf_60_speciator.txt')


get_ipython().system('grep PF0345 ~/pf_60_speciator.txt')


# Looks like PA0271-C isn't P. falciparum
get_ipython().system('cat /lustre/scratch116/malaria/pfalciparum/output/9/7/3/0/48138/1_speciator/6936_2_nonhuman_4.pe.markdup.recal.speciator.tab')


# Looks like PF0345-C is P. falciparum
get_ipython().system('cat /lustre/scratch116/malaria/pfalciparum/output/f/3/b/0/47447/1_speciator/8516_5_29.pe.markdup.recal.speciator.tab')


# # P. malariae/P./ vivax mixed infections
# PA0271-C reads blast very well to P. malariae. Looking at K13 in PF0345-C shows a subset of reads with many variants. Blasting the sequences from a few of these reads shows perfect or almost perfect matches to p. malariae K13 - hence this looks liked a mixed infection. This is evidence that mixed infections cause many heterozygous SNPs. The questions are how to identify and what to do about such samples. Some initial ideas:
# 
# How to identify mixed p. malariae (or other human malaria) infections:
# 1. Rerun specicator including malariae and ovale
# 2. Use Olivo/Chris's speciation on sequencing data?
# 3. Identify samples with excess heterozygosity (how extreme are malariae mixed infections?)
# 
# What to do about mixed infections
# 1. Remove from analyses
# 2. Attempt to classify each read by species, and run separately on each species
# 3. Map all reads to a combination of all reference genomes
# 
# Note that these findings also have implications for P. vivax analyses, given that we have a reasonable number of known Pf/Pv mixtures. In fact PH0914-Cx and QC0280-C are both Pf/Pv samples, and both have excess K13 hets (though slightly less dramatic than Pf/Pm mixtures)
# 
# Given the way bwa mem works, it is likely that many reads could map to both species. It will be interesting to see if mixed infections have excess heterozygosity in both species.
# 




# See emails from:
# 
# Jim 15/06/2016 14:19
# 
# Cinzia 22/03/2016 07:47
# 
# Roberto 14/06/2016 15:17

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_context('paper')
sns.set_style('white')
sns.set_style('ticks')


output_dir = '/nfs/team112_internal/rp7/data/Pf/hrp'
get_ipython().system('mkdir -p {output_dir}/fofns')
get_ipython().system('mkdir -p {output_dir}/metadata')
get_ipython().system('mkdir -p {output_dir}/scripts')

cinzia_metadata_fn = '%s/metadata/PF_metadata_base.csv' % output_dir # From Cinzia 22/03/2016 07:47
v4_metadata_fn = '%s/metadata/PGV4_mk5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
v5_metadata_fn = '%s/metadata/v5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
iso_country_code_fn = '%s/metadata/country-codes.csv' % output_dir # https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv
regions_in_dataset_fn = '%s/metadata/regions_in_dataset.xlsx' % output_dir
sub_contintent_fn = '%s/metadata/region_sub_continents.xlsx' % output_dir
manifest_fn = '%s/metadata/hrp_manifest_20160621.txt' % output_dir
# jim_manifest_fn = '%s/metadata/manifest_for_jim_20160620.txt' % output_dir
lookseq_fn = '%s/metadata/lookseq.txt' % output_dir
vrpipe_metadata_script_fn = "%s/scripts/vrpipe_metadata.sh" % output_dir
vrpipe_metadata_fn = "%s/metadata/vrpipe_metadata_20160621.txt" % output_dir

lab_studies = list(range(1032, 1044, 1)) + [1104, 1133, 1150, 1153]

# cinzia_extra_metadata_fn = '/nfs/team112_internal/rp7/data/Pf/4_0/meta/PF_extrametadata.csv' # From Cinzia 22/03/2016 08:22


fofns = collections.OrderedDict()

fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '%s/fofns/olivo_TRAC.fofn' % output_dir
fofns['fanello'] = '%s/fofns/olivo_fanello.fofn' % output_dir


import glob

bam_dirs = collections.OrderedDict()
bam_dirs['trac'] = '/nfs/team112_internal/production_files/Pf/olivo_TRAC/remapped'
bam_dirs['fanello'] = '/nfs/team112_internal/production_files/Pf/olivo_fanello'

for bam_dir in bam_dirs:
    get_ipython().system('rm {fofns[bam_dir]}')
    with open(fofns[bam_dir], "a") as fofn:
# glob.glob('%s/*.bam' % fofns['trac'])
        print("path\tsample", file=fofn)
        for x in glob.glob('%s/*.bam' % bam_dirs[bam_dir]):
            print("%s\t%s" % (x, os.path.basename(x).replace('_', '-').replace('.bam', '')), file=fofn)
# [os.path.basename(x) for x in glob.glob('%s/*.bam' % fofns['trac'])]


for i, fofn in enumerate(fofns):
    if i == 0:
        tbl_all_bams = (
            etl.fromtsv(fofns[fofn])
            .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
            .rename('bases_of_5X_coverage', 'orig_bases_of_5X_coverage')
            .rename('mean_coverage', 'orig_mean_coverage')
            .addfield('dataset', fofn)
        )
    else:
        if fofn == 'pf3k_pilot_5_0_broad':
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .selecteq('study', 'Pf3k_Senegal')
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .rename('bases_of_5X_coverage', 'orig_bases_of_5X_coverage')
                    .rename('mean_coverage', 'orig_mean_coverage')
                    .addfield('dataset', fofn)
                )
            )
        elif fofn in ['pf_community_5_0', 'conway']:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .rename('bases_of_5X_coverage', 'orig_bases_of_5X_coverage')
                    .rename('mean_coverage', 'orig_mean_coverage')
                    .addfield('dataset', fofn)
                )
            )
        else:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample'])
                    .addfield('dataset', fofn)
                )
            )
        


len(tbl_all_bams.data())


fo = open(vrpipe_metadata_script_fn, "w")
print("vrpipe-fileinfo \\", file=fo)
fo.close()
fo = open(vrpipe_metadata_script_fn, "a")
for path in tbl_all_bams.values('path'):
    print("--path %s \\" % path, file=fo)
print("--metadata sample,bases_of_1X_coverage,bases_of_5X_coverage,bases_of_10X_coverage,bases_of_20X_coverage,mean_coverage \\", file=fo)
print("--display tab > %s" % vrpipe_metadata_fn, file=fo)
fo.close()

# I then manually ran this script from the command line


len(tbl_all_bams.duplicates('sample').data())


len(tbl_all_bams.unique('sample').data())


tbl_all_bams


tbl_all_bams.valuecounts('dataset').displayall()


tbl_solaris = (
    etl.fromcsv(cinzia_metadata_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'country_code', 'oxford_src_code', 'oxford_donor_code', 'alfresco_code'])
    .convert('alfresco_code', int)
    .rename('country', 'solaris_country')
    .rename('country_code', 'solaris_country_code')
    .distinct('oxford_code')
)
tbl_solaris.selectne('alfresco_code', None)


tbl_solaris.duplicates('oxford_code').displayall()


tbl_v4_metadata = etl.fromxlsx(v4_metadata_fn, 'PGV4.0').cut(['Sample', 'Region']).rename('Region', 'v4_region')
tbl_v5_metadata = etl.fromxlsx(v5_metadata_fn).cut(['Sample', 'Region']).rename('Region', 'v5_region')


tbl_v4_metadata


tbl_v4_metadata.selecteq('Sample', 'PF0542-C')


tbl_v5_metadata


tbl_v5_metadata.selecteq('Sample', 'PF0542-C')


def determine_region(rec, null_vals=(None, 'NULL', '-')):
#     if (
#         rec['sample'].startswith('PG') or
#         rec['sample'].startswith('PL') or
#         rec['sample'].startswith('PF') or
#         rec['sample'].startswith('WL') or
#         rec['sample'].startswith('WH') or
#         rec['sample'].startswith('WS')
#     ):
    if rec['alfresco_code'] in lab_studies:
        return('Lab')
    if rec['v5_region'] not in null_vals:
        return(rec['v5_region'])
    elif rec['v4_region'] not in null_vals:
        return(rec['v4_region'])
    elif rec['sample'].startswith('PJ'):
        return('ID')
#     elif rec['sample'].startswith('QM'):
#         return('MG')
#     elif rec['sample'].startswith('QS'):
#         return('MG')
    elif rec['solaris_country_code'] not in null_vals:
        return(rec['solaris_country_code'])
    elif rec['solaris_country'] not in null_vals:
        return(rec['solaris_country'])
    else:
        return('unknown')
       


tbl_regions = (
    etl.fromxlsx(v4_metadata_fn, 'Locations')
    .cut(['Country', 'Region'])
    .rename('Country', 'country_from_region')
    .rename('Region', 'region')
    .selectne('region', '-')
    .distinct(['country_from_region', 'region'])
)
tbl_regions.selecteq('country_from_region', 'KH')


tbl_country_code = (
    etl.fromxlsx(v4_metadata_fn, 'CountryCodes')
    .rename('County', 'country')
    .rename('Code', 'code_from_country')
    .rename('SubContinent', 'subcontintent')
    .selectne('country', '-')
)
tbl_country_code.displayall()


def determine_region_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['code_from_country'] not in null_vals:
        return(rec['code_from_country'])
    elif rec['dataset'] == 'pf3k_pilot_5_0_broad':
        return('SN')
    else:
        return(rec['region'])


def determine_country_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['country_from_region'] not in null_vals:
        return(rec['country_from_region'])
    else:
        return(rec['region_code'])


tbl_iso_country_codes = (
    etl.fromcsv(iso_country_code_fn)
    .cut(['official_name', 'ISO3166-1-Alpha-2'])
    .rename('official_name', 'country_name')
)
tbl_iso_country_codes


tbl_sub_continents = etl.fromxlsx(sub_contintent_fn)
tbl_sub_continents


tbl_sub_continent_names = etl.fromxlsx(sub_contintent_fn, 'Names').convertnumbers()
tbl_sub_continent_names


tbl_vrpipe_metadata = etl.fromtsv(vrpipe_metadata_fn)
print(len(tbl_vrpipe_metadata.data()))
tbl_vrpipe_metadata


tbl_manifest.selecteq('country_code', 'CI')


final_fields = [
    'path', 'sample', 'oxford_src_code', 'oxford_donor_code', 'dataset', 'type', 'region_code', 'country_code',
    'country_name', 'sub_continent', 'sub_continent_name', 'sub_continent_number',
    'bases_of_5X_coverage', 'mean_coverage',
    'bases_of_1X_coverage', 'bases_of_10X_coverage', 'bases_of_20X_coverage'
]

tbl_manifest = (
    tbl_all_bams
    .leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code')
    .replace('type', None, 'unknown')
    .leftjoin(tbl_v4_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_v5_metadata, lkey='sample', rkey='Sample')
    .addfield('region', determine_region)
    .leftjoin(tbl_country_code.cut(['country', 'code_from_country']), lkey='region', rkey='country')
    .addfield('region_code', determine_region_code)
    .replace('region_code', 'Benin', 'BJ')
    .replace('region_code', 'Mauritania', 'MR')
    .replace('region_code', "Cote d'Ivoire (Ivory Coast)", 'CI')
    .replace('region_code', 'Ethiopia', 'ET')
    .replace('region_code', 'US', 'Lab')
    .leftjoin(tbl_regions, lkey='region_code', rkey='region')
    .addfield('country_code', determine_country_code)
    .leftjoin(tbl_iso_country_codes, lkey='country_code', rkey='ISO3166-1-Alpha-2')
    .replace('country_name', "Côte d'Ivoire", "Ivory Coast")
    .leftjoin(tbl_sub_continents.cut(['region_code', 'sub_continent']), key='region_code')
    .leftjoin(tbl_sub_continent_names, key='sub_continent')
    .leftjoin(tbl_vrpipe_metadata, key='sample')
    .selectne('region_code', 'unknown') # NB this removes vivax samples, of which there are lots in Ethiopia
    .sort(['sub_continent_number', 'country_name', 'sample'])
    .cut(final_fields)
)


tbl_manifest


tbl_manifest.selectnone('bases_of_10X_coverage')


tbl_manifest_all_metrics = (
    tbl_manifest
    .selectne('mean_coverage', 'unknown')
    .selectnotnone('mean_coverage')
    .selectne('bases_of_10X_coverage', 'unknown')
    .selectnotnone('bases_of_10X_coverage')
    .selectne('bases_of_20X_coverage', 'unknown')
    .selectnotnone('bases_of_20X_coverage')
)

mean_coverage = (
    tbl_manifest_all_metrics
    .convert('mean_coverage', float)
    .values('mean_coverage')
    .array()
)
bases_of_1X_coverage = (
    tbl_manifest_all_metrics
    .convert('bases_of_1X_coverage', int)
    .values('bases_of_1X_coverage')
    .array()
)
bases_of_5X_coverage = (
    tbl_manifest_all_metrics
    .convert('bases_of_5X_coverage', int)
    .values('bases_of_5X_coverage')
    .array()
)
bases_of_10X_coverage = (
    tbl_manifest_all_metrics
    .convert('bases_of_10X_coverage', int)
    .values('bases_of_10X_coverage')
    .array()
)
bases_of_20X_coverage = (
    tbl_manifest_all_metrics
    .convert('bases_of_20X_coverage', int)
    .values('bases_of_20X_coverage')
    .array()
)
print(len(mean_coverage))
print(len(bases_of_1X_coverage))
print(len(bases_of_5X_coverage))
print(len(bases_of_10X_coverage))
print(len(bases_of_20X_coverage))


import pandas as pd
df = pd.DataFrame({
        'Mean coverage': mean_coverage,
        'Bases of 1x coverage': bases_of_1X_coverage,
        'Bases of 5x coverage': bases_of_5X_coverage,
        'Bases of 10x coverage': bases_of_10X_coverage,
        'Bases of 20x coverage': bases_of_20X_coverage,
})


mean_coverage


bases_of_10X_coverage.astype(float)


sns.jointplot('Mean coverage', 'Bases of 1x coverage', df, kind="hex")


sns.jointplot('Mean coverage', 'Bases of 5x coverage', df, kind="hex")


sns.jointplot('Mean coverage', 'Bases of 10x coverage', df, kind="hex")


sns.jointplot('Mean coverage', 'Bases of 20x coverage', df, kind="hex")


sns.jointplot(mean_coverage, bases_of_5X_coverage, kind="hex")


sns.jointplot(mean_coverage, bases_of_10X_coverage, kind="hex")


sns.jointplot(mean_coverage, bases_of_20X_coverage, kind="hex")


sns.jointplot(bases_of_10X_coverage, bases_of_20X_coverage, kind="hex")


bases_of_10X_coverage


tbl_manifest.selectnotnone('orig_bases_of_5X_coverage')


tbl_manifest.select(lambda rec: rec['orig_bases_of_5X_coverage'] is not None and rec['orig_bases_of_5X_coverage'] != rec['bases_of_5X_coverage'])


tbl_manifest.select(lambda rec: rec['orig_mean_coverage'] is not None and rec['orig_mean_coverage'] != rec['mean_coverage'])


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.selectnone('sub_continent').displayall()


tbl_manifest.selecteq('sub_continent', 'Lab')


len(tbl_all_bams.leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code').data())


manifest_fn


tbl_manifest.totsv(manifest_fn)


tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).totsv(jim_manifest_fn)


manifest_fn


len(tbl_manifest.data())


len(tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).data())


len(tbl_manifest.distinct('sample').data())


tbl_temp = tbl_manifest.addfield('bam_exists', lambda rec: os.path.exists(rec['path']))
tbl_temp.valuecounts('bam_exists')


with open(lookseq_fn, "w") as fo:
    for rec in tbl_manifest:
        bam_fn = rec[0]
        sample_name = "%s_%s" % (rec[1].replace('-', '_'), rec[3])
        group_name = "%s %s %s %s" % (rec[9], rec[7], rec[6], rec[4])
        print(
            '"%s" : { "bam":"%s", "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "%s" } ,' % (sample_name, bam_fn, group_name),
            file=fo
        )
# "PG0049_CW2" : { "bam":"/lustre/scratch109/malaria/pfalciparum/output/e/b/8/6/144292/1_bam_merge/pe.1.bam" , "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "1104-PF-LAB-WENDLER" } ,


2+2


tbl_manifest


tbl_manifest.tail(5)


tbl_manifest.select(lambda rec: rec['sample'].startswith('QZ')).displayall()


tbl_manifest.selecteq('sample', 'WL0071-C').displayall()


tbl_manifest.selecteq('country_code', 'UK').displayall()


tbl_manifest.valuecounts('type', 'dataset').displayall()


lkp_country_code = etl.lookup(tbl_country_code, 'code', 'country')


tbl_manifest.selecteq('type', 'unknown').selecteq('dataset', 'pf_community_5_1')


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None and rec['v4_region'] != rec['v5_region'])


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is None)


lkp_code_country = etl.lookup(tbl_country_code, 'country', 'code')


lkp_country_code['BZ']


tbl_manifest.valuecounts('region').displayall()


tbl_manifest.selecteq('region', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('region_code').displayall()


tbl_manifest.selecteq('region_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_code').displayall()


tbl_manifest.selecteq('country_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_name').displayall()


tbl_manifest.valuecounts('region_code', 'country_name').toxlsx(regions_in_dataset_fn)


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.cut(['path', 'sample', 'dataset', 'type', 'region_code', 'country_code', 'country_name', 'sub_continent'])





# # Introduction
# The purpose of this notebook is to compare the results of running Olivo's speceis tools between 5.1 and 6.1 releases, and against results from Magnus's speciator code.
# 
# See 20161117_run_Olivo_GRC.ipynb for details of running Olivo's GRC code on the Pf6.0 release.

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_species_results_Pf6"
get_ipython().system('mkdir -p {output_dir}')
olivo_7979_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC/samplesMeta5x-V1.0.xlsx"
# reads_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"
# vcf_6_0_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161115_GRC_from_VCF/Pf_6_GRC_from_vcf.xlsx"

all_calls_crosstab_fn = "%s/all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/discordant_nonmissing_calls_crosstab.xlsx" % output_dir

speciator_pv_unique_coverage_fn = "/nfs/team112_internal/rp7/data/pv/analysis/20161003_pv_3_0_sample_metadata/pv_unique_coverage.txt"
reads_6_0_species_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/species/AllSamples-AllTargets.classes.tab"


sample_swaps_51_60 = [
    'PP0011-C', 'PP0010-C', 'PC0003-0', 'PH0027-C', 'PH0027-C', 'PG0313-C', 'PG0305-C', 'PG0304-C', 'PG0282-C',
    'PG0309-C', 'PG0334-C', 'PG0306-C', 'PG0332-C', 'PG0311-C', 'PG0330-C', 'PG0310-C', 'PG0312-C', 'PG0335-C',
    'PG0280-C', 'PG0281-C', 'PG0308-C', 'PP0010-C', 'PP0011-C', 'PC0003-0', 'PH0027-C', 'PH0027-C', 'PG0335-C',
    'PG0308-C', 'PG0332-C', 'PG0330-C', 'PG0311-C', 'PG0305-C', 'PG0309-C', 'PG0304-C', 'PG0313-C', 'PG0280-C',
    'PG0312-C', 'PG0334-C', 'PG0306-C', 'PG0281-C', 'PG0282-C', 'PG0310-C'
]


tbl_5_1_species = (
    etl
    .fromxlsx(olivo_7979_results_fn)
)
print(len(tbl_5_1_species.data()))
tbl_5_1_species


tbl_6_0_species = (
    etl
    .fromtsv(reads_6_0_species_results_fn)
    .convertnumbers()
)
print(len(tbl_6_0_species.data()))
tbl_6_0_species


aggregation = collections.OrderedDict()
aggregation['count'] = len
aggregation['sum'] = 'pv_unique_coverage', sum

tbl_speciator_pv_unique_coverage = (
    etl
    .fromtsv(speciator_pv_unique_coverage_fn)
    .convertnumbers()
    .aggregate('ox_code', aggregation)
    .addfield('pv_unique_coverage', lambda rec: rec['sum'] / rec['count'])
)
print(len(tbl_speciator_pv_unique_coverage.data()))
tbl_speciator_pv_unique_coverage


len(tbl_speciator_pv_unique_coverage.duplicates('ox_code'))


tbl_species_5_1_vs_6_0 = (
    tbl_5_1_species
    .join(tbl_6_0_species, key='Sample')
    .cut(['Sample', 'Species', 'SampleClass'])
    .selectnotin('Sample', sample_swaps_51_60)
)
print(len(tbl_species_5_1_vs_6_0.data()))
tbl_species_5_1_vs_6_0


df_species_5_1_vs_6_0 = tbl_species_5_1_vs_6_0.todataframe()


writer = pd.ExcelWriter(all_calls_crosstab_fn)
df_species_5_1_vs_6_0
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1],
    df_species_5_1_vs_6_0.ix[:, 2],
    margins=True
).to_excel(writer, 'All')
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1][df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]],
    df_species_5_1_vs_6_0.ix[:, 2][df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]],
    margins=True
).to_excel(writer, 'Discordant')
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1][
        (df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]) &
        (df_species_5_1_vs_6_0.ix[:, 1] != '-') &
        (df_species_5_1_vs_6_0.ix[:, 2] != '-')
    ],
    df_species_5_1_vs_6_0.ix[:, 2][
        (df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]) &
        (df_species_5_1_vs_6_0.ix[:, 1] != '-') &
        (df_species_5_1_vs_6_0.ix[:, 2] != '-')
    ],
    margins=True
).to_excel(writer, 'Non_missing')
writer.save()


# ## Comparison of 5.1 and 6.0
# In general there was good agreement between the two. There were originally only 32 discordant calls, and 31 of these were either the difference between a non-consensus and non-call or between consensus and non-consensus. One sample (PV0025-C) was Pf,Pv in 5.1 but Pv only in 6.0. Speciator results suggest there probably is a low level of Pf in this sample.
# 
# There were no cases where there was a consensus call in 6.0 but non-consensus or no-call in 5.1, but there were 4 cases where there was a consensus call in 5.1 but non-consensus or no-call in 6.0. Missingness was similar in each call set.

df_species_5_1_vs_6_0[
    (df_species_5_1_vs_6_0['Species'] == 'Pf,Pv') &
    (df_species_5_1_vs_6_0['SampleClass'] == 'Pv')
]


tbl_speciator_pv_unique_coverage.selecteq('ox_code', 'PV0025-C')


get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PV0025_C_5371_1_nonhuman.tab')


df_species_5_1_vs_6_0[
    (df_species_5_1_vs_6_0['Species'] == 'Pf,Pv') &
    (df_species_5_1_vs_6_0['SampleClass'] == 'Pf*,Pv')
]


get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PN0157_C_12483_3_32.tab')


get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PV0255_C_8129_2_20.tab')


# ## Comparison of Olivo's species code with Magnus's speciator
# 

tbl_species_6_0_vs_speciator = (
    tbl_6_0_species
    .join(tbl_speciator_pv_unique_coverage, lkey='Sample', rkey='ox_code')
    .cut(['Sample', 'SampleClass', 'pv_unique_coverage'])
    .addfield('Pv classification', lambda rec: 'Consensus' if 'Pv' in rec['SampleClass'] and not 'Pv*' in rec['SampleClass']
        else 'Evidence' if 'Pv*' in rec['SampleClass']
        else 'Unknown' if rec['SampleClass'] == '-'
        else 'No Pv'
    )
)
print(len(tbl_species_6_0_vs_speciator.data()))
tbl_species_6_0_vs_speciator


tbl_species_6_0_vs_speciator.valuecounts('Pv classification').displayall()


tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()


tbl_species_6_0_vs_speciator.selecteq('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()


tbl_species_6_0_vs_speciator.valuecounts('Pv classification').toxlsx("%s/Pv_classification.xlsx" % output_dir)


# Note I have removed samples with zero Pv coverage as 
df_species_6_0_vs_speciator = tbl_species_6_0_vs_speciator.todataframe()
df_species_6_0_vs_speciator = df_species_6_0_vs_speciator[df_species_6_0_vs_speciator['pv_unique_coverage'] > 0]
df_species_6_0_vs_speciator['Log10(Pv coverage)'] = np.log10(df_species_6_0_vs_speciator['pv_unique_coverage'])
# df_species_6_0_vs_speciator = df_species_6_0_vs_speciator[np.logical_not(np.isinf(df_species_6_0_vs_speciator['Log10(Pv coverage)']))]
# df_species_6_0_vs_speciator['Log10(Pv coverage)'][np.isinf(df_species_6_0_vs_speciator['Log10(Pv coverage)'])] = -5.0


df_species_6_0_vs_speciator['Log10(Pv coverage)'].describe()


ax = sns.violinplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator)


fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(1, 1, 1)
ax = sns.boxplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator, ax=ax)
# ax = sns.swarmplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator,
#                    color="white", edgecolor="gray")
ax.set_xticklabels(['No Pv\nn=7,595', 'Unknown\nn=31', 'Evidence\nn=106', 'Consensus\nn=139'])
fig.savefig("%s/Pv_in_Pf.png" % output_dir, dpi=300)


tbl_species_6_0_vs_speciator.selecteq('Pv classification', 'No Pv').selectgt('pv_unique_coverage', 0.3).displayall()


np.log10(0.507)


tbl_species_6_0_vs_speciator


# ## Species calls for Pf samples which have been included in Pv 3.0 build
# 29 Pf samples were included in Pv 3.0 build because they had pv_unique_converage > 5.0 according to Speciator results (see https://github.com/malariagen/pv/blob/master/notes/release_3_0/20161003_pv_3_0_sample_metadata.ipynb). The following few cells show results from Olivo's code on this, and makes a quick but unsuccessful attempt to see if the 5 samples that don't receive a call have mutations in a few common variants in anchor sequences. I decided to abandon this as really needs to be done properly after discussion.

# This is the threshold for inclusion in Pv build
tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 5.0).valuecounts('Pv classification').displayall()


# The following are inlcuded in Pv 3.0 build, but are not given a species
# The following few cells look at some common variants in Pf 6.0, to see if they affect anchors in any of these
# samples, but they don't
tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 5.0).selecteq('Pv classification', 'Unknown')


vcf_file_format = "/nfs/team112_internal/production_files/Pf/6_0/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz"
def non_ref_sample(chrom='Pf_M76611', pos=960):
    vcf_reader = vcf.Reader(filename=vcf_file_format % chrom)
    samples = vcf_reader.samples
    for record in vcf_reader.fetch(chrom, pos-1, pos):
        print(record, record.FILTER)
        for sample in samples:
            GT = record.genotype(sample)['GT']
            if not GT in ['0/0', './.']:
                print(sample, record.genotype(sample))

non_ref_sample()


non_ref_sample(pos=1033)


non_ref_sample(pos=1100)


# ## Discordant results between lanes
# Before I realised that we had speciator results for each lane, I had noticed that one sample that was called as Pf,Pv by Olivo's code, had zero pv_unique_coverage by Magnus's code. I then realised this sample had two lanes, only one of which had zero pv_unique_coverage. One of the lanes had hardly any coverage of anything (zero unique coverage for Pf as well as Pv).

# Note that was run before I recreated the table to have mean coverage for each ox_code
tbl_species_6_0_vs_speciator.selecteq('Sample', 'PH1186-C')


tbl_species_6_0_vs_speciator.selecteq('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()


(
    etl
    .fromtsv(speciator_pv_unique_coverage_fn)
    .convertnumbers()
    .selecteq('ox_code', 'PH1186-C')
)


get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PH1186_C_13253_5_7.tab')


get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PH1186_C_14323_2_6.tab')





pv_unique_coverage = collections.OrderedDict()
pv_unique_coverage['Pv consensus'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: 'Pv' in rec['SampleClass'] and not 'Pv*' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)
pv_unique_coverage['Pv evidence'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: 'Pv*' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)
pv_unique_coverage['No Pv'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: not 'Pv' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)


# # Conclusion
# In general there was good agreement between the 5.1 and 6.0. There were originally only 32 discordant calls, and 31 of these were either the difference between a non-consensus and non-call or between consensus and non-consensus. One sample (PV0025-C) was Pf,Pv in 5.1 but Pv only in 6.0. Speciator results suggest there probably is a low level of Pf in this sample.
# 
# There were no cases where there was a consensus call in 6.0 but non-consensus or no-call in 5.1, but there were 4 cases where there was a consensus call in 5.1 but non-consensus or no-call in 6.0. Missingness was similar in each call set.
# 




# # Introduction
# This notebook creates a file of all Pf samples, showing whether they are in Pf 5.0, 5.1, 6.0, in progress or other

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release"
get_ipython().system('mkdir -p {output_dir}')

report_sample_status_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_parse_sample_status_report/2016_12_07_report_sample_status.txt"
solaris_fn = "%s/PF_metadata_base.csv" % output_dir
olivo_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC/samplesMeta5x-V1.0.xlsx"
sample_5_0_fn = "/nfs/team112_internal/production/release_build/5_0_study_samples.tab"
sample_5_1_fn = "/nfs/team112_internal/production/release_build/5_1_study_samples_all.tab"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"

study_summary_fn = "%s/Pf_6_0_study_summary.xlsx" % output_dir


tbl_report_sample_status = (
    etl
    .fromtsv(report_sample_status_fn)
    .selecteq('Taxon', 'PF')
)
print(len(tbl_report_sample_status.data()))
tbl_report_sample_status


tbl_report_sample_status.valuecounts('State').displayall()


tbl_report_sample_status.valuecounts('Note').displayall()


tbl_solaris = (etl
    .fromcsv(solaris_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'sampling_date'])
#     .rename('sampling_date', 'cinzia_sampling_date')
    .unique('oxford_code')
)
print(len(tbl_solaris.data()))
tbl_solaris.tail()


tbl_olivo = (etl
    .fromxlsx(olivo_fn)
)
print(len(tbl_olivo.data()))
tbl_olivo.tail()


tbl_sample_5_0 = (etl
    .fromtsv(sample_5_0_fn)
    .pushheader(['study', 'oxford_code'])
)
print(len(tbl_sample_5_0.data()))
tbl_sample_5_0.tail()
samples_5_0 = tbl_sample_5_0.values('oxford_code').array()
print(len(samples_5_0))


tbl_sample_5_1 = (etl
    .fromtsv(sample_5_1_fn)
    .pushheader(['study', 'oxford_code'])
)
print(len(tbl_sample_5_1.data()))
tbl_sample_5_1.tail()
samples_5_1 = tbl_sample_5_1.values('oxford_code').array()
print(len(samples_5_1))


tbl_sample_6_0 = (etl
    .fromtsv(sample_6_0_fn)
)
print(len(tbl_sample_6_0.data()))
samples_6_0 = tbl_sample_6_0.values('sample').array()
print(len(samples_6_0))
tbl_sample_6_0.tail()


def which_release(rec):
    if rec['in_5_0']:
        if (not rec['in_5_1']) or (not rec['in_6_0']):
            return('5_0_NOT_SUBSEQUENT')
        else:
            return('5_0')
    elif rec['in_5_1']:
        if (not rec['in_6_0']):
            return('5_1_NOT_SUBSEQUENT')
        else:
            return('5_1')
    elif rec['in_6_0']:
        return('6_0')
    elif rec['State'] == 'in progress':
        return('in_progress')
    else:
        return('no_release')


tbl_metadata = (
    tbl_report_sample_status
    .leftjoin(tbl_solaris, lkey='Oxford Code', rkey='oxford_code')
    .addfield('in_5_0', lambda rec: rec['Oxford Code'] in samples_5_0)
    .addfield('in_5_1', lambda rec: rec['Oxford Code'] in samples_5_1)
    .addfield('in_6_0', lambda rec: rec['Oxford Code'] in samples_6_0)
    .addfield('release', which_release)
    .leftjoin(tbl_olivo, lkey='Oxford Code', rkey='Sample')
    .outerjoin(tbl_sample_6_0.rename('study', 'study2'), lkey='Oxford Code', rkey='sample')
)
print(len(tbl_metadata.data()))
tbl_metadata.tail()


tbl_metadata.valuecounts('Country of Origin', 'country').displayall()


tbl_sample_6_0.antijoin(tbl_report_sample_status, rkey='Oxford Code', lkey='sample')


tbl_metadata.valuecounts('release').displayall()


tbl_metadata.pivot('study', 'release', 'release', len).displayall()


tbl_metadata.pivot('study', 'release', 'release', len).toxlsx(study_summary_fn)


tbl_panoptes_samples = (
    tbl_metadata
    .selectnotnone('study2')
)
print(len(tbl_panoptes_samples.data()))


get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161205_sample_level_summaries'
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pf60')
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pv30')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

GENOME_FN = collections.OrderedDict()
genome_fn = collections.OrderedDict()
genome = collections.OrderedDict()
GENOME_FN['Pf60'] = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
GENOME_FN['Pv30'] = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
genome_fn['Pf60'] = "%s/Pfalciparum.genome.fasta" % output_dir
genome_fn['Pv30'] = "%s/PvivaxP01.genome.fasta" % output_dir

run_create_sample_summary_job_fn = "%s/scripts/run_create_sample_summary_job.sh" % output_dir
submit_create_sample_summary_jobs_fn = "%s/scripts/submit_create_sample_summary_jobs.sh" % output_dir


# sites_annotation_pf60_fn = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161125_Pf60_final_vcfs/vcf/SNP_INDEL_WG.combined.filtered.annotation.vcf.gz'
hdf_fn = collections.OrderedDict()
hdf_fn['Pf60'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'
hdf_fn['Pv30'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161201_Pv_30_HDF5_build/hdf5/Pv_30.h5'


for release in GENOME_FN:
    get_ipython().system('cp {GENOME_FN[release]} {genome_fn[release]}')
    genome[release] = pyfasta.Fasta(genome_fn[release])
    print(sorted(genome[release].keys())[0])


hdf = collections.OrderedDict()
for release in hdf_fn:
    hdf[release] = h5py.File(hdf_fn[release], 'r')
    print(release, len(hdf[release]['samples']))
    


get_ipython().run_cell_magic('time', '', 'import pickle\nimport allel\ncalldata_subset = collections.OrderedDict()\nfor release in hdf_fn:\n    calldata_subset[release] = collections.OrderedDict()\n    for variable in [\'genotype\', \'GQ\', \'DP\', \'PGT\']:\n        calldata_subset[release][variable] = collections.OrderedDict()\n        calldata = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][variable])\n        \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first_pass.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_pass.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_all.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n            \n#             print(release, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n#             print(release, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n        pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))')


# %%time
# import pickle
# genotypes_subset_fn = "%s/genotypes_subset.p" % output_dir
# if os.path.exists(genotypes_subset_fn):
#     genotypes_subset = pickle.load(open(genotypes_subset_fn, "rb"))
# else:
#     genotypes = collections.OrderedDict()
#     genotypes_subset = collections.OrderedDict()
#     import allel
#     for release in hdf_fn:
#     # for release in ['Pv30']:
#         genotypes[release] = allel.GenotypeChunkedArray(hdf[release]['calldata']['genotype'])
#         genotypes_subset[release] = collections.OrderedDict()
#         print(release, 'first')
#         genotypes_subset[release]['first'] = genotypes[release].subset(
#             (hdf[release]['variants']['FILTER_PASS'][:]) &
#             (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
#         )
#         print(release, 'pass')
#         genotypes_subset[release]['pass'] = genotypes[release].subset(hdf[release]['variants']['FILTER_PASS'][:])
#         print(release, 'all')
#         genotypes_subset[release]['all'] = genotypes[release].subset()
#     pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))


get_ipython().run_cell_magic('time', '', 'import pickle\nGQ_subset_fn = "%s/GQ_subset.p" % output_dir\nif os.path.exists(GQ_subset_fn):\n    genotypes_subset = pickle.load(GQ_subset_fn)\nelse:\n    GQ = collections.OrderedDict()\n    GQ_subset = collections.OrderedDict()\n    for release in hdf_fn:\n        GQ[release] = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][\'GQ\'])\n        GQ_subset[release] = collections.OrderedDict()\n        print(release, \'first\')\n        GQ_subset[release][\'first\'] = GQ[release].subset(\n            (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n            (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n        )\n        print(release, \'pass\')\n        GQ_subset[release][\'pass\'] = GQ[release].subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n        print(release, \'all\')\n        GQ_subset[release][\'all\'] = GQ[release].subset()\n    pickle.dump(GQ_subset, open(GQ_subset_fn, "wb"))')


pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))


temp = (
    (hdf[release]['variants']['FILTER_PASS'][:]) &
    (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
)
pd.value_counts(temp)


sorted(genome[release].keys())[0].encode('ascii')


pd.value_counts(hdf[release]['variants']['CHROM'][:])


hdf[release]['samples'][:]


genotypes_subset['Pf60']


genotypes_subset['Pv30']


import allel
def create_sample_summary(hdf5_fn=hdf_fn['Pf60'], index=0, output_filestem="%s/sample_summaries/Pf60" % output_dir):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
    output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
    fo = open(output_fn, 'w')
    print(0)
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
#     genotypes = allel.GenotypeChunkedArray(hdf['calldata']['genotype'])
    print(1)
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    print(2)
    genotypes_pass = genotypes[is_pass]
    
    print(3)
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    print(4)
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    
    print(5)
    results['num_variants']      = genotypes.shape[0]
    results['num_pass_variants'] = np.count_nonzero(is_pass)
    results['num_missing']       = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']  = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']        = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']   = (results['num_pass_variants'] - results['num_pass_missing'])
    print(6)
    results['num_het']           = genotypes.count_het(axis=0)[0]
    results['num_pass_het']      = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']       = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']  = genotypes_pass.count_hom_alt(axis=0)[0]
    print(7)
    results['num_snp_hom_ref']   = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']       = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']   = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref'] = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']     = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt'] = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]
    print(8)    
    results['num_ins_hom_ref']   = genotypes_pass.subset(is_ins).count_hom_ref(axis=0)[0]
    results['num_ins_het']       = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins']           = (results['num_ins_hom_ref'] + results['num_ins_het'])
    results['num_del_hom_ref']   = genotypes_pass.subset(is_del).count_hom_ref(axis=0)[0]
    results['num_del_het']       = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del']           = (results['num_del_hom_ref'] + results['num_del_het'])
    
    print(9)
    results['pc_pass']           = results['num_pass_called'] / results['num_called']
    results['pc_missing']        = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']   = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']            = results['num_het'] / results['num_called']
    results['pc_pass_het']       = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']        = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']   = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']            = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_ins']            = (results['num_ins'] / (results['num_ins'] + results['num_del']))

    print(10)
    
    print('\t'.join([str(x) for x in list(results.keys())]), file=fo)
    print('\t'.join([str(x) for x in list(results.values())]), file=fo)
    fo.close()
    
    df_sample_summary = pd.DataFrame(
            {
                'Sample': pd.Series(results['sample_id']),
                'Variants called': pd.Series(results['num_called']),
                'Variants missing': pd.Series(results['num_missing']),
                'Proportion missing': pd.Series(results['pc_missing']),
                'Proportion pass missing': pd.Series(results['pc_pass_missing']),
                'Proportion heterozygous': pd.Series(results['pc_het']),
                'Proportion pass heterozygous': pd.Series(results['pc_pass_het']),
                'Proportion homozygous alternative': pd.Series(results['pc_hom_alt']),
                'Proportion pass homozygous alternative': pd.Series(results['pc_pass_hom_alt']),
                'Proportion variants SNPs': pd.Series(results['pc_snp']),
                'Proportion indels insertions': pd.Series(results['pc_ins']),
            }
        )  
    return(df_sample_summary, results)


hdf_fn['Pf60']


import allel
def create_sample_summary(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:][is_pass])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:][is_pass] >= 6.0)
    is_vhq_snp = (is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'SYNONYMOUS_CODING')
    is_frameshift = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'FRAME_SHIFT')
    is_inframe = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:][is_pass], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )[is_pass]
    
    is_pass_nonref = (is_pass & ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0)))
    is_biallelic_snp_nonref = (is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    is_biallelic_indel_nonref = (~is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    
    GQ = hdf['calldata']['GQ'][:, [index]][is_pass]
    DP = hdf['calldata']['DP'][:, [index]][is_pass]
    PGT = hdf['calldata']['PGT'][:, [index]][is_pass]
    
    mutations = np.char.add(hdf['variants']['REF'][:][is_pass][is_biallelic_snp_nonref], hdf['variants']['ALT'][:, 0][is_pass][is_biallelic_snp_nonref])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']         = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_het']                  = genotypes.count_het(axis=0)[0]
    results['num_pass_het']             = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']              = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']         = genotypes_pass.count_hom_alt(axis=0)[0]
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass_nonref)
    
    results['num_biallelic_het']        = genotypes_pass.subset(is_bi).count_het(axis=0)[0]
    results['num_biallelic_hom_alt']    = genotypes_pass.subset(is_bi).count_hom_alt(axis=0)[0]
    results['num_spanning_del_het']     = genotypes_pass.subset(is_sd).count_het(axis=0)[0]
    results['num_spanning_del_hom_alt'] = genotypes_pass.subset(is_sd).count_hom_alt(axis=0)[0]
    results['num_multiallelic_het']     = genotypes_pass.subset(is_mu).count_het(axis=0)[0]
    results['num_multiallelic_hom_alt'] = genotypes_pass.subset(is_mu).count_hom_alt(axis=0)[0]
    
    results['num_snp_hom_ref']          = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']              = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']          = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt']        = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]

    results['num_ins_het']              = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins_hom_alt']          = genotypes_pass.subset(is_ins).count_hom_alt(axis=0)[0]
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del_hom_alt']          = genotypes_pass.subset(is_del).count_hom_alt(axis=0)[0]
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = genotypes_pass.subset(is_coding).count_het(axis=0)[0]
    results['num_coding_hom_alt']       = genotypes_pass.subset(is_coding).count_hom_alt(axis=0)[0]
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_vhq_snp_hom_ref']      = genotypes_pass.subset(is_vhq_snp).count_hom_ref(axis=0)[0]
    results['num_vhq_snp_het']          = genotypes_pass.subset(is_vhq_snp).count_het(axis=0)[0]
    results['num_vhq_snp_hom_alt']      = genotypes_pass.subset(is_vhq_snp).count_hom_alt(axis=0)[0]
    
    results['num_singleton']            = np.count_nonzero(is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_bi & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_biallelic_snp_nonref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_biallelic_snp_nonref & is_synonymous)
    results['num_bi_frameshift']        = np.count_nonzero(is_biallelic_indel_nonref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_biallelic_indel_nonref & is_inframe)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['pc_pass']                  = results['num_pass_called'] / results['num_called']
    results['pc_missing']               = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = results['num_het'] / results['num_called']
    results['pc_pass_het']              = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']               = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_bi_frameshift']         = results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_bi_transition']         = results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    
    results['mean_GQ']                  = np.mean(GQ)
    results['mean_GQ_2']                = np.nanmean(GQ)
    results['mean_DP']                  = np.mean(DP)
    results['mean_DP_2']                = np.nanmean(DP)
    
    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

    return(results, PGT)


import allel
def create_sample_summary_2(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
#     genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    indel_len = svlen1
    het_indels = (svlen1 != svlen2)
    indel_len[het_indels] = svlen1[het_indels] + svlen2[het_indels]
    
    is_indel = (indel_len != 0)
    is_inframe = ((indel_len != 0) & (indel_len%3 == 0))
    is_frameshift = ((indel_len != 0) & (indel_len%3 != 0))
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))
    is_del = ((svlen1 < 0) | (svlen2 < 0))
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0))) # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:] >= 6.0)
    is_hq_snp = (is_pass & is_snp & is_bi & is_coding)
    is_vhq_snp = (is_pass & is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'SYNONYMOUS_CODING')
    is_frameshift_snpeff = (hdf['variants']['SNPEFF_EFFECT'][:] == b'FRAME_SHIFT')
    is_inframe_snpeff = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )
    
    is_hom_ref = ((genotypes[:, 0, 0] == 0) & (genotypes[:, 0, 1] == 0))
    is_het = ((genotypes[:, 0, 0] != genotypes[:, 0, 1]))
    is_hom_alt = ((genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 0] == genotypes[:, 0, 1]))
    is_non_ref = ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0))
    is_missing = ((genotypes[:, 0, 0] == -1))
    is_called = ((genotypes[:, 0, 0] >= 0))
    
    GQ = hdf['calldata']['GQ'][:, index]
    is_GQ_30 = (GQ >= 30)
    is_GQ_99 = (GQ >= 99)
    DP = hdf['calldata']['DP'][:, index]
    PGT = hdf['calldata']['PGT'][:, index]
    is_phased = np.in1d(PGT, [b'.', b''], invert=True)
    
    mutations = np.char.add(hdf['variants']['REF'][:][(is_pass & is_snp & is_bi & is_non_ref)], hdf['variants']['ALT'][:, 0][(is_pass & is_snp & is_bi & is_non_ref)])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = np.count_nonzero(is_missing)
    results['num_pass_missing']         = np.count_nonzero(is_pass & is_missing)
    results['num_called']               = np.count_nonzero(~is_missing)
#     results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = np.count_nonzero(is_pass & is_called)
#     results['num_pass_called_2']        = np.count_nonzero(is_pass & ~is_missing)
#     results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_hom_ref']              = np.count_nonzero(is_hom_ref)
    results['num_het']                  = np.count_nonzero(is_het)
    results['num_pass_het']             = np.count_nonzero(is_pass & is_het)
    results['num_hom_alt']              = np.count_nonzero(is_hom_alt)
    results['num_pass_hom_alt']         = np.count_nonzero(is_pass & is_hom_alt)
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass & is_non_ref)
#     results['num_variants_2']           = results['num_hom_ref'] + results['num_het'] + results['num_hom_alt'] + results['num_missing']
    
    results['num_biallelic_het']        = np.count_nonzero(is_pass & is_bi & is_het)
    results['num_biallelic_hom_alt']    = np.count_nonzero(is_pass & is_bi & is_hom_alt)
    results['num_spanning_del_het']     = np.count_nonzero(is_pass & is_sd & is_het)
    results['num_spanning_del_hom_alt'] = np.count_nonzero(is_pass & is_sd & is_hom_alt)
    results['num_multiallelic_het']     = np.count_nonzero(is_pass & is_mu & is_het)
    results['num_multiallelic_hom_alt'] = np.count_nonzero(is_pass & is_mu & is_hom_alt)
    
#     results['num_snp_hom_ref']          = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_het']              = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_hom_alt']          = np.count_nonzero(is_pass & is_snp & is_hom_alt)
    results['num_snp']                  = (results['num_snp_het'] + results['num_snp_hom_alt'])
#     results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel_hom_alt']        = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel']                  = (results['num_indel_het'] + results['num_indel_hom_alt'])

    results['num_ins_het']              = np.count_nonzero(is_pass & is_ins & is_het)
    results['num_ins_hom_alt']          = np.count_nonzero(is_pass & is_ins & is_hom_alt)
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = np.count_nonzero(is_pass & is_del & is_het)
    results['num_del_hom_alt']          = np.count_nonzero(is_pass & is_del & is_hom_alt)
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = np.count_nonzero(is_pass & is_coding & is_het)
    results['num_coding_hom_alt']       = np.count_nonzero(is_pass & is_coding & is_hom_alt)
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_hq_snp_called']        = np.count_nonzero(is_hq_snp & ~is_missing)
    results['num_hq_snp_hom_ref']       = np.count_nonzero(is_hq_snp & is_hom_ref)
    results['num_hq_snp_het']           = np.count_nonzero(is_hq_snp & is_het)
    results['num_hq_snp_hom_alt']       = np.count_nonzero(is_hq_snp & is_hom_alt)
    results['num_vhq_snp_called']       = np.count_nonzero(is_vhq_snp & ~is_missing)
    results['num_vhq_snp_hom_ref']      = np.count_nonzero(is_vhq_snp & is_hom_ref)
    results['num_vhq_snp_het']          = np.count_nonzero(is_vhq_snp & is_het)
    results['num_vhq_snp_hom_alt']      = np.count_nonzero(is_vhq_snp & is_hom_alt)
    
    results['num_singleton']            = np.count_nonzero(is_pass & is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_pass & is_bi & is_singleton)
    results['num_hq_snp_singleton']     = np.count_nonzero(is_hq_snp & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_synonymous)
#     results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_frameshift)
#     results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_inframe)
    results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_coding & is_frameshift)
    results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_coding & is_inframe)
    results['num_bi_frameshift']        = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_hq_frameshift']        = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_hq_inframe']           = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_bi_frameshift_snpeff'] = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_frameshift_snpeff)
    results['num_bi_inframe_snpeff']    = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_inframe_snpeff)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['num_phased']               = np.count_nonzero(is_pass & is_phased)
    results['num_phased_non_ref']       = np.count_nonzero(is_pass & is_phased & is_non_ref)
    results['num_phased_hom_ref']       = np.count_nonzero(is_pass & is_phased & is_hom_ref)
    results['num_phased_missing']       = np.count_nonzero(is_pass & is_phased & is_missing)
    
    results['num_GQ_30']                = np.count_nonzero(is_pass & is_called & is_GQ_30)
    results['num_het_GQ_30']            = np.count_nonzero(is_pass & is_het & is_GQ_30)
    results['num_hom_alt_GQ_30']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_30)
    results['num_GQ_99']                = np.count_nonzero(is_pass & is_called & is_GQ_99)
    results['num_het_GQ_99']            = np.count_nonzero(is_pass & is_het & is_GQ_99)
    results['num_hom_alt_GQ_99']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_99)

    results['pc_pass']                  = 0.0 if results['num_called'] == 0 else         results['num_pass_called'] / results['num_called']
    results['pc_missing']               = 0.0 if results['num_variants'] == 0 else         results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = 0.0 if results['num_pass_variants'] == 0 else         results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = 0.0 if results['num_called'] == 0 else         results['num_het'] / results['num_called']
    results['pc_pass_het']              = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_het'] / results['num_pass_called']
    results['pc_hq_snp_het']            = 0.0 if results['num_hq_snp_called'] == 0 else         results['num_hq_snp_het'] / results['num_hq_snp_called']
    results['pc_vhq_snp_het']           = 0.0 if results['num_vhq_snp_called'] == 0 else         results['num_vhq_snp_het'] / results['num_vhq_snp_called']
    results['pc_hom_alt']               = 0.0 if results['num_called'] == 0 else         results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = 0.0 if (results['num_ins'] + results['num_del']) == 0 else         (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = 0.0 if (results['num_bi_nonsynonymous'] + results['num_bi_synonymous']) == 0 else         results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_frameshift']            = 0.0 if (results['num_frameshift'] + results['num_inframe']) == 0 else         results['num_frameshift'] / (results['num_frameshift'] + results['num_inframe'])
    results['pc_bi_frameshift']         = 0.0 if (results['num_bi_frameshift'] + results['num_bi_inframe']) == 0 else         results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_hq_frameshift']         = 0.0 if (results['num_hq_frameshift'] + results['num_hq_inframe']) == 0 else         results['num_hq_frameshift'] / (results['num_hq_frameshift'] + results['num_hq_inframe'])
    results['pc_bi_frameshift_snpeff']  = 0.0 if (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff']) == 0 else         results['num_bi_frameshift_snpeff'] / (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff'])
    results['pc_bi_transition']         = 0.0 if (results['num_bi_transition'] + results['num_bi_transversion']) == 0 else         results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_phased']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_phased_non_ref'] / results['num_pass_non_ref']
    results['pc_phased_hom_ref']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_hom_ref'] / results['num_phased']
    results['pc_phased_missing']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_missing'] / results['num_phased']
    results['pc_GQ_30']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_30'] / results['num_pass_called']
    results['pc_het_GQ_30']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_30'] / results['num_pass_het']
    results['pc_hom_alt_GQ_30']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_30'] / results['num_pass_hom_alt']
    results['pc_GQ_99']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_99'] / results['num_pass_called']
    results['pc_het_GQ_99']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_99'] / results['num_pass_het']
    results['pc_hom_alt_GQ_99']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_99'] / results['num_pass_hom_alt']
     
    results['mean_GQ']                  = np.mean(GQ[is_pass])
    results['mean_GQ_hom_ref']          = np.mean(GQ[is_pass & is_hom_ref])
    results['mean_GQ_het']              = np.mean(GQ[is_pass & is_het])
    results['mean_GQ_hom_alt']          = np.mean(GQ[is_pass & is_hom_alt])
    results['mean_DP']                  = np.mean(DP[is_pass])
    results['mean_DP_hom_ref']          = np.mean(DP[is_pass & is_hom_ref])
    results['mean_DP_het']              = np.mean(DP[is_pass & is_het])
    results['mean_DP_hom_alt']          = np.mean(DP[is_pass & is_hom_alt])
#     results['mean_GQ_2']                = np.nanmean(GQ[is_pass])
#     results['mean_DP_2']                = np.nanmean(DP[is_pass])
#     results['mean_DP']                  = np.mean(DP)
#     results['mean_DP_2']                = np.nanmean(DP)

    results['mean_indel_len']           = np.mean(indel_len[is_pass])
    results['total_indel_len']          = np.sum(indel_len[is_pass])

    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

#     return(results, is_pass, is_phased, is_non_ref, is_hom_ref, is_missing)
#     return(results, svlen, svlen1, svlen2, indel_len, is_indel, is_inframe, is_frameshift, is_pass, is_bi, is_non_ref, is_frameshift_snpeff, is_inframe_snpeff, is_coding, is_vqslod6)
#     return(results, is_pass, is_called, is_GQ_30)
    return(results)


# - Create sample level summaries of genotypes
#     - #Pass/Fail SNP/INDEL BI/MU/SD het/hom
#     - Pass/Fail
#     - SNP/INDEL
#     - INS/DEL
#     - Coding/non-coding
#     - singletons
#     - singleton mean VQSLOD
#     - het/hom
#     - NS/S
#     - mean GQ
#     - mean DP
#     - %phased
#     - num_pass_biallelic_coding_snp_het, num_VQSLOD6_biallelic_coding_snp_het
#     - mean indel size, total_indel_size
#     - %coding indel mod3
#     - #transitions/transversions, Ti/Tv
#     

results = create_sample_summary_2(index=275)


results, is_pass, is_called, is_GQ_30 = create_sample_summary_2()


print('pc_GQ_30', results['pc_GQ_30'])
print('pc_het_GQ_30', results['pc_het_GQ_30'])
print('pc_hom_alt_GQ_30', results['pc_hom_alt_GQ_30'])
print('pc_GQ_99', results['pc_GQ_99'])
print('pc_het_GQ_99', results['pc_het_GQ_99'])
print('pc_hom_alt_GQ_99', results['pc_hom_alt_GQ_99'])


results = create_sample_summary_2()


print('num_frameshift', results['num_frameshift'])
print('num_inframe', results['num_inframe'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('num_hq_frameshift', results['num_bi_frameshift'])
print('num_hq_inframe', results['num_bi_inframe'])
print('num_bi_frameshift_snpeff', results['num_bi_frameshift_snpeff'])
print('num_bi_inframe_snpeff', results['num_bi_inframe_snpeff'])
print()
print('pc_frameshift', results['pc_frameshift'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('pc_hq_frameshift', results['pc_bi_frameshift'])
print('pc_bi_frameshift_snpeff', results['pc_bi_frameshift_snpeff'])


np.count_nonzero(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)


np.where(is_pass & is_vqslod6 & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]


np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]


np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]


np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]
)


np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]
)


index=140202
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['CDS'][index])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=61186
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


is_phased_hom_ref = (is_pass & is_phased & is_hom_ref)
is_phased_missing = (is_pass & is_phased & is_missing)
print(np.count_nonzero(is_phased_hom_ref))
print(np.count_nonzero(is_phased_missing))


np.where(is_phased_hom_ref)


index=71926
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=79318
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])


index=79318
print(hdf['Pf60']['variants']['REF'][71926])
print(hdf['Pf60']['variants']['ALT'][71926, :])
print(hdf['Pf60']['calldata']['genotype'][71926, 0, :])
print(hdf['Pf60']['calldata']['AD'][71926, 0, :])
print(hdf['Pf60']['calldata']['PL'][71926, 0, :])
print(hdf['Pf60']['calldata']['PID'][71926, 0])
print(hdf['Pf60']['calldata']['PGT'][71926, 0])


hdf['Pf60']['variants']['ALT'][:][71926, :]


hdf['Pf60']['calldata']['genotype'][:, 0, :][71926, :]


hdf['Pf60']['calldata']['genotype'][71926, 0, :]


hdf['Pf60']['calldata']['AD'][71926, 0, :]


hdf['Pf60']['calldata']['PL'][71926, 0, :]


hdf['Pf60']['calldata']['PID'][71926, 0]


hdf['Pf60']['calldata']['PGT'][71926, 0]


np.where(is_phased_missing)


pd.value_counts(PGT[:,0])


results, mutations = create_sample_summary()


is_pass = hdf['Pf60']['variants']['FILTER_PASS'][:]
is_snp = (hdf['Pf60']['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
is_bi = (hdf['Pf60']['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
temp = hdf['Pf60']['variants']['REF'][:][is_pass][is_snp]


mutations = np.char.add(
    hdf['Pf60']['variants']['REF'][:][is_pass][(is_snp & is_bi)],
    hdf['Pf60']['variants']['ALT'][:, 0][is_pass][(is_snp & is_bi)]
)


pd.value_counts(mutations)


pd.value_counts(mutations)


list(hdf['Pf60'].keys())


pd.value_counts(hdf['Pf60']['variants']['AC'][:,0] == 0)


pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:])


7182/20


pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:][(
            hdf['Pf60']['variants']['FILTER_PASS'][:] &
            (hdf['Pf60']['variants']['MULTIALLELIC'][:] == b'BI') &
            (hdf['Pf60']['variants']['AC'][:, 0] > 359) &
            (hdf['Pf60']['variants']['AC'][:, 0] < (7182-359))
        )])


print('num_pass_non_ref', results['num_pass_non_ref'])
print('num_pass_non_ref_2', results['num_pass_non_ref_2'])


print('pc_bi_transition', results['pc_bi_transition'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('pc_biallelic', results['pc_biallelic'])
print('pc_spanning_del', results['pc_spanning_del'])
print('pc_mutliallelic', results['pc_mutliallelic'])
print('pc_bi_nonsynonymous', results['pc_bi_nonsynonymous'])
print('pc_bi_nonsynonymous_2', results['pc_bi_nonsynonymous_2'])


pd.value_counts(mutations)


print('num_snp_hom_ref', results['num_snp_hom_ref'])
print('num_snp_het', results['num_snp_het'])
print('num_snp_hom_alt', results['num_snp_hom_alt'])
print('num_indel_hom_ref', results['num_indel_hom_ref'])
print('num_indel_het', results['num_indel_het'])
print('num_indel_hom_alt', results['num_indel_hom_alt'])
print()
print('pc_snp', results['pc_snp'])
# print('pc_snp_v2', results['pc_snp_v2'])


create_sample_summary()


get_ipython().system('/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\\ 6.0/scripts/create_sample_summary.py')


fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX

''', file=fo)
fo.close()


fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
RELEASE=$2
# LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX
OUTPUT_FN=%s/sample_summaries/$RELEASE/results_$INDEX.txt

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX > $OUTPUT_FN

''' % (output_dir), file=fo)
fo.close()


get_ipython().system('LSB_JOBINDEX=2 && bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')


MEMORY=8000
# Kick off Pf 6.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-7182]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')


MEMORY=8000
# Kick off Pv 3.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-1001]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} {hdf_fn[\'Pv30\']} Pv30')


get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pf60/results_0.txt &   cat {output_dir}/sample_summaries/Pf60/results_*.txt | grep -v '^sample_id') > {output_dir}/pf_60_summaries.txt")


get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pv30/results_0.txt &   cat {output_dir}/sample_summaries/Pv30/results_*.txt | grep -v '^sample_id') > {output_dir}/pv_30_summaries.txt")


output_dir





create_sample_summary(1)


df_sample_summary, results = create_sample_summary()


df_sample_summary, results = create_sample_summary(index=100)


b'FP0008-C'.decode('ascii')


'\t'.join([str(x) for x in list(results.values())])


'\t'.join([str(x) for x in list(results.values())])


'\t'.join(list(results.values()))


list(results.values())


list(results.values())[0]


hdf = h5py.File(hdf_fn['Pv30'], 'r')
hdf['calldata']['genotype'].shape


svlen = hdf['variants']['svlen'][:]
svlen


genotype = hdf['calldata']['genotype'][:, 0, :]
genotype


pd.value_counts(genotype[:,0])


print(genotype.shape)
print(svlen.shape)


svlen1 = svlen[np.arange(svlen.shape[0]), genotype[:, 0] - 1]
svlen1[np.in1d(genotype[:, 0], [-1, 0])] = 0


pd.Series(svlen1).describe()


pd.value_counts(svlen1)


svlen2 = svlen[genotype]


svlen


genotype[:,0]


svlen[:, genotype[:,0]]


np.take(svlen[0:100000], genotype[0:100000,0]-1, axis=1)


alt_indexes = genotype[:, 0] - 1
alt_indexes[alt_indexes < 0] = 0


pd.value_counts(alt_indexes)


np.take(svlen[0:10000], alt_indexes[0:10000], axis=0).shape


svlen[0:1002]


alt_indexes[0:1002]


np.take(svlen[0:1002], alt_indexes[0:1002], axis=0)


svlen[0:1002][np.arange(1002), alt_indexes[0:1002]].shape


alt_indexes[0:10000].shape


svlen[0:10000].shape


svlen


print(svlen2.shape)
svlen2


temp = hdf['calldata']['genotype'][:, [0], :]


temp2=allel.GenotypeArray(temp)
temp2


temp.shape


get_ipython().run_cell_magic('time', '', "df_sample_summary = collections.OrderedDict()\n# for release in genotypes_subset:\nfor release in ['Pv30', 'Pf60']:\n    print(release)\n    samples = hdf[release]['samples'][:]\n    pass_variants = hdf[release]['variants']['FILTER_PASS'][:]\n    \n    print(0)\n    is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')\n    is_bi = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'BI')\n    is_sd = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'SD')\n    is_mu = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'MU')\n    is_ins = (hdf[release]['variants']['svlen'][:][pass_variants] > 0)\n    is_del = (hdf[release]['variants']['svlen'][:][pass_variants] < 0)\n    \n    print(1)\n    num_variants = genotypes_subset[release]['all'].shape[0]\n    num_pass_variants = genotypes_subset[release]['pass'].shape[0]\n    num_missing = genotypes_subset[release]['all'].count_missing(axis=0)[:]\n    num_pass_missing = genotypes_subset[release]['pass'].count_missing(axis=0)[:]\n    num_called = (num_variants - num_missing)\n    num_pass_called = (num_pass_variants - num_pass_missing)\n    print(2)\n    num_het = genotypes_subset[release]['all'].count_het(axis=0)[:]\n    num_pass_het = genotypes_subset[release]['pass'].count_het(axis=0)[:]\n    num_hom_alt = genotypes_subset[release]['all'].count_hom_alt(axis=0)[:]\n    num_pass_hom_alt = genotypes_subset[release]['pass'].count_hom_alt(axis=0)[:]\n    print(3)\n    num_snp_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_snp_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_snp_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    num_indel_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_indel_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_indel_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    print(4)    \n    num_ins_hom_ref = genotypes_subset[release]['pass'].subset(is_ins).count_hom_ref(axis=0)[:]\n    num_ins_het = genotypes_subset[release]['pass'].subset(is_ins).count_het(axis=0)[:]\n    num_ins = (num_ins_hom_ref + num_ins_het)\n    num_del_hom_ref = genotypes_subset[release]['pass'].subset(is_del).count_hom_ref(axis=0)[:]\n    num_del_het = genotypes_subset[release]['pass'].subset(is_del).count_het(axis=0)[:]\n    num_del = (num_del_hom_ref + num_del_het)\n    \n    print(5)\n    pc_pass = num_pass_called / num_called\n    pc_missing = num_missing / num_variants\n    pc_pass_missing = num_pass_missing / num_pass_variants\n    pc_het = num_het / num_called\n    pc_pass_het = num_pass_het / num_pass_called\n    pc_hom_alt = num_hom_alt / num_called\n    pc_pass_hom_alt = num_pass_hom_alt / num_pass_called\n    pc_snp = (num_snp_het + num_snp_homalt) / (num_snp_het + num_snp_homalt + num_indel_het + num_indel_homalt)\n    pc_ins = (num_ins / (num_ins + num_del))\n\n    print(6)\n    df_sample_summary[release] = pd.DataFrame(\n            {\n                'Sample': pd.Series(samples),\n                'Variants called': pd.Series(num_called),\n                'Variants missing': pd.Series(num_called),\n                'Proportion missing': pd.Series(pc_missing),\n                'Proportion pass missing': pd.Series(pc_pass_missing),\n                'Proportion heterozygous': pd.Series(pc_het),\n                'Proportion pass heterozygous': pd.Series(pc_pass_het),\n                'Proportion homozygous alternative': pd.Series(pc_hom_alt),\n                'Proportion pass homozygous alternative': pd.Series(pc_pass_hom_alt),\n                'Proportion variants SNPs': pd.Series(pc_snp),\n                'Proportion indels insertions': pd.Series(pc_ins),\n            }\n        )")


is_ins


num_snp_hom_ref = genotypes_subset['Pv30']['pass'][is_snp, :, :].count_hom_ref(axis=0)[:]


genotypes_subset['Pv30']['pass'].subset(is_snp)


pd.value_counts(is_snp)


len(is_snp)


is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')


2+2


df_sample_summary['Pv30']


df_sample_summary['Pf60']





# #Background
# This notebook describes work on a release based on Pf reads found in human sequence data
# 
# See emails from Jim:
# 
# 17/03/2016 14:22 - this has details of files of 5332 'lanelets' in /nfs/team112_internal/production_files/Hs/x10/metrics

# #Plan
# - Try to map all reads from human file for GF5122-C (best covered sample) to 3D7
# - Also the above for phix
# - Put the above into lookseq

get_ipython().run_line_magic('run', 'standard_imports.ipynb')


input_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/input'
output_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/output'
test_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/test'
get_ipython().system('mkdir -p {input_dir}')
get_ipython().system('mkdir -p {output_dir}')
get_ipython().system('mkdir -p {test_dir}')

lanelets_fn = '/nfs/team112_internal/production_files/Hs/x10/metrics/oxcode_cram.tab'
GF5122_C_irods = "%s/GF5122_C.cram.irods" % test_dir
GF5122_C_fofn = "/nfs/team112_internal/production/release_build/Pf/Hs_X10_Pf_1/Hs_X10_Pf_1.lanelets.fofn"


get_ipython().system('grep GF5122 {lanelets_fn}')


get_ipython().system('grep GF5122 {lanelets_fn} > {GF5122_C_irods}')


get_ipython().system('cat {GF5122_C_irods}')


cwd = get_ipython().getoutput('pwd')
cwd = cwd[0]


get_ipython().run_line_magic('cd', '{test_dir}')


tbl_GF5122_C_lanelets = etl.fromtsv(GF5122_C_irods).pushheader(['sample', 'file'])
for rec in tbl_GF5122_C_lanelets.data():
    get_ipython().system('iget {rec[1]}')


cwd


get_ipython().run_line_magic('cd', '{cwd}')
get_ipython().system('pwd')


tbl_fofn = (tbl_GF5122_C_lanelets
 .sub('sample', '-', '_')
 .sub('file', '/seq/[0-9]+/(.*)', '%s/\\1' % test_dir)
 .rename('file', 'path')
 .cut(['path', 'sample'])
)
tbl_fofn


tbl_fofn.totsv(GF5122_C_fofn)


GF5122_C_fofn











# #Earlier failed setups

# # This notebook abandoned as after realising many samples weren't in Pf community 5.1. See 20160323_WillH_1_setup for pipeline actually run

# #Plan
# - Make fofn of Pf community 5.1 sample bams
# - Create various setups (see Dushi email 02/10/2015 16:47)
#     - vrpipe-setup —based_on pf3kgatk_mapping (for example) 4 fofn_with_metadata
#     - wait for Dushi before running pf3kgatk_combine_gvcfs
# - output to /lustre/scratch109/malaria/WillH_1
# - final output to /nfs/team112_internal/production/release_build/Pf/WillH_1
# 
# 

get_ipython().run_line_magic('run', 'standard_imports.ipynb')


pf_51_manifest_fn = '/nfs/team112_internal/production/release_build/Pf/5_1_release_packages/pf_51_freeze_manifest.tab'
WillH_1_samples_fn = '/lustre/scratch109/malaria/WillH_1/meta/Antoine_samples_vrpipe2.txt'


tbl_pf_51_manifest = etl.fromtsv(pf_51_manifest_fn)
tbl_pf_51_manifest


tbl_WillH_1_samples = etl.fromtsv(WillH_1_samples_fn).pushheader(['sample'])
tbl_WillH_1_samples


tbl_vrpipe = (tbl_WillH_1_samples
    .join(tbl_pf_51_manifest, key='sample')
    .cut(['path', 'sample'])
)
print(len(tbl_vrpipe.data()))
tbl_vrpipe


tbl_missing = (tbl_WillH_1_samples
    .antijoin(tbl_pf_51_manifest, key='sample')
)
print(len(tbl_missing.data()))
tbl_missing.displayall()





# This notebook must be run directly from MacBook after running ~/bin/sanger-tunneling.sh in order to connect
# to Sanger network. I haven't figured out a way to do this from Docker container

get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


get_ipython().system('mkdir -p {os.path.dirname(INTERIM5_VCF_FOFN)}')
get_ipython().system('rsync -av malsrv2:{INTERIM5_VCF_FOFN} {os.path.dirname(INTERIM5_VCF_FOFN)}')


for release in CHROM_VCF_FNS.keys():
    for chrom in CHROM_VCF_FNS[release].keys():
        vcf_fn = CHROM_VCF_FNS[release][chrom]
        get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
        get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')


vcf_fn = WG_VCF_FNS['release3']
get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')


RELEASE_3_VCF_FN = '/nfs/team112_internal/production/release_build/Pf3K/pilot_3_0/all_merged_with_calls_vfp_v4.vcf.gz'
RELEASE_4_DIR = '/nfs/team112_internal/production/release_build/Pf3K/pilot_4_0'
INTERIM_5_VCF_FOFN = '/lustre/scratch109/malaria/pf3k_methods/input/output_fofn/pf3kgatk_variant_filtration_ps583for_2640samples.output'





get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = '/lustre/scratch111/malaria/rp7/data/methods-dev/pf3k_techbm/20170111_Pf3k_50_HDF5_build'
vcf_stem = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_{chrom}.combined.filtered.vcf.gz'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
nfs_final_hdf5_dir = '%s/hdf5' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_hdf5_dir}')

GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
genome_fn = "%s/Pfalciparum.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/hdf5')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/npy')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

get_ipython().system('cp {GENOME_FN} {genome_fn}')


genome = pyfasta.Fasta(genome_fn)
genome


fo = open("%s/scripts/vcfnp_variants.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_calldata.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type calldata_2d \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity AD:7 \
    --arity PL:28 \
    --dtype PGT:a3 \
    --dtype PID:a12 \
    --exclude-field MIN_DP \
    --exclude-field RGQ \
    --exclude-field SB''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
inputs=$3
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''', file=fo)
fo.close()


task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

    task = "%s/scripts/vcfnp_calldata.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')


task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "hdf" -n8 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'Pf3D7_01_v3\')} {output_dir}/hdf5/Pf3K_pilot_5_0 {output_dir}/npy')


get_ipython().system('cp {output_dir}/hdf5/* {nfs_final_hdf5_dir}/')





output_dir


task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "full" -R"select[mem>16000] rusage[mem=16000] span[hosts=1]" -M 16000     -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'Pf3D7_01_v3\')}     {output_dir}/hdf5/Pf_60     /lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161124_HDF5_build/npy')


y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT.h5' % output_dir, 'r')


(etl.wrap(
    np.unique(y['variants']['SNPEFF_EFFECT'], return_counts=True)
)
    .transpose()
    .pushheader('SNPEFF_EFFECT', 'number')
    .sort('number', reverse=True)
    .displayall()
)


task_size = 20000
for chrom in ['PvP01_00'] + sorted(genome.keys()):
    if chrom.startswith('Pv'):
        vcf_fn = vcf_stem.format(chrom=chrom)
        if chrom == 'PvP01_00':
            chrom_length = transfer_length
        else:
            chrom_length = len(genome[chrom])
        n_tasks = '1-%s' % ((chrom_length // task_size) + 1)
        print(chrom, n_tasks)

        task = "%s/scripts/vcfnp_variants.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

        task = "%s/scripts/vcfnp_calldata.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')


get_ipython().system('cp /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/hdf5/ ')





(etl.wrap(
    np.unique(y['variants']['CDS'], return_counts=True)
)
    .transpose()
    .pushheader('CDS', 'number')
    .sort('number', reverse=True)
    .displayall()
)


CDS = y['variants']['CDS'][:]
SNPEFF_EFFECT = y['variants']['SNPEFF_EFFECT'][:]
SNP = (y['variants']['VARIANT_TYPE'][:] == b'SNP')
INDEL = (y['variants']['VARIANT_TYPE'][:] == b'INDEL')


np.unique(CDS[SNP], return_counts=True)


2+2


y['variants']['VARIANT_TYPE']


pd.value_counts(INDEL)


pd.crosstab(SNPEFF_EFFECT[SNP], CDS[SNP])


2+2


df = pd.DataFrame({'CDS': CDS, 'SNPEFF_EFFECT':SNPEFF_EFFECT})


writer = pd.ExcelWriter("/nfs/users/nfs_r/rp7/SNPEFF_for_Rob.xlsx")
pd.crosstab(SNPEFF_EFFECT, CDS).to_excel(writer)
writer.save()





pd.crosstab(SNPEFF_EFFECT, y['variants']['CHROM'])


np.unique(y['variants']['svlen'], return_counts=True)


y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT_10pc.h5.h5' % output_dir, 'r')
y


# for field in y['variants'].keys():
for field in ['svlen']:
    print(field, np.unique(y['variants'][field], return_counts=True))














get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_no_PID_PGT_10pc     --output {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5 > {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.md5 ')








get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset_1pc     --output {output_dir}/hdf5/Pf_60_subset_1pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_1pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_1pc.h5 > {output_dir}/hdf5/Pf_60_subset_1pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('{output_dir}/scripts/vcfnp_concat.sh {vcf_fn} {output_dir}/hdf5/Pf_60')


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
# inputs=${vcf}.vcfnp_cache
inputs=%s/npy
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''' % (
        output_dir,
        )
      , file=fo)
fo.close()

#     nv=$(ls -1 ${inputs}/v* | wc -l)
#     nc=$(ls -1 ${inputs}/c* | wc -l)
#     echo variants files $nv >> $log
#     echo calldata files $nc >> $log
#     if [ "$nv" -ne "$nc" ]
#     then
#         echo missing npy files
#         exit 1
#     fi


# # Copy files to /nfs

get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5 {nfs_final_hdf5_dir}/')
get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5.md5 {nfs_final_hdf5_dir}/')





get_ipython().run_line_magic('run', '_standard_imports.ipynb')


scratch_dir = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC"
output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC"
get_ipython().system('mkdir -p {scratch_dir}/grc')
get_ipython().system('mkdir -p {scratch_dir}/species')
get_ipython().system('mkdir -p {scratch_dir}/log')
get_ipython().system('mkdir -p {output_dir}/grc')
get_ipython().system('mkdir -p {output_dir}/species')

bam_fn = "%s/pf_60_mergelanes.txt" % output_dir
bam_list_fn = "%s/pf_60_mergelanes_bamfiles.txt" % output_dir
chromosomeMap_fn = "%s/chromosomeMap.tab" % output_dir
grc_properties_fn = "%s/grc/grc.properties" % output_dir
species_properties_fn = "%s/species/species.properties" % output_dir
submitArray_fn = "%s/grc/submitArray.sh" % output_dir
submitSpeciesArray_fn = "%s/species/submitArray.sh" % output_dir
runArrayJob_fn = "%s/grc/runArrayJob.sh" % output_dir
runSpeciesArrayJob_fn = "%s/species/runArrayJob.sh" % output_dir
mergeGrcResults_fn = "%s/grc/mergeGrcResults.sh" % output_dir
mergeSpeciesResults_fn = "%s/species/mergeSpeciesResults.sh" % output_dir

ref_fasta_fn = "%s/Pfalciparum.genome.fasta" % output_dir
old_ref_fasta_fn = "%s/3D7_V3.fasta" % output_dir


get_ipython().system('cp /lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta {ref_fasta_fn}')
get_ipython().system('cp /lustre/scratch109/malaria/pfalciparum/resources/3D7_V3.fasta {old_ref_fasta_fn}')


# Check old and new MT sequences are identical (they are for the region of interest and indeed for whole sequence)
# The only difference between old and new refs is the apicoplast sequence

from pyfasta import Fasta
from Bio.Seq import Seq

print(len(Seq(Fasta(ref_fasta_fn)['Pf_M76611'][:])))
print(len(Seq(Fasta(old_ref_fasta_fn)['M76611'][:])))

ref_sequence = Seq(Fasta(ref_fasta_fn)['Pf_M76611'][520:1251])
old_ref_sequence = Seq(Fasta(old_ref_fasta_fn)['M76611'][520:1251])
print(str(ref_sequence).upper() == str(old_ref_sequence))

ref_sequence = Seq(Fasta(ref_fasta_fn)['Pf_M76611'][:])
old_ref_sequence = Seq(Fasta(old_ref_fasta_fn)['M76611'][:])
print(str(ref_sequence).upper() == str(old_ref_sequence))

print()

print(len(Seq(Fasta(ref_fasta_fn)['Pf3D7_API_v3'][:])))
print(len(Seq(Fasta(old_ref_fasta_fn)['PFC10_API_IRAB'][:])))

ref_sequence = Seq(Fasta(ref_fasta_fn)['Pf3D7_API_v3'][:])
old_ref_sequence = Seq(Fasta(old_ref_fasta_fn)['PFC10_API_IRAB'][:])


get_ipython().system('cp /nfs/users/nfs_r/rp7/pf_60_mergelanes.txt {bam_fn}')


table1 = [['foo', 'bar'],
          ['C', 2],
          ['A', 9],
          ['A', 6],
          ['F', 1],
          ['D', 10]]


(etl.wrap(table1).addrownumbers().sort('row', reverse=True).cutout('row'))


# Create list of bam files in format required
tbl_bam_file = (etl
    .fromtsv(bam_fn)
    .addfield('ChrMap', 'default')
    .rename('path', 'BamFile')
    .rename('sample', 'Sample')
    .cut(['Sample', 'BamFile', 'ChrMap'])
    .addrownumbers()
    .sort('row', reverse=True) # we reverse it because the largest files are at the end of the list, and we want these kicked off first
    .cutout('row')
)
tbl_bam_file.totsv(bam_list_fn)
get_ipython().system('dos2unix -o {bam_list_fn}')


get_ipython().system('wc -l {bam_list_fn}')


fo = open(grc_properties_fn, 'w')
print('''grc.loci=crt_core,crt_ex01,crt_ex02,crt_ex03,crt_ex04,crt_ex06,crt_ex09,crt_ex10,crt_ex11,dhfr_1,dhfr_2,dhfr_3,dhps_1,dhps_2,dhps_3,dhps_4,mdr1_1,mdr1_2,mdr1_3,arps10,mdr2,fd,exo

# CRT
grc.locus.crt_core.region=Pf3D7_07_v3:403500-403800
grc.locus.crt_core.targets=crt_72-76@403612-403626
grc.locus.crt_core.anchors=403593@TATTATTTATTTAAGTGTA,403627@ATTTTTGCTAAAAGAAC

grc.locus.crt_ex01.region=Pf3D7_07_v3:403150-404420
grc.locus.crt_ex01.targets=crt_24@403291-403293
grc.locus.crt_ex01.anchors=403273@GAGCGTTATA.[AG]GAATTA...AATTTA.TACAAGAA[GA]GAA

grc.locus.crt_ex02.region=Pf3D7_07_v3:403550-403820
grc.locus.crt_ex02.targets=crt_97@403687-403689
grc.locus.crt_ex02.anchors=403657@GGTAACTATAGTTTTGT.[AT]CATC[CT]GAAAC,403690@AACTTTATTTGTATGATTA[TA]GTTCTTTATT

grc.locus.crt_ex03.region=Pf3D7_07_v3:403850-404170
grc.locus.crt_ex03.targets=crt_144@404007-404009,crt_148@404019-404021
grc.locus.crt_ex03.anchors=404022@ACAAGAACTACTGGAAA[TC]AT[CT]CA[AG]TCATTT,403977@TC[CT]AT.TTA.AT[GT]CCTGTTCA.T[CA]ATT

grc.locus.crt_ex04.region=Pf3D7_07_v3:404200-404500
grc.locus.crt_ex04.targets=crt_194@404329-404331,crt_220@404407-404409
grc.locus.crt_ex04.anchors=404304@CGGAGCA[GC]TTATTATTGTTGTAACA...GCTC,404338@GTAGAAATGAAATTATC[TA]TTTGAAACAC,404359@GAAACACAAGAAGAAAATTCTATC[AG]TATTTAATC,404382@C[AG]TATTTAATCTTGTCTTA[AT]TTAGT...TTAATTG

grc.locus.crt_ex06.region=Pf3D7_07_v3:404700-405000
grc.locus.crt_ex06.targets=crt_271@404836-404838
grc.locus.crt_ex06.anchors=404796@TTGTCTTATATT.CCTGTATACACCCTTCCATT[TC]TTAAAA...C

grc.locus.crt_ex09.region=Pf3D7_07_v3:405200-405500
grc.locus.crt_ex09.targets=crt_326@405361-405363,crt_333@405382-405384
grc.locus.crt_ex09.anchors=405334@AAAACCTT[CT]G[CT]ATTGTTTTCCTTCTTT,405364@A.TTGTGATAATTTAATA...AGCTAT

grc.locus.crt_ex10.region=Pf3D7_07_v3:405400-405750
grc.locus.crt_ex10.targets=crt_342@405557-405559,crt_356@405599-405601
grc.locus.crt_ex10.anchors=405539@ATTATCGACAAATTTTCT...[AT]TGACATATAC,405573@TTGTTAGTTGTATACAAG[GT]TCCA[GA]CA,405602@GCAATT[GT]CTTATTACTTTAAATTCTTA[GA]CC

grc.locus.crt_ex11.region=Pf3D7_07_v3:405700-406000
grc.locus.crt_ex11.targets=crt_371@405837-405839
grc.locus.crt_ex11.anchors=405825@[GT]GTGATGTT.[TA]A...G.ACCAAGATTATTAG,405840@G.ACCAAGATTATTAGATTTCGTAACTTTG

# DHFR
grc.locus.dhfr_1.region=Pf3D7_04_v3:748100-748400
grc.locus.dhfr_1.targets=dhfr_51@748238-748240,dhfr_59@748262-748264
grc.locus.dhfr_1.anchors=748200@GAGGTCTAGGAAATAAAGGAGTATTACCATGGAA,748241@TCCCTAGATATGAAATATTTT...GCAG,748265@GCAGTTACAACATATGTGAATGAATC

grc.locus.dhfr_2.region=Pf3D7_04_v3:748250-748550
grc.locus.dhfr_2.targets=dhfr_108@748409-748411
grc.locus.dhfr_2.anchors=748382@CAAAATGTTGTAGTTATGGGAAGAACA,748412@TGGGAAAGCATTCCAAAAAAATTT

grc.locus.dhfr_3.region=Pf3D7_04_v3:748400-748720
grc.locus.dhfr_3.targets=dhfr_164@748577-748579
grc.locus.dhfr_3.anchors=748382@GGGAAATTAAATTACTATAAATG,748382@CTATAAATGTTTTATT...GGAGGTTC,748412@GGAGGTTCCGTTGTTTATCAAG


# DHPS
grc.locus.dhps_1.region=Pf3D7_08_v3:549550-549750
grc.locus.dhps_1.targets=dhps_436@549681-549683,dhps_437@549684-549686
grc.locus.dhps_1.anchors=549657@GTTATAGAT[AG]TAGGTGGAGAATCC,549669@GGTGGAGAATCC..TG.TCC,549687@CCTTTTGTTAT[AG]CCTAATCCAAAAATTAGTG

grc.locus.dhps_2.region=Pf3D7_08_v3:549850-550150
grc.locus.dhps_2.targets=dhps_540@549993-549995
grc.locus.dhps_2.anchors=549949@GTGTAGTTCTAATGCATAAAAGAGG,549970@GAGGAAATCCACATACAATGGAT,549985@CAATGGAT...CTAACAAATTA[TA]GATA,549996@CTAACAAATTA[TA]GATAATCTAGT

grc.locus.dhps_3.region=Pf3D7_08_v3:549950-550250
grc.locus.dhps_3.targets=dhps_581@550116-550118
grc.locus.dhps_3.anchors=550092@CTATTTGATATTGGATTAGGATTT,550119@AAGAAACATGATCAATCT[AT]TTAAACTC

grc.locus.dhps_4.region=Pf3D7_08_v3:550050-550350
grc.locus.dhps_4.targets=dhps_613@550212-550214
grc.locus.dhps_4.anchors=550167@GATGAGTATCCACTTTTTATTGG,550188@GGATATTCAAGAAAAAGATTTATT,550215@CATTGCATGAATGATCAAAATGTTG


# MDR1
grc.locus.mdr1_1.region=Pf3D7_05_v3:957970-958280
grc.locus.mdr1_1.targets=mdr1_86@958145-958147
grc.locus.mdr1_1.anchors=958120@GTTTG[GT]TGTAATATTAAA[GA]AACATG,958141@CATG...TTAGGTGATGATATTAATCCT

grc.locus.mdr1_2.region=Pf3D7_05_v3:958300-958600
grc.locus.mdr1_2.targets=mdr1_184@958439-958441
grc.locus.mdr1_2.anchors=958413@CATATGC[CA]AGTTCCTTTTTAGG,958446@GGTC[AG]TTAATAAAAAAT[GA]CACGTTTGAC

grc.locus.mdr1_3.region=Pf3D7_05_v3:961470-961770
grc.locus.mdr1_3.targets=mdr1_1246@961625-961627
grc.locus.mdr1_3.anchors=961595@GTTATAGAT[AG]TAGGTGGAGAATCC,961628@CTTAGAAA[CT][TA]TATTTTC[AT]ATAGTTAGTC

# ARPS10
grc.locus.arps10.region=Pf3D7_14_v3:2480900-2481200
grc.locus.arps10.targets=arps10_127@2481070-2481072
grc.locus.arps10.anchors=2481045@ATTTAC[CA]TTTTTGCGATCTCCCCAT...[GC],2481079@GACAGT[AC]G[AG]GA[GA]CAATTCGAAATAAAAC

# MDR2
grc.locus.mdr2.region=Pf3D7_14_v3:1956070-1956370
grc.locus.mdr2.targets=mdr2_484@-1956224-1956226
grc.locus.mdr2.anchors=1956203@ACATGTTATTAATCCT[TC]TAT...TGCC,1956227@TGCCGGAATAAT[AG]TACATTAAAACAGAAC

# Ferredoxin
grc.locus.fd.region=Pf3D7_13_v3:748250-748550
grc.locus.fd.targets=fd_193@-748393-748395
grc.locus.fd.anchors=748396@[GA]TGTAGTTCGTCTTCCTTGTG[CT]GTTTC

# Exo
grc.locus.exo.region=Pf3D7_13_v3:2504400-2504700
grc.locus.exo.targets=exo_415@2504559-2504561
grc.locus.exo.anchors=2504526@[GC]ATGATTTTA[AG][CA]AATATGGT[TC]ATAA[CT]GATAAAA,2504562@GAA[GT]TAAA[CT][AC]ATCATTGG[GA]AAAA[TC]AATATATAC
''', file=fo)
fo.close() 


# Sanity check anchors for MDR2 and FD
print(str(Seq(Fasta(ref_fasta_fn)['Pf3D7_14_v3'][2481044:2481073])).upper()) # ARPS10
print(str(Seq(Fasta(ref_fasta_fn)['Pf3D7_14_v3'][2481078:2481106])).upper()) # ARPS10
print()
print(str(Seq(Fasta(ref_fasta_fn)['Pf3D7_14_v3'][1956202:1956230])).upper()) # MDR2
print(str(Seq(Fasta(ref_fasta_fn)['Pf3D7_14_v3'][1956226:1956255])).upper()) # MDR2
print()
print(str(Seq(Fasta(ref_fasta_fn)['Pf3D7_13_v3'][748395:748422])).upper()) # Ferredoxin


fo = open(species_properties_fn, 'w')
print('''sampleClass.classes=Pf,Pv,Pm,Pow,Poc,Pk
sampleClass.loci=mito1,mito2,mito3,mito4,mito5,mito6 

sampleClass.locus.mito1.region=Pf_M76611:520-820 
sampleClass.locus.mito1.anchors=651@CCTTACGTACTCTAGCT....ACACAA
sampleClass.locus.mito1.targets=species1@668-671&678-683
sampleClass.locus.mito1.target.species1.alleles=Pf@ATGATTGTCT|ATGATTGTTT,Pv@TTTATATTAT,Pm@TTGTATTAAT,Pow@ATTTACATAA,Poc@ATTTATATAT,Pk@TTTTTATTAT

sampleClass.locus.mito2.region=Pf_M76611:600-900 
sampleClass.locus.mito2.anchors=741@GAATAGAA...GAACTCTATAAATAACCA
sampleClass.locus.mito2.targets=species2@728-733&740-740&749-751&770-773
sampleClass.locus.mito2.target.species2.alleles=Pf@GTTCATTTAAGATT|GTTCATTTAAGACT,Pv|Pk@TATTCATAAATACA,Pm@GTTCAATTAGTACT,Pow|Poc@GTTACAATAATATT

sampleClass.locus.mito3.region=Pf_M76611:720-1020 
sampleClass.locus.mito3.anchors=842@(?:GAAAGAATTTATAA|ATATA[AG]TGAATATG)ACCAT
sampleClass.locus.mito3.targets=species3@861-869&878-881&884-887
sampleClass.locus.mito3.target.species3.alleles=Pf@TCGGTAGAATATTTATT,Pv@TCACTATTACATTAACT,Pm@TCACTATTTAATATATC,Pow@CCCTTATTTAACTAACC|TCCTTATTTAACTAACC,Poc@TCGTTATTAAACTAACC,Pk@TCACAATTAAACTTATT

sampleClass.locus.mito4.region=Pf_M76611:820-1120 
sampleClass.locus.mito4.anchors=948@CCTGTAACACAATAAAATAATGT
sampleClass.locus.mito4.targets=species4@971-982
sampleClass.locus.mito4.target.species4.alleles=Pf@AGTATATACAGT,Pv|Pow|Poc@ACCAGATATAGC,Pm@TCCTGAAACTCC,Pk@ACCTGATATAGC

sampleClass.locus.mito5.region=Pf_M76611:900-1200 
sampleClass.locus.mito5.anchors=1029@GATGCAAAACATTCTCC
sampleClass.locus.mito5.targets=species5@1025-1028&1046-1049
sampleClass.locus.mito5.target.species5.alleles=Pf@TAGATAAT,Pv|Pk@AAGTAAGT,Pm@TAATAAGT,Pow@TAATAAGA,Poc@TAATAAGG

sampleClass.locus.mito6.region=Pf_M76611:950-1250
sampleClass.locus.mito6.anchors=1077@ATTTC[AT]AAACTCAT[TA]CCTTTTTCTA
sampleClass.locus.mito6.targets=species6@1062-1066&1073-1073&1076-1076&1082-1082&1091-1091&1102-1108
sampleClass.locus.mito6.target.species6.alleles=Pf@CAAATAGATTAAATAC,Pv|Pk@AATACAATTTTAGAAA|AATATAATTTTAGAAA,Pm@AATATTTAAAAAGAAA,Pow|Poc@AATATTTTTTGAGAAA|AATATTTTTTAAGAAA
''', file=fo)
fo.close() 


fo = open(chromosomeMap_fn, 'w')
print('''default
Pf3D7_01_v3
Pf3D7_02_v3
Pf3D7_03_v3
Pf3D7_04_v3
Pf3D7_05_v3
Pf3D7_06_v3
Pf3D7_07_v3
Pf3D7_08_v3
Pf3D7_09_v3
Pf3D7_10_v3
Pf3D7_11_v3
Pf3D7_12_v3
Pf3D7_13_v3
Pf3D7_14_v3
Pf_M76611
Pf3D7_API_v3''', file=fo)
fo.close()


fo = open(runArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()


# Note this was orginally run with 2GB RAM, but 32 jobs failed with TERM_MEMLIMIT errors so reran with 4GB, got
# java.lang.OutOfMemoryError then ran with 8GB
fo = open(runSpeciesArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx12000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()


fo = open(submitArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=normal
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%1000" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runArrayJob_fn,
        ),
     file=fo)
fo.close()


# Note this was orginally run with 2GB RAM, but 32 jobs failed with TERM_MEMLIMIT errors so reran with 4GB, which still gave errors
# so finally ran with 8GB here (but 4GB in -Xmx of run script)
# During 8GB run, I got an email from Sanger farm team to say I was running multi-threaded code, so reran with -n9
fo = open(submitSpeciesArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/species
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=normal
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%1000" -n9 -R"select[mem>16000] rusage[mem=16000] span[hosts=1]" -M 16000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runSpeciesArrayJob_fn,
        ),
     file=fo)
fo.close()


fo = open(mergeGrcResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()


fo = open(mergeSpeciesResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/species
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()


# # Kicking off pipeline

get_ipython().system('bash {submitArray_fn}')


get_ipython().system('bash {mergeGrcResults_fn}')


# Original run with 2GB RAM
get_ipython().system('bash {submitSpeciesArray_fn}')


# New run with 4GB RAM
get_ipython().system('bash {submitSpeciesArray_fn}')


# New run with 8GB RAM
get_ipython().system('bash {submitSpeciesArray_fn}')


# New run with 8GB RAM and -n9 (multi-threaded)
get_ipython().system('bash {submitSpeciesArray_fn}')


# New run with 8GB RAM and -n9 (multi-threaded) and -Xmx8000m
get_ipython().system('bash {submitSpeciesArray_fn}')


# New run with 12GB RAM and -n9 (multi-threaded) and -Xmx8000m
get_ipython().system('bash {submitSpeciesArray_fn}')


# Which jobs failed in the above?
get_ipython().system("grep -i 'error\\|memlimit' /lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/log/old_logs/output_8259152*")


get_ipython().system('sed "1039 q;d" {bam_list_fn}')


get_ipython().system('sed "5571 q;d" {bam_list_fn}')


get_ipython().system('sed "7743 q;d" {bam_list_fn}')


# New run with 16GB RAM and -n9 (multi-threaded) and -Xmx12000m
get_ipython().system('bash {submitSpeciesArray_fn}')





# Original run with 2GB RAM
get_ipython().system('bash {mergeSpeciesResults_fn}')


# New run with 8GB RAM and -n9 (multi-threaded)
get_ipython().system('bash {mergeSpeciesResults_fn}')


# New run with 16GB RAM and -n9 (multi-threaded) and -Xmx12000m
get_ipython().system('bash {mergeSpeciesResults_fn}')


get_ipython().system('grep -nr PA0312 {bam_list_fn}')





# # Introduction
# The purpose of this notebook is to compare the results of running Olivo's GRC tools on the "7979" bam files which form a superset of Pf5.1, and the more recent Pf6.0 release bams. Perhaps the most significant difference in these two runs is using bwa aln (5.1) vs bwa mem (6.0). I would need to look at full pipeline of 5.1 to understand other differences, e.g. was BQSR used? What about snp-o-matic?
# 
# See 20161117_run_Olivo_GRC.ipynb for details of running Olivo's GRC code on the Pf6.0 release. Olivo emailed me the results on the 7979 samples (samplesMeta5x-V1.0.xlsx) 10/11/2016 14:44

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC"
get_ipython().system('mkdir -p {output_dir}')
olivo_7979_results_fn = "%s/samplesMeta5x-V1.0.xlsx" % output_dir
richard_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"

all_calls_crosstab_fn = "%s/Pf6_vs_7979_all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/Pf6_vs_7979_discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/Pf6_vs_7979_discordant_nonmissing_calls_crosstab.xlsx" % output_dir

bwa_aln_fofn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'
bwa_mem_fofn = '/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/pf_60_mergelanes.txt'


get_ipython().system('wc -l {bwa_aln_fofn}')


get_ipython().system('wc -l {bwa_mem_fofn}')


tbl_bwa_aln = (
    etl
    .fromxlsx(olivo_7979_results_fn)
)
print(len(tbl_bwa_aln.data()))
tbl_bwa_aln


tbl_bwa_mem = (
    etl
    .fromtsv(richard_6_0_results_fn)
    .rename('mdr2_484[P]', 'mdr2_484[T]')
    .rename('fd_193[P]', 'fd_193[D]')
)
print(len(tbl_bwa_mem.data()))
tbl_bwa_mem


loci = list(tbl_bwa_mem.header()[2:])
print(len(loci))


tbl_both_results = (
    tbl_bwa_aln
    .cut(['Sample'] + loci)
    .join(tbl_bwa_mem.cut(['Sample'] + loci), key='Sample', lprefix='bwa_aln_', rprefix='bwa_mem_')
)
print(len(tbl_both_results.data()))


tbl_both_results


df_both_results = tbl_both_results.todataframe()


writer = pd.ExcelWriter(all_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i],
        df_both_results.ix[:, i+29],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0]) # Excel doesn't like the [CMNVK] endings
writer.save()


writer = pd.ExcelWriter(discordant_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        df_both_results.ix[:, i+29][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()


writer = pd.ExcelWriter(discordant_nonmissing_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        df_both_results.ix[:, i+29][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()





# ## Homozygous discordant calls
# - dhfr_108, S vs N
# - dhps_437, G vs A
# - mdr1_184, F vs Y
# 
# In the following we first identify the samples that are discordant, then look to see if there is different metadata in the bam file.

tbl_both_results.selecteq(16, 'S').selecteq(16+29, 'N')


tbl_both_results.selecteq(16, 'N').selecteq(16+29, 'S')


tbl_both_results.selecteq(19, 'G').selecteq(19+29, 'A')


tbl_both_results.selecteq(19, 'A').selecteq(19+29, 'G')


tbl_both_results.selecteq(24, 'F').selecteq(24+29, 'Y')


tbl_both_results.selecteq(24, 'Y').selecteq(24+29, 'F')


# See methods-dev/notebooks/20160621_HRP_sample_metadata.ipynb
fofns = collections.OrderedDict()

fofns['bwa_aln'] = bwa_aln_fofn
fofns['bwa_mem'] = bwa_mem_fofn
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '/nfs/team112_internal/rp7/data/Pf/hrp/fofns/olivo_TRAC.fofn'
fofns['fanello'] = '/nfs/team112_internal/rp7/data/Pf/hrp/fofns/olivo_fanello.fofn'

for fofn in fofns:
    print(fofn)
    get_ipython().system('grep PG0282 {fofns[fofn]}')


def show_rg(sample='PG0282'):
    line = get_ipython().getoutput('grep {sample} {bwa_aln_fofn}')
    bam_fn = line[0].split('\t')[0]
    rg = get_ipython().getoutput('samtools view -H {bam_fn} | grep RG')
    print(rg[0])
    line = get_ipython().getoutput('grep {sample} {bwa_mem_fofn}')
    bam_fn = line[0].split('\t')[0]
    rg = get_ipython().getoutput('samtools view -H {bam_fn} | grep RG')
    print(rg[0])


for sample_id in ['PG0282-C', 'PG0304-C', 'PG0312-C', 'PG0313-C', 'PG0330-C', 'PG0332-C', 'PG0334-C', 'PG0335-C']:
    print(sample_id)
    show_rg(sample_id)


# # Conclusion
# 
# There were four pairs of samples ('PG0282-C', 'PG0304-C', 'PG0312-C', 'PG0313-C') and ('PG0330-C', 'PG0332-C', 'PG0334-C', 'PG0335-C') which had discordant results at dhfr_108, dhps_437 and mdr1_184. The above analysis shows that these have different read group IDs, suggested sample metadata has been swapped at some point. I have followed these up with Jim, and it seems that the core team at Sanger modified some metadata without letting Jim know.
# 
# Manual inspection of the output crosstab spreadsheets shows that, other than these sample swaps there were no homozygous call discordances between 5.1 and 6.0. There were small numbers of discordances in the following categories: a) Hom vs het b) upper case vs lower case, c) different ordering of alleles, d) two identical alleles (e.g. "S,S") vs one allele (e.g. "S"), and e) called vs missing. Levels of missingness were very similar between the two runs. For most loci missingness was very slightly higher in Pf 6.0, but for some loci was slightly higher in 5.1. However, there was nothing dramatic, for example the most extreme difference was at crt_144 where 24 sample were called in 5.1 but not 6.0 (19 as "a" and 5 as "A"), whereas 7 samples were called in 6.0 but not in 5.1 (6 as "A" and 1 as "a"). All in all, I think we can safely assume that the results from running on bwa aln or bwa mem mapped reads are essentially the same.
# 




# # Introduction
# 
# At the time of writing 85 samples had failed pf_60_haplotype_caller setup, and a further 16 were still running. This is thought to be due to a known bug in GATK
# 
# Jim sent ox_codes for these and whether they were in 5.1. This notebook checks if the 101 failures are included in Pf3k pilot 5.0 (and hence ran fine with GATK 3.4), and summarises by study

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


results_dir = '/nfs/team112_internal/rp7/data/methods-dev/analysis/20161005_Pf_60_failures'
get_ipython().system('mkdir {results_dir}')

jim_fn = '%s/pf_60_fails_studies.tab' % results_dir
results_fn = '%s/pf_60_failures_summary.xlsx' % results_dir
pf3k_vcf_fn = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_Pf3D7_01_v3.combined.filtered.vcf.gz'


# vcf_reader = vcf.Reader(open(pf3k_vcf_fn, 'rz'))
vcf_reader = vcf.Reader(filename=pf3k_vcf_fn)
pf3k_samples = vcf_reader.samples


samples_line = get_ipython().getoutput("zcat /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_Pf3D7_01_v3.combined.filtered.vcf.gz | head -n 500 | grep '#CHROM'")


pf3k_samples = samples_line[0].split('\t')[9:]
print(len(pf3k_samples))
pf3k_samples[0:10]


tbl_failed_samples = (
    etl
    .fromtsv(jim_fn)
    .pushheader(['study', 'ox_code', 'in_pf_5.1'])
    .convert('in_pf_5.1', lambda x: bool(int(x)))
    .addfield('in_pf3k', lambda rec: rec[1] in pf3k_samples)
)
print(len(tbl_failed_samples.data()))
tbl_failed_samples.displayall()


tbl_failed_sample_summary = (
    tbl_failed_samples
#     .valuecounts('study', 'in_pf_5.1', 'in_pf3k')
    .valuecounts('study', 'in_pf_5.1')
    .cutout('frequency')
    .rename('count', 'Number of samples')
#     .sort(('study', 'in_pf_5.1', 'in_pf3k'))
    .sort(('study', 'in_pf_5.1'))
)


tbl_failed_sample_summary.displayall()


tbl_failed_sample_summary.toxlsx(results_fn)





# #Plan
# - Similar to 20160720_mendelian_error_duplicate_concordance but
# - Classify as SNP/INDEL before deciding which are multiallelic

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
crosses_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)
sites_only_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.sites.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160721_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'
gff_fn = "%s/Pfalciparum.noseq.gff3.gz" % output_dir
cds_gff_fn = "%s/Pfalciparum.noseq.gff3.cds.gz" % output_dir

results_table_fn = "%s/genotype_quality.xlsx" % output_dir
counts_table_fn = "%s/variant_counts.xlsx" % output_dir
simplifed_counts_table_fn = "%s/simplified_variant_counts.xlsx" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
GATK = '/software/jre1.7.0_25/bin/java -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.4-46/GenomeAnalysisTK.jar'


gff_fn


get_ipython().system('wget ftp://ftp.sanger.ac.uk/pub/project/pathogens/gff3/2016-06/Pfalciparum.noseq.gff3.gz     -O {gff_fn}')


get_ipython().system("zgrep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")


crosses_vcf_fn


multiallelic_header_fn = "%s/vcf/MULTIALLELIC.hdr" % (output_dir)
fo=open(multiallelic_header_fn, 'w')
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()


variant_type_header_fn = "%s/vcf/VARIANT_TYPE.hdr" % (output_dir)
fo=open(variant_type_header_fn, 'w')
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (IND)">', file=fo)
fo.close()


cds_header_fn = "%s/vcf/CDS.hdr" % (output_dir)
fo=open(cds_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
fo.close()


def create_analysis_vcf(input_vcf_fn=crosses_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['nonref', 'multiallelic', 'triallelic', 'bi_allelic', 'spanning_deletion', 'triallelic_no_sd', 'multiallelics',
                              'biallelic', 'str', 'snps', 'indels', 'strs', 'variant_type', 'coding', 'analysis',
                             'site_snps', 'site_indels', 'site_variant_type', 'site_analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)
    
    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        if region is not None:
            get_ipython().system('{BCFTOOLS} annotate --regions {region} --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        else:
            get_ipython().system('{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'nonref\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'nonref\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

#     if rewrite or not os.path.exists(intermediate_fns['strs']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tSTR\n' --include 'TYPE!="snp" && STR=1' {intermediate_fns['str']} | bgzip -c > {intermediate_fns['strs']} && tabix -s1 -b2 -e2 -f {intermediate_fns['strs']}

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['nonref']} |        {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

        #         {BCFTOOLS} annotate -a {intermediate_fns['strs']} -c CHROM,POS,INFO/VARIANT_TYPE \

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {intermediate_fns['variant_type']} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")


create_analysis_vcf(region=None)


output_dir


tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


# Note the version created in this notebook doesn't work. I think this is because of R in Number of FORMAT field for AD,
# which is part of spec for v4.2, but think GATK must have got rid of this in previous notebook
analysis_vcf_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_mendelian_error_duplicate_concordance/vcf/SNP_INDEL_WG.analysis.vcf.gz"
vcf_reader = vcf.Reader(filename=analysis_vcf_fn)
sample_ids = np.array(vcf_reader.samples)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
                'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
                'VQSLOD', 'FILTER'],
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'VARIANT_TYPE':             'a3',
            'RU':                       'a40',
            'SNPEFF_EFFECT':            'a33',
            'CDS':                      bool,
            'MULTIALLELIC':             'a2',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
        },
        flatten_filter=True,
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD', 'DP', 'GQ'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )


create_variants_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))
create_calldata_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))


analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
variants = np.load("%s.vcfnp_cache/variants.npy" % analysis_vcf_fn)
calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % analysis_vcf_fn)


print(np.unique(variants['VARIANT_TYPE'], return_counts=True))


print(np.unique(variants['VARIANT_TYPE'], return_counts=True))


np.unique(variants['MULTIALLELIC'], return_counts=True)


np.unique(variants['MULTIALLELIC'], return_counts=True)


def genotype_concordance_gatk(calldata=calldata,
                              ix = ((variants['VARIANT_TYPE'] == b'SNP') & (variants['MULTIALLELIC'] == b'BI') &
                                    (variants['CDS']) & variants['FILTER_PASS']),
                              GQ_threshold=30,
                              rep_index_first=rep_index_first, rep_index_second=rep_index_second,
                              verbose=False):
    GT = calldata['GT'][ix, :]
    GT[calldata['GQ'][ix, :] < GQ_threshold] = b'./.'
    
    all_samples = sample_ids
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = GT[:, all_samples==parents[0]]
        parent2_calls = GT[:, all_samples==parents[1]]
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = GT[:, all_samples==ox_code]
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    GT_both = (np.in1d(GT[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(GT[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    GT_both = (
        ((GT[:, rep_index_first] == b'0/0') | (GT[:, rep_index_first] == b'1/1')) &
        ((GT[:, rep_index_second] == b'0/0') | (GT[:, rep_index_second] == b'1/1'))
    )
    GT_discordant = (
        ((GT[:, rep_index_first] == b'0/0') & (GT[:, rep_index_second] == b'1/1')) |
        ((GT[:, rep_index_first] == b'1/1') & (GT[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(GT == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (GT.shape[0] * GT.shape[1])
    heterozygosity_per_sample = np.sum(GT == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (GT.shape[0] * GT.shape[1])
    num_discordance_per_sample_pair = np.sum(GT_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(GT_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(GT_discordant, 1)
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    num_of_alleles = np.sum(ix)
    return(
#         num_of_alleles,
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
#         prop_discordances_per_sample_pair,
#         GT.shape
    )
    


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


results_list = list()
# GQ_thresholds = [30, 99, 0]
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

for GQ_threshold in GQ_thresholds:
    for filter_pass in filter_passes:
        for variant_type in variant_types:
            for coding in codings:
                for multiallelic in multiallelics:
                    print(GQ_threshold, filter_pass, variant_type, coding, multiallelic)
                    ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
                    number_of_alleles = np.sum(ix)
                    number_of_sites = len(np.unique(variants[['CHROM', 'POS']][ix]))
                    mean_nraf = np.sum(variants['AC'][ix]) / np.sum(variants['AN'][ix])
                    genotype_quality_results = list(genotype_concordance_gatk(ix=ix, GQ_threshold=GQ_threshold))
#                     sites_ix = (
#                         (site_variants['VARIANT_TYPE'] == variant_type) &
#                         (site_variants['MULTIALLELIC'] == multiallelic) &
#                         (site_variants['CDS'] == coding) &
#                         (site_variants['FILTER_PASS'] == filter_pass)
#                     )
#                     num_sites = np.sum(sites_ix)
                    results_list.append(
                        [GQ_threshold, filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf] +
                        genotype_quality_results
                    )

# print(results_list)


# Sanity check. Previously this was showing 2 variants, which was due to a bug
variant_type = b'SNP'
multiallelic = b'BI'
coding = False
filter_pass = True
ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
temp = variants[['CHROM', 'POS']][ix]
s = np.sort(temp, axis=None)
s[s[1:] == s[:-1]]


# Sanity check. Previously this was 1 of the two variants shown above and multiallelic was b'BI' not b'MU'
variants[(variants['CHROM']==b'Pf3D7_01_v3') & (variants['POS']==514753)]


headers = ['GQ threshold', 'PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF', 'Missingness',
           'Heterozygosity', 'Discordance', 'MER']
etl.wrap(results_list).pushheader(headers).displayall()


np.sum(etl.wrap(results_list).pushheader(headers).values('Variants').array())


np.sum(etl.wrap(results_list).pushheader(headers).values('Variants').array())





# etl.wrap(results_list).pushheader(headers).convert('Alleles', int).toxlsx(results_table_fn)
etl.wrap(results_list).pushheader(headers).cutout('Alleles').cutout('Mean NRAF').toxlsx(results_table_fn)
results_table_fn








# variant_type_header_2_fn = "%s/vcf/VARIANT_TYPE_2.hdr" % (output_dir)
# fo=open(variant_type_header_2_fn, 'w')
# print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or INDEL">', file=fo)
# fo.close()


def create_variant_counts_vcf(input_vcf_fn=sites_only_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.sites.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['multiallelic', 'multiallelics', 'snps', 'indels', 'triallelic', 'bi_allelic', 'spanning_deletion',
                              'triallelic_no_sd', 'biallelic',
                              'variant_type', 'analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {input_vcf_fn} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {input_vcf_fn} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {input_vcf_fn} |         {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {intermediate_fns['variant_type']} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set        -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {intermediate_fns['variant_type']} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
                
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")


sites_only_vcf_fn


create_variant_counts_vcf(region=None)


create_variants_npy("%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir))


sites_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
variants_all = np.load("%s.vcfnp_cache/variants.npy" % sites_vcf_fn)


counts_list = list()
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

# GQ_thresholds = [30, 99, 0]
# variant_types = [b'SNP', b'IND']
# multiallelics = [False, True]
# codings = [True, False]
# filter_passes = [True, False]


for filter_pass in filter_passes:
    for variant_type in variant_types:
        for coding in codings:
            for multiallelic in multiallelics:
                print(filter_pass, variant_type, coding, multiallelic)
                ix = (
                    (variants_all['VARIANT_TYPE'] == variant_type) &
                    (variants_all['MULTIALLELIC'] == multiallelic) &
                    (variants_all['CDS'] == coding) &
                    (variants_all['FILTER_PASS'] == filter_pass)
                )
                number_of_alleles = np.sum(ix)
                number_of_sites = len(np.unique(variants_all[['CHROM', 'POS']][ix]))
                mean_nraf = np.sum(variants_all['AC'][ix]) / np.sum(variants_all['AN'][ix])
#                 number_of_variants = np.sum(ix)
                counts_list.append(
                    [filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf]
                )

print(counts_list)


headers = ['PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF']
(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .displayall()
)


(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .toxlsx(counts_table_fn)
)
# etl.wrap(counts_list).pushheader(headers).convertnumbers().toxlsx(counts_table_fn)
counts_table_fn


np.sum(etl
       .fromtsv('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/passcounts')
       .pushheader(('CHROM', 'count'))
       .convertnumbers()
       .values('count')
       .array()
       )


simplified_counts_list = list()
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

for filter_pass in filter_passes:
    for variant_type in variant_types:
        for coding in codings:
            print(filter_pass, variant_type, coding)
            ix = (
                (variants_all['VARIANT_TYPE'] == variant_type) &
                (variants_all['CDS'] == coding) &
                (variants_all['FILTER_PASS'] == filter_pass)
            )
            number_of_alleles = np.sum(ix)
            number_of_sites = len(np.unique(variants_all[['CHROM', 'POS']][ix]))
            mean_nraf = np.sum(variants_all['AC'][ix]) / np.sum(variants_all['AN'][ix])
            simplified_counts_list.append(
                [filter_pass, variant_type, coding, number_of_sites, number_of_alleles, mean_nraf]
            )


headers = ['PASS', 'Type', 'Coding', 'Variants', 'Alleles', 'Mean NRAF']
(etl
 .wrap(simplified_counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .displayall()
)


headers = ['PASS', 'Type', 'Coding', 'Variants', 'Alleles', 'Mean NRAF']
(etl
 .wrap(simplified_counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .toxlsx(simplifed_counts_table_fn)
)


np.sum(etl
       .fromtsv('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/passcounts')
       .pushheader(('CHROM', 'count'))
       .convertnumbers()
       .values('count')
       .array()
       )


2+2





# # Plan
# - dict of boolean array of core genome
# - for each sample determine % core callable
# - plot histogram of this
# - dict of number of samples callable at each position
# - for each well-covered samples, add to number of callable samples
# - plot this genome-wide

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


# see 20160525_CallableLoci_bed_release_5.ipynb
callable_loci_bed_fn_format = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_%s.bed"

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160713_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'


etl.fromtsv(REGIONS_FN).pushheader('chrom', 'start', 'end', 'region')


core_genome_dict = collections.OrderedDict()
for chrom in ['Pf3D7_%02d_v3' % i for i in range(1, 15)]:
    this_chrom_regions = (etl
                          .fromtabix(core_regions_fn, chrom)
                          .pushheader('chrom', 'start', 'end', 'region')
                          .convertnumbers()
                          )
    chrom_length = np.max(this_chrom_regions.convert('end', int).values('end').array())
    core_genome_dict[chrom] = np.zeros(chrom_length, dtype=bool)
    for rec in this_chrom_regions:
        if rec[3] == 'Core':
            core_genome_dict[chrom][rec[1]:rec[2]] = True


core_genome_length = 0
for chrom in core_genome_dict:
    print(chrom, len(core_genome_dict[chrom]), np.sum(core_genome_dict[chrom]))
    core_genome_length = core_genome_length + np.sum(core_genome_dict[chrom])
print(core_genome_length)


tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)


tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])


len(tbl_field_samples.data())


bases_callable = collections.OrderedDict()
core_bases_callable = collections.OrderedDict()
autosomes = ['Pf3D7_%02d_v3' % i for i in range(1, 15)]
for i, ox_code in enumerate(tbl_field_samples.values('sample')):
#     print(i, ox_code)
    this_sample_callable_loci = collections.OrderedDict()
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    for chrom in core_genome_dict.keys():
        chrom_length = len(core_genome_dict[chrom])
        this_sample_callable_loci[chrom] = np.zeros(chrom_length, dtype=bool)
    tbl_this_sample_callable_loci = (etl
                                     .fromtsv(callable_loci_bed_fn)
                                     .pushheader('chrom', 'start', 'end', 'region')
                                     .selecteq('region', 'CALLABLE')
                                     .selectin('chrom', autosomes)
                                     .convertnumbers()
                                    )
    for rec in tbl_this_sample_callable_loci.data():
        this_sample_callable_loci[rec[0]][rec[1]:rec[2]] = True
    bases_callable[ox_code] = 0
    core_bases_callable[ox_code] = 0
    for chrom in core_genome_dict.keys():
        bases_callable[ox_code] = bases_callable[ox_code] + np.sum(this_sample_callable_loci[chrom])
        core_bases_callable[ox_code] = core_bases_callable[ox_code] + np.sum((this_sample_callable_loci[chrom] & core_genome_dict[chrom]))
#     print(ox_code, bases_callable, core_bases_callable)
#     print(i, type(i))
    print('%d' % (i%10), end='', flush=True)
    
        


20296931 / 20782107 


20782107 * 0.95


proportion_core_callable = np.array(core_bases_callable.values())


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.0, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram.pdf" % plot_dir)


WG_VCF_FN = "/nfs/team112_internal/production_files/Pf/1147_Conway/conway_5_1_annot_gt.vcf.gz"
FINAL_VCF_FN = "/nfs/team112_internal/production_files/Pf/1147_Conway/conway_5_1_annot_gt_nonref.vcf.gz"
# BCFTOOLS = 'bcftools'


get_ipython().system('tabix -p vcf {WG_VCF_FN}')
get_ipython().system('md5sum {WG_VCF_FN} > {WG_VCF_FN}.md5')


get_ipython().system("bcftools view --include 'AC>0' --output-type z --output-file {FINAL_VCF_FN} {WG_VCF_FN}")
get_ipython().system('bcftools index --tbi {FINAL_VCF_FN}')
get_ipython().system('md5sum {FINAL_VCF_FN} > {FINAL_VCF_FN}.md5')


number_of_variants = get_ipython().getoutput("bcftools query -f '%CHROM\\t%POS\\n' {FINAL_VCF_FN} | wc -l")
number_of_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'TYPE="snp"\' {FINAL_VCF_FN} | wc -l')
number_of_ref_only = get_ipython().getoutput("bcftools query -f '%CHROM\\t%POS\\n' --include 'N_ALT=0' {FINAL_VCF_FN} | wc -l")
number_of_pass_variants = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_ref_only = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && N_ALT=0\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_variants = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && N_ALT=1\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_ref_only = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1 && N_ALT=0\' {FINAL_VCF_FN} | wc -l')

print("%s variants" % ("{:,}".format(int(number_of_variants[0]))))
print("%s SNPs" % ("{:,}".format(int(number_of_snps[0]))))
print("%s ref only" % ("{:,}".format(int(number_of_ref_only[0]))))
print()
print("%s PASS variants" % ("{:,}".format(int(number_of_pass_variants[0]))))
print("%s PASS SNPs" % ("{:,}".format(int(number_of_pass_snps[0]))))
print("%s PASS ref only" % ("{:,}".format(int(number_of_pass_ref_only[0]))))
print()
print("%s PASS biallelic variants" % ("{:,}".format(int(number_of_pass_biallelic_variants[0]))))
print("%s PASS biallelic SNPs" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]))))
print("%s PASS biallelic ref only" % ("{:,}".format(int(number_of_pass_biallelic_ref_only[0]))))
print()
     


number_of_pass_inc_noncoding_variants = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER!="Biallelic" && FILTER!="HetUniq" && FILTER!="HyperHet" && FILTER!="MaxCoverage" && FILTER!="MinAlt" && FILTER!="MinCoverage" && FILTER!="MonoAllelic" && FILTER!="NoAltAllele" && FILTER!="Region" && FILTER!="triallelic"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_inc_noncoding_variants


number_of_hq_noncoding_variants = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER!="PASS" && FILTER!="Biallelic" && FILTER!="HetUniq" && FILTER!="HyperHet" && FILTER!="MaxCoverage" && FILTER!="MinAlt" && FILTER!="MinCoverage" && FILTER!="MonoAllelic" && FILTER!="NoAltAllele" && FILTER!="Region" && FILTER!="triallelic"\' {FINAL_VCF_FN} | wc -l')
number_of_hq_noncoding_variants


number_of_samples = get_ipython().getoutput('bcftools query --list-samples {FINAL_VCF_FN} | wc -l')
number_of_samples


print('''
===================================================================================
MalariaGEN P. falciparum Community Project - Biallelic SNP genotypes for study 1147
===================================================================================

Date: 2017-01-30


Description 
-----------

Through an analysis of 3,394 parasite samples collected at 42 different locations in Africa, Asia, America and Oceania, we identified single nucleotide polymorphisms (SNPs). This download includes genotyping data for samples contributed to the MalariaGEN Plasmodium falciparum Community Project under study 1147 DC-MRC-Mauritania that were genotyped at these SNPs.

Potential data users are asked to respect the legitimate interest of the Community Project and its partners by abiding any restrictions on the use of a data as described in the Terms of Use: http://www.malariagen.net/projects/parasite/pf/use-p-falciparum-community-project-data

For more information on the P. falciparum Community Project that generated these data, please visit: https://www.malariagen.net/projects/p-falciparum-community-project

Genotyping data is currently released for all identified biallelic single nucleotide polymorphisms (SNPs) that are segregating amongst the {number_of_samples} samples of study 1147. 

The methods used to generate the data are described in detail in MalariaGEN Plasmodium falciparum Community Project, eLife (2016), DOI: 10.7554/eLife.08714.

This data was created as an ad-hoc build and hasn't been quality assessed by the MalariaGEN team.


Citation information
--------------------

Publications using these data should acknowledge and cite the source of the data using the following format: "This publication uses data from the MalariaGEN Plasmodium falciparum Community Project as described in Genomic epidemiology of artemisinin resistant malaria, eLife, 2016 (DOI: 10.7554/eLife.08714)."


File descriptions
-----------------

- conway_5_1_annot_gt_nonref.vcf.gz

The data file ("*.vcf.gz") is a zipped VCF format file containing all samples in the study.  The file, once unzipped, is a tab-separated text file, but may be too big to open in Excel.  

The format is described in https://github.com/samtools/hts-specs

Tools to assist in handling VCF files are freely available from
https://vcftools.github.io/index.html
http://samtools.github.io/bcftools/

- conway_5_1_annot_gt_nonref.vcf.gz.tbi

This is a tabix index file for conway_5_1_annot_gt_nonref.vcf.gz

Further details on tabix indexes are available at
http://www.htslib.org/doc/tabix.html

- conway_5_1_annot_gt_nonref.vcf.gz.md5

This is an MD5 checksum for conway_5_1_annot_gt_nonref.vcf.gz


Contents of the VCF file
------------------------

The VCF file contains details of {number_of_variants} SNPs in {number_of_samples} samples. These are all the SNPs discovered in the MalariaGEN 5.0 release to partners that are segregating and biallelic in the {number_of_samples} samples.

It is important to note that many of these SNPs are considered low quality. Only the variants for which the FILTER column is set to PASS should be considered of high quality. There are {number_of_pass_variants} such high-quality PASS SNPs. Note that this set only includes coding SNPs (those in exons). There are an additional {number_of_hq_noncoding_variants} SNPs that are in non-coding regions but which pass all other variant filters.

Columns 10 and onwards of the VCF contain the information for each sample. The first component of this (GT) is the genotype call. A value of 0 indicates a homozygous reference call (at least 5 reads in total and <= 1 read with alternative allele). A value of 1 indicates a homozygous alternative call (at least 5 reads in total and <= 1 read with reference allele). A value of 0/1 indicates the sample has a heterozygous call (at least 5 reads in total, >=2 reads with reference allele and >=2 reads with alternative allele). A value of . indicates a missing genotype call (<5 reads in total).

'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0])),
        number_of_samples="{:,}".format(int(number_of_samples[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0])),
        number_of_hq_noncoding_variants="{:,}".format(int(number_of_hq_noncoding_variants[0])),
    )
)


# Copy and pasted above 2 cells into /nfs/team112_internal/production_files/Pf/1147_Conway/README_for_FTP then ran following code:




# # Introduction
# This notebook is a test of scikit-allel on the Pf 6.0 release HDF5 file. It is stolen from http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html (https://github.com/alimanfoo/alimanfoo.github.io/blob/master/_posts/2016-06-10-scikit-allel-tour.ipynb)

# ## Setup
# 
# Let's import the libraries we'll be using.

import numpy as np
import scipy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import h5py
import allel; print('scikit-allel', allel.__version__)
import collections


# The data we'll be analysing originally came from a [VCF format file](https://en.wikipedia.org/wiki/Variant_Call_Format), however these data have previously been pre-processed into an [HDF5 file](https://www.hdfgroup.org/HDF5/) which improves data access speed for a range of access patterns. I won't cover this pre-processing step here, for more information see [vcfnp](https://github.com/alimanfoo/vcfnp). 
# 
# Open an HDF5 file containing variant calls from the [P. falciparum Community Project](https://www.malariagen.net/projects/p-falciparum-community-project). Note that this contains **raw data**, i.e., all putative SNPs and indels are included. As part of the PfCP we have done a detailed analysis of data quality and so this dataset already contains filter annotations to tell you which SNPs we believe are real. However, for the purpose of this tour, I am going to pretend that we have never seen these data before, and so need to perform our own analysis of data quality.

callset_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5'
callset = h5py.File(callset_fn, mode='r')
callset


# ## Visualize variant density
# 
# As a first step into getting to know these data, let's make a plot of variant density, which will simply show us how many SNPs there are and how they are distributed along the chromosome.
# 
# The data on SNP positions and various other attributes of the SNPs are stored in the HDF5 file. Each of these can be treated as a column in a table, so let's set up a table with some of the columns we'll need for this and subsequent analyses.

variants = allel.VariantChunkedTable(callset['variants'], 
                                     names=['CHROM', 'POS', 'REF', 'ALT', 'DP', 'MQ', 'QD', 'num_alleles', 'is_snp'],
                                     index=['CHROM', 'POS'])
variants


np.unique(callset['variants']['is_snp'][:], return_counts=True)


# The caption for this table tells us that we have 6,051,695 variants (rows) in this dataset.
# 
# Now let's extract the variant positions and load into a numpy array.

chrom = variants['CHROM'][:]
chrom


chroms = np.unique(chrom)
chroms


pos = variants['POS'][:]
pos


# Define a function to plot variant density in windows over the chromosome.

def plot_windowed_variant_density(pos, window_size, title=None):
    
    # setup windows 
    bins = np.arange(0, pos.max(), window_size)
    
    # use window midpoints as x coordinate
    x = (bins[1:] + bins[:-1])/2
    
    # compute variant density in each window
    h, _ = np.histogram(pos, bins=bins)
    y = h / window_size
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y)
    ax.set_xlabel('Chromosome position (bp)')
    ax.set_ylabel('Variant density (bp$^{-1}$)')
    if title:
        ax.set_title(title)


# Now make a plot with the SNP positions from our chosen chromosome.

for current_chrom in chroms:
    plot_windowed_variant_density(
        pos[chrom==current_chrom], window_size=1000, title='Raw variant density %s' % current_chrom.decode('ascii')
    )


# From this we can see that variant density is around 0.2 over much of the genome, which means the raw data contains a variant about every 5 bases of the reference genome. Variant density much higher in var gene regions, as expected, with almost every base being variant.

# ## Explore variant attributes
# 
# As I mentioned above, each variant also has a number "annotations", which are data attributes that originally came from the "INFO" field in the VCF file. These are important for data quality, so let's begin by getting to know a bit more about the numerical range and distribution of some of these attributes.
# 
# Each attribute can be loaded from the table we setup earlier into a numpy array. E.g., load the "DP" field into an array.

dp = variants['DP'][:]
dp


# Define a function to plot a frequency distribution for any variant attribute.

def plot_variant_hist(f, bins=30):
    x = variants[f][:]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')
    ax.set_title('Variant %s distribution' % f)


# "DP" is total depth of coverage across all samples.

plot_variant_hist('DP', bins=50)


# "MQ" is average mapping quality across all samples.

plot_variant_hist('MQ')


# "QD" is a slightly odd statistic but turns out to be very useful for finding poor quality variants. Roughly speaking, high numbers mean that evidence for variation is strong (concentrated), low numbers mean that evidence is weak (dilute).

plot_variant_hist('QD')


# Finally let's see how many biallelic, triallelic, quadriallelic, etc variants we have.

plot_variant_hist('num_alleles', bins=np.arange(1.5, 8.5, 1))
plt.gca().set_xticks([2, 3, 4, 5, 6, 7]);


# We can also look at the joint frequency distribution of two attributes.

def plot_variant_hist_2d(f1, f2, downsample):
    x = variants[f1][:][::downsample]
    y = variants[f2][:][::downsample]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.despine(ax=ax, offset=10)
    ax.hexbin(x, y, gridsize=40)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s joint distribution' % (f1, f2))


# To make the plotting go faster I've downsampled to use every 10th variant.

plot_variant_hist_2d('QD', 'MQ', downsample=10)


# ## Investigate variant quality
# 
# The DP, MQ and QD attributes are potentially informative about SNP quality. For example, we have a prior expectation that putative SNPs with very high or very low DP may coincide with some form of larger structural variation, and may therefore be unreliable. However, it would be great to have some empirical indicator of data quality, which could guide our choices about how to filter the data.
# 
# There are several possible quality indicators that could be used, and in general it's a good idea to use more than one if available. Here, to illustrate the general idea, let's use just one indicator, which is the number of [transitions]() divided by the number of [transversions](), which I will call Ti/Tv.
# 
# ![Transitions and transversions](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Transitions-transversions-v4.svg/500px-Transitions-transversions-v4.svg.png)
# 
# If mutations were completely random we would expect a Ti/Tv of 0.5, because there are twice as many possible transversions as transitions. However, in most species a mutation bias has been found towards transitions, and so we expect the true Ti/Tv to be higher. We can therefore look for features of the raw data that are associated with low Ti/Tv (close to 0.5) and be fairly confident that these contain a lot of noise. 
# 
# To do this, let's first set up an array of mutations, where each entry contains two characters representing the reference and alternate allele. For simplicity of presentation I'm going to ignore the fact that some SNPs are multiallelic, but if doing this for real this should be restricted to biallelic variants only.

mutations = np.char.add(variants['REF'].subset(variants['is_snp']), variants['ALT'].subset(variants['is_snp'])[:, 0])
mutations


# Define a function to locate transition mutations within a mutations array.

def locate_transitions(x):
    x = np.asarray(x)
    return (x == b'AG') | (x == b'GA') | (x == b'CT') | (x == b'TC')


# Demonstrate how the ``locate_transitions`` function generates a boolean array from a mutations array.

is_ti = locate_transitions(mutations)
is_ti


# Define a function to compute Ti/Tv.

def ti_tv(x):
    if len(x) == 0:
        return np.nan
    is_ti = locate_transitions(x)
    n_ti = np.count_nonzero(is_ti)
    n_tv = np.count_nonzero(~is_ti)
    if n_tv > 0:
        return n_ti / n_tv
    else:
        return np.nan


# Demonstrate the ``ti_tv`` function by computing Ti/Tv over all SNPs.

ti_tv(mutations)


# Define a function to plot Ti/Tv in relation to a variant attribute like DP or MQ.

def plot_ti_tv(f, downsample, bins):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    x = variants[f].subset(variants['is_snp'])[:][::downsample]
    
    # plot a histogram
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')

    # plot Ti/Tv
    ax = ax.twinx()
    sns.despine(ax=ax, bottom=True, left=True, right=False, offset=10)
    values = mutations[::downsample]
    with np.errstate(over='ignore'):
        # binned_statistic generates an annoying overflow warning which we can ignore
        y1, _, _ = scipy.stats.binned_statistic(x, values, statistic=ti_tv, bins=bins)
    bx = (bins[1:] + bins[:-1]) / 2
    ax.plot(bx, y1, color='k')
    ax.set_ylabel('Ti/Tv')
    ax.set_ylim(0.6, 1.3)

    ax.set_title('Variant %s and Ti/Tv' % f)


# Example the relationship between the QD, MQ and DP attributes and Ti/Tv. 

plot_ti_tv('QD', downsample=5, bins=np.arange(0, 40, 1))


plot_ti_tv('MQ', downsample=5, bins=np.arange(0, 60, 1))


plot_ti_tv('DP', downsample=5, bins=np.linspace(0, 50000, 50))


# Ti/Tv is not a simple variable and so some care is required when interpreting these plots. However, we can see that there is a trend towards low Ti/Tv for low values of QD, MQ and DP.
# 
# To investigate further, let's look at Ti/Tv in two dimensions. 

def plot_joint_ti_tv(f1, f2, downsample, gridsize=20, mincnt=20, vmin=0.6, vmax=1.4, extent=None):
    fig, ax = plt.subplots()
    sns.despine(ax=ax, offset=10)
    x = variants[f1].subset(variants['is_snp'])[:][::downsample]
    y = variants[f2].subset(variants['is_snp'])[:][::downsample]
    C = mutations[::downsample]
    im = ax.hexbin(x, y, C=C, reduce_C_function=ti_tv, mincnt=mincnt, extent=extent,
                   gridsize=gridsize, cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s and Ti/Tv' % (f1, f2))


plot_joint_ti_tv('QD', 'MQ', downsample=5, mincnt=400, extent=(0, 40, 0, 80))


plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400, extent=(0, 40, 0, 8e+5))
# plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400)


plot_joint_ti_tv('MQ', 'DP', downsample=5, mincnt=400, extent=(0, 80, 0, 8e+5))


# This information may be useful when designing a variant filtering strategy. If you have other data that could be used as a quality indicator, such as Mendelian errors in a trio or cross, and/or data on genotype discordances between replicate samples, a similar analysis could be performed.

# ## Filtering variants
# 
# There are many possible approaches to filtering variants. The simplest approach is define thresholds on variant attributes like DP, MQ and QD, and exclude SNPs that fall outside of a defined range (a.k.a. "hard filtering"). This is crude but simple to implement and in many cases may suffice, at least for an initial exploration of the data. 
# 
# Let's implement a simple hard filter. First, a reminder that we have a table containing all these variant attributes.

variants


# Define the hard filter using an expression. This is just a string of Python code, which we will evaluate in a moment.

filter_expression = '(QD > 5) & (MQ > 40) & (DP > 3e+5) & (DP < 8e+5)'


# Now evaluate the filter using the columns from the table

variant_selection = variants.eval(filter_expression)[:]
variant_selection


# How many variants to we keep?

np.count_nonzero(variant_selection)


# How many variants do we filter out?

np.count_nonzero(~variant_selection)


# Now that we have our variant filter, let's make a new variants table with only rows for variants that pass our filter.

variants_pass = variants.compress(variant_selection)
variants_pass


# ## Cleanup to reduce RAM usage

import gc
gc.collect()


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))


del(variants)
del(mutations)
del(pos)
del(dp)
del(chrom)
del(_3)
del(_31)
del(_18)
del(_7)
del(_10)
del(_5)
gc.collect()


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))


# 
# 
# ## Subset genotypes
# 
# Now that we have some idea of variant quality, let's look at our samples and at the genotype calls.
# 
# All data relating to the genotype calls is stored in the HDF5.

calldata = callset['calldata']
calldata


list(calldata.keys())


# Each of these is a separate dataset in the HDF5 file. To make it easier to work with the genotype dataset, let's wrap it using a class from scikit-allel.

genotypes = allel.GenotypeChunkedArray(calldata['genotype'])
genotypes


# N.B., at this point we have not loaded any data into memory, it is still in the HDF5 file. From the representation above we have some diagnostic information about the genotypes, for example, we have calls for 6,051,695 variants in 7,182 samples with ploidy 2 (i.e., diploid). Uncompressed these data would be 81.0G but the data are compressed and so actually use 6.4G on disk.
# 
# We can also see genotype calls for the last 3 variants in the first and last 5 samples, which are all missing ("./.").
# 
# Before we go any furter, let's also pull in some data about the 7,182 samples we've genotyped.

samples_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt'
samples = pandas.DataFrame.from_csv(samples_fn, sep='\t')
samples.head()


# The "study" column defines which of 54 studies the parasites came from. How many parasites come from each of these studies?

samples.study.value_counts()


# These don't tell us much about geography, so let's assign each study to an approximate continental grouping

continents = collections.OrderedDict()
continents['1001-PF-ML-DJIMDE']               = '1_WA'
continents['1004-PF-BF-OUEDRAOGO']            = '1_WA'
continents['1006-PF-GM-CONWAY']               = '1_WA'
continents['1007-PF-TZ-DUFFY']                = '2_EA'
continents['1008-PF-SEA-RINGWALD']            = '4_SEA'
continents['1009-PF-KH-PLOWE']                = '4_SEA'
continents['1010-PF-TH-ANDERSON']             = '4_SEA'
continents['1011-PF-KH-SU']                   = '4_SEA'
continents['1012-PF-KH-WHITE']                = '4_SEA'
continents['1013-PF-PEGB-BRANCH']             = '6_SA'
continents['1014-PF-SSA-SUTHERLAND']          = '3_AF'
continents['1015-PF-KE-NZILA']                = '2_EA'
continents['1016-PF-TH-NOSTEN']               = '4_SEA'
continents['1017-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1018-PF-GB-NEWBOLD']              = '9_Lab'
continents['1020-PF-VN-BONI']                 = '4_SEA'
continents['1021-PF-PG-MUELLER']              = '5_OC'
continents['1022-PF-MW-OCHOLLA']              = '2_EA'
continents['1023-PF-CO-ECHEVERRI-GARCIA']     = '6_SA'
continents['1024-PF-UG-BOUSEMA']              = '2_EA'
continents['1025-PF-KH-PLOWE']                = '4_SEA'
continents['1026-PF-GN-CONWAY']               = '1_WA'
continents['1027-PF-KE-BULL']                 = '2_EA'
continents['1031-PF-SEA-PLOWE']               = '4_SEA'
continents['1044-PF-KH-FAIRHURST']            = '4_SEA'
continents['1052-PF-TRAC-WHITE']              = '4_SEA'
continents['1062-PF-PG-BARRY']                = '5_OC'
continents['1083-PF-GH-CONWAY']               = '1_WA'
continents['1093-PF-CM-APINJOH']              = '1_WA'
continents['1094-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1095-PF-TZ-ISHENGOMA']            = '2_EA'
continents['1096-PF-GH-GHANSAH']              = '1_WA'
continents['1097-PF-ML-MAIGA']                = '1_WA'
continents['1098-PF-ET-GOLASSA']              = '2_EA'
continents['1100-PF-CI-YAVO']                 = '1_WA'
continents['1101-PF-CD-ONYAMBOKO']            = '1_WA'
continents['1102-PF-MG-RANDRIANARIVELOJOSIA'] = '2_EA'
continents['1103-PF-PDN-GMSN-NGWA']           = '1_WA'
continents['1107-PF-KEN-KAMAU']               = '2_EA'
continents['1125-PF-TH-NOSTEN']               = '4_SEA'
continents['1127-PF-ML-SOULEYMANE']           = '1_WA'
continents['1131-PF-BJ-BERTIN']               = '1_WA'
continents['1133-PF-LAB-MERRICK']             = '9_Lab'
continents['1134-PF-ML-CONWAY']               = '1_WA'
continents['1135-PF-SN-CONWAY']               = '1_WA'
continents['1136-PF-GM-NGWA']                 = '1_WA'
continents['1137-PF-GM-DALESSANDRO']          = '1_WA'
continents['1138-PF-CD-FANELLO']              = '1_WA'
continents['1141-PF-GM-CLAESSENS']            = '1_WA'
continents['1145-PF-PE-GAMBOA']               = '6_SA'
continents['1147-PF-MR-CONWAY']               = '1_WA'
continents['1151-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1152-PF-DBS-GH-AMENGA-ETEGO']     = '1_WA'
continents['1155-PF-ID-PRICE']                = '5_OC'


samples['continent'] = pandas.Series([continents[x] for x in samples.study], index=samples.index)


samples.continent.value_counts()


samples


# Let's work with two populations only for simplicity. These are *Plasmodium falciparum* populations from Oceania (5_OC) and South America (6_SA).

sample_selection = samples.continent.isin({'5_OC', '6_SA'}).values
sample_selection[:5]


sample_selection = samples.study.isin(
    {'1010-PF-TH-ANDERSON', '1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE'}
).values
sample_selection[:5]


# Now restrict the samples table to only these two populations.

samples_subset = samples[sample_selection]
samples_subset.reset_index(drop=True, inplace=True)
samples_subset.head()


samples_subset.continent.value_counts()


# Now let's subset the genotype calls to keep only variants that pass our quality filters and only samples in our two populations of interest.

get_ipython().run_cell_magic('time', '', 'genotypes_subset = genotypes.subset(variant_selection, sample_selection)')


# This takes a few minutes, so time for a quick tea break.

genotypes_subset


# The new genotype array we've made has 1,816,619 variants and 98 samples, as expected.

# ## Sample QC
# 
# Before we go any further, let's do some sample QC. This is just to check if any of the 98 samples we're working with have major quality issues that might confound an analysis. 
# 
# Compute the percent of missing and heterozygous genotype calls for each sample.

get_ipython().run_cell_magic('time', '', 'n_variants = len(variants_pass)\npc_missing = genotypes_subset.count_missing(axis=0)[:] * 100 / n_variants\npc_het = genotypes_subset.count_het(axis=0)[:] * 100 / n_variants')


# Define a function to plot genotype frequencies for each sample.

def plot_genotype_frequency(pc, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    left = np.arange(len(pc))
    palette = sns.color_palette()
    pop2color = {'1_WA': palette[0], '6_SA': palette[1], '4_SEA': palette[2]}
    colors = [pop2color[p] for p in samples_subset.continent]
    ax.bar(left, pc, color=colors)
    ax.set_xlim(0, len(pc))
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title(title)
    handles = [mpl.patches.Patch(color=palette[0]),
               mpl.patches.Patch(color=palette[1])]
    ax.legend(handles=handles, labels=['1_WA', '6_SA', '4_SEA'], title='Population',
              bbox_to_anchor=(1, 1), loc='upper left')


# Let's look at missingness first.

plot_genotype_frequency(pc_missing, 'Missing')


# All samples have pretty low missingness, though generally slightly higher in South America than in West Africa, as might be expected given the 3D7 reference is thought to originate from West Africa. Just for comparison with Alistair's original notebook on which this is based, let's look at the sample with the highest missingness.

np.argsort(pc_missing)[-1]


# Let's dig a little more into this sample. Is the higher missingness spread over the whole genome, or only in a specific region? Choose two other samples to compare with.

g_strange = genotypes_subset.take([30, 62, 63], axis=1)
g_strange


# Locate missing calls.

is_missing = g_strange.is_missing()[:]
is_missing


# Plot missingness for each sample over the chromosome.

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_pass['CHROM'][:]==current_chrom
    pos = variants_pass['POS'][:][this_chrom_variant]
    window_size = 10000
    y1, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 0], statistic=np.count_nonzero, size=window_size)
    y2, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 1], statistic=np.count_nonzero, size=window_size)
    y3, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 2], statistic=np.count_nonzero, size=window_size)
    x = windows.mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1 * 100 / window_size, lw=1)
    ax.plot(x, y2 * 100 / window_size, lw=1)
    ax.plot(x, y3 * 100 / window_size, lw=1)
    ax.set_title(current_chrom.decode('ascii'))
    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Percent calls');


# The sample with higher missingness (in red) has generally higher missingness in the same places as other samples (i.e. in var gene regions)
# 
# Let's look at heterozygosity.

plot_genotype_frequency(pc_het, 'Heterozygous')


# No samples stand out, although it looks like there is a general trend for lower heterozogysity in the South American population.
# 

# ## Allele count
# 
# As a first step into doing some population genetic analyses, let's perform an allele count within each of the two populations we've selected. This just means, for each SNP, counting how many copies of the reference allele (0) and each of the alternate alleles (1, 2, 3) are observed.
# 
# To set this up, define a dictionary mapping population names onto the indices of samples within them.

subpops = {
    'all': list(range(len(samples_subset))),
    'WA': samples_subset[samples_subset.continent == '1_WA'].index.tolist(),
    'SA': samples_subset[samples_subset.continent == '6_SA'].index.tolist(),
    'SEA': samples_subset[samples_subset.continent == '4_SEA'].index.tolist(),
}
subpops['WA'][:5]


# Now perform the allele count.

get_ipython().run_cell_magic('time', '', 'ac_subpops = genotypes_subset.count_alleles_subpops(subpops, max_allele=6)')


ac_subpops


# Each column in the table above has allele counts for a population, where "all" means the union of both populations. We can pull out a single column, e.g.:

ac_subpops['SA'][:5]


# So in the SA population, at the first variant (index 0) we observe 82 copies of the reference allele (0) and 2 copies of the first alternate allele (1).

# ## Locate segregating variants
# 
# There are lots of SNPs which do not segregate in either of these populations are so are not interesting for any analysis of these populations. We might as well get rid of them.
# 
# How many segregating SNPs are there in each population?

for pop in 'all', 'WA', 'SA', 'SEA':
    print(pop, ac_subpops[pop].count_segregating())


# Locate SNPs that are segregating in the union of our two selected populations.

is_seg = ac_subpops['all'].is_segregating()[:]
is_seg


# Subset genotypes again to keep only the segregating SNPs.

genotypes_seg = genotypes_subset.compress(is_seg, axis=0)
genotypes_seg


# Subset the variants and allele counts too.

variants_seg = variants_pass.compress(is_seg)
variants_seg


ac_seg = ac_subpops.compress(is_seg)
ac_seg


for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, int(eval("%s.nbytes" % object_name) / 1e+6))


# ## Population differentiation
# 
# Are these two populations genetically different? To get a first impression, let's plot the alternate allele counts from each population.

jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SA');


jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SEA');
jsfs


jsfs = allel.stats.joint_sfs(ac_seg['SA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, SA')
ax.set_ylabel('Alternate allele count, SEA');


jsfs


# So the alternate allele counts are correlated, meaning there is some relationship between these two populations, however there are plenty of SNPs off the diagonal, suggesting there is also some differentiation.
# 
# Let's compute average Fst, a statistic which summarises the difference in allele frequencies averaged over all SNPs. This also includes an estimate of standard error via jacknifing in blocks of 100,000 SNPs.

fst, fst_se, _, _ = allel.stats.blockwise_hudson_fst(ac_seg['WA'], ac_seg['SA'], blen=100000)
print("Hudson's Fst: %.3f +/- %.3f" % (fst, fst_se))


# Define a function to plot Fst in windows over the chromosome.

def plot_fst(ac1, ac2, pos, blen=2000, current_chrom=b'Pf3D7_01_v3'):
    
    fst, se, vb, _ = allel.stats.blockwise_hudson_fst(ac1, ac2, blen=blen)
    
    # use the per-block average Fst as the Y coordinate
    y = vb
    
    # use the block centres as the X coordinate
    x = allel.stats.moving_statistic(pos, statistic=lambda v: (v[0] + v[-1]) / 2, size=blen)
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y, 'k-', lw=.5)
    ax.set_ylabel('$F_{ST}$')
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())


# Are any chromosome regions particularly differentiated?

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    plot_fst(
        ac_seg['WA'].subset(this_chrom_variant),
        ac_seg['SA'].subset(this_chrom_variant),
        variants_seg['POS'][:][this_chrom_variant],
        100,
        current_chrom
    )


# Maybe some interesting signals of differentiation here.
# 
# There are a number of subtleties to Fst analysis which I haven't mentioned here, but you can read more about [estimating Fst](http://alimanfoo.github.io/2015/09/21/estimating-fst.html) on my blog.
# 
# ## Site frequency spectra
# 
# While we're looking at allele counts, let's also plot a site frequency spectrum for each population, which gives another summary of the data and is also informative about demographic history.
# 
# To do this we really do need to restrict to biallelic variants, so let's do that first.

is_biallelic_01 = ac_seg['all'].is_biallelic_01()[:]
ac1 = ac_seg['WA'].compress(is_biallelic_01, axis=0)[:, :2]
ac2 = ac_seg['SA'].compress(is_biallelic_01, axis=0)[:, :2]
ac3 = ac_seg['SEA'].compress(is_biallelic_01, axis=0)[:, :2]
ac1


# OK, now plot folded site frequency spectra, scaled such that populations with constant size should have a spectrum close to horizontal (constant across allele frequencies).

fig, ax = plt.subplots(figsize=(8, 5))
sns.despine(ax=ax, offset=10)
sfs1 = allel.stats.sfs_folded_scaled(ac1)
allel.stats.plot_sfs_folded_scaled(sfs1, ax=ax, label='WA', n=ac1.sum(axis=1).max())
sfs2 = allel.stats.sfs_folded_scaled(ac2)
allel.stats.plot_sfs_folded_scaled(sfs2, ax=ax, label='SA', n=ac2.sum(axis=1).max())
sfs3 = allel.stats.sfs_folded_scaled(ac3)
allel.stats.plot_sfs_folded_scaled(sfs3, ax=ax, label='SEA', n=ac3.sum(axis=1).max())
ax.legend()
ax.set_title('Scaled folded site frequency spectra')
# workaround bug in scikit-allel re axis naming
ax.set_xlabel('minor allele frequency');


# The spectra are very different for the three populations. WA has an excess of rare variants, suggesting a population expansion, while SA and SEA are closer to neutral expectation, suggesting a more stable population size.
# 
# We can also plot Tajima's D, which is a summary of the site frequency spectrum, over the chromosome, to see if there are any interesting localised variations in this trend.

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    # compute windows with equal numbers of SNPs
    pos = variants_seg['POS'][:][this_chrom_variant]
    windows = allel.stats.moving_statistic(pos, statistic=lambda v: [v[0], v[-1]], size=100)
    x = np.asarray(windows).mean(axis=1)

    # compute Tajima's D
    y1, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['WA'].subset(this_chrom_variant), windows=windows)
    y2, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SA'].subset(this_chrom_variant), windows=windows)
    y3, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SEA'].subset(this_chrom_variant), windows=windows)

    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1, lw=.5, label='WA')
    ax.plot(x, y2, lw=.5, label='SA')
    ax.plot(x, y3, lw=.5, label='SEA')
    ax.set_ylabel("Tajima's $D$")
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1));


# Not really sure what to make of the above!

# ## Principal components analysis
# 
# Finally, let's to a quick-and-dirty PCA to confirm our evidence for differentiation between these two populations and check if there is any other genetic structure within populations that we might have missed.
# 
# First grab the allele counts for the union of the two populations.

ac = ac_seg['all'][:]
ac


# Select the variants to use for the PCA, including only biallelic SNPs with a minor allele count above 2.

pca_selection = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 2)
pca_selection


np.count_nonzero(pca_selection)


# Now randomly downsample these SNPs purely for speed.

indices = np.nonzero(pca_selection)[0]
indices


len(indices)


indices_ds = np.random.choice(indices, size=50000, replace=False)
indices_ds.sort()
indices_ds


# Subset the genotypes to keep only our selected SNPs for PCA.

genotypes_pca = genotypes_seg.take(indices_ds, axis=0)
genotypes_pca


# Transform the genotypes into an array of alternate allele counts per call. 

gn = genotypes_pca.to_n_alt()[:]
gn


# Run the PCA.

coords, model = allel.stats.pca(gn)


coords


coords.shape


# Plot the results.

def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for pop in ['1_WA', '6_SA', '4_SEA']:
        flt = (samples_subset.continent == pop).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=pop, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))


fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();


def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for study in ['1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE', '1010-PF-TH-ANDERSON']:
        flt = (samples_subset.study == study).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=study, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))


fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();


samples.index[sample_selection][(coords[:, 0] > -50) & (coords[:, 0] < -20)]


samples.index[sample_selection][coords[:, 0] > 0]


samples.index[sample_selection][coords[:, 1] > 0]


coords[(samples.index == 'PP0012-C')[sample_selection]]


coords[(samples.index == 'PP0022-C')[sample_selection]]


coords[(samples.index == 'PP0022-Cx')[sample_selection]]


coords[(samples.index == 'PD0047-C')[sample_selection]]


coords[(samples.index == 'PP0018-C')[sample_selection]]


fig, ax = plt.subplots(figsize=(5, 4))
sns.despine(ax=ax, offset=10)
y = 100 * model.explained_variance_ratio_
x = np.arange(len(y))
ax.set_xticks(x + .4)
ax.set_xticklabels(x + 1)
ax.bar(x, y)
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance explained (%)');


# From this PCA we can see that PC1 and PC2 separate the three populations. Some of the South American samples are clustering together with South-East Asia - so maybe these are the lab sample (Dd2) contaminants mentioned by Julian?
# 
# For running PCA with more populations there are a number of subtleties which I haven't covered here, for all the gory details see the [fast PCA](http://alimanfoo.github.io/2015/09/28/fast-pca.html) article on my blog.

# ## Under the hood
# 
# Here's a few notes on what's going on under the hood. If you want to know more, the best place to look is the [scikit-allel source code](https://github.com/cggh/scikit-allel).

# ### NumPy arrays
# 
# NumPy is the foundation for everything in scikit-allel. A NumPy array is an N-dimensional container for binary data.

x = np.array([0, 4, 7])
x


x.ndim


x.shape


x.dtype


# item access
x[1]


# slicing
x[0:2]


# NumPy support array-oriented programming, which is both convenient and efficient, because looping is implemented internally in C code. 

y = np.array([1, 6, 9])
x + y


# Scikit-allel defines a number of conventions for storing variant call data using NumPy arrays. For example, a set of diploid genotype calls over *m* variants in *n* samples is stored as a NumPy array of integers with shape (m, n, 2). 

g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[0, 2], [-1, -1]]], dtype='i1')
g


# The ``allel.GenotypeArray`` class is a sub-class of ``np.ndarray``.

isinstance(g, np.ndarray)


# All the usual properties and methods of an ndarray are inherited.

g.ndim


g.shape


# obtain calls for the second variant in all samples
g[1, :]


# obtain calls for the second sample in all variants
g[:, 1]


# obtain the genotype call for the second variant, second sample
g[1, 1]


# make a subset with only the first and third variants
g.take([0, 2], axis=0)


# find missing calls
np.any(g < 0, axis=2)


# Instances of ``allel.GenotypeArray`` also have some extra properties and methods. 

g.n_variants, g.n_samples, g.ploidy


g.count_alleles()


# ### Chunked, compressed arrays
# 
# The ``scikit-allel`` genotype array convention is flexible, allowing for multiallelic and polyploid genotype calls. However, it is not very compact, requiring 2 bytes of memory for each call. A set of calls for 10,000,000 SNPs in 1,000 samples thus requires 20G of memory.
# 
# One option to work with large arrays is to use bit-packing, i.e., to pack two or more items of data into a single byte. E.g., this is what the [plink BED format](http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml) does. If you have have diploid calls that are only ever biallelic, then it is possible to fit 4 genotype calls into a single byte. This is 8 times smaller than the NumPy unpacked representation.
# 
# However, coding against bit-packed data is not very convenient. Also, there are several libraries available for Python which allow N-dimensional arrays to be stored using **compression**: [h5py](http://www.h5py.org/), [bcolz](http://bcolz.blosc.org/en/latest/) and [zarr](http://zarr.readthedocs.io). Genotype data is usually extremely compressible due to sparsity - most calls are homozygous ref, i.e., (0, 0), so there are a lot of zeros. 
# 
# For example, the ``genotypes`` data we used above has calls for 16 million variants in 765 samples, yet requires only 1.2G of storage. In other words, there are more than 9 genotype calls per byte, which means that each genotype call requires less than a single bit on average.

genotypes


# The data for this array are stored in an HDF5 file on disk and compressed using zlib, and achieve a compression ratio of 19.1 over an equivalent uncompressed NumPy array.
# 
# To avoid having to decompress the entire dataset every time you want to access any part of it, the data are divided into chunks and each chunk is compressed. You have to choose the chunk shape, and there are some trade-offs regarding both the shape and size of a chunk. 
# 
# Here is the chunk shape for the ``genotypes`` dataset.

genotypes.chunks


# This means that the dataset is broken into chunks where each chunk has data for 6553 variants and 10 samples.
# 
# This gives a chunk size of ~128K (6553 \* 10 \* 2) which we have since found is not optimal - better performance is usually achieved with chunks that are at least 1M. However, performance is not bad and the data are publicly released so I haven't bothered to rechunk them.
# 
# Chunked, compressed arrays can be stored either on disk (as for the ``genotypes`` dataset) or in main memory. E.g., in the tour above, I stored all the intermediate genotype arrays in memory, such as the ``genotypes_subset`` array, which can speed things up a bit.

genotypes_subset


# To perform some operation over a chunked arrays, the best way is to compute the result for each chunk separately then combine the results for each chunk if needed. All functions in ``scikit-allel`` try to use a chunked implementation wherever possible, to avoid having to load large data uncompressed into memory.

# ## Further reading
# 
# 
# * [scikit-allel reference documentation](http://scikit-allel.readthedocs.io/)
# * [Introducing scikit-allel](http://alimanfoo.github.io/2015/09/15/introducing-scikit-allel.html)
# * [Estimating Fst](http://alimanfoo.github.io/2015/09/21/estimating-fst.html)
# * [Fast PCA](http://alimanfoo.github.io/2015/09/28/fast-pca.html)
# * [To HDF5 and beyond](http://alimanfoo.github.io/2016/04/14/to-hdf5-and-beyond.html)
# * [CPU blues](http://alimanfoo.github.io/2016/05/16/cpu-blues.html)
# * [vcfnp](https://github.com/alimanfoo/vcfnp)
# * [numpy](http://www.numpy.org/)
# * [matplotlib](http://matplotlib.org/)
# * [pandas](http://pandas.pydata.org/)
# 

import datetime
print(datetime.datetime.now().isoformat())


# # Introduction
# Sample status report created weekly by Jim and put in https://alfresco.malariagen.net/share/page/site/malariagen-analysis/documentlibrary#filter=path%7C%2FSampleStatusReport%7C&page=3
# 
# This notebook parses this to get all Pf and Pv samples

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_parse_sample_status_report"
get_ipython().system('mkdir -p {output_dir}')

sample_status_report_fn = "%s/2016_12_07_report_sample_status.xls" % output_dir
output_fn = "%s/2016_12_07_report_sample_status_pf_pv.xlsx" % output_dir
output_fn = "%s/2016_12_07_report_sample_status.txt" % output_dir


tbl_studies = (
    etl
    .fromxls(sample_status_report_fn, 'Report')
    .skip(2)
    .selectin('Taxon', ['PF', 'PV'])
    .selectne('Project Code', '')
)
project_codes = tbl_studies.values('Project Code').array()
alfresco_codes = tbl_studies.values('Alfresco Code').array()
print(len(tbl_studies.data()))
tbl_studies.tail()


project_codes


tbl_sample_status_report = (
    etl
    .fromxls(sample_status_report_fn, project_codes[0])
    .skip(10)
    .selectne('Oxford Code', '')
    .addfield('study', alfresco_codes[0])
)
for i, project_code in enumerate(project_codes):
    if i > 0:
        tbl_sample_status_report = (
            tbl_sample_status_report
            .cat(
                etl
                .fromxls(sample_status_report_fn, project_code)
                .skip(10)
                .selectne('Oxford Code', '')
                .addfield('study', alfresco_codes[i])
            )
        )


tbl_sample_status_report


len(tbl_sample_status_report.data())


tbl_sample_status_report.totsv(output_fn, lineterminator='\n')


# # Sanity check the data

tbl_sample_status_reloaded = etl.fromtsv(output_fn)
print(len(tbl_sample_status_reloaded.data()))
tbl_sample_status_reloaded


output_fn


tbl_sample_status_reloaded.tail()


tbl_sample_status_reloaded.duplicates('Oxford Code')


tbl_sample_status_reloaded.valuecounts('study').displayall()


tbl_sample_status_reloaded.valuecounts('Country of Origin').displayall()


tbl_sample_status_reloaded.selecteq('Country of Origin', '').displayall()





# # Plan
# - Create annotated versions (CDS, VARIANT_TYPE, MULTIALLELIC) of chromosome vcfs ala Pf3k, including only fields from 31/10/2016 11:56 email
# - Create WG sites files
# - Create summary tables ala Pf3k

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161125_Pf60_final_vcfs'
vrpipe_vcfs_dir = '/nfs/team112_internal/production_files/Pf/6_0'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages'
nfs_final_vcf_dir = '%s/vcf' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_vcf_dir}')

gff_fn = "/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3"
cds_gff_fn = "%s/gff/Pfalciparum_GeneDB_Oct2016.Pfalciparum.noseq.gff3.cds.gz" % output_dir
annotations_header_fn = "%s/intermediate_files/annotations.hdr" % (output_dir)

run_create_multiallelics_file_job_fn = "%s/scripts/run_create_multiallelics_file_job.sh" % output_dir
submit_create_multiallelics_file_jobs_fn = "%s/scripts/submit_create_multiallelics_file_jobs.sh" % output_dir
create_study_vcf_job_fn = "%s/scripts/create_study_vcf_job.sh" % output_dir

vrpipe_metadata_fn = "%s/Pf_6.0_vrpipe_bam_summaries.txt" % output_dir

GENOME_FN = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

genome_fn = "%s/Pfalciparum.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/gff')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/study_vcfs')
get_ipython().system('mkdir -p {output_dir}/intermediate_files')
get_ipython().system('mkdir -p {output_dir}/tables')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')


cds_gff_fn


get_ipython().system("grep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")


fo=open(annotations_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (INDEL)">', file=fo)
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()


fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
JOB=$LSB_JOBINDEX
# JOB=16
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_SITES_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_FULL_VCF_FN=%s/vcf/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.txt.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pf_60_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN
 
python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\t%%CDS\t%%VARIANT_TYPE\t%%MULTIALLELIC\n' | \
bgzip -c > $ANNOTATION_FN

tabix -s1 -b2 -e2 $ANNOTATION_FN

#/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
#$ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools view \
--include 'ALT!="*"' | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CHROM,POS,REF,ALT,CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_FULL_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AF,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC,^FORMAT/GT,FORMAT/AD,FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL,\
^FILTER/PASS,FILTER/Centromere,FILTER/InternalHypervariable,FILTER/SubtelomericHypervariable,\
FILTER/SubtelomericRepeat,FILTER/Low_VQSLOD \
-Oz -o $OUTPUT_VCF_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$OUTPUT_VCF_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        vrpipe_vcfs_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        cds_gff_fn,
        annotations_header_fn,
        GENOME_FN,
        annotations_header_fn,
        cds_gff_fn,
        annotations_header_fn,
        )
        , file=fo)
fo.close()


fo = open(submit_create_multiallelics_file_jobs_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
LOG_DIR=%s/log
 
NUM_CHROMS=`wc -l < $FASTA_FAI_FILE`
QUEUE=long

bsub -q $QUEUE -G malaria-dk -J "ma[1-$NUM_CHROMS]" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s
''' % (
        GENOME_FN,
        output_dir,
        "bash %s" % run_create_multiallelics_file_job_fn,
        ),
     file=fo)
fo.close()


get_ipython().system('bash {submit_create_multiallelics_file_jobs_fn}')


2**24


# # Create study-specific VCFs

get_ipython().system('cp {GENOME_FN} {genome_fn}')
genome = pyfasta.Fasta(genome_fn)
genome


genome_length = 0
for chrom in genome.keys():
    genome_length += len(genome[chrom])
genome_length


758718931/23332839


7377894+208885


7377894+208885+208885


vrpipe_columns = [
    'path', 'sample', 'study', 'bases_of_1X_coverage', 'bases_of_2X_coverage', 'bases_of_5X_coverage',
    'mean_coverage', 'mean_insert_size', 'sd_insert_size', 'avg_read_length', 'bases_callable_percent',
    'bases_no_coverage_percent', 'bases_low_coverage_percent', 'bases_excessive_coverage_percent',
    'bases_poor_mapping_quality_percent', 'bases_ref_n_percent', 'reads', 'reads_mapped', 'reads_mapped_and_paired',
    'reads_properly_paired', 'reads_qc_failed', 'pairs_on_different_chromosomes', 'non_primary_alignments',
    'center_name'
]
print(",".join(vrpipe_columns[1:]))


metadata_columns = [
    'sample', 'study', 'center_name', 'bases_callable_proportion', 'bases_no_coverage_proportion', 'bases_low_coverage_proportion',
#     'bases_excessive_coverage_proportion', 'bases_poor_mapping_quality_proportion', 'bases_ref_n_proportion',
    'bases_poor_mapping_quality_proportion',
    'proportion_genome_covered_at_1x', 'proportion_genome_covered_at_5x', 'mean_coverage',
    'mean_insert_size', 'sd_insert_size', 'avg_read_length', 
    'reads_mapped_proportion', 'mapped_reads_properly_paired_proportion', 'pairs_on_different_chromosomes_proportion',
    'non_primary_alignments_proportion',
]


get_ipython().system('vrpipe-fileinfo --setup pf_60_mergelanes --metadata {",".join(vrpipe_columns[1:])} | sort -k 2,2 > {vrpipe_metadata_fn}')

# | grep '\.summary' \


get_ipython().system('vrpipe-fileinfo --setup pf_60_mergelanes --metadata sample,study,bases_of_1X_coverage,bases_of_2X_coverage,bases_of_5X_coverage,mean_coverage,mean_insert_size,sd_insert_size,avg_read_length,bases_callable_percent,bases_no_coverage_percent,bases_low_coverage_percent,bases_excessive_coverage_percent,bases_poor_mapping_quality_percent,bases_ref_n_percent,reads,reads_mapped,reads_mapped_and_paired,reads_properly_paired,reads_qc_failed,pairs_on_different_chromosomes,non_primary_alignments,center_name > {vrpipe_metadata_fn}')

# | grep '\.summary' \
# | sort -k 2,2 \


'%s/vcf/Pf_60_Pf3D7_01_v3.final.vcf.gz' % output_dir


vcf.VERSION


pysam.__version__


sys.version


vcf_samples = vcf.Reader(filename='%s/vcf/Pf_60_Pf3D7_01_v3.final.vcf.gz' % output_dir).samples
print(len(vcf_samples))
vcf_samples[0:10]


tbl_vcf_samples = etl.fromcolumns([vcf_samples]).setheader(['sample'])
print(len(tbl_vcf_samples.data()))


tbl_vcf_samples.duplicates('sample')


tbl_vcf_samples.antijoin(tbl_sample_metadata, key='sample')


def genome_coverage(rec, variable='bases_of_1X_coverage'):
    if rec[variable] == 'unknown':
        return(0.0)
    else:
        return(round(rec[variable] / genome_length, 4))


metadata_columns


tbl_sample_metadata = (
    etl
    .fromtsv(vrpipe_metadata_fn)
    .pushheader(vrpipe_columns)
    .select(lambda rec: 'pe' in rec['path'] or rec['sample'] == 'PN0002-C')
    .convertnumbers()
    .convert('avg_read_length', lambda val: val+1)
#     .addfield('bases_callable_proportion', lambda rec: 0.0 if rec['bases_callable_percent'] == 'unknown' else round(rec['bases_callable_percent'] / 100, 4))
#     .addfield('bases_no_coverage_proportion', lambda rec: 0.0 if rec['bases_no_coverage_percent'] == 'unknown' else round(rec['bases_no_coverage_percent'] / 100, 4))
#     .addfield('bases_low_coverage_proportion', lambda rec: 0.0 if rec['bases_low_coverage_percent'] == 'unknown' else round(rec['bases_low_coverage_percent'] / 100, 4))
#     .addfield('bases_excessive_coverage_proportion', lambda rec: 0.0 if rec['bases_excessive_coverage_percent'] == 'unknown' else round(rec['bases_excessive_coverage_percent'] / 100, 4))
#     .addfield('bases_poor_mapping_quality_proportion', lambda rec: 0.0 if rec['bases_poor_mapping_quality_percent'] == 'unknown' else round(rec['bases_poor_mapping_quality_percent'] / 100, 4))
#     .addfield('bases_ref_n_proportion', lambda rec: 0.0 if rec['bases_ref_n_percent'] == 'unknown' else round(rec['bases_ref_n_percent'] / 100, 4))
#     .addfield('proportion_genome_covered_at_1x', lambda rec: 0.0 if rec['bases_of_1X_coverage'] == 'unknown' else round(rec['bases_of_1X_coverage'] / genome_length, 4))
#     .addfield('proportion_genome_covered_at_5x', lambda rec: 0.0 if rec['bases_of_5X_coverage'] == 'unknown' else round(rec['bases_of_5X_coverage'] / genome_length, 4))
#     .addfield('reads_mapped_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_mapped'] / rec['reads'], 4))
#     .addfield('mapped_reads_properly_paired_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_properly_paired'] / rec['reads_mapped'], 4))
#     # Note in the following we use reads_properly_paired/2 to get numbers of pairs of reads
#     .addfield('pairs_on_different_chromosomes_proportion', lambda rec: 0.0 if rec['pairs_on_different_chromosomes'] == 'unknown' or rec['pairs_on_different_chromosomes'] == 0.0 else round(rec['pairs_on_different_chromosomes'] / (rec['pairs_on_different_chromosomes'] + ( rec['reads_properly_paired'] / 2)), 4))
#     .addfield('non_primary_alignments_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['non_primary_alignments'] / rec['reads_mapped'], 4))
#     .addfield('reads_qc_failed_proportion', lambda rec: 0.0 if rec['reads_qc_failed'] == 'unknown' else round(rec['reads_qc_failed'] / rec['reads'], 4))
    #  .leftjoin(tbl_solaris_metadata, lkey='sample', rkey='ox_code')
    #  .convert('run_accessions', 'NULL', where=lambda rec: rec['study'] == '1156-PV-ID-PRICE') # These were wrongly accessioned and are currently being removed from ENA
    #  .cut(['sample', 'study', 'src_code', 'run_accessions', 'genome_covered_at_1x', 'genome_covered_at_5x',
    #        'mean_coverage', 'avg_read_length'])
#     .cut(metadata_columns)
    .selectin('sample', vcf_samples)
    .sort('sample')
)
print(len(tbl_sample_metadata.data()))
tbl_sample_metadata.display(index_header=True)


tbl_sample_metadata.selecteq('sample', 'PN0002-C')


tbl_sample_metadata.select(lambda rec: type(rec['bases_callable_percent']) == str)


tbl_sample_metadata.selectgt('reads_qc_failed_proportion', 0.0)


tbl_sample_metadata.selectgt('bases_ref_n_proportion', 0.0)


tbl_sample_metadata.selectgt('bases_excessive_coverage_proportion', 0.0)


tbl_sample_metadata.selectgt('non_primary_alignments_proportion', 0.2)


0.9565+0.0389+0.0047


tbl_sample_metadata.duplicates('sample').displayall()





tbl_sample_metadata.select(lambda rec: 'se' in rec['path']).selectin('sample', vcf_samples).displayall()


tbl_sample_metadata.select(lambda rec: rec['sample'] in ['PM0006-C', 'PM0007-C', 'PM0008-C', 'PN0002-C']).displayall()


print(len(tbl_sample_metadata.selectin('sample', vcf_samples).data()))


tbl_sample_metadata.selecteq('bases_of_1X_coverage', 22897930)


tbl_sample_metadata.select(lambda rec: type(rec[3]) == str and rec[3] != 'unknown')


tbl_sample_metadata.selectin('sample', vcf_samples).select(lambda rec: type(rec[3]) == str and rec[3] == 'unknown')


tbl_sample_metadata.selecteq('bases_of_1X_coverage', 'unknown')


type('a') == str


tbl_sample_metadata.valuecounts('avg_read_length').sort('avg_read_length').displayall()


tbl_sample_metadata.valuecounts('study').sort('study').displayall()


studies = tbl_sample_metadata.distinct('study').values('study').array()
studies


study_vcf_jobs_manifest = '%s/study_vcf_jobs_manifest.txt' % output_dir
fo = open(study_vcf_jobs_manifest, 'w')
for study in studies:
    sample_ids = ",".join(tbl_sample_metadata.selecteq('study', study).values('sample'))
    for chrom in sorted(genome.keys()):
        print('%s\t%s\t%s' % (study, chrom, sample_ids), file=fo)
fo.close()


get_ipython().system('cat {study_vcf_jobs_manifest}')


get_ipython().system('which bcftools')


get_ipython().system('bcftools')


fo = open(create_study_vcf_job_fn, 'w')
print('''STUDY_VCF_JOBS_FILE=%s
 
JOB=$LSB_JOBINDEX
# JOB=16
 
IN=`sed "$JOB q;d" $STUDY_VCF_JOBS_FILE`
read -a LINE <<< "$IN"
STUDY=${LINE[0]}
CHROM=${LINE[1]}
SAMPLES=${LINE[2]}

OUTPUT_DIR=%s

mkdir -p $OUTPUT_DIR/study_vcfs/$STUDY

INPUT_VCF_FN=$OUTPUT_DIR/vcf/Pf_60_$CHROM.final.vcf.gz
OUTPUT_VCF_FN=$OUTPUT_DIR/study_vcfs/$STUDY/Pf_60__$STUDY\__$CHROM.vcf.gz

echo $OUTPUT_VCF_FN
echo $STUDY

bcftools view --samples $SAMPLES --output-file $OUTPUT_VCF_FN --output-type z $INPUT_VCF_FN
bcftools index --tbi $OUTPUT_VCF_FN
md5sum $OUTPUT_VCF_FN > $OUTPUT_VCF_FN.md5

''' % (
        study_vcf_jobs_manifest,
        output_dir,
        )
        , file=fo)
fo.close()


get_ipython().system('bash {create_study_vcf_job_fn}')


QUEUE = 'normal'
wc_output = get_ipython().getoutput('wc -l {study_vcf_jobs_manifest}')
NUM_JOBS = wc_output[0].split(' ')[0]
MEMORY = 8000
LOG_DIR = "%s/log" % output_dir

print(NUM_JOBS, LOG_DIR)

get_ipython().system('bsub -q {QUEUE} -G malaria-dk -J "s_vcf[1-{NUM_JOBS}]" -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {LOG_DIR}/output_%J-%I.log bash {create_study_vcf_job_fn}')


# # Copy files to /nfs

get_ipython().system('cp {output_dir}/vcf/* {nfs_final_vcf_dir}/')


get_ipython().system('cp -R {output_dir}/study_vcfs/* {nfs_release_dir}/')


2+2


for study in studies:
    get_ipython().system('cp /lustre/scratch116/malaria/pfalciparum/resources/regions-20130225.bed.gz* {nfs_release_dir}/{study}/')





# # Introduction
# The purpose of this notebook is to compare the results of running Olivo's GRC tools and my code to call the same loci from Pf 6.0 VCF.
# 
# See 20161117_run_Olivo_GRC.ipynb for details of running Olivo's GRC code on the Pf6.0 release and 20161118_GRC_from_VCF.ipynb for VCF code.

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_Olivo_vs_VCF_GRC_Pf6"
get_ipython().system('mkdir -p {output_dir}')
reads_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"
vcf_6_0_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_GRC_from_VCF/Pf_6_GRC_from_vcf.xlsx"

all_calls_crosstab_fn = "%s/all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/discordant_nonmissing_calls_crosstab.xlsx" % output_dir


tbl_read_results = (
    etl
    .fromtsv(reads_6_0_results_fn)
    .distinct('Sample')
    .rename('mdr2_484[P]', 'mdr2_484[T]')
    .rename('fd_193[P]', 'fd_193[D]')
)
print(len(tbl_read_results.data()))
tbl_read_results


tbl_vcf_results = (
    etl
    .fromxlsx(vcf_6_0_results_fn)
)
print(len(tbl_vcf_results.data()))
tbl_vcf_results


tbl_both_results = (
    tbl_read_results
    .cutout('Num')
    .join(tbl_vcf_results.replaceall('.', '-'), key='Sample', lprefix='reads_', rprefix='vcf_')
    .convertall(lambda x: ','.join(sorted(x.split(','))))
)
print(len(tbl_both_results.data()))


df_both_results = tbl_both_results.todataframe()


loci = list(tbl_read_results.header()[2:])
print(len(loci))


writer = pd.ExcelWriter(all_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i],
        df_both_results.ix[:, i+29],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0]) # Excel doesn't like the [CMNVK] like endings
writer.save()


writer = pd.ExcelWriter(discordant_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        df_both_results.ix[:, i+29][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()


writer = pd.ExcelWriter(discordant_nonmissing_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        df_both_results.ix[:, i+29][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()





# ## Different allele discordances reads vs vcf
# 
# Manual inspection of discordant_nonmissing_calls_crosstab_fn showed the following 8 discordant alleles (ignoring all het vs hom discordances)
# 
# - crt_72-76[CVMNK]: CVIDT,CVIET vs CVIDK,CVIET
# - crt_72-76[CVMNK]: CVIET,CVMNK,SVMNT vs CVMNK,CVMNT
# - dhps_436[S]: A,F vs A,S
# - dhps_540[K]: E,N vs E,K (5)
# 
# The following few cells show which samples were discordant at the above. See 20161118_GRC_from_VCF.ipynb
# for details of VCF calls at these. Comments below refer to manual inspection of these calls

# Het call made at final SNP despite only one read
# Unclear which result really correct
df_both_results[
    (df_both_results['reads_crt_72-76[CVMNK]'] == 'CVIDT,CVIET') &
    (df_both_results['vcf_crt_72-76[CVMNK]'] == 'CVIDK,CVIET')]


# Many variants here had small numbers of ALT reads but were called het
# Reads results probably correct
df_both_results[
    (df_both_results['reads_crt_72-76[CVMNK]'] == 'CVIET,CVMNK,SVMNT') &
    (df_both_results['vcf_crt_72-76[CVMNK]'] == 'CVMNK,CVMNT')]


# Get wrong call here becuase middle GT is 0/0. Would get correct call if we used PGT call which is 1|0
# Reads results probably correct
df_both_results[(df_both_results['reads_dhps_436[S]'] == 'A,F') & (df_both_results['vcf_dhps_436[S]'] == 'A,S')]


# For all of the following, final variant looks like a het (always 2+ reads) but called hom
# Reads results probably correct for all of these
df_both_results[(df_both_results['reads_dhps_540[K]'] == 'E,N') & (df_both_results['vcf_dhps_540[K]'] == 'E,K')]





# # Conclusion
# I have written code to call the same set of variants as Olivo's read-based analysis. This uses the same input file (grc.properties) so can easily be applied to future call sets and loci. The code to create calls from the vcf is very quick, with all loci called in all samples in about 2 minutes.
# 
# The results were highly concordant. There were no homozygous call differences. There were 8 cases where different alleles were called as a part of a het call. I've manually checked all these, and in 7 cases I think the calls in the vcf are wrong, for example two different variants have what look from read counts like het calls, but one is called as het and one as hom. The eighth sample with a different allele had only one alternate read which VCF called as het but Olivo's code as hom - so not really clear what truth is here. Other than that the only differences are het vs hom calls (and the few I've checked looked like borderline calls), or called vs missing. I inspected the reads for three sample where VCF had hom calls at crt_97, whereas Olivo's code had het calls. In each sample, there were clearly good quality reads and bases with the minor allele, but these weren't reflected in the VCF, for reasons I don't understand. Olivo's read-based calls have slightly higher missingness, but this is as expected as I have done no filtering on the vcf-based calls.
# 
# The VCF-based calls can be found in /nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_GRC_from_VCF/Pf_6_GRC_from_vcf.xlsx. Missed calls are denoted by ".". Frameshift mutations are denoted by "!". Samples that have two or more het calls at the locus which are unphased are denoted by 'X'. Samples which have two or more phase groups at the locus are denoted by "?".
# 
# Calls are only set to missing if all genotypes at locus missing. If only some missing, have assumed hom ref (this is important due to pecularities at crt_72-76, where genotypes between insertion at 403618 and deletion at 403622 are set to missing in samples that have the insertion/deletion.
# 
# With the VCF-based code, two samples appeared to have frameshift mutations in crt_72-76. Manual inspection showed borderline het/hom calls. PA0490-C has hom alt call (4 ref, 134 alt reads) at 403618 insertion and het call (5 ref, 137 alt reads) at 403622 deletion. QV0090-C has het call (3 ref, 104 alt reads) at 403618 insertion and hom alt call (3 ref, 106 alt reads) at 403622 deletion. For all these calls, the genotype quality is low (GQ < 30) reflecting similar likelihoods of het and hom alt calls.
# 
# The following loci had samples with unphased hets (number in brackets are numbers of samples): crt_72-76 (6), crt_333 (4), dhps_436 (17), dhps_540 (28). A further five samples had hets in different phase groups at crt_72-76. All of these samples could not be called by VCF-based code.
# 
# Of all the variants discovered in these loci, only two (at Pf3D7_07_v3:404836 and Pf3D7_07_v3:405559) failed variant filters. In each case there was only one sample with a heterozygous call at the variant, and the position was the last in the codon and the different alleles made no change in the amino acid.
# 
# In summary, the calls from the vcf and from the reads are very similar, and the problems of calling CRT seem to be a thing of the past. Having said that, in the few cases where there was a discrepancy between calls, in general the reads-based calls are right. There were also a few cases that couldn't be called from the VCF, e.g. due to unresolved phasing, and two cases of the VCF incorrectly predicting a frameshift. Therefore I would recommend using Olivo's reads-based calls code for the GRC.
# 
# 




get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161201_Pv_30_HDF5_build'
vrpipe_fileinfo_fn = "%s/pv_30_genotype_gvcfs_200kb.txt" % output_dir
vcf_fofn = "%s/pv_30_genotype_gvcfs_20kb.fofn" % output_dir
vcf_stem = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pv3.0/20161130_Pv30_final_vcfs/vcf/Pv_30_{chrom}.final.vcf.gz'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pv/3_0_release_packages'
nfs_final_hdf5_dir = '%s/hdf5' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_hdf5_dir}')

GENOME_FN = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
genome_fn = "%s/PvivaxP01.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/hdf5')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/npy')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

get_ipython().system('cp {GENOME_FN} {genome_fn}')


genome = pyfasta.Fasta(genome_fn)
genome


transfer_length = 0
for chrom in genome.keys():
    if chrom.startswith('Transfer'):
        transfer_length += len(genome[chrom])
transfer_length


fo = open("%s/scripts/vcfnp_variants.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_calldata.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type calldata_2d \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity AD:7 \
    --arity PL:28 \
    --dtype PGT:a3 \
    --dtype PID:a12 \
    --exclude-field MIN_DP \
    --exclude-field RGQ \
    --exclude-field SB''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
inputs=$3
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''', file=fo)
fo.close()


task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

    task = "%s/scripts/vcfnp_calldata.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')


fo = open("%s/scripts/vcfnp_variants_temp.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy_temp \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


# Three variants jobs from the above didn't complete. Killed them, then ran the following
get_ipython().system('mkdir -p {output_dir}/npy_temp')

task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants_temp.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

#     task = "%s/scripts/vcfnp_calldata.sh" % output_dir
#     !bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} 


get_ipython().system('mv {output_dir}/npy_temp/v*.npy {output_dir}/npy/')


task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "hdf" -n8 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'PvP01_01_v1\')} {output_dir}/hdf5/Pv_30 {output_dir}/npy')


output_dir


task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "full" -R"select[mem>16000] rusage[mem=16000] span[hosts=1]" -M 16000     -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'Pf3D7_01_v3\')}     {output_dir}/hdf5/Pf_60     /lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161124_HDF5_build/npy')


y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT.h5' % output_dir, 'r')


(etl.wrap(
    np.unique(y['variants']['SNPEFF_EFFECT'], return_counts=True)
)
    .transpose()
    .pushheader('SNPEFF_EFFECT', 'number')
    .sort('number', reverse=True)
    .displayall()
)


task_size = 20000
for chrom in ['PvP01_00'] + sorted(genome.keys()):
    if chrom.startswith('Pv'):
        vcf_fn = vcf_stem.format(chrom=chrom)
        if chrom == 'PvP01_00':
            chrom_length = transfer_length
        else:
            chrom_length = len(genome[chrom])
        n_tasks = '1-%s' % ((chrom_length // task_size) + 1)
        print(chrom, n_tasks)

        task = "%s/scripts/vcfnp_variants.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

        task = "%s/scripts/vcfnp_calldata.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')








(etl.wrap(
    np.unique(y['variants']['CDS'], return_counts=True)
)
    .transpose()
    .pushheader('CDS', 'number')
    .sort('number', reverse=True)
    .displayall()
)


CDS = y['variants']['CDS'][:]
SNPEFF_EFFECT = y['variants']['SNPEFF_EFFECT'][:]
SNP = (y['variants']['VARIANT_TYPE'][:] == b'SNP')
INDEL = (y['variants']['VARIANT_TYPE'][:] == b'INDEL')


np.unique(CDS[SNP], return_counts=True)


2+2


y['variants']['VARIANT_TYPE']


pd.value_counts(INDEL)


pd.crosstab(SNPEFF_EFFECT[SNP], CDS[SNP])


2+2


df = pd.DataFrame({'CDS': CDS, 'SNPEFF_EFFECT':SNPEFF_EFFECT})


writer = pd.ExcelWriter("/nfs/users/nfs_r/rp7/SNPEFF_for_Rob.xlsx")
pd.crosstab(SNPEFF_EFFECT, CDS).to_excel(writer)
writer.save()





pd.crosstab(SNPEFF_EFFECT, y['variants']['CHROM'])


np.unique(y['variants']['svlen'], return_counts=True)


y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT_10pc.h5.h5' % output_dir, 'r')
y


# for field in y['variants'].keys():
for field in ['svlen']:
    print(field, np.unique(y['variants'][field], return_counts=True))














get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_no_PID_PGT_10pc     --output {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5 > {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.md5 ')








get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset_1pc     --output {output_dir}/hdf5/Pf_60_subset_1pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_1pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_1pc.h5 > {output_dir}/hdf5/Pf_60_subset_1pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('{output_dir}/scripts/vcfnp_concat.sh {vcf_fn} {output_dir}/hdf5/Pf_60')


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
# inputs=${vcf}.vcfnp_cache
inputs=%s/npy
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''' % (
        output_dir,
        )
      , file=fo)
fo.close()

#     nv=$(ls -1 ${inputs}/v* | wc -l)
#     nc=$(ls -1 ${inputs}/c* | wc -l)
#     echo variants files $nv >> $log
#     echo calldata files $nc >> $log
#     if [ "$nv" -ne "$nc" ]
#     then
#         echo missing npy files
#         exit 1
#     fi


# # Copy files to /nfs

get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5 {nfs_final_hdf5_dir}/')
get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5.md5 {nfs_final_hdf5_dir}/')





# # Plan
# - Create annotated versions (CDS, VARIANT_TYPE, MULTIALLELIC) of chromosome vcfs ala Pf3k, including only fields from 31/10/2016 11:56 email
# - Create WG sites files
# - Create summary tables ala Pf3k

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161122_Pf60_final_vcfs'
vrpipe_vcfs_dir = '/nfs/team112_internal/production_files/Pf/6_0'

gff_fn = "/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3"
cds_gff_fn = "%s/gff/Pfalciparum_GeneDB_Oct2016.Pfalciparum.noseq.gff3.cds.gz" % output_dir
annotations_header_fn = "%s/intermediate_files/annotations.hdr" % (output_dir)

run_create_multiallelics_file_job_fn = "%s/scripts/run_create_multiallelics_file_job.sh" % output_dir
submit_create_multiallelics_file_jobs_fn = "%s/scripts/submit_create_multiallelics_file_jobs.sh" % output_dir


GENOME_FN = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

get_ipython().system('mkdir -p {output_dir}/gff')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/intermediate_files')
get_ipython().system('mkdir -p {output_dir}/tables')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')


cds_gff_fn


run_create_multiallelics_file_job_fn


get_ipython().system("grep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")


fo=open(annotations_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (INDEL)">', file=fo)
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()


fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
# JOB=$LSB_JOBINDEX
JOB=16
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_SITES_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_FULL_VCF_FN=%s/vcf/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.vcf.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pf_60_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN
 
/nfs/users/nfs_r/rp7/anaconda3/bin/python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC \
-Oz -o $ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools view \
--include 'ALT!="*"' | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

# /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
# $NORMALISED_ANNOTATION_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        vrpipe_vcfs_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        cds_gff_fn,
        annotations_header_fn,
        GENOME_FN,
        annotations_header_fn,
        )
        , file=fo)
fo.close()


fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
JOB=$LSB_JOBINDEX
# JOB=16
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_SITES_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_FULL_VCF_FN=%s/vcf/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.txt.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pf_60_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN
 
/nfs/users/nfs_r/rp7/anaconda3/bin/python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\t%%CDS\t%%VARIANT_TYPE\t%%MULTIALLELIC\n' | \
bgzip -c > $ANNOTATION_FN

tabix -s1 -b2 -e2 $ANNOTATION_FN

#/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
#$ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools view \
--include 'ALT!="*"' | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CHROM,POS,REF,ALT,CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_FULL_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC,FORMAT/BCS \
-Oz -o $OUTPUT_VCF_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$OUTPUT_VCF_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        vrpipe_vcfs_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        cds_gff_fn,
        annotations_header_fn,
        GENOME_FN,
        annotations_header_fn,
        cds_gff_fn,
        annotations_header_fn,
        )
        , file=fo)
fo.close()


fo = open(submit_create_multiallelics_file_jobs_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
LOG_DIR=%s/log
 
NUM_CHROMS=`wc -l < $FASTA_FAI_FILE`
QUEUE=long

bsub -q $QUEUE -G malaria-dk -J "ma[1-$NUM_CHROMS]" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s
''' % (
        GENOME_FN,
        output_dir,
        "bash %s" % run_create_multiallelics_file_job_fn,
        ),
     file=fo)
fo.close()


get_ipython().system('bash {submit_create_multiallelics_file_jobs_fn}')





# # Introduction
# The purpose of this notebook is to create an updated version of sample metadata file with:
# - ox_codes instead of Individual ID
# - remove year from unselected dup (for panoptes)
# - add GPS coordinates
# - create new files for Vikki
# - create new Panoptes file for Ben

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


panoptes_previous_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170206.txt.gz"
sites_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_sites_20170206.xlsx"
study_summary_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/study_summary_20170206.xlsx"
samples_by_study_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/samples_by_study_20170206.xlsx"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"
panoptes_samples_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170206.txt"


tbl_panoptes_samples_previous = (
    etl
    .fromtsv(panoptes_previous_metadata_fn)
    .convertnumbers()
)
print(len(tbl_panoptes_samples_previous.data()))
print(len(tbl_panoptes_samples_previous.distinct('Sample').data()))
tbl_panoptes_samples_previous


tbl_duplicates_ox_code = (
    tbl_panoptes_samples_previous
    .aggregate('IndividualGroup', etl.strjoin(','), 'Sample')
    .rename('value', 'AllSamplesThisIndividual')
)


tbl_duplicates_ox_code.select(lambda rec: len(rec['AllSamplesThisIndividual']) > 10)


tbl_sites = (
    etl.fromxlsx(sites_fn)
    .rename('Source', 'Site_source')
)
print(len(tbl_sites.data()))
print(len(tbl_sites.distinct('Site').data()))
tbl_sites


tbl_panoptes_samples_final = (
    tbl_panoptes_samples_previous
    .join(tbl_duplicates_ox_code, key='IndividualGroup')
    .join(tbl_sites, key='Site')
    .sort(['AlfrescoStudyCode', 'Sample'])
)
print(len(tbl_panoptes_samples_final.data()))
print(len(tbl_panoptes_samples_final.distinct('Sample').data()))
tbl_panoptes_samples_final


# File to send to Olivo
tbl_panoptes_samples_final.totsv(panoptes_final_metadata_fn, lineterminator='\n')


# # Create spreadsheet for Vikki

tbl_study_summary = (
    (
        tbl_panoptes_samples_final
        .selecteq('InV5', 'True')
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Total in 5.0')
        .cutout('frequency')
    ).outerjoin(
    (
        tbl_panoptes_samples_final
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Total in 6.0')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .addfield('New in 6.0', lambda rec: rec['Total in 6.0'] if rec['Total in 5.0'] is None else rec['Total in 6.0'] - rec['Total in 5.0'])
    .outerjoin(
    (
        tbl_panoptes_samples_final
            .selectin('Year', ['-', 'N/A'])
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Missing year')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .outerjoin(
    (
        tbl_panoptes_samples_final
            .selectin('Site', ['-'])
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Missing site')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .replaceall(None, 0)
    .sort(['AlfrescoStudyCode'])
)
tbl_study_summary.toxlsx(study_summary_fn)
tbl_study_summary.displayall()


partner_columns = ['Sample', 'OxfordSrcCode', 'Site', 'Year', 'AllSamplesThisIndividual']


from pandas import ExcelWriter

studies = tbl_study_summary.values('AlfrescoStudyCode').list()
writer = ExcelWriter(samples_by_study_fn)
for study in studies:
    print(study)
    if tbl_study_summary.selecteq('AlfrescoStudyCode', study).values('Total in 5.0')[0] > 0:
        df = (
            tbl_panoptes_samples_final
            .selecteq('AlfrescoStudyCode', study)
            .selecteq('InV5', 'True')
            .cut(partner_columns)
            .todataframe()
        )
        sheet_name = "%s_old" % study[0:4]
        df.to_excel(writer, sheet_name, index=False)
    if tbl_study_summary.selecteq('AlfrescoStudyCode', study).values('New in 6.0')[0] > 0:
        df = (
            tbl_panoptes_samples_final
            .selecteq('AlfrescoStudyCode', study)
            .selecteq('InV5', 'False')
            .cut(partner_columns)
            .todataframe()
        )
        sheet_name = "%s_new" % study[0:4]
        df.to_excel(writer, sheet_name, index=False)
writer.save()
    


# # Create new Panaoptes file

tbl_sample_6_0 = (
    etl
    .fromtsv(sample_6_0_fn)
    .cutout('study')
    .cutout('source_code')
    .cutout('run_accessions')
)
print(len(tbl_sample_6_0.data()))
print(len(tbl_sample_6_0.distinct('sample').data()))
tbl_sample_6_0


tbl_panoptes_samples = (
    tbl_panoptes_samples_final
    .addfield('PreferredSample', lambda rec: rec['DiscardAsDuplicate'] == 'False')
    .convert('Year', lambda v: '', where=lambda r: r['PreferredSample'] == False)
    .convert('Date', lambda v: '', where=lambda r: r['PreferredSample'] == False)
    .cutout('DiscardAsDuplicate')
    .join(tbl_sample_6_0, lkey='Sample', rkey='sample')
    .sort(['AlfrescoStudyCode', 'Sample'])
)
print(len(tbl_panoptes_samples.data()))
print(len(tbl_panoptes_samples.distinct('Sample').data()))
tbl_panoptes_samples


tbl_panoptes_samples.valuecounts('DiscardAsDuplicate')


tbl_panoptes_samples.valuecounts('PreferredSample')


tbl_panoptes_samples.valuecounts('Year').displayall()


tbl_panoptes_samples.totsv(panoptes_samples_fn, lineterminator='\n')


# After running the above, ran the following from oscar.well.ox.ac.uk

# rsync -av rp7@malsrv2:/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170206.txt /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/ >> ~/bin/log/Pf_60_Panoptes_20170206.log
# 




get_ipython().run_line_magic('run', '_standard_imports.ipynb')


panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"

hdf_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5"
gff_fn = "/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3"
genome_fn = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
panoptes_samples_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170130.txt"


tbl_panoptes_samples_final = (
    etl
    .fromtsv(panoptes_final_metadata_fn)
    .cutout('pc_pass_missing')
    .cutout('pc_genome_covered_at_1x')
    .cutout('sort_AlfrescoStudyCode')
)
print(len(tbl_panoptes_samples_final.data()))
print(len(tbl_panoptes_samples_final.distinct('Sample').data()))
tbl_panoptes_samples_final


tbl_sample_6_0 = (
    etl
    .fromtsv(sample_6_0_fn)
    .cutout('study')
    .cutout('source_code')
    .cutout('run_accessions')
)
print(len(tbl_sample_6_0.data()))
print(len(tbl_sample_6_0.distinct('sample').data()))
tbl_sample_6_0


tbl_panoptes_samples = (
    tbl_panoptes_samples_final
    .join(tbl_sample_6_0, lkey='Sample', rkey='sample')
)
print(len(tbl_panoptes_samples.data()))
print(len(tbl_panoptes_samples.distinct('Sample').data()))
tbl_panoptes_samples


len(tbl_panoptes_samples.header())


callset = h5py.File(hdf_fn, mode='r')
callset['samples'][:]


v_decode_ascii = np.vectorize(lambda x: x.decode('ascii'))


sample_concordance = (v_decode_ascii(callset['samples'][:]) == tbl_panoptes_samples.values('Sample').array())


np.unique(sample_concordance, return_counts=True)


np.all(v_decode_ascii(callset['samples'][:]) == tbl_panoptes_samples.values('Sample').array())


tbl_panoptes_samples.totsv(panoptes_samples_fn, lineterminator='\n')


tbl_panoptes_samples


# After running the above, ran the following from oscar.well.ox.ac.uk

# mkdir -p /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5
# mkdir -p /kwiat/2/grassi/sanger_mirror/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016
# rsync -av rp7@malsrv2:/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170130.txt /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/ >> ~/bin/log/Pf_60_Panoptes_20170130.log
# rsync -av rp7@malsrv2:/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta /kwiat/2/grassi/sanger_mirror/lustre/scratch116/malaria/pfalciparum/resources/ >> ~/bin/log/Pf_60_Panoptes_20170130.log
# rsync -av rp7@malsrv2:/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3 /kwiat/2/grassi/sanger_mirror/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/ >> ~/bin/log/Pf_60_Panoptes_20170130.log
# rsync -av rp7@malsrv2:/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5 /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/ >> ~/bin/log/Pf_60_Panoptes_20170130.log
# chgrp -R malariagen /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages
# chgrp -R malariagen /kwiat/2/grassi/sanger_mirror/lustre/scratch116/malaria/pfalciparum/resources
# chmod -R 550 /kwiat/2/grassi/sanger_mirror/nfs/team112_internal/production/release_build/Pf/6_0_release_packages
# chmod -R 550 /kwiat/2/grassi/sanger_mirror/lustre/scratch116/malaria/pfalciparum/resources
# 

# # Introduction
# As part of cleanup of lustre, I decided we should copy various WillH_1 outputs used or created by vrpipe.
# 
# Setups used were: 729-747
# 
# I decided it might be good to keep inputs, resources, outputs of HaplotypeCaller and final vcfs
# 
# I have asked Jim and Dushi to archive improved sample bams as a priority, but also outputs of HaplotypeCaller and final vcfs

archive_dir = '/nfs/team112_internal/rp7/data/Pf/WillH_1'
for chrom in ['Pf3D7_%02d_v3' % n for n in range(1, 15)] + ['Pf3D7_API_v3', 'Pf_M76611']:
    get_ipython().system('mkdir -p {"%s/vcf/vcf_symlinks/%s" % (archive_dir, chrom)}')


get_ipython().system('cp -R /lustre/scratch109/malaria/WillH_1/meta {archive_dir}/')


# See emails from:
# Jim 15/06/2016 14:19
# Cinzia 22/03/2016 07:47
# Roberto 14/06/2016 15:17

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = '/nfs/team112_internal/rp7/data/Pf/hrp'
get_ipython().system('mkdir -p {output_dir}/fofns')
get_ipython().system('mkdir -p {output_dir}/metadata')
cinzia_metadata_fn = '%s/metadata/PF_metadata_base.csv' % output_dir # From Cinzia 22/03/2016 07:47
v4_metadata_fn = '%s/metadata/PGV4_mk5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
v5_metadata_fn = '%s/metadata/v5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
country_code_fn = '%s/metadata/country-codes.csv' % output_dir # https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv
regions_in_dataset_fn = '%s/metadata/regions_in_dataset.xlsx' % output_dir
sub_contintent_fn = '%s/metadata/region_sub_continents.xlsx' % output_dir
manifest_fn = '%s/metadata/hrp_manifest.txt' % output_dir
lookseq_fn = '%s/metadata/lookseq.txt' % output_dir


# cinzia_extra_metadata_fn = '/nfs/team112_internal/rp7/data/Pf/4_0/meta/PF_extrametadata.csv' # From Cinzia 22/03/2016 08:22


fofns = collections.OrderedDict()

fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '%s/fofns/olivo_TRAC.fofn' % output_dir
fofns['fanello'] = '%s/fofns/olivo_fanello.fofn' % output_dir


import glob

bam_dirs = collections.OrderedDict()
bam_dirs['trac'] = '/nfs/team112_internal/production_files/Pf/olivo_TRAC/remapped'
bam_dirs['fanello'] = '/nfs/team112_internal/production_files/Pf/olivo_fanello'

for bam_dir in bam_dirs:
    get_ipython().system('rm {fofns[bam_dir]}')
    with open(fofns[bam_dir], "a") as fofn:
# glob.glob('%s/*.bam' % fofns['trac'])
        print("path\tsample", file=fofn)
        for x in glob.glob('%s/*.bam' % bam_dirs[bam_dir]):
            print("%s\t%s" % (x, os.path.basename(x).replace('_', '-').replace('.bam', '')), file=fofn)
# [os.path.basename(x) for x in glob.glob('%s/*.bam' % fofns['trac'])]


for fofn in fofns:
    print(fofn)
    get_ipython().system('head -n 1 {fofns[fofn]}')
    get_ipython().system('wc -l {fofns[fofn]}')
    print()


for i, fofn in enumerate(fofns):
    if i == 0:
        tbl_all_bams = etl.fromtsv(fofns[fofn]).cut(['path', 'sample']).addfield('dataset', fofn)
    else:
        if fofn == 'pf3k_pilot_5_0_broad':
            tbl_all_bams = tbl_all_bams.cat(etl.fromtsv(fofns[fofn]).selecteq('study', 'Pf3k_Senegal').cut(['path', 'sample']).addfield('dataset', fofn))
        else:
            tbl_all_bams = tbl_all_bams.cat(etl.fromtsv(fofns[fofn]).cut(['path', 'sample']).addfield('dataset', fofn))
        


len(tbl_all_bams.data())


len(tbl_all_bams.duplicates('sample').data())


len(tbl_all_bams.unique('sample').data())


tbl_all_bams


tbl_all_bams.valuecounts('dataset').displayall()


tbl_solaris = (
    etl.fromcsv(cinzia_metadata_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'country_code'])
    .rename('country', 'solaris_country')
    .rename('country_code', 'solaris_country_code')
    .distinct('oxford_code')
)
tbl_solaris


tbl_solaris.duplicates('oxford_code').displayall()


tbl_v4_metadata = etl.fromxlsx(v4_metadata_fn, 'PGV4.0').cut(['Sample', 'Region']).rename('Region', 'v4_region')
tbl_v5_metadata = etl.fromxlsx(v5_metadata_fn).cut(['Sample', 'Region']).rename('Region', 'v5_region')


tbl_v4_metadata


tbl_v5_metadata


def determine_region(rec, null_vals=(None, 'NULL', '-')):
    if (
        rec['sample'].startswith('PG') or
        rec['sample'].startswith('PL') or
        rec['sample'].startswith('PF') or
        rec['sample'].startswith('WL') or
        rec['sample'].startswith('WH') or
        rec['sample'].startswith('WS')
    ):
        return('Lab')
    if rec['v5_region'] not in null_vals:
        return(rec['v5_region'])
    elif rec['v4_region'] not in null_vals:
        return(rec['v4_region'])
    elif rec['sample'].startswith('PJ'):
        return('ID')
#     elif rec['sample'].startswith('QM'):
#         return('MG')
#     elif rec['sample'].startswith('QS'):
#         return('MG')
    elif rec['solaris_country_code'] not in null_vals:
        return(rec['solaris_country_code'])
    elif rec['solaris_country'] not in null_vals:
        return(rec['solaris_country'])
    else:
        return('unknown')
       


tbl_regions = (
    etl.fromxlsx(v4_metadata_fn, 'Locations')
    .cut(['Country', 'Region'])
    .rename('Country', 'country_from_region')
    .rename('Region', 'region')
    .selectne('region', '-')
    .distinct(['country_from_region', 'region'])
)
tbl_regions.displayall()


tbl_country_code = (
    etl.fromxlsx(v4_metadata_fn, 'CountryCodes')
    .rename('County', 'country')
    .rename('Code', 'code_from_country')
    .rename('SubContinent', 'subcontintent')
    .selectne('country', '-')
)
tbl_country_code


def determine_region_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['code_from_country'] not in null_vals:
        return(rec['code_from_country'])
    elif rec['dataset'] == 'pf3k_pilot_5_0_broad':
        return('SN')
    else:
        return(rec['region'])


def determine_country_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['country_from_region'] not in null_vals:
        return(rec['country_from_region'])
    else:
        return(rec['region_code'])


tbl_country_codes = (
    etl.fromcsv(country_code_fn)
    .cut(['official_name', 'ISO3166-1-Alpha-2'])
    .rename('official_name', 'country_name')
)
tbl_country_codes


tbl_sub_continents = etl.fromxlsx(sub_contintent_fn)
tbl_sub_continents


tbl_sub_continent_names = etl.fromxlsx(sub_contintent_fn, 'Names').convertnumbers()
tbl_sub_continent_names


final_fields = [
    'path', 'sample', 'dataset', 'type', 'region_code', 'country_code', 'country_name', 'sub_continent', 'sub_continent_name', 'sub_continent_number']

tbl_manifest = (
    tbl_all_bams
    .leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code')
    .replace('type', None, 'unknown')
    .leftjoin(tbl_v4_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_v5_metadata, lkey='sample', rkey='Sample')
    .addfield('region', determine_region)
    .leftjoin(tbl_country_code.cut(['country', 'code_from_country']), lkey='region', rkey='country')
    .addfield('region_code', determine_region_code)
    .replace('region_code', 'Benin', 'BJ')
    .replace('region_code', 'Mauritania', 'MR')
    .replace('region_code', "Cote d'Ivoire (Ivory Coast)", 'CI')
    .replace('region_code', 'Ethiopia', 'ET')
    .leftjoin(tbl_regions, lkey='region_code', rkey='region')
    .addfield('country_code', determine_country_code)
    .leftjoin(tbl_country_codes, lkey='country_code', rkey='ISO3166-1-Alpha-2')
    .leftjoin(tbl_sub_continents.cut(['region_code', 'sub_continent']), key='region_code')
    .leftjoin(tbl_sub_continent_names, key='sub_continent')
    .selectne('region_code', 'unknown')
    .sort(['sub_continent_number', 'country_name', 'sample'])
    .cut(final_fields)
)


len(tbl_all_bams.leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code').data())


tbl_manifest.totsv(manifest_fn)


manifest_fn


len(tbl_manifest.data())


len(tbl_manifest.distinct('sample').data())


tbl_temp = tbl_manifest.addfield('bam_exists', lambda rec: os.path.exists(rec['path']))
tbl_temp.valuecounts('bam_exists')


with open(lookseq_fn, "w") as fo:
    for rec in tbl_manifest:
        bam_fn = rec[0]
        sample_name = "%s_%s" % (rec[1].replace('-', '_'), rec[3])
        group_name = "%s %s %s %s" % (rec[9], rec[7], rec[6], rec[4])
        print(
            '"%s" : { "bam":"%s", "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "%s" } ,' % (sample_name, bam_fn, group_name),
            file=fo
        )
# "PG0049_CW2" : { "bam":"/lustre/scratch109/malaria/pfalciparum/output/e/b/8/6/144292/1_bam_merge/pe.1.bam" , "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "1104-PF-LAB-WENDLER" } ,


2+2


tbl_manifest


tbl_manifest.tail(5)


tbl_manifest.select(lambda rec: rec['sample'].startswith('QZ')).displayall()


tbl_manifest.selecteq('sample', 'WL0071-C').displayall()


tbl_manifest.selecteq('country_code', 'UK').displayall()


tbl_manifest.valuecounts('type', 'dataset').displayall()


lkp_country_code = etl.lookup(tbl_country_code, 'code', 'country')


tbl_manifest.selecteq('type', 'unknown').selecteq('dataset', 'pf_community_5_1')


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None and rec['v4_region'] != rec['v5_region'])


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is None)


lkp_code_country = etl.lookup(tbl_country_code, 'country', 'code')


lkp_country_code['BZ']


tbl_manifest.valuecounts('region').displayall()


tbl_manifest.selecteq('region', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('region_code').displayall()


tbl_manifest.selecteq('region_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_code').displayall()


tbl_manifest.selecteq('country_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_name').displayall()


tbl_manifest.valuecounts('region_code', 'country_name').toxlsx(regions_in_dataset_fn)


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.cut(['path', 'sample', 'dataset', 'type', 'region_code', 'country_code', 'country_name', 'sub_continent'])





# See emails from:
# 
# Jim 15/06/2016 14:19
# 
# Cinzia 22/03/2016 07:47
# 
# Roberto 14/06/2016 15:17

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = '/nfs/team112_internal/rp7/data/Pf/hrp'
get_ipython().system('mkdir -p {output_dir}/fofns')
get_ipython().system('mkdir -p {output_dir}/metadata')
cinzia_metadata_fn = '%s/metadata/PF_metadata_base.csv' % output_dir # From Cinzia 22/03/2016 07:47
v4_metadata_fn = '%s/metadata/PGV4_mk5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
v5_metadata_fn = '%s/metadata/v5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
iso_country_code_fn = '%s/metadata/country-codes.csv' % output_dir # https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv
regions_in_dataset_fn = '%s/metadata/regions_in_dataset.xlsx' % output_dir
sub_contintent_fn = '%s/metadata/region_sub_continents.xlsx' % output_dir
manifest_fn = '%s/metadata/hrp_manifest_20160620.txt' % output_dir
jim_manifest_fn = '%s/metadata/manifest_for_jim_20160620.txt' % output_dir
lookseq_fn = '%s/metadata/lookseq.txt' % output_dir

lab_studies = list(range(1032, 1044, 1)) + [1104, 1133, 1150, 1153]

# cinzia_extra_metadata_fn = '/nfs/team112_internal/rp7/data/Pf/4_0/meta/PF_extrametadata.csv' # From Cinzia 22/03/2016 08:22


fofns = collections.OrderedDict()

fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '%s/fofns/olivo_TRAC.fofn' % output_dir
fofns['fanello'] = '%s/fofns/olivo_fanello.fofn' % output_dir


import glob

bam_dirs = collections.OrderedDict()
bam_dirs['trac'] = '/nfs/team112_internal/production_files/Pf/olivo_TRAC/remapped'
bam_dirs['fanello'] = '/nfs/team112_internal/production_files/Pf/olivo_fanello'

for bam_dir in bam_dirs:
    get_ipython().system('rm {fofns[bam_dir]}')
    with open(fofns[bam_dir], "a") as fofn:
# glob.glob('%s/*.bam' % fofns['trac'])
        print("path\tsample", file=fofn)
        for x in glob.glob('%s/*.bam' % bam_dirs[bam_dir]):
            print("%s\t%s" % (x, os.path.basename(x).replace('_', '-').replace('.bam', '')), file=fofn)
# [os.path.basename(x) for x in glob.glob('%s/*.bam' % fofns['trac'])]


for i, fofn in enumerate(fofns):
    if i == 0:
        tbl_all_bams = etl.fromtsv(fofns[fofn]).cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage']).addfield('dataset', fofn)
    else:
        if fofn == 'pf3k_pilot_5_0_broad':
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .selecteq('study', 'Pf3k_Senegal')
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .addfield('dataset', fofn)
                )
            )
        elif fofn in ['pf_community_5_0', 'conway']:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .addfield('dataset', fofn)
                )
            )
        else:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample'])
                    .addfield('dataset', fofn)
                )
            )
        


len(tbl_all_bams.data())


len(tbl_all_bams.duplicates('sample').data())


len(tbl_all_bams.unique('sample').data())


tbl_all_bams


tbl_all_bams.valuecounts('dataset').displayall()


tbl_solaris = (
    etl.fromcsv(cinzia_metadata_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'country_code', 'oxford_src_code', 'oxford_donor_code', 'alfresco_code'])
    .convert('alfresco_code', int)
    .rename('country', 'solaris_country')
    .rename('country_code', 'solaris_country_code')
    .distinct('oxford_code')
)
tbl_solaris.selectne('alfresco_code', None)


tbl_solaris.duplicates('oxford_code').displayall()


tbl_v4_metadata = etl.fromxlsx(v4_metadata_fn, 'PGV4.0').cut(['Sample', 'Region']).rename('Region', 'v4_region')
tbl_v5_metadata = etl.fromxlsx(v5_metadata_fn).cut(['Sample', 'Region']).rename('Region', 'v5_region')


tbl_v4_metadata


tbl_v4_metadata.selecteq('Sample', 'PF0542-C')


tbl_v5_metadata


tbl_v5_metadata.selecteq('Sample', 'PF0542-C')


def determine_region(rec, null_vals=(None, 'NULL', '-')):
#     if (
#         rec['sample'].startswith('PG') or
#         rec['sample'].startswith('PL') or
#         rec['sample'].startswith('PF') or
#         rec['sample'].startswith('WL') or
#         rec['sample'].startswith('WH') or
#         rec['sample'].startswith('WS')
#     ):
    if rec['alfresco_code'] in lab_studies:
        return('Lab')
    if rec['v5_region'] not in null_vals:
        return(rec['v5_region'])
    elif rec['v4_region'] not in null_vals:
        return(rec['v4_region'])
    elif rec['sample'].startswith('PJ'):
        return('ID')
#     elif rec['sample'].startswith('QM'):
#         return('MG')
#     elif rec['sample'].startswith('QS'):
#         return('MG')
    elif rec['solaris_country_code'] not in null_vals:
        return(rec['solaris_country_code'])
    elif rec['solaris_country'] not in null_vals:
        return(rec['solaris_country'])
    else:
        return('unknown')
       


tbl_regions = (
    etl.fromxlsx(v4_metadata_fn, 'Locations')
    .cut(['Country', 'Region'])
    .rename('Country', 'country_from_region')
    .rename('Region', 'region')
    .selectne('region', '-')
    .distinct(['country_from_region', 'region'])
)
tbl_regions.selecteq('country_from_region', 'KH')


tbl_country_code = (
    etl.fromxlsx(v4_metadata_fn, 'CountryCodes')
    .rename('County', 'country')
    .rename('Code', 'code_from_country')
    .rename('SubContinent', 'subcontintent')
    .selectne('country', '-')
)
tbl_country_code.displayall()


def determine_region_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['code_from_country'] not in null_vals:
        return(rec['code_from_country'])
    elif rec['dataset'] == 'pf3k_pilot_5_0_broad':
        return('SN')
    else:
        return(rec['region'])


def determine_country_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['country_from_region'] not in null_vals:
        return(rec['country_from_region'])
    else:
        return(rec['region_code'])


tbl_iso_country_codes = (
    etl.fromcsv(iso_country_code_fn)
    .cut(['official_name', 'ISO3166-1-Alpha-2'])
    .rename('official_name', 'country_name')
)
tbl_iso_country_codes


tbl_sub_continents = etl.fromxlsx(sub_contintent_fn)
tbl_sub_continents


tbl_sub_continent_names = etl.fromxlsx(sub_contintent_fn, 'Names').convertnumbers()
tbl_sub_continent_names


final_fields = [
    'path', 'sample', 'oxford_src_code', 'oxford_donor_code', 'dataset', 'type', 'region_code', 'country_code',
    'country_name', 'sub_continent', 'sub_continent_name', 'sub_continent_number', 'bases_of_5X_coverage', 'mean_coverage'
]

tbl_manifest = (
    tbl_all_bams
    .leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code')
    .replace('type', None, 'unknown')
    .leftjoin(tbl_v4_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_v5_metadata, lkey='sample', rkey='Sample')
    .addfield('region', determine_region)
    .leftjoin(tbl_country_code.cut(['country', 'code_from_country']), lkey='region', rkey='country')
    .addfield('region_code', determine_region_code)
    .replace('region_code', 'Benin', 'BJ')
    .replace('region_code', 'Mauritania', 'MR')
    .replace('region_code', "Cote d'Ivoire (Ivory Coast)", 'CI')
    .replace('region_code', 'Ethiopia', 'ET')
    .replace('region_code', 'US', 'Lab')
    .leftjoin(tbl_regions, lkey='region_code', rkey='region')
    .addfield('country_code', determine_country_code)
    .leftjoin(tbl_iso_country_codes, lkey='country_code', rkey='ISO3166-1-Alpha-2')
    .replace('country_name', "Côte d'Ivoire", "Ivory Coast")
    .leftjoin(tbl_sub_continents.cut(['region_code', 'sub_continent']), key='region_code')
    .leftjoin(tbl_sub_continent_names, key='sub_continent')
    .selectne('region_code', 'unknown')
    .sort(['sub_continent_number', 'country_name', 'sample'])
    .cut(final_fields)
)


tbl_manifest.selecteq('country_code', 'CI')


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.selectnone('sub_continent').displayall()


tbl_manifest.selecteq('sub_continent', 'Lab')


len(tbl_all_bams.leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code').data())


manifest_fn


tbl_manifest.totsv(manifest_fn)


tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).totsv(jim_manifest_fn)


manifest_fn


len(tbl_manifest.data())


len(tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).data())


len(tbl_manifest.distinct('sample').data())


tbl_temp = tbl_manifest.addfield('bam_exists', lambda rec: os.path.exists(rec['path']))
tbl_temp.valuecounts('bam_exists')


with open(lookseq_fn, "w") as fo:
    for rec in tbl_manifest:
        bam_fn = rec[0]
        sample_name = "%s_%s" % (rec[1].replace('-', '_'), rec[3])
        group_name = "%s %s %s %s" % (rec[9], rec[7], rec[6], rec[4])
        print(
            '"%s" : { "bam":"%s", "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "%s" } ,' % (sample_name, bam_fn, group_name),
            file=fo
        )
# "PG0049_CW2" : { "bam":"/lustre/scratch109/malaria/pfalciparum/output/e/b/8/6/144292/1_bam_merge/pe.1.bam" , "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "1104-PF-LAB-WENDLER" } ,


2+2


tbl_manifest


tbl_manifest.tail(5)


tbl_manifest.select(lambda rec: rec['sample'].startswith('QZ')).displayall()


tbl_manifest.selecteq('sample', 'WL0071-C').displayall()


tbl_manifest.selecteq('country_code', 'UK').displayall()


tbl_manifest.valuecounts('type', 'dataset').displayall()


lkp_country_code = etl.lookup(tbl_country_code, 'code', 'country')


tbl_manifest.selecteq('type', 'unknown').selecteq('dataset', 'pf_community_5_1')


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None and rec['v4_region'] != rec['v5_region'])


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is None and rec['v5_region'] is not None)


tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is None)


lkp_code_country = etl.lookup(tbl_country_code, 'country', 'code')


lkp_country_code['BZ']


tbl_manifest.valuecounts('region').displayall()


tbl_manifest.selecteq('region', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('region_code').displayall()


tbl_manifest.selecteq('region_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_code').displayall()


tbl_manifest.selecteq('country_code', 'unknown').valuecounts('dataset').displayall()


tbl_manifest.valuecounts('country_name').displayall()


tbl_manifest.valuecounts('region_code', 'country_name').toxlsx(regions_in_dataset_fn)


tbl_manifest.valuecounts('sub_continent').displayall()


tbl_manifest.cut(['path', 'sample', 'dataset', 'type', 'region_code', 'country_code', 'country_name', 'sub_continent'])





# # Introduction
# As part of cleanup of lustre, I decided we should copy the output created by Jim (see email 23/06/2016 19:57) using vrpipe pipelines pf_hrp_haplotype_caller, pf_hrp_combine_gvcfs, pf_hrp_genotype_gvcfs and pf_hrp_variant_annotation_using_snpeff_gatk_vcf_annotate
# 
# I decided it might be good to keep both outputs of HaplotypeCaller and final vcf

archive_dir = '/nfs/team112_internal/rp7/data/Pf/hrp/vcf'
get_ipython().system('mkdir -p {archive_dir}')


# # Introduction
# On looking at results of faceaway reads analysis, noticed there were many samples with very low proportions. As one example PG0025-C has potential calls in 5 different MDR1 duplications. Decided to donwload the bam to look in IGV

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


get_ipython().system('grep PG0025 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD0720 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD1496 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD1333 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep QC0147 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD1125 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD0789 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('grep PD1515 /nfs/team112_internal/rp7/data/methods-dev/analysis/20160926_pf_60_duplications/pf_6_0_bams.txt')


get_ipython().system('ls -al /lustre/scratch116/malaria/pfalciparum/output/b/0/5/3/37922/4_bam_mark_duplicates_v2/pe.1.markdup.bam')


# # Current plans for 2016
# 
# 
# ## Copy all data to MacBook and create raw data files
# - Create _standard_imports (based on Docker), plotting_setup and _shared_setup (VCF_FNS, NPY_FNS, HDF5_FNS, GENOME_FASTA_FNS, etc.) in notebook 2016
# - Notebook to rsync all relevant data (3/4/5 and 5 samples vcfs, fasta, gff, mask bed, snpeff)
# - Notebook to merge vcfs, create npy and hdf5 files
# 
# 
# ## Find problems with Interim 5 and create Interim 5.1
# - Create scatter plots of VQSLOD and input metrics of Release 4 vs Interim 5
# - Separate plots for training and other SNPs for all input variable and for VQSLOD
# - Breakdown of missingness in all input variables (ReadPosRankSum presumably very high?)
# - Do different peaks in VQSLOD scores correspond to presence/lack of ReadPosRankSum?
# - Redo VQSR without ReadPosRankSum - improvement in distribution of VQSLOD scores?
# - Manually run VQSR on Release 4 and Interim 5 call sets, with and without ReadPosRankSum. Improvement in VQSLOD scores? Still discrepancy between 4 and 5?
# - Run VQSR with GATK hard thresholds as training set (excluding RankSum variables) for 4 and 5
# - Manually run VQSR on intersection of SNPs between Release 4 and Interim 5 call sets
# - Run VQSR on intersection SNPs on subsets of input parameters
# - Once key problems fixed, manually create new vcf file, and give to Joe as interim
# 
# 
# ## Build release 5 candidate
# - Follow up with Valentin about converting between haploid and diploid calls. Are these simple functions of PL? Is it possible to covert diploid to haploid, but not vice-versa?
# - Check with Cinzia on status of 16 lab samples
# - Add StrandAlleleCountsBySample annotation to HaplotypeCaller step. Check that this does indeed give filtered depths (i.e. always same or lower than AD).
# - Run with GATK 3.5+ (in haploid mode?)
# - Ensure we now have a clear merging strategy
# - Implement changes identified to fix VQSLOD problems
# - Create release 5 candidate inc 16 samples if ready
# 
# 
# ## Single call set evaluation machinery
# - Sensitivity and FDR using “nucmer” truth set
# - Sensitivity and FDR using “unclustered nucmer” truth set
# - Sensitivity and FDR using “bwa/GATK” truth set
# - Mendelian error rate(s) in crosses
# - Genotype error rate(s) in mixtures
# - Duplicate discordance rate(s)
# - Mean heterozygosity or FWS in clonal lab samples
# - Ti/Tv ratio (SNP coding/noncoding only)
# - % frameshifts (indel coding only)
# - % coding/noncoding genome accessible
# 
# 
# ## Evaluation metrics for release 3, 4 and 5
# - Run above evaluation metrics on all vcfs
# - Present results back to group
# 
# 
# ## Release 5 candidate deltas
# - Inclusion of hyperhet filter
# - Core definition based on dust score and uniqueness
# - Genotyping by GATK GT, 5/2 rule, 5/3 rule
# - Run above evaluation metrics on all vcfs
# - Present results back to group
# 
# 
# ## Two call set comparison machinery
# - Scatter of variant metrics (VQSLOD and other) for intersection variants in both call sets
# - Above metrics for variant intersection and setdiffs
# - Above metrics where callset 2 is re-thresholded to have same number of variants as callset 1
# - Histograms of setdiff variant metrics
# - Scatter of heterozygosity, missingness, etc for samples in both call sets
# - Comparisons of discordant unfiltered calls for 5 validation samples
# - Run above evaluation metrics on all vcfs
# - Present results back to group
# 
# 
# ## Create evaluation samples set
# - To include two versions of 5 lab strains, crosses, mixtures and ~100 mixed field samples (diff read lengths)
# - Run above evaluation metrics on release 5 vs evaluation samples
# - Present results back to group
# 
# 
# ## Explore use of filtered allele depths
# - If response at http://gatkforums.broadinstitute.org/gatk/discussion/comment/25870/#Comment_25870 add StrandAlleleCountsBySample to future builds
# - If not, re-create 5 samples call set
# 
# 
# ## Decision on Docker strategy
# - Image on Docker hub that covers all the above
# - Discuss with Nick and Alistair forward Docker strategy
# 




# #Plan
# - Figure out how /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz was created and document here
# - Filter out AC=0 to get file size down and remove unnecessary INFO fields
# - Annotate each variant with a multiallelic flag by filter multiallelic to sep vcf then bcftools annotate with this
# - Create biallelic (and left aligned?) version using bcftools norm
# - Annotate STRs with GATK
# - Create SNP/indel/STR by filtering out different sets and setting appropriate INFO with bcftools annotate
# - Annotate coding flag by filtering to new vcf and then bcftools annotate
# 
# 
# - Create function to create biallelic, 5/2 rule, new AF, segregating, minimal, renormalised VCF
# - Split the above into SNPs and INDELs
# - Test function on small subset of chr14
# - Run function on chrom 14
# - New function to also create npy file
# - Read in chr14 npy file, and calculate Mendelian error and genotype concordance
# - Attempt to reannotate above with STR and SNPEFF annotations
# - Rerun scripts to get breakdown by SNP/STR/nonSTR coding/noncoding

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
crosses_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)
sites_only_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.sites.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'
gff_fn = "%s/Pfalciparum.noseq.gff3.gz" % output_dir
cds_gff_fn = "%s/Pfalciparum.noseq.gff3.cds.gz" % output_dir

results_table_fn = "%s/genotype_quality.xlsx" % output_dir
counts_table_fn = "%s/variant_counts.xlsx" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
GATK = '/software/jre1.7.0_25/bin/java -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.4-46/GenomeAnalysisTK.jar'


gff_fn


# !wget ftp://ftp.sanger.ac.uk/pub/project/pathogens/gff3/2016-06/Pfalciparum.noseq.gff3.gz \
#     -O {gff_fn}


# !zgrep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}


crosses_vcf_fn


multiallelic_header_fn = "%s/vcf/MULTIALLELIC.hdr" % (output_dir)
fo=open(multiallelic_header_fn, 'w')
print('##INFO=<ID=MULTIALLELIC,Number=0,Type=Flag,Description="Is position multiallelic">', file=fo)
fo.close()


variant_type_header_fn = "%s/vcf/VARIANT_TYPE.hdr" % (output_dir)
fo=open(variant_type_header_fn, 'w')
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP, INDEL or STR">', file=fo)
fo.close()


cds_header_fn = "%s/vcf/CDS.hdr" % (output_dir)
fo=open(cds_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
fo.close()


def create_analysis_vcf(input_vcf_fn=crosses_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['nonref', 'multiallelic', 'multiallelics', 'biallelic', 'str', 'snps', 'indels', 'strs',
                              'variant_type', 'coding', 'analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)
    
    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        if region is not None:
            get_ipython().system('{BCFTOOLS} annotate --regions {region} --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        else:
            get_ipython().system('{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t1\\n' --include 'N_ALT>1' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
#         !{BCFTOOLS} index --tbi {intermediate_fns['multiallelic']}
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC         -h {multiallelic_header_fn}         -Oz -o {intermediate_fns['multiallelics']} {intermediate_fns['nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['str']):
        get_ipython().system("{GATK} -T VariantAnnotator             -R {GENOME_FN}             -o {intermediate_fns['str']}             -A TandemRepeatAnnotator              -V {intermediate_fns['biallelic']}")
#         !{BCFTOOLS} index --tbi {intermediate_fns['str']}

    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'str\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp" && STR=0\' {intermediate_fns[\'str\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

    if rewrite or not os.path.exists(intermediate_fns['strs']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSTR\\n\' --include \'TYPE!="snp" && STR=1\' {intermediate_fns[\'str\']} | bgzip -c > {intermediate_fns[\'strs\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'strs\']}')

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['str']} |        {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE |        {BCFTOOLS} annotate -a {intermediate_fns['strs']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")


create_analysis_vcf()


create_analysis_vcf(region=None)


output_dir





tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
vcf_reader = vcf.Reader(filename=analysis_vcf_fn)
sample_ids = np.array(vcf_reader.samples)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE', 'RU',
                'SNPEFF_EFFECT', 'AC', 'AN', 'RPA', 'CDS', 'MULTIALLELIC',
                'VQSLOD', 'FILTER'],
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'VARIANT_TYPE':             'a3',
            'RU':                       'a40',
            'SNPEFF_EFFECT':            'a33',
            'CDS':                      bool,
            'MULTIALLELIC':             bool,
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
        },
        flatten_filter=True,
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD', 'DP', 'GQ'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )


# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
# create_variants_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
# create_calldata_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
create_variants_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))
create_calldata_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))


analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
variants = np.load("%s.vcfnp_cache/variants.npy" % analysis_vcf_fn)
calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % analysis_vcf_fn)


def genotype_concordance_gatk(calldata=calldata,
                              ix = ((variants['VARIANT_TYPE'] == b'SNP') & (variants['MULTIALLELIC'] == False) &
                                    (variants['CDS']) & variants['FILTER_PASS']),
                              GQ_threshold=30,
                              rep_index_first=rep_index_first, rep_index_second=rep_index_second,
                              verbose=False):
    GT = calldata['GT'][ix, :]
    GT[calldata['GQ'][ix, :] < GQ_threshold] = b'./.'
    
    all_samples = sample_ids
#     all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = GT[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = GT[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = GT[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    GT_both = (np.in1d(GT[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(GT[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    GT_both = (
        ((GT[:, rep_index_first] == b'0/0') | (GT[:, rep_index_first] == b'1/1')) &
        ((GT[:, rep_index_second] == b'0/0') | (GT[:, rep_index_second] == b'1/1'))
    )
    GT_discordant = (
        ((GT[:, rep_index_first] == b'0/0') & (GT[:, rep_index_second] == b'1/1')) |
        ((GT[:, rep_index_first] == b'1/1') & (GT[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(GT == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (GT.shape[0] * GT.shape[1])
    heterozygosity_per_sample = np.sum(GT == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (GT.shape[0] * GT.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(GT_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(GT_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(GT_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
#         prop_discordances_per_sample_pair,
#         GT.shape
    )
    


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


results_list = list()
GQ_thresholds = [30, 99, 0]
variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [False, True]
codings = [True, False]
filter_passes = [True, False]

for GQ_threshold in GQ_thresholds:
    for filter_pass in filter_passes:
        for variant_type in variant_types:
            for coding in codings:
                for multiallelic in multiallelics:
                    print(GQ_threshold, filter_pass, variant_type, coding, multiallelic)
                    ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
                    number_of_alleles = np.sum(ix)
                    mean_nraf = np.sum(variants['AC'][ix]) / np.sum(variants['AN'][ix])
                    genotype_quality_results = list(genotype_concordance_gatk(ix=ix, GQ_threshold=GQ_threshold))
                    results_list.append(
                        [GQ_threshold, filter_pass, variant_type, coding, multiallelic, number_of_alleles, mean_nraf] +
                        genotype_quality_results
                    )

print(results_list)


results_list = list()
GQ_thresholds = [30, 99, 0]
variant_types = collections.OrderedDict()
variant_types['SNP'] = [b'SNP']
variant_types['indel'] = [b'IND', b'STR']
multiallelics = [False, True]
codings = [True, False]
filter_passes = [True, False]

for GQ_threshold in GQ_thresholds:
    for filter_pass in filter_passes:
        for variant_type in variant_types:
            for coding in codings:
                for multiallelic in multiallelics:
                    print(GQ_threshold, filter_pass, variant_type, coding, multiallelic)
                    ix = (
                        (np.in1d(variants['VARIANT_TYPE'], variant_types[variant_type])) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
                    number_of_alleles = np.sum(ix)
                    mean_nraf = np.sum(variants['AC'][ix]) / np.sum(variants['AN'][ix])
                    genotype_quality_results = list(genotype_concordance_gatk(ix=ix, GQ_threshold=GQ_threshold))
                    results_list.append(
                        [GQ_threshold, filter_pass, variant_type, coding, multiallelic, number_of_alleles, mean_nraf] +
                        genotype_quality_results
                    )

# print(results_list)


headers = ['GQ threshold', 'PASS', 'Type', 'Coding', 'Multiallelic', 'Alleles', 'Mean NRAF', 'Missingness',
           'Heterozygosity', 'Discordance', 'MER']
etl.wrap(results_list).pushheader(headers).displayall()


# etl.wrap(results_list).pushheader(headers).convert('Alleles', int).toxlsx(results_table_fn)
etl.wrap(results_list).pushheader(headers).cutout('Alleles').cutout('Mean NRAF').toxlsx(results_table_fn)
results_table_fn





variant_type_header_2_fn = "%s/vcf/VARIANT_TYPE_2.hdr" % (output_dir)
fo=open(variant_type_header_2_fn, 'w')
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or INDEL">', file=fo)
fo.close()


def create_variant_counts_vcf(input_vcf_fn=sites_only_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.sites.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['multiallelic', 'multiallelics', 'snps', 'indels',
                              'variant_type', 'analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)
    
    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t1\\n' --include 'N_ALT>1' {input_vcf_fn} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set        -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC        -h {multiallelic_header_fn}         -Oz -o {intermediate_fns['multiallelics']} {input_vcf_fn}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
                
    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'multiallelics\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'multiallelics\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_2_fn} {intermediate_fns['multiallelics']} |         {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")


create_variant_counts_vcf()


create_variant_counts_vcf(region=None)


def create_variants_npy_2(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
                'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
                'VQSLOD', 'FILTER'],
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'VARIANT_TYPE':             'a3',
            'SNPEFF_EFFECT':            'a33',
            'CDS':                      bool,
            'MULTIALLELIC':             bool,
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
        },
        flatten_filter=True,
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )


create_variants_npy_2("%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir))


sites_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
variants_all = np.load("%s.vcfnp_cache/variants.npy" % sites_vcf_fn)


counts_list = list()
GQ_thresholds = [30, 99, 0]
variant_types = [b'SNP', b'IND']
multiallelics = [False, True]
codings = [True, False]
filter_passes = [True, False]

for filter_pass in filter_passes:
    for variant_type in variant_types:
        for coding in codings:
            for multiallelic in multiallelics:
                print(filter_pass, variant_type, coding, multiallelic)
                ix = (
                    (variants_all['VARIANT_TYPE'] == variant_type) &
                    (variants_all['MULTIALLELIC'] == multiallelic) &
                    (variants_all['CDS'] == coding) &
                    (variants_all['FILTER_PASS'] == filter_pass)
                )
                number_of_variants = np.sum(ix)
                counts_list.append(
                    [filter_pass, variant_type, coding, multiallelic, number_of_variants]
                )

print(counts_list)


headers = ['PASS', 'Type', 'Coding', 'Multiallelic', 'Variants']
etl.wrap(counts_list).pushheader(headers).displayall()


etl.wrap(counts_list).pushheader(headers).convert('Variants', int).toxlsx(counts_table_fn)
counts_table_fn


np.sum(etl
       .fromtsv('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/passcounts')
       .pushheader(('CHROM', 'count'))
       .convertnumbers()
       .values('count')
       .array()
       )





# # Plan
# - Determine invariant sequence near start of exon 4
# - grep this in PD0479-C and PD0471-C, both Pf3k and Pf 5.0
# - If we find any , how well do they map in their other location? Does CRT appear as alternative mapping location?
# - If reads in 1) and 2), are there any bwa parameters that might stop this?
# - Look into Thomas assemblies to see what true variation around here looks like. Is 3D7II similar to 3D7? Is intron between 3 and 4 very different between isolates?
# 

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


bam_fns = collections.OrderedDict()
bam_fns['PD0479-C Pf3k'] = '/lustre/scratch109/malaria/pf3k_methods/output/2/8/4/f/290788/4_bam_mark_duplicates_v2/pe.1.markdup.bam'
bam_fns['PD0471-C Pf3k'] = '/lustre/scratch109/malaria/pf3k_methods/output/8/3/4/7/290780/4_bam_mark_duplicates_v2/pe.1.markdup.bam'
bam_fns['PD0479-C Pf 5.0'] = '/lustre/scratch109/malaria/pfalciparum/output/4/4/3/3/43216/1_bam_merge/pe.1.bam'
bam_fns['PD0471-C Pf 5.0'] = '/lustre/scratch109/malaria/pfalciparum/output/f/5/2/7/43208/1_bam_merge/pe.1.bam'

vcf_fns = collections.OrderedDict()
vcf_fns['Pf3k'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.vcf.gz'
vcf_fns['Pf 5.0'] = '/nfs/team112_internal/production_files/Pf/5_0/pf_50_vfp1.newCoverageFilters_pass_5_99.5.vcf.gz'

exon4_coordinates = 'Pf3D7_07_v3:404283-404415'

seeds = collections.OrderedDict()
seeds['Pf3D7_07_v3:404290-404310'] = 'TTATACAATTATCTCGGAGCA'
seeds['Pf3D7_07_v3:404356-404376'] = 'TTTGAAACACAAGAAGAAAAT'


from Bio.Seq import Seq
for seed in seeds:
    print(seed)
    print(Seq(seeds[seed]).reverse_complement())


for vcf_fn in vcf_fns:
    print(vcf_fn)
    get_ipython().system('tabix {vcf_fns[vcf_fn]} {exon4_coordinates} | cut -f 1-7')


for vcf_fn in vcf_fns:
    print(vcf_fn)
    get_ipython().system('tabix {vcf_fns[vcf_fn]} {exon4_coordinates} | grep -v MinAlt | cut -f 1-7')


for seed in seeds:
    for bam_fn in bam_fns:
        print('\n\n', seed, bam_fn)
        seed_sequence = "'%s|%s'" % (
            seeds[seed],
            Seq(seeds[seed]).reverse_complement()
        )
        get_ipython().system('samtools view {bam_fns[bam_fn]} | grep -E {seed_sequence} | cut -f 1-9')
    


for seed in seeds:
    for bam_fn in bam_fns:
        print('\n\n', seed, bam_fn)
        seed_sequence = "'%s|%s'" % (
            seeds[seed],
            Seq(seeds[seed]).reverse_complement()
        )
        get_ipython().system('samtools view -f 4 {bam_fns[bam_fn]} | grep -E {seed_sequence} | cut -f 1-9')
    


# # Conclusion
# It seems that we really do only see reads with seed sequences where the mate maps to the right of the seed. There must be some bias against fragments where the mate would expect to map to the left, or at the very least, either the mate or even both reads don't map. Presumably something to do with the very high AT content (100% AT for 130+ bases) in the intron to the left.
# 
# I didn't go into looking at Thomas assemblies, as don't think sequence variation in the intron is relevant after all.
# 




# #Plan
# - Create function to create biallelic, 5/2 rule, new AF, segregating, minimal, renormalised VCF
# - Split the above into SNPs and INDELs
# - Test function on small subset of chr14
# - Run function on chrom 14
# - New function to also create npy file
# - Read in chr14 npy file, and calculate Mendelian error and genotype concordance
# - Attempt to reannotate above with STR and SNPEFF annotations
# - Rerun scripts to get breakdown by SNP/STR/nonSTR coding/noncoding

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
chrom_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160607_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')
chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3.analysis.vcf.gz" % output_dir

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'


vfp_tool_configs = collections.OrderedDict()
vfp_tool_configs['5_2'] = '/nfs/team112/software/htslib/vfp/just_call.config'
vfp_tool_configs['6_3'] = '%s/call_3het.config' % output_dir


get_ipython().run_cell_magic('writefile', "{vfp_tool_configs['6_3']}", 'testing=0\nfilters=gtcall\ninput_is_vcf=1\noutput_is_vcf=1\ncallMinCov=6\ncallMinAlleleCov=3\n')


chrom_vcf_fn


tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


all_samples = ','.join(tbl_release5_crosses_metadata.values('sample'))


all_samples


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 1).values('sample').list()


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 0).values('sample').list()


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


np.in1d(replicates_first, tbl_release5_crosses_metadata.values('sample').array())


rep_index_first = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_first)
rep_index_second = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_second)
print(np.sum(rep_index_first))
print(np.sum(rep_index_second))


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.duplicates('clone').displayall()


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


(tbl_release5_crosses_metadata
 .selecteq('study_title', '3D7xHB3 cross progeny')
 .selecteq('parent_or_progeny', 'parent')
 .values('sample')
 .array()
)


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'MLEAF': 1,
            'MLEAC': 1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        verbose=False,
        cache=True,
        cachedir=output_dir
    )


def create_analysis_vcf(input_vcf_fn=chrom_vcf_fn, region='Pf3D7_14_v3:1000000-1100000', vfp_tool_config='5_2',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        output_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['biallelic', 'regenotyped', 'new_af', 'nonref', 'pass', 'minimal', 'analysis', 'SNP', 'INDEL',
                             'SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                             'gatk_new_af', 'gatk_nonref', 'gatk_pass', 'gatk_minimal', 'gatk_analysis', 'gatk_SNP', 'gatk_INDEL',
                             'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

#     if not os.path.exists(subset_vcf_fn):
#         !{BCFTOOLS} view -Oz -o {subset_vcf_fn} -s {validation_samples} {chrom_vcf_fn}
#         !{BCFTOOLS} index --tbi {subset_vcf_fn}

    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        if region is not None:
            get_ipython().system("{BCFTOOLS} annotate --regions {region} --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        else:
            get_ipython().system("{BCFTOOLS} annotate --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['regenotyped']):
        get_ipython().system("/nfs/team112/software/htslib/vfp/vfp_tool {intermediate_fns['biallelic']} {vfp_tool_configs[vfp_tool_config]} |         bgzip -c > {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['regenotyped']}")

    if rewrite or not os.path.exists(intermediate_fns['new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['new_af']} {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']} {intermediate_fns[\'new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['pass']} {intermediate_fns['nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['pass']}")

    if rewrite or not os.path.exists(intermediate_fns['minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['minimal']} {intermediate_fns['pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['analysis']} {intermediate_fns['minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'SNP\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'SNP_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MIXED\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'INDEL\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_MULTIALLELIC']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['gatk_new_af']} {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'gatk_nonref\']} {intermediate_fns[\'gatk_new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['gatk_pass']} {intermediate_fns['gatk_nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_pass']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['gatk_minimal']} {intermediate_fns['gatk_pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['gatk_analysis']} {intermediate_fns['gatk_minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_SNP\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MIXED\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_INDEL\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_MULTIALLELIC']}")
        
    for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                        'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        if rewrite or not os.path.exists("%s/.vcfnp_cache/variants.npy" % intermediate_fns[variant_type]):
            create_variants_npy(intermediate_fns[variant_type])
        if rewrite or not os.path.exists("%s/.vcfnp_cache/calldata_2d.npy" % intermediate_fns[variant_type]):
            create_calldata_npy(intermediate_fns[variant_type])
        


# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
create_analysis_vcf()


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        print(region, vfp_tool_config)
        create_analysis_vcf(region=region, vfp_tool_config=vfp_tool_config)


chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3_5_2.analysis.vcf.gz" % output_dir
variants_SNP_BIALLELIC = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))


calldata_SNP_BIALLELIC['GT'].shape[1]


calldata_SNP_BIALLELIC['GT'][:, 1] == b'0'


np.unique(variants_SNP_BIALLELIC['SNPEFF_EFFECT'], return_counts=True)


np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)


4788/(np.sum([  1533, 196926,   4788,  95261]))


hets_per_sample = np.sum(calldata_SNP_BIALLELIC['GT'] == b'0/1', 0)
print(len(hets_per_sample))


hets_per_sample


def genotype_concordance(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0') & (parent2_calls == b'0') & (progeny_calls == b'1')) |
                ((parent1_calls == b'1') & (parent2_calls == b'1') & (progeny_calls == b'0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0') | (parent1_calls == b'1' )) &
                ((parent2_calls == b'0') | (parent2_calls == b'1' )) &
                ((progeny_calls == b'0') | (progeny_calls == b'1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0', b'1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0', b'1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0') | (calldata[:, rep_index_first] == b'1')) &
        ((calldata[:, rep_index_second] == b'0') | (calldata[:, rep_index_second] == b'1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0') & (calldata[:, rep_index_second] == b'1')) |
        ((calldata[:, rep_index_first] == b'1') & (calldata[:, rep_index_second] == b'0'))
    )
    missingness_per_sample = np.sum(calldata == b'.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    


def genotype_concordance_gatk(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0/0') | (calldata[:, rep_index_first] == b'1/1')) &
        ((calldata[:, rep_index_second] == b'0/0') | (calldata[:, rep_index_second] == b'1/1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0/0') & (calldata[:, rep_index_second] == b'1/1')) |
        ((calldata[:, rep_index_first] == b'1/1') & (calldata[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(calldata == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'gatk_SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)


1136/(1899+173626+1136+98719)


chrom_analysis_vcf_fn


calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)


np.sum(np.array([  1899, 173626,   1136,  98719]))


np.sum(np.array([  1802, 184456,   3279,  96427]))


np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)


genotype_concordance()


calldata_SNP_BIALLELIC[:, rep_index_first]


variants_crosses = np.load('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz.vcfnp_cache/variants.npy')


variants_crosses.dtype.names


np.unique(variants_crosses['VariantType'])


del(variants_crosses)
gc.collect()


2+2





# This notebook must be run directly from MacBook after running ~/bin/sanger-tunneling.sh in order to connect
# to Sanger network. I haven't figured out a way to do this from Docker container

get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


get_ipython().system('rsync -avL /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0  malsrv2:/nfs/team112_internal/production/release_build/Pf3K/')


get_ipython().system('rsync -avL {DATA_DIR} malsrv2:{os.path.dirname(DATA_DIR)}')





# Previously the setup pf3kgatk_callable_loci cleaned up output files after running. The output of this was used in sample metadata that was released, so we don't want to simply create a new setup based on this that doesn't remove the files, as we want to change parameters, and this will change released metadata. As such we will do this manually here. In the future, we should probably not do the cleanup so bed files are still available.

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


import stat
from sh import ssh
bsub = sh.Command('bsub')


output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
bam_fofn = "%s/pf3k_sample_bams.txt" % output_dir
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/results')
get_ipython().system('mkdir -p {output_dir}/logs')


get_ipython().system('vrpipe-fileinfo --setup pf3kgatk_mergelanes --step 4 --display tab --metadata sample > {bam_fofn}')


GenomeAnalysisTK="/software/jre1.7.0_25/bin/java -Xmx4G -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.5/GenomeAnalysisTK.jar"


GENOME_FN


tbl_bams = etl.fromtsv(bam_fofn)
print(len(tbl_bams.data()))
tbl_bams


for bam_fn, sample in tbl_bams.data():
    print('.', sep='')
    bed_fn = "%s/results/callable_loci_%s.bed" % (output_dir, sample)
    summary_fn = "%s/results/summary_table_%s.txt" % (output_dir, sample)

    if not os.path.exists(bed_fn):
#     if True:
        script_fn = "%s/scripts/CallableLoci_%s.sh" % (output_dir, sample)
        fo = open(script_fn, 'w')
        print('''%s -T CallableLoci -R %s -I %s -summary %s -o %s
''' % (
                GenomeAnalysisTK,
                GENOME_FN,
                bam_fn,
                summary_fn,
                bed_fn,
            ),
            file = fo
        )
        fo.close()
        st = os.stat(script_fn)
        os.chmod(script_fn, st.st_mode | stat.S_IEXEC)
        bsub(
            '-G', 'malaria-dk',
            '-P', 'malaria-dk',
            '-q', 'normal',
            '-o', '%s/logs/CL_%s.out' % (output_dir, sample),
            '-e', '%s/logs/CL_%s.err' % (output_dir, sample),
            '-J', 'CL_%s' % (sample),
            '-R', "'select[mem>8000] rusage[mem=8000]'",
            '-M', '8000',
            script_fn)


2+2


# # Inroduction
# This is to check numbers sent by Thomas 19/07/2016 16:17

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
five_strain_vcf_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160718_Thomas_5_validation_vcf/SNP_INDEL_WG.for_thomas.vcf.gz'


output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_sanity_check_Thomas_numbers"
get_ipython().system('mkdir -p {output_dir}')


vcf_7G8 = "%s/7G8.vcf.gz" % output_dir
get_ipython().system('{BCFTOOLS} view -s 7G8 {five_strain_vcf_fn} | {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {vcf_7G8}')
get_ipython().system('{BCFTOOLS} index --tbi {vcf_7G8}')


6319+33433


6319+33433+4381


6319+33433+4381-909





# # Introduction
# Zam wanted to know how to get all Pf3k variants >= 5% (see email 02/08/2016 16:29)

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
wg_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.vcf.gz" % (release5_final_files_dir)

output_dir = '/nfs/team112_internal/rp7/data/pf3k/analysis/20160802_5PC_MAF_for_Zam'
get_ipython().system('mkdir -p {output_dir}')
output_fn = "%s/SNP_INDEL_WG.combined.filtered.maf_ge_5pc.vcf.gz" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'


get_ipython().system('{BCFTOOLS} view --include \'MAF[0]>=0.05 & FILTER="PASS" & ALT[1]!="*"\' --output-file {output_fn} --output-type z {wg_vcf_fn}')
get_ipython().system('{BCFTOOLS} index -f --tbi {output_fn}')

# --regions Pf3D7_01_v3:100000-110000 \





2+2





# # Introduction
# This is details of a VR-PIPE setup of full calling pipeline based on mappings done by Thomas of 100bp reads from 16 "multiple reference strain" (MRS) samples. See email from Thomas at 14:13 on 23/05/2016.

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


RELEASE_DIR = "%s/mrs_1" % DATA_DIR
RESOURCES_DIR = '%s/resources' % RELEASE_DIR

# GENOME_FN = "/nfs/pathogen003/tdo/Pfalciparum/3D7/Reference/Oct2011/Pf3D7_v3.fasta" # Note this ref used by Thomas is different to other refs we have used, e.g. chromsomes aren't in numerical order
GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
SNPEFF_DIR = "/lustre/scratch109/malaria/pf3k_methods/resources/snpEff"
REGIONS_FN = "/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz"

RELEASE_METADATA_FN = "%s/pf3k_mrs_1_sample_metadata.txt" % RELEASE_DIR
WG_VCF_FN = "%s/vcf/pf3k_mrs_1.vcf.gz" % RELEASE_DIR

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'


print(WG_VCF_FN)


chromosomes = ["Pf3D7_%02d_v3" % x for x in range(1, 15, 1)] + [
    'Pf3D7_API_v3', 'Pf_M76611'
]
chromosome_vcfs = ["%s/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz" % (RELEASE_DIR, x) for x in chromosomes]


if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


get_ipython().system('cp {GENOME_FN}* {RESOURCES_DIR}')
get_ipython().system('cp -R {SNPEFF_DIR} {RESOURCES_DIR}')
get_ipython().system('cp -R {REGIONS_FN} {RESOURCES_DIR}')


for lustre_dir in ['input', 'output', 'meta']:
    os.makedirs("/lustre/scratch109/malaria/pf3k_mrs_1/%s" % lustre_dir)





get_ipython().run_line_magic('run', 'imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


# ## Create sites only vcfs

for release in CHROM_VCF_FNS.keys():
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for chrom in CHROM_VCF_FNS[release].keys():
        input_vcf_fn = CHROM_VCF_FNS[release][chrom]
        output_vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, chrom)
        if not os.path.exists(output_vcf_fn):
            get_ipython().system('bcftools view --drop-genotypes --output-type z --output-file {output_vcf_fn} {input_vcf_fn}')


# ## Merge sites only vcfs

for release in CHROM_VCF_FNS.keys():
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    input_files = ' '.join(
        ['%s/%s_%s_sites.vcf.gz' % (output_dir, release, chrom) for chrom in CHROM_VCF_FNS[release].keys()]
    )
    output_vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, 'WG')
    if not os.path.exists(output_vcf_fn):
        get_ipython().system('bcftools concat --output-type z --output {output_vcf_fn} {input_files}')
        get_ipython().system('bcftools index --tbi {output_vcf_fn}')


# ## Create npy sites files

for release in WG_VCF_FNS.keys():
    print(release)
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    if release == 'release3':
        vcf_fn = WG_VCF_FNS['release3']
    else:
        vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, 'WG')
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   6,
            'AF':    6,
            'AC':    6,
            'MLEAF': 6,
            'MLEAC': 6,
            'RPA':   7,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )





# #Plan
# - Create function to create biallelic, 5/2 rule, new AF, segregating, minimal, renormalised VCF
# - Split the above into SNPs and INDELs
# - Test function on small subset of chr14
# - Run function on chrom 14
# - New function to also create npy file
# - Read in chr14 npy file, and calculate Mendelian error and genotype concordance
# - Attempt to reannotate above with STR and SNPEFF annotations
# - Rerun scripts to get breakdown by SNP/STR/nonSTR coding/noncoding

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
chrom_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160712_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')
chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3.analysis.vcf.gz" % output_dir

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'


vfp_tool_configs = collections.OrderedDict()
vfp_tool_configs['5_2'] = '/nfs/team112/software/htslib/vfp/just_call.config'
vfp_tool_configs['6_3'] = '%s/call_3het.config' % output_dir


get_ipython().run_cell_magic('writefile', "{vfp_tool_configs['6_3']}", 'testing=0\nfilters=gtcall\ninput_is_vcf=1\noutput_is_vcf=1\ncallMinCov=6\ncallMinAlleleCov=3\n')


chrom_vcf_fn


tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


all_samples = ','.join(tbl_release5_crosses_metadata.values('sample'))


all_samples


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 1).values('sample').list()


tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 0).values('sample').list()


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


np.in1d(replicates_first, tbl_release5_crosses_metadata.values('sample').array())


rep_index_first = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_first)
rep_index_second = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_second)
print(np.sum(rep_index_first))
print(np.sum(rep_index_second))


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.duplicates('clone').displayall()


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


(tbl_release5_crosses_metadata
 .selecteq('study_title', '3D7xHB3 cross progeny')
 .selecteq('parent_or_progeny', 'parent')
 .values('sample')
 .array()
)


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'MLEAF': 1,
            'MLEAC': 1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        verbose=False,
        cache=True,
        cachedir=output_dir
    )


def create_analysis_vcf(input_vcf_fn=chrom_vcf_fn, region='Pf3D7_14_v3:1000000-1100000', vfp_tool_config='5_2',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        output_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['biallelic', 'regenotyped', 'new_af', 'nonref', 'pass', 'minimal', 'analysis', 'SNP', 'INDEL',
                             'SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                             'gatk_new_af', 'gatk_nonref', 'gatk_pass', 'gatk_minimal', 'gatk_analysis', 'gatk_SNP', 'gatk_INDEL',
                             'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

#     if not os.path.exists(subset_vcf_fn):
#         !{BCFTOOLS} view -Oz -o {subset_vcf_fn} -s {validation_samples} {chrom_vcf_fn}
#         !{BCFTOOLS} index --tbi {subset_vcf_fn}

    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        if region is not None:
            get_ipython().system("{BCFTOOLS} annotate --regions {region} --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        else:
            get_ipython().system("{BCFTOOLS} annotate --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['regenotyped']):
        get_ipython().system("/nfs/team112/software/htslib/vfp/vfp_tool {intermediate_fns['biallelic']} {vfp_tool_configs[vfp_tool_config]} |         bgzip -c > {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['regenotyped']}")

    if rewrite or not os.path.exists(intermediate_fns['new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['new_af']} {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']} {intermediate_fns[\'new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['pass']} {intermediate_fns['nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['pass']}")

    if rewrite or not os.path.exists(intermediate_fns['minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['minimal']} {intermediate_fns['pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['analysis']} {intermediate_fns['minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'SNP\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'SNP_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MIXED\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'INDEL\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_MULTIALLELIC']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['gatk_new_af']} {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'gatk_nonref\']} {intermediate_fns[\'gatk_new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['gatk_pass']} {intermediate_fns['gatk_nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_pass']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['gatk_minimal']} {intermediate_fns['gatk_pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['gatk_analysis']} {intermediate_fns['gatk_minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_SNP\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MIXED\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_INDEL\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_MULTIALLELIC']}")
        
    for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                        'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        if rewrite or not os.path.exists("%s/.vcfnp_cache/variants.npy" % intermediate_fns[variant_type]):
            create_variants_npy(intermediate_fns[variant_type])
        if rewrite or not os.path.exists("%s/.vcfnp_cache/calldata_2d.npy" % intermediate_fns[variant_type]):
            create_calldata_npy(intermediate_fns[variant_type])
        


# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
create_analysis_vcf()


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        print(region, vfp_tool_config)
        create_analysis_vcf(region=region, vfp_tool_config=vfp_tool_config)


chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3_5_2.analysis.vcf.gz" % output_dir
variants_SNP_BIALLELIC = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))


calldata_SNP_BIALLELIC['GT'].shape[1]


calldata_SNP_BIALLELIC['GT'][:, 1] == b'0'


np.unique(variants_SNP_BIALLELIC['SNPEFF_EFFECT'], return_counts=True)


np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)


4788/(np.sum([  1533, 196926,   4788,  95261]))


hets_per_sample = np.sum(calldata_SNP_BIALLELIC['GT'] == b'0/1', 0)
print(len(hets_per_sample))


hets_per_sample


def genotype_concordance(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0') & (parent2_calls == b'0') & (progeny_calls == b'1')) |
                ((parent1_calls == b'1') & (parent2_calls == b'1') & (progeny_calls == b'0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0') | (parent1_calls == b'1' )) &
                ((parent2_calls == b'0') | (parent2_calls == b'1' )) &
                ((progeny_calls == b'0') | (progeny_calls == b'1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0', b'1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0', b'1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0') | (calldata[:, rep_index_first] == b'1')) &
        ((calldata[:, rep_index_second] == b'0') | (calldata[:, rep_index_second] == b'1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0') & (calldata[:, rep_index_second] == b'1')) |
        ((calldata[:, rep_index_first] == b'1') & (calldata[:, rep_index_second] == b'0'))
    )
    missingness_per_sample = np.sum(calldata == b'.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    


def genotype_concordance_gatk(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0/0') | (calldata[:, rep_index_first] == b'1/1')) &
        ((calldata[:, rep_index_second] == b'0/0') | (calldata[:, rep_index_second] == b'1/1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0/0') & (calldata[:, rep_index_second] == b'1/1')) |
        ((calldata[:, rep_index_first] == b'1/1') & (calldata[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(calldata == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)


calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'gatk_SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)


1136/(1899+173626+1136+98719)


chrom_analysis_vcf_fn


calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)


np.sum(np.array([  1899, 173626,   1136,  98719]))


np.sum(np.array([  1802, 184456,   3279,  96427]))


np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)


genotype_concordance()


calldata_SNP_BIALLELIC[:, rep_index_first]


variants_crosses = np.load('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz.vcfnp_cache/variants.npy')


variants_crosses.dtype.names


np.unique(variants_crosses['VariantType'])


del(variants_crosses)
gc.collect()


2+2





# #Plan
# - Similar to 20160720_mendelian_error_duplicate_concordance but
# - Split multiallelics in biallelic+spanning deletion and "true" multiallelic

# See 20160203_release5_npy_hdf5.ipynb for creation of VCF specific to crosses

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
crosses_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)
sites_only_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.sites.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160720_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'
gff_fn = "%s/Pfalciparum.noseq.gff3.gz" % output_dir
cds_gff_fn = "%s/Pfalciparum.noseq.gff3.cds.gz" % output_dir

results_table_fn = "%s/genotype_quality.xlsx" % output_dir
counts_table_fn = "%s/variant_counts.xlsx" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
GATK = '/software/jre1.7.0_25/bin/java -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.4-46/GenomeAnalysisTK.jar'


gff_fn


get_ipython().system('wget ftp://ftp.sanger.ac.uk/pub/project/pathogens/gff3/2016-06/Pfalciparum.noseq.gff3.gz     -O {gff_fn}')


get_ipython().system("zgrep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")


crosses_vcf_fn


multiallelic_header_fn = "%s/vcf/MULTIALLELIC.hdr" % (output_dir)
fo=open(multiallelic_header_fn, 'w')
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()


variant_type_header_fn = "%s/vcf/VARIANT_TYPE.hdr" % (output_dir)
fo=open(variant_type_header_fn, 'w')
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (IND)">', file=fo)
fo.close()


cds_header_fn = "%s/vcf/CDS.hdr" % (output_dir)
fo=open(cds_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
fo.close()


def create_analysis_vcf(input_vcf_fn=crosses_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['nonref', 'multiallelic', 'triallelic', 'bi_allelic', 'spanning_deletion', 'triallelic_no_sd', 'multiallelics',
                              'biallelic', 'str', 'snps', 'indels', 'strs', 'variant_type', 'coding', 'analysis',
                             'site_snps', 'site_indels', 'site_variant_type', 'site_analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)
    
    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        if region is not None:
            get_ipython().system('{BCFTOOLS} annotate --regions {region} --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        else:
            get_ipython().system('{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {intermediate_fns['nonref']} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

#     if rewrite or not os.path.exists(intermediate_fns['str']):
#         !{GATK} -T VariantAnnotator \
#             -R {GENOME_FN} \
#             -o {intermediate_fns['str']} \
#             -A TandemRepeatAnnotator  \
#             -V {intermediate_fns['biallelic']}
# #         !{BCFTOOLS} index --tbi {intermediate_fns['str']}

    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

#     if rewrite or not os.path.exists(intermediate_fns['strs']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tSTR\n' --include 'TYPE!="snp" && STR=1' {intermediate_fns['str']} | bgzip -c > {intermediate_fns['strs']} && tabix -s1 -b2 -e2 -f {intermediate_fns['strs']}

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['biallelic']} |        {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

        #         {BCFTOOLS} annotate -a {intermediate_fns['strs']} -c CHROM,POS,INFO/VARIANT_TYPE \

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")

#     if rewrite or not os.path.exists(intermediate_fns['site_snps']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tSNP\n' --include 'TYPE="snp"' {intermediate_fns['multiallelics']} | bgzip -c > {intermediate_fns['site_snps']} && tabix -s1 -b2 -e2 -f {intermediate_fns['site_snps']}

#     if rewrite or not os.path.exists(intermediate_fns['site_indels']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tINDEL\n' --include 'TYPE!="snp"' {intermediate_fns['multiallelics']} | bgzip -c > {intermediate_fns['site_indels']} && tabix -s1 -b2 -e2 -f {intermediate_fns['site_indels']}

#     if rewrite or not os.path.exists(intermediate_fns['site_variant_type']):
#         !{BCFTOOLS} annotate -a {intermediate_fns['site_snps']} -c CHROM,POS,INFO/VARIANT_TYPE \
#         -h {variant_type_header_fn} {intermediate_fns['multiallelics']} | \
#         {BCFTOOLS} annotate -a {intermediate_fns['site_indels']} -c CHROM,POS,INFO/VARIANT_TYPE \
#         -Oz -o {intermediate_fns['site_variant_type']} 
#         !{BCFTOOLS} index --tbi {intermediate_fns['site_variant_type']}

#     if rewrite or not os.path.exists(intermediate_fns['site_analysis']):
#         !{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS \
#         -h {cds_header_fn} \
#         -Oz -o {intermediate_fns['site_analysis']} {intermediate_fns['site_variant_type']}
#         !{BCFTOOLS} index --tbi {intermediate_fns['site_analysis']}


# create_analysis_vcf()


create_analysis_vcf(region=None)


output_dir


tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata


replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']


quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']


# Note the version created in this notebook doesn't work. I think this is because of R in Number of FORMAT field for AD,
# which is part of spec for v4.2, but think GATK must have got rid of this in previous notebook
analysis_vcf_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_mendelian_error_duplicate_concordance/vcf/SNP_INDEL_WG.analysis.vcf.gz"
vcf_reader = vcf.Reader(filename=analysis_vcf_fn)
sample_ids = np.array(vcf_reader.samples)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)


# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)


tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()


def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE', 'RU',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'RPA', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
        fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
                'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
                'VQSLOD', 'FILTER'],
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'VARIANT_TYPE':             'a3',
            'RU':                       'a40',
            'SNPEFF_EFFECT':            'a33',
            'CDS':                      bool,
            'MULTIALLELIC':             'a2',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
        },
        flatten_filter=True,
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD', 'DP', 'GQ'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )


# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
# create_variants_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
# create_calldata_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
create_variants_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))
create_calldata_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))


# def create_variants_npy_2(vcf_fn):
#     output_dir = '%s.vcfnp_cache' % vcf_fn
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     vcfnp.variants(
#         vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
#         dtypes={
#             'REF':                      'a10',
#             'ALT':                      'a10',
#             'RegionType':               'a25',
#             'VariantType':              'a40',
#             'VARIANT_TYPE':             'a3',
#             'SNPEFF_EFFECT':            'a33',
#             'CDS':                      bool,
#             'MULTIALLELIC':             'a2',
#         },
#         arities={
#             'ALT':   1,
#             'AF':    1,
#             'AC':    1,
#             'ANN':   1,
#         },
#         fills={
#             'VQSLOD': np.nan,
#         },
#         flatten_filter=True,
#         progress=100000,
#         verbose=True,
#         cache=True,
#         cachedir=output_dir
#     )


# create_variants_npy_2("%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir))
# create_calldata_npy("%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir))


analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
variants = np.load("%s.vcfnp_cache/variants.npy" % analysis_vcf_fn)
calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % analysis_vcf_fn)


print(np.unique(variants['VARIANT_TYPE'], return_counts=True))


# site_analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir)
# site_variants = np.load("%s.vcfnp_cache/variants.npy" % site_analysis_vcf_fn)


# print(np.unique(site_variants['VARIANT_TYPE'], return_counts=True))
# print(np.unique(site_variants['MULTIALLELIC'], return_counts=True))
# print(np.unique(site_variants['CDS'], return_counts=True))
# print(np.unique(site_variants['FILTER_PASS'], return_counts=True))


np.unique(variants['MULTIALLELIC'], return_counts=True)


def genotype_concordance_gatk(calldata=calldata,
                              ix = ((variants['VARIANT_TYPE'] == b'SNP') & (variants['MULTIALLELIC'] == b'BI') &
                                    (variants['CDS']) & variants['FILTER_PASS']),
                              GQ_threshold=30,
                              rep_index_first=rep_index_first, rep_index_second=rep_index_second,
                              verbose=False):
    GT = calldata['GT'][ix, :]
    GT[calldata['GQ'][ix, :] < GQ_threshold] = b'./.'
    
    all_samples = sample_ids
#     all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = GT[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = GT[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = GT[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    GT_both = (np.in1d(GT[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(GT[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    GT_both = (
        ((GT[:, rep_index_first] == b'0/0') | (GT[:, rep_index_first] == b'1/1')) &
        ((GT[:, rep_index_second] == b'0/0') | (GT[:, rep_index_second] == b'1/1'))
    )
    GT_discordant = (
        ((GT[:, rep_index_first] == b'0/0') & (GT[:, rep_index_second] == b'1/1')) |
        ((GT[:, rep_index_first] == b'1/1') & (GT[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(GT == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (GT.shape[0] * GT.shape[1])
    heterozygosity_per_sample = np.sum(GT == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (GT.shape[0] * GT.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(GT_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(GT_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(GT_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    num_of_alleles = np.sum(ix)
    return(
#         num_of_alleles,
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
#         prop_discordances_per_sample_pair,
#         GT.shape
    )
    


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


genotype_concordance_gatk()


results_list = list()
# GQ_thresholds = [30, 99, 0]
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

for GQ_threshold in GQ_thresholds:
    for filter_pass in filter_passes:
        for variant_type in variant_types:
            for coding in codings:
                for multiallelic in multiallelics:
                    print(GQ_threshold, filter_pass, variant_type, coding, multiallelic)
                    ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
                    number_of_alleles = np.sum(ix)
                    number_of_sites = len(np.unique(variants[['CHROM', 'POS']][ix]))
                    mean_nraf = np.sum(variants['AC'][ix]) / np.sum(variants['AN'][ix])
                    genotype_quality_results = list(genotype_concordance_gatk(ix=ix, GQ_threshold=GQ_threshold))
#                     sites_ix = (
#                         (site_variants['VARIANT_TYPE'] == variant_type) &
#                         (site_variants['MULTIALLELIC'] == multiallelic) &
#                         (site_variants['CDS'] == coding) &
#                         (site_variants['FILTER_PASS'] == filter_pass)
#                     )
#                     num_sites = np.sum(sites_ix)
                    results_list.append(
                        [GQ_threshold, filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf] +
                        genotype_quality_results
                    )

# print(results_list)


# Sanity check. Previously this was showing 2 variants, which was due to a bug
variant_type = b'SNP'
multiallelic = b'BI'
coding = False
filter_pass = True
ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
temp = variants[['CHROM', 'POS']][ix]
s = np.sort(temp, axis=None)
s[s[1:] == s[:-1]]


# Sanity check. Previously this was 1 of the two variants shown above and multiallelic was b'BI' not b'MU'
variants[(variants['CHROM']==b'Pf3D7_01_v3') & (variants['POS']==514753)]


headers = ['GQ threshold', 'PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF', 'Missingness',
           'Heterozygosity', 'Discordance', 'MER']
etl.wrap(results_list).pushheader(headers).displayall()


np.sum(etl.wrap(results_list).pushheader(headers).values('Variants').array())





# etl.wrap(results_list).pushheader(headers).convert('Alleles', int).toxlsx(results_table_fn)
etl.wrap(results_list).pushheader(headers).cutout('Alleles').cutout('Mean NRAF').toxlsx(results_table_fn)
results_table_fn








# variant_type_header_2_fn = "%s/vcf/VARIANT_TYPE_2.hdr" % (output_dir)
# fo=open(variant_type_header_2_fn, 'w')
# print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or INDEL">', file=fo)
# fo.close()


def create_variant_counts_vcf(input_vcf_fn=sites_only_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.sites.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['multiallelic', 'multiallelics', 'snps', 'indels', 'triallelic', 'bi_allelic', 'spanning_deletion',
                              'triallelic_no_sd', 'biallelic',
                              'variant_type', 'analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {input_vcf_fn} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {input_vcf_fn} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {input_vcf_fn} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set        -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {input_vcf_fn} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
                
#     if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
#         !{BCFTOOLS} annotate -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC \
#         -h {multiallelic_header_fn} {intermediate_fns['nonref']} | \
#         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |\
#         {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |\
#         {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC \
#         -Oz -o {intermediate_fns['multiallelics']}
#         !{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

#         {BCFTOOLS} view --include 'AC>0 && ALT!="*"' -Oz -o {intermediate_fns['biallelic']}
        
    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['biallelic']} |         {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")


sites_only_vcf_fn


# create_variant_counts_vcf()


create_variant_counts_vcf(region=None)


# def create_variants_npy_2(vcf_fn):
#     output_dir = '%s.vcfnp_cache' % vcf_fn
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     vcfnp.variants(
#         vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
#         dtypes={
#             'REF':                      'a10',
#             'ALT':                      'a10',
#             'RegionType':               'a25',
#             'VariantType':              'a40',
#             'VARIANT_TYPE':             'a3',
#             'SNPEFF_EFFECT':            'a33',
#             'CDS':                      bool,
#             'MULTIALLELIC':             bool,
#         },
#         arities={
#             'ALT':   1,
#             'AF':    1,
#             'AC':    1,
#             'ANN':   1,
#         },
#         fills={
#             'VQSLOD': np.nan,
#         },
#         flatten_filter=True,
#         progress=100000,
#         verbose=True,
#         cache=True,
#         cachedir=output_dir
#     )


create_variants_npy("%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir))


sites_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
variants_all = np.load("%s.vcfnp_cache/variants.npy" % sites_vcf_fn)


counts_list = list()
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

# GQ_thresholds = [30, 99, 0]
# variant_types = [b'SNP', b'IND']
# multiallelics = [False, True]
# codings = [True, False]
# filter_passes = [True, False]


for filter_pass in filter_passes:
    for variant_type in variant_types:
        for coding in codings:
            for multiallelic in multiallelics:
                print(filter_pass, variant_type, coding, multiallelic)
                ix = (
                    (variants_all['VARIANT_TYPE'] == variant_type) &
                    (variants_all['MULTIALLELIC'] == multiallelic) &
                    (variants_all['CDS'] == coding) &
                    (variants_all['FILTER_PASS'] == filter_pass)
                )
                number_of_alleles = np.sum(ix)
                number_of_sites = len(np.unique(variants_all[['CHROM', 'POS']][ix]))
                mean_nraf = np.sum(variants_all['AC'][ix]) / np.sum(variants_all['AN'][ix])
#                 number_of_variants = np.sum(ix)
                counts_list.append(
                    [filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf]
                )

print(counts_list)


headers = ['PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF']
(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .displayall()
)


(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .toxlsx(counts_table_fn)
)
# etl.wrap(counts_list).pushheader(headers).convertnumbers().toxlsx(counts_table_fn)
counts_table_fn


np.sum(etl
       .fromtsv('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/passcounts')
       .pushheader(('CHROM', 'count'))
       .convertnumbers()
       .values('count')
       .array()
       )


2+2





# # Plan
# - dict of boolean array of core genome
# - for each sample determine % core callable
# - plot histogram of this
# - dict of number of samples callable at each position
# - for each well-covered samples, add to number of callable samples
# - plot this genome-wide

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


# see 20160525_CallableLoci_bed_release_5.ipynb
lustre_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
callable_loci_bed_fn_format = "%s/results/callable_loci_%%s.bed" % lustre_dir

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160718_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

core_regions_fn = "%s/core_regions_20130225.bed" % lustre_dir

callable_loci_fn = "%s/callable_loci_high_coverage_samples.bed" % plot_dir
callable_loci_merged_fn = "%s/callable_loci_merged_samples.bed" % plot_dir

multiIntersectBed = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/multiIntersectBed'
bedtools = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools'

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'


core_genome_dict = collections.OrderedDict()
for chrom in ['Pf3D7_%02d_v3' % i for i in range(1, 15)]:
    this_chrom_regions = (etl
                          .fromtabix(REGIONS_FN, chrom)
                          .pushheader('chrom', 'start', 'end', 'region')
                          .convertnumbers()
                          )
    chrom_length = np.max(this_chrom_regions.convert('end', int).values('end').array())
    core_genome_dict[chrom] = np.zeros(chrom_length, dtype=bool)
    for rec in this_chrom_regions:
        if rec[3] == 'Core':
            core_genome_dict[chrom][rec[1]:rec[2]] = True


tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)


tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])


len(tbl_field_samples.data())


def count_symbol(i=1):
    if i%10 == 0:
        return(str((i//10)*10))
    else:
        return('.')


# ox_codes = tbl_field_samples_extended.selectge('core_bases_callable', core_genome_length*0.95).values('sample').array(dtype='U12')
# len(ox_codes)


ox_codes = tbl_field_samples.values('sample').array(dtype='U12')
len(ox_codes)


ox_codes.dtype


callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged.bed'
for i, ox_code in enumerate(ox_codes):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')


get_ipython().system("sort -T /lustre/scratch111/malaria/rp7/temp -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")


# !/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov \
# -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} \
# -g {GENOME_FN+'.fai'} -bga \
# > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.bed


get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt")


# !bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed
get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt')

# !tabix -f -p bed /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed.gz
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt.gz')


merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)


print(len(accessibility_array))
print(accessibility_array[0])
accessibility_array


accessibility_array_fn = "%s/accessibility_array_2512_field.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array)


del(accessibility_array)
gc.collect()


accessibility_array_fn = "%s/accessibility_array_2512_field.npy" % plot_dir
accessibility_array = np.load(accessibility_array_fn)


accessibility_colors = {
    'Core': 'white',
    'SubtelomericHypervariable': 'red',
    'InternalHypervariable': 'orange',
    'SubtelomericRepeat': 'brown',
    'Centromere': 'black'
#     'InternalHypervariable': '#b20000',
}


def plot_accessibility(accessibility=accessibility_array, callset='2512_field', bin_size=1000, number_of_samples = 2512):

    fig = plt.figure(figsize=(11.69*1, 8.27*1))
    gs = GridSpec(2*14, 1, height_ratios=([1.0, 1.0])*14)
    gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

    print('\n', bin_size)
    for i in range(14):
        print(i+1, end=" ")
        chrom = 'Pf3D7_%02d_v3' % (i + 1)
        pos = accessibility[accessibility['chrom']==chrom.encode('ascii')]['pos']
        coverage = accessibility[accessibility['chrom']==chrom.encode('ascii')]['coverage']
        max_pos = np.max(pos)
        if bin_size == 1:
            binned_coverage, bin_centres = coverage, pos
        else:
            binned_coverage, bins, _ = scipy.stats.binned_statistic(pos, coverage, bins=np.arange(1, max_pos, bin_size))
            bin_centres = (bins[:-1]+bins[1:]) / 2
        ax = fig.add_subplot(gs[i*2])
        ax.plot(bin_centres, binned_coverage/number_of_samples)
    #     ax.plot(pos, coverage/number_of_samples)
        ax.set_xlim(0, 3300000)
        ax.set_xticks(range(0, len(core_genome_dict[chrom]), 100000))
        ax.set_xticklabels(np.arange(0, len(core_genome_dict[chrom])/1e+6, 0.1))
        tbl_regions = (etl
            .fromtabix(REGIONS_FN, chrom)
            .pushheader('chrom', 'start', 'end', 'region')
            .convertnumbers()
        )
        for region_chrom, start_pos, end_pos, region_type in tbl_regions.data():
            if region_type != 'Core':
                ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
        for s in 'left', 'right', 'top':
            ax.spines[s].set_visible(False)
    #         ax.set_yticklabels([])
        ax.get_xaxis().tick_bottom()
        ax.set_yticks([])

        ax.set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')

        ax.set_xlabel('')
        if i < 13:
            ax.set_xticklabels([])
    #     ax.spines['top'].set_bounds(0, len(core_genome_dict[chrom]))    
        ax.spines['bottom'].set_bounds(0, len(core_genome_dict[chrom]))
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%s_%dbp_windows.png' % (callset, bin_size)), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%s_%dbp_windows.pdf' % (callset, bin_size)))

plot_accessibility()





ox_codes_5 = ['7G8', 'GB4', 'ERS740940', 'ERS740937', 'ERS740936']
callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged.bed'
for i, ox_code in enumerate(ox_codes_5):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')


get_ipython().system("sort -T /lustre/scratch111/malaria/rp7/temp -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")


get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt")


get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt')
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt.gz')


merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt.gz"

accessibility_array_5_validation = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)


print(len(accessibility_array_5_validation))
print(accessibility_array_5_validation[0])
accessibility_array_5_validation


accessibility_array_fn = "%s/accessibility_array_5_validation.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array_5_validation)


plot_accessibility(accessibility=accessibility_array_5_validation, callset='5_validation')


plot_dir





# # Introduction
# As part of cleanup of lustre, I decided we should copy various pf3k_pacbio_2 outputs used or created by vrpipe.
# 
# Setups used were: 843-847, 849-854
# 
# I decided it might be good to keep inputs, resources, outputs of HaplotypeCaller and final vcfs
# 

archive_dir = '/nfs/team112_internal/rp7/data/pf3k/pacbio_2'
for chrom in ['Pf3D7_%02d_v3' % n for n in range(1, 15)] + ['Pf3D7_API_v3', 'Pf_M76611']:
    get_ipython().system('mkdir -p {"%s/vcf/vcf_symlinks/%s" % (archive_dir, chrom)}')


get_ipython().system('cp -R /lustre/scratch109/malaria/pf3k_pacbio/input {archive_dir}/')


pf3k_pacbio_2_haplotype_caller


# # Not run!
# This never got run as it was decided not to use the interim build based on Thomas's defintion of the core genome

get_ipython().run_line_magic('run', '_standard_imports.ipynb')


output_dir = '/lustre/scratch118/malaria/team112/personal/rp7/data/methods-dev/pf3k_techbm/20170216_Pf3k_60_HDF5_build'
vcf_stem = '/nfs/team112_internal/production/release_build/Pf3K/pilot_6_0/SNP_INDEL_{chrom}.combined.filtered.vcf.gz'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_6_0'
nfs_final_hdf5_dir = '%s/hdf5' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_hdf5_dir}')

GENOME_FN = "/lustre/scratch118/malaria/team112/pipelines/resources/pf3k_methods/resources/Pfalciparum.genome.fasta"
genome_fn = "%s/Pfalciparum.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/hdf5')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/npy')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

get_ipython().system('cp {GENOME_FN} {genome_fn}')


genome = pyfasta.Fasta(genome_fn)
genome


fo = open("%s/scripts/vcfnp_variants.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype NewRegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_calldata.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type calldata_2d \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity AD:7 \
    --arity PL:28 \
    --dtype PGT:a3 \
    --dtype PID:a12 \
    --exclude-field MIN_DP \
    --exclude-field RGQ \
    --exclude-field SB''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
inputs=$3
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''', file=fo)
fo.close()


task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

    task = "%s/scripts/vcfnp_calldata.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')


task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "hdf" -n8 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'Pf3D7_01_v3\')} {output_dir}/hdf5/Pf3K_pilot_6_0 {output_dir}/npy')


get_ipython().system('cp {output_dir}/hdf5/* {nfs_final_hdf5_dir}/')





# # Sanity checks

y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT.h5' % output_dir, 'r')


(etl.wrap(
    np.unique(y['variants']['SNPEFF_EFFECT'], return_counts=True)
)
    .transpose()
    .pushheader('SNPEFF_EFFECT', 'number')
    .sort('number', reverse=True)
    .displayall()
)


task_size = 20000
for chrom in ['PvP01_00'] + sorted(genome.keys()):
    if chrom.startswith('Pv'):
        vcf_fn = vcf_stem.format(chrom=chrom)
        if chrom == 'PvP01_00':
            chrom_length = transfer_length
        else:
            chrom_length = len(genome[chrom])
        n_tasks = '1-%s' % ((chrom_length // task_size) + 1)
        print(chrom, n_tasks)

        task = "%s/scripts/vcfnp_variants.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

        task = "%s/scripts/vcfnp_calldata.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')


get_ipython().system('cp /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/hdf5/ ')





(etl.wrap(
    np.unique(y['variants']['CDS'], return_counts=True)
)
    .transpose()
    .pushheader('CDS', 'number')
    .sort('number', reverse=True)
    .displayall()
)


CDS = y['variants']['CDS'][:]
SNPEFF_EFFECT = y['variants']['SNPEFF_EFFECT'][:]
SNP = (y['variants']['VARIANT_TYPE'][:] == b'SNP')
INDEL = (y['variants']['VARIANT_TYPE'][:] == b'INDEL')


np.unique(CDS[SNP], return_counts=True)


2+2


y['variants']['VARIANT_TYPE']


pd.value_counts(INDEL)


pd.crosstab(SNPEFF_EFFECT[SNP], CDS[SNP])


2+2


df = pd.DataFrame({'CDS': CDS, 'SNPEFF_EFFECT':SNPEFF_EFFECT})


writer = pd.ExcelWriter("/nfs/users/nfs_r/rp7/SNPEFF_for_Rob.xlsx")
pd.crosstab(SNPEFF_EFFECT, CDS).to_excel(writer)
writer.save()





pd.crosstab(SNPEFF_EFFECT, y['variants']['CHROM'])


np.unique(y['variants']['svlen'], return_counts=True)


y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT_10pc.h5.h5' % output_dir, 'r')
y


# for field in y['variants'].keys():
for field in ['svlen']:
    print(field, np.unique(y['variants'][field], return_counts=True))














get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_no_PID_PGT_10pc     --output {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5 > {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.md5 ')








get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset_1pc     --output {output_dir}/hdf5/Pf_60_subset_1pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_1pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_1pc.h5 > {output_dir}/hdf5/Pf_60_subset_1pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')


get_ipython().system('{output_dir}/scripts/vcfnp_concat.sh {vcf_fn} {output_dir}/hdf5/Pf_60')


fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
# inputs=${vcf}.vcfnp_cache
inputs=%s/npy
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''' % (
        output_dir,
        )
      , file=fo)
fo.close()

#     nv=$(ls -1 ${inputs}/v* | wc -l)
#     nc=$(ls -1 ${inputs}/c* | wc -l)
#     echo variants files $nv >> $log
#     echo calldata files $nc >> $log
#     if [ "$nv" -ne "$nc" ]
#     then
#         echo missing npy files
#         exit 1
#     fi


# # Copy files to /nfs

get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5 {nfs_final_hdf5_dir}/')
get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5.md5 {nfs_final_hdf5_dir}/')





# This notebook must be run directly from MacBook after running ~/bin/sanger-tunneling.sh in order to connect
# to Sanger network. I haven't figured out a way to do this from Docker container

get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')


get_ipython().system('rsync -avL {DATA_DIR} malsrv2:{os.path.dirname(DATA_DIR)}')


# # Plan
# - dict of boolean array of core genome
# - for each sample determine % core callable
# - plot histogram of this
# - dict of number of samples callable at each position
# - for each well-covered samples, add to number of callable samples
# - plot this genome-wide

get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')


# see 20160525_CallableLoci_bed_release_5.ipynb
lustre_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
callable_loci_bed_fn_format = "%s/results/callable_loci_%%s.bed" % lustre_dir

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160713_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

core_regions_fn = "%s/core_regions_20130225.bed" % lustre_dir

callable_loci_fn = "%s/callable_loci_high_coverage_samples.bed" % plot_dir
callable_loci_merged_fn = "%s/callable_loci_merged_samples.bed" % plot_dir

multiIntersectBed = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/multiIntersectBed'
bedtools = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools'

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'


core_regions_fn


REGIONS_FN


get_ipython().system('zgrep Core {REGIONS_FN} > {core_regions_fn}')


core_genome_dict = collections.OrderedDict()
for chrom in ['Pf3D7_%02d_v3' % i for i in range(1, 15)]:
    this_chrom_regions = (etl
                          .fromtabix(REGIONS_FN, chrom)
                          .pushheader('chrom', 'start', 'end', 'region')
                          .convertnumbers()
                          )
    chrom_length = np.max(this_chrom_regions.convert('end', int).values('end').array())
    core_genome_dict[chrom] = np.zeros(chrom_length, dtype=bool)
    for rec in this_chrom_regions:
        if rec[3] == 'Core':
            core_genome_dict[chrom][rec[1]:rec[2]] = True


core_genome_length = 0
for chrom in core_genome_dict:
    print(chrom, len(core_genome_dict[chrom]), np.sum(core_genome_dict[chrom]))
    core_genome_length = core_genome_length + np.sum(core_genome_dict[chrom])
print(core_genome_length)


tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)


tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])


len(tbl_field_samples.data())


# # Calculate number of core bases callable in each sample

for sample in tbl_field_samples.values('sample'):
    print('.', end='')
    callable_loci_bed_fn = "%s/results/callable_loci_%s.bed" % (lustre_dir, sample)
    core_bases_callable_fn = "%s/results/core_bases_callable_%s.txt" % (lustre_dir, sample)

    if not os.path.exists(core_bases_callable_fn):
        script_fn = "%s/scripts/core_bases_callable_%s.sh" % (lustre_dir, sample)
        fo = open(script_fn, 'w')
        print('''grep CALLABLE %s | %s intersect -a - -b %s | %s genomecov -i - -g %s | grep -P 'genome\t1' | cut -f 3 > %s
''' % (
                callable_loci_bed_fn,
                bedtools,
                core_regions_fn,
                bedtools,
                GENOME_FN+'.fai',
                core_bases_callable_fn,
            ),
            file = fo
        )
        fo.close()
        st = os.stat(script_fn)
        os.chmod(script_fn, st.st_mode | stat.S_IEXEC)
        bsub(
            '-G', 'malaria-dk',
            '-P', 'malaria-dk',
            '-q', 'normal',
            '-o', '%s/logs/CL_%s.out' % (lustre_dir, sample),
            '-e', '%s/logs/CL_%s.err' % (lustre_dir, sample),
            '-J', 'CBC_%s' % (sample),
            '-R', "'select[mem>1000] rusage[mem=1000]'",
            '-M', '1000',
            script_fn)


def read_core_base_callable_file(sample='PF0249-C'):
    core_bases_callable_fn = "%s/results/core_bases_callable_%s.txt" % (lustre_dir, sample)
    with open(core_bases_callable_fn, 'r') as f:
        bases_callable = f.readline()
        if bases_callable == '':
            return(0)
        else:
            return(int(bases_callable))
   


read_core_base_callable_file()


tbl_field_samples_extended = tbl_field_samples.addfield('core_bases_callable', lambda rec: read_core_base_callable_file(rec[0]))
tbl_field_samples_extended.cut(['sample', 'core_bases_callable'])


len(tbl_field_samples.data())


ox_codes = tbl_field_samples.values('sample').array()


def calc_callable_core(callable_loci_bed_fn='/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_PA0007-C.bed'):
    core_bases_callable_fn = callable_loci_bed_fn.replace('.bed', '.core_callable.txt')
    get_ipython().system("grep CALLABLE {callable_loci_bed_fn} |     {bedtools} intersect -a - -b {core_regions_fn} |     {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} |     grep -P 'genome\\t1' |     cut -f 3 > {core_bases_callable_fn}")


calc_callable_core()


callable_loci_bed_fns = [callable_loci_bed_fn_format % ox_code for ox_code in ox_codes]
print(len(callable_loci_bed_fns))
callable_loci_bed_fns[0:2]


def count_symbol(i=1):
    if i%10 == 0:
        return(str((i//10)*10))
    else:
        return('.')


for i, callable_loci_bed_fn in enumerate(callable_loci_bed_fns):
    print('%s' % count_symbol(i), end='', flush=True)
    calc_callable_core(callable_loci_bed_fn)


for callable_loci_bed_fn in callable_loci_bed_fns:
    callable_loci_callable_fn = callable_loci_bed_fn.replace('.bed', '.callable.bed')
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} > {callable_loci_callable_fn}')


for callable_loci_bed_fn in callable_loci_bed_fns:
    callable_loci_callable_fn = callable_loci_bed_fn.replace('.bed', '.callable.bed')
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} > {callable_loci_callable_fn}')


get_ipython().system("{bedtools} intersect -a {callable_loci_bed_fn.replace('.bed', '.callable.bed')} -b {core_regions_fn} | {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} | grep -P 'genome\\t1'")
# {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} | grep -P 'genome\t1'


callable_loci_callable_fns = [(callable_loci_bed_fn_format % ox_code).replace('.bed', '.callable.bed') for ox_code in ox_codes]
callable_loci_callable_fns[0:2]


callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_merged.bed'
for i, callable_loci_callable_fn in enumerate(callable_loci_callable_fns):
    get_ipython().system('cat {callable_loci_callable_fn} >> {callable_loci_merged_fn}')
    print('%d' % (i%10), end='', flush=True)


get_ipython().system("sort -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")


get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -bga > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_merged_coverage.bed")


bedtools_command = '''{multiIntersectBed} -i {bed_files_list} -empty -g {genome} | cut -f 1-4 > {output_filename}'''.format(
    multiIntersectBed = multiIntersectBed,
    bed_files_list = " ".join(callable_loci_callable_fns),
    genome = GENOME_FN+'.fai',
    output_filename = callable_loci_fn
)
get_ipython().system('{bedtools_command}')


117*13


bases_callable = collections.OrderedDict()
core_bases_callable = collections.OrderedDict()
autosomes = ['Pf3D7_%02d_v3' % i for i in range(1, 15)]
for i, ox_code in enumerate(tbl_field_samples.values('sample')):
#     print(i, ox_code)
    this_sample_callable_loci = collections.OrderedDict()
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    for chrom in core_genome_dict.keys():
        chrom_length = len(core_genome_dict[chrom])
        this_sample_callable_loci[chrom] = np.zeros(chrom_length, dtype=bool)
    tbl_this_sample_callable_loci = (etl
                                     .fromtsv(callable_loci_bed_fn)
                                     .pushheader('chrom', 'start', 'end', 'region')
                                     .selecteq('region', 'CALLABLE')
                                     .selectin('chrom', autosomes)
                                     .convertnumbers()
                                    )
    for rec in tbl_this_sample_callable_loci.data():
        this_sample_callable_loci[rec[0]][rec[1]:rec[2]] = True
    bases_callable[ox_code] = 0
    core_bases_callable[ox_code] = 0
    for chrom in core_genome_dict.keys():
        bases_callable[ox_code] = bases_callable[ox_code] + np.sum(this_sample_callable_loci[chrom])
        core_bases_callable[ox_code] = core_bases_callable[ox_code] + np.sum((this_sample_callable_loci[chrom] & core_genome_dict[chrom]))
#     print(ox_code, bases_callable, core_bases_callable)
#     print(i, type(i))
    print('%d' % (i%10), end='', flush=True)
    
        


20296931 / 20782107 


20782107 * 0.95


proportion_core_callable = tbl_field_samples_extended.values('core_bases_callable').array()/core_genome_length


20155438 / core_genome_length


proportion_core_callable


for x in [0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.5, 0.1, 0.01]:
    print(x, np.sum(proportion_core_callable >= x), np.sum(proportion_core_callable >= x)/2512)


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.0, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram.pdf" % plot_dir)


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.9, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram_90.pdf" % plot_dir)


ox_codes = tbl_field_samples_extended.selectge('core_bases_callable', core_genome_length*0.95).values('sample').array(dtype='U12')
len(ox_codes)


ox_codes.dtype


2+2


callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged.bed'
for i, ox_code in enumerate(ox_codes):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')


get_ipython().system("sort -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")


get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -bga > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed")


get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt")


get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed')
get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt')

get_ipython().system('tabix -f -p bed /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed.gz')
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz')


merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)


merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"
accessibility_array = np.loadtxt(merged_coverage_fn,
                                 dtype={'names': ('chrom', 'pos', 'coverage'), 'formats': ('U11', 'i4', 'i4')})





merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtabix(merged_coverage_fn, region = 'Pf3D7_01_v3', header=['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray()
)


print(len(accessibility_array))
print(accessibility_array[0])
accessibility_array


accessibility_array_fn = "%s/accessibility_array.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array)


del(accessibility_array)
gc.collect()


accessibility_array_fn = "%s/accessibility_array.npy" % plot_dir
accessibility_array = np.load(accessibility_array_fn)


accessibility_array


2+2


chrom.encode('ascii')


accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['pos']


(etl
            .fromtsv(REGIONS_FN)
            .pushheader('chrom', 'start', 'end', 'region')
            .convertnumbers()
#              .valuecounts('region').displayall()
        )


accessibility_colors = {
    'Core': 'white',
    'SubtelomericHypervariable': 'red',
    'InternalHypervariable': 'orange',
    'SubtelomericRepeat': 'brown',
    'Centromere': 'black'
#     'InternalHypervariable': '#b20000',
}


def plot_accessibility(bin_size=1000, number_of_samples = 1848):

    fig = plt.figure(figsize=(11.69*1, 8.27*1))
    gs = GridSpec(2*14, 1, height_ratios=([1.0, 1.0])*14)
    gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

    print('\n', bin_size)
    for i in range(14):
        print(i+1, end=" ")
        chrom = 'Pf3D7_%02d_v3' % (i + 1)
        pos = accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['pos']
        coverage = accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['coverage']
        max_pos = np.max(pos)
        if bin_size == 1:
            binned_coverage, bin_centres = coverage, pos
        else:
            binned_coverage, bins, _ = scipy.stats.binned_statistic(pos, coverage, bins=np.arange(1, max_pos, bin_size))
            bin_centres = (bins[:-1]+bins[1:]) / 2
        ax = fig.add_subplot(gs[i*2])
        ax.plot(bin_centres, binned_coverage/number_of_samples)
    #     ax.plot(pos, coverage/number_of_samples)
        ax.set_xlim(0, 3300000)
        ax.set_xticks(range(0, len(core_genome_dict[chrom]), 100000))
        ax.set_xticklabels(np.arange(0, len(core_genome_dict[chrom])/1e+6, 0.1))
        tbl_regions = (etl
            .fromtabix(REGIONS_FN, chrom)
            .pushheader('chrom', 'start', 'end', 'region')
            .convertnumbers()
        )
        for region_chrom, start_pos, end_pos, region_type in tbl_regions.data():
            if region_type != 'Core':
                ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
        for s in 'left', 'right', 'top':
            ax.spines[s].set_visible(False)
    #         ax.set_yticklabels([])
        ax.get_xaxis().tick_bottom()
        ax.set_yticks([])

        ax.set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')

        ax.set_xlabel('')
        if i < 13:
            ax.set_xticklabels([])
    #     ax.spines['top'].set_bounds(0, len(core_genome_dict[chrom]))    
        ax.spines['bottom'].set_bounds(0, len(core_genome_dict[chrom]))
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows.png' % bin_size), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows.pdf' % bin_size))

plot_accessibility()


# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [100000, 10000, 1000, 1]:
    plot_accessibility(bin_size)


# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [500, 300]:
    plot_accessibility(bin_size)


# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [100, 10]:
    plot_accessibility(bin_size)


def plot_accessibility_region(chrom='Pf3D7_10_v3', start=1.4e+6, end=1.44e+6, bin_size=1000, number_of_samples = 1848,
                              tick_distance=5000):

    fig = plt.figure(figsize=(8, 3))

    pos_array = (
        (accessibility_array['chrom']==chrom.encode('ascii')) &
        (accessibility_array['pos']>=start) &
        (accessibility_array['pos']<=end)
    )
    pos = accessibility_array[pos_array]['pos']
    coverage = accessibility_array[pos_array]['coverage']
    min_pos = np.min(pos)
    max_pos = np.max(pos)
    print(min_pos, max_pos)
    if bin_size == 1:
        binned_coverage, bin_centres = coverage, pos
    else:
        binned_coverage, bins, _ = scipy.stats.binned_statistic(pos, coverage, bins=np.arange(min_pos, max_pos, bin_size))
        bin_centres = (bins[:-1]+bins[1:]) / 2
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(bin_centres, binned_coverage/number_of_samples)
    ax.set_xlim(min_pos, max_pos)
#     ax.set_xlim(0, 3300000)
    ax.set_xticks(range(min_pos, max_pos+1, tick_distance))
    ax.set_xticklabels(np.arange(min_pos/1e+6, (max_pos+1)/1e+6, tick_distance/1e+6))
#         for region_chrom, start_pos, end_pos, region_type, region_size in tbl_regions.data():
#             if chrom == region_chrom and region_type != 'Core':
#                 ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
    for s in 'left', 'right', 'top':
        ax.spines[s].set_visible(False)
#         ax.set_yticklabels([])
    ax.get_xaxis().tick_bottom()
    ax.set_yticks([])

#     ax.set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')

    ax.set_xlabel('')
    ax.spines['bottom'].set_bounds(min_pos, max_pos)
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows_%s_%d_%d.png' % (bin_size, chrom, start, end)), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows_%s_%d_%d.pdf' % (bin_size, chrom, start, end)))


for bin_size in [1000, 500, 300, 100]:
    plot_accessibility_region(bin_size=bin_size)


plot_accessibility_region(start=1.2e+6, end=1.6e+6, bin_size=300)


# RH2a and RH2b
plot_accessibility_region(chrom='Pf3D7_13_v3', start=1.4e+6, end=1.5e+6, bin_size=300, tick_distance=10000)


# RH2a and RH2b
plot_accessibility_region(chrom='Pf3D7_13_v3', start=1.41e+6, end=1.46e+6, bin_size=300, tick_distance=5000)


# MSP region
plot_accessibility_region(bin_size=300)


# CRT region
plot_accessibility_region(chrom='Pf3D7_07_v3', start=403000, end=406500, bin_size=1, tick_distance=500)


# WG for Thomas
for bin_size in [10000, 1000, 100, 1]:
    plot_accessibility(bin_size)


number_of_samples = 1848

fig = plt.figure(figsize=(11.69*2, 8.27*2))
gs = GridSpec(2*14, 1, height_ratios=([1.0, 0.5])*14)
gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

for i in range(14):
    print(i, end=" ")
    chrom = 'Pf3D7_%02d_v3' % (i + 1)
    accessibility_array = (etl
        .fromtabix(merged_coverage_fn, region = chrom, header=['chrom', 'pos', 'coverage'])
        .cut(['pos', 'coverage'])
        .convertnumbers()
        .toarray()
    )
    ax = fig.add_subplot(gs[i*2])
    ax.plot(accessibility_array['pos'], accessibility_array['coverage']/number_of_samples)
    ax.set_xlim(0, 3300000)
    ax.set_xticks(range(0, len(core_genome_dict[chrom]), 100000))
    ax.set_xticklabels(range(0, int(len(core_genome_dict[chrom])/1000), 100))
#         for region_chrom, start_pos, end_pos, region_type, region_size in tbl_regions.data():
#             if chrom == region_chrom and region_type != 'Core':
#                 ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
#         for s in 'left', 'right':
#             ax.spines[s].set_visible(False)
#             ax.set_yticklabels([])
    ax.set_yticks([])

    ax.set_title(chrom, loc='left')

    ax.set_xlabel('')
    if i < 13:
        ax.set_xticklabels([])
    ax.spines['top'].set_bounds(0, len(core_genome_dict[chrom]))    
    ax.spines['bottom'].set_bounds(0, len(core_genome_dict[chrom]))    


2+2





