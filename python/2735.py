# ## Code presented in the question
# 

from copy import copy

def merge(arr, left_lo, left_hi, right_lo, right_hi, dct):
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in range(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
        for x in range(index, left_hi+1):
            dct[arr[x]] += 1

    for i in range(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def merge_sort(arr, lo, hi, dct):
    mid = (lo+hi) // 2
    if lo <= mid < hi:
        merge_sort(arr, lo, mid, dct)
        merge_sort(arr, mid+1, hi, dct)
        merge(arr, lo, mid, mid+1, hi, dct)
    return

def count_inversion(arr, N):
    lo = 0
    hi = N-1
    dct = {i:0 for i in arr}
    arr2 = copy(arr)
    merge_sort(arr, lo, hi, dct)
    return ' '.join([str(dct[num]) for num in arr2])


# ## Improved code: uses lists and slicing
# 

from copy import copy
from operator import add

def new_merge(arr, left_lo, left_hi, right_lo, right_hi, out):
    # docstring goes here
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in xrange(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
        sublist_length = left_hi+1 - index
        ones = [1]*sublist_length
        out[index:left_hi+1] = map(add, out[index:left_hi+1], ones)

    for i in xrange(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def new_merge_sort(arr, lo, hi, out):
    # docstring goes here
    mid = (lo+hi) / 2
    if lo <= mid < hi:
        new_merge_sort(arr, lo, mid, out)
        new_merge_sort(arr, mid+1, hi, out)
        new_merge(arr, lo, mid, mid+1, hi, out)
    return

def new_count_inversion(arr):
    N = len(arr)
    lo = 0
    hi = N-1
    out = [0] * N
    arr2 = copy(arr)
    new_merge_sort(arr, lo, hi, out)
    return ' '.join([str(num) for num in out])


# ## Improved code: using numpy, but only for the output variable
# 

from copy import copy
import numpy as np

def d_merge(arr, left_lo, left_hi, right_lo, right_hi, out):
    # docstring goes here
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in xrange(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
            sublist_length = left_hi+1 - index
            out[index:left_hi+1] += np.ones(sublist_length, dtype = int)

    for i in xrange(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def d_merge_sort(arr, lo, hi, out):
    # docstring goes here
    mid = (lo+hi) / 2
    if lo <= mid < hi:
        d_merge_sort(arr, lo, mid, out)
        d_merge_sort(arr, mid+1, hi, out)
        d_merge(arr, lo, mid, mid+1, hi, out)
    return

def d_count_inversion(arr):
    N = len(arr)
    lo = 0
    hi = N-1
    out = np.array(([0] * N))
    arr2 = copy(arr)
    d_merge_sort(arr, lo, hi, out)
    return ' '.join([str(num) for num in out])


# ## Tests
# 

arr = [2, 3, 1, 4]
arr2 = [2, 1, 4, 3]
arr3 = [20]
arr4 = [1, 2, 3, 4, 5, 6]
arr5 = [87, 78, 16, 94]
arr6 = [5, 4, 3, 2, 5, 6, 7]

arrs_to_test = [arr, arr2, arr3, arr4, arr5, arr6]

print [d_count_inversion(copy(test)) for test in arrs_to_test]
print [new_count_inversion(copy(test)) for test in arrs_to_test]
print [count_inversion(copy(test), len(test)) for test in arrs_to_test]


get_ipython().magic('timeit [d_count_inversion(copy(test)) for test in arrs_to_test]')
get_ipython().magic('timeit [new_count_inversion(copy(test)) for test in arrs_to_test]')
get_ipython().magic('timeit [count_inversion(copy(test), len(test)) for test in arrs_to_test]')


from random import randint
big_test = [randint(0, 100) for _ in range(10000)]

get_ipython().magic('timeit x = d_count_inversion(copy(big_test))')
get_ipython().magic('timeit x = new_count_inversion(copy(big_test))')
get_ipython().magic('timeit x = count_inversion(copy(big_test), len(big_test))')


get_ipython().magic('load_ext line_profiler')


assert False


get_ipython().magic('lprun -f d_merge d_count_inversion(copy(big_test))')


get_ipython().magic('lprun -f new_merge new_count_inversion(copy(big_test))')


get_ipython().magic('lprun -f merge count_inversion(copy(big_test), len(big_test))')


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit.Chem import TemplateAlign

IPythonConsole.use_SVG=False
# IPythonConsole.use_SVG=True  ## comment out this line for GitHub


# ## Two related macrocycles
# 

dhz_smiles = 'C[C@H]1C/C=C/[C@H](CCC/C=C/C2=CC(=CC(=C2C(=O)O1)O)O)O'
dhz = Chem.MolFromSmiles(dhz_smiles)  # from smiles

radicicol_smiles = 'C[C@@H]1C[C@@H]2[C@H](O2)/C=C\C=C\C(=O)Cc3c(c(cc(c3Cl)O)O)C(=O)O1'
radicicol = Chem.MolFromSmiles(radicicol_smiles)

my_molecules = [dhz, radicicol]

Draw.MolsToGridImage(my_molecules)


# The default depiction of the 14-membered ring is very round, an idiosyncracy of several (not just rdkit) chemoinformatic packages that [has been noted before](https://www.slideshare.net/NextMoveSoftware/rdkit-ugm-2016-higher-quality-chemical-depictions) [see slide 31 in the linked deck].
# 
# In this notebook I want to compare ways of forcing these depictions to be less round.
# 

# ## Loading sdfs from PubChem
# 

suppl = Chem.SDMolSupplier('Structure3D_CID_5359013.sdf')  # from SDF via pubchem for 3D conformation
radicicol_2 = [mol for mol in suppl][0]
Chem.SanitizeMol(radicicol_2)
radicicol_2


# As a first try, I manually downloaded the Pubchem 3D conformer SDF file for one of the molecules.  Since the SDF file specifies a 3D conformation, the rdkit-generated 2D depiction tries to use the 3D information to generate a different 2D depiction.  
# 
# To my eye, this is a small but real improvement over the prior version, but is not very scalable or general.  The resulting depiction is still imperfect (e.g. the ester/lactide moiety looks pretty funky, etc.)
# 

# ## Align to a non-macrocyclic common substructure
# 

aligner = Chem.MolFromSmarts('[*r14][cR2]2[cR1][cR1](-[#8])[cR1][cR1](-[#8])[cR2]2[#6r14](=[#8])[#8r14][#6r14]')
Chem.GetSSSR(aligner)
AllChem.Compute2DCoords(aligner)

for mol in my_molecules:
    AllChem.GenerateDepictionMatching2DStructure(mol, 
                                                 aligner,
                                                 acceptFailure = True)

highlight_lists = [mol.GetSubstructMatch(aligner) for mol in my_molecules]
Draw.MolsToGridImage(my_molecules,
                     # highlightAtomLists = highlight_lists  ## uncomment this line to show the substructure match
                    )


# A second strategy is to align macrocycles to a non-cyclic substructure.  Since the depiction for the non-cyclic substructure will generally not be "rounded", this forces *parts* of the macrocycles to be depicted in a more regular fashion.
# 
# The results above show an improvement for the lactide part of the ring, but some rounding is still evident.  To my eye, these are probably the best depictions of these molecules in the notebook.
# 
# However, a caveat of this approach is that by using a bigger and bigger substructure, more of the macrocyclic ring is normalized gets, but at some point, the rest of the ring gets worse instead of better.
# 

aligner_bigger = Chem.MolFromSmarts('[*r14][cR2]2[cR1][cR1](-[#8])[cR1][cR1](-[#8])[cR2]2[#6r14](=[#8])[#8r14]@[#6r14]@[#6r14]')
Chem.GetSSSR(aligner_bigger)
AllChem.Compute2DCoords(aligner_bigger)

for mol in my_molecules:
    AllChem.GenerateDepictionMatching2DStructure(mol, 
                                                 aligner_bigger,
                                                 acceptFailure = True)

highlight_lists = [mol.GetSubstructMatch(aligner_bigger) for mol in my_molecules]
Draw.MolsToGridImage(my_molecules,
                     # highlightAtomLists = highlight_lists  ## uncomment this line to show the substructure match
                    )


# # Aligning to a conformer of cyclotetradecane
# 

suppl = Chem.SDMolSupplier('Structure2D_CID_67524.sdf')  # from SDF via pubchem for 3D conformation
ring_templ = [mol for mol in suppl][0]
Chem.GetSSSR(ring_templ)

ring_templ


# A third strategy is to use `AlignMolToTemplate2D()` and to force parts of the molecule to have a desired depiction.  Here, I download a 2D sdf file from Pubchem for cyclotetradecane and try to use it as a template for the macrocyclic part of my target molecules.
# 

# identify atoms in the 14-membered ring

ring_info_list = [mol.GetRingInfo() for mol in my_molecules]
mol_atoms_in_rings = [ring_info.AtomRings() for ring_info in ring_info_list]

size_14_rings = []
for rings in mol_atoms_in_rings:
    for ring in rings:
        if len(ring) == 14:
            size_14_rings.append(ring)
            
print(size_14_rings)


# **ASIDE:**
# 
# I had to use `RingInfo` to get the macrocyclic ring atoms because SMARTS `[*r14]` queries will _not_ hit the ring atoms in the oxirane ring or in the benzene ring. `[*r14]` will only hit atoms for which the _smallest_ ring they are a part of is a 14-cycle.  Is there a better SMARTS query to use here?

for idx, mol in enumerate(my_molecules):
    TemplateAlign.AlignMolToTemplate2D(mol, 
                                       ring_templ, 
                                       match=size_14_rings[idx],
                                       clearConfs=True)

Draw.MolsToGridImage(my_molecules,
                    highlightAtomLists = size_14_rings
                    )


# The macrocycles look great!  However, the rest of the molecule, in particular the benzene ring, is a bit stretched.
# 

# # Aligning to a Template from ChemDraw
# 
# I made a template from ChemDraw and saved it as an .sdf file.
# 

cd_templ = [mol for mol in Chem.SDMolSupplier('chemdraw_template.sdf')][0]

print(Chem.MolToSmarts(cd_templ))

cd_templ


matches = [mol.GetSubstructMatch(cd_templ) for mol in my_molecules]
print(matches)


for idx, mol in enumerate(my_molecules):
    TemplateAlign.AlignMolToTemplate2D(mol, 
                                       cd_templ, 
                                       match=matches[idx],
                                       clearConfs=True)

Draw.MolsToGridImage(my_molecules,
                    # highlightAtomLists = size_14_or_6_rings
                    )


# These depictions look really good.  The only problem is the overlap of the explicit hydrogens of the radicicol ring.  They can be eliminated manually, although this is also not an ultra-scalable strategy.
# 

for atom in radicicol.GetAtoms():
    atom.SetNumExplicitHs(0)
    atom.UpdatePropertyCache()

radicicol


# The other problem with the radicicol representation is the weird bond angle of the oxirane ring.  
# 
# Looking at the documentation for `AllChem.Compute2DCoords()`, it seems there is a ` coordMap` object that would allow us to specify atoms to keep fixed when generating new 2D coordinates.  However, I couldn't figure out how to use it.
# 

radicicol.GetConformer(0).GetAtomPosition(0)


my_dict = {idx: radicicol.GetConformer(0).GetAtomPosition(idx) for idx in matches[1]}
print(my_dict)

AllChem.Compute2DCoords(radicicol, coordMap = my_dict)





# $$\require{mhchem}$$
# # Overlaying optical and mass spectrometry imaging datasets using OpenMSI and IPython
# 
# This notebook shows an example of comparing optical and mass spectrometry images in IPython using OpenMSI components to read the MSI data files.
# 
# The data here is very simple: it is a manually spotted 384-well MALDI plate containing spots with **canola oil** (top row), or  **toasted sesame oil** (middle row) or sample blanks (i.e. MALDI matrix with no sample; bottom row).  Each sample was replicated six times within each row.
# 
# Samples were prepared by pre-spotting MALDI matrix (an equimolar mixture of <a href="http://en.wikipedia.org/wiki/Gentisic_acid">DHBA</a> and <a href="http://en.wikipedia.org/wiki/Alpha-Cyano-4-hydroxycinnamic_acid">&alpha;-cyano-4-hydroxycinnamic acid (CHCA)</a>) in a stainless steel MALDI plate.  After the spots dried, 1 &mu;L aliquots of the oil samples, each diluted 1000x in chloroform containing 0.1% trifluoroacetic acid, were manually pipetted on top of the MALDI spots.
# 
# MALDI acquisition was with a Thermo LTQ-XL Orbitrap mass spectrometer operating in the Orbitrap mode.  
# 
# MS<sup>2</sup> data was obtained for two ions: from a (triolein+Na<sup>+</sup>) ion at 907.73 Da ($\cf{C57H104O6Na+}$), and from a related ion at 906.76 Da, which arises primarily from <sup>13</sup>C substitution on a singly-unsaturated variant of triolein, $\cf{^{12}C56~^{13}C1H102O6Na+}$).  The observed ions in each MS<sup>2</sup> spectrum are primarily from loss of a single acyl chain.
# 
# In canola oil, triolein is the most abundant triacylglycerol.  In contrast, in toasted sesame oil triolein is present but in a relatively lower amount (lower-molecular weight TAGs are more abundant).
# 
# Although the samples were spotted on a 384-well plate, data were acquired in imaging mode (~400 &mu;m spatial resolution) to serve as an example of OpenMSI and IPython processing of mass spectrometry imaging and optical data.  Spatial resolution was kept low to keep the file size small for this example analysis.
# 

# ## Preliminaries: importing required Python modules, providing paths to data
# 

### Importing required python modules for these computations ####
# General stuff:
get_ipython().magic('matplotlib inline')
import numpy as np
import mkl
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.cm as cm

import sys
def isneeded(x):
    if x not in sys.path:
        sys.path.append(x)

isneeded('/Users/curt/openMSI_SVN/openmsi-tk/')
isneeded('/Users/curt/openMSI_localdata/')

#OpenMSI stuff for getting my source images:
from omsi.dataformat.mzml_file_CF import *

# Image registration
import imreg_dft as ird


# ## Reading MSI data into IPython using OpenMSI's `mzml_file()` reader.
# 

#Reading a datafile of spotted samples in MALDI
omsi_ms2 = mzml_file(basename="/Users/curt/openMSI_localdata/re-reimaging_TI.mzML")


# ##### Plotting the TIC images and example spectra from the MSI data
# 

#Plotting TIC images for each scan type
f, axarr = plt.subplots(3, 1, figsize=(6, 9))
for ind in range(len(omsi_ms2.data)):
    axarr[ind].imshow(omsi_ms2.data[ind][:, :, :].sum(axis=2).T)  #total ion chromatogram images
    axarr[ind].set_title('TIC for scan type '+omsi_ms2.scan_types[ind])


#Plotting spectra at the most intense pixel for each scan type:
f, axarr = plt.subplots(3, 1, figsize=(9, 12))
for ind in range(len(omsi_ms2.data)):
    nx, ny, nmz = omsi_ms2.data[ind].shape
    maxx, maxy = np.unravel_index(omsi_ms2.data[ind][:, :, :].sum(axis=2).argmax(), dims=[nx, ny])
    axarr[ind].plot(omsi_ms2.data[ind][maxx, maxy, :])  
    axarr[ind].set_title(('Spectrum at pixel (%s, %s) for scan type ' % (maxx, maxy))+omsi_ms2.scan_types[ind])


# ## Importing an optical image into Python
# 
# In this example, we align an optical image with the MSI image.  Although this data is for a region of a spotted MALDI plate, in the future the same code could be applied to tissue images, such as histochemical iamges.
# 

#grayscale image of plate:
photo = misc.imread('/Users/curt/openMSI_localdata/Simple_canolaTAG_vs_sesameTAG_vs_blank_plate_image.bmp')
plt.figure(figsize=(20,10))
f = plt.imshow(photo, cmap=cm.Greys_r)


# ## Registration (i.e. matching & lining up) the optical image with the MSI image.
# ##### Prepration for registration: resizing images and normalizing MS image by chosen representative ion
# 
# In this "big crystal" MALDI experiment, the presence of "hot spots" means that intensity of a given ion varies wildly from pixel to pixel.  Normalizing the MS image by a particular ion helps eliminate this effect and shows more clearly the spot locations in the MSI data.  For tissue images where MALDI "hot spots" are not an issue, this step would probably not be required.
# 

#plotting optical image resized to be the same size as MS image

ms1ticImage = omsi_ms2.data[1][:, :, :].sum(axis=2).T
photoSmall = misc.imresize(photo,ms1ticImage.shape)

f, ax = plt.subplots(2, 1)
ax[0].imshow(photoSmall, cmap=cm.Greys_r)
ax[1].imshow(ms1ticImage, cmap=cm.Greys_r)


#Normalize the MS image by the intensity of the triolein MS1 peak
mzind = abs(omsi_ms2.mz[1]-907.7738).argmin()
ms1_907 = omsi_ms2.data[1][:, :, mzind-2:mzind+2].sum(axis=2).T
normimage = ((ms1_907)) / (ms1ticImage+1.)
print normimage.dtype
f, ax = plt.subplots(2, 1)
ax[0].imshow(photoSmall, cmap=cm.Greys_r)
ax[1].imshow(normimage)


# ##### Doing image registration: finding the optimal overlay of the images, and plotting the result
# 

result = ird.similarity(photoSmall, normimage, numiter=20)
f, axarr = plt.subplots(2,2)
axarr[0, 0].imshow(photoSmall)
axarr[0, 1].imshow(normimage)
axarr[1, 0].imshow(photoSmall-normimage)
axarr[1, 1].imshow(result['timg'])
print result['tvec']


# ## Final result: 
# ## an overlay of interpolated MS<sup>1</sup> intensity for the triolein ion on the optical image
# 

#Overlaying the "registered" MSI data on the optical image to make the final plot
msBig = misc.imresize(result['timg'], photo.shape, 'bicubic')

msBig_masked = np.ma.masked_where(msBig <= 35, msBig)


plt.figure(figsize=(30,15))
plt.imshow(photo, cmap=cm.Greys_r, alpha=1)
plt.imshow(msBig_masked, alpha=0.7)


# # Analysis of chemical similarity in `python` using `rdkit`
# 
# This notebook shows simple examples of finding and comparing the chemical [fingerprints](http://rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf) of molecular structures using [`rdkit`](http://www.rdkit.org/docs/GettingStartedInPython.html).
# 
# 
# 
# ### Required libraries and modules
# 

# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.cm as cm

# scipy
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hc

# seaborn -- for better looking plots
import seaborn as sns

# pandas 
import pandas as pd

# rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


# ## Clustering molecules by fingerprint
# 

smiles_list = ['CC',
               'CCC',
               'CCO',
               'c1ccccc1',
               '[Na+].[Cl-]',
               '[Na+].[O-]C(=O)C',
               'c1ccsc1',
               'CC(O)=O',
               'C=CC(=O)N',
               'Nc1ccccc1']

names_list = ['ethane',
              'propane',
              'ethanol',
              'benzene',
              'sodium chloride',
              'sodium acetate',
              'thiophene',
              'acetic acid', 
              'acrylamide',
              'aniline']

names_dict = {smiles: name for (smiles, name) in zip(smiles_list, names_list)}
print names_dict

mols = []
fingerprints = []

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mols.append(mol)

fingerprint_mat = np.vstack(np.asarray(rdmolops.RDKFingerprint(mol, fpSize = 2048), dtype = 'bool') for mol in mols)


dist_mat = pdist(fingerprint_mat, 'jaccard')

dist_df = pd.DataFrame(squareform(dist_mat), index = smiles_list, columns= smiles_list)

dist_df


# set a mask so that only the lower-triangular part of the distance matrix is plotted
mask = np.zeros_like(dist_df, dtype = 'bool')
mask[np.triu_indices_from(mask, k = 1)] = True

# make the plot
sns.heatmap(dist_df, 
            mask=mask, 
            annot=True,
            cmap = cm.viridis,
            linewidths = 0.25,
            xticklabels=names_list,
            yticklabels=names_list
           )


# use the distance matrix to hierarchically cluster the results
z = hc.linkage(dist_mat)
dendrogram = hc.dendrogram(z, labels=dist_df.columns, leaf_rotation=90)
plt.show()


# find the order of leaves in the clustering results
new_order = dendrogram['ivl']

# reorder both rows and columns according to the clustering order
reordered_dist_df = dist_df[new_order].reindex(new_order)
reordered_names = [names_list[smiles_list.index(item)] for item in new_order]



# plot again
sns.heatmap(reordered_dist_df, 
            mask=mask, 
            annot=True, 
            cmap = cm.viridis, 
            linewidths = 0.25,
            xticklabels=reordered_names,
            yticklabels=reordered_names,
            )


# ## Plotting the fingerprint and distance matrices for various fingerprint sizes
# 

def calc_fingerprint_mat(mols, fingerprint_size, show_plot=True):
    # Given a list of rdkit molecules of length m, and a fingerprint size n, calculate a m-by-n fingerpint matrix
    # :param:    mols               list of rdkit molecules
    # :param:    fingerprint_size   integer, must be a power of 2 for results to be meaningful
    # :param:    show_plot          boolean, whether to show the plots
    # :return:   fingerprint_mat    a boolean matrix of size m-by-fingerprint_size
    
    # calculate fingerprint matrix
    fp_tuple = (np.asarray(rdmolops.RDKFingerprint(mol, fpSize = fingerprint_size), dtype = 'bool') for mol in mols)
    fingerprint_mat = np.vstack(fp_tuple)
    
    if show_plot:
        f, axes = plt.subplots(1, 2, figsize = (12, 4))
    
        # construct fingerprint matrix plot
        sns.heatmap(fingerprint_mat, 
                    cmap = cm.viridis,
                    yticklabels=names_list,
                    ax = axes[0]
                   )
        axes[0].set_title('Fingerprint matrix with size %s' % fingerprint_size)
    
    # construct distance matrix
    dist_mat = pdist(fingerprint_mat, 'jaccard')
    dist_df = pd.DataFrame(squareform(dist_mat), 
                           index = smiles_list, 
                           columns = smiles_list)
    
    # plot distance matrix
    if show_plot:
        mask = np.zeros_like(dist_df, dtype = 'bool')
        mask[np.triu_indices_from(mask, k = 1)] = True

        # make the plot
        ax2 = sns.heatmap(dist_df, 
                    mask=mask, 
                    annot=True,
                    cmap = cm.viridis,
                    linewidths = 0.25,
                    xticklabels=names_list,
                    yticklabels=names_list,
                    ax = axes[1]
                   )
        axes[1].set_title('distance matrix with size %s' % fingerprint_size)

   
    return fingerprint_mat


foo = calc_fingerprint_mat(mols, fingerprint_size=8)


foo = calc_fingerprint_mat(mols, fingerprint_size=16)


foo = calc_fingerprint_mat(mols, fingerprint_size=32)


foo = calc_fingerprint_mat(mols, fingerprint_size=64)


foo = calc_fingerprint_mat(mols, fingerprint_size=2048)


# ## Overview
# 
# Most clustering methods are designed to work well when the number of points in each cluster is approximately the same.  What about when one cluster has a far larger number of data points?

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Generating toy data
# Here we start with an "equipartioned" sample data set of 600 points (200 per cluster) and then add 6000 more points ten-fold more points to one of the blobs.  (I.e. the total number of points in the largest blob is ~30-fold higher than in the other clusters.
# 

# Generate equipartitioned sample data
centers = [[0, 0], [20, 18], [18, 20]]

X, labels_true = make_blobs(n_samples=600, 
                            centers=centers, 
                            cluster_std=0.4, 
                            random_state=0)

# Generate 10x more data for one cluster
extra_X0, labels_true_extra = make_blobs(n_samples=6000, 
                                    centers=centers[0], 
                                    cluster_std=0.4, 
                                    random_state=0)

# Combine the datasets
X = np.vstack([X, extra_X0])
labels_true = np.concatenate([labels_true, labels_true_extra])


# Scale all the data
X = StandardScaler().fit_transform(X) 

# Plot the data
xx, yy = zip(*X)
plt.scatter(xx, yy)
plt.show()


# ## Clustering this data with DBSCAN
# 

db = DBSCAN(eps=0.15, min_samples=50).fit(X)
core_samples = db.core_sample_indices_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print n_clusters_


# ## Plotting the results
# 

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# This notebook shows that for well-resolved clusters like these, DBSCAN can handle an 11-fold excess of points in one of the clusters and still return reliable results.
# 

# ## Timing
# 
# I'm not sure if DBSCAN is $O(n\log n)$ or if it is $O(n^2)$ in the number of data points, but it is at least one of those.  For a data set of 6600 data points, the code below shows that the core clustering step takes less than a second.
# 

get_ipython().magic('timeit db = DBSCAN(eps=0.15, min_samples=50).fit(X)')


# ## Sources for this code:
# 
# * A [question](http://stackoverflow.com/questions/18237479/dbscan-in-scikit-learn-of-python-save-the-cluster-points-in-an-array) on StackOverflow.
# * A [scikit-learn help page](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html) on DBSCAN.
# 




# # Purpose of this notebook
# 

# This notebook demonstrates how to parse molecular structures from the web using a Jupyter notebook.  In addition to standard modules such as `numpy`, this notebook uses these specialized Python modules:
# 
# * `rdkit` - the best available open python package for representing and computing on molecules in 2D and in 3D
# * `pandas` - this the python package that adds support for `R`-like data frames (_annotated_ tabular data) in python.  It also includes some nice tools for scraping HTML data tables, as we shall see.
# 
# The notebook begins with `import` statements, telling the interpreter what files to load.
# 

# # Setup: importing required python modules
# 

# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.cm as cm

# scipy
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hc

# seaborn -- for better looking plots
import seaborn as sns

# pandas 
import pandas as pd

# rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


# # Scraping data from the web
# 

# The demonstration data for this notebook comes from the URL **`http://sgtc-svr102.stanford.edu/hiplab/compounds/browse/2/`** .  This page lists SMILES structures for compounds from the TimTec natural products library of ~800 compounds.  More information on the TimTec library is available from the vendor at **`http://www.timtec.net/natural-compound-library.html`** .  To access the stanford URL, VPN access to Stanford's network is likely required.
# 
# The table is parsed into python's `pandas` package using the convenient `pandas` function `read_html()`.  This function returns a _list_ of all the tables found on the given page.  Our page of interest contains only one table, so this list is only one element long.  Each element of the list is a `pandas` DataFrame.
# 

url = 'http://sgtc-svr102.stanford.edu/hiplab/compounds/browse/2/'

list_of_tables = pd.read_html(url)

# the _only_ table on the page is the first table in the list
table = list_of_tables[0]


# # Turning strings into rdkit molecules
# 

# The **SMILES** column contains molecule structural information in the [simplified molecular input line entry system](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) nomenclature.  Click the link for more info.
# 
# Now that the table is structured into a data table, the strings that represent each structure can be parsed into molecules that the `rdkit` package understands.  Only the **SMILES** column of the DataFrame needs to be parsed.  For each SMILES structure (i.e. each row of the DataFrame), we both instantiate a `rdkit` molecule object for the structure as well as compute a fingerprint for that structure.  A molecular `fingerpint` is a bit vector that is computed from the molecular structure.  Since it is just a numerical vector, it allows rapid comparison of two molecules.  You can read some more about how fingerprints are computed in `rdkit`'s documentation [here](http://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity).  There is also useful information in a 2012 [slide deck](http://rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf) (pdf) by the principal author of `rdkit`, Greg Landrum.  
# * There is a large library in `rdkit` called `Fingerprints.FingerprintMols`, which [seems]( http://www.rdkit.org/Python_Docs/rdkit.Chem.Fingerprints.FingerprintMols-module.html) like it would have the requisite functionality for fingerprinting, but according to [this 2014 archived email correspondence from Greg Landrum](https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CAOSYiOLVNk3eUzEm-oWmWN81qY8WBFzvmjeg_1j1od_QJcgF7w@mail.gmail.com/), this entire module should be deprecated.  The `rdmolops.RDKFingerprint()` function is the one to use.
# 
# If `rdkit` cannot parse a SMILES string for whatever reason, it will return `None`.  If you try to do **`FingerprintMols.FingerprintMol(None)`**, it will generate an error.  Thus an rdkit failure to parse must specifically be checked and handled separately.  This is what the `if`...`else`...`continue`... loop does below.
# 

smiles_list = table['SMILES']

mols = []
fingerprints = []

for idx, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mols.append(mol)
    else:
        print 'Unable to parse item %s, SMILES string %s and so this molecule will be skipped.' % (idx, smiles)
        print 'Please check validity of this string or manually remove it from smiles_list.'
        continue
    
fingerprint_mat = np.vstack(np.asarray(rdmolops.RDKFingerprint(mol, fpSize = 2048), dtype = 'bool') for mol in mols)


smiles_list.pop(215)


# Now that all fingerprints have been computed, we can find the distances between all the fingerprints.  The "fingerprint" is a bit vector, i.e. a boolean vector.  Distance metrics for comparing them should be appropriate for boolean data.  The most popular option is probably the "Tanimoto" similarity.  But as the Wikipedia page describes, "Tanimoto" distance metrics, despite being the most popular term in the chemoinformatics literature, are not true distance metrics and are very closely related to "Jaccard" similarity / distance metrics.
# 

dist_mat = pdist(fingerprint_mat, 'jaccard')

dist_df = pd.DataFrame(squareform(dist_mat), index = smiles_list, columns= smiles_list)

# set a mask
mask = np.zeros_like(dist_df, dtype = 'bool')
mask[np.triu_indices_from(mask, k = 1)] = True


# plot
ax = sns.heatmap(dist_df, mask=mask, cmap = cm.viridis, xticklabels=20, yticklabels=20, )
ax.tick_params(axis='both', which='major', labelsize=3)


# With a distance matrix, we can perform hierarchical clustering.  This also requires choosing a linkage method.  Here I simply use the `scipy.cluster.hierarchy.linkage()` default method, which is single linkage.  
# 

# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
z = hc.linkage(dist_mat, metric='jaccard')
plt.figure(figsize=[4, 20])
dendrogram = hc.dendrogram(z, 
                           orientation = 'left',

                           labels = dist_df.columns,
                           show_leaf_counts = True,
                           show_contracted = True,
                           leaf_font_size = 2
                          )

plt.show()


# reorder dist_df according to clustering results
new_order = dendrogram['ivl']

# reorder both rows and columns
reordered_dist_df = dist_df[new_order].reindex(new_order)

# plot again
ax = sns.heatmap(reordered_dist_df, mask=mask, cmap = cm.viridis, xticklabels=20, yticklabels=20)
ax.tick_params(axis='both', which='major', labelsize=3)


# The distance matrix plot is now much more interpretable.  The order of molecules along the rows (and columns) is now shifted so that the similar molecules are near each other.  Off-diagonal elements of the distance matrix with very low values are as close to the diagonal as possible.  Thus, highly "blue" regions indicate groups of related molecules.  We can extract and plot a few of those.   
# 
# As a way to limit the number of structure-pairs plotted, here are choose a very stringent similarity cutoff, 0.02.  
# 

low_distance = np.where(np.logical_and(dist_df <= 0.02,
                                       dist_df > 0)
                        )

similar_smiles_pairs = [indices for indices in zip(low_distance[0], low_distance[1]) 
                                          if indices[0] < indices[1]]


f, ax = plt.subplots(len(similar_smiles_pairs), 2, figsize = (10, 90))

for idx, pair in enumerate(similar_smiles_pairs):
    mol1 = Chem.MolFromSmiles(smiles_list[pair[0]])
    mol2 = Chem.MolFromSmiles(smiles_list[pair[1]])
    ax[idx, 0].imshow(Draw.MolToImage(mol1, size=(200, 200), fitImage=True))
    ax[idx, 1].imshow(Draw.MolToImage(mol2, size=(200, 200), fitImage=True))
    ax[idx, 0].grid(False)
    ax[idx, 1].grid(False)
    ax[idx, 0].axis('off')
    ax[idx, 1].axis('off')





