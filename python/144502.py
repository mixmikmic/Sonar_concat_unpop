# # jammer
# ## 1 Extract, transform, and load
# ### I- Data for Starfish
# 
# Michael Gully-Santiago  
# Friday, March 31, 2017  
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns


from astropy.io import fits
import h5py


import astropy.units as u


from os import listdir


files = listdir('../data/raw/')


# We need to convert the units to what Starfish expects:
# - **wavelengths:** Angstroms
# - **fluxes:** erg/s/cm$^2$/A
# 

# update 4/21/2017:
# Some sigmas have negative or zero values!!!  Fix this!
# 

for i, file in enumerate(files):
    with open('../data/raw/{}'.format(file), 'rb') as f:
        wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')
        
    out_name = '../data/reduced/{}.hdf5'.format(file[:-4])
    fls_out = (Flux*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
    sig_out = (Flux_err*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
    #print(out_name, np.min(sig_out), np.sum(sig_out==0), np.percentile(fls_out/sig_out, 80))
    bi = sig_out <= 0
    sig_out[bi] = np.abs(fls_out[bi])
    wls_out = wlgrid*10000.0
    msk_out = np.ones(len(wls_out), dtype=int)
    f_new = h5py.File(out_name, 'w')
    f_new.create_dataset('fls', data=fls_out)
    f_new.create_dataset('wls', data=wls_out)
    f_new.create_dataset('sigmas', data=sig_out)
    f_new.create_dataset('masks', data=msk_out)
    print("{:03d}: {:.0f}  -  {:.0f}   {}".format(i, wls_out[0], wls_out[-1], out_name))
    f_new.close()


# ### The end!
# 

# # jammer
# ## 12 Extract, transform, and load the full-sampling Gl570D data
# ### I- Data for Starfish
# 
# Michael Gully-Santiago  
# Thursday, March 27, 2017
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns


from astropy.io import fits
import h5py


import astropy.units as u


from os import listdir


file = '../../../other_GitHub/splat/reference/Spectra/10020_11100.fits'


fits_dat = fits.open(file)
fd_0 = fits_dat[0]
fd_0.data.shape


with open('../data/raw/Gl570D.pic', 'rb') as f:
    wlgrid_orig, Flux_orig, Flux_err_orig = np.load(f, allow_pickle=True, encoding='bytes')


# It seems like the default SpeX Prism Library spectrum is normalized to its peak.
# 

plt.step(fd_0.data[0, :], fd_0.data[1, :]*np.max(Flux_orig))
plt.step(fd_0.data[0, :], fd_0.data[2, :]*np.max(Flux_orig))
plt.plot(wlgrid_orig, Flux_orig, '-')


# Also, the third column looks like $S/N$ not just $N$, unless I'm mistaken.  Weird.
# 

plt.plot(fd_0.data[0, :],  fd_0.data[1, :]/fd_0.data[2, :], 'o')
plt.ylabel('$\sigma ?$')


# We need to convert the units to what Starfish expects:
# - **wavelengths:** Angstroms
# - **fluxes:** erg/s/cm$^2$/A
# 

# update 4/21/2017:
# Some sigmas have negative or zero values!!!  Fix this!
# 

wlgrid = fd_0.data[0, :]
Flux = fd_0.data[1, :]*np.max(Flux_orig)
Flux_err = np.abs(fd_0.data[1, :]/fd_0.data[2, :])*np.max(Flux_orig)


plt.step(wlgrid, Flux)
plt.plot(wlgrid, Flux_err)


plt.plot(wlgrid, Flux/Flux_err, '.')


# Let's enforce $S/N <75$ and $\sigma \ne 0$.  
# That is, $N=S/75$.
# 

bi = ((Flux/Flux_err) > 75)


bi.sum()


Flux_err[bi] = Flux[bi]/75.0


bi2 = np.abs(Flux_err) == np.inf
Flux_err[bi2] = np.abs(Flux[bi2]*3.0)


out_name = '../data/reduced/Gl570D_full.hdf5'
fls_out = (Flux*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
sig_out = (Flux_err*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
#print(out_name, np.min(sig_out), np.sum(sig_out==0), np.percentile(fls_out/sig_out, 80))
bi = sig_out <= 0
sig_out[bi] = np.abs(fls_out[bi])
wls_out = wlgrid*10000.0
msk_out = np.ones(len(wls_out), dtype=int)
f_new = h5py.File(out_name, 'w')
f_new.create_dataset('fls', data=fls_out)
f_new.create_dataset('wls', data=wls_out)
f_new.create_dataset('sigmas', data=sig_out)
f_new.create_dataset('masks', data=msk_out)
print("{:.0f}  -  {:.0f}   {}".format(wls_out[0], wls_out[-1], out_name))
f_new.close()


# ### The end!
# 

# # jammer
# ## 1 Exploratory
# ### I- Look at the spectra
# 
# Michael Gully-Santiago  
# Friday, March 31, 2017  
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns


from astropy.io import fits
import h5py


# It looks like the pickle file was written in Python 2, which requires some special attention to the `encoding=` keyword argument (*kwarg*).  
# We must read the file as binary (`'rb'` argument to `open`).
# 

with open('../data/raw/2M_J0050.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')


# From Mike Line:
# > You'll probably want the units...same as in Line+2015.  
# 
# > Wavelength=microns  
# > Flux=W/m2/m  
# > err=W/m2/m  
# 

plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')


# ### Spot-check another one:
# 

with open('../data/raw/2M_J0415.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')


plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')


# ### And another:
# 

with open('../data/raw/HD3651B.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')

plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')


len(wlgrid)


# Such low resolution.
# 

# ### The end!
# 

# # jammer
# ## 06- `star_marley.py`  on Gl570D
# ### II. Run01 success
# 

import pandas as pd
from matplotlib.ticker import MaxNLocator


label = [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$", 
         r"$c^1$", r"$c^2$", r"$c^3$", r"sigAmp", r"logAmp", r"$l$"] 


# ```bash
# gully at gigayear in ~/GitHub/jammer/sf/Gl570D/output/marley_grid/run01 on master [!]
# $ $jammer/code/star_marley.py --samples=5000 --incremental_save=10
# keeping grid as is
# Using the user defined prior in $jammer/sf/Gl570D/user_prior.py
# 2017 Apr 24,12:10 PM: 9/5000 = 0.2%
# 2017 Apr 24,12:11 PM: 19/5000 = 0.4%
# 2017 Apr 24,12:11 PM: 29/5000 = 0.6%
# 2017 Apr 24,12:11 PM: 39/5000 = 0.8%
# 2017 Apr 24,12:11 PM: 49/5000 = 1.0%
# [...]
# 2017 Apr 24, 1:33 PM: 4979/5000 = 99.6%
# 2017 Apr 24, 1:33 PM: 4989/5000 = 99.8%
# 2017 Apr 24, 1:33 PM: 4999/5000 = 100.0%
# The end.
# ```
# 

ws = np.load("../sf/Gl570D/output/marley_grid/run01/temp_emcee_chain.npy")


burned = ws[:, -1000:,:]
xs, ys, zs = burned.shape
fc = burned.reshape(xs*ys, zs)
nx, ny = fc.shape


fig, axes = plt.subplots(11, 1, sharex=True, figsize=(8, 14))
for i in range(0, 11, 1):
    axes[i].plot(burned[:, :, i].T, color="k", alpha=0.2)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(label[i])

axes[10].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)


# Seems reasonable.
# 

# ## What is the Cheb spectrum doing, in light of the multi-dimensional prior?
# 

x_vec = np.arange(-1, 1, 0.01)


from numpy.polynomial import Chebyshev as Ch


# Plot a bunch of random draws from the Cheb polynomials
# 

n_samples, n_dim = fc.shape
n_draws = 900
rand_i = np.random.randint(0, n_samples, size=n_draws)


for i in range(n_draws):

    ch_vec = np.array([0]+list(fc[rand_i[i], 5:7+1]))
    ch_tot = Ch(ch_vec)
    ch_spec = ch_tot(x_vec)

    plt.plot(x_vec, ch_spec, 'r', alpha=0.05)


# This is probably creating a bias that we don't really want.  We might have to get rid of the Chebyshev polynomials altogether.
# 

# ## What are we getting compared to *Saumon et al 2006*?
# 

truth_vals = [810.0, 5.15, 0.0, 30.0] # Saumon et al. 2006;  v_z, and vsini made up from plausible values.


import corner
fig = corner.corner(fc[:, 0:2], labels=label[0:2], show_titles=True, truths=truth_vals[0:2])
fig.savefig('../results/Gl570D_exp1.png', dpi=300)


# Systematically off, but maybe not surprisingly so.
# 

# ## What do the spectra look like?
# 

dat1 = pd.read_csv('../sf/Gl570D/output/marley_grid/run01/spec_config.csv')
dat2 = pd.read_csv('../sf/Gl570D/output/marley_grid/run01/models_draw.csv')


sns.set_style('ticks')


plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.step(dat1.wl, dat2.model_comp50, 'b', label='draw')
plt.step(dat1.wl, dat1.model_composite, 'r',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('../results/Gl570D_exp1_fit.png', dpi=300, bbox_inches='tight')


plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.step(dat1.wl, dat2.model_comp50, 'b', label='draw')
plt.step(dat1.wl, dat1.model_composite, 'r',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D')
plt.legend(loc='best')
plt.yscale('linear')


CC = np.load('../sf/Gl570D/output/marley_grid/run01/CC_new.npy')


from scipy.stats import multivariate_normal


#sns.heatmap(CC, xticklabels=False, yticklabels=False)


nz_draw = multivariate_normal(dat2.model_comp50, CC)


plt.figure(figsize=(12, 7))
plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.plot(dat1.wl, dat2.model_comp50, 'b-', label='draw')

plt.plot(dat1.wl, nz_draw.rvs(), 'g-', label='noise draw')
for i in range(10):
    plt.plot(dat1.wl, nz_draw.rvs(), 'g-', alpha=0.3)

plt.plot(dat1.wl, dat1.model_composite, 'r:',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D with draws from GP')
plt.legend(loc='best')
plt.yscale('linear')


# The `logAmp` on the Gaussian Process is **way too strong**!  It should be down by a factor of $10\times$.  
# It looks like the scale should be smaller by a factor of about $5\times$.  I will change the priors.  
# I should also put a prior on the $v_z$ and $v\sin{i}$, or at least re-interpret them as calibration nuisance parameters...  
# I also lowered the Cheb variation to a maximum of 1%.
# 

# ## Next steps:
# 1. ~~Re-run with adjusted prior on the GP parameters~~
# 2. Fix and re-instantiate the `part1` of the spectral emulator covariance matrix
# 3. Make a variation of grid_tools.py that convolves with a wavelength-dependent resolution kernel.
# 4. Fit with a resolution kernel, not vsini, re-interpret the 4$^{th}$ parameter as $\sigma_R$.
# 5. Put a prior on $v_z$ equal to a fraction of a pixel.
# 

# # jammer
# ## 13- Turning off the Chebyshev polynomials
# ### Part I. Starfish outcome
# 
# (We also happen to have used the wavelength-dependent spectral resolution for the first time...)
# 

import pandas as pd
from matplotlib.ticker import MaxNLocator


label = [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$", 
         r"sigAmp", r"logAmp", r"$l$"] 


ws = np.load("../sf/Gl570D_resg/output/marley_grid/run01/temp_emcee_chain.npy")


burned = ws[:, :1399,:]
xs, ys, zs = burned.shape
fc = burned.reshape(xs*ys, zs)
nx, ny = fc.shape


burned.shape


fig, axes = plt.subplots(8, 1, sharex=True, figsize=(8, 9))
for i in range(0, 8, 1):
    axes[i].plot(burned[:, :, i].T, color="k", alpha=0.2)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(label[i])

axes[7].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)


# Seems fine, but it doesn't use the spectral emulator matrix.  Let's halt the run, and re-execute with the correct spectral emulator matrix.
# 

# The end.
# 

