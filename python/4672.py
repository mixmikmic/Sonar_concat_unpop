# Read in the output catalogs and make some summary plots showing the SNANA host galaxy sample
# 

get_ipython().magic('matplotlib inline')


import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob


reload(snhostspec)
sim = snhostspec.SnanaSimData()
sim.load_simdata_catalog('/Users/rodney/Dropbox/src/wfirst/wfirst_snhostspec3.cat')
sim.plot_efficiency_curves(dz=0.2)
print(len(sim.simdata))


reload(snhostspec)
etcout = snhostspec.SubaruObsSim('etc.output/subaruPFS_SNR.40hr.000510.dat', 1, 25, 40)
print(etcout.wave_obs.min())
print(etcout.wave_obs.max())


etcout.check_redshift(showplot=False)


plt.plot(etcout.wave_rest, etcout.mAB)


# Host galaxy magnitude vs redshift compared to 3DHST
photdat3d = fits.open('3DHST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')
f160 = photdat3d[1].data['f_F160W']
zspec = photdat3d[1].data['z_spec']
zphot = photdat3d[1].data['z_peak']
zbest = np.where(zspec>0, zspec, zphot)
usephot = photdat3d[1].data['use_phot']

ivalid = np.where(((f160>0) & (zbest>0)) & (usephot==1) )[0]
mH3D = -2.5*np.log10(f160[ivalid])+25
z3D = zbest[ivalid]
plt.plot(z3D, mH3D, 'b.', ls=' ', ms=1, alpha=0.1)
#plt.plot(hostz_med, hostmag_med, 'g.', ls=' ', ms=3, alpha=0.3)
plt.plot(hostz_deep, hostmag_deep, 'r.', ls=' ', ms=3, alpha=0.3)
ax = plt.gca()
xlim = ax.set_xlim(0,2.5)
ylim = ax.set_ylim(28,20)
ax.set_xlabel('redshift')
ax.set_ylabel('host galaxy AB magnitude')


# Examples of some spectrum plots
reload(snhostspec)
# etcoutfilelist = glob('etcout/subaruPFS*dat')
# etcoutfile_random = np.random.choice(etcoutfilelist)
# subarusim = snhostspec.SubaruObsSim(etcoutfile_random)
i=33
etcoutfile = 'etc.output/subaruPFS_SNR.40hr.000{:03d}.dat'.format(i)
subarusim = snhostspec.SubaruObsSim(etcoutfile, 1, 25, 40)
subarusim.check_redshift(snrthresh=4, showplot=False)
subarusim.plot(showspec=True)


np.where(subarusim.wave_rest>350)[0][0]


np.median(subarusim.signaltonoise[4939:7347])


np.median(np.diff(subarusim.wave_obs))


50*0.1





reload(snhostspec)
sim1 = snhostspec.SnanaSimData()
sim1.load_simdata_catalog('wfirst_snhostspec1.cat')
zcheck = []
etcoutlist=[]
fout = open("/Users/rodney/Dropbox/src/wfirst/wfirst_snhostspec4.cat", 'a')
fout.write(
    "# index zsim magsim zmatch magmatch 1hr 5hr 10hr 40hr SNRbest SNRwave SNRbin\n")
for index in sim1.simdata['index']:
    fout = open("/Users/rodney/Dropbox/src/wfirst/wfirst_snhostspec4.cat", 'a')
    hostgal = sim1.simdata[index]
    etcoutstr = ''
    for et in [1,5,10,40]:
        etcoutfile = 'etc.output/subaruPFS_SNR.{:d}hr.{:06d}.dat'.format(et, index)
        if not os.path.isfile(etcoutfile):
            etcoutstr += '  -1'
            continue
        etcout = snhostspec.SubaruObsSim(
            etcoutfile, hostgal['zmatch'], hostgal['magmatch'], et, verbose=0)
        etcout.check_redshift(snrthresh=4, showplot=False)
        if etcout.redshift_detected:
            etcoutstr += '   1'
        else:
            etcoutstr += '   0'                    
    fout.write('{:6d} {:.2f} {:.2f} {:.2f} {:.2f} {:s} {:.2f} {:.1f} {:d}\n'.format(
               hostgal['index'], hostgal['zsim'], hostgal['magsim'], 
               hostgal['zmatch'], hostgal['magmatch'], 
               etcoutstr, etcout.bestsnr, etcout.bestsnr_waverest, 
               etcout.bestbinsize))
    fout.close()





# Part IIa :  (a quick check with just 15 representative galaxy SEDs)
# 
# (1) Read in simulated host galaxy data from the catalog built in Part I
# 
# (2) Pull out a subset of galaxies around each redshift in [0.8, 1.2, 1.5, 1.8, 2.2]
# 
# (3) For each redshift subset, identify the host galaxy magnitude that is fainter than 50%, 80% and 95% of the subset in that redshift bin.
# 
# (4) Use EAZY to make a simulated spectrum for each representative galaxy.
# 
# (5) Run the Subaru ETC for each of those "percentile marker" galaxies. The ETC outputs a "S/N spectrum", with S/N per pixel vs. wavelength
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('pdb')
#import time
#tstart = time.time()


import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob


reload(snhostspec)
sim2 = snhostspec.SnanaSimData()
sim2.load_simdata_catalog('wfirst_snhostspec2.cat')


idsubset = sim2.get_host_percentile_indices(zlist=[1.5,1.8,2.2])
sim2.simulate_subaru_snr_curves(np.ravel(idsubset))


fig = plt.figure(figsize=[13,3])
for et, iax  in zip([1, 5, 10,40], [1,2,3,4]):
    ax = fig.add_subplot(1,4,iax)
    ietwin = np.where((zcheck['exptime']==et) & (zcheck['gotz']>0))[0]
    ietfail = np.where((zcheck['exptime']==et) & (zcheck['gotz']<1))[0]
    ax.plot(zcheck['z'][ietwin], zcheck['mag'][ietwin], 'ro', ls=' ')
    ax.plot(zcheck['z'][ietfail], zcheck['mag'][ietfail], 'ks', ls=' ')
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax1.set_title('1hr')
ax2.set_title('5hr')
ax3.set_title('10hr')
ax1.set_ylabel('host gal H band mag')
ax2.set_xlabel('redshift')


mastercat = wfirst.WfirstMasterHostCatalog()
mastercat.load_all_simdata()
mastercat.write('wfirst_snhostspec_master.cat')


mastercat2 = wfirst.WfirstMasterHostCatalog()
mastercat2.read('wfirst_snhostspec_master.cat')


mastercat2.mastercat = ascii.read('wfirst_snhostspec_master.cat', format='commented_header')


mastercat2.simulate_all_seds()


reload(wfirst)
sim = wfirst.WfirstSimData('SNANA.SIM.OUTPUT/IMG_2T_4FILT_MD_SLT3_Z08_Ia-01_HEAD.FITS')
sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
sim.get_matchlists()
sim.simdata.write("wfirst_snhostgal_sim.dat", format='ascii.commented_header')


# Read in the simulated SN data from the SNANA sim data files.
# Each SNANA simulation has generated a HEAD.FITS file that contains a binary table with metadata for each simulated SN and host galaxy.  The high-z host galaxy magnitudes have been drawn from distributions that match the CANDELS+CLASH sample -- so there is some selection bias built in, but it will be similar to the selection biases of the WFIRST SN survey (?). 
# 

reload(wfirst)


get_ipython().magic('pwd')


get_ipython().magic('pinfo sim.snanadata.read')


simlist = []
simfilelist_med = glob('SNANA.SIM.OUTPUT/*Z08*HEAD.FITS')
simfilelist_deep = glob('SNANA.SIM.OUTPUT/*Z17*HEAD.FITS')
hostz_med, hostmag_med = np.array([]), np.array([])
for simfile in simfilelist_med:
    sim = wfirst.WfirstSimData(simfile)
    sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
    sim.get_matchlists()
    hostz_med = np.append(hostz_med, sim.zsim)
    hostmag_med = np.append(hostmag_med, sim.mag)
    simlist.append(sim)

hostz_deep, hostmag_deep = np.array([]), np.array([])
for simfile in simfilelist_deep:
    sim = wfirst.WfirstSimData(simfile)
    sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
    sim.get_matchlists()
    hostz_deep = np.append(hostz_deep, sim.zsim)
    hostmag_deep = np.append(hostmag_deep, sim.mag)
    simlist.append(sim)    


# Now for each SNANA sim file, load in the catalog of galaxy SED data from 3DHST and use EAZY to simulate an SED.  The output simulated SEDs are stored in the sub-directory '3dHST/sedsim.output'
# 

if not os.path.isdir('3DHST/sedsim.output'):
    os.mkdir('3DHST/sedsim.output')
for sim in simlist:
    sim.load_sed_data()
    sim.simulate_seds()


# TODO NEXT : run the Subaru ETC on each simulated galaxy spectrum and determine the S/N achieved after 1 hour 5 hour, 10 hour exposures
# 

# Example of a spectrum plot
eazyspecsim = wfirst.EazySpecSim('3DHST/sedsim.output/wfirst_simsed.AEGIS.0185.dat')
eazyspecsim.plot()


photdat3d = fits.open('3DHST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')
f160 = photdat3d[1].data['f_F160W']
zspec = photdat3d[1].data['z_spec']
zphot = photdat3d[1].data['z_peak']
zbest = np.where(zspec>0, zspec, zphot)
usephot = photdat3d[1].data['use_phot']

ivalid = np.where(((f160>0) & (zbest>0)) & (usephot==1) )[0]
mH3D = -2.5*np.log10(f160[ivalid])+25
z3D = zbest[ivalid]
plt.plot(z3D, mH3D, 'b.', ls=' ', ms=1, alpha=0.1)
#plt.plot(hostz_med, hostmag_med, 'g.', ls=' ', ms=3, alpha=0.3)
plt.plot(hostz_deep, hostmag_deep, 'r.', ls=' ', ms=3, alpha=0.3)
ax = plt.gca()
xlim = ax.set_xlim(0,2.5)
ylim = ax.set_ylim(28,20)
ax.set_xlabel('redshift')
ax.set_ylabel('host galaxy AB magnitude')


# Part II :  (long version, generating simulated SEDs for all the galaxies)
# 
# (1) Read in the master catalog generated in Part I 
# 
# (2) For each simulated SN host galaxy, use the EAZY code to make a simulated host galaxy spectrum (from the best-fitting photoz template)
# 
# (3) Store each simulated spectrum as an ascii .dat file with wavelength in nm and AB mag (suitable input to the Subaru ETC).
# 
# (4) Store the revised master catalog (now updated with SED .dat file names) as a simple ascii file "wfirst_snhostspec_master2.cat"
# 

get_ipython().magic('matplotlib inline')


import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob


reload(snhostspec)
sim1 = snhostspec.SnanaSimData()
sim1.load_simdata_catalog('wfirst_snhostspec1.cat')
sim1.load_sed_data()


sim1.verbose=2
sim1.simulate_host_spectra(indexlist=[2,4], clobber=True)


sim1.write_catalog('wfirst_snhostspec2.dat')





get_ipython().magic('matplotlib inline')


import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob


# Load the catalog produced in part I and updated in part II
# 

reload(snhostspec)
sim1 = snhostspec.SnanaSimData()
sim1.load_simdata_catalog('wfirst_snhostspec1.cat')


# Read in all the Subaru PFS S/N spectra, and evaluate whether we get a redshift.  Store the outputs in a simple ascii table.
# 

reload(snhostspec)
zcheck = []
etcoutlist=[]
fout = open("/Users/rodney/Dropbox/src/wfirst/wfirst_snhostspec3.cat", 'a')
fout.write(
    "# index zsim magsim zmatch magmatch 1hr 5hr 10hr 40hr SNRbest SNRwave SNRbin\n")
for index in sim1.simdata['index']:
    fout = open("/Users/rodney/Dropbox/src/wfirst/wfirst_snhostspec3.cat", 'a')
    hostgal = sim1.simdata[index]
    etcoutstr = ''
    for et in [1,5,10,40]:
        etcoutfile = 'etc.output/subaruPFS_SNR.{:d}hr.{:06d}.dat'.format(et, index)
        if not os.path.isfile(etcoutfile):
            etcoutstr += '  -1'
            continue
        etcout = snhostspec.SubaruObsSim(
            etcoutfile, hostgal['zmatch'], hostgal['magmatch'], et, verbose=0)
        etcout.check_redshift(snrthresh=4, showplot=False)
        if etcout.redshift_detected:
            etcoutstr += '   1'
        else:
            etcoutstr += '   0'                    
    fout.write('{:6d} {:.2f} {:.2f} {:.2f} {:.2f} {:s} {:.2f} {:.1f} {:d}\n'.format(
               hostgal['index'], hostgal['zsim'], hostgal['magsim'], 
               hostgal['zmatch'], hostgal['magmatch'], 
               etcoutstr, etcout.bestsnr, etcout.bestsnr_waverest, 
               etcout.bestbinsize))
    fout.close()


len(zcheck)





reload(snhostspec)
snrspeclist = glob('etc.output/subaruPFS_SNR*dat')
z = []
mag = []
gotz = []
snrbest = []
snrbest_waverest = []
snrbest_binsize = []
exptime = []
for snrspecfile in snrspeclist:
    snrspec = snhostspec.SubaruObsSim(snrspecfile)
    snrspec.verbose=0
    # snrspec.plot(marker=' ', frame='rest', showspec=True, ls='-', lw=0.7)
    snrspec.check_redshift(snrthresh=4, showplot=False)
    z.append(snrspec.z)
    mag.append(snrspec.mag)
    exptime.append(snrspec.exptime_hours)
    gotz.append(snrspec.redshift_detected)
    snrbest.append(snrspec.bestsnr)
    snrbest_waverest.append(snrspec.bestsnr_waverest)
    snrbest_binsize.append(snrspec.bestbinsize)
snrchecktable = Table(data=[z, mag, exptime, gotz, snrbest, snrbest_waverest, snrbest_binsize], 
                     names=['z','mag','exptime','gotz','snrbest','snrbest_waverest', 'snrbest_binsize'])
snrchecktable.write('wfirst_snhostspec_redshiftcheck.dat', format='ascii.commented_header')


# Make a plot showing the host spectra that get a redshift detection and those that don't, as a function of z.
# 

zcheck = ascii.read('wfirst_snhostspec_redshiftcheck.dat', 
                    format='commented_header')
fig = plt.figure(figsize=[13,3])
ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(1,4,3, sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(1,4,4, sharex=ax1, sharey=ax1)
for et, ax  in zip([1, 5, 10,40], [ax1,ax2,ax3,ax4]):
    ietwin = np.where((zcheck['exptime']==et) & (zcheck['gotz']>0))[0]
    ietfail = np.where((zcheck['exptime']==et) & (zcheck['gotz']<1))[0]
    ax.plot(zcheck['z'][ietwin], zcheck['mag'][ietwin], 'ro', ls=' ', label='Got z')
    ax.plot(zcheck['z'][ietfail], zcheck['mag'][ietfail], 'kx', ms=8, mew=0.8, ls=' ', label='No z')
ax1.invert_yaxis()
ax1.set_title('1hr')
ax2.set_title('5hr')
ax3.set_title('10hr')
ax4.set_title('40hr')
ax1.set_ylabel('host gal H band mag')
ax2.set_xlabel('redshift')
ax4.legend(loc='lower left', numpoints=1, handletextpad=0.1, frameon=False,
           bbox_to_anchor=[1.0,1.0], bbox_transform=ax4.transAxes)
ax1.text(0.85, 23.35, "-m$>$50%", ha='left', va='center')
ax1.text(0.85, 23.85, "-80%", ha='left', va='center')
ax1.text(0.85, 24.0, "-95%", ha='left', va='center')
ax1.set_xlim(0.7,2.4)


sim = snhostspec.SnanaSimData()
sim.load_simdata_catalog('wfirst_snhostspec_master.cat')
ax = pl.gca()
x = sim.simdata['zsim']
y = sim.simdata['magsim']
hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='Blues')
ax.invert_yaxis()
ietwin = np.where((zcheck['exptime']==40) & (zcheck['gotz']>0))[0]
ietfail = np.where((zcheck['exptime']==40) & (zcheck['gotz']<1))[0]
ax.plot(zcheck['z'][ietwin], zcheck['mag'][ietwin], 'ro', ls=' ', label='Got z')
ax.plot(zcheck['z'][ietfail], zcheck['mag'][ietfail], 'kx', ms=8, mew=0.8, ls=' ', label='No z')
ax.set_xlabel('redshift')
ax.set_ylabel('Host gal H band AB mag')





# Part I :
# 
# (1) Read in host galaxy data from SNANA simulation outputs (from HEAD.FITS files)
# 
# (2) For each simulated SN host galaxy, find a real galaxy in the 3DHST catalog that has a similar redshift and H band magnitude
# 
# (3) For each simulated SN host galaxy, use the EAZY code to make a simulated host galaxy spectrum (from the best-fitting photoz template)
# 
# (4) Store each simulated spectrum as an ascii .dat file with wavelength in nm and AB mag (suitable for input to the Subaru ETC).
# 
# (5) Store the master catalog as a simple ascii file "wfirst_snhostspec1.cat"
# 

get_ipython().magic('matplotlib inline')


import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob
start = time.time()


reload(snhostspec)
sim = snhostspec.SnanaSimData()
sim.add_all_snana_simdata()


sim.load_matchdata()
sim.pick_random_matches()


sim.load_sed_data()
sim.verbose=2
sim.simulate_host_spectra(clobber=True)


sim.write_catalog("wfirst_snhostspec1.cat")


end = time.time()
print("Finished in {:.1f} sec".format(end-start))


