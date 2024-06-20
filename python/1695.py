# # The time-frequency tradeoff in signal processing
# While physics may be famous for its "uncertainty principle", the world of signal processing has a very similar rule at its core. Often called the "time-frequency tradeoff", this idea states that the more we understand how a signal changes in time, the less detail we have about its frequency makeup.
# 
# To illustrate this, I'm going to use two of my favorite new tools - Jupyter widgets and Binder. We'll play around with a time-varying signal (a recording of my voice), and see how the time-frequency tradeoff works.
# 
# First, we'll load some data
# 

# For loading and processing data
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import wavfile

# For making nifty widgets
from IPython.display import display
from ipywidgets import interact, widgets
get_ipython().magic('matplotlib inline')


# Here's a wavfile of me speaking
fs, data = wavfile.read('../data/science_is_awesome.wav')
times = np.arange(data.shape[0]) / float(fs)


# Let's look at a quick plot of the data. 
# 

f, ax = plt.subplots()
ax.plot(times, data)
_ = ax.set_title('The raw signal', fontsize=20)


# As we can see, it is a single, time-varying signal. This represents the changing air pressure that occurs as a result of my vocal chords vibrating. There's clearly something going on, but its' a bit hard to decipher.
# 
# However, we have years of research in auditory processing on our side, and so we know that underneath this 1-D signal is a much more complicated story in frequency space.
# 
# To show this, I'll plot the raw signal, as well as a spectrogram of the signal:
# 

f, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(times, data)
_ = axs[0].set_title('The raw signal', fontsize=20)
spec, freqs, spec_times, _ = axs[1].specgram(data, Fs=fs)
_ = axs[1].set_title('A spectrogram of the same thing', fontsize=20)
_ = plt.setp(axs, xlim=[times.min(), times.max()])
_ = plt.setp(axs[1], ylim=[freqs.min(), freqs.max()])


# The spectrogram attempts to do the following things:
# 
# 1. Break down our 1-D signal into a bunch of sine waves, moving from low-frequency to high-frequency
# 1. Create a **window** around a subset of time in the 1-D signal. Report how much each sine wave is "contributing" to the signal within that window. E.g., if the signal is changing very slowly, it will result in hot colors near the bottom of the plot (the lower frequencies).
# 1. Now do the same thing for another window a little bit later. The second window might be **overlapping** with the first window by some amount.
# 1. Continue this process until we have stepped through the entire original signal. Each window gives us a column of the spectrogram, and when we stitch them all together, we get the nice plot above
# 
# This is often called a **time-frequency decomposition**. This means that we have taken a single time-varying signal, and decomposed it into a bunch of time-varying frequency bands.
# 
# ## Now on to time-frequency tradeoffs
# As you may have noticed, there are two things that are particularly easy to vary when you calculate the spectrogram. One is the **window size**, and the other is the **amount of overlap** between two windows. These are parameters that we can pass to the spectrogram function, and they allow us to control the time-frequency tradeoff of our plot.
# 
# As an example, I'll show the same plot above, but now with a couple sliders that let you play around with the window size, and the window overlap. See what happens when you drag the window size to be larger and larger, as well as smaller and smaller.
# 
# The parameter **n_fft** controls how large the window is. The spectrogram function calculates the frequency content in each window using something called a Fast Fourier Transform. **n_fft** controls how many points are used in the FFT, AKA how large the window is.
# 

f, ax = plt.subplots(figsize=(10, 5))

nfft_widget = widgets.IntSlider(min=3, max=15, step=1, value=10)
n_overlap_widget = widgets.IntSlider(min=3, max=15, step=1, value=9)

def update_n_overlap(*args):
    n_overlap_widget.value = np.clip(n_overlap_widget.value, None, nfft_widget.value-1)
nfft_widget.observe(update_n_overlap, 'value')
n_overlap_widget.observe(update_n_overlap, 'value')

def func(n_fft, n_overlap):
    spec, freqs, spec_times, _ = ax.specgram(data, Fs=fs,
                                        NFFT=2**n_fft, noverlap=2**n_overlap,
                                        animated=True)
    ax.set(xlim=[spec_times.min(), spec_times.max()],
           ylim=[freqs.min(), freqs.max()])
    plt.close(f)
    display(f)
w = interact(func, n_fft=nfft_widget,
             n_overlap=n_overlap_widget)


# In this case, the window size (n_fft) is the biggest factor in determining our frequency / time tradeoff. If you use really large windows, then you lose time detail. If you use really small windows, then you lose frequency detail. This is the time-frequency tradeoff in action.
# 
# # Other time-frequency methods
# There are lots of other methods out there for doing time-frequency decompositions, and many of them exist to try and work out a better balance between time and frequency resolution. One really popular method uses little window functions called "morlet wavelets". This is similar to the windows that we used above, but with a more complicated window.
# 
# With wavelets, we can define an arbitrary set of frequency bands to include in the time-frequency decomposition. We can also define the **number of cycles** to keep in each wavelet. We'll create a wavelet for each frequency band, and the number of cycles will determine how long it is.
# 
# If we keep the number of cycles fixed at 5. Wavelets with a lower frequency will naturally be longer, while wavelets with smaller frequency will be shorter. This is because lower frequencies vary over longer stretches of time, so completing 5 cycles takes a while.
# 
# In this way, we're trying to tune the size of the window to the frequency band. We're basically saying "frequencies that vary over long periods of time should have a longer window" (and vice versa).
# 
# We can play around with this effect in the same way as above. Below, we'll plot the **continuous wavelet transform** using Morlet wavelets. You can play around both with the number of frequencies used in this transform, as well as the number of cycles in each wavelet. See how this affects the time-frequency tradeoff.
# 
# (note that we are now showing the y-axis on a log scale, to account for the fact that in general, we care more about frequency resolution at lower frequencies, and time resolution at higher frequencies)
# 

# Note that now the sliders won't update until you release the mouse to save time.
f, ax = plt.subplots(figsize=(10, 5))
def func(n_cycles, n_freqs):
    plt.close(f)
    freqs = np.logspace(np.log10(100), np.log10(20000), n_freqs)
    amps = mne.time_frequency.cwt_morlet(data[np.newaxis, :], fs,
                                         freqs, n_cycles=n_cycles)
    amps = np.log(np.abs(amps))[0]
    ax.imshow(amps, animated=True, origin='lower', aspect='auto')
    display(f)
    
n_cycles_widget = widgets.IntSlider(min=5, max=50, step=1, value=3, continuous_update=False)
n_freqs_widget = widgets.IntSlider(min=10, max=150, step=10, value=50, continuous_update=False)
w = interact(func, n_cycles=n_cycles_widget, n_freqs=n_freqs_widget)


# As you can see above, changing the cycles and number of frequencies in the wavelets also tends to smear the plot along either the time dimension (x-axis) or frequency dimension (y-axis). You might have also noticed that it doesn't smear things quite as bad as the `specgram` function above. That's because the wavelet transform tends to do a better job at balancing time-frequency resolution than the FFT.
# 
# ## Wrap up
# There are lots of methods out there for doing time-frequency decompositions, but they all have to deal with the same fundamental tradeoff of time vs. frequency. The details of each algorithm will determine how elegantly they handle this, and some are more well-suited to accentuate one component of your signal over another.
# 
# If you want to learn more about the time-frequency tradeoff, check out the following links:
# 
# 1. [Here's a good stackoverflow question on this topic](http://stackoverflow.com/questions/5887366/matlab-spectrogram-params) (though note that it uses Matlab instead of python)
# 1. Here are wikipedia pages on [window functions](https://en.wikipedia.org/wiki/Window_function) and [spectral leakage](https://en.wikipedia.org/wiki/Spectral_leakage)
# 




# # Signal processing and time-frequency analysis
# 
# Signal processing is a fundamental component of any neuroscientific analysis, and sadly is one of the more poorly-taught fields out there (for scientists, anyway). Basically, this is just a way of manipulating and understanding signals (usually that vary in time). In predictive modeling of language, the two signals we tend to care about are ones that come from the brain (e.g., EEG, ECoG) and ones that come from our mouths (e.g., speech, language). Signal processing is a HUGE field, and in neuroscience we usually only scratch the surface. It’s most important to be familiar with these techniques, and particularly know when it might affect your data.
# 
# > The most basic package for doing this kind of thing in python is called “[scipy.signal](http://docs.scipy.org/doc/scipy/reference/signal.html)”. It’s the “signal” submodule of the scipy package. It’s got a lot of great features, though it can be more confusing than it needs to be.
# 
# In practice, it's often easier to use signal processing tools that have been pre-crafted for neuroscience analysis. A good place to start is [MNE-python](http://martinos.org/mne/stable/mne-python.html), which has a `time_frequency` module that has many of these functions in more user-friendly form. In particular, [these are some functions](http://martinos.org/mne/stable/python_reference.html#time-frequency) that can create spectrograms of sound or brain activity. I recommend checking out their "examples gallery" [here](http://martinos.org/mne/stable/auto_examples/index.html) to get an idea for what's possible.
# 
# Of particular interest to neuroscience are "time-frequency" decompositions. Extracting the frequency content of a signal allows you to take a time-domain signal (e.g., air pressure over time) into a frequency-domain signal (e.g., frequency power over frequency value). This tells us “how much” of each frequency is in the original signal.
# In neuroscience, we do this on sliding windows of a signal, and slid those windows across time (computing the frequency representation each time) in order to build a spectrogram of the signal, aka, how the frequency content changes across time.
# 
# > To perform these things on a signal, check out the "periodogram" and "specgram" functions of pyplot. Also check out the time-frequency module that I linked to earlier.
# 
# In general, you need to provide the signal itself, as well as the sampling rate of the signal (e.g., how many points per second are we recording). This lets the function know the highest frequency that can be detected in the signal.
# 
# There are often other parameters to play around with, these often correspond to the parameters for the windows, or how the Fourier Transform is performed. Play around with them and see how it changes things.
# 
# A quick note - if you've got a 3-D matrix (e.g., a spectrogram that is time x frequency x amplitude), then a useful way of visualizing is to plot it as an image (aka, time and frequency are the two sides of the image, and the color intensity is the amplitude). Check out the "imshow" function for this.
# 
# ## On to the data
# For a first look at time-frequency processing, we'll first look at a raw audio file. This will show us how we can extract time-frequency information from a raw waveform. Then, we'll perform the same analysis on brain activity and see what there is to learn.
# 

import mne
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')


# Read in an MNE epochs file
data = mne.io.Raw('./data/audio_clean_raw.fif', add_eeg_ref=False)


# Load our times file, which is stored as a CSV of events
mtime = pd.read_csv('./data/time_info.csv', index_col=0)
mtime = mtime.query('t_type=="mid" and t_num > 0')

# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * data.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T

# This is just a metadata dictionary.
# If we had multiple event types, we'd specify here.
einfo = dict(myevent=1)


# Now create an epochs object so we're time-locked to sound onset
tmin, tmax = -.5, 2
epochs = mne.Epochs(data, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(epochs._data.shape)


# As you'll see below, audio is represented as changing air pressure over time. It looks like audio has structure if you look at it in raw form, but it's quite difficult to make any sense of this:
# 

data_plt = epochs._data[0]
plt.plot(epochs.times, data_plt.T)


# Fortunately for us, we've got a long history of audio analysis and signal processing to draw on. One thing that scientists have learned about audio is that it's actually a combination of lots of sine waves that oscillate at different frequencies. Signal processing gives us a set of tools to uncover this underlying structure in the data.
# 
# We'll use a *Morlet Wavelet* to extract the extent to which each of these sine waves contributes to the signal. A wavelet is basically a tiny "window" signal that you use as a filter for the audio waveform. The extent to which the waveform overlaps with your wavelet, the output will be larger.
# 
# So, we construct wavelets for a range of frequencies and see which are represented most strongly in the signal. Moreover, we will slide this window across our audio waveform to see how the frequency representation changes over time. This is accomplished with the `cwt_morlet` function in MNE. 
# 

freqs = np.logspace(2, np.log10(5000), 128)
spec = mne.time_frequency.cwt_morlet(data_plt, epochs.info['sfreq'], freqs)


# Our output is now n_epochs x n_freqs x time
spec.shape


# And now we reveal the spectral content that was present in the sound
plt.pcolormesh(epochs.times, freqs, spec[0])


# Doesn't look too clean. This is because of something called the *1/f power law*. Basically, the total power in a frequency drops off as the frequency is larger (by a factor of `1/f`). Because of this, the power at low frequencies tends to dominate our visualization. So, we can fix this by taking the log of the output:
# 

# Whoops, that looks pretty messy. Let's try taking the log...
f, ax = plt.subplots()
ax.pcolormesh(epochs.times, freqs, np.log(spec[0]))


# Now we've revealed the underlying spectro-temporal structure that was present in the sound. By using a time-frequency decomposition, we've taken a really complicated looking time-varying signal, and we've revealed some interesting stuff that was happening under the surface. Let's do the same for our brain signal and see if we can gain similar insights.
# 

# ## Brain activity
# 

# So it's clear that we can learn interesting underlying structure in sounds by taking looking at their spectral content. What about the brain? There's a long history of spectral analysis in neuroscience. This is because most signals we record from the brain also vary in time. Using the same analytical methods, we can parse apart different kinds of signal in our brain data.
# 

# First we'll load some brain data
brain = mne.io.Raw('./data/ecog_clean_raw.fif', add_eeg_ref=False)


# We will create an "events" object by turning the start times into indices
ev = mtime['start'] * brain.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T
einfo = dict(myevent=1)


# Now we have time-locked Epochs
tmin, tmax = -.5, 2
brain_epochs = mne.Epochs(brain, ev, einfo, tmin, tmax,
                          baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(brain_epochs._data.shape)


# Pull a subset of epochs to speed things up
n_ep = 10
use_epochs = np.random.choice(range(len(brain_epochs)), replace=False, size=n_ep)
brain_epochs = brain_epochs[use_epochs]


# Before getting into a full fancy spectral analysis. We'll first look at the *Power Spectral Density*. This is basically a "snapshot" of spectral power for the entire trial (aka, not moving in time).
# 
# We'll use a method known as *Multitaper Spectral Estimation*. In this case, rather than using a sliding window as we did with Morlet Wavelets, we'll construct a set of filters that are all overlapping in time. These filters are constructed so that they are *orthogonal* to one another. Basically, this means that they are designed to extract different kinds of information from the signal, which in turn means that we get a more stable estimate of the spectral content in the signal.
# 

# Multitaper
psd = []
for ep in brain_epochs._data:
    ipsd, freqs = mne.time_frequency.multitaper._psd_multitaper(
        ep, sfreq=brain_epochs.info['sfreq'])
    psd.append(ipsd)
psd = np.array(psd)
psd = pd.DataFrame(psd.mean(0), columns=freqs)
psd.index.name = 'elec'
psd['kind'] = 'mt'
psd.set_index('kind', append=True, inplace=True)

# Collect them
psd.columns.name = 'freq'


# Just as before, we'll apply the log and plot
psd.apply(np.log).groupby(level='kind').mean().T.plot(figsize=(15, 5))


# So what can we learn from this? Well, it seems like there is more power at low frequencies (that's the `1/f` bit we covered earlier). Moreover, at some point the power seems to level off. This is known as the "noise floor" and it basically means that we can't make meaningful statements about the brain past that point.
# 

# ## TFR
# 

# OK, so we've estimated the *Power Spectral Density* of the brain activity, but let's take a look at how this changes over time. For this, we can use the same method above
# 

# Here we'll define the range of frequencies we care about
freqs = np.logspace(1, np.log10(150), 20)

# This determines the length of the filter we use to extract spectral content
n_cycles = 5


freqs.shape


# Now we'll extract the TFR of our brain data
df_tfr = []
tfr, itc = mne.time_frequency.tfr_morlet(brain_epochs, freqs, n_cycles)
for i, elec in enumerate(tfr.data):
    ielec = pd.DataFrame(elec, index=freqs, columns=brain_epochs.times)
    ielec['elec'] = i
    ielec.index.name = 'freq'
    ielec.set_index(['elec'], inplace=True, append=True)
    df_tfr.append(ielec)
df_tfr = pd.concat(df_tfr, axis=0)


f, ax = plt.subplots()
plt_df = df_tfr.xs(20, level='elec')
y_axis = plt_df.index.values
# y_axis = np.arange(plt_df.shape[0])
ax.pcolormesh(plt_df.columns.values, y_axis, plt_df.values,
              cmap=plt.cm.RdBu_r)





# *Note - you can find the nbviewer of this post [here](https://github.com/choldgraf/write-ups/blob/master/neuro/coherence_correlation.ipynb)*
# 
# # Coherence vs. Correlation - a simple simulation
# A big question that I've always wrestled with is the difference between correlation and coherence. Intuitively, I think of these two things as very similar to one another. Correlation is a way to determine the extent to which two variables covary (normalized to be between -1 and 1). Coherence is similar, but instead assesses "similarity" by looking at the similarity for two variables in frequency space, rather than time space.
# 
# There was a nice paper that came out a while back that basically compared these two methods in order to see when they'd produce the same result, when they'd produce different results, and when they'd break down [1]. They made a lot of nice plots like this:
# 

# <img src='http://chrisholdgraf.com/wp-content/uploads/2015/05/eeg_coh.png', style='width:300px'><img>
# 

# Here I am recreating this result in the hopes of giving people a set of scripts to play around with, and giving a bit more intuition.
# 
# [1] http://www.ncbi.nlm.nih.gov/pubmed/8947780
# 
# ---
# 
# First things first, we'll import some tools to use
# 

import scipy as sp
import pandas as pd
import mne
from itertools import product
from nitime import viz
from itertools import combinations
import numpy as np


# ## Creating our sine waves
# Recall that the equation for a sinusoidal wave is:
# 
# $$ Asin(2{\pi}ft + 2\pi\phi)$$
# 
# Where $f$ is the frequency of the wave, $$t$$ indexes time, and $$2\pi\phi$$ defines a phase offset of the wave. Then, $$A$$ scales the wave's amplitude.
# 

# We can generate these sine wave parameters, then stitch them together
amplitude_values = [1, 3, 10, 15]
phase_values = [0, .25, .33, .5]
freq = 2
signal_vals = list(product(amplitude_values, phase_values))
amps, phases = zip(*signal_vals)

# We'll also define some noise levels to see how this affects results
noise_levels = [0, 2, 4, 8]

# Now define how long these signals will be
t_stop = 50
time = np.arange(0, t_stop, .01)

# We're storing everything in dataframes, so create some indices
ix_amp = pd.Index(amps, name='amp')
ix_phase = pd.Index(phases, name='phase')

# Create all our signals
signals = []
for noise_level in noise_levels:
    sig_ = np.array([amp * np.sin(freq*2*np.pi*time + 2*np.pi*phase) for amp, phase in signal_vals])
    noise = noise_level * np.random.randn(*sig_.shape)
    sig_ += noise
    ix_noise = pd.Index([noise_level]*sig_.shape[0], name='noise_level')
    ix_multi = pd.MultiIndex.from_arrays([ix_amp, ix_phase, ix_noise])
    signals.append(pd.DataFrame(sig_, index=ix_multi))
signals = pd.concat(signals, 0)
signals.columns.name = 'time'


# ## Computing connectivity 
# Now we've got a bunch of sinewaves with the parameters chosen above. Next, we will calculate the coherence and the correlation between all pairs of signals. This way we can see how these values change for different kinds of input signals.
# 

con_all = []
for ix_noise, sig in signals.groupby(level='noise_level'):
    # Setting up output indices
    this_noise_level = sig.index.get_level_values('noise_level').unique()[0]
    ix_ref = np.where(sig.eval('amp==3 and phase==0'))[0][0]
    ix_time = pd.Index(range(sig.shape[0]), name='time')
    ix_cc = pd.Index(['cc']*sig.shape[0], name='con')
    ix_coh = pd.Index(['coh']*sig.shape[0], name='con')

    # Calculating correlation is easy with pandas
    cc = sig.T.corr().astype(float).iloc[:, ix_ref]
    cc.name = None
    cc = pd.DataFrame(cc)
    # We'll use MNE for coherenece
    indices = (np.arange(sig.shape[0]), ix_ref.repeat(sig.shape[0]))
    con, freqs, times, epochs, tapers = mne.connectivity.spectral_connectivity(
        sig.values[None, :, :], sfreq=freq, indices=indices)
    con_mn = con.mean(-1)
    con_mn = pd.DataFrame(con_mn, index=cc.index)
    
    # Final prep
    con_mn = con_mn.set_index(ix_coh, append=True)
    cc = cc.set_index(ix_cc, append=True)
    con_all += ([con_mn, cc])
con_all = pd.concat(con_all, axis=0).squeeze().unstack('noise_level')


# ## Visualizing results
# First off, we'll look at what happens to sine waves of varying parameters, for different levels of noise.
# 
# Remember, each tuple is (amplitude, phase_lag). The first number controls how large the signal is, and the second controls the difference in phase between two sine waves. 
# 

f, axs = plt.subplots(2, 2, figsize=(15, 10))
for ax, (noise, vals) in zip(axs.ravel(), con_all.iteritems()):
    ax = vals.unstack('con').plot(ax=ax)
    ax.set_title('Noise level: {0}'.format(noise))
    ax.set_ylim([-1.1, 1.1])


# That's already an interesting picture - as you can see, coherence is far more robust to differences between the two signals. Here are a few thoughts:
# 
# 1. Correlation varies widely (between 0 and 1) for differences in phase lag. However, coherence remains relatively stable.
# 1. Coherence values are smaller in general for a signal with any noise
# 1. However, coherence is more robust for increasing levels of noise, while correlations start to drop to 0
# 
# To illustrate number 1, let's plot correlation and coherence against each other:
# 

plt_df = con_all.stack('noise_level').unstack('con').reset_index('noise_level')
ax = plt_df.plot('cc', 'coh', c='noise_level', kind='scatter',
                 cmap=plt.cm.Reds, figsize=(10, 5), alpha=.5, s=50)
ax.set_title('CC vs Coherence')


# As you can see here, coherence remains the same (except for when it occasionally increases to 1) while correlation is much more dependent on the phase relationship between the signals. Moreover, as the signal SNR degrades, the correlation shrinks to 0, while the coherence remains the same.
# 
# Let's take a look at how the correlation and coherence relate to the actual shape of the signals:
# 

# Set up a dataframe for plotting
noise_level = noise_levels[1]
plt_df = con_all.copy()
plt_df = con_all[noise_level].unstack('con')

# Define 16 signals to plot
sig_combinations = list(product([ix_ref], range(16)))

plt_sig = signals.xs(noise_level, level='noise_level')
n_combs = len(sig_combinations)
f, axs = plt.subplots(n_combs/4, 4, figsize=(15, n_combs/3*5))
for (comp_a, comp_b), ax in zip(sig_combinations, axs.ravel()):
    plt_sig.iloc[[comp_a, comp_b]].T.head(250).plot(ax=ax, legend=None)
    ax.set_title('CC: {0}\nCoh: {1}'.format(*plt_df.iloc[comp_b, :].values))


# Finally, for a more direct comparison, we can look directly at the difference between the two as a function of both noise level and the sine wave parameters.
# 

diff_con = con_all.stack('noise_level').unstack('con')
diff = diff_con['cc'] - diff_con['coh']
diff = diff.unstack('noise_level')


diff.plot(figsize=(15, 5))


# So what does this mean? Well, the relationship between coherence and correlation is too complicated to sum it up in a single line or two. However, it is clear that correlation is more sensitive to differences between signals in time. Coherence, on the other hand, is more reliable for these differences. Moreover, correlation degrades quickly with an increase in noise, while coherence remains the same.
# 
# As such, if you care about understanding the relationship between two signals as it pertains to time, then perhaps correlation is the way to go. On the other hand, if you want a robust estimate of the amount of overlap in the structure of two signals, then coherence may be the best bet.
# 




from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd
sys.path.insert(0, '/Users/choldgraf/github/mne-python/')
import mne
from mne.viz.utils import ClickableImage
from mne.channels.layout import generate_2d_layout


plt.rcParams['image.cmap'] = 'gray'


im_path = '/Users/choldgraf/github/mne-python/mne/data/image/mni_brain.gif'
layout_path = '/Users/choldgraf/github/mne-python/mne/data/image/custom_layout.lay'


im = imread(im_path)


# ## Display image, then click/store positions
# 

# Make sure that inline plotting is off before clicking
get_ipython().magic('matplotlib qt')
click = ClickableImage(im)


get_ipython().magic('matplotlib inline')


# The click coordinates are stored as a list of tuples
click.plot_clicks()
coords = click.coords
print coords


# ##Show results
# 

# Generate a layout from our clicks and normalize by the image
# lt = generate_2d_layout(np.vstack(coords), bg_image=im) 
# lt.save(layout_path + 'custom_layout.lay')  # To save if we want

# Or if we've already got the layout, load it
lt = mne.channels.read_layout(layout_path)


# Create some fake data
nchans = len(coords)
nepochs = 50
sr = 1000
nsec = 5
events = np.arange(nepochs).reshape([-1, 1])
events = np.hstack([events, np.zeros([nepochs, 2])])
data = np.random.randn(nepochs, nchans, sr * nsec)
info = mne.create_info(nchans, sr, ch_types='eeg')
epochs = mne.EpochsArray(data, info, events)


# Using the native plot_topo function
f = mne.viz.plot_topo(epochs.average(), layout=lt)


# Now with the image plotted in the background
f = mne.viz.plot_topo(epochs.average(), layout=lt)
ax = f.add_axes([0, 0, 1, 1])
ax.imshow(im)
ax.set_zorder(-1)


# ## Using Craigslist to compare prices in the Bay Area
# In the [last post](http://chrisholdgraf.com/querying-craigslist-with-python/) I showed how to use a simple python bot to scrape data from Criagslist. This is a quick follow-up to take a peek at the data.
# 
# > Note - data that you scrape from Craigslist is pretty limited. They tend to clear out old posts, and you can only scrape from recent posts anyway to avoid them blocking you. 
# 
# Now that we've got some craigslist data, what questions can we ask? Well, a good start would be to see where we want (or don't want) to rent our house. Let's compare the housing market in a few different regions of the Bay Area.
# 

# Seaborn can help create some pretty plots
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
get_ipython().magic('matplotlib inline')
sns.set_palette('colorblind')
sns.set_style('white')


# First we'll load the data we pulled from before
results = pd.read_csv('../data/craigslist_results.csv')


# # Price distributions
# As a start, let's take a look at the distribution of home prices to get an idea for what we're dealing with:
# 

f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(results['price'].dropna())


# That's not super useful - it looks like we have a highly skewed distribution with a few instances way out to the right. We'll convert to a log scale from here on to make it easier to comprehend:
# 

f, ax = plt.subplots(figsize=(10, 5))
results['logprice'] = results['price'].apply(np.log10)
sns.distplot(results['logprice'].dropna())
ax.set(title="Log plots are nicer for skewed data")


# Don't forget the log mappings:
print(['10**{0} = {1}'.format(i, 10**i) for i in ax.get_xlim()])


# However, what we really want is to look at the breakdown of prices for a few locations in the bay area. Luckily Craigslist stores this in the URL of our search, so we can easily split this up with a pandas `groupby`:
# 

f, ax_hist = plt.subplots(figsize=(10, 5))
for loc, vals in results.groupby('loc'):
    sns.distplot(vals['logprice'].dropna(), label=loc, ax=ax_hist)
    ax_hist.legend()
    ax_hist.set(title='San Francisco is too damn expensive')


summary = results.groupby('loc').describe()['logprice'].unstack('loc')
for loc, vals in summary.iteritems():
    print('{0}: {1}+/-{2}'.format(loc, vals['mean'], vals['std']/vals.shape[0]))
    
print('Differences on the order of: $' + str(10**3.65 - 10**3.4))


# That's a bit unsurprising - San Francisco is significantly more expensive than any other region in the area. Note that this is a log scale, so small differences at this scale == large differences in the raw values.
# 
# However, it looks like the shapes of these prices are different as well. If any of these distributions aren't symmetric around the center, then describing it with the mean +/- standard deviation isn't so great.
# 
# Perhaps a better way to get an idea for what kind of deal we're getting is to directly calculate price per square foot. Let's see how this scales as the houses go up.
# 

# We'll quickly create a new variable to use here
results['ppsf'] = results['price'] / results['size']


# These switches will turn on/off the KDE vs. histogram
kws_dist = dict(kde=True, hist=False)
n_loc = results['loc'].unique().shape[0]
f, (ax_ppsf, ax_sze) = plt.subplots(1, 2, figsize=(10, 5))
for loc, vals in results.groupby('loc'):
    sns.distplot(vals['ppsf'].dropna(), ax=ax_ppsf,
                 bins=np.arange(0, 10, .5), label=loc, **kws_dist)
    sns.distplot(vals['size'].dropna(), ax=ax_sze,
                 bins=np.arange(0, 4000, 100), **kws_dist)
ax_ppsf.set(xlim=[0, 10], title='Price per square foot')
ax_sze.set(title='Size')


# So it looks like size-wise, there aren't many differences here. However, with price per square foot, you'll be paying a lot more for the same space in SF.
# 
# Finally, let's take a look at how the price scales with the size. For this, we'll use a `regplot` to fit a line to each distribution.
# 

# Split up by location, then plot summaries of the data for each
n_loc = results['loc'].unique().shape[0]
f, axs = plt.subplots(n_loc, 3, figsize=(15, 5*n_loc))
for (loc, vals), (axr) in zip(results.groupby('loc'), axs):
    sns.regplot('size', 'ppsf', data=vals, order=1, ax=axr[0])
    sns.distplot(vals['ppsf'].dropna(), kde=True, ax=axr[1],
                 bins=np.arange(0, 10, .5))
    sns.distplot(vals['size'].dropna(), kde=True, ax=axr[2],
                 bins=np.arange(0, 4000, 100))
    axr[0].set_title('Location: {0}'.format(loc))

_ = plt.setp(axs[:, 0], xlim=[0, 4000], ylim=[0, 10])
_ = plt.setp(axs[:, 1], xlim=[0, 10], ylim=[0, 1])
_ = plt.setp(axs[:, 2], xlim=[0, 4000], ylim=[0, .002])


# And now on top of one another
# 

f, ax = plt.subplots()
locs = [res[0] for res in results.groupby('loc')]
for loc, vals in results.groupby('loc'):
    sns.regplot('size', 'ppsf', data=vals, order=1, ax=ax,
                scatter=True, label=loc, scatter_kws={'alpha':.3})

# If we want to turn off the scatterplot
scats = [isct for isct in ax.collections
         if isinstance(isct, mpl.collections.PathCollection)]
# plt.setp(scats, visible=False)

ax.legend(locs)
ax.set_xlim([0, 4000])
ax.set_ylim([0, 10])


# Basically, lines that go down more steeply mean you get a better deal the bigger the place is.
# 
# For instance, if you're in the southbay you might be paying \$6/sqf for a 600 sq. ft. place, but $1/sqf for a 2000 sq. ft. place. On the other hand, San Francisco is pretty consistent, with a relatively flat line. This means that you'll be paying pretty much the same per square foot regardless of how big your place is. In fact, all of the other regions seem to follow the same trend - so if you're looking for more efficient big-place finds, go with the South Bay.
# 
# Also note that this gives us information about the uncertainty in these estimates. The error bars are so wide for San Francisco because we don't have many data points at high values (because there aren't that many places >2000 square feet in SF). It's anyone's guess as to what this would cost.
# 

# # Text analysis
# Finally, we can also learn a bit from the text in the post titles. We could probably get better information by using the post text itself, but this would require some extra legwork looking up the URL of each entry and pulling the body of text from this. We'll stick with titles for now.
# 
# To do this, we'll use some text analysis tools in `scikit-learn`. This is good enough for our purposes, though if we wanted to do something fancier we could use something like `gensim`, `word2vec`, or `nltk`. (we'd also probably need a lot more data).
# 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string


# First we'll do some quick data cleaning - we'll only keep datapoints with a title, and then define some characters to remove so that the definition of "a word" makes more sense.
# 

word_data = results.dropna(subset=['title'])

# Remove special characters
rem = string.digits + '/\-+.'
rem_chars = lambda a: ''.join([i for i in a if i not in rem])
word_data['title'] = word_data['title'].apply(rem_chars)


# Next, we'll remove words that are too specific (from a geographical standpoint) to the regions we're using. Otherwise you'll just get a bunch of clusters with streetnames etc. predicting the Bay Area region.
# 

loc_words = {'eby': ['antioch', 'berkeley', 'dublin', 'fremont', 'rockridge',
                     'livermore', 'mercer', 'ramon'],
             'nby': ['sausalito', 'marin', 'larkspur', 'novato', 'petaluma', 'bennett', 
                     'tiburon', 'sonoma', 'anselmo', 'healdsburg', 'rafael'],
             'sby': ['campbell', 'clara', 'cupertino', 'jose'],
             'scz': ['aptos', 'capitola', 'cruz', 'felton', 'scotts',
                     'seabright', 'soquel', 'westside', 'ucsc'],
             'sfc': ['miraloma', 'soma', 'usf', 'ashbury', 'marina',
                     'mission', 'noe']}

# We can append these to sklearn's collection of english "stop" words
rand_words = ['th', 'xs', 'x', 'bd', 'ok', 'bdr']
stop_words = [i for j in loc_words.values() for i in j] + rand_words
stop_words = ENGLISH_STOP_WORDS.union(stop_words)


# Finally, we will vectorize this data so that it can be used with sklearn algorithms. This takes a list of "bags" of words, and turns it into a list of vectors, where the length of each vector is the total number of words we've got. Each position of the vector corresponds to 1 word. It will be "1" if that word is present in the current item, and 0 otherwise:
# 

vec = CountVectorizer(max_df=.6, stop_words=stop_words)
vec_tar = LabelEncoder()

counts = vec.fit_transform(word_data['title'])
targets = vec_tar.fit_transform(word_data['loc'])
plt.plot(counts[:3].toarray().T)
plt.ylim([-1, 2])
plt.title('Each row is a post, with 1s representing presence of a word in that post')


# Let's do a quick description of the most common words in each region. We can use our vectorized vocabulary and see which words were most common.
# 

top_words = {}
for itrg in np.unique(targets):
    loc = vec_tar.classes_[itrg]
    # Pull only the data points assigned to the current loction
    icounts = counts[targets == itrg, :].sum(0).squeeze()
    
    # Which counts had at least five occurrences
    msk_top_words = icounts > 5
    
    # The inverse transform turns the vectors back into actual words
    top_words[loc] = vec.inverse_transform(msk_top_words)[0]


# Then, we'll print the words that are unique to each area by filtering out ones that are common across locations:
# 

unique_words = {}
for loc, words in top_words.iteritems():
    others = top_words.copy()
    others.pop(loc)
    unique_words[loc] = [wrd for wrd in top_words[loc]
                         if wrd not in np.hstack(others.values())]
for loc, words in unique_words.iteritems():
    print('{0}: {1}\n\n---\n'.format(loc, words))


# Apparently people in the North Bay like appliances, people in Santa Cruz like the beach, people in the East Bay need the Bart, and people in San Francisco have victorians...who knew.
# 
# Just for fun we'll also do a quick classification algorithm to see if some machine learning can find structure in these words that separates one location from another:
# 

mod = LinearSVC(C=.1)
cv = StratifiedShuffleSplit(targets, n_iter=10, test_size=.2)

coefs = []
for tr, tt in cv:
    mod.fit(counts[tr], targets[tr])
    coefs.append(mod.coef_)
    print(mod.score(counts[tt], targets[tt]))
coefs = np.array(coefs).mean(0)


# Doesn't look like it (those are horrible generalization scores), but we'll look at what coefficients it considered important anyway:
# 

for loc, icoef in zip(vec_tar.classes_, coefs):
    cut = np.percentile(icoef, 99)
    important = icoef > cut
    print('{0}: {1}'.format(loc, vec.inverse_transform(important)))


# You may note that these are quite similar to the words that were unique to each location as noted above - such is the power of machine learning :)
# 

# ## So what have we learned?
# Well, you might say that we've merely quantified what everybody already knows: San Francisco is expensive, really expensive. If you're looking for a place in the Bay Area, you can expect to shell out a lot more for the same square footage.
# 
# However, what's also interesting is that apartments in the Bay Area don't seem to obey the same rules that other regions do - they don't necessarily become more economically efficient as the place gets bigger. This is in stark contrast to the south bay, where places are pretty expensive in general, but in ways that you'd expect for an apartment.
# 
# Finally, there are probably lots of other cool things that you could do with these datasets, especially if you wanted to break things down by neighborhood and collect more data.
# 

# The first step of any analysis in neuroscience is took look at your raw data. This is important for all kinds of reasons - most notably masking sure that you haven't made a mistake in labeling channels, timepoints, etc.
# 
# ## Background reading
# The ecog signal is really interesting, quite complicated, and poorly understood. It's best to have a background in signal processing, though if you don't then try to learn this as you do your first analyses...there are quite a few "gotchas" if you don't know what you're doing in time-domain signals.
# 
# That said, here we'll focus on some general analysis techniques as well as the ecog signal specifically. There's a lot of attention paid to high-gamma activity in ecog. Here are a few papers that discuss where it comes from what what it means.
# 
# * [Nathan Crone's review paper](http://www.ncbi.nlm.nih.gov/pubmed/21081143) on high gamma activity and our understanding of it is a good place to start.
# * [Kai Miller's paper on broadband high-gamma](http://www.jneurosci.org/content/29/10/3132.full) suggests that high-gamma activity isn't really an "oscillatory" signal, and isn't restricted to >70Hz regions of the power spectrum. Instead, it's a broadband signal that reflects as a global power increase across all frequencies.
# * [Supratim Ray's paper](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000610) on the cellular origin of high-gamma covers what kinds of neurophysiology might create this signal. 
# 
# ## Now on to the data
# First we'll just look at some raw data. We've collected a dataset where the subject was listening to spoken sentences. We have an audio file with those sentences, a brain file with their brain activity, and a list of times for when the sentences began. (actually, rather than a raw audio file we have a spectrogram of audio, but we'll get to that later).
# 
# We'll load some brain activity, and audio recorded at the same time. We'll use the package `MNE-python` for keeping track of brain/audio timeseries data.
# 

import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ecogtools as et
import seaborn as sns
sns.set_style('white')

# This ensures that you'll be able to use interactive plots
get_ipython().magic('matplotlib notebook')


# These are stored as "fif" files, which MNE reads easily
brain = mne.io.Raw('./data/ecog_clean_raw.fif', add_eeg_ref=False, preload=True)
audio = mne.io.Raw('./data/spectrogram.fif', add_eeg_ref=False, preload=True)


# First we'll just glance through the brain activity.
# For plot visualizations
scale = brain._data[0].max()

# This should pop out an interactive plot, scroll through the data
f = brain.plot(scalings={'eeg': scale}, show=False)
f.set_size_inches(8, 5)


# OK, looks like a lot of stuff is going on. You can scroll left and right to see more times, and up/down to see more channels. This is useful for getting a general view of the data, but we care specifically about the response to sound. We'll pull out only the timepoints corresponding to audio onsets. We can do this using an `Epochs` object and the time onsets.
# 

# Load our times file, which is stored as a CSV of events
mtime = pd.read_csv('./data/time_info.csv', index_col=0)

# Pull only the trials we care about
mtime = mtime.query('t_type=="mid" and t_num > 0')

# These are the start/stop times for sound
mtime.head()


# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * brain.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T

# This is just a metadata dictionary.
# If we had multiple event types, we'd specify here.
einfo = dict(myevent=1)

# First columns == event start ix, last columns == event id
ev[:5]


# Epochs objects are a way to represent data that is time-locked to events of interest. We can pull a window before/after an event, which is useful for things like baselining, averaging, and seeing a time-locked effect.
# 

# First low-pass filter to remove some noise
brain_epochs = brain.copy()
brain_epochs.filter(0, 20)

# Now we'll turn the raw array into epochs shape
tmin, tmax = -.5, 2
epochs = mne.Epochs(brain_epochs, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# shape (trials, channels, time)
print(epochs._data.shape)


# Let's average across all epochs to see if any channels are responsive
epochs.average().plot()


# This looks...complicated. This is because we're using the raw brain signals we recorded from the patient. These signals have all kinds of things going on in them, which is why the outputs look so noisy. To get around this, we can try to isolate specific parts of the raw signals.
# 
# In particular, "high-frequency" components of the signal have been shown to be responsive to stimuli in the world. We can reveal this information by doing a "high-pass" filter that removes all of the lower-frequency components.
# 

# We'll do this on the raw data for reasons we can talk about later
brain.filter(70, 150)


# Now we will take the *amplitude* of this filtered data. It tells us how large/small is the signal in general. We do this using a "Hilbert Transform", which is a mathematical trick to calculate the amplitude of an oscillatory signal (as well as it's phase but we won't worry about that here)
# 

# We'll also add zeros to our data so that it's of a length 2**N.
# In signal processing, everything goes faster if your data is length 2**N :-)
next_pow2 = int(np.ceil(np.log2(brain.n_times)))
brain.apply_hilbert(range(len(brain.ch_names)), envelope=True,
                    n_fft=2**next_pow2)

# Now that we've extracted the amplitude, we'll low-pass filter it to remove noise
brain.filter(None, 20)


# Now take another look at the data
scale = brain._data[0].max()
brain.plot(scalings={'eeg': scale})


# It looks quite different, and a bit cleaner. This is partially because now we've calculated **amplitude**, which goes from 0 and upward. It's never negative. This is because the amplitude of high-gamma activity is reflective of "stuff going on in general", and you can't have negative "stuff going on".
# 
# Te recap, we've:
# 
# 1. Filtered out raw signal between 70 - 150Hz
# 2. Calculated the Hilbert Transform of this signal
# 3. Taken the absolute value of the transform (this returns the amplitude)
# 4. Low-pass filtered the output to clean it up.
# 
# We'll create epochs again and see if this changes our average plots:
# 

tmin, tmax = -.5, 2
epochs = mne.Epochs(brain, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(epochs._data.shape)


# We'll rescale the epochs to show the increase over baseline using a
# "z" score. This subtracts the baseline mean, and divides by baseline
# standard deviation
_ = mne.baseline.rescale(epochs._data, epochs.times, [-.5, 0], 'zscore', copy=False)


# Let's look at the average plots again
epochs.average().plot()


# Interesting - now it seems like, while some channels are still all over the place, there are a few that have an increased amplitude above everything else. If you want to visualize each channel individually, do it like this:
# 

# Use the arrow keys to move around.
# Green line == time 0. Dotted line == epoch start
# See if some electrodes seem to "turn on" in response to the sound.
scale = 10
f = epochs.plot(scalings=scale, n_epochs=10)
f.set_size_inches(8, 5)


# Looking through the data, maybe you note some channels that are more "active" than others. AKA, ones that show a response when the sound begins (green lines are sound onsets). Let's look at one closer to see what it looks like.
# 

# Another way to look at this is with an image.
# Here are the trial activations for one electrode:
# It looks like this channel is responsive
use_chan = 'TG37'
ix_elec = mne.pick_channels(epochs.ch_names, [use_chan])[0]
plt_elec = epochs._data[:, ix_elec, :]

f, ax = plt.subplots()
ax.imshow(plt_elec, aspect='auto', cmap=plt.cm.RdBu_r, vmin=-5, vmax=5)
f


# You can see a clear increase in activity near t== 0 (500 on the xaxis above). So this seems like an electrode that **does** respond to sound.
# 
# I should note, this is one of the really powerful things about electrocorticography recordings. It allows you to see a clear effect on an individual trial. Most other methods in human neuroscience require you to average across many trials before anything clear pops up.
# 

# ## Sound we played
# 

f, ax = plt.subplots()

# This will plot 10 seconds.
ax.imshow(audio._data[:, :10*audio.info['sfreq']],
          aspect='auto', origin='lower', cmap=plt.cm.Reds)


# Those regions of "hot" colors show when the sound is present. Rows near the bottom are slow-moving parts of the sound, and rows near the top are fast-moving parts.
# 

# Maybe the presence of these spectral features can be used to predict the ecog activity. To address this question, we'll use the same regression technique. In this case, our inputs will be the spectrogram above (as X) and the output will be the high-gamma activity in the electrode (as y).
# 
# There is one extra step - we don't expect the brain to respond *immediately* to sounds in the world. That's because it takes time for the signal to get from your ears to your auditory cortex. As such, we'll include "time-lagged" versions of the above spectrogram as extra features in our model:
# 

# First we'll cut up the data so that we don't overload memory
# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * audio.info['sfreq']
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T
einfo = dict(myevent=1)

# Now we'll turn the raw array into epochs shape
tmin, tmax = -.5, 2
epochs_audio = mne.Epochs(audio, ev, einfo, tmin, tmax,
                          baseline=(None, 0), preload=True)

# We'll decimate the data because we've got more datapoints than we need
epochs_audio.decimate(10)


# # Wrapping up
# OK, so what have we learned here? We accomplished the following steps:
# 
# 1. Look at raw brain data
# 1. Look at the raw activity in response to sound
# 1. Filter our brain data to extract high-gamma amplitude
# 1. See that high-gamma has a more reliable response to sound
# 
# That's a very, very tip-of-the-iceberg introduction to fitting encoding models in the brain, which is the flip-side to fitting "decoding" models in the brain. We'll go into more detail on that later.
# 







# So you want to learn more about ecog and predictive models in the brain? Well I don't blame you, they're pretty cool. Whether it be fitting spectro (or spatio) temporal receptive fields, mapping semantic processing with brain activity, or trying to decode the stuff of thought, predictive models are a really powerful tool.
# 
# However, they aren't always the easiest things to learn. This is partially because the concept of a predictive model is actually quite general, so there are bits and pieces scattered throughout the web but nothing systematic, easy to understand, and focused on practicality.
# 
# These are a collection of notebooks that cover the basic processing of ecog data, some common computational techniques for feature extraction, and a bit about fitting models using these features. To begin, here are a few good general resources (with topic-specific resources found in each notebook).
# 

# ** Getting python **
# 
# First things first, you'll need to program in order to make these models work for you. It's possible to do predictive modeling in any language, though my preference is (unsurprisingly) in python. Using a combination of numpy and scikit-learn, you can build all kinds of great modeling tools. And along with the neurophysiology package "MNE", you can also represent brain electrophysiology data quite easily. I'd recommend checking out the [Anaconda python distribution](https://www.continuum.io/downloads), which has most of these packages pre-included.
# 
# **Learning materials**
# 
# A lot of people have written IPython notebooks specifically for tutorials, and they're really useful to learn from. [Check out this page](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks), which has a curated list of some of the better ipython notebooks. In particular, you should check out the "[Python for Scientific Computing](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks#scientific-computing-and-data-analysis-with-the-scipy-stack)" section. It'll be really helpful in both making you more comfortable with python, and with scientific analysis in general.
# 
# **Psychopy**
# 
# Psychopy is really useful for stimulus presentation and running your experiments. I'd recommend using the "Standalone PsychoPy" installation. This basically comes with its own python distribution, and all packages that it needs pre-installed. This makes it much easier to use on lots of computers without worrying about whether you have the right library installed.
# 
# [Here's the main website](http://www.psychopy.org/) for psychopy. In order to get familiar with psychopy, you might check out [this](https://www.youtube.com/watch?v=WKJBbVnQkj0), and [this](https://www.youtube.com/watch?v=VV6qhuQgsiI) youtube video. You can skip around to see how things work. It should be relatively straightforward, but there may be an early learning curve. I've also found that the [psychopy-users](https://groups.google.com/forum/#!forum/psychopy-users) google group is useful as well. 
# 
# **Visualization**
# 
# For some quick visualizations, the [pyplot](http://matplotlib.org/api/pyplot_api.html) module of matplotlib is really useful too. Matplotlib is a general plotting package, and pyplot includes some helper functions to do things quickly.
# 
# There's also [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/), which is prettier than matplotlib, and has some defaults and functions that are more appropriate for scientific visualizations.
# 

# Now take a look at the other notebooks contained here. The first one covers basic data visualization and exploration, and we'll get more complex from there.
# 




# Note - this is a static version of a jupyter notebook. You can find the original here.
# 

# There are a lot of great blogging platforms out there, and I know it's all the rage these days to have your website hosted on github and push new pages directly from your repository. However, for those of us stuck in the stone age using platforms like Wordpress, there hasn't been as much talk about using newfangled tools like the Jupyter (IPython) Notbebook.
# 
# That said, it's relatively simple to convert jupyter notebooks into a format that can be used in something like Wordpress. This is a quick tutorial on how I got it working for this site. So without further ado...
# 
# ## Step 0 - Write content into your notebook
# You can use markdown cells, code cells, whatever. It's also fine if you leave outputs in the notebook as well. The only caveat is that if you're hard-linking to things that are on your computer, then it won't be rendered when you put it on the internet unless you update the links too.
# 
# ## Step 1 - Download the notebook file
# You can download your notebook file directly to your computer by going to File -> Download as -> Ipython Notebook. (note, this might be Jupyter Notebook one day).
# 
# This will download an `.ipynb` file to your computer, which contains 
# all of the information in your original interactive notebook. Importantly - any images, plots, etc will be serialized and store with the notebook itself, which means you can still display them later on.
# 
# ## Step 2 - Convert the notebook to an html file
# For this, `nbconvert` is a magnificent tool. Check out their instructions for using it [here](https://ipython.org/ipython-doc/3/notebook/nbconvert.html).
# 
# To convert a notebook for HTML, run `nbconvert` with the following syntax:
# 
# ```python
# ipython nbconvert --to html --template basic *source_file.ipynb* *target_file.html*
# ```
# 
# Make note of the `basic` command - this strips a lot of the fancy shmancy ipython formatting and just gives you the basic tags etc. This is important because we'll just create our own styles in wordpress, which brings me to...
# 
# ## Step 3 - Add notebook styles to your site CSS file
# There are a few typical tags that nbconvert will output, and you can capture these tags to style notebooks however you like. Go to your custom CSS file in wordpress, and add the following tags to mimic what this site does:
# 
# ```css
# .input_prompt {
# 	color: #8476FF;
# }
# 
# .navbar {
# 	margin-bottom: 20px;
# 	border-radius: 0;
# }
# 
# .c {
# 	color: #989898;
# }
# 
# .k {
# 	color: #338822;
# 	font-weight: bold;
# }
# 
# .kn {
# 	color: #338822;
# 	font-weight: bold;
# }
# 
# .mi {
# 	color: #000000;
# }
# 
# .o {
# 	color: #000000;
# }
# 
# .ow {
# 	color: #BA22FF;
# 	font-weight: bold;
# }
# 
# .nb {
# 	color: #338822;
# }
# 
# .n {
# 	color: #000000;
# }
# 
# .s {
# 	color: #cc2222;
# }
# 
# .se {
# 	color: #cc2222;
# 	font-weight: bold;
# }
# 
# .si {
# 	color: #C06688;
# 	font-weight: bold;
# }
# 
# .nn {
# 	color: #4D00FF;
# 	font-weight: bold;
# }
# 
# .output_area pre {
# 	background-color: #FFFFFF;
# 	padding-left: 5%;
# }
# 
# .site-description {
# 	margin: 10px auto;
# }
# 
# #site-title {
# 	margin-top: 10px;
# }
# 
# .blog-header {
# 	padding-top: 20px;
# 	padding-bottom: 20px;
# }
# 
# .code_cell {
# 	padding-left: 2%;
# }
# 
# .cell {
# 	margin-top: 20px;
# 	margin-bottom: 20px;
# }
# 
# br {
# 	line-height: 2;
# }
# 
# .cell h1 h2 h3 h4 {
# 	margin-top: 30px;
# 	margin-bottom: 10px;
# }
# ```
# 
# ## Step 4 - Turn off `p` and `br` tags in wordpress (maybe)
# I'm not sure why this is, but I had trouble with Wordpress automatically creating these tags when I pasted in my notebook HTML code. You can turn these off by adding the following lines to your `functions.php` file:
# 
# ```php
# (/, Disable, automatic, p, and, br, tags)
# remove_filter( 'the_content', 'wpautop' );
# remove_filter( 'the_excerpt', 'wpautop' );
# ```
# It's possible that this won't be an issue for you, so I'd say only try this if these tags seem to be a problem.
# 
# ## Step 4.1 - Get mathjax working on your wordpress install
# Mathjax is really useful for displaying mathematical equations, notation, etc that are often used in notebooks. Check out the [Automatic mathjax plugin](http://docs.mathjax.org/en/v1.1-latest/platforms/wordpress.html) and see [this stackexchange post](http://tex.stackexchange.com/questions/27633/mathjax-inline-mode-not-rendering) on getting it set up correctly.
# 
# ## Step 5 - Paste in the HTML to a new post
# Now we're ready to create the new post in wordpress. On the post editing page, switch the editor view to 'Text'. Then, open up your newly-created HTML file in a text editor of your choice. Copy the entire thing, then paste it all into the 'Text' window. It'll look like gibberish, but if your CSS is done right, it should all "just work".
# 
# # That's it
# You're ready to post to your blog.
# 
# Note that there are a lot of things you could tweak here (most obviously the CSS that styles your notebooks). This is just a starting point, but I've found it sufficient to create notebook posts that don't look like garbage. I hope you find it useful!
# 
# ## Remember
# Here a few quick 'gotchas' that I ran into. Keep these in mind:
# 
# * If you hard-link to any images or files on your computer, make sure that you upload these to your website and link to them in the code, otherwise the links may break. If you're plotting the images w/ data in the notebook itself, it should serialize the images and won't be a problem.
# * I'd highly recommend keeping a version of the post as a notebook on Github anyway. It's a great way to keep track of your drafts, and the conversion process above is pretty painless in case you need to make changes.
# * If you display images within a text cell, it may appear inline with your text. In that case, just split the cell up into multiple ones, and give the image its own cell so it's displayed correctly.
# * You may want to match your website content container width to roughly the same width that ipython notebooks use to display, this will keep the output more natural-looking
# * *Note - this is largely inspired by a post I found [here](http://prooffreaderplus.blogspot.com/2014/11/how-to-quickly-turn-ipython-notebook.html). Check it out for more ideas*
# 

# ## Overview
# In this notebook, I'll show you how to make a simple query on Craigslist using some nifty python modules. You can take advantage of all the structure data that exists on webpages to collect interesting datasets.
# 

import pandas as pd
get_ipython().magic('pylab inline')


# First we need to figure out how to submit a query to Craigslist. As with many websites, one way you can do this is simply by constructing the proper URL and sending it to Craigslist. Here's a sample URL that is returned after manually typing in a search to Craigslist:
# > `http://sfbay.craigslist.org/search/eby/apa?bedrooms=1&pets_cat=1&pets_dog=1&is_furnished=1`
# 
# This is actually two separate things. The first tells craigslist what kind of thing we're searching for:
# 
# > `http://sfbay.craigslist.org/search/eby/apa` says we're searching in the sfbay area (`sfbay`) for apartments (`apa`) in the east bay (`eby`).
# 
# The second part contains the parameters that we pass to the search:
# 
# > `?bedrooms=1&pets_cat=1&pets_dog=1&is_furnished=1` says we want 1+ bedrooms, cats allowed, dogs allowed, and furnished apartments. You can manually change these fields in order to create new queries.
# 
# ## Getting a single posting
# 
# So, we'll use this knowledge to send some custom URLs to Craigslist. We'll do this using the `requests` python module, which is really useful for querying websites.
# 

import requests


# In internet lingo, we're posting a `get` requests to the website, which simply says that we'd like to get some information from the Craigslist website.  With requests, we can easily create a dictionary that specifies parameters in the URL:
# 

url_base = 'http://sfbay.craigslist.org/search/eby/apa'
params = dict(bedrooms=1, is_furnished=1)
rsp = requests.get(url_base, params=params)


# Note that requests automatically created the right URL:
print(rsp.url)


# We can access the content of the response that Craigslist sent back here:
print(rsp.text[:500])


# Wow, that's a lot of code. Remember, websites serve HTML documents, and usually your browser will automatically render this into a nice webpage for you. Since we're doing this with python, we get back the raw text. This is really useful, but how can we possibly parse it all?
# 
# For this, we'll turn to another great package, BeautifulSoup:

from bs4 import BeautifulSoup as bs4

# BS4 can quickly parse our text, make sure to tell it that you're giving html
html = bs4(rsp.text, 'html.parser')

# BS makes it easy to look through a document
print(html.prettify()[:1000])


# Beautiful soup lets us quickly search through an HTML document. We can pull out whatever information we want.
# 
# Scanning through this text, we see a common structure repeated `<p class="row">`. This seems to be the container that has information for a single apartment.
# 
# In BeautifulSoup, we can quickly get all instances of this container:
# 

# find_all will pull entries that fit your search criteria.
# Note that we have to use brackets to define the `attrs` dictionary
# Because "class" is a special word in python, so we need to give a string.
apts = html.find_all('p', attrs={'class': 'row'})
print(len(apts))


# Now let's look inside the values of a single apartment listing:
# 

# We can see that there's a consistent structure to a listing.
# There is a 'time', a 'name', a 'housing' field with size/n_brs, etc.
this_appt = apts[15]
print(this_appt.prettify())


# So now we'll pull out a couple of things we might be interested in:
# It looks like "housing" contains size information. We'll pull that.
# Note that `findAll` returns a list, since there's only one entry in
# this HTML, we'll just pull the first item.
size = this_appt.findAll(attrs={'class': 'housing'})[0].text
print(size)


# We can query split this into n_bedrooms and the size. However, note that sometimes one of these features might be missing. So we'll use an `if` statement to try and capture this variability:
# 

def find_size_and_brs(size):
    split = size.strip('/- ').split(' - ')
    if len(split) == 2:
        n_brs = split[0].replace('br', '')
        this_size = split[1].replace('ft2', '')
    elif 'br' in split[0]:
        # It's the n_bedrooms
        n_brs = split[0].replace('br', '')
        this_size = np.nan
    elif 'ft2' in split[0]:
        # It's the size
        this_size = split[0].replace('ft2', '')
        n_brs = np.nan
    return float(this_size), float(n_brs)
this_size, n_brs = find_size_and_brs(size)


# Now we'll also pull a few other things:
this_time = this_appt.find('time')['datetime']
this_time = pd.to_datetime(this_time)
this_price = float(this_appt.find('span', {'class': 'price'}).text.strip('$'))
this_title = this_appt.find('a', attrs={'class': 'hdrlnk'}).text


# Now we've got the n_bedrooms, size, price, and time of listing
print('\n'.join([str(i) for i in [this_size, n_brs, this_time, this_price, this_title]]))


# ## Querying lots of postings
# 
# Cool - so now we've got some useful information about one listing. Now let's loop through many listings across several locations.
# 
# It looks like there is a "city code" that distinguishes where you're searching. Here is a **not** up to date list: [link](https://sites.google.com/site/clsiteinfo/city-site-code-sort)
# 
# Within the Bay Area, there are also a lot of sub-regional locations, which we'll define here, then loop through them all.
# 
# Note that the `s` parameter tells Craiglist where to start in terms of the number of results given back. E.g., if s==100, then it starts at the 100th entry.
# 

loc_prefixes = ['eby', 'nby', 'sfc', 'sby', 'scz']


# We'll define a few helper functions to handle edge cases and make sure that we don't get any errors.
# 

def find_prices(results):
    prices = []
    for rw in results:
        price = rw.find('span', {'class': 'price'})
        if price is not None:
            price = float(price.text.strip('$'))
        else:
            price = np.nan
        prices.append(price)
    return prices

def find_times(results):
    times = []
    for rw in apts:
        if time is not None:
            time = time['datetime']
            time = pd.to_datetime(time)
        else:
            time = np.nan
        times.append(time)
    return times


# Now we're ready to go. We'll loop through all of our locations, and pull a number of entries for each one. We'll use a pandas dataframe to store everything, because this will be useful for future analysis.
# 
# **Note** - Craigslist won't take kindly to you querying their server a bunch of times at once. Make sure not to pull too much data too quickly. Another option is to add a delay to each loop iteration. Otherwise your IP might get banned.
# 

# Now loop through all of this and store the results
results = []  # We'll store the data here
# Careful with this...too many queries == your IP gets banned temporarily
search_indices = np.arange(0, 300, 100)
for loc in loc_prefixes:
    print loc
    for i in search_indices:
        url = 'http://sfbay.craigslist.org/search/{0}/apa'.format(loc)
        resp = requests.get(url, params={'bedrooms': 1, 's': i})
        txt = bs4(resp.text, 'html.parser')
        apts = txt.findAll(attrs={'class': "row"})
        
        # Find the size of all entries
        size_text = [rw.findAll(attrs={'class': 'housing'})[0].text
                     for rw in apts]
        sizes_brs = [find_size_and_brs(stxt) for stxt in size_text]
        sizes, n_brs = zip(*sizes_brs)  # This unzips into 2 vectors
     
        # Find the title and link
        title = [rw.find('a', attrs={'class': 'hdrlnk'}).text
                      for rw in apts]
        links = [rw.find('a', attrs={'class': 'hdrlnk'})['href']
                 for rw in apts]
        
        # Find the time
        time = [pd.to_datetime(rw.find('time')['datetime']) for rw in apts]
        price = find_prices(apts)
        
        # We'll create a dataframe to store all the data
        data = np.array([time, price, sizes, n_brs, title, links])
        col_names = ['time', 'price', 'size', 'brs', 'title', 'link']
        df = pd.DataFrame(data.T, columns=col_names)
        df = df.set_index('time')
        
        # Add the location variable to all entries
        df['loc'] = loc
        results.append(df)
        
# Finally, concatenate all the results
results = pd.concat(results, axis=0)


# We'll make sure that the right columns are represented numerically:
results[['price', 'size', 'brs']] = results[['price', 'size', 'brs']].convert_objects(convert_numeric=True)


# And there you have it:
results.head()


ax = results.hist('price', bins=np.arange(0, 10000, 100))[0, 0]
ax.set_title('Mother of god.', fontsize=20)
ax.set_xlabel('Price', fontsize=18)
ax.set_ylabel('Count', fontsize=18)


# Finally, we can save this data to a CSV to play around with it later.
# We'll have to remove some annoying characters first:
import string
use_chars = string.ascii_letters +    ''.join([str(i) for i in range(10)]) +    ' /\.'
results['title'] = results['title'].apply(
    lambda a: ''.join([i for i in a if i in use_chars]))

results.to_csv('../data/craigslist_results.csv')


# ## RECAP
# To sum up what we just did:
# 
# * We defined the ability to query a website using a custom URL. This is usually the same in structure for website, but the parameter names will be different.
# * We sent a `get` request to Craigslist using the `requests` module of python.
# * We parsed the response using `BeautifulSoup4`.
# * We then looped through a bunch of apartment listings, pulled some relevant data, and combined it all into a cleaned and usable dataframe with `pandas`.
# 
# Next up I'll take a look at the data, and see if there's anything interesting to make of it.
# 

# ## Bonus - auto-emailing yourself w/ notifications
# A few people have asked me about using this kind of process to make a bot that scrapes craigslist periodically. This is actually quite simple, as it basically involves pulling the top listings from craigslist, checking this against an "old" list, and detecting if there's anything new that has popped up since the last time you checked.
# 
# Here's a simple script that will get the job done. Once again, don't pull too much data at once, and don't query Craigslist too frequently, or you're gonna get banned.
# 

# We'll use the gmail module (there really is a module for everything in python)
import gmail
import time


gm = gmail.GMail('my_username', 'my_password')
gm.connect()

# Define our URL and a query we want to post
base_url = 'http://sfbay.craigslist.org/'
url = base_url + 'search/eby/apa?nh=48&anh=49&nh=112&nh=58&nh=61&nh=62&nh=66&max_price=2200&bedrooms=1'

# This will remove weird characters that people put in titles like ****!***!!!
use_chars = string.ascii_letters + ''.join([str(i) for i in range(10)]) + ' '


link_list = []  # We'll store the data here
link_list_send = []  # This is a list of links to be sent
send_list = []  # This is what will actually be sent in the email

# Careful with this...too many queries == your IP gets banned temporarily
while True:
    resp = requests.get(url)
    txt = bs4(resp.text, 'html.parser')
    apts = txt.findAll(attrs={'class': "row"})
    
    # We're just going to pull the title and link
    for apt in apts:
        title = apt.find_all('a', attrs={'class': 'hdrlnk'})[0]
        name = ''.join([i for i in title.text if i in use_chars])
        link = title.attrs['href']
        if link not in link_list and link not in link_list_send:
            print('Found new listing')
            link_list_send.append(link)
            send_list.append(name + '  -  ' + base_url+link)
            
    # Flush the cache if we've found new entries
    if len(link_list_send) > 0:
        print('Sending mail!')
        msg = '\n'.join(send_list)
        m = email.message.Message()
        m.set_payload(msg)
        gm.send(m, ['recipient_email@mydomain.com'])
        link_list += link_list_send
        link_list_send = []
        send_list = []
    
    # Sleep a bit so CL doesn't ban us
    sleep_amt = np.random.randint(60, 120)
    time.sleep(sleep_amt)


# And there you have it - your own little bot to keep you on the top of the rental market.
# 

# When we discuss "computational efficiency", you often hear people throw around phrases like $O(n^2)$ or $O(nlogn)$. We talk about them in the abstract, and it can be hard to appreciate what these distinctions mean and how important they are. So let's take a quick look at what computational efficiency looks like in the context of a very famous algorithm: The Fourier Transform.
# 
# ## A short primer on the Fourier Transform
# Briefly, A Fourier Transform is used for uncovering the spectral information that is present in a signal. AKA, it tells us about oscillatory components in the signal, and has [a wide range](http://dsp.stackexchange.com/questions/69/why-is-the-fourier-transform-so-important) of uses in communications, signal processing, and even neuroscience analysis.
# 
# Here's a [Quora post](https://www.quora.com/What-is-an-intuitive-way-of-explaining-how-the-Fourier-transform-works) that discusses Fourier Transforms more generally. The first explanation is fantastic and full of history and detail.
# 
# The challenge with the Fourier Transform is that it can take a really long time to compute. If you h ave a signal of length $n$, then you're calculating $n$ Fourier components for each point in the (length $n$) signal. This means that the number of operations required to calculate a fourier transform is $n * n$ or $O(n^2)$.
# 
# For a quick intuition into what a difference this makes. Consider two signals, one of length 10, and the other of length 100. Since the Fourier Transform is $O(n^2)$, the length 100 signal will take *2 orders of magnitude* longer to compute, even though it is only *1 order of magnitude longer in length*.
# 
# Think this isn't a big deal? Let's see what happens when the signal gets longer. First off, a very short signal:
# 

# We can use the `time` and the `numpy` module to time how long it takes to do an FFT
from time import time
import numpy as np
import seaborn as sns
sns.set_style('white')

# For a signal of length ~1000. Say, 100ms of a 10KHz audio sample.
signal = np.random.randn(1009)
start = time()
_ = np.fft.fft(signal)
stop = time()
print('It takes {} seconds to do the FFT'.format(stop-start))


# That's not too bad - ~.003 seconds is pretty fast. But here's where the $O(n^2)$ thing really gets us...
# 

# We'll test out how long the FFT takes for a few lengths
test_primes = [11, 101, 1009, 10009, 100019]


# Let's try a few slightly longer signals
for i_length in test_primes:
    # Calculate the number of factors for this length (we'll see why later)
    factors = [ii for ii in range(1, 1000) if i_length % ii == 0]
    # Generate a random signal w/ this length
    signal = np.random.randn(i_length)
    # Now time the FFT
    start = time()
    _ = np.fft.fft(signal)
    stop = time()
    print('With data of length {} ({} factors), it takes {} seconds to do the FFT'.format(
            i_length, len(factors), stop-start))


# Whoah wait a sec, that last one took way longer than everything else. We increased the length of the data by a factor of 10, but the time it took went up by a factor of 100. Not good. That means that if we want to perform an FFT on a signal that was 10 times longer, it'd take us about 42 minutes. 100 times longer? That'd take *~3 days.*
# 
# Given how important the Fourier Transform is, it'd be great if we could speed it up somehow. 
# 
# > *You'll notice that I chose a very particular set of numbers above. Specifically, I chose numbers that were primes (or nearly primes) meaning that they couldn't be broken down into products of smaller numbers. That turns out to be really important in allowing the FFT to do its magic. When your signal length is a prime number, then you don't gain any speedup from the FFT, as I'll show below.*
# 
# ## Enter the Fast Fourier Transform
# The Fast Fourier Transform (FFT) is one of the most important algorithms to come out of the last century because it drastically speeds up the performance of the Fourier Transform. It accomplishes this by breaking down all those $n^2$ computations into a smaller number of computations, and then putting them together at the end to get the same result. This is called **factorizing**.
# 
# You can think of factorizing like trying to move a bunch of groceries from your car to your fridge. Say you have 20 items in your car. One way to do this is to individually take each item, pull it from the car, walk to the house, place it in the fridge. It'd take you 20 trips to do this. Factorizing is like putting your 20 items into 2 grocery bags. Now you only need to make 2 trips to the house - one for each grocery bag. The first approach requires 20 trips to the house, and the second requires 2 trips. You've just reduced the number of trips by an order of magnitude!
# 
# The FFT accomplishes its factorization by recognizing that signals of a certain length can be broken down (factorized) into smaller signals. How many smaller signals? Well, that depends on the length of the original signal. If a number has many *factors*, it means that it can be broken down into a product of many different, smaller, signals.
# 
# In practice, this means that if the input to an FFT has a lot of factors, then you gain a bigger speedup from the FFT algorithm. On one end, a signal with a length == a power of two will have a ton of factors, and yield the greatest speedups. A signal with length == a prime number will be the slowest because it has no factors. Below is a quick simulation to see how much of a difference this makes.
# 
# > Here are some useful links explaining Fourier Transforms, as well as the FFT:
# > * [A Quora post](https://www.quora.com/What-is-an-intuitive-explanation-of-the-FFT-algorithm) with some great answers on the intuition behind the Fast Fourier Transform.
# > * [The wikipedia entry](https://en.wikipedia.org/wiki/Fast_Fourier_transform) for FFTs also has some nice links.
# > * [A post on the FFT](https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/) from Jake Vanderplas is also a great explanation of how it works.
# 

import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# ## The beautiful efficiency of the FFT
# To see what the FFT's efficiency looks like, we'll simulate data of different lengths and see how long it takes to compute the FFT at each length. We'll create a random vector of gaussian noise ranging from length 1 to 10,000. For each vector, we'll compute the FFT, and time how long it took to compute. I've already taken the liberty of doing this (repeated 3 times, and then averaged together). Those times are stored in fft_times.csv.
# 

# Let's read in the data and see how it looks
df = pd.read_csv('../data/fft_times.csv', index_col=0)
df = df.apply(pd.to_numeric)
df = df.mean(0).to_frame('time')
df.index = df.index.astype(int)
df['length'] = df.index.values


# First off, it's clear that computation time grows nonlinearly with signal length
df.plot('length', 'time', figsize=(10, 5))


# However, upon closer inspection, it's clear that there's much variability
winsize = 500
i = 0
j = i + winsize
df.iloc[i:j]['time'].plot(figsize=(10, 5))


# As you can see, there appear to be multiple trends in the data. There seems to be a "most inefficient" line of growth in the data, as well as a "more efficient" and a "most efficient" trend. These correspond to lengths that are particularly good for an FFT.
# 
# We can use regression to find the "linear" relationship between length of signal and time of FFT. However, if there are any trends in the data that are **nonlinear**, then they should show up as errors in the regression model. Let's see if that happens...
# 

# We'll use a regression model to try and fit how length predicts time
mod = linear_model.LinearRegression()
xfit = df['length']
xfit = np.vstack([xfit, xfit**2, xfit**3, xfit**4]).T
yfit = df['time'].reshape([-1, 1])


# Now fit to our data, and calculate the error for each datapoint
mod.fit(xfit, yfit)
df['ypred'] = mod.predict(xfit)
df['diff'] = df['time'] - df.ypred


# As the length grows, the trends in the data begin to diverge more and more
ax = df.plot('length', 'diff', kind='scatter',
             style='.', alpha=.5, figsize=(10, 5))
ax.set_ylim([0, .05])
ax.set_title('Error of linear fit for varying signal lengths')


# It looks like there are some clear components of the data that *don't* follow a linear relationship. Moreover, this seems to be systematic. We clearly see several separate traces in the error plot, which means that there are patterns in the data that follow different non-linear trends.
# 
# But we already know that the FFT efficiency will differ depending on the number of factors of the signal's length. Let's see if that's related to the plot above...
# 

# We'll write a helper function that shows how many (<100) factors each length has
find_n_factors = lambda n: len([i for i in range(1, min(100, n-1)) if n % i == 0])

# This tells us the number of factors for all lengths we tried
df['n_factors'] = df['length'].map(find_n_factors)

# We now have a column that tells us how many factors each iteration had
df.tail()


# Finally, we can plot time to compue the FFT as a function of the number of factors for that signal length.
# 

# As we can see, the FFT time drops quickly as a function of the number of factors
ax = df.plot('n_factors', 'time', style=['.'], figsize=(10, 5), alpha=.1)
ax.set_xlim([0, 15])
ax.set_ylabel('Time for FFT (s)')
ax.set_title('Time of FFT for varying numbers of factors')


# The fewer factors in the length of the signal, the longer the FFT takes.
# 
# Finally, we can show how the length of computation time changes for each group of factors. We'll plot the signal length along with the time to compute the FFT, this time colored by the number of factors for each point.
# 

# We'll plot two zoom levels to see the detail
f, axs = plt.subplots(2, 1, figsize=(10, 5))
vmin, vmax = 1, 18
for ax in axs:
    ax = df.plot.scatter('length', 'time', c='n_factors', lw=0, cmap=plt.cm.get_cmap('RdYlBu', vmax),
                                           figsize=(10, 10), vmin=vmin, vmax=vmax, ax=ax, alpha=.5)
    ax.set_xlabel('Length of signal (samples)')
    ax.set_ylabel('Time to complete FFT (s)')
    ax.set_title('Time to compute the FFT, colored by n_factors')
_ = plt.setp(axs, xlim=[0, df['length'].max()])
_ = plt.setp(axs[0], ylim=[0, .2])
_ = plt.setp(axs[1], ylim=[0, .005])
plt.tight_layout()


# Each of those colored traces spreading upwards represents a particular strategy that the FFT uses for that number of factors. As you can see, the FFT will take a lot longer (and scales exponentially) with fewer factors (see the red lines). It takes much less time (and scales more linearly) with more factors (see the blue lines).
# 
# And that right there is the beauty of methods like the FFT. They leverage the structure of mathematics to take a computation that goes on for days, and figure out how to do it in seconds.
# 

