# # Harmonic-percussive-residual source separation
# 
# 

from __future__ import print_function


import librosa
import IPython.display
import numpy as np


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Load the example track
#y, sr = librosa.load('audio/Kevin_MacLeod_-_Camille_Saint-Sans_Danse_Macabre_-_Sad_Part.mp3')
#y, sr = librosa.load('audio/Karissa_Hobbs_-_09_-_Lets_Go_Fishin.mp3', offset=40, duration=10)
y, sr = librosa.load('/home/bmcfee/data/CAL500/mp3/james_taylor-fire_and_rain.mp3', duration=15)


D = librosa.stft(y)


# Decompose D into harmonic and percussive components
D_harmonic, D_percussive = librosa.decompose.hpss(D)


# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))**2

plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
librosa.display.specshow(librosa.logamplitude(D**2, ref_power=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')
plt.subplot(3,1,2)
librosa.display.specshow(librosa.logamplitude(D_harmonic**2, ref_power=rp), y_axis='log')
plt.colorbar()
plt.title('Harmonic spectrogram')
plt.subplot(3,1,3)
librosa.display.specshow(librosa.logamplitude(D_percussive**2, ref_power=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')
plt.tight_layout()


# We can make this stricter by using a larger margins.
# The default (above) corresponds to margin=1

D_harmonic2, D_percussive2 = librosa.decompose.hpss(D, margin=2)
D_harmonic4, D_percussive4 = librosa.decompose.hpss(D, margin=4)
D_harmonic8, D_percussive8 = librosa.decompose.hpss(D, margin=8)
D_harmonic16, D_percussive16 = librosa.decompose.hpss(D, margin=16)


plt.figure(figsize=(10, 10))

plt.subplot(5,2,1)
librosa.display.specshow(librosa.logamplitude(D_harmonic**2, ref_power=rp), y_axis='log')
plt.title('Harmonic')
plt.yticks([])
plt.ylabel('margin=1')

plt.subplot(5,2,2)
librosa.display.specshow(librosa.logamplitude(D_percussive**2, ref_power=rp), y_axis='log')
plt.title('Percussive')
plt.yticks([]) ,plt.ylabel('')

plt.subplot(5,2,3)
librosa.display.specshow(librosa.logamplitude(D_harmonic2**2, ref_power=rp), y_axis='log')
plt.yticks([])
plt.ylabel('margin=2')

plt.subplot(5,2,4)
librosa.display.specshow(librosa.logamplitude(D_percussive2**2, ref_power=rp), y_axis='log')
plt.yticks([]) ,plt.ylabel('')

plt.subplot(5,2,5)
librosa.display.specshow(librosa.logamplitude(D_harmonic4**2, ref_power=rp), y_axis='log')
plt.yticks([])
plt.ylabel('margin=4')

plt.subplot(5,2,6)
librosa.display.specshow(librosa.logamplitude(D_percussive4**2, ref_power=rp), y_axis='log')
plt.yticks([]) ,plt.ylabel('')

plt.subplot(5,2,7)
librosa.display.specshow(librosa.logamplitude(D_harmonic8**2, ref_power=rp), y_axis='log')
plt.yticks([])
plt.ylabel('margin=8')

plt.subplot(5,2,8)
librosa.display.specshow(librosa.logamplitude(D_percussive8**2, ref_power=rp), y_axis='log')
plt.yticks([]) ,plt.ylabel('')

plt.subplot(5,2,9)
librosa.display.specshow(librosa.logamplitude(D_harmonic16**2, ref_power=rp), y_axis='log')
plt.yticks([])
plt.ylabel('margin=16')

plt.subplot(5,2,10)
librosa.display.specshow(librosa.logamplitude(D_percussive16**2, ref_power=rp), y_axis='log')
plt.yticks([]) ,plt.ylabel('')

plt.tight_layout()


from IPython.display import Audio


Audio(data=y, rate=sr)


Audio(data=librosa.istft(D_harmonic), rate=sr)


Audio(data=librosa.istft(D_harmonic2), rate=sr)


Audio(data=librosa.istft(D_harmonic4), rate=sr)


Audio(data=librosa.istft(D_harmonic8), rate=sr)


Audio(data=librosa.istft(D_percussive), rate=sr)


Audio(data=librosa.istft(D_percussive2), rate=sr)


Audio(data=librosa.istft(D_percussive4), rate=sr)


Audio(data=librosa.istft(D_percussive8), rate=sr)


# 
# [Holzapfel and Stylianou](https://www.ics.forth.gr/netlab/data/J13.pdf)
# 

from __future__ import print_function


import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from IPython.display import Audio


y, sr = librosa.load('audio/BFJazz_-_Stargazer_from_Texas.mp3', sr=44100)
#y, sr = librosa.load('/home/bmcfee/working/Battles - Tonto-it1CCNCHPc0.mp3', sr=44100, duration=240, offset=240)


# Compute a log-CQT for visualization
C = librosa.logamplitude(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3, real=False)**2, ref_power=np.max)


# We'll use a superflux-style onset strength envelope
oenv = librosa.onset.onset_strength(y=y, sr=sr, lag=2, max_size=5)


# Get the tempogram
tgram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, win_length=512)


Audio(data=y, rate=sr)


plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
librosa.display.specshow(C, y_axis='cqt_hz', bins_per_octave=12*3, sr=sr)

plt.subplot(2,1,2)
librosa.display.specshow(tgram[:256], y_axis='tempo', x_axis='time', sr=sr)

plt.tight_layout()


# The above figure reveals that the track alternates between sections in 6/8 and 4/4.
# 

# Let's beat-synchronize to reduce dimensionality
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, trim=False)


# Let's plot the "average" onset autocorrelation on log-lag axis
# We use the median to suppress outliers / breaks
# We skip over the lag=0 point, since it corresponds to infinite tempo

# Compute the inter-quartile range for each lag position
tlb = np.percentile(tgram[1:], 25, axis=1)
tub = np.percentile(tgram[1:], 75, axis=1)

plt.figure(figsize=(8, 4))
plt.semilogx(librosa.tempo_frequencies(len(tgram))[1:], np.median(tgram[1:], axis=1),
             label='Median onset autocorrelation', basex=2)

plt.fill_between(librosa.tempo_frequencies(len(tgram))[1:], tlb, tub, alpha=0.25,
                 label='Inter-quartile range')

plt.axvline(tempo, color='r', label='Tempo={:.1f} BPM'.format(tempo))
plt.xlabel('Tempo (BPM)')
plt.axis('tight')

plt.grid()
plt.legend(loc='upper right')
plt.tight_layout()


# We can clean up some bleed by a vertical median filter
tgram_clean = np.maximum(0.0, tgram - scipy.ndimage.median_filter(tgram, size=(15, 1)))


plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
librosa.display.specshow(tgram[:256], y_axis='tempo', sr=sr)

plt.subplot(2,1,2)
librosa.display.specshow(tgram_clean[:256], y_axis='tempo', x_axis='time', sr=sr)
plt.tight_layout()


tst = librosa.fmt(tgram_clean, axis=0)


plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
librosa.display.specshow(tgram_clean[:256], y_axis='tempo', sr=sr)
plt.title('Tempogram')
plt.subplot(3,1,2)
librosa.display.specshow(librosa.logamplitude(tst**2, ref_power=np.max)[:16],  n_xticks=12, sr=sr)
plt.title('Scale transform magnitude')
plt.subplot(3,1,3)
librosa.display.specshow(np.angle(tst)[:16], x_axis='time', n_xticks=12, sr=sr, cmap='hsv')
plt.title('Scale transform phase')
plt.tight_layout()


# Examining the first 16 components of the scale transform, some discontinuties show up at around 1:17, 2:10, and 3:30, corresponding to the time signature changes in the track.
# 

tgram_sync = librosa.util.sync(tgram_clean, beats)[1:]
tst_sync = librosa.util.sync(np.abs(tst), beats)[:32]


# And plot a distance matrix for each feature.
Rtgram = scipy.spatial.distance.cdist(tgram_sync.T, tgram_sync.T,
                                      metric='seuclidean', V=1e-3 + np.std(tgram_sync, axis=1))

Rtst = scipy.spatial.distance.cdist(tst_sync.T, tst_sync.T, metric='seuclidean')


plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
librosa.display.specshow(Rtgram, cmap='magma_r', aspect='equal')
plt.title('Tempogram distance')
plt.colorbar()

plt.subplot(1,2,2)
librosa.display.specshow(Rtst, cmap='magma_r', aspect='equal')
plt.title('Scale transform distance')
plt.colorbar()
plt.tight_layout()





# # Audio effects and playback with Librosa and IPython Notebook
# 
# This notebook will demonstrate how to do audio effects processing with librosa and IPython notebook.  You will need IPython 2.0 or later.
# 
# By the end of this notebook, you'll know how to do the following:
# 
#   - Play audio in the browser
#   - Effect transformations such as harmonic/percussive source separation, time stretching, and pitch shifting
#   - Decompose and reconstruct audio signals with non-negative matrix factorization
#   - Visualize spectrogram data
# 

from __future__ import print_function


import librosa
import IPython.display
import numpy as np


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Load the example track
y, sr = librosa.load(librosa.util.example_audio_file())


# Play it back!
IPython.display.Audio(data=y, rate=sr)


# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)


# Play the harmonic component
IPython.display.Audio(data=y_h, rate=sr)


# Play the percussive component
IPython.display.Audio(data=y_p, rate=sr)


# Pitch shifting?  Let's gear-shift by a major third (4 semitones)
y_shift = librosa.effects.pitch_shift(y, sr, 7)

IPython.display.Audio(data=y_shift, rate=sr)


# Or time-stretching?  Let's slow it down
y_slow = librosa.effects.time_stretch(y, 0.5)

IPython.display.Audio(data=y_slow, rate=sr)


# How about something more advanced?  Let's decompose a spectrogram with NMF, and then resynthesize an individual component
D = librosa.stft(y)

# Separate the magnitude and phase
S, phase = librosa.magphase(D)

# Decompose by nmf
components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)


# Visualize the components and activations, just for fun

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(librosa.logamplitude(components**2.0, ref_power=np.max), y_axis='log')
plt.xlabel('Component')
plt.ylabel('Frequency')
plt.title('Components')

plt.subplot(1,2,2)
librosa.display.specshow(activations)
plt.xlabel('Time')
plt.ylabel('Component')
plt.title('Activations')

plt.tight_layout()


print(components.shape, activations.shape)


# Play back the reconstruction
# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = components.dot(activations)

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Full reconstruction')

IPython.display.Audio(data=y_k, rate=sr)


# Resynthesize.  How about we isolate just first (lowest) component?
k = 0

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)


# Resynthesize.  How about we isolate a middle-frequency component?
k = len(activations) / 2

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)


# Resynthesize.  How about we isolate just last (highest) component?
k = -1

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)


# # Superflux onsets
# 
# This notebook demonstrates how to recover the Superflux onset detection algorithm of [Boeck and Widmer, 2013](http://dafx13.nuim.ie/papers/09.dafx2013_submission_12.pdf) from librosa.  This algorithm improves onset detection accuracy in the presence of vibrato.
# 

from __future__ import print_function


import librosa
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


y, sr = librosa.load('audio/Karissa_Hobbs_-_09_-_Lets_Go_Fishin.mp3', sr=44100, duration=2, offset=35)


# Parameters from the paper:

n_fft = 1024
hop_length = int(librosa.time_to_samples(1./200, sr=sr))
lag = 2
n_mels = 138
fmin = 27.5
fmax = 16000.
max_size = 3


hop_length


S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)


plt.figure(figsize=(6, 4))
librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max),
                         y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
plt.tight_layout()


odf_default = librosa.onset.onset_strength(y=y, sr=sr)
onset_default = librosa.onset.onset_detect(y=y, sr=sr)


odf_sf = librosa.onset.onset_strength(S=librosa.logamplitude(S), sr=sr, hop_length=hop_length,
                                      lag=lag, max_size=max_size)

onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf, sr=sr, hop_length=hop_length)


plt.figure(figsize=(6, 6))

plt.subplot(2,1,2)
librosa.display.specshow(librosa.logamplitude(S, top_db=50, ref_power=np.max),
                         y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
                         n_xticks=9)

plt.subplot(4,1,1)
plt.plot(odf_default, label='Spectral flux')
plt.vlines(onset_default, 0, odf_default.max(), color='r', label='Onsets')
plt.yticks([])
plt.xticks([])
plt.axis('tight')
plt.legend()


plt.subplot(4,1,2)
plt.plot(odf_sf, color='g', label='Superflux')
plt.vlines(onset_sf, 0, odf_sf.max(), color='r', label='Onsets')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.axis('tight')

plt.tight_layout()


# # Comments
# 
# If you look carefully, the default onset detector (top sub-plot) has several false positives in high-vibrato regions, eg around 0.62s or 1.80s. 
# 
# The superflux method (middle plot) is less susceptible to vibrato, and does not detect onset events at those points.
# 

# # Presets
# 
# This notebook illustrates how to use the [presets](https://pypi.python.org/pypi/presets) package to globally override the default parameter settings of `librosa`.
# 
# `Presets` lets you specify default parameter values by a global dictionary interface.  Note that this interface is tied to the variable name, and not to any particular function, so some care must be taken not to override variable names that mean different things in different functions.
# 

from __future__ import print_function


# Note the _ prefix: we use this to alias the unmodified module
import librosa as _librosa

from presets import Preset

# The Preset object wraps a module (and its submodules) with a dictionary interface
librosa = Preset(_librosa)


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# We can now change some default settings via the dictionary

# Use 44.1KHz as the default sampling rate
librosa['sr'] = 44100

# Change the FFT parameters
librosa['n_fft'] = 4096
librosa['hop_length'] = librosa['n_fft'] // 4


y, sr = librosa.load(librosa.util.example_audio_file())


print(sr)


# Voila! The default sampling rate works.
# 

S = librosa.stft(y)


print(S.shape)


print(librosa.get_duration(S=S), librosa.get_duration(y=y, sr=sr))


plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), x_axis='time', y_axis='log')
plt.tight_layout();


# Presets can be explicitly overridden just like any other default value:

S = librosa.stft(y, hop_length=2048)


print(S.shape)


plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), x_axis='time', hop_length=2048, y_axis='log')
plt.tight_layout();


# # Librosa demo
# 
# This notebook demonstrates some of the basic functionality of librosa version 0.4.
# 
# Following through this example, you'll learn how to:
# 
# * Load audio input
# * Compute mel spectrogram, MFCC, delta features, chroma
# * Locate beat events
# * Compute beat-synchronous features
# * Display features
# 

from __future__ import print_function


# We'll need numpy for some mathematical operations
import numpy as np

# Librosa for audio
import librosa

# matplotlib for displaying the output
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# and IPython.display for audio output
import IPython.display


audio_path = librosa.util.example_audio_file()

# or uncomment the line below and point it at your favorite song:
#
# audio_path = '/path/to/your/favorite/song.mp3'

y, sr = librosa.load(audio_path)


# By default, librosa will resample the signal to 22050Hz.
# 
# You can change this behavior by saying:
# ```
# librosa.load(audio_path, sr=44100)
# ```
# to resample at 44.1KHz, or
# ```
# librosa.load(audio_path, sr=None)
# ```
# to disable resampling.
# 

# # Mel spectrogram
# This first step will show how to compute a [Mel](http://en.wikipedia.org/wiki/Mel_scale) spectrogram from an audio waveform.
# 

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()


# # Harmonic-percussive source separation
# 
# Before doing any signal analysis, let's pull apart the harmonic and percussive components of the audio.  This is pretty easy to do with the `effects` module.
# 

y_harmonic, y_percussive = librosa.effects.hpss(y)


# What do the spectrograms look like?
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

# Convert to log scale (dB). We'll use the peak power as reference.
log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
# Display the spectrogram on a mel scale
librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Harmonic)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

plt.subplot(2,1,2)
librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Percussive)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()


# # Chromagram
# 
# Next, we'll extract [Chroma](http://en.wikipedia.org/wiki/Pitch_class) features to represent pitch class information.
# 

# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax
librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')
plt.colorbar()

plt.tight_layout()


# # MFCC
# 
# [Mel-frequency cepstral coefficients](http://en.wikipedia.org/wiki/Mel-frequency_cepstrum) are commonly used to represent texture or timbre of sound.
# 

# Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# Let's pad on the first and second deltas while we're at it
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# How do they look?  We'll show each in its own subplot
plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

plt.subplot(3,1,2)
librosa.display.specshow(delta_mfcc)
plt.ylabel('MFCC-$\Delta$')
plt.colorbar()

plt.subplot(3,1,3)
librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
plt.ylabel('MFCC-$\Delta^2$')
plt.colorbar()

plt.tight_layout()

# For future use, we'll stack these together into one matrix
M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])


# # Beat tracking
# 
# The beat tracker returns an estimate of the tempo (in beats per minute) and frame indices of beat events.
# 
# The input can be either an audio time series (as we do below), or an onset strength envelope as calculated by `librosa.onset.onset_strength()`.
# 

# Now, let's run the beat tracker.
# We'll use the percussive component for this part
plt.figure(figsize=(12, 6))
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

# Let's re-draw the spectrogram, but this time, overlay the detected beats
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Let's draw transparent lines over the beat frames
plt.vlines(beats, 0, log_S.shape[0], colors='r', linestyles='-', linewidth=2, alpha=0.5)

plt.axis('tight')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()


# By default, the beat tracker will trim away any leading or trailing beats that don't appear strong enough.  
# 
# To disable this behavior, call `beat_track()` with `trim=False`.
# 

print('Estimated tempo:        %.2f BPM' % tempo)

print('First 5 beat frames:   ', beats[:5])

# Frame numbers are great and all, but when do those beats occur?
print('First 5 beat times:    ', librosa.frames_to_time(beats[:5], sr=sr))

# We could also get frame numbers from times by librosa.time_to_frames()


# # Beat-synchronous feature aggregation
# 
# Once we've located the beat events, we can use them to summarize the feature content of each beat.
# 
# This can be useful for reducing data dimensionality, and removing transient noise from the features.
# 

# feature.sync will summarize each beat event by the mean feature vector within that beat

M_sync = librosa.feature.sync(M, beats)

plt.figure(figsize=(12,6))

# Let's plot the original and beat-synchronous features against each other
plt.subplot(2,1,1)
librosa.display.specshow(M)
plt.title('MFCC-$\Delta$-$\Delta^2$')

# We can also use pyplot *ticks directly
# Let's mark off the raw MFCC and the delta features
plt.yticks(np.arange(0, M.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])

plt.colorbar()

plt.subplot(2,1,2)
librosa.display.specshow(M_sync)

# librosa can generate axis ticks from arbitrary timestamps and beat events also
librosa.display.time_ticks(librosa.frames_to_time(beats, sr=sr))

plt.yticks(np.arange(0, M_sync.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])             
plt.title('Beat-synchronous MFCC-$\Delta$-$\Delta^2$')
plt.colorbar()

plt.tight_layout()


# Beat synchronization is flexible.
# Instead of computing the mean delta-MFCC within each beat, let's do beat-synchronous chroma
# We can replace the mean with any statistical aggregation function, such as min, max, or median.

C_sync = librosa.feature.sync(C, beats, aggregate=np.median)

plt.figure(figsize=(12,6))

plt.subplot(2, 1, 1)
librosa.display.specshow(C, sr=sr, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time')
plt.title('Chroma')
plt.colorbar()

plt.subplot(2, 1, 2)
librosa.display.specshow(C_sync, y_axis='chroma', vmin=0.0, vmax=1.0)

beat_times = librosa.frames_to_time(beats, sr=sr)
librosa.display.time_ticks(beat_times)

plt.title('Beat-synchronous Chroma (median aggregation)')

plt.colorbar()
plt.tight_layout()


# # Vocal separation
# 
# This notebook demonstrates a simple, but effective technique for separating vocals (and other sporadic foreground signals) from accompanying instrumentation.
# 
# Based on the method of [Rafii and Pardo, 2012](http://www.cs.northwestern.edu/~zra446/doc/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%20ISMIR%202012.pdf), but includes a couple of modifications and extensions:
# 
# - fft windows are 1/4 overlap, instead of 1/2
# - non-local filtering is converted into a soft mask by Wiener filtering.  This is similar in spirit to the soft-masking method used by [Fitzgerald, 2012](http://arrow.dit.ie/cgi/viewcontent.cgi?article=1086&context=argcon), but is a bit more numerically stable in practice.
# 

from __future__ import print_function


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from IPython.display import Audio


# Load an example with vocals
y, sr = librosa.load('audio/Cheese_N_Pot-C_-_16_-_The_Raps_Well_Clean_Album_Version.mp3',
                     sr=44100, offset=20, duration=30)


#S_full, phase = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=2048, window=np.ones))
S_full, phase = librosa.magphase(librosa.stft(y, hop_length=2048, window=np.ones))


# Plot a 5-second slice of the spectrum
idx = slice(*librosa.time_to_frames([10, 15], hop_length=2048, sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S_full[:, idx]**2, ref_power=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()


# Plot a 5-second slice of the spectrum
idx = slice(*librosa.time_to_frames([10, 15], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S_full[:, idx]**2, ref_power=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()


# #### The wiggly lines above are due to the vocal component.  We can separate them by using non-local median filtering.
# 

Audio(data=y, rate=sr)


# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
#
# To avoid being biased by local continuity, we constrain similar frames to be
# separated by at least 2 seconds.
#
# This suppresses sparse/non-repetetitive deviations from the average spectrum,
# and works well to discard vocal elements.

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       k=20,
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input if we assume signals are additive
S_filter = np.minimum(S_full, S_filter)


# The raw filter output can be used as a mask, but it sounds better if we use soft-masking.
# We can also use a margin to reduce bleed between the vocals and instrumentation masks.

# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Using the masks we get a cleaner signal

S_foreground = mask_v * S_full
S_background = mask_i * S_full


# Plot the same slice, but separated into its foreground and background

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
librosa.display.specshow(librosa.logamplitude(S_background[:, idx]**2, ref_power=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(librosa.logamplitude(S_foreground[:, idx]**2, ref_power=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()


# #### And play them back in order: full, background, foreground (vocals)
# 

Audio(data=y, rate=sr)


Audio(data=librosa.istft(S_background * phase), rate=sr)


Audio(data=librosa.istft(S_foreground * phase), rate=sr)


