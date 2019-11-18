""" This script combines birdsong from the Berlin nature archive with background tracks downloaded from https://mc2method.org/white-noise/."""
""" Irina Tolkova, Nov 2019."""

import numpy as np
import os
import librosa
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy import signal as scipysig
from scipy.io import wavfile

# loads in all MP3s in folder
def loadAllMP3sInFolder(folder, normalize=True):
    files = os.listdir(folder)
    mp3s = [f for f in files if f[-4:] == '.mp3']
    tracks = [librosa.core.load(folder + f, sr=None) for f in mp3s]
    return tracks

# split up into segments of length track_length and put in list
def split(track, subtrack_length, num_birds_per_subtrack, target_noise_ratio):
    total_subtrack_list = []
    bird_subtrack_list = []
    bg_subtrack_list = []
    
    num_subtracks = np.floor(len(track)/subtrack_length)
    
    for i in np.arange(num_subtracks):
        print(i * subtrack_length)
        print((i+1) * subtrack_length)
        # get subtrack
        subtrack = track[int(i * subtrack_length):int((i+1) * subtrack_length)]
        # normalize background subtrack to target noise amount
        bg_subtrack = target_noise_ratio * subtrack/np.max(subtrack)
        
        # get bird tracks for this subtrack
        birdsong_inds = np.random.randint(low=0, high=len(fg), size=(num_birds_per_subtrack))
        
        # randomly sample start locations within track
        bird_locations = np.random.randint(low=0, high=subtrack_length, size=(num_birds_per_subtrack))
        
        # make a track of foreground sounds
        bird_subtrack = np.zeros(subtrack_length)
        
        # fill up bird subtrack with birdsongs at chosen indeces
        for j in np.arange(num_birds_per_subtrack):
            
            # take current birdsong segment and normalize
            birdsong = fg[birdsong_inds[j]][0]
            birdsong = birdsong/np.max(birdsong)
            
            # find maximum index for insertion (to not overflow)
            max_index = np.min((bird_locations[j] + len(birdsong), subtrack_length))
            
            # add to cumulative birdsong track
            bird_subtrack[bird_locations[j]:max_index] = bird_subtrack[bird_locations[j]:max_index] + birdsong[0:np.min((len(birdsong), subtrack_length - bird_locations[j]))]
        
        # make cumulative subtrack (background noise plus birds)
        total_subtrack = bg_subtrack + bird_subtrack
        
        # add to list
        total_subtrack_list.append(total_subtrack)
        bird_subtrack_list.append(bird_subtrack)
        bg_subtrack_list.append(bg_subtrack)
    
    return [total_subtrack_list, bird_subtrack_list, bg_subtrack_list]

# directory paths
dir = '/Users/ira/Documents/bioacoustics/cs281/'
bg_dir = dir + 'data/background/'
fg_dir = dir + 'data/birdsong/'

# load all background sounds
bg = loadAllMP3sInFolder(bg_dir)
fg = loadAllMP3sInFolder(fg_dir)

# how long are background sounds?
for b in bg:
    print('Length of background in seconds: ' + str(len(b[0])/(10 * b[1])))

# how long are foreground sounds?
for f in fg:
    print('Length of foreground in seconds: ' + str(len(f[0])/(10 * f[1])))

# amplitudes of background sounds?
for b in bg:
    print('Max amplitude of background: ' + str(np.max(b[0])))

# amplitudes of foreground sounds?
for f in fg:
    print('Max amplitude of foreground: ' + str(np.max(f[0])))

# define length of each track, number of audio tracks, number of birdsongs per track, target noise ratio
sampling_rate = bg[0][1]
subtrack_length = 5 * sampling_rate # (60 seconds)
num_birds_per_subtrack = 2
target_noise_ratio = 1.5

all_total_subtracks = []
all_bird_subtracks = []
all_bg_subtracks = []

for b in bg:
    [total_subtracks, bird_subtracks, bg_subtracks] = split(b[0], subtrack_length, num_birds_per_subtrack, target_noise_ratio)
    all_total_subtracks = all_total_subtracks + total_subtracks
    all_bird_subtracks = all_bird_subtracks + bird_subtracks
    all_bg_subtracks = all_bg_subtracks + bg_subtracks

"""
def makeSpectrogram(data, sampling_rate, n=500):
    # Compute spectrogram from your data
    freqs, t, full_specs = scipysig.spectrogram(data, fs=sampling_rate, window=np.hanning(n), nperseg=n, noverlap=4*n//5, nfft=2**9, detrend=False, mode='magnitude')
    bandwidth_clip = [0., 5000.]
    specs = full_specs[freqs.searchsorted(bandwidth_clip[0], side='left'):freqs.searchsorted(bandwidth_clip[1], side='right'), :]
    return [t, freqs, full_specs]
"""

def makeSpectrogram(s, framerate):
    # create hanning window of size n
    N = len(s)
    n = 400
    g = np.hanning(n)
    freqs = np.fft.fftfreq(n, 1/framerate)
    t = np.arange(N) * 1/framerate
    
    # move by more than one per iteration
    inds = np.arange(0, N - n, n/4)
    
    # save amplitudes at specific times
    spec_abs = np.zeros((len(inds), np.int(n/2)))
    spec_real = np.zeros((len(inds), np.int(n/2)))
    spec_imag = np.zeros((len(inds), np.int(n/2)))
    
    for i in np.arange(len(inds)):
        #print(str(inds[i]) + "  " + str(inds[i]+n))
        fw = np.fft.fft(s[int(inds[i]):int(inds[i]+n)] * g)
        fw_abs = np.abs(fw)
        spec_abs[i, :] = fw_abs[0:np.int(n/2)]
        spec_imag[i, :] = fw.imag[0:np.int(n/2)]
        spec_real[i, :] = fw.real[0:np.int(n/2)]
    
    return [t, freqs, spec_abs]

# calculate spectrogram
subtrack_index = 200

[t_fg, freqs_fg, specs_fg] = makeSpectrogram(all_bird_subtracks[subtrack_index], sampling_rate)
[t_bg, freqs_bg, specs_bg] = makeSpectrogram(all_bg_subtracks[subtrack_index], sampling_rate)
[t_mix, freqs_mix, specs_mix] = makeSpectrogram(all_total_subtracks[subtrack_index], sampling_rate)

# write to file
wavfile.write(dir + 'testing_fg_subtrack.wav', sampling_rate, all_bird_subtracks[subtrack_index])
wavfile.write(dir + 'testing_bg_subtrack.wav', sampling_rate, all_bg_subtracks[subtrack_index])
wavfile.write(dir + 'testing_mix_subtrack.wav', sampling_rate, all_total_subtracks[subtrack_index])

# show spectrogram
plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.transpose(specs_fg[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')

plt.subplot(3, 1, 2)
plt.imshow(np.transpose(specs_bg[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')

plt.subplot(3, 1, 3)
plt.imshow(np.transpose(specs_mix[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')

plt.tight_layout()
plt.show()

# show log spectrogram
plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.log(np.transpose(specs_fg[:, 0:100])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')

plt.subplot(3, 1, 2)
plt.imshow(np.log(np.transpose(specs_bg[:, 0:100])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')

plt.subplot(3, 1, 3)
plt.imshow(np.log(np.transpose(specs_mix[:, 0:100])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')

plt.show()

# What next?

# Is there a visible difference?
# Plot log-scale?

