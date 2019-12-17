""" This script combines birdsong from the Berlin nature archive with background tracks from BirdVox. """
""" Irina Tolkova, Dec 2019."""

import time
import numpy as np
import os
import librosa
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy import signal as scipysig
from scipy.io import wavfile
import pandas as pd
import random

# loads in all MP3s in folder
def loadAllMP3sInFolder(folder, normalize=True):
    files = os.listdir(folder)
    files = files[0:5]
    mp3s = [f for f in files if f[-4:] == '.mp3']
    tracks = [librosa.core.load(folder + f, sr=None) for f in mp3s]
    return tracks

# loads in all WAVs in folder
def loadAllBirdVoxBackgrounds(folder, normalize=True):
    labels = pd.read_csv(folder + 'BirdVoxDCASE20k_labels.csv', header=0)
    files = os.listdir(folder)
    wavfilenames = [f for f in files if f[-4:] == '.wav']
    tracks = []
    for file in wavfilenames:
        hasbird = int(labels.loc[labels['itemid'] == file[:-4]]['hasbird'])
        if not hasbird:
            tracks.append(librosa.core.load(folder + file, sr=None))
    return tracks

# make a spectrogram out of a signal
def makeSpectrogram(s, framerate):
    # create hanning window of size n
    N = len(s)
    n = 512
    g = np.hanning(n)
    freqs = np.fft.fftfreq(n, 1/framerate)
    t = np.arange(N) * 1/framerate
    
    # move by more than one per iteration
    inds = np.arange(0, N - n, n/5)
    
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
    
    spec_abs = spec_abs / np.max(spec_abs)
    return [inds, freqs, spec_abs]


def makeSpectrogramScipy(s, framerate, n=512):
    # Compute spectrogram from your data
    freqs, t, full_specs = scipysig.spectrogram(s, fs=framerate, window=np.hanning(n), nperseg=n, noverlap=4*n//5, nfft=2**9, detrend=False, mode='magnitude')
    bandwidth_clip = [0., 5000.]
    specs = full_specs[freqs.searchsorted(bandwidth_clip[0], side='left'):freqs.searchsorted(bandwidth_clip[1], side='right'), :]
    return [t, freqs, full_specs]



# split up into segments of length track_length and put in list
def split(track, subtrack_length, num_birds_per_subtrack, target_noise_ratio):
    total_subtrack_list = []
    bird_subtrack_list = []
    bg_subtrack_list = []
    
    num_subtracks = np.floor(len(track)/subtrack_length)
    
    for i in np.arange(num_subtracks):
        # get subtrack
        subtrack = track[int(i * subtrack_length):int((i+1) * subtrack_length)]
        # normalize background subtrack to target noise amount
        bg_subtrack = target_noise_ratio * subtrack/np.max(subtrack)
        
        # get bird tracks for this subtrack
        birdsong_inds = np.random.randint(low=0, high=len(fg), size=(num_birds_per_subtrack))
        
        # randomly sample start locations within track
        bird_locations = np.random.randint(low=0, high=np.round(0.75 * subtrack_length), size=(num_birds_per_subtrack))
        
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


def writeCheckEraseAndRewrite(filename, spec):
    np.savetxt(filename, spec, fmt='%1.3e')
    try:
        # check if it is possible to load txt
        check = np.loadtxt(filename)
    except ValueError:
        # delete file
        os.remove(filename)
        # rewrite file
        print('Rewriting file ' + filename)
        # recursively check and rewrite?
        writeCheckEraseAndRewrite(filename, spec)



## ---------------- ACTUAL SCRIPT ----------------

# directory paths
dir = '/Users/ira/Documents/bioacoustics/cs281/'

# set background and foreground
bg_dir = './data/birdvox/'
fg_dir = './data/birdsong/'

# load all background sounds
t0 = time.time()
bg = loadAllBirdVoxBackgrounds(bg_dir)
t1 = time.time()
print("Loading in background took: " + str(t1 - t0))

t0 = time.time()
fg = loadAllMP3sInFolder(fg_dir)
t1 = time.time()
print("Loading in foreground took: " + str(t1 - t0))

# set seed
np.random.seed(42)

# define parameters for creating a subtrack
sampling_rate = bg[0][1] # should be 44100
subtrack_length = int(1 * sampling_rate)
num_birds_per_subtrack = 2 # number of birdsong tracks per subtrack
target_noise_ratio = 1 # relative ratio of noise to signal after normalization

# prepare lists to store tracks
all_total_subtracks = []
all_bird_subtracks = []
all_bg_subtracks = []

# for each background track...
for b in bg:
    # split it into subtracks
    [total_subtracks, bird_subtracks, bg_subtracks] = split(b[0], subtrack_length, num_birds_per_subtrack, target_noise_ratio)
    
    # add to list
    all_total_subtracks = all_total_subtracks + total_subtracks
    all_bird_subtracks = all_bird_subtracks + bird_subtracks
    all_bg_subtracks = all_bg_subtracks + bg_subtracks

# shuffle the tracks!
zipped_lists = list(zip(all_total_subtracks, all_bird_subtracks, all_bg_subtracks))
random.shuffle(zipped_lists)
all_total_subtracks, all_bird_subtracks, all_bg_subtracks = zip(*zipped_lists)

# for each subtrack...
for i in np.arange(1779, len(all_total_subtracks)):
    print('Writing subtrack and spectrograms to file: ' + str(i))
    
    # save wav files to files
    #wavfile.write(dir + 'constructed_data/audio_fg/' + str(i) + '.wav', sampling_rate, all_bird_subtracks[i])
    #wavfile.write(dir + 'constructed_data/audio_bg/' + str(i) + '.wav', sampling_rate, all_bg_subtracks[i])
    #wavfile.write(dir + 'constructed_data/audio_mix/' + str(i) + '.wav', sampling_rate, all_total_subtracks[i])
    
    # create spectrograms
    [t_fg, freqs_fg, specs_fg] = makeSpectrogram(all_bird_subtracks[i], sampling_rate)
    #[t_bg, freqs_bg, specs_bg] = makeSpectrogram(all_bg_subtracks[i], sampling_rate)
    [t_mix, freqs_mix, specs_mix] = makeSpectrogram(all_total_subtracks[i], sampling_rate)
    
    # save foreground spectrogram
    writeCheckEraseAndRewrite(dir + 'constructed_data/spec_fg/' + str(i) + '.csv', specs_fg[:, 0:100])
    #writeCheckEraseAndRewrite(dir + 'constructed_data/spec_bg/' + str(i) + '.csv', specs_bg[:, 0:100])
    writeCheckEraseAndRewrite(dir + 'constructed_data/spec_mix/' + str(i) + '.csv', specs_mix[:, 0:100])


"""
    # plot every hundred values
    if (np.mod(i, 10) == 0):
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(np.transpose(specs_fg[:, 0:100]), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram of Foreground (Birdsong) (i = ' + str(i) + ')')
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.imshow(np.transpose(specs_bg[:, 0:100]), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram of Background (Noise) (i = ' + str(i) + ')')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.imshow(np.transpose(specs_mix[:, 0:100]), aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram of Mix (Sum) (i = ' + str(i) + ')')
        plt.colorbar()
        plt.savefig(dir + 'constructed_data/pngs/' + str(i) + '.png')
        plt.clf()

"""



"""
# for each background track...
b = bg[10]

testing = 0

# split it into subtracks
[total_subtracks, bird_subtracks, bg_subtracks] = split(b[0], subtrack_length, num_birds_per_subtrack, target_noise_ratio)

# add to list
all_total_subtracks = all_total_subtracks + total_subtracks
all_bird_subtracks = all_bird_subtracks + bird_subtracks
all_bg_subtracks = all_bg_subtracks + bg_subtracks

i = 4

[t_fg, freqs_fg, specs_fg] = makeSpectrogram(all_bird_subtracks[i], sampling_rate)
[t_bg, freqs_bg, specs_bg] = makeSpectrogram(all_bg_subtracks[i], sampling_rate)
[t_mix, freqs_mix, specs_mix] = makeSpectrogram(all_total_subtracks[i], sampling_rate)

[t_fg_2, freqs_fg_2, specs_fg_2] = makeSpectrogramScipy(all_bird_subtracks[i], sampling_rate)
[t_bg_2, freqs_bg_2, specs_bg_2] = makeSpectrogramScipy(all_bg_subtracks[i], sampling_rate)
[t_mix_2, freqs_mix_2, specs_mix_2] = makeSpectrogramScipy(all_total_subtracks[i], sampling_rate)

# plot things!
plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.transpose(specs_fg[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(np.transpose(specs_bg[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(np.transpose(specs_mix[:, 0:100]), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')
plt.colorbar()

plt.tight_layout()

plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(specs_fg_2[0:150,:], aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(specs_bg_2[0:150,:], aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(specs_mix_2[0:150,:], aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')
plt.colorbar()

plt.tight_layout()

plt.show()

# show log spectrogram

plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.log(np.transpose(specs_fg[:, 0:150])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(np.log(np.transpose(specs_bg[:, 0:150])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(np.log(np.transpose(specs_mix[:, 0:150])), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')
plt.colorbar()

# maybe try converting to decibels (librosa) and using those spectrograms as input?

plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.log(100 * np.transpose(specs_fg[:, 0:150] + np.ones(specs_fg[:, 0:150].shape))), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(np.log(100 * np.transpose(specs_bg[:, 0:150] + np.ones(specs_fg[:, 0:150].shape))), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(np.log(100 * np.transpose(specs_mix[:, 0:150] + np.ones(specs_fg[:, 0:150].shape))), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')
plt.colorbar()

plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(np.log(10 * specs_fg_2[0:150,:] + np.ones(specs_fg_2[0:150,:].shape)), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Foreground (Birdsong)')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(np.log(10 * specs_bg_2[0:150,:] + np.ones(specs_fg_2[0:150,:].shape)), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Background (Noise)')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(np.log(10 * specs_mix_2[0:150,:] + np.ones(specs_fg_2[0:150,:].shape)), aspect='auto')
plt.gca().invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram of Mix (Sum)')
plt.colorbar()

plt.show()

"""

"""
# calculate spectrogram
subtrack_index = 400

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


# What next?

# Is there a visible difference?
# Plot log-scale?

# write to file
wavfile.write(dir + 'testing_fg_subtrack.wav', sampling_rate, all_bird_subtracks[subtrack_index])
wavfile.write(dir + 'testing_bg_subtrack.wav', sampling_rate, all_bg_subtracks[subtrack_index])
wavfile.write(dir + 'testing_mix_subtrack.wav', sampling_rate, all_total_subtracks[subtrack_index])

# next steps: generate files and save in folder, then push to git.
"""
