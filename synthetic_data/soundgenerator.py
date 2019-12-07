# Tyler Piazza
# Python 3
# November-December 2019
# goal is to artificially generate sound waves that sound like birds, using Python

"""
various links/sources:

starting point was
https://dsp.stackexchange.com/questions/53125/write-a-440-hz-sine-wave-to-wav-file-using-python-and-scipy

(note: in our FB chat, an older file I sent started out with a different set of links, but the frequency was a bit messed up there; that file is now called deprecated_soundgenerator.py)
"""

import random
import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, spectrogram
from matplotlib import pyplot as plt

#SAMPLERATE = 100000 # this was originall set to 44100; maybe tweaking this will get better results with higher frequency? It's a performance tradeoff
SAMPLERATE = 44100 # this is the standard
C4FREQ = 261.63 # 261.63 Frequency should be C4 on the Piano

def normalize_data(data):
  if np.max(data) == np.min(data):
    return np.zeros(data.shape[0])
  # data is an np array, and for consistency sake put in the range (-1,1)
  return 2. * ((data - np.min(data)) / (np.max(data) - np.min(data))) - 1.

def random_func(tval):
  # should just be noise
  return random.uniform(-1,1)

def naive_sine_func(tval, freq=C4FREQ):
  # a single tone
  return np.sin(freq * 2 * np.pi * tval)

def sin_plus_rand(tval, freq=C4FREQ):
  # a tone plus noise
  return np.sin(freq * 2 * np.pi * tval) +  random.uniform(-1,1)

def func_to_file(func, filename, length=1., amplitude=1.):
  # take in func, which maps real numbers to real numbers, and create a .wav file saved in filename
  t = np.linspace(0, int(length), int(SAMPLERATE * length))
  # at some point it may make sense to apply the "normalize_data" function to this array before scaling by amplitude
  y = amplitude * np.array([func(tval) for tval in t])
  wavfile.write(filename, SAMPLERATE, y)

def create_uniform_chirp_func(freqrange=[C4FREQ, C4FREQ], timerange=[0., 3.], countrange=[3,5], lengthrange=[0.1, 0.2]):
  # return value: a function that maps real numbers to real numbers, which is supposed to simulate birds chirping
  # all ranges are inclusive

  # figure out how many chirps (use chirpcountrange)
  chirpcount = random.randint(countrange[0], countrange[1])
  chirpdictlist = []
  for i in range(chirpcount):
    # then figure out when the chirps start, end, and their frequencies, using the ranges chirptimerange, chirplengthrange, freqrange
    tempdict = {}
    tempdict["chirpstart"] = random.uniform(timerange[0], timerange[1])
    tempdict["chirpend"] = tempdict["chirpstart"] + random.uniform(lengthrange[0], lengthrange[1])
    tempdict["freq"] = random.uniform(freqrange[0], freqrange[1])
    chirpdictlist.append(tempdict)

  def tempfunc(tval):
    # note that the chirps add to eachother
    tempval = 0.
    for chirpdict in chirpdictlist:
      if tval >= chirpdict["chirpstart"] and tval <= chirpdict["chirpend"]:
        tempval += naive_sine_func(tval, freq=chirpdict["freq"])
    # return a 0 if nothing else panned out, i.e. it's a chirp or nothing
    return tempval
  return tempfunc

def produce_samples_v1(freqrange=[C4FREQ, C4FREQ], timerange=[0., 3.],
  countrange=[3,5], lengthrange=[0.1, 0.2], noiselevel=0.05, length=3., amplitude=1., fileprefix="samplesv1/", numsamples=5):
  # freqrange, timerange, countrange, and lengthrange are all from create_uniform_chirp_func, check that function for what these parameters do
  # length and amplitude are hopefully self explanitory (time in seconds, and something proportional to amplitude)
  # noiselevel is a number in [0.,1.], it represents how much to bias the noise when it comes to adding the sounds
  # numsamples is how many samples, file_prefix is the prefix for the files that you save these with

  t = np.linspace(0, int(length), int(SAMPLERATE * length))
  for i in range(numsamples):
    chirp_func = create_uniform_chirp_func(freqrange=freqrange, timerange=timerange, countrange=countrange, lengthrange=lengthrange)
    y_chirp = amplitude * (1. - noiselevel) * normalize_data(np.array([chirp_func(tval) for tval in t]))
    y_random = amplitude * noiselevel * np.array([random_func(tval) for tval in t])
    # the bird alone
    wavfile.write(fileprefix + "bird_alone_" + str(i) + ".wav", SAMPLERATE, y_chirp)
    # noise alone
    wavfile.write(fileprefix + "noise_alone_" + str(i) + ".wav", SAMPLERATE, y_random)
    # both combined
    wavfile.write(fileprefix + "bird_noise_mix_" + str(i) + ".wav", SAMPLERATE, y_chirp + y_random)

def write_single_advanced_chirp(filename):
  # writes a chirp to filename
  t = np.linspace(0, 0.2, int(SAMPLERATE * 0.2))
  # method can be tinkered between 'linear', 'quadratic', 'logorithmic', and 'hyperbolic'
  # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html for details
  # f0 and f1 are the range of frequencies that the single chirp will go through
  w = chirp(t, f0=10000, f1=5000, t1=0.2, method='linear')
  wavfile.write(filename, SAMPLERATE, w)

def create_advanced_chirp_array(freqrange=[C4FREQ, C4FREQ], timerange=[0., 3.], length=3, countrange=[3,5], lengthrange=[0.1, 0.2]):
  # create an array that is a bunch of advanced chirps going off
  # freqrange, timerange, length, countrange, lengthrange are all analagous to what they are in create_uniform_chirp_func
  accumulator = np.zeros(int(length*SAMPLERATE))
  chirpcount = random.randint(countrange[0], countrange[1])
  # for now, the endfreqis from freqrange[0] to freqmidpoint, beginningfreq is in freqmidpoint to freqrange[1] (so frequency goes down)
  freqmidpoint = int(float(freqrange[0] + freqrange[1]) / 2.)
  assert(freqrange[0] <= freqmidpoint)
  assert(freqrange[1] >= freqmidpoint)
  for _ in range(chirpcount):
    # sample the random features of each chirp
    chirpstart = random.uniform(timerange[0], timerange[1])
    chirplen = random.uniform(lengthrange[0], lengthrange[1])
    endfreq = random.uniform(freqrange[0], freqmidpoint)
    beginningfreq = random.uniform(freqmidpoint, freqrange[1])
    # construct time and chirps
    t = np.linspace(0, chirplen, int(SAMPLERATE * chirplen))
    w = chirp(t, f0=beginningfreq, f1=endfreq, t1=chirplen, method='linear')
    # the resize and roll just lets me set a new starttime for the chirp (so it doesn't start at the beginning)
    w.resize(int(length*SAMPLERATE))
    w = np.roll(w, int(chirpstart*SAMPLERATE))
    accumulator = accumulator + w
  # make sure it's in [-1,1] range
  accumulator = normalize_data(accumulator)
  return accumulator

def produce_samples_v2(freqrange=[C4FREQ, C4FREQ],
  countrange=[3,5], lengthrange=[0.1, 0.2], noiselevel=0.05, length=3., timerange=[0., 3.], amplitude=1., fileprefix="samplesv2/", numsamples=5):
  # freqrange, timerange, countrange, and lengthrange are all from create_advanced_chirp_array, check that function for what these parameters do
  # length and amplitude are hopefully self explanitory (time in seconds, and something proportional to amplitude)
  # noiselevel is a number in [0.,1.], it represents how much to bias the noise when it comes to adding the sounds
  # numsamples is how many samples, file_prefix is the prefix for the files that you save these with
  returnarray = []
  t = np.linspace(0, int(length), int(SAMPLERATE * length))
  for i in range(numsamples):
    # use my create_advanced_chirp_array, which is already normalized
    y_chirp = amplitude * (1. - noiselevel) * create_advanced_chirp_array(freqrange=freqrange, timerange=timerange, length=length, countrange=countrange, lengthrange=lengthrange)
    y_random = amplitude * noiselevel * np.array([random_func(tval) for tval in t])
    # the bird alone
    wavfile.write(fileprefix + "bird_alone_" + str(i) + ".wav", SAMPLERATE, y_chirp)
    # noise alone
    wavfile.write(fileprefix + "noise_alone_" + str(i) + ".wav", SAMPLERATE, y_random)
    # both combined
    wavfile.write(fileprefix + "bird_noise_mix_" + str(i) + ".wav", SAMPLERATE, y_chirp + y_random)

    returnarray.append([y_chirp, y_random, y_chirp + y_random])
  # also return an array of the triplets of values
  return returnarray

def makeSpectrogram(s, framerate=SAMPLERATE):
    """
    note: this code was originally from Irina's construct_dataset.py
    framerate should be samplingrate, which I have as SAMPLERATE
    s should be
    """

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

def showspectrogram(spec_abs, label="Sound"):
  # show spectrogram
  plt.figure()

  plt.subplot(3, 1, 1)
  plt.imshow(np.transpose(spec_abs[:, 0:100]), aspect='auto')
  plt.gca().invert_yaxis()
  plt.xlabel('Time')
  plt.ylabel('Frequency')
  plt.title('Spectrogram of ' + label)
  """
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
  """
  plt.tight_layout()
  plt.show()

def quick_asserts():
  # just to check some basic logic, not super exhaustive
  print("Starting asserts...")
  assert((normalize_data(np.array([0., 2.])) == np.array([-1., 1.])).all())
  assert((normalize_data(np.array([-5., 23.])) == np.array([-1., 1.])).all())
  assert((normalize_data(np.array([-4., -1., 2.])) == np.array([-1., 0.0, 1.0])).all())
  assert((normalize_data(np.array([7.2, 7.2, 7.2, 7.2])) == np.array([0.0, 0.0, 0.0, 0.0])).all())
  print("All asserts passed successfully!")

def main():
  # func_to_file will convert a function into a sound, saving it as a .wav file
  #func_to_file(func=naive_sine_func, filename="singleCtone.wav")
  print("Feel free to change the parameters, or ask Tyler if you're confused about how to construct a certain data set")
  # change the numsamples parameter to change the number of triples of .wav files that are saved
  #produce_samples_v1(freqrange=[10000,15000],fileprefix="samplesv1_", numsamples=1, length=5, timerange=[0., 5.], countrange=[8,12], noiselevel=0.6, amplitude=2.5) # this was an arbitrary frequency range that might sound like bird chirps, feel free to tinker
  #write_single_advanced_chirp("advanced_chirp1.wav")
  tripletlist = produce_samples_v2(freqrange=[5000,10000],fileprefix="samplesv2_", numsamples=1, length=5, timerange=[0., 5.], countrange=[8,12], noiselevel=0.6, amplitude=2.5)
  showspectrogram(makeSpectrogram(tripletlist[0][0])[2])
if __name__ == "__main__":
  main()
