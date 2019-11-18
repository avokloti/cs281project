# Tyler Piazza
# Python 3
# 11/10/19
# goal is to generate sound waves that sound like birds, using Python

# note: as of 11/17, this file is not useful!!!! go to the other sound generator file
print("Do not use this file (instead of soundgenerator.py) unless you have good reason. If in doubt, ask Tyler!")

"""
various links/sources:

starting point was
https://stackoverflow.com/questions/9770073/sound-generation-synthesis-with-python
which played (on the computer) a simple sine wave out as audio

this file helps with some of the download issues I had with pyaudio
https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3

a possible way to record sounds to files (listens to microphone, saves to .wav)
https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

another microphone to file helper
https://www.programcreek.com/python/example/52624/pyaudio.PyAudio

this one takes short values, let's you save a wav file --> so function to file is possible here
https://www.tutorialspoint.com/read-and-write-wav-files-using-python-wave
"""

import math, random, struct        #import needed modules
import pyaudio     #sudo apt-get install python-pyaudio
import wave
PyAudio = pyaudio.PyAudio     #initialize pyaudio
import numpy as np
from scipy.io import wavfile

# some of the params

BITRATE = 16000 # number of frames per second/frameset.
#FREQUENCY = 5000 # Hz, waves per second, 261.63=C4-note. Originally set to 500
FREQUENCY = 600
# 2000 gets a cool pitch
if FREQUENCY > BITRATE:
  BITRATE = FREQUENCY+100
# number of seconds
LENGTH = 1
NUMBEROFFRAMES = int(BITRATE * LENGTH)
WIDTH = 1
FORMAT = PyAudio().get_format_from_width(WIDTH)

# the sound functions, mapping natural numbers to integers in [0,256) I believe
def naivesinefunc(i):
  # takes in natural number i, returns basic sine wave - should be constant tone
  return int(math.sin(i/((BITRATE/FREQUENCY)/math.pi))*127+128)

def randomfunc(i):
  # should just be noise
  return random.randint(0, 256)

def functofile(func, filename):
  """
  func: a function that takes in a natural number, returns a value in [0,256] I believe
  filename: example is "sound.wav", it's a .wav file for the sound, to save
  """
  obj = wave.open(filename,'w')
  obj.setnchannels(1) # mono
  obj.setsampwidth(WIDTH)
  obj.setframerate(BITRATE)
  # what is this 99999 value for?
  for i in range(NUMBEROFFRAMES):
     value = func(i)
     data = struct.pack('<h', value)
     obj.writeframesraw( data )
  obj.close()

def functonoise(func):
  """
  func: a function that takes in a natural number, returns a value in [0,256] I believe
  """
  # just generate noise on your computer
  # note that the sound here is a little different than functofile

  WAVEDATA = ''
  # apply the function for the number of frames
  for i in range(NUMBEROFFRAMES):
    WAVEDATA = WAVEDATA+chr(func(i))

  # it may or may not be necessary to deal with this extra restframe business
  #RESTFRAMES = NUMBEROFFRAMES % BITRATE
  #for x in range(RESTFRAMES):
    #WAVEDATA = WAVEDATA+chr(128)

  p = PyAudio()
  # note that format was chosen to match a width, defined in parameters at beginning of file
  stream = p.open(format = FORMAT,
                  channels = 1,
                  rate = BITRATE,
                  output = True)

  stream.write(WAVEDATA)
  stream.stop_stream()
  stream.close()
  p.terminate()

def newwriter():
  sampleRate = 44100
  #frequency = 440
  frequency = 261.63
  length = 5

  t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
  y = np.sin(frequency * 2 * np.pi * t)  #  Has frequency of 440Hz

  wavfile.write('newsine.wav', sampleRate, y)

def main():
  #functonoise(func=randomfunc)
  #functonoise(func=naivesinefunc)
  #functofile(func=randomfunc, filename="whitenoise.wav")
  #functofile(func=naivesinefunc, filename="onetone.wav")
  newwriter()

if __name__ == "__main__":
  main()


