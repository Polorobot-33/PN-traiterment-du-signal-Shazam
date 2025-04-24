import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from algorithm import *

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    # 1: Encoder
    nperseg=8196
    noverlap=1024
    min_distance=4#25
    time_window=1.
    freq_window=1500
    encoder = Encoding(nperseg=nperseg, noverlap=noverlap, 
      time_window=time_window, 
      freq_window=freq_window,
      min_distance=min_distance)
      
   
    # 2: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    # 3: Use the encoder to extract a signature from the extract
    encoder.process(fs, s[tmin:tmin + duration])
    encoder.display_spectrogram(True)
    






