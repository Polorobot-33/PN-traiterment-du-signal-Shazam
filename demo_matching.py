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
      
   
    # 2.1: Randomly get an extract from one of the songs of the database and encode it
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    encoder.process(fs, s[tmin:tmin + duration])
    hashes1 = encoder.hashes

    # 2.2: Get a slightly offsetted extract of the same sample
    offset = int(np.random.randint(-5, 5) * fs)
    encoder.process(fs, s[tmin + offset:tmin + offset + duration])
    hashes2 = encoder.hashes

    # 3: Get a random extract from a song, which should hopefully be different from the first one, and encode it too
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    encoder.process(fs, s[tmin:tmin + duration])
    hashes3 = encoder.hashes

    # 4: display the maching tables in both cases
    matching = Matching(hashes1, hashes1)
    matching.display_scatterplot()
    matching.display_histogram()

    matching = Matching(hashes1, hashes2)
    matching.display_scatterplot()
    matching.display_histogram()
    
    matching = Matching(hashes1, hashes3)
    matching.display_scatterplot()
    matching.display_histogram()






