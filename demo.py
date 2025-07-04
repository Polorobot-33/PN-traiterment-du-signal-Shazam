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

    # 1: Load the database
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Encoder
    nperseg=8196
    noverlap=1024
    min_distance=4#25
    time_window=1.
    freq_window=1500
    encoder = Encoding(nperseg=nperseg, noverlap=noverlap, 
      time_window=time_window, 
      freq_window=freq_window,
      min_distance=min_distance)
      
   
    # 3: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    # 4: Use the encoder to extract a signature from the extract
    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.hashes

    encoder.display_spectrogram(True)

    # 5: TODO: Using the class Matching, compare the fingerprint to all the 
    # fingerprints in the database
    tab=[]
    for song_test in songs:
        filename = './samples/' + song_test
        fs, s = read(filename)
        encoder.process(fs, s)
        hashes_test = encoder.hashes
        matching=Matching(hashes,hashes_test)
        tab.append(matching.match)
    i=np.argmax(tab)
    print('Correspondance avec : ' + songs[i][:-4])
        
        
        #matching.display_histogram()
    






