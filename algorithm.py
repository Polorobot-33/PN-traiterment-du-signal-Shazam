"""
Algorithm implementation
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

    """
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """
   



    def __init__(self, nperseg=128, noverlap=32, min_distance = 50, time_window = 1., freq_window = 1500):

        """
        Class constructor

        To Do
        -----

        Initialize in the constructor all the parameters required for
        creating the signature of the audio files. These parameters include for
        instance:
        - the window selected for computing the spectrogram
        - the size of the temporal window 
        - the size of the overlap between subsequent windows
        - etc.

        All these parameters should be kept as attributes of the class.
        """

        # Insert code here
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.time_window = time_window
        self.freq_window = freq_window
        self.min_distance = min_distance

    def process(self, fs, s):

        """

        To Do
        -----

        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")


        Parameters
        ----------

        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """

        self.fs = fs
        self.s = s

        # Insert code here
        self.f, self.t, self.S = spectrogram(s, fs, nperseg=self.nperseg, noverlap=self.noverlap)
        self.anchors = np.array([[t, f] for [f, t] in peak_local_max(self.S, min_distance=self.min_distance, exclude_border=False, threshold_rel = 0.01)])


        # extract hash from anchors

        hash = []
        for (itime, ifreq) in self.anchors :
            time = self.t[itime]
            freq = self.f[ifreq]

            for (target_itime, target_ifreq) in self.anchors :            
               target_time = self.t[target_itime]
               target_freq = self.f[target_ifreq]

               if target_time < time or target_time > time + self.time_window or abs(target_freq - freq) > self.freq_window : continue

               hash.append({'t' : time, 'hash':np.array([target_time - time, freq, target_freq])})


        self.hashes = hash
         
         # This hash representation is time invariant as only the time difference between the anchor and the target is saved.
         # Thus, the pattern or melody saved in the hash can be recognized at any time of song.



    def display_spectrogram(self, display_anchors) :

        """
        Display the spectrogram of the audio signal

        Parameters
        ----------
        display_anchors: boolean
           when set equal to True, the anchors are displayed on the
           spectrogram
        """

        plt.pcolormesh(self.t, self.f/1e3, self.S, shading='gouraud')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [kHz]')
        plt.ylim([0, 2])
        if(display_anchors) :
            plt.scatter(self.t[self.anchors[:, 0]], self.f[self.anchors[:, 1]]/1e3)
        plt.show()



# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):

        """
        Compare the hashes from two audio files to determine if these
        files match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
          
        """


        self.hashes1 = hashes1
        self.hashes2 = hashes2

        times = np.array([item['t'] for item in self.hashes1])
        hashcodes = np.array([item['hash'] for item in self.hashes1])

        # Establish matches
        self.matching = []
        for hc in self.hashes2:
             t = hc['t']
             h = hc['hash'][np.newaxis, :]
             dist = np.sum(np.abs(hashcodes - h), axis=1)
             mask = (dist < 1e-6)
             if (mask != 0).any():
                 self.matching.append(np.array([times[mask][0], t]))
        self.matching = np.array(self.matching)

        # TODO: complete the implementation of the class by
        # 1. creating an array "offset" containing the time offsets of the 
        #    hashcodes that match

        self.offsets = self.matching[:, 1] - self.matching[:, 0]


        # 2. implementing a criterion to decide whether or not both extracts
        #    match     
           
        # The criterion that can be used is if the max of the histogram is greater
        # than a given n number of times the average value of the non null values.
        # Null values are not considered as they depend on the number of match,
        # which is not a good indicator of whether or not two samples match.
        # We use n = 3 here

        histogram = [h for h in np.histogram(self.offsets, bins=100, density=True)[0] if h > 0]
        mean = np.mean(histogram)
        max = np.max(histogram)
        self.match = max/mean
        print(f'Mean : {mean}, max : {max}, samples match : {self.match}')
       
             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
    
        plt.scatter(self.matching[:, 0], self.matching[:, 1])
        plt.show()


    def display_histogram(self):

        """
        Display the offset histogram
        """
    
        plt.hist(self.offsets, bins=100, density=True)
        plt.xlabel('Offset (s)')
        plt.show()


