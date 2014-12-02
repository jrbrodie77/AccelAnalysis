#### *This spreadsheet* lets the user import vibration data form the NI accelerometer rig and plot.

# Setup the environment
import sys
import pylab as py
import numpy as np
import math

def calc_fft(signal):

    fft = np.fft.rfft(signal)
    fft /= len(fft)  # Normalize by length
    freq_spacing = 1 / (times[-1] - times[0])
    freqs = np.arange(len(fft)) * freq_spacing

    return np.vstack((freqs, fft))


def fft_abs_norm(times, signal):
    """Takes array of form [times, magnitudes] and returns [freq, magnitude] of signal FFT"""

    fft = np.fft.rfft(signal)
    fft /= len(fft)  # Normalize by length
    fft = np.absolute(fft)

    freq_spacing = 1 / (times[-1] - times[0])
    freqs = np.arange(len(fft)) * freq_spacing

    return np.vstack((freqs, fft)).T


def bin_spectrum(bin_spec, velocity_data):
    """Bins FFT lines, using power sum"""

    binned_keys = np.digitize(velocity_data[:, 0], bin_spec[:, 1], right=True)
    binned_spectrum = {}

    for index, bin_num in enumerate(binned_keys):
        if bin_num < len(bin_spec):
            freq = bin_spec[bin_num, 0]
            if not binned_spectrum.has_key(freq):
                binned_spectrum[freq] = 0

            binned_spectrum[freq] += velocity_data[index, 1]

    binned_spectrum = np.array([[k, v] for k, v in binned_spectrum.iteritems()])
    binned_spectrum = binned_spectrum[binned_spectrum[:, 0].argsort()]
    binned_spectrum = [[v[0], v[1] ** 0.5] for v in binned_spectrum]
    binned_spectrum = np.array(binned_spectrum)

    return binned_spectrum[1:]  # RMS of velocity, dump first bin


def linear_bins(low=0, hi=200, inc=5):
    """Returns array of lin. spaced frequency values for binning"""
    center_freq = np.arange(low, hi, inc)
    limit_freq = center_freq + inc / 2
    bin_spec = np.array([center_freq, limit_freq]).T
    return bin_spec


# In[3]:

def gen_test_tones():
    """This generates a signal for our test case"""
    samprate = 1024
    freqs = [4, 5, 6, 50]

    u = np.arange(0, samprate) * 1.0 / samprate  #Generate 1 second of time data
    y = np.zeros(len(u))

    for freq in freqs:
        print(freq)
        y += [math.sin(num * 2 * math.pi * freq) for num in u]  #*2*freq

    return np.array([u, y]).T


class DataSet:
    """DataSet( 'optional filename' )"""

    def __init__(self, filename=None):
        self.data = []
        self.file_contents = np.array([])

        if filename: self.load_ni_csv_file(filename)

    def load_ni_csv_file(self, filename, skiprows=8, verbose=True):
        """This function reads in National Instruments SignalExpress csv files"""
        self.file_contents = np.loadtxt(filename, skiprows=skiprows, delimiter=',')
        times = self.file_contents[:, 0]  # Slice first column
        num_cols = self.file_contents.shape[1]

        if verbose:
            print str.format("Verbose mode:\n Skipping {s} rows as header", s=skiprows)
            print " Opening: " + filename
            print str.format(" Read {shape} array of data", shape=self.file_contents.shape)
            print " First row of data: ", self.file_contents[0, :]
            print(" num cols:" + str(num_cols))

        for col in range(1, num_cols):
            cur_data_series = DataSeries(times=times, values=self.file_contents[:,col])
            cur_data_series.name = "Series:" + str(col)
            self.data.append(cur_data_series)
        return True


class DataSeries:
    def __init__(self, times=None, values=None):
        """
        :rtype : object, DataSeries
        """
        self.name = ""
        self.time_data = []
        self.freq_data = []

        if times.any():
            self.time_data.append(times)
            self.time_data.append(values)
            self.time_data = np.array(self.time_data)


    def __str__(self):
        return str.format(" {0}: times({1}), data({2}) ", self.name, len(self.time_data[0]), len(self.time_data[1]))


    def __calc_fft(times, signal):
        """Takes array of form [times, magnitudes] and returns [freq, magnitude] of signal FFT"""

        fft = np.fft.rfft(signal)
        fft /= len(fft)  # Normalize by length
        fft = np.absolute(fft)

        freq_spacing = 1 / (times[-1] - times[0])
        freqs = np.arange(len(fft)) * freq_spacing

        return np.vstack((freqs, fft)).T

