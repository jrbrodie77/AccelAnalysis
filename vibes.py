"""Vibes.py - Vibration Data Module

This module provides a class called DataSeries for containing time and frequency data from
vibration or acoustics measurements.

  Example:
    filename = 'shelf_inverted.txt'
    data_columns = vibes.load_ni_txt_file(filename)

    # Set name of each column and plot graph
    data_columns[0].name = "On table"
    data_columns[0].color = "red"   # blue, green, red, cyan, magenta, yellow, black, white

    data_columns[1].name = "Above load cell"
    data_columns[1].color = "grey"

    plt1 = vibes.plot_time_series(data_columns)
    py.show()  #Show plots
"""

import matplotlib.pyplot as py
import numpy as np
import math
import collections

# Creating subclasses of namedtuple
FreqSeries = collections.namedtuple('FreqSeries', ['freqs', 'values'])
TimeSeries = collections.namedtuple('TimeSeries', ['times', 'values'])
BinSpec = collections.namedtuple('BinSpec', ['center_freqs', 'limit_freqs'])

def linear_bins(low=0, hi=200, inc=5):
    """Returns BinSpec instance that specifies center and bandlimit freqs"""
    center_freqs = np.arange(low, hi, inc)
    limit_freqs = center_freqs + inc / 2

    return BinSpec(center_freqs, limit_freqs)

class DataSeries:
    """Container object, initialized with time series data, also
    calculates FFT, spectrum, PSD, transfer function"""

    def __init__(self, times=None, values=None, name="", bin_spec=linear_bins()):
        self.name = name
        if type(times) is not None:
            times, values = np.array(times), np.array(values)

            if times.any():
                self.time_series = TimeSeries(times, values)
                self.__add_samprate()
                self.fft_series = calc_fft(self.time_series, samprate=self.samprate )
                self.spectrum = calc_spectrum(self.fft_series)
                self.binned_spec = bin_spectrum(self.spectrum, bin_spec)
                self.color = None  # blue, green, red, cyan, magenta, yellow, black, white

    def __add_samprate(self):
        num_samples = len(self.time_series.times)
        start_time = self.time_series.times[0]
        end_time = self.time_series.times[-1]
        samprate = (num_samples-1)/(end_time - start_time)

        self.samprate = samprate

    def __str__(self):
        pretty = """.name='{0}'
 times({1}), data({2}) \n"""

        return str.format(pretty, self.name, len(self.time_series.times), len(self.time_series.values))

    def calc_tf(self, ref_series):
        tf = self.fft_series.values / (ref_series.fft_series.values+1e-15)
        freqs = self.fft_series.freqs
        self.tf = FreqSeries(freqs, tf)

        return self.tf

def plot_time_series(series, showplot=False, xlim = None, ylim = None ):
    """plot_time_series( <DataSeries object>, <showplot=True/False> )"""

    if type(series) is not list: #for the case that a single series is passed
        series = [series]

    fig, ax = py.subplots(figsize=[8,6])
    ax.set_title("Time Series")
    ax.set_xlabel("time, s")
    ax.set_ylabel("acceleration, g")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    for ds in series:
        if ds.color:
            ax.plot(ds.time_series.times, ds.time_series.values, label=ds.name, color=ds.color)
        else:
            ax.plot(ds.time_series.times, ds.time_series.values, label=ds.name)

    legend = ax.legend()
    return fig, ax

def plot_spectrum(series, showplot=False, xlim = None, ylim = None ):
    """plot_spectrum( <DataSeries object>, <showplot=True/False> )"""

    if type(series) is not list: series = [series]

    fig, ax = py.subplots(figsize=[8,6])
    ax.set_title("Spectrum")
    ax.set_xlabel("freq, Hz")
    ax.set_ylabel("acceleration, g^2/Hz")
    ax.set_yscale('log')
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    for ds in series:
        if ds.color:
            ax.plot(ds.spectrum.freqs, ds.spectrum.values, label=ds.name, color=ds.color)
        else:
            ax.plot(ds.spectrum.freqs, ds.spectrum.values, label=ds.name)

    legend = ax.legend()
    return fig, ax

def plot_binned_spectrum(series, showplot=False, xlim = None, ylim = None ):
    """plot_binned_spec( <DataSeries object>, <showplot=True/False> )"""
    if type(series) is not list: series = [series]

    fig, ax = py.subplots(figsize=[8,6])
    ax.set_title("Binned Spectrum")
    ax.set_xlabel("freq, Hz")
    ax.set_ylabel("acceleration, g^2/Hz")
    ax.set_yscale('log')
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    for ds in series:
        if ds.color:
            ax.plot(ds.binned_spec.freqs, ds.binned_spec.values, label=ds.name, color=ds.color)
        else:
            ax.plot(ds.binned_spec.freqs, ds.binned_spec.values, label=ds.name)

    legend = ax.legend()
    return fig, ax

def plot_tf(series, showplot=False, xlim = None, ylim = None):
    """plot_tf( <DataSeries object w/ tf>, <showplot=True/False> )"""
    if type(series) is not list: series = [series]

    fig, ax = py.subplots(2,1, figsize=[8,12])
    ax[0].set_title("Transfer Function")
    ax[0].set_xlabel("freq, Hz")
    ax[0].set_ylabel("dB")
    if xlim is not None: ax[0].set_xlim(xlim)
    if ylim is not None: ax[0].set_ylim(ylim)

    ax[1].set_ylim([-180,180])
    if xlim is not None: ax[1].set_xlim(xlim)

    ax[1].set_ylabel('phase(deg)', color='r')
    for tl in ax[1].get_yticklabels():
        tl.set_color('r')

    for ds in series:
        angles = np.angle(ds.tf.values, deg=True)
        mags = 20*np.log10(np.abs(ds.tf.values))
        ax[0].plot(ds.tf.freqs, mags, label=ds.name)
        ax[1].plot(ds.tf.freqs, angles, label=ds.name, color='r')
        legend = ax[0].legend()

    if showplot: py.show()

    return fig, ax

def calc_fft(time_series, samprate=1):

    n = len(time_series.times)
    fft = np.fft.rfft(time_series.values)/n
    freqs = np.fft.rfftfreq(n, 1.0/samprate)

    if n % 2:  # odd-length
        fft[1:-1]=fft[1:]*2**0.5  #include last point
    else:  # even-length
        fft[1:-1]=fft[1:-1]*2**0.5 #exclude last point

    return FreqSeries(freqs, fft)

def calc_spectrum(fft_series, bin_spec=linear_bins()):
    """Takes raw FFT data, returns binned energy spectral density in freq bins"""
    values = np.abs(fft_series.values)**2
    spectrum = FreqSeries(fft_series.freqs, values)
    return spectrum

def bin_spectrum(freq_series, bin_spec=linear_bins()):
    """Sums values according to frequency bin"""

    values = np.array(freq_series.values)

    #freqs_to_bins_map: array of bin index that each freq corresponds to
    freqs_to_bins_map = np.digitize(freq_series.freqs, bin_spec.limit_freqs, right=True)

    summed_values = []
    for bin_idx, freq in enumerate(bin_spec.center_freqs):
        summed_bin_value = values[freqs_to_bins_map == bin_idx].sum() #values sliced with boolean
        summed_values.append(summed_bin_value)

    return FreqSeries(bin_spec.center_freqs, np.array(summed_values))

def load_ni_csv_file(filename, skiprows=8, delimiter=',', verbose=False):
    """This function reads in National Instruments SignalExpress csv files"""

    file_contents = np.loadtxt(filename, skiprows=skiprows, delimiter=delimiter)
    times = file_contents[:, 0]  # Slice first column
    num_cols = file_contents.shape[1]

    if verbose:
        print(" \nOpening: " + filename)
        print(str.format(" Read {shape} array of data", shape=file_contents.shape))
        print(" number of columns:" + str(num_cols))
        print(" First row of data: ", file_contents[0, :])


    series_list = []

    for col in range(1, num_cols):
        cur_series = DataSeries(times=times, values=file_contents[:,col])
        cur_series.name = "Series" + str(col)
        series_list.append(cur_series)

    return series_list

def load_ni_txt_file(filename):
    return load_ni_csv_file(filename, skiprows=22, delimiter='\t')

def main():
    dataset = load_ni_csv_file(filename = "vibedata.csv")
    dataset[0].name = "First Meas Column"
    dataset[1].name = "Second Meas Column"

    plot_time_series(dataset)
    plot_spectrum(dataset)
    plot_binned_spec(dataset)
    py.show()


if __name__ == "__main__":
    main()