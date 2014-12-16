import matplotlib.pyplot as py
import numpy as np
import math
import collections


# namedtuple returns a class definition
FreqSeries = collections.namedtuple('FreqSeries', ['freqs', 'values'])
TimeSeries = collections.namedtuple('TimeSeries', ['times', 'values'])
BinSpec = collections.namedtuple('BinSpec', ['center_freqs', 'limit_freqs'])

def linear_bins(low=0, hi=200, inc=5):
    """Returns array of lin. spaced frequency values for binning"""
    center_freqs = np.arange(low, hi, inc)
    limit_freqs = center_freqs + inc / 2

    return BinSpec(center_freqs, limit_freqs)

class DataSeries:
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

    @property
    def fft_abs(self):
        return np.abs(self.fft_series[1])


def plot_time_series(series, showplot=False):

    if type(series) is not list: #for the case that a single series is passed
        series = [series]

    fig, ax = py.subplots()
    ax.set_title("Raw Signal")
    ax.set_xlabel("time, s")
    ax.set_ylabel("acceleration, g")

    for ds in series:
        ax.plot(ds.time_series.times, ds.time_series.values, label=ds.name)
    legend = ax.legend(loc='upper center')
    return fig

def plot_spectrum(series, showplot=False):

    if type(series) is not list: series = [series]

    fig, ax = py.subplots()
    ax.set_title("Spectrum")
    ax.set_xlabel("freq, Hz")
    ax.set_ylabel("acceleration, g^2/Hz")

    for ds in series:
        ax.plot(ds.spectrum.freqs, ds.spectrum.values, label=ds.name)
    legend = ax.legend(loc='upper center')
    return fig

def plot_binned_spec(series, showplot=False):

    if type(series) is not list: series = [series]

    fig, ax = py.subplots()
    ax.set_title("Binned Spectrum")
    ax.set_xlabel("freq, Hz")
    ax.set_ylabel("acceleration, g^2/Hz")

    for ds in series:
        ax.plot(ds.binned_spec.freqs, ds.binned_spec.values, label=ds.name)
    legend = ax.legend(loc='upper center')
    return fig

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

def gen_test_tones(freqs=[256], samprate=1024, num_samps=1024):
    """This generates a signal for our test case"""
    times = np.arange(0, num_samps) * 1.0 / samprate
    values = np.zeros(num_samps)

    for freq in freqs:
        values += [math.cos(num * 2 * math.pi * freq) for num in times]  #*2*freq

    return (times, values)

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


def load_ni_csv_file(filename, skiprows=8, delimiter=',', verbose=True):
    """This function reads in National Instruments SignalExpress csv files"""
    file_contents = np.loadtxt(filename, skiprows=skiprows, delimiter=',')
    times = file_contents[:, 0]  # Slice first column
    num_cols = file_contents.shape[1]

    if verbose:
        print(str.format("Verbose mode:\n Skipping {s} rows as header", s=skiprows))
        print(" Opening: " + filename)
        print(str.format(" Read {shape} array of data", shape=file_contents.shape))
        print(" First row of data: ", file_contents[0, :])
        print(" num cols:" + str(num_cols))

    series = []

    for col in range(1, num_cols):
        cur_series = DataSeries(times=times, values=file_contents[:,col])
        cur_series.name = "Series" + str(col)
        series.append(cur_series)

    return series


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