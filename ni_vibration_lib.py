import matplotlib.pyplot as py
import numpy as np
import math

def calc_fft(signal, samprate=1):
    times = signal[0]
    n = len(times)
    sig = signal[1]
    fft = np.fft.rfft(sig)/n
    freqs = np.fft.rfftfreq(n, 1.0/samprate)

    if n % 2:  # odd-length
        fft[1:-1]=fft[1:]*2**0.5  #include last point
    else:  # even-length
        fft[1:-1]=fft[1:-1]*2**0.5 #exclude last point

    return (freqs, fft)

def gen_test_tones(freqs=[256], samprate=1024, num_samps=1024):
    """This generates a signal for our test case"""
    times = np.arange(0, num_samps) * 1.0 / samprate
    values = np.zeros(num_samps)

    for freq in freqs:
        values += [math.cos(num * 2 * math.pi * freq) for num in times]  #*2*freq

    return (times, values)


def linear_bins(low=0, hi=200, inc=5):
    """Returns array of lin. spaced frequency values for binning"""
    center_freq = np.arange(low, hi, inc, dtype=float)
    limit_freq = center_freq + float(inc) / 2
    bin_spec = (center_freq, limit_freq)
    return bin_spec


def calc_spectrum(fft_data, bin_spec=linear_bins()):
    """Takes raw FFT data, returns binned energy spectral density in freq bins"""

    freqs = fft_data[0]
    values = np.abs(fft_data[1])**2
    spectrum = (freqs, values)
    return spectrum


def bin_spectrum(freq_data, bin_spec=linear_bins()):
    """Bins FFT lines, using power sum"""

    binned_keys = np.digitize(freq_data[0], bin_spec[1], right=True)
    binned_spectrum = {}

    for index, bin_num in enumerate(binned_keys):
        if bin_num < len(bin_spec[1]):
            freq = bin_spec[0][bin_num]
            if not binned_spectrum.has_key(freq):
                binned_spectrum[freq] = 0

            binned_spectrum[freq] += freq_data[1][index]

    binned_spectrum = np.array([[k, v] for k, v in binned_spectrum.iteritems()])
    binned_spectrum = binned_spectrum[binned_spectrum[0].argsort()]
    binned_spectrum = [[v[0], v[1]] for v in binned_spectrum]
    binned_spectrum = np.array(binned_spectrum)
    freqs, values = binned_spectrum[0], binned_spectrum[1]
    return (freqs, values)


def rms(vals):
    return np.sqrt(np.mean(np.square(vals)))


class DataSet:
    """DataSet( 'optional filename' )"""

    def __init__(self, filename=None, verbose=True):
        self.data = []
        self.file_contents = np.array([])
        if filename: self.load_ni_csv_file(filename, verbose=verbose)

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
            cur_data_series.name = "Series" + str(col)
            self.data.append(cur_data_series)
        return True


class DataSeries:
    def __init__(self, times=None, values=None):
        """
        :rtype : object, DataSeries
        """
        self.name = ""
        self.time_data = []
        self.fft_data = []
        self.spectrum = []
        if type(times)!='NoneType':
            times, values = np.array(times), np.array(values)

        if times.any():
            self.time_data=(times, values)
            self.__add_samprate()
            self.fft_data = calc_fft(self.time_data, samprate=self.samprate )
            self.spectrum = calc_spectrum(self.fft_data)

    def __add_samprate(self):
        self.samprate = float((len(self.time_data[0])-1))/(self.time_data[0][-1] - self.time_data[0][0])

    def __str__(self):
        pretty = """.name='{0}'
 times({1}), data({2}) \n"""

        return str.format(pretty, self.name, len(self.time_data[0]), len(self.time_data[1]))

    @property
    def fft_abs(self):
        return np.abs(self.fft_data[1])


def plot_time_data(dataseries, showplot=False):

    if type(dataseries) != list: dataseries = [dataseries]

    fig, ax = py.subplots()
    ax.set_title("Raw Signal")
    ax.set_xlabel("time, s")
    ax.set_ylabel("acceleration, g")

    for ds in dataseries:
        ax.plot(ds.time_data[0], ds.time_data[1], label=ds.name)
    legend = ax.legend(loc='upper center')
    return fig


def plot_spectrum(dataseries, showplot=False):

    if type(dataseries) != list: dataseries = [dataseries]

    fig, ax = py.subplots()
    ax.set_title("Spectrum")
    ax.set_xlabel("freq, Hz")
    ax.set_ylabel("acceleration, g^2/Hz")

    for ds in dataseries:
        print(len(ds.spectrum[0]))
        print(len(ds.spectrum[1]))
        ax.plot(ds.spectrum[0], ds.spectrum[1], label=ds.name)
    legend = ax.legend(loc='upper center')
    return fig

#a = [3,3,3,3,-3,-3,-3,-3]
#a = [1,-1,1,-1,1,-1,1,-1]
#a = [3,3,3,3,3,3,3,3]
#b= [0,1,2,3,4,5,6,7]
#ds1=DataSeries(b,a)
#ds2 = gen_test_tones_8()

#print(ds2.fft_abs)
def main():
    dataset=DataSet("vibedata.csv")
    dataset.data[0].name = "First Meas Column"
    dataset.data[1].name = "Second Meas Column"

    #print(ds.data[0].spectrum[1])
    print(type(dataset.data))
    plot_time_data(dataset.data)
    plot_spectrum(dataset.data)
    py.show()


if __name__ == "__main__":
    main()

