from unittest import TestCase
import unittest
import numpy as np
import ni_vibration_lib as ni

__author__ = 'jrbrodie77'



def gen_test_tones_8():
    """This generates a signal for our test case"""
    samprate = 8
    num_samples = 8
    freqs = [2]

    times = np.arange(0, num_samples) * 1.0 / samprate  #Generate 1 second of time data
    values = np.zeros(len(u))

    for freq in freqs:
        y += [math.sin(num * 2 * math.pi * freq) for num in u]  #*2*freq


    return (times, values)


class TestCalc_fft(unittest.TestCase):
    # def test_fft_dc(self):
    #     values = [3, 3, 3, 3, 3, 3, 3, 3]
    #     times = [0,1,2,3,4,5,6,7]
    #     fft = ni.calc_fft([times,values], 8)
    #     self.assertListEqual(fft[0].tolist(), [ 0.0,  1.0,  2.0,  3.0,  4.0], "Frequencies wrong or wrong type")

    #def test_gen_test_tones(self):
    #    times, values=ni.gen_test_tones()

    def test_bin_spectrum(self):
        freqs = np.arange(0,200,2.5)
        values=np.array([1.0]*len(freqs))
        values[15]=10.5

        freq_series = ni.FreqSeries(freqs, values)
        binned_spectrum = ni.bin_spectrum(freq_series)

        print( binned_spectrum['freqs'])
        self.assertEqual(values.sum(), binned_spectrum.values.sum(), "Total summation doesn't add up")

    def test_load_csv(self):
        filename = 'vibedata.csv'
        series_list = ni.SeriesList(filename = 'vibedata.csv')
        print(series_list)

    def test_series_list_w_initial_series(self):
        freqs = np.arange(0,200,2.5)
        values=np.array([1.0]*len(freqs))
        values[15]=10.5
        freq_series = ni.FreqSeries(freqs, values)
        series_list = ni.SeriesList()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalc_fft)
    unittest.TextTestRunner(verbosity=2).run(suite)