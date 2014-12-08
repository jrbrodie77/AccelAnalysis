from unittest import TestCase
import unittest
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

    def test_dc(self):
        values = [3, 3, 3, 3, 3, 3, 3, 3]
        times = [0,1,2,3,4,5,6,7]
        fft = ni.calc_fft([times,values], 8)
        self.assertListEqual(fft[0].tolist(), [ 0.0,  1.0,  2.0,  3.0,  4.0], "Frequencies wrong or wrong type")

    def test_gen_test_tones(self):

        times, values=ni.gen_test_tones()






#a = [3,3,3,3,-3,-3,-3,-3]
#a = [1,-1,1,-1,1,-1,1,-1]
#a = [3,3,3,3,3,3,3,3]
#b= [0,1,2,3,4,5,6,7]
#ds1=DataSeries(b,a)
#ds2 = gen_test_tones_8()

#print(ds2.fft_abs)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalc_fft)
    unittest.TextTestRunner(verbosity=2).run(suite)