# import csv
#
#
# with open('vibedata.csv', 'rb') as f:
#
# file_contents = f.readlines()
#
# reader = csv.reader(file_contents, delimiter=',')
# header=[]
#
# for row in range(8):
# header.append(reader.next())
#

import sys
sys.dont_write_bytecode = True

from ni_vibration_lib import *
ds = DataSet('vibedata.csv')


cs = calc_spectrum(ds.data[0].fft_data)


freq_data = cs
bin_spec=linear_bins()

binned_keys = np.digitize(freq_data[0], bin_spec[1], right=True)

print(bin_spec[1][0:10])
print(binned_keys[0:10])

binned_spectrum = {}

for index, bin_num in enumerate(binned_keys):
    if bin_num < len(bin_spec[1]):
        freq = bin_spec[0][bin_num]
        if not binned_spectrum.has_key(freq):
            binned_spectrum[freq] = 0

        binned_spectrum[freq] += freq_data[1][index]

sorted_args = np.array(binned_spectrum[0]).argsort() #create sort order for freqs

binned_spectrum[0] = [binned_spectrum[0][index] for index in sorted_args]
binned_spectrum[1] = [binned_spectrum[1][index] for index in sorted_args]

# binned_spectrum = [[v[0], v[1]] for v in binned_spectrum]
# binned_spectrum = np.array(binned_spectrum)
# freqs, values = binned_spectrum[0], binned_spectrum[1]
# return (freqs, values)
