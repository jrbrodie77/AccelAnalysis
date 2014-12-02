# *This spreadsheet* lets the user import vibration data form the NI accelerometer rig and plot.
# This spreadsheet takes NI .csv file output and plots time domain and frequency spectrums of the accelerometer outputs.

from ni_vibration_lib import *
import pylab as py
import numpy as np
import math


file_to_load = './vibedata.csv'

# <codecell>

# This code loads the csv file and plots
signals = load_ni_csv_file_2ch(file_to_load, verbose=False)

spectrum = fft_abs_norm(signals[0])

spectrum2 = fft_abs_norm(signals[1])

binned_spectrum = bin_spectrum(linear_bins(0, 200, 2.5), spectrum)
binned_spectrum2 = bin_spectrum(linear_bins(0, 200, 2.5), spectrum2)

# Create plot
fig, axes = py.subplots(1, 2)
# fig.set_size_inches(14,6)

axes[0].plot(signals[0][:, 0], signals[0][:, 1], color='blue', label='vibedata1')
# axes[0].plot(signal_2[:,0], signal_2[:,1], color='red', label='vibedata2' )

axes[0].set_title("Raw Signal")
axes[0].set_xlabel("Time(s)")
axes[0].set_ylabel("acceleration, g")

# axes[0].set_xlim( 0,0.5 )
#axes[0].ylim( 1, 2000 )

axes[1].plot(binned_spectrum[:, 0], binned_spectrum[:, 1], color='blue')
#axes[1].plot(binned_spectrum2[:,0], binned_spectrum2[:,1], color='red'  )

axes[1].set_title("Acceleration Spectrum")
axes[1].set_xlabel("frequency(Hz)")
axes[1].set_yscale('log')
axes[1].set_ylabel("acceleration, g")
#fig.tight_layout()

#axes[1].xlim( 3, 200 )
#axes[1].ylim( 1, 2000 )

py.show()

print binned_spectrum
print "2nd series"
print binned_spectrum2