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

#print(ds.data[0].esd_data.T)


print bin_spectrum(ds.data[0].esd_data)