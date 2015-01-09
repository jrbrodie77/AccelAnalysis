from vibes import *

import matplotlib.pyplot as py
import numpy as np
import math
import sys


def main():
    dataset =  load_1col_txt_file(filename = "warehouse_table1_up-down_nofan_010715.txt")
    dataset[0].name = "First Meas Column"

    plot_time_series(dataset)
    plot_spectrum(dataset)
    plot_binned_spectrum(dataset)
    py.show()

main()
