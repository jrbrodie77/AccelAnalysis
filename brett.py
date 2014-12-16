import numpy as np
import matplotlib.pyplot as py
import ni_vibration_lib as ni


np.set_printoptions(precision=6)  #Set the number of decimal places when numpy numbers are printed

dataset = ni.DataSet('vibedata.csv')
dataset.data[0].name = "Accel on table"
dataset.data[1].name = "Accel on shelf"
ni.plot_time_data(dataset.data)

a=np.matrix([1,2,3,4])
b=np.matrix([5,6,7,8])
print("The product of matrices a and b")
print( a.T * b )  #print the matrix multiply of the two vectors

py.show()  #this shows graphs.


