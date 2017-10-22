import pickle
import numpy
import scipy.io as sio

a=pickle.load(open("intermediate_query_rooster_2.pickle","rb"))
b=numpy.array(a)
print(b)

#sio.savemat('train_sound.mat',{'b':b})
a1=pickle.load(open("intermediate_query_rooster_1.pickle","rb"))
b1=numpy.array(a1)
print(b1)

print(b==b1)

