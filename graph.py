import matplotlib
matplotlib.use('Agg')
import numpy
from tsne import bh_sne
from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
import scipy.io as sio
import pickle
from sklearn.manifold import TSNE
path='/usr2/elizalde/NELS/users/rohan/fingerprinting/fp_NN_all/'
a=pickle.load(open(path+'filespickle/train_sound.pickle',"rb"))
b=numpy.array(a)
bh1=b.shape[1]
b00=numpy.reshape(b,(bh1,128))

#sio.loadmat('')
a1=pickle.load(open(path+'filespickle/rooster_sound.pickle',"rb"))
b1=numpy.array(a1)
b1_pram=b1.shape[1]
b11=numpy.reshape(b1,(b1_pram,128))

a2=pickle.load(open(path+'filespickle/airplane_sound.pickle',"rb"))
b2=numpy.array(a2)
b221=b2.shape[1]
b22=numpy.reshape(b2,(b221,128))

#print(type(b221))
print(bh1)
print(b1_pram)
print(b221)
#print(b22.shape)
#print(b.shape)
f=numpy.concatenate((b00,b11,b22),axis=0)
print(f.shape)

#f1=numpy.reshape(f,(30,128))
a=numpy.empty([f.shape[0],1])
print(f.shape[0])
for i in range(f.shape[0]):
	print(i)
	if i>=0 and i<bh1:
		a[i]=0
	elif i>=bh1 and i<b1_pram+bh1:
		a[i]=1
	elif i>=b1_pram+bh1 and i<b221+bh1+b1_pram:
		a[i]=2
#a32=a.astype('float64')
print(a)
#f1=f.astype('float64')
X_embedded=TSNE(n_components=2,perplexity=1).fit_transform(f)
#print(X_embedded.shape)
#vis_data=bh_sne(f1,perplexity=10000)
#fig=plt.figure()
vis_x=X_embedded[:,0]
vis_y=X_embedded[:,1]
plt.scatter(vis_x,vis_y,c=a,cmap=plt.cm.get_cmap("jet",3))
#plt.colorbar(ticks=range(3))
#savefig('foo.png')
plt.show()
plt.savefig('foo.png')
#plt.savefig('cluster.png')
#print(f1[1,:])
#fig.savefig('plot.png')
