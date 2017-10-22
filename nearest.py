import matplotlib
matplotlib.use('Agg')
import numpy
import scipy
from tsne import bh_sne
from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
import scipy.io as sio
import pickle
from sklearn.manifold import TSNE

a=pickle.load(open("intermediate_train_sound.pickle","rb"))
b=numpy.array(a)
b00=numpy.reshape(b,(10,128))

#sio.loadmat('')
a1=pickle.load(open("intermediate_rooster_sound.pickle","rb"))
b1=numpy.array(a1)
b11=numpy.reshape(b1,(10,128))

a2=pickle.load(open("intermediate_airplane.pickle","rb"))
b2=numpy.array(a2)
b22=numpy.reshape(b2,(10,128))


#print(b22.shape)
#print(b.shape)
f=numpy.concatenate((b00,b11,b22),axis=0)
#print(f.shape)
q=pickle.load(open("intermediate_query_train.pickle","rb"))
q1=numpy.array(q)
q22=numpy.reshape(q1,(1,128))
#q23=numpy.repeat(q22,30,axis=0)

#print(q23.shape)
dist=(f-q22)**2
qr=numpy.sum(dist,axis=1)
rq=numpy.sqrt(qr)
wer=numpy.argsort(rq)
print(wer)
count1=0
count2=0
count3=0

for i in range(5):
	if wer[i] >=0 and wer[i]<=9:
		count1=count1+1
	elif wer[i] >=10 and wer[i]<=19:
		count2=count2+1
	elif wer[i] >=20 and wer[i]<=29:
		count3=count3+1
print(count1)
print(count2)
print(count3)
print('\n')




#	print(wer)
#mag=numpy.sqrt(pq.dot(pq))
#dist=scipy.linalg.norm(f-q23)
#print(mag.shape)
#print(q22.shape)

#f1=numpy.reshape(f,(30,128))
#a=numpy.empty([30,1])
#for i in range(30):
#	if i>=0 and i<=9:
#		a[i,0]=0
#	elif i>=10 and i<=19:
#		a[i,0]=1
#	elif i>=20 and i<=29:
#		a[i,0]=2
#print(a)
#f1=f.astype('float64')
#X_embedded=TSNE(n_components=2,perplexity=7).fit_transform(f1)
#print(X_embedded.shape)
#vis_data=bh_sne(f1,perplexity=10000)
#fig=plt.figure()
#vis_x=X_embedded[:,0]
#vis_y=X_embedded[:,1]
#plt.scatter(vis_x,vis_y,c=a,cmap=plt.cm.get_cmap("jet",3))
#plt.colorbar(ticks=range(3))
#savefig('foo.png')
#plt.show()
#plt.savefig('foo.png')
#plt.savefig('cluster.png')
#print(f1[1,:])
#fig.savefig('plot.png')
