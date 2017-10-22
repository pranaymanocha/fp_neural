'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
:150
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy import stats
from numpy import linalg as LA
import theano
import scipy.io as sio
from keras.layers.convolutional import Conv2D
from keras.models import model_from_json
import sklearn
from sklearn import preprocessing, metrics
from keras import optimizers
from keras.models import load_model
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras import backend as K
from scipy.spatial import distance
from six.moves import cPickle
import tensorflow
from theano.ifelse import ifelse
from theano import tensor as T
from theano.tensor import _shared
import numpy as np
from theano import shared
from keras.utils import plot_model
import pickle

def euclidean_distance(inputs):
    x, y = inputs
    eucl_dist = K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
   #bin_val = K.switch(eucl_dist > 2, theano.tensor.constant(0), theano.tensor.constant(1))
    return eucl_dist
    #assert len(inputs) == 2, \
    #    'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    #u, v = inputs
    #return np.sqrt(M
#    a1= np.array((np.square(x - y)).sum(axis=1, keepdims=True))
#    a1=a1+1e-07
#    a=np.sqrt(a1)
    #x=shared(np.array(a))
 #  from theano.tensor import _shared
  # import numpy as np909a1    #xt = _shared(np.sqrt(a1))
    #x=theano.tensor.dscalar()
    #f = theano.function([x],2*x)
#    return a

   # a3,b3 = T.scalars('a3', 'b3') 
   # x,y = T.matrices('x', 'y')
   # z_lazy = ifelse(T.lt(a3, b3), big_mat=1)
   # f_lazyifelse = theano.function([a, b, x], z_lazy,
  #                             mode=theano.Mode(linker='vm'))
    #f_lazyifelse(a, 1, big_mat1)
   # f_switch(a,1, big_mat1, big_mat2)
   
   # return a
   # x,y = T.matrices('x', 'y')
    #a2,b2 = T.scalars('a2','b2')
    # threshold = b
    #b=0.80 
    #z_lazy = ifelse(T.lt(a2, b2),a2=0,a2=1)
    #f_lazyifelse = theano.function([a2, b2], z_lazy,mode=theano.Mode(linker='vm'))
    #f_switch(a,b)
    #T.scalars
    
    
   # b=T.scalars('np.sqrt(a)')
   # threshold =T.scalars('threshold')
   # z_lazy = ifelse(T.lt(b, threshold), T., T.)
    
   # return np.array(a) 
    
  # a=np.square(x-y).sum()
   # return np.maximum(a,1e-07) 
    #print ("%s" % (x.get_shape()))
    #f = open('obj.save', 'wb')
    #cPickle.dump(K.sqrt(K.maximum(K.sum(K.square(x - y),axis=1, keepdims=True), K.epsilon())), f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()
    #K.sqrt(K.maximum(K.sum(K.square(x - y),axis=1, keepdims=True), K.epsilon()))
    #return np.sqrt(np.maximum(np.sum(np.square(x-y),axis=1,keepdims=True)))
    #a=LA.norm(x-y)
    #return K.maximum(a)
    #return K.sqrt(K.maximum(K.sum(
    #return 
    #return K.sqrt(K.sum(K.square(x - y),axis=1))
    #threshold=np.array([0.8])
    #upper=np.array([1])
    #lower=np.array([0])
    #b[b>threshold] = lower
    #b[b<=threshold] = upper
    #return np.array(b)
    #return np.sqrt(a)
    #var=K.variable(value=dist)
    #dist[dist>threshold] = upper
    #x=np.where(dist>threshold, lower, upper)
    #var=K.variable(value=B)
    #b=sklearn.prep<t_k>

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

#def binary_distance_shpe(shape):
#    return (shape[0],1)
    
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
   
   
def create_pairs(x, digit_indices,qw):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    #a=len(x)
    print ("%d" % qw)
    n = min([len(digit_indices[d]) for d in range(qw)])-1
    print ("%d" % n)
    for d in range(qw):
        for i in range(n):
	    #print(i)
            z1, z2 = digit_indices[d][0], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, qw)
            dn = ((d + inc) % qw)[0]
            io=random.randrange(0,n)
	   #print ("%d" % dn)
            z1, z2 = digit_indices[d][0], digit_indices[dn][io]
            pairs += [[x[z1], x[z2]]]
            labels += [1,0]
    print(labels)
    #print ("%d" % len(pairs))
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''

    #seq=Sequential()
    #seq.add(Conv2D(15,kernel_size=(6,6),activation='relu',input_shape=(input_dim,)))
    #seq.add(Conv2D(, (3, 3), activation='relu'))
    #seq.add(MaxPooling2D(pool_size=(19, 19)))
    #seq.add(Dropout(0.25))
    #seq.add(Conv2D(30,kernel_size=(9,9),activation='relu')   
    #seq.add(Flatten())
    #seq.add(Dense(256, activation='relu'))
    #seq.add(Dropout(0.5))
#   #seq.add(Dense(128, activation='softmax'))
    seq = Sequential()
    seq.add(Dense(512,input_shape=(input_dim,),activation='relu'))
    seq.add(Dropout(0.3))
    seq.add(Dense(256,activation='relu'))
#    seq.add(Dropout(0.2))
    seq.add(Dropout(0.3))
#    seq.add(Dropout(0.2))
    seq.add(Dense(128,activation='relu'))

    #seq.add(Dropout(0.1))
#    seq.add(Flatten())
 #   seq.add(Dense(256,activation='relu'))
    #seq.add(Lambda(euclidean_distance,
     #             output_shape=eucl_dist_output_shape))
   #seq.add(Dropout(0.1))
   # seq.add(Dense(1024,activation='softmax'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    #print ("%d" % labels[predictions.ravel()<0.80].mean())
    return labels[predictions.ravel() <0.1].mean()


#the data, shuffled and split between train and test sets
#print ("A very special dog")
mat_file=sio.loadmat('snippets_from_train.mat')
x_train=mat_file['S']
y_train=mat_file['y_t']

#print ("done %d"% len(x_train))
mat_file_1=sio.loadmat('snippets_from_validation.mat')
x_test=mat_file_1['S']
y_test=mat_file_1['y_t']

#print ("done %d" % y_test)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = mat_S.reshape(2,31293)
#x_test = x_test.reshape(2,784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 13509
epochs = 200

# create training+test positive and negative pairs
a=len(y_train)
a1=y_train[a-1]
#qw2=len(x_train)
#print ("%d" %qw2)
print ('%d' % a1)
digit_indices = [np.where(y_train == i)[0] for i in range(1,a1+1)]
print ("%d" % len(digit_indices))
tr_pairs, tr_y = create_pairs(x_train, digit_indices,a1)
print (" size %d" % len(tr_pairs))
#sio.savemat('train_lib.mat', {'tr_pairs':tr_pairs})
#sio.savemat('train_labels.mat', {'tr_y':tr_y})


b=len(y_test)
b1=y_test[b-1]
digit_indices = [np.where(y_test == i)[0] for i in range(1,b1+1)]
te_pairs, te_y = create_pairs(x_test, digit_indices,b1)
print (" size %d" % len(te_pairs))

# network definition
print('started')
base_network = create_base_network(input_dim)
input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

#x = keras.layers.concatenate([lstm_out, auxiliary_input])

#model=merge([processed_a,processed_b], mode = 'concat')
#model = Dense(512,activation='relu')(model)
#model= Dropout(0.4)(model)
#model= Dense(1,activation='relu')(model)

#model=Dense(1,activation='relu')(model)
#print ("%s" % theano.tensor.shape.eval(processed_a))
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

#keras.layers.merge.Concatenate(axis=-1)
#binary_distance=lambda x: 0 if np.array(x)>1 else 1


#distance1 = Lambda(binary_distance,output_shape=binary_distance_shpe)(np.array(distance))
model = Model([input_a, input_b], distance)

#train
rms = RMSprop()
#sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=False)
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=512,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

model.save('my_model_ref_ref.h5')
#json_string=model.to_json()
#with open("model.json",'w') as json_file:
#	json_file.write(json_string)
#model.save_weights('my_model_weights.h5')
# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])


sio.savemat('np_vector.mat', {'pred':pred})

tr_acc = compute_accuracy(pred, tr_y)
pred1 = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred1, te_y)

print ('Model %s' % model.summary())
#plot_model(model, to_file='model.png')
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

#print(model.summary())

#model = ...  # create the original model
#layer_name = 'my_layer'


intermediate_layer_model = Model(inputs=input_a,
                                 outputs=processed_a)

json_string=intermediate_layer_model.to_json()
with open('model.json','w') as json_file:
	json_file.write(json_string)
intermediate_layer_model.save_weights('my_model_weights.h5')

#intermediate_output = intermediate_layer_model.predict([x_test])
#
#intermediate_layer_model.save('intermediate_model.h5')

#with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
#	pickle.dump([intermediate_output], f)

#print ('intermediate layer output %s' % intermediate_output)
#sio.savemat('np_vector_intermediate.mat', {'intermediate_output':intermediate_output})

fpr, tpr, thresholds = metrics.roc_curve(te_y,pred1,pos_label=1)
print ('%s' % fpr)
A=np.absolute(np.subtract(fpr,1-tpr))
print ('%s' % A)
B=min(A, key=abs)
print ('%s' % B)
#sio.savemat('np_vector.mat', {'pred':pred})
#EER=stats.threshold(np.argmin(abs(fpr-1+tpr)))
#print ('%s' % EE



