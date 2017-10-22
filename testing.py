from keras.models import load_model
import scipy.io as sio
import pickle
import numpy
#import code
from keras.models import model_from_json
import glob

def contrastive_loss(y_true,y_pred):
	margin=1
	return K.mean(y_true * K.square(y_pred)+(1-y_true)*K.square(K.maximum(margin - y_pred,0)))

json_file=open('model.json','r')

loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights('my_model_weights.h5')
#model=model_from_json(json_string)
#model.load_weights('my_model_weights.h5')
#Model=load_model('my_model_ref_ref.h5')

a=glob.glob('filesmat/*.mat')
#print(a)
path='/usr2/elizalde/NELS/users/rohan/fingerprinting/fp_NN_all/'
for i in range(len(a)):
	print(i)
	fi=path+a[i]
	print(fi)
	mat_file=sio.loadmat(fi)
	x_int_out=mat_file['S']
#	x1=numpy.array(x_int_out)
	pred1=loaded_model.predict([x_int_out])
#intermediate_layer_model=loaded_model(inputs=input_a, outputs=processed_a)
#intermediate_output=intermediate_layer_model.predict([x_int_out])

	#print(pred)
      # 	as=a[1]
#	as1=as.replace('.mat','')
	pred=numpy.array(pred1)
	pi=path+'filespickle'+'/'+a[i].replace('filesmat/snippets_from_intermediate_outputs_','').replace('.mat','')+'.pickle'
	print(pi)
	with open(pi,'w') as f:
		pickle.dump([pred],f)

