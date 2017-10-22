from keras.models import load_model
import scipy.io as sio
import pickle

import numpy#import code
from keras.models import model_from_json
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

mat_file=sio.loadmat('76_queries_new.mat')
x_int_out=mat_file['S']
x1=mat_file['y_t']
x2=numpy.array(x1)
#mat_file1=sio.loadmat('snippets_from_query_rooster_2.mat')
#x_int_out1=mat_file1['S']

#x2=numpy.array(x_int_out1)
#red1=loaded_model.predict([x1])

pred=loaded_model.predict([x_int_out])

pred12=numpy.array(pred)
#print(pred==pred1)
#intermediate_layer_model=loaded_model(inputs=input_a, outputs=processed_a)
#intermediate_output=intermediate_layer_model.predict([x_int_out])

#print(pred==pred1)
with open('intermediate_76_queries_new.pickle','w') as f:
	pickle.dump([pred12,x2],f)

