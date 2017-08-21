from __future__ import division
import random
import numpy as np 
from keras import losses
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model 
from keras.layers import SimpleRNN, LSTM, Embedding
from keras.layers import Dense, Input, Lambda, Merge, merge
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pdb

from preprocessing import *

RNNTYPE = 0
MAX_SEQUENCE_LENGTH = 160
BATCH_SIZE = 256
EPOCHS = 10
input_dim = (BATCH_SIZE,100)

def create_base_network(input_dim, rnntype):
	seq = Sequential()
	seq.add(Embedding(len(word_index.item()) + 1,
					  EMBEDDING_DIM,
					  weights = [embedding_matrix],
					  input_length = MAX_SEQUENCE_LENGTH,
					  trainable=False))
	if rnntype == 0:
		seq.add(SimpleRNN(50, input_shape=(input_dim), unroll=True))
	if rnntype == 1:
		seq.add(LSTM(200, input_shape=(input_dim), unroll=True))
	return seq

def b_init(shape,name=None):
	values = np.random.normal(loc=0.5, scale=1e-02, size=shape)
	return K.variable(values, name=name)

def get_data(s,f,data):
	print 'Fetching validation/testing data ...'
	contexts = np.zeros([f*10,MAX_SEQUENCE_LENGTH])
	responses = np.zeros([f*10,MAX_SEQUENCE_LENGTH])
	labels = []
	cnt = 0
	con = data[s:s+11*f:11]
	for i in range(f):
		for j in range(1,11):
			response = data[s+(i*10+j)]
			contexts[cnt] = con[i]
			responses[cnt] = response
			cnt += 1
			if j==1:
				labels.append(1)
			else:
				labels.append(0)

	return contexts,responses,labels

def get_labels(n):
	print 'Fetching training labels ...'
	samples = []
	labels = []
	with open('./train.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			# if len(samples) > n:
			# 	break
		for i in range(1,len(samples)):
			labels.append(float(samples[i][2]))

	return labels


data, word_index, embedding_matrix = compute_embedding_matrix()
np.save('word_index.npy',word_index)
np.save('embedding_matrix.npy',embedding_matrix)
np.save('data.npy',data)
# data = np.load('data.npy')
# word_index = np.load('word_index.npy')
# embedding_matrix = np.load('embedding_matrix.npy')

def train():
	base_network = create_base_network(input_dim,RNNTYPE)
	input_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
	input_b = Input(shape=(MAX_SEQUENCE_LENGTH,))

	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	similarity = merge([Dense(int(processed_b.shape[1]), bias=False)(processed_a), processed_b], mode='dot')
	prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(similarity)

	model = Model([input_a,input_b], output=prediction)

	adam = Adam()
	model.compile(loss=losses.binary_crossentropy, optimizer=adam)

	train_a = data[0:2000000:2]
	train_b = data[1:2000000:2]
	labels = get_labels(1000000)
	valid_contexts, valid_responses, valid_labels = get_data(2000000,19560,data)

	filepath = './models/RNN/weights.{epoch:02d}-{val_loss:.5f}.h5'
	checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
	stopping = EarlyStopping(monitor='val_loss', min_delta=1e-05, patience=2, verbose=1, mode='min')

	model.fit([train_a,train_b],np.array(labels).reshape(-1,1), epochs=EPOCHS, batch_size=BATCH_SIZE, 
			   validation_data=([valid_contexts,valid_responses],np.array(valid_labels).reshape(-1,1)),
			   callbacks = [checkpointer],#stopping],
			   initial_epoch=0)

	model.save('models/dual_encoder_%d.h5'%(RNNTYPE))

def test():
	from keras.models import load_model
	base_network = create_base_network(input_dim,RNNTYPE)
	input_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
	input_b = Input(shape=(MAX_SEQUENCE_LENGTH,))

	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	similarity = merge([Dense(int(processed_b.shape[1]), bias=False)(processed_a), processed_b], mode='dot')
	prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(similarity)

	model = Model([input_a,input_b], output=prediction)
	# model.load_weights('./models/RNN/weights.01-0.61147.h5')

	test_contexts, test_responses, test_labels = get_data(2000000+11*19560,18920,data)
	print 'Done fetching data.'

	predictions = model.predict([test_contexts,test_responses])
	print 'Done predicting.'
	old_context = test_contexts[0]
	top1 = 0
	top3 = 0
	total = 0
	for i in range(0,len(test_contexts),10):
		total += 1
		top3idx = (predictions[i:i+10].reshape(1,10)[0]).argsort()[-3:][::-1]
		if 0 in top3idx:
			top3 += 1
		if top3idx[0] == 0:
			top1 += 1
	print 'Total: %d\nTop1: %d\nTop3: %d'%(total,top1,top3)
	top1 = top1/total
	top3 = top3/total
	print 'Classification accuracy for RNN %d is %f'%(RNNTYPE,top1)
	print 'Accuracy for top 3 is %f'%(top3)

train()
print 'Training complete.'
# test()
