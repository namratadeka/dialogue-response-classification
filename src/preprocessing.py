import csv
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pdb

MAX_NB_WORDS = 1000000
MAX_SEQUENCE_LENGTH = 160
EMBEDDING_DIM = 100

def prepare():
	print 'Reading data ...'
	texts = []
	samples = []
	with open('../data/train.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			# if len(samples) > 200000:
			# 	break
		for i in range(1,len(samples)):
			texts.append(samples[i][0])
			texts.append(samples[i][1])
	print len(texts)
	samples = []
	with open('../data/valid.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			# if len(samples) > 40000:
			# 	break
		for i in range(1,len(samples)):
			for j in range(11):
				texts.append(samples[i][j])
	samples = []
	with open('../data/test.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			# if len(samples) > 40000:
			# 	break
		for i in range(1,len(samples)):
			for j in range(11):
				texts.append(samples[i][j])

	return texts


def tokenize_and_pad():
	print 'Tokenizing and padding ...'
	texts = prepare()
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	print 'Found %s unique tokens.'%len(word_index)

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	print 'Shape of data tensor:',data.shape

	return data, word_index

def compute_embedding_map():
	print 'Computing embedding map ...'
	embeddings_index = {}
	f = open('./data/glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print 'Found %s word vectors.'%len(embeddings_index)

	return embeddings_index

def compute_embedding_matrix():
	print 'Computing embedding matrix ...'
	data, word_index = tokenize_and_pad()
	embeddings_index = compute_embedding_map()
	embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return data, word_index, embedding_matrix
