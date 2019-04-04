import os
import re
import numpy as np 
import torch
import collections
from collections import Counter


def preparevocab(corpus):
	words = []
	for sentence in corpus:
		words+=sentence.split()

	allwords = Counter(words).most_common()

	vocabulary = {}
	vocabulary['<PAD>'] = 0
	index = 1
	for word,_ in allwords:
		vocabulary[word] = index
		index+=1

	return vocabulary


def get_matrix(vocab,embedmatrix,embeddim):
	embeddingmatrix =  torch.zeros(len(vocab),embeddim)
	
	for word,i in enumerate(list(vocab.keys())):
		if(word in embedmatrix):
			embeddingmatrix[i] = embedmatrix[word]
	return embeddingmatrix

def get_indices(vocabulary,corpus,maxlen):
	curind =  torch.zeros(len(corpus),maxlen)
	for i in range(len(corpus)):
		corpusind = [vocabulary[word] for word in corpus[i].split() if word in vocabulary]
		padind = [0]*maxlen
		curlen = len(corpusind)
		if(maxlen-curlen<0):
			padind = corpusind[:maxlen]
		else:
			padind[maxlen-curlen:] = corpusind
		curind[i] = torch.from_numpy(np.asarray(padind,dtype='int32'))

	return curind



def generate_indices(train,val,test,embedmatrix,embeddim,maxlen):
	vocabulary = preparevocab(train)

	embeddingmatrix = get_matrix(vocabulary,embedmatrix,embeddim)

	trainind = get_indices(vocabulary,train,maxlen)
	testind = get_indices(vocabulary,test,maxlen)
	valind = get_indices(vocabulary,val,maxlen)

	return trainind,valind,testind,len(vocabulary),embeddingmatrix