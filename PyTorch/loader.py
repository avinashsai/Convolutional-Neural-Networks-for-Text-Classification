import re
import numpy as np
import torch

def clean_str(text,TREC=False):
	return text


datapath = '../Datasets/'
vecpath = '../glove.840B.300d.txt'

def loadrt():
	corpus = []
	poscount = 0
	negcount = 0
	with open(datapath+'rt-polarity/rt-polarity.pos','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			poscount+=1

	with open(datapath+'rt-polarity/rt-polarity.neg','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			negcount+=1

	labels = np.zeros(poscount+negcount)
	labels[:poscount] = 1
	print("Training Data Loaded ")
	return corpus,labels


def loadso():
	corpus = []
	poscount = 0
	negcount = 0
	with open(datapath+'Subj_Obj/plot.tok.gt9.5000','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			poscount+=1

	with open(datapath+'Subj_Obj/quote.tok.gt9.5000','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			negcount+=1

	labels = np.zeros(poscount+negcount)
	labels[:poscount] = 1
	return corpus,labels


def loadmpqa():
	corpus = []
	poscount = 0
	negcount = 0
	with open(datapath+'mpqa/mpqa.pos','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			poscount+=1

	with open(datapath+'mpqa/mpqa.neg','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			negcount+=1

	labels = np.zeros(poscount+negcount)
	labels[:poscount] = 1
	return corpus,labels


def loadcr():
	corpus = []
	poscount = 0
	negcount = 0
	with open(datapath+'cr/custrev.pos','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			poscount+=1

	with open(datapath+'cr/custrev.neg','r',encoding='latin1') as f:
		for line in f.readlines():
			corpus.append(clean_str(line[:-1]))
			negcount+=1

	labels = np.zeros(poscount+negcount)
	labels[:poscount] = 1
	return corpus,labels


def loadtrec():
	lab = {'DESC':0,'ENTY':1,'ABBR':2,'HUM':3,'LOC':4,'NUM':5}

	Xtrain = []
	ytrain = []
	with open(datapath+'TREC/train.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			words = line.split()
			label = words[0][0:words[0].find(":")]
			sentence = " ".join(words[1:])
			ytrain.append(int(lab[label]))
			Xtrain.append(clean_str(sentence,True))

	Xtest = []
	ytest = []
	with open(datapath+'TREC/test.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			words = line.split()
			label = words[0][0:words[0].find(":")]
			sentence = " ".join(words[1:])
			ytest.append(int(lab[label]))
			Xtest.append(clean_str(sentence,True))

	return Xtrain,np.asarray(ytrain),Xtest,np.asarray(ytest)


def load_embeddings():

	embedding_index = {}
	with open(vecpath,'r',encoding='utf-8') as f:
		for line in f.readlines():
			words = line.split()
			word = words[0]
			vector = torch.FloatTensor(np.asarray(words[1:],'float32'))
			embedding_index[word] = vector

		embed_dim = vector.size(0)

	return embedding_index,embed_dim