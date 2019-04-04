import os
import re
import sys
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score

import argparse


from loader import *
from train import *


def main():
	dataset = sys.argv[1]
	if(dataset=='rt'):
		data,labels = loadrt()
	elif(dataset=='so'):
		data,labels = loadso()
	elif(dataset=='mpqa'):
		data,labels = loadmpqa()
	elif(dataset=='cr'):
		data,labels = loadcr()
	elif(dataset=='trec'):
		Xtrain,ytrain,Xtest,ytest = loadtrec()

	embed_matrix,embed_dim = load_embeddings()

	trainall(data,labels,embed_matrix,embed_dim,name=dataset,maxlen=10)


if __name__ == '__main__':
	main()