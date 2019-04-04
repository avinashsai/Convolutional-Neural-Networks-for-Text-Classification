import os
import re
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
import copy

from convert import *
from model import *

def get_loss(net,loader,device):
	net.eval()
	with torch.no_grad():
		val_loss = 0.0
		for inds,lbs in loader:
			inds = inds.long().to(device)
			lbs = lbs.long().to(device)

			out = net(inds)

			curloss = F.cross_entropy(out,lbs)
			val_loss+=curloss.item()
		return val_loss/len(loader)


def get_acc(net,loader,device):
	net.eval()
	with torch.no_grad():
		val_acc = 0
		total = 0
		for inds,lbs in loader:
			inds = inds.long().to(device)
			lbs = lbs.long().to(device)

			out = net(inds)

			total+=inds.size(0)
			preds = torch.max(out,1)[1]
			val_acc+=torch.sum(preds==lbs.data).item()

		return (val_acc/total)*100


def getloaders(trainind,valind,testind,train_labels,val_labels,test_labels,batchsize):

	trainarray = torch.utils.data.TensorDataset(trainind,train_labels)
	trainloader = torch.utils.data.DataLoader(trainarray,batch_size=batchsize)

	valarray = torch.utils.data.TensorDataset(valind,val_labels)
	valloader = torch.utils.data.DataLoader(valarray,batch_size=batchsize)

	testarray = torch.utils.data.TensorDataset(testind,test_labels)
	testloader = torch.utils.data.DataLoader(testarray,batch_size=batchsize)

	return trainloader,valloader,testloader



def trainmodel(net,train_loader,val_loader,test_loader,numepochs,device):
	numepochs = 3

	optimizer = torch.optim.Adadelta(net.parameters())
	criterion = nn.CrossEntropyLoss()

	best_model_wts = copy.deepcopy(net.state_dict())

	for epoch in range(numepochs):
		net.train()
		curloss = 0.0
		val_best_loss = np.Inf
		for indices,labels in train_loader:
			indices = indices.long().to(device)
			labels = labels.long().to(device)

			net.zero_grad()

			output = net(indices)

			loss = criterion(output,labels)
			curloss+=loss

			loss.backward()
			clip_grad_norm_(net.parameters(),3)
			optimizer.step()

		print("Epoch {} Loss {} ".format(epoch+1,loss/len(train_loader)))
		valloss = get_loss(net,val_loader,device)
		if(valloss<val_best_loss):
			val_best_loss = valloss
			best_model_wts = copy.deepcopy(net.state_dict())

	net.load_state_dict(best_model_wts)

	testacc = get_acc(net,test_loader,device)

	return testacc



def trainall(data,labels,embed_matrix,embed_dim,name,maxlen,numclasses=2,numepochs=10,batchsize=32):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	rand_acc = 0.0
	static_acc = 0.0
	nonstatic_acc = 0.0
	multichannel_acc = 0.0

	nsplits = 10
	count = 1
	kf = StratifiedKFold(n_splits=nsplits,random_state=0)
	for train_index,test_index in kf.split(data,labels):
		Xtrain = [data[i] for i in train_index]
		ytrain = labels[train_index]

		Xtest = [data[i] for i in test_index]
		ytest = labels[test_index]

		Xtrain,Xval,ytrain,yval = train_test_split(Xtrain,ytrain,test_size=0.1,random_state=0)

		train_labels = torch.from_numpy(np.asarray(ytrain))
		val_labels = torch.from_numpy(np.asarray(yval))
		test_labels = torch.from_numpy(np.asarray(ytest))

		trainind,valind,testind,vocablen,embedmatrix = generate_indices(Xtrain,Xval,Xtest,embed_matrix,embed_dim,maxlen)

		trainloader,valloader,testloader = getloaders(trainind,valind,testind,train_labels,val_labels,test_labels,batchsize)

		print("Training Fold {} ".format(count))

		print("Training Random Model ")

		randmodel = cnn_rand(vocablen,maxlen,embed_dim,numclasses)

		rand_acc+=trainmodel(randmodel,trainloader,valloader,testloader,numepochs,device)


		print("---------------")

		print("Training Static Model ")

		staticmodel = cnn_static(maxlen,embedmatrix,numclasses,embed_dim)

		static_acc+=trainmodel(staticmodel,trainloader,valloader,testloader,numepochs,device)


		print("---------------")

		print("Training Non Static Model ")

		nonstaticmodel = cnn_nonstatic(maxlen,embedmatrix,numclasses,embed_dim)

		nonstatic_acc+=trainmodel(nonstaticmodel,trainloader,valloader,testloader,numepochs,device)

		print("---------------")

		print("Training Multi Channel Model ")

		multichannelmodel = cnn_multichannel(maxlen,embedmatrix,numclasses,embed_dim)

		multichannel_acc+=trainmodel(multichannelmodel,trainloader,valloader,testloader,numepochs,device)

		print("----------------")

		count+=1

	print("Random Model Accuracy {} ".format(rand_acc/nsplits))

	print("Static Model Accuracy {} ".format(static_acc/nsplits))

	print("Non-Static Model Accuracy {} ".format(nonstatic_acc/nsplits))

	print("Multi Channel Model Accuracy {} ".format(multichannel_acc/nsplits))