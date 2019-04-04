import torch
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data


numfilters = 100
filtersizes = [3,4,5]


### Random Model

class cnn_rand(nn.Module):
	def __init__(self,vocablen,maxlen,embeddim,numclasses):
		super(cnn_rand,self).__init__()
		self.numfilters = numfilters
		self.filtersizes = filtersizes
		self.vocablen = vocablen
		self.embeddim = embeddim
		self.numclasses = numclasses
		self.embed = nn.Embedding(self.vocablen,self.embeddim)
		self.conv1 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[0])
		self.conv2 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[1])
		self.conv3 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[2])
		self.pool1 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[0]+1))
		self.pool2 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[1]+1))
		self.pool3 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[2]+1))
		self.dense = nn.Linear(numfilters*3,self.numclasses)
		self.act = nn.ReLU()
		self.drop = nn.Dropout(0.5)

	def forward(self,x):
		x = self.embed(x)
		x = x.transpose(1,2)
		out1 = self.act(self.conv1(x))
		out2 = self.act(self.conv2(x))
		out3 = self.act(self.conv3(x))

		out1 = self.pool1(out1)
		out2 = self.pool2(out2)
		out3 = self.pool3(out3)
		
		out = torch.cat([out1,out2,out3],dim=1)
		out = out.view(out.size(0),-1)
		out = self.drop(out)
		out = self.dense(out)

		return out

### Static Model

class cnn_static(nn.Module):
	def __init__(self,maxlen,embedmatrix,numclasses,embeddim):
		super(cnn_static,self).__init__()
		self.numfilters = numfilters
		self.filtersizes = filtersizes
		self.embeddim = embeddim
		self.numclasses = numclasses
		self.embed = nn.Embedding.from_pretrained(embedmatrix)
		self.conv1 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[0])
		self.conv2 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[1])
		self.conv3 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[2])
		self.pool1 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[0]+1))
		self.pool2 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[1]+1))
		self.pool3 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[2]+1))
		self.dense = nn.Linear(numfilters*3,self.numclasses)
		self.act = nn.ReLU()
		self.drop = nn.Dropout(0.5)

	def forward(self,x):
		x = self.embed(x)
		x = x.transpose(1,2)
		out1 = self.act(self.conv1(x))
		out2 = self.act(self.conv2(x))
		out3 = self.act(self.conv3(x))

		out1 = self.pool1(out1)
		out2 = self.pool2(out2)
		out3 = self.pool3(out3)

		out = torch.cat([out1,out2,out3],dim=1)
		out = out.view(out.size(0),-1)
		out = self.drop(out)
		out = self.dense(out)

		return out

### Non-Static Model

class cnn_nonstatic(nn.Module):
	def __init__(self,maxlen,embedmatrix,numclasses,embeddim):
		super(cnn_nonstatic,self).__init__()
		self.numfilters = numfilters
		self.filtersizes = filtersizes
		self.embeddim = embeddim
		self.numclasses = numclasses
		self.embed = nn.Embedding.from_pretrained(embedmatrix,freeze=False)
		self.conv1 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[0])
		self.conv2 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[1])
		self.conv3 = nn.Conv1d(self.embeddim,self.numfilters,kernel_size=self.filtersizes[2])
		self.pool1 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[0]+1))
		self.pool2 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[1]+1))
		self.pool3 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[2]+1))
		self.dense = nn.Linear(numfilters*3,self.numclasses)
		self.act = nn.ReLU()
		self.drop = nn.Dropout(0.5)

	def forward(self,x):
		x = self.embed(x)
		x = x.transpose(1,2)
		out1 = self.act(self.conv1(x))
		out2 = self.act(self.conv2(x))
		out3 = self.act(self.conv3(x))

		out1 = self.pool1(out1)
		out2 = self.pool2(out2)
		out3 = self.pool3(out3)

		out = torch.cat([out1,out2,out3],dim=1)
		out = out.view(out.size(0),-1)
		out = self.drop(out)
		out = self.dense(out)

		return out

##  Multi Channel Model

class cnn_multichannel(nn.Module):
	def __init__(self,maxlen,embedmatrix,numclasses,embeddim):
		super(cnn_multichannel,self).__init__()
		self.numfilters = numfilters
		self.filtersizes = filtersizes
		self.embeddim = embeddim
		self.numclasses = numclasses
		self.embed1 = nn.Embedding.from_pretrained(embedmatrix,freeze=False)
		self.embed2 = nn.Embedding.from_pretrained(embedmatrix,freeze=True)
		self.conv1 = nn.Conv1d(self.embeddim*2,self.numfilters,kernel_size=self.filtersizes[0])
		self.conv2 = nn.Conv1d(self.embeddim*2,self.numfilters,kernel_size=self.filtersizes[1])
		self.conv3 = nn.Conv1d(self.embeddim*2,self.numfilters,kernel_size=self.filtersizes[2])
		self.pool1 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[0]+1))
		self.pool2 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[1]+1))
		self.pool3 = nn.MaxPool1d(kernel_size=(maxlen-filtersizes[2]+1))
		self.dense = nn.Linear(numfilters*3,self.numclasses)
		self.act = nn.ReLU()
		self.drop = nn.Dropout(0.5)

	def forward(self,x):
		x1 = self.embed1(x)
		x2 = self.embed2(x)
		xcom = torch.cat([x1,x2],dim=-1)
		xcom = xcom.transpose(1,2)

		out1 = self.act(self.conv1(xcom))
		out2 = self.act(self.conv2(xcom))
		out3 = self.act(self.conv3(xcom))

		out1 = self.pool1(out1)
		out2 = self.pool2(out2)
		out3 = self.pool3(out3)

		out = torch.cat([out1,out2,out3],dim=1)
		out = out.view(out.size(0),-1)
		out = self.drop(out)
		out = self.dense(out)

		return out