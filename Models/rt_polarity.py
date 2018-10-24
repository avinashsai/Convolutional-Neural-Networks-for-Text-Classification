import nltk

import tensorflow as tf
import keras
from keras.layers import Dense,Input
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Model,Sequential
from keras import metrics
from keras.layers import Embedding
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate,Concatenate,Flatten

import sys
import os
import re
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')

def preprocess(text):
  text = re.sub(r"it\'s","it is",str(text))
  text = re.sub(r"i\'d","i would",str(text))
  text = re.sub(r"don\'t","do not",str(text))
  text = re.sub(r"he\'s","he is",str(text))
  text = re.sub(r"there\'s","there is",str(text))
  text = re.sub(r"that\'s","that is",str(text))
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"cannot", "can not ", text)
  text = re.sub(r"what\'s", "what is", text)
  text = re.sub(r"What\'s", "what is", text)
  text = re.sub(r"\'ve ", " have ", text)
  text = re.sub(r"n\'t", " not ", text)
  text = re.sub(r"i\'m", "i am ", text)
  text = re.sub(r"I\'m", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'s"," is",text)
  text = re.sub(r"[^a-zA-Z]"," ",str(text))
  sents = word_tokenize(text)
  return " ".join(word.lower() for word in sents if word.lower() not in stopword)

def load_data(path):
  pos_file = path+'rt-polarity.pos'
  neg_file = path+'rt-polarity.neg'
    
  corpus = []
  with open(pos_file,'r',encoding='latin1') as f:
    for each_line in f:
      corpus.append(preprocess(each_line))
  

  with open(neg_file,'r',encoding='latin1') as f:
    for each_line in f:
      corpus.append(preprocess(each_line))
      
  return corpus

path = '../Datasets/rt-polarity/'

corpus = load_data(path)

labels = np.zeros(10662)
labels[0:5331] = 1

Xtrain,Xtest,ytrain,ytest = train_test_split(corpus,labels,test_size=0.2,random_state=42)

Xtrain,ytrain = shuffle(Xtrain,ytrain)
Xtest,ytest = shuffle(Xtest,ytest)


tokenizer = Tokenizer(num_words=18765)
tokenizer.fit_on_texts(Xtrain)

word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(Xtrain)
train_indices = pad_sequences(train_seq,maxlen=20)

test_seq = tokenizer.texts_to_sequences(Xtest)
test_indices = pad_sequences(test_seq,maxlen=20)

GLOVEDIR = ''

embeddings_index = {}
f = open(os.path.join(GLOVEDIR,'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
      embedding_matrix[i] = np.random.rand(300)

def conv_model(embedding_layer):
  inp = Input(shape=(20,))
  embedding_out = embedding_layer(inp)
  
  
  conv1 = Conv1D(100,3,activation='relu')(embedding_out)
  pool1 = MaxPooling1D(2,padding='valid')(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu')(embedding_out)
  pool2 = MaxPooling1D(2,padding='valid')(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu')(embedding_out)
  pool3 = MaxPooling1D(2,padding='valid')(conv3)
  out3 = Flatten()(pool3)
  
  concat_out = Concatenate()([out1,out2,out3])
  
  drop = Dropout(0.5)(concat_out)
  
  final_out = Dense(1,activation='sigmoid')(drop)
  
  model = Model(inputs=inp,outputs=final_out)
  
  return model

def train_rand_model():
  
  embedding_random = np.random.rand(len(word_index)+1,300)
  embedding_layer = Embedding(input_dim=len(word_index)+1,output_dim=300,weights=[embedding_random],trainable=True)
  return conv_model(embedding_layer)

model_rand = train_rand_model()

model_rand.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

model_rand.fit(train_indices,ytrain,batch_size=50,epochs=50,shuffle=True,verbose=1)

model_rand_acc = model_rando

def train_static_model():
  embedding_layer = Embedding(input_dim=len(word_index)+1,output_dim=300,weights=[embedding_matrix],trainable=False)
  return conv_model(embedding_layer)

model_static = train_static_model()
model_static.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_static.fit(train_indices,ytrain,batch_size=50,epochs=50)

model_static_acc = model_static.evaluate(test_indices,ytest)[1]
print(model_static_acc)

def train_non_static_model():
  embedding_layer = Embedding(input_dim=len(word_index)+1,output_dim=300,weights=[embedding_matrix],trainable=True)
  return conv_model(embedding_layer)

model_non_static = train_non_static_model()
model_non_static.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_non_static.fit(train_indices,ytrain,batch_size=50,epochs=50)

model_non_static_acc = model_non_static.evaluate(test_indices,ytest)[1]
print(model_non_static_acc)

def multi_channel_model():
  embedding1 = np.random.rand(len(word_index)+1,300)
  embedding1 =  Embedding(input_dim=len(word_index)+1,output_dim=300,weights=[embedding1],trainable=True)
  
  embedding2 =  Embedding(input_dim=len(word_index)+1,output_dim=300,weights=[embedding_matrix],trainable=False)
  
  inp = Input(shape=(20,))
  embedding1_out = embedding1(inp)
  embedding2_out = embedding2(inp)
  
  final_embedding = Concatenate()([embedding1_out,embedding2_out])
  
  conv1 = Conv1D(100,3,activation='relu')(final_embedding)
  pool1 = MaxPooling1D(2,padding='valid')(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu')(final_embedding)
  pool2 = MaxPooling1D(2,padding='valid')(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu')(final_embedding)
  pool3 = MaxPooling1D(2,padding='valid')(conv3)
  out3 = Flatten()(pool3)
  
  concat_out = Concatenate()([out1,out2,out3])
  
  drop = Dropout(0.5)(concat_out)
  
  final_out = Dense(1,activation='sigmoid')(drop)
  
  model = Model(inputs=inp,outputs=final_out)
  
  return model

model_multi_channel = multi_channel_model()
model_multi_channel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_multi_channel.fit(train_indices,ytrain,batch_size=50,epochs=20)

model_multi_channel_acc = model_multi_channel.evaluate(test_indices,ytest)[1]
print(model_multi_channel_acc)

