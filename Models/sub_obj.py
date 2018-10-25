import nltk
nltk.download('stopwords')
nltk.download('punkt')

import sys
import os
import re
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import tensorflow as tf
import keras
from keras.layers import Dense,Input
from keras.models import Model,Sequential
from keras import metrics
from keras.layers import Embedding
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate,Concatenate,Flatten

from keras.constraints import max_norm


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
  pos_file = path+'quote.tok.gt9.5000'
  neg_file = path+'plot.tok.gt9.5000'
    
  corpus = []
  with open(pos_file,'r',encoding='latin1') as f:
    for each_line in f:
      corpus.append(preprocess(each_line))
  

  with open(neg_file,'r',encoding='latin1') as f:
    for each_line in f:
      corpus.append(preprocess(each_line))
      
  return corpus

path = '../Datasets/Subj_Obj/'

corpus = load_data(path)

labels = np.zeros(10000)
labels[0:5000] = 1


def generate_indices(Xtrain,ytrain):
  
  tokenizer = Tokenizer(num_words=21323)
  tokenizer.fit_on_texts(Xtrain)
  
  train_seq = tokenizer.texts_to_sequences(Xtrain)
  train_indices = pad_sequences(train_seq,maxlen=23)
  
  test_seq = tokenizer.texts_to_sequences(Xtest)
  test_indices = pad_sequences(test_seq,maxlen=23)
  
  word_index = tokenizer.word_index
  
  return train_indices,test_indices,word_index

GLOVEDIR = ''

embeddings_index = {}
f = open(os.path.join(GLOVEDIR,'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

def generate_embeddings(word_index):
  
  total_length = 21323
  embedding_matrix = np.random.rand(total_length+1,300)
  
  count = 1
  
  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    if(count==total_length):
      break
    count+=1
    
  return embedding_matrix

def train_random_model():
  
  total_length = 21324
  
  embeddings_random = np.random.rand(total_length,300)
  embedding_layer = Embedding(input_dim=total_length,output_dim=300,weights=[embeddings_random],trainable=True)
  
  input_vec = Input(shape=(23,))
  embedding_out = embedding_layer(input_vec)
  
  conv1 = Conv1D(100,3,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool1 = MaxPooling1D(2)(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool2 = MaxPooling1D(2)(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool3 = MaxPooling1D(2)(conv3)
  out3 = Flatten()(pool3)
  
  final_out = Concatenate()([out1,out2,out3])
  
  final_out = Dropout(0.5)(final_out)
  
  final_out = Dense(1,activation='sigmoid')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_static_model(embedding_pretrained):
  
  total_length = 21324
  
  embedding_layer = Embedding(input_dim=total_length,output_dim=300,weights=[embeddings_pretrained],trainable=False)
  
  input_vec = Input(shape=(23,))
  embedding_out = embedding_layer(input_vec)
  
  conv1 = Conv1D(100,3,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool1 = MaxPooling1D(2)(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool2 = MaxPooling1D(2)(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool3 = MaxPooling1D(2)(conv3)
  out3 = Flatten()(pool3)
  
  final_out = Concatenate()([out1,out2,out3])
  
  final_out = Dropout(0.5)(final_out)
  
  final_out = Dense(1,activation='sigmoid')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_non_static_model(embedding_pretrained):
  
  total_length = 21324
  
  embedding_layer = Embedding(input_dim=total_length,output_dim=300,weights=[embeddings_pretrained],trainable=True)
  
  input_vec = Input(shape=(23,))
  embedding_out = embedding_layer(input_vec)
  
  conv1 = Conv1D(100,3,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool1 = MaxPooling1D(2)(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool2 = MaxPooling1D(2)(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool3 = MaxPooling1D(2)(conv3)
  out3 = Flatten()(pool3)
  
  final_out = Concatenate()([out1,out2,out3])
  
  final_out = Dropout(0.5)(final_out)
  
  final_out = Dense(1,activation='sigmoid')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_multichannel_model(embedding_pretrained):
  
  total_length = 21324
  
  embeddings_random = np.random.rand(total_length,300)
  embedding_layer1 = Embedding(input_dim=total_length,output_dim=300,weights=[embeddings_random],trainable=True)
   
  embedding_layer2 = Embedding(input_dim=total_length,output_dim=300,weights=[embeddings_pretrained],trainable=False)
  
  input_vec = Input(shape=(23,))
  embedding_out1 = embedding_layer1(input_vec)
  
  embedding_out2 = embedding_layer2(input_vec)
  
  embedding_out = Concatenate()([embedding_out1,embedding_out2])
  
  conv1 = Conv1D(100,3,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool1 = MaxPooling1D(2)(conv1)
  out1 = Flatten()(pool1)
  
  conv2 = Conv1D(100,4,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool2 = MaxPooling1D(2)(conv2)
  out2 = Flatten()(pool2)
  
  conv3 = Conv1D(100,5,activation='relu',kernel_constraint=max_norm(3))(embedding_out)
  pool3 = MaxPooling1D(2)(conv3)
  out3 = Flatten()(pool3)
  
  final_out = Concatenate()([out1,out2,out3])
  
  final_out = Dropout(0.5)(final_out)
  
  final_out = Dense(1,activation='sigmoid')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def calculate_accuracy(net,X,y):
  
  test_length = len(X)
  pred = net.predict(X)
  pred = pred.reshape(test_length)
  ypred = (pred>0.5)
  
  return sum(ypred==y)

kf = StratifiedKFold(n_splits=2)

random_accuracy = 0
static_accuracy = 0
nonstatic_accuracy = 0
multichannel_accuracy = 0
fold = 1

for train_index,test_index in kf.split(corpus,labels):
  
  Xtrain = [corpus[i] for i in train_index]
  ytrain = labels[train_index]
  
  Xtest = [corpus[i] for i in test_index]
  ytest = labels[test_index]
  
  train_ind,test_ind,word_index = generate_indices(Xtrain,Xtest)
  
  embeddings_pretrained = generate_embeddings(word_index)
  
  print("Fold Number is :{}".format(fold))
  print("Training CNN random model")
  
  random_model = train_random_model()
  random_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  random_model.fit(train_ind,ytrain,batch_size=50,epochs=100)
  random_accuracy+=calculate_accuracy(random_model,test_ind,ytest)
  
  print("\n")
  
  print("Training CNN static model")
  
  static_model = train_static_model(embeddings_pretrained)
  static_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  static_model.fit(train_ind,ytrain,batch_size=50,epochs=100)
  static_accuracy+=calculate_accuracy(static_model,test_ind,ytest)
  
  print("\n")
  
  print("Training CNN Non static model")
  
  non_static_model = train_non_static_model(embeddings_pretrained)
  non_static_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  non_static_model.fit(train_ind,ytrain,batch_size=50,epochs=100)
  nonstatic_accuracy+=calculate_accuracy(non_static_model,test_ind,ytest)
  
  print("\n")
  
  print("Training CNN Multi Channel model")
  
  multichannel_model = train_multichannel_model(embeddings_pretrained)
  multichannel_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  multichannel_model.fit(train_ind,ytrain,batch_size=50,epochs=100)
  multichannel_accuracy+=calculate_accuracy(multichannel_model,test_ind,ytest)
  
  
  
  fold+=1
  print("\n")
  
print("Accuracy of all models :")
print("CNN Random Model accuracy is :{}".format((random_accuracy/10000)*100))
print("CNN Static Model accuracy is :{}".format((static_accuracy/10000)*100))
print("CNN Non-Static Model accuracy is :{}".format((nonstatic_accuracy/10000)*100))
print("CNN Multi Channel Model accuracy is :{}".format((multichannel_accuracy/10000)*100))

