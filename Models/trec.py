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
  return text.lower()

train_length = 5452 
test_length = 500

labels = {'DESC':0,'ENTY':1,'ABBR':2,'HUM':3,'NUM':4,'LOC':5}

def load_data(path):
  
  train_file = path+'train.txt'
  test_file = path+'test.txt'
  
  count = 0
  
  train_corpus = []
  train_labels = np.zeros(train_length)
  
  with open(train_file,'r',encoding='latin1') as f:
    for line in f.readlines():
      words = line.split(" ")
      word = words[0]
      label = word[:word.find(":")]
      sentence = " ".join(words[1:-1])
      train_corpus.append(preprocess(sentence))
      train_labels[count] = labels[label]
      count+=1
      
  count = 0
      
  test_corpus = []
  test_labels = np.zeros(test_length)
  
  with open(test_file,'r',encoding='latin1') as f:
    for line in f.readlines():
      words = line.split(" ")
      word = words[0]
      label = word[:word.find(":")]
      sentence = " ".join(words[1:-1])
      test_corpus.append(preprocess(sentence))
      test_labels[count] = labels[label]
      count+=1
      
  return train_corpus,test_corpus,train_labels,test_labels

path = '../Datasets/TREC/'

train_corpus,test_corpus,train_labels,test_labels = load_data(path)

print(train_corpus[0:2])
print(test_corpus[0:2])

assert len(train_corpus)==len(train_labels)
assert len(test_corpus)==len(test_labels)

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

def generate_indices(Xtrain,Xtest):
  
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(Xtrain)
  
  train_seq = tokenizer.texts_to_sequences(Xtrain)
  train_indices = pad_sequences(train_seq,maxlen=10)
  
  test_seq = tokenizer.texts_to_sequences(Xtest)
  test_indices = pad_sequences(test_seq,maxlen=10)
  
  wordindex = tokenizer.word_index
  
  return train_indices,test_indices,wordindex

GLOVEDIR = ''

embeddings_index = {}
f = open(os.path.join(GLOVEDIR,'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

def generate_embeddings(total_length):
  
  embedding_matrix = np.random.uniform(-0.5,0.5,(total_length+1,300))
  
  count = 1
  
  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    
    if(count==total_length):
      break
    count+=1
    
  return embedding_matrix

def train_random_model(total_length):
  
  
  embeddings_random = np.random.uniform(-0.5,0.5,(total_length+1,300))
  embedding_layer = Embedding(input_dim=total_length+1,output_dim=300,weights=[embeddings_random],trainable=True)
  
  input_vec = Input(shape=(10,))
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
  
  final_out = Dense(6,activation='softmax')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_static_model(total_length,embedding_pretrained):
  
  
  embedding_layer = Embedding(input_dim=total_length+1,output_dim=300,weights=[embeddings_pretrained],trainable=False)
  
  input_vec = Input(shape=(10,))
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
  
  final_out = Dense(6,activation='softmax')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_non_static_model(total_length,embedding_pretrained):
  
  
  
  embedding_layer = Embedding(input_dim=total_length+1,output_dim=300,weights=[embeddings_pretrained],trainable=True)
  
  input_vec = Input(shape=(10,))
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
  
  final_out = Dense(6,activation='softmax')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def train_multichannel_model(total_length,embedding_pretrained):
  
  
  embeddings_random = np.random.uniform(-0.5,0.5,(total_length+1,300))
  embedding_layer1 = Embedding(input_dim=total_length+1,output_dim=300,weights=[embeddings_random],trainable=True)
   
  embedding_layer2 = Embedding(input_dim=total_length+1,output_dim=300,weights=[embeddings_pretrained],trainable=False)
  
  input_vec = Input(shape=(10,))
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
  
  final_out = Dense(6,activation='softmax')(final_out)
  
  model = Model(inputs=input_vec,outputs=final_out)
  
  return model

def calculate_accuracy(net,X,y):
  
  test_length = len(X)
  pred = net.predict(X)
  ypred = np.argmax(pred,1)
  ypred = ypred.reshape(test_length)
  
  return sum(ypred==y)

random_accuracy = 0
static_accuracy = 0
nonstatic_accuracy = 0
multichannel_accuracy = 0

ytrain = keras.utils.to_categorical(train_labels,num_classes=6)

train_ind,test_ind,wordindex = generate_indices(train_corpus,test_corpus)
totallength = max(len(wordindex),9592)
embeddings_pretrained = generate_embeddings(totallength)

print("Training CNN random model")
  
random_model = train_random_model(totallength)
random_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  
random_model.fit(train_ind,ytrain,batch_size=50,epochs=30)
random_accuracy=calculate_accuracy(random_model,test_ind,test_labels)

print("\n")
  
print("Training CNN static model")
  
static_model = train_static_model(totallength,embeddings_pretrained)
static_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  
static_model.fit(train_ind,ytrain,batch_size=50,epochs=30)
static_accuracy=calculate_accuracy(static_model,test_ind,test_labels)
  
print("\n")
  
print("Training CNN Non static model")
  
non_static_model = train_non_static_model(totallength,embeddings_pretrained)
non_static_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  
non_static_model.fit(train_ind,ytrain,batch_size=50,epochs=30)
nonstatic_accuracy=calculate_accuracy(non_static_model,test_ind,test_labels)
  
print("\n")
  
print("Training CNN Multi Channel model")
  
multichannel_model = train_multichannel_model(totallength,embeddings_pretrained)
multichannel_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  
multichannel_model.fit(train_ind,ytrain,batch_size=50,epochs=30)
multichannel_accuracy=calculate_accuracy(multichannel_model,test_ind,test_labels)
  
  
  
  
print("\n")
  
print("Accuracy of all models :")
print("CNN Random Model accuracy is :{}".format((random_accuracy/500)*100))
print("CNN Static Model accuracy is :{}".format((static_accuracy/500)*100))
print("CNN Non-Static Model accuracy is :{}".format((nonstatic_accuracy/500)*100))
print("CNN Multi Channel Model accuracy is :{}".format((multichannel_accuracy/500)*100))

