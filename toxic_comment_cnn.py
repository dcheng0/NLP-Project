#!/usr/bin/env python
'''This is an implementation of a basic CNN for our project.

It is directly adapted from imdb_cnn.py from keras/exmaples/
Project specific code syntax orginally written by Ahmad and adapted by David.

Original imdb_cnn.py included in folder ref/ref-code for easy diff

Gets to 0.98 test accuracy after 1 epochs.
around 100 seconds total for nvidia gtx 1050ti gpu.
'''
from __future__ import print_function

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from argparse import ArgumentParser

# to add adjustable cli arguments, like in Denny Britz's CNN
# see ref/ for his project
parser = ArgumentParser(description='For adjusting hyperparameters.')
# various arguments
parser.add_argument('--embedding_dim', '-e', type=int,
    help='Dimensionality of character embedding (default: 128)',
    default=128)
parser.add_argument('--num_filters', type=int,
    help='Number of filters per filter size. (default: 250)',
    default=250)
parser.add_argument('--batch_size', type=int,
    help='Batch Size (default: 64)',
    default=64)
parser.add_argument('--num_epochs', type=int,
    help='Number of training epochs. (default: 1)',
    default=1)

# set args
args = parser.parse_args()

# set train and test values
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values

# set parameters:
max_features = 25000
maxlen = 250
batch_size = args.batch_size
embedding_dims = args.embedding_dim
filters = args.num_filters
kernel_size = 3
hidden_dims = 250
epochs = args.num_epochs

list_sentences_train = train["comment_text"]
list_sentences_test = train["comment_text"]

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

#
x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
x_tst = pad_sequences(list_tokenized_test, maxlen=maxlen)
print(x_train[0])

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(list_tokenized_train), 'train sequences')
print(len(list_tokenized_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
x_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=.3)

# Make Prediction to send to Kaggle for evaluation
y_pred = model.predict(x_test)
print(y_pred[0:5])

