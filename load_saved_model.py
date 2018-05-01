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
from keras.models import load_model

model = load_model('toxic_cnn_model')

max_features = 25000
maxlen = 250
batch_size = args.batch_size
embedding_dims = args.embedding_dim
filters = args.num_filters
kernel_size = 3
hidden_dims = 250
epochs = args.num_epochs

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

y_pred = model.predict(x_test)
print(y_pred[0:5])

submission = pd.read_csv("sample_submission.csv")
submission.ix[:,1:6] = y_pred
submission.to_csv("./sub_1.csv", index=False)

