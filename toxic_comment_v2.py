import keras
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

submission = pd.read_csv('data/sample_submission.csv')
def split_data_train_validate_test(data, train_percent=.7, validate_percent=.3, seed=None):
    np.random.seed(seed)
    shuffled_data = np.random.permutation(data)
    
    train = shuffled_data[0:int(train_percent*(data.shape[0])),:]
    validate = shuffled_data[int(train_percent*(data.shape[0])):int((train_percent+validate_percent)*(data.shape[0])),:]
    return train, validate

def read_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]
    return list_sentences_train, list_sentences_test, y

def tokenizeComments(list_sentences_train, list_sentences_test, n):
    max_words = n
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    return list_tokenized_train, list_tokenized_test

def getCommentLengthsDistribution(comments):
    commentsList = []
    for i in range(0, len(comments)):
        commentsList.append(len(comments[i]))
    
    plt.hist(commentsList,bins = np.arange(0,500,10))
    plt.show()

def padComments(list_tokenized_train, list_tokenized_test):
    maxlen = 250
    X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    return X_train, X_test


def buildFeedForwardModel(X_train, y, input_len, e, batch, l, opt, valid_split):
    # define the architecture for the simple feed forward network
    model = Sequential()
    model.add(Dense(1024, input_dim=input_len, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(0.25))
    model.add(Dense(512, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(0.25))
    model.add(Dense(6))
    model.add(Activation("softmax"))
    model.compile(loss=l, optimizer=opt, metrics=["accuracy"])
    model.fit(X_train, y, epochs=e, batch_size=batch, verbose=1, validation_split=valid_split)
    return model

def buildCNNModel(x_train, y_train, max_features, filters, stride, kernel_size, embedding_dim, bs, hidden_dim, maxlen):
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters,
                 kernel_size,
                 activation='relu',
                 strides=stride))
    
    model.add(GlobalMaxPool1D())
    model.add(Dense(hidden_dim))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=bs,
              epochs=1,
              validation_split = 0.3)
              #validation_data=(x_test, y_test))
    
    return model

def predictModel(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


#test run
MAX_FEATURES = 250000
list_sentences_train, list_sentences_test, y = read_data()
list_tokenized_train, list_tokenized_test = tokenizeComments(list_sentences_train, list_sentences_test, 250000)
x_train, x_test = padComments(list_tokenized_train, list_tokenized_test)


#ff_model = buildFeedForwardModel(x_train, y, 250, 1, 256, "binary_crossentropy", "adam",0.3)
#cnn_model = buildCNNModel(x_train, y, MAX_FEATURES, 32, 3, 250)
cnn_model = buildCNNModel(x_train, y, MAX_FEATURES, 512,1,7,1024,64,1024,250)


y_pred = predictModel(cnn_model, x_test)
print(len(y_pred))

print(y_pred[0:2])
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

submission[list_classes] = y_pred
print(submission.head())
submission.to_csv('/data/baseline_model_v4.csv', index = False)
