
# coding: utf-8

# In[ ]:

import keras
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values


# In[ ]:

train['comment_text'].head()
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]
print(list_sentences_train[0])


# In[ ]:

max_features = 25000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[ ]:

print(list_tokenized_train[0])


# In[ ]:

def getCommentLengthsDistribution(comments):
    commentsList = []
    for i in range(0, len(comments)):
        commentsList.append(len(comments[i]))
    
    plt.hist(commentsList,bins = np.arange(0,500,10))
    plt.show()


# In[ ]:

maxlen = 250
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
print(X_train[0])


# In[ ]:

##the neural net architecture for an lSTM
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

##training
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
batch_size = 32
epochs = 1
model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[ ]:

# define the architecture for the simple feed forward network
model = Sequential()
model.add(Dense(1024, input_dim=maxlen, kernel_initializer="uniform",
	activation="signmoid"))
model.add(Dropout(0.25))
model.add(Dense(512, kernel_initializer="uniform", activation="sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(6))
model.add(Activation("softmax"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y, epochs=5, batch_size=256, verbose=1, validation_split=0.2)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



