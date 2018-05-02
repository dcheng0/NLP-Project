import keras
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, History
from keras import initializers, regularizers, constraints, optimizers, layers

submission = pd.read_csv('../data/sample_submission.csv')
def split_data_train_validate_test(data, train_percent=.8, validate_percent=.2, seed=None):
    np.random.seed(seed)
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    #shuffled_data = np.random.permutation(data)
    train = shuffled_data.iloc[0:int(train_percent*(data.shape[0])),:]
    validate = shuffled_data.iloc[int(train_percent*(data.shape[0])):int((train_percent+validate_percent)*(data.shape[0])),:]
    return train, validate

def read_data(valid_prob):
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] 
    train, validate = split_data_train_validate_test(train, valid_prob, 1-valid_prob)
    y_train = train[list_classes].values
    y_validate = validate[list_classes].values
    
    list_sentences_train = train["comment_text"]
    list_sentences_valid = validate["comment_text"]
    list_sentences_test = test["comment_text"]
    
    return list_sentences_train, list_sentences_test, list_sentences_valid, y_train, y_validate

def tokenizeComments(list_sentences_train, list_sentences_test, list_sentences_valid, max_words):
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts((list(list_sentences_train)+list(list_sentences_valid)))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_valid = tokenizer.texts_to_sequences(list_sentences_valid)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    
    return list_tokenized_train, list_tokenized_test, list_tokenized_valid

def getCommentLengthsDistribution(comments):
    commentsList = []
    for i in range(0, len(comments)):
        commentsList.append(len(comments[i]))
    
    plt.hist(commentsList,bins = np.arange(0,500,10))
    plt.show()

def padComments(list_tokenized_train, list_tokenized_test,list_tokenized_valid, max_comment_length):
    X_train = pad_sequences(list_tokenized_train, maxlen=max_comment_length)
    X_test = pad_sequences(list_tokenized_test, maxlen=max_comment_length)
    X_valid = pad_sequences(list_tokenized_valid, maxlen=max_comment_length)
    return X_train, X_test, X_valid


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
 
def buildCNNModel(x_train, y_train, x_valid, y_valid
                  , max_features, filters, stride, kernel_size
                  , embedding_dim, bs, hidden_dim, convolved_layers
                  , act_function, loss_function, epoch_num, maxlen):

    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters,
                 kernel_size,
                 activation='relu',
                 strides=stride))
    
    if convolved_layers != 1:
        model.add(MaxPooling1D(pool_size=2))
    else:
        model.add(GlobalMaxPool1D())


    if convolved_layers == 2:
        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(GlobalMaxPool1D())

    if convolved_layers == 3:
        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(GlobalMaxPool1D())

    model.add(Dense(hidden_dim))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = History()

    nn_model = model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
              #validation_split = 0.3,
              validation_data = (x_valid, y_valid),
              callbacks=[early_stopping, history],
              shuffle = True,
              verbose = 1)
    
    #val_accuracy = model.evaluate(x_valid, y_valid, verbose=0)
    
    return nn_model, nn_model.history

def predictModel(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


#test run
MAX_WORDS = 200000
max_comment_length = 200
list_sentences_train, list_sentences_test, list_sentences_valid, y_train, y_validate = read_data(0.8)
list_tokenized_train, list_tokenized_test, list_tokenized_valid,  = tokenizeComments(list_sentences_train, list_sentences_test, list_sentences_valid, MAX_WORDS)
x_train, x_test, x_valid = padComments(list_tokenized_train, list_tokenized_test, list_tokenized_valid, max_comment_length)

print("TRAINING A SINGLE CONV LAYER CNN")
cnn_model_1layer, cnn_model_1layer_hist = buildCNNModel(x_train, y_train
    , x_valid, y_validate
    , max_features = MAX_WORDS
    , filters = 32
    , stride = 1
    , kernel_size = 5
    , embedding_dim = 128
    , bs =128
    , hidden_dim = 128
    , convolved_layers = 1
    , act_function = 'relu'
    , loss_function = 'binary_crossentropy'
    , epoch_num = 2
    , maxlen = max_comment_length)

print(cnn_model_1layer_hist)
print(cnn_model_1layer_hist.keys())

nb_epochs=2
val_loss = cnn_model_1layer_hist["val_loss"][nb_epochs - 1]
val_acc = cnn_model_1layer_hist["val_acc"][nb_epochs - 1]

print(val_loss)
print(val_acc)


y_pred = predictModel(cnn_model_1layer, x_test)
print(len(y_pred))

print(y_pred[0])
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

submission[list_classes] = y_pred
print(submission.head())
submission.to_csv('/data/baseline_model_v5.csv', index = False)


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submission[list_classes] = y_pred_1
submission.to_csv('../data/cnn_v5.csv', index = False)