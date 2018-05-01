    model.add(Dropout(0.25))
    model.add(Dense(6))
    model.add(Activation("softmax"))
    model.compile(loss=l, optimizer=opt, metrics=["accuracy"])
    model.fit(X_train, y, epochs=e, batch_size=batch, verbose=1, validation_split=valid_split)
    return model

#optimizer
#loss function
#activation function    

def buildCNNModel(x_train, y_train, max_features, filters, stride, kernel_size, embedding_dim, bs, hidden_dim, convolved_layers, 
    act_function, loss_function, epoch_num, maxlen):

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
    
    model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
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
#cnn_model_1 = buildCNNModel(x_train, y, MAX_FEATURES, 32, 3, 250)
#cnn_model_2 = buildCNNModel(x_train, y, MAX_FEATURES, 512,1,7,1024,64,1024,250)
print("TRAINING A SINGLE CONV LAYER CNN")
cnn_model_1layer = buildCNNModel(x_train, y, max_features = MAX_FEATURES, filters = 32, stride = 1, kernel_size = 5, embedding_dim = 128, bs =128, hidden_dim = 128, convolved_layers = 1
                                 , act_function = 'relu', loss_function = 'binary_crossentropy', epoch_num = 10, maxlen = 250)

print("TRAINING A DOUBLE CONV LAYER CNN")
cnn_model_2layer = buildCNNModel(x_train, y, max_features = MAX_FEATURES, filters = 32, stride = 1, kernel_size = 5, embedding_dim = 128, bs =128, hidden_dim = 128, convolved_layers = 2
                                 , act_function = 'relu', loss_function = 'binary_crossentropy', epoch_num = 10, maxlen = 250)

print("TRAINING A TRIPLE CONV LAYER CNN")
cnn_model_3layer = buildCNNModel(x_train, y, max_features = MAX_FEATURES, filters = 32, stride = 1, kernel_size = 5, embedding_dim = 128, bs =128, hidden_dim = 128, convolved_layers = 3
                                 , act_function = 'relu', loss_function = 'binary_crossentropy', epoch_num = 10, maxlen = 250)



y_pred_1 = predictModel(cnn_model_1layer, x_test)
y_pred_2 = predictModel(cnn_model_2layer, x_test)
y_pred_3 = predictModel(cnn_model_3layer, x_test)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submission[list_classes] = y_pred_1
submission.to_csv('data/cnn_1_layer.csv', index = False)

submission[list_classes] = y_pred_2
submission.to_csv('data/cnn_2_layer.csv', index = False)

submission[list_classes] = y_pred_3
submission.to_csv('data/cnn_3_layer.csv', index = False)
