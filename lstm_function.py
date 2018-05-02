def build_LSTM_Model(x_train, y_train, max_features, filters, stride, kernel_size, embedding_dim, bs, 
    act_function, loss_function, epoch_num, maxlen):

    print('Build model...')
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dim))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
              validation_split = 0.3)
              #validation_data=(x_test, y_test))
    
    return model
