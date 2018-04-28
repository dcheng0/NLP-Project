#!/usr/bin/env python

from keras.models import Sequential

model = Sequential()

model.add(Dense(units=64, activation=relu, input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])
 

