#!/usr/bin/env python

from __future__ import absolute import 

from keras import preprocessing

# from https://keras.io/preproccesing/text

preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_'{|]~', lower=True, split=' ',
char_level=False, oov_Token=None)

hashing_trick(text, n, hash_functions=None, filters=
