#!/usr/bin/env python

# from __future__ import absolute import 

# import keras
from keras.preprocessing import text 
from keras.preprocessing.text import hashing_trick
import configparser

# Configuration
config = configparser.ConfigParser()

# Tokenizer, from Keras Docs 

text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ', char_level=False, oov_token=None)

'''
# Hashing Trick, from Keras Docs

hashing_trick(text, config.read('vocab_size'), hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')

# One_hot, from Keras Docs

text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~   ', lower=True, split=' ')


# text_to_word_sequence, from Keras Docs

ktxt.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')
'''
