#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import glob

file_names = glob.glob('books/hp1.txt')
len(file_names)

def get_sentences(file_name):
    with open(file_name, 'r') as f:
        return f.read().split('.')

MIN_LENGTH = 15
sentences = []
for file_name in file_names:
    sentences+=get_sentences(file_name)

sentences = [sentence.replace('\n','') for sentence in sentences]
sentences = [sentence.replace('\t','') for sentence in sentences]
sentences = [sentence for sentence in sentences if len(sentence)>MIN_LENGTH]

lengths = [len(sentence) for sentence in sentences]
lengths = pd.Series(lengths)
lengths.quantile(.8)
lengths.describe()


#--------- Load Whole Corpus----------------#

corpus = ""
for file_name in file_names:
    with open(file_name, 'r') as f:
            corpus+=f.read()

corpus = corpus.replace('\n',' ')
corpus = corpus.replace('\t',' ')
corpus = corpus.replace('“', ' " ')
corpus = corpus.replace('”', ' " ')

for spaced in ['.','-',',','!','?','(','—',')','\"']:
    corpus = corpus.replace(spaced, ' {0} '.format(spaced))

print(f"length of corpus = {len(corpus)}")
#print(f"Corpus[10000:15000] = {corpus[10000:15000]}")

corpus_words = corpus.split(' ')
corpus_words= [word for word in corpus_words if word != '']

print(f"len(corpus_words = {len(corpus_words)}")


distinct_words = list(set(corpus_words))
word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
distinct_words_count = len(list(set(corpus_words)))
print(f"Num of distinct words = {distinct_words_count}")


k = 2
sets_of_k_words = [' '.join(corpus_words[i:i+k]) for i, _ in enumerate(corpus_words[:-k])]

from random import random 

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects', 
        the likelihood of the objects is weighted according 
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]

from scipy.sparse import dok_matrix

sets_count = len(list(set(sets_of_k_words)))
next_after_k_words_matrix = dok_matrix((sets_count, len(distinct_words)))
distinct_sets_of_k_words = list(set(sets_of_k_words))
print(f"Distinct sets of {k} words = {len(distinct_sets_of_k_words)}")
k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}

for i, word in enumerate(sets_of_k_words[:-k]):
    word_sequence_idx = k_words_idx_dict[word]
    next_word_idx = word_idx_dict[corpus_words[i+k]]
    next_after_k_words_matrix[word_sequence_idx, next_word_idx] += 1

def sample_next_word_after_sequence(word_sequence, alpha = 0):
    next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
    likelihoods = next_word_vector/next_word_vector.sum()
    return weighted_choice(distinct_words, likelihoods.toarray())

def stochastic_chain(seed, chain_length = 15, seed_length = 2):
    current_words = seed.split(' ')
    if len(current_words) != seed_length:
        raise ValueError(f"Wrong number of words; expected {seed_length}, got {len(current_words)}")
    sentence = seed
    for _ in range(chain_length):
        sentence += ' '
        next_word = sample_next_word_after_sequence(' '.join(current_words))
        sentence += next_word
        current_words = current_words[1:] + [next_word]

    return sentence


things_to_try = ['his glasses', 'Ron said', 'He Who']

for s in things_to_try:
    print(stochastic_chain(s))
