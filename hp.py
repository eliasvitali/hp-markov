#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import glob

file_names = glob.glob('Books/hp1.txt')
len(file_names)

#Get all sentences in the book by splitting on full stops
def get_sentences(file_name):
    with open(file_name, 'r') as f:
        return f.read().split('.')

MIN_LENGTH = 15                                                                     #Min sentence length
sentences = []

#Store sentences in a list
for file_name in file_names:
    sentences+=get_sentences(file_name)

#Loop through sentences and replace line breaks with spaces
sentences = [sentence.replace('\n','') for sentence in sentences]
sentences = [sentence.replace('\t','') for sentence in sentences]
sentences = [sentence for sentence in sentences if len(sentence)>MIN_LENGTH]        #Minimum sentence length (so we don't have sentences like "I do. Yes." etc


lengths = [len(sentence) for sentence in sentences]                                 #list which stores the lengths of various parameters
lengths = pd.Series(lengths)
lengths.quantile(.8)
lengths.describe()                                                                  #prints some parameters to the console


#--------- Load Whole Corpus----------------#

corpus = ""                                                                         #string that stores the corpus
for file_name in file_names:
    with open(file_name, 'r') as f:
            corpus+=f.read()

#pad evert punctuation mark with spaces to treat them as tokens (i.e. treat them as words)
corpus = corpus.replace('\n',' ')
corpus = corpus.replace('\t',' ')
corpus = corpus.replace('“', ' " ')
corpus = corpus.replace('”', ' " ')
for spaced in ['.','-',',','!','?','(','—',')']:
    corpus = corpus.replace(spaced, ' {0} '.format(spaced))

print(f"length of corpus = {len(corpus)}")

#Split corpus on words to get list of all words
corpus_words = corpus.split(' ')
corpus_words= [word for word in corpus_words if word != '']

print(f"len(corpus_words = {len(corpus_words)}")


#Get number of distinct words and store them in a dictionary. Words are the key and the value is the index
distinct_words = list(set(corpus_words))
word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
distinct_words_count = len(list(set(corpus_words)))
print(f"Num of distinct words = {distinct_words_count}")





#-----------------Start Markov Chain Training---------------------#


k = 2                                                                                                   #k = number of words it reads before it starts predicting the next one
sets_of_k_words = [' '.join(corpus_words[i:i+k]) for i, _ in enumerate(corpus_words[:-k])]              #List containing all the distinct sets of k words in the corpus

from random import random 

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects', 
        the likelihood of the objects is weighted according 
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtypei = np.float64)
    sum_of_weights = weights.sum()
    
    #Standardize
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]

from scipy.sparse import dok_matrix                                                                     #Dok matrix (dic of keys) stores zeroes more efficiently to use less memory
                                                                                                        #Matrix will have very few (<1%) non-zero elements so dataset is very sparse

sets_count = len(list(set(sets_of_k_words)))                                                            #number of distinct sets of k words
next_after_k_words_matrix = dok_matrix((sets_count, len(distinct_words)))                               #Initialize matrix of zeroes of size n x n where n = # of distinct sets of k words
distinct_sets_of_k_words = list(set(sets_of_k_words))
print(f"Distinct sets of {k} words = {len(distinct_sets_of_k_words)}")
k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}                         #turns the list into a dictionary


#This loop populates the matrix
#Each row represents a word in the corpus. e.g. if the corpus was only the sentence: "I am your father" would be a 4x4 matrix:
# /       I AM YOUR FATHER
#       I 0  0   0    0
#      AM 0  0   0    0
#    YOUR 0  0   0    0
#  FATHER 0  0   0    0
#
#The columns will have a 1 if the row-column set of words exists in the distinct set of words list. So in the above example it would look like
# /       I AM YOUR FATHER
#       I 0  1   0    0
#      AM 0  0   1    0
#    YOUR 0  0   0    1
#  FATHER 0  0   0    0
#
#We do this for all the distinct words and end up with a very sparse matrix where <1% of elements are non-zero. 
#Hence we use a sparse matrix implementation called dok matrix to store our values
#Code:
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
