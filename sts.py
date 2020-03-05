# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:32:08 2020

@author: Muhammad Ali
"""


import numpy as np
import pandas as pd


dataset=pd.read_csv('C:/Users/Muhammad Ali/Downloads/similarity/Text_Similarity_Dataset.csv')
X=dataset.iloc[:,1:2].values



import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Muhammad Ali/Downloads/similarity/GoogleNews-vectors-negative300.bin.gz', binary=True)

index2word_set=set(word2vec_model.index2word)

def avg_sentence_vector(words, model, num_feature, index2word_set):
    
    featureVec= np.zeros((num_feature,), dtype="float32")
    nwords = 0
    
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec


from scipy import spatial

list=[]
list1=[]

for i in range(0,4023):

    text_1 = dataset['text1'][i]
    text_1_avg_vector = avg_sentence_vector(text_1.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    text_1_avg_vector=text_1_avg_vector.reshape(-1,1)

    text_2 = dataset['text2'][i]
    text_2_avg_vector = avg_sentence_vector(text_2.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    text_2_avg_vector=text_2_avg_vector.reshape(-1,1)

    text1_text2_similarity = 1-spatial.distance.cosine(text_1_avg_vector,text_2_avg_vector)
    
    list1.append(i)
    list.append(text1_text2_similarity)
    
output=pd.DataFrame((list),columns=['Similarity_Score'])
output.to_csv('Semantic_train.csv')