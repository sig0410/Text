#!/usr/bin/env python
# coding: utf-8

# In[87]:


import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[88]:


data_file = './/Amazon_Unlocked_Mobile.csv'
n = 413000  
s = 20000 
skip = sorted(random.sample(range(1,n),n-s))


df = pd.read_csv( data_file, delimiter = ",", skiprows = skip)


# In[89]:


df


# In[90]:


df.isnull().sum()


# In[91]:


df['Reviews']


# In[92]:


df = df.dropna(subset= ['Reviews'])


# In[93]:


df


# ### Rating Mapping

# In[94]:


Review_mapping =  {0: '0',1 : '0', 2 :'0',3 : '0', 4 : '1', 5 : '1'}
Review_mapping


# In[95]:


Rating = lambda x: Review_mapping.get(x,x)
df['Rating']=df.Rating.map(Rating)


# In[96]:


df['Rating'].unique()


# ### Text Analysis

# In[98]:


df.columns


# In[99]:


a = df.iloc[:,3]
print(a)


# In[100]:


b = df.iloc[:,4]


# In[101]:


doc = pd.concat([a,b], axis = 1)


# In[102]:


train_docs, test_docs = train_test_split(doc, test_size = 0.3)


# In[103]:


def review_to_wordlist(review, remove_stopwords = True):

    
    review_text = re.sub('[^a-zA-Z]'," ", review)
    #review_text에 영어만 넣기 
    
    words = review_text.lower().split()
    #소문자로 바꿔주고 그것들을 분리해준다 
    
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]
        #stops에 영어의 불용어를 넣어줌 
        #words는 소문자로 변환되고 띄어져있는 것이며 
        #stops에 있는 불용어를 제외하고 넣어줌 
        
    b = []
    stemmer = english_stemmer
    for word in words:
        b.append(stemmer.stem(word))
        #words에 전처리된것들의 어간들만 추출 
        
    return(b)


# In[104]:


clean_train_reviews = []
for review in train_docs['Reviews']:
    clean_train_reviews.append( " ".join(review_to_wordlist(review)))
    
clean_test_reviews = []
for review in test_docs['Reviews']:
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))


# In[19]:


c = train_docs['Rating']
d = test_docs['Rating']


# In[20]:


c.isnull().sum()


# In[21]:


len(c)


# In[22]:


len(d)


# In[23]:


train_docs.isnull().sum()


# In[24]:


test_docs.isnull().sum()


# # ================ 5/1

# ### Tf-Idf

# In[118]:


X_train = clean_train_reviews
X_test = clean_test_reviews 

y_train = train_docs['Rating']
y_test = test_docs['Rating']


# In[122]:


X_train


# In[119]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import words


# In[120]:


vectorizer = CountVectorizer(analyzer = 'word', 
                             lowercase = True,
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = 'english',
                             min_df = 2, # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋다. 
                             ngram_range=(1, 3),
                             vocabulary = set(words.words()), # nltk의 words를 사용하거나 문서 자체의 사전을 만들거나 선택한다. 
                             max_features = 20000
                            )


# In[121]:


pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf = False))
])


# In[123]:


X_train_tf_idf_vector = pipeline.fit_transform(X_train)


# In[124]:


X_test_tf_idf_vector = pipeline.fit_transform(X_test)


# In[125]:


from sklearn.ensemble import RandomForestClassifier


# In[126]:


forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)
forest


# In[127]:


get_ipython().run_line_magic('time', 'forest = forest.fit(X_train_tf_idf_vector, y_train)')


# In[128]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[129]:


k_fold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

score = np.mean(cross_val_score(                               forest, X_train_tf_idf_vector,                                y_train, cv = k_fold, scoring = 'roc_auc', n_jobs = -1))


# In[130]:


format(score)


# In[131]:


result = forest.predict(X_test_tf_idf_vector)


# In[132]:


y_test


# In[133]:


a = sum(y_test == result)
print("테스트 셋 정확도 :",a / len(y_test))


# ### 전체 데이터셋 활용

# In[134]:


df = pd.read_csv('./Amazon_Unlocked_Mobile.csv')


# In[135]:


df = df.dropna(subset= ['Reviews'])


# In[136]:


df.isnull().sum()


# In[137]:


Review_mapping =  {0: '0',1 : '0', 2 :'0',3 : '0', 4 : '1', 5 : '1'}
Review_mapping


# In[138]:


Rating = lambda x: Review_mapping.get(x,x)
df['Rating']=df.Rating.map(Rating)


# In[139]:


df


# In[140]:


a = df.iloc[:,3]
b = df.iloc[:,4]


# In[141]:


doc = pd.concat([a,b], axis = 1)
train_docs, test_docs = train_test_split(doc, test_size = 0.3)


# In[142]:


def review_to_wordlist(review, remove_stopwords = True):

    
    review_text = re.sub('[^a-zA-Z]'," ", review)
    #review_text에 영어만 넣기 
    
    words = review_text.lower().split()
    #소문자로 바꿔주고 그것들을 분리해준다 
    
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]
        #stops에 영어의 불용어를 넣어줌 
        #words는 소문자로 변환되고 띄어져있는 것이며 
        #stops에 있는 불용어를 제외하고 넣어줌 
        
    b = []
    stemmer = english_stemmer
    for word in words:
        b.append(stemmer.stem(word))
        #words에 전처리된것들의 어간들만 추출 
        
    return(b)


# In[143]:


clean_train_reviews = []
for review in train_docs['Reviews']:
    clean_train_reviews.append( " ".join(review_to_wordlist(review)))
    
clean_test_reviews = []
for review in test_docs['Reviews']:
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))


# In[144]:


X_train = clean_train_reviews
X_test = clean_test_reviews

y_train = train_docs['Rating']
y_test = test_docs['Rating']


# In[145]:


len(clean_train_reviews)


# In[146]:


vectorizer = CountVectorizer(analyzer = 'word', 
                             lowercase = True,
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = 'english',
                             min_df = 2, # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋다. 
                             ngram_range=(1, 5),
                             vocabulary = set(words.words()), # nltk의 words를 사용하거나 문서 자체의 사전을 만들거나 선택한다. 
                             max_features = 300000
                            )


# In[147]:


pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf = False))
])


# In[148]:


X_train_tf_idf_vector = pipeline.fit_transform(X_train)
X_test_tf_idf_vector = pipeline.fit_transform(X_test)


# In[149]:


forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)
forest


# In[150]:


get_ipython().run_line_magic('time', 'forest = forest.fit(X_train_tf_idf_vector, y_train)')


# In[151]:


k_fold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

score = np.mean(cross_val_score(                               forest, X_train_tf_idf_vector,                                y_train, cv = k_fold, scoring = 'roc_auc', n_jobs = -1))
print(format(score))


# In[152]:


result = forest.predict(X_test_tf_idf_vector)
a = sum(y_test == result)
print("테스트 셋 정확도 :",a / len(y_test))


# In[ ]:




