#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.version


# In[2]:

# !pip install gensim
# !pip install tensorflow==1.12.0
import tensorflow
print(tensorflow.keras.__version__)
print(tensorflow.__version__)


# In[3]:


import pandas
import numpy as np
import pickle
import time
import json
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, Activation, LSTM,Dropout, InputLayer
from gensim.models import KeyedVectors


# In[4]:


# from keras.models import model_from_json
df = pandas.read_table('asmsr/unigram_input.utf8', header=None)
label = pandas.read_table('asmsr/unigram_label.txt', header=None)


# In[8]:


df['label'] = label[0]
df.columns = ['character', 'label']


# In[9]:


df.sample(5)


# In[10]:


### Create sequence

vocabulary_size = 128

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['character'])
sequences = tokenizer.texts_to_sequences(df['character'])
data = pad_sequences(sequences, maxlen=50)

data.shape


# In[12]:


labels = LabelEncoder().fit_transform(df.label)
labels[:500]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)


# In[14]:


X_train.shape, y_train.shape


# In[18]:


####Load Embedding
embedding_model = KeyedVectors.load_word2vec_format('/root/asmsr/wang.txt')
embedding_model


# In[19]:


embedding_dim = len(embedding_model[next(iter(embedding_model.vocab))])
embedding_dim


# In[20]:


embedding_matrix = np.random.rand(256, embedding_dim)
embedding_matrix


# In[21]:


word_index = tokenizer.word_index

for word, i in word_index.items():
    if i < vocabulary_size:
        try:
          embedding_vector = embedding_model.get_vector(word)
          embedding_matrix[i] = embedding_vector
        except:
          pass


# In[22]:


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# In[24]:


CLASSES = 4
max_len = 50
word_size = 100

def build_model(char_size, dropout, lr, optimizer):
    model = Sequential()
    model.add(Embedding(char_size, word_size, input_length=max_len, weights=[embedding_matrix],mask_zero=True))
    model.add(Bidirectional(LSTM(char_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False), merge_mode='sum'))
    model.add(Dropout(dropout))
    model.add((Dense(CLASSES, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[precision, 'accuracy'])
    return model


# In[25]:


def save(model):
    timestring = "".join(str(time.time()).split("."))  ###Save each epoch with a timestamp
    model_json_name = '/root/resources/modelnew_{}.json'.format(timestring)
    model_json = model.model.to_json()
    with open(model_json_name, "w") as json_file:
        json_file.write(model_json)
    return model_json

def get_metrics(y_pred, y_true):
    chosen_metrics = {
        'accuracy': metrics.accuracy_score,
        'precision' : metrics.precision_score
    }

    results = {}
    for metric_name, metric_func in chosen_metrics.items():
        try:
            if metric_name == "precision":
                inter_res = metric_func(y_pred, y_true, average=None)
            else:
                inter_res = metric_func(y_pred, y_true)
        except Exception as ex:
            inter_res = None
            print("Couldn't evaluate %s because of %s", metric_name, ex)
        results[metric_name] = inter_res
    return results

def scorer_callback(model, X, y):
    # do all the work and return some of the metrics
    y_pred_val = model.predict(X)
    results = get_metrics(y_pred_val, y)
    model_savepath = save(model)
    return results['accuracy']


# In[ ]:


params_dict = {
    'lr': [0.04, 0.03],
    'char_size': [256],
    'dropout': [0.20, 0.25,],
    'optimizer': ['adam', 'sgd'],
}

grid = GridSearchCV(model,
                    cv=3,
                    param_grid=params_dict,
                    return_train_score=True,
                    scoring=scorer_callback)

grid_results = grid.fit(X_train,y_train)

print('Parameters of the best model: ')
print(grid_results.best_params_)


# In[28]:


model_json = grid.best_estimator_.model.to_json()
with open("/root/resources/grid_model.json", "w") as file:
    file.write(model_json)

grid.best_estimator_.model.save_weights('/root/resources/weights/grid_weights.h5', overwrite=True)


# In[47]:


grid.best_score_


# In[48]:


print('Parameters of the best model: ')
print(grid_results.best_params_)


# In[ ]:
