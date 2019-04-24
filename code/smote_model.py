#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
print(tensorflow.keras.__version__)
print(tensorflow.__version__)


# In[2]:


import pandas
import numpy as np
import pickle
import time
import json
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, Activation, LSTM,Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from gensim.models import KeyedVectors
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# In[3]:


df = pandas.read_table('asmsr/unigram_input.utf8', header=None)
label = pandas.read_table('asmsr/unigram_label.txt', header=None)


#validation data

df_val = pandas.read_table('asmsr_val/asmsr_unigram_input.txt', header=None)
label_val = pandas.read_table('asmsr_val/asmsr_unigram_label.txt', header=None)

df['label'] = label[0]
df.columns = ['character', 'label']

df = df.sample(50000)

df_val['val_label'] = label_val[0]
df_val.columns = ['val_character', 'val_label']

df_val = df_val.sample(20000)


# In[4]:


### Create sequence for training

vocabulary_size = 128

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['character'])
sequences = tokenizer.texts_to_sequences(df['character'])
data = pad_sequences(sequences, maxlen=50)


# In[5]:


### Create sequence for validation

vocabulary_size = 128

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df_val['val_character'])
sequences = tokenizer.texts_to_sequences(df_val['val_character'])
val_data = pad_sequences(sequences, maxlen=50)

val_data.shape


# In[6]:


labels = LabelEncoder().fit_transform(df.label)
val_labels = LabelEncoder().fit_transform(df_val.val_label)


# In[7]:


from collections import Counter

Counter(labels)


# In[8]:


Counter(val_labels)


# In[9]:


sm = SMOTE(random_state=42, ratio=1.0)


# In[10]:


train_data = data
train_labels = labels
validation_data = val_data
validation_labels = val_labels


# In[11]:


train_data_resampled, train_labels_resampled = SMOTE().fit_resample(train_data, train_labels)
validation_data_resampled, validation_label_resampled = SMOTE().fit_resample(validation_data, validation_labels)


# In[12]:


Counter(train_labels), Counter(validation_labels)


# In[14]:


unique, counts = np.unique(train_labels_resampled, return_counts=True)
print(np.asarray((unique, counts)).T)


# In[15]:


####Load Embedding
embedding_model = KeyedVectors.load_word2vec_format('asmsr/wang.txt')
embedding_model


# In[16]:


embedding_dim = len(embedding_model[next(iter(embedding_model.vocab))])
embedding_dim


# In[25]:


embedding_matrix = np.random.rand(256, embedding_dim)
word_index = tokenizer.word_index

for word, i in word_index.items():
    if i < vocabulary_size:
        try:
          embedding_vector = embedding_model.get_vector(word)
          embedding_matrix[i] = embedding_vector
        except:
          pass


# In[17]:


word_index = tokenizer.word_index

for word, i in word_index.items():
    if i < vocabulary_size:
        try:
          embedding_vector = embedding_model.get_vector(word)
          embedding_matrix[i] = embedding_vector
        except:
          pass


# In[18]:


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


# In[19]:


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


# In[22]:


y = to_categorical(train_labels_resampled, 4)
y_val_test = to_categorical(validation_label_resampled,4)


# In[23]:


print(y.shape)
print(y_val_test.shape)


# In[31]:



class_weights_ = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)


class_weights_


# In[ ]:
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

filepath="resources/weights-improvement-{epoch:02d}-{precision:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=precision, verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('resources/asmsr_smote_val_model.log', separator=',', append=False)
callbacks_list = [checkpoint,csv_logger,es]

model = build_model(256, 0.2, 0.04, 'sgd')

history = model.fit(train_data_resampled, y, validation_data = ( validation_data_resampled, y_val_test ), epochs=5, class_weight=class_weights_, batch_size=10, callbacks=callbacks_list, verbose=1)


# serialize model to JSON
model_json = model.to_json()
with open("resources/asmsr_smote_val_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('resources/asmsr_smote_val_weights.h5', overwrite=True)


# In[39]:


print(history.history)


# In[40]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
# plt.savefig('resources/Accuracy plot.png')


# In[41]:


# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
# plt.savefig('resources/Precision plot.png')


# In[42]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
# plt.savefig('resources/Loss plot.png')


# In[44]:


y_pred = model.predict_classes(validation_data_resampled[:500])
y_pred


# In[50]:


y_true = validation_label_resampled[:500]
y_true


# In[51]:


print(classification_report(y_true, y_pred, target_names=['0', '1', '2', '3']))


# In[ ]:
