
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from gensim.models import Word2Vec

import tensorflow as tf

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional,Activation

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[2]:


data = pd.read_csv("DATA_PATH")


# In[3]:


text_train, text_test, y_train, y_test = train_test_split(data['text'], 
                                                          data['label'], 
                                                          random_state = 42, 
                                                          test_size=0.2)


# In[4]:


maxlen = max([len(s.split()) for s in text_train.values.tolist()])


# In[6]:


def Tokenizing(data, maxlen, train, tokenizer = "") :        
    
    if train == "train" : 
        tokenizer = Tokenizer(num_words=5000) # 빈도 높은 단어 5000개만 사용
        tokenizer.fit_on_texts(data) # train 데이터에 적용

        x_train = tokenizer.texts_to_sequences(data) # 문장 내에 각 단어에 tokenizer에 따른 index 부여
        x_train = sequence.pad_sequences(x_train, value=0.0, padding='post', maxlen=maxlen) #모두 같은 길이가 되도록 0으로 채움
        
        vocab_size = len(tokenizer.word_index) + 1        

        train_words = []
        for line in data.values.tolist() :
            words = text_to_word_sequence(line)
            train_words.append(words)
        return train_words, x_train, tokenizer, vocab_size
    
    elif train == "test" :
        tokenizer = tokenizer
        x_test = tokenizer.texts_to_sequences(data) 
        x_test = sequence.pad_sequences(x_test, value=0.0, padding='post', maxlen=maxlen)
        return x_test

train_words, X_train, tokenizer, vocab_size  = Tokenizing(text_train, maxlen, "train")
x_test = Tokenizing(text_test, maxlen, "test", tokenizer)


# In[7]:


word_index = tokenizer.word_index


# In[8]:


def load_embedding(filename):
    embeddings_index = {}

    f = open(filename, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def get_weight_matrix(embeddings_index, vocab):
    vocab_size = len(vocab) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in vocab.items():
        if i > vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        #print(embedding_vector)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    print(embedding_matrix.shape)
    return embedding_matrix


EMBEDDING_DIM = 128

w2v = Word2Vec(sentences = train_words, size= EMBEDDING_DIM, window = 5, iter = 10,
                workers = 5, # number of threads
                min_count=10)
print('w2v vocabulary size: ', len(list(w2v.wv.vocab)))
w2v.wv.save_word2vec_format('W2V_RESULT.txt', binary=False)

# load pretrained embeddings
raw_embedding = load_embedding('W2V_RESULT.txt')
embedding_matrix = get_weight_matrix(raw_embedding, word_index)


# In[9]:


def build_bilstm(word_index, embeddings_dict, nclasses,  
                 MAX_SEQUENCE_LENGTH=500, 
                 EMBEDDING_DIM=128, 
                 dropout=0.5, hidden_layer = 0, lstm_node = 64):
    # Initialize a sequebtial model
    model = Sequential()
            
    # Add embedding layer
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=False))
    # Add hidden layers 
    for i in range(0,hidden_layer):
        model.add(Bidirectional(LSTM(lstm_node, return_sequences=True, recurrent_dropout=0.2)))
        model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(lstm_node, recurrent_dropout=0.2)))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(nclasses, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


# In[10]:


model = build_bilstm(word_index, raw_embedding, 1)
model.summary()


# In[11]:


history = model.fit(X_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=20,
                    batch_size=1024,
                    callbacks=[callback],
                    verbose=1)


# In[13]:


predicted = model.predict_classes(x_test)


# In[15]:


print(classification_report(y_test, predicted, digits = 4))


# In[17]:


model.save("MODEL_SAVE.h5")

