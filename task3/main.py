# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
         'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combined = [train_df, test_df]
train_df


# %%
labels = {}
for i, val in enumerate(AMINO_ACIDS):
    labels[val] = i

labels


# %%
def encode_seq(df):
    data_encoded = []
    for row in df['Sequence'].values:
        row_encoded = []
        for c in row:
            row_encoded.append(labels[c])
        data_encoded.append(np.array(row_encoded))

    return data_encoded

X = np.array(encode_seq(train_df))
y = to_categorical(train_df['Active'].values)


# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# %%
embedding_dim = 8

model = Sequential()
model.add(Embedding(20, embedding_dim, input_length=4))
model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Conv1D(filters=8, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# %%
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=256)


# %%
# make submission
X_test = np.array(encode_seq(test_df))

y_test = model.predict_classes(X_test)
submission = pd.DataFrame(y_test)
submission.to_csv('submission.csv', header=False, index=False)


# %%


