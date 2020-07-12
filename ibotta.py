import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# set seed
np.random.seed(578)

# import data
ibotta_train = pd.read_csv("ibotta_train.csv")
ibotta_test = pd.read_csv("ibotta_test.csv")
print("training data set:", ibotta_train.shape)
print("test data set:", ibotta_test.shape)

# combine training data set and test data set
ibotta_all = pd.concat([ibotta_train, ibotta_test])
ibotta_all = ibotta_all.reset_index(drop=True)
print(ibotta_all.shape)

# preprocessing of ibotta data (both training and test data sets)
max_words = 10000
max_len = 25
training_samples = 6000
validation_samples = 1000
test_samples = 1000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(ibotta_all.Name)
sequences = tokenizer.texts_to_sequences(ibotta_all.Name)

word_index = tokenizer.word_index
print("found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(ibotta_train.Cat_code)
print("shape of data tensor:", data.shape)
print("shape of label tensor:", labels.shape)

# split the data into training and test
data_train = data[:8000]
data_test = data[-1999:]
print("shape of data tensor that we'll make predictions on:", data_test.shape)

# split training data into training and validation
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_train = data_train[indices]
labels = labels[indices]

x_train = data_train[:training_samples]
y_train = labels[:training_samples]
y_train = to_categorical(y_train, 7)
x_val = data_train[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
y_val = to_categorical(y_val, 7)
x_test = data_train[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = to_categorical(y_test, 7)
print("shape of training data tensor:", x_train.shape)
print("shape of validation data tensor:", x_val.shape)
print("shape of test data tensor:", x_test.shape)

# check: reverse translate from integers to text
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decode_name = ' '.join(
    reverse_word_index.get(i, "?") for i in data_test[0]
)

# model architecture
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))
model.summary()

# compile
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit model
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(x_val, y_val))

# test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test loss:', test_loss)
print('test accuracy:', test_acc)

# plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

# submission
prediction = model.predict_classes(data_test)
submission = pd.DataFrame({'Id': range(8001, 10000), 'Cat_code': prediction})
submission.to_csv("submission3.csv", index=False)
