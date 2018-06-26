import os
import numpy as np
from HTK import HTKFile
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Reshape, GRU, Lambda, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TerminateOnNaN, LambdaCallback

batch_size = 32
epochs = 50

label_map = {"liN": 0, "#i": 1, "#er": 2, "san": 3, "sy": 4, "#u": 5,
             "liou": 6, "qi": 7, "ba": 8, "jiou": 9, "blank": 10}

# load labels
labels = {}
MFCC = HTKFile()
with open("labels/Clean08TR.mlf") as f:
    f.readline()
    key = None
    value = []
    for line in f:
        if line[0] == '\"':
            key = line[4:-6]
        elif line[0] == '.':
            labels[key] = value
            value = []
        elif line != "sil\n":
            value.append(label_map[line[:-1]])

# load training data
train_x = []
train_y = []
input_lengths = []
label_lengths = []
filelist = os.listdir(os.getcwd() + "/MFCC/training")
filelist.sort()
for filename in filelist:
    MFCC.load("MFCC/training/" + filename)
    # train_x.append(np.swapaxes(np.reshape(np.array(MFCC.data), (-1, 3, 13)), 1, 2))
    train_x.append(np.array(MFCC.data))
    train_y.append(np.array(labels[filename[1:-4]]))
    input_lengths.append(len(MFCC.data))
    label_lengths.append(len(labels[filename[1:-4]]))
    # label_lengths.append(7)

# print (len(train_x))
# print (train_x[0], train_y[0])
train_x = pad_sequences(train_x, dtype='float', padding='post')
train_y = pad_sequences(train_y, dtype='int32', padding='post')
input_lengths = np.array(input_lengths)
label_lengths = np.array(label_lengths)
print(train_x.shape, train_y.shape, input_lengths.shape, label_lengths.shape)

# wrapper for the Lambda layer
def ctc_wrapper(args):
    return K.ctc_batch_cost(*args)

inputs = Input(shape=(None, train_x.shape[2]))
train_labels = Input(shape=(None,))
input_length = Input(shape=(1,))
label_length = Input(shape=(1,))
# x = Conv1D(16, 3, activation='relu')(inputs)
# x = Conv1D(16, 3, activation='relu')(x)
# x = Conv1D(16, 3, activation='relu')(x)
x = TimeDistributed(Dense(128, activation='relu'))(inputs)
x = TimeDistributed(Dense(128, activation='relu'))(x)
x = TimeDistributed(Dense(128, activation='relu'))(x)
x = GRU(128, return_sequences=True, activation='relu')(x)
x = TimeDistributed(Dense(128, activation='relu'))(x)
y = TimeDistributed(Dense(11, activation='softmax'))(x)
loss_output = Lambda(ctc_wrapper)([train_labels, y, input_length, label_length])
model = Model(inputs=[inputs, train_labels, input_length, label_length], outputs=loss_output)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
# model.summary()

# the model for testing which outputs the softmax result of each timestep
test_model = Model(inputs=inputs, outputs=y)
# def debug_pred(batch, logs):
#     pred = test_model.predict(train_x[0:1])
#     print(pred)
# debug_cb = LambdaCallback(on_batch_end=debug_pred)

model.fit([train_x, train_y, input_lengths, label_lengths], np.zeros(train_x.shape[0]),
          batch_size=batch_size, epochs=epochs, validation_split=0.05)

# only save the model for testing
test_model.save("model.h5")
