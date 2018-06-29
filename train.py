import numpy as np
import pandas as pd
import pickle
import collections
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, concatenate
from keras.layers import Embedding, Flatten, BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.metrics import f1_score

ModelCheckPoint_filename='googlereg0.01_model.{epoch:02d}-{acc:.4f}.h5'
log_filename='log.csv'

with open('./pickles/train_set_google.pickle', 'rb') as handle:
    yTrain, sentenceTrain, positionTrain1, positionTrain2 = pickle.load(handle)
with open('./pickles/test_set_google.pickle', 'rb') as handle:
    yTest, sentenceTest, positionTest1, positionTest2 = pickle.load(handle)
with open('./pickles/embeddings_google.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)


max_position = max(np.max(positionTrain1), np.max(positionTrain2)) + 1
position_dims = 50
filters = 100
kernel_size = [3,4,5]
batch_size = 64
epochs = 100

Y_train = np_utils.to_categorical(yTrain, 19)

distanceModel1_input = Input(shape = (positionTrain1.shape[1],))
distanceModel1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(distanceModel1_input)
distanceModel2_input = Input(shape = (positionTrain2.shape[1],))
distanceModel2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])(distanceModel2_input)
wordModel_input = Input(shape = (sentenceTrain.shape[1],))
wordModel = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False)(wordModel_input)

cat = concatenate([wordModel, distanceModel1, distanceModel2])
conv0 = Conv1D(filters, kernel_size[0], padding='valid', activation='linear', strides=1)(cat)
conv0 = LeakyReLU(alpha = 0.1)(conv0)
conv0 = GlobalMaxPooling1D()(conv0)
conv0 = Dropout(0.7)(conv0)
conv1 = Conv1D(filters, kernel_size[1], padding='valid', activation='linear', strides=1)(cat)
conv1 = LeakyReLU(alpha = 0.1)(conv1)
conv1 = GlobalMaxPooling1D()(conv1)
conv1 = Dropout(0.7)(conv1)
conv2 = Conv1D(filters, kernel_size[2], padding='valid', activation='linear', strides=1)(cat)
conv2 = LeakyReLU(alpha = 0.1)(conv2)
conv2 = GlobalMaxPooling1D()(conv2)
conv2 = Dropout(0.7)(conv2)
fc = concatenate([conv0, conv1, conv2])
fc = Dense(19, activation = 'softmax')(fc)

model = Model(inputs = [wordModel_input, distanceModel1_input, distanceModel2_input], outputs = fc)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
model.summary()

# cllbks = [
#     CSVLogger(log_filename, append=True, separator=';'),
#     TensorBoard(log_dir='./Graph'),
#     ModelCheckpoint(ModelCheckPoint_filename, monitor='acc', verbose=1, save_best_only=False, mode='auto', period=5)
# 		]
model.fit([sentenceTrain, positionTrain1, positionTrain2], Y_train, batch_size=batch_size, verbose=True,epochs=59) 
model.save('model.h5')









