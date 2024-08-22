import pickle  # 데이터 저장
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv1D, MaxPool1D, Flatten, BatchNormalization, LSTM, Add, GlobalAvgPool1D
from matplotlib import pyplot as plt

MAE = tf.keras.losses.mean_absolute_error
MSE = tf.keras.losses.mean_squared_error
#Error_Function = 'MAE'


data_input = pd.read_csv(r"C:\Users\fhrm5\Desktop\gait\딥러닝_수정2\data_input.csv")
data_target = pd.read_csv(r"C:\Users\fhrm5\Desktop\gait\딥러닝_수정2\data_target.csv")


train_input = np.array(data_input.iloc[:278,1:601])
train_target = np.array(data_target.iloc[:278,1:101])

test_input = np.array(data_input.iloc[278:,1:601])
test_target = np.array(data_target.iloc[278:,1:101])


train_input = np.array(train_input)
train_input = train_input.reshape(len(train_target), 100, 6)
test_input = test_input.reshape(len(test_target), 100, 6)
input_shape = (100, 6)

start = time.time()
epochs = 200

'------------------------------------------------CNN-LSTM------------------------------------------------------'
#F1, F2, F3, F4 = 32, 32, 64, 8
Error_Function = 'MSE'
for Error_Function in ['MAE', 'MSE']:
# train_input = train_stack.reshape(len(train_target), 50, 8)
# test_input = test_stack.reshape(len(test_target), 50, 8)
input_shape = (100, 6)


# 인공신경망 생성 padding='same',
model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu', strides=1, input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, kernel_size=2, activation='relu', strides=1, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, kernel_size=2, activation='relu', strides=1, padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(keras.layers.LSTM(64, return_sequences=True))
model.add(keras.layers.LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(100))

MAE = tf.keras.losses.mean_absolute_error
model.compile(optimizer='adam', loss=MAE)

# '--------------------------------------------------------------------------------------------------------------------'
model.summary()
'---------------------------------Model Check Point Setting------------------------------------------------------------'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=r'C:\Users\fhrm5\Desktop\gait\딥러닝_수정2\GRF\Conv_%s_2/'
                                                            '{epoch}-{val_loss:.5f}.h5' %(Error_Function),
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   verbose=1)
'---------------------------------------------------------------------------------------------------------------------'
early_stop = keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')

# 인공신경망 학습
hist = model.fit(train_input, train_target, epochs=epochs, batch_size=1,
                 validation_data=(test_input, test_target),
                 callbacks=[model_checkpoint, early_stop])

round(((time.time() - start))/60, 1)