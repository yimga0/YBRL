import pickle  # 데이터 저장
import sqlite3  # 데이터 베이스 불러오는 모듈
import matplotlib.pyplot as plt  # 그래프 그리기
import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns



train_data = np.array(train_data.iloc[:,:959])
train_target = np.array(train_data.iloc[:,959:])

test_data = np.array(data_input.iloc[278:,1:601])
test_target = np.array(data_target.iloc[278:,1:101])

train_data =pd.DataFrame(train_data)
train_target =pd.DataFrame(train_target)

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