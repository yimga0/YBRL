import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from keras.layers import Input, Conv1D, MaxPooling2D, UpSampling2D, Flatten, Dense, LSTM, MaxPooling1D
from keras.models import Model
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv1D, MaxPool1D, Flatten, BatchNormalization, LSTM, Add, GlobalAveragePooling1D
from keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from scipy import signal



train_data = pd.read_csv(r'C:/Users/abbey/Desktop/20명 데이터 split/train_data.csv')
test_data = pd.read_csv(r'C:/Users/abbey/Desktop/20명 데이터 split/test_data.csv')
train_data = train_data.iloc[:,:]
test_data = test_data.iloc[:,:]


#nan값찾기
# train_data.isnull().sum().sum()
# errorfind = np.transpose(train_data).isnull().sum()
# plt.plot(errorfind)
# plt.show()


data_range = 960  # 120*8
train_input = np.array(train_data.iloc[:,:data_range])
train_target_10class = np.array(train_data.iloc[:,data_range:data_range+1])

test_input = np.array(test_data.iloc[:,:data_range])
test_target_10class = np.array(test_data.iloc[:,data_range:data_range+1])



from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()

onehot_encoder_train = OneHotEncoder().fit(train_target_10class)
raw_data_cat_onehot = onehot_encoder.fit_transform(train_target_10class)
train_target_onehot = raw_data_cat_onehot.toarray()

onehot_encoder = OneHotEncoder().fit(test_target_10class)
raw_data_cat_onehot = onehot_encoder.fit_transform(test_target_10class)
test_target_onehot = raw_data_cat_onehot.toarray()

train_input = np.array(train_input)
train_input = train_input.reshape(len(train_input), 120, 8)
test_input = test_input.reshape(len(test_input), 120, 8)
input_shape = (120 ,8)
# Normalize data


x_train = train_input
x_test = test_input


split_index = int(len(x_test) * 0.5)
x_test_split = x_test[:split_index]
x_val_split = x_test[split_index:]
test_target_split = test_target_onehot[:split_index]
val_target_split = test_target_onehot[split_index:]

X_train = x_train
X_val = x_val_split
X_test = x_test_split

y_train = train_target_onehot
y_val = val_target_split
y_test = test_target_split

'---------------transformer model--------------------------------------------'
n_classes = 100


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
    output = Dense(10, activation='softmax')(x)
    return keras.Model(inputs, output)


epochs = 100
input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    dropout=0.25,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)
model.summary()


model.fit(X_train, y_train, validation_data=[X_val,y_val],epochs=epochs, batch_size=64)


model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=r'C:/Users/abbey/Desktop/20명 데이터 split/test5/{epoch}-{loss:.5f}.keras',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=1)


'---------------------------Result-----------------------------'

pred = model.predict(test_input)
pred_trans = pd.DataFrame(np.transpose(pred))
y_pred = np.zeros((len(pred),1))

for tmp in range(len(pred)):
    y_pred [tmp] = np.argmax(pred_trans[tmp]) + 1

y_true = test_target_10class
y_true = y_true.tolist()
y_pred = y_pred.tolist()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score
accuracy_score = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred,average= "macro")
recall = recall_score(y_true, y_pred,average= "macro")
f1_score1 = f1_score(y_true, y_pred, average='macro')

plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
#plt.title('HAR Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10'))
plt.xticks(rotation=45)
plt.yticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10'))
plt.yticks(rotation=0)
plt.show()
plt.tight_layout()

print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1_score1: {3:.4f}'.format(accuracy_score, precision, recall, f1_score1))

y_true_df = pd.DataFrame(y_true)
y_pred_df = pd.DataFrame(y_pred)
y_result = pd.concat([y_true_df,y_pred_df], axis=1)
y_result.columns = ['true', 'pred']
y_result['diff'] = y_result.true - y_result.pred




