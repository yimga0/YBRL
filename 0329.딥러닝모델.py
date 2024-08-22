import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


train_data = pd.read_csv(r"C:/Users/abbey/Desktop/test1(120+non)/train_data.csv")
test_data = pd.read_csv(r"C:/Users/abbey/Desktop/test1(120+non)/test_data.csv")
train_data = train_data.iloc[:,1:]
test_data = test_data.iloc[:,1:]

# import matplotlib.pyplot as plt
# # nan값찾기
# train_data.isnull().sum().sum()
# errorfind = np.transpose(train_data).isnull().sum()
# plt.plot(errorfind)
# plt.show()


data_range = 960 # 120*8
train_input = np.array(train_data.iloc[:,:data_range])
train_target_7class = np.array(train_data.iloc[:,data_range:data_range+1])

test_input = np.array(test_data.iloc[:,:data_range])
test_target_7class = np.array(test_data.iloc[:,data_range:data_range+1])



from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()

onehot_encoder_train = OneHotEncoder().fit(train_target_7class)
raw_data_cat_onehot = onehot_encoder.fit_transform(train_target_7class)
train_targe_onehot = raw_data_cat_onehot.toarray()

onehot_encoder = OneHotEncoder().fit(test_target_7class)
raw_data_cat_onehot = onehot_encoder.fit_transform(test_target_7class)
test_targe_onehot = raw_data_cat_onehot.toarray()

train_input = np.array(train_input)
train_input = train_input.reshape(len(train_input), 120, 8)
test_input = test_input.reshape(len(test_input), 120, 8)
input_shape = (120 ,8)
# test = pd.DataFrame(train_input[101])
# test.columns = ['1','2','3','4','5','6','7','8']
# plt.figure()
# plt.plot(test['7'])


# Normalize data

x_train = train_input
x_test = test_input

epochs = 200

# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv1D(256, kernel_size=1, strides=1, input_shape=input_shape, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling1D(pool_size=(2)),
        layers.Conv1D(512, kernel_size=1, strides=1, input_shape=input_shape, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling1D(pool_size=(2)),
        layers.Conv1D(512, kernel_size=1, strides=1, padding='same'),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Flatten(),
        layers.Dense(7, activation='softmax'),
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv1D(16, kernel_size=1, strides=1, input_shape=input_shape, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling1D(pool_size=(2)),
        layers.Conv1D(32, kernel_size=1, strides=1, padding='same'),
        #layers.LSTM(64, return_sequences=True),
        layers.Flatten(),
        layers.Dense(7, activation='softmax'),
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)


# Prepare the train and test dataset.
batch_size = 64
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()




# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train and evaluate teacher on data.

'---------------------------------Model Check Point Setting------------------------------------------------------------'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=r'C:/Users/abbey/Desktop/test1(120+non)/epoch/{epoch}-{loss:.5f}.keras',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=1)
'---------------------------------------------------------------------------------------------------------------------'
early_stop = keras.callbacks.EarlyStopping(patience=20, monitor='loss')

# 인공신경망 학습
hist = teacher.fit(x_train, train_targe_onehot, epochs=epochs, batch_size=64, validation_split=0.2,
                 callbacks=[model_checkpoint, early_stop])

#teacher.fit(x_train, train_targe_onehot, epochs=10)
teacher.evaluate(x_test, test_targe_onehot)



#여기까지만 돌려
'---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


teacher.summary()
student.summary()

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student

distiller.fit(x_train, train_targe_onehot, batch_size=64, epochs=epochs, validation_split=0.2)


student.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
student.evaluate(x_test, test_targe_onehot)


# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, train_targe_onehot, epochs=10)
student_scratch.evaluate(x_test, test_targe_onehot)


# student.save(r'C:\Users\fhrm5\Desktop\동작분류\DL지식\teacher2/result_knowledge.h5')
# student_scratch.save(r'C:\Users\fhrm5\Desktop\동작분류\DL지식\teacher2/result_normal.h5')
'------------------------------------------------------결과-----------------------------------------------------------'
teacher = tf.keras.models.load_model(r'C:/Users/abbey/Desktop/test1(120+non)/epoch/151-0.00369.keras')

model = teacher

pred = model.predict(test_input)
pred_trans = pd.DataFrame(np.transpose(pred))
y_pred = np.zeros((len(pred),1))

for tmp in range(len(pred)):
    y_pred [tmp] = np.argmax(pred_trans[tmp]) + 1

y_true = test_target_7class
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
plt.xticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'))
plt.xticks(rotation=45)
plt.yticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'))
plt.yticks(rotation=0)
plt.show()
plt.tight_layout()

print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1_score1: {3:.4f}'.format(accuracy_score, precision, recall, f1_score1))

y_true_df = pd.DataFrame(y_true)
y_pred_df = pd.DataFrame(y_pred)
y_result = pd.concat([y_true_df,y_pred_df], axis=1)
y_result.columns = ['true', 'pred']
y_result['diff'] = y_result.true - y_result.pred
#
test_data.iloc[929]
'--------------------------------------------------모델 비교용-------------------------------------------------------'

# pred_n = student_scratch.predict(test_input)
# pred_trans_n = pd.DataFrame(np.transpose(pred_n))
# y_pred_n = np.zeros((len(pred_n),1))
# for tmp in range(len(pred_n)):
#     y_pred_n [tmp] = np.argmax(pred_trans_n[tmp]) + 1
# y_true = test_target_7class
# y_true = y_true.tolist()
# y_pred_n = y_pred_n.tolist()
# from sklearn.metrics import confusion_matrix
# cm_n = confusion_matrix(y_true, y_pred_n)
# import seaborn as sns
# #
# plt.figure(figsize=(3,3))
# sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
# #plt.title('HAR Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.xticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'))
# plt.xticks(rotation=45)
# plt.yticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5), ('class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'))
# plt.yticks(rotation=0)
# plt.show()
# plt.tight_layout()

# from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score
# accuracy_score_n = accuracy_score(y_true, y_pred_n)
# precision_n = precision_score(y_true, y_pred_n,average= "macro")
# recall_n = recall_score(y_true, y_pred_n,average= "macro")
# f1_score1_n = f1_score(y_true, y_pred_n, average='macro')
# print('Knowledge \n accuracy: {0:.4f}, precision: {0:.4f}, recall: {1:.4f}, f1_score1: {1:.4f}'.format(accuracy_score_k, precision_k, recall_k, f1_score1_k))
# print('normal \n accuracy: {0:.4f}, precision: {0:.4f}, recall: {1:.4f}, f1_score1: {1:.4f}'.format(accuracy_score_n, precision_n, recall_n, f1_score1_n))


