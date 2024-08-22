'''
예제 출처
https://hmkim312.github.io/posts/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C_%ED%95%B4%EB%B3%B4%EB%8A%94_CNN(Convolution_Neral_Network)/
'''

'''
데이터셋 불러오기
'''
from tensorflow.keras import datasets

mnist = datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # min-max scaling

X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

'''
모델 구성
'''
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                  padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

import time

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
다중 분류를 위한 손실함수에
categorical_crossentropy --> 훈련 데이터의 label (y/target) 값이 원-핫 벡터 형태인 경우 
sparse_categorical_crossentropy --> 훈련 데이터의 label (y/target) 값이 정수(int) 형태인 경우

본 훈련데이터는 정수(0~255) 형태이므로 후자를 사용함.

출처: https://bigdaheta.tistory.com/65
다른 손실 함수에 대한 정보: https://wikidocs.net/36033


옵티마이저: 일반적으로는 Optimizer라고 합니다. 뉴럴넷의 가중치를 업데이트하는 알고리즘이라고 생각하시면 이해가 간편하실 것 같습니다.

자세한 내용: https://wikidocs.net/152765
'''

start_time = time.time()

hist = model.fit(X_train, y_train, epochs=5, verbose = 1, validation_data=(X_test, y_test))

'''
함수 인자로 verbose 옵션은 함수를 실행하면서 발생하는 정보들을 상세하게 출력할 것인지, 표준 출력으로 나타낼 것인지, 출력하지 않을 것인지를 선택할 수 있습니다.

verbose = 0  (출력하지 않음 X)
verbose = 1  (정보를 상세하게 출력함)
verbose = 2  (정보를 함축적으로 출력함)

출처: https://lungfish.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-verbose-%EC%98%B5%EC%85%98
'''

print(f'Fit Time :{time.time() - start_time}')

'''
그래프로 보기
'''

import matplotlib.pyplot as plt
import seaborn as sns

plot_target = ['loss' , 'accuracy', 'val_loss', 'val_accuracy']
plt.figure(figsize=(12, 8))

for each in plot_target:
    plt.plot(hist.history[each], label = each)
plt.legend()
plt.grid()
plt.show()

'''
Test  
'''

score = model.evaluate(X_test, y_test)
print(f'Test Loss : {score[0]}')
print(f'Test Accuracy  : {score[1]}')

'''
데이터 예측
'''

import numpy as np

predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result,  axis=1)
predicted_labels[:10]


'''
틀린 데이터만 모으기
'''
wrong_result = []
for n in range(0, len(y_test)):
    if predicted_labels[n] != y_test[n]:
        wrong_result.append(n)

len(wrong_result)



'''
틀린 데이터 16개만 직접 그려보기
'''

import random

samples = random.choices(population=wrong_result, k=16)

plt.figure(figsize=(14, 12))

for idx, n in enumerate(samples):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.title('Label ' + str(y_test[n]) + ', Predict ' + str(predicted_labels[n]))
    plt.axis('off')

plt.show()

model.save('MNIST_CNN_model.h5')
