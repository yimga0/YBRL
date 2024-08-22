import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt
# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns

path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/S02D12R03_SW.csv"

# e , m, t = 9, 11, 1
#
# e = str(e).zfill(2)
# m = str(m).zfill(2)
# t = str(t).zfill(2)
#
#
# name = "P09D11R01"
# name_csv = "%s%s.csv" % (path, name)
# name_png = "%s.png" % (name)

data = pd.read_csv(str(path))
data = pd.DataFrame(data)
data = data.iloc[:, 2:8]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

Ax = -data.Acc_Z
Ay = data.Acc_X
Az = -data.Acc_Y

Gx = -data.Gyr_Z
Gy = data.Gyr_X
Gz = -data.Gyr_Y

# data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
# data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출
# data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

'----------------------------------가속도로 각도 연산--------------------------------------------'
Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi  # 가속도계 데이터 롤 각도
Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi  # 가속도계 데이터 피치 각도

# 상보필터
Fs = 50
dt = 1 / Fs

# Gx가 1차원 배열인지 확인하고 1차원 배열로 변환
Gx = np.array(Gx).flatten()
Roll_w = np.zeros((len(Gx), 1))  # 자이로스코프 데이터 롤 각도
for n in range(len(Gx) - 2):
    Roll_w[n + 2, 0] = Roll_w[n, 0] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3  # 심슨룰

Gy = np.array(Gy).flatten()
Pitch_w = np.zeros((len(Gy), 1))  # 자이로스코프 데이터 피치 각도
for n in range(len(Gy) - 2):
    Pitch_w[n + 2, 0] = Pitch_w[n, 0] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3  # 심슨룰

Roll_w = Roll_w.flatten()
Pitch_w = Pitch_w.flatten()

a = 0.2  # 가속도계 데이터의 가중치   (가속도계 데이터가 20%의 영향, 자이로스코프 데이터가 80%의 영향)
Roll_a = Roll * a + Roll_w * (1 - a)  # 최종적으로 결합된 롤 각도
Pitch_a = Pitch * a + Pitch_w * (1 - a)  # 최종적으로 결합된 피치 각도



# 그래프 한번에 여러개 그리기

plt.figure(figsize=(16, 10))
plt.subplot(6, 1, 1)
plt.ylabel('$m/s^2$')
# plt.plot(acc ,label='ACC_X''Acc_Y''Acc_Z' )
plt.plot(data['Acc_X'], label='Acc_X')
plt.plot(data['Acc_Y'], label='Acc_Y')
plt.plot(data['Acc_Z'], label='Acc_Z')
plt.gca().set_yticklabels([f'{x:.1f}' for x in np.arange(0, np.ceil(data.ASVM.max()) + 1, 0.5)])  # 눈금 라벨을 소수점 한 자리로 설정
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


plt.subplot(6, 1, 2)
plt.ylabel('$g$')
plt.plot(data.ASVM, label='ASVM')
plt.legend(loc='upper right', fontsize=8)
plt.gca().set_yticklabels([f'{x:.1f}' for x in np.arange(0, np.ceil(data.ASVM.max()) + 1, 0.5)])  # 눈금 라벨을 소수점 한 자리로 설정
plt.tight_layout()

plt.subplot(6, 1, 3)
plt.ylabel('degree/s')
plt.plot(data['Gyr_X'], label='Gyr_X')
plt.plot(data['Gyr_Y'], label='Gyr_Y')
plt.plot(data['Gyr_Z'], label='Gyr_Z')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


plt.subplot(6, 1, 4)
plt.ylabel('degree/s')
plt.plot(data.GSVM, label='GSVM')
# plt.xlim(100, 500)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


plt.subplot(6, 1, 5)
plt.ylabel('degree')
plt.plot(Roll_a, label='Roll')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

plt.subplot(6, 1,6)
plt.ylabel('degree')
plt.plot(Pitch_a, label='Roll')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

'-------------------------------------------------------------------------------------------------------------------'

max_index = data['ASVM'].idxmax()

split = max_index + 100

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.title('S02D12R03_SW')
plt.plot(data1.ASVM, label='ASVM')
plt.ylabel('ASVM')
plt.legend(loc='upper right', fontsize=8)
plt.axvline(split, color='r', linestyle='-')
plt.gca().set_yticklabels([f'{x:.1f}' for x in np.arange(0, np.ceil(data.ASVM.max()) + 1, 0.5)])  # 눈금 라벨을 소수점 한 자리로 설정
plt.tight_layout()


plt.subplot(3, 1, 2)
plt.ylabel('GSVM')
plt.plot(data1.GSVM, label='GSVM')
plt.axvline(split, color='r', linestyle='-')
# plt.xlim(100, 500)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


plt.subplot(3, 1, 3)
plt.ylabel('Angle (degree)')
plt.plot( Roll, label='Roll')
plt.plot( Pitch, label= 'Pitch')
plt.legend(loc='upper right', fontsize=8)
plt.axvline(split, color='r', linestyle='-')
plt.tight_layout()
plt.show()



