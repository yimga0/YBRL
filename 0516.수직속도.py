import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt


e, m, t =10,1,2
e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = r"C:/Users/abbey/Desktop/어플실험/S10/sw/"
name = "S%sF%sR%s_SW" % (e,m,t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=1)
data = pd.DataFrame(data)
data = data.iloc[:,1:7]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
data['ASVM'] = np.sqrt(data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2)
Acc_V = np.sqrt(data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) - 1   #정지 상태에서의 가속도(1g)를 제거

#가속도를 적분하여 수직속도 구하기
dt = 0.02
'----------------------------------------------------근사 수직 가속도----------------------------------------------------'

VV1 = np.zeros((len(Acc_V), 1))
for i in range(len(Acc_V) - 1):
    VV1[i + 1] = VV1[i] + (Acc_V[i] + Acc_V[i + 1]) * dt / 2


'----------------------------------------------------심슨룰-------------------------------------------------------------'
#가속도가 일정 이상일 때 심슨의 법칙을 사용하여 적분하고, 가속도가 낮을 때는 감쇠 처리를 통해 속도를 점차 줄어들게 함
y = np.zeros((len(data.Acc_X), 1))    #data.Acc_X와 동일한 길이의 0으로 채워진 배열y
for n in range(len(data.Acc_X) - 2):  #data.Acc_X 배열의 길이에서 2를 뺀 횟수만큼 반복
  y[n + 2] = y[n] + (data.Acc_X[n] + 4*data.Acc_X[n + 1] + data.Acc_X[n + 2]) * dt / 3  # 가속도 적분하여 y배열에 누적


#심슨룰 변형 - 상쇄 조건 적용 적분
Vel_V = np.zeros((len(Acc_V), 1))   #Acc_V와 동일한 길이의 0으로 채워진 배열 Vel_V
for k in range(len(Acc_V) - 2):     #Acc_V 배열의 길이에서 2를 뺀 횟수만큼 반복
    if Acc_V[k] > 0.24:
        Vel_V[k + 2] = Vel_V[k] + 9.8 * (Acc_V[k] + 4 * Acc_V[k + 1] + Acc_V[k + 2]) * dt / 3   #수직가속도가 0.24g 보다 클 때 적분하고 Vel_V 배열에 누적
    else:
        Vel_V[k + 2] = Vel_V[k + 1] * 0.9    #수직 가속도가 0.24g보다 작을 경우, 이전 값에 0.9를 곱하여 Vel_V 배열에 저장



'------------------------------------------------그래프 두개씩 비교--------------------------------------------------------'

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(data.ASVM )
plt.title(name)
plt.xlabel('Frames')
plt.ylabel('ASVM(g)')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.subplot(2,1,2)
plt.plot(VV1)
plt.xlabel('Frames')
plt.ylabel('Vertical Velocity(m/s)')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(data.ASVM )
plt.title(name)
plt.xlabel('Frames')
plt.ylabel('ASVM(g)')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.subplot(2,1,2)
plt.plot(Vel_V)
plt.xlabel('Frames')
plt.ylabel('Simpsons rule(m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



'------------------------------------------------그래프 한번에 3개---------------------------------------------------------'
plt.figure(figsize=(10, 6))
plt.subplot(3,1,1)
plt.plot(data.ASVM )
plt.title('ASVM')
plt.xlabel('Time')
plt.ylabel('Vertical Velocity')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.subplot(3,1,2)
plt.plot(VV1)
plt.title('Vertical Velocity over Time')
plt.xlabel('Time')
plt.ylabel('Vertical Velocity')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.subplot(3,1,3)
plt.plot(Vel_V)
plt.title('Simpsons rule')
plt.xlabel('Time')
plt.ylabel('Vertical Velocity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
