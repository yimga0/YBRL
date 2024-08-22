import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


e, m, t =1,4,1
for e in range(1,13):
    for m in range(1,12):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/어플실험/S%s/SW/" % e
                name ="S%sD%sR%s_SW" % (e,m,t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[1:, 1:7]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                Ax = -data.Acc_Z
                Ay = data.Acc_X
                Az = -data.Acc_Y

                Gx = -data.Gyr_Z
                Gy = data.Gyr_X
                Gz = -data.Gyr_Y

                # data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                # data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출
                # data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                data1 = pd.concat([Ax,Ay,Az,Gx,Gy,Gz], axis = 1)
                data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                '----------------------------------가속도로 각도 연산--------------------------------------------'
                Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi   #가속도계 데이터 롤 각도
                Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi  # 가속도계 데이터 피치 각도

                # 상보필터
                Fs = 50
                dt = 1/Fs

                # Gx가 1차원 배열인지 확인하고 1차원 배열로 변환
                Gx = np.array(Gx).flatten()
                Roll_w = np.zeros((len(Gx), 1))  # 자이로스코프 데이터 롤 각도
                for n in range(len(Gx) - 2):
                  Roll_w[n + 2, 0] = Roll_w[n, 0] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3   #심슨룰

                Gy = np.array(Gy).flatten()
                Pitch_w = np.zeros((len(Gy), 1))   # 자이로스코프 데이터 피치 각도
                for n in range(len(Gy) - 2):
                  Pitch_w[n + 2, 0] = Pitch_w[n, 0] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3    #심슨룰

                Roll_w = Roll_w.flatten()
                Pitch_w = Pitch_w.flatten()

                a = 0.2 #가속도계 데이터의 가중치   (가속도계 데이터가 20%의 영향, 자이로스코프 데이터가 80%의 영향)
                Roll_a = Roll * a + Roll_w * (1 - a)  #최종적으로 결합된 롤 각도
                Pitch_a = Pitch * a + Pitch_w * (1 - a)   #최종적으로 결합된 피치 각도

                path_graph = r"C:/Users/abbey/Desktop/SW(각도연산)/"
                plt.figure(figsize=(10,8))
                plt.subplot(3, 1, 1)
                plt.title(name)
                plt.plot(Roll_a, label='Roll')
                plt.plot(Pitch_a, label='Pitch')
                plt.xlabel('Frames')
                plt.ylabel('Angle $(\degree)$')
                plt.legend(loc='upper right')
                plt.tight_layout()

                plt.subplot(3, 1, 2)
                plt.plot(data1.ASVM)  # 그래프그리기
                plt.ylabel('ASVM (g)')
                plt.xlabel('Frames')
                plt.tight_layout()

                plt.subplot(3, 1, 3)
                plt.plot(data1.GSVM)  # 그래프그리기
                plt.ylabel('GSVM (g)')
                plt.xlabel('Frames')
                plt.tight_layout()

                plt.show()

                plt.savefig(path_graph + name_png)
                plt.close()
            except:
                pass




'--------------------------------------------z방향 가속도-----------------------------------------------'

Acc_V_2 = []   # 가속도 z 축이 -, 단위는 g
for m in range(len(Pitch)):
    if (Az[m] > 0) and (abs(Pitch[m]) < 90):
        AccV = -np.sin(Pitch[m] / 180 * np.pi) * Ax[m] + np.cos(
            Pitch[m] / 180 * np.pi) * np.sin(
            Roll[m] / 180 * np.pi) * Ay[m] - np.cos(
            Roll[m] / 180 * np.pi) * np.cos(
            Pitch[m] / 180 * np.pi) * Az[m] + 1
        Acc_V_2.append(AccV)
    else:
        AccV = -np.sin(Pitch[m] / 180 * np.pi) * Ax[m] + np.cos(
            Pitch[m] / 180 * np.pi) * np.sin(
            Roll[m] / 180 * np.pi) * Ay[m] + np.cos(
            Roll[m] / 180 * np.pi) * np.cos(
            Pitch[m] / 180 * np.pi) * Az[m] + 1
        Acc_V_2.append(AccV)

# 그려보기
plt.title('Acc_V (Angle O)')
plt.ylabel('Acc. $(m/s^2)$')
plt.xlabel('Frames')
plt.plot(Acc_V_2)
plt.show()



'-------------------------------------------가속도 각속도 그리기(확인용)-----------------------------------------------'

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2, 1, 1)
plt.title('Acceleration')
plt.plot(Ax, label='Acc_X')
plt.plot(Ay, label='Acc_Y')
plt.plot(Az, label='Acc_Z')
plt.xlabel('Frames')
plt.ylabel('Acc. $(g)$')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.title('Angular Velocity')
plt.plot(Gx, label='Gyr_X')
plt.plot(Gy, label='Gyr_Y')
plt.plot(Gz, label='Gyr_Z')
plt.xlabel('Frames')
plt.ylabel('Gyro. $(\degree/s)$')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
