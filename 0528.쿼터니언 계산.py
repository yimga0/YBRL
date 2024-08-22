from ahrs.common.orientation import acc2q
from ahrs.filters import Complementary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 각도 연산 함수 정의
def compleQ2Euler(gain, acc, gyr, num_samples):
    comple = Complementary(gain=gain)
    Q_c = np.zeros((num_samples, 4))
    Q_c[0] = acc2q(acc[0])
    for i in range(1, num_samples):
        Q_c[i] = comple.update(Q_c[i - 1], gyr=gyr[i], acc=acc[i])

    Q_c = pd.DataFrame(Q_c, columns=['w', 'x', 'y', 'z'])

    # Roll 계산
    Roll = np.arctan2(2 * (Q_c['w'] * Q_c['x'] + Q_c['y'] * Q_c['z']),
                      1 - 2 * (Q_c['x'] ** 2 + Q_c['y'] ** 2)) * 180 / np.pi

    # Pitch 계산
    Pitch = np.arcsin(2 * (Q_c['w'] * Q_c['y'] - Q_c['z'] * Q_c['x'])) * 180 / np.pi

    return Roll, Pitch


e, m, t =1,3,2
# 데이터 불러오기
for e in range(1,13):
    for m in range(1,12):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/어플실험/S%s/SW/" % e
                name ="S%sF%sR%s_SW" % (e,m,t)
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

                data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
                data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출



                #각도연산
                gain = 0.2  # 상보필터 gain
                num_samples = len(Ax)
                acc_data = np.stack((Ax, Ay, Az), axis=1)
                gyro_data = np.stack((Gx, Gy, Gz), axis=1)
                Roll, Pitch = compleQ2Euler(gain=gain, acc=acc_data, gyr=gyro_data, num_samples=num_samples)


                # 그래프 그리기
                plt.figure()
                plt.title('Roll, Pitch')
                plt.plot(Roll, label='Roll')
                plt.plot(Pitch, label='Pitch')
                plt.xlabel('Frames')
                plt.ylabel('Angle $(\degree)$')
                plt.legend(loc='upper right')

                plt.tight_layout()
                plt.show()


            except:
                pass



