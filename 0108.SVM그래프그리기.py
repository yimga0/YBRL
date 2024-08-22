import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt
# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns

path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P10/ble_re/P10F06R01.csv"

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
data = data.iloc[:, 1:7]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

data['Acc_X'] = data.Acc_X
data['Acc_Y'] = data.Acc_Y
data['Acc_Z'] = data.Acc_Z
data['Gyr_X'] = data.Gyr_X
data['Gyr_Y'] = data.Gyr_Y
data['Gyr_Z'] = data.Gyr_Z
data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

data = data.iloc[1:, :]

plt.figure()
plt.plot(data.GSVM)  # 그래프그리기
plt.ylabel('GSVM (g)')
plt.xlabel('Frames')
plt.show()




'---------------------------------------------------------------------------------------------------------------------'


#
#
#
# 그래프 한번에 여러개 그리기
# plt.figure(figsize=(10, 8))
# plt.subplot(4, 1, 1)  #세로, 가로, 몇번째
# plt.title('Acceleration SVM')
# plt.ylabel('ASVM($g$)')
# plt.xlabel('Frames')
# plt.plot(Acc_RMS, label='ASVM')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_ACC, label='Threshold')
# # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)
#
# plt.subplot(4, 1, 2)
# plt.title('Angular Velocity SVM')
# plt.ylabel('GSVM($\degree/s$)')
# plt.xlabel('Frames')
# plt.plot(Ang_RMS, label='GSVM')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_GYR, label='Threshold')
# # # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)
#
# plt.subplot(4, 1, 3)
# plt.title('Angle')
# plt.ylabel('Angle($\degree$)')
# plt.xlabel('Frames')
# plt.plot(Modified_Pitch, label='Pitch')
# plt.plot(Modified_Roll, label='Roll')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_Angle, label='Threshold')
# plt.axhline(-TH_Angle)
# # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)
#
# plt.subplot(4, 1, 4)
# plt.title('Vertical Velocity')
# plt.ylabel('Vel_V($m/s$)')
# plt.xlabel('Frames')
# plt.plot(Vel_V, label='Vel_V')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_VV, label='Threshold')
# plt.axhline(TH_VV2, label='Threshold2')
# # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)
#
# plt.tight_layout()
# plt.savefig('23 12.28 RIS 아두이노 그래프/검출 그래프/' + file.replace('.csv', '.png'))
# plt.close()
#
#
# xticks(), yticks()  #x,y축 눈금 그리기