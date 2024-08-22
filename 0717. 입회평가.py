import numpy as np
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
import random


Th_Acc = 0.9
Th_Gyro = 80
Th_Ang = 20
Grid_list = []

exception = list([7, 8, 9, 17])
Elderly = list(set(range(1,26)) - set(exception))


subject = range(13,20)
movement_exception = list([23])
movement = list(set(range(1,40)) - set(movement_exception))

# for Th_Acc in np.arange(0.7, 0.91,0.05):
#     for Th_Gyro in range(80, 151, 10):
#         for Th_Ang in range(10, 31,1):
for Th_Acc in np.arange(0.6, 0.9, 0.1):
    for Th_Gyro in range(80, 111, 10):
        for Th_Ang in range(20, 31, 5):
            fs = 50  # 50Hz
            #print(Th_Ang)
            file_name = []
            Lead_Time = []
            FN_Error = []
            FP_Error = []
            TP_name=[]

            count = 0
            TP, TN, FP, FN = 0, 0, 0, 0

            S = 13
            D, R = 18, 2
            for S in subject:
                for D in movement:
                    for R in range(1, 4):
                        try:
                            detection_frame = 0
                            impact = 0
                            lead_time = 0

                            S = str(S).zfill(2)
                            R = str(R).zfill(2)


                            path = r"C:/Users/abbey/Desktop/입회평가/"

                            if D <= 21:
                                ADL = D
                                d = str(ADL).zfill(2)
                                name = 'S%sD%sR%s_SW_v2.csv' % (S, d, R)
                            elif D >= 22:
                                Fall = D - 22
                                d = str(Fall).zfill(2)
                                name = 'S%sF%sR%s_SW_v2.csv' % (S, d, R)

                            name_png = '%s.png' % name

                            data = pd.read_csv(path + name).iloc[:, 2:8]


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

                            # path_graph = r"C:/Users/abbey/Desktop/SW(각도연산)/"
                            # plt.figure(figsize=(10, 8))
                            # plt.subplot(3, 1, 1)
                            # plt.title(name)
                            # plt.plot(Roll_a, label='Roll')
                            # plt.plot(Pitch_a, label='Pitch')
                            # plt.xlabel('Frames')
                            # plt.ylabel('Angle $(\degree)$')
                            # plt.legend(loc='upper right')
                            # plt.tight_layout()
                            #
                            # plt.subplot(3, 1, 2)
                            # plt.plot(data1.ASVM)  # 그래프그리기
                            # plt.ylabel('ASVM (g)')
                            # plt.xlabel('Frames')
                            # plt.tight_layout()
                            #
                            # plt.subplot(3, 1, 3)
                            # plt.plot(data1.GSVM)  # 그래프그리기
                            # plt.ylabel('GSVM (g)')
                            # plt.xlabel('Frames')
                            # plt.tight_layout()
                            #
                            # plt.show()

                            '---------------------------------------검출 알고리즘---------------------------------------'
                            data['Roll'] = Roll_a
                            data['Pitch'] = Pitch_a


                            Roll = abs(data.Roll)
                            Pitch = abs(data.Pitch)
                            impact = np.argmax(data.ASVM)

                            n = 3
                            n_samples = len(data)
                            detection_frame = 0
                            tmp = 200
                            for tmp in range(len(data)):
                                if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                                    fall_flag = tmp
                                    for tmp_2 in range(tmp, tmp + 6):
                                        if data.GSVM[tmp_2] >= Th_Gyro:
                                            fall_flag_2 = tmp_2
                                            #tmp_3 =206
                                            for tmp_3 in range(tmp_2, tmp_2 + 6):
                                                if Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang:
                                                    detection_frame = tmp_3
                                                    break

                            #
                            # plt.subplot(3,1,1)
                            # plt.title(name)
                            # plt.plot(data.ASVM)
                            #
                            # plt.subplot(3, 1, 2)
                            # plt.plot(data.GSVM)
                            # plt.subplot(3,1,3)
                            # plt.plot(Roll)
                            # plt.plot(Pitch)

                            if detection_frame > 0:
                                lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms

                            if lead_time > 0 and lead_time <= 2000:

                                if D <= 21:
                                    FP += 1
                                    FP_Error.append(name)
                                elif D >= 22:
                                    TP += 1
                                    TP_name.append(name)
                                    Lead_Time.append(lead_time)

                            else:
                                if D <= 21:
                                    TN += 1
                                elif D >= 22:
                                    FN += 1
                                    FN_Error.append(name)

                            count += 1

                            file_name.append(name)

                        except:
                            pass

            TP + TN + FN + FP

            Accuracy = round(((TP) + (TN)) / ((TP) + (FP) + (FN) + (TN)) * 100, 2)
            Sensitivity = round(TP / ((TP) + (FN)) * 100, 2)
            Specificity = round((TN) / ((TN) + (FP)) * 100, 2)
            Lead_Time_mean = round(np.mean(Lead_Time), 2)
            Lead_Time_std = round(np.std(Lead_Time), 2)
            print('\nAccuracy  Sensitivity  Specificity  LeadTime ')
            print('%s     %s        %s        %s ± %s     Thresholds:  %s, %i, %i' % (
                round(Accuracy, 2), round(Sensitivity, 2), round(Specificity, 2), Lead_Time_mean, Lead_Time_std, Th_Acc, Th_Gyro, Th_Ang))
            Grid = [Th_Acc, Th_Gyro, Th_Ang, Accuracy, Sensitivity, Specificity, Lead_Time_mean, Lead_Time_std]
            # Grid = [Wn1, n2, Accuracy, Sensitivity, Specificity, Lead_Time_mean, Lead_Time_std]

            Grid_list.append(Grid)
Grid_list_df = pd.DataFrame(Grid_list)
Grid_list_df.columns = ['Th_Acc', 'Th_Gyro', 'Th_Ang', 'Accuracy', 'Sensitivity', 'Specificity', 'Lead_Time_mean',
                        'Lead_Time_std']























# #
# accuracy_max = np.max(Grid_list_df.Accuracy)
# sensitivity_max = np.max(Grid_list_df.Sensitivity)
# accuracy_max_index = np.argmax(Grid_list_df.Accuracy)
# sensitivity_max_index = np.argmax(Grid_list_df.Sensitivity)
#
# print(f"최고 정확도: {accuracy_max}")
# print(f"최고 민감도: {sensitivity_max}")
#
# print(f"전도 검출 민감도: {round(Sensitivity, 2)}")
# print(f"전도 검출 특이도: {round(Specificity, 2)}")
# print(f"전도 검출 정확도: {round(Accuracy, 2)}")
# print(f"Lead time: {round(Lead_Time_mean, 2)}")