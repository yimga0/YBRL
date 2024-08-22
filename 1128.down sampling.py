import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample_poly
from scipy.signal import butter, filtfilt
import numpy as np
import os
import natsort
import seaborn as sns

for n in range(9, 13):
    #################### 피험자 번호 ####################
    subject_num = 23

    if subject_num < 10:
        subject_num = str(subject_num).zfill(2)
    else:
        subject_num = subject_num

    #################### 파일 이름 리스트 만들기 ####################
    count = 0
    a = []
    for type in ['D', 'F']:

        if type == 'D':

            for motion_num in range(1, 15):
                m = type + str(motion_num).zfill(2)

                for trial_num in range(1, 4):
                    r = 'R' + str(trial_num).zfill(2)

                    if motion_num == 1:
                        filename = 'P%s%s%s.csv' % (subject_num, m, r)
                        count += 1
                        a.append(filename)
                        break
                    else:
                        filename = 'P%s%s%s.csv' % (subject_num, m, r)
                        count += 1
                        a.append(filename)

        elif type == 'F':

            for motion_num in range(1, 12):
                m = type + str(motion_num).zfill(2)

                for trial_num in range(1, 4):
                    r = 'R' + str(trial_num).zfill(2)
                    filename = 'P%s%s%s.csv' % (subject_num, m, r)
                    count += 1
                    a.append(filename)


    #################### 폴더 경로 불러오기 ####################
    folder_path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리//P%s/ble/' % subject_num

    export_path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/ble_re/' % subject_num # 내보내기 경로

    folder = os.listdir(folder_path)

    folder_list = pd.DataFrame(folder, columns=['name'])
    folder_list = folder_list[~folder_list['name'].str.contains('.zip')]
    folder_list = folder_list.reset_index(drop=True)

    i = 1

    #################### 파일 불러오기 및 전처리 ####################
    for i in range(len(folder_list)):

        raw_path = folder_path + folder_list.name[i] + '/Raw data (g).csv'
        raw = pd.read_csv(raw_path)
        raw.columns = ['Time', 'X', 'Y', 'Z', 'Tag']

        # 가속도 데이터
        acc = raw[raw['Tag'] == 1]
        acc = acc.iloc[:, :4]
        acc.columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z']
        acc = acc.reset_index(drop=True)
        # 가속도 리샘플링
        acc_Fs = (len(acc)) / (acc.Time[len(acc)-1] - acc.Time[0])
        acc_new_samples = int(len(acc) * (30 / acc_Fs))  # 30Hz 샘플 수
        acc_resample = resample_poly(acc, acc_new_samples, len(acc))
        acc_resample = pd.DataFrame(acc_resample)
        acc_resample = acc_resample.iloc[:, 1:4]
        acc_resample.columns = ['Acc_X', 'Acc_Y', 'Acc_Z']

        # plt.figure()
        # plt.plot(acc_resample)
        # len(acc)
        # len(gyr)

        # 각속도 데이터
        gyr = raw[raw['Tag'] == 2]
        gyr = gyr.iloc[:, :4]
        gyr.columns = ['Time', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        gyr = gyr.reset_index(drop=True)
        # 각속도 리샘플링
        gyr_Fs = (len(gyr) - 1) / (gyr.Time[len(gyr) - 1] - gyr.Time[0])
        gyr_new_samples = int(len(gyr) * (30 / gyr_Fs))  # 30Hz 샘플 수
        gyr_resample = resample_poly(gyr, gyr_new_samples, len(gyr))
        gyr_resample = pd.DataFrame(gyr_resample)
        gyr_resample = gyr_resample.iloc[:, 1:4]
        gyr_resample.columns = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']

        # plt.figure()
        # plt.plot(gyr_resample)

        BLE_data = pd.concat([acc_resample, gyr_resample], axis=1)


        BLE_data.to_csv(export_path + a[i])


        #################### 확인용 그래프 ####################
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title(a[i].replace('.csv', ''))
        plt.plot(acc_resample)
        plt.xlabel('Frames')
        plt.ylabel('Acceleration (g)')

        plt.subplot(2, 1, 2)
        plt.plot(gyr_resample)
        plt.xlabel('Frames')
        plt.ylabel('Angular velocity (degree/s)')

        plt.tight_layout()

        plt.savefig('C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/ble_re/그래프/%s' % (subject_num, a[i].replace('.csv', '.png')))
        plt.close()




        gyr_Fs = (len(gyr) - 1) / (gyr.Time[len(gyr) - 1] - gyr.Time[0])
        gyr_new_samples = int(len(gyr) * (30 / gyr_Fs))  # 30Hz 샘플 수
