import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample_poly
from scipy.signal import butter, filtfilt
import numpy as np
import os
import natsort   #숫자정렬
import seaborn as sns   #matplotlib보다 그래프를 빠르게 그릴 수 있음

#################### 피험자 번호 ####################
subject_num = 20

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
                    filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                    count += 1
                    a.append(filename)
                    break
                else:
                    filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                    count += 1
                    a.append(filename)

    elif type =='F':

        for motion_num in range(1, 12):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)
                filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                count += 1
                a.append(filename)


#################### 폴더 경로 불러오기 ####################
folder_path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P20/ble/'

export_path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P20/ble_out/'  # 내보내기 경로

folder = os.listdir(folder_path)

folder_list = pd.DataFrame(folder, columns=['name'])
folder_list = folder_list[~folder_list['name'].str.contains('.zip')]
folder_list = folder_list.reset_index(drop=True)


#################### 파일 불러오기 및 전처리 ####################
for i in range(len(folder_list)):

    raw_path = folder_path + folder_list.name[i] + '/Raw data (g).csv'
    raw = pd.read_csv(raw_path)
    raw.columns = ['Time', 'X', 'Y', 'Z', 'Tag']

    # 가속도 데이터
    acc = raw[raw['Tag'] == 1]
    acc = acc.iloc[:, 1:4]
    acc.columns = ['Acc_X', 'Acc_Y', 'Acc_Z']
    acc = acc.reset_index(drop=True)
    # plt.figure()
    # plt.plot(acc)

    # 각속도 데이터
    gyr = raw[raw['Tag'] == 2]
    gyr = gyr.iloc[:, 1:4]
    gyr.columns = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
    gyr = gyr.reset_index(drop=True)
    # plt.figure()
    # plt.plot(gyr)

    BLE_data = pd.concat([acc, gyr], axis=1)


    BLE_data.to_csv(export_path + a[i])


    #################### 확인용 그래프 ####################
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(a[i].replace('.csv', ''))
    plt.plot(acc)
    plt.xlabel('Frames')
    plt.ylabel('Acceleration (g)')

    plt.subplot(2, 1, 2)
    plt.plot(gyr)
    plt.xlabel('Frames')
    plt.ylabel('Angular velocity (degree/s)')

    plt.tight_layout()

    plt.savefig('C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P20/ble_out/그래프/%s' % a[i].replace('.csv', '.png'))
    plt.close()






