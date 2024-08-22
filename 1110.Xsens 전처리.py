import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num =11

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P11/xsens/'


folder = os.listdir(path)
filelist = pd.DataFrame(folder, columns=['name'])
filelist = natsort.natsorted(filelist.name)
filelist = pd.DataFrame(filelist, columns=['name'])
filelist = filelist[filelist.name != '.DS_Store']
filelist = filelist.reset_index(drop=True)

# 테스트용
a = []
type = 'D'
motion_num = 1
trial_num = 1
count = 0

#################### 이름 바꾸기 ####################
for type in ['D', 'F']:

    if type == 'D':

        for motion_num in range(1, 15):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)

                if motion_num == 1:
                    filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                    os.rename(path+filelist.name[count], path+filename)
                    count += 1
                    a.append(filename)
                    break
                else:
                    filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                    os.rename(path+filelist.name[count], path+filename)
                    count += 1
                    a.append(filename)

    elif type == 'F':

        for motion_num in range(1, 12):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)
                filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path+filelist.name[count], path+filename)
                count += 1
                a.append(filename)
#

#################### 확인용 그래프 ####################
i = 0
for i in range(len(a)):

    data = pd.read_csv(path+a[i], header=1)
    acc = data.iloc[:, 5:8]
    gyr = data.iloc[:, 8:11]

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

    plt.savefig('C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P11/xsens/그래프/%s' % a[i].replace('.csv', '.png'))
    plt.close()
