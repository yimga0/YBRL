import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num =5

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = "rC:/Users/abbey/Desktop/어플실험(기존)/S05/sw/"


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

        for motion_num in range(1,15):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)

                filename = 'S%s%s%s_SW.csv' % (str(subject_num).zfill(2), m, r)
                # os.rename(path + filelist.name[count], path + filename)
                count += 1
                a.append(filename)

    elif type == 'F':

        for motion_num in range(1, 12):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)
                filename = 'S%s%s%s_SW.csv' % (str(subject_num).zfill(2), m, r)
                # os.rename(path+filelist.name[count], path+filename)
                count += 1
                a.append(filename)



for i in range(len(a)):
    data = pd.read_csv(path+a[i], header=1)
    acc = data.iloc[:, 2:5]
    gyr = data.iloc[:, 5:8]

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
    plt.show()

    # plt.savefig(r"C:/Users/abbey/Desktop/S06 SW/변환/그래프/%s" % a[i].replace('.csv', '.png'))
    # plt.close()


