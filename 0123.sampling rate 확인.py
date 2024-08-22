import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = 'C:/Users/lsoor/OneDrive/바탕 화면/test'

folder = os.listdir(path)
filelist = pd.DataFrame(folder, columns=['name'])
filelist = natsort.natsorted(filelist.name)
filelist = pd.DataFrame(filelist, columns=['name'])
filelist = filelist[filelist.name != '.DS_Store']
filelist = filelist.reset_index(drop=True)

# 테스트용
a = []
# type = 'D'
motion_num = 1
# trial_num = 1
count = 0

#################### 이름 바꾸기 ####################
for motion_num in range(1, 20):
    motion_num = count + 1
    m = str(motion_num).zfill(2)
    filename = '%s.csv' % (m)

    while os.path.exists(os.path.join(path, filename)):
        motion_num += 1
        m = str(motion_num).zfill(2)
        filename = '%s.csv' % (m)

    os.rename(os.path.join(path, filelist.name[count]), os.path.join(path, filename))
    count += 1
    a.append(filename)



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

        plt.savefig('C:/Users/lsoor/OneDrive/바탕 화면/test/그래프/%s' % a[i].replace('.csv', '.png'))
        plt.close()

