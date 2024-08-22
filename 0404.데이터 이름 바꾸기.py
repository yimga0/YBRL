import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num =2

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = 'C:/Users/abbey/Desktop/20240404_190630/'

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
                filename = 'P%s%s%s_waist.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path + filelist.name[count], path + filename)
                count += 1
                a.append(filename)
                # if motion_num == 1:
                #     filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                #     os.rename(path+filelist.name[count], path+filename)
                #     count += 1
                #     a.append(filename)
                #     break
                # else:
                #     filename = 'P%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                #     os.rename(path+filelist.name[count], path+filename)
                #     count += 1
                #     a.append(filename)

    elif type == 'F':

        for motion_num in range(1, 12):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)
                filename = 'P%s%s%s_waist.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path+filelist.name[count], path+filename)
                count += 1
                a.append(filename)



path_graph = 'C:/Users/abbey/Desktop/20240404_190630/그래프(waist)/'

count=0
e,m,t=2,2,1
for m in range(1, 15):
    for t in range(1,4):
        try:

            e = str(e).zfill(2)
            m = str(m).zfill(2)
            t = str(t).zfill(2)

            path = 'C:/Users/abbey/Desktop/20240404_190630/'

            name = "P%sD%sR%s_waist" % (e, m, t)
            name_csv = "%s%s.csv" % (path, name)
            name_png = "%s.png" % (name)


            data = pd.read_csv(str(name_csv), header=1)
            data = pd.DataFrame(data)
            data = data.iloc[1:, 5:11]  #Acc_X ~ Gyr_z
            data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']


            data['Acc_X'] = data.Acc_X / 9.8
            data['Acc_Y'] = data.Acc_Y / 9.8
            data['Acc_Z'] = data.Acc_Z / 9.8

            data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
            data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

            plt.figure(figsize=(10,6))

            plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
            plt.title(name)
            # plt.title('Acceleration SVM')
            plt.ylabel('ASVM(g)')
            plt.xlabel('Frames')
            plt.plot(data.ASVM, label='ASVM')
            plt.legend(loc='upper right', fontsize=8)

            plt.subplot(2, 1, 2)
            # plt.title('Angular Velocity SVM')
            plt.ylabel('GSVM (degree/s)')
            plt.xlabel('Frames')
            plt.plot(data.GSVM, label='GSVM')
            plt.legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.show()

            plt.savefig(path_graph + name_png)
            plt.close()

        except:
            pass

