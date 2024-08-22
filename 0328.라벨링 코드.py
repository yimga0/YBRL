import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num =11

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = 'C:/Users/abbey/Desktop/어플실험/S11/'

folder = os.listdir(path)
filelist = pd.DataFrame(folder, columns=['name'])
filelist = natsort.natsorted(filelist.name)
filelist = pd.DataFrame(filelist, columns=['name'])
filelist = filelist[filelist.name != '.DS_Store']
filelist = filelist.reset_index(drop=True)

# # 테스트용
a = []
# type = 'D'
# motion_num = 1
# trial_num = 1
count = 0

#################### 이름 바꾸기 ####################
for type in ['D', 'F']:

    if type == 'D':

        for motion_num in range(1, 15):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)

                filename = 'S%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path + filelist.name[count], path + filename)
                count += 1
                a.append(filename)

                # if motion_num == 1:
                #     filename = 'S%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                #     os.rename(path+filelist.name[count], path+filename)
                #     count += 1
                #     a.append(filename)
                #     break
                # else:
                #     filename = 'S%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                #     os.rename(path+filelist.name[count], path+filename)
                #     count += 1
                #     a.append(filename)

    elif type == 'F':

        for motion_num in range(1, 12):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)
                filename = 'S%s%s%s.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path+filelist.name[count], path+filename)
                count += 1
                a.append(filename)
#