import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num = 20

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path = 'C:/Users/abbey/Desktop/어플실험(기존)/S20/Xsens_pocket/'



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
        pass

        for motion_num in range(1,23):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)

                filename = 'S%s%s%s_pocket.csv' % (str(subject_num).zfill(2), m, r)
                # os.rename(path + filelist.name[count], path + filename)
                count += 1
                a.append(filename)

    elif type == 'F':

            for motion_num in range(1, 18):
                m = type + str(motion_num).zfill(2)

                for trial_num in range(1, 4):
                    r = 'R' + str(trial_num).zfill(2)
                    filename = 'S%s%s%s_pocket.csv' % (str(subject_num).zfill(2), m, r)
                    # os.rename(path+filelist.name[count], path+filename)
                    count += 1
                    a.append(filename)


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

    plt.savefig( 'C:/Users/abbey/Desktop/어플실험(기존)/S20/Xsens_pocket/그래프(pocket)/%s' % a[i].replace('.csv', '.png'))
    plt.close()






'----------------------------------------라벨링------------------------------------------------------------'


import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
subject_num = 13

#################### 폴더 경로 지정 및 파일 불러오기 ####################
path =  r"C:/Users/abbey/Desktop/새 폴더 (3)/"

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
subject_start=13

subject_end=18

#################### 이름 바꾸기 ####################
for subject_num in range(subject_start, subject_end + 1):

    if type == 'D':

        for motion_num in range(14,15):
            m = type + str(motion_num).zfill(2)

            for trial_num in range(1, 4):
                r = 'R' + str(trial_num).zfill(2)

                filename = 'S%s%s%s_SW_v2.csv' % (str(subject_num).zfill(2), m, r)
                os.rename(path + filelist.name[count], path + filename)
                count += 1
                a.append(filename)



'----------------------------------------v2 떼기---------------------------------------------------------'

import os

# 폴더 경로 지정
path = 'C:/Users/abbey/Desktop/20명 데이터셋//'

# 폴더 내 파일 목록 가져오기
folder = os.listdir(path)

# 파일 이름에서 '_v2' 제거 및 이름 변경
for filename in folder:
    if '_v2' in filename:
        new_filename = filename.replace('_v2', '')
        old_filepath = os.path.join(path, filename)
        new_filepath = os.path.join(path, new_filename)

        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')




'----------------------------------------v2 떼기---------------------------------------------------------'

import os

# 경로 설정
# path = 'C:/Users/abbey/Desktop/SW실험/SW변환(S01~S12)/변환/'

try:
    # 디렉토리 내 모든 파일명 가져오기
    file_names = os.listdir(path)

    # 파일명 변경
    for file_name in file_names:
        # 파일명에서 "Raw"라는 세 글자를 제거합니다.
        new_file_name = file_name.replace('Raw', '')

        # 파일명 변경
        os.rename(os.path.join(path, file_name), os.path.join(path, new_file_name))
        print(f"{file_name} -> {new_file_name}")

    print("파일명 변경 완료!")

except Exception as e:
    print(f"파일명 변경 중 오류 발생: {e}")


'----------------------------------------T_ 떼기---------------------------------------------------------'


import os

# 폴더 경로 지정
# path =r"C:/Users/abbey/Desktop/SW실험/SW변환(S01~S12)/변환/"

# 폴더 내 파일 목록 가져오기
folder = os.listdir(path)

# 파일 이름에서 'T_' 제거 및 이름 변경
for filename in folder:
    if filename.startswith('T_'):
        new_filename = filename[2:]  # 'T_' 제거
        old_filepath = os.path.join(path, filename)
        new_filepath = os.path.join(path, new_filename)

        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')
