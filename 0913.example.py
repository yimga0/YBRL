#if문 사용해서 100프레임 안채워지는 부분 제로패딩 하는법

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pk
import matplotlib.pyplot as plt


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
'----------------------------------------------------01. 파라미터 선택---------------------------------------------------'
full_subject_list = np.arange(1, 22)
no_subject_list = np.array([11])
subject_list = np.setdiff1d(full_subject_list, no_subject_list)

# full_subject_list = np.arange(1, 21)
np.random.seed(7)
subject_list_test = np.random.choice(subject_list, 4, replace=False)
subject_list_training = np.setdiff1d(subject_list, subject_list_test)

file_path = 'D:/01. 개인연구/02. 데이터/2022_2 RIS/'
data_folder_path = '삼성 전처리/csv/'
variable_folder_path = '삼성 전처리/변수/'

'----------------------------------------------------02. Raw Data Load-------------------------------------------------'
D1 = {}
for subject in tqdm(subject_list):
    for motion in range(1, 26):
        for trial in range(1, 4):
            if subject < 10:
                subject_num = '0%d' % subject
            else:
                subject_num = '%d' % subject

            if motion <= 14:
                motion_type = 'D'
                real_motion = motion
            else:
                motion_type = 'F'
                real_motion = motion - 14

            if real_motion < 10:
                motion_num = '0%d' % real_motion
            else:
                motion_num = '%d' % real_motion

            if trial < 10:
                trial_num = '0%d' % trial
            else:
                trial_num = '%d' % trial

            try:
                file_name = 'S' + subject_num + motion_type + motion_num + 'R' + trial_num

                # 원본데이터 불러오기
                data = pd.read_csv(file_path + data_folder_path + file_name + '.csv', header=0)
            except FileNotFoundError:
                print('S' + subject_num + motion_type + motion_num + 'R' + trial_num + 'is not existed')
            else:
                data = data.iloc[:, 1:7]
                D1.setdefault(file_name, data)

with open('D1_Raw_Data_Samsung.p', 'wb') as file:
    pk.dump(D1, file)

'---------------------------------------------------03. Pre-processing-------------------------------------------------'

with open('D1_Raw_Data_Samsung.p', 'rb') as file:
    D1 = pk.load(file)

Dataset = []
Target1 = []
Target2 = []

for subject in tqdm(subject_list):
    for motion in range(1, 26):
        for trial in range(1, 4):
            # subject, motion, trial = 2, 9, 3
            subject, motion, trial = 6, 14+8, 2
            if subject < 10:
                subject_num = '0%d' % subject
            else:
                subject_num = '%d' % subject

            if motion <= 14:
                motion_type = 'D'
                real_motion = motion
            else:
                motion_type = 'F'
                real_motion = motion - 14

            if real_motion < 10:
                motion_num = '0%d' % real_motion
            else:
                motion_num = '%d' % real_motion

            if trial < 10:
                trial_num = '0%d' % trial
            else:
                trial_num = '%d' % trial

            try:
                file_name = 'S' + subject_num + motion_type + motion_num + 'R' + trial_num
                data = D1[file_name]
            except FileNotFoundError:
                print('S' + subject_num + motion_type + motion_num + 'R' + trial_num + 'is not existed')
            else:
                if motion_type == 'F':
                    data_svm = np.array(((data.Acc_X**2) + (data.Acc_Y**2) + (data.Acc_Z**2))**0.5)

                    # irregular 처리
                    if subject == 1 and motion == 14+3 and trial == 1:
                        impact_point = 231
                    elif subject == 2 and motion == 14+3 and trial == 2:
                        impact_point = 288
                    elif subject == 2 and motion == 14+11 and trial == 3:
                        impact_point = 678
                    else:
                        impact_point = np.argmax(data_svm)

                    new_data = data[impact_point-100:impact_point+200]
                    new_data.reset_index(drop=True, inplace=True)

                else:
                    median_point = int(round(len(data)/2, 0))

                    new_data = data[median_point - 150:median_point + 150]
                    new_data.reset_index(drop=True, inplace=True)

                    if len(new_data) != 300:
                        print(file_name)

                if len(Dataset) == 0:
                    Dataset = new_data
                    Target1 = np.array([subject, motion, trial])
                    Target2 = np.array([motion])
                else:
                    Dataset = np.dstack([Dataset, new_data])
                    Target1 = np.dstack([Target1, [subject, motion, trial]])
                    Target2 = np.dstack([Target2, [motion]])

revised_Dataset = Dataset.transpose(2, 0, 1)
revised_Target1 = Target1.transpose(2, 0, 1).reshape(1500, 3)
revised_Target2 = Target2.transpose(2, 0, 1).reshape(1500, )

np.save(file_path+variable_folder_path+'data_x', revised_Dataset)
np.save(file_path+variable_folder_path+'data_y1', revised_Target1)
np.save(file_path+variable_folder_path+'data_y2', revised_Target2)

plt.plot(new_data.iloc[:, 0:3])
data_svm[impact_point-100:impact_point+200]
plt.plot(data_svm[median_point - 150:median_point + 150])
plt.title('S02D09R03')
plt.xlabel('Frames')
plt.ylabel('ASVM (g)')

plt.plot(new_data.iloc[:, 0], label='Acc_X')
plt.plot(new_data.iloc[:, 1], label='Acc_Y')
plt.plot(new_data.iloc[:, 2], label='Acc_Z')
plt.plot(new_data.iloc[:, 3], label='Gyro_X')
plt.plot(new_data.iloc[:, 4], label='Gyro_Y')
plt.plot(new_data.iloc[:, 5], label='Gyro_Z')
plt.legend(loc='upper right')
plt.title('S02D09R03')
plt.xlabel('Frames')

'------------------------------------------------04. Training-Test Split-----------------------------------------------'

data_x = np.load(file_path+variable_folder_path+'data_x.npy')
data_y1 = np.load(file_path+variable_folder_path+'data_y1.npy')
data_y2 = np.load(file_path+variable_folder_path+'data_y2.npy')

train_x = data_x[np.isin(data_y1[:, 0], subject_list_training)]
train_y1 = data_y1[np.isin(data_y1[:, 0], subject_list_training)]
train_y2 = data_y2[np.isin(data_y1[:, 0], subject_list_training)]

test_x = data_x[np.isin(data_y1[:, 0], subject_list_test)]
test_y1 = data_y1[np.isin(data_y1[:, 0], subject_list_test)]
test_y2 = data_y2[np.isin(data_y1[:, 0], subject_list_test)]

np.save(file_path+variable_folder_path+'train_x', train_x)
np.save(file_path+variable_folder_path+'train_y1', train_y1)
np.save(file_path+variable_folder_path+'train_y2', train_y2)

np.save(file_path+variable_folder_path+'test_x', test_x)
np.save(file_path+variable_folder_path+'test_y1', test_y1)
np.save(file_path+variable_folder_path+'test_y2', test_y2)

'------------------------------------------------05. Model------------------------------------------------------------'
