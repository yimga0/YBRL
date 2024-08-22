import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
from tqdm import tqdm
from ahrs.common.orientation import acc2q
from ahrs.filters import Complementary
import matplotlib.pyplot as plt
import natsort
import seaborn

path = '1.2 xsens데이터csv/csv/'

folder = os.listdir(path)

filelist = pd.DataFrame(folder, columns=['name'])
filelist = natsort.natsorted(filelist.name)
filelist = pd.DataFrame(filelist, columns=['name'])
filelist = filelist[filelist.name != '.DS_Store']
filelist = filelist.reset_index()

# 필요한 그래프만 뽑기: num을 지정(인덱스번호), for문만 뺴고 아래 돌리기(num에 인덱스 번호가 있으니까)

# num = 591
Impact_list = []
File_list = []

impact_frame_df = pd.read_csv('C:/Users/gaga2/Desktop/학부연구/삼성개발데이터/삼성 개발 데이터셋/csv.csv')

for num in range(len(filelist)):
    file = filelist.name[num]
    data = pd.read_csv(path + filelist.name[num], header=3)
    data = data.iloc[:, 1:]
    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM', 'Roll', 'Pitch']
ㅁ
    if file[3:4] == 'F':

        for i in range(len(impact_frame_df)):
            if impact_frame_df.name[i] == file.replace('.csv', ''):
                impact_frame = impact_frame_df.impact[i]
                break

        ASVM = np.sqrt(data.Acc_X**2 + data.Acc_Y**2 + data.Acc_Z**2)
        # impact_frame = np.argmax(ASVM)
        # Impact_list.append(impact_frame)
        # File_list.append(file.replace('.csv', ''))


        plt.figure(figsize=(8, 4)) #그래프 크기지정
        plt.title(file.replace('.csv', '') + ' ASVM')  #file이라는 곳에 .csv를 안보이게 하려고(지저분하니까)
        plt.plot(ASVM, lw=1)
        plt.axvline(impact_frame, color='red', lw=0.5)
        plt.axvline(impact_frame - 6 - 60, color='red', lw=1)  #impact_frame의 -6부터-60까지 확인하기 위해 구간 축 지정1
        plt.axvline(impact_frame - 6, color='red', lw=1)       # impact_frame의 -6부터-60까지 확인하기 위해 구간 축 지정2
        plt.xlim(impact_frame - 100, impact_frame + 20)  #무시해도됨
        plt.ylabel('Acceleration SVM (m/s^2)')
        plt.xlabel('Frames')
        plt.tight_layout()
        plt.show()
        plt.savefig('1.2 xsens데이터csv/그래프3/%s' % file.replace('.csv', '.png'))  # 이 파일에 %s부분에 csv를 png로 바꿈
        plt.savefig(r'C:\Users\gaga2\Desktop\학부연구\삼성개발데이터\graph\%s' % file.replace('.csv', '.png'))
        plt.close()


a = 'hello'
a = a.replace('o', 'w')