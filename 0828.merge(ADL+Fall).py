#콘솔창 유지
#import os
#os.system("pause")
'----------------------------------------------------01. ADL FALL 합치기------------------------------------------------'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#파일경로
path = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\삼성개발데이터\삼성 개발 데이터셋\Xsens_csv_rowdata\\'
path_graph = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\삼성개발데이터\gragh\graph_Xsens_rowdata\\'

#데이터 2D배열 설정
data_saved = []
data_saved = np.zeros((840,1200))

#count변수 초기화
count = 0
s, m, t = 1, 1, 1

data_name_Fall = []
data_name_ADL = []

s, m, t = 1, 1, 1
for s in range(1,21):
    for m in range(1,15):
        for t in range(1,4):
            try:
                S = str(s).zfill(2)
                M = str(m).zfill(2)
                T = str(t).zfill(2)
                file_name = 'S%sD%sR%s.csv' % (S, M, T)
                data = pd.read_csv(path + file_name)

                mid = int((len(data)-1)/2)
                sampling_rate = 100
                data_trim = data.iloc[mid-sampling_rate : mid+sampling_rate , 1:7]

                #str(data_saved).rjust(200, '0')


                # 데이터 1D로 변환
                data_trim = np.array(data_trim)
                data_trim = data_trim.flatten()

                data_saved[count] = data_trim

                count += 1

                data_name_ADL.append(file_name)  #리스트 추가


            except:
                data_name_ADL.append(file_name)  #리스트 추가
                count += 1


#데이터 2D배열 설정
data_savef = []
data_savef = np.zeros((660,1200))

#count변수 초기화
count1 = 0
s, m, t = 1, 1, 1

#subject, movement, trial 각 작업 수행
for s in range(1,21):   #subject
    for m in range(1,12):   #movement
        for t in range(1,4):   #trial
            try:
                #두자리중에 왼쪽부터 0 채움(포맷팅)
                S = str(s).zfill(2)
                M = str(m).zfill(2)
                T = str(t).zfill(2)
                file_name = 'S%sF%sR%s.csv' % (S, M, T)
                file_png = 'S%sF%sR%s.png' % (S, M, T)
                data = pd.read_csv(path + file_name)

                #max_frame 기준 계산
                max_frame = np.argmax((data.ASVM))
                #1초동안 샘플링 100번
                sampling_rate = 100

                data_trim = data.iloc[max_frame - sampling_rate:max_frame + sampling_rate, 1:7]

                #데이터 1D로 변환
                data_trim = np.array(data_trim)
                data_trim = data_trim.flatten()

                data_savef[count1] = data_trim

                data_name_Fall.append(file_name)

                '--------------------graph-----------------------'

                # plt.figure(figsize=(8, 4))  # 그래프 크기지정
                # plt.title(file_name)
                # plt.plot(data['ASVM'], lw=1)  # ASVM 축 두께 지정
                # plt.axvline(max_frame, color='r', lw=0.5)
                # plt.ylabel(r'Acceleration SVM $(m/s^2)$')
                # plt.xlabel('Frames')
                # plt.tight_layout()
                # plt.savefig(path_graph + file_png)
                # plt.close()

                count1 += 1

            except:
                data_name_Fall.append(file_name)
                count1 += 1



data.ASVM  #엑셀asvm값만불러오기
import matplotlib.pyplot as plt
plt.plot(data.ASVM)   #그래프그리기
plt.show()


#np.vstack: 두 배열 위에서 아래로 붙이기
#data_all = np.vstack((data_saved, data_savef))
data_saved = pd.DataFrame(data_saved)
data_savef = pd.DataFrame(data_savef)


data_name_ADL = pd.DataFrame(data_name_ADL)  #data_name_ADL 데이터프레임 생성
data_name_Fall = pd.DataFrame(data_name_Fall)   #data_name_Fall 데이터프레임 생성


#objs: 다중 데이터 프레임 한번에 합칠때 용이
data_ADL = pd.concat([data_saved, data_name_ADL], axis=1)   #ADL데이터 + ADL 파일이름 열방향으로 프레임합치기
data_Fall = pd.concat([data_savef, data_name_Fall], axis=1)  #Fall데이터 + Fall 파일이름 열방향으로 프레임합치기

data_all = pd.concat([data_ADL, data_Fall], axis=0)  #Fall데이터 + ADL데이터 행방향으로 프레임합치기


#그래프 확인
#보내주신 데이터 그래프 그리기
#누락파일 확인

