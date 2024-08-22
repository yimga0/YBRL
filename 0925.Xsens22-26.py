import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#파일경로
path = r'C:\Users\gaga2\Desktop\학부연구\삼성개발데이터\삼성 개발 데이터셋\Xsens_22-26\\'
path_graph = r'C:\Users\gaga2\Desktop\학부연구\삼성개발데이터\gragh\graph_22-26\\'

#데이터 2D배열 설정
data_saved = []
data_saved = np.zeros((840,1200))

#count변수 초기화
count = 0
s, m, t = 1, 1, 1

data_name_Fall = []
data_name_ADL = []

s, m, t = 1, 1, 1
for s in range(25,27):
    for m in range(1,16):
        for t in range(1,4):
            try:
                S = str(s).zfill(2)
                M = str(m).zfill(2)
                T = str(t).zfill(2)
                file_name = 'S%sD%sR%s.csv' % (S, M, T)
                file_png = 'S%sD%sR%s.png' % (S, M, T)
                data = pd.read_csv(path + file_name)

                mid = int((len(data)-1)/2)
                sampling_rate = 100
                data_trim = data.iloc[mid-sampling_rate : mid+sampling_rate , 1:7]

                #str(data_saved).rjust(200, '0')


                # 데이터 1D로 변환
                data_trim = np.array(data_trim)
                data_trim = data_trim.flatten()

                data_saved[count] = data_trim


                '--------------------graph-----------------------'

                plt.figure(figsize=(8, 4))  # 그래프 크기지정
                plt.title(file_name)
                plt.plot(data['ASVM'])  # ASVM 축 두께 지정
                #plt.axvline(max_frame, color='r', lw=0.5)
                plt.ylabel(r'Acceleration SVM $(m/s^2)$')
                plt.xlabel('Frames')
                plt.tight_layout()
                plt.savefig(path_graph + file_png)
                plt.close()

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

                plt.figure(figsize=(8, 4))  # 그래프 크기지정
                plt.title(file_name)
                plt.plot(data['ASVM'], lw=1)  # ASVM 축 두께 지정
                plt.axvline(max_frame, color='r', lw=0.5)
                plt.ylabel(r'Acceleration SVM $(m/s^2)$')
                plt.xlabel('Frames')
                plt.tight_layout()
                plt.savefig(path_graph + file_png)
                plt.close()

                count1 += 1

            except:
                data_name_Fall.append(file_name)
                count1 += 1

#파일명 설정X 23,24
#ASVM 정리X 26제외
#26번 피험자 동작수 1개적음
#그래프 그리기