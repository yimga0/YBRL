
import pandas as pd
import numpy as np


#파일경로
path = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\삼성개발데이터\삼성 개발 데이터셋\Xsens_csv_rowdata\\'

#데이터 2D배열 설정
data_save = []
data_save = np.zeros((660,1200))

#count변수 초기화
count = 0
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

                data = pd.read_csv(path + file_name)

                #mas_frame 기준 계산
                max_frame = np.argmax((data.ASVM))
                #1초동안 샘플링 100번
                sampling_rate = 100
                data_trim = data.iloc[max_frame - sampling_rate:max_frame + sampling_rate, 1:7]


                #데이터 1D로 변환
                data_trim = np.array(data_trim)
                data_trim = data_trim.flatten()

                data_save[count] = data_trim

                count += 1

            except:
                count += 1


#data_test = data_save[5]
#data_test = data_test.reshape(200,6)

#
# data.ASVM  #엑셀asvm값만불러오기
# import matplotlib.pyplot as plt
# plt.plot(data.ASVM)   #그래프그리기
# plt.show()


# point_data = pd.read_csv(path + file_name)
#
# point_data.ASVM  #엑셀asvm값만불러오기
# import matplotlib.pyplot as plt
# plt.plot(point_data.ASVM)  #그래프그리기
# plt.show()

# import matplotlib.pyplot as plt
# plt.plot(data.iloc[:, 7:8])
# # data_svm[impact_point-100:impact_point+200]
# # plt.plot(data_svm[median_point - 150:median_point + 150])
# plt.title('')
# plt.xlabel('Frames')
# plt.ylabel('ASVM (g)')
