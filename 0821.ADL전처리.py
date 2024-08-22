
import pandas as pd
import numpy as np


path = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\삼성개발데이터\삼성 개발 데이터셋\Xsens_csv_rowdata\\'

data_save = []
data_save = np.zeros((840,1200))
count = 0
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
                data_trim = data.iloc[mid - sampling_rate:mid + sampling_rate, 1:7]

                data_trim = np.array(data_trim)
                data_trim = data_trim.flatten()

                data_save[count] = data_trim

                count += 1

            except:
                count += 1


#data_save['file_name'] = 0

#data_test = data_save[5]
#data_test = data_test.reshape(200,6)




