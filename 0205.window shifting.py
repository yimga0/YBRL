import numpy as np
import pandas as pd

e,m,t= 1,2,1


e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/"

name = "P%sD%sR%s_a" % (e, m, t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=1)
data = pd.DataFrame(data)
data = data.iloc[:, 1:7]  #Acc_X ~ Gyr_z
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

# 윈도우 크기와 슬라이드 간격 정의
window_size = 100                             # 예: 100개의 샘플로 구성된 윈도우
overlap_ratio =0.3
overlap = int(window_size * overlap_ratio )   # 예: 매 10개 샘플마다 윈도우 슬라이드

# 결과를 저장할 리스트 초기화
windowed_data = []

# # 윈도우를 통한 샘플링 및 특징 추출
features = []
labels = []
i = 0


# calculate the number of windows
num_windows = int(np.ceil((len(data) - window_size) / (window_size - overlap))) + 1  #window 총 개수/window가 넘어가지 않도록 +1을 함

# create an empty array to store the segmented data
# segments = np.zeros((num_windows, window_size, 6))  #윈도우 수에 해당하는 크기


# for i in range(num_windows):
#     start = i * (window_size - overlap)
#     end = start + window_size
#     # segments[i] = data[start:end]
#     if end > len(data):  # 마지막 윈도우의 경우 데이터가 부족할 수 있으므로 윈도우 크기를 조절
#         end = len(data)
#         start = end - window_size
#     window = data.iloc[i:i+window_size]
#     windowed_data.append(window)


for i in range(0, len(data) - window_size + 1, window_size - overlap):
    window = data.iloc[i:i + window_size]
    if len(window) < window_size:  # 윈도우 크기가 부족한 경우 건너뜀
        continue
    windowed_data.append(window.values)

windowed_data = np.concatenate(windowed_data)  # 리스트를 NumPy 배열로 변환
print(windowed_data.shape)  # 윈도우 데이터 배열의 형태 출력

window_index = 2
selected_window = pd.DataFrame(windowed_data[window_index])
print(selected_window)


import matplotlib.pyplot as plt
# windowed_data를 순회하면서 각 윈도우 데이터를 시각화하고 레이블을 추가


'--------------------------------------------------graph-------------------------------------------------------------'
# # windowed_data의 첫 번째 윈도우 데이터를 시각화하는 예제
# plt.plot(windowed_data[0])
# plt.title('First Window Data')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
#
# plt.legend()
# plt.show()

# calculate the number of windows





# num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap))))
#
# # create an empty array to store the segmented data
# segments = np.zeros((num_windows, window_length, 6))
#
# # segment the data using the sliding window technique
# for i in range(num_windows):
#     start = int(i * (window_length * (1 - overlap)))
#     end = start + window_length
#     segments[i] = data[start:end]
# segments = np.array(segments)
# print(segments.shape)
#
# x_train01 = rearrange(segments, 'a b c -> (a b) c')