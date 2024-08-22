import numpy as np
import matplotlib.pyplot as plt
from math import log2

def time_series_entropy(data):           #시계열 데이터 data를 받음
    n = len(data)                        #n은 데이터의 길이
    unique_data = np.unique(data)        #데이터에서 중복 제거, 고유한 값만 남기고 unique_data에 저장
    entropy = 0                          #초기 엔트로피 0으로 설정
    for val in unique_data:              #각 고유한 값 반복문
        p_val = np.sum(data == val) / n  #val이 나타난 횟수 / 데이터 길이 = 확률
        entropy -= p_val * log2(p_val)   #엔트로피 공식(이벤트 확률에 로그 취하고(이벤트 정도 양) x 확률 =  엔트로피 계산)
    return entropy                       #최종적으로 계산된 엔트로피 값을 반환

def sample_entropy(data, m, r):  #시계열데이터, 패턴길이, 비교 가능한 거리(비슷한 패턴을 찾을 때 허용되는 최대 거리)
    n = len(data)
    B = 0.0  #유사한 패턴개수
    x_m = [data[i:i + m] for i in range(n - m + 1)]  #i=0부터시작하므로 전체길이-패턴길이+1 = m개의 원소 , (n - m + 1)윈도우 이동시킬 때 생기는 가능한 윈도우 개수
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            if i != j and np.max(np.abs(np.array(x_m[i]) - np.array(x_m[j]))) <= r:  #두 시계열이 다르고 절대값의 최대가 r이하인 경우
                B += 1.0
    sample_entropy = -np.log(B / ((n - m + 1) * (n - m)))   #(B / ((n - m + 1) * (n - m)): 각 윈도우에서 유사한 패턴의 평균 개수, -ln:음의 자연로그에 넣어 최종 샘플 엔트로피 계산
    return sample_entropy   #샘플 엔트로피 높을 수록 무질서한 패턴이 많이 존재

# def plot_time_series_and_entropy(time_series_data, m, r):    # 시계열 데이터 시각화
#     plt.figure(figsize=(10, 4))
#     plt.subplot(2, 1, 1)
#     plt.plot(time_series_data)  #시계열 데이터를 선 그래프로 시각화
#     plt.title("Time Series Data")
#
#     # 시계열 엔트로피 및 샘플 엔트로피 계산
#     ts_entropy = time_series_entropy(time_series_data)
#     samp_entropy = sample_entropy(time_series_data, m, r)
#
#     # 엔트로피 값 시각화
#     plt.subplot(2, 1, 2)
#     plt.bar(["Time Series Entropy", "Sample Entropy"], [ts_entropy, samp_entropy], color=['blue', 'green'])
#     plt.ylabel("Entropy Value")
#     plt.title("Entropy Analysis")
#
#     plt.tight_layout()
#     plt.show()

# # 예시 데이터
# time_series_data = np.random.rand(100)   #0부터 1사이의 균일 분포에서 난수 matrix array 생성
# pattern_length = 4  #윈도우 크기 결정
# comparison_distance = 0.2  #패턴이 얼마나 유사한지 결정하는 임계값(시계열 쌍 간의 거리)
#
# # 그래프 시각화
# plot_time_series_and_entropy(time_series_data, pattern_length, comparison_distance)
#
#

("--------------------------------------------------------------------------------------------------------------------")

import pandas as pd
# 예시 데이터 (랜덤한 시계열)
# 예시 데이터 (랜덤한 시계열)
# time_series_data = np.random.rand(1000)
e,m,t = 7,10,1

# for e in range(1,21):

e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = "C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/xsens/" % e
name = "P%sD%sR%s" % (e,m,t)
name_csv = "%s%s.csv" % (path, name)

data = pd.read_csv(name_csv, header=1)
data = pd.DataFrame(data)
data = data.iloc[:, 5:11]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

data['Acc_X'] = data.Acc_X
data['Acc_Y'] = data.Acc_Y
data['Acc_Z'] = data.Acc_Z
data['Gyr_X'] = data.Gyr_X
data['Gyr_Y'] = data.Gyr_Y
data['Gyr_Z'] = data.Gyr_Z
data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


time_series_data = data.ASVM
asvm = pd.DataFrame(data.ASVM)
# time_series_data = np.random.rand(100)
pattern_length = 2        #윈도우 크기 결정
comparison_distance = 2  #패턴이 얼마나 유사한지 결정하는 임계값(시계열 쌍 간의 거리)

# 그래프 시각화
def plot_time_series_and_entropy(time_series_data, pattern_length, comparison_distance)

'--------------------------------------------------------------------------------------------------------------------'
# window shift

# def plot_time_series_and_entropy(time_series_data, m, r):    # 시계열 데이터 시각화
#     plt.figure(figsize=(10, 4))
#     plt.subplot(2, 1, 1)
#     plt.plot(time_series_data)  #시계열 데이터를 선 그래프로 시각화
#     plt.title("Time Series Data")
#
#     # 시계열 엔트로피 및 샘플 엔트로피 계산
#     ts_entropy = time_series_entropy(time_series_data)
#     samp_entropy = sample_entropy(time_series_data, m, r)
#
#     # 엔트로피 값 시각화
#     plt.subplot(2, 1, 2)
#     plt.bar(["Time Series Entropy", "Sample Entropy"], [ts_entropy, samp_entropy], color=['blue', 'green'])
#     plt.ylabel("Entropy Value")
#     plt.title("Entropy Analysis")
#
#     plt.tight_layout()
#     plt.show()
