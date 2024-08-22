import numpy as np
import pandas as pd
# 예시 데이터 (랜덤한 시계열)
# time_series_data = np.random.rand(1000)
e = 1
m = 1
t = 1
e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)
path = "C:/Users/lsoor/OneDrive/바탕 화면/임가영/삼성개발데이터/삼성 개발 데이터셋/1.2 xsens데이터csv/csv/"
name = "S%sD%sR%s" % (e,m,t)
name_csv = "%s%s.csv" % (path, name)

data = pd.read_csv(str(name_csv))
data = pd.DataFrame(data)
data = data.iloc[:, 1:7]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

data['Acc_X'] = data.Acc_X / 9.8
data['Acc_Y'] = data.Acc_Y / 9.8
data['Acc_Z'] = data.Acc_Z / 9.8
data['Gyr_X'] = data.Gyr_X / 9.8
data['Gyr_Y'] = data.Gyr_Y / 9.8
data['Gyr_Z'] = data.Gyr_Z / 9.8
data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

data = data.iloc[1:, :]

def shannon_entropy(data):
    """
    주어진 데이터에 대한 Shannon 엔트로피를 계산합니다.

    Parameters:
    - data: 리스트 또는 배열 형태의 데이터

    Returns:
    - entropy: Shannon 엔트로피 값
    """
    unique_elements, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 예시 데이터
sample_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

# Shannon 엔트로피 계산
entropy_value = shannon_entropy(data.ASVM)

print(f"Shannon 엔트로피: {entropy_value}")


import numpy as np

def window_shifting(data, window_size, shift):
    """
    시계열 데이터에 대해 window shifting을 수행하는 함수

    Parameters:
    - data: 시계열 데이터
    - window_size: 윈도우의 크기
    - shift: 윈도우를 이동시키는 간격

    Returns:
    - shifted_windows: 각 윈도우에 대한 결과를 저장한 리스트
    """
    n = len(data)
    shifted_windows = []

    for i in range(0, n - window_size + 1, shift):
        window = data[i:i + window_size]
        # 여기서 윈도우에 대한 작업을 수행
        window_mean = np.mean(window)
        shifted_windows.append(window_mean)

    return shifted_windows

# 예시 데이터
time_series_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 윈도우 크기와 이동 간격 설정
window_size = 100
shift = 10

# window shifting 수행
result = window_shifting(data.ASVM, window_size, shift)

# 결과 출력
print(result)
