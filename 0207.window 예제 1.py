from scipy.stats import mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

e, m, trial = 19, 3, 1   #필요한거..
t = trial

e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = 'D:/장지원/낙상/Ongoing/낙상 데이터 전처리_장지원/P19/ble_re/'

name = "P%sD%sR%s" % (e, m, t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=0)
data = pd.DataFrame(data)
data = data.iloc[:, 1:7]  #Acc_X ~ Gyr_z
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

# data_1 = data['Acc_X']
# data_2 = data['Acc_X'].shift(periods=2) # 단순 프레임 이동

# 윈도우 크기와 슬라이드 간격 정의
window_size = 100  # 예: 100개의 샘플로 구성된 윈도우
step_size = 10  # 예: 매 10개 샘플마다 윈도우 슬라이드

# 결과를 저장할 리스트 초기화
windowed_data = []

# 윈도우를 통한 샘플링 및 특징 추출
features = []
labels = []
i = 0

# 슬라이딩 윈도우 샘플링
for i in range(0, len(data) - window_size + 1, step_size):
    end = i + window_size
    window = data.iloc[i:end]  # 현재 윈도우에 대한 데이터 추출
    # 윈도우에 대한 통계 계산 (예: 평균, 표준편차)
    window_mean = window.mean()
    window_std = window.std()
    # 결과 저장
    windowed_data.append((window_mean, window_std))

# 결과를 데이터 프레임으로 변환
windowed_data_df = pd.DataFrame(windowed_data, columns=['Window_Mean', 'Window_Std'])
windowed_data_df['Mean'] = pd.to_numeric(windowed_data_df['Mean'], errors='coerce')
windowed_data_df['Std'] = pd.to_numeric(windowed_data_df['Std'], errors='coerce')
windowed_data_df.dropna(inplace=True)  # 결측값이 있는 행을 제거

windowed_data_df.tail()
plt.figure()
plt.plot(windowed_data_df, label='windowed_data_df')


for i in range(0, data.shape[0] - window_size + 1, step_size):
    window_data = data.iloc[i:i + window_size]

    # 특징 추출 (여기서는 평균과 표준편차를 사용)
    feature = [
        window_data['ASVM'].mean(),
        window_data['GSVM'].std(),
        # ... 추가 특징을 계산할 수 있습니다.
    ]
    features.append(feature)

data.iloc[0, 0:1]

'--------------------graph-----------------------'
# plt.close()
# mid_index=414   #원하는 프레임 설정
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)  #세로, 가로, 몇번째

plt.title(name)
plt.ylabel('ASVM(g)')
plt.xlabel('Frames')
plt.plot(data.ASVM, label='ASVM')
# plt.axvline(mid_index, label='Mid', color='red')
plt.legend(loc='upper right', fontsize=8)

plt.subplot(2, 1, 2)
# plt.title('Angular Velocity SVM')
plt.ylabel('GSVM (degree/s)')
plt.xlabel('Frames')
plt.plot(data.GSVM, label='GSVM')
# plt.axvline(mid_index, label='Mid', color='red')
plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()


