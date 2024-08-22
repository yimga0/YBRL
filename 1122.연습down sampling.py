import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P02/ble/Test 2023-11-13 18-31-15/Raw data (g).csv'
# path = 'C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P02/ble/Test 2023-11-13 18-31-15/Raw data (g).csv'
# flie_name = 'Test 2023-11-09 19-11-18'

raw = pd.read_csv(path)
raw.columns = ['Time', 'X', 'Y', 'Z', 'Tag']
len(raw)
sample_rate = ((raw.index[-1] - raw.index[0]) / 2) / (raw.Time[len(raw)-1] - raw.Time[0])
total_time = raw.Time[len(raw)-1]- raw.Time[0]
resample = 30

df = signal.resample(raw, int(resample*total_time)*2)




# df = pd.DataFrame(df)
# df_resampled = raw.resample('30hz', on='time')
#
# print(df_resampled)


#plt.plot( raw.X  )
plt.plot(df)
plt.show()
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

#  df_resampled = df.resample('30', on='index')
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 원본 데이터 생성
# original_data = np.sin(2 * np.pi * 0.1 * np.arange(0, 1000, 1)) + 0.2 * np.random.randn(1000)
#
# # 다운샘플링할 비율 설정
# downsample_factor = 30
#
# # 다운샘플링 수행 (평균 활용)
# downsampled_data = np.mean(original_data.reshape(-1, downsample_factor), axis=1)
#
# # 그래프로 시각화
# plt.figure(figsize=(10, 5))
#
# plt.subplot(2, 1, 1)
# plt.plot(original_data, label='원본 데이터')
# plt.title('원본 데이터')
#
# plt.subplot(2, 1, 2)
# plt.plot(downsampled_data, label=f'다운샘플링 (비율: {downsample_factor})', color='orange')
# plt.title('다운샘플링된 데이터')
#
# plt.tight_layout()
# plt.show()
#