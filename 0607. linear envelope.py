import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

'----------------------------------------------------------------------------------------------------------------------'

e, m, t = 2,4,2
for e in range(1, 23):
    for m in range(2, 15):
        for t in range(1, 4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/SW데이터(임계값)/"
                name = "S%sD01R01_SW" % e
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[1:, 1:7]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                mean = np.mean(data.ASVM)
                variance = np.var(data.ASVM)
                std = variance ** (1 / 2)
                threshold = mean + std * 6


                plot = "S%sD%sR%s_SW" % (e, m, t)
                plot_csv = "%s%s.csv" % (path, plot)
                plot_png = "%s.png" % (plot)

                plot_data = pd.read_csv(str(plot_csv))
                plot_data = pd.DataFrame(plot_data)
                # plot_data = plot_data.dropna(axis=0)
                plot_data = plot_data.iloc[:, 1:9]
                plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                ASVM_filter = plot_data.ASVM

                '---------------------------------------그래프----------------------------------------------'
                path_graph = r"C:/Users/abbey/Desktop/SW데이터(임계값)/그래프/"

                # 고역 필터 (50 Hz)
                fs = 50 # 샘플링 주파수 (필요시 데이터에 맞게 조정)
                hp_cutoff = 50
                b, a = ASVM_filter.butter(4, hp_cutoff / (fs / 2), btype='high')
                filtered_signal = ASVM_filter.filtfilt(b, a, plot_data['ASVM'])

                # 노치 필터 (59.5-60.5 Hz)
                notch_freq = 60.0  # 노치 필터 중심 주파수
                bandwidth = 1.0  # 대역폭
                b, a = a.iirnotch(notch_freq / (fs / 2), Q=notch_freq / bandwidth)
                filtered_signal = ASVM_filter.filtfilt(b, a, filtered_signal)

                # 전체 파형 정류
                rectified_signal = np.abs(filtered_signal)

                # 저역 필터 (8 Hz)
                lp_cutoff = 8
                b, a = a.butter(4, lp_cutoff / (fs / 2), btype='low')
                envelope_signal = ASVM_filter.filtfilt(b, a, rectified_signal)


                plt.figure(figsize=(10, 6))
                plt.title(plot)
                plt.plot(plot_data['ASVM'], label='Raw ASVM', color='blue')  # 원시 ASVM 신호, 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)
                plt.plot(envelope_signal, label='Linear Envelope', color='black')  # 선형 포락선 신호

                plt.axhline(threshold, color='red', linewidth=1)
                plt.ylabel('ASVM (g)')
                plt.xlabel('Frames')

                # 교차점 찾기
                crossing_points = np.where(np.diff(np.sign(plot_data.ASVM - threshold)))[0]
                crossing_frames = crossing_points

                # 교차점 표시
                for cp in crossing_points:
                    plt.plot(cp, plot_data.ASVM[cp], 'ro', markersize=1)

                frames_df = pd.DataFrame({'frame': crossing_frames})

                first_frame = frames_df['frame'].iloc[0]
                last_frame = frames_df['frame'].iloc[-1]

                result_frame = pd.DataFrame({'name': [plot], 'first': [first_frame], 'last': [last_frame]})

                plt.axvline(first_frame, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(last_frame, color='green', linestyle='-')  # 마지막 교차점

                plt.tight_layout()
                plt.show()
                plt.savefig(path_graph + plot_png)
                plt.close()

            except:
                pass
