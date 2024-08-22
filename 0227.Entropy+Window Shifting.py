import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/entropy_values_des/"


# 엔트로피 변화를 확인할 임계값 설정
threshold = 4.786

# 윈도우 크기와 슬라이드 간격 정의
window_length = 120
overlap_ratio = 0.5
overlap = int(window_length * overlap_ratio)

e, m, trial = 1, 2,1
count = 0

for e in range(1, 21):
    for m in range(1, 15):
        for trial in range(1, 4):
            try:
                t = trial
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/xsens/" % e
                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[1:, 5:11]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                # 데이터 정규화
                data['Acc_X'] = data.Acc_X / 9.8
                data['Acc_Y'] = data.Acc_Y / 9.8
                data['Acc_Z'] = data.Acc_Z / 9.8
                data['Gyr_X'] = data.Gyr_X / 9.8
                data['Gyr_Y'] = data.Gyr_Y / 9.8
                data['Gyr_Z'] = data.Gyr_Z / 9.8

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                ASVM = data[['ASVM']]
                ASVM_en = np.zeros((num_windows, window_length, 1))

                for i in range(num_windows):
                    start = int(i * (window_length * (1 - overlap_ratio)))
                    end = start + window_length
                    ASVM_en[i] = ASVM[start:end]

                ASVM_en = np.array(ASVM_en)
                entropy_values = np.zeros((num_windows, 1))

                for i in range(num_windows):
                    entropy_values[i] = entropy(ASVM_en[i])

                entropy_df = np.array(entropy_values)
                entropy_plt = np.zeros((num_windows * 10, 1))

                for i in range(len(entropy_plt)):
                    num = i // 10
                    entropy_plt[i] = entropy_df[int(num)]

                '----------------------------------------------------------------------------------'


                entropy_values_scale = entropy_values / max(entropy_values)
                plt.figure()
                plt.plot(entropy_values_scale)
                plt.show()

                entropy_values_des = np.zeros((len(entropy_values), 1))
                for i in range(len(entropy_values_scale) - 1):
                    entropy_values_des[i] = entropy_values_scale[i] - entropy_values_scale[i + 1]
                plt.figure()
                plt.plot(entropy_values_des)
                plt.show()

                entropy_values_des2 = entropy_values_des / max(entropy_values_des)
                plt.figure()
                plt.plot(entropy_values_des2)
                plt.axhline(0)
                plt.show()

                '-----------------------------------------------------------------------------------'

                plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                plt.plot(ASVM, label='ASVM')
                plt.legend()
                plt.title(name)

                plt.subplot(2, 1, 2)
                plt.plot(entropy_values_des2, label='entropy_values_des')
                plt.axhline(0)
                plt.legend()

                plt.tight_layout()
                plt.show()

                plt.savefig(path_graph + name_png)
                plt.close()

            except:
                count += 1


# entropy_values_scale = entropy_values / max(entropy_values)
#
# entropy_values_des = np.zeros((len(entropy_values), 1))
# for i in range(len(entropy_values_scale) - 1):
# entropy_values_des[i] = entropy_values_scale[i] - entropy_values_scale[i + 1]
# plt.figure()
# plt.plot(entropy_values_des)
# plt.show()
#
# entropy_values_des2 = entropy_values_des / max(entropy_values_des)
# plt.figure()
# plt.plot(entropy_values_des2)
# plt.axhline(0)
# plt.show()


# # 엔트로피 값의 변화 지점 표시
# threshold_crossings = np.where(np.diff((entropy_plt <= threshold).astype(int)) == 1)[0]
# # threshold_crossings를 기준으로 수직선 그리기
# for idx in threshold_crossings:
#     plt.axvline(x=idx * 10, color='r', linestyle='--', linewidth=0.5)
#
# threshold_crossings_end = np.where(np.diff((entropy_plt >= threshold).astype(int)) == 1)[0]
# # threshold_crossings_end를 기준으로 수직선 그리기
# for idx in threshold_crossings_end:
#     plt.axvline(x=idx * 10, color='g', linestyle='--', linewidth=0.5)