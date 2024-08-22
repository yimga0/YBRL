import pandas as pd  # 데이터 프레임
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

path = r"C:/Users/abbey/Desktop/xsens 데이터 조정/P%sD%sR%s"

# 윈도우 크기와 슬라이드 간격 정의
window_length = 120
overlap_ratio = 0.5
overlap = int(window_length * overlap_ratio)

e, m, t = 1,1,1

for e in range(1,3):
    for m in range(1, 15):
            for t in range(1,4):
                try:
                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/abbey/Desktop/xsens 데이터 조정/"

                    name = "P%sD%sR%s" % (e, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)

                    data = pd.read_csv(str(name_csv))
                    data = pd.DataFrame(data)
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

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

                    entropy_values_scale = entropy_values / max(entropy_values)

                    entropy_values_des = np.zeros((len(entropy_values), 1))
                    for i in range(len(entropy_values_scale) - 1):
                        entropy_values_des[i] = entropy_values_scale[i] - entropy_values_scale[i + 1]

                    entropy_values_des2 = entropy_values_des / max(entropy_values_des)

                    '---------------------------graph---------------------------------------'

                    path_graph = r"C:/Users/abbey/Desktop/xsens 데이터 조정/그래프/"

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

