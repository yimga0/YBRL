import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

'------------------------------------------------------------------------------------------------------------------'
#그래프 저장 주소
path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/entropy_ADL(BLE)/그래프_GSVM/"

# folder_path = r"C:/Users/lsoor/OneDrive/바탕 화면/entropy_ADL/"
# path = glob.glob(folder_path + "*.csv")

# path = glob.glob("C:/Users/lsoor/OneDrive/바탕 화면/entropy_ADL/*.csv")
# csv_file = "C:/Users/lsoor/OneDrive/바탕 화면/entropy_ADL/P01D02R01_b.csv"
count=0
e, m, t = 1,2,2
# count=0
for e in range(1,21):
    for m in range(1, 21):
        if m in [2, 3, 4, 10, 11, 12, 13, 14]:
            for t in range(1,4) :
                try:

                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/lsoor/OneDrive/바탕 화면/entropy_ADL(BLE)/"

                    name = "P%sD%sR%s_a" % (e, m, t)
                    name_csv = "%s%s.csv" % (path,name)
                    name_png = "%s.png" % (name)


                    data = pd.read_csv(name_csv)
                    data = data.iloc[:, 1:]

                    # 시계열 데이터 추출
                    ASVM =data[['ASVM']]
                    GSVM =data[['GSVM']]

                    # def detect_signal_change(signal, window_size, threshold):
                    # 엔트로피 변화 감지에 사용할 슬라이딩 윈도우 크기
                    # window_size = 10

                    # 엔트로피 변화를 감지하기 위한 임계값
                    threshold = 2

                    window_length = 10
                    overlap = 0

                    num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap))))

                    # create an empty array to store the segmented data
                    ASVM_en = np.zeros((num_windows, window_length, 1))

                    # segment the data using the sliding window technique
                    for i in range(num_windows):
                        start = int(i * (window_length * (1 - overlap)))
                        end = start + window_length
                        ASVM_en[i] = ASVM[start:end]

                    ASVM_en = np.array(ASVM_en)
                    # print(ASVM_en.shape)
                    # plt.plot(ASVM_en.flatten())

                    entropy_values = np.zeros((num_windows, 1))
                    for i in range(num_windows):
                        entropy_values[i] = entropy(ASVM_en[i])

                    entropy_df = np.array(entropy_values)

                    entropy_plt = np.zeros((num_windows * window_length, 1))
                    # i=40
                    for i in range(len(entropy_plt)):
                        num = i // 10
                        entropy_plt[i] = entropy_df[int(num)]

                    plt.figure(figsize=(10, 8))

                    plt.subplot(2, 1, 1)
                    plt.plot(ASVM_en.flatten(), label='ASVM')
                    plt.legend()
                    plt.title(name)

                    plt.subplot(2, 1, 2)
                    plt.plot(entropy_plt, label='Entropy_ASVM')
                    plt.legend()

                    plt.tight_layout()
                    plt.show

                    plt.savefig(path_graph + name_png)
                    plt.close()

                except:
                    count += 1
