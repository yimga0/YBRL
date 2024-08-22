
import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt
from tqdm import tqdm  # 시간
from scipy import signal
import seaborn as sns

# 파일이 위치한 경로 설정
path = r'D:\장지원\낙상\csv\csv\\'

# 데이터를 저장할 2D 배열 생성
data_saved = np.zeros((660, 1200), dtype=np.float64)  # 데이터 형식 추가

# 데이터 개수를 세는 변수 초기화
count = 0
count_miss = 0

window_length =
overlap =

# 세 개의 반복문을 사용하여 모든 가능한 파일에 대해 작업 수행
for s in range(1, 21):  # subject
    for m in range(1, 12):  # movement
        for t in range(1, 4):  # trial

                # 두자리중에 왼쪽부터 0 채움(포맷팅)
                S = str(s).zfill(2)
                M = str(m).zfill(2)
                T = str(t).zfill(2)
                file_name = 'S%sF%sR%s.csv' % (S, M, T)
                data = pd.read_csv(path + file_name)

                # calculate the number of windows
                num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap))))

                # create an empty array to store the segmented data
                segments = np.zeros((num_windows, window_length, 6))

                # segment the data using the sliding window technique
                for i in range(num_windows):
                    start = int(i * (window_length * (1 - overlap)))
                    end = start + window_length
                    segments[i] = data[start:end]
                segments = np.array(segments)
                print(segments.shape)

                x_train01 = rearrange(segments, 'a b c -> (a b) c')