import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 로드
path = pd.read_csv("C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/P01D02R01_a.csv")
data = path.iloc[:, 1:]

# 시계열 데이터 추출
ASVM=data[['ASVM']]
GSVM=data[['GSVM']]


# 엔트로피 변화 감지에 사용할 슬라이딩 윈도우 크기
window_size = 10

# 엔트로피 변화를 감지하기 위한 임계값
threshold = 0.5

def detect_signal_change(signal, window_size, threshold):
    """
    시계열 신호에서 엔트로피 변화를 감지하여 해당 부분의 데이터를 추출하는 함수

    :param signal: 입력 시계열 신호
    :param window_size: 슬라이딩 윈도우 크기
    :param threshold: 엔트로피 변화를 감지하기 위한 임계값
    :return: 엔트로피 변화가 감지된 부분의 데이터
    """

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

    # Create a DataFrame to store i values and corresponding entropy_values
    # entropy_df = pd.DataFrame({'i': np.arange(num_windows), 'entropy_values': entropy_values})
    # print(entropy_df)
    entropy_df = np.array(entropy_values)

    entropy_plt = np.zeros((num_windows * window_length, 1))
    i = 40
    for i in range(len(entropy_plt)):
        num = i // 10
        entropy_plt[i] = entropy_df[int(num)]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(ASVM_en.flatten(), label='ASVM')
    plt.legend()
    plt.title("ASVM and Entropy")

    plt.subplot(2, 1, 2)
    plt.plot(entropy_plt, label='Entropy_ASVM')
    plt.legend()

    plt.tight_layout()
    plt.show()


'-----------------------------------------------------------------------------------------------------------------------'
# import numpy as np
# from scipy.stats import entropy
# import pandas as pd
# import matplotlib.pyplot as plt
# #
# # CSV 파일 로드
# data = pd.read_csv("C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/P01D02R01_a.csv")
# data = data.iloc[:, 1:]
#
#
# # # 입력 시계열 신호 (임의로 생성)
# # signal = np.random.rand(100)
#
# # 엔트로피 변화 감지에 사용할 슬라이딩 윈도우 크기
# window_size = 10
#
# # 엔트로피 변화를 감지하기 위한 임계값
# threshold = 0.5
#
# def detect_signal_change(signal, window_size, threshold):
#     """
#     시계열 신호에서 엔트로피 변화를 감지하여 해당 부분의 데이터를 추출하는 함수
#
#     :param signal: 입력 시계열 신호
#     :param window_size: 슬라이딩 윈도우 크기
#     :param threshold: 엔트로피 변화를 감지하기 위한 임계값
#     :return: 엔트로피 변화가 감지된 부분의 데이터
#     """
#
#     window_length = 60
#     overlap = 0.5
#
#     num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap))))
#
#     # create an empty array to store the segmented data
#     segments = np.zeros((num_windows, window_length, 8))
#
#     i=0
#     # segment the data using the sliding window technique
#     for i in range(num_windows):
#         start = int(i * (window_length * (1 - overlap)))
#         end = start + window_length
#         segments[i] = data[start:end]
#
#     segments = np.array(segments)
#     print(segments.shape)
#
#     # Calculate entropy for each segment
#     entropy_values = []
#     for segment in segments:
#         segment = np.where(segment == 0, 1e-10, segment)  # segment 배열에서 0인 값을 아주 작은 값인 1e-10으로 대체
#         segment_entropy = entropy(segment)
#         entropy_values.append(segment_entropy)
#
#     # Plot entropy values
#     plt.figure(figsize=(10, 5))
#     for i in range(len(entropy_values)):
#         plt.plot(np.arange(i, i + 1, 1), entropy_values[i])
#
#     plt.xlabel('Window Index')
#     plt.ylabel('Entropy')
#     plt.title('Entropy Change over Windows')
#     plt.show()


 '----------------------------------------------------------------------------------------------------------------------'
    # segments = np.array(segments)
    # print(segments.shape)

    # segments_2d = segments.reshape(-1, segments.shape[-1])
    # columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']
    # segments_df = pd.DataFrame(segments_2d, columns=columns)
    # print(segments_df.head())
'----------------------------------------------------------------------------------------------------------------------'
