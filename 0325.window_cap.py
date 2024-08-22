import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt


# 경로 설정
# path_graph = r"C:/Users/abbey/Desktop/extract_data/"


# 윈도우 크기와 슬라이드 간격 정의
window_length = 120

overlap_ratio = 0
overlap = int(window_length * overlap_ratio)
# 결과를 저장할 리스트 초기화
windowed_data = []

'---------------------------------------------------------------------------------------------------------------------'

subject_list = np.arange(1, 23)
subject_list = np.delete(np.arange(1, 23), [1, 17])  # 2와 18을 뺀 리스트 생성
np.random.seed(7) #
subject_list_test = np.random.choice(subject_list, 4, replace=False) #임의표본추출(모집단,선택할개수,중복허용여부)
list_test = list(subject_list_test)
list_train = [x for x in list(subject_list) if x not in list(subject_list_test)]
'---------------------------------------------------------------------------------------------------------------------'


count = 0

# train_stack = []
train_data_list = []

for e in list_train:
    for m in range(1, 15):
        for t in range(1, 4):
            train_class = []
            train_stack = []

            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/extract_data/"
                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[:, 0:8]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM','GSVM']

                num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                data_unit = np.zeros((num_windows, window_length, 8))


                for i in range(num_windows):
                    start = int(i * (window_length * (1 - overlap_ratio)))
                    end = start + window_length
                    data_unit[i] = data[start:end]

                data_unit = np.array(data_unit)

                df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                df = pd.DataFrame(df)


                if count == 0:  # 데이터 쌓기
                    train_stack = df
                else:
                    train_stack = pd.concat([train_stack, df], axis=0)

                train_stack = pd.DataFrame(train_stack)



                if 'D%s' % m == 'D01':
                    train_class.append(1)

                elif 'D%s' % m in ['D02', 'D03', 'D04', 'D11', 'D12', 'D13', 'D14']:
                    train_class.append(2)

                elif 'D%s' % m in ['D05', 'D06', 'D07', 'D10']:
                    train_class.append(3)

                elif 'D%s' % m in ['D08', 'D09']:
                    train_class.append(4)


                train_class_df = pd.DataFrame(train_class, columns=['class'])
                train_class_indicator = pd.concat([train_class_df] * num_windows, ignore_index=True)
                # train_class_df = pd.DataFrame(for train_class  in range(1, num_windows + 1)

                i=1
                for i in range(1, num_windows + 1):
                    window_name = f"{name}_{i}"
                    train_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})

                train_merge = pd.concat([train_class_indicator, train_file, train_stack], axis=1, sort=False)
                train_data_list.append(train_merge)
                # 데이터 합치기


            except:
                pass


train_data = pd.concat(train_data_list, axis=0, ignore_index=True)  # 모든 데이터를 하나로 합치기

'--------------------------------------------------------------------------------------------------------------------'

count = 0

# train_stack = []
test_data_list = []

for e in list_test:
    for m in range(1, 15):
        for t in range(1, 4):
            test_class = []
            test_stack = []

            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/extract_data/"
                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[:, 0:8]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM','GSVM']

                num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                data_unit = np.zeros((num_windows, window_length, 8))


                for i in range(num_windows):
                    start = int(i * (window_length * (1 - overlap_ratio)))
                    end = start + window_length
                    data_unit[i] = data[start:end]

                data_unit = np.array(data_unit)

                df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                df = pd.DataFrame(df)


                if count == 0:  # 데이터 쌓기
                    test_stack = df
                else:
                    test = pd.concat([test_stack, df], axis=0)

                test_stack = pd.DataFrame(test_stack)



                if 'D%s' % m == 'D01':
                    test_class.append(1)

                elif 'D%s' % m in ['D02', 'D03', 'D04', 'D11', 'D12', 'D13', 'D14']:
                    test_class.append(2)

                elif 'D%s' % m in ['D05', 'D06', 'D07', 'D10']:
                    test_class.append(3)

                elif 'D%s' % m in ['D08', 'D09']:
                    test_class.append(4)


                test_class_df = pd.DataFrame(test_class, columns=['class'])
                test_class_indicator = pd.concat([test_class_df] * num_windows, ignore_index=True)
                # train_class_df = pd.DataFrame(for train_class  in range(1, num_windows + 1)

                i=1
                for i in range(1, num_windows + 1):
                    window_name = f"{name}_{i}"
                    test_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})

                test_merge = pd.concat([test_class_indicator, test_file, test_stack], axis=1, sort=False)
                test_data_list.append(test_merge)
                # 데이터 합치기


            except:
                pass


test_data = pd.concat(test_data_list, axis=0, ignore_index=True)  # 모든 데이터를 하나로 합치기
