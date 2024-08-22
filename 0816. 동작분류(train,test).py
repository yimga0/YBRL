import numpy as np
import pandas as pd
import glob
import os

# 윈도우 크기와 슬라이드 간격 정의
window_length = 120
overlap_ratio = 0.5
overlap = int(window_length * overlap_ratio)

# 주체 번호 리스트 생성 및 train/test 분리
subject_list = np.arange(1, 21)
np.random.seed(7)  # 랜덤 시드 고정
subject_list_test = np.random.choice(subject_list, 4, replace=False)
list_test = list(subject_list_test)
list_train = [x for x in subject_list if x not in list_test]

# 결과를 저장할 리스트 초기화
train_data_list = []
test_data_list = []

# 데이터 경로 설정
path = r"C:/Users/abbey/Desktop/20명 데이터 split/"
csv_files = glob.glob(os.path.join(path, "*.csv"))

# 각 파일에 대해 데이터 처리
for file in csv_files:
    try:
        # 파일명에서 필요한 정보 추출
        name = os.path.basename(file)
        movement = name[3]  # 4번째 글자
        subject_number = int(name[1:3])  # 2번째부터 3번째까지가 주체 번호
        identifier = name[3:8]  # 4번째부터 8번째까지 (예: D01_1, F02_)

        # 데이터 로드 및 전처리
        data = pd.read_csv(file, header=1)
        data = data.iloc[:, 0:8]
        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

        # 데이터 분할 및 클래스 부여
        if movement == 'D':
            num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))
            data_unit = np.zeros((num_windows, window_length, 8))

            for i in range(num_windows):
                start = int(i * (window_length * (1 - overlap_ratio)))
                end = start + window_length
                data_unit[i] = data[start:end]

            df = pd.DataFrame(data_unit.reshape(num_windows, -1))

            # 클래스 부여
            if identifier.startswith('D01'):
                train_class = 1
            elif identifier in ['D02_2', 'D11_2']:
                train_class = 2
            elif identifier in ['D02_1', 'D03_1', 'D04_1', 'D11_1', 'D12_1', 'D13_1','D14_1']:
                train_class = 3
            elif identifier in ['D02_3', 'D03_3', 'D04_3', 'D11_3', 'D12_2', 'D13_3','D14_3']:
                train_class = 4
                print(f"Class 4 data found in file: {name}")
            elif identifier.startswith('D05'):
                train_class = 5
            elif identifier == 'D10_1':
                train_class = 6
            elif identifier == 'D10_2':
                train_class = 7
            else:
                continue  # 조건에 맞지 않으면 skip

            train_class_df = pd.DataFrame([train_class] * num_windows, columns=['class'])
            train_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})

        elif movement == 'F':
            impact_frame = np.argmax(data.ASVM)
            df = pd.DataFrame(data.iloc[impact_frame - 60:impact_frame + 60, :].values.reshape(1, -1))

            # 클래스 부여
            if identifier[1:3] in ['02', '03', '05', '08', '09']:
                train_class = 8
            elif identifier[1:3] in ['01', '04', '07', '11']:
                train_class = 9
            elif identifier[1:3] in ['06', '10']:
                train_class = 10
            else:
                continue  # 조건에 맞지 않으면 skip

            train_class_df = pd.DataFrame([train_class], columns=['class'])
            train_file = pd.DataFrame({'window': [name]})

        # 최종 데이터 병합 및 train/test 분리
        train_merge = pd.concat([df, train_class_df, train_file], axis=1)

        if subject_number in list_train:
            train_data_list.append(train_merge)
        elif subject_number in list_test:
            test_data_list.append(train_merge)

    except Exception as e:
        print(f"Error processing file {name}: {e}")
        continue

# 모든 데이터 병합 및 CSV 파일로 저장
train_data = pd.concat(train_data_list, axis=0, ignore_index=True)
train_data.to_csv('C:/Users/abbey/Desktop/20명 데이터 split/train_data.csv', index=False)

test_data = pd.concat(test_data_list, axis=0, ignore_index=True)
test_data.to_csv('C:/Users/abbey/Desktop/20명 데이터 split/test_data.csv', index=False)


