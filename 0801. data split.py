
'-------------------------------------------S01~S12 split--------------------------------------------------------'
import pandas as pd
import os
import matplotlib.pyplot as plt
import re


data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/"
split_file_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/"
os.makedirs(split_file_path, exist_ok=True)

e,m,t = 1,11,1
for e in range(13, 21):
    for m in range(2, 15):
        for t in range(1, 4):
            try:
                e_str = str(e).zfill(2)
                m_str = str(m).zfill(2)
                t_str = str(t).zfill(2)

                name = f"S{e_str}D{m_str}R{t_str}_SW"
                name_csv = os.path.join(data_path, f"{name}.csv")
                name_png = f"{name}.png"

                # 파일이 존재하는지 확인
                if not os.path.isfile(name_csv):
                    print(f"File not found: {name_csv}")
                    continue

                # 데이터 읽기
                data = pd.read_csv(name_csv)
                data = data.iloc[:, 2:8]  # Acc_X ~ Gyr_z
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                mid = len(data) // 2


                match = re.search(r'D\d+', name)
                if match:
                    d_part = match.group(0)  # 예: D04
                    file_name_1 = f"{name.replace(d_part, f'{d_part}_1').replace('_SW', '')}.csv"
                    file_name_2 = f"{name.replace(d_part, f'{d_part}_2').replace('_SW', '')}.csv"
                    file_name_3 = f"{name.replace(d_part, f'{d_part}_3').replace('_SW', '')}.csv"


            # m_str을 정수로 변환 후 비교
                m_int = int(m_str)
                if m_int in [3,4,13,14]:
                    data_1 = data.iloc[:mid, :]
                    data_3 = data.iloc[mid:, :]

                    data_1.to_csv(os.path.join(split_file_path, file_name_1), index=False)
                    data_3.to_csv(os.path.join(split_file_path, file_name_3), index=False)


                # if m_int in [10]:
                #     data_1 = data.iloc[:mid, :]
                #     data_2 = data.iloc[mid:, :]
                #
                #     data_1.to_csv(os.path.join(split_file_path, file_name_1), index=False)
                #     data_2.to_csv(os.path.join(split_file_path, file_name_2), index=False)


                # if m_int in [3, 4, 10, 13]:
                #     data_1 = data.iloc[:mid, :]
                #     data_2 = data.iloc[mid:, :]
                #
                #     data_1.to_csv(os.path.join(split_file_path, file_name_1), index=False)
                #     data_2.to_csv(os.path.join(split_file_path, file_name_2), index=False)


                    # # 그래프 저장
                    # plt.figure(figsize=(10, 8))
                    # plt.subplot(2, 1, 1)
                    # plt.plot(data['ASVM'], label='ASVM')
                    # plt.title(f'{name}')
                    # plt.ylabel('ASVM')
                    # plt.xlabel('Frames')
                    # plt.legend()
                    #
                    # plt.subplot(2, 1, 2)
                    # plt.plot(data['GSVM'], label='GSVM')
                    # plt.ylabel('GSVM')
                    # plt.xlabel('Frames')
                    # plt.legend()
                    #
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(split_file_path, name_png))
                    # plt.close()


            except Exception as e:
                print(f"Error processing file {name_csv}: {e}")






'-------------------------------------------D12 split--------------------------------------------------------'
import pandas as pd
import os
import matplotlib.pyplot as plt
import re


data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/"
split_file_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/"
os.makedirs(split_file_path, exist_ok=True)
# e,m,t = 1,12,1

for e in range(1, 21):
    for m in range(12, 13):
        for t in range(1, 4):
            try:
                e_str = str(e).zfill(2)
                m_str = str(m).zfill(2)
                t_str = str(t).zfill(2)

                name = f"S{e_str}D{m_str}R{t_str}_SW"
                name_csv = os.path.join(data_path, f"{name}.csv")
                name_png = f"{name}.png"


                # 데이터 읽기
                data = pd.read_csv(name_csv)
                data = data.iloc[:, 2:8]  # Acc_X ~ Gyr_z
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                max_index = data['ASVM'].idxmax()

                split = max_index + 100



                match = re.search(r'D\d+', name)
                if match:
                    d_part = match.group(0)
                    file_name_1 = f"{name.replace(d_part, f'{d_part}_1').replace('_SW', '')}.csv"
                    file_name_3 = f"{name.replace(d_part, f'{d_part}_2').replace('_SW', '')}.csv"

                    data_1 = data.iloc[:split, :]
                    data_3 = data.iloc[split:, :]

                    data_1.to_csv(os.path.join(split_file_path, file_name_1), index=False)
                    data_3.to_csv(os.path.join(split_file_path, file_name_3), index=False)


                    # # 그래프 저장
                    # plt.figure(figsize=(10, 8))
                    # plt.subplot(2, 1, 1)
                    # plt.plot(data['ASVM'], label='ASVM')
                    # plt.title(f'{name} - ASVM')
                    # plt.ylabel('ASVM')
                    # plt.xlabel('Frames')
                    # plt.legend()
                    #
                    # plt.subplot(2, 1, 2)
                    # plt.plot(data['GSVM'], label='GSVM')
                    # plt.title(f'{name} - GSVM')
                    # plt.ylabel('GSVM')
                    # plt.xlabel('Frames')
                    # plt.legend()
                    #
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(split_file_path, name_png))
                    # plt.close()

            except Exception as e:
                print(f"Error processing file {name_csv}: {e}")




'-------------------------------------------S13~S20 D02,10,11 split--------------------------------------------------------'
import pandas as pd
import os

data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/"
split_file_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/세분화/"
os.makedirs(split_file_path, exist_ok=True)

# D02, D10, D11에 해당하는 패턴
patterns = ['D02', 'D10', 'D11']

# 파일 목록을 가져옴
file_list = os.listdir(data_path)

for file_name in file_list:
    # 파일이 CSV 파일인지 확인
    if file_name.endswith('.csv'):
        # 파일명이 S13~S20로 시작하는지 확인
        if re.match(r"S1[3-9]|S20", file_name):
            # 파일명이 D02, D10, D11로 시작하는지 확인
            if any(file_name[3:].startswith(pattern) for pattern in patterns):
                try:
                    # 파일 경로 생성
                    name_csv = os.path.join(data_path, file_name)

                    # 데이터 읽기
                    data = pd.read_csv(name_csv)
                    data = data.iloc[:, 2:8]  # 첫 8행 및 Acc_X ~ Gyr_Z 컬럼 선택
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                    # 새로운 파일 이름 생성 (뒤의 '_SW'만 제거)
                    new_file_name = file_name.replace('_SW', '')
                    data.to_csv(os.path.join(split_file_path, new_file_name), index=False)
                    print(f"Saved file: {new_file_name}")

                except Exception as e:
                    print(f"Error processing file {name_csv}: {e}")





'-------------------------------------------S1~S20 D01, D05 가져오기--------------------------------------------------------'
import pandas as pd
import os

data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/"
split_file_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/세분화/"
os.makedirs(split_file_path, exist_ok=True)

# D02, D10, D11에 해당하는 패턴
patterns = ['D01', 'D05']

# 파일 목록을 가져옴
file_list = os.listdir(data_path)

for file_name in file_list:
    # 파일이 CSV 파일인지 확인
    if file_name.endswith('.csv'):

        if any(file_name[3:].startswith(pattern) for pattern in patterns):
            try:
                # 파일 경로 생성
                name_csv = os.path.join(data_path, file_name)

                # 데이터 읽기
                data = pd.read_csv(name_csv)
                data = data.iloc[:, 2:8]  # 첫 8행 및 Acc_X ~ Gyr_Z 컬럼 선택
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                # 새로운 파일 이름 생성 (뒤의 '_SW'만 제거)
                new_file_name = file_name.replace('_SW', '')
                data.to_csv(os.path.join(split_file_path, new_file_name), index=False)
                print(f"Saved file: {new_file_name}")

            except Exception as e:
                print(f"Error processing file {name_csv}: {e}")





'-------------------------------------------S1~S20 Fall 가져오기----------------------------------------------------'
import pandas as pd
import os

data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (원본)/"
split_file_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/fall/"
os.makedirs(split_file_path, exist_ok=True)

# # D02, D10, D11에 해당하는 패턴
patterns = ['F']

# 파일 목록을 가져옴
file_list = os.listdir(data_path)

for file_name in file_list:
    # 파일이 CSV 파일인지 확인
    if file_name.endswith('.csv'):

        if any(file_name[3:].startswith(pattern) for pattern in patterns):
            try:
                # 파일 경로 생성
                name_csv = os.path.join(data_path, file_name)

                # 데이터 읽기
                data = pd.read_csv(name_csv)
                data = data.iloc[:, 2:8]  # 첫 8행 및 Acc_X ~ Gyr_Z 컬럼 선택
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                # 새로운 파일 이름 생성 (뒤의 '_SW'만 제거)
                new_file_name = file_name.replace('_SW', '')
                data.to_csv(os.path.join(split_file_path, new_file_name), index=False)
                print(f"Saved file: {new_file_name}")

            except Exception as e:
                print(f"Error processing file {name_csv}: {e}")





'-------------------------------------------- SW 떼기-----------------------------------------------------'
import os
# 폴더 경로 설정
folder_path = r"C:\Users\abbey\Desktop\20명 데이터 세분화 (split)\새 폴더"

# 폴더 내의 모든 파일명 가져오기
for filename in os.listdir(folder_path):
    # 파일명에 _SW가 포함되어 있는지 확인
    if '_SW' in filename:
        # 새로운 파일명 생성 (_SW 제거)
        new_filename = filename.replace('_SW', '')
        # 기존 파일 경로
        old_file = os.path.join(folder_path, filename)
        # 새로운 파일 경로
        new_file = os.path.join(folder_path, new_filename)
        # 파일명 변경
        os.rename(old_file, new_file)

print("파일명 변경 완료")





'--------------------------------------------S13~S20 D02,10,11 복붙 -----------------------------------------------------'
import os
import shutil

# 원본 폴더 경로 설정
source_folder = r"C:\Users\abbey\Desktop\새 폴더 (2)"

# 대상 폴더 경로 설정
target_folder1 = r"C:\Users\abbey\Desktop\20명 데이터 세분화 (split)\새 폴더"
target_folder2 = r"C:\Users\abbey\Desktop\20명 데이터 세분화 (split)\새 폴더"

# 폴더 내의 모든 파일명 가져오기
for filename in os.listdir(source_folder):
    # 첫 번째 조건에 맞는 파일들 복사
    if len(filename) > 4 and filename[3] == 'D' and filename[4] == '0' and filename[5] == '2':
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder1, filename)
        shutil.copy2(source_file, target_file)

    # 두 번째 조건에 맞는 파일들 복사
    elif len(filename) > 4 and filename[3] == 'D' and filename[4] == '1' and filename[5] in ['0', '1']:
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder2, filename)
        shutil.copy2(source_file, target_file)

print("파일 복사 완료")






'-------------------------------------------------D01만 가져오기--------------------------------------------------------'
import pandas as pd
import os

data_path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split)/"
split_file_path = r"C:/Users/abbey/Desktop/D01 평균/"
os.makedirs(split_file_path, exist_ok=True)

# D02, D10, D11에 해당하는 패턴
patterns = ['D01']

# 파일 목록을 가져옴
file_list = os.listdir(data_path)

for file_name in file_list:
    # 파일이 CSV 파일인지 확인
    if file_name.endswith('.csv'):

        if any(file_name[3:].startswith(pattern) for pattern in patterns):
            try:
                # 파일 경로 생성
                name_csv = os.path.join(data_path, file_name)

                # 데이터 읽기
                data = pd.read_csv(name_csv)

                data.to_csv(os.path.join(split_file_path, file_name), index=False)
                print(f"Saved file: {file_name}")

            except Exception as e:
                print(f"Error processing file {name_csv}: {e}")






'------------------------------------------------D01 ASVM 평균 분산 csv 파일만들기 -----------------------------------------------------'
import numpy as np
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt


data_list = []
csvpath = r"C:/Users/abbey/Desktop/D01 평균/"

for e in range(1, 21):
    for m in range(1, 2):
        for t in range(1, 4):
            e = str(e).zfill(2)
            m = str(m).zfill(2)
            t = str(t).zfill(2)

            path = r"C:/Users/abbey/Desktop/D01 평균/"
            name = "S%sD%sR%s" % (e, m, t)
            name_csv = "%s%s.csv" % (path, name)
            name_png = "%s.png" % (name)

            data = pd.read_csv(str(name_csv), header=2)
            data = pd.DataFrame(data)
            data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

            mean = np.mean(data.ASVM)
            variance = np.var(data.ASVM)

            mean_file = pd.DataFrame({'Mean': [mean]})
            variance_file = pd.DataFrame({'Variance': [variance]})
            data_name = pd.DataFrame({'Name': [name]})

            data_merge = pd.concat((data_name, mean_file, variance_file), axis=1, ignore_index=False)
            data_list.append(data_merge)


            merged_data = pd.concat(data_list, ignore_index=True)

            merged_data.to_csv(csvpath + 'mean_ASVM.csv', index=False)






'------------------------------------------------D01 GSVM 평균 분산 csv 파일만들기 -----------------------------------------------------'
import numpy as np
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt


data_list = []
csvpath = r"C:/Users/abbey/Desktop/D01 평균/"

for e in range(1, 21):
    for m in range(1, 2):
        for t in range(1, 4):
            e = str(e).zfill(2)
            m = str(m).zfill(2)
            t = str(t).zfill(2)

            path = r"C:/Users/abbey/Desktop/D01 평균/"
            name = "S%sD%sR%s" % (e, m, t)
            name_csv = "%s%s.csv" % (path, name)
            name_png = "%s.png" % (name)

            data = pd.read_csv(str(name_csv), header=2)
            data = pd.DataFrame(data)
            data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

            mean = np.mean(data.GSVM)
            variance = np.var(data.GSVM)

            mean_file = pd.DataFrame({'Mean': [mean]})
            variance_file = pd.DataFrame({'Variance': [variance]})
            data_name = pd.DataFrame({'Name': [name]})

            data_merge = pd.concat((data_name, mean_file, variance_file), axis=1, ignore_index=False)
            data_list.append(data_merge)


            merged_data = pd.concat(data_list, ignore_index=True)
            #
            # merged_data.to_csv(csvpath + 'mean_GSVM.csv', index=False)





'-----------------------------------------D01 평균 분산 피험자별로 연산한 csv 파일만들기 ------------------------------'
import pandas as pd

# 파일 경로 설정
input_path = r'C:\Users\abbey\Desktop\D01 평균\mean_final.csv'
output_path = r'C:\Users\abbey\Desktop\D01 평균\mean_final_processed.csv'

# CSV 파일을 데이터 프레임으로 읽기
df = pd.read_csv(input_path)

# 새로운 데이터 프레임 생성
output_df = pd.DataFrame({
    'Subject': [f'S{str(i).zfill(2)}D01' for i in range(1, 21)]
})

# 열마다 평균 계산 및 데이터 프레임에 추가
for col in df.columns[1:]:  # 첫 번째 열(인덱스 0)은 제외하고 나머지 열들에 대해 반복
    means = []
    for start_row in range(0, 60, 3):  # 1, 4, 7, ..., 58 번째 행에서 시작
        mean_value = df.iloc[start_row:start_row + 3][col].mean()
        means.append(mean_value)
    output_df[col] = means

# 계산된 결과를 새로운 CSV 파일로 저장
output_df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")





'------------------------------------------------D01 표준편차 계산 -----------------------------------------------------'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 파일 경로 설정
std_path = r'C:\Users\abbey\Desktop\D01 평균\mean_final_processed.csv'

# CSV 파일을 데이터 프레임으로 읽기
std_data = pd.read_csv(std_path)
std_data.columns = ['Name', 'Mean(A)', 'Variance(A)','Mean(G)', 'Variance(G)']

# 숫자를 과학적 표기법으로 출력하도록 설정
pd.set_option('display.float_format', '{:.2e}'.format)

# 데이터 파일 경로 설정
path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split) _ADL/*.csv"
csv_files = glob.glob(path)

# 결과를 저장할 리스트
results = []

for file in csv_files:
    try:
        # 파일명 추출
        name = os.path.basename(file).split('.')[0]
        name_csv = "%s%s.csv" % (path, name)
        name_png = "%s.png" % (name)

        data = pd.read_csv(file, header=1)
        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

        # name의 앞 3자리 추출
        name_prefix = name[:3]
        matching_rows = std_data[std_data['Name'].str[:3] == name_prefix]

        if not matching_rows.empty:
            # 'A'와 'G'에 대해 각각 계산
            thresholds = {}
            for var_type in ['A', 'G']:
                standard_deviation = matching_rows[f'Variance({var_type})'].iloc[0]
                final_mean = matching_rows[f'Mean({var_type})'].iloc[0]

                if standard_deviation is not None:
                    std = standard_deviation ** 0.5
                    threshold = final_mean + (std * 3)
                    thresholds[var_type] = threshold
                else:
                    thresholds[var_type] = None

            threshold_A = thresholds['A']
            threshold_G = thresholds['G']

            # 그래프 경로 설정
            path_graph = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split) _ADL/그래프(A+G)/"

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.title(name)
            plt.plot(data.ASVM)  # 그래프 그리기
            plt.axhline(threshold_A, color='red', linewidth=1)
            plt.ylabel('ASVM (g)')

            # 교차점 찾기
            crossing_points = np.where(np.diff(np.sign(data.ASVM - threshold_A)))[0]

            # 교차점 표시
            for cp in crossing_points:
                plt.plot(cp, data.ASVM[cp], 'ro', markersize=1)

            # 첫 번째 교차점과 마지막 교차점을 변수에 저장
            ASVM_first = crossing_points[0] if crossing_points.size > 0 else None
            ASVM_last = crossing_points[-1] if crossing_points.size > 0 else None

            if crossing_points.size > 0:
                plt.axvline(ASVM_first, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(ASVM_last, color='green', linestyle='-')  # 마지막 교차점

            plt.subplot(2, 1, 2)
            plt.plot(data.GSVM)  # 그래프 그리기
            plt.axhline(threshold_G, color='red', linewidth=1)
            plt.ylabel('GSVM (g)')
            plt.xlabel('Frames')

            # 교차점 찾기
            crossing_points1 = np.where(np.diff(np.sign(data.GSVM - threshold_G)))[0]

            # 교차점 표시
            for cp in crossing_points1:
                plt.plot(cp, data.GSVM[cp], 'ro', markersize=1)

            # 첫 번째 교차점과 마지막 교차점을 변수에 저장
            GSVM_first = crossing_points1[0] if crossing_points1.size > 0 else None
            GSVM_last = crossing_points1[-1] if crossing_points1.size > 0 else None

            if crossing_points1.size > 0:
                plt.axvline(GSVM_first, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(GSVM_last, color='green', linestyle='-')  # 마지막 교차점


            plt.close()

            # plt.tight_layout()
            # # plt.show()
            # # plt.savefig(path_graph + name_png)
            # # plt.close()

            # 결과를 데이터프레임에 저장
            result_frame = pd.DataFrame({
                'name': [name],
                'ASVM_first': [ASVM_first],
                'ASVM_last': [ASVM_last],
                'GSVM_first': [GSVM_first],
                'GSVM_last': [GSVM_last]
            })

            results.append(result_frame)

        else:
            print(f"No matching row for {name_prefix}")

    except Exception as e:
        print(f"An error occurred with file {file}: {e}")
        pass

# 최종 결과를 데이터프레임으로 합치고 CSV 파일로 저장
final_result_df = pd.concat(results, ignore_index=True)
final_result_df.to_csv(r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split) _ADL/그래프(A+G)/frame.csv", index=False)






'-----------------------------------------ASVM, GSVM 프레임 min max-----------------------------------------------'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# # CSV 파일 경로
path_csv = "C:/Users/abbey\Desktop/20명 데이터 split/그래프(A+G)/frame.csv"

# CSV 파일 불러오기
raw = pd.read_csv(path_csv)
raw.columns = ['name', 'ASVM_first', 'ASVM_last', 'GSVM_first', 'GSVM_last']

# 계산 결과를 저장할 리스트
results = []

for i in range(len(raw)):
    a = raw.loc[i, 'ASVM_first']
    b = raw.loc[i, 'GSVM_first']
    c = raw.loc[i, 'ASVM_last']
    d = raw.loc[i, 'GSVM_last']

    # NaN 값을 제외하고 최소값, 최대값 계산
    first = min([a, b], key=lambda x: np.inf if np.isnan(x) else x)
    last = max([c, d], key=lambda x: -np.inf if np.isnan(x) else x)

    # 결과를 리스트에 추가
    results.append({'name': raw.loc[i, 'name'], 'first': first, 'last': last})

# 결과를 데이터프레임으로 변환
result_df = pd.DataFrame(results)

save_path = r"C:/Users/abbey\Desktop/20명 데이터 split/그래프(A+G)/min_max.csv"
result_df.to_csv(save_path, index=False)


# 데이터 파일 경로 설정
path = r"C:/Users/abbey/Desktop/20명 데이터 세분화 (split) _ADL/*.csv"
csv_files = glob.glob(path)
# 결과를 저장할 리스트
results = []
for file in csv_files:
    try:
        # 파일명 추출
        name = os.path.basename(file).split('.')[0]
        name_csv = "%s%s.csv" % (path, name)
        name_png = "%s.png" % (name)

        data = pd.read_csv(file, header=1)
        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']


        match = result_df[result_df['name'] == name]

        if match.empty or np.isnan(match['first'].values[0]) or np.isnan(match['last'].values[0]):
            # first_value 또는 last_value가 NaN인 경우: 전체 데이터를 extract에 저장
            extract = data
        else:
            # first_value와 last_value를 정수로 변환
            first_value = int(match['first'].values[0])
            last_value = int(match['last'].values[0])

            # 해당 구간 데이터를 extract에 저장
            extract = data.iloc[first_value:last_value + 1, :]

        # 파일 저장
        save_data = "C:/Users/abbey/Desktop/20명 데이터 split/"
        save_csv = "%s%s.csv" % (save_data, name)
        extract.to_csv(save_csv, index=False)


        path_graph = "C:/Users/abbey\Desktop/20명 데이터 split/그래프/"

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
        plt.title(name)
        # plt.title('Acceleration SVM')
        plt.ylabel('ASVM(g)')
        plt.xlabel('Frames')
        plt.plot(extract.ASVM, label='ASVM')

        plt.subplot(2, 1, 2)
        # plt.title('Angular Velocity SVM')
        plt.ylabel('GSVM (degree/s)')
        plt.xlabel('Frames')
        plt.plot(extract.GSVM, label='GSVM')

        plt.tight_layout()
        plt.show()

        plt.savefig(path_graph + name_png)
        plt.close()


    except:
        pass



















