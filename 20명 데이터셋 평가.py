
'----------------------------------------------------alarm 개수 세기---------------------------------------------------------'

import pandas as pd

count = 0
name_alarm=[]

for e in range(13, 20):
    for m in range(1,18):
        for t in range(1,4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/어플실험(기존)/S%s/SW_v2/" %e
                name = "T_S%sF%sR%s_SW_v2" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


                if not (data.alarm == 1).any():
                    count += 1
                    name_alarm.append(name)

            except:
                pass

name_alarm_df = pd.DataFrame(name_alarm, columns=['File Names'])





'-----------------------------------------파일명바꾸기---------------------------------------------------------------'
import os
import re

# 디렉토리 경로 설정
directory = r"C:/Users/abbey/Desktop/어플실험(기존)/S19/SW_v2/"

# 파일명 변경 함수
def rename_files(directory):
    # 지정된 패턴 (S17F02R01 부터 S17F17R03 까지)
    pattern = re.compile(r'^S19F(0[2-9]|1[0-7])R(\d{2})')

    for filename in os.listdir(directory):
        # 패턴 매칭
        match = pattern.match(filename)
        if match:
            new_filename = filename.replace('F', 'D', 1)  # 첫 번째 'F'만 'D'로 변경
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')

# 파일명 변경 함수 호출
rename_files(directory)


'-----------------------------------------T_파일명바꾸기---------------------------------------------------------------'
import os
import re

# 디렉토리 경로 설정
directory =  r"C:/Users/abbey/Desktop/S06 SW/변환/"


# 파일명 변경 함수
def rename_files(directory):
    # 지정된 패턴 (T_S18F01R01_SW_v2 부터 T_S18F17R03_SW_v2 까지)
    pattern = re.compile(r'^T_S06F(0[1-9]|1[0-7])R(\d{2})_SW')

    for filename in os.listdir(directory):
        # 패턴 매칭
        match = pattern.match(filename)
        if match:
            new_filename = filename.replace('T_', '', 1)  # 첫 번째 'T_'만 제거
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')

# 파일명 변경 함수 호출
rename_files(directory)


'-----------------------------------------v2 데이터 기존으로 라벨링(ADL)----------------------------------------------'

import os
import re
import shutil

# 디렉토리 경로 설정
input_directory =  r"C:/Users/abbey/Desktop/새 폴더 (2)"
output_directory =  r"C:/Users/abbey/Desktop/새 폴더 (2)/라벨링"

# 출력 디렉토리가 없으면 생성
os.makedirs(output_directory, exist_ok=True)

# D 숫자 변경 매핑
d_mapping = {
    "01": "01",
    "02": "03",
    "03": "04",
    "04": "02_1",
    "05": "02_2",
    "06": "02_3",
    "08": "05",
    "09": "11_2",
    "10": "11_1",
    "11": "11_3",
    "14": "12",
    "17": "13",
    "20": "10_1",  # D20의 경우 두 개의 매핑이 있어 하나만 유지했습니다.
    "21": "10_2",
    "22": "14"
}

# 파일명 변경 함수
def rename_files(input_directory, output_directory):
    # 지정된 패턴 (S13D01R01_SW_v2 형태)
    pattern = re.compile(r'^S20D(\d{2})R(\d{2})_SW_v2')

    for filename in os.listdir(input_directory):
        match = pattern.match(filename)
        if match:
            d_num = match.group(1)
            if d_num in d_mapping:
                new_d_num = d_mapping[d_num]
                new_filename = filename.replace(f"D{d_num}", f"D{new_d_num}", 1)
                old_file = os.path.join(input_directory, filename)
                new_file = os.path.join(output_directory, new_filename)
                shutil.copy(old_file, new_file)
                print(f'Copied: {old_file} -> {new_file}')
            else:
                # 매핑되지 않은 경우는 복사하지 않음
                print(f'Skipped: {filename}')

# 파일명 변경 함수 호출
rename_files(input_directory, output_directory)


'-----------------------------------------v2 데이터 기존으로 라벨링(Fall)----------------------------------------------'

import os
import re
import shutil

# 디렉토리 경로 설정
input_directory =  r"C:/Users/abbey/Desktop/새 폴더 (2)"
output_directory =  r"C:/Users/abbey/Desktop/새 폴더 (2)/라벨링"


# 출력 디렉토리가 없으면 생성
os.makedirs(output_directory, exist_ok=True)

# D 숫자 변경 매핑
f_mapping = {
    "02": "04",
    "04": "08",
    "06": "05",
    "07": "06",
    "08": "07",
    "11": "02",
    "14": "01",
    "15": "09",
    "16": "10",
    "17": "11"
}

# 파일명 변경 함수
def rename_files(input_directory, output_directory):
    # 지정된 패턴 (S13F01R01_SW_v2 형태)
    pattern = re.compile(r'^S20F(\d{2})R(\d{2})_SW_v2')

    for filename in os.listdir(input_directory):
        match = pattern.match(filename)
        if match:
            f_num = match.group(1)
            if f_num in f_mapping:
                new_f_num = f_mapping[f_num]
                new_filename = filename.replace(f"F{f_num}", f"F{new_f_num}", 1)
                old_file = os.path.join(input_directory, filename)
                new_file = os.path.join(output_directory, new_filename)
                shutil.copy(old_file, new_file)
                print(f'Copied: {old_file} -> {new_file}')
            else:
                # 매핑되지 않은 경우는 복사하지 않음
                print(f'Skipped: {filename}')

# 파일명 변경 함수 호출
rename_files(input_directory, output_directory)




'--------------------------------------------------   _v2 제거 -------------------------------------------------------'

import os

# 폴더 경로 지정
path = r"C:/Users/abbey/Desktop/S20 SW/라벨링/ADL"


# 폴더 내 파일 목록 가져오기
folder = os.listdir(path)

# 파일 이름에서 '_v2' 제거 및 이름 변경
for filename in folder:
    if '_v2' in filename:
        new_filename = filename.replace('_v2', '')
        old_filepath = os.path.join(path, filename)
        new_filepath = os.path.join(path, new_filename)

        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')



'----------------------------------------------- Raw 떼기  ---------------------------------------------------------'

import os

# 경로 설정
path = r'C:\Users\abbey\Desktop\20명 평가\12명'

try:
    # 디렉토리 내 모든 파일명 가져오기
    file_names = os.listdir(path)

    # 파일명 변경
    for file_name in file_names:
        # 파일명에서 "Raw"라는 세 글자를 제거합니다.
        new_file_name = file_name.replace('Raw', '')

        # 파일명 변경
        os.rename(os.path.join(path, file_name), os.path.join(path, new_file_name))
        print(f"{file_name} -> {new_file_name}")

    print("파일명 변경 완료!")

except Exception as e:
    print(f"파일명 변경 중 오류 발생: {e}")





'-----------------------------------------Leadtime 계산1----------------------------------------------------'

import pandas as pd
name_alarm = []
leadtime =[]
freams=[]

e, m, t = 1,1,1
for e in range(13, 19):
    for m in range(1,18):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path =  r"C:/Users/abbey/Desktop/SW실험/V2 S13-S18/"
                name = "S%sF%sR%s_SW_v2" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                if (data.alarm == 1).any():
                    name_alarm.append(name)

                    detect = data[data.alarm == 1].index[0]
                    app = data['ASVM'].idxmax()
                    frame = app-detect
                    freams.append(frame)
                    time = (app-detect)/50
                    leadtime.append(time)

                else:
                    pass

            except:
                pass

name_alarm_df = pd.DataFrame(name_alarm, columns=['File Names'])
leadtime_df = pd.DataFrame(leadtime, columns=['Leadtime'])
frame_df = pd.DataFrame(freams, columns=['Frames'])
merge1 = pd.concat([name_alarm_df, leadtime_df, frame_df], axis=1)


'---------------------Leadtime 계산2-----------------------'
name_alarm = []
leadtime =[]
freams=[]

e, m, t = 1,1,1
for e in range(13, 19):
    for m in range(1,18):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path =  r"C:/Users/abbey/Desktop/SW실험/V2 S13-S18/"
                name = "T_S%sF%sR%s_SW_v2" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                if (data.alarm == 1).any():
                    name_alarm.append(name)

                    detect = data[data.alarm == 1].index[0]
                    app = data['ASVM'].idxmax()
                    frame = app-detect
                    freams.append(frame)
                    time = (app-detect)/50
                    leadtime.append(time)

                else:
                    pass

            except:
                pass

name_alarm_df = pd.DataFrame(name_alarm, columns=['File Names'])
leadtime_df = pd.DataFrame(leadtime, columns=['Leadtime'])
frame_df = pd.DataFrame(freams, columns=['Frames'])
merge2 = pd.concat([name_alarm_df, leadtime_df, frame_df], axis=1)

merge = pd.concat([merge1, merge2], axis =0)


path = r"C:/Users/abbey/Desktop/SW실험/V2 S13-S18/"
filename = "merge.csv"
merge.to_csv(path+filename, index=False)







'------------------------------------------------Leadtime----------------------------------------------------------'
import os
import pandas as pd

# 폴더 경로를 설정합니다
folder_path = r'C:/Users/abbey/Desktop/20명 데이터셋/20명 데이터셋(수정)/'
csv_path = r'C:/Users/abbey/Desktop/20명 데이터셋/leadtime2'  # 결과를 저장할 경로와 파일명

# 폴더 안의 파일명을 가져옵니다 (네번째 자리가 'F'인 파일만)
file_names = [f for f in os.listdir(folder_path) if len(f) > 3 and f[3] == 'F']

# 결과를 저장할 리스트 초기화
name_alarm = []
leadtime = []
freams = []
no_alarm_files = []



# 각 파일을 읽고 처리합니다
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']
    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

    # 'alarm' 열에 1이 존재하는지 확인합니다
    if (data.alarm == 1).any():
        detect = data[data.alarm == 1].index[0]
        max_frame = data['ASVM'].idxmax()
        frame = max_frame - detect
        freams.append(frame)
        time = frame / 50
        leadtime.append(time)
        name_alarm.append(file_name)

    else:
        no_alarm_files.append(file_name)

# DataFrame으로 변환합니다
name_alarm_df = pd.DataFrame(name_alarm, columns=['File Names'])
leadtime_df = pd.DataFrame(leadtime, columns=['Leadtime'])
frame_df = pd.DataFrame(freams, columns=['Frames'])

# 결과를 하나의 DataFrame으로 병합합니다
merge = pd.concat([name_alarm_df, leadtime_df, frame_df], axis=1)

# 결과를 CSV 파일로 저장합니다
merge.to_csv(csv_path + '.csv', index=False)

print(f"결과가 {csv_path}에 저장되었습니다.")
print(f"Alarm이 없는 파일 개수: {len(no_alarm_files)}")
print("Alarm이 없는 파일 목록:")
for file_name in no_alarm_files:
    print(file_name)










'------------------------------------------------SVM 그래프 그리기--------------------------------------------------------'

import os
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\abbey\Desktop\20명 데이터셋(정리중)\20명 데이터셋"

# 폴더 안의 파일명을 가져옵니다 (네번째 자리가 'F'인 파일만)
file_names = [f for f in os.listdir(path) if len(f) > 3 and f[3] == 'F']

# 결과를 저장할 리스트 초기화WEST 하반기 단기 일정
name_alarm = []
leadtime = []
freams = []
no_alarm_files = []


# 각 파일을 읽고 처리합니다
for file_name in file_names:
    file_path = os.path.join(path, file_name)
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']
    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

    # 파일명에서 .csv를 제거하여 이름만 추출
    name = file_name[:-4]  # 마지막 4글자 (.csv) 제거

    path_graph = r"C:/Users/abbey/Desktop/20명 데이터셋(정리중)/SVM 그래프/"

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.title(name)  # .csv를 제거한 이름으로 제목 설정
    plt.plot(data.ASVM)  # 그래프 그리기
    plt.ylabel('ASVM (g)')
    plt.xlabel('Frames')
    if (data.alarm == 1).any():
        plt.axvline(data.index[data.alarm == 1][0], label='Mid', color='red', linewidth=1.0)  # 첫 번째 1의 위치에 세로선 추가
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(data.GSVM)  # 그래프 그리기
    plt.ylabel('GSVM (g)')
    plt.xlabel('Frames')
    if (data.alarm == 1).any():
        plt.axvline(data.index[data.alarm == 1][0], label='Mid', color='red', linewidth=1.0)  # 첫 번째 1의 위치에 세로선 추가
    plt.tight_layout()

    plt.show()
    plt.savefig(path_graph + name + '.png')  # .csv를 제거한 이름으로 저장
    plt.close()













