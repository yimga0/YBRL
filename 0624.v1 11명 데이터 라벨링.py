import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


e, m, t = 13,1,1
for e in range(13, 17):
    for m in range(1,22):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/SW실험/V2 S13-S15/"
                name = "T_S%sF%sR%s_SW_v2" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                # path_graph = r"C:/Users/abbey/Desktop/SW실험/V2 S13-S15/T_그래프/"
                # plt.figure(figsize=(8, 6))
                # plt.subplot(2, 1, 1)
                # plt.title(name)
                # plt.plot(data.ASVM)  # 그래프그리기
                # plt.ylabel('ASVM (g)')
                # plt.xlabel('Frames')
                # if (data.alarm == 1).any():
                #     plt.axvline(data.index[data.alarm == 1][0], label='Mid', color='red',linewidth=1.0)  # 첫 번째 1의 위치에 세로선 추가
                # plt.tight_layout()
                #
                # plt.subplot(2, 1, 2)
                # plt.plot(data.GSVM)  # 그래프그리기
                # plt.ylabel('GSVM (g)')
                # plt.xlabel('Frames')
                # if (data.alarm == 1).any():
                #     plt.axvline(data.index[data.alarm == 1][0], label='Mid', color='red',linewidth=1.0)  # 첫 번째 1의 위치에 세로선 추가
                # plt.tight_layout()
                #
                # plt.show()
                # plt.savefig(path_graph + name_png)
                # plt.close()

            except:
                pass



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





'----------------------------------------x,y,z 그래프 그리기---------------------------------------------------'


import pandas as pd
import matplotlib.pyplot as plt

#################### 피험자 번호 ####################
e, m, t = 4,9,3

e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = r"C:/Users/abbey/Desktop/SW실험/V1 11명 데이터/"
name = "T_S%sF%sR%s_SWRaw" % (e, m, t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv))
data = pd.DataFrame(data)
acc = data.iloc[:, 2:5]
gyr = data.iloc[:, 5:8]

plt.figure()
plt.subplot(2, 1, 1)
plt.title(name)
plt.plot(acc)
plt.xlabel('Frames')
plt.ylabel('Acceleration (g)')

plt.subplot(2, 1, 2)
plt.plot(gyr)
plt.xlabel('Frames')
plt.ylabel('Angular velocity (degree/s)')

plt.tight_layout()
plt.show()

# plt.savefig('C:/Users/abbey/Desktop/어플실험/S11/sw/그래프(sw)/%s' % a[i].replace('.csv', '.png'))
# plt.close()
#


'-----------------------------------------------T_ 개수(20명)---------------------------------------------------'
import os

# 폴더 경로를 설정합니다
folder_path = r'C:\Users\abbey\Desktop\20명 평가\12명\Q(x)'

# 폴더 안의 파일명을 가져옵니다
file_names = os.listdir(folder_path)

# 'D'와 'F'의 개수를 세는 변수
count_D = 0
count_F = 0

# 'D'와 'F'가 나오는 파일명을 저장할 리스트
files_D = []
files_F = []

# 파일명을 순회하면서 네 번째 자리를 확인합니다
for file_name in file_names:
    if len(file_name) > 3:  # 파일명이 충분히 긴지 확인합니다
        if file_name[3] == 'D':
            count_D += 1
            files_D.append(file_name)
        elif file_name[3] == 'F':
            count_F += 1
            files_F.append(file_name)

print(f"네 번째 자리에 'D'가 나오는 파일의 개수: {count_D}")
print("파일명 목록:", files_D)
print(f"네 번째 자리에 'F'가 나오는 파일의 개수: {count_F}")
print("파일명 목록:", files_F)




'-----------------------------------------------ALARM 개수(20명)---------------------------------------------------'
import os
import pandas as pd

# 폴더 경로를 설정합니다
folder_path = r'C:\Users\abbey\Desktop\20명 평가\12명\Q(x)'

# 폴더 안의 파일명을 가져옵니다
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 'alarm' 열의 값이 1인 파일을 저장할 리스트와 개수 세기
count = 0
name_alarm = []

# 파일명을 순회하면서 CSV 파일을 읽고 'alarm' 열의 값을 확인합니다
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    data = data.iloc[:, 2:9]  # 가속도, 각속도 3축 데이터만 불러오게 자름
    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'alarm']

    # 'alarm' 열에 1이 존재하는지 확인합니다
    if not (data['alarm'] == 1).any():
            count += 1
            name_alarm.append(file_name)


# 'D'와 'F'가 네 번째 자리에 오는 파일명을 저장할 리스트와 개수 세기
count_D = 0
count_F = 0
files_D = []
files_F = []

for file_name in name_alarm:
    if len(file_name) > 3:  # 파일명이 충분히 긴지 확인합니다
        if file_name[3] == 'D':
            count_D += 1
            files_D.append(file_name)
        elif file_name[3] == 'F':
            count_F += 1
            files_F.append(file_name)

# 결과 출력
print(f"'alarm' 열에 1이 있는 파일의 개수: {count}")
print("파일명 목록:", name_alarm)
print(f"네 번째 자리에 'D'가 나오는 파일의 개수: {count_D}")
print("파일명 목록:", files_D)
print(f"네 번째 자리에 'F'가 나오는 파일의 개수: {count_F}")
print("파일명 목록:", files_F)







'-----------------------------------------------12명 동작 지우기---------------------------------------------------'

import os

# 폴더 경로를 설정합니다
folder_path = r'C:\Users\abbey\Desktop\20명 평가\v1 라벨링'

# 폴더 안의 파일명을 가져옵니다
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# "D06", "D07", "D08", "D09"이 포함된 파일명을 저장할 리스트와 개수 세기
patterns = ["D06", "D07", "D08", "D09"]
files_to_delete = []

for file_name in file_names:
    if any(pattern in file_name for pattern in patterns):
        files_to_delete.append(file_name)

# 결과 출력
print(f"D06, D07, D08, D09 중 하나가 포함된 파일의 개수: {len(files_to_delete)}")
print("파일명 목록:", files_to_delete)

# 파일을 삭제합니다
for file_name in files_to_delete:
    file_path = os.path.join(folder_path, file_name)
    try:
        os.remove(file_path)
        print(f"삭제됨: {file_name}")
    except Exception as e:
        print(f"삭제 실패: {file_name}, 에러: {e}")
