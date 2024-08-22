import numpy as np
import pandas as pd
import os
import glob

Th_Acc = 0.9
Th_Gyro = 100
Th_Ang = 30
Grid_list = []

# 폴더 경로 설정
path = r'C:\Users\abbey\Desktop\20명 데이터셋\20명 데이터셋'

# 모든 CSV 파일의 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(path, '*.csv'))

# 각 파일을 순차적으로 처리
for Th_Acc in np.arange(0.6, 0.9, 0.1):
    for Th_Gyro in range(80, 111, 10):
        for Th_Ang in range(20, 31, 5):
            fs = 50  # 50Hz
            file_name = []
            Lead_Time = []
            FN_Error = []
            FP_Error = []
            TP_name = []

            count = 0
            TP, TN, FP, FN = 0, 0, 0, 0

            for file_path in file_paths:
                try:
                    name = os.path.basename(file_path)  # 파일명 저장
                    data = pd.read_csv(file_path).iloc[:, 2:8]

                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                    Ax = -data.Acc_Z
                    Ay = data.Acc_X
                    Az = -data.Acc_Y

                    Gx = -data.Gyr_Z
                    Gy = data.Gyr_X
                    Gz = -data.Gyr_Y

                    data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
                    data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                    data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                    # 가속도로 각도 연산
                    Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
                    Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

                    # 상보필터
                    Fs = 50
                    dt = 1 / Fs

                    Gx = np.array(Gx).flatten()
                    Roll_w = np.zeros(len(Gx))
                    for n in range(len(Gx) - 2):
                        Roll_w[n + 2] = Roll_w[n] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3

                    Gy = np.array(Gy).flatten()
                    Pitch_w = np.zeros(len(Gy))
                    for n in range(len(Gy) - 2):
                        Pitch_w[n + 2] = Pitch_w[n] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3

                    Roll_w = Roll_w.flatten()
                    Pitch_w = Pitch_w.flatten()

                    a = 0.2
                    Roll_a = Roll * a + Roll_w * (1 - a)
                    Pitch_a = Pitch * a + Pitch_w * (1 - a)

                    # 검출 알고리즘
                    data['Roll'] = Roll_a
                    data['Pitch'] = Pitch_a

                    Roll = abs(data.Roll)
                    Pitch = abs(data.Pitch)
                    impact = np.argmax(data.ASVM)

                    detection_frame = 0
                    lead_time = None  # 초기화

                    for tmp in range(len(data)):
                        if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                            fall_flag = tmp
                            for tmp_2 in range(tmp, tmp + 6):
                                if tmp_2 < len(data) and data.GSVM[tmp_2] >= Th_Gyro:
                                    fall_flag_2 = tmp_2
                                    for tmp_3 in range(tmp_2, tmp_2 + 6):
                                        if tmp_3 < len(data) and (Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang):
                                            detection_frame = tmp_3
                                            break

                    if detection_frame > 0:
                        lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms

                    if lead_time is not None and lead_time > 0 and lead_time <= 2000:
                        if 'D' in name:
                            FP += 1
                            FP_Error.append(name)
                        elif 'F' in name:
                            TP += 1
                            TP_name.append(name)
                            Lead_Time.append(lead_time)
                    else:
                        if 'D' in name:
                            TN += 1
                        elif 'F' in name:
                            FN += 1
                            FN_Error.append(name)

                    count += 1
                    file_name.append(name)

                except Exception as e:
                    print(f"Error processing file {name}: {e}")
                    pass

            Accuracy = round(((TP) + (TN)) / ((TP) + (FP) + (FN) + (TN)) * 100, 2)
            Sensitivity = round(TP / ((TP) + (FN)) * 100, 2)
            Specificity = round((TN) / ((TN) + (FP)) * 100, 2)
            Lead_Time_mean = round(np.mean(Lead_Time), 2) if Lead_Time else 0
            Lead_Time_std = round(np.std(Lead_Time), 2) if Lead_Time else 0
            print('\nAccuracy  Sensitivity  Specificity  LeadTime ')
            print('%s     %s        %s        %s ± %s     Thresholds:  %s, %i, %i' % (
                round(Accuracy, 2), round(Sensitivity, 2), round(Specificity, 2), Lead_Time_mean, Lead_Time_std, Th_Acc,
                Th_Gyro, Th_Ang))
            Grid = [Th_Acc, Th_Gyro, Th_Ang, Accuracy, Sensitivity, Specificity, Lead_Time_mean, Lead_Time_std]

            Grid_list.append(Grid)

Grid_list_df = pd.DataFrame(Grid_list)
Grid_list_df.columns = ['Th_Acc', 'Th_Gyro', 'Th_Ang', 'Accuracy', 'Sensitivity', 'Specificity', 'Lead_Time_mean',
                        'Lead_Time_std']

accuracy_max = np.max(Grid_list_df.Accuracy)
np.max(Grid_list_df.Sensitivity)
np.argmax(Grid_list_df.Accuracy)
np.argmax(Grid_list_df.Sensitivity)

print(Grid_list_df)

# lead_time_df = pd.DataFrame(Lead_Time)
# lead_time_df.to_csv(path + '\leadtime2.csv', index=False)





'--------------------------------------------------leadtime 50 이하------------------------------------------------------'

import numpy as np
import pandas as pd
import os
import glob

Th_Acc = 0.8
Th_Gyro = 100
Th_Ang = 20
Grid_list = []
except_filename = []  # 리드타임이 50ms 이하인 낙상 동작의 파일명을 저장하는 리스트

# 폴더 경로 설정
path = r'C:\Users\abbey\Desktop\20명 데이터셋\20명 데이터셋'

# 모든 CSV 파일의 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(path, '*.csv'))

# 각 파일을 순차적으로 처리
for Th_Acc in np.arange(0.6, 0.9, 0.1):
    for Th_Gyro in range(80, 111, 10):
        for Th_Ang in range(20, 31, 5):
            fs = 50  # 50Hz
            file_name = []
            Lead_Time = []
            FN_Error = []
            FP_Error = []
            TP_name = []

            count = 0
            TP, TN, FP, FN = 0, 0, 0, 0

            for file_path in file_paths:
                try:
                    name = os.path.basename(file_path)  # 파일명 저장
                    data = pd.read_csv(file_path).iloc[:, 2:8]

                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                    Ax = -data.Acc_Z
                    Ay = data.Acc_X
                    Az = -data.Acc_Y

                    Gx = -data.Gyr_Z
                    Gy = data.Gyr_X
                    Gz = -data.Gyr_Y

                    data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
                    data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                    data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                    # 가속도로 각도 연산
                    Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
                    Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

                    # 상보필터
                    Fs = 50
                    dt = 1 / Fs

                    Gx = np.array(Gx).flatten()
                    Roll_w = np.zeros(len(Gx))
                    for n in range(len(Gx) - 2):
                        Roll_w[n + 2] = Roll_w[n] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3

                    Gy = np.array(Gy).flatten()
                    Pitch_w = np.zeros(len(Gy))
                    for n in range(len(Gy) - 2):
                        Pitch_w[n + 2] = Pitch_w[n] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3

                    Roll_w = Roll_w.flatten()
                    Pitch_w = Pitch_w.flatten()

                    a = 0.2
                    Roll_a = Roll * a + Roll_w * (1 - a)
                    Pitch_a = Pitch * a + Pitch_w * (1 - a)

                    # 검출 알고리즘
                    data['Roll'] = Roll_a
                    data['Pitch'] = Pitch_a

                    Roll = abs(data.Roll)
                    Pitch = abs(data.Pitch)
                    impact = np.argmax(data.ASVM)

                    detection_frame = 0
                    lead_time = None  # 초기화

                    for tmp in range(len(data)):
                        if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                            fall_flag = tmp
                            for tmp_2 in range(tmp, tmp + 6):
                                if tmp_2 < len(data) and data.GSVM[tmp_2] >= Th_Gyro:
                                    fall_flag_2 = tmp_2
                                    for tmp_3 in range(tmp_2, tmp_2 + 6):
                                        if tmp_3 < len(data) and (Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang):
                                            detection_frame = tmp_3
                                            break

                    if detection_frame > 0:
                        lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms

                    if lead_time is not None and lead_time > 0 and lead_time <= 2000:
                        if lead_time <= 50:  # 리드타임이 50ms 이하인 경우
                            FN += 1
                            FN_Error.append(name)
                        else:
                            if 'D' in name:
                                FP += 1
                                FP_Error.append(name)
                            elif 'F' in name:
                                TP += 1
                                TP_name.append(name)
                                Lead_Time.append(lead_time)
                    else:
                        if 'D' in name:
                            TN += 1
                        elif 'F' in name:
                            FN += 1
                            FN_Error.append(name)

                    count += 1
                    file_name.append(name)

                except Exception as e:
                    print(f"Error processing file {name}: {e}")
                    pass

            Accuracy = round(((TP) + (TN)) / ((TP) + (FP) + (FN) + (TN)) * 100, 2)
            Sensitivity = round(TP / ((TP) + (FN)) * 100, 2)
            Specificity = round((TN) / ((TN) + (FP)) * 100, 2)
            Lead_Time_mean = round(np.mean(Lead_Time), 2) if Lead_Time else 0
            Lead_Time_std = round(np.std(Lead_Time), 2) if Lead_Time else 0
            print('\nAccuracy  Sensitivity  Specificity  LeadTime ')
            print('%s     %s        %s        %s ± %s     Thresholds:  %s, %i, %i' % (
                round(Accuracy, 2), round(Sensitivity, 2), round(Specificity, 2), Lead_Time_mean, Lead_Time_std, Th_Acc,
                Th_Gyro, Th_Ang))
            Grid = [Th_Acc, Th_Gyro, Th_Ang, Accuracy, Sensitivity, Specificity, Lead_Time_mean, Lead_Time_std]

            Grid_list.append(Grid)

Grid_list_df = pd.DataFrame(Grid_list)
Grid_list_df.columns = ['Th_Acc', 'Th_Gyro', 'Th_Ang', 'Accuracy', 'Sensitivity', 'Specificity', 'Lead_Time_mean',
                        'Lead_Time_std']

accuracy_max = np.max(Grid_list_df.Accuracy)
np.max(Grid_list_df.Sensitivity)
np.argmax(Grid_list_df.Accuracy)
np.argmax(Grid_list_df.Sensitivity)

print(Grid_list_df)

lead_time_df = pd.DataFrame(Lead_Time)
lead_time_df.to_csv(path + '\leadtime2.csv', index=False)



# 리드타임이 50ms 이하인 낙상 동작의 파일명을 출력
print(f"Files with fall detection but lead time <= 50ms: {except_filename}")





'--------------------------------------------------90,100,30-----------------------------------------------------'

import numpy as np
import pandas as pd
import os
import glob

# 임계값 설정
Th_Acc = 0.9
Th_Gyro = 100
Th_Ang = 30

# 검출되지 않은 Fall 데이터 파일명을 저장할 리스트
not_detection = []

# 폴더 경로 설정
path = r'C:\Users\abbey\Desktop\20명 데이터셋\20명 데이터셋'

# 모든 CSV 파일의 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(path, '*.csv'))

# 각 파일을 순차적으로 처리
fs = 50  # 50Hz
file_name = []
Lead_Time = []
FN_Error = []
FP_Error = []
TP_name = []

count = 0
TP, TN, FP, FN = 0, 0, 0, 0

for file_path in file_paths:
    try:
        name = os.path.basename(file_path)  # 파일명 저장
        data = pd.read_csv(file_path).iloc[:, 2:8]

        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
        data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

        Ax = -data.Acc_Z
        Ay = data.Acc_X
        Az = -data.Acc_Y

        Gx = -data.Gyr_Z
        Gy = data.Gyr_X
        Gz = -data.Gyr_Y

        data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
        data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        data1['ASVM'] = (data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
        data1['GSVM'] = (data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

        # 가속도로 각도 연산
        Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
        Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

        # 상보필터
        Fs = 50
        dt = 1 / Fs

        Gx = np.array(Gx).flatten()
        Roll_w = np.zeros(len(Gx))
        for n in range(len(Gx) - 2):
            Roll_w[n + 2] = Roll_w[n] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3

        Gy = np.array(Gy).flatten()
        Pitch_w = np.zeros(len(Gy))
        for n in range(len(Gy) - 2):
            Pitch_w[n + 2] = Pitch_w[n] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3

        Roll_w = Roll_w.flatten()
        Pitch_w = Pitch_w.flatten()

        a = 0.2
        Roll_a = Roll * a + Roll_w * (1 - a)
        Pitch_a = Pitch * a + Pitch_w * (1 - a)

        # 검출 알고리즘
        data['Roll'] = Roll_a
        data['Pitch'] = Pitch_a

        Roll = abs(data.Roll)
        Pitch = abs(data.Pitch)
        impact = np.argmax(data.ASVM)

        detection_frame = 0
        lead_time = None  # 초기화

        for tmp in range(len(data)):
            if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                fall_flag = tmp
                for tmp_2 in range(tmp, tmp + 6):
                    if tmp_2 < len(data) and data.GSVM[tmp_2] >= Th_Gyro:
                        fall_flag_2 = tmp_2
                        for tmp_3 in range(tmp_2, tmp_2 + 6):
                            if tmp_3 < len(data) and (Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang):
                                detection_frame = tmp_3
                                break

        if detection_frame > 0:
            lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms

        if lead_time is not None and lead_time > 0 and lead_time <= 2000:
            if 'D' in name:
                FP += 1
                FP_Error.append(name)
            elif 'F' in name:
                TP += 1
                TP_name.append(name)
                Lead_Time.append(lead_time)
        else:
            if 'D' in name:
                TN += 1
            elif 'F' in name:
                FN += 1
                FN_Error.append(name)
                not_detection.append(name)  # 검출되지 않은 Fall 데이터 파일명을 저장

        count += 1
        file_name.append(name)

    except Exception as e:
        print(f"Error processing file {name}: {e}")
        pass

# 검출되지 않은 Fall 데이터 파일명을 데이터프레임으로 저장
not_detection_df = pd.DataFrame(not_detection, columns=['Not Detected Files'])

# 검출되지 않은 파일명 출력
print(not_detection_df)




import matplotlib.pyplot as plt
import os

# 그래프를 저장할 폴더 경로 설정
output_path = r'C:\Users\abbey\Desktop\오검출그래프'

# ASVM, GSVM이 임계값과 만나는 순간의 프레임을 저장할 리스트
threshold_frames = []

# 검출되지 않은 Fall 데이터 파일에 대해 그래프를 그림
for file_name in not_detection:
    try:
        # CSV 파일 불러오기
        file_path = os.path.join(path, file_name)
        data = pd.read_csv(file_path).iloc[:, 2:8]

        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
        data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

        # 가속도로 각도 연산
        Ax = -data.Acc_Z
        Ay = data.Acc_X
        Az = -data.Acc_Y

        Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
        Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

        # ASVM 임계값과 만나는 순간의 프레임을 찾음
        asvm_threshold_frame = (data['ASVM'] <= Th_Acc).idxmax()

        # GSVM 임계값과 만나는 순간의 프레임을 찾음
        gsvm_threshold_frame = (data['GSVM'] >= Th_Gyro).idxmax()

        # 결과를 리스트에 저장
        threshold_frames.append({
            'File Name': file_name,
            'ASVM Threshold Frame': asvm_threshold_frame,
            'GSVM Threshold Frame': gsvm_threshold_frame
        })

        # 그래프 그리기
        plt.figure(figsize=(12, 8))

        # 첫 번째 그래프: ASVM
        plt.subplot(3, 1, 1)
        plt.plot(data['ASVM'], label='ASVM')
        plt.axhline(y=Th_Acc, color='r', linestyle='--', label=f'Threshold = {Th_Acc}')
        plt.title(f'{file_name}')
        plt.legend()

        # 두 번째 그래프: GSVM
        plt.subplot(3, 1, 2)
        plt.plot(data['GSVM'], label='GSVM')
        plt.axhline(y=Th_Gyro, color='r', linestyle='--', label=f'Threshold = {Th_Gyro}')
        plt.legend()

        # 세 번째 그래프: Roll and Pitch
        plt.subplot(3, 1, 3)
        plt.plot(abs(Roll), label='Roll')
        plt.plot(abs(Pitch), label='Pitch')
        plt.axhline(y=Th_Ang, color='r', linestyle='--', label=f'Threshold = {Th_Ang}')
        plt.legend()

        # 그래프를 파일로 저장
        save_path = os.path.join(output_path, f'{os.path.splitext(file_name)[0]}.png')
        plt.savefig(save_path)
        plt.close()

        print(f'Saved plot: {save_path}')

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# 결과를 데이터프레임으로 변환
threshold_frames_df = pd.DataFrame(threshold_frames)

# 결과 데이터프레임 출력
print(threshold_frames_df)
