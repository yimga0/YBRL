import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os

# 설정
Th_Acc = 0.9
Th_Gyro = 100
Th_Ang = 30
fs = 50  # 50Hz 샘플링 속도
data_path = r'C:/Users/abbey/Desktop/20명 데이터/'

# 파일 검색 및 필터링
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f[3] == 'F']
print(f"Filtered CSV files: {csv_files}")

# 결과 저장 변수 초기화
Lead_Time = []
TP_name = []
FP_Error = []
FN_Error = []
Grid_list = []

# 파일 처리
for file in csv_files:
    try:
        file_path = os.path.join(data_path, file)
        data = pd.read_csv(file_path).iloc[:, 2:8]
        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

        # 데이터 전처리
        data['ASVM'] = np.sqrt(data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2)
        data['GSVM'] = np.sqrt(data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2)

        Ax = -data.Acc_Z
        Ay = data.Acc_X
        Az = -data.Acc_Y
        Gx = -data.Gyr_Z
        Gy = data.Gyr_X
        Gz = -data.Gyr_Y

        data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
        data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        data1['ASVM'] = np.sqrt(data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2)
        data1['GSVM'] = np.sqrt(data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2)

        Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
        Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

        dt = 1 / fs
        Gx = np.array(Gx).flatten()
        Roll_w = np.zeros(len(Gx))
        for n in range(len(Gx) - 2):
            Roll_w[n + 2] = Roll_w[n] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3

        Gy = np.array(Gy).flatten()
        Pitch_w = np.zeros(len(Gy))
        for n in range(len(Gy) - 2):
            Pitch_w[n + 2] = Pitch_w[n] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3

        Roll_a = Roll * 0.2 + Roll_w * 0.8
        Pitch_a = Pitch * 0.2 + Pitch_w * 0.8

        data['Roll'] = Roll_a
        data['Pitch'] = Pitch_a

        Roll = abs(data.Roll)
        Pitch = abs(data.Pitch)
        impact = np.argmax(data.ASVM)

        # 검출 알고리즘
        detection_frame = 0
        for tmp in range(len(data)):
            if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                for tmp_2 in range(tmp, tmp + 6):
                    if data.GSVM[tmp_2] >= Th_Gyro:
                        for tmp_3 in range(tmp_2, tmp_2 + 6):
                            if Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang:
                                detection_frame = tmp_3
                                break

        if detection_frame > 0:
            lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms
            if lead_time > 0 and lead_time <= 2000:
                TP_name.append(file)
                Lead_Time.append(lead_time)
            else:
                FP_Error.append(file)
        else:
            FN_Error.append(file)


        '-----------------------------------------------------------------------------------------------'
        #그래프그리기
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 8))
        #
        # # ASVM 그래프
        # plt.subplot(3, 1, 1)
        # plt.plot(data['ASVM'], label='ASVM')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.title(f'{file}')
        # plt.xlabel('Frames')
        # plt.ylabel('ASVM')
        # plt.legend()
        #
        # # GSVM 그래프
        # plt.subplot(3, 1, 2)
        # plt.plot(data['GSVM'], label='GSVM')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.xlabel('Frames')
        # plt.ylabel('GSVM')
        # plt.legend()
        #
        # # Roll, Pitch 그래프
        # plt.subplot(3, 1, 3)
        # plt.plot(data['Roll'], label='Roll')
        # plt.plot(data['Pitch'], label='Pitch')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.xlabel('Frames')
        # plt.ylabel('Angle (degrees)')
        # plt.legend()
        #
        # plt.tight_layout()
        #
        # graph_path = "C:/Users/abbey/Desktop/20명 데이터/그래프/"
        # graph_file_path = os.path.join(graph_path, f"{os.path.splitext(file)[0]}.png")
        # plt.savefig(graph_file_path)
        # plt.close()
        '-----------------------------------------------------------------------------------------------'



    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 결과 출력
print(f"True Positives (TP): {TP_name}")
print(f"False Positives (FP): {FP_Error}")
print(f"False Negatives (FN): {FN_Error}")
print(f"Lead Times: {Lead_Time}")
print(f"Average Lead Time: {np.mean(Lead_Time) if Lead_Time else 'N/A'} ms")



# 리드타임 데이터 프레임 생성
lead_time_df = pd.DataFrame({'File Name': TP_name, 'Lead Time (ms)': Lead_Time})

# CSV 파일로 저장
output_path = r'C:\Users\abbey\Desktop\20명 데이터\문서\lead_times3.csv'
lead_time_df.to_csv(output_path, index=False)

print(f"Lead times saved to {output_path}")



'------------------------------------------------------------------------------------------------------------'




import matplotlib.pyplot as plt
# 그래프 그리기
plt.figure(figsize=(12, 8))

# ASVM 그래프
plt.subplot(3, 1, 1)
plt.plot(data['ASVM'], label='ASVM')
plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
if detection_frame > 0:
    plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
plt.title('ASVM')
plt.xlabel('Frames')
plt.ylabel('ASVM')
plt.legend()

# GSVM 그래프
plt.subplot(3, 1, 2)
plt.plot(data['GSVM'], label='GSVM')
plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
if detection_frame > 0:
    plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
plt.title('GSVM')
plt.xlabel('Frames')
plt.ylabel('GSVM')
plt.legend()

# Roll, Pitch 그래프
plt.subplot(3, 1, 3)
plt.plot(data['Roll'], label='Roll')
plt.plot(data['Pitch'], label='Pitch')
plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
if detection_frame > 0:
    plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
plt.title('Roll and Pitch')
plt.xlabel('Frames')
plt.ylabel('Angle (degrees)')
plt.legend()

plt.tight_layout()
plt.show()





'------------------------------------------------------------------------------------------------------------'




import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 설정
Th_Acc = 0.9
Th_Gyro = 100
Th_Ang = 30
fs = 50  # 50Hz 샘플링 속도
data_path = r'C:\Users\abbey\Desktop\20명 데이터셋\20명 데이터셋'

# 파일 검색 및 필터링
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f[3] == 'F']
print(f"Filtered CSV files: {csv_files}")

# 결과 저장 변수 초기화
Lead_Time = []
TP_name = []
FP_Error = []
FN_Error = []

# 그래프 저장 경로 설정
graph_path = r'C:/Users/abbey/Desktop/20명 데이터셋/그래프(90,100,30)/'
os.makedirs(graph_path, exist_ok=True)

# 파일 처리
for file in csv_files:
    try:
        file_path = os.path.join(data_path, file)
        data = pd.read_csv(file_path).iloc[:, 2:8]
        data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

        # 데이터 전처리
        data['ASVM'] = np.sqrt(data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2)
        data['GSVM'] = np.sqrt(data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2)

        Ax = -data.Acc_Z
        Ay = data.Acc_X
        Az = -data.Acc_Y
        Gx = -data.Gyr_Z
        Gy = data.Gyr_X
        Gz = -data.Gyr_Y

        data1 = pd.concat([Ax, Ay, Az, Gx, Gy, Gz], axis=1)
        data1.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
        data1['ASVM'] = np.sqrt(data1.Acc_X ** 2 + data1.Acc_Y ** 2 + data1.Acc_Z ** 2)
        data1['GSVM'] = np.sqrt(data1.Gyr_X ** 2 + data1.Gyr_Y ** 2 + data1.Gyr_Z ** 2)

        Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi
        Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi

        dt = 1 / fs
        Gx = np.array(Gx).flatten()
        Roll_w = np.zeros(len(Gx))
        for n in range(len(Gx) - 2):
            Roll_w[n + 2] = Roll_w[n] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3

        Gy = np.array(Gy).flatten()
        Pitch_w = np.zeros(len(Gy))
        for n in range(len(Gy) - 2):
            Pitch_w[n + 2] = Pitch_w[n] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3

        Roll_a = Roll * 0.2 + Roll_w * 0.8
        Pitch_a = Pitch * 0.2 + Pitch_w * 0.8

        data['Roll'] = Roll_a
        data['Pitch'] = Pitch_a

        Roll = abs(data.Roll)
        Pitch = abs(data.Pitch)
        impact = np.argmax(data.ASVM)

        # 검출 알고리즘
        detection_frame = 0
        for tmp in range(len(data)):
            if data.ASVM[tmp] <= Th_Acc and detection_frame == 0:
                for tmp_2 in range(tmp, min(tmp + 6, len(data))):
                    if data.GSVM[tmp_2] >= Th_Gyro:
                        for tmp_3 in range(tmp_2, min(tmp_2 + 6, len(data))):
                            if Roll[tmp_3] >= Th_Ang or Pitch[tmp_3] >= Th_Ang:
                                detection_frame = tmp_3
                                break

        if detection_frame > 0:
            lead_time = round((impact - detection_frame) / fs * 1000, 3)  # ms
            if lead_time > 0 and lead_time <= 2000:
                TP_name.append(file)
                Lead_Time.append(lead_time)
            else:
                FP_Error.append(file)
        else:
            FN_Error.append(file)

        # 그래프 그리기
        # plt.figure(figsize=(12, 8))
        #
        # # ASVM 그래프
        # plt.subplot(3, 1, 1)
        # plt.plot(data['ASVM'], label='ASVM')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.title(f'{file}')
        # plt.xlabel('Frames')
        # plt.ylabel('ASVM')
        # plt.legend()
        #
        # # GSVM 그래프
        # plt.subplot(3, 1, 2)
        # plt.plot(data['GSVM'], label='GSVM')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.xlabel('Frames')
        # plt.ylabel('GSVM')
        # plt.legend()
        #
        # # Roll, Pitch 그래프
        # plt.subplot(3, 1, 3)
        # plt.plot(data['Roll'], label='Roll')
        # plt.plot(data['Pitch'], label='Pitch')
        # plt.axvline(x=impact, color='r', linestyle='--', label='Impact')
        # if detection_frame > 0:
        #     plt.axvline(x=detection_frame, color='g', linestyle='--', label='Detection')
        # plt.xlabel('Frames')
        # plt.ylabel('Angle (degrees)')
        # plt.legend()
        #
        # plt.tight_layout()
        #
        # # 그래프 저장
        # graph_file_path = os.path.join(graph_path, f"{os.path.splitext(file)[0]}.png")
        # plt.savefig(graph_file_path)
        # plt.close()  # plt.close() 호출을 통해 현재 그래프를 닫습니다.

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 결과 출력
print(f"True Positives (TP): {TP_name}")
print(f"False Positives (FP): {FP_Error}")
print(f"False Negatives (FN): {FN_Error}")
print(f"Lead Times: {Lead_Time}")
print(f"Average Lead Time: {np.mean(Lead_Time) if Lead_Time else 'N/A'} ms")

# 리드타임 데이터 프레임 생성
lead_time_df = pd.DataFrame({'File Name': TP_name, 'Lead Time (ms)': Lead_Time})

# # CSV 파일로 저장
# output_path = r'C:\Users\abbey\Desktop\20명 데이터\문서\lead_times3.csv'
# lead_time_df.to_csv(output_path, index=False)
#
# print(f"Lead times saved to {output_path}")
