import os
import pandas as pd
import numpy as np
import struct
import csv


def transform_text(input_text):
    # "SensorDataSWRaw_" 부분을 "TransRaw_"로 대체합니다.
    transformed_text = input_text.replace("SensorDataSWRaw_", "TransRaw_")
    return transformed_text


def convertBytesToUInt16(byte1, byte2):
    # 두 byte를 int8로 변환
    byte1 = np.int8(byte1)
    byte2 = np.int8(byte2)

    # 상위 바이트를 8비트 왼쪽으로 시프트하고 하위 바이트와 결합
    value = (np.int16(byte2) << 8) | (np.int16(byte1) & 0xFF)

    # 음수 처리를 위한 조건문 추가
    if value >= 0x8000:
        value -= 0x10000

    return np.int16(value)


def process_values(value1, value2):
    # value1과 value2를 16진수로 변환
    hex_value1 = format(value1 & 0xFF, '02X')
    hex_value2 = format(value2 & 0xFF, '02X')

    # 두 16진수 값을 합침
    combined_hex = hex_value2 + hex_value1

    # 상태 값을 확인 (앞자리 값이 8이면 Trigger를 on으로 표시)
    state = combined_hex[0]
    if state == '8':
        trigger_status = 1
    else:
        trigger_status = 0

    # 나머지 값을 숫자로 변환
    remaining_hex = combined_hex[1:]
    remaining_value = int(remaining_hex, 16)

    return trigger_status, remaining_value


def parseIMUDataSW(row):
    global results
    global save_filename

    data = np.array(row, dtype=np.int8)
    byte_data = data.tobytes()

    # 바이트 배열을 언패킹
    # buffer = struct.unpack('<' + 'B' * len(byte_data), byte_data)
    # bufferLength = len(data)
    # buffer = struct.unpack('<' + 'B'*len(data), data)
    bufferLength = len(data)
    A = data[2:4]
    startMarker = ''.join(map(chr, data[0:2]))
    Time = struct.unpack('<H', data[2:4])[0]

    # alarmInfo = seqNumber // 4096

    if startMarker == "SD":
        dataBlockNum = 0
        if bufferLength == 130:
            dataBlockNum = 9
        elif bufferLength == 144:
            dataBlockNum = 10
        else:
            return

        for i in range(dataBlockNum):
            alarmInfo, seq_num = process_values(data[4 + 14 * i], data[5 + 14 * i])

            accX = convertBytesToUInt16(data[6 + 14 * i], data[7 + 14 * i]) / 8192.0
            accY = convertBytesToUInt16(data[8 + 14 * i], data[9 + 14 * i]) / 8192.0
            accZ = convertBytesToUInt16(data[10 + 14 * i], data[11 + 14 * i]) / 8192.0
            gyroX = convertBytesToUInt16(data[12 + 14 * i], data[13 + 14 * i]) / 65.536
            gyroY = convertBytesToUInt16(data[14 + 14 * i], data[15 + 14 * i]) / 65.536
            gyroZ = convertBytesToUInt16(data[16 + 14 * i], data[17 + 14 * i]) / 65.536

            results.append({
                'Time': Time,
                'seq': seq_num,
                'accX': accX,
                'accY': accY,
                'accZ': accZ,
                'gyroX': gyroX,
                'gyroY': gyroY,
                'gyroZ': gyroZ,
                'alarmInfo': alarmInfo  # 0: Not detect / 1: detect
            })
    elif startMarker == "Q=":
        trigger_marker = ''.join(map(chr, data[2:3]))
        if trigger_marker == "2":
            if "T_" in save_filename:  # 낙상이 감지가 되더라도, Q=2 string이 들어오지 않으면 파일명 변경 안됨
                pass
            else:

                save_filename = "T_" + save_filename

            return True
    return False


#raw_path = r"D:\낙상\2024 낙상 (재발견)\2차\SWRaw_v2\SensorDataSWRaw_v2"  # RAW 파일 저장 경로
#save_path = r"D:\낙상\2024 낙상 (재발견)\2차\SWRaw_v2\전처리"  # 변환 파일 저장할 경로


raw_path = r'C:\Users\abbey\Desktop\20명 평가\v1 라벨링'  # RAW 파일 저장 경로
save_path = r'C:\Users\abbey\Desktop\20명 평가\12명'   # 변환 파일 저장할 경로



file_list = os.listdir(raw_path)

#f = 'S13F17R03_SW_v2.csv'
for f in file_list:

    results = []
    save_filename = transform_text(f)
    file_path = os.path.join(raw_path, f)

    # CSV 파일을 읽고 데이터와 최대 열 수를 확인
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        if "Data" in first_row:
            pass  # 첫 번째 줄이 문자열인 경우 건너뛰기

        for row in reader:
            parseIMUDataSW(row[1:])

    # 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)
    result_df = result_df.dropna()
    # DataFrame을 CSV 파일로 저장
    result_df.to_csv(os.path.join(save_path, save_filename), index=False)
