import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import glob
import os

# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns
'---------------------------------------------------------------------------------------------------------------------'


# 윈도우 크기와 슬라이드 간격 정의
window_length = 120
overlap_ratio = 0.5
overlap = int(window_length * overlap_ratio)
# 결과를 저장할 리스트 초기화
windowed_data = []

'---------------------------------------------------------------------------------------------------------------------'


subject_list = np.setdiff1d(np.arange(1, 23), [2, 18])

np.random.seed(7) #
subject_list_test = np.random.choice(subject_list, 4, replace=False) #임의표본추출(모집단,선택할개수,중복허용여부)
list_test = list(subject_list_test)
list_train = [x for x in list(subject_list) if x not in list(subject_list_test)]


'---------------------------------------------------------------------------------------------------------------------'


train_data_list = []
impact_frame = []
Window = []
train_motion = []
train_m_type = []
train_name = []
train_class_ADL  = []
train_class_indicator=[]
#
e,m,t =1,10,1
movement = 'D'
count = 0

csv_files = []

for e in list_train:
    for movement in ['D','F']:
        for m in range(1, 12 if movement == 'F' else 15):
            for t in range(1, 2 if m == 1 else 4):
                try:
                    train_class = []
                    train_stack = pd.DataFrame()
                    train_merge = []

                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/abbey/Desktop/동작분류(10class)/"
                    name = "P%s%s%sR%s" % (e, movement, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)
                    name_a = name + "_a"
                    name_b = name + "_b"
                    name_a_csv = "%s%s.csv" % (path, name_a)
                    name_b_csv = "%s%s.csv" % (path, name_b)

                    if os.path.exists(name_b_csv):
                        name_result = name_b
                    else:
                        pass

                    '------------------------------------a,b 할때마다 변경------------------------------------'
                    # if os.path.exists(name_csv):
                    #     data = pd.read_csv(str(name_csv))
                    #     data = pd.DataFrame(data)
                    #     data = data.iloc[0:, 1:9]
                    #     data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']
                    #
                    # elif os.path.exists(name_a_csv):
                    #     data = pd.read_csv(str(name_a_csv))
                    #     data = pd.DataFrame(data)
                    #     data = data.iloc[0:, 1:9]
                    #     data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                    data = pd.read_csv(str(name_b_csv), header=1)
                    data = pd.DataFrame(data)
                    data = data.iloc[:, 0:8]
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                    '-----------------------------------------ADL--------------------------------------------'

                    num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                    data_unit = np.zeros((num_windows, window_length, 8))

                    for i in range(num_windows):
                        start = int(i * (window_length * (1 - overlap_ratio)))
                        end = start + window_length
                        data_unit[i] = data[start:end]

                        data_unit = np.array(data_unit)

                        df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                        df = pd.DataFrame(df)

                    '-----------------------------------------FALL-------------------------------------------'

                    if movement == 'F':
                        impact_frame = np.argmax(data.ASVM)
                        df = data.iloc[impact_frame - 60:impact_frame + 60, :]
                        df = np.array(df).reshape(1, -1)# 1열(한줄로 쌓음)
                        df = pd.DataFrame(df)

                    else:
                        pass

                    '-----------------------------------------merge-------------------------------------------'

                    if movement == 'D':
                        train_stack = pd.concat([train_stack, df], ignore_index=True)  # train_stack에 데이터 추가
                    else:
                        train_stack = df

                    '-----------------------------------------class------------------------------------------'
                    # class 2,3 / 5,6 a,b 할때마다 바꾸기

                    train_name.append(name)
                    train_motion.append(movement+m)

                    if movement =='D':
                        train_m_type = []
                        train_m_type.append(0)

                        if movement+str(m).zfill(2) == 'D01':
                            train_class.append(1)

                        # elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                        #     train_class.append(2)

                        elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                            train_class.append(3)

                        elif movement+str(m).zfill(2) in ['D05', 'D06','D07']:
                            train_class.append(4)

                        elif movement + str(m).zfill(2) in ['D08', 'D09']:
                            train_class.append(5)

                        # elif movement+str(m).zfill(2) in ['D10']:
                        #     train_class.append(6)

                        elif movement+str(m).zfill(2) in ['D10']:
                            train_class.append(7)

                        train_class_df = pd.DataFrame(train_class, columns=['class'])
                        train_class_indicator = pd.concat([train_class_df] * num_windows, ignore_index=True)

                        train_file = pd.DataFrame({'window': [f"{name_result}_{i}" for i in range(1, num_windows + 1)]})

                        # train_fall_indicator = []
                        # train_fall_indicator = pd.DataFrame(train_m_type, columns=['FD'])
                        # train_fall_indicator = pd.concat([train_fall_indicator] * num_windows).reset_index(drop=True)

                    else:
                        train_m_type = []
                        train_m_type.append(1)

                        if movement + str(m).zfill(2) in ['F02', 'F03', 'F05', 'F08', 'F09']:
                            train_class.append(8)

                        elif movement + str(m).zfill(2) in ['F01', 'F04', 'F07', 'F11']:
                            train_class.append(9)

                        elif movement + str(m).zfill(2) in ['F06', 'F10']:
                            train_class.append(10)

                        train_class_df = pd.DataFrame(train_class, columns=['class'])

                        train_class_indicator = train_class_df

                        train_file = pd.DataFrame({'window': [name_result]})

                        # train_fall_indicator = []
                        # train_fall_indicator = pd.DataFrame(train_m_type, columns=['FD'])


                    '---------------------------------------P01D01R01----------------------------------------------'

                    if name == 'P01D01R01':
                        train_class = []
                        train_stack = pd.DataFrame()
                        train_merge = []
                        num_windows = 0
                        num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                        num_windows = num_windows - 2

                        data_unit = np.zeros((num_windows, window_length, 8))

                        for i in range(num_windows):
                            start = int(i * (window_length * (1 - overlap_ratio)))
                            end = start + window_length
                            data_unit[i] = data[start:end]

                            data_unit = np.array(data_unit)

                            df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                            df = pd.DataFrame(df)

                        if movement == 'D':
                            train_stack = pd.concat([train_stack, df], ignore_index=True)  # train_stack에 데이터 추가
                        else:
                            train_stack = df

                        train_name.append(name)
                        train_motion.append(movement + m)

                        train_m_type = []
                        train_m_type.append(0)

                        if movement + str(m).zfill(2) == 'D01':
                            train_class.append(1)

                        train_class_df = pd.DataFrame(train_class, columns=['class'])
                        train_class_indicator = pd.concat([train_class_df] * num_windows, ignore_index=True)

                        train_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})

                    '-------------------------------------------------------------------------------------------'
                    if name == 'P06D01R01':
                        train_class = []
                        train_stack = pd.DataFrame()
                        train_merge = []
                        num_windows = 0
                        num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                        num_windows = num_windows - 2

                        data_unit = np.zeros((num_windows, window_length, 8))

                        for i in range(num_windows):
                            start = int(i * (window_length * (1 - overlap_ratio)))
                            end = start + window_length
                            data_unit[i] = data[start:end]

                            data_unit = np.array(data_unit)

                            df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                            df = pd.DataFrame(df)

                        if movement == 'D':
                            train_stack = pd.concat([train_stack, df], ignore_index=True)  # train_stack에 데이터 추가
                        else:
                            train_stack = df

                        train_name.append(name)
                        train_motion.append(movement + m)

                        train_m_type = []
                        train_m_type.append(0)

                        if movement + str(m).zfill(2) == 'D01':
                            train_class.append(1)

                        train_class_df = pd.DataFrame(train_class, columns=['class'])
                        train_class_indicator = pd.concat([train_class_df] * num_windows, ignore_index=True)

                        train_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})


                    if name == 'P17D01R01':
                        train_class = []
                        train_stack = pd.DataFrame()
                        train_merge = []
                        num_windows = 0
                        num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                        num_windows = num_windows - 2

                        data_unit = np.zeros((num_windows, window_length, 8))

                        for i in range(num_windows):
                            start = int(i * (window_length * (1 - overlap_ratio)))
                            end = start + window_length
                            data_unit[i] = data[start:end]

                            data_unit = np.array(data_unit)

                            df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                            df = pd.DataFrame(df)

                        if movement == 'D':
                            train_stack = pd.concat([train_stack, df], ignore_index=True)  # train_stack에 데이터 추가
                        else:
                            train_stack = df

                        train_name.append(name)
                        train_motion.append(movement + m)

                        train_m_type = []
                        train_m_type.append(0)

                        if movement + str(m).zfill(2) == 'D01':
                            train_class.append(1)

                        train_class_df = pd.DataFrame(train_class, columns=['class'])
                        train_class_indicator = pd.concat([train_class_df] * num_windows, ignore_index=True)

                        train_file = pd.DataFrame({'window': [f"{name}_{i}" for i in range(1, num_windows + 1)]})

                    '-------------------------------------------------------------------------------------------'

                    train_merge = pd.concat([train_stack, train_class_indicator, train_file], axis=1, sort=False)
                    train_data_list.append(train_merge)


                except:
                   pass
                count += 1

train_data = pd.concat(train_data_list, axis=0, ignore_index=True)
train_data_list = [pd.DataFrame(data) for data in train_data_list]
train_data2 = pd.concat([train_data] + train_data_list, axis=0, ignore_index=True)

train_data2.to_csv('C:/Users/abbey/Desktop/동작분류(10class)/train_data_10class.csv')







'------------------------------------------------------test---------------------------------------------------------'


import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import glob
import os


# 윈도우 크기와 슬라이드 간격 정의
window_length = 120
overlap_ratio = 0.5
overlap = int(window_length * overlap_ratio)
# 결과를 저장할 리스트 초기화
windowed_data = []


subject_list = np.setdiff1d(np.arange(1, 23), [2, 18])

np.random.seed(7) #
subject_list_test = np.random.choice(subject_list, 4, replace=False) #임의표본추출(모집단,선택할개수,중복허용여부)
list_test = list(subject_list_test)
list_train = [x for x in list(subject_list) if x not in list(subject_list_test)]


test_data_list = []
impact_frame = []
Window = []
test_motion = []
test_m_type = []
test_name = []
test_class_ADL  = []
test_class_indicator=[]

e,m,t = 3,1,1
movement = 'D'
count = 0

for e in list_test:
    for movement in ['D','F']:
        for m in range(1, 12 if movement == 'F' else 15):
            for t in range(1, 2 if m == 1 else 4):
                try:
                    test_class = []
                    test_stack = pd.DataFrame()
                    test_merge = []

                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/abbey/Desktop/동작분류(10class)/"
                    name = "P%s%s%sR%s" % (e, movement, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)
                    name_a = name + "_a"
                    name_b = name + "_b"
                    name_a_csv = "%s%s.csv" % (path, name_a)
                    name_b_csv = "%s%s.csv" % (path, name_b)

                    if os.path.exists(name_b_csv):
                        name_result = name_b
                    else:
                        pass


                    '------------------------------------a,b 할때마다 변경------------------------------------'
                    # if os.path.exists(name_csv):
                    #     data = pd.read_csv(str(name_csv))
                    #     data = pd.DataFrame(data)
                    #     data = data.iloc[0:, 1:9]
                    #     data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']
                    #
                    # elif os.path.exists(name_a_csv):
                    #     data = pd.read_csv(str(name_a_csv))
                    #     data = pd.DataFrame(data)
                    #     data = data.iloc[0:, 1:9]
                    #     data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                    data = pd.read_csv(str(name_b_csv), header=1)
                    data = pd.DataFrame(data)
                    data = data.iloc[:, 0:8]
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']



                    '-----------------------------------------ADL--------------------------------------------'
                    num_windows = int(np.ceil((len(data) - window_length) / (window_length * (1 - overlap_ratio))))

                    data_unit = np.zeros((num_windows, window_length, 8))

                    for i in range(num_windows):
                        start = int(i * (window_length * (1 - overlap_ratio)))
                        end = start + window_length
                        data_unit[i] = data[start:end]

                        data_unit = np.array(data_unit)

                        df = data_unit.reshape(num_windows, -1)  # 1열(한줄로 쌓음)
                        df = pd.DataFrame(df)

                    '-----------------------------------------FALL-------------------------------------------'

                    if movement == 'F':
                        impact_frame = np.argmax(data.ASVM)
                        df = data.iloc[impact_frame - 60:impact_frame + 60, :]
                        df = np.array(df).reshape(1, -1)# 1열(한줄로 쌓음)
                        df = pd.DataFrame(df)

                    else:
                        pass

                    '-----------------------------------------merge-------------------------------------------'

                    if movement == 'D':
                        test_stack = pd.concat([test_stack, df], ignore_index=True)  # train_stack에 데이터 추가
                    else:
                        test_stack = df

                    '-----------------------------------------class------------------------------------------'

                    # class 2,3 / 5,6 a,b 할때마다 바꾸기
                    test_name.append(name)
                    test_motion.append(movement+m)

                    if movement =='D':
                        test_m_type = []
                        test_m_type.append(0)

                        if movement+str(m).zfill(2) == 'D01':
                            test_class.append(1)

                        # elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                        #     test_class.append(2)

                        elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                            test_class.append(3)

                        elif movement+str(m).zfill(2) in ['D05', 'D06','D07']:
                            test_class.append(4)

                        elif movement + str(m).zfill(2) in ['D08', 'D09']:
                            test_class.append(5)

                        # elif movement+str(m).zfill(2) in ['D10']:
                        #     test_class.append(6)

                        elif movement+str(m).zfill(2) in ['D10']:
                            test_class.append(7)

                        test_class_df = pd.DataFrame(test_class, columns=['class'])
                        test_class_indicator = pd.concat([test_class_df] * num_windows, ignore_index=True)

                        test_file = pd.DataFrame({'window': [f"{name_result}_{i}" for i in range(1, num_windows + 1)]})

                    else:
                        test_m_type = []
                        test_m_type.append(1)

                        if movement + str(m).zfill(2) in ['F02', 'F03', 'F05', 'F08', 'F09']:
                            test_class.append(8)

                        elif movement + str(m).zfill(2) in ['F01', 'F04', 'F07', 'F11']:
                            test_class.append(9)

                        elif movement + str(m).zfill(2) in ['F06', 'F10']:
                            test_class.append(10)

                        test_class_df = pd.DataFrame(test_class, columns=['class'])

                        test_class_indicator = test_class_df

                        test_file = pd.DataFrame({'window': [name_result]})

                    '-------------------------------------------------------------------------------------------'

                    test_merge = pd.concat([test_stack, test_class_indicator, test_file], axis=1, sort=False)
                    test_data_list.append(test_merge)


                except:
                   pass
                count += 1


test_data = pd.concat(test_data_list, axis=0, ignore_index=True)
test_data_list = [pd.DataFrame(data) for data in test_data_list]
test_data2 = pd.concat([test_data] + test_data_list, axis=0, ignore_index=True)

test_data2.to_csv('C:/Users/abbey/Desktop/동작분류(10class)/test_data_10class.csv')



'--------------------------------------------------------------------------------------------------------------------'

#
# # #모델만들기위해
# # X_train = train_data.iloc[:, :480]
# # Y_train = train_data.FD
train_data.to_csv('C:/Users/abbey/Desktop/동작 분류/train_data_8축.csv')

# # X_test = test_data.iloc[:, :480]
# # Y_test = test_data.FD
test_data.to_csv('C:/Users/abbey/Desktop/동작 분류/test_data_8축.csv')
