import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns

subject_list = np.arange(1, 21)
np.random.seed(7) #
subject_list_test = np.random.choice(subject_list, 4, replace=False) #임의표본추출(모집단,선택할개수,중복허용여부)
list_test = list(subject_list_test)
list_train = [x for x in list(subject_list) if x not in list(subject_list_test)]

count = 0
train_target = []
train_name = []
train_sort = []
train_stack = []
train_target_noise = []
train_target_resample = []
train_target_scale = []
train_target_frame = []
Fall01 = [] #추락
Fall02 = [] #전도
NotFall = [] #일상동작
M = range(1,22)

train_motion = []
train_m_type = []

train_stack_FFH = []
train_stack_Fall = []
train_stack_NF = []
train_class = []

# e = 1 #피험자
# m = 1# 동작번호
# trial = 1 # 트라이얼
# movement = 'D'

# e = 1,m = 3,trial = 1, movement = 'D'
e , m, trial = 1, 1, 1
#impact = pd.read_csv('C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P01/ble_re/P01D01R01.csv')

for e in list_train:
    for movement in ['D', 'F']:
        for m in range(1, 15):
            for trial in range(1,4):
                try:

                    t = trial

                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/abbey/Desktop/낙상 실험 전처리/P%s/ble_re/" %(e)

                    name = "P%s%s%sR%s" % (e, movement, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)
                    #name = "V%s%s%sR%s.csv" % (path, e)


                    data = pd.read_csv(str(name_csv))
                    data = pd.DataFrame(data)
                    data = data.iloc[:, 1:7]
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']


                    data['Acc_X'] = data.Acc_X / 9.8
                    data['Acc_Y'] = data.Acc_Y / 9.8
                    data['Acc_Z'] = data.Acc_Z / 9.8
                    #
                    # plt.plot(data.Acc_Z)
                    #
                    # plt.show()
                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


                    # 충격 시점 앞 뒤 1초 자르기
                    if movement == 'F':
                        # 충격 시점 찾기 Fall
                       impact_frame = np.argmax(data.ASVM)
                       Window = data.iloc[impact_frame - 30:impact_frame + 30, :]

                    else:  #D라면
                        # 중간 지점 찾기 ADL
                        median_frame = np.argsort(data.index)[len(data) // 2]           # median_frame 기준
                        Window = data.iloc[median_frame - 30:median_frame + 30, :]



                    # plt.subplot(2,1,1)
                    # plt.plot(data.ASVM)
                    # plt.axvline(impact_frame, color='red')
                    # plt.subplot(2, 1, 2)
                    # plt.plot(Window.ASVM)
                    # plt.axvline(impact_frame, color='red')
                    # plt.show


                    # if movement == 'F':
                    #     Max_Frame = data.sort_values('ASVM', ascending=False)  # max값 찾기
                    #     Max_Frame = Max_Frame.index[0]
                    #     Max = max(data.ASVM)  # max값 찾기
                    #     Window = data.iloc[Max_Frame - 60:Max_Frame - 30,:6]  # data max-71값에서부터 max-21까지 자르기
                    # elif movement == 'D':
                    #     Max_Frame = data.index[int(len(data)/2)]
                    #     Max = data.ASVM[int(len(data)/2)]
                    #     Window = data.iloc[Max_Frame - 60:Max_Frame - 30,:6]  # data max-71값에서부터 max-21까지 자르기
                    # plt.plot(Window.ASVM)


                    Window = np.array(Window)

                    df = Window.reshape(1, -1) #1열(한줄로 쌓음)
                    df = pd.DataFrame(df)



                    #df_scale = pd.concat([df_scale, Max_scale], axis=True)

                    if count == 0:  # 데이터 쌓기
                        train_stack = df

                    else:
                        train_stack = pd.concat([train_stack, df], axis=0)


                    train_name.append(name)
                    train_motion.append(movement+m)
                    if movement =='D':
                        train_m_type.append(0)

                        if movement+str(m).zfill(2) == 'D01':
                            train_class.append('c1')

                        elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                            train_class.append('c2')

                        elif movement+str(m).zfill(2) in ['D05', 'D06','D07', 'D10']:
                            train_class.append('c3')

                        elif movement+str(m).zfill(2) in ['D08', 'D09']:
                            train_class.append('c4')


                    else:
                        train_m_type.append(1)
                    # train_target.append(Max)
                    #
                        if movement + str(m).zfill(2) in ['F02', 'F03', 'F05', 'F08', 'F09']:
                            train_class.append('c5')

                        elif movement + str(m).zfill(2) in ['F01', 'F04', 'F07', 'F11']:
                            train_class.append('c6')

                        elif movement + str(m).zfill(2) in ['F06', 'F10']:
                            train_class.append('c7')
                    count += 1
                except:
                    pass



# a = train_stack.isnull().sum() # nan값 있는지 확인
# np.sum(a)

#train_stack = train_stack.fillna(0)              # NaN값 -> 0으로 치환 (Zero Padding)
train_fall_indicator = pd.DataFrame(train_m_type)
train_fall_indicator.columns = ['FD']
# train_motion_class =


# train_target = pd.DataFrame(train_target)           #Train 데이터 전처리
# train_target_frame = pd.DataFrame(train_target_frame)
# test_target_frame = pd.DataFrame(test_target_frame)
train_name = pd.DataFrame(train_name)
train_name.columns = ['name']
#train_sort = pd.DataFrame(train_sort)


#train_class
train_class_indicator = pd.DataFrame(train_class)
train_class_indicator.columns = ['class']

# train_stack = pd.DataFrame(train_stack)
# train_stack.reset_index(drop=False, inplace=True)
# train_stack = train_stack.iloc[:, 1:]
#train_target.columns = ['736']       #Train 데이터 전처리
train_data = pd.concat([train_stack, train_fall_indicator, train_class_indicator, train_name], axis=1, sort=False)  # 데이터 합치기
# train_frame = pd.concat([train_name, train_target_frame], axis=1, sort=False)  # 두 데이터 합치기
# train_frame.columns = ['name','frame']
# train_frame.frame[0]



# test_frame = pd.concat([test_name, test_target_frame], axis=1, sort=False)  # 두 데이터 합치기
# test_frame.columns = ['name','frame']
# test_frame.frame[0]


test_target = []
test_name = []
test_sort = []
test_stack = []

Fall01 = [] #추락
Fall02 = [] #전도
NotFall = [] #일상동작

test_stack_FFH = []
test_stack_Fall = []
test_stack_NF = []
test_target_frame = []

test_motion = []
test_m_type = []
test_class = []
count = 0

e = 1 #피험자
m = 1# 동작번호
trial = 1 # 트라이얼
movement = 'F'

for e in list_test:
    for movement in ['D', 'F']:
        for m in range(1, 15):
            for trial in range(1,4):
                try:

                    t = trial

                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/ble_re/" %(e)
                    name = "P%s%s%sR%s" % (e, movement, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)
                    # name = "V%s%s%sR%s.csv" % (path, e)

                    data = pd.read_csv(str(name_csv))
                    data = pd.DataFrame(data)
                    data = data.iloc[:, 1:7]
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                    data['Acc_X'] = data.Acc_X / 9.8
                    data['Acc_Y'] = data.Acc_Y / 9.8
                    data['Acc_Z'] = data.Acc_Z / 9.8
                    #
                    # plt.plot(data.Acc_Z)
                    #
                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                    # #수동으로 데이터 자르기
                    # train_frame = pd.read_csv("train_frame.csv")
                    #
                    # Max_Frame = train_frame.frame[count]
                    # Max = max(data.ASVM)  # max값 찾기
                    # Window = data.iloc[Max_Frame - 54:Max_Frame - 24,:]  # data max-71값에서부터 max-21까지 자르기



                    # 충격 시점 앞 뒤 1초 자르기
                    if movement == 'F':
                        # 충격 시점 찾기 Fall
                        impact_frame = np.argmax(data.ASVM)
                        Window = data.iloc[impact_frame - 30:impact_frame + 30, :]
                    else:
                        # 중간 지점 찾기 ADL
                        median_frame = np.argsort(data.index)[len(data) // 2]
                        Window = data.iloc[median_frame - 30:median_frame + 30, :]

                    # plt.subplot(2,1,1)
                    # plt.plot(data.ASVM)
                    # plt.axvline(impact_frame, color='red')
                    # plt.subplot(2, 1, 2)
                    # plt.plot(Window.ASVM)
                    # plt.axvline(impact_frame, color='red')

                    # if movement == 'F':
                    #     Max_Frame = data.sort_values('ASVM', ascending=False)  # max값 찾기
                    #     Max_Frame = Max_Frame.index[0]
                    #     Max = max(data.ASVM)  # max값 찾기
                    #     Window = data.iloc[Max_Frame - 60:Max_Frame - 30,:6]  # data max-71값에서부터 max-21까지 자르기
                    # elif movement == 'D':
                    #     Max_Frame = data.index[int(len(data)/2)]
                    #     Max = data.ASVM[int(len(data)/2)]
                    #     Window = data.iloc[Max_Frame - 60:Max_Frame - 30,:6]  # data max-71값에서부터 max-21까지 자르기
                    # plt.plot(Window.ASVM)

                    Window = np.array(Window)

                    df = Window.reshape(1, -1)
                    df = pd.DataFrame(df)

                    # df_scale = pd.concat([df_scale, Max_scale], axis=True)

                    if count == 0:  # 데이터 쌓기
                        test_stack = df

                    else:
                        test_stack = pd.concat([test_stack, df], axis=0)

                    test_name.append(name)
                    test_motion.append(movement + m)
                    if movement == 'D':
                        test_m_type.append(0)

                        if movement+str(m).zfill(2) == 'D01':
                            test_class.append('c1')

                        elif movement+str(m).zfill(2) in ['D02', 'D03','D04', 'D11','D12', 'D13','D14']:
                            test_class.append('c2')

                        elif movement+str(m).zfill(2) in ['D05', 'D06','D07', 'D10']:
                            test_class.append('c3')

                        elif movement+str(m).zfill(2) in ['D08', 'D09']:
                            test_class.append('c4')

                    else:
                        test_m_type.append(1)
                    # train_target.append(Max)

                    if movement + str(m).zfill(2) in ['F02', 'F03', 'F05', 'F08', 'F09']:
                        test_class.append('c5')

                    elif movement + str(m).zfill(2) in ['F01', 'F04', 'F07', 'F11']:
                        test_class.append('c6')

                    elif movement + str(m).zfill(2) in ['F06', 'F10']:
                        test_class.append('c7')
                    count += 1

                except:
                    pass

test_stack = test_stack.reset_index(drop=True)  #reset_index:인덱스 없애고 drop=true:재정렬

test_fall_indicator = pd.DataFrame(test_m_type)
test_fall_indicator.columns = ['FD']

test_name = pd.DataFrame(test_name)
test_name.columns = ['name']

test_class_indicator = pd.DataFrame(test_class)
test_class_indicator.columns = ['class']


test_data = pd.concat([test_stack, test_fall_indicator, test_class_indicator, test_name], axis=1, sort=False)  # 두 데이터 합치기


# a = pd.read_csv('/test_data.csv')


# #모델만들기위해
# X_train = train_data.iloc[:, :480]
# Y_train = train_data.FD
# train_data.to_csv('train_data.csv')
#
# X_test = test_data.iloc[:, :480]
# Y_test = test_data.FD
# test_data.to_csv('test_data.csv')


# with open(r'C:\Users\fhrm5\Desktop\착용형 로봇 낙상 실험\낙상실험\실험데이터\Xsens전처리\pickle\All(ADL_rev)/'
#           'test_All_g.pkl', 'wb') as f:
#     pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(r'C:\Users\fhrm5\Desktop\착용형 로봇 낙상 실험\낙상실험\실험데이터\Xsens전처리\pickle\All(ADL_rev)/'
#           'train_All_g.pkl', 'wb') as f:
#     pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)