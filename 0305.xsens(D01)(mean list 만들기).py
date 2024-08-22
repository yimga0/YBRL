import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt

path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/xsens D01조정/그래프/"
path = r"C:/Users/lsoor/OneDrive/바탕 화면/xsens D01조정/"

e,m,t = 10,1,1

count=0

for e in range(1,11):
    for m in range(1, 2):
        for trial in range(1,2):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                name = "P%sD%sR%s" % (e,m,t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=2)
                data = pd.DataFrame(data)
                # data = data.iloc[:, 5:11]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM','GSVM']

                data['Acc_X'] = data.Acc_X / 9.8
                data['Acc_Y'] = data.Acc_Y / 9.8
                data['Acc_Z'] = data.Acc_Z / 9.8

                data['Gyr_X'] = data.Gyr_X / 9.8
                data['Gyr_Y'] = data.Gyr_Y / 9.8
                data['Gyr_Z'] = data.Gyr_Z / 9.8

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                name_save_csv = path + name + ".csv"

                # 데이터 저장
                data.to_csv(name_save_csv, index=False)

                plt.figure(figsize=(10, 6))
                plt.title(name)
                plt.plot(data.ASVM)
                plt.ylabel('ASVM (g)')
                plt.xlabel('Frames')

                plt.tight_layout()
                plt.show()

                # plt.savefig(path_graph + name_png)
                # plt.close()

            except:

'-------------------------------------------------------------------------------------------------------------------'
import numpy as np
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt

path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/xsens D01조정/그래프/"
data_list = []

for e in range (1,11):
    m, t = 1, 1
    e = str(e).zfill(2)
    m = str(m).zfill(2)
    t = str(t).zfill(2)

    path = r"C:/Users/lsoor/OneDrive/바탕 화면/xsens D01조정/"
    name = "P%sD%sR%s" % (e, m, t)
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

merged_data.to_csv(path + 'mean.csv', index=False)
'---------------------------------------------------------------------------------------------------------------------'


#
#
#
# 그래프 한번에 여러개 그리기
# plt.figure(figsize=(10, 8))
# plt.subplot(4, 1, 1)  #세로, 가로, 몇번째
# plt.title('Acceleration SVM')
# plt.ylabel('ASVM($g$)')
# plt.xlabel('Frames')
# plt.plot(Acc_RMS, label='ASVM')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_ACC, label='Threshold')
# # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)
#
# plt.subplot(4, 1, 2)
# plt.title('Angular Velocity SVM')
# plt.ylabel('GSVM($\degree/s$)')
# plt.xlabel('Frames')
# plt.plot(Ang_RMS, label='GSVM')
# plt.axvline(np.argmax(Acc_RMS), label='Impact time', color='red')
# plt.axvline(Detection_Time, label='Detection time', color='blue')
# plt.axhline(TH_GYR, label='Threshold')
# # # plt.xlim(100, 500)
# plt.legend(loc='upper right', fontsize=8)



'--------------------------------------------------------------------------------------------------------------------'
'''
e, m , t = 1,1,1
for e in range(1,11):
    for m in range(1,2):
        for t in range(1,2):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/낙상 실험 전처리/P%s/xsens/" % e

                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=2)
                data = pd.DataFrame(data)
                data = data.iloc[:, 5:11]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                path_save = r"C:/Users/abbey/Desktop/xsens(D01조정)/"
                name_save_csv = path_save + name + ".csv"

                # 데이터 저장
                data.to_csv(name_save_csv, index=False)

'''

