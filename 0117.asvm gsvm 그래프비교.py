import numpy as np  # 연산+배열 모듈
import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt
# from tqdm import tqdm  # 시간
# from scipy import signal
# import seaborn as sns

a = []

path_graph = 'C:/Users/abbey/Desktop/extract_data/'   #그래프 어디에 그릴지

e, m, trial = 4, 11,2
count=0
for e in range(1,21):
    for m in range(1, 15):
        for trial in range(1,4):
            try:

                t = trial

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/extract_data/"

                name = "P%sF%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)


                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[:, 1:7]  #Acc_X ~ Gyr_z
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                # plt.plot(data.Acc_Z)
                # plt.show()

                data['Acc_X'] = data.Acc_X / 9.8
                data['Acc_Y'] = data.Acc_Y / 9.8
                data['Acc_Z'] = data.Acc_Z / 9.8


                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


                '-------------------- graph 비교 -----------------------'

                # mid_index1 = np.argsort(data.ASVM)[len(data) // 2]
                # mid_index2 = np.argsort(data.ASVM)[len(data) // 2]
                mid_index = data.index[int(len(data) // 2)]  #전체 데이터 길이 중간점
                a.append(mid_index)

                plt.figure(figsize=(10, 8))

                plt.subplot(2, 1, 1)  #세로, 가로, 몇번째
                plt.title(name)
                # plt.title('Acceleration SVM')
                plt.ylabel('ASVM(g)')
                plt.xlabel('Frames')
                plt.plot(data.ASVM, label='ASVM')
                # plt.axvline(mid_index, label='Mid', color='red')
                plt.legend(loc='upper right', fontsize=8)

                plt.subplot(2, 1, 2)
                # plt.title('Angular Velocity SVM')
                plt.ylabel('GSVM (degree/s)')
                plt.xlabel('Frames')
                plt.plot(data.GSVM, label='GSVM')
                # plt.axvline(mid_index, label='Mid', color='red')
                plt.legend(loc='upper right', fontsize=8)

                plt.tight_layout()
                plt.show()

                # plt.savefig(path_graph + name_png)
                # plt.close()

            except:
                count += 1

Acc_X,Acc_Y,Acc_Z=1468,1468,1468
asvm = (Acc_X ** 2 + Acc_Y ** 2 + Acc_Z ** 2) ** 0.5
