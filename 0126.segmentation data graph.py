 import pandas as pd  # 데이터 프레임
i mport matplotlib.pyplot as plt

csv_file_path= r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/mid_data_수정중.csv"
graph_path = r"C:/Users/lsoor/OneDrive/바탕 화면/ble 그래프 최종/"

index = pd.read_csv(str(csv_file_path))
index = pd.DataFrame(index)
index = index.iloc[:, 1:3]
index.columns = ['frame', 'name']

a = []
e, m, t = 1, 1,1
count=0
count_error=0
for e in range(1,21):
    for m in range(1, 15):
        for t in range(1,4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/ble_re/" % e

                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)


                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[:, 1:7]  #Acc_X ~ Gyr_z
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                data['Acc_X'] = data.Acc_X
                data['Acc_Y'] = data.Acc_Y
                data['Acc_Z'] = data.Acc_Z

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


                '-------------------- graph 비교 -----------------------'


                if name == index['name'].values[count]:
                    frame_values = index['frame'].iloc[count]

                    line = data.index[int(frame_values)]

                    plt.figure(figsize=(10, 8))

                    plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
                    plt.title(name)
                    # plt.title('Acceleration SVM')
                    plt.ylabel('ASVM(g)')
                    plt.xlabel('Frames')
                    plt.plot(data.ASVM, label='ASVM')
                    plt.axvline(line, label='Mid', color='red')
                    plt.legend(loc='upper right', fontsize=8)

                    plt.subplot(2, 1, 2)
                    # plt.title('Angular Velocity SVM')
                    plt.ylabel('GSVM (degree/s)')
                    plt.xlabel('Frames')
                    plt.plot(data.GSVM, label='GSVM')
                    plt.axvline(line, label='Mid', color='red')
                    plt.legend(loc='upper right', fontsize=8)

                    plt.tight_layout()
                    plt.show()

                    # plt.savefig(graph_path + name_png)
                    # plt.close()

                count += 1

            except:
                count_error += 1


