import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt


a = []
b = []
data_a = []
data_b = []

e, m, t = 1,12,1
count=0
count_error=0
for e in range(21,23):
    for m in range(1, 15):
        if m in [10]:
            for t in range(1,4):
                try:
                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/abbey/Desktop/extract_data/"
                    name = "P%sD%sR%s" % (e, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)


                    data = pd.read_csv(str(name_csv))
                    data = pd.DataFrame(data)
                    data = data.iloc[:, :]  #Acc_X ~ Gyr_z
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM', 'GSVM']

                    max_value = max(data.ASVM)
                    max_index = data[data.ASVM == max_value].index[0] + 100

                    mid_index = data.index[int(len(data) // 2)]  # 전체 데이터 길이 중간점
                    a.append(mid_index)
                    b.append(name)

                    data_mid = pd.DataFrame(a)
                    data_mid.columns = ['frame']

                    data_name = pd.DataFrame(b)
                    data_name.columns = ['name']

                    index = pd.concat((data_mid, data_name), axis=1)

                    index.to_csv(r"C:/Users/abbey/Desktop/동작분류(10class)/max_data.csv")

                    if name == index['name'].values[count]:
                        frame_values = index['frame'].iloc[count]

                        data_a = data.iloc[:frame_values+1, :]
                        data_b = data.iloc[frame_values+1:, :]

                        split_flie_path = r"C:/Users/abbey/Desktop/동작분류(10class)/"

                        file_name_a = name + '_a.csv'
                        file_name_b = name + '_b.csv'

                        data_a.to_csv(split_flie_path + file_name_a)
                        data_b.to_csv(split_flie_path + file_name_b)

                        '---------------------------graph---------------------------------------'

                        file_a = name + '_a'
                        file_b = name + '_b'
                        file_a_png = name + '_a.png'
                        file_b_png = name + '_b.png'


                        path_graph = r"C:/Users/abbey/Desktop/동작분류(10class)/그래프/"

                        plt.figure(figsize=(10, 8))

                        plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
                        plt.title(file_a)
                        # plt.title('Acceleration SVM')
                        plt.ylabel('ASVM(g)')
                        plt.xlabel('Frames')
                        plt.plot(data_a.ASVM, label='ASVM')
                        plt.tight_layout()
                        plt.show()

                        plt.subplot(2, 1, 2)  # 세로, 가로, 몇번째
                        # plt.title('Acceleration SVM')
                        plt.ylabel('GSVM (degree/s)')
                        plt.xlabel('Frames')
                        plt.plot(data_a.GSVM, label='GSVM')
                        plt.tight_layout()
                        plt.show()

                        plt.savefig(path_graph + file_a_png)
                        plt.close()

                        '-----------------------------------------------------------------------'

                        plt.figure(figsize=(10, 8))
                        plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
                        plt.title(file_b)
                        # plt.title('Acceleration SVM')
                        plt.ylabel('ASVM(g)')
                        plt.xlabel('Frames')
                        plt.plot(data_b.ASVM, label='ASVM')
                        plt.tight_layout()
                        plt.show()

                        plt.subplot(2, 1, 2)  # 세로, 가로, 몇번째
                        # plt.title('Acceleration SVM')
                        plt.ylabel('GSVM (degree/s)')
                        plt.xlabel('Frames')
                        plt.plot(data_b.GSVM, label='GSVM')
                        plt.tight_layout()
                        plt.show()

                        plt.savefig(path_graph + file_b_png)
                        plt.close()

                    count += 1

                except:
                    count_error += 1

'---------------------------------------------------------------------------------------------------------------------'

#
# path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/그래프/"
# # data_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/P%SD%sR%s_%s.csv"
#
#
# data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
# data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출
#
# plt.figure(figsize=(10, 8))
#
# plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
# plt.title(name)
# # plt.title('Acceleration SVM')
# plt.ylabel('ASVM(g)')
# plt.xlabel('Frames')
# plt.plot(data_a.ASVM, label='ASVM')
# plt.tight_layout()
# plt.show()
# plt.savefig(path_graph + file_name_a)
# plt.close()
#
# plt.subplot(2, 1, 2)  # 세로, 가로, 몇번째
# plt.title(name)
# # plt.title('Acceleration SVM')
# plt.ylabel('GSVM (degree/s)')
# plt.xlabel('Frames')
# plt.plot(data_a.GSVM, label='GSVM')
# plt.tight_layout()
# plt.show()
# plt.savefig(path_graph + file_name_a)
# plt.close()



'------------------------------------------------csv파일 만들기-----------------------------------------------'
# import pandas as pd  # 데이터 프레임
# import matplotlib.pyplot as plt
#
#
# path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/그래프/"
#
#
# data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
# data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출
#
# data = data.iloc[1:, :]
#
# plt.figure(figsize=(10, 8))
#
# plt.subplot(2, 1, 1)  #세로, 가로, 몇번째
# plt.title(name)
# # plt.title('Acceleration SVM')
# plt.ylabel('ASVM(g)')
# plt.xlabel('Frames')
# plt.plot(data.ASVM, label='ASVM')
#
# plt.subplot(2, 1, 2)
# # plt.title('Angular Velocity SVM')
# plt.ylabel('GSVM (degree/s)')
# plt.xlabel('Frames')
# plt.plot(data.GSVM, label='GSVM')
#
# plt.tight_layout()
# plt.show()
#
# plt.savefig(path_graph + name_png)
# plt.close()
#
#
