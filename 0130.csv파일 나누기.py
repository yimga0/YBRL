import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt


csv_file_path= r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/split_기준.csv"

index = pd.read_csv(str(csv_file_path))
index = pd.DataFrame(index)
index = index.iloc[:, 1:3]
index.columns = ['frame', 'name']


data_a = []
data_b = []

e, m, t = 1,2,1
count=0
count_error=0
for e in range(1,21):
    for m in range(1, 15):
        if m in [2, 3, 4, 10, 11, 12, 13, 14]:
            for t in range(1,4):
                try:
                    e = str(e).zfill(2)
                    m = str(m).zfill(2)
                    t = str(t).zfill(2)

                    path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리/P%s/ble_re/" % e

                    name = "P%sD%sR%s" % (e, m, t)
                    name_csv = "%s%s.csv" % (path, name)
                    name_png = "%s.png" % (name)


                    data = pd.read_csv(str(name_csv))
                    data = pd.DataFrame(data)
                    data = data.iloc[:, 1:7]  #Acc_X ~ Gyr_z
                    data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                    data['Acc_X'] = data.Acc_X
                    data['Acc_Y'] = data.Acc_Y
                    data['Acc_Z'] = data.Acc_Z

                    data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                    data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5


                    if name == index['name'].values[count]:
                        frame_values = index['frame'].iloc[count]

                        data_a = data.iloc[:frame_values+1, :]
                        data_b = data.iloc[frame_values+1:, :]

                        split_flie_path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/"

                        file_name_a = name + '_a.png'
                        file_name_b = name + '_b.png'

                        # data_a.to_csv(split_flie_path + file_name_a)
                        # data_b.to_csv(split_flie_path + file_name_b)

                        '---------------------------graph---------------------------------------'

                        file_a = name + '_a'
                        file_b = name + '_b'

                        path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 데이터 split/그래프/"

                        plt.figure(figsize=(10, 8))

                        # plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
                        # plt.title(file_a)
                        # # plt.title('Acceleration SVM')
                        # plt.ylabel('ASVM(g)')
                        # plt.xlabel('Frames')
                        # plt.plot(data_a.ASVM, label='ASVM')
                        # plt.tight_layout()
                        # plt.show()
                        #
                        # plt.subplot(2, 1, 2)  # 세로, 가로, 몇번째
                        # # plt.title('Acceleration SVM')
                        # plt.ylabel('GSVM (degree/s)')
                        # plt.xlabel('Frames')
                        # plt.plot(data_a.GSVM, label='GSVM')
                        # plt.tight_layout()
                        # plt.show()
                        #
                        # plt.savefig(path_graph + file_name_a)
                        # plt.close()



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

                        plt.savefig(path_graph + file_name_b)
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
