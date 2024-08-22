import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt

'----------------------------------------------------------------------------------------------------------------------'

path_graph = r"C:/Users/abbey/Desktop/낙상데이터 8축/GSVM그래프/"

e, m, t =1,1,1
for e in range(1, 23):
    for m in range(1, 15):
        for t in range(1, 4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/낙상데이터 8축/"
                name = "P%sD01R01" % e
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                mean = np.mean(data.ASVM)
                variance = np.var(data.ASVM)
                std = variance ** (1 / 2)
                threshold = mean + std * 3

                plot = "P%sD%sR%s" % (e, m, t)
                plot_csv = "%s%s.csv" % (path, plot)
                plot_png = "%s.png" % (plot)

                plot_data = pd.read_csv(str(plot_csv), header=1)
                plot_data = pd.DataFrame(plot_data)
                plot_data = plot_data.dropna(axis=0)
                plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                # plt.figure(figsize=(10, 6))
                # plt.title(plot)
                # plt.plot(plot_data.ASVM)  # 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)
                #
                # plt.axhline(threshold, color='red', linewidth=1)
                # plt.ylabel('ASVM (g)')
                # plt.xlabel('Frames')
                #
                # # 교차점 찾기
                # crossing_points = np.where(np.diff(np.sign(plot_data.ASVM - threshold)))[0]
                # crossing_frames = crossing_points
                #
                # # 교차점 표시
                # for cp in crossing_points:
                #     plt.plot(cp, plot_data.ASVM[cp], 'ro', markersize=1)
                #
                # frames_df = pd.DataFrame({'frame': crossing_frames})
                #
                # first_frame = frames_df['frame'].iloc[0]
                # last_frame = frames_df['frame'].iloc[-1]
                #
                # result_frame = pd.DataFrame({'name': [plot], 'first': [first_frame], 'last': [last_frame]})
                #
                # plt.axvline(first_frame, color='green', linestyle='-')  # 첫 번째 교차점
                # plt.axvline(last_frame, color='green', linestyle='-')  # 마지막 교차점
                #
                # plt.tight_layout()
                # plt.show()
                # plt.savefig(path_graph + plot_png)
                # plt.close()

                '----------------------------------------------------------------------------------------------'

                mean1 = np.mean(data.GSVM)
                variance1 = np.var(data.GSVM)
                std1 = variance1 ** (1 / 2)
                threshold1 = mean1 + std1 * 3

                plt.figure(figsize=(10, 6))
                plt.title(plot)
                plt.plot(plot_data.GSVM)  # 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)

                plt.axhline(threshold1, color='red', linewidth=1)
                plt.ylabel('GSVM (g)')
                plt.xlabel('Frames')

                # 교차점 찾기
                crossing_points1 = np.where(np.diff(np.sign(plot_data.GSVM - threshold1)))[0]
                crossing_frames1 = crossing_points1

                # 교차점 표시
                for cp in crossing_points1:
                    plt.plot(cp, plot_data.GSVM[cp], 'ro', markersize=1)

                frames_df1 = pd.DataFrame({'frame': crossing_frames1})

                first_frame1 = frames_df1['frame'].iloc[0]
                last_frame1 = frames_df1['frame'].iloc[-1]

                result_frame1 = pd.DataFrame({'name': [plot], 'first': [first_frame1], 'last': [last_frame1]})

                plt.axvline(first_frame1, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(last_frame1, color='green', linestyle='-')  # 마지막 교차점

                plt.tight_layout()
                plt.show()
                plt.savefig(path_graph + plot_png)
                plt.close()

                # print("Crossing frames:", crossing_frames1)

                # print(result_frame)

            except:
                pass




'''
import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt


path_graph = r"C:/Users/abbey/Desktop/낙상데이터 8축/ASVM+GSVM/"

e, m, t =1,1,1
for e in range(1, 23):
    for m in range(1, 15):
        for t in range(1, 4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/낙상데이터 8축/"
                name = "P%sD01R01" % e
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                mean = np.mean(data.ASVM)
                variance = np.var(data.ASVM)
                std = variance ** (1 / 2)
                threshold = mean + std * 3

                plot = "P%sD%sR%s" % (e, m, t)
                plot_csv = "%s%s.csv" % (path, plot)
                plot_png = "%s.png" % (plot)

                plot_data = pd.read_csv(str(plot_csv), header=1)
                plot_data = pd.DataFrame(plot_data)
                plot_data = plot_data.dropna(axis=0)
                plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                plt.title(plot)
                plt.plot(plot_data.ASVM)  # 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)

                plt.axhline(threshold, color='red', linewidth=1)
                plt.ylabel('ASVM (g)')
                plt.xlabel('Frames')

                # 교차점 찾기
                crossing_points = np.where(np.diff(np.sign(plot_data.ASVM - threshold)))[0]
                crossing_frames = crossing_points

                # 교차점 표시
                for cp in crossing_points:
                    plt.plot(cp, plot_data.ASVM[cp], 'ro', markersize=1)

                frames_df = pd.DataFrame({'frame': crossing_frames})

                first_frame = frames_df['frame'].iloc[0]
                last_frame = frames_df['frame'].iloc[-1]

                result_frame = pd.DataFrame({'name': [plot], 'first': [first_frame], 'last': [last_frame]})

                plt.axvline(first_frame, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(last_frame, color='green', linestyle='-')  # 마지막 교차점

                '----------------------------------------------------------------------------------------------'

                mean1 = np.mean(data.GSVM)
                variance1 = np.var(data.GSVM)
                std1 = variance1 ** (1 / 2)
                threshold1 = mean1 + std1 * 3

                plt.subplot(2,1,2)
                plt.plot(plot_data.GSVM)  # 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)

                plt.axhline(threshold1, color='red', linewidth=1)
                plt.ylabel('GSVM (g)')
                plt.xlabel('Frames')

                # 교차점 찾기
                crossing_points1 = np.where(np.diff(np.sign(plot_data.GSVM - threshold1)))[0]
                crossing_frames1 = crossing_points1

                # 교차점 표시
                for cp in crossing_points1:
                    plt.plot(cp, plot_data.GSVM[cp], 'ro', markersize=1)

                frames_df1 = pd.DataFrame({'frame': crossing_frames1})

                first_frame1 = frames_df1['frame'].iloc[0]
                last_frame1 = frames_df1['frame'].iloc[-1]

                result_frame1 = pd.DataFrame({'name': [plot], 'first': [first_frame1], 'last': [last_frame1]})

                plt.axvline(first_frame1, color='green', linestyle='-')  # 첫 번째 교차점
                plt.axvline(last_frame1, color='green', linestyle='-')  # 마지막 교차점

                plt.tight_layout()
                plt.show()
                plt.savefig(path_graph + plot_png)
                plt.close()

                # print("Crossing frames:", crossing_frames1)

                # print(result_frame)

            except:
                pass
'''