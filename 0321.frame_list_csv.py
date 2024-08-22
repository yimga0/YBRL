import numpy as np
# from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import csv


e, m, t = 1,2,3
frame_list =[]
data_list = []

for e in range(1,23):
    for m in range(2, 15):
        for t in range(1, 4):
            try:

                m = str(m).zfill(2)
                e = str(e).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/낙상데이터 8축/"
                one = "P%sD01R01" % e
                one_csv = "%s%s.csv" % (path, one)
                one_png = "%s.png" % (one)

                data = pd.read_csv(str(one_csv), header=1)
                data = pd.DataFrame(data)
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM', 'GSVM']

                mean = np.mean(data.ASVM)
                variance = np.var(data.ASVM)
                std = variance ** (1 / 2)
                threshold = mean + std * 3

                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                plot_data = pd.read_csv(str(name_csv), header=1)
                plot_data = pd.DataFrame(plot_data)
                plot_data = plot_data.dropna(axis=0)
                plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']


                #ASVM
                # 교차점 찾기
                crossing_points = np.where(np.diff(np.sign(plot_data.ASVM - threshold)))[0]
                crossing_frames = crossing_points

                # 교차점 표시
                for cp in crossing_points:
                    plt.plot(cp, plot_data.ASVM[cp], 'ro', markersize=1)

                frames_df = pd.DataFrame({'frame': crossing_frames})

                first_frame = frames_df['frame'].iloc[0]
                last_frame = frames_df['frame'].iloc[-1]
                result_frame = pd.DataFrame({'name': [name], 'first': [first_frame], 'last': [last_frame]})

                '------------------------------------------------------------------------------------------------------'
                #GSVM
                mean1 = np.mean(data.GSVM)
                variance1 = np.var(data.GSVM)
                std1 = variance1 ** (1 / 2)
                threshold1 = mean1 + std1 * 3

                # 교차점 찾기
                crossing_points1 = np.where(np.diff(np.sign(plot_data.GSVM - threshold1)))[0]
                crossing_frames1 = crossing_points1

                # 교차점 표시
                for cp in crossing_points1:
                    plt.plot(cp, plot_data.GSVM[cp], 'ro', markersize=1)

                frames_df1 = pd.DataFrame({'frame': crossing_frames1})

                first_frame1 = frames_df1['frame'].iloc[0]
                last_frame1 = frames_df1['frame'].iloc[-1]
                result_frame1 = pd.DataFrame({'name': [name], 'first': [first_frame1], 'last': [last_frame1]})

                columns = ['name', 'ASVM_first', 'ASVM_last', 'GSVM_first', 'GSVM_last']
                values = [[name], [first_frame], [last_frame], [first_frame1], [last_frame1]]
                data_list = pd.DataFrame(dict(zip(columns, values)))
                frame_list.append(data_list)
                total_frame_df = pd.concat(frame_list, ignore_index=True)



            except:
                pass


# total_frame_df.to_csv(r"C:/Users/abbey/Desktop/낙상데이터 8축/frame_list.csv")

