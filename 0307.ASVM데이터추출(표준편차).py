import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt


e, m, trial =1,1,1
count = 0

e = str(e).zfill(2)
m = str(m).zfill(2)
trial = str(trial).zfill(2)

path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상데이터 8축/"
name = "P%sD%sR%s" % (e, m, trial)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=1)
data = pd.DataFrame(data)
data = data.iloc[:, :]
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM','GSVM']

mean = np.mean(data.ASVM)
variance = np.var(data.ASVM)

std = variance ** (1 / 2)


'----------------------------------------------------------------------------------------------------------------------'

path_graph = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상데이터 8축/ASVM그래프/"

s, d, t =1,2,1
# for s in range(1, 2):
for d in range(2, 15):
    for t in range(1, 4):
        try:

            s = str(s).zfill(2)
            d = str(d).zfill(2)
            t = str(t).zfill(2)

            plot = "P%sD%sR%s" % (s, d, t)
            plot_csv = "%s%s.csv" % (path, plot)
            plot_png = "%s.png" % (plot)

            plot_data = pd.read_csv(str(plot_csv), header=1)
            plot_data = pd.DataFrame(plot_data)
            plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

            std = variance ** (1 / 2)

            plt.figure(figsize=(10, 6))
            plt.title(plot)
            plt.plot(plot_data.ASVM)  # 그래프그리기 (((data_test.ASVM-mean)/9.8)/std)
            threshold = mean + std * 3

            plt.axhline(threshold, color='red', linewidth=1)
            plt.ylabel('ASVM (g)')
            plt.xlabel('Frames')

            # 교차점 찾기
            crossing_points = np.where(np.diff(np.sign(plot_data.ASVM - threshold)))[0]
            crossing_frames = crossing_points

            # 교차점 표시
            for cp in crossing_points:
                plt.plot(cp, plot_data.ASVM[cp], 'ro', markersize=1)

            plt.axvline(crossing_frames[0], color='green', linestyle='-')  # 첫 번째 교차점
            plt.axvline(crossing_frames[-1], color='green', linestyle='-')  # 마지막 교차점

            plt.tight_layout()
            plt.show()
            plt.savefig(path_graph + plot_png)
            plt.close()

            print("Crossing frames:", crossing_frames)

            frames_df = pd.DataFrame({'frame': crossing_frames})

            first_frame = frames_df['frame'].iloc[0]
            last_frame = frames_df['frame'].iloc[-1]

            result_frame = pd.DataFrame({'name': [plot], 'first': [first_frame], 'last': [last_frame]})

            # print(result_frame)

        except:
            pass

result_frame.to_csv(r"C:/Users/lsoor/OneDrive/바탕 화면/낙상데이터 8축/frame.csv")

# print("Crossing frames count:", len(crossing_frames))

# points = np.where(np.isclose(plot_data.ASVM, mean + std * 4, atol=0.001))  #다 체크되어버림
# plt.scatter(points, plot_data.ASVM.iloc[points], color='red')
