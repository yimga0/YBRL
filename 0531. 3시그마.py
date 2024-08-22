import numpy as np
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt

'----------------------------------------------------------------------------------------------------------------------'

e, m, t = 2,4,2
for e in range(1, 23):
    for m in range(2, 15):
        for t in range(1, 4):
            try:

                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/SW데이터(임계값)/"
                name = "S%sD01R01_SW" % e
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[1:, 1:7]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출

                mean = np.mean(data.ASVM)
                variance = np.var(data.ASVM)
                std = variance ** (1 / 2)
                threshold = mean + std * 6


                plot = "S%sD%sR%s_SW" % (e, m, t)
                plot_csv = "%s%s.csv" % (path, plot)
                plot_png = "%s.png" % (plot)

                plot_data = pd.read_csv(str(plot_csv))
                plot_data = pd.DataFrame(plot_data)
                # plot_data = plot_data.dropna(axis=0)
                plot_data = plot_data.iloc[:, 1:9]
                plot_data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']


                '---------------------------------------그래프----------------------------------------------'
                path_graph = r"C:/Users/abbey/Desktop/SW데이터(임계값)/그래프/"

                plt.figure(figsize=(10, 6))
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

                plt.tight_layout()
                plt.show()
                plt.savefig(path_graph + plot_png)
                plt.close()

            except:
                pass


#csv저장
# result_frame.to_csv(r"C:/Users/lsoor/OneDrive/바탕 화면/낙상데이터 8축/frame.csv", mode='a', header=False)


# print("Crossing frames count:", len(crossing_frames))

# points = np.where(np.isclose(plot_data.ASVM, mean + std * 4, atol=0.001))  #다 체크되어버림
# plt.scatter(points, plot_data.ASVM.iloc[points], color='red')
