import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

e, m, t =1,3,3
for e in range(1,13):
    for m in range(1,12):
        for t in range(1,4):
            try:
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/SW(각도연산10)/"
                name = "S%sD%sR%s_SW" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv))
                data = pd.DataFrame(data)
                data = data.iloc[:, 1:11]  # 가속도, 각속도 3축 데이터만 불러오게 자름
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM', 'Roll', 'Pitch']

                Angle = np.sqrt(data.Roll ** 2 + data.Pitch ** 2)

                path_graph = r"C:/Users/abbey/Desktop/SW(각도연산10)/그래프/"
                plt.figure(figsize=(10, 8))
                plt.subplot(3, 1, 1)
                plt.title(name)
                plt.plot(data.ASVM)  # 그래프그리기
                plt.ylabel('ASVM (g)')
                plt.xlabel('Frames')
                plt.axhline(y=0.8, label='0.8', color='red')
                # plt.axvline(x=first_crossing1, color='green', linestyle='-', label='First Crossing')
                plt.tight_layout()

                plt.subplot(3, 1, 2)
                plt.plot(data.GSVM)  # 그래프그리기
                plt.ylabel('GSVM (g)')
                plt.xlabel('Frames')
                plt.axhline(y=80, label='80', color='red')
                crossing_points = np.where(np.diff(np.sign(data.GSVM - 100)))[0]
                plt.tight_layout()

                plt.subplot(3, 1, 3)
                # plt.plot(data.Roll, label='Roll')
                # plt.plot(data.Pitch, label='Pitch')
                plt.plot(Angle, label='Angle')
                plt.xlabel('Frames')
                plt.ylabel('Angle $(\degree)$')
                plt.legend(loc='upper right')
                plt.axhline(y=30, label='25', color='red')
                plt.tight_layout()

                plt.show()
                plt.savefig(path_graph + name_png)
                plt.close()


            except:
                pass

                '----------------------------------가속도로 각도 연산--------------------------------------------'
                Roll = np.arctan2(Ay, np.sqrt(Ax ** 2 + Az ** 2)) * 180 / np.pi  # 가속도계 데이터 롤 각도
                Pitch = np.arctan2(-Ax, np.sqrt(Ay ** 2 + Az ** 2)) * 180 / np.pi  # 가속도계 데이터 피치 각도

                # 상보필터
                # Fs = 50
                # dt = 1 / Fs
                #
                # # Gx가 1차원 배열인지 확인하고 1차원 배열로 변환
                # Gx = np.array(Gx).flatten()
                # Roll_w = np.zeros((len(Gx), 1))  # 자이로스코프 데이터 롤 각도
                # for n in range(len(Gx) - 2):
                #     Roll_w[n + 2, 0] = Roll_w[n, 0] + (Gx[n] + 4 * Gx[n + 1] + Gx[n + 2]) * dt / 3  # 심슨룰
                #
                # Gy = np.array(Gy).flatten()
                # Pitch_w = np.zeros((len(Gy), 1))  # 자이로스코프 데이터 피치 각도
                # for n in range(len(Gy) - 2):
                #     Pitch_w[n + 2, 0] = Pitch_w[n, 0] + (Gy[n] + 4 * Gy[n + 1] + Gy[n + 2]) * dt / 3  # 심슨룰
                #
                # Roll_w = Roll_w.flatten()
                # Pitch_w = Pitch_w.flatten()
                #
                # a = 0.01  # 가속도계 데이터의 가중치   (가속도계 데이터가 20%의 영향, 자이로스코프 데이터가 80%의 영향)
                # Roll_a = Roll * a + Roll_w * (1 - a)  # 최종적으로 결합된 롤 각도
                # Pitch_a = Pitch * a + Pitch_w * (1 - a)  # 최종적으로 결합된 피치 각도
                #
                # Angle = np.sqrt(Roll_a ** 2 + Pitch_a ** 2)
                # #
                # '------------------------------------csv파일 만들기----------------------------------------'
                # #
                # # data_RP = pd.concat( (Roll_a,Pitch_a), axis=1)
                # # data_RP.columns = ['Roll', 'Pitch']
                # #
                # # data_merge = pd.concat((data1,data_RP ), axis=1)
                # # path_data = r"C:/Users/abbey/Desktop/SW(각도연산2)/"
                # # data_merge.to_csv(path_data+name+'.csv')
                # #
            #
            #
            #     '-------------------------------그래프 그리기(3가지)------------------------------------------'
            #
            #     crossing_points1 = np.where(np.diff(np.sign(data1.ASVM - 0.9)))[0]
            #     if crossing_points1.size > 0:
            #         first_crossing1 = crossing_points1[0]
            #
            #     crossing_points2 = np.where(np.diff(np.sign(data1.GSVM - 100)))[0]
            #     if crossing_points2.size > 0:
            #         first_crossing2 = crossing_points2[0]
            #
            #     crossing_points = np.where((data1.ASVM < 0.9) & (data1.GSVM > 100))[0]
            #     if crossing_points.size > 0:
            #         first_crossing = crossing_points[0]
            #
            #
            #     path_graph = r"C:/Users/abbey/Desktop/SW(각도연산2)/그래프/"
            #     plt.figure(figsize=(10, 8))
            #     plt.subplot(3, 1, 3)
            #     plt.title(name)
            #     # plt.plot(Roll_a, label='Angle')
            #     # plt.plot(Pitch_a, label='Angle')
            #     plt.xlabel('Frames')
            #     plt.ylabel('Angle $(\degree)$')
            #     plt.legend(loc='upper right')
            #     plt.axvline(x=first_crossing1, color='green', linestyle='-', label='First Crossing')
            #     plt.axvline(x=first_crossing2, color='red', linestyle='-', label='First Crossing')
            #     plt.tight_layout()
            #     plt.show()
            #     plt.subplot(3, 1, 3)
            #     # plt.plot(data.Roll, label='Roll')
            #     # plt.plot(data.Pitch, label='Pitch')
            #
            #     plt.subplot(3, 1, 1)
            #     plt.plot(data1.ASVM)  # 그래프그리기
            #     plt.ylabel('ASVM (g)')
            #     plt.xlabel('Frames')
            #     plt.axhline(y=0.9, label='0.9', color='orange')
            #     plt.axvline(x=first_crossing1, color='green', linestyle='-', label='First Crossing')
            #     plt.tight_layout()
            #
            #     plt.subplot(3, 1, 2)
            #     plt.plot(data1.GSVM)  # 그래프그리기
            #     plt.ylabel('GSVM (g)')
            #     plt.xlabel('Frames')
            #     plt.axhline(y=100, label='100', color='orange')
            #     crossing_points = np.where(np.diff(np.sign(data1.GSVM - 100)))[0]
            #     if crossing_points.size > 0:
            #         first_crossing = crossing_points[0]
            #         plt.axvline(x=first_crossing2, color='purple', linestyle='-', label='First Crossing')
            #     plt.tight_layout()
            #
            #     plt.show()
            #
            #     # plt.savefig(path_graph + name_png)
            #     plt.close()
            #
            # except:
            #     pass
            #
