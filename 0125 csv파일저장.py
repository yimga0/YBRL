import pandas as pd  # 데이터 프레임

a = []
b = []

e, m, trial = 1, 1, 1
count=0
for e in range(1,21):
    for m in range(1, 15):
        for trial in range(1,4):
            try:

                t = trial

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

                mid_index = data.index[int(len(data) // 2)]  # 전체 데이터 길이 중간점
                a.append(mid_index)
                b.append(name)

            except:
                count += 1


'------------------------------------------------csv파일 만들기-----------------------------------------------'
data_mid = pd.DataFrame(a)
data_mid.columns = ['frame']

data_name = pd.DataFrame(b)
data_name.columns = ['name']

data_merge = pd.concat((data_mid, data_name), axis=1)

data_merge.to_csv(r"C:/Users/lsoor/OneDrive/바탕 화면/낙상 실험 전처리\mid_data.csv")