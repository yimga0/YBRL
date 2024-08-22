import numpy as np
from scipy.stats import entropy
import pandas as pd

csv_path = r"C:/Users/abbey/Desktop/extract_data/"

e, m, trial = 1,1,1
count = 0
for e in range(1, 23):
    for m in range(1, 12):
        for trial in range(1, 4):
            try:
                t = trial
                e = str(e).zfill(2)
                m = str(m).zfill(2)
                t = str(t).zfill(2)

                path = r"C:/Users/abbey/Desktop/낙상 실험 전처리/P%s/xsens/" % e
                name = "P%sF%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data = pd.DataFrame(data)
                data = data.iloc[1:, 5:11]
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

                # 데이터 정규화
                data['Acc_X'] = data.Acc_X / 9.8
                data['Acc_Y'] = data.Acc_Y / 9.8
                data['Acc_Z'] = data.Acc_Z / 9.8
                data['Gyr_X'] = data.Gyr_X / 9.8
                data['Gyr_Y'] = data.Gyr_Y / 9.8
                data['Gyr_Z'] = data.Gyr_Z / 9.8

                data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5
                data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5

                data.to_csv(csv_path + name + '.csv', index=False)

            except:
                pass
