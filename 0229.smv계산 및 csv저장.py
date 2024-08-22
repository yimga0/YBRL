import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt

e, m, trial = 1,6,1
t = trial

e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = r"C:/Users/abbey/Desktop/낙상 실험 전처리//P%s/xsens/" % e

name = "P%sD%sR%s" % (e, m, t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=2)
data = pd.DataFrame(data)
data = data.iloc[:, :8]  #Acc_X ~ Gyr_z
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z','ASVM', 'GSVM']

data['ASVM'] = (data.Acc_X ** 2 + data.Acc_Y ** 2 + data.Acc_Z ** 2) ** 0.5  # SVM 특성 추출
data['GSVM'] = (data.Gyr_X ** 2 + data.Gyr_Y ** 2 + data.Gyr_Z ** 2) ** 0.5  # SVM 특성 추출


path_save = r"C:/Users/abbey/Desktop/xsens 데이터 조정/"

name_save_csv = path_save + name + ".csv"

# 데이터 저장
data.to_csv(name_save_csv, index=False)
