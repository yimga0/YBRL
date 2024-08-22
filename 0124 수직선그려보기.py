import pandas as pd  # 데이터 프레임
import matplotlib.pyplot as plt

e, m, t = 6,14,1


e = str(e).zfill(2)
m = str(m).zfill(2)
t = str(t).zfill(2)

path = r"C:/Users/lsoor/OneDrive/바탕 화면/낙상데이터 8축/"

name = "P%sD%sR%s" % (e, m, t)
name_csv = "%s%s.csv" % (path, name)
name_png = "%s.png" % (name)

data = pd.read_csv(str(name_csv), header=2)
data = pd.DataFrame(data)
data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

'--------------------graph-----------------------'
plt.close()
line =130
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)  #세로, 가로, 몇번째

plt.title(name)
plt.ylabel('ASVM(g)')
plt.xlabel('Frames')
plt.plot(data.ASVM, label='ASVM')
plt.axvline(line, label='Mid', color='red')
plt.legend(loc='upper right', fontsize=8)

plt.subplot(2, 1, 2)
# plt.title('Angular Velocity SVM')
plt.ylabel('GSVM (degree/s)')
plt.xlabel('Frames')
plt.plot(data.GSVM, label='GSVM')
plt.axvline(line, label='Mid', color='red')
plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
