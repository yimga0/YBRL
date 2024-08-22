import pandas as pd
import matplotlib.pyplot as plt

############ Nano33BLE 데이터 불러오기 ############
# 가속도 데이터
path = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\231106 Test\\'
file_name = 'Raw data (g).csv'

# CSV 파일을 읽어옴
data = pd.read_csv(path + file_name)


acc = data.iloc[0::2, 1:4] # 짝수 열만
acc.columns = ['Acc_X', 'Acc_Y', 'Acc_Z']
acc = acc.reset_index(drop=True)
# plt.plot(acc)

gyr = data.iloc[1::2, 1:4] # 홀수 열만
gyr.columns = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
gyr = gyr.reset_index(drop=True)
# plt.plot(gyr)

BLE_data = pd.concat([acc, gyr], axis=1)

# 가속도 그래프
plt.subplot(1, 2, 1)
plt.plot(BLE_data.Acc_X, label='Acc_X')
plt.plot(BLE_data.Acc_Y, label='Acc_Y')
plt.plot(BLE_data.Acc_Z, label='Acc_Z')
plt.ylabel('Acceleration (g)')
plt.xlabel('Frames')
plt.legend()
plt.show()


# 각속도 그래프
plt.subplot(1, 2, 2)
plt.plot(BLE_data.Gyr_X, label='Gyr_X')
plt.plot(BLE_data.Gyr_Y, label='Gyr_Y')
plt.plot(BLE_data.Gyr_Z, label='Gyr_Z')
plt.ylabel('Acceleration (g)')
plt.xlabel('Frames')
plt.legend()
plt.show()


'-----------------------------------------------test---------------------------------------------------------------'



import pandas as pd
import matplotlib.pyplot as plt

############ Nano33BLE 데이터 불러오기 ############
# 가속도 데이터
path = r'C:\Users\lsoor\OneDrive\바탕 화면\임가영\231106 Test\\'
file_name = 'Raw data (g).csv'

# CSV 파일을 읽어옴
data = pd.read_csv(path + file_name)


acc = data.iloc[0::2, 1:4] # 짝수 열만
acc.columns = ['Acc_X', 'Acc_Y', 'Acc_Z']
acc = acc.reset_index(drop=True)
# plt.plot(acc)

gyr = data.iloc[1::2, 1:4] # 홀수 열만
gyr.columns = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
gyr = gyr.reset_index(drop=True)
# plt.plot(gyr)

BLE_data = pd.concat([acc, gyr], axis=1)

# 가속도 그래프
plt.subplot(1, 2, 1)
plt.plot(BLE_data.Acc_X, label='Acc_X')
plt.plot(BLE_data.Acc_Y, label='Acc_Y')
plt.plot(BLE_data.Acc_Z, label='Acc_Z')
plt.ylabel('Acceleration (g)')
plt.xlabel('Frames')
plt.legend()
plt.show()


# 각속도 그래프
plt.subplot(1, 2, 2)
plt.plot(BLE_data.Gyr_X, label='Gyr_X')
plt.plot(BLE_data.Gyr_Y, label='Gyr_Y')
plt.plot(BLE_data.Gyr_Z, label='Gyr_Z')
plt.ylabel('Acceleration (g)')
plt.xlabel('Frames')
plt.legend()
plt.show()




