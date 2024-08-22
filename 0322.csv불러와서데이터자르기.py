import pandas as pd
import matplotlib.pyplot as plt


path_csv = "C:/Users/abbey/Desktop/낙상데이터 8축/frame_list.csv"
frame_list =[]
i=0
for i in range(0,780):

    raw = pd.read_csv(path_csv)
    raw.columns = ['name', 'ASVM_first', 'ASVM_last', 'GSVM_first', 'GSVM_last']
    a = raw.loc[i, 'ASVM_first']
    b = raw.loc[i, 'GSVM_first']
    first = min(a,b)

    c = raw.loc[i, 'ASVM_last']
    d = raw.loc[i, 'GSVM_last']
    last = max(c,d)

    name = raw.loc[i, 'name']
    columns = ['name', 'first', 'last']
    values = [[name], [first], [last]]
    data_list = pd.DataFrame(dict(zip(columns, values)))
    frame_list.append(data_list)
    total_frame_df = pd.concat(frame_list, ignore_index=True)

    # total_frame_df.to_csv(r"C:/Users/abbey/Desktop/낙상데이터 8축/final_list.csv")


e,m,t =1,3,2
for e in range(6,23):
    for m in range(2, 15):
        for t in range(1, 4):
            try:

                m = str(m).zfill(2)
                e = str(e).zfill(2)
                t = str(t).zfill(2)

                path_data = "C:/Users/abbey/Desktop/낙상데이터 8축/"
                name = "P%sD%sR%s" % (e, m, t)
                name_csv = "%s%s.csv" % (path_data, name)
                name_png = "%s.png" % (name)

                data = pd.read_csv(str(name_csv), header=1)
                data.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'ASVM', 'GSVM']

                match = total_frame_df[total_frame_df['name'] == name]

                first_value = match ['first'].values[0]
                last_value = match ['last'].values[0]

                extract = data.iloc[first_value:last_value+1 ,: ]

                save_data = "C:/Users/abbey/Desktop/extract_data/"
                save_csv  = "%s%s.csv" % (save_data, name)
                # extract.to_csv(save_csv, index=False)


                path_graph = "C:/Users/abbey/Desktop/extract_data/그래프/"

                plt.figure(figsize=(10, 8))

                plt.subplot(2, 1, 1)  # 세로, 가로, 몇번째
                plt.title(name)
                # plt.title('Acceleration SVM')
                plt.ylabel('ASVM(g)')
                plt.xlabel('Frames')
                plt.plot(extract.ASVM, label='ASVM')

                plt.subplot(2, 1, 2)
                # plt.title('Angular Velocity SVM')
                plt.ylabel('GSVM (degree/s)')
                plt.xlabel('Frames')
                plt.plot(extract.GSVM, label='GSVM')

                plt.tight_layout()
                plt.show()

                plt.savefig(path_graph + name_png)
                plt.close()


            except:
                pass
