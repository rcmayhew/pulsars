import Feature_select as Fs
import pandas as pd
import operator

file_path = "C:\Data\predicting-a-pulsar-star\pulsar_stars.csv"
original_data = pd.read_csv(file_path)

ClassLabel = original_data.iloc[:, 8]
data = original_data.drop(['target_class'], axis=1)
normData = Fs.normalize(data)
max_acc = 0
best_set = []
set_data = []
best_n = 0
for n in range(1, 20):
    test_set, test_acc = Fs.forward_selection(data, ClassLabel, n=n, verbose=False)
    set_data.append((test_set, test_acc, n))
    if test_acc > max_acc:
        best_set = test_set
        max_acc = test_acc

set_data.sort(key=operator.itemgetter(1), reverse=True)
print("best set:", best_set, max_acc, n)
print(set_data)
