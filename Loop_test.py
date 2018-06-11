import Feature_select as Fs
import pandas as pd
import operator

"""
Used to exhaustively loop through every n, and for every n, search through the group through the 
greedy search of the feature set. The exact search of the feature set is feature selection.
"""

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
    print(n)
    test_set, test_acc = Fs.forward_selection(data, ClassLabel, n=n, verbose=True)
    set_data.append((test_set, test_acc, n))
    if test_acc > max_acc:
        best_set = test_set
        max_acc = test_acc

set_data.sort(key=operator.itemgetter(1), reverse=True)
print("best set:", best_set, max_acc, n)
print(set_data)
"""
best set: [2, 1, 6, 3] 0.9685439713934518 1
    ([2, 1, 6, 3], 0.9685439713934518, 1), ([2, 6, 3, 1], 0.9536819756397363, 2), ([2, 6, 1, 3], 0.9406637613141133, 3), 
    ([2, 5, 6], 0.9258576377248855, 4), ([2, 6, 1, 3], 0.9186501285059783, 5), ([2, 7, 6], 0.9046262152195776, 6), 
    ([2, 7, 6], 0.8952955637501396, 7), ([2, 7, 6], 0.8869147390769918, 8), ([2, 6, 7], 0.879651357693597, 9), 
    ([2, 6, 7], 0.8704324505531345, 10), ([2, 6, 7], 0.8630573248407644, 11), ([2, 6, 7], 0.8556263269639066, 12), 
    ([2, 6, 7], 0.8482512012515365, 13), ([2, 7, 6], 0.8410436920326293, 14), ([2, 7, 6], 0.8324952508660185, 15), 
    ([2, 7, 6], 0.8254553581405744, 16), ([2, 7, 6], 0.8185272097441055, 17), ([2, 7, 6], 0.8121019108280255, 18), 
    ([2, 7, 6], 0.8054531232539949, 19)
    
    Through the search, there are a two different groups of features that are used, [1, 2, 3, 6] for smaller ns,
    and [2, 6, 7] for all larger n up to 19. features 2 and 6 are strong in both groups. These features are
    Excess kurtosis of the integrated profile, and  Excess kurtosis of the DM-SNR curve. In small k clustering,
    Standard deviation of the integrated profile and Skewness of the integrated profile is used, but in larger
    K cluster, those are replaced with Skewness of the DM-SNR curve. 
    When discussing the choice of n, even numbers should never be chosen despite its accuracy, because it can then 
    predict the class label based on order of the instances, and not based on clustering if there is a tie. So looking 
    at the the 
"""
