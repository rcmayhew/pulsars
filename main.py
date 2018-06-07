import pandas as pd
import Feature_select as Fs

file_path = "C:\Data\predicting-a-pulsar-star\pulsar_stars.csv"
original_data = pd.read_csv(file_path)
#  print(data.columns)

ClassLabel = original_data.iloc[:, 8]
data = original_data.drop(['target_class'], axis=1)


normData = Fs.normalize(data)
print(normData.shape)
print(ClassLabel.shape)

"""
best_features = Fs.forward_selection(data, ClassLabel)
print(best_features)
"""
"""
(17898, 8)
(17898,)
Features being tested [0]
0.8921108503743435
Features being tested [1]
0.7784109956419711
Features being tested [2]
0.9266957201922003
Features being tested [3]
0.897809811152084
Features being tested [4]
0.7928260140797855
Features being tested [5]
0.7991395686668902
Features being tested [6]
0.7922114202704212
Features being tested [7]
0.7914850821320818
Features being tested [2, 0]
0.932506425298916
Features being tested [2, 1]
0.9335121242596938
Features being tested [2, 3]
0.9320035758185272
Features being tested [2, 4]
0.9347971840429098
Features being tested [2, 5]
0.9345178232204716
Features being tested [2, 6]
0.9360263716616382
Features being tested [2, 7]
0.9343502067270086
Features being tested [2, 6, 0]
0.9284277572913174
Features being tested [2, 6, 1]
0.9383730025701196
Features being tested [2, 6, 3]
0.9370320706224159
Features being tested [2, 6, 4]
0.9361939881551011
Features being tested [2, 6, 5]
0.9377584087607554
Features being tested [2, 6, 7]
0.9368085819644653
Features being tested [2, 6, 1, 0]
0.9226729243490893
Features being tested [2, 6, 1, 3]
0.9406637613141133
Features being tested [2, 6, 1, 4]
0.9218907140462621
Features being tested [2, 6, 1, 5]
0.9239021119678177
Features being tested [2, 6, 1, 7]
0.9287071181137557
Features being tested [2, 6, 1, 3, 0]
0.9353559056877864
Features being tested [2, 6, 1, 3, 4]
0.9336238685886691
Features being tested [2, 6, 1, 3, 5]
0.9359146273326628
Features being tested [2, 6, 1, 3, 7]
0.9379260252542183
Features being tested [2, 6, 1, 3, 7, 0]
0.9265839758632249
Features being tested [2, 6, 1, 3, 7, 4]
0.9339032294111074
Features being tested [2, 6, 1, 3, 7, 5]
0.9308302603642865
Features being tested [2, 6, 1, 3, 7, 4, 0]
0.9271426975081014
Features being tested [2, 6, 1, 3, 7, 4, 5]
0.926416359369762
Features being tested [2, 6, 1, 3, 7, 4, 0, 5]
0.9286512459492681
([2, 6, 1, 3], 0.9406637613141133)
"""
bestlist = [1, 2, 3, 6]
# next we are going to loop and then we are going to create a class model that does this for us
# final list is [1,2,3,6] with accuracy of 94.07% with n = 3
# we are going to cluster. buyt then do a feature selection by moving down the list to find the best
# grouping of features. because of all the featuers sets, we will not be able to graph it
#
