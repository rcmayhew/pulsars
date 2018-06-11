from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
import random


def normalize(data):
    """
    standardizes the data
    :param data: data without the class label
    :return: returns the data without class labels standardized
    """
    return (data - data.mean()) / (data.max() - data.min())


def leave_one_out(data, classlabel, n=3, loop=True):
    """
    tests k-fold accuracy when k = n
    :param data: the test data without the class identifier
    :param classlabel: the lable of the class for each instance
    :param n: the number of neighbors to be tested, default is 3
    :param loop: loop true is normal, loop false is for testing
    :return: the average overall accuracy for leave one out
    """
    knn = KNeighborsRegressor(n_neighbors=n)
    loo = LeaveOneOut()
    size = loo.get_n_splits(data, classlabel)

    rate = 0
    if loop:
        for training, testing in loo.split(data):
            x_train, x_test = data.iloc[training], data.iloc[testing]
            y_train, y_test = classlabel[training], classlabel[testing]

            knn.fit(x_train, y_train)
            if knn.predict(x_test)[0] == y_test.iloc[0]:
                rate = rate + 1
        error = rate / size
    else:
        error = random.uniform(1.0, 0.0)

    return error


def forward_selection(data, classlabel, featurelist=[], bestlist=[], best_accuracy=0, n=3, verbose=True):
    """
    Uses leave one out to search through the whole of the feature set to determine what set
    of features create the most accurate model for KNN. The functions recurs at the start
    "layer" in the loop for greater readability.
    :param data: The entirety of the feature data
    :param classlabel: The class labels for the data
    :param featurelist: The current list of features the algorithm is searching through
    :param featurelist: The current list of features the algorithm is searching through
    :param bestlist: Always a subset of featurelist, the best set of features (most accurate)
    :param best_accuracy: the accuracy of bestlist
    :param n: the number of neighbors to test with
    :param verbose: determines if you want a print out of each feature set and its accuracy
    :return: returns the final bestlist and accuracy
    """

    num_of_neighbors = n
    current_accuracy = 0

    # Exit condition
    if len(featurelist) == 8:
        return bestlist, best_accuracy
    # loop through all features that aren't in the feature list
    # 8 - len(featurelist) loops
    looplist = featurelist
    for column in range(len(data.columns)):
        # if the feature is already in feature list then skip feature
        if column in looplist:
            continue
        # initialize the list to test the current set of features not in the feature list
        uselist = list(looplist)

        # if else then add to use list
        uselist.append(column)

        # print to see the loops
        if verbose:
            print('Features being tested', uselist)

        # test accuracy of the set of features
        test_accuracy = leave_one_out(data.iloc[:, uselist], classlabel, n=num_of_neighbors, loop=True)

        if verbose:
            print(test_accuracy)
        # keeps a running total of the best set of features within the use list
        if test_accuracy > current_accuracy:
            featurelist = uselist
            current_accuracy = test_accuracy

    # checks if the best of the
    if current_accuracy > best_accuracy:
        bestlist = featurelist
        best_accuracy = current_accuracy

    return forward_selection(data, classlabel, featurelist, bestlist=bestlist, best_accuracy=best_accuracy,
                             n=n, verbose=verbose)




