# Pulsars
Using the dataset from the Kaggle dataset list, https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star, this repository is meant to 
explore differnet models to classify pulsars. After looking at the data and seeing it was not easliy separable, I decided that clustering
would be a better choice to start out with.

I started with K = 3 for the K-NN clustering as there are 8 features, a good place to start is the square root of the feature space 
dimension. As you don't want an even number with for k, because it can then cause the classifier to predict the class label based on order 
of the instances, and not based on its features if there is a tie. With odd numbers, there is never a tie. 3 was chosen as the starting 
place for the clustering. The feature space was then searched through using feature selection and accuracy was determined with Leave one
out cross fold validation. This gave the featureset of [2, 1, 6, 3] as the best feature set with an accuracy of 94.06%. This wasn't as
good as I expected so I searched through all k from 1-20 to find the best k/feature set combination. results below:

| Feature Set | Accuracy | K |
| ------------ | ------------------ | - |
| [1, 2, 3, 6] | 0.9685439713934518 | 1 |
| [1, 2, 3, 6] | 0.9406637613141133 | 3 |
| [1, 2, 3, 6] | 0.9186501285059783 | 5 | 
| [2, 6, 7]    | 0.8952955637501396 | 7 |
| [2, 6, 7]    | 0.879651357693597  | 9 |
| [2, 6, 7]    | 0.8630573248407644 | 11|
| [2, 6, 7]    | 0.8482512012515365 | 13|
| [2, 6, 7]    | 0.8324952508660185 | 15|
| [2, 6, 7]    | 0.8185272097441055 | 17| 
| [2, 6, 7]    | 0.8054531232539949 | 19|

The best feature set was [2, 1, 6, 3] with a K of 1, and that give an accuracy of 96.85 %.

This was without standarizing that data first. After running the test again on standardized data on the best k's from the earlier test 
I got:

| Feature Set | Accuracy | K |
| ------------ | ------------------ | - |
| [2, 3, 6, 7, 4, 1, 5, 0] | 0.9713934517823221 | 1 |
| [2, 7, 1, 4, 5, 6, 3]    | 0.9442954520058107 | 3 |
| [2, 7, 3, 4, 0, 1, 6, 5] | 0.922728796513577  | 5 |

While these are modest increases, it lowers the 3.5% gap from the best k=1 model to 2.9% gap. I currently believe the best way to imporve 
this model is to switch to a neurel network, but I am still running searchs to find out.
