# pulsars
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
| [2, 7, 6]    | 0.8324952508660185 | 15|
| [2, 7, 6]    | 0.8185272097441055 | 17| 
| [2, 7, 6]    | 0.8054531232539949 | 19|

The best feature set was [2, 1, 6, 3] with a K of 1, and that give an accuracy of 96.85 %.

Through the search, there are a two different groups of features that are used, [1, 2, 3, 6] for smaller n's,
and [2, 6, 7] for all larger n up to 19. Features 2 and 6 are strong in both groups. These features are
Excess kurtosis of the integrated profile, and  Excess kurtosis of the DM-SNR curve. In small k clusterings,
Standard deviation of the integrated profile and Skewness of the integrated profile are used, but in larger
K cluster, those are replaced with Skewness of the DM-SNR curve. 
So looking at all the numbers, only k = [1, 3, 5] are above the 90% threshhold, with k=1 being the greatest accuracy.
