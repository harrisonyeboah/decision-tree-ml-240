credit card data set https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

## 8. Analysis: Decision Tree vs. Random Forest on Credit Card Fraud (with and without PCA)

**Setup.** There is a total of about 560k transaction in this dataset, The data set is perfectly balanced meaning that we could have used accuracy instead of f1 as an option, The data set was split into thirds, so about 190k transactions per set. I dropped the id column since that was a data leak, since the top half was all not fraud and the bottom half was all fraud. The model could have easily learn off of that.  For every loop in each model I decided to use min_leaf sample as my regularization parameter. 

**1. Regularization: validation F1 never turned down.** Since trees are greedy to be overfitted, and that raising min_leaf_samples will regularize it. Like a lot of machine learning models, one could expect a u shaped graph in terms of f1 score for validation set as the graph grows, but since my validation set f1 never decrease at first I though it was underfitting, but if my f1 increases monotonically, this could potentially just be a good data set with solid decision boundaries an minimal noise.


**2. My hypothesis was wrong: PCA hurt.** 
- The columns of v1 - v28 possibly uncorrelated since it is a mixture of many columns to prevent maybe fraudseters from determining how to weave around fraud based systems. 


.

## 9. Analysis: K-Means vs. Random Forest vs. CatBoost on Breast Cancer (with and without PCA)

**Setup.** Theres is 569 patients in the data, I chose a random stat of 42 for the state shuffle so all random sets are the same: Of the sets there is 189 training, 189 validation, and 191 test. Within the testing set there is (115 benign, 76 malignant). The sample set is tiny so f1 is easily impacted, so **one misclassified test row moves F1 by about 0.5 to 1 point.** I read every difference below with that in mind.

| Model | Best setting | Val F1 | Test F1 | FP | FN |
|---|---|---|---|---|---|
| K Means (mine), raw | k=16 | 0.941 | 0.933 | 4 | 6 |
| Random Forest Tree, raw | leaf=1 | 0.933 | 0.927 | 5 | 6 |
| Cat Boost Tree, raw | depth=3 | 0.949 | 0.922 | 7 | 5 |
| K-Means with PCA Dimension Reduciton | k=14 | 0.896 | 0.808 | 11 | 17 |
| Random Forest with PCA PCA Dimension Reduction | leaf=2 | 0.926 | 0.901 | 7 | 8 |
| CatBoost with PCA Dimension Reduction | depth=8 | 0.910 | 0.889 | 9 | 8 |

**1. Variance dominates on small data.** Each the Cat Tree and the Random Forest reaches an F1 score of 1, When it is time to test the dataset there is only .92 and .93. Géron's In our book, it is described that decision trees are greedy thereore it makes sense that constricted trees lead up to 1 in f1 score. This differs from the creid card data set in a sense where, there is not 190 rows in a training set like how it is in the cancer datatset. Therefore, bagging cannot solve the problem. In the cancer data set a lot of the errors overlap. 


**2. K-Means.** Based on my implementation each cluseter is assigned to the majority voted class. Based off the print statements f1 score are also very volatile from k = 2 to 12, which potentially could mean that the assignment of initial clusters are potentially very impactful in how tghe model performes in terms of finding the right k. This shows that K means could also be very erratic and you dont always get the global optium. 


**3. Medical interpretation.** The models without PCA miss about 5 of the 76 malignant cases This achieved a recall of about 92 percent. A tree-based model estimates class probabilities from the class ratio in a leaf, Based off what we talked about in class regarding type i errors an type ii errors, it would make sense to potentially decrease the threshold for more false positives in terms of malicious tumor, since it better safe than sorry to classify as malignant than benign. In a real word scenario, the doctor should play the role as an audit, while my forest while the tree plays as a white/black box algorithm. 

**6. Connecting both experiments.** In terms of sample size these were the two opposites. The credit card fraud has 190,000 rows, therefore the issue become bias: The breast cancer set had 190 rows, therfore the variance was a huge problem: In the credit card fraud dataset as tree grew more the f1 gap between validation and training narrowed down, but since there was a lot of noise in cancer dataset, validation sometimes had a hard time catching up to training set f1 score.
