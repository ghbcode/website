---
layout: default
title: Topics in Machine Learning
description: posted by ghbcode on 2016/05/31
---


Each topic is covered within a self contained python notebook. To begin I'm noting the main ideas first and updates/revisions will follow.

There are various ways to categorize machine learning methods but a common way to think about it is via the following categories. Note that the is another distinction, whether your output is a real-valued number or some sort of classification. For the former, an example would be to predict the price of a stock. This is commonly referred to as 'regression.' The later is referred to as 'classification' and it would answer the question 'does the input belong to class A or B?'. Lastly note that classification can also contain probability measure as in, 'input X is of class A with a probability of 80%'.

* **Supervised Learning** - use labeled data to train your algorithm
  * [Bayes, MLE and MAP Preliminaries](/website/notebooks/Bayes-MLE-MAP.html)
    * Takeaway: Compute a point estimate 
  * [Linear Regression using Ordinary Least Squares and the Normal Equation](/website/notebooks/linear-regression-ols-normal-equation.html), 
  * Linear Regression with Regularization
    * [Ridge (l2) Regression](/website/notebooks/Ridge-regression.html)
      * Takeaway: Penalize complexity for the linear regression
    * [Lasso(l1) Regression](/website/notebooks/Lasso-l1-regression.html)
      * Takeaway: Promote sparsity
    * https://sdsawtelle.github.io/blog/output/week2-andrew-ng-machine-learning-with-python.html
  * [Polynomial regression](/website/notebooks/polynomial-regression.html)
  * Logistic Regression
  * Naive Bayes
  * Decision Trees (CART)
  * Support Vector Machine (SVM)
  * Ensemble Methods
    * Random Forest
    * Boosting
  * [Sample machine learning on a house sale price data set](/website/notebooks/ml_house_sale_price.html)
* **Semi-supervised Learning** - use some labeled data to train your algorithm. Use unlabeled data to possibly infer even more information such as distribution of classes
* **Unsupervised Learning** - tries to find some sort of structure in the data. Depends heavily on similarity (ex: distance) and dissimilarity measures (ex: inverse distance)
  * PCA
  * k-means
  * https://sdsawtelle.github.io/blog/output/week8-andrew-ng-machine-learning-with-python.html


Long Version
* 15 bias – variance, overfitting (variance), lebesge
* 18: L1 regularization, promotes sparsity
* 19: L0 and Bayesian feature selection
* 20: p-norms, Bridge regression is the family of penalized regressions (where Lasso is gamma=1, ridge is gamma=2), Validation and test set with very good E_out and bounding math. AML chapter 4.3 and on has good material on this
* 21: More on Training and validation with bounding math. Adaptive basis-function models, CART (decision trees) forms a tree and a set of regions in feature space
* 22: Very thorough analysis of CART
* 23: 1st page has a very good pros/cons on CART. Random forest is an averaging over individual trees to get better results. 
  * Pseudo algorithm. 
    * Some part is a Bagging (or bootstrap aggregating) procedure. Reduces variance (see averaging function)
    * correlation of the trees is small/er and this also reduces variance
    * formulas for regression and classification provided
* 24: Boosting, another adaptive basis-function model
  * a weak learner that can classify entire feature space 
  * Objective function of AdaBoost provided
    * uses exponential loss as the base to find the argmin Lec24p5
* 25: Semi-supervised learning
  * inductive - supervised learning, i.e. learn from labeled data
  * transductive semi-supervised learning- predict the labels on the unlabeled instances of the training data.
  * Self training algo: use labeled data to find f_hat, then using f_hat label unlabeled data
  * propagating 1-NN
  * Mixture models
    * Great video: https://www.youtube.com/watch?v=REypj2sy_5U&list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt
* 26: More mixture models and then EM (expectation maximization)
  * Great video: https://www.youtube.com/watch?v=iQoXFmbXRJA&index=2&list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt
  * Page 4 shows difficulty in mixture models and reason for EM
  * Page 5 shows use of EM (find missing data, find MLE, estimate quantities in mixture models)
  * Shows entire derivation of E and M steps
* 27: Unsupervised learning : first page has good uses 
  * data clustering or grouping (helps find features or centroids)
  * Based on similarity or dissimilarity measures
    * euclidean distance (x-y)^2, l1 norm |x-y|
    * for categorical features it can be the Hamming distance (number of features that are different between x and y
    * a couple of other measures
  * hierarchical graphical clustering 
  * aglomerative hierarchical clustering 
    * dendogram
* 28: aglomerative hierarchical clustering 
  * useful measures
  * NN algorithm
  * Farthest Neighbor algorithm
