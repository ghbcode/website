---
layout: default_sidebar
title: Topics in Machine Learning
description: posted by ghbcode on 2014/11/23
---


Each topic is covered within a self contained python notebook. In some cases the notebooks run code using the [scikit-learn library](http://scikit-learn.org/stable/modules/classes.html). To begin I'm noting the main ideas first and updates/revisions will follow.

There are various ways to categorize machine learning methods but a common way to think about it is via the following  categories, i.e. supervised learning, semi-supervised learning and unsupervised learning. Note that there is another distinction, whether your output is a real-valued number or some sort of classification. Classification can also contain probability measure as in, 'input X is of class A with a probability of 80%'. Other ways to think of machine learning involve categorizing the methods used by the models or the general idea that the model is using. For now I will use the former method to categorize and list machine learning algorithms.

* **Supervised Learning** - use labeled data to train your algorithm
  * [Data Processing and Preliminaries](/website/notebooks/data-processing-preliminaries.html)
    * Data Splitting/Partitioning
    * K-fold cross validation
    * Data pre-processing
    * Other topics
  * [Bayes, MLE and MAP Preliminaries](/website/notebooks/Bayes-MLE-MAP.html)
  * [Linear Regression using Ordinary Least Squares and the Normal Equation](/website/notebooks/linear-regression-ols-normal-equation.html), 
  * Linear Regression with Regularization
    * [Ridge (l2) Regression](/website/notebooks/Ridge-regression.html)
    * [Lasso(l1) Regression](/website/notebooks/Lasso-l1-regression.html)
  * [Polynomial regression](/website/notebooks/polynomial-regression.html)
  * [Logistic Regression Classification](/website/notebooks/logistic-regression-classification.html)
  * [Naive Bayes Classification](/website/notebooks/Naive-bayes.html)
  * [Decision Trees (CART)](/website/notebooks/decision-tree-cart.html)
  * [Support Vector Machine (SVM)](/website/notebooks/svm.html)
  * [Ensemble Methods](https://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/)
    * [Random Forest](/website/notebooks/random-forest.html)
    * [Boosting using AdaBoost](/website/notebooks/boosting.html) 
  * [Sample machine learning on a house sale price data set](/website/notebooks/ml_house_sale_price.html)
* **Semi-supervised Learning** - many times acquiring labeled data is either expensive or impractical. So you use the labeled data to train your algorithm as you would with supervised learning. And you use the unlabeled data to possibly infer more information and therefore increase the predictive strength of your model.
  * [More on Semi-Supervised Learning](/website/notebooks/ssl.html)
  * [Mixture models](/website/notebooks/ssl.html#Mixture-Models)
  * [Expectation Maximization](/website/notebooks/ssl.html#Expectation-Maximization)
* **Unsupervised Learning** - tries to find some sort of structure in the data. Depends heavily on similarity (ex: distance) and/or dissimilarity measures (ex: inverse distance)
  * [More on Unsupervised Learning](/website/notebooks/usl.html)
  * [PCA to reduce dimensionality](/website/notebooks/pca.html)
  * [K-Means to find subsets in the data](/website/notebooks/k-means.html)

