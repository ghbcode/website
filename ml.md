---
layout: default_sidebar
title: Topics in Machine Learning
description: posted by ghbcode on 2014/11/23
---


There are various ways to categorize machine learning methods but a common way to think about it is via the following  categories, i.e. supervised learning, semi-supervised learning and unsupervised learning. Note that there is another distinction, whether your output is a real-valued number or some sort of classification. Classification can also contain probability measure as in, 'input X is of class A with a probability of 80%'. Other ways to think of machine learning involve categorizing the methods used by the models or the general idea that the model is using. For now I will use the former method to categorize and list machine learning algorithms. Lastly please note that Artificial Intelligence is the overarching theme, inside of that is machine learning and that deep learning is considered to be a subset of machine learning. 

* **Supervised Learning** - use labeled data to train your algorithm
  * [Data Processing and Preliminaries](https://github.com/ghbcode/website/blob/master/notebooks/SL00 Data Processing and Preliminaries.ipynb)
    * Data Splitting/Partitioning
    * K-fold cross validation
    * Data pre-processing
    * Other topics
  * [Bayes, MLE and MAP Preliminaries](https://github.com/ghbcode/website/blob/master/notebooks/SL01 Bayes, MLE and MAP.ipynb)
  * [Linear Regression using Ordinary Least Squares and the Normal Equation](https://github.com/ghbcode/website/blob/master/notebooks/SL02 Linear Regression OLS.ipynb), 
  * Linear Regression with Regularization
    * [Ridge (l2) Regression](https://github.com/ghbcode/website/blob/master/notebooks/SL02 Linear Regression OLS.ipynb)
    * [Lasso(l1) Regression](https://github.com/ghbcode/website/blob/master/notebooks/SL04 Lasso L1 Regression.ipynb)
  * [Polynomial regression](https://github.com/ghbcode/website/blob/master/notebooks/SL05 Polynomial Regression.ipynb)
  * [Logistic Regression Classification](https://github.com/ghbcode/website/blob/master/notebooks/SL06 Logistic Regression Classification.ipynb)
  * [Naive Bayes Classification](https://github.com/ghbcode/website/blob/master/notebooks/SL07 Naive Bayes Classification.ipynb)
  * [Decision Trees (CART)](https://github.com/ghbcode/website/blob/master/notebooks/SL08 Decision Trees -CART.ipynb)
  * [Support Vector Machine (SVM)](https://github.com/ghbcode/website/blob/master/notebooks/SL09 Support Vector Machine (SVM).ipynbl)
  * [Ensemble Methods](https://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/)
    * [Random Forest](https://github.com/ghbcode/website/blob/master/notebooks/SL10 Random Forest.ipynb)
    * [Boosting using AdaBoost](https://github.com/ghbcode/website/blob/master/notebooks/SL11 Boosting with AdaBoost.ipynb) 
  * [Sample machine learning on a house sale price data set](/website/notebooks/ml_house_sale_price.html)
* **Semi-supervised Learning** - many times acquiring labeled data is either expensive or impractical. So you use the labeled data to train your algorithm as you would with supervised learning. And you use the unlabeled data to possibly infer more information and therefore increase the predictive strength of your model.
  * [More on Semi-Supervised Learning](https://github.com/ghbcode/website/blob/master/notebooks/SL12 Semi Supervised Learning.ipynb)
  * [Mixture models](https://github.com/ghbcode/website/blob/master/notebooks/SL12 Semi Supervised Learning.ipynb)
  * [Expectation Maximization](https://github.com/ghbcode/website/blob/master/notebooks/SL12 Semi Supervised Learning.ipynb)
* **Unsupervised Learning** - tries to find some sort of structure in the data. Depends heavily on similarity (ex: distance) and/or dissimilarity measures (ex: inverse distance)
  * [More on Unsupervised Learning](https://github.com/ghbcode/website/blob/master/notebooks/SL13 Unsupervised Learning.ipynb)
  * [K-Means to find subsets in the data](https://github.com/ghbcode/website/blob/master/notebooks/SL14 USL K-Means Clustering.ipynb)
  * [PCA to reduce dimensionality](https://github.com/ghbcode/website/blob/master/notebooks/Sl15 USL Principal Component Analysis.ipynb)
* **Deep Learning** - A method that improves on traditional machine learning methods, Deep Learning makes use of neural networks that essentially give the algorithm the ability to process the data in a non-linear fashion. This often results in a performance that beats traditional methods although this comes at the expense of complexity and needing large amounts of data.
  * [Introduction to Deep Learning using Tensorflow](/website/notebooks/deep-learning.html)

