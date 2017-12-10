# Notes on Machine Learning Topics




* 15 bias – variance, overfitting (variance), lebesge
* 16: midterm review so start of l2 regularization
* 17: l2 regularization
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

You can use the [editor on GitHub](https://github.com/ghbcode/github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ghbcode/github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
